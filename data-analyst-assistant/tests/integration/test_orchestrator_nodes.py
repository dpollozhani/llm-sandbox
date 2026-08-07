"""Tests the orchestrator's node-level wiring in isolation: that delegating to
a specialist seeds a fresh child conversation from the user's actual latest
question (not the orchestrator's full history, and not whatever message
happens to be last after an earlier specialist already ran this turn), folds
the specialist's answer back in as a single labeled message, records
`data_context` so a later specialist/turn can reuse already-fetched data, and
short-circuits straight to a clarifying question if the specialist asks one.
"""
from __future__ import annotations

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from data_analyst.agents.datasource.models import DataSourceQueryResult
from data_analyst.agents.orchestrator.nodes import build_analysis_node, build_datasource_node, build_supervisor_node
from data_analyst.clients.llm.factory import FakeToolCallingChatModel
from data_analyst.config.settings import PowerBiCatalog, SemanticModelConfig

_FETCHED = DataSourceQueryResult(
    dataset_id="dataset_1", model_name="Sales Analytics", group_by=["Sales.Region"], measures=["Total Revenue"], row_count=5
).model_dump()

_MULTI_MODEL_CATALOG = PowerBiCatalog(
    semantic_models=[
        SemanticModelConfig(model_name="Sales Analytics", dataset_id="ds-1"),
        SemanticModelConfig(model_name="HQ financial costs", dataset_id="ds-2"),
    ]
)


class _RecordingLLM(FakeToolCallingChatModel):
    """Records the message list of every `_generate` call, so a test can
    check what a specialist was actually seeded with - `FakeToolCallingChatModel`
    itself only scripts responses, it doesn't expose its inputs."""

    recorded_messages: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.recorded_messages.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


async def test_datasource_node_seeds_fresh_child_and_folds_back_summary():
    llm = FakeToolCallingChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "pbi_mcp_get_semantic_metadata", "args": {"model_name": "Sales Analytics"}, "id": "c1"}],
            ),
            AIMessage(content="There is one semantic model: Sales Analytics."),
        ]
    )
    node = build_datasource_node(llm)

    state = {
        "messages": [
            HumanMessage(content="irrelevant earlier turn, should not reach the specialist"),
            HumanMessage(content="what semantic models are available?"),
        ],
        "turns": 1,
        "next": "datasource",
        "session_id": "sess-node-1",
        "data_context": None,
    }
    update = await node(state)

    assert len(update["messages"]) == 1
    folded = update["messages"][0]
    assert isinstance(folded, AIMessage)
    assert folded.content == "[datasource] There is one semantic model: Sales Analytics."


async def test_datasource_node_scopes_its_own_prompt_to_the_callers_allowed_model_names():
    """A caller scoping this session to part of the full catalog (see
    app/api.py's ChatRequest.model_names) should narrow what the datasource
    agent's own prompt describes as available - computed fresh at
    delegation time (the child subgraph is rebuilt every delegation
    anyway), not baked into a build-time-only closure."""
    llm = _RecordingLLM(responses=[AIMessage(content="There is one semantic model: HQ financial costs.")])
    node = build_datasource_node(llm, catalog=_MULTI_MODEL_CATALOG)

    state = {
        "messages": [HumanMessage(content="what semantic models are available?")],
        "turns": 1,
        "next": "datasource",
        "session_id": "sess-node-scoped",
        "data_context": None,
        "allowed_model_names": ["HQ financial costs"],
    }
    await node(state)

    system_prompt = llm.recorded_messages[0][0].content
    assert "HQ financial costs" in system_prompt
    assert "Sales Analytics" not in system_prompt


async def test_specialist_uses_latest_human_message_not_a_prior_specialists_fold():
    """Simulates the supervisor delegating to a second specialist within the
    same turn: by then, state["messages"][-1] is the first specialist's own
    folded-back summary, not the user's question - the second specialist must
    still see the real question."""
    llm = FakeToolCallingChatModel(responses=[AIMessage(content="Reused prior data as requested.")])
    node = build_analysis_node(llm)

    state = {
        "messages": [
            HumanMessage(content="what's the average revenue by region?"),
            AIMessage(content="[datasource] Fetched revenue by region, dataset_id=dataset_1."),
        ],
        "turns": 1,
        "next": "analysis",
        "session_id": "sess-node-2",
        "data_context": _FETCHED,
    }
    update = await node(state)

    assert update["messages"][0].content == "[analysis] Reused prior data as requested."


async def test_analysis_node_does_not_overwrite_data_context():
    llm = FakeToolCallingChatModel(responses=[AIMessage(content="Computed the average.")])
    node = build_analysis_node(llm)

    state = {
        "messages": [HumanMessage(content="what's the average?")],
        "turns": 1,
        "next": "analysis",
        "session_id": "sess-node-3",
        "data_context": _FETCHED,
    }
    update = await node(state)

    assert "data_context" not in update


async def test_specialist_flagged_ambiguity_sets_pending_clarification():
    ambiguity_call = {
        "name": "flag_ambiguity",
        "args": {"reason": "Which region do you mean?", "options": ["North", "South"]},
        "id": "c1",
    }
    llm = FakeToolCallingChatModel(
        responses=[AIMessage(content="", tool_calls=[ambiguity_call]), AIMessage(content="I need more detail.")]
    )
    node = build_datasource_node(llm)

    state = {
        "messages": [HumanMessage(content="how much revenue")],
        "turns": 1,
        "next": "datasource",
        "session_id": "sess-node-4",
        "data_context": None,
    }
    update = await node(state)

    # No "next" key at all - agents/orchestrator/graph.py's
    # _after_specialist decides whether to skip to END from
    # pending_clarification alone, not a "next" value.
    assert "next" not in update
    # Composed deterministically from the tool call's own reason/options -
    # not the specialist's own final freeform message ("I need more detail.")
    assert update["messages"][0].content == "Which region do you mean? (North / South)"
    assert update["pending_clarification"] == {
        "agent": "datasource",
        "reason": "Which region do you mean?",
        "options": ["North", "South"],
    }
    assert "data_context" not in update
    assert update["followup_suggestion"] is None


async def test_specialist_suggests_followup_alongside_completed_answer():
    """Unlike flag_ambiguity, suggest_followup never blocks or replaces the
    specialist's own final answer - both reach the orchestrator together."""
    followup_call = {
        "name": "suggest_followup",
        "args": {"reason": "Want this broken down further?", "options": ["By region", "By product"]},
        "id": "c1",
    }
    llm = FakeToolCallingChatModel(
        responses=[
            AIMessage(content="", tool_calls=[followup_call]),
            AIMessage(content="Computed the top 5 accounts."),
        ]
    )
    node = build_analysis_node(llm)

    state = {
        "messages": [HumanMessage(content="top 5 accounts")],
        "turns": 1,
        "next": "analysis",
        "session_id": "sess-node-10",
        "data_context": None,
    }
    update = await node(state)

    # The real answer is preserved, not discarded the way flag_ambiguity's
    # deterministic composition discards the specialist's own final message.
    assert update["messages"][0].content == "[analysis] Computed the top 5 accounts."
    assert update["pending_clarification"] is None
    assert update["followup_suggestion"] == {
        "agent": "analysis",
        "question": "Want this broken down further?",
        "options": ["By region", "By product"],
    }


async def test_specialist_success_without_followup_clears_any_stale_suggestion():
    """followup_suggestion is always explicitly set (to a new value or
    None), the same convention as pending_clarification, so a suggestion
    from an earlier turn never lingers once superseded."""
    llm = FakeToolCallingChatModel(responses=[AIMessage(content="Done.")])
    node = build_analysis_node(llm)

    state = {
        "messages": [HumanMessage(content="something else")],
        "turns": 1,
        "next": "analysis",
        "session_id": "sess-node-11",
        "data_context": None,
        "followup_suggestion": {"agent": "analysis", "question": "stale", "options": ["a", "b"]},
    }
    update = await node(state)

    assert update["followup_suggestion"] is None


async def test_successful_result_clears_pending_clarification_and_records_it_resolved():
    llm = FakeToolCallingChatModel(responses=[AIMessage(content="Done.")])
    node = build_analysis_node(llm)

    state = {
        "messages": [
            HumanMessage(content="top 10 by inventory"),
            AIMessage(content="Which metric do you mean?"),
            HumanMessage(content="Inventory on-hand"),
        ],
        "turns": 1,
        "next": "analysis",
        "session_id": "sess-node-5",
        "data_context": None,
        "pending_clarification": {"agent": "analysis", "reason": "Which metric do you mean?", "options": ["a", "b"]},
    }
    update = await node(state)

    assert update["pending_clarification"] is None
    assert update["resolved_clarifications"] == [
        {"question": "Which metric do you mean?", "answer": "Inventory on-hand"}
    ]


async def test_specialist_seeds_original_task_and_resolved_clarifications_not_full_history():
    """A reply to a clarifying question isn't a fresh task - the specialist
    needs the original ask plus what's now been answered, not just the
    isolated reply. Unlike the old design, it's seeded with a compact
    rendering of what's resolved, never the full raw message history (which
    would forward - and re-pay for - every prior turn's content forever)."""
    llm = _RecordingLLM(responses=[AIMessage(content="Got it.")])
    node = build_datasource_node(llm)

    state = {
        "messages": [
            HumanMessage(content="top 10 by inventory"),
            AIMessage(content="Which metric do you mean?"),
            HumanMessage(content="Inventory on-hand"),
        ],
        "turns": 1,
        "next": "datasource",
        "session_id": "sess-node-6",
        "data_context": None,
        "pending_clarification": {
            "agent": "datasource",
            "reason": "Which metric do you mean?",
            "options": ["Inventory on-hand", "Inventory in-transit"],
        },
        "resolved_clarifications": [{"question": "Which region?", "answer": "North"}],
    }
    await node(state)

    seeded = llm.recorded_messages[0][1:]  # drop the chain's own prepended SystemMessage
    assert len(seeded) == 1
    content = seeded[0].content
    assert "top 10 by inventory" in content  # the original task, not the reply alone
    assert "Which region? -> North" in content  # already-resolved, from an earlier round
    assert "Inventory on-hand" in content  # this round's reply
    # The full raw exchange (the specialist's own prior clarifying-question
    # message) is not forwarded verbatim - only the compact renderings above.
    assert "Which metric do you mean?" in content  # referenced, but via the compact note
    assert content.count("Which metric do you mean?") == 1


async def test_supervisor_resumes_directly_into_the_specialist_awaiting_a_reply():
    """No fresh routing decision (and no extra LLM call) when a specific
    specialist is the one waiting on a clarification reply - resuming
    straight into it also avoids the risk of the supervisor routing
    somewhere else."""
    llm = FakeToolCallingChatModel(responses=[])  # never called - would raise if it were
    node = build_supervisor_node(llm)

    state = {
        "messages": [HumanMessage(content="North")],
        "turns": 2,
        "session_id": "sess-node-7",
        "pending_clarification": {"agent": "analysis", "reason": "Which region?", "options": ["North", "South"]},
    }
    update = await node(state)

    assert update == {"next": "analysis", "turns": 3}


async def test_specialist_seeds_only_the_latest_task_for_a_fresh_request():
    llm = _RecordingLLM(responses=[AIMessage(content="Answer.")])
    node = build_datasource_node(llm)

    state = {
        "messages": [
            HumanMessage(content="irrelevant earlier turn, should not reach the specialist"),
            HumanMessage(content="what semantic models are available?"),
        ],
        "turns": 1,
        "next": "datasource",
        "session_id": "sess-node-7",
        "data_context": None,
        "pending_clarification": None,
    }
    await node(state)

    seeded = llm.recorded_messages[0][1:]
    assert [m.content for m in seeded] == ["what semantic models are available?"]


class _FakeRestClient:
    async def run_dax_query(self, access_token: str, spec):
        return "EVALUATE SUMMARIZECOLUMNS(...)", pd.DataFrame([{"Region": "North", "Total Revenue": 100}])


async def test_datasource_node_sets_data_context_from_the_tool_result_not_the_summary():
    """data_context comes from pbi_rest_run_dax_query's own structured
    result (dataset_id/model_name/query/row_count), not the specialist's
    freeform final summary - so the supervisor and the next specialist see
    the real group_by/filters/measures/row_count even if that summary
    omitted or misstated them."""
    dax_call = {
        "name": "pbi_rest_run_dax_query",
        "args": {
            "model_name": "Sales Analytics",
            "group_by": [{"table": "Sales", "column": "Region"}],
            "filters": [],
            "measures": [{"name": "Total Revenue", "aggregation": "SUM", "table": "Sales", "column": "Revenue"}],
        },
        "id": "c1",
    }
    llm = FakeToolCallingChatModel(
        responses=[AIMessage(content="", tool_calls=[dax_call]), AIMessage(content="Done.")]
    )
    node = build_datasource_node(llm, rest_client=_FakeRestClient())

    state = {
        "messages": [HumanMessage(content="revenue by region")],
        "turns": 1,
        "next": "datasource",
        "session_id": "sess-node-8",
        "data_context": None,
        "pbi_token": "tok-pbi",
    }
    update = await node(state)

    fetched = update["data_context"]
    assert isinstance(fetched, dict)  # checkpoint-safe - see OrchestratorState.data_context
    assert fetched["dataset_id"] == "dataset_1"
    assert fetched["model_name"] == "Sales Analytics"
    assert fetched["group_by"] == ["Sales.Region"]
    assert fetched["measures"] == ["Total Revenue = SUM(Sales.Revenue)"]
    assert fetched["row_count"] == 1


async def test_specialist_hitting_the_recursion_limit_returns_a_clean_failure_not_a_crash():
    """A specialist that never reaches a final answer - here, a model that
    always calls a tool, never just replies - runs into LangGraph's hard
    step cap (GraphRecursionError) inside `child_graph.ainvoke()`. Left
    uncaught, that would crash the whole turn with a raw framework error
    instead of a normal, bounded reply."""
    always_call = {"name": "python_sandbox_execute", "args": {"code": "result = 1 / 0"}, "id": "c1"}
    llm = FakeToolCallingChatModel(responses=[AIMessage(content="", tool_calls=[always_call])])
    node = build_analysis_node(llm)

    state = {
        "messages": [HumanMessage(content="compute something impossible")],
        "turns": 1,
        "next": "analysis",
        "session_id": "sess-node-9",
        "data_context": None,
    }
    update = await node(state)

    assert "couldn't complete this" in update["messages"][0].content.lower()
    assert update["pending_clarification"] is None
    assert update["followup_suggestion"] is None
    assert "data_context" not in update
