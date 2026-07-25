"""Tests the orchestrator's node-level wiring in isolation: that delegating to
a specialist seeds a fresh child conversation from the user's actual latest
question (not the orchestrator's full history, and not whatever message
happens to be last after an earlier specialist already ran this turn), folds
the specialist's answer back in as a single labeled message, records
`data_context` so a later specialist/turn can reuse already-fetched data, and
short-circuits straight to a clarifying question if the specialist asks one.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from data_analyst.agents.orchestrator.nodes import build_analysis_node, build_datasource_node
from data_analyst.clients.llm.factory import FakeToolCallingChatModel


async def test_datasource_node_seeds_fresh_child_and_folds_back_summary():
    llm = FakeToolCallingChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "pbi_mcp_list_semantic_models", "args": {}, "id": "c1"}]),
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
    assert update["data_context"] == "There is one semantic model: Sales Analytics."


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
            AIMessage(content="[datasource] Fetched revenue by region, sandbox_ref=df_1."),
        ],
        "turns": 1,
        "next": "analysis",
        "session_id": "sess-node-2",
        "data_context": "Fetched revenue by region, sandbox_ref=df_1.",
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
        "data_context": "Fetched revenue by region, sandbox_ref=df_1.",
    }
    update = await node(state)

    assert "data_context" not in update


async def test_specialist_self_clarification_sets_next_to_clarify():
    clarify_call = {"name": "request_clarification", "args": {"question": "Which region do you mean?"}, "id": "c1"}
    llm = FakeToolCallingChatModel(
        responses=[AIMessage(content="", tool_calls=[clarify_call]), AIMessage(content="Which region do you mean?")]
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

    assert update["next"] == "clarify"
    assert update["messages"][0].content == "Which region do you mean?"
    assert "data_context" not in update
