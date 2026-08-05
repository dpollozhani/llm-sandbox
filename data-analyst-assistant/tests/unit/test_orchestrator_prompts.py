from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from data_analyst.agents.datasource.models import DataSourceQueryResult
from data_analyst.agents.orchestrator.nodes import Route, build_clarify_node, build_respond_node, build_supervisor_node
from data_analyst.clients.llm.factory import FakeToolCallingChatModel
from data_analyst.config.settings import Glossary, GlossaryEntry, PowerBiCatalog, SemanticModelConfig

_GLOSSARY = Glossary(terms=[GlossaryEntry(term="BRIC", definition="An item attribute, not BRICS.")])
_CATALOG = PowerBiCatalog(semantic_models=[SemanticModelConfig(model_name="Sales Analytics", dataset_id="ds-1")])
_MULTI_MODEL_CATALOG = PowerBiCatalog(
    semantic_models=[
        SemanticModelConfig(model_name="Sales Analytics", dataset_id="ds-1"),
        SemanticModelConfig(model_name="HQ financial costs", dataset_id="ds-2"),
    ]
)


class _RecordingLLM(FakeToolCallingChatModel):
    """Records the message list of every `_generate` call."""

    recorded_messages: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.recorded_messages.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class _RecordingStructuredLLM(FakeToolCallingChatModel):
    """Like `_RecordingLLM`, but for the `with_structured_output` path
    (`build_supervisor_node`/`build_clarify_node`) - the base fake's own
    `with_structured_output` stand-in ignores its input entirely, so it
    can't be used to check what prompt a structured-output call actually
    received."""

    recorded_messages: list = Field(default_factory=list)

    def with_structured_output(self, schema, **kwargs):
        def _invoke(messages, *_args, **_kwargs):
            self.recorded_messages.append(list(messages))
            return schema()

        return RunnableLambda(_invoke)


class _ScriptedRouteLLM(FakeToolCallingChatModel):
    """Like `_RecordingStructuredLLM`, but returns a specific `Route`
    instead of the schema's bare default - needed to script the router into
    picking "clarify" itself, to check `build_supervisor_node`'s
    deterministic override of that choice."""

    route: Route = Route()

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda *_args, **_kwargs: self.route)


async def test_respond_node_injects_glossary():
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    node = build_respond_node(llm, glossary=_GLOSSARY)

    await node({"messages": [HumanMessage(content="hi")]})

    assert "- BRIC: An item attribute, not BRICS." in llm.recorded_messages[0][0].content


async def test_respond_node_has_no_glossary_section_when_absent():
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    node = build_respond_node(llm)

    await node({"messages": [HumanMessage(content="hi")]})

    assert "Glossary:" not in llm.recorded_messages[0][0].content


async def test_supervisor_node_injects_glossary():
    """The supervisor sees raw user language before ever delegating - a
    term it doesn't recognize can misroute, or trigger an unnecessary
    upfront clarifying question, just as easily as it can confuse the
    datasource agent."""
    llm = _RecordingStructuredLLM(responses=[])
    node = build_supervisor_node(llm, glossary=_GLOSSARY)

    await node({"messages": [HumanMessage(content="top 10 by BRIC")], "turns": 0})

    assert "- BRIC: An item attribute, not BRICS." in llm.recorded_messages[0][0].content


async def test_clarify_node_injects_glossary():
    llm = _RecordingStructuredLLM(responses=[])
    node = build_clarify_node(llm, glossary=_GLOSSARY)

    await node({"messages": [HumanMessage(content="hi")]})

    assert "- BRIC: An item attribute, not BRICS." in llm.recorded_messages[0][0].content


async def test_respond_node_lists_catalog_model_names():
    """So "which models are available" can be answered from real config,
    never guessed - see RESPOND_SYSTEM_PROMPT's anti-confabulation rule."""
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    node = build_respond_node(llm, catalog=_CATALOG)

    await node({"messages": [HumanMessage(content="which models are available?")]})

    assert 'Available semantic models: "Sales Analytics".' in llm.recorded_messages[0][0].content


async def test_respond_node_describes_only_the_callers_allowed_model_names():
    """A caller can scope a session to part of the full catalog (see
    app/api.py's ChatRequest.model_names) - the respond node should
    describe only that subset, computed fresh per-invocation from state,
    not the full catalog it was built with."""
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    node = build_respond_node(llm, catalog=_MULTI_MODEL_CATALOG)

    await node(
        {
            "messages": [HumanMessage(content="which models are available?")],
            "allowed_model_names": ["Sales Analytics"],
        }
    )

    prompt = llm.recorded_messages[0][0].content
    assert 'Available semantic models: "Sales Analytics".' in prompt
    assert "HQ financial costs" not in prompt


async def test_respond_node_describes_full_catalog_when_no_scoping_requested():
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    node = build_respond_node(llm, catalog=_MULTI_MODEL_CATALOG)

    await node({"messages": [HumanMessage(content="which models are available?")]})

    prompt = llm.recorded_messages[0][0].content
    assert "Sales Analytics" in prompt
    assert "HQ financial costs" in prompt


async def test_respond_node_has_no_catalog_section_when_absent():
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    node = build_respond_node(llm)

    await node({"messages": [HumanMessage(content="hi")]})

    assert "Available semantic models" not in llm.recorded_messages[0][0].content


async def test_respond_node_injects_currently_available_data():
    """Concrete grounding for RESPOND_SYSTEM_PROMPT's "only suggest a
    follow-up when it's grounded in the currently available data" rule -
    without this, the model has no structured signal for what's actually
    in play this conversation to scope a suggestion to."""
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    node = build_respond_node(llm)
    data_context = DataSourceQueryResult(
        dataset_id="dataset_1", model_name="Sales Analytics", group_by=["Sales.Region"], row_count=5
    ).model_dump()

    await node({"messages": [HumanMessage(content="what's our revenue by region?")], "data_context": data_context})

    prompt = llm.recorded_messages[0][0].content
    assert "Currently available data in this session:" in prompt
    assert "Sales Analytics" in prompt


async def test_respond_node_has_no_data_context_section_when_absent():
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    node = build_respond_node(llm)

    await node({"messages": [HumanMessage(content="hi")]})

    # The prompt's own static text mentions "Currently available data" in
    # passing (see RESPOND_SYSTEM_PROMPT) - check for the actual injected
    # line, not that substring, which would be present either way.
    assert "Currently available data in this session:" not in llm.recorded_messages[0][0].content


async def test_supervisor_node_lists_catalog_model_names():
    """The supervisor has no schema access at all - only these names - so a
    request about a model's contents must route to "datasource" instead of
    the supervisor guessing; see SUPERVISOR_SYSTEM_PROMPT."""
    llm = _RecordingStructuredLLM(responses=[])
    node = build_supervisor_node(llm, catalog=_CATALOG)

    await node({"messages": [HumanMessage(content="which models are available?")], "turns": 0})

    assert 'Available semantic models: "Sales Analytics".' in llm.recorded_messages[0][0].content


async def test_supervisor_node_describes_only_the_callers_allowed_model_names():
    llm = _RecordingStructuredLLM(responses=[])
    node = build_supervisor_node(llm, catalog=_MULTI_MODEL_CATALOG)

    await node(
        {
            "messages": [HumanMessage(content="which models are available?")],
            "turns": 0,
            "allowed_model_names": ["HQ financial costs"],
        }
    )

    prompt = llm.recorded_messages[0][0].content
    assert 'Available semantic models: "HQ financial costs".' in prompt
    assert "Sales Analytics" not in prompt


async def test_supervisor_node_mentions_a_pending_followup_suggestion():
    """Unlike pending_clarification's deterministic resume, a non-blocking
    followup_suggestion still goes through the normal routing LLM call -
    this is context for that decision, not a bypass of it."""
    llm = _RecordingStructuredLLM(responses=[])
    node = build_supervisor_node(llm)

    await node(
        {
            "messages": [HumanMessage(content="the first one")],
            "turns": 1,
            "followup_suggestion": {"agent": "analysis", "question": "Break down further?", "options": ["By region", "By product"]},
        }
    )

    prompt = llm.recorded_messages[0][0].content
    assert "Break down further?" in prompt
    assert '"analysis"' in prompt


async def test_supervisor_node_overrides_a_same_turn_reclarify_of_a_followup_suggestion():
    """A production trace caught the routing LLM picking "clarify" right
    after a specialist's own non-blocking suggest_followup, with no new
    human message in between - turning it into a blocking question and
    defeating the whole point of the non-blocking mechanism. The prompt
    tells the router not to do this, but this is the deterministic
    backstop: even if the router still says "clarify" in that exact
    situation, it's overridden to "respond"."""
    llm = _ScriptedRouteLLM(responses=[], route=Route(next="clarify"))
    node = build_supervisor_node(llm)

    result = await node(
        {
            "messages": [AIMessage(content="[analysis] here's your answer")],
            "turns": 1,
            "followup_suggestion": {"agent": "analysis", "question": "Break down further?", "options": ["By region", "By product"]},
        }
    )

    assert result["next"] == "respond"


async def test_supervisor_node_allows_clarify_after_a_fresh_human_message_despite_a_stale_followup():
    """The override above is scoped to the same-turn case only - once the
    user has actually sent a new message, "clarify" is a legitimate routing
    decision again (e.g. a genuinely new, unrelated ambiguity), even if a
    followup_suggestion from earlier hasn't been superseded yet."""
    llm = _ScriptedRouteLLM(responses=[], route=Route(next="clarify"))
    node = build_supervisor_node(llm)

    result = await node(
        {
            "messages": [HumanMessage(content="something unrelated")],
            "turns": 1,
            "followup_suggestion": {"agent": "analysis", "question": "Break down further?", "options": ["By region", "By product"]},
        }
    )

    assert result["next"] == "clarify"


async def test_supervisor_node_has_no_followup_section_when_absent():
    llm = _RecordingStructuredLLM(responses=[])
    node = build_supervisor_node(llm)

    await node({"messages": [HumanMessage(content="hi")], "turns": 0})

    assert "already completed its answer and suggested a follow-up" not in llm.recorded_messages[0][0].content


async def test_respond_node_is_told_not_to_restate_an_existing_followup_suggestion():
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    node = build_respond_node(llm)

    await node(
        {
            "messages": [HumanMessage(content="hi")],
            "followup_suggestion": {"agent": "analysis", "question": "Break down further?", "options": ["By region", "By product"]},
        }
    )

    prompt = llm.recorded_messages[0][0].content
    assert "Break down further?" in prompt
    assert "Don't repeat or list them yourself" in prompt
    assert "don't ask the user which one they want either" in prompt


async def test_clarify_node_mentions_already_resolved_clarifications():
    """So the supervisor's own upfront path doesn't re-ask something a
    specialist (or an earlier upfront clarify) already settled."""
    llm = _RecordingStructuredLLM(responses=[])
    node = build_clarify_node(llm)

    await node(
        {
            "messages": [HumanMessage(content="hi")],
            "resolved_clarifications": [{"question": "Which region?", "answer": "North"}],
        }
    )

    prompt = llm.recorded_messages[0][0].content
    assert '"Which region?" -> "North"' in prompt
