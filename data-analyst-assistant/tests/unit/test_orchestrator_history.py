from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from data_analyst.agents.orchestrator.history import build_prompt_messages, refresh_context
from data_analyst.clients.llm.factory import FakeToolCallingChatModel

# ~1000+ tokens at count_tokens_approximately's default chars_per_token=4.0 -
# a handful of these comfortably crosses SUMMARIZE_TOKEN_THRESHOLD without
# the test needing to hand-derive the exact token/character formula.
_BIG_CONTENT = "x" * 4000


class _RecordingLLM(FakeToolCallingChatModel):
    """Records the message list of every `_generate` call, so a test can
    check exactly what `refresh_context` fed the model."""

    recorded_messages: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.recorded_messages.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _messages(n: int, *, big: bool = False) -> list:
    suffix = f" {_BIG_CONTENT}" if big else ""
    return [HumanMessage(content=f"m{i}{suffix}") for i in range(n)]


def test_build_prompt_messages_returns_everything_since_the_last_summary():
    messages = _messages(5)
    assert build_prompt_messages(messages, None, 0) == messages


def test_build_prompt_messages_excludes_already_summarized_messages():
    messages = _messages(5)
    assert build_prompt_messages(messages, None, 2) == messages[2:]


def test_build_prompt_messages_prepends_the_summary():
    messages = _messages(5)
    context = build_prompt_messages(messages, "the user asked about revenue", 2)
    assert "the user asked about revenue" in context[0].content
    assert context[1:] == messages[2:]


async def test_refresh_context_leaves_history_untouched_below_token_threshold():
    """A handful of small messages since the last summary - nowhere near
    SUMMARIZE_TOKEN_THRESHOLD - shouldn't trigger a summarization call at
    all, however many of them there are."""
    llm = _RecordingLLM(responses=[AIMessage(content="should not be called")])
    messages = _messages(5)

    context, update = await refresh_context(llm, {"messages": messages})

    assert update == {}
    assert context == messages
    assert llm.recorded_messages == []


async def test_refresh_context_folds_messages_since_the_last_summary():
    llm = _RecordingLLM(responses=[AIMessage(content="folded summary")])
    messages = _messages(4, big=True)

    context, update = await refresh_context(llm, {"messages": messages})

    assert update == {"history_summary": "folded summary", "history_summarized_through": len(messages)}
    # Everything just got folded in, so this turn's context is the fresh
    # summary alone - nothing left unfolded yet.
    assert len(context) == 1
    assert "folded summary" in context[0].content
    fed = llm.recorded_messages[0][1:]  # drop the summarizer's own SystemMessage
    assert [m.content for m in fed] == [m.content for m in messages]


async def test_refresh_context_only_folds_messages_since_the_last_summary():
    """Already-summarized messages are never re-measured against the
    threshold or re-fed to the model - only the delta since
    history_summarized_through."""
    llm = _RecordingLLM(responses=[AIMessage(content="updated summary")])
    already_summarized = _messages(2)  # small - already folded in, irrelevant to this call
    new_backlog = _messages(4, big=True)  # crosses SUMMARIZE_TOKEN_THRESHOLD on its own
    messages = already_summarized + new_backlog
    state = {
        "messages": messages,
        "history_summary": "earlier summary",
        "history_summarized_through": len(already_summarized),
    }

    context, update = await refresh_context(llm, state)

    assert update == {"history_summary": "updated summary", "history_summarized_through": len(messages)}
    assert len(context) == 1
    assert "updated summary" in context[0].content
    fed = llm.recorded_messages[0][1:]
    # Only the new backlog, plus the "previous summary" lead-in - not the
    # already-summarized messages.
    assert "earlier summary" in fed[0].content
    assert [m.content for m in fed[1:]] == [m.content for m in new_backlog]
