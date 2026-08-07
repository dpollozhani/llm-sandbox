from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from data_analyst.agents.orchestrator.history import build_prompt_messages, refresh_context
from data_analyst.clients.llm.factory import FakeToolCallingChatModel


class _RecordingLLM(FakeToolCallingChatModel):
    """Records the message list of every `_generate` call, so a test can
    check exactly what `refresh_context` fed the model."""

    recorded_messages: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.recorded_messages.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _messages(n: int, *, chars: int = 0) -> list:
    """`chars` pads each message's content so its approximate token count is
    predictable - see the fold test below for why 3200 chars each, 10 of
    them, is the size used to exercise a fold."""
    suffix = f" {'x' * chars}" if chars else ""
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
    MAX_RECENT_TOKENS - shouldn't trigger a summarization call at all,
    however many of them there are."""
    llm = _RecordingLLM(responses=[AIMessage(content="should not be called")])
    messages = _messages(5)

    context, update = await refresh_context(llm, {"messages": messages})

    assert update == {}
    assert context == messages
    assert llm.recorded_messages == []


async def test_refresh_context_folds_the_older_part_keeping_a_raw_recent_window():
    """10 messages of 3200 filler chars each total ~8050 approximate tokens -
    just over MAX_RECENT_TOKENS (8000), triggering a fold. With
    RECENT_WINDOW_TOKENS at 4000, the last 4 of them (~3220 tokens) fit raw
    (the last 5, ~4025 tokens, don't) - so the fold should only feed the
    first 6 to the summarizer, and this turn's context should carry the
    fresh summary plus those last 4 raw, not just the summary alone."""
    llm = _RecordingLLM(responses=[AIMessage(content="folded summary")])
    messages = _messages(10, chars=3200)
    old, recent = messages[:6], messages[6:]

    context, update = await refresh_context(llm, {"messages": messages})

    assert update == {"history_summary": "folded summary", "history_summarized_through": len(old)}
    assert "folded summary" in context[0].content
    assert [m.content for m in context[1:]] == [m.content for m in recent]
    fed = llm.recorded_messages[0][1:]  # drop the summarizer's own SystemMessage
    assert [m.content for m in fed] == [m.content for m in old]


async def test_refresh_context_only_measures_messages_since_the_last_summary():
    """Already-summarized messages are never re-measured against the
    threshold or re-fed to the model - only the delta since
    history_summarized_through."""
    llm = _RecordingLLM(responses=[AIMessage(content="updated summary")])
    already_summarized = _messages(2)  # small - already folded in, irrelevant to this call
    new_backlog = _messages(10, chars=3200)  # ~8050 tokens, crosses MAX_RECENT_TOKENS on its own
    old, recent = new_backlog[:6], new_backlog[6:]
    messages = already_summarized + new_backlog
    state = {
        "messages": messages,
        "history_summary": "earlier summary",
        "history_summarized_through": len(already_summarized),
    }

    context, update = await refresh_context(llm, state)

    assert update == {
        "history_summary": "updated summary",
        "history_summarized_through": len(already_summarized) + len(old),
    }
    assert [m.content for m in context[1:]] == [m.content for m in recent]
    fed = llm.recorded_messages[0][1:]
    # Only the new backlog's older part, plus the "previous summary"
    # lead-in - not the already-summarized messages, not the raw recent tail.
    assert "earlier summary" in fed[0].content
    assert [m.content for m in fed[1:]] == [m.content for m in old]
