from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from data_analyst.agents.orchestrator.history import (
    RECENT_MESSAGE_COUNT,
    build_prompt_messages,
    summarize_history,
)
from data_analyst.clients.llm.factory import FakeToolCallingChatModel

# ~1000+ tokens at count_tokens_approximately's default chars_per_token=4.0 -
# a handful of these comfortably crosses SUMMARIZE_TOKEN_THRESHOLD without
# the test needing to hand-derive the exact token/character formula.
_BIG_CONTENT = "x" * 4000


class _RecordingLLM(FakeToolCallingChatModel):
    """Records the message list of every `_generate` call, so a test can
    check exactly what `summarize_history` fed the model."""

    recorded_messages: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.recorded_messages.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _messages(n: int, *, big: bool = False) -> list:
    suffix = f" {_BIG_CONTENT}" if big else ""
    return [HumanMessage(content=f"m{i}{suffix}") for i in range(n)]


def test_build_prompt_messages_returns_everything_when_short_and_no_summary():
    messages = _messages(5)
    assert build_prompt_messages(messages, None) == messages


def test_build_prompt_messages_trims_to_the_recent_window():
    messages = _messages(RECENT_MESSAGE_COUNT + 5)
    context = build_prompt_messages(messages, None)
    assert context == messages[-RECENT_MESSAGE_COUNT:]


def test_build_prompt_messages_prepends_the_summary():
    messages = _messages(RECENT_MESSAGE_COUNT + 5)
    context = build_prompt_messages(messages, "the user asked about revenue")
    assert "the user asked about revenue" in context[0].content
    assert context[1:] == messages[-RECENT_MESSAGE_COUNT:]


async def test_summarize_history_does_nothing_below_token_threshold():
    """A handful of small foldable messages - nowhere near
    SUMMARIZE_TOKEN_THRESHOLD - shouldn't trigger a summarization call at
    all, however many of them there are."""
    llm = _RecordingLLM(responses=[AIMessage(content="summary")])
    state = {"messages": _messages(RECENT_MESSAGE_COUNT + 5)}

    result = await summarize_history(llm, state)

    assert result is None
    assert llm.recorded_messages == []


async def test_summarize_history_folds_only_the_older_messages():
    llm = _RecordingLLM(responses=[AIMessage(content="folded summary")])
    messages = _messages(RECENT_MESSAGE_COUNT + 3, big=True)
    state = {"messages": messages}

    result = await summarize_history(llm, state)

    foldable_up_to = len(messages) - RECENT_MESSAGE_COUNT
    assert result == {"history_summary": "folded summary", "history_summarized_through": foldable_up_to}
    fed = llm.recorded_messages[0][1:]  # drop the summarizer's own SystemMessage
    assert [m.content for m in fed] == [m.content for m in messages[:foldable_up_to]]


async def test_summarize_history_skips_when_nothing_new_since_last_summary():
    """The "nothing new to fold in" check fires before any token counting -
    a backlog well past SUMMARIZE_TOKEN_THRESHOLD still shouldn't trigger a
    second summarization call if it was already folded in last time."""
    llm = _RecordingLLM(responses=[AIMessage(content="should not be called")])
    messages = _messages(RECENT_MESSAGE_COUNT + 3, big=True)
    foldable_up_to = len(messages) - RECENT_MESSAGE_COUNT
    state = {"messages": messages, "history_summary": "already summarized", "history_summarized_through": foldable_up_to}

    result = await summarize_history(llm, state)

    assert result is None
    assert llm.recorded_messages == []


async def test_summarize_history_only_feeds_the_new_delta_next_time():
    """Only the messages added since the last summary are ever re-measured
    against the threshold or fed to the model - not the whole foldable
    range again, and not anything already folded in."""
    llm = _RecordingLLM(responses=[AIMessage(content="updated summary")])
    already_summarized = _messages(2)  # small - already folded in, irrelevant to this call
    new_backlog = _messages(4, big=True)  # crosses SUMMARIZE_TOKEN_THRESHOLD on its own
    recent = _messages(RECENT_MESSAGE_COUNT)  # never token-counted at all
    messages = already_summarized + new_backlog + recent
    state = {
        "messages": messages,
        "history_summary": "earlier summary",
        "history_summarized_through": len(already_summarized),
    }

    result = await summarize_history(llm, state)

    foldable_up_to = len(messages) - RECENT_MESSAGE_COUNT
    assert result["history_summarized_through"] == foldable_up_to
    fed = llm.recorded_messages[0][1:]
    # Only the new backlog, plus the "previous summary" lead-in - not the
    # already-summarized messages, and not the recent window either.
    assert "earlier summary" in fed[0].content
    assert [m.content for m in fed[1:]] == [m.content for m in new_backlog]
