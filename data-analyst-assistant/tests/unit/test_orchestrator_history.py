from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from data_analyst.agents.orchestrator.history import (
    RECENT_MESSAGE_COUNT,
    build_prompt_messages,
    pending_backlog,
    summarize_backlog,
)
from data_analyst.clients.llm.factory import FakeToolCallingChatModel

# ~1000+ tokens at count_tokens_approximately's default chars_per_token=4.0 -
# a handful of these comfortably crosses SUMMARIZE_TOKEN_THRESHOLD without
# the test needing to hand-derive the exact token/character formula.
_BIG_CONTENT = "x" * 4000


class _RecordingLLM(FakeToolCallingChatModel):
    """Records the message list of every `_generate` call, so a test can
    check exactly what `summarize_backlog` fed the model."""

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


def test_pending_backlog_is_none_below_token_threshold():
    """A handful of small foldable messages - nowhere near
    SUMMARIZE_TOKEN_THRESHOLD - isn't worth folding in, however many of
    them there are. Purely synchronous: no LLM involved in this decision
    at all."""
    messages = _messages(RECENT_MESSAGE_COUNT + 5)
    assert pending_backlog({"messages": messages}) is None


def test_pending_backlog_is_none_when_nothing_new_since_last_summary():
    """Fires before any token counting - a backlog well past
    SUMMARIZE_TOKEN_THRESHOLD still isn't "pending" if it was already
    folded in last time."""
    messages = _messages(RECENT_MESSAGE_COUNT + 3, big=True)
    foldable_up_to = len(messages) - RECENT_MESSAGE_COUNT
    state = {"messages": messages, "history_summary": "already summarized", "history_summarized_through": foldable_up_to}

    assert pending_backlog(state) is None


def test_pending_backlog_returns_only_the_new_delta():
    """Only the messages added since the last summary are ever considered
    - not the whole foldable range again, and not anything already folded
    in."""
    already_summarized = _messages(2)  # small - already folded in, irrelevant to this call
    new_backlog = _messages(4, big=True)  # crosses SUMMARIZE_TOKEN_THRESHOLD on its own
    recent = _messages(RECENT_MESSAGE_COUNT)  # never considered at all
    messages = already_summarized + new_backlog + recent
    state = {"messages": messages, "history_summarized_through": len(already_summarized)}

    pending = pending_backlog(state)

    assert pending is not None
    new_messages, foldable_up_to = pending
    assert [m.content for m in new_messages] == [m.content for m in new_backlog]
    assert foldable_up_to == len(messages) - RECENT_MESSAGE_COUNT


async def test_summarize_backlog_folds_the_given_messages_unconditionally():
    """No threshold, no bookkeeping - summarize_backlog always calls the
    model for whatever it's handed; that decision lives in pending_backlog
    instead."""
    llm = _RecordingLLM(responses=[AIMessage(content="folded summary")])
    new_messages = _messages(2)

    result = await summarize_backlog(llm, new_messages, previous_summary=None)

    assert result == "folded summary"
    fed = llm.recorded_messages[0][1:]  # drop the summarizer's own SystemMessage
    assert [m.content for m in fed] == [m.content for m in new_messages]


async def test_summarize_backlog_includes_the_previous_summary_as_lead_in():
    llm = _RecordingLLM(responses=[AIMessage(content="updated summary")])
    new_messages = _messages(2)

    result = await summarize_backlog(llm, new_messages, previous_summary="earlier summary")

    assert result == "updated summary"
    fed = llm.recorded_messages[0][1:]
    assert "earlier summary" in fed[0].content
    assert [m.content for m in fed[1:]] == [m.content for m in new_messages]
