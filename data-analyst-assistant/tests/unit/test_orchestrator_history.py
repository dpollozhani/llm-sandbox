from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from data_analyst.agents.orchestrator.history import (
    RECENT_MESSAGE_COUNT,
    SUMMARIZE_THRESHOLD,
    build_prompt_messages,
    maybe_summarize_history,
)
from data_analyst.clients.llm.factory import FakeToolCallingChatModel


class _RecordingLLM(FakeToolCallingChatModel):
    """Records the message list of every `_generate` call, so a test can
    check exactly what `maybe_summarize_history` fed the model."""

    recorded_messages: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.recorded_messages.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _messages(n: int) -> list:
    return [HumanMessage(content=f"m{i}") for i in range(n)]


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


async def test_maybe_summarize_history_does_nothing_below_threshold():
    llm = _RecordingLLM(responses=[AIMessage(content="summary")])
    state = {"messages": _messages(SUMMARIZE_THRESHOLD)}

    result = await maybe_summarize_history(llm, state)

    assert result is None
    assert llm.recorded_messages == []


async def test_maybe_summarize_history_folds_only_the_older_messages():
    llm = _RecordingLLM(responses=[AIMessage(content="folded summary")])
    messages = _messages(SUMMARIZE_THRESHOLD + 4)
    state = {"messages": messages}

    result = await maybe_summarize_history(llm, state)

    cutoff = len(messages) - RECENT_MESSAGE_COUNT
    assert result == {"history_summary": "folded summary", "history_summarized_through": cutoff}
    fed = llm.recorded_messages[0][1:]  # drop the summarizer's own SystemMessage
    assert [m.content for m in fed] == [m.content for m in messages[:cutoff]]


async def test_maybe_summarize_history_skips_when_nothing_new_since_last_summary():
    llm = _RecordingLLM(responses=[AIMessage(content="should not be called")])
    messages = _messages(SUMMARIZE_THRESHOLD + 4)
    cutoff = len(messages) - RECENT_MESSAGE_COUNT
    state = {"messages": messages, "history_summary": "already summarized", "history_summarized_through": cutoff}

    result = await maybe_summarize_history(llm, state)

    assert result is None
    assert llm.recorded_messages == []


async def test_maybe_summarize_history_only_feeds_the_new_delta_next_time():
    llm = _RecordingLLM(responses=[AIMessage(content="updated summary")])
    messages = _messages(SUMMARIZE_THRESHOLD + 8)
    already = SUMMARIZE_THRESHOLD + 8 - RECENT_MESSAGE_COUNT - 4  # summarized up through 4 messages ago
    state = {"messages": messages, "history_summary": "earlier summary", "history_summarized_through": already}

    result = await maybe_summarize_history(llm, state)

    new_cutoff = len(messages) - RECENT_MESSAGE_COUNT
    assert result["history_summarized_through"] == new_cutoff
    fed = llm.recorded_messages[0][1:]
    # Only the delta since `already`, plus the "previous summary" lead-in -
    # not the whole history again.
    assert "earlier summary" in fed[0].content
    assert [m.content for m in fed[1:]] == [m.content for m in messages[already:new_cutoff]]
