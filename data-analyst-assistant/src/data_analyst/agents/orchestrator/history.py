"""Rolling history summarization: `OrchestratorState["messages"]` stays
unbounded (it's the checkpointed conversation record, needed for
audit/resume - never trimmed), but the supervisor/respond/clarify nodes
get a bounded prompt context instead - a running summary of older turns
plus the most recent messages verbatim - so per-turn token cost stops
growing linearly with conversation length.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately

from data_analyst.agents.orchestrator.state import OrchestratorState

RECENT_MESSAGE_COUNT = 8
"""How many of the most recent raw messages are always included verbatim in
prompt context, on top of `history_summary` - generous enough to cover the
current exchange (a task, any specialist fold-backs, a clarification round
trip) without relying solely on the summary for it. By message count, not
tokens: what this window exists to guarantee is "N whole recent exchanges
stay intact", regardless of how big any one of them happens to be -
`SUMMARIZE_TOKEN_THRESHOLD` below is what actually bounds cost."""

SUMMARIZE_TOKEN_THRESHOLD = 2000
"""Once the *foldable* backlog (everything older than `RECENT_MESSAGE_COUNT`,
not yet folded into `history_summary`) holds more than this many tokens -
approximated via `count_tokens_approximately`, the same rough
characters-per-token metric LangChain's/LangMem's own summarization
utilities default to - `summarize_history` folds it in. Token count, not
message count: token count is what actually drives per-turn cost, and a
handful of large tool-result-laden messages can cost as much as dozens of
short ones. 2000 is an order-of-magnitude estimate - comfortably above what
one ordinary turn's own exchange would use, far below typical model context
windows - not a value tuned against real usage."""

_SUMMARIZE_PROMPT = """Summarize this part of a data-analyst conversation in
a few sentences, preserving anything a later turn would need: what the user
asked for, what data/analyses were already produced (dataset ids, models,
metrics), and any open threads or unresolved questions. Be concise - this
replaces, not supplements, the messages it covers."""


def build_prompt_messages(messages: list[AnyMessage], history_summary: str | None) -> list[AnyMessage]:
    """`[summary-as-message?, *recent raw messages]` instead of the full,
    ever-growing `messages` list - what the supervisor/respond/clarify node
    builders (`agents/orchestrator/nodes.py`) actually send to the model."""
    recent = messages[-RECENT_MESSAGE_COUNT:]
    if not history_summary:
        return recent
    return [HumanMessage(content=f"Summary of earlier turns: {history_summary}"), *recent]


async def summarize_history(llm: BaseChatModel, state: OrchestratorState) -> dict | None:
    """A `{"history_summary": ..., "history_summarized_through": ...}` state
    update if the foldable backlog has grown past `SUMMARIZE_TOKEN_THRESHOLD`
    since the last summary - `None` if there's nothing worth folding in yet
    (the common case), so `build_supervisor_node` can skip touching either
    field at all most turns.

    Only ever summarizes the *delta* since `history_summarized_through`
    (not the whole history again each time) - otherwise this call's own
    input would grow right along with the thing it's meant to bound.
    """
    messages = state["messages"]
    already_summarized_through = state.get("history_summarized_through", 0)
    # Everything from here on stays outside the summary, in
    # build_prompt_messages's always-verbatim recent window - only messages
    # before this index are ever eligible to be folded in.
    foldable_up_to = len(messages) - RECENT_MESSAGE_COUNT
    if foldable_up_to <= already_summarized_through:
        return None  # nothing new since the last summary to fold in

    new_messages = messages[already_summarized_through:foldable_up_to]
    if count_tokens_approximately(new_messages) <= SUMMARIZE_TOKEN_THRESHOLD:
        return None  # not enough of a backlog yet to bother summarizing at all

    previous_summary = state.get("history_summary")
    lead_in = [HumanMessage(content=f"Previous summary: {previous_summary}")] if previous_summary else []
    response = await llm.ainvoke([SystemMessage(content=_SUMMARIZE_PROMPT), *lead_in, *new_messages])
    return {"history_summary": response.content, "history_summarized_through": foldable_up_to}
