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

from data_analyst.agents.orchestrator.state import OrchestratorState

RECENT_MESSAGE_COUNT = 8
"""How many of the most recent raw messages are always included verbatim in
prompt context, on top of `history_summary` - small enough to bound
per-turn token cost, generous enough to cover the current exchange (a task,
any specialist fold-backs, a clarification round trip) without relying
solely on the summary for it."""

SUMMARIZE_THRESHOLD = 16
"""Once `messages` grows past this many entries, `maybe_summarize_history`
folds the messages older than `RECENT_MESSAGE_COUNT` into `history_summary`
- comfortably above `MAX_TURNS * 2` (agents/orchestrator/nodes.py) so this
doesn't trigger inside a single supervisor turn's own back-and-forth."""

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


async def maybe_summarize_history(llm: BaseChatModel, state: OrchestratorState) -> dict | None:
    """A `{"history_summary": ..., "history_summarized_through": ...}` state
    update if `state["messages"]` has grown enough since the last summary to
    warrant folding more of it in - `None` if nothing needs to change yet
    (the common case), so `build_supervisor_node` can skip touching either
    field at all most turns.

    Only ever summarizes the *delta* since `history_summarized_through`
    (not the whole history again each time) - otherwise this call's own
    input would grow right along with the thing it's meant to bound.
    """
    messages = state["messages"]
    cutoff = len(messages) - RECENT_MESSAGE_COUNT
    if len(messages) <= SUMMARIZE_THRESHOLD or cutoff <= state.get("history_summarized_through", 0):
        return None
    new_messages = messages[state.get("history_summarized_through", 0) : cutoff]
    if not new_messages:
        return None

    previous = state.get("history_summary")
    lead_in = [HumanMessage(content=f"Previous summary: {previous}")] if previous else []
    response = await llm.ainvoke([SystemMessage(content=_SUMMARIZE_PROMPT), *lead_in, *new_messages])
    return {"history_summary": response.content, "history_summarized_through": cutoff}
