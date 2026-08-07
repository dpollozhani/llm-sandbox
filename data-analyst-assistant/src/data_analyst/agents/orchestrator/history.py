"""Rolling history summarization, four steps, repeated every supervisor turn:

1. Context for any model call is `[summary-message?, *messages since the
   last summary]` - `build_prompt_messages`.
2. Check whether that backlog since the last summary has grown past
   `SUMMARIZE_TOKEN_THRESHOLD`.
3. If it has, fold it into a fresh summary and advance the "last summary"
   point to now - `refresh_context` does 2 and 3, then returns the step 1
   context built from whichever summary is current (fresh or unchanged).
4. Repeat next turn.

`OrchestratorState["messages"]` itself stays unbounded (it's the
checkpointed record, needed for audit/resume - never trimmed); only the
prompt context handed to a model is bounded this way, so per-turn token
cost stops growing linearly with conversation length. There's no separate
"always keep the last N messages verbatim" window on top of this - the
messages since the last summary *are* the recent window, whatever size
that happens to be at the time.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately

from data_analyst.agents.orchestrator.state import OrchestratorState

SUMMARIZE_TOKEN_THRESHOLD = 2000
"""Once the messages since the last summary hold more than this many
tokens - approximated via `count_tokens_approximately`, the same rough
characters-per-token metric LangChain's/LangMem's own summarization
utilities default to - `refresh_context` folds them in. Token count, not
message count: token count is what actually drives per-turn cost, and a
handful of large tool-result-laden messages can cost as much as dozens of
short ones. 2000 is an order-of-magnitude estimate - comfortably above
what one ordinary turn's own exchange would use, far below typical model
context windows - not a value tuned against real usage."""

_SUMMARIZE_PROMPT = """Summarize this part of a data-analyst conversation in
a few sentences, preserving anything a later turn would need: what the user
asked for, what data/analyses were already produced (dataset ids, models,
metrics), and any open threads or unresolved questions. Be concise - this
replaces, not supplements, the messages it covers."""


def build_prompt_messages(
    messages: list[AnyMessage], history_summary: str | None, summarized_through: int
) -> list[AnyMessage]:
    """`[summary-as-message?, *messages since the last summary]` - the
    context every model-calling node (supervisor/respond/clarify) actually
    sends, whether or not this particular call is the one that just
    refreshed the summary (only the supervisor's own `refresh_context`
    ever does that; the other two just use whatever's current)."""
    since_last_summary = messages[summarized_through:]
    if not history_summary:
        return since_last_summary
    return [HumanMessage(content=f"Summary of earlier turns: {history_summary}"), *since_last_summary]


async def refresh_context(llm: BaseChatModel, state: OrchestratorState) -> tuple[list[AnyMessage], dict]:
    """Steps 2-3-1 above, in order: measure the backlog since the last
    summary; fold it into a fresh one if it's grown past
    `SUMMARIZE_TOKEN_THRESHOLD`; return the context to use this turn
    (built from whichever summary is now current) alongside the state
    update needed to persist a fresh one - empty if nothing changed, safe
    to merge into a larger return dict unconditionally.
    """
    messages = state["messages"]
    summarized_through = state.get("history_summarized_through", 0)
    history_summary = state.get("history_summary")
    since_last_summary = messages[summarized_through:]
    update: dict = {}

    if count_tokens_approximately(since_last_summary) > SUMMARIZE_TOKEN_THRESHOLD:
        lead_in = [HumanMessage(content=f"Previous summary: {history_summary}")] if history_summary else []
        response = await llm.ainvoke([SystemMessage(content=_SUMMARIZE_PROMPT), *lead_in, *since_last_summary])
        history_summary = response.content
        summarized_through = len(messages)
        update = {"history_summary": history_summary, "history_summarized_through": summarized_through}

    return build_prompt_messages(messages, history_summary, summarized_through), update
