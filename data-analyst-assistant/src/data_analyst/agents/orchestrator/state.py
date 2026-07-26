from __future__ import annotations

from data_analyst.agents.common.state import ChatState


class OrchestratorState(ChatState):
    """Extends the shared `messages` state with supervisor bookkeeping.

    `turns`, `next`, and `data_context` are plain (non-reduced) fields: the
    supervisor/specialist nodes are their only writers each step, so
    LangGraph's default "overwrite" behavior is exactly what's wanted, no
    `Annotated` reducer needed.
    """

    turns: int
    next: str | None
    data_context: str | None
    """Human-readable summary of the most recently fetched dataset (set by
    `_run_specialist` after a successful datasource call, e.g. "Sales grouped
    by Region with Total Revenue (sandbox_ref=df_3)"). Threaded into the
    supervisor's routing prompt and into the analysis specialist's seed
    message, so a follow-up question can reuse already-fetched data instead
    of triggering a new datasource delegation."""
    clarification_options: list[str] | None
    """Set alongside a "clarify" outcome - either the supervisor's own
    upfront decision (`build_clarify_node`) or a specialist's
    `request_clarification` tool call (`_run_specialist`) - to the 2-3
    options the user can pick from. Read by `app/api.py`'s `ChatResponse` so
    a frontend can render them as buttons instead of requiring free text."""
