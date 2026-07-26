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
    data_context: dict | None
    """The most recently fetched dataset (set by `_run_specialist` from the
    datasource tool's own structured result, not a specialist's freeform
    summary of it), as a plain dict - `FetchedDataset(**data_context)` to
    work with it (e.g. `.describe()` to render it into a prompt). Kept as a
    dict rather than the `FetchedDataset` model itself because this field is
    checkpointed: LangGraph's serializer only has provisional, soon-to-be-
    removed support for arbitrary custom types, but a plain dict of
    strings/ints/lists is unconditionally safe to checkpoint on any
    backend. Threaded into the supervisor's routing prompt and into the
    analysis specialist's seed message, so a follow-up question can reuse
    already-fetched data instead of triggering a new datasource delegation."""
    clarification_options: list[str] | None
    """Set alongside a "clarify" outcome - either the supervisor's own
    upfront decision (`build_clarify_node`) or a specialist's
    `request_clarification` tool call (`_run_specialist`) - to the 2-3
    options the user can pick from. Read by `app/api.py`'s `ChatResponse` so
    a frontend can render them as buttons instead of requiring free text."""
    awaiting_clarification: bool
    """True right after any clarifying question was asked (supervisor's own
    upfront one, or a specialist's mid-task one), cleared once a turn
    finishes without asking another. `_run_specialist` reads this to decide
    how much history to seed a specialist with: a fresh task gets only the
    latest user message (see `_latest_user_task`), but a reply to a
    clarifying question needs the specialist to see the *whole* exchange
    (the original ask, its own question, the user's answer) - a specialist
    subgraph is rebuilt from scratch on every delegation and remembers
    nothing between turns on its own, so without this a clarification reply
    would land as an entirely new, context-free request instead of a
    continuation of the one that prompted the question."""
