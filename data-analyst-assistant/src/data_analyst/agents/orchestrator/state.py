from __future__ import annotations

from typing_extensions import NotRequired

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
    summary of it), as a plain dict - `DataSourceQueryResult(**data_context)`
    (`agents/datasource/models.py`) to work with it (e.g. `.describe()` to
    render it into a prompt). Kept as a dict rather than the
    `DataSourceQueryResult` model itself because this field is checkpointed:
    LangGraph's serializer only has provisional, soon-to-be-removed support
    for arbitrary custom types, but a plain dict of strings/ints/lists is
    unconditionally safe to checkpoint on any backend. Threaded into the
    supervisor's routing prompt and into the analysis specialist's seed
    message, so a follow-up question can reuse already-fetched data instead
    of triggering a new datasource delegation."""
    pending_clarification: dict | None
    """{"agent": "datasource" | "analysis" | "supervisor", "reason": str,
    "options": list[str]} - who is waiting on a reply and why, whether from
    the supervisor's own upfront decision (`build_clarify_node`) or a
    specialist's `flag_ambiguity` tool call (`_run_specialist`). The single
    source of truth for "is a clarification outstanding": replaces the
    former separate `awaiting_clarification`/`clarification_options` pair
    so this fact lives in one place, not two that can drift out of sync.
    Read by `build_supervisor_node` to resume straight into the specialist
    that asked (skipping a fresh routing decision) and by `app/api.py`'s
    `ChatResponse` (`options`) so a frontend can render buttons instead of
    requiring free text. Cleared (set to `None`) once the reply resolves it."""
    resolved_clarifications: list[dict]
    """[{"question": str, "answer": str}, ...] - every clarification
    settled so far this conversation, appended to (read-append-overwrite,
    same convention as `turns`) by `_run_specialist` right before it clears
    `pending_clarification`. Given to a specialist at seed time (in place of
    the full raw message history a clarification reply used to be seeded
    with) so it knows what's already been settled instead of re-deriving -
    or re-asking about - it from scratch."""
    followup_suggestion: dict | None
    """{"agent": "analysis", "question": str, "options": list[str]} - the
    analysis specialist's `suggest_followup` tool call this run (see
    `agents/common/tools.py`), if any - only that specialist has this tool;
    the datasource agent's job is complete once it's fetched data at the
    right grain, with no genuine fork of its own worth suggesting (see
    `agents/datasource/prompts.py`). Unlike `pending_clarification`, this
    is non-blocking: the specialist already produced a real, complete
    answer this run - this is purely a supplementary "here's an optional
    next step" alongside it, not something the orchestrator waits on before
    doing anything else. Read by `build_supervisor_node` (as one input to
    its normal routing decision, not a deterministic bypass the way
    `pending_clarification` is - the user's next message might pick this
    up, or might not, so an LLM call still decides) and by `app/api.py`'s
    `ChatResponse` (`suggested_options`) so a frontend can render buttons
    alongside the real answer instead of the model restating them in prose.
    Always explicitly set (to a new value or `None`) on every
    `_run_specialist` return path, the same convention as
    `pending_clarification`, so a stale suggestion from an earlier turn
    never lingers."""
    history_summary: NotRequired[str]
    """A running summary of conversation turns old enough to have been
    folded out of the supervisor/respond/clarify nodes' own prompt context
    (see `agents/orchestrator/history.py`) - `messages` itself is never
    trimmed (it's the checkpointed record), only what gets sent to the
    model each call. `NotRequired`: absent until a conversation is long
    enough to need one."""
    history_summarized_through: NotRequired[int]
    """How many of `messages` (by index) are already folded into
    `history_summary` - lets `pending_backlog` only ever consider the
    delta since the last summary, not the whole foldable range again each
    time. `NotRequired` for the same reason as `history_summary`."""
    allowed_model_names: NotRequired[list[str] | None]
    """The caller's own request (see `app/api.py`'s `ChatRequest.model_names`)
    to scope this session to part of the full configured catalog - e.g. a
    frontend embedded in a specific Power BI app, pre-resolving "this app ->
    these models" itself before ever calling this API. Set once, directly in
    the initial state of every `/chat`/`/chat/stream` call (the same
    convention as `pbi_token`), never written by a node's own return dict.
    Read by `agents/orchestrator/nodes.py::_effective_catalog` to narrow both
    what the supervisor/respond/datasource-agent prompts *describe* as
    available and what the datasource tools can actually resolve a
    `model_name` against (threaded into `build_datasource_graph` - see
    `build_datasource_node`) - a real restriction for the life of this
    request, not just a description one. "Overridable" means the caller can
    change or lift it on a later request (a different `model_names`, or
    omitting it) - not that the agent can reach around it within this one.
    `NotRequired`/`None`: the common case, no scoping at all."""
