from __future__ import annotations

from data_analyst.agents.common.state import ChatState


class OrchestratorState(ChatState):
    """Extends the shared `messages` state with supervisor bookkeeping.

    `turns` and `next` are plain (non-reduced) fields: the supervisor is
    their only writer each step, so LangGraph's default "overwrite" behavior
    is exactly what's wanted, no `Annotated` reducer needed.
    """

    turns: int
    next: str | None
