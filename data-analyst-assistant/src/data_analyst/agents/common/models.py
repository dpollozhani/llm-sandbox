"""Pydantic models shared across agent packages."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """What a specialist subgraph (datasource/analysis) hands back to the
    orchestrator: enough to keep the conversation going without re-exposing
    every intermediate tool call to the supervisor's own context."""

    agent: str
    summary: str


class FetchedDataset(BaseModel):
    """Everything about one fetched-and-staged dataset in one place - the id
    to reference it by, the query that produced it (`group_by`/`filters`/
    `measures` in plain `table.column` form, from
    `clients/powerbi/dax.py::describe_query`), and its row count. This is
    what flows through the orchestrator's `data_context` between specialist
    delegations (`agents/orchestrator/nodes.py::_run_specialist`), read
    directly from the datasource tool's own structured result rather than a
    specialist's own freeform summary, which has no guarantee of mentioning
    all of it."""

    dataset_id: str
    model_name: str
    group_by: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    measures: list[str] = Field(default_factory=list)
    row_count: int

    def describe(self) -> str:
        """A short natural-language line for prompts (the supervisor's
        routing prompt, and the next specialist's seed message)."""
        parts = [f"{self.row_count} row(s) from '{self.model_name}'"]
        if self.group_by:
            parts.append(f"grouped by {', '.join(self.group_by)}")
        if self.filters:
            parts.append(f"filtered to {', '.join(self.filters)}")
        if self.measures:
            parts.append(f"with {', '.join(self.measures)}")
        return f"{'; '.join(parts)} (dataset_id={self.dataset_id})"


class Clarification(BaseModel):
    """A clarifying question with 2-3 clearly distinct options a user can
    pick from, so a frontend can render them as buttons instead of relying
    on free text. Produced by the supervisor's own upfront "clarify"
    decision (`agents/orchestrator/chains.py::build_clarify_chain`, via
    structured output) - see `Ambiguity` for what a specialist reports
    instead, mid-task."""

    question: str = ""
    options: list[str] = Field(default_factory=lambda: ["", ""], min_length=2, max_length=3)


class Ambiguity(BaseModel):
    """What a specialist reports when it can't proceed confidently, via the
    `flag_ambiguity` tool (`agents/common/tools.py`) - a `reason` (a
    complete, user-readable sentence describing the ambiguity) plus 2-3
    clearly distinct candidate options. Unlike `Clarification`, this is not
    itself a user-facing question: `agents/orchestrator/nodes.py::_run_specialist`
    is the only place that decides how (and whether) to surface it, composing
    the final message deterministically rather than relaying the specialist's
    own phrasing as an LLM-authored question would imply."""

    reason: str = ""
    options: list[str] = Field(default_factory=lambda: ["", ""], min_length=2, max_length=3)
