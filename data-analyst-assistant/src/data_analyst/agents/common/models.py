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
    """A short question plus 2-3 clearly distinct options a user can pick
    from, so a frontend can render them as buttons instead of relying on
    free text. Produced two ways, both read into this same shape: the
    supervisor's own upfront "clarify" decision
    (`agents/orchestrator/chains.py::build_clarify_chain`, via structured
    output), where `question` is already the ready-to-send text; and a
    specialist's `flag_ambiguity` tool call (`agents/common/tools.py`),
    where `question` is really just the specialist's own reason for the
    ambiguity - not itself user-facing.
    `agents/orchestrator/nodes.py::_run_specialist` is the only place that
    decides how (and whether) to surface the latter, composing the final
    message deterministically (`_compose_ambiguity_message`) rather than
    relaying the specialist's own phrasing as if a model had written it for
    the user. One shared model rather than two near-identical ones, since
    the distinction is in which code path produced it and what that path
    does with it next, not in the shape itself."""

    question: str = ""
    options: list[str] = Field(default_factory=lambda: ["", ""], min_length=2, max_length=3)
