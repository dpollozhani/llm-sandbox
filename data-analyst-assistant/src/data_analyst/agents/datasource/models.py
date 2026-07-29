"""Structured shape for the datasource agent's query-result tool, and for the
subset of it that flows on into the orchestrator's `data_context`."""
from __future__ import annotations

from pydantic import BaseModel, Field


class DataSourceQueryResult(BaseModel):
    """What `pbi_rest_run_dax_query` (`agents/datasource/nodes.py`) hands
    back for one successful query - the id to reference the staged result
    by, the query that produced it (`group_by`/`filters`/`measures` in plain
    `table.column` form, formatted by the tool from its `DaxQuerySpec`
    before constructing this), its row count, a `preview` of the first few
    rows, whether it was served from this session's cache (`reused`), and
    the DAX text actually sent (`dax_query`, `None` on a cache hit - nothing
    was sent). This is also what `agents/orchestrator/nodes.py::_run_specialist`
    reads straight from the tool's own structured result (rather than a
    specialist's own freeform summary, which has no guarantee of mentioning
    all of it) and stores in `data_context` between specialist delegations -
    `preview` and `dax_query` just go along for the ride there, unused by
    `.describe()`.

    Deliberately holds no more than a `preview` of the actual rows - the
    full result set lives only in the session's `SandboxClient` store
    (`clients/sandbox/client.py`), staged under `dataset_id`, and is loaded
    from there by `dataset_id` (never re-embedded here) whenever the
    analysis agent's `python_sandbox_execute` tool runs code against it."""

    dataset_id: str
    model_name: str
    group_by: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    measures: list[str] = Field(default_factory=list)
    row_count: int
    preview: list[dict] = Field(default_factory=list)
    reused: bool = False
    dax_query: str | None = None

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
