"""Structured shape for the datasource agent's query-result tool, and for the
subset of it that flows on into the orchestrator's `data_context`."""
from __future__ import annotations

from pydantic import BaseModel, Field

from data_analyst.clients.powerbi.dax import DaxMeasure, DaxQuerySpec


class DataSourceQueryResult(BaseModel):
    """What `pbi_rest_run_dax_query` (`agents/datasource/nodes.py`) hands
    back for one successful query - the id to reference the staged result
    by, the query that produced it (`group_by`/`filters`/`measures` in plain
    `table.column` form - see `.from_query()`), its row count, a `preview`
    of the first few rows, whether it was served from this session's cache
    (`reused`), and the DAX text actually sent (`dax_query`, `None` on a
    cache hit - nothing was sent). This is also what
    `agents/orchestrator/nodes.py::_run_specialist` reads straight from the
    tool's own structured result (rather than a specialist's own freeform
    summary, which has no guarantee of mentioning all of it) and stores in
    `data_context` between specialist delegations - `preview` and
    `dax_query` just go along for the ride there, unused by `.describe()`.

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

    @classmethod
    def from_query(
        cls,
        spec: DaxQuerySpec,
        *,
        dataset_id: str,
        row_count: int,
        preview: list[dict],
        reused: bool,
        dax_query: str | None = None,
    ) -> "DataSourceQueryResult":
        """Build the result from the `DaxQuerySpec` that produced it - the
        one place that formats its structured `group_by`/`filters`/
        `measures` into the plain `table.column` strings a user reads,
        replacing the former standalone `clients/powerbi/dax.py::describe_query`
        (which had exactly this one call site and nothing else)."""

        def _measure(m: DaxMeasure) -> str:
            return m.name if m.aggregation is None else f"{m.name} = {m.aggregation}({m.table}.{m.column})"

        return cls(
            dataset_id=dataset_id,
            model_name=spec.model_name,
            group_by=[f"{c.table}.{c.column}" for c in spec.group_by],
            filters=[f"{f.table}.{f.column} {f.operator} {f.value!r}" for f in spec.filters],
            measures=[_measure(m) for m in spec.measures],
            row_count=row_count,
            preview=preview,
            reused=reused,
            dax_query=dax_query,
        )

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
