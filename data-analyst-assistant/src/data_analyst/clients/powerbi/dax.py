"""Structured DAX query support: every query the datasource agent can run is
built as a single `SUMMARIZECOLUMNS(...)` call from structured group-by
columns, filters, and measures - never free-form DAX text from the model.

`DaxQuerySpec` is the structural request; `build_summarizecolumns` renders it
to DAX text; `validate_dax_query` checks that text (and the spec behind it)
structurally before it would be sent to the REST endpoint - it doesn't know
the real model's columns, so an unknown column still comes back as an error
from Power BI itself (see `rest.py::PBIRestClient.run_dax_query`), just like
any other query mistake the agent can retry after; `parse_execute_queries_response`
turns the REST API's `executeQueries` response back into a DataFrame.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

FilterOperator = Literal["=", "!=", ">", ">=", "<", "<=", "IN"]
Aggregation = Literal["SUM", "AVERAGE", "COUNT", "MIN", "MAX"]


class DaxFilter(BaseModel):
    column: str = Field(description="Column to filter on, e.g. 'Region'.")
    operator: FilterOperator = Field(description="Comparison operator.")
    value: str | float | int | list[str | float | int] = Field(
        description="Value to compare against. Use a list only with operator 'IN'."
    )


class DaxMeasure(BaseModel):
    """Either a reference to a measure that already exists in the model
    (give `name` only - schemas commonly ship these, e.g. under a
    "_Measures" table - and it's addressed directly, never re-aggregated),
    or an ad-hoc aggregation over a raw column (`aggregation` + `column`,
    with `name` as the output label)."""

    name: str = Field(description="An existing model measure's name, or an output label for the aggregation below.")
    aggregation: Aggregation | None = Field(
        default=None, description="Omit to reference an existing model measure by `name`; set to aggregate `column`."
    )
    column: str | None = Field(default=None, description="Column to aggregate. Required only if `aggregation` is set.")

    @model_validator(mode="after")
    def _check_aggregation_needs_column(self) -> "DaxMeasure":
        if self.aggregation is not None and self.column is None:
            raise ValueError("`column` is required when `aggregation` is set")
        return self


class DaxQuerySpec(BaseModel):
    model_name: str
    table: str
    group_by: list[str] = Field(default_factory=list, description="Columns to group by.")
    filters: list[DaxFilter] = Field(default_factory=list)
    measures: list[DaxMeasure] = Field(default_factory=list)

    def cache_key(self) -> str:
        """A stable key for two structurally-equivalent specs, regardless of
        incidental list ordering - used to detect "we already fetched this"."""
        payload = {
            "model_name": self.model_name,
            "table": self.table,
            "group_by": sorted(self.group_by),
            "filters": sorted((f.column, f.operator, str(f.value)) for f in self.filters),
            "measures": sorted((m.name, m.aggregation, m.column) for m in self.measures),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _literal(value: object) -> str:
    if isinstance(value, list):
        return "{" + ", ".join(_literal(v) for v in value) + "}"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def build_summarizecolumns(spec: DaxQuerySpec) -> str:
    """Render a `DaxQuerySpec` to a `SUMMARIZECOLUMNS(...)` DAX string."""
    parts: list[str] = [f"'{spec.table}'[{col}]" for col in spec.group_by]
    for f in spec.filters:
        keyword = "IN" if f.operator == "IN" else f.operator
        parts.append(f"FILTER(ALL('{spec.table}'), '{spec.table}'[{f.column}] {keyword} {_literal(f.value)})")
    for m in spec.measures:
        if m.aggregation is None:
            parts.append(f'"{m.name}", [{m.name}]')
        else:
            parts.append(f'"{m.name}", {m.aggregation}(\'{spec.table}\'[{m.column}])')
    inner = ",\n    ".join(parts)
    return f"SUMMARIZECOLUMNS(\n    {inner}\n)"


def validate_dax_query(dax_query: str, spec: DaxQuerySpec) -> None:
    """Structural validation before the query would be sent to the REST
    endpoint. Raises ValueError with a specific reason on failure."""
    text = dax_query.strip()
    if not text.startswith("SUMMARIZECOLUMNS(") or not text.endswith(")"):
        raise ValueError("DAX query must be a single SUMMARIZECOLUMNS(...) call")
    if text.count("(") != text.count(")"):
        raise ValueError("DAX query has unbalanced parentheses")
    if not spec.group_by and not spec.measures:
        raise ValueError("Query must select at least one group-by column or measure")


def _result_key(keys: list[str], name: str, table: str | None = None) -> str:
    """Match `name` (a group-by column or measure name from the spec) against
    one of the column headers Power BI actually returned. Group-by columns
    can come back as `'Table'[Column]`, `Table[Column]`, or bare `[Column]`
    depending on the model; measures come back as the plain name given in
    the query. Raises ValueError (surfaced to the agent, not raised past the
    tool) if nothing matches."""
    candidates = [name, f"[{name}]"]
    if table:
        candidates += [f"'{table}'[{name}]", f"{table}[{name}]"]
    for candidate in candidates:
        if candidate in keys:
            return candidate
    suffix = f"[{name}]"
    for key in keys:
        if key.endswith(suffix):
            return key
    raise ValueError(f"Column '{name}' not found in query result columns: {keys}")


def parse_execute_queries_response(response: dict, spec: DaxQuerySpec) -> pd.DataFrame:
    """Turn a Power BI REST `executeQueries` response body back into a
    DataFrame with friendly column names (the spec's own `group_by`/measure
    names), for the analysis agent to work with by `sandbox_ref`."""
    try:
        rows = response["results"][0]["tables"][0]["rows"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected executeQueries response shape: {response!r}") from exc

    wanted = [*spec.group_by, *(m.name for m in spec.measures)]
    if not rows:
        return pd.DataFrame(columns=wanted)

    keys = list(rows[0].keys())
    rename = {_result_key(keys, col, table=spec.table): col for col in spec.group_by}
    rename.update({_result_key(keys, m.name): m.name for m in spec.measures})

    return pd.DataFrame(rows).rename(columns=rename)[wanted]
