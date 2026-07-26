"""Structured DAX query support: every query the datasource agent can run is
built from structured group-by columns, filters, and measures - never
free-form DAX text from the model - as one of two query shapes:

- with at least one `group_by` column: `EVALUATE SUMMARIZECOLUMNS(...)`,
  filters folded in as `FILTER(ALL(...), ...)` table arguments.
- with none (a grand total, not broken out by anything):
  `EVALUATE ROW(...)` instead - `SUMMARIZECOLUMNS` requires at least one
  group-by column syntactically, it has no "just give me the totals" mode,
  and `ROW` is DAX's own idiom for that. Filters have nowhere to attach on
  `ROW` (no table arguments), so each measure expression is wrapped in
  `CALCULATE(<expr>, <filter>, ...)` instead.

`DaxQuerySpec` is the structural request; `build_dax_query` renders it to
DAX text; `validate_dax_query` checks that text (and the spec behind it)
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


def _bracketed(name: str) -> str:
    """Wrap `name` in exactly one pair of square brackets - an existing
    model measure is always referenced this way, with no table qualifier.
    Strips a pair the caller (or an overly-helpful model) already added,
    so a name like "[Inventory on-hand]" doesn't double up into
    "[[Inventory on-hand]]", which Power BI would reject."""
    if name.startswith("[") and name.endswith("]"):
        name = name[1:-1]
    return f"[{name}]"


def _filter_parts(spec: DaxQuerySpec) -> list[str]:
    parts = []
    for f in spec.filters:
        keyword = "IN" if f.operator == "IN" else f.operator
        parts.append(f"FILTER(ALL('{spec.table}'), '{spec.table}'[{f.column}] {keyword} {_literal(f.value)})")
    return parts


def _measure_expr(spec: DaxQuerySpec, m: DaxMeasure) -> str:
    expr = _bracketed(m.name) if m.aggregation is None else f"{m.aggregation}('{spec.table}'[{m.column}])"
    if spec.group_by:
        return expr  # a group-by column is already in scope to filter by; SUMMARIZECOLUMNS's own filter-table args apply
    filters = _filter_parts(spec)
    return f"CALCULATE({expr}, {', '.join(filters)})" if filters else expr


def build_dax_query(spec: DaxQuerySpec) -> str:
    """Render a `DaxQuerySpec` to DAX text - see this module's docstring for
    why the shape (`SUMMARIZECOLUMNS` vs `ROW`) depends on whether
    `group_by` is empty. The `EVALUATE` is not optional decoration either
    way - `executeQueries` parses the query text as a full DAX query, and
    every live call without it failed with a generic "Invalid query syntax.
    A valid MDX or DAX query was expected." (confirmed against Power BI, and
    matching every official request-body example in Microsoft's own docs)."""
    measure_parts = [f'"{m.name}", {_measure_expr(spec, m)}' for m in spec.measures]

    if spec.group_by:
        parts = [f"'{spec.table}'[{col}]" for col in spec.group_by] + _filter_parts(spec) + measure_parts
        inner = ",\n    ".join(parts)
        return f"EVALUATE SUMMARIZECOLUMNS(\n    {inner}\n)"

    inner = ",\n    ".join(measure_parts)
    return f"EVALUATE ROW(\n    {inner}\n)"


def validate_dax_query(dax_query: str, spec: DaxQuerySpec) -> None:
    """Structural validation before the query would be sent to the REST
    endpoint. Raises ValueError with a specific reason on failure."""
    if not spec.group_by and not spec.measures:
        raise ValueError("Query must select at least one group-by column or measure")

    text = dax_query.strip()
    expected_prefix = "EVALUATE SUMMARIZECOLUMNS(" if spec.group_by else "EVALUATE ROW("
    if not text.startswith(expected_prefix) or not text.endswith(")"):
        raise ValueError(f"DAX query must be a single {expected_prefix}...) call")
    if text.count("(") != text.count(")"):
        raise ValueError("DAX query has unbalanced parentheses")


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
