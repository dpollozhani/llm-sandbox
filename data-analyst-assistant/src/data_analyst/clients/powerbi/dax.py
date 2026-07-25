"""Structured DAX query support: every query the datasource agent can run is
built as a single `SUMMARIZECOLUMNS(...)` call from structured group-by
columns, filters, and measures - never free-form DAX text from the model.

`DaxQuerySpec` is the structural request; `build_summarizecolumns` renders it
to DAX text; `validate_dax_query` checks that text (and the spec behind it)
before it would be sent to the REST endpoint; `execute_query` is the mocked
query engine standing in for what a real semantic model would compute.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

FilterOperator = Literal["=", "!=", ">", ">=", "<", "<=", "IN"]
Aggregation = Literal["SUM", "AVERAGE", "COUNT", "MIN", "MAX"]

_PANDAS_AGG = {"SUM": "sum", "AVERAGE": "mean", "COUNT": "count", "MIN": "min", "MAX": "max"}


class DaxFilter(BaseModel):
    column: str = Field(description="Column to filter on, e.g. 'Region'.")
    operator: FilterOperator = Field(description="Comparison operator.")
    value: str | float | int | list[str | float | int] = Field(
        description="Value to compare against. Use a list only with operator 'IN'."
    )


class DaxMeasure(BaseModel):
    name: str = Field(description="Output name for the aggregated value, e.g. 'Total Revenue'.")
    aggregation: Aggregation = Field(description="Aggregation function to apply.")
    column: str = Field(description="Column to aggregate, e.g. 'Revenue'.")


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
        parts.append(f'"{m.name}", {m.aggregation}(\'{spec.table}\'[{m.column}])')
    inner = ",\n    ".join(parts)
    return f"SUMMARIZECOLUMNS(\n    {inner}\n)"


def validate_dax_query(dax_query: str, spec: DaxQuerySpec, known_columns: set[str]) -> None:
    """Structural validation before the query would be sent to the REST
    endpoint. Raises ValueError with a specific reason on failure."""
    text = dax_query.strip()
    if not text.startswith("SUMMARIZECOLUMNS(") or not text.endswith(")"):
        raise ValueError("DAX query must be a single SUMMARIZECOLUMNS(...) call")
    if text.count("(") != text.count(")"):
        raise ValueError("DAX query has unbalanced parentheses")
    if not spec.group_by and not spec.measures:
        raise ValueError("Query must select at least one group-by column or measure")

    referenced = set(spec.group_by) | {f.column for f in spec.filters} | {m.column for m in spec.measures}
    unknown = referenced - known_columns
    if unknown:
        raise ValueError(f"Unknown column(s) for table '{spec.table}': {sorted(unknown)}")


def _apply_filter(df: pd.DataFrame, f: DaxFilter) -> pd.DataFrame:
    column = df[f.column]
    if f.operator == "=":
        mask = column == f.value
    elif f.operator == "!=":
        mask = column != f.value
    elif f.operator == ">":
        mask = column > f.value
    elif f.operator == ">=":
        mask = column >= f.value
    elif f.operator == "<":
        mask = column < f.value
    elif f.operator == "<=":
        mask = column <= f.value
    else:  # "IN"
        values = f.value if isinstance(f.value, list) else [f.value]
        mask = column.isin(values)
    return df[mask]


def execute_query(df: pd.DataFrame, spec: DaxQuerySpec) -> pd.DataFrame:
    """Mocked query engine: applies `spec` to `df` the way a real semantic
    model would evaluate the equivalent SUMMARIZECOLUMNS query."""
    result = df
    for f in spec.filters:
        result = _apply_filter(result, f)

    if spec.measures:
        agg_spec = {m.column: _PANDAS_AGG[m.aggregation] for m in spec.measures}
        rename = {m.column: m.name for m in spec.measures}
        if spec.group_by:
            result = result.groupby(spec.group_by, as_index=False).agg(agg_spec).rename(columns=rename)
        else:
            row = {m.name: getattr(result[m.column], _PANDAS_AGG[m.aggregation])() for m in spec.measures}
            result = pd.DataFrame([row])
    elif spec.group_by:
        result = result[spec.group_by].drop_duplicates().reset_index(drop=True)

    return result
