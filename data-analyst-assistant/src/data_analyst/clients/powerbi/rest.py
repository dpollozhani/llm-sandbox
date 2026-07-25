"""Mocked Power BI REST API client - read-only (workspace/dataset metadata,
refresh history, DAX query execution). No write/admin operations (e.g.
triggering a refresh) are exposed here; the datasource agent is
intentionally read-only.

A real client would call `https://api.powerbi.com/v1.0/myorg/...`; here it
reads workspace/dataset metadata from the catalog config, refresh history
from an in-memory fixture, and "executes" DAX queries (mirroring the real
REST API's dataset "Execute Queries" endpoint) against a handful of fake
tables. Queries are always structured `SUMMARIZECOLUMNS(...)` calls built
and validated from a `DaxQuerySpec` (see `dax.py`) - never free-form DAX
text handed in directly.
"""
from __future__ import annotations

import pandas as pd

from data_analyst.clients.powerbi.auth import get_bearer_token
from data_analyst.clients.powerbi.dax import DaxQuerySpec, build_summarizecolumns, execute_query, validate_dax_query
from data_analyst.config.settings import PowerBiCatalog, get_catalog
from data_analyst.telemetry.tracing import trace_span

_REFRESH_HISTORY: dict[str, list[dict]] = {
    "ds-001": [
        {
            "request_id": "r-1",
            "status": "Completed",
            "start_time": "2026-07-24T02:00:00Z",
            "end_time": "2026-07-24T02:04:12Z",
        },
        {
            "request_id": "r-2",
            "status": "Completed",
            "start_time": "2026-07-23T02:00:00Z",
            "end_time": "2026-07-23T02:03:47Z",
        },
    ]
}

_SALES = pd.DataFrame(
    [
        {"Date": "2026-01-05", "Region": "North", "Product": "Widget A", "Quantity": 120, "Revenue": 3600.0},
        {"Date": "2026-01-05", "Region": "South", "Product": "Widget B", "Quantity": 80, "Revenue": 2400.0},
        {"Date": "2026-02-12", "Region": "North", "Product": "Widget A", "Quantity": 95, "Revenue": 2850.0},
        {"Date": "2026-02-12", "Region": "East", "Product": "Widget C", "Quantity": 60, "Revenue": 2700.0},
        {"Date": "2026-03-01", "Region": "South", "Product": "Widget B", "Quantity": 110, "Revenue": 3300.0},
        {"Date": "2026-03-15", "Region": "West", "Product": "Widget C", "Quantity": 75, "Revenue": 3375.0},
    ]
)

_PRODUCTS = pd.DataFrame(
    [
        {"Product": "Widget A", "Category": "Hardware", "UnitCost": 18.0},
        {"Product": "Widget B", "Category": "Hardware", "UnitCost": 20.0},
        {"Product": "Widget C", "Category": "Premium", "UnitCost": 32.0},
    ]
)

_REGIONS = pd.DataFrame(
    [
        {"Region": "North", "Country": "Sweden", "Manager": "A. Lind"},
        {"Region": "South", "Country": "Denmark", "Manager": "B. Holm"},
        {"Region": "East", "Country": "Finland", "Manager": "C. Saari"},
        {"Region": "West", "Country": "Norway", "Manager": "D. Berg"},
    ]
)

_TABLES: dict[str, pd.DataFrame] = {"Sales": _SALES, "Products": _PRODUCTS, "Regions": _REGIONS}


class PBIRestClient:
    def __init__(self, catalog: PowerBiCatalog | None = None) -> None:
        self._catalog = catalog or get_catalog()

    async def list_workspaces(self) -> list[dict]:
        with trace_span("pbi_rest.list_workspaces"):
            await get_bearer_token()
            return [w.model_dump() for w in self._catalog.workspaces]

    async def get_refresh_history(self, dataset_id: str) -> list[dict]:
        with trace_span("pbi_rest.get_refresh_history", dataset_id=dataset_id):
            await get_bearer_token()
            return list(_REFRESH_HISTORY.get(dataset_id, []))

    async def run_dax_query(self, spec: DaxQuerySpec) -> tuple[str, pd.DataFrame]:
        """Build, validate, and "execute" a structured DAX query.

        Returns the resolved DAX text (for transparency in the tool result)
        and the resulting DataFrame. Raises ValueError if the model/table is
        unknown or the built query fails validation. Building/validating/
        executing the query itself is pure, fast, in-memory work (no real
        I/O to await), unlike `get_bearer_token` - a real implementation
        would await an HTTP call here instead.
        """
        with trace_span("pbi_rest.run_dax_query", model_name=spec.model_name, table=spec.table):
            await get_bearer_token()
            model = self._catalog.find_model(spec.model_name)
            if model is None:
                raise ValueError(f"Unknown semantic model '{spec.model_name}'")
            if spec.table not in _TABLES:
                raise ValueError(f"Unknown table '{spec.table}' in model '{spec.model_name}'")

            dax_query = build_summarizecolumns(spec)
            validate_dax_query(dax_query, spec, known_columns=set(_TABLES[spec.table].columns))
            return dax_query, execute_query(_TABLES[spec.table], spec)
