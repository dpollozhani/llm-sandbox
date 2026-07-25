"""Mocked Power BI REST API client (workspaces, datasets, refresh operations).

A real client would call `https://api.powerbi.com/v1.0/myorg/...`; here it
reads workspace/dataset metadata from the catalog config and keeps refresh
history in memory.
"""
from __future__ import annotations

from ...config.settings import PowerBiCatalog, get_catalog
from ...telemetry.tracing import trace_span
from .auth import get_bearer_token

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


class PBIRestClient:
    def __init__(self, catalog: PowerBiCatalog | None = None) -> None:
        self._catalog = catalog or get_catalog()

    def list_workspaces(self) -> list[dict]:
        with trace_span("pbi_rest.list_workspaces"):
            get_bearer_token()
            return [w.model_dump() for w in self._catalog.workspaces]

    def get_refresh_history(self, dataset_id: str) -> list[dict]:
        with trace_span("pbi_rest.get_refresh_history", dataset_id=dataset_id):
            get_bearer_token()
            return list(_REFRESH_HISTORY.get(dataset_id, []))

    def trigger_refresh(self, dataset_id: str) -> dict:
        with trace_span("pbi_rest.trigger_refresh", dataset_id=dataset_id):
            get_bearer_token()
            history = _REFRESH_HISTORY.setdefault(dataset_id, [])
            entry = {
                "request_id": f"r-{len(history) + 1}",
                "status": "Completed",
                "start_time": "2026-07-25T00:00:00Z",
                "end_time": "2026-07-25T00:03:00Z",
            }
            history.insert(0, entry)
            return entry
