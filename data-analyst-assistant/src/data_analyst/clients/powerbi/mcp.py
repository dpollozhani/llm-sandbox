"""Mocked Power BI MCP client - semantic model discovery/metadata only.

In production this would be a Model Context Protocol client talking to a PBI
MCP server. Query execution is handled through the REST client instead (see
`rest.py::PBIRestClient.run_dax_query`, mirroring the real Power BI REST
API's "Execute Queries" endpoint), not through MCP.
"""
from __future__ import annotations

from ...config.settings import PowerBiCatalog, get_catalog
from ...telemetry.tracing import trace_span
from .auth import get_bearer_token


class PBIMcpClient:
    def __init__(self, catalog: PowerBiCatalog | None = None) -> None:
        self._catalog = catalog or get_catalog()

    def list_semantic_models(self) -> list[dict]:
        with trace_span("pbi_mcp.list_semantic_models"):
            get_bearer_token()
            return [m.model_dump() for m in self._catalog.semantic_models]
