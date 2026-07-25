"""Mocked Power BI MCP client - semantic model discovery/metadata only.

In production this would be a Model Context Protocol client talking to a PBI
MCP server over the network, hence async. Query execution is handled through
the REST client instead (see `rest.py::PBIRestClient.run_dax_query`,
mirroring the real Power BI REST API's "Execute Queries" endpoint), not
through MCP.
"""
from __future__ import annotations

from data_analyst.clients.powerbi.auth import get_bearer_token
from data_analyst.config.settings import PowerBiCatalog, get_catalog
from data_analyst.telemetry.tracing import trace_span


class PBIMcpClient:
    def __init__(self, catalog: PowerBiCatalog | None = None) -> None:
        self._catalog = catalog or get_catalog()

    async def list_semantic_models(self) -> list[dict]:
        with trace_span("pbi_mcp.list_semantic_models"):
            await get_bearer_token()
            return [m.model_dump() for m in self._catalog.semantic_models]
