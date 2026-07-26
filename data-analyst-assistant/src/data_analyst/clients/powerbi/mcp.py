"""Power BI MCP client - calls the official remote Power BI MCP
server's `GetSemanticMetadata` tool (tables/columns/measures/relationships
for one semantic model) over streamable HTTP. That's the only capability of
this server this app uses. There's no "list all models" tool anywhere in
this build - the model learns valid `model_name` values from the static
catalog (`config/semantic_models.yaml`), injected into the datasource
agent's system prompt (see `agents/datasource/chains.py`).

Delegated auth only (see `clients/powerbi/auth.py`'s module docstring) - the
caller's own access token is sent as a Bearer header on the MCP transport,
same as any other call to this server; this client holds no auth state.

The exact shape of `GetSemanticMetadata`'s JSON payload (tables/columns/
measures/relationships) isn't pinned down here beyond "valid JSON" - it's
passed through as-is for the datasource agent's model to read, rather than
forced into a schema that might not match the live server.
"""
from __future__ import annotations

import json

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from data_analyst.config.settings import PowerBiCatalog, get_catalog, get_settings
from data_analyst.telemetry.tracing import trace_span


class PBIMcpClient:
    def __init__(self, catalog: PowerBiCatalog | None = None, server_url: str | None = None) -> None:
        self._catalog = catalog or get_catalog()
        self._server_url = server_url or get_settings().pbi_mcp_server_url

    async def get_semantic_metadata(self, access_token: str, model_name: str) -> dict:
        """Fetch `GetSemanticMetadata` for the semantic model named
        `model_name` (resolved to a dataset id via the catalog config).
        Raises ValueError if the model is unknown or the tool call fails."""
        with trace_span("pbi_mcp.get_semantic_metadata", model_name=model_name):
            model = self._catalog.find_model(model_name)
            if model is None:
                raise ValueError(f"Unknown semantic model '{model_name}'")

            headers = {"Authorization": f"Bearer {access_token}"}
            async with streamablehttp_client(self._server_url, headers=headers) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("GetSemanticMetadata", {"datasetId": model.dataset_id})

            texts = [c.text for c in result.content if hasattr(c, "text")]
            if result.isError:
                raise ValueError(f"GetSemanticMetadata failed: {'; '.join(texts) or 'unknown error'}")
            if not texts:
                raise ValueError("GetSemanticMetadata returned no content")
            return json.loads(texts[0])
