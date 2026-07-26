"""Power BI MCP client - calls the remote Power BI MCP server's semantic
metadata tool (tables/columns/measures/relationships for one semantic
model) over streamable HTTP. That's the only capability of this server this
app uses. There's no "list all models" tool anywhere in this build - the
model learns valid `model_name` values from the static catalog
(`config/semantic_models.yaml`), injected into the datasource agent's
system prompt (see `agents/datasource/chains.py`).

The tool's exact machine name and its dataset-id argument's exact key
aren't hardcoded - Microsoft's own docs refer to it inconsistently (e.g.
"Get Semantic Model Schema" in some places, "GetSemanticMetadata" in
others), and neither is confirmed against every tenant/server version. So
each call first asks the live server for its own tool list
(`session.list_tools()`) and picks whichever tool looks like the semantic
metadata one, then reads that tool's own `inputSchema` to find which
property should hold the dataset id - ground truth from the server itself,
not another guess to verify by redeploying.

Delegated auth only (see `clients/powerbi/auth.py`'s module docstring) - the
caller's own access token is sent as a Bearer header on the MCP transport,
same as any other call to this server; this client holds no auth state.

The exact shape of the tool's JSON payload (tables/columns/measures/
relationships) isn't pinned down here beyond "valid JSON" - it's passed
through as-is for the datasource agent's model to read, rather than forced
into a schema that might not match the live server.
"""
from __future__ import annotations

import json

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool

from data_analyst.config.settings import PowerBiCatalog, get_catalog, get_settings
from data_analyst.telemetry.logging import get_logger
from data_analyst.telemetry.tracing import trace_span

_logger = get_logger("clients.powerbi.mcp")


def _find_semantic_metadata_tool(tools: list[Tool]) -> Tool | None:
    for t in tools:
        name = t.name.lower()
        if "semantic" in name and ("metadata" in name or "schema" in name):
            return t
    return None


def _dataset_arg_name(tool: Tool) -> str:
    properties = (tool.inputSchema or {}).get("properties", {})
    for key in properties:
        if any(word in key.lower() for word in ("dataset", "model", "semantic", "artifact")):
            return key
    if len(properties) == 1:
        # Whatever it's called, a single-argument tool has nowhere else the
        # dataset id could go (e.g. Fabric's MCP server calls it
        # "artifactId" - its term for a workspace item, not a keyword we'd
        # otherwise think to look for).
        return next(iter(properties))
    return "datasetId"  # best-effort fallback if the schema doesn't say


class PBIMcpClient:
    def __init__(self, catalog: PowerBiCatalog | None = None, server_url: str | None = None) -> None:
        self._catalog = catalog or get_catalog()
        self._server_url = server_url or get_settings().pbi_mcp_server_url

    async def get_semantic_metadata(self, access_token: str, model_name: str) -> dict:
        """Fetch the semantic model schema for `model_name` (resolved to a
        dataset id via the catalog config). Raises ValueError if the model
        is unknown, no matching tool is advertised by the server, or the
        tool call fails."""
        with trace_span("pbi_mcp.get_semantic_metadata", model_name=model_name):
            model = self._catalog.find_model(model_name)
            if model is None:
                raise ValueError(f"Unknown semantic model '{model_name}'")

            headers = {"Authorization": f"Bearer {access_token}"}
            async with streamablehttp_client(self._server_url, headers=headers) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    available = (await session.list_tools()).tools
                    _logger.info(
                        "MCP server tools: %s",
                        [(t.name, list((t.inputSchema or {}).get("properties", {}))) for t in available],
                    )
                    tool = _find_semantic_metadata_tool(available)
                    if tool is None:
                        raise ValueError(
                            "No semantic-metadata tool advertised by the MCP server "
                            f"(available: {[t.name for t in available]})"
                        )

                    arg_name = _dataset_arg_name(tool)
                    result = await session.call_tool(tool.name, {arg_name: model.dataset_id})

            texts = [c.text for c in result.content if hasattr(c, "text")]
            # INFO, not DEBUG (trace_span's own logging is DEBUG-only, and
            # LOG_LEVEL defaults to INFO) - a 200 OK at the HTTP layer says
            # nothing about whether the *tool call* itself actually
            # succeeded or returned anything useful.
            preview = "; ".join(texts)[:2000]
            _logger.info("%s(%s=%s) isError=%s result=%s", tool.name, arg_name, model.dataset_id, result.isError, preview)

            if result.isError:
                raise ValueError(f"{tool.name} failed: {'; '.join(texts) or 'unknown error'}")
            if not texts:
                raise ValueError(f"{tool.name} returned no content")
            return json.loads(texts[0])
