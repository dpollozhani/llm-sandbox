"""Datasource agent: read-only PBI MCP + PBI REST tools, plus the graph nodes
that use them. No tool here mutates anything in Power BI - metadata lookups
and structured DAX queries only.

Every PBI tool reads the current request's delegated access token from
`InjectedState` (`pbi_token`, set by `app/api.py` from whichever the caller
signed in with - see `clients/powerbi/auth.py`'s module docstring for why a
delegated user token is required at all: row-level security depends on
whose identity the call runs as, so there's no app-only/service-principal
fallback here) rather than holding any auth of their own.
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState, ToolNode

from data_analyst.agents.common.tools import request_clarification
from data_analyst.agents.datasource.chains import build_agent_chain
from data_analyst.agents.datasource.state import DatasourceState
from data_analyst.clients.powerbi.dax import DaxFilter, DaxMeasure, DaxQuerySpec
from data_analyst.clients.powerbi.mcp import PBIMcpClient
from data_analyst.clients.powerbi.rest import PBIRestClient
from data_analyst.clients.sandbox.client import get_sandbox_client
from data_analyst.config.settings import PowerBiCatalog

_NOT_SIGNED_IN = "Not signed in with Power BI access for this - ask the user to sign in again (/auth/login)."


def build_tools(mcp_client: PBIMcpClient | None = None, rest_client: PBIRestClient | None = None) -> list[BaseTool]:
    """Builds the datasource agent's tools bound to `mcp_client`/`rest_client`
    (real clients by default; tests inject fakes here instead of reaching
    past the `@tool` decorators)."""
    mcp = mcp_client or PBIMcpClient()
    rest = rest_client or PBIRestClient()

    @tool
    async def pbi_mcp_get_semantic_metadata(model_name: str, state: Annotated[DatasourceState, InjectedState]) -> dict:
        """Get a semantic model's schema (tables, columns, measures,
        relationships) via the Power BI MCP server's GetSemanticMetadata.
        Call this for a model before querying it if you haven't already
        seen its schema this conversation."""
        token = state.get("pbi_token")
        if not token:
            return {"error": _NOT_SIGNED_IN}
        try:
            return await mcp.get_semantic_metadata(token, model_name)
        except ValueError as exc:
            return {"error": str(exc)}

    @tool
    async def pbi_rest_run_dax_query(
        model_name: str,
        table: str,
        group_by: list[str],
        filters: list[DaxFilter],
        measures: list[DaxMeasure],
        state: Annotated[DatasourceState, InjectedState],
    ) -> dict:
        """Run a structured query against a Power BI semantic model via the PBI REST API.

        The query is always built as a single SUMMARIZECOLUMNS(...) call from
        `group_by` columns, `filters`, and `measures` - never free-form DAX
        text - and is validated before being sent. Pass empty lists for
        anything not needed, but at least one of `group_by` or `measures` is
        required.

        If this exact query (same table/columns/filters/measures) was already
        run earlier in this conversation, the cached result is reused instead
        of issuing a new query - check the `reused` field in the response.

        Returns a preview of the resulting rows plus a `sandbox_ref` that the
        analysis agent can use to load the full result as a DataFrame.
        """
        token = state.get("pbi_token")
        if not token:
            return {"error": _NOT_SIGNED_IN}

        try:
            spec = DaxQuerySpec(
                model_name=model_name, table=table, group_by=group_by, filters=filters, measures=measures
            )
        except ValueError as exc:
            return {"error": str(exc)}

        store = get_sandbox_client(state["session_id"])
        cache_key = spec.cache_key()
        cached_ref = store.find_cached(cache_key)
        if cached_ref is not None:
            df = store.peek(cached_ref)
            return {
                "sandbox_ref": cached_ref,
                "row_count": len(df),
                "preview": df.head(5).to_dict(orient="records"),
                "reused": True,
            }

        try:
            dax_query, df = await rest.run_dax_query(token, spec)
        except ValueError as exc:
            return {"error": str(exc)}

        ref = store.stage(df)
        store.remember(cache_key, ref)
        return {
            "sandbox_ref": ref,
            "row_count": len(df),
            "preview": df.head(5).to_dict(orient="records"),
            "reused": False,
            "dax_query": dax_query,
        }

    return [
        pbi_mcp_get_semantic_metadata,
        pbi_rest_run_dax_query,
        request_clarification,
    ]


def build_agent_node(llm: BaseChatModel, tools: list[BaseTool] | None = None, catalog: PowerBiCatalog | None = None):
    tools = tools if tools is not None else build_tools()
    chain = build_agent_chain(llm, tools, catalog=catalog)

    async def agent_node(state: DatasourceState):
        response = await chain.ainvoke(state["messages"])
        return {"messages": [response]}

    return agent_node


def build_tool_node(tools: list[BaseTool] | None = None) -> ToolNode:
    return ToolNode(tools if tools is not None else build_tools())
