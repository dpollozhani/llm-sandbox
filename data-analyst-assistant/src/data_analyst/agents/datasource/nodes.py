"""Datasource agent: read-only PBI MCP + PBI REST tools, plus the graph nodes
that use them. No tool here mutates anything in Power BI - metadata lookups
and structured DAX queries only.

Tools are async because a real implementation would await network calls
here (Power BI MCP/REST, both already async - see clients/powerbi/); the
mocked bodies just don't have anything to actually await.
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, ToolNode

from data_analyst.agents.common.tools import request_clarification
from data_analyst.agents.datasource.chains import build_agent_chain
from data_analyst.agents.datasource.state import DatasourceState
from data_analyst.clients.powerbi.dax import DaxFilter, DaxMeasure, DaxQuerySpec
from data_analyst.clients.powerbi.mcp import PBIMcpClient
from data_analyst.clients.powerbi.rest import PBIRestClient
from data_analyst.clients.sandbox.client import get_sandbox_client

_mcp_client = PBIMcpClient()
_rest_client = PBIRestClient()


@tool
async def pbi_mcp_list_semantic_models() -> list[dict]:
    """List semantic models (datasets) reachable through the Power BI MCP server."""
    return await _mcp_client.list_semantic_models()


@tool
async def pbi_rest_list_workspaces() -> list[dict]:
    """List Power BI workspaces and their datasets via the PBI REST API."""
    return await _rest_client.list_workspaces()


@tool
async def pbi_rest_get_refresh_history(dataset_id: str) -> list[dict]:
    """Get the refresh history for a dataset via the PBI REST API."""
    return await _rest_client.get_refresh_history(dataset_id)


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
    `group_by` columns, `filters`, and `measures` - never free-form DAX text
    - and is validated before being sent. Pass empty lists for anything not
    needed, but at least one of `group_by` or `measures` is required.

    If this exact query (same table/columns/filters/measures) was already run
    earlier in this conversation, the cached result is reused instead of
    issuing a new query - check the `reused` field in the response.

    Returns a preview of the resulting rows plus a `sandbox_ref` that the
    analysis agent can use to load the full result as a DataFrame.
    """
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
        dax_query, df = await _rest_client.run_dax_query(spec)
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


TOOLS = [
    pbi_mcp_list_semantic_models,
    pbi_rest_list_workspaces,
    pbi_rest_get_refresh_history,
    pbi_rest_run_dax_query,
    request_clarification,
]


def build_agent_node(llm: BaseChatModel):
    chain = build_agent_chain(llm, TOOLS)

    async def agent_node(state: DatasourceState):
        response = await chain.ainvoke(state["messages"])
        return {"messages": [response]}

    return agent_node


def build_tool_node() -> ToolNode:
    return ToolNode(TOOLS)
