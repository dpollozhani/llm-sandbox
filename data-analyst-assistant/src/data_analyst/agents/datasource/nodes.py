"""Datasource agent: read-only PBI MCP + PBI REST tools, plus the graph nodes
that use them. No tool here mutates anything in Power BI - metadata lookups
and DAX queries only."""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from ...clients.powerbi.mcp import PBIMcpClient
from ...clients.powerbi.rest import PBIRestClient
from ...clients.sandbox.client import sandbox_client
from .chains import build_agent_chain
from .state import DatasourceState

_mcp_client = PBIMcpClient()
_rest_client = PBIRestClient()


@tool
def pbi_mcp_list_semantic_models() -> list[dict]:
    """List semantic models (datasets) reachable through the Power BI MCP server."""
    return _mcp_client.list_semantic_models()


@tool
def pbi_rest_list_workspaces() -> list[dict]:
    """List Power BI workspaces and their datasets via the PBI REST API."""
    return _rest_client.list_workspaces()


@tool
def pbi_rest_get_refresh_history(dataset_id: str) -> list[dict]:
    """Get the refresh history for a dataset via the PBI REST API."""
    return _rest_client.get_refresh_history(dataset_id)


@tool
def pbi_rest_run_dax_query(model_name: str, dax_query: str) -> dict:
    """Run a DAX query against a Power BI semantic model via the PBI REST API.

    Returns a preview of the resulting rows plus a `sandbox_ref` that the
    analysis agent can use to load the full result as a DataFrame.
    """
    df = _rest_client.run_dax_query(model_name, dax_query)
    ref = sandbox_client.stage(df)
    return {"sandbox_ref": ref, "row_count": len(df), "preview": df.head(5).to_dict(orient="records")}


TOOLS = [
    pbi_mcp_list_semantic_models,
    pbi_rest_list_workspaces,
    pbi_rest_get_refresh_history,
    pbi_rest_run_dax_query,
]


def build_agent_node(llm: BaseChatModel):
    chain = build_agent_chain(llm, TOOLS)

    def agent_node(state: DatasourceState):
        response = chain.invoke(state["messages"])
        return {"messages": [response]}

    return agent_node


def build_tool_node() -> ToolNode:
    return ToolNode(TOOLS)
