"""Datasource subgraph: a small ReAct-style loop scoped to the read-only PBI MCP/REST tools."""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tools_condition

from data_analyst.agents.datasource.nodes import build_agent_node, build_tool_node, build_tools
from data_analyst.agents.datasource.state import DatasourceState
from data_analyst.clients.powerbi.mcp import PBIMcpClient
from data_analyst.clients.powerbi.rest import PBIRestClient


def build_datasource_graph(
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
    mcp_client: PBIMcpClient | None = None,
    rest_client: PBIRestClient | None = None,
) -> CompiledStateGraph:
    tools = build_tools(mcp_client=mcp_client, rest_client=rest_client)
    graph = StateGraph(DatasourceState)
    graph.add_node("agent", build_agent_node(llm, tools))
    graph.add_node("tools", build_tool_node(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    # No checkpointer by default: see agents/orchestrator/graph.py for how this
    # subgraph is invoked from within an orchestrator node. Pass a
    # checkpointer explicitly to run and test this subgraph standalone (see
    # tests/integration).
    return graph.compile(checkpointer=checkpointer)
