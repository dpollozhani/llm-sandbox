"""Analysis subgraph: a small ReAct-style loop scoped to the sandbox tool."""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tools_condition

from .nodes import build_agent_node, build_tool_node
from .state import AnalysisState


def build_analysis_graph(
    llm: BaseChatModel, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    graph = StateGraph(AnalysisState)
    graph.add_node("agent", build_agent_node(llm))
    graph.add_node("tools", build_tool_node())

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)
