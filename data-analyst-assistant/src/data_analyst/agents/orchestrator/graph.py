"""Orchestrator: a supervisor loop that delegates to the datasource/analysis
specialists and answers once it has enough information.

    START -> supervisor -> {datasource, analysis} -> supervisor -> ... -> respond -> END

Only this graph is compiled with a checkpointer. Because a specialist's
`.invoke()` inside a node function picks up the *ambient* LangChain
`RunnableConfig` (checkpointer + thread_id) when it isn't given its own
`config=`, an `interrupt()` raised deep inside the datasource specialist's
`pbi_rest_trigger_dataset_refresh` tool still pauses (and, on
`Command(resume=...)`, resumes) this orchestrator run - no extra plumbing
needed to bridge parent and child.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import build_analysis_node, build_datasource_node, build_respond_node, build_supervisor_node
from .state import OrchestratorState


def build_orchestrator_graph(
    llm: BaseChatModel, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    graph = StateGraph(OrchestratorState)
    graph.add_node("supervisor", build_supervisor_node(llm))
    graph.add_node("datasource", build_datasource_node(llm))
    graph.add_node("analysis", build_analysis_node(llm))
    graph.add_node("respond", build_respond_node(llm))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {"datasource": "datasource", "analysis": "analysis", "respond": "respond"},
    )
    graph.add_edge("datasource", "supervisor")
    graph.add_edge("analysis", "supervisor")
    graph.add_edge("respond", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
