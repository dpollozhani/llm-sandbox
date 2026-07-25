"""Orchestrator: a supervisor loop that delegates to the datasource/analysis
specialists and answers once it has enough information.

    START -> supervisor -> {datasource, analysis} -> supervisor -> ... -> {respond, clarify} -> END

This graph is compiled with a checkpointer so a `thread_id` scopes a
resumable, multi-turn conversation (see app/lifespan.py). Both specialists
are read-only in this build (see agents/datasource/nodes.py), so there's
currently nothing that pauses mid-run - but because a specialist's
`.invoke()` inside a node function picks up the *ambient* LangChain
`RunnableConfig` (checkpointer + thread_id) when it isn't given its own
`config=`, a future mutating tool could call `langgraph.types.interrupt()`
and have it pause (and, on `Command(resume=...)`, resume) this orchestrator
run with no extra plumbing to bridge parent and child.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from data_analyst.agents.orchestrator.nodes import (
    build_analysis_node,
    build_clarify_node,
    build_datasource_node,
    build_respond_node,
    build_supervisor_node,
)
from data_analyst.agents.orchestrator.state import OrchestratorState


def build_orchestrator_graph(
    llm: BaseChatModel, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    graph = StateGraph(OrchestratorState)
    graph.add_node("supervisor", build_supervisor_node(llm))
    graph.add_node("datasource", build_datasource_node(llm))
    graph.add_node("analysis", build_analysis_node(llm))
    graph.add_node("respond", build_respond_node(llm))
    graph.add_node("clarify", build_clarify_node(llm))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {"datasource": "datasource", "analysis": "analysis", "respond": "respond", "clarify": "clarify"},
    )
    graph.add_edge("datasource", "supervisor")
    graph.add_edge("analysis", "supervisor")
    graph.add_edge("respond", END)
    graph.add_edge("clarify", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
