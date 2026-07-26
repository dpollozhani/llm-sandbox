"""Orchestrator: a supervisor loop that delegates to the datasource/analysis
specialists and answers once it has enough information.

    START -> supervisor -> {datasource, analysis} -> supervisor -> ... -> {respond, clarify} -> END
                                  |            |
                                  +-- END <----+   (a specialist can short-circuit straight to a
                                                     clarifying question - see nodes.py)

This graph is compiled with a checkpointer so a `thread_id` scopes a
resumable, multi-turn conversation (see app/lifespan.py). Both specialists
are read-only in this build (see agents/datasource/nodes.py), so there's
currently nothing that pauses mid-run - but because a specialist's
`.ainvoke()` inside a node function picks up the *ambient* LangChain
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
from data_analyst.clients.powerbi.mcp import PBIMcpClient
from data_analyst.clients.powerbi.rest import PBIRestClient
from data_analyst.config.settings import Glossary, get_glossary


def _after_specialist(state: OrchestratorState) -> str:
    # A specialist normally leaves `next` as whatever the supervisor set it
    # to ("datasource"/"analysis") and loops back for the next routing
    # decision. It's only ever "clarify" here if `_run_specialist` just set
    # it because the specialist itself asked a clarifying question (see
    # nodes.py) - in which case there's nothing left for the supervisor to
    # decide this turn, so skip straight to END instead of paying for an
    # extra supervisor call (and the separate "clarify" node, which only
    # handles the supervisor's *own* upfront clarify decision).
    return "end" if state.get("next") == "clarify" else "supervisor"


def build_orchestrator_graph(
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
    mcp_client: PBIMcpClient | None = None,
    rest_client: PBIRestClient | None = None,
    glossary: Glossary | None = None,
) -> CompiledStateGraph:
    # Fetched once and threaded to every node, rather than each node/subgraph
    # independently calling get_glossary() - one source of truth per graph
    # build, and a single override point for callers/tests.
    glossary = glossary or get_glossary()
    graph = StateGraph(OrchestratorState)
    graph.add_node("supervisor", build_supervisor_node(llm, glossary=glossary))
    graph.add_node(
        "datasource", build_datasource_node(llm, mcp_client=mcp_client, rest_client=rest_client, glossary=glossary)
    )
    graph.add_node("analysis", build_analysis_node(llm, glossary=glossary))
    graph.add_node("respond", build_respond_node(llm, glossary=glossary))
    graph.add_node("clarify", build_clarify_node(llm, glossary=glossary))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {"datasource": "datasource", "analysis": "analysis", "respond": "respond", "clarify": "clarify"},
    )
    graph.add_conditional_edges("datasource", _after_specialist, {"supervisor": "supervisor", "end": END})
    graph.add_conditional_edges("analysis", _after_specialist, {"supervisor": "supervisor", "end": END})
    graph.add_edge("respond", END)
    graph.add_edge("clarify", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
