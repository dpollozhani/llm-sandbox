"""Orchestrator nodes: a routing supervisor plus wrappers that seed a fresh
specialist subgraph from the current task and fold its answer back in.

IMPORTANT: `interrupt()` (used by the datasource agent's dataset-refresh
approval) raises `GraphInterrupt`, which is a plain `Exception` subclass.
Never wrap a specialist's `.invoke()` call below in a broad `except
Exception`, and never apply `utils/retry.py`'s `@retry` to these node
functions - either would silently swallow the pause instead of letting it
bubble up to the orchestrator's own checkpointer.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from ..analysis.graph import build_analysis_graph
from ..common.models import AgentResult
from ..datasource.graph import build_datasource_graph
from .chains import build_respond_chain, build_supervisor_chain
from .state import OrchestratorState

MAX_TURNS = 6


def build_supervisor_node(llm: BaseChatModel):
    chain = build_supervisor_chain(llm)

    def supervisor_node(state: OrchestratorState):
        turns = state.get("turns", 0)
        if turns >= MAX_TURNS:
            return {"next": "respond", "turns": turns}
        route = chain.invoke(state["messages"])
        return {"next": route.next, "turns": turns + 1}

    return supervisor_node


def _run_specialist(agent_name: str, build_graph_fn, llm: BaseChatModel, state: OrchestratorState) -> dict:
    # Seed a *fresh* child conversation from only the latest task, not the
    # orchestrator's full history - keeps each specialist's context scoped to
    # what it needs and keeps the orchestrator's own transcript short.
    task_message = state["messages"][-1]
    child_graph = build_graph_fn(llm)
    result = child_graph.invoke({"messages": [task_message]})
    last_message = result["messages"][-1]
    agent_result = AgentResult(agent=agent_name, summary=getattr(last_message, "content", str(last_message)))
    return {"messages": [AIMessage(content=f"[{agent_name}] {agent_result.summary}")]}


def build_datasource_node(llm: BaseChatModel):
    def datasource_node(state: OrchestratorState):
        return _run_specialist("datasource", build_datasource_graph, llm, state)

    return datasource_node


def build_analysis_node(llm: BaseChatModel):
    def analysis_node(state: OrchestratorState):
        return _run_specialist("analysis", build_analysis_graph, llm, state)

    return analysis_node


def build_respond_node(llm: BaseChatModel):
    chain = build_respond_chain(llm)

    def respond_node(state: OrchestratorState):
        response = chain.invoke(state["messages"])
        return {"messages": [response]}

    return respond_node
