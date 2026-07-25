"""Orchestrator nodes: a routing supervisor plus wrappers that seed a fresh
specialist subgraph from the current task and fold its answer back in.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from data_analyst.agents.analysis.graph import build_analysis_graph
from data_analyst.agents.common.models import AgentResult
from data_analyst.agents.datasource.graph import build_datasource_graph
from data_analyst.agents.orchestrator.chains import build_clarify_chain, build_respond_chain, build_supervisor_chain
from data_analyst.agents.orchestrator.state import OrchestratorState

MAX_TURNS = 6


def build_supervisor_node(llm: BaseChatModel):
    chain = build_supervisor_chain(llm)

    def supervisor_node(state: OrchestratorState):
        turns = state.get("turns", 0)
        if turns >= MAX_TURNS:
            return {"next": "respond", "turns": turns}
        route = chain.invoke({"messages": state["messages"], "data_context": state.get("data_context")})
        return {"next": route.next, "turns": turns + 1}

    return supervisor_node


def _latest_user_task(messages: list[AnyMessage]) -> AnyMessage:
    """The specialists get one scoped task per delegation, not the orchestrator's
    full history - but that task must always be the user's actual question, not
    whatever message happens to be last (which, after a first specialist has
    already run this turn, would be that specialist's own folded-back summary)."""
    for message in reversed(messages):
        if message.type == "human":
            return message
    return messages[-1]


def _run_specialist(agent_name: str, build_graph_fn, llm: BaseChatModel, state: OrchestratorState) -> dict:
    task_message = _latest_user_task(state["messages"])
    data_context = state.get("data_context")
    seed_content = f"(Available data in this session: {data_context})\n\n{task_message.content}" if data_context else task_message.content

    child_graph = build_graph_fn(llm)
    result = child_graph.invoke({"messages": [HumanMessage(content=seed_content)], "session_id": state["session_id"]})
    last_message = result["messages"][-1]
    agent_result = AgentResult(agent=agent_name, summary=getattr(last_message, "content", str(last_message)))

    update: dict = {"messages": [AIMessage(content=f"[{agent_name}] {agent_result.summary}")]}
    if agent_name == "datasource":
        # Lets a follow-up question route straight to "analysis" (or skip
        # re-delegating altogether) instead of always re-fetching - see
        # SUPERVISOR_SYSTEM_PROMPT and clients/sandbox/client.py's cache.
        update["data_context"] = agent_result.summary
    return update


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


def build_clarify_node(llm: BaseChatModel):
    chain = build_clarify_chain(llm)

    def clarify_node(state: OrchestratorState):
        response = chain.invoke(state["messages"])
        return {"messages": [response]}

    return clarify_node
