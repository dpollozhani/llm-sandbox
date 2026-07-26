"""Orchestrator nodes: a routing supervisor plus wrappers that seed a fresh
specialist subgraph from the current task and fold its answer back in.
"""
from __future__ import annotations

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from data_analyst.agents.analysis.graph import build_analysis_graph
from data_analyst.agents.common.models import AgentResult, Clarification, FetchedDataset
from data_analyst.agents.common.tools import request_clarification
from data_analyst.agents.datasource.graph import build_datasource_graph
from data_analyst.agents.orchestrator.chains import build_clarify_chain, build_respond_chain, build_supervisor_chain
from data_analyst.agents.orchestrator.state import OrchestratorState
from data_analyst.clients.powerbi.mcp import PBIMcpClient
from data_analyst.clients.powerbi.rest import PBIRestClient

MAX_TURNS = 6

SPECIALIST_RECURSION_LIMIT = 20
"""Caps a specialist's own internal agent<->tools loop (`_run_specialist`'s
`child_graph.ainvoke()`) - explicitly, rather than relying on LangGraph's
own default, which is version-dependent and has been observed to differ
by orders of magnitude between installs. Generous enough for several
legitimate tool-call/retry round trips, but still a bounded, intentional
cap rather than an implicit one - see the `GraphRecursionError` handling
below for what happens when it's hit."""


def build_supervisor_node(llm: BaseChatModel):
    chain = build_supervisor_chain(llm)

    async def supervisor_node(state: OrchestratorState):
        turns = state.get("turns", 0)
        if turns >= MAX_TURNS:
            return {"next": "respond", "turns": turns, "awaiting_clarification": False}
        route = await chain.ainvoke({"messages": state["messages"], "data_context": state.get("data_context")})
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


def _specialist_clarification(messages: list[AnyMessage]) -> Clarification | None:
    """The `Clarification` a specialist requested during this run (see
    agents/common/tools.py::request_clarification), if any - read directly
    from the tool call's own structured result rather than the model's own
    (freeform, easy-to-drift) restatement of it, so the question/options a
    frontend renders always exactly match what the tool was actually called
    with."""
    for message in messages:
        if message.type == "tool" and message.name == request_clarification.name:
            return Clarification(**json.loads(message.content))
    return None


def _fetched_dataset(messages: list[AnyMessage]) -> FetchedDataset | None:
    """The `FetchedDataset` behind the most recent successful
    `pbi_rest_run_dax_query` call during this run, if any - read from the
    tool's own structured result (`dataset_id`/`model_name`/`query`/
    `row_count`), not a specialist's own freeform summary of it, which isn't
    guaranteed to mention all of it. Searched newest-first so a failed
    attempt followed by a successful retry resolves to the retry."""
    for message in reversed(messages):
        if message.type != "tool" or message.name != "pbi_rest_run_dax_query":
            continue
        payload = json.loads(message.content)
        if "dataset_id" not in payload:
            continue  # an {"error": ...} result, not a successful fetch
        return FetchedDataset(
            dataset_id=payload["dataset_id"],
            model_name=payload["model_name"],
            row_count=payload["row_count"],
            **payload["query"],
        )
    return None


async def _run_specialist(agent_name: str, build_graph_fn, llm: BaseChatModel, state: OrchestratorState) -> dict:
    data_context = state.get("data_context")

    if state.get("awaiting_clarification"):
        # The latest human message is a reply to a clarifying question
        # (this specialist's own, or the supervisor's upfront one), not a
        # fresh request - a rebuilt-from-scratch specialist subgraph has no
        # memory of that question (or the original ask before it) unless we
        # hand it the whole exchange, so it can actually use the answer
        # instead of re-deriving (or re-asking) everything from one isolated
        # reply.
        messages = list(state["messages"])
    else:
        task_message = _latest_user_task(state["messages"])
        seed_content = (
            f"(Available data in this session: {FetchedDataset(**data_context).describe()})\n\n{task_message.content}"
            if data_context
            else task_message.content
        )
        messages = [HumanMessage(content=seed_content)]

    child_graph = build_graph_fn(llm)
    try:
        result = await child_graph.ainvoke(
            {
                "messages": messages,
                "session_id": state["session_id"],
                "pbi_token": state.get("pbi_token"),
            },
            config={"recursion_limit": SPECIALIST_RECURSION_LIMIT},
        )
    except GraphRecursionError:
        # The specialist's own agent<->tools loop kept calling tools without
        # ever reaching a final answer (e.g. retrying a computation that
        # keeps failing) - LangGraph's hard step cap turned that into a raw
        # exception that would otherwise crash this whole turn. Fold back a
        # plain failure instead: the supervisor's own MAX_TURNS still bounds
        # how many times this can happen before "respond" takes over.
        return {
            "messages": [
                AIMessage(
                    content=f"[{agent_name}] Couldn't complete this after many attempts - "
                    "try a simpler or narrower request."
                )
            ],
            "awaiting_clarification": False,
        }

    clarification = _specialist_clarification(result["messages"])
    if clarification is not None:
        # The specialist itself decided it couldn't proceed confidently and
        # asked its own clarifying question - surface it to the user
        # directly. Setting `next` here (rather than going back to
        # "supervisor") is what lets agents/orchestrator/graph.py route
        # straight to END: no extra supervisor round-trip, no separate
        # "clarify" node call, just the specialist's question (and options)
        # as the reply.
        return {
            "messages": [AIMessage(content=clarification.question)],
            "next": "clarify",
            "clarification_options": clarification.options,
            "awaiting_clarification": True,
        }

    last_message = result["messages"][-1]
    summary = getattr(last_message, "content", str(last_message))
    agent_result = AgentResult(agent=agent_name, summary=summary)
    update: dict = {
        "messages": [AIMessage(content=f"[{agent_name}] {agent_result.summary}")],
        "awaiting_clarification": False,
    }
    if agent_name == "datasource":
        fetched = _fetched_dataset(result["messages"])
        if fetched is not None:
            # Lets a follow-up question route straight to "analysis" (or skip
            # re-delegating altogether) instead of always re-fetching - see
            # SUPERVISOR_SYSTEM_PROMPT and clients/sandbox/client.py's cache.
            # Left out of `update` (not overwritten with None) when this run
            # didn't fetch anything (e.g. only browsed the schema), so a
            # dataset from an earlier turn stays available. Stored as a
            # plain dict, not the FetchedDataset itself - see
            # OrchestratorState.data_context's docstring for why.
            update["data_context"] = fetched.model_dump()
    return update


def build_datasource_node(
    llm: BaseChatModel, mcp_client: PBIMcpClient | None = None, rest_client: PBIRestClient | None = None
):
    def build_graph_fn(agent_llm: BaseChatModel):
        return build_datasource_graph(agent_llm, mcp_client=mcp_client, rest_client=rest_client)

    async def datasource_node(state: OrchestratorState):
        return await _run_specialist("datasource", build_graph_fn, llm, state)

    return datasource_node


def build_analysis_node(llm: BaseChatModel):
    async def analysis_node(state: OrchestratorState):
        return await _run_specialist("analysis", build_analysis_graph, llm, state)

    return analysis_node


def build_respond_node(llm: BaseChatModel):
    chain = build_respond_chain(llm)

    async def respond_node(state: OrchestratorState):
        response = await chain.ainvoke(state["messages"])
        return {"messages": [response], "awaiting_clarification": False}

    return respond_node


def build_clarify_node(llm: BaseChatModel):
    chain = build_clarify_chain(llm)

    async def clarify_node(state: OrchestratorState):
        clarification = await chain.ainvoke(state["messages"])
        return {
            "messages": [AIMessage(content=clarification.question)],
            "clarification_options": clarification.options,
            "awaiting_clarification": True,
        }

    return clarify_node
