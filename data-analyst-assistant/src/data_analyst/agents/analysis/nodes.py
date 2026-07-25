"""Analysis agent: the Python sandbox tool, plus the graph nodes that use it."""
from __future__ import annotations

from typing import Annotated

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, ToolNode

from data_analyst.agents.analysis.chains import build_agent_chain
from data_analyst.agents.analysis.state import AnalysisState
from data_analyst.agents.common.tools import request_clarification
from data_analyst.clients.sandbox.client import get_sandbox_client


@tool
async def python_sandbox_execute(
    code: str, state: Annotated[AnalysisState, InjectedState], sandbox_ref: str | None = None
) -> dict:
    """Execute Python/pandas code in an isolated sandbox.

    If `sandbox_ref` is given, the staged DataFrame is bound to the local
    variable `df` before running `code`. Assign to a variable named `result`
    to return a value; anything printed is captured as stdout.
    """
    store = get_sandbox_client(state["session_id"])
    result = await store.execute(code, sandbox_ref)
    return result.model_dump()


TOOLS = [python_sandbox_execute, request_clarification]


def build_agent_node(llm: BaseChatModel):
    chain = build_agent_chain(llm, TOOLS)

    async def agent_node(state: AnalysisState):
        response = await chain.ainvoke(state["messages"])
        return {"messages": [response]}

    return agent_node


def build_tool_node() -> ToolNode:
    return ToolNode(TOOLS)
