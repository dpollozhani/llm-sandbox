"""Analysis agent: the Python sandbox tool, plus the graph nodes that use it."""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from ...clients.sandbox.client import sandbox_client
from .chains import build_agent_chain
from .state import AnalysisState


@tool
def python_sandbox_execute(code: str, sandbox_ref: str | None = None) -> dict:
    """Execute Python/pandas code in an isolated sandbox.

    If `sandbox_ref` is given, the staged DataFrame is bound to the local
    variable `df` before running `code`. Assign to a variable named `result`
    to return a value; anything printed is captured as stdout.
    """
    result = sandbox_client.execute(code, sandbox_ref)
    return result.model_dump()


TOOLS = [python_sandbox_execute]


def build_agent_node(llm: BaseChatModel):
    chain = build_agent_chain(llm, TOOLS)

    def agent_node(state: AnalysisState):
        response = chain.invoke(state["messages"])
        return {"messages": [response]}

    return agent_node


def build_tool_node() -> ToolNode:
    return ToolNode(TOOLS)
