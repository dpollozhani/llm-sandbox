"""Analysis agent: the Python sandbox tool, plus the graph nodes that use it."""
from __future__ import annotations

from typing import Annotated

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, ToolNode

from data_analyst.agents.analysis.prompts import SYSTEM_PROMPT
from data_analyst.agents.analysis.state import AnalysisState
from data_analyst.agents.common.tools import flag_ambiguity
from data_analyst.clients.sandbox.client import get_sandbox_client
from data_analyst.config.settings import Glossary, inject_glossary


@tool
async def python_sandbox_execute(
    code: str, state: Annotated[AnalysisState, InjectedState], dataset_id: str | None = None
) -> dict:
    """Execute Python code in a restricted sandbox - pandas/numpy/scipy.stats/
    math are already imported as pd/np/stats/math, no other imports work.

    If `dataset_id` is given, the staged DataFrame is bound to the local
    variable `df` before running `code`. Assign to a variable named `result`
    to return a value; anything printed is captured as stdout.
    """
    store = get_sandbox_client(state["session_id"])
    result = await store.execute(code, dataset_id)
    return result.model_dump()


TOOLS = [python_sandbox_execute, flag_ambiguity]


def build_agent_node(llm: BaseChatModel, glossary: Glossary | None = None):
    llm_with_tools = llm.bind_tools(TOOLS)
    system_prompt = inject_glossary(SYSTEM_PROMPT, glossary)

    async def agent_node(state: AnalysisState):
        response = await llm_with_tools.ainvoke([SystemMessage(content=system_prompt), *state["messages"]])
        return {"messages": [response]}

    return agent_node


def build_tool_node() -> ToolNode:
    return ToolNode(TOOLS)
