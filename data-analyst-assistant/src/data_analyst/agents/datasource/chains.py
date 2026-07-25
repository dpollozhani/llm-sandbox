"""LCEL runnable that binds the datasource tools to the chat model."""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool

from data_analyst.agents.datasource.prompts import SYSTEM_PROMPT


def build_agent_chain(llm: BaseChatModel, tools: list[BaseTool]) -> Runnable:
    llm_with_tools = llm.bind_tools(tools)

    def _invoke(messages: list[AnyMessage]):
        return llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT), *messages])

    return RunnableLambda(_invoke)
