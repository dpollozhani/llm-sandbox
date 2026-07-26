"""LCEL runnable that binds the sandbox tool to the chat model."""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool

from data_analyst.agents.analysis.prompts import SYSTEM_PROMPT
from data_analyst.config.settings import Glossary, inject_glossary


def build_agent_chain(llm: BaseChatModel, tools: list[BaseTool], glossary: Glossary | None = None) -> Runnable:
    llm_with_tools = llm.bind_tools(tools)
    system_prompt = inject_glossary(SYSTEM_PROMPT, glossary)

    async def _invoke(messages: list[AnyMessage]):
        return await llm_with_tools.ainvoke([SystemMessage(content=system_prompt), *messages])

    return RunnableLambda(_invoke)
