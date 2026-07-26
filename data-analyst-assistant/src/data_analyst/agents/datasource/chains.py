"""LCEL runnable that binds the datasource tools to the chat model."""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool

from data_analyst.agents.datasource.prompts import SYSTEM_PROMPT
from data_analyst.config.settings import PowerBiCatalog


def build_agent_chain(llm: BaseChatModel, tools: list[BaseTool], catalog: PowerBiCatalog | None = None) -> Runnable:
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = SYSTEM_PROMPT
    if catalog is not None and catalog.semantic_models:
        # There's no "list models" tool (removed along with workspace
        # listing/refresh history - out of scope for this build), so this is
        # the only way the model learns which `model_name` values are valid
        # to pass to pbi_mcp_get_semantic_metadata/pbi_rest_run_dax_query.
        names = ", ".join(f'"{m.model_name}"' for m in catalog.semantic_models)
        system_prompt += f"\n\nAvailable semantic models: {names}."

    async def _invoke(messages: list[AnyMessage]):
        return await llm_with_tools.ainvoke([SystemMessage(content=system_prompt), *messages])

    return RunnableLambda(_invoke)
