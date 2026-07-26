"""The supervisor's routing chain (structured output) and the final-answer/
clarification chains."""
from __future__ import annotations

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from data_analyst.agents.common.models import Clarification
from data_analyst.agents.orchestrator.prompts import (
    CLARIFY_SYSTEM_PROMPT,
    RESPOND_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
)


class Route(BaseModel):
    """Every field needs a default: `FakeToolCallingChatModel`'s
    `with_structured_output` stand-in constructs this with no arguments (see
    clients/llm/factory.py), which tests rely on."""

    next: Literal["datasource", "analysis", "respond", "clarify"] = "respond"
    reason: str = ""


def build_supervisor_chain(llm: BaseChatModel) -> Runnable:
    router = llm.with_structured_output(Route)

    async def _invoke(args: dict) -> Route:
        prompt = SUPERVISOR_SYSTEM_PROMPT
        if data_context := args.get("data_context"):
            prompt += f"\n\nCurrently available data in this session: {data_context.describe()}"
        return await router.ainvoke([SystemMessage(content=prompt), *args["messages"]])

    return RunnableLambda(_invoke)


def build_respond_chain(llm: BaseChatModel) -> Runnable:
    async def _invoke(messages: list[AnyMessage]):
        return await llm.ainvoke([SystemMessage(content=RESPOND_SYSTEM_PROMPT), *messages])

    return RunnableLambda(_invoke)


def build_clarify_chain(llm: BaseChatModel) -> Runnable:
    chain = llm.with_structured_output(Clarification)

    async def _invoke(messages: list[AnyMessage]) -> Clarification:
        return await chain.ainvoke([SystemMessage(content=CLARIFY_SYSTEM_PROMPT), *messages])

    return RunnableLambda(_invoke)
