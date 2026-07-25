"""The supervisor's routing chain (structured output) and the final-answer/
clarification chains."""
from __future__ import annotations

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from data_analyst.agents.orchestrator.prompts import (
    CLARIFY_SYSTEM_PROMPT,
    RESPOND_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
)


class Route(BaseModel):
    """Every field needs a default: the demo chat model's `with_structured_output`
    stand-in constructs this with no arguments (see clients/llm/factory.py)."""

    next: Literal["datasource", "analysis", "respond", "clarify"] = "respond"
    reason: str = ""


def build_supervisor_chain(llm: BaseChatModel) -> Runnable:
    router = llm.with_structured_output(Route)

    def _invoke(messages: list[AnyMessage], data_context: str | None) -> Route:
        prompt = SUPERVISOR_SYSTEM_PROMPT
        if data_context:
            prompt += f"\n\nCurrently available data in this session: {data_context}"
        return router.invoke([SystemMessage(content=prompt), *messages])

    return RunnableLambda(lambda args: _invoke(args["messages"], args.get("data_context")))


def build_respond_chain(llm: BaseChatModel) -> Runnable:
    def _invoke(messages: list[AnyMessage]):
        return llm.invoke([SystemMessage(content=RESPOND_SYSTEM_PROMPT), *messages])

    return RunnableLambda(_invoke)


def build_clarify_chain(llm: BaseChatModel) -> Runnable:
    def _invoke(messages: list[AnyMessage]):
        return llm.invoke([SystemMessage(content=CLARIFY_SYSTEM_PROMPT), *messages])

    return RunnableLambda(_invoke)
