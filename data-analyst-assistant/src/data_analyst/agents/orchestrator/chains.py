"""The supervisor's routing chain (structured output) and the final-answer chain."""
from __future__ import annotations

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from data_analyst.agents.orchestrator.prompts import RESPOND_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPT


class Route(BaseModel):
    """Every field needs a default: the demo chat model's `with_structured_output`
    stand-in constructs this with no arguments (see clients/llm/factory.py)."""

    next: Literal["datasource", "analysis", "respond"] = "respond"
    reason: str = ""


def build_supervisor_chain(llm: BaseChatModel) -> Runnable:
    router = llm.with_structured_output(Route)

    def _invoke(messages: list[AnyMessage]) -> Route:
        return router.invoke([SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT), *messages])

    return RunnableLambda(_invoke)


def build_respond_chain(llm: BaseChatModel) -> Runnable:
    def _invoke(messages: list[AnyMessage]):
        return llm.invoke([SystemMessage(content=RESPOND_SYSTEM_PROMPT), *messages])

    return RunnableLambda(_invoke)
