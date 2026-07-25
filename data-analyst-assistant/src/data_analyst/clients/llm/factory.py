"""Builds the chat model used across all agents, chosen by `settings.llm_provider`.

"demo" needs no API key or network access: it's a scripted fake model that
answers immediately without delegating to a specialist, so `POST /chat`
works out of the box. To exercise the full tool-calling / delegation /
approval flow, either point this at a real provider or see the fake models
built directly inside tests/integration and tests/e2e, which script a
specific multi-step scenario for the subgraph or endpoint under test.
"""
from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from ...config.settings import Settings
from .azure_openai import build_azure_chat_openai


class FakeToolCallingChatModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel doesn't implement tool binding or structured
    output, since it's meant for plain chat scripting. Both overrides here
    are intentionally generic (no knowledge of a specific tool/schema) so
    this class can stand in for any agent's model in tests or demo mode."""

    def bind_tools(self, tools, **kwargs) -> "FakeToolCallingChatModel":
        return self

    def with_structured_output(self, schema, **kwargs) -> Runnable:
        # Requires `schema` to be constructible with no arguments, i.e. every
        # field has a default - true for this project's routing schemas.
        return RunnableLambda(lambda *_args, **_kwargs: schema())


def build_demo_chat_model() -> FakeToolCallingChatModel:
    return FakeToolCallingChatModel(
        responses=[
            AIMessage(
                content=(
                    "Running in demo mode (LLM_PROVIDER=demo): I can't reach a real "
                    "model, so this is a canned answer. Set LLM_PROVIDER to "
                    "'anthropic' or 'azure_openai' to see the full multi-agent flow, "
                    "or check tests/ for scripted scenarios."
                )
            )
        ]
    )


def get_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "azure_openai":
        return build_azure_chat_openai(settings)
    if settings.llm_provider == "anthropic":
        return init_chat_model(f"anthropic:{settings.anthropic_model}", api_key=settings.anthropic_api_key)
    return build_demo_chat_model()
