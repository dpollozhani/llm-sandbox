"""Builds the chat model used across all agents, chosen by `settings.llm_provider`."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda

from data_analyst.clients.llm.azure_openai import build_azure_chat_openai
from data_analyst.config.settings import Settings


class FakeToolCallingChatModel(FakeMessagesListChatModel):
    """A scripted chat model for tests: `responses` is the fixed sequence of
    messages to hand back, one per call. `FakeMessagesListChatModel` doesn't
    implement tool binding or structured output, since it's meant for plain
    chat scripting; both overrides here are intentionally generic (no
    knowledge of a specific tool/schema) so this class can stand in for any
    agent's model in a test."""

    def bind_tools(self, tools, **kwargs) -> "FakeToolCallingChatModel":
        return self

    def with_structured_output(self, schema, **kwargs) -> Runnable:
        # Requires `schema` to be constructible with no arguments, i.e. every
        # field has a default - true for this project's routing schemas.
        return RunnableLambda(lambda *_args, **_kwargs: schema())

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # FakeMessagesListChatModel returns the *same* message object (by
        # reference) every time a response is reused - e.g. a test that
        # sends a second message on the same thread and expects the same
        # scripted reply again. LangGraph's `add_messages` reducer assigns
        # that object an `id` and mutates it in place the first time it's
        # added to a thread's history; handing back the identical object
        # (with that same id already set) on a later turn makes
        # `add_messages` treat it as an *update* to the earlier message
        # instead of a new one, so nothing gets appended - the human message
        # ends up last, and callers that read `messages[-1]` see their own
        # message echoed back. Returning a fresh copy with no id every call
        # avoids that.
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        fresh = [g.message.model_copy(update={"id": None}) for g in result.generations]
        return ChatResult(generations=[ChatGeneration(message=m) for m in fresh])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # FakeMessagesListChatModel has no real `_stream`, so a test using
        # this model against app/api.py's /chat/stream would otherwise only
        # ever see a single all-at-once chunk, unlike a real provider's
        # token-by-token output. A tool call is scripted as one complete
        # decision, not something to simulate token-by-token, so it's still
        # emitted as a single chunk.
        message = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs).generations[0].message
        if message.tool_calls:
            yield ChatGenerationChunk(message=AIMessageChunk(content=message.content, tool_calls=message.tool_calls))
            return
        words = message.content.split(" ")
        for i, word in enumerate(words):
            piece = word if i == len(words) - 1 else f"{word} "
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))


def get_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "azure_openai":
        return build_azure_chat_openai(settings)

    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is required for llm_provider=anthropic")
    return init_chat_model(f"anthropic:{settings.anthropic_model}", api_key=settings.anthropic_api_key)
