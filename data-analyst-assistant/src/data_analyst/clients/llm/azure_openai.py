"""Builds the Azure OpenAI chat model used in production."""
from __future__ import annotations

from langchain_openai import AzureChatOpenAI

from ...config.settings import Settings


def build_azure_chat_openai(settings: Settings) -> AzureChatOpenAI:
    if not settings.azure_openai_endpoint or not settings.azure_openai_api_key:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are required for llm_provider=azure_openai"
        )
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_deployment,
    )
