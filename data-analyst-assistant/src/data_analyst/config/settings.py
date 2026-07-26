"""Application settings (env-driven) and the static Power BI catalog config."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_CATALOG_PATH = Path(__file__).parent / "semantic_models.yaml"
_DEFAULT_GLOSSARY_PATH = Path(__file__).parent / "glossary.yaml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    app_name: str = "data-analyst-assistant"
    log_level: str = "INFO"

    # No default: pick one explicitly. Fails fast (a pydantic ValidationError
    # at Settings() construction, i.e. app startup) if LLM_PROVIDER isn't set,
    # rather than silently falling back to anything.
    llm_provider: Literal["azure_openai", "anthropic"]

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-5"

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_deployment: str = "gpt-4o"

    semantic_models_path: Path = _DEFAULT_CATALOG_PATH
    glossary_path: Path = _DEFAULT_GLOSSARY_PATH

    # Entra ID app registration used for delegated (per-user) sign-in. This
    # is required: Power BI's ExecuteQueries and the remote PBI MCP server's
    # GetSemanticMetadata both enforce row-level security using the calling
    # user's own identity - a service-principal (app-only) token is rejected
    # outright on datasets with RLS - so every request needs a signed-in
    # user's delegated token, not a client-credentials token. A public
    # client (PKCE, no client secret) - see clients/powerbi/auth.py's module
    # docstring for the Entra app registration requirements that go with
    # that (redirect URI platform type, "Allow public client flows").
    entra_tenant_id: str
    entra_client_id: str
    entra_redirect_uri: str

    pbi_mcp_server_url: str = "https://api.fabric.microsoft.com/v1/mcp/powerbi"


class SemanticModelConfig(BaseModel):
    model_name: str
    dataset_id: str


class PowerBiCatalog(BaseModel):
    semantic_models: list[SemanticModelConfig]

    def find_model(self, model_name: str) -> SemanticModelConfig | None:
        return next((m for m in self.semantic_models if m.model_name == model_name), None)


class GlossaryEntry(BaseModel):
    term: str
    definition: str


class Glossary(BaseModel):
    """Domain terms/concepts the model can't reliably infer from the
    semantic model's schema alone (an abbreviation, a business meaning
    behind a cryptically-named column, a term that collides with a more
    common meaning) - injected into every agent's system prompt (see
    `inject_glossary`) so the user doesn't have to re-explain them every
    conversation, and so a term that would confuse the datasource agent
    doesn't just as easily confuse the supervisor's routing or the
    analysis agent first."""

    terms: list[GlossaryEntry] = Field(default_factory=list)

    def render(self) -> str:
        """A compact `term: definition` line per entry, for injecting into
        a prompt - not a big structured block, since a glossary is meant to
        be a handful of terms, not a dataset in its own right."""
        return "\n".join(f"- {e.term}: {e.definition}" for e in self.terms)


def inject_glossary(prompt: str, glossary: Glossary | None) -> str:
    """Append `glossary`'s rendered terms to `prompt`, if there are any.
    Shared by every agent chain that builds a system prompt (datasource,
    analysis, and the orchestrator's supervisor/respond/clarify chains) so
    this stays in one place rather than re-implemented per chain."""
    if glossary is not None and glossary.terms:
        return prompt + f"\n\nGlossary:\n{glossary.render()}"
    return prompt


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_catalog(path: Path | None = None) -> PowerBiCatalog:
    catalog_path = path or get_settings().semantic_models_path
    with open(catalog_path) as f:
        raw = yaml.safe_load(f)
    return PowerBiCatalog.model_validate(raw)


@lru_cache
def get_glossary(path: Path | None = None) -> Glossary:
    """Unlike `get_catalog` (required - no semantic models means no data
    queries at all), a glossary is optional supplementary context: a
    missing or empty file just means no terms to inject, not a startup
    failure."""
    glossary_path = path or get_settings().glossary_path
    if not glossary_path.exists():
        return Glossary()
    with open(glossary_path) as f:
        raw = yaml.safe_load(f) or {}
    return Glossary.model_validate(raw)
