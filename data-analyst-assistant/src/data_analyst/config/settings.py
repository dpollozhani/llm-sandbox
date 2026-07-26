"""Application settings (env-driven) and the static Power BI catalog config."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_CATALOG_PATH = Path(__file__).parent / "semantic_models.yaml"


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
    tables: list[str]


class PowerBiCatalog(BaseModel):
    semantic_models: list[SemanticModelConfig]

    def find_model(self, model_name: str) -> SemanticModelConfig | None:
        return next((m for m in self.semantic_models if m.model_name == model_name), None)


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_catalog(path: Path | None = None) -> PowerBiCatalog:
    catalog_path = path or get_settings().semantic_models_path
    with open(catalog_path) as f:
        raw = yaml.safe_load(f)
    return PowerBiCatalog.model_validate(raw)
