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


class DatasetConfig(BaseModel):
    dataset_id: str
    dataset_name: str


class WorkspaceConfig(BaseModel):
    workspace_id: str
    workspace_name: str
    datasets: list[DatasetConfig]


class SemanticModelConfig(BaseModel):
    model_name: str
    dataset_id: str
    tables: list[str]


class PowerBiCatalog(BaseModel):
    workspaces: list[WorkspaceConfig]
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
