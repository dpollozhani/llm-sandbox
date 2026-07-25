"""Structured shapes for the datasource agent's tool results."""
from __future__ import annotations

from pydantic import BaseModel


class SemanticModelInfo(BaseModel):
    model_name: str
    dataset_id: str
    tables: list[str]


class DaxQueryResult(BaseModel):
    sandbox_ref: str
    row_count: int
    preview: list[dict]
    dax_query: str
    reused: bool
    """True if this result was served from the session's data store instead
    of issuing a new query - see `clients/sandbox/client.py`."""


class WorkspaceInfo(BaseModel):
    workspace_id: str
    workspace_name: str
    datasets: list[dict]


class RefreshHistoryEntry(BaseModel):
    request_id: str
    status: str
    start_time: str
    end_time: str
