"""Pydantic models shared across agent packages."""
from __future__ import annotations

from pydantic import BaseModel


class AgentResult(BaseModel):
    """What a specialist subgraph (datasource/analysis) hands back to the
    orchestrator: enough to keep the conversation going without re-exposing
    every intermediate tool call to the supervisor's own context."""

    agent: str
    summary: str
