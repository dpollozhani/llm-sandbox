"""Pydantic models shared across agent packages."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """What a specialist subgraph (datasource/analysis) hands back to the
    orchestrator: enough to keep the conversation going without re-exposing
    every intermediate tool call to the supervisor's own context."""

    agent: str
    summary: str


class Clarification(BaseModel):
    """A clarifying question with 2-3 clearly distinct options a user can
    pick from, so a frontend can render them as buttons instead of relying
    on free text. Produced two ways, both read into this same shape by
    `agents/orchestrator/nodes.py`: the supervisor's own upfront "clarify"
    decision (`agents/orchestrator/chains.py::build_clarify_chain`, via
    structured output) and a specialist's `request_clarification` tool call
    (`agents/common/tools.py`)."""

    question: str = ""
    options: list[str] = Field(default_factory=lambda: ["", ""], min_length=2, max_length=3)
