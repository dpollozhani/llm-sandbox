"""Structured shapes for the analysis agent's tool results."""
from __future__ import annotations

from pydantic import BaseModel


class SandboxExecutionResult(BaseModel):
    stdout: str = ""
    result: object = None
    error: str | None = None
