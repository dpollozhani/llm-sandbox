"""Structured shapes for the analysis agent's tool results."""
from __future__ import annotations

from pydantic import BaseModel


class ExecutionResult(BaseModel):
    """What `clients/sandbox/executor.py::execute` (and, in turn,
    `clients/sandbox/client.py::SandboxClient.execute`) hands back for one
    `python_sandbox_execute` tool call - captured stdout, the executed code's
    `result` variable (already record-ified via `utils/dataframe.py::to_records`),
    or an `error` string if execution raised. Lives here rather than in
    `clients/sandbox/` because it's an agent-facing tool-result shape, not a
    client-internal one - `executor.py` imports it back."""

    stdout: str = ""
    result: object = None
    error: str | None = None
