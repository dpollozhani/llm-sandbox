"""Structured shapes for the analysis agent's tool results."""
from __future__ import annotations

from pydantic import BaseModel, Field


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


class AnalysisResult(BaseModel):
    """What `agents/orchestrator/nodes.py::_run_specialist` persists in
    `OrchestratorState.last_analysis_result` from the analysis agent's own
    most recent successful `python_sandbox_execute` call - the same role
    `DataSourceQueryResult` plays for the datasource agent's last fetch (see
    its own docstring): letting a *later*, freshly-seeded specialist
    delegation see what was already concluded, since specialist seeding
    (`_seed_content`) never forwards the raw orchestrator message history a
    conclusion like this would otherwise only ever live in."""

    summary: str
    """The analysis agent's own final natural-language answer this run.
    Unlike `DataSourceQueryResult.describe()`, this can't be generated
    deterministically from `preview` alone - arbitrary code can compute
    anything, so there's no fixed schema (like group_by/filters/measures)
    to render a description from. This is the specialist's own words, the
    same text that already became a `[analysis] ...`-prefixed message in
    `state["messages"]`, just persisted somewhere a later specialist's seed
    can read it too."""

    preview: list[dict] = Field(default_factory=list)
    """A bounded snapshot of the executed code's own `result` variable, if
    it was already row-shaped (a dict, or a list of dicts) - capped the
    same way `DataSourceQueryResult.preview` is, so a concrete value (an
    account code, a specific figure) survives as structured data a later
    specialist can match against exactly, not just prose it has to
    re-parse and hope is precise. Empty when `result` wasn't already this
    shape (a scalar, a plain string, or `None`) - `summary` is the only
    record of those cases."""

    def describe(self) -> str:
        """A short natural-language line for a later specialist's seed
        message - `summary` is already natural language, so this only adds
        the concrete preview values on top when there are any."""
        if not self.preview:
            return self.summary
        return f"{self.summary} (concrete value(s): {self.preview})"
