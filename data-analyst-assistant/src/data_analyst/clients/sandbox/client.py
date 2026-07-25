"""Client for the (mocked) sandbox execution service.

Stages DataFrames under a reference id so tool results can be handed between
agents by reference instead of round-tripping full result sets through the
LLM's context window, then delegates actual execution to `executor.py`. In
production this class would be an HTTP client to an isolated
code-execution service instead of an in-process dict + `exec()`.
"""
from __future__ import annotations

import itertools

import pandas as pd

from data_analyst.clients.sandbox.executor import ExecutionResult, execute
from data_analyst.telemetry.tracing import trace_span

_ref_counter = itertools.count(1)


class SandboxClient:
    def __init__(self) -> None:
        self._store: dict[str, pd.DataFrame] = {}

    def stage(self, df: pd.DataFrame) -> str:
        ref = f"df_{next(_ref_counter)}"
        self._store[ref] = df
        return ref

    def execute(self, code: str, sandbox_ref: str | None = None) -> ExecutionResult:
        with trace_span("sandbox.execute", sandbox_ref=sandbox_ref):
            dataframe = None
            if sandbox_ref is not None:
                if sandbox_ref not in self._store:
                    return ExecutionResult(error=f"Unknown sandbox_ref '{sandbox_ref}'")
                dataframe = self._store[sandbox_ref]
            return execute(code, dataframe)


# Process-wide instance: mirrors talking to one shared sandbox service.
# A per-session sandbox would be created per thread_id in a real deployment.
sandbox_client = SandboxClient()
