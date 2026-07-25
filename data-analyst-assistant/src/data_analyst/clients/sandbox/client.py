"""Client for the (mocked) sandbox execution service.

Stages DataFrames under a reference id so tool results can be handed between
agents by reference instead of round-tripping full result sets through the
LLM's context window, then delegates actual execution to `executor.py`. In
production this class would be an HTTP client to an isolated code-execution
service instead of an in-process dict + `exec()`.

Each session (one per conversation `thread_id`) gets its own `SandboxClient`
via `get_sandbox_client`, so staged data and the query-result cache are
session-bound: a follow-up question in the same conversation can reuse
already-fetched data, but sessions never see each other's data.
"""
from __future__ import annotations

import itertools

import pandas as pd

from data_analyst.clients.sandbox.executor import ExecutionResult, execute
from data_analyst.telemetry.tracing import trace_span


class SandboxClient:
    def __init__(self) -> None:
        self._store: dict[str, pd.DataFrame] = {}
        self._query_cache: dict[str, str] = {}  # query cache_key -> sandbox_ref
        self._ref_counter = itertools.count(1)

    def stage(self, df: pd.DataFrame) -> str:
        ref = f"df_{next(self._ref_counter)}"
        self._store[ref] = df
        return ref

    def peek(self, sandbox_ref: str) -> pd.DataFrame | None:
        return self._store.get(sandbox_ref)

    def find_cached(self, cache_key: str) -> str | None:
        """Return the sandbox_ref for a previously staged result with this
        cache key, if this session has already fetched it."""
        return self._query_cache.get(cache_key)

    def remember(self, cache_key: str, sandbox_ref: str) -> None:
        self._query_cache[cache_key] = sandbox_ref

    def execute(self, code: str, sandbox_ref: str | None = None) -> ExecutionResult:
        with trace_span("sandbox.execute", sandbox_ref=sandbox_ref):
            dataframe = None
            if sandbox_ref is not None:
                if sandbox_ref not in self._store:
                    return ExecutionResult(error=f"Unknown sandbox_ref '{sandbox_ref}'")
                dataframe = self._store[sandbox_ref]
            return execute(code, dataframe)


_sessions: dict[str, SandboxClient] = {}


def get_sandbox_client(session_id: str) -> SandboxClient:
    """Process-wide registry of one SandboxClient per session. Mirrors how
    the orchestrator's checkpointer keeps per-thread_id conversation state -
    both are process-local and lost on restart; swap in a shared backing
    store (e.g. Redis) for a real multi-instance deployment."""
    if session_id not in _sessions:
        _sessions[session_id] = SandboxClient()
    return _sessions[session_id]
