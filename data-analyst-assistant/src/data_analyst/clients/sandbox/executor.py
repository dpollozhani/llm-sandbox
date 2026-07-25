"""Restricted Python execution. A real sandbox would be a network-isolated
worker process/container; this runs `exec()` against a minimal namespace as a
stand-in, and should not be pointed at untrusted input as-is."""
from __future__ import annotations

import contextlib
import io

import pandas as pd
from pydantic import BaseModel

from ...utils.dataframe import to_records


class ExecutionResult(BaseModel):
    stdout: str = ""
    result: object = None
    error: str | None = None


def execute(code: str, dataframe: pd.DataFrame | None = None) -> ExecutionResult:
    local_vars: dict = {}
    if dataframe is not None:
        local_vars["df"] = dataframe.copy()

    safe_globals = {"__builtins__": {"len": len, "range": range, "sum": sum, "round": round}, "pd": pd}
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, safe_globals, local_vars)  # noqa: S102 - mocked sandbox
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller, not raised
        return ExecutionResult(stdout=stdout.getvalue(), error=str(exc))

    return ExecutionResult(stdout=stdout.getvalue(), result=to_records(local_vars.get("result")))
