"""Restricted Python execution. A real sandbox would be a network-isolated
worker process/container; this runs `exec()` against a minimal namespace as a
stand-in, and should not be pointed at untrusted input as-is."""
from __future__ import annotations

import contextlib
import io
import math

import numpy as np
import pandas as pd
from scipy import stats

from data_analyst.agents.analysis.models import ExecutionResult
from data_analyst.utils.dataframe import to_records

# Ordinary, pure-computation builtins a pandas/numpy/scipy analysis script
# would reasonably reach for - deliberately excludes anything that touches
# the filesystem/network/process, or that could reintroduce arbitrary code
# execution (open, __import__, exec, eval, compile, input, globals/locals/
# vars, getattr/setattr/delattr). That said, this was never a real security
# boundary either way (see this module's own docstring) - the point of this
# list is giving a data-analysis script what it needs, not gatekeeping.
_SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "range": range,
    "sum": sum,
    "round": round,
    "min": min,
    "max": max,
    "abs": abs,
    "sorted": sorted,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "reversed": reversed,
    "all": all,
    "any": any,
    "isinstance": isinstance,
    "next": next,
    "iter": iter,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "ZeroDivisionError": ZeroDivisionError,
    "AttributeError": AttributeError,
    "StopIteration": StopIteration,
}


def execute(code: str, dataframe: pd.DataFrame | None = None) -> ExecutionResult:
    local_vars: dict = {}
    if dataframe is not None:
        local_vars["df"] = dataframe.copy()

    safe_globals = {"__builtins__": _SAFE_BUILTINS, "pd": pd, "np": np, "math": math, "stats": stats}
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, safe_globals, local_vars)  # noqa: S102 - mocked sandbox
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller, not raised
        return ExecutionResult(stdout=stdout.getvalue(), error=str(exc))

    return ExecutionResult(stdout=stdout.getvalue(), result=to_records(local_vars.get("result")))
