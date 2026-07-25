"""Mocked tool surface: PBI MCP, PBI REST, and a Python sandbox.

Each group of tools stands in for a real integration:
  - PBI MCP    -> a Model Context Protocol server exposing the semantic model
                  (list models, run DAX queries)
  - PBI REST   -> direct calls to the Power BI REST API (workspace/dataset
                  metadata, refresh operations)
  - Sandbox    -> an isolated code-execution service for pandas analysis

They return canned/deterministic data so the LangGraph wiring can be
exercised without any external dependency.
"""
from __future__ import annotations

import contextlib
import io
import itertools

import pandas as pd
from langchain_core.tools import tool
from langgraph.types import interrupt

from . import mock_data

# Shared "sandbox" dataframe store. Mimics how a real agent would stage query
# results (e.g. into a scratch table or object store) so the python sandbox
# tool can reference them by id instead of round-tripping full result sets
# through the LLM's context window.
_SANDBOX_STORE: dict[str, pd.DataFrame] = {}
_ref_counter = itertools.count(1)


def _stage(df: pd.DataFrame) -> str:
    ref = f"df_{next(_ref_counter)}"
    _SANDBOX_STORE[ref] = df
    return ref


# --- 1. PBI MCP (mocked) ----------------------------------------------------

@tool
def pbi_mcp_list_semantic_models() -> list[dict]:
    """List semantic models (datasets) reachable through the Power BI MCP server."""
    return mock_data.SEMANTIC_MODELS


@tool
def pbi_mcp_run_dax_query(model_name: str, dax_query: str) -> dict:
    """Run a DAX query against a Power BI semantic model via the MCP server.

    Args:
        model_name: Name of the semantic model, e.g. "Sales Analytics".
        dax_query: The DAX query text, e.g. "EVALUATE Sales".

    Returns a small preview of the resulting rows plus a `sandbox_ref` that
    can be passed to `python_sandbox_execute` to load the full result as a
    pandas DataFrame (bound to `df`).
    """
    table_name = mock_data.guess_table_from_dax(dax_query)
    df = mock_data.get_table(table_name)
    ref = _stage(df)
    return {
        "sandbox_ref": ref,
        "row_count": len(df),
        "preview": df.head(5).to_dict(orient="records"),
    }


# --- 2. PBI REST API (mocked) -----------------------------------------------

@tool
def pbi_rest_list_workspaces() -> list[dict]:
    """List Power BI workspaces and their datasets via the PBI REST API."""
    return mock_data.WORKSPACES


@tool
def pbi_rest_get_refresh_history(dataset_id: str) -> list[dict]:
    """Get the refresh history for a dataset via the PBI REST API."""
    return mock_data.get_refresh_history(dataset_id)


@tool
def pbi_rest_trigger_dataset_refresh(dataset_id: str) -> dict:
    """Trigger an on-demand refresh of a Power BI dataset via the PBI REST API.

    This mutates a shared resource, so it pauses the graph and asks a human
    to approve before actually "calling" the REST API.
    """
    approved = interrupt(
        {
            "type": "approval_required",
            "action": "pbi_rest_trigger_dataset_refresh",
            "dataset_id": dataset_id,
            "message": f"Approve triggering a refresh for dataset '{dataset_id}'?",
        }
    )
    if not approved:
        return {"status": "cancelled", "dataset_id": dataset_id}
    return mock_data.trigger_refresh(dataset_id)


# --- 3. Python sandbox (mocked) ----------------------------------------------

@tool
def python_sandbox_execute(code: str, sandbox_ref: str | None = None) -> dict:
    """Execute Python/pandas code in an isolated sandbox.

    If `sandbox_ref` is given (from a prior PBI MCP query), the staged
    DataFrame is bound to the local variable `df` before running `code`.
    Assign to a variable named `result` to return a value; anything printed
    is captured as stdout.
    """
    local_vars: dict = {}
    if sandbox_ref is not None:
        if sandbox_ref not in _SANDBOX_STORE:
            return {"error": f"Unknown sandbox_ref '{sandbox_ref}'"}
        local_vars["df"] = _SANDBOX_STORE[sandbox_ref].copy()

    # A real sandbox would run this in a network-isolated worker; here we
    # just restrict the available names for a bit of blast-radius control.
    safe_globals = {"__builtins__": {"len": len, "range": range, "sum": sum, "round": round}, "pd": pd}
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, safe_globals, local_vars)  # noqa: S102 - mocked sandbox
    except Exception as exc:  # noqa: BLE001 - surfaced to the agent, not raised
        return {"stdout": stdout.getvalue(), "error": str(exc)}

    result = local_vars.get("result")
    if isinstance(result, pd.DataFrame):
        result = result.to_dict(orient="records")

    return {"stdout": stdout.getvalue(), "result": result}


PBI_MCP_TOOLS = [pbi_mcp_list_semantic_models, pbi_mcp_run_dax_query]
PBI_REST_TOOLS = [
    pbi_rest_list_workspaces,
    pbi_rest_get_refresh_history,
    pbi_rest_trigger_dataset_refresh,
]
SANDBOX_TOOLS = [python_sandbox_execute]

ALL_TOOLS = [*PBI_MCP_TOOLS, *PBI_REST_TOOLS, *SANDBOX_TOOLS]
