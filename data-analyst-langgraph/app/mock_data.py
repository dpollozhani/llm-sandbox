"""Fake Power BI workspaces, semantic models, and underlying tables.

Stands in for what a real PBI MCP server / PBI REST API would return, so the
tool layer has something deterministic to work with.
"""
from __future__ import annotations

import pandas as pd

WORKSPACES = [
    {
        "workspace_id": "ws-001",
        "workspace_name": "Retail Analytics",
        "datasets": [{"dataset_id": "ds-001", "dataset_name": "Sales Analytics"}],
    }
]

SEMANTIC_MODELS = [
    {"model_name": "Sales Analytics", "tables": ["Sales", "Products", "Regions"]}
]

_SALES = pd.DataFrame(
    [
        {"Date": "2026-01-05", "Region": "North", "Product": "Widget A", "Quantity": 120, "Revenue": 3600.0},
        {"Date": "2026-01-05", "Region": "South", "Product": "Widget B", "Quantity": 80, "Revenue": 2400.0},
        {"Date": "2026-02-12", "Region": "North", "Product": "Widget A", "Quantity": 95, "Revenue": 2850.0},
        {"Date": "2026-02-12", "Region": "East", "Product": "Widget C", "Quantity": 60, "Revenue": 2700.0},
        {"Date": "2026-03-01", "Region": "South", "Product": "Widget B", "Quantity": 110, "Revenue": 3300.0},
        {"Date": "2026-03-15", "Region": "West", "Product": "Widget C", "Quantity": 75, "Revenue": 3375.0},
    ]
)

_PRODUCTS = pd.DataFrame(
    [
        {"Product": "Widget A", "Category": "Hardware", "UnitCost": 18.0},
        {"Product": "Widget B", "Category": "Hardware", "UnitCost": 20.0},
        {"Product": "Widget C", "Category": "Premium", "UnitCost": 32.0},
    ]
)

_REGIONS = pd.DataFrame(
    [
        {"Region": "North", "Country": "Sweden", "Manager": "A. Lind"},
        {"Region": "South", "Country": "Denmark", "Manager": "B. Holm"},
        {"Region": "East", "Country": "Finland", "Manager": "C. Saari"},
        {"Region": "West", "Country": "Norway", "Manager": "D. Berg"},
    ]
)

_TABLES = {"Sales": _SALES, "Products": _PRODUCTS, "Regions": _REGIONS}

_REFRESH_HISTORY: dict[str, list[dict]] = {
    "ds-001": [
        {
            "request_id": "r-1",
            "status": "Completed",
            "start_time": "2026-07-24T02:00:00Z",
            "end_time": "2026-07-24T02:04:12Z",
        },
        {
            "request_id": "r-2",
            "status": "Completed",
            "start_time": "2026-07-23T02:00:00Z",
            "end_time": "2026-07-23T02:03:47Z",
        },
    ]
}


def get_table(table_name: str) -> pd.DataFrame:
    if table_name not in _TABLES:
        raise ValueError(f"Unknown table '{table_name}'")
    return _TABLES[table_name]


def guess_table_from_dax(dax_query: str) -> str:
    """Very naive stand-in for actually executing a DAX query: pick whichever
    known table name appears in the query text, defaulting to Sales."""
    for name in _TABLES:
        if name.lower() in dax_query.lower():
            return name
    return "Sales"


def get_refresh_history(dataset_id: str) -> list[dict]:
    return _REFRESH_HISTORY.get(dataset_id, [])


def trigger_refresh(dataset_id: str) -> dict:
    history = _REFRESH_HISTORY.setdefault(dataset_id, [])
    entry = {
        "request_id": f"r-{len(history) + 1}",
        "status": "Completed",
        "start_time": "2026-07-25T00:00:00Z",
        "end_time": "2026-07-25T00:03:00Z",
    }
    history.insert(0, entry)
    return entry
