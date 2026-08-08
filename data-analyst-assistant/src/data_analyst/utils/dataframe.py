"""Small pandas helpers shared by the Power BI and sandbox clients."""
from __future__ import annotations

import numpy as np
import pandas as pd


def preview_records(df: pd.DataFrame, limit: int = 5) -> list[dict]:
    return df.head(limit).to_dict(orient="records")


def to_records(value: object) -> object:
    """Convert a DataFrame/Series result to plain records, and numpy scalars/
    arrays - however deeply nested in a plain dict/list/tuple - to native
    Python types, so a sandbox result is always JSON-safe crossing the tool
    boundary. numpy/scipy functions commonly hand back a bare `np.float64`,
    an `ndarray`, or a dict/tuple containing either - none of which
    `json.dumps` can serialize on its own. Anything else passes through
    unchanged."""
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):  # a numpy scalar, e.g. np.float64/np.int64/np.bool_
        return value.item()
    if isinstance(value, dict):
        return {k: to_records(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_records(v) for v in value]
    return value


def to_preview_records(value: object, limit: int = 5) -> list[dict]:
    """`value` (already `to_records`-normalized, so JSON-safe) coerced to a
    bounded list of dict rows if it's already shaped that way (a list of
    dicts) or trivially could be (a single dict, wrapped as one row) - `[]`
    for anything else (a scalar, a string, `None`), since there's no
    meaningful "row" to make of those. Same cap as `preview_records`, for
    the same reason: a concrete value should survive as structured data
    without risking an unbounded payload."""
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value[:limit]
    return []
