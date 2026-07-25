"""Small pandas helpers shared by the Power BI and sandbox clients."""
from __future__ import annotations

import pandas as pd


def preview_records(df: pd.DataFrame, limit: int = 5) -> list[dict]:
    return df.head(limit).to_dict(orient="records")


def to_records(value: object) -> object:
    """Convert a DataFrame result to plain records; pass through anything else."""
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value
