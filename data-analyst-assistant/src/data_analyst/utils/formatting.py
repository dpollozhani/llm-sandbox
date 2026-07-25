"""Formatting helpers for turning numbers into the kind of text an analyst
assistant should say out loud."""
from __future__ import annotations


def format_currency(value: float, currency: str = "") -> str:
    formatted = f"{value:,.0f}"
    return f"{currency}{formatted}" if currency else formatted


def format_number(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"
