"""Tools shared across specialist agents (not the orchestrator, which has no
tools of its own - see agents/orchestrator/chains.py)."""
from __future__ import annotations

from langchain_core.tools import tool


@tool
async def request_clarification(question: str) -> dict:
    """Ask the user a clarifying question instead of guessing.

    Call this - instead of any other tool - when you're not confident which
    data/columns/filters the user means, or what analysis would answer
    their question. After calling it, end your turn by relaying `question`
    as your final answer; don't call another tool afterwards.
    """
    return {"clarification_requested": True, "question": question}
