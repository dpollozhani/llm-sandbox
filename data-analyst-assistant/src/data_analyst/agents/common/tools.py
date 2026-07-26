"""Tools shared across specialist agents (not the orchestrator, which has no
tools of its own - see agents/orchestrator/chains.py)."""
from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

from data_analyst.agents.common.models import Clarification


@tool
async def request_clarification(
    question: str,
    options: Annotated[
        list[str],
        Field(
            description="2-3 clearly distinct, mutually exclusive answers the user could pick instead of typing a reply.",
            min_length=2,
            max_length=3,
        ),
    ],
) -> dict:
    """Ask the user a clarifying question instead of guessing.

    Call this - instead of any other tool - when you're not confident which
    data/columns/filters the user means, or what analysis would answer
    their question. Always give 2-3 clearly distinct options (e.g. specific
    regions, time periods, or metrics) rather than leaving it open-ended - a
    frontend may render them as buttons instead of requiring free text.
    After calling it, end your turn by relaying `question` as your final
    answer; don't call another tool afterwards.
    """
    return Clarification(question=question, options=options).model_dump()
