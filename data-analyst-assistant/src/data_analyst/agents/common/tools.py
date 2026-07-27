"""Tools shared across specialist agents (not the orchestrator, which has no
tools of its own - see agents/orchestrator/chains.py)."""
from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

from data_analyst.agents.common.models import Clarification


@tool
async def flag_ambiguity(
    reason: str,
    options: Annotated[
        list[str],
        Field(
            description="2-3 clearly distinct, mutually exclusive candidate answers.",
            min_length=2,
            max_length=3,
        ),
    ],
) -> dict:
    """Flag that you can't proceed confidently, instead of guessing.

    Call this - instead of any other tool - when you're not confident which
    data/columns/filters the user means, or what analysis would answer
    their question. `reason` should be a complete, user-readable sentence
    describing the ambiguity (it may be relayed to the user as-is), plus
    2-3 clearly distinct options (e.g. specific regions, time periods, or
    metrics) rather than leaving it open-ended. This does not itself ask
    the user anything - the orchestrator decides how (and whether) to
    surface it. After calling it, end your turn with a brief final message;
    don't call another tool afterwards.
    """
    # The tool's own parameter is named `reason` (what a specialist gives),
    # but it's read back into the same `Clarification` shape the
    # supervisor's own upfront path produces - see that model's docstring
    # for why one shape serves both without implying this is already
    # ready-to-send text.
    return Clarification(question=reason, options=options).model_dump()
