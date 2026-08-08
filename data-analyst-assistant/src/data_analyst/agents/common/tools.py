"""Tools shared across specialist agents (not the orchestrator, which has no
tools of its own - see agents/orchestrator/nodes.py)."""
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
    surface it, discarding whatever you say next. After calling it, end
    your turn with a brief final message (it won't be shown, but don't
    call another tool afterwards).

    Every option must be a complete answer you could act on immediately if
    the user picked it - never a stand-in for "let me type the real answer
    myself" (e.g. "You provide the exact account code"). A user can always
    reply with free text instead of picking an option, so that fallback
    doesn't need - and shouldn't be - an option of its own; phrase it as
    part of `reason`'s question instead (e.g. "...or tell me the exact
    account code") and only list options that are themselves genuinely
    one-click answers.
    """
    # The tool's own parameter is named `reason` (what a specialist gives),
    # but it's read back into the same `Clarification` shape the
    # supervisor's own upfront path produces - see that model's docstring
    # for why one shape serves both without implying this is already
    # ready-to-send text.
    return Clarification(question=reason, options=options).model_dump()


@tool
async def suggest_followup(
    reason: str,
    options: Annotated[
        list[str],
        Field(
            description="2-3 clearly distinct, concrete next steps the user could ask for.",
            min_length=2,
            max_length=3,
        ),
    ],
) -> dict:
    """Offer 2-3 concrete next steps after you've already completed the
    current task - not instead of guessing, but instead of asking in your
    own final message's prose.

    Call this only once you have a real, complete answer AND there's a
    genuine, concretely-grounded fork in what to do next - not a generic
    "anything else?" catch-all, and not "I fetched/computed something but
    still need to know how to finish" (that's `flag_ambiguity`, since
    nothing's actually complete yet). Unlike `flag_ambiguity`, this never
    blocks or replaces your final answer: give your actual answer as
    normal, this is purely a supplementary suggestion alongside it. `reason`
    should be a short, complete sentence (may be relayed to the user as-is),
    plus 2-3 clearly distinct options. Your final message must NOT also ask
    which option the user wants or restate the options in prose - these are
    shown to the user separately as clickable choices, so doing both just
    duplicates the same question. Call this at most once per turn.
    """
    # Same shape as flag_ambiguity's Clarification for the same reason -
    # `_run_specialist` reads this one non-blocking (see
    # `_specialist_followup`/`followup_suggestion`), never short-circuiting
    # the turn the way a flagged ambiguity does.
    return Clarification(question=reason, options=options).model_dump()
