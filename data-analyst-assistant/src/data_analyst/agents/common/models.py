"""Pydantic models shared across agent packages."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Clarification(BaseModel):
    """A short question plus 2-3 clearly distinct options a user can pick
    from, so a frontend can render them as buttons instead of relying on
    free text. Produced two ways, both read into this same shape: the
    supervisor's own upfront "clarify" decision
    (`agents/orchestrator/nodes.py::build_clarify_node`, via structured
    output), where `question` is already the ready-to-send text; and a
    specialist's `flag_ambiguity` tool call (`agents/common/tools.py`),
    where `question` is really just the specialist's own reason for the
    ambiguity - not itself user-facing.
    `agents/orchestrator/nodes.py::_run_specialist` is the only place that
    decides how (and whether) to surface the latter, composing the final
    message deterministically (`_compose_ambiguity_message`) rather than
    relaying the specialist's own phrasing as if a model had written it for
    the user. One shared model rather than two near-identical ones, since
    the distinction is in which code path produced it and what that path
    does with it next, not in the shape itself."""

    question: str = ""
    options: list[str] = Field(default_factory=lambda: ["", ""], min_length=2, max_length=3)
