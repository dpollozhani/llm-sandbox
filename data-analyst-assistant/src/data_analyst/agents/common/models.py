"""Pydantic models shared across agent packages."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Clarification(BaseModel):
    """A short question plus 2-3 clearly distinct options a user can pick
    from, so a frontend can render them as buttons instead of relying on
    free text. Produced three ways, all read into this same shape: the
    supervisor's own upfront "clarify" decision
    (`agents/orchestrator/nodes.py::build_clarify_node`, via structured
    output), where `question` is already the ready-to-send text; a
    specialist's `flag_ambiguity` tool call (`agents/common/tools.py`),
    where `question` is really just the specialist's own reason for the
    ambiguity - not itself user-facing; and the analysis specialist's
    `suggest_followup` tool call (same module), same shape again but
    non-blocking - offered *alongside* an already-complete answer rather
    than instead of one. `flag_ambiguity` is shared by every specialist;
    `suggest_followup` isn't - see agents/datasource/prompts.py for why the
    datasource agent has no tool for it at all.
    `agents/orchestrator/nodes.py::_run_specialist` is the only place that
    decides how (and whether) to surface either tool's result, composing
    the final message deterministically (`_compose_ambiguity_message`)
    for the blocking case rather than relaying the specialist's own
    phrasing as if a model had written it for the user. One shared model
    rather than three near-identical ones, since the distinction is in
    which code path produced it and what that path does with it next, not
    in the shape itself."""

    question: str = ""
    options: list[str] = Field(default_factory=lambda: ["", ""], min_length=2, max_length=3)
