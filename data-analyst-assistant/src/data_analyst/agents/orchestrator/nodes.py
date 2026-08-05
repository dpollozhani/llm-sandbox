"""Orchestrator nodes: a routing supervisor plus wrappers that seed a fresh
specialist subgraph from the current task and fold its answer back in.
"""
from __future__ import annotations

import json
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

from data_analyst.agents.analysis.graph import build_analysis_graph
from data_analyst.agents.common.models import Clarification
from data_analyst.agents.common.tools import flag_ambiguity, suggest_followup
from data_analyst.agents.datasource.graph import build_datasource_graph
from data_analyst.agents.datasource.models import DataSourceQueryResult
from data_analyst.agents.orchestrator.history import build_prompt_messages, maybe_summarize_history
from data_analyst.agents.orchestrator.prompts import (
    CLARIFY_SYSTEM_PROMPT,
    RESPOND_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
)
from data_analyst.agents.orchestrator.state import OrchestratorState
from data_analyst.clients.powerbi.mcp import PBIMcpClient
from data_analyst.clients.powerbi.rest import PBIRestClient
from data_analyst.config.settings import Glossary, PowerBiCatalog, inject_glossary
from data_analyst.telemetry.logging import get_logger
from data_analyst.telemetry.tracing import trace_span

_logger = get_logger("agents.orchestrator.nodes")

MAX_TURNS = 6

SPECIALIST_RECURSION_LIMIT = 20
"""Caps a specialist's own internal agent<->tools loop (`_run_specialist`'s
`child_graph.ainvoke()`) - explicitly, rather than relying on LangGraph's
own default, which is version-dependent and has been observed to differ
by orders of magnitude between installs. Generous enough for several
legitimate tool-call/retry round trips, but still a bounded, intentional
cap rather than an implicit one - see the `GraphRecursionError` handling
below for what happens when it's hit."""


class Route(BaseModel):
    """Every field needs a default: `FakeToolCallingChatModel`'s
    `with_structured_output` stand-in constructs this with no arguments (see
    clients/llm/factory.py), which tests rely on."""

    next: Literal["datasource", "analysis", "respond", "clarify"] = "respond"
    reason: str = ""


def _render_resolved(resolved: list[dict] | None) -> str:
    """A compact `"question" -> "answer"` line per already-settled
    clarification, for injecting into a prompt - shared by the supervisor's
    routing decision and its own upfront clarify decision, so neither
    re-asks something already answered earlier in the conversation."""
    if not resolved:
        return ""
    lines = "\n".join(f'- "{r["question"]}" -> "{r["answer"]}"' for r in resolved)
    return f"\n\nAlready clarified earlier in this conversation:\n{lines}"


def _render_followup_suggestion(followup: dict | None) -> str:
    """A compact rendering of a specialist's non-blocking `suggest_followup`
    suggestion (if any), for the supervisor's routing prompt: unlike
    `pending_clarification`'s deterministic resume, the next message might
    pick this up or might not, so the routing LLM call still needs to
    decide - this is context for that decision, not a bypass of it."""
    if not followup:
        return ""
    options = ", ".join(f'"{o}"' for o in followup["options"])
    return (
        f"\n\nThe {followup['agent']} specialist already completed its answer and suggested a "
        f'follow-up: "{followup["question"]}" ({options}). If the user\'s reply picks one of these '
        f"up, route back to \"{followup['agent']}\" to continue it; otherwise route normally. Do not "
        'route to "clarify" to re-ask this same fork yourself - it is non-blocking by design.'
    )


def _describe_catalog(catalog: PowerBiCatalog | None) -> str:
    """The real, config-backed list of semantic model names, appended to a
    prompt so the supervisor/respond nodes can answer "which models are
    available" directly from actual config rather than guessing - they see
    no schema at all, only these names (see SUPERVISOR_SYSTEM_PROMPT/
    RESPOND_SYSTEM_PROMPT). Empty string (no-op when appended) if there's no
    catalog or it's empty."""
    if not catalog or not catalog.semantic_models:
        return ""
    names = ", ".join(f'"{m.model_name}"' for m in catalog.semantic_models)
    return f"\n\nAvailable semantic models: {names}."


def build_supervisor_node(llm: BaseChatModel, glossary: Glossary | None = None, catalog: PowerBiCatalog | None = None):
    router = llm.with_structured_output(Route)

    async def supervisor_node(state: OrchestratorState):
        turns = state.get("turns", 0)
        pending = state.get("pending_clarification")
        if pending and pending["agent"] in ("datasource", "analysis"):
            # A reply to a clarification a specific specialist asked - go
            # straight back to it instead of paying for (and risking a
            # mis-route from) a fresh supervisor routing decision. The
            # specialist itself (_run_specialist) reads `pending` to seed
            # itself with the original task plus what's now been answered.
            return {"next": pending["agent"], "turns": turns + 1}
        if turns >= MAX_TURNS:
            return {"next": "respond", "turns": turns}

        summary_update = await maybe_summarize_history(llm, state)
        history_summary = (summary_update or {}).get("history_summary", state.get("history_summary"))

        prompt = inject_glossary(SUPERVISOR_SYSTEM_PROMPT, glossary) + _describe_catalog(catalog)
        if data_context := state.get("data_context"):
            prompt += f"\n\nCurrently available data in this session: {DataSourceQueryResult(**data_context).describe()}"
        prompt += _render_resolved(state.get("resolved_clarifications"))
        prompt += _render_followup_suggestion(state.get("followup_suggestion"))
        context = build_prompt_messages(state["messages"], history_summary)
        route = await router.ainvoke([SystemMessage(content=prompt), *context])

        next_step = route.next
        if next_step == "clarify" and state.get("followup_suggestion") and state["messages"][-1].type != "human":
            # The prompt above already tells the router not to do this, but
            # a same-turn re-clarify of a suggestion that was explicitly
            # meant to be non-blocking is cheap and easy for the model to
            # get wrong under pressure - worth a deterministic backstop
            # rather than relying on prompt wording alone. "No new human
            # message since" is what distinguishes this from a genuinely
            # fresh routing decision the user actually asked for.
            _logger.debug(
                "supervisor picked 'clarify' right after a non-blocking followup_suggestion with no "
                "new human message in between - overriding to 'respond'"
            )
            next_step = "respond"

        update: dict = {"next": next_step, "turns": turns + 1}
        if summary_update:
            update.update(summary_update)
        return update

    return supervisor_node


def _latest_user_task(messages: list[AnyMessage]) -> AnyMessage:
    """The specialists get one scoped task per delegation, not the orchestrator's
    full history - but that task must always be the user's actual question, not
    whatever message happens to be last (which, after a first specialist has
    already run this turn, would be that specialist's own folded-back summary)."""
    for message in reversed(messages):
        if message.type == "human":
            return message
    return messages[-1]


def _original_task(messages: list[AnyMessage]) -> AnyMessage:
    """The human message that started the current exchange - the latest
    human message *before* the most recent one. When resuming a reply to a
    clarification, the most recent message is that reply itself, not the
    original ask; this walks one step further back to find it, so a
    specialist can be re-seeded with the actual task instead of just the
    isolated reply."""
    seen_latest = False
    for message in reversed(messages):
        if message.type == "human":
            if not seen_latest:
                seen_latest = True
                continue
            return message
    return messages[-1]


def _specialist_ambiguity(messages: list[AnyMessage]) -> Clarification | None:
    """The `Clarification` a specialist flagged during this run (see
    agents/common/tools.py::flag_ambiguity), if any - read directly from
    the tool call's own structured result rather than the model's own
    (freeform, easy-to-drift) restatement of it, so the orchestrator always
    composes the user-facing message from exactly what the tool was called
    with. `.question` here is really just the specialist's own reason for
    the ambiguity, not ready-to-send text - see `_compose_ambiguity_message`."""
    for message in messages:
        if message.type == "tool" and message.name == flag_ambiguity.name:
            return Clarification(**json.loads(message.content))
    return None


def _specialist_followup(messages: list[AnyMessage]) -> Clarification | None:
    """The `Clarification`-shaped follow-up a specialist suggested this run
    (see agents/common/tools.py::suggest_followup), if any - same
    read-from-the-tool-call approach as `_specialist_ambiguity`, but this
    one is non-blocking: the specialist still produced a real final answer
    this run, so unlike ambiguity this never short-circuits the turn."""
    for message in messages:
        if message.type == "tool" and message.name == suggest_followup.name:
            return Clarification(**json.loads(message.content))
    return None


def _fetched_dataset(messages: list[AnyMessage]) -> DataSourceQueryResult | None:
    """The `DataSourceQueryResult` behind the most recent successful
    `pbi_rest_run_dax_query` call during this run, if any - read from the
    tool's own structured result, not a specialist's own freeform summary of
    it, which isn't guaranteed to mention all of it. Searched newest-first
    so a failed attempt followed by a successful retry resolves to the
    retry."""
    for message in reversed(messages):
        if message.type != "tool" or message.name != "pbi_rest_run_dax_query":
            continue
        payload = json.loads(message.content)
        if "dataset_id" not in payload:
            continue  # an {"error": ...} result, not a successful fetch
        return DataSourceQueryResult(**payload)
    return None


def _compose_ambiguity_message(ambiguity: Clarification) -> str:
    """The user-facing text for a specialist's flagged ambiguity, composed
    deterministically - no extra LLM call. The whole point of a specialist
    reporting ambiguity (`flag_ambiguity`) rather than phrasing a question
    itself is that the orchestrator, not another model call, decides what
    the user sees; doing that with a template here keeps this path exactly
    as cheap as the specialist's own final answer, matching the supervisor's
    own upfront `clarify` path only in shape, not in cost."""
    return f"{ambiguity.question} ({' / '.join(ambiguity.options)})"


def _append_resolved(state: OrchestratorState) -> list[dict]:
    """`resolved_clarifications` with the currently pending clarification
    (if any) appended - resolved by the latest message, which is always the
    user's most recent reply whenever a clarification was pending. Returns
    the list unchanged if nothing was pending. Shared by every node that can
    clear/replace `pending_clarification`, so this bookkeeping lives in one
    place rather than being reimplemented at each call site."""
    pending = state.get("pending_clarification")
    resolved = list(state.get("resolved_clarifications") or [])
    if pending is None:
        return resolved
    reply = state["messages"][-1]
    resolved.append({"question": pending["reason"], "answer": getattr(reply, "content", str(reply))})
    return resolved


def _seed_content(state: OrchestratorState, task_content: str) -> str:
    """The full seed message body for a specialist delegation: what data is
    already available, plus a compact rendering of what's already been
    clarified this conversation (never the raw messages that established
    it) so a specialist doesn't re-derive - or re-ask - either from
    scratch, plus the task itself."""
    parts = []
    if data_context := state.get("data_context"):
        parts.append(f"(Available data in this session: {DataSourceQueryResult(**data_context).describe()})")
    if resolved := state.get("resolved_clarifications"):
        lines = "\n".join(f"- {r['question']} -> {r['answer']}" for r in resolved)
        parts.append(f"(Already clarified earlier in this conversation:\n{lines})")
    if followup := state.get("followup_suggestion"):
        parts.append(f"(You previously suggested this follow-up: \"{followup['question']}\" ({', '.join(followup['options'])}))")
    parts.append(task_content)
    return "\n\n".join(parts)


def _specialist_update(
    state: OrchestratorState,
    message: AIMessage,
    *,
    next: str | None = None,
    pending_clarification: dict | None = None,
    followup_suggestion: dict | None = None,
) -> dict:
    """The shape every `_run_specialist` return path shares: the folded-back
    message plus the same three clarification-bookkeeping fields, always
    explicitly set rather than left out (see `OrchestratorState`'s
    `pending_clarification`/`followup_suggestion` docstrings for why a
    stale value must never linger). Each call site passes only what's
    different about its own case; `resolved_clarifications` always goes
    through `_append_resolved` since any path can be resolving whatever was
    pending before the delegation that triggered it."""
    update: dict = {
        "messages": [message],
        "pending_clarification": pending_clarification,
        "resolved_clarifications": _append_resolved(state),
        "followup_suggestion": followup_suggestion,
    }
    if next is not None:
        update["next"] = next
    return update


async def _run_specialist(agent_name: str, build_graph_fn, llm: BaseChatModel, state: OrchestratorState) -> dict:
    pending = state.get("pending_clarification")

    if pending is not None:
        # The latest human message is a reply to a clarifying question
        # (this specialist's own, or the supervisor's upfront one), not a
        # fresh request - a rebuilt-from-scratch specialist subgraph has no
        # memory of that question (or the original ask before it) unless we
        # hand it the actual task plus what's now been answered.
        original = _original_task(state["messages"])
        reply = state["messages"][-1]
        task_content = (
            f"{original.content}\n\n"
            f'(Replying to: "{pending["reason"]}" -> "{getattr(reply, "content", str(reply))}")'
        )
    else:
        task_content = _latest_user_task(state["messages"]).content
    messages = [HumanMessage(content=_seed_content(state, task_content))]

    child_graph = build_graph_fn(llm)
    with trace_span("specialist.run", agent=agent_name):
        try:
            result = await child_graph.ainvoke(
                {
                    "messages": messages,
                    "session_id": state["session_id"],
                    "pbi_token": state.get("pbi_token"),
                },
                config={"recursion_limit": SPECIALIST_RECURSION_LIMIT},
            )
        except GraphRecursionError:
            # The specialist's own agent<->tools loop kept calling tools without
            # ever reaching a final answer (e.g. retrying a computation that
            # keeps failing) - LangGraph's hard step cap turned that into a raw
            # exception that would otherwise crash this whole turn. Fold back a
            # plain failure instead: the supervisor's own MAX_TURNS still bounds
            # how many times this can happen before "respond" takes over.
            return _specialist_update(
                state,
                AIMessage(
                    content=f"[{agent_name}] Couldn't complete this after many attempts - "
                    "try a simpler or narrower request."
                ),
            )
        # The specialist's own internal trace (tool calls, intermediate
        # reasoning) is never exposed to the orchestrator's own context - it's
        # folded into one summary below - but is still worth a DEBUG record
        # for observability, since otherwise it leaves no trace anywhere at
        # all once discarded.
        _logger.debug(
            "%s specialist trace: %d message(s), tools=%s",
            agent_name,
            len(result["messages"]),
            [m.name for m in result["messages"] if m.type == "tool"],
        )

    ambiguity = _specialist_ambiguity(result["messages"])
    if ambiguity is not None:
        # The specialist itself decided it couldn't proceed confidently and
        # flagged an ambiguity - the orchestrator (not the specialist)
        # composes the user-facing message and owns whatever was pending
        # before this delegation is now folded into `resolved_clarifications`
        # (the reply that triggered this run did resolve *that* much, even
        # though a new, narrower ambiguity has now come up). Setting `next`
        # here (rather than going back to "supervisor") is what lets
        # agents/orchestrator/graph.py route straight to END: no extra
        # supervisor round-trip, no separate "clarify" node call, just this
        # specialist's question (and options) as the reply.
        return _specialist_update(
            state,
            AIMessage(content=_compose_ambiguity_message(ambiguity)),
            next="clarify",
            pending_clarification={"agent": agent_name, "reason": ambiguity.question, "options": ambiguity.options},
        )

    last_message = result["messages"][-1]
    summary = getattr(last_message, "content", str(last_message))
    followup = _specialist_followup(result["messages"])
    update = _specialist_update(
        state,
        AIMessage(content=f"[{agent_name}] {summary}"),
        followup_suggestion={"agent": agent_name, "question": followup.question, "options": followup.options}
        if followup is not None
        else None,
    )
    if agent_name == "datasource":
        fetched = _fetched_dataset(result["messages"])
        if fetched is not None:
            # Lets a follow-up question route straight to "analysis" (or skip
            # re-delegating altogether) instead of always re-fetching - see
            # SUPERVISOR_SYSTEM_PROMPT and clients/sandbox/client.py's cache.
            # Left out of `update` (not overwritten with None) when this run
            # didn't fetch anything (e.g. only browsed the schema), so a
            # dataset from an earlier turn stays available. Stored as a
            # plain dict, not the DataSourceQueryResult itself - see
            # OrchestratorState.data_context's docstring for why.
            update["data_context"] = fetched.model_dump()
    return update


def build_datasource_node(
    llm: BaseChatModel,
    mcp_client: PBIMcpClient | None = None,
    rest_client: PBIRestClient | None = None,
    glossary: Glossary | None = None,
    catalog: PowerBiCatalog | None = None,
):
    def build_graph_fn(agent_llm: BaseChatModel):
        return build_datasource_graph(
            agent_llm, mcp_client=mcp_client, rest_client=rest_client, catalog=catalog, glossary=glossary
        )

    async def datasource_node(state: OrchestratorState):
        return await _run_specialist("datasource", build_graph_fn, llm, state)

    return datasource_node


def build_analysis_node(llm: BaseChatModel, glossary: Glossary | None = None):
    def build_graph_fn(agent_llm: BaseChatModel):
        return build_analysis_graph(agent_llm, glossary=glossary)

    async def analysis_node(state: OrchestratorState):
        return await _run_specialist("analysis", build_graph_fn, llm, state)

    return analysis_node


def build_respond_node(llm: BaseChatModel, glossary: Glossary | None = None, catalog: PowerBiCatalog | None = None):
    base_prompt = inject_glossary(RESPOND_SYSTEM_PROMPT, glossary) + _describe_catalog(catalog)

    async def respond_node(state: OrchestratorState):
        # Concrete grounding for RESPOND_SYSTEM_PROMPT's "only suggest a
        # follow-up when it's grounded in the currently available data"
        # rule - without this, the model's only signal for "what's in play
        # right now" is prose buried in the message history, not something
        # to build a scoped suggestion from.
        prompt = base_prompt
        if data_context := state.get("data_context"):
            prompt += f"\n\nCurrently available data in this session: {DataSourceQueryResult(**data_context).describe()}"
        if followup := state.get("followup_suggestion"):
            # Already shown to the user separately as clickable options
            # (see app/api.py's `suggested_options`) - restating them in
            # prose here, or asking the user to pick one before you'll
            # continue, would just duplicate that non-blocking suggestion
            # as if it were a blocking question.
            prompt += (
                f"\n\nThe {followup['agent']} specialist already suggested a follow-up this turn - "
                f"\"{followup['question']}\" ({', '.join(followup['options'])}) - it's already shown to the "
                "user separately as clickable options. Don't repeat or list them yourself, and don't ask "
                "the user which one they want either - just give the answer; the suggestion is an "
                "optional next step, not something that needs picking before you can finish."
            )
        context = build_prompt_messages(state["messages"], state.get("history_summary"))
        response = await llm.ainvoke([SystemMessage(content=prompt), *context])
        return {
            "messages": [response],
            "pending_clarification": None,
            "resolved_clarifications": _append_resolved(state),
        }

    return respond_node


def build_clarify_node(llm: BaseChatModel, glossary: Glossary | None = None):
    router = llm.with_structured_output(Clarification)
    base_prompt = inject_glossary(CLARIFY_SYSTEM_PROMPT, glossary)

    async def clarify_node(state: OrchestratorState):
        prompt = base_prompt + _render_resolved(state.get("resolved_clarifications"))
        context = build_prompt_messages(state["messages"], state.get("history_summary"))
        clarification = await router.ainvoke([SystemMessage(content=prompt), *context])
        return {
            "messages": [AIMessage(content=clarification.question)],
            "pending_clarification": {
                "agent": "supervisor",
                "reason": clarification.question,
                "options": clarification.options,
            },
            "resolved_clarifications": _append_resolved(state),
            # A fresh broad clarification takes priority - a stale
            # non-blocking suggestion from an earlier specialist run is
            # now moot.
            "followup_suggestion": None,
        }

    return clarify_node
