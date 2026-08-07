"""FastAPI surface for the assistant.

Each HTTP request is stateless; conversations are resumed by passing back the
`thread_id` returned from a previous call, which the checkpointer uses to
keep the message history for that conversation, and which also scopes the
session-bound data store (see clients/sandbox/client.py) so a follow-up
question can reuse already-fetched data.

Every `/chat*` call also needs the caller's own delegated Power BI token
(see `clients/powerbi/auth.py` for why - row-level security requires it),
resolved by `get_pbi_tokens` from either the browser's signed-in session
(`app/auth.py`) or an `X-PBI-Token` header a client (e.g. `cli.py`) already
holds from its own sign-in flow.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from data_analyst.agents.orchestrator.nodes import MAX_TURNS
from data_analyst.app.auth import get_token_broker, save_broker
from data_analyst.app.auth import router as auth_router
from data_analyst.app.dependencies import get_graph
from data_analyst.app.lifespan import lifespan
from data_analyst.app.web import CHAT_PAGE_HTML
from data_analyst.config.settings import get_catalog, get_settings
from data_analyst.telemetry.logging import get_logger

app = FastAPI(title="Data Analyst Assistant", lifespan=lifespan)
app.include_router(auth_router)

_logger = get_logger("app.api")

_NOT_SIGNED_IN = {"login_url": "/auth/login", "message": "Sign in with Power BI access to use the assistant."}

# Explicit rather than relying on LangGraph's own default recursion limit,
# which is version-dependent (observed to differ by orders of magnitude
# between installs). A generous multiple of MAX_TURNS, not a magic number -
# each supervisor turn is a supervisor-node visit plus one delegate/respond/
# clarify visit, and specialist subgraphs are bounded separately (see
# agents/orchestrator/nodes.py::SPECIALIST_RECURSION_LIMIT).
ORCHESTRATOR_RECURSION_LIMIT = MAX_TURNS * 4

_HEARTBEAT_INTERVAL_SECONDS = 15
"""How often /chat/stream emits a bare SSE comment while no real event has
fired - a specialist's own tool call (e.g. the analysis agent's sandbox)
can legitimately run silent for tens of seconds with nothing to report yet.
Without something on the wire, an idle-timing-out proxy or gateway between
here and the client can - and in practice does - drop that "silent"
connection outright, which the client only ever sees as a raw network
failure (Safari's fetch() surfaces this as "Load failed", not as a caught
error this app produced - see app/web.py's fetch catch block). A short
comment line costs nothing and needs no client-side change to tolerate."""


@dataclass
class PBITokens:
    token: str


async def get_pbi_tokens(request: Request) -> PBITokens:
    """CLI clients (see cli.py) get their own token via a device-code flow
    and send it directly as a header; the browser instead relies on the
    signed-in session from app/auth.py."""
    header_token = request.headers.get("X-PBI-Token")
    if header_token:
        return PBITokens(token=header_token)

    broker = get_token_broker(request, get_settings())
    if broker is None:
        raise HTTPException(status_code=401, detail=_NOT_SIGNED_IN)

    try:
        token = await broker.get_token()
    except RuntimeError:
        raise HTTPException(status_code=401, detail=_NOT_SIGNED_IN) from None
    save_broker(request, broker)
    return PBITokens(token=token)

# Human-readable status shown while a given orchestrator node is running -
# see /chat/stream. Keyed by node name (LangGraph's `metadata.langgraph_node`
# on each streamed event), not by tool or subgraph-internal node names, so
# only these five top-level transitions are ever surfaced to the client.
_NODE_STATUS = {
    "supervisor": "Thinking about what to do next...",
    "datasource": "Delegating to data source agent...",
    "analysis": "Delegating to analysis agent...",
    "respond": "Composing answer...",
    "clarify": "Composing a clarifying question...",
}


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None
    model_names: list[str] | None = None
    """Scopes this session to part of the full configured catalog
    (`config/semantic_models.yaml`) - each entry is either a `model_name` or
    a `dataset_id` from that config. Resolving *which* model(s) a given
    caller should be scoped to (e.g. "this Power BI app maps to these two
    models") is entirely the caller's own concern - this field takes the
    already-resolved result, never an app/caller identifier this API would
    have to look up itself, so a caller with no notion of "app" (a
    standalone portal, the CLI) never pays for or touches that lookup.
    A real restriction for the life of this request - the assistant can
    neither describe nor actually query a model outside this subset (see
    `OrchestratorState.allowed_model_names`) - not just a default it can
    reach around. "Overridable" means the caller can send a different
    `model_names` (or omit it) on its *next* request, not that the
    assistant can override it mid-conversation. `None`: no scoping, the
    full catalog applies, unchanged from before this field existed."""


def _resolve_allowed_model_names(model_names: list[str] | None) -> list[str] | None:
    """`body.model_names` (each a `model_name` or `dataset_id`) resolved
    against the full configured catalog into the plain `model_name` list
    `OrchestratorState.allowed_model_names` expects - `None` through
    unchanged (no scoping requested). Raises 400 if an entry doesn't match
    anything: a caller passing this at all is asserting these are real,
    resolvable models, so silently dropping an unrecognized one would hide
    what's likely a frontend integration bug (a stale link, a typo) rather
    than surfacing it immediately."""
    if model_names is None:
        return None  # not provided at all - no scoping, the full catalog applies
    if not model_names:
        # Provided, but empty - distinct from omitting the field entirely
        # (handled above): asking to be scoped to nothing is a caller bug,
        # not a request for "no scoping."
        raise HTTPException(status_code=400, detail="model_names, if provided, must be non-empty")
    subset = get_catalog().subset(model_names)
    resolved = {m.model_name for m in subset.semantic_models} | {m.dataset_id for m in subset.semantic_models}
    unknown = [m for m in model_names if m not in resolved]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown semantic model(s): {unknown}")
    return [m.model_name for m in subset.semantic_models]


class ChatResponse(BaseModel):
    thread_id: str
    status: Literal["completed", "clarification_needed"]
    reply: str | None = None
    options: list[str] | None = None
    """2-3 clearly distinct options a frontend can render as buttons instead
    of requiring a typed reply - only set when `status` is
    "clarification_needed" (see agents/common/models.py::Clarification).
    Blocking: `reply` here is not a real answer, just the clarifying
    question itself - picking one of these (or replying in free text) is
    the only way to make progress."""
    suggested_options: list[str] | None = None
    """2-3 optional next-step suggestions from a specialist alongside a
    completed answer (see agents/common/tools.py::suggest_followup and
    OrchestratorState.followup_suggestion) - non-blocking: `reply` is
    already the real, complete answer, and picking one of these (or
    ignoring them and asking something else) are equally valid next moves.
    Only ever set when `status` is "completed" - a blocking clarification
    always takes priority over a supplementary suggestion."""


def _to_chat_response(thread_id: str, final_state: dict) -> ChatResponse:
    pending = final_state.get("pending_clarification")
    status = "clarification_needed" if pending is not None else "completed"
    followup = final_state.get("followup_suggestion") if pending is None else None
    return ChatResponse(
        thread_id=thread_id,
        status=status,
        reply=final_state["messages"][-1].content,
        options=pending["options"] if pending is not None else None,
        suggested_options=followup["options"] if followup is not None else None,
    )


def _log_graph_update(node: str, update: dict) -> None:
    """`Settings.debug_graph_state`-gated: logs exactly what one node's
    return dict changed - the same per-step payload LangGraph's own
    `stream_mode="updates"` yields (see `chat`), or the equivalent already
    sitting in an `astream_events` `on_chain_end` event's `output` (see
    `_stream_chat_events`) - not a custom log format standing in for
    either."""
    _logger.info("graph update: node=%s update=%s", node, update)


@app.get("/", response_class=HTMLResponse)
async def chat_page() -> str:
    """A minimal browser chat UI (see app/web.py) for trying the assistant
    by hand - the same /chat endpoints any other client would use. Whether
    the visitor is signed in is checked client-side via GET /auth/whoami
    (see app/auth.py), not here, since this just serves the static page."""
    return CHAT_PAGE_HTML


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    graph: CompiledStateGraph = Depends(get_graph),
    tokens: PBITokens = Depends(get_pbi_tokens),
) -> ChatResponse:
    thread_id = body.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": ORCHESTRATOR_RECURSION_LIMIT}
    input_state = {
        "messages": [HumanMessage(content=body.message)],
        "turns": 0,
        "session_id": thread_id,
        "pbi_token": tokens.token,
        "allowed_model_names": _resolve_allowed_model_names(body.model_names),
    }

    if get_settings().debug_graph_state:
        # stream_mode="updates" yields exactly what each node's own return
        # dict changed, one node at a time, as the run actually happens -
        # LangGraph's own tool for this, not `ainvoke` plus a separate
        # after-the-fact history replay. The final full state still needs
        # one `aget_state` call once the run's done, since "updates" only
        # ever gives per-node diffs, never the merged whole.
        async for update in graph.astream(input_state, config=config, stream_mode="updates"):
            for node, node_update in update.items():
                _log_graph_update(node, node_update)
        result = (await graph.aget_state(config)).values
    else:
        result = await graph.ainvoke(input_state, config=config)
    return _to_chat_response(thread_id, result)


async def _stream_chat_events(
    body: ChatRequest, graph: CompiledStateGraph, thread_id: str, tokens: PBITokens, allowed_model_names: list[str] | None
) -> AsyncIterator[str]:
    """Drive one /chat turn via `astream_events` instead of `ainvoke`, emitting
    Server-Sent Events as the orchestrator progresses:

    - "status": a top-level node (supervisor/datasource/analysis/respond/
      clarify) started - including ones inside a specialist's own subgraph,
      since LangGraph's ambient event propagation surfaces nested-graph
      events the same way it already does for the checkpointer (see
      docs/architecture.md).
    - "tool": a tool call started (name + args), for visibility into what
      the datasource/analysis agent is doing.
    - "token": a piece of the *final* answer as the model generates it -
      only for the respond/clarify node, so supervisor routing chatter and
      specialist-internal reasoning never leak into what looks like "the
      answer". Only fires with a chat model that actually implements
      streaming - both real providers (Anthropic, Azure OpenAI) do.
    - "done": the final payload, identical in shape to what POST /chat
      returns - the authoritative result, regardless of which/how many
      token events arrived before it.
    - "error": something raised before a result was produced.

    Also emits a bare SSE comment line (`_HEARTBEAT_INTERVAL_SECONDS`'s
    docstring explains why) whenever nothing else has been emitted in a
    while - invisible to a spec-compliant SSE parser and to this app's own
    manual reader (app/web.py's `readEvents` only ever looks for "data: "
    lines), so no client change is needed to tolerate it.
    """

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    yield sse({"type": "start", "thread_id": thread_id})

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": ORCHESTRATOR_RECURSION_LIMIT}
    debug_graph_state = get_settings().debug_graph_state
    final_state: dict | None = None

    # astream_events itself is consumed by a background task, feeding a
    # queue this generator reads with a timeout - so a long silent stretch
    # between events (e.g. the analysis agent's sandbox tool actually
    # running) still lets the loop below wake up on the timeout and emit a
    # heartbeat, instead of blocking on the next real event for however
    # long that takes.
    queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    async def _produce() -> None:
        try:
            async for event in graph.astream_events(
                {
                    "messages": [HumanMessage(content=body.message)],
                    "turns": 0,
                    "session_id": thread_id,
                    "pbi_token": tokens.token,
                    "allowed_model_names": allowed_model_names,
                },
                config=config,
                version="v2",
            ):
                await queue.put(event)
        except Exception as exc:  # noqa: BLE001 - surfaced to the client, not raised
            await queue.put(exc)
        finally:
            await queue.put(_DONE)

    producer = asyncio.create_task(_produce())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if item is _DONE:
                break
            if isinstance(item, Exception):
                yield sse({"type": "error", "message": str(item)})
                return

            event = item
            kind = event["event"]
            node = event.get("metadata", {}).get("langgraph_node")

            # A single node visit fires several nested on_chain_start events
            # (the node's own runnable, its inner chain's _invoke, etc.), all
            # tagged with the same `langgraph_node` metadata - only the one
            # whose *own* name matches the node name is that node's outermost
            # invocation, so this fires exactly once per visit. Node names
            # aren't deduped across the whole run: the same node (e.g.
            # "supervisor") legitimately runs, and should report status,
            # multiple times in one turn.
            if kind == "on_chain_start" and node in _NODE_STATUS and event.get("name") == node:
                yield sse({"type": "status", "node": node, "message": _NODE_STATUS[node]})
            elif debug_graph_state and kind == "on_chain_end" and node in _NODE_STATUS and event.get("name") == node:
                # The same outermost-invocation filter as on_chain_start above,
                # applied to its matching end event - `output` here is the
                # node's own return dict, the same per-node payload
                # stream_mode="updates" would yield (see `chat`), already
                # sitting in this event rather than needing a second,
                # after-the-fact aget_state_history walk once the run's done.
                _log_graph_update(node, event["data"].get("output"))
            elif kind == "on_tool_start":
                yield sse({"type": "tool", "name": event.get("name"), "input": event["data"].get("input")})
            elif kind == "on_chat_model_stream" and node in ("respond", "clarify"):
                content = event["data"]["chunk"].content
                if content:
                    yield sse({"type": "token", "content": content})
            elif kind == "on_chain_end" and event.get("name") == "LangGraph" and node is None:
                final_state = event["data"]["output"]
    finally:
        if not producer.done():
            producer.cancel()

    yield sse({"type": "done", **_to_chat_response(thread_id, final_state).model_dump()})


@app.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    graph: CompiledStateGraph = Depends(get_graph),
    tokens: PBITokens = Depends(get_pbi_tokens),
) -> StreamingResponse:
    thread_id = body.thread_id or str(uuid.uuid4())
    # Resolved before the stream starts (not inside _stream_chat_events'
    # own try/except) so an invalid model_names entry is a real 400
    # response, not an "error" SSE event after the stream's already begun.
    allowed_model_names = _resolve_allowed_model_names(body.model_names)
    return StreamingResponse(
        _stream_chat_events(body, graph, thread_id, tokens, allowed_model_names),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
