"""FastAPI surface for the assistant.

Each HTTP request is stateless; conversations are resumed by passing back the
`thread_id` returned from a previous call, which the checkpointer uses to
keep the message history for that conversation, and which also scopes the
session-bound data store (see clients/sandbox/client.py) so a follow-up
question can reuse already-fetched data.

Every `/chat*` call also needs the caller's own delegated Power BI tokens
(see `clients/powerbi/auth.py` for why - row-level security requires them),
resolved by `get_pbi_tokens` from either the browser's signed-in session
(`app/auth.py`) or `X-PBI-*-Token` headers a client (e.g. `cli.py`) already
holds from its own sign-in flow.
"""
from __future__ import annotations

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

from data_analyst.app.auth import get_token_broker, save_broker
from data_analyst.app.auth import router as auth_router
from data_analyst.app.dependencies import get_graph
from data_analyst.app.lifespan import lifespan
from data_analyst.app.web import CHAT_PAGE_HTML
from data_analyst.clients.powerbi.auth import PBI_MCP_SCOPE, PBI_REST_SCOPE
from data_analyst.config.settings import get_settings

app = FastAPI(title="Data Analyst Assistant", lifespan=lifespan)
app.include_router(auth_router)

_NOT_SIGNED_IN = {"login_url": "/auth/login", "message": "Sign in with Power BI access to use the assistant."}


@dataclass
class PBITokens:
    rest: str | None
    mcp: str | None


async def get_pbi_tokens(request: Request) -> PBITokens:
    """CLI clients (see cli.py) get their own tokens via a device-code flow
    and send them directly as headers; the browser instead relies on the
    signed-in session from app/auth.py. Either way, at least one token is
    required - a chat turn that never touches Power BI still needs *a*
    signed-in identity, since the supervisor decides mid-run whether the
    datasource specialist is needed at all."""
    header_rest = request.headers.get("X-PBI-Rest-Token")
    header_mcp = request.headers.get("X-PBI-Mcp-Token")
    if header_rest or header_mcp:
        return PBITokens(rest=header_rest, mcp=header_mcp)

    broker = get_token_broker(request, get_settings())
    if broker is None:
        raise HTTPException(status_code=401, detail=_NOT_SIGNED_IN)

    async def _try(scope: str) -> str | None:
        try:
            return await broker.get_token(scope)
        except RuntimeError:
            return None

    rest_token = await _try(PBI_REST_SCOPE)
    mcp_token = await _try(PBI_MCP_SCOPE)
    save_broker(request, broker)

    if rest_token is None and mcp_token is None:
        raise HTTPException(status_code=401, detail=_NOT_SIGNED_IN)
    return PBITokens(rest=rest_token, mcp=mcp_token)

# Human-readable status shown while a given orchestrator node is running -
# see /chat/stream. Keyed by node name (LangGraph's `metadata.langgraph_node`
# on each streamed event), not by tool or subgraph-internal node names, so
# only these five top-level transitions are ever surfaced to the client.
_NODE_STATUS = {
    "supervisor": "Thinking about what to do next...",
    "datasource": "Querying Power BI...",
    "analysis": "Running analysis...",
    "respond": "Composing answer...",
    "clarify": "Composing a clarifying question...",
}


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    status: Literal["completed", "clarification_needed"]
    reply: str | None = None


def _to_chat_response(thread_id: str, final_state: dict) -> ChatResponse:
    status = "clarification_needed" if final_state.get("next") == "clarify" else "completed"
    return ChatResponse(thread_id=thread_id, status=status, reply=final_state["messages"][-1].content)


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
    config = {"configurable": {"thread_id": thread_id}}
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=body.message)],
            "turns": 0,
            "session_id": thread_id,
            "pbi_rest_token": tokens.rest,
            "pbi_mcp_token": tokens.mcp,
        },
        config=config,
    )
    return _to_chat_response(thread_id, result)


async def _stream_chat_events(
    body: ChatRequest, graph: CompiledStateGraph, thread_id: str, tokens: PBITokens
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
    """

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    yield sse({"type": "start", "thread_id": thread_id})

    config = {"configurable": {"thread_id": thread_id}}
    final_state: dict | None = None
    try:
        async for event in graph.astream_events(
            {
                "messages": [HumanMessage(content=body.message)],
                "turns": 0,
                "session_id": thread_id,
                "pbi_rest_token": tokens.rest,
                "pbi_mcp_token": tokens.mcp,
            },
            config=config,
            version="v2",
        ):
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
            elif kind == "on_tool_start":
                yield sse({"type": "tool", "name": event.get("name"), "input": event["data"].get("input")})
            elif kind == "on_chat_model_stream" and node in ("respond", "clarify"):
                content = event["data"]["chunk"].content
                if content:
                    yield sse({"type": "token", "content": content})
            elif kind == "on_chain_end" and event.get("name") == "LangGraph" and node is None:
                final_state = event["data"]["output"]
    except Exception as exc:  # noqa: BLE001 - surfaced to the client, not raised
        yield sse({"type": "error", "message": str(exc)})
        return

    yield sse({"type": "done", **_to_chat_response(thread_id, final_state).model_dump()})


@app.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    graph: CompiledStateGraph = Depends(get_graph),
    tokens: PBITokens = Depends(get_pbi_tokens),
) -> StreamingResponse:
    thread_id = body.thread_id or str(uuid.uuid4())
    return StreamingResponse(
        _stream_chat_events(body, graph, thread_id, tokens),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
