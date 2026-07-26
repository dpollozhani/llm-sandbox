"""FastAPI surface for the assistant.

Each HTTP request is stateless; conversations are resumed by passing back the
`thread_id` returned from a previous call, which the checkpointer uses to
keep the message history for that conversation, and which also scopes the
session-bound data store (see clients/sandbox/client.py) so a follow-up
question can reuse already-fetched data.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from data_analyst.app.dependencies import get_graph
from data_analyst.app.lifespan import lifespan
from data_analyst.app.web import CHAT_PAGE_HTML

app = FastAPI(title="Data Analyst Assistant", lifespan=lifespan)

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
    by hand - the same /chat endpoints any other client would use."""
    return CHAT_PAGE_HTML


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, graph: CompiledStateGraph = Depends(get_graph)) -> ChatResponse:
    thread_id = body.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=body.message)], "turns": 0, "session_id": thread_id},
        config=config,
    )
    return _to_chat_response(thread_id, result)


async def _stream_chat_events(body: ChatRequest, graph: CompiledStateGraph, thread_id: str) -> AsyncIterator[str]:
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
      streaming (real providers do; the LLM_PROVIDER=demo fake model
      doesn't, so demo mode shows status updates but not live typing).
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
            {"messages": [HumanMessage(content=body.message)], "turns": 0, "session_id": thread_id},
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
async def chat_stream(body: ChatRequest, graph: CompiledStateGraph = Depends(get_graph)) -> StreamingResponse:
    thread_id = body.thread_id or str(uuid.uuid4())
    return StreamingResponse(
        _stream_chat_events(body, graph, thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
