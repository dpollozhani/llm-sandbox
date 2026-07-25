"""FastAPI surface for the assistant.

Each HTTP request is stateless; conversations are resumed by passing back the
`thread_id` returned from the first call. When a specialist's tool pauses for
human approval (see agents/datasource/nodes.py), `/chat` returns
`status: "approval_required"` with the interrupt payload instead of a reply,
and the caller resumes via `/chat/{thread_id}/approve`.
"""
from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel

from .dependencies import get_graph
from .lifespan import lifespan

app = FastAPI(title="Data Analyst Assistant", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ApproveRequest(BaseModel):
    approved: bool


class ChatResponse(BaseModel):
    thread_id: str
    status: str
    reply: str | None = None
    interrupt: dict | None = None


def _to_response(thread_id: str, result: dict) -> ChatResponse:
    if interrupts := result.get("__interrupt__"):
        return ChatResponse(thread_id=thread_id, status="approval_required", interrupt=interrupts[0].value)
    return ChatResponse(thread_id=thread_id, status="completed", reply=result["messages"][-1].content)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, graph: CompiledStateGraph = Depends(get_graph)) -> ChatResponse:
    thread_id = body.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=body.message)], "turns": 0},
        config=config,
    )
    return _to_response(thread_id, result)


@app.post("/chat/{thread_id}/approve", response_model=ChatResponse)
async def approve(
    thread_id: str, body: ApproveRequest, graph: CompiledStateGraph = Depends(get_graph)
) -> ChatResponse:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="unknown thread_id")
    if not snapshot.next:
        raise HTTPException(status_code=409, detail="thread is not awaiting approval")

    result = await graph.ainvoke(Command(resume=body.approved), config=config)
    return _to_response(thread_id, result)
