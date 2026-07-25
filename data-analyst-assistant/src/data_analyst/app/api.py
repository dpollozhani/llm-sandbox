"""FastAPI surface for the assistant.

Each HTTP request is stateless; conversations are resumed by passing back the
`thread_id` returned from a previous call, which the checkpointer uses to
keep the message history for that conversation, and which also scopes the
session-bound data store (see clients/sandbox/client.py) so a follow-up
question can reuse already-fetched data.
"""
from __future__ import annotations

import uuid
from typing import Literal

from fastapi import Depends, FastAPI
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from data_analyst.app.dependencies import get_graph
from data_analyst.app.lifespan import lifespan

app = FastAPI(title="Data Analyst Assistant", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    status: Literal["completed", "clarification_needed"]
    reply: str | None = None


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
    status = "clarification_needed" if result.get("next") == "clarify" else "completed"
    return ChatResponse(thread_id=thread_id, status=status, reply=result["messages"][-1].content)
