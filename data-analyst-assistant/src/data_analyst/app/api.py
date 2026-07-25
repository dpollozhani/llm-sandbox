"""FastAPI surface for the assistant.

Each HTTP request is stateless; conversations are resumed by passing back the
`thread_id` returned from a previous call, which the checkpointer uses to
keep the message history for that conversation.
"""
from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from .dependencies import get_graph
from .lifespan import lifespan

app = FastAPI(title="Data Analyst Assistant", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    status: str
    reply: str | None = None


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
    return ChatResponse(thread_id=thread_id, status="completed", reply=result["messages"][-1].content)
