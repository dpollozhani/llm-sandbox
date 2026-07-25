"""Builds the chat model and compiled orchestrator graph once per process,
storing them on `app.state` for the dependencies in `dependencies.py`."""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.memory import InMemorySaver

from data_analyst.agents.orchestrator.graph import build_orchestrator_graph
from data_analyst.clients.llm.factory import get_chat_model
from data_analyst.config.settings import get_settings
from data_analyst.telemetry.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)

    llm = get_chat_model(settings)
    # One InMemorySaver for the process lifetime: this *is* the session
    # store. It's process-local and lost on restart - swap in a
    # Postgres/Redis-backed checkpointer here for a real deployment without
    # touching graph.py or api.py.
    app.state.settings = settings
    app.state.graph = build_orchestrator_graph(llm, checkpointer=InMemorySaver())
    yield
