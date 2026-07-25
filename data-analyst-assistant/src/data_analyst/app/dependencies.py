from __future__ import annotations

from fastapi import Request
from langgraph.graph.state import CompiledStateGraph

from ..config.settings import Settings


def get_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.graph


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings
