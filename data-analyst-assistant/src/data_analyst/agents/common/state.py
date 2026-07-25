"""Shared state shape: every agent graph (orchestrator, datasource, analysis)
keeps its own running conversation as a `messages` list using LangGraph's
`add_messages` reducer, so ToolNode/tools_condition work the same way in
each of them."""
from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
