"""Shared state shape: every agent graph (orchestrator, datasource, analysis)
keeps its own running conversation as a `messages` list using LangGraph's
`add_messages` reducer, so ToolNode/tools_condition work the same way in
each of them. `session_id` scopes the session-bound data store (see
`clients/sandbox/client.py::get_sandbox_client`) so tools can stage and
reuse fetched data across turns of the same conversation via
`langgraph.prebuilt.InjectedState`, without the model ever seeing or
supplying it."""
from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    session_id: str
