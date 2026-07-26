"""Shared state shape: every agent graph (orchestrator, datasource, analysis)
keeps its own running conversation as a `messages` list using LangGraph's
`add_messages` reducer, so ToolNode/tools_condition work the same way in
each of them. `session_id` scopes the session-bound data store (see
`clients/sandbox/client.py::get_sandbox_client`) so tools can stage and
reuse fetched data across turns of the same conversation via
`langgraph.prebuilt.InjectedState`, without the model ever seeing or
supplying it. `pbi_rest_token`/`pbi_mcp_token` are the same idea applied to
auth: the current request's delegated Power BI tokens (see
`app/api.py::get_pbi_tokens`), injected into the datasource agent's tools so
row-level security is enforced as the signed-in user - never the
model's business, and never put in the message history. Both are
`NotRequired`: callers that never touch Power BI (the analysis subgraph,
most tests) don't have to supply them, and the tools that do read them
(`agents/datasource/nodes.py`) use `.get()` and treat a missing key the same
as `None` - required keys here would instead fail pydantic's schema
validation for *any* `InjectedState` tool the moment a caller omitted them,
regardless of whether that tool cares about auth at all."""
from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    session_id: str
    pbi_rest_token: NotRequired[str | None]
    pbi_mcp_token: NotRequired[str | None]
