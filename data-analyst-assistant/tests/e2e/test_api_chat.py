"""End-to-end tests against the real FastAPI app (lifespan included), covering
the full orchestrator -> datasource -> analysis -> respond round trip over
HTTP. Every test drives its own scripted model via
`app.dependency_overrides[get_graph]` (see tests/conftest.py for why the app
can still start up without one), and a fake Power BI REST/MCP client swapped
into the orchestrator graph's datasource specialist so nothing here needs a
Power BI tenant or sign-in. `get_pbi_tokens` (the dependency that otherwise
requires a signed-in session - see app/auth.py) is overridden for every test
in this module by the `_fake_pbi_tokens` fixture below."""
from __future__ import annotations

import json

import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from fastapi.testclient import TestClient
from pydantic import Field

from data_analyst.agents.common.models import Clarification
from data_analyst.agents.orchestrator.graph import build_orchestrator_graph
from data_analyst.app.api import PBITokens, app, get_pbi_tokens
from data_analyst.app.dependencies import get_graph
from data_analyst.clients.llm.factory import FakeToolCallingChatModel
from data_analyst.clients.powerbi.dax import DaxQuerySpec, build_dax_query, validate_dax_query


class ScriptedRoutingModel(FakeToolCallingChatModel):
    """Extends FakeToolCallingChatModel so the supervisor's routing (`Route`)
    and clarify (`Clarification`) structured outputs can be scripted too,
    not just tool-calling responses. Each schema gets its own independent
    call counter/list: `build_supervisor_chain` and `build_clarify_chain`
    each call `with_structured_output` once, at chain-build time, not once
    per turn, so a single shared counter would make the two schemas fight
    over the same list index."""

    routes: list[dict] = Field(default_factory=list)
    clarifications: list[dict] = Field(default_factory=list)

    def with_structured_output(self, schema, **kwargs):
        items = self.clarifications if schema is Clarification else self.routes
        state = {"i": 0}

        def _invoke(*_args, **_kwargs):
            item = items[state["i"]]
            state["i"] += 1
            return schema(**item)

        return RunnableLambda(_invoke)


class _FakeRestClient:
    """Stands in for PBIRestClient's real HTTP calls - see the identical
    fake in tests/integration/test_datasource_graph.py."""

    async def run_dax_query(self, access_token: str, spec: DaxQuerySpec):
        assert access_token == "tok-pbi"
        dax_query = build_dax_query(spec)
        validate_dax_query(dax_query, spec)
        return dax_query, pd.DataFrame([{"Total Revenue": 18225}])


class _FakeMcpClient:
    async def get_semantic_metadata(self, access_token: str, model_name: str):
        assert access_token == "tok-pbi"
        return {"tables": [{"name": "Sales", "columns": ["Region", "Revenue"]}]}


def _build_graph(llm):
    return build_orchestrator_graph(llm, mcp_client=_FakeMcpClient(), rest_client=_FakeRestClient())


@pytest.fixture(autouse=True)
def _fake_pbi_tokens():
    app.dependency_overrides[get_pbi_tokens] = lambda: PBITokens(token="tok-pbi")
    yield
    app.dependency_overrides.pop(get_pbi_tokens, None)


def _stream_events(client: TestClient, body: dict) -> list[dict]:
    """Collects every SSE `data:` payload from POST /chat/stream into a list,
    in arrival order - mirrors how app/web.py's JS parses the same stream."""
    events = []
    with client.stream("POST", "/chat/stream", json=body) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


def test_chat_page_is_served_at_root():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Data Analyst Assistant" in response.text
    assert "/chat" in response.text


def test_chat_without_signing_in_returns_401_with_a_login_url():
    app.dependency_overrides.pop(get_pbi_tokens, None)  # undo the autouse override for this one test
    try:
        with TestClient(app) as client:
            response = client.post("/chat", json={"message": "hello"})
    finally:
        app.dependency_overrides[get_pbi_tokens] = lambda: PBITokens(token="tok-pbi")

    assert response.status_code == 401
    assert response.json()["detail"]["login_url"] == "/auth/login"


def test_chat_endpoint_answers_using_a_simple_scripted_model():
    llm = FakeToolCallingChatModel(responses=[AIMessage(content="Here's what I can do.")])
    graph = _build_graph(llm)
    app.dependency_overrides[get_graph] = lambda: graph

    try:
        with TestClient(app) as client:
            response = client.post("/chat", json={"message": "hello, what can you do?"})
    finally:
        app.dependency_overrides.pop(get_graph, None)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["reply"] == "Here's what I can do."


def test_scripted_reply_reused_across_turns_does_not_echo_the_next_message():
    """Regression test: FakeToolCallingChatModel's single scripted response is
    the same message object reused on every call (e.g. a model with only one
    item in `responses`, invoked again on a second turn of the same thread).
    A prior version returned that object as-is, and LangGraph's add_messages
    reducer assigns it an id and mutates it in place on first use - so
    reusing the identical (already-id'd) object on a later turn got treated
    as an update to the earlier message instead of a new one, leaving the
    just-sent human message last. The API then read that back as "the
    reply", i.e. the assistant appeared to echo whatever the user just typed
    on every turn after the first.
    """
    llm = FakeToolCallingChatModel(responses=[AIMessage(content="Here's what I can do.")])
    graph = _build_graph(llm)
    app.dependency_overrides[get_graph] = lambda: graph

    try:
        with TestClient(app) as client:
            first = client.post("/chat", json={"message": "hello, what can you do?"})
            thread_id = first.json()["thread_id"]
            second = client.post("/chat", json={"message": "banana banana banana", "thread_id": thread_id})
    finally:
        app.dependency_overrides.pop(get_graph, None)

    assert second.json()["reply"] == first.json()["reply"]
    assert second.json()["reply"] != "banana banana banana"


def test_full_flow_delegates_through_both_specialists():
    llm = ScriptedRoutingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "pbi_mcp_get_semantic_metadata", "args": {"model_name": "Sales Analytics"}, "id": "c1"}],
            ),
            AIMessage(content="There is one semantic model: Sales Analytics."),
            AIMessage(
                content="",
                tool_calls=[{"name": "python_sandbox_execute", "args": {"code": "result = 1 + 1"}, "id": "c2"}],
            ),
            AIMessage(content="Computed 1 + 1 = 2."),
            AIMessage(content="You have one semantic model available, and 1 + 1 is 2."),
        ],
        routes=[{"next": "datasource"}, {"next": "analysis"}, {"next": "respond"}],
    )
    graph = _build_graph(llm)
    app.dependency_overrides[get_graph] = lambda: graph

    try:
        with TestClient(app) as client:
            response = client.post("/chat", json={"message": "what data is available, and what's 1 + 1?"})
    finally:
        app.dependency_overrides.pop(get_graph, None)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["reply"] == "You have one semantic model available, and 1 + 1 is 2."


def test_supervisor_asks_for_clarification_when_uncertain():
    llm = ScriptedRoutingModel(
        responses=[],
        routes=[{"next": "clarify", "reason": "ambiguous request"}],
        clarifications=[
            {
                "question": "Which region and time period do you mean?",
                "options": ["North, last month", "South, last quarter"],
            }
        ],
    )
    graph = _build_graph(llm)
    app.dependency_overrides[get_graph] = lambda: graph

    try:
        with TestClient(app) as client:
            response = client.post("/chat", json={"message": "how are we doing?"})
    finally:
        app.dependency_overrides.pop(get_graph, None)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "clarification_needed"
    assert body["reply"] == "Which region and time period do you mean?"
    assert body["options"] == ["North, last month", "South, last quarter"]


def test_specialist_asks_for_clarification_without_a_second_supervisor_call():
    """A single scripted route is enough: if the orchestrator needed a second
    supervisor decision after the datasource specialist ran, this would raise
    IndexError instead of returning - proving the specialist's own
    clarifying question short-circuits straight to the reply."""
    clarify_call = {
        "name": "request_clarification",
        "args": {"question": "Which time period do you mean?", "options": ["Last month", "Last quarter"]},
        "id": "c1",
    }
    llm = ScriptedRoutingModel(
        responses=[
            AIMessage(content="", tool_calls=[clarify_call]),
            AIMessage(content="Which time period do you mean?"),
        ],
        routes=[{"next": "datasource"}],
    )
    graph = _build_graph(llm)
    app.dependency_overrides[get_graph] = lambda: graph

    try:
        with TestClient(app) as client:
            response = client.post("/chat", json={"message": "how much revenue did we make"})
    finally:
        app.dependency_overrides.pop(get_graph, None)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "clarification_needed"
    assert body["reply"] == "Which time period do you mean?"
    assert body["options"] == ["Last month", "Last quarter"]


def test_follow_up_reuses_fetched_data_without_a_new_datasource_call():
    dax_call = {
        "name": "pbi_rest_run_dax_query",
        "args": {
            "model_name": "Sales Analytics",
            "group_by": [{"table": "Sales", "column": "Region"}],
            "filters": [],
            "measures": [{"name": "Total Revenue", "aggregation": "SUM", "table": "Sales", "column": "Revenue"}],
        },
        "id": "c1",
    }
    llm = ScriptedRoutingModel(
        responses=[
            AIMessage(content="", tool_calls=[dax_call]),
            AIMessage(content="Fetched revenue by region, sandbox_ref=df_1."),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "python_sandbox_execute", "args": {"code": "result = df['Total Revenue'].sum()", "sandbox_ref": "df_1"}, "id": "c2"}
                ],
            ),
            AIMessage(content="Total is 18225."),
            AIMessage(content="Total revenue across regions is 18225."),
            # follow-up: supervisor skips "datasource" entirely, reuses df_1
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "python_sandbox_execute", "args": {"code": "result = df['Total Revenue'].mean()", "sandbox_ref": "df_1"}, "id": "c3"}
                ],
            ),
            AIMessage(content="Average is 4556.25."),
            AIMessage(content="On average, about 4556 per region."),
        ],
        routes=[
            {"next": "datasource"},
            {"next": "analysis"},
            {"next": "respond"},
            {"next": "analysis"},
            {"next": "respond"},
        ],
    )
    graph = _build_graph(llm)
    app.dependency_overrides[get_graph] = lambda: graph

    try:
        with TestClient(app) as client:
            first = client.post("/chat", json={"message": "total revenue by region?"})
            thread_id = first.json()["thread_id"]
            second = client.post(
                "/chat", json={"message": "what's the average across those regions?", "thread_id": thread_id}
            )
    finally:
        app.dependency_overrides.pop(get_graph, None)

    assert first.json()["reply"] == "Total revenue across regions is 18225."
    assert second.json()["thread_id"] == thread_id
    assert second.json()["reply"] == "On average, about 4556 per region."


def test_stream_simple_reply_emits_status_tokens_and_a_matching_done_event():
    llm = FakeToolCallingChatModel(responses=[AIMessage(content="Here's what I can do.")])
    graph = _build_graph(llm)
    app.dependency_overrides[get_graph] = lambda: graph

    try:
        with TestClient(app) as client:
            events = _stream_events(client, {"message": "hello, what can you do?"})
    finally:
        app.dependency_overrides.pop(get_graph, None)

    kinds = [e["type"] for e in events]
    assert kinds[0] == "start"
    assert "status" in kinds
    assert "token" in kinds  # FakeToolCallingChatModel._stream tokenizes the scripted reply
    assert kinds[-1] == "done"

    done = events[-1]
    streamed_reply = "".join(e["content"] for e in events if e["type"] == "token")
    assert done["status"] == "completed"
    assert done["reply"] == streamed_reply == "Here's what I can do."


def test_stream_full_flow_emits_status_and_tool_events_for_both_specialists():
    llm = ScriptedRoutingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "pbi_mcp_get_semantic_metadata", "args": {"model_name": "Sales Analytics"}, "id": "c1"}],
            ),
            AIMessage(content="There is one semantic model: Sales Analytics."),
            AIMessage(
                content="",
                tool_calls=[{"name": "python_sandbox_execute", "args": {"code": "result = 1 + 1"}, "id": "c2"}],
            ),
            AIMessage(content="Computed 1 + 1 = 2."),
            AIMessage(content="You have one semantic model available, and 1 + 1 is 2."),
        ],
        routes=[{"next": "datasource"}, {"next": "analysis"}, {"next": "respond"}],
    )
    graph = _build_graph(llm)
    app.dependency_overrides[get_graph] = lambda: graph

    try:
        with TestClient(app) as client:
            events = _stream_events(client, {"message": "what data is available, and what's 1 + 1?"})
    finally:
        app.dependency_overrides.pop(get_graph, None)

    status_nodes = [e["node"] for e in events if e["type"] == "status"]
    assert status_nodes == ["supervisor", "datasource", "supervisor", "analysis", "supervisor", "respond"]

    tool_names = [e["name"] for e in events if e["type"] == "tool"]
    assert tool_names == ["pbi_mcp_get_semantic_metadata", "python_sandbox_execute"]

    done = events[-1]
    assert done["type"] == "done"
    assert done["status"] == "completed"
    assert done["reply"] == "You have one semantic model available, and 1 + 1 is 2."


def test_stream_clarify_path_reports_clarification_needed():
    llm = ScriptedRoutingModel(
        responses=[],
        routes=[{"next": "clarify", "reason": "ambiguous request"}],
        clarifications=[
            {
                "question": "Which region and time period do you mean?",
                "options": ["North, last month", "South, last quarter"],
            }
        ],
    )
    graph = _build_graph(llm)
    app.dependency_overrides[get_graph] = lambda: graph

    try:
        with TestClient(app) as client:
            events = _stream_events(client, {"message": "how are we doing?"})
    finally:
        app.dependency_overrides.pop(get_graph, None)

    assert [e["node"] for e in events if e["type"] == "status"] == ["supervisor", "clarify"]
    done = events[-1]
    assert done["status"] == "clarification_needed"
    assert done["reply"] == "Which region and time period do you mean?"
    assert done["options"] == ["North, last month", "South, last quarter"]
