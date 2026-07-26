"""End-to-end tests against the real FastAPI app (lifespan included), covering
the full orchestrator -> datasource -> analysis -> respond round trip over
HTTP, plus the demo-mode default path with no scripting required.
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from fastapi.testclient import TestClient

from data_analyst.agents.orchestrator.graph import build_orchestrator_graph
from data_analyst.app.api import app
from data_analyst.app.dependencies import get_graph
from data_analyst.clients.llm.factory import FakeToolCallingChatModel


class ScriptedRoutingModel(FakeToolCallingChatModel):
    """Extends the demo fake model so the supervisor's structured-output
    routing can be scripted too, not just tool-calling responses."""

    routes: list[dict]

    def with_structured_output(self, schema, **kwargs):
        state = {"i": 0}

        def _invoke(*_args, **_kwargs):
            route = self.routes[state["i"]]
            state["i"] += 1
            return schema(**route)

        return RunnableLambda(_invoke)


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


def test_demo_mode_answers_without_any_scripting():
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "hello, what can you do?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["reply"]


def test_demo_mode_gives_the_same_canned_reply_on_a_second_turn():
    """Regression test: the demo model's single canned response is the same
    message object reused on every call. A prior version of
    FakeToolCallingChatModel returned that object as-is, and LangGraph's
    add_messages reducer assigns it an id and mutates it in place on first
    use - so reusing the identical (already-id'd) object on a later turn of
    the same thread got treated as an update to the earlier message instead
    of a new one, leaving the just-sent human message last. The API then
    read that back as "the reply", i.e. the assistant appeared to echo
    whatever the user just typed on every turn after the first.
    """
    with TestClient(app) as client:
        first = client.post("/chat", json={"message": "hello, what can you do?"})
        thread_id = first.json()["thread_id"]
        second = client.post("/chat", json={"message": "banana banana banana", "thread_id": thread_id})

    assert second.json()["reply"] == first.json()["reply"]
    assert second.json()["reply"] != "banana banana banana"


def test_full_flow_delegates_through_both_specialists():
    llm = ScriptedRoutingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "pbi_mcp_list_semantic_models", "args": {}, "id": "c1"}],
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
    graph = build_orchestrator_graph(llm)
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
        responses=[AIMessage(content="Which region and time period do you mean?")],
        routes=[{"next": "clarify", "reason": "ambiguous request"}],
    )
    graph = build_orchestrator_graph(llm)
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


def test_specialist_asks_for_clarification_without_a_second_supervisor_call():
    """A single scripted route is enough: if the orchestrator needed a second
    supervisor decision after the datasource specialist ran, this would raise
    IndexError instead of returning - proving the specialist's own
    clarifying question short-circuits straight to the reply."""
    clarify_call = {"name": "request_clarification", "args": {"question": "Which time period do you mean?"}, "id": "c1"}
    llm = ScriptedRoutingModel(
        responses=[
            AIMessage(content="", tool_calls=[clarify_call]),
            AIMessage(content="Which time period do you mean?"),
        ],
        routes=[{"next": "datasource"}],
    )
    graph = build_orchestrator_graph(llm)
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


def test_follow_up_reuses_fetched_data_without_a_new_datasource_call():
    dax_call = {
        "name": "pbi_rest_run_dax_query",
        "args": {
            "model_name": "Sales Analytics",
            "table": "Sales",
            "group_by": ["Region"],
            "filters": [],
            "measures": [{"name": "Total Revenue", "aggregation": "SUM", "column": "Revenue"}],
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
    graph = build_orchestrator_graph(llm)
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


def test_stream_demo_mode_emits_status_tokens_and_a_matching_done_event():
    with TestClient(app) as client:
        events = _stream_events(client, {"message": "hello, what can you do?"})

    kinds = [e["type"] for e in events]
    assert kinds[0] == "start"
    assert "status" in kinds
    assert "token" in kinds  # FakeToolCallingChatModel._stream tokenizes the canned reply
    assert kinds[-1] == "done"

    done = events[-1]
    streamed_reply = "".join(e["content"] for e in events if e["type"] == "token")
    assert done["status"] == "completed"
    assert done["reply"] == streamed_reply


def test_stream_full_flow_emits_status_and_tool_events_for_both_specialists():
    llm = ScriptedRoutingModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "pbi_mcp_list_semantic_models", "args": {}, "id": "c1"}]),
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
    graph = build_orchestrator_graph(llm)
    app.dependency_overrides[get_graph] = lambda: graph

    try:
        with TestClient(app) as client:
            events = _stream_events(client, {"message": "what data is available, and what's 1 + 1?"})
    finally:
        app.dependency_overrides.pop(get_graph, None)

    status_nodes = [e["node"] for e in events if e["type"] == "status"]
    assert status_nodes == ["supervisor", "datasource", "supervisor", "analysis", "supervisor", "respond"]

    tool_names = [e["name"] for e in events if e["type"] == "tool"]
    assert tool_names == ["pbi_mcp_list_semantic_models", "python_sandbox_execute"]

    done = events[-1]
    assert done["type"] == "done"
    assert done["status"] == "completed"
    assert done["reply"] == "You have one semantic model available, and 1 + 1 is 2."


def test_stream_clarify_path_reports_clarification_needed():
    llm = ScriptedRoutingModel(
        responses=[AIMessage(content="Which region and time period do you mean?")],
        routes=[{"next": "clarify", "reason": "ambiguous request"}],
    )
    graph = build_orchestrator_graph(llm)
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
