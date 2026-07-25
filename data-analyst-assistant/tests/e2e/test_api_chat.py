"""End-to-end tests against the real FastAPI app (lifespan included), covering
the full orchestrator -> datasource -> analysis -> respond round trip over
HTTP, plus the demo-mode default path with no scripting required.
"""
from __future__ import annotations

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


def test_demo_mode_answers_without_any_scripting():
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "hello, what can you do?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["reply"]


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
