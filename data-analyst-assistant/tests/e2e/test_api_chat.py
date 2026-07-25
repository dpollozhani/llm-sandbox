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
