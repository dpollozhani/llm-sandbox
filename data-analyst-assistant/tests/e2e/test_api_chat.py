"""End-to-end tests against the real FastAPI app (lifespan included), covering
the full orchestrator -> datasource -> approval -> respond round trip over
HTTP, plus the demo-mode default path with no scripting required.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
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


def test_full_flow_delegates_pauses_for_approval_and_resumes():
    llm = ScriptedRoutingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "pbi_rest_trigger_dataset_refresh", "args": {"dataset_id": "ds-001"}, "id": "c1"}],
            ),
            AIMessage(content="Refresh triggered for Sales Analytics."),
            AIMessage(content="Done - I refreshed the Sales Analytics dataset for you."),
        ],
        routes=[{"next": "datasource"}, {"next": "respond"}],
    )
    graph = build_orchestrator_graph(llm, checkpointer=InMemorySaver())
    app.dependency_overrides[get_graph] = lambda: graph

    try:
        with TestClient(app) as client:
            chat_response = client.post("/chat", json={"message": "please refresh the sales dataset"})
            assert chat_response.status_code == 200
            chat_body = chat_response.json()
            assert chat_body["status"] == "approval_required"
            assert chat_body["interrupt"]["resource_id"] == "ds-001"
            thread_id = chat_body["thread_id"]

            approve_response = client.post(f"/chat/{thread_id}/approve", json={"approved": True})
            assert approve_response.status_code == 200
            approve_body = approve_response.json()
            assert approve_body["status"] == "completed"
            assert "refreshed" in approve_body["reply"].lower()

            repeat_response = client.post(f"/chat/{thread_id}/approve", json={"approved": True})
            assert repeat_response.status_code == 409

            unknown_response = client.post("/chat/does-not-exist/approve", json={"approved": True})
            assert unknown_response.status_code == 404
    finally:
        app.dependency_overrides.pop(get_graph, None)
