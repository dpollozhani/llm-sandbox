"""Tests the orchestrator's node-level wiring in isolation: that delegating to
a specialist seeds a fresh child conversation from the latest task (not the
orchestrator's full history) and folds the specialist's answer back in as a
single labeled message.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from data_analyst.agents.orchestrator.nodes import build_datasource_node
from data_analyst.clients.llm.factory import FakeToolCallingChatModel


def test_datasource_node_seeds_fresh_child_and_folds_back_summary():
    llm = FakeToolCallingChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "pbi_mcp_list_semantic_models", "args": {}, "id": "c1"}],
            ),
            AIMessage(content="There is one semantic model: Sales Analytics."),
        ]
    )
    node = build_datasource_node(llm)

    state = {
        "messages": [
            HumanMessage(content="irrelevant earlier turn, should not reach the specialist"),
            HumanMessage(content="what semantic models are available?"),
        ],
        "turns": 1,
        "next": "datasource",
    }
    update = node(state)

    assert len(update["messages"]) == 1
    folded = update["messages"][0]
    assert isinstance(folded, AIMessage)
    assert folded.content == "[datasource] There is one semantic model: Sales Analytics."
