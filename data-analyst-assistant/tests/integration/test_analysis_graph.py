import json

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage

from data_analyst.agents.analysis.graph import build_analysis_graph
from data_analyst.clients.llm.factory import FakeToolCallingChatModel
from data_analyst.clients.sandbox.client import get_sandbox_client


async def test_sandbox_execute_uses_staged_dataframe():
    session_id = "sess-analysis-1"
    dataset_id = get_sandbox_client(session_id).stage(
        pd.DataFrame([{"Region": "North", "Revenue": 10.0}, {"Region": "South", "Revenue": 5.0}])
    )

    llm = FakeToolCallingChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "python_sandbox_execute",
                        "args": {"code": "result = df['Revenue'].sum()", "dataset_id": dataset_id},
                        "id": "c1",
                    }
                ],
            ),
            AIMessage(content="Total revenue is 15."),
        ]
    )
    graph = build_analysis_graph(llm)

    result = await graph.ainvoke({"messages": [HumanMessage(content="sum revenue")], "session_id": session_id})

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert '"result": 15.0' in tool_messages[0].content
    assert result["messages"][-1].content == "Total revenue is 15."


async def test_sandbox_execute_dataset_id_is_scoped_to_its_own_session():
    dataset_id = get_sandbox_client("sess-analysis-owner").stage(pd.DataFrame([{"x": 1}]))

    llm = FakeToolCallingChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "python_sandbox_execute", "args": {"code": "result = 1", "dataset_id": dataset_id}, "id": "c1"}
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    graph = build_analysis_graph(llm)

    result = await graph.ainvoke({"messages": [HumanMessage(content="use it")], "session_id": "sess-analysis-other"})

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert "unknown dataset_id" in tool_messages[0].content.lower()


async def test_can_flag_ambiguity_instead_of_guessing():
    ambiguity_call = {
        "name": "flag_ambiguity",
        "args": {"reason": "Which metric do you want analyzed?", "options": ["Total revenue", "Average revenue"]},
        "id": "c1",
    }
    llm = FakeToolCallingChatModel(
        responses=[
            AIMessage(content="", tool_calls=[ambiguity_call]),
            AIMessage(content="I need to know which metric."),
        ]
    )
    graph = build_analysis_graph(llm)

    result = await graph.ainvoke({"messages": [HumanMessage(content="analyze it")], "session_id": "sess-analysis-clarify"})

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert tool_messages[0].name == "flag_ambiguity"
    assert '"error"' not in tool_messages[0].content
    assert json.loads(tool_messages[0].content) == {
        # Serialized from the shared Clarification model (see
        # agents/common/tools.py::flag_ambiguity) - "question" here is
        # really just the specialist's own reason, not ready-to-send text.
        "question": "Which metric do you want analyzed?",
        "options": ["Total revenue", "Average revenue"],
    }
