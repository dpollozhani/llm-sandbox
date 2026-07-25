from langchain_core.messages import AIMessage, HumanMessage

from data_analyst.agents.analysis.graph import build_analysis_graph
from data_analyst.clients.llm.factory import FakeToolCallingChatModel
from data_analyst.clients.sandbox.client import sandbox_client


def test_sandbox_execute_uses_staged_dataframe():
    import pandas as pd

    ref = sandbox_client.stage(pd.DataFrame([{"Region": "North", "Revenue": 10.0}, {"Region": "South", "Revenue": 5.0}]))

    llm = FakeToolCallingChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "python_sandbox_execute",
                        "args": {"code": "result = df['Revenue'].sum()", "sandbox_ref": ref},
                        "id": "c1",
                    }
                ],
            ),
            AIMessage(content="Total revenue is 15."),
        ]
    )
    graph = build_analysis_graph(llm)

    result = graph.invoke({"messages": [HumanMessage(content="sum revenue")]})

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert '"result": 15.0' in tool_messages[0].content
    assert result["messages"][-1].content == "Total revenue is 15."
