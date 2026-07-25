from langchain_core.messages import AIMessage, HumanMessage

from data_analyst.agents.datasource.graph import build_datasource_graph
from data_analyst.clients.llm.factory import FakeToolCallingChatModel


def test_run_dax_query_stages_a_sandbox_ref():
    llm = FakeToolCallingChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "pbi_mcp_run_dax_query",
                        "args": {"model_name": "Sales Analytics", "dax_query": "EVALUATE Sales"},
                        "id": "c1",
                    }
                ],
            ),
            AIMessage(content="Fetched the Sales table."),
        ]
    )
    graph = build_datasource_graph(llm)

    result = graph.invoke({"messages": [HumanMessage(content="get sales data")]})

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    assert "sandbox_ref" in tool_messages[0].content
    assert result["messages"][-1].content == "Fetched the Sales table."


def test_get_refresh_history_is_read_only_no_trigger_tool_available():
    llm = FakeToolCallingChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "pbi_rest_get_refresh_history", "args": {"dataset_id": "ds-001"}, "id": "c1"}],
            ),
            AIMessage(content="The dataset last refreshed successfully."),
        ]
    )
    graph = build_datasource_graph(llm)

    result = graph.invoke({"messages": [HumanMessage(content="when did the dataset last refresh?")]})

    tool_names = {tc["name"] for m in result["messages"] if m.type == "ai" for tc in (m.tool_calls or [])}
    assert "pbi_rest_trigger_dataset_refresh" not in tool_names
    assert result["messages"][-1].content == "The dataset last refreshed successfully."
