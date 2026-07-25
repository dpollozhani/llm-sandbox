from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

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


def test_trigger_refresh_pauses_for_approval_and_resumes():
    llm = FakeToolCallingChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "pbi_rest_trigger_dataset_refresh", "args": {"dataset_id": "ds-001"}, "id": "c1"}],
            ),
            AIMessage(content="Refresh triggered."),
        ]
    )
    graph = build_datasource_graph(llm, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "test-thread"}}

    paused = graph.invoke({"messages": [HumanMessage(content="refresh please")]}, config=config)
    assert paused.get("__interrupt__")
    payload = paused["__interrupt__"][0].value
    assert payload["resource_id"] == "ds-001"

    resumed = graph.invoke(Command(resume=True), config=config)
    tool_messages = [m for m in resumed["messages"] if m.type == "tool"]
    assert '"status": "completed"' in tool_messages[-1].content


def test_trigger_refresh_can_be_rejected():
    llm = FakeToolCallingChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "pbi_rest_trigger_dataset_refresh", "args": {"dataset_id": "ds-001"}, "id": "c1"}],
            ),
            AIMessage(content="Okay, not refreshing."),
        ]
    )
    graph = build_datasource_graph(llm, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "test-thread-reject"}}

    graph.invoke({"messages": [HumanMessage(content="refresh please")]}, config=config)
    resumed = graph.invoke(Command(resume=False), config=config)

    tool_messages = [m for m in resumed["messages"] if m.type == "tool"]
    assert '"status": "cancelled"' in tool_messages[-1].content
