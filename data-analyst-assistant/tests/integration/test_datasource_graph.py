from langchain_core.messages import AIMessage, HumanMessage

from data_analyst.agents.datasource.graph import build_datasource_graph
from data_analyst.clients.llm.factory import FakeToolCallingChatModel

_DAX_TOOL_CALL = {
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


async def test_run_dax_query_stages_a_sandbox_ref():
    llm = FakeToolCallingChatModel(
        responses=[
            AIMessage(content="", tool_calls=[_DAX_TOOL_CALL]),
            AIMessage(content="Fetched revenue by region."),
        ]
    )
    graph = build_datasource_graph(llm)

    result = await graph.ainvoke({"messages": [HumanMessage(content="revenue by region")], "session_id": "sess-dax-1"})

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    assert "sandbox_ref" in tool_messages[0].content
    assert '"reused": false' in tool_messages[0].content
    assert result["messages"][-1].content == "Fetched revenue by region."


async def test_run_dax_query_reuses_cached_result_within_same_session():
    llm = FakeToolCallingChatModel(
        responses=[AIMessage(content="", tool_calls=[_DAX_TOOL_CALL]), AIMessage(content="First fetch.")]
    )
    graph = build_datasource_graph(llm)
    await graph.ainvoke({"messages": [HumanMessage(content="revenue by region")], "session_id": "sess-dax-2"})

    llm2 = FakeToolCallingChatModel(
        responses=[AIMessage(content="", tool_calls=[_DAX_TOOL_CALL]), AIMessage(content="Second fetch.")]
    )
    result = await build_datasource_graph(llm2).ainvoke(
        {"messages": [HumanMessage(content="revenue by region again")], "session_id": "sess-dax-2"}
    )

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert '"reused": true' in tool_messages[0].content


async def test_run_dax_query_does_not_reuse_across_different_sessions():
    llm = FakeToolCallingChatModel(
        responses=[AIMessage(content="", tool_calls=[_DAX_TOOL_CALL]), AIMessage(content="Fetch.")]
    )
    result = await build_datasource_graph(llm).ainvoke(
        {"messages": [HumanMessage(content="revenue by region")], "session_id": "sess-dax-isolated"}
    )
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert '"reused": false' in tool_messages[0].content


async def test_run_dax_query_invalid_columns_returns_error_not_raise():
    bad_call = {
        "name": "pbi_rest_run_dax_query",
        "args": {"model_name": "Sales Analytics", "table": "Sales", "group_by": ["Bogus"], "filters": [], "measures": []},
        "id": "c1",
    }
    llm = FakeToolCallingChatModel(
        responses=[AIMessage(content="", tool_calls=[bad_call]), AIMessage(content="Let me fix that.")]
    )
    graph = build_datasource_graph(llm)

    result = await graph.ainvoke({"messages": [HumanMessage(content="bad request")], "session_id": "sess-dax-error"})

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert "error" in tool_messages[0].content.lower()


async def test_get_refresh_history_is_read_only_no_trigger_tool_available():
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

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="when did the dataset last refresh?")], "session_id": "sess-refresh"}
    )

    tool_names = {tc["name"] for m in result["messages"] if m.type == "ai" for tc in (m.tool_calls or [])}
    assert "pbi_rest_trigger_dataset_refresh" not in tool_names
    assert result["messages"][-1].content == "The dataset last refreshed successfully."


async def test_can_ask_for_clarification_instead_of_guessing():
    clarify_call = {"name": "request_clarification", "args": {"question": "Which time period do you mean?"}, "id": "c1"}
    llm = FakeToolCallingChatModel(
        responses=[
            AIMessage(content="", tool_calls=[clarify_call]),
            AIMessage(content="Which time period do you mean?"),
        ]
    )
    graph = build_datasource_graph(llm)

    result = await graph.ainvoke({"messages": [HumanMessage(content="how much revenue")], "session_id": "sess-clarify"})

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert tool_messages[0].name == "request_clarification"
    assert result["messages"][-1].content == "Which time period do you mean?"
