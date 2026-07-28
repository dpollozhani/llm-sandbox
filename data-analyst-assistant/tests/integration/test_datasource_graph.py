import json

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage

from data_analyst.agents.datasource.graph import build_datasource_graph
from data_analyst.clients.llm.factory import FakeToolCallingChatModel
from data_analyst.clients.powerbi.dax import DaxQuerySpec, build_dax_query, validate_dax_query

_DAX_TOOL_CALL = {
    "name": "pbi_rest_run_dax_query",
    "args": {
        "model_name": "Sales Analytics",
        "group_by": [{"table": "Sales", "column": "Region"}],
        "filters": [],
        "measures": [{"name": "Total Revenue", "aggregation": "SUM", "table": "Sales", "column": "Revenue"}],
    },
    "id": "c1",
}


class _FakeRestClient:
    """Stands in for PBIRestClient's real HTTP calls: still builds/validates
    the structured query the way the real client does, just returns a canned
    DataFrame instead of calling Power BI - and asserts the delegated token
    threaded through from the graph's initial state actually arrives here."""

    async def run_dax_query(self, access_token: str, spec: DaxQuerySpec):
        assert access_token == "tok-pbi"
        dax_query = build_dax_query(spec)
        validate_dax_query(dax_query, spec)
        if [c.column for c in spec.group_by] == ["Bogus"]:
            raise ValueError("Unknown column(s) for table 'Sales': ['Bogus']")
        return dax_query, pd.DataFrame([{"Region": "North", "Total Revenue": 18225}])


class _FailingRestClient:
    """Simulates a connection/protocol failure below the ValueError level -
    e.g. the mcp SDK/httpx's background tasks (anyio task groups) raising
    something other than ValueError, which used to escape the tool
    uncaught and crash the whole graph run instead of returning a normal
    {"error": ...} tool result."""

    async def run_dax_query(self, access_token: str, spec: DaxQuerySpec):
        raise ExceptionGroup("unhandled errors in a TaskGroup", [ConnectionError("boom")])


class _FailingMcpClient:
    async def get_semantic_metadata(self, access_token: str, model_name: str):
        raise ExceptionGroup("unhandled errors in a TaskGroup", [ConnectionError("boom")])


class _CountingMcpClient:
    def __init__(self) -> None:
        self.calls = 0

    async def get_semantic_metadata(self, access_token: str, model_name: str):
        self.calls += 1
        return {"tables": [{"name": "Sales"}]}


def _graph(llm):
    return build_datasource_graph(llm, rest_client=_FakeRestClient())


def _state(session_id: str, messages: list) -> dict:
    return {"messages": messages, "session_id": session_id, "pbi_token": "tok-pbi"}


async def test_run_dax_query_stages_a_dataset_id():
    llm = FakeToolCallingChatModel(
        responses=[
            AIMessage(content="", tool_calls=[_DAX_TOOL_CALL]),
            AIMessage(content="Fetched revenue by region."),
        ]
    )
    graph = _graph(llm)

    result = await graph.ainvoke(_state("sess-dax-1", [HumanMessage(content="revenue by region")]))

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    assert "dataset_id" in tool_messages[0].content
    assert '"reused": false' in tool_messages[0].content
    assert result["messages"][-1].content == "Fetched revenue by region."


async def test_run_dax_query_returns_a_friendly_query_summary():
    """For transparency: every fetch (new or reused) tells the caller
    exactly what group-by/filters/measures were used, in plain
    table.column form, not just the raw DAX text."""
    llm = FakeToolCallingChatModel(
        responses=[
            AIMessage(content="", tool_calls=[_DAX_TOOL_CALL]),
            AIMessage(content="Fetched revenue by region."),
        ]
    )
    graph = _graph(llm)

    result = await graph.ainvoke(_state("sess-dax-summary", [HumanMessage(content="revenue by region")]))

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert '"query": {"group_by": ["Sales.Region"], "filters": [], "measures": ["Total Revenue = SUM(Sales.Revenue)"]}' in tool_messages[0].content


async def test_run_dax_query_reuses_cached_result_within_same_session():
    llm = FakeToolCallingChatModel(
        responses=[AIMessage(content="", tool_calls=[_DAX_TOOL_CALL]), AIMessage(content="First fetch.")]
    )
    await _graph(llm).ainvoke(_state("sess-dax-2", [HumanMessage(content="revenue by region")]))

    llm2 = FakeToolCallingChatModel(
        responses=[AIMessage(content="", tool_calls=[_DAX_TOOL_CALL]), AIMessage(content="Second fetch.")]
    )
    result = await _graph(llm2).ainvoke(_state("sess-dax-2", [HumanMessage(content="revenue by region again")]))

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert '"reused": true' in tool_messages[0].content


async def test_run_dax_query_does_not_reuse_across_different_sessions():
    llm = FakeToolCallingChatModel(
        responses=[AIMessage(content="", tool_calls=[_DAX_TOOL_CALL]), AIMessage(content="Fetch.")]
    )
    result = await _graph(llm).ainvoke(_state("sess-dax-isolated", [HumanMessage(content="revenue by region")]))
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert '"reused": false' in tool_messages[0].content


async def test_run_dax_query_invalid_columns_returns_error_not_raise():
    bad_call = {
        "name": "pbi_rest_run_dax_query",
        "args": {"model_name": "Sales Analytics", "group_by": [{"table": "Sales", "column": "Bogus"}], "filters": [], "measures": []},
        "id": "c1",
    }
    llm = FakeToolCallingChatModel(
        responses=[AIMessage(content="", tool_calls=[bad_call]), AIMessage(content="Let me fix that.")]
    )
    graph = _graph(llm)

    result = await graph.ainvoke(_state("sess-dax-error", [HumanMessage(content="bad request")]))

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert "error" in tool_messages[0].content.lower()


async def test_run_dax_query_without_a_signed_in_token_returns_error_not_raise():
    llm = FakeToolCallingChatModel(
        responses=[AIMessage(content="", tool_calls=[_DAX_TOOL_CALL]), AIMessage(content="Please sign in.")]
    )
    graph = _graph(llm)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="revenue by region")], "session_id": "sess-no-token"}
    )

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert "sign" in tool_messages[0].content.lower()


async def test_run_dax_query_network_failure_returns_error_not_raise():
    llm = FakeToolCallingChatModel(
        responses=[AIMessage(content="", tool_calls=[_DAX_TOOL_CALL]), AIMessage(content="Something went wrong.")]
    )
    graph = build_datasource_graph(llm, rest_client=_FailingRestClient())

    result = await graph.ainvoke(_state("sess-dax-network-fail", [HumanMessage(content="revenue by region")]))

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert "boom" in tool_messages[0].content
    assert result["messages"][-1].content == "Something went wrong."


async def test_run_dax_query_failure_is_logged_with_the_query_that_triggered_it(caplog):
    """A `run_dax_query` failure (e.g. dax.py's `_result_key` not
    recognizing a returned column header) used to be visible nowhere but
    the tool's own `{"error": ...}` result, folded into the model's
    paraphrase - invisible in production logs. This also guards against a
    regression where logging it crashed instead: the query text has to be
    rebuilt independently for the log line, since `run_dax_query` raising
    before returning means it never got the chance to hand one back."""

    class _ColumnMismatchRestClient:
        async def run_dax_query(self, access_token: str, spec):
            raise ValueError("Column 'Region' not found in query result columns: ['NewRegion']")

    llm = FakeToolCallingChatModel(
        responses=[AIMessage(content="", tool_calls=[_DAX_TOOL_CALL]), AIMessage(content="Something went wrong.")]
    )
    graph = build_datasource_graph(llm, rest_client=_ColumnMismatchRestClient())

    with caplog.at_level("INFO"):
        result = await graph.ainvoke(_state("sess-dax-column-mismatch", [HumanMessage(content="revenue by region")]))

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert "not found in query result columns" in tool_messages[0].content
    assert any("not found in query result columns" in r.message and "EVALUATE" in r.message for r in caplog.records)


async def test_get_semantic_metadata_network_failure_returns_error_not_raise():
    call = {"name": "pbi_mcp_get_semantic_metadata", "args": {"model_name": "Sales Analytics"}, "id": "c1"}
    llm = FakeToolCallingChatModel(
        responses=[AIMessage(content="", tool_calls=[call]), AIMessage(content="Something went wrong.")]
    )
    graph = build_datasource_graph(llm, mcp_client=_FailingMcpClient())

    result = await graph.ainvoke(_state("sess-mcp-network-fail", [HumanMessage(content="show schema")]))

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert "boom" in tool_messages[0].content
    assert result["messages"][-1].content == "Something went wrong."


async def test_get_semantic_metadata_is_cached_across_fresh_specialist_delegations():
    """A specialist subgraph is rebuilt from scratch on every delegation and
    has no memory of an earlier delegation's own tool calls - without a
    session-scoped cache, a follow-up question needing the same model's
    schema would re-fetch it from the MCP server every single time."""
    call = {"name": "pbi_mcp_get_semantic_metadata", "args": {"model_name": "Sales Analytics"}, "id": "c1"}
    mcp_client = _CountingMcpClient()

    llm1 = FakeToolCallingChatModel(
        responses=[AIMessage(content="", tool_calls=[call]), AIMessage(content="Here's the schema.")]
    )
    graph1 = build_datasource_graph(llm1, mcp_client=mcp_client)
    await graph1.ainvoke(_state("sess-metadata-cache", [HumanMessage(content="show schema")]))

    llm2 = FakeToolCallingChatModel(
        responses=[AIMessage(content="", tool_calls=[call]), AIMessage(content="Here's the schema again.")]
    )
    graph2 = build_datasource_graph(llm2, mcp_client=mcp_client)
    result2 = await graph2.ainvoke(_state("sess-metadata-cache", [HumanMessage(content="show schema again")]))

    assert mcp_client.calls == 1
    tool_messages = [m for m in result2["messages"] if m.type == "tool"]
    assert tool_messages[0].content == '{"tables": [{"name": "Sales"}]}'


async def test_can_flag_ambiguity_instead_of_guessing():
    ambiguity_call = {
        "name": "flag_ambiguity",
        "args": {"reason": "Which time period do you mean?", "options": ["Last month", "Last quarter"]},
        "id": "c1",
    }
    llm = FakeToolCallingChatModel(
        responses=[
            AIMessage(content="", tool_calls=[ambiguity_call]),
            AIMessage(content="I need to know which time period."),
        ]
    )
    graph = _graph(llm)

    result = await graph.ainvoke(_state("sess-clarify", [HumanMessage(content="how much revenue")]))

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert tool_messages[0].name == "flag_ambiguity"
    assert '"error"' not in tool_messages[0].content
    assert json.loads(tool_messages[0].content) == {
        # Serialized from the shared Clarification model (see
        # agents/common/tools.py::flag_ambiguity) - "question" here is
        # really just the specialist's own reason, not ready-to-send text.
        "question": "Which time period do you mean?",
        "options": ["Last month", "Last quarter"],
    }
