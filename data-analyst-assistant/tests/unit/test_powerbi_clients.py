import json
from contextlib import asynccontextmanager

import httpx
import pytest

from data_analyst.clients.powerbi.dax import DaxQuerySpec
from data_analyst.clients.powerbi.mcp import PBIMcpClient
from data_analyst.clients.powerbi.rest import PBIRestClient


class _FakeContent:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeToolResult:
    def __init__(self, text: str, is_error: bool = False) -> None:
        self.content = [_FakeContent(text)]
        self.isError = is_error


class _FakeSession:
    def __init__(self, result: _FakeToolResult) -> None:
        self._result = result
        self.calls: list[tuple[str, dict]] = []

    async def initialize(self) -> None:
        pass

    async def call_tool(self, name: str, args: dict) -> _FakeToolResult:
        self.calls.append((name, args))
        return self._result

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False


@asynccontextmanager
async def _fake_streamablehttp_client(url, headers=None):
    yield (None, None, None)

_EXECUTE_QUERIES_RESPONSE = {
    "results": [{"tables": [{"rows": [{"Products[Category]": "Hardware"}, {"Products[Category]": "Premium"}]}]}]
}


def _rest_client(handler) -> PBIRestClient:
    return PBIRestClient(transport=httpx.MockTransport(handler))


async def test_run_dax_query_executes_and_parses_the_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1.0/myorg/datasets/ds-001/executeQueries"
        assert request.headers["authorization"] == "Bearer tok-123"
        body = json.loads(request.content)
        assert body["queries"][0]["query"].startswith("SUMMARIZECOLUMNS(")
        return httpx.Response(200, json=_EXECUTE_QUERIES_RESPONSE)

    spec = DaxQuerySpec(model_name="Sales Analytics", table="Products", group_by=["Category"])
    dax_query, df = await _rest_client(handler).run_dax_query("tok-123", spec)

    assert dax_query.startswith("SUMMARIZECOLUMNS(")
    assert df.to_dict(orient="records") == [{"Category": "Hardware"}, {"Category": "Premium"}]


async def test_run_dax_query_unknown_model_raises_without_any_http_call():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not have made an HTTP call for an unknown model")

    spec = DaxQuerySpec(model_name="Nonexistent Model", table="Sales", group_by=["Region"])
    with pytest.raises(ValueError, match="Unknown semantic model"):
        await _rest_client(handler).run_dax_query("tok-123", spec)


async def test_run_dax_query_surfaces_power_bi_error_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Column 'Bogus' does not exist")

    spec = DaxQuerySpec(model_name="Sales Analytics", table="Sales", group_by=["Bogus"])
    with pytest.raises(ValueError, match="Bogus"):
        await _rest_client(handler).run_dax_query("tok-123", spec)


async def test_list_workspaces_returns_the_value_array():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1.0/myorg/groups"
        return httpx.Response(200, json={"value": [{"id": "ws-001", "name": "Retail Analytics"}]})

    workspaces = await _rest_client(handler).list_workspaces("tok-123")
    assert workspaces == [{"id": "ws-001", "name": "Retail Analytics"}]


async def test_get_refresh_history_returns_the_value_array():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1.0/myorg/datasets/ds-001/refreshes"
        return httpx.Response(200, json={"value": [{"status": "Completed"}]})

    history = await _rest_client(handler).get_refresh_history("tok-123", "ds-001")
    assert history == [{"status": "Completed"}]


def test_rest_client_has_no_trigger_refresh():
    assert not hasattr(PBIRestClient(), "trigger_refresh")


async def test_get_semantic_metadata_unknown_model_raises_without_any_mcp_call(monkeypatch):
    def _unreachable(*args, **kwargs):
        raise AssertionError("should not have connected to the MCP server for an unknown model")

    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.streamablehttp_client", _unreachable)

    with pytest.raises(ValueError, match="Unknown semantic model"):
        await PBIMcpClient().get_semantic_metadata("tok-123", "Nonexistent Model")


def test_mcp_client_has_no_query_execution():
    assert not hasattr(PBIMcpClient(), "run_dax_query")


async def test_get_semantic_metadata_calls_get_semantic_metadata_tool_and_parses_json(monkeypatch):
    fake_session = _FakeSession(_FakeToolResult(json.dumps({"tables": [{"name": "Products"}]})))
    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.streamablehttp_client", _fake_streamablehttp_client)
    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.ClientSession", lambda read, write: fake_session)

    result = await PBIMcpClient().get_semantic_metadata("tok-123", "Sales Analytics")

    assert result == {"tables": [{"name": "Products"}]}
    tool_name, args = fake_session.calls[0]
    assert tool_name == "GetSemanticMetadata"
    assert args == {"datasetId": "ds-001"}


async def test_get_semantic_metadata_raises_on_tool_error(monkeypatch):
    fake_session = _FakeSession(_FakeToolResult("permission denied", is_error=True))
    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.streamablehttp_client", _fake_streamablehttp_client)
    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.ClientSession", lambda read, write: fake_session)

    with pytest.raises(ValueError, match="permission denied"):
        await PBIMcpClient().get_semantic_metadata("tok-123", "Sales Analytics")
