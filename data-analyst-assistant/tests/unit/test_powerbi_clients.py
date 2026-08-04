import json
from contextlib import asynccontextmanager

import httpx
import pytest

from data_analyst.clients.powerbi.dax import DaxColumn, DaxQuerySpec
from data_analyst.clients.powerbi.mcp import PBIMcpClient, get_metadata_cache
from data_analyst.clients.powerbi.rest import PBIRestClient
from data_analyst.config.settings import PowerBiCatalog, SemanticModelConfig

# A catalog of our own, independent of the real shipped
# config/semantic_models.yaml - these are client-layer tests, not tests of
# that specific config's content.
_CATALOG = PowerBiCatalog(semantic_models=[SemanticModelConfig(model_name="Test Model", dataset_id="ds-test")])


class _FakeContent:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeToolResult:
    def __init__(self, text: str, is_error: bool = False) -> None:
        self.content = [_FakeContent(text)]
        self.isError = is_error


class _FakeTool:
    def __init__(self, name: str, properties: dict) -> None:
        self.name = name
        self.inputSchema = {"properties": properties}


class _FakeListToolsResult:
    def __init__(self, tools: list[_FakeTool]) -> None:
        self.tools = tools


_DEFAULT_TOOLS = [_FakeTool("GetSemanticMetadata", {"datasetId": {"type": "string"}})]


class _FakeSession:
    def __init__(self, result: _FakeToolResult, tools: list[_FakeTool] | None = None) -> None:
        self._result = result
        self._tools = tools if tools is not None else _DEFAULT_TOOLS
        self.calls: list[tuple[str, dict]] = []

    async def initialize(self) -> None:
        pass

    async def list_tools(self) -> _FakeListToolsResult:
        return _FakeListToolsResult(self._tools)

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
    return PBIRestClient(catalog=_CATALOG, transport=httpx.MockTransport(handler))


async def _no_sleep(*args, **kwargs) -> None:
    """Patched over `utils.retry`'s `asyncio.sleep` so retry-backoff tests
    don't actually wait out the real delay."""


async def test_run_dax_query_executes_and_parses_the_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1.0/myorg/datasets/ds-test/executeQueries"
        assert request.headers["authorization"] == "Bearer tok-123"
        body = json.loads(request.content)
        assert body["queries"][0]["query"].startswith("EVALUATE SUMMARIZECOLUMNS(")
        return httpx.Response(200, json=_EXECUTE_QUERIES_RESPONSE)

    spec = DaxQuerySpec(model_name="Test Model", group_by=[DaxColumn(table="Products", column="Category")])
    dax_query, df = await _rest_client(handler).run_dax_query("tok-123", spec)

    assert dax_query.startswith("EVALUATE SUMMARIZECOLUMNS(")
    assert df.to_dict(orient="records") == [{"Category": "Hardware"}, {"Category": "Premium"}]


async def test_run_dax_query_unknown_model_raises_without_any_http_call():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not have made an HTTP call for an unknown model")

    spec = DaxQuerySpec(model_name="Nonexistent Model", group_by=[DaxColumn(table="Sales", column="Region")])
    with pytest.raises(ValueError, match="Unknown semantic model"):
        await _rest_client(handler).run_dax_query("tok-123", spec)


async def test_run_dax_query_surfaces_power_bi_error_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Column 'Bogus' does not exist")

    spec = DaxQuerySpec(model_name="Test Model", group_by=[DaxColumn(table="Sales", column="Bogus")])
    with pytest.raises(ValueError, match="Bogus"):
        await _rest_client(handler).run_dax_query("tok-123", spec)


async def test_run_dax_query_retries_a_transient_connection_error_then_succeeds(monkeypatch):
    """A dropped connection reaching Power BI shouldn't fail the whole
    request on the first blip - see PBIRestClient.run_dax_query's `@retry`."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json=_EXECUTE_QUERIES_RESPONSE)

    monkeypatch.setattr("data_analyst.utils.retry.asyncio.sleep", _no_sleep)
    spec = DaxQuerySpec(model_name="Test Model", group_by=[DaxColumn(table="Products", column="Category")])

    dax_query, df = await _rest_client(handler).run_dax_query("tok-123", spec)

    assert calls["n"] == 3
    assert df.to_dict(orient="records") == [{"Category": "Hardware"}, {"Category": "Premium"}]


async def test_run_dax_query_gives_up_after_max_attempts_on_a_persistent_connection_error(monkeypatch):
    """A real outage - not just a blip - still surfaces a clear error within
    a bounded number of attempts instead of retrying forever."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("boom", request=request)

    monkeypatch.setattr("data_analyst.utils.retry.asyncio.sleep", _no_sleep)
    spec = DaxQuerySpec(model_name="Test Model", group_by=[DaxColumn(table="Products", column="Category")])

    with pytest.raises(httpx.ConnectError):
        await _rest_client(handler).run_dax_query("tok-123", spec)

    assert calls["n"] == 3


async def test_run_dax_query_does_not_retry_a_power_bi_error_response():
    """A query Power BI itself rejects (bad column, etc.) is a real answer,
    not a transient failure - retrying would only repeat it identically."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="Column 'Bogus' does not exist")

    spec = DaxQuerySpec(model_name="Test Model", group_by=[DaxColumn(table="Sales", column="Bogus")])
    with pytest.raises(ValueError, match="Bogus"):
        await _rest_client(handler).run_dax_query("tok-123", spec)

    assert calls["n"] == 1


def test_rest_client_has_no_trigger_refresh():
    assert not hasattr(PBIRestClient(), "trigger_refresh")


def test_rest_client_has_no_workspace_or_refresh_history_listing():
    # Not needed for this build - see PBIRestClient's module docstring.
    client = PBIRestClient()
    assert not hasattr(client, "list_workspaces")
    assert not hasattr(client, "get_refresh_history")


async def test_get_semantic_metadata_unknown_model_raises_without_any_mcp_call(monkeypatch):
    def _unreachable(*args, **kwargs):
        raise AssertionError("should not have connected to the MCP server for an unknown model")

    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.streamablehttp_client", _unreachable)

    with pytest.raises(ValueError, match="Unknown semantic model"):
        await PBIMcpClient(catalog=_CATALOG).get_semantic_metadata("tok-123", "Nonexistent Model")


def test_mcp_client_has_no_query_execution():
    assert not hasattr(PBIMcpClient(), "run_dax_query")


async def test_get_semantic_metadata_calls_get_semantic_metadata_tool_and_parses_json(monkeypatch):
    fake_session = _FakeSession(_FakeToolResult(json.dumps({"tables": [{"name": "Products"}]})))
    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.streamablehttp_client", _fake_streamablehttp_client)
    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.ClientSession", lambda read, write: fake_session)

    result = await PBIMcpClient(catalog=_CATALOG).get_semantic_metadata("tok-123", "Test Model")

    assert result == {"tables": [{"name": "Products"}]}
    tool_name, args = fake_session.calls[0]
    assert tool_name == "GetSemanticMetadata"
    assert args == {"datasetId": "ds-test"}


async def test_get_semantic_metadata_raises_on_tool_error(monkeypatch):
    fake_session = _FakeSession(_FakeToolResult("permission denied", is_error=True))
    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.streamablehttp_client", _fake_streamablehttp_client)
    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.ClientSession", lambda read, write: fake_session)

    with pytest.raises(ValueError, match="permission denied"):
        await PBIMcpClient(catalog=_CATALOG).get_semantic_metadata("tok-123", "Test Model")


async def test_get_semantic_metadata_retries_a_transient_connection_error_then_succeeds(monkeypatch):
    """A dropped connection to the MCP server surfaces as an ExceptionGroup
    (its transport runs in an anyio task group, see `_describe()` in
    agents/datasource/nodes.py) - shouldn't fail the whole request on the
    first blip, see PBIMcpClient.get_semantic_metadata's `@retry`."""
    calls = {"n": 0}

    @asynccontextmanager
    async def _flaky_streamablehttp_client(url, headers=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ExceptionGroup("unhandled errors in a TaskGroup", [ConnectionError("boom")])
        yield (None, None, None)

    fake_session = _FakeSession(_FakeToolResult(json.dumps({"tables": []})))
    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.streamablehttp_client", _flaky_streamablehttp_client)
    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.ClientSession", lambda read, write: fake_session)
    monkeypatch.setattr("data_analyst.utils.retry.asyncio.sleep", _no_sleep)

    result = await PBIMcpClient(catalog=_CATALOG).get_semantic_metadata("tok-123", "Test Model")

    assert result == {"tables": []}
    assert calls["n"] == 3


async def test_get_semantic_metadata_gives_up_after_max_attempts_on_a_persistent_connection_error(monkeypatch):
    calls = {"n": 0}

    @asynccontextmanager
    async def _always_failing_streamablehttp_client(url, headers=None):
        calls["n"] += 1
        raise ExceptionGroup("unhandled errors in a TaskGroup", [ConnectionError("boom")])
        yield  # pragma: no cover - unreachable; keeps this an async generator

    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.streamablehttp_client", _always_failing_streamablehttp_client)
    monkeypatch.setattr("data_analyst.utils.retry.asyncio.sleep", _no_sleep)

    with pytest.raises(ExceptionGroup):
        await PBIMcpClient(catalog=_CATALOG).get_semantic_metadata("tok-123", "Test Model")

    assert calls["n"] == 3


async def test_get_semantic_metadata_does_not_retry_a_tool_error(monkeypatch):
    """A tool call the server itself rejects is a real answer, not a
    transient failure - retrying would only repeat it identically."""
    calls = {"n": 0}
    fake_session = _FakeSession(_FakeToolResult("permission denied", is_error=True))

    @asynccontextmanager
    async def _counting_streamablehttp_client(url, headers=None):
        calls["n"] += 1
        yield (None, None, None)

    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.streamablehttp_client", _counting_streamablehttp_client)
    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.ClientSession", lambda read, write: fake_session)

    with pytest.raises(ValueError, match="permission denied"):
        await PBIMcpClient(catalog=_CATALOG).get_semantic_metadata("tok-123", "Test Model")

    assert calls["n"] == 1


async def test_get_semantic_metadata_adapts_to_a_differently_named_server_tool(monkeypatch):
    """The tool's exact machine name and dataset-id argument key aren't
    hardcoded - Microsoft's own docs disagree on the name across pages -
    so this should work against whatever the live server actually
    advertises via list_tools(), not just the one name/key we've seen."""
    tools = [_FakeTool("Get Semantic Model Schema", {"semanticModelId": {"type": "string"}})]
    fake_session = _FakeSession(_FakeToolResult(json.dumps({"tables": []})), tools=tools)
    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.streamablehttp_client", _fake_streamablehttp_client)
    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.ClientSession", lambda read, write: fake_session)

    result = await PBIMcpClient(catalog=_CATALOG).get_semantic_metadata("tok-123", "Test Model")

    assert result == {"tables": []}
    tool_name, args = fake_session.calls[0]
    assert tool_name == "Get Semantic Model Schema"
    assert args == {"semanticModelId": "ds-test"}


async def test_get_semantic_metadata_recognizes_the_artifact_id_argument(monkeypatch):
    """Live regression: Microsoft's Fabric MCP server names the tool
    "GetSemanticModelSchema" with a single argument, "artifactId" (Fabric's
    term for a workspace item), which the original keyword list ("dataset",
    "model", "semantic") didn't recognize - the client fell back to a
    hardcoded "datasetId" key the live server rejected."""
    tools = [_FakeTool("GetSemanticModelSchema", {"artifactId": {"type": "string"}})]
    fake_session = _FakeSession(_FakeToolResult(json.dumps({"tables": []})), tools=tools)
    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.streamablehttp_client", _fake_streamablehttp_client)
    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.ClientSession", lambda read, write: fake_session)

    result = await PBIMcpClient(catalog=_CATALOG).get_semantic_metadata("tok-123", "Test Model")

    assert result == {"tables": []}
    tool_name, args = fake_session.calls[0]
    assert tool_name == "GetSemanticModelSchema"
    assert args == {"artifactId": "ds-test"}


async def test_get_semantic_metadata_uses_the_sole_argument_when_its_name_is_unrecognized(monkeypatch):
    """Belt-and-suspenders for a name we haven't seen yet: with only one
    argument in the schema, it's the only place the dataset id could go
    regardless of what it's called."""
    tools = [_FakeTool("GetSemanticModelSchema", {"itemId": {"type": "string"}})]
    fake_session = _FakeSession(_FakeToolResult(json.dumps({"tables": []})), tools=tools)
    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.streamablehttp_client", _fake_streamablehttp_client)
    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.ClientSession", lambda read, write: fake_session)

    result = await PBIMcpClient(catalog=_CATALOG).get_semantic_metadata("tok-123", "Test Model")

    assert result == {"tables": []}
    tool_name, args = fake_session.calls[0]
    assert tool_name == "GetSemanticModelSchema"
    assert args == {"itemId": "ds-test"}


async def test_get_semantic_metadata_raises_clearly_when_no_matching_tool_is_advertised(monkeypatch):
    tools = [_FakeTool("SomeUnrelatedTool", {})]
    fake_session = _FakeSession(_FakeToolResult("unused"), tools=tools)
    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.streamablehttp_client", _fake_streamablehttp_client)
    monkeypatch.setattr("data_analyst.clients.powerbi.mcp.ClientSession", lambda read, write: fake_session)

    with pytest.raises(ValueError, match="No semantic-metadata tool"):
        await PBIMcpClient(catalog=_CATALOG).get_semantic_metadata("tok-123", "Test Model")


def test_metadata_cache_remembers_by_model_name():
    cache = get_metadata_cache("sess-cache-1")
    assert cache.get("Sales Analytics") is None

    cache.remember("Sales Analytics", {"tables": ["Sales"]})

    assert cache.get("Sales Analytics") == {"tables": ["Sales"]}
    assert cache.get("Other Model") is None


def test_metadata_cache_is_scoped_per_session():
    get_metadata_cache("sess-cache-owner").remember("Sales Analytics", {"tables": ["Sales"]})

    assert get_metadata_cache("sess-cache-other").get("Sales Analytics") is None
    # Same session id returns the same cache instance, so what was
    # remembered is still there on a later call.
    assert get_metadata_cache("sess-cache-owner").get("Sales Analytics") == {"tables": ["Sales"]}
