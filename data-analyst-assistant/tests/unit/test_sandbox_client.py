import pandas as pd

from data_analyst.clients.sandbox.client import SandboxClient, get_sandbox_client


async def test_stage_and_execute_roundtrip():
    client = SandboxClient()
    df = pd.DataFrame([{"Region": "North", "Revenue": 10.0}, {"Region": "North", "Revenue": 5.0}])
    dataset_id = client.stage(df)

    result = await client.execute("result = df.groupby('Region')['Revenue'].sum().reset_index()", dataset_id=dataset_id)

    assert result.error is None
    assert result.result == [{"Region": "North", "Revenue": 15.0}]


async def test_execute_unknown_dataset_id_returns_error_not_raise():
    client = SandboxClient()
    result = await client.execute("result = 1", dataset_id="does-not-exist")
    assert result.error is not None


async def test_execute_bad_code_captures_error():
    client = SandboxClient()
    result = await client.execute("result = 1 / 0")
    assert result.error is not None
    assert "division" in result.error.lower() or "zero" in result.error.lower()


def test_query_cache_roundtrip():
    client = SandboxClient()
    assert client.find_cached("some-key") is None

    dataset_id = client.stage(pd.DataFrame([{"a": 1}]))
    client.remember("some-key", dataset_id)

    assert client.find_cached("some-key") == dataset_id
    assert client.peek(dataset_id) is not None
    assert client.peek("unknown-id") is None


def test_get_sandbox_client_is_session_scoped():
    a1 = get_sandbox_client("session-a")
    a2 = get_sandbox_client("session-a")
    b = get_sandbox_client("session-b")

    assert a1 is a2
    assert a1 is not b
