import pandas as pd

from data_analyst.clients.sandbox.client import SandboxClient


def test_stage_and_execute_roundtrip():
    client = SandboxClient()
    df = pd.DataFrame([{"Region": "North", "Revenue": 10.0}, {"Region": "North", "Revenue": 5.0}])
    ref = client.stage(df)

    result = client.execute("result = df.groupby('Region')['Revenue'].sum().reset_index()", sandbox_ref=ref)

    assert result.error is None
    assert result.result == [{"Region": "North", "Revenue": 15.0}]


def test_execute_unknown_ref_returns_error_not_raise():
    client = SandboxClient()
    result = client.execute("result = 1", sandbox_ref="does-not-exist")
    assert result.error is not None


def test_execute_bad_code_captures_error():
    client = SandboxClient()
    result = client.execute("result = 1 / 0")
    assert result.error is not None
    assert "division" in result.error.lower() or "zero" in result.error.lower()
