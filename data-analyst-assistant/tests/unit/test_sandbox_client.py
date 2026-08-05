import pandas as pd
import pytest

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


async def test_execute_print_is_captured_as_stdout():
    """Regression: `print` was missing from the sandbox's restricted
    builtins even though the tool's own docstring always claimed printed
    output is captured - any code reaching for it failed outright."""
    client = SandboxClient()
    result = await client.execute("print('hello'); result = 1")
    assert result.error is None
    assert "hello" in result.stdout


async def test_execute_common_builtins_are_available():
    client = SandboxClient()
    result = await client.execute("result = sorted([3, 1, 2]) + [min(1, 2), max(1, 2), abs(-5), str(1), int('2')]")
    assert result.error is None
    assert result.result == [1, 2, 3, 1, 2, 5, "1", 2]


async def test_execute_helper_function_sees_top_level_variables():
    """Regression: exec() with two separate globals/locals dicts runs
    top-level code like a class body - a def (or comprehension, a hidden
    nested function) created there gets __globals__ set to the globals
    dict, not locals, so it couldn't see a variable a top-level assignment
    put in locals. This broke exactly the kind of code a real analysis
    script writes: a helper function, called later via groupby().apply(),
    referencing a variable assigned earlier in the very same script."""
    client = SandboxClient()
    code = (
        "threshold = 2\n"
        "def above(row):\n"
        "    return row['x'] > threshold\n"
        "result = df[df.apply(above, axis=1)].to_dict(orient='records')"
    )
    df = pd.DataFrame([{"x": 1}, {"x": 3}])
    result = await client.execute(code, dataset_id=client.stage(df))

    assert result.error is None
    assert result.result == [{"x": 3}]


async def test_execute_cannot_import_additional_modules():
    """The sandbox pre-imports pd/np/math/stats - anything else is
    unreachable, including via the model's own `import` statement."""
    client = SandboxClient()
    result = await client.execute("import os\nresult = 1")
    assert result.error is not None


async def test_execute_can_use_numpy_math_and_scipy_stats():
    client = SandboxClient()
    df = pd.DataFrame([{"x": 1.0}, {"x": 2.0}, {"x": 3.0}, {"x": 4.0}])
    result = await client.execute(
        "result = {'mean': np.mean(df['x']), 'sqrt_sum': math.sqrt(df['x'].sum()), "
        "'zscores': list(stats.zscore(df['x']))}",
        dataset_id=client.stage(df),
    )

    assert result.error is None
    assert result.result["mean"] == 2.5
    assert result.result["sqrt_sum"] == pytest.approx(3.1622776601683795)
    assert len(result.result["zscores"]) == 4


async def test_execute_normalizes_numpy_scalar_and_array_results():
    """numpy/scipy functions commonly hand back bare np scalars/arrays,
    which json.dumps can't serialize on its own - these must come back as
    plain Python types crossing the tool boundary, however deeply nested."""
    client = SandboxClient()
    result = await client.execute("result = {'scalar': np.float64(3.5), 'array': np.array([1, 2, 3])}")

    assert result.error is None
    assert result.result == {"scalar": 3.5, "array": [1, 2, 3]}
    assert isinstance(result.result["scalar"], float)
    assert isinstance(result.result["array"], list)


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
