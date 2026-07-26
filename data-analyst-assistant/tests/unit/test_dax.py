import pytest

from data_analyst.clients.powerbi.dax import (
    DaxFilter,
    DaxMeasure,
    DaxQuerySpec,
    build_summarizecolumns,
    parse_execute_queries_response,
    validate_dax_query,
)


def test_build_summarizecolumns_shape():
    spec = DaxQuerySpec(
        model_name="m",
        table="Sales",
        group_by=["Region"],
        filters=[DaxFilter(column="Region", operator="!=", value="South")],
        measures=[DaxMeasure(name="Total Revenue", aggregation="SUM", column="Revenue")],
    )
    dax = build_summarizecolumns(spec)
    assert dax.startswith("EVALUATE SUMMARIZECOLUMNS(")
    assert dax.endswith(")")
    assert "'Sales'[Region]" in dax
    assert 'FILTER(ALL(\'Sales\'), \'Sales\'[Region] != "South")' in dax
    assert '"Total Revenue", SUM(\'Sales\'[Revenue])' in dax


def test_build_summarizecolumns_references_an_existing_model_measure_directly():
    """A measure with no `aggregation` is a reference to a measure that
    already exists in the model (e.g. under a "_Measures" table) - it's
    addressed directly by name, never wrapped in an aggregation function
    (wrapping it, as if it were a raw column, is what Power BI's
    executeQueries rejected with a 400 in production)."""
    spec = DaxQuerySpec(
        model_name="m",
        table="Sales",
        group_by=["Region"],
        measures=[DaxMeasure(name="Inventory on-hand")],
    )
    dax = build_summarizecolumns(spec)
    assert '"Inventory on-hand", [Inventory on-hand]' in dax


def test_measure_with_aggregation_requires_column():
    with pytest.raises(ValueError, match="column"):
        DaxMeasure(name="Total Revenue", aggregation="SUM")


def test_build_summarizecolumns_does_not_double_bracket_an_already_bracketed_measure_name():
    """If the caller (or model) already wrapped the measure name in
    brackets, e.g. "[Inventory on-hand]", the reference must still come
    out as a single [Inventory on-hand] - not [[Inventory on-hand]], which
    Power BI would reject."""
    spec = DaxQuerySpec(
        model_name="m",
        table="Sales",
        group_by=["Region"],
        measures=[DaxMeasure(name="[Inventory on-hand]")],
    )
    dax = build_summarizecolumns(spec)
    assert '"[Inventory on-hand]", [Inventory on-hand]' in dax
    assert "[[Inventory on-hand]]" not in dax


def test_validate_rejects_non_summarizecolumns_text():
    spec = DaxQuerySpec(model_name="m", table="Sales", group_by=["Region"])
    with pytest.raises(ValueError, match="SUMMARIZECOLUMNS"):
        validate_dax_query("EVALUATE Sales", spec)


def test_validate_rejects_empty_selection():
    spec = DaxQuerySpec(model_name="m", table="Sales")
    dax = build_summarizecolumns(spec)
    with pytest.raises(ValueError, match="at least one"):
        validate_dax_query(dax, spec)


def test_validate_accepts_a_well_formed_query():
    spec = DaxQuerySpec(model_name="m", table="Sales", group_by=["Region"])
    dax = build_summarizecolumns(spec)
    validate_dax_query(dax, spec)  # doesn't raise


def test_parse_execute_queries_response_renames_group_by_and_measures():
    spec = DaxQuerySpec(
        model_name="m",
        table="Sales",
        group_by=["Region"],
        measures=[DaxMeasure(name="Total Revenue", aggregation="SUM", column="Revenue")],
    )
    response = {
        "results": [
            {
                "tables": [
                    {
                        "rows": [
                            {"Sales[Region]": "North", "Total Revenue": 150},
                            {"Sales[Region]": "South", "Total Revenue": 30},
                        ]
                    }
                ]
            }
        ]
    }
    df = parse_execute_queries_response(response, spec)
    assert list(df.columns) == ["Region", "Total Revenue"]
    assert df.to_dict(orient="records") == [
        {"Region": "North", "Total Revenue": 150},
        {"Region": "South", "Total Revenue": 30},
    ]


def test_parse_execute_queries_response_handles_quoted_table_column_headers():
    spec = DaxQuerySpec(model_name="m", table="Sales", group_by=["Region"])
    response = {"results": [{"tables": [{"rows": [{"'Sales'[Region]": "North"}]}]}]}
    df = parse_execute_queries_response(response, spec)
    assert df.to_dict(orient="records") == [{"Region": "North"}]


def test_parse_execute_queries_response_empty_rows_returns_empty_frame_with_expected_columns():
    spec = DaxQuerySpec(
        model_name="m",
        table="Sales",
        group_by=["Region"],
        measures=[DaxMeasure(name="Total", aggregation="SUM", column="Revenue")],
    )
    response = {"results": [{"tables": [{"rows": []}]}]}
    df = parse_execute_queries_response(response, spec)
    assert list(df.columns) == ["Region", "Total"]
    assert len(df) == 0


def test_parse_execute_queries_response_unexpected_shape_raises():
    spec = DaxQuerySpec(model_name="m", table="Sales", group_by=["Region"])
    with pytest.raises(ValueError, match="Unexpected executeQueries response"):
        parse_execute_queries_response({"unexpected": True}, spec)


def test_cache_key_ignores_list_order():
    a = DaxQuerySpec(
        model_name="m",
        table="Sales",
        group_by=["Region", "Product"],
        measures=[DaxMeasure(name="Total", aggregation="SUM", column="Revenue")],
    )
    b = DaxQuerySpec(
        model_name="m",
        table="Sales",
        group_by=["Product", "Region"],
        measures=[DaxMeasure(name="Total", aggregation="SUM", column="Revenue")],
    )
    assert a.cache_key() == b.cache_key()


def test_cache_key_differs_for_different_specs():
    a = DaxQuerySpec(model_name="m", table="Sales", group_by=["Region"])
    b = DaxQuerySpec(model_name="m", table="Sales", group_by=["Product"])
    assert a.cache_key() != b.cache_key()
