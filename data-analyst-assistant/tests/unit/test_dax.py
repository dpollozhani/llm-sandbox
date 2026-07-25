import pandas as pd
import pytest

from data_analyst.clients.powerbi.dax import (
    DaxFilter,
    DaxMeasure,
    DaxQuerySpec,
    build_summarizecolumns,
    execute_query,
    validate_dax_query,
)

_DF = pd.DataFrame(
    [
        {"Region": "North", "Product": "A", "Revenue": 100},
        {"Region": "North", "Product": "B", "Revenue": 50},
        {"Region": "South", "Product": "A", "Revenue": 30},
    ]
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
    assert dax.startswith("SUMMARIZECOLUMNS(")
    assert dax.endswith(")")
    assert "'Sales'[Region]" in dax
    assert 'FILTER(ALL(\'Sales\'), \'Sales\'[Region] != "South")' in dax
    assert '"Total Revenue", SUM(\'Sales\'[Revenue])' in dax


def test_validate_rejects_non_summarizecolumns_text():
    spec = DaxQuerySpec(model_name="m", table="Sales", group_by=["Region"])
    with pytest.raises(ValueError, match="SUMMARIZECOLUMNS"):
        validate_dax_query("EVALUATE Sales", spec, {"Region"})


def test_validate_rejects_empty_selection():
    spec = DaxQuerySpec(model_name="m", table="Sales")
    dax = build_summarizecolumns(spec)
    with pytest.raises(ValueError, match="at least one"):
        validate_dax_query(dax, spec, {"Region"})


def test_validate_rejects_unknown_column():
    spec = DaxQuerySpec(model_name="m", table="Sales", group_by=["Bogus"])
    dax = build_summarizecolumns(spec)
    with pytest.raises(ValueError, match="Unknown column"):
        validate_dax_query(dax, spec, {"Region"})


def test_execute_query_groups_filters_and_aggregates():
    spec = DaxQuerySpec(
        model_name="m",
        table="Sales",
        group_by=["Region"],
        filters=[DaxFilter(column="Region", operator="!=", value="South")],
        measures=[DaxMeasure(name="Total Revenue", aggregation="SUM", column="Revenue")],
    )
    result = execute_query(_DF, spec)
    assert result.to_dict(orient="records") == [{"Region": "North", "Total Revenue": 150}]


def test_execute_query_no_measures_returns_distinct_group_by():
    spec = DaxQuerySpec(model_name="m", table="Sales", group_by=["Region"])
    result = execute_query(_DF, spec)
    assert sorted(result["Region"].tolist()) == ["North", "South"]


def test_execute_query_no_group_by_returns_single_row_aggregate():
    spec = DaxQuerySpec(model_name="m", table="Sales", measures=[DaxMeasure(name="Total", aggregation="SUM", column="Revenue")])
    result = execute_query(_DF, spec)
    assert result.to_dict(orient="records") == [{"Total": 180}]


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
