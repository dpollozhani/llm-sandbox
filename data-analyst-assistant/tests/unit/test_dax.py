import pytest

from data_analyst.clients.powerbi.dax import (
    DaxColumn,
    DaxFilter,
    DaxMeasure,
    DaxQuerySpec,
    build_dax_query,
    parse_execute_queries_response,
    validate_dax_query,
)


def test_build_dax_query_shape():
    spec = DaxQuerySpec(
        model_name="m",
        group_by=[DaxColumn(table="Sales", column="Region")],
        filters=[DaxFilter(table="Sales", column="Region", operator="!=", value="South")],
        measures=[DaxMeasure(name="Total Revenue", aggregation="SUM", table="Sales", column="Revenue")],
    )
    dax = build_dax_query(spec)
    assert dax.startswith("EVALUATE SUMMARIZECOLUMNS(")
    assert dax.endswith(")")
    assert "'Sales'[Region]" in dax
    assert 'FILTER(ALL(\'Sales\'), \'Sales\'[Region] != "South")' in dax
    assert '"Total Revenue", SUM(\'Sales\'[Revenue])' in dax


def test_build_dax_query_mixes_columns_from_different_related_tables():
    """The real bug this guards against: a group-by column and an
    aggregated measure column can (and often do) belong to different,
    related tables in a star-schema model - e.g. grouping by a dimension
    table's column while summing a fact table's column. An earlier design
    forced every column onto one spec-level table, which had no correct way
    to express this and produced invalid, doubly-qualified references like
    'Facts'[dimItemMaster[Article key]] in production."""
    spec = DaxQuerySpec(
        model_name="m",
        group_by=[DaxColumn(table="dimItemMaster", column="BRIC")],
        measures=[DaxMeasure(name="On-hand sum", aggregation="SUM", table="Facts", column="Actual stock quantity")],
    )
    dax = build_dax_query(spec)
    assert "'dimItemMaster'[BRIC]" in dax
    assert '"On-hand sum", SUM(\'Facts\'[Actual stock quantity])' in dax


def test_build_dax_query_references_an_existing_model_measure_directly():
    """A measure with no `aggregation` is a reference to a measure that
    already exists in the model (e.g. under a "_Measures" table) - it's
    addressed directly by name, never wrapped in an aggregation function
    (wrapping it, as if it were a raw column, is what Power BI's
    executeQueries rejected with a 400 in production)."""
    spec = DaxQuerySpec(
        model_name="m",
        group_by=[DaxColumn(table="Sales", column="Region")],
        measures=[DaxMeasure(name="Inventory on-hand")],
    )
    dax = build_dax_query(spec)
    assert '"Inventory on-hand", [Inventory on-hand]' in dax


def test_measure_with_aggregation_requires_table_and_column():
    with pytest.raises(ValueError, match="table"):
        DaxMeasure(name="Total Revenue", aggregation="SUM")
    with pytest.raises(ValueError, match="column"):
        DaxMeasure(name="Total Revenue", aggregation="SUM", table="Sales")


def test_build_dax_query_does_not_double_bracket_an_already_bracketed_measure_name():
    """If the caller (or model) already wrapped the measure name in
    brackets, e.g. "[Inventory on-hand]", the reference must still come
    out as a single [Inventory on-hand] - not [[Inventory on-hand]], which
    Power BI would reject."""
    spec = DaxQuerySpec(
        model_name="m",
        group_by=[DaxColumn(table="Sales", column="Region")],
        measures=[DaxMeasure(name="[Inventory on-hand]")],
    )
    dax = build_dax_query(spec)
    assert '"[Inventory on-hand]", [Inventory on-hand]' in dax
    assert "[[Inventory on-hand]]" not in dax


def test_validate_rejects_non_summarizecolumns_text():
    spec = DaxQuerySpec(model_name="m", group_by=[DaxColumn(table="Sales", column="Region")])
    with pytest.raises(ValueError, match="SUMMARIZECOLUMNS"):
        validate_dax_query("EVALUATE Sales", spec)


def test_validate_rejects_empty_selection():
    spec = DaxQuerySpec(model_name="m")
    dax = build_dax_query(spec)
    with pytest.raises(ValueError, match="at least one"):
        validate_dax_query(dax, spec)


def test_validate_accepts_a_well_formed_query():
    spec = DaxQuerySpec(model_name="m", group_by=[DaxColumn(table="Sales", column="Region")])
    dax = build_dax_query(spec)
    validate_dax_query(dax, spec)  # doesn't raise


def test_build_dax_query_uses_row_for_a_grand_total_with_no_group_by():
    """SUMMARIZECOLUMNS requires at least one group-by column - it has no
    "just give me the totals" mode - so a spec with only measures (e.g. "total
    inventory on-hand across everything", the exact case that 400'd in
    production) must use EVALUATE ROW(...) instead, DAX's own idiom for a
    single ungrouped row of measures."""
    spec = DaxQuerySpec(model_name="m", measures=[DaxMeasure(name="Inventory on-hand")])
    dax = build_dax_query(spec)
    assert dax.startswith("EVALUATE ROW(")
    assert "SUMMARIZECOLUMNS" not in dax
    assert '"Inventory on-hand", [Inventory on-hand]' in dax
    validate_dax_query(dax, spec)  # doesn't raise


def test_build_dax_query_wraps_filtered_grand_total_measures_in_calculate():
    """ROW() has no table-filter arguments like SUMMARIZECOLUMNS does, so a
    filtered grand total (no group_by, but a filter) has to fold the filter
    into each measure's own expression via CALCULATE instead."""
    spec = DaxQuerySpec(
        model_name="m",
        filters=[DaxFilter(table="Sales", column="Region", operator="=", value="North")],
        measures=[DaxMeasure(name="Total Revenue", aggregation="SUM", table="Sales", column="Revenue")],
    )
    dax = build_dax_query(spec)
    assert dax.startswith("EVALUATE ROW(")
    assert '"Total Revenue", CALCULATE(SUM(\'Sales\'[Revenue]), FILTER(ALL(\'Sales\'), \'Sales\'[Region] = "North"))' in dax
    validate_dax_query(dax, spec)  # doesn't raise


def test_validate_rejects_row_text_when_group_by_is_present():
    spec = DaxQuerySpec(model_name="m", group_by=[DaxColumn(table="Sales", column="Region")])
    with pytest.raises(ValueError, match="SUMMARIZECOLUMNS"):
        validate_dax_query("EVALUATE ROW(\n    \"x\", 1\n)", spec)


def test_validate_rejects_summarizecolumns_text_when_group_by_is_absent():
    spec = DaxQuerySpec(model_name="m", measures=[DaxMeasure(name="Total")])
    with pytest.raises(ValueError, match="ROW"):
        validate_dax_query('EVALUATE SUMMARIZECOLUMNS(\n    "Total", [Total]\n)', spec)


def test_parse_execute_queries_response_renames_group_by_and_measures():
    spec = DaxQuerySpec(
        model_name="m",
        group_by=[DaxColumn(table="Sales", column="Region")],
        measures=[DaxMeasure(name="Total Revenue", aggregation="SUM", table="Sales", column="Revenue")],
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
    spec = DaxQuerySpec(model_name="m", group_by=[DaxColumn(table="Sales", column="Region")])
    response = {"results": [{"tables": [{"rows": [{"'Sales'[Region]": "North"}]}]}]}
    df = parse_execute_queries_response(response, spec)
    assert df.to_dict(orient="records") == [{"Region": "North"}]


def test_parse_execute_queries_response_empty_rows_returns_empty_frame_with_expected_columns():
    spec = DaxQuerySpec(
        model_name="m",
        group_by=[DaxColumn(table="Sales", column="Region")],
        measures=[DaxMeasure(name="Total", aggregation="SUM", table="Sales", column="Revenue")],
    )
    response = {"results": [{"tables": [{"rows": []}]}]}
    df = parse_execute_queries_response(response, spec)
    assert list(df.columns) == ["Region", "Total"]
    assert len(df) == 0


def test_parse_execute_queries_response_unexpected_shape_raises():
    spec = DaxQuerySpec(model_name="m", group_by=[DaxColumn(table="Sales", column="Region")])
    with pytest.raises(ValueError, match="Unexpected executeQueries response"):
        parse_execute_queries_response({"unexpected": True}, spec)


def test_cache_key_ignores_list_order():
    a = DaxQuerySpec(
        model_name="m",
        group_by=[DaxColumn(table="Sales", column="Region"), DaxColumn(table="Sales", column="Product")],
        measures=[DaxMeasure(name="Total", aggregation="SUM", table="Sales", column="Revenue")],
    )
    b = DaxQuerySpec(
        model_name="m",
        group_by=[DaxColumn(table="Sales", column="Product"), DaxColumn(table="Sales", column="Region")],
        measures=[DaxMeasure(name="Total", aggregation="SUM", table="Sales", column="Revenue")],
    )
    assert a.cache_key() == b.cache_key()


def test_cache_key_differs_for_different_specs():
    a = DaxQuerySpec(model_name="m", group_by=[DaxColumn(table="Sales", column="Region")])
    b = DaxQuerySpec(model_name="m", group_by=[DaxColumn(table="Sales", column="Product")])
    assert a.cache_key() != b.cache_key()
