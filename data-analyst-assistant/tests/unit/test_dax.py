import io

import pyarrow as pa
import pytest

from data_analyst.clients.powerbi.dax import (
    DaxColumn,
    DaxFilter,
    DaxMeasure,
    DaxQuerySpec,
    build_dax_query,
    parse_arrow_query_response,
    validate_dax_query,
)


def _arrow_bytes(rows: list[dict], schema: pa.Schema | None = None, is_error: bool = False) -> bytes:
    """Build an in-memory Apache Arrow IPC stream matching what the Execute
    DAX Queries (Arrow) endpoint returns, for tests with no live Power BI
    tenant to call against. `schema` is required when `rows` is empty (there's
    nothing to infer column types/names from otherwise)."""
    table = pa.Table.from_pylist(rows, schema=schema)
    table = table.replace_schema_metadata({b"IsError": b"true" if is_error else b"false"})
    buf = io.BytesIO()
    with pa.ipc.new_stream(buf, table.schema) as writer:
        writer.write_table(table)
    return buf.getvalue()


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
    """A group-by column and an aggregated measure column can belong to
    different, related tables - e.g. grouping by a dimension table's
    column while summing a fact table's column - so each must be
    independently table-qualified rather than sharing one table for the
    whole query."""
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
    (Power BI's executeQueries rejects wrapping one as if it were a raw
    column)."""
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


def test_parse_arrow_query_response_renames_group_by_and_measures():
    spec = DaxQuerySpec(
        model_name="m",
        group_by=[DaxColumn(table="Sales", column="Region")],
        measures=[DaxMeasure(name="Total Revenue", aggregation="SUM", table="Sales", column="Revenue")],
    )
    content = _arrow_bytes(
        [
            {"Sales[Region]": "North", "Total Revenue": 150},
            {"Sales[Region]": "South", "Total Revenue": 30},
        ]
    )
    df = parse_arrow_query_response(content, spec)
    assert list(df.columns) == ["Region", "Total Revenue"]
    assert df.to_dict(orient="records") == [
        {"Region": "North", "Total Revenue": 150},
        {"Region": "South", "Total Revenue": 30},
    ]


def test_parse_arrow_query_response_handles_quoted_table_column_headers():
    spec = DaxQuerySpec(model_name="m", group_by=[DaxColumn(table="Sales", column="Region")])
    content = _arrow_bytes([{"'Sales'[Region]": "North"}])
    df = parse_arrow_query_response(content, spec)
    assert df.to_dict(orient="records") == [{"Region": "North"}]


def test_parse_arrow_query_response_empty_rows_returns_empty_frame_with_expected_columns():
    spec = DaxQuerySpec(
        model_name="m",
        group_by=[DaxColumn(table="Sales", column="Region")],
        measures=[DaxMeasure(name="Total", aggregation="SUM", table="Sales", column="Revenue")],
    )
    schema = pa.schema([("Sales[Region]", pa.string()), ("Total", pa.float64())])
    content = _arrow_bytes([], schema=schema)
    df = parse_arrow_query_response(content, spec)
    assert list(df.columns) == ["Region", "Total"]
    assert len(df) == 0


def test_parse_arrow_query_response_unexpected_bytes_raises():
    spec = DaxQuerySpec(model_name="m", group_by=[DaxColumn(table="Sales", column="Region")])
    with pytest.raises(ValueError, match="Unexpected Arrow executeQueries response"):
        parse_arrow_query_response(b"not a valid arrow stream", spec)


def test_parse_arrow_query_response_raises_on_error_rowset():
    """A query error comes back as HTTP 200 with an error rowset embedded in
    the Arrow stream (an `IsError` schema metadata flag), not an HTTP error
    status - the parser has to surface that itself."""
    spec = DaxQuerySpec(model_name="m", group_by=[DaxColumn(table="Sales", column="Region")])
    content = _arrow_bytes([{"ErrorMessage": "Column 'Bogus' does not exist"}], is_error=True)
    with pytest.raises(ValueError, match="Column 'Bogus' does not exist"):
        parse_arrow_query_response(content, spec)


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
