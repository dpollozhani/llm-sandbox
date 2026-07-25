import pytest

from data_analyst.clients.powerbi.dax import DaxQuerySpec
from data_analyst.clients.powerbi.mcp import PBIMcpClient
from data_analyst.clients.powerbi.rest import PBIRestClient


async def test_list_semantic_models_reads_catalog():
    models = await PBIMcpClient().list_semantic_models()
    assert {"model_name": "Sales Analytics", "dataset_id": "ds-001", "tables": ["Sales", "Products", "Regions"]} in models


async def test_run_dax_query_returns_dax_text_and_dataframe():
    spec = DaxQuerySpec(model_name="Sales Analytics", table="Products", group_by=["Category"])
    dax_query, df = await PBIRestClient().run_dax_query(spec)
    assert dax_query.startswith("SUMMARIZECOLUMNS(")
    assert "Category" in df.columns


async def test_run_dax_query_unknown_model_raises():
    spec = DaxQuerySpec(model_name="Nonexistent Model", table="Sales", group_by=["Region"])
    with pytest.raises(ValueError):
        await PBIRestClient().run_dax_query(spec)


async def test_run_dax_query_unknown_table_raises():
    spec = DaxQuerySpec(model_name="Sales Analytics", table="Bogus", group_by=["X"])
    with pytest.raises(ValueError):
        await PBIRestClient().run_dax_query(spec)


def test_mcp_client_has_no_query_execution():
    assert not hasattr(PBIMcpClient(), "run_dax_query")


async def test_get_refresh_history_is_read_only():
    rest = PBIRestClient()
    assert not hasattr(rest, "trigger_refresh")
    history = await rest.get_refresh_history("ds-001")
    assert history[0]["status"] == "Completed"
