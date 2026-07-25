import pytest

from data_analyst.clients.powerbi.mcp import PBIMcpClient
from data_analyst.clients.powerbi.rest import PBIRestClient


def test_list_semantic_models_reads_catalog():
    models = PBIMcpClient().list_semantic_models()
    assert {"model_name": "Sales Analytics", "dataset_id": "ds-001", "tables": ["Sales", "Products", "Regions"]} in models


def test_run_dax_query_matches_table_from_query_text():
    df = PBIMcpClient().run_dax_query("Sales Analytics", "EVALUATE Products")
    assert "Category" in df.columns


def test_run_dax_query_unknown_model_raises():
    with pytest.raises(ValueError):
        PBIMcpClient().run_dax_query("Nonexistent Model", "EVALUATE Sales")


def test_get_refresh_history_is_read_only():
    rest = PBIRestClient()
    assert not hasattr(rest, "trigger_refresh")
    history = rest.get_refresh_history("ds-001")
    assert history[0]["status"] == "Completed"
