from data_analyst.agents.datasource.nodes import TOOLS


def test_datasource_has_no_mutating_tools():
    tool_names = {t.name for t in TOOLS}
    assert tool_names == {
        "pbi_mcp_list_semantic_models",
        "pbi_rest_list_workspaces",
        "pbi_rest_get_refresh_history",
        "pbi_rest_run_dax_query",
    }
