from data_analyst.agents.datasource.nodes import build_tools


def test_datasource_has_no_mutating_tools():
    tool_names = {t.name for t in build_tools()}
    assert tool_names == {
        "pbi_mcp_get_semantic_metadata",
        "pbi_rest_run_dax_query",
        "request_clarification",
    }
