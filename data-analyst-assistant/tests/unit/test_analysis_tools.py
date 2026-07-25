from data_analyst.agents.analysis.nodes import TOOLS


def test_analysis_tools_include_clarification():
    tool_names = {t.name for t in TOOLS}
    assert tool_names == {"python_sandbox_execute", "request_clarification"}
