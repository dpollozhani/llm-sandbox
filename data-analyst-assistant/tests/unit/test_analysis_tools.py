from data_analyst.agents.analysis.nodes import TOOLS


def test_analysis_tools_include_ambiguity_flag():
    tool_names = {t.name for t in TOOLS}
    assert tool_names == {"python_sandbox_execute", "flag_ambiguity", "suggest_followup"}
