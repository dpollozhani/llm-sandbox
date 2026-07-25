"""LangGraph wiring for the simplified data analyst assistant.

Shape: a single agent node bound to all tools, a tool-execution node, and a
conditional loop between them until the model answers without calling a
tool. A checkpointer gives each `thread_id` persistent, resumable state,
which is also what makes the human-in-the-loop approval in
`pbi_rest_trigger_dataset_refresh` possible (the graph can pause mid-run and
be resumed later, even after a process restart if the checkpointer is
swapped for a persistent one).
"""
from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .tools import ALL_TOOLS

SYSTEM_PROMPT = """You are a data analyst assistant for Power BI users.

You can:
- discover semantic models and run DAX queries against them (PBI MCP tools)
- inspect workspaces, datasets, and refresh history, and trigger refreshes (PBI REST tools)
- stage query results into a Python sandbox and run pandas code to analyze them

Always fetch data before analyzing it, use the sandbox for any math or
aggregation instead of computing it yourself, and explain results in plain
language."""


def build_graph(model: BaseChatModel | str = "anthropic:claude-sonnet-4-5") -> CompiledStateGraph:
    llm = init_chat_model(model) if isinstance(model, str) else model
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: MessagesState):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *state["messages"]]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=InMemorySaver())
