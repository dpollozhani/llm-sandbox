"""CLI entry point for the simplified data analyst assistant.

Usage:
    python run_cli.py --demo                      # scripted fake model, no API key/network needed
    python run_cli.py                              # real LLM via init_chat_model (needs ANTHROPIC_API_KEY)
    python run_cli.py --model openai:gpt-4o-mini "How did North region do?"
"""
from __future__ import annotations

import argparse
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from app.graph import build_graph

DEFAULT_QUESTION = (
    "What was total revenue by region, and please refresh the Sales Analytics "
    "dataset afterwards?"
)


def build_demo_model():
    """A scripted model that walks through the same tool-calling loop a real
    LLM would, so the whole graph (including the human-in-the-loop approval)
    can be exercised with no API key and no network access."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class FakeToolCallingChatModel(FakeMessagesListChatModel):
        """FakeMessagesListChatModel ignores tool schemas anyway, so `bind_tools`
        just needs to be a no-op instead of raising NotImplementedError."""

        def bind_tools(self, tools, **kwargs):
            return self

    scripted_responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "pbi_mcp_list_semantic_models", "args": {}, "id": "call_1"}],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "pbi_mcp_run_dax_query",
                    "args": {"model_name": "Sales Analytics", "dax_query": "EVALUATE Sales"},
                    "id": "call_2",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "python_sandbox_execute",
                    "args": {
                        "code": "result = df.groupby('Region')['Revenue'].sum().reset_index()",
                        "sandbox_ref": "df_1",
                    },
                    "id": "call_3",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "pbi_rest_trigger_dataset_refresh",
                    "args": {"dataset_id": "ds-001"},
                    "id": "call_4",
                }
            ],
        ),
        AIMessage(
            content=(
                "Revenue by region: North 6,450, South 5,700, East 2,700, West 3,375. "
                "I also kicked off a refresh of the Sales Analytics dataset as requested."
            )
        ),
    ]
    return FakeToolCallingChatModel(responses=scripted_responses)


def run(question: str, model: str, demo: bool, auto_approve: bool) -> None:
    llm = build_demo_model() if demo else model
    graph = build_graph(llm)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    state = {"messages": [HumanMessage(content=question)]}
    result = graph.invoke(state, config=config)

    while result.get("__interrupt__"):
        payload = result["__interrupt__"][0].value
        print(f"\n[approval requested] {payload['message']}")
        if auto_approve:
            approved = True
            print("(auto-approved via --auto-approve)")
        else:
            approved = input("Approve? [y/N] ").strip().lower() == "y"
        result = graph.invoke(Command(resume=approved), config=config)

    for message in result["messages"]:
        role = getattr(message, "type", "?")
        content = getattr(message, "content", "")
        if content:
            print(f"\n[{role}] {content}")
        for call in getattr(message, "tool_calls", None) or []:
            print(f"    -> tool call: {call['name']}({call['args']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--model", default="anthropic:claude-sonnet-4-5", help="init_chat_model provider:model string")
    parser.add_argument("--demo", action="store_true", help="use a scripted fake model instead of a real LLM")
    parser.add_argument("--auto-approve", action="store_true", help="auto-approve human-in-the-loop tool calls")
    args = parser.parse_args()
    run(args.question, args.model, args.demo, args.auto_approve)


if __name__ == "__main__":
    main()
