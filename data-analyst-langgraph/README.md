# Data Analyst Assistant — LangChain/LangGraph version

A simplified rebuild of a data-analyst agent (Power BI + code execution)
using only **LangChain** (tools, chat model) and **LangGraph** (the agent
loop, memory, human-in-the-loop), instead of the Azure AI SDK's Agents
Service. The three external integrations — Power BI MCP, Power BI REST API,
and a Python code-execution sandbox — are mocked so the whole thing runs
with no cloud dependency.

## Run it

```bash
pip install -r requirements.txt

# no API key, no network — walks a scripted tool-calling sequence
python run_cli.py --demo

# with a real model (needs ANTHROPIC_API_KEY or OPENAI_API_KEY)
python run_cli.py --model anthropic:claude-sonnet-4-5 "How did North region do?"
```

`--demo` mode auto-runs the same graph with a scripted fake chat model, so
you can see the full loop — including the human-approval pause — without
any credentials.

## Layout

```
app/
  mock_data.py   fake PBI workspaces / semantic model / underlying tables
  tools.py       the three mocked tool groups (PBI MCP, PBI REST, sandbox)
  graph.py       the LangGraph StateGraph: agent node + tool node + loop
run_cli.py       CLI: real or scripted model, interrupt/resume handling
```

## How the graph works

```
START -> agent -> (tool_calls?) -> tools -> agent -> ... -> END
```

- **`agent` node**: the chat model, bound to all tools via `bind_tools`,
  invoked with a system prompt + the running message list.
- **`tools` node**: LangGraph's prebuilt `ToolNode`, which executes whatever
  tool calls the model just made and appends `ToolMessage`s.
- **`tools_condition`**: prebuilt conditional edge — routes to `tools` if the
  last AI message has tool calls, otherwise to `END`.
- **Checkpointer** (`InMemorySaver`): gives each `thread_id` durable,
  resumable state — this is what a "thread" is in the Azure AI SDK, made
  explicit and swappable (Postgres/Redis-backed checkpointers exist for
  production).
- **`interrupt()`** inside `pbi_rest_trigger_dataset_refresh`: pauses the
  graph before a *mutating* action and waits for a human decision, resumed
  via `Command(resume=approved)`. This is the one part of the demo that
  doesn't have a clean one-liner equivalent in the Azure AI SDK — it's a
  first-class LangGraph primitive.

## Mapping from the Azure AI SDK version

| Azure AI SDK concept | LangChain/LangGraph equivalent |
|---|---|
| Agent + function tools (`FunctionTool`, `ToolSet`) | Python functions decorated with `@tool` (`langchain_core.tools`), collected into a list and passed to `bind_tools` |
| `AgentsClient.create_agent(model=..., instructions=..., tools=...)` | `init_chat_model(model)` + a system prompt string + `llm.bind_tools(tools)` — no server-side agent resource, it's just local objects |
| Threads / runs (`client.agents.threads`, `client.agents.runs`) | A `StateGraph` compiled with a checkpointer; a `thread_id` in `config["configurable"]` scopes persisted state — same idea, but you own the graph shape instead of a fixed run loop |
| Polling a run for `requires_action` / submitting tool outputs | The graph *is* the loop: `agent -> tools -> agent` via `tools_condition`, no polling required — `graph.invoke` just runs until it hits `END` or an `interrupt()` |
| Custom code for "pause and ask a human before doing X" | `langgraph.types.interrupt()` inside the tool function itself, resumed with `Command(resume=...)` |
| MCP tool integration via Azure AI Foundry's MCP tool type | Same MCP server could be wrapped with `langchain-mcp-adapters` to auto-generate `@tool`-compatible functions from the server's tool list; here it's mocked directly as plain functions |
| Streaming agent updates via SDK event handlers | `graph.stream(...)` / `graph.astream(...)`, or `stream_mode="values"` for incremental state, or `"updates"` for per-node diffs |

The biggest structural difference: Azure AI Agents Service treats the agent
as a managed server-side resource you configure and poll. LangGraph makes
the control flow a graph you define yourself — more code up front, but the
loop, memory, and interrupt points are all explicit and inspectable rather
than living inside a managed run object.

## Where this is simplified vs. the real assistant

- **PBI MCP**: real DAX execution and MCP transport are replaced by
  `mock_data.guess_table_from_dax`, which just pattern-matches a table name
  out of the query text.
- **PBI REST**: workspace/dataset/refresh-history calls return canned data;
  only the refresh trigger goes through the approval `interrupt()`.
- **Python sandbox**: `python_sandbox_execute` really does `exec()` pandas
  code (so the numbers you see are genuinely computed), but with a
  minimal restricted namespace instead of a network-isolated worker process
  — do not point this at untrusted input as-is.
