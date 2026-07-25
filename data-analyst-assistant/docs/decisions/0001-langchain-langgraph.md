# ADR 0001: Rebuild the assistant on LangChain/LangGraph instead of the Azure AI SDK

## Context

The original assistant was built on the Azure AI SDK's Agents Service
(`AgentsClient`, threads, runs, function tools). This project rebuilds a
simplified version of the same assistant (Power BI + a code sandbox) using
only LangChain (tools, chat model abstraction) and LangGraph (the agent
loop, memory, human-in-the-loop), to evaluate what that stack looks like in
practice.

## Decision

Use LangGraph's multi-agent supervisor pattern: an `orchestrator` graph
routes each turn to a `datasource` or `analysis` specialist subgraph (or
responds directly), each specialist being its own small ReAct-style loop
(agent node bound to a scoped tool set + `ToolNode`, looping via
`tools_condition`). See `docs/architecture.md` for the full shape.

## Mapping from the Azure AI SDK version

| Azure AI SDK concept | LangChain/LangGraph equivalent |
|---|---|
| Agent + function tools (`FunctionTool`, `ToolSet`) | Python functions decorated with `@tool` (`langchain_core.tools`), collected into a list and passed to `bind_tools` |
| `AgentsClient.create_agent(model=..., instructions=..., tools=...)` | `init_chat_model(...)` / a provider-specific client (`AzureChatOpenAI`) + a system prompt string + `llm.bind_tools(tools)` - no server-side agent resource, it's local objects built by `clients/llm/factory.py` |
| Threads / runs (`client.agents.threads`, `client.agents.runs`) | A `StateGraph` compiled with a checkpointer; a `thread_id` in `config["configurable"]` scopes persisted state - same idea, but the graph shape is yours to define instead of a fixed run loop |
| Multiple specialized agents handed off via the Assistants/Agents "handoff" pattern | LangGraph subgraphs added as nodes in a supervisor `StateGraph` - each specialist is a fully independent compiled graph, composed rather than orchestrated by a special SDK feature |
| Polling a run for `requires_action` / submitting tool outputs | The graph *is* the loop: `agent -> tools -> agent` via `tools_condition`, no polling required - `graph.invoke` just runs until it hits `END` or an `interrupt()` |
| Custom code for "pause and ask a human before doing X" | `langgraph.types.interrupt()` inside the tool function itself, resumed with `Command(resume=...)` - propagates up through nested/subgraph calls to whichever graph is holding the checkpoint |
| MCP tool integration via Azure AI Foundry's MCP tool type | Same MCP server could be wrapped with `langchain-mcp-adapters` to auto-generate `@tool`-compatible functions from the server's tool list; here it's mocked directly in `clients/powerbi/mcp.py` |
| Streaming agent updates via SDK event handlers | `graph.stream(...)` / `graph.astream(...)`, with `stream_mode="values"` for incremental state or `"updates"` for per-node diffs |

The biggest structural difference: Azure AI Agents Service treats the agent
as a managed server-side resource you configure and poll. LangGraph makes
the control flow a graph you define yourself - more code up front, but the
loop, memory, and interrupt points are explicit and inspectable rather than
living inside a managed run object.

## Consequences

- More boilerplate per agent (five files instead of one Assistants API
  call), but each piece (prompt, tool binding, node wiring, state shape) is
  independently testable - see `tests/unit` and `tests/integration`.
- `InMemorySaver` is process-local, fine for this demo and for local
  development, but a production deployment needs a persistent checkpointer
  (Postgres/Redis-backed) so approvals and conversations survive restarts
  and work across multiple app instances.
- The Python sandbox and Power BI clients are mocked; swapping in the real
  MCP/REST/sandbox integrations only touches `src/data_analyst/clients/`,
  none of the agent or graph code.
