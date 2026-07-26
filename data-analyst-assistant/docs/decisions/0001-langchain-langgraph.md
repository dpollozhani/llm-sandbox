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
`tools_condition`), and every node/chain/tool async throughout. See
`docs/architecture.md` for the full shape.

## Mapping from the Azure AI SDK version

| Azure AI SDK concept | LangChain/LangGraph equivalent |
|---|---|
| Agent + function tools (`FunctionTool`, `ToolSet`) | Python functions decorated with `@tool` (`langchain_core.tools`), collected into a list and passed to `bind_tools` |
| `AgentsClient.create_agent(model=..., instructions=..., tools=...)` | `init_chat_model(...)` / a provider-specific client (`AzureChatOpenAI`) + a system prompt string + `llm.bind_tools(tools)` - no server-side agent resource, it's local objects built by `clients/llm/factory.py` |
| Threads / runs (`client.agents.threads`, `client.agents.runs`) | A `StateGraph` compiled with a checkpointer; a `thread_id` in `config["configurable"]` scopes persisted state - same idea, but the graph shape is yours to define instead of a fixed run loop |
| Multiple specialized agents handed off via the Assistants/Agents "handoff" pattern | LangGraph subgraphs added as nodes in a supervisor `StateGraph` - each specialist is a fully independent compiled graph, composed rather than orchestrated by a special SDK feature |
| Polling a run for `requires_action` / submitting tool outputs | The graph *is* the loop: `agent -> tools -> agent` via `tools_condition`, no polling required - `graph.ainvoke` just runs until it hits `END` or an `interrupt()` |
| Custom code for "pause and ask a human before doing X" | `langgraph.types.interrupt()` inside the tool function itself, resumed with `Command(resume=...)` - propagates up through nested/subgraph calls to whichever graph is holding the checkpoint |
| MCP tool integration via Azure AI Foundry's MCP tool type | `clients/powerbi/mcp.py` calls the remote Power BI MCP server directly via the `mcp` SDK's streamable-HTTP client, rather than going through `langchain-mcp-adapters`'s auto-generated `@tool` wrapping - only one tool (`GetSemanticMetadata`) is used, so a thin hand-written client was simpler than pulling in the adapter for one call |
| Streaming agent updates via SDK event handlers | `graph.stream(...)` / `graph.astream(...)`, with `stream_mode="values"` for incremental state or `"updates"` for per-node diffs |
| Constraining a function tool's arguments to a safe, structured shape (custom JSON-schema validation in your function body) | Pydantic models as tool parameter types (`clients/powerbi/dax.py::DaxFilter`/`DaxMeasure`), which LangChain turns into the tool's JSON schema automatically - the model can only submit `group_by`/`filters`/`measures`, never a raw query string, and the built query is validated again server-side (`validate_dax_query`) before use |
| Passing request-scoped context (user id, session) into a function tool (typically a custom parameter or thread-local) | `langgraph.prebuilt.InjectedState` - a tool parameter annotated `Annotated[StateT, InjectedState]` that LangGraph fills in from the graph's state and removes from the schema the model sees (`tool.tool_call_schema` vs. `tool.args_schema`) |
| An agent asking a clarifying question (custom logic, e.g. a special function tool or a specific instruction in the system prompt) | Two paths, deliberately kept separate: a fourth `Route` option (`"clarify"`) the supervisor can pick for broad ambiguity, and a shared `request_clarification` tool (`agents/common/tools.py`) a *specialist* can call for narrower ambiguity it only discovers mid-task - the latter short-circuits straight to the reply via a conditional edge, skipping a supervisor round-trip (see "Two places a clarifying question can come from" in `docs/architecture.md`) |
| The SDK's own async client handling the whole call under the hood | Every node/chain/tool/client method here is `async def` and invoked via `.ainvoke()`/`await` by hand - LangGraph enforces this once *any* node is async (a sync `.invoke()` on a graph with an async node raises immediately), so getting the async boundary right is the graph author's job, not something a single SDK call guarantees for you - see "Async, end to end" in `docs/architecture.md` |

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
  (Postgres/Redis-backed) so conversations survive restarts and work across
  multiple app instances.
- The datasource agent is read-only (metadata + DAX queries only, no PBI
  REST write/admin calls), so the `interrupt()`-based human-approval row
  above describes a capability the stack has, not one this codebase
  currently exercises - see "No mutating actions (by design)" in
  `docs/architecture.md`.
- Power BI access is REST + the remote MCP server's `GetSemanticMetadata`,
  both requiring delegated Entra ID sign-in - see `clients/powerbi/auth.py`;
  only the Python sandbox's data layer is still mocked/in-process. Swapping
  the sandbox for an isolated execution service only touches
  `src/data_analyst/clients/sandbox/`, none of the agent or graph code.
- The session-bound data store (`clients/sandbox/client.py`) is, like
  `InMemorySaver`, process-local and lost on restart - the same "swap for a
  shared backing store in production" caveat applies to both.
- Going fully async touches nearly every file (chains, nodes, tools, client
  methods) and removes sync `.invoke()` as an option anywhere in the graph -
  a real cost for a "simplified" build, taken on because a FastAPI app that
  quietly blocks its event loop on every request defeats the point of using
  FastAPI. `clients/sandbox/client.py::SandboxClient.execute` is the one
  place doing genuinely CPU-bound work (arbitrary code execution), offloaded
  via `asyncio.to_thread` rather than naively awaited.
