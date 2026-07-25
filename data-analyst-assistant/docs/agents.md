# Agents

## Orchestrator (`agents/orchestrator/`)

The supervisor. Doesn't have its own tools - it only ever does one of three
things: ask the model to route (`chains.py::build_supervisor_chain`, a
`Route(next=..., reason=...)` structured output over `"datasource"` /
`"analysis"` / `"respond"` / `"clarify"`), produce the final answer
(`build_respond_chain`, a plain chat call over the accumulated
conversation), or ask a clarifying question (`build_clarify_chain`, used
when the supervisor picks `"clarify"` - see `CLARIFY_SYSTEM_PROMPT` in
`prompts.py`). `nodes.py` wires all three into graph nodes, plus the two
delegation wrappers (`build_datasource_node`, `build_analysis_node`) that
seed and fold specialist subgraphs - see `docs/architecture.md` for why that
folding is manual rather than a native LangGraph subgraph node, and for how
`data_context` and the session-bound data store let a follow-up question
reuse already-fetched data.

`state.py`'s `OrchestratorState` adds `turns` (loop guard, `MAX_TURNS = 6` in
`nodes.py`), `next` (the last routing decision - also what `app/api.py` uses
to tell a `"clarification_needed"` response from a `"completed"` one), and
`data_context` (a human-readable summary of the most recently fetched
dataset) on top of the shared `ChatState`.

## Datasource (`agents/datasource/`)

Power BI specialist, **read-only by design**: metadata lookups and
structured queries only, nothing that changes state in Power BI. Query
execution goes through the REST client (mirroring the real Power BI REST
API's "Execute Queries" endpoint), not MCP - the MCP client is
discovery/metadata only. Tools (`nodes.py`):

| Tool | Backing client | Notes |
|---|---|---|
| `pbi_mcp_list_semantic_models` | `clients/powerbi/mcp.py` | reads `config/semantic_models.yaml` |
| `pbi_rest_list_workspaces` | `clients/powerbi/rest.py` | |
| `pbi_rest_get_refresh_history` | `clients/powerbi/rest.py` | |
| `pbi_rest_run_dax_query` | `clients/powerbi/rest.py` + `clients/powerbi/dax.py` | takes structured `group_by`/`filters`/`measures`, never free-form DAX; builds and validates a SUMMARIZECOLUMNS query, checks the session's cache before running it, stages the result and returns a `sandbox_ref` |

`pbi_rest_run_dax_query` also receives `state` via
`langgraph.prebuilt.InjectedState` (invisible to the model's tool schema -
see `tool_call_schema` vs. `args_schema`) to reach its session's data store
by `session_id`. `models.py` defines the structured shapes these tools
conceptually return (`SemanticModelInfo`, `DaxQueryResult`, `WorkspaceInfo`,
`RefreshHistoryEntry`); the tools themselves return plain dicts at the
LangChain tool boundary, catching `ValueError` from spec construction/query
validation and returning `{"error": ...}` rather than raising, so the model
can see what was wrong and retry. `clients/powerbi/rest.py` has no
write/admin method (e.g. triggering a refresh) at all - this isn't just a
tool that's unexposed, the capability doesn't exist in the client layer
either.

## Analysis (`agents/analysis/`)

Sandbox specialist, one tool: `python_sandbox_execute`, which runs pandas
code against a DataFrame staged earlier (by the datasource agent, possibly
in an earlier turn) via `clients/sandbox/client.py`, reached the same way as
the datasource agent - `state["session_id"]` injected via `InjectedState`.
`models.py` defines `SandboxExecutionResult` for the same reason as above.

## Common (`agents/common/`)

- `state.py::ChatState` - the `messages` + `session_id` shape every agent
  graph shares. `session_id` scopes the data store in
  `clients/sandbox/client.py` and is only ever read via `InjectedState`
  inside a tool, never exposed to the model.
- `models.py::AgentResult` - what a specialist hands back to the
  orchestrator (`agent` name + `summary` text).

## Adding a new specialist

1. Copy the shape of `agents/analysis/` (it's the smallest): `state.py`
   (extend `ChatState`), `prompts.py` (a system prompt scoped to the new
   tools), `chains.py` (bind the tools to the model), `nodes.py` (the
   `@tool` functions + `agent`/`tools` node builders), `graph.py` (the
   `agent <-> tools` loop).
2. Add a `build_<name>_node` to `agents/orchestrator/nodes.py` following
   `build_analysis_node`, wire it into `agents/orchestrator/graph.py`
   (`add_node` + edges back to `supervisor`), and mention it in
   `SUPERVISOR_SYSTEM_PROMPT` (`agents/orchestrator/prompts.py`) so the
   router knows when to pick it.
3. Any tool that mutates something outside the conversation should call
   `langgraph.types.interrupt()` for human approval before acting - see
   "No mutating actions (by design)" in `docs/architecture.md` for how that
   propagates through the orchestrator, and note that no tool in this
   codebase currently does this (the datasource agent is read-only).
4. Any tool that needs the session's data store (or anything else in
   `state`) should take a `state: Annotated[YourState, InjectedState]`
   parameter (see `pbi_rest_run_dax_query` or `python_sandbox_execute`) -
   LangGraph populates it from the graph state and strips it from the
   schema shown to the model, so it can't be spoofed or need to be supplied
   by the LLM.
