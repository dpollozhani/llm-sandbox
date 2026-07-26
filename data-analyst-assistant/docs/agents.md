# Agents

## Orchestrator (`agents/orchestrator/`)

The supervisor. Doesn't have its own tools - it only ever does one of three
things: ask the model to route (`chains.py::build_supervisor_chain`, a
`Route(next=..., reason=...)` structured output over `"datasource"` /
`"analysis"` / `"respond"` / `"clarify"`), produce the final answer
(`build_respond_chain`, a plain chat call over the accumulated
conversation), or ask a clarifying question itself (`build_clarify_chain`,
used when the supervisor picks `"clarify"` - see `CLARIFY_SYSTEM_PROMPT` in
`prompts.py`, reserved for when it's unclear even which specialist should
handle the request). `nodes.py` wires all three into graph nodes, plus the
two delegation wrappers (`build_datasource_node`, `build_analysis_node`)
that seed and fold specialist subgraphs - see `docs/architecture.md` for why
that folding is manual rather than a native LangGraph subgraph node, how
`data_context` and the session-bound data store let a follow-up question
reuse already-fetched data, and how those same two wrappers detect and
short-circuit on a *specialist's own* clarifying question (narrower
ambiguity than the supervisor could have caught upfront) without an extra
supervisor round-trip.

`state.py`'s `OrchestratorState` adds `turns` (loop guard, `MAX_TURNS = 6` in
`nodes.py`), `next` (the last routing decision - also what `app/api.py` uses
to tell a `"clarification_needed"` response from a `"completed"` one),
`data_context` (a human-readable summary of the most recently fetched
dataset), and `clarification_options` (the 2-3 options alongside a
`"clarify"` outcome - see "Two places a clarifying question can come from")
on top of the shared `ChatState`.

## Datasource (`agents/datasource/`)

Power BI specialist, **read-only by design**: metadata lookups and
structured queries only, nothing that changes state in Power BI. Query
execution goes through the REST client (the Power BI REST API's
"Execute Queries" endpoint), not MCP - the MCP client only calls the remote
Power BI MCP server's `GetSemanticMetadata` tool. Both clients require the
caller's own delegated access token (see `clients/powerbi/auth.py`'s module
docstring - RLS requires it), read from `state` via `InjectedState`, not a
client-held credential. Tools (`nodes.py::build_tools`):

There's no workspace/refresh-history tool - both were dropped as unneeded
for this build. Without a "list models" tool either, the model instead
learns valid `model_name` values from the static catalog
(`config/semantic_models.yaml`), appended to the system prompt by
`chains.py::build_agent_chain`.

| Tool | Backing client | Notes |
|---|---|---|
| `pbi_mcp_get_semantic_metadata` | `clients/powerbi/mcp.py` | resolves `model_name` to a dataset id via `config/semantic_models.yaml`, then calls the MCP server's `GetSemanticMetadata` |
| `pbi_rest_run_dax_query` | `clients/powerbi/rest.py` + `clients/powerbi/dax.py` | takes structured `group_by`/`filters`/`measures`, never free-form DAX; builds and structurally validates a SUMMARIZECOLUMNS (or, with no `group_by`, a ROW grand-total) query, checks the session's cache before running it, calls the `executeQueries` endpoint, stages the parsed result and returns a `sandbox_ref` |
| `request_clarification` | `agents/common/tools.py` | shared with the analysis agent; asks the user a question when the specialist itself is unsure what's meant - see `docs/architecture.md`'s "Two places a clarifying question can come from" |

Every tool receives `state` via `langgraph.prebuilt.InjectedState` (invisible
to the model's tool schema - see `tool_call_schema` vs. `args_schema`): to
read the delegated `pbi_token` (returning `{"error": ...}` if missing
rather than calling Power BI with no token), and, for
`pbi_rest_run_dax_query`, to also reach its session's data store by
`session_id`. `build_tools(mcp_client=..., rest_client=...)` takes the two
Power BI clients as parameters specifically so tests can inject fakes
without reaching past the `@tool` decorators - see
`tests/integration/test_datasource_graph.py`. Tools return plain dicts at
the LangChain tool boundary, catching `ValueError` from spec construction,
structural validation, or the REST/MCP call itself, and returning
`{"error": ...}` rather than raising, so the model can see what was wrong
and retry. `clients/powerbi/rest.py` has no write/admin method (e.g.
triggering a refresh) at all - this isn't just a tool that's unexposed, the
capability doesn't exist in the client layer either.

## Analysis (`agents/analysis/`)

Sandbox specialist, two tools: `python_sandbox_execute`, which runs pandas
code against a DataFrame staged earlier (by the datasource agent, possibly
in an earlier turn) via `clients/sandbox/client.py`, reached the same way as
the datasource agent - `state["session_id"]` injected via `InjectedState`;
and the same shared `request_clarification` as the datasource agent, for
when the requested analysis itself is ambiguous. `models.py` defines
`SandboxExecutionResult` for the same reason as above.

## Common (`agents/common/`)

- `state.py::ChatState` - the `messages` + `session_id` shape every agent
  graph shares. `session_id` scopes the data store in
  `clients/sandbox/client.py` and is only ever read via `InjectedState`
  inside a tool, never exposed to the model.
- `models.py::AgentResult` - what a specialist hands back to the
  orchestrator (`agent` name + `summary` text).
- `models.py::Clarification` - a `question` plus 2-3 clearly distinct
  options, the shape both clarifying-question paths produce (see
  `docs/architecture.md`'s "Two places a clarifying question can come
  from").
- `tools.py::request_clarification` - the tool both specialists share for
  asking the user a question (with those 2-3 options) instead of guessing.

## Adding a new specialist

1. Copy the shape of `agents/analysis/` (it's the smallest): `state.py`
   (extend `ChatState`), `prompts.py` (a system prompt scoped to the new
   tools), `chains.py` (bind the tools to the model), `nodes.py` (the
   `@tool` functions + `agent`/`tools` node builders), `graph.py` (the
   `agent <-> tools` loop). Every tool function, chain `_invoke`, and node
   function should be `async def` (see "Async, end to end" in
   `docs/architecture.md`) - a graph with even one sync node function can no
   longer be invoked at all once other nodes are async.
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
5. Add `agents/common/tools.py::request_clarification` to the new
   specialist's `TOOLS` list (see `docs/architecture.md`'s "Two places a
   clarifying question can come from") and mention it in the specialist's
   own system prompt, following `agents/datasource/prompts.py`.
