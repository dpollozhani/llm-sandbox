# Agents

## Orchestrator (`agents/orchestrator/`)

The supervisor. Doesn't have its own tools - it only ever does one of two
things: ask the model to route (`chains.py::build_supervisor_chain`, a
`Route(next=..., reason=...)` structured output), or produce the final
answer (`build_respond_chain`, a plain chat call over the accumulated
conversation). `nodes.py` wires both into graph nodes, plus the two
delegation wrappers (`build_datasource_node`, `build_analysis_node`) that
seed and fold specialist subgraphs - see `docs/architecture.md` for why that
folding is manual rather than a native LangGraph subgraph node.

`state.py`'s `OrchestratorState` adds `turns` (loop guard, `MAX_TURNS = 6` in
`nodes.py`) and `next` (the last routing decision) on top of the shared
`ChatState`.

## Datasource (`agents/datasource/`)

Power BI specialist. Tools (`nodes.py`):

| Tool | Backing client | Notes |
|---|---|---|
| `pbi_mcp_list_semantic_models` | `clients/powerbi/mcp.py` | reads `config/semantic_models.yaml` |
| `pbi_mcp_run_dax_query` | `clients/powerbi/mcp.py` | stages the result into the sandbox (`clients/sandbox/client.py`) and returns a `sandbox_ref` |
| `pbi_rest_list_workspaces` | `clients/powerbi/rest.py` | |
| `pbi_rest_get_refresh_history` | `clients/powerbi/rest.py` | |
| `pbi_rest_trigger_dataset_refresh` | `clients/powerbi/rest.py` | mutating - calls `interrupt()` for approval before executing |

`models.py` defines the structured shapes these tools conceptually return
(`SemanticModelInfo`, `DaxQueryResult`, `WorkspaceInfo`,
`RefreshHistoryEntry`, `RefreshResult`); the tools themselves return plain
dicts at the LangChain tool boundary.

## Analysis (`agents/analysis/`)

Sandbox specialist, one tool: `python_sandbox_execute`, which runs pandas
code against a DataFrame staged earlier by the datasource agent (referenced
by `sandbox_ref`) via `clients/sandbox/client.py`. `models.py` defines
`SandboxExecutionResult` for the same reason as above.

## Common (`agents/common/`)

- `state.py::ChatState` - the `messages` shape every agent graph shares.
- `models.py::ApprovalRequest` - the payload shape surfaced through
  `interrupt()`.
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
   `interrupt()` before acting, following
   `pbi_rest_trigger_dataset_refresh` - see the propagation rules in
   `docs/architecture.md`.
