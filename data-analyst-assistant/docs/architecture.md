# Architecture

## Shape

```
FastAPI (app/api.py)
        |
        v
 orchestrator graph  (checkpointed - this is the unit of persistence/resume)
   START -> supervisor -> {datasource | analysis} -> supervisor -> ... -> {respond | clarify} -> END
        |                        |            |
        |             (delegates, one task at a time)
        v                        v            v
   Route(next=...)      datasource graph   analysis graph
                         agent<->tools loop  agent<->tools loop
                         (PBI MCP/REST)      (Python sandbox)
                                \            /
                                 v          v
                          session-scoped data store
                          (clients/sandbox/client.py)
```

- **`agents/orchestrator`**: a supervisor loop. `supervisor` asks the model
  for a structured `Route` decision (`datasource` / `analysis` / `respond` /
  `clarify`); the corresponding node runs, then control returns to
  `supervisor` for the next decision, up to `MAX_TURNS`.
  - `respond` produces the final answer.
  - `clarify` asks the user a short clarifying question instead - used when
    the supervisor isn't confident how to build the next query (which
    table/columns/filters/measures) or how to perform the requested
    analysis. The API surfaces this as `status: "clarification_needed"`; the
    user's reply continues the same conversation (same `thread_id`).
- **`agents/datasource`** and **`agents/analysis`**: each is an independent,
  small ReAct-style `StateGraph` - an `agent` node bound to a scoped set of
  tools via `bind_tools`, and a `ToolNode`, looping through
  `tools_condition` until the model answers without calling a tool. The
  datasource agent is read-only: metadata lookups and structured queries
  only, no tool that changes anything in Power BI.
- **`agents/common`**: the `ChatState` shape (`messages` + `session_id` with
  LangGraph's `add_messages` reducer) shared by all three graphs, and
  `AgentResult`, the small model a specialist hands back to the
  orchestrator.
- **`clients/`**: everything that would talk to a real system in
  production - Power BI (MCP + REST + auth), the code sandbox, and the LLM
  provider. Agent code never imports a provider SDK directly; it only calls
  into `clients/`.

## How delegation works

`agents/orchestrator/nodes.py`'s `_run_specialist` seeds a **fresh** child
conversation from only the user's actual latest question (not the
orchestrator's full history), builds the specialist's compiled subgraph, and
invokes it. This is a deliberate choice over LangGraph's native "add a
compiled graph directly as a node" support: every graph here shares the same
`messages`/`ChatState` convention, so a native subgraph node would silently
pass the orchestrator's *entire* transcript into the specialist and merge
the specialist's *entire* transcript back - the opposite of "delegate one
scoped task, get back one summary." Manual invocation from a plain node
function is what makes that translation possible.

"The user's actual latest question" is deliberately not just
`state["messages"][-1]`: once a first specialist has already run within the
same supervisor turn (e.g. datasource, then analysis), the last message is
that specialist's own folded-back summary, not the user's question.
`_latest_user_task` walks backward to find the most recent human message
instead. Alongside it, `_run_specialist` threads `state["data_context"]`
(see below) into the seed message, so a specialist with no memory of earlier
turns still knows what data is already available.

## Structured, validated DAX queries

`pbi_rest_run_dax_query` (`agents/datasource/nodes.py`) never accepts DAX
text from the model - only structured `group_by` columns, `filters`, and
`measures` (`clients/powerbi/dax.py::DaxQuerySpec`). The tool:

1. Builds a single `SUMMARIZECOLUMNS(...)` string from the spec
   (`build_summarizecolumns`).
2. Validates it (`validate_dax_query`) before it would be sent to the REST
   endpoint: the text must actually be a well-formed `SUMMARIZECOLUMNS(...)`
   call, at least one group-by column or measure must be selected, and every
   referenced column must exist on the target table.
3. Only then "executes" it (`execute_query`, a small pandas engine standing
   in for the real semantic model) against the mocked table.

Validation failures are caught inside the tool and returned as `{"error":
...}` rather than raised, so the model sees the specific problem (e.g. an
unknown column) and can correct its next tool call - the same pattern
`clients/sandbox/executor.py` already uses for sandbox code errors.

## Session-bound data reuse

Every conversation (`thread_id`) gets its own `SandboxClient`
(`clients/sandbox/client.py::get_sandbox_client`), which holds both the
staged DataFrames (referenced by `sandbox_ref`) *and* a cache from a query's
structural signature (`DaxQuerySpec.cache_key()`) to the `sandbox_ref` that
already answers it. `pbi_rest_run_dax_query` checks that cache before
running a query at all: an unchanged data requirement - even across
separate HTTP turns - is served from the cache (`reused: true` in the tool's
response) instead of hitting the REST client again.

Tools reach their session's store via `session_id`, carried in `ChatState`
and injected automatically with `langgraph.prebuilt.InjectedState` - the
model never sees or supplies it, so it can't be spoofed or forgotten. This
is also how the analysis agent finds data the datasource agent staged in an
*earlier* turn: `_run_specialist` seeds every specialist's fresh child state
with the same `session_id`, so the store carries over even though the
child's own `messages` don't.

This cache is the hard guarantee against redundant fetches. On top of it,
`SUPERVISOR_SYSTEM_PROMPT` tells the model what's already available
(`data_context`) so it can often skip delegating to "datasource" at all for
a follow-up that only needs analysis - a soft optimization, not something
the correctness of "no duplicate fetch" depends on.

## No mutating actions (by design)

The datasource agent only reads: semantic model/table discovery, DAX
queries, workspace listing, refresh history. There's no tool anywhere in
this build that changes state in Power BI (or elsewhere), so there's
nothing that needs a human-in-the-loop approval gate.

That said, the orchestrator's checkpointing already supports one for free.
A node function's call to `child_graph.invoke(seed)` (no `config=` passed)
picks up LangChain's *ambient* `RunnableConfig` - including the
orchestrator's checkpointer and `thread_id` - so a future mutating tool
could call `langgraph.types.interrupt()` and have the pause bubble up
through the child graph, through `_run_specialist`
(`agents/orchestrator/nodes.py`), and out of
`orchestrator.invoke()`/`ainvoke()` as `result["__interrupt__"]`, with no
extra plumbing to bridge parent and child. Two things to get right if you
add one:

1. **Never wrap a specialist's `.invoke()` call in a broad `except
   Exception`** (or apply `utils/retry.py`'s `@retry` to a node function that
   calls one). `interrupt()` works by raising `GraphInterrupt`, which *is* a
   plain `Exception` subclass - a broad catch would silently swallow the
   pause.
2. **The orchestrator node function reruns from its own top on resume.** From
   the parent graph's point of view, `_run_specialist` is one atomic task;
   when you resume with `Command(resume=...)`, LangGraph re-enters that node
   function from scratch. The child graph's own already-completed steps are
   *not* redone (tracked transparently via the same ambient checkpointer),
   but any plain Python before the interrupt point inside that task does
   re-run. Keep that code cheap and side-effect-free.

## What's mocked vs. real

- **Real**: the LangGraph control flow (supervisor loop, ReAct loops,
  checkpointing), the FastAPI request/response cycle, the sandbox's
  `exec()`-based code execution.
- **Mocked**: Power BI MCP/REST calls (`clients/powerbi/`, backed by
  `config/semantic_models.yaml` and in-memory fake tables) and Azure AD auth
  (`clients/powerbi/auth.py`). Swapping these for real integrations only
  touches `clients/`, not the agents or graphs.
