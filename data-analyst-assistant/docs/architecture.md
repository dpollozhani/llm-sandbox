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
    the supervisor can't even tell which specialist should handle the
    request. Narrower ambiguity (which table/columns/filters/measures, or
    which computation) is instead handled by the specialist itself asking -
    see "Two places a clarifying question can come from" below. Either way
    the API surfaces it as `status: "clarification_needed"`; the user's
    reply continues the same conversation (same `thread_id`).
- **`agents/datasource`** and **`agents/analysis`**: each is an independent,
  small ReAct-style `StateGraph` - an `agent` node bound to a scoped set of
  tools via `bind_tools`, and a `ToolNode`, looping through
  `tools_condition` until the model answers without calling a tool. The
  datasource agent is read-only: metadata lookups and structured queries
  only, no tool that changes anything in Power BI. Both specialists also
  have a `request_clarification` tool (`agents/common/tools.py`).
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

## Two places a clarifying question can come from

Asking for clarification isn't only a supervisor-level decision. There are
two distinct paths, deliberately kept separate because they resolve
different kinds of uncertainty at different costs:

1. **The supervisor asks upfront** (`Route(next="clarify")`,
   `build_clarify_node`/`clarify` node): for when the request is so vague
   the supervisor can't even tell which specialist should handle it. This
   costs one supervisor call and ends the turn.
2. **A specialist asks mid-task** (`request_clarification` tool,
   `agents/common/tools.py`): for narrower ambiguity only visible once a
   specialist is actually trying to build the query or pick the
   computation - which the supervisor has no good way to predict before
   delegating. Requiring the supervisor to anticipate every possible
   query-building or analysis ambiguity upfront would mean either being
   overcautious (clarifying when the specialist would have managed fine) or
   still delegating and hoping - so the specialist is given the tool to
   bail out itself, exactly when it discovers it needs to.

`_run_specialist` (`agents/orchestrator/nodes.py`) detects the second case
by checking whether `request_clarification` was called during the
specialist's run (`_specialist_asked_for_clarification`), and if so sets
`next="clarify"` itself and returns the specialist's own question directly.
`agents/orchestrator/graph.py`'s conditional edges out of `datasource`/
`analysis` check that: normally they loop back to `supervisor`, but if
`next` is now `"clarify"`, they go straight to `END` instead. This is the
cheaper path precisely because it skips the extra supervisor call *and* the
separate `clarify` node entirely - one specialist invocation, one reply -
rather than always paying for a full supervisor round-trip regardless of
where the ambiguity was actually discovered.

## Structured, validated DAX queries

`pbi_rest_run_dax_query` (`agents/datasource/nodes.py`) never accepts DAX
text from the model - only structured `group_by` columns, `filters`, and
`measures` (`clients/powerbi/dax.py::DaxQuerySpec`). The tool:

1. Builds a single `SUMMARIZECOLUMNS(...)` string from the spec
   (`build_summarizecolumns`).
2. Validates it structurally (`validate_dax_query`) before it would be sent
   to the REST endpoint: the text must actually be a well-formed
   `SUMMARIZECOLUMNS(...)` call, and at least one group-by column or measure
   must be selected. It can't check that a referenced column actually exists
   on the target table - there's no live schema lookup here - so that class
   of error surfaces from Power BI's own response instead (see next point).
3. Sends it to the real Power BI REST `executeQueries` endpoint
   (`PBIRestClient.run_dax_query`) using the caller's delegated access
   token, and parses the JSON response back into a DataFrame
   (`parse_execute_queries_response`).

Failures (a validation problem, an unknown model, or Power BI's own error
response - e.g. an unknown column) are caught inside the tool and returned
as `{"error": ...}` rather than raised, so the model sees the specific
problem and can correct its next tool call - the same pattern
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
A node function's call to `child_graph.ainvoke(seed)` (no `config=` passed)
picks up LangChain's *ambient* `RunnableConfig` - including the
orchestrator's checkpointer and `thread_id` - so a future mutating tool
could call `langgraph.types.interrupt()` and have the pause bubble up
through the child graph, through `_run_specialist`
(`agents/orchestrator/nodes.py`), and out of `orchestrator.ainvoke()` as
`result["__interrupt__"]`, with no extra plumbing to bridge parent and
child. Two things to get right if you add one:

1. **Never wrap a specialist's `.ainvoke()` call in a broad `except
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

## Async, end to end

Every node function, chain, and tool in this codebase is `async def`, and
every graph is invoked with `.ainvoke()` - not just `app/api.py`'s endpoint.
LangGraph requires this consistency: a graph with even one async node
function raises `TypeError: No synchronous function provided to "..."` if
you call its sync `.invoke()` - see `tests/` for how that plays out (they're
all async tests, using `pytest-asyncio`, except `tests/e2e`, which stays
sync because FastAPI's `TestClient` already bridges sync test code to the
async ASGI app for you).

Two things worth being precise about:

- **I/O-bound vs. CPU-bound.** The Power BI client methods
  (`clients/powerbi/mcp.py`, `rest.py`, `auth.py`) are async because they
  really do `await` a network call (real Power BI REST/MCP calls, real MSAL
  token acquisition) - that's I/O-bound work, so `await`ing it directly is
  correct and doesn't block the event loop. The sandbox's code execution
  (`clients/sandbox/executor.py`)
  is different: it's genuinely CPU-bound (it runs arbitrary code, which
  could be slow), so merely marking it `async def` wouldn't help - it would
  still block the event loop for its duration. `SandboxClient.execute`
  (`clients/sandbox/client.py`) instead offloads it with
  `asyncio.to_thread(execute, ...)`, which is the correct pattern for
  "async-friendly" CPU-bound work: other requests keep being served while
  it runs. Getting this distinction backwards - `await`ing CPU-bound work
  directly, or `to_thread`-ing something that's actually I/O-bound - is a
  common way real async codebases quietly lose their concurrency.
- **Why it matters here specifically.** FastAPI's whole reason for being
  async is serving concurrent requests without one blocking another; a
  sync-underneath implementation wrapped in `async def` just for the
  endpoint signature would defeat that the moment it talked to a real
  Power BI/sandbox backend - every request would tie up a thread for the
  duration of each call instead of yielding the event loop while waiting.

## Streaming (SSE)

`POST /chat/stream` (`app/api.py::chat_stream`) drives the same turn as
`POST /chat`, but via `graph.astream_events(..., version="v2")` instead of
`ainvoke`, streaming Server-Sent Events as the run progresses instead of
waiting for the whole thing to finish. Both endpoints exist side by side -
`/chat` for simple request/response callers, `/chat/stream` for the web page
(`app/web.py`) and `cli.py`'s default mode.

The interesting part is how little of the rest of the codebase had to change
to support this - none of the chains, since `astream_events` already exists
on the compiled graph object and needs nothing from the call sites beneath
it:

- **Node-level progress comes from event metadata, for free.** Every event
  `astream_events` yields carries `metadata["langgraph_node"]`, so
  `chat_stream` can tell "the `datasource` node just started" apart from
  "the `analysis` node just started" without any node needing to say so
  itself. This works for nested subgraph nodes too (e.g. the datasource
  specialist's own internal `agent`/`tools` nodes) via the same ambient
  propagation that already makes the orchestrator's checkpointer reach
  child graphs (see "How delegation works" above) - callbacks and config
  propagate through nested `.ainvoke()` calls the same way.
- **A single node visit fires several nested `on_chain_start` events**
  (the node's own runnable, its chain's inner `_invoke`, etc.), all tagged
  with the same `langgraph_node` metadata. `chat_stream` only treats the one
  whose own name matches the node name as "this node started" - filtering
  by `node not in seen_nodes` instead (i.e. only the *first* visit) would be
  wrong, since a node like `supervisor` legitimately runs more than once in
  a turn and each visit should report status.
- **Token-level streaming needs zero chain changes, but does need a real
  `_stream`/`_astream` on the model.** Every chain here calls `llm.ainvoke()`,
  never `.astream()` - yet `on_chat_model_stream` events still carry
  per-token chunks under `astream_events`, because LangChain's event
  machinery routes the call through the model's streaming code path
  regardless of which method the calling code used, *if* the model
  implements one. `clients/llm/factory.py::FakeToolCallingChatModel` (the
  scripted model every test drives) adds a `_stream` override for exactly
  this reason - without it, a test's scripted model would only ever produce
  whole-message `on_chat_model_end` events, and `/chat/stream` would show
  status updates but never live-typed tokens, an easy thing to miss if you
  only test with a model that already streams. Real providers (Anthropic,
  Azure OpenAI) implement real streaming natively, so this only matters for
  the test double.
- **Token events are filtered to the user-facing answer.** `chat_stream`
  only forwards `on_chat_model_stream` chunks when
  `metadata["langgraph_node"]` is `"respond"` or `"clarify"` - otherwise the
  supervisor's routing decision and a specialist's internal reasoning would
  leak into the stream looking like partial answers.
- **The final `"done"` event is the authoritative result**, read from the
  outermost graph's own `on_chain_end` (`event["name"] == "LangGraph"` with
  no `langgraph_node` in its metadata - that's what distinguishes the whole
  run's own start/end from any node's), not reconstructed from accumulated
  tokens - those two happen to add up to the same text in practice, but
  `"done"` doesn't depend on that being true.
- **The browser can't use `EventSource` for this.** `EventSource` only
  supports `GET` with no request body, and this endpoint needs a JSON POST
  body (`message`/`thread_id`). `app/web.py`'s JS instead reads the
  `text/event-stream` body by hand via `fetch()` + `ReadableStream`,
  splitting on blank lines the same way any SSE parser would.

## What's mocked vs. real

- **Real**: the LangGraph control flow (supervisor loop, ReAct loops,
  checkpointing), the async structure throughout (see above), streaming via
  `astream_events` (see above), the FastAPI request/response cycle, the
  sandbox's `exec()`-based code execution, and Power BI itself - real REST
  API calls (`clients/powerbi/rest.py`), a real remote MCP server call for
  `GetSemanticMetadata` (`clients/powerbi/mcp.py`), and real Entra ID
  delegated auth (`clients/powerbi/auth.py`, `app/auth.py`, `cli.py`'s
  device-code flow) - see that auth module's docstring for why a delegated
  user token is required rather than an app-only one.
- **Mocked**: only the Python sandbox's data layer - `clients/sandbox/`
  really executes code via `exec()`, but against an in-process dict of
  staged DataFrames rather than an isolated execution service.
