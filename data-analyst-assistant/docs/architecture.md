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
    which computation) is instead flagged by the specialist itself - see
    "Clarifications are the orchestrator's alone to surface" below. Either way the
    API surfaces it as `status: "clarification_needed"`; the user's reply
    continues the same conversation (same `thread_id`).
- **`agents/datasource`** and **`agents/analysis`**: each is an independent,
  small ReAct-style `StateGraph` - an `agent` node bound to a scoped set of
  tools via `bind_tools`, and a `ToolNode`, looping through
  `tools_condition` until the model answers without calling a tool. The
  datasource agent is read-only: metadata lookups and structured queries
  only, no tool that changes anything in Power BI. Both specialists also
  have a `flag_ambiguity` tool (`agents/common/tools.py`) to report - not
  ask - an ambiguity they can't resolve themselves.
- **`agents/common`**: the `ChatState` shape (`messages` + `session_id` with
  LangGraph's `add_messages` reducer) shared by all three graphs.
- **`clients/`**: everything that would talk to a real system in
  production - Power BI (MCP + REST + auth), the code sandbox, and the LLM
  provider. Agent code never imports a provider SDK directly; it only calls
  into `clients/`. The one deliberate exception to that direction:
  `clients/sandbox/executor.py::execute` returns
  `agents/analysis/models.py::ExecutionResult` - an agent-facing tool-result
  shape, not a client-internal one - imported back into the client layer
  rather than duplicated there, since `agents/datasource/nodes.py`'s
  `pbi_rest_run_dax_query` tool builds its own equivalent
  `agents/datasource/models.py::DataSourceQueryResult` directly rather than
  through a client return type.

## How delegation works

`agents/orchestrator/nodes.py`'s `_run_specialist` seeds a **fresh** child
conversation (not the orchestrator's full history), builds the specialist's
compiled subgraph, and invokes it. This is a deliberate choice over
LangGraph's native "add a compiled graph directly as a node" support: every
graph here shares the same `messages`/`ChatState` convention, so a native
subgraph node would silently pass the orchestrator's *entire* transcript into
the specialist and merge the specialist's *entire* transcript back - the
opposite of "delegate one scoped task, get back one summary." Manual
invocation from a plain node function is what makes that translation
possible.

For a **fresh task**, the seed is only the user's actual latest question -
deliberately not just `state["messages"][-1]`, since once a first specialist
has already run within the same supervisor turn (e.g. datasource, then
analysis), the last message is that specialist's own folded-back summary, not
the user's question. `_latest_user_task` walks backward to find the most
recent human message instead. Alongside it, `_run_specialist` threads
`state["data_context"]` and `state["last_analysis_result"]` (see below) into
the seed message, so a specialist with no memory of earlier turns still
knows what data is already available and what's already been concluded -
without either, a conclusion the analysis agent reached in prose (e.g. "the
top account is 654000") would only ever live in `state["messages"]`, which
specialist seeding never forwards; a later datasource delegation asked to
filter on "that account" would have no way to know what it refers to.

For a **reply to a clarifying question**, though, a single isolated message
isn't enough - a specialist subgraph is rebuilt from scratch on every
delegation, so with only "Total across all products and locations" (say) and
none of the preceding exchange, it has no way to know that's an answer about
the "Inventory on-hand" measure asked about three messages ago. Rather than
forwarding the *entire* raw orchestrator history to solve this (which would
also mean forwarding it - and re-paying for it - every subsequent turn),
`state["pending_clarification"]` (see below) tells `_run_specialist` exactly
what's outstanding, so it can seed the specialist with the original task
plus a one-line "replying to: X -> Y" note instead of the whole transcript.

## Clarifications are the orchestrator's alone to surface

Asking for clarification isn't only a supervisor-level decision. There are
three distinct paths that can produce one, deliberately kept separate
because they resolve different kinds of uncertainty at different costs -
but only the orchestrator ever decides what the user actually sees:

1. **The supervisor asks upfront** (`Route(next="clarify")`,
   `build_clarify_node`/`clarify` node): for when the request is so vague
   the supervisor can't even tell which specialist should handle it -
   routing-level uncertainty only. A data or analysis request's own
   specifics being unclear (which model/table/columns/filters/measures, or
   which computation) is never grounds for this path, no matter how vague -
   that's squarely path 2's job, never the supervisor's to ask about, even
   indirectly. `SUPERVISOR_SYSTEM_PROMPT`/`CLARIFY_SYSTEM_PROMPT`
   (`agents/orchestrator/prompts.py`) both say so explicitly, since a
   vaguer framing here previously let the supervisor's own clarify path
   drift into asking about data specifics that were properly a
   specialist's to flag. This costs one supervisor call
   (`build_clarify_node`'s structured output, producing a `Clarification` -
   `question` + options) and ends the turn.
2. **A specialist flags ambiguity mid-task** (`flag_ambiguity` tool,
   `agents/common/tools.py`): for narrower ambiguity only visible once a
   specialist is actually trying to build the query or pick the
   computation - which the supervisor has no good way to predict before
   delegating. The tool's result is read into the same `Clarification`
   shape as the supervisor's own path, but its `question` field is really
   just the specialist's own reason for the ambiguity, not a ready-to-send
   question: `_run_specialist` composes the actual user-facing message
   deterministically (`_compose_ambiguity_message`, no extra LLM call)
   rather than relaying the specialist's own tool-call arguments as if a
   model had phrased them for the user. Requiring the
   supervisor to anticipate every possible query-building or analysis
   ambiguity upfront would mean either being overcautious (clarifying when
   the specialist would have managed fine) or still delegating and hoping -
   so the specialist is given the tool to bail out itself, exactly when it
   discovers it needs to, at the same cost as its own final answer.
3. **The analysis specialist suggests a follow-up alongside a completed
   answer** (`suggest_followup` tool, `agents/common/tools.py`): for when
   it has *already* produced a real, complete answer, but there's a
   genuine, concretely-grounded fork in what to compute next (e.g. several
   equally valid further breakdowns) - the case that emerged in practice as
   specialists writing "which would you like?" straight into their own
   final message's prose instead of using any tool, which meant it never
   got `pending_clarification` tracking, never rendered as clickable
   buttons, and was lost to `resolved_clarifications` bookkeeping. Unlike
   path 2, this one is non-blocking: it never replaces or short-circuits
   the specialist's own final answer, both reach the orchestrator together
   (see `state["followup_suggestion"]` below).

   Only the analysis specialist has this tool - the datasource agent
   doesn't. Fetching data at the right grain is its whole job, full stop;
   even when the user's request also implies a ranking/trend/computation
   over that data, that's not "a real, complete answer with an optional
   next step" the way a computed result is - it's still an open question
   about how to finish, which is exactly what routing (datasource ->
   analysis) already exists to resolve, not something worth surfacing to
   the user as a suggestion. Giving the datasource agent `suggest_followup`
   turned out to invite exactly that misuse in practice: it would fetch
   raw data and immediately offer "how should I summarize this?" options
   before anything was actually computed - a `flag_ambiguity`-shaped
   situation dressed up as a non-blocking one.

Paths 1 and 2 converge on one field: `state["pending_clarification"]`
(`agents/orchestrator/state.py`) - `{"agent", "reason", "options"}`,
identifying who's waiting on a reply and why. This replaced a former
`awaiting_clarification`/`clarification_options` pair so that fact lives in
one place, not two that could drift out of sync. `ChatResponse`
(`app/api.py`) carries `pending_clarification["options"]` as `options`
alongside `reply` whenever `status` is `"clarification_needed"`, so a
frontend can render them as buttons instead of requiring a typed reply -
see `app/web.py`'s `renderOptions`, where picking one both removes the
buttons (so only one is ever selectable) and submits it exactly as if it
had been typed and sent.

`_run_specialist` (`agents/orchestrator/nodes.py`) detects the specialist's
own case by looking for a `flag_ambiguity` tool call in the specialist's run
and parsing its own structured result (`_specialist_ambiguity`) - rather
than trusting the specialist's own final freeform message to faithfully
restate the reason and options - and if found sets `pending_clarification`
directly (no `next` value involved).  `agents/orchestrator/graph.py`'s
`_after_specialist` conditional edge out of `datasource`/`analysis` checks
`pending_clarification` itself: normally it loops back to `supervisor`, but
if a specialist just set one, it goes straight to `END` instead. This is
the cheaper path precisely because it skips the extra supervisor call *and*
the separate `clarify` node entirely - one specialist invocation, one
reply, zero extra LLM calls - rather than always paying for a full
supervisor round-trip (or an extra rephrasing call) regardless of where the
ambiguity was actually discovered. Earlier versions of this signaled the
same thing via `next="clarify"` - a value that meant something entirely
different to the supervisor's own routing edge (see `Route(next="clarify")`
above) and invited exactly that confusion; `_after_specialist` now reads
`pending_clarification` directly instead.

Once a `pending_clarification` is resolved (or superseded by a new one),
`_append_resolved` folds it into `state["resolved_clarifications"]` - a
running `[{"question", "answer"}, ...]` list. Every specialist delegation
(`_seed_content`) and the supervisor's own routing/clarify prompts
(`_render_resolved` in `agents/orchestrator/nodes.py`) are given a compact
rendering of this list, so nothing re-asks (or re-derives from scratch)
something already settled earlier in the conversation - without needing the
full raw message history to do so.

On resume, `build_supervisor_node` checks `pending_clarification` before
doing anything else: if a specific specialist (not the supervisor itself)
is the one awaiting a reply, it routes straight back to that specialist -
skipping the routing decision's LLM call entirely - rather than re-deciding
from scratch and risking a mis-route.

### The non-blocking case: `followup_suggestion`

Path 3 above gets its own state field, `state["followup_suggestion"]`
(`agents/orchestrator/state.py`) - `{"agent", "question", "options"}`, same
shape as `pending_clarification` but a deliberately separate field, because
its semantics really are different: `pending_clarification` means "nothing
else happens until this is answered"; `followup_suggestion` means "here's
a real answer, and optionally, here's a next step if you want it." Mixing
the two into one field would force every reader of `pending_clarification`
to also handle a "well, actually, there's already a real answer this time"
case that doesn't apply to it.

That semantic difference changes how each one resumes:
- `pending_clarification` bypasses the routing LLM call entirely (see
  above) - a fair bet, since nothing else could have happened yet.
- `followup_suggestion` cannot make that same bet - the user's next message
  might pick up the suggestion, or might be about something else entirely,
  so `build_supervisor_node` still runs its normal routing call, just with
  the suggestion rendered into the prompt as extra context
  (`_render_followup_suggestion`) so the model can route back to the
  specialist that offered it if the reply actually matches.

Consistently with `pending_clarification`, every `_run_specialist` return
path sets `followup_suggestion` explicitly (to a new value or `None`) -
never left out of the update dict - so a stale suggestion from an earlier
turn can't linger once superseded. `app/api.py`'s `ChatResponse` surfaces
it as `suggested_options` (kept separate from `options`, which only means
something when `status` is `"clarification_needed"`) - `status` stays
`"completed"` and `reply` is the real answer either way; `app/web.py`
renders `suggested_options` with the same `renderOptions` button UI as a
blocking clarification's `options`, just styled differently (`.suggestion`)
since picking one - or ignoring them and asking something else - are
equally valid. `build_respond_node`'s own prompt is told about a pending
suggestion too, specifically so it doesn't restate the same options in
prose once they're already shown as buttons.

Because `followup_suggestion` still goes through a real routing LLM call
(unlike `pending_clarification`'s bypass), that call can get it wrong: a
production trace caught the router picking `"clarify"` immediately after a
specialist's own non-blocking suggestion, with no new human message in
between - reasking, as a *blocking* question, the exact fork the specialist
had just offered non-blockingly. `SUPERVISOR_SYSTEM_PROMPT` tells the model
not to do this, and `build_supervisor_node` also backstops it
deterministically: if the router still returns `"clarify"` while
`followup_suggestion` is set and the latest message isn't a fresh human
one, it's overridden to `"respond"`. The prompt instruction is cheap
insurance, not the actual guarantee - the code-level check is.

## No schema without a schema lookup

Only the datasource agent's `pbi_mcp_get_semantic_metadata` tool ever
actually sees a semantic model's schema (tables/columns/measures/
relationships). Every other node - the supervisor's routing decision, its
own `respond` answer, the `clarify` question - has zero schema access, and
their prompts (`agents/orchestrator/prompts.py`) say so explicitly: a
"which models are available" question can be answered directly from the
static catalog's model *names* (`config/semantic_models.yaml`, injected via
`nodes.py::_describe_catalog` - real config, not a guess), but anything
about a model's *contents* must route to "datasource" rather than being
answered from assumption or general knowledge of what a "typical" schema
might contain. `agents/datasource/prompts.py::SYSTEM_PROMPT` carries the
matching rule for the specialist itself: call
`pbi_mcp_get_semantic_metadata` before stating anything about a model's
tables/columns/measures - not just before running a query against it -
unless that model's schema was already fetched earlier in the same
conversation (the per-session cache in `clients/powerbi/mcp.py` is what
makes "already fetched" cheap to check for real, not just a prompt
instruction to trust).

This is a prompt-level guarantee, not a code-enforced one - unlike the
cache-key/`InjectedState` mechanisms elsewhere in this doc, there's no
practical way to programmatically verify "this response's table/column
names actually came from a tool result" without re-implementing schema
validation the MCP server already owns. The mitigation is keeping every
node's stated capabilities honest about what it can and can't see, so a
model has no prompt-level cover for guessing instead of looking something
up.

## Scoping a session to part of the catalog

`ChatRequest.model_names` (`app/api.py`) lets a caller restrict a session to
part of the full configured catalog (`config/semantic_models.yaml`) - e.g. a
frontend embedded in a specific Power BI app, where only one or two models
are actually relevant to that context. Each entry is a `model_name` or
`dataset_id`; `_resolve_allowed_model_names` resolves them against
`get_catalog()` and 400s on anything unrecognized (a likely frontend
integration bug - a stale link, a typo - worth surfacing immediately rather
than silently dropping).

Deciding *which* model(s) a given caller should see is entirely that
caller's own concern, deliberately kept out of this service: there's no
"app name" concept anywhere in this codebase, no lookup table mapping one to
the other. A caller with no notion of "app" at all (a standalone portal, the
CLI) never resolves anything and never pays for this feature; `model_names`
is `None` and every model in the catalog applies, exactly as before this
existed.

The resolved plain `list[str]` of names goes into
`OrchestratorState.allowed_model_names` - set once, directly in the initial
state of the call (the same convention as `pbi_token`), never written by a
node's own return dict. `agents/orchestrator/nodes.py::_effective_catalog`
narrows `PowerBiCatalog` to it, and that narrowed catalog is used for two
things, not one: what `_describe_catalog` renders as "which models are
available" into a prompt (supervisor, respond, and - since the datasource
specialist's own subgraph is rebuilt fresh on every delegation anyway
(`_run_specialist`) - the datasource agent's own prompt too), *and* what
`build_datasource_graph` passes into `build_tools`, which is what its
`PBIMcpClient`/`PBIRestClient` instances actually resolve a `model_name`
against. A model outside the subset doesn't just go undescribed - asking
for it fails exactly the same way asking for a model that was never in the
catalog at all would (`PBIMcpClient`/`PBIRestClient`'s own "Unknown
semantic model" `ValueError`, before any network call). From the agent's
own side this looks exactly like a smaller configured catalog - there's no
way for it to tell the difference between that and a genuine restriction,
and no way to reach around it from inside the conversation.

This is a real restriction for the life of one request, not a soft
default: "overridable" (see the earlier design discussion this came out
of) means the *caller* can change or lift the scoping by sending a
different `model_names` (or omitting it) on its *next* request - not that
the agent can override it mid-conversation by naming a different model
itself. Row-level security separately scopes what data a query can
actually return, per the caller's own Power BI identity; this is an
additional, independent restriction on which models are in play at all.

## Structured, validated DAX queries

`pbi_rest_run_dax_query` (`agents/datasource/nodes.py`) never accepts DAX
text from the model - only structured `group_by` columns, `filters`, and
`measures` (`clients/powerbi/dax.py::DaxQuerySpec`). The tool:

1. Builds a single DAX query string from the spec (`build_dax_query`): an
   `EVALUATE SUMMARIZECOLUMNS(...)` call if `group_by` is non-empty, or an
   `EVALUATE ROW(...)` grand total if it's empty - `SUMMARIZECOLUMNS`
   requires at least one group-by column syntactically, it has no
   "just give me the totals" mode.
2. Validates it structurally (`validate_dax_query`) before it would be sent
   to the REST endpoint: the text must actually be a well-formed call to
   whichever of those two the spec implies, and at least one group-by column
   or measure must be selected. It can't check that a referenced column
   actually exists on the target table - there's no live schema lookup here
   - so that class of error surfaces from Power BI's own response instead
   (see next point).
3. Sends it to the Power BI REST `executeQueries` endpoint
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
staged DataFrames (referenced by `dataset_id`) *and* a cache from a query's
structural signature (`DaxQuerySpec.cache_key()`) to the `dataset_id` that
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
the correctness of "no duplicate fetch" depends on. `data_context` is a
`DataSourceQueryResult` (`agents/datasource/models.py`) - the same model
`pbi_rest_run_dax_query` itself returns (`dataset_id`, `model_name`, the
query's `group_by`/`filters`/`measures`/`row_count`, plus `preview`/
`reused`/`dax_query`) - built by `_run_specialist` straight from that
tool's own structured result, not the datasource specialist's freeform
final summary, which has no guarantee of mentioning all of it.
`.describe()` renders it to the short line threaded into prompts - also
into `build_respond_node`'s own prompt now, the concrete grounding
`RESPOND_SYSTEM_PROMPT` needs for its "only suggest a follow-up when it's
scoped to the model actually in play" rule (see "Clarifications are the
orchestrator's alone to surface" for the parallel problem this was
avoiding on the clarify side: a prompt with no concrete grounding tends to
fill the gap with something plausible-sounding but ungrounded - here, a
generic suggestion pulled from the glossary or "typical" business
questions instead of what was actually fetched).

## No mutating actions (by design)

The datasource agent only reads: semantic model schema lookups and DAX
queries. There's no tool anywhere in this build that changes state in Power
BI (or elsewhere), so there's
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

Every node function and tool in this codebase is `async def`, and
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
  `await` a network call (Power BI REST/MCP calls, MSAL token acquisition) -
  that's I/O-bound work, so `await`ing it directly is correct and doesn't
  block the event loop. The sandbox's code execution
  (`clients/sandbox/executor.py`)
  is different: it's CPU-bound (it runs arbitrary code, which
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
to support this - none of the node functions' own prompt-building or model
calls, since `astream_events` already exists on the compiled graph object
and needs nothing from the call sites beneath it:

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
  (the node's own runnable, the model call inside it, etc. - LangChain's
  event machinery calls any `Runnable` invocation a "chain" event
  regardless of what it is), all tagged with the same `langgraph_node`
  metadata. `chat_stream` only treats the one whose own name matches the
  node name as "this node started" - filtering by `node not in seen_nodes`
  instead (i.e. only the *first* visit) would be wrong, since a node like
  `supervisor` legitimately runs more than once in a turn and each visit
  should report status.
- **Token-level streaming needs zero node changes, but does need a real
  `_stream`/`_astream` on the model.** Every node function here calls
  `llm.ainvoke()`, never `.astream()` - yet `on_chat_model_stream` events still carry
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
- **A long silent stretch gets a heartbeat, not nothing.** A specialist's
  own tool call (e.g. the analysis agent's sandbox) can legitimately run
  for tens of seconds between events. Left silent, some intermediate proxy
  or gateway between the client and this server can treat that as an idle
  connection and drop it - the client only ever sees this as a raw network
  failure (Safari's `fetch()` surfaces it as `TypeError: Load failed`, not
  as this app's own `"error"` event), so it looks like a bug in the
  assistant rather than in the transport. `chat_stream` consumes
  `astream_events` from a background task over a queue instead of directly,
  so the main generator loop can fall back to `asyncio.wait_for`'s timeout
  and emit a bare `: keep-alive` comment line every
  `_HEARTBEAT_INTERVAL_SECONDS` whenever nothing else has fired - invisible
  to any spec-compliant SSE parser (and to `app/web.py`'s own manual reader,
  which only ever looks for `"data: "` lines), so no client change is
  needed to tolerate it.

## Inspecting graph state (`DEBUG_GRAPH_STATE`)

`Settings.debug_graph_state` (env `DEBUG_GRAPH_STATE`, off by default) logs
exactly what each node's own return dict changed, live as the run happens,
via `app/api.py::_log_graph_update` - one line per node visit, in order.
This is LangGraph's own mechanism for the job, not a custom log format
standing in for it, though the two endpoints get there slightly
differently since they already drive a turn two different ways:

- **`POST /chat`** normally runs the turn via a single `graph.ainvoke(...)`
  call. With the flag on, it instead drives the same turn via
  `graph.astream(..., stream_mode="updates")` - LangGraph's own per-node
  diff stream - logging each `{node: update}` pair as it's yielded, then
  makes one `aget_state(config)` call once the run finishes to get the
  full merged state `_to_chat_response` needs (`stream_mode="updates"`
  only ever yields per-node diffs, never the merged whole).
- **`POST /chat/stream`** already drives the turn via `astream_events` (see
  "Streaming (SSE)" above) for its own status/tool/token events - switching
  it to `stream_mode="updates"` too would mean two competing ways of
  running the same turn, or running it twice. Instead, the flag makes the
  existing loop also match each node's `on_chain_end` event (the same
  outermost-invocation filter already used for `on_chain_start`/status) and
  log its `output` - the identical per-node payload `stream_mode="updates"`
  would give, already sitting in an event this loop was consuming anyway.

Either way, this needs no new instrumentation inside the graph itself -
no node function changes, nothing added to `OrchestratorState` - only a
call at the point where a request already has `graph`/`config` (or the
event stream) in scope, the same "for free" property `astream_events`
already gives node-level status/tool events (see "Streaming (SSE)").

## What's mocked vs. real

- **Real**: the LangGraph control flow (supervisor loop, ReAct loops,
  checkpointing), the async structure throughout (see above), streaming via
  `astream_events` (see above), the FastAPI request/response cycle, the
  sandbox's `exec()`-based code execution, and Power BI itself - REST API
  calls (`clients/powerbi/rest.py`), a remote MCP server call for
  `GetSemanticMetadata` (`clients/powerbi/mcp.py`), and Entra ID delegated
  auth (`clients/powerbi/auth.py`, `app/auth.py`, `cli.py`'s device-code
  flow) - see that auth module's docstring for why a delegated user token is
  required rather than an app-only one.
- **Mocked**: only the Python sandbox's data layer - `clients/sandbox/`
  executes code via `exec()`, but against an in-process dict of staged
  DataFrames rather than an isolated execution service.

## Retrying a dropped connection

`PBIRestClient.run_dax_query` and `PBIMcpClient.get_semantic_metadata` are
each wrapped in `utils/retry.py`'s `@retry` (3 attempts, 1s/2s backoff),
scoped only to transport-level failures - `httpx.TransportError` for the
REST client, plus `ExceptionGroup` for the MCP client (its transport runs
in an anyio task group, so a dropped connection surfaces as one - see
`_describe()` in `agents/datasource/nodes.py`). A real answer from either
service - an unknown model, a query Power BI itself rejects, a tool call
the MCP server errors on - is a plain `ValueError` outside that scope and
is never retried; retrying an answer that isn't going to change would just
delay the agent seeing it.

This answers "what happens if the connection drops mid-call": neither an
indefinite hang nor an instant failure. A few quick attempts (bounded at
~3s of added backoff total) ride out a brief blip; a real outage still
surfaces a clear `{"error": ...}` tool result within the same chat turn
(via each tool's own broad `except Exception`, unchanged by this) rather
than leaving the request hanging past what the caller would consider
"still loading."
