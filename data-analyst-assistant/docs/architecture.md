# Architecture

## Shape

```
FastAPI (app/api.py)
        |
        v
 orchestrator graph  (checkpointed - this is the unit of persistence/resume)
   START -> supervisor -> {datasource | analysis} -> supervisor -> ... -> respond -> END
        |                        |            |
        |             (delegates, one task at a time)
        v                        v            v
   Route(next=...)      datasource graph   analysis graph
                         agent<->tools loop  agent<->tools loop
                         (PBI MCP/REST)      (Python sandbox)
```

- **`agents/orchestrator`**: a supervisor loop. `supervisor` asks the model
  for a structured `Route` decision (`datasource` / `analysis` / `respond`);
  the corresponding node runs, then control returns to `supervisor` for the
  next decision, up to `MAX_TURNS`, until it routes to `respond`.
- **`agents/datasource`** and **`agents/analysis`**: each is an independent,
  small ReAct-style `StateGraph` - an `agent` node bound to a scoped set of
  tools via `bind_tools`, and a `ToolNode`, looping through
  `tools_condition` until the model answers without calling a tool.
- **`agents/common`**: the `ChatState` shape (`messages` with LangGraph's
  `add_messages` reducer) shared by all three graphs, and the small models
  (`ApprovalRequest`, `AgentResult`) exchanged between them.
- **`clients/`**: everything that would talk to a real system in
  production - Power BI (MCP + REST + auth), the code sandbox, and the LLM
  provider. Agent code never imports a provider SDK directly; it only calls
  into `clients/`.

## How delegation works

`agents/orchestrator/nodes.py`'s `_run_specialist` seeds a **fresh** child
conversation from only the current task message (not the orchestrator's full
history), builds the specialist's compiled subgraph, and invokes it. This is
a deliberate choice over LangGraph's native "add a compiled graph directly as
a node" support: every graph here shares the same `messages`/`ChatState`
convention, so a native subgraph node would silently pass the orchestrator's
*entire* transcript into the specialist and merge the specialist's *entire*
transcript back - the opposite of "delegate one scoped task, get back one
summary." Manual invocation from a plain node function is what makes that
translation possible.

## How the human-approval interrupt propagates

`agents/datasource/nodes.py`'s `pbi_rest_trigger_dataset_refresh` tool calls
`langgraph.types.interrupt()` before "calling" the REST API. Only the
**orchestrator** graph is compiled with a checkpointer
(`agents/orchestrator/graph.py`); the datasource/analysis subgraphs are not.
That's enough for the pause to work correctly: a node function's call to
`child_graph.invoke(seed)` (no `config=` passed) picks up LangChain's
*ambient* `RunnableConfig` - including the orchestrator's checkpointer and
`thread_id` - so the interrupt bubbles up through the child graph, through
`_run_specialist`, and out of `orchestrator.invoke()`/`ainvoke()` as
`result["__interrupt__"]`, exactly as if it had been raised directly inside
the orchestrator.

Two correctness rules fall out of this:

1. **Never wrap a specialist's `.invoke()` call in a broad `except
   Exception`** (or apply `utils/retry.py`'s `@retry` to a node function that
   calls one). `interrupt()` works by raising `GraphInterrupt`, which *is* a
   plain `Exception` subclass - a broad catch silently swallows the pause.
2. **The orchestrator node function reruns from its own top on resume.** From
   the parent graph's point of view, `_run_specialist` is one atomic task;
   when you resume with `Command(resume=...)`, LangGraph re-enters that node
   function from scratch. The child graph's own already-completed steps are
   *not* redone (this is tracked transparently via the same ambient
   checkpointer), but any plain Python before the interrupt point inside that
   task does re-run. Keep that code cheap and side-effect-free.

## FastAPI pause/resume

`app/api.py`'s `POST /chat` returns `status: "approval_required"` plus the
interrupt payload instead of a reply when the graph pauses. The caller
resumes via `POST /chat/{thread_id}/approve`, which first calls
`graph.aget_state(config)` to validate the thread is actually paused
(`404` if the thread is unknown, `409` if it isn't currently awaiting
approval) before resuming with `Command(resume=body.approved)`.

## What's mocked vs. real

- **Real**: the LangGraph control flow (supervisor loop, ReAct loops,
  checkpointing, interrupts), the FastAPI request/response cycle, the
  sandbox's `exec()`-based code execution.
- **Mocked**: Power BI MCP/REST calls (`clients/powerbi/`, backed by
  `config/semantic_models.yaml` and in-memory fake tables) and Azure AD auth
  (`clients/powerbi/auth.py`). Swapping these for real integrations only
  touches `clients/`, not the agents or graphs.
