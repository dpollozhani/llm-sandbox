# Data Analyst Assistant

A simplified, multi-agent rebuild of a data-analyst assistant (Power BI +
code execution) using only **LangChain** (tools, chat model abstraction) and
**LangGraph** (agent loops, checkpointed memory, session-scoped state), served
over **FastAPI** - built to see what that stack looks like in practice
instead of the Azure AI SDK's Agents Service.

The Power BI agent is read-only (metadata + structured queries only, no
write/admin actions), and every query is a validated, structurally-built
`SUMMARIZECOLUMNS(...)` call - never free-form DAX text from the model. Both
the supervisor and the specialist agents can ask a clarifying question
instead of guessing - the supervisor for broad "which specialist even
handles this" uncertainty, a specialist itself for narrower ambiguity (which
columns/filters, or which computation) it only discovers mid-task, which
skips an extra supervisor round-trip. Data already fetched in a conversation
is cached per-session and reused by follow-up questions instead of
triggering a new fetch. Every node, chain, tool, and client call is async
end to end, not just the FastAPI endpoint. See
[`docs/decisions/0001-langchain-langgraph.md`](docs/decisions/0001-langchain-langgraph.md)
for the concept-by-concept mapping, and [`docs/architecture.md`](docs/architecture.md)
for how the pieces fit together.

The three external integrations - Power BI MCP, Power BI REST, and a Python
code-execution sandbox - are mocked (see `src/data_analyst/clients/`), so
the whole thing runs with no cloud dependency in `LLM_PROVIDER=demo` mode.

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env   # defaults to LLM_PROVIDER=demo, no keys needed

uvicorn data_analyst.app.api:app --reload
```

```bash
curl -s localhost:8000/chat -X POST -H 'content-type: application/json' \
  -d '{"message": "what can you do?"}'
```

To see the full multi-agent flow (routing, tool calls across both
specialists), set `LLM_PROVIDER=anthropic` or `LLM_PROVIDER=azure_openai` in
`.env` with a real key - or read `tests/e2e/test_api_chat.py`, which drives
the same flow end-to-end with a scripted model and needs no API key.

## Layout

```
src/data_analyst/
  app/           FastAPI: api.py (routes), lifespan.py (builds the graph once), dependencies.py
  agents/
    orchestrator/  supervisor loop: routes to a specialist, responds, or asks for clarification
    datasource/    Power BI specialist (PBI MCP + PBI REST tools, structured DAX queries only)
    analysis/      Python sandbox specialist
    common/        shared state shape (messages + session_id) + models used across all three
  clients/       Power BI (mocked, incl. the DAX query builder/validator), sandbox (mocked, session-bound), and LLM provider clients
  config/        env-driven settings + the (mocked) Power BI catalog
  telemetry/     logging/tracing/metrics stand-ins
  utils/         small shared helpers
tests/
  unit/          client layer only (no LangGraph)
  integration/   one subgraph / one orchestrator node at a time, scripted models
  e2e/           the real FastAPI app end-to-end, across both specialists
deploy/          Dockerfile + docker-compose, an Azure DevOps pipeline
infrastructure/  illustrative Bicep/Terraform for a Container App + Azure OpenAI
docs/            architecture, per-agent reference, decision records
```

## Testing

```bash
pytest
```

All 40 tests run offline with scripted fake models - no API key or network
needed. Everything except `tests/e2e` is `async def` (LangGraph requires
`.ainvoke()` once any node is async); `tests/e2e` stays sync because
FastAPI's `TestClient` already bridges to the async app for you.

## What's simplified vs. a real deployment

- Power BI MCP/REST calls are mocked in `clients/powerbi/` against
  `config/semantic_models.yaml` and a handful of in-memory fake tables; the
  DAX query engine (`clients/powerbi/dax.py`) is a small pandas-based
  stand-in, not a real semantic model, and MCP transport isn't implemented.
- The sandbox (`clients/sandbox/`) really does `exec()` pandas code, but in
  a minimally-restricted namespace rather than a network-isolated worker -
  don't point it at untrusted input as-is.
- Both `InMemorySaver` (conversation state, `app/lifespan.py`) and the
  session-bound data store (`clients/sandbox/client.py`) are process-local;
  swap in a Postgres/Redis-backed checkpointer and a shared cache/object
  store, respectively, for anything that needs to survive a restart or run
  across multiple instances.
