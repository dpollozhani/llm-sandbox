# Data Analyst Assistant

A simplified, multi-agent rebuild of a data-analyst assistant (Power BI +
code execution) using only **LangChain** (tools, chat model abstraction) and
**LangGraph** (agent loops, checkpointed memory), served over **FastAPI** -
built to see what that stack looks like in practice instead of the Azure AI
SDK's Agents Service. The Power BI agent is read-only (metadata + DAX
queries only, no write/admin actions). See
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
    orchestrator/  supervisor loop: routes to a specialist or responds
    datasource/    Power BI specialist (PBI MCP + PBI REST tools)
    analysis/      Python sandbox specialist
    common/        shared state shape + models used across all three
  clients/       Power BI (mocked), sandbox (mocked), and LLM provider clients
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

All 15 tests run offline with scripted fake models - no API key or network
needed.

## What's simplified vs. a real deployment

- Power BI MCP/REST calls are mocked in `clients/powerbi/` against
  `config/semantic_models.yaml` and a handful of in-memory fake tables; real
  DAX execution and MCP transport aren't implemented.
- The sandbox (`clients/sandbox/`) really does `exec()` pandas code, but in
  a minimally-restricted namespace rather than a network-isolated worker -
  don't point it at untrusted input as-is.
- Session state (`InMemorySaver` in `app/lifespan.py`) is process-local;
  swap in a Postgres/Redis-backed checkpointer for anything that needs to
  survive a restart or run across multiple instances.
