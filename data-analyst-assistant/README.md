# Data Analyst Assistant

A simplified, multi-agent rebuild of a data-analyst assistant (Power BI +
code execution) using only **LangChain** (tools, chat model abstraction) and
**LangGraph** (agent loops, checkpointed memory, session-scoped state), served
over **FastAPI** - built to see what that stack looks like in practice
instead of the Azure AI SDK's Agents Service.

The Power BI agent is read-only (metadata + structured queries only, no
write/admin actions), and every query is a validated, structurally-built
`SUMMARIZECOLUMNS(...)`/`ROW(...)` call - never free-form DAX text from the
model. Both
the supervisor and the specialist agents can ask a clarifying question
instead of guessing - the supervisor for broad "which specialist even
handles this" uncertainty, a specialist itself for narrower ambiguity (which
columns/filters, or which computation) it only discovers mid-task, which
skips an extra supervisor round-trip. Data already fetched in a conversation
is cached per-session and reused by follow-up questions instead of
triggering a new fetch. Every node, chain, tool, and client call is async
end to end, not just the FastAPI endpoint - which is also what makes
`POST /chat/stream` possible: live status updates ("Delegating to data
source agent...") and the final answer typed out token by token over
Server-Sent Events,
without changing a single chain. See
[`docs/decisions/0001-langchain-langgraph.md`](docs/decisions/0001-langchain-langgraph.md)
for the concept-by-concept mapping, and [`docs/architecture.md`](docs/architecture.md)
for how the pieces fit together.

`clients/powerbi/rest.py` calls the Power BI REST API, and
`clients/powerbi/mcp.py` calls the official remote Power BI MCP server's
`GetSemanticMetadata` tool. Both require the caller's own **delegated**
Entra ID access token, never an app-only service-principal one -
row-level security depends on whose identity the call runs as, and Power
BI's `executeQueries` rejects a service-principal token outright on any
dataset with RLS configured. Despite the MCP server being hosted at an
`api.fabric.microsoft.com` URL, it authenticates against the same Entra
resource as the classic REST API, so one scope/one sign-in covers both - see
[`clients/powerbi/auth.py`](src/data_analyst/clients/powerbi/auth.py)'s
module docstring for all of this. So every `/chat` call needs a signed-in user: the
browser flow (`app/auth.py`) is a PKCE authorization-code sign-in (public
client, no client secret anywhere - delegated permissions don't require
one), and `cli.py` gets its own tokens via a device-code flow. The Python sandbox
(`clients/sandbox/`) is still in-process rather than an isolated service -
see "What's simplified" below. `LLM_PROVIDER` (`anthropic` or
`azure_openai`, with the matching API key) is also required - there's no
offline/no-key fallback.

## Quickstart

You'll need one Entra ID (Azure AD) app registration with:
- **Allow public client flows** enabled - this app is a public client (PKCE)
  everywhere, no client secret to configure or rotate
- `http://localhost:8000/auth/callback` registered as a redirect URI under
  the **"Mobile and desktop applications"** platform - not "Web" (Entra
  treats that as confidential-client-only and rejects a secret-less token
  exchange, `AADSTS7000218`), and not "Single-page application" either
  (Entra requires an SPA redirect URI's code to be redeemed via a
  cross-origin browser `fetch()`, which rejects our server-side exchange
  with `AADSTS9002327`)
- delegated API permissions for the Power BI Service, and for the remote
  Power BI MCP server if you use it - with admin consent granted

```bash
pip install -e ".[dev]"
cp .env.example .env   # set LLM_PROVIDER + API key, and the ENTRA_* values

uvicorn data_analyst.app.api:app --reload
```

Then try it any of three ways:

- **Browser**: open <http://localhost:8000/> - a minimal, mobile-friendly
  chat page (`app/web.py`, no build step, no extra dependency) that prompts
  you to sign in with Microsoft first, then streams live status and the
  answer token by token via `POST /chat/stream`.
- **Terminal**: `python cli.py --tenant-id ... --client-id ...` (or set
  `$ENTRA_TENANT_ID`/`$ENTRA_CLIENT_ID`) - a small interactive chat client
  (stdlib-only), streaming by default; `--no-stream` for plain
  request/response. Prints a device-code sign-in URL the first time, then
  caches the token locally and refreshes it silently after that.
- **curl**: needs the same delegated token as a header, so it's only
  practical once you already have one (e.g. copied from `cli.py`'s token
  cache at `~/.cache/data-analyst-assistant/pbi_token.json`, or a debugger
  breakpoint in `app/auth.py`):
  ```bash
  curl -s localhost:8000/chat -X POST -H 'content-type: application/json' \
    -H 'X-PBI-Token: ...' \
    -d '{"message": "what can you do?"}'
  ```

To try it from your phone without deploying anywhere: run
`uvicorn data_analyst.app.api:app --host 0.0.0.0`, add
`http://<your-laptop's-LAN-IP>:8000/auth/callback` as another redirect URI
on the app registration, and open `http://<your-laptop's-LAN-IP>:8000/`
from your phone on the same Wi-Fi.

## Deploy (Render)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/dpollozhani/llm-sandbox)

For a real public URL (e.g. to open from your phone off Wi-Fi), this repo
has a Render Blueprint (`/render.yaml` at the repo root - this is a
monorepo, so it points at `data-analyst-assistant/` via `rootDir`) that
builds `deploy/docker/Dockerfile`.

- **Button above**: works once `render.yaml` is on the repo's default
  branch (Render's Blueprint button deploys from the default branch, not
  an arbitrary one) - merge this first, or use the manual path below to
  deploy this branch directly.
- **Manual (works right now, any branch)**: Render dashboard -> New -> Web
  Service -> connect `dpollozhani/llm-sandbox`, pick this branch -> Runtime
  "Docker" -> Root Directory `data-analyst-assistant` -> Dockerfile Path
  `./deploy/docker/Dockerfile` -> Docker Build Context Directory `.` ->
  Health Check Path `/health` -> free plan.

Either way, **the service won't start without `LLM_PROVIDER` and the
matching `ANTHROPIC_API_KEY`/`AZURE_OPENAI_*` vars, plus `ENTRA_TENANT_ID`,
`ENTRA_CLIENT_ID`, and `ENTRA_REDIRECT_URI`, set in the service's
Environment tab** - there's no default provider or Power BI sign-in to fall
back to. Set `ENTRA_REDIRECT_URI` to this service's public URL +
`/auth/callback` (e.g.
`https://data-analyst-assistant.onrender.com/auth/callback`) and register
that exact URI, under the "Mobile and desktop applications" platform (not
"Web", not "Single-page application" - see
[`clients/powerbi/auth.py`](src/data_analyst/clients/powerbi/auth.py)'s
module docstring for why), on the Entra ID app registration too. The free
plan also spins the instance down after ~15 minutes of inactivity (30-60s
cold start on the next request).

If you just want to see the full multi-agent flow (routing, tool calls
across both specialists) without a Power BI tenant or LLM provider at all,
read `tests/e2e/test_api_chat.py`, which drives it end-to-end with a
scripted model and a fake Power BI client - no sign-in, no API key.

## Layout

```
src/data_analyst/
  app/           FastAPI: api.py (routes incl. /chat/stream SSE, via astream_events), auth.py (Entra ID browser sign-in), lifespan.py (builds the graph once), dependencies.py
  agents/
    orchestrator/  supervisor loop: routes to a specialist, responds, or asks for clarification
    datasource/    Power BI specialist (PBI MCP + PBI REST tools, structured DAX queries only)
    analysis/      Python sandbox specialist
    common/        shared state shape (messages + session_id + delegated PBI token) + models used across all three
  clients/       Power BI (REST + MCP calls, delegated auth only), sandbox (mocked, session-bound), and LLM provider clients
  config/        env-driven settings + the Power BI catalog (model name -> dataset id mapping)
  telemetry/     logging/tracing/metrics stand-ins
  utils/         small shared helpers
tests/
  unit/          client layer only (no LangGraph) - Power BI REST calls mocked via httpx.MockTransport, MCP via a fake ClientSession
  integration/   one subgraph / one orchestrator node at a time, scripted models + a fake Power BI client
  e2e/           the FastAPI app end-to-end, across both specialists, with fake Power BI clients and a stubbed sign-in
deploy/          Dockerfile + docker-compose, an Azure DevOps pipeline
infrastructure/  illustrative Bicep/Terraform for a Container App + Azure OpenAI
docs/            architecture, per-agent reference, decision records
```

## Testing

```bash
pytest
```

All tests run offline - no Power BI tenant, no LLM API key, no network.
Power BI's HTTP/MCP clients are exercised against a fake transport/session
in `tests/unit/`, and the datasource/orchestrator graphs are tested against
injected fake `PBIRestClient`/`PBIMcpClient` instances
(`build_datasource_graph(llm, mcp_client=..., rest_client=...)`) rather than
the network calls, with `app/api.py::get_pbi_tokens` overridden the same
way `get_graph` already is. Everything except `tests/e2e` is `async def`
(LangGraph requires `.ainvoke()` once any node is async); `tests/e2e` stays
sync because FastAPI's `TestClient` already bridges to the async app for
you.

## What's simplified vs. a real deployment

- The exact JSON shape of the remote PBI MCP server's `GetSemanticMetadata`
  response isn't pinned down in code (see `clients/powerbi/mcp.py`) - it's
  passed through to the model as-is rather than forced into a schema that
  might not match what your tenant's server actually returns.
- `config/semantic_models.yaml` maps a friendly model name (what the model
  uses in a `DaxQuerySpec`) to a dataset id - populate it with your own
  tenant's values.
- The sandbox (`clients/sandbox/`) really does `exec()` pandas code, but in
  a minimally-restricted namespace rather than a network-isolated worker -
  don't point it at untrusted input as-is.
- Sign-in sessions (`app/auth.py`), `InMemorySaver` (conversation state,
  `app/lifespan.py`), and the session-bound data store
  (`clients/sandbox/client.py`) are all process-local; swap in a shared
  session store, a Postgres/Redis-backed checkpointer, and a shared
  cache/object store, respectively, for anything that needs to survive a
  restart or run across multiple instances.
