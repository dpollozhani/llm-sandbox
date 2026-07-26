"""Browser sign-in: Entra ID (Azure AD) OAuth authorization-code flow with
PKCE, as a public client (no client secret) - see
`clients/powerbi/auth.py`'s module docstring for why, and for the redirect
URI platform-type requirement that goes with it. One scope (`PBI_SCOPE`)
covers both the Power BI REST API and the remote MCP server - see that same
docstring - so this is a single sign-in, not one per resource.

`/auth/login` starts the flow (`TokenBroker.start_login`), redirects to
Microsoft, and stashes the returned flow dict (which carries the PKCE
verifier and expected `state`, among other things MSAL needs to complete
the exchange) server-side, keyed by its own `state` value. `/auth/callback`
looks that flow back up by the `state` query param Entra sends back,
exchanges the code for tokens (`TokenBroker.redeem_code`), and stores the
resulting MSAL token cache server-side, keyed by an opaque session id kept
in a cookie - the cookie itself never holds a token or the flow dict, only
that lookup key, mirroring how `clients/sandbox/client.py` keys its
per-session store (process-local; lost on restart).

`cli.py` doesn't use any of this - it gets its own token directly via a
device-code flow and sends it as a request header instead. See
`app/api.py::get_pbi_tokens` for how the two paths converge.
"""
from __future__ import annotations

import secrets

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from data_analyst.clients.powerbi.auth import TokenBroker
from data_analyst.config.settings import Settings, get_settings

router = APIRouter(prefix="/auth", tags=["auth"])

SESSION_COOKIE = "das_session"

# Process-local, mirroring clients/sandbox/client.py's per-session registry.
_sessions: dict[str, str] = {}  # session_id -> serialized MSAL token cache
_pending_flows: dict[str, tuple[str, dict]] = {}  # oauth "state" -> (session_id, flow)


def get_token_broker(request: Request, settings: Settings) -> TokenBroker | None:
    """The current browser session's TokenBroker, or None if not signed in."""
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id is None or session_id not in _sessions:
        return None
    return TokenBroker(settings, serialized_cache=_sessions[session_id])


def save_broker(request: Request, broker: TokenBroker) -> None:
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id is None:
        return
    serialized = broker.serialize()
    if serialized is not None:
        _sessions[session_id] = serialized


def is_signed_in(request: Request) -> bool:
    session_id = request.cookies.get(SESSION_COOKIE)
    return session_id is not None and session_id in _sessions


@router.get("/whoami")
async def whoami(request: Request) -> dict:
    return {"signed_in": is_signed_in(request)}


@router.get("/login")
async def login(request: Request) -> RedirectResponse:
    settings = get_settings()
    session_id = request.cookies.get(SESSION_COOKIE) or secrets.token_urlsafe(32)

    broker = TokenBroker(settings, serialized_cache=_sessions.get(session_id))
    flow = broker.start_login()
    _pending_flows[flow["state"]] = (session_id, flow)

    response = RedirectResponse(flow["auth_uri"])
    response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax")
    return response


@router.get("/callback")
async def callback(request: Request) -> RedirectResponse:
    params = dict(request.query_params)
    if "error" in params:
        raise HTTPException(status_code=400, detail=params.get("error_description", params["error"]))

    state = params.get("state")
    pending = _pending_flows.pop(state, None) if state else None
    if pending is None:
        raise HTTPException(status_code=400, detail="Invalid or expired sign-in attempt")
    session_id, flow = pending

    settings = get_settings()
    broker = TokenBroker(settings, serialized_cache=_sessions.get(session_id))
    try:
        await broker.redeem_code(flow, params)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _sessions[session_id] = broker.cache.serialize()
    response = RedirectResponse("/")
    response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax")
    return response


@router.post("/logout")
async def logout(request: Request) -> RedirectResponse:
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id is not None:
        _sessions.pop(session_id, None)
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response
