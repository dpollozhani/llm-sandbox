"""Browser sign-in: Entra ID (Azure AD) OAuth authorization-code flow.

This app is a confidential OAuth client (it holds `entra_client_secret`).
`/auth/login` redirects to Microsoft, requesting consent for both Power BI
resources a chat turn might need (`PBI_REST_SCOPE` up front,
`PBI_MCP_SCOPE` via `extra_scopes_to_consent` so the one consent screen
covers both - see `TokenBroker.get_token` for how the second resource's
token is later minted silently from the same refresh token).
`/auth/callback` exchanges the resulting code for tokens and stores the
resulting MSAL token cache server-side, keyed by an opaque session id kept
in a cookie - the cookie itself never holds a token, only that lookup key,
mirroring how `clients/sandbox/client.py` keys its per-session store
(process-local; lost on restart).

`cli.py` doesn't use any of this - it gets its own tokens directly via a
device-code flow and sends them as request headers instead. See
`app/api.py::get_pbi_tokens` for how the two paths converge.
"""
from __future__ import annotations

import secrets

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from data_analyst.clients.powerbi.auth import PBI_MCP_SCOPE, PBI_REST_SCOPE, TokenBroker, build_msal_app
from data_analyst.config.settings import Settings, get_settings

router = APIRouter(prefix="/auth", tags=["auth"])

SESSION_COOKIE = "das_session"

# Process-local, mirroring clients/sandbox/client.py's per-session registry.
_sessions: dict[str, str] = {}  # session_id -> serialized MSAL token cache
_pending_states: dict[str, str] = {}  # oauth "state" -> session_id


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
    state = secrets.token_urlsafe(16)
    _pending_states[state] = session_id

    app_ = build_msal_app(settings)
    auth_url = app_.get_authorization_request_url(
        scopes=[PBI_REST_SCOPE],
        state=state,
        redirect_uri=settings.entra_redirect_uri,
        extra_scopes_to_consent=[PBI_MCP_SCOPE],
    )
    response = RedirectResponse(auth_url)
    response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax")
    return response


@router.get("/callback")
async def callback(
    request: Request, code: str | None = None, state: str | None = None, error_description: str | None = None
) -> RedirectResponse:
    if error_description:
        raise HTTPException(status_code=400, detail=error_description)
    if not code or not state or state not in _pending_states:
        raise HTTPException(status_code=400, detail="Invalid or expired sign-in attempt")

    session_id = _pending_states.pop(state)
    settings = get_settings()
    broker = TokenBroker(settings)
    try:
        await broker.redeem_code(code)
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
