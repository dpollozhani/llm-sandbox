"""Entra ID (Azure AD) auth for Power BI: delegated (per-user) tokens via
MSAL, not app-only client-credentials.

Both the Power BI REST API's `executeQueries` and the remote PBI MCP
server's `GetSemanticMetadata` enforce row-level security using the calling
user's own identity - `executeQueries` rejects a service-principal token
outright (401) on any dataset with RLS configured, and the MCP server is
documented as delegated-only. So there is no app-only fallback here: every
call needs a signed-in user's delegated access token.

This app is a **public** OAuth client everywhere - the browser sign-in flow
(`app/auth.py`) and `cli.py`'s device-code flow both authenticate the user
without any client secret, via PKCE (MSAL's `initiate_auth_code_flow`/
`acquire_token_by_auth_code_flow` generate and validate the PKCE
code_verifier/challenge automatically). Delegated permissions don't require
a confidential client - the client secret only proves the *app's* identity
during a code exchange, it has nothing to do with whether the resulting
token represents the user's own permissions (delegated) or the app's own
(application/service-principal). Since there's no server-held secret to
protect here, a public client is simpler and one less thing to configure/
rotate.

This does require the Entra ID app registration's `ENTRA_REDIRECT_URI` to be
registered under a public-client platform - either **"Single-page
application"** or **"Mobile and desktop applications"** both work - not
"Web": Entra classifies a "Web" redirect URI as belonging to a confidential
client and will reject a secret-less token exchange against it
(`AADSTS7000218`) regardless of what MSAL class the code uses. "Allow public
client flows" also needs to be enabled (already required for `cli.py`'s
device-code flow).

`TokenBroker` wraps one user's MSAL token cache and mints scoped access
tokens from it, refreshing silently via the cached refresh token. Two
different callers build one:
- `app/auth.py`'s browser sign-in flow (stores each user's serialized cache
  server-side, keyed by a session cookie).
- `cli.py`'s device-code flow builds tokens itself and sends them as
  request headers instead of going through a TokenBroker at all - see
  `app/api.py::get_pbi_tokens`.
"""
from __future__ import annotations

import asyncio

import msal

from data_analyst.config.settings import Settings

PBI_REST_SCOPE = "https://analysis.windows.net/powerbi/api/.default"
PBI_MCP_SCOPE = "https://api.fabric.microsoft.com/.default"


def build_msal_app(
    settings: Settings, token_cache: msal.SerializableTokenCache | None = None
) -> msal.PublicClientApplication:
    return msal.PublicClientApplication(
        client_id=settings.entra_client_id,
        authority=f"https://login.microsoftonline.com/{settings.entra_tenant_id}",
        token_cache=token_cache,
    )


class TokenBroker:
    def __init__(self, settings: Settings, serialized_cache: str | None = None) -> None:
        self._settings = settings
        self.cache = msal.SerializableTokenCache()
        if serialized_cache:
            self.cache.deserialize(serialized_cache)
        self._app = build_msal_app(settings, self.cache)

    def serialize(self) -> str | None:
        """The cache's serialized state, if it changed since it was loaded
        (or created) - None if nothing changed, so a caller like
        `app/auth.py`'s session store can skip rewriting an unchanged
        session on every request."""
        return self.cache.serialize() if self.cache.has_state_changed else None

    def start_login(self, scope: str) -> dict:
        """Begin a PKCE authorization-code flow for one resource `scope`.

        `.default` scopes can't be combined across resources in a single
        authorization request - Entra rejects it outright
        (`AADSTS70011: ... static scope limit exceeded`), since `.default`
        already means "every statically configured permission for *this*
        resource", and that only makes sense for one resource at a time.
        So `PBI_REST_SCOPE` and `PBI_MCP_SCOPE` each need their own
        interactive consent round (see `app/auth.py`'s `/auth/login?
        resource=`) - there's no single-screen shortcut for two resources
        the way there is for multiple *named* scopes on the same resource.

        Returns the flow dict `app/auth.py` must store server-side (keyed
        by its own "state") until the matching `/auth/callback` request
        arrives - this is a local, synchronous call, no network I/O."""
        return self._app.initiate_auth_code_flow(scopes=[scope], redirect_uri=self._settings.entra_redirect_uri)

    async def redeem_code(self, flow: dict, auth_response: dict) -> None:
        """Complete a PKCE flow started by `start_login`, exchanging the
        authorization code in `auth_response` (the raw `/auth/callback`
        query params) for tokens, populating this broker's cache. Raises
        RuntimeError on failure (including a state/CSRF mismatch, which MSAL
        raises as ValueError - normalized here to one exception type)."""
        try:
            result = await asyncio.to_thread(self._app.acquire_token_by_auth_code_flow, flow, auth_response)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if "access_token" not in result:
            raise RuntimeError(result.get("error_description") or result.get("error") or "Sign-in failed")

    async def get_token(self, scope: str) -> str:
        """Return a fresh delegated access token for `scope`, silently
        refreshing via the cached refresh token. Raises RuntimeError if
        there's no signed-in account, or this user hasn't consented to
        `scope` yet (needs another interactive `/auth/login` round)."""
        accounts = self._app.get_accounts()
        if not accounts:
            raise RuntimeError("Not signed in")
        result = await asyncio.to_thread(self._app.acquire_token_silent, [scope], account=accounts[0])
        if not result or "access_token" not in result:
            raise RuntimeError(f"Sign-in required for scope '{scope}'")
        return result["access_token"]
