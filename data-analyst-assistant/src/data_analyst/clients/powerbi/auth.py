"""Real Entra ID (Azure AD) auth for Power BI: delegated (per-user) tokens
via MSAL, not app-only client-credentials.

Both the Power BI REST API's `executeQueries` and the remote PBI MCP
server's `GetSemanticMetadata` enforce row-level security using the calling
user's own identity - `executeQueries` rejects a service-principal token
outright (401) on any dataset with RLS configured, and the MCP server is
documented as delegated-only. So there is no app-only fallback here: every
call needs a real signed-in user's delegated access token.

`TokenBroker` wraps one user's MSAL token cache and mints scoped access
tokens from it, refreshing silently via the cached refresh token. Two
different callers build one:
- `app/auth.py`'s browser sign-in flow (this app is a confidential OAuth
  client - it holds `entra_client_secret` - and stores each user's
  serialized cache server-side, keyed by a session cookie).
- `cli.py`'s device-code flow builds tokens itself (as a public client, no
  secret) and sends them as request headers instead of going through a
  TokenBroker at all - see `app/api.py::get_pbi_tokens`.
"""
from __future__ import annotations

import asyncio

import msal

from data_analyst.config.settings import Settings

PBI_REST_SCOPE = "https://analysis.windows.net/powerbi/api/.default"
PBI_MCP_SCOPE = "https://api.fabric.microsoft.com/.default"


def build_msal_app(
    settings: Settings, token_cache: msal.SerializableTokenCache | None = None
) -> msal.ConfidentialClientApplication:
    return msal.ConfidentialClientApplication(
        client_id=settings.entra_client_id,
        client_credential=settings.entra_client_secret,
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

    async def redeem_code(self, code: str) -> None:
        """Exchange an authorization code (from `/auth/callback`) for tokens,
        populating this broker's cache. Raises RuntimeError on failure."""
        result = await asyncio.to_thread(
            self._app.acquire_token_by_authorization_code,
            code,
            scopes=[PBI_REST_SCOPE],
            redirect_uri=self._settings.entra_redirect_uri,
        )
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
