"""Mocked Azure AD auth for Power BI. A real client would use MSAL
(confidential client credentials flow) against `https://analysis.windows.net/powerbi/api/.default`;
here we just fabricate a bearer token with an expiry so callers exercise the
same "get a cached token, refresh if stale" shape.
"""
from __future__ import annotations

import time

from ...utils.retry import retry

_TOKEN_TTL_SECONDS = 3600


class _CachedToken:
    def __init__(self) -> None:
        self.value: str | None = None
        self.expires_at: float = 0.0


_token = _CachedToken()


@retry(attempts=3)
def get_bearer_token() -> str:
    """Return a cached (mocked) bearer token, "refreshing" it once expired."""
    now = time.monotonic()
    if _token.value is None or now >= _token.expires_at:
        _token.value = f"mock-token-{int(now)}"
        _token.expires_at = now + _TOKEN_TTL_SECONDS
    return _token.value
