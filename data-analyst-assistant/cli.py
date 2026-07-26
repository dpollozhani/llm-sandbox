#!/usr/bin/env python3
"""Simple terminal chat client for the Data Analyst Assistant API.

Talks to a running `uvicorn data_analyst.app.api:app` instance over plain
HTTP (stdlib only, no extra dependency) - by default via the streaming
POST /chat/stream endpoint (same one the web UI at "/" uses), printing
status updates and the answer as it's generated; --no-stream falls back to
the plain POST /chat request/response endpoint.

Every request needs the caller's own delegated Power BI access tokens (see
`clients/powerbi/auth.py`'s module docstring on the server for why - the
server enforces sign-in for every /chat* call, not just Power BI-specific
questions). Rather than driving the server's browser-based sign-in
(app/auth.py, which needs a real browser), this CLI gets its own tokens
directly from Entra ID via the device-code flow: it prints a URL and a
short code, you enter that code in any browser (even on another device),
and this process polls until Entra ID issues the tokens - then sends them
to the server as `X-PBI-Rest-Token`/`X-PBI-Mcp-Token` headers on every
request. Tokens (and their refresh tokens) are cached locally so you're not
re-prompted every run.

Usage:
    python cli.py [--url http://localhost:8000] [--no-stream]
        [--tenant-id ...] [--client-id ...]

`--tenant-id`/`--client-id` default to the ENTRA_TENANT_ID/ENTRA_CLIENT_ID
env vars - the same Entra ID app registration the server uses (it must have
"Allow public client flows" enabled for this device-code flow to work
alongside the server's own confidential-client browser flow).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path

# Mirrors clients/powerbi/auth.py's scope constants - duplicated rather than
# imported so this file stays stdlib-only (no dependency on the installed
# `data_analyst` package, or its `msal`/`mcp` dependencies, just to run the CLI).
PBI_REST_SCOPE = "https://analysis.windows.net/powerbi/api/.default"
PBI_MCP_SCOPE = "https://api.fabric.microsoft.com/.default"

_TOKEN_CACHE_PATH = Path.home() / ".cache" / "data-analyst-assistant" / "pbi_tokens.json"
_EXPIRY_MARGIN_SECONDS = 60


def _post_form(url: str, params: dict) -> dict:
    data = urllib.parse.urlencode(params).encode()
    request = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read())


def _device_code_flow(tenant_id: str, client_id: str, scope: str) -> dict:
    """Interactive device-code sign-in for one resource `scope`. Returns the
    token response dict (access_token, refresh_token, expires_in, ...)."""
    start = _post_form(
        f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/devicecode",
        {"client_id": client_id, "scope": f"{scope} offline_access"},
    )
    if "device_code" not in start:
        raise RuntimeError(f"Could not start device sign-in: {start.get('error_description', start)}")

    print(f"\n{start['message']}\n")
    interval = start.get("interval", 5)
    deadline = time.monotonic() + start.get("expires_in", 900)

    while time.monotonic() < deadline:
        time.sleep(interval)
        result = _post_form(
            f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": client_id,
                "device_code": start["device_code"],
            },
        )
        if "access_token" in result:
            return result
        error = result.get("error")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
            continue
        raise RuntimeError(f"Sign-in failed: {result.get('error_description', result)}")
    raise RuntimeError("Sign-in timed out - please try again")


def _refresh(tenant_id: str, client_id: str, refresh_token: str, scope: str) -> dict:
    return _post_form(
        f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
        {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh_token,
            "scope": f"{scope} offline_access",
        },
    )


def _load_cache() -> dict:
    if _TOKEN_CACHE_PATH.exists():
        return json.loads(_TOKEN_CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    _TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TOKEN_CACHE_PATH.write_text(json.dumps(cache))


def _get_token(cache: dict, key: str, scope: str, tenant_id: str, client_id: str) -> str:
    """Return a valid access token for `scope`, using the cache (refreshing
    silently if possible), or falling back to an interactive device-code
    sign-in - once per (tenant, client, scope), not once per chat request."""
    entry = cache.get(key)
    if entry and entry.get("tenant_id") == tenant_id and entry.get("client_id") == client_id:
        if entry["expires_at"] > time.time() + _EXPIRY_MARGIN_SECONDS:
            return entry["access_token"]
        refreshed = _refresh(tenant_id, client_id, entry["refresh_token"], scope)
        if "access_token" in refreshed:
            entry.update(
                access_token=refreshed["access_token"],
                refresh_token=refreshed.get("refresh_token", entry["refresh_token"]),
                expires_at=time.time() + refreshed.get("expires_in", 3600),
            )
            cache[key] = entry
            return entry["access_token"]
        # Refresh token no longer valid (revoked/expired) - fall through to a fresh interactive sign-in.

    result = _device_code_flow(tenant_id, client_id, scope)
    cache[key] = {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "access_token": result["access_token"],
        "refresh_token": result["refresh_token"],
        "expires_at": time.time() + result.get("expires_in", 3600),
    }
    return cache[key]["access_token"]


def get_pbi_headers(tenant_id: str, client_id: str) -> dict[str, str]:
    """Signs in (or reuses/refreshes a cached sign-in) for both Power BI
    resources this app needs, returning the headers to send on every
    request. May prompt for two separate interactive device-code sign-ins
    the first time (one per resource) - see the module docstring."""
    cache = _load_cache()
    rest_token = _get_token(cache, "rest", PBI_REST_SCOPE, tenant_id, client_id)
    mcp_token = _get_token(cache, "mcp", PBI_MCP_SCOPE, tenant_id, client_id)
    _save_cache(cache)
    return {"X-PBI-Rest-Token": rest_token, "X-PBI-Mcp-Token": mcp_token}


def post_chat(base_url: str, message: str, thread_id: str | None, headers: dict[str, str]) -> dict:
    return _request(base_url, "/chat", message, thread_id, headers)


def _request(base_url: str, path: str, message: str, thread_id: str | None, headers: dict[str, str]):
    body: dict = {"message": message}
    if thread_id:
        body["thread_id"] = thread_id
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    return urllib.request.urlopen(request)


def stream_chat(base_url: str, message: str, thread_id: str | None, headers: dict[str, str]) -> Iterator[dict]:
    """Yields each parsed SSE `data:` payload from POST /chat/stream, in
    arrival order - the same events app/web.py's JS parses."""
    with _request(base_url, "/chat/stream", message, thread_id, headers) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if line.startswith("data: "):
                yield json.loads(line[len("data: ") :])


def run_streaming(base_url: str, message: str, thread_id: str | None, headers: dict[str, str]) -> str | None:
    printed_prefix = False
    reply = None
    new_thread_id = thread_id
    for event in stream_chat(base_url, message, thread_id, headers):
        kind = event["type"]
        if kind == "status" and not printed_prefix:
            print(f"... {event['message']}")
        elif kind == "tool" and not printed_prefix:
            print(f"... calling {event['name']}")
        elif kind == "token":
            if not printed_prefix:
                print("assistant> ", end="", flush=True)
                printed_prefix = True
            print(event["content"], end="", flush=True)
        elif kind == "done":
            new_thread_id = event["thread_id"]
            reply = event["reply"]
            if not printed_prefix:
                label = "assistant (clarifying)" if event["status"] == "clarification_needed" else "assistant"
                print(f"{label}> {reply}", end="")
            print("\n")
        elif kind == "error":
            print(f"\nerror: {event['message']}\n")
    return new_thread_id


def run_blocking(base_url: str, message: str, thread_id: str | None, headers: dict[str, str]) -> str | None:
    with post_chat(base_url, message, thread_id, headers) as response:
        result = json.loads(response.read())
    label = "assistant (clarifying)" if result["status"] == "clarification_needed" else "assistant"
    print(f"{label}> {result['reply']}\n")
    return result["thread_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000", help="base URL of the running API")
    parser.add_argument("--no-stream", action="store_true", help="use plain POST /chat instead of streaming")
    parser.add_argument("--tenant-id", default=os.environ.get("ENTRA_TENANT_ID"), help="Entra ID tenant (or $ENTRA_TENANT_ID)")
    parser.add_argument("--client-id", default=os.environ.get("ENTRA_CLIENT_ID"), help="Entra ID app client id (or $ENTRA_CLIENT_ID)")
    args = parser.parse_args()

    if not args.tenant_id or not args.client_id:
        parser.error("--tenant-id/--client-id are required (or set ENTRA_TENANT_ID/ENTRA_CLIENT_ID)")

    print(f"Data Analyst Assistant CLI - {args.url}{' (no streaming)' if args.no_stream else ''}")
    print("Signing in with Power BI access...")
    try:
        headers = get_pbi_headers(args.tenant_id, args.client_id)
    except RuntimeError as exc:
        print(f"error: {exc}")
        return
    print("Signed in. Type a message and press enter ('exit' to quit).\n")

    thread_id: str | None = None
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message.lower() in {"exit", "quit"}:
            break

        try:
            if args.no_stream:
                thread_id = run_blocking(args.url, message, thread_id, headers)
            else:
                thread_id = run_streaming(args.url, message, thread_id, headers)
        except urllib.error.URLError as exc:
            print(f"error: could not reach {args.url} ({exc})\n")


if __name__ == "__main__":
    main()
