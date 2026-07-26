#!/usr/bin/env python3
"""Simple terminal chat client for the Data Analyst Assistant API.

Talks to a running `uvicorn data_analyst.app.api:app` instance over plain
HTTP (stdlib only, no extra dependency) - by default via the streaming
POST /chat/stream endpoint (same one the web UI at "/" uses), printing
status updates and the answer as it's generated; --no-stream falls back to
the plain POST /chat request/response endpoint.

Usage:
    python cli.py [--url http://localhost:8000] [--no-stream]
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from collections.abc import Iterator


def post_chat(base_url: str, message: str, thread_id: str | None) -> dict:
    return _request(base_url, "/chat", message, thread_id)


def _request(base_url: str, path: str, message: str, thread_id: str | None):
    body: dict = {"message": message}
    if thread_id:
        body["thread_id"] = thread_id
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request)


def stream_chat(base_url: str, message: str, thread_id: str | None) -> Iterator[dict]:
    """Yields each parsed SSE `data:` payload from POST /chat/stream, in
    arrival order - the same events app/web.py's JS parses."""
    with _request(base_url, "/chat/stream", message, thread_id) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if line.startswith("data: "):
                yield json.loads(line[len("data: ") :])


def run_streaming(base_url: str, message: str, thread_id: str | None) -> str | None:
    printed_prefix = False
    reply = None
    new_thread_id = thread_id
    for event in stream_chat(base_url, message, thread_id):
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


def run_blocking(base_url: str, message: str, thread_id: str | None) -> str | None:
    with post_chat(base_url, message, thread_id) as response:
        result = json.loads(response.read())
    label = "assistant (clarifying)" if result["status"] == "clarification_needed" else "assistant"
    print(f"{label}> {result['reply']}\n")
    return result["thread_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000", help="base URL of the running API")
    parser.add_argument("--no-stream", action="store_true", help="use plain POST /chat instead of streaming")
    args = parser.parse_args()

    print(f"Data Analyst Assistant CLI - {args.url}{' (no streaming)' if args.no_stream else ''}")
    print("Type a message and press enter ('exit' to quit).\n")

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
                thread_id = run_blocking(args.url, message, thread_id)
            else:
                thread_id = run_streaming(args.url, message, thread_id)
        except urllib.error.URLError as exc:
            print(f"error: could not reach {args.url} ({exc})\n")


if __name__ == "__main__":
    main()
