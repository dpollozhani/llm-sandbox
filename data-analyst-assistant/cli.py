#!/usr/bin/env python3
"""Simple terminal chat client for the Data Analyst Assistant API.

Talks to a running `uvicorn data_analyst.app.api:app` instance over plain
HTTP (stdlib only, no extra dependency) - the same /chat endpoint the web
UI at "/" uses.

Usage:
    python cli.py [--url http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def post_chat(base_url: str, message: str, thread_id: str | None) -> dict:
    body: dict = {"message": message}
    if thread_id:
        body["thread_id"] = thread_id
    request = urllib.request.Request(
        f"{base_url}/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000", help="base URL of the running API")
    args = parser.parse_args()

    print(f"Data Analyst Assistant CLI - {args.url}")
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
            result = post_chat(args.url, message, thread_id)
        except urllib.error.URLError as exc:
            print(f"error: could not reach {args.url} ({exc})\n")
            continue

        thread_id = result["thread_id"]
        label = "assistant (clarifying)" if result["status"] == "clarification_needed" else "assistant"
        print(f"{label}> {result['reply']}\n")


if __name__ == "__main__":
    main()
