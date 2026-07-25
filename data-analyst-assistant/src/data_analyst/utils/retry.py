"""A small async retry decorator for client calls that talk to external
services (Power BI, the sandbox). Mocked clients don't fail, but real ones
will - and since those calls are I/O-bound, the whole client layer is async
(see docs/architecture.md), so this only wraps async functions."""
from __future__ import annotations

import asyncio
import functools
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


def retry(attempts: int = 3, backoff_seconds: float = 0.5, exceptions: tuple[type[Exception], ...] = (Exception,)):
    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs) -> T:
            last_exc: Exception | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as exc:  # noqa: BLE001 - re-raised after retries
                    last_exc = exc
                    if attempt < attempts:
                        await asyncio.sleep(backoff_seconds * attempt)
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator
