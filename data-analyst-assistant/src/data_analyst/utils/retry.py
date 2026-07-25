"""A small retry decorator for client calls that talk to external services
(Power BI, the sandbox). Mocked clients don't fail, but real ones will."""
from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry(attempts: int = 3, backoff_seconds: float = 0.5, exceptions: tuple[type[Exception], ...] = (Exception,)):
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            last_exc: Exception | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:  # noqa: BLE001 - re-raised after retries
                    last_exc = exc
                    if attempt < attempts:
                        time.sleep(backoff_seconds * attempt)
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator
