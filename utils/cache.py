"""Tiny in-process TTL cache + a threadpool map.

No external services (Redis, etc.), so it works on a single free-tier instance.
The cache is process-local and best-effort: it speeds up repeated coordinates /
queries while a worker stays warm and simply misses after a restart. Failures
(falsy results) are NOT cached by default, so a transient outage isn't
remembered. `parallel_map` runs blocking `requests` calls concurrently to cut
end-to-end latency.
"""

from __future__ import annotations

import functools
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable

CACHE_TTL = float(os.getenv("CACHE_TTL_SECONDS", "86400"))
CACHE_MAX = int(os.getenv("CACHE_MAX_ENTRIES", "2048"))
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))

_lock = threading.Lock()
_store: dict[Any, tuple[float, Any]] = {}


def _evict_if_needed() -> None:
    if len(_store) < CACHE_MAX:
        return
    # Drop the oldest ~10% of entries.
    oldest = sorted(_store, key=lambda k: _store[k][0])[: max(1, CACHE_MAX // 10)]
    for k in oldest:
        _store.pop(k, None)


def cache_get(key: Any) -> Any:
    """Return a cached value for `key`, or None if missing/expired."""
    now = time.monotonic()
    with _lock:
        hit = _store.get(key)
        if hit and now - hit[0] < CACHE_TTL:
            return hit[1]
    return None


def cache_set(key: Any, value: Any) -> None:
    with _lock:
        _evict_if_needed()
        _store[key] = (time.monotonic(), value)


def clear() -> None:
    with _lock:
        _store.clear()


def cached(func: Callable | None = None, *, cache_empty: bool = False):
    """Memoize a function's result by its arguments (TTL-bounded).

    `cache_empty=False` (default) means falsy results are not stored, so a
    fail-soft None/[]/{} from a network hiccup won't be cached.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (fn.__qualname__, args, tuple(sorted(kwargs.items())))
            found = cache_get(key)
            if found is not None:
                return found
            value = fn(*args, **kwargs)
            if value or cache_empty:
                cache_set(key, value)
            return value

        return wrapper

    return decorator(func) if func is not None else decorator


def parallel_map(
    func: Callable, items: Iterable, workers: int | None = None
) -> list:
    """Run `func` over `items` concurrently (threads; good for blocking I/O).

    `workers` caps concurrency below PARALLEL_WORKERS for jobs that are heavier
    than a single HTTP request (e.g. a whole per-image pipeline), so one request
    cannot saturate a small instance.
    """
    items = list(items)
    if not items:
        return []
    if len(items) == 1:
        return [func(items[0])]
    limit = min(workers or PARALLEL_WORKERS, PARALLEL_WORKERS, len(items))
    with ThreadPoolExecutor(max_workers=max(1, limit)) as ex:
        return list(ex.map(func, items))
