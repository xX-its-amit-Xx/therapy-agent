"""Process-local TTL cache for the idempotent retrieval tools.

UniProt, Reactome, ClinVar, and ChEMBL responses for a given gene don't
change minute-to-minute. Caching their results inside the process for a
short TTL cuts the benchmark wall-time substantially (we see the same
~10 genes queried multiple times per case across nodes), and also
reduces traffic against the upstream public APIs (good citizen behavior).

The cache is process-local because:
  - benchmark runs are short-lived (~5 min on GPT-4o)
  - LLM determinism is a stronger property if the cache is fresh per
    process (no risk of a stale entry leaking across PRs)

TTL is 300s (5 min) by default; override with `THERAPY_AGENT_CACHE_TTL`.

Usage:

    from therapy_agent.tools._cache import cached_async

    @cached_async("uniprot")
    async def my_uniprot_lookup(gene): ...
"""
from __future__ import annotations

import asyncio
import functools
import os
import time
from typing import Any, Callable, Coroutine


_DEFAULT_TTL = float(os.environ.get("THERAPY_AGENT_CACHE_TTL", "300"))


class _AsyncTTLCache:
    """Tiny async-safe TTL cache keyed by (namespace, args)."""

    def __init__(self, ttl: float = _DEFAULT_TTL) -> None:
        self.ttl = ttl
        self._store: dict[tuple, tuple[float, Any]] = {}
        self._locks: dict[tuple, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    async def get_or_compute(
        self,
        key: tuple,
        compute: Callable[[], Coroutine[Any, Any, Any]],
    ) -> Any:
        now = time.time()
        # Check cache (fast path, no lock for hot reads).
        cached = self._store.get(key)
        if cached and (now - cached[0]) < self.ttl:
            self.hits += 1
            return cached[1]
        # Slow path: acquire per-key lock to dedup concurrent computes.
        async with self._global_lock:
            lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self._store.get(key)
            if cached and (time.time() - cached[0]) < self.ttl:
                self.hits += 1
                return cached[1]
            value = await compute()
            self._store[key] = (time.time(), value)
            self.misses += 1
            return value


_GLOBAL_CACHE = _AsyncTTLCache()


def cached_async(namespace: str):
    """Decorator: cache an async function by (namespace, args)."""
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            # Build a hashable key from positional args + sorted kwargs.
            key = (namespace, args, tuple(sorted(kwargs.items())))
            return await _GLOBAL_CACHE.get_or_compute(key, lambda: fn(*args, **kwargs))
        wrapper.cache_stats = lambda: {"hits": _GLOBAL_CACHE.hits, "misses": _GLOBAL_CACHE.misses}
        return wrapper
    return deco


def cache_stats() -> dict:
    return {"hits": _GLOBAL_CACHE.hits, "misses": _GLOBAL_CACHE.misses}
