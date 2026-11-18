"""TTL cache for expensive graph queries — subgraph extraction and centrality."""

from __future__ import annotations

from typing import Any

from cachetools import TTLCache


class CacheService:
    """In-process TTL cache with hit/miss tracking."""

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 1000) -> None:
        self._cache: TTLCache = TTLCache(maxsize=max_entries, ttl=ttl_seconds)
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        value = self._cache.get(key)
        if value is not None:
            self._hits += 1
        else:
            self._misses += 1
        return value

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear EVERY tenant's entries. Deliberately NOT reachable from the HTTP API.

        Kept for process-local lifecycle use (``dependencies.clear_cache``) and tests.
        Anything serving an HTTP caller must use :meth:`clear_tenant`, or one tenant's
        request empties the shared singleton for all of them.
        """
        self._cache.clear()

    def clear_tenant(self, tenant_id: str) -> int:
        """Drop only the entries owned by ``tenant_id``; return how many were removed.

        Ownership is EXACT equality on the tenant segment, never a prefix or substring
        match: ``acme`` and ``acme2`` are unrelated tenants, and an ``entity_id`` may
        legitimately equal another tenant's id. See :meth:`key_tenant`.

        Keys are materialised into a list first: ``TTLCache.keys()`` is a live view and
        mutating the cache while iterating it raises.
        """
        self._require_real_tenant(tenant_id, "clear_tenant")
        owned = [key for key in list(self._cache.keys()) if self.key_tenant(key) == tenant_id]
        for key in owned:
            self._cache.pop(key, None)
        return len(owned)

    def tenant_size(self, tenant_id: str) -> int:
        """Number of live entries owned by ``tenant_id`` (exact segment match)."""
        self._require_real_tenant(tenant_id, "tenant_size")
        return sum(1 for key in list(self._cache.keys()) if self.key_tenant(key) == tenant_id)

    @staticmethod
    def _require_real_tenant(tenant_id: str, operation: str) -> None:
        """Refuse a falsy tenant id on any per-tenant operation.

        ``key_tenant`` returns ``None`` for a key with no tenant segment and ``""`` for
        a key built with an empty tenant (``centrality::pagerank`` — what the services
        emit for ``tenant_id or ""``). Both compare equal to a falsy argument, so
        ``clear_tenant(None)`` would sweep every untenanted key and ``clear_tenant("")``
        every empty-tenant key — entries that belong to nobody and that no caller is
        entitled to remove.

        No caller can reach this today (``resolve_tenant_id`` never returns a falsy
        string), and the ``str`` annotation hides the ``None`` case from ``ty``, which
        is exactly why it needs a runtime guard: it is a silent cross-tenant sweep one
        refactor away, and it fails loud instead.
        """
        if not tenant_id:
            raise ValueError(f"{operation} requires a concrete tenant id, got {tenant_id!r}")

    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._cache),
            "max_entries": self._cache.maxsize,
            "ttl_seconds": self._cache.ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(self._hits + self._misses, 1), 4),
        }

    @staticmethod
    def key_tenant(key: str) -> str | None:
        """Owning tenant of a cache key, or ``None`` if the key is not tenant-scoped.

        THE single definition of the ``<namespace>:<tenant_id>:<rest>`` grammar every
        ``make_*_key`` builder below emits, so the owner is colon-segment 1. Only that
        segment is read, which keeps an ``entity_id`` or ``algorithm`` containing
        colons harmless. Lives here, beside the builders, so the grammar cannot drift
        between the routers that parse it and the builders that produce it.

        Segment 1 identifies the owner UNAMBIGUOUSLY only because a tenant id can
        never itself contain ``:``. That is enforced twice: at the process boundary
        (``auth.middleware._TENANT_ID_RE``, which refuses the request outright) and
        here by :meth:`_tenant_segment`, so no builder can emit an ambiguous key even
        if a future caller path forgets the boundary check. Without it, tenant
        ``acme:eu`` emitted ``subgraph:acme:eu:e1:2``, which parses back as ``acme``.

        Every comparison against this value must be EXACT equality — a prefix or
        substring test makes ``acme`` the owner of ``acme2``'s entries.
        """
        segments = key.split(":")
        if len(segments) < 2:
            return None
        return segments[1]

    @staticmethod
    def _tenant_segment(tenant_id: str) -> str:
        """Validate ``tenant_id`` as a key segment, or raise.

        ``:`` is the grammar's delimiter; a tenant id containing one shifts every
        following segment and makes :meth:`key_tenant` report the wrong owner. The
        boundary check in ``auth.middleware`` rejects such ids before they reach here,
        so this is the fail-loud backstop for any non-HTTP caller.
        """
        if ":" in tenant_id:
            raise ValueError(
                f"tenant id must not contain the cache-key delimiter ':' (got {tenant_id!r}) — "
                "it would make the key parse as a different tenant"
            )
        return tenant_id

    @staticmethod
    def make_subgraph_key(tenant_id: str, entity_id: str, max_hops: int) -> str:
        return f"subgraph:{CacheService._tenant_segment(tenant_id)}:{entity_id}:{max_hops}"

    @staticmethod
    def make_centrality_key(tenant_id: str, algorithm: str) -> str:
        return f"centrality:{CacheService._tenant_segment(tenant_id)}:{algorithm}"

    @staticmethod
    def make_communities_key(tenant_id: str, algorithm: str, min_community_size: int) -> str:
        return f"communities:{CacheService._tenant_segment(tenant_id)}:{algorithm}:{min_community_size}"
