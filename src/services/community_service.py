"""Community detection service — delegates to Graph Adapter."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from clients.graph_adapter import GraphAdapterClient
from log_config import get_logger
from models.topology import CommunityResponse
from services.cache import CacheService

logger = get_logger(__name__)


class CommunityService:
    def __init__(self, graph_client: GraphAdapterClient, cache: CacheService | None = None) -> None:
        self.graph_client = graph_client
        self._cache = cache
        # Per-key locks give single-flight semantics: within one verify_batch,
        # ~50 concurrent _community_score calls would otherwise each fire an
        # identical full-tenant Louvain. The first computes+caches; the rest wait
        # on the lock and hit the cache.
        self._locks: dict[str, asyncio.Lock] = {}

    async def detect_communities(
        self,
        algorithm: str = "louvain",
        min_community_size: int = 3,
        parameters: dict[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> CommunityResponse:
        # Only the default (parameterless) global detection is cached — the
        # per-entity integrity path always calls it that way, and it keeps the
        # key space from exploding across ad-hoc parameter dicts.
        if self._cache is None or parameters is not None:
            return await self._compute(algorithm, min_community_size, parameters, tenant_id)

        cache_key = CacheService.make_communities_key(tenant_id or "", algorithm, min_community_size)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        lock = self._locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            # Re-check: another coroutine may have populated the cache while we waited.
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
            result = await self._compute(algorithm, min_community_size, parameters, tenant_id)
            self._cache.set(cache_key, result)
            return result

    async def _compute(
        self,
        algorithm: str,
        min_community_size: int,
        parameters: dict[str, Any] | None,
        tenant_id: str | None,
    ) -> CommunityResponse:
        start = time.monotonic()

        result = await self.graph_client.get_communities(
            algorithm=algorithm,
            min_community_size=min_community_size,
            parameters=parameters,
            tenant_id=tenant_id,
        )

        # Map Graph Adapter response to our model
        entity_communities: dict[str, int] = result.get("entity_communities", result.get("communities", {}))
        modularity: float = result.get("modularity", 0.0)

        # Compute community sizes
        community_sizes: dict[int, int] = {}
        for comm_id in entity_communities.values():
            community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1

        elapsed = (time.monotonic() - start) * 1000

        return CommunityResponse(
            algorithm=algorithm,
            entity_communities=entity_communities,
            community_sizes=community_sizes,
            number_of_communities=len(community_sizes),
            modularity=modularity,
            computation_time_ms=round(elapsed, 2),
        )
