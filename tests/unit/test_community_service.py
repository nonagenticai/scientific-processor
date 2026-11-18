"""CommunityService — cache + single-flight around the full-graph Louvain pass.

The N+1 this pins: ``POST /api/pipeline/batch {"batch_size": N}`` fans out to N
concurrent ``IntegrityService._community_score`` calls, each of which asks for a
full-tenant community detection. Without memoisation that is N identical Louvain
passes over the whole graph for one batch.

These tests assert the *invocation count*, not merely that a cache key exists.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from services.cache import CacheService
from services.community_service import CommunityService

pytestmark = pytest.mark.unit

BATCH_SIZE = 50


def _graph_client(delay: float = 0.0) -> AsyncMock:
    """A graph client whose Louvain call is slow enough to overlap concurrently."""
    mock = AsyncMock()

    async def _get_communities(**_kwargs):
        if delay:
            await asyncio.sleep(delay)
        return {"entity_communities": {f"e{i}": i % 3 for i in range(9)}, "modularity": 0.42}

    mock.get_communities = AsyncMock(side_effect=_get_communities)
    return mock


async def test_batch_of_n_triggers_exactly_one_louvain_pass():
    """N concurrent detections for one tenant must collapse into ONE graph call."""
    client = _graph_client(delay=0.01)
    service = CommunityService(graph_client=client, cache=CacheService(ttl_seconds=300))

    results = await asyncio.gather(*[service.detect_communities(tenant_id="t1") for _ in range(BATCH_SIZE)])

    assert len(results) == BATCH_SIZE
    assert client.get_communities.call_count == 1, (
        f"{client.get_communities.call_count} full-graph Louvain passes for a batch of {BATCH_SIZE}"
    )
    assert all(r.entity_communities == results[0].entity_communities for r in results)


async def test_sequential_calls_hit_the_cache():
    """A warm cache serves later calls without recomputing."""
    client = _graph_client()
    service = CommunityService(graph_client=client, cache=CacheService(ttl_seconds=300))

    for _ in range(BATCH_SIZE):
        await service.detect_communities(tenant_id="t1")

    assert client.get_communities.call_count == 1


async def test_cache_key_is_scoped_per_tenant():
    """Tenant A's community assignment must never be served to tenant B."""
    client = _graph_client()
    service = CommunityService(graph_client=client, cache=CacheService(ttl_seconds=300))

    await service.detect_communities(tenant_id="t1")
    await service.detect_communities(tenant_id="t2")
    await service.detect_communities(tenant_id="t1")

    assert client.get_communities.call_count == 2
    keys = {c.kwargs["tenant_id"] for c in client.get_communities.call_args_list}
    assert keys == {"t1", "t2"}


async def test_cache_key_distinguishes_algorithm_and_min_size():
    """Different detection parameters are different measurements, not cache hits."""
    client = _graph_client()
    service = CommunityService(graph_client=client, cache=CacheService(ttl_seconds=300))

    await service.detect_communities(tenant_id="t1", algorithm="louvain", min_community_size=3)
    await service.detect_communities(tenant_id="t1", algorithm="leiden", min_community_size=3)
    await service.detect_communities(tenant_id="t1", algorithm="louvain", min_community_size=5)
    await service.detect_communities(tenant_id="t1", algorithm="louvain", min_community_size=3)

    assert client.get_communities.call_count == 3


async def test_ad_hoc_parameters_are_not_cached():
    """Explicit ``parameters`` bypass the cache rather than exploding the key space."""
    client = _graph_client()
    service = CommunityService(graph_client=client, cache=CacheService(ttl_seconds=300))

    await service.detect_communities(tenant_id="t1", parameters={"resolution": 1.2})
    await service.detect_communities(tenant_id="t1", parameters={"resolution": 1.2})

    assert client.get_communities.call_count == 2


async def test_invalidating_the_key_forces_recompute():
    """A graph mutation invalidates the key; the next call must not serve stale data."""
    cache = CacheService(ttl_seconds=300)
    client = _graph_client()
    service = CommunityService(graph_client=client, cache=cache)

    await service.detect_communities(tenant_id="t1")
    cache.invalidate(CacheService.make_communities_key("t1", "louvain", 3))
    await service.detect_communities(tenant_id="t1")

    assert client.get_communities.call_count == 2


async def test_no_cache_configured_still_works():
    client = _graph_client()
    service = CommunityService(graph_client=client, cache=None)

    await service.detect_communities(tenant_id="t1")
    await service.detect_communities(tenant_id="t1")

    assert client.get_communities.call_count == 2
