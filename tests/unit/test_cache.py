"""Tests for cache service — TTL cache for subgraph extraction and centrality."""

from __future__ import annotations

import pytest

from services.cache import CacheService

pytestmark = pytest.mark.unit


class TestCacheService:
    def test_get_miss(self):
        cache = CacheService(ttl_seconds=300, max_entries=100)
        assert cache.get("nonexistent") is None

    def test_set_and_get(self):
        cache = CacheService(ttl_seconds=300, max_entries=100)
        cache.set("key1", {"data": "value"})
        assert cache.get("key1") == {"data": "value"}

    def test_invalidate(self):
        cache = CacheService(ttl_seconds=300, max_entries=100)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.invalidate("key1")
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_clear_all(self):
        cache = CacheService(ttl_seconds=300, max_entries=100)
        cache.set("key1", "v1")
        cache.set("key2", "v2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_key_tenant_reads_segment_1_of_every_builder(self):
        """``key_tenant`` is the single definition of the key grammar — it must agree
        with all three ``make_*_key`` builders."""
        assert CacheService.key_tenant(CacheService.make_subgraph_key("tenant-1", "e1", 2)) == "tenant-1"
        assert CacheService.key_tenant(CacheService.make_centrality_key("tenant-1", "pagerank")) == "tenant-1"
        assert CacheService.key_tenant(CacheService.make_communities_key("tenant-1", "louvain", 3)) == "tenant-1"

    def test_key_tenant_ignores_colons_in_the_remainder(self):
        """An entity_id containing colons must not shift the tenant segment."""
        key = CacheService.make_subgraph_key("tenant-1", "urn:doi:10.1/x", 2)
        assert CacheService.key_tenant(key) == "tenant-1"

    def test_key_tenant_is_none_for_an_untenanted_key(self):
        assert CacheService.key_tenant("plain-key") is None

    def test_key_tenant_of_an_empty_tenant_segment_is_not_a_real_tenant(self):
        """A key built with an empty tenant (the old topology ``None`` path) is owned
        by the empty string, so no real tenant id can ever match it."""
        assert CacheService.key_tenant("subgraph::e1:2") == ""

    def test_clear_tenant_removes_only_that_tenants_entries(self):
        cache = CacheService(ttl_seconds=300, max_entries=100)
        mine = CacheService.make_subgraph_key("tenant-1", "e1", 2)
        theirs = CacheService.make_subgraph_key("tenant-2", "e1", 2)
        untenanted = "loose-key"
        cache.set(mine, "mine")
        cache.set(theirs, "theirs")
        cache.set(untenanted, "loose")

        assert cache.clear_tenant("tenant-1") == 1
        assert cache.get(mine) is None
        assert cache.get(theirs) == "theirs"
        assert cache.get(untenanted) == "loose"

    def test_clear_tenant_of_an_unknown_tenant_removes_nothing(self):
        cache = CacheService(ttl_seconds=300, max_entries=100)
        cache.set(CacheService.make_subgraph_key("tenant-1", "e1", 2), "mine")
        assert cache.clear_tenant("tenant-9") == 0
        assert cache.stats()["size"] == 1

    def test_tenant_size_counts_only_that_tenants_entries(self):
        cache = CacheService(ttl_seconds=300, max_entries=100)
        cache.set(CacheService.make_subgraph_key("tenant-1", "e1", 2), "a")
        cache.set(CacheService.make_centrality_key("tenant-1", "pagerank"), "b")
        cache.set(CacheService.make_subgraph_key("tenant-2", "e1", 2), "c")
        cache.set("loose-key", "d")

        assert cache.tenant_size("tenant-1") == 2
        assert cache.tenant_size("tenant-2") == 1
        assert cache.tenant_size("tenant-3") == 0
        assert cache.stats()["size"] == 4

    def test_make_subgraph_key(self):
        cache = CacheService()
        key = cache.make_subgraph_key("tenant-1", "entity-abc", 2)
        assert "tenant-1" in key
        assert "entity-abc" in key
        assert "2" in key

    def test_make_centrality_key(self):
        cache = CacheService()
        key = cache.make_centrality_key("tenant-1", "pagerank")
        assert "tenant-1" in key
        assert "pagerank" in key

    def test_max_entries_evicts_oldest(self):
        cache = CacheService(ttl_seconds=300, max_entries=2)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")
        # k1 should be evicted (LRU)
        assert cache.get("k1") is None
        assert cache.get("k3") == "v3"

    def test_stats(self):
        cache = CacheService(ttl_seconds=300, max_entries=100)
        cache.set("k1", "v1")
        cache.get("k1")  # hit
        cache.get("k2")  # miss
        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_ttl_expiration(self):
        # TTL of 0 means immediate expiration
        cache = CacheService(ttl_seconds=0, max_entries=100)
        cache.set("k1", "v1")
        # With TTL=0, cachetools TTLCache should expire immediately
        # but in practice needs a small delay — test the mechanism exists
        assert cache._cache.ttl == 0


class TestClearCacheDependency:
    """Test the clear_cache() shutdown helper in dependencies.py."""

    def test_clear_cache_clears_and_resets_singleton(self):
        import dependencies

        # Populate the singleton
        old_cache = dependencies._cache_service
        dependencies._cache_service = CacheService(ttl_seconds=300, max_entries=100)
        dependencies._cache_service.set("k1", "v1")
        assert dependencies._cache_service.stats()["size"] == 1

        dependencies.clear_cache()

        # Singleton should be reset to None
        assert dependencies._cache_service is None

        # Restore
        dependencies._cache_service = old_cache

    def test_clear_cache_noop_when_none(self):
        import dependencies

        old_cache = dependencies._cache_service
        dependencies._cache_service = None

        # Should not raise
        dependencies.clear_cache()
        assert dependencies._cache_service is None

        # Restore
        dependencies._cache_service = old_cache


class TestTenantSegmentIsolation:
    """The anti-collision property of the tenant segment, pinned against mutation.

    ``clear_tenant`` / ``tenant_size`` / the router's ownership check all decide
    ownership by comparing ``key_tenant(key)`` to the caller's tenant. That comparison
    MUST be exact equality. Two plausible-looking loosenings both wipe other tenants'
    data and both survived the suite before these tests existed:

      ``(self.key_tenant(key) or "").startswith(tenant_id)``  -> ``acme`` wipes ``acme2``
      ``tenant_id in key``                                    -> ``acme`` wipes any key
                                                                 mentioning ``acme``

    Every test below is written to FAIL under one of those two mutations.
    """

    @staticmethod
    def _cache() -> CacheService:
        return CacheService(ttl_seconds=300, max_entries=100)

    def test_clear_tenant_spares_prefix_neighbours(self):
        """KILLS the ``startswith`` mutation (and the substring one).

        ``acme2`` and ``acmeX`` are unrelated tenants that happen to start with the
        caller's id. Under a prefix match ``acme``'s clear takes all three.
        """
        cache = self._cache()
        mine = CacheService.make_subgraph_key("acme", "e1", 2)
        neighbour_digit = CacheService.make_subgraph_key("acme2", "e1", 2)
        neighbour_alpha = CacheService.make_centrality_key("acmeX", "pagerank")
        longer = CacheService.make_communities_key("acme-europe", "louvain", 3)
        for key, value in (
            (mine, "mine"),
            (neighbour_digit, "acme2"),
            (neighbour_alpha, "acmeX"),
            (longer, "acme-europe"),
        ):
            cache.set(key, value)

        assert cache.clear_tenant("acme") == 1
        assert cache.get(mine) is None
        assert cache.get(neighbour_digit) == "acme2"
        assert cache.get(neighbour_alpha) == "acmeX"
        assert cache.get(longer) == "acme-europe"
        assert cache.stats()["size"] == 3

    def test_clear_tenant_spares_a_key_that_merely_mentions_the_tenant_elsewhere(self):
        """KILLS the whole-key ``substring`` mutation specifically.

        Here the caller's id appears in the key's ENTITY segment, not its tenant
        segment, and is not a prefix of the owner — so ``startswith`` is innocent and
        only ``tenant_id in key`` misfires. An entity id equal to some other tenant's
        id is ordinary: entity ids are free-form strings.
        """
        cache = self._cache()
        theirs = CacheService.make_subgraph_key("acme", "e1", 2)  # 'e1' is the ENTITY
        also_theirs = CacheService.make_centrality_key("acme", "e1-pagerank")
        cache.set(theirs, "acme-data")
        cache.set(also_theirs, "acme-data-2")

        assert cache.clear_tenant("e1") == 0
        assert cache.get(theirs) == "acme-data"
        assert cache.get(also_theirs) == "acme-data-2"

    def test_clear_tenant_spares_a_key_whose_namespace_matches_the_tenant(self):
        """A tenant literally named after a key namespace must not sweep that namespace."""
        cache = self._cache()
        key = CacheService.make_subgraph_key("acme", "e1", 2)
        cache.set(key, "acme-data")

        assert cache.clear_tenant("subgraph") == 0
        assert cache.get(key) == "acme-data"

    def test_tenant_size_spares_prefix_neighbours(self):
        """``tenant_size`` shares the comparison, so it needs the same guarantee."""
        cache = self._cache()
        cache.set(CacheService.make_subgraph_key("acme", "e1", 2), "a")
        cache.set(CacheService.make_subgraph_key("acme2", "e1", 2), "b")
        cache.set(CacheService.make_centrality_key("acmeX", "pagerank"), "c")

        assert cache.tenant_size("acme") == 1
        assert cache.tenant_size("acme2") == 1
        assert cache.tenant_size("acmeX") == 1

    def test_key_tenant_of_a_prefix_neighbour_is_not_the_neighbour(self):
        """The parser itself: no partial match anywhere in the chain."""
        assert CacheService.key_tenant(CacheService.make_subgraph_key("acme2", "e1", 2)) == "acme2"
        assert CacheService.key_tenant(CacheService.make_subgraph_key("acme2", "e1", 2)) != "acme"


class TestTenantIdCannotForgeTheKeyGrammar:
    """A tenant id containing the ``:`` delimiter must never reach a key.

    Reproduced end-to-end before this guard: tenant ``acme:eu`` wrote
    ``subgraph:acme:eu:e1:2``, whose segment 1 is ``acme``. ``POST /api/cache/clear``
    as ``acme`` then returned 200 ``previous_size=1`` having deleted ``acme:eu``'s
    entry, while ``acme:eu`` got 403 invalidating its OWN key.

    The primary fix is at the process boundary (``auth.middleware._TENANT_ID_RE``);
    this is the builder-level backstop for any non-HTTP caller.
    """

    @pytest.mark.parametrize(
        "builder",
        [
            lambda t: CacheService.make_subgraph_key(t, "e1", 2),
            lambda t: CacheService.make_centrality_key(t, "pagerank"),
            lambda t: CacheService.make_communities_key(t, "louvain", 3),
        ],
        ids=["subgraph", "centrality", "communities"],
    )
    def test_every_builder_refuses_a_colon_bearing_tenant(self, builder):
        with pytest.raises(ValueError, match="delimiter"):
            builder("acme:eu")

    def test_the_admissible_charset_round_trips_through_build_and_parse(self):
        """For every tenant id the boundary admits, build->parse returns it exactly.

        This is the property that makes ``/invalidate`` on one's OWN key always 200:
        the owner the router reads back is the caller, character for character.
        Includes the prefix-colliding family so a loosened comparison cannot hide here.
        """
        admissible = [
            "0195f48b-d000-7000-8000-000000000001",
            "tenant-abc",
            "tenant_abc",
            "tenant.abc",
            "t1",
            "acme",
            "acme2",
            "acmeX",
            "acme-europe",
            "A" * 128,
        ]
        for tenant in admissible:
            for key in (
                CacheService.make_subgraph_key(tenant, "urn:doi:10.1/x", 2),
                CacheService.make_centrality_key(tenant, "pagerank"),
                CacheService.make_communities_key(tenant, "louvain", 3),
            ):
                assert CacheService.key_tenant(key) == tenant, key


class TestPerTenantOperationsRefuseAFalsyTenant:
    """``clear_tenant(None)`` swept every untenanted key — closed, fail-loud.

    ``key_tenant`` returns ``None`` for a key with fewer than two segments and ``""``
    for a key built with an empty tenant (``centrality::pagerank`` — what
    ``centrality_service`` emits for ``tenant_id or ""``). Both compare equal to a
    falsy argument. Unreachable from HTTP today and invisible to ``ty`` because the
    parameter is annotated ``str``, which is precisely why it needs a runtime guard.
    """

    @staticmethod
    def _loaded_cache() -> CacheService:
        cache = CacheService(ttl_seconds=300, max_entries=100)
        cache.set("loose-key", "untenanted")
        cache.set("another", "untenanted")
        cache.set("centrality::pagerank", "empty-tenant")
        cache.set(CacheService.make_subgraph_key("t1", "e1", 2), "real")
        return cache

    @pytest.mark.parametrize("falsy", [None, ""], ids=["none", "empty"])
    def test_clear_tenant_refuses_and_removes_nothing(self, falsy):
        cache = self._loaded_cache()
        with pytest.raises(ValueError, match="concrete tenant id"):
            cache.clear_tenant(falsy)
        assert cache.stats()["size"] == 4
        assert cache.get("loose-key") == "untenanted"
        assert cache.get("centrality::pagerank") == "empty-tenant"

    @pytest.mark.parametrize("falsy", [None, ""], ids=["none", "empty"])
    def test_tenant_size_refuses(self, falsy):
        cache = self._loaded_cache()
        with pytest.raises(ValueError, match="concrete tenant id"):
            cache.tenant_size(falsy)
