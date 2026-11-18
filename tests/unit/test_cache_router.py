"""Unit tests for cache management endpoints — /api/cache/stats, invalidate, clear.

Tenancy contract under test:
- ``CacheService`` is a process-global singleton shared by EVERY tenant, and its
  keys carry the owning tenant in their second colon-separated segment
  (``subgraph:<tenant>:<entity>:<hops>`` — see ``CacheService.make_*_key``).
- ``POST /invalidate`` must therefore refuse a key whose tenant segment is not
  the caller's, or tenant A can evict tenant B's entries.
- ``POST /clear`` is TENANT-SCOPED: it removes only the caller's entries and must
  leave every other tenant's — and every untenanted key's — entries intact. It is
  deliberately NOT role-gated: neither identity path in ``auth.middleware`` can
  ever produce a realm role (the in-cluster path hard-codes ``roles: []``, and the
  licenses introspection response carries no ``roles`` field), so a role gate here
  would 403 every real caller forever.
- ``GET /stats`` stays open to any authenticated tenant — coarse global metadata.
"""

from __future__ import annotations

import os

os.environ["ENABLE_AUTH"] = "false"
os.environ["ENABLE_DOCS"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from app import create_app
from auth.dependencies import DEFAULT_TENANT_ID, _require_uuid_tenant
from config import Settings
from dependencies import get_cache_service
from services.cache import CacheService

pytestmark = pytest.mark.unit

OTHER_TENANT_ID = "0195f48b-d000-7000-8000-0000000000ff"

# Distinct canonical tenants, plus two NON-CANONICAL spellings of TENANT_A.
# uuid.UUID() ignores dash placement, so all of TENANT_A / PREFIX_COLLIDING_SPELLING
# / UNDASHED_SPELLING parse to the same uuid while being different strings. The
# boundary canonicalises them onto TENANT_A (TestBoundaryCanonicalisesTenantSpelling);
# they survive here as the only way to build a raw prefix-colliding pair for the
# service-layer mutation tests.
TENANT_A = "0195f48b-d000-7000-8000-0000000000a1"
PREFIX_COLLIDING_SPELLING = TENANT_A + "-"
UNDASHED_SPELLING = TENANT_A.replace("-", "")
TENANT_B = "0195f48b-d000-7000-8000-0000000000b2"
TENANT_C = "0195f48b-d000-7000-8000-0000000000c3"


@pytest.fixture
def cache():
    return CacheService(ttl_seconds=300, max_entries=100)


@pytest.fixture(autouse=True)
def auth_disabled(monkeypatch):
    """Pin ENABLE_AUTH off regardless of which test module imported auth first.

    ``auth.dependencies.ENABLE_AUTH`` is read from the environment once at import
    time and test_auth.py sets it to "true", so the module global — not the env
    var this module sets — is what these endpoints actually see.
    """
    monkeypatch.setattr("auth.dependencies.ENABLE_AUTH", False)


def _auth_disabled_settings() -> Settings:
    """Settings with auth pinned off, independent of module IMPORT order.

    The module-level ``os.environ["ENABLE_AUTH"] = "false"`` above only wins if this
    module is imported after test_auth.py (which sets it to "true"); pytest imports
    in collection order, so running this file first flipped every tenant-scoping test
    here to 403 — AuthMiddleware got registered and overwrote the injected
    ``request.state.tenant_id``. Pin the field on the object instead of racing on an
    env var, so the tenant-isolation assertions below hold in any collection order.
    """
    settings = Settings()
    settings.enable_auth = False
    return settings


@pytest.fixture
def app(cache):
    application = create_app(_auth_disabled_settings())
    application.dependency_overrides[get_cache_service] = lambda: cache
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def tenant_client(cache):
    """Factory: a TestClient whose requests resolve to an arbitrary tenant id.

    Same mechanism as ``other_tenant_client`` (stand in for AuthMiddleware, which is
    not registered when ENABLE_AUTH is false) but parameterised, so a test can pit
    prefix-colliding tenants against each other over real HTTP.
    """
    applications = []

    def _make(tenant_id: str) -> TestClient:
        application = create_app(_auth_disabled_settings())
        application.dependency_overrides[get_cache_service] = lambda: cache

        @application.middleware("http")
        async def _inject_tenant(request: Request, call_next):
            request.state.tenant_id = tenant_id
            return await call_next(request)

        applications.append(application)
        return TestClient(application)

    yield _make
    for application in applications:
        application.dependency_overrides.clear()


@pytest.fixture
def other_tenant_client(cache):
    """Client whose requests resolve to OTHER_TENANT_ID.

    Stands in for AuthMiddleware (not registered when ENABLE_AUTH is false) by
    setting the same ``request.state.tenant_id`` it would set. This is what
    proves the ownership check reads the *caller's* tenant rather than comparing
    against a constant.
    """
    application = create_app(_auth_disabled_settings())
    application.dependency_overrides[get_cache_service] = lambda: cache

    @application.middleware("http")
    async def _inject_tenant(request: Request, call_next):
        request.state.tenant_id = OTHER_TENANT_ID
        return await call_next(request)

    yield TestClient(application)
    application.dependency_overrides.clear()


class TestCacheStats:
    def test_stats_empty_cache(self, client):
        resp = client.get("/api/cache/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["size"] == 0
        assert data["hits"] == 0
        assert data["misses"] == 0
        assert data["hit_rate"] == 0.0

    def test_stats_after_operations(self, client, cache):
        cache.set("k1", "v1")
        cache.get("k1")  # hit
        cache.get("k2")  # miss

        resp = client.get("/api/cache/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["size"] == 1
        assert data["hits"] == 1
        assert data["misses"] == 1
        assert data["hit_rate"] == 0.5

    def test_stats_open_to_any_ordinary_tenant(self, client):
        """No authorization gate may creep onto /stats — it is coarse global metadata."""
        resp = client.get("/api/cache/stats")
        assert resp.status_code == 200


class TestCacheInvalidate:
    def test_invalidate_own_tenant_key(self, client, cache):
        key = CacheService.make_subgraph_key(DEFAULT_TENANT_ID, "e1", 2)
        cache.set(key, "my-value")
        assert cache.get(key) == "my-value"

        resp = client.post("/api/cache/invalidate", json={"key": key})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "invalidated"
        assert data["key"] == key

        # Key should be gone (the next get counts as a miss)
        assert cache.get(key) is None

    def test_invalidate_nonexistent_own_key(self, client):
        key = CacheService.make_centrality_key(DEFAULT_TENANT_ID, "pagerank")
        resp = client.post("/api/cache/invalidate", json={"key": key})
        assert resp.status_code == 200
        assert resp.json()["status"] == "invalidated"

    def test_invalidate_missing_body_returns_422(self, client):
        resp = client.post("/api/cache/invalidate", json={})
        assert resp.status_code == 422

    def test_invalidate_other_tenant_key_is_forbidden(self, client, cache):
        """Tenant A must not be able to evict tenant B's entry from the shared cache."""
        victim_key = CacheService.make_subgraph_key(OTHER_TENANT_ID, "e1", 2)
        cache.set(victim_key, "victim-value")

        resp = client.post("/api/cache/invalidate", json={"key": victim_key})
        assert resp.status_code == 403

        # The other tenant's entry survives.
        assert cache.get(victim_key) == "victim-value"

    def test_invalidate_other_tenant_communities_key_is_forbidden(self, client, cache):
        """All three key namespaces share the grammar — cover the third one too."""
        victim_key = CacheService.make_communities_key(OTHER_TENANT_ID, "louvain", 3)
        cache.set(victim_key, "victim-value")

        resp = client.post("/api/cache/invalidate", json={"key": victim_key})
        assert resp.status_code == 403
        assert cache.get(victim_key) == "victim-value"

    def test_invalidate_unscoped_key_is_forbidden(self, client, cache):
        """A key with no tenant segment belongs to nobody — fail closed."""
        cache.set("my-key", "my-value")

        resp = client.post("/api/cache/invalidate", json={"key": "my-key"})
        assert resp.status_code == 403
        assert cache.get("my-key") == "my-value"

    def test_invalidate_tenant_prefix_is_not_a_substring_match(self, client, cache):
        """A tenant id that merely starts with the caller's must still be rejected."""
        key = CacheService.make_subgraph_key(f"{DEFAULT_TENANT_ID}-extra", "e1", 2)
        cache.set(key, "not-mine")

        resp = client.post("/api/cache/invalidate", json={"key": key})
        assert resp.status_code == 403
        assert cache.get(key) == "not-mine"

    def test_invalidate_is_scoped_to_the_calling_tenant_not_a_constant(self, other_tenant_client, cache):
        """The same key flips from allowed to forbidden purely by who is calling.

        Guards against a fix that hardcodes DEFAULT_TENANT_ID instead of
        resolving the tenant per request.
        """
        own_key = CacheService.make_subgraph_key(OTHER_TENANT_ID, "e1", 2)
        foreign_key = CacheService.make_subgraph_key(DEFAULT_TENANT_ID, "e1", 2)
        cache.set(own_key, "mine")
        cache.set(foreign_key, "theirs")

        assert other_tenant_client.post("/api/cache/invalidate", json={"key": own_key}).status_code == 200
        assert cache.get(own_key) is None

        assert other_tenant_client.post("/api/cache/invalidate", json={"key": foreign_key}).status_code == 403
        assert cache.get(foreign_key) == "theirs"


class TestCacheClear:
    def test_clear_removes_only_the_calling_tenants_entries(self, client, cache):
        """THE regression test: one tenant's clear must not wipe another tenant's entries.

        Before the fix ``/clear`` called ``CacheService.clear()`` on the process-global
        singleton, so a single caller emptied the cache for everyone (and was 403-ed
        besides). Both halves are asserted: mine gone, theirs intact.
        """
        mine_a = CacheService.make_subgraph_key(DEFAULT_TENANT_ID, "e1", 2)
        mine_b = CacheService.make_centrality_key(DEFAULT_TENANT_ID, "pagerank")
        theirs_a = CacheService.make_subgraph_key(OTHER_TENANT_ID, "e9", 2)
        theirs_b = CacheService.make_communities_key(OTHER_TENANT_ID, "louvain", 3)
        cache.set(mine_a, "mine-a")
        cache.set(mine_b, "mine-b")
        cache.set(theirs_a, "theirs-a")
        cache.set(theirs_b, "theirs-b")

        resp = client.post("/api/cache/clear")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cleared"] is True
        # previous_size counts only the caller's entries, not the global size of 4.
        assert data["previous_size"] == 2

        # My entries are gone...
        assert cache.get(mine_a) is None
        assert cache.get(mine_b) is None
        # ...and the other tenant's entries SURVIVE. This is the half that was broken.
        assert cache.get(theirs_a) == "theirs-a"
        assert cache.get(theirs_b) == "theirs-b"

    def test_clear_is_reachable_by_an_ordinary_caller_holding_no_roles(self, client, cache):
        """No role gate.

        ``AuthMiddleware`` hard-codes ``realm_access={"roles": []}`` on the in-cluster
        path and the licenses introspection payload has no ``roles`` field, so a
        role-gated /clear is unreachable for EVERY real caller — permanently 403.
        """
        cache.set(CacheService.make_subgraph_key(DEFAULT_TENANT_ID, "e1", 2), "v")

        resp = client.post("/api/cache/clear")
        assert resp.status_code == 200
        assert resp.json()["previous_size"] == 1

    def test_clear_is_scoped_to_the_calling_tenant_not_a_constant(self, other_tenant_client, cache):
        """Same endpoint, different caller.

        Guards against a fix that hardcodes ``DEFAULT_TENANT_ID``: the OTHER tenant's
        clear must drop its own entry and leave the default tenant's alone.
        """
        mine = CacheService.make_subgraph_key(OTHER_TENANT_ID, "e1", 2)
        theirs = CacheService.make_subgraph_key(DEFAULT_TENANT_ID, "e1", 2)
        cache.set(mine, "mine")
        cache.set(theirs, "theirs")

        resp = other_tenant_client.post("/api/cache/clear")
        assert resp.status_code == 200
        assert resp.json()["previous_size"] == 1
        assert cache.get(mine) is None
        assert cache.get(theirs) == "theirs"

    def test_clear_leaves_untenanted_keys_alone(self, client, cache):
        """A key with no tenant segment is owned by nobody, so no tenant may sweep it.

        Same fail-closed rule ``/invalidate`` applies to an unscoped key.
        """
        cache.set("untenanted", "v")
        cache.set(CacheService.make_subgraph_key(DEFAULT_TENANT_ID, "e1", 2), "mine")

        resp = client.post("/api/cache/clear")
        assert resp.status_code == 200
        assert resp.json()["previous_size"] == 1
        assert cache.get("untenanted") == "v"

    def test_clear_empty_cache(self, client):
        resp = client.post("/api/cache/clear")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cleared"] is True
        assert data["previous_size"] == 0


class TestBoundaryCanonicalisesTenantSpelling:
    """One tenant must have exactly ONE cache namespace, whatever spelling it sent.

    ``uuid.UUID()`` accepts dashes wherever they fall, so a single tenant has
    unlimited valid spellings of its id. When ``_require_uuid_tenant`` handed the
    ORIGINAL string back, every spelling became its own key prefix in the shared
    ``CacheService`` — a caller that sent the undashed form wrote entries it could
    never read back through the canonical form, and vice versa. Postgres was never
    affected (it casts to ``uuid``); the key grammars here were.

    The boundary now returns ``str(uuid.UUID(x))``, which collapses them.
    """

    def test_every_spelling_resolves_to_the_canonical_string(self):
        """The unit-level statement of the contract, for all three shapes."""
        assert _require_uuid_tenant(TENANT_A) == TENANT_A
        assert _require_uuid_tenant(PREFIX_COLLIDING_SPELLING) == TENANT_A
        assert _require_uuid_tenant(UNDASHED_SPELLING) == TENANT_A
        assert _require_uuid_tenant(TENANT_A.upper()) == TENANT_A

    @pytest.mark.parametrize("spelling", [PREFIX_COLLIDING_SPELLING, UNDASHED_SPELLING])
    def test_a_noncanonical_caller_sees_its_own_cached_entries(self, tenant_client, cache, spelling):
        """The defect this fixes, asserted where it actually bit: over HTTP.

        The entry is written under the canonical id. A caller that authenticates
        with a different-but-equivalent spelling of the SAME uuid must reach it —
        before canonicalisation it got its own namespace and cleared nothing.
        """
        key = CacheService.make_subgraph_key(TENANT_A, "e1", 2)
        cache.set(key, "mine")

        resp = tenant_client(spelling).post("/api/cache/clear")
        assert resp.status_code == 200
        assert resp.json()["previous_size"] == 1, f"spelling {spelling!r} got its own cache namespace"
        assert cache.get(key) is None


class TestClearCannotReachAnotherTenantByPartialMatch:
    """``/clear`` is BULK deletion — its ownership test is the security boundary.

    Both of these loosenings of ``CacheService.clear_tenant`` survived the whole
    suite before this class existed, because no test used a prefix-colliding tenant
    pair:

        if (self.key_tenant(key) or "").startswith(tenant_id):   # T wipes T's superstrings
        if tenant_id in key:                                     # T wipes anything
                                                                 # whose key mentions T

    Each test below fails under at least one of them; together they cover both.

    ⚠️ CANONICALISATION AT THE HTTP BOUNDARY DOES NOT RETIRE THIS CLASS, and the
    argument that it does is a trap. It runs: tenant ids are canonical uuids now,
    canonical uuids are all 36 chars, no 36-char string is a strict prefix of
    another, therefore ``startswith`` is equivalent to ``==`` and the mutation is
    dead. Every step is true and the conclusion is still wrong, because it holds
    only for callers that came through ``_require_uuid_tenant``. ``CacheService``
    is a process-global singleton also driven by callers that never touch the HTTP
    boundary — background workers, the eviction path, anything constructing keys
    directly — and ``clear_tenant`` is what stands between them and another
    tenant's data. Deleting the coverage because one entry path now sanitises its
    input would leave the function itself unguarded for every other entry path.

    So the prefix-pair coverage moved DOWN a layer rather than being dropped: the
    two tests below call ``clear_tenant`` directly with a raw colliding pair, which
    is the only layer that can still express one. The HTTP-level guarantee that no
    such pair can arrive from a request is asserted separately, in
    ``TestBoundaryCanonicalisesTenantSpelling``. Both mutations remain red.
    """

    def test_clear_spares_prefix_neighbours_at_the_service_layer(self, cache):
        """KILLS the ``startswith`` mutation. Service-level, per the class docstring."""
        keys = {
            "a": CacheService.make_subgraph_key(TENANT_A, "e1", 2),
            "a_suffixed": CacheService.make_subgraph_key(PREFIX_COLLIDING_SPELLING, "e1", 2),
            "b": CacheService.make_centrality_key(TENANT_B, "pagerank"),
            "c": CacheService.make_communities_key(TENANT_C, "louvain", 3),
        }
        for name, key in keys.items():
            cache.set(key, name)

        assert cache.clear_tenant(TENANT_A) == 1, "clear leaked into a prefix-colliding tenant"

        assert cache.get(keys["a"]) is None
        assert cache.get(keys["a_suffixed"]) == "a_suffixed"
        assert cache.get(keys["b"]) == "b"
        assert cache.get(keys["c"]) == "c"

    def test_clear_by_the_longer_id_spares_the_shorter(self, cache):
        """The reverse direction: the longer id must not sweep the shorter one either."""
        mine = CacheService.make_subgraph_key(PREFIX_COLLIDING_SPELLING, "e1", 2)
        theirs = CacheService.make_subgraph_key(TENANT_A, "e1", 2)
        cache.set(mine, "mine")
        cache.set(theirs, "theirs")

        assert cache.clear_tenant(PREFIX_COLLIDING_SPELLING) == 1
        assert cache.get(mine) is None
        assert cache.get(theirs) == "theirs"

    def test_clear_spares_keys_that_only_mention_the_caller_elsewhere(self, tenant_client, cache):
        """KILLS the whole-key ``substring`` mutation specifically.

        Stays at the HTTP layer: it needs no colliding pair, only two distinct
        canonical tenants, one of which appears in the other's ENTITY and ALGORITHM
        segments — a real shape, since an entity id is caller-supplied free text.
        """
        entity_hit = CacheService.make_subgraph_key(TENANT_A, TENANT_B, 2)
        algo_hit = CacheService.make_centrality_key(TENANT_A, f"{TENANT_B}-pagerank")
        cache.set(entity_hit, "a")
        cache.set(algo_hit, "a")

        resp = tenant_client(TENANT_B).post("/api/cache/clear")
        assert resp.status_code == 200
        assert resp.json()["previous_size"] == 0
        assert cache.get(entity_hit) == "a"
        assert cache.get(algo_hit) == "a"

    def test_a_tenant_named_after_a_namespace_clears_nothing(self, tenant_client, cache):
        """``subgraph`` is a key namespace, not an owner.

        Two assertions, for the same reason the prefix tests moved down a layer.
        Over HTTP the id is now refused outright (400) — ``subgraph`` is not a uuid.
        But ``CacheService`` is also reachable from non-HTTP callers, which never
        pass ``_require_uuid_tenant``, so the segment-0 rule still has to hold in
        the service itself. Asserting only the 400 would quietly drop the real
        invariant.
        """
        key = CacheService.make_subgraph_key(TENANT_A, "e1", 2)
        cache.set(key, "a")

        resp = tenant_client("subgraph").post("/api/cache/clear")
        assert resp.status_code == 400

        assert cache.clear_tenant("subgraph") == 0
        assert cache.get(key) == "a"

    @pytest.mark.parametrize("tenant", [TENANT_A, TENANT_B, TENANT_C, DEFAULT_TENANT_ID])
    def test_any_tenant_can_always_invalidate_its_own_key(self, tenant_client, cache, tenant):
        """The other half of the ``:``-in-tenant defect: no caller is 403-ed on its OWN key.

        Tenant ``acme:eu`` used to be, because its key parsed back as ``acme``. With
        ``:`` refused at the boundary, ``key_tenant(make_*_key(t, ...)) == t`` holds for
        every admissible ``t`` — asserted here over HTTP, including for an entity id
        that is itself full of colons.
        """
        client_ = tenant_client(tenant)
        for key in (
            CacheService.make_subgraph_key(tenant, "urn:doi:10.1/x", 2),
            CacheService.make_centrality_key(tenant, "pagerank"),
            CacheService.make_communities_key(tenant, "louvain", 3),
        ):
            cache.set(key, "mine")
            resp = client_.post("/api/cache/invalidate", json={"key": key})
            assert resp.status_code == 200, f"{tenant} was refused its own key {key}"
            assert cache.get(key) is None
