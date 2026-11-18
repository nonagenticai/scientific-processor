"""Tests for SubgraphService — Cypher-based subgraph extraction."""

from unittest.mock import AsyncMock

import pytest

from config import Settings
from services.cache import CacheService
from services.subgraph_service import SubgraphService


def _make_cypher_response(nodes, relationships):
    """Build a Graph Adapter /query response with path data."""
    return {
        "results": [{"path": {"nodes": nodes, "relationships": relationships}}],
        "execution_time_ms": 10.0,
        "result_count": 1,
    }


def _make_multi_path_response(paths):
    """Build a Graph Adapter response with multiple path results."""
    return {
        "results": [{"path": p} for p in paths],
        "execution_time_ms": 10.0,
        "result_count": len(paths),
    }


@pytest.fixture
def graph_client():
    client = AsyncMock()
    client.execute_cypher = AsyncMock()
    return client


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def service(graph_client, settings):
    return SubgraphService(graph_client=graph_client, settings=settings)


@pytest.mark.unit
async def test_extract_subgraph_basic(service, graph_client):
    """Verify Cypher query contains entity_id/max_hops and edge data lands in the NX graph."""
    graph_client.get_subgraph.return_value = _make_cypher_response(
        nodes=[
            {"id": "n1", "labels": ["Claim"], "properties": {"name": "A"}},
            {"id": "n2", "labels": ["Source"], "properties": {"name": "B"}},
        ],
        relationships=[
            {"start_node_id": "n2", "end_node_id": "n1", "type": "SUPPORTS", "properties": {"confidence": 0.9}},
        ],
    )

    data = await service.extract_subgraph(entity_id="n1", max_hops=2)

    # Verify get_subgraph was called with the right structured args
    call_kwargs = graph_client.get_subgraph.call_args.kwargs
    assert call_kwargs["entity_id"] == "n1"
    assert call_kwargs["max_hops"] == 2

    # Verify response metadata
    assert data.response.entity_id == "n1"
    assert data.response.node_count == 2
    assert data.response.edge_count == 1
    assert data.response.max_hops_used == 2

    # Verify NetworkX graph has the edge with correct attributes
    assert data.graph.number_of_nodes() == 2
    assert data.graph.has_edge("n2", "n1")
    edge_data = data.graph.edges["n2", "n1"]
    assert edge_data["type"] == "SUPPORTS"
    assert edge_data["confidence"] == pytest.approx(0.9)

    # Verify node data in response
    nodes_by_id = {n["id"]: n for n in data.response.nodes}
    assert nodes_by_id["n1"]["label"] == "Claim"
    assert nodes_by_id["n1"]["properties"]["name"] == "A"
    assert nodes_by_id["n2"]["label"] == "Source"


@pytest.mark.unit
async def test_deduplicates_nodes(service, graph_client):
    """Same node appearing in multiple paths should be deduplicated, preserving first occurrence data."""
    graph_client.get_subgraph.return_value = {
        "results": [
            {
                "path": {
                    "nodes": [{"id": "n1", "labels": ["Claim"], "properties": {"name": "First"}}],
                    "relationships": [],
                }
            },
            {
                "path": {
                    "nodes": [{"id": "n1", "labels": ["Claim"], "properties": {"name": "Duplicate"}}],
                    "relationships": [],
                }
            },
        ],
    }

    data = await service.extract_subgraph(entity_id="n1")
    assert data.response.node_count == 1

    # First-occurrence properties are preserved (not overwritten by duplicate)
    node = data.response.nodes[0]
    assert node["id"] == "n1"
    assert node["label"] == "Claim"
    assert node["labels"] == ["Claim"]
    assert node["properties"]["name"] == "First"


@pytest.mark.unit
async def test_min_confidence_filter(service, graph_client):
    """Verify WHICH edge survives the confidence filter (0.9 kept, 0.3 dropped)."""
    graph_client.get_subgraph.return_value = _make_cypher_response(
        nodes=[
            {"id": "n1", "labels": ["Claim"], "properties": {}},
            {"id": "n2", "labels": ["Source"], "properties": {}},
            {"id": "n3", "labels": ["Source"], "properties": {}},
        ],
        relationships=[
            {"start_node_id": "n2", "end_node_id": "n1", "type": "SUPPORTS", "properties": {"confidence": 0.9}},
            {"start_node_id": "n3", "end_node_id": "n1", "type": "SUPPORTS", "properties": {"confidence": 0.3}},
        ],
    )

    data = await service.extract_subgraph(entity_id="n1", min_confidence=0.5)

    assert data.response.edge_count == 1
    surviving_edge = data.response.edges[0]
    assert surviving_edge["source_id"] == "n2", "The 0.9-confidence edge (n2->n1) should survive"
    assert surviving_edge["target_id"] == "n1"
    assert surviving_edge["properties"]["confidence"] == pytest.approx(0.9)

    # The n3->n1 edge (0.3 confidence) should be gone from both response and NX graph
    assert data.graph.has_edge("n2", "n1")
    assert not data.graph.has_edge("n3", "n1")


@pytest.mark.unit
async def test_max_nodes_cap(service, graph_client):
    """n0 (hub node, degree 9) must be kept; surviving edges connect only kept nodes."""
    nodes = [{"id": f"n{i}", "labels": ["Node"], "properties": {}} for i in range(10)]
    rels = [{"start_node_id": f"n{i}", "end_node_id": "n0", "type": "REF", "properties": {}} for i in range(1, 10)]
    graph_client.get_subgraph.return_value = _make_cypher_response(nodes, rels)

    data = await service.extract_subgraph(entity_id="n0", max_nodes=5)

    assert data.response.node_count <= 5

    # n0 has degree 9 (highest) — it must always be kept
    kept_ids = {n["id"] for n in data.response.nodes}
    assert "n0" in kept_ids, "Hub node n0 (degree 9) must survive the cap"

    # All surviving edges must connect only kept nodes
    for edge in data.response.edges:
        assert edge["source_id"] in kept_ids, f"Edge source {edge['source_id']} not in kept nodes"
        assert edge["target_id"] in kept_ids, f"Edge target {edge['target_id']} not in kept nodes"

    # NX graph should also reflect the cap
    assert data.graph.number_of_nodes() <= 5
    assert set(data.graph.nodes()) == kept_ids


@pytest.mark.unit
async def test_empty_response(service, graph_client):
    graph_client.get_subgraph.return_value = {"results": []}

    data = await service.extract_subgraph(entity_id="missing")
    assert data.response.node_count == 0
    assert data.response.edge_count == 0
    assert data.graph.number_of_nodes() == 0


@pytest.mark.unit
async def test_relationship_types_filter(service, graph_client):
    """Only edges matching the requested relationship_types should appear in the Cypher query."""
    graph_client.get_subgraph.return_value = _make_cypher_response(
        nodes=[
            {"id": "n1", "labels": ["Claim"], "properties": {}},
            {"id": "n2", "labels": ["Source"], "properties": {}},
        ],
        relationships=[
            {"start_node_id": "n2", "end_node_id": "n1", "type": "SUPPORTS", "properties": {"confidence": 0.8}},
        ],
    )

    data = await service.extract_subgraph(entity_id="n1", max_hops=2, relationship_types=["SUPPORTS", "REFUTES"])

    # Verify the relationship type filter is forwarded to the structured endpoint
    call_kwargs = graph_client.get_subgraph.call_args.kwargs
    assert call_kwargs["relationship_types"] == ["SUPPORTS", "REFUTES"]

    # Verify result still works
    assert data.response.edge_count == 1
    assert data.response.edges[0]["type"] == "SUPPORTS"


@pytest.mark.unit
async def test_multiple_paths_overlapping_nodes_and_edges(service, graph_client):
    """Realistic scenario: multiple paths share nodes and edges, no duplicates in output."""
    graph_client.get_subgraph.return_value = _make_multi_path_response(
        [
            {
                "nodes": [
                    {"id": "A", "labels": ["Paper"], "properties": {"title": "Paper A"}},
                    {"id": "B", "labels": ["Claim"], "properties": {"text": "Claim B"}},
                ],
                "relationships": [
                    {"start_node_id": "A", "end_node_id": "B", "type": "SUPPORTS", "properties": {"confidence": 0.95}},
                ],
            },
            {
                "nodes": [
                    {"id": "B", "labels": ["Claim"], "properties": {"text": "Claim B"}},
                    {"id": "C", "labels": ["Paper"], "properties": {"title": "Paper C"}},
                ],
                "relationships": [
                    {"start_node_id": "C", "end_node_id": "B", "type": "REFUTES", "properties": {"confidence": 0.7}},
                ],
            },
            {
                # Third path re-traverses A->B (should be deduplicated)
                "nodes": [
                    {"id": "A", "labels": ["Paper"], "properties": {"title": "Paper A"}},
                    {"id": "B", "labels": ["Claim"], "properties": {"text": "Claim B"}},
                    {"id": "D", "labels": ["Dataset"], "properties": {"name": "Dataset D"}},
                ],
                "relationships": [
                    {"start_node_id": "A", "end_node_id": "B", "type": "SUPPORTS", "properties": {"confidence": 0.95}},
                    {"start_node_id": "B", "end_node_id": "D", "type": "CITES", "properties": {}},
                ],
            },
        ]
    )

    data = await service.extract_subgraph(entity_id="B")

    # 4 unique nodes: A, B, C, D
    assert data.response.node_count == 4
    node_ids = {n["id"] for n in data.response.nodes}
    assert node_ids == {"A", "B", "C", "D"}

    # 3 unique edges: A->B SUPPORTS, C->B REFUTES, B->D CITES
    # (A->B SUPPORTS from path 3 is a duplicate of path 1)
    assert data.response.edge_count == 3
    edge_keys = {(e["source_id"], e["target_id"], e["type"]) for e in data.response.edges}
    assert ("A", "B", "SUPPORTS") in edge_keys
    assert ("C", "B", "REFUTES") in edge_keys
    assert ("B", "D", "CITES") in edge_keys

    # NX graph matches
    assert data.graph.number_of_nodes() == 4
    assert data.graph.number_of_edges() == 3


@pytest.mark.unit
async def test_max_hops_clamped_to_settings_limit(service, graph_client, settings):
    """max_hops passed by caller is clamped to settings.subgraph_max_hops."""
    graph_client.get_subgraph.return_value = {"results": []}

    # Request 10 hops — should be clamped to settings.subgraph_max_hops (default 3)
    await service.extract_subgraph(entity_id="n1", max_hops=10)

    call_kwargs = graph_client.get_subgraph.call_args.kwargs
    # max_hops forwarded to the endpoint should be clamped, not 10
    assert call_kwargs["max_hops"] == settings.subgraph_max_hops
    assert call_kwargs["max_hops"] != 10


@pytest.mark.unit
async def test_malicious_relationship_types_filtered(service, graph_client):
    """Cypher injection attempts via relationship_types are sanitized out."""
    graph_client.get_subgraph.return_value = _make_cypher_response(
        nodes=[
            {"id": "n1", "labels": ["Claim"], "properties": {}},
            {"id": "n2", "labels": ["Source"], "properties": {}},
        ],
        relationships=[
            {"start_node_id": "n2", "end_node_id": "n1", "type": "SUPPORTS", "properties": {}},
        ],
    )

    # Mix valid types with injection attempts
    malicious_types = [
        "SUPPORTS",  # valid
        "REFUTES",  # valid
        "]-() DETACH DELETE n //",  # injection attempt
        "foo",  # lowercase — invalid
        "VALID_TYPE_2",  # valid
        "HAS SPACE",  # space — invalid
        "",  # empty — invalid
        "SUPPORTS|DETACH DELETE n",  # pipe injection — invalid (contains space)
    ]

    await service.extract_subgraph(entity_id="n1", relationship_types=malicious_types)

    call_kwargs = graph_client.get_subgraph.call_args.kwargs
    forwarded = call_kwargs["relationship_types"]

    # Only the three valid types are forwarded to the endpoint
    assert forwarded == ["SUPPORTS", "REFUTES", "VALID_TYPE_2"]

    # None of the injection strings survive the whitelist
    blob = "".join(forwarded)
    assert "DETACH" not in blob
    assert "DELETE" not in blob
    assert "//" not in blob


@pytest.mark.unit
async def test_all_invalid_relationship_types_produces_no_filter(service, graph_client):
    """If all relationship_types are invalid, the query has no type filter."""
    graph_client.get_subgraph.return_value = {"results": []}

    await service.extract_subgraph(
        entity_id="n1",
        relationship_types=["lower", "has space", "]-() DELETE n //"],
    )

    call_kwargs = graph_client.get_subgraph.call_args.kwargs
    # All types invalid → no filter forwarded (None), so the endpoint walks all
    # relationship types. No injection fragment is passed through.
    assert call_kwargs["relationship_types"] is None


# ---------------------------------------------------------------------------
# Cache integration tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_cache_miss_then_hit(graph_client, settings):
    """First call is a cache miss (queries graph), second call is a cache hit (skips graph)."""
    cache = CacheService(ttl_seconds=300, max_entries=100)
    svc = SubgraphService(graph_client=graph_client, settings=settings, cache=cache)

    graph_client.get_subgraph.return_value = _make_cypher_response(
        nodes=[
            {"id": "n1", "labels": ["Claim"], "properties": {"name": "A"}},
            {"id": "n2", "labels": ["Source"], "properties": {"name": "B"}},
        ],
        relationships=[
            {"start_node_id": "n2", "end_node_id": "n1", "type": "SUPPORTS", "properties": {"confidence": 0.9}},
        ],
    )

    # First call — cache miss
    data1 = await svc.extract_subgraph(entity_id="n1", max_hops=2, tenant_id="t1")
    assert graph_client.get_subgraph.await_count == 1
    assert data1.response.node_count == 2

    # Second call — cache hit, graph_client should NOT be called again
    data2 = await svc.extract_subgraph(entity_id="n1", max_hops=2, tenant_id="t1")
    assert graph_client.get_subgraph.await_count == 1  # still 1
    assert data2.response.node_count == 2
    assert data2 is data1  # same object from cache

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


@pytest.mark.unit
async def test_cache_none_means_no_caching(graph_client, settings):
    """When cache=None, every call queries the graph (backward compat)."""
    svc = SubgraphService(graph_client=graph_client, settings=settings, cache=None)

    graph_client.get_subgraph.return_value = _make_cypher_response(
        nodes=[{"id": "n1", "labels": ["Claim"], "properties": {}}],
        relationships=[],
    )

    await svc.extract_subgraph(entity_id="n1", max_hops=2, tenant_id="t1")
    await svc.extract_subgraph(entity_id="n1", max_hops=2, tenant_id="t1")
    assert graph_client.get_subgraph.await_count == 2


@pytest.mark.unit
async def test_cache_key_varies_by_params(graph_client, settings):
    """Different entity_id / max_hops / tenant_id produce separate cache entries."""
    cache = CacheService(ttl_seconds=300, max_entries=100)
    svc = SubgraphService(graph_client=graph_client, settings=settings, cache=cache)

    graph_client.get_subgraph.return_value = _make_cypher_response(
        nodes=[{"id": "n1", "labels": ["Claim"], "properties": {}}],
        relationships=[],
    )

    await svc.extract_subgraph(entity_id="n1", max_hops=2, tenant_id="t1")
    await svc.extract_subgraph(entity_id="n1", max_hops=1, tenant_id="t1")  # different max_hops
    await svc.extract_subgraph(entity_id="n2", max_hops=2, tenant_id="t1")  # different entity_id
    await svc.extract_subgraph(entity_id="n1", max_hops=2, tenant_id="t2")  # different tenant

    # All 4 calls should have hit the graph (all cache misses)
    assert graph_client.get_subgraph.await_count == 4
    assert cache.stats()["misses"] == 4
    assert cache.stats()["hits"] == 0
