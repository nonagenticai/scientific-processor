"""Tests for CentralityService — PageRank/eigenvector centrality."""

from unittest.mock import AsyncMock

import networkx as nx
import pytest

from models.topology import SubgraphResponse
from services.cache import CacheService
from services.centrality_service import CentralityService
from services.subgraph_service import SubgraphData, SubgraphService


def _make_subgraph_data_from_graph(graph: nx.DiGraph, entity_id: str = "center") -> SubgraphData:
    """Build SubgraphData from an existing NetworkX graph."""
    nodes = []
    for nid, ndata in graph.nodes(data=True):
        nodes.append(
            {
                "id": nid,
                "label": ndata.get("label", "Node"),
                "labels": ndata.get("labels", ["Node"]),
                "properties": ndata.get("properties", {}),
            }
        )
    edges = []
    for src, tgt, edata in graph.edges(data=True):
        edges.append(
            {
                "source_id": src,
                "target_id": tgt,
                "type": edata.get("type", "REF"),
            }
        )

    response = SubgraphResponse(
        entity_id=entity_id,
        nodes=nodes,
        edges=edges,
        node_count=len(nodes),
        edge_count=len(edges),
        max_hops_used=2,
        computation_time_ms=1.0,
    )
    return SubgraphData(response=response, graph=graph)


def _make_star_graph(n_spokes: int = 5) -> nx.DiGraph:
    """Build a star graph where all spokes point to center. Center should get highest PageRank."""
    g = nx.DiGraph()
    g.add_node("center", label="Hub", labels=["Hub"], properties={})
    for i in range(n_spokes):
        nid = f"spoke{i}"
        g.add_node(nid, label="Leaf", labels=["Leaf"], properties={})
        g.add_edge(nid, "center", type="REF")
    return g


def _make_chain_graph(n: int = 5) -> nx.DiGraph:
    """Build a simple chain graph n0->n1->...->n(n-1)."""
    g = nx.DiGraph()
    for i in range(n):
        g.add_node(f"n{i}", label="Node", labels=["Node"], properties={})
    for i in range(n - 1):
        g.add_edge(f"n{i}", f"n{i + 1}", type="REF")
    return g


@pytest.fixture
def subgraph_service():
    return AsyncMock(spec=SubgraphService)


@pytest.fixture
def graph_client():
    return AsyncMock()


@pytest.fixture
def service(subgraph_service, graph_client):
    return CentralityService(subgraph_service=subgraph_service, graph_client=graph_client)


@pytest.mark.unit
async def test_local_pagerank_star_graph(service, subgraph_service):
    """Star graph: all spokes point to center — center must have the HIGHEST pagerank score."""
    star = _make_star_graph(n_spokes=5)
    subgraph_service.extract_subgraph.return_value = _make_subgraph_data_from_graph(star, entity_id="center")

    result = await service.compute_authority(entity_id="center", algorithm="pagerank", top_k=10)

    assert result.algorithm == "pagerank"
    assert result.entity_count == 6  # center + 5 spokes

    # Center must be the top-ranked entity
    assert result.top_entities[0]["id"] == "center"
    assert result.top_entities[0]["label"] == "Hub"

    # Verify the scores match what nx.pagerank actually returns
    expected_scores = nx.pagerank(star)
    for entity in result.top_entities:
        assert entity["score"] == pytest.approx(expected_scores[entity["id"]], abs=1e-5)

    # Center score must be strictly greater than any spoke score
    center_score = result.scores["center"]
    for i in range(5):
        assert center_score > result.scores[f"spoke{i}"]


@pytest.mark.unit
async def test_global_delegation(service, graph_client):
    """Global centrality delegates to graph_client; entities returned in descending score order."""
    # Real graph-adapter CentralityResponse contract: entity_scores + top_entities.
    # No "scores"/"entities" keys — the endpoint carries scores+ids only, no labels.
    graph_client.get_centrality.return_value = {
        "entity_scores": {"a": 0.5, "b": 0.3, "c": 0.8, "d": 0.1},
        "top_entities": [["c", 0.8], ["a", 0.5]],
    }

    result = await service.compute_authority(entity_id=None, algorithm="pagerank", top_k=10)

    graph_client.get_centrality.assert_awaited_once()
    assert result.entity_count == 4  # one per entity_scores entry

    # Verify descending score order
    scores_in_order = [e["score"] for e in result.top_entities]
    assert scores_in_order == sorted(scores_in_order, reverse=True)

    # Top entity should be "c" (score 0.8)
    assert result.top_entities[0]["id"] == "c"
    assert result.top_entities[0]["score"] == pytest.approx(0.8)

    # Last entity should be "d" (score 0.1)
    assert result.top_entities[-1]["id"] == "d"
    assert result.top_entities[-1]["score"] == pytest.approx(0.1)


@pytest.mark.unit
async def test_top_k_truncation(service, subgraph_service):
    chain = _make_chain_graph(10)
    subgraph_service.extract_subgraph.return_value = _make_subgraph_data_from_graph(chain, entity_id="n0")

    result = await service.compute_authority(entity_id="n0", top_k=3)
    assert len(result.top_entities) == 3
    # All 10 nodes should still be in scores
    assert result.entity_count == 10


@pytest.mark.unit
async def test_empty_subgraph(service, subgraph_service):
    empty = SubgraphData(
        response=SubgraphResponse(
            entity_id="x", nodes=[], edges=[], node_count=0, edge_count=0, max_hops_used=2, computation_time_ms=0.0
        ),
        graph=nx.DiGraph(),
    )
    subgraph_service.extract_subgraph.return_value = empty

    result = await service.compute_authority(entity_id="x")
    assert result.entity_count == 0
    assert result.top_entities == []


@pytest.mark.unit
async def test_eigenvector_centrality_connected_graph(service, subgraph_service):
    """Eigenvector centrality succeeds on a connected graph (no fallback needed)."""
    star = _make_star_graph(n_spokes=4)
    subgraph_service.extract_subgraph.return_value = _make_subgraph_data_from_graph(star, entity_id="center")

    result = await service.compute_authority(entity_id="center", algorithm="eigenvector", top_k=10)

    assert result.algorithm == "eigenvector"
    assert result.entity_count == 5

    # Verify scores are valid (all positive, sum to something reasonable)
    assert all(s > 0 for s in result.scores.values())

    # The center (only node with incoming edges) should rank highest
    assert result.top_entities[0]["id"] == "center"


@pytest.mark.unit
async def test_eigenvector_centrality_disconnected_graph_falls_back(service, subgraph_service):
    """Eigenvector centrality on a DISCONNECTED graph falls back to pagerank."""
    # Create a disconnected graph: two isolated components with no connecting edges
    g = nx.DiGraph()
    # Component 1
    g.add_node("a1", label="A1", labels=["Node"], properties={})
    g.add_node("a2", label="A2", labels=["Node"], properties={})
    g.add_edge("a1", "a2", type="REF")
    # Component 2 (disconnected from component 1)
    g.add_node("b1", label="B1", labels=["Node"], properties={})
    g.add_node("b2", label="B2", labels=["Node"], properties={})
    g.add_edge("b1", "b2", type="REF")

    subgraph_service.extract_subgraph.return_value = _make_subgraph_data_from_graph(g, entity_id="a1")

    result = await service.compute_authority(entity_id="a1", algorithm="eigenvector", top_k=10)

    # Algorithm name stays "eigenvector" (it's the requested algo, even though fallback happened)
    assert result.algorithm == "eigenvector"
    assert result.entity_count == 4

    # Scores should still be valid (pagerank fallback works on disconnected graphs)
    assert all(s > 0 for s in result.scores.values())
    # Verify the scores match nx.pagerank (the fallback)
    expected_scores = nx.pagerank(g)
    for nid, score in result.scores.items():
        assert score == pytest.approx(expected_scores[nid], abs=1e-5)


# ---------------------------------------------------------------------------
# Cache integration tests for _global_centrality
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_global_centrality_cache_miss_then_hit(subgraph_service, graph_client):
    """First global call is a cache miss, second is a hit — graph_client only called once."""
    cache = CacheService(ttl_seconds=300, max_entries=100)
    svc = CentralityService(subgraph_service=subgraph_service, graph_client=graph_client, cache=cache)

    graph_client.get_centrality.return_value = {
        "entity_scores": {"a": 0.5, "b": 0.3},
        "top_entities": [["a", 0.5], ["b", 0.3]],
    }

    result1 = await svc.compute_authority(entity_id=None, algorithm="pagerank", tenant_id="t1")
    assert graph_client.get_centrality.await_count == 1
    assert result1.entity_count == 2

    result2 = await svc.compute_authority(entity_id=None, algorithm="pagerank", tenant_id="t1")
    assert graph_client.get_centrality.await_count == 1  # still 1 — cache hit
    assert result2.entity_count == 2

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


@pytest.mark.unit
async def test_global_centrality_cache_none_no_caching(subgraph_service, graph_client):
    """When cache=None, every global call queries the graph_client."""
    svc = CentralityService(subgraph_service=subgraph_service, graph_client=graph_client, cache=None)

    graph_client.get_centrality.return_value = {
        "entity_scores": {"a": 0.5},
        "top_entities": [["a", 0.5]],
    }

    await svc.compute_authority(entity_id=None, algorithm="pagerank", tenant_id="t1")
    await svc.compute_authority(entity_id=None, algorithm="pagerank", tenant_id="t1")
    assert graph_client.get_centrality.await_count == 2


@pytest.mark.unit
async def test_global_centrality_cache_key_varies(subgraph_service, graph_client):
    """Different algorithm / tenant_id produce separate cache entries."""
    cache = CacheService(ttl_seconds=300, max_entries=100)
    svc = CentralityService(subgraph_service=subgraph_service, graph_client=graph_client, cache=cache)

    graph_client.get_centrality.return_value = {
        "entity_scores": {"a": 0.5},
        "top_entities": [["a", 0.5]],
    }

    await svc.compute_authority(entity_id=None, algorithm="pagerank", tenant_id="t1")
    await svc.compute_authority(entity_id=None, algorithm="eigenvector", tenant_id="t1")  # different algo
    await svc.compute_authority(entity_id=None, algorithm="pagerank", tenant_id="t2")  # different tenant

    assert graph_client.get_centrality.await_count == 3
    assert cache.stats()["misses"] == 3
    assert cache.stats()["hits"] == 0


@pytest.mark.unit
async def test_local_centrality_not_cached_by_centrality_service(subgraph_service, graph_client):
    """_local_centrality delegates to subgraph_service which has its own cache — centrality cache is for global only."""
    cache = CacheService(ttl_seconds=300, max_entries=100)
    svc = CentralityService(subgraph_service=subgraph_service, graph_client=graph_client, cache=cache)

    star = _make_star_graph(n_spokes=3)
    subgraph_service.extract_subgraph.return_value = _make_subgraph_data_from_graph(star, entity_id="center")

    await svc.compute_authority(entity_id="center", algorithm="pagerank")
    await svc.compute_authority(entity_id="center", algorithm="pagerank")

    # Both calls go through subgraph_service (not centrality cache)
    assert subgraph_service.extract_subgraph.await_count == 2
    # Centrality cache should not be touched for local queries
    assert cache.stats()["hits"] == 0
    assert cache.stats()["misses"] == 0


# ---------------------------------------------------------------------------
# Event-loop floor. Recovered from the stranded PR #51, whose production fix
# (asyncio.to_thread around _run_centrality) landed via #39 WITHOUT this guard —
# so nothing on main could catch its removal.
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_local_centrality_runs_off_the_event_loop(service, subgraph_service, monkeypatch):
    """PageRank on a subgraph of up to ``max_nodes=5000`` is CPU-bound.

    Running it inline froze the event loop for the whole worker — every concurrent
    request, ``/health`` included, stalled until it finished. It must be offloaded
    to a worker thread.

    Asserts on the THREAD the compute actually ran on rather than on the presence
    of an ``asyncio.to_thread`` call: a wrapper that awaits the coroutine on the
    loop would still satisfy a call-count assertion.
    """
    import threading

    star = _make_star_graph(n_spokes=5)
    subgraph_service.extract_subgraph.return_value = _make_subgraph_data_from_graph(star, entity_id="center")

    loop_thread = threading.current_thread()
    observed: list[threading.Thread] = []

    def _recording_centrality(graph, algorithm):
        observed.append(threading.current_thread())
        return nx.pagerank(graph)

    monkeypatch.setattr(CentralityService, "_run_centrality", staticmethod(_recording_centrality))

    result = await service.compute_authority(entity_id="center", algorithm="pagerank", top_k=3)

    assert result.top_entities[0]["id"] == "center"
    assert observed, "_run_centrality was never called"
    assert observed[0] is not loop_thread, "_run_centrality ran on the event loop thread — it must be offloaded"
