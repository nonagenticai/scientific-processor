"""Tests for topology router endpoints — full-stack integration through real services.

Instead of mocking entire services, we only mock get_graph_client (the HTTP boundary)
and let the real SubgraphService, ContradictionService, CentralityService, and
CommunityService run end-to-end through the FastAPI router.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

os.environ["ENABLE_AUTH"] = "false"
os.environ["ENABLE_DOCS"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from app import create_app
from config import Settings
from dependencies import get_graph_client

pytestmark = pytest.mark.unit


def _cypher_response_subgraph():
    """Realistic Cypher response for subgraph extraction."""
    return {
        "results": [
            {
                "path": {
                    "nodes": [
                        {"id": "claim1", "labels": ["Claim"], "properties": {"text": "X causes Y"}},
                        {"id": "paper1", "labels": ["Paper"], "properties": {"title": "Study Alpha"}},
                        {"id": "paper2", "labels": ["Paper"], "properties": {"title": "Study Beta"}},
                    ],
                    "relationships": [
                        {
                            "start_node_id": "paper1",
                            "end_node_id": "claim1",
                            "type": "SUPPORTS",
                            "properties": {"confidence": 0.95},
                        },
                        {
                            "start_node_id": "paper2",
                            "end_node_id": "claim1",
                            "type": "REFUTES",
                            "properties": {"confidence": 0.80},
                        },
                    ],
                }
            }
        ],
    }


def _cypher_response_star():
    """Cypher response for a star graph (spokes -> center) for authority scoring."""
    nodes = [{"id": "hub", "labels": ["Claim"], "properties": {"text": "Central claim"}}]
    rels = []
    for i in range(5):
        nodes.append({"id": f"spoke{i}", "labels": ["Paper"], "properties": {"title": f"Paper {i}"}})
        rels.append(
            {
                "start_node_id": f"spoke{i}",
                "end_node_id": "hub",
                "type": "SUPPORTS",
                "properties": {"confidence": 0.9},
            }
        )
    return {"results": [{"path": {"nodes": nodes, "relationships": rels}}]}


@pytest.fixture
def mock_graph_client():
    """Mock GraphAdapterClient -- the only thing we mock."""
    client = AsyncMock()
    client.get_subgraph = AsyncMock()
    client.get_centrality = AsyncMock()
    client.get_communities = AsyncMock()
    return client


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def app(settings, mock_graph_client):
    application = create_app(settings)
    # Override ONLY the graph client -- all services are real
    application.dependency_overrides[get_graph_client] = lambda: mock_graph_client
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
def client(app):
    return TestClient(app)


# ---- Subgraph endpoint ----


def test_subgraph_full_stack(client, mock_graph_client):
    """Full stack: router -> DI -> SubgraphService -> mocked graph client."""
    mock_graph_client.get_subgraph.return_value = _cypher_response_subgraph()

    resp = client.post("/api/topology/subgraph", json={"entity_id": "claim1", "max_hops": 2})

    assert resp.status_code == 200
    data = resp.json()

    # Verify the service produced correct structure
    assert data["entity_id"] == "claim1"
    assert data["node_count"] == 3
    assert data["edge_count"] == 2
    assert data["max_hops_used"] == 2

    # Verify actual node data (not just counts)
    nodes_by_id = {n["id"]: n for n in data["nodes"]}
    assert "claim1" in nodes_by_id
    assert nodes_by_id["claim1"]["label"] == "Claim"
    assert nodes_by_id["claim1"]["properties"]["text"] == "X causes Y"
    assert nodes_by_id["paper1"]["label"] == "Paper"
    assert nodes_by_id["paper1"]["properties"]["title"] == "Study Alpha"

    # Verify actual edge data
    edges_by_type = {(e["source_id"], e["type"]): e for e in data["edges"]}
    assert ("paper1", "SUPPORTS") in edges_by_type
    assert edges_by_type[("paper1", "SUPPORTS")]["properties"]["confidence"] == pytest.approx(0.95)
    assert ("paper2", "REFUTES") in edges_by_type

    # Verify the structured endpoint was called with the right args
    call_kwargs = mock_graph_client.get_subgraph.call_args.kwargs
    assert call_kwargs["entity_id"] == "claim1"
    assert call_kwargs["max_hops"] == 2


def test_subgraph_empty_response(client, mock_graph_client):
    mock_graph_client.get_subgraph.return_value = {"results": []}
    resp = client.post("/api/topology/subgraph", json={"entity_id": "missing"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_count"] == 0
    assert data["edge_count"] == 0


def test_subgraph_graph_error_returns_500(client, mock_graph_client):
    mock_graph_client.get_subgraph.side_effect = KeyError("unexpected error")
    resp = client.post("/api/topology/subgraph", json={"entity_id": "n1"})
    assert resp.status_code == 500


def test_subgraph_graph_unavailable_returns_503(client, mock_graph_client):
    from clients.graph_adapter import GraphServiceUnavailableError

    mock_graph_client.get_subgraph.side_effect = GraphServiceUnavailableError("Graph adapter unreachable")
    resp = client.post("/api/topology/subgraph", json={"entity_id": "n1"})
    assert resp.status_code == 503
    assert "unreachable" in resp.json()["detail"].lower()


def test_subgraph_graph_adapter_error_passes_status(client, mock_graph_client):
    from clients.graph_adapter import GraphAdapterError

    mock_graph_client.get_subgraph.side_effect = GraphAdapterError("Not found in graph", status_code=404)
    resp = client.post("/api/topology/subgraph", json={"entity_id": "n1"})
    assert resp.status_code == 404


def test_contradictions_graph_unavailable_returns_503(client, mock_graph_client):
    from clients.graph_adapter import GraphServiceUnavailableError

    mock_graph_client.get_subgraph.side_effect = GraphServiceUnavailableError()
    resp = client.post("/api/topology/contradictions", json={"entity_id": "n1"})
    assert resp.status_code == 503


def test_authority_graph_unavailable_returns_503(client, mock_graph_client):
    from clients.graph_adapter import GraphServiceUnavailableError

    mock_graph_client.get_subgraph.side_effect = GraphServiceUnavailableError()
    resp = client.post("/api/topology/authority", json={"entity_id": "n1"})
    assert resp.status_code == 503


def test_communities_graph_unavailable_returns_503(client, mock_graph_client):
    from clients.graph_adapter import GraphServiceUnavailableError

    mock_graph_client.get_communities.side_effect = GraphServiceUnavailableError()
    resp = client.post("/api/topology/communities", json={})
    assert resp.status_code == 503


# ---- Contradictions endpoint ----


def test_contradictions_full_stack(client, mock_graph_client):
    """Full stack: router -> DI -> ContradictionService -> SubgraphService -> mocked client."""
    mock_graph_client.get_subgraph.return_value = _cypher_response_subgraph()

    resp = client.post("/api/topology/contradictions", json={"entity_id": "claim1"})

    assert resp.status_code == 200
    data = resp.json()

    # claim1 has both SUPPORTS (paper1) and REFUTES (paper2) -- one contradiction pair
    assert data["total_contradictions"] == 1
    assert data["entity_id"] == "claim1"

    pair = data["contradiction_pairs"][0]
    assert pair["target_id"] == "claim1"
    assert pair["target_label"] == "Claim"

    # paper1 SUPPORTS claim1
    assert len(pair["supporting_nodes"]) == 1
    assert pair["supporting_nodes"][0]["id"] == "paper1"
    assert pair["supporting_nodes"][0]["label"] == "Paper"

    # paper2 REFUTES claim1
    assert len(pair["refuting_nodes"]) == 1
    assert pair["refuting_nodes"][0]["id"] == "paper2"

    # net_support_score = (1-1)/(1+1) = 0
    assert pair["net_support_score"] == pytest.approx(0.0)


# ---- Authority endpoint ----


def test_authority_full_stack_local(client, mock_graph_client):
    """Full stack: router -> CentralityService -> SubgraphService -> mocked client."""
    mock_graph_client.get_subgraph.return_value = _cypher_response_star()

    resp = client.post("/api/topology/authority", json={"entity_id": "hub", "algorithm": "pagerank", "top_k": 3})

    assert resp.status_code == 200
    data = resp.json()

    assert data["algorithm"] == "pagerank"
    assert data["entity_count"] == 6  # hub + 5 spokes

    # top_k=3 -- only 3 entities returned
    assert len(data["top_entities"]) == 3

    # Hub (all 5 spokes point to it) must be #1
    assert data["top_entities"][0]["id"] == "hub"
    assert data["top_entities"][0]["label"] == "Claim"
    assert data["top_entities"][0]["score"] > 0

    # Entities are in descending score order
    scores = [e["score"] for e in data["top_entities"]]
    assert scores == sorted(scores, reverse=True)


def test_authority_full_stack_global(client, mock_graph_client):
    """Global authority (no entity_id) delegates to graph_client.get_centrality."""
    # Real graph-adapter CentralityResponse contract: entity_scores + top_entities
    # (scores+ids only, no per-entity labels).
    mock_graph_client.get_centrality.return_value = {
        "entity_scores": {"x": 0.7, "y": 0.2, "z": 0.1},
        "top_entities": [["x", 0.7], ["y", 0.2], ["z", 0.1]],
    }

    resp = client.post("/api/topology/authority", json={"algorithm": "pagerank", "top_k": 10})

    assert resp.status_code == 200
    data = resp.json()

    assert data["entity_count"] == 3
    assert data["top_entities"][0]["id"] == "x"
    assert data["top_entities"][0]["score"] == pytest.approx(0.7)
    assert data["top_entities"][-1]["id"] == "z"

    # get_subgraph should NOT have been called (global path skips subgraph)
    mock_graph_client.get_subgraph.assert_not_awaited()
    mock_graph_client.get_centrality.assert_awaited_once()


# ---- Communities endpoint ----


def test_communities_full_stack(client, mock_graph_client):
    """Full stack: router -> CommunityService -> mocked graph client."""
    mock_graph_client.get_communities.return_value = {
        "entity_communities": {"a": 0, "b": 0, "c": 1, "d": 1, "e": 1},
        "modularity": 0.42,
    }

    resp = client.post("/api/topology/communities", json={"algorithm": "louvain", "min_community_size": 2})

    assert resp.status_code == 200
    data = resp.json()

    assert data["algorithm"] == "louvain"
    assert data["number_of_communities"] == 2
    assert data["modularity"] == pytest.approx(0.42)

    # Community sizes computed correctly
    assert data["community_sizes"]["0"] == 2  # a, b
    assert data["community_sizes"]["1"] == 3  # c, d, e

    # Entity-community mapping preserved
    assert data["entity_communities"]["a"] == 0
    assert data["entity_communities"]["c"] == 1

    mock_graph_client.get_communities.assert_awaited_once()


# ---- Validation error tests ----


def test_subgraph_missing_entity_id_returns_422(client):
    resp = client.post("/api/topology/subgraph", json={})
    assert resp.status_code == 422


def test_subgraph_max_hops_out_of_range(client):
    resp = client.post("/api/topology/subgraph", json={"entity_id": "n1", "max_hops": 0})
    assert resp.status_code == 422
    resp = client.post("/api/topology/subgraph", json={"entity_id": "n1", "max_hops": 6})
    assert resp.status_code == 422


def test_subgraph_max_nodes_out_of_range(client):
    resp = client.post("/api/topology/subgraph", json={"entity_id": "n1", "max_nodes": 0})
    assert resp.status_code == 422
    resp = client.post("/api/topology/subgraph", json={"entity_id": "n1", "max_nodes": 5001})
    assert resp.status_code == 422


def test_subgraph_min_confidence_out_of_range(client):
    resp = client.post("/api/topology/subgraph", json={"entity_id": "n1", "min_confidence": -0.1})
    assert resp.status_code == 422
    resp = client.post("/api/topology/subgraph", json={"entity_id": "n1", "min_confidence": 1.1})
    assert resp.status_code == 422


def test_authority_invalid_algorithm(client):
    resp = client.post("/api/topology/authority", json={"algorithm": "invalid"})
    assert resp.status_code == 422


def test_authority_betweenness_not_accepted(client):
    resp = client.post("/api/topology/authority", json={"algorithm": "betweenness"})
    assert resp.status_code == 422


def test_authority_top_k_out_of_range(client):
    resp = client.post("/api/topology/authority", json={"top_k": 0})
    assert resp.status_code == 422
    resp = client.post("/api/topology/authority", json={"top_k": 501})
    assert resp.status_code == 422


def test_communities_invalid_algorithm(client):
    resp = client.post("/api/topology/communities", json={"algorithm": "spectral"})
    assert resp.status_code == 422


def test_communities_valid_algorithms_accepted(client, mock_graph_client):
    mock_graph_client.get_communities.return_value = {"entity_communities": {}, "modularity": 0.0}
    for algo in ("louvain", "leiden", "label_propagation"):
        resp = client.post("/api/topology/communities", json={"algorithm": algo})
        assert resp.status_code == 200, f"Algorithm {algo} should be accepted"


def test_communities_min_community_size_out_of_range(client):
    resp = client.post("/api/topology/communities", json={"min_community_size": 0})
    assert resp.status_code == 422


def test_contradictions_missing_entity_id_returns_422(client):
    resp = client.post("/api/topology/contradictions", json={})
    assert resp.status_code == 422


# ---- Tenant scoping ----


def test_topology_sends_a_concrete_tenant_id_to_the_graph_adapter(client, mock_graph_client):
    """Every /api/topology/* endpoint must resolve a real tenant, never None.

    ``AuthMiddleware`` is only registered when ENABLE_AUTH=true, so with auth off
    ``request.state.tenant_id`` is unset. The router used to hand-roll a
    ``_tenant_id()`` returning ``str | None``, which meant all four endpoints issued
    UNSCOPED cross-tenant graph reads AND wrote cache keys with an empty tenant
    segment (``subgraph::e1:2``) that no caller could subsequently invalidate.
    ``resolve_tenant_id`` is the chokepoint every sibling router already uses.
    """
    from auth.dependencies import DEFAULT_TENANT_ID
    from dependencies import clear_cache

    # The CacheService is a process-global singleton that outlives the app fixture —
    # reset it so these calls reach the (per-test) mock instead of a previous entry.
    clear_cache()

    mock_graph_client.get_subgraph.return_value = _cypher_response_subgraph()
    mock_graph_client.get_communities.return_value = {"entity_communities": {}, "modularity": 0.0}
    mock_graph_client.get_centrality.return_value = {"entity_scores": {}, "top_entities": []}

    assert client.post("/api/topology/subgraph", json={"entity_id": "tenant-probe-a"}).status_code == 200
    assert mock_graph_client.get_subgraph.call_args.kwargs["tenant_id"] == DEFAULT_TENANT_ID

    assert client.post("/api/topology/contradictions", json={"entity_id": "tenant-probe-b"}).status_code == 200
    assert mock_graph_client.get_subgraph.call_args.kwargs["tenant_id"] == DEFAULT_TENANT_ID

    assert client.post("/api/topology/authority", json={"algorithm": "eigenvector"}).status_code == 200
    assert mock_graph_client.get_centrality.call_args.kwargs["tenant_id"] == DEFAULT_TENANT_ID

    assert client.post("/api/topology/communities", json={}).status_code == 200
    assert mock_graph_client.get_communities.call_args.kwargs["tenant_id"] == DEFAULT_TENANT_ID
