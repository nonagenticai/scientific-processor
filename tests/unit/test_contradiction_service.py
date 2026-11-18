"""Tests for ContradictionService — SUPPORTS+REFUTES detection."""

from unittest.mock import AsyncMock

import networkx as nx
import pytest

from models.topology import SubgraphResponse
from services.contradiction_service import ContradictionService
from services.subgraph_service import SubgraphData, SubgraphService


def _make_subgraph_data(nodes, edges):
    """Build SubgraphData from simple node/edge lists."""
    graph = nx.DiGraph()
    nodes_list = []
    for n in nodes:
        graph.add_node(n["id"], **n)
        nodes_list.append(n)
    for e in edges:
        graph.add_edge(e["source_id"], e["target_id"], type=e["type"], **e.get("properties", {}))

    response = SubgraphResponse(
        entity_id=nodes[0]["id"] if nodes else "",
        nodes=nodes_list,
        edges=edges,
        node_count=len(nodes),
        edge_count=len(edges),
        max_hops_used=2,
        computation_time_ms=1.0,
    )
    return SubgraphData(response=response, graph=graph)


@pytest.fixture
def subgraph_service():
    svc = AsyncMock(spec=SubgraphService)
    return svc


@pytest.fixture
def service(subgraph_service):
    return ContradictionService(subgraph_service=subgraph_service)


@pytest.mark.unit
async def test_detects_contradiction_pair(service, subgraph_service):
    """Verify contradiction detection AND actual node data in supporting/refuting lists."""
    data = _make_subgraph_data(
        nodes=[
            {"id": "target", "label": "Claim", "labels": ["Claim"], "properties": {"text": "X causes Y"}},
            {"id": "s1", "label": "Study-A", "labels": ["Source"], "properties": {"year": 2024}},
            {"id": "s2", "label": "Study-B", "labels": ["Source"], "properties": {"year": 2025}},
        ],
        edges=[
            {"source_id": "s1", "target_id": "target", "type": "SUPPORTS"},
            {"source_id": "s2", "target_id": "target", "type": "REFUTES"},
        ],
    )
    subgraph_service.extract_subgraph.return_value = data

    result = await service.detect_contradictions(entity_id="target")

    assert result.total_contradictions == 1
    pair = result.contradiction_pairs[0]
    assert pair.target_id == "target"
    assert pair.target_label == "Claim"
    assert pair.net_support_score == pytest.approx(0.0)  # (1-1)/2 = 0

    # Verify actual supporting node data
    assert len(pair.supporting_nodes) == 1
    sup = pair.supporting_nodes[0]
    assert sup["id"] == "s1"
    assert sup["label"] == "Study-A"
    assert sup["properties"]["year"] == 2024

    # Verify actual refuting node data
    assert len(pair.refuting_nodes) == 1
    ref = pair.refuting_nodes[0]
    assert ref["id"] == "s2"
    assert ref["label"] == "Study-B"
    assert ref["properties"]["year"] == 2025


@pytest.mark.unit
async def test_net_support_score_calculation(service, subgraph_service):
    data = _make_subgraph_data(
        nodes=[
            {"id": "t1", "label": "Claim", "labels": ["Claim"], "properties": {}},
            {"id": "a", "label": "S", "labels": ["S"], "properties": {}},
            {"id": "b", "label": "S", "labels": ["S"], "properties": {}},
            {"id": "c", "label": "S", "labels": ["S"], "properties": {}},
        ],
        edges=[
            {"source_id": "a", "target_id": "t1", "type": "SUPPORTS"},
            {"source_id": "b", "target_id": "t1", "type": "SUPPORTS"},
            {"source_id": "c", "target_id": "t1", "type": "REFUTES"},
        ],
    )
    subgraph_service.extract_subgraph.return_value = data

    result = await service.detect_contradictions(entity_id="t1")
    pair = result.contradiction_pairs[0]
    # (2 - 1) / (2 + 1) = 0.3333
    assert pair.net_support_score == pytest.approx(0.3333, abs=0.001)


@pytest.mark.unit
async def test_no_contradictions_supports_only(service, subgraph_service):
    """Graph with only SUPPORTS edges produces zero contradictions."""
    data = _make_subgraph_data(
        nodes=[
            {"id": "t1", "label": "Claim", "labels": ["Claim"], "properties": {}},
            {"id": "s1", "label": "Source", "labels": ["Source"], "properties": {}},
            {"id": "s2", "label": "Source", "labels": ["Source"], "properties": {}},
        ],
        edges=[
            {"source_id": "s1", "target_id": "t1", "type": "SUPPORTS"},
            {"source_id": "s2", "target_id": "t1", "type": "SUPPORTS"},
        ],
    )
    subgraph_service.extract_subgraph.return_value = data

    result = await service.detect_contradictions(entity_id="t1")
    assert result.total_contradictions == 0
    assert result.contradiction_pairs == []


@pytest.mark.unit
async def test_multiple_contradiction_targets(service, subgraph_service):
    """Multiple targets each with both SUPPORTS and REFUTES edges produce separate pairs."""
    data = _make_subgraph_data(
        nodes=[
            {"id": "claim1", "label": "Claim-1", "labels": ["Claim"], "properties": {}},
            {"id": "claim2", "label": "Claim-2", "labels": ["Claim"], "properties": {}},
            {"id": "a", "label": "Source-A", "labels": ["Source"], "properties": {}},
            {"id": "b", "label": "Source-B", "labels": ["Source"], "properties": {}},
            {"id": "c", "label": "Source-C", "labels": ["Source"], "properties": {}},
            {"id": "d", "label": "Source-D", "labels": ["Source"], "properties": {}},
        ],
        edges=[
            # claim1 has contradiction: a supports, b refutes
            {"source_id": "a", "target_id": "claim1", "type": "SUPPORTS"},
            {"source_id": "b", "target_id": "claim1", "type": "REFUTES"},
            # claim2 has contradiction: c supports, d refutes
            {"source_id": "c", "target_id": "claim2", "type": "SUPPORTS"},
            {"source_id": "d", "target_id": "claim2", "type": "REFUTES"},
        ],
    )
    subgraph_service.extract_subgraph.return_value = data

    result = await service.detect_contradictions(entity_id="claim1")

    assert result.total_contradictions == 2
    pair_targets = {p.target_id for p in result.contradiction_pairs}
    assert pair_targets == {"claim1", "claim2"}

    # Verify each pair has the correct nodes
    for pair in result.contradiction_pairs:
        if pair.target_id == "claim1":
            assert pair.target_label == "Claim-1"
            assert pair.supporting_nodes[0]["id"] == "a"
            assert pair.refuting_nodes[0]["id"] == "b"
        elif pair.target_id == "claim2":
            assert pair.target_label == "Claim-2"
            assert pair.supporting_nodes[0]["id"] == "c"
            assert pair.refuting_nodes[0]["id"] == "d"


@pytest.mark.unit
async def test_ignores_non_support_refute_types(service, subgraph_service):
    """CITES, REF, and other relationship types should NOT count as contradictions."""
    data = _make_subgraph_data(
        nodes=[
            {"id": "target", "label": "Claim", "labels": ["Claim"], "properties": {}},
            {"id": "s1", "label": "Source", "labels": ["Source"], "properties": {}},
            {"id": "s2", "label": "Source", "labels": ["Source"], "properties": {}},
            {"id": "s3", "label": "Source", "labels": ["Source"], "properties": {}},
            {"id": "s4", "label": "Source", "labels": ["Source"], "properties": {}},
        ],
        edges=[
            # Only one SUPPORTS edge — these other types should be ignored
            {"source_id": "s1", "target_id": "target", "type": "SUPPORTS"},
            {"source_id": "s2", "target_id": "target", "type": "CITES"},
            {"source_id": "s3", "target_id": "target", "type": "REF"},
            {"source_id": "s4", "target_id": "target", "type": "REFERENCES"},
        ],
    )
    subgraph_service.extract_subgraph.return_value = data

    result = await service.detect_contradictions(entity_id="target")

    # Only SUPPORTS exists, no REFUTES — no contradiction pair
    assert result.total_contradictions == 0
    assert result.contradiction_pairs == []
