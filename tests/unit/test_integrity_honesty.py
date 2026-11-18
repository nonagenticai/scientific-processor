"""Integrity scoring must never fabricate a score — and a fabricated score must
never reach the knowledge graph or the verification history.

Companion to the RAGAS honesty fix (#36). The failure mode here is strictly worse
than the RAGAS one: a fabricated composite is written back onto the graph node
(``integrity_score``) and into the MinIO/DB history, where it is indistinguishable
from a real measurement forever.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import networkx as nx
import pytest

from clients.graph_adapter import GraphAdapterError
from config import Settings
from models.topology import CommunityResponse, ContradictionResponse, SubgraphResponse
from services.integrity_service import IntegrityService, Unmeasurable
from services.subgraph_service import SubgraphData

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _star_subgraph(entity_id: str = "e1", supporters: int = 3, refs: int = 3) -> SubgraphData:
    graph = nx.DiGraph()
    graph.add_node(entity_id, label="Claim")
    nodes: list[dict] = [{"id": entity_id, "label": "Claim", "labels": ["Claim"], "properties": {}}]
    for i in range(supporters):
        graph.add_edge(f"sup_{i}", entity_id, type="SUPPORTS")
        nodes.append({"id": f"sup_{i}", "label": "Source", "labels": ["Source"], "properties": {}})
    for i in range(refs):
        graph.add_edge(f"xref_{i}", entity_id, type="CITES")
        nodes.append({"id": f"xref_{i}", "label": "Reference", "labels": ["Reference"], "properties": {}})

    response = SubgraphResponse(
        entity_id=entity_id,
        nodes=nodes,
        edges=[],
        node_count=len(nodes),
        edge_count=graph.number_of_edges(),
        max_hops_used=2,
        computation_time_ms=1.0,
    )
    return SubgraphData(response=response, graph=graph)


@pytest.fixture
def subgraph_service():
    mock = AsyncMock()
    mock.extract_subgraph = AsyncMock(return_value=_star_subgraph())
    return mock


@pytest.fixture
def contradiction_service():
    mock = AsyncMock()
    mock.detect_contradictions = AsyncMock(
        return_value=ContradictionResponse(
            entity_id="e1", contradiction_pairs=[], total_contradictions=0, computation_time_ms=1.0
        )
    )
    return mock


@pytest.fixture
def community_service():
    mock = AsyncMock()
    mock.detect_communities = AsyncMock(
        return_value=CommunityResponse(
            algorithm="louvain",
            entity_communities={"e1": 0},
            community_sizes={0: 5},
            number_of_communities=1,
            modularity=0.5,
            computation_time_ms=1.0,
        )
    )
    return mock


@pytest.fixture
def graph_client():
    return AsyncMock()


@pytest.fixture
def service(subgraph_service, contradiction_service, community_service, graph_client):
    return IntegrityService(
        subgraph_service=subgraph_service,
        contradiction_service=contradiction_service,
        centrality_service=AsyncMock(),
        community_service=community_service,
        settings=Settings(),
        graph_client=graph_client,
    )


# ---------------------------------------------------------------------------
# RED PROOF 1a — a community-detection outage produces a fully plausible,
# fully fabricated response: a score is present and NOTHING signals degradation.
# ---------------------------------------------------------------------------


async def test_community_outage_must_not_return_a_plausible_fabricated_score(service, community_service):
    """Community detection is down. The composite cannot be computed.

    On pristine main the caller receives ``integrity_score=0.549`` (with
    ``community_membership=0.5`` invented) and no degradation signal whatsoever —
    a fabricated number that reads exactly like a measurement.
    """
    community_service.detect_communities.side_effect = GraphAdapterError("Community service down")

    result = await service.compute_integrity(entity_id="e1")

    # The response must carry a degradation signal at all (RagasResponse/CoVeResponse shape).
    assert getattr(result, "degraded", None) is True, (
        f"IntegrityResponse signalled no degradation. Fabricated response: "
        f"integrity_score={result.integrity_score!r}, "
        f"components={ {c.name: c.score for c in result.components} }"
    )
    # And it must not invent a composite out of an uncomputable component.
    assert result.integrity_score is None, (
        f"integrity_score={result.integrity_score!r} was fabricated from an uncomputable component"
    )
    # The uncomputable component itself must be null, not a mid-range guess.
    community = next(c for c in result.components if c.name == "community_membership")
    assert community.score is None, f"community_membership.score={community.score!r} is a fabricated value"
    assert result.degraded_components == ["community_membership"]


# ---------------------------------------------------------------------------
# RED PROOF 1b — THE POINT: the fabricated value reaches the graph write.
# ---------------------------------------------------------------------------


async def test_fabricated_score_must_never_reach_the_graph_write(service, community_service, graph_client):
    """``write_back=True`` + an uncomputable component must NOT persist a score.

    On pristine main ``update_node`` is called with
    ``{"integrity_score": 0.6725, "integrity_computed_at": ...}`` — the fabrication
    is written onto the node and is indistinguishable from a real score forever.
    """
    community_service.detect_communities.side_effect = GraphAdapterError("Community service down")

    result = await service.compute_integrity(entity_id="e1", write_back=True, tenant_id="t1")

    # Persistence assertion FIRST — this is the point of the finding.
    if graph_client.update_node.called:
        props = graph_client.update_node.call_args[0][1]
        raise AssertionError(f"a degraded integrity score was persisted to the graph: {props!r}")
    assert result.degraded is True


async def test_healthy_score_is_still_written_back(service, graph_client):
    """The guard must not break the happy path: a real score still persists."""
    result = await service.compute_integrity(entity_id="e1", write_back=True, tenant_id="t1")

    assert result.degraded is False
    assert result.integrity_score is not None
    graph_client.update_node.assert_called_once()
    props = graph_client.update_node.call_args[0][1]
    assert props["integrity_score"] == result.integrity_score
    assert props.get("integrity_degraded") is False


# ---------------------------------------------------------------------------
# RED PROOF 1c — the authority component has the same fabrication.
# ---------------------------------------------------------------------------


async def test_authority_failure_must_not_fabricate(service, subgraph_service, monkeypatch):
    """A networkx failure inside PageRank must degrade, not return 0.5."""

    def _boom(*_args, **_kwargs):
        raise nx.NetworkXError("power iteration failed to converge")

    monkeypatch.setattr(nx, "pagerank", _boom)
    # >= 10 nodes forces the pagerank branch.
    subgraph_service.extract_subgraph.return_value = _star_subgraph(supporters=8, refs=8)

    result = await service.compute_integrity(entity_id="e1")

    authority = next(c for c in result.components if c.name == "authority")
    assert authority.score is None, f"authority.score={authority.score!r} is a fabricated value"
    assert result.degraded is True
    assert result.integrity_score is None


# ---------------------------------------------------------------------------
# RED PROOF 1d — the two static-method fabrications.
# ---------------------------------------------------------------------------


class TestCitationSupportDoesNotFabricate:
    """The invariant these tests exist for is "never 0.5", and it still holds.

    The *representation* of the undefined ratio was refined afterwards: it is
    ``Unmeasurable.NOT_APPLICABLE`` rather than a bare ``None``, because a bare
    ``None`` was indistinguishable from a backend outage and therefore suppressed
    the graph write-back, trapping ordinary uncited nodes in the rescan pool. Both
    assertions below still forbid the fabrication.
    """

    def test_entity_absent_from_graph_is_not_half(self):
        """The entity is not in the extracted subgraph — the ratio is undefined."""
        g = nx.DiGraph()
        g.add_node("other")
        assert IntegrityService._citation_support(g, "e1") is Unmeasurable.NOT_APPLICABLE
        assert IntegrityService._citation_support(g, "e1") != 0.5

    def test_no_in_edges_is_not_half(self):
        """0 SUPPORTS / 0 edges is undefined; 0.5 claims 'half the evidence supports'."""
        g = nx.DiGraph()
        g.add_node("e1")
        assert IntegrityService._citation_support(g, "e1") is Unmeasurable.NOT_APPLICABLE
        assert IntegrityService._citation_support(g, "e1") != 0.5

    def test_real_ratio_still_computed(self):
        g = nx.DiGraph()
        g.add_edge("a", "e1", type="SUPPORTS")
        g.add_edge("b", "e1", type="REFUTES")
        assert IntegrityService._citation_support(g, "e1") == pytest.approx(0.5)


async def test_entity_absent_from_subgraph_is_reported_as_inapplicable_not_degraded(service, subgraph_service):
    """An entity missing from its own subgraph is scored, and says which part was undefined.

    Originally this asserted the whole response degrades. That was too strong: the
    subgraph extraction *succeeded*, and cross_reference/authority both measured 0.0
    off the same graph — only citation_support had no denominator. Degrading the
    composite here meant no write-back, so the node was re-offered by
    `find_unscored_entities` on every batch, re-firing CoVe and the sandbox forever.
    """
    empty = nx.DiGraph()
    empty.add_node("someone_else")
    subgraph_service.extract_subgraph.return_value = SubgraphData(
        response=SubgraphResponse(
            entity_id="e1",
            nodes=[],
            edges=[],
            node_count=1,
            edge_count=0,
            max_hops_used=2,
            computation_time_ms=1.0,
        ),
        graph=empty,
    )

    result = await service.compute_integrity(entity_id="e1")

    assert result.degraded is False
    assert result.integrity_score is not None
    assert result.degraded_components == []
    assert "citation_support" in result.not_applicable_components
    citation = next(c for c in result.components if c.name == "citation_support")
    assert citation.score is None
    assert citation.status == "not_applicable"
