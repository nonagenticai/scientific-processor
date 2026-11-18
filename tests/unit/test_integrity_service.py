"""Tests for IntegrityService -- exact mathematical verification of composite scoring."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock

import networkx as nx
import pytest

from clients.graph_adapter import GraphAdapterError
from config import Settings
from models.topology import CommunityResponse, ContradictionPair, ContradictionResponse, SubgraphResponse
from services.integrity_service import IntegrityService, Unmeasurable
from services.subgraph_service import SubgraphData

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ABS_TOL = 1e-7  # tolerance for floating-point comparisons


def _make_settings(**overrides) -> Settings:
    return Settings(**overrides)


def _make_graph_with_entity(entity_id: str, supporters: int = 2, refuters: int = 0, refs: int = 3) -> SubgraphData:
    """Build a simple star graph: all source nodes point at *entity_id*."""
    graph = nx.DiGraph()
    graph.add_node(entity_id, label="Claim")
    nodes: list[dict] = [{"id": entity_id, "label": "Claim", "labels": ["Claim"], "properties": {}}]
    edges: list[dict] = []

    for i in range(supporters):
        nid = f"sup_{i}"
        graph.add_node(nid, label="Source")
        graph.add_edge(nid, entity_id, type="SUPPORTS")
        nodes.append({"id": nid, "label": "Source", "labels": ["Source"], "properties": {}})
        edges.append({"source_id": nid, "target_id": entity_id, "type": "SUPPORTS"})

    for i in range(refuters):
        nid = f"ref_{i}"
        graph.add_node(nid, label="Source")
        graph.add_edge(nid, entity_id, type="REFUTES")
        nodes.append({"id": nid, "label": "Source", "labels": ["Source"], "properties": {}})
        edges.append({"source_id": nid, "target_id": entity_id, "type": "REFUTES"})

    for i in range(refs):
        nid = f"xref_{i}"
        graph.add_node(nid, label="Reference")
        graph.add_edge(nid, entity_id, type="CITES")
        nodes.append({"id": nid, "label": "Reference", "labels": ["Reference"], "properties": {}})
        edges.append({"source_id": nid, "target_id": entity_id, "type": "CITES"})

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def subgraph_service():
    return AsyncMock()


@pytest.fixture
def contradiction_service():
    return AsyncMock()


@pytest.fixture
def centrality_service():
    return AsyncMock()


@pytest.fixture
def community_service():
    return AsyncMock()


@pytest.fixture
def graph_client():
    mock = AsyncMock()
    mock.execute_cypher = AsyncMock(return_value={"results": []})
    return mock


@pytest.fixture
def service(subgraph_service, contradiction_service, centrality_service, community_service, graph_client):
    settings = _make_settings()
    return IntegrityService(
        subgraph_service=subgraph_service,
        contradiction_service=contradiction_service,
        centrality_service=centrality_service,
        community_service=community_service,
        settings=settings,
        graph_client=graph_client,
    )


# ---------------------------------------------------------------------------
# test_composite_score_well_supported
# ---------------------------------------------------------------------------


async def test_composite_score_well_supported(service, subgraph_service, contradiction_service, community_service):
    """5 SUPPORTS + 10 CITES into e1 (16-node graph).  Verify exact composite score."""
    data = _make_graph_with_entity("e1", supporters=5, refuters=0, refs=10)
    subgraph_service.extract_subgraph.return_value = data
    contradiction_service.detect_contradictions.return_value = ContradictionResponse(
        entity_id="e1", contradiction_pairs=[], total_contradictions=0, computation_time_ms=1.0
    )
    community_service.detect_communities.return_value = CommunityResponse(
        algorithm="louvain",
        entity_communities={"e1": 0},
        community_sizes={0: 10},
        number_of_communities=1,
        modularity=0.5,
        computation_time_ms=1.0,
    )

    result = await service.compute_integrity(entity_id="e1")

    # --- expected component values ---
    # citation: 5 SUPPORTS / 15 total in-edges = 1/3
    expected_citation = 5 / 15
    # crossref: 15 distinct predecessors, log-normalized against 50
    expected_crossref = min(1.0, math.log1p(15) / math.log1p(50))
    # community: entity's community size (10) / largest community size (10) = 1.0
    expected_community = 1.0
    # authority: 16 nodes >= 10, so PageRank is used.
    # Star topology: all 15 leaf nodes point to e1, e1 has highest PageRank.
    pr = nx.pagerank(data.graph)
    expected_authority = pr["e1"] / max(pr.values())
    # contradiction: no pairs -> 0.0
    expected_contradiction = 0.0

    # composite (default weights: citation=0.25, crossref=0.20, community=0.15, authority=0.25, contradiction=0.15)
    raw = (
        0.25 * expected_citation
        + 0.20 * expected_crossref
        + 0.15 * expected_community
        + 0.25 * expected_authority
        - 0.15 * expected_contradiction
    )
    expected_score = max(0.0, min(1.0, raw))

    assert result.integrity_score == pytest.approx(round(expected_score, 4), abs=ABS_TOL)
    assert len(result.components) == 5

    # Verify each component individually
    by_name = {c.name: c for c in result.components}

    assert by_name["citation_support"].score == pytest.approx(round(expected_citation, 4), abs=ABS_TOL)
    assert by_name["citation_support"].weight == 0.25
    assert by_name["citation_support"].weighted_score == pytest.approx(round(0.25 * expected_citation, 4), abs=ABS_TOL)

    assert by_name["cross_reference"].score == pytest.approx(round(expected_crossref, 4), abs=ABS_TOL)
    assert by_name["cross_reference"].weight == 0.20
    assert by_name["cross_reference"].weighted_score == pytest.approx(round(0.20 * expected_crossref, 4), abs=ABS_TOL)

    assert by_name["community_membership"].score == pytest.approx(round(expected_community, 4), abs=ABS_TOL)
    assert by_name["authority"].score == pytest.approx(round(expected_authority, 4), abs=ABS_TOL)
    assert by_name["contradiction_penalty"].score == pytest.approx(0.0, abs=ABS_TOL)


# ---------------------------------------------------------------------------
# test_contradiction_lowers_score
# ---------------------------------------------------------------------------


async def test_contradiction_lowers_score(service, subgraph_service, contradiction_service, community_service):
    """2 SUPPORTS + 2 REFUTES + 2 CITES, one contradiction pair with net_support=0.0."""
    data = _make_graph_with_entity("e1", supporters=2, refuters=2, refs=2)
    subgraph_service.extract_subgraph.return_value = data
    contradiction_service.detect_contradictions.return_value = ContradictionResponse(
        entity_id="e1",
        contradiction_pairs=[
            ContradictionPair(
                target_id="e1",
                target_label="Claim",
                supporting_nodes=[{"id": "s1"}],
                refuting_nodes=[{"id": "r1"}],
                net_support_score=0.0,
            )
        ],
        total_contradictions=1,
        computation_time_ms=1.0,
    )
    community_service.detect_communities.return_value = CommunityResponse(
        algorithm="louvain",
        entity_communities={"e1": 0},
        community_sizes={0: 5},
        number_of_communities=1,
        modularity=0.3,
        computation_time_ms=1.0,
    )

    result = await service.compute_integrity(entity_id="e1")

    # --- expected component values ---
    # citation: 2 SUPPORTS / 6 total in-edges = 1/3
    expected_citation = 2 / 6
    # crossref: 6 distinct predecessors
    expected_crossref = min(1.0, math.log1p(6) / math.log1p(50))
    # community: 5/5 = 1.0
    expected_community = 1.0
    # authority: 7 nodes < 10, uses in_degree. e1 in_degree=6, max_in_degree=6 -> 1.0
    expected_authority = 1.0
    # contradiction: (1 - 0.0) / 2 = 0.5
    expected_contradiction = 0.5

    raw = (
        0.25 * expected_citation
        + 0.20 * expected_crossref
        + 0.15 * expected_community
        + 0.25 * expected_authority
        - 0.15 * expected_contradiction
    )
    expected_score = max(0.0, min(1.0, raw))

    assert result.integrity_score == pytest.approx(round(expected_score, 4), abs=ABS_TOL)

    # Verify the contradiction component directly
    contra = next(c for c in result.components if c.name == "contradiction_penalty")
    assert contra.score == pytest.approx(round(expected_contradiction, 4), abs=ABS_TOL)
    assert contra.weighted_score == pytest.approx(round(-0.15 * expected_contradiction, 4), abs=ABS_TOL)
    assert contra.weighted_score < 0  # penalty is negative


# ---------------------------------------------------------------------------
# test_score_clamped_to_unit_range
# ---------------------------------------------------------------------------


def _one_refuting_contradiction() -> ContradictionResponse:
    """A single fully-refuted pair -> penalty = (1 - (-1)) / 2 = 1.0."""
    return ContradictionResponse(
        entity_id="e1",
        contradiction_pairs=[
            ContradictionPair(
                target_id="e1",
                target_label="Claim",
                supporting_nodes=[],
                refuting_nodes=[{"id": "r1"}, {"id": "r2"}],
                net_support_score=-1.0,
            )
        ],
        total_contradictions=1,
        computation_time_ms=1.0,
    )


async def test_score_clamped_to_unit_range(
    subgraph_service, contradiction_service, centrality_service, community_service, graph_client
):
    """A negative raw composite clamps to 0.0 (never below the unit range).

    Every component must be *measurable* here, otherwise the composite is None by
    design and this asserts nothing about clamping. So the entity carries a real
    in-edge (citation_support = 0/1 = 0.0) and the contradiction weight dominates.
    """
    svc = IntegrityService(
        subgraph_service=subgraph_service,
        contradiction_service=contradiction_service,
        centrality_service=centrality_service,
        community_service=community_service,
        settings=_make_settings(
            weight_citation=0.05,
            weight_crossref=0.05,
            weight_community=0.05,
            weight_authority=0.05,
            weight_contradiction=0.80,
        ),
        graph_client=graph_client,
    )
    subgraph_service.extract_subgraph.return_value = _make_graph_with_entity("e1", supporters=0, refuters=1, refs=0)
    contradiction_service.detect_contradictions.return_value = _one_refuting_contradiction()
    community_service.detect_communities.return_value = CommunityResponse(
        algorithm="louvain",
        entity_communities={},
        community_sizes={},
        number_of_communities=0,
        modularity=0.0,
        computation_time_ms=1.0,
    )

    result = await svc.compute_integrity(entity_id="e1")

    # --- expected component values ---
    expected_citation = 0.0  # 1 in-edge, 0 SUPPORTS
    expected_crossref = math.log1p(1) / math.log1p(50)  # 1 distinct predecessor
    expected_community = 0.0  # e1 not in entity_communities
    expected_authority = 1.0  # 2 nodes; e1 has the only in-edge
    expected_contradiction = 1.0

    raw = (
        0.05 * expected_citation
        + 0.05 * expected_crossref
        + 0.05 * expected_community
        + 0.05 * expected_authority
        - 0.80 * expected_contradiction
    )
    assert raw < 0, f"raw should be negative but got {raw}"
    assert result.degraded is False, f"components unexpectedly unmeasurable: {result.degraded_components}"
    assert result.integrity_score == 0.0


async def test_isolated_node_scores_zero_over_its_measurable_components(
    service, subgraph_service, contradiction_service, community_service
):
    """An entity with no edges at all is measurably unsupported — that is 0.0, not unknown.

    citation_support is 0/0 and therefore NOT_APPLICABLE, but cross_reference,
    community_membership and authority all successfully measured 0.0 for this same
    graph. Renormalizing over them yields a real 0.0 (a refuting contradiction then
    drives it below the clamp). Reporting `None` instead — as #42 did — suppressed
    the write-back, so `find_unscored_entities` re-offered this node on every batch
    and re-fired CoVe and the sandbox forever. See
    tests/unit/test_integrity_no_evidence_is_measurable.py.
    """
    subgraph_service.extract_subgraph.return_value = _make_graph_with_entity("e1", supporters=0, refuters=0, refs=0)
    contradiction_service.detect_contradictions.return_value = _one_refuting_contradiction()
    community_service.detect_communities.return_value = CommunityResponse(
        algorithm="louvain",
        entity_communities={},
        community_sizes={},
        number_of_communities=0,
        modularity=0.0,
        computation_time_ms=1.0,
    )

    result = await service.compute_integrity(entity_id="e1")

    assert result.integrity_score == 0.0
    assert result.degraded is False
    assert result.degraded_components == []
    assert result.not_applicable_components == ["citation_support"]


# ---------------------------------------------------------------------------
# test_custom_weights
# ---------------------------------------------------------------------------


async def test_custom_weights(
    subgraph_service, contradiction_service, centrality_service, community_service, graph_client
):
    """Non-default weights change the composite score according to the formula."""
    custom = _make_settings(
        weight_citation=0.40,
        weight_crossref=0.10,
        weight_community=0.10,
        weight_authority=0.30,
        weight_contradiction=0.10,
    )
    svc = IntegrityService(
        subgraph_service=subgraph_service,
        contradiction_service=contradiction_service,
        centrality_service=centrality_service,
        community_service=community_service,
        settings=custom,
        graph_client=graph_client,
    )

    data = _make_graph_with_entity("e1", supporters=3, refuters=1, refs=4)
    subgraph_service.extract_subgraph.return_value = data
    contradiction_service.detect_contradictions.return_value = ContradictionResponse(
        entity_id="e1",
        contradiction_pairs=[
            ContradictionPair(
                target_id="e1",
                target_label="Claim",
                supporting_nodes=[{"id": "s1"}],
                refuting_nodes=[{"id": "r1"}],
                net_support_score=0.5,
            )
        ],
        total_contradictions=1,
        computation_time_ms=1.0,
    )
    community_service.detect_communities.return_value = CommunityResponse(
        algorithm="louvain",
        entity_communities={"e1": 0},
        community_sizes={0: 8},
        number_of_communities=1,
        modularity=0.4,
        computation_time_ms=1.0,
    )

    result = await svc.compute_integrity(entity_id="e1")

    # --- expected component values ---
    # citation: 3 SUPPORTS / 8 total (3 sup + 1 ref + 4 cites) = 3/8
    expected_citation = 3 / 8
    # crossref: 8 predecessors
    expected_crossref = min(1.0, math.log1p(8) / math.log1p(50))
    # community: 8/8 = 1.0
    expected_community = 1.0
    # authority: 9 nodes < 10, in_degree. e1=8, max=8 -> 1.0
    expected_authority = 1.0
    # contradiction: (1 - 0.5) / 2 = 0.25
    expected_contradiction = 0.25

    raw = (
        0.40 * expected_citation
        + 0.10 * expected_crossref
        + 0.10 * expected_community
        + 0.30 * expected_authority
        - 0.10 * expected_contradiction
    )
    expected_score = max(0.0, min(1.0, raw))

    assert result.integrity_score == pytest.approx(round(expected_score, 4), abs=ABS_TOL)

    # Also verify the weights stored in components match the custom values
    by_name = {c.name: c for c in result.components}
    assert by_name["citation_support"].weight == 0.40
    assert by_name["cross_reference"].weight == 0.10
    assert by_name["community_membership"].weight == 0.10
    assert by_name["authority"].weight == 0.30
    assert by_name["contradiction_penalty"].weight == 0.10


# ---------------------------------------------------------------------------
# test_each_component_formula
# ---------------------------------------------------------------------------


class TestCitationSupport:
    """Test IntegrityService._citation_support static method."""

    def test_all_supports(self):
        g = nx.DiGraph()
        g.add_edge("a", "e1", type="SUPPORTS")
        g.add_edge("b", "e1", type="SUPPORTS")
        assert IntegrityService._citation_support(g, "e1") == pytest.approx(1.0, abs=ABS_TOL)

    def test_mixed(self):
        g = nx.DiGraph()
        g.add_edge("a", "e1", type="SUPPORTS")
        g.add_edge("b", "e1", type="REFUTES")
        g.add_edge("c", "e1", type="CITES")
        # 1 SUPPORTS / 3 total = 1/3
        assert IntegrityService._citation_support(g, "e1") == pytest.approx(1 / 3, abs=ABS_TOL)

    def test_no_in_edges_is_not_applicable(self):
        """0 SUPPORTS / 0 edges is undefined, so the component reports NOT_APPLICABLE.

        This used to return a "neutral" 0.5, which is a claim ("half the evidence
        supports this") that no evidence was ever gathered to support. It must also
        not be UNAVAILABLE: the count succeeded and came back zero, which is a
        finding about the entity, not an outage. See
        tests/unit/test_integrity_honesty.py and
        tests/unit/test_integrity_no_evidence_is_measurable.py.
        """
        g = nx.DiGraph()
        g.add_node("e1")
        assert IntegrityService._citation_support(g, "e1") is Unmeasurable.NOT_APPLICABLE
        assert IntegrityService._citation_support(g, "e1") != 0.5

    def test_all_refutes(self):
        g = nx.DiGraph()
        g.add_edge("a", "e1", type="REFUTES")
        g.add_edge("b", "e1", type="REFUTES")
        # 0 SUPPORTS / 2 total = 0.0
        assert IntegrityService._citation_support(g, "e1") == pytest.approx(0.0, abs=ABS_TOL)


class TestCrossReference:
    """Test IntegrityService._cross_reference static method."""

    def test_known_predecessor_count(self):
        g = nx.DiGraph()
        for i in range(20):
            g.add_edge(f"n{i}", "e1")
        expected = min(1.0, math.log1p(20) / math.log1p(50))
        assert IntegrityService._cross_reference(g, "e1") == pytest.approx(expected, abs=ABS_TOL)

    def test_no_predecessors(self):
        g = nx.DiGraph()
        g.add_node("e1")
        assert IntegrityService._cross_reference(g, "e1") == pytest.approx(0.0, abs=ABS_TOL)

    def test_saturates_at_50_plus(self):
        g = nx.DiGraph()
        for i in range(60):
            g.add_edge(f"n{i}", "e1")
        # log1p(60) / log1p(50) > 1.0, clamped to 1.0
        assert IntegrityService._cross_reference(g, "e1") == pytest.approx(1.0, abs=ABS_TOL)

    def test_duplicate_predecessor_not_counted(self):
        """Multiple edges from the same predecessor count as one distinct predecessor."""
        g = nx.DiGraph()
        # DiGraph only allows one edge per (u,v), so predecessor count = 1
        g.add_edge("a", "e1", type="SUPPORTS")
        expected = min(1.0, math.log1p(1) / math.log1p(50))
        assert IntegrityService._cross_reference(g, "e1") == pytest.approx(expected, abs=ABS_TOL)


class TestContradictionPenalty:
    """Test IntegrityService._contradiction_penalty static method."""

    def test_no_pairs(self):
        assert IntegrityService._contradiction_penalty([]) == pytest.approx(0.0, abs=ABS_TOL)

    def test_single_pair_neutral(self):
        pair = ContradictionPair(
            target_id="e1",
            target_label="Claim",
            supporting_nodes=[],
            refuting_nodes=[],
            net_support_score=0.0,
        )
        # (1 - 0.0) / 2 = 0.5
        assert IntegrityService._contradiction_penalty([pair]) == pytest.approx(0.5, abs=ABS_TOL)

    def test_single_pair_fully_supporting(self):
        pair = ContradictionPair(
            target_id="e1",
            target_label="Claim",
            supporting_nodes=[{"id": "s1"}],
            refuting_nodes=[],
            net_support_score=1.0,
        )
        # (1 - 1.0) / 2 = 0.0
        assert IntegrityService._contradiction_penalty([pair]) == pytest.approx(0.0, abs=ABS_TOL)

    def test_single_pair_fully_refuting(self):
        pair = ContradictionPair(
            target_id="e1",
            target_label="Claim",
            supporting_nodes=[],
            refuting_nodes=[{"id": "r1"}],
            net_support_score=-1.0,
        )
        # (1 - (-1)) / 2 = 1.0
        assert IntegrityService._contradiction_penalty([pair]) == pytest.approx(1.0, abs=ABS_TOL)

    def test_multiple_pairs_averaged(self):
        p1 = ContradictionPair(
            target_id="e1",
            target_label="C",
            supporting_nodes=[],
            refuting_nodes=[],
            net_support_score=0.6,
        )
        p2 = ContradictionPair(
            target_id="e1",
            target_label="C",
            supporting_nodes=[],
            refuting_nodes=[],
            net_support_score=-0.4,
        )
        # p1 penalty = (1 - 0.6) / 2 = 0.2
        # p2 penalty = (1 - (-0.4)) / 2 = 0.7
        # average = (0.2 + 0.7) / 2 = 0.45
        assert IntegrityService._contradiction_penalty([p1, p2]) == pytest.approx(0.45, abs=ABS_TOL)


class TestAuthorityScore:
    """Test IntegrityService._authority_score static method."""

    def test_empty_graph(self):
        g = nx.DiGraph()
        assert IntegrityService._authority_score(g, "e1") == pytest.approx(0.0, abs=ABS_TOL)

    def test_small_graph_uses_in_degree(self):
        """< 10 nodes -> uses in_degree as proxy."""
        g = nx.DiGraph()
        g.add_node("e1")
        for i in range(5):
            g.add_edge(f"n{i}", "e1")
        # 6 nodes < 10. e1 in_degree=5, max_in_degree=5 -> 1.0
        assert IntegrityService._authority_score(g, "e1") == pytest.approx(1.0, abs=ABS_TOL)

    def test_small_graph_non_max(self):
        """Entity with fewer in-edges than the max."""
        g = nx.DiGraph()
        for i in range(2):
            g.add_edge(f"n{i}", "e1")
        for i in range(4):
            g.add_edge(f"m{i}", "other")
        # nodes: e1, other, n0, n1, m0, m1, m2, m3 = 8 < 10 -> in_degree
        # e1 in_degree=2, other in_degree=4, max=4
        assert IntegrityService._authority_score(g, "e1") == pytest.approx(2 / 4, abs=ABS_TOL)

    def test_large_graph_uses_pagerank(self):
        """>=10 nodes -> uses PageRank."""
        g = nx.DiGraph()
        g.add_node("e1")
        # 12 nodes pointing to e1 -> 13 total nodes >= 10
        for i in range(12):
            g.add_edge(f"n{i}", "e1")
        pr = nx.pagerank(g)
        expected = pr["e1"] / max(pr.values())
        assert IntegrityService._authority_score(g, "e1") == pytest.approx(expected, abs=ABS_TOL)


# ---------------------------------------------------------------------------
# test_community_service_failure_uses_fallback
# ---------------------------------------------------------------------------


async def test_community_service_failure_degrades_rather_than_fabricating(
    service, subgraph_service, contradiction_service, community_service
):
    """When community detection raises, the component and the composite are None.

    This test previously asserted ``community.score == 0.5`` — it *encoded* the
    fabrication it should have caught. An outage is not a measurement.
    """
    subgraph_service.extract_subgraph.return_value = _make_graph_with_entity("e1", supporters=3, refuters=0, refs=3)
    contradiction_service.detect_contradictions.return_value = ContradictionResponse(
        entity_id="e1", contradiction_pairs=[], total_contradictions=0, computation_time_ms=1.0
    )
    community_service.detect_communities.side_effect = GraphAdapterError("Community service down")

    result = await service.compute_integrity(entity_id="e1")

    assert result.integrity_score is None
    assert result.degraded is True
    assert result.degraded_components == ["community_membership"]
    community = next(c for c in result.components if c.name == "community_membership")
    assert community.score is None
    assert community.weighted_score is None
    # The components that *were* measurable are still reported.
    citation = next(c for c in result.components if c.name == "citation_support")
    assert citation.score is not None


# ---------------------------------------------------------------------------
# test_write_back
# ---------------------------------------------------------------------------


def _setup_basic_mocks(subgraph_service, contradiction_service, community_service):
    """Configure mocks for a simple integrity computation."""
    subgraph_service.extract_subgraph.return_value = _make_graph_with_entity("e1", supporters=2, refuters=0, refs=2)
    contradiction_service.detect_contradictions.return_value = ContradictionResponse(
        entity_id="e1", contradiction_pairs=[], total_contradictions=0, computation_time_ms=1.0
    )
    community_service.detect_communities.return_value = CommunityResponse(
        algorithm="louvain",
        entity_communities={"e1": 0},
        community_sizes={0: 5},
        number_of_communities=1,
        modularity=0.5,
        computation_time_ms=1.0,
    )


async def test_write_back_true_calls_cypher(
    service,
    subgraph_service,
    contradiction_service,
    community_service,
    graph_client,
):
    """When write_back=True, update_node merges integrity_score onto the node."""
    _setup_basic_mocks(subgraph_service, contradiction_service, community_service)

    result = await service.compute_integrity(entity_id="e1", write_back=True, tenant_id="t1")

    assert 0.0 <= result.integrity_score <= 1.0
    graph_client.update_node.assert_called_once()
    call_args = graph_client.update_node.call_args
    # update_node(node_id, {properties}, tenant_id)
    node_id = call_args[0][0]
    props = call_args[0][1]
    tenant = call_args[0][2]

    assert node_id == "e1"
    assert props["integrity_score"] == result.integrity_score
    assert "integrity_computed_at" in props
    assert tenant == "t1"


async def test_write_back_false_does_not_call_cypher(
    service,
    subgraph_service,
    contradiction_service,
    community_service,
    graph_client,
):
    """When write_back=False (default), update_node is NOT called."""
    _setup_basic_mocks(subgraph_service, contradiction_service, community_service)

    result = await service.compute_integrity(entity_id="e1", tenant_id="t1")

    assert 0.0 <= result.integrity_score <= 1.0
    graph_client.update_node.assert_not_called()


async def test_write_back_without_graph_client(
    subgraph_service,
    contradiction_service,
    centrality_service,
    community_service,
):
    """When graph_client is None, write_back=True is silently ignored."""
    svc = IntegrityService(
        subgraph_service=subgraph_service,
        contradiction_service=contradiction_service,
        centrality_service=centrality_service,
        community_service=community_service,
        settings=_make_settings(),
        graph_client=None,
    )
    _setup_basic_mocks(subgraph_service, contradiction_service, community_service)

    result = await svc.compute_integrity(entity_id="e1", write_back=True)

    # `integrity_score` is now `float | None` — None means "could not be measured".
    # On this healthy path it must be a real number, so assert that FIRST rather
    # than letting a None silently satisfy a range check that never ran.
    assert result.integrity_score is not None
    assert result.degraded is False
    assert 0.0 <= result.integrity_score <= 1.0
