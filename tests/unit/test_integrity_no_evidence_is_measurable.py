""" "No evidence yet" is a measurement; "the backend is down" is not.

#42 correctly stopped fabricating ``0.5`` for an undefined citation ratio, but it
routed BOTH reasons a component can lack a number down the same ``None`` path:

* the community/authority backends raised — we genuinely cannot know, and
* the entity simply has no incoming edges — we looked, and there is nothing there.

Collapsing the second into the first has an unbounded-cost consequence, each link
verified on ``origin/main`` at 7b39b9c:

    integrity_service ``specs``  any None component ⇒ composite None
      → ``if response.integrity_score is None`` ⇒ **write-back suppressed**
        → ``find_unscored_entities`` keys off a *missing* ``integrity_score``
          ⇒ the node is re-offered in **every** batch
            → ``pipeline_service`` ``score is None`` ⇒ CoVe (LLM spend) **and**
              formal verification (sandbox exec) fire again, forever.

An ordinary uncited claim node can never leave the pool. These tests pin the
distinction: NOT_APPLICABLE renormalizes and persists, UNAVAILABLE still refuses.
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


def _subgraph(entity_id: str = "e1", *, present: bool = True, supporters: int = 0, refs: int = 0) -> SubgraphData:
    """Subgraph around ``entity_id``. ``present=False`` omits the entity entirely."""
    graph = nx.DiGraph()
    nodes: list[dict] = []
    if present:
        graph.add_node(entity_id, label="Claim")
        nodes.append({"id": entity_id, "label": "Claim", "labels": ["Claim"], "properties": {}})
    else:
        graph.add_node("someone_else")
    for i in range(supporters):
        graph.add_edge(f"sup_{i}", entity_id, type="SUPPORTS")
        nodes.append({"id": f"sup_{i}", "label": "Source", "labels": ["Source"], "properties": {}})
    for i in range(refs):
        graph.add_edge(f"xref_{i}", entity_id, type="CITES")
        nodes.append({"id": f"xref_{i}", "label": "Reference", "labels": ["Reference"], "properties": {}})

    return SubgraphData(
        response=SubgraphResponse(
            entity_id=entity_id,
            nodes=nodes,
            edges=[],
            node_count=max(len(nodes), 1),
            edge_count=graph.number_of_edges(),
            max_hops_used=2,
            computation_time_ms=1.0,
        ),
        graph=graph,
    )


@pytest.fixture
def subgraph_service():
    mock = AsyncMock()
    mock.extract_subgraph = AsyncMock(return_value=_subgraph())
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
def settings():
    return Settings()


@pytest.fixture
def service(subgraph_service, contradiction_service, community_service, graph_client, settings):
    return IntegrityService(
        subgraph_service=subgraph_service,
        contradiction_service=contradiction_service,
        centrality_service=AsyncMock(),
        community_service=community_service,
        settings=settings,
        graph_client=graph_client,
    )


# ---------------------------------------------------------------------------
# RED 1 — the rescan loop. An uncited node must receive a real, persisted score.
# ---------------------------------------------------------------------------


async def test_uncited_node_gets_a_real_persisted_score(service, graph_client):
    """0 in-edges is a finding, not an outage — it must not suppress the write-back.

    On 7b39b9c this response is ``integrity_score=None``, ``update_node`` is never
    called, and ``find_unscored_entities`` therefore hands the same node back on
    every batch, re-firing CoVe and the sandbox each time.
    """
    result = await service.compute_integrity(entity_id="e1", write_back=True, tenant_id="t1")

    assert result.integrity_score is not None, (
        "an uncited node was reported unmeasurable, so it is never scored, never "
        "written back, and is re-offered by find_unscored_entities forever"
    )
    assert result.degraded is False
    assert result.degraded_components == []
    assert result.not_applicable_components == ["citation_support"]

    assert graph_client.update_node.called, "not persisted ⇒ the node stays in the unscored pool forever"
    graph_client.update_node.assert_called_once()
    props = graph_client.update_node.call_args[0][1]
    assert props["integrity_score"] == result.integrity_score
    assert props["integrity_degraded"] is False
    assert props["integrity_not_applicable"] == "citation_support"


async def test_entity_absent_from_its_own_subgraph_is_also_scored(service, subgraph_service, graph_client):
    """Absent from the extracted subgraph means "no citations", exactly like 0 in-edges.

    ``_cross_reference`` and ``_authority_score`` already report 0.0 for this same
    graph — citation_support was the only component treating it as unknowable.
    """
    subgraph_service.extract_subgraph.return_value = _subgraph(present=False)

    result = await service.compute_integrity(entity_id="e1", write_back=True, tenant_id="t1")

    assert result.integrity_score is not None
    assert result.degraded is False
    assert result.not_applicable_components == ["citation_support"]
    graph_client.update_node.assert_called_once()


# ---------------------------------------------------------------------------
# RED 2 — the renormalization is the honest part: weights must be redistributed,
# not silently left summing to less than 1.
# ---------------------------------------------------------------------------


async def test_composite_renormalizes_over_the_measurable_components(service, settings):
    """Dropping a component must not quietly shrink the score's scale.

    With citation_support (0.25) inapplicable, the surviving positive weights
    (crossref 0.20, community 0.15, authority 0.25 = 0.60) are scaled by
    0.85/0.60 so the positive side still spans the same range. The contradiction
    penalty is a penalty, not a share of the total, so it is never rescaled.

    Fixture graph: crossref 0.0 (no predecessors), community 5/5 = 1.0,
    authority 0/0 ⇒ 0.0, contradiction 0.0.
    """
    result = await service.compute_integrity(entity_id="e1")

    positive_total = settings.weight_citation + settings.weight_crossref
    positive_total += settings.weight_community + settings.weight_authority
    present = settings.weight_crossref + settings.weight_community + settings.weight_authority
    expected = round((positive_total / present) * settings.weight_community * 1.0, 4)

    assert result.integrity_score == pytest.approx(expected)

    citation = next(c for c in result.components if c.name == "citation_support")
    assert citation.score is None
    assert citation.status == "not_applicable"
    assert citation.effective_weight == 0.0

    community = next(c for c in result.components if c.name == "community_membership")
    assert community.status == "measured"
    assert community.effective_weight == pytest.approx(round((positive_total / present) * settings.weight_community, 4))
    assert community.weight == settings.weight_community, "the configured weight must stay auditable"


async def test_fully_measured_response_is_unscaled(service, subgraph_service, settings):
    """No inapplicable component ⇒ effective weights equal configured weights."""
    subgraph_service.extract_subgraph.return_value = _subgraph(supporters=3, refs=3)

    result = await service.compute_integrity(entity_id="e1")

    assert result.not_applicable_components == []
    for component in result.components:
        assert component.status == "measured"
        assert component.effective_weight == pytest.approx(component.weight)


# ---------------------------------------------------------------------------
# GUARD — #42's win must survive. A real outage is still unmeasurable.
# ---------------------------------------------------------------------------


async def test_community_outage_still_degrades_and_never_persists(service, community_service, graph_client):
    community_service.detect_communities.side_effect = GraphAdapterError("Community service down")

    result = await service.compute_integrity(entity_id="e1", write_back=True, tenant_id="t1")

    assert result.integrity_score is None
    assert result.degraded is True
    assert result.degraded_components == ["community_membership"]
    assert graph_client.update_node.called is False


async def test_outage_dominates_an_inapplicable_component(service, community_service, graph_client):
    """citation N/A + community DOWN ⇒ unknown wins. Never renormalize around an outage."""
    community_service.detect_communities.side_effect = GraphAdapterError("Community service down")

    result = await service.compute_integrity(entity_id="e1", write_back=True, tenant_id="t1")

    assert result.integrity_score is None
    assert result.degraded is True
    assert result.degraded_components == ["community_membership"]
    assert result.not_applicable_components == ["citation_support"]
    assert graph_client.update_node.called is False


async def test_authority_failure_still_degrades(service, subgraph_service, graph_client, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise nx.NetworkXError("power iteration failed to converge")

    monkeypatch.setattr(nx, "pagerank", _boom)
    subgraph_service.extract_subgraph.return_value = _subgraph(supporters=8, refs=8)

    result = await service.compute_integrity(entity_id="e1", write_back=True, tenant_id="t1")

    assert result.integrity_score is None
    assert result.degraded is True
    assert "authority" in result.degraded_components
    assert graph_client.update_node.called is False


# ---------------------------------------------------------------------------
# Boundary — nothing left to renormalize over must not divide by zero.
# ---------------------------------------------------------------------------


async def test_no_measurable_positive_component_is_degraded_not_a_zero_division(
    service, community_service, monkeypatch, graph_client
):
    """If every positive component is inapplicable there is no scale to project onto.

    Contrived (crossref and authority always yield a number today), so it is
    forced at the component level — the arithmetic guard must exist regardless.
    """
    monkeypatch.setattr(IntegrityService, "_cross_reference", staticmethod(lambda *_: Unmeasurable.NOT_APPLICABLE))
    monkeypatch.setattr(IntegrityService, "_authority_score", staticmethod(lambda *_: Unmeasurable.NOT_APPLICABLE))
    community_service.detect_communities = AsyncMock(
        return_value=CommunityResponse(
            algorithm="louvain",
            entity_communities={},
            community_sizes={},
            number_of_communities=0,
            modularity=0.0,
            computation_time_ms=1.0,
        )
    )
    monkeypatch.setattr(
        IntegrityService,
        "_community_score",
        AsyncMock(return_value=Unmeasurable.NOT_APPLICABLE),
    )

    result = await service.compute_integrity(entity_id="e1", write_back=True, tenant_id="t1")

    assert result.integrity_score is None
    assert result.degraded is True
    assert result.degraded_reason is not None
    assert graph_client.update_node.called is False


# ---------------------------------------------------------------------------
# The static helper's own contract.
# ---------------------------------------------------------------------------


class TestCitationSupportSeparatesUnknownFromAbsent:
    def test_entity_absent_from_graph_is_not_applicable(self):
        g = nx.DiGraph()
        g.add_node("other")
        assert IntegrityService._citation_support(g, "e1") is Unmeasurable.NOT_APPLICABLE

    def test_no_in_edges_is_not_applicable(self):
        g = nx.DiGraph()
        g.add_node("e1")
        assert IntegrityService._citation_support(g, "e1") is Unmeasurable.NOT_APPLICABLE

    def test_it_is_still_never_a_fabricated_half(self):
        """The #42 invariant, restated: the undefined ratio never becomes 0.5."""
        g = nx.DiGraph()
        g.add_node("e1")
        assert IntegrityService._citation_support(g, "e1") != 0.5

    def test_real_ratio_still_computed(self):
        g = nx.DiGraph()
        g.add_edge("a", "e1", type="SUPPORTS")
        g.add_edge("b", "e1", type="REFUTES")
        assert IntegrityService._citation_support(g, "e1") == pytest.approx(0.5)
