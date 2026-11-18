"""Scientific integrity scoring — composite of citation, crossref, community, authority, contradiction."""

from __future__ import annotations

import asyncio
import math
import time
from datetime import UTC, datetime
from enum import Enum

import httpx
import networkx

from clients.graph_adapter import GraphAdapterClient, GraphAdapterError
from config import Settings
from log_config import get_logger
from models.verification import ComponentStatus, IntegrityComponent, IntegrityResponse
from services.centrality_service import CentralityService
from services.community_service import CommunityService
from services.contradiction_service import ContradictionService
from services.subgraph_service import SubgraphService

logger = get_logger(__name__)


class Unmeasurable(Enum):
    """Why a component helper returned no number.

    #42 removed the fabricated ``0.5`` by returning ``None`` from every such
    branch, which was right about the fabrication and wrong about the taxonomy:
    it made "the community backend is down" and "this claim has no incoming
    edges" the same value. They are not the same, and the difference is not
    cosmetic — a ``None`` composite suppresses the graph write-back, and
    ``find_unscored_entities`` keys off a *missing* ``integrity_score``, so an
    ordinary uncited node was re-offered on every batch and re-fired CoVe (LLM)
    plus formal verification (sandbox) forever.

    ``UNAVAILABLE``
        We could not look. Backend outage, failed computation. The composite is
        genuinely unknown, stays ``None``, and is never persisted (#42's win).

    ``NOT_APPLICABLE``
        We looked and the metric has no denominator here (0 in-edges ⇒ 0/0).
        That is a finding, not an outage. The component contributes nothing and
        its weight is redistributed over the applicable components, so the entity
        still receives a real, persisted score and leaves the rescan pool.
    """

    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


#: What a component helper may return: a measured ratio, or a reason it has none.
ComponentValue = float | Unmeasurable


def _status(score: ComponentValue) -> ComponentStatus:
    """Map a component's value onto the wire-level status."""
    if isinstance(score, float):
        return "measured"
    return "unavailable" if score is Unmeasurable.UNAVAILABLE else "not_applicable"


class IntegrityService:
    def __init__(
        self,
        subgraph_service: SubgraphService,
        contradiction_service: ContradictionService,
        centrality_service: CentralityService,
        community_service: CommunityService,
        settings: Settings,
        graph_client: GraphAdapterClient | None = None,
    ) -> None:
        self.subgraph_service = subgraph_service
        self.contradiction_service = contradiction_service
        self.centrality_service = centrality_service
        self.community_service = community_service
        self.graph_client = graph_client
        self.w_citation = settings.weight_citation
        self.w_crossref = settings.weight_crossref
        self.w_community = settings.weight_community
        self.w_authority = settings.weight_authority
        self.w_contradiction = settings.weight_contradiction

    async def compute_integrity(
        self,
        entity_id: str,
        max_hops: int = 2,
        tenant_id: str | None = None,
        write_back: bool = False,
    ) -> IntegrityResponse:
        start = time.monotonic()

        # Extract subgraph once — reused for citation + crossref
        data = await self.subgraph_service.extract_subgraph(
            entity_id=entity_id,
            max_hops=max_hops,
            tenant_id=tenant_id,
        )
        graph = data.graph

        # 1. Citation support: SUPPORTS edges into entity / total edges into entity
        citation_score = self._citation_support(graph, entity_id)

        # 2. Cross-reference: distinct sources referencing entity (log-normalized)
        crossref_score = self._cross_reference(graph, entity_id)

        # 3. Community: relative size of entity's community
        community_score = await self._community_score(entity_id, tenant_id)

        # 4. Authority: entity's centrality normalized to max in subgraph.
        # PageRank is CPU-bound (iterative power method) — offload to a worker
        # thread so it never blocks the event loop on large subgraphs.
        authority_score = await asyncio.to_thread(self._authority_score, graph, entity_id)

        # 5. Contradiction penalty
        contradiction_result = await self.contradiction_service.detect_contradictions(
            entity_id=entity_id,
            max_hops=max_hops,
            tenant_id=tenant_id,
        )
        contradiction_score = self._contradiction_penalty(contradiction_result.contradiction_pairs)

        # Each entry: (name, measured score or an Unmeasurable reason, weight, sign)
        specs: list[tuple[str, ComponentValue, float, float]] = [
            ("citation_support", citation_score, self.w_citation, 1.0),
            ("cross_reference", crossref_score, self.w_crossref, 1.0),
            ("community_membership", community_score, self.w_community, 1.0),
            ("authority", authority_score, self.w_authority, 1.0),
            ("contradiction_penalty", contradiction_score, self.w_contradiction, -1.0),
        ]

        degraded_components = [n for n, score, _, _ in specs if score is Unmeasurable.UNAVAILABLE]
        not_applicable = [n for n, score, _, _ in specs if score is Unmeasurable.NOT_APPLICABLE]

        # Redistribute an inapplicable component's share over the ones that DO have
        # a number, so the positive side of the composite still spans its full
        # configured range. Without this, dropping citation_support (0.25 of 0.85)
        # would silently cap every uncited entity at 0.60 — a different fabrication.
        # The contradiction penalty is subtracted, not a share of the total, so it
        # is never rescaled.
        positive_total = sum(w for _, _, w, sign in specs if sign > 0)
        positive_present = sum(w for _, score, w, sign in specs if sign > 0 and isinstance(score, float))
        # 1.0 keeps the fully-measured path bit-identical to pre-change arithmetic.
        scale = 1.0
        if not_applicable and positive_present > 0:
            scale = positive_total / positive_present

        def _effective(score: ComponentValue, weight: float, sign: float) -> float:
            """The multiplier this component actually contributes.

            Rounded, and the composite below multiplies by *this* value rather than
            an unrounded one, so `effective_weight` on the response is exactly the
            number used — the renormalized score is reproducible from the payload
            alone. A no-op on the fully-measured path, where scale is 1.0.
            """
            if not isinstance(score, float):
                return 0.0
            return round(weight * (scale if sign > 0 else 1.0), 4)

        components = [
            IntegrityComponent(
                name=name,
                score=round(score, 4) if isinstance(score, float) else None,
                status=_status(score),
                weight=weight,
                weighted_score=(
                    round(sign * _effective(score, weight, sign) * score, 4) if isinstance(score, float) else None
                ),
                effective_weight=_effective(score, weight, sign),
            )
            for name, score, weight, sign in specs
        ]

        # Composite score. A weighted sum over components where one term is
        # UNKNOWN is not a measurement of anything — do not invent one. This
        # mirrors RagasResponse.factual_accuracy, which is None when either of
        # its inputs is None (#36). A term that is merely INAPPLICABLE is a
        # different case: it is a real observation about the entity, so the
        # composite is renormalized rather than discarded.
        integrity: float | None
        degraded_reason: str | None = None
        if degraded_components:
            integrity = None
            degraded_reason = f"integrity_score is unavailable: {', '.join(degraded_components)} could not be measured"
            logger.warning(
                "Integrity degraded — refusing to fabricate a composite",
                entity_id=entity_id,
                degraded_components=degraded_components,
                not_applicable_components=not_applicable,
            )
        elif positive_present <= 0:
            # Nothing positive was applicable, so there is no scale to project
            # onto. Reporting 0.0 here would assert "measured, and it is as bad as
            # possible", which is exactly the fabrication class #42 removed.
            integrity = None
            degraded_reason = (
                "integrity_score is unavailable: no positive component was applicable "
                f"({', '.join(not_applicable) or 'none measurable'}), so there is nothing to renormalize over"
            )
            logger.warning(
                "Integrity has no applicable positive component — refusing to report 0.0 as a measurement",
                entity_id=entity_id,
                not_applicable_components=not_applicable,
            )
        else:
            raw = sum(
                sign * _effective(score, weight, sign) * score
                for _, score, weight, sign in specs
                if isinstance(score, float)
            )
            integrity = round(max(0.0, min(1.0, raw)), 4)
            if not_applicable:
                logger.info(
                    "Integrity renormalized over the applicable components",
                    entity_id=entity_id,
                    not_applicable_components=not_applicable,
                    scale=round(scale, 4),
                    integrity_score=integrity,
                )

        elapsed = (time.monotonic() - start) * 1000

        response = IntegrityResponse(
            entity_id=entity_id,
            integrity_score=integrity,
            components=components,
            computation_time_ms=round(elapsed, 2),
            degraded=integrity is None,
            degraded_components=degraded_components,
            degraded_reason=degraded_reason,
            not_applicable_components=not_applicable,
        )

        if write_back and self.graph_client is not None:
            # NEVER persist a degraded score. Once on the node it is
            # indistinguishable from a real measurement, and downstream
            # (find_unscored_entities, history, dashboards) treats it as one.
            if response.integrity_score is None:
                logger.warning(
                    "Skipping integrity write-back — score is degraded, not measured",
                    entity_id=entity_id,
                    degraded_components=degraded_components,
                )
            else:
                try:
                    # Structured node-property merge (reuses PUT /nodes/{id}) — raw
                    # Cypher is gated off in prod. SET n += {properties}.
                    await self.graph_client.update_node(
                        entity_id,
                        {
                            "integrity_score": response.integrity_score,
                            "integrity_computed_at": datetime.now(UTC).isoformat(),
                            # Self-describing record: a consumer reading the node can
                            # tell this was a full measurement, not a partial one.
                            "integrity_degraded": False,
                            # Which components had no denominator for this entity, so a
                            # renormalized score is never mistaken for a full one. Stored
                            # as a scalar comma-joined string, not a list — the graph
                            # adapter's PUT /nodes/{id} contract is scalar properties.
                            "integrity_not_applicable": ",".join(not_applicable),
                        },
                        tenant_id,
                    )
                except (GraphAdapterError, httpx.HTTPError) as e:
                    logger.warning("Integrity write-back failed for %s: %s", entity_id, e)

        return response

    @staticmethod
    def _citation_support(graph, entity_id: str) -> ComponentValue:
        """SUPPORTS edges into entity / total edges into entity.

        Never 0.5 for an undefined ratio (#42): 0/0 is not "half the evidence
        supports this claim". But it is not UNAVAILABLE either — the subgraph
        extraction succeeded, we counted, and the count is zero. That is
        NOT_APPLICABLE: a real observation about the entity, consistent with
        ``_cross_reference`` and ``_authority_score``, which both report 0.0 for
        this same graph rather than claiming they could not look.
        """
        if entity_id not in graph:
            return Unmeasurable.NOT_APPLICABLE
        in_edges = list(graph.in_edges(entity_id, data=True))
        if not in_edges:
            return Unmeasurable.NOT_APPLICABLE
        supports = sum(1 for _, _, d in in_edges if d.get("type", "").upper() == "SUPPORTS")
        return supports / len(in_edges)

    @staticmethod
    def _cross_reference(graph, entity_id: str) -> float:
        """Distinct source nodes referencing entity, log-normalized."""
        if entity_id not in graph:
            return 0.0
        predecessors = set(graph.predecessors(entity_id))
        if not predecessors:
            return 0.0
        # log1p normalization: log(1+n)/log(1+max_reasonable)
        # Assume 50+ distinct references is maximum quality
        return min(1.0, math.log1p(len(predecessors)) / math.log1p(50))

    async def _community_score(self, entity_id: str, tenant_id: str | None) -> ComponentValue:
        """Entity's community size relative to largest community.

        Returns UNAVAILABLE when community detection failed — an outage is not a
        score, and unlike an inapplicable metric it makes the composite unknowable.
        """
        try:
            result = await self.community_service.detect_communities(tenant_id=tenant_id)
            comm_id = result.entity_communities.get(entity_id)
            if comm_id is None:
                return 0.0
            entity_comm_size = result.community_sizes.get(comm_id, 0)
            max_comm_size = max(result.community_sizes.values()) if result.community_sizes else 1
            return entity_comm_size / max_comm_size if max_comm_size > 0 else 0.0
        except GraphAdapterError, KeyError, ZeroDivisionError:
            logger.warning("Community detection failed — reporting degraded, not a fabricated 0.5", exc_info=True)
            return Unmeasurable.UNAVAILABLE

    @staticmethod
    def _authority_score(graph, entity_id: str) -> ComponentValue:
        """Entity's PageRank score normalized to max in subgraph.

        Returns UNAVAILABLE when the centrality computation itself failed. An empty
        subgraph is 0.0, not unavailable — that is a measurement, not an outage.
        """
        if graph.number_of_nodes() == 0:
            return 0.0
        try:
            scores = dict(graph.in_degree())
            # Use in-degree as a fast proxy when graph is small
            if graph.number_of_nodes() >= 10:
                from networkx import pagerank

                scores = pagerank(graph)
            entity_score = scores.get(entity_id, 0)
            max_score = max(scores.values()) if scores else 1
            return entity_score / max_score if max_score > 0 else 0.0
        except networkx.NetworkXError, ZeroDivisionError:
            logger.warning("Authority/PageRank failed — reporting degraded, not a fabricated 0.5", exc_info=True)
            return Unmeasurable.UNAVAILABLE

    @staticmethod
    def _contradiction_penalty(contradiction_pairs: list) -> float:
        """Average inverted net_support_score across contradiction pairs. Higher = worse."""
        if not contradiction_pairs:
            return 0.0
        # net_support_score ranges from -1 (all refuting) to 1 (all supporting)
        # Penalty = average of (1 - net_support_score) / 2, so fully supported = 0, fully refuted = 1
        penalties = [(1 - pair.net_support_score) / 2 for pair in contradiction_pairs]
        return sum(penalties) / len(penalties)
