"""Contradiction detection service — finds SUPPORTS+REFUTES edges to same target."""

from __future__ import annotations

import time
from typing import Any

from log_config import get_logger
from models.topology import ContradictionPair, ContradictionResponse
from services.subgraph_service import SubgraphService

logger = get_logger(__name__)


class ContradictionService:
    def __init__(self, subgraph_service: SubgraphService) -> None:
        self.subgraph_service = subgraph_service

    async def detect_contradictions(
        self,
        entity_id: str,
        max_hops: int = 2,
        min_confidence: float = 0.0,
        tenant_id: str | None = None,
    ) -> ContradictionResponse:
        start = time.monotonic()

        data = await self.subgraph_service.extract_subgraph(
            entity_id=entity_id,
            max_hops=max_hops,
            max_nodes=5000,
            relationship_types=None,
            min_confidence=min_confidence,
            tenant_id=tenant_id,
        )

        graph = data.graph
        nodes_map = {n["id"]: n for n in data.response.nodes}

        # Group edges by target, split into supporting/refuting
        target_edges: dict[str, dict[str, list[dict[str, Any]]]] = {}

        for src, tgt, edge_data in graph.edges(data=True):
            rel_type = edge_data.get("type", "").upper()
            if rel_type in ("SUPPORTS", "REFUTES"):
                bucket = target_edges.setdefault(tgt, {"supports": [], "refutes": []})
                node_info = nodes_map.get(src, {"id": src, "label": "", "properties": {}})
                if rel_type == "SUPPORTS":
                    bucket["supports"].append(node_info)
                else:
                    bucket["refutes"].append(node_info)

        # Build contradiction pairs where both supports and refutes exist
        pairs: list[ContradictionPair] = []
        for target_id, groups in target_edges.items():
            supporting = groups["supports"]
            refuting = groups["refutes"]
            if supporting and refuting:
                total = len(supporting) + len(refuting)
                net_score = (len(supporting) - len(refuting)) / total
                target_info = nodes_map.get(target_id, {})
                pairs.append(
                    ContradictionPair(
                        target_id=target_id,
                        target_label=target_info.get("label", ""),
                        supporting_nodes=supporting,
                        refuting_nodes=refuting,
                        net_support_score=round(net_score, 4),
                    )
                )

        elapsed = (time.monotonic() - start) * 1000

        return ContradictionResponse(
            entity_id=entity_id,
            contradiction_pairs=pairs,
            total_contradictions=len(pairs),
            computation_time_ms=round(elapsed, 2),
        )
