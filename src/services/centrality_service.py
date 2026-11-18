"""Centrality/authority scoring service — local NetworkX or delegate to Graph Adapter."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import networkx as nx

from clients.graph_adapter import GraphAdapterClient
from log_config import get_logger
from models.topology import AuthorityResponse
from services.cache import CacheService
from services.subgraph_service import SubgraphService

logger = get_logger(__name__)


class CentralityService:
    def __init__(
        self,
        subgraph_service: SubgraphService,
        graph_client: GraphAdapterClient,
        cache: CacheService | None = None,
    ) -> None:
        self.subgraph_service = subgraph_service
        self.graph_client = graph_client
        self._cache = cache

    async def compute_authority(
        self,
        entity_id: str | None = None,
        algorithm: str = "pagerank",
        max_hops: int = 2,
        top_k: int = 20,
        tenant_id: str | None = None,
    ) -> AuthorityResponse:
        start = time.monotonic()

        if entity_id is not None:
            scores, nodes_map = await self._local_centrality(entity_id, algorithm, max_hops, tenant_id)
        else:
            scores, nodes_map = await self._global_centrality(algorithm, tenant_id)

        # Sort and take top-K
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        top_entities: list[dict[str, Any]] = []
        for nid, score in sorted_scores:
            info = nodes_map.get(nid, {"id": nid, "label": "", "properties": {}})
            top_entities.append(
                {
                    "id": nid,
                    "label": info.get("label", ""),
                    "score": round(score, 6),
                    "properties": info.get("properties", {}),
                }
            )

        elapsed = (time.monotonic() - start) * 1000

        return AuthorityResponse(
            algorithm=algorithm,
            scores={k: round(v, 6) for k, v in scores.items()},
            top_entities=top_entities,
            entity_count=len(scores),
            computation_time_ms=round(elapsed, 2),
        )

    async def _local_centrality(
        self,
        entity_id: str,
        algorithm: str,
        max_hops: int,
        tenant_id: str | None,
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
        """Run centrality on locally-extracted subgraph."""
        data = await self.subgraph_service.extract_subgraph(
            entity_id=entity_id,
            max_hops=max_hops,
            max_nodes=5000,
            tenant_id=tenant_id,
        )
        graph = data.graph
        nodes_map = {n["id"]: n for n in data.response.nodes}

        if graph.number_of_nodes() == 0:
            return {}, nodes_map

        # PageRank / eigenvector are CPU-bound (iterative) over subgraphs up to
        # max_nodes=5000 — offload to a worker thread so we never block the event
        # loop for seconds (mirrors IntegrityService._authority_score).
        scores = await asyncio.to_thread(self._run_centrality, graph, algorithm)
        return scores, nodes_map

    async def _global_centrality(
        self,
        algorithm: str,
        tenant_id: str | None,
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
        """Delegate to Graph Adapter for global centrality."""
        # Check cache
        cache_key: str | None = None
        if self._cache is not None:
            cache_key = CacheService.make_centrality_key(tenant_id or "", algorithm)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        result = await self.graph_client.get_centrality(
            algorithm=algorithm,
            tenant_id=tenant_id,
        )
        scores: dict[str, float] = result.get("entity_scores", {})
        # The global centrality endpoint returns scores keyed by entity id only; it does
        # not carry per-entity labels/properties. compute_authority's fallback
        # ({"id": nid, "label": "", "properties": {}}) supplies id-only entries — we do
        # not fabricate labels here.
        nodes_map: dict[str, dict[str, Any]] = {}

        if self._cache is not None and cache_key is not None:
            self._cache.set(cache_key, (scores, nodes_map))

        return scores, nodes_map

    @staticmethod
    def _run_centrality(graph: nx.DiGraph, algorithm: str) -> dict[str, float]:
        """Run NetworkX centrality, falling back to pagerank if eigenvector fails."""
        if algorithm == "eigenvector":
            try:
                return nx.eigenvector_centrality_numpy(graph)
            except nx.NetworkXError, nx.NetworkXException, nx.PowerIterationFailedConvergence:
                logger.warning("Eigenvector centrality failed on disconnected graph, falling back to pagerank")
                return nx.pagerank(graph)
        return nx.pagerank(graph)
