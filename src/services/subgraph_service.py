"""Subgraph extraction service — k-hop neighborhoods via Cypher → NetworkX."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from clients.graph_adapter import GraphAdapterClient
from config import Settings
from log_config import get_logger
from models.topology import SubgraphResponse
from services.cache import CacheService

logger = get_logger(__name__)

# Whitelist for Neo4j relationship type names — prevents Cypher injection.
_VALID_REL_TYPE = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass
class SubgraphData:
    """Internal representation holding both response data and NetworkX graph."""

    response: SubgraphResponse
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)


class SubgraphService:
    def __init__(self, graph_client: GraphAdapterClient, settings: Settings, cache: CacheService | None = None) -> None:
        self.graph_client = graph_client
        self.max_hops_limit = settings.subgraph_max_hops
        self.max_nodes_limit = settings.subgraph_max_nodes
        self._cache = cache

    async def extract_subgraph(
        self,
        entity_id: str,
        max_hops: int = 2,
        max_nodes: int = 500,
        relationship_types: list[str] | None = None,
        min_confidence: float = 0.0,
        tenant_id: str | None = None,
    ) -> SubgraphData:
        start = time.monotonic()

        max_hops = min(max_hops, self.max_hops_limit)
        max_nodes = min(max_nodes, self.max_nodes_limit)

        # Check cache
        cache_key: str | None = None
        if self._cache is not None:
            cache_key = CacheService.make_subgraph_key(tenant_id or "", entity_id, max_hops)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Filter relationship types to the safe whitelist before sending. The
        # structured /nodes/subgraph endpoint escapes them server-side too, but
        # we keep the silent drop so a stray type never turns into a 400.
        safe_types: list[str] | None = None
        if relationship_types:
            filtered = [t for t in relationship_types if _VALID_REL_TYPE.match(t)]
            safe_types = filtered or None

        # Tenant-scoped k-hop walk via the structured endpoint (raw Cypher is
        # gated off in prod). Returns the same serialized-path envelope the raw
        # /query produced, so the path-row parser below is unchanged.
        result = await self.graph_client.get_subgraph(
            entity_id=entity_id,
            max_hops=max_hops,
            max_nodes=max_nodes,
            relationship_types=safe_types,
            tenant_id=tenant_id,
        )

        nodes_map: dict[str, dict[str, Any]] = {}
        edges_list: list[dict[str, Any]] = []
        edge_keys: set[tuple[str, str, str]] = set()

        for row in result.get("results", []):
            self._parse_result_row(row, nodes_map, edges_list, edge_keys)

        # Filter edges by min_confidence
        if min_confidence > 0:
            edges_list = [e for e in edges_list if e.get("properties", {}).get("confidence", 1.0) >= min_confidence]

        # Cap nodes by removing lowest-degree if needed
        if len(nodes_map) > max_nodes:
            nodes_map, edges_list = self._cap_nodes(nodes_map, edges_list, max_nodes)

        # Build NetworkX DiGraph
        graph = nx.DiGraph()
        for nid, ndata in nodes_map.items():
            graph.add_node(nid, **ndata)
        for edge in edges_list:
            graph.add_edge(
                edge["source_id"],
                edge["target_id"],
                type=edge.get("type", ""),
                **edge.get("properties", {}),
            )

        nodes_out = list(nodes_map.values())
        elapsed = (time.monotonic() - start) * 1000

        response = SubgraphResponse(
            entity_id=entity_id,
            nodes=nodes_out,
            edges=edges_list,
            node_count=len(nodes_out),
            edge_count=len(edges_list),
            max_hops_used=max_hops,
            computation_time_ms=round(elapsed, 2),
        )

        result = SubgraphData(response=response, graph=graph)

        if self._cache is not None and cache_key is not None:
            self._cache.set(cache_key, result)

        return result

    def _parse_result_row(
        self,
        row: dict[str, Any],
        nodes_map: dict[str, dict[str, Any]],
        edges_list: list[dict[str, Any]],
        edge_keys: set[tuple[str, str, str]],
    ) -> None:
        """Parse a single Cypher result row containing path data."""
        # Result may be a path dict or wrapped in a key
        path = row.get("path", row)

        for node in path.get("nodes", []):
            nid = str(node.get("id", node.get("elementId", "")))
            if nid and nid not in nodes_map:
                labels = node.get("labels", [])
                nodes_map[nid] = {
                    "id": nid,
                    "label": labels[0] if labels else "",
                    "labels": labels,
                    "properties": node.get("properties", {}),
                }

        for rel in path.get("relationships", []):
            src = str(rel.get("start_node_id", rel.get("startNodeElementId", "")))
            tgt = str(rel.get("end_node_id", rel.get("endNodeElementId", "")))
            rtype = rel.get("type", "")
            key = (src, tgt, rtype)
            if key not in edge_keys:
                edge_keys.add(key)
                edges_list.append(
                    {
                        "source_id": src,
                        "target_id": tgt,
                        "type": rtype,
                        "properties": rel.get("properties", {}),
                    }
                )

    @staticmethod
    def _cap_nodes(
        nodes_map: dict[str, dict[str, Any]],
        edges_list: list[dict[str, Any]],
        max_nodes: int,
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        """Keep only the top max_nodes nodes by degree, prune orphan edges."""
        degree: dict[str, int] = {nid: 0 for nid in nodes_map}
        for edge in edges_list:
            src, tgt = edge["source_id"], edge["target_id"]
            if src in degree:
                degree[src] += 1
            if tgt in degree:
                degree[tgt] += 1

        keep = set(sorted(degree, key=lambda k: degree[k], reverse=True)[:max_nodes])
        nodes_map = {k: v for k, v in nodes_map.items() if k in keep}
        edges_list = [e for e in edges_list if e["source_id"] in keep and e["target_id"] in keep]
        return nodes_map, edges_list
