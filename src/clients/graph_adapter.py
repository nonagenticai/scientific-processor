"""
HTTP client for the Graph Adapter service.

Provides access to Neo4j knowledge graph operations via the graph-adapter API.
"""

from __future__ import annotations

from typing import Any

import httpx
from kag_common.attribution import attribution_event_hook
from kag_common.auth import outbound_headers

from auth.tenant import license_id_var
from config import get_settings
from log_config import get_logger

logger = get_logger(__name__)


class GraphAdapterError(Exception):
    """Base error for graph adapter operations."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class GraphServiceUnavailableError(GraphAdapterError):
    """Graph adapter service is unreachable."""

    def __init__(self, message: str = "Graph adapter service unavailable") -> None:
        super().__init__(message, status_code=503)


class GraphAdapterClient:
    """Async HTTP client for graph-adapter service."""

    def __init__(self, base_url: str, timeout: float = 60.0, service_token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # WS-4 on-behalf-of service identity (see config.graph_service_token).
        self.service_token = service_token or None
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            s = get_settings()
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                # Forward the caller's X-Project-Id / X-Job-Id (kag_common 0.33): read per
                # request from the bound context, so this cached client never leaks one
                # request's project onto another. Unbound (background work) adds nothing.
                event_hooks={"request": [attribution_event_hook]},
                transport=httpx.AsyncHTTPTransport(retries=s.http_transport_retries),
                timeout=httpx.Timeout(
                    connect=s.http_connect_timeout_s,
                    read=self.timeout,
                    write=s.http_write_timeout_s,
                    pool=s.http_pool_timeout_s,
                ),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _headers(self, tenant_id: str | None) -> dict[str, str]:
        """Outbound identity headers, via the fleet's shared builder.

        ``authorization_service_token`` is the WS-4 on-behalf-of slot: it proves
        OUR own service identity separately so the plain ``X-Tenant-Id`` (the
        data tenant) is trusted by graph-adapter, without touching the tenant
        ladder. Same two headers as the hand-rolled version it replaces.

        ``license_id`` is the caller's *resolved* licence, forwarded purely as
        cost-attribution data — never as a credential; graph-adapter still
        authenticates on the tenant/Authorization pair above. Omitted entirely
        when the caller presented no licence, which is what
        ``billing_license_unattributed_total`` is there to count. It is never
        replaced by ``self.service_token``: that would attribute a customer's
        graph work to scientific-processor's own service licence.
        """
        return outbound_headers(
            tenant_id=tenant_id,
            authorization_service_token=self.service_token,
            license_id=license_id_var.get(),
        )

    def _handle_response(self, response: httpx.Response, operation: str) -> dict:
        """Validate response and return JSON, wrapping httpx errors."""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            detail = e.response.text[:200] if e.response.text else str(e)
            raise GraphAdapterError(
                f"Graph adapter {operation} failed (HTTP {status}): {detail}",
                status_code=status,
            ) from e
        return response.json()

    async def _request(
        self,
        method: str,
        path: str,
        operation: str,
        *,
        json: dict | None = None,
        tenant_id: str | None = None,
    ) -> dict:
        """Execute an HTTP request with standardized error handling."""
        client = self._ensure_client()
        try:
            if method == "GET":
                response = await client.get(path, headers=self._headers(tenant_id))
            elif method == "PUT":
                response = await client.put(path, json=json, headers=self._headers(tenant_id))
            else:
                response = await client.post(path, json=json, headers=self._headers(tenant_id))
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException) as e:
            raise GraphServiceUnavailableError(f"Graph adapter unreachable during {operation}: {e}") from e
        return self._handle_response(response, operation)

    async def execute_cypher(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> dict:
        """Execute a Cypher query via the Graph Adapter /query endpoint."""
        return await self._request(
            "POST",
            "/query",
            "execute_cypher",
            json={
                "query": query,
                "parameters": parameters or {},
                "query_type": "cypher",
            },
            tenant_id=tenant_id,
        )

    async def get_node(
        self,
        node_id: str,
        tenant_id: str | None = None,
    ) -> dict:
        """Fetch a single node by ID."""
        return await self._request(
            "GET",
            f"/nodes/{node_id}",
            "get_node",
            tenant_id=tenant_id,
        )

    async def get_centrality(
        self,
        algorithm: str = "pagerank",
        parameters: dict[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> dict:
        """Compute centrality scores for nodes."""
        return await self._request(
            "POST",
            "/analytics/centrality",
            "get_centrality",
            json={"algorithm": algorithm, "parameters": parameters},
            tenant_id=tenant_id,
        )

    async def get_communities(
        self,
        algorithm: str = "louvain",
        min_community_size: int = 3,
        parameters: dict[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> dict:
        """Detect communities in the graph."""
        return await self._request(
            "POST",
            "/analytics/community",
            "get_communities",
            json={
                "algorithm": algorithm,
                "min_community_size": min_community_size,
                "parameters": parameters,
            },
            tenant_id=tenant_id,
        )

    async def get_paths(
        self,
        source_entity_id: str,
        target_entity_id: str,
        path_type: str = "shortest",
        max_paths: int = 5,
        min_confidence: float = 0.0,
        tenant_id: str | None = None,
    ) -> dict:
        """Find paths between two nodes."""
        return await self._request(
            "POST",
            "/analytics/path",
            "get_paths",
            json={
                "source_entity_id": source_entity_id,
                "target_entity_id": target_entity_id,
                "path_type": path_type,
                "max_paths": max_paths,
                "min_confidence": min_confidence,
            },
            tenant_id=tenant_id,
        )

    async def get_subgraph(
        self,
        entity_id: str,
        max_hops: int = 2,
        max_nodes: int = 500,
        relationship_types: list[str] | None = None,
        tenant_id: str | None = None,
    ) -> dict:
        """Extract a tenant-scoped k-hop subgraph around a single entity.

        Structured replacement for the raw-Cypher path traversal (``/query`` is
        gated off in prod). Returns the serialized-path envelope
        ``{"results": [{"path": {"nodes": [...], "relationships": [...]}}], ...}``.
        """
        return await self._request(
            "POST",
            "/nodes/subgraph",
            "get_subgraph",
            json={
                "entity_id": entity_id,
                "max_hops": max_hops,
                "max_nodes": max_nodes,
                "relationship_types": relationship_types or [],
            },
            tenant_id=tenant_id,
        )

    async def fulltext_search(
        self,
        query: str,
        min_score: float = 0.0,
        limit: int = 10,
        tenant_id: str | None = None,
    ) -> dict:
        """Tenant-scoped fulltext search over the graph (nodeFulltextIndex).

        Structured replacement for the raw-Cypher CoVe source search. Returns
        ``{"results": [{"id","name","labels","confidence","score"}], ...}``.
        """
        return await self._request(
            "POST",
            "/nodes/fulltext-search",
            "fulltext_search",
            json={"query": query, "min_score": min_score, "limit": limit},
            tenant_id=tenant_id,
        )

    async def find_nodes_missing_property(
        self,
        property_name: str,
        limit: int = 50,
        tenant_id: str | None = None,
    ) -> dict:
        """Tenant-scoped scan for node ids whose ``property_name`` is absent/NULL.

        Structured replacement for the raw-Cypher unscored-entity scan. Returns
        ``{"results": [{"id": ...}], "result_count": N}``.
        """
        return await self._request(
            "POST",
            "/nodes/missing-property",
            "find_nodes_missing_property",
            json={"property": property_name, "limit": limit},
            tenant_id=tenant_id,
        )

    async def update_node(
        self,
        node_id: str,
        properties: dict[str, Any],
        tenant_id: str | None = None,
    ) -> dict:
        """Merge ``properties`` onto a node by its business id (tenant-scoped).

        Reuses the existing PUT /nodes/{id} (SET n += $properties). Used for the
        integrity score write-back instead of raw Cypher.
        """
        return await self._request(
            "PUT",
            f"/nodes/{node_id}",
            "update_node",
            json={"properties": properties},
            tenant_id=tenant_id,
        )

    async def health_check(self) -> dict:
        """Check graph-adapter health."""
        return await self._request("GET", "/health", "health_check")
