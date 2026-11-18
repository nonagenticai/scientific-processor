"""
HTTP client for the Discovery Engine service.

Stubbed — returns empty results until discovery-engine is implemented.
"""

from __future__ import annotations

import httpx
from kag_common.attribution import attribution_event_hook

from config import get_settings
from log_config import get_logger

logger = get_logger(__name__)


class DiscoveryEngineClient:
    """Async HTTP client for discovery-engine service (stubbed)."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
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

    async def search(
        self,
        query: str,
        limit: int = 10,
        tenant_id: str | None = None,
    ) -> dict:
        """Search for related scientific entities. Stubbed — returns empty results."""
        logger.debug("discovery_engine.search stubbed", query=query)
        return {"results": [], "total": 0}

    async def get_related_claims(
        self,
        claim_id: str,
        tenant_id: str | None = None,
    ) -> dict:
        """Get claims related to a given claim. Stubbed — returns empty results."""
        logger.debug("discovery_engine.get_related_claims stubbed", claim_id=claim_id)
        return {"claims": [], "total": 0}

    async def health_check(self) -> dict:
        """Check discovery-engine health."""
        try:
            client = self._ensure_client()
            response = await client.get("/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError, OSError:
            return {"status": "unavailable", "service": "discovery-engine"}
