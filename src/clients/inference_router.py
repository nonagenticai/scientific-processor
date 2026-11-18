"""
HTTP client for the Inference Router service.

Provides access to LLM chat completion for verification and formal reasoning.

LICENCE ATTRIBUTION — WHY THIS HOP MATTERS MOST
-----------------------------------------------
This is the METERED LLM hop: every call here produces a cost event downstream.
It used to build its headers by hand as ``X-Tenant-Id`` + ``X-Schema-Id`` only,
so inference-router received a resolved tenant and no licence and had nothing to
attribute the spend to — ``billing_service`` counts that exact shape in
``billing_license_unattributed_total``. The licence now rides
``auth.tenant.license_id_var``, bound by ``AuthMiddleware`` in lockstep with the
tenant.

Headers are built by ``kag_common.auth.outbound_headers``, the fleet's single
definition of the inter-service header contract (this repo already depends on
``kag-common``; ``auth/vouch.py`` uses the same package). It emits
``X-License-Id`` only for a truthy value, so a caller that presented no licence
sends no header at all — the honest NULL billing actually measures — and it never
substitutes the tenant or a service token for a missing licence.
"""

from __future__ import annotations

import httpx
from kag_common.attribution import attribution_event_hook
from kag_common.auth import outbound_headers

from auth.tenant import license_id_var
from config import get_settings
from log_config import get_logger

logger = get_logger(__name__)


class InferenceRouterClient:
    """Async HTTP client for inference-router service."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
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
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str = "default",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        tenant_id: str | None = None,
        schema_id: str | None = None,
    ) -> dict:
        """Send a chat completion request to the inference router.

        Args:
            schema_id: Optional routing schema ID. Passed as X-Schema-Id header
                so inference-router looks up the corresponding routing rule
                (provider chain, json_schema, system_prompt).
        """
        client = self._ensure_client()
        # `X-Schema-Id` is not part of the shared contract, so it rides `extra`,
        # which `outbound_headers` merges LAST — a header a caller has already
        # set is never overwritten by the builder.
        extra = {"X-Schema-Id": schema_id} if schema_id else None
        headers = outbound_headers(tenant_id=tenant_id, license_id=license_id_var.get(), extra=extra)
        response = await client.post(
            "/api/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> dict:
        """Check inference-router health."""
        client = self._ensure_client()
        response = await client.get("/health")
        response.raise_for_status()
        return response.json()
