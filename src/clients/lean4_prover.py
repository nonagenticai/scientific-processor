"""HTTP client for the lean4-prover service."""

from __future__ import annotations

import asyncio
import math
import random
import time
from typing import Any

import httpx
from kag_common.attribution import attribution_event_hook
from kag_common.auth import outbound_headers

from auth.tenant import license_id_var, license_token_var
from config import get_settings
from log_config import get_logger

logger = get_logger(__name__)


class Lean4LicenceMissingError(RuntimeError):
    """Raised when a lean4-prover call would leave without any licence token."""


#: Statuses that mean "GPU busy, come back later" (lean4-prover#15 answers
#: 503 gpu_busy + Retry-After; 429 is treated identically per the fleet contract).
_BUSY_STATUSES = frozenset({429, 503})
#: Exponential backoff used when the server gives no Retry-After hint.
_BACKOFF_BASE_S = 2.0
_BACKOFF_MAX_S = 60.0
#: Retry-After values the contract allows (seconds).
_RETRY_AFTER_MIN_S = 1
_RETRY_AFTER_MAX_S = 120

#: Indirection points so tests can drive a fake clock without patching asyncio/time globally.
_sleep = asyncio.sleep
_monotonic = time.monotonic


def _jitter() -> float:
    return random.uniform(0.0, 1.0)  # noqa: S311 - backoff jitter, not crypto


class Lean4ProverBusyError(RuntimeError):
    """lean4-prover stayed busy (503/429) past this call's retry deadline.

    TRANSIENT, never terminal: the router maps it to HTTP 503 with
    ``Retry-After: <retry_after>`` and body ``{"error": "gpu_busy", ...}``.
    It deliberately does NOT subclass ``httpx.HTTPError``, so the existing
    ``except httpx.HTTPError`` fallbacks cannot swallow it.
    """

    def __init__(self, retry_after: int, detail: str = "lean4-prover GPU busy") -> None:
        super().__init__(f"{detail} (retry after {retry_after}s)")
        self.retry_after = retry_after


def _clamp_retry_after(seconds: float) -> int:
    return max(_RETRY_AFTER_MIN_S, min(_RETRY_AFTER_MAX_S, math.ceil(seconds)))


def _server_retry_after(response: httpx.Response) -> float | None:
    """Server hint: ``Retry-After`` header (seconds) first, then body ``retry_after``."""
    header = response.headers.get("Retry-After")
    if header is not None:
        try:
            value = float(header.strip())
        except ValueError:
            value = -1.0
        if value >= 0:
            return value
    try:
        body = response.json()
    except ValueError:
        return None
    if isinstance(body, dict):
        raw = body.get("retry_after")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw >= 0:
            return float(raw)
    return None


class Lean4ProverClient:
    """Async HTTP client for lean4-prover:8812.

    Every call to a GPU backend must carry a licence (owner ruling). lean4-prover
    introspects ``X-Tenant-Id: Bearer <token>`` and 401s anything else, so this
    client ALWAYS sends a Bearer licence token: the end tenant's own token when
    the inbound request presented one (``license_token_var``), else this
    service's own service licence (``SCIENTIFIC_PROCESSOR_DOWNSTREAM_LICENSE_TOKEN``).
    With neither it raises before touching the network — never a bare tenant.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        service_license_token: str | None = None,
        max_concurrency: int = 4,
        busy_deadline_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.service_license_token = (service_license_token or "").strip() or None
        self.max_concurrency = max(1, max_concurrency)
        self.busy_deadline_s = max(0.0, busy_deadline_s)
        # Bounds this pod's in-flight lean4 calls (the server's prove lane is
        # 2 active + 8 waiting); held only for the HTTP call, not the backoff.
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
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
        """Bearer licence + licence-id, via the fleet's shared outbound-header builder.

        The token is the end tenant's own (it was introspected on the way in, so
        lean4-prover attributes the work to that tenant) or, failing that, this
        service's service licence. ``service_token=`` is the ``outbound_headers``
        rung that ALWAYS Bearer-prefixes; ``tenant_token=`` only prefixes
        JWT-shaped values and would send an opaque licence bare, i.e. a 401.

        ``tenant_id`` is the caller's data scope; it is not sent, because the
        ``X-Tenant-Id`` slot carries the credential and lean4-prover derives the
        tenant from introspection, never from a header.
        """
        del tenant_id
        token = license_token_var.get() or self.service_license_token
        if not token:
            raise Lean4LicenceMissingError(
                "lean4-prover call has no licence token: the request carried none and "
                "SCIENTIFIC_PROCESSOR_DOWNSTREAM_LICENSE_TOKEN is empty"
            )
        return outbound_headers(service_token=token, license_id=license_id_var.get())

    async def _post(self, path: str, payload: dict[str, Any], tenant_id: str | None) -> dict:
        """POST with GPU back-pressure handling.

        503/429 are retried, honouring the server's hint (``Retry-After`` header,
        then body ``retry_after``, else exponential backoff 2->60 s with jitter),
        until ``busy_deadline_s`` would be overrun; then ``Lean4ProverBusyError``.
        The semaphore wait counts against the same deadline.
        """
        client = self._ensure_client()
        headers = self._headers(tenant_id)
        deadline = _monotonic() + self.busy_deadline_s
        attempt = 0
        while True:
            remaining = deadline - _monotonic()
            try:
                async with asyncio.timeout(max(remaining, 0.0)):
                    await self._semaphore.acquire()
            except TimeoutError:
                raise Lean4ProverBusyError(
                    _clamp_retry_after(_BACKOFF_BASE_S), "lean4-prover concurrency limit saturated"
                ) from None
            try:
                response = await client.post(path, json=payload, headers=headers)
            finally:
                self._semaphore.release()

            if response.status_code not in _BUSY_STATUSES:
                response.raise_for_status()
                return response.json()

            hint = _server_retry_after(response)
            backoff = min(_BACKOFF_MAX_S, _BACKOFF_BASE_S * (2**attempt))
            wait = (hint if hint is not None else backoff) + _jitter()
            attempt += 1
            if _monotonic() + wait > deadline:
                retry_after = _clamp_retry_after(hint if hint is not None else backoff)
                logger.warning(
                    "lean4-prover busy past retry deadline",
                    path=path,
                    status=response.status_code,
                    attempts=attempt,
                    retry_after=retry_after,
                )
                raise Lean4ProverBusyError(retry_after)
            logger.info(
                "lean4-prover busy, backing off",
                path=path,
                status=response.status_code,
                attempt=attempt,
                wait_s=round(wait, 2),
            )
            await _sleep(wait)

    async def check(self, lean_code: str, timeout: int = 30, tenant_id: str | None = None) -> dict:
        """Type-check Lean 4 code. Maps to POST /api/check."""
        return await self._post("/api/check", {"source": lean_code, "timeout": timeout}, tenant_id)

    async def elaborate(self, lean_code: str, timeout: int = 30, tenant_id: str | None = None) -> dict:
        """Elaborate Lean 4 code (type-check + return type info). Maps to POST /api/elaborate."""
        return await self._post("/api/elaborate", {"term": lean_code, "timeout": timeout}, tenant_id)

    async def prove(
        self,
        claim_text: str,
        strategy: str = "auto",
        timeout: int = 60,
        tenant_id: str | None = None,
    ) -> dict:
        """End-to-end claim → Lean 4 proof. Maps to POST /api/prove."""
        return await self._post(
            "/api/prove", {"theorem": claim_text, "tactics": [strategy], "timeout": timeout}, tenant_id
        )

    async def health_check(self) -> dict:
        """Check lean4-prover health."""
        try:
            client = self._ensure_client()
            response = await client.get("/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError, OSError:
            return {"status": "unavailable", "service": "lean4-prover"}
