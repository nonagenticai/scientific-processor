"""GPU back-pressure: lean4-prover 503 gpu_busy + Retry-After (lean4-prover#15).

A stub lean4-prover (``httpx.MockTransport``) answers busy, and these tests pin:
the client honours Retry-After (header, then body, then 2->60 s backoff) within a
bounded deadline; exhaustion raises ``Lean4ProverBusyError``; the formal router
turns that into HTTP 503 + Retry-After + ``{"error": "gpu_busy"}`` instead of 500;
and the per-pod semaphore caps in-flight calls.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

os.environ["ENABLE_AUTH"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

import clients.lean4_prover as lean4_mod
from app import create_app
from clients.lean4_prover import Lean4ProverBusyError, Lean4ProverClient
from config import Settings
from dependencies import get_formal_service
from services.formal_service import FormalService

pytestmark = pytest.mark.unit

BASE_URL = "http://lean4-prover:8812"
SVC_TOKEN = "svc-licence-token-sp"  # noqa: S105 - test fixture, not a credential
OK_BODY = {"success": True, "errors": [], "messages": ["OK"]}


def _busy(retry_after: int | None = None, *, header: bool = True, status: int = 503) -> httpx.Response:
    body: dict[str, Any] = {"error": "gpu_busy", "lane": "prove"}
    headers: dict[str, str] = {}
    if retry_after is not None:
        body["retry_after"] = retry_after
        if header:
            headers["Retry-After"] = str(retry_after)
    return httpx.Response(status, json=body, headers=headers)


class FakeClock:
    """Drives the client's clock and sleep so retries run instantly."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    c = FakeClock()
    monkeypatch.setattr(lean4_mod, "_monotonic", c.monotonic)
    monkeypatch.setattr(lean4_mod, "_sleep", c.sleep)
    monkeypatch.setattr(lean4_mod, "_jitter", lambda: 0.0)
    return c


def _stub_client(
    responses: list[httpx.Response] | Callable[[httpx.Request], Any],
    **kwargs: Any,
) -> tuple[Lean4ProverClient, list[httpx.Request]]:
    """A Lean4ProverClient talking to a stub lean4-prover; returns (client, seen requests)."""
    seen: list[httpx.Request] = []
    if callable(responses):
        handler = responses
    else:
        queue = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return queue.pop(0) if len(queue) > 1 else queue[0]

    client = Lean4ProverClient(BASE_URL, service_license_token=SVC_TOKEN, **kwargs)
    client._client = httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    return client, seen


class TestBusyRetry:
    async def test_503_with_retry_after_header_then_200(self, clock: FakeClock) -> None:
        client, seen = _stub_client([_busy(7), httpx.Response(200, json=OK_BODY)])
        result = await client.check("theorem t : True := trivial")
        assert result == OK_BODY
        assert len(seen) == 2
        assert clock.sleeps == [7.0]

    async def test_header_wins_over_body(self, clock: FakeClock) -> None:
        resp = httpx.Response(503, json={"error": "gpu_busy", "retry_after": 30}, headers={"Retry-After": "3"})
        client, _ = _stub_client([resp, httpx.Response(200, json=OK_BODY)])
        await client.prove("1 + 1 = 2")
        assert clock.sleeps == [3.0]

    async def test_body_retry_after_when_no_header(self, clock: FakeClock) -> None:
        client, _ = _stub_client([_busy(9, header=False), httpx.Response(200, json=OK_BODY)])
        await client.elaborate("theorem t : True := trivial")
        assert clock.sleeps == [9.0]

    async def test_429_treated_like_503(self, clock: FakeClock) -> None:
        client, seen = _stub_client([_busy(4, status=429), httpx.Response(200, json=OK_BODY)])
        assert await client.check("x") == OK_BODY
        assert len(seen) == 2
        assert clock.sleeps == [4.0]

    async def test_no_hint_exponential_backoff(self, clock: FakeClock) -> None:
        bare = httpx.Response(503, text="no healthy upstream")
        client, seen = _stub_client([bare, bare, bare, httpx.Response(200, json=OK_BODY)])
        assert await client.check("x") == OK_BODY
        assert clock.sleeps == [2.0, 4.0, 8.0]
        assert len(seen) == 4

    async def test_backoff_capped_at_60(self, clock: FakeClock) -> None:
        bare = httpx.Response(503)
        client, _ = _stub_client([bare] * 7 + [httpx.Response(200, json=OK_BODY)], busy_deadline_s=1000.0)
        await client.check("x")
        assert clock.sleeps == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]

    async def test_always_busy_raises_busy_error_within_deadline(self, clock: FakeClock) -> None:
        client, seen = _stub_client([_busy(5)], busy_deadline_s=12.0)
        start = clock.now
        with pytest.raises(Lean4ProverBusyError) as exc_info:
            await client.prove("1 + 1 = 2")
        assert exc_info.value.retry_after == 5
        # 3 attempts at t=0,5,10; the 4th wait would land at 15 > 12.
        assert len(seen) == 3
        assert clock.now - start <= 12.0
        assert not isinstance(exc_info.value, httpx.HTTPError)

    async def test_retry_after_clamped_to_contract_range(self, clock: FakeClock) -> None:
        client, _ = _stub_client([_busy(600)], busy_deadline_s=10.0)
        with pytest.raises(Lean4ProverBusyError) as exc_info:
            await client.check("x")
        assert exc_info.value.retry_after == 120

    async def test_non_busy_error_is_not_retried(self, clock: FakeClock) -> None:
        client, seen = _stub_client([httpx.Response(500, json={"detail": "boom"})])
        with pytest.raises(httpx.HTTPStatusError):
            await client.check("x")
        assert len(seen) == 1
        assert clock.sleeps == []


class TestSemaphore:
    async def test_in_flight_calls_bounded_by_max_concurrency(self) -> None:
        in_flight = 0
        peak = 0
        release = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await release.wait()
            in_flight -= 1
            return httpx.Response(200, json=OK_BODY)

        client, _ = _stub_client(handler, max_concurrency=3)
        tasks = [asyncio.create_task(client.check(f"t{i}")) for i in range(10)]
        for _ in range(20):
            await asyncio.sleep(0)
        assert in_flight == 3
        release.set()
        results = await asyncio.gather(*tasks)
        assert len(results) == 10
        assert peak == 3

    async def test_semaphore_wait_counts_against_deadline(self) -> None:
        release = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            await release.wait()
            return httpx.Response(200, json=OK_BODY)

        client, _ = _stub_client(handler, max_concurrency=1, busy_deadline_s=0.05)
        holder = asyncio.create_task(client.check("holder"))
        await asyncio.sleep(0)
        with pytest.raises(Lean4ProverBusyError):
            await client.check("starved")
        release.set()
        assert await holder == OK_BODY

    def test_default_concurrency_is_4(self) -> None:
        c = Lean4ProverClient(BASE_URL)
        assert c.max_concurrency == 4
        assert c.busy_deadline_s == 120.0

    def test_settings_feed_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCIENTIFIC_PROCESSOR_LEAN4_PROVER_MAX_CONCURRENCY", "2")
        monkeypatch.setenv("SCIENTIFIC_PROCESSOR_LEAN4_PROVER_BUSY_DEADLINE_S", "45")
        s = Settings()
        assert s.lean4_prover_max_concurrency == 2
        assert s.lean4_prover_busy_deadline_s == 45.0


class TestRouterMapsBusyTo503:
    """End to end: real FormalService + client against an always-busy stub."""

    @pytest.fixture
    def http(self, clock: FakeClock):
        lean4, _ = _stub_client([_busy(6)], busy_deadline_s=10.0)
        inference = AsyncMock()
        inference.chat_completion.return_value = {
            "choices": [{"message": {"content": "theorem t : 1 + 1 = 2 := by norm_num"}}]
        }
        service = FormalService(
            inference_client=inference,
            z3_verifier=AsyncMock(),
            sympy_verifier=AsyncMock(),
            minio_client=None,
            lean4_client=lean4,
        )
        app = create_app(Settings())
        app.dependency_overrides[get_formal_service] = lambda: service
        yield TestClient(app), inference
        app.dependency_overrides.clear()

    def test_verify_lean4_busy_is_503_with_retry_after(self, http) -> None:
        client, _ = http
        resp = client.post(
            "/api/formal/verify", json={"formal_expression": "theorem t : True := trivial", "language": "lean4"}
        )
        assert resp.status_code == 503
        assert resp.headers["Retry-After"] == "6"
        assert resp.json() == {"error": "gpu_busy", "retry_after": 6}

    def test_translate_lean4_busy_is_503_not_swallowed(self, http) -> None:
        client, inference = http
        resp = client.post("/api/formal/translate", json={"claim_text": "1 + 1 = 2", "target_language": "lean4"})
        assert resp.status_code == 503
        assert resp.headers["Retry-After"] == "6"
        assert resp.json() == {"error": "gpu_busy", "retry_after": 6}
        # Busy is not an elaboration failure: no second LLM round-trip is spent on it.
        assert inference.chat_completion.await_count == 1
