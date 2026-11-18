"""Transport-hardening tests for upstream HTTP clients.

The 2026-05-16 trust.pdf ingest surfaced ``httpx.RemoteProtocolError:
Server disconnected`` against inference-router AND graph-adapter — the
root cause was bare ``httpx.AsyncClient`` construction with no transport
retries, so a single transient ambient-mesh handshake blip became a
terminal "Server disconnected" failure. Mirrors the worker_management F3
fix (PR #46).

These tests assert that every upstream client (inference-router,
graph-adapter, pytorch-backend, discovery-engine, lean4-prover) carries
the configured ``AsyncHTTPTransport(retries=N)`` and the explicit
``httpx.Timeout`` derived from settings.
"""

from __future__ import annotations

import httpx
import pytest

from clients.discovery_engine import DiscoveryEngineClient
from clients.graph_adapter import GraphAdapterClient
from clients.inference_router import InferenceRouterClient
from clients.lean4_prover import Lean4ProverClient
from clients.pytorch_backend import PyTorchBackendClient
from config import get_settings

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("factory", "read_timeout"),
    [
        (lambda: InferenceRouterClient(base_url="http://ir.test:8000", timeout=120.0), 120.0),
        (lambda: GraphAdapterClient(base_url="http://ga.test:8000", timeout=60.0), 60.0),
        (lambda: PyTorchBackendClient(base_url="http://pt.test:8000", timeout=120.0), 120.0),
        (lambda: DiscoveryEngineClient(base_url="http://de.test:8000", timeout=60.0), 60.0),
        (lambda: Lean4ProverClient(base_url="http://l4.test:8000", timeout=120.0), 120.0),
    ],
    ids=["inference_router", "graph_adapter", "pytorch_backend", "discovery_engine", "lean4_prover"],
)
def test_ensure_client_carries_transport_retries_and_timeouts(factory, read_timeout) -> None:
    """Every upstream client must build its AsyncClient with the
    settings-driven AsyncHTTPTransport(retries=N) and explicit Timeout
    (connect/read/write/pool). Bare clients caused the trust.pdf
    "Server disconnected" cascade — never regress to that.
    """
    settings = get_settings()
    client_obj = factory()

    underlying = client_obj._ensure_client()

    # Transport-level retries must be configured (the load-bearing change).
    transport = underlying._transport
    assert isinstance(transport, httpx.AsyncHTTPTransport)
    # The retries kwarg is stored on the AsyncHTTPTransport's pool config.
    # httpx exposes it on the internal _pool, but a clean assertion is to
    # rebuild the transport with the same retries and compare; here we
    # rely on the documented httpx contract: AsyncHTTPTransport(retries=N)
    # constructs a connection pool with retries=N. The pool object stores
    # retries on `.retries` since httpx 0.27.
    assert getattr(transport._pool, "_retries", None) == settings.http_transport_retries

    # Explicit Timeout must be set from settings (not the bare default).
    timeout = underlying.timeout
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == settings.http_connect_timeout_s
    assert timeout.read == read_timeout
    assert timeout.write == settings.http_write_timeout_s
    assert timeout.pool == settings.http_pool_timeout_s


def test_settings_defaults_are_sane() -> None:
    """Sanity-bound the settings so a future env-only "tuning" can't
    accidentally regress us to the pre-fix shape.
    """
    s = get_settings()
    assert s.http_transport_retries >= 1, "must retry at least once"
    assert s.http_connect_timeout_s >= 5.0, "connect timeout too short for cross-ns TLS handshake"
    assert s.http_pool_timeout_s >= 1.0, "pool timeout too aggressive"
    assert s.http_write_timeout_s >= 5.0, "write timeout too aggressive"
