"""
Health endpoint unit tests for scientific_processor.

Validates all health endpoints via TestClient with exact assertions.
"""

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

# Disable auth and enable docs for unit tests
os.environ["ENABLE_AUTH"] = "false"
os.environ["ENABLE_DOCS"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
# Health endpoint iterates only over services with non-empty URLs (the
# production fail-fast validator allows blank URLs in test env). Populate
# stub values so the mocked _check_service is actually invoked per backend.
os.environ.setdefault("SCIENTIFIC_PROCESSOR_GRAPH_ADAPTER_URL", "http://stub-graph-adapter:0")
os.environ.setdefault("SCIENTIFIC_PROCESSOR_PYTORCH_BACKEND_URL", "http://stub-pytorch-backend:0")
os.environ.setdefault("SCIENTIFIC_PROCESSOR_INFERENCE_ROUTER_URL", "http://stub-inference-router:0")
os.environ.setdefault("SCIENTIFIC_PROCESSOR_DISCOVERY_ENGINE_URL", "http://stub-discovery-engine:0")

from fastapi.testclient import TestClient

from app import create_app
from routers.health import health, liveness, readiness, root

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def client():
    """Create a test client for the scientific processor app."""
    app = create_app()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Direct function tests (fast, no network)
# ---------------------------------------------------------------------------


async def test_root_redirects():
    """Root endpoint returns a RedirectResponse to /docs."""
    result = await root()
    assert result.status_code == 307
    assert result.headers["location"] == "/docs"


@patch("routers.health._check_service", new_callable=AsyncMock)
async def test_health_all_healthy(mock_check):
    """Health endpoint reports healthy when all backends are up."""
    mock_check.return_value = {"status": "healthy", "response_time_ms": 5.0}
    result = await health()
    assert result.status == "healthy"
    assert isinstance(result.services, dict)
    assert result.timestamp


@patch("routers.health._check_service", new_callable=AsyncMock)
async def test_health_degraded_when_some_backends_down(mock_check):
    """Health endpoint reports degraded when some backends are unhealthy."""

    async def side_effect(name, url):
        if name == "graph_adapter":
            return {"status": "unhealthy", "response_time_ms": 5000.0, "error": "timeout"}
        return {"status": "healthy", "response_time_ms": 5.0}

    mock_check.side_effect = side_effect
    result = await health()
    assert result.status == "degraded"


@patch("routers.health._check_service", new_callable=AsyncMock)
async def test_health_unhealthy_when_all_backends_down(mock_check):
    """Health endpoint reports unhealthy when all backends are down."""
    mock_check.return_value = {"status": "unhealthy", "response_time_ms": 5000.0, "error": "timeout"}
    result = await health()
    assert result.status == "unhealthy"


@patch("routers.health._check_service", new_callable=AsyncMock)
async def test_readiness_returns_ready(mock_check):
    """Readiness probe returns ready (200) when critical backends are healthy."""
    mock_check.return_value = {"status": "healthy", "response_time_ms": 5.0}
    resp = await readiness()
    assert resp.status_code == 200
    body = json.loads(bytes(resp.body))
    assert body["ready"] is True
    assert body.get("reason") is None


@patch("routers.health._check_service", new_callable=AsyncMock)
async def test_readiness_not_ready_when_critical_down(mock_check):
    """A critical backend being down must produce 503, not a 200 with ready=false.

    This test used to call readiness() and assert only `result.ready is False`,
    which passed happily while the endpoint answered HTTP 200 for that same
    verdict. Since the Deployment wires this path as an httpGet readinessProbe —
    and an httpGet probe reads ONLY the status code, never the body — asserting
    the body alone could never catch the bug. Assert the status code.
    """
    mock_check.return_value = {"status": "unhealthy", "response_time_ms": 5000.0}
    resp = await readiness()
    assert resp.status_code == 503, (
        "readiness answered 200 while graph_adapter was unhealthy; an httpGet "
        "probe cannot see the body, so this pod would stay in rotation"
    )
    body = json.loads(bytes(resp.body))
    assert body["ready"] is False
    assert body["reason"]


async def test_liveness():
    """Liveness probe always returns alive."""
    result = await liveness()
    assert result.alive is True


# ---------------------------------------------------------------------------
# Integration tests via TestClient
# ---------------------------------------------------------------------------


@patch("routers.health._check_service", new_callable=AsyncMock)
def test_health_returns_200(mock_check, client) -> None:
    """GET /health returns 200 with standard HealthStatus body."""
    mock_check.return_value = {"status": "healthy", "response_time_ms": 5.0}
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "services" in body
    assert "timestamp" in body


@patch("routers.health._check_service", new_callable=AsyncMock)
def test_health_ready_returns_200(mock_check, client) -> None:
    """GET /health/ready returns 200 with ReadinessStatus body."""
    mock_check.return_value = {"status": "healthy", "response_time_ms": 5.0}
    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True


@patch("routers.health._check_service", new_callable=AsyncMock)
def test_health_ready_returns_503_when_critical_down(mock_check, client) -> None:
    """End-to-end through the HTTP layer: unhealthy critical dep -> 503."""
    mock_check.return_value = {"status": "unhealthy", "response_time_ms": 5000.0}
    response = client.get("/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert "graph_adapter" in body["reason"]


def test_health_live_returns_200(client) -> None:
    """GET /health/live returns 200 with LivenessStatus body."""
    response = client.get("/health/live")
    assert response.status_code == 200
    body = response.json()
    assert body["alive"] is True


def test_health_content_type(client) -> None:
    """Health endpoints return application/json."""
    response = client.get("/health/live")
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]


def test_openapi_json_available(client) -> None:
    """OpenAPI schema is served at /openapi.json."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "Scientific Processor API"
