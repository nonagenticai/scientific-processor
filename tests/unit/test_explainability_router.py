"""Tests for explainability router endpoints -- /api/explain/subgraphx, /api/explain/jobs/{job_id}."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

os.environ["ENABLE_AUTH"] = "false"
os.environ["ENABLE_DOCS"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from app import create_app
from config import Settings
from dependencies import get_job_manager, get_subgraphx_service
from services.job_manager import JobManager

pytestmark = pytest.mark.unit

# tenant ids are native postgres uuids since migration 005; the boundary
# (auth.dependencies._require_uuid_tenant) 400s anything that is not one, so the
# slugs these tests used to send ("t1", "my-tenant", "attacker") are no longer
# admissible input and would only ever assert the rejection path.
TENANT_ID = "0195f48b-d000-7000-8000-0000000000a1"
OTHER_TENANT_ID = "0195f48b-d000-7000-8000-0000000000a2"


@pytest.fixture
def job_manager():
    return JobManager()


@pytest.fixture
def subgraphx_mock():
    mock = AsyncMock()
    mock.explain = AsyncMock(return_value="test-job-id-1234")
    return mock


@pytest.fixture
def app(subgraphx_mock, job_manager, monkeypatch):
    # Auth ON so the middleware→router tenant flow runs: routers now read the
    # authoritative request.state.tenant_id (set by AuthMiddleware from the
    # trusted-net raw X-Tenant-Id), never the raw header directly. The ASGI
    # test peer (127.0.0.1) is a trusted network. monkeypatch auto-reverts.
    monkeypatch.setenv("ENABLE_AUTH", "true")
    settings = Settings()
    application = create_app(settings)
    application.dependency_overrides[get_subgraphx_service] = lambda: subgraphx_mock
    application.dependency_overrides[get_job_manager] = lambda: job_manager
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestSubgraphXEndpoint:
    async def test_post_subgraphx_returns_job_id(self, client, subgraphx_mock):
        resp = await client.post(
            "/api/explain/subgraphx",
            json={"node_id": "entity-42", "model_type": "gcn", "num_iterations": 10, "subgraph_size": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "test-job-id-1234"
        subgraphx_mock.explain.assert_called_once()

    async def test_post_subgraphx_defaults(self, client, subgraphx_mock):
        """Only node_id is required; others have defaults."""
        resp = await client.post(
            "/api/explain/subgraphx",
            json={"node_id": "entity-1"},
        )
        assert resp.status_code == 200
        call_kwargs = subgraphx_mock.explain.call_args.kwargs
        assert call_kwargs["model_type"] == "gcn"
        assert call_kwargs["num_iterations"] == 20
        assert call_kwargs["subgraph_size"] == 5

    async def test_post_subgraphx_missing_node_id(self, client):
        resp = await client.post("/api/explain/subgraphx", json={})
        assert resp.status_code == 422

    async def test_post_subgraphx_invalid_model_type(self, client):
        resp = await client.post(
            "/api/explain/subgraphx",
            json={"node_id": "e1", "model_type": "invalid_model"},
        )
        assert resp.status_code == 422

    async def test_post_subgraphx_iterations_too_low(self, client):
        resp = await client.post(
            "/api/explain/subgraphx",
            json={"node_id": "e1", "num_iterations": 2},
        )
        assert resp.status_code == 422

    async def test_post_subgraphx_iterations_too_high(self, client):
        resp = await client.post(
            "/api/explain/subgraphx",
            json={"node_id": "e1", "num_iterations": 200},
        )
        assert resp.status_code == 422

    async def test_post_subgraphx_size_too_small(self, client):
        resp = await client.post(
            "/api/explain/subgraphx",
            json={"node_id": "e1", "subgraph_size": 1},
        )
        assert resp.status_code == 422

    async def test_post_subgraphx_size_too_large(self, client):
        resp = await client.post(
            "/api/explain/subgraphx",
            json={"node_id": "e1", "subgraph_size": 50},
        )
        assert resp.status_code == 422

    async def test_post_subgraphx_gat_model(self, client, subgraphx_mock):
        resp = await client.post(
            "/api/explain/subgraphx",
            json={"node_id": "e1", "model_type": "gat"},
        )
        assert resp.status_code == 200
        assert subgraphx_mock.explain.call_args.kwargs["model_type"] == "gat"

    async def test_post_subgraphx_graphsage_model(self, client, subgraphx_mock):
        resp = await client.post(
            "/api/explain/subgraphx",
            json={"node_id": "e1", "model_type": "graphsage"},
        )
        assert resp.status_code == 200
        assert subgraphx_mock.explain.call_args.kwargs["model_type"] == "graphsage"

    async def test_post_subgraphx_uses_tenant_header(self, client, subgraphx_mock):
        resp = await client.post(
            "/api/explain/subgraphx",
            json={"node_id": "e1"},
            headers={"X-Tenant-Id": TENANT_ID},
        )
        assert resp.status_code == 200
        assert subgraphx_mock.explain.call_args.kwargs["tenant_id"] == TENANT_ID

    async def test_post_subgraphx_service_error_returns_500(self, client, subgraphx_mock):
        from clients.graph_adapter import GraphAdapterError

        subgraphx_mock.explain.side_effect = GraphAdapterError("Boom")
        resp = await client.post(
            "/api/explain/subgraphx",
            json={"node_id": "e1"},
        )
        assert resp.status_code == 500
        assert "failed" in resp.json()["detail"].lower()


class TestJobStatusEndpoint:
    async def test_get_job_not_found(self, client):
        resp = await client.get("/api/explain/jobs/nonexistent-id")
        assert resp.status_code == 404

    async def test_get_job_pending(self, client, job_manager):
        job_id = await job_manager.create_job("subgraphx", tenant_id=TENANT_ID)
        resp = await client.get(f"/api/explain/jobs/{job_id}", headers={"X-Tenant-Id": TENANT_ID})
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["status"] == "pending"
        assert data["progress"] == 0.0
        assert data["result"] is None
        assert data["computation_time_ms"] is None

    async def test_get_job_cross_tenant_is_404(self, client, job_manager):
        """IDOR guard: another tenant cannot read this job."""
        job_id = await job_manager.create_job("subgraphx", tenant_id=TENANT_ID)
        resp = await client.get(f"/api/explain/jobs/{job_id}", headers={"X-Tenant-Id": OTHER_TENANT_ID})
        assert resp.status_code == 404

    async def test_get_job_running(self, client, job_manager):
        job_id = await job_manager.create_job("subgraphx", tenant_id=TENANT_ID)
        await job_manager.update_job(job_id, status="running", progress=0.5)
        resp = await client.get(f"/api/explain/jobs/{job_id}", headers={"X-Tenant-Id": TENANT_ID})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert data["progress"] == 0.5

    async def test_get_job_completed_with_result(self, client, job_manager):
        job_id = await job_manager.create_job("subgraphx", tenant_id=TENANT_ID)
        result = {
            "node_id": "A",
            "model_type": "gcn",
            "baseline_score": 0.85,
            "influential_subgraphs": [{"nodes": ["A", "B"], "reward": 0.3, "shapley_values": {"A": 0.2, "B": 0.1}}],
            "num_iterations": 20,
            "computation_time_ms": 1234.56,
        }
        await job_manager.update_job(job_id, status="completed", result=result)
        resp = await client.get(f"/api/explain/jobs/{job_id}", headers={"X-Tenant-Id": TENANT_ID})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["progress"] == 1.0
        assert data["result"]["baseline_score"] == 0.85
        assert data["computation_time_ms"] == 1234.56

    async def test_get_job_failed(self, client, job_manager):
        job_id = await job_manager.create_job("subgraphx", tenant_id=TENANT_ID)
        await job_manager.update_job(job_id, status="failed", error="OOM killed")
        resp = await client.get(f"/api/explain/jobs/{job_id}", headers={"X-Tenant-Id": TENANT_ID})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
