"""Unit tests for sandbox router endpoints — /api/sandbox/execute, /jobs, /results."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

os.environ["ENABLE_AUTH"] = "false"
os.environ["ENABLE_DOCS"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from app import create_app
from config import Settings
from dependencies import get_sandbox_service

pytestmark = pytest.mark.unit

# tenant ids are native postgres uuids since migration 005 — the boundary
# (auth.dependencies._require_uuid_tenant) 400s a slug like "custom-tenant".
CUSTOM_TENANT_ID = "0195f48b-d000-7000-8000-0000000000b1"


@pytest.fixture
def sandbox_service_mock():
    mock = AsyncMock()
    return mock


@pytest.fixture
def app(sandbox_service_mock, monkeypatch):
    # Auth ON so the middleware→router tenant flow runs: routers now read the
    # authoritative request.state.tenant_id (set by AuthMiddleware from the
    # trusted-net raw X-Tenant-Id), never the raw header directly. The peer is
    # injected explicitly by the `client` fixture below. monkeypatch auto-reverts.
    monkeypatch.setenv("ENABLE_AUTH", "true")
    settings = Settings()
    application = create_app(settings)
    application.dependency_overrides[get_sandbox_service] = lambda: sandbox_service_mock
    yield application
    application.dependency_overrides.clear()


#: A REAL loopback peer. ``auth.middleware._is_trusted_network`` no longer
#: honours Starlette's default ``"testclient"`` sentinel — a test-only escape
#: hatch has no business in production auth — so every test that needs the
#: trusted-network branch must present a peer production would also trust.
TRUSTED_PEER = ("127.0.0.1", 50000)


@pytest.fixture
def client(app):
    return TestClient(app, client=TRUSTED_PEER)


class TestExecuteEndpoint:
    def test_execute_returns_job_id(self, client, sandbox_service_mock):
        sandbox_service_mock.execute = AsyncMock(return_value="job-uuid-123")

        resp = client.post(
            "/api/sandbox/execute",
            json={"code": "print('hello')"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "job-uuid-123"

        # Verify service was called with correct args
        sandbox_service_mock.execute.assert_awaited_once()
        call_kwargs = sandbox_service_mock.execute.call_args.kwargs
        assert call_kwargs["code"] == "print('hello')"
        assert call_kwargs["language"] == "python"
        assert call_kwargs["timeout_seconds"] == 60
        assert call_kwargs["input_data"] is None
        assert call_kwargs["tenant_id"] == "0195f48b-d000-7000-8000-000000000001"

    def test_execute_with_input_data(self, client, sandbox_service_mock):
        sandbox_service_mock.execute = AsyncMock(return_value="job-uuid-456")

        resp = client.post(
            "/api/sandbox/execute",
            json={
                "code": "import json; print(json.load(open('input.json')))",
                "language": "python",
                "timeout_seconds": 120,
                "input_data": {"x": [1, 2, 3]},
            },
        )
        assert resp.status_code == 200
        call_kwargs = sandbox_service_mock.execute.call_args.kwargs
        assert call_kwargs["input_data"] == {"x": [1, 2, 3]}
        assert call_kwargs["timeout_seconds"] == 120

    def test_execute_with_tenant_header(self, client, sandbox_service_mock):
        sandbox_service_mock.execute = AsyncMock(return_value="job-uuid-789")

        resp = client.post(
            "/api/sandbox/execute",
            json={"code": "1+1"},
            headers={"X-Tenant-Id": CUSTOM_TENANT_ID},
        )
        assert resp.status_code == 200
        call_kwargs = sandbox_service_mock.execute.call_args.kwargs
        assert call_kwargs["tenant_id"] == CUSTOM_TENANT_ID

    def test_execute_missing_code_returns_422(self, client):
        resp = client.post("/api/sandbox/execute", json={})
        assert resp.status_code == 422

    def test_execute_code_too_long_returns_422(self, client):
        resp = client.post("/api/sandbox/execute", json={"code": "x" * 50001})
        assert resp.status_code == 422

    def test_execute_timeout_too_low_returns_422(self, client):
        resp = client.post("/api/sandbox/execute", json={"code": "1+1", "timeout_seconds": 1})
        assert resp.status_code == 422

    def test_execute_timeout_too_high_returns_422(self, client):
        resp = client.post("/api/sandbox/execute", json={"code": "1+1", "timeout_seconds": 999})
        assert resp.status_code == 422

    def test_execute_invalid_language_returns_422(self, client):
        resp = client.post("/api/sandbox/execute", json={"code": "1+1", "language": "ruby"})
        assert resp.status_code == 422

    def test_execute_service_error_returns_500(self, client, sandbox_service_mock):
        sandbox_service_mock.execute = AsyncMock(side_effect=OSError("k8s down"))

        resp = client.post("/api/sandbox/execute", json={"code": "print(1)"})
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Sandbox execution failed"


class TestGetJobEndpoint:
    def test_get_job_returns_status(self, client, sandbox_service_mock):
        from datetime import UTC, datetime

        from models.jobs import JobStatus

        job = JobStatus(
            job_id="j1",
            job_type="sandbox",
            status="running",
            created_at=datetime(2026, 3, 19, 12, 0, 0, tzinfo=UTC),
            tenant_id=CUSTOM_TENANT_ID,
        )
        sandbox_service_mock.get_status = AsyncMock(return_value=job)

        resp = client.get("/api/sandbox/jobs/j1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "j1"
        assert data["status"] == "running"
        assert data["created_at"] is not None
        assert data["completed_at"] is None
        assert data["error"] is None

    def test_get_job_completed_with_execution_time(self, client, sandbox_service_mock):
        from datetime import UTC, datetime

        from models.jobs import JobStatus

        job = JobStatus(
            job_id="j2",
            job_type="sandbox",
            status="completed",
            created_at=datetime(2026, 3, 19, 12, 0, 0, tzinfo=UTC),
            completed_at=datetime(2026, 3, 19, 12, 1, 0, tzinfo=UTC),
            result={"logs": "output", "execution_time_ms": 1234.5},
            result_url="sandbox/j2/output.txt",
            tenant_id=CUSTOM_TENANT_ID,
        )
        sandbox_service_mock.get_status = AsyncMock(return_value=job)

        resp = client.get("/api/sandbox/jobs/j2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["completed_at"] is not None
        assert data["execution_time_ms"] == 1234.5
        assert data["result_url"] == "sandbox/j2/output.txt"

    def test_get_job_not_found_returns_404(self, client, sandbox_service_mock):
        sandbox_service_mock.get_status = AsyncMock(return_value=None)

        resp = client.get("/api/sandbox/jobs/nonexistent")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Job not found"

    def test_get_job_failed_with_error(self, client, sandbox_service_mock):
        from datetime import UTC, datetime

        from models.jobs import JobStatus

        job = JobStatus(
            job_id="j3",
            job_type="sandbox",
            status="failed",
            created_at=datetime(2026, 3, 19, 12, 0, 0, tzinfo=UTC),
            completed_at=datetime(2026, 3, 19, 12, 0, 30, tzinfo=UTC),
            error="ZeroDivisionError: division by zero",
            tenant_id=CUSTOM_TENANT_ID,
        )
        sandbox_service_mock.get_status = AsyncMock(return_value=job)

        resp = client.get("/api/sandbox/jobs/j3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "ZeroDivisionError" in data["error"]


class TestGetResultsEndpoint:
    def test_get_results_returns_url(self, client, sandbox_service_mock):
        sandbox_service_mock.get_results = AsyncMock(return_value="https://minio.example/sandbox/j1/output.txt?sig=abc")

        resp = client.get("/api/sandbox/results/j1")
        assert resp.status_code == 200
        data = resp.json()
        assert "minio.example" in data["result_url"]

    def test_get_results_not_available_returns_404(self, client, sandbox_service_mock):
        sandbox_service_mock.get_results = AsyncMock(return_value=None)

        resp = client.get("/api/sandbox/results/nonexistent")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Results not available"

    def test_get_results_pending_job_returns_404(self, client, sandbox_service_mock):
        sandbox_service_mock.get_results = AsyncMock(return_value=None)

        resp = client.get("/api/sandbox/results/pending-job")
        assert resp.status_code == 404
