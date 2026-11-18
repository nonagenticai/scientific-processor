"""Tests for stub routers (501 stubs), error handler branches in topology/verification,
health root redirect, log_config coverage, and app factory branches.
"""

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
from dependencies import (
    get_centrality_service,
    get_community_service,
    get_contradiction_service,
    get_integrity_service,
    get_ragas_service,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def app(settings):
    application = create_app(settings)
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
def client(app):
    return TestClient(app)


# ===========================================================================
# Explainability router — no longer stubs (implemented in Phase 2.0-C)
# Validation tests only
# ===========================================================================


class TestExplainabilityValidation:
    def test_subgraphx_missing_body_returns_422(self, client):
        resp = client.post("/api/explain/subgraphx")
        assert resp.status_code == 422

    def test_get_explanation_job_not_found_returns_404(self, client):
        resp = client.get("/api/explain/jobs/some-job-id")
        assert resp.status_code == 404


# ===========================================================================
# Formal router — no longer stubs (implemented in Phase 2.0-D)
# Validation tests only (missing body → 422)
# ===========================================================================


class TestFormalValidation:
    def test_translate_missing_body_returns_422(self, client):
        resp = client.post("/api/formal/translate")
        assert resp.status_code == 422

    def test_verify_missing_body_returns_422(self, client):
        resp = client.post("/api/formal/verify")
        assert resp.status_code == 422


# ===========================================================================
# Sandbox router — no longer stubs (implemented in Phase 2.0-B)
# Validation tests only
# ===========================================================================


class TestSandboxValidation:
    def test_execute_missing_body_returns_422(self, client):
        resp = client.post("/api/sandbox/execute")
        assert resp.status_code == 422

    def test_get_sandbox_job_not_found_returns_404(self, client):
        resp = client.get("/api/sandbox/jobs/job-456")
        assert resp.status_code == 404

    def test_get_sandbox_results_not_found_returns_404(self, client):
        resp = client.get("/api/sandbox/results/job-456")
        assert resp.status_code == 404


# ===========================================================================
# Health router — root redirect (missing lines 21-23)
# ===========================================================================


class TestHealthRoot:
    def test_root_redirects_to_docs(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 307
        assert resp.headers["location"] == "/docs"


# ===========================================================================
# Topology router — error handler branches (lines 75-77, 95-97, 114-116)
# ===========================================================================


class TestTopologyErrorPaths:
    def _make_failing_service(self):
        svc = AsyncMock()
        svc.side_effect = RuntimeError("boom")
        return svc

    def test_contradictions_service_error_returns_500(self, app):
        """Trigger the except branch in detect_contradictions."""
        failing = AsyncMock()
        failing.detect_contradictions = AsyncMock(side_effect=KeyError("db down"))

        app.dependency_overrides[get_contradiction_service] = lambda: failing
        with TestClient(app) as c:
            resp = c.post("/api/topology/contradictions", json={"entity_id": "n1"})
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Contradiction detection failed"

    def test_authority_service_error_returns_500(self, app):
        """Trigger the except branch in compute_authority."""
        failing = AsyncMock()
        failing.compute_authority = AsyncMock(side_effect=KeyError("timeout"))

        app.dependency_overrides[get_centrality_service] = lambda: failing
        with TestClient(app) as c:
            resp = c.post("/api/topology/authority", json={"entity_id": "n1", "algorithm": "pagerank"})
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Authority computation failed"

    def test_communities_service_error_returns_500(self, app):
        """Trigger the except branch in detect_communities."""
        failing = AsyncMock()
        failing.detect_communities = AsyncMock(side_effect=KeyError("oom"))

        app.dependency_overrides[get_community_service] = lambda: failing
        with TestClient(app) as c:
            resp = c.post("/api/topology/communities", json={"algorithm": "louvain"})
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Community detection failed"


# ===========================================================================
# Verification router — error handler branches (lines 73-75, 98-100)
# ===========================================================================


class TestVerificationErrorPaths:
    def test_ragas_service_error_returns_500(self, app):
        """Trigger the except branch in verify_ragas."""
        failing = AsyncMock()
        failing.evaluate = AsyncMock(side_effect=KeyError("embedding service down"))

        app.dependency_overrides[get_ragas_service] = lambda: failing
        with TestClient(app) as c:
            resp = c.post(
                "/api/verify/ragas",
                json={
                    "question": "What causes X?",
                    "answer": "Y causes X.",
                    "contexts": ["Y causes X."],
                },
            )
        assert resp.status_code == 500
        assert resp.json()["detail"] == "RAGAS evaluation failed"

    def test_integrity_service_error_returns_500(self, app):
        """Trigger the except branch in compute_integrity."""
        failing = AsyncMock()
        failing.compute_integrity = AsyncMock(side_effect=KeyError("graph unreachable"))

        app.dependency_overrides[get_integrity_service] = lambda: failing
        with TestClient(app) as c:
            resp = c.post("/api/verify/integrity", json={"entity_id": "e1"})
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Integrity computation failed"


# ===========================================================================
# log_config.py — console format branch and service context (lines 19-20, 28-31, 77)
# ===========================================================================


class TestLogConfig:
    def test_get_logger_returns_logger(self):
        from log_config import get_logger

        logger = get_logger("test_logger")
        assert logger is not None

    def test_add_service_context_keys(self, monkeypatch):
        monkeypatch.setenv("OTEL_SERVICE_NAME", "test-svc")
        monkeypatch.setenv("ENVIRONMENT", "testing")
        from log_config import add_service_context

        event_dict: dict = {}
        result = add_service_context(None, None, event_dict)
        assert result["service"] == "test-svc"
        assert result["environment"] == "testing"

    def test_configure_logging_idempotent(self):
        import log_config

        # Reset the guard
        log_config._configured = False
        log_config.configure_logging()
        assert log_config._configured is True
        # Second call is a no-op (no error)
        log_config.configure_logging()
        assert log_config._configured is True
        # Clean up: leave it configured (safe for other tests)

    def test_configure_logging_console_format(self, monkeypatch):
        import log_config

        # Reset guard so configure_logging actually runs
        log_config._configured = False
        monkeypatch.setenv("LOG_FORMAT", "console")
        log_config.configure_logging()
        assert log_config._configured is True
        # Reset for subsequent tests
        log_config._configured = False
        monkeypatch.setenv("LOG_FORMAT", "json")
        log_config.configure_logging()

    def test_add_trace_context_with_otel_available(self, monkeypatch):
        """Exercise lines 28-31: trace context injection with a recording span."""
        from unittest.mock import MagicMock

        import log_config

        mock_span_ctx = MagicMock()
        mock_span_ctx.is_valid = True
        mock_span_ctx.trace_id = 0x1234567890ABCDEF1234567890ABCDEF
        mock_span_ctx.span_id = 0x1234567890ABCDEF

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.get_span_context.return_value = mock_span_ctx

        # Ensure OTEL_AVAILABLE is True and trace module is set
        orig_avail = log_config.OTEL_AVAILABLE
        orig_trace = log_config.trace
        try:
            log_config.OTEL_AVAILABLE = True
            mock_trace = MagicMock()
            mock_trace.get_current_span.return_value = mock_span
            log_config.trace = mock_trace

            event_dict: dict = {}
            result = log_config.add_trace_context(None, None, event_dict)
            assert "trace_id" in result
            assert "span_id" in result
            assert len(result["trace_id"]) == 32
            assert len(result["span_id"]) == 16
        finally:
            log_config.OTEL_AVAILABLE = orig_avail
            log_config.trace = orig_trace

    def test_add_trace_context_otel_not_available(self):
        """When OTEL_AVAILABLE=False, trace context is not injected (lines 19-20 path)."""
        import log_config

        orig_avail = log_config.OTEL_AVAILABLE
        try:
            log_config.OTEL_AVAILABLE = False
            event_dict: dict = {}
            result = log_config.add_trace_context(None, None, event_dict)
            assert "trace_id" not in result
            assert "span_id" not in result
        finally:
            log_config.OTEL_AVAILABLE = orig_avail

    def test_add_trace_context_span_not_recording(self):
        """When span is not recording, no trace context added."""
        from unittest.mock import MagicMock

        import log_config

        orig_avail = log_config.OTEL_AVAILABLE
        orig_trace = log_config.trace
        try:
            log_config.OTEL_AVAILABLE = True
            mock_span = MagicMock()
            mock_span.is_recording.return_value = False
            mock_trace = MagicMock()
            mock_trace.get_current_span.return_value = mock_span
            log_config.trace = mock_trace

            event_dict: dict = {}
            result = log_config.add_trace_context(None, None, event_dict)
            assert "trace_id" not in result
        finally:
            log_config.OTEL_AVAILABLE = orig_avail
            log_config.trace = orig_trace


# ===========================================================================
# app.py — auth middleware enabled branch (lines 84-85)
# ===========================================================================


class TestAppFactory:
    def test_auth_middleware_enabled(self, monkeypatch):
        """When ENABLE_AUTH=true, AuthMiddleware is added to the app."""
        monkeypatch.setenv("ENABLE_AUTH", "true")
        from config import Settings, get_settings

        get_settings.cache_clear()
        s = Settings()
        assert s.enable_auth is True
        application = create_app(s)
        middleware_classes = [getattr(m.cls, "__name__", "") for m in application.user_middleware]
        assert "AuthMiddleware" in middleware_classes

    def test_cors_wildcard_disables_credentials(self, monkeypatch):
        """When CORS_ALLOW_ORIGINS=*, allow_credentials should be False."""
        monkeypatch.setenv("CORS_ALLOW_ORIGINS", "*")
        monkeypatch.setenv("ENABLE_AUTH", "false")
        from config import Settings, get_settings

        get_settings.cache_clear()
        s = Settings()
        application = create_app(s)
        # Find the CORSMiddleware and verify credentials are disabled
        cors_mw = None
        for m in application.user_middleware:
            if getattr(m.cls, "__name__", "") == "CORSMiddleware":
                cors_mw = m
                break
        assert cors_mw is not None
        assert cors_mw.kwargs.get("allow_credentials") is False

    def test_openapi_customization(self, monkeypatch):
        """Calling the OpenAPI customization produces BearerAuth and TenantIdHeader."""
        monkeypatch.setenv("ENABLE_AUTH", "false")
        monkeypatch.setenv("ENABLE_DOCS", "true")
        from config import Settings, get_settings

        get_settings.cache_clear()
        s = Settings()
        application = create_app(s)
        with TestClient(application) as c:
            resp = c.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "BearerAuth" in schema["components"]["securitySchemes"]
        assert "TenantIdHeader" in schema["components"]["parameters"]

    def test_openapi_cached_second_call(self, monkeypatch):
        """Second call to custom_openapi returns cached schema (line 91-92)."""
        monkeypatch.setenv("ENABLE_AUTH", "false")
        monkeypatch.setenv("ENABLE_DOCS", "true")
        from config import Settings, get_settings

        get_settings.cache_clear()
        s = Settings()
        application = create_app(s)
        with TestClient(application) as c:
            resp1 = c.get("/openapi.json")
            resp2 = c.get("/openapi.json")
        assert resp1.json() == resp2.json()

    def test_openapi_non_dict_operation_skipped(self, monkeypatch):
        """Line 131: non-dict entries in path_item.values() are skipped."""
        monkeypatch.setenv("ENABLE_AUTH", "false")
        monkeypatch.setenv("ENABLE_DOCS", "true")
        from config import Settings, get_settings

        get_settings.cache_clear()
        s = Settings()
        application = create_app(s)
        # Force openapi_schema to None so custom_openapi rebuilds it
        application.openapi_schema = None

        # First build the schema normally, then inject a non-dict value and rebuild
        from unittest.mock import patch

        from fastapi.openapi.utils import get_openapi

        real_schema = get_openapi(
            title=application.title,
            version=application.version,
            description=application.description,
            routes=application.routes,
        )
        # Inject a non-dict value to exercise the isinstance(operation, dict) guard
        first_path = next(iter(real_schema["paths"]))
        real_schema["paths"][first_path]["summary"] = "A string, not a dict"

        with patch("fastapi.openapi.utils.get_openapi", return_value=real_schema):
            schema = application.openapi()
        assert "paths" in schema
        # Verify the non-dict entry was skipped (no crash) and schema was built
        assert "BearerAuth" in schema["components"]["securitySchemes"]
