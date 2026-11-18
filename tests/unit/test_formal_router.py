"""Tests for formal reasoning router — /api/formal endpoints."""

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
from dependencies import get_formal_service

pytestmark = pytest.mark.unit


@pytest.fixture
def formal_service_mock():
    mock = AsyncMock()
    return mock


@pytest.fixture
def app(formal_service_mock):
    settings = Settings()
    application = create_app(settings)
    application.dependency_overrides[get_formal_service] = lambda: formal_service_mock
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
def client(app):
    return TestClient(app)


class TestTranslateEndpoint:
    def test_translate_basic(self, client, formal_service_mock):
        from models.formal import FormalTranslateResponse

        formal_service_mock.translate.return_value = FormalTranslateResponse(
            claim_text="p implies q",
            formal_expression="Implies(Bool('p'), Bool('q'))",
            target_language="z3",
            confidence=0.8,
            computation_time_ms=42.0,
        )

        resp = client.post("/api/formal/translate", json={"claim_text": "p implies q", "target_language": "z3"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["claim_text"] == "p implies q"
        assert data["formal_expression"] == "Implies(Bool('p'), Bool('q'))"
        assert data["target_language"] == "z3"
        assert data["confidence"] == 0.8
        assert data["computation_time_ms"] == 42.0

    def test_translate_missing_claim_text_returns_422(self, client):
        resp = client.post("/api/formal/translate", json={"target_language": "z3"})
        assert resp.status_code == 422

    def test_translate_claim_too_long_returns_422(self, client):
        resp = client.post("/api/formal/translate", json={"claim_text": "x" * 5001})
        assert resp.status_code == 422

    def test_translate_invalid_language_returns_422(self, client):
        resp = client.post("/api/formal/translate", json={"claim_text": "test", "target_language": "coq"})
        assert resp.status_code == 422

    def test_translate_default_language(self, client, formal_service_mock):
        from models.formal import FormalTranslateResponse

        formal_service_mock.translate.return_value = FormalTranslateResponse(
            claim_text="test",
            formal_expression="Bool('test')",
            target_language="z3",
            confidence=0.8,
            computation_time_ms=10.0,
        )
        resp = client.post("/api/formal/translate", json={"claim_text": "test"})
        assert resp.status_code == 200
        # Default language should be z3
        formal_service_mock.translate.assert_called_once()
        call_args = formal_service_mock.translate.call_args
        assert call_args[0][1] == "z3"  # target_language argument

    def test_translate_service_error_returns_500(self, client, formal_service_mock):
        formal_service_mock.translate.side_effect = ValueError("LLM down")
        resp = client.post("/api/formal/translate", json={"claim_text": "test"})
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Formal translation failed"


class TestVerifyEndpoint:
    def test_verify_z3(self, client, formal_service_mock):
        from models.formal import FormalVerifyResponse

        formal_service_mock.verify.return_value = FormalVerifyResponse(
            expression="Or(Bool('p'), Not(Bool('p')))",
            language="z3",
            is_valid=True,
            proof="Valid by Z3",
            computation_time_ms=5.0,
        )

        resp = client.post(
            "/api/formal/verify",
            json={"formal_expression": "Or(Bool('p'), Not(Bool('p')))", "language": "z3"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_valid"] is True
        assert data["language"] == "z3"

    def test_verify_sympy(self, client, formal_service_mock):
        from models.formal import FormalVerifyResponse

        formal_service_mock.verify.return_value = FormalVerifyResponse(
            expression="Eq(1, 1)",
            language="sympy",
            is_valid=True,
            proof="Expression simplifies to True",
            computation_time_ms=3.0,
        )

        resp = client.post(
            "/api/formal/verify",
            json={"formal_expression": "Eq(1, 1)", "language": "sympy"},
        )
        assert resp.status_code == 200
        assert resp.json()["is_valid"] is True

    def test_verify_invalid_language_returns_422(self, client):
        resp = client.post(
            "/api/formal/verify",
            json={"formal_expression": "test", "language": "coq"},
        )
        assert resp.status_code == 422

    def test_verify_missing_expression_returns_422(self, client):
        resp = client.post("/api/formal/verify", json={"language": "z3"})
        assert resp.status_code == 422

    def test_verify_service_error_returns_500(self, client, formal_service_mock):
        formal_service_mock.verify.side_effect = ValueError("Z3 crashed")
        resp = client.post(
            "/api/formal/verify",
            json={"formal_expression": "Or(Bool('p'), Not(Bool('p')))", "language": "z3"},
        )
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Formal verification failed"


class TestProofEndpoints:
    def test_create_proof(self, client, formal_service_mock):
        formal_service_mock.store_proof.return_value = "proofs/claim-1/20260319T000000Z.json"

        resp = client.post(
            "/api/formal/proofs/claim-1",
            json={"expression": "Or(p, Not(p))", "language": "z3", "is_valid": True, "proof": "Valid by Z3"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "proofs/claim-1/20260319T000000Z.json"
        formal_service_mock.store_proof.assert_called_once()

    def test_create_proof_service_error_returns_500(self, client, formal_service_mock):
        formal_service_mock.store_proof.side_effect = RuntimeError("MinIO down")
        resp = client.post(
            "/api/formal/proofs/claim-1",
            json={"expression": "Or(p, q)", "language": "z3"},
        )
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Proof storage failed"

    def test_get_proofs(self, client, formal_service_mock):
        formal_service_mock.get_proofs.return_value = [
            {"claim_id": "claim-1", "key": "proofs/claim-1/a.json", "is_valid": True},
        ]

        resp = client.get("/api/formal/proofs/claim-1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["claim_id"] == "claim-1"

    def test_get_proofs_empty(self, client, formal_service_mock):
        formal_service_mock.get_proofs.return_value = []
        resp = client.get("/api/formal/proofs/claim-1")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_proofs_service_error_returns_500(self, client, formal_service_mock):
        formal_service_mock.get_proofs.side_effect = OSError("MinIO unreachable")
        resp = client.get("/api/formal/proofs/claim-1")
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Proof retrieval failed"
