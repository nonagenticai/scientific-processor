"""Guardrail tests for the scientific_processor OpenAPI schema.

Asserts the canonical fleet shape: every operation declares an explicit,
non-auto-generated operation_id; operation_ids are unique across the schema;
the id set matches this module's plan; and BearerAuth + TenantIdHeader are
declared in components.

This matters because the `kag` gateway merges this service's `/openapi.json`
under the `/scientific` prefix and re-derives n8n operation names from each
operationId. Auto-generated FastAPI ids (e.g. `cache_clear_api_cache_clear_post`)
produce ugly node operations like `scientificCacheClearApiCacheClearPost`;
explicit ids (`cache_clear`) produce clean `scientificCacheClear`.
"""

from __future__ import annotations

import os
import re

import pytest

# Enable docs (so /openapi.json is served) and disable auth before importing the
# app factory — mirrors tests/unit/test_health.py.
os.environ["ENABLE_AUTH"] = "false"
os.environ["ENABLE_DOCS"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from fastapi.testclient import TestClient

import config
from app import create_app

pytestmark = pytest.mark.unit

EXPECTED_OPERATION_IDS: set[str] = {
    # Health (tag "health")
    "health_check",
    "health_readiness",
    "health_liveness",
    "health_status",
    # Cache (tag "cache")
    "cache_stats",
    "cache_invalidate",
    "cache_clear",
    # Topology (tag "topology")
    "topology_extract_subgraph",
    "topology_detect_contradictions",
    "topology_compute_authority",
    "topology_detect_communities",
    # Verification (tag "verification")
    "verification_cove",
    "verification_ragas",
    "verification_integrity",
    # Statistics (tag "statistics")
    "statistics_meta_analysis",
    "statistics_confidence_interval",
    "statistics_trend",
    # Explainability (tag "explainability")
    "explainability_subgraphx",
    "explainability_get_job",
    # Formal (tag "formal")
    "formal_translate",
    "formal_verify",
    "formal_create_proof",
    "formal_get_proofs",
    # Sandbox (tag "sandbox")
    "sandbox_execute",
    "sandbox_get_job",
    "sandbox_get_results",
    # Pipeline (tag "pipeline")
    "pipeline_trigger",
    "pipeline_webhook",
    "pipeline_batch",
    "pipeline_status",
    "pipeline_history",
    "pipeline_anomalies",
}


@pytest.fixture
def openapi_schema() -> dict:
    config.get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        return resp.json()


def _iter_operations(schema: dict):
    for path, path_item in schema.get("paths", {}).items():
        for method, op in path_item.items():
            if method.lower() in {"get", "post", "put", "patch", "delete", "options", "head"}:
                yield path, method.lower(), op


def test_every_route_has_explicit_operation_id(openapi_schema: dict) -> None:
    """Every operation declares a non-auto-generated operation_id."""
    auto_pattern = re.compile(r"_[a-z0-9_]+_api_[a-z0-9_]+_[a-z0-9_]+_(get|post|put|patch|delete)$")
    failures: list[str] = []
    for path, method, op in _iter_operations(openapi_schema):
        op_id = op.get("operationId")
        if not op_id:
            failures.append(f"{method.upper()} {path}: missing operationId")
        elif auto_pattern.search(op_id):
            failures.append(f"{method.upper()} {path}: operationId looks auto-generated ({op_id})")
    assert not failures, "Routes without an explicit operation_id:\n  " + "\n  ".join(failures)


def test_operation_ids_match_plan(openapi_schema: dict) -> None:
    """The set of operation IDs equals EXPECTED_OPERATION_IDS."""
    actual = {op.get("operationId") for _path, _m, op in _iter_operations(openapi_schema)}
    missing = EXPECTED_OPERATION_IDS - actual
    extra = actual - EXPECTED_OPERATION_IDS
    assert not missing, f"Expected operation_ids not in OpenAPI: {sorted(missing)}"
    assert not extra, f"OpenAPI has operation_ids not in the plan: {sorted(extra)}"


def test_operation_ids_are_unique(openapi_schema: dict) -> None:
    seen: dict[str, str] = {}
    for path, method, op in _iter_operations(openapi_schema):
        op_id = op.get("operationId")
        assert op_id, f"{method.upper()} {path}: missing operationId"
        if op_id in seen:
            pytest.fail(f"Duplicate operationId {op_id!r} on {seen[op_id]} and {method.upper()} {path}")
        seen[op_id] = f"{method.upper()} {path}"


def test_bearer_auth_security_scheme(openapi_schema: dict) -> None:
    schemes = openapi_schema["components"]["securitySchemes"]
    assert "BearerAuth" in schemes
    assert schemes["BearerAuth"]["type"] == "http"
    assert schemes["BearerAuth"]["scheme"] == "bearer"


def test_tenant_id_header_parameter(openapi_schema: dict) -> None:
    params = openapi_schema["components"].get("parameters", {})
    assert "TenantIdHeader" in params
    th = params["TenantIdHeader"]
    assert th["name"] == "X-Tenant-Id"
    assert th["in"] == "header"
