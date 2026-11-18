"""The pipeline's per-entity result must tell the caller WHY a score is missing.

#42 stopped `IntegrityResponse` fabricating a 0.5 and gave it `degraded`,
`degraded_components` and `degraded_reason`. `EntityVerificationResult` — what
`/api/pipeline/trigger` actually hands back — never carried those fields, so the
whole diagnosis was dropped at the pipeline boundary: the caller saw
`integrity_score: null` and nothing else.

These tests use REAL `IntegrityResponse` objects rather than `MagicMock`. A
`MagicMock` auto-creates `.degraded` as a truthy Mock, which would make a
degraded-detection test pass no matter what the code does.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from models.verification import IntegrityComponent, IntegrityResponse
from services.pipeline_service import PipelineService

pytestmark = pytest.mark.unit


def _make_settings():
    settings = MagicMock()
    settings.pipeline_auto_cove_threshold = 0.6
    settings.pipeline_auto_formal_threshold = 0.3
    settings.pipeline_batch_size = 50
    settings.pipeline_concurrency = 10
    return settings


def _degraded_integrity(entity_id: str = "e1") -> IntegrityResponse:
    """What integrity_service returns when PageRank/authority cannot be measured."""
    return IntegrityResponse(
        entity_id=entity_id,
        integrity_score=None,
        components=[
            IntegrityComponent(name="citation_support", score=0.5, weight=0.4, weighted_score=0.2),
            IntegrityComponent(name="authority", score=None, weight=0.3, weighted_score=None),
        ],
        computation_time_ms=10.0,
        degraded=True,
        degraded_components=["authority"],
        degraded_reason="integrity_score is unavailable: authority could not be measured",
    )


def _healthy_integrity(entity_id: str = "e1", score: float = 0.85) -> IntegrityResponse:
    return IntegrityResponse(
        entity_id=entity_id,
        integrity_score=score,
        components=[IntegrityComponent(name="authority", score=score, weight=1.0, weighted_score=score)],
        computation_time_ms=10.0,
    )


@pytest.fixture
def integrity_service():
    mock = AsyncMock()
    mock.compute_integrity = AsyncMock()
    return mock


@pytest.fixture
def graph_client():
    mock = AsyncMock()
    mock.get_node = AsyncMock(return_value={"id": "e1", "properties": {"name": "Test Claim"}})
    mock.execute_cypher = AsyncMock(return_value={"results": []})
    return mock


@pytest.fixture
def cove_service():
    mock = AsyncMock()
    resp = MagicMock()
    resp.overall_verdict = "verified"
    resp.confidence = 0.8
    mock.verify_claim = AsyncMock(return_value=resp)
    return mock


@pytest.fixture
def formal_service():
    mock = AsyncMock()
    translated = MagicMock()
    translated.formal_expression = "(assert true)"
    translated.target_language = "z3"
    mock.translate = AsyncMock(return_value=translated)
    verified = MagicMock()
    verified.model_dump = MagicMock(return_value={"is_valid": True})
    mock.verify = AsyncMock(return_value=verified)
    return mock


@pytest.fixture
def pipeline(integrity_service, cove_service, formal_service, graph_client):
    # A degraded (null) score escalates into CoVe and formal verification by
    # design (#42: "an unmeasurable entity is exactly the one worth a second
    # opinion"), so those collaborators must be wired up here or the tests would
    # fail on an unconfigured mock rather than on the behaviour under test.
    return PipelineService(
        integrity_service=integrity_service,
        cove_service=cove_service,
        formal_service=formal_service,
        graph_client=graph_client,
        minio_client=AsyncMock(),
        settings=_make_settings(),
    )


class TestDegradedSurfacing:
    async def test_degraded_reason_reaches_the_batch_caller(self, pipeline, integrity_service):
        """The caller must be able to name the component that could not be measured."""
        integrity_service.compute_integrity.return_value = _degraded_integrity("e1")

        result = await pipeline.verify_entity("e1", tenant_id="t1")

        assert result.integrity_score is None
        assert result.degraded is True
        assert result.degraded_components == ["authority"]
        assert result.degraded_reason
        assert "authority" in result.degraded_reason

    async def test_degraded_is_distinguishable_from_a_measured_score(self, pipeline, integrity_service):
        """A degraded run and a healthy run must not serialise to the same shape."""
        integrity_service.compute_integrity.return_value = _degraded_integrity("e1")
        degraded = (await pipeline.verify_entity("e1", tenant_id="t1")).model_dump()

        integrity_service.compute_integrity.return_value = _healthy_integrity("e2")
        healthy = (await pipeline.verify_entity("e2", tenant_id="t1")).model_dump()

        assert degraded["degraded"] is True
        assert healthy["degraded"] is False
        assert healthy["degraded_components"] == []
        assert healthy["degraded_reason"] is None

    async def test_degraded_is_not_reported_as_a_pipeline_error(self, pipeline, integrity_service):
        """Degradation is not an exception — `error` stays None, which is exactly
        why the dedicated fields are needed to explain the null score."""
        integrity_service.compute_integrity.return_value = _degraded_integrity("e1")

        result = await pipeline.verify_entity("e1", tenant_id="t1")

        assert result.error is None
        assert result.degraded is True

    async def test_healthy_entity_is_never_marked_degraded(self, pipeline, integrity_service):
        """Negative control — a real measurement must not be flagged as an outage."""
        integrity_service.compute_integrity.return_value = _healthy_integrity("e1", 0.85)

        result = await pipeline.verify_entity("e1", tenant_id="t1")

        assert result.integrity_score == 0.85
        assert result.degraded is False
        assert result.degraded_components == []
        assert result.degraded_reason is None

    async def test_verify_batch_carries_degradation_per_entity(self, pipeline, integrity_service):
        """A batch mixes healthy and degraded entities; each keeps its own verdict."""

        async def _per_entity(entity_id: str, **_kwargs):
            return _degraded_integrity(entity_id) if entity_id == "bad" else _healthy_integrity(entity_id)

        integrity_service.compute_integrity.side_effect = _per_entity

        results = await pipeline.verify_batch(entity_ids=["good", "bad"], tenant_id="t1")
        by_id = {r.entity_id: r for r in results}

        assert by_id["good"].degraded is False
        assert by_id["bad"].degraded is True
        assert by_id["bad"].degraded_components == ["authority"]
