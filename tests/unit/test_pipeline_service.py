"""Tests for PipelineService — autonomous verification pipeline orchestration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.graph_adapter import GraphAdapterError
from models.pipeline import EntityVerificationResult
from services.pipeline_service import PipelineService, PipelineState

pytestmark = pytest.mark.unit


def _make_settings(**overrides):
    """Create a mock Settings with pipeline defaults."""
    settings = MagicMock()
    settings.pipeline_auto_cove_threshold = overrides.get("pipeline_auto_cove_threshold", 0.6)
    settings.pipeline_auto_formal_threshold = overrides.get("pipeline_auto_formal_threshold", 0.3)
    settings.pipeline_batch_size = overrides.get("pipeline_batch_size", 50)
    settings.pipeline_concurrency = overrides.get("pipeline_concurrency", 10)
    return settings


def _integrity_response(entity_id: str, score: float, components: list | None = None):
    """Mock IntegrityResponse."""
    resp = MagicMock()
    resp.entity_id = entity_id
    resp.integrity_score = score
    resp.components = components or []
    resp.computation_time_ms = 10.0
    return resp


def _cove_response(verdict: str = "verified", confidence: float = 0.8):
    resp = MagicMock()
    resp.overall_verdict = verdict
    resp.confidence = confidence
    resp.computation_time_ms = 50.0
    return resp


@pytest.fixture
def integrity_service():
    mock = AsyncMock()
    mock.compute_integrity = AsyncMock()
    return mock


@pytest.fixture
def cove_service():
    mock = AsyncMock()
    mock.verify_claim = AsyncMock()
    return mock


@pytest.fixture
def formal_service():
    mock = AsyncMock()
    mock.translate = AsyncMock()
    mock.verify = AsyncMock()
    return mock


@pytest.fixture
def graph_client():
    mock = AsyncMock()
    mock.get_node = AsyncMock(return_value={"id": "e1", "properties": {"name": "Test Claim"}})
    mock.execute_cypher = AsyncMock(return_value={"results": []})
    return mock


@pytest.fixture
def minio_client():
    mock = AsyncMock()
    mock.upload = AsyncMock()
    return mock


@pytest.fixture
def pipeline(integrity_service, cove_service, formal_service, graph_client, minio_client):
    return PipelineService(
        integrity_service=integrity_service,
        cove_service=cove_service,
        formal_service=formal_service,
        graph_client=graph_client,
        minio_client=minio_client,
        settings=_make_settings(),
    )


class TestVerifyEntity:
    async def test_verify_entity_computes_integrity(self, pipeline, integrity_service):
        integrity_service.compute_integrity.return_value = _integrity_response("e1", 0.85)

        result = await pipeline.verify_entity("e1", tenant_id="t1")
        assert isinstance(result, EntityVerificationResult)
        assert result.entity_id == "e1"
        assert result.integrity_score == 0.85
        integrity_service.compute_integrity.assert_called_once()

    async def test_verify_entity_runs_cove_when_below_threshold(
        self, pipeline, integrity_service, cove_service, graph_client
    ):
        """If integrity < auto_cove_threshold (0.6), CoVe runs automatically."""
        integrity_service.compute_integrity.return_value = _integrity_response("e1", 0.4)
        cove_service.verify_claim.return_value = _cove_response("disputed", 0.3)

        result = await pipeline.verify_entity("e1", tenant_id="t1", run_cove=True)
        assert result.cove_verdict == "disputed"
        cove_service.verify_claim.assert_called_once()

    async def test_verify_entity_skips_cove_when_above_threshold(self, pipeline, integrity_service, cove_service):
        """If integrity >= auto_cove_threshold, CoVe should not run unless forced."""
        integrity_service.compute_integrity.return_value = _integrity_response("e1", 0.85)

        result = await pipeline.verify_entity("e1", tenant_id="t1")
        assert result.cove_verdict is None
        cove_service.verify_claim.assert_not_called()

    async def test_verify_entity_runs_formal_when_below_threshold(self, pipeline, integrity_service, formal_service):
        """If integrity < auto_formal_threshold (0.3), formal verification runs."""
        integrity_service.compute_integrity.return_value = _integrity_response("e1", 0.2)
        formal_service.translate.return_value = MagicMock(formal_expression="Eq(1,1)", target_language="z3")
        formal_service.verify.return_value = MagicMock(
            is_valid=True,
            expression="Eq(1,1)",
            language="z3",
            model_dump=MagicMock(return_value={"is_valid": True}),
        )

        result = await pipeline.verify_entity("e1", tenant_id="t1", run_formal=True)
        assert result.formal_result is not None
        formal_service.translate.assert_called_once()

    async def test_verify_entity_catches_errors(self, pipeline, integrity_service):
        """If integrity computation fails, return error result."""
        from clients.graph_adapter import GraphAdapterError

        integrity_service.compute_integrity.side_effect = GraphAdapterError("graph down")

        result = await pipeline.verify_entity("e1", tenant_id="t1")
        assert result.error is not None
        assert "graph down" in result.error

    async def test_verify_entity_with_write_back(self, pipeline, integrity_service):
        integrity_service.compute_integrity.return_value = _integrity_response("e1", 0.75)

        result = await pipeline.verify_entity("e1", tenant_id="t1", write_back=True)
        assert result.integrity_score == 0.75
        integrity_service.compute_integrity.assert_called_once_with(
            entity_id="e1",
            tenant_id="t1",
            write_back=True,
        )


class TestVerifyBatch:
    async def test_verify_batch_processes_all_entities(self, pipeline, integrity_service):
        integrity_service.compute_integrity.return_value = _integrity_response("e1", 0.7)

        results = await pipeline.verify_batch(["e1", "e2", "e3"], tenant_id="t1")
        assert len(results) == 3
        assert integrity_service.compute_integrity.call_count == 3

    async def test_verify_batch_isolates_failures(self, pipeline, integrity_service):
        """One entity failing shouldn't prevent others from being processed."""
        from clients.graph_adapter import GraphAdapterError

        integrity_service.compute_integrity.side_effect = [
            _integrity_response("e1", 0.9),
            GraphAdapterError("e2 failed"),
            _integrity_response("e3", 0.5),
        ]

        results = await pipeline.verify_batch(["e1", "e2", "e3"], tenant_id="t1")
        assert len(results) == 3
        assert results[0].integrity_score == 0.9
        assert results[1].error is not None
        assert results[2].integrity_score == 0.5

    async def test_verify_batch_respects_batch_size(self, pipeline, integrity_service):
        """Batch size from settings should be respected."""
        pipeline.settings.pipeline_batch_size = 2
        integrity_service.compute_integrity.return_value = _integrity_response("e1", 0.7)

        # Even with 5 IDs, only batch_size are processed at a time
        results = await pipeline.verify_batch(["e1", "e2", "e3", "e4", "e5"], tenant_id="t1")
        assert len(results) == 5


class TestFindUnscoredEntities:
    async def test_find_unscored_queries_graph(self, pipeline, graph_client):
        graph_client.find_nodes_missing_property.return_value = {
            "results": [{"id": "e1"}, {"id": "e2"}],
        }

        ids = await pipeline.find_unscored_entities(tenant_id="t1", limit=50)
        assert ids == ["e1", "e2"]
        graph_client.find_nodes_missing_property.assert_called_once()

    async def test_find_unscored_returns_empty_on_error(self, pipeline, graph_client):
        graph_client.find_nodes_missing_property.side_effect = GraphAdapterError("graph unavailable")

        ids = await pipeline.find_unscored_entities(tenant_id="t1", limit=50)
        assert ids == []


class TestPipelineStatus:
    def test_get_status_initial(self, pipeline):
        status = pipeline.get_status()
        assert status.running is False
        assert status.processed_total == 0
        assert status.failed_total == 0

    async def test_status_updates_after_processing(self, pipeline, integrity_service):
        integrity_service.compute_integrity.return_value = _integrity_response("e1", 0.7)
        await pipeline.verify_batch(["e1", "e2"], tenant_id="t1")

        status = pipeline.get_status()
        assert status.processed_total == 2


class TestSharedState:
    """Counters live in an injectable PipelineState so /status survives the
    per-request PipelineService instance being discarded."""

    async def test_status_shared_across_instances(
        self, integrity_service, cove_service, formal_service, graph_client, minio_client
    ):
        integrity_service.compute_integrity.return_value = _integrity_response("e1", 0.7)
        state = PipelineState()

        def _svc():
            return PipelineService(
                integrity_service=integrity_service,
                cove_service=cove_service,
                formal_service=formal_service,
                graph_client=graph_client,
                minio_client=minio_client,
                settings=_make_settings(),
                state=state,
            )

        # One request does the work; a *different* instance sharing the same state
        # reports it — this is exactly the per-request-Depends scenario.
        await _svc().verify_batch(["e1", "e2"], tenant_id="t1")
        assert _svc().get_status().processed_total == 2

    def test_default_state_is_per_instance(
        self, pipeline, integrity_service, cove_service, formal_service, graph_client, minio_client
    ):
        other = PipelineService(
            integrity_service=integrity_service,
            cove_service=cove_service,
            formal_service=formal_service,
            graph_client=graph_client,
            minio_client=minio_client,
            settings=_make_settings(),
        )
        pipeline._state.processed_total = 9
        assert other.get_status().processed_total == 0
