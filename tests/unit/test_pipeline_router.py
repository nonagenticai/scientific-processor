"""Tests for pipeline router endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from models.pipeline import EntityVerificationResult, PipelineStatusResponse

pytestmark = pytest.mark.unit

TENANT = "n8n-dev"


def _make_pipeline_service():
    svc = MagicMock()
    svc.verify_entity = AsyncMock()
    svc.verify_batch = AsyncMock(return_value=[])
    svc.find_unscored_entities = AsyncMock(return_value=[])
    svc.get_status = MagicMock(return_value=PipelineStatusResponse())
    return svc


@pytest.fixture
def pipeline_svc():
    return _make_pipeline_service()


@pytest.fixture
def client(pipeline_svc):
    import os

    os.environ["ENABLE_AUTH"] = "false"

    from config import get_settings

    get_settings.cache_clear()

    from app import create_app
    from routers.pipeline import get_pipeline_service

    app = create_app()
    app.dependency_overrides[get_pipeline_service] = lambda: pipeline_svc
    return TestClient(app)


class TestPipelineTrigger:
    def test_trigger_returns_completed(self, client, pipeline_svc):
        mock_results = [
            EntityVerificationResult(entity_id="e1", integrity_score=0.8, computation_time_ms=10.0),
        ]
        pipeline_svc.verify_batch.return_value = mock_results

        resp = client.post(
            "/api/pipeline/trigger",
            json={"entity_ids": ["e1"]},
            headers={"X-Tenant-Id": TENANT},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert len(data["results"]) == 1
        assert data["results"][0]["entity_id"] == "e1"

    def test_trigger_empty_entity_ids_rejected(self, client):
        resp = client.post(
            "/api/pipeline/trigger",
            json={"entity_ids": []},
            headers={"X-Tenant-Id": TENANT},
        )
        assert resp.status_code == 422


class TestPipelineWebhook:
    def test_webhook_accepts_node_created(self, client, pipeline_svc):
        pipeline_svc.verify_batch.return_value = []

        resp = client.post(
            "/api/pipeline/webhook",
            json={"event_type": "node_created", "entity_ids": ["e1"]},
            headers={"X-Tenant-Id": TENANT},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"

    def test_webhook_rejects_invalid_event_type(self, client):
        resp = client.post(
            "/api/pipeline/webhook",
            json={"event_type": "invalid", "entity_ids": ["e1"]},
            headers={"X-Tenant-Id": TENANT},
        )
        assert resp.status_code == 422

    def test_webhook_rejects_oversize_entity_ids(self, client):
        """entity_ids beyond max_length=100 must be rejected (matches the trigger
        cap) so the fire-and-forget verify_batch cannot be handed an unbounded list."""
        resp = client.post(
            "/api/pipeline/webhook",
            json={"event_type": "node_created", "entity_ids": [f"e{i}" for i in range(101)]},
            headers={"X-Tenant-Id": TENANT},
        )
        assert resp.status_code == 422

    def test_webhook_schedules_batch_not_awaited(self, client, pipeline_svc, monkeypatch):
        """A large webhook must schedule verification and return "accepted"
        immediately — never hold the connection open for the whole batch (which
        times the sender out into duplicate-retry double-verifies). If the handler
        awaited verify_batch inline, _spawn would never be called."""
        from routers import pipeline as pmod

        captured: list = []

        def _capture(coro):
            captured.append(coro)
            coro.close()  # we don't run it here; avoids a "never awaited" warning
            return None

        monkeypatch.setattr(pmod, "_spawn", _capture)

        resp = client.post(
            "/api/pipeline/webhook",
            json={"event_type": "node_created", "entity_ids": ["e1"]},
            headers={"X-Tenant-Id": TENANT},
        )

        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"
        assert len(captured) == 1  # verify_batch was scheduled, not awaited inline
        pipeline_svc.verify_batch.assert_called_once()


class TestFactoryWiring:
    def test_factory_injects_process_wide_state(self):
        """Every request-scoped PipelineService must share one PipelineState so
        /status reflects real work instead of the discarded instance's zeros."""
        from routers import pipeline as pmod

        svc1 = pmod.get_pipeline_service(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        svc2 = pmod.get_pipeline_service(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        assert svc1._state is svc2._state
        assert svc1._state is pmod._pipeline_state


class TestPipelineBatch:
    def test_batch_finds_and_processes_entities(self, client, pipeline_svc):
        pipeline_svc.find_unscored_entities.return_value = ["e1", "e2"]
        pipeline_svc.verify_batch.return_value = []

        resp = client.post(
            "/api/pipeline/batch",
            json={},
            headers={"X-Tenant-Id": TENANT},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["entities_queued"] == 2


class TestPipelineStatus:
    def test_status_returns_metrics(self, client, pipeline_svc):
        pipeline_svc.get_status.return_value = PipelineStatusResponse(
            running=False,
            processed_total=10,
            failed_total=1,
        )

        resp = client.get("/api/pipeline/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["processed_total"] == 10
        assert data["failed_total"] == 1


class TestBackgroundTaskRetention:
    """Recovered from the stranded PR #50.

    ``routers.pipeline._spawn`` landed with the fire-and-forget webhook/batch work,
    but nothing asserted the strong reference is taken and released. The event loop
    holds only a weak reference, so an unretained task can be garbage collected
    mid-flight and silently cancelled — the webhook would answer "accepted" for a
    batch that never ran.

    ``test_webhook_schedules_batch_not_awaited`` monkeypatches ``_spawn`` away, so it
    can never see this; this drives the real helper.
    """

    async def test_spawned_task_is_retained_then_released(self):
        import asyncio

        from routers import pipeline as pmod

        gate = asyncio.Event()

        async def _work() -> None:
            await gate.wait()

        before = set(pmod._background_tasks)
        task = pmod._spawn(_work())

        new_tasks = set(pmod._background_tasks) - before
        assert new_tasks == {task}, "the spawned task was not retained — it can be GC'd mid-flight"
        assert not task.done()

        gate.set()
        await task
        await asyncio.sleep(0)  # let the done-callback run

        assert task not in pmod._background_tasks, "the done callback must release the reference"


class TestPipelineStatusRealWiring:
    """Recovered from the stranded PR #50.

    ``TestPipelineStatus.test_status_returns_metrics`` overrides the whole service
    with a MagicMock, so it asserts the stub and can never fail on a real regression.
    These drive the genuine ``get_pipeline_service`` factory and the real
    ``PipelineService``, overriding only the LEAF dependencies — which is what makes
    the process-wide ``_pipeline_state`` observable end to end over HTTP.
    """

    @pytest.fixture
    def real_wiring(self):
        import os

        os.environ["ENABLE_AUTH"] = "false"

        from config import get_settings

        get_settings.cache_clear()

        from app import create_app
        from dependencies import (
            get_cove_service,
            get_formal_service,
            get_graph_client,
            get_integrity_service,
            get_minio_client,
        )
        from models.verification import IntegrityResponse
        from routers import pipeline as pmod
        from services.pipeline_service import PipelineState

        # The counters are process-wide by design; reset them so each test starts cold.
        original_state = pmod._pipeline_state
        pmod._pipeline_state = PipelineState()

        integrity = AsyncMock()
        # 0.9 clears both the auto-CoVe and auto-formal thresholds, so the pipeline
        # runs the integrity step only and the counters are the only thing under test.
        integrity.compute_integrity = AsyncMock(
            return_value=IntegrityResponse(
                entity_id="e1",
                integrity_score=0.9,
                components=[],
                computation_time_ms=1.0,
            )
        )

        app = create_app()
        app.dependency_overrides[get_integrity_service] = lambda: integrity
        app.dependency_overrides[get_cove_service] = lambda: AsyncMock()
        app.dependency_overrides[get_formal_service] = lambda: AsyncMock()
        app.dependency_overrides[get_graph_client] = lambda: AsyncMock()
        app.dependency_overrides[get_minio_client] = lambda: None

        yield TestClient(app)

        app.dependency_overrides.clear()
        pmod._pipeline_state = original_state

    def test_status_is_zero_before_any_run(self, real_wiring):
        """The inverse guard: a fresh process must not fabricate counts."""
        data = real_wiring.get("/api/pipeline/status").json()

        assert data["processed_total"] == 0
        assert data["failed_total"] == 0
        assert data["last_run_at"] is None

    def test_counters_survive_across_requests(self, real_wiring):
        """POST /trigger then GET /status over real HTTP.

        The counter written by one request must be visible to the next. Reports
        ``0 != 2`` if the counters ever move back onto the request-scoped
        ``PipelineService`` instance.
        """
        trigger = real_wiring.post(
            "/api/pipeline/trigger",
            json={"entity_ids": ["e1", "e2"]},
            headers={"X-Tenant-Id": TENANT},
        )
        assert trigger.status_code == 200
        assert trigger.json()["status"] == "completed"

        status = real_wiring.get("/api/pipeline/status")

        assert status.status_code == 200
        data = status.json()
        assert data["processed_total"] == 2
        assert data["failed_total"] == 0
        assert data["last_run_at"] is not None

    def test_counters_accumulate_over_multiple_batches(self, real_wiring):
        real_wiring.post("/api/pipeline/trigger", json={"entity_ids": ["e1"]}, headers={"X-Tenant-Id": TENANT})
        real_wiring.post("/api/pipeline/trigger", json={"entity_ids": ["e2", "e3"]}, headers={"X-Tenant-Id": TENANT})

        assert real_wiring.get("/api/pipeline/status").json()["processed_total"] == 3
