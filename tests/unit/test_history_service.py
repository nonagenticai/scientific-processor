"""Tests for HistoryService — temporal trust tracking via per-record MinIO objects."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from botocore.exceptions import ClientError

from services.history_service import HistoryService


def _not_found_error() -> ClientError:
    """Create a botocore ClientError simulating a 404 (NoSuchKey) from S3/MinIO."""
    return ClientError({"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject")


pytestmark = pytest.mark.unit


@pytest.fixture
def minio_mock():
    mock = AsyncMock()
    mock.upload = AsyncMock()
    mock.download = AsyncMock()
    mock.list_objects = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def service(minio_mock):
    return HistoryService(minio_client=minio_mock)


@pytest.fixture
def service_no_minio():
    return HistoryService(minio_client=None)


def _record_object(**overrides) -> bytes:
    """Serialize a single-record .jsonl object body."""
    entry = {
        "timestamp": overrides.get("timestamp", "2026-03-01T00:00:00+00:00"),
        "integrity_score": overrides.get("integrity_score", 0.5),
        "trigger": overrides.get("trigger", "batch"),
        "components": overrides.get("components", {}),
    }
    return (json.dumps(entry) + "\n").encode()


class TestRecord:
    async def test_record_writes_one_object_no_read(self, service, minio_mock):
        await service.record(
            entity_id="e1",
            tenant_id="t1",
            integrity_score=0.75,
            components={"citation_support": 0.8},
            trigger="pipeline_batch",
        )

        # Per-object write must NOT do a read-modify-write.
        minio_mock.download.assert_not_called()
        minio_mock.upload.assert_called_once()
        call_args = minio_mock.upload.call_args
        key = call_args.kwargs.get("key") or call_args[0][0]
        assert key.startswith("history/t1/e1/")
        assert key.endswith(".jsonl")

        data = call_args.kwargs.get("data") or call_args[0][1]
        entry = json.loads(data.decode().strip())
        assert entry["integrity_score"] == 0.75
        assert entry["trigger"] == "pipeline_batch"

    async def test_concurrent_records_write_distinct_objects(self, service, minio_mock):
        """Concurrent verifies of the same entity must not lose updates — each
        record is an independent object with a unique key (no append race)."""
        await asyncio.gather(
            *[
                service.record(
                    entity_id="e1",
                    tenant_id="t1",
                    integrity_score=0.1 * i,
                    components={},
                    trigger="pipeline",
                )
                for i in range(5)
            ]
        )

        assert minio_mock.upload.await_count == 5
        minio_mock.download.assert_not_called()
        keys = {c.kwargs.get("key") or c.args[0] for c in minio_mock.upload.await_args_list}
        assert len(keys) == 5  # every write went to its own object

    async def test_record_noop_without_minio(self, service_no_minio):
        # Should not raise
        await service_no_minio.record(
            entity_id="e1",
            tenant_id="t1",
            integrity_score=0.5,
            components={},
            trigger="test",
        )


class TestGetHistory:
    async def test_get_history_aggregates_objects(self, service, minio_mock):
        minio_mock.list_objects.return_value = [
            "history/t1/e1/20260302T000000000000-b.jsonl",
            "history/t1/e1/20260301T000000000000-a.jsonl",
        ]

        def _download(key):
            if key.endswith("a.jsonl"):
                return _record_object(timestamp="2026-03-01T00:00:00+00:00", integrity_score=0.5)
            return _record_object(timestamp="2026-03-02T00:00:00+00:00", integrity_score=0.7)

        minio_mock.download.side_effect = lambda key: _download(key)

        history = await service.get_history("e1", "t1")
        assert len(history) == 2
        # Sorted chronologically regardless of listing order.
        assert history[0]["integrity_score"] == 0.5
        assert history[1]["integrity_score"] == 0.7

    async def test_get_history_returns_empty_on_missing(self, service, minio_mock):
        minio_mock.list_objects.return_value = []
        history = await service.get_history("e999", "t1")
        assert history == []

    async def test_get_history_no_minio(self, service_no_minio):
        history = await service_no_minio.get_history("e1", "t1")
        assert history == []


class TestGetAnomalies:
    async def test_detects_score_drop(self, service, minio_mock):
        """Entities whose score dropped by > threshold should be flagged."""
        minio_mock.list_objects.return_value = [
            "history/t1/e1/20260301T000000000000-a.jsonl",
            "history/t1/e1/20260302T000000000000-b.jsonl",
        ]

        def _download(key):
            if key.endswith("a.jsonl"):
                return _record_object(timestamp="2026-03-01T00:00:00+00:00", integrity_score=0.9)
            return _record_object(timestamp="2026-03-02T00:00:00+00:00", integrity_score=0.5)

        minio_mock.download.side_effect = lambda key: _download(key)

        anomalies = await service.get_anomalies("t1", threshold_delta=0.2)
        assert len(anomalies) == 1
        assert anomalies[0]["entity_id"] == "e1"
        assert anomalies[0]["delta"] == pytest.approx(-0.4, abs=0.01)

    async def test_no_anomaly_when_stable(self, service, minio_mock):
        minio_mock.list_objects.return_value = [
            "history/t1/e1/20260301T000000000000-a.jsonl",
            "history/t1/e1/20260302T000000000000-b.jsonl",
        ]

        def _download(key):
            if key.endswith("a.jsonl"):
                return _record_object(timestamp="2026-03-01T00:00:00+00:00", integrity_score=0.8)
            return _record_object(timestamp="2026-03-02T00:00:00+00:00", integrity_score=0.75)

        minio_mock.download.side_effect = lambda key: _download(key)

        anomalies = await service.get_anomalies("t1", threshold_delta=0.2)
        assert len(anomalies) == 0
