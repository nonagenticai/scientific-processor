"""Unit tests for JobManager -- in-memory job lifecycle."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from services.job_manager import JobManager

pytestmark = pytest.mark.unit


class TestJobManager:
    async def test_create_job_returns_id(self):
        mgr = JobManager()
        job_id = await mgr.create_job("sandbox", tenant_id="t1")
        assert isinstance(job_id, str)
        assert len(job_id) == 36  # UUID

    async def test_get_job_returns_created_job(self):
        mgr = JobManager()
        job_id = await mgr.create_job("subgraphx", tenant_id="t1")
        job = await mgr.get_job(job_id)
        assert job is not None
        assert job.job_id == job_id
        assert job.job_type == "subgraphx"
        assert job.status == "pending"
        assert job.tenant_id == "t1"
        assert job.progress == 0.0

    async def test_get_job_missing_returns_none(self):
        mgr = JobManager()
        assert await mgr.get_job("nonexistent") is None

    async def test_get_job_cross_tenant_returns_none(self):
        """IDOR guard: a caller supplying its tenant cannot read another tenant's job."""
        mgr = JobManager()
        job_id = await mgr.create_job("sandbox", tenant_id="tenant-a")
        assert await mgr.get_job(job_id, tenant_id="tenant-b") is None
        # owner still reads it
        job = await mgr.get_job(job_id, tenant_id="tenant-a")
        assert job is not None and job.tenant_id == "tenant-a"

    async def test_get_job_no_tenant_arg_is_unscoped(self):
        """Internal callers (no tenant supplied) keep the existing lookup semantics."""
        mgr = JobManager()
        job_id = await mgr.create_job("sandbox", tenant_id="tenant-a")
        assert await mgr.get_job(job_id) is not None

    async def test_update_job_status(self):
        mgr = JobManager()
        job_id = await mgr.create_job("sandbox")
        await mgr.update_job(job_id, status="running", progress=0.5)
        job = await mgr.get_job(job_id)
        assert job is not None
        assert job.status == "running"
        assert job.progress == 0.5
        assert job.completed_at is None

    async def test_update_job_completed_sets_timestamp(self):
        mgr = JobManager()
        job_id = await mgr.create_job("sandbox")
        await mgr.update_job(job_id, status="completed", result={"output": "ok"})
        job = await mgr.get_job(job_id)
        assert job is not None
        assert job.status == "completed"
        assert job.progress == 1.0
        assert job.completed_at is not None
        assert job.result == {"output": "ok"}

    async def test_update_job_failed_sets_timestamp_and_error(self):
        mgr = JobManager()
        job_id = await mgr.create_job("sandbox")
        await mgr.update_job(job_id, status="failed", error="OOM killed")
        job = await mgr.get_job(job_id)
        assert job is not None
        assert job.status == "failed"
        assert job.error == "OOM killed"
        assert job.completed_at is not None

    async def test_update_job_timeout_sets_timestamp(self):
        mgr = JobManager()
        job_id = await mgr.create_job("sandbox")
        await mgr.update_job(job_id, status="timeout")
        job = await mgr.get_job(job_id)
        assert job is not None
        assert job.status == "timeout"
        assert job.completed_at is not None

    async def test_update_nonexistent_returns_none(self):
        mgr = JobManager()
        assert await mgr.update_job("missing", status="running") is None

    async def test_update_result_url(self):
        mgr = JobManager()
        job_id = await mgr.create_job("sandbox")
        await mgr.update_job(job_id, status="completed", result_url="http://minio/result.json")
        job = await mgr.get_job(job_id)
        assert job is not None
        assert job.result_url == "http://minio/result.json"

    async def test_list_jobs_all(self):
        mgr = JobManager()
        await mgr.create_job("sandbox", tenant_id="t1")
        await mgr.create_job("subgraphx", tenant_id="t2")
        await mgr.create_job("sandbox", tenant_id="t1")
        assert len(await mgr.list_jobs()) == 3

    async def test_list_jobs_by_type(self):
        mgr = JobManager()
        await mgr.create_job("sandbox", tenant_id="t1")
        await mgr.create_job("subgraphx", tenant_id="t2")
        await mgr.create_job("sandbox", tenant_id="t1")
        sandbox_jobs = await mgr.list_jobs(job_type="sandbox")
        assert len(sandbox_jobs) == 2
        assert all(j.job_type == "sandbox" for j in sandbox_jobs)

    async def test_list_jobs_by_tenant(self):
        mgr = JobManager()
        await mgr.create_job("sandbox", tenant_id="t1")
        await mgr.create_job("subgraphx", tenant_id="t2")
        t1_jobs = await mgr.list_jobs(tenant_id="t1")
        assert len(t1_jobs) == 1

    async def test_cleanup_expired(self):
        mgr = JobManager()
        job_id = await mgr.create_job("sandbox")
        # Backdate the job to 25 hours ago
        job = await mgr.get_job(job_id)
        assert job is not None
        job.created_at = datetime.now(UTC) - timedelta(hours=25)
        mgr._jobs[job_id] = job

        fresh_id = await mgr.create_job("sandbox")

        removed = await mgr.cleanup_expired(max_age_hours=24)
        assert removed == 1
        assert await mgr.get_job(job_id) is None
        assert await mgr.get_job(fresh_id) is not None

    async def test_cleanup_no_expired(self):
        mgr = JobManager()
        await mgr.create_job("sandbox")
        removed = await mgr.cleanup_expired(max_age_hours=24)
        assert removed == 0
