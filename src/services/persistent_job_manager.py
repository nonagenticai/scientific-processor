"""PostgreSQL-backed job manager -- replaces in-memory store for production persistence."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from database import JobRow
from log_config import get_logger
from models.jobs import JobStatus, JobStatusType, JobType
from services.job_manager import JobManager

logger = get_logger(__name__)


class PersistentJobManager(JobManager):
    """Durable job tracking via PostgreSQL."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        super().__init__()
        self._session_factory = session_factory

    async def create_job(
        self, job_type: JobType, tenant_id: str | None = None, params: dict[str, Any] | None = None
    ) -> str:
        job_id = str(uuid.uuid7())
        async with self._session_factory() as session:
            row = JobRow(
                id=job_id,
                job_type=job_type,
                status="pending",
                tenant_id=tenant_id,
                params=json.dumps(params) if params else None,
                progress=0.0,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            session.add(row)
            await session.commit()
        logger.info("Job created (persistent)", job_id=job_id, job_type=job_type)
        return job_id

    async def get_job(self, job_id: str, tenant_id: str | None = None) -> JobStatus | None:
        async with self._session_factory() as session:
            row = await session.get(JobRow, job_id)
            if row is None:
                return None
            # Fail-closed cross-tenant guard (independent of RLS mode): job ids
            # are enumerable uuid7, so a caller may only read its own tenant's job.
            if tenant_id is not None and row.tenant_id != tenant_id:
                return None
            return self._row_to_status(row)

    async def update_job(
        self,
        job_id: str,
        *,
        status: JobStatusType | None = None,
        progress: float | None = None,
        result: dict[str, Any] | None = None,
        result_url: str | None = None,
        error: str | None = None,
    ) -> JobStatus | None:
        async with self._session_factory() as session:
            row = await session.get(JobRow, job_id)
            if row is None:
                return None

            if status is not None:
                row.status = status
            if progress is not None:
                row.progress = progress
            if result is not None:
                # Store result as JSON in params column
                row.params = json.dumps(result)
            if result_url is not None:
                row.result_url = result_url
            if error is not None:
                row.error = error

            if status in ("completed", "failed", "timeout"):
                row.completed_at = datetime.now(UTC)
                if status == "completed":
                    row.progress = 1.0

            row.updated_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(row)
            return self._row_to_status(row)

    async def list_jobs(self, job_type: JobType | None = None, tenant_id: str | None = None) -> list[JobStatus]:
        async with self._session_factory() as session:
            stmt = select(JobRow)
            if job_type:
                stmt = stmt.where(JobRow.job_type == job_type)
            if tenant_id:
                stmt = stmt.where(JobRow.tenant_id == tenant_id)
            db_result = await session.execute(stmt)
            return [self._row_to_status(row) for row in db_result.scalars().all()]

    async def cleanup_expired(self, max_age_hours: int = 24) -> int:
        cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
        async with self._session_factory() as session:
            stmt = delete(JobRow).where(JobRow.created_at < cutoff)
            db_result = await session.execute(stmt)
            await session.commit()
            count: int = getattr(db_result, "rowcount", 0) or 0
        if count:
            logger.info("Cleaned up expired jobs (persistent)", count=count)
        return count

    @staticmethod
    def _row_to_status(row: JobRow) -> JobStatus:
        # Parse result from params JSON if present
        result_data = None
        if row.params:
            try:
                parsed = json.loads(row.params) if isinstance(row.params, str) else row.params
                # Only treat as result if it looks like one (dict with non-param keys)
                if isinstance(parsed, dict):
                    result_data = parsed
            except json.JSONDecodeError, TypeError:
                pass

        return JobStatus(
            job_id=row.id,
            job_type=cast(JobType, row.job_type),
            status=cast(JobStatusType, row.status),
            progress=row.progress or 0.0,
            created_at=row.created_at,
            completed_at=row.completed_at,
            result=result_data,
            result_url=row.result_url,
            error=row.error,
            tenant_id=row.tenant_id,
        )
