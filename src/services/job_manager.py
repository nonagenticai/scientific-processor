"""In-memory async job manager for long-running tasks."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Protocol

from log_config import get_logger
from models.jobs import JobStatus, JobStatusType, JobType

logger = get_logger(__name__)

#: A job in one of these states is finished; anything else is still in flight.
TERMINAL_STATUSES: Final[frozenset[str]] = frozenset({"completed", "failed", "timeout"})


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}

    async def create_job(
        self, job_type: JobType, tenant_id: str | None = None, params: dict[str, Any] | None = None
    ) -> str:
        job_id = str(uuid.uuid7())
        job = JobStatus(
            job_id=job_id,
            job_type=job_type,
            status="pending",
            created_at=datetime.now(UTC),
            tenant_id=tenant_id,
        )
        self._jobs[job_id] = job
        logger.info("Job created", job_id=job_id, job_type=job_type)
        return job_id

    async def get_job(self, job_id: str, tenant_id: str | None = None) -> JobStatus | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        # Fail-closed cross-tenant guard: a caller that supplies its tenant can
        # only read its own jobs (job ids are enumerable uuid7).
        if tenant_id is not None and job.tenant_id != tenant_id:
            return None
        return job

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
        job = self._jobs.get(job_id)
        if job is None:
            return None

        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = progress
        if result is not None:
            job.result = result
        if result_url is not None:
            job.result_url = result_url
        if error is not None:
            job.error = error

        if status in ("completed", "failed", "timeout"):
            job.completed_at = datetime.now(UTC)
            if status == "completed":
                job.progress = 1.0

        self._jobs[job_id] = job
        return job

    async def list_jobs(self, job_type: JobType | None = None, tenant_id: str | None = None) -> list[JobStatus]:
        results = list(self._jobs.values())
        if job_type:
            results = [j for j in results if j.job_type == job_type]
        if tenant_id:
            results = [j for j in results if j.tenant_id == tenant_id]
        return results

    async def cleanup_expired(self, max_age_hours: int = 24) -> int:
        cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
        expired = [jid for jid, j in self._jobs.items() if j.created_at < cutoff]
        for jid in expired:
            del self._jobs[jid]
        if expired:
            logger.info("Cleaned up expired jobs", count=len(expired))
        return len(expired)


class _JobStore(Protocol):
    """The slice of JobManager the terminal-state guard needs."""

    async def get_job(self, job_id: str, tenant_id: str | None = None) -> JobStatus | None: ...

    async def update_job(
        self,
        job_id: str,
        *,
        status: JobStatusType | None = None,
        progress: float | None = None,
        result: dict[str, Any] | None = None,
        result_url: str | None = None,
        error: str | None = None,
    ) -> JobStatus | None: ...


async def _force_failed(job_store: _JobStore, job_id: str, error: str) -> None:
    """Best-effort transition to ``failed``. Never raises."""
    try:
        await job_store.update_job(job_id, status="failed", error=error[:4000])
    except Exception:  # noqa: BLE001 - the guard is the last line of defence
        logger.exception("Could not mark job failed — it may be stranded", job_id=job_id)


@asynccontextmanager
async def job_terminal_state(job_store: _JobStore, job_id: str, *, label: str) -> AsyncIterator[None]:
    """Guarantee a job never remains non-terminal once its task has ended.

    Enumerating exception types in a fire-and-forget task's ``except`` clause is a
    losing game. ``IndexError`` is a ``LookupError``, not a ``KeyError``; a tuple
    that forgets it lets the task die silently, and the job row stays ``running``
    with nothing left in the process that will ever move it. Adding the missing
    type only defers the problem to the next unlisted one.

    So this enforces the *invariant* instead of the taxonomy:

    * any ``Exception`` escaping the body marks the job ``failed`` (recording the
      exception type and message) and is swallowed — the job row is now the record;
    * ``asyncio.CancelledError`` marks the job ``failed`` and is re-raised, so
      shutdown/cancellation semantics are preserved;
    * a body that returns *without* reaching a terminal status is closed out as
      ``failed`` rather than left dangling.

    Use it to wrap the whole body of every background job task.
    """
    try:
        yield
    except asyncio.CancelledError:
        logger.warning("%s cancelled — marking job failed", label, job_id=job_id)
        await _force_failed(job_store, job_id, f"{label} was cancelled before completion")
        raise
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all; see docstring
        logger.exception("%s failed", label, job_id=job_id)
        await _force_failed(job_store, job_id, f"{type(exc).__name__}: {exc}")
        return

    # No exception, but the body may still have returned early without settling
    # the job. The contract is terminal-or-failed, so close it out.
    try:
        job = await job_store.get_job(job_id)
    except Exception:  # noqa: BLE001
        logger.exception("Could not read job to verify terminal state", job_id=job_id)
        return
    if job is not None and job.status not in TERMINAL_STATUSES:
        logger.error("%s ended without settling its job — forcing failed", label, job_id=job_id)
        await _force_failed(job_store, job_id, f"{label} ended without reporting a result")
