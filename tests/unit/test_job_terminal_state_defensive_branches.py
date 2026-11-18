"""Direct coverage for the last-line-of-defence branches of `job_terminal_state`.

`tests/unit/test_job_terminal_state_invariant.py` drives the guard through
`SubgraphXService` and proves the headline invariant: an unenumerated exception
still lands the job in a terminal state.

What it does not reach is the guard's own defensive machinery — the branches that
have to work when the *job store itself* is misbehaving, or when the task is being
torn down rather than failing. Those were entirely unexercised
(`job_manager.py:122-123, 150-152, 162-164, 166-167`), which is the worst place
to have no tests: this code only ever runs once something else has already gone
wrong, so a defect in it is invisible until the exact moment it is needed.

Test-only; no production code changes.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from services.job_manager import JobManager, job_terminal_state

pytestmark = pytest.mark.unit

TERMINAL = ("completed", "failed", "timeout")


@pytest.fixture
def job_manager():
    return JobManager()


async def _new_job(job_manager: JobManager) -> str:
    job_id = await job_manager.create_job("subgraphx", tenant_id="t1")
    await job_manager.update_job(job_id, status="running")
    return job_id


class TestCancellation:
    async def test_cancelled_task_is_marked_failed_and_cancellation_propagates(self, job_manager):
        """Cancellation must settle the job AND still cancel the task.

        `asyncio.CancelledError` derives from `BaseException`, so a bare
        `except Exception` would not catch it at all and the job would strand.
        The guard handles it explicitly: mark failed, then re-raise so shutdown
        semantics are preserved. Swallowing it would be a worse bug than the
        stranding it fixes — the event loop would never learn the task stopped.
        """
        job_id = await _new_job(job_manager)
        started = asyncio.Event()

        async def _body() -> None:
            async with job_terminal_state(job_manager, job_id, label="Cancellable job"):
                started.set()
                await asyncio.sleep(3600)

        task = asyncio.create_task(_body())
        await started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # The cancellation actually took effect on the task ...
        assert task.cancelled()

        # ... and the job did not strand at `running`.
        job = await job_manager.get_job(job_id)
        assert job is not None
        assert job.status == "failed"
        assert job.status in TERMINAL
        assert "cancel" in (job.error or "").lower()


class TestBodyReturnsWithoutSettling:
    async def test_body_that_returns_without_a_terminal_status_is_forced_failed(self, job_manager):
        """An early `return` raises nothing, so the exception handlers never fire.

        The contract is terminal-or-failed, not "no exception means success".
        """
        job_id = await _new_job(job_manager)

        async with job_terminal_state(job_manager, job_id, label="Early-return job"):
            pass  # never sets a terminal status

        job = await job_manager.get_job(job_id)
        assert job is not None
        assert job.status == "failed"
        assert "without reporting a result" in (job.error or "")

    async def test_a_body_that_completed_normally_is_left_alone(self, job_manager):
        """Negative control — the guard must not overwrite a real terminal status."""
        job_id = await _new_job(job_manager)

        async with job_terminal_state(job_manager, job_id, label="Successful job"):
            await job_manager.update_job(job_id, status="completed", result={"ok": True})

        job = await job_manager.get_job(job_id)
        assert job is not None
        assert job.status == "completed"
        assert job.error is None


class TestJobStoreItselfFails:
    async def test_unreadable_job_store_does_not_mask_a_successful_body(self, job_manager):
        """If the post-check read raises, the guard logs and returns rather than
        propagating a brand-new exception out of a body that already succeeded."""
        store = AsyncMock()
        store.get_job = AsyncMock(side_effect=ConnectionError("job store unreachable"))
        store.update_job = AsyncMock()

        async with job_terminal_state(store, "job-1", label="Unreadable-store job"):
            pass

        # It swallowed the read failure and did not invent a status transition.
        store.update_job.assert_not_awaited()

    async def test_unwritable_job_store_never_raises_out_of_the_guard(self):
        """`_force_failed` is the last line of defence — if IT raises, the original
        exception is replaced by a misleading one and the job strands anyway."""
        store = AsyncMock()
        store.get_job = AsyncMock(return_value=None)
        store.update_job = AsyncMock(side_effect=ConnectionError("job store unreachable"))

        # The body's RuntimeError is still swallowed (the job row is meant to be
        # the record), and the store's failure does not surface in its place.
        async with job_terminal_state(store, "job-1", label="Unwritable-store job"):
            raise RuntimeError("body blew up")

        store.update_job.assert_awaited()

    async def test_unwritable_job_store_still_lets_cancellation_propagate(self):
        """Even when the failure write is impossible, cancellation must not be lost."""
        store = AsyncMock()
        store.get_job = AsyncMock(return_value=None)
        store.update_job = AsyncMock(side_effect=ConnectionError("job store unreachable"))
        started = asyncio.Event()

        async def _body() -> None:
            async with job_terminal_state(store, "job-1", label="Doomed job"):
                started.set()
                await asyncio.sleep(3600)

        task = asyncio.create_task(_body())
        await started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
