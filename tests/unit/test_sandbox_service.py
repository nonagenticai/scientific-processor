"""Unit tests for SandboxService -- K8s job creation & polling logic."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import Settings
from services.job_manager import JobManager
from services.sandbox_service import SandboxService

pytestmark = pytest.mark.unit


async def _wait_for_terminal(job_manager: JobManager, job_id: str, timeout: float = 5.0) -> None:
    """Poll the job manager until the job reaches a terminal status (not running/pending)."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        job = await job_manager.get_job(job_id)
        if job and job.status not in ("running", "pending"):
            return
        await asyncio.sleep(0.05)


@pytest.fixture
def job_manager():
    return JobManager()


@pytest.fixture
def minio_client():
    mock = AsyncMock()
    mock.upload = AsyncMock()
    mock.presigned_url = AsyncMock(return_value="https://minio.example/sandbox/abc/output.txt?sig=xxx")
    return mock


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def k8s_settings():
    """Settings with sandbox K8s mode enabled."""
    s = Settings()
    s.sandbox_enabled = True
    return s


@pytest.fixture
def service(job_manager, minio_client, k8s_settings):
    svc = SandboxService(job_manager=job_manager, minio_client=minio_client, settings=k8s_settings)
    svc._k8s_loaded = True  # Skip real k8s config loading
    return svc


def _make_k8s_job_status(succeeded=None, failed=None, conditions=None):
    """Build a mock K8s Job object with the given status fields."""
    status = MagicMock()
    status.succeeded = succeeded
    status.failed = failed
    status.conditions = conditions
    job = MagicMock()
    job.status = status
    return job


def _make_pod_list(pod_name="sandbox-pod-abc"):
    """Build a mock pod list with one pod."""
    pod = MagicMock()
    pod.metadata.name = pod_name
    pod_list = MagicMock()
    pod_list.items = [pod]
    return pod_list


class TestSandboxServiceExecute:
    async def test_execute_creates_job_and_returns_id(self, service, job_manager):
        """execute() should create a job, create K8s resources, and return job_id."""
        mock_core_v1 = AsyncMock()
        mock_batch_v1 = AsyncMock()

        mock_api_ctx = AsyncMock()
        mock_api_ctx.__aenter__ = AsyncMock(return_value=mock_api_ctx)
        mock_api_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("kubernetes_asyncio.client.ApiClient", return_value=mock_api_ctx),
            patch("kubernetes_asyncio.client.CoreV1Api", return_value=mock_core_v1),
            patch("kubernetes_asyncio.client.BatchV1Api", return_value=mock_batch_v1),
            patch("asyncio.create_task") as mock_create_task,
        ):
            job_id = await service.execute(
                code="print('hello')",
                language="python",
                timeout_seconds=60,
                input_data=None,
                tenant_id="test-tenant",
            )

        assert isinstance(job_id, str)
        assert len(job_id) == 36  # UUID

        # Job should exist in manager
        job = await job_manager.get_job(job_id)
        assert job is not None
        assert job.status == "running"
        assert job.tenant_id == "test-tenant"

        # K8s resources should have been created
        mock_core_v1.create_namespaced_config_map.assert_awaited_once()
        mock_batch_v1.create_namespaced_job.assert_awaited_once()

        # Background poll task should have been started
        mock_create_task.assert_called_once()

    async def test_execute_with_input_data(self, service):
        """execute() with input_data includes input.json in ConfigMap."""
        mock_core_v1 = AsyncMock()
        mock_batch_v1 = AsyncMock()

        mock_api_ctx = AsyncMock()
        mock_api_ctx.__aenter__ = AsyncMock(return_value=mock_api_ctx)
        mock_api_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("kubernetes_asyncio.client.ApiClient", return_value=mock_api_ctx),
            patch("kubernetes_asyncio.client.CoreV1Api", return_value=mock_core_v1),
            patch("kubernetes_asyncio.client.BatchV1Api", return_value=mock_batch_v1),
            patch("asyncio.create_task"),
        ):
            await service.execute(
                code="import json; print(json.load(open('input.json')))",
                language="python",
                timeout_seconds=30,
                input_data={"key": "value"},
                tenant_id=None,
            )

        # Verify ConfigMap included input.json
        call_args = mock_core_v1.create_namespaced_config_map.call_args
        configmap_body = call_args.kwargs["body"]
        assert "input.json" in configmap_body.data
        assert '"key"' in configmap_body.data["input.json"]


class TestSandboxServicePoll:
    async def test_poll_success_completes_job(self, service, job_manager, minio_client):
        """When K8s Job succeeds, the internal job should be marked completed."""
        job_id = await job_manager.create_job("sandbox", tenant_id="t1")
        await job_manager.update_job(job_id, status="running")

        k8s_job_running = _make_k8s_job_status(succeeded=None, failed=None)
        k8s_job_done = _make_k8s_job_status(succeeded=1, failed=None)
        pod_list = _make_pod_list()

        mock_batch_v1 = AsyncMock()
        mock_batch_v1.read_namespaced_job = AsyncMock(side_effect=[k8s_job_running, k8s_job_done])

        mock_core_v1 = AsyncMock()
        mock_core_v1.list_namespaced_pod = AsyncMock(return_value=pod_list)
        mock_core_v1.read_namespaced_pod_log = AsyncMock(return_value="result: 42\n")
        mock_core_v1.delete_namespaced_config_map = AsyncMock()

        mock_api_ctx = AsyncMock()
        mock_api_ctx.__aenter__ = AsyncMock(return_value=mock_api_ctx)
        mock_api_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("kubernetes_asyncio.client.ApiClient", return_value=mock_api_ctx),
            patch("kubernetes_asyncio.client.CoreV1Api", return_value=mock_core_v1),
            patch("kubernetes_asyncio.client.BatchV1Api", return_value=mock_batch_v1),
            patch("services.sandbox_service.asyncio.sleep", new_callable=AsyncMock),
        ):
            await service._poll_job(job_id, f"sandbox-{job_id}", f"sandbox-code-{job_id}")

        job = await job_manager.get_job(job_id)
        assert job.status == "completed"
        assert job.completed_at is not None
        assert job.result is not None
        assert "result: 42" in job.result["logs"]
        # execution_time_ms must be PERSISTED into the result dict, not mutated
        # onto a detached JobStatus after update_job (regression guard: get_job
        # returns a fresh detached copy under PersistentJobManager).
        assert "execution_time_ms" in job.result
        assert isinstance(job.result["execution_time_ms"], (int, float))
        assert job.result["execution_time_ms"] >= 0

        # MinIO upload should have been called
        minio_client.upload.assert_awaited_once()

    async def test_poll_failure_marks_job_failed(self, service, job_manager):
        """When K8s Job fails, the internal job should be marked failed."""
        job_id = await job_manager.create_job("sandbox", tenant_id="t1")
        await job_manager.update_job(job_id, status="running")

        k8s_job_failed = _make_k8s_job_status(succeeded=None, failed=1, conditions=None)
        pod_list = _make_pod_list()

        mock_batch_v1 = AsyncMock()
        mock_batch_v1.read_namespaced_job = AsyncMock(return_value=k8s_job_failed)

        mock_core_v1 = AsyncMock()
        mock_core_v1.list_namespaced_pod = AsyncMock(return_value=pod_list)
        mock_core_v1.read_namespaced_pod_log = AsyncMock(return_value="Traceback: ZeroDivisionError\n")
        mock_core_v1.delete_namespaced_config_map = AsyncMock()

        mock_api_ctx = AsyncMock()
        mock_api_ctx.__aenter__ = AsyncMock(return_value=mock_api_ctx)
        mock_api_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("kubernetes_asyncio.client.ApiClient", return_value=mock_api_ctx),
            patch("kubernetes_asyncio.client.CoreV1Api", return_value=mock_core_v1),
            patch("kubernetes_asyncio.client.BatchV1Api", return_value=mock_batch_v1),
            patch("services.sandbox_service.asyncio.sleep", new_callable=AsyncMock),
        ):
            await service._poll_job(job_id, f"sandbox-{job_id}", f"sandbox-code-{job_id}")

        job = await job_manager.get_job(job_id)
        assert job.status == "failed"
        assert "ZeroDivisionError" in job.error
        assert job.completed_at is not None

    async def test_poll_timeout_marks_job_timeout(self, service, job_manager):
        """When K8s Job exceeds activeDeadlineSeconds, job should be marked timeout."""
        job_id = await job_manager.create_job("sandbox", tenant_id="t1")
        await job_manager.update_job(job_id, status="running")

        condition = MagicMock()
        condition.type = "Failed"
        condition.reason = "DeadlineExceeded"

        k8s_job_timeout = _make_k8s_job_status(
            succeeded=None,
            failed=1,
            conditions=[condition],
        )
        pod_list = _make_pod_list()

        mock_batch_v1 = AsyncMock()
        mock_batch_v1.read_namespaced_job = AsyncMock(return_value=k8s_job_timeout)

        mock_core_v1 = AsyncMock()
        mock_core_v1.list_namespaced_pod = AsyncMock(return_value=pod_list)
        mock_core_v1.read_namespaced_pod_log = AsyncMock(return_value="")
        mock_core_v1.delete_namespaced_config_map = AsyncMock()

        mock_api_ctx = AsyncMock()
        mock_api_ctx.__aenter__ = AsyncMock(return_value=mock_api_ctx)
        mock_api_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("kubernetes_asyncio.client.ApiClient", return_value=mock_api_ctx),
            patch("kubernetes_asyncio.client.CoreV1Api", return_value=mock_core_v1),
            patch("kubernetes_asyncio.client.BatchV1Api", return_value=mock_batch_v1),
            patch("services.sandbox_service.asyncio.sleep", new_callable=AsyncMock),
        ):
            await service._poll_job(job_id, f"sandbox-{job_id}", f"sandbox-code-{job_id}")

        job = await job_manager.get_job(job_id)
        assert job.status == "timeout"
        assert "deadline" in job.error.lower()

    async def test_poll_crash_marks_job_failed(self, service, job_manager):
        """If the poll task itself crashes, the job should be marked failed."""
        job_id = await job_manager.create_job("sandbox", tenant_id="t1")
        await job_manager.update_job(job_id, status="running")

        mock_api_ctx = AsyncMock()
        mock_api_ctx.__aenter__ = AsyncMock(side_effect=OSError("k8s unreachable"))
        mock_api_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("kubernetes_asyncio.client.ApiClient", return_value=mock_api_ctx),
        ):
            await service._poll_job(job_id, "sandbox-x", "sandbox-code-x")

        job = await job_manager.get_job(job_id)
        assert job.status == "failed"
        assert "poll error" in job.error.lower()

    async def test_poll_no_minio_skips_upload(self, service, job_manager):
        """When minio_client is None, results upload is skipped gracefully."""
        service.minio_client = None
        job_id = await job_manager.create_job("sandbox", tenant_id="t1")
        await job_manager.update_job(job_id, status="running")

        k8s_job_done = _make_k8s_job_status(succeeded=1, failed=None)
        pod_list = _make_pod_list()

        mock_batch_v1 = AsyncMock()
        mock_batch_v1.read_namespaced_job = AsyncMock(return_value=k8s_job_done)

        mock_core_v1 = AsyncMock()
        mock_core_v1.list_namespaced_pod = AsyncMock(return_value=pod_list)
        mock_core_v1.read_namespaced_pod_log = AsyncMock(return_value="output line")
        mock_core_v1.delete_namespaced_config_map = AsyncMock()

        mock_api_ctx = AsyncMock()
        mock_api_ctx.__aenter__ = AsyncMock(return_value=mock_api_ctx)
        mock_api_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("kubernetes_asyncio.client.ApiClient", return_value=mock_api_ctx),
            patch("kubernetes_asyncio.client.CoreV1Api", return_value=mock_core_v1),
            patch("kubernetes_asyncio.client.BatchV1Api", return_value=mock_batch_v1),
            patch("services.sandbox_service.asyncio.sleep", new_callable=AsyncMock),
        ):
            await service._poll_job(job_id, f"sandbox-{job_id}", f"sandbox-code-{job_id}")

        job = await job_manager.get_job(job_id)
        assert job.status == "completed"
        assert job.result_url is None  # No MinIO, no URL


class TestSandboxServiceGetStatus:
    async def test_get_status_returns_job(self, service, job_manager):
        job_id = await job_manager.create_job("sandbox", tenant_id="t1")
        result = await service.get_status(job_id)
        assert result is not None
        assert result.job_id == job_id

    async def test_get_status_returns_none_for_missing(self, service):
        result = await service.get_status("nonexistent-id")
        assert result is None


class TestSandboxServiceGetResults:
    async def test_get_results_returns_presigned_url(self, service, job_manager, minio_client):
        job_id = await job_manager.create_job("sandbox", tenant_id="t1")
        await job_manager.update_job(job_id, status="completed", result_url="sandbox/abc/output.txt")

        url = await service.get_results(job_id)
        assert url is not None
        assert "minio.example" in url
        minio_client.presigned_url.assert_awaited_once_with("sandbox/abc/output.txt")

    async def test_get_results_returns_none_for_pending_job(self, service, job_manager):
        job_id = await job_manager.create_job("sandbox", tenant_id="t1")
        # Job is still pending, no result_url
        url = await service.get_results(job_id)
        assert url is None

    async def test_get_results_returns_none_for_missing_job(self, service):
        url = await service.get_results("nonexistent")
        assert url is None

    async def test_get_results_returns_none_for_completed_no_url(self, service, job_manager):
        job_id = await job_manager.create_job("sandbox", tenant_id="t1")
        await job_manager.update_job(job_id, status="completed")
        # completed but no result_url
        url = await service.get_results(job_id)
        assert url is None

    async def test_get_results_without_minio_returns_raw_url(self, service, job_manager):
        service.minio_client = None
        job_id = await job_manager.create_job("sandbox", tenant_id="t1")
        await job_manager.update_job(job_id, status="completed", result_url="sandbox/abc/output.txt")

        url = await service.get_results(job_id)
        assert url == "sandbox/abc/output.txt"


# ---------------------------------------------------------------------------
# SANDBOX_ENABLED toggle
# ---------------------------------------------------------------------------


class TestSandboxEnabledToggle:
    async def test_sandbox_enabled_false_by_default(self, settings, job_manager):
        svc = SandboxService(job_manager=job_manager, minio_client=None, settings=settings)
        assert svc.sandbox_enabled is False

    async def test_sandbox_enabled_true(self, k8s_settings, job_manager):
        svc = SandboxService(job_manager=job_manager, minio_client=None, settings=k8s_settings)
        assert svc.sandbox_enabled is True


# ---------------------------------------------------------------------------
# Local fallback tests (SANDBOX_ENABLED=false)
# ---------------------------------------------------------------------------


class TestLocalExecution:
    @pytest.fixture
    def local_service(self, job_manager, minio_client, settings):
        """Service in local dev mode: K8s sandbox off, in-process opt-in ON."""
        settings.sandbox_allow_local = True
        return SandboxService(job_manager=job_manager, minio_client=minio_client, settings=settings)

    async def test_local_execute_refused_without_optin(self, job_manager, minio_client, settings):
        """Without the explicit dev opt-in, in-process execution must refuse (fail closed)."""
        settings.sandbox_allow_local = False  # default
        svc = SandboxService(job_manager=job_manager, minio_client=minio_client, settings=settings)
        with pytest.raises(RuntimeError, match="disabled"):
            await svc.execute(
                code="import os; print(os.environ)",
                language="python",
                timeout_seconds=10,
                input_data=None,
                tenant_id="t1",
            )

    async def test_local_execute_success(self, local_service, job_manager):
        """Local execution of valid Python code should complete successfully."""
        job_id = await local_service.execute(
            code="print('hello world')",
            language="python",
            timeout_seconds=10,
            input_data=None,
            tenant_id="t1",
        )

        assert isinstance(job_id, str)

        await _wait_for_terminal(job_manager, job_id)

        job = await job_manager.get_job(job_id)
        assert job is not None
        assert job.status == "completed"
        assert "hello world" in job.result["logs"]
        assert job.result["execution_time_ms"] > 0

    async def test_local_execute_with_input_data(self, local_service, job_manager):
        """Local execution should make INPUT_DATA available to the code."""
        job_id = await local_service.execute(
            code="print(INPUT_DATA['key'])",
            language="python",
            timeout_seconds=10,
            input_data={"key": "test_value"},
            tenant_id="t1",
        )

        await _wait_for_terminal(job_manager, job_id)

        job = await job_manager.get_job(job_id)
        assert job.status == "completed"
        assert "test_value" in job.result["logs"]

    async def test_local_execute_failure(self, local_service, job_manager):
        """Code that raises an exception should mark the job as failed."""
        job_id = await local_service.execute(
            code="raise ValueError('boom')",
            language="python",
            timeout_seconds=10,
            input_data=None,
            tenant_id="t1",
        )

        await _wait_for_terminal(job_manager, job_id)

        job = await job_manager.get_job(job_id)
        assert job.status == "failed"
        assert "ValueError" in job.error

    async def test_local_execute_timeout(self, local_service, job_manager):
        """Code that exceeds the timeout should be killed and marked as timeout."""
        job_id = await local_service.execute(
            code="import time; time.sleep(30)",
            language="python",
            timeout_seconds=1,
            input_data=None,
            tenant_id="t1",
        )

        await _wait_for_terminal(job_manager, job_id, timeout=5.0)

        job = await job_manager.get_job(job_id)
        assert job.status == "timeout"
        assert "deadline" in job.error.lower()

    async def test_local_execute_syntax_error(self, local_service, job_manager):
        """Code with a syntax error should fail."""
        job_id = await local_service.execute(
            code="def f(\n",
            language="python",
            timeout_seconds=10,
            input_data=None,
            tenant_id="t1",
        )

        await _wait_for_terminal(job_manager, job_id)

        job = await job_manager.get_job(job_id)
        assert job.status == "failed"
        assert "SyntaxError" in job.error


class TestBackgroundTaskRetention:
    """Recovered from the stranded PR #51.

    ``sandbox_service._spawn`` landed via #39/#40 but nothing on main asserted the
    strong reference is (a) taken and (b) released. The event loop holds only a weak
    reference to a task, so a bare ``asyncio.create_task(...)`` can be garbage
    collected mid-flight and silently cancelled — orphaning the K8s Job + ConfigMap
    and pinning the sandbox job at "running" forever.
    """

    async def test_local_execution_holds_a_strong_reference_until_done(self, job_manager, minio_client):
        from services import sandbox_service as module

        s = Settings()
        s.sandbox_allow_local = True
        svc = SandboxService(job_manager=job_manager, minio_client=minio_client, settings=s)

        before = set(module._background_tasks)
        job_id = await svc.execute(
            code="print('hi')",
            language="python",
            timeout_seconds=10,
            input_data=None,
            tenant_id="t1",
        )
        new_tasks = set(module._background_tasks) - before
        assert len(new_tasks) == 1, "the _run_local task was not retained — it can be GC'd mid-flight"

        task = new_tasks.pop()
        await task
        assert task not in module._background_tasks, "the done callback must release the reference"

        job = await job_manager.get_job(job_id)
        assert job.status == "completed"

    async def test_k8s_poll_task_is_retained_then_released(self, service):
        """The K8s path spawns ``_poll_job``; the same retention rule applies.

        ``test_execute_creates_job_and_returns_id`` patches ``asyncio.create_task``
        away entirely, so it proves a task was CREATED but says nothing about whether
        anything holds it — which is the whole defect.
        """
        from services import sandbox_service as module

        mock_core_v1 = AsyncMock()
        mock_batch_v1 = AsyncMock()
        mock_api_ctx = AsyncMock()
        mock_api_ctx.__aenter__ = AsyncMock(return_value=mock_api_ctx)
        mock_api_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("kubernetes_asyncio.client.ApiClient", return_value=mock_api_ctx),
            patch("kubernetes_asyncio.client.CoreV1Api", return_value=mock_core_v1),
            patch("kubernetes_asyncio.client.BatchV1Api", return_value=mock_batch_v1),
            patch.object(SandboxService, "_poll_job", new=AsyncMock()) as poll,
        ):
            before = set(module._background_tasks)
            await service.execute(
                code="print('hi')",
                language="python",
                timeout_seconds=10,
                input_data=None,
                tenant_id="t1",
            )
            new_tasks = set(module._background_tasks) - before
            assert len(new_tasks) == 1, "the _poll_job task was not retained — it can be GC'd mid-flight"

            task = new_tasks.pop()
            await task
            assert task not in module._background_tasks, "the done callback must release the reference"
            poll.assert_awaited_once()
