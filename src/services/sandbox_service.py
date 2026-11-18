"""Sandboxed code execution via Kubernetes Jobs (or local fallback)."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any

from botocore.exceptions import ClientError

from config import Settings
from log_config import get_logger
from models.jobs import JobStatus
from services.job_manager import JobManager, job_terminal_state

logger = get_logger(__name__)

NAMESPACE = "scientific-processor"
POLL_INTERVAL_SECONDS = 5

# Retain references to fire-and-forget background tasks so the event loop cannot
# garbage-collect them mid-flight.
_background_tasks: set[asyncio.Task[Any]] = set()


def _spawn(coro: Any) -> asyncio.Task[Any]:
    """Schedule a background task and keep a strong reference until it finishes."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


class SandboxService:
    def __init__(
        self,
        job_manager: JobManager,
        minio_client: Any | None,
        settings: Settings,
    ) -> None:
        self.job_manager = job_manager
        self.minio_client = minio_client
        self.settings = settings
        self._k8s_loaded = False

    @property
    def sandbox_enabled(self) -> bool:
        return self.settings.sandbox_enabled

    async def _ensure_k8s_config(self) -> None:
        if not self._k8s_loaded:
            from kubernetes_asyncio import config as k8s_config

            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                await k8s_config.load_kube_config()
            self._k8s_loaded = True

    async def execute(
        self,
        code: str,
        language: str,
        timeout_seconds: int,
        input_data: dict[str, Any] | None,
        tenant_id: str | None,
    ) -> str:
        """Submit code for sandboxed execution. Returns job_id."""
        if self.sandbox_enabled:
            return await self._execute_k8s(code, language, timeout_seconds, input_data, tenant_id)
        return await self._execute_local(code, language, timeout_seconds, input_data, tenant_id)

    # -----------------------------------------------------------------------
    # K8s execution path
    # -----------------------------------------------------------------------

    async def _execute_k8s(
        self,
        code: str,
        language: str,
        timeout_seconds: int,
        input_data: dict[str, Any] | None,
        tenant_id: str | None,
    ) -> str:
        from kubernetes_asyncio import client as k8s_client

        job_id = await self.job_manager.create_job("sandbox", tenant_id=tenant_id)
        await self.job_manager.update_job(job_id, status="running")

        await self._ensure_k8s_config()

        configmap_name = f"sandbox-code-{job_id}"
        k8s_job_name = f"sandbox-{job_id}"

        # Build ConfigMap data
        cm_data: dict[str, str] = {"code.py": code}
        if input_data is not None:
            cm_data["input.json"] = json.dumps(input_data)

        configmap = k8s_client.V1ConfigMap(
            metadata=k8s_client.V1ObjectMeta(
                name=configmap_name,
                namespace=NAMESPACE,
                labels={"app": "scientific-sandbox", "job-id": job_id},
            ),
            data=cm_data,
        )

        # Build K8s Job
        container = k8s_client.V1Container(
            name="sandbox",
            image=self.settings.sandbox_image,
            resources=k8s_client.V1ResourceRequirements(
                limits={
                    "cpu": self.settings.sandbox_cpu_limit,
                    "memory": self.settings.sandbox_memory_limit,
                },
                requests={
                    "cpu": "100m",
                    "memory": "64Mi",
                },
            ),
            volume_mounts=[
                k8s_client.V1VolumeMount(
                    name="code-volume",
                    mount_path="/workspace",
                    read_only=True,
                ),
            ],
        )

        pod_spec = k8s_client.V1PodSpec(
            containers=[container],
            restart_policy="Never",
            volumes=[
                k8s_client.V1Volume(
                    name="code-volume",
                    config_map=k8s_client.V1ConfigMapVolumeSource(name=configmap_name),
                ),
            ],
        )

        job_spec = k8s_client.V1JobSpec(
            template=k8s_client.V1PodTemplateSpec(
                metadata=k8s_client.V1ObjectMeta(
                    labels={"app": "scientific-sandbox", "job-id": job_id},
                ),
                spec=pod_spec,
            ),
            active_deadline_seconds=timeout_seconds,
            backoff_limit=0,
            ttl_seconds_after_finished=300,
        )

        k8s_job = k8s_client.V1Job(
            metadata=k8s_client.V1ObjectMeta(
                name=k8s_job_name,
                namespace=NAMESPACE,
                labels={"app": "scientific-sandbox", "job-id": job_id},
            ),
            spec=job_spec,
        )

        # Create ConfigMap and Job
        async with k8s_client.ApiClient() as api:
            core_v1 = k8s_client.CoreV1Api(api)
            batch_v1 = k8s_client.BatchV1Api(api)

            await core_v1.create_namespaced_config_map(namespace=NAMESPACE, body=configmap)
            await batch_v1.create_namespaced_job(namespace=NAMESPACE, body=k8s_job)

        logger.info("Sandbox job created", job_id=job_id, k8s_job=k8s_job_name, timeout=timeout_seconds)

        # Start background poll task
        _spawn(self._poll_job(job_id, k8s_job_name, configmap_name))

        return job_id

    async def _poll_job(self, job_id: str, k8s_job_name: str, configmap_name: str) -> None:
        """Poll K8s Job status until completion, failure, or timeout.

        Wrapped in ``job_terminal_state`` so an exception type the inner handler
        does not list cannot strand the job at ``running`` (see #39/#42).
        """
        from kubernetes_asyncio import client as k8s_client

        start_time = time.monotonic()

        async with job_terminal_state(self.job_manager, job_id, label="Sandbox poll"):
            try:
                async with k8s_client.ApiClient() as api:
                    batch_v1 = k8s_client.BatchV1Api(api)
                    core_v1 = k8s_client.CoreV1Api(api)

                    while True:
                        await asyncio.sleep(POLL_INTERVAL_SECONDS)

                        try:
                            k8s_job = await batch_v1.read_namespaced_job(name=k8s_job_name, namespace=NAMESPACE)
                        except k8s_client.exceptions.ApiException:
                            logger.exception("Failed to read K8s job", job_id=job_id)
                            continue

                        status = k8s_job.status

                        # Check for completion
                        if status.succeeded and status.succeeded > 0:
                            elapsed_ms = (time.monotonic() - start_time) * 1000
                            logs = await self._read_pod_logs(core_v1, job_id)
                            result_url = await self._upload_results(job_id, logs)
                            # Fold execution_time_ms into the persisted result dict.
                            # Under PersistentJobManager, get_job() returns a fresh
                            # detached JobStatus, so mutating it post-update never
                            # persisted (GET /sandbox/jobs/{id} reported null).
                            await self.job_manager.update_job(
                                job_id,
                                status="completed",
                                result={"logs": logs[:10000], "execution_time_ms": elapsed_ms},
                                result_url=result_url,
                            )
                            logger.info("Sandbox job completed", job_id=job_id, elapsed_ms=elapsed_ms)
                            break

                        # Check for failure
                        if status.failed and status.failed > 0:
                            elapsed_ms = (time.monotonic() - start_time) * 1000
                            logs = await self._read_pod_logs(core_v1, job_id)

                            # Detect timeout via conditions
                            is_timeout = False
                            if status.conditions:
                                for cond in status.conditions:
                                    if cond.type == "Failed" and cond.reason == "DeadlineExceeded":
                                        is_timeout = True
                                        break

                            if is_timeout:
                                await self.job_manager.update_job(
                                    job_id,
                                    status="timeout",
                                    error="Execution exceeded deadline",
                                )
                                logger.warning("Sandbox job timed out", job_id=job_id)
                            else:
                                error_msg = logs[-2000:] if logs else "Unknown error"
                                await self.job_manager.update_job(
                                    job_id,
                                    status="failed",
                                    error=error_msg,
                                )
                                logger.warning("Sandbox job failed", job_id=job_id)
                            break

                    # Cleanup ConfigMap
                    try:
                        await core_v1.delete_namespaced_config_map(name=configmap_name, namespace=NAMESPACE)
                    except k8s_client.exceptions.ApiException:
                        logger.debug("ConfigMap cleanup failed (may already be deleted)", configmap=configmap_name)

            except k8s_client.exceptions.ApiException, OSError:
                logger.exception("Sandbox poll task crashed", job_id=job_id)
                await self.job_manager.update_job(job_id, status="failed", error="Internal poll error")

    async def _read_pod_logs(self, core_v1: Any, job_id: str) -> str:
        """Read logs from the sandbox pod."""
        from kubernetes_asyncio.client.exceptions import ApiException

        try:
            pods = await core_v1.list_namespaced_pod(
                namespace=NAMESPACE,
                label_selector=f"job-id={job_id}",
            )
            if pods.items:
                logs = await core_v1.read_namespaced_pod_log(
                    name=pods.items[0].metadata.name,
                    namespace=NAMESPACE,
                )
                return logs or ""
        except ApiException:
            logger.debug("Failed to read pod logs", job_id=job_id)
        return ""

    # -----------------------------------------------------------------------
    # Local execution fallback (SANDBOX_ENABLED=false)
    # -----------------------------------------------------------------------

    async def _execute_local(
        self,
        code: str,
        language: str,
        timeout_seconds: int,
        input_data: dict[str, Any] | None,
        tenant_id: str | None,
    ) -> str:
        """Run code locally in a subprocess. Used when K8s sandbox is disabled."""
        # Fail closed: in-process execution runs caller code in this container
        # with all its secrets. Require an explicit dev opt-in so a production
        # misconfiguration (sandbox_enabled unset) errors instead of silently
        # granting arbitrary code execution.
        if not self.settings.sandbox_allow_local:
            raise RuntimeError(
                "In-process sandbox execution is disabled. Enable the K8s sandbox "
                "(SCIENTIFIC_PROCESSOR_SANDBOX_ENABLED=true) or, for local dev only, "
                "set SCIENTIFIC_PROCESSOR_SANDBOX_ALLOW_LOCAL=true."
            )
        job_id = await self.job_manager.create_job("sandbox", tenant_id=tenant_id)
        await self.job_manager.update_job(job_id, status="running")

        # Fire-and-forget background task
        _spawn(self._run_local(job_id, code, timeout_seconds, input_data))
        return job_id

    async def _run_local(
        self,
        job_id: str,
        code: str,
        timeout_seconds: int,
        input_data: dict[str, Any] | None,
    ) -> None:
        """Execute code in a subprocess with timeout."""
        start = time.monotonic()

        # Prepend input_data injection if provided
        preamble = ""
        if input_data is not None:
            escaped = json.dumps(input_data)
            preamble = f"import json as __json; INPUT_DATA = __json.loads({escaped!r})\n"

        full_code = preamble + code

        async with job_terminal_state(self.job_manager, job_id, label="Local sandbox execution"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    full_code,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    stdin=asyncio.subprocess.DEVNULL,
                )

                try:
                    stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
                except TimeoutError:
                    proc.kill()
                    await proc.wait()
                    await self.job_manager.update_job(
                        job_id,
                        status="timeout",
                        error="Execution exceeded deadline",
                    )
                    logger.warning("Local sandbox job timed out", job_id=job_id)
                    return

                elapsed_ms = (time.monotonic() - start) * 1000
                output = stdout_bytes.decode(errors="replace")[:10000] if stdout_bytes else ""

                if proc.returncode == 0:
                    result_url = await self._upload_results(job_id, output)
                    await self.job_manager.update_job(
                        job_id,
                        status="completed",
                        result={"logs": output, "execution_time_ms": elapsed_ms},
                        result_url=result_url,
                    )
                    logger.info("Local sandbox job completed", job_id=job_id, elapsed_ms=elapsed_ms)
                else:
                    error_msg = output[-2000:] if output else "Unknown error"
                    await self.job_manager.update_job(
                        job_id,
                        status="failed",
                        error=error_msg,
                    )
                    logger.warning("Local sandbox job failed", job_id=job_id, returncode=proc.returncode)

            except OSError:
                logger.exception("Local sandbox execution crashed", job_id=job_id)
                await self.job_manager.update_job(job_id, status="failed", error="Internal execution error")

    # -----------------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------------

    async def _upload_results(self, job_id: str, logs: str) -> str | None:
        """Upload execution logs to MinIO. Returns the object key, or None."""
        if not self.minio_client or not logs:
            return None
        key = f"sandbox/{job_id}/output.txt"
        try:
            await self.minio_client.upload(key, logs.encode(), content_type="text/plain")
            return key
        except ClientError, OSError:
            logger.debug("Failed to upload results to MinIO", job_id=job_id)
            return None

    async def get_status(self, job_id: str, tenant_id: str | None = None) -> JobStatus | None:
        return await self.job_manager.get_job(job_id, tenant_id)

    async def get_results(self, job_id: str, tenant_id: str | None = None) -> str | None:
        """Return presigned MinIO URL for results, or None."""
        job = await self.job_manager.get_job(job_id, tenant_id)
        if not job or job.status != "completed" or not job.result_url:
            return None
        if self.minio_client:
            return await self.minio_client.presigned_url(job.result_url)
        return job.result_url
