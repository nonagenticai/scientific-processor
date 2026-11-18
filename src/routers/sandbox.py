"""Sandboxed code execution endpoints — run user-supplied analysis code
in resource-limited Kubernetes containers."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from kubernetes_asyncio.client.exceptions import ApiException

from auth.dependencies import resolve_tenant_id
from dependencies import get_sandbox_service
from log_config import get_logger
from models.sandbox import SandboxJobStatus, SandboxRequest
from services.sandbox_service import SandboxService

logger = get_logger(__name__)

router = APIRouter()


@router.post("/execute", operation_id="sandbox_execute")
async def execute_sandbox(
    body: SandboxRequest,
    request: Request,
    service: SandboxService = Depends(get_sandbox_service),
) -> dict[str, str]:
    """Submit code for sandboxed execution."""
    tenant_id = resolve_tenant_id(request)
    try:
        job_id = await service.execute(
            code=body.code,
            language=body.language,
            timeout_seconds=body.timeout_seconds,
            input_data=body.input_data,
            tenant_id=tenant_id,
        )
        return {"job_id": job_id}
    except ApiException, OSError, ValueError, RuntimeError:
        logger.exception("Sandbox execution failed")
        raise HTTPException(status_code=500, detail="Sandbox execution failed")


@router.get("/jobs/{job_id}", response_model=SandboxJobStatus, operation_id="sandbox_get_job")
async def get_sandbox_job(
    job_id: str,
    request: Request,
    service: SandboxService = Depends(get_sandbox_service),
) -> SandboxJobStatus:
    """Get status of a sandboxed execution job."""
    job = await service.get_status(job_id, tenant_id=resolve_tenant_id(request))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return SandboxJobStatus(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        result_url=job.result_url,
        error=job.error,
        execution_time_ms=job.result.get("execution_time_ms") if job.result else None,
    )


@router.get("/results/{job_id}", operation_id="sandbox_get_results")
async def get_sandbox_results(
    job_id: str,
    request: Request,
    service: SandboxService = Depends(get_sandbox_service),
) -> dict[str, str]:
    """Retrieve results from a completed sandboxed execution."""
    result_url = await service.get_results(job_id, tenant_id=resolve_tenant_id(request))
    if result_url is None:
        raise HTTPException(status_code=404, detail="Results not available")
    return {"result_url": result_url}
