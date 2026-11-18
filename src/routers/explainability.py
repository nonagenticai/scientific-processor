"""Explainability endpoints — SubgraphX GNN explanations."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request

from auth.dependencies import resolve_tenant_id
from clients.graph_adapter import GraphAdapterError, GraphServiceUnavailableError
from dependencies import get_job_manager, get_subgraphx_service
from log_config import get_logger
from models.explainability import SubgraphXJobStatus, SubgraphXRequest
from services.job_manager import JobManager
from services.subgraphx_service import SubgraphXService

logger = get_logger(__name__)

router = APIRouter()


@router.post("/subgraphx", operation_id="explainability_subgraphx")
async def run_subgraphx(
    body: SubgraphXRequest,
    request: Request,
    service: SubgraphXService = Depends(get_subgraphx_service),
) -> dict[str, str]:
    """Launch a SubgraphX explanation job for a GNN prediction."""
    tenant_id = resolve_tenant_id(request)
    try:
        job_id = await service.explain(
            node_id=body.node_id,
            model_type=body.model_type,
            num_iterations=body.num_iterations,
            subgraph_size=body.subgraph_size,
            tenant_id=tenant_id,
        )
        return {"job_id": job_id}
    except GraphServiceUnavailableError, GraphAdapterError, httpx.HTTPError, KeyError, ValueError:
        logger.exception("SubgraphX explanation launch failed")
        raise HTTPException(status_code=500, detail="SubgraphX explanation launch failed")


@router.get("/jobs/{job_id}", response_model=SubgraphXJobStatus, operation_id="explainability_get_job")
async def get_explanation_job(
    job_id: str,
    request: Request,
    job_manager: JobManager = Depends(get_job_manager),
) -> SubgraphXJobStatus:
    """Get status and results of a SubgraphX explanation job."""
    job = await job_manager.get_job(job_id, tenant_id=resolve_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    computation_time_ms = None
    if job.result and "computation_time_ms" in job.result:
        computation_time_ms = job.result["computation_time_ms"]

    # SubgraphXJobStatus doesn't have "timeout" — map to "failed"
    explain_status: str = "failed" if job.status == "timeout" else job.status
    return SubgraphXJobStatus(
        job_id=job.job_id,
        status=explain_status,
        progress=job.progress,
        result=job.result,
        computation_time_ms=computation_time_ms,
    )
