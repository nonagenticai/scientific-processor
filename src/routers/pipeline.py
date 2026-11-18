"""Pipeline endpoints — autonomous verification triggers, batch processing, status."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, Request

from auth.dependencies import resolve_tenant_id
from config import Settings, get_settings
from dependencies import (
    get_cove_service,
    get_formal_service,
    get_graph_client,
    get_integrity_service,
    get_minio_client,
)
from log_config import get_logger
from models.pipeline import (
    PipelineBatchRequest,
    PipelineResponse,
    PipelineStatusResponse,
    PipelineTriggerRequest,
    PipelineWebhookPayload,
)
from services.history_service import HistoryService
from services.pipeline_service import PipelineService, PipelineState

logger = get_logger(__name__)

router = APIRouter()

# Process-wide pipeline counters. get_pipeline_service builds a fresh service per
# request, so the counters must live outside the request-scoped instance for
# /status to reflect real work.
_pipeline_state = PipelineState()

# Retain references to fire-and-forget background tasks so the event loop does not
# garbage-collect them mid-flight (see https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task).
_background_tasks: set[asyncio.Task[Any]] = set()


def _spawn(coro: Any) -> asyncio.Task[Any]:
    """Schedule a background task and keep a strong reference until it finishes."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def get_history_service(
    minio_client=Depends(get_minio_client),
) -> HistoryService:
    return HistoryService(minio_client=minio_client)


def _tenant_id(request: Request) -> str:
    # Authoritative tenant set by AuthMiddleware — never re-read the raw
    # X-Tenant-Id header (it is `Bearer <token>` on the introspection path).
    return resolve_tenant_id(request)


def get_pipeline_service(
    integrity_service=Depends(get_integrity_service),
    cove_service=Depends(get_cove_service),
    formal_service=Depends(get_formal_service),
    graph_client=Depends(get_graph_client),
    minio_client=Depends(get_minio_client),
    settings: Settings = Depends(get_settings),
    history_service: HistoryService = Depends(get_history_service),
) -> PipelineService:
    return PipelineService(
        integrity_service=integrity_service,
        cove_service=cove_service,
        formal_service=formal_service,
        graph_client=graph_client,
        minio_client=minio_client,
        settings=settings,
        history_service=history_service,
        state=_pipeline_state,
    )


@router.post("/trigger", operation_id="pipeline_trigger")
async def trigger_verification(
    body: PipelineTriggerRequest,
    request: Request,
    service: PipelineService = Depends(get_pipeline_service),
) -> dict[str, Any]:
    """Manually trigger verification for specific entity IDs."""
    tenant_id = _tenant_id(request)
    results = await service.verify_batch(
        entity_ids=body.entity_ids,
        tenant_id=tenant_id,
        write_back=body.write_back,
        run_cove=body.run_cove,
        run_formal=body.run_formal,
    )
    return {
        "status": "completed",
        "entities_queued": len(body.entity_ids),
        "results": [r.model_dump() for r in results],
    }


@router.post("/webhook", operation_id="pipeline_webhook")
async def webhook_notification(
    body: PipelineWebhookPayload,
    request: Request,
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineResponse:
    """Receive graph change notifications and trigger verification."""
    tenant_id = _tenant_id(request)
    logger.info("Webhook received", event_type=body.event_type, entity_count=len(body.entity_ids))

    # Fire-and-forget batch processing — a large webhook must not hold the sender's
    # connection open for minutes (which would time it out into duplicate retries).
    # Same pattern as /batch.
    if body.entity_ids:
        _spawn(
            service.verify_batch(
                entity_ids=body.entity_ids,
                tenant_id=tenant_id,
                write_back=True,
            )
        )

    return PipelineResponse(
        status="accepted",
        entities_queued=len(body.entity_ids),
        message=f"Processing {len(body.entity_ids)} entities from {body.event_type}",
    )


@router.post("/batch", operation_id="pipeline_batch")
async def batch_verification(
    body: PipelineBatchRequest,
    request: Request,
    service: PipelineService = Depends(get_pipeline_service),
) -> dict[str, Any]:
    """Find and verify all unscored entities in the graph."""
    tenant_id = _tenant_id(request)

    entity_ids = await service.find_unscored_entities(tenant_id=tenant_id, limit=body.batch_size)

    if entity_ids:
        # Fire-and-forget — same pattern as the webhook endpoint
        _spawn(
            service.verify_batch(
                entity_ids=entity_ids,
                tenant_id=tenant_id,
                write_back=body.write_back,
            )
        )

    return {
        "status": "accepted",
        "entities_queued": len(entity_ids),
    }


@router.get("/status", response_model=PipelineStatusResponse, operation_id="pipeline_status")
async def pipeline_status(
    request: Request,
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineStatusResponse:
    """Get pipeline health, queue depth, throughput."""
    return service.get_status()


@router.get("/history/{entity_id}", operation_id="pipeline_history")
async def entity_history(
    entity_id: str,
    request: Request,
    history: HistoryService = Depends(get_history_service),
) -> dict[str, Any]:
    """Get verification score history for an entity."""
    tenant_id = _tenant_id(request)
    entries = await history.get_history(entity_id, tenant_id)
    return {"entity_id": entity_id, "entries": entries, "total": len(entries)}


@router.get("/anomalies", operation_id="pipeline_anomalies")
async def get_anomalies(
    request: Request,
    threshold: float = 0.2,
    history: HistoryService = Depends(get_history_service),
) -> dict[str, Any]:
    """Get entities with significant score drops."""
    tenant_id = _tenant_id(request)
    anomalies = await history.get_anomalies(tenant_id, threshold_delta=threshold)
    return {"anomalies": anomalies, "total": len(anomalies)}
