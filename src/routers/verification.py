"""
Verification endpoints — CoVe chain-of-verification, RAGAS faithfulness,
and composite integrity scoring.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request

from auth.dependencies import resolve_tenant_id
from clients.graph_adapter import GraphAdapterError, GraphServiceUnavailableError
from dependencies import get_cove_service, get_integrity_service, get_ragas_service
from log_config import get_logger
from models.verification import (
    CoVeRequest,
    CoVeResponse,
    IntegrityRequest,
    IntegrityResponse,
    RagasRequest,
    RagasResponse,
)
from services.cove_service import CoVeService
from services.integrity_service import IntegrityService
from services.ragas_service import RagasService

logger = get_logger(__name__)

router = APIRouter()


@router.post("/cove", response_model=CoVeResponse, operation_id="verification_cove")
async def verify_cove(
    body: CoVeRequest,
    request: Request,
    service: CoVeService = Depends(get_cove_service),
) -> CoVeResponse:
    """Run Chain-of-Verification (CoVe) on a claim.

    Generates verification questions, searches the knowledge graph for
    supporting/refuting evidence, and aggregates a verdict.
    """
    tenant_id = resolve_tenant_id(request)
    try:
        return await service.verify_claim(body, tenant_id)
    except GraphAdapterError, httpx.HTTPError, KeyError, ValueError, TypeError:
        logger.exception("CoVe verification failed")
        raise HTTPException(status_code=500, detail="CoVe verification failed")


@router.post("/ragas", response_model=RagasResponse, operation_id="verification_ragas")
async def verify_ragas(
    body: RagasRequest,
    request: Request,
    service: RagasService = Depends(get_ragas_service),
) -> RagasResponse:
    """Run RAGAS faithfulness evaluation on a question-answer-context tuple.

    Computes faithfulness, answer relevancy, context precision, and factual accuracy.
    """
    tenant_id = resolve_tenant_id(request)
    try:
        return await service.evaluate(body, tenant_id)
    except httpx.HTTPError, KeyError, ValueError, TypeError:
        logger.exception("RAGAS evaluation failed")
        raise HTTPException(status_code=500, detail="RAGAS evaluation failed")


@router.post("/integrity", response_model=IntegrityResponse, operation_id="verification_integrity")
async def compute_integrity(
    body: IntegrityRequest,
    request: Request,
    service: IntegrityService = Depends(get_integrity_service),
) -> IntegrityResponse:
    """Compute composite integrity score combining citation, cross-ref, community, authority, and contradiction signals."""
    tenant_id = resolve_tenant_id(request)
    try:
        return await service.compute_integrity(
            entity_id=body.entity_id,
            max_hops=body.max_hops,
            tenant_id=tenant_id,
            write_back=body.write_back,
        )
    except GraphServiceUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except GraphAdapterError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except httpx.HTTPError, KeyError, ValueError:
        logger.exception("Integrity computation failed")
        raise HTTPException(status_code=500, detail="Integrity computation failed")
