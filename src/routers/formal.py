"""Formal reasoning endpoints — LLM-based translation to formal logic,
automated verification, and proof management.
"""

from __future__ import annotations

from typing import Any

import httpx
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from auth.dependencies import resolve_tenant_id
from clients.lean4_prover import Lean4ProverBusyError
from dependencies import get_formal_service
from log_config import get_logger
from models.formal import (
    FormalTranslateRequest,
    FormalTranslateResponse,
    FormalVerifyRequest,
    FormalVerifyResponse,
    ProofStoreRequest,
)
from services.formal_service import FormalService

logger = get_logger(__name__)

router = APIRouter()


#: OpenAPI doc for the GPU back-pressure answer (lean4-prover busy past the deadline).
_GPU_BUSY_RESPONSES: dict[int | str, dict[str, Any]] = {
    503: {
        "description": "lean4-prover GPU busy; retry after the `Retry-After` header (seconds).",
        "content": {"application/json": {"example": {"error": "gpu_busy", "retry_after": 10}}},
    }
}


def _gpu_busy_response(exc: Lean4ProverBusyError) -> JSONResponse:
    """TRANSIENT 503 + Retry-After, never a 500: the caller should re-drive."""
    logger.warning("Formal request shed: lean4-prover busy", retry_after=exc.retry_after)
    return JSONResponse(
        status_code=503,
        content={"error": "gpu_busy", "retry_after": exc.retry_after},
        headers={"Retry-After": str(exc.retry_after)},
    )


def _tenant_id(request: Request) -> str:
    # Authoritative tenant set by AuthMiddleware — never re-read the raw
    # X-Tenant-Id header (it is `Bearer <token>` on the introspection path).
    return resolve_tenant_id(request)


@router.post(
    "/translate",
    response_model=FormalTranslateResponse,
    operation_id="formal_translate",
    responses=_GPU_BUSY_RESPONSES,
)
async def translate_to_formal(
    body: FormalTranslateRequest,
    request: Request,
    service: FormalService = Depends(get_formal_service),
) -> FormalTranslateResponse | JSONResponse:
    """Translate a natural-language claim to a formal logic representation."""
    tenant_id = _tenant_id(request)
    try:
        return await service.translate(body.claim_text, body.target_language, tenant_id)
    except Lean4ProverBusyError as exc:
        return _gpu_busy_response(exc)
    except httpx.HTTPError, ValueError, KeyError:
        logger.exception("Formal translation failed")
        raise HTTPException(status_code=500, detail="Formal translation failed")


@router.post(
    "/verify",
    response_model=FormalVerifyResponse,
    operation_id="formal_verify",
    responses=_GPU_BUSY_RESPONSES,
)
async def verify_formal(
    body: FormalVerifyRequest,
    request: Request,
    service: FormalService = Depends(get_formal_service),
) -> FormalVerifyResponse | JSONResponse:
    """Verify a formal-logic expression using Z3, SymPy, or Lean 4."""
    tenant_id = _tenant_id(request)
    try:
        return await service.verify(body.formal_expression, body.language, tenant_id=tenant_id)
    except Lean4ProverBusyError as exc:
        return _gpu_busy_response(exc)
    except httpx.HTTPError, ValueError, KeyError:
        logger.exception("Formal verification failed")
        raise HTTPException(status_code=500, detail="Formal verification failed")


@router.post("/proofs/{claim_id}", operation_id="formal_create_proof")
async def create_proof(
    claim_id: str,
    body: ProofStoreRequest,
    request: Request,
    service: FormalService = Depends(get_formal_service),
) -> dict[str, str]:
    """Store a formal proof for a claim."""
    tenant_id = _tenant_id(request)
    try:
        proof_data = body.model_dump()
        key = await service.store_proof(claim_id, proof_data, tenant_id)
        return {"key": key}
    except RuntimeError, ClientError, OSError:
        logger.exception("Proof storage failed")
        raise HTTPException(status_code=500, detail="Proof storage failed")


@router.get("/proofs/{claim_id}", operation_id="formal_get_proofs")
async def get_proofs(
    claim_id: str,
    request: Request,
    service: FormalService = Depends(get_formal_service),
) -> list[dict[str, Any]]:
    """Retrieve existing proofs for a claim."""
    tenant_id = _tenant_id(request)
    try:
        return await service.get_proofs(claim_id, tenant_id=tenant_id)
    except ClientError, OSError, KeyError:
        logger.exception("Proof retrieval failed")
        raise HTTPException(status_code=500, detail="Proof retrieval failed")
