"""Health check endpoints for the Scientific Processor."""

import time

import httpx
from fastapi import APIRouter
from starlette.responses import JSONResponse
from starlette.status import HTTP_200_OK, HTTP_503_SERVICE_UNAVAILABLE

from config import get_settings
from models.common import HealthStatus, LivenessStatus, ReadinessStatus

router = APIRouter()

_HEALTH_TIMEOUT = httpx.Timeout(5.0)


async def _check_service(name: str, url: str) -> dict:
    """Check a single backend service health."""
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT, verify=False) as client:  # nosec B501 — internal service, self-signed certs
            resp = await client.get(f"{url}/health")
            elapsed = (time.time() - start) * 1000
            return {
                "status": "healthy" if resp.status_code < 500 else "unhealthy",
                "response_time_ms": round(elapsed, 2),
            }
    except (httpx.HTTPError, OSError) as exc:
        elapsed = (time.time() - start) * 1000
        return {
            "status": "unhealthy",
            "response_time_ms": round(elapsed, 2),
            "error": str(exc),
        }


@router.get("/", include_in_schema=False, operation_id="redirect_root_to_docs")
async def root():
    """Root endpoint - redirects to API documentation."""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/docs")


@router.get("/health", response_model=HealthStatus, status_code=HTTP_200_OK, operation_id="health_check")
async def health() -> HealthStatus:
    """Health check — queries backend services and returns aggregate status."""
    settings = get_settings()

    service_urls = {
        "graph_adapter": settings.graph_adapter_url,
        "pytorch_backend": settings.pytorch_backend_url,
        "inference_router": settings.inference_router_url,
        "discovery_engine": settings.discovery_engine_url,
    }

    services: dict = {}
    for name, url in service_urls.items():
        if url:
            services[name] = await _check_service(name, url)

    unhealthy_count = sum(1 for s in services.values() if s.get("status") == "unhealthy")
    if unhealthy_count == 0:
        overall = "healthy"
    elif unhealthy_count < len(services):
        overall = "degraded"
    else:
        overall = "unhealthy"

    return HealthStatus(
        status=overall,
        services=services,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def _readiness_response(body: ReadinessStatus) -> JSONResponse:
    """Serialise a readiness verdict at the status code the verdict implies.

    This exists because returning the bare model does NOT do that. FastAPI
    serialises any non-Response return value at the decorator's `status_code`,
    so a `status_code=HTTP_200_OK` decorator made `/health/ready` answer 200 for
    `ready=False` too. The Deployment wires this path as an httpGet
    readinessProbe, and an httpGet probe reads ONLY the status code, never the
    body — so `{"ready": false}` was computed correctly and then thrown away,
    and a pod whose graph_adapter was down stayed in the Service's endpoint
    rotation indefinitely.
    """
    return JSONResponse(
        content=body.model_dump(exclude_none=True),
        status_code=HTTP_200_OK if body.ready else HTTP_503_SERVICE_UNAVAILABLE,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessStatus,
    response_model_exclude_none=True,
    status_code=HTTP_200_OK,
    responses={HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessStatus}},
    operation_id="health_readiness",
)
async def readiness() -> JSONResponse:
    """Readiness probe — checks critical backend dependencies.

    Returns 503 when a critical dependency is unhealthy, so an httpGet probe
    can actually observe the failure.
    """
    settings = get_settings()
    critical = {
        "graph_adapter": settings.graph_adapter_url,
    }

    for name, url in critical.items():
        if not url:
            continue
        result = await _check_service(name, url)
        if result.get("status") == "unhealthy":
            return _readiness_response(ReadinessStatus(ready=False, reason=f"{name} is unhealthy"))

    return _readiness_response(ReadinessStatus(ready=True))


@router.get("/health/live", response_model=LivenessStatus, status_code=HTTP_200_OK, operation_id="health_liveness")
async def liveness() -> LivenessStatus:
    """Liveness probe — always returns alive."""
    return LivenessStatus(alive=True)


@router.get("/status", status_code=HTTP_200_OK, operation_id="health_status")
async def status_check() -> dict:
    """Lightweight status endpoint for gateway proxy health checks."""
    return {"status": "ok"}
