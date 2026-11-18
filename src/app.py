"""FastAPI application factory for the scientific processor service."""

import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from kag_common.attribution import AttributionContextMiddleware
from starlette.routing import BaseRoute

from auth.middleware import PROTECTED_PREFIXES, PUBLIC_ENDPOINTS, AuthMiddleware
from config import Settings, get_settings
from dependencies import clear_cache, close_clients, dispose_db_engine, set_db_session_factory
from log_config import get_logger
from middleware.logging import RequestLoggingMiddleware
from routers.cache import router as cache_router
from routers.explainability import router as explainability_router
from routers.formal import router as formal_router
from routers.health import router as health_router
from routers.pipeline import router as pipeline_router
from routers.sandbox import router as sandbox_router
from routers.statistics import router as statistics_router
from routers.topology import router as topology_router
from routers.verification import router as verification_router

logger = get_logger(__name__)


class _HealthCheckFilter(logging.Filter):
    """Filter out health check requests from uvicorn access logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_HealthCheckFilter())


def _route_operation_id(route: BaseRoute) -> str:
    """Fallback operation_id generator — used only if a route forgot to set its own."""
    name = getattr(route, "name", "") or "unnamed"
    tags: list[str] = list(getattr(route, "tags", []) or []) or ["default"]
    tag = tags[0].lower().replace(" ", "_")
    return f"{tag}_{name}"


class _SecuredFastAPI(FastAPI):
    """FastAPI subclass with openapi() override for security schemes and tenant headers."""

    def __init__(self, app_settings: Settings, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._app_settings = app_settings

    def openapi(self) -> dict[str, Any]:
        if self.openapi_schema:
            return self.openapi_schema

        from fastapi.openapi.utils import get_openapi

        openapi_schema = get_openapi(
            title=self.title,
            version=self.version,
            description=self.description,
            routes=self.routes,
        )

        components = openapi_schema.setdefault("components", {})
        security_schemes = components.setdefault("securitySchemes", {})
        security_schemes.setdefault(
            "BearerAuth",
            {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "Enter Bearer <token>",
            },
        )

        parameters = components.setdefault("parameters", {})
        parameters.setdefault(
            "TenantIdHeader",
            {
                "name": "X-Tenant-Id",
                "in": "header",
                "required": False,
                "description": "Tenant identifier (defaults to configured DEFAULT_TENANT_ID).",
                "schema": {"type": "string", "default": self._app_settings.default_tenant_id},
                "example": self._app_settings.default_tenant_id,
            },
        )

        for path, path_item in openapi_schema.get("paths", {}).items():
            normalized_path = path.rstrip("/") or "/"
            requires_auth = (
                any(normalized_path.startswith(prefix) for prefix in PROTECTED_PREFIXES)
                and normalized_path not in PUBLIC_ENDPOINTS
            )

            for operation in path_item.values():
                if not isinstance(operation, dict):
                    continue

                if requires_auth:
                    operation.setdefault("security", [{"BearerAuth": []}])
                else:
                    operation.pop("security", None)

                op_params = operation.setdefault("parameters", [])
                has_tenant = any(
                    p.get("name", "").lower() == "x-tenant-id" and p.get("in") == "header" for p in op_params
                )
                if not has_tenant:
                    op_params.append({"$ref": "#/components/parameters/TenantIdHeader"})

        self.openapi_schema = openapi_schema
        return self.openapi_schema


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        """Application lifespan events (startup/shutdown)."""
        # Initialize OpenTelemetry tracing (before other startup tasks)
        from telemetry import setup_telemetry

        otel_enabled = setup_telemetry(app, settings.service_name or "scientific-processor")

        # Set up database engine if DATABASE_URL is configured
        db_url = settings.database_url
        if db_url:
            from database import create_async_engine_from_url, create_session_factory

            engine = create_async_engine_from_url(db_url, pool_pre_ping=True)
            factory = create_session_factory(engine)
            set_db_session_factory(engine, factory)
            logger.info("Database engine initialized", url=db_url.split("@")[-1] if "@" in db_url else db_url)

        logger.info(
            "Scientific processor ready",
            version=settings.version,
            otel_enabled=otel_enabled,
        )

        yield

        # Shutdown — clear cache, close persistent HTTP clients and database engine
        clear_cache()
        await close_clients()
        await dispose_db_engine()
        logger.info("Shutting down scientific processor")

    docs_enabled = (settings.environment == "development") or settings.enable_docs
    app = _SecuredFastAPI(
        app_settings=settings,
        title="Scientific Processor API",
        description=(
            "Scientific data processing, topology analysis, verification (CoVe, RAGAS), "
            "statistical meta-analysis, explainability (SubgraphX), formal reasoning, "
            "and sandboxed code execution over knowledge graphs. "
            "Auth via Istio-forwarded headers; controlled by ENABLE_AUTH."
        ),
        version=settings.version,
        docs_url=settings.docs_url if docs_enabled else None,
        redoc_url=settings.redoc_url if docs_enabled else None,
        openapi_url=settings.openapi_url if docs_enabled else None,
        lifespan=lifespan,
        generate_unique_id_function=_route_operation_id,
    )

    # --- Middleware (order: CORS -> RequestLogging -> Auth) ---
    # Innermost: bind the caller's X-Project-Id / X-Job-Id (X-W2V-* win) for the
    # endpoint, so outbound clients forward it via attribution_event_hook. Owner
    # ruling 2026-10-07: cost project = the calling app's own project; this
    # shared service forwards it and owns none. Reset after every request.
    app.add_middleware(AttributionContextMiddleware)

    raw_origins = os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:8921,http://localhost:3000,http://127.0.0.1:3000",
    )
    allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    allow_credentials = "*" not in allowed_origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if "*" in allowed_origins else allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(RequestLoggingMiddleware)

    if settings.enable_auth:
        app.add_middleware(AuthMiddleware)
        logger.info("Authentication middleware enabled")
    else:
        logger.warning("Authentication middleware DISABLED (ENABLE_AUTH=false)")

    # --- Global exception handler ---
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Global exception handler returning standardized error responses."""
        request_id: str = request.state.request_id if hasattr(request.state, "request_id") else str(uuid.uuid7())

        if isinstance(exc, HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "code": f"HTTP_{exc.status_code}",
                        "message": exc.detail,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "request_id": request_id,
                    }
                },
                headers=exc.headers,
            )

        if isinstance(exc, BaseExceptionGroup):
            for sub_exc in exc.exceptions:
                if isinstance(sub_exc, HTTPException):
                    return JSONResponse(
                        status_code=sub_exc.status_code,
                        content={
                            "error": {
                                "code": f"HTTP_{sub_exc.status_code}",
                                "message": sub_exc.detail,
                                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "request_id": request_id,
                            }
                        },
                        headers=sub_exc.headers,
                    )
                if isinstance(sub_exc, BaseExceptionGroup):
                    for nested_exc in sub_exc.exceptions:
                        if isinstance(nested_exc, HTTPException):
                            return JSONResponse(
                                status_code=nested_exc.status_code,
                                content={
                                    "error": {
                                        "code": f"HTTP_{nested_exc.status_code}",
                                        "message": nested_exc.detail,
                                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                        "request_id": request_id,
                                    }
                                },
                                headers=nested_exc.headers,
                            )

        is_debug = settings.environment == "development" or settings.enable_docs
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred",
                    "details": str(exc) if is_debug else None,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "request_id": request_id,
                }
            },
        )

    # Include routers
    app.include_router(health_router, tags=["health"])
    app.include_router(cache_router, prefix="/api/cache", tags=["cache"])
    app.include_router(topology_router, prefix="/api/topology", tags=["topology"])
    app.include_router(verification_router, prefix="/api/verify", tags=["verification"])
    app.include_router(statistics_router, prefix="/api/stats", tags=["statistics"])
    app.include_router(explainability_router, prefix="/api/explain", tags=["explainability"])
    app.include_router(formal_router, prefix="/api/formal", tags=["formal"])
    app.include_router(sandbox_router, prefix="/api/sandbox", tags=["sandbox"])
    app.include_router(pipeline_router, prefix="/api/pipeline", tags=["pipeline"])

    return app
