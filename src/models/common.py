"""Standard API response models (health, errors)."""

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response format."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp (ISO 8601)")
    request_id: str = Field(..., description="Unique request identifier for tracing")


class HealthStatus(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status: healthy, degraded, or unhealthy")
    services: dict[str, Any] = Field(default_factory=dict, description="Status of backend services")
    timestamp: str = Field(..., description="Timestamp (ISO 8601)")


class ReadinessStatus(BaseModel):
    """Readiness probe response."""

    ready: bool = Field(..., description="Whether service is ready")
    reason: str | None = Field(None, description="Reason if not ready")


class LivenessStatus(BaseModel):
    """Liveness probe response."""

    alive: bool = Field(..., description="Whether service is alive")
