"""Pydantic models for async job tracking."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

JobType = Literal["sandbox", "subgraphx", "formal"]
JobStatusType = Literal["pending", "running", "completed", "failed", "timeout"]


class JobStatus(BaseModel):
    job_id: str
    job_type: JobType
    status: JobStatusType = "pending"
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    result_url: str | None = None
    error: str | None = None
    tenant_id: str | None = None
