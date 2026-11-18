"""Pydantic models for sandboxed code execution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SandboxRequest(BaseModel):
    code: str = Field(..., max_length=50000)
    language: Literal["python"] = "python"
    timeout_seconds: int = Field(default=60, ge=5, le=300)
    input_data: dict[str, Any] | None = None


class SandboxJobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "completed", "failed", "timeout"]
    created_at: str
    completed_at: str | None = None
    result_url: str | None = None
    error: str | None = None
    execution_time_ms: float | None = None
