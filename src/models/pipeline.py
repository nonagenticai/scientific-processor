"""Pydantic models for the autonomous verification pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PipelineTriggerRequest(BaseModel):
    entity_ids: list[str] = Field(..., min_length=1, max_length=100)
    write_back: bool = True
    run_cove: bool = False
    run_formal: bool = False


class PipelineWebhookPayload(BaseModel):
    event_type: Literal["node_created", "node_updated", "edge_created", "edge_updated"]
    # Cap matches PipelineTriggerRequest.entity_ids: the webhook fire-and-forgets
    # verify_batch over the whole list, so an uncapped payload = unbounded load.
    entity_ids: list[str] = Field(..., min_length=1, max_length=100)
    timestamp: str | None = None


class PipelineBatchRequest(BaseModel):
    batch_size: int = Field(default=50, ge=1, le=500)
    max_hops: int = Field(default=2, ge=1, le=5)
    write_back: bool = True


class EntityVerificationResult(BaseModel):
    entity_id: str
    integrity_score: float | None = Field(
        default=None,
        description=(
            "None when the score could not be measured. A null score is NOT a low "
            "score — see `degraded_reason` for why it is missing."
        ),
    )
    components: list[dict[str, Any]] = Field(default_factory=list)
    cove_verdict: str | None = None
    formal_result: dict[str, Any] | None = None
    error: str | None = None
    computation_time_ms: float = 0.0
    # Mirrors IntegrityResponse.degraded* (#42). Without these the pipeline
    # dropped the whole diagnosis at its boundary: /api/pipeline/trigger returned
    # `integrity_score: null` with `error: null`, giving the caller no way to tell
    # a measured absence from a backend outage.
    degraded: bool = Field(
        default=False,
        description="True when `integrity_score` could not be measured, so it is null.",
    )
    degraded_components: list[str] = Field(
        default_factory=list,
        description="Names of the integrity components that could not be measured.",
    )
    degraded_reason: str | None = Field(
        default=None,
        description="Human-readable explanation of why the score could not be computed.",
    )
    not_applicable_components: list[str] = Field(
        default_factory=list,
        description=(
            "Integrity components whose metric is undefined for this entity (e.g. it "
            "has no incoming edges). NOT a degradation — the score is renormalized "
            "over the remaining components and is persisted, so the entity leaves the "
            "unscored pool instead of being re-verified on every batch."
        ),
    )


class PipelineResponse(BaseModel):
    job_id: str | None = None
    status: str = "accepted"
    entities_queued: int = 0
    message: str = ""


class PipelineStatusResponse(BaseModel):
    running: bool = False
    queue_depth: int = 0
    processed_total: int = 0
    failed_total: int = 0
    last_run_at: str | None = None
    throughput_per_minute: float = 0.0


class PipelineHistoryEntry(BaseModel):
    entity_id: str
    tenant_id: str
    integrity_score: float
    components: dict[str, Any] = Field(default_factory=dict)
    trigger: str
    cove_verdict: str | None = None
    formal_result: dict[str, Any] | None = None
    computed_at: str


class PipelineHistoryResponse(BaseModel):
    entity_id: str
    entries: list[PipelineHistoryEntry] = Field(default_factory=list)
    total: int = 0
