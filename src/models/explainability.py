"""Pydantic models for SubgraphX GNN explainability."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SubgraphXRequest(BaseModel):
    node_id: str
    model_type: Literal["gcn", "gat", "graphsage"] = "gcn"
    num_iterations: int = Field(default=20, ge=5, le=100)
    subgraph_size: int = Field(default=5, ge=2, le=20)


class SubgraphXJobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    progress: float = 0.0
    result: dict[str, Any] | None = None
    computation_time_ms: float | None = None
