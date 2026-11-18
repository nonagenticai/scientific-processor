"""Pydantic models for statistical analysis endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StudyResult(BaseModel):
    effect_size: float
    standard_error: float = Field(gt=0)
    sample_size: int = Field(gt=0)
    weight: float | None = Field(default=None, description="Optional manual weight override")
    label: str | None = None


# --- Meta-Analysis ---


class MetaAnalysisRequest(BaseModel):
    studies: list[StudyResult] = Field(min_length=2)
    method: str = Field(default="random", pattern="^(fixed|random)$")
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)


class MetaAnalysisResponse(BaseModel):
    pooled_effect: float
    pooled_se: float
    ci_lower: float
    ci_upper: float
    heterogeneity_q: float
    heterogeneity_i_squared: float
    method: str
    study_weights: list[float]
    computation_time_ms: float


# --- Confidence Interval ---


class ConfidenceIntervalRequest(BaseModel):
    # ``values`` is capped so bootstrap resampling (n_bootstrap * len(values)
    # pure-Python work) stays bounded; ``n_bootstrap`` ceiling likewise keeps a
    # single request from pinning a worker thread for minutes.
    values: list[float] = Field(min_length=2, max_length=100_000)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    method: str = Field(default="bootstrap", pattern="^(bootstrap|normal)$")
    n_bootstrap: int = Field(default=10000, ge=100, le=100_000)


class ConfidenceIntervalResponse(BaseModel):
    mean: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    method: str
    computation_time_ms: float


# --- Trend Detection ---


class TimeSeriesPoint(BaseModel):
    timestamp: float
    value: float


class TrendRequest(BaseModel):
    # ``_mann_kendall`` is O(n^2) over ``data``; cap the length so a single
    # request cannot schedule an unbounded amount of CPU work.
    data: list[TimeSeriesPoint] = Field(min_length=3, max_length=10_000)
    test: str = Field(default="mann_kendall", pattern="^(mann_kendall|linear_regression)$")


class TrendResponse(BaseModel):
    trend: str = Field(description="increasing | decreasing | no_trend")
    p_value: float
    statistic: float
    slope: float | None = None
    intercept: float | None = None
    computation_time_ms: float
