"""
Statistical analysis endpoints — meta-analysis, confidence intervals,
and trend detection.
"""

from fastapi import APIRouter, Depends

from dependencies import get_stats_service
from models.statistics import (
    ConfidenceIntervalRequest,
    ConfidenceIntervalResponse,
    MetaAnalysisRequest,
    MetaAnalysisResponse,
    TrendRequest,
    TrendResponse,
)
from services.stats_service import StatsService

router = APIRouter()


@router.post("/meta-analysis", response_model=MetaAnalysisResponse, operation_id="statistics_meta_analysis")
async def run_meta_analysis(
    body: MetaAnalysisRequest,
    service: StatsService = Depends(get_stats_service),
) -> MetaAnalysisResponse:
    """Run fixed/random-effects meta-analysis over a set of study results."""
    return service.meta_analysis(body)


@router.post(
    "/confidence-interval", response_model=ConfidenceIntervalResponse, operation_id="statistics_confidence_interval"
)
async def compute_confidence_interval(
    body: ConfidenceIntervalRequest,
    service: StatsService = Depends(get_stats_service),
) -> ConfidenceIntervalResponse:
    """Compute confidence intervals for aggregated effect sizes."""
    return await service.confidence_interval(body)


@router.post("/trend", response_model=TrendResponse, operation_id="statistics_trend")
async def detect_trend(
    body: TrendRequest,
    service: StatsService = Depends(get_stats_service),
) -> TrendResponse:
    """Detect temporal trends in scientific evidence."""
    return await service.detect_trend(body)
