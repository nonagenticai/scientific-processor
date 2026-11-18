"""Tests for StatsService — meta-analysis, CI, trend detection.

All assertions use exact mathematically computed expected values
verified against the formulas in the service implementation.
"""

import numpy as np
import pytest
from scipy import stats as sp_stats

from models.statistics import (
    ConfidenceIntervalRequest,
    MetaAnalysisRequest,
    StudyResult,
    TimeSeriesPoint,
    TrendRequest,
)
from services.stats_service import StatsService


@pytest.fixture
def service():
    return StatsService()


# ---------------------------------------------------------------------------
# Meta-Analysis — Fixed Effects
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fixed_effects_meta_analysis(service):
    """Three studies: yi=[0.5, 0.6, 0.4], sei=[0.1, 0.15, 0.12].

    Hand-computed:
      wi = [100, 44.444444, 69.444444]
      pooled = sum(wi*yi)/sum(wi) = 104.4444.../213.8888... = 0.488312
      pooled_se = 1/sqrt(213.888...) = 0.068376
      Q = sum(wi*(yi-pooled)^2) = 1.1097
      I^2 = max(0, (Q-2)/Q) = 0.0  (Q < k-1=2)
      z_crit(0.95) = 1.959964
      CI = [0.354297, 0.622327]
    """
    studies = [
        StudyResult(effect_size=0.5, standard_error=0.1, sample_size=100),
        StudyResult(effect_size=0.6, standard_error=0.15, sample_size=80),
        StudyResult(effect_size=0.4, standard_error=0.12, sample_size=120),
    ]
    req = MetaAnalysisRequest(studies=studies, method="fixed", confidence_level=0.95)
    result = service.meta_analysis(req)

    assert result.method == "fixed"
    assert result.pooled_effect == pytest.approx(0.488312, abs=1e-6)
    assert result.pooled_se == pytest.approx(0.068376, abs=1e-6)
    assert result.ci_lower == pytest.approx(0.354297, abs=1e-6)
    assert result.ci_upper == pytest.approx(0.622327, abs=1e-6)
    assert result.heterogeneity_q == pytest.approx(1.1097, abs=1e-4)
    assert result.heterogeneity_i_squared == 0.0
    assert len(result.study_weights) == 3
    assert result.study_weights[0] == pytest.approx(100.0, abs=1e-6)
    assert result.study_weights[1] == pytest.approx(44.444444, abs=1e-6)
    assert result.study_weights[2] == pytest.approx(69.444444, abs=1e-6)


@pytest.mark.unit
def test_fixed_effects_90_percent_ci(service):
    """Fixed-effects with 90% CI: yi=[0.3, 0.5, 0.7], sei=[0.2, 0.2, 0.2].

    Hand-computed:
      wi = [25, 25, 25], sum_wi=75
      pooled = (7.5+12.5+17.5)/75 = 0.5
      pooled_se = 1/sqrt(75) = 0.11547
      Q = 25*(0.04+0+0.04) = 2.0
      I^2 = max(0, (2-2)/2) = 0.0
      z_crit(0.90) = 1.644854
      CI = [0.5 - 1.644854*0.11547, 0.5 + 1.644854*0.11547] = [0.310069, 0.689931]
    """
    studies = [
        StudyResult(effect_size=0.3, standard_error=0.2, sample_size=50),
        StudyResult(effect_size=0.5, standard_error=0.2, sample_size=50),
        StudyResult(effect_size=0.7, standard_error=0.2, sample_size=50),
    ]
    req = MetaAnalysisRequest(studies=studies, method="fixed", confidence_level=0.90)
    result = service.meta_analysis(req)

    assert result.method == "fixed"
    assert result.pooled_effect == pytest.approx(0.5, abs=1e-6)
    assert result.pooled_se == pytest.approx(0.11547, abs=1e-5)
    assert result.ci_lower == pytest.approx(0.310069, abs=1e-6)
    assert result.ci_upper == pytest.approx(0.689931, abs=1e-6)
    assert result.heterogeneity_q == pytest.approx(2.0, abs=1e-4)
    assert result.heterogeneity_i_squared == 0.0


# ---------------------------------------------------------------------------
# Meta-Analysis — Random Effects (DerSimonian-Laird)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_random_effects_meta_analysis(service):
    """Two studies: yi=[0.5, 0.8], sei=[0.1, 0.2].

    Hand-computed:
      wi = [100, 25], sum_wi=125
      pooled_fixed = (50+20)/125 = 0.56
      Q = 100*(0.5-0.56)^2 + 25*(0.8-0.56)^2 = 0.36+1.44 = 1.8
      I^2 = max(0, (1.8-1)/1.8) = 0.4444
      C = 125 - (10000+625)/125 = 125-85 = 40
      tau^2 = max(0, (1.8-1)/40) = 0.02
      wi* = [1/(0.01+0.02), 1/(0.04+0.02)] = [33.333333, 16.666667]
      pooled = (33.333*0.5+16.667*0.8)/50 = (16.667+13.333)/50 = 0.6
      pooled_se = 1/sqrt(50) = 0.141421
      CI = [0.6-1.96*0.1414, 0.6+1.96*0.1414] = [0.322819, 0.877181]
    """
    studies = [
        StudyResult(effect_size=0.5, standard_error=0.1, sample_size=100),
        StudyResult(effect_size=0.8, standard_error=0.2, sample_size=50),
    ]
    req = MetaAnalysisRequest(studies=studies, method="random", confidence_level=0.95)
    result = service.meta_analysis(req)

    assert result.method == "random"
    assert result.pooled_effect == pytest.approx(0.6, abs=1e-6)
    assert result.pooled_se == pytest.approx(0.141421, abs=1e-6)
    assert result.ci_lower == pytest.approx(0.322819, abs=1e-6)
    assert result.ci_upper == pytest.approx(0.877181, abs=1e-6)
    assert result.heterogeneity_q == pytest.approx(1.8, abs=1e-4)
    assert result.heterogeneity_i_squared == pytest.approx(0.4444, abs=1e-4)
    assert len(result.study_weights) == 2
    assert result.study_weights[0] == pytest.approx(33.333333, abs=1e-6)
    assert result.study_weights[1] == pytest.approx(16.666667, abs=1e-6)


@pytest.mark.unit
def test_random_effects_high_heterogeneity(service):
    """Two studies with large effect size gap: yi=[0.2, 1.8], sei=[0.1, 0.1].

    Hand-computed:
      wi = [100, 100], sum_wi=200
      pooled_fixed = 200/200 = 1.0
      Q = 100*(0.2-1.0)^2 + 100*(1.8-1.0)^2 = 64+64 = 128
      I^2 = (128-1)/128 = 0.9922
      C = 200 - 20000/200 = 200-100 = 100
      tau^2 = (128-1)/100 = 1.27
      wi* = [1/(0.01+1.27), 1/(0.01+1.27)] = [0.78125, 0.78125]
      pooled = 0.78125*(0.2+1.8)/(2*0.78125) = 1.0
      pooled_se = 1/sqrt(1.5625) = 0.8
      CI = [1.0-1.96*0.8, 1.0+1.96*0.8] = [-0.567971, 2.567971]
    """
    studies = [
        StudyResult(effect_size=0.2, standard_error=0.1, sample_size=100),
        StudyResult(effect_size=1.8, standard_error=0.1, sample_size=100),
    ]
    req = MetaAnalysisRequest(studies=studies, method="random", confidence_level=0.95)
    result = service.meta_analysis(req)

    assert result.pooled_effect == pytest.approx(1.0, abs=1e-6)
    assert result.pooled_se == pytest.approx(0.8, abs=1e-6)
    assert result.ci_lower == pytest.approx(-0.567971, abs=1e-6)
    assert result.ci_upper == pytest.approx(2.567971, abs=1e-6)
    assert result.heterogeneity_q == pytest.approx(128.0, abs=1e-4)
    assert result.heterogeneity_i_squared == pytest.approx(0.9922, abs=1e-4)
    assert result.study_weights[0] == pytest.approx(0.78125, abs=1e-6)
    assert result.study_weights[1] == pytest.approx(0.78125, abs=1e-6)


@pytest.mark.unit
def test_random_effects_collapses_to_fixed_when_no_heterogeneity(service):
    """When Q < k-1, tau^2 = 0 and random-effects equals fixed-effects.

    Studies: yi=[0.5, 0.6], sei=[0.1, 0.15]
      Q = 0.3077 < 1  =>  tau^2 = max(0, ...) = 0
      So wi* = wi = [100, 44.444444]
      pooled = 0.530769
    """
    studies = [
        StudyResult(effect_size=0.5, standard_error=0.1, sample_size=100),
        StudyResult(effect_size=0.6, standard_error=0.15, sample_size=80),
    ]
    req_random = MetaAnalysisRequest(studies=studies, method="random", confidence_level=0.95)
    req_fixed = MetaAnalysisRequest(studies=studies, method="fixed", confidence_level=0.95)

    result_random = service.meta_analysis(req_random)
    result_fixed = service.meta_analysis(req_fixed)

    # When tau^2=0, random and fixed should produce identical results
    assert result_random.pooled_effect == result_fixed.pooled_effect
    assert result_random.pooled_se == result_fixed.pooled_se
    assert result_random.ci_lower == result_fixed.ci_lower
    assert result_random.ci_upper == result_fixed.ci_upper
    assert result_random.pooled_effect == pytest.approx(0.530769, abs=1e-6)


@pytest.mark.unit
def test_heterogeneity_with_identical_studies(service):
    """Identical studies: yi=[0.5, 0.5], sei=[0.1, 0.1].

    Q = 100*(0)^2 + 100*(0)^2 = 0  =>  I^2 = 0.0
    pooled = 0.5 exactly
    """
    studies = [
        StudyResult(effect_size=0.5, standard_error=0.1, sample_size=100),
        StudyResult(effect_size=0.5, standard_error=0.1, sample_size=100),
    ]
    req = MetaAnalysisRequest(studies=studies, method="random")
    result = service.meta_analysis(req)

    assert result.heterogeneity_q == 0.0
    assert result.heterogeneity_i_squared == 0.0
    assert result.pooled_effect == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Confidence Intervals — Bootstrap
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_bootstrap_ci(service):
    """Bootstrap CI for [1..10] with default seed=42, n_bootstrap=5000.

    The service always uses seed=42, so results are deterministic.
    Pre-computed with numpy default_rng(42):
      mean = 5.5
      ci_lower = 3.7
      ci_upper = 7.3
    """
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    req = ConfidenceIntervalRequest(values=values, method="bootstrap", n_bootstrap=5000)
    result = await service.confidence_interval(req)

    assert result.method == "bootstrap"
    assert result.mean == pytest.approx(5.5, abs=1e-6)
    assert result.ci_lower == pytest.approx(3.7, abs=1e-6)
    assert result.ci_upper == pytest.approx(7.3, abs=1e-6)
    assert result.confidence_level == 0.95


@pytest.mark.unit
async def test_bootstrap_ci_1000_samples(service):
    """Bootstrap CI for [1..10] with n_bootstrap=1000.

    Pre-computed with numpy default_rng(42):
      ci_lower = 3.7
      ci_upper = 7.3
    """
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    req = ConfidenceIntervalRequest(values=values, method="bootstrap", n_bootstrap=1000)
    result = await service.confidence_interval(req)

    assert result.mean == pytest.approx(5.5, abs=1e-6)
    assert result.ci_lower == pytest.approx(3.7, abs=1e-6)
    assert result.ci_upper == pytest.approx(7.3, abs=1e-6)


@pytest.mark.unit
async def test_bootstrap_ci_is_deterministic(service):
    """Two calls with the same data must produce identical results (seed=42)."""
    values = [2.0, 4.0, 6.0, 8.0]
    req = ConfidenceIntervalRequest(values=values, method="bootstrap", n_bootstrap=2000)
    r1 = await service.confidence_interval(req)
    r2 = await service.confidence_interval(req)

    assert r1.ci_lower == r2.ci_lower
    assert r1.ci_upper == r2.ci_upper


# ---------------------------------------------------------------------------
# Confidence Intervals — Normal (t-distribution)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_normal_ci(service):
    """Normal CI for [1, 2, 3, 4, 5] at 95%.

    Hand-computed:
      mean = 3.0
      std(ddof=1) = sqrt(2.5) = 1.58114
      se = 1.58114 / sqrt(5) = 0.70711
      t_crit(0.95, df=4) from scipy.stats.t.interval
      ci_lower = 1.036757
      ci_upper = 4.963243
    """
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    req = ConfidenceIntervalRequest(values=values, method="normal", confidence_level=0.95)
    result = await service.confidence_interval(req)

    assert result.method == "normal"
    assert result.mean == pytest.approx(3.0, abs=1e-6)
    assert result.ci_lower == pytest.approx(1.036757, abs=1e-6)
    assert result.ci_upper == pytest.approx(4.963243, abs=1e-6)
    assert result.confidence_level == 0.95


@pytest.mark.unit
async def test_normal_ci_two_values(service):
    """Normal CI for [10, 20] at 95% — minimum sample size.

    mean = 15.0
    std(ddof=1) = sqrt(50) = 7.07107
    se = 7.07107 / sqrt(2) = 5.0
    t_crit(0.95, df=1) = 12.7062  (from scipy)
    ci = [15 - 12.7062*5, 15 + 12.7062*5] = [-48.531, 78.531]
    """
    values = [10.0, 20.0]
    req = ConfidenceIntervalRequest(values=values, method="normal", confidence_level=0.95)
    result = await service.confidence_interval(req)

    # Compute exact expected values
    arr = np.array([10.0, 20.0])
    se = float(np.std(arr, ddof=1) / np.sqrt(2))
    ci_l, ci_u = sp_stats.t.interval(0.95, df=1, loc=15.0, scale=se)

    assert result.mean == pytest.approx(15.0, abs=1e-6)
    assert result.ci_lower == pytest.approx(round(float(ci_l), 6), abs=1e-6)
    assert result.ci_upper == pytest.approx(round(float(ci_u), 6), abs=1e-6)


# ---------------------------------------------------------------------------
# Trend Detection — Mann-Kendall
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_mann_kendall_increasing(service):
    """Perfect increasing sequence: value = i*2 for i=0..19.

    Hand-computed:
      S = n*(n-1)/2 = 190  (all pairs concordant)
      var_S = 20*19*45/18 = 950
      z = (190-1)/sqrt(950) = 189/30.822 = 6.131970
      p = 2*(1 - Phi(6.132)) ~ 8.68e-10, rounds to 0.0
      tau = 2*190/(20*19) = 1.0
      Theil-Sen slope = 2.0
    """
    data = [TimeSeriesPoint(timestamp=float(i), value=float(i * 2)) for i in range(20)]
    req = TrendRequest(data=data, test="mann_kendall")
    result = await service.detect_trend(req)

    assert result.trend == "increasing"
    assert result.statistic == pytest.approx(1.0, abs=1e-6)  # tau
    assert result.p_value == pytest.approx(0.0, abs=1e-6)  # rounds to 0
    assert result.slope == pytest.approx(2.0, abs=1e-6)
    assert result.intercept is None


@pytest.mark.unit
async def test_mann_kendall_decreasing(service):
    """Perfect decreasing sequence: value = 100 - i*3 for i=0..19.

    Hand-computed:
      S = -190  (all pairs discordant)
      z = (-190+1)/sqrt(950) = -6.131970
      p ~ 8.68e-10, rounds to 0.0
      tau = -1.0
      Theil-Sen slope = -3.0
    """
    data = [TimeSeriesPoint(timestamp=float(i), value=float(100 - i * 3)) for i in range(20)]
    req = TrendRequest(data=data, test="mann_kendall")
    result = await service.detect_trend(req)

    assert result.trend == "decreasing"
    assert result.statistic == pytest.approx(-1.0, abs=1e-6)  # tau
    assert result.p_value == pytest.approx(0.0, abs=1e-6)
    assert result.slope == pytest.approx(-3.0, abs=1e-6)


@pytest.mark.unit
async def test_mann_kendall_no_trend(service):
    """Alternating 0/1 values: value = i%2 for i=0..19.

    Hand-computed:
      S = 10, var_S = 950
      z = (10-1)/sqrt(950) = 9/30.822 = 0.29200
      p = 2*(1-Phi(0.292)) = 0.770288
      tau = 2*10/(20*19) = 0.052632
      Theil-Sen slope = 0.0
    """
    data = [TimeSeriesPoint(timestamp=float(i), value=float(i % 2)) for i in range(20)]
    req = TrendRequest(data=data, test="mann_kendall")
    result = await service.detect_trend(req)

    assert result.trend == "no_trend"
    assert result.p_value == pytest.approx(0.770288, abs=1e-6)
    assert result.statistic == pytest.approx(0.052632, abs=1e-6)  # tau
    assert result.slope == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
async def test_mann_kendall_minimum_points(service):
    """Three increasing points [1, 2, 3] — minimum allowed.

    Hand-computed:
      S = 3, var_S = 3*(2)*(11)/18 = 3.6667
      z = (3-1)/sqrt(3.6667) = 2/1.9149 = 1.04447
      p = 2*(1-Phi(1.04447)) = 0.29627
      tau = 2*3/(3*2) = 1.0
      slope = 1.0

    Note: p > 0.05 so trend = 'no_trend' despite tau=1.0 (too few points).
    """
    data = [TimeSeriesPoint(timestamp=float(i), value=float(i + 1)) for i in range(3)]
    req = TrendRequest(data=data, test="mann_kendall")
    result = await service.detect_trend(req)

    assert result.trend == "no_trend"  # p=0.296 > 0.05
    assert result.statistic == pytest.approx(1.0, abs=1e-6)
    assert result.p_value == pytest.approx(0.29627, abs=1e-5)
    assert result.slope == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Trend Detection — Linear Regression
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_linear_regression_trend(service):
    """Perfect linear: value = i*1.5 + 10 for i=0..19.

    scipy.linregress on a perfect line gives exact slope/intercept, p ~ 0.
    """
    data = [TimeSeriesPoint(timestamp=float(i), value=float(i * 1.5 + 10)) for i in range(20)]
    req = TrendRequest(data=data, test="linear_regression")
    result = await service.detect_trend(req)

    assert result.trend == "increasing"
    assert result.slope == pytest.approx(1.5, abs=1e-6)
    assert result.intercept == pytest.approx(10.0, abs=1e-6)
    assert result.statistic == pytest.approx(1.5, abs=1e-6)  # statistic == slope
    assert result.p_value == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
async def test_linear_regression_decreasing(service):
    """Perfect decreasing linear: value = 100 - i*5 for i=0..9.

    slope = -5.0, intercept = 100.0, p ~ 0.
    """
    data = [TimeSeriesPoint(timestamp=float(i), value=float(100 - i * 5)) for i in range(10)]
    req = TrendRequest(data=data, test="linear_regression")
    result = await service.detect_trend(req)

    assert result.trend == "decreasing"
    assert result.slope == pytest.approx(-5.0, abs=1e-6)
    assert result.intercept == pytest.approx(100.0, abs=1e-6)
    assert result.p_value == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
async def test_linear_regression_no_trend(service):
    """Flat line: all values = 42.0.

    slope = 0.0, intercept = 42.0, p = NaN for constant (scipy returns NaN).
    The service should classify as 'no_trend' since p >= 0.05 (NaN is not < 0.05).
    """
    data = [TimeSeriesPoint(timestamp=float(i), value=42.0) for i in range(10)]
    req = TrendRequest(data=data, test="linear_regression")
    result = await service.detect_trend(req)

    assert result.trend == "no_trend"
    assert result.slope == pytest.approx(0.0, abs=1e-6)
    assert result.intercept == pytest.approx(42.0, abs=1e-6)


# ---------------------------------------------------------------------------
# NEW: Additional edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_mann_kendall_constant_values(service):
    """All same value -> S=0, z=0, p=1.0, no trend.

    Hand-computed:
      S = 0 (all pairs equal, no concordant or discordant)
      z = 0.0
      p = 2*(1-Phi(0)) = 1.0
      tau = 0.0
      Theil-Sen slope: all slopes are 0 -> median = 0.0
    """
    data = [TimeSeriesPoint(timestamp=float(i), value=5.0) for i in range(10)]
    req = TrendRequest(data=data, test="mann_kendall")
    result = await service.detect_trend(req)

    assert result.trend == "no_trend"
    assert result.statistic == pytest.approx(0.0, abs=1e-6)
    assert result.p_value == pytest.approx(1.0, abs=1e-6)
    assert result.slope == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
async def test_bootstrap_ci_reproducible(service):
    """With fixed seed (42), calling twice produces identical CIs."""
    values = [3.0, 7.0, 2.0, 9.0, 5.0, 1.0, 8.0, 4.0, 6.0, 10.0]
    req = ConfidenceIntervalRequest(values=values, method="bootstrap", n_bootstrap=5000)
    r1 = await service.confidence_interval(req)
    r2 = await service.confidence_interval(req)

    assert r1.mean == r2.mean
    assert r1.ci_lower == r2.ci_lower
    assert r1.ci_upper == r2.ci_upper


@pytest.mark.unit
async def test_linear_regression_perfect_line(service):
    """y = 2x + 1 -> slope=2.0, intercept=1.0 exactly.

    scipy.linregress on perfect data gives exact values, p~0.
    """
    data = [TimeSeriesPoint(timestamp=float(x), value=float(2 * x + 1)) for x in range(10)]
    req = TrendRequest(data=data, test="linear_regression")
    result = await service.detect_trend(req)

    assert result.trend == "increasing"
    assert result.slope == pytest.approx(2.0, abs=1e-6)
    assert result.intercept == pytest.approx(1.0, abs=1e-6)
    assert result.p_value == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_meta_analysis_response_has_all_fields(service):
    """Verify ALL MetaAnalysisResponse fields are present and correctly typed."""
    studies = [
        StudyResult(effect_size=0.3, standard_error=0.1, sample_size=50),
        StudyResult(effect_size=0.7, standard_error=0.2, sample_size=100),
    ]
    req = MetaAnalysisRequest(studies=studies, method="random", confidence_level=0.95)
    result = service.meta_analysis(req)

    # All fields exist and have correct types
    assert isinstance(result.pooled_effect, float)
    assert isinstance(result.pooled_se, float)
    assert isinstance(result.ci_lower, float)
    assert isinstance(result.ci_upper, float)
    assert isinstance(result.heterogeneity_q, float)
    assert isinstance(result.heterogeneity_i_squared, float)
    assert isinstance(result.method, str)
    assert isinstance(result.study_weights, list)
    assert isinstance(result.computation_time_ms, float)

    # Structural checks
    assert result.method == "random"
    assert len(result.study_weights) == 2
    assert result.ci_lower < result.pooled_effect < result.ci_upper
    assert result.pooled_se > 0
    assert 0.0 <= result.heterogeneity_i_squared <= 1.0
    assert result.heterogeneity_q >= 0.0
    assert result.computation_time_ms > 0


@pytest.mark.unit
async def test_trend_detection_with_timestamps(service):
    """Verify that non-sequential timestamps work with linear_regression.

    timestamps = [0, 10, 20, 30, 40], values = [0, 10, 20, 30, 40]
    -> slope = 1.0, intercept = 0.0
    """
    data = [TimeSeriesPoint(timestamp=float(i * 10), value=float(i * 10)) for i in range(5)]
    req = TrendRequest(data=data, test="linear_regression")
    result = await service.detect_trend(req)

    assert result.trend == "increasing"
    assert result.slope == pytest.approx(1.0, abs=1e-6)
    assert result.intercept == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# CPU-bound compute is offloaded off the event loop
# ---------------------------------------------------------------------------


async def test_confidence_interval_offloads_to_thread(service, monkeypatch):
    """The bootstrap compute must run via asyncio.to_thread, never inline on the loop."""
    called = {}

    async def fake_to_thread(fn, *args, **kwargs):
        called["fn"] = fn
        return fn(*args, **kwargs)

    monkeypatch.setattr("services.stats_service.asyncio.to_thread", fake_to_thread)
    req = ConfidenceIntervalRequest(values=[1.0, 2.0, 3.0], method="bootstrap", n_bootstrap=200)
    result = await service.confidence_interval(req)

    assert called["fn"] == service._confidence_interval
    assert result.method == "bootstrap"


async def test_detect_trend_offloads_to_thread(service, monkeypatch):
    """The Mann-Kendall compute must run via asyncio.to_thread, never inline on the loop."""
    called = {}

    async def fake_to_thread(fn, *args, **kwargs):
        called["fn"] = fn
        return fn(*args, **kwargs)

    monkeypatch.setattr("services.stats_service.asyncio.to_thread", fake_to_thread)
    data = [TimeSeriesPoint(timestamp=float(i), value=float(i)) for i in range(5)]
    req = TrendRequest(data=data, test="mann_kendall")
    result = await service.detect_trend(req)

    assert called["fn"] == service._detect_trend
    assert result.trend in ("increasing", "decreasing", "no_trend")
