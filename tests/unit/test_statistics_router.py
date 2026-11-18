"""Tests for statistics router endpoints.

All assertions use exact mathematically computed expected values
to validate both the HTTP layer and the underlying StatsService.
"""

import os

import pytest
from fastapi.testclient import TestClient

from app import create_app
from config import Settings


@pytest.fixture
def settings():
    os.environ["ENABLE_AUTH"] = "false"
    os.environ["ENABLE_DOCS"] = "true"
    os.environ["OTEL_SDK_DISABLED"] = "true"
    return Settings()


@pytest.fixture
def client(settings):
    app = create_app(settings)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Meta-Analysis Endpoint
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_meta_analysis_endpoint(client):
    """Random-effects with yi=[0.5, 0.6], sei=[0.1, 0.15].

    Q = 0.3077 < k-1=1  =>  tau^2 = 0, so random == fixed.
    pooled = (100*0.5 + 44.444*0.6) / 144.444 = 76.667/144.444 = 0.530769
    pooled_se = 1/sqrt(144.444) = 0.083205
    CI(95%) = [0.36769, 0.693848]
    """
    resp = client.post(
        "/api/stats/meta-analysis",
        json={
            "studies": [
                {"effect_size": 0.5, "standard_error": 0.1, "sample_size": 100},
                {"effect_size": 0.6, "standard_error": 0.15, "sample_size": 80},
            ],
            "method": "random",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["pooled_effect"] == pytest.approx(0.530769, abs=1e-6)
    assert data["pooled_se"] == pytest.approx(0.083205, abs=1e-6)
    assert data["ci_lower"] == pytest.approx(0.36769, abs=1e-5)
    assert data["ci_upper"] == pytest.approx(0.693848, abs=1e-6)
    assert data["heterogeneity_q"] == pytest.approx(0.3077, abs=1e-4)
    assert data["heterogeneity_i_squared"] == 0.0
    assert data["method"] == "random"
    assert len(data["study_weights"]) == 2
    assert data["study_weights"][0] == pytest.approx(100.0, abs=1e-6)
    assert data["study_weights"][1] == pytest.approx(44.444444, abs=1e-6)


@pytest.mark.unit
def test_meta_analysis_fixed_effects_endpoint(client):
    """Fixed-effects with three studies through the API."""
    resp = client.post(
        "/api/stats/meta-analysis",
        json={
            "studies": [
                {"effect_size": 0.5, "standard_error": 0.1, "sample_size": 100},
                {"effect_size": 0.6, "standard_error": 0.15, "sample_size": 80},
                {"effect_size": 0.4, "standard_error": 0.12, "sample_size": 120},
            ],
            "method": "fixed",
            "confidence_level": 0.95,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["method"] == "fixed"
    assert data["pooled_effect"] == pytest.approx(0.488312, abs=1e-6)
    assert data["pooled_se"] == pytest.approx(0.068376, abs=1e-6)


@pytest.mark.unit
def test_meta_analysis_high_heterogeneity_endpoint(client):
    """Random-effects with large effect size gap via API.

    yi=[0.2, 1.8], sei=[0.1, 0.1]
    I^2 = 0.9922, pooled = 1.0
    """
    resp = client.post(
        "/api/stats/meta-analysis",
        json={
            "studies": [
                {"effect_size": 0.2, "standard_error": 0.1, "sample_size": 100},
                {"effect_size": 1.8, "standard_error": 0.1, "sample_size": 100},
            ],
            "method": "random",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["pooled_effect"] == pytest.approx(1.0, abs=1e-6)
    assert data["heterogeneity_i_squared"] == pytest.approx(0.9922, abs=1e-4)
    assert data["pooled_se"] == pytest.approx(0.8, abs=1e-6)


# ---------------------------------------------------------------------------
# Confidence Interval Endpoint
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_confidence_interval_endpoint(client):
    """Bootstrap CI for [1..10] with seed=42, n_bootstrap=1000.

    Pre-computed: mean=5.5, ci_lower=3.7, ci_upper=7.3
    """
    resp = client.post(
        "/api/stats/confidence-interval",
        json={"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "method": "bootstrap", "n_bootstrap": 1000},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mean"] == pytest.approx(5.5, abs=1e-6)
    assert data["ci_lower"] == pytest.approx(3.7, abs=1e-6)
    assert data["ci_upper"] == pytest.approx(7.3, abs=1e-6)
    assert data["method"] == "bootstrap"
    assert data["confidence_level"] == 0.95


@pytest.mark.unit
def test_confidence_interval_normal_endpoint(client):
    """Normal CI for [1,2,3,4,5] at 95%.

    mean=3.0, ci_lower=1.036757, ci_upper=4.963243
    """
    resp = client.post(
        "/api/stats/confidence-interval",
        json={"values": [1, 2, 3, 4, 5], "method": "normal", "confidence_level": 0.95},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mean"] == pytest.approx(3.0, abs=1e-6)
    assert data["ci_lower"] == pytest.approx(1.036757, abs=1e-6)
    assert data["ci_upper"] == pytest.approx(4.963243, abs=1e-6)
    assert data["method"] == "normal"


# ---------------------------------------------------------------------------
# Trend Detection Endpoint
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_trend_endpoint(client):
    """Mann-Kendall for perfect increasing i*2, i=0..19.

    tau=1.0, p~0, slope=2.0
    """
    data_points = [{"timestamp": float(i), "value": float(i * 2)} for i in range(20)]
    resp = client.post("/api/stats/trend", json={"data": data_points, "test": "mann_kendall"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["trend"] == "increasing"
    assert data["statistic"] == pytest.approx(1.0, abs=1e-6)
    assert data["p_value"] == pytest.approx(0.0, abs=1e-6)
    assert data["slope"] == pytest.approx(2.0, abs=1e-6)


@pytest.mark.unit
def test_trend_linear_regression_endpoint(client):
    """Linear regression for i*1.5+10, i=0..19 through API.

    slope=1.5, intercept=10.0
    """
    data_points = [{"timestamp": float(i), "value": float(i * 1.5 + 10)} for i in range(20)]
    resp = client.post("/api/stats/trend", json={"data": data_points, "test": "linear_regression"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["trend"] == "increasing"
    assert data["slope"] == pytest.approx(1.5, abs=1e-6)
    assert data["intercept"] == pytest.approx(10.0, abs=1e-6)
    assert data["statistic"] == pytest.approx(1.5, abs=1e-6)


@pytest.mark.unit
def test_trend_decreasing_endpoint(client):
    """Mann-Kendall decreasing: value=100-i*3, i=0..19.

    tau=-1.0, slope=-3.0
    """
    data_points = [{"timestamp": float(i), "value": float(100 - i * 3)} for i in range(20)]
    resp = client.post("/api/stats/trend", json={"data": data_points, "test": "mann_kendall"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["trend"] == "decreasing"
    assert data["statistic"] == pytest.approx(-1.0, abs=1e-6)
    assert data["slope"] == pytest.approx(-3.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Validation — Error Responses
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_meta_analysis_validation_requires_min_studies(client):
    resp = client.post(
        "/api/stats/meta-analysis",
        json={"studies": [{"effect_size": 0.5, "standard_error": 0.1, "sample_size": 100}]},
    )
    assert resp.status_code == 422


@pytest.mark.unit
def test_meta_analysis_validation_rejects_zero_se(client):
    """standard_error must be > 0."""
    resp = client.post(
        "/api/stats/meta-analysis",
        json={
            "studies": [
                {"effect_size": 0.5, "standard_error": 0.0, "sample_size": 100},
                {"effect_size": 0.6, "standard_error": 0.1, "sample_size": 80},
            ],
        },
    )
    assert resp.status_code == 422


@pytest.mark.unit
def test_meta_analysis_validation_rejects_invalid_method(client):
    """method must be 'fixed' or 'random'."""
    resp = client.post(
        "/api/stats/meta-analysis",
        json={
            "studies": [
                {"effect_size": 0.5, "standard_error": 0.1, "sample_size": 100},
                {"effect_size": 0.6, "standard_error": 0.1, "sample_size": 80},
            ],
            "method": "bayesian",
        },
    )
    assert resp.status_code == 422


@pytest.mark.unit
def test_trend_validation_requires_min_points(client):
    resp = client.post("/api/stats/trend", json={"data": [{"timestamp": 1.0, "value": 2.0}]})
    assert resp.status_code == 422


@pytest.mark.unit
def test_ci_validation_requires_min_values(client):
    resp = client.post("/api/stats/confidence-interval", json={"values": [1.0]})
    assert resp.status_code == 422


@pytest.mark.unit
def test_ci_validation_rejects_invalid_method(client):
    """method must be 'bootstrap' or 'normal'."""
    resp = client.post(
        "/api/stats/confidence-interval",
        json={"values": [1.0, 2.0, 3.0], "method": "bayesian"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# NEW: Additional response field verification
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_meta_analysis_returns_all_fields(client):
    """Verify ALL MetaAnalysisResponse fields are present and correctly typed via API."""
    resp = client.post(
        "/api/stats/meta-analysis",
        json={
            "studies": [
                {"effect_size": 0.3, "standard_error": 0.1, "sample_size": 50},
                {"effect_size": 0.7, "standard_error": 0.2, "sample_size": 100},
            ],
            "method": "random",
            "confidence_level": 0.95,
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    # All fields must be present
    required_fields = [
        "pooled_effect",
        "pooled_se",
        "ci_lower",
        "ci_upper",
        "heterogeneity_q",
        "heterogeneity_i_squared",
        "method",
        "study_weights",
        "computation_time_ms",
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"

    # Type checks
    assert isinstance(data["pooled_effect"], float)
    assert isinstance(data["pooled_se"], float)
    assert isinstance(data["ci_lower"], float)
    assert isinstance(data["ci_upper"], float)
    assert isinstance(data["heterogeneity_q"], float)
    assert isinstance(data["heterogeneity_i_squared"], float)
    assert isinstance(data["method"], str)
    assert isinstance(data["study_weights"], list)
    assert isinstance(data["computation_time_ms"], float)

    # Structural validation
    assert data["ci_lower"] < data["pooled_effect"] < data["ci_upper"]
    assert data["computation_time_ms"] > 0


@pytest.mark.unit
def test_trend_detection_with_timestamps_endpoint(client):
    """Non-sequential timestamps work correctly through the API.

    timestamps = [100, 200, 300], values = [10, 20, 30]
    slope = 0.1, intercept = 0.0
    """
    data_points = [
        {"timestamp": 100.0, "value": 10.0},
        {"timestamp": 200.0, "value": 20.0},
        {"timestamp": 300.0, "value": 30.0},
    ]
    resp = client.post("/api/stats/trend", json={"data": data_points, "test": "linear_regression"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["trend"] == "increasing"
    assert data["slope"] == pytest.approx(0.1, abs=1e-6)
    assert data["intercept"] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.unit
def test_trend_validation_rejects_invalid_test(client):
    """test must be 'mann_kendall' or 'linear_regression'."""
    data_points = [{"timestamp": float(i), "value": float(i)} for i in range(5)]
    resp = client.post("/api/stats/trend", json={"data": data_points, "test": "arima"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Resource-exhaustion caps (reject unbounded CPU-bound inputs with 422)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ci_validation_rejects_oversize_values(client):
    """values beyond max_length=100_000 must be rejected before compute runs."""
    resp = client.post(
        "/api/stats/confidence-interval",
        json={"values": [1.0] * 100_001, "method": "normal"},
    )
    assert resp.status_code == 422


@pytest.mark.unit
def test_ci_validation_rejects_oversize_n_bootstrap(client):
    """n_bootstrap above the 100_000 ceiling must be rejected (no worker pinning)."""
    resp = client.post(
        "/api/stats/confidence-interval",
        json={"values": [1.0, 2.0, 3.0], "method": "bootstrap", "n_bootstrap": 1_000_001},
    )
    assert resp.status_code == 422


@pytest.mark.unit
def test_trend_validation_rejects_oversize_data(client):
    """data beyond max_length=10_000 must be rejected before O(n^2) compute runs."""
    data_points = [{"timestamp": float(i), "value": float(i)} for i in range(10_001)]
    resp = client.post("/api/stats/trend", json={"data": data_points, "test": "mann_kendall"})
    assert resp.status_code == 422
