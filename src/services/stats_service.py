"""Statistical analysis service — meta-analysis, confidence intervals, trend detection."""

from __future__ import annotations

import asyncio
import math
import time

import numpy as np
from scipy import stats as sp_stats

from models.statistics import (
    ConfidenceIntervalRequest,
    ConfidenceIntervalResponse,
    MetaAnalysisRequest,
    MetaAnalysisResponse,
    TrendRequest,
    TrendResponse,
)


class StatsService:
    def meta_analysis(self, req: MetaAnalysisRequest) -> MetaAnalysisResponse:
        start = time.monotonic()

        yi = np.array([s.effect_size for s in req.studies])
        sei = np.array([s.standard_error for s in req.studies])
        k = len(yi)

        # Inverse-variance weights
        wi = 1.0 / (sei**2)

        # Fixed-effects pooled estimate
        pooled_fixed = float(np.sum(wi * yi) / np.sum(wi))

        # Cochran's Q
        q = float(np.sum(wi * (yi - pooled_fixed) ** 2))

        # I-squared
        i_squared = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0

        if req.method == "random":
            # DerSimonian-Laird tau²
            c = float(np.sum(wi) - np.sum(wi**2) / np.sum(wi))
            tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
            wi_star = 1.0 / (sei**2 + tau2)
            pooled = float(np.sum(wi_star * yi) / np.sum(wi_star))
            pooled_se = float(1.0 / np.sqrt(np.sum(wi_star)))
            weights_out = wi_star.tolist()
        else:
            pooled = pooled_fixed
            pooled_se = float(1.0 / np.sqrt(np.sum(wi)))
            weights_out = wi.tolist()

        z_crit = sp_stats.norm.ppf((1 + req.confidence_level) / 2)
        ci_lower = pooled - z_crit * pooled_se
        ci_upper = pooled + z_crit * pooled_se

        elapsed = (time.monotonic() - start) * 1000
        return MetaAnalysisResponse(
            pooled_effect=round(pooled, 6),
            pooled_se=round(pooled_se, 6),
            ci_lower=round(ci_lower, 6),
            ci_upper=round(ci_upper, 6),
            heterogeneity_q=round(q, 4),
            heterogeneity_i_squared=round(i_squared, 4),
            method=req.method,
            study_weights=[round(w, 6) for w in weights_out],
            computation_time_ms=round(elapsed, 2),
        )

    async def confidence_interval(self, req: ConfidenceIntervalRequest) -> ConfidenceIntervalResponse:
        # Bootstrap CI is a pure-Python comprehension up to ``n_bootstrap``
        # iterations — offload to a worker thread so it never blocks the event
        # loop (mirrors CentralityService._run_centrality). Input size is capped
        # in ConfidenceIntervalRequest.
        return await asyncio.to_thread(self._confidence_interval, req)

    def _confidence_interval(self, req: ConfidenceIntervalRequest) -> ConfidenceIntervalResponse:
        start = time.monotonic()
        arr = np.array(req.values)
        mean_val = float(np.mean(arr))

        if req.method == "bootstrap":
            rng = np.random.default_rng(42)
            boot_means = np.array(
                [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(req.n_bootstrap)]
            )
            alpha = 1 - req.confidence_level
            ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
            ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        else:
            # Normal / t-distribution
            n = len(arr)
            se = float(np.std(arr, ddof=1) / np.sqrt(n))
            ci_lower_raw, ci_upper_raw = sp_stats.t.interval(req.confidence_level, df=n - 1, loc=mean_val, scale=se)
            ci_lower = float(ci_lower_raw)
            ci_upper = float(ci_upper_raw)

        elapsed = (time.monotonic() - start) * 1000
        return ConfidenceIntervalResponse(
            mean=round(mean_val, 6),
            ci_lower=round(ci_lower, 6),
            ci_upper=round(ci_upper, 6),
            confidence_level=req.confidence_level,
            method=req.method,
            computation_time_ms=round(elapsed, 2),
        )

    async def detect_trend(self, req: TrendRequest) -> TrendResponse:
        # ``_mann_kendall`` is O(n^2) over ``req.data`` — offload the sync compute
        # to a worker thread so it never blocks the event loop. Input size is
        # capped in TrendRequest.
        return await asyncio.to_thread(self._detect_trend, req)

    def _detect_trend(self, req: TrendRequest) -> TrendResponse:
        start = time.monotonic()
        timestamps = np.array([p.timestamp for p in req.data])
        values = np.array([p.value for p in req.data])

        if req.test == "mann_kendall":
            result = self._mann_kendall(values)
            elapsed = (time.monotonic() - start) * 1000
            return TrendResponse(
                trend=result["trend"],
                p_value=round(result["p_value"], 6),
                statistic=round(result["tau"], 6),
                slope=round(result["slope"], 6) if result["slope"] is not None else None,
                intercept=None,
                computation_time_ms=round(elapsed, 2),
            )
        else:
            slope, intercept, _r, p_value, _se = sp_stats.linregress(timestamps, values)
            if p_value < 0.05:
                trend = "increasing" if slope > 0 else "decreasing"
            else:
                trend = "no_trend"
            elapsed = (time.monotonic() - start) * 1000
            return TrendResponse(
                trend=trend,
                p_value=round(float(p_value), 6),
                statistic=round(float(slope), 6),
                slope=round(float(slope), 6),
                intercept=round(float(intercept), 6),
                computation_time_ms=round(elapsed, 2),
            )

    @staticmethod
    def _mann_kendall(values: np.ndarray) -> dict:
        """Mann-Kendall trend test with Theil-Sen slope."""
        n = len(values)
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                diff = values[j] - values[i]
                if diff > 0:
                    s += 1
                elif diff < 0:
                    s -= 1

        # Variance of S
        var_s = n * (n - 1) * (2 * n + 5) / 18.0

        # Z statistic
        if s > 0:
            z = (s - 1) / math.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / math.sqrt(var_s)
        else:
            z = 0.0

        p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))

        # Kendall's tau
        tau = 2 * s / (n * (n - 1)) if n > 1 else 0.0

        # Theil-Sen slope
        slopes = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                dx = j - i
                if dx != 0:
                    slopes.append((values[j] - values[i]) / dx)
        slope = float(np.median(slopes)) if slopes else None

        if p_value < 0.05:
            trend = "increasing" if s > 0 else "decreasing"
        else:
            trend = "no_trend"

        return {"trend": trend, "p_value": float(p_value), "tau": float(tau), "slope": slope}
