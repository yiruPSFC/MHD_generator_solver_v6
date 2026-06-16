from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BaseSigmaInterval:
    ok: bool
    sigma_lo: float
    sigma_hi: float
    lower_source: str
    upper_source: str
    sigma_slope_lower: float
    sigma_slope_upper: float
    sigma_logA_lower: float
    sigma_logA_upper: float
    sigma_curvature_lower: float
    sigma_curvature_upper: float
    error: str = ""


def max_with_source(current: float, source: str, candidate: float, candidate_source: str) -> tuple[float, str]:
    if float(candidate) > float(current):
        return float(candidate), str(candidate_source)
    return float(current), str(source)


def min_with_source(current: float, source: str, candidate: float, candidate_source: str) -> tuple[float, str]:
    if float(candidate) < float(current):
        return float(candidate), str(candidate_source)
    return float(current), str(source)


def build_base_sigma_interval(
    *,
    logA_current: float,
    dx: float,
    direction: int,
    sigma_min: float,
    sigma_max: float,
    curvature_max: float | None,
    sigma_prev: float | None,
    logA_min: float,
    logA_max: float,
) -> BaseSigmaInterval:
    step = float(dx)
    if step <= 0.0:
        return _empty_base_interval("dx must be positive.")
    direction_value = int(direction)
    if direction_value not in {-1, 1}:
        return _empty_base_interval("direction must be -1 or 1.")

    sigma_slope_lower = float(sigma_min)
    sigma_slope_upper = float(sigma_max)
    area_candidates = [
        (float(logA_min) - float(logA_current)) / (float(direction_value) * step),
        (float(logA_max) - float(logA_current)) / (float(direction_value) * step),
    ]
    sigma_logA_lower = float(min(area_candidates))
    sigma_logA_upper = float(max(area_candidates))
    if (
        curvature_max is None
        or not np.isfinite(float(curvature_max))
        or sigma_prev is None
        or not np.isfinite(float(sigma_prev))
    ):
        sigma_curvature_lower = -float("inf")
        sigma_curvature_upper = float("inf")
    else:
        width = abs(float(curvature_max))
        sigma_curvature_lower = float(sigma_prev) - width
        sigma_curvature_upper = float(sigma_prev) + width

    lo = -float("inf")
    hi = float("inf")
    lower_source = "none"
    upper_source = "none"
    lo, lower_source = max_with_source(lo, lower_source, sigma_slope_lower, "slope_min")
    hi, upper_source = min_with_source(hi, upper_source, sigma_slope_upper, "slope_max")
    area_lower_source = "area_max" if direction_value == -1 else "area_min"
    area_upper_source = "area_min" if direction_value == -1 else "area_max"
    lo, lower_source = max_with_source(lo, lower_source, sigma_logA_lower, area_lower_source)
    hi, upper_source = min_with_source(hi, upper_source, sigma_logA_upper, area_upper_source)
    lo, lower_source = max_with_source(lo, lower_source, sigma_curvature_lower, "curvature_min")
    hi, upper_source = min_with_source(hi, upper_source, sigma_curvature_upper, "curvature_max")

    return BaseSigmaInterval(
        ok=bool(lo <= hi),
        sigma_lo=float(lo),
        sigma_hi=float(hi),
        lower_source=lower_source,
        upper_source=upper_source,
        sigma_slope_lower=sigma_slope_lower,
        sigma_slope_upper=sigma_slope_upper,
        sigma_logA_lower=sigma_logA_lower,
        sigma_logA_upper=sigma_logA_upper,
        sigma_curvature_lower=sigma_curvature_lower,
        sigma_curvature_upper=sigma_curvature_upper,
        error="" if lo <= hi else "empty geometry interval",
    )


def build_forward_sigma_interval(
    *,
    logA_current: float,
    dx: float,
    sigma_prev: float | None,
    sigma_min: float,
    sigma_max: float,
    curvature_max: float | None,
    logA_min: float,
    logA_max: float,
) -> BaseSigmaInterval:
    return build_base_sigma_interval(
        logA_current=logA_current,
        dx=dx,
        direction=1,
        sigma_prev=sigma_prev,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        curvature_max=curvature_max,
        logA_min=logA_min,
        logA_max=logA_max,
    )


def _empty_base_interval(error: str) -> BaseSigmaInterval:
    return BaseSigmaInterval(
        ok=False,
        sigma_lo=float("nan"),
        sigma_hi=float("nan"),
        lower_source="none",
        upper_source="none",
        sigma_slope_lower=float("nan"),
        sigma_slope_upper=float("nan"),
        sigma_logA_lower=float("nan"),
        sigma_logA_upper=float("nan"),
        sigma_curvature_lower=float("nan"),
        sigma_curvature_upper=float("nan"),
        error=str(error),
    )
