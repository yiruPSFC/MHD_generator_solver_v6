from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sigma_interval import build_base_sigma_interval, max_with_source, min_with_source


@dataclass(frozen=True)
class ReverseCoefficients:
    p0: float
    p1: float
    q0: float
    q1: float

    @property
    def p1q1(self) -> float:
        return float(self.p1 * self.q1)


@dataclass(frozen=True)
class ReverseInterval:
    ok: bool
    sigma_lo: float
    sigma_hi: float
    lower_source: str
    upper_source: str
    reverse_G_bound_kind: str
    sigma_G_bound: float
    sigma_slope_lower: float
    sigma_slope_upper: float
    sigma_logA_lower: float
    sigma_logA_upper: float
    sigma_curvature_lower: float
    sigma_curvature_upper: float
    error: str = ""


@dataclass(frozen=True)
class EndpointDecision:
    sigma: float
    objective_bound_kind: str
    endpoint_source: str


def reverse_coefficients_from_forward(
    *,
    dx: float,
    G_current: float,
    G_floor: float,
    a0: float,
    a1: float,
    b0: float,
    b1: float,
) -> ReverseCoefficients:
    step = float(dx)
    return ReverseCoefficients(
        p0=float(step * float(a0)),
        p1=float(step * float(a1)),
        q0=float(float(G_current) - float(G_floor) - step * float(b0)),
        q1=float(-step * float(b1)),
    )


def build_reverse_sigma_interval(
    *,
    A_current: float,
    logA_current: float,
    dx: float,
    sigma_prev: float | None,
    sigma_min: float,
    sigma_max: float,
    curvature_max: float | None,
    logA_min: float,
    logA_max: float,
    q0: float,
    q1: float,
    q1_tol: float = 1.0e-14,
    g_margin_tol: float = 1.0e-12,
) -> ReverseInterval:
    step = float(dx)
    area = float(A_current)
    if step <= 0.0:
        return _empty_interval("dx must be positive.")
    if area <= 0.0 or not np.isfinite(area):
        return _empty_interval("A_current must be positive and finite.")

    base = build_base_sigma_interval(
        logA_current=logA_current,
        dx=step,
        direction=-1,
        sigma_prev=sigma_prev,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        curvature_max=curvature_max,
        logA_min=logA_min,
        logA_max=logA_max,
    )
    sigma_slope_lower = float(base.sigma_slope_lower)
    sigma_slope_upper = float(base.sigma_slope_upper)
    sigma_logA_lower = float(base.sigma_logA_lower)
    sigma_logA_upper = float(base.sigma_logA_upper)
    sigma_curvature_lower = float(base.sigma_curvature_lower)
    sigma_curvature_upper = float(base.sigma_curvature_upper)
    lo = float(base.sigma_lo)
    hi = float(base.sigma_hi)
    lower_source = str(base.lower_source)
    upper_source = str(base.upper_source)
    if lo > hi:
        return ReverseInterval(
            ok=False,
            sigma_lo=float(lo),
            sigma_hi=float(hi),
            lower_source=lower_source,
            upper_source=upper_source,
            reverse_G_bound_kind="not_applied",
            sigma_G_bound=float("nan"),
            sigma_slope_lower=sigma_slope_lower,
            sigma_slope_upper=sigma_slope_upper,
            sigma_logA_lower=sigma_logA_lower,
            sigma_logA_upper=sigma_logA_upper,
            sigma_curvature_lower=sigma_curvature_lower,
            sigma_curvature_upper=sigma_curvature_upper,
            error="empty geometry interval before reverse G intersection",
        )

    reverse_G_bound_kind = "none"
    sigma_G_bound = float("nan")
    if float(q1) > float(q1_tol):
        sigma_G_bound = float(-float(q0) / (float(q1) * area))
        lo, lower_source = max_with_source(lo, lower_source, sigma_G_bound, "G_lower")
        reverse_G_bound_kind = "lower"
    elif float(q1) < -float(q1_tol):
        sigma_G_bound = float(-float(q0) / (float(q1) * area))
        hi, upper_source = min_with_source(hi, upper_source, sigma_G_bound, "G_upper")
        reverse_G_bound_kind = "upper"
    elif float(q0) < -float(g_margin_tol):
        return ReverseInterval(
            ok=False,
            sigma_lo=float(lo),
            sigma_hi=float(hi),
            lower_source=lower_source,
            upper_source=upper_source,
            reverse_G_bound_kind="infeasible_flat",
            sigma_G_bound=float("nan"),
            sigma_slope_lower=sigma_slope_lower,
            sigma_slope_upper=sigma_slope_upper,
            sigma_logA_lower=sigma_logA_lower,
            sigma_logA_upper=sigma_logA_upper,
            sigma_curvature_lower=sigma_curvature_lower,
            sigma_curvature_upper=sigma_curvature_upper,
            error="reverse G margin is flat and infeasible at affine order",
        )

    if lo > hi:
        return ReverseInterval(
            ok=False,
            sigma_lo=float(lo),
            sigma_hi=float(hi),
            lower_source=lower_source,
            upper_source=upper_source,
            reverse_G_bound_kind=reverse_G_bound_kind,
            sigma_G_bound=sigma_G_bound,
            sigma_slope_lower=sigma_slope_lower,
            sigma_slope_upper=sigma_slope_upper,
            sigma_logA_lower=sigma_logA_lower,
            sigma_logA_upper=sigma_logA_upper,
            sigma_curvature_lower=sigma_curvature_lower,
            sigma_curvature_upper=sigma_curvature_upper,
            error="empty interval after reverse G intersection",
        )

    return ReverseInterval(
        ok=True,
        sigma_lo=float(lo),
        sigma_hi=float(hi),
        lower_source=lower_source,
        upper_source=upper_source,
        reverse_G_bound_kind=reverse_G_bound_kind,
        sigma_G_bound=sigma_G_bound,
        sigma_slope_lower=sigma_slope_lower,
        sigma_slope_upper=sigma_slope_upper,
        sigma_logA_lower=sigma_logA_lower,
        sigma_logA_upper=sigma_logA_upper,
        sigma_curvature_lower=sigma_curvature_lower,
        sigma_curvature_upper=sigma_curvature_upper,
    )


def choose_objective_endpoint(
    *,
    interval: ReverseInterval,
    p1: float,
    p1_tol: float = 1.0e-14,
    regularizer_sigma: float = 0.0,
) -> EndpointDecision:
    if float(p1) > float(p1_tol):
        return EndpointDecision(
            sigma=float(interval.sigma_hi),
            objective_bound_kind="upper",
            endpoint_source=str(interval.upper_source),
        )
    if float(p1) < -float(p1_tol):
        return EndpointDecision(
            sigma=float(interval.sigma_lo),
            objective_bound_kind="lower",
            endpoint_source=str(interval.lower_source),
        )
    return EndpointDecision(
        sigma=float(np.clip(float(regularizer_sigma), float(interval.sigma_lo), float(interval.sigma_hi))),
        objective_bound_kind="flat",
        endpoint_source="regularizer",
    )


def classify_endpoint_support(
    *,
    endpoint_source: str,
    objective_bound_kind: str,
    p1: float,
    q1: float,
    p1q1_tol: float = 1.0e-14,
) -> str:
    if str(objective_bound_kind) == "flat":
        return "singular_flat"

    source = str(endpoint_source)
    p1q1 = float(p1) * float(q1)
    if source in {"G_lower", "G_upper"}:
        if p1q1 < -float(p1q1_tol):
            return "G_limited_reverse"
        if p1q1 > float(p1q1_tol):
            return "G_permissive_reverse_selected_by_other_effect"
        return "G_flat_reverse"
    if source in {"slope_min", "slope_max"}:
        return "geometry_limited"
    if source in {"area_min", "area_max"}:
        return "area_limited"
    if source in {"curvature_min", "curvature_max"}:
        return "curvature_limited"
    if source in {"temperature_lower", "temperature_upper"}:
        return "temperature_limited"
    return "interior_or_regularized"


def interval_diagnostics(interval: ReverseInterval) -> dict[str, float | str | bool]:
    return {
        "reverse_interval_ok": bool(interval.ok),
        "sigma_interval_lower": float(interval.sigma_lo),
        "sigma_interval_upper": float(interval.sigma_hi),
        "lower_source": str(interval.lower_source),
        "upper_source": str(interval.upper_source),
        "reverse_G_bound_kind": str(interval.reverse_G_bound_kind),
        "sigma_G_bound": float(interval.sigma_G_bound),
        "sigma_slope_lower": float(interval.sigma_slope_lower),
        "sigma_slope_upper": float(interval.sigma_slope_upper),
        "sigma_logA_lower": float(interval.sigma_logA_lower),
        "sigma_logA_upper": float(interval.sigma_logA_upper),
        "sigma_curvature_lower": float(interval.sigma_curvature_lower),
        "sigma_curvature_upper": float(interval.sigma_curvature_upper),
        "reverse_interval_error": str(interval.error),
    }


def _empty_interval(error: str) -> ReverseInterval:
    return ReverseInterval(
        ok=False,
        sigma_lo=float("nan"),
        sigma_hi=float("nan"),
        lower_source="none",
        upper_source="none",
        reverse_G_bound_kind="not_applied",
        sigma_G_bound=float("nan"),
        sigma_slope_lower=float("nan"),
        sigma_slope_upper=float("nan"),
        sigma_logA_lower=float("nan"),
        sigma_logA_upper=float("nan"),
        sigma_curvature_lower=float("nan"),
        sigma_curvature_upper=float("nan"),
        error=str(error),
    )
