from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .algebra import solve_primitive_from_hlt
from .models import FreidbergConfig, PrimitivePoint
from .rhs import freidberg_rhs


@dataclass(frozen=True)
class FixedTeRolloutResult:
    x: np.ndarray
    T_e: np.ndarray
    H_p: np.ndarray
    L_p: np.ndarray
    mach: np.ndarray
    method: str
    failed_at_interval: int | None = None
    failure_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.failed_at_interval is None

    def finite_prefix_size(self) -> int:
        finite = np.isfinite(self.H_p) & np.isfinite(self.L_p) & np.isfinite(self.mach)
        if not np.any(finite):
            return 0
        invalid = np.flatnonzero(~finite)
        return int(invalid[0]) if invalid.size else int(finite.size)

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "succeeded": self.succeeded,
            "failed_at_interval": self.failed_at_interval,
            "failure_message": self.failure_message,
            "finite_prefix_size": self.finite_prefix_size(),
            "x": self.x.tolist(),
            "T_e": self.T_e.tolist(),
            "H_p": self.H_p.tolist(),
            "L_p": self.L_p.tolist(),
            "mach": self.mach.tolist(),
        }


def _solve_point(
    *,
    x: float,
    H_p: float,
    L_p: float,
    T_e: float,
    config: FreidbergConfig,
    mach_hint: float | None,
    branch: str,
    closure_tolerance_K: float,
) -> PrimitivePoint:
    point, _ = solve_primitive_from_hlt(
        H_p=H_p,
        L_p=L_p,
        T_e=T_e,
        config=config,
        x=x,
        branch=branch,
        mach_hint=mach_hint,
        tolerance_K=closure_tolerance_K,
    )
    return point


def integrate_fixed_te(
    *,
    x: np.ndarray,
    T_e: np.ndarray,
    H_p0: float,
    L_p0: float,
    config: FreidbergConfig,
    mach_hints: np.ndarray | None = None,
    method: str = "heun",
    branch: str = "subsonic",
    closure_tolerance_K: float = 1e-3,
) -> FixedTeRolloutResult:
    x_arr = np.asarray(x, dtype=float)
    te_arr = np.asarray(T_e, dtype=float)
    if x_arr.shape != te_arr.shape:
        raise ValueError("x and T_e arrays must have identical shape")
    if x_arr.size < 2:
        raise ValueError("at least two x nodes are required")
    if not np.all(np.diff(x_arr) > 0.0):
        raise ValueError("x grid must be strictly increasing")
    hints = None if mach_hints is None else np.asarray(mach_hints, dtype=float)
    if hints is not None and hints.shape != x_arr.shape:
        raise ValueError("mach_hints must match x shape")
    if method not in {"euler", "heun"}:
        raise ValueError("method must be 'euler' or 'heun'")

    H = np.full_like(x_arr, np.nan, dtype=float)
    L = np.full_like(x_arr, np.nan, dtype=float)
    mach = np.full_like(x_arr, np.nan, dtype=float)
    H[0] = float(H_p0)
    L[0] = float(L_p0)
    mach[0] = float(hints[0]) if hints is not None and math.isfinite(float(hints[0])) else np.nan

    failed_at: int | None = None
    failure_message: str | None = None
    for idx in range(x_arr.size - 1):
        dx = float(x_arr[idx + 1] - x_arr[idx])
        hint_i = None
        if hints is not None and math.isfinite(float(hints[idx])):
            hint_i = float(hints[idx])
        elif math.isfinite(float(mach[idx])):
            hint_i = float(mach[idx])
        try:
            point_i = _solve_point(
                x=float(x_arr[idx]),
                H_p=float(H[idx]),
                L_p=float(L[idx]),
                T_e=float(te_arr[idx]),
                config=config,
                mach_hint=hint_i,
                branch=branch,
                closure_tolerance_K=closure_tolerance_K,
            )
            dH_i, dL_i = freidberg_rhs(point_i, config)
            if method == "euler":
                next_H = H[idx] + dx * dH_i
                next_L = L[idx] + dx * dL_i
            else:
                predictor_H = H[idx] + dx * dH_i
                predictor_L = L[idx] + dx * dL_i
                hint_next = None if hints is None else float(hints[idx + 1])
                predictor = _solve_point(
                    x=float(x_arr[idx + 1]),
                    H_p=float(predictor_H),
                    L_p=float(predictor_L),
                    T_e=float(te_arr[idx + 1]),
                    config=config,
                    mach_hint=hint_next,
                    branch=branch,
                    closure_tolerance_K=closure_tolerance_K,
                )
                dH_p, dL_p = freidberg_rhs(predictor, config)
                next_H = H[idx] + 0.5 * dx * (dH_i + dH_p)
                next_L = L[idx] + 0.5 * dx * (dL_i + dL_p)
            point_next = _solve_point(
                x=float(x_arr[idx + 1]),
                H_p=float(next_H),
                L_p=float(next_L),
                T_e=float(te_arr[idx + 1]),
                config=config,
                mach_hint=None if hints is None else float(hints[idx + 1]),
                branch=branch,
                closure_tolerance_K=closure_tolerance_K,
            )
            H[idx + 1] = float(next_H)
            L[idx + 1] = float(next_L)
            mach[idx] = float(point_i.mach)
            mach[idx + 1] = float(point_next.mach)
        except Exception as exc:  # noqa: BLE001 - keep the first failed interval as a diagnostic artifact.
            failed_at = idx
            failure_message = str(exc)
            break

    return FixedTeRolloutResult(
        x=x_arr,
        T_e=te_arr,
        H_p=H,
        L_p=L,
        mach=mach,
        method=method,
        failed_at_interval=failed_at,
        failure_message=failure_message,
    )


def compare_rollout_to_reference(
    result: FixedTeRolloutResult,
    *,
    H_p_reference: np.ndarray,
    L_p_reference: np.ndarray,
    config: FreidbergConfig,
) -> dict[str, float | int | bool | None]:
    H_ref = np.asarray(H_p_reference, dtype=float)
    L_ref = np.asarray(L_p_reference, dtype=float)
    if H_ref.shape != result.H_p.shape or L_ref.shape != result.L_p.shape:
        raise ValueError("reference arrays must match rollout shape")
    finite = np.isfinite(result.H_p) & np.isfinite(result.L_p)
    if not np.any(finite):
        return {
            "succeeded": result.succeeded,
            "failed_at_interval": result.failed_at_interval,
            "finite_prefix_size": 0,
            "max_abs_H_error_MW": float("nan"),
            "max_abs_L_error": float("nan"),
            "final_H_error_MW": float("nan"),
            "final_L_error": float("nan"),
        }
    A0 = config.inlet_area_m2
    last = int(np.flatnonzero(finite)[-1])
    return {
        "succeeded": result.succeeded,
        "failed_at_interval": result.failed_at_interval,
        "finite_prefix_size": result.finite_prefix_size(),
        "max_abs_H_error_MW": float(np.nanmax(np.abs(result.H_p[finite] - H_ref[finite])) * A0 / 1e6),
        "max_abs_L_error": float(np.nanmax(np.abs(result.L_p[finite] - L_ref[finite]))),
        "final_H_error_MW": float((result.H_p[last] - H_ref[last]) * A0 / 1e6),
        "final_L_error": float(result.L_p[last] - L_ref[last]),
    }
