from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import casadi as ca
import numpy as np

from v6_casadi.optimize_area_profile_casadi_v6 import (
    FeasibilityThresholds,
    InletConstants,
    _compute_feasibility_diagnostics,
    _evaluate_profile_numeric,
    _jsonify_stats,
    _make_stage_function,
)
from v6_firedrake_reduced.design import load_case_config
from v6_firedrake_reduced.geometry import LogAreaSplineControl

from .policy import AnchorState, _physics_params
from .reachability_common import (
    anchor_payload,
    json_default,
    load_anchor_json,
    load_profile,
    load_profile_anchor,
    save_profile_npz,
    write_csv,
    write_json,
)


def _case_name(length: float) -> str:
    return f"L_{float(length):.6g}".replace("-", "m").replace(".", "p")


def _anchor_vector(anchor: AnchorState, *, config) -> np.ndarray:
    return np.array(
        [
            float(anchor.state.n_p),
            float(anchor.state.T_e),
            float(anchor.state.area(config)),
        ],
        dtype=float,
    )


def _case_inlet_constants(config) -> InletConstants:
    params = _physics_params(config)
    v_ref = float(params.dot_N) / max(float(config.design.n_p_in) * float(config.area_scale_m2), 1e-300)
    return InletConstants(
        seed_fraction=float(params.seed_fraction),
        seed_mode="case_config",
        dot_N=float(params.dot_N),
        I_0=float(params.I_0),
        v_in=v_ref,
    )


def _load_target_anchor(args: argparse.Namespace, *, config) -> AnchorState:
    if args.target_anchor_json is not None:
        return load_anchor_json(Path(args.target_anchor_json), config=config)
    return load_profile_anchor(
        None if args.target_profile_npz is None else Path(args.target_profile_npz),
        index=int(args.target_profile_index),
        config=config,
        source=str(args.target_profile_npz or f"{config.case}:built_in_profile"),
    )


def _interval_sigma_from_area(x: np.ndarray, A: np.ndarray, *, fallback: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float).reshape(-1)
    if x.size < 2:
        return np.zeros(0, dtype=float)
    dx = np.diff(x)
    sigma = np.full(x.size - 1, float(fallback), dtype=float)
    valid = np.abs(dx) > 1e-300
    sigma[valid] = np.diff(np.log(np.maximum(A, 1e-300)))[valid] / dx[valid]
    return sigma


def _warm_start_arrays(
    *,
    source: AnchorState,
    target: AnchorState,
    config,
    length: float,
    n_intervals: int,
    warm_profile_npz: Path | None,
    sigma_min: float,
    sigma_max: float,
) -> dict[str, np.ndarray]:
    x_nodes = np.linspace(0.0, float(length), int(n_intervals) + 1, dtype=float)
    source_vec = _anchor_vector(source, config=config)
    target_vec = _anchor_vector(target, config=config)

    if warm_profile_npz is None:
        alpha = x_nodes / max(float(length), 1e-300)
        n_p = (1.0 - alpha) * source_vec[0] + alpha * target_vec[0]
        T_e = (1.0 - alpha) * source_vec[1] + alpha * target_vec[1]
        logA = (1.0 - alpha) * math.log(max(source_vec[2], 1e-300)) + alpha * math.log(
            max(target_vec[2], 1e-300)
        )
        A = np.exp(logA)
        sigma = _interval_sigma_from_area(x_nodes, A)
        return {
            "x": x_nodes,
            "n_p": n_p,
            "T_e": T_e,
            "A": A,
            "sigma_logA": np.clip(sigma, float(sigma_min), float(sigma_max)),
            "source": np.asarray(["linear_endpoint"], dtype=object),
        }

    profile = load_profile(Path(warm_profile_npz), case=config.case)
    x_old = np.asarray(profile["x"], dtype=float).reshape(-1)
    order = np.argsort(x_old)
    x_old = x_old[order]
    x_old = x_old - float(x_old[0])
    old_span = float(x_old[-1]) if x_old.size > 1 else 0.0
    if old_span <= 0.0:
        raise ValueError(f"{warm_profile_npz} has a degenerate x grid.")
    x_old = x_old / old_span * float(length)
    n_p_old = np.asarray(profile["n_p"], dtype=float).reshape(-1)[order]
    T_e_old = np.asarray(profile["T_e"], dtype=float).reshape(-1)[order]
    A_old = np.asarray(profile["A"], dtype=float).reshape(-1)[order]
    n_p = np.interp(x_nodes, x_old, n_p_old)
    T_e = np.interp(x_nodes, x_old, T_e_old)
    A = np.interp(x_nodes, x_old, A_old)
    n_p[0], T_e[0], A[0] = source_vec
    n_p[-1], T_e[-1], A[-1] = target_vec
    sigma = _interval_sigma_from_area(x_nodes, A)
    return {
        "x": x_nodes,
        "n_p": n_p,
        "T_e": T_e,
        "A": A,
        "sigma_logA": np.clip(sigma, float(sigma_min), float(sigma_max)),
        "source": np.asarray([str(warm_profile_npz)], dtype=object),
    }


def _safe_value(value_fn, expr, *, default=None):
    try:
        return value_fn(expr)
    except Exception:
        return default


def _dual_array(opti: ca.Opti, value_fn, handles: list[ca.MX]) -> np.ndarray:
    if not handles:
        return np.zeros(0, dtype=float)
    exprs = []
    for handle in handles:
        dual_expr = opti.dual(handle)
        exprs.append(ca.reshape(dual_expr, dual_expr.numel(), 1))
    return np.asarray(value_fn(ca.vertcat(*exprs)), dtype=float).reshape(-1)


def _summarize_array(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "size": int(arr.size),
            "finite_count": 0,
            "max_abs": None,
            "l1": None,
            "active_fraction_gt_1e_8": None,
        }
    max_abs_idx = int(np.nanargmax(np.abs(arr)))
    return {
        "size": int(arr.size),
        "finite_count": int(finite.size),
        "max_abs": float(np.nanmax(np.abs(arr))),
        "max_abs_index": max_abs_idx,
        "l1": float(np.nansum(np.abs(arr))),
        "active_fraction_gt_1e_8": float(np.mean(np.abs(finite) > 1e-8)),
    }


def _midpoint_path_values(
    *,
    transcription: str,
    x_nodes: np.ndarray,
    n_p: np.ndarray,
    T_e: np.ndarray,
    A: np.ndarray,
    sigma_logA: np.ndarray,
    stage_fun: ca.Function,
) -> dict[str, np.ndarray]:
    n_intervals = int(np.asarray(sigma_logA).size)
    T_p = np.full(n_intervals, np.nan, dtype=float)
    G = np.full(n_intervals, np.nan, dtype=float)
    mach = np.full(n_intervals, np.nan, dtype=float)
    x_mid = np.full(n_intervals, np.nan, dtype=float)
    for k in range(n_intervals):
        dx = float(x_nodes[k + 1] - x_nodes[k])
        xk = np.array([n_p[k], T_e[k], A[k]], dtype=float)
        xkp1 = np.array([n_p[k + 1], T_e[k + 1], A[k + 1]], dtype=float)
        out_k = np.asarray(stage_fun(xk, float(sigma_logA[k])), dtype=float).reshape(-1)
        out_kp1 = np.asarray(stage_fun(xkp1, float(sigma_logA[k])), dtype=float).reshape(-1)
        if transcription == "hermite-simpson":
            mid_state = 0.5 * (xk + xkp1) + 0.125 * dx * np.array(
                [out_k[0] - out_kp1[0], out_k[1] - out_kp1[1], out_k[2] - out_kp1[2]],
                dtype=float,
            )
        else:
            mid_state = 0.5 * (xk + xkp1)
        out_mid = np.asarray(stage_fun(mid_state, float(sigma_logA[k])), dtype=float).reshape(-1)
        x_mid[k] = 0.5 * (float(x_nodes[k]) + float(x_nodes[k + 1]))
        T_p[k] = float(out_mid[3])
        mach[k] = float(out_mid[12])
        G[k] = float(out_mid[13])
    return {"x_mid": x_mid, "T_p_mid": T_p, "G_mid": G, "mach_mid": mach}


def _compute_policy_euler_diagnostics(
    *,
    x_nodes: np.ndarray,
    n_p: np.ndarray,
    T_e: np.ndarray,
    A: np.ndarray,
    sigma_logA: np.ndarray,
    T_p: np.ndarray,
    mach: np.ndarray,
    velikhov_margin: np.ndarray,
    stage_fun: ca.Function,
    inlet_target: tuple[float, float, float],
    state_bounds: dict[str, float | None],
    sigma_bounds: tuple[float, float],
    thresholds: FeasibilityThresholds,
) -> dict[str, Any]:
    np_scale = float(inlet_target[0])
    te_scale = float(max(inlet_target[1], 1.0))
    A_scale = float(inlet_target[2])
    n_intervals = int(np.asarray(sigma_logA).size)
    defects = np.zeros((n_intervals, 3), dtype=float)
    for k in range(n_intervals):
        dx = float(x_nodes[k + 1] - x_nodes[k])
        xk = np.array([n_p[k], T_e[k], A[k]], dtype=float)
        out_k = np.asarray(stage_fun(xk, float(sigma_logA[k])), dtype=float).reshape(-1)
        defects[k, 0] = (float(n_p[k + 1]) - float(n_p[k])) / np_scale - dx * float(out_k[0]) / np_scale
        defects[k, 1] = (float(T_e[k + 1]) - float(T_e[k])) / te_scale - dx * float(out_k[1]) / te_scale
        defects[k, 2] = math.log(max(float(A[k + 1]), 1e-300) / max(float(A[k]), 1e-300)) - dx * float(
            sigma_logA[k]
        )

    initial_residual = np.array(
        [n_p[0] - inlet_target[0], T_e[0] - inlet_target[1], A[0] - inlet_target[2]],
        dtype=float,
    )
    initial_residual_scaled = np.array(
        [initial_residual[0] / np_scale, initial_residual[1] / te_scale, initial_residual[2] / A_scale],
        dtype=float,
    )
    finite_profile = bool(
        np.all(np.isfinite(n_p))
        and np.all(np.isfinite(T_e))
        and np.all(np.isfinite(A))
        and np.all(np.isfinite(T_p))
        and np.all(np.isfinite(mach))
        and np.all(np.isfinite(velikhov_margin))
        and np.all(np.isfinite(defects))
    )
    tp_min = float(np.nanmin(T_p))
    margin_min = float(np.nanmin(velikhov_margin))
    mach_min_val = float(np.nanmin(mach))
    mach_max_val = float(np.nanmax(mach))
    finite_margin_mask = np.isfinite(velikhov_margin)
    if np.any(finite_margin_mask):
        margin_indices = np.flatnonzero(finite_margin_mask)
        local_idx = int(np.argmin(np.asarray(velikhov_margin)[finite_margin_mask]))
        margin_min_index = int(margin_indices[local_idx])
        margin_min_x = float(np.asarray(x_nodes, dtype=float)[margin_min_index])
        margin_lt_threshold_fraction = float(np.mean(np.asarray(velikhov_margin)[finite_margin_mask] < 1e-3))
    else:
        margin_min_index = -1
        margin_min_x = float("nan")
        margin_lt_threshold_fraction = float("nan")

    state_np_min = float(state_bounds["np_floor"])
    state_np_max = float(state_bounds["np_ceil"])
    te_min = float(state_bounds["te_floor"])
    te_max = float(state_bounds["te_ceil"])
    A_min = float(state_bounds["A_floor"])
    A_max = float(state_bounds["A_ceil"])
    tp_floor = float(state_bounds["tp_floor"])
    margin_floor = float(state_bounds["margin_floor"])
    mach_floor = state_bounds["mach_floor"]
    mach_ceil = state_bounds["mach_ceil"]

    violations = [
        max(0.0, float(np.nanmax(state_np_min - n_p))),
        max(0.0, float(np.nanmax(n_p - state_np_max))),
        max(0.0, float(np.nanmax(te_min - T_e))),
        max(0.0, float(np.nanmax(T_e - te_max))),
        max(0.0, float(np.nanmax(A_min - A))),
        max(0.0, float(np.nanmax(A - A_max))),
        max(0.0, tp_floor - tp_min),
        max(0.0, margin_floor - margin_min),
    ]
    if mach_floor is not None:
        violations.append(max(0.0, float(mach_floor) - mach_min_val))
    if mach_ceil is not None:
        violations.append(max(0.0, mach_max_val - float(mach_ceil)))

    defect_inf = float(np.nanmax(np.abs(defects))) if defects.size else 0.0
    defect_rms = float(np.sqrt(np.nanmean(defects * defects))) if defects.size else 0.0
    boundary_inf = float(np.nanmax(np.abs(initial_residual_scaled)))
    max_constraint_violation = float(max(violations)) if violations else 0.0
    sigma = np.asarray(sigma_logA, dtype=float).reshape(-1)
    sigma_lower, sigma_upper = sigma_bounds
    sigma_slew = np.diff(sigma) / max(float(np.mean(np.diff(x_nodes))), 1e-300) if sigma.size > 1 else np.zeros(0)
    acceptable = bool(
        finite_profile
        and defect_inf <= float(thresholds.defect_inf_tol)
        and defect_rms <= float(thresholds.defect_rms_tol)
        and boundary_inf <= float(thresholds.boundary_inf_tol)
        and max_constraint_violation <= float(thresholds.path_slack_tol)
    )
    return {
        "finite_profile": finite_profile,
        "dynamic_defect_inf": defect_inf,
        "dynamic_defect_rms": defect_rms,
        "midpoint_defect_inf": 0.0,
        "midpoint_defect_rms": 0.0,
        "boundary_residual_inf": boundary_inf,
        "initial_state_residual": initial_residual.tolist(),
        "tp_min": tp_min,
        "velikhov_margin_min": margin_min,
        "velikhov_margin_min_index": margin_min_index,
        "velikhov_margin_min_x_m": margin_min_x,
        "velikhov_margin_lt_1e_3_fraction": margin_lt_threshold_fraction,
        "mach_min": mach_min_val,
        "mach_max": mach_max_val,
        "max_constraint_violation": max_constraint_violation,
        "margin_slack_max": 0.0,
        "margin_slack_l1": 0.0,
        "margin_slack_active_fraction": 0.0,
        "sigma_slew_rms": float(np.sqrt(np.mean(sigma_slew * sigma_slew))) if sigma_slew.size else 0.0,
        "sigma_near_lower_bound_fraction": float(np.mean(sigma <= float(sigma_lower) + 1e-8)) if sigma.size else 0.0,
        "sigma_near_upper_bound_fraction": float(np.mean(sigma >= float(sigma_upper) - 1e-8)) if sigma.size else 0.0,
        "regularity_ok": True,
        "acceptable": acceptable,
        "thresholds": {
            "defect_inf_tol": float(thresholds.defect_inf_tol),
            "defect_rms_tol": float(thresholds.defect_rms_tol),
            "boundary_inf_tol": float(thresholds.boundary_inf_tol),
            "path_slack_tol": float(thresholds.path_slack_tol),
            "velikhov_margin_activity_threshold": 1e-3,
        },
    }


def _activity_summary(
    *,
    margins: dict[str, np.ndarray],
    duals: dict[str, np.ndarray],
    active_tols: dict[str, float],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for name, margin in margins.items():
        arr = np.asarray(margin, dtype=float).reshape(-1)
        finite = arr[np.isfinite(arr)]
        tol = float(active_tols.get(name, active_tols.get("*", 1e-8)))
        if finite.size:
            near = finite <= tol
            min_margin = float(np.nanmin(finite))
            near_count = int(np.sum(near))
            near_fraction = float(np.mean(near))
        else:
            min_margin = float("nan")
            near_count = 0
            near_fraction = float("nan")
        dual_summary = _summarize_array(duals.get(name, np.zeros(0, dtype=float)))
        dual_arr = np.asarray(duals.get(name, np.zeros(0, dtype=float)), dtype=float).reshape(-1)
        active_dual_l1 = None
        inactive_dual_l1 = None
        if finite.size and dual_arr.size == arr.size:
            finite_mask = np.isfinite(arr) & np.isfinite(dual_arr)
            near_full = np.zeros(arr.size, dtype=bool)
            near_full[finite_mask] = arr[finite_mask] <= tol
            active_dual_l1 = float(np.nansum(np.abs(dual_arr[near_full])))
            inactive_dual_l1 = float(np.nansum(np.abs(dual_arr[finite_mask & ~near_full])))
        row = {
            "constraint": name,
            "size": int(arr.size),
            "active_tol": tol,
            "min_margin": min_margin,
            "near_active_count": near_count,
            "near_active_fraction": near_fraction,
            "dual_max_abs": dual_summary.get("max_abs"),
            "dual_l1": dual_summary.get("l1"),
            "active_dual_l1": active_dual_l1,
            "inactive_dual_l1": inactive_dual_l1,
            "dual_active_fraction_gt_1e_8": dual_summary.get("active_fraction_gt_1e_8"),
        }
        by_name[name] = row
        rows.append(row)

    def dual_key(row: dict[str, Any]) -> float:
        if int(row.get("near_active_count", 0)) <= 0:
            return -1.0
        value = row.get("active_dual_l1")
        return -1.0 if value is None or not np.isfinite(float(value)) else float(value)

    def support_key(row: dict[str, Any]) -> tuple[int, float]:
        return (int(row.get("near_active_count", 0)), dual_key(row))

    inequality_rows = [row for row in rows if not str(row["constraint"]).endswith("_eq")]
    ranked_by_dual = sorted(inequality_rows, key=dual_key, reverse=True)
    ranked_by_support = sorted(inequality_rows, key=support_key, reverse=True)
    return {
        "rows": rows,
        "by_name": by_name,
        "ranked_by_dual_l1": ranked_by_dual[:10],
        "ranked_by_primal_support": ranked_by_support[:10],
        "dominant_dual_boundary": ranked_by_dual[0]["constraint"] if ranked_by_dual and dual_key(ranked_by_dual[0]) > 0 else None,
        "dominant_primal_boundary": ranked_by_support[0]["constraint"]
        if ranked_by_support and int(ranked_by_support[0].get("near_active_count", 0)) > 0
        else None,
    }


def _solve_one_length(
    *,
    config,
    source: AnchorState,
    target: AnchorState,
    length: float,
    n_intervals: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if length <= 0.0:
        raise ValueError("length must be positive.")
    if n_intervals < 2:
        raise ValueError("--n-intervals must be at least 2.")

    inlet = _case_inlet_constants(config)
    stage = _make_stage_function(
        dot_N=float(inlet.dot_N),
        I_0=float(inlet.I_0),
        seed_fraction=float(inlet.seed_fraction),
        B=float(config.B_T),
    )
    source_vec = _anchor_vector(source, config=config)
    target_vec = _anchor_vector(target, config=config)
    x_nodes = np.linspace(0.0, float(length), int(n_intervals) + 1, dtype=float)
    dx = float(length) / int(n_intervals)

    np_scale = max(float(source_vec[0]), 1e10)
    te_scale = max(float(source_vec[1]), 1.0)
    A_scale = max(float(source_vec[2]), 1e-12)
    np_floor = max(min(float(source_vec[0]), float(target_vec[0])) * float(args.np_min_ratio), 1e10)
    np_ceil = max(float(source_vec[0]), float(target_vec[0])) * float(args.np_max_ratio)
    te_floor = float(args.te_min)
    te_ceil = max(float(source_vec[1]), float(target_vec[1])) * float(args.te_max_ratio)
    area_box_floor = min(float(source_vec[2]), float(target_vec[2])) * float(args.A_min_ratio)
    area_box_ceil = max(float(source_vec[2]), float(target_vec[2])) * float(args.A_max_ratio)
    if bool(args.use_logA_bounds):
        logA_floor = LogAreaSplineControl.lower_bound()
        logA_ceil = LogAreaSplineControl.upper_bound()
        area_box_floor = max(area_box_floor, float(config.area_scale_m2) * math.exp(float(logA_floor)))
        area_box_ceil = min(area_box_ceil, float(config.area_scale_m2) * math.exp(float(logA_ceil)))
    sigma_min = float(args.sigma_min)
    sigma_max = float(args.sigma_max)
    if sigma_min > sigma_max:
        raise ValueError("--sigma-min cannot exceed --sigma-max.")

    warm = _warm_start_arrays(
        source=source,
        target=target,
        config=config,
        length=float(length),
        n_intervals=int(n_intervals),
        warm_profile_npz=None if args.warm_profile_npz is None else Path(args.warm_profile_npz),
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    warm["n_p"] = np.clip(np.asarray(warm["n_p"], dtype=float), np_floor, np_ceil)
    warm["T_e"] = np.clip(np.asarray(warm["T_e"], dtype=float), te_floor, te_ceil)
    warm["A"] = np.clip(np.asarray(warm["A"], dtype=float), area_box_floor, area_box_ceil)
    warm["n_p"][0], warm["T_e"][0], warm["A"][0] = source_vec
    warm["n_p"][-1], warm["T_e"][-1], warm["A"][-1] = target_vec

    opti = ca.Opti()
    X = opti.variable(3, int(n_intervals) + 1)
    U = opti.variable(1, int(n_intervals))
    dual_handles: dict[str, list[ca.MX]] = {
        "source_endpoint_eq": [],
        "target_endpoint_eq": [],
        "n_p_lower_node": [],
        "n_p_upper_node": [],
        "T_e_lower_node": [],
        "T_e_upper_node": [],
        "A_lower_node": [],
        "A_upper_node": [],
        "sigma_lower_interval": [],
        "sigma_upper_interval": [],
        "sigma_step_lower": [],
        "sigma_step_upper": [],
        "Tp_lower_node": [],
        "Tp_lower_mid": [],
        "G_lower_node": [],
        "G_lower_mid": [],
        "Mach_lower_node": [],
        "Mach_lower_mid": [],
        "Mach_upper_node": [],
        "Mach_upper_mid": [],
    }

    def subject_to(name: str | None, expr):
        opti.subject_to(expr)
        if name is not None:
            dual_handles[name].append(expr)

    n_hat = X[0, :]
    te_hat = X[1, :]
    A_hat = X[2, :]
    source_hat = ca.DM([source_vec[0] / np_scale, source_vec[1] / te_scale, source_vec[2] / A_scale])
    target_hat = ca.DM([target_vec[0] / np_scale, target_vec[1] / te_scale, target_vec[2] / A_scale])
    subject_to("source_endpoint_eq", X[:, 0] == source_hat)
    subject_to("target_endpoint_eq", X[:, -1] == target_hat)
    subject_to("n_p_lower_node", n_hat >= np_floor / np_scale)
    subject_to("n_p_upper_node", n_hat <= np_ceil / np_scale)
    subject_to("T_e_lower_node", te_hat >= te_floor / te_scale)
    subject_to("T_e_upper_node", te_hat <= te_ceil / te_scale)
    subject_to("A_lower_node", A_hat >= area_box_floor / A_scale)
    subject_to("A_upper_node", A_hat <= area_box_ceil / A_scale)
    subject_to("sigma_lower_interval", U >= sigma_min)
    subject_to("sigma_upper_interval", U <= sigma_max)
    if args.sigma_step_max is not None and int(n_intervals) > 1:
        dU = U[:, 1:] - U[:, :-1]
        subject_to("sigma_step_lower", dU >= -float(args.sigma_step_max))
        subject_to("sigma_step_upper", dU <= float(args.sigma_step_max))

    objective = 0.0
    has_objective_terms = False
    if float(args.smooth_weight) > 0.0:
        objective += float(args.smooth_weight) * dx * ca.sumsqr(U)
        has_objective_terms = True
    if float(args.control_slew_weight) > 0.0 and int(n_intervals) > 1:
        objective += float(args.control_slew_weight) * dx * ca.sumsqr((U[:, 1:] - U[:, :-1]) / dx)
        has_objective_terms = True
    if float(args.control_curvature_weight) > 0.0 and int(n_intervals) > 2:
        objective += float(args.control_curvature_weight) * dx * ca.sumsqr(
            (U[:, 2:] - 2.0 * U[:, 1:-1] + U[:, :-2]) / (dx * dx)
        )
        has_objective_terms = True
    if float(args.state_curvature_weight) > 0.0 and int(n_intervals) > 2:
        objective += float(args.state_curvature_weight) * dx * (
            ca.sumsqr((n_hat[2:] - 2.0 * n_hat[1:-1] + n_hat[:-2]) / (dx * dx))
            + ca.sumsqr((te_hat[2:] - 2.0 * te_hat[1:-1] + te_hat[:-2]) / (dx * dx))
            + ca.sumsqr((A_hat[2:] - 2.0 * A_hat[1:-1] + A_hat[:-2]) / (dx * dx))
        )
        has_objective_terms = True
    if not has_objective_terms:
        objective = 1e-12 * ca.sumsqr(U)

    for k in range(int(n_intervals)):
        xk_phys = ca.vertcat(np_scale * n_hat[k], te_scale * te_hat[k], A_scale * A_hat[k])
        xkp1_phys = ca.vertcat(np_scale * n_hat[k + 1], te_scale * te_hat[k + 1], A_scale * A_hat[k + 1])
        out_k = stage(xk_phys, U[0, k])
        out_kp1 = stage(xkp1_phys, U[0, k])
        f_k = ca.vertcat(out_k[0] / np_scale, out_k[1] / te_scale, out_k[2] / A_scale)
        f_kp1 = ca.vertcat(out_kp1[0] / np_scale, out_kp1[1] / te_scale, out_kp1[2] / A_scale)
        if str(args.transcription) == "policy-euler":
            subject_to(None, n_hat[k + 1] == n_hat[k] + dx * out_k[0] / np_scale)
            subject_to(None, te_hat[k + 1] == te_hat[k] + dx * out_k[1] / te_scale)
            subject_to(None, ca.log(A_hat[k + 1]) - ca.log(A_hat[k]) == dx * U[0, k])
            mid_state = 0.5 * (xk_phys + xkp1_phys)
        elif str(args.transcription) == "hermite-simpson":
            mid_state = 0.5 * (xk_phys + xkp1_phys) + 0.125 * dx * ca.vertcat(
                out_k[0] - out_kp1[0],
                out_k[1] - out_kp1[1],
                out_k[2] - out_kp1[2],
            )
            out_mid_for_dyn = stage(mid_state, U[0, k])
            f_mid = ca.vertcat(
                out_mid_for_dyn[0] / np_scale,
                out_mid_for_dyn[1] / te_scale,
                out_mid_for_dyn[2] / A_scale,
            )
            subject_to(None, X[:, k + 1] == X[:, k] + dx / 6.0 * (f_k + 4.0 * f_mid + f_kp1))
        else:
            mid_state = 0.5 * (xk_phys + xkp1_phys)
            subject_to(None, X[:, k + 1] == X[:, k] + 0.5 * dx * (f_k + f_kp1))

        subject_to("Tp_lower_node", out_k[3] >= float(args.tp_floor))
        subject_to("G_lower_node", out_k[13] >= float(args.g_floor))
        if args.mach_min is not None:
            subject_to("Mach_lower_node", out_k[12] >= float(args.mach_min))
        if args.mach_max is not None:
            subject_to("Mach_upper_node", out_k[12] <= float(args.mach_max))

        out_mid = stage(mid_state, U[0, k])
        subject_to("Tp_lower_mid", out_mid[3] >= float(args.tp_floor))
        subject_to("G_lower_mid", out_mid[13] >= float(args.g_floor))
        if args.mach_min is not None:
            subject_to("Mach_lower_mid", out_mid[12] >= float(args.mach_min))
        if args.mach_max is not None:
            subject_to("Mach_upper_mid", out_mid[12] <= float(args.mach_max))

    out_end = stage(
        ca.vertcat(np_scale * n_hat[-1], te_scale * te_hat[-1], A_scale * A_hat[-1]),
        U[0, -1],
    )
    subject_to("Tp_lower_node", out_end[3] >= float(args.tp_floor))
    subject_to("G_lower_node", out_end[13] >= float(args.g_floor))
    if args.mach_min is not None:
        subject_to("Mach_lower_node", out_end[12] >= float(args.mach_min))
    if args.mach_max is not None:
        subject_to("Mach_upper_node", out_end[12] <= float(args.mach_max))

    opti.minimize(objective)
    opti.set_initial(X[0, :], np.asarray(warm["n_p"], dtype=float) / np_scale)
    opti.set_initial(X[1, :], np.asarray(warm["T_e"], dtype=float) / te_scale)
    opti.set_initial(X[2, :], np.asarray(warm["A"], dtype=float) / A_scale)
    opti.set_initial(U, np.asarray(warm["sigma_logA"], dtype=float).reshape(1, -1))
    opti.solver(
        "ipopt",
        {
            "expand": True,
            "print_time": 0,
            "ipopt.print_level": int(args.ipopt_print_level),
            "ipopt.max_iter": int(args.ipopt_max_iter),
            "ipopt.tol": float(args.ipopt_tol),
            "ipopt.acceptable_tol": max(float(args.ipopt_tol) * 10.0, 1e-6),
            "ipopt.sb": "yes",
        },
        {},
    )

    sol = None
    value_fn = None
    solve_error = ""
    try:
        sol = opti.solve_limited()
        value_fn = sol.value
    except RuntimeError as exc:
        solve_error = str(exc)
        value_fn = opti.debug.value

    stats = opti.stats()
    if value_fn is None:
        value_fn = opti.debug.value
    X_sol = np.asarray(value_fn(X), dtype=float)
    U_sol = np.asarray(value_fn(U), dtype=float).reshape(-1)
    n_p_sol = np_scale * X_sol[0, :]
    T_e_sol = te_scale * X_sol[1, :]
    A_sol = A_scale * X_sol[2, :]
    profile = _evaluate_profile_numeric(
        x=x_nodes,
        n_p=n_p_sol,
        T_e=T_e_sol,
        A=A_sol,
        inlet=inlet,
        B=float(config.B_T),
        sigma_logA=U_sol,
    )
    thresholds = FeasibilityThresholds(
        defect_inf_tol=float(args.defect_inf_tol),
        defect_rms_tol=float(args.defect_rms_tol),
        boundary_inf_tol=float(args.boundary_inf_tol),
        path_slack_tol=float(args.path_slack_tol),
    )
    state_bounds = {
        "np_floor": np_floor,
        "np_ceil": np_ceil,
        "te_floor": te_floor,
        "te_ceil": te_ceil,
        "A_floor": area_box_floor,
        "A_ceil": area_box_ceil,
        "tp_floor": float(args.tp_floor),
        "margin_floor": float(args.g_floor),
        "mach_floor": None if args.mach_min is None else float(args.mach_min),
        "mach_ceil": None if args.mach_max is None else float(args.mach_max),
    }
    if str(args.transcription) == "policy-euler":
        diagnostics = _compute_policy_euler_diagnostics(
            x_nodes=x_nodes,
            n_p=profile["n_p"],
            T_e=profile["T_e"],
            A=profile["A"],
            sigma_logA=profile["sigma_logA"],
            T_p=profile["T_p"],
            mach=profile["mach"],
            velikhov_margin=profile["velikhov_margin"],
            stage_fun=stage,
            inlet_target=(float(source_vec[0]), float(source_vec[1]), float(source_vec[2])),
            state_bounds=state_bounds,
            sigma_bounds=(sigma_min, sigma_max),
            thresholds=thresholds,
        )
    else:
        diagnostics = _compute_feasibility_diagnostics(
            transcription=str(args.transcription),
            x_nodes=x_nodes,
            n_p=profile["n_p"],
            T_e=profile["T_e"],
            A=profile["A"],
            sigma_logA=profile["sigma_logA"],
            T_p=profile["T_p"],
            mach=profile["mach"],
            velikhov_margin=profile["velikhov_margin"],
            stage_fun=stage,
            inlet_target=(float(source_vec[0]), float(source_vec[1]), float(source_vec[2])),
            state_bounds=state_bounds,
            sigma_bounds=(sigma_min, sigma_max),
            thresholds=thresholds,
        )
    terminal_residual = np.array(
        [n_p_sol[-1] - target_vec[0], T_e_sol[-1] - target_vec[1], A_sol[-1] - target_vec[2]],
        dtype=float,
    )
    terminal_residual_scaled = np.array(
        [terminal_residual[0] / np_scale, terminal_residual[1] / te_scale, terminal_residual[2] / A_scale],
        dtype=float,
    )
    terminal_inf = float(np.nanmax(np.abs(terminal_residual_scaled)))
    acceptable = bool(diagnostics["acceptable"] and terminal_inf <= float(args.boundary_inf_tol))

    dual_arrays: dict[str, np.ndarray] = {}
    dual_errors: dict[str, str] = {}
    for name, handles in dual_handles.items():
        try:
            dual_arrays[name] = _dual_array(opti, value_fn, handles)
        except Exception as exc:
            dual_arrays[name] = np.zeros(0, dtype=float)
            dual_errors[name] = str(exc)

    mid = _midpoint_path_values(
        transcription=str(args.transcription),
        x_nodes=x_nodes,
        n_p=profile["n_p"],
        T_e=profile["T_e"],
        A=profile["A"],
        sigma_logA=profile["sigma_logA"],
        stage_fun=stage,
    )
    margins: dict[str, np.ndarray] = {
        "n_p_lower_node": profile["n_p"] - np_floor,
        "n_p_upper_node": np_ceil - profile["n_p"],
        "T_e_lower_node": profile["T_e"] - te_floor,
        "T_e_upper_node": te_ceil - profile["T_e"],
        "A_lower_node": profile["A"] - area_box_floor,
        "A_upper_node": area_box_ceil - profile["A"],
        "sigma_lower_interval": profile["sigma_logA"] - sigma_min,
        "sigma_upper_interval": sigma_max - profile["sigma_logA"],
        "Tp_lower_node": profile["T_p"] - float(args.tp_floor),
        "Tp_lower_mid": mid["T_p_mid"] - float(args.tp_floor),
        "G_lower_node": profile["velikhov_margin"] - float(args.g_floor),
        "G_lower_mid": mid["G_mid"] - float(args.g_floor),
    }
    if args.mach_min is not None:
        margins["Mach_lower_node"] = profile["mach"] - float(args.mach_min)
        margins["Mach_lower_mid"] = mid["mach_mid"] - float(args.mach_min)
    if args.mach_max is not None:
        margins["Mach_upper_node"] = float(args.mach_max) - profile["mach"]
        margins["Mach_upper_mid"] = float(args.mach_max) - mid["mach_mid"]
    if args.sigma_step_max is not None and profile["sigma_logA"].size > 1:
        d_sigma = np.diff(profile["sigma_logA"])
        margins["sigma_step_lower"] = d_sigma + float(args.sigma_step_max)
        margins["sigma_step_upper"] = float(args.sigma_step_max) - d_sigma

    active_tols = {
        "*": float(args.active_tol),
        "G_lower_node": float(args.g_active_tol),
        "G_lower_mid": float(args.g_active_tol),
        "Tp_lower_node": float(args.tp_active_tol),
        "Tp_lower_mid": float(args.tp_active_tol),
        "A_lower_node": max(float(args.active_tol), 1e-8 * max(abs(area_box_floor), 1.0)),
        "A_upper_node": max(float(args.active_tol), 1e-8 * max(abs(area_box_ceil), 1.0)),
    }
    active_set = _activity_summary(margins=margins, duals=dual_arrays, active_tols=active_tols)
    objective_value = _safe_value(value_fn, objective, default=float("nan"))

    nodes: list[dict[str, Any]] = []
    sigma_node = np.concatenate([profile["sigma_logA"], profile["sigma_logA"][-1:]])
    for i in range(x_nodes.size):
        T_p = float(profile["T_p"][i])
        nodes.append(
            {
                "i": int(i),
                "x": float(x_nodes[i]),
                "n_p": float(profile["n_p"][i]),
                "T_e": float(profile["T_e"][i]),
                "T_p": T_p,
                "Delta": float(profile["T_e"][i] / max(T_p, 1e-300) - 1.0),
                "A": float(profile["A"][i]),
                "sigma_logA": float(sigma_node[i]),
                "G": float(profile["velikhov_margin"][i]),
                "mach": float(profile["mach"][i]),
                "beta": float(profile["beta"][i]),
                "Z": float(profile["Z"][i]),
            }
        )
    intervals: list[dict[str, Any]] = []
    for k in range(int(n_intervals)):
        intervals.append(
            {
                "k": int(k),
                "x_left": float(x_nodes[k]),
                "x_right": float(x_nodes[k + 1]),
                "x_mid": float(mid["x_mid"][k]),
                "sigma_logA": float(profile["sigma_logA"][k]),
                "T_p_mid": float(mid["T_p_mid"][k]),
                "G_mid": float(mid["G_mid"][k]),
                "mach_mid": float(mid["mach_mid"][k]),
            }
        )

    return {
        "ok": bool(acceptable and bool(stats.get("success", False))),
        "acceptable": acceptable,
        "solver_success": bool(stats.get("success", False)),
        "return_status": str(stats.get("return_status", "")),
        "solve_error": solve_error,
        "length_m": float(length),
        "n_intervals": int(n_intervals),
        "dx_m": dx,
        "transcription": str(args.transcription),
        "case_config": config.to_dict(),
        "source_anchor": anchor_payload(source, config=config),
        "target_anchor": anchor_payload(target, config=config),
        "inlet_constants": {
            "seed_fraction": float(inlet.seed_fraction),
            "seed_mode": str(inlet.seed_mode),
            "dot_N": float(inlet.dot_N),
            "I_0": float(inlet.I_0),
            "v_in_reference": float(inlet.v_in),
        },
        "state_bounds": {
            "np_floor": np_floor,
            "np_ceil": np_ceil,
            "te_floor": te_floor,
            "te_ceil": te_ceil,
            "A_floor": area_box_floor,
            "A_ceil": area_box_ceil,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "sigma_step_max": None if args.sigma_step_max is None else float(args.sigma_step_max),
            "tp_floor": float(args.tp_floor),
            "g_floor": float(args.g_floor),
            "mach_min": None if args.mach_min is None else float(args.mach_min),
            "mach_max": None if args.mach_max is None else float(args.mach_max),
        },
        "objective_value": float(objective_value),
        "warm_start_source": str(np.asarray(warm["source"], dtype=object).reshape(-1)[0]),
        "diagnostics": {
            **diagnostics,
            "terminal_state_residual": terminal_residual.tolist(),
            "terminal_state_residual_scaled": terminal_residual_scaled.tolist(),
            "terminal_boundary_residual_inf": terminal_inf,
            "acceptable_with_terminal": acceptable,
        },
        "active_set": active_set,
        "dual_summary": {name: _summarize_array(values) for name, values in dual_arrays.items()},
        "dual_errors": dual_errors,
        "solver_stats": _jsonify_stats(stats),
        "profile": profile,
        "midpoint": mid,
        "dual_arrays": dual_arrays,
        "nodes": nodes,
        "intervals": intervals,
    }


def _summary_payload(result: dict[str, Any]) -> dict[str, Any]:
    profile = result["profile"]
    return {
        "ok": bool(result["ok"]),
        "acceptable": bool(result["acceptable"]),
        "solver_success": bool(result["solver_success"]),
        "return_status": str(result["return_status"]),
        "length_m": float(result["length_m"]),
        "n_intervals": int(result["n_intervals"]),
        "dx_m": float(result["dx_m"]),
        "source_Delta": float(result["nodes"][0]["Delta"]),
        "target_Delta": float(result["nodes"][-1]["Delta"]),
        "forward_Delta_gain": float(result["nodes"][-1]["Delta"] - result["nodes"][0]["Delta"]),
        "min_G_node": float(np.nanmin(profile["velikhov_margin"])),
        "min_Tp_node_K": float(np.nanmin(profile["T_p"])),
        "max_mach_node": float(np.nanmax(profile["mach"])),
        "min_mach_node": float(np.nanmin(profile["mach"])),
        "terminal_boundary_residual_inf": float(result["diagnostics"]["terminal_boundary_residual_inf"]),
        "dynamic_defect_inf": float(result["diagnostics"]["dynamic_defect_inf"]),
        "dominant_dual_boundary": result["active_set"].get("dominant_dual_boundary"),
        "dominant_primal_boundary": result["active_set"].get("dominant_primal_boundary"),
    }


def _strip_arrays_for_json(result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    payload.pop("profile", None)
    payload.pop("midpoint", None)
    payload.pop("dual_arrays", None)
    return payload


def _write_case_outputs(case_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    summary = _strip_arrays_for_json(result)
    write_json(case_dir / "ipopt_endpoint_summary.json", summary)
    write_csv(case_dir / "nodes.csv", list(result["nodes"]))
    write_csv(case_dir / "intervals.csv", list(result["intervals"]))
    write_csv(case_dir / "active_sets.csv", list(result["active_set"]["rows"]))
    profile = dict(result["profile"])
    dual_arrays = {f"dual_{name}": values for name, values in result["dual_arrays"].items()}
    midpoint = {key: np.asarray(value, dtype=float) for key, value in result["midpoint"].items()}
    save_profile_npz(case_dir / "profile.npz", {**profile, **midpoint, **dual_arrays})
    return {
        "summary_json": str(case_dir / "ipopt_endpoint_summary.json"),
        "profile_npz": str(case_dir / "profile.npz"),
        "nodes_csv": str(case_dir / "nodes.csv"),
        "active_sets_csv": str(case_dir / "active_sets.csv"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use CasADi/IPOPT direct transcription to test whether a fixed source "
            "anchor can reach a fixed Freidberg target anchor in shorter channels."
        )
    )
    parser.add_argument("--case", default="freidberg_reference")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-anchor-json", type=Path, required=True)
    parser.add_argument("--target-anchor-json", type=Path, default=None)
    parser.add_argument("--target-profile-npz", type=Path, default=None)
    parser.add_argument("--target-profile-index", type=int, default=0)
    parser.add_argument("--warm-profile-npz", type=Path, default=None)
    parser.add_argument("--lengths", type=float, nargs="+", required=True)
    parser.add_argument("--n-intervals", type=int, default=80)
    parser.add_argument(
        "--transcription",
        choices=("trapezoid", "hermite-simpson", "policy-euler"),
        default="hermite-simpson",
    )
    parser.add_argument(
        "--g-floor",
        type=float,
        default=-1e-7,
        help="Velikhov margin floor. Default tolerates the Freidberg reference node's roundoff-level G<0.",
    )
    parser.add_argument("--tp-floor", type=float, default=300.0)
    parser.add_argument("--mach-min", type=float, default=None)
    parser.add_argument("--mach-max", type=float, default=None)
    parser.add_argument("--sigma-min", type=float, default=-0.5)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--sigma-step-max", type=float, default=None)
    parser.add_argument("--A-min-ratio", type=float, default=0.25)
    parser.add_argument("--A-max-ratio", type=float, default=4.0)
    parser.add_argument("--use-logA-bounds", action="store_true")
    parser.add_argument("--np-min-ratio", type=float, default=1e-6)
    parser.add_argument("--np-max-ratio", type=float, default=100.0)
    parser.add_argument("--te-min", type=float, default=100.0)
    parser.add_argument("--te-max-ratio", type=float, default=20.0)
    parser.add_argument("--smooth-weight", type=float, default=1e-8)
    parser.add_argument("--control-slew-weight", type=float, default=1e-10)
    parser.add_argument("--control-curvature-weight", type=float, default=0.0)
    parser.add_argument("--state-curvature-weight", type=float, default=0.0)
    parser.add_argument("--ipopt-max-iter", type=int, default=1000)
    parser.add_argument("--ipopt-tol", type=float, default=1e-7)
    parser.add_argument("--ipopt-print-level", type=int, default=0)
    parser.add_argument("--defect-inf-tol", type=float, default=1e-4)
    parser.add_argument("--defect-rms-tol", type=float, default=1e-5)
    parser.add_argument("--boundary-inf-tol", type=float, default=1e-8)
    parser.add_argument("--path-slack-tol", type=float, default=1e-6)
    parser.add_argument("--active-tol", type=float, default=1e-7)
    parser.add_argument("--g-active-tol", type=float, default=1e-3)
    parser.add_argument("--tp-active-tol", type=float, default=1e-3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_case_config(case=str(args.case))
    source = load_anchor_json(Path(args.source_anchor_json), config=config)
    target = _load_target_anchor(args, config=config)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "source_anchor.json", anchor_payload(source, config=config))
    write_json(out_root / "target_anchor.json", anchor_payload(target, config=config))

    aggregate_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for length in [float(v) for v in args.lengths]:
        result = _solve_one_length(
            config=config,
            source=source,
            target=target,
            length=length,
            n_intervals=int(args.n_intervals),
            args=args,
        )
        case_dir = out_root / _case_name(length)
        paths = _write_case_outputs(case_dir, result)
        row = {**_summary_payload(result), **paths}
        aggregate_rows.append(row)
        manifests.append({"length_m": float(length), "case_dir": str(case_dir), **paths})

    aggregate_rows = sorted(aggregate_rows, key=lambda row: float(row["length_m"]))
    feasible = [row for row in aggregate_rows if bool(row["ok"])]
    shortest = min(feasible, key=lambda row: float(row["length_m"])) if feasible else None
    write_csv(out_root / "ipopt_reachability_summary.csv", aggregate_rows)
    write_json(
        out_root / "ipopt_reachability_summary.json",
        {
            "case": config.case,
            "source_anchor_json": str(out_root / "source_anchor.json"),
            "target_anchor_json": str(out_root / "target_anchor.json"),
            "shortest_ok_case": shortest,
            "rows": aggregate_rows,
            "manifests": manifests,
        },
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_root),
                "n_cases": len(aggregate_rows),
                "shortest_ok_case": shortest,
                "rows": aggregate_rows,
            },
            indent=2,
            sort_keys=True,
            default=json_default,
        )
    )
    return 0 if feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())
