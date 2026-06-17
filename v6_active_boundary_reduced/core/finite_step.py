from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from v6_firedrake_reduced.design import CaseConfig

from .numba_physics import dynamic_terms_numba


PhysicsParamsFn = Callable[[CaseConfig], Any]
StateFactory = Callable[..., Any]
ClosureMetricsFn = Callable[..., dict[str, float]]
StepObjectivePayloadFn = Callable[..., dict[str, float | str]]
SolveNextStateFn = Callable[..., tuple[Any, dict[str, float | bool | int | str]]]


def rk4_rhs_mode(settings: Any) -> str:
    mode = str(getattr(settings, "rk4_rhs_mode", "raw")).strip().lower().replace("-", "_")
    aliases = {
        "raw": "raw",
        "primitive": "raw",
        "legacy": "raw",
        "log": "log",
        "log_columns": "log",
        "log_column": "log",
        "nondim": "nondim",
        "non_dim": "nondim",
        "row_scaled": "nondim",
        "scaled": "nondim",
    }
    if mode not in aliases:
        raise ValueError("rk4_rhs_mode must be 'raw', 'log', or 'nondim'.")
    return aliases[mode]


def safe_matrix_cond(matrix: np.ndarray) -> float:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(np.linalg.cond(np.asarray(matrix, dtype=float)))
    except Exception:
        return float("inf")
    return value if np.isfinite(value) else float("inf")


def safe_matrix_det(matrix: np.ndarray) -> float:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(np.linalg.det(np.asarray(matrix, dtype=float)))
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def safe_singular_values(matrix: np.ndarray) -> np.ndarray:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    except Exception:
        return np.asarray([], dtype=float)
    return np.asarray(values, dtype=float)


def row_norms(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    max_abs = np.max(np.abs(values), axis=1)
    safe_max = np.maximum(max_abs, 1.0e-300)
    scaled = values / safe_max[:, None]
    return safe_max * np.sqrt(np.sum(scaled * scaled, axis=1))


def row_normalized_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    scales = row_norms(values)
    return values / np.maximum(scales[:, None], 1.0e-300)


def row_cosine(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=float)
    row_norm_values = row_norms(values)
    denom = float(row_norm_values[0] * row_norm_values[1])
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    value = float(np.dot(values[0], values[1]) / denom)
    return value if np.isfinite(value) else float("nan")


def solve_linear_rhs(matrix: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, str, int]:
    values = np.asarray(matrix, dtype=float)
    target = np.asarray(rhs, dtype=float)
    try:
        solution = np.linalg.solve(values, target)
        method = "solve"
        rank = int(values.shape[1])
    except np.linalg.LinAlgError:
        solution, _residuals, rank_value, _singular_values = np.linalg.lstsq(values, target, rcond=None)
        method = "lstsq"
        rank = int(rank_value)
    return np.asarray(solution, dtype=float), method, rank


def _solve_2x2_fast(
    a00: float,
    a01: float,
    a10: float,
    a11: float,
    rhs0: float,
    rhs1: float,
) -> tuple[float, float]:
    det = float(a00 * a11 - a01 * a10)
    if np.isfinite(det) and det != 0.0:
        return (
            float((rhs0 * a11 - a01 * rhs1) / det),
            float((a00 * rhs1 - rhs0 * a10) / det),
        )
    matrix = np.array([[a00, a01], [a10, a11]], dtype=float)
    rhs = np.array([rhs0, rhs1], dtype=float)
    solution, _method, _rank = solve_linear_rhs(matrix, rhs)
    return float(solution[0]), float(solution[1])


def _row_norm2(a: float, b: float) -> float:
    max_abs = max(abs(float(a)), abs(float(b)))
    safe_max = float(np.maximum(max_abs, 1.0e-300))
    row_norm = safe_max * math.sqrt((float(a) / safe_max) ** 2 + (float(b) / safe_max) ** 2)
    return float(np.maximum(row_norm, 1.0e-300))


def _exp_clip(value: float) -> float:
    return math.exp(min(max(float(value), -700.0), 700.0))


def evaluate_sigma(
    *,
    current: Any,
    sigma: float,
    dx: float,
    direction: int,
    config: CaseConfig,
    settings: Any,
    solve_next_state_fn: SolveNextStateFn,
    closure_metrics_fn: ClosureMetricsFn,
    step_objective_payload_fn: StepObjectivePayloadFn,
) -> dict[str, Any]:
    logA_next = float(current.logA + float(direction) * float(dx) * float(sigma))
    next_state, residual = solve_next_state_fn(
        current=current,
        logA_next=logA_next,
        sigma=float(sigma),
        dx=dx,
        direction=direction,
        config=config,
        settings=settings,
    )
    metrics = closure_metrics_fn(next_state, config=config)
    objective = step_objective_payload_fn(
        current_metrics=closure_metrics_fn(current, config=config),
        next_metrics=metrics,
        dx=dx,
        settings=settings,
    )
    residual_tol = float(getattr(settings, "rk4_error_tol", 1.0e-6))
    rk4_error_estimate = float(residual.get("rk4_error_estimate", residual.get("max_abs_scaled_residual", float("nan"))))
    margins = {
        "G": float(metrics["G"] - float(settings.g_floor)),
        "Tp": float(metrics["T_p"] - float(settings.tp_floor_K)),
        "residual": float(residual_tol - rk4_error_estimate),
    }
    if dict(residual.get("rk4_stage_constraint_margins", {}) or {}):
        margins.update(dict(residual.get("rk4_stage_constraint_margins", {}) or {}))
    margin_values = [float(value) for value in margins.values()]
    feasible = bool(all(np.isfinite(value) and value >= -float(settings.active_tol) for value in margin_values))
    violation = float(sum(max(-value, 0.0) if np.isfinite(value) else float("inf") for value in margin_values))
    return {
        "ok": bool(residual["ok"]),
        "feasible": bool(feasible and residual["ok"]),
        "sigma": float(sigma),
        "next_state": next_state,
        **objective,
        "constraint_margins": margins,
        "constraint_violation": violation,
        "step_error_kind": "rk4_step_doubling",
        "rk4_error_margin": float(residual_tol - rk4_error_estimate),
        "physical_residual_scaled": float("nan"),
        "physical_residual_ok": False,
        **metrics,
        **residual,
    }


def primitive_log_rhs(
    *,
    state: Any,
    sigma: float,
    config: CaseConfig,
    rhs_mode: str = "raw",
    physics_params_fn: PhysicsParamsFn,
) -> np.ndarray:
    return primitive_log_rhs_fast(
        state=state,
        sigma=sigma,
        config=config,
        rhs_mode=rhs_mode,
        physics_params_fn=physics_params_fn,
    )


def primitive_log_rhs_fast(
    *,
    state: Any,
    sigma: float,
    config: CaseConfig,
    rhs_mode: str,
    physics_params_fn: PhysicsParamsFn,
) -> np.ndarray:
    params = physics_params_fn(config)
    dlog0, dlog1, dlogA = _primitive_log_rhs_fast_values(
        log_n=float(state.log_n),
        log_te=float(state.log_Te),
        logA=float(state.logA),
        sigma=float(sigma),
        rhs_mode=rhs_mode,
        params=params,
    )
    return np.array([dlog0, dlog1, dlogA], dtype=float)


def _primitive_log_rhs_fast_values(
    *,
    log_n: float,
    log_te: float,
    logA: float,
    sigma: float,
    rhs_mode: str,
    params: Any,
) -> tuple[float, float, float]:
    n = _exp_clip(log_n)
    te = _exp_clip(log_te)
    area = float(params.area_scale_m2) * _exp_clip(logA)
    terms = dynamic_terms_numba(
        n,
        te,
        area,
        float(sigma),
        float(params.dot_N),
        float(params.I_0),
        float(params.seed_fraction),
        float(params.B),
        float(params.heavy_particle_mass_kg),
        float(params.seed_ionization_energy_J),
        float(params.sigma_ep),
    )
    M11 = float(terms[0])
    M12 = float(terms[1])
    E11 = float(terms[3])
    E12 = float(terms[4])
    rhs_m = float(terms[7])
    rhs_e = float(terms[8])
    n_scale = max(n, 1.0e-300)
    te_scale = max(te, 1.0e-300)
    mode = str(rhs_mode).strip().lower()
    if mode == "raw":
        dn_dx, dte_dx = _solve_2x2_fast(M11, M12, E11, E12, rhs_m, rhs_e)
        dlog0 = float(dn_dx) / n_scale
        dlog1 = float(dte_dx) / te_scale
    elif mode == "log":
        dlog0, dlog1 = _solve_2x2_fast(
            M11 * n_scale,
            M12 * te_scale,
            E11 * n_scale,
            E12 * te_scale,
            rhs_m,
            rhs_e,
        )
    elif mode == "nondim":
        l00 = M11 * n_scale
        l01 = M12 * te_scale
        l10 = E11 * n_scale
        l11 = E12 * te_scale
        row0 = _row_norm2(l00, l01)
        row1 = _row_norm2(l10, l11)
        dlog0, dlog1 = _solve_2x2_fast(
            l00 / row0,
            l01 / row0,
            l10 / row1,
            l11 / row1,
            rhs_m / row0,
            rhs_e / row1,
        )
    else:
        raise ValueError("rk4_rhs_mode must be 'raw', 'log', or 'nondim'.")
    # REVIEW: Keep fused-numba RHS deferred: profiling says the 2x2 solve is not the bottleneck.
    # A separate raw/log/nondim numba path would offer limited gain and raise maintenance drift risk.
    return float(dlog0), float(dlog1), float(sigma)


def primitive_log_rhs_with_diagnostics(
    *,
    state: Any,
    sigma: float,
    config: CaseConfig,
    rhs_mode: str,
    physics_params_fn: PhysicsParamsFn,
    physics_params: Any | None = None,
) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    params = physics_params if physics_params is not None else physics_params_fn(config)
    n = float(state.n_p)
    te = float(state.T_e)
    area = float(state.area(config))
    terms = dynamic_terms_numba(
        n,
        te,
        area,
        float(sigma),
        float(params.dot_N),
        float(params.I_0),
        float(params.seed_fraction),
        float(params.B),
        float(params.heavy_particle_mass_kg),
        float(params.seed_ionization_energy_J),
        float(params.sigma_ep),
    )
    matrix = np.array([[float(terms[0]), float(terms[1])], [float(terms[3]), float(terms[4])]], dtype=float)
    rhs = np.array([float(terms[7]), float(terms[8])], dtype=float)
    n_scale = max(n, 1.0e-300)
    te_scale = max(te, 1.0e-300)
    # REVIEW: Column scaling rewrites M*[n', Te'] = rhs as
    # (M*diag(n, Te))*[dlogn/dx, dlogTe/dx] = rhs.
    log_matrix = matrix @ np.diag([n_scale, te_scale])
    log_row_matrix = row_normalized_matrix(log_matrix)
    row_scales = row_norms(log_matrix)
    row_scales = np.maximum(row_scales, 1.0e-300)
    scaled_rhs = rhs / row_scales
    singular_values = safe_singular_values(log_row_matrix)
    mode = str(rhs_mode).strip().lower()
    if mode == "raw":
        derivatives, solve_method, solve_rank = solve_linear_rhs(matrix, rhs)
        dlog = np.array([float(derivatives[0]) / n_scale, float(derivatives[1]) / te_scale], dtype=float)
    elif mode == "log":
        dlog, solve_method, solve_rank = solve_linear_rhs(log_matrix, rhs)
    elif mode == "nondim":
        dlog, solve_method, solve_rank = solve_linear_rhs(log_row_matrix, scaled_rhs)
    else:
        raise ValueError("rk4_rhs_mode must be 'raw', 'log', or 'nondim'.")
    values = np.array([float(dlog[0]), float(dlog[1]), float(sigma)], dtype=float)
    stage_residual = log_matrix @ dlog - rhs
    stage_scaled_residual = stage_residual / row_scales
    diagnostics: dict[str, float | bool | str] = {
        "rk4_rhs_mode": mode,
        "rk4_stage_linear_solve_method": solve_method,
        "rk4_stage_linear_solve_rank": int(solve_rank),
        "rk4_stage_Tp_K": float(terms[13]),
        "rk4_stage_mach": float(terms[14]),
        "rk4_stage_G": float(terms[15]),
        "rk4_stage_det_raw": float(terms[9]),
        "rk4_stage_cond_raw": safe_matrix_cond(matrix),
        "rk4_stage_det_log_columns": safe_matrix_det(log_matrix),
        "rk4_stage_cond_log_columns": safe_matrix_cond(log_matrix),
        "rk4_stage_cond_row_norm_log": safe_matrix_cond(log_row_matrix),
        "rk4_stage_cos_rows_log": row_cosine(log_row_matrix),
        "rk4_stage_singular_min_row_norm_log": float(np.min(singular_values)) if singular_values.size else float("nan"),
        "rk4_stage_singular_max_row_norm_log": float(np.max(singular_values)) if singular_values.size else float("nan"),
        "rk4_stage_differential_replay_residual": float(np.max(np.abs(stage_scaled_residual))),
        "rk4_stage_abs_dlogn_dx": abs(float(values[0])),
        "rk4_stage_abs_dlogTe_dx": abs(float(values[1])),
        "rk4_stage_log_rhs_norm": float(np.linalg.norm(values[:2])),
        "rk4_stage_rhs_ok": bool(np.all(np.isfinite(values))),
    }
    return values, diagnostics


def empty_rk4_stage_summary(*, rhs_mode: str) -> dict[str, float | int | bool | str]:
    return {
        "rk4_rhs_mode": str(rhs_mode),
        "rk4_stage_count": 0,
        "rk4_stage_rhs_ok": True,
        # REVIEW: Stage solve method/rank are per-stage today;
        # aggregate them if lstsq fallback frequency matters.
        "rk4_stage_min_Tp_K": float("inf"),
        "rk4_stage_max_mach": -float("inf"),
        "rk4_stage_min_G": float("inf"),
        "rk4_stage_max_G": -float("inf"),
        "rk4_stage_min_det_raw": float("inf"),
        "rk4_stage_min_abs_det_raw": float("inf"),
        "rk4_stage_max_cond_raw": 0.0,
        "rk4_stage_max_cond_log_columns": 0.0,
        "rk4_stage_max_cond_row_norm_log": 0.0,
        "rk4_stage_min_singular_row_norm_log": float("inf"),
        "rk4_stage_max_differential_replay_residual": 0.0,
        "rk4_stage_min_abs_one_minus_cos_rows_log": float("inf"),
        "rk4_stage_max_abs_dlogn_dx": 0.0,
        "rk4_stage_max_abs_dlogTe_dx": 0.0,
        "rk4_stage_max_log_rhs_norm": 0.0,
    }


def finite_or_default(value: float, default: float) -> float:
    return value if np.isfinite(value) else default


def update_rk4_stage_summary(
    summary: dict[str, float | int | bool | str],
    diagnostics: dict[str, float | bool | str],
) -> None:
    summary["rk4_stage_count"] = int(summary["rk4_stage_count"]) + 1
    summary["rk4_stage_rhs_ok"] = bool(summary["rk4_stage_rhs_ok"]) and bool(
        diagnostics.get("rk4_stage_rhs_ok", False)
    )
    tp = float(diagnostics.get("rk4_stage_Tp_K", float("nan")))
    mach = float(diagnostics.get("rk4_stage_mach", float("nan")))
    g_value = float(diagnostics.get("rk4_stage_G", float("nan")))
    det = float(diagnostics.get("rk4_stage_det_raw", float("nan")))
    cos_value = float(diagnostics.get("rk4_stage_cos_rows_log", float("nan")))
    singular_min = float(diagnostics.get("rk4_stage_singular_min_row_norm_log", float("nan")))
    summary["rk4_stage_min_Tp_K"] = min(float(summary["rk4_stage_min_Tp_K"]), finite_or_default(tp, float("inf")))
    summary["rk4_stage_max_mach"] = max(float(summary["rk4_stage_max_mach"]), finite_or_default(mach, -float("inf")))
    summary["rk4_stage_min_G"] = min(float(summary["rk4_stage_min_G"]), finite_or_default(g_value, float("inf")))
    summary["rk4_stage_max_G"] = max(float(summary["rk4_stage_max_G"]), finite_or_default(g_value, -float("inf")))
    summary["rk4_stage_min_singular_row_norm_log"] = min(
        float(summary["rk4_stage_min_singular_row_norm_log"]),
        finite_or_default(singular_min, float("inf")),
    )
    summary["rk4_stage_min_det_raw"] = min(
        float(summary["rk4_stage_min_det_raw"]),
        finite_or_default(det, float("inf")),
    )
    summary["rk4_stage_min_abs_det_raw"] = min(
        float(summary["rk4_stage_min_abs_det_raw"]),
        finite_or_default(abs(det), float("inf")),
    )
    for key in (
        "rk4_stage_cond_raw",
        "rk4_stage_cond_log_columns",
        "rk4_stage_cond_row_norm_log",
        "rk4_stage_differential_replay_residual",
        "rk4_stage_abs_dlogn_dx",
        "rk4_stage_abs_dlogTe_dx",
        "rk4_stage_log_rhs_norm",
    ):
        out_key = key.replace("rk4_stage_cond", "rk4_stage_max_cond").replace(
            "rk4_stage_abs", "rk4_stage_max_abs"
        ).replace("rk4_stage_log_rhs_norm", "rk4_stage_max_log_rhs_norm").replace(
            "rk4_stage_differential_replay_residual", "rk4_stage_max_differential_replay_residual"
        )
        value = float(diagnostics.get(key, float("nan")))
        summary[out_key] = max(float(summary[out_key]), finite_or_default(value, 0.0))
    if np.isfinite(cos_value):
        summary["rk4_stage_min_abs_one_minus_cos_rows_log"] = min(
            float(summary["rk4_stage_min_abs_one_minus_cos_rows_log"]),
            abs(1.0 - cos_value),
        )


def finalize_rk4_stage_summary(summary: dict[str, float | int | bool | str]) -> dict[str, float | int | bool | str]:
    finalized = dict(summary)
    replacements = {
        float("inf"): float("nan"),
        -float("inf"): float("nan"),
    }
    for key, value in list(finalized.items()):
        if isinstance(value, float):
            finalized[key] = replacements.get(value, value)
    return finalized


def rk4_stage_gate_margins(
    *,
    summary: dict[str, float | int | bool | str],
    settings: Any,
) -> dict[str, float]:
    margins: dict[str, float] = {}
    tp_floor = getattr(settings, "rk4_stage_tp_floor_K", None)
    if tp_floor is None:
        tp_floor = float(settings.tp_floor_K)
    margins["stage_Tp"] = float(summary.get("rk4_stage_min_Tp_K", float("nan"))) - float(tp_floor)
    g_floor = getattr(settings, "rk4_stage_g_floor", None)
    if g_floor is None:
        g_floor = float(settings.g_floor)
    margins["stage_G"] = float(summary.get("rk4_stage_min_G", float("nan"))) - float(g_floor)
    cond_max = float(getattr(settings, "rk4_stage_cond_max", float("inf")))
    if np.isfinite(cond_max):
        margins["stage_cond"] = cond_max - float(summary.get("rk4_stage_max_cond_row_norm_log", float("inf")))
    mach_max = float(getattr(settings, "rk4_stage_mach_max", float("inf")))
    if np.isfinite(mach_max):
        margins["stage_mach"] = mach_max - float(summary.get("rk4_stage_max_mach", float("inf")))
    stage_replay_tol = float(getattr(settings, "rk4_stage_replay_tol", float("inf")))
    if np.isfinite(stage_replay_tol):
        margins["stage_replay"] = stage_replay_tol - float(
            summary.get("rk4_stage_max_differential_replay_residual", float("inf"))
        )
    return margins


def rk4_integrate_state(
    *,
    current: Any,
    sigma: float,
    dx_signed: float,
    config: CaseConfig,
    substeps: int,
    rhs_mode: str,
    collect_diagnostics: bool,
    physics_params_fn: PhysicsParamsFn,
    state_factory: StateFactory,
    physics_params: Any | None = None,
) -> tuple[Any, dict[str, float | int | bool | str]]:
    n_substeps = max(int(substeps), 1)
    h = float(dx_signed) / float(n_substeps)
    params = physics_params if physics_params is not None else physics_params_fn(config)
    stage_summary = empty_rk4_stage_summary(rhs_mode=rhs_mode)

    if not collect_diagnostics:
        log_n = float(current.log_n)
        log_te = float(current.log_Te)
        logA = float(current.logA)
        sigma_value = float(sigma)

        def rhs_values(n_value: float, te_value: float, logA_value: float) -> tuple[float, float, float]:
            return _primitive_log_rhs_fast_values(
                log_n=n_value,
                log_te=te_value,
                logA=logA_value,
                sigma=sigma_value,
                rhs_mode=rhs_mode,
                params=params,
            )

        # REVIEW: RK4 substeps hold sigma fixed over the candidate segment; they
        # sample the implied linear logA path rather than reselecting sigma.
        for _ in range(n_substeps):
            k1n, k1te, k1a = rhs_values(log_n, log_te, logA)
            k2n, k2te, k2a = rhs_values(
                log_n + 0.5 * h * k1n,
                log_te + 0.5 * h * k1te,
                logA + 0.5 * h * k1a,
            )
            k3n, k3te, k3a = rhs_values(
                log_n + 0.5 * h * k2n,
                log_te + 0.5 * h * k2te,
                logA + 0.5 * h * k2a,
            )
            k4n, k4te, k4a = rhs_values(
                log_n + h * k3n,
                log_te + h * k3te,
                logA + h * k3a,
            )
            log_n += (h / 6.0) * (k1n + 2.0 * k2n + 2.0 * k3n + k4n)
            log_te += (h / 6.0) * (k1te + 2.0 * k2te + 2.0 * k3te + k4te)
            logA += (h / 6.0) * (k1a + 2.0 * k2a + 2.0 * k3a + k4a)
        return state_factory(log_n=float(log_n), log_Te=float(log_te), logA=float(logA)), finalize_rk4_stage_summary(
            stage_summary
        )

    y = np.array([float(current.log_n), float(current.log_Te), float(current.logA)], dtype=float)

    def rhs(values: np.ndarray) -> np.ndarray:
        state = state_factory(log_n=float(values[0]), log_Te=float(values[1]), logA=float(values[2]))
        rhs_values, diagnostics = primitive_log_rhs_with_diagnostics(
            state=state,
            sigma=float(sigma),
            config=config,
            rhs_mode=rhs_mode,
            physics_params_fn=physics_params_fn,
            physics_params=params,
        )
        update_rk4_stage_summary(stage_summary, diagnostics)
        return rhs_values

    # REVIEW: RK4 substeps hold sigma fixed over the candidate segment; they
    # sample the implied linear logA path rather than reselecting sigma.
    for _ in range(n_substeps):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * h * k1)
        k3 = rhs(y + 0.5 * h * k2)
        k4 = rhs(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state_factory(log_n=float(y[0]), log_Te=float(y[1]), logA=float(y[2])), finalize_rk4_stage_summary(
        stage_summary
    )


def solve_next_state_rk4(
    *,
    current: Any,
    logA_next: float,
    sigma: float,
    dx: float,
    direction: int,
    config: CaseConfig,
    settings: Any,
    physics_params_fn: PhysicsParamsFn,
    state_factory: StateFactory,
) -> tuple[Any, dict[str, float | bool | int | str]]:
    dx_signed = float(direction) * float(dx)
    substeps = max(int(getattr(settings, "rk4_substeps", 1)), 1)
    rhs_mode = rk4_rhs_mode(settings)
    collect_stage_diagnostics = bool(
        getattr(settings, "rk4_stage_diagnostics", False) or getattr(settings, "rk4_stage_gate", False)
    )
    try:
        params = physics_params_fn(config)
        coarse, _coarse_stage = rk4_integrate_state(
            current=current,
            sigma=float(sigma),
            dx_signed=dx_signed,
            config=config,
            substeps=substeps,
            rhs_mode=rhs_mode,
            collect_diagnostics=False,
            physics_params_fn=physics_params_fn,
            state_factory=state_factory,
            physics_params=params,
        )
        fine, fine_stage = rk4_integrate_state(
            current=current,
            sigma=float(sigma),
            dx_signed=dx_signed,
            config=config,
            substeps=2 * substeps,
            rhs_mode=rhs_mode,
            collect_diagnostics=collect_stage_diagnostics,
            physics_params_fn=physics_params_fn,
            state_factory=state_factory,
            physics_params=params,
        )
        coarse = state_factory(log_n=coarse.log_n, log_Te=coarse.log_Te, logA=float(logA_next))
        fine = state_factory(log_n=fine.log_n, log_Te=fine.log_Te, logA=float(logA_next))
        err = max(abs(float(fine.log_n - coarse.log_n)), abs(float(fine.log_Te - coarse.log_Te)))
        finite = bool(
            np.isfinite(fine.log_n)
            and np.isfinite(fine.log_Te)
            and np.isfinite(fine.logA)
            and fine.n_p > 0.0
            and fine.T_e > 0.0
            and fine.area(config) > 0.0
        )
        tol = float(getattr(settings, "rk4_error_tol", 1.0e-6))
        stage_margins = (
            rk4_stage_gate_margins(summary=fine_stage, settings=settings)
            if bool(getattr(settings, "rk4_stage_gate", False))
            else {}
        )
        stage_ok = bool(
            not bool(getattr(settings, "rk4_stage_gate", False))
            or all(float(value) >= -float(settings.active_tol) for value in stage_margins.values())
        )
        rk4_error_ok = bool(finite and err <= tol)
        # REVIEW: Compatibility aliases below contain the RK4 step-doubling
        # error, not a physical residual.
        return fine, {
            "rk4_error_ok": rk4_error_ok,
            "ok": bool(rk4_error_ok and stage_ok),
            "max_abs_scaled_residual": float(err),
            "residual_ok": rk4_error_ok,
            "step_error_kind": "rk4_step_doubling",
            "rk4_error_estimate": float(err),
            "rk4_error_tol": tol,
            "rk4_substeps": int(substeps),
            "rk4_rhs_mode": rhs_mode,
            "rk4_stage_diagnostics_enabled": bool(collect_stage_diagnostics),
            "rk4_stage_gate_enabled": bool(getattr(settings, "rk4_stage_gate", False)),
            "rk4_stage_ok": bool(stage_ok),
            "rk4_stage_constraint_margins": stage_margins,
            **fine_stage,
        }
    except Exception as exc:
        return current, {
            "rk4_error_ok": False,
            "residual_ok": False,
            "ok": False,
            "max_abs_scaled_residual": float("inf"),
            "step_error_kind": "rk4_step_doubling",
            "rk4_error_estimate": float("inf"),
            "rk4_error_tol": float(getattr(settings, "rk4_error_tol", 1.0e-6)),
            "rk4_substeps": int(substeps),
            "rk4_rhs_mode": rhs_mode,
            "rk4_stage_diagnostics_enabled": bool(collect_stage_diagnostics),
            "rk4_stage_gate_enabled": bool(getattr(settings, "rk4_stage_gate", False)),
            "rk4_stage_ok": False,
            "rk4_stage_constraint_margins": {},
            "error": f"rk4 step failed: {exc}",
        }
