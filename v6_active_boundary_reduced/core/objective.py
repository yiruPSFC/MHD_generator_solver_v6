from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from v6_firedrake_reduced.design import (
    DESIGN_VARIABLE_NAMES,
    CaseConfig,
    DesignVector,
    load_case_config,
)

from .numba_physics import closure_state_numba, inlet_design_numba
from .physics_constants import K_B
from .policy import AnchorState, PreparationSettings, State, _physics_params, recover_preparation_profile
from .scoring import soft_square


AREA_DESIGN_VARIABLE_NAMES = ("a1", "a2", "a3")
SEARCH_DESIGN_VARIABLE_NAMES = tuple(
    name for name in DESIGN_VARIABLE_NAMES if name not in AREA_DESIGN_VARIABLE_NAMES
)


@dataclass(frozen=True)
class AnchorOptions:
    x: float = 0.0
    logA: float = 0.0
    sigma_logA: float | None = None
    source: str = "design_anchor"


@dataclass(frozen=True)
class PreparationObjectiveWeights:
    delta_improvement: float = 1.0
    mhd_output_power_MW: float = 0.0
    enthalpy_extraction_percent: float = 0.0
    inlet_delta: float = 0.05
    inlet_te_floor_K: float = 6000.0
    inlet_tp_floor_K: float = 3000.0
    inlet_te_shortfall: float = 1.0
    inlet_tp_shortfall: float = 1.0
    temperature_scale_K: float = 1000.0
    failure_penalty: float = 1.0e6


@dataclass(frozen=True)
class ProfileMetrics:
    mhd_output_power_W: float
    raw_enthalpy_extraction_percent: float
    inlet_enthalpy_flux_W: float
    hall_voltage_V: float
    electric_power_from_hall_W: float
    min_T_p_K: float
    max_Te_over_Tp: float
    min_mach: float
    finite_profile: bool

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "mhd_output_power_W": float(self.mhd_output_power_W),
            "mhd_output_power_MW": float(self.mhd_output_power_W) / 1.0e6,
            "raw_enthalpy_extraction_percent": float(self.raw_enthalpy_extraction_percent),
            "inlet_enthalpy_flux_W": float(self.inlet_enthalpy_flux_W),
            "hall_voltage_V": float(self.hall_voltage_V),
            "electric_power_from_hall_W": float(self.electric_power_from_hall_W),
            "min_T_p_K": float(self.min_T_p_K),
            "max_Te_over_Tp": float(self.max_Te_over_Tp),
            "min_mach": float(self.min_mach),
            "finite_profile": bool(self.finite_profile),
        }


def load_base_config(*, case: str, n_intervals: int | None = None) -> CaseConfig:
    return load_case_config(case=case, n_intervals=n_intervals)


def design_from_overrides(
    base_design: DesignVector,
    overrides: dict[str, float],
    *,
    allow_area_overrides: bool = False,
) -> DesignVector:
    unknown = sorted(set(overrides) - set(DESIGN_VARIABLE_NAMES))
    if unknown:
        raise ValueError(f"unknown design variable(s): {unknown}")
    if not allow_area_overrides:
        area = sorted(set(overrides) & set(AREA_DESIGN_VARIABLE_NAMES))
        if area:
            raise ValueError(
                "area spline variables are not search variables for the active-boundary solver: "
                + ", ".join(area)
            )
    values = base_design.to_dict()
    for name, value in overrides.items():
        values[str(name)] = float(value)
    return DesignVector.from_dict(values)


def config_with_design(config: CaseConfig, design: DesignVector) -> CaseConfig:
    return replace(config, design=design, B_T=float(design.B_T))


def anchor_from_design(
    config: CaseConfig,
    *,
    options: AnchorOptions | None = None,
) -> AnchorState:
    opts = options or AnchorOptions()
    design = config.design
    if float(design.T_e_in) <= 0.0:
        raise ValueError("design.T_e_in must be positive.")
    return AnchorState(
        state=State(
            log_n=float(design.log_n_p_in),
            log_Te=float(np.log(float(design.T_e_in))),
            logA=float(opts.logA),
        ),
        sigma_logA=None if opts.sigma_logA is None else float(opts.sigma_logA),
        x=float(opts.x),
        source=str(opts.source),
        source_index=-1,
    )


def evaluate_preparation_design(
    *,
    base_config: CaseConfig,
    design_overrides: dict[str, float],
    settings: PreparationSettings,
    weights: PreparationObjectiveWeights | None = None,
    anchor_options: AnchorOptions | None = None,
    case_id: int | None = None,
    return_payload: bool = False,
) -> dict[str, Any]:
    objective_weights = weights or PreparationObjectiveWeights()
    started = time.perf_counter()
    design_payload: dict[str, float]
    try:
        design = design_from_overrides(
            base_config.design,
            {str(k): float(v) for k, v in design_overrides.items()},
            allow_area_overrides=False,
        )
        config = config_with_design(base_config, design)
        anchor = anchor_from_design(config, options=anchor_options)
        design_payload = design.to_dict()
        payload = recover_preparation_profile(config=config, anchor=anchor, settings=settings)
        result = _score_rollout(
            payload=payload,
            design=design,
            config=config,
            weights=objective_weights,
            elapsed_s=time.perf_counter() - started,
            case_id=case_id,
            failure=None,
        )
        if return_payload:
            result["payload"] = payload
        return result
    except Exception as exc:
        try:
            design_payload = design_from_overrides(
                base_config.design,
                {str(k): float(v) for k, v in design_overrides.items()},
                allow_area_overrides=False,
            ).to_dict()
        except Exception:
            design_payload = dict(base_config.design.to_dict())
            for key, value in design_overrides.items():
                design_payload[str(key)] = float(value)
        return _failure_result(
            design=design_payload,
            weights=objective_weights,
            elapsed_s=time.perf_counter() - started,
            case_id=case_id,
            failure=str(exc),
        )


def _score_rollout(
    *,
    payload: dict[str, Any],
    design: DesignVector,
    config: CaseConfig,
    weights: PreparationObjectiveWeights,
    elapsed_s: float,
    case_id: int | None,
    failure: str | None,
) -> dict[str, Any]:
    nodes = list(payload.get("nodes", []))
    if len(nodes) < 2:
        return _failure_result(
            design=design.to_dict(),
            weights=weights,
            elapsed_s=elapsed_s,
            case_id=case_id,
            failure=failure or "rollout produced fewer than two nodes",
        )
    outlet = dict(nodes[0])
    inlet = dict(nodes[-1])
    active_summary = dict(payload.get("active_summary", {}) or {})
    support_counts = dict(active_summary.get("support_counts", {}) or {})
    settings_payload = dict(payload.get("settings", {}) or {})
    g_floor = float(settings_payload.get("g_floor", 0.0))
    tp_floor = float(settings_payload.get("tp_floor_K", 300.0))
    active_tol = float(settings_payload.get("active_tol", 1e-6))
    delta_improvement = float(outlet["Delta"]) - float(inlet["Delta"])
    inlet_delta_penalty = float(weights.inlet_delta) * max(float(inlet["Delta"]), 0.0)
    temperature_scale = max(float(weights.temperature_scale_K), 1e-300)
    te_shortfall_penalty = float(weights.inlet_te_shortfall) * soft_square(
        float(weights.inlet_te_floor_K) - float(inlet["T_e"]), temperature_scale
    )
    tp_shortfall_penalty = float(weights.inlet_tp_shortfall) * soft_square(
        float(weights.inlet_tp_floor_K) - float(inlet["T_p"]), temperature_scale
    )
    target_g_margin = float(outlet["G"]) - g_floor
    target_tp_margin = float(outlet["T_p"]) - tp_floor
    profile_metrics = _profile_metric_terms(payload=payload, design=design, config=config)
    profile_metrics_ok = bool(float(profile_metrics.get("profile_metrics_ok", 0.0)) >= 0.5)
    profile_metrics_required = bool(
        float(weights.mhd_output_power_MW) != 0.0
        or float(weights.enthalpy_extraction_percent) != 0.0
    )
    # REVIEW: Profile metrics are optional for delta-only scans, but required when they affect reward.
    mhd_output_MW = float(profile_metrics.get("mhd_output_power_W", float("nan"))) / 1.0e6
    enthalpy_extraction = float(profile_metrics.get("raw_enthalpy_extraction_percent", float("nan")))
    mhd_power_reward = (
        float(weights.mhd_output_power_MW) * mhd_output_MW if np.isfinite(mhd_output_MW) else 0.0
    )
    enthalpy_reward = (
        float(weights.enthalpy_extraction_percent) * enthalpy_extraction
        if np.isfinite(enthalpy_extraction)
        else 0.0
    )
    target_ok = bool(target_g_margin >= -active_tol and target_tp_margin >= -active_tol)
    rollout_ok = (
        bool(payload.get("ok", False))
        and target_ok
        and (profile_metrics_ok or not profile_metrics_required)
    )
    failure_diagnostics = _failure_diagnostics(
        payload_ok=bool(payload.get("ok", False)),
        target_ok=target_ok,
        active_summary=active_summary,
        target_g_margin=target_g_margin,
        target_tp_margin=target_tp_margin,
        active_tol=active_tol,
    )
    if profile_metrics_required and not profile_metrics_ok:
        failure_diagnostics = dict(failure_diagnostics)
        blockers = list(failure_diagnostics.get("primary_blockers", []) or [])
        blockers.append("profile_metrics")
        failure_diagnostics.update(
            {
                "kind": failure_diagnostics.get("kind") or "profile_metrics",
                "primary_blockers": sorted(set(str(item) for item in blockers)),
                "profile_metrics_failure": str(
                    profile_metrics.get("profile_metrics_failure", "profile_metrics_unavailable")
                ),
            }
        )
    failure_reason = failure
    if failure_reason is None and not target_ok:
        failure_reason = (
            "target_anchor_infeasible: "
            f"G_margin={target_g_margin:.6g}, Tp_margin={target_tp_margin:.6g}"
        )
    if failure_reason is None and profile_metrics_required and not profile_metrics_ok:
        failure_reason = "profile_metrics_unavailable: " + str(
            profile_metrics.get("profile_metrics_failure", "unknown")
        )
    if failure_reason is None and not bool(payload.get("ok", False)):
        failure_reason = _rollout_failure_message(failure_diagnostics)
    failure_penalty = 0.0 if rollout_ok else float(weights.failure_penalty)
    raw_score = float(weights.delta_improvement) * delta_improvement
    raw_score += mhd_power_reward + enthalpy_reward
    raw_score -= inlet_delta_penalty + te_shortfall_penalty + tp_shortfall_penalty
    score = raw_score if rollout_ok and np.isfinite(raw_score) else -float(weights.failure_penalty)
    return {
        "case_id": None if case_id is None else int(case_id),
        "ok": rollout_ok,
        "score": float(score),
        "failure": failure_reason,
        "failure_diagnostics": failure_diagnostics,
        "elapsed_s": float(elapsed_s),
        "design": design.to_dict(),
        "search_variables": {name: float(getattr(design, name)) for name in SEARCH_DESIGN_VARIABLE_NAMES},
        "fixed_area_variables": {name: float(getattr(design, name)) for name in AREA_DESIGN_VARIABLE_NAMES},
        "objective_terms": {
            "delta_improvement": float(delta_improvement),
            "delta_improvement_reward": float(weights.delta_improvement) * delta_improvement,
            "mhd_output_power_MW_reward": mhd_power_reward,
            "enthalpy_extraction_percent_reward": enthalpy_reward,
            "inlet_delta_penalty": inlet_delta_penalty,
            "inlet_te_shortfall_penalty": te_shortfall_penalty,
            "inlet_tp_shortfall_penalty": tp_shortfall_penalty,
            "failure_penalty": failure_penalty,
            "raw_score_before_failure_gate": float(raw_score),
            "target_g_margin": target_g_margin,
            "target_tp_margin": target_tp_margin,
            "target_ok": float(1.0 if target_ok else 0.0),
            **profile_metrics,
        },
        "outlet": _node_summary(outlet),
        "preparation_inlet": _node_summary(inlet),
        "active_summary": active_summary,
        "support_counts": support_counts,
        "max_abs_scaled_residual": float(active_summary.get("max_abs_scaled_residual", np.nan)),
        "n_steps_completed": max(len(nodes) - 1, 0),
    }


def _inlet_enthalpy_flux(*, design: DesignVector, config: CaseConfig) -> float:
    params = _physics_params(config)
    inlet = inlet_design_numba(
        float(design.n_p_in),
        float(design.T_e_in),
        float(design.Z_in),
        float(design.I_0),
        float(design.seed_fraction),
        float(design.B_T),
        float(config.area_scale_m2),
        float(params.heavy_particle_mass_kg),
        float(params.seed_ionization_energy_J),
        float(params.sigma_ep),
    )
    n_p = max(float(inlet[0]), 1.0)
    n_e = max(float(inlet[1]), 0.0)
    T_e = max(float(inlet[2]), 1.0)
    T_p = max(float(inlet[3]), 1.0)
    v_in = max(float(inlet[7]), 1.0e-30)
    area = max(float(inlet[11]), 1.0e-30)
    thermal_density = 2.5 * K_B * (n_p * T_p + n_e * T_e)
    kinetic_density = 0.5 * float(params.heavy_particle_mass_kg) * n_p * v_in * v_in
    return float(area * v_in * (thermal_density + kinetic_density))


def evaluate_profile_metrics(
    *,
    profile: dict[str, Any],
    design: DesignVector,
    config: CaseConfig,
) -> ProfileMetrics:
    x = np.asarray(profile["x"], dtype=float).reshape(-1)
    n_p = np.asarray(profile["n_p"], dtype=float).reshape(-1)
    T_e = np.asarray(profile["T_e"], dtype=float).reshape(-1)
    area = np.asarray(profile["A"], dtype=float).reshape(-1)
    sigma = np.asarray(profile["sigma_logA"], dtype=float).reshape(-1)
    if not (x.size == n_p.size == T_e.size == area.size == sigma.size):
        raise ValueError("profile arrays x, n_p, T_e, A, and sigma_logA must have matching lengths.")
    if x.size < 2:
        raise ValueError("profile metrics require at least two profile points.")

    params = _physics_params(config)
    T_p_values: list[float] = []
    mach_values: list[float] = []
    power_density: list[float] = []
    hall_field: list[float] = []
    for n_val, te_val, area_val in zip(n_p, T_e, area, strict=True):
        closure = closure_state_numba(
            float(n_val),
            float(te_val),
            float(area_val),
            float(params.dot_N),
            float(params.I_0),
            float(params.seed_fraction),
            float(params.B),
            float(params.heavy_particle_mass_kg),
            float(params.seed_ionization_energy_J),
            float(params.sigma_ep),
        )
        A_safe = float(closure[2])
        J_x = float(closure[14])
        E_x = float(closure[16])
        T_p_values.append(float(closure[10]))
        mach_values.append(float(closure[17]))
        power_density.append(float(-A_safe * J_x * E_x))
        hall_field.append(float(-E_x))

    T_p = np.asarray(T_p_values, dtype=float)
    mach = np.asarray(mach_values, dtype=float)
    power_density_arr = np.asarray(power_density, dtype=float)
    hall_field_arr = np.asarray(hall_field, dtype=float)
    mhd_output_power_W = float(np.trapezoid(power_density_arr, x))
    hall_voltage = float(np.trapezoid(hall_field_arr, x))
    inlet_flux_W = _inlet_enthalpy_flux(design=design, config=config)
    te_over_tp = T_e / np.maximum(T_p, 1.0)
    return ProfileMetrics(
        mhd_output_power_W=mhd_output_power_W,
        raw_enthalpy_extraction_percent=float(100.0 * mhd_output_power_W / max(inlet_flux_W, 1e-30)),
        inlet_enthalpy_flux_W=inlet_flux_W,
        hall_voltage_V=hall_voltage,
        electric_power_from_hall_W=float(float(design.I_0) * hall_voltage),
        min_T_p_K=float(np.nanmin(T_p)),
        max_Te_over_Tp=float(np.nanmax(te_over_tp)),
        min_mach=float(np.nanmin(mach)),
        finite_profile=bool(
            np.all(np.isfinite(x))
            and np.all(np.isfinite(n_p))
            and np.all(np.isfinite(T_e))
            and np.all(np.isfinite(area))
            and np.all(np.isfinite(sigma))
            and np.all(np.isfinite(T_p))
            and np.all(np.isfinite(mach))
            and np.all(np.isfinite(power_density_arr))
            and np.all(np.isfinite(hall_field_arr))
            and np.all(np.isfinite(te_over_tp))
        ),
    )


def _profile_metric_terms(
    *,
    payload: dict[str, Any],
    design: DesignVector,
    config: CaseConfig,
) -> dict[str, Any]:
    arrays = dict(payload.get("profile_arrays", {}) or {})
    required = ("x", "n_p", "T_e", "A", "sigma_logA")
    missing = [name for name in required if name not in arrays]
    if missing:
        return {
            "profile_metrics_ok": 0.0,
            "profile_metrics_failure": "missing_profile_arrays:" + ",".join(missing),
        }
    profile = {name: np.asarray(arrays[name], dtype=float).reshape(-1) for name in required}
    sizes = {profile[name].size for name in required}
    if len(sizes) != 1 or next(iter(sizes)) < 2:
        return {
            "profile_metrics_ok": 0.0,
            "profile_metrics_failure": "invalid_profile_array_sizes",
        }
    order = np.argsort(profile["x"])
    sorted_profile = {name: values[order] for name, values in profile.items()}
    try:
        metrics = evaluate_profile_metrics(profile=sorted_profile, design=design, config=config)
    except Exception as exc:
        return {
            "profile_metrics_ok": 0.0,
            "profile_metrics_failure": type(exc).__name__ + ": " + str(exc),
        }
    values = metrics.to_dict()
    return {
        "profile_metrics_ok": 1.0,
        "profile_metrics_failure": "",
        "mhd_output_power_W": float(values["mhd_output_power_W"]),
        "mhd_output_power_MW": float(values["mhd_output_power_W"]) / 1.0e6,
        "raw_enthalpy_extraction_percent": float(values["raw_enthalpy_extraction_percent"]),
        "inlet_enthalpy_flux_W": float(values["inlet_enthalpy_flux_W"]),
        "hall_voltage_V": float(values["hall_voltage_V"]),
        "electric_power_from_hall_W": float(values["electric_power_from_hall_W"]),
        "profile_min_T_p_K": float(values["min_T_p_K"]),
        "profile_max_Te_over_Tp": float(values["max_Te_over_Tp"]),
        "profile_min_mach": float(values["min_mach"]),
    }


def _failure_diagnostics(
    *,
    payload_ok: bool,
    target_ok: bool,
    active_summary: dict[str, Any],
    target_g_margin: float,
    target_tp_margin: float,
    active_tol: float,
) -> dict[str, Any]:
    if bool(payload_ok) and bool(target_ok):
        return {}
    termination = dict(active_summary.get("termination", {}) or {})
    margins = dict(termination.get("constraint_margins", {}) or {})
    primary_blockers = [
        str(name) for name, value in margins.items() if float(value) < -float(active_tol)
    ]
    if not target_ok:
        if float(target_g_margin) < -float(active_tol):
            primary_blockers.append("target_G")
        if float(target_tp_margin) < -float(active_tol):
            primary_blockers.append("target_Tp")
    if not primary_blockers and not bool(termination.get("ok", bool(payload_ok))):
        reason = str(termination.get("reason") or termination.get("support_type") or "rollout")
        primary_blockers.append(reason)

    scan_diagnostics = dict(termination.get("scan_diagnostics", {}) or {})
    best_residual = dict(scan_diagnostics.get("best_residual", {}) or {})
    best_violation = dict(scan_diagnostics.get("best_violation", {}) or {})
    return {
        "kind": "target_anchor" if not target_ok else "rollout",
        "primary_blockers": sorted(set(primary_blockers)),
        "termination_reason": termination.get("reason"),
        "termination_error": termination.get("error"),
        "failed_step": termination.get("step"),
        "support_type": termination.get("support_type"),
        "solver_method": termination.get("solver_method"),
        "sigma": termination.get("sigma"),
        "sigma_interval_lower": termination.get("sigma_interval_lower"),
        "sigma_interval_upper": termination.get("sigma_interval_upper"),
        "constraint_margins": margins,
        "max_abs_scaled_residual": termination.get("max_abs_scaled_residual"),
        "reverse_interval_error": termination.get("reverse_interval_error"),
        "reverse_interval_conflict": termination.get("reverse_interval_conflict"),
        "reverse_interval_conflict_kind": termination.get("reverse_interval_conflict_kind"),
        "reverse_interval_conflict_summary": termination.get("reverse_interval_conflict_summary"),
        "reverse_interval_conflict_lower_source": termination.get("reverse_interval_conflict_lower_source"),
        "reverse_interval_conflict_upper_source": termination.get("reverse_interval_conflict_upper_source"),
        "reverse_interval_conflict_sigma_lower": termination.get("reverse_interval_conflict_sigma_lower"),
        "reverse_interval_conflict_sigma_upper": termination.get("reverse_interval_conflict_sigma_upper"),
        "reverse_interval_conflict_sigma_gap": termination.get("reverse_interval_conflict_sigma_gap"),
        "reverse_interval_conflict_Aprime_lower": termination.get("reverse_interval_conflict_Aprime_lower"),
        "reverse_interval_conflict_Aprime_upper": termination.get("reverse_interval_conflict_Aprime_upper"),
        "reverse_interval_conflict_Aprime_gap": termination.get("reverse_interval_conflict_Aprime_gap"),
        "scan_feasible_count": scan_diagnostics.get("feasible_count"),
        "scan_n": scan_diagnostics.get("n_scan"),
        "best_residual": best_residual,
        "best_violation": best_violation,
        "target_g_margin": float(target_g_margin),
        "target_tp_margin": float(target_tp_margin),
    }


def _rollout_failure_message(diagnostics: dict[str, Any]) -> str:
    if not diagnostics:
        return "reverse_rollout_failed"
    blockers = ",".join(str(item) for item in diagnostics.get("primary_blockers", []) or [])
    pieces = [
        "reverse_rollout_failed",
        f"step={diagnostics.get('failed_step')}",
        f"reason={diagnostics.get('termination_reason')}",
        f"blockers={blockers or 'unknown'}",
        f"support={diagnostics.get('support_type')}",
    ]
    best_residual = dict(diagnostics.get("best_residual", {}) or {})
    margins = dict(best_residual.get("constraint_margins", {}) or {})
    if best_residual:
        pieces.extend(
            [
                f"best_residual_sigma={best_residual.get('sigma')}",
                f"best_residual={best_residual.get('max_abs_scaled_residual')}",
                f"residual_margin={margins.get('residual')}",
            ]
        )
    interval_summary = str(diagnostics.get("reverse_interval_conflict_summary") or "")
    if interval_summary:
        pieces.append(f"interval_conflict={interval_summary}")
    return "; ".join(pieces)


def _failure_result(
    *,
    design: dict[str, float],
    weights: PreparationObjectiveWeights,
    elapsed_s: float,
    case_id: int | None,
    failure: str,
) -> dict[str, Any]:
    return {
        "case_id": None if case_id is None else int(case_id),
        "ok": False,
        "score": -float(weights.failure_penalty),
        "failure": str(failure),
        "failure_diagnostics": {
            "kind": "exception",
            "primary_blockers": ["exception"],
            "termination_error": str(failure),
        },
        "elapsed_s": float(elapsed_s),
        "design": {str(k): float(v) for k, v in design.items()},
        "search_variables": {
            name: float(design[name]) for name in SEARCH_DESIGN_VARIABLE_NAMES if name in design
        },
        "fixed_area_variables": {
            name: float(design[name]) for name in AREA_DESIGN_VARIABLE_NAMES if name in design
        },
        "objective_terms": {
            "delta_improvement": float("nan"),
            "delta_improvement_reward": float("nan"),
            "mhd_output_power_MW_reward": float("nan"),
            "enthalpy_extraction_percent_reward": float("nan"),
            "inlet_delta_penalty": float("nan"),
            "inlet_te_shortfall_penalty": float("nan"),
            "inlet_tp_shortfall_penalty": float("nan"),
            "failure_penalty": float(weights.failure_penalty),
        },
        "outlet": {},
        "preparation_inlet": {},
        "active_summary": {},
        "support_counts": {},
        "max_abs_scaled_residual": float("nan"),
        "n_steps_completed": 0,
    }


def _node_summary(node: dict[str, Any]) -> dict[str, float]:
    fields = (
        "x",
        "n_p",
        "T_e",
        "T_p",
        "Delta",
        "mach",
        "A",
        "logA",
        "sigma_logA",
        "G",
        "beta",
        "Z",
        "power_density_W_per_m",
        "hall_field_V_per_m",
        "J_x",
        "E_x",
    )
    return {name: float(node[name]) for name in fields if name in node}


def flatten_result_for_csv(result: dict[str, Any]) -> dict[str, Any]:
    active_summary = dict(result.get("active_summary", {}) or {})
    termination = dict(active_summary.get("termination", {}) or {})
    bound_sources = dict(termination.get("bound_sources", {}) or {})
    constraint_margins = dict(termination.get("constraint_margins", {}) or {})
    scan_diagnostics = dict(termination.get("scan_diagnostics", {}) or {})
    row: dict[str, Any] = {
        "case_id": result.get("case_id"),
        "ok": bool(result.get("ok", False)),
        "score": float(result.get("score", np.nan)),
        "failure": result.get("failure") or "",
        "elapsed_s": float(result.get("elapsed_s", np.nan)),
        "max_abs_scaled_residual": float(result.get("max_abs_scaled_residual", np.nan)),
        "n_steps_completed": int(result.get("n_steps_completed", 0)),
    }
    failure_diagnostics = dict(result.get("failure_diagnostics", {}) or {})
    for name in (
        "kind",
        "termination_reason",
        "termination_error",
        "failed_step",
        "support_type",
        "selected_support_type",
        "selected_sigma_origin",
        "selected_sigma_source",
        "affine_support_type",
        "affine_objective_bound_kind",
        "affine_selected_endpoint_source",
        "objective_bound_kind",
        "selected_endpoint_source",
        "solver_method",
        "sign_aware_fallback_status",
        "sign_aware_fallback_attempted",
        "sign_aware_fallback_used",
        "sign_aware_fallback_recovered",
        "sign_aware_fallback_solver_method",
        "sign_aware_fallback_validation_failure",
        "sign_aware_endpoint_sigma",
        "sign_aware_endpoint_ok",
        "sign_aware_endpoint_feasible",
        "sign_aware_endpoint_solver_method",
        "sign_aware_endpoint_validation_failure",
        "sign_aware_endpoint_constraint_violation",
        "sonic_objective_score",
        "sonic_direction_ok",
        "sonic_direction_gate",
        "sonic_compatibility_status",
        "sonic_compatibility_selected_sigma",
        "sonic_compatibility_selected_scaled_residual",
        "sonic_compatibility_best_interval_sigma",
        "sonic_compatibility_best_interval_scaled_residual",
        "sonic_compatibility_variation_scaled",
        "sonic_compatibility_root_sigma",
        "sigma",
        "sigma_interval_lower",
        "sigma_interval_upper",
        "max_abs_scaled_residual",
        "reverse_interval_error",
        "reverse_interval_conflict",
        "reverse_interval_conflict_kind",
        "reverse_interval_conflict_summary",
        "reverse_interval_conflict_lower_source",
        "reverse_interval_conflict_upper_source",
        "reverse_interval_conflict_sigma_lower",
        "reverse_interval_conflict_sigma_upper",
        "reverse_interval_conflict_sigma_gap",
        "reverse_interval_conflict_Aprime_lower",
        "reverse_interval_conflict_Aprime_upper",
        "reverse_interval_conflict_Aprime_gap",
        "profile_metrics_failure",
        "scan_feasible_count",
        "scan_n",
        "target_g_margin",
        "target_tp_margin",
    ):
        if name in failure_diagnostics:
            row[f"failure_{name}"] = failure_diagnostics[name]
    if "primary_blockers" in failure_diagnostics:
        row["failure_primary_blockers"] = ",".join(
            str(item) for item in failure_diagnostics.get("primary_blockers", []) or []
        )
    for name, value in dict(failure_diagnostics.get("constraint_margins", {}) or {}).items():
        row[f"failure_margin_{name}"] = value
    for label in ("best_residual", "best_violation"):
        item = dict(failure_diagnostics.get(label, {}) or {})
        for name in (
            "sigma",
            "feasible",
            "constraint_violation",
            "max_abs_scaled_residual",
        ):
            if name in item:
                row[f"failure_{label}_{name}"] = item[name]
        for name, value in dict(item.get("constraint_margins", {}) or {}).items():
            row[f"failure_{label}_margin_{name}"] = value
    for name in (
        "n_steps_requested",
        "n_steps_completed",
        "logA_min",
        "logA_max",
        "A_min",
        "A_max",
        "Te_min_K",
        "Te_max_K",
        "Tp_min_K",
        "Tp_max_K",
        "mach_min",
        "mach_max",
        "objective",
        "power_density_min_W_per_m",
        "power_density_max_W_per_m",
        "mhd_output_power_W",
        "mhd_output_power_MW",
        "sigma_min",
        "sigma_max",
        "G_min_excluding_anchor",
        "G_active_count_excluding_anchor",
        "G_margin_near_count_excluding_anchor",
        "Tp_min_excluding_anchor_K",
        "Tp_floor_active_count_excluding_anchor",
        "Tp_floor_margin_near_count_excluding_anchor",
    ):
        if name in active_summary:
            row[f"active_{name}"] = active_summary[name]
    for name in (
        "ok",
        "reason",
        "step",
        "support_type",
        "solver_method",
        "sigma",
        "sigma_interval_lower",
        "sigma_interval_upper",
        "max_abs_scaled_residual",
        "error",
        "reverse_interval_error",
        "reverse_interval_conflict",
        "reverse_interval_conflict_kind",
        "reverse_interval_conflict_summary",
        "reverse_interval_conflict_lower_source",
        "reverse_interval_conflict_upper_source",
        "reverse_interval_conflict_sigma_lower",
        "reverse_interval_conflict_sigma_upper",
        "reverse_interval_conflict_sigma_gap",
        "reverse_interval_conflict_Aprime_lower",
        "reverse_interval_conflict_Aprime_upper",
        "reverse_interval_conflict_Aprime_gap",
    ):
        if name in termination:
            row[f"termination_{name}"] = termination[name]
    for side, source in bound_sources.items():
        row[f"termination_bound_source_{side}"] = source
    for name, value in constraint_margins.items():
        row[f"termination_margin_{name}"] = value
    for name in (
        "n_scan",
        "feasible_count",
        "sigma_min",
        "sigma_max",
        "feasible_sigma_min",
        "feasible_sigma_max",
    ):
        if name in scan_diagnostics:
            row[f"scan_{name}"] = scan_diagnostics[name]
    for label in (
        "best_violation",
        "best_residual",
        "left_endpoint",
        "right_endpoint",
        "best_objective_feasible",
    ):
        item = dict(scan_diagnostics.get(label, {}) or {})
        if not item:
            continue
        for name in (
            "sigma",
            "ok",
            "feasible",
            "objective_value",
            "delta_gain",
            "constraint_violation",
            "max_abs_scaled_residual",
        ):
            if name in item:
                row[f"scan_{label}_{name}"] = item[name]
        for name, value in dict(item.get("constraint_margins", {}) or {}).items():
            row[f"scan_{label}_margin_{name}"] = value
    for name, value in dict(result.get("design", {}) or {}).items():
        row[f"design_{name}"] = float(value)
    for name, value in dict(result.get("objective_terms", {}) or {}).items():
        if isinstance(value, (bool, np.bool_)):
            row[f"objective_{name}"] = bool(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            row[f"objective_{name}"] = float(value)
        elif value is None:
            row[f"objective_{name}"] = ""
        else:
            row[f"objective_{name}"] = str(value)
    for prefix in ("outlet", "preparation_inlet"):
        for name, value in dict(result.get(prefix, {}) or {}).items():
            row[f"{prefix}_{name}"] = float(value)
    for name, value in dict(result.get("support_counts", {}) or {}).items():
        row[f"support_{name}"] = int(value)
    return row
