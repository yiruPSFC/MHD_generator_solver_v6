from __future__ import annotations

from typing import Any

import numpy as np


def eval_public(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "objective_kind": str(item.get("objective_kind", "")),
        "objective_value": float(item.get("objective_value", float("nan"))),
        "delta_gain": float(item.get("delta_gain", float("nan"))),
        "power_next_W_per_m": float(item.get("power_next_W_per_m", float("nan"))),
        "power_density_gain_W_per_m": float(item.get("power_density_gain_W_per_m", float("nan"))),
        "step_power_W": float(item.get("step_power_W", float("nan"))),
        "G": float(item.get("G", float("nan"))),
        "T_p": float(item.get("T_p", float("nan"))),
        "Delta": float(item.get("Delta", float("nan"))),
        "mach": float(item.get("mach", float("nan"))),
        "power_density_W_per_m": float(item.get("power_density_W_per_m", float("nan"))),
        "hall_field_V_per_m": float(item.get("hall_field_V_per_m", float("nan"))),
        "J_x": float(item.get("J_x", float("nan"))),
        "J_y": float(item.get("J_y", float("nan"))),
        "E_x": float(item.get("E_x", float("nan"))),
        "constraint_margins": dict(item.get("constraint_margins", {})),
        "boundary_blockers": list(item.get("boundary_blockers", [])),
        "boundary_bracket_width": float(item.get("boundary_bracket_width", float("nan"))),
        "boundary_infeasible_sigma": float(item.get("boundary_infeasible_sigma", float("nan"))),
        "boundary_infeasible_margins": dict(item.get("boundary_infeasible_margins", {})),
        "solver_method": str(item.get("solver_method", "legacy_scan")),
        "step_error_kind": str(item.get("step_error_kind", "")),
        "max_abs_scaled_residual": float(item.get("max_abs_scaled_residual", float("nan"))),
        "physical_residual_scaled": float(item.get("physical_residual_scaled", float("nan"))),
        "physical_residual_ok": bool(item.get("physical_residual_ok", False)),
        "rk4_error_ok": bool(item.get("rk4_error_ok", False)),
        "rk4_error_estimate": float(item.get("rk4_error_estimate", float("nan"))),
        "rk4_error_margin": float(item.get("rk4_error_margin", float("nan"))),
        "rk4_substeps": int(item.get("rk4_substeps", -1)),
        "rk4_rhs_mode": str(item.get("rk4_rhs_mode", "")),
        "rk4_stage_diagnostics_enabled": bool(item.get("rk4_stage_diagnostics_enabled", False)),
        "rk4_stage_gate_enabled": bool(item.get("rk4_stage_gate_enabled", False)),
        "rk4_stage_ok": bool(item.get("rk4_stage_ok", True)),
        "rk4_stage_constraint_margins": dict(item.get("rk4_stage_constraint_margins", {}) or {}),
        "rk4_stage_count": int(item.get("rk4_stage_count", 0)),
        "rk4_stage_min_Tp_K": float(item.get("rk4_stage_min_Tp_K", float("nan"))),
        "rk4_stage_max_mach": float(item.get("rk4_stage_max_mach", float("nan"))),
        "rk4_stage_min_G": float(item.get("rk4_stage_min_G", float("nan"))),
        "rk4_stage_max_G": float(item.get("rk4_stage_max_G", float("nan"))),
        "rk4_stage_min_abs_det_raw": float(item.get("rk4_stage_min_abs_det_raw", float("nan"))),
        "rk4_stage_max_cond_raw": float(item.get("rk4_stage_max_cond_raw", float("nan"))),
        "rk4_stage_max_cond_log_columns": float(item.get("rk4_stage_max_cond_log_columns", float("nan"))),
        "rk4_stage_max_cond_row_norm_log": float(item.get("rk4_stage_max_cond_row_norm_log", float("nan"))),
        "rk4_stage_min_singular_row_norm_log": float(
            item.get("rk4_stage_min_singular_row_norm_log", float("nan"))
        ),
        "rk4_stage_max_differential_replay_residual": float(
            item.get("rk4_stage_max_differential_replay_residual", float("nan"))
        ),
        "rk4_stage_min_abs_one_minus_cos_rows_log": float(
            item.get("rk4_stage_min_abs_one_minus_cos_rows_log", float("nan"))
        ),
        "rk4_stage_max_abs_dlogn_dx": float(item.get("rk4_stage_max_abs_dlogn_dx", float("nan"))),
        "rk4_stage_max_abs_dlogTe_dx": float(item.get("rk4_stage_max_abs_dlogTe_dx", float("nan"))),
        "rk4_stage_max_log_rhs_norm": float(item.get("rk4_stage_max_log_rhs_norm", float("nan"))),
    }


def scan_diagnostics(scan: list[dict[str, Any]]) -> dict[str, Any]:
    if not scan:
        return {"n_scan": 0, "feasible_count": 0}
    feasible = [item for item in scan if bool(item.get("feasible", False))]
    by_violation = min(scan, key=lambda item: float(item.get("constraint_violation", 1e300)))
    by_residual = min(scan, key=lambda item: float(item.get("max_abs_scaled_residual", 1e300)))
    by_step_error = min(
        scan,
        key=lambda item: float(item.get("rk4_error_estimate", item.get("max_abs_scaled_residual", 1e300))),
    )
    payload: dict[str, Any] = {
        "n_scan": int(len(scan)),
        "feasible_count": int(len(feasible)),
        "sigma_min": float(min(float(item["sigma"]) for item in scan)),
        "sigma_max": float(max(float(item["sigma"]) for item in scan)),
        "best_violation": scan_item_summary(by_violation),
        "best_residual": scan_item_summary(by_residual),
        "best_step_error": scan_item_summary(by_step_error),
        "left_endpoint": scan_item_summary(scan[0]),
        "right_endpoint": scan_item_summary(scan[-1]),
    }
    if feasible:
        payload["feasible_sigma_min"] = float(min(float(item["sigma"]) for item in feasible))
        payload["feasible_sigma_max"] = float(max(float(item["sigma"]) for item in feasible))
        payload["best_objective_feasible"] = scan_item_summary(
            max(feasible, key=lambda item: float(item.get("objective_value", -1e300)))
        )
    return payload


def scan_item_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "sigma": float(item.get("sigma", float("nan"))),
        "ok": bool(item.get("ok", False)),
        "feasible": bool(item.get("feasible", False)),
        "objective_kind": str(item.get("objective_kind", "")),
        "objective_value": float(item.get("objective_value", float("nan"))),
        "delta_gain": float(item.get("delta_gain", float("nan"))),
        "power_next_W_per_m": float(item.get("power_next_W_per_m", float("nan"))),
        "step_power_W": float(item.get("step_power_W", float("nan"))),
        "constraint_violation": float(item.get("constraint_violation", float("nan"))),
        "constraint_margins": dict(item.get("constraint_margins", {}) or {}),
        "max_abs_scaled_residual": float(item.get("max_abs_scaled_residual", float("nan"))),
        "step_error_kind": str(item.get("step_error_kind", "")),
        "physical_residual_scaled": float(item.get("physical_residual_scaled", float("nan"))),
        "physical_residual_ok": bool(item.get("physical_residual_ok", False)),
        "rk4_error_ok": bool(item.get("rk4_error_ok", False)),
        "rk4_error_estimate": float(item.get("rk4_error_estimate", float("nan"))),
        "rk4_error_margin": float(item.get("rk4_error_margin", float("nan"))),
        "rk4_substeps": int(item.get("rk4_substeps", -1)),
        "rk4_rhs_mode": str(item.get("rk4_rhs_mode", "")),
        "rk4_stage_ok": bool(item.get("rk4_stage_ok", True)),
        "rk4_stage_constraint_margins": dict(item.get("rk4_stage_constraint_margins", {}) or {}),
        "rk4_stage_min_Tp_K": float(item.get("rk4_stage_min_Tp_K", float("nan"))),
        "rk4_stage_max_mach": float(item.get("rk4_stage_max_mach", float("nan"))),
        "rk4_stage_min_G": float(item.get("rk4_stage_min_G", float("nan"))),
        "rk4_stage_max_cond_row_norm_log": float(item.get("rk4_stage_max_cond_row_norm_log", float("nan"))),
        "rk4_stage_min_singular_row_norm_log": float(
            item.get("rk4_stage_min_singular_row_norm_log", float("nan"))
        ),
        "rk4_stage_max_differential_replay_residual": float(
            item.get("rk4_stage_max_differential_replay_residual", float("nan"))
        ),
        "rk4_stage_max_log_rhs_norm": float(item.get("rk4_stage_max_log_rhs_norm", float("nan"))),
    }


def active_summary(
    *,
    nodes: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    settings: Any,
    objective: str,
) -> dict[str, Any]:
    support_counts: dict[str, int] = {}
    for seg in segments:
        key = str(seg.get("support_type", "unknown"))
        support_counts[key] = support_counts.get(key, 0) + 1
    g_margin_near_count = int(
        sum(1 for node in nodes[1:] if float(node["G"]) - float(settings.g_floor) <= float(settings.active_tol))
    )
    tp_margin_near_count = int(
        sum(1 for node in nodes[1:] if float(node["T_p"]) - float(settings.tp_floor_K) <= float(settings.active_tol))
    )
    sigmas = [float(node["sigma_logA"]) for node in nodes if np.isfinite(float(node["sigma_logA"]))]
    power_values = [
        float(node.get("power_density_W_per_m", float("nan")))
        for node in nodes
        if np.isfinite(float(node.get("power_density_W_per_m", float("nan"))))
    ]
    mhd_output_power_W = integrated_power_W(nodes)
    stage_cond_values = [
        float(seg.get("rk4_stage_max_cond_row_norm_log", float("nan")))
        for seg in segments
        if np.isfinite(float(seg.get("rk4_stage_max_cond_row_norm_log", float("nan"))))
    ]
    stage_tp_values = [
        float(seg.get("rk4_stage_min_Tp_K", float("nan")))
        for seg in segments
        if np.isfinite(float(seg.get("rk4_stage_min_Tp_K", float("nan"))))
    ]
    stage_mach_values = [
        float(seg.get("rk4_stage_max_mach", float("nan")))
        for seg in segments
        if np.isfinite(float(seg.get("rk4_stage_max_mach", float("nan"))))
    ]
    stage_replay_values = [
        float(seg.get("rk4_stage_max_differential_replay_residual", float("nan")))
        for seg in segments
        if np.isfinite(float(seg.get("rk4_stage_max_differential_replay_residual", float("nan"))))
    ]
    rk4_error_values = [
        float(seg.get("rk4_error_estimate", float("nan")))
        for seg in segments
        if np.isfinite(float(seg.get("rk4_error_estimate", float("nan"))))
    ]
    physical_residual_values = [
        float(seg.get("physical_residual_scaled", float("nan")))
        for seg in segments
        if np.isfinite(float(seg.get("physical_residual_scaled", float("nan"))))
    ]
    return {
        "objective": str(objective),
        "support_counts": support_counts,
        "termination": termination_summary(segments),
        "n_steps_requested": int(settings.n_steps),
        "n_steps_completed": int(len(segments)),
        "Delta_start": float(nodes[0]["Delta"]),
        "Delta_end": float(nodes[-1]["Delta"]),
        "Delta_gain": float(nodes[-1]["Delta"] - nodes[0]["Delta"]),
        "G_min_excluding_anchor": float(min(float(node["G"]) for node in nodes[1:])) if len(nodes) > 1 else None,
        "Tp_min_excluding_anchor_K": float(min(float(node["T_p"]) for node in nodes[1:])) if len(nodes) > 1 else None,
        "logA_min": float(min(float(node["logA"]) for node in nodes)),
        "logA_max": float(max(float(node["logA"]) for node in nodes)),
        "A_min": float(min(float(node["A"]) for node in nodes)),
        "A_max": float(max(float(node["A"]) for node in nodes)),
        "Te_min_K": float(min(float(node["T_e"]) for node in nodes)),
        "Te_max_K": float(max(float(node["T_e"]) for node in nodes)),
        "Tp_min_K": float(min(float(node["T_p"]) for node in nodes)),
        "Tp_max_K": float(max(float(node["T_p"]) for node in nodes)),
        "mach_min": float(min(float(node["mach"]) for node in nodes)),
        "mach_max": float(max(float(node["mach"]) for node in nodes)),
        "power_density_min_W_per_m": float(min(power_values)) if power_values else None,
        "power_density_max_W_per_m": float(max(power_values)) if power_values else None,
        "mhd_output_power_W": mhd_output_power_W,
        "mhd_output_power_MW": float(mhd_output_power_W / 1.0e6) if np.isfinite(mhd_output_power_W) else None,
        "sigma_min": float(min(sigmas)) if sigmas else None,
        "sigma_max": float(max(sigmas)) if sigmas else None,
        "max_abs_scaled_residual": (
            float(max(float(seg.get("max_abs_scaled_residual", 0.0)) for seg in segments))
            if segments
            else 0.0
        ),
        "rk4_error_estimate_max": float(max(rk4_error_values)) if rk4_error_values else None,
        "physical_residual_scaled_max": float(max(physical_residual_values)) if physical_residual_values else None,
        "rk4_stage_max_cond_row_norm_log": float(max(stage_cond_values)) if stage_cond_values else None,
        "rk4_stage_min_Tp_K": float(min(stage_tp_values)) if stage_tp_values else None,
        "rk4_stage_max_mach": float(max(stage_mach_values)) if stage_mach_values else None,
        "rk4_stage_max_differential_replay_residual": (
            float(max(stage_replay_values)) if stage_replay_values else None
        ),
        "G_active_count_excluding_anchor": int(
            support_counts.get("G_supported", 0)
            + support_counts.get("G_limited_reverse", 0)
            + support_counts.get("G_flat_reverse", 0)
        ),
        "G_margin_near_count_excluding_anchor": g_margin_near_count,
        "Tp_floor_active_count_excluding_anchor": int(support_counts.get("Tp_floor_supported", 0)),
        "Tp_floor_margin_near_count_excluding_anchor": tp_margin_near_count,
    }


def integrated_power_W(nodes: list[dict[str, Any]]) -> float:
    if len(nodes) < 2:
        return 0.0
    pairs = [(float(node["x"]), float(node.get("power_density_W_per_m", float("nan")))) for node in nodes]
    if not all(np.isfinite(x) and np.isfinite(p) for x, p in pairs):
        return float("nan")
    pairs.sort(key=lambda item: item[0])
    x = np.asarray([item[0] for item in pairs], dtype=float)
    p = np.asarray([item[1] for item in pairs], dtype=float)
    if float(x[-1]) <= float(x[0]):
        return 0.0
    return float(np.trapezoid(p, x))


def termination_summary(segments: list[dict[str, Any]]) -> dict[str, Any]:
    if not segments:
        return {"ok": True, "reason": "no_segments", "step": None}
    last = dict(segments[-1])
    ok = bool(all(bool(item.get("ok", False)) for item in segments))
    reason = (
        "completed_requested_steps"
        if ok
        else str(last.get("termination_reason") or last.get("support_type") or last.get("error") or "failed_step")
    )
    return {
        "ok": bool(ok),
        "reason": reason,
        "step": int(last.get("k", len(segments) - 1)),
        "support_type": str(last.get("support_type", "unknown")),
        "selected_support_type": str(last.get("selected_support_type", "")),
        "selected_sigma_origin": str(last.get("selected_sigma_origin", "")),
        "selected_sigma_source": str(last.get("selected_sigma_source", "")),
        "affine_support_type": str(last.get("affine_support_type", "")),
        "affine_objective_bound_kind": str(last.get("affine_objective_bound_kind", "")),
        "affine_selected_endpoint_source": str(last.get("affine_selected_endpoint_source", "")),
        "objective_bound_kind": str(last.get("objective_bound_kind", "")),
        "selected_endpoint_source": str(last.get("selected_endpoint_source", "")),
        "solver_method": str(last.get("solver_method", "unknown")),
        "sigma": float(last.get("sigma", float("nan"))),
        "sigma_interval_lower": float(last.get("sigma_interval_lower", float("nan"))),
        "sigma_interval_upper": float(last.get("sigma_interval_upper", float("nan"))),
        "bound_sources": dict(last.get("bound_sources", {}) or {}),
        "constraint_margins": dict(last.get("constraint_margins", {}) or {}),
        "boundary_blockers": list(last.get("boundary_blockers", []) or []),
        "step_error_kind": str(last.get("step_error_kind", "")),
        "max_abs_scaled_residual": float(last.get("max_abs_scaled_residual", float("nan"))),
        "physical_residual_scaled": float(last.get("physical_residual_scaled", float("nan"))),
        "physical_residual_ok": bool(last.get("physical_residual_ok", False)),
        "rk4_error_ok": bool(last.get("rk4_error_ok", False)),
        "rk4_error_estimate": float(last.get("rk4_error_estimate", float("nan"))),
        "rk4_error_margin": float(last.get("rk4_error_margin", float("nan"))),
        "rk4_substeps": int(last.get("rk4_substeps", -1)),
        "sonic_objective_score": float(last.get("sonic_objective_score", float("nan"))),
        "sonic_direction_ok": bool(last.get("sonic_direction_ok", False)),
        "sonic_direction_gate": str(last.get("sonic_direction_gate", "")),
        "sonic_compatibility_status": str(last.get("sonic_compatibility_status", "")),
        "sonic_compatibility_selected_sigma": float(last.get("sonic_compatibility_selected_sigma", float("nan"))),
        "sonic_compatibility_selected_scaled_residual": float(
            last.get("sonic_compatibility_selected_scaled_residual", float("nan"))
        ),
        "sonic_compatibility_best_interval_sigma": float(
            last.get("sonic_compatibility_best_interval_sigma", float("nan"))
        ),
        "sonic_compatibility_best_interval_scaled_residual": float(
            last.get("sonic_compatibility_best_interval_scaled_residual", float("nan"))
        ),
        "sonic_compatibility_variation_scaled": float(
            last.get("sonic_compatibility_variation_scaled", float("nan"))
        ),
        "sonic_compatibility_root_sigma": float(last.get("sonic_compatibility_root_sigma", float("nan"))),
        "sign_aware_fallback_status": str(last.get("sign_aware_fallback_status", "")),
        "sign_aware_fallback_attempted": bool(last.get("sign_aware_fallback_attempted", False)),
        "sign_aware_fallback_used": bool(last.get("sign_aware_fallback_used", False)),
        "sign_aware_fallback_recovered": bool(last.get("sign_aware_fallback_recovered", False)),
        "sign_aware_fallback_solver_method": str(last.get("sign_aware_fallback_solver_method", "")),
        "sign_aware_fallback_validation_failure": str(last.get("sign_aware_fallback_validation_failure", "")),
        "sign_aware_endpoint_sigma": float(last.get("sign_aware_endpoint_sigma", float("nan"))),
        "sign_aware_endpoint_ok": bool(last.get("sign_aware_endpoint_ok", False)),
        "sign_aware_endpoint_feasible": bool(last.get("sign_aware_endpoint_feasible", False)),
        "sign_aware_endpoint_solver_method": str(last.get("sign_aware_endpoint_solver_method", "")),
        "sign_aware_endpoint_validation_failure": str(last.get("sign_aware_endpoint_validation_failure", "")),
        "sign_aware_endpoint_constraint_violation": float(
            last.get("sign_aware_endpoint_constraint_violation", float("nan"))
        ),
        "sign_aware_endpoint_constraint_margins": dict(
            last.get("sign_aware_endpoint_constraint_margins", {}) or {}
        ),
        "error": None if last.get("error") is None else str(last.get("error")),
        "reverse_interval_error": str(last.get("reverse_interval_error", "")),
        "reverse_interval_conflict": bool(last.get("reverse_interval_conflict", False)),
        "reverse_interval_conflict_kind": str(last.get("reverse_interval_conflict_kind", "")),
        "reverse_interval_conflict_summary": str(last.get("reverse_interval_conflict_summary", "")),
        "reverse_interval_conflict_lower_source": str(last.get("reverse_interval_conflict_lower_source", "")),
        "reverse_interval_conflict_upper_source": str(last.get("reverse_interval_conflict_upper_source", "")),
        "reverse_interval_conflict_sigma_lower": float(last.get("reverse_interval_conflict_sigma_lower", float("nan"))),
        "reverse_interval_conflict_sigma_upper": float(last.get("reverse_interval_conflict_sigma_upper", float("nan"))),
        "reverse_interval_conflict_sigma_gap": float(last.get("reverse_interval_conflict_sigma_gap", float("nan"))),
        "reverse_interval_conflict_Aprime_lower": float(
            last.get("reverse_interval_conflict_Aprime_lower", float("nan"))
        ),
        "reverse_interval_conflict_Aprime_upper": float(
            last.get("reverse_interval_conflict_Aprime_upper", float("nan"))
        ),
        "reverse_interval_conflict_Aprime_gap": float(last.get("reverse_interval_conflict_Aprime_gap", float("nan"))),
        "scan_diagnostics": dict(last.get("scan_diagnostics", {}) or {}),
    }
