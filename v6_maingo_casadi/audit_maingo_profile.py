#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_maingo_casadi.constants import _G_HARD_MARGIN, _TP_MIN
from v6_maingo_casadi.geometry import SplineAreaDesign
from v6_maingo_casadi.numerics import _ops_for_numeric
from v6_maingo_casadi.physics import _dynamic_system_terms
from v6_maingo_casadi.profiles import _normalize_working_fluid_profile


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonify(value: Any):
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonify(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _resolve_summary_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "maingo_summary.json"
    return candidate.resolve()


def _resolve_profile_path(summary_path: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    for name in ("maingo_handoff_profile.npz", "maingo_best_profile.npz"):
        candidate = summary_path.with_name(name).resolve()
        if candidate.exists():
            return candidate
    return summary_path.with_name("maingo_handoff_profile.npz").resolve()


def _tp_min_from_summary(summary: dict[str, Any]) -> float:
    baseline = dict(summary.get("baseline_seed", {}) or {})
    baseline_path = baseline.get("summary_path")
    if baseline_path:
        path = Path(str(baseline_path))
        if path.exists():
            schedule = list(_load_json(path).get("schedule", []) or [])
            if schedule and "tp_min" in schedule[0]:
                return float(schedule[0]["tp_min"])
    return float(_TP_MIN)


def _working_fluid_from_summary(summary: dict[str, Any]):
    for key in ("working_fluid_profile", "working_fluid"):
        if key in summary:
            return _normalize_working_fluid_profile(summary[key])
    baseline = dict(summary.get("baseline_seed", {}) or {})
    if "working_fluid" in baseline:
        return _normalize_working_fluid_profile(baseline["working_fluid"])
    return _normalize_working_fluid_profile(None)


def _min_record(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    if not records:
        return {}
    return min(records, key=lambda item: float(item[field]))


def _closure_records(
    *,
    n_p: np.ndarray,
    T_e: np.ndarray,
    A: np.ndarray,
    sigma: np.ndarray,
    x: np.ndarray,
    dot_N: float,
    I_0: float,
    seed_fraction: float,
    B: float,
    working_fluid,
) -> list[dict[str, Any]]:
    ops = _ops_for_numeric()
    if sigma.size == A.size - 1:
        sigma_nodes = np.concatenate([sigma, sigma[-1:]])
    else:
        sigma_nodes = np.resize(sigma, A.size)
    records = []
    for idx in range(A.size):
        closure, _ = _dynamic_system_terms(
            ops=ops,
            n_p=float(n_p[idx]),
            T_e=float(T_e[idx]),
            A=float(A[idx]),
            sigma=float(sigma_nodes[idx]),
            dot_N=float(dot_N),
            I_0=float(I_0),
            seed_fraction=float(seed_fraction),
            B=float(B),
            working_fluid=working_fluid,
        )
        records.append(
            {
                "index": int(idx),
                "x_m": float(x[idx]),
                "A_m2": float(A[idx]),
                "n_p_A": float(n_p[idx] * A[idx]),
                "T_e_K": float(T_e[idx]),
                "T_p_K": float(closure["T_p"]),
                "v_p_m_s": float(closure["v_p"]),
                "Z": float(closure["Z"]),
                "velikhov_margin": float(closure["G"]),
            }
        )
    return records


def _midpoint_records(
    *,
    n_p: np.ndarray,
    T_e: np.ndarray,
    A_mid: np.ndarray,
    sigma_mid: np.ndarray,
    x: np.ndarray,
    dot_N: float,
    I_0: float,
    seed_fraction: float,
    B: float,
    working_fluid,
) -> list[dict[str, Any]]:
    ops = _ops_for_numeric()
    records = []
    for idx in range(A_mid.size):
        n_mid = 0.5 * (float(n_p[idx]) + float(n_p[idx + 1]))
        te_mid = 0.5 * (float(T_e[idx]) + float(T_e[idx + 1]))
        x_mid = 0.5 * (float(x[idx]) + float(x[idx + 1]))
        closure, _ = _dynamic_system_terms(
            ops=ops,
            n_p=n_mid,
            T_e=te_mid,
            A=float(A_mid[idx]),
            sigma=float(sigma_mid[idx]),
            dot_N=float(dot_N),
            I_0=float(I_0),
            seed_fraction=float(seed_fraction),
            B=float(B),
            working_fluid=working_fluid,
        )
        records.append(
            {
                "interval": int(idx),
                "x_m": float(x_mid),
                "A_m2": float(A_mid[idx]),
                "n_p_A": float(n_mid * A_mid[idx]),
                "T_e_K": float(te_mid),
                "T_p_K": float(closure["T_p"]),
                "v_p_m_s": float(closure["v_p"]),
                "Z": float(closure["Z"]),
                "velikhov_margin": float(closure["G"]),
            }
        )
    return records


def _area_reconstruction_audit(
    *,
    summary: dict[str, Any],
    x: np.ndarray,
    A: np.ndarray,
    length: float,
    area_scale: float,
) -> dict[str, Any]:
    coarse = dict(summary.get("coarse_best", {}) or {})
    area_design_payload = dict(coarse.get("area_design", {}) or {})
    baseline = dict(summary.get("baseline_seed", {}) or {})
    legacy_area_reference = bool(dict(baseline.get("area_reference", {}) or {}).get("enabled", False))
    x_norm = (x - float(x[0])) / max(float(x[-1] - x[0]), 1e-30)
    audit: dict[str, Any] = {
        "legacy_area_reference_enabled": legacy_area_reference,
        "stored_A_min_m2": float(np.min(A)),
        "stored_A_max_m2": float(np.max(A)),
        "stored_outlet_to_inlet_ratio": float(A[-1] / max(float(A[0]), 1e-30)),
    }
    if area_design_payload:
        direct = SplineAreaDesign(
            a1=float(area_design_payload["a1"]),
            a2=float(area_design_payload["a2"]),
            a3=float(area_design_payload["a3"]),
        )
        direct_profile = direct.evaluate_on_normalized_grid(x_norm, length=float(length), area_scale=float(area_scale))
        A_direct = np.asarray(direct_profile["A"], dtype=float)
        log_ratio = np.log(np.maximum(A_direct, 1e-300)) - np.log(np.maximum(A, 1e-300))
        rel = np.abs(A_direct - A) / np.maximum(np.abs(A), 1e-300)
        worst = int(np.argmax(np.abs(log_ratio)))
        audit["summary_area_design"] = direct.to_dict()
        audit["direct_spline_outlet_to_inlet_ratio"] = float(A_direct[-1] / max(float(A_direct[0]), 1e-30))
        audit["max_abs_logA_mismatch"] = float(np.max(np.abs(log_ratio)))
        audit["max_relative_A_mismatch"] = float(np.max(rel))
        audit["worst_direct_spline_mismatch"] = {
            "index": worst,
            "x_m": float(x[worst]),
            "stored_A_m2": float(A[worst]),
            "direct_spline_A_m2": float(A_direct[worst]),
            "relative_mismatch": float(rel[worst]),
        }
        audit["direct_spline_self_consistent"] = bool(np.max(np.abs(log_ratio)) <= 1e-3)
    else:
        audit["direct_spline_self_consistent"] = None

    projected = SplineAreaDesign.project_from_profile(x=x, A=A)
    projected_profile = projected.evaluate_on_normalized_grid(x_norm, length=float(length), area_scale=float(area_scale))
    A_projected = np.asarray(projected_profile["A"], dtype=float)
    projected_log_ratio = np.log(np.maximum(A_projected, 1e-300)) - np.log(np.maximum(A, 1e-300))
    audit["projected_area_design_from_npz"] = projected.to_dict()
    audit["projected_outlet_to_inlet_ratio"] = float(A_projected[-1] / max(float(A_projected[0]), 1e-30))
    audit["projected_max_abs_logA_mismatch"] = float(np.max(np.abs(projected_log_ratio)))
    return audit


def audit_maingo_profile(summary_path: str | Path, profile_path: str | Path | None = None) -> dict[str, Any]:
    summary_path = _resolve_summary_path(summary_path)
    profile_path = _resolve_profile_path(summary_path, str(profile_path) if profile_path else None)
    summary = _load_json(summary_path)
    with np.load(profile_path) as data:
        x = np.asarray(data["x"], dtype=float)
        n_p = np.asarray(data["n_p"], dtype=float)
        T_e = np.asarray(data["T_e"], dtype=float)
        T_p_stored = np.asarray(data["T_p"], dtype=float)
        A = np.asarray(data["A"], dtype=float)
        sigma = np.asarray(data["sigma_logA"], dtype=float)
        v_p = np.asarray(data["v_p"], dtype=float)
        J_x = np.asarray(data["J_x"], dtype=float)
        velikhov = np.asarray(data["velikhov_margin"], dtype=float)
        seed_profile = np.asarray(data["seed_fraction"], dtype=float) if "seed_fraction" in data.files else np.array([])

    coarse = dict(summary.get("coarse_best", {}) or {})
    decision = dict(coarse.get("decision_vector", {}) or {})
    baseline = dict(summary.get("baseline_seed", {}) or {})
    B = float(baseline.get("B", 3.0))
    length = float(baseline.get("L", x[-1] - x[0]))
    area_scale = float(baseline.get("area_scale_m2", A[0]))
    I_0 = float(decision.get("I_0", J_x[0] * A[0]))
    if "log_seed_fraction" in decision:
        seed_fraction = float(math.exp(float(decision["log_seed_fraction"])))
    elif seed_profile.size:
        seed_fraction = float(np.median(seed_profile))
    else:
        seed_fraction = 1e-4
    dot_N_nodes = n_p * v_p * A
    dot_N = float(np.median(dot_N_nodes))
    tp_min = _tp_min_from_summary(summary)
    working_fluid = _working_fluid_from_summary(summary)

    node_records = _closure_records(
        n_p=n_p,
        T_e=T_e,
        A=A,
        sigma=sigma,
        x=x,
        dot_N=dot_N,
        I_0=I_0,
        seed_fraction=seed_fraction,
        B=B,
        working_fluid=working_fluid,
    )
    linear_mid = _midpoint_records(
        n_p=n_p,
        T_e=T_e,
        A_mid=0.5 * (A[:-1] + A[1:]),
        sigma_mid=sigma if sigma.size == A.size - 1 else 0.5 * (sigma[:-1] + sigma[1:]),
        x=x,
        dot_N=dot_N,
        I_0=I_0,
        seed_fraction=seed_fraction,
        B=B,
        working_fluid=working_fluid,
    )
    log_mid = _midpoint_records(
        n_p=n_p,
        T_e=T_e,
        A_mid=np.exp(0.5 * (np.log(np.maximum(A[:-1], 1e-300)) + np.log(np.maximum(A[1:], 1e-300)))),
        sigma_mid=sigma if sigma.size == A.size - 1 else 0.5 * (sigma[:-1] + sigma[1:]),
        x=x,
        dot_N=dot_N,
        I_0=I_0,
        seed_fraction=seed_fraction,
        B=B,
        working_fluid=working_fluid,
    )
    area_audit = _area_reconstruction_audit(
        summary=summary,
        x=x,
        A=A,
        length=length,
        area_scale=area_scale,
    )

    direct_mid_records: list[dict[str, Any]] = []
    area_design_payload = dict(coarse.get("area_design", {}) or {})
    if area_design_payload:
        x_mid_norm = (np.arange(A.size - 1, dtype=float) + 0.5) / float(A.size - 1)
        direct = SplineAreaDesign(
            a1=float(area_design_payload["a1"]),
            a2=float(area_design_payload["a2"]),
            a3=float(area_design_payload["a3"]),
        )
        direct_mid = direct.evaluate_on_normalized_grid(
            x_mid_norm,
            length=float(length),
            area_scale=float(area_scale),
        )
        direct_mid_records = _midpoint_records(
            n_p=n_p,
            T_e=T_e,
            A_mid=np.asarray(direct_mid["A"], dtype=float),
            sigma_mid=np.asarray(direct_mid["sigma_logA"], dtype=float),
            x=x,
            dot_N=dot_N,
            I_0=I_0,
            seed_fraction=seed_fraction,
            B=B,
            working_fluid=working_fluid,
        )

    stored_area_min_tp = min(
        float(np.min(T_p_stored)),
        float(_min_record(node_records, "T_p_K")["T_p_K"]),
        float(_min_record(linear_mid, "T_p_K")["T_p_K"]),
        float(_min_record(log_mid, "T_p_K")["T_p_K"]),
    )
    stored_area_min_g = min(
        float(np.min(velikhov)),
        float(_min_record(node_records, "velikhov_margin")["velikhov_margin"]),
        float(_min_record(linear_mid, "velikhov_margin")["velikhov_margin"]),
        float(_min_record(log_mid, "velikhov_margin")["velikhov_margin"]),
    )
    direct_min_tp = float(_min_record(direct_mid_records, "T_p_K")["T_p_K"]) if direct_mid_records else None
    direct_consistent = area_audit.get("direct_spline_self_consistent")
    if stored_area_min_tp < tp_min:
        verdict = "profile_collapses_tp_min_under_stored_area"
    elif stored_area_min_g < float(_G_HARD_MARGIN):
        verdict = "stored_profile_passes_tp_check_but_fails_dense_velikhov_check"
    elif direct_consistent is False:
        verdict = "stored_profile_passes_tp_check_but_summary_area_design_is_not_self_consistent"
    else:
        verdict = "stored_profile_passes_tp_check_and_area_reconstruction_is_self_consistent"

    return {
        "summary_path": str(summary_path),
        "profile_path": str(profile_path),
        "objective_profile": summary.get("objective_profile"),
        "B_T": B,
        "L_m": length,
        "area_scale_m2": area_scale,
        "I_0_A": I_0,
        "seed_fraction": seed_fraction,
        "tp_min_threshold_K": tp_min,
        "velikhov_hard_margin": float(_G_HARD_MARGIN),
        "dot_N_consistency": {
            "median": dot_N,
            "min_over_median": float(np.min(dot_N_nodes) / max(abs(dot_N), 1e-300)),
            "max_over_median": float(np.max(dot_N_nodes) / max(abs(dot_N), 1e-300)),
        },
        "stored_profile_checks": {
            "stored_npz_min_T_p_K": float(np.min(T_p_stored)),
            "recomputed_node_min": _min_record(node_records, "T_p_K"),
            "stored_linear_midpoint_min": _min_record(linear_mid, "T_p_K"),
            "stored_log_midpoint_min": _min_record(log_mid, "T_p_K"),
            "stored_npz_min_velikhov_margin": float(np.min(velikhov)),
            "recomputed_node_min_velikhov": _min_record(node_records, "velikhov_margin"),
            "stored_linear_midpoint_min_velikhov": _min_record(linear_mid, "velikhov_margin"),
            "stored_log_midpoint_min_velikhov": _min_record(log_mid, "velikhov_margin"),
            "stored_area_min_T_p_all_checks_K": stored_area_min_tp,
            "stored_area_min_velikhov_all_checks": stored_area_min_g,
            "passes_tp_min_under_stored_area": bool(stored_area_min_tp >= tp_min),
            "passes_velikhov_under_stored_area": bool(stored_area_min_g >= float(_G_HARD_MARGIN)),
        },
        "area_reconstruction": area_audit,
        "direct_spline_midpoint_probe": {
            "enabled": bool(direct_mid_records),
            "min": _min_record(direct_mid_records, "T_p_K") if direct_mid_records else {},
            "min_T_p_K": direct_min_tp,
            "note": (
                "This probe is valid only when summary area_design is already in direct spline semantics; "
                "legacy area_reference artifacts should be regenerated before they are used for new searches."
            ),
        },
        "verdict": verdict,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CasADi-free audit of a MAiNGO profile's stored-area T_p and area-design semantics."
    )
    parser.add_argument("summary_or_dir", type=str, help="maingo_summary.json path or its output directory")
    parser.add_argument("--profile", type=str, default="", help="optional MAiNGO profile NPZ path")
    parser.add_argument("--out", type=str, default="", help="optional audit JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary_path = _resolve_summary_path(args.summary_or_dir)
    audit = audit_maingo_profile(summary_path, args.profile or None)
    out_path = Path(args.out) if args.out else summary_path.with_name("maingo_profile_audit.json")
    out_path.write_text(json.dumps(_jsonify(audit), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonify(audit), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
