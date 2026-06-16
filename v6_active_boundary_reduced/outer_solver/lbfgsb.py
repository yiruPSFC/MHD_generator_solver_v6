from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from ..objective import (
    AnchorOptions,
    PreparationObjectiveWeights,
    evaluate_preparation_design,
    flatten_result_for_csv,
    load_base_config,
)
from ..policy import PreparationSettings
from .prescreen import (
    CONTROL_VARIABLE_NAMES,
    PrescreenSettings,
    from_normalized,
    overrides_from_values,
    prescreen_candidates,
    variable_bounds,
)
from .reward import OuterRewardWeights, score_outer_result


@dataclass(frozen=True)
class LbfgsbOuterSolverConfig:
    variable_names: tuple[str, ...] = CONTROL_VARIABLE_NAMES
    maxiter: int = 24
    maxfun: int = 160
    finite_diff_eps: float = 1e-3
    neighborhood_eps: float = 1e-3
    require_robust_seed: bool = True
    run_lbfgsb_without_robust_seed: bool = False
    certify_top_k: int = 8


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def _payload_without_heavy_result(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "result"}


def _profile_arrays_from_nodes(nodes: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    fields = ("x", "n_p", "T_e", "T_p", "A", "logA", "sigma_logA", "G", "Delta", "mach", "beta", "Z")
    arrays: dict[str, np.ndarray] = {}
    for field in fields:
        values = [float(node[field]) for node in nodes if field in node]
        if len(values) == len(nodes):
            arrays[field] = np.asarray(values, dtype=float)
    return arrays


def _write_best_outputs(out_dir: Path, result: dict[str, Any]) -> dict[str, str]:
    payload = dict(result.get("payload", {}) or {})
    nodes = list(payload.get("nodes", []) or [])
    segments = list(payload.get("segments", []) or [])
    outputs: dict[str, str] = {}
    if nodes:
        nodes_path = out_dir / "best_nodes.csv"
        _write_csv(nodes_path, [dict(node) for node in nodes])
        profile_path = out_dir / "best_profile.npz"
        np.savez(profile_path, **_profile_arrays_from_nodes([dict(node) for node in nodes]))
        outputs["best_nodes_csv"] = str(nodes_path)
        outputs["best_profile_npz"] = str(profile_path)
    if segments:
        segments_path = out_dir / "best_segments.csv"
        _write_csv(segments_path, [dict(seg) for seg in segments])
        outputs["best_segments_csv"] = str(segments_path)
    result_path = out_dir / "best_result.json"
    _write_json(result_path, result)
    outputs["best_result_json"] = str(result_path)
    return outputs


def _feasible_result(result: dict[str, Any]) -> bool:
    return bool(result.get("ok", False)) and np.isfinite(float(result.get("outer_score", np.nan)))


def _neighbor_points(
    y: np.ndarray,
    *,
    eps: float,
    variable_names: tuple[str, ...],
) -> list[tuple[str, np.ndarray]]:
    points: list[tuple[str, np.ndarray]] = []
    y = np.clip(np.asarray(y, dtype=float), 0.0, 1.0)
    step = max(float(eps), 0.0)
    if step <= 0.0:
        return points
    for index, name in enumerate(variable_names):
        for sign, suffix in ((-1.0, "minus"), (1.0, "plus")):
            candidate = np.array(y, dtype=float)
            candidate[index] += sign * step
            if 0.0 <= float(candidate[index]) <= 1.0:
                points.append((f"{name}_{suffix}", candidate))
    return points


def _certification_public(diag: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for item in list(diag.get("evaluations", []) or []):
        result = dict(item.get("result", {}) or {})
        rows.append(
            {
                "label": item.get("label"),
                "case_id": result.get("case_id"),
                "ok": bool(result.get("ok", False)),
                "outer_score": float(result.get("outer_score", np.nan)),
                "failure": result.get("failure"),
                "failure_diagnostics": result.get("failure_diagnostics", {}),
                "optimizer_normalized": result.get("optimizer_normalized", []),
            }
        )
    return {
        "label": diag.get("label"),
        "ok": bool(diag.get("ok", False)),
        "center_case_id": diag.get("center_case_id"),
        "center_ok": bool(diag.get("center_ok", False)),
        "center_outer_score": float(diag.get("center_outer_score", np.nan)),
        "neighbor_ok_count": int(diag.get("neighbor_ok_count", 0)),
        "neighbor_count": int(diag.get("neighbor_count", 0)),
        "failed_labels": list(diag.get("failed_labels", []) or []),
        "evaluations": rows,
    }


def run_outer_lbfgsb(
    *,
    base_config,
    lower: np.ndarray,
    upper: np.ndarray,
    out_dir: Path,
    solver_config: LbfgsbOuterSolverConfig,
    prescreen_settings: PrescreenSettings,
    rollout_settings: PreparationSettings,
    rollout_weights: PreparationObjectiveWeights,
    reward_weights: OuterRewardWeights,
    anchor_options: AnchorOptions,
    fixed_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    variable_names = tuple(solver_config.variable_names)
    fixed_overrides = {str(k): float(v) for k, v in dict(fixed_overrides or {}).items()}
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    prescreen = prescreen_candidates(
        base_config=base_config,
        lower=lower,
        upper=upper,
        variable_names=variable_names,
        settings=prescreen_settings,
        rollout_settings=rollout_settings,
        rollout_weights=rollout_weights,
        reward_weights=reward_weights,
        anchor_options=anchor_options,
        fixed_overrides=fixed_overrides,
    )
    if not prescreen:
        raise RuntimeError(
            "prescreen produced no optimizer seeds. Relax the Te/Tp/G/gradient thresholds "
            "or pass --allow-prescreen-fallback."
        )
    _write_jsonl(out_dir / "prescreen.jsonl", [_payload_without_heavy_result(row) for row in prescreen])

    evaluations: list[dict[str, Any]] = []
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    best_seen: dict[str, Any] | None = None
    counter = 0

    def evaluate_y(y_values: np.ndarray, *, source: str) -> dict[str, Any]:
        nonlocal counter, best_seen
        y = np.clip(np.asarray(y_values, dtype=float), 0.0, 1.0)
        key = tuple(round(float(value), 12) for value in y)
        if key in cache:
            return cache[key]
        values = from_normalized(y, lower=lower, upper=upper)
        result = evaluate_preparation_design(
            base_config=base_config,
            design_overrides=overrides_from_values(
                values,
                variable_names=variable_names,
                fixed_overrides=fixed_overrides,
            ),
            settings=rollout_settings,
            weights=rollout_weights,
            anchor_options=anchor_options,
            case_id=counter,
            return_payload=True,
        )
        scored = score_outer_result(result, weights=reward_weights)
        result["outer_score"] = float(scored["score"])
        result["outer_reward_terms"] = dict(scored["terms"])
        result["optimizer_source"] = str(source)
        result["optimizer_normalized"] = [float(value) for value in y]
        evaluations.append(result)
        counter += 1
        if _feasible_result(result) and (
            best_seen is None or float(result["outer_score"]) > float(best_seen["outer_score"])
        ):
            best_seen = result
        cache[key] = result
        return result

    def objective_y(y_values: np.ndarray, *, source: str) -> float:
        result = evaluate_y(y_values, source=source)
        if not _feasible_result(result):
            return float(reward_weights.failure_penalty)
        fun = -float(result["outer_score"])
        if not np.isfinite(fun):
            fun = float(reward_weights.failure_penalty)
        return float(fun)

    def certify_neighborhood(y_values: np.ndarray, *, label: str) -> dict[str, Any]:
        y = np.clip(np.asarray(y_values, dtype=float), 0.0, 1.0)
        center = evaluate_y(y, source=f"{label}:center")
        records = [{"label": "center", "result": center}]
        failed_labels: list[str] = []
        if not _feasible_result(center):
            failed_labels.append("center")
        for neighbor_label, neighbor_y in _neighbor_points(
            y,
            eps=float(solver_config.neighborhood_eps),
            variable_names=variable_names,
        ):
            item = evaluate_y(neighbor_y, source=f"{label}:{neighbor_label}")
            records.append({"label": neighbor_label, "result": item})
            if not _feasible_result(item):
                failed_labels.append(neighbor_label)
        neighbor_count = max(len(records) - 1, 0)
        neighbor_ok_count = sum(1 for item in records[1:] if _feasible_result(dict(item["result"])))
        return {
            "label": str(label),
            "ok": not failed_labels,
            "center_case_id": center.get("case_id"),
            "center_ok": _feasible_result(center),
            "center_outer_score": float(center.get("outer_score", np.nan)),
            "neighbor_ok_count": int(neighbor_ok_count),
            "neighbor_count": int(neighbor_count),
            "failed_labels": failed_labels,
            "evaluations": records,
        }

    seed_certifications: list[dict[str, Any]] = []
    optimizer_seeds: list[tuple[int, np.ndarray]] = []
    for seed_index, seed_row in enumerate(prescreen):
        y0 = np.asarray(seed_row["normalized"], dtype=float)
        cert = certify_neighborhood(y0, label=f"seed_{seed_index}_neighborhood")
        seed_certifications.append(cert)
        if bool(cert["ok"]):
            optimizer_seeds.append((seed_index, y0))

    skipped_reason = None
    if not bool(solver_config.require_robust_seed):
        optimizer_seeds = [
            (seed_index, np.asarray(seed_row["normalized"], dtype=float))
            for seed_index, seed_row in enumerate(prescreen)
        ]
    elif not optimizer_seeds and bool(solver_config.run_lbfgsb_without_robust_seed):
        skipped_reason = "running L-BFGS-B without robust seed by explicit config"
        optimizer_seeds = [
            (seed_index, np.asarray(seed_row["normalized"], dtype=float))
            for seed_index, seed_row in enumerate(prescreen)
        ]
    elif not optimizer_seeds:
        skipped_reason = "no robust feasible seed passed neighborhood certification"

    scipy_runs: list[dict[str, Any]] = []
    for seed_index, y0 in optimizer_seeds:
        result = minimize(
            lambda y, idx=seed_index: objective_y(y, source=f"lbfgsb_seed_{idx}"),
            y0,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * len(variable_names),
            options={
                "maxiter": int(solver_config.maxiter),
                "maxfun": int(solver_config.maxfun),
                "eps": float(solver_config.finite_diff_eps),
                "ftol": 1e-8,
                "gtol": 1e-5,
            },
        )
        final_certification = certify_neighborhood(
            np.asarray(result.x, dtype=float),
            label=f"lbfgsb_seed_{seed_index}_final_neighborhood",
        )
        final_values = from_normalized(np.asarray(result.x, dtype=float), lower=lower, upper=upper)
        scipy_runs.append(
            {
                "seed_index": int(seed_index),
                "success": bool(result.success),
                "message": str(result.message),
                "fun": float(result.fun),
                "nit": int(result.nit),
                "nfev": int(result.nfev),
                "initial_normalized": [float(value) for value in y0],
                "final_normalized": [float(value) for value in np.asarray(result.x, dtype=float)],
                "final_values": [float(value) for value in final_values],
                "final_overrides": overrides_from_values(
                    final_values,
                    variable_names=variable_names,
                    fixed_overrides=fixed_overrides,
                ),
                "final_neighborhood": _certification_public(final_certification),
            }
        )

    feasible_seen = sorted(
        [item for item in evaluations if _feasible_result(item)],
        key=lambda item: float(item["outer_score"]),
        reverse=True,
    )
    post_certifications: list[dict[str, Any]] = []
    certified_candidates: list[dict[str, Any]] = []
    for item in feasible_seen[: max(int(solver_config.certify_top_k), 0)]:
        y = np.asarray(item.get("optimizer_normalized", []), dtype=float)
        if y.size != len(variable_names):
            continue
        cert = certify_neighborhood(y, label=f"post_best_case_{item.get('case_id')}_neighborhood")
        post_certifications.append(cert)
        if bool(cert["ok"]):
            certified_candidates.append(dict(cert["evaluations"][0]["result"]))

    certified_best = (
        max(certified_candidates, key=lambda item: float(item.get("outer_score", -np.inf)))
        if certified_candidates
        else None
    )
    best_result = certified_best or {}
    flat_rows = []
    for item in evaluations:
        row = flatten_result_for_csv(item)
        row["outer_score"] = float(item.get("outer_score", np.nan))
        row["optimizer_source"] = str(item.get("optimizer_source", ""))
        for name, value in dict(item.get("outer_reward_terms", {}) or {}).items():
            row[f"outer_{name}"] = float(value)
        flat_rows.append(row)
    _write_csv(out_dir / "evaluations.csv", flat_rows)
    _write_jsonl(out_dir / "evaluations.jsonl", evaluations)
    best_outputs = _write_best_outputs(out_dir, best_result) if best_result else {}

    summary = {
        "ok": bool(best_result.get("ok", False)),
        "elapsed_s": float(time.perf_counter() - started),
        "method": "feasibility-gated scipy.optimize.minimize:L-BFGS-B",
        "variable_names": list(variable_names),
        "bounds": {
            name: {"min": float(lo), "max": float(hi)}
            for name, lo, hi in zip(variable_names, lower, upper, strict=True)
        },
        "fixed_overrides": fixed_overrides,
        "solver_config": solver_config.__dict__,
        "prescreen_settings": prescreen_settings.__dict__,
        "rollout_settings": rollout_settings.__dict__,
        "rollout_weights": rollout_weights.__dict__,
        "reward_weights": reward_weights.__dict__,
        "anchor_options": anchor_options.__dict__,
        "lbfgsb_skipped_reason": skipped_reason,
        "seed_certifications": [_certification_public(item) for item in seed_certifications],
        "scipy_runs": scipy_runs,
        "post_best_certifications": [_certification_public(item) for item in post_certifications],
        "best_seen_uncertified": (
            {
                key: value
                for key, value in dict(best_seen or {}).items()
                if key not in {"payload"}
            }
            if best_seen is not None
            else {}
        ),
        "best": {
            key: value
            for key, value in best_result.items()
            if key not in {"payload"}
        },
        "outputs": {
            "summary_json": str(out_dir / "optimization_summary.json"),
            "prescreen_jsonl": str(out_dir / "prescreen.jsonl"),
            "evaluations_csv": str(out_dir / "evaluations.csv"),
            "evaluations_jsonl": str(out_dir / "evaluations.jsonl"),
            **best_outputs,
        },
    }
    _write_json(out_dir / "optimization_summary.json", summary)
    return summary


def _parse_pairs(items: list[list[str]] | None) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw in items or []:
        values[str(raw[0])] = float(raw[1])
    return values


def _parse_bounds(items: list[list[str]] | None) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for raw in items or []:
        name = str(raw[0])
        lo = float(raw[1])
        hi = float(raw[2])
        if name in {"np_in", "n_p_in"}:
            if lo <= 0.0 or hi <= 0.0:
                raise ValueError(f"{name} bounds must be positive before log transform.")
            name = "log_n_p_in"
            lo = float(np.log(lo))
            hi = float(np.log(hi))
        if hi < lo:
            raise ValueError(f"upper bound is smaller than lower bound for {name}.")
        bounds[name] = (lo, hi)
    return bounds


def _settings_from_args(args: argparse.Namespace) -> PreparationSettings:
    return PreparationSettings(
        n_steps=int(args.n_steps),
        dx=float(args.dx),
        objective=str(args.objective),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        curvature_max=None if bool(args.no_curvature_bound) else float(args.curvature_max),
        g_floor=float(args.g_floor),
        tp_floor_K=float(args.tp_floor),
        scan_points=int(args.scan_points),
        refine_iterations=int(args.refine_iterations),
        active_tol=float(args.active_tol),
        sonic_mode=str(args.sonic_mode),
        sonic_mach_tol=float(args.sonic_mach_tol),
        sonic_det_abs_tol=float(args.sonic_det_abs_tol),
        sonic_compatibility_tol=float(args.sonic_compatibility_tol),
        sonic_residual_tol=float(args.sonic_residual_tol),
        rk4_substeps=int(args.rk4_substeps),
        rk4_error_tol=float(args.rk4_error_tol),
        g_boundary_fallback_mode=str(args.g_boundary_fallback_mode),
    )


def _rollout_weights_from_args(args: argparse.Namespace) -> PreparationObjectiveWeights:
    return PreparationObjectiveWeights(
        delta_improvement=float(args.rollout_delta_improvement_weight),
        inlet_delta=float(args.rollout_inlet_delta_weight),
        inlet_te_floor_K=float(args.rollout_inlet_te_floor),
        inlet_tp_floor_K=float(args.rollout_inlet_tp_floor),
        inlet_te_shortfall=float(args.rollout_inlet_te_shortfall_weight),
        inlet_tp_shortfall=float(args.rollout_inlet_tp_shortfall_weight),
        temperature_scale_K=float(args.temperature_scale),
        failure_penalty=float(args.failure_penalty),
    )


def _reward_weights_from_args(args: argparse.Namespace) -> OuterRewardWeights:
    return OuterRewardWeights(
        delta_improvement=float(args.delta_improvement_weight),
        mhd_output_power_MW=float(args.mhd_output_power_weight),
        min_tp_floor_K=float(args.min_tp_floor),
        min_tp_shortfall=float(args.min_tp_shortfall_weight),
        max_te_ceiling_K=float(args.max_te_ceiling),
        max_te_excess=float(args.max_te_excess_weight),
        temperature_scale_K=float(args.temperature_scale),
        area_ratio_min=float(args.area_ratio_min),
        area_ratio_max=float(args.area_ratio_max),
        area_ratio_penalty=float(args.area_ratio_penalty_weight),
        magnetic_field_min_T=float(args.magnetic_field_min),
        magnetic_field_max_T=float(args.magnetic_field_max),
        magnetic_field_penalty=float(args.magnetic_field_penalty_weight),
        g_floor=float(args.g_floor),
        g_shortfall=float(args.g_shortfall_weight),
        g_scale=float(args.g_scale),
        mach_ceiling=float(args.mach_ceiling),
        mach_excess=float(args.mach_excess_weight),
        incomplete_rollout=float(args.incomplete_rollout_weight),
        failure_penalty=float(args.failure_penalty),
    )


def _anchor_options_from_args(args: argparse.Namespace) -> AnchorOptions:
    return AnchorOptions(
        x=float(args.anchor_x),
        logA=float(args.anchor_logA),
        sigma_logA=None if args.anchor_sigma is None else float(args.anchor_sigma),
        source="outer_lbfgsb_anchor",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run L-BFGS-B outer optimization for v6_active_boundary_reduced controls."
    )
    parser.add_argument("--case", default="freidberg_reference")
    parser.add_argument("--n-intervals", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--bound",
        action="append",
        nargs=3,
        metavar=("NAME", "MIN", "MAX"),
        help=(
            f"Override a control bound. Supported controls: {', '.join(CONTROL_VARIABLE_NAMES)}. "
            "Aliases np_in and n_p_in are accepted and log-transformed."
        ),
    )
    parser.add_argument(
        "--fixed",
        action="append",
        nargs=2,
        metavar=("NAME", "VALUE"),
        help="Fixed design override, e.g. B_T 12.0. The five primary controls remain optimized.",
    )
    parser.add_argument("--maxiter", type=int, default=24)
    parser.add_argument("--maxfun", type=int, default=160)
    parser.add_argument("--finite-diff-eps", type=float, default=1e-3)
    parser.add_argument(
        "--neighborhood-eps",
        type=float,
        default=1e-3,
        help="Normalized +/- perturbation used to certify local rollout feasibility before trusting gradients.",
    )
    parser.add_argument(
        "--certify-top-k",
        type=int,
        default=8,
        help="Number of top feasible evaluations to re-check with neighborhood certification.",
    )
    parser.add_argument(
        "--allow-nonrobust-lbfgsb",
        action="store_true",
        help="Run L-BFGS-B even when no seed passes neighborhood feasibility certification.",
    )
    parser.add_argument(
        "--disable-robust-seed-gate",
        action="store_true",
        help="Use all prescreen seeds for L-BFGS-B; final best still requires post-run certification.",
    )
    parser.add_argument("--prescreen-candidates", type=int, default=64)
    parser.add_argument("--prescreen-top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--te-over-tp-ceiling", type=float, default=1.2)
    parser.add_argument("--min-te-over-tp-gradient", type=float, default=0.0)
    parser.add_argument("--allow-prescreen-fallback", action="store_true")
    parser.add_argument("--accept-failed-rollout-seeds", action="store_true")
    parser.add_argument("--dx", type=float, required=True)
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument(
        "--objective",
        choices=("delta_drop", "power_next"),
        default="delta_drop",
        help="Local greedy target used inside each active-boundary rollout.",
    )
    parser.add_argument("--sigma-min", type=float, default=-0.5)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--curvature-max", type=float, default=0.05)
    parser.add_argument("--no-curvature-bound", action="store_true")
    parser.add_argument("--g-floor", type=float, default=0.0)
    parser.add_argument("--tp-floor", type=float, default=300.0)
    parser.add_argument("--scan-points", type=int, default=41)
    parser.add_argument("--refine-iterations", type=int, default=24)
    parser.add_argument("--active-tol", type=float, default=1e-6)
    parser.add_argument("--sonic-mode", choices=("auto", "off", "on"), default="auto")
    parser.add_argument("--sonic-mach-tol", type=float, default=1.0e-3)
    parser.add_argument("--sonic-det-abs-tol", type=float, default=1.0e-2)
    parser.add_argument("--sonic-compatibility-tol", type=float, default=1.0e-7)
    parser.add_argument("--sonic-residual-tol", type=float, default=1.0e-6)
    parser.add_argument("--rk4-substeps", type=int, default=1)
    parser.add_argument("--rk4-error-tol", type=float, default=1.0e-6)
    parser.add_argument(
        "--g-boundary-fallback-mode",
        default="endpoint_brentq",
        metavar="{endpoint_brentq,affine_expand_then_endpoint_brentq}",
        help="G-boundary fallback mode; legacy and affine_expand are accepted aliases.",
    )
    parser.add_argument("--anchor-x", type=float, default=0.0)
    parser.add_argument("--anchor-logA", type=float, default=0.0)
    parser.add_argument("--anchor-sigma", type=float, default=None)
    parser.add_argument("--delta-improvement-weight", type=float, default=1.0)
    parser.add_argument("--mhd-output-power-weight", type=float, default=0.0)
    parser.add_argument("--min-tp-floor", type=float, default=3000.0)
    parser.add_argument("--min-tp-shortfall-weight", type=float, default=1.0)
    parser.add_argument("--max-te-ceiling", type=float, default=10000.0)
    parser.add_argument("--max-te-excess-weight", type=float, default=1.0)
    parser.add_argument("--temperature-scale", type=float, default=1000.0)
    parser.add_argument("--area-ratio-min", type=float, default=1.0)
    parser.add_argument("--area-ratio-max", type=float, default=25.0)
    parser.add_argument("--area-ratio-penalty-weight", type=float, default=0.25)
    parser.add_argument("--magnetic-field-min", type=float, default=1.0)
    parser.add_argument("--magnetic-field-max", type=float, default=20.0)
    parser.add_argument("--magnetic-field-penalty-weight", type=float, default=1.0)
    parser.add_argument("--g-shortfall-weight", type=float, default=1.0)
    parser.add_argument("--g-scale", type=float, default=1.0e5)
    parser.add_argument("--mach-ceiling", type=float, default=0.98)
    parser.add_argument("--mach-excess-weight", type=float, default=10.0)
    parser.add_argument("--incomplete-rollout-weight", type=float, default=100.0)
    parser.add_argument("--failure-penalty", type=float, default=1.0e6)
    parser.add_argument("--rollout-delta-improvement-weight", type=float, default=1.0)
    parser.add_argument("--rollout-inlet-delta-weight", type=float, default=0.05)
    parser.add_argument("--rollout-inlet-te-floor", type=float, default=6000.0)
    parser.add_argument("--rollout-inlet-tp-floor", type=float, default=3000.0)
    parser.add_argument("--rollout-inlet-te-shortfall-weight", type=float, default=1.0)
    parser.add_argument("--rollout-inlet-tp-shortfall-weight", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_config = load_base_config(case=str(args.case), n_intervals=args.n_intervals)
    explicit_bounds = _parse_bounds(args.bound)
    unknown_bounds = sorted(set(explicit_bounds) - set(CONTROL_VARIABLE_NAMES))
    if unknown_bounds:
        raise SystemExit(f"--bound only supports primary controls {CONTROL_VARIABLE_NAMES}; got {unknown_bounds}.")
    fixed = _parse_pairs(args.fixed)
    overlap = sorted(set(fixed) & set(CONTROL_VARIABLE_NAMES))
    if overlap:
        raise SystemExit(f"do not pass primary optimized controls via --fixed: {overlap}. Use --bound instead.")
    lower, upper = variable_bounds(
        base_config,
        variable_names=CONTROL_VARIABLE_NAMES,
        explicit_bounds=explicit_bounds,
    )
    summary = run_outer_lbfgsb(
        base_config=base_config,
        lower=lower,
        upper=upper,
        out_dir=Path(args.out_dir),
        solver_config=LbfgsbOuterSolverConfig(
            variable_names=CONTROL_VARIABLE_NAMES,
            maxiter=int(args.maxiter),
            maxfun=int(args.maxfun),
            finite_diff_eps=float(args.finite_diff_eps),
            neighborhood_eps=float(args.neighborhood_eps),
            require_robust_seed=not bool(args.disable_robust_seed_gate),
            run_lbfgsb_without_robust_seed=bool(args.allow_nonrobust_lbfgsb),
            certify_top_k=int(args.certify_top_k),
        ),
        prescreen_settings=PrescreenSettings(
            candidates=int(args.prescreen_candidates),
            top_k=int(args.prescreen_top_k),
            seed=int(args.seed),
            te_over_tp_ceiling=float(args.te_over_tp_ceiling),
            min_te_over_tp_gradient=float(args.min_te_over_tp_gradient),
            g_floor=float(args.g_floor),
            require_rollout_ok=not bool(args.accept_failed_rollout_seeds),
            allow_fallback=bool(args.allow_prescreen_fallback),
        ),
        rollout_settings=_settings_from_args(args),
        rollout_weights=_rollout_weights_from_args(args),
        reward_weights=_reward_weights_from_args(args),
        anchor_options=_anchor_options_from_args(args),
        fixed_overrides=fixed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return 0 if bool(summary.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
