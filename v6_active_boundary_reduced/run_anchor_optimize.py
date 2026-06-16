from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from .objective import (
    SEARCH_DESIGN_VARIABLE_NAMES,
    AnchorOptions,
    PreparationObjectiveWeights,
    evaluate_preparation_design,
    flatten_result_for_csv,
    load_base_config,
)
from .preparation_recovery_diagnostics import write_preparation_diagnostics
from .policy import PreparationSettings


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


def _weights_from_args(args: argparse.Namespace) -> PreparationObjectiveWeights:
    return PreparationObjectiveWeights(
        delta_improvement=float(args.delta_improvement_weight),
        mhd_output_power_MW=float(args.mhd_output_power_weight),
        enthalpy_extraction_percent=float(args.enthalpy_extraction_weight),
        inlet_delta=float(args.inlet_delta_weight),
        inlet_te_floor_K=float(args.inlet_te_floor),
        inlet_tp_floor_K=float(args.inlet_tp_floor),
        inlet_te_shortfall=float(args.inlet_te_shortfall_weight),
        inlet_tp_shortfall=float(args.inlet_tp_shortfall_weight),
        temperature_scale_K=float(args.temperature_scale),
        failure_penalty=float(args.failure_penalty),
    )


def _anchor_options_from_args(args: argparse.Namespace) -> AnchorOptions:
    return AnchorOptions(
        x=float(args.anchor_x),
        logA=float(args.anchor_logA),
        sigma_logA=None if args.anchor_sigma is None else float(args.anchor_sigma),
        source="design_anchor_optimize",
    )


def _parse_pairs(items: list[list[str]] | None, *, flag_name: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw in items or []:
        name = str(raw[0])
        if name not in SEARCH_DESIGN_VARIABLE_NAMES:
            raise ValueError(
                f"{flag_name} received non-search variable {name!r}. "
                f"Use one of {SEARCH_DESIGN_VARIABLE_NAMES}."
            )
        values[name] = float(raw[1])
    return values


def _parse_bounds(items: list[list[str]] | None, base_config) -> tuple[list[str], list[tuple[float, float]], dict[str, float]]:
    explicit: dict[str, tuple[float, float]] = {}
    for raw in items or []:
        name = str(raw[0])
        if name not in SEARCH_DESIGN_VARIABLE_NAMES:
            raise ValueError(
                f"{name!r} is not a searchable active-boundary design variable. "
                f"Use one of {SEARCH_DESIGN_VARIABLE_NAMES}."
            )
        lo = float(raw[1])
        hi = float(raw[2])
        if hi < lo:
            raise ValueError(f"upper bound is smaller than lower bound for {name}.")
        explicit[name] = (lo, hi)

    active_names: list[str] = []
    bounds: list[tuple[float, float]] = []
    fixed: dict[str, float] = {}
    for name in SEARCH_DESIGN_VARIABLE_NAMES:
        if explicit:
            if name not in explicit:
                fixed[name] = float(getattr(base_config.design, name))
                continue
            lo, hi = explicit[name]
        else:
            lo = float(getattr(base_config.bounds.lower, name))
            hi = float(getattr(base_config.bounds.upper, name))
        if hi > lo:
            active_names.append(name)
            bounds.append((lo, hi))
        else:
            fixed[name] = 0.5 * (lo + hi)
    return active_names, bounds, fixed


def _write_best_payload(out_dir: Path, payload: dict[str, Any], *, diagnose: bool) -> dict[str, str]:
    summary_path = out_dir / "best_preparation_recovery_summary.json"
    _write_json(summary_path, payload)
    _write_csv(out_dir / "best_nodes.csv", list(payload.get("nodes", [])))
    _write_csv(out_dir / "best_segments.csv", list(payload.get("segments", [])))
    arrays = dict(payload.get("profile_arrays", {}) or {})
    if arrays:
        np.savez(
            out_dir / "best_profile.npz",
            x=np.asarray(arrays["x"], dtype=float),
            n_p=np.asarray(arrays["n_p"], dtype=float),
            T_e=np.asarray(arrays["T_e"], dtype=float),
            A=np.asarray(arrays["A"], dtype=float),
            sigma_logA=np.asarray(arrays["sigma_logA"], dtype=float),
        )
    outputs = {
        "best_preparation_recovery_summary_json": str(summary_path),
        "best_nodes_csv": str(out_dir / "best_nodes.csv"),
        "best_segments_csv": str(out_dir / "best_segments.csv"),
        "best_profile_npz": str(out_dir / "best_profile.npz"),
    }
    if diagnose:
        manifest = write_preparation_diagnostics(summary_path)
        outputs["best_diagnostics_manifest_json"] = str(out_dir / "diagnostics_manifest.json")
        for key, value in manifest.items():
            if isinstance(value, str):
                outputs[f"diagnostic_{key}"] = value
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize reduced_firedrake-style target anchor variables for reverse preparation recovery."
    )
    parser.add_argument("--case", default="freidberg_reference")
    parser.add_argument("--n-intervals", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--bound",
        action="append",
        nargs=3,
        metavar=("NAME", "MIN", "MAX"),
        help=f"Optimization bound for one non-area design variable. Names: {', '.join(SEARCH_DESIGN_VARIABLE_NAMES)}.",
    )
    parser.add_argument(
        "--fixed",
        action="append",
        nargs=2,
        metavar=("NAME", "VALUE"),
        help="Fix a non-area design variable away from its case baseline.",
    )
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--popsize", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--polish", action="store_true")
    parser.add_argument("--diagnose-best", action="store_true")
    parser.add_argument("--dx", type=float, required=True, help="Positive upstream marching step [m].")
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument(
        "--objective",
        choices=("delta_drop", "power_next"),
        default="delta_drop",
        help="Local greedy target used inside each active-boundary rollout.",
    )
    parser.add_argument("--sigma-min", type=float, default=-0.5)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--curvature-max", type=float, default=8.0)
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
    parser.add_argument(
        "--anchor-sigma",
        type=float,
        default=None,
        help="Target dlogA/dx. Default uses Freidberg profile index 0 when available, otherwise 0.",
    )
    parser.add_argument("--delta-improvement-weight", type=float, default=1.0)
    parser.add_argument("--mhd-output-power-weight", type=float, default=0.0)
    parser.add_argument("--enthalpy-extraction-weight", type=float, default=0.0)
    parser.add_argument("--inlet-delta-weight", type=float, default=0.05)
    parser.add_argument("--inlet-te-floor", type=float, default=6000.0)
    parser.add_argument("--inlet-tp-floor", type=float, default=3000.0)
    parser.add_argument("--inlet-te-shortfall-weight", type=float, default=1.0)
    parser.add_argument("--inlet-tp-shortfall-weight", type=float, default=1.0)
    parser.add_argument("--temperature-scale", type=float, default=1000.0)
    parser.add_argument("--failure-penalty", type=float, default=1.0e6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_base_config(case=str(args.case), n_intervals=args.n_intervals)
    settings = _settings_from_args(args)
    weights = _weights_from_args(args)
    anchor_options = _anchor_options_from_args(args)
    active_names, bounds, fixed = _parse_bounds(args.bound, base_config)
    fixed.update(_parse_pairs(args.fixed, flag_name="--fixed"))
    overlap = sorted(set(active_names) & set(fixed))
    if overlap:
        raise SystemExit(f"variable(s) cannot be both bounded and fixed: {overlap}")

    evaluations: list[dict[str, Any]] = []
    counter = 0

    def objective(values: np.ndarray) -> float:
        nonlocal counter
        overrides = dict(fixed)
        for idx, name in enumerate(active_names):
            overrides[name] = float(values[idx])
        result = evaluate_preparation_design(
            base_config=base_config,
            design_overrides=overrides,
            settings=settings,
            weights=weights,
            anchor_options=anchor_options,
            case_id=counter,
        )
        evaluations.append(result)
        counter += 1
        score = float(result.get("score", -float(weights.failure_penalty)))
        return -score if np.isfinite(score) else float(weights.failure_penalty)

    started = time.perf_counter()
    optimizer_payload: dict[str, Any]
    if active_names:
        scipy_result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=int(args.maxiter),
            popsize=int(args.popsize),
            seed=int(args.seed),
            polish=bool(args.polish),
            updating="immediate",
            workers=1,
        )
        best_overrides = dict(fixed)
        for idx, name in enumerate(active_names):
            best_overrides[name] = float(scipy_result.x[idx])
        optimizer_payload = {
            "method": "scipy.optimize.differential_evolution",
            "success": bool(scipy_result.success),
            "message": str(scipy_result.message),
            "fun": float(scipy_result.fun),
            "nit": int(scipy_result.nit),
            "nfev": int(scipy_result.nfev),
            "active_names": list(active_names),
            "bounds": {name: {"min": float(lo), "max": float(hi)} for name, (lo, hi) in zip(active_names, bounds, strict=True)},
            "best_vector": [float(value) for value in np.asarray(scipy_result.x, dtype=float)],
        }
    else:
        best_overrides = dict(fixed)
        objective(np.asarray([], dtype=float))
        optimizer_payload = {
            "method": "single_evaluation_no_active_variables",
            "success": True,
            "message": "No non-degenerate optimization bounds were provided.",
            "fun": float(-evaluations[-1]["score"]),
            "nit": 0,
            "nfev": 1,
            "active_names": [],
            "bounds": {},
            "best_vector": [],
        }

    best_result = evaluate_preparation_design(
        base_config=base_config,
        design_overrides=best_overrides,
        settings=settings,
        weights=weights,
        anchor_options=anchor_options,
        case_id=counter,
        return_payload=True,
    )
    best_payload = best_result.pop("payload", None)
    elapsed_s = time.perf_counter() - started
    flat_rows = [flatten_result_for_csv(item) for item in evaluations]
    _write_csv(out_dir / "evaluations.csv", flat_rows)
    with (out_dir / "evaluations.jsonl").open("w", encoding="utf-8") as fh:
        for item in evaluations:
            fh.write(json.dumps(item, sort_keys=True, default=_json_default) + "\n")
    best_outputs = _write_best_payload(out_dir, best_payload, diagnose=bool(args.diagnose_best)) if best_payload else {}
    summary = {
        "ok": bool(best_result.get("ok", False)),
        "case": str(args.case),
        "elapsed_s": float(elapsed_s),
        "search_variables": list(SEARCH_DESIGN_VARIABLE_NAMES),
        "fixed_overrides": {name: float(value) for name, value in fixed.items()},
        "settings": settings.__dict__,
        "weights": weights.__dict__,
        "anchor_options": anchor_options.__dict__,
        "optimizer": optimizer_payload,
        "best": best_result,
        "outputs": {
            "optimization_summary_json": str(out_dir / "optimization_summary.json"),
            "evaluations_csv": str(out_dir / "evaluations.csv"),
            "evaluations_jsonl": str(out_dir / "evaluations.jsonl"),
            **best_outputs,
        },
    }
    _write_json(out_dir / "optimization_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return 0 if bool(summary["ok"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
