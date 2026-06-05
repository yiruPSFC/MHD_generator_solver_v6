from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .objective import (
    SEARCH_DESIGN_VARIABLE_NAMES,
    AnchorOptions,
    PreparationObjectiveWeights,
    evaluate_preparation_design,
    flatten_result_for_csv,
    load_base_config,
)
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


def _parse_ranges(items: list[list[str]] | None) -> dict[str, np.ndarray]:
    ranges: dict[str, np.ndarray] = {}
    for raw in items or []:
        name = str(raw[0])
        if name not in SEARCH_DESIGN_VARIABLE_NAMES:
            raise ValueError(
                f"{name!r} is not a searchable active-boundary design variable. "
                f"Use one of {SEARCH_DESIGN_VARIABLE_NAMES}."
            )
        lo = float(raw[1])
        hi = float(raw[2])
        count = int(raw[3])
        if count <= 0:
            raise ValueError(f"range count for {name} must be positive.")
        ranges[name] = np.linspace(lo, hi, count, dtype=float)
    return ranges


def _design_overrides_from_ranges(ranges: dict[str, np.ndarray]) -> list[dict[str, float]]:
    if not ranges:
        return [{}]
    names = list(ranges)
    rows: list[dict[str, float]] = []
    for values in itertools.product(*(ranges[name] for name in names)):
        rows.append({name: float(value) for name, value in zip(names, values, strict=True)})
    return rows


def _load_design_jsonl(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_no} must be a JSON object.")
        unknown = sorted(set(payload) - set(SEARCH_DESIGN_VARIABLE_NAMES))
        if unknown:
            raise ValueError(
                f"{path}:{line_no} contains non-search variable(s) {unknown}; "
                f"use one of {SEARCH_DESIGN_VARIABLE_NAMES}."
            )
        rows.append({str(k): float(v) for k, v in payload.items()})
    return rows


def _settings_from_args(args: argparse.Namespace) -> PreparationSettings:
    return PreparationSettings(
        n_steps=int(args.n_steps),
        dx=float(args.dx),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        curvature_max=None if bool(args.no_curvature_bound) else float(args.curvature_max),
        g_floor=float(args.g_floor),
        tp_floor_K=float(args.tp_floor),
        scan_points=int(args.scan_points),
        refine_iterations=int(args.refine_iterations),
        active_tol=float(args.active_tol),
        residual_tol=float(args.residual_tol),
        sonic_mode=str(args.sonic_mode),
        sonic_mach_tol=float(args.sonic_mach_tol),
        sonic_det_abs_tol=float(args.sonic_det_abs_tol),
        sonic_compatibility_tol=float(args.sonic_compatibility_tol),
        sonic_residual_tol=float(args.sonic_residual_tol),
        step_backend=str(args.step_backend),
        rk4_substeps=int(args.rk4_substeps),
        rk4_error_tol=float(args.rk4_error_tol),
    )


def _refined_settings_from_args(args: argparse.Namespace, coarse: PreparationSettings) -> PreparationSettings | None:
    if int(args.refine_top_k) <= 0:
        return None
    if args.refine_dx is None and args.refine_n_steps is None:
        raise ValueError("--refine-top-k requires --refine-dx, --refine-n-steps, or both.")
    length = float(coarse.dx) * float(coarse.n_steps)
    if args.refine_dx is None:
        n_steps = int(args.refine_n_steps)
        if n_steps <= 0:
            raise ValueError("--refine-n-steps must be positive.")
        dx = length / float(n_steps)
    elif args.refine_n_steps is None:
        requested_dx = float(args.refine_dx)
        if requested_dx <= 0.0:
            raise ValueError("--refine-dx must be positive.")
        n_steps = max(1, int(np.ceil(length / requested_dx)))
        dx = length / float(n_steps)
    else:
        dx = float(args.refine_dx)
        n_steps = int(args.refine_n_steps)
        if dx <= 0.0 or n_steps <= 0:
            raise ValueError("--refine-dx and --refine-n-steps must be positive.")
    return PreparationSettings(
        n_steps=n_steps,
        dx=dx,
        sigma_min=float(coarse.sigma_min),
        sigma_max=float(coarse.sigma_max),
        curvature_max=coarse.curvature_max,
        g_floor=float(coarse.g_floor),
        tp_floor_K=float(coarse.tp_floor_K),
        scan_points=int(coarse.scan_points),
        refine_iterations=int(coarse.refine_iterations),
        active_tol=float(coarse.active_tol),
        residual_tol=float(coarse.residual_tol),
        sonic_mode=str(coarse.sonic_mode),
        sonic_mach_tol=float(coarse.sonic_mach_tol),
        sonic_det_abs_tol=float(coarse.sonic_det_abs_tol),
        sonic_compatibility_tol=float(coarse.sonic_compatibility_tol),
        sonic_residual_tol=float(coarse.sonic_residual_tol),
        step_backend=str(coarse.step_backend),
        rk4_substeps=int(coarse.rk4_substeps),
        rk4_error_tol=float(coarse.rk4_error_tol),
    )


def _weights_from_args(args: argparse.Namespace) -> PreparationObjectiveWeights:
    return PreparationObjectiveWeights(
        delta_improvement=float(args.delta_improvement_weight),
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
        source="design_anchor_scan",
    )


def _scan_worker(job: dict[str, Any]) -> dict[str, Any]:
    base_config = load_base_config(case=str(job["case"]), n_intervals=job["n_intervals"])
    result = evaluate_preparation_design(
        base_config=base_config,
        design_overrides=dict(job["design_overrides"]),
        settings=job["settings"],
        weights=job["weights"],
        anchor_options=job["anchor_options"],
        case_id=int(job["case_id"]),
    )
    if "coarse_case_id" in job:
        result["coarse_case_id"] = job["coarse_case_id"]
    return result


def _evaluate_jobs(
    *,
    jobs: list[dict[str, Any]],
    workers: int,
    jsonl_path: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        if int(workers) <= 1:
            for job in jobs:
                result = _scan_worker(job)
                results.append(result)
                fh.write(json.dumps(result, sort_keys=True, default=_json_default) + "\n")
                fh.flush()
        else:
            with ProcessPoolExecutor(max_workers=int(workers)) as pool:
                future_to_id = {pool.submit(_scan_worker, job): int(job["case_id"]) for job in jobs}
                for future in as_completed(future_to_id):
                    result = future.result()
                    results.append(result)
                    fh.write(json.dumps(result, sort_keys=True, default=_json_default) + "\n")
                    fh.flush()
    return results


def _flatten_rows(results: list[dict[str, Any]], *, mesh_phase: str) -> list[dict[str, Any]]:
    rows = []
    for result in sorted(results, key=lambda item: int(item["case_id"])):
        row = flatten_result_for_csv(result)
        row["mesh_phase"] = str(mesh_phase)
        if "coarse_case_id" in result:
            row["coarse_case_id"] = result["coarse_case_id"]
        rows.append(row)
    return rows


def _ranked(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(results, key=lambda item: float(item.get("score", -np.inf)), reverse=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel scan over reduced_firedrake-style target anchor design variables."
    )
    parser.add_argument("--case", default="freidberg_reference")
    parser.add_argument("--n-intervals", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--design-jsonl", type=Path, default=None)
    parser.add_argument(
        "--range",
        action="append",
        nargs=4,
        metavar=("NAME", "MIN", "MAX", "COUNT"),
        help=f"Grid range for one non-area design variable. Names: {', '.join(SEARCH_DESIGN_VARIABLE_NAMES)}.",
    )
    parser.add_argument("--dx", type=float, required=True, help="Positive upstream marching step [m].")
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument(
        "--refine-top-k",
        type=int,
        default=0,
        help="Re-evaluate the top K coarse candidates on a refined mesh.",
    )
    parser.add_argument(
        "--refine-dx",
        type=float,
        default=None,
        help="Refined upstream step [m]. If --refine-n-steps is omitted, total length is preserved.",
    )
    parser.add_argument(
        "--refine-n-steps",
        type=int,
        default=None,
        help="Refined step count. If --refine-dx is omitted, total length is preserved.",
    )
    parser.add_argument(
        "--refine-workers",
        type=int,
        default=None,
        help="Worker count for the refined pass. Defaults to --workers.",
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
    parser.add_argument("--residual-tol", type=float, default=1e-8)
    parser.add_argument("--sonic-mode", choices=("auto", "off", "on"), default="auto")
    parser.add_argument("--sonic-mach-tol", type=float, default=1.0e-3)
    parser.add_argument("--sonic-det-abs-tol", type=float, default=1.0e-2)
    parser.add_argument("--sonic-compatibility-tol", type=float, default=1.0e-7)
    parser.add_argument("--sonic-residual-tol", type=float, default=1.0e-6)
    parser.add_argument("--step-backend", choices=("implicit_be", "rk4"), default="implicit_be")
    parser.add_argument("--rk4-substeps", type=int, default=1)
    parser.add_argument("--rk4-error-tol", type=float, default=1.0e-6)
    parser.add_argument("--anchor-x", type=float, default=0.0)
    parser.add_argument("--anchor-logA", type=float, default=0.0)
    parser.add_argument(
        "--anchor-sigma",
        type=float,
        default=None,
        help="Target dlogA/dx. Default uses Freidberg profile index 0 when available, otherwise 0.",
    )
    parser.add_argument("--delta-improvement-weight", type=float, default=1.0)
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
    if args.design_jsonl is not None and args.range:
        raise SystemExit("--design-jsonl and --range are mutually exclusive for this scan runner.")
    settings = _settings_from_args(args)
    refined_settings = _refined_settings_from_args(args, settings)
    weights = _weights_from_args(args)
    anchor_options = _anchor_options_from_args(args)
    if args.design_jsonl is not None:
        overrides = _load_design_jsonl(args.design_jsonl)
    else:
        overrides = _design_overrides_from_ranges(_parse_ranges(args.range))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    jobs = [
        {
            "case_id": idx,
            "case": str(args.case),
            "n_intervals": None if args.n_intervals is None else int(args.n_intervals),
            "design_overrides": row,
            "settings": settings,
            "weights": weights,
            "anchor_options": anchor_options,
        }
        for idx, row in enumerate(overrides)
    ]
    jsonl_path = out_dir / "scan_results.jsonl"
    results = _evaluate_jobs(jobs=jobs, workers=int(args.workers), jsonl_path=jsonl_path)
    coarse_elapsed_s = time.perf_counter() - started
    _write_csv(out_dir / "scan_results.csv", _flatten_rows(results, mesh_phase="coarse"))
    ranked = _ranked(results)

    refined_results: list[dict[str, Any]] = []
    refined_elapsed_s = 0.0
    refined_jsonl_path = out_dir / "refined_results.jsonl"
    refined_csv_path = out_dir / "refined_results.csv"
    if refined_settings is not None:
        refine_started = time.perf_counter()
        top_k = max(0, min(int(args.refine_top_k), len(ranked)))
        refined_jobs = []
        for refine_case_id, coarse_result in enumerate(ranked[:top_k]):
            refined_jobs.append(
                {
                    "case_id": refine_case_id,
                    "coarse_case_id": coarse_result.get("case_id"),
                    "case": str(args.case),
                    "n_intervals": None if args.n_intervals is None else int(args.n_intervals),
                    "design_overrides": dict(coarse_result.get("search_variables", {}) or {}),
                    "settings": refined_settings,
                    "weights": weights,
                    "anchor_options": anchor_options,
                }
            )
        refined_results = _evaluate_jobs(
            jobs=refined_jobs,
            workers=int(args.workers if args.refine_workers is None else args.refine_workers),
            jsonl_path=refined_jsonl_path,
        )
        refined_elapsed_s = time.perf_counter() - refine_started
        _write_csv(refined_csv_path, _flatten_rows(refined_results, mesh_phase="refined"))

    elapsed_s = time.perf_counter() - started
    final_ranked = _ranked(refined_results) if refined_results else ranked
    summary = {
        "ok": bool(any(bool(item.get("ok", False)) for item in final_ranked)),
        "case": str(args.case),
        "n_cases": int(len(results)),
        "ok_count": int(sum(1 for item in results if bool(item.get("ok", False)))),
        "elapsed_s": float(elapsed_s),
        "coarse_elapsed_s": float(coarse_elapsed_s),
        "workers": int(args.workers),
        "search_variables": list(SEARCH_DESIGN_VARIABLE_NAMES),
        "settings": settings.__dict__,
        "weights": weights.__dict__,
        "anchor_options": anchor_options.__dict__,
        "best_source": "refined" if refined_results else "coarse",
        "best": final_ranked[0] if final_ranked else None,
        "top10": final_ranked[:10],
        "coarse": {
            "n_cases": int(len(results)),
            "ok_count": int(sum(1 for item in results if bool(item.get("ok", False)))),
            "elapsed_s": float(coarse_elapsed_s),
            "best": ranked[0] if ranked else None,
            "top10": ranked[:10],
        },
        "refined": {
            "enabled": bool(refined_settings is not None),
            "top_k": int(args.refine_top_k),
            "n_cases": int(len(refined_results)),
            "ok_count": int(sum(1 for item in refined_results if bool(item.get("ok", False)))),
            "elapsed_s": float(refined_elapsed_s),
            "settings": None if refined_settings is None else refined_settings.__dict__,
            "workers": int(args.workers if args.refine_workers is None else args.refine_workers),
            "best": _ranked(refined_results)[0] if refined_results else None,
            "top10": _ranked(refined_results)[:10],
        },
        "outputs": {
            "scan_results_jsonl": str(jsonl_path),
            "scan_results_csv": str(out_dir / "scan_results.csv"),
            "refined_results_jsonl": str(refined_jsonl_path) if refined_results else None,
            "refined_results_csv": str(refined_csv_path) if refined_results else None,
            "scan_summary_json": str(out_dir / "scan_summary.json"),
        },
    }
    _write_json(out_dir / "scan_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return 0 if bool(summary["ok"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
