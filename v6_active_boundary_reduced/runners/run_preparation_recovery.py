from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from v6_firedrake_reduced.cases.freidberg_reference import load_reference_profile
from v6_firedrake_reduced.design import load_case_config

from ..core.policy import PreparationSettings, anchor_from_dict, anchor_from_profile, recover_preparation_profile
from ..diagnostics.preparation_recovery import write_preparation_diagnostics


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


def _load_profile(path: Path | None, *, case: str) -> dict[str, np.ndarray]:
    if path is None:
        if str(case).strip().lower().replace("-", "_") != "freidberg_reference":
            raise ValueError("--profile-npz is required unless case is freidberg_reference.")
        return load_reference_profile()
    with np.load(path) as data:
        return {name: np.asarray(data[name], dtype=float) for name in data.files}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover an upstream preparation profile from a target MHD state."
    )
    parser.add_argument("--case", default="freidberg_reference")
    parser.add_argument("--profile-npz", type=Path, default=None)
    anchor = parser.add_mutually_exclusive_group(required=True)
    anchor.add_argument(
        "--anchor-json",
        type=Path,
        help="JSON target state. Fields: n_p or log_n, T_e or log_Te, A or logA; optional sigma_logA and x.",
    )
    anchor.add_argument(
        "--anchor-profile-index",
        type=int,
        help="Use one node from --profile-npz or the built-in Freidberg profile as the target anchor.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dx", type=float, required=True, help="Positive upstream marching step [m].")
    parser.add_argument("--n-steps", type=int, default=60)
    # REVIEW: reverse preparation currently only supports delta_drop; power_next is kept as a visible legacy CLI option.
    parser.add_argument(
        "--objective",
        choices=("delta_drop", "power_next"),
        default="delta_drop",
        help="Local greedy target. delta_drop preserves the existing reverse-preparation policy.",
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
    parser.add_argument(
        "--write-diagnostics",
        action="store_true",
        help="Write diagnostic tables and plots. Disabled by default for batch runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_case_config(case=str(args.case))
    if args.anchor_json is not None:
        anchor_payload = json.loads(Path(args.anchor_json).read_text(encoding="utf-8"))
        anchor = anchor_from_dict(anchor_payload, config=config)
    else:
        profile = _load_profile(args.profile_npz, case=config.case)
        anchor = anchor_from_profile(
            profile,
            index=int(args.anchor_profile_index),
            config=config,
            source=str(args.profile_npz or f"{config.case}:built_in_profile"),
        )
    settings = PreparationSettings(
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
    payload = recover_preparation_profile(config=config, anchor=anchor, settings=settings)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "preparation_recovery_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "nodes.csv", list(payload["nodes"]))
    _write_csv(out_dir / "segments.csv", list(payload["segments"]))
    arrays = payload["profile_arrays"]
    np.savez(
        out_dir / "profile.npz",
        x=np.asarray(arrays["x"], dtype=float),
        n_p=np.asarray(arrays["n_p"], dtype=float),
        T_e=np.asarray(arrays["T_e"], dtype=float),
        A=np.asarray(arrays["A"], dtype=float),
        sigma_logA=np.asarray(arrays["sigma_logA"], dtype=float),
    )
    diagnostic_manifest = (
        write_preparation_diagnostics(summary_path) if bool(args.write_diagnostics) else None
    )
    short = {
        "ok": bool(payload["ok"]),
        "out_dir": str(out_dir),
        "diagnostics": diagnostic_manifest,
        **payload["active_summary"],
    }
    print(json.dumps(short, indent=2, sort_keys=True, default=_json_default))
    return 0 if bool(payload["ok"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
