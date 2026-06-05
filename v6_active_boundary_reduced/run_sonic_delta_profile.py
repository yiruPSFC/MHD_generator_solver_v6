from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from v6_firedrake_reduced.design import load_case_config

from .sonic_delta_profile import SonicDeltaSettings, build_sonic_delta_profile


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, payload: dict[str, Any]) -> None:
    arrays = dict(payload["profile_arrays"])
    x = np.asarray(arrays["x"], dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=(10.0, 9.0), sharex=True)
    axes[0].plot(x, np.asarray(arrays["mach"], dtype=float), marker="o", color="#6f3fb5")
    axes[0].axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Mach")
    axes[1].plot(x, np.asarray(arrays["Delta"], dtype=float), marker="o", color="#111111")
    axes[1].set_ylabel("Delta")
    axes[2].plot(x, np.asarray(arrays["G"], dtype=float), marker="o", color="#267a3e")
    axes[2].axhline(float(payload["settings"]["g_floor"]), color="#b22222", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("G")
    axes[3].plot(x, np.asarray(arrays["sigma_logA"], dtype=float), marker="o", color="#b2552d")
    axes[3].set_ylabel("sigma")
    axes[3].set_xlabel("local x around sonic point [m]")
    for ax in axes:
        ax.axvline(0.0, color="#555555", linewidth=1.0, alpha=0.6)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Sonic-compatible active-boundary profile")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a sonic-compatible active-boundary profile and select steepest Delta change under G>=0."
    )
    parser.add_argument("--case", default="freidberg_reference")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dx", type=float, default=1.0e-4)
    parser.add_argument("--n-steps-each-side", type=int, default=2)
    parser.add_argument("--sigma-min", type=float, default=-0.5)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--curvature-max", type=float, default=None)
    parser.add_argument("--g-floor", type=float, default=0.0)
    parser.add_argument("--tp-floor", type=float, default=300.0)
    parser.add_argument("--scan-points", type=int, default=41)
    parser.add_argument("--residual-tol", type=float, default=1.0e-6)
    parser.add_argument("--active-tol", type=float, default=1.0e-7)
    parser.add_argument("--branch-mach-tol", type=float, default=1.0e-7)
    parser.add_argument("--branch-mode", choices=("fixed", "agnostic"), default="fixed")
    parser.add_argument("--objective", choices=("pedal", "abs", "drop", "rise"), default="pedal")
    parser.add_argument("--selection-mode", choices=("continuation", "steepest"), default="continuation")
    parser.add_argument("--target-mach-slope", type=float, default=20.0)
    parser.add_argument("--target-mach-offset-max", type=float, default=0.015)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_case_config(case=str(args.case))
    settings = SonicDeltaSettings(
        dx=float(args.dx),
        n_steps_each_side=int(args.n_steps_each_side),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        curvature_max=None if args.curvature_max is None else float(args.curvature_max),
        g_floor=float(args.g_floor),
        tp_floor_K=float(args.tp_floor),
        scan_points=int(args.scan_points),
        residual_tol=float(args.residual_tol),
        active_tol=float(args.active_tol),
        branch_mach_tol=float(args.branch_mach_tol),
        branch_mode=str(args.branch_mode),
        objective=str(args.objective),
        selection_mode=str(args.selection_mode),
        target_mach_slope_1_per_m=float(args.target_mach_slope),
        target_mach_offset_max=float(args.target_mach_offset_max),
    )
    payload = build_sonic_delta_profile(config=config, settings=settings)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "sonic_delta_profile_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "nodes.csv", list(payload["nodes"]))
    _write_csv(out_dir / "segments.csv", list(payload["segments"]))
    arrays = dict(payload["profile_arrays"])
    np.savez(out_dir / "profile.npz", **{key: np.asarray(value, dtype=float) for key, value in arrays.items()})
    if not bool(args.no_plot):
        _write_plot(out_dir / "sonic_delta_profile.png", payload)
    short = {
        "ok": bool(payload["ok"]),
        "out_dir": str(out_dir),
        **dict(payload["active_summary"]),
        "sigma_sonic": payload["sonic_primitive_compatibility"]["sigma_sonic"],
        "compatibility_scaled_residual": payload["sonic_primitive_compatibility"]["compatibility_scaled_residual"],
    }
    print(json.dumps(short, indent=2, sort_keys=True, default=_json_default))
    return 0 if bool(payload["ok"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
