from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from v6_firedrake_reduced.design import load_case_config

from ..diagnostics.preparation_recovery import write_preparation_diagnostics
from ..core.policy import PreparationSettings, anchor_from_dict, recover_preparation_profile
from .common import (
    anchor_from_node_payload,
    anchor_payload,
    json_default,
    load_profile_anchor,
    save_profile_npz,
    write_csv,
    write_json,
)


def _case_name(length: float) -> str:
    return f"L_{float(length):.6g}".replace("-", "m").replace(".", "p")


def _target_anchor_from_args(args: argparse.Namespace, *, config):
    if args.target_anchor_json is not None:
        payload = json.loads(Path(args.target_anchor_json).read_text(encoding="utf-8"))
        return anchor_from_dict(payload, config=config)
    return load_profile_anchor(
        None if args.target_profile_npz is None else Path(args.target_profile_npz),
        index=int(args.target_profile_index),
        config=config,
        source=str(args.target_profile_npz or f"{config.case}:built_in_profile"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reverse active-boundary/marginal baseline for fixed target "
            "anchors and channel lengths."
        )
    )
    parser.add_argument("--case", default="freidberg_reference")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lengths", type=float, nargs="+", required=True)
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument(
        "--objective",
        choices=("delta_drop", "power_next"),
        default="delta_drop",
        help="Local greedy target used inside each active-boundary rollout.",
    )
    parser.add_argument("--target-anchor-json", type=Path, default=None)
    parser.add_argument("--target-profile-npz", type=Path, default=None)
    parser.add_argument("--target-profile-index", type=int, default=0)
    parser.add_argument("--sigma-min", type=float, default=-0.5)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--curvature-max", type=float, default=8.0)
    parser.add_argument("--no-curvature-bound", action="store_true")
    parser.add_argument("--g-floor", type=float, default=0.0)
    parser.add_argument("--tp-floor", type=float, default=300.0)
    parser.add_argument("--scan-points", type=int, default=41)
    parser.add_argument("--refine-iterations", type=int, default=24)
    parser.add_argument("--active-tol", type=float, default=1e-6)
    parser.add_argument("--rk4-substeps", type=int, default=1)
    parser.add_argument("--rk4-error-tol", type=float, default=1.0e-6)
    parser.add_argument(
        "--g-boundary-fallback-mode",
        default="endpoint_brentq",
        metavar="{endpoint_brentq,affine_expand_then_endpoint_brentq}",
        help="G-boundary fallback mode; legacy and affine_expand are accepted aliases.",
    )
    parser.add_argument("--write-diagnostics", action="store_true")
    return parser


def _settings_for_length(args: argparse.Namespace, *, length: float) -> PreparationSettings:
    requested_dx = float(args.dx)
    if requested_dx <= 0.0:
        raise ValueError("--dx must be positive.")
    n_steps = max(1, int(np.ceil(float(length) / requested_dx)))
    actual_dx = float(length) / float(n_steps)
    return PreparationSettings(
        n_steps=n_steps,
        dx=actual_dx,
        objective=str(args.objective),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        curvature_max=None if bool(args.no_curvature_bound) else float(args.curvature_max),
        g_floor=float(args.g_floor),
        tp_floor_K=float(args.tp_floor),
        scan_points=int(args.scan_points),
        refine_iterations=int(args.refine_iterations),
        active_tol=float(args.active_tol),
        rk4_substeps=int(args.rk4_substeps),
        rk4_error_tol=float(args.rk4_error_tol),
        g_boundary_fallback_mode=str(args.g_boundary_fallback_mode),
    )


def _row_from_payload(payload: dict[str, Any], *, out_dir: Path, length: float) -> dict[str, Any]:
    nodes = list(payload["nodes"])
    source = nodes[-1]
    target = nodes[0]
    active = dict(payload["active_summary"])
    support_counts = dict(active.get("support_counts", {}))
    return {
        "length_m": float(length),
        "n_steps": int(payload["settings"]["n_steps"]),
        "dx_m": float(payload["settings"]["dx"]),
        "ok": bool(payload["ok"]),
        "target_Delta": float(target["Delta"]),
        "source_Delta": float(source["Delta"]),
        "forward_Delta_gain": float(target["Delta"]) - float(source["Delta"]),
        "target_Te_K": float(target["T_e"]),
        "source_Te_K": float(source["T_e"]),
        "target_Tp_K": float(target["T_p"]),
        "source_Tp_K": float(source["T_p"]),
        "G_min_excluding_target": active.get("G_min_excluding_anchor"),
        "Tp_min_excluding_target_K": active.get("Tp_min_excluding_anchor_K"),
        "max_abs_scaled_residual": active.get("max_abs_scaled_residual"),
        "G_supported_count": int(support_counts.get("G_supported", 0)),
        "Tp_floor_supported_count": int(support_counts.get("Tp_floor_supported", 0)),
        "sigma_lower_count": int(support_counts.get("sigma_lower", 0)),
        "sigma_upper_count": int(support_counts.get("sigma_upper", 0)),
        "summary_json": str(out_dir / "preparation_recovery_summary.json"),
        "profile_npz": str(out_dir / "profile.npz"),
        "source_anchor_json": str(out_dir / "source_anchor.json"),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_case_config(case=str(args.case))
    target_anchor = _target_anchor_from_args(args, config=config)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "target_anchor.json", anchor_payload(target_anchor, config=config))

    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for length in [float(v) for v in args.lengths]:
        if length <= 0.0:
            raise ValueError("--lengths must all be positive.")
        settings = _settings_for_length(args, length=length)
        payload = recover_preparation_profile(config=config, anchor=target_anchor, settings=settings)
        case_dir = out_root / _case_name(length)
        case_dir.mkdir(parents=True, exist_ok=True)
        summary_path = case_dir / "preparation_recovery_summary.json"
        write_json(summary_path, payload)
        write_csv(case_dir / "nodes.csv", list(payload["nodes"]))
        write_csv(case_dir / "segments.csv", list(payload["segments"]))
        save_profile_npz(case_dir / "profile.npz", dict(payload["profile_arrays"]))

        source_anchor = anchor_from_node_payload(
            dict(payload["nodes"][-1]),
            config=config,
            source=f"{summary_path}:nodes[-1]",
            source_index=int(payload["nodes"][-1].get("k", -1)),
        )
        write_json(case_dir / "source_anchor.json", anchor_payload(source_anchor, config=config))
        if bool(args.write_diagnostics):
            write_preparation_diagnostics(summary_path)
        row = _row_from_payload(payload, out_dir=case_dir, length=length)
        rows.append(row)
        manifests.append(
            {
                "length_m": float(length),
                "case_dir": str(case_dir),
                "summary_json": str(summary_path),
                "source_anchor_json": str(case_dir / "source_anchor.json"),
                "ok": bool(payload["ok"]),
            }
        )

    rows = sorted(rows, key=lambda row: float(row["length_m"]))
    write_csv(out_root / "reachability_baseline_summary.csv", rows)
    write_json(
        out_root / "reachability_baseline_summary.json",
        {
            "case": config.case,
            "target_anchor_json": str(out_root / "target_anchor.json"),
            "rows": rows,
            "manifests": manifests,
        },
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_root),
                "n_cases": len(rows),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
            default=json_default,
        )
    )
    return 0 if all(bool(row["ok"]) for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
