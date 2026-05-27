from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .design import CASE_NAMES, GEOMETRY_LENGTH_MODES, CaseConfig, DesignVector, load_case_config
from .sonic_compatibility import build_sonic_mesh_matching_diagnostic
from .transport import ELECTRON_TRANSPORT_MODELS, normalize_electron_transport


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        out = float(value)
        if not np.isfinite(out):
            return None
        return out
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _with_design(config: CaseConfig, design: DesignVector, *, metadata: dict[str, Any]) -> CaseConfig:
    return CaseConfig(
        case=config.case,
        objective_profile=config.objective_profile,
        length_m=config.length_m,
        area_scale_m2=config.area_scale_m2,
        B_T=float(design.B_T),
        working_fluid_profile=config.working_fluid_profile,
        n_intervals=config.n_intervals,
        design=design,
        bounds=config.bounds,
        metadata=dict(metadata),
    )


def _format_bound_violations(violations: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"{item['name']}={float(item['value']):.12g} outside [{float(item['min']):.12g}, {float(item['max']):.12g}]"
        for item in violations
    )


def _design_from_json(path: Path, *, config: CaseConfig, allow_out_of_bounds: bool) -> DesignVector:
    design = DesignVector.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    violations = config.bounds.violations(design)
    if violations and not bool(allow_out_of_bounds):
        raise ValueError(
            "--design-json is outside the active case bounds. "
            "Use --allow-out-of-bounds-design-json for exact failed-point replay. "
            f"Violations: {_format_bound_violations(violations)}"
        )
    return design


def _write_npz(path: Path, report: dict[str, Any]) -> None:
    if not bool(report.get("ok", False)):
        return
    mesh = dict(report["suggested_local_mesh"])
    area = dict(report["area_launch_profile"])
    np.savez(
        path,
        mesh_x_m=np.asarray(mesh["x_m"], dtype=float),
        mesh_x_fraction=np.asarray(mesh["x_fraction"], dtype=float),
        logA_launch=np.asarray(area["logA_launch"], dtype=float),
        A_over_A0_launch=np.asarray(area["A_over_A0_launch"], dtype=float),
        sigma_logA_launch_1_per_m=np.asarray(area["sigma_logA_launch_1_per_m"], dtype=float),
        logA_original=np.asarray(area["logA_original"], dtype=float),
        A_over_A0_original=np.asarray(area["A_over_A0_original"], dtype=float),
        sigma_logA_original_1_per_m=np.asarray(area["sigma_logA_original_1_per_m"], dtype=float),
    )


def _write_plot(path: Path, report: dict[str, Any]) -> None:
    if not bool(report.get("ok", False)):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    match = dict(report["local_sonic_match"])
    area = dict(report["area_launch_profile"])
    launch = dict(report["right_branch_launch_condition"])
    sonic = dict(report["sonic_point"])
    x_m = np.asarray(area["x_m"], dtype=float)
    local_um = 1.0e6 * (x_m - float(sonic["x_m"]))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    candidates = match.get("candidate_solutions", [])
    left_x = []
    left_m = []
    for item in candidates:
        anchor = item.get("left_anchor") or {}
        if anchor.get("x_fraction") is not None and anchor.get("mach") is not None:
            left_x.append(float(anchor["x_fraction"]))
            left_m.append(float(anchor["mach"]))
    axes[0, 0].plot(left_x, left_m, "o-", label="accepted left anchors")
    axes[0, 0].plot([float(sonic["x_fraction"])], [1.0], "s", label="matched sonic point")
    first = dict(launch["first_right_node"])
    axes[0, 0].plot([float(first["x_fraction"])], [float(first["mach"])], "^", label="first right node")
    axes[0, 0].axhline(1.0, color="0.25", linewidth=1.0)
    axes[0, 0].set_xlabel("x / L")
    axes[0, 0].set_ylabel("Mach")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(local_um, np.asarray(area["sigma_logA_original_1_per_m"], dtype=float), label="original A'/A")
    axes[0, 1].plot(local_um, np.asarray(area["sigma_logA_launch_1_per_m"], dtype=float), label="local launch A'/A")
    axes[0, 1].axvline(0.0, color="0.25", linewidth=1.0)
    axes[0, 1].set_xlabel("x - x* [um]")
    axes[0, 1].set_ylabel("sigma_logA [1/m]")
    axes[0, 1].legend(fontsize=8)

    A_star = float(sonic["A_over_A0"])
    axes[1, 0].plot(local_um, 1.0e6 * (np.asarray(area["A_over_A0_original"], dtype=float) - A_star), label="original")
    axes[1, 0].plot(local_um, 1.0e6 * (np.asarray(area["A_over_A0_launch"], dtype=float) - A_star), label="local launch")
    axes[1, 0].axvline(0.0, color="0.25", linewidth=1.0)
    axes[1, 0].set_xlabel("x - x* [um]")
    axes[1, 0].set_ylabel("delta(A/A0) [ppm]")
    axes[1, 0].legend(fontsize=8)

    labels = ["sigma*", "sigma original", "sigma'", "M first - 1"]
    values = [
        float(sonic["sigma_required_1_per_m"]),
        float(sonic["sigma_original_1_per_m"]),
        float(launch["required_dsigma_logA_dx_1_per_m2"]),
        float(first["mach"]) - 1.0,
    ]
    axes[1, 1].bar(labels, values)
    axes[1, 1].set_yscale("symlog", linthresh=1.0e-3)
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].set_ylabel("value")
    axes[1, 1].set_title("local compatibility numbers")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose a smooth sonic mesh/matching point for Freidberg H/L marching.")
    parser.add_argument("--case", default="yamasaki2004", choices=CASE_NAMES)
    parser.add_argument("--objective", default="enthalpy_extraction")
    parser.add_argument("--n-intervals", type=int, default=None)
    parser.add_argument("--geometry-length-mode", default="radial", choices=GEOMETRY_LENGTH_MODES)
    parser.add_argument("--design-json", type=Path, default=None)
    parser.add_argument("--allow-out-of-bounds-design-json", action="store_true")
    parser.add_argument(
        "--electron-transport",
        default=None,
        help=f"Electron-heavy collision model, one of {ELECTRON_TRANSPORT_MODELS}; defaults to case metadata.",
    )
    parser.add_argument("--reference-residual-tol", type=float, default=1e-7)
    parser.add_argument("--reference-substeps-per-interval", type=int, default=10)
    parser.add_argument("--reference-max-log-step", type=float, default=0.25)
    parser.add_argument("--target-M-prime", type=float, default=1000.0)
    parser.add_argument("--launch-mach-increment", type=float, default=1.0e-3)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_case_config(
        case=args.case,
        objective_profile=args.objective,
        n_intervals=args.n_intervals,
        geometry_length_mode=str(args.geometry_length_mode),
    )
    electron_transport = normalize_electron_transport(
        args.electron_transport
        if args.electron_transport is not None
        else config.metadata.get("electron_transport", "e-He")
    )
    metadata = {
        **config.metadata,
        "electron_transport": electron_transport,
        "sonic_mesh_target_M_prime_1_per_m": float(args.target_M_prime),
        "sonic_mesh_launch_mach_increment": float(args.launch_mach_increment),
    }
    design = config.design
    if args.design_json is not None:
        design = _design_from_json(
            Path(args.design_json),
            config=config,
            allow_out_of_bounds=bool(args.allow_out_of_bounds_design_json),
        )
        metadata = {
            **metadata,
            "design_json": str(Path(args.design_json)),
            "design_json_bounds_policy": "allow" if bool(args.allow_out_of_bounds_design_json) else "error",
        }
    config = _with_design(config, design, metadata=metadata)
    report = build_sonic_mesh_matching_diagnostic(
        design=config.design,
        config=config,
        residual_tol=float(args.reference_residual_tol),
        initial_substeps_per_interval=int(args.reference_substeps_per_interval),
        max_log_step=float(args.reference_max_log_step),
        target_M_prime_1_per_m=float(args.target_M_prime),
        launch_mach_increment=float(args.launch_mach_increment),
    )
    payload = {
        **report,
        "case_config": config.to_dict(),
    }
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "summary.json", payload)
    _write_npz(out_dir / "local_mesh_area_launch.npz", report)
    if not bool(args.no_plot):
        _write_plot(out_dir / "sonic_mesh_matching.png", report)
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return 0 if bool(report.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
