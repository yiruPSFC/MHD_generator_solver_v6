from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .design import GEOMETRY_LENGTH_MODES, DesignVector, load_case_config
from .forward import solve_forward
from .transport import ELECTRON_TRANSPORT_MODELS, normalize_electron_transport


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _format_bound_violations(violations: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"{item['name']}={float(item['value']):.12g} outside [{float(item['min']):.12g}, {float(item['max']):.12g}]"
        for item in violations
    )


def _design_from_json(path: Path, *, config, allow_out_of_bounds: bool) -> DesignVector:
    design = DesignVector.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    violations = config.bounds.violations(design)
    if violations and not bool(allow_out_of_bounds):
        raise ValueError(
            "--design-json is outside the active case bounds. "
            "Use --allow-out-of-bounds-design-json only when exact out-of-window replay is intended. "
            f"Violations: {_format_bound_violations(violations)}"
        )
    return design


def run_continuation(
    *,
    design: DesignVector,
    base_config,
    electron_transport_modes: list[str],
    out_dir: Path,
) -> dict[str, Any]:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    previous_profile: dict[str, np.ndarray] | None = None
    rows: list[dict[str, Any]] = []
    final_payload: dict[str, Any] | None = None

    modes = [normalize_electron_transport(mode) for mode in electron_transport_modes]
    for index, electron_transport in enumerate(modes):
        step_dir = out_dir / f"step_{index:03d}_{electron_transport.lower().replace('-', '_')}"
        metadata = {
            **base_config.metadata,
            "electron_transport": electron_transport,
            "electron_transport_transition_index": int(index),
            "electron_transport_transition_steps": int(len(modes)),
            "residual_scaling": str(base_config.metadata.get("residual_scaling", "inlet")),
            "velikhov_mode": str(base_config.metadata.get("velikhov_mode", "diagnostic")),
        }
        config = replace(base_config, design=design, metadata=metadata)
        result = solve_forward(
            design=design,
            config=config,
            initial_profile=previous_profile,
        )
        row: dict[str, Any] = {
            "index": int(index),
            "electron_transport": electron_transport,
            "ok": bool(result.ok),
            "step_dir": str(step_dir),
            "initial_guess": result.diagnostics.get("initial_guess"),
        }
        fluid_diag = dict(result.diagnostics.get("working_fluid", {}) or {})
        if "sigma_ep" in fluid_diag:
            row["sigma_ep_m2"] = float(fluid_diag["sigma_ep"])
        if result.ok:
            assert result.profile is not None
            assert result.metrics is not None
            step_dir.mkdir(parents=True, exist_ok=True)
            np.savez(step_dir / "profile.npz", **result.profile)
            _write_json(step_dir / "run_summary.json", {
                "ok": True,
                "case_config": config.to_dict(),
                "metrics": result.metrics.to_dict(),
                "diagnostics": result.diagnostics,
                "profile_npz": str(step_dir / "profile.npz"),
            })
            _write_json(step_dir / "best_design.json", design.to_dict())
            previous_profile = result.profile
            row.update(
                {
                    "raw_enthalpy_extraction_percent": float(result.metrics.raw_enthalpy_extraction_percent),
                    "mhd_output_power_W": float(result.metrics.mhd_output_power_W),
                    "hall_voltage_V": float(result.metrics.hall_voltage_V),
                    "min_T_p_K": float(result.metrics.min_T_p_K),
                    "outlet_mach": float(result.metrics.outlet_mach),
                    "min_velikhov_margin": float(result.metrics.min_velikhov_margin),
                    "velikhov_passes_floor": bool(result.metrics.velikhov_passes_floor),
                }
            )
        else:
            row["error"] = result.error
            _write_json(step_dir / "run_summary.json", {
                "ok": False,
                "case_config": config.to_dict(),
                "diagnostics": result.diagnostics,
                "error": result.error,
            })
            rows.append(row)
            final_payload = {
                "ok": False,
                "failed_index": int(index),
                "failed_electron_transport": electron_transport,
                "error": result.error,
                "rows": rows,
            }
            break
        rows.append(row)

    if final_payload is None:
        final_payload = {
            "ok": True,
            "final_electron_transport": modes[-1],
            "rows": rows,
        }
    _write_json(out_dir / "continuation_summary.json", final_payload)
    return final_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a two-model electron-transport warm-start replay.")
    parser.add_argument("--case", default="yamasaki2004", choices=("yamasaki2004",))
    parser.add_argument("--objective", default="enthalpy_extraction")
    parser.add_argument("--n-intervals", type=int, default=20)
    parser.add_argument("--geometry-length-mode", default="inferred_swirl", choices=GEOMETRY_LENGTH_MODES)
    parser.add_argument("--design-json", type=Path, required=True)
    parser.add_argument("--allow-out-of-bounds-design-json", action="store_true")
    parser.add_argument(
        "--start-electron-transport",
        default="e-Argon",
        help=f"Initial electron transport model, one of {ELECTRON_TRANSPORT_MODELS}.",
    )
    parser.add_argument(
        "--target-electron-transport",
        default="e-He",
        help=f"Target electron transport model, one of {ELECTRON_TRANSPORT_MODELS}.",
    )
    parser.add_argument("--residual-scaling", default="inlet", choices=("inlet", "characteristic", "dimensional"))
    parser.add_argument("--velikhov-mode", default="diagnostic", choices=("diagnostic", "penalty"))
    parser.add_argument("--velikhov-floor", type=float, default=5e-7)
    parser.add_argument("--velikhov-penalty-scale", type=float, default=1e-2)
    parser.add_argument("--velikhov-penalty-weight", type=float, default=25.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    config = load_case_config(
        case=str(args.case),
        objective_profile=str(args.objective),
        n_intervals=int(args.n_intervals),
        geometry_length_mode=str(args.geometry_length_mode),
    )
    metadata = {
        **config.metadata,
        "design_json": str(args.design_json),
        "design_json_bounds_policy": "allow" if bool(args.allow_out_of_bounds_design_json) else "error",
        "residual_scaling": str(args.residual_scaling),
        "velikhov_mode": str(args.velikhov_mode),
        "velikhov_floor": float(args.velikhov_floor),
        "velikhov_penalty_scale": float(args.velikhov_penalty_scale),
        "velikhov_penalty_weight": float(args.velikhov_penalty_weight),
    }
    design = _design_from_json(
        Path(args.design_json),
        config=config,
        allow_out_of_bounds=bool(args.allow_out_of_bounds_design_json),
    )
    config = replace(config, design=design, metadata=metadata)
    start_transport = normalize_electron_transport(args.start_electron_transport)
    target_transport = normalize_electron_transport(args.target_electron_transport)
    modes = [start_transport] if start_transport == target_transport else [start_transport, target_transport]
    payload = run_continuation(
        design=design,
        base_config=config,
        electron_transport_modes=modes,
        out_dir=Path(args.out_dir),
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
