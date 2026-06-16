from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cases.freidberg_reference import load_reference_profile
from .design import DESIGN_VARIABLE_NAMES, DesignVector
from .freidberg_area_only import (
    AREA_CONTROL_NAMES,
    compare_candidate_to_reference,
    freidberg_net_power_MWe,
    load_freidberg_area_only_config,
    reference_profile_metrics,
)
from .forward import solve_forward
from .run_firedrake_reduced import _json_default, _run_optimize, _write_json
from .transport import normalize_electron_transport


THERMAL_HISTORY_FIELDS = (
    "min_T_p_K",
    "max_Te_over_Tp",
    "thermal_window_penalty",
    "thermal_window_penalty_candidate",
    "thermal_Tp_in_penalty_candidate",
    "thermal_Tp_low_penalty_candidate",
    "thermal_Tp_high_penalty_candidate",
    "thermal_stagnation_Tp_high_penalty_candidate",
    "thermal_Te_over_Tp_penalty_candidate",
    "min_stagnation_T_p_K",
    "max_stagnation_T_p_K",
)


def _add_thermal_window_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--thermal-window-mode",
        default="diagnostic",
        choices=("diagnostic", "penalty"),
        help="Use reconstructed T_p / T_e/T_p only as diagnostics, or subtract a soft thermal-window penalty.",
    )
    parser.add_argument(
        "--thermal-tp-in-max",
        type=float,
        default=2000.0,
        help="Soft inlet T_p ceiling [K] used by --thermal-window-mode penalty.",
    )
    parser.add_argument(
        "--thermal-tp-floor",
        type=float,
        default=300.0,
        help="Soft path floor for reconstructed T_p nodes [K].",
    )
    parser.add_argument(
        "--thermal-tp-path-max",
        type=float,
        default=None,
        help="Optional soft path ceiling for reconstructed T_p nodes [K].",
    )
    parser.add_argument(
        "--thermal-stagnation-tp-path-max",
        type=float,
        default=None,
        help="Optional soft path ceiling for isentropic heavy-particle stagnation temperature T_p0 [K].",
    )
    parser.add_argument(
        "--thermal-te-over-tp-min",
        type=float,
        default=None,
        help="Optional soft lower band edge for reconstructed T_e/T_p.",
    )
    parser.add_argument(
        "--thermal-te-over-tp-max",
        type=float,
        default=None,
        help="Optional soft upper band edge for reconstructed T_e/T_p.",
    )
    parser.add_argument("--thermal-tp-penalty-scale", type=float, default=100.0)
    parser.add_argument("--thermal-te-over-tp-penalty-scale", type=float, default=1.0)
    parser.add_argument("--thermal-tp-in-penalty-weight", type=float, default=1.0)
    parser.add_argument("--thermal-tp-path-penalty-weight", type=float, default=1.0)
    parser.add_argument("--thermal-stagnation-tp-penalty-weight", type=float, default=1.0)
    parser.add_argument("--thermal-te-over-tp-penalty-weight", type=float, default=1.0)


def _thermal_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "thermal_window_mode": str(args.thermal_window_mode),
        "thermal_tp_in_max_K": None if args.thermal_tp_in_max is None else float(args.thermal_tp_in_max),
        "thermal_tp_floor_K": None if args.thermal_tp_floor is None else float(args.thermal_tp_floor),
        "thermal_tp_path_max_K": None if args.thermal_tp_path_max is None else float(args.thermal_tp_path_max),
        "thermal_stagnation_tp_path_max_K": (
            None
            if args.thermal_stagnation_tp_path_max is None
            else float(args.thermal_stagnation_tp_path_max)
        ),
        "thermal_te_over_tp_min": (
            None if args.thermal_te_over_tp_min is None else float(args.thermal_te_over_tp_min)
        ),
        "thermal_te_over_tp_max": (
            None if args.thermal_te_over_tp_max is None else float(args.thermal_te_over_tp_max)
        ),
        "thermal_tp_penalty_scale_K": float(args.thermal_tp_penalty_scale),
        "thermal_te_over_tp_penalty_scale": float(args.thermal_te_over_tp_penalty_scale),
        "thermal_tp_in_penalty_weight": float(args.thermal_tp_in_penalty_weight),
        "thermal_tp_path_penalty_weight": float(args.thermal_tp_path_penalty_weight),
        "thermal_stagnation_tp_penalty_weight": float(args.thermal_stagnation_tp_penalty_weight),
        "thermal_te_over_tp_penalty_weight": float(args.thermal_te_over_tp_penalty_weight),
    }


def _metric_history_fields(metrics) -> dict[str, float | bool]:
    payload = metrics.to_dict()
    names = (
        "objective_score",
        "mhd_output_power_W",
        "raw_enthalpy_extraction_percent",
        "min_velikhov_margin",
        *THERMAL_HISTORY_FIELDS,
    )
    return {name: payload[name] for name in names}


def _area_control_box_summary(config) -> dict[str, Any]:
    values = config.design.as_array()
    lower = config.bounds.lower.as_array()
    upper = config.bounds.upper.as_array()
    controls = []
    for name in AREA_CONTROL_NAMES:
        idx = DESIGN_VARIABLE_NAMES.index(name)
        span = max(float(upper[idx] - lower[idx]), 1e-300)
        fraction = float((values[idx] - lower[idx]) / span)
        if fraction <= 1e-6:
            support = "lower_bound"
        elif fraction >= 1.0 - 1e-6:
            support = "upper_bound"
        else:
            support = "interior"
        controls.append(
            {
                "name": name,
                "value": float(values[idx]),
                "lower": float(lower[idx]),
                "upper": float(upper[idx]),
                "box_fraction": fraction,
                "support": support,
            }
        )
    return {
        "control_names": list(AREA_CONTROL_NAMES),
        "controls": controls,
        "summary": {item["name"]: item["support"] for item in controls},
    }


def _attach_area_control_summary(payload: dict[str, Any], config) -> dict[str, Any]:
    enriched = dict(payload)
    final_config = config
    if isinstance(enriched.get("case_config"), dict) and isinstance(enriched["case_config"].get("design"), dict):
        final_config = _config_with_design(config, DesignVector.from_dict(enriched["case_config"]["design"]))
    enriched["area_control_box_summary"] = _area_control_box_summary(final_config)
    return enriched


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an area-only Freidberg reference benchmark with fixed inlet/load controls."
    )
    parser.add_argument("--mode", default="optimize", choices=("reference", "optimize"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-intervals", type=int, default=None)
    parser.add_argument("--area-window-half-width", type=float, default=0.35)
    parser.add_argument("--reference-profile-npz", type=Path, default=None)
    parser.add_argument(
        "--optimizer",
        default="coordinate_search",
        choices=("coordinate_search", "projected_gradient", "constrained_slsqp"),
    )
    parser.add_argument("--coordinate-initial-step", type=float, default=0.05)
    parser.add_argument("--coordinate-min-step", type=float, default=1e-3)
    parser.add_argument("--velikhov-constraint-mode", default="none", choices=("none", "hard"))
    parser.add_argument("--velikhov-hard-floor", type=float, default=0.0)
    parser.add_argument("--multistart", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument(
        "--slsqp-state-restarts",
        type=int,
        default=1,
        help="For constrained_slsqp, rebuild from the last converged profile this many times.",
    )
    parser.add_argument(
        "--slsqp-trial-continuation",
        action="store_true",
        help="For constrained_slsqp, warm-start each trial by design-space continuation before local adjoint evaluation.",
    )
    parser.add_argument(
        "--slsqp-continuation-T-p-floor-K",
        type=float,
        default=1.0,
        help="Hard T_p floor used by --slsqp-trial-continuation while accepting continuation trial states.",
    )
    parser.add_argument(
        "--slsqp-rebuild-tape-per-trial",
        action="store_true",
        help="With --slsqp-trial-continuation, rebuild a local pyadjoint tape for every trial instead of reusing a Placeholder-backed tape.",
    )
    parser.add_argument("--residual-scaling", default="inlet", choices=("inlet", "characteristic", "dimensional"))
    parser.add_argument("--equation-form", default="primitive", choices=("primitive", "freidberg_hl"))
    parser.add_argument("--snes-max-it", type=int, default=None)
    parser.add_argument("--snes-type", default=None)
    parser.add_argument("--snes-dtol", type=float, default=None)
    parser.add_argument("--snes-linesearch-type", default=None)
    parser.add_argument("--electron-transport", default=None)
    parser.add_argument("--freidberg-branch-audit", action="store_true")
    _add_thermal_window_args(parser)
    return parser


def _with_runtime_metadata(config, args: argparse.Namespace):
    electron_transport = normalize_electron_transport(
        args.electron_transport
        if args.electron_transport is not None
        else config.metadata.get("electron_transport", "e-Argon")
    )
    metadata: dict[str, Any] = {
        **config.metadata,
        "benchmark": "freidberg_reference_area_only",
        "electron_transport": electron_transport,
        "residual_scaling": str(args.residual_scaling),
        "equation_form": str(args.equation_form),
        "optimizer": str(args.optimizer),
        "velikhov_constraint_mode": str(args.velikhov_constraint_mode),
        "velikhov_hard_floor": float(args.velikhov_hard_floor),
        "velikhov_constraint_sampling": "nodes",
        "slsqp_trial_continuation": bool(args.slsqp_trial_continuation),
        "slsqp_continuation_T_p_floor_K": float(args.slsqp_continuation_T_p_floor_K),
        "slsqp_rebuild_tape_per_trial": bool(args.slsqp_rebuild_tape_per_trial),
        **_thermal_metadata(args),
    }
    if args.snes_max_it is not None:
        metadata["snes_max_it"] = int(args.snes_max_it)
    if args.snes_type is not None:
        metadata["snes_type"] = str(args.snes_type)
    if args.snes_dtol is not None:
        metadata["snes_dtol"] = float(args.snes_dtol)
    if args.snes_linesearch_type is not None:
        metadata["snes_linesearch_type"] = str(args.snes_linesearch_type)
    return type(config)(
        case=config.case,
        objective_profile=config.objective_profile,
        length_m=config.length_m,
        area_scale_m2=config.area_scale_m2,
        B_T=config.B_T,
        working_fluid_profile=config.working_fluid_profile,
        n_intervals=config.n_intervals,
        design=config.design,
        bounds=config.bounds,
        metadata=metadata,
    )


def _config_with_design(config, design: DesignVector):
    return type(config)(
        case=config.case,
        objective_profile=config.objective_profile,
        length_m=config.length_m,
        area_scale_m2=config.area_scale_m2,
        B_T=float(design.B_T),
        working_fluid_profile=config.working_fluid_profile,
        n_intervals=config.n_intervals,
        design=design,
        bounds=config.bounds,
        metadata=config.metadata,
    )


def _run_coordinate_search(config, out_dir: Path, *, initial_profile: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    area_indices = [DESIGN_VARIABLE_NAMES.index(name) for name in AREA_CONTROL_NAMES]
    lower = config.bounds.lower.as_array()
    upper = config.bounds.upper.as_array()
    history: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    hard_velikhov = str(args.velikhov_constraint_mode) == "hard"
    hard_floor = float(args.velikhov_hard_floor)

    def evaluate(
        values: np.ndarray,
        *,
        source: str,
        step: float,
        initial_guess_profile: dict[str, Any],
    ):
        design = DesignVector.from_array(values)
        trial_config = _config_with_design(config, design)
        result = solve_forward(design=design, config=trial_config, initial_profile=initial_guess_profile)
        if not result.ok or result.profile is None or result.metrics is None:
            failures.append(
                {
                    "source": source,
                    "step": float(step),
                    "design": design.to_dict(),
                    "error": result.error or "forward solve failed",
                    "diagnostics": result.diagnostics,
                }
            )
            return None
        if hard_velikhov and float(result.metrics.min_velikhov_margin) < hard_floor:
            failures.append(
                {
                    "source": source,
                    "step": float(step),
                    "design": design.to_dict(),
                    "error": "velikhov hard floor violated",
                    "min_velikhov_margin": float(result.metrics.min_velikhov_margin),
                    "velikhov_hard_floor": hard_floor,
                    "diagnostics": result.diagnostics,
                }
            )
            return None
        return result

    current_values = config.design.as_array()
    current_result = evaluate(
        current_values,
        source="initial",
        step=0.0,
        initial_guess_profile=initial_profile,
    )
    if current_result is None:
        payload = {
            "ok": False,
            "case_config": config.to_dict(),
            "error": "initial coordinate-search forward solve failed",
            "optimizer": {
                "optimizer": "coordinate_search",
                "trial_failures": failures,
            },
        }
        _write_json(out_dir / "run_summary.json", payload)
        return payload

    current_score = float(current_result.metrics.objective_score)
    current_profile = current_result.profile
    best_result = current_result
    best_values = current_values.copy()
    step = float(args.coordinate_initial_step)
    min_step = float(args.coordinate_min_step)
    for iteration in range(int(args.max_iterations)):
        improved = False
        for idx in area_indices:
            for direction in (1.0, -1.0):
                trial_values = current_values.copy()
                trial_values[idx] = np.clip(trial_values[idx] + direction * step, lower[idx], upper[idx])
                if np.allclose(trial_values, current_values, rtol=0.0, atol=1e-15):
                    continue
                result = evaluate(
                    trial_values,
                    source=f"iter_{iteration}_{DESIGN_VARIABLE_NAMES[idx]}_{direction:+.0f}",
                    step=step,
                    initial_guess_profile=current_profile,
                )
                if result is None:
                    continue
                score = float(result.metrics.objective_score)
                history.append(
                    {
                        "iteration": int(iteration),
                        "control": DESIGN_VARIABLE_NAMES[idx],
                        "direction": float(direction),
                        "step": float(step),
                        "accepted": bool(score > current_score + 1e-10),
                        **_metric_history_fields(result.metrics),
                        "design": DesignVector.from_array(trial_values).to_dict(),
                    }
                )
                if score > current_score + 1e-10:
                    current_values = trial_values
                    current_score = score
                    current_profile = result.profile
                    best_result = result
                    best_values = trial_values.copy()
                    improved = True
                    break
            if improved:
                break
        if not improved:
            step *= 0.5
            if step < min_step:
                break

    best_design = DesignVector.from_array(best_values)
    np.savez(out_dir / "profile.npz", **current_profile)
    _write_json(out_dir / "best_design.json", best_design.to_dict())
    history_path = out_dir / "objective_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iteration",
                "control",
                "direction",
                "step",
                "accepted",
                "objective_score",
                "mhd_output_power_W",
                "raw_enthalpy_extraction_percent",
                "min_T_p_K",
                "max_Te_over_Tp",
                "min_velikhov_margin",
                "thermal_window_penalty",
                "thermal_window_penalty_candidate",
                "thermal_Tp_in_penalty_candidate",
                "thermal_Tp_low_penalty_candidate",
                "thermal_Tp_high_penalty_candidate",
                "thermal_stagnation_Tp_high_penalty_candidate",
                "thermal_Te_over_Tp_penalty_candidate",
                "min_stagnation_T_p_K",
                "max_stagnation_T_p_K",
                "design",
            ],
        )
        writer.writeheader()
        for row in history:
            writer.writerow({**row, "design": json.dumps(row["design"], sort_keys=True)})
    metrics = best_result.metrics.to_dict()
    metrics["estimated_total_plant_power_MWe"] = freidberg_net_power_MWe(
        metrics["mhd_output_power_W"],
        config=config,
    )
    payload = {
        "ok": True,
        "case_config": _config_with_design(config, best_design).to_dict(),
        "metrics": metrics,
        "diagnostics": best_result.diagnostics,
        "profile_npz": str(out_dir / "profile.npz"),
        "optimizer": {
            "optimizer": "coordinate_search",
            "max_iterations": int(args.max_iterations),
            "initial_step": float(args.coordinate_initial_step),
            "min_step": min_step,
            "velikhov_constraint_mode": str(args.velikhov_constraint_mode),
            "velikhov_hard_floor": hard_floor,
            "trial_initial_guess": "current_accepted_profile",
            "history_csv": str(history_path),
            "accepted_step_count": int(sum(1 for row in history if row["accepted"])),
            "trial_failure_count": len(failures),
            "trial_failures": failures,
        },
    }
    payload = _attach_area_control_summary(payload, _config_with_design(config, best_design))
    _write_json(out_dir / "run_summary.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if bool(args.slsqp_trial_continuation) and int(args.slsqp_state_restarts) > 1:
        raise ValueError("--slsqp-trial-continuation currently replaces --slsqp-state-restarts; use one or the other.")
    config = load_freidberg_area_only_config(
        n_intervals=args.n_intervals,
        area_half_width=float(args.area_window_half_width),
    )
    config = _with_runtime_metadata(config, args)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_profile_path = None if args.reference_profile_npz is None else str(args.reference_profile_npz)
    reference_profile = load_reference_profile(reference_profile_path)
    reference_metrics = reference_profile_metrics(config, profile_path=reference_profile_path)
    _write_json(out_dir / "reference_profile_metrics.json", reference_metrics)
    np.savez(out_dir / "reference_initial_profile.npz", **reference_profile)
    if str(args.mode) == "reference":
        benchmark_payload = {
            "ok": True,
            "mode": "reference",
            "case_config": config.to_dict(),
            "reference": reference_metrics,
            "reference_profile_metrics_json": str(out_dir / "reference_profile_metrics.json"),
            "reference_initial_profile_npz": str(out_dir / "reference_initial_profile.npz"),
        }
        _write_json(out_dir / "benchmark_summary.json", benchmark_payload)
        print(json.dumps(benchmark_payload, indent=2, sort_keys=True, default=_json_default))
        return 0

    if str(args.optimizer) == "coordinate_search":
        optimize_payload = _run_coordinate_search(config, out_dir, initial_profile=reference_profile, args=args)
    else:
        optimize_payload = _run_optimize(
            config,
            out_dir,
            optimizer=str(args.optimizer),
            velikhov_constraint_mode=str(args.velikhov_constraint_mode),
            multistart=int(args.multistart),
            seed=int(args.seed),
            max_iterations=int(args.max_iterations),
            initial_profile=reference_profile,
            initial_profile_context={
                "source": "freidberg_reference_profile",
                "path": str(args.reference_profile_npz or config.metadata.get("reference_profile_npz")),
            },
            slsqp_state_restarts=int(args.slsqp_state_restarts),
            slsqp_trial_continuation=bool(args.slsqp_trial_continuation),
            slsqp_continuation_T_p_floor_K=float(args.slsqp_continuation_T_p_floor_K),
            slsqp_rebuild_tape_per_trial=bool(args.slsqp_rebuild_tape_per_trial),
            freidberg_branch_audit=bool(args.freidberg_branch_audit),
        )
        optimize_payload = _attach_area_control_summary(optimize_payload, config)
        _write_json(out_dir / "run_summary.json", optimize_payload)
    benchmark_payload: dict[str, Any] = {
        "ok": bool(optimize_payload.get("ok", False)),
        "case_config": config.to_dict(),
        "reference_profile_metrics_json": str(out_dir / "reference_profile_metrics.json"),
        "optimizer_run_summary_json": str(out_dir / "run_summary.json"),
        "area_control_box_summary": optimize_payload.get("area_control_box_summary"),
    }
    if optimize_payload.get("metrics") is not None:
        benchmark_payload["comparison"] = compare_candidate_to_reference(
            reference=reference_metrics,
            candidate=dict(optimize_payload["metrics"]),
            config=config,
        )
    else:
        benchmark_payload["optimizer_error"] = optimize_payload.get("error", "optimizer did not return final metrics")
    _write_json(out_dir / "benchmark_summary.json", benchmark_payload)
    print(json.dumps(benchmark_payload, indent=2, sort_keys=True, default=_json_default))
    return 0 if bool(benchmark_payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
