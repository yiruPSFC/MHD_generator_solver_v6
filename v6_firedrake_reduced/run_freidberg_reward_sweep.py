from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .design import DESIGN_VARIABLE_NAMES
from .freidberg_area_only import AREA_CONTROL_NAMES
from .run_firedrake_reduced import _json_default


METRIC_FIELDS = (
    "objective_score",
    "raw_enthalpy_extraction_percent",
    "mhd_output_power_W",
    "min_T_p_K",
    "max_Te_over_Tp",
    "min_velikhov_margin",
    "thermal_window_penalty",
    "thermal_window_penalty_candidate",
    "thermal_Tp_in_penalty_candidate",
    "thermal_Tp_low_penalty_candidate",
    "thermal_Tp_high_penalty_candidate",
    "thermal_Te_over_Tp_penalty_candidate",
)

METADATA_FIELDS = (
    "thermal_window_mode",
    "thermal_tp_floor_K",
    "thermal_tp_path_penalty_weight",
    "thermal_te_over_tp_max",
    "thermal_te_over_tp_penalty_weight",
)

COMBINED_PRESETS = (
    (450.0, 10.0, 12.0, 10.0),
    (600.0, 10.0, 12.0, 10.0),
    (450.0, 100.0, 10.0, 100.0),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _float_label(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _area_summary_from_case(case_config: dict[str, Any]) -> dict[str, Any]:
    design = dict(case_config.get("design", {}))
    bounds = dict(case_config.get("bounds", {}))
    controls = []
    for name in AREA_CONTROL_NAMES:
        value = float(design.get(name, float("nan")))
        item_bounds = dict(bounds.get(name, {}))
        lower = float(item_bounds.get("min", float("nan")))
        upper = float(item_bounds.get("max", float("nan")))
        span = max(upper - lower, 1e-300)
        fraction = (value - lower) / span
        if fraction <= 1e-6:
            support = "lower_bound"
        elif fraction >= 1.0 - 1e-6:
            support = "upper_bound"
        else:
            support = "interior"
        controls.append(
            {
                "name": name,
                "value": value,
                "lower": lower,
                "upper": upper,
                "box_fraction": fraction,
                "support": support,
            }
        )
    return {
        "control_names": list(AREA_CONTROL_NAMES),
        "controls": controls,
        "summary": {item["name"]: item["support"] for item in controls},
    }


def _base_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    command = [
        str(args.python),
        "-m",
        "v6_firedrake_reduced.run_freidberg_area_only_benchmark",
        "--mode",
        "optimize",
        "--out-dir",
        str(run_dir),
        "--area-window-half-width",
        str(args.area_window_half_width),
        "--optimizer",
        str(args.optimizer),
        "--max-iterations",
        str(args.max_iterations),
        "--multistart",
        str(args.multistart),
        "--seed",
        str(args.seed),
    ]
    if args.n_intervals is not None:
        command.extend(["--n-intervals", str(args.n_intervals)])
    velikhov_mode = args.velikhov_constraint_mode
    if velikhov_mode is None:
        velikhov_mode = "hard" if str(args.optimizer) == "constrained_slsqp" else "none"
    command.extend(["--velikhov-constraint-mode", str(velikhov_mode)])
    if str(velikhov_mode) == "hard":
        command.extend(["--velikhov-hard-floor", str(args.velikhov_hard_floor)])
    if str(args.optimizer) == "constrained_slsqp" and bool(args.slsqp_trial_continuation):
        command.append("--slsqp-trial-continuation")
        command.extend(["--slsqp-continuation-T-p-floor-K", str(args.slsqp_continuation_T_p_floor_K)])
    if bool(args.slsqp_rebuild_tape_per_trial):
        command.append("--slsqp-rebuild-tape-per-trial")
    if bool(args.freidberg_branch_audit):
        command.append("--freidberg-branch-audit")
    return command


def _experiments(args: argparse.Namespace) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = [
        {
            "name": "baseline_diagnostic",
            "family": "baseline",
            "thermal_args": ["--thermal-window-mode", "diagnostic"],
        }
    ]
    for floor in _parse_float_list(args.thermal_floors):
        for weight in _parse_float_list(args.thermal_tp_weights):
            experiments.append(
                {
                    "name": f"thermal_floor{_float_label(floor)}_w{_float_label(weight)}",
                    "family": "thermal_guard",
                    "thermal_args": [
                        "--thermal-window-mode",
                        "penalty",
                        "--thermal-tp-floor",
                        str(floor),
                        "--thermal-tp-path-penalty-weight",
                        str(weight),
                    ],
                }
            )
    for ratio_max in _parse_float_list(args.ratio_max_values):
        for weight in _parse_float_list(args.ratio_weights):
            experiments.append(
                {
                    "name": f"ratio_max{_float_label(ratio_max)}_w{_float_label(weight)}",
                    "family": "ratio_availability",
                    "thermal_args": [
                        "--thermal-window-mode",
                        "penalty",
                        "--thermal-te-over-tp-max",
                        str(ratio_max),
                        "--thermal-te-over-tp-penalty-weight",
                        str(weight),
                        "--thermal-tp-floor",
                        str(args.ratio_guard_tp_floor),
                        "--thermal-tp-path-penalty-weight",
                        str(args.ratio_guard_tp_weight),
                    ],
                }
            )
    if bool(args.include_combined):
        for floor, tp_weight, ratio_max, ratio_weight in COMBINED_PRESETS:
            experiments.append(
                {
                    "name": (
                        f"combined_floor{_float_label(floor)}_tw{_float_label(tp_weight)}"
                        f"_rmax{_float_label(ratio_max)}_rw{_float_label(ratio_weight)}"
                    ),
                    "family": "combined",
                    "thermal_args": [
                        "--thermal-window-mode",
                        "penalty",
                        "--thermal-tp-floor",
                        str(floor),
                        "--thermal-tp-path-penalty-weight",
                        str(tp_weight),
                        "--thermal-te-over-tp-max",
                        str(ratio_max),
                        "--thermal-te-over-tp-penalty-weight",
                        str(ratio_weight),
                    ],
                }
            )
    return experiments


def _run_logged(command: list[str], *, cwd: Path, log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        result = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(result.returncode)


def _run_kkt(args: argparse.Namespace, run_dir: Path, *, cwd: Path) -> int | None:
    summary_path = run_dir / "run_summary.json"
    if bool(args.skip_kkt) or not summary_path.exists():
        return None
    try:
        summary = _read_json(summary_path)
    except Exception:
        return None
    if not bool(summary.get("ok", False)):
        return None
    command = [
        str(args.python),
        "-m",
        "v6_firedrake_reduced.analyze_kkt",
        str(summary_path),
        "--out",
        str(run_dir / "kkt_analysis.json"),
        "--active-tol",
        str(args.kkt_active_tol),
    ]
    return _run_logged(command, cwd=cwd, log_path=run_dir / "kkt_analysis.log")


def _summary_row(run_dir: Path, experiment: dict[str, Any], *, returncode: int | None, kkt_returncode: int | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_name": experiment["name"],
        "family": experiment["family"],
        "returncode": returncode,
        "kkt_returncode": kkt_returncode,
        "ok": False,
        "kkt_ok": False,
        "error": None,
    }
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        row["error"] = "missing run_summary.json"
        return row
    try:
        summary = _read_json(summary_path)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    row["ok"] = bool(summary.get("ok", False))
    row["error"] = summary.get("error")
    metrics = dict(summary.get("metrics") or {})
    for field in METRIC_FIELDS:
        row[field] = metrics.get(field)
    metadata = dict((summary.get("case_config") or {}).get("metadata") or {})
    for field in METADATA_FIELDS:
        row[field] = metadata.get(field)

    area_summary = summary.get("area_control_box_summary")
    if not isinstance(area_summary, dict):
        area_summary = _area_summary_from_case(dict(summary.get("case_config") or {}))
    area_controls = {item["name"]: item for item in area_summary.get("controls", [])}
    for name in AREA_CONTROL_NAMES:
        item = dict(area_controls.get(name, {}))
        row[f"{name}_box_fraction"] = item.get("box_fraction")
        row[f"{name}_support"] = item.get("support")

    kkt_path = run_dir / "kkt_analysis.json"
    if not kkt_path.exists():
        return row
    try:
        kkt = _read_json(kkt_path)
    except Exception as exc:
        row["kkt_error"] = f"{type(exc).__name__}: {exc}"
        return row
    row["kkt_ok"] = True
    quality = dict((kkt.get("recovered_multipliers") or {}).get("quality") or {})
    row["kkt_stationarity_inf"] = quality.get("stationarity_inf")
    row["kkt_stationarity_l2"] = quality.get("stationarity_l2")
    controls = {item["name"]: item for item in kkt.get("control_stationarity", [])}
    area_residuals = []
    for name in AREA_CONTROL_NAMES:
        item = dict(controls.get(name, {}))
        residual = item.get("stationarity_residual")
        row[f"{name}_stationarity_residual"] = residual
        row[f"{name}_path_constraint_support"] = item.get("path_constraint_support")
        row[f"{name}_bound_constraint_support"] = item.get("bound_constraint_support")
        if residual is not None:
            area_residuals.append(abs(float(residual)))
    row["area_stationarity_inf"] = max(area_residuals) if area_residuals else None
    row["active_path_constraint_count"] = len(kkt.get("active_path_constraints", []))
    row["active_bound_constraint_count"] = len(kkt.get("active_bound_constraints", []))
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_name",
        "family",
        "returncode",
        "kkt_returncode",
        "ok",
        "kkt_ok",
        "error",
        *METADATA_FIELDS,
        *METRIC_FIELDS,
        "a1_box_fraction",
        "a1_support",
        "a2_box_fraction",
        "a2_support",
        "a3_box_fraction",
        "a3_support",
        "kkt_stationarity_inf",
        "kkt_stationarity_l2",
        "area_stationarity_inf",
        "a1_stationarity_residual",
        "a2_stationarity_residual",
        "a3_stationarity_residual",
        "a1_path_constraint_support",
        "a2_path_constraint_support",
        "a3_path_constraint_support",
        "a1_bound_constraint_support",
        "a2_bound_constraint_support",
        "a3_bound_constraint_support",
        "active_path_constraint_count",
        "active_bound_constraint_count",
    ]
    extras = sorted({key for row in rows for key in row if key not in fieldnames})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extras])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Freidberg area-only alternative-reward sweep and summarize KKT diagnostics."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--n-intervals", type=int, default=None)
    parser.add_argument("--area-window-half-width", type=float, default=3.0)
    parser.add_argument(
        "--optimizer",
        default="constrained_slsqp",
        choices=("coordinate_search", "projected_gradient", "constrained_slsqp"),
    )
    parser.add_argument("--velikhov-constraint-mode", default=None, choices=("none", "hard"))
    parser.add_argument("--velikhov-hard-floor", type=float, default=0.0)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--multistart", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--slsqp-trial-continuation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slsqp-continuation-T-p-floor-K", type=float, default=1.0)
    parser.add_argument("--slsqp-rebuild-tape-per-trial", action="store_true")
    parser.add_argument("--thermal-floors", default="300,450,600")
    parser.add_argument("--thermal-tp-weights", default="1,10,100")
    parser.add_argument("--ratio-max-values", default="10,12,15")
    parser.add_argument("--ratio-weights", default="1,10,100")
    parser.add_argument("--ratio-guard-tp-floor", type=float, default=300.0)
    parser.add_argument("--ratio-guard-tp-weight", type=float, default=1.0)
    parser.add_argument("--include-combined", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kkt-active-tol", type=float, default=1e-6)
    parser.add_argument("--skip-kkt", action="store_true")
    parser.add_argument("--freidberg-branch-audit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    experiments = _experiments(args)
    manifest = []
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        run_dir = out_dir / str(experiment["name"])
        command = [*_base_command(args, run_dir), *experiment["thermal_args"]]
        manifest.append(
            {
                "name": experiment["name"],
                "family": experiment["family"],
                "run_dir": str(run_dir),
                "command": command,
            }
        )
    _write_json(out_dir / "reward_sweep_manifest.json", {"experiments": manifest})
    if bool(args.dry_run):
        _write_json(out_dir / "reward_sweep_summary.json", {"dry_run": True, "rows": []})
        _write_csv(out_dir / "reward_sweep_summary.csv", [])
        print(json.dumps({"dry_run": True, "experiment_count": len(manifest), "manifest": str(out_dir / "reward_sweep_manifest.json")}, indent=2))
        return 0

    for item, experiment in zip(manifest, experiments, strict=True):
        run_dir = Path(item["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "experiment_definition.json", item)
        returncode = _run_logged(item["command"], cwd=cwd, log_path=run_dir / "run.log")
        kkt_returncode = _run_kkt(args, run_dir, cwd=cwd)
        rows.append(_summary_row(run_dir, experiment, returncode=returncode, kkt_returncode=kkt_returncode))
        _write_json(out_dir / "reward_sweep_summary.json", {"dry_run": False, "rows": rows})
        _write_csv(out_dir / "reward_sweep_summary.csv", rows)
        if bool(args.fail_fast) and returncode not in (0, None):
            break
    print(
        json.dumps(
            {
                "experiment_count": len(rows),
                "summary_json": str(out_dir / "reward_sweep_summary.json"),
                "summary_csv": str(out_dir / "reward_sweep_summary.csv"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
