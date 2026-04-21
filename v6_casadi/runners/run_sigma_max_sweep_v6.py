#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_CASADI_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_casadi.run_casadi_continuation_v6 import run_continuation


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Sweep final-stage max_abs_dlogA_dx for one inlet candidate and summarize "
            "performance plus IPOPT/CasADi constraint duals."
        )
    )
    p.add_argument(
        "--source-summary-json",
        type=str,
        default=str(_CASADI_DIR / "outputs" / "continuation" / "scenario_compare_coal_mhd_htah_strict_gate" / "search_summary.json"),
        help="search_summary.json containing the source candidate and schedule",
    )
    p.add_argument(
        "--candidate-index",
        type=int,
        default=-1,
        help="candidate_index to use from source summary; default uses best_candidate",
    )
    p.add_argument(
        "--sigma-values",
        type=str,
        default="0.5,0.75,1.0,1.5,2.0",
        help="comma-separated final-stage max_abs_dlogA_dx values",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(_CASADI_DIR / "outputs" / "continuation" / "sigma_max_sweep"),
    )
    p.add_argument("--warm-start-dx", type=float, default=0.01)
    p.add_argument("--adaptive-bridge-count", type=int, default=-1)
    p.add_argument("--adaptive-bridge-max-count", type=int, default=-1)
    return p


def _parse_sigma_values(raw: str) -> list[float]:
    values = []
    for item in str(raw).split(","):
        token = item.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0.0:
            raise ValueError("sigma values must be positive.")
        values.append(value)
    if not values:
        raise ValueError("at least one sigma value is required.")
    return values


def _source_candidate(source: dict, *, candidate_index: int) -> dict:
    if candidate_index < 0:
        best = source.get("best_candidate")
        if not isinstance(best, dict) or not isinstance(best.get("candidate"), dict):
            raise ValueError("source summary has no best_candidate.candidate.")
        return dict(best["candidate"])

    for section_name in ("evaluated_candidates", "prefilter_ranked_candidates"):
        for item in source.get(section_name, []) or []:
            candidate = dict(item.get("candidate", {}) or {})
            if int(candidate.get("candidate_index", -999999)) == int(candidate_index):
                return candidate
    raise ValueError(f"candidate_index={candidate_index} not found in source summary.")


def _source_continuation_settings(source: dict, *, candidate_index: int) -> dict:
    if candidate_index < 0:
        best = source.get("best_candidate")
        if isinstance(best, dict) and isinstance(best.get("continuation"), dict):
            return dict(best["continuation"])
    for item in source.get("evaluated_candidates", []) or []:
        candidate = dict(item.get("candidate", {}) or {})
        if int(candidate.get("candidate_index", -999999)) == int(candidate_index):
            return dict(item.get("continuation", {}) or {})
    return {}


def _schedule_for_sigma(source_schedule: list[dict], *, sigma_max: float) -> list[dict]:
    schedule = [dict(stage) for stage in source_schedule]
    if not schedule:
        raise ValueError("source schedule is empty.")
    final_idx = None
    for i, stage in enumerate(schedule):
        if str(stage.get("name", "")).endswith("arbitrary_final"):
            final_idx = i
    if final_idx is None:
        final_idx = len(schedule) - 1
    schedule[final_idx]["max_abs_dlogA_dx"] = float(sigma_max)
    schedule[final_idx]["name"] = f"{schedule[final_idx]['name']}__sigma_{float(sigma_max):g}".replace(".", "p")
    return schedule


def _finite_float(value, default=float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out


def _dual_max(stage_record: dict, name: str) -> float:
    summary = (
        dict(stage_record.get("artifacts", {}) or {})
        .get("dual_summary", {})
    )
    if not isinstance(summary, dict):
        return float("nan")
    return _finite_float(dict(summary.get(name, {}) or {}).get("max_abs", float("nan")))


def _top_duals(stage_record: dict, *, limit: int = 6) -> list[dict[str, object]]:
    summary = dict(dict(stage_record.get("artifacts", {}) or {}).get("dual_summary", {}) or {})
    rows = []
    for name, payload in summary.items():
        value = _finite_float(dict(payload or {}).get("max_abs", float("nan")))
        if np.isfinite(value):
            rows.append(
                {
                    "name": str(name),
                    "max_abs": value,
                    "max_abs_index": int(dict(payload or {}).get("max_abs_index", -1)),
                    "active_fraction_gt_1e_8": _finite_float(
                        dict(payload or {}).get("active_fraction_gt_1e_8", float("nan"))
                    ),
                }
            )
    rows.sort(key=lambda item: float(item["max_abs"]), reverse=True)
    return rows[: int(limit)]


def _summarize_final_stage(*, sigma_max: float, payload: dict, case_dir: Path) -> dict:
    stages = [dict(stage) for stage in payload.get("stages", []) or []]
    final = stages[-1] if stages else {}
    artifacts = dict(final.get("artifacts", {}) or {})
    diagnostics = dict(artifacts.get("diagnostics", {}) or payload.get("final_diagnostics", {}) or {})
    return {
        "sigma_max": float(sigma_max),
        "ok": bool(payload.get("ok", False)),
        "final_stage": str(final.get("name", "")),
        "success": bool(final.get("success", False)),
        "acceptable": bool(final.get("acceptable", False)),
        "return_status": str(final.get("return_status", "")),
        "objective_delta_Te_K": _finite_float(final.get("objective_delta_Te_K", float("nan"))),
        "sigma_bound_hit_fraction": _finite_float(diagnostics.get("sigma_bound_hit_fraction", float("nan"))),
        "sigma_sign_changes": int(diagnostics.get("sigma_sign_changes", 0)),
        "area_step_sign_changes": int(diagnostics.get("area_step_sign_changes", 0)),
        "min_velikhov_margin": _finite_float(final.get("min_velikhov_margin", diagnostics.get("velikhov_margin_min", float("nan")))),
        "min_mach": _finite_float(final.get("min_mach", diagnostics.get("mach_min", float("nan")))),
        "tp_min": _finite_float(diagnostics.get("tp_min", float("nan"))),
        "outlet_area_m2": _finite_float(final.get("outlet_area_m2", float("nan"))),
        "dynamic_defect_inf": _finite_float(final.get("dynamic_defect_inf", diagnostics.get("dynamic_defect_inf", float("nan")))),
        "max_constraint_violation": _finite_float(final.get("max_constraint_violation", diagnostics.get("max_constraint_violation", float("nan")))),
        "dual_sigma_upper_max_abs": _dual_max(final, "sigma_upper_interval"),
        "dual_G_lower_node_max_abs": _dual_max(final, "G_lower_node"),
        "dual_G_lower_mid_max_abs": _dual_max(final, "G_lower_mid"),
        "dual_Mach_lower_node_max_abs": _dual_max(final, "Mach_lower_node"),
        "dual_Tp_lower_node_max_abs": _dual_max(final, "Tp_lower_node"),
        "dual_A_upper_node_max_abs": _dual_max(final, "A_upper_node"),
        "top_duals": _top_duals(final),
        "case_dir": str(case_dir),
        "plot_path": str(artifacts.get("plot_path", "")),
        "dual_plot_path": str(artifacts.get("dual_plot_path", "")),
        "npz_path": str(artifacts.get("npz_path", "")),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "sigma_max",
        "ok",
        "success",
        "acceptable",
        "return_status",
        "objective_delta_Te_K",
        "sigma_bound_hit_fraction",
        "min_velikhov_margin",
        "min_mach",
        "tp_min",
        "outlet_area_m2",
        "dynamic_defect_inf",
        "max_constraint_violation",
        "dual_sigma_upper_max_abs",
        "dual_G_lower_node_max_abs",
        "dual_G_lower_mid_max_abs",
        "dual_Mach_lower_node_max_abs",
        "dual_Tp_lower_node_max_abs",
        "dual_A_upper_node_max_abs",
        "case_dir",
        "plot_path",
        "dual_plot_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    args = _build_parser().parse_args()
    source_path = Path(args.source_summary_json)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    candidate = _source_candidate(source, candidate_index=int(args.candidate_index))
    continuation_settings = _source_continuation_settings(source, candidate_index=int(args.candidate_index))
    sigma_values = _parse_sigma_values(args.sigma_values)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adaptive_bridge_count = (
        int(args.adaptive_bridge_count)
        if int(args.adaptive_bridge_count) >= 0
        else int(continuation_settings.get("adaptive_bridge_count", 3))
    )
    adaptive_bridge_max_count = (
        int(args.adaptive_bridge_max_count)
        if int(args.adaptive_bridge_max_count) >= 0
        else int(continuation_settings.get("adaptive_bridge_max_count", 12))
    )

    rows = []
    cases = []
    for sigma_max in sigma_values:
        case_dir = out_dir / f"sigma_{float(sigma_max):g}".replace(".", "p")
        schedule = _schedule_for_sigma(list(source["schedule"]), sigma_max=float(sigma_max))
        payload = run_continuation(
            n_p_in=float(candidate["n_p_in"]),
            Z_in=float(candidate["Z_in"]),
            T_p_in=float(candidate["T_p_in"]),
            T_e_in=float(candidate["T_e_in"]),
            A_in=float(candidate["A_in"]),
            B=float(source["B_T"]),
            L=float(source["L_m"]),
            seed_fraction=None,
            warm_start_dx=float(args.warm_start_dx),
            stage_schedule=schedule,
            out_dir=case_dir,
            stop_on_unacceptable=False,
            warm_start_policy=str(continuation_settings.get("warm_start_policy", "regular")),
            adaptive_bridge_count=adaptive_bridge_count,
            adaptive_bridge_max_count=adaptive_bridge_max_count,
        )
        row = _summarize_final_stage(sigma_max=float(sigma_max), payload=payload, case_dir=case_dir)
        rows.append(row)
        cases.append(
            {
                "sigma_max": float(sigma_max),
                "schedule": schedule,
                "payload": payload,
                "summary": row,
            }
        )
        print(
            json.dumps(
                {
                    "sigma_max": float(sigma_max),
                    "acceptable": row["acceptable"],
                    "dTe_K": row["objective_delta_Te_K"],
                    "sigma_bound_hit_fraction": row["sigma_bound_hit_fraction"],
                    "top_duals": row["top_duals"][:3],
                },
                ensure_ascii=False,
            )
        )

    summary = {
        "source_summary_json": str(source_path),
        "candidate": candidate,
        "B_T": float(source["B_T"]),
        "L_m": float(source["L_m"]),
        "adaptive_bridge_count": adaptive_bridge_count,
        "adaptive_bridge_max_count": adaptive_bridge_max_count,
        "sigma_values": sigma_values,
        "rows": rows,
        "cases": cases,
    }
    summary_path = out_dir / "sigma_max_sweep_summary.json"
    csv_path = out_dir / "sigma_max_sweep_summary.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows)
    print(json.dumps({"summary_path": str(summary_path), "csv_path": str(csv_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
