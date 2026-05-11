#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze a v6 CasADi continuation_summary.json and identify the first failure point."
    )
    p.add_argument("summary_json", type=str, help="Path to continuation_summary.json")
    p.add_argument(
        "--out-md",
        type=str,
        default="",
        help="Optional Markdown report path; default prints only to stdout.",
    )
    return p


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _diag(stage: dict[str, Any]) -> dict[str, Any]:
    artifacts = dict(stage.get("artifacts", {}) or {})
    return dict(artifacts.get("diagnostics", {}) or {})


def _dual_summary(stage: dict[str, Any]) -> dict[str, Any]:
    artifacts = dict(stage.get("artifacts", {}) or {})
    return dict(artifacts.get("dual_summary", {}) or {})


def _top_duals(stage: dict[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, payload in _dual_summary(stage).items():
        item = dict(payload or {})
        value = _as_float(item.get("max_abs"))
        if value == value:
            rows.append(
                {
                    "name": str(name),
                    "max_abs": value,
                    "max_abs_index": int(item.get("max_abs_index", -1)),
                    "active_fraction_gt_1e_8": _as_float(item.get("active_fraction_gt_1e_8")),
                }
            )
    rows.sort(key=lambda item: float(item["max_abs"]), reverse=True)
    return rows[:limit]


def _stage_row(stage: dict[str, Any]) -> dict[str, Any]:
    diag = _diag(stage)
    return {
        "name": stage.get("name", ""),
        "stage_kind": stage.get("stage_kind", ""),
        "alpha": stage.get("adaptive_bridge_alpha", None),
        "success": bool(stage.get("success", False)),
        "acceptable": bool(stage.get("acceptable", False)),
        "return_status": stage.get("return_status", ""),
        "dTe_K": _as_float(stage.get("objective_delta_Te_K")),
        "tp_min": _as_float(diag.get("tp_min")),
        "max_constraint_violation": _as_float(stage.get("max_constraint_violation", diag.get("max_constraint_violation"))),
        "dynamic_defect_inf": _as_float(stage.get("dynamic_defect_inf", diag.get("dynamic_defect_inf"))),
        "sigma_bound_hit_fraction": _as_float(
            stage.get("sigma_bound_hit_fraction", diag.get("sigma_bound_hit_fraction"))
        ),
        "sigma_sign_changes": int(stage.get("sigma_sign_changes", diag.get("sigma_sign_changes", 0))),
        "area_step_sign_changes": int(
            stage.get("area_step_sign_changes", diag.get("area_step_sign_changes", 0))
        ),
        "min_velikhov_margin": _as_float(
            stage.get("min_velikhov_margin", diag.get("velikhov_margin_min"))
        ),
        "min_mach": _as_float(stage.get("min_mach", diag.get("mach_min"))),
        "warm_start_adopted": bool(stage.get("warm_start_adopted", False)),
        "warm_start_rejection_reason": stage.get("warm_start_rejection_reason", ""),
        "top_duals": _top_duals(stage),
    }


def _first_index(rows: list[dict[str, Any]], predicate) -> int | None:
    for i, row in enumerate(rows):
        if predicate(row):
            return i
    return None


def _classify_failure(first_bad: dict[str, Any] | None, final: dict[str, Any] | None) -> list[str]:
    if first_bad is None:
        return ["No failing or unacceptable stage was found in this summary."]

    reasons: list[str] = []
    status = str(first_bad.get("return_status", ""))
    if status and status != "Solve_Succeeded":
        reasons.append(f"solver_status={status}")
    if _as_float(first_bad.get("tp_min")) <= 0.0:
        reasons.append("nonphysical_Tp_min")
    if _as_float(first_bad.get("max_constraint_violation")) > 1e-2:
        reasons.append("large_constraint_violation")
    if _as_float(first_bad.get("dynamic_defect_inf")) > 5e-2:
        reasons.append("large_dynamic_defect")
    if int(first_bad.get("sigma_sign_changes", 0)) > 0 or int(first_bad.get("area_step_sign_changes", 0)) > 0:
        reasons.append("oscillatory_area_or_sigma")

    top_names = [str(item.get("name", "")) for item in first_bad.get("top_duals", [])[:3]]
    if top_names:
        reasons.append("top_duals=" + ",".join(top_names))

    if final is not None and final is not first_bad:
        if _as_float(final.get("tp_min")) <= 0.0:
            reasons.append("final_profile_has_negative_Tp")
        if _as_float(final.get("max_constraint_violation")) > _as_float(first_bad.get("max_constraint_violation")):
            reasons.append("final_constraint_violation_worsens")
    return reasons


def analyze(summary: dict[str, Any]) -> dict[str, Any]:
    rows = [_stage_row(dict(stage)) for stage in summary.get("stages", []) or []]
    first_non_success_i = _first_index(rows, lambda row: not bool(row["success"]))
    first_unacceptable_i = _first_index(rows, lambda row: not bool(row["acceptable"]))
    first_bad_i_candidates = [i for i in (first_non_success_i, first_unacceptable_i) if i is not None]
    first_bad_i = min(first_bad_i_candidates) if first_bad_i_candidates else None
    first_bad = None if first_bad_i is None else rows[first_bad_i]

    last_good_before_bad = None
    if first_bad_i is not None:
        for row in reversed(rows[:first_bad_i]):
            if bool(row["success"]) and bool(row["acceptable"]) and bool(row["warm_start_adopted"]):
                last_good_before_bad = row
                break

    final = rows[-1] if rows else None
    return {
        "ok": bool(summary.get("ok", False)),
        "solver_success": bool(summary.get("solver_success", False)),
        "final_return_status": summary.get("final_return_status", ""),
        "stopped_after_failed_bridge": bool(summary.get("stopped_after_failed_bridge", False)),
        "bridge_stop": dict(summary.get("bridge_stop", {}) or {}),
        "last_trusted_stage_summary": dict(summary.get("last_trusted_stage", {}) or {}),
        "stage_count": len(rows),
        "first_non_success_index": first_non_success_i,
        "first_unacceptable_index": first_unacceptable_i,
        "last_good_before_first_bad": last_good_before_bad,
        "first_bad_stage": first_bad,
        "final_stage": final,
        "failure_reasons": _classify_failure(first_bad, final),
    }


def _format_stage(row: dict[str, Any] | None) -> str:
    if row is None:
        return "- none\n"
    top_duals = ", ".join(
        f"{item['name']}={item['max_abs']:.3g}@{item['max_abs_index']}"
        for item in row.get("top_duals", [])[:5]
    )
    return "\n".join(
        [
            f"- name: `{row['name']}`",
            f"- kind: `{row['stage_kind']}`, alpha: `{row['alpha']}`",
            f"- success / acceptable: `{row['success']}` / `{row['acceptable']}`",
            f"- status: `{row['return_status']}`",
            f"- dTe_K: `{row['dTe_K']:.6g}`",
            f"- Tp_min: `{row['tp_min']:.6g}`",
            f"- max_constraint_violation: `{row['max_constraint_violation']:.6g}`",
            f"- dynamic_defect_inf: `{row['dynamic_defect_inf']:.6g}`",
            f"- sigma_hit_fraction: `{row['sigma_bound_hit_fraction']:.6g}`",
            f"- sigma / area sign changes: `{row['sigma_sign_changes']}` / `{row['area_step_sign_changes']}`",
            f"- min Velikhov margin / min Mach: `{row['min_velikhov_margin']:.6g}` / `{row['min_mach']:.6g}`",
            f"- warm-start adopted: `{row['warm_start_adopted']}` `{row['warm_start_rejection_reason']}`",
            f"- top duals: {top_duals or 'none'}",
        ]
    ) + "\n"


def format_markdown(report: dict[str, Any], *, source: Path) -> str:
    reasons = "\n".join(f"- `{reason}`" for reason in report["failure_reasons"])
    bridge_stop = dict(report.get("bridge_stop", {}) or {})
    bridge_lines = ["- none"]
    if bridge_stop:
        bridge_lines = [
            f"- blocked_target_stage: `{bridge_stop.get('blocked_target_stage', '')}`",
            f"- reason: `{bridge_stop.get('reason', '')}`",
            f"- max_stable_alpha: `{bridge_stop.get('max_stable_alpha', None)}`",
            f"- next_failed_alpha: `{bridge_stop.get('next_failed_alpha', None)}`",
            f"- last_trusted_stage: `{dict(bridge_stop.get('last_trusted_stage', {}) or {}).get('name', '')}`",
            f"- failed_stage: `{dict(bridge_stop.get('failed_stage', {}) or {}).get('name', '')}`",
        ]
    return "\n".join(
        [
            f"# Continuation Failure Analysis: {source}",
            "",
            "## Summary",
            "",
            f"- ok: `{report['ok']}`",
            f"- solver_success: `{report['solver_success']}`",
            f"- final_return_status: `{report['final_return_status']}`",
            f"- stopped_after_failed_bridge: `{report['stopped_after_failed_bridge']}`",
            f"- stage_count: `{report['stage_count']}`",
            f"- first_non_success_index: `{report['first_non_success_index']}`",
            f"- first_unacceptable_index: `{report['first_unacceptable_index']}`",
            f"- last_trusted_stage: `{dict(report.get('last_trusted_stage_summary', {}) or {}).get('name', '')}`",
            "",
            "## Bridge Stop",
            "",
            "\n".join(bridge_lines),
            "",
            "## Failure Reasons",
            "",
            reasons,
            "",
            "## Last Good Warm Start Before First Bad Stage",
            "",
            _format_stage(report["last_good_before_first_bad"]),
            "",
            "## First Bad Stage",
            "",
            _format_stage(report["first_bad_stage"]),
            "",
            "## Final Stage",
            "",
            _format_stage(report["final_stage"]),
        ]
    )


def main() -> int:
    args = _build_parser().parse_args()
    source = Path(args.summary_json)
    summary = json.loads(source.read_text(encoding="utf-8"))
    report = analyze(summary)
    text = format_markdown(report, source=source)
    print(text)
    if args.out_md:
        out_path = Path(args.out_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
