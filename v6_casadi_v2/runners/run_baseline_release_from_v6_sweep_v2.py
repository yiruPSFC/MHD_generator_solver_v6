#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_CASADI_V2_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_casadi_v2.run_casadi_continuation_v2 import load_warm_profile_npz, run_continuation


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Reconstruct a v2 continuation run from a v6 sigma sweep case, using the "
            "selected v6 profile as a baseline anchor and then releasing into a v2 objective."
        )
    )
    p.add_argument(
        "--source-sweep-json",
        type=str,
        default=str(
            REPO_DIR
            / "v6_casadi"
            / "outputs"
            / "continuation"
            / "sigma_max_sweep_candidate_022"
            / "sigma_max_sweep_summary.json"
        ),
        help="v6 sigma_max sweep summary json produced by run_sigma_max_sweep_v6",
    )
    p.add_argument(
        "--sigma-value",
        type=float,
        default=0.5,
        help="sigma_max entry to extract from the v6 sweep summary",
    )
    p.add_argument(
        "--candidate-index",
        type=int,
        default=-1,
        help="candidate_index to recover from the source summary; default uses the sweep summary candidate",
    )
    p.add_argument(
        "--inlet-margin-mode",
        type=str,
        choices=("equality", "lower-bound"),
        default="lower-bound",
        help="lower-bound is safer when reproducing a v6 warm profile with small positive inlet G",
    )
    p.add_argument("--np-relative-window", type=float, default=0.05)
    p.add_argument("--te-relative-window", type=float, default=0.05)
    p.add_argument("--z-relative-window", type=float, default=0.10)
    p.add_argument("--jx-relative-window", type=float, default=0.10)
    p.add_argument("--seed-lower-factor", type=float, default=0.5)
    p.add_argument("--seed-upper-factor", type=float, default=2.0)
    p.add_argument(
        "--anchor-sigma-extra",
        type=float,
        default=0.05,
        help="extra sigma headroom used in the initial baseline anchor stage",
    )
    p.add_argument("--anchor-warm-profile-track-weight", type=float, default=4096.0)
    p.add_argument("--anchor-warm-control-track-weight", type=float, default=1024.0)
    p.add_argument("--anchor-ipopt-max-iter", type=int, default=2200)
    p.add_argument("--release-warm-profile-track-weight", type=float, default=12.0)
    p.add_argument("--release-warm-control-track-weight", type=float, default=3.0)
    p.add_argument("--release-ipopt-max-iter", type=int, default=2600)
    p.add_argument("--final-warm-profile-track-weight", type=float, default=1.0)
    p.add_argument("--final-warm-control-track-weight", type=float, default=0.25)
    p.add_argument("--final-ipopt-max-iter", type=int, default=3200)
    p.add_argument("--adaptive-bridge-count", type=int, default=2)
    p.add_argument("--adaptive-bridge-max-count", type=int, default=8)
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="directory for the aligned v2 continuation artifacts",
    )
    p.add_argument("--out-json", type=str, default="")
    return p


def _format_sigma_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _relative_bounds(center: float, half_window: float, *, floor: float) -> tuple[float, float]:
    half_window = max(float(half_window), 0.0)
    lower = float(center) * (1.0 - half_window)
    upper = float(center) * (1.0 + half_window)
    lower = max(float(floor), lower)
    upper = max(lower + max(abs(float(center)) * 1e-12, 1e-12), upper)
    return float(lower), float(upper)


def _seed_bounds(seed: float, *, lower_factor: float, upper_factor: float) -> tuple[float, float]:
    lower_factor = max(float(lower_factor), 1e-6)
    upper_factor = max(float(upper_factor), lower_factor + 1e-6)
    lower = max(1e-13, float(seed) * lower_factor)
    upper = min(5e-2, float(seed) * upper_factor)
    upper = max(lower * (1.0 + 1e-6), upper)
    return float(lower), float(upper)


def _select_case_for_sigma(sweep_summary: dict, *, sigma_value: float) -> dict:
    cases = list(sweep_summary.get("cases", []) or [])
    if not cases:
        raise ValueError("source sweep summary has no cases.")
    best = min(
        cases,
        key=lambda item: abs(float(item.get("sigma_max", float("inf"))) - float(sigma_value)),
    )
    best_sigma = float(best.get("sigma_max", float("nan")))
    if not np.isfinite(best_sigma):
        raise ValueError("could not identify a finite sigma_max case in the sweep summary.")
    if abs(best_sigma - float(sigma_value)) > 1e-9:
        raise ValueError(
            f"sigma_value={sigma_value:g} not found in sweep summary; nearest entry is {best_sigma:g}."
        )
    return dict(best)


def _candidate_index_from_sweep(sweep_summary: dict, *, override: int) -> int:
    if int(override) >= 0:
        return int(override)
    candidate = dict(sweep_summary.get("candidate", {}) or {})
    if "candidate_index" not in candidate:
        raise ValueError("sweep summary has no top-level candidate.candidate_index; pass --candidate-index.")
    return int(candidate["candidate_index"])


def _source_candidate_entry(source_summary: dict, *, candidate_index: int) -> dict:
    best = dict(source_summary.get("best_candidate", {}) or {})
    best_candidate = dict(best.get("candidate", {}) or {})
    if int(best_candidate.get("candidate_index", -999999)) == int(candidate_index):
        return best

    for section_name in ("evaluated_candidates", "prefilter_ranked_candidates"):
        for item in source_summary.get(section_name, []) or []:
            item_dict = dict(item or {})
            candidate = dict(item_dict.get("candidate", {}) or {})
            if int(candidate.get("candidate_index", -999999)) == int(candidate_index):
                return item_dict
    raise ValueError(f"candidate_index={candidate_index} not found in the source summary.")


def _preferred_warm_npz(case: dict) -> Path:
    summary = dict(case.get("summary", {}) or {})
    for key in ("trusted_npz_path", "npz_path"):
        raw = str(summary.get(key, "")).strip()
        if raw:
            return Path(raw)
    raise ValueError("selected sweep case has no trusted_npz_path or npz_path.")


def _build_stage_schedule(
    *,
    source_schedule: list[dict],
    selected_sigma: float,
    warm_A: np.ndarray,
    anchor_sigma_extra: float,
    anchor_warm_profile_track_weight: float,
    anchor_warm_control_track_weight: float,
    anchor_ipopt_max_iter: int,
    release_warm_profile_track_weight: float,
    release_warm_control_track_weight: float,
    release_ipopt_max_iter: int,
    final_warm_profile_track_weight: float,
    final_warm_control_track_weight: float,
    final_ipopt_max_iter: int,
) -> list[dict]:
    schedule = [dict(stage) for stage in source_schedule]
    if len(schedule) < 3:
        raise ValueError("expected at least three stages in the source schedule.")

    source_anchor = dict(schedule[0])
    source_release = dict(schedule[1])
    source_final = dict(schedule[-1])
    warm_A_max = float(np.nanmax(np.asarray(warm_A, dtype=float)))

    anchor = dict(source_anchor)
    anchor["name"] = "stage_1_hs_v6_baseline_anchor"
    anchor["A_max_ratio"] = max(float(anchor.get("A_max_ratio", 0.0)), warm_A_max * 1.10, 2.2)
    anchor["max_abs_dlogA_dx"] = max(
        float(anchor.get("max_abs_dlogA_dx", 0.0)),
        float(selected_sigma) + max(float(anchor_sigma_extra), 0.0),
    )
    anchor["smooth_weight"] = 0.0
    anchor["control_slew_weight"] = 0.0
    anchor["control_curvature_weight"] = 0.0
    anchor["state_curvature_weight"] = 0.0
    anchor["warm_profile_track_weight"] = max(
        float(anchor.get("warm_profile_track_weight", 0.0)),
        float(anchor_warm_profile_track_weight),
    )
    anchor["warm_control_track_weight"] = max(
        float(anchor.get("warm_control_track_weight", 0.0)),
        float(anchor_warm_control_track_weight),
    )
    anchor["objective_weight"] = 0.0
    anchor["ipopt_max_iter"] = max(int(anchor.get("ipopt_max_iter", 0)), int(anchor_ipopt_max_iter))

    release = dict(source_release)
    release["name"] = "stage_2_hs_objective_release"
    release["max_abs_dlogA_dx"] = max(float(release.get("max_abs_dlogA_dx", 0.0)), float(selected_sigma))
    release["warm_profile_track_weight"] = max(
        float(release.get("warm_profile_track_weight", 0.0)),
        float(release_warm_profile_track_weight),
    )
    release["warm_control_track_weight"] = max(
        float(release.get("warm_control_track_weight", 0.0)),
        float(release_warm_control_track_weight),
    )
    release["ipopt_max_iter"] = max(int(release.get("ipopt_max_iter", 0)), int(release_ipopt_max_iter))

    final = dict(source_final)
    final["name"] = "stage_3_hs_objective_final"
    final["max_abs_dlogA_dx"] = float(selected_sigma)
    final["warm_profile_track_weight"] = max(
        float(final.get("warm_profile_track_weight", 0.0)),
        float(final_warm_profile_track_weight),
    )
    final["warm_control_track_weight"] = max(
        float(final.get("warm_control_track_weight", 0.0)),
        float(final_warm_control_track_weight),
    )
    final["ipopt_max_iter"] = max(int(final.get("ipopt_max_iter", 0)), int(final_ipopt_max_iter))

    return [anchor, release, final]


def main() -> int:
    args = _build_parser().parse_args()
    sweep_summary_path = Path(args.source_sweep_json)
    sweep_summary = _load_json(sweep_summary_path)
    raw_source_summary_path = str(sweep_summary.get("source_summary_json", "")).strip()
    if not raw_source_summary_path:
        raise ValueError("source sweep summary is missing source_summary_json.")
    source_summary_path = Path(raw_source_summary_path)
    source_summary = _load_json(source_summary_path)

    selected_case = _select_case_for_sigma(sweep_summary, sigma_value=float(args.sigma_value))
    candidate_index = _candidate_index_from_sweep(
        sweep_summary,
        override=int(args.candidate_index),
    )
    source_entry = _source_candidate_entry(source_summary, candidate_index=int(candidate_index))
    candidate = dict(source_entry.get("candidate", {}) or {})
    inlet_metrics = dict(source_entry.get("inlet_metrics", {}) or {})
    selected_sigma = float(selected_case.get("sigma_max", args.sigma_value))

    n_p_in_guess = float(candidate["n_p_in"])
    T_e_in_guess = float(candidate["T_e_in"])
    Z_in_guess = float(candidate["Z_in"])
    J_x_in_guess = float(inlet_metrics.get("J_x_in_A_m2", float("nan")))
    seed_fraction_guess = float(source_entry.get("seed_fraction", float("nan")))
    if not np.isfinite(J_x_in_guess) or J_x_in_guess <= 0.0:
        warm_probe = np.load(_preferred_warm_npz(selected_case))
        try:
            J_x_in_guess = float(np.asarray(warm_probe["J_x"], dtype=float).reshape(-1)[0])
        finally:
            warm_probe.close()
    if not np.isfinite(seed_fraction_guess) or seed_fraction_guess <= 0.0:
        seed_fraction_guess = 1e-4

    np_in_min, np_in_max = _relative_bounds(
        n_p_in_guess,
        float(args.np_relative_window),
        floor=1.0,
    )
    te_in_min, te_in_max = _relative_bounds(
        T_e_in_guess,
        float(args.te_relative_window),
        floor=100.0,
    )
    z_in_min, z_in_max = _relative_bounds(
        Z_in_guess,
        float(args.z_relative_window),
        floor=1.0,
    )
    jx_in_min, jx_in_max = _relative_bounds(
        J_x_in_guess,
        float(args.jx_relative_window),
        floor=1e-12,
    )
    seed_fraction_min, seed_fraction_max = _seed_bounds(
        seed_fraction_guess,
        lower_factor=float(args.seed_lower_factor),
        upper_factor=float(args.seed_upper_factor),
    )

    warm_profile_path = _preferred_warm_npz(selected_case)
    warm_profile = load_warm_profile_npz(
        warm_profile_path,
        n_p_in_guess=n_p_in_guess,
        n_p_in_min=np_in_min,
        n_p_in_max=np_in_max,
        T_e_in_guess=T_e_in_guess,
        T_e_in_min=te_in_min,
        T_e_in_max=te_in_max,
        Z_in_guess=Z_in_guess,
        Z_in_min=z_in_min,
        Z_in_max=z_in_max,
        J_x_in_guess=J_x_in_guess,
        J_x_in_min=jx_in_min,
        J_x_in_max=jx_in_max,
        seed_fraction_guess=seed_fraction_guess,
        seed_fraction_min=seed_fraction_min,
        seed_fraction_max=seed_fraction_max,
        B=float(sweep_summary["B_T"]),
    )
    stage_schedule = _build_stage_schedule(
        source_schedule=list(selected_case.get("schedule", []) or []),
        selected_sigma=selected_sigma,
        warm_A=np.asarray(warm_profile.A, dtype=float),
        anchor_sigma_extra=float(args.anchor_sigma_extra),
        anchor_warm_profile_track_weight=float(args.anchor_warm_profile_track_weight),
        anchor_warm_control_track_weight=float(args.anchor_warm_control_track_weight),
        anchor_ipopt_max_iter=int(args.anchor_ipopt_max_iter),
        release_warm_profile_track_weight=float(args.release_warm_profile_track_weight),
        release_warm_control_track_weight=float(args.release_warm_control_track_weight),
        release_ipopt_max_iter=int(args.release_ipopt_max_iter),
        final_warm_profile_track_weight=float(args.final_warm_profile_track_weight),
        final_warm_control_track_weight=float(args.final_warm_control_track_weight),
        final_ipopt_max_iter=int(args.final_ipopt_max_iter),
    )

    out_dir = (
        Path(args.out_dir)
        if str(args.out_dir).strip()
        else _CASADI_V2_DIR
        / "outputs"
        / "continuation"
        / f"baseline_release_from_v6_candidate_{candidate_index:03d}_sigma_{_format_sigma_token(selected_sigma)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = out_dir / "aligned_release_schedule.json"
    schedule_path.write_text(json.dumps(stage_schedule, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = run_continuation(
        n_p_in_guess=n_p_in_guess,
        n_p_in_min=np_in_min,
        n_p_in_max=np_in_max,
        T_e_in_guess=T_e_in_guess,
        T_e_in_min=te_in_min,
        T_e_in_max=te_in_max,
        Z_in_guess=Z_in_guess,
        Z_in_min=z_in_min,
        Z_in_max=z_in_max,
        J_x_in_guess=J_x_in_guess,
        J_x_in_min=jx_in_min,
        J_x_in_max=jx_in_max,
        seed_fraction_guess=seed_fraction_guess,
        seed_fraction_min=seed_fraction_min,
        seed_fraction_max=seed_fraction_max,
        inlet_margin_mode=str(args.inlet_margin_mode),
        B=float(sweep_summary["B_T"]),
        L=float(sweep_summary["L_m"]),
        stage_schedule=stage_schedule,
        out_dir=out_dir,
        out_json=args.out_json,
        stop_on_unacceptable=False,
        warm_start_policy="regular",
        adaptive_bridge_count=int(args.adaptive_bridge_count),
        adaptive_bridge_max_count=int(args.adaptive_bridge_max_count),
        warm_profile=warm_profile,
    )

    payload["schedule"] = stage_schedule
    payload["schedule_source"] = f"derived_from_v6_sigma_sweep:{sweep_summary_path}"
    payload["schedule_path"] = str(schedule_path)
    payload["source_alignment"] = {
        "source_sweep_json": str(sweep_summary_path),
        "source_summary_json": str(source_summary_path),
        "candidate_index": int(candidate_index),
        "candidate": candidate,
        "source_seed_fraction": float(source_entry.get("seed_fraction", float("nan"))),
        "source_inlet_metrics": inlet_metrics,
        "selected_sigma": float(selected_sigma),
        "warm_profile_npz": str(warm_profile_path),
        "aligned_inlet_window": {
            "np_in": {"guess": n_p_in_guess, "min": np_in_min, "max": np_in_max},
            "te_in": {"guess": T_e_in_guess, "min": te_in_min, "max": te_in_max},
            "z_in": {"guess": Z_in_guess, "min": z_in_min, "max": z_in_max},
            "jx_in": {"guess": J_x_in_guess, "min": jx_in_min, "max": jx_in_max},
            "seed_fraction": {
                "guess": seed_fraction_guess,
                "min": seed_fraction_min,
                "max": seed_fraction_max,
            },
        },
    }

    if args.out_json:
        out_json_path = Path(args.out_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        summary_path = out_dir / "continuation_summary.json"
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "schedule_path": str(schedule_path),
                "ok": bool(payload.get("ok", False)),
                "last_trusted_stage": payload.get("last_trusted_stage"),
                "first_failed_stage": payload.get("first_failed_stage"),
                "final_trusted_inlet_design": payload.get("final_trusted_inlet_design"),
                "final_trusted_objective_delta_Te_K": payload.get("final_trusted_objective_delta_Te_K"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
