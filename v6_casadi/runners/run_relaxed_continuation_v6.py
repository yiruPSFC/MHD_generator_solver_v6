#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
_CASADI_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_casadi.run_casadi_continuation_v6 import run_continuation


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run a progressively relaxed continuation schedule for the post-Jeffrey "
            "NLP workflow; suitable for exploring more arbitrary nozzle curves"
        )
    )
    p.add_argument("--np-in", type=float, default=3.05e25)
    p.add_argument("--z-in", type=float, default=75.954994)
    p.add_argument("--tp-in", type=float, default=429.0)
    p.add_argument("--te-in", type=float, default=4420.0)
    p.add_argument("--A-in", type=float, default=0.447)
    p.add_argument("--B", type=float, default=10.2)
    p.add_argument("--L", type=float, default=5.4)
    p.add_argument("--seed-fraction", type=float, default=None)
    p.add_argument("--warm-start-dx", type=float, default=0.01)
    p.add_argument("--n-intervals", type=int, default=80)
    p.add_argument(
        "--schedule-profile",
        type=str,
        choices=("conservative", "balanced", "aggressive"),
        default="balanced",
        help="built-in relaxation profile used when --schedule-json is not provided",
    )
    p.add_argument(
        "--schedule-json",
        type=str,
        default="",
        help="optional json file (list of stage dicts) to fully override the built-in schedule",
    )
    p.add_argument(
        "--stop-on-unacceptable",
        action="store_true",
        help="stop at the first unacceptable stage (default is to keep relaxing)",
    )
    p.add_argument(
        "--include-stage1-assess",
        action="store_true",
        help="include an optional zero-objective assessment stage before the main HS continuation",
    )
    p.add_argument(
        "--adaptive-bridge-count",
        type=int,
        default=3,
        help="number of interpolated bridge stages inserted between stable and target relaxed stages",
    )
    p.add_argument(
        "--adaptive-bridge-max-count",
        type=int,
        default=12,
        help="maximum bridge-stage count after automatic doubling retries",
    )
    p.add_argument(
        "--transition-margin-slack-max",
        type=float,
        default=0.0,
        help="optional slack cap applied to non-final relaxed stages for G >= 0 continuation",
    )
    p.add_argument(
        "--transition-margin-slack-weight",
        type=float,
        default=0.0,
        help="L1 penalty weight used with --transition-margin-slack-max",
    )
    p.add_argument(
        "--final-mach-min",
        type=float,
        default=1.0,
        help="final-stage Mach lower bound; pass <0 to disable it",
    )
    p.add_argument("--final-a-min-ratio", type=float, default=0.40)
    p.add_argument("--final-a-max-ratio", type=float, default=5.0)
    p.add_argument("--final-max-abs-dlogA-dx", type=float, default=0.50)
    p.add_argument("--final-np-min-ratio", type=float, default=1e-8)
    p.add_argument("--final-np-max-ratio", type=float, default=150.0)
    p.add_argument("--final-te-min", type=float, default=100.0)
    p.add_argument("--final-te-max-ratio", type=float, default=24.0)
    p.add_argument("--final-smooth-weight", type=float, default=0.004)
    p.add_argument("--final-control-slew-weight", type=float, default=0.06)
    p.add_argument("--final-control-curvature-weight", type=float, default=0.015)
    p.add_argument("--final-state-curvature-weight", type=float, default=0.003)
    p.add_argument("--final-warm-profile-track-weight", type=float, default=1.0)
    p.add_argument("--final-warm-control-track-weight", type=float, default=0.25)
    p.add_argument("--final-objective-weight", type=float, default=0.08)
    p.add_argument("--final-ipopt-max-iter", type=int, default=2600)
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(_CASADI_DIR / "outputs" / "continuation" / "relaxed_reference_case"),
        help="directory for the relaxed continuation artifacts",
    )
    p.add_argument("--out-json", type=str, default="")
    return p


def _stage(
    name: str,
    *,
    n_intervals: int,
    transcription: str,
    mach_min: float | None,
    A_min_ratio: float,
    A_max_ratio: float,
    max_abs_dlogA_dx: float,
    np_min_ratio: float,
    np_max_ratio: float,
    te_min: float,
    te_max_ratio: float,
    margin_slack_max: float = 0.0,
    margin_slack_weight: float = 0.0,
    smooth_weight: float,
    control_slew_weight: float,
    control_curvature_weight: float,
    state_curvature_weight: float,
    warm_profile_track_weight: float,
    warm_control_track_weight: float,
    objective_weight: float,
    ipopt_max_iter: int,
) -> dict:
    return {
        "name": name,
        "n_intervals": n_intervals,
        "transcription": transcription,
        "min_margin": 0.0,
        "mach_min": mach_min,
        "A_min_ratio": A_min_ratio,
        "A_max_ratio": A_max_ratio,
        "max_abs_dlogA_dx": max_abs_dlogA_dx,
        "np_min_ratio": np_min_ratio,
        "np_max_ratio": np_max_ratio,
        "te_min": te_min,
        "te_max_ratio": te_max_ratio,
        "margin_slack_max": margin_slack_max,
        "margin_slack_weight": margin_slack_weight,
        "smooth_weight": smooth_weight,
        "control_slew_weight": control_slew_weight,
        "control_curvature_weight": control_curvature_weight,
        "state_curvature_weight": state_curvature_weight,
        "warm_profile_track_weight": warm_profile_track_weight,
        "warm_control_track_weight": warm_control_track_weight,
        "objective_weight": objective_weight,
        "ipopt_max_iter": ipopt_max_iter,
    }


def _profile_prefix_stages(
    *,
    profile: str,
    n_intervals: int,
    include_stage1_assess: bool = False,
    transition_margin_slack_max: float = 0.0,
    transition_margin_slack_weight: float = 0.0,
) -> list[dict]:
    if n_intervals < 2:
        raise ValueError("n_intervals must be at least 2.")
    assess_stage = _stage(
        "stage_1_reference_assess",
        n_intervals=n_intervals,
        transcription="hermite-simpson",
        mach_min=1.4,
        A_min_ratio=0.95,
        A_max_ratio=2.1,
        max_abs_dlogA_dx=0.20,
        np_min_ratio=1e-4,
        np_max_ratio=30.0,
        te_min=100.0,
        te_max_ratio=6.0,
        smooth_weight=0.0,
        control_slew_weight=0.0,
        control_curvature_weight=0.0,
        state_curvature_weight=0.0,
        warm_profile_track_weight=0.0,
        warm_control_track_weight=0.0,
        objective_weight=0.0,
        ipopt_max_iter=0,
    )

    if profile == "conservative":
        stages = [
            _stage(
                "stage_2_hs_anchor",
                n_intervals=n_intervals,
                transcription="hermite-simpson",
                mach_min=1.36,
                A_min_ratio=0.95,
                A_max_ratio=2.12,
                max_abs_dlogA_dx=0.17,
                np_min_ratio=1e-4,
                np_max_ratio=26.0,
                te_min=100.0,
                te_max_ratio=6.2,
                margin_slack_max=transition_margin_slack_max,
                margin_slack_weight=transition_margin_slack_weight,
                smooth_weight=0.01,
                control_slew_weight=0.70,
                control_curvature_weight=0.22,
                state_curvature_weight=0.028,
                warm_profile_track_weight=48.0,
                warm_control_track_weight=13.0,
                objective_weight=0.002,
                ipopt_max_iter=1500,
            ),
            _stage(
                "stage_3_hs_release",
                n_intervals=n_intervals,
                transcription="hermite-simpson",
                mach_min=1.22,
                A_min_ratio=0.72,
                A_max_ratio=3.00,
                max_abs_dlogA_dx=0.28,
                np_min_ratio=1e-5,
                np_max_ratio=60.0,
                te_min=100.0,
                te_max_ratio=10.5,
                margin_slack_max=transition_margin_slack_max,
                margin_slack_weight=transition_margin_slack_weight,
                smooth_weight=0.006,
                control_slew_weight=0.20,
                control_curvature_weight=0.055,
                state_curvature_weight=0.010,
                warm_profile_track_weight=7.0,
                warm_control_track_weight=2.0,
                objective_weight=0.014,
                ipopt_max_iter=2100,
            ),
        ]
        return ([assess_stage] + stages) if include_stage1_assess else stages

    if profile == "aggressive":
        stages = [
            _stage(
                "stage_2_hs_anchor",
                n_intervals=n_intervals,
                transcription="hermite-simpson",
                mach_min=1.32,
                A_min_ratio=0.93,
                A_max_ratio=2.2,
                max_abs_dlogA_dx=0.20,
                np_min_ratio=5e-5,
                np_max_ratio=35.0,
                te_min=100.0,
                te_max_ratio=7.0,
                margin_slack_max=transition_margin_slack_max,
                margin_slack_weight=transition_margin_slack_weight,
                smooth_weight=0.01,
                control_slew_weight=0.35,
                control_curvature_weight=0.10,
                state_curvature_weight=0.015,
                warm_profile_track_weight=18.0,
                warm_control_track_weight=5.0,
                objective_weight=0.006,
                ipopt_max_iter=1600,
            ),
            _stage(
                "stage_3_hs_release",
                n_intervals=n_intervals,
                transcription="hermite-simpson",
                mach_min=1.08,
                A_min_ratio=0.58,
                A_max_ratio=3.60,
                max_abs_dlogA_dx=0.36,
                np_min_ratio=1e-5,
                np_max_ratio=95.0,
                te_min=100.0,
                te_max_ratio=14.0,
                margin_slack_max=transition_margin_slack_max,
                margin_slack_weight=transition_margin_slack_weight,
                smooth_weight=0.004,
                control_slew_weight=0.10,
                control_curvature_weight=0.028,
                state_curvature_weight=0.005,
                warm_profile_track_weight=2.5,
                warm_control_track_weight=0.7,
                objective_weight=0.030,
                ipopt_max_iter=2200,
            ),
        ]
        return ([assess_stage] + stages) if include_stage1_assess else stages

    stages = [
        _stage(
            "stage_2_hs_anchor",
            n_intervals=n_intervals,
            transcription="hermite-simpson",
            mach_min=1.35,
            A_min_ratio=0.95,
            A_max_ratio=2.12,
            max_abs_dlogA_dx=0.17,
            np_min_ratio=1e-4,
            np_max_ratio=28.0,
            te_min=100.0,
            te_max_ratio=6.4,
            margin_slack_max=transition_margin_slack_max,
            margin_slack_weight=transition_margin_slack_weight,
            smooth_weight=0.01,
            control_slew_weight=0.60,
            control_curvature_weight=0.18,
            state_curvature_weight=0.025,
            warm_profile_track_weight=32.0,
            warm_control_track_weight=9.0,
            objective_weight=0.0035,
            ipopt_max_iter=1600,
        ),
        _stage(
            "stage_3_hs_release",
            n_intervals=n_intervals,
            transcription="hermite-simpson",
            mach_min=1.12,
            A_min_ratio=0.60,
            A_max_ratio=3.25,
            max_abs_dlogA_dx=0.32,
            np_min_ratio=8e-6,
            np_max_ratio=75.0,
            te_min=100.0,
            te_max_ratio=12.0,
            margin_slack_max=transition_margin_slack_max,
            margin_slack_weight=transition_margin_slack_weight,
            smooth_weight=0.005,
            control_slew_weight=0.12,
            control_curvature_weight=0.035,
            state_curvature_weight=0.006,
            warm_profile_track_weight=4.0,
            warm_control_track_weight=1.0,
            objective_weight=0.024,
            ipopt_max_iter=2100,
        ),
    ]
    return ([assess_stage] + stages) if include_stage1_assess else stages


def _relaxed_stage_schedule(
    *,
    profile: str,
    n_intervals: int,
    final_mach_min: float | None,
    final_a_min_ratio: float,
    final_a_max_ratio: float,
    final_max_abs_dlogA_dx: float,
    final_np_min_ratio: float,
    final_np_max_ratio: float,
    final_te_min: float,
    final_te_max_ratio: float,
    final_smooth_weight: float,
    final_control_slew_weight: float,
    final_control_curvature_weight: float,
    final_state_curvature_weight: float,
    final_warm_profile_track_weight: float,
    final_warm_control_track_weight: float,
    final_objective_weight: float,
    final_ipopt_max_iter: int,
    include_stage1_assess: bool,
    transition_margin_slack_max: float,
    transition_margin_slack_weight: float,
) -> list[dict]:
    stages = _profile_prefix_stages(
        profile=profile,
        n_intervals=n_intervals,
        include_stage1_assess=include_stage1_assess,
        transition_margin_slack_max=transition_margin_slack_max,
        transition_margin_slack_weight=transition_margin_slack_weight,
    )
    stages.append(
        {
            "name": f"stage_{len(stages) + 2}_arbitrary_final",
            "n_intervals": n_intervals,
            "transcription": "hermite-simpson",
            "min_margin": 0.0,
            "mach_min": final_mach_min,
            "A_min_ratio": final_a_min_ratio,
            "A_max_ratio": final_a_max_ratio,
            "max_abs_dlogA_dx": final_max_abs_dlogA_dx,
            "np_min_ratio": final_np_min_ratio,
            "np_max_ratio": final_np_max_ratio,
            "te_min": final_te_min,
            "te_max_ratio": final_te_max_ratio,
            "smooth_weight": final_smooth_weight,
            "control_slew_weight": final_control_slew_weight,
            "control_curvature_weight": final_control_curvature_weight,
            "state_curvature_weight": final_state_curvature_weight,
            "warm_profile_track_weight": final_warm_profile_track_weight,
            "warm_control_track_weight": final_warm_control_track_weight,
            "objective_weight": final_objective_weight,
            "ipopt_max_iter": final_ipopt_max_iter,
        }
    )
    return stages


def _load_schedule_json(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("schedule json must be a non-empty list of stage dicts.")
    stages: list[dict] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"schedule[{i}] must be a dict.")
        if not isinstance(item.get("name"), str) or not item["name"]:
            raise ValueError(f"schedule[{i}] must include a non-empty 'name' string.")
        stages.append(dict(item))
    return stages


def main() -> int:
    args = _build_parser().parse_args()
    final_mach_min = None if float(args.final_mach_min) < 0.0 else float(args.final_mach_min)
    if args.schedule_json:
        schedule_path = Path(args.schedule_json)
        schedule = _load_schedule_json(schedule_path)
        schedule_source = f"json:{schedule_path}"
        schedule_profile = "custom"
    else:
        schedule = _relaxed_stage_schedule(
            profile=str(args.schedule_profile),
            n_intervals=int(args.n_intervals),
            final_mach_min=final_mach_min,
            final_a_min_ratio=float(args.final_a_min_ratio),
            final_a_max_ratio=float(args.final_a_max_ratio),
            final_max_abs_dlogA_dx=float(args.final_max_abs_dlogA_dx),
            final_np_min_ratio=float(args.final_np_min_ratio),
            final_np_max_ratio=float(args.final_np_max_ratio),
            final_te_min=float(args.final_te_min),
            final_te_max_ratio=float(args.final_te_max_ratio),
            final_smooth_weight=float(args.final_smooth_weight),
            final_control_slew_weight=float(args.final_control_slew_weight),
            final_control_curvature_weight=float(args.final_control_curvature_weight),
            final_state_curvature_weight=float(args.final_state_curvature_weight),
            final_warm_profile_track_weight=float(args.final_warm_profile_track_weight),
            final_warm_control_track_weight=float(args.final_warm_control_track_weight),
            final_objective_weight=float(args.final_objective_weight),
            final_ipopt_max_iter=int(args.final_ipopt_max_iter),
            include_stage1_assess=bool(args.include_stage1_assess),
            transition_margin_slack_max=float(args.transition_margin_slack_max),
            transition_margin_slack_weight=float(args.transition_margin_slack_weight),
        )
        schedule_source = "built-in"
        schedule_profile = str(args.schedule_profile)
    payload = run_continuation(
        n_p_in=float(args.np_in),
        Z_in=float(args.z_in),
        T_p_in=float(args.tp_in),
        T_e_in=float(args.te_in),
        A_in=float(args.A_in),
        B=float(args.B),
        L=float(args.L),
        seed_fraction=None if args.seed_fraction is None else float(args.seed_fraction),
        warm_start_dx=float(args.warm_start_dx),
        stage_schedule=schedule,
        out_dir=args.out_dir,
        out_json=args.out_json,
        stop_on_unacceptable=bool(args.stop_on_unacceptable),
        warm_start_policy="regular",
        adaptive_bridge_count=int(args.adaptive_bridge_count),
        adaptive_bridge_max_count=int(args.adaptive_bridge_max_count),
    )
    payload["schedule"] = schedule
    payload["schedule_source"] = schedule_source
    payload["schedule_profile"] = schedule_profile
    payload["stop_on_unacceptable"] = bool(args.stop_on_unacceptable)
    payload["adaptive_bridge_count"] = int(args.adaptive_bridge_count)
    payload["adaptive_bridge_max_count"] = int(args.adaptive_bridge_max_count)
    payload["transition_margin_slack_max"] = float(args.transition_margin_slack_max)
    payload["transition_margin_slack_weight"] = float(args.transition_margin_slack_weight)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "relaxed_schedule.json").write_text(
            json.dumps(schedule, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
