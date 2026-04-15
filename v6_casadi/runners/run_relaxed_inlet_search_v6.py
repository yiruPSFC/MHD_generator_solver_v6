#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_CASADI_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_casadi.run_casadi_continuation_v6 import run_continuation
from v6_casadi.optimize_area_profile_casadi_v6 import (
    _make_stage_function,
    _prepare_inlet_constants,
)
from v6_casadi.runners.run_relaxed_continuation_v6 import _relaxed_stage_schedule
from v6_global_marginal.global_postprocess_v6 import (
    design_value_weights_lab_poc,
    evaluate_design_value,
)
from v6_global_marginal.pde_solver_v6_batch_global import (
    ForwardPDESolverV6BatchGlobal,
    event_name_from_code,
)


SCENARIO_PRESETS: dict[str, dict[str, float | str]] = {
    "coal_mhd_htah": {
        "tp_in_min": 300.0,
        "tp_in_max": 1900.0,
        "description": "coal-fired MHD topping with a dedicated high-temperature air heater",
    },
    "model_unconstrained": {
        "tp_in_min": 300.0,
        "tp_in_max": 3000.0,
        "description": "broad single-segment model sweep with a weak engineering cap on inlet Tp",
    },
    "lab_hot_gas": {
        "tp_in_min": 300.0,
        "tp_in_max": 2200.0,
        "description": "laboratory hot-gas inlet without a dedicated plasma-torch upper-bound case",
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Search inlet conditions for the single-segment relaxed CasADi workflow. "
            "The outer loop varies inlet state parameters; the inner loop runs the "
            "relaxed continuation schedule."
        )
    )
    p.add_argument(
        "--scenario",
        type=str,
        choices=tuple(SCENARIO_PRESETS) + ("custom",),
        default="coal_mhd_htah",
        help=(
            "preset inlet-Tp search scenario; use custom to disable presets or pass "
            "--tp-in-min/--tp-in-max explicitly to override the preset bounds"
        ),
    )
    p.add_argument("--B", type=float, default=10.2)
    p.add_argument("--L", type=float, default=1.0)
    p.add_argument("--seed-fraction", type=float, default=None)
    p.add_argument("--warm-start-dx", type=float, default=0.01)
    p.add_argument("--n-intervals", type=int, default=80)
    p.add_argument(
        "--schedule-profile",
        type=str,
        choices=("conservative", "balanced", "aggressive"),
        default="balanced",
    )
    p.add_argument("--sample-count", type=int, default=24)
    p.add_argument("--eval-count", type=int, default=8)
    p.add_argument("--random-seed", type=int, default=7)
    p.add_argument("--np-in-min", type=float, default=1.0e24)
    p.add_argument("--np-in-max", type=float, default=8.0e25)
    p.add_argument("--z-in-min", type=float, default=30.0)
    p.add_argument("--z-in-max", type=float, default=110.0)
    p.add_argument("--tp-in-min", type=float, default=300.0)
    p.add_argument("--tp-in-max", type=float, default=1200.0)
    p.add_argument(
        "--inlet-delta-ratio-min",
        type=float,
        default=0.05,
        help="sample Te_in via Te_in = Tp_in * (1 + inlet_delta_ratio)",
    )
    p.add_argument("--inlet-delta-ratio-max", type=float, default=4.0)
    p.add_argument("--A-in-min", type=float, default=0.10)
    p.add_argument("--A-in-max", type=float, default=0.80)
    p.add_argument(
        "--prefilter-weight-inlet-delta-ratio",
        type=float,
        default=1.0,
        help="cheap ranking weight before running continuation",
    )
    p.add_argument("--prefilter-weight-inlet-mach", type=float, default=0.75)
    p.add_argument("--prefilter-weight-inlet-f-ion", type=float, default=0.75)
    p.add_argument(
        "--score-weight-delta-ratio-uplift",
        type=float,
        default=1.0,
        help="extra reward on (outlet_delta_ratio - inlet_delta_ratio)",
    )
    p.add_argument(
        "--score-penalty-inlet-f-ion",
        type=float,
        default=0.5,
        help="extra penalty on inlet ionization fraction in addition to lab_poc score",
    )
    p.add_argument(
        "--adaptive-bridge-count",
        type=int,
        default=3,
    )
    p.add_argument(
        "--adaptive-bridge-max-count",
        type=int,
        default=12,
    )
    p.add_argument(
        "--transition-margin-slack-max",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--transition-margin-slack-weight",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--final-mach-min",
        type=float,
        default=1.0,
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
        default=str(_CASADI_DIR / "outputs" / "continuation" / "relaxed_inlet_search"),
    )
    p.add_argument("--out-json", type=str, default="")
    return p


def _arg_was_provided(argv: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in argv)


def _apply_scenario_preset(args, *, argv: list[str]) -> dict[str, object]:
    scenario_name = str(args.scenario)
    if scenario_name == "custom":
        return {
            "name": "custom",
            "description": "manual inlet-Tp bounds from explicit command-line arguments",
            "tp_in_min": float(args.tp_in_min),
            "tp_in_max": float(args.tp_in_max),
            "tp_in_min_source": "explicit_or_parser_default",
            "tp_in_max_source": "explicit_or_parser_default",
        }

    preset = dict(SCENARIO_PRESETS[scenario_name])
    tp_min_flag = _arg_was_provided(argv, "--tp-in-min")
    tp_max_flag = _arg_was_provided(argv, "--tp-in-max")
    if not tp_min_flag:
        args.tp_in_min = float(preset["tp_in_min"])
    if not tp_max_flag:
        args.tp_in_max = float(preset["tp_in_max"])
    return {
        "name": scenario_name,
        "description": str(preset["description"]),
        "tp_in_min": float(args.tp_in_min),
        "tp_in_max": float(args.tp_in_max),
        "tp_in_min_source": "cli_override" if tp_min_flag else "scenario_preset",
        "tp_in_max_source": "cli_override" if tp_max_flag else "scenario_preset",
    }


def _safe_ratio(num: float, den: float, *, floor: float = 1e-30) -> float:
    if (not math.isfinite(num)) or (not math.isfinite(den)):
        return float("nan")
    den_safe = den if abs(den) >= floor else (floor if den >= 0.0 else -floor)
    return float(num / den_safe)


def _log_uniform(rng: np.random.Generator, lo: float, hi: float, size: int) -> np.ndarray:
    if lo <= 0.0 or hi <= 0.0:
        raise ValueError("log-uniform bounds must be positive.")
    if hi < lo:
        raise ValueError("invalid log-uniform bounds.")
    if hi == lo:
        return np.full(size, lo, dtype=float)
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))


def _uniform(rng: np.random.Generator, lo: float, hi: float, size: int) -> np.ndarray:
    if hi < lo:
        raise ValueError("invalid uniform bounds.")
    if hi == lo:
        return np.full(size, lo, dtype=float)
    return rng.uniform(lo, hi, size=size)


def _sample_inlets(args, *, rng: np.random.Generator) -> list[dict[str, float]]:
    n = max(1, int(args.sample_count))
    n_p = _log_uniform(rng, float(args.np_in_min), float(args.np_in_max), n)
    Z = _uniform(rng, float(args.z_in_min), float(args.z_in_max), n)
    T_p = _uniform(rng, float(args.tp_in_min), float(args.tp_in_max), n)
    delta = _uniform(
        rng,
        float(args.inlet_delta_ratio_min),
        float(args.inlet_delta_ratio_max),
        n,
    )
    A = _uniform(rng, float(args.A_in_min), float(args.A_in_max), n)

    out: list[dict[str, float]] = []
    for i in range(n):
        tp = float(T_p[i])
        inlet_delta_ratio = float(delta[i])
        out.append(
            {
                "candidate_index": int(i),
                "n_p_in": float(n_p[i]),
                "Z_in": float(Z[i]),
                "T_p_in": tp,
                "T_e_in": float(tp * (1.0 + inlet_delta_ratio)),
                "A_in": float(A[i]),
                "inlet_delta_ratio_target": inlet_delta_ratio,
            }
        )
    return out


def _prefilter_candidates(args, *, candidates: list[dict[str, float]]) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    if args.seed_fraction is None:
        solver = ForwardPDESolverV6BatchGlobal(B=float(args.B), length=float(args.L))
        inlet = solver.evaluate_inlet_batch(
            n_p_in=np.array([c["n_p_in"] for c in candidates], dtype=float),
            Z_in=np.array([c["Z_in"] for c in candidates], dtype=float),
            T_p_in=np.array([c["T_p_in"] for c in candidates], dtype=float),
            T_e_in=np.array([c["T_e_in"] for c in candidates], dtype=float),
            A_in=np.array([c["A_in"] for c in candidates], dtype=float),
        )

        for i, candidate in enumerate(candidates):
            seed_fraction = float(inlet.seed_fraction[i])
            inlet_f_ion = _safe_ratio(
                float(inlet.n_e[i]),
                seed_fraction * float(inlet.n_p[i]),
            )
            inlet_delta_ratio = float(inlet.T_e[i] / max(float(inlet.T_p[i]), 1e-30) - 1.0)
            prefilter_score = -(
                float(args.prefilter_weight_inlet_delta_ratio) * inlet_delta_ratio
                + float(args.prefilter_weight_inlet_mach) * float(inlet.mach[i])
                + float(args.prefilter_weight_inlet_f_ion) * inlet_f_ion
            )
            ranked.append(
                {
                    "candidate": dict(candidate),
                    "inlet_success": bool(inlet.success[i]),
                    "inlet_event_code": int(inlet.event_code[i]),
                    "inlet_event_name": event_name_from_code(int(inlet.event_code[i])),
                    "seed_fraction": seed_fraction,
                    "inlet_metrics": {
                        "mach": float(inlet.mach[i]),
                        "inlet_delta_ratio": inlet_delta_ratio,
                        "inlet_f_ion": inlet_f_ion,
                        "velikhov_margin": float(inlet.inlet_velikhov_margin[i]),
                        "T_p_effective_K": float(inlet.T_p[i]),
                        "J_x_in_A_m2": float(inlet.J_x[i]),
                        "E_x_in_V_m": float(inlet.E_x[i]),
                        "dTe_rel_grad_1_m": float(inlet.dTe_rel_grad[i]),
                        "dA_rel_grad_1_m": float(inlet.dA_rel_grad[i]),
                    },
                    "prefilter_score": float(prefilter_score),
                }
            )
    else:
        for candidate in candidates:
            try:
                inlet = _prepare_inlet_constants(
                    n_p_in=float(candidate["n_p_in"]),
                    Z_in=float(candidate["Z_in"]),
                    T_p_in=float(candidate["T_p_in"]),
                    T_e_in=float(candidate["T_e_in"]),
                    A_in=float(candidate["A_in"]),
                    B=float(args.B),
                    seed_fraction=float(args.seed_fraction),
                )
                stage = _make_stage_function(
                    dot_N=float(inlet.dot_N),
                    I_0=float(inlet.I_0),
                    seed_fraction=float(inlet.seed_fraction),
                    B=float(args.B),
                )
                state = np.array(
                    [
                        float(candidate["n_p_in"]),
                        float(candidate["T_e_in"]),
                        float(candidate["A_in"]),
                    ],
                    dtype=float,
                )
                out = np.asarray(stage(state, 0.0), dtype=float).reshape(-1)
                inlet_success = bool(np.all(np.isfinite(out)))
                inlet_event_code = 0 if inlet_success else -1
                inlet_event_name = "EVENT_NONE" if inlet_success else "NON_FINITE_STAGE_EVAL"
                tp_effective = float(out[3]) if inlet_success else float("nan")
                mach = float(out[12]) if inlet_success else float("nan")
                velikhov_margin = float(out[13]) if inlet_success else float("nan")
                n_e = float(out[5]) if inlet_success else float("nan")
                j_x = float(out[9]) if inlet_success else float("nan")
                e_x = float(out[11]) if inlet_success else float("nan")
                inlet_delta_ratio = (
                    float(candidate["T_e_in"] / max(tp_effective, 1e-30) - 1.0)
                    if inlet_success
                    else float("nan")
                )
                inlet_f_ion = _safe_ratio(
                    n_e,
                    float(inlet.seed_fraction) * float(candidate["n_p_in"]),
                )
                prefilter_score = -(
                    float(args.prefilter_weight_inlet_delta_ratio) * inlet_delta_ratio
                    + float(args.prefilter_weight_inlet_mach) * mach
                    + float(args.prefilter_weight_inlet_f_ion) * inlet_f_ion
                ) if inlet_success else float("-inf")
                ranked.append(
                    {
                        "candidate": dict(candidate),
                        "inlet_success": inlet_success,
                        "inlet_event_code": int(inlet_event_code),
                        "inlet_event_name": str(inlet_event_name),
                        "seed_fraction": float(inlet.seed_fraction),
                        "inlet_metrics": {
                            "mach": mach,
                            "inlet_delta_ratio": inlet_delta_ratio,
                            "inlet_f_ion": inlet_f_ion,
                            "velikhov_margin": velikhov_margin,
                            "T_p_effective_K": tp_effective,
                            "J_x_in_A_m2": j_x,
                            "E_x_in_V_m": e_x,
                            "dTe_rel_grad_1_m": float("nan"),
                            "dA_rel_grad_1_m": float("nan"),
                        },
                        "prefilter_score": float(prefilter_score),
                    }
                )
            except ValueError as exc:
                ranked.append(
                    {
                        "candidate": dict(candidate),
                        "inlet_success": False,
                        "inlet_event_code": -1,
                        "inlet_event_name": str(exc),
                        "seed_fraction": float(args.seed_fraction),
                        "inlet_metrics": {
                            "mach": float("nan"),
                            "inlet_delta_ratio": float("nan"),
                            "inlet_f_ion": float("nan"),
                            "velikhov_margin": float("nan"),
                            "T_p_effective_K": float("nan"),
                            "J_x_in_A_m2": float("nan"),
                            "E_x_in_V_m": float("nan"),
                            "dTe_rel_grad_1_m": float("nan"),
                            "dA_rel_grad_1_m": float("nan"),
                        },
                        "prefilter_score": float("-inf"),
                    }
                )
    ranked.sort(
        key=lambda item: (
            0 if item["inlet_success"] else 1,
            -float(item["prefilter_score"]),
        )
    )
    return ranked


def _search_score(args, *, final_stage_artifacts: dict[str, object], inlet_metrics: dict[str, float]) -> dict[str, object]:
    value_terms = dict(final_stage_artifacts["value_terms"])
    lab_poc = dict(final_stage_artifacts["value_profiles"]["lab_poc"])
    outlet_delta_ratio = float(value_terms["outlet_delta_ratio"])
    inlet_delta_ratio = float(inlet_metrics["inlet_delta_ratio"])
    inlet_f_ion = float(inlet_metrics["inlet_f_ion"])
    delta_ratio_uplift = outlet_delta_ratio - inlet_delta_ratio
    contributions = {
        "base_lab_poc_score": float(lab_poc["total_score"]),
        "reward_delta_ratio_uplift": float(args.score_weight_delta_ratio_uplift) * delta_ratio_uplift,
        "penalty_inlet_f_ion": -float(args.score_penalty_inlet_f_ion) * inlet_f_ion,
    }
    return {
        "total_score": float(sum(contributions.values())),
        "contributions": contributions,
        "outlet_delta_ratio": outlet_delta_ratio,
        "inlet_delta_ratio": inlet_delta_ratio,
        "delta_ratio_uplift": delta_ratio_uplift,
        "inlet_f_ion": inlet_f_ion,
        "base_lab_poc_score": float(lab_poc["total_score"]),
    }


def _build_schedule(args) -> list[dict]:
    final_mach_min = None if float(args.final_mach_min) < 0.0 else float(args.final_mach_min)
    return _relaxed_stage_schedule(
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
        include_stage1_assess=False,
        transition_margin_slack_max=float(args.transition_margin_slack_max),
        transition_margin_slack_weight=float(args.transition_margin_slack_weight),
    )


def _run_candidate(args, *, ranked_item: dict[str, object], schedule: list[dict], out_dir: Path) -> dict[str, object]:
    candidate = dict(ranked_item["candidate"])
    candidate_dir = out_dir / f"candidate_{int(candidate['candidate_index']):03d}"
    payload = run_continuation(
        n_p_in=float(candidate["n_p_in"]),
        Z_in=float(candidate["Z_in"]),
        T_p_in=float(candidate["T_p_in"]),
        T_e_in=float(candidate["T_e_in"]),
        A_in=float(candidate["A_in"]),
        B=float(args.B),
        L=float(args.L),
        seed_fraction=None if args.seed_fraction is None else float(args.seed_fraction),
        warm_start_dx=float(args.warm_start_dx),
        stage_schedule=schedule,
        out_dir=candidate_dir,
        stop_on_unacceptable=False,
        warm_start_policy="regular",
        adaptive_bridge_count=int(args.adaptive_bridge_count),
        adaptive_bridge_max_count=int(args.adaptive_bridge_max_count),
    )

    record: dict[str, object] = {
        "candidate": candidate,
        "inlet_success": bool(ranked_item["inlet_success"]),
        "inlet_event_code": int(ranked_item["inlet_event_code"]),
        "inlet_event_name": str(ranked_item["inlet_event_name"]),
        "seed_fraction": float(ranked_item["seed_fraction"]),
        "inlet_metrics": dict(ranked_item["inlet_metrics"]),
        "prefilter_score": float(ranked_item["prefilter_score"]),
        "continuation_summary_path": str(candidate_dir / "continuation_summary.json"),
        "candidate_dir": str(candidate_dir),
        "continuation": payload,
    }

    if payload.get("stages"):
        stages = [dict(stage) for stage in payload["stages"]]
        final_stage = dict(stages[-1])
        record["final_stage"] = final_stage
        best_stage = None
        best_score = None
        for stage in stages:
            artifacts = stage.get("artifacts")
            if (not bool(stage.get("acceptable", False))) or (not isinstance(artifacts, dict)):
                continue
            score = _search_score(
                args,
                final_stage_artifacts=artifacts,
                inlet_metrics=dict(ranked_item["inlet_metrics"]),
            )
            if (best_score is None) or (float(score["total_score"]) > float(best_score["total_score"])):
                best_stage = dict(stage)
                best_score = dict(score)
        record["best_acceptable_stage"] = best_stage
        record["search_score"] = best_score
    return record


def main() -> int:
    args = _build_parser().parse_args()
    scenario_info = _apply_scenario_preset(args, argv=sys.argv[1:])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.random_seed))
    candidates = _sample_inlets(args, rng=rng)
    ranked = _prefilter_candidates(args, candidates=candidates)
    feasible_ranked = [item for item in ranked if bool(item["inlet_success"])]
    eval_count = min(max(0, int(args.eval_count)), len(feasible_ranked))
    schedule = _build_schedule(args)

    evaluated: list[dict[str, object]] = []
    for ranked_item in feasible_ranked[:eval_count]:
        evaluated.append(
            _run_candidate(
                args,
                ranked_item=ranked_item,
                schedule=schedule,
                out_dir=out_dir,
            )
        )

    scored = [item for item in evaluated if isinstance(item.get("search_score"), dict)]
    scored.sort(
        key=lambda item: float(item["search_score"]["total_score"]),
        reverse=True,
    )

    summary = {
        "search_goal": (
            "maximize single-segment Te/Tp uplift under relaxed inlet search "
            "with penalties on low-experiment-feasibility inlet conditions"
        ),
        "B_T": float(args.B),
        "L_m": float(args.L),
        "scenario": scenario_info,
        "schedule_profile": str(args.schedule_profile),
        "sample_count": int(args.sample_count),
        "eval_count": int(eval_count),
        "seed_fraction_mode": "specified" if args.seed_fraction is not None else "projected_marginal",
        "schedule": schedule,
        "prefilter_ranked_candidates": ranked,
        "evaluated_candidates": evaluated,
        "best_candidate": None if not scored else scored[0],
    }

    if args.out_json:
        out_path = Path(args.out_json)
    else:
        out_path = out_dir / "search_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
