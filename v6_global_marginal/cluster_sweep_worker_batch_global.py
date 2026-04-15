#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_batch.cluster_sweep_worker_batch import (
    RewardConfig,
    _lin_map,
    _log_map,
    _score_profile,
    _unit_sample_blocks,
    _write_rows_atomic,
)
from v6_batch.pde_solver_v6_batch import EVENT_INLET_ERROR
from v6_global_marginal.pde_solver_v6_batch_global import (
    EVENT_NO_MARGINAL_SEED,
    EVENT_SEED_OUT_OF_RANGE,
    ForwardPDESolverV6BatchGlobal,
    event_name_from_code,
)

FIXED_A_IN = 1.0


@dataclass
class Candidate:
    n_p_in: float
    Z_in: float
    T_p_in: float
    T_e_in: float
    B: float


@dataclass
class EvalSummary:
    ok: bool
    feasible: bool
    reason: str
    objective: float
    outlet_te: float
    outlet_tp: float
    outlet_ratio: float
    min_velikhov_margin: float
    event: str | None
    params: Dict[str, float]
    metrics: Dict[str, float | bool | str] = field(default_factory=dict)


def _candidate_from_unit(
    u: np.ndarray,
    bounds: Dict[str, Tuple[float, float]],
    T_p_in: float,
    B: float,
) -> Candidate:
    return Candidate(
        n_p_in=_log_map(float(u[0]), *bounds["n_p_in"]),
        Z_in=_lin_map(float(u[1]), *bounds["Z_in"]),
        T_p_in=float(T_p_in),
        T_e_in=_lin_map(float(u[2]), *bounds["T_e_in"]),
        B=float(B),
    )


def _reward_config_from_args_global(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        mach_buffer=float(args.mach_buffer),
        truncation_tol=float(args.trunc_tol),
        te_growth_min=float(args.te_growth_min),
        te_growth_bonus_target=float(args.te_growth_bonus_target),
        te_growth_bonus_dominance=float(args.te_growth_bonus_dominance),
        front_length_frac=float(args.front_length_frac),
        front_fraction_max=float(args.front_fraction_max),
        te_step_jump_frac_max=float(args.te_step_jump_frac_max),
        z_step_jump_frac_max=float(args.z_step_jump_frac_max),
        diff2_ref_mult=float(args.diff2_ref_mult),
        w_stab=float(args.w_stab),
        w_trunc=float(args.w_trunc),
        w_mach=float(args.w_mach),
        w_te_growth=float(args.w_te_growth),
        w_te_step_jump=float(args.w_te_step_jump),
        w_z_step_jump=float(args.w_z_step_jump),
        w_front_fraction=float(args.w_front_fraction),
        w_diff2=float(args.w_diff2),
        w_feasible_te_growth_bonus=(
            None
            if args.w_feasible_te_growth_bonus is None
            else float(args.w_feasible_te_growth_bonus)
        ),
        w_feasible_mach_bonus=float(args.w_feasible_mach_bonus),
        w_feasible_margin_bonus=float(args.w_feasible_margin_bonus),
    )


def _apply_inlet_prefilter_global(
    solver: ForwardPDESolverV6BatchGlobal,
    chunk: List[Candidate],
    dte_rel_min: float | None,
    mach_inlet_min: float | None,
    check_inlet_velikhov: bool,
) -> tuple[List[Candidate], Dict[str, int]]:
    stats: Dict[str, int] = {
        "total": len(chunk),
        "kept": 0,
        "rejected": 0,
        "rejected_inlet_error": 0,
        "rejected_invalid_state": 0,
        "rejected_low_mach_inlet": 0,
        "rejected_inlet_unstable": 0,
        "rejected_low_dte_rel_grad": 0,
    }
    if not chunk:
        return [], stats

    metrics = solver.evaluate_inlet_batch(
        n_p_in=np.array([c.n_p_in for c in chunk], dtype=float),
        Z_in=np.array([c.Z_in for c in chunk], dtype=float),
        T_p_in=np.array([c.T_p_in for c in chunk], dtype=float),
        T_e_in=np.array([c.T_e_in for c in chunk], dtype=float),
        A_in=np.full(len(chunk), FIXED_A_IN, dtype=float),
        parallel=True,
    )

    kept: List[Candidate] = []
    for i, cand in enumerate(chunk):
        if not bool(metrics.success[i]):
            code = int(metrics.event_code[i])
            if code in (EVENT_INLET_ERROR, EVENT_NO_MARGINAL_SEED, EVENT_SEED_OUT_OF_RANGE):
                stats["rejected_inlet_error"] += 1
            else:
                stats["rejected_invalid_state"] += 1
            stats["rejected"] += 1
            continue

        if mach_inlet_min is not None and float(metrics.mach[i]) < mach_inlet_min:
            stats["rejected_low_mach_inlet"] += 1
            stats["rejected"] += 1
            continue

        if check_inlet_velikhov and float(metrics.inlet_velikhov_margin[i]) < 0.0:
            stats["rejected_inlet_unstable"] += 1
            stats["rejected"] += 1
            continue

        if dte_rel_min is not None and float(metrics.dTe_rel_grad[i]) < dte_rel_min:
            stats["rejected_low_dte_rel_grad"] += 1
            stats["rejected"] += 1
            continue

        kept.append(cand)
        stats["kept"] += 1

    return kept, stats


def _evaluate_chunk(
    solver: ForwardPDESolverV6BatchGlobal,
    chunk: List[Candidate],
    dx: float,
    velikhov_required: bool,
    reward_config: RewardConfig,
) -> List[EvalSummary]:
    out = solver.solve_batch(
            n_p_in=np.array([c.n_p_in for c in chunk], dtype=float),
            Z_in=np.array([c.Z_in for c in chunk], dtype=float),
            T_p_in=np.array([c.T_p_in for c in chunk], dtype=float),
            T_e_in=np.array([c.T_e_in for c in chunk], dtype=float),
            A_in=np.full(len(chunk), FIXED_A_IN, dtype=float),
            dx=dx,
            store_profiles=True,
        )

    rows: List[EvalSummary] = []
    for i, cand in enumerate(chunk):
        event_name = event_name_from_code(int(out.event_code[i]))
        if not bool(out.success[i]) or int(out.valid_points[i]) <= 0:
            rows.append(
                EvalSummary(
                    ok=False,
                    feasible=False,
                    reason=f"batch_failed:{event_name or 'unknown'}",
                    objective=1e30,
                    outlet_te=np.nan,
                    outlet_tp=np.nan,
                    outlet_ratio=np.nan,
                    min_velikhov_margin=np.nan,
                    event=event_name,
                    params=asdict(cand),
                    metrics={},
                )
            )
            continue

        idx_last = int(out.valid_points[i]) - 1
        scored = _score_profile(
            x=out.x[: idx_last + 1],
            T_e=out.T_e[i, : idx_last + 1],
            T_p=out.T_p[i, : idx_last + 1],
            Z=out.Z[i, : idx_last + 1],
            mach=out.mach[i, : idx_last + 1],
            beta=out.beta[i, : idx_last + 1],
            n_e=out.n_e[i, : idx_last + 1],
            n_p=out.n_p[i, : idx_last + 1],
            seed_fraction=float(out.seed_fraction[i]),
            length=float(solver.length),
            reached_end=bool(out.reached_end[i]),
            velikhov_required=velikhov_required,
            reward_config=reward_config,
        )
        rows.append(
            EvalSummary(
                ok=True,
                feasible=bool(scored["feasible"]),
                reason=str(scored["reason"]),
                objective=float(scored["objective"]),
                outlet_te=float(scored["outlet_te"]),
                outlet_tp=float(scored["outlet_tp"]),
                outlet_ratio=float(scored["outlet_ratio"]),
                min_velikhov_margin=float(scored["min_velikhov_margin"]),
                event=event_name,
                params=asdict(cand),
                metrics={
                    **dict(scored["metrics"]),
                    "projected_seed_fraction": float(out.seed_fraction[i]),
                    "outlet_area": float(out.A[i, idx_last]),
                    "area_ratio": float(out.A[i, idx_last] / out.A[i, 0]),
                },
            )
        )

    return rows


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="V6 global batch sweep worker with fixed inlet area A_in=1"
    )
    p.add_argument("--n-total", type=int, default=10000, help="total LHS points")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--top-k", type=int, default=200, help="keep best K results per shard")
    p.add_argument("--out", type=str, default="global_sweep_results.jsonl")

    p.add_argument("--B", type=float, default=10.2, help="magnetic field [T]")
    p.add_argument("--L-max", type=float, default=10.0, help="integration horizon [m]")
    p.add_argument("--tp-in", type=float, default=429.0, help="fixed inlet heavy-particle temperature [K]")
    p.add_argument("--dx", type=float, default=5e-3, help="RK4 step size [m]")
    p.add_argument("--batch-size", type=int, default=64, help="LHS block size")

    p.add_argument("--np-min", type=float, default=1e23, help="min inlet n_p [m^-3]")
    p.add_argument("--np-max", type=float, default=1e26, help="max inlet n_p [m^-3]")
    p.add_argument("--z-min", type=float, default=1.0, help="min inlet Hall loading Z")
    p.add_argument("--z-max", type=float, default=120.0, help="max inlet Hall loading Z")
    p.add_argument("--te-min", type=float, default=2500.0, help="min inlet T_e [K]")
    p.add_argument("--te-max", type=float, default=8000.0, help="max inlet T_e [K]")
    p.add_argument(
        "--mach-buffer",
        type=float,
        default=0.03,
        help="require min |Mach-1| to exceed this buffer",
    )
    p.add_argument(
        "--trunc-tol",
        type=float,
        default=0.01,
        help="allowed truncation fraction before penalizing",
    )
    p.add_argument(
        "--te-growth-min",
        type=float,
        default=0.01,
        help="require (Te_out-Te_in)/Te_in above this fraction",
    )
    p.add_argument(
        "--te-growth-bonus-target",
        type=float,
        default=0.10,
        help="relative Te growth fraction that defines one unit of feasible Te-growth bonus score",
    )
    p.add_argument(
        "--te-growth-bonus-dominance",
        type=float,
        default=10.0,
        help="auto-scale feasible Te-growth bonus when explicit weight is omitted",
    )
    p.add_argument(
        "--front-length-frac",
        type=float,
        default=0.05,
        help="front segment fraction for front-loaded heating check",
    )
    p.add_argument(
        "--front-fraction-max",
        type=float,
        default=0.2,
        help="max allowed heating fraction accumulated in front segment",
    )
    p.add_argument(
        "--te-step-jump-frac-max",
        type=float,
        default=0.1,
        help="max first-step Te jump as a fraction of full-profile Te excursion",
    )
    p.add_argument(
        "--z-step-jump-frac-max",
        type=float,
        default=0.1,
        help="max first-step Z jump as a fraction of full-profile Z excursion",
    )
    p.add_argument(
        "--diff2-ref-mult",
        type=float,
        default=50.0,
        help="max |diff2(Te)| relative to a profile-scale quadratic reference",
    )
    p.add_argument("--w-stab", type=float, default=200.0, help="weight on Velikhov instability")
    p.add_argument("--w-trunc", type=float, default=0.0, help="weight on early truncation")
    p.add_argument("--w-mach", type=float, default=50.0, help="weight on near-sonic penalty")
    p.add_argument("--w-te-growth", type=float, default=300.0, help="weight on insufficient Te growth")
    p.add_argument("--w-te-step-jump", type=float, default=300.0, help="weight on first-step Te jump")
    p.add_argument("--w-z-step-jump", type=float, default=200.0, help="weight on first-step Z jump")
    p.add_argument("--w-front-fraction", type=float, default=250.0, help="weight on front-loaded heating")
    p.add_argument("--w-diff2", type=float, default=250.0, help="weight on rough Te profile")
    p.add_argument(
        "--w-feasible-te-growth-bonus",
        type=float,
        default=None,
        help="bonus weight for feasible profiles with larger Te growth; auto-scaled when omitted",
    )
    p.add_argument("--w-feasible-mach-bonus", type=float, default=2.0, help="bonus weight for larger Mach buffer")
    p.add_argument(
        "--w-feasible-margin-bonus",
        type=float,
        default=0.2,
        help="bonus weight for larger positive Velikhov margin",
    )
    p.add_argument("--no-velikhov-check", action="store_true")
    p.add_argument(
        "--prefilter-dte-rel-min",
        type=float,
        default=None,
        help="minimum inlet (dTe/dx)/Te_in [1/m] for batch prefilter; default is 0.01/L_max unless disabled",
    )
    p.add_argument(
        "--no-prefilter-dte-rel",
        action="store_true",
        help="disable the inlet normalized-Te-growth prefilter",
    )
    p.add_argument(
        "--prefilter-mach-inlet-min",
        type=float,
        default=None,
        help="optional minimum inlet Mach for cheap prefilter",
    )
    p.add_argument(
        "--prefilter-check-inlet-velikhov",
        action="store_true",
        help="also reject candidates with negative inlet Velikhov margin in the cheap prefilter",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    reward_config = _reward_config_from_args_global(args)

    bounds = {
        "n_p_in": (float(args.np_min), float(args.np_max)),
        "Z_in": (float(args.z_min), float(args.z_max)),
        "T_e_in": (float(args.te_min), float(args.te_max)),
    }
    for key, (lo, hi) in bounds.items():
        if hi <= lo:
            raise ValueError(f"{key} bounds invalid: {lo}, {hi}")
        if key == "n_p_in" and lo <= 0.0:
            raise ValueError(f"{key} must be positive")
    solver = ForwardPDESolverV6BatchGlobal(B=float(args.B), length=float(args.L_max))
    dte_rel_min = None if args.no_prefilter_dte_rel else (
        float(args.prefilter_dte_rel_min)
        if args.prefilter_dte_rel_min is not None
        else (0.01 / float(args.L_max))
    )
    use_prefilter = (
        dte_rel_min is not None
        or args.prefilter_mach_inlet_min is not None
        or bool(args.prefilter_check_inlet_velikhov)
    )
    rows: List[EvalSummary] = []
    prefilter_stats: Dict[str, int] = {
        "total": 0,
        "kept": 0,
        "rejected": 0,
        "rejected_inlet_error": 0,
        "rejected_invalid_state": 0,
        "rejected_low_mach_inlet": 0,
        "rejected_inlet_unstable": 0,
        "rejected_low_dte_rel_grad": 0,
    }
    for block in _unit_sample_blocks(
        n_total=int(args.n_total),
        shard_index=int(args.shard_index),
        shard_count=max(1, int(args.shard_count)),
        seed=int(args.seed),
        d=3,
        block_size=max(1, int(args.batch_size)),
    ):
        chunk = [
            _candidate_from_unit(
                block[i],
                bounds=bounds,
                T_p_in=float(args.tp_in),
                B=float(args.B),
            )
            for i in range(block.shape[0])
        ]
        if use_prefilter:
            chunk, chunk_stats = _apply_inlet_prefilter_global(
                solver=solver,
                chunk=chunk,
                dte_rel_min=dte_rel_min,
                mach_inlet_min=(
                    None
                    if args.prefilter_mach_inlet_min is None
                    else float(args.prefilter_mach_inlet_min)
                ),
                check_inlet_velikhov=bool(args.prefilter_check_inlet_velikhov),
            )
            for key, value in chunk_stats.items():
                prefilter_stats[key] += int(value)
            if not chunk:
                continue
        rows.extend(
            _evaluate_chunk(
                solver=solver,
                chunk=chunk,
                dx=float(args.dx),
                velikhov_required=(not args.no_velikhov_check),
                reward_config=reward_config,
            )
        )

    rows_sorted = sorted(rows, key=lambda r: r.objective)
    best = rows_sorted[: max(1, int(args.top_k))]
    out_path = Path(args.out)
    _write_rows_atomic(best, out_path)

    feasible_n = sum(1 for r in rows if r.feasible)
    ok_n = sum(1 for r in rows if r.ok)
    print(
        json.dumps(
            {
                "mode": "single_shard",
                "n_total": int(args.n_total),
                "shard_index": int(args.shard_index),
                "shard_count": int(args.shard_count),
                "evaluated": int(len(rows)),
                "ok": int(ok_n),
                "feasible": int(feasible_n),
                "saved": int(len(best)),
                "output": str(out_path),
                "dx": float(args.dx),
                "L_max": float(args.L_max),
                "A_in_fixed": float(FIXED_A_IN),
                "prefilter_enabled": bool(use_prefilter),
                "prefilter_dte_rel_min": (None if dte_rel_min is None else float(dte_rel_min)),
                "prefilter_mach_inlet_min": (
                    None
                    if args.prefilter_mach_inlet_min is None
                    else float(args.prefilter_mach_inlet_min)
                ),
                "prefilter_check_inlet_velikhov": bool(args.prefilter_check_inlet_velikhov),
                "prefilter_stats": {k: int(v) for k, v in prefilter_stats.items()},
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
