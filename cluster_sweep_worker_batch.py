#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats.qmc import LatinHypercube

from local_algebraic_closure import B_FIELD, E_I, K_B
from cluster_sweep_worker import (
    Candidate,
    _build_parser,
    _candidate_from_unit,
    _shard_window,
)
from pde_solver_v6_batch import ForwardPDESolverV6Batch, event_name_from_code


@dataclass
class BatchEvalSummary:
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
    metrics: Dict[str, Any] = field(default_factory=dict)


def _apply_inlet_prefilter(
    solver: ForwardPDESolverV6Batch,
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
        seed_fraction=np.array([c.seed_fraction for c in chunk], dtype=float),
        parallel=True,
    )

    kept: List[Candidate] = []
    for i, cand in enumerate(chunk):
        if not bool(metrics.success[i]):
            code = int(metrics.event_code[i])
            if code == 4:
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


def _velikhov_margin_profile(
    beta: np.ndarray,
    T_e: np.ndarray,
    T_p: np.ndarray,
    n_e: np.ndarray,
    n_p: np.ndarray,
    seed_fraction: float,
) -> np.ndarray:
    ns = seed_fraction * np.maximum(n_p, 1e-300)
    fI = np.clip(n_e / np.maximum(ns, 1e-300), 1e-12, 1.0 - 1e-12)

    delta = np.maximum(T_e / np.maximum(T_p, 1e-300) - 1.0, 1e-12)
    alpha = (K_B * T_e / (2.0 * E_I)) * (2.0 - fI) / np.maximum(1.0 - fI, 1e-12)

    rhs = 4.0 * alpha * (2.0 + 1.0 / delta) * (1.0 + alpha * (1.0 + 1.0 / delta))
    lhs = beta * beta
    return rhs - lhs


def _evaluate_profile_objective(
    cand: Candidate,
    out,
    idx: int,
    *,
    channel_length: float,
    velikhov_enabled: bool,
    stability_mode: str,
    te_out_target: float,
    inlet_te_max_hard: float,
    inlet_te_rel_tol: float,
    mach_buffer: float,
    mach_inlet_min: float,
    trunc_tol: float,
    front_length_frac: float,
    front_fraction_max: float,
    te_step_jump_mult: float,
    z_step_jump_max: float,
    diff2_ref_mult: float,
    w_inlet_cap: float,
    w_inlet_match: float,
    w_te: float,
    w_stab: float,
    w_trunc: float,
    w_mach: float,
    w_mach_inlet: float,
    w_te_step_jump: float,
    w_z_step_jump: float,
    w_front_fraction: float,
    w_diff2: float,
    w_feasible_te_bonus: float,
    w_feasible_mach_bonus: float,
    w_feasible_margin_bonus: float,
    w_mach_inlet_bonus: float,
    mach_inlet_reward_start: float,
    te_growth_min: float,
    w_te_growth: float,
) -> BatchEvalSummary:
    event_name = event_name_from_code(int(out.event_code[idx]))
    valid_points = int(out.valid_points[idx])
    x_end = float(out.x[max(valid_points - 1, 0)]) if valid_points > 0 else 0.0

    if valid_points < 2:
        return BatchEvalSummary(
            ok=False,
            feasible=False,
            reason="too_short_profile",
            objective=1e9,
            outlet_te=np.nan,
            outlet_tp=np.nan,
            outlet_ratio=np.nan,
            min_velikhov_margin=np.nan,
            event=event_name,
            params=asdict(cand),
            metrics={"valid_points": valid_points, "x_end": x_end},
        )

    if not bool(out.success[idx]):
        return BatchEvalSummary(
            ok=False,
            feasible=False,
            reason=f"batch_failed:{event_name or 'unknown'}",
            objective=1e9,
            outlet_te=np.nan,
            outlet_tp=np.nan,
            outlet_ratio=np.nan,
            min_velikhov_margin=np.nan,
            event=event_name,
            params=asdict(cand),
            metrics={"valid_points": valid_points, "x_end": x_end},
        )

    sl = slice(0, valid_points)
    x = np.asarray(out.x[sl], dtype=float)
    n_p = np.asarray(out.n_p[idx, sl], dtype=float)
    T_e = np.asarray(out.T_e[idx, sl], dtype=float)
    T_p = np.asarray(out.T_p[idx, sl], dtype=float)
    n_e = np.asarray(out.n_e[idx, sl], dtype=float)
    beta = np.asarray(out.beta[idx, sl], dtype=float)
    Z = np.asarray(out.Z[idx, sl], dtype=float)
    mach = np.asarray(out.mach[idx, sl], dtype=float)

    Te_out = float(T_e[-1])
    Tp_out = float(T_p[-1])
    ratio_out = Te_out / max(Tp_out, 1e-300)
    Te_in_real = float(T_e[0])
    rel_inlet_mismatch = abs(Te_in_real - cand.T_e_in) / max(cand.T_e_in, 1e-12)
    Z_in_real = float(Z[0])
    Z_step_jump = abs(float(Z[1] - Z[0]))
    Te_step_jump = abs(float(T_e[1] - T_e[0]))

    mach_inlet = float(mach[0])
    min_abs_mach_gap = float(np.min(np.abs(mach - 1.0)))

    margin_profile = _velikhov_margin_profile(
        beta=beta,
        T_e=T_e,
        T_p=T_p,
        n_e=n_e,
        n_p=n_p,
        seed_fraction=cand.seed_fraction,
    )
    stable_mask = np.asarray(margin_profile >= 0.0, dtype=bool)
    stable_fraction_raw = float(np.mean(stable_mask))
    outlet_stable_raw = bool(stable_mask[-1])
    min_margin = float(np.nanmin(margin_profile))

    if velikhov_enabled:
        if stability_mode == "profile":
            stable_ok = bool(np.all(stable_mask))
        elif stability_mode == "outlet":
            stable_ok = outlet_stable_raw
        else:
            raise ValueError(f"invalid stability_mode: {stability_mode}")
        stable_fraction = stable_fraction_raw
        outlet_stable = outlet_stable_raw
    else:
        stable_ok = True
        stable_fraction = 1.0
        outlet_stable = True

    truncated = x_end < channel_length * (1.0 - trunc_tol)

    n_eff = max(valid_points, 2)
    dTe_target = max(te_out_target - cand.T_e_in, 1e-6)
    te_step_ref = dTe_target / ((n_eff - 1) ** 2)
    diff2_ref = 2.0 * dTe_target / ((n_eff - 1) ** 2)
    te_step_jump_ok = Te_step_jump <= te_step_jump_mult * te_step_ref
    z_step_jump_ok = Z_step_jump <= z_step_jump_max

    x0 = float(x[0])
    x1 = float(x[-1])
    span = max(x1 - x0, 1e-12)
    x_front = x0 + front_length_frac * span
    front_mask = x <= x_front
    if not np.any(front_mask):
        front_mask = np.zeros_like(x, dtype=bool)
        front_mask[0] = True
    Te_front_end = float(np.max(T_e[front_mask]))
    total_rise = max(Te_out - Te_in_real, 1e-12)
    front_rise = max(Te_front_end - Te_in_real, 0.0)
    front_fraction = front_rise / total_rise
    front_fraction_ok = front_fraction <= front_fraction_max

    if valid_points >= 3:
        diff2 = np.diff(T_e, n=2)
        max_abs_diff2 = float(np.max(np.abs(diff2)))
    else:
        max_abs_diff2 = np.inf
    diff2_ok = max_abs_diff2 <= diff2_ref_mult * diff2_ref

    inlet_cap_ok = Te_in_real <= inlet_te_max_hard
    inlet_match_ok = rel_inlet_mismatch <= inlet_te_rel_tol
    te_ok = Te_out >= te_out_target
    mach_ok = min_abs_mach_gap >= mach_buffer
    mach_inlet_ok = mach_inlet >= mach_inlet_min
    trunc_ok = not truncated
    te_growth_frac = (Te_out - Te_in_real) / max(Te_in_real, 1e-12)
    te_growth_ok = te_growth_frac >= te_growth_min

    feasible = bool(
        inlet_cap_ok
        and inlet_match_ok
        and te_ok
        and stable_ok
        and mach_ok
        and mach_inlet_ok
        and trunc_ok
        and te_growth_ok
        and te_step_jump_ok
        and z_step_jump_ok
        and front_fraction_ok
        and diff2_ok
    )

    viol_inlet_cap = max(0.0, Te_in_real - inlet_te_max_hard) / max(inlet_te_max_hard, 1.0)
    viol_inlet_match = max(0.0, rel_inlet_mismatch - inlet_te_rel_tol) / max(inlet_te_rel_tol, 1e-12)
    viol_te = max(0.0, te_out_target - Te_out) / max(te_out_target, 1.0)
    viol_stab = 0.0 if stable_ok else (1.0 - stable_fraction)
    viol_mach = max(0.0, mach_buffer - min_abs_mach_gap) / max(mach_buffer, 1e-6)
    viol_mach_inlet = max(0.0, mach_inlet_min - mach_inlet) / max(mach_inlet_min, 1e-6)
    viol_trunc = max(0.0, (channel_length - x_end) / max(channel_length, 1e-12))
    viol_te_growth = max(0.0, te_growth_min - te_growth_frac) / max(te_growth_min, 1e-12)
    viol_te_step_jump = max(0.0, Te_step_jump / max(te_step_jump_mult * te_step_ref, 1e-12) - 1.0)
    viol_z_step_jump = max(0.0, Z_step_jump / max(z_step_jump_max, 1e-12) - 1.0)
    viol_front_fraction = max(0.0, front_fraction / max(front_fraction_max, 1e-12) - 1.0)
    viol_diff2 = max(0.0, max_abs_diff2 / max(diff2_ref_mult * diff2_ref, 1e-12) - 1.0)

    objective = (
        w_inlet_cap * viol_inlet_cap
        + w_inlet_match * viol_inlet_match
        + w_te * viol_te
        + w_stab * viol_stab
        + w_trunc * viol_trunc
        + w_mach * viol_mach
        + w_mach_inlet * viol_mach_inlet
        + w_te_growth * viol_te_growth
        + w_te_step_jump * viol_te_step_jump
        + w_z_step_jump * viol_z_step_jump
        + w_front_fraction * viol_front_fraction
        + w_diff2 * viol_diff2
    )
    inlet_mach_reward = max(
        0.0,
        min(1.0, (mach_inlet - mach_inlet_reward_start) / max(1.0 - mach_inlet_reward_start, 1e-6)),
    )
    objective -= w_mach_inlet_bonus * inlet_mach_reward
    if feasible:
        objective -= w_feasible_te_bonus * (Te_out - te_out_target) / max(te_out_target, 1.0)
        objective -= w_feasible_mach_bonus * min_abs_mach_gap
        if velikhov_enabled:
            objective -= w_feasible_margin_bonus * min_margin

    metrics: Dict[str, Any] = {
        "valid_points": valid_points,
        "step_size": float(out.step_size),
        "x_end": x_end,
        "Te_in_real": Te_in_real,
        "Te_in_target": float(cand.T_e_in),
        "rel_inlet_mismatch": rel_inlet_mismatch,
        "Z_in_real": Z_in_real,
        "Te_step_jump": Te_step_jump,
        "Z_step_jump": Z_step_jump,
        "te_step_ref": te_step_ref,
        "max_abs_diff2": max_abs_diff2,
        "diff2_ref": diff2_ref,
        "front_fraction": front_fraction,
        "Te_out": Te_out,
        "te_growth_frac": te_growth_frac,
        "Te_out_over_Tp_out": ratio_out,
        "mach_inlet": mach_inlet,
        "min_abs_mach_gap": min_abs_mach_gap,
        "stable_fraction": stable_fraction,
        "outlet_stable": float(outlet_stable),
        "velikhov_min_margin": min_margin,
        "truncated": float(truncated),
        "viol_inlet_cap": viol_inlet_cap,
        "viol_inlet_match": viol_inlet_match,
        "viol_te": viol_te,
        "viol_stab": viol_stab,
        "viol_mach": viol_mach,
        "viol_mach_inlet": viol_mach_inlet,
        "viol_te_growth": viol_te_growth,
        "viol_trunc": viol_trunc,
        "viol_te_step_jump": viol_te_step_jump,
        "viol_z_step_jump": viol_z_step_jump,
        "viol_front_fraction": viol_front_fraction,
        "viol_diff2": viol_diff2,
    }

    return BatchEvalSummary(
        ok=True,
        feasible=feasible,
        reason="feasible" if feasible else "constraint_violated",
        objective=float(objective),
        outlet_te=Te_out,
        outlet_tp=Tp_out,
        outlet_ratio=ratio_out,
        min_velikhov_margin=min_margin,
        event=event_name,
        params=asdict(cand),
        metrics=metrics,
    )


def _evaluate_chunk(
    solver: ForwardPDESolverV6Batch,
    chunk: List[Candidate],
    dx: float,
    *,
    channel_length: float,
    velikhov_enabled: bool,
    stability_mode: str,
    te_out_target: float,
    inlet_te_max_hard: float,
    inlet_te_rel_tol: float,
    mach_buffer: float,
    mach_inlet_min: float,
    trunc_tol: float,
    front_length_frac: float,
    front_fraction_max: float,
    te_step_jump_mult: float,
    z_step_jump_max: float,
    diff2_ref_mult: float,
    w_inlet_cap: float,
    w_inlet_match: float,
    w_te: float,
    w_stab: float,
    w_trunc: float,
    w_mach: float,
    w_mach_inlet: float,
    w_te_step_jump: float,
    w_z_step_jump: float,
    w_front_fraction: float,
    w_diff2: float,
    w_feasible_te_bonus: float,
    w_feasible_mach_bonus: float,
    w_feasible_margin_bonus: float,
    w_mach_inlet_bonus: float,
    mach_inlet_reward_start: float,
    te_growth_min: float,
    w_te_growth: float,
) -> List[BatchEvalSummary]:
    out = solver.solve_batch(
        n_p_in=np.array([c.n_p_in for c in chunk], dtype=float),
        Z_in=np.array([c.Z_in for c in chunk], dtype=float),
        T_p_in=np.array([c.T_p_in for c in chunk], dtype=float),
        T_e_in=np.array([c.T_e_in for c in chunk], dtype=float),
        seed_fraction=np.array([c.seed_fraction for c in chunk], dtype=float),
        dx=dx,
        store_profiles=True,
    )

    rows: List[BatchEvalSummary] = []
    for i, cand in enumerate(chunk):
        rows.append(
            _evaluate_profile_objective(
                cand,
                out,
                i,
                channel_length=channel_length,
                velikhov_enabled=velikhov_enabled,
                stability_mode=stability_mode,
                te_out_target=te_out_target,
                inlet_te_max_hard=inlet_te_max_hard,
                inlet_te_rel_tol=inlet_te_rel_tol,
                mach_buffer=mach_buffer,
                mach_inlet_min=mach_inlet_min,
                trunc_tol=trunc_tol,
                front_length_frac=front_length_frac,
                front_fraction_max=front_fraction_max,
                te_step_jump_mult=te_step_jump_mult,
                z_step_jump_max=z_step_jump_max,
                diff2_ref_mult=diff2_ref_mult,
                w_inlet_cap=w_inlet_cap,
                w_inlet_match=w_inlet_match,
                w_te=w_te,
                w_stab=w_stab,
                w_trunc=w_trunc,
                w_mach=w_mach,
                w_mach_inlet=w_mach_inlet,
                w_te_step_jump=w_te_step_jump,
                w_z_step_jump=w_z_step_jump,
                w_front_fraction=w_front_fraction,
                w_diff2=w_diff2,
                w_feasible_te_bonus=w_feasible_te_bonus,
                w_feasible_mach_bonus=w_feasible_mach_bonus,
                w_feasible_margin_bonus=w_feasible_margin_bonus,
                w_mach_inlet_bonus=w_mach_inlet_bonus,
                mach_inlet_reward_start=mach_inlet_reward_start,
                te_growth_min=te_growth_min,
                w_te_growth=w_te_growth,
            )
        )

    return rows


def main() -> int:
    p = _build_parser()
    p.description = "V6 cluster sweep worker (batch fixed-step)"
    p.add_argument("--batch-size", type=int, default=64, help="number of candidates per batch solve")
    p.add_argument("--dx", type=float, default=2e-5, help="fixed spatial step for batch RK4")
    p.add_argument(
        "--prefilter-dte-rel-min",
        type=float,
        default=None,
        help="minimum inlet (dTe/dx)/Te_in [1/m] for batch prefilter; default is 0.01/L unless disabled",
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
    p.add_argument("--stability-mode", choices=["profile", "outlet"], default="profile")
    p.add_argument("--te-out-target", type=float, default=4000.0)
    p.add_argument("--inlet-te-max-hard", type=float, default=3000.0)
    p.add_argument("--inlet-te-rel-tol", type=float, default=0.05)
    p.add_argument("--mach-buffer", type=float, default=0.03)
    p.add_argument("--mach-inlet-min", type=float, default=0.8)
    p.add_argument("--trunc-tol", type=float, default=0.01)
    p.add_argument("--front-length-frac", type=float, default=0.05)
    p.add_argument("--front-fraction-max", type=float, default=0.2)
    p.add_argument("--te-step-jump-mult", type=float, default=500.0)
    p.add_argument("--z-step-jump-max", type=float, default=20.0)
    p.add_argument("--diff2-ref-mult", type=float, default=50.0)
    p.add_argument("--w-inlet-cap", type=float, default=1200.0)
    p.add_argument("--w-inlet-match", type=float, default=800.0)
    p.add_argument("--w-te", type=float, default=100.0)
    p.add_argument("--w-stab", type=float, default=200.0)
    p.add_argument("--w-trunc", type=float, default=100.0)
    p.add_argument("--w-mach", type=float, default=50.0)
    p.add_argument("--w-mach-inlet", type=float, default=150.0)
    p.add_argument("--w-te-step-jump", type=float, default=300.0)
    p.add_argument("--w-z-step-jump", type=float, default=200.0)
    p.add_argument("--w-front-fraction", type=float, default=250.0)
    p.add_argument("--w-diff2", type=float, default=250.0)
    p.add_argument("--w-feasible-te-bonus", type=float, default=5.0)
    p.add_argument("--w-feasible-mach-bonus", type=float, default=2.0)
    p.add_argument("--w-feasible-margin-bonus", type=float, default=0.2)
    p.add_argument("--w-mach-inlet-bonus", type=float, default=30.0)
    p.add_argument("--mach-inlet-reward-start", type=float, default=0.8)
    p.add_argument("--te-growth-min", type=float, default=0.01)
    p.add_argument("--w-te-growth", type=float, default=300.0)
    args = p.parse_args()

    bounds = {
        "n_p_in": (float(args.np_min), float(args.np_max)),
        "Z_in": (float(args.z_min), float(args.z_max)),
        "T_e_in": (float(args.te_min), float(args.te_max)),
        "seed_fraction": (float(args.seedf_min), float(args.seedf_max)),
    }
    for k, (lo, hi) in bounds.items():
        if lo <= 0.0 and k in ("n_p_in", "seed_fraction"):
            raise ValueError(f"{k} must be positive")
        if hi <= lo:
            raise ValueError(f"{k} bounds invalid: {lo}, {hi}")

    start, count = _shard_window(int(args.n_total), int(args.shard_index), int(args.shard_count))
    lhs = LatinHypercube(d=4, seed=int(args.seed))
    U = lhs.random(n=int(args.n_total))
    U = U[start:start + count]

    solver = ForwardPDESolverV6Batch(B=float(args.B), length=float(args.L))
    batch_size = max(1, int(args.batch_size))
    rows: List[BatchEvalSummary] = []
    dte_rel_min = None if args.no_prefilter_dte_rel else (
        float(args.prefilter_dte_rel_min) if args.prefilter_dte_rel_min is not None else (0.01 / float(args.L))
    )
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
    use_prefilter = (
        dte_rel_min is not None
        or args.prefilter_mach_inlet_min is not None
        or bool(args.prefilter_check_inlet_velikhov)
    )

    for offset in range(0, U.shape[0], batch_size):
        block = U[offset : offset + batch_size]
        chunk = [
            _candidate_from_unit(block[i], bounds=bounds, T_p_in=float(args.tp_in), B=float(args.B))
            for i in range(block.shape[0])
        ]
        if use_prefilter:
            chunk, chunk_stats = _apply_inlet_prefilter(
                solver=solver,
                chunk=chunk,
                dte_rel_min=dte_rel_min,
                mach_inlet_min=(
                    None if args.prefilter_mach_inlet_min is None else float(args.prefilter_mach_inlet_min)
                ),
                check_inlet_velikhov=bool(args.prefilter_check_inlet_velikhov),
            )
            for key, value in chunk_stats.items():
                prefilter_stats[key] += int(value)
        else:
            prefilter_stats["total"] += len(chunk)
            prefilter_stats["kept"] += len(chunk)

        if not chunk:
            continue

        rows.extend(
            _evaluate_chunk(
                solver=solver,
                chunk=chunk,
                dx=float(args.dx),
                channel_length=float(args.L),
                velikhov_enabled=(not args.no_velikhov_check),
                stability_mode=str(args.stability_mode),
                te_out_target=float(args.te_out_target),
                inlet_te_max_hard=float(args.inlet_te_max_hard),
                inlet_te_rel_tol=float(args.inlet_te_rel_tol),
                mach_buffer=float(args.mach_buffer),
                mach_inlet_min=float(args.mach_inlet_min),
                trunc_tol=float(args.trunc_tol),
                front_length_frac=float(args.front_length_frac),
                front_fraction_max=float(args.front_fraction_max),
                te_step_jump_mult=float(args.te_step_jump_mult),
                z_step_jump_max=float(args.z_step_jump_max),
                diff2_ref_mult=float(args.diff2_ref_mult),
                w_inlet_cap=float(args.w_inlet_cap),
                w_inlet_match=float(args.w_inlet_match),
                w_te=float(args.w_te),
                w_stab=float(args.w_stab),
                w_trunc=float(args.w_trunc),
                w_mach=float(args.w_mach),
                w_mach_inlet=float(args.w_mach_inlet),
                w_te_step_jump=float(args.w_te_step_jump),
                w_z_step_jump=float(args.w_z_step_jump),
                w_front_fraction=float(args.w_front_fraction),
                w_diff2=float(args.w_diff2),
                w_feasible_te_bonus=float(args.w_feasible_te_bonus),
                w_feasible_mach_bonus=float(args.w_feasible_mach_bonus),
                w_feasible_margin_bonus=float(args.w_feasible_margin_bonus),
                w_mach_inlet_bonus=float(args.w_mach_inlet_bonus),
                mach_inlet_reward_start=float(args.mach_inlet_reward_start),
                te_growth_min=float(args.te_growth_min),
                w_te_growth=float(args.w_te_growth),
            )
        )

    rows_sorted = sorted(rows, key=lambda r: r.objective)
    k = max(1, int(args.top_k))
    best = rows_sorted[:k]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in best:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    feasible_n = sum(1 for r in rows if r.feasible)
    ok_n = sum(1 for r in rows if r.ok)
    print(
        json.dumps(
            {
                "n_total": int(args.n_total),
                "shard_index": int(args.shard_index),
                "shard_count": int(args.shard_count),
                "evaluated": int(len(rows)),
                "ok": int(ok_n),
                "feasible": int(feasible_n),
                "saved": int(len(best)),
                "output": str(out_path),
                "batch_size": batch_size,
                "dx": float(args.dx),
                "velikhov_enabled": bool(not args.no_velikhov_check),
                "stability_mode": str(args.stability_mode),
                "prefilter_enabled": bool(use_prefilter),
                "prefilter_dte_rel_min": (None if dte_rel_min is None else float(dte_rel_min)),
                "prefilter_mach_inlet_min": (
                    None if args.prefilter_mach_inlet_min is None else float(args.prefilter_mach_inlet_min)
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
