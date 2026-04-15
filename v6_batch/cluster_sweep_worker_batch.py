#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_core.local_algebraic_closure import B_FIELD, E_I, K_B
from v6_batch.pde_solver_v6_batch import ForwardPDESolverV6Batch, event_name_from_code


TASK_FILE_RE = re.compile(r"^shard_(\d+)(?:\..+)?\.task$")


@dataclass
class Candidate:
    n_p_in: float
    Z_in: float
    T_p_in: float
    T_e_in: float
    seed_fraction: float
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


@dataclass(frozen=True)
class RewardConfig:
    te_out_target: float = 4000.0
    mach_buffer: float = 0.03
    truncation_tol: float = 0.01
    te_growth_min: float = 0.01
    te_growth_bonus_target: float = 0.10
    te_growth_bonus_dominance: float = 10.0
    front_length_frac: float = 0.05
    front_fraction_max: float = 0.2
    te_step_jump_mult: float = 500.0
    z_step_jump_max: float = 20.0
    te_step_jump_frac_max: float = 0.1
    z_step_jump_frac_max: float = 0.1
    diff2_ref_mult: float = 50.0
    w_te: float = 0.0
    w_stab: float = 200.0
    w_trunc: float = 0.0
    w_mach: float = 50.0
    w_te_growth: float = 300.0
    w_te_step_jump: float = 300.0
    w_z_step_jump: float = 200.0
    w_front_fraction: float = 250.0
    w_diff2: float = 250.0
    w_feasible_te_bonus: float = 0.0
    w_feasible_te_growth_bonus: float | None = None
    w_feasible_mach_bonus: float = 2.0
    w_feasible_margin_bonus: float = 0.2


def _log_map(u: float, lo: float, hi: float) -> float:
    return float(10.0 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo))))


def _lin_map(u: float, lo: float, hi: float) -> float:
    return float(lo + u * (hi - lo))


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
        seed_fraction=_log_map(float(u[3]), *bounds["seed_fraction"]),
        B=float(B),
    )


def _velikhov_margin(
    beta: np.ndarray,
    T_e: np.ndarray,
    T_p: np.ndarray,
    n_e: np.ndarray,
    n_p: np.ndarray,
    seed_fraction: float,
) -> float:
    ns = seed_fraction * np.maximum(n_p, 1e-300)
    fI = np.clip(n_e / np.maximum(ns, 1e-300), 1e-12, 1.0 - 1e-12)

    delta = np.maximum(T_e / np.maximum(T_p, 1e-300) - 1.0, 1e-12)
    alpha = (K_B * T_e / (2.0 * E_I)) * (2.0 - fI) / np.maximum(1.0 - fI, 1e-12)

    rhs = 4.0 * alpha * (2.0 + 1.0 / delta) * (1.0 + alpha * (1.0 + 1.0 / delta))
    lhs = beta * beta
    margin = rhs - lhs
    return float(np.min(margin))


def _reward_config_from_args(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        te_out_target=float(args.te_out_target),
        mach_buffer=float(args.mach_buffer),
        truncation_tol=float(args.trunc_tol),
        te_growth_min=float(args.te_growth_min),
        te_growth_bonus_target=float(args.te_growth_bonus_target),
        te_growth_bonus_dominance=float(args.te_growth_bonus_dominance),
        front_length_frac=float(args.front_length_frac),
        front_fraction_max=float(args.front_fraction_max),
        te_step_jump_mult=float(args.te_step_jump_mult),
        z_step_jump_max=float(args.z_step_jump_max),
        te_step_jump_frac_max=float(args.te_step_jump_frac_max),
        z_step_jump_frac_max=float(args.z_step_jump_frac_max),
        diff2_ref_mult=float(args.diff2_ref_mult),
        w_te=float(args.w_te),
        w_stab=float(args.w_stab),
        w_trunc=float(args.w_trunc),
        w_mach=float(args.w_mach),
        w_te_growth=float(args.w_te_growth),
        w_te_step_jump=float(args.w_te_step_jump),
        w_z_step_jump=float(args.w_z_step_jump),
        w_front_fraction=float(args.w_front_fraction),
        w_diff2=float(args.w_diff2),
        w_feasible_te_bonus=float(args.w_feasible_te_bonus),
        w_feasible_te_growth_bonus=(
            None
            if args.w_feasible_te_growth_bonus is None
            else float(args.w_feasible_te_growth_bonus)
        ),
        w_feasible_mach_bonus=float(args.w_feasible_mach_bonus),
        w_feasible_margin_bonus=float(args.w_feasible_margin_bonus),
    )


def _dominant_violation_reason(violations: Dict[str, float]) -> str:
    key = max(violations, key=violations.get)
    return key if violations[key] > 0.0 else "other_constraint"


def _resolved_feasible_te_growth_bonus_weight(reward_config: RewardConfig) -> float:
    if reward_config.w_feasible_te_growth_bonus is not None:
        return float(reward_config.w_feasible_te_growth_bonus)

    other_weight_scale = (
        abs(reward_config.w_stab)
        + abs(reward_config.w_trunc)
        + abs(reward_config.w_mach)
        + abs(reward_config.w_te_growth)
        + abs(reward_config.w_te_step_jump)
        + abs(reward_config.w_z_step_jump)
        + abs(reward_config.w_front_fraction)
        + abs(reward_config.w_diff2)
        + abs(reward_config.w_feasible_mach_bonus)
        + abs(reward_config.w_feasible_margin_bonus)
    )
    return float(reward_config.te_growth_bonus_dominance) * max(other_weight_scale, 1.0)


def _score_profile(
    *,
    x: np.ndarray,
    T_e: np.ndarray,
    T_p: np.ndarray,
    Z: np.ndarray,
    mach: np.ndarray,
    beta: np.ndarray,
    n_e: np.ndarray,
    n_p: np.ndarray,
    seed_fraction: float,
    length: float,
    reached_end: bool,
    velikhov_required: bool,
    reward_config: RewardConfig,
) -> Dict[str, object]:
    if len(x) == 0:
        return {
            "feasible": False,
            "reason": "too_short_profile",
            "objective": 1e30,
            "outlet_te": np.nan,
            "outlet_tp": np.nan,
            "outlet_ratio": np.nan,
            "min_velikhov_margin": np.nan,
            "metrics": {"reached_end": bool(reached_end)},
        }

    mach_gap_full = np.abs(mach - 1.0)
    unsafe_idx = np.flatnonzero(mach_gap_full < reward_config.mach_buffer)
    if unsafe_idx.size > 0:
        effective_last_idx = int(unsafe_idx[0]) - 1
    else:
        effective_last_idx = len(x) - 1

    if effective_last_idx < 0:
        return {
            "feasible": False,
            "reason": "near_sonic",
            "objective": 1e30,
            "outlet_te": np.nan,
            "outlet_tp": np.nan,
            "outlet_ratio": np.nan,
            "min_velikhov_margin": np.nan,
            "metrics": {
                "reached_end": False,
                "effective_last_idx": -1,
                "unsafe_mach_idx": int(unsafe_idx[0]),
                "min_abs_mach_gap_full": float(np.min(mach_gap_full)),
            },
        }

    x = x[: effective_last_idx + 1]
    T_e = T_e[: effective_last_idx + 1]
    T_p = T_p[: effective_last_idx + 1]
    Z = Z[: effective_last_idx + 1]
    mach = mach[: effective_last_idx + 1]
    beta = beta[: effective_last_idx + 1]
    n_e = n_e[: effective_last_idx + 1]
    n_p = n_p[: effective_last_idx + 1]

    effective_reached_end = bool(
        reached_end and effective_last_idx == (len(mach_gap_full) - 1)
    )

    te_out = float(T_e[-1])
    tp_out = float(T_p[-1])
    te_in = float(T_e[0])
    ratio_out = te_out / max(tp_out, 1e-300)
    x_end = float(x[-1])
    truncated = x_end < float(length) * (1.0 - reward_config.truncation_tol)

    if len(x) >= 2:
        dx_typ = float(np.median(np.diff(x)))
    else:
        dx_typ = max(float(length), 1e-12)
    dx_typ = max(dx_typ, 1e-12)
    te_step_jump = abs(float(T_e[1] - T_e[0])) if len(T_e) > 1 else np.inf
    z_step_jump = abs(float(Z[1] - Z[0])) if len(Z) > 1 else np.inf

    x0 = float(x[0])
    span = max(x_end - x0, 1e-12)
    te_profile_excursion = max(float(np.max(T_e) - np.min(T_e)), 1e-6)
    z_profile_excursion = max(float(np.max(Z) - np.min(Z)), 1e-6)
    te_step_limit = reward_config.te_step_jump_frac_max * te_profile_excursion
    z_step_limit = reward_config.z_step_jump_frac_max * z_profile_excursion
    diff2_ref = 2.0 * te_profile_excursion * (dx_typ / span) ** 2
    x_front = x0 + reward_config.front_length_frac * span
    front_mask = x <= x_front
    if not np.any(front_mask):
        front_mask = np.zeros_like(x, dtype=bool)
        front_mask[0] = True
    te_front_end = float(np.max(T_e[front_mask]))
    total_rise = max(te_out - te_in, 1e-12)
    front_rise = max(te_front_end - te_in, 0.0)
    front_fraction = front_rise / total_rise

    if len(T_e) >= 3:
        diff2 = np.diff(T_e, n=2)
        max_abs_diff2 = float(np.max(np.abs(diff2)))
    else:
        max_abs_diff2 = np.inf

    min_abs_mach_gap = float(np.min(np.abs(mach - 1.0)))
    min_abs_mach_gap_full = float(np.min(mach_gap_full))
    te_growth_frac = (te_out - te_in) / max(te_in, 1e-12)
    min_margin = _velikhov_margin(
        beta=beta,
        T_e=T_e,
        T_p=T_p,
        n_e=n_e,
        n_p=n_p,
        seed_fraction=seed_fraction,
    )
    stable_ok = (min_margin >= 0.0) if velikhov_required else True

    mach_ok = min_abs_mach_gap >= reward_config.mach_buffer
    trunc_ok = not truncated
    te_growth_ok = te_growth_frac >= reward_config.te_growth_min
    te_step_jump_ok = te_step_jump <= te_step_limit
    z_step_jump_ok = z_step_jump <= z_step_limit
    front_fraction_ok = front_fraction <= reward_config.front_fraction_max
    diff2_ok = max_abs_diff2 <= reward_config.diff2_ref_mult * diff2_ref

    feasible = bool(
        stable_ok
        and mach_ok
        and trunc_ok
        and te_growth_ok
        and te_step_jump_ok
        and z_step_jump_ok
        and front_fraction_ok
        and diff2_ok
    )

    viol_te = 0.0
    viol_stab = 0.0 if stable_ok else 1.0
    viol_trunc = max(0.0, (length - x_end) / max(length, 1e-12))
    viol_mach = 0.0 if unsafe_idx.size > 0 else max(
        0.0, reward_config.mach_buffer - min_abs_mach_gap
    ) / max(reward_config.mach_buffer, 1e-6)
    viol_te_growth = max(0.0, reward_config.te_growth_min - te_growth_frac) / max(
        reward_config.te_growth_min, 1e-12
    )
    viol_te_step_jump = max(0.0, te_step_jump / max(te_step_limit, 1e-12) - 1.0)
    viol_z_step_jump = max(0.0, z_step_jump / max(z_step_limit, 1e-12) - 1.0)
    viol_front_fraction = max(
        0.0, front_fraction / max(reward_config.front_fraction_max, 1e-12) - 1.0
    )
    viol_diff2 = max(
        0.0, max_abs_diff2 / max(reward_config.diff2_ref_mult * diff2_ref, 1e-12) - 1.0
    )
    growth_bonus_weight = _resolved_feasible_te_growth_bonus_weight(reward_config)
    growth_bonus_score = te_growth_frac / max(reward_config.te_growth_bonus_target, 1e-12)
    margin_bonus_score = math.log10(1.0 + max(min_margin, 0.0))

    objective = (
        reward_config.w_stab * viol_stab
        + reward_config.w_trunc * viol_trunc
        + reward_config.w_mach * viol_mach
        + reward_config.w_te_growth * viol_te_growth
        + reward_config.w_te_step_jump * viol_te_step_jump
        + reward_config.w_z_step_jump * viol_z_step_jump
        + reward_config.w_front_fraction * viol_front_fraction
        + reward_config.w_diff2 * viol_diff2
    )
    if feasible:
        objective -= growth_bonus_weight * growth_bonus_score
        objective -= reward_config.w_feasible_mach_bonus * min_abs_mach_gap
        objective -= reward_config.w_feasible_margin_bonus * margin_bonus_score

    reason = "ok" if feasible else _dominant_violation_reason(
        {
            "velikhov_unstable": viol_stab,
            "near_sonic": viol_mach,
            "truncated_early": viol_trunc,
            "te_growth_too_low": viol_te_growth,
            "step1_Te_jump": viol_te_step_jump,
            "step1_Z_jump": viol_z_step_jump,
            "front_loaded_heating": viol_front_fraction,
            "rough_Te_curve": viol_diff2,
        }
    )

    return {
        "feasible": feasible,
        "reason": reason,
        "objective": float(objective),
        "outlet_te": te_out,
        "outlet_tp": tp_out,
        "outlet_ratio": ratio_out,
        "min_velikhov_margin": float(min_margin),
        "metrics": {
            "x_end": x_end,
            "reached_end": bool(effective_reached_end),
            "reached_end_full": bool(reached_end),
            "truncated": bool(truncated),
            "effective_last_idx": int(effective_last_idx),
            "unsafe_mach_idx": (int(unsafe_idx[0]) if unsafe_idx.size > 0 else -1),
            "te_growth_frac": float(te_growth_frac),
            "min_abs_mach_gap": float(min_abs_mach_gap),
            "min_abs_mach_gap_full": float(min_abs_mach_gap_full),
            "te_profile_excursion": float(te_profile_excursion),
            "z_profile_excursion": float(z_profile_excursion),
            "te_step_jump": float(te_step_jump),
            "z_step_jump": float(z_step_jump),
            "te_step_ref": float(te_step_limit),
            "te_step_limit": float(te_step_limit),
            "z_step_limit": float(z_step_limit),
            "te_step_jump_frac": float(te_step_jump / te_profile_excursion),
            "z_step_jump_frac": float(z_step_jump / z_profile_excursion),
            "diff2_ref": float(diff2_ref),
            "front_fraction": float(front_fraction),
            "max_abs_diff2": float(max_abs_diff2),
            "viol_te": float(viol_te),
            "viol_stab": float(viol_stab),
            "viol_trunc": float(viol_trunc),
            "viol_mach": float(viol_mach),
            "viol_te_growth": float(viol_te_growth),
            "viol_te_step_jump": float(viol_te_step_jump),
            "viol_z_step_jump": float(viol_z_step_jump),
            "viol_front_fraction": float(viol_front_fraction),
            "viol_diff2": float(viol_diff2),
            "te_growth_bonus_score": float(growth_bonus_score),
            "margin_bonus_score": float(margin_bonus_score),
            "effective_feasible_te_growth_bonus_weight": float(growth_bonus_weight),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="V6 cluster sweep worker (LHS)")
    p.add_argument("--n-total", type=int, default=10000, help="total LHS points")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--top-k", type=int, default=200, help="keep best K results per shard")
    p.add_argument("--out", type=str, default="sweep_results.jsonl")

    p.add_argument("--B", type=float, default=B_FIELD)
    p.add_argument("--L", type=float, default=0.039)
    p.add_argument("--tp-in", type=float, default=1000.0)

    p.add_argument("--np-min", type=float, default=1e21)
    p.add_argument("--np-max", type=float, default=1e24)
    p.add_argument("--z-min", type=float, default=1.0)
    p.add_argument("--z-max", type=float, default=120.0)
    p.add_argument("--te-min", type=float, default=1500.0)
    p.add_argument("--te-max", type=float, default=3500.0)
    p.add_argument("--seedf-min", type=float, default=1e-4)
    p.add_argument("--seedf-max", type=float, default=5e-2)

    p.add_argument(
        "--te-out-target",
        type=float,
        default=4000.0,
        help="legacy absolute-Te target kept for compatibility; current reward does not use it",
    )
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
        help="if feasible Te-growth bonus weight is omitted, auto-scale it so reaching --te-growth-bonus-target dominates the other configured reward weights by at least this factor",
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
        "--te-step-jump-mult",
        type=float,
        default=500.0,
        help="legacy first-step Te multiplier kept for compatibility; current reward uses ratio-based step limits",
    )
    p.add_argument(
        "--z-step-jump-max",
        type=float,
        default=20.0,
        help="legacy first-step Z bound kept for compatibility; current reward uses ratio-based step limits",
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
    p.add_argument(
        "--w-te",
        type=float,
        default=0.0,
        help="legacy weight kept for compatibility; current reward does not penalize absolute Te_out",
    )
    p.add_argument(
        "--w-stab",
        type=float,
        default=200.0,
        help="weight for Velikhov instability violation",
    )
    p.add_argument(
        "--w-trunc",
        type=float,
        default=0.0,
        help="weight for early truncation violation",
    )
    p.add_argument(
        "--w-mach",
        type=float,
        default=50.0,
        help="weight for Mach-buffer violation",
    )
    p.add_argument(
        "--w-te-growth",
        type=float,
        default=300.0,
        help="weight for low Te growth",
    )
    p.add_argument(
        "--w-te-step-jump",
        type=float,
        default=300.0,
        help="weight for first-step Te jump",
    )
    p.add_argument(
        "--w-z-step-jump",
        type=float,
        default=200.0,
        help="weight for first-step Z jump",
    )
    p.add_argument(
        "--w-front-fraction",
        type=float,
        default=250.0,
        help="weight for front-loaded heating",
    )
    p.add_argument(
        "--w-diff2",
        type=float,
        default=250.0,
        help="weight for rough Te curve",
    )
    p.add_argument(
        "--w-feasible-te-bonus",
        type=float,
        default=0.0,
        help="legacy bonus kept for compatibility; current reward does not bonus absolute Te_out",
    )
    p.add_argument(
        "--w-feasible-te-growth-bonus",
        type=float,
        default=None,
        help="weight for larger feasible Te growth after normalizing by --te-growth-bonus-target; default auto-scales from the other configured reward weights",
    )
    p.add_argument(
        "--w-feasible-mach-bonus",
        type=float,
        default=2.0,
        help="bonus for larger Mach gap on feasible profiles",
    )
    p.add_argument(
        "--w-feasible-margin-bonus",
        type=float,
        default=0.2,
        help="bonus for larger Velikhov margin on feasible profiles, applied to log10(1 + margin)",
    )
    p.add_argument("--no-velikhov-check", action="store_true")
    return p


def _shard_window(n_total: int, shard_index: int, shard_count: int) -> Tuple[int, int]:
    if shard_count <= 0:
        raise ValueError("shard_count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("invalid shard_index")
    base = n_total // shard_count
    rem = n_total % shard_count
    start = shard_index * base + min(shard_index, rem)
    size = base + (1 if shard_index < rem else 0)
    return start, size


def _shard_rng(
    n_total: int,
    shard_index: int,
    shard_count: int,
    seed: int,
) -> np.random.Generator:
    # Mix the global problem size and shard metadata into the seed so each shard
    # gets an independent but fully reproducible sample stream.
    seed_seq = np.random.SeedSequence(
        [int(seed), int(n_total), int(shard_index), int(shard_count)]
    )
    return np.random.default_rng(seed_seq)


def _unit_sample_blocks(
    *,
    n_total: int,
    shard_index: int,
    shard_count: int,
    seed: int,
    d: int,
    block_size: int,
):
    if block_size <= 0:
        raise ValueError("block_size must be >= 1")

    _, count = _shard_window(n_total, shard_index, shard_count)
    rng = _shard_rng(
        n_total=n_total,
        shard_index=shard_index,
        shard_count=shard_count,
        seed=seed,
    )

    remaining = count
    while remaining > 0:
        n = min(block_size, remaining)
        yield rng.random((n, d))
        remaining -= n


def _task_name(shard_index: int) -> str:
    return f"shard_{int(shard_index)}.task"


def _parse_task_index(task_path: Path) -> int | None:
    m = TASK_FILE_RE.match(task_path.name)
    if not m:
        return None
    return int(m.group(1))


def _pick_coprime_step(rng: random.Random, modulo: int) -> int:
    if modulo <= 1:
        return 1
    if modulo == 2:
        return 1

    for _ in range(8):
        step = rng.randrange(1, modulo)
        if math.gcd(step, modulo) == 1:
            return step
    return 1


def _try_claim_task(
    todo_dir: Path,
    processing_dir: Path,
    worker_id: str,
    shard_count: int,
    rng: random.Random,
) -> Path | None:
    if shard_count <= 0:
        return None

    start = rng.randrange(shard_count)
    step = _pick_coprime_step(rng, shard_count)

    for k in range(shard_count):
        shard_index = (start + k * step) % shard_count
        src = todo_dir / _task_name(shard_index)
        dst = processing_dir / f"shard_{shard_index}.{worker_id}.task"
        try:
            os.rename(src, dst)
        except FileNotFoundError:
            continue
        except OSError:
            continue
        return dst
    return None


def _touch(path: Path) -> None:
    try:
        os.utime(path, None)
    except FileNotFoundError:
        return


def _make_heartbeat(task_path: Path, interval_s: float) -> Callable[[bool], None]:
    interval = max(0.1, float(interval_s))
    last_touch = 0.0

    def heartbeat(force: bool = False) -> None:
        nonlocal last_touch
        now = time.time()
        if force or (now - last_touch) >= interval:
            _touch(task_path)
            last_touch = now

    return heartbeat


def _has_task_files(queue_dir: Path) -> bool:
    return any(queue_dir.glob("shard_*.task"))


def _requeue_stale_processing(
    processing_dir: Path,
    todo_dir: Path,
    stale_timeout_s: float,
    max_requeue_per_scan: int,
) -> int:
    if stale_timeout_s <= 0.0:
        return 0

    now = time.time()
    requeued_n = 0
    for task_path in sorted(processing_dir.glob("shard_*.task"), key=lambda p: p.name):
        if requeued_n >= max_requeue_per_scan:
            break

        shard_index = _parse_task_index(task_path)
        if shard_index is None:
            continue

        try:
            age_s = now - float(task_path.stat().st_mtime)
        except FileNotFoundError:
            continue

        if age_s < stale_timeout_s:
            continue

        todo_path = todo_dir / _task_name(shard_index)
        try:
            os.replace(task_path, todo_path)
        except FileNotFoundError:
            continue
        except OSError:
            continue
        requeued_n += 1

    return requeued_n


def _mark_done_task(processing_task_path: Path, done_dir: Path, shard_index: int) -> None:
    done_path = done_dir / _task_name(shard_index)
    try:
        os.replace(processing_task_path, done_path)
    except FileNotFoundError:
        return
    except OSError:
        try:
            processing_task_path.unlink()
        except FileNotFoundError:
            pass
        done_path.touch(exist_ok=True)


def _attempt_file(attempts_dir: Path, shard_index: int) -> Path:
    return attempts_dir / f"shard_{int(shard_index)}.attempt"


def _read_attempt_count(attempts_dir: Path, shard_index: int) -> int:
    path = _attempt_file(attempts_dir, shard_index)
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return 0
    except OSError:
        return 0
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _write_attempt_count(attempts_dir: Path, shard_index: int, count: int) -> None:
    path = _attempt_file(attempts_dir, shard_index)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(str(max(0, int(count))))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _clear_attempt_count(attempts_dir: Path, shard_index: int) -> None:
    path = _attempt_file(attempts_dir, shard_index)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


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


def _evaluate_chunk(
    solver: ForwardPDESolverV6Batch,
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
        seed_fraction=np.array([c.seed_fraction for c in chunk], dtype=float),
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
            seed_fraction=cand.seed_fraction,
            length=float(out.x[-1]),
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
                metrics=dict(scored["metrics"]),
            )
        )

    return rows


def _evaluate_rows_for_shard(
    *,
    solver: ForwardPDESolverV6Batch,
    n_total: int,
    seed: int,
    shard_index: int,
    shard_count: int,
    bounds: Dict[str, tuple[float, float]],
    tp_in: float,
    B: float,
    dx: float,
    velikhov_required: bool,
    reward_config: RewardConfig,
    batch_size: int,
    use_prefilter: bool,
    dte_rel_min: float | None,
    prefilter_mach_inlet_min: float | None,
    prefilter_check_inlet_velikhov: bool,
    heartbeat: Callable[[bool], None] | None,
) -> tuple[List[EvalSummary], Dict[str, int]]:
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
        n_total=n_total,
        shard_index=shard_index,
        shard_count=shard_count,
        seed=seed,
        d=4,
        block_size=batch_size,
    ):
        if heartbeat is not None:
            heartbeat(force=False)

        chunk = [
            _candidate_from_unit(block[i], bounds=bounds, T_p_in=tp_in, B=B)
            for i in range(block.shape[0])
        ]
        if use_prefilter:
            chunk, chunk_stats = _apply_inlet_prefilter(
                solver=solver,
                chunk=chunk,
                dte_rel_min=dte_rel_min,
                mach_inlet_min=prefilter_mach_inlet_min,
                check_inlet_velikhov=prefilter_check_inlet_velikhov,
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
                dx=dx,
                velikhov_required=velikhov_required,
                reward_config=reward_config,
            )
        )

    return rows, prefilter_stats


def _write_rows_atomic(best: List[EvalSummary], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(
        f".{out_path.name}.tmp.{os.getpid()}.{int(time.time() * 1_000_000)}"
    )
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            for row in best:
                f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _run_single_shard(
    *,
    solver: ForwardPDESolverV6Batch,
    args,
    reward_config: RewardConfig,
    bounds: Dict[str, tuple[float, float]],
    shard_index: int,
    shard_count: int,
    out_path: Path,
    batch_size: int,
    use_prefilter: bool,
    dte_rel_min: float | None,
    heartbeat: Callable[[bool], None] | None,
) -> Dict[str, object]:
    rows, prefilter_stats = _evaluate_rows_for_shard(
        solver=solver,
        n_total=int(args.n_total),
        seed=int(args.seed),
        shard_index=int(shard_index),
        shard_count=int(shard_count),
        bounds=bounds,
        tp_in=float(args.tp_in),
        B=float(args.B),
        dx=float(args.dx),
        velikhov_required=(not args.no_velikhov_check),
        reward_config=reward_config,
        batch_size=batch_size,
        use_prefilter=use_prefilter,
        dte_rel_min=dte_rel_min,
        prefilter_mach_inlet_min=(
            None
            if args.prefilter_mach_inlet_min is None
            else float(args.prefilter_mach_inlet_min)
        ),
        prefilter_check_inlet_velikhov=bool(args.prefilter_check_inlet_velikhov),
        heartbeat=heartbeat,
    )

    rows_sorted = sorted(rows, key=lambda r: r.objective)
    k = max(1, int(args.top_k))
    best = rows_sorted[:k]
    _write_rows_atomic(best, out_path)

    feasible_n = sum(1 for r in rows if r.feasible)
    ok_n = sum(1 for r in rows if r.ok)

    return {
        "n_total": int(args.n_total),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "evaluated": int(len(rows)),
        "ok": int(ok_n),
        "feasible": int(feasible_n),
        "saved": int(len(best)),
        "output": str(out_path),
        "batch_size": int(batch_size),
        "dx": float(args.dx),
        "prefilter_enabled": bool(use_prefilter),
        "prefilter_dte_rel_min": (
            None if dte_rel_min is None else float(dte_rel_min)
        ),
        "prefilter_mach_inlet_min": (
            None
            if args.prefilter_mach_inlet_min is None
            else float(args.prefilter_mach_inlet_min)
        ),
        "prefilter_check_inlet_velikhov": bool(args.prefilter_check_inlet_velikhov),
        "prefilter_stats": {k: int(v) for k, v in prefilter_stats.items()},
    }


def _run_task_pool(
    *,
    solver: ForwardPDESolverV6Batch,
    args,
    reward_config: RewardConfig,
    bounds: Dict[str, tuple[float, float]],
    batch_size: int,
    use_prefilter: bool,
    dte_rel_min: float | None,
) -> int:
    out_dir = Path(args.out).parent
    pool_root = Path(args.task_pool_root)
    todo_dir = pool_root / "todo"
    processing_dir = pool_root / "processing"
    done_dir = pool_root / "done"
    failed_dir = pool_root / "failed"
    attempts_dir = pool_root / "attempts"
    todo_dir.mkdir(parents=True, exist_ok=True)
    processing_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir.mkdir(parents=True, exist_ok=True)

    shard_count = max(1, int(args.shard_count))
    poll_interval_s = max(0.1, float(args.task_poll_interval_s))
    stale_timeout_s = float(args.task_stale_timeout_s)
    max_requeue_per_scan = max(1, int(args.task_max_requeue_per_scan))
    max_attempts = max(1, int(args.task_max_attempts))
    worker_id = args.task_worker_id.strip() or (
        f"{os.environ.get('SLURM_JOB_ID', 'noj')}_"
        f"{os.environ.get('SLURM_ARRAY_TASK_ID', 'na')}_"
        f"pid{os.getpid()}"
    )
    rng = random.Random(f"{worker_id}:{os.getpid()}:{time.time_ns()}")

    claimed_n = 0
    completed_n = 0
    failed_n = 0
    failed_permanent_n = 0
    requeued_stale_n = 0
    skipped_existing_n = 0

    print(
        json.dumps(
            {
                "event": "task_pool_start",
                "worker_id": worker_id,
                "pool_root": str(pool_root),
                "shard_count": int(shard_count),
                "poll_interval_s": float(poll_interval_s),
                "stale_timeout_s": float(stale_timeout_s),
                "max_attempts": int(max_attempts),
            },
            ensure_ascii=False,
        )
    )

    while True:
        requeued_now = _requeue_stale_processing(
            processing_dir=processing_dir,
            todo_dir=todo_dir,
            stale_timeout_s=stale_timeout_s,
            max_requeue_per_scan=max_requeue_per_scan,
        )
        requeued_stale_n += int(requeued_now)

        claimed_path = _try_claim_task(
            todo_dir=todo_dir,
            processing_dir=processing_dir,
            worker_id=worker_id,
            shard_count=shard_count,
            rng=rng,
        )
        if claimed_path is None:
            todo_left = _has_task_files(todo_dir)
            processing_left = _has_task_files(processing_dir)
            if (not todo_left) and (not processing_left):
                break
            sleep_s = poll_interval_s * (1.0 + 0.2 * rng.random())
            time.sleep(sleep_s)
            continue

        shard_index = _parse_task_index(claimed_path)
        if shard_index is None:
            try:
                claimed_path.unlink()
            except FileNotFoundError:
                pass
            continue

        claimed_n += 1
        out_path = out_dir / f"sweep_shard_{shard_index}.jsonl"

        try:
            out_exists = out_path.exists()
        except OSError:
            out_exists = False

        if out_exists:
            skipped_existing_n += 1
            _clear_attempt_count(attempts_dir, shard_index)
            _mark_done_task(
                processing_task_path=claimed_path,
                done_dir=done_dir,
                shard_index=shard_index,
            )
            print(
                json.dumps(
                    {
                        "event": "task_skip_existing_output",
                        "worker_id": worker_id,
                        "shard_index": int(shard_index),
                        "output": str(out_path),
                    },
                    ensure_ascii=False,
                )
            )
            continue

        heartbeat = _make_heartbeat(
            task_path=claimed_path,
            interval_s=float(args.task_heartbeat_interval_s),
        )
        heartbeat(force=True)
        started_at = time.time()
        try:
            summary = _run_single_shard(
                solver=solver,
                args=args,
                reward_config=reward_config,
                bounds=bounds,
                shard_index=int(shard_index),
                shard_count=int(shard_count),
                out_path=out_path,
                batch_size=batch_size,
                use_prefilter=use_prefilter,
                dte_rel_min=dte_rel_min,
                heartbeat=heartbeat,
            )
        except Exception as exc:
            failed_n += 1
            attempts = _read_attempt_count(attempts_dir, shard_index) + 1
            _write_attempt_count(attempts_dir, shard_index, attempts)
            if attempts >= max_attempts:
                failed_permanent_n += 1
                failed_task_path = failed_dir / _task_name(shard_index)
                try:
                    os.replace(claimed_path, failed_task_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    try:
                        claimed_path.unlink()
                    except FileNotFoundError:
                        pass
                    failed_task_path.touch(exist_ok=True)
                print(
                    json.dumps(
                        {
                            "event": "task_failed_permanent",
                            "worker_id": worker_id,
                            "shard_index": int(shard_index),
                            "attempts": int(attempts),
                            "max_attempts": int(max_attempts),
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                )
            else:
                todo_path = todo_dir / _task_name(shard_index)
                try:
                    os.replace(claimed_path, todo_path)
                except FileNotFoundError:
                    pass
                print(
                    json.dumps(
                        {
                            "event": "task_failed_requeued",
                            "worker_id": worker_id,
                            "shard_index": int(shard_index),
                            "attempts": int(attempts),
                            "max_attempts": int(max_attempts),
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                )
            if args.task_fail_fast:
                raise
            continue

        elapsed_s = time.time() - started_at
        _mark_done_task(
            processing_task_path=claimed_path,
            done_dir=done_dir,
            shard_index=shard_index,
        )
        _clear_attempt_count(attempts_dir, shard_index)
        completed_n += 1

        summary["mode"] = "task_pool"
        summary["worker_id"] = worker_id
        summary["elapsed_s"] = float(elapsed_s)
        print(json.dumps(summary, ensure_ascii=False))

    print(
        json.dumps(
            {
                "event": "task_pool_exit",
                "worker_id": worker_id,
                "claimed": int(claimed_n),
                "completed": int(completed_n),
                "failed": int(failed_n),
                "failed_permanent": int(failed_permanent_n),
                "requeued_stale": int(requeued_stale_n),
                "skipped_existing": int(skipped_existing_n),
            },
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    p = _build_parser()
    p.description = "V6 cluster sweep worker (batch fixed-step)"
    p.add_argument("--batch-size", type=int, default=256, help="number of candidates per batch solve")
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
    p.add_argument(
        "--task-pool-root",
        type=str,
        default="",
        help="enable pull-based task pool with root directory containing todo/processing/done",
    )
    p.add_argument(
        "--task-poll-interval-s",
        type=float,
        default=1.0,
        help="sleep duration while waiting for claimable tasks",
    )
    p.add_argument(
        "--task-stale-timeout-s",
        type=float,
        default=900.0,
        help="requeue processing tasks older than this many seconds; <=0 disables stale requeue",
    )
    p.add_argument(
        "--task-heartbeat-interval-s",
        type=float,
        default=15.0,
        help="touch claimed task file every N seconds while computing",
    )
    p.add_argument(
        "--task-max-requeue-per-scan",
        type=int,
        default=4,
        help="upper bound of stale processing tasks requeued per scan",
    )
    p.add_argument(
        "--task-max-attempts",
        type=int,
        default=3,
        help="max retries per shard in task-pool mode before marking failed/",
    )
    p.add_argument(
        "--task-fail-fast",
        action="store_true",
        help="exit non-zero on first shard exception in task-pool mode",
    )
    p.add_argument(
        "--task-worker-id",
        type=str,
        default="",
        help="optional identifier embedded in processing lock filenames",
    )

    args = p.parse_args()
    reward_config = _reward_config_from_args(args)

    bounds = {
        "n_p_in": (float(args.np_min), float(args.np_max)),
        "Z_in": (float(args.z_min), float(args.z_max)),
        "T_e_in": (float(args.te_min), float(args.te_max)),
        "seed_fraction": (float(args.seedf_min), float(args.seedf_max)),
    }
    for key, (lo, hi) in bounds.items():
        if lo <= 0.0 and key in ("n_p_in", "seed_fraction"):
            raise ValueError(f"{key} must be positive")
        if hi <= lo:
            raise ValueError(f"{key} bounds invalid: {lo}, {hi}")

    solver = ForwardPDESolverV6Batch(B=float(args.B), length=float(args.L))
    batch_size = max(1, int(args.batch_size))
    dte_rel_min = None if args.no_prefilter_dte_rel else (
        float(args.prefilter_dte_rel_min)
        if args.prefilter_dte_rel_min is not None
        else (0.01 / float(args.L))
    )
    use_prefilter = (
        dte_rel_min is not None
        or args.prefilter_mach_inlet_min is not None
        or bool(args.prefilter_check_inlet_velikhov)
    )

    if args.task_pool_root:
        return _run_task_pool(
            solver=solver,
            args=args,
            reward_config=reward_config,
            bounds=bounds,
            batch_size=batch_size,
            use_prefilter=use_prefilter,
            dte_rel_min=dte_rel_min,
        )

    summary = _run_single_shard(
        solver=solver,
        args=args,
        reward_config=reward_config,
        bounds=bounds,
        shard_index=int(args.shard_index),
        shard_count=max(1, int(args.shard_count)),
        out_path=Path(args.out),
        batch_size=batch_size,
        use_prefilter=use_prefilter,
        dte_rel_min=dte_rel_min,
        heartbeat=None,
    )
    summary["mode"] = "single_shard"
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
