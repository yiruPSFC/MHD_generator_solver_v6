#!/usr/bin/env python3
from __future__ import annotations

"""
Scalar (non-batch) fallback sweep worker.

Shared sampling/reward/parser logic is owned by cluster_sweep_worker_batch.py.
This file keeps only the scalar-solver execution path.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from cluster_sweep_worker_batch import (  # noqa: E402
    Candidate,
    EvalSummary,
    RewardConfig,
    _build_parser,
    _candidate_from_unit,
    _reward_config_from_args,
    _score_profile,
    _unit_sample_blocks,
)
from non_batch.pde_solver_v6 import ForwardPDESolverV6  # noqa: E402


def evaluate_candidate(
    c: Candidate,
    L: float,
    velikhov_required: bool = True,
    reward_config: RewardConfig | None = None,
) -> EvalSummary:
    try:
        reward_config = reward_config or RewardConfig()
        solver = ForwardPDESolverV6(B=c.B, length=L)
        out = solver.solve(
            n_p_in=c.n_p_in,
            Z_in=c.Z_in,
            T_p_in=c.T_p_in,
            T_e_in=c.T_e_in,
            seed_fraction=c.seed_fraction,
            rtol=1e-6,
            atol=1e-8,
        )

        if not out.success or len(out.x) == 0:
            return EvalSummary(
                ok=False,
                feasible=False,
                reason=f"ivp_failed:{out.message}",
                objective=1e30,
                outlet_te=np.nan,
                outlet_tp=np.nan,
                outlet_ratio=np.nan,
                min_velikhov_margin=np.nan,
                event=out.event_name,
                params=asdict(c),
                metrics={},
            )

        scored = _score_profile(
            x=out.x,
            T_e=out.T_e,
            T_p=out.T_p,
            Z=out.Z,
            mach=out.mach,
            beta=out.beta,
            n_e=out.n_e,
            n_p=out.n_p,
            seed_fraction=c.seed_fraction,
            length=L,
            reached_end=(
                float(out.x[-1]) >= float(L) * (1.0 - reward_config.truncation_tol)
            ),
            velikhov_required=velikhov_required,
            reward_config=reward_config,
        )

        return EvalSummary(
            ok=True,
            feasible=bool(scored["feasible"]),
            reason=str(scored["reason"]),
            objective=float(scored["objective"]),
            outlet_te=float(scored["outlet_te"]),
            outlet_tp=float(scored["outlet_tp"]),
            outlet_ratio=float(scored["outlet_ratio"]),
            min_velikhov_margin=float(scored["min_velikhov_margin"]),
            event=out.event_name,
            params=asdict(c),
            metrics=dict(scored["metrics"]),
        )

    except Exception as exc:
        return EvalSummary(
            ok=False,
            feasible=False,
            reason=f"exception:{exc.__class__.__name__}:{exc}",
            objective=1e30,
            outlet_te=np.nan,
            outlet_tp=np.nan,
            outlet_ratio=np.nan,
            min_velikhov_margin=np.nan,
            event=None,
            params=asdict(c),
            metrics={},
        )


def main() -> int:
    args = _build_parser().parse_args()
    reward_config = _reward_config_from_args(args)

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

    rows: List[EvalSummary] = []
    for block in _unit_sample_blocks(
        n_total=int(args.n_total),
        shard_index=int(args.shard_index),
        shard_count=int(args.shard_count),
        seed=int(args.seed),
        d=4,
        block_size=4096,
    ):
        for i in range(block.shape[0]):
            cand = _candidate_from_unit(
                block[i],
                bounds=bounds,
                T_p_in=float(args.tp_in),
                B=float(args.B),
            )
            res = evaluate_candidate(
                cand,
                L=float(args.L),
                velikhov_required=(not args.no_velikhov_check),
                reward_config=reward_config,
            )
            rows.append(res)

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
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
