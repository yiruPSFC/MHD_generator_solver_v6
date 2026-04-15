#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_global_marginal.global_postprocess_v6 import (
    compute_global_metrics,
    required_mhd_output_power,
    trim_profile_to_mhd_target,
)
from v6_global_marginal.pde_solver_v6_batch_global import ForwardPDESolverV6BatchGlobal, event_name_from_code


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one V6 global-design reference case")
    p.add_argument("--np-in", type=float, required=True)
    p.add_argument("--z-in", type=float, required=True)
    p.add_argument("--tp-in", type=float, required=True)
    p.add_argument("--te-in", type=float, required=True)
    p.add_argument("--A-in", type=float, required=True, help="inlet cross-section area [m^2]")
    p.add_argument("--B", type=float, default=10.2)
    p.add_argument("--L", type=float, default=5.4)
    p.add_argument("--dx", type=float, default=5e-3)
    p.add_argument("--mach-low", type=float, default=0.99)
    p.add_argument("--mach-high", type=float, default=1.01)
    p.add_argument("--furnace-power-MW", type=float, default=363.0)
    p.add_argument("--steam-efficiency", type=float, default=0.35)
    p.add_argument(
        "--total-plant-power-target",
        type=float,
        default=None,
        help="if set, integrate to --L then trim the profile to the length needed to hit this plant output target [MWe]",
    )
    p.add_argument("--out-json", type=str, default="", help="optional metrics json path")
    p.add_argument("--out-npz", type=str, default="", help="optional profile npz path")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    solver = ForwardPDESolverV6BatchGlobal(B=float(args.B), length=float(args.L))
    out = solver.solve_batch(
        n_p_in=np.array([args.np_in], dtype=float),
        Z_in=np.array([args.z_in], dtype=float),
        T_p_in=np.array([args.tp_in], dtype=float),
        T_e_in=np.array([args.te_in], dtype=float),
        A_in=np.array([args.A_in], dtype=float),
        dx=float(args.dx),
        mach_low=float(args.mach_low),
        mach_high=float(args.mach_high),
        store_profiles=True,
    )

    idx_last = int(out.valid_points[0]) - 1
    if (not bool(out.success[0])) or idx_last < 0:
        payload = {
            "ok": False,
            "event": event_name_from_code(int(out.event_code[0])),
            "valid_points": int(out.valid_points[0]),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    x = out.x[: idx_last + 1]
    profile = {
        "x": x,
        "n_p": out.n_p[0, : idx_last + 1],
        "T_e": out.T_e[0, : idx_last + 1],
        "T_p": out.T_p[0, : idx_last + 1],
        "A": out.A[0, : idx_last + 1],
        "v_p": out.v_p[0, : idx_last + 1],
        "n_e": out.n_e[0, : idx_last + 1],
        "beta": out.beta[0, : idx_last + 1],
        "eta": out.eta[0, : idx_last + 1],
        "Z": out.Z[0, : idx_last + 1],
        "J_x": out.J_x[0, : idx_last + 1],
        "J_y": out.J_y[0, : idx_last + 1],
        "E_x": out.E_x[0, : idx_last + 1],
        "mach": out.mach[0, : idx_last + 1],
        "velikhov_margin": out.velikhov_margin[0, : idx_last + 1],
    }
    if args.total_plant_power_target is not None:
        target_mhd = required_mhd_output_power(
            total_plant_power_MWe=float(args.total_plant_power_target),
            furnace_power_MW=float(args.furnace_power_MW),
            steam_cycle_efficiency=float(args.steam_efficiency),
        )
        trimmed = trim_profile_to_mhd_target(
            target_mhd_output_MWe=target_mhd,
            x=profile["x"],
            A=profile["A"],
            J_x=profile["J_x"],
            E_x=profile["E_x"],
            n_p=profile["n_p"],
            T_e=profile["T_e"],
            T_p=profile["T_p"],
            v_p=profile["v_p"],
            n_e=profile["n_e"],
            beta=profile["beta"],
            eta=profile["eta"],
            Z=profile["Z"],
            J_y=profile["J_y"],
            mach=profile["mach"],
            velikhov_margin=profile["velikhov_margin"],
        )
        if trimmed is None:
            payload = {
                "ok": False,
                "event": event_name_from_code(int(out.event_code[0])),
                "reason": "insufficient_mhd_power_before_profile_end",
                "valid_points": int(out.valid_points[0]),
                "target_mhd_output_MWe": float(target_mhd),
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 1
        profile = trimmed

    metrics = compute_global_metrics(
        x=profile["x"],
        A=profile["A"],
        v_p=profile["v_p"],
        eta=profile["eta"],
        J_x=profile["J_x"],
        J_y=profile["J_y"],
        E_x=profile["E_x"],
        B=float(args.B),
        velikhov_margin=profile["velikhov_margin"],
        furnace_power_MW=float(args.furnace_power_MW),
        steam_cycle_efficiency=float(args.steam_efficiency),
    )

    payload = {
        "ok": True,
        "event": event_name_from_code(int(out.event_code[0])),
        "reached_end": bool(out.reached_end[0]),
        "valid_points": int(out.valid_points[0]),
        "step_size": float(out.step_size),
        "projected_seed_fraction": float(out.seed_fraction[0]),
        "metrics": metrics.to_dict(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.out_npz:
        out_path = Path(args.out_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            x=profile["x"],
            n_p=profile["n_p"],
            T_e=profile["T_e"],
            T_p=profile["T_p"],
            A=profile["A"],
            v_p=profile["v_p"],
            n_e=profile["n_e"],
            beta=profile["beta"],
            eta=profile["eta"],
            Z=profile["Z"],
            J_x=profile["J_x"],
            J_y=profile["J_y"],
            E_x=profile["E_x"],
            mach=profile["mach"],
            velikhov_margin=profile["velikhov_margin"],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
