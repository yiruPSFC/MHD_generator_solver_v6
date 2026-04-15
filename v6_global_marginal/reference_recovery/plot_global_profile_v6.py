#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from v6_global_marginal.reference_recovery.global_plotting_v6 import plot_global_results_v6


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a saved V6 global profile NPZ")
    parser.add_argument("--npz", type=str, required=True, help="profile npz created by a V6 global runner")
    parser.add_argument("--out", type=str, required=True, help="output figure path")
    parser.add_argument("--B", type=float, default=None, help="override magnetic field [T]")
    parser.add_argument("--seed-fraction", type=float, default=None, help="override seed fraction")
    parser.add_argument("--title", type=str, default="", help="optional figure title")
    parser.add_argument("--furnace-power-MW", type=float, default=None)
    parser.add_argument("--steam-efficiency", type=float, default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    data = np.load(Path(args.npz))

    B = args.B
    if B is None:
        if "B" not in data:
            raise ValueError("NPZ does not contain B; pass --B explicitly")
        B = float(np.asarray(data["B"]).ravel()[0])

    seed_fraction = args.seed_fraction
    if seed_fraction is None:
        if "seed_fraction" not in data:
            raise ValueError("NPZ does not contain seed_fraction; pass --seed-fraction explicitly")
        seed_fraction = float(np.asarray(data["seed_fraction"]).ravel()[0])

    profile = {
        "x": np.asarray(data["x"], dtype=float),
        "n_p": np.asarray(data["n_p"], dtype=float),
        "T_e": np.asarray(data["T_e"], dtype=float),
        "T_p": np.asarray(data["T_p"], dtype=float),
        "A": np.asarray(data["A"], dtype=float),
        "v_p": np.asarray(data["v_p"], dtype=float),
        "n_e": np.asarray(data["n_e"], dtype=float),
        "beta": np.asarray(data["beta"], dtype=float),
        "eta": np.asarray(data["eta"], dtype=float),
        "Z": np.asarray(data["Z"], dtype=float),
        "J_x": np.asarray(data["J_x"], dtype=float),
        "J_y": np.asarray(data["J_y"], dtype=float),
        "E_x": np.asarray(data["E_x"], dtype=float),
        "mach": np.asarray(data["mach"], dtype=float),
        "velikhov_margin": np.asarray(data["velikhov_margin"], dtype=float),
    }

    stats = plot_global_results_v6(
        profile,
        args.out,
        B=float(B),
        seed_fraction=float(seed_fraction),
        title=args.title,
        furnace_power_MW=args.furnace_power_MW,
        steam_cycle_efficiency=args.steam_efficiency,
    )
    for key, value in stats.items():
        print(f"{key}: {value:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
