#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_casadi_v2.optimize_area_profile_casadi_v2 import (
    _design_value_weights_lab_poc_v2_objective,
    _evaluate_inlet_design_numeric,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate the v6_casadi_v2 experimental objective on a saved profile npz."
    )
    p.add_argument("--profile-npz", type=str, required=True)
    p.add_argument("--B", type=float, default=10.2)
    p.add_argument("--seed-fraction", type=float, default=1e-4)
    p.add_argument("--np-in", type=float, default=None)
    p.add_argument("--te-in", type=float, default=None)
    p.add_argument("--z-in", type=float, default=None)
    p.add_argument("--jx-in", type=float, default=None)
    p.add_argument(
        "--normalize-inlet-area",
        action="store_true",
        help="evaluate external profiles in the v2 A_in=1 intensity convention",
    )
    p.add_argument("--out-json", type=str, default="")
    return p


def _finite_summary(values: np.ndarray) -> dict[str, object]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "size": int(arr.size),
            "finite_count": 0,
            "min": float("nan"),
            "max": float("nan"),
            "first": float("nan"),
            "last": float("nan"),
        }
    return {
        "size": int(arr.size),
        "finite_count": int(finite.size),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "first": float(arr[0]) if arr.size else float("nan"),
        "last": float(arr[-1]) if arr.size else float("nan"),
    }


def _dual_summary(data: np.lib.npyio.NpzFile) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for key in sorted(name for name in data.files if name.startswith("dual_")):
        arr = np.asarray(data[key], dtype=float).reshape(-1)
        finite = arr[np.isfinite(arr)]
        name = key.removeprefix("dual_")
        if finite.size == 0:
            out[name] = {
                "size": int(arr.size),
                "finite_count": 0,
                "max_abs": float("nan"),
                "max_abs_index": -1,
                "l1": float("nan"),
                "active_fraction_gt_1e_8": float("nan"),
            }
            continue
        out[name] = {
            "size": int(arr.size),
            "finite_count": int(finite.size),
            "max_abs": float(np.nanmax(np.abs(arr))),
            "max_abs_index": int(np.nanargmax(np.abs(arr))),
            "l1": float(np.nansum(np.abs(arr))),
            "active_fraction_gt_1e_8": float(np.mean(np.abs(finite) > 1e-8)),
        }
    return out


def evaluate_profile_objective(
    *,
    profile_npz: str | Path,
    B: float,
    seed_fraction: float,
    n_p_in: float | None = None,
    T_e_in: float | None = None,
    Z_in: float | None = None,
    J_x_in: float | None = None,
    normalize_inlet_area: bool = False,
) -> dict[str, object]:
    path = Path(profile_npz)
    with np.load(path) as data:
        required = ["x", "n_p", "T_e", "T_p", "A", "n_e", "mach", "J_x", "E_x"]
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"profile npz missing required arrays: {missing}")
        x = np.asarray(data["x"], dtype=float).reshape(-1)
        n_p = np.asarray(data["n_p"], dtype=float).reshape(-1)
        T_e = np.asarray(data["T_e"], dtype=float).reshape(-1)
        T_p = np.asarray(data["T_p"], dtype=float).reshape(-1)
        A = np.asarray(data["A"], dtype=float).reshape(-1)
        n_e = np.asarray(data["n_e"], dtype=float).reshape(-1)
        mach = np.asarray(data["mach"], dtype=float).reshape(-1)
        J_x = np.asarray(data["J_x"], dtype=float).reshape(-1)
        E_x = np.asarray(data["E_x"], dtype=float).reshape(-1)
        Z = np.asarray(data["Z"], dtype=float).reshape(-1) if "Z" in data.files else np.zeros(0, dtype=float)
        duals = _dual_summary(data)

    sizes = {arr.size for arr in (x, n_p, T_e, T_p, A, n_e, mach, J_x, E_x)}
    if len(sizes) != 1:
        raise ValueError("all profile arrays must have the same node length.")
    if x.size < 2:
        raise ValueError("profile must contain at least two nodes.")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("profile x must be strictly increasing.")
    if not all(np.all(np.isfinite(arr)) for arr in (x, n_p, T_e, T_p, A, n_e, mach, J_x, E_x)):
        raise ValueError("profile arrays must be finite.")

    inlet_np = float(n_p[0] if n_p_in is None else n_p_in)
    inlet_te = float(T_e[0] if T_e_in is None else T_e_in)
    inlet_z = float((Z[0] if Z.size else 0.0) if Z_in is None else Z_in)
    inlet_jx = float(J_x[0] if J_x_in is None else J_x_in)
    area_scale = max(float(A[0]), 1e-30)
    A_eval = A / area_scale if normalize_inlet_area else A
    inlet_I0 = inlet_jx if normalize_inlet_area else inlet_jx * area_scale
    inlet = _evaluate_inlet_design_numeric(
        n_p_in=inlet_np,
        T_e_in=inlet_te,
        Z_in=inlet_z,
        I_0=inlet_I0,
        seed_fraction=float(seed_fraction),
        B=float(B),
    )

    weights = _design_value_weights_lab_poc_v2_objective()
    outlet_delta_te_per_kK = (float(T_e[-1]) - float(inlet.T_e)) / 1e3
    outlet_delta_ratio = float(T_e[-1]) / max(float(T_p[-1]), 1.0) - 1.0
    outlet_f_ion = float(n_e[-1]) / max(float(seed_fraction) * float(n_p[-1]), 1e-30)
    power_density_samples = -A_eval * J_x * E_x / 1e8
    outlet_mhd_output_per_100MWe = float(np.trapezoid(power_density_samples, x))
    inlet_delta_ratio = float(inlet.T_e) / max(float(inlet.T_p), 1.0) - 1.0
    device_length_per_5m = float(x[-1] - x[0]) / 5.0

    contributions = {
        "outlet_delta_te_per_kK": float(weights.outlet_delta_te_per_kK) * outlet_delta_te_per_kK,
        "outlet_delta_ratio": float(weights.outlet_delta_ratio) * outlet_delta_ratio,
        "outlet_f_ion": float(weights.outlet_f_ion) * outlet_f_ion,
        "outlet_mhd_output_per_100MWe": float(weights.outlet_mhd_output_per_100MWe)
        * outlet_mhd_output_per_100MWe,
        "inlet_delta_ratio_penalty": -float(weights.inlet_delta_ratio_penalty) * inlet_delta_ratio,
        "inlet_mach_penalty": -float(weights.inlet_mach_penalty) * float(inlet.mach),
        "magnetic_field_T_penalty": -float(weights.magnetic_field_T_penalty) * abs(float(B)),
        "device_length_per_5m_penalty": -float(weights.device_length_per_5m_penalty) * device_length_per_5m,
    }
    score = float(sum(contributions.values()))
    raw_terms = {
        "outlet_delta_te_per_kK": float(outlet_delta_te_per_kK),
        "outlet_delta_ratio": float(outlet_delta_ratio),
        "outlet_f_ion": float(outlet_f_ion),
        "outlet_mhd_output_per_100MWe": float(outlet_mhd_output_per_100MWe),
        "inlet_delta_ratio": float(inlet_delta_ratio),
        "inlet_mach": float(inlet.mach),
        "magnetic_field_T": float(B),
        "device_length_per_5m": float(device_length_per_5m),
    }
    return {
        "profile_npz": str(path),
        "objective": "lab_poc_v2_objective",
        "normalize_inlet_area": bool(normalize_inlet_area),
        "score": score,
        "raw_terms": raw_terms,
        "weighted_contributions": contributions,
        "weights": weights.to_dict(),
        "inlet_design": {
            "n_p_in": float(inlet.n_p),
            "T_e_in": float(inlet.T_e),
            "T_p_in": float(inlet.T_p),
            "Z_in": float(inlet.Z),
            "J_x_in": float(inlet.J_x),
            "I_0": float(inlet.I_0),
            "seed_fraction": float(seed_fraction),
            "mach_in": float(inlet.mach),
            "velikhov_margin_in": float(inlet.velikhov_margin),
        },
        "profile_summary": {
            "x": _finite_summary(x),
            "n_p": _finite_summary(n_p),
            "T_e": _finite_summary(T_e),
            "T_p": _finite_summary(T_p),
            "A": _finite_summary(A),
            "A_eval": _finite_summary(A_eval),
            "mach": _finite_summary(mach),
        },
        "dual_summary": duals,
        "finite": bool(np.isfinite(score) and all(np.isfinite(v) for v in raw_terms.values())),
    }


def main() -> int:
    args = _build_parser().parse_args()
    payload = evaluate_profile_objective(
        profile_npz=args.profile_npz,
        B=float(args.B),
        seed_fraction=float(args.seed_fraction),
        n_p_in=None if args.np_in is None else float(args.np_in),
        T_e_in=None if args.te_in is None else float(args.te_in),
        Z_in=None if args.z_in is None else float(args.z_in),
        J_x_in=None if args.jx_in is None else float(args.jx_in),
        normalize_inlet_area=bool(args.normalize_inlet_area),
    )
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
