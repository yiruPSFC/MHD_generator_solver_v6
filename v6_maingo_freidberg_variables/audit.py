from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .algebra import (
    bridge_diagnostics,
    profile_to_freidberg_arrays,
    reconstruct_profile_from_hl_arrays,
)
from .models import FreidbergConfig, PrimitivePoint
from .rhs import freidberg_rhs_arrays


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(np.asarray(y, dtype=float), np.asarray(x, dtype=float)))


def _load_original_points(profile_path: Path) -> list[PrimitivePoint]:
    with np.load(profile_path) as data:
        return [PrimitivePoint.from_npz(data, idx) for idx in range(np.asarray(data["x"], dtype=float).size)]


def audit_profile(
    *,
    profile_path: str | Path,
    summary_path: str | Path,
    bridge_tolerance_K: float = 1e-4,
) -> dict[str, Any]:
    profile_path = Path(profile_path).resolve()
    summary_path = Path(summary_path).resolve()
    config = FreidbergConfig.from_summary_and_profile(summary_path, profile_path)
    arrays = profile_to_freidberg_arrays(profile_path, config)
    original = _load_original_points(profile_path)
    reconstructed, residuals = reconstruct_profile_from_hl_arrays(
        H_p=arrays["H_p"],
        L_p=arrays["L_p"],
        T_e=arrays["T_e"],
        x=arrays["x"],
        config=config,
        mach_hints=arrays["mach"],
        branch="any",
        tolerance_K=bridge_tolerance_K,
    )
    bridge = bridge_diagnostics(original, reconstructed, residuals)
    rhs = freidberg_rhs_arrays(original, config)
    x = arrays["x"]
    H_p = arrays["H_p"]
    L_p = arrays["L_p"]
    dHdx = rhs["dHdx"]
    dLdx = rhs["dLdx"]
    A0 = config.inlet_area_m2
    with np.load(profile_path) as data:
        A = np.asarray(data["A"], dtype=float)
        J_x = np.asarray(data["J_x"], dtype=float)
        E_x = np.asarray(data["E_x"], dtype=float)
        eta = np.asarray(data["eta"], dtype=float)
        J_y = np.asarray(data["J_y"], dtype=float)
        v_p = np.asarray(data["v_p"], dtype=float)
    J2 = np.asarray([point.J_x * point.J_x + point.J_y * point.J_y for point in original], dtype=float)
    interval_H_rhs_MW = 0.5 * np.diff(x) * (dHdx[:-1] + dHdx[1:]) * A0 / 1e6
    interval_L_rhs = 0.5 * np.diff(x) * (dLdx[:-1] + dLdx[1:])
    H_delta_MW = float((H_p[-1] - H_p[0]) * A0 / 1e6)
    H_rhs_integral_MW = float(_trapz(dHdx, x) * A0 / 1e6)
    L_delta = float(L_p[-1] - L_p[0])
    L_rhs_integral = _trapz(dLdx, x)
    mhd_output_MW = float(_trapz(-A * J_x * E_x, x) / 1e6)
    joule_MW = float(_trapz(A * eta * J2, x) / 1e6)
    lorentz_MW = float(_trapz(A * v_p * J_y * config.B_T, x) / 1e6)
    inlet_primary_enthalpy_MW = float(H_p[0] * A0 / 1e6)

    return {
        "profile_path": str(profile_path),
        "summary_path": str(summary_path),
        "config": config.to_dict(),
        "n_points": int(x.size),
        "mach_min": float(np.nanmin(arrays["mach"])),
        "mach_max": float(np.nanmax(arrays["mach"])),
        "mhd_output_MW": mhd_output_MW,
        "inlet_primary_enthalpy_MW": inlet_primary_enthalpy_MW,
        "reported_style_primary_extraction_percent": float(100.0 * mhd_output_MW / max(inlet_primary_enthalpy_MW, 1e-300)),
        "bridge": bridge.to_dict(),
        "freidberg_H": {
            "delta_MW": H_delta_MW,
            "rhs_integral_MW": H_rhs_integral_MW,
            "residual_MW": H_delta_MW - H_rhs_integral_MW,
            "max_interval_residual_MW": float(np.nanmax(np.abs(np.diff(H_p) * A0 / 1e6 - interval_H_rhs_MW))),
        },
        "freidberg_L": {
            "delta": L_delta,
            "rhs_integral": L_rhs_integral,
            "residual": L_delta - L_rhs_integral,
            "max_interval_residual": float(np.nanmax(np.abs(np.diff(L_p) - interval_L_rhs))),
        },
        "power_terms_MW": {
            "lorentz_integral_A_vJyB": lorentz_MW,
            "joule_integral_A_etaJ2": joule_MW,
            "lorentz_plus_joule": lorentz_MW + joule_MW,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a stored primitive profile through the Freidberg H_p/L_p bridge."
    )
    parser.add_argument("profile", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bridge-tolerance-K", type=float, default=1e-4)
    args = parser.parse_args()
    report = audit_profile(
        profile_path=args.profile,
        summary_path=args.summary,
        bridge_tolerance_K=float(args.bridge_tolerance_K),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
