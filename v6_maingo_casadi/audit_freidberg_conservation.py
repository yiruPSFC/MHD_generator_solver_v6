from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


K_B = 1.380649e-23
AMU_KG = 1.66053906660e-27


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _summary_defaults(summary_path: Path | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "B_T": None,
        "area_scale_m2": None,
        "heavy_particle_mass_kg": None,
    }
    if summary_path is None:
        return defaults
    payload = _load_json(summary_path)
    baseline = dict(payload.get("baseline_seed", {}) or {})
    fluid = dict(payload.get("working_fluid_profile", {}) or baseline.get("working_fluid", {}) or {})
    defaults["B_T"] = payload.get("B", baseline.get("B"))
    defaults["area_scale_m2"] = baseline.get("area_scale_m2")
    defaults["heavy_particle_mass_kg"] = fluid.get("heavy_particle_mass_kg")
    return defaults


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(np.asarray(y, dtype=float), np.asarray(x, dtype=float)))


def audit_profile(
    *,
    profile_path: Path,
    summary_path: Path | None,
    B_T: float | None,
    area_scale_m2: float | None,
    heavy_particle_mass_kg: float | None,
) -> dict[str, Any]:
    defaults = _summary_defaults(summary_path)
    B = float(B_T if B_T is not None else defaults["B_T"])
    mp = float(
        heavy_particle_mass_kg
        if heavy_particle_mass_kg is not None
        else defaults["heavy_particle_mass_kg"]
    )
    with np.load(profile_path) as data:
        x = np.asarray(data["x"], dtype=float)
        A = np.asarray(data["A"], dtype=float)
        n_p = np.asarray(data["n_p"], dtype=float)
        T_p = np.asarray(data["T_p"], dtype=float)
        v_p = np.asarray(data["v_p"], dtype=float)
        eta = np.asarray(data["eta"], dtype=float)
        J_x = np.asarray(data["J_x"], dtype=float)
        J_y = np.asarray(data["J_y"], dtype=float)
        E_x = np.asarray(data["E_x"], dtype=float)
        n_e = np.asarray(data["n_e"], dtype=float) if "n_e" in data else np.zeros_like(n_p)
        T_e = np.asarray(data["T_e"], dtype=float) if "T_e" in data else np.zeros_like(n_p)

    A0 = float(area_scale_m2 if area_scale_m2 is not None else defaults["area_scale_m2"] or A[0])
    J2 = J_x * J_x + J_y * J_y
    M2 = 3.0 * v_p * v_p / (5.0 * K_B * T_p / mp)
    M = np.sqrt(np.maximum(M2, 0.0))
    p_p = n_p * K_B * T_p

    # Freidberg slide 38. H_p is primary stagnation enthalpy flux per inlet area.
    H_p = (A * n_p * v_p / A0) * (2.5 * K_B * T_p + 0.5 * mp * v_p * v_p)
    L_p = M / np.maximum((M2 + 3.0) ** 2, 1e-300) * (A / A0)

    # Freidberg slide 39, using the signed J_y convention stored in the profile.
    rhs_H = (A / A0) * (v_p * J_y * B + eta * J2)
    rhs_L = (
        -(12.0 / 5.0)
        * L_p
        / np.maximum((M2 + 3.0) * p_p * v_p, 1e-300)
        * (v_p * J_y * B - ((5.0 * M2 + 3.0) / 12.0) * eta * J2)
    )

    H_delta_MW = float((H_p[-1] - H_p[0]) * A0 / 1e6)
    H_rhs_integral_MW = float(_trapz(rhs_H, x) * A0 / 1e6)
    L_delta = float(L_p[-1] - L_p[0])
    L_rhs_integral = _trapz(rhs_L, x)

    dx = np.diff(x)
    interval_H_rhs_MW = 0.5 * dx * (rhs_H[:-1] + rhs_H[1:]) * A0 / 1e6
    interval_L_rhs = 0.5 * dx * (rhs_L[:-1] + rhs_L[1:])

    inlet_primary_enthalpy_MW = float(H_p[0] * A0 / 1e6)
    outlet_primary_enthalpy_MW = float(H_p[-1] * A0 / 1e6)
    inlet_stagnation_enthalpy_with_electrons_MW = float(
        A[0]
        * v_p[0]
        * (2.5 * K_B * (n_p[0] * T_p[0] + n_e[0] * T_e[0]) + 0.5 * mp * n_p[0] * v_p[0] ** 2)
        / 1e6
    )
    mhd_output_MW = float(_trapz(-A * J_x * E_x, x) / 1e6)
    joule_heating_MW = float(_trapz(A * eta * J2, x) / 1e6)
    lorentz_power_MW = float(_trapz(A * v_p * J_y * B, x) / 1e6)

    return {
        "profile_path": str(profile_path),
        "summary_path": None if summary_path is None else str(summary_path),
        "B_T": B,
        "area_scale_m2": A0,
        "heavy_particle_mass_kg": mp,
        "n_points": int(x.size),
        "mach_min": float(np.nanmin(M)),
        "mach_max": float(np.nanmax(M)),
        "mhd_output_MW": mhd_output_MW,
        "inlet_primary_enthalpy_MW": inlet_primary_enthalpy_MW,
        "outlet_primary_enthalpy_MW": outlet_primary_enthalpy_MW,
        "inlet_stagnation_enthalpy_with_electrons_MW": inlet_stagnation_enthalpy_with_electrons_MW,
        "reported_style_enthalpy_extraction_percent": float(
            100.0 * mhd_output_MW / max(inlet_stagnation_enthalpy_with_electrons_MW, 1e-300)
        ),
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
            "lorentz_integral_A_vJyB": lorentz_power_MW,
            "joule_integral_A_etaJ2": joule_heating_MW,
            "lorentz_plus_joule": lorentz_power_MW + joule_heating_MW,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Freidberg slide-38/39 conservation balances for a stored MHD profile.")
    parser.add_argument("profile", type=Path)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--B", type=float, default=None)
    parser.add_argument("--area-scale", type=float, default=None)
    parser.add_argument("--heavy-particle-mass-kg", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = audit_profile(
        profile_path=args.profile.resolve(),
        summary_path=None if args.summary is None else args.summary.resolve(),
        B_T=args.B,
        area_scale_m2=args.area_scale,
        heavy_particle_mass_kg=args.heavy_particle_mass_kg,
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
