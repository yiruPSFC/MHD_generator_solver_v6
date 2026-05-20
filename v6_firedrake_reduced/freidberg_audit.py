from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .design import CaseConfig, DesignBounds, DesignVector
from .legacy_physics import closure_state, inlet_design_generic, ops_for_numeric
from .transport import working_fluid_for_config


K_B = 1.380649e-23


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(np.asarray(y, dtype=float), np.asarray(x, dtype=float)))


def _case_config_from_summary(payload: dict[str, Any]) -> CaseConfig:
    case = dict(payload["case_config"])
    bounds_payload = case["bounds"]
    lower = DesignVector.from_dict({name: values["min"] for name, values in bounds_payload.items()})
    upper = DesignVector.from_dict({name: values["max"] for name, values in bounds_payload.items()})
    return CaseConfig(
        case=str(case["case"]),
        objective_profile=str(case["objective_profile"]),
        length_m=float(case["length_m"]),
        area_scale_m2=float(case["area_scale_m2"]),
        B_T=float(case["B_T"]),
        working_fluid_profile=str(case["working_fluid_profile"]),
        n_intervals=int(case["n_intervals"]),
        design=DesignVector.from_dict(case["design"]),
        bounds=DesignBounds(lower=lower, upper=upper),
        metadata=dict(case.get("metadata", {}) or {}),
    )


def _closure_arrays(*, profile: dict[str, np.ndarray], design: DesignVector, config: CaseConfig) -> dict[str, np.ndarray]:
    ops = ops_for_numeric()
    fluid = working_fluid_for_config(config)
    inlet = inlet_design_generic(
        ops=ops,
        n_p_in=design.n_p_in,
        T_e_in=design.T_e_in,
        Z_in=design.Z_in,
        I_0=design.I_0,
        seed_fraction=design.seed_fraction,
        B=float(config.B_T),
        inlet_A=float(config.area_scale_m2),
        working_fluid=fluid,
    )
    fields: dict[str, list[float]] = {
        "T_p": [],
        "v_p": [],
        "eta": [],
        "J_x": [],
        "J_y": [],
        "E_x": [],
        "n_e": [],
        "mach": [],
        "G": [],
    }
    for n_p, T_e, A, sigma in zip(
        profile["n_p"],
        profile["T_e"],
        profile["A"],
        profile["sigma_logA"],
        strict=True,
    ):
        closure = closure_state(
            ops=ops,
            n_p=float(n_p),
            T_e=float(T_e),
            A=float(A),
            dot_N=float(inlet["dot_N"]),
            I_0=float(design.I_0),
            seed_fraction=design.seed_fraction,
            B=float(config.B_T),
            working_fluid=fluid,
        )
        for name in fields:
            fields[name].append(float(closure[name]))
    return {name: np.asarray(values, dtype=float) for name, values in fields.items()}


def audit_run_directory(run_dir: str | Path, *, write_arrays: bool = True) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    summary_path = run_path / "run_summary.json"
    profile_path = run_path / "profile.npz"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    config = _case_config_from_summary(payload)
    design = config.design
    fluid = working_fluid_for_config(config)
    mp = float(fluid.heavy_particle_mass_kg)

    with np.load(profile_path) as data:
        profile = {name: np.asarray(data[name], dtype=float) for name in data.files}

    closures = _closure_arrays(profile=profile, design=design, config=config)
    x = np.asarray(profile["x"], dtype=float)
    A = np.asarray(profile["A"], dtype=float)
    n_p = np.asarray(profile["n_p"], dtype=float)
    T_e = np.asarray(profile["T_e"], dtype=float)
    T_p = closures["T_p"]
    v_p = closures["v_p"]
    eta = closures["eta"]
    J_x = closures["J_x"]
    J_y = closures["J_y"]
    E_x = closures["E_x"]
    n_e = closures["n_e"]

    A0 = float(config.area_scale_m2)
    J2 = J_x * J_x + J_y * J_y
    M2 = 3.0 * v_p * v_p / np.maximum(5.0 * K_B * T_p / mp, 1e-300)
    M = np.sqrt(np.maximum(M2, 0.0))
    p_p = n_p * K_B * T_p

    H_p = (A * n_p * v_p / A0) * (2.5 * K_B * T_p + 0.5 * mp * v_p * v_p)
    L_p = M / np.maximum((M2 + 3.0) ** 2, 1e-300) * (A / A0)
    rhs_H = (A / A0) * (v_p * J_y * float(config.B_T) + eta * J2)
    rhs_L = (
        -(12.0 / 5.0)
        * L_p
        / np.maximum((M2 + 3.0) * p_p * v_p, 1e-300)
        * (v_p * J_y * float(config.B_T) - ((5.0 * M2 + 3.0) / 12.0) * eta * J2)
    )

    dx = np.diff(x)
    interval_H_delta_MW = np.diff(H_p) * A0 / 1e6
    interval_H_rhs_MW = 0.5 * dx * (rhs_H[:-1] + rhs_H[1:]) * A0 / 1e6
    interval_H_residual_MW = interval_H_delta_MW - interval_H_rhs_MW
    interval_L_delta = np.diff(L_p)
    interval_L_rhs = 0.5 * dx * (rhs_L[:-1] + rhs_L[1:])
    interval_L_residual = interval_L_delta - interval_L_rhs

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
    lorentz_power_MW = float(_trapz(A * v_p * J_y * float(config.B_T), x) / 1e6)
    H_delta_MW = float((H_p[-1] - H_p[0]) * A0 / 1e6)
    H_rhs_integral_MW = float(_trapz(rhs_H, x) * A0 / 1e6)
    L_delta = float(L_p[-1] - L_p[0])
    L_rhs_integral = _trapz(rhs_L, x)

    denom_primary = max(abs(inlet_primary_enthalpy_MW), 1e-300)
    denom_l = max(float(np.nanmax(np.abs(L_p))), 1e-300)
    report = {
        "run_dir": str(run_path),
        "profile_path": str(profile_path),
        "n_points": int(x.size),
        "B_T": float(config.B_T),
        "area_scale_m2": A0,
        "heavy_particle_mass_kg": mp,
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
            "residual_fraction_of_inlet_primary": float((H_delta_MW - H_rhs_integral_MW) / denom_primary),
            "max_interval_residual_MW": float(np.nanmax(np.abs(interval_H_residual_MW))),
            "max_interval_residual_fraction_of_inlet_primary": float(
                np.nanmax(np.abs(interval_H_residual_MW)) / denom_primary
            ),
        },
        "freidberg_L": {
            "delta": L_delta,
            "rhs_integral": L_rhs_integral,
            "residual": L_delta - L_rhs_integral,
            "residual_fraction_of_max_abs_L": float((L_delta - L_rhs_integral) / denom_l),
            "max_interval_residual": float(np.nanmax(np.abs(interval_L_residual))),
            "max_interval_residual_fraction_of_max_abs_L": float(
                np.nanmax(np.abs(interval_L_residual)) / denom_l
            ),
        },
        "power_terms_MW": {
            "lorentz_integral_A_vJyB": lorentz_power_MW,
            "joule_integral_A_etaJ2": joule_heating_MW,
            "lorentz_plus_joule": lorentz_power_MW + joule_heating_MW,
        },
    }
    if write_arrays:
        np.savez(
            run_path / "freidberg_audit_profile.npz",
            x=x,
            H_p=H_p,
            L_p=L_p,
            rhs_H=rhs_H,
            rhs_L=rhs_L,
            interval_H_residual_MW=interval_H_residual_MW,
            interval_L_residual=interval_L_residual,
            mach=M,
            T_p=T_p,
            v_p=v_p,
            eta=eta,
            J_x=J_x,
            J_y=J_y,
            E_x=E_x,
            n_e=n_e,
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Freidberg H/L balances for a Firedrake reduced run directory.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-arrays", action="store_true")
    args = parser.parse_args(argv)

    report = audit_run_directory(args.run_dir, write_arrays=not args.no_arrays)
    text = json.dumps(report, indent=2, sort_keys=True)
    out_path = args.out or (Path(args.run_dir) / "freidberg_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
