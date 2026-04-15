#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from v6_global_marginal.reference_recovery.global_plotting_v6 import plot_global_results_v6
from v6_global_marginal.global_postprocess_v6 import (
    compute_global_metrics,
    required_mhd_output_power,
    trim_profile_to_mhd_target,
)
from v6_core.local_algebraic_closure import SIGMA_EP
from v6_global_marginal.pde_solver_v6_batch_global import ForwardPDESolverV6BatchGlobal, event_name_from_code

_V5_GLOBAL = _ROOT.parent / "mhd_generator_solver_v5" / "global_reproduce"
if str(_V5_GLOBAL) not in sys.path:
    sys.path.insert(0, str(_V5_GLOBAL))

from jeffrey_params import load_jeffrey_params  # type: ignore  # noqa: E402


ATM = 101325.0
K_IONIZATION_ENERGY_EV = 4.3407


def _extract(text: str, pattern: str) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"missing field for pattern: {pattern}")
    return float(match.group(1).replace(",", ""))


def load_jeffrey_reference(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {
        "P_TOT": _extract(text, r"P_TOT.*?:\s*([0-9.+\-eE,]+)"),
        "T_in": _extract(text, r"T_in.*?:\s*([0-9.+\-eE,]+)"),
        "p_in": _extract(text, r"p_in.*?:\s*([0-9.+\-eE,]+)"),
        "M_0": _extract(text, r"Mach number.*?:\s*([0-9.+\-eE,]+)"),
        "eta_T": _extract(text, r"eta_T\)\s*:?\s*([0-9.+\-eE,]+)"),
        "f": _extract(text, r"MHD conversion efficiency.*?:\s*([0-9.+\-eE,]+)"),
        "P_w": _extract(text, r"P_w\)\s*:?\s*([0-9.+\-eE,]+)"),
        "E_max": _extract(text, r"E_max\).*?:\s*([0-9.+\-eE,\-]+)"),
        "P_Fur": _extract(text, r"P_Fur.*?:\s*([0-9.+\-eE,]+)"),
        "P_MHD": _extract(text, r"P_MHD.*?:\s*([0-9.+\-eE,]+)"),
        "P_Rin": _extract(text, r"P_Rin.*?:\s*([0-9.+\-eE,]+)"),
        "P_Rout": _extract(text, r"P_Rout.*?:\s*([0-9.+\-eE,]+)"),
        "P_Loss": _extract(text, r"P_Loss.*?:\s*([0-9.+\-eE,]+)"),
        "eta_TOT": _extract(text, r"eta_TOT.*?:\s*([0-9.+\-eE,]+)"),
        "p_p0": _extract(text, r"Inlet Argon pressure.*?:\s*([0-9.+\-eE,]+)"),
        "T_p0": _extract(text, r"Inlet Argon temperature.*?:\s*([0-9.+\-eE,]+)"),
        "n_p0": _extract(text, r"Inlet Argon density.*?:\s*([0-9.+\-eE,]+)"),
        "v_p0": _extract(text, r"Inlet Argon velocity.*?:\s*([0-9.+\-eE,]+)"),
        "A0": _extract(text, r"Inlet cross section area.*?:\s*([0-9.+\-eE,]+)"),
        "S_MHD0": _extract(text, r"Inlet electric power density.*?:\s*([0-9.+\-eE,]+)"),
        "T_e0": _extract(text, r"Inlet electron temperature.*?:\s*([0-9.+\-eE,]+)"),
        "B_0": _extract(text, r"Applied magnetic field.*?:\s*([0-9.+\-eE,]+)"),
        "beta_0": _extract(text, r"Inlet Hall parameter.*?:\s*([0-9.+\-eE,]+)"),
        "n_e0": _extract(text, r"Inlet electron density.*?:\s*([0-9.+\-eE,]+)"),
        "n_s0": _extract(text, r"seed density.*?:\s*([0-9.+\-eE,]+)"),
        "J_x0": _extract(text, r"Inlet Hall current density.*?:\s*([0-9.+\-eE,]+)"),
        "J_y0": _extract(text, r"Inlet Faraday current density.*?:\s*([0-9.+\-eE,\-]+)"),
        "L": _extract(text, r"Generator length.*?:\s*([0-9.+\-eE,]+)"),
        "h0": _extract(text, r"Inlet generator height.*?:\s*([0-9.+\-eE,]+)"),
        "h1": _extract(text, r"Outlet generator height.*?:\s*([0-9.+\-eE,]+)"),
        "V": _extract(text, r"Generator volume.*?:\s*([0-9.+\-eE,]+)"),
        "W_mag": _extract(text, r"W_mag.*?:\s*([0-9.+\-eE,]+)"),
    }


def _residual(value: float, reference: float) -> float:
    if reference == 0.0 or (not np.isfinite(reference)):
        return float("nan")
    return float(value / reference - 1.0)


def _compare_map(values: dict[str, float], reference: dict[str, float]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, value in values.items():
        ref = float(reference.get(key, float("nan")))
        out[key] = {
            "value": float(value),
            "reference": ref,
            "residual": _residual(float(value), ref),
        }
    return out


def _fmt_compare_line(
    label: str,
    *,
    jeffrey: float,
    actual: float,
    unit: str = "",
) -> str:
    res = _residual(actual, jeffrey)
    unit_str = f" {unit}" if unit else ""
    left = f"Jeffrey={jeffrey:.6g}{unit_str}"
    right = f"V6={actual:.6g}{unit_str}"
    res_str = f"res={res:+.3e}" if np.isfinite(res) else "res=n/a"
    return f"{label:<48}: {left:<28} | {right:<28} | {res_str}"


def write_comparison_report(
    path: Path,
    *,
    reference_path: Path,
    reference: dict[str, float],
    compare_values: dict[str, float],
    projected_seed_fraction: float,
    reference_seed_fraction: float,
    Z_in_reference: float,
    full_metrics,
    trimmed_metrics,
    plot_stats: dict[str, float],
) -> None:
    lines: list[str] = []
    lines.append("=" * 88)
    lines.append(f"FILE: {path.name}")
    lines.append("DESCRIPTION: Jeffrey reference recovery report (V6 global solver)")
    lines.append(f"SOURCE: {reference_path}")
    lines.append(f"Seed ionization energy E_I = {K_IONIZATION_ENERGY_EV:.4f} eV (potassium branch)")
    lines.append(f"Electron-argon sigma_ep = {SIGMA_EP:.9e} m^2")
    lines.append("=" * 88)
    lines.append("")
    lines.append("NOTE: Residual is (V6 / Jeffrey - 1).")
    lines.append("NOTE: V6 values below are taken from the profile trimmed to Jeffrey's P_TOT target.")
    lines.append("")

    lines.append("-" * 88)
    lines.append("1. BASIC INPUTS (基本输入参数)")
    lines.append("-" * 88)
    lines.append(_fmt_compare_line("Total electric power out (P_TOT)", jeffrey=reference["P_TOT"], actual=compare_values["P_TOT"], unit="MWe"))
    lines.append(_fmt_compare_line("HTGR Argon furnace input Temp (T_in)", jeffrey=reference["T_in"], actual=compare_values["T_in"], unit="K"))
    lines.append(_fmt_compare_line("HTGR Argon furnace input Press (p_in)", jeffrey=reference["p_in"], actual=compare_values["p_in"], unit="atm"))
    lines.append(_fmt_compare_line("MHD generator input Mach number (M_0)", jeffrey=reference["M_0"], actual=compare_values["M_0"]))
    lines.append(_fmt_compare_line("Efficiency of steam cycle (eta_T)", jeffrey=reference["eta_T"], actual=compare_values["eta_T"]))
    lines.append(_fmt_compare_line("MHD conversion efficiency (f)", jeffrey=reference["f"], actual=compare_values["f"]))
    lines.append("")

    lines.append("-" * 88)
    lines.append("2. ENGINEERING CONSTRAINT INPUTS (工程约束输入)")
    lines.append("-" * 88)
    lines.append(_fmt_compare_line("Electrode wall loading limit (P_w)", jeffrey=reference["P_w"], actual=compare_values["P_w"], unit="MW/m^2"))
    lines.append(_fmt_compare_line("Electric field breakdown limit (E_max)", jeffrey=reference["E_max"], actual=compare_values["E_max"], unit="V/m"))
    lines.append("")

    lines.append("-" * 88)
    lines.append("3. OVERALL POWER BALANCE (整体功率平衡)")
    lines.append("-" * 88)
    lines.append(_fmt_compare_line("Furnace thermal power (P_Fur)", jeffrey=reference["P_Fur"], actual=compare_values["P_Fur"], unit="MW"))
    lines.append(_fmt_compare_line("MHD electric output power (P_MHD)", jeffrey=reference["P_MHD"], actual=compare_values["P_MHD"], unit="MWe"))
    lines.append(_fmt_compare_line("Input thermal power to steam generator (P_Rin)", jeffrey=reference["P_Rin"], actual=compare_values["P_Rin"], unit="MW"))
    lines.append(_fmt_compare_line("Output electric power from steam gen (P_Rout)", jeffrey=reference["P_Rout"], actual=compare_values["P_Rout"], unit="MWe"))
    lines.append(_fmt_compare_line("Thermal power lost up the chimney (P_Loss)", jeffrey=reference["P_Loss"], actual=compare_values["P_Loss"], unit="MW"))
    lines.append(_fmt_compare_line("Overall plant efficiency (eta_TOT)", jeffrey=reference["eta_TOT"], actual=compare_values["eta_TOT"]))
    lines.append("")

    lines.append("-" * 88)
    lines.append("4. INLET QUANTITIES TO MHD GENERATOR (入口参数)")
    lines.append("-" * 88)
    lines.append(_fmt_compare_line("Inlet Argon pressure (p_p0)", jeffrey=reference["p_p0"], actual=compare_values["p_p0"], unit="atm"))
    lines.append(_fmt_compare_line("Inlet Argon temperature (T_p0)", jeffrey=reference["T_p0"], actual=compare_values["T_p0"], unit="K"))
    lines.append(_fmt_compare_line("Inlet Argon density (n_p0)", jeffrey=reference["n_p0"], actual=compare_values["n_p0"], unit="m^-3"))
    lines.append(_fmt_compare_line("Inlet Argon velocity (v_p0)", jeffrey=reference["v_p0"], actual=compare_values["v_p0"], unit="m/s"))
    lines.append(_fmt_compare_line("Inlet cross section area (s_0)", jeffrey=reference["A0"], actual=compare_values["A0"], unit="m^2"))
    lines.append(_fmt_compare_line("Inlet electric power density (S_MHD0)", jeffrey=reference["S_MHD0"], actual=compare_values["S_MHD0"], unit="MW/m^3"))
    lines.append(_fmt_compare_line("Inlet electron temperature (T_e0)", jeffrey=reference["T_e0"], actual=compare_values["T_e0"], unit="K"))
    lines.append(_fmt_compare_line("Applied magnetic field (B_0)", jeffrey=reference["B_0"], actual=compare_values["B_0"], unit="T"))
    lines.append(_fmt_compare_line("Inlet Hall parameter (beta_0)", jeffrey=reference["beta_0"], actual=compare_values["beta_0"]))
    lines.append(_fmt_compare_line("Inlet electron density (n_e0)", jeffrey=reference["n_e0"], actual=compare_values["n_e0"], unit="m^-3"))
    lines.append(_fmt_compare_line("Initial pre-ionization seed density (n_s0)", jeffrey=reference["n_s0"], actual=compare_values["n_s0"], unit="m^-3"))
    lines.append(_fmt_compare_line("Inlet Hall current density (J_x0)", jeffrey=reference["J_x0"], actual=compare_values["J_x0"], unit="A/m^2"))
    lines.append(_fmt_compare_line("Inlet Faraday current density (J_y0)", jeffrey=reference["J_y0"], actual=compare_values["J_y0"], unit="A/m^2"))
    lines.append("")

    lines.append("-" * 88)
    lines.append("5. OUTLET/GEOMETRY QUANTITIES (出口/几何)")
    lines.append("-" * 88)
    lines.append(_fmt_compare_line("Generator length (l)", jeffrey=reference["L"], actual=compare_values["L"], unit="m"))
    lines.append(_fmt_compare_line("Inlet generator height (h_0)", jeffrey=reference["h0"], actual=compare_values["h0"], unit="m"))
    lines.append(_fmt_compare_line("Outlet generator height (h_1)", jeffrey=reference["h1"], actual=compare_values["h1"], unit="m"))
    lines.append(_fmt_compare_line("Generator volume (approx.) (V)", jeffrey=reference["V"], actual=compare_values["V"], unit="m^3"))
    lines.append(_fmt_compare_line("Magnetic energy (approx.) (W_mag)", jeffrey=reference["W_mag"], actual=compare_values["W_mag"], unit="MJ"))
    lines.append("")

    lines.append("-" * 88)
    lines.append("6. V6-SPECIFIC NOTES (V6 特有信息)")
    lines.append("-" * 88)
    lines.append(f"Reference seed fraction inferred from Jeffrey numbers      : {reference_seed_fraction:.6e}")
    lines.append(f"Projected marginal seed fraction used by V6 solver        : {projected_seed_fraction:.6e}")
    lines.append(f"Seed projection residual (V6 / Jeffrey - 1)               : {_residual(projected_seed_fraction, reference_seed_fraction):+.3e}")
    lines.append(f"Reference inlet Z inferred from Jeffrey beta/Jx/Jy        : {Z_in_reference:.6f}")
    lines.append(f"Trimmed profile inlet Velikhov margin                     : {trimmed_metrics.inlet_velikhov_margin:+.3e}")
    lines.append(f"Trimmed profile outlet Velikhov margin                    : {trimmed_metrics.outlet_velikhov_margin:+.3e}")
    lines.append(f"Trimmed profile max |Velikhov drift|                      : {trimmed_metrics.max_abs_velikhov_drift:.3e}")
    lines.append(f"Mass-flux conservation max residual                       : {plot_stats['mass_flux_res_max']:.3e}")
    lines.append(f"Current-flux conservation max residual                    : {plot_stats['current_flux_res_max']:.3e}")
    lines.append("")

    lines.append("-" * 88)
    lines.append("7. FULL-LENGTH (UNTRIMMED) V6 RESULT (未截断到 200 MWe 的结果)")
    lines.append("-" * 88)
    lines.append(f"Full-length channel length                                : {full_metrics.length_m:.6g} m")
    lines.append(f"Full-length MHD output power                              : {full_metrics.mhd_output_power_MWe:.6g} MWe")
    lines.append(f"Full-length total plant power                             : {full_metrics.total_plant_power_MWe:.6g} MWe")
    lines.append(f"Full-length outlet height                                 : {full_metrics.outlet_height_m:.6g} m")
    lines.append(f"Full-length magnetic energy                               : {full_metrics.magnetic_energy_MJ:.6g} MJ")
    lines.append("")

    lines.append("-" * 88)
    lines.append("8. FORMULAS (计算说明)")
    lines.append("-" * 88)
    lines.append("P_MHD  = -∫ A(x) * Jx(x) * Ex(x) dx")
    lines.append("P_Rin  = P_Fur - P_MHD")
    lines.append("P_Rout = eta_T * P_Rin")
    lines.append("P_TOT  = P_MHD + P_Rout")
    lines.append("P_Loss = P_Fur - P_MHD - P_Rout")
    lines.append("E_max  = min_x Ex(x)")
    lines.append("P_w    = max_x [eta * (Jx^2 + Jy^2) * A / (4*sqrt(A))] for A = h^2")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _derive_reference_inputs(reference: dict[str, float]) -> dict[str, float]:
    Z_in = -reference["beta_0"] * reference["J_y0"] / reference["J_x0"] - 1.0
    seed_fraction = reference["n_s0"] / reference["n_p0"]
    return {
        "Z_in": Z_in,
        "seed_fraction": seed_fraction,
    }


def _profile_from_result(out, index: int) -> dict[str, np.ndarray]:
    idx_last = int(out.valid_points[index]) - 1
    return {
        "x": out.x[: idx_last + 1],
        "n_p": out.n_p[index, : idx_last + 1],
        "T_e": out.T_e[index, : idx_last + 1],
        "T_p": out.T_p[index, : idx_last + 1],
        "A": out.A[index, : idx_last + 1],
        "v_p": out.v_p[index, : idx_last + 1],
        "n_e": out.n_e[index, : idx_last + 1],
        "beta": out.beta[index, : idx_last + 1],
        "eta": out.eta[index, : idx_last + 1],
        "Z": out.Z[index, : idx_last + 1],
        "J_x": out.J_x[index, : idx_last + 1],
        "J_y": out.J_y[index, : idx_last + 1],
        "E_x": out.E_x[index, : idx_last + 1],
        "mach": out.mach[index, : idx_last + 1],
        "velikhov_margin": out.velikhov_margin[index, : idx_last + 1],
    }


def _trim_profile_to_reference_target(
    profile: dict[str, np.ndarray],
    *,
    reference: dict[str, float],
) -> dict[str, np.ndarray] | None:
    target_mhd = required_mhd_output_power(
        total_plant_power_MWe=reference["P_TOT"],
        furnace_power_MW=reference["P_Fur"],
        steam_cycle_efficiency=reference["eta_T"],
    )
    return trim_profile_to_mhd_target(
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


def _metrics_to_reference_space(
    profile: dict[str, np.ndarray],
    metrics,
    *,
    B: float,
    seed_fraction: float,
) -> dict[str, float]:
    n_p0 = float(profile["n_p"][0])
    T_p0 = float(profile["T_p"][0])
    T_e0 = float(profile["T_e"][0])
    v_p0 = float(profile["v_p"][0])
    n_e0 = float(profile["n_e"][0])
    beta0 = float(profile["beta"][0])
    J_x0 = float(profile["J_x"][0])
    J_y0 = float(profile["J_y"][0])
    return {
        "P_TOT": float(metrics.total_plant_power_MWe),
        "T_in": float("nan"),
        "p_in": float("nan"),
        "M_0": float("nan"),
        "eta_T": float(metrics.steam_cycle_efficiency),
        "f": float("nan"),
        "P_w": float(metrics.wall_loading_MW_m2),
        "E_max": float(metrics.electric_field_min_V_m),
        "P_Fur": float(metrics.furnace_power_MW),
        "P_MHD": float(metrics.mhd_output_power_MWe),
        "P_Rin": float(metrics.steam_input_power_MW),
        "P_Rout": float(metrics.steam_output_power_MWe),
        "P_Loss": float(metrics.chimney_loss_MW),
        "eta_TOT": float(metrics.total_plant_efficiency),
        "p_p0": float(n_p0 * T_p0 * 1.380649e-23 / ATM),
        "T_p0": T_p0,
        "n_p0": n_p0,
        "v_p0": v_p0,
        "A0": float(profile["A"][0]),
        "S_MHD0": float(-v_p0 * J_y0 * B / 1e6),
        "T_e0": T_e0,
        "B_0": float(B),
        "beta_0": beta0,
        "n_e0": n_e0,
        "n_s0": float(seed_fraction * n_p0),
        "J_x0": J_x0,
        "J_y0": J_y0,
        "L": float(metrics.length_m),
        "h0": float(metrics.inlet_height_m),
        "h1": float(metrics.outlet_height_m),
        "V": float(metrics.volume_m3),
        "W_mag": float(metrics.magnetic_energy_MJ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recover Jeffrey reference case with the V6 global solver")
    parser.add_argument(
        "--reference-file",
        type=str,
        default=str(_V5_GLOBAL / "JeffreyParameter.txt"),
        help="reference txt extracted from Jeffrey slides",
    )
    parser.add_argument("--dx", type=float, default=0.01, help="integration step size [m]")
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(_THIS_DIR / "summary.json"),
        help="summary json output path",
    )
    parser.add_argument(
        "--out-npz",
        type=str,
        default=str(_THIS_DIR / "profile_trimmed.npz"),
        help="trimmed profile npz output path",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default=str(_THIS_DIR / "profile_trimmed.png"),
        help="trimmed profile plot path",
    )
    parser.add_argument(
        "--out-txt",
        type=str,
        default=str(_THIS_DIR / "summary.txt"),
        help="human-readable comparison report path",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    reference_path = Path(args.reference_file)
    reference = load_jeffrey_reference(reference_path)
    params = load_jeffrey_params(reference_path)
    derived = _derive_reference_inputs(reference)

    solver = ForwardPDESolverV6BatchGlobal(B=float(reference["B_0"]), length=float(reference["L"]))
    out = solver.solve_batch(
        n_p_in=np.array([params.np0], dtype=float),
        Z_in=np.array([derived["Z_in"]], dtype=float),
        T_p_in=np.array([params.Tp0], dtype=float),
        T_e_in=np.array([reference["T_e0"]], dtype=float),
        A_in=np.array([params.A0], dtype=float),
        dx=float(args.dx),
        store_profiles=True,
    )

    if (not bool(out.success[0])) or int(out.valid_points[0]) <= 1:
        payload = {
            "ok": False,
            "event": event_name_from_code(int(out.event_code[0])),
            "valid_points": int(out.valid_points[0]),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    full_profile = _profile_from_result(out, 0)
    trimmed_profile = _trim_profile_to_reference_target(full_profile, reference=reference)
    if trimmed_profile is None:
        payload = {
            "ok": False,
            "event": event_name_from_code(int(out.event_code[0])),
            "reason": "insufficient_mhd_power_before_reference_length",
            "valid_points": int(out.valid_points[0]),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    full_metrics = compute_global_metrics(
        x=full_profile["x"],
        A=full_profile["A"],
        v_p=full_profile["v_p"],
        eta=full_profile["eta"],
        J_x=full_profile["J_x"],
        J_y=full_profile["J_y"],
        E_x=full_profile["E_x"],
        B=float(reference["B_0"]),
        velikhov_margin=full_profile["velikhov_margin"],
        furnace_power_MW=float(reference["P_Fur"]),
        steam_cycle_efficiency=float(reference["eta_T"]),
    )
    trimmed_metrics = compute_global_metrics(
        x=trimmed_profile["x"],
        A=trimmed_profile["A"],
        v_p=trimmed_profile["v_p"],
        eta=trimmed_profile["eta"],
        J_x=trimmed_profile["J_x"],
        J_y=trimmed_profile["J_y"],
        E_x=trimmed_profile["E_x"],
        B=float(reference["B_0"]),
        velikhov_margin=trimmed_profile["velikhov_margin"],
        furnace_power_MW=float(reference["P_Fur"]),
        steam_cycle_efficiency=float(reference["eta_T"]),
    )

    compare_values = _metrics_to_reference_space(
        trimmed_profile,
        trimmed_metrics,
        B=float(reference["B_0"]),
        seed_fraction=float(out.seed_fraction[0]),
    )
    compare_values["T_in"] = float(reference["T_in"])
    compare_values["p_in"] = float(reference["p_in"])
    compare_values["M_0"] = float(reference["M_0"])
    compare_values["f"] = float(reference["f"])

    plot_stats = plot_global_results_v6(
        trimmed_profile,
        args.out_plot,
        B=float(reference["B_0"]),
        seed_fraction=float(out.seed_fraction[0]),
        title="Jeffrey Reference Case Recovered by V6 Global Solver",
        furnace_power_MW=float(reference["P_Fur"]),
        steam_cycle_efficiency=float(reference["eta_T"]),
    )

    payload: dict[str, Any] = {
        "ok": True,
        "event": event_name_from_code(int(out.event_code[0])),
        "step_size_m": float(args.dx),
        "assumed_seed_species": "K",
        "seed_species_note": (
            "v5/global_reproduce aligns Jeffrey with E_I=4.3407 eV; "
            "the old JeffreyParameter_Cs.txt / JeffreyParameter_K.txt labels are inverted."
        ),
        "ionization_energy_eV": float(K_IONIZATION_ENERGY_EV),
        "sigma_ep_m2": float(SIGMA_EP),
        "reference_inputs": {
            "n_p0": float(params.np0),
            "T_p0": float(params.Tp0),
            "v_p0": float(params.vp0),
            "A0": float(params.A0),
            "B0": float(params.B0),
            "T_e0": float(reference["T_e0"]),
            "J_x0": float(reference["J_x0"]),
            "J_y0": float(reference["J_y0"]),
            "beta_0": float(reference["beta_0"]),
            "seed_fraction_reference": float(derived["seed_fraction"]),
            "Z_in_reference": float(derived["Z_in"]),
        },
        "solver_inlet_projection": {
            "seed_fraction_projected": float(out.seed_fraction[0]),
            "seed_fraction_projection_residual": float(out.seed_fraction[0] / derived["seed_fraction"] - 1.0),
        },
        "full_length_metrics": full_metrics.to_dict(),
        "trimmed_reference_target_metrics": trimmed_metrics.to_dict(),
        "reference_comparison": _compare_map(compare_values, reference),
        "plot_stats": plot_stats,
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_txt = Path(args.out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    write_comparison_report(
        out_txt,
        reference_path=reference_path,
        reference=reference,
        compare_values=compare_values,
        projected_seed_fraction=float(out.seed_fraction[0]),
        reference_seed_fraction=float(derived["seed_fraction"]),
        Z_in_reference=float(derived["Z_in"]),
        full_metrics=full_metrics,
        trimmed_metrics=trimmed_metrics,
        plot_stats=plot_stats,
    )

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        x=trimmed_profile["x"],
        n_p=trimmed_profile["n_p"],
        T_e=trimmed_profile["T_e"],
        T_p=trimmed_profile["T_p"],
        A=trimmed_profile["A"],
        v_p=trimmed_profile["v_p"],
        n_e=trimmed_profile["n_e"],
        beta=trimmed_profile["beta"],
        eta=trimmed_profile["eta"],
        Z=trimmed_profile["Z"],
        J_x=trimmed_profile["J_x"],
        J_y=trimmed_profile["J_y"],
        E_x=trimmed_profile["E_x"],
        mach=trimmed_profile["mach"],
        velikhov_margin=trimmed_profile["velikhov_margin"],
        B=np.array([reference["B_0"]], dtype=float),
        seed_fraction=np.array([out.seed_fraction[0]], dtype=float),
        Z_in_reference=np.array([derived["Z_in"]], dtype=float),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
