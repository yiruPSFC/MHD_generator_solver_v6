from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Mapping

import matplotlib

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from v6_global_marginal.global_postprocess_v6 import (
    compute_global_metrics,
    cumulative_mhd_output_power_MWe,
)
from v6_core.local_algebraic_closure import K_B


ATM = 101325.0


def _to_1d(name: str, values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return arr


def _match_B_profile(B, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(B, dtype=float)
    if arr.ndim == 0:
        return np.full_like(x, float(arr))
    if arr.ndim != 1 or arr.shape != x.shape:
        raise ValueError("B must be scalar or same-shape 1D array")
    return arr


def _margin_linthresh(*arrays: np.ndarray) -> float:
    finite_parts = []
    for arr in arrays:
        flat = np.asarray(arr, dtype=float).reshape(-1)
        mask = np.isfinite(flat) & (flat != 0.0)
        if np.any(mask):
            finite_parts.append(np.abs(flat[mask]))
    if not finite_parts:
        return 1e-8
    values = np.concatenate(finite_parts)
    return float(max(1e-8, np.nanpercentile(values, 5.0)))


def _plot_margin_symlog(ax, x_values: np.ndarray, margin: np.ndarray, *, x_label: str) -> None:
    linthresh = _margin_linthresh(margin)
    ax.plot(x_values, np.asarray(margin, dtype=float))
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.35)
    ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0)
    ax.set_title("Velikhov margin G (symlog)", fontsize=10)
    ax.set_xlabel(x_label)
    ax.grid(True, alpha=0.3, which="both")


def plot_global_results_v6(
    profile: Mapping[str, np.ndarray],
    save_path: str | os.PathLike[str],
    *,
    B,
    seed_fraction: float,
    title: str = "",
    furnace_power_MW: float | None = None,
    steam_cycle_efficiency: float | None = None,
) -> dict[str, float]:
    x = _to_1d("x", profile["x"])
    n_p = _to_1d("n_p", profile["n_p"])
    T_e = _to_1d("T_e", profile["T_e"])
    T_p = _to_1d("T_p", profile["T_p"])
    A = _to_1d("A", profile["A"])
    v_p = _to_1d("v_p", profile["v_p"])
    n_e = _to_1d("n_e", profile["n_e"])
    beta = _to_1d("beta", profile["beta"])
    eta = _to_1d("eta", profile["eta"])
    Z = _to_1d("Z", profile["Z"])
    J_x = _to_1d("J_x", profile["J_x"])
    J_y = _to_1d("J_y", profile["J_y"])
    E_x = _to_1d("E_x", profile["E_x"])
    mach = _to_1d("mach", profile["mach"])
    velikhov = _to_1d("velikhov_margin", profile["velikhov_margin"])
    B_profile = _match_B_profile(B, x)

    x_mm = x * 1e3
    pressure_atm = n_p * K_B * T_p / ATM
    extracted_power = -J_x * E_x / 1e6
    ohmic_power = eta * (J_x * J_x + J_y * J_y) / 1e6
    power_ratio = np.divide(
        ohmic_power,
        extracted_power,
        out=np.full_like(ohmic_power, np.nan),
        where=np.abs(extracted_power) > 0.0,
    )
    conductivity = np.divide(1.0, eta, out=np.full_like(eta, np.nan), where=eta > 0.0)
    n_s = float(seed_fraction) * n_p
    f_I = np.divide(n_e, n_s, out=np.full_like(n_e, np.nan), where=n_s > 0.0)
    wall_loading = eta * (J_x * J_x + J_y * J_y) * A / np.maximum(4.0 * np.sqrt(A), 1e-30) / 1e6

    mass_flux = n_p * v_p * A
    current_flux = J_x * A
    mass_flux_res = np.divide(
        mass_flux,
        mass_flux[0],
        out=np.full_like(mass_flux, np.nan),
        where=np.abs(mass_flux[0]) > 0.0,
    ) - 1.0
    current_flux_res = np.divide(
        current_flux,
        current_flux[0],
        out=np.full_like(current_flux, np.nan),
        where=np.abs(current_flux[0]) > 0.0,
    ) - 1.0
    cumulative_mhd = cumulative_mhd_output_power_MWe(x=x, A=A, J_x=J_x, E_x=E_x)

    metrics = compute_global_metrics(
        x=x,
        A=A,
        v_p=v_p,
        eta=eta,
        J_x=J_x,
        J_y=J_y,
        E_x=E_x,
        B=B_profile,
        velikhov_margin=velikhov,
        furnace_power_MW=furnace_power_MW,
        steam_cycle_efficiency=steam_cycle_efficiency,
    )

    series = [
        ("pressure (atm)", pressure_atm, None),
        ("T_p (K)", T_p, None),
        ("T_e (K)", T_e, None),
        ("n_p (m^-3)", n_p, "log"),
        ("n_e (m^-3)", n_e, "log"),
        ("v_p (m/s)", v_p, None),
        ("A (m^2)", A, None),
        ("Mach number", mach, None),
        ("load power density (MW/m^3)", extracted_power, None),
        ("ohmic power density (MW/m^3)", ohmic_power, None),
        ("S_ohm / S_load", power_ratio, None),
        ("conductivity (S/m)", conductivity, "log"),
        ("beta / Z / beta^2", None, None),
        ("f_I", f_I, None),
        ("Velikhov margin G", velikhov, "symlog"),
        ("cum. MHD output (MWe)", cumulative_mhd, None),
        ("n v A / const - 1", mass_flux_res, None),
        ("Jx A / const - 1", current_flux_res, None),
        ("E_x (V/m)", E_x, None),
        ("wall loading (MW/m^2)", wall_loading, None),
    ]

    fig, axes = plt.subplots(5, 4, figsize=(18, 13), constrained_layout=True)
    axes_flat = axes.ravel()

    for idx, (label, data, yscale) in enumerate(series):
        ax = axes_flat[idx]
        if label == "beta / Z / beta^2":
            ax.semilogy(x_mm, np.where(beta > 0.0, beta, np.nan), label="beta")
            ax.semilogy(x_mm, np.where(Z > 0.0, Z, np.nan), label="Z")
            ax.semilogy(x_mm, np.where(beta * beta > 0.0, beta * beta, np.nan), label="beta^2")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, which="both")
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("x (mm)")
            continue

        arr = np.asarray(data, dtype=float)
        if yscale == "log":
            arr = np.where(arr > 0.0, arr, np.nan)
            ax.semilogy(x_mm, arr)
        elif yscale == "symlog":
            _plot_margin_symlog(ax, x_mm, arr, x_label="x (mm)")
            continue
        else:
            ax.plot(x_mm, arr)

        if label == "A (m^2)":
            note = (
                f"A_in={metrics.inlet_area_m2:.3f} m^2, A_out={metrics.outlet_area_m2:.3f} m^2\n"
                f"h_in={metrics.inlet_height_m:.3f} m, h_out={metrics.outlet_height_m:.3f} m\n"
                f"seed={seed_fraction:.3e}"
            )
            ax.text(0.02, 0.98, note, transform=ax.transAxes, va="top", fontsize=8)

        if label == "cum. MHD output (MWe)":
            note = (
                f"P_MHD={metrics.mhd_output_power_MWe:.3f} MWe\n"
                f"L={metrics.length_m:.3f} m, V={metrics.volume_m3:.3f} m^3\n"
                f"W_mag={metrics.magnetic_energy_MJ:.3f} MJ"
            )
            ax.text(0.02, 0.98, note, transform=ax.transAxes, va="top", fontsize=8)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("x (mm)")
        ax.grid(True, alpha=0.3, which="both")

    for idx in range(len(series), len(axes_flat)):
        axes_flat[idx].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return {
        "mass_flux_res_max": float(np.nanmax(np.abs(mass_flux_res))),
        "current_flux_res_max": float(np.nanmax(np.abs(current_flux_res))),
        "velikhov_margin_max_abs": float(np.nanmax(np.abs(velikhov))),
        "wall_loading_max_MW_m2": float(np.nanmax(wall_loading)),
        "electric_field_min_V_m": float(np.nanmin(E_x)),
        "mhd_output_power_MWe": float(metrics.mhd_output_power_MWe),
        "total_plant_power_MWe": float(metrics.total_plant_power_MWe),
        "magnetic_energy_MJ": float(metrics.magnetic_energy_MJ),
    }


__all__ = ["plot_global_results_v6"]
