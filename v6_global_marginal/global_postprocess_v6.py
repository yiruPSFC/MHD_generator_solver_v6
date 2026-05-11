from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np

from v6_core.local_algebraic_closure import K_B, M_P


MU0 = 4.0e-7 * math.pi


@dataclass(frozen=True)
class GlobalEngineeringMetrics:
    length_m: float
    inlet_area_m2: float
    outlet_area_m2: float
    inlet_height_m: float
    outlet_height_m: float
    volume_m3: float
    magnetic_energy_MJ: float
    inlet_power_density_MW_m3: float
    mhd_output_power_MWe: float
    wall_loading_MW_m2: float
    electric_field_min_V_m: float
    electric_field_max_abs_V_m: float
    inlet_velikhov_margin: float
    outlet_velikhov_margin: float
    max_abs_velikhov_drift: float
    furnace_power_MW: float
    steam_input_power_MW: float
    steam_output_power_MWe: float
    total_plant_power_MWe: float
    chimney_loss_MW: float
    steam_cycle_efficiency: float
    total_plant_efficiency: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DesignValueTerms:
    outlet_delta_te_K: float
    outlet_delta_te_per_kK: float
    outlet_delta_ratio: float
    outlet_f_ion: float
    outlet_mhd_output_per_100MWe: float
    mhd_output_power_MWe: float
    inlet_enthalpy_flux_MW: float
    outlet_enthalpy_extraction_ratio: float
    outlet_enthalpy_extraction_percent: float
    inlet_delta_ratio: float
    inlet_mach: float
    magnetic_field_T: float
    device_length_m: float
    device_length_per_5m: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DesignValueWeights:
    outlet_delta_te_per_kK: float = 1.0
    outlet_delta_ratio: float = 0.0
    outlet_f_ion: float = 0.0
    outlet_mhd_output_per_100MWe: float = 0.0
    inlet_delta_ratio_penalty: float = 0.0
    inlet_mach_penalty: float = 0.0
    magnetic_field_T_penalty: float = 0.0
    device_length_per_5m_penalty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DesignValueBreakdown:
    profile_name: str
    total_score: float
    contributions: Dict[str, float]
    terms: DesignValueTerms
    weights: DesignValueWeights

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "total_score": float(self.total_score),
            "contributions": {str(k): float(v) for k, v in self.contributions.items()},
            "terms": self.terms.to_dict(),
            "weights": self.weights.to_dict(),
        }


def _to_float_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("expected 1D array")
    return arr


def _safe_ratio(num: float, den: float, *, floor: float = 1e-30) -> float:
    if not np.isfinite(num) or not np.isfinite(den):
        return float("nan")
    den_safe = den if abs(den) >= floor else (floor if den >= 0.0 else -floor)
    return float(num / den_safe)


def compute_wall_loading_square(A: np.ndarray, eta: np.ndarray, J_x: np.ndarray, J_y: np.ndarray) -> float:
    A = _to_float_array(A)
    eta = _to_float_array(eta)
    J_x = _to_float_array(J_x)
    J_y = _to_float_array(J_y)
    h = np.sqrt(np.maximum(A, 0.0))
    J_sq = J_x * J_x + J_y * J_y
    q_wall = eta * J_sq * A / np.maximum(4.0 * h, 1e-30)
    return float(np.nanmax(q_wall))


def required_mhd_output_power(
    *,
    total_plant_power_MWe: float,
    furnace_power_MW: float,
    steam_cycle_efficiency: float,
) -> float:
    denom = 1.0 - float(steam_cycle_efficiency)
    if abs(denom) < 1e-12:
        raise ValueError("steam_cycle_efficiency must not be 1")
    return (
        float(total_plant_power_MWe) - float(steam_cycle_efficiency) * float(furnace_power_MW)
    ) / denom


def cumulative_mhd_output_power_MWe(*, x, A, J_x, E_x) -> np.ndarray:
    x = _to_float_array(x)
    A = _to_float_array(A)
    J_x = _to_float_array(J_x)
    E_x = _to_float_array(E_x)
    if len(x) == 0:
        return np.zeros(0, dtype=float)
    power_density = -A * J_x * E_x / 1e6
    out = np.zeros_like(x, dtype=float)
    for i in range(1, len(x)):
        dx = float(x[i] - x[i - 1])
        out[i] = out[i - 1] + 0.5 * dx * (power_density[i - 1] + power_density[i])
    return out


def inlet_stagnation_enthalpy_flux_W(
    *,
    n_p_in: float,
    T_p_in: float,
    T_e_in: float,
    n_e_in: float,
    v_p_in: float,
    A_in: float,
    heavy_particle_mass_kg: float = M_P,
) -> float:
    thermal_density = 2.5 * K_B * (
        float(n_p_in) * float(T_p_in) + float(n_e_in) * float(T_e_in)
    )
    kinetic_density = 0.5 * float(heavy_particle_mass_kg) * float(n_p_in) * float(v_p_in) * float(v_p_in)
    return float(A_in) * float(v_p_in) * (thermal_density + kinetic_density)


def trim_profile_to_mhd_target(
    *,
    target_mhd_output_MWe: float,
    x,
    A,
    J_x,
    E_x,
    **extra_arrays,
) -> dict[str, np.ndarray] | None:
    x = _to_float_array(x)
    A = _to_float_array(A)
    J_x = _to_float_array(J_x)
    E_x = _to_float_array(E_x)
    if len(x) == 0:
        return None

    cumulative = cumulative_mhd_output_power_MWe(x=x, A=A, J_x=J_x, E_x=E_x)
    target = float(target_mhd_output_MWe)
    if target <= 0.0:
        return None
    if cumulative[-1] < target:
        return None

    idx = int(np.searchsorted(cumulative, target, side="left"))
    if idx == 0:
        frac = 0.0
        x_star = float(x[0])
    else:
        p0 = float(cumulative[idx - 1])
        p1 = float(cumulative[idx])
        if abs(p1 - p0) < 1e-30:
            frac = 1.0
        else:
            frac = (target - p0) / (p1 - p0)
        frac = min(max(frac, 0.0), 1.0)
        x_star = float(x[idx - 1] + frac * (x[idx] - x[idx - 1]))

    trimmed: dict[str, np.ndarray] = {
        "x": np.concatenate([x[:idx], np.array([x_star], dtype=float)]),
        "cumulative_mhd_output_MWe": np.concatenate(
            [cumulative[:idx], np.array([target], dtype=float)]
        ),
    }

    base_arrays = {"A": A, "J_x": J_x, "E_x": E_x}
    base_arrays.update({name: _to_float_array(arr) for name, arr in extra_arrays.items()})

    for name, arr in base_arrays.items():
        if len(arr) != len(x):
            raise ValueError(f"{name} must have same length as x")
        if idx == 0:
            y_star = float(arr[0])
        else:
            y_star = float(arr[idx - 1] + frac * (arr[idx] - arr[idx - 1]))
        trimmed[name] = np.concatenate([arr[:idx], np.array([y_star], dtype=float)])

    return trimmed


def compute_global_metrics(
    *,
    x,
    A,
    v_p,
    eta,
    J_x,
    J_y,
    E_x,
    B,
    velikhov_margin=None,
    furnace_power_MW: float | None = None,
    steam_cycle_efficiency: float | None = None,
) -> GlobalEngineeringMetrics:
    x = _to_float_array(x)
    A = _to_float_array(A)
    v_p = _to_float_array(v_p)
    eta = _to_float_array(eta)
    J_x = _to_float_array(J_x)
    J_y = _to_float_array(J_y)
    E_x = _to_float_array(E_x)
    if len(x) == 0:
        raise ValueError("profile is empty")

    B_arr = np.asarray(B, dtype=float)
    if B_arr.ndim == 0:
        B_arr = np.full_like(x, float(B_arr))
    elif B_arr.ndim != 1 or B_arr.shape != x.shape:
        raise ValueError("B must be scalar or same-shape 1D array")

    length_m = float(x[-1] - x[0]) if len(x) >= 2 else 0.0
    inlet_area_m2 = float(A[0])
    outlet_area_m2 = float(A[-1])
    inlet_height_m = math.sqrt(max(inlet_area_m2, 0.0))
    outlet_height_m = math.sqrt(max(outlet_area_m2, 0.0))
    volume_m3 = float(np.trapezoid(A, x))
    magnetic_energy_MJ = float(np.trapezoid(0.5 * B_arr * B_arr * A / MU0, x)) / 1e6
    inlet_power_density_MW_m3 = float(-v_p[0] * J_y[0] * B_arr[0]) / 1e6
    mhd_output_power_MWe = float(np.trapezoid(-A * J_x * E_x, x)) / 1e6
    wall_loading_MW_m2 = compute_wall_loading_square(A, eta, J_x, J_y) / 1e6
    electric_field_min_V_m = float(np.nanmin(E_x))
    electric_field_max_abs_V_m = float(np.nanmax(np.abs(E_x)))

    if velikhov_margin is None:
        inlet_velikhov_margin = float("nan")
        outlet_velikhov_margin = float("nan")
        max_abs_velikhov_drift = float("nan")
    else:
        G = _to_float_array(velikhov_margin)
        inlet_velikhov_margin = float(G[0])
        outlet_velikhov_margin = float(G[-1])
        max_abs_velikhov_drift = float(np.nanmax(np.abs(G - G[0])))

    if furnace_power_MW is None:
        furnace_power_MW = float("nan")
        steam_input_power_MW = float("nan")
        steam_output_power_MWe = float("nan")
        total_plant_power_MWe = float("nan")
        chimney_loss_MW = float("nan")
        total_plant_efficiency = float("nan")
    else:
        steam_input_power_MW = float(furnace_power_MW) - mhd_output_power_MWe
        if steam_cycle_efficiency is None:
            steam_output_power_MWe = float("nan")
            total_plant_power_MWe = float("nan")
            chimney_loss_MW = float("nan")
            total_plant_efficiency = float("nan")
        else:
            steam_output_power_MWe = float(steam_cycle_efficiency) * steam_input_power_MW
            total_plant_power_MWe = mhd_output_power_MWe + steam_output_power_MWe
            chimney_loss_MW = float(furnace_power_MW) - total_plant_power_MWe
            total_plant_efficiency = (
                total_plant_power_MWe / float(furnace_power_MW)
                if furnace_power_MW != 0.0
                else float("nan")
            )

    return GlobalEngineeringMetrics(
        length_m=length_m,
        inlet_area_m2=inlet_area_m2,
        outlet_area_m2=outlet_area_m2,
        inlet_height_m=inlet_height_m,
        outlet_height_m=outlet_height_m,
        volume_m3=volume_m3,
        magnetic_energy_MJ=magnetic_energy_MJ,
        inlet_power_density_MW_m3=inlet_power_density_MW_m3,
        mhd_output_power_MWe=mhd_output_power_MWe,
        wall_loading_MW_m2=wall_loading_MW_m2,
        electric_field_min_V_m=electric_field_min_V_m,
        electric_field_max_abs_V_m=electric_field_max_abs_V_m,
        inlet_velikhov_margin=inlet_velikhov_margin,
        outlet_velikhov_margin=outlet_velikhov_margin,
        max_abs_velikhov_drift=max_abs_velikhov_drift,
        furnace_power_MW=float(furnace_power_MW),
        steam_input_power_MW=float(steam_input_power_MW),
        steam_output_power_MWe=float(steam_output_power_MWe),
        total_plant_power_MWe=float(total_plant_power_MWe),
        chimney_loss_MW=float(chimney_loss_MW),
        steam_cycle_efficiency=(
            float(steam_cycle_efficiency)
            if steam_cycle_efficiency is not None
            else float("nan")
        ),
        total_plant_efficiency=float(total_plant_efficiency),
    )


def compute_design_value_terms(
    *,
    x,
    T_e,
    T_p,
    n_p,
    n_e,
    mach,
    A,
    J_x,
    E_x,
    B,
    seed_fraction: float,
    v_p=None,
    heavy_particle_mass_kg: float = M_P,
) -> DesignValueTerms:
    x = _to_float_array(x)
    T_e = _to_float_array(T_e)
    T_p = _to_float_array(T_p)
    n_p = _to_float_array(n_p)
    n_e = _to_float_array(n_e)
    mach = _to_float_array(mach)
    A = _to_float_array(A)
    J_x = _to_float_array(J_x)
    E_x = _to_float_array(E_x)
    v_p_arr = None if v_p is None else _to_float_array(v_p)

    if len(x) == 0:
        raise ValueError("profile is empty")
    if v_p_arr is not None and v_p_arr.shape != x.shape:
        raise ValueError("v_p must be omitted or a same-shape 1D array")

    B_arr = np.asarray(B, dtype=float)
    if B_arr.ndim == 0:
        B_mag = abs(float(B_arr))
    elif B_arr.ndim == 1 and B_arr.shape == x.shape:
        B_mag = float(np.nanmean(np.abs(B_arr)))
    else:
        raise ValueError("B must be scalar or same-shape 1D array")

    outlet_delta_te_K = float(T_e[-1] - T_e[0])
    outlet_delta_ratio = float(T_e[-1] / max(T_p[-1], 1e-30) - 1.0)
    inlet_delta_ratio = float(T_e[0] / max(T_p[0], 1e-30) - 1.0)
    seed_density_out = float(seed_fraction) * float(n_p[-1])
    outlet_f_ion = _safe_ratio(float(n_e[-1]), seed_density_out)
    mhd_output_power_W = float(np.trapezoid(-A * J_x * E_x, x))
    mhd_output_power_MWe = mhd_output_power_W / 1e6
    outlet_mhd_output_per_100MWe = mhd_output_power_W / 1e8
    if v_p_arr is None:
        inlet_enthalpy_flux_MW = float("nan")
        outlet_enthalpy_extraction_ratio = float("nan")
    else:
        inlet_enthalpy_flux_W = inlet_stagnation_enthalpy_flux_W(
            n_p_in=float(n_p[0]),
            T_p_in=float(T_p[0]),
            T_e_in=float(T_e[0]),
            n_e_in=float(n_e[0]),
            v_p_in=float(v_p_arr[0]),
            A_in=float(A[0]),
            heavy_particle_mass_kg=float(heavy_particle_mass_kg),
        )
        inlet_enthalpy_flux_MW = inlet_enthalpy_flux_W / 1e6
        outlet_enthalpy_extraction_ratio = _safe_ratio(
            mhd_output_power_W,
            inlet_enthalpy_flux_W,
        )
    length_m = float(x[-1] - x[0]) if len(x) >= 2 else 0.0

    return DesignValueTerms(
        outlet_delta_te_K=outlet_delta_te_K,
        outlet_delta_te_per_kK=outlet_delta_te_K / 1e3,
        outlet_delta_ratio=outlet_delta_ratio,
        outlet_f_ion=outlet_f_ion,
        outlet_mhd_output_per_100MWe=outlet_mhd_output_per_100MWe,
        mhd_output_power_MWe=mhd_output_power_MWe,
        inlet_enthalpy_flux_MW=inlet_enthalpy_flux_MW,
        outlet_enthalpy_extraction_ratio=outlet_enthalpy_extraction_ratio,
        outlet_enthalpy_extraction_percent=100.0 * outlet_enthalpy_extraction_ratio,
        inlet_delta_ratio=inlet_delta_ratio,
        inlet_mach=float(mach[0]),
        magnetic_field_T=B_mag,
        device_length_m=length_m,
        device_length_per_5m=length_m / 5.0,
    )


def design_value_weights_delta_te_only() -> DesignValueWeights:
    return DesignValueWeights(
        outlet_delta_te_per_kK=1.0,
    )


def design_value_weights_lab_poc() -> DesignValueWeights:
    return DesignValueWeights(
        outlet_delta_te_per_kK=1.0,
        outlet_delta_ratio=0.35,
        outlet_f_ion=0.35,
        outlet_mhd_output_per_100MWe=0.15,
        inlet_delta_ratio_penalty=0.75,
        inlet_mach_penalty=0.10,
        magnetic_field_T_penalty=0.05,
        device_length_per_5m_penalty=0.50,
    )


def evaluate_design_value(
    terms: DesignValueTerms,
    *,
    weights: DesignValueWeights,
    profile_name: str,
) -> DesignValueBreakdown:
    contributions = {
        "reward_outlet_delta_te": float(weights.outlet_delta_te_per_kK) * float(terms.outlet_delta_te_per_kK),
        "reward_outlet_delta_ratio": float(weights.outlet_delta_ratio) * float(terms.outlet_delta_ratio),
        "reward_outlet_f_ion": float(weights.outlet_f_ion) * float(terms.outlet_f_ion),
        "reward_outlet_mhd_output": float(weights.outlet_mhd_output_per_100MWe) * float(terms.outlet_mhd_output_per_100MWe),
        "penalty_inlet_delta_ratio": -float(weights.inlet_delta_ratio_penalty) * float(terms.inlet_delta_ratio),
        "penalty_inlet_mach": -float(weights.inlet_mach_penalty) * float(terms.inlet_mach),
        "penalty_magnetic_field": -float(weights.magnetic_field_T_penalty) * float(terms.magnetic_field_T),
        "penalty_device_length": -float(weights.device_length_per_5m_penalty) * float(terms.device_length_per_5m),
    }
    total = float(sum(contributions.values()))
    return DesignValueBreakdown(
        profile_name=str(profile_name),
        total_score=total,
        contributions=contributions,
        terms=terms,
        weights=weights,
    )
