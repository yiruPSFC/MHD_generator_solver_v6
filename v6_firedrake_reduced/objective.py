from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .design import CaseConfig, DesignVector
from .legacy_physics import closure_state, inlet_design_generic, ops_for_numeric
from .transport import working_fluid_for_config


DEFAULT_VELIKHOV_FLOOR = 5e-7
DEFAULT_VELIKHOV_PENALTY_SCALE = 1e-2
DEFAULT_VELIKHOV_PENALTY_WEIGHT = 25.0


@dataclass(frozen=True)
class ProfileMetrics:
    objective_score: float
    raw_enthalpy_extraction_percent: float
    outlet_enthalpy_extraction_percent: float
    outlet_Te_over_Tp: float
    outlet_T_e_K: float
    outlet_T_p_K: float
    outlet_mach: float
    min_T_p_K: float
    min_mach: float
    min_velikhov_margin: float
    velikhov_passes_floor: bool
    velikhov_floor: float
    velikhov_penalty: float
    velikhov_penalty_candidate: float
    velikhov_penalty_scale: float
    velikhov_penalty_weight: float
    hall_voltage_V: float
    electric_power_from_hall_W: float
    mhd_output_power_W: float
    inlet_enthalpy_flux_W: float
    finite_profile: bool

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "objective_score": float(self.objective_score),
            "raw_enthalpy_extraction_percent": float(self.raw_enthalpy_extraction_percent),
            "outlet_enthalpy_extraction_percent": float(self.outlet_enthalpy_extraction_percent),
            "outlet_Te_over_Tp": float(self.outlet_Te_over_Tp),
            "outlet_T_e_K": float(self.outlet_T_e_K),
            "outlet_T_p_K": float(self.outlet_T_p_K),
            "outlet_mach": float(self.outlet_mach),
            "min_T_p_K": float(self.min_T_p_K),
            "min_mach": float(self.min_mach),
            "min_velikhov_margin": float(self.min_velikhov_margin),
            "velikhov_passes_floor": bool(self.velikhov_passes_floor),
            "velikhov_floor": float(self.velikhov_floor),
            "velikhov_penalty": float(self.velikhov_penalty),
            "velikhov_penalty_candidate": float(self.velikhov_penalty_candidate),
            "velikhov_penalty_scale": float(self.velikhov_penalty_scale),
            "velikhov_penalty_weight": float(self.velikhov_penalty_weight),
            "hall_voltage_V": float(self.hall_voltage_V),
            "electric_power_from_hall_W": float(self.electric_power_from_hall_W),
            "mhd_output_power_W": float(self.mhd_output_power_W),
            "inlet_enthalpy_flux_W": float(self.inlet_enthalpy_flux_W),
            "finite_profile": bool(self.finite_profile),
        }


def velikhov_settings(config: CaseConfig) -> dict[str, float | bool | str]:
    mode = str(config.metadata.get("velikhov_mode", "diagnostic")).lower()
    if mode not in {"diagnostic", "penalty"}:
        raise ValueError("velikhov_mode metadata must be 'diagnostic' or 'penalty'.")
    return {
        "mode": mode,
        "active": mode == "penalty",
        "floor": float(config.metadata.get("velikhov_floor", DEFAULT_VELIKHOV_FLOOR)),
        "scale": float(config.metadata.get("velikhov_penalty_scale", DEFAULT_VELIKHOV_PENALTY_SCALE)),
        "weight": float(config.metadata.get("velikhov_penalty_weight", DEFAULT_VELIKHOV_PENALTY_WEIGHT)),
    }


def _velikhov_penalty_from_array(*, x: np.ndarray, velikhov: np.ndarray, config: CaseConfig) -> tuple[float, float, float, float]:
    settings = velikhov_settings(config)
    floor = float(settings["floor"])
    scale = max(float(settings["scale"]), 1e-300)
    weight = float(settings["weight"])
    shortfall = np.maximum(floor - np.asarray(velikhov, dtype=float), 0.0)
    scaled_sq = (shortfall / scale) ** 2
    length = max(float(x[-1] - x[0]), 1e-300)
    mean_scaled_sq = float(np.trapezoid(scaled_sq, np.asarray(x, dtype=float)) / length)
    return weight * mean_scaled_sq, floor, scale, weight


def _inlet_enthalpy_flux(*, inlet: dict[str, float], config: CaseConfig) -> float:
    fluid = working_fluid_for_config(config)
    n_p = max(float(inlet["n_p"]), 1.0)
    n_e = max(float(inlet["n_e"]), 0.0)
    T_p = max(float(inlet["T_p"]), 1.0)
    T_e = max(float(inlet["T_e"]), 1.0)
    v = max(float(inlet["v_in"]), 1e-30)
    A = max(float(inlet["A_in"]), 1e-30)
    thermal_density = 2.5 * 1.380649e-23 * (n_p * T_p + n_e * T_e)
    kinetic_density = 0.5 * float(fluid.heavy_particle_mass_kg) * n_p * v * v
    return float(A * v * (thermal_density + kinetic_density))


def evaluate_profile_metrics(
    *,
    profile: dict[str, Any],
    design: DesignVector,
    config: CaseConfig,
) -> ProfileMetrics:
    ops = ops_for_numeric()
    fluid = working_fluid_for_config(config)
    x = np.asarray(profile["x"], dtype=float).reshape(-1)
    n_p = np.asarray(profile["n_p"], dtype=float).reshape(-1)
    T_e = np.asarray(profile["T_e"], dtype=float).reshape(-1)
    A = np.asarray(profile["A"], dtype=float).reshape(-1)
    sigma = np.asarray(profile["sigma_logA"], dtype=float).reshape(-1)
    if not (x.size == n_p.size == T_e.size == A.size == sigma.size):
        raise ValueError("profile arrays x, n_p, T_e, A, and sigma_logA must have matching lengths.")

    inlet = inlet_design_generic(
        ops=ops,
        n_p_in=design.n_p_in,
        T_e_in=float(design.T_e_in),
        Z_in=float(design.Z_in),
        I_0=float(design.I_0),
        seed_fraction=design.seed_fraction,
        B=float(config.B_T),
        inlet_A=float(config.area_scale_m2),
        working_fluid=fluid,
    )
    closures = []
    power_density = []
    hall_field = []
    for n_val, te_val, area_val, sigma_val in zip(n_p, T_e, A, sigma, strict=True):
        closure = closure_state(
            ops=ops,
            n_p=float(n_val),
            T_e=float(te_val),
            A=float(area_val),
            dot_N=float(inlet["dot_N"]),
            I_0=float(design.I_0),
            seed_fraction=design.seed_fraction,
            B=float(config.B_T),
            working_fluid=fluid,
        )
        closures.append(closure)
        power_density.append(float(-float(area_val) * float(closure["J_x"]) * float(closure["E_x"])))
        hall_field.append(float(-float(closure["E_x"])))

    T_p = np.array([float(item["T_p"]) for item in closures], dtype=float)
    mach = np.array([float(item["mach"]) for item in closures], dtype=float)
    velikhov = np.array([float(item["G"]) for item in closures], dtype=float)
    power_density_arr = np.asarray(power_density, dtype=float)
    hall_field_arr = np.asarray(hall_field, dtype=float)
    mhd_output_power_W = float(np.trapezoid(power_density_arr, x))
    hall_voltage = float(np.trapezoid(hall_field_arr, x))
    electric_power_from_hall_W = float(float(design.I_0) * hall_voltage)
    inlet_flux_W = _inlet_enthalpy_flux(inlet={key: float(value) for key, value in inlet.items()}, config=config)
    enthalpy_percent = float(100.0 * mhd_output_power_W / max(inlet_flux_W, 1e-30))
    penalty_candidate, velikhov_floor, velikhov_scale, velikhov_weight = _velikhov_penalty_from_array(
        x=x,
        velikhov=velikhov,
        config=config,
    )
    settings = velikhov_settings(config)
    velikhov_penalty = float(penalty_candidate if bool(settings["active"]) else 0.0)
    outlet_Tp = float(T_p[-1])
    metrics = ProfileMetrics(
        objective_score=float(enthalpy_percent - velikhov_penalty),
        raw_enthalpy_extraction_percent=enthalpy_percent,
        outlet_enthalpy_extraction_percent=enthalpy_percent,
        outlet_Te_over_Tp=float(T_e[-1] / max(outlet_Tp, 1.0)),
        outlet_T_e_K=float(T_e[-1]),
        outlet_T_p_K=outlet_Tp,
        outlet_mach=float(mach[-1]),
        min_T_p_K=float(np.nanmin(T_p)),
        min_mach=float(np.nanmin(mach)),
        min_velikhov_margin=float(np.nanmin(velikhov)),
        velikhov_passes_floor=bool(np.nanmin(velikhov) >= velikhov_floor),
        velikhov_floor=velikhov_floor,
        velikhov_penalty=velikhov_penalty,
        velikhov_penalty_candidate=float(penalty_candidate),
        velikhov_penalty_scale=velikhov_scale,
        velikhov_penalty_weight=velikhov_weight,
        hall_voltage_V=hall_voltage,
        electric_power_from_hall_W=electric_power_from_hall_W,
        mhd_output_power_W=mhd_output_power_W,
        inlet_enthalpy_flux_W=inlet_flux_W,
        finite_profile=bool(
            np.all(np.isfinite(x))
            and np.all(np.isfinite(n_p))
            and np.all(np.isfinite(T_e))
            and np.all(np.isfinite(A))
            and np.all(np.isfinite(T_p))
            and np.all(np.isfinite(mach))
            and np.all(np.isfinite(velikhov))
        ),
    )
    return metrics
