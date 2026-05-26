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
DEFAULT_THERMAL_TP_PENALTY_SCALE_K = 100.0
DEFAULT_THERMAL_RATIO_PENALTY_SCALE = 1.0
DEFAULT_THERMAL_PENALTY_WEIGHT = 1.0


@dataclass(frozen=True)
class ProfileMetrics:
    objective_score: float
    raw_enthalpy_extraction_percent: float
    outlet_enthalpy_extraction_percent: float
    outlet_Te_over_Tp: float
    outlet_T_e_K: float
    outlet_T_p_K: float
    outlet_mach: float
    inlet_T_p_K: float
    min_T_p_K: float
    max_T_p_K: float
    min_Te_over_Tp: float
    max_Te_over_Tp: float
    min_mach: float
    min_velikhov_margin: float
    velikhov_passes_floor: bool
    velikhov_floor: float
    velikhov_penalty: float
    velikhov_penalty_candidate: float
    velikhov_penalty_scale: float
    velikhov_penalty_weight: float
    thermal_window_active: bool
    thermal_window_passes: bool
    thermal_window_penalty: float
    thermal_window_penalty_candidate: float
    thermal_Tp_in_penalty_candidate: float
    thermal_Tp_low_penalty_candidate: float
    thermal_Tp_high_penalty_candidate: float
    thermal_Te_over_Tp_penalty_candidate: float
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
            "inlet_T_p_K": float(self.inlet_T_p_K),
            "min_T_p_K": float(self.min_T_p_K),
            "max_T_p_K": float(self.max_T_p_K),
            "min_Te_over_Tp": float(self.min_Te_over_Tp),
            "max_Te_over_Tp": float(self.max_Te_over_Tp),
            "min_mach": float(self.min_mach),
            "min_velikhov_margin": float(self.min_velikhov_margin),
            "velikhov_passes_floor": bool(self.velikhov_passes_floor),
            "velikhov_floor": float(self.velikhov_floor),
            "velikhov_penalty": float(self.velikhov_penalty),
            "velikhov_penalty_candidate": float(self.velikhov_penalty_candidate),
            "velikhov_penalty_scale": float(self.velikhov_penalty_scale),
            "velikhov_penalty_weight": float(self.velikhov_penalty_weight),
            "thermal_window_active": bool(self.thermal_window_active),
            "thermal_window_passes": bool(self.thermal_window_passes),
            "thermal_window_penalty": float(self.thermal_window_penalty),
            "thermal_window_penalty_candidate": float(self.thermal_window_penalty_candidate),
            "thermal_Tp_in_penalty_candidate": float(self.thermal_Tp_in_penalty_candidate),
            "thermal_Tp_low_penalty_candidate": float(self.thermal_Tp_low_penalty_candidate),
            "thermal_Tp_high_penalty_candidate": float(self.thermal_Tp_high_penalty_candidate),
            "thermal_Te_over_Tp_penalty_candidate": float(self.thermal_Te_over_Tp_penalty_candidate),
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


def _optional_finite_float(config: CaseConfig, key: str) -> float | None:
    value = config.metadata.get(key)
    if value is None:
        return None
    value_float = float(value)
    if not np.isfinite(value_float):
        return None
    return value_float


def thermal_window_settings(config: CaseConfig) -> dict[str, float | bool | str | None]:
    mode = str(config.metadata.get("thermal_window_mode", "diagnostic")).lower()
    if mode not in {"diagnostic", "penalty"}:
        raise ValueError("thermal_window_mode metadata must be 'diagnostic' or 'penalty'.")
    return {
        "mode": mode,
        "active": mode == "penalty",
        "tp_in_max_K": _optional_finite_float(config, "thermal_tp_in_max_K"),
        "tp_floor_K": _optional_finite_float(config, "thermal_tp_floor_K"),
        "tp_path_max_K": _optional_finite_float(config, "thermal_tp_path_max_K"),
        "te_over_tp_min": _optional_finite_float(config, "thermal_te_over_tp_min"),
        "te_over_tp_max": _optional_finite_float(config, "thermal_te_over_tp_max"),
        "tp_scale_K": max(
            float(config.metadata.get("thermal_tp_penalty_scale_K", DEFAULT_THERMAL_TP_PENALTY_SCALE_K)),
            1e-300,
        ),
        "ratio_scale": max(
            float(config.metadata.get("thermal_te_over_tp_penalty_scale", DEFAULT_THERMAL_RATIO_PENALTY_SCALE)),
            1e-300,
        ),
        "tp_in_weight": float(
            config.metadata.get("thermal_tp_in_penalty_weight", DEFAULT_THERMAL_PENALTY_WEIGHT)
        ),
        "tp_path_weight": float(
            config.metadata.get("thermal_tp_path_penalty_weight", DEFAULT_THERMAL_PENALTY_WEIGHT)
        ),
        "ratio_weight": float(
            config.metadata.get("thermal_te_over_tp_penalty_weight", DEFAULT_THERMAL_PENALTY_WEIGHT)
        ),
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


def _mean_path_penalty(*, x: np.ndarray, scaled_violation: np.ndarray, weight: float) -> float:
    length = max(float(x[-1] - x[0]), 1e-300)
    return float(weight) * float(np.trapezoid(np.asarray(scaled_violation, dtype=float) ** 2, x) / length)


def _thermal_window_penalty_from_arrays(
    *,
    x: np.ndarray,
    T_e: np.ndarray,
    T_p: np.ndarray,
    inlet_T_p: float,
    config: CaseConfig,
) -> dict[str, float | bool]:
    settings = thermal_window_settings(config)
    tp_scale = float(settings["tp_scale_K"])
    ratio_scale = float(settings["ratio_scale"])
    tp_in_weight = float(settings["tp_in_weight"])
    tp_path_weight = float(settings["tp_path_weight"])
    ratio_weight = float(settings["ratio_weight"])
    x_arr = np.asarray(x, dtype=float)
    T_e_arr = np.asarray(T_e, dtype=float)
    T_p_arr = np.asarray(T_p, dtype=float)
    ratio = T_e_arr / np.maximum(T_p_arr, 1.0)

    tp_in_penalty = 0.0
    tp_in_max = settings["tp_in_max_K"]
    if tp_in_max is not None:
        tp_in_penalty = tp_in_weight * (max(float(inlet_T_p) - float(tp_in_max), 0.0) / tp_scale) ** 2

    tp_low_penalty = 0.0
    tp_floor = settings["tp_floor_K"]
    if tp_floor is not None:
        tp_low_penalty = _mean_path_penalty(
            x=x_arr,
            scaled_violation=np.maximum(float(tp_floor) - T_p_arr, 0.0) / tp_scale,
            weight=tp_path_weight,
        )

    tp_high_penalty = 0.0
    tp_path_max = settings["tp_path_max_K"]
    if tp_path_max is not None:
        tp_high_penalty = _mean_path_penalty(
            x=x_arr,
            scaled_violation=np.maximum(T_p_arr - float(tp_path_max), 0.0) / tp_scale,
            weight=tp_path_weight,
        )

    ratio_low = np.zeros_like(ratio)
    ratio_min = settings["te_over_tp_min"]
    if ratio_min is not None:
        ratio_low = np.maximum(float(ratio_min) - ratio, 0.0) / ratio_scale
    ratio_high = np.zeros_like(ratio)
    ratio_max = settings["te_over_tp_max"]
    if ratio_max is not None:
        ratio_high = np.maximum(ratio - float(ratio_max), 0.0) / ratio_scale
    ratio_penalty = _mean_path_penalty(
        x=x_arr,
        scaled_violation=ratio_low + ratio_high,
        weight=ratio_weight,
    )

    candidate = float(tp_in_penalty + tp_low_penalty + tp_high_penalty + ratio_penalty)
    active = bool(settings["active"])
    return {
        "active": active,
        "passes": bool(candidate <= 0.0),
        "penalty": float(candidate if active else 0.0),
        "candidate": candidate,
        "tp_in": float(tp_in_penalty),
        "tp_low": float(tp_low_penalty),
        "tp_high": float(tp_high_penalty),
        "ratio": float(ratio_penalty),
    }


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
        B=float(design.B_T),
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
            B=float(design.B_T),
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
    thermal_penalty = _thermal_window_penalty_from_arrays(
        x=x,
        T_e=T_e,
        T_p=T_p,
        inlet_T_p=float(inlet["T_p"]),
        config=config,
    )
    outlet_Tp = float(T_p[-1])
    te_over_tp = T_e / np.maximum(T_p, 1.0)
    metrics = ProfileMetrics(
        objective_score=float(enthalpy_percent - velikhov_penalty - float(thermal_penalty["penalty"])),
        raw_enthalpy_extraction_percent=enthalpy_percent,
        outlet_enthalpy_extraction_percent=enthalpy_percent,
        outlet_Te_over_Tp=float(T_e[-1] / max(outlet_Tp, 1.0)),
        outlet_T_e_K=float(T_e[-1]),
        outlet_T_p_K=outlet_Tp,
        outlet_mach=float(mach[-1]),
        inlet_T_p_K=float(inlet["T_p"]),
        min_T_p_K=float(np.nanmin(T_p)),
        max_T_p_K=float(np.nanmax(T_p)),
        min_Te_over_Tp=float(np.nanmin(te_over_tp)),
        max_Te_over_Tp=float(np.nanmax(te_over_tp)),
        min_mach=float(np.nanmin(mach)),
        min_velikhov_margin=float(np.nanmin(velikhov)),
        velikhov_passes_floor=bool(np.nanmin(velikhov) >= velikhov_floor),
        velikhov_floor=velikhov_floor,
        velikhov_penalty=velikhov_penalty,
        velikhov_penalty_candidate=float(penalty_candidate),
        velikhov_penalty_scale=velikhov_scale,
        velikhov_penalty_weight=velikhov_weight,
        thermal_window_active=bool(thermal_penalty["active"]),
        thermal_window_passes=bool(thermal_penalty["passes"]),
        thermal_window_penalty=float(thermal_penalty["penalty"]),
        thermal_window_penalty_candidate=float(thermal_penalty["candidate"]),
        thermal_Tp_in_penalty_candidate=float(thermal_penalty["tp_in"]),
        thermal_Tp_low_penalty_candidate=float(thermal_penalty["tp_low"]),
        thermal_Tp_high_penalty_candidate=float(thermal_penalty["tp_high"]),
        thermal_Te_over_Tp_penalty_candidate=float(thermal_penalty["ratio"]),
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
            and np.all(np.isfinite(te_over_tp))
        ),
    )
    return metrics
