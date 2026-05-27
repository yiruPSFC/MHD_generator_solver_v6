from __future__ import annotations

from dataclasses import dataclass, field
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

from .design import CaseConfig, DesignVector, OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION
from .legacy_physics import (
    dynamic_system_terms,
    inlet_design_generic,
    ops_for_numeric,
)
from .objective import ProfileMetrics, evaluate_profile_metrics, thermal_window_settings, velikhov_settings
from .transport import working_fluid_for_config


K_B = 1.380649e-23
EQUATION_FORMS = ("primitive", "freidberg_hl")


class FiredrakeUnavailableError(RuntimeError):
    """Raised when a Firedrake-only path is requested without Firedrake."""


@dataclass
class ForwardResult:
    ok: bool
    design: DesignVector
    config: CaseConfig
    profile: dict[str, np.ndarray] | None
    metrics: ProfileMetrics | None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    fd_objective: Any | None = None
    fd_controls: dict[str, Any] = field(default_factory=dict)
    fd_control_space: Any | None = None


def _require_firedrake():
    original_argv = sys.argv[:]
    try:
        sys.argv = ["v6_firedrake_reduced"]
        import firedrake as fd  # type: ignore
    except ImportError as exc:
        raise FiredrakeUnavailableError(
            "Firedrake is not installed in this Python environment. "
            "Create/use .venv_firedrake and install Firedrake before running the forward solver."
        ) from exc
    finally:
        sys.argv = original_argv
    return fd


def _ops_for_firedrake(fd):
    log_fn = getattr(fd, "ln", None)
    if log_fn is None:
        log_fn = getattr(fd, "log")
    return SimpleNamespace(
        exp=fd.exp,
        log=log_fn,
        sqrt=fd.sqrt,
        fabs=lambda value: abs(value),
        max=fd.max_value,
        min=fd.min_value,
        pos=None,
        neg=None,
        lb_func=None,
        ub_func=None,
        bounding_func=None,
    )


def _basis_function(fd, V, values: np.ndarray, *, name: str):
    func = fd.Function(V, name=name)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if func.dat.data.shape[0] != arr.shape[0]:
        raise RuntimeError(
            f"cannot seed Firedrake basis function {name!r}: expected {func.dat.data.shape[0]} values, got {arr.shape[0]}"
        )
    func.dat.data[:] = arr
    return func


def _initial_delta_arrays(
    *,
    initial_profile: dict[str, Any],
    design: DesignVector,
    target_x_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_target = int(np.asarray(target_x_norm, dtype=float).size)
    if "n_p" not in initial_profile or "T_e" not in initial_profile:
        raise ValueError("initial_profile must contain n_p and T_e arrays.")
    n_p_values = np.asarray(initial_profile["n_p"], dtype=float).reshape(-1)
    T_e_values = np.asarray(initial_profile["T_e"], dtype=float).reshape(-1)
    if "x_norm" in initial_profile:
        source_x = np.asarray(initial_profile["x_norm"], dtype=float).reshape(-1)
    elif "x" in initial_profile:
        source_x_raw = np.asarray(initial_profile["x"], dtype=float).reshape(-1)
        span = float(source_x_raw[-1] - source_x_raw[0]) if source_x_raw.size >= 2 else 0.0
        source_x = (source_x_raw - float(source_x_raw[0])) / max(span, 1e-300)
    else:
        source_x = np.linspace(0.0, 1.0, n_p_values.size, dtype=float)
    if not (source_x.size == n_p_values.size == T_e_values.size):
        raise ValueError("initial_profile x/x_norm, n_p, and T_e arrays must have matching lengths.")
    if source_x.size < 2:
        raise ValueError("initial_profile must contain at least two nodes.")
    if not (
        np.all(np.isfinite(source_x))
        and np.all(np.isfinite(n_p_values))
        and np.all(np.isfinite(T_e_values))
        and np.all(n_p_values > 0.0)
        and np.all(T_e_values > 0.0)
    ):
        raise ValueError("initial_profile contains non-finite or non-positive n_p/T_e values.")

    order = np.argsort(source_x)
    source_x = np.asarray(source_x[order], dtype=float)
    n_p_values = np.asarray(n_p_values[order], dtype=float)
    T_e_values = np.asarray(T_e_values[order], dtype=float)
    target = np.asarray(target_x_norm, dtype=float).reshape(n_target)
    n_interp = np.interp(target, source_x, n_p_values)
    T_interp = np.interp(target, source_x, T_e_values)
    delta_log_n = np.log(np.maximum(n_interp, 1e-300)) - float(design.log_n_p_in)
    delta_log_Te = np.log(np.maximum(T_interp, 1e-300)) - float(np.log(max(float(design.T_e_in), 1.0)))
    delta_log_n[0] = 0.0
    delta_log_Te[0] = 0.0
    return delta_log_n, delta_log_Te


def _design_coefficients(fd, mesh, design: DesignVector, *, as_functions: bool) -> tuple[dict[str, Any], Any | None]:
    if not as_functions:
        return {key: fd.Constant(float(value)) for key, value in design.to_dict().items()}, None

    control_space = fd.FunctionSpace(mesh, "R", 0)
    controls = {}
    for key, value in design.to_dict().items():
        control = fd.Function(control_space, name=key)
        control.assign(float(value))
        controls[key] = control
    return controls, control_space


def _residual_scaling_mode(config: CaseConfig) -> str:
    mode = str(config.metadata.get("residual_scaling", "inlet")).lower()
    if mode not in {"inlet", "characteristic", "dimensional"}:
        raise ValueError("residual_scaling metadata must be 'inlet', 'characteristic', or 'dimensional'.")
    return mode


def _equation_form(config: CaseConfig) -> str:
    form = str(config.metadata.get("equation_form", "primitive")).strip().lower()
    if form not in EQUATION_FORMS:
        raise ValueError(f"equation_form metadata must be one of {EQUATION_FORMS!r}.")
    return form


def _max_abs_expr(ops, *values: Any, floor: float = 1.0) -> Any:
    scale = float(floor)
    for value in values:
        scale = ops.max(scale, ops.fabs(value))
    return scale


def _row_scale_exprs(
    *,
    ops,
    terms: dict[str, Any],
    n_p: Any,
    T_e: Any,
    length_m: float,
) -> tuple[Any, Any]:
    inverse_length = 1.0 / max(float(length_m), 1e-30)
    momentum_scale = _max_abs_expr(
        ops,
        terms["M11"] * n_p * inverse_length,
        terms["M12"] * T_e * inverse_length,
        terms["rhs_m"],
    )
    energy_scale = _max_abs_expr(
        ops,
        terms["E11"] * n_p * inverse_length,
        terms["E12"] * T_e * inverse_length,
        terms["rhs_e"],
    )
    return momentum_scale, energy_scale


def _freidberg_balance_terms(
    *,
    ops,
    closure: dict[str, Any],
    A: Any,
    B_T: Any,
    area_scale_m2: float,
    heavy_particle_mass_kg: float,
    length_m: float | None = None,
) -> dict[str, Any]:
    A0 = max(float(area_scale_m2), 1e-300)
    mp = float(heavy_particle_mass_kg)
    J2 = closure["J_x"] * closure["J_x"] + closure["J_y"] * closure["J_y"]
    v_p = closure["v_p"]
    T_p = closure["T_p"]
    T_p_safe = ops.max(T_p, 1.0)
    n_p = closure["n_p_safe"]
    mach = closure["mach"]
    M2 = mach * mach
    H_p = (A * n_p * v_p / A0) * (2.5 * K_B * T_p_safe + 0.5 * mp * v_p * v_p)
    L_p = mach * (A / A0) / ((M2 + 3.0) * (M2 + 3.0))
    rhs_H = (A / A0) * (v_p * closure["J_y"] * B_T + closure["eta"] * J2)
    p_p = n_p * K_B * T_p_safe
    denom = ops.max((M2 + 3.0) * p_p * v_p, 1e-300)
    rhs_L = (
        -(12.0 / 5.0)
        * L_p
        / denom
        * (v_p * closure["J_y"] * B_T - ((5.0 * M2 + 3.0) / 12.0) * closure["eta"] * J2)
    )
    balances = {"H_p": H_p, "L_p": L_p, "rhs_H": rhs_H, "rhs_L": rhs_L, "M2": M2}
    if length_m is not None:
        h_scale, l_scale = _freidberg_row_scale_exprs(
            ops=ops,
            balances=balances,
            length_m=float(length_m),
        )
        balances["H_scale"] = h_scale
        balances["L_scale"] = l_scale
    return balances


def _freidberg_row_scale_exprs(*, ops, balances: dict[str, Any], length_m: float) -> tuple[Any, Any]:
    inverse_length = 1.0 / max(float(length_m), 1e-30)
    h_scale = _max_abs_expr(
        ops,
        balances["H_p"] * inverse_length,
        balances["rhs_H"],
        floor=1.0,
    )
    l_scale = _max_abs_expr(
        ops,
        balances["L_p"] * inverse_length,
        balances["rhs_L"],
        floor=inverse_length,
    )
    return h_scale, l_scale


def _reference_residual_scale_arrays(*, design: DesignVector, config: CaseConfig) -> tuple[np.ndarray, np.ndarray]:
    ops = ops_for_numeric()
    fluid = working_fluid_for_config(config)
    area = design.area_control.evaluate_profile(
        length=float(config.length_m),
        n_intervals=int(config.n_intervals),
        area_scale=float(config.area_scale_m2),
    )
    inlet = inlet_design_generic(
        ops=ops,
        n_p_in=design.n_p_in,
        T_e_in=design.T_e_in,
        Z_in=design.Z_in,
        I_0=design.I_0,
        seed_fraction=design.seed_fraction,
        B=float(design.B_T),
        inlet_A=float(config.area_scale_m2),
        working_fluid=fluid,
    )
    momentum_scales = []
    energy_scales = []
    for A, sigma in zip(area["A"], area["sigma_logA"], strict=True):
        _, terms = dynamic_system_terms(
            ops=ops,
            n_p=design.n_p_in,
            T_e=design.T_e_in,
            A=float(A),
            sigma=float(sigma),
            dot_N=float(inlet["dot_N"]),
            I_0=float(design.I_0),
            seed_fraction=design.seed_fraction,
            B=float(design.B_T),
            working_fluid=fluid,
        )
        momentum_scale, energy_scale = _row_scale_exprs(
            ops=ops,
            terms=terms,
            n_p=design.n_p_in,
            T_e=design.T_e_in,
            length_m=float(config.length_m),
        )
        momentum_scales.append(float(momentum_scale))
        energy_scales.append(float(energy_scale))
    return np.asarray(momentum_scales, dtype=float), np.asarray(energy_scales, dtype=float)


def _reference_freidberg_scale_arrays(*, design: DesignVector, config: CaseConfig) -> tuple[np.ndarray, np.ndarray]:
    ops = ops_for_numeric()
    fluid = working_fluid_for_config(config)
    area = design.area_control.evaluate_profile(
        length=float(config.length_m),
        n_intervals=int(config.n_intervals),
        area_scale=float(config.area_scale_m2),
    )
    inlet = inlet_design_generic(
        ops=ops,
        n_p_in=design.n_p_in,
        T_e_in=design.T_e_in,
        Z_in=design.Z_in,
        I_0=design.I_0,
        seed_fraction=design.seed_fraction,
        B=float(design.B_T),
        inlet_A=float(config.area_scale_m2),
        working_fluid=fluid,
    )
    h_scales = []
    l_scales = []
    for A, sigma in zip(area["A"], area["sigma_logA"], strict=True):
        closure, _ = dynamic_system_terms(
            ops=ops,
            n_p=design.n_p_in,
            T_e=design.T_e_in,
            A=float(A),
            sigma=float(sigma),
            dot_N=float(inlet["dot_N"]),
            I_0=float(design.I_0),
            seed_fraction=design.seed_fraction,
            B=float(design.B_T),
            working_fluid=fluid,
        )
        balances = _freidberg_balance_terms(
            ops=ops,
            closure=closure,
            A=float(A),
            B_T=float(design.B_T),
            area_scale_m2=float(config.area_scale_m2),
            heavy_particle_mass_kg=float(fluid.heavy_particle_mass_kg),
        )
        h_scale, l_scale = _freidberg_row_scale_exprs(
            ops=ops,
            balances=balances,
            length_m=float(config.length_m),
        )
        h_scales.append(float(h_scale))
        l_scales.append(float(l_scale))
    return np.asarray(h_scales, dtype=float), np.asarray(l_scales, dtype=float)


def _inlet_residual_scale_values(*, design: DesignVector, config: CaseConfig) -> tuple[float, float]:
    ops = ops_for_numeric()
    fluid = working_fluid_for_config(config)
    area = design.area_control.evaluate_profile(
        length=float(config.length_m),
        n_intervals=int(config.n_intervals),
        area_scale=float(config.area_scale_m2),
    )
    inlet = inlet_design_generic(
        ops=ops,
        n_p_in=design.n_p_in,
        T_e_in=design.T_e_in,
        Z_in=design.Z_in,
        I_0=design.I_0,
        seed_fraction=design.seed_fraction,
        B=float(design.B_T),
        inlet_A=float(config.area_scale_m2),
        working_fluid=fluid,
    )
    _, terms = dynamic_system_terms(
        ops=ops,
        n_p=design.n_p_in,
        T_e=design.T_e_in,
        A=float(area["A"][0]),
        sigma=float(area["sigma_logA"][0]),
        dot_N=float(inlet["dot_N"]),
        I_0=float(design.I_0),
        seed_fraction=design.seed_fraction,
        B=float(design.B_T),
        working_fluid=fluid,
    )
    momentum_scale, energy_scale = _row_scale_exprs(
        ops=ops,
        terms=terms,
        n_p=design.n_p_in,
        T_e=design.T_e_in,
        length_m=float(config.length_m),
    )
    return float(momentum_scale), float(energy_scale)


def _inlet_freidberg_scale_values(*, design: DesignVector, config: CaseConfig) -> tuple[float, float]:
    h_scales, l_scales = _reference_freidberg_scale_arrays(design=design, config=config)
    return float(h_scales[0]), float(l_scales[0])


def _strong_residual_diagnostics(
    *,
    profile: dict[str, np.ndarray],
    design: DesignVector,
    config: CaseConfig,
) -> dict[str, Any]:
    ops = ops_for_numeric()
    fluid = working_fluid_for_config(config)
    x = np.asarray(profile["x"], dtype=float)
    n_p = np.asarray(profile["n_p"], dtype=float)
    T_e = np.asarray(profile["T_e"], dtype=float)
    A = np.asarray(profile["A"], dtype=float)
    sigma = np.asarray(profile["sigma_logA"], dtype=float)
    dn_dx = np.gradient(n_p, x, edge_order=1)
    dTe_dx = np.gradient(T_e, x, edge_order=1)
    inlet = inlet_design_generic(
        ops=ops,
        n_p_in=design.n_p_in,
        T_e_in=design.T_e_in,
        Z_in=design.Z_in,
        I_0=design.I_0,
        seed_fraction=design.seed_fraction,
        B=float(design.B_T),
        inlet_A=float(config.area_scale_m2),
        working_fluid=fluid,
    )
    momentum = []
    energy = []
    momentum_scales = []
    energy_scales = []
    for idx in range(x.size):
        _, terms = dynamic_system_terms(
            ops=ops,
            n_p=float(n_p[idx]),
            T_e=float(T_e[idx]),
            A=float(A[idx]),
            sigma=float(sigma[idx]),
            dot_N=float(inlet["dot_N"]),
            I_0=float(design.I_0),
            seed_fraction=design.seed_fraction,
            B=float(design.B_T),
            working_fluid=fluid,
        )
        m_scale, e_scale = _row_scale_exprs(
            ops=ops,
            terms=terms,
            n_p=float(n_p[idx]),
            T_e=float(T_e[idx]),
            length_m=float(config.length_m),
        )
        m_res = float(terms["M11"] * dn_dx[idx] + terms["M12"] * dTe_dx[idx] - terms["rhs_m"])
        e_res = float(terms["E11"] * dn_dx[idx] + terms["E12"] * dTe_dx[idx] - terms["rhs_e"])
        momentum.append(m_res)
        energy.append(e_res)
        momentum_scales.append(float(m_scale))
        energy_scales.append(float(e_scale))

    momentum_arr = np.asarray(momentum, dtype=float)
    energy_arr = np.asarray(energy, dtype=float)
    momentum_scale_arr = np.asarray(momentum_scales, dtype=float)
    energy_scale_arr = np.asarray(energy_scales, dtype=float)
    scaled_momentum = momentum_arr / np.maximum(momentum_scale_arr, 1e-300)
    scaled_energy = energy_arr / np.maximum(energy_scale_arr, 1e-300)
    return {
        "finite_difference_note": (
            "strong residual diagnostic only; solver residual is weak CG form; "
            "scaled values use local characteristic diagnostic scales"
        ),
        "max_abs_momentum_residual": float(np.nanmax(np.abs(momentum_arr))),
        "max_abs_energy_residual": float(np.nanmax(np.abs(energy_arr))),
        "max_abs_scaled_momentum_residual": float(np.nanmax(np.abs(scaled_momentum))),
        "max_abs_scaled_energy_residual": float(np.nanmax(np.abs(scaled_energy))),
        "momentum_scale_min": float(np.nanmin(momentum_scale_arr)),
        "momentum_scale_max": float(np.nanmax(momentum_scale_arr)),
        "energy_scale_min": float(np.nanmin(energy_scale_arr)),
        "energy_scale_max": float(np.nanmax(energy_scale_arr)),
    }


def _freidberg_strong_residual_diagnostics(
    *,
    profile: dict[str, np.ndarray],
    design: DesignVector,
    config: CaseConfig,
) -> dict[str, Any]:
    ops = ops_for_numeric()
    fluid = working_fluid_for_config(config)
    x = np.asarray(profile["x"], dtype=float)
    n_p = np.asarray(profile["n_p"], dtype=float)
    T_e = np.asarray(profile["T_e"], dtype=float)
    A = np.asarray(profile["A"], dtype=float)
    sigma = np.asarray(profile["sigma_logA"], dtype=float)
    inlet = inlet_design_generic(
        ops=ops,
        n_p_in=design.n_p_in,
        T_e_in=design.T_e_in,
        Z_in=design.Z_in,
        I_0=design.I_0,
        seed_fraction=design.seed_fraction,
        B=float(design.B_T),
        inlet_A=float(config.area_scale_m2),
        working_fluid=fluid,
    )
    H_values = []
    L_values = []
    rhs_H_values = []
    rhs_L_values = []
    h_scales = []
    l_scales = []
    mach = []
    for n_val, te_val, area_val, sigma_val in zip(n_p, T_e, A, sigma, strict=True):
        closure, _ = dynamic_system_terms(
            ops=ops,
            n_p=float(n_val),
            T_e=float(te_val),
            A=float(area_val),
            sigma=float(sigma_val),
            dot_N=float(inlet["dot_N"]),
            I_0=float(design.I_0),
            seed_fraction=design.seed_fraction,
            B=float(design.B_T),
            working_fluid=fluid,
        )
        balances = _freidberg_balance_terms(
            ops=ops,
            closure=closure,
            A=float(area_val),
            B_T=float(design.B_T),
            area_scale_m2=float(config.area_scale_m2),
            heavy_particle_mass_kg=float(fluid.heavy_particle_mass_kg),
        )
        h_scale, l_scale = _freidberg_row_scale_exprs(
            ops=ops,
            balances=balances,
            length_m=float(config.length_m),
        )
        H_values.append(float(balances["H_p"]))
        L_values.append(float(balances["L_p"]))
        rhs_H_values.append(float(balances["rhs_H"]))
        rhs_L_values.append(float(balances["rhs_L"]))
        h_scales.append(float(h_scale))
        l_scales.append(float(l_scale))
        mach.append(float(closure["mach"]))

    dH_dx = np.gradient(np.asarray(H_values, dtype=float), x, edge_order=1)
    dL_dx = np.gradient(np.asarray(L_values, dtype=float), x, edge_order=1)
    h_res = dH_dx - np.asarray(rhs_H_values, dtype=float)
    l_res = dL_dx - np.asarray(rhs_L_values, dtype=float)
    h_scale_arr = np.asarray(h_scales, dtype=float)
    l_scale_arr = np.asarray(l_scales, dtype=float)
    return {
        "finite_difference_note": (
            "Freidberg H/L strong residual diagnostic only; solver residual is weak CG form."
        ),
        "max_abs_H_residual": float(np.nanmax(np.abs(h_res))),
        "max_abs_L_residual": float(np.nanmax(np.abs(l_res))),
        "max_abs_scaled_H_residual": float(np.nanmax(np.abs(h_res / np.maximum(h_scale_arr, 1e-300)))),
        "max_abs_scaled_L_residual": float(np.nanmax(np.abs(l_res / np.maximum(l_scale_arr, 1e-300)))),
        "H_scale_min": float(np.nanmin(h_scale_arr)),
        "H_scale_max": float(np.nanmax(h_scale_arr)),
        "L_scale_min": float(np.nanmin(l_scale_arr)),
        "L_scale_max": float(np.nanmax(l_scale_arr)),
        "H_min": float(np.nanmin(H_values)),
        "H_max": float(np.nanmax(H_values)),
        "L_min": float(np.nanmin(L_values)),
        "L_max": float(np.nanmax(L_values)),
        "mach_min": float(np.nanmin(mach)),
        "mach_max": float(np.nanmax(mach)),
    }


def _inlet_enthalpy_flux_density_expr(*, ops, inlet: dict[str, Any], fluid: Any, length_m: float) -> Any:
    n_p = ops.max(inlet["n_p"], 1.0)
    n_e = ops.max(inlet["n_e"], 0.0)
    T_p = ops.max(inlet["T_p"], 1.0)
    T_e = ops.max(inlet["T_e"], 1.0)
    v_in = ops.max(inlet["v_in"], 1e-30)
    A_in = float(max(float(inlet["A_in"]), 1e-30))
    thermal_density = 2.5 * 1.380649e-23 * (n_p * T_p + n_e * T_e)
    kinetic_density = 0.5 * float(fluid.heavy_particle_mass_kg) * n_p * v_in * v_in
    return A_in * v_in * (thermal_density + kinetic_density) / max(float(length_m), 1e-30)


def _enthalpy_extraction_objective_expr(
    *,
    fd,
    ops,
    A: Any,
    closure: dict[str, Any],
    inlet: dict[str, Any],
    fluid: Any,
    length_m: float,
    measure: Any,
) -> Any:
    power_density = -A * closure["J_x"] * closure["E_x"]
    inlet_flux_density = _inlet_enthalpy_flux_density_expr(
        ops=ops,
        inlet=inlet,
        fluid=fluid,
        length_m=float(length_m),
    )
    mhd_output_power_W = fd.assemble(power_density * measure)
    inlet_enthalpy_flux_W = fd.assemble(inlet_flux_density * measure)
    return 100.0 * mhd_output_power_W / (inlet_enthalpy_flux_W + 1e-30)


def _velikhov_penalty_expr(
    *,
    fd,
    ops,
    G: Any,
    config: CaseConfig,
    measure: Any,
) -> Any:
    settings = velikhov_settings(config)
    if not bool(settings["active"]):
        return 0.0
    floor = float(settings["floor"])
    scale = max(float(settings["scale"]), 1e-300)
    weight = float(settings["weight"])
    shortfall = ops.max(floor - G, 0.0)
    length = max(float(config.length_m), 1e-300)
    mean_scaled_sq = fd.assemble(((shortfall / scale) ** 2) * measure) / length
    return weight * mean_scaled_sq


def _thermal_window_penalty_expr(
    *,
    fd,
    ops,
    T_e: Any,
    T_p: Any,
    inlet_T_p: Any,
    config: CaseConfig,
    measure: Any,
) -> Any:
    settings = thermal_window_settings(config)
    if not bool(settings["active"]):
        return 0.0

    length = max(float(config.length_m), 1e-300)
    tp_scale = max(float(settings["tp_scale_K"]), 1e-300)
    ratio_scale = max(float(settings["ratio_scale"]), 1e-300)
    tp_in_weight = float(settings["tp_in_weight"])
    tp_path_weight = float(settings["tp_path_weight"])
    ratio_weight = float(settings["ratio_weight"])

    penalty = 0.0
    tp_in_max = settings["tp_in_max_K"]
    if tp_in_max is not None:
        excess = ops.max(inlet_T_p - float(tp_in_max), 0.0) / tp_scale
        penalty = penalty + tp_in_weight * fd.assemble((excess**2) * measure) / length

    tp_floor = settings["tp_floor_K"]
    if tp_floor is not None:
        shortfall = ops.max(float(tp_floor) - T_p, 0.0) / tp_scale
        penalty = penalty + tp_path_weight * fd.assemble((shortfall**2) * measure) / length

    tp_path_max = settings["tp_path_max_K"]
    if tp_path_max is not None:
        excess = ops.max(T_p - float(tp_path_max), 0.0) / tp_scale
        penalty = penalty + tp_path_weight * fd.assemble((excess**2) * measure) / length

    ratio = T_e / ops.max(T_p, 1.0)
    ratio_violation = 0.0
    ratio_min = settings["te_over_tp_min"]
    if ratio_min is not None:
        ratio_violation = ratio_violation + ops.max(float(ratio_min) - ratio, 0.0) / ratio_scale
    ratio_max = settings["te_over_tp_max"]
    if ratio_max is not None:
        ratio_violation = ratio_violation + ops.max(ratio - float(ratio_max), 0.0) / ratio_scale
    if ratio_min is not None or ratio_max is not None:
        penalty = penalty + ratio_weight * fd.assemble((ratio_violation**2) * measure) / length

    return penalty


def solve_forward(
    *,
    design: DesignVector,
    config: CaseConfig,
    annotate_objective: bool = False,
    initial_profile: dict[str, Any] | None = None,
) -> ForwardResult:
    """Solve the quasi-1D implicit reduced forward problem with Firedrake."""

    fd = _require_firedrake()
    ops = _ops_for_firedrake(fd)
    fluid = working_fluid_for_config(config)
    equation_form = _equation_form(config)
    mesh = fd.IntervalMesh(int(config.n_intervals), float(config.length_m))
    measure = fd.dx(domain=mesh)
    V = fd.FunctionSpace(mesh, "CG", 1)
    W = V * V
    state = fd.Function(W, name="delta_log_state")
    delta_log_n, delta_log_Te = fd.split(state)
    test_n, test_Te = fd.TestFunctions(W)

    controls, control_space = _design_coefficients(fd, mesh, design, as_functions=bool(annotate_objective))
    n_p_in = ops.exp(controls["log_n_p_in"])
    T_e_in = controls["T_e_in"]
    seed_fraction = ops.exp(controls["log_seed_fraction"])
    inlet = inlet_design_generic(
        ops=ops,
        n_p_in=n_p_in,
        T_e_in=T_e_in,
        Z_in=controls["Z_in"],
        I_0=controls["I_0"],
        seed_fraction=seed_fraction,
        B=controls["B_T"],
        inlet_A=float(config.area_scale_m2),
        working_fluid=fluid,
    )

    x_norm = np.linspace(0.0, 1.0, int(config.n_intervals) + 1, dtype=float)
    basis, slopes = design.area_control.basis_matrices(x_norm)
    b1 = _basis_function(fd, V, basis[:, 0], name="area_basis_a1")
    b2 = _basis_function(fd, V, basis[:, 1], name="area_basis_a2")
    b3 = _basis_function(fd, V, basis[:, 2], name="area_basis_a3")
    s1 = _basis_function(fd, V, slopes[:, 0], name="area_slope_a1")
    s2 = _basis_function(fd, V, slopes[:, 1], name="area_slope_a2")
    s3 = _basis_function(fd, V, slopes[:, 2], name="area_slope_a3")

    log_Te_in = ops.log(ops.max(controls["T_e_in"], 1.0))
    log_n = controls["log_n_p_in"] + delta_log_n
    log_Te = log_Te_in + delta_log_Te
    logA = controls["a1"] * b1 + controls["a2"] * b2 + controls["a3"] * b3
    sigma = (controls["a1"] * s1 + controls["a2"] * s2 + controls["a3"] * s3) / float(config.length_m)
    A = float(config.area_scale_m2) * ops.exp(logA)
    n_p = ops.exp(log_n)
    T_e = ops.exp(log_Te)
    dn_dx = n_p * delta_log_n.dx(0)
    dTe_dx = T_e * delta_log_Te.dx(0)

    closure, terms = dynamic_system_terms(
        ops=ops,
        n_p=n_p,
        T_e=T_e,
        A=A,
        sigma=sigma,
        dot_N=inlet["dot_N"],
        I_0=controls["I_0"],
        seed_fraction=seed_fraction,
        B=controls["B_T"],
        working_fluid=fluid,
    )
    residual_scaling = _residual_scaling_mode(config)
    solver_row1_scale_values = None
    solver_row2_scale_values = None
    solver_row_names = ("momentum", "energy")
    if equation_form == "freidberg_hl":
        balances = _freidberg_balance_terms(
            ops=ops,
            closure=closure,
            A=A,
            B_T=controls["B_T"],
            area_scale_m2=float(config.area_scale_m2),
            heavy_particle_mass_kg=float(fluid.heavy_particle_mass_kg),
        )
        row1 = balances["H_p"].dx(0) - balances["rhs_H"]
        row2 = balances["L_p"].dx(0) - balances["rhs_L"]
        solver_row_names = ("freidberg_H", "freidberg_L")
        if residual_scaling == "inlet":
            row1_scale_value, row2_scale_value = _inlet_freidberg_scale_values(design=design, config=config)
            row1_scale = fd.Constant(row1_scale_value)
            row2_scale = fd.Constant(row2_scale_value)
            solver_row1_scale_values = np.full(int(config.n_intervals) + 1, row1_scale_value, dtype=float)
            solver_row2_scale_values = np.full(int(config.n_intervals) + 1, row2_scale_value, dtype=float)
            residual = ((row1 / row1_scale) * test_n + (row2 / row2_scale) * test_Te) * measure
        elif residual_scaling == "characteristic":
            row1_scale_values, row2_scale_values = _reference_freidberg_scale_arrays(design=design, config=config)
            row1_scale = _basis_function(fd, V, row1_scale_values, name="freidberg_H_residual_scale")
            row2_scale = _basis_function(fd, V, row2_scale_values, name="freidberg_L_residual_scale")
            solver_row1_scale_values = row1_scale_values
            solver_row2_scale_values = row2_scale_values
            residual = ((row1 / row1_scale) * test_n + (row2 / row2_scale) * test_Te) * measure
        else:
            residual = (row1 * test_n + row2 * test_Te) * measure
    else:
        momentum = terms["M11"] * dn_dx + terms["M12"] * dTe_dx - terms["rhs_m"]
        energy = terms["E11"] * dn_dx + terms["E12"] * dTe_dx - terms["rhs_e"]
        if residual_scaling == "inlet":
            momentum_scale_value, energy_scale_value = _inlet_residual_scale_values(design=design, config=config)
            momentum_scale = fd.Constant(momentum_scale_value)
            energy_scale = fd.Constant(energy_scale_value)
            solver_row1_scale_values = np.full(int(config.n_intervals) + 1, momentum_scale_value, dtype=float)
            solver_row2_scale_values = np.full(int(config.n_intervals) + 1, energy_scale_value, dtype=float)
            residual = ((momentum / momentum_scale) * test_n + (energy / energy_scale) * test_Te) * measure
        elif residual_scaling == "characteristic":
            momentum_scale_values, energy_scale_values = _reference_residual_scale_arrays(design=design, config=config)
            momentum_scale = _basis_function(fd, V, momentum_scale_values, name="momentum_residual_scale")
            energy_scale = _basis_function(fd, V, energy_scale_values, name="energy_residual_scale")
            solver_row1_scale_values = momentum_scale_values
            solver_row2_scale_values = energy_scale_values
            residual = ((momentum / momentum_scale) * test_n + (energy / energy_scale) * test_Te) * measure
        else:
            residual = (momentum * test_n + energy * test_Te) * measure

    state.sub(0).interpolate(fd.Constant(0.0))
    state.sub(1).interpolate(fd.Constant(0.0))
    initial_guess = "zero_delta"
    if initial_profile is not None:
        try:
            delta_log_n_values, delta_log_Te_values = _initial_delta_arrays(
                initial_profile=initial_profile,
                design=design,
                target_x_norm=x_norm,
            )
            initial_state = fd.Function(W, name="initial_profile_delta")
            initial_state.subfunctions[0].dat.data[:] = delta_log_n_values
            initial_state.subfunctions[1].dat.data[:] = delta_log_Te_values
            try:
                state.assign(initial_state, annotate=annotate_objective)
            except TypeError:
                state.assign(initial_state)
            initial_guess = "profile_interpolated_delta"
        except Exception as exc:
            return ForwardResult(
                ok=False,
                design=design,
                config=config,
                profile=None,
                metrics=None,
                diagnostics={
                    "solver": "firedrake_snes_newtonls",
                    "equation_form": equation_form,
                    "residual_scaling": residual_scaling,
                    "initial_guess": "invalid_profile",
                    "working_fluid": fluid.to_dict(),
                },
                error=f"{type(exc).__name__}: {exc}",
                fd_controls=controls,
                fd_control_space=control_space,
            )
    bcs = [
        fd.DirichletBC(W.sub(0), fd.Constant(0.0), 1),
        fd.DirichletBC(W.sub(1), fd.Constant(0.0), 1),
    ]
    problem = fd.NonlinearVariationalProblem(residual, state, bcs=bcs)
    solver_parameters = {
        "snes_type": str(config.metadata.get("snes_type", "newtonls")),
        "snes_rtol": 1e-8,
        "snes_atol": 1e-9,
        "snes_max_it": int(config.metadata.get("snes_max_it", 50)),
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    if "snes_dtol" in config.metadata:
        solver_parameters["snes_dtol"] = float(config.metadata["snes_dtol"])
    if "snes_linesearch_type" in config.metadata:
        solver_parameters["snes_linesearch_type"] = str(config.metadata["snes_linesearch_type"])
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)

    try:
        solver.solve()
    except Exception as exc:
        failure_profile = None
        failure_metrics = None
        failure_extra: dict[str, Any] = {}
        try:
            delta_log_n_fn, delta_log_Te_fn = state.subfunctions
            log_Te_in_float = float(np.log(max(float(design.T_e_in), 1.0)))
            failure_x = x_norm * float(config.length_m)
            failure_area = design.area_control.evaluate_profile(
                length=float(config.length_m),
                n_intervals=int(config.n_intervals),
                area_scale=float(config.area_scale_m2),
            )
            failure_profile = {
                "x": failure_x,
                "x_norm": x_norm,
                "n_p": np.exp(float(design.log_n_p_in) + np.asarray(delta_log_n_fn.dat.data_ro, dtype=float).copy()),
                "T_e": np.exp(log_Te_in_float + np.asarray(delta_log_Te_fn.dat.data_ro, dtype=float).copy()),
                "A": np.asarray(failure_area["A"], dtype=float),
                "sigma_logA": np.asarray(failure_area["sigma_logA"], dtype=float),
            }
            if (
                np.all(np.isfinite(failure_profile["n_p"]))
                and np.all(np.isfinite(failure_profile["T_e"]))
                and np.all(failure_profile["n_p"] > 0.0)
                and np.all(failure_profile["T_e"] > 0.0)
            ):
                failure_metrics = evaluate_profile_metrics(profile=failure_profile, design=design, config=config)
                failure_extra = {
                    "failed_iterate_metrics": failure_metrics.to_dict(),
                    "failed_iterate_strong_residual_diagnostics": _strong_residual_diagnostics(
                        profile=failure_profile,
                        design=design,
                        config=config,
                    ),
                    "failed_iterate_freidberg_strong_residual_diagnostics": _freidberg_strong_residual_diagnostics(
                        profile=failure_profile,
                        design=design,
                        config=config,
                    ),
                }
            else:
                failure_extra = {"failed_iterate_note": "last Newton iterate contained non-finite or non-positive n_p/T_e"}
        except Exception as diag_exc:
            failure_extra = {"failed_iterate_diagnostic_error": f"{type(diag_exc).__name__}: {diag_exc}"}
        return ForwardResult(
            ok=False,
            design=design,
            config=config,
            profile=failure_profile,
            metrics=failure_metrics,
            diagnostics={
                "solver": "firedrake_snes_newtonls",
                "equation_form": equation_form,
                "residual_scaling": residual_scaling,
                "initial_guess": initial_guess,
                "working_fluid": fluid.to_dict(),
                **failure_extra,
            },
            error=f"{type(exc).__name__}: {exc}",
            fd_controls=controls,
            fd_control_space=control_space,
        )

    delta_log_n_fn, delta_log_Te_fn = state.subfunctions
    log_Te_in_float = float(np.log(max(float(design.T_e_in), 1.0)))
    x = x_norm * float(config.length_m)
    area = design.area_control.evaluate_profile(
        length=float(config.length_m),
        n_intervals=int(config.n_intervals),
        area_scale=float(config.area_scale_m2),
    )
    profile = {
        "x": x,
        "x_norm": x_norm,
        "n_p": np.exp(float(design.log_n_p_in) + np.asarray(delta_log_n_fn.dat.data_ro, dtype=float).copy()),
        "T_e": np.exp(log_Te_in_float + np.asarray(delta_log_Te_fn.dat.data_ro, dtype=float).copy()),
        "A": np.asarray(area["A"], dtype=float),
        "sigma_logA": np.asarray(area["sigma_logA"], dtype=float),
    }
    metrics = evaluate_profile_metrics(profile=profile, design=design, config=config)
    residual_diagnostics = _strong_residual_diagnostics(profile=profile, design=design, config=config)
    freidberg_residual_diagnostics = _freidberg_strong_residual_diagnostics(profile=profile, design=design, config=config)
    fd_objective = None
    objective_kind = None
    if annotate_objective:
        if config.objective_profile != OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION:
            raise ValueError(f"unsupported taped objective profile: {config.objective_profile!r}")
        closure, _ = dynamic_system_terms(
            ops=ops,
            n_p=n_p,
            T_e=T_e,
            A=A,
            sigma=sigma,
            dot_N=inlet["dot_N"],
            I_0=controls["I_0"],
            seed_fraction=seed_fraction,
            B=controls["B_T"],
            working_fluid=fluid,
        )
        fd_objective = _enthalpy_extraction_objective_expr(
            fd=fd,
            ops=ops,
            A=A,
            closure=closure,
            inlet=inlet,
            fluid=fluid,
            length_m=float(config.length_m),
            measure=measure,
        )
        fd_objective = fd_objective - _velikhov_penalty_expr(
            fd=fd,
            ops=ops,
            G=closure["G"],
            config=config,
            measure=measure,
        )
        fd_objective = fd_objective - _thermal_window_penalty_expr(
            fd=fd,
            ops=ops,
            T_e=T_e,
            T_p=closure["T_p"],
            inlet_T_p=inlet["T_p"],
            config=config,
            measure=measure,
        )
        objective_kind = OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION
    velikhov_config = velikhov_settings(config)
    thermal_config = thermal_window_settings(config)

    return ForwardResult(
        ok=True,
        design=design,
        config=config,
        profile=profile,
        metrics=metrics,
        diagnostics={
            "solver": "firedrake_snes_newtonls",
            "equation_form": equation_form,
            "initial_guess": initial_guess,
            "working_fluid": fluid.to_dict(),
            "taped_objective": objective_kind,
            "velikhov_mode": str(velikhov_config["mode"]),
            "velikhov_penalty_formula": (
                "score = raw_enthalpy_extraction_percent - "
                "weight * mean(max(floor - G, 0) / scale)^2"
            ),
            "thermal_window_mode": str(thermal_config["mode"]),
            "thermal_window_penalty_formula": (
                "score -= Tp_in_weight*max(Tp_in - Tp_in_max, 0)^2/Tp_scale^2 "
                "+ Tp_path_weight*mean(path Tp shortfalls/excesses)^2 "
                "+ ratio_weight*mean(Te/Tp band violations)^2"
            ),
            "residual_scaling": residual_scaling,
            "residual_scaling_formula": (
                "inlet: one inlet-row scale for each equation; "
                "characteristic: nodal reference scales from current area and inlet n/T; "
                "dimensional: no row scaling"
            ),
            "strong_residual_diagnostics": residual_diagnostics,
            "freidberg_strong_residual_diagnostics": freidberg_residual_diagnostics,
            "solver_residual_scale": (
                None
                if solver_row1_scale_values is None or solver_row2_scale_values is None
                else {
                    "row1": solver_row_names[0],
                    "row2": solver_row_names[1],
                    "row1_scale_min": float(np.nanmin(solver_row1_scale_values)),
                    "row1_scale_max": float(np.nanmax(solver_row1_scale_values)),
                    "row2_scale_min": float(np.nanmin(solver_row2_scale_values)),
                    "row2_scale_max": float(np.nanmax(solver_row2_scale_values)),
                }
            ),
        },
        fd_objective=fd_objective,
        fd_controls=controls,
        fd_control_space=control_space,
    )
