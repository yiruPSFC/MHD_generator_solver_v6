from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .design import CaseConfig, DesignVector
from .legacy_physics import dynamic_system_terms, inlet_design_generic, ops_for_numeric
from .reference_profile import (
    ReferenceProfileResult,
    _area_at_x,
    _area_profile_arrays,
    _freidberg_terms_for_log_state,
    _safe_exp,
    build_freidberg_reference_profile,
)
from .transport import working_fluid_for_config


_EPS = 1e-300
K_B = 1.380649e-23
_ELL_SECOND_AT_SONIC = -3.0 / 32.0


def _json_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _compatibility_values(row: dict[str, Any]) -> dict[str, float | None]:
    L_p = _json_float(row.get("L_p"))
    rhs_L = _json_float(row.get("rhs_L"))
    sigma = _json_float(row.get("sigma_logA"))
    if L_p is None or rhs_L is None or sigma is None or abs(L_p) <= _EPS:
        return {
            "required_sigma_logA": None,
            "compatibility_residual": None,
            "scaled_compatibility_residual": None,
            "sigma_gap_required_minus_current": None,
        }
    required = rhs_L / L_p
    residual = rhs_L - sigma * L_p
    scale = max(1.0, abs(rhs_L), abs(sigma * L_p))
    return {
        "required_sigma_logA": float(required),
        "compatibility_residual": float(residual),
        "scaled_compatibility_residual": float(residual / scale),
        "sigma_gap_required_minus_current": float(required - sigma),
    }


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_clean(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if np.isfinite(out) else None
    return value


def _freidberg_terms_for_explicit_area(
    *,
    ops,
    fluid,
    design: DesignVector,
    config: CaseConfig,
    inlet: dict[str, Any],
    x: float,
    z: np.ndarray,
    A: float,
    sigma_logA: float,
) -> dict[str, Any]:
    """Evaluate the Freidberg H/L quantities with explicit local area data."""

    log_n = float(design.log_n_p_in) + float(z[0])
    log_Te = float(np.log(max(float(design.T_e_in), 1.0))) + float(z[1])
    n_p = _safe_exp(log_n)
    T_e = _safe_exp(log_Te)
    closure, terms = dynamic_system_terms(
        ops=ops,
        n_p=n_p,
        T_e=T_e,
        A=float(A),
        sigma=float(sigma_logA),
        dot_N=float(inlet["dot_N"]),
        I_0=float(design.I_0),
        seed_fraction=design.seed_fraction,
        B=float(design.B_T),
        working_fluid=fluid,
    )
    J2 = float(closure["J_x"]) * float(closure["J_x"]) + float(closure["J_y"]) * float(closure["J_y"])
    v_p = float(closure["v_p"])
    T_p = float(closure["T_p"])
    T_p_safe = max(T_p, 1.0)
    mach = float(closure["mach"])
    M2 = mach * mach
    A0 = max(float(config.area_scale_m2), _EPS)
    H_p = (float(A) * n_p * v_p / A0) * (
        2.5 * K_B * T_p_safe + 0.5 * float(fluid.heavy_particle_mass_kg) * v_p * v_p
    )
    L_p = mach * (float(A) / A0) / max((M2 + 3.0) * (M2 + 3.0), _EPS)
    rhs_H = (float(A) / A0) * (v_p * float(closure["J_y"]) * float(design.B_T) + float(closure["eta"]) * J2)
    p_p = max(n_p * K_B * T_p_safe, _EPS)
    rhs_L = (
        -(12.0 / 5.0)
        * L_p
        / max((M2 + 3.0) * p_p * v_p, _EPS)
        * (v_p * float(closure["J_y"]) * float(design.B_T) - ((5.0 * M2 + 3.0) / 12.0) * float(closure["eta"]) * J2)
    )
    return {
        "x": float(x),
        "z": np.asarray(z, dtype=float).copy(),
        "n_p": float(n_p),
        "T_e": float(T_e),
        "A": float(A),
        "sigma_logA": float(sigma_logA),
        "closure": closure,
        "primitive_terms": terms,
        "H_p": float(H_p),
        "L_p": float(L_p),
        "rhs_H": float(rhs_H),
        "rhs_L": float(rhs_L),
        "mach": float(mach),
        "T_p": float(T_p),
        "det": float(terms["det"]),
        "J2": float(J2),
    }


def _freidberg_terms_with_area_slope(
    *,
    ops,
    fluid,
    design: DesignVector,
    config: CaseConfig,
    inlet: dict[str, Any],
    x: float,
    z: np.ndarray,
    area_profile: dict[str, Any] | None = None,
    sigma_logA: float | None = None,
) -> dict[str, Any]:
    A, sigma = _area_at_x(design=design, config=config, x=float(x), area_profile=area_profile)
    return _freidberg_terms_for_explicit_area(
        ops=ops,
        fluid=fluid,
        design=design,
        config=config,
        inlet=inlet,
        x=float(x),
        z=np.asarray(z, dtype=float),
        A=float(A),
        sigma_logA=float(sigma if sigma_logA is None else sigma_logA),
    )


def _z_from_state(*, design: DesignVector, n_p: float, T_e: float) -> np.ndarray:
    return np.array(
        [
            float(np.log(max(float(n_p), _EPS))) - float(design.log_n_p_in),
            float(np.log(max(float(T_e), 1.0))) - float(np.log(max(float(design.T_e_in), 1.0))),
        ],
        dtype=float,
    )


def _area_arrays_for_match(
    *,
    design: DesignVector,
    config: CaseConfig,
    area_profile: dict[str, Any] | None,
    n_intervals: int | None,
) -> dict[str, np.ndarray]:
    if area_profile is not None:
        return _area_profile_arrays(area_profile, config=config)
    n = max(400, 8 * int(config.n_intervals if n_intervals is None else n_intervals))
    return design.area_control.evaluate_profile(
        length=float(config.length_m),
        n_intervals=n,
        area_scale=float(config.area_scale_m2),
    )


def _interp_area_field(area: dict[str, Any], field: str, x: float) -> float:
    return float(np.interp(float(x), np.asarray(area["x"], dtype=float), np.asarray(area[field], dtype=float)))


def audit_sonic_compatibility(
    *,
    profile: dict[str, Any],
    design: DesignVector,
    config: CaseConfig,
) -> dict[str, Any]:
    """Evaluate the smooth Freidberg H/L sonic compatibility proxy on a profile."""

    ops = ops_for_numeric()
    fluid = working_fluid_for_config(config)
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
    area = _area_profile_arrays(profile, config=config)
    n_p = np.asarray(profile["n_p"], dtype=float).reshape(-1)
    T_e = np.asarray(profile["T_e"], dtype=float).reshape(-1)
    if not (area["x"].size == n_p.size == T_e.size):
        raise ValueError("profile x/A, n_p, and T_e arrays must have matching sizes.")
    log_Te_in = float(np.log(max(float(design.T_e_in), 1.0)))
    rows: list[dict[str, Any]] = []
    for idx, (x_value, n_value, te_value) in enumerate(zip(area["x"], n_p, T_e, strict=True)):
        if not (np.isfinite(n_value) and np.isfinite(te_value) and n_value > 0.0 and te_value > 0.0):
            rows.append({"index": int(idx), "x": _json_float(x_value), "ok": False, "error": "non-finite state"})
            continue
        terms = _freidberg_terms_for_log_state(
            ops=ops,
            fluid=fluid,
            design=design,
            config=config,
            inlet=inlet,
            x=float(x_value),
            z=np.array(
                [
                    np.log(float(n_value)) - float(design.log_n_p_in),
                    np.log(float(te_value)) - log_Te_in,
                ],
                dtype=float,
            ),
            area_profile=area,
        )
        row = {
            "index": int(idx),
            "x": _json_float(x_value),
            "x_fraction": _json_float(float(x_value) / max(float(config.length_m), _EPS)),
            "ok": True,
            "mach": _json_float(terms["mach"]),
            "T_p": _json_float(terms["T_p"]),
            "A": _json_float(terms["A"]),
            "sigma_logA": _json_float(terms["sigma_logA"]),
            "L_p": _json_float(terms["L_p"]),
            "rhs_L": _json_float(terms["rhs_L"]),
        }
        row.update(_compatibility_values(row))
        rows.append(row)

    finite_mach = [
        (idx, float(row["mach"]))
        for idx, row in enumerate(rows)
        if row.get("mach") is not None and np.isfinite(float(row["mach"]))
    ]
    closest = None
    if finite_mach:
        row_index, mach_value = min(finite_mach, key=lambda item: abs(item[1] - 1.0))
        closest = dict(rows[row_index])
        closest["abs_mach_minus_one"] = float(abs(mach_value - 1.0))
    residuals = [
        abs(float(row["scaled_compatibility_residual"]))
        for row in rows
        if row.get("scaled_compatibility_residual") is not None
    ]
    mach_values = np.asarray([item[1] for item in finite_mach], dtype=float) if finite_mach else np.array([])
    summary = {
        "ok": bool(closest is not None),
        "row_count": int(len(rows)),
        "finite_mach_count": int(len(finite_mach)),
        "mach_min": None if mach_values.size == 0 else float(np.nanmin(mach_values)),
        "mach_max": None if mach_values.size == 0 else float(np.nanmax(mach_values)),
        "max_abs_scaled_compatibility_residual": None if not residuals else float(max(residuals)),
        "closest_to_sonic": closest,
        "compatibility_formula": "rhs_L - sigma_logA * L_p = 0",
    }
    return {
        "audit": "sonic_freidberg_hl_compatibility",
        "schema_version": 1,
        "summary": summary,
        "rows": rows,
    }


def _solve_sonic_match_from_anchor(
    *,
    ops,
    fluid,
    design: DesignVector,
    config: CaseConfig,
    inlet: dict[str, Any],
    area_profile: dict[str, Any] | None,
    anchor_row: dict[str, Any],
    anchor_index: int,
    max_log_step: float,
    h_min: float,
    h_upper: float,
    x_star_guess: float | None = None,
    sigma_guess_override: float | None = None,
) -> dict[str, Any]:
    x_prev = float(anchor_row["x"])
    z_prev = _z_from_state(design=design, n_p=float(anchor_row["n_p"]), T_e=float(anchor_row["T_e"]))
    prev = _freidberg_terms_with_area_slope(
        ops=ops,
        fluid=fluid,
        design=design,
        config=config,
        inlet=inlet,
        x=x_prev,
        z=z_prev,
        area_profile=area_profile,
    )
    row_comp = _compatibility_values({**anchor_row, "L_p": prev["L_p"], "rhs_L": prev["rhs_L"]})
    sigma_guess = sigma_guess_override if sigma_guess_override is not None else row_comp.get("required_sigma_logA")
    if sigma_guess is None or not np.isfinite(float(sigma_guess)):
        sigma_guess = float(prev["sigma_logA"])
    h_guess = abs(float(anchor_row.get("h", 0.0) or 0.0))
    if x_star_guess is not None and np.isfinite(float(x_star_guess)) and float(x_star_guess) > x_prev:
        h_guess = float(x_star_guess) - x_prev
    if not np.isfinite(h_guess) or h_guess <= 0.0:
        h_guess = max(float(config.length_m) * 1.0e-6, h_min)
    h_upper = min(max(float(h_upper), 2.0 * h_min), max(float(config.length_m) - x_prev, 2.0 * h_min))
    h_guess = min(max(h_guess, 2.0 * h_min), 0.75 * h_upper)
    lower = np.array(
        [
            z_prev[0] - float(max_log_step),
            z_prev[1] - float(max_log_step),
            float(np.log(max(h_min, _EPS))),
            -1.0e7,
        ],
        dtype=float,
    )
    upper = np.array(
        [
            z_prev[0] + float(max_log_step),
            z_prev[1] + float(max_log_step),
            float(np.log(max(h_upper, 2.0 * h_min))),
            1.0e7,
        ],
        dtype=float,
    )
    guess = np.array([z_prev[0], z_prev[1], float(np.log(h_guess)), float(sigma_guess)], dtype=float)
    guess = np.minimum(np.maximum(guess, lower), upper)

    inv_l = 1.0 / max(float(config.length_m), _EPS)

    def residual(candidate: np.ndarray) -> np.ndarray:
        try:
            z_star = np.asarray(candidate[:2], dtype=float)
            h = float(np.exp(float(candidate[2])))
            sigma_star = float(candidate[3])
            x_star = x_prev + h
            if not (x_prev < x_star < float(config.length_m)):
                return np.full(4, 1e30, dtype=float)
            star = _freidberg_terms_with_area_slope(
                ops=ops,
                fluid=fluid,
                design=design,
                config=config,
                inlet=inlet,
                x=x_star,
                z=z_star,
                area_profile=area_profile,
                sigma_logA=sigma_star,
            )
            H_res = (star["H_p"] - prev["H_p"]) / max(h, _EPS) - star["rhs_H"]
            L_res = (star["L_p"] - prev["L_p"]) / max(h, _EPS) - star["rhs_L"]
            C_res = star["rhs_L"] - sigma_star * star["L_p"]
            H_scale = max(1.0, abs(star["H_p"] * inv_l), abs(prev["H_p"] * inv_l), abs(star["rhs_H"]))
            L_scale = max(inv_l, abs(star["L_p"] * inv_l), abs(prev["L_p"] * inv_l), abs(star["rhs_L"]))
            C_scale = max(inv_l, abs(star["rhs_L"]), abs(sigma_star * star["L_p"]))
            values = np.array(
                [
                    H_res / H_scale,
                    L_res / L_scale,
                    (star["mach"] - 1.0) / 1.0e-7,
                    C_res / C_scale,
                ],
                dtype=float,
            )
            if not np.all(np.isfinite(values)):
                return np.full(4, 1e30, dtype=float)
            return values
        except Exception:
            return np.full(4, 1e30, dtype=float)

    candidate_guesses = [guess]
    for factor in (0.25, 0.5, 2.0, 4.0):
        h_alt = min(max(h_guess * factor, h_min), h_upper)
        candidate_guesses.append(
            np.minimum(
                np.maximum(
                    np.array([z_prev[0], z_prev[1], float(np.log(h_alt)), float(sigma_guess)], dtype=float),
                    lower,
                ),
                upper,
            )
        )
    solutions = [
        least_squares(
            residual,
            candidate,
            bounds=(lower, upper),
            x_scale=np.array([1.0, 1.0, 1.0, max(abs(float(sigma_guess)), 100.0)], dtype=float),
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            max_nfev=300,
        )
        for candidate in candidate_guesses
    ]
    sol = min(solutions, key=lambda item: float(np.max(np.abs(residual(np.asarray(item.x, dtype=float))))))
    values = residual(np.asarray(sol.x, dtype=float))
    z_star = np.asarray(sol.x[:2], dtype=float)
    h = float(np.exp(float(sol.x[2])))
    x_star = x_prev + h
    sigma_star = float(sol.x[3])
    star = _freidberg_terms_with_area_slope(
        ops=ops,
        fluid=fluid,
        design=design,
        config=config,
        inlet=inlet,
        x=x_star,
        z=z_star,
        area_profile=area_profile,
        sigma_logA=sigma_star,
    )
    A_original, sigma_original = _area_at_x(
        design=design,
        config=config,
        x=x_star,
        area_profile=area_profile,
    )
    H_res_raw = (star["H_p"] - prev["H_p"]) / max(h, _EPS) - star["rhs_H"]
    L_res_raw = (star["L_p"] - prev["L_p"]) / max(h, _EPS) - star["rhs_L"]
    C_res_raw = star["rhs_L"] - sigma_star * star["L_p"]
    residual_inf = float(np.max(np.abs(values))) if values.size else float("inf")
    return _json_clean(
        {
            "ok": bool(sol.success and np.isfinite(residual_inf) and residual_inf <= 1.0e-3),
            "anchor_index": int(anchor_index),
            "least_squares_success": bool(sol.success),
            "least_squares_status": int(sol.status),
            "least_squares_message": str(sol.message),
            "scaled_residual_inf": residual_inf,
            "scaled_residuals": values,
            "raw_residuals": {
                "H_balance": H_res_raw,
                "L_balance": L_res_raw,
                "mach_minus_one": star["mach"] - 1.0,
                "compatibility": C_res_raw,
            },
            "left_anchor": {
                "x_m": x_prev,
                "x_fraction": x_prev / max(float(config.length_m), _EPS),
                "h_to_sonic_m": h,
                "h_to_sonic_over_uniform_dx": h
                / max(float(config.length_m) / max(int(config.n_intervals), 1), _EPS),
                "mach": prev["mach"],
                "T_p_K": prev["T_p"],
                "n_p": prev["n_p"],
                "T_e": prev["T_e"],
                "delta_log_n": z_prev[0],
                "delta_log_Te": z_prev[1],
                "A_over_A0": prev["A"] / max(float(config.area_scale_m2), _EPS),
                "sigma_logA": prev["sigma_logA"],
            },
            "sonic_point": {
                "x_m": x_star,
                "x_fraction": x_star / max(float(config.length_m), _EPS),
                "mach": star["mach"],
                "T_p_K": star["T_p"],
                "n_p": star["n_p"],
                "T_e": star["T_e"],
                "delta_log_n": z_star[0],
                "delta_log_Te": z_star[1],
                "A_m2": star["A"],
                "A_over_A0": star["A"] / max(float(config.area_scale_m2), _EPS),
                "logA": np.log(max(star["A"] / max(float(config.area_scale_m2), _EPS), _EPS)),
                "sigma_required_1_per_m": sigma_star,
                "sigma_original_1_per_m": sigma_original,
                "sigma_gap_required_minus_original_1_per_m": sigma_star - sigma_original,
                "rhs_L": star["rhs_L"],
                "L_p": star["L_p"],
                "rhs_H": star["rhs_H"],
                "H_p": star["H_p"],
                "det": star["det"],
                "A_original_m2": A_original,
            },
        }
    )


def solve_local_sonic_match(
    *,
    design: DesignVector,
    config: CaseConfig,
    n_intervals: int | None = None,
    residual_tol: float = 1e-7,
    initial_substeps_per_interval: int = 10,
    max_log_step: float = 0.25,
    min_step_fraction: float = 1e-8,
    max_steps: int = 20000,
    area_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Find a local smooth sonic endpoint from the accepted subsonic branch."""

    reference = build_freidberg_reference_profile(
        design=design,
        config=config,
        n_intervals=n_intervals,
        residual_tol=float(residual_tol),
        initial_substeps_per_interval=int(initial_substeps_per_interval),
        max_log_step=float(max_log_step),
        min_step_fraction=float(min_step_fraction),
        max_steps=int(max_steps),
        area_profile=area_profile,
    )
    diagnostics = dict(reference.diagnostics or {})
    rows = [dict(row) for row in list(diagnostics.get("accepted_tail", []) or [])]
    usable_rows = [
        row
        for row in rows
        if row.get("x") is not None
        and row.get("n_p") is not None
        and row.get("T_e") is not None
        and float(row.get("x", float("nan"))) < float(config.length_m)
        and np.isfinite(float(row.get("n_p", float("nan"))))
        and np.isfinite(float(row.get("T_e", float("nan"))))
        and float(row.get("n_p", 0.0)) > 0.0
        and float(row.get("T_e", 0.0)) > 0.0
    ]
    if not usable_rows:
        return {
            "diagnostic": "local_sonic_match",
            "schema_version": 1,
            "ok": False,
            "error": "Freidberg marcher did not provide usable accepted_tail anchors.",
            "reference_ok": bool(reference.ok),
            "reference_error": reference.error,
            "reference_diagnostics": _json_clean(diagnostics),
        }

    ops = ops_for_numeric()
    fluid = working_fluid_for_config(config)
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
    h_base = float(config.length_m) / max(
        (int(config.n_intervals if n_intervals is None else n_intervals)) * int(initial_substeps_per_interval),
        1,
    )
    h_min = max(float(config.length_m) * float(min_step_fraction), float(config.length_m) * 1.0e-12)
    x_star_guess = None
    sorted_rows = sorted(usable_rows, key=lambda item: float(item.get("x", 0.0)))
    for prev_row, next_row in zip(sorted_rows[:-1], sorted_rows[1:], strict=False):
        try:
            m0 = float(prev_row["mach"])
            m1 = float(next_row["mach"])
            if m1 > m0 and m0 < 1.0 and m1 < 1.0:
                x0 = float(prev_row["x"])
                x1 = float(next_row["x"])
                x_star_guess = x1 + (1.0 - m1) * (x1 - x0) / max(m1 - m0, _EPS)
        except Exception:
            continue
    sigma_guess_override = None
    try:
        last = sorted_rows[-1]
        last_comp = _compatibility_values(last)
        if last_comp["required_sigma_logA"] is not None:
            sigma_guess_override = float(last_comp["required_sigma_logA"])
    except Exception:
        sigma_guess_override = None
    candidates = []
    for anchor_index, row in enumerate(usable_rows):
        h_row = abs(float(row.get("h", h_base) or h_base))
        h_upper = max(8.0 * h_row, 4.0 * h_min)
        if x_star_guess is not None and float(x_star_guess) > float(row["x"]):
            h_upper = max(h_upper, 2.0 * (float(x_star_guess) - float(row["x"])))
        candidates.append(
            _solve_sonic_match_from_anchor(
                ops=ops,
                fluid=fluid,
                design=design,
                config=config,
                inlet=inlet,
                area_profile=area_profile,
                anchor_row=row,
                anchor_index=anchor_index,
                max_log_step=max_log_step,
                h_min=h_min,
                h_upper=h_upper,
                x_star_guess=x_star_guess,
                sigma_guess_override=sigma_guess_override,
            )
        )

    best = min(candidates, key=lambda item: float(item.get("scaled_residual_inf") or float("inf")))
    ok_candidates = [item for item in candidates if bool(item.get("ok"))]
    selected = min(ok_candidates, key=lambda item: float(item.get("scaled_residual_inf") or float("inf"))) if ok_candidates else best
    return {
        "diagnostic": "local_sonic_match",
        "schema_version": 1,
        "ok": bool(selected.get("ok", False)),
        "selected_anchor_index": int(selected.get("anchor_index", -1)),
        "reference_ok": bool(reference.ok),
        "reference_error": reference.error,
        "reference_marcher": {
            "method": diagnostics.get("method"),
            "x_fraction": _json_float(diagnostics.get("x_fraction")),
            "attempted_h": _json_float(diagnostics.get("attempted_h")),
            "residual_inf": _json_float(diagnostics.get("residual_inf")),
            "accepted_tail_count": int(len(usable_rows)),
            "area_profile_source": diagnostics.get("area_profile_source", "design_spline"),
        },
        "left_anchor": selected.get("left_anchor"),
        "sonic_point": selected.get("sonic_point"),
        "selected_solution": selected,
        "candidate_solutions": candidates,
        "compatibility_formula": "rhs_L - sigma_logA * L_p = 0 at M=1",
        "note": (
            "This is a local endpoint match. It constrains A'/A at x_* only; "
            "it does not prescribe the downstream A(x) curve."
        ),
    }


def estimate_right_launch_curvature(
    *,
    design: DesignVector,
    config: CaseConfig,
    sonic_point: dict[str, Any],
    target_M_prime_1_per_m: float = 1000.0,
    launch_mach_increment: float = 1.0e-3,
    finite_difference_step_m: float | None = None,
) -> dict[str, Any]:
    """Estimate the local A''/A degree of freedom needed to leave M=1 smoothly."""

    mu = float(target_M_prime_1_per_m)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("target_M_prime_1_per_m must be finite and positive for a supersonic right launch.")
    ops = ops_for_numeric()
    fluid = working_fluid_for_config(config)
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
    x_star = float(sonic_point["x_m"])
    z_star = np.array([float(sonic_point["delta_log_n"]), float(sonic_point["delta_log_Te"])], dtype=float)
    logA_star = float(sonic_point["logA"])
    sigma_star = float(sonic_point["sigma_required_1_per_m"])
    A0 = max(float(config.area_scale_m2), _EPS)

    def terms_at_offset(s: float, p: np.ndarray) -> dict[str, Any]:
        z = z_star + float(s) * np.asarray(p[:2], dtype=float)
        tau = float(p[2])
        logA = logA_star + sigma_star * float(s) + 0.5 * tau * float(s) * float(s)
        sigma = sigma_star + tau * float(s)
        return _freidberg_terms_for_explicit_area(
            ops=ops,
            fluid=fluid,
            design=design,
            config=config,
            inlet=inlet,
            x=x_star + float(s),
            z=z,
            A=A0 * float(np.exp(np.clip(logA, -700.0, 700.0))),
            sigma_logA=sigma,
        )

    star = terms_at_offset(0.0, np.array([0.0, 0.0, 0.0], dtype=float))
    target_C_prime = float(star["A"] / A0) * _ELL_SECOND_AT_SONIC * mu * mu
    step = (
        max(float(config.length_m) * 1.0e-7, 1.0e-9)
        if finite_difference_step_m is None
        else float(finite_difference_step_m)
    )
    H_scale = max(1.0, abs(float(star["H_p"]) / max(float(config.length_m), _EPS)), abs(float(star["rhs_H"])))
    M_scale = max(1.0, abs(mu))
    C_scale = max(1.0 / max(float(config.length_m) ** 2, _EPS), abs(target_C_prime))

    def path_quantities(s: float, p: np.ndarray) -> np.ndarray:
        terms = terms_at_offset(s, p)
        C = float(terms["rhs_L"]) - float(terms["sigma_logA"]) * float(terms["L_p"])
        return np.array([float(terms["H_p"]), float(terms["mach"]), C], dtype=float)

    def residual(p: np.ndarray) -> np.ndarray:
        try:
            q_plus = path_quantities(step, p)
            q_minus = path_quantities(-step, p)
            deriv = (q_plus - q_minus) / (2.0 * step)
            values = np.array(
                [
                    (deriv[0] - float(star["rhs_H"])) / H_scale,
                    (deriv[1] - mu) / M_scale,
                    (deriv[2] - target_C_prime) / C_scale,
                ],
                dtype=float,
            )
            if not np.all(np.isfinite(values)):
                return np.full(3, 1e30, dtype=float)
            return values
        except Exception:
            return np.full(3, 1e30, dtype=float)

    guess = np.array([-0.9 * mu, 1.5 * mu, max(mu * mu, 1.0)], dtype=float)
    sol = least_squares(
        residual,
        guess,
        x_scale=np.array([max(abs(mu), 1.0), max(abs(mu), 1.0), max(mu * mu, 1.0)], dtype=float),
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        max_nfev=300,
    )
    values = residual(np.asarray(sol.x, dtype=float))
    residual_inf = float(np.max(np.abs(values)))
    launch_dx = float(launch_mach_increment) / mu
    first = terms_at_offset(launch_dx, np.asarray(sol.x, dtype=float))
    return _json_clean(
        {
            "ok": bool(sol.success and np.isfinite(residual_inf) and residual_inf <= 1.0e-4),
            "least_squares_success": bool(sol.success),
            "least_squares_status": int(sol.status),
            "least_squares_message": str(sol.message),
            "target_M_prime_1_per_m": mu,
            "target_dC_dx_1_per_m2": target_C_prime,
            "ell_second_at_M1": _ELL_SECOND_AT_SONIC,
            "finite_difference_step_m": step,
            "scaled_residual_inf": residual_inf,
            "scaled_residuals": values,
            "dlogn_dx_1_per_m": float(sol.x[0]),
            "dlogTe_dx_1_per_m": float(sol.x[1]),
            "required_dsigma_logA_dx_1_per_m2": float(sol.x[2]),
            "first_right_node": {
                "dx_from_sonic_m": launch_dx,
                "x_m": x_star + launch_dx,
                "x_fraction": (x_star + launch_dx) / max(float(config.length_m), _EPS),
                "mach": first["mach"],
                "T_p_K": first["T_p"],
                "n_p": first["n_p"],
                "T_e": first["T_e"],
                "A_over_A0": first["A"] / A0,
                "sigma_logA_1_per_m": first["sigma_logA"],
                "delta_log_n": z_star[0] + launch_dx * float(sol.x[0]),
                "delta_log_Te": z_star[1] + launch_dx * float(sol.x[1]),
            },
            "note": (
                "This is the local right-branch launch condition. It sets the initial curvature "
                "of log A near x_*, not the downstream area slope after the launch buffer."
            ),
        }
    )


def _suggest_sonic_mesh_nodes(
    *,
    x_left: float,
    x_star: float,
    first_right_dx: float,
    length_m: float,
    n_intervals: int,
    left_intervals: int = 16,
    right_intervals: int = 12,
    growth: float = 1.35,
) -> np.ndarray:
    left = np.linspace(float(x_left), float(x_star), int(left_intervals) + 1, dtype=float)
    right_nodes = [float(x_star)]
    left_dx = (float(x_star) - float(x_left)) / max(int(left_intervals), 1)
    dx = max(min(float(first_right_dx) / 8.0, max(left_dx, float(length_m) * 1.0e-9)), float(length_m) * 1.0e-10)
    for _ in range(int(right_intervals)):
        right_nodes.append(min(float(length_m), right_nodes[-1] + dx))
        dx = min(dx * float(growth), max(float(first_right_dx), dx))
    nodes = np.unique(np.concatenate([left, np.asarray(right_nodes, dtype=float)]))
    return nodes[(nodes >= 0.0) & (nodes <= float(length_m))]


def _local_launch_area_on_nodes(
    *,
    design: DesignVector,
    config: CaseConfig,
    area_profile: dict[str, Any] | None,
    nodes: np.ndarray,
    left_anchor: dict[str, Any],
    sonic_point: dict[str, Any],
    right_launch: dict[str, Any],
) -> dict[str, Any]:
    base = _area_arrays_for_match(
        design=design,
        config=config,
        area_profile=area_profile,
        n_intervals=config.n_intervals,
    )
    x_left = float(left_anchor["x_m"])
    x_star = float(sonic_point["x_m"])
    logA_left = _interp_area_field(base, "logA", x_left)
    sigma_left = _interp_area_field(base, "sigma_logA", x_left)
    logA_star = float(sonic_point["logA"])
    sigma_star = float(sonic_point["sigma_required_1_per_m"])
    tau = float(right_launch["required_dsigma_logA_dx_1_per_m2"])
    nodes = np.asarray(nodes, dtype=float)
    logA = np.empty_like(nodes)
    sigma = np.empty_like(nodes)
    left_mask = nodes <= x_star
    if np.any(left_mask):
        logA[left_mask], sigma[left_mask] = _hermite_segment(
            nodes[left_mask],
            x0=x_left,
            x1=x_star,
            y0=logA_left,
            y1=logA_star,
            m0=sigma_left,
            m1=sigma_star,
        )
    if np.any(~left_mask):
        s = nodes[~left_mask] - x_star
        logA[~left_mask] = logA_star + sigma_star * s + 0.5 * tau * s * s
        sigma[~left_mask] = sigma_star + tau * s
    A0 = max(float(config.area_scale_m2), _EPS)
    base_logA = np.interp(nodes, np.asarray(base["x"], dtype=float), np.asarray(base["logA"], dtype=float))
    base_sigma = np.interp(nodes, np.asarray(base["x"], dtype=float), np.asarray(base["sigma_logA"], dtype=float))
    return _json_clean(
        {
            "x_m": nodes,
            "x_fraction": nodes / max(float(config.length_m), _EPS),
            "logA_launch": logA,
            "A_over_A0_launch": np.exp(np.clip(logA, -700.0, 700.0)),
            "sigma_logA_launch_1_per_m": sigma,
            "logA_original": base_logA,
            "A_over_A0_original": np.exp(np.clip(base_logA, -700.0, 700.0)),
            "sigma_logA_original_1_per_m": base_sigma,
            "note": "Left side is a Hermite bridge into x_*; right side is only the local quadratic launch buffer.",
        }
    )


def build_sonic_mesh_matching_diagnostic(
    *,
    design: DesignVector,
    config: CaseConfig,
    n_intervals: int | None = None,
    residual_tol: float = 1e-7,
    initial_substeps_per_interval: int = 10,
    max_log_step: float = 0.25,
    min_step_fraction: float = 1e-8,
    max_steps: int = 20000,
    area_profile: dict[str, Any] | None = None,
    target_M_prime_1_per_m: float = 1000.0,
    launch_mach_increment: float = 1.0e-3,
) -> dict[str, Any]:
    """Build the local mesh/curvature diagnostic around the smooth sonic point."""

    match = solve_local_sonic_match(
        design=design,
        config=config,
        n_intervals=n_intervals,
        residual_tol=residual_tol,
        initial_substeps_per_interval=initial_substeps_per_interval,
        max_log_step=max_log_step,
        min_step_fraction=min_step_fraction,
        max_steps=max_steps,
        area_profile=area_profile,
    )
    if not bool(match.get("ok", False)):
        return {
            "diagnostic": "sonic_mesh_matching",
            "schema_version": 1,
            "ok": False,
            "local_sonic_match": match,
        }
    launch = estimate_right_launch_curvature(
        design=design,
        config=config,
        sonic_point=dict(match["sonic_point"]),
        target_M_prime_1_per_m=target_M_prime_1_per_m,
        launch_mach_increment=launch_mach_increment,
    )
    left_anchor = dict(match["left_anchor"])
    sonic_point = dict(match["sonic_point"])
    first_dx = float(launch["first_right_node"]["dx_from_sonic_m"])
    nodes = _suggest_sonic_mesh_nodes(
        x_left=float(left_anchor["x_m"]),
        x_star=float(sonic_point["x_m"]),
        first_right_dx=first_dx,
        length_m=float(config.length_m),
        n_intervals=int(config.n_intervals if n_intervals is None else n_intervals),
    )
    dx = np.diff(nodes)
    area_launch = _local_launch_area_on_nodes(
        design=design,
        config=config,
        area_profile=area_profile,
        nodes=nodes,
        left_anchor=left_anchor,
        sonic_point=sonic_point,
        right_launch=launch,
    )
    uniform_dx = float(config.length_m) / max(int(config.n_intervals if n_intervals is None else n_intervals), 1)
    return {
        "diagnostic": "sonic_mesh_matching",
        "schema_version": 1,
        "ok": bool(launch.get("ok", False)),
        "local_sonic_match": match,
        "left_anchor": left_anchor,
        "sonic_point": sonic_point,
        "right_branch_launch_condition": launch,
        "suggested_local_mesh": _json_clean(
            {
                "node_count": int(nodes.size),
                "x_m": nodes,
                "x_fraction": nodes / max(float(config.length_m), _EPS),
                "min_dx_m": None if dx.size == 0 else float(np.min(dx)),
                "max_dx_m": None if dx.size == 0 else float(np.max(dx)),
                "uniform_dx_m": uniform_dx,
                "left_anchor_to_sonic_m": float(sonic_point["x_m"]) - float(left_anchor["x_m"]),
                "left_anchor_to_sonic_over_uniform_dx": (
                    float(sonic_point["x_m"]) - float(left_anchor["x_m"])
                )
                / max(uniform_dx, _EPS),
                "first_right_dx_m": first_dx,
            }
        ),
        "area_launch_profile": area_launch,
        "equation_contract": {
            "sonic_point": [
                "H/L backward-Euler match from the accepted left anchor",
                "M(x_*) = 1",
                "rhs_L(x_*) - sigma_logA(x_*) * L_p(x_*) = 0",
            ],
            "right_launch": [
                "dH_p/dx = rhs_H at x_*",
                "dM/dx is chosen positive to leave the sonic point",
                "d(rhs_L - sigma_logA L_p)/dx = (A/A0) ell''(1) (dM/dx)^2",
            ],
        },
        "note": (
            "The output is a local sonic chart diagnostic. The right-launch quadratic "
            "should be replaced by the original or optimized A_rest after a short buffer."
        ),
    }


def _hermite_segment(
    x: np.ndarray,
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    m0: float,
    m1: float,
) -> tuple[np.ndarray, np.ndarray]:
    h = max(float(x1) - float(x0), 1e-12)
    t = np.clip((np.asarray(x, dtype=float) - float(x0)) / h, 0.0, 1.0)
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    y = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
    dh00 = (6.0 * t**2 - 6.0 * t) / h
    dh10 = 3.0 * t**2 - 4.0 * t + 1.0
    dh01 = (-6.0 * t**2 + 6.0 * t) / h
    dh11 = 3.0 * t**2 - 2.0 * t
    dydx = dh00 * y0 + dh10 * m0 + dh01 * y1 + dh11 * m1
    return y, dydx


def make_sonic_matched_area_profile(
    *,
    design: DesignVector,
    config: CaseConfig,
    x_sonic: float,
    required_sigma_logA: float,
    n_intervals: int | None = None,
    base_area_profile: dict[str, Any] | None = None,
    rejoin_fraction: float = 0.08,
) -> dict[str, np.ndarray]:
    """Construct a fixed C1 log-area curve that hits the local sonic slope."""

    n_out = int(config.n_intervals if n_intervals is None else n_intervals)
    n_area = max(8 * n_out, 400)
    x_nodes = np.linspace(0.0, float(config.length_m), n_area + 1, dtype=float)
    if base_area_profile is None:
        base = design.area_control.evaluate_profile(
            length=float(config.length_m),
            n_intervals=max(4 * n_out, 80),
            area_scale=float(config.area_scale_m2),
        )
    else:
        base = _area_profile_arrays(base_area_profile, config=config)
    base_x = np.asarray(base["x"], dtype=float)
    base_logA = np.asarray(base["logA"], dtype=float)
    base_sigma = np.asarray(base["sigma_logA"], dtype=float)

    length = max(float(config.length_m), 1e-12)
    x_star = float(np.clip(float(x_sonic), 0.0, length))
    required_sigma = float(required_sigma_logA)
    dx_nominal = length / max(n_area, 1)
    rejoin_width = max(float(rejoin_fraction) * length, 4.0 * dx_nominal)
    x_rejoin = min(length, x_star + rejoin_width)
    if x_rejoin <= x_star + 1e-12:
        x_rejoin = length

    def base_y(x: float | np.ndarray) -> np.ndarray:
        return np.interp(x, base_x, base_logA)

    def base_m(x: float | np.ndarray) -> np.ndarray:
        return np.interp(x, base_x, base_sigma)

    logA = np.asarray(base_y(x_nodes), dtype=float)
    sigma = np.asarray(base_m(x_nodes), dtype=float)
    y_star = float(base_y(x_star))
    y_rejoin = float(base_y(x_rejoin))
    m_rejoin = float(base_m(x_rejoin))

    if x_star <= 1e-10 * length:
        mask = x_nodes <= x_rejoin
        logA[mask], sigma[mask] = _hermite_segment(
            x_nodes[mask],
            x0=0.0,
            x1=x_rejoin,
            y0=0.0,
            y1=y_rejoin,
            m0=required_sigma,
            m1=m_rejoin,
        )
    else:
        left_mask = x_nodes <= x_star
        logA[left_mask], sigma[left_mask] = _hermite_segment(
            x_nodes[left_mask],
            x0=0.0,
            x1=x_star,
            y0=0.0,
            y1=y_star,
            m0=float(base_m(0.0)),
            m1=required_sigma,
        )
        mid_mask = (x_nodes > x_star) & (x_nodes <= x_rejoin)
        logA[mid_mask], sigma[mid_mask] = _hermite_segment(
            x_nodes[mid_mask],
            x0=x_star,
            x1=x_rejoin,
            y0=y_star,
            y1=y_rejoin,
            m0=required_sigma,
            m1=m_rejoin,
        )

    A = float(config.area_scale_m2) * np.exp(np.clip(logA, -700.0, 700.0))
    return {
        "x": x_nodes,
        "x_norm": x_nodes / length,
        "logA": logA,
        "A": A,
        "sigma_logA": sigma,
    }


def _seed_from_failed_freidberg(result: ReferenceProfileResult) -> dict[str, Any]:
    diagnostics = dict(result.diagnostics or {})
    tail = list(diagnostics.get("accepted_tail", []) or [])
    row = dict(tail[-1]) if tail else dict(diagnostics.get("detail", {}) or {})
    if "x" not in row and diagnostics.get("x") is not None:
        row["x"] = diagnostics.get("x")
    row.update(_compatibility_values(row))
    return row


def build_sonic_matched_freidberg_reference_profile(
    *,
    design: DesignVector,
    config: CaseConfig,
    n_intervals: int | None = None,
    residual_tol: float = 1e-7,
    initial_substeps_per_interval: int = 10,
    max_log_step: float = 0.25,
    min_step_fraction: float = 1e-8,
    max_steps: int = 20000,
    max_match_iterations: int = 3,
) -> ReferenceProfileResult:
    """Try fixed-area sonic matching before handing a profile to Firedrake."""

    area_profile: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    last_result: ReferenceProfileResult | None = None
    for attempt in range(int(max_match_iterations) + 1):
        result = build_freidberg_reference_profile(
            design=design,
            config=config,
            n_intervals=n_intervals,
            residual_tol=float(residual_tol),
            initial_substeps_per_interval=int(initial_substeps_per_interval),
            max_log_step=float(max_log_step),
            min_step_fraction=float(min_step_fraction),
            max_steps=int(max_steps),
            area_profile=area_profile,
        )
        last_result = result
        if result.ok and result.profile is not None:
            audit = audit_sonic_compatibility(profile=result.profile, design=design, config=config)
            diagnostics = {
                **dict(result.diagnostics or {}),
                "method": "sonic_matched_freidberg_hl",
                "sonic_matcher": {
                    "attempt_count": int(attempt),
                    "attempts": attempts,
                    "final_audit_summary": audit["summary"],
                },
            }
            return ReferenceProfileResult(ok=True, profile=result.profile, diagnostics=diagnostics)

        seed = _seed_from_failed_freidberg(result)
        attempts.append(
            {
                "attempt": int(attempt),
                "ok": bool(result.ok),
                "error": result.error,
                "x": _json_float(seed.get("x")),
                "x_fraction": _json_float(
                    None
                    if seed.get("x") is None
                    else float(seed["x"]) / max(float(config.length_m), _EPS)
                ),
                "mach": _json_float(seed.get("mach")),
                "T_p": _json_float(seed.get("T_p")),
                "sigma_logA": _json_float(seed.get("sigma_logA")),
                "required_sigma_logA": _json_float(seed.get("required_sigma_logA")),
                "sigma_gap_required_minus_current": _json_float(seed.get("sigma_gap_required_minus_current")),
                "scaled_compatibility_residual": _json_float(seed.get("scaled_compatibility_residual")),
            }
        )
        required_sigma = seed.get("required_sigma_logA")
        x_sonic = seed.get("x")
        if required_sigma is None or x_sonic is None:
            break
        area_profile = make_sonic_matched_area_profile(
            design=design,
            config=config,
            x_sonic=float(x_sonic),
            required_sigma_logA=float(required_sigma),
            n_intervals=int(config.n_intervals if n_intervals is None else n_intervals),
            base_area_profile=area_profile,
        )

    diagnostics = {
        "method": "sonic_matched_freidberg_hl",
        "ok": False,
        "sonic_matcher": {
            "attempt_count": int(len(attempts)),
            "attempts": attempts,
        },
    }
    if last_result is not None:
        diagnostics["last_reference_diagnostics"] = dict(last_result.diagnostics or {})
    return ReferenceProfileResult(
        ok=False,
        profile=None,
        diagnostics=diagnostics,
        error=None if last_result is None else last_result.error,
    )


def build_front_loaded_area_initial_profile(
    *,
    design: DesignVector,
    config: CaseConfig,
    n_intervals: int | None = None,
    area_ratio: float | None = None,
    width_fraction: float = 0.05,
) -> ReferenceProfileResult:
    """Generate an evaluate-only fixed-area initial guess with early expansion.

    This is intentionally not a solved reference profile.  It is a controlled
    bracket curve for checking whether the forward solver can solve any nearby
    same-area-ratio geometry before we move the curve back toward the reported
    Yamasaki shape.
    """

    n_out = int(config.n_intervals if n_intervals is None else n_intervals)
    x = np.linspace(0.0, float(config.length_m), n_out + 1, dtype=float)
    base_area = design.area_control.evaluate_profile(
        length=float(config.length_m),
        n_intervals=n_out,
        area_scale=float(config.area_scale_m2),
    )
    default_ratio = float(base_area["A"][-1] / max(float(base_area["A"][0]), _EPS))
    ratio = default_ratio if area_ratio is None else float(area_ratio)
    if not np.isfinite(ratio) or ratio <= 1.0:
        raise ValueError("front-loaded area_ratio must be finite and greater than 1.")
    width = max(float(width_fraction) * float(config.length_m), 1e-9)
    denom = max(1.0 - np.exp(-float(config.length_m) / width), 1e-12)
    log_ratio = float(np.log(ratio))
    logA = log_ratio * (1.0 - np.exp(-x / width)) / denom
    sigma = log_ratio * np.exp(-x / width) / (width * denom)
    A = float(config.area_scale_m2) * np.exp(np.clip(logA, -700.0, 700.0))
    profile = {
        "x": x,
        "x_norm": x / max(float(config.length_m), _EPS),
        "n_p": np.full_like(x, float(design.n_p_in), dtype=float),
        "T_e": np.full_like(x, float(design.T_e_in), dtype=float),
        "A": A,
        "sigma_logA": sigma,
    }
    diagnostics = {
        "method": "front_loaded_area_initial_guess",
        "initial_guess_only": True,
        "area_ratio": float(ratio),
        "width_fraction": float(width_fraction),
        "width_m": float(width),
        "sigma_logA_inlet": float(sigma[0]),
        "sigma_logA_min": float(np.nanmin(sigma)),
        "sigma_logA_max": float(np.nanmax(sigma)),
        "note": (
            "This profile is not a forward-marched solution; it is a fixed-area "
            "initial guess for evaluate-only forward-solver bracketing."
        ),
    }
    return ReferenceProfileResult(ok=True, profile=profile, diagnostics=diagnostics)
