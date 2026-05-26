from __future__ import annotations

import math
from typing import Any

import numpy as np

from .design import CaseConfig, DesignVector
from .legacy_physics import closure_state, inlet_design_generic, ops_for_numeric
from .transport import working_fluid_for_config


K_B = 1.380649e-23
_EPS = 1e-300
_BRANCHES = ("subsonic", "supersonic")
_BRANCH_POLICIES = ("continuity", "subsonic", "supersonic", "any")


def _json_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_abs(value: float | None) -> float | None:
    return None if value is None else abs(float(value))


def _abs_residual_score(result: dict[str, Any]) -> float:
    value = result.get("abs_residual_K")
    return float("inf") if value is None else float(value)


def _branch_bounds(branch: str) -> tuple[float, float]:
    if branch == "subsonic":
        return 1e-5, 1.0 - 1e-10
    if branch == "supersonic":
        return 1.0 + 1e-10, 50.0
    raise ValueError(f"unknown branch={branch!r}; expected one of {_BRANCHES}")


def _profile_arrays(profile: dict[str, Any], config: CaseConfig) -> dict[str, np.ndarray]:
    required = ("x", "n_p", "T_e", "A", "sigma_logA")
    arrays = {name: np.asarray(profile[name], dtype=float).reshape(-1) for name in required}
    n = arrays["x"].size
    if any(values.size != n for values in arrays.values()):
        raise ValueError("profile arrays x, n_p, T_e, A, and sigma_logA must have matching lengths.")
    if "x_norm" in profile:
        x_norm = np.asarray(profile["x_norm"], dtype=float).reshape(-1)
        if x_norm.size != n:
            raise ValueError("profile x_norm must match x length when present.")
    else:
        x_norm = arrays["x"] / max(float(config.length_m), _EPS)
    arrays["x_norm"] = x_norm
    return arrays


def _primitive_from_mach(
    *,
    H_p: float,
    L_p: float,
    T_e: float,
    mach: float,
    context: dict[str, Any],
) -> tuple[dict[str, float], float]:
    if H_p <= 0.0 or L_p <= 0.0 or T_e <= 0.0 or mach <= 0.0:
        raise ValueError("H_p, L_p, T_e, and mach must be positive.")
    M = float(mach)
    M2 = M * M
    dot_N = float(context["dot_N"])
    A0 = float(context["area_scale_m2"])
    mp = float(context["heavy_particle_mass_kg"])
    T_p_from_H = 6.0 * float(H_p) * A0 / max(5.0 * K_B * dot_N * (M2 + 3.0), _EPS)
    if not math.isfinite(T_p_from_H) or T_p_from_H <= 0.0:
        raise ValueError("H/L reconstruction produced nonpositive T_p.")
    v_p = math.sqrt(max(5.0 * K_B * T_p_from_H * M2 / max(3.0 * mp, _EPS), 0.0))
    A = A0 * float(L_p) * (M2 + 3.0) * (M2 + 3.0) / M
    if not (math.isfinite(v_p) and math.isfinite(A)) or v_p <= 0.0 or A <= 0.0:
        raise ValueError("H/L reconstruction produced nonpositive velocity or area.")
    n_p = dot_N / max(A * v_p, _EPS)
    if not math.isfinite(n_p) or n_p <= 0.0:
        raise ValueError("H/L reconstruction produced nonpositive density.")
    closure = closure_state(
        ops=context["ops"],
        n_p=float(n_p),
        T_e=float(T_e),
        A=float(A),
        dot_N=dot_N,
        I_0=float(context["I_0"]),
        seed_fraction=float(context["seed_fraction"]),
        B=float(context["B_T"]),
        working_fluid=context["working_fluid"],
    )
    residual = T_p_from_H - float(closure["T_p"])
    if not math.isfinite(residual):
        raise ValueError("H/L closure residual is nonfinite.")
    return (
        {
            "n_p": float(n_p),
            "T_e": float(T_e),
            "T_p_from_H_K": float(T_p_from_H),
            "closure_T_p_K": float(closure["T_p"]),
            "A_m2": float(A),
            "v_p_m_per_s": float(v_p),
            "mach": float(M),
        },
        float(residual),
    )


def _bisect_root(fn, lower: float, upper: float, *, iterations: int = 90) -> float:
    f_lower = float(fn(lower))
    f_upper = float(fn(upper))
    if not math.isfinite(f_lower) or not math.isfinite(f_upper) or f_lower * f_upper > 0.0:
        raise ValueError("invalid bisection bracket")
    lo = float(lower)
    hi = float(upper)
    flo = f_lower
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = float(fn(mid))
        if not math.isfinite(f_mid):
            break
        if abs(f_mid) <= 1e-12:
            return mid
        if flo * f_mid <= 0.0:
            hi = mid
        else:
            lo = mid
            flo = f_mid
    return 0.5 * (lo + hi)


def _refine_min_abs(fn, center: float, lower: float, upper: float) -> float:
    lo = max(float(lower), float(center) / 1.8)
    hi = min(float(upper), float(center) * 1.8)
    if not lo < hi:
        return float(center)
    a = math.log(lo)
    b = math.log(hi)
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)

    def score(log_mach: float) -> float:
        try:
            value = fn(math.exp(log_mach))
        except (OverflowError, ValueError):
            return float("inf")
        return value * value if math.isfinite(value) else float("inf")

    fc = score(c)
    fd = score(d)
    for _ in range(100):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = score(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = score(d)
    return math.exp(0.5 * (a + b))


def _solve_branch(
    *,
    H_p: float,
    L_p: float,
    T_e: float,
    branch: str,
    context: dict[str, Any],
    mach_hint: float | None,
    tolerance_K: float,
) -> dict[str, Any]:
    lower, upper = _branch_bounds(branch)
    if H_p <= 0.0 or L_p <= 0.0 or T_e <= 0.0:
        return {
            "branch": branch,
            "success": False,
            "mach": None,
            "residual_K": None,
            "abs_residual_K": None,
            "candidate_count": 0,
            "bracket_count": 0,
            "message": "H_p, L_p, and T_e must be positive",
            "reconstructed": None,
        }

    def residual(mach: float) -> float:
        _, value = _primitive_from_mach(H_p=H_p, L_p=L_p, T_e=T_e, mach=mach, context=context)
        return value

    candidates: list[float] = []
    bracket_count = 0
    if mach_hint is not None and math.isfinite(float(mach_hint)) and float(mach_hint) > 0.0:
        hint = min(max(float(mach_hint), lower), upper)
        candidates.append(hint)
        candidates.append(_refine_min_abs(residual, hint, lower, upper))
        lo = max(lower, hint / 2.0)
        hi = min(upper, hint * 2.0)
        if lo < hi:
            try:
                if residual(lo) * residual(hi) <= 0.0:
                    bracket_count += 1
                    candidates.append(_bisect_root(residual, lo, hi))
            except (OverflowError, ValueError):
                pass

    grid = np.exp(np.linspace(math.log(lower), math.log(upper), 180))
    vals = []
    for mach in grid:
        try:
            vals.append(residual(float(mach)))
        except (OverflowError, ValueError):
            vals.append(float("nan"))
    values = np.asarray(vals, dtype=float)
    finite = np.isfinite(values)
    if np.any(finite):
        best_idx = int(np.nanargmin(np.where(finite, np.abs(values), np.inf)))
        candidates.append(float(grid[best_idx]))
        candidates.append(_refine_min_abs(residual, float(grid[best_idx]), lower, upper))
    for idx in range(grid.size - 1):
        if not (finite[idx] and finite[idx + 1]):
            continue
        if values[idx] == 0.0:
            candidates.append(float(grid[idx]))
        elif values[idx] * values[idx + 1] < 0.0:
            try:
                bracket_count += 1
                candidates.append(_bisect_root(residual, float(grid[idx]), float(grid[idx + 1])))
            except ValueError:
                pass

    best_mach = None
    best_residual = None
    best_primitive = None
    best_abs = float("inf")
    for candidate in candidates:
        if not math.isfinite(float(candidate)) or float(candidate) <= 0.0:
            continue
        try:
            primitive, value = _primitive_from_mach(
                H_p=H_p,
                L_p=L_p,
                T_e=T_e,
                mach=float(candidate),
                context=context,
            )
        except (OverflowError, ValueError):
            continue
        abs_value = abs(float(value))
        if abs_value < best_abs:
            best_abs = abs_value
            best_mach = float(candidate)
            best_residual = float(value)
            best_primitive = primitive

    if best_mach is None or best_residual is None:
        return {
            "branch": branch,
            "success": False,
            "mach": None,
            "residual_K": None,
            "abs_residual_K": None,
            "candidate_count": int(len(candidates)),
            "bracket_count": int(bracket_count),
            "message": "no finite Mach candidate",
            "reconstructed": None,
        }
    return {
        "branch": branch,
        "success": bool(abs(best_residual) <= float(tolerance_K)),
        "mach": _json_float(best_mach),
        "residual_K": _json_float(best_residual),
        "abs_residual_K": _json_abs(_json_float(best_residual)),
        "candidate_count": int(len(candidates)),
        "bracket_count": int(bracket_count),
        "message": None if abs(best_residual) <= float(tolerance_K) else "best candidate exceeds tolerance",
        "reconstructed": None if best_primitive is None else {key: _json_float(value) for key, value in best_primitive.items()},
    }


def _choose_branch(
    branch_results: dict[str, dict[str, Any]],
    *,
    branch_policy: str,
    original_mach: float | None,
    previous_chosen_mach: float | None,
) -> str | None:
    if branch_policy in _BRANCHES:
        return branch_policy
    finite = [name for name in _BRANCHES if branch_results[name].get("mach") is not None]
    if not finite:
        return None
    successful = [name for name in finite if bool(branch_results[name].get("success"))]
    pool = successful or finite
    if branch_policy == "any":
        return min(pool, key=lambda name: _abs_residual_score(branch_results[name]))
    target = previous_chosen_mach if previous_chosen_mach is not None else original_mach
    if target is None or not math.isfinite(float(target)) or float(target) <= 0.0:
        return min(pool, key=lambda name: _abs_residual_score(branch_results[name]))
    return min(
        pool,
        key=lambda name: (
            abs(math.log(max(float(branch_results[name]["mach"]), _EPS) / max(float(target), _EPS))),
            _abs_residual_score(branch_results[name]),
        ),
    )


def audit_freidberg_branches(
    *,
    profile: dict[str, Any],
    design: DesignVector,
    config: CaseConfig,
    branch_policy: str = "continuity",
    tolerance_K: float = 1e-3,
) -> dict[str, Any]:
    """Audit algebraic branch choices induced by a primitive Firedrake profile.

    This is diagnostic only: it does not alter the forward solve.  The H/L
    transform intentionally follows the current Firedrake residual's T_p floor
    so the report matches the code path being audited.
    """

    policy = str(branch_policy).strip().lower().replace("-", "_")
    if policy not in _BRANCH_POLICIES:
        raise ValueError(f"unknown branch_policy={branch_policy!r}; expected one of {_BRANCH_POLICIES}")
    tol = float(tolerance_K)
    if not math.isfinite(tol) or tol < 0.0:
        raise ValueError("tolerance_K must be finite and nonnegative.")
    arrays = _profile_arrays(profile, config)
    ops = ops_for_numeric()
    fluid = working_fluid_for_config(config)
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
    context = {
        "ops": ops,
        "working_fluid": fluid,
        "dot_N": float(inlet["dot_N"]),
        "I_0": float(design.I_0),
        "seed_fraction": float(design.seed_fraction),
        "B_T": float(design.B_T),
        "area_scale_m2": float(config.area_scale_m2),
        "heavy_particle_mass_kg": float(fluid.heavy_particle_mass_kg),
    }

    rows: list[dict[str, Any]] = []
    previous_chosen_mach = None
    previous_chosen_branch = None
    branch_switch_count = 0
    original_mach_values: list[float] = []
    original_mach_entries: list[tuple[int, float]] = []
    original_T_p_values: list[float] = []
    chosen_abs_residuals: list[float] = []

    for idx in range(arrays["x"].size):
        n_p = float(arrays["n_p"][idx])
        T_e = float(arrays["T_e"][idx])
        A = float(arrays["A"][idx])
        try:
            closure = closure_state(
                ops=ops,
                n_p=n_p,
                T_e=T_e,
                A=A,
                dot_N=float(inlet["dot_N"]),
                I_0=float(design.I_0),
                seed_fraction=design.seed_fraction,
                B=float(design.B_T),
                working_fluid=fluid,
            )
            v_p = float(closure["v_p"])
            T_p_raw = float(closure["T_p"])
            T_p_for_H = max(T_p_raw, 1.0)
            mach = float(closure["mach"])
            M2 = mach * mach
            H_p = (A * n_p * v_p / float(config.area_scale_m2)) * (
                2.5 * K_B * T_p_for_H + 0.5 * float(fluid.heavy_particle_mass_kg) * v_p * v_p
            )
            L_p = mach * (A / float(config.area_scale_m2)) / max((M2 + 3.0) * (M2 + 3.0), _EPS)
            original_mach_values.append(mach)
            original_mach_entries.append((idx, mach))
            original_T_p_values.append(T_p_raw)
            branch_results = {
                branch: _solve_branch(
                    H_p=float(H_p),
                    L_p=float(L_p),
                    T_e=T_e,
                    branch=branch,
                    context=context,
                    mach_hint=mach,
                    tolerance_K=tol,
                )
                for branch in _BRANCHES
            }
            chosen_branch = _choose_branch(
                branch_results,
                branch_policy=policy,
                original_mach=mach,
                previous_chosen_mach=previous_chosen_mach,
            )
            chosen_result = None if chosen_branch is None else branch_results[chosen_branch]
            chosen_success = bool(chosen_result.get("success")) if chosen_result is not None else False
            chosen_mach = None if chosen_result is None else chosen_result.get("mach")
            if chosen_result is not None and chosen_result.get("abs_residual_K") is not None:
                chosen_abs_residuals.append(float(chosen_result["abs_residual_K"]))
            if chosen_branch is not None and previous_chosen_branch is not None and chosen_branch != previous_chosen_branch:
                branch_switch_count += 1
            if chosen_mach is not None:
                previous_chosen_mach = float(chosen_mach)
            if chosen_branch is not None:
                previous_chosen_branch = chosen_branch
            row = {
                "index": int(idx),
                "x": _json_float(arrays["x"][idx]),
                "x_norm": _json_float(arrays["x_norm"][idx]),
                "original": {
                    "n_p": _json_float(n_p),
                    "T_e_K": _json_float(T_e),
                    "A_m2": _json_float(A),
                    "T_p_raw_K": _json_float(T_p_raw),
                    "T_p_used_for_H_K": _json_float(T_p_for_H),
                    "v_p_m_per_s": _json_float(v_p),
                    "mach": _json_float(mach),
                    "H_p": _json_float(H_p),
                    "L_p": _json_float(L_p),
                },
                "branches": branch_results,
                "chosen_branch": chosen_branch,
                "chosen_success": chosen_success,
            }
        except Exception as exc:
            row = {
                "index": int(idx),
                "x": _json_float(arrays["x"][idx]),
                "x_norm": _json_float(arrays["x_norm"][idx]),
                "original": {
                    "n_p": _json_float(n_p),
                    "T_e_K": _json_float(T_e),
                    "A_m2": _json_float(A),
                },
                "branches": {
                    branch: {
                        "branch": branch,
                        "success": False,
                        "mach": None,
                        "residual_K": None,
                        "abs_residual_K": None,
                        "candidate_count": 0,
                        "bracket_count": 0,
                        "message": f"{type(exc).__name__}: {exc}",
                        "reconstructed": None,
                    }
                    for branch in _BRANCHES
                },
                "chosen_branch": None,
                "chosen_success": False,
            }
        rows.append(row)

    chosen_failures = [row for row in rows if not bool(row.get("chosen_success"))]
    finite_original_mach = np.asarray(original_mach_values, dtype=float)
    closest_to_sonic = None
    if finite_original_mach.size and original_mach_entries:
        sonic_idx = int(np.nanargmin(np.abs(finite_original_mach - 1.0)))
        row_index, sonic_mach = original_mach_entries[sonic_idx]
        row = rows[row_index]
        closest_to_sonic = {
            "index": int(row["index"]),
            "x": row["x"],
            "x_norm": row["x_norm"],
            "original_mach": _json_float(sonic_mach),
        }
    first_failure = None
    if chosen_failures:
        failed = chosen_failures[0]
        chosen = failed.get("chosen_branch")
        chosen_result = None if chosen is None else failed["branches"][chosen]
        first_failure = {
            "index": int(failed["index"]),
            "x": failed["x"],
            "x_norm": failed["x_norm"],
            "chosen_branch": chosen,
            "abs_residual_K": None if chosen_result is None else chosen_result.get("abs_residual_K"),
            "message": None if chosen_result is None else chosen_result.get("message"),
        }

    summary = {
        "ok": bool(len(chosen_failures) == 0),
        "n_points": int(len(rows)),
        "branch_policy": policy,
        "tolerance_K": tol,
        "chosen_failure_count": int(len(chosen_failures)),
        "subsonic_success_count": int(sum(bool(row["branches"]["subsonic"]["success"]) for row in rows)),
        "supersonic_success_count": int(sum(bool(row["branches"]["supersonic"]["success"]) for row in rows)),
        "branch_switch_count": int(branch_switch_count),
        "first_chosen_failure": first_failure,
        "closest_to_sonic": closest_to_sonic,
        "min_original_T_p_K": _json_float(np.nanmin(original_T_p_values)) if original_T_p_values else None,
        "min_original_mach": _json_float(np.nanmin(original_mach_values)) if original_mach_values else None,
        "max_original_mach": _json_float(np.nanmax(original_mach_values)) if original_mach_values else None,
        "negative_original_T_p_count": int(sum(value < 0.0 for value in original_T_p_values)),
        "max_abs_chosen_residual_K": _json_float(np.nanmax(chosen_abs_residuals)) if chosen_abs_residuals else None,
    }
    return {
        "audit": "freidberg_branch_audit",
        "schema_version": 1,
        "summary": summary,
        "rows": rows,
    }
