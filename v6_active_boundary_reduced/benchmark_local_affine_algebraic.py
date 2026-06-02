from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import numpy as np


def _install_numba_stub_if_requested(argv: list[str]) -> None:
    if "--python-numba-stub" not in argv:
        return
    import types

    def njit(*args, **_kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn

    sys.modules.setdefault("numba", types.SimpleNamespace(njit=njit))


_install_numba_stub_if_requested(sys.argv)

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v6_firedrake_reduced.cases.freidberg_reference import load_reference_profile
from v6_firedrake_reduced.design import load_case_config

from v6_active_boundary_reduced import policy as policy_mod
from v6_active_boundary_reduced.local_affine import (
    ForwardAffineCoefficients,
    _closure_G_gradients,
    compute_forward_affine_coefficients,
)
from v6_active_boundary_reduced.numba_physics import closure_state_numba, dynamic_terms_numba
from v6_active_boundary_reduced.policy import (
    AnchorState,
    PreparationSettings,
    anchor_from_dict,
    anchor_from_profile,
    recover_preparation_profile,
)
from v6_active_boundary_reduced.policy_types import PhysicsParamsLike


def compute_forward_affine_coefficients_algebraic(
    *,
    n_p: float,
    T_e: float,
    A: float,
    logA: float,
    params: PhysicsParamsLike,
    log_gradient_eps: float = 1.0e-5,
) -> ForwardAffineCoefficients:
    """Same local-affine operator as local_affine.py, written in closed 2x2 form.

    Momentum/energy residual:

        M11*n' + M12*Te' + M13*u - Hm = 0
        E11*n' + E12*Te' + E13*u - He = 0

    with u=A'.  Therefore

        [n', Te'] = D^{-1} [Hm, He] + u*D^{-1}[-M13, -E13].
    """

    n = float(n_p)
    te = float(T_e)
    area = float(A)
    if n <= 0.0 or te <= 0.0 or area <= 0.0:
        raise ValueError("n_p, T_e, and A must be positive for local affine coefficients.")

    terms = dynamic_terms_numba(
        n,
        te,
        area,
        0.0,
        float(params.dot_N),
        float(params.I_0),
        float(params.seed_fraction),
        float(params.B),
        float(params.heavy_particle_mass_kg),
        float(params.seed_ionization_energy_J),
        float(params.sigma_ep),
    )
    M11 = float(terms[0])
    M12 = float(terms[1])
    M13 = float(terms[2])
    E11 = float(terms[3])
    E12 = float(terms[4])
    E13 = float(terms[5])
    Hm = float(terms[7])
    He = float(terms[8])
    det_D = float(terms[9])

    if not np.isfinite(det_D) or abs(det_D) <= 0.0:
        y0_n = y0_te = y1_n = y1_te = float("nan")
    else:
        inv_det = 1.0 / det_D
        y0_n = (E12 * Hm - M12 * He) * inv_det
        y0_te = (-E11 * Hm + M11 * He) * inv_det
        y1_n = (-E12 * M13 + M12 * E13) * inv_det
        y1_te = (E11 * M13 - M11 * E13) * inv_det

    closure = closure_state_numba(
        n,
        te,
        area,
        float(params.dot_N),
        float(params.I_0),
        float(params.seed_fraction),
        float(params.B),
        float(params.heavy_particle_mass_kg),
        float(params.seed_ionization_energy_J),
        float(params.sigma_ep),
    )
    T_p = float(closure[10])
    dTp_dTe = float(closure[11])
    dTp_dn = float(closure[12])
    dTp_dA = float(closure[13])
    G = float(closure[18])
    phi = float(te / max(T_p, 1.0e-300) - 1.0)

    inv_Tp2 = 1.0 / max(T_p * T_p, 1.0e-300)
    dPhi_dn = -te * inv_Tp2 * dTp_dn
    dPhi_dTe = 1.0 / max(T_p, 1.0e-300) - te * inv_Tp2 * dTp_dTe
    dPhi_dA = -te * inv_Tp2 * dTp_dA
    a0 = float(dPhi_dn * y0_n + dPhi_dTe * y0_te)
    a1 = float(dPhi_dn * y1_n + dPhi_dTe * y1_te + dPhi_dA)

    dG_dn, dG_dTe, dG_dA = _closure_G_gradients(
        n_p=n,
        T_e=te,
        A=area,
        params=params,
        log_gradient_eps=float(log_gradient_eps),
    )
    b0 = float(dG_dn * y0_n + dG_dTe * y0_te)
    b1 = float(dG_dn * y1_n + dG_dTe * y1_te + dG_dA)

    return ForwardAffineCoefficients(
        a0=a0,
        a1=a1,
        b0=b0,
        b1=b1,
        det_D=det_D,
        phi_current=phi,
        G_current=G,
        T_p_current=T_p,
        A_current=area,
        logA_current=float(logA),
        n_prime_a0=float(y0_n),
        n_prime_a1=float(y1_n),
        Te_prime_a0=float(y0_te),
        Te_prime_a1=float(y1_te),
        f0_momentum=Hm,
        f0_energy=He,
        f1_momentum=float(-M13),
        f1_energy=float(-E13),
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _load_anchor(args: argparse.Namespace, *, config) -> AnchorState:
    if args.anchor_json is not None:
        payload = json.loads(Path(args.anchor_json).read_text(encoding="utf-8"))
        return anchor_from_dict(payload, config=config)

    if args.profile_npz is None:
        if str(args.case).strip().lower().replace("-", "_") != "freidberg_reference":
            raise ValueError("--profile-npz is required unless --case is freidberg_reference.")
        profile = load_reference_profile()
        source = f"{config.case}:built_in_profile"
    else:
        with np.load(args.profile_npz) as data:
            profile = {name: np.asarray(data[name], dtype=float) for name in data.files}
        source = str(args.profile_npz)
    return anchor_from_profile(profile, index=int(args.anchor_profile_index), config=config, source=source)


def _settings_from_args(args: argparse.Namespace) -> PreparationSettings:
    return PreparationSettings(
        n_steps=int(args.n_steps),
        dx=float(args.dx),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        curvature_max=None if bool(args.no_curvature_bound) else float(args.curvature_max),
        g_floor=float(args.g_floor),
        tp_floor_K=float(args.tp_floor),
        scan_points=int(args.scan_points),
        refine_iterations=int(args.refine_iterations),
        active_tol=float(args.active_tol),
        residual_tol=float(args.residual_tol),
    )


def _run_with_operator(
    *,
    operator: Callable[..., ForwardAffineCoefficients],
    config,
    anchor: AnchorState,
    settings: PreparationSettings,
) -> dict[str, Any]:
    original = policy_mod.compute_forward_affine_coefficients
    policy_mod.compute_forward_affine_coefficients = operator
    try:
        return recover_preparation_profile(config=config, anchor=anchor, settings=settings)
    finally:
        policy_mod.compute_forward_affine_coefficients = original


def _time_operator(
    *,
    label: str,
    operator: Callable[..., ForwardAffineCoefficients],
    config,
    anchor: AnchorState,
    settings: PreparationSettings,
    repeat: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    times = []
    payload = _run_with_operator(operator=operator, config=config, anchor=anchor, settings=settings)
    for _ in range(max(int(repeat), 1)):
        t0 = time.perf_counter()
        payload = _run_with_operator(operator=operator, config=config, anchor=anchor, settings=settings)
        times.append(time.perf_counter() - t0)
    return payload, {
        "label": str(label),
        "repeat": int(max(int(repeat), 1)),
        "times_s": times,
        "min_s": min(times),
        "median_s": statistics.median(times),
        "mean_s": statistics.fmean(times),
        "max_s": max(times),
        "ok": bool(payload.get("ok", False)),
        "n_segments": int(len(payload.get("segments", []))),
    }


def _max_abs_diff(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], fields: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    n = min(len(rows_a), len(rows_b))
    for field in fields:
        values = []
        for idx in range(n):
            va = rows_a[idx].get(field)
            vb = rows_b[idx].get(field)
            try:
                fa = float(va)
                fb = float(vb)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fa) and math.isfinite(fb):
                values.append(abs(fa - fb))
        out[field] = max(values) if values else None
    return out


def _compare_payloads(baseline: dict[str, Any], algebraic: dict[str, Any]) -> dict[str, Any]:
    base_segments = list(baseline.get("segments", []))
    alg_segments = list(algebraic.get("segments", []))
    base_nodes = list(baseline.get("nodes", []))
    alg_nodes = list(algebraic.get("nodes", []))
    segment_count = min(len(base_segments), len(alg_segments))
    node_count = min(len(base_nodes), len(alg_nodes))

    decision_fields = [
        "support_type",
        "affine_support_type",
        "selected_endpoint_source",
        "objective_bound_kind",
        "validation_status",
        "finite_step_solver_method",
    ]
    decision_mismatches = []
    for idx in range(segment_count):
        for field in decision_fields:
            if str(base_segments[idx].get(field)) != str(alg_segments[idx].get(field)):
                decision_mismatches.append(
                    {
                        "segment": idx,
                        "field": field,
                        "baseline": base_segments[idx].get(field),
                        "algebraic": alg_segments[idx].get(field),
                    }
                )

    return {
        "ok_match": bool(baseline.get("ok")) == bool(algebraic.get("ok")),
        "segment_count_baseline": int(len(base_segments)),
        "segment_count_algebraic": int(len(alg_segments)),
        "node_count_baseline": int(len(base_nodes)),
        "node_count_algebraic": int(len(alg_nodes)),
        "decision_mismatch_count": int(len(decision_mismatches)),
        "decision_mismatches_first10": decision_mismatches[:10],
        "segment_max_abs_diff": _max_abs_diff(
            base_segments,
            alg_segments,
            [
                "sigma",
                "sigma_selected",
                "A_prime_selected",
                "a0",
                "a1",
                "b0",
                "b1",
                "p0",
                "p1",
                "q0",
                "q1",
                "p1q1_reverse",
                "objective_drop_predicted",
                "G_margin_upstream_predicted",
                "G_margin_upstream",
                "delta_gain",
                "max_abs_scaled_residual",
            ],
        ),
        "node_max_abs_diff": _max_abs_diff(
            base_nodes,
            alg_nodes,
            ["n_p", "T_e", "A", "sigma_logA", "Delta", "G", "T_p", "mach", "beta", "Z"],
        ),
    }


def _operator_only_benchmark(
    *,
    baseline_payload: dict[str, Any],
    config,
    repeat: int,
) -> dict[str, Any]:
    params = policy_mod._physics_params(config)
    nodes = list(baseline_payload.get("nodes", []))
    states = [
        (
            float(node["n_p"]),
            float(node["T_e"]),
            float(node["A"]),
            float(node["logA"]),
        )
        for node in nodes
    ]

    def run_operator(operator: Callable[..., ForwardAffineCoefficients]) -> list[ForwardAffineCoefficients]:
        return [
            operator(n_p=n, T_e=te, A=area, logA=log_area, params=params)
            for n, te, area, log_area in states
        ]

    baseline_coeffs = run_operator(compute_forward_affine_coefficients)
    algebraic_coeffs = run_operator(compute_forward_affine_coefficients_algebraic)
    coefficient_diff = {
        field: max(
            abs(float(getattr(base, field)) - float(getattr(alg, field)))
            for base, alg in zip(baseline_coeffs, algebraic_coeffs, strict=True)
        )
        for field in ("a0", "a1", "b0", "b1", "n_prime_a0", "n_prime_a1", "Te_prime_a0", "Te_prime_a1")
    }

    timings: dict[str, dict[str, Any]] = {}
    for label, operator in (
        ("baseline_np_linalg_solve", compute_forward_affine_coefficients),
        ("algebraic_2x2_determinant", compute_forward_affine_coefficients_algebraic),
    ):
        # One warm pass keeps allocation/import/cache effects out of the repeated loop.
        run_operator(operator)
        times = []
        for _ in range(max(int(repeat), 1)):
            t0 = time.perf_counter()
            run_operator(operator)
            times.append(time.perf_counter() - t0)
        n_calls = max(len(states), 1)
        timings[label] = {
            "repeat": int(max(int(repeat), 1)),
            "states_per_repeat": int(len(states)),
            "total_calls": int(len(states) * max(int(repeat), 1)),
            "min_s": min(times),
            "median_s": statistics.median(times),
            "mean_s": statistics.fmean(times),
            "max_s": max(times),
            "median_us_per_call": 1.0e6 * statistics.median(times) / n_calls,
            "times_s": times,
        }

    baseline_median = float(timings["baseline_np_linalg_solve"]["median_s"])
    algebraic_median = float(timings["algebraic_2x2_determinant"]["median_s"])
    return {
        "n_states": int(len(states)),
        "coefficient_max_abs_diff": coefficient_diff,
        "timing": timings,
        "median_speedup_baseline_over_algebraic": baseline_median / max(algebraic_median, 1.0e-300),
        "median_percent_change": (algebraic_median / max(baseline_median, 1.0e-300) - 1.0) * 100.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the existing local-affine coefficient operator against "
            "the closed-form MR/ER determinant version on a reverse preparation rollout."
        )
    )
    parser.add_argument("--case", default="freidberg_reference")
    parser.add_argument("--profile-npz", type=Path, default=None)
    parser.add_argument("--anchor-json", type=Path, default=None)
    parser.add_argument("--anchor-profile-index", type=int, default=1)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--operator-repeat", type=int, default=100)
    parser.add_argument("--sigma-min", type=float, default=-0.5)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--curvature-max", type=float, default=8.0)
    parser.add_argument("--no-curvature-bound", action="store_true")
    parser.add_argument("--g-floor", type=float, default=0.0)
    parser.add_argument("--tp-floor", type=float, default=300.0)
    parser.add_argument("--scan-points", type=int, default=41)
    parser.add_argument("--refine-iterations", type=int, default=24)
    parser.add_argument("--active-tol", type=float, default=1e-6)
    parser.add_argument("--residual-tol", type=float, default=1e-8)
    parser.add_argument(
        "--python-numba-stub",
        action="store_true",
        help="Replace numba.njit with a no-op before imports. Useful only when the local Python env cannot import numba.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_case_config(case=str(args.case))
    anchor = _load_anchor(args, config=config)
    settings = _settings_from_args(args)

    baseline, baseline_timing = _time_operator(
        label="baseline_np_linalg_solve",
        operator=compute_forward_affine_coefficients,
        config=config,
        anchor=anchor,
        settings=settings,
        repeat=int(args.repeat),
    )
    algebraic, algebraic_timing = _time_operator(
        label="algebraic_2x2_determinant",
        operator=compute_forward_affine_coefficients_algebraic,
        config=config,
        anchor=anchor,
        settings=settings,
        repeat=int(args.repeat),
    )
    comparison = _compare_payloads(baseline, algebraic)
    operator_only = _operator_only_benchmark(
        baseline_payload=baseline,
        config=config,
        repeat=int(args.operator_repeat),
    )

    speedup_median = baseline_timing["median_s"] / max(float(algebraic_timing["median_s"]), 1.0e-300)
    summary = {
        "case": str(config.case),
        "anchor": {
            "source": str(anchor.source),
            "source_index": int(anchor.source_index),
            "x": float(anchor.x),
            "n_p": float(anchor.state.n_p),
            "T_e": float(anchor.state.T_e),
            "A": float(anchor.state.area(config)),
            "sigma_logA": float(anchor.sigma_logA),
        },
        "settings": settings.__dict__,
        "timing": {
            "baseline": baseline_timing,
            "algebraic": algebraic_timing,
            "median_speedup_baseline_over_algebraic": float(speedup_median),
            "median_percent_change": float((1.0 / max(speedup_median, 1.0e-300) - 1.0) * 100.0),
        },
        "comparison": comparison,
        "operator_only": operator_only,
    }

    if args.out_json is not None:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return 0 if comparison["ok_match"] and int(comparison["decision_mismatch_count"]) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
