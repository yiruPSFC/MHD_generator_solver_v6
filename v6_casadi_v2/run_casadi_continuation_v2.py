#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_core.local_algebraic_closure import E_I, H_P, K_B, M_E
from v6_global_marginal.pde_solver_v6_batch_global import project_seed_fraction_to_marginal_inlet
from v6_casadi_v2.optimize_area_profile_casadi_v2 import (
    WarmStartProfile,
    _design_value_weights_lab_poc_v2_objective,
    _evaluate_inlet_design_numeric,
    _evaluate_profile_numeric,
    optimize_area_profile,
)
from v6_global_marginal.global_postprocess_v6 import (
    compute_design_value_terms,
    design_value_weights_delta_te_only,
    design_value_weights_lab_poc,
    evaluate_design_value,
)
from v6_global_marginal.reference_recovery.global_plotting_v6 import plot_global_results_v6

_THIS_DIR = Path(__file__).resolve().parent


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a continuation schedule for the CasADi v2 area optimizer")
    p.add_argument("--np-in-guess", type=float, required=True)
    p.add_argument("--np-in-min", type=float, required=True)
    p.add_argument("--np-in-max", type=float, required=True)
    p.add_argument("--te-in-guess", type=float, required=True)
    p.add_argument("--te-in-min", type=float, required=True)
    p.add_argument("--te-in-max", type=float, required=True)
    p.add_argument("--z-in-guess", type=float, required=True)
    p.add_argument("--z-in-min", type=float, required=True)
    p.add_argument("--z-in-max", type=float, required=True)
    p.add_argument("--jx-in-guess", type=float, required=True)
    p.add_argument("--jx-in-min", type=float, required=True)
    p.add_argument("--jx-in-max", type=float, required=True)
    p.add_argument("--seed-fraction-guess", type=float, default=1e-4)
    p.add_argument("--seed-fraction-min", type=float, default=1e-13)
    p.add_argument("--seed-fraction-max", type=float, default=5e-2)
    p.add_argument(
        "--inlet-margin-mode",
        type=str,
        choices=("equality", "lower-bound"),
        default="equality",
        help="enforce G_in == min_margin or only G_in >= min_margin",
    )
    p.add_argument(
        "--warm-profile-npz",
        type=str,
        default="",
        help="optional external profile npz used as the initial v2 warm start",
    )
    p.add_argument("--B", type=float, default=10.2)
    p.add_argument(
        "--L",
        type=float,
        default=None,
        help="channel length [m]; defaults to external warm profile length when --warm-profile-npz is used, else 5.4",
    )
    p.add_argument(
        "--warm-start-policy",
        type=str,
        choices=("finite", "acceptable", "regular"),
        default="regular",
        help="criterion used before a stage result is propagated as the next warm start",
    )
    p.add_argument(
        "--continue-on-unacceptable",
        action="store_true",
        help="do not stop early on an unacceptable intermediate stage; continue relaxing",
    )
    p.add_argument(
        "--adaptive-bridge-count",
        type=int,
        default=0,
        help="number of interpolated bridge stages inserted before each scheduled stage when possible",
    )
    p.add_argument(
        "--adaptive-bridge-max-count",
        type=int,
        default=0,
        help="maximum number of bridge stages to use after repeated bridge-count doubling retries",
    )
    p.add_argument("--out-json", type=str, default="")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(_THIS_DIR / "outputs" / "continuation" / "reference_case_v2"),
        help="directory for per-stage profiles, plots, and summary json",
    )
    return p


def _stage_schedule() -> list[dict]:
    return [
        {
            "name": "stage_1",
            "n_intervals": 80,
            "transcription": "trapezoid",
            "min_margin": 0.0,
            "mach_min": None,
            "A_min_ratio": 0.95,
            "A_max_ratio": 2.1,
            "max_abs_dlogA_dx": 0.20,
            "smooth_weight": 0.0,
            "warm_profile_track_weight": 0.0,
            "warm_control_track_weight": 0.0,
            "objective_weight": 0.0,
            "ipopt_max_iter": 0,
        },
        {
            "name": "stage_2",
            "n_intervals": 80,
            "transcription": "trapezoid",
            "min_margin": 0.0,
            "mach_min": None,
            "A_min_ratio": 0.95,
            "A_max_ratio": 2.1,
            "max_abs_dlogA_dx": 0.20,
            "smooth_weight": 0.01,
            "warm_profile_track_weight": 20.0,
            "warm_control_track_weight": 5.0,
            "objective_weight": 0.01,
            "ipopt_max_iter": 1200,
        },
        {
            "name": "stage_3",
            "n_intervals": 80,
            "transcription": "hermite-simpson",
            "min_margin": 0.0,
            "mach_min": None,
            "A_min_ratio": 0.90,
            "A_max_ratio": 2.5,
            "max_abs_dlogA_dx": 0.25,
            "smooth_weight": 0.01,
            "warm_profile_track_weight": 50.0,
            "warm_control_track_weight": 10.0,
            "objective_weight": 0.02,
            "ipopt_max_iter": 1200,
        },
    ]


def _profile_payload(result) -> dict[str, np.ndarray]:
    return {
        "x": result.x,
        "n_p": result.n_p,
        "T_e": result.T_e,
        "T_p": result.T_p,
        "A": result.A,
        "v_p": result.v_p,
        "n_e": result.n_e,
        "beta": result.beta,
        "eta": result.eta,
        "Z": result.Z,
        "J_x": result.J_x,
        "J_y": result.J_y,
        "E_x": result.E_x,
        "mach": result.mach,
        "velikhov_margin": result.velikhov_margin,
        "sigma_logA": result.sigma_logA,
    }


def _dual_npz_payload(result) -> dict[str, np.ndarray]:
    arrays = {}
    dual_arrays = dict((getattr(result, "duals", {}) or {}).get("arrays", {}) or {})
    for name, values in dual_arrays.items():
        arrays[f"dual_{name}"] = np.asarray(values, dtype=float)
    return arrays


def _focused_dual_summary(result) -> dict[str, object]:
    summary = dict((getattr(result, "duals", {}) or {}).get("summary", {}) or {})
    names = (
        "sigma_upper_interval",
        "sigma_lower_interval",
        "G_lower_node",
        "G_lower_mid",
        "Inlet_G_equal_zero",
        "Inlet_G_lower_bound",
        "Mach_lower_node",
        "Mach_lower_mid",
        "A_lower_node",
        "A_upper_node",
    )
    return {name: summary.get(name, {}) for name in names if name in summary}


def _save_dual_plot(*, out_dir: Path, stage_name: str, result) -> Path | None:
    dual_arrays = dict((getattr(result, "duals", {}) or {}).get("arrays", {}) or {})
    if not dual_arrays:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_node = np.asarray(result.x, dtype=float)
    if x_node.size < 2:
        return None
    x_mid = 0.5 * (x_node[:-1] + x_node[1:])

    def _x_for(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == x_node.size:
            return x_node
        if arr.size == x_mid.size:
            return x_mid
        return np.linspace(float(x_node[0]), float(x_node[-1]), arr.size, dtype=float)

    def _plot_group(ax, names: list[str], title: str) -> None:
        plotted = False
        for name in names:
            values = np.asarray(dual_arrays.get(name, []), dtype=float).reshape(-1)
            if values.size == 0 or not np.any(np.isfinite(values)):
                continue
            ax.plot(_x_for(values), values, lw=1.8, label=name)
            plotted = True
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.grid(True, alpha=0.3, which="both")
        if plotted:
            ax.set_yscale("symlog", linthresh=1e-8, linscale=1.0)
            ax.legend(loc="best", fontsize=8)
        else:
            ax.text(0.5, 0.5, "no active series", transform=ax.transAxes, ha="center", va="center")

    save_path = out_dir / f"{stage_name}_duals.png"
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), constrained_layout=True)
    _plot_group(
        axes[0],
        ["G_lower_node", "G_lower_mid", "Inlet_G_equal_zero", "Inlet_G_lower_bound"],
        "Velikhov margin multipliers",
    )
    _plot_group(
        axes[1],
        ["Tp_lower_node", "Tp_lower_mid", "Inlet_Tp_lower"],
        "T_p lower-bound multipliers",
    )
    _plot_group(
        axes[2],
        ["A_lower_node", "A_upper_node", "sigma_lower_interval", "sigma_upper_interval"],
        "Area and dlogA/dx bound multipliers",
    )
    _plot_group(
        axes[3],
        ["Mach_lower_node", "Mach_lower_mid", "Mach_upper_node", "Mach_upper_mid", "Inlet_Mach_lower", "Inlet_Mach_upper"],
        "Mach bound multipliers",
    )
    status = str((getattr(result, "duals", {}) or {}).get("status", ""))
    fig.suptitle(f"{stage_name}: IPOPT/CasADi constraint duals ({status})", fontsize=13)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def load_warm_profile_npz(
    path: str | Path,
    *,
    n_p_in_guess: float,
    n_p_in_min: float,
    n_p_in_max: float,
    T_e_in_guess: float,
    T_e_in_min: float,
    T_e_in_max: float,
    Z_in_guess: float,
    Z_in_min: float,
    Z_in_max: float,
    J_x_in_guess: float,
    J_x_in_min: float,
    J_x_in_max: float,
    seed_fraction_guess: float,
    seed_fraction_min: float,
    seed_fraction_max: float,
    B: float = 10.2,
) -> WarmStartProfile:
    warm_path = Path(path)
    with np.load(warm_path) as data:
        missing = [name for name in ("x", "n_p", "T_e", "A") if name not in data.files]
        if missing:
            raise ValueError(f"warm profile npz missing required arrays: {missing}")
        x = np.asarray(data["x"], dtype=float).reshape(-1)
        n_p = np.asarray(data["n_p"], dtype=float).reshape(-1)
        T_e = np.asarray(data["T_e"], dtype=float).reshape(-1)
        A_raw = np.asarray(data["A"], dtype=float).reshape(-1)
        if "sigma_logA" in data.files:
            sigma = np.asarray(data["sigma_logA"], dtype=float).reshape(-1)
        else:
            sigma = np.diff(np.log(np.maximum(A_raw, 1e-20))) / np.diff(x)
        Z_arr = np.asarray(data["Z"], dtype=float).reshape(-1) if "Z" in data.files else np.zeros(0, dtype=float)
        J_x_arr = np.asarray(data["J_x"], dtype=float).reshape(-1) if "J_x" in data.files else np.zeros(0, dtype=float)
        T_p_arr = np.asarray(data["T_p"], dtype=float).reshape(-1) if "T_p" in data.files else np.zeros(0, dtype=float)
        n_e_arr = np.asarray(data["n_e"], dtype=float).reshape(-1) if "n_e" in data.files else np.zeros(0, dtype=float)
        seed_arr = (
            np.asarray(data["seed_fraction"], dtype=float).reshape(-1)
            if "seed_fraction" in data.files
            else np.zeros(0, dtype=float)
        )

    if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError("warm profile x must be strictly increasing with at least two points.")
    if n_p.size != x.size or T_e.size != x.size or A_raw.size != x.size:
        raise ValueError("warm profile n_p, T_e, and A arrays must match x length.")
    if sigma.size != x.size - 1:
        raise ValueError("warm profile sigma_logA length must be len(x)-1.")
    arrays = (x, n_p, T_e, A_raw, sigma)
    if not all(np.all(np.isfinite(arr)) for arr in arrays):
        raise ValueError("warm profile arrays must be finite.")
    inlet_A = float(A_raw[0]) if A_raw.size else 1.0
    if inlet_A <= 0.0 or not np.isfinite(inlet_A):
        raise ValueError("warm profile inlet A must be positive and finite.")
    A = A_raw / inlet_A

    def _infer_seed_fraction() -> float:
        if seed_arr.size and np.isfinite(seed_arr[0]) and float(seed_arr[0]) > 0.0:
            return float(seed_arr[0])
        if n_e_arr.size and np.isfinite(n_e_arr[0]) and float(n_e_arr[0]) > 0.0:
            n_p0 = float(n_p[0])
            T_e0 = float(T_e[0])
            n_e0 = float(n_e_arr[0])
            saha_a = 2.0 * np.pi * M_E * K_B * T_e0 / (H_P * H_P)
            saha_k = (saha_a ** 1.5) * np.exp(-E_I / (K_B * T_e0))
            if np.isfinite(saha_k) and saha_k > 0.0 and n_p0 > 0.0:
                seed_density = n_e0 + n_e0 * n_e0 / saha_k
                seed = seed_density / n_p0
                if np.isfinite(seed) and seed > 0.0:
                    return float(seed)
        if T_p_arr.size and Z_arr.size:
            status, projected = project_seed_fraction_to_marginal_inlet(
                n_p_in=float(n_p[0]),
                T_p_in=float(T_p_arr[0]),
                T_e_in=float(T_e[0]),
                B=float(B),
            )
            if int(status) == 0 and np.isfinite(projected) and float(projected) > 0.0:
                return float(projected)
        return float(seed_fraction_guess)

    inlet_J_x = float(J_x_arr[0]) if J_x_arr.size else float(J_x_in_guess)
    inferred_seed_fraction = _infer_seed_fraction()
    return WarmStartProfile(
        x=x.copy(),
        n_p=n_p.copy(),
        T_e=T_e.copy(),
        A=A.copy(),
        sigma_logA=sigma.copy(),
        inlet_n_p=float(np.clip(n_p[0] if n_p.size else n_p_in_guess, n_p_in_min, n_p_in_max)),
        inlet_T_e=float(np.clip(T_e[0] if T_e.size else T_e_in_guess, T_e_in_min, T_e_in_max)),
        inlet_Z=float(np.clip(Z_arr[0] if Z_arr.size else Z_in_guess, Z_in_min, Z_in_max)),
        inlet_I0=float(np.clip(inlet_J_x, J_x_in_min, J_x_in_max)),
        inlet_seed_fraction=float(np.clip(inferred_seed_fraction, seed_fraction_min, seed_fraction_max)),
        source=f"external_npz:{warm_path}",
    )


def _save_stage_artifacts(*, out_dir: Path, stage_name: str, result, B: float) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / stage_name
    plots_dir = out_dir / "plots"
    npz_path = base.with_suffix(".npz")
    png_path = (plots_dir / stage_name).with_suffix(".png")
    json_path = base.with_suffix(".json")

    np.savez(npz_path, **_profile_payload(result), **_dual_npz_payload(result))

    stage_status = "TRUSTED" if bool(result.success) and bool(result.acceptable) else "FAILED"
    plot_stats = plot_global_results_v6(
        _profile_payload(result),
        png_path,
        B=B,
        seed_fraction=result.inlet.seed_fraction,
        title=(
            f"{stage_status}: {stage_name}: {result.transcription}, status={result.return_status}, "
            f"acceptable={result.acceptable}, dTe={result.objective_delta_Te:.2f} K"
        ),
        dual_arrays=(getattr(result, "duals", {}) or {}).get("arrays", {}),
    )
    value_terms = compute_design_value_terms(
        x=result.x,
        T_e=result.T_e,
        T_p=result.T_p,
        n_p=result.n_p,
        n_e=result.n_e,
        mach=result.mach,
        A=result.A,
        J_x=result.J_x,
        E_x=result.E_x,
        B=float(B),
        seed_fraction=result.inlet.seed_fraction,
    )
    stage_payload = {
        "stage": stage_name,
        "success": bool(result.success),
        "acceptable": bool(result.acceptable),
        "return_status": result.return_status,
        "objective_delta_Te_K": float(result.objective_delta_Te),
        "inlet_design": {
            "n_p_in": float(result.inlet.n_p),
            "T_e_in": float(result.inlet.T_e),
            "T_p_in": float(result.inlet.T_p),
            "Z_in": float(result.inlet.Z),
            "J_x_in": float(result.inlet.J_x),
            "I_0": float(result.inlet.I_0),
            "dot_N": float(result.inlet.dot_N),
            "v_in": float(result.inlet.v_in),
            "seed_fraction": float(result.inlet.seed_fraction),
            "mach_in": float(result.inlet.mach),
            "velikhov_margin_in": float(result.inlet.velikhov_margin),
            "A_in": 1.0,
        },
        "diagnostics": result.diagnostics,
        "value_terms": value_terms.to_dict(),
        "value_profiles": {
            "delta_te_only": evaluate_design_value(
                value_terms,
                weights=design_value_weights_delta_te_only(),
                profile_name="delta_te_only",
            ).to_dict(),
            "lab_poc": evaluate_design_value(
                value_terms,
                weights=design_value_weights_lab_poc(),
                profile_name="lab_poc",
            ).to_dict(),
            "lab_poc_v2_objective": evaluate_design_value(
                value_terms,
                weights=_design_value_weights_lab_poc_v2_objective(),
                profile_name="lab_poc_v2_objective",
            ).to_dict(),
        },
        "plot_stats": {k: float(v) for k, v in plot_stats.items()},
        "dual_status": (getattr(result, "duals", {}) or {}).get("status", ""),
        "dual_summary": (getattr(result, "duals", {}) or {}).get("summary", {}),
        "focused_dual_summary": _focused_dual_summary(result),
        "dual_errors": (getattr(result, "duals", {}) or {}).get("errors", {}),
        "npz_path": str(npz_path),
        "plot_path": str(png_path),
    }
    dual_plot_path = _save_dual_plot(out_dir=out_dir, stage_name=stage_name, result=result)
    if dual_plot_path is not None:
        stage_payload["dual_plot_path"] = str(dual_plot_path)
    json_path.write_text(json.dumps(stage_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return stage_payload


def _save_baseline_comparison_plot(*, out_dir: Path, baseline_result, final_result) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path = out_dir / "plots" / "baseline_vs_final.png"
    xb = np.asarray(baseline_result.x, dtype=float)
    xf = np.asarray(final_result.x, dtype=float)
    series = [
        ("A (m^2)", baseline_result.A, final_result.A),
        ("T_e (K)", baseline_result.T_e, final_result.T_e),
        ("T_p (K)", baseline_result.T_p, final_result.T_p),
        ("Mach", baseline_result.mach, final_result.mach),
        ("Velikhov margin G", baseline_result.velikhov_margin, final_result.velikhov_margin),
        (
            "load power density (MW/m^3)",
            -(baseline_result.J_x * baseline_result.E_x) / 1e6,
            -(final_result.J_x * final_result.E_x) / 1e6,
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (label, yb, yf) in zip(axes.ravel(), series):
        if label == "Velikhov margin G":
            yb_arr = np.asarray(yb, dtype=float)
            yf_arr = np.asarray(yf, dtype=float)
            linthresh = max(
                1e-8,
                float(
                    np.nanpercentile(
                        np.concatenate(
                            [
                                np.abs(yb_arr[np.isfinite(yb_arr) & (yb_arr != 0.0)]),
                                np.abs(yf_arr[np.isfinite(yf_arr) & (yf_arr != 0.0)]),
                            ]
                        )
                        if np.any(np.isfinite(yb_arr) & (yb_arr != 0.0)) or np.any(np.isfinite(yf_arr) & (yf_arr != 0.0))
                        else np.array([1e-8], dtype=float),
                        5.0,
                    )
                ),
            )
            ax.plot(xb, yb_arr, label="baseline", lw=2)
            ax.plot(xf, yf_arr, label="optimized final", lw=2)
            ax.axhline(0.0, color="k", lw=0.8, alpha=0.35)
            ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0)
            ax.set_title("Velikhov margin G (symlog)")
            ax.set_xlabel("x (m)")
            ax.grid(True, alpha=0.3, which="both")
            continue

        ax.plot(xb, np.asarray(yb, dtype=float), label="baseline", lw=2)
        ax.plot(xf, np.asarray(yf, dtype=float), label="optimized final", lw=2)
        ax.set_title(label)
        ax.set_xlabel("x (m)")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=9)
    fig.suptitle("v2 reference case: baseline vs optimized final profile", fontsize=14)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def _make_external_baseline_result(warm: WarmStartProfile, *, B: float):
    inlet = _evaluate_inlet_design_numeric(
        n_p_in=float(warm.inlet_n_p),
        T_e_in=float(warm.inlet_T_e),
        Z_in=float(warm.inlet_Z),
        I_0=float(warm.inlet_I0),
        seed_fraction=float(warm.inlet_seed_fraction),
        B=float(B),
    )
    profile = _evaluate_profile_numeric(
        x=np.asarray(warm.x, dtype=float),
        n_p=np.asarray(warm.n_p, dtype=float),
        T_e=np.asarray(warm.T_e, dtype=float),
        A=np.asarray(warm.A, dtype=float),
        inlet=inlet,
        B=float(B),
        sigma_logA=np.asarray(warm.sigma_logA, dtype=float),
    )
    return SimpleNamespace(inlet=inlet, **profile)


_BLEND_FLOAT_FIELDS = (
    "min_margin",
    "mach_min",
    "mach_max",
    "A_min_ratio",
    "A_max_ratio",
    "max_abs_dlogA_dx",
    "np_min_ratio",
    "np_max_ratio",
    "te_min",
    "te_max_ratio",
    "tp_min",
    "margin_slack_max",
    "margin_slack_weight",
    "smooth_weight",
    "control_slew_weight",
    "control_curvature_weight",
    "state_curvature_weight",
    "warm_profile_track_weight",
    "warm_control_track_weight",
    "objective_weight",
    "ipopt_tol",
)


def _supports_warm_start(result) -> bool:
    arrays = (
        np.asarray(result.x, dtype=float),
        np.asarray(result.n_p, dtype=float),
        np.asarray(result.T_e, dtype=float),
        np.asarray(result.A, dtype=float),
        np.asarray(result.sigma_logA, dtype=float),
    )
    if not all(np.all(np.isfinite(arr)) for arr in arrays):
        return False
    inlet_scalars = (
        float(result.inlet.n_p),
        float(result.inlet.T_e),
        float(result.inlet.Z),
        float(result.inlet.I_0),
        float(result.inlet.seed_fraction),
    )
    if not all(np.isfinite(value) for value in inlet_scalars):
        return False
    if arrays[0].ndim != 1 or arrays[0].size < 2:
        return False
    if any(arr.ndim != 1 for arr in arrays[1:]):
        return False
    if arrays[1].size != arrays[0].size:
        return False
    if arrays[2].size != arrays[0].size:
        return False
    if arrays[3].size != arrays[0].size:
        return False
    if arrays[4].size != arrays[0].size - 1:
        return False
    return True


def _is_reasonable_warm_start_candidate(result) -> bool:
    if not _supports_warm_start(result):
        return False
    diagnostics = dict(getattr(result, "diagnostics", {}) or {})
    if not bool(diagnostics.get("finite_profile", False)):
        return False

    max_constraint_violation = float(diagnostics.get("max_constraint_violation", float("inf")))
    dynamic_defect_inf = float(diagnostics.get("dynamic_defect_inf", float("inf")))
    boundary_residual_inf = float(diagnostics.get("boundary_residual_inf", float("inf")))
    tp_min = float(diagnostics.get("tp_min", float("-inf")))

    if not np.isfinite(max_constraint_violation):
        return False
    if not np.isfinite(dynamic_defect_inf):
        return False
    if not np.isfinite(boundary_residual_inf):
        return False
    if not np.isfinite(tp_min):
        return False
    if not bool(diagnostics.get("regularity_ok", False)):
        return False

    return bool(
        tp_min > 0.0
        and max_constraint_violation <= 1e-2
        and dynamic_defect_inf <= 5e-2
        and boundary_residual_inf <= 5e-3
    )


def _warm_start_gate(result, *, policy: str) -> tuple[bool, str]:
    if not _supports_warm_start(result):
        return False, "unsupported_profile"

    diagnostics = dict(getattr(result, "diagnostics", {}) or {})
    solver_success = bool(getattr(result, "success", False))
    finite_profile = bool(diagnostics.get("finite_profile", False))
    acceptable = bool(diagnostics.get("acceptable", False))
    regularity_ok = bool(diagnostics.get("regularity_ok", False))

    if policy == "finite":
        return finite_profile, "" if finite_profile else "non_finite_profile"
    if policy == "acceptable":
        if not finite_profile:
            return False, "non_finite_profile"
        if not solver_success:
            return False, "solver_not_successful"
        return acceptable, "" if acceptable else "unacceptable_profile"
    if policy == "regular":
        if not finite_profile:
            return False, "non_finite_profile"
        if not solver_success:
            return False, "solver_not_successful"
        if not acceptable:
            return False, "unacceptable_profile"
        if not regularity_ok:
            return False, "irregular_profile"
        if not _is_reasonable_warm_start_candidate(result):
            return False, "failed_reasonable_warm_start_checks"
        return True, ""
    raise ValueError(f"unsupported warm_start_policy={policy!r}")


def _blend_optional_float(lhs, rhs, *, alpha: float):
    if lhs is None and rhs is None:
        return None
    if lhs is None:
        return float(rhs)
    if rhs is None:
        return float(lhs)
    return (1.0 - alpha) * float(lhs) + alpha * float(rhs)


def _supports_adaptive_bridging(source_stage: dict, target_stage: dict) -> bool:
    return bool(
        source_stage.get("transcription") == target_stage.get("transcription")
        and int(source_stage.get("n_intervals", 0)) == int(target_stage.get("n_intervals", -1))
    )


def _build_adaptive_bridge_stage(
    *,
    source_stage: dict,
    target_stage: dict,
    alpha: float,
    index: int,
    total: int,
    refinement_round: int = 0,
) -> dict:
    if not _supports_adaptive_bridging(source_stage, target_stage):
        raise ValueError("adaptive bridge requires matching transcription and n_intervals.")

    name = f"{target_stage['name']}__bridge_{index + 1}_of_{total}"
    if refinement_round > 0:
        name = f"{target_stage['name']}__bridge_r{refinement_round + 1}_{index + 1}_of_{total}"

    bridge = {
        "name": name,
        "n_intervals": int(target_stage["n_intervals"]),
        "transcription": str(target_stage["transcription"]),
        "adaptive_bridge": True,
        "adaptive_bridge_alpha": float(alpha),
        "adaptive_bridge_source": str(source_stage["name"]),
        "adaptive_bridge_target": str(target_stage["name"]),
        "adaptive_bridge_refinement_round": int(refinement_round),
    }
    for field in _BLEND_FLOAT_FIELDS:
        value = _blend_optional_float(source_stage.get(field), target_stage.get(field), alpha=float(alpha))
        if value is not None:
            bridge[field] = value

    source_iter = int(source_stage.get("ipopt_max_iter", target_stage.get("ipopt_max_iter", 0)))
    target_iter = int(target_stage.get("ipopt_max_iter", source_iter))
    bridge["ipopt_max_iter"] = int(round((1.0 - float(alpha)) * source_iter + float(alpha) * target_iter))
    return bridge


def _build_adaptive_bridge_sequence(
    *,
    source_stage: dict,
    target_stage: dict,
    bridge_count: int,
    refinement_round: int = 0,
) -> list[dict]:
    if bridge_count <= 0 or not _supports_adaptive_bridging(source_stage, target_stage):
        return []
    alphas = [(i + 1) / float(bridge_count + 1) for i in range(int(bridge_count))]
    return [
        _build_adaptive_bridge_stage(
            source_stage=source_stage,
            target_stage=target_stage,
            alpha=float(alpha),
            index=i,
            total=int(bridge_count),
            refinement_round=int(refinement_round),
        )
        for i, alpha in enumerate(alphas)
    ]


def _initial_warm_bridge_source_stage(target_stage: dict) -> dict:
    source = {
        "name": "initial_external_warm_profile",
        "n_intervals": int(target_stage["n_intervals"]),
        "transcription": str(target_stage["transcription"]),
        "objective_weight": 0.0,
        "warm_profile_track_weight": max(128.0, float(target_stage.get("warm_profile_track_weight", 0.0))),
        "warm_control_track_weight": max(32.0, float(target_stage.get("warm_control_track_weight", 0.0))),
        "ipopt_max_iter": int(target_stage.get("ipopt_max_iter", 1000)),
    }
    for field in _BLEND_FLOAT_FIELDS:
        if field not in source and field in target_stage:
            source[field] = target_stage[field]
    return source


def _save_stage_record(
    *,
    stages_out: list[dict],
    stage: dict,
    stage_input_warm_source: str,
    result,
    out_dir_path: Path | None,
    B: float,
) -> dict[str, object]:
    record = {
        "name": stage["name"],
        "success": bool(result.success),
        "acceptable": bool(result.acceptable),
        "return_status": result.return_status,
        "objective_delta_Te_K": float(result.objective_delta_Te),
        "min_velikhov_margin": float(np.nanmin(result.velikhov_margin))
        if result.velikhov_margin.size
        else float("nan"),
        "min_velikhov_margin_x_m": float(result.diagnostics.get("velikhov_margin_min_x_m", float("nan"))),
        "velikhov_margin_lt_1e_3_fraction": float(
            result.diagnostics.get("velikhov_margin_lt_1e_3_fraction", float("nan"))
        ),
        "min_mach": float(np.nanmin(result.mach)) if result.mach.size else float("nan"),
        "outlet_area_m2": float(result.A[-1]),
        "dynamic_defect_inf": float(result.diagnostics["dynamic_defect_inf"]),
        "max_constraint_violation": float(result.diagnostics["max_constraint_violation"]),
        "regularity_ok": bool(result.diagnostics.get("regularity_ok", False)),
        "sigma_sign_changes": int(result.diagnostics.get("sigma_sign_changes", 0)),
        "area_step_sign_changes": int(result.diagnostics.get("area_step_sign_changes", 0)),
        "sigma_bound_hit_fraction": float(result.diagnostics.get("sigma_bound_hit_fraction", 0.0)),
        "warm_start_input_source": stage_input_warm_source,
        "stage_kind": "adaptive_bridge" if bool(stage.get("adaptive_bridge", False)) else "scheduled",
        "focused_dual_summary": _focused_dual_summary(result),
        "inlet_design": {
            "n_p_in": float(result.inlet.n_p),
            "T_e_in": float(result.inlet.T_e),
            "T_p_in": float(result.inlet.T_p),
            "Z_in": float(result.inlet.Z),
            "J_x_in": float(result.inlet.J_x),
            "I_0": float(result.inlet.I_0),
            "dot_N": float(result.inlet.dot_N),
            "seed_fraction": float(result.inlet.seed_fraction),
        },
    }
    if bool(stage.get("adaptive_bridge", False)):
        record["adaptive_bridge_alpha"] = float(stage["adaptive_bridge_alpha"])
        record["adaptive_bridge_source"] = str(stage["adaptive_bridge_source"])
        record["adaptive_bridge_target"] = str(stage["adaptive_bridge_target"])
        record["adaptive_bridge_refinement_round"] = int(stage.get("adaptive_bridge_refinement_round", 0))
    stages_out.append(record)
    if out_dir_path is not None:
        stages_out[-1]["artifacts"] = _save_stage_artifacts(
            out_dir=out_dir_path,
            stage_name=stage["name"],
            result=result,
            B=float(B),
        )
    return stages_out[-1]


def _run_stage(
    *,
    stage: dict,
    warm_profile: WarmStartProfile | None,
    n_p_in_guess: float,
    n_p_in_min: float,
    n_p_in_max: float,
    T_e_in_guess: float,
    T_e_in_min: float,
    T_e_in_max: float,
    Z_in_guess: float,
    Z_in_min: float,
    Z_in_max: float,
    J_x_in_guess: float,
    J_x_in_min: float,
    J_x_in_max: float,
    seed_fraction_guess: float,
    seed_fraction_min: float,
    seed_fraction_max: float,
    inlet_margin_mode: str,
    B: float,
    L: float,
):
    return optimize_area_profile(
        n_p_in_guess=float(n_p_in_guess),
        n_p_in_min=float(n_p_in_min),
        n_p_in_max=float(n_p_in_max),
        T_e_in_guess=float(T_e_in_guess),
        T_e_in_min=float(T_e_in_min),
        T_e_in_max=float(T_e_in_max),
        Z_in_guess=float(Z_in_guess),
        Z_in_min=float(Z_in_min),
        Z_in_max=float(Z_in_max),
        J_x_in_guess=float(J_x_in_guess),
        J_x_in_min=float(J_x_in_min),
        J_x_in_max=float(J_x_in_max),
        seed_fraction_guess=float(seed_fraction_guess),
        seed_fraction_min=float(seed_fraction_min),
        seed_fraction_max=float(seed_fraction_max),
        inlet_margin_mode=str(stage.get("inlet_margin_mode", inlet_margin_mode)),
        B=float(B),
        length=float(L),
        warm_profile=warm_profile,
        **{
            k: v
            for k, v in stage.items()
            if k not in ("name", "inlet_margin_mode") and not str(k).startswith("adaptive_bridge")
        },
    )


def _make_warm_profile(result, *, source: str) -> WarmStartProfile:
    return WarmStartProfile(
        x=result.x,
        n_p=result.n_p,
        T_e=result.T_e,
        A=result.A,
        sigma_logA=result.sigma_logA,
        inlet_n_p=float(result.inlet.n_p),
        inlet_T_e=float(result.inlet.T_e),
        inlet_Z=float(result.inlet.Z),
        inlet_I0=float(result.inlet.I_0),
        inlet_seed_fraction=float(result.inlet.seed_fraction),
        source=source,
    )


def _stage_ref(record: dict | None) -> dict[str, object] | None:
    if record is None:
        return None
    out = {
        "name": str(record.get("name", "")),
        "success": bool(record.get("success", False)),
        "acceptable": bool(record.get("acceptable", False)),
        "return_status": str(record.get("return_status", "")),
        "objective_delta_Te_K": record.get("objective_delta_Te_K"),
        "stage_kind": str(record.get("stage_kind", "")),
        "warm_start_adopted": bool(record.get("warm_start_adopted", False)),
        "warm_start_rejection_reason": str(record.get("warm_start_rejection_reason", "")),
    }
    if "adaptive_bridge_alpha" in record:
        out["adaptive_bridge_alpha"] = float(record["adaptive_bridge_alpha"])
        out["adaptive_bridge_target"] = str(record.get("adaptive_bridge_target", ""))
        out["adaptive_bridge_refinement_round"] = int(record.get("adaptive_bridge_refinement_round", 0))
    if "inlet_design" in record:
        out["inlet_design"] = dict(record.get("inlet_design", {}) or {})
    return out


def _partition_stage_records(stages: list[dict]) -> dict[str, object]:
    trusted = [stage for stage in stages if bool(stage.get("warm_start_adopted", False))]
    failed = [
        stage
        for stage in stages
        if (
            not bool(stage.get("success", False))
            or not bool(stage.get("acceptable", False))
            or not bool(stage.get("warm_start_adopted", False))
        )
    ]
    return {
        "trusted_stages": [_stage_ref(stage) for stage in trusted],
        "failed_attempts": [_stage_ref(stage) for stage in failed],
        "last_trusted_stage": _stage_ref(trusted[-1] if trusted else None),
        "first_failed_stage": _stage_ref(failed[0] if failed else None),
    }


def _next_bridge_count(current: int, *, max_count: int) -> int:
    if current <= 0 or current >= max_count:
        return current
    return min(max_count, max(current * 2, current + 1))


def run_continuation(
    *,
    n_p_in_guess: float,
    n_p_in_min: float,
    n_p_in_max: float,
    T_e_in_guess: float,
    T_e_in_min: float,
    T_e_in_max: float,
    Z_in_guess: float,
    Z_in_min: float,
    Z_in_max: float,
    J_x_in_guess: float,
    J_x_in_min: float,
    J_x_in_max: float,
    seed_fraction_guess: float = 1e-4,
    seed_fraction_min: float = 1e-13,
    seed_fraction_max: float = 5e-2,
    inlet_margin_mode: str = "equality",
    B: float = 10.2,
    L: float = 5.4,
    stage_schedule: list[dict] | None = None,
    out_dir: str | Path | None = None,
    out_json: str | Path = "",
    stop_on_unacceptable: bool = True,
    warm_start_policy: str = "regular",
    adaptive_bridge_count: int = 0,
    adaptive_bridge_max_count: int = 0,
    warm_profile: WarmStartProfile | None = None,
) -> dict[str, object]:
    out_dir_path = None if out_dir in (None, "") else Path(out_dir)
    schedule = _stage_schedule() if stage_schedule is None else list(stage_schedule)
    initial_warm_profile_source = "" if warm_profile is None else str(warm_profile.source)
    stages_out: list[dict] = []
    baseline_result = (
        _make_external_baseline_result(warm_profile, B=float(B))
        if warm_profile is not None
        else None
    )
    final_attempt = None
    final_trusted = None
    stopped_early = False
    stopped_after_failed_bridge = False
    bridge_stop: dict[str, object] | None = None
    continued_unacceptable_stages = 0
    last_adopted_stage: dict | None = None
    effective_bridge_max_count = max(int(adaptive_bridge_count), int(adaptive_bridge_max_count))
    completed_schedule = True

    for stage in schedule:
        if adaptive_bridge_count > 0 and warm_profile is not None and (
            last_adopted_stage is not None or bool(initial_warm_profile_source)
        ):
            backup_warm_profile = warm_profile
            backup_source_stage = (
                dict(last_adopted_stage)
                if last_adopted_stage is not None
                else _initial_warm_bridge_source_stage(stage)
            )
            current_bridge_count = int(adaptive_bridge_count)
            refinement_round = 0
            bridge_attempted = False
            bridge_sequence_completed = False
            last_failed_bridge_record: dict | None = None
            while current_bridge_count > 0:
                bridge_sequence = _build_adaptive_bridge_sequence(
                    source_stage=backup_source_stage,
                    target_stage=stage,
                    bridge_count=int(current_bridge_count),
                    refinement_round=int(refinement_round),
                )
                if not bridge_sequence:
                    break

                bridge_attempted = True
                round_warm_profile = backup_warm_profile
                round_source_stage = dict(backup_source_stage)
                sequence_completed = True
                for bridge_stage in bridge_sequence:
                    bridge_input_warm_source = str(round_warm_profile.source)
                    bridge_result = _run_stage(
                        stage=bridge_stage,
                        warm_profile=round_warm_profile,
                        n_p_in_guess=float(n_p_in_guess),
                        n_p_in_min=float(n_p_in_min),
                        n_p_in_max=float(n_p_in_max),
                        T_e_in_guess=float(T_e_in_guess),
                        T_e_in_min=float(T_e_in_min),
                        T_e_in_max=float(T_e_in_max),
                        Z_in_guess=float(Z_in_guess),
                        Z_in_min=float(Z_in_min),
                        Z_in_max=float(Z_in_max),
                        J_x_in_guess=float(J_x_in_guess),
                        J_x_in_min=float(J_x_in_min),
                        J_x_in_max=float(J_x_in_max),
                        seed_fraction_guess=float(seed_fraction_guess),
                        seed_fraction_min=float(seed_fraction_min),
                        seed_fraction_max=float(seed_fraction_max),
                        inlet_margin_mode=str(inlet_margin_mode),
                        B=float(B),
                        L=float(L),
                    )
                    bridge_record = _save_stage_record(
                        stages_out=stages_out,
                        stage=bridge_stage,
                        stage_input_warm_source=bridge_input_warm_source,
                        result=bridge_result,
                        out_dir_path=out_dir_path,
                        B=float(B),
                    )
                    final_attempt = bridge_result
                    if baseline_result is None:
                        baseline_result = bridge_result
                    bridge_adopted, bridge_reason = _warm_start_gate(
                        bridge_result,
                        policy=str(warm_start_policy),
                    )
                    bridge_record["warm_start_adopted"] = bool(bridge_adopted)
                    bridge_record["warm_start_rejection_reason"] = bridge_reason
                    if not bridge_adopted:
                        sequence_completed = False
                        last_failed_bridge_record = bridge_record
                        break
                    round_warm_profile = _make_warm_profile(
                        bridge_result,
                        source=f"continuation:{bridge_stage['name']}",
                    )
                    round_source_stage = dict(bridge_stage)
                    final_trusted = bridge_result

                if sequence_completed:
                    warm_profile = round_warm_profile
                    last_adopted_stage = round_source_stage
                    bridge_sequence_completed = True
                    break
                next_bridge_count = _next_bridge_count(
                    int(current_bridge_count),
                    max_count=int(effective_bridge_max_count),
                )
                if next_bridge_count <= current_bridge_count:
                    break
                current_bridge_count = next_bridge_count
                refinement_round += 1
            if bridge_attempted and not bridge_sequence_completed:
                stopped_early = True
                stopped_after_failed_bridge = True
                completed_schedule = False
                bridge_stop = {
                    "blocked_target_stage": str(stage.get("name", "")),
                    "reason": "adaptive_bridge_failed_before_target_stage",
                    "last_trusted_stage": None,
                    "failed_stage": _stage_ref(last_failed_bridge_record),
                    "max_stable_alpha": None,
                    "next_failed_alpha": None,
                }
                trusted_for_target = [
                    item
                    for item in stages_out
                    if (
                        bool(item.get("warm_start_adopted", False))
                        and str(item.get("adaptive_bridge_target", "")) == str(stage.get("name", ""))
                    )
                ]
                if trusted_for_target:
                    bridge_stop["last_trusted_stage"] = _stage_ref(trusted_for_target[-1])
                    bridge_stop["max_stable_alpha"] = float(trusted_for_target[-1]["adaptive_bridge_alpha"])
                if last_failed_bridge_record is not None and "adaptive_bridge_alpha" in last_failed_bridge_record:
                    bridge_stop["next_failed_alpha"] = float(last_failed_bridge_record["adaptive_bridge_alpha"])
                break

        stage_input_warm_source = "constant_auto" if warm_profile is None else str(warm_profile.source)
        result = _run_stage(
            stage=stage,
            warm_profile=warm_profile,
            n_p_in_guess=float(n_p_in_guess),
            n_p_in_min=float(n_p_in_min),
            n_p_in_max=float(n_p_in_max),
            T_e_in_guess=float(T_e_in_guess),
            T_e_in_min=float(T_e_in_min),
            T_e_in_max=float(T_e_in_max),
            Z_in_guess=float(Z_in_guess),
            Z_in_min=float(Z_in_min),
            Z_in_max=float(Z_in_max),
            J_x_in_guess=float(J_x_in_guess),
            J_x_in_min=float(J_x_in_min),
            J_x_in_max=float(J_x_in_max),
            seed_fraction_guess=float(seed_fraction_guess),
            seed_fraction_min=float(seed_fraction_min),
            seed_fraction_max=float(seed_fraction_max),
            inlet_margin_mode=str(inlet_margin_mode),
            B=float(B),
            L=float(L),
        )
        stage_record = _save_stage_record(
            stages_out=stages_out,
            stage=stage,
            stage_input_warm_source=stage_input_warm_source,
            result=result,
            out_dir_path=out_dir_path,
            B=float(B),
        )
        final_attempt = result
        if baseline_result is None:
            baseline_result = result

        warm_start_adopted, warm_start_rejection_reason = _warm_start_gate(
            result,
            policy=str(warm_start_policy),
        )
        stage_record["warm_start_adopted"] = bool(warm_start_adopted)
        stage_record["warm_start_rejection_reason"] = warm_start_rejection_reason
        if warm_start_adopted:
            warm_profile = _make_warm_profile(
                result,
                source=f"continuation:{stage['name']}",
            )
            last_adopted_stage = dict(stage)
            final_trusted = result

        if (not result.acceptable) and bool(stop_on_unacceptable):
            stopped_early = True
            completed_schedule = False
            break
        if not result.acceptable:
            continued_unacceptable_stages += 1

    record_partitions = _partition_stage_records(stages_out)
    payload = {
        "ok": bool(completed_schedule and final_attempt is not None and final_attempt.acceptable),
        "solver_success": bool(final_attempt.success) if final_attempt is not None else False,
        "stopped_early": bool(stopped_early),
        "stopped_after_failed_bridge": bool(stopped_after_failed_bridge),
        "stop_on_unacceptable": bool(stop_on_unacceptable),
        "continued_unacceptable_stages": int(continued_unacceptable_stages),
        "warm_start_policy": str(warm_start_policy),
        "adaptive_bridge_count": int(adaptive_bridge_count),
        "adaptive_bridge_max_count": int(effective_bridge_max_count),
        "inlet_design_problem": {
            "np_in": {"guess": float(n_p_in_guess), "min": float(n_p_in_min), "max": float(n_p_in_max)},
            "te_in": {"guess": float(T_e_in_guess), "min": float(T_e_in_min), "max": float(T_e_in_max)},
            "z_in": {"guess": float(Z_in_guess), "min": float(Z_in_min), "max": float(Z_in_max)},
            "jx_in": {"guess": float(J_x_in_guess), "min": float(J_x_in_min), "max": float(J_x_in_max)},
            "seed_fraction": {
                "guess": float(seed_fraction_guess),
                "min": float(seed_fraction_min),
                "max": float(seed_fraction_max),
            },
            "inlet_margin_mode": str(inlet_margin_mode),
            "A_in": 1.0,
        },
        "B": float(B),
        "L": float(L),
        "initial_warm_profile_source": initial_warm_profile_source,
        "stages": stages_out,
        **record_partitions,
        "bridge_stop": bridge_stop,
        "final_attempt_return_status": "" if final_attempt is None else final_attempt.return_status,
        "final_attempt_objective_delta_Te_K": None
        if final_attempt is None
        else float(final_attempt.objective_delta_Te),
        "final_attempt_diagnostics": None if final_attempt is None else final_attempt.diagnostics,
        "final_trusted_return_status": "" if final_trusted is None else final_trusted.return_status,
        "final_trusted_objective_delta_Te_K": None
        if final_trusted is None
        else float(final_trusted.objective_delta_Te),
        "final_trusted_diagnostics": None if final_trusted is None else final_trusted.diagnostics,
        "final_focused_dual_summary": None if final_attempt is None else _focused_dual_summary(final_attempt),
        "final_trusted_focused_dual_summary": None if final_trusted is None else _focused_dual_summary(final_trusted),
        "final_return_status": "" if final_attempt is None else final_attempt.return_status,
        "final_objective_delta_Te_K": None if final_attempt is None else float(final_attempt.objective_delta_Te),
        "final_diagnostics": None if final_attempt is None else final_attempt.diagnostics,
        "final_inlet_design": None
        if final_attempt is None
        else {
            "n_p_in": float(final_attempt.inlet.n_p),
            "T_e_in": float(final_attempt.inlet.T_e),
            "T_p_in": float(final_attempt.inlet.T_p),
            "Z_in": float(final_attempt.inlet.Z),
            "J_x_in": float(final_attempt.inlet.J_x),
            "seed_fraction": float(final_attempt.inlet.seed_fraction),
        },
        "final_trusted_inlet_design": None
        if final_trusted is None
        else {
            "n_p_in": float(final_trusted.inlet.n_p),
            "T_e_in": float(final_trusted.inlet.T_e),
            "T_p_in": float(final_trusted.inlet.T_p),
            "Z_in": float(final_trusted.inlet.Z),
            "J_x_in": float(final_trusted.inlet.J_x),
            "seed_fraction": float(final_trusted.inlet.seed_fraction),
        },
    }

    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif out_dir_path is not None:
        summary_path = out_dir_path / "continuation_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if out_dir_path is not None and final_trusted is not None:
        final_npz = out_dir_path / "final_acceptable.npz"
        final_png = out_dir_path / "plots" / "final_acceptable.png"
        np.savez(final_npz, **_profile_payload(final_trusted), **_dual_npz_payload(final_trusted))
        plot_global_results_v6(
            _profile_payload(final_trusted),
            final_png,
            B=float(B),
            seed_fraction=final_trusted.inlet.seed_fraction,
            title=(
                f"final trusted profile: {final_trusted.transcription}, status={final_trusted.return_status}, "
                f"acceptable={final_trusted.acceptable}, dTe={final_trusted.objective_delta_Te:.2f} K"
            ),
            dual_arrays=(getattr(final_trusted, "duals", {}) or {}).get("arrays", {}),
        )
        _save_dual_plot(out_dir=out_dir_path, stage_name="final_acceptable", result=final_trusted)
        if baseline_result is not None:
            _save_baseline_comparison_plot(
                out_dir=out_dir_path,
                baseline_result=baseline_result,
                final_result=final_trusted,
            )

    return payload


def main() -> int:
    args = _build_parser().parse_args()
    warm_profile = None
    if args.warm_profile_npz:
        warm_profile = load_warm_profile_npz(
            args.warm_profile_npz,
            n_p_in_guess=float(args.np_in_guess),
            n_p_in_min=float(args.np_in_min),
            n_p_in_max=float(args.np_in_max),
            T_e_in_guess=float(args.te_in_guess),
            T_e_in_min=float(args.te_in_min),
            T_e_in_max=float(args.te_in_max),
            Z_in_guess=float(args.z_in_guess),
            Z_in_min=float(args.z_in_min),
            Z_in_max=float(args.z_in_max),
            J_x_in_guess=float(args.jx_in_guess),
            J_x_in_min=float(args.jx_in_min),
            J_x_in_max=float(args.jx_in_max),
            seed_fraction_guess=float(args.seed_fraction_guess),
            seed_fraction_min=float(args.seed_fraction_min),
            seed_fraction_max=float(args.seed_fraction_max),
            B=float(args.B),
        )
    run_length = (
        float(args.L)
        if args.L is not None
        else float(warm_profile.x[-1] - warm_profile.x[0])
        if warm_profile is not None
        else 5.4
    )
    payload = run_continuation(
        n_p_in_guess=float(args.np_in_guess),
        n_p_in_min=float(args.np_in_min),
        n_p_in_max=float(args.np_in_max),
        T_e_in_guess=float(args.te_in_guess),
        T_e_in_min=float(args.te_in_min),
        T_e_in_max=float(args.te_in_max),
        Z_in_guess=float(args.z_in_guess),
        Z_in_min=float(args.z_in_min),
        Z_in_max=float(args.z_in_max),
        J_x_in_guess=float(args.jx_in_guess),
        J_x_in_min=float(args.jx_in_min),
        J_x_in_max=float(args.jx_in_max),
        seed_fraction_guess=float(args.seed_fraction_guess),
        seed_fraction_min=float(args.seed_fraction_min),
        seed_fraction_max=float(args.seed_fraction_max),
        inlet_margin_mode=str(args.inlet_margin_mode),
        B=float(args.B),
        L=float(run_length),
        out_dir=args.out_dir,
        out_json=args.out_json,
        stop_on_unacceptable=not bool(args.continue_on_unacceptable),
        warm_start_policy=str(args.warm_start_policy),
        adaptive_bridge_count=int(args.adaptive_bridge_count),
        adaptive_bridge_max_count=int(args.adaptive_bridge_max_count),
        warm_profile=warm_profile,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
