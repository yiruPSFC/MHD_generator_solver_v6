from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .design import DESIGN_VARIABLE_NAMES, CaseConfig, DesignBounds, DesignVector
from .legacy_physics import closure_state, inlet_design_generic, ops_for_numeric
from .transport import working_fluid_for_config


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_profile(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        profile = {name: np.asarray(data[name], dtype=float) for name in data.files}
    if "sigma_logA" not in profile:
        x = np.asarray(profile["x"], dtype=float).reshape(-1)
        A = np.asarray(profile["A"], dtype=float).reshape(-1)
        profile["sigma_logA"] = np.gradient(np.log(np.maximum(A / max(float(A[0]), 1e-300), 1e-300)), x)
    if "x_norm" not in profile:
        x = np.asarray(profile["x"], dtype=float).reshape(-1)
        profile["x_norm"] = (x - float(x[0])) / max(float(x[-1] - x[0]), 1e-300)
    return profile


def _config_from_case_payload(payload: dict[str, Any]) -> CaseConfig:
    bounds_payload = dict(payload["bounds"])
    lower = DesignVector.from_dict({name: values["min"] for name, values in bounds_payload.items()})
    upper = DesignVector.from_dict({name: values["max"] for name, values in bounds_payload.items()})
    design = DesignVector.from_dict(dict(payload["design"]))
    return CaseConfig(
        case=str(payload["case"]),
        objective_profile=str(payload["objective_profile"]),
        length_m=float(payload["length_m"]),
        area_scale_m2=float(payload["area_scale_m2"]),
        B_T=float(design.B_T),
        working_fluid_profile=str(payload["working_fluid_profile"]),
        n_intervals=int(payload["n_intervals"]),
        design=design,
        bounds=DesignBounds(lower=lower, upper=upper),
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def _profile_path_from_run(run_dir: Path, run_summary: dict[str, Any]) -> Path:
    candidate = run_summary.get("profile_npz") or run_dir / "profile.npz"
    path = Path(str(candidate))
    if not path.is_absolute():
        path = run_dir / path
    return path.resolve()


def _fields_from_profile(*, profile: dict[str, np.ndarray], config: CaseConfig) -> dict[str, np.ndarray]:
    ops = ops_for_numeric()
    fluid = working_fluid_for_config(config)
    design = config.design
    x = np.asarray(profile["x"], dtype=float).reshape(-1)
    n_p = np.asarray(profile["n_p"], dtype=float).reshape(-1)
    T_e = np.asarray(profile["T_e"], dtype=float).reshape(-1)
    A = np.asarray(profile["A"], dtype=float).reshape(-1)
    sigma = np.asarray(profile["sigma_logA"], dtype=float).reshape(-1)
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
    closures = [
        closure_state(
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
        for n_val, te_val, area_val in zip(n_p, T_e, A, strict=True)
    ]
    J_x = np.asarray([float(item["J_x"]) for item in closures], dtype=float)
    E_x = np.asarray([float(item["E_x"]) for item in closures], dtype=float)
    out = {
        "x": x,
        "x_norm": (x - float(x[0])) / max(float(x[-1] - x[0]), 1e-300),
        "n_p": n_p,
        "T_e": T_e,
        "A": A,
        "sigma_logA": sigma,
        "T_p": np.asarray([float(item["T_p"]) for item in closures], dtype=float),
        "mach": np.asarray([float(item["mach"]) for item in closures], dtype=float),
        "G": np.asarray([float(item["G"]) for item in closures], dtype=float),
        "J_x": J_x,
        "J_y": np.asarray([float(item["J_y"]) for item in closures], dtype=float),
        "E_x": E_x,
        "power_density_W_per_m": -A * J_x * E_x,
    }
    return out


def _field_summary(fields: dict[str, np.ndarray], *, floor: float) -> dict[str, Any]:
    x = np.asarray(fields["x"], dtype=float)
    x_norm = np.asarray(fields["x_norm"], dtype=float)
    G = np.asarray(fields["G"], dtype=float)
    argmin = int(np.nanargmin(G))
    return {
        "node_count": int(x.size),
        "G_floor": float(floor),
        "G_min": float(G[argmin]),
        "G_argmin_index": argmin,
        "G_argmin_x_m": float(x[argmin]),
        "G_argmin_x_norm": float(x_norm[argmin]),
        "G_inlet": float(G[0]),
        "G_outlet": float(G[-1]),
        "G_violated_count": int(np.count_nonzero(G < float(floor))),
        "G_active_count_1e_minus_6": int(np.count_nonzero(np.abs(G - float(floor)) <= 1e-6)),
        "T_p_min_K": float(np.nanmin(fields["T_p"])),
        "T_p_inlet_K": float(fields["T_p"][0]),
        "mach_min": float(np.nanmin(fields["mach"])),
        "mach_inlet": float(fields["mach"][0]),
        "power_integral_W": float(np.trapezoid(fields["power_density_W_per_m"], x)),
    }


def _plot(
    *,
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    floor: float,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    ref_x = reference["x_norm"]
    cand_x = candidate["x_norm"]

    axes[0, 0].plot(ref_x, reference["A"] / reference["A"][0], label="reference")
    axes[0, 0].plot(cand_x, candidate["A"] / candidate["A"][0], "o-", label="candidate")
    axes[0, 0].set_title("Area ratio")
    axes[0, 0].set_ylabel("A / A_in")

    axes[0, 1].plot(ref_x, reference["mach"], label="reference")
    axes[0, 1].plot(cand_x, candidate["mach"], "o-", label="candidate")
    axes[0, 1].set_title("Mach")
    axes[0, 1].set_ylabel("M")

    axes[1, 0].plot(ref_x, reference["T_p"], label="reference")
    axes[1, 0].plot(cand_x, candidate["T_p"], "o-", label="candidate")
    axes[1, 0].set_title("Heavy-particle temperature")
    axes[1, 0].set_ylabel("T_p [K]")

    axes[1, 1].plot(ref_x, reference["T_e"], label="reference")
    axes[1, 1].plot(cand_x, candidate["T_e"], "o-", label="candidate")
    axes[1, 1].set_title("Electron temperature")
    axes[1, 1].set_ylabel("T_e [K]")

    axes[2, 0].plot(ref_x, reference["G"], label="reference")
    axes[2, 0].plot(cand_x, candidate["G"], "o-", label="candidate")
    axes[2, 0].axhline(floor, color="black", linewidth=0.9, linestyle="--", label="hard floor")
    axes[2, 0].set_title("Velikhov margin")
    axes[2, 0].set_ylabel("G")
    axes[2, 0].set_yscale("symlog", linthresh=1e-6)

    axes[2, 1].plot(ref_x, reference["power_density_W_per_m"] / 1e6, label="reference")
    axes[2, 1].plot(cand_x, candidate["power_density_W_per_m"] / 1e6, "o-", label="candidate")
    axes[2, 1].set_title("MHD power density")
    axes[2, 1].set_ylabel("-A J_x E_x [MW/m]")

    for ax in axes.flat:
        ax.set_xlabel("x / L")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Freidberg area-only reference/candidate profiles.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-png", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--g-floor", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    run_summary = _load_json(run_dir / "run_summary.json")
    benchmark_summary = _load_json(run_dir / "benchmark_summary.json")
    reference_config = _config_from_case_payload(dict(benchmark_summary["case_config"]))
    candidate_config = _config_from_case_payload(dict(run_summary["case_config"]))
    floor = (
        float(args.g_floor)
        if args.g_floor is not None
        else float(candidate_config.metadata.get("velikhov_hard_floor", 0.0))
    )

    reference_profile = _load_profile(run_dir / "reference_initial_profile.npz")
    candidate_profile = _load_profile(_profile_path_from_run(run_dir, run_summary))
    reference_fields = _fields_from_profile(profile=reference_profile, config=reference_config)
    candidate_fields = _fields_from_profile(profile=candidate_profile, config=candidate_config)

    out_png = args.out_png or run_dir / "profile_comparison.png"
    out_json = args.out_json or run_dir / "profile_comparison_summary.json"
    out_png = Path(out_png).resolve()
    out_json = Path(out_json).resolve()
    _plot(reference=reference_fields, candidate=candidate_fields, floor=floor, out_png=out_png)
    payload = {
        "reference": _field_summary(reference_fields, floor=floor),
        "candidate": _field_summary(candidate_fields, floor=floor),
        "candidate_minus_reference": {
            "G_min": float(_field_summary(candidate_fields, floor=floor)["G_min"])
            - float(_field_summary(reference_fields, floor=floor)["G_min"]),
            "power_integral_W": float(_field_summary(candidate_fields, floor=floor)["power_integral_W"])
            - float(_field_summary(reference_fields, floor=floor)["power_integral_W"]),
        },
        "plot_png": str(out_png),
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
