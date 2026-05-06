from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .constants import B_FIELD, REPO_DIR, _A_IN, _EPS
from .geometry import SplineAreaDesign
from .numerics import _json_load
from .profiles import WorkingFluidProfile, _normalize_working_fluid_profile

@dataclass(frozen=True)
class InletDesign:
    n_p: float
    T_e: float
    T_p: float
    Z: float
    I_0: float
    dot_N: float
    v_in: float
    seed_fraction: float
    mach: float
    velikhov_margin: float
    A_in: float = _A_IN

    def to_dict(self) -> dict[str, float]:
        return {
            "n_p_in": float(self.n_p),
            "T_e_in": float(self.T_e),
            "T_p_in": float(self.T_p),
            "Z_in": float(self.Z),
            "J_x_in": float(self.I_0),
            "I_0": float(self.I_0),
            "dot_N": float(self.dot_N),
            "v_in": float(self.v_in),
            "seed_fraction": float(self.seed_fraction),
            "mach_in": float(self.mach),
            "velikhov_margin_in": float(self.velikhov_margin),
            "A_in": float(self.A_in),
        }


@dataclass(frozen=True)
class BaselineSeed:
    summary_path: Path
    warm_profile_npz_path: Path
    working_fluid: WorkingFluidProfile
    B: float
    L: float
    area_scale_m2: float
    schedule: list[dict[str, Any]]
    adaptive_bridge_count: int
    adaptive_bridge_max_count: int
    n_p_in_nominal: float
    T_e_in_nominal: float
    Z_in_nominal: float
    I_0_nominal: float
    seed_fraction_nominal: float
    inlet_windows: dict[str, dict[str, float]]
    area_design_windows: dict[str, dict[str, float]]
    area_design_nominal: SplineAreaDesign
    area_reference_x_norm: np.ndarray | None = None
    area_reference_factor: np.ndarray | None = None
    area_reference_sigma_logA: np.ndarray | None = None

    @classmethod
    def from_summary(cls, path: str | Path) -> "BaselineSeed":
        summary_path = Path(path).resolve()
        summary = _json_load(summary_path)
        source_alignment = dict(summary.get("source_alignment", {}) or {})
        if not source_alignment:
            raise ValueError("baseline summary is missing source_alignment.")
        warm_profile_npz_path = Path(str(source_alignment.get("warm_profile_npz", "")))
        if not warm_profile_npz_path.is_absolute():
            warm_profile_npz_path = (REPO_DIR / warm_profile_npz_path).resolve()
        aligned = dict(source_alignment.get("aligned_inlet_window", {}) or {})
        if not aligned:
            raise ValueError("baseline summary is missing aligned_inlet_window.")
        schedule = [dict(item) for item in list(summary.get("schedule", []) or [])]
        length = float(summary.get("L", 5.4))
        area_reference_x_norm = None
        area_reference_factor = None
        area_reference_sigma_logA = None
        area_reference_mode = str(source_alignment.get("area_reference_mode", "")).strip().lower()
        with np.load(warm_profile_npz_path) as warm_data:
            warm_x = np.asarray(warm_data["x"], dtype=float)
            warm_A = np.asarray(warm_data["A"], dtype=float)
            if area_reference_mode == "multiplicative":
                area_reference_x_norm = (warm_x - float(warm_x[0])) / max(float(warm_x[-1] - warm_x[0]), _EPS)
                if "area_reference_factor" in warm_data:
                    area_reference_factor = np.asarray(warm_data["area_reference_factor"], dtype=float)
                else:
                    area_reference_factor = warm_A / max(float(warm_A[0]), _EPS)
                if "area_reference_sigma_logA" in warm_data:
                    area_reference_sigma_logA = np.asarray(warm_data["area_reference_sigma_logA"], dtype=float)
                else:
                    area_reference_sigma_logA = np.gradient(np.log(np.maximum(area_reference_factor, _EPS)), warm_x)
                area_design = SplineAreaDesign(a1=0.0, a2=0.0, a3=0.0)
            else:
                area_design = SplineAreaDesign.project_from_profile(x=warm_x, A=warm_A)
        if schedule:
            sigma_limit = float(schedule[0].get("max_abs_dlogA_dx", np.inf))
            if math.isfinite(sigma_limit) and sigma_limit > 0.0:
                projected_profile = area_design.evaluate_profile(length=length, n_intervals=40)
                sigma_peak = float(np.max(np.abs(np.asarray(projected_profile["sigma_logA"], dtype=float))))
                if sigma_peak > sigma_limit:
                    scale = 0.98 * sigma_limit / max(sigma_peak, _EPS)
                    area_design = SplineAreaDesign(
                        a1=float(area_design.a1 * scale),
                        a2=float(area_design.a2 * scale),
                        a3=float(area_design.a3 * scale),
                    )
        area_window_payload = dict(source_alignment.get("aligned_area_window", {}) or {})
        if area_window_payload:
            area_design_windows = {
                key: {
                    "guess": float(area_window_payload[key]["guess"]),
                    "min": float(area_window_payload[key]["min"]),
                    "max": float(area_window_payload[key]["max"]),
                }
                for key in ("a1", "a2", "a3")
            }
        else:
            area_design_windows = {
                "a1": {
                    "guess": float(area_design.a1),
                    "min": SplineAreaDesign.lower_bound(),
                    "max": SplineAreaDesign.upper_bound(),
                },
                "a2": {
                    "guess": float(area_design.a2),
                    "min": SplineAreaDesign.lower_bound(),
                    "max": SplineAreaDesign.upper_bound(),
                },
                "a3": {
                    "guess": float(area_design.a3),
                    "min": SplineAreaDesign.lower_bound(),
                    "max": SplineAreaDesign.upper_bound(),
                },
            }
        return cls(
            summary_path=summary_path,
            warm_profile_npz_path=warm_profile_npz_path,
            working_fluid=_normalize_working_fluid_profile(
                summary.get("working_fluid_profile", summary.get("working_fluid"))
            ),
            B=float(summary.get("B", source_alignment.get("B_T", B_FIELD))),
            L=length,
            area_scale_m2=float(summary.get("area_scale_m2", source_alignment.get("area_scale_m2", _A_IN))),
            schedule=schedule,
            adaptive_bridge_count=int(summary.get("adaptive_bridge_count", 0)),
            adaptive_bridge_max_count=int(summary.get("adaptive_bridge_max_count", 0)),
            n_p_in_nominal=float(aligned["np_in"]["guess"]),
            T_e_in_nominal=float(aligned["te_in"]["guess"]),
            Z_in_nominal=float(aligned["z_in"]["guess"]),
            I_0_nominal=float(aligned["jx_in"]["guess"]),
            seed_fraction_nominal=float(aligned["seed_fraction"]["guess"]),
            inlet_windows={
                "n_p_in": {
                    "guess": float(aligned["np_in"]["guess"]),
                    "min": float(aligned["np_in"]["min"]),
                    "max": float(aligned["np_in"]["max"]),
                },
                "T_e_in": {
                    "guess": float(aligned["te_in"]["guess"]),
                    "min": float(aligned["te_in"]["min"]),
                    "max": float(aligned["te_in"]["max"]),
                },
                "Z_in": {
                    "guess": float(aligned["z_in"]["guess"]),
                    "min": float(aligned["z_in"]["min"]),
                    "max": float(aligned["z_in"]["max"]),
                },
                "I_0": {
                    "guess": float(aligned["jx_in"]["guess"]),
                    "min": float(aligned["jx_in"]["min"]),
                    "max": float(aligned["jx_in"]["max"]),
                },
                "seed_fraction": {
                    "guess": float(aligned["seed_fraction"]["guess"]),
                    "min": float(aligned["seed_fraction"]["min"]),
                    "max": float(aligned["seed_fraction"]["max"]),
                },
            },
            area_design_windows=area_design_windows,
            area_design_nominal=area_design,
            area_reference_x_norm=area_reference_x_norm,
            area_reference_factor=area_reference_factor,
            area_reference_sigma_logA=area_reference_sigma_logA,
        )

    def with_inlet_bound_factors(
        self,
        *,
        n_p_in_lower: float = 1.0,
        n_p_in_upper: float = 1.0,
        T_e_in_lower: float = 1.0,
        T_e_in_upper: float = 1.0,
        Z_in_lower: float = 1.0,
        Z_in_upper: float = 1.0,
        I_0_lower: float = 1.0,
        I_0_upper: float = 1.0,
        seed_fraction_lower: float = 1.0,
        seed_fraction_upper: float = 1.0,
    ) -> "BaselineSeed":
        lower_factor_map = {
            "n_p_in": float(n_p_in_lower),
            "T_e_in": float(T_e_in_lower),
            "Z_in": float(Z_in_lower),
            "I_0": float(I_0_lower),
            "seed_fraction": float(seed_fraction_lower),
        }
        upper_factor_map = {
            "n_p_in": float(n_p_in_upper),
            "T_e_in": float(T_e_in_upper),
            "Z_in": float(Z_in_upper),
            "I_0": float(I_0_upper),
            "seed_fraction": float(seed_fraction_upper),
        }
        for label, factor_map in (("lower", lower_factor_map), ("upper", upper_factor_map)):
            for key, factor in factor_map.items():
                if factor <= 0.0:
                    raise ValueError(f"{label} inlet bound factor for {key} must be positive, got {factor!r}.")
        new_windows: dict[str, dict[str, float]] = {}
        for key, window in self.inlet_windows.items():
            new_window = dict(window)
            new_window["min"] = float(window["min"]) * lower_factor_map[key]
            new_window["max"] = float(window["max"]) * upper_factor_map[key]
            if new_window["max"] <= float(new_window["min"]):
                raise ValueError(
                    f"scaled bounds for {key} must satisfy min < max: "
                    f"min={new_window['min']!r}, max={new_window['max']!r}."
                )
            new_windows[key] = new_window
        return replace(self, inlet_windows=new_windows)

    def with_inlet_upper_bound_factors(
        self,
        *,
        n_p_in: float = 1.0,
        T_e_in: float = 1.0,
        Z_in: float = 1.0,
        I_0: float = 1.0,
        seed_fraction: float = 1.0,
    ) -> "BaselineSeed":
        return self.with_inlet_bound_factors(
            n_p_in_upper=n_p_in,
            T_e_in_upper=T_e_in,
            Z_in_upper=Z_in,
            I_0_upper=I_0,
            seed_fraction_upper=seed_fraction,
        )

    def with_working_fluid_profile(self, profile: str | WorkingFluidProfile | None) -> "BaselineSeed":
        return replace(self, working_fluid=_normalize_working_fluid_profile(profile))

    def area_reference_kwargs(self) -> dict[str, np.ndarray | None]:
        return {
            "area_reference_x_norm": self.area_reference_x_norm,
            "area_reference_factor": self.area_reference_factor,
            "area_reference_sigma_logA": self.area_reference_sigma_logA,
        }

    def initial_point(self) -> list[float]:
        return [
            math.log(self.n_p_in_nominal),
            float(self.T_e_in_nominal),
            float(self.Z_in_nominal),
            float(self.I_0_nominal),
            math.log(self.seed_fraction_nominal),
            float(self.area_design_nominal.a1),
            float(self.area_design_nominal.a2),
            float(self.area_design_nominal.a3),
        ]

    def optimization_variable_bounds(self) -> list[tuple[float, float, str]]:
        windows = self.inlet_windows
        return [
            (
                math.log(float(windows["n_p_in"]["min"])),
                math.log(float(windows["n_p_in"]["max"])),
                "log_n_p_in",
            ),
            (
                float(windows["T_e_in"]["min"]),
                float(windows["T_e_in"]["max"]),
                "T_e_in",
            ),
            (
                float(windows["Z_in"]["min"]),
                float(windows["Z_in"]["max"]),
                "Z_in",
            ),
            (
                float(windows["I_0"]["min"]),
                float(windows["I_0"]["max"]),
                "I_0",
            ),
            (
                math.log(float(windows["seed_fraction"]["min"])),
                math.log(float(windows["seed_fraction"]["max"])),
                "log_seed_fraction",
            ),
            (
                float(windows.get("a1", self.area_design_windows["a1"])["min"]),
                float(windows.get("a1", self.area_design_windows["a1"])["max"]),
                "a1",
            ),
            (
                float(windows.get("a2", self.area_design_windows["a2"])["min"]),
                float(windows.get("a2", self.area_design_windows["a2"])["max"]),
                "a2",
            ),
            (
                float(windows.get("a3", self.area_design_windows["a3"])["min"]),
                float(windows.get("a3", self.area_design_windows["a3"])["max"]),
                "a3",
            ),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_path": str(self.summary_path),
            "warm_profile_npz_path": str(self.warm_profile_npz_path),
            "working_fluid": self.working_fluid.to_dict(),
            "B": float(self.B),
            "L": float(self.L),
            "area_scale_m2": float(self.area_scale_m2),
            "adaptive_bridge_count": int(self.adaptive_bridge_count),
            "adaptive_bridge_max_count": int(self.adaptive_bridge_max_count),
            "inlet_windows": self.inlet_windows,
            "area_design_windows": self.area_design_windows,
            "area_design_nominal": self.area_design_nominal.to_dict(),
            "area_reference": {
                "enabled": self.area_reference_x_norm is not None and self.area_reference_factor is not None,
                "factor_min": None
                if self.area_reference_factor is None
                else float(np.min(np.asarray(self.area_reference_factor, dtype=float))),
                "factor_max": None
                if self.area_reference_factor is None
                else float(np.max(np.asarray(self.area_reference_factor, dtype=float))),
            },
        }


@dataclass(frozen=True)
class CoarseProfileResult:
    decision_vector: dict[str, float]
    inlet_design: InletDesign
    area_design: SplineAreaDesign
    objective_score: float
    objective_to_minimize: float
    diagnostics: dict[str, Any]
    x: np.ndarray
    n_p: np.ndarray
    T_e: np.ndarray
    T_p: np.ndarray
    A: np.ndarray
    sigma_logA: np.ndarray
    v_p: np.ndarray
    n_e: np.ndarray
    beta: np.ndarray
    eta: np.ndarray
    Z: np.ndarray
    J_x: np.ndarray
    J_y: np.ndarray
    E_x: np.ndarray
    mach: np.ndarray
    velikhov_margin: np.ndarray
    value_terms: dict[str, Any]
    value_profile: dict[str, Any]

    def to_npz_payload(self) -> dict[str, np.ndarray]:
        sigma_interval = np.diff(np.log(np.maximum(np.asarray(self.A, dtype=float), _EPS))) / np.diff(
            np.asarray(self.x, dtype=float)
        )
        return {
            "x": np.asarray(self.x, dtype=float),
            "n_p": np.asarray(self.n_p, dtype=float),
            "T_e": np.asarray(self.T_e, dtype=float),
            "T_p": np.asarray(self.T_p, dtype=float),
            "A": np.asarray(self.A, dtype=float),
            "sigma_logA": np.asarray(sigma_interval, dtype=float),
            "v_p": np.asarray(self.v_p, dtype=float),
            "n_e": np.asarray(self.n_e, dtype=float),
            "beta": np.asarray(self.beta, dtype=float),
            "eta": np.asarray(self.eta, dtype=float),
            "Z": np.asarray(self.Z, dtype=float),
            "J_x": np.asarray(self.J_x, dtype=float),
            "J_y": np.asarray(self.J_y, dtype=float),
            "E_x": np.asarray(self.E_x, dtype=float),
            "mach": np.asarray(self.mach, dtype=float),
            "velikhov_margin": np.asarray(self.velikhov_margin, dtype=float),
            "seed_fraction": np.full_like(np.asarray(self.x, dtype=float), self.inlet_design.seed_fraction),
        }

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "decision_vector": self.decision_vector,
            "inlet_design": self.inlet_design.to_dict(),
            "area_design": self.area_design.to_dict(),
            "objective_score": float(self.objective_score),
            "objective_to_minimize": float(self.objective_to_minimize),
            "diagnostics": self.diagnostics,
            "value_terms": self.value_terms,
            "value_profile": self.value_profile,
        }


@dataclass(frozen=True)
class HybridRunResult:
    baseline_seed: BaselineSeed
    maingo_status: dict[str, Any]
    maingo_best: CoarseProfileResult
    handoff_bounds: dict[str, dict[str, float]]
    maingo_summary_path: Path
    maingo_best_profile_path: Path
    continuation_out_dir: Path
    continuation_summary: dict[str, Any]
    hybrid_summary_path: Path

    def to_dict(self) -> dict[str, Any]:
        continuation_final = dict(self.continuation_summary.get("final_trusted_inlet_design", {}) or {})
        continuation_summary_path = self.continuation_summary.get(
            "summary_path",
            str(self.continuation_out_dir / "continuation_summary.json"),
        )
        return {
            "objective_profile": self.maingo_best.diagnostics.get("objective_profile"),
            "working_fluid_profile": self.baseline_seed.working_fluid.to_dict(),
            "baseline_seed": self.baseline_seed.to_dict(),
            "maingo_status": self.maingo_status,
            "maingo_best": self.maingo_best.to_summary_dict(),
            "handoff_bounds": self.handoff_bounds,
            "continuation": {
                "out_dir": str(self.continuation_out_dir),
                "summary_path": None if continuation_summary_path is None else str(continuation_summary_path),
                "skipped": bool(self.continuation_summary.get("skipped", False)),
                "skip_reason": self.continuation_summary.get("reason"),
                "ok": bool(self.continuation_summary.get("ok", False)),
                "final_return_status": str(self.continuation_summary.get("final_return_status", "")),
                "final_objective_delta_Te_K": self.continuation_summary.get("final_objective_delta_Te_K"),
                "final_trusted_inlet_design": continuation_final,
            },
            "artifacts": {
                "maingo_summary_json": str(self.maingo_summary_path),
                "maingo_best_profile_npz": str(self.maingo_best_profile_path),
                "hybrid_summary_json": str(self.hybrid_summary_path),
                "continuation_out_dir": str(self.continuation_out_dir),
            },
        }
