"""Freidberg slide-derived reference case for area-only reduced tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...geometry import LogAreaSplineControl


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class FreidbergReferenceParameters:
    """Freidberg/Jeffrey reference values after the V6 marginal recovery.

    The inlet values come from the April 30, 2024 Freidberg slide deck.  The
    seed fraction and total Hall current below use the recovered marginal
    projection, so the inlet algebraic closure reproduces T_p ~= 429 K and
    Mach ~= 2.004 in the current model.
    """

    n_p_in_m3: float = 3.05e25
    T_e_in_K: float = 4420.0
    T_p_in_K: float = 429.0
    Z_in: float = 75.95499375780275
    seed_fraction: float = 8.985685931899876e-7
    slide_seed_fraction: float = 8.936065573770492e-7
    I_0_A: float = 1445.2521362624675
    slide_J_x0_A_m2: float = 3204.0
    recovered_J_x0_A_m2: float = 3233.2262556207324
    J_y0_A_m2: float = -6073.0
    B_T: float = 10.2
    area_scale_m2: float = 0.447
    length_m: float = 5.325180763867631
    slide_length_m: float = 5.4
    furnace_power_MW: float = 363.0
    steam_cycle_efficiency: float = 0.35
    reference_total_plant_power_MWe: float = 200.0
    reference_mhd_output_MWe: float = 112.230765690579
    reference_area_a1: float = 0.1734697727255258
    reference_area_a2: float = 0.37530533787502696
    reference_area_a3: float = 0.6163516395254951
    reference_profile_relpath: str = "v6_global_marginal/reference_recovery/profile_trimmed.npz"

    @property
    def area_control(self) -> LogAreaSplineControl:
        return LogAreaSplineControl(
            a1=float(self.reference_area_a1),
            a2=float(self.reference_area_a2),
            a3=float(self.reference_area_a3),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "reference_source": "Freidberg slide reference recovered by v6_global_marginal/reference_recovery",
            "reference_profile_npz": str(reference_profile_path()),
            "reference_profile_trimmed_to_total_plant_power": True,
            "slide_length_m": float(self.slide_length_m),
            "furnace_power_MW": float(self.furnace_power_MW),
            "steam_cycle_efficiency": float(self.steam_cycle_efficiency),
            "reference_total_plant_power_MWe": float(self.reference_total_plant_power_MWe),
            "reference_mhd_output_MWe": float(self.reference_mhd_output_MWe),
            "slide_seed_fraction": float(self.slide_seed_fraction),
            "seed_fraction_projection_residual": float(self.seed_fraction / self.slide_seed_fraction - 1.0),
            "slide_J_x0_A_m2": float(self.slide_J_x0_A_m2),
            "recovered_J_x0_A_m2": float(self.recovered_J_x0_A_m2),
            "control_scope": "area_only",
            "electron_transport": "e-Argon",
        }


@dataclass(frozen=True)
class FreidbergReferenceSeed:
    area_log_window_half_width: float = 0.35
    schedule_n_intervals: int = 80


FREIDBERG_REFERENCE = FreidbergReferenceParameters()
FREIDBERG_REFERENCE_MODEL_SEED = FreidbergReferenceSeed()


def reference_profile_path() -> Path:
    return REPO_ROOT / FREIDBERG_REFERENCE.reference_profile_relpath


def load_reference_profile(path: str | Path | None = None) -> dict[str, np.ndarray]:
    profile_path = reference_profile_path() if path is None else Path(path)
    with np.load(profile_path) as data:
        profile = {name: np.asarray(data[name], dtype=float) for name in data.files}
    if "sigma_logA" not in profile:
        x = np.asarray(profile["x"], dtype=float).reshape(-1)
        A = np.asarray(profile["A"], dtype=float).reshape(-1)
        profile["sigma_logA"] = np.gradient(np.log(np.maximum(A / max(float(A[0]), 1e-300), 1e-300)), x)
    if "x_norm" not in profile:
        x = np.asarray(profile["x"], dtype=float).reshape(-1)
        profile["x_norm"] = (x - float(x[0])) / max(float(x[-1] - x[0]), 1e-300)
    return profile
