from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from v6_global_marginal.global_postprocess_v6 import DesignValueWeights, evaluate_design_value

from .constants import (
    E_CHARGE,
    SIGMA_EP,
    _AMU_KG,
    OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION,
    OBJECTIVE_PROFILE_LAB_POC_V2,
    OBJECTIVE_PROFILES,
    WORKING_FLUID_PROFILE_ARGON_POTASSIUM,
    WORKING_FLUID_PROFILE_HELIUM_CESIUM,
    WORKING_FLUID_PROFILES,
)

@dataclass(frozen=True)
class WorkingFluidProfile:
    key: str
    working_gas: str
    seed_species: str
    heavy_particle_mass_amu: float
    seed_ionization_energy_eV: float
    sigma_ep: float
    sigma_ep_note: str

    @property
    def heavy_particle_mass_kg(self) -> float:
        return float(self.heavy_particle_mass_amu) * _AMU_KG

    @property
    def seed_ionization_energy_J(self) -> float:
        return float(self.seed_ionization_energy_eV) * E_CHARGE

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": str(self.key),
            "working_gas": str(self.working_gas),
            "seed_species": str(self.seed_species),
            "heavy_particle_mass_amu": float(self.heavy_particle_mass_amu),
            "heavy_particle_mass_kg": float(self.heavy_particle_mass_kg),
            "seed_ionization_energy_eV": float(self.seed_ionization_energy_eV),
            "seed_ionization_energy_J": float(self.seed_ionization_energy_J),
            "sigma_ep": float(self.sigma_ep),
            "sigma_ep_note": str(self.sigma_ep_note),
        }


_WORKING_FLUID_PROFILE_MAP = {
    WORKING_FLUID_PROFILE_ARGON_POTASSIUM: WorkingFluidProfile(
        key=WORKING_FLUID_PROFILE_ARGON_POTASSIUM,
        working_gas="Ar",
        seed_species="K",
        heavy_particle_mass_amu=39.948,
        seed_ionization_energy_eV=4.3407,
        sigma_ep=SIGMA_EP,
        sigma_ep_note="original v6 closure value",
    ),
    WORKING_FLUID_PROFILE_HELIUM_CESIUM: WorkingFluidProfile(
        key=WORKING_FLUID_PROFILE_HELIUM_CESIUM,
        working_gas="He",
        seed_species="Cs",
        heavy_particle_mass_amu=4.002602,
        seed_ionization_energy_eV=3.89390572743,
        sigma_ep=SIGMA_EP,
        sigma_ep_note=(
            "first-pass He/Cs run: mass and seed ionization energy updated; "
            "electron-heavy-particle momentum cross-section still uses original v6 closure value"
        ),
    ),
}


_WORKING_FLUID_ALIASES = {
    "default": WORKING_FLUID_PROFILE_ARGON_POTASSIUM,
    "ar": WORKING_FLUID_PROFILE_ARGON_POTASSIUM,
    "argon": WORKING_FLUID_PROFILE_ARGON_POTASSIUM,
    "argon_potassium": WORKING_FLUID_PROFILE_ARGON_POTASSIUM,
    "ar/k": WORKING_FLUID_PROFILE_ARGON_POTASSIUM,
    "k": WORKING_FLUID_PROFILE_ARGON_POTASSIUM,
    "he": WORKING_FLUID_PROFILE_HELIUM_CESIUM,
    "helium": WORKING_FLUID_PROFILE_HELIUM_CESIUM,
    "helium_cesium": WORKING_FLUID_PROFILE_HELIUM_CESIUM,
    "he/cs": WORKING_FLUID_PROFILE_HELIUM_CESIUM,
    "cs": WORKING_FLUID_PROFILE_HELIUM_CESIUM,
}


_DEFAULT_WORKING_FLUID_PROFILE = _WORKING_FLUID_PROFILE_MAP[WORKING_FLUID_PROFILE_ARGON_POTASSIUM]


def _design_value_weights_lab_poc_v2_objective() -> DesignValueWeights:
    return DesignValueWeights(
        outlet_delta_te_per_kK=1.0,
        outlet_delta_ratio=0.35,
        outlet_f_ion=0.35,
        outlet_mhd_output_per_100MWe=0.0,
        inlet_delta_ratio_penalty=0.75,
        inlet_mach_penalty=0.10,
        magnetic_field_T_penalty=0.0,
        device_length_per_5m_penalty=0.0,
    )


def _normalize_objective_profile(objective_profile: str) -> str:
    profile = str(objective_profile or OBJECTIVE_PROFILE_LAB_POC_V2).strip().lower()
    aliases = {
        "lab": OBJECTIVE_PROFILE_LAB_POC_V2,
        "lab_poc": OBJECTIVE_PROFILE_LAB_POC_V2,
        "lab_poc_v2_objective": OBJECTIVE_PROFILE_LAB_POC_V2,
        "enthalpy": OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION,
        "enthalpy_extraction_percent": OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION,
    }
    profile = aliases.get(profile, profile)
    if profile not in OBJECTIVE_PROFILES:
        raise ValueError(f"unknown objective_profile={objective_profile!r}; expected one of {OBJECTIVE_PROFILES!r}")
    return profile


def _normalize_working_fluid_profile(profile: str | WorkingFluidProfile | None) -> WorkingFluidProfile:
    if isinstance(profile, WorkingFluidProfile):
        return profile
    if isinstance(profile, dict):
        if "key" in profile:
            return _normalize_working_fluid_profile(str(profile["key"]))
        if "working_fluid" in profile:
            return _normalize_working_fluid_profile(str(profile["working_fluid"]))
    key = str(profile or WORKING_FLUID_PROFILE_ARGON_POTASSIUM).strip().lower()
    key = _WORKING_FLUID_ALIASES.get(key, key)
    if key not in _WORKING_FLUID_PROFILE_MAP:
        raise ValueError(f"unknown working_fluid_profile={profile!r}; expected one of {WORKING_FLUID_PROFILES!r}")
    return _WORKING_FLUID_PROFILE_MAP[key]


def _objective_profile_name(objective_profile: str) -> str:
    profile = _normalize_objective_profile(objective_profile)
    if profile == OBJECTIVE_PROFILE_LAB_POC_V2:
        return "lab_poc_v2_objective"
    return "enthalpy_extraction_objective"


def _value_profile_dict(value_terms, *, objective_profile: str):
    profile = _normalize_objective_profile(objective_profile)
    if profile == OBJECTIVE_PROFILE_LAB_POC_V2:
        return evaluate_design_value(
            value_terms,
            weights=_design_value_weights_lab_poc_v2_objective(),
            profile_name=_objective_profile_name(profile),
        ).to_dict()
    score = float(value_terms.outlet_enthalpy_extraction_percent)
    return {
        "profile_name": _objective_profile_name(profile),
        "total_score": score,
        "score_units": "percent_of_inlet_stagnation_enthalpy_flux",
        "contributions": {
            "reward_outlet_enthalpy_extraction_percent": score,
        },
        "terms": value_terms.to_dict(),
        "weights": {
            "outlet_enthalpy_extraction_percent": 1.0,
        },
    }


def _augment_value_terms_with_hall_diagnostics(
    value_terms_dict: dict[str, Any],
    *,
    x: np.ndarray,
    E_x: np.ndarray,
    I_0: float,
) -> dict[str, Any]:
    hall_voltage = float(-np.trapezoid(np.asarray(E_x, dtype=float), np.asarray(x, dtype=float)))
    value_terms_dict["hall_voltage_V"] = hall_voltage
    value_terms_dict["hall_voltage_abs_V"] = float(abs(hall_voltage))
    value_terms_dict["power_from_hall_voltage_MWe"] = float(float(I_0) * hall_voltage / 1e6)
    return value_terms_dict
