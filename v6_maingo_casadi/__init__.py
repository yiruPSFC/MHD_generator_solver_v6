"""Hybrid MAiNGO + CasADi workflows for the v6 active-segment model."""

from .core import (
    BaselineSeed,
    HybridRunResult,
    InletDesign,
    OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION,
    OBJECTIVE_PROFILE_LAB_POC_V2,
    OBJECTIVE_PROFILES,
    SplineAreaDesign,
    WORKING_FLUID_PROFILE_ARGON_POTASSIUM,
    WORKING_FLUID_PROFILE_HELIUM_CESIUM,
    WORKING_FLUID_PROFILES,
    WorkingFluidProfile,
    evaluate_inlet_design_numeric,
    run_hybrid_maingo_casadi,
)

__all__ = [
    "BaselineSeed",
    "HybridRunResult",
    "InletDesign",
    "OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION",
    "OBJECTIVE_PROFILE_LAB_POC_V2",
    "OBJECTIVE_PROFILES",
    "SplineAreaDesign",
    "WORKING_FLUID_PROFILE_ARGON_POTASSIUM",
    "WORKING_FLUID_PROFILE_HELIUM_CESIUM",
    "WORKING_FLUID_PROFILES",
    "WorkingFluidProfile",
    "evaluate_inlet_design_numeric",
    "run_hybrid_maingo_casadi",
]
