"""Hybrid MAiNGO + CasADi workflows for the v6 active-segment model."""

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


def __getattr__(name: str):
    if name in __all__:
        from . import core

        return getattr(core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
