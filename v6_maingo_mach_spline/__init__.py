from __future__ import annotations

from .geometry import MachSplineDesign

__all__ = [
    "MachSplineDesign",
    "mach_closure_from_state",
    "reconstruct_points_from_mach",
]


def __getattr__(name: str):
    if name in {"mach_closure_from_state", "reconstruct_points_from_mach"}:
        from .closure import mach_closure_from_state, reconstruct_points_from_mach

        return {
            "mach_closure_from_state": mach_closure_from_state,
            "reconstruct_points_from_mach": reconstruct_points_from_mach,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
