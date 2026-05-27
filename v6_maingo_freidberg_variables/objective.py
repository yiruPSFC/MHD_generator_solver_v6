from __future__ import annotations


def primary_enthalpy_extraction_percent(*, mhd_output_MW: float, inlet_primary_enthalpy_MW: float) -> float:
    return 100.0 * float(mhd_output_MW) / max(float(inlet_primary_enthalpy_MW), 1e-300)


def conservative_power_objective_MW(*, H_in: float, H_out: float, inlet_area_m2: float) -> float:
    return (float(H_in) - float(H_out)) * float(inlet_area_m2) / 1e6
