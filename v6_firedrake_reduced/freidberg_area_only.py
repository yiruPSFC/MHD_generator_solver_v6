from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .cases.freidberg_reference import FREIDBERG_REFERENCE, load_reference_profile
from .design import DESIGN_VARIABLE_NAMES, CaseConfig, DesignBounds, DesignVector, load_case_config
from .geometry import LogAreaSplineControl
from .objective import evaluate_profile_metrics


AREA_CONTROL_NAMES = ("a1", "a2", "a3")


def with_area_only_bounds(config: CaseConfig, *, area_half_width: float | None = None) -> CaseConfig:
    """Return a config where only a1/a2/a3 have nonzero search width."""

    half_width = (
        float(config.metadata.get("area_log_window_half_width", 0.35))
        if area_half_width is None
        else float(area_half_width)
    )
    values = config.design.as_array()
    lower = values.copy()
    upper = values.copy()
    for name in AREA_CONTROL_NAMES:
        idx = DESIGN_VARIABLE_NAMES.index(name)
        lower[idx] = max(float(values[idx] - half_width), LogAreaSplineControl.lower_bound())
        upper[idx] = min(float(values[idx] + half_width), LogAreaSplineControl.upper_bound())
    metadata = {
        **config.metadata,
        "control_scope": "area_only",
        "area_log_window_half_width": half_width,
        "frozen_control_names": [name for name in DESIGN_VARIABLE_NAMES if name not in AREA_CONTROL_NAMES],
    }
    return replace(
        config,
        bounds=DesignBounds(
            lower=DesignVector.from_array(lower),
            upper=DesignVector.from_array(upper),
        ),
        metadata=metadata,
    )


def load_freidberg_area_only_config(
    *,
    n_intervals: int | None = None,
    area_half_width: float | None = None,
    objective_profile: str = "enthalpy_extraction",
) -> CaseConfig:
    return with_area_only_bounds(
        load_case_config(
            case="freidberg_reference",
            objective_profile=objective_profile,
            n_intervals=n_intervals,
        ),
        area_half_width=area_half_width,
    )


def freidberg_net_power_MWe(mhd_output_power_W: float, *, config: CaseConfig) -> float:
    furnace_power_MW = float(config.metadata.get("furnace_power_MW", FREIDBERG_REFERENCE.furnace_power_MW))
    steam_efficiency = float(config.metadata.get("steam_cycle_efficiency", FREIDBERG_REFERENCE.steam_cycle_efficiency))
    mhd_power_MW = float(mhd_output_power_W) / 1.0e6
    return float(mhd_power_MW + steam_efficiency * (furnace_power_MW - mhd_power_MW))


def reference_profile_metrics(config: CaseConfig, *, profile_path: str | None = None) -> dict[str, Any]:
    profile = load_reference_profile(profile_path)
    metrics = evaluate_profile_metrics(profile=profile, design=config.design, config=config)
    payload = metrics.to_dict()
    payload["estimated_total_plant_power_MWe"] = freidberg_net_power_MWe(
        payload["mhd_output_power_W"],
        config=config,
    )
    payload["profile_npz"] = str(profile_path or config.metadata.get("reference_profile_npz", ""))
    payload["area_control"] = config.design.area_control.to_dict()
    payload["area_ratio_outlet_to_inlet"] = float(
        np.asarray(profile["A"], dtype=float)[-1] / max(float(np.asarray(profile["A"], dtype=float)[0]), 1e-300)
    )
    return payload


def compare_candidate_to_reference(*, reference: dict[str, Any], candidate: dict[str, Any], config: CaseConfig) -> dict[str, Any]:
    reference_power = float(reference["mhd_output_power_W"])
    candidate_power = float(candidate["mhd_output_power_W"])
    reference_extraction = float(reference["raw_enthalpy_extraction_percent"])
    candidate_extraction = float(candidate["raw_enthalpy_extraction_percent"])
    reference_net = float(reference.get("estimated_total_plant_power_MWe", freidberg_net_power_MWe(reference_power, config=config)))
    candidate_net = freidberg_net_power_MWe(candidate_power, config=config)
    return {
        "reference": reference,
        "candidate": {
            **candidate,
            "estimated_total_plant_power_MWe": candidate_net,
        },
        "improvement": {
            "mhd_output_power_W": candidate_power - reference_power,
            "mhd_output_power_MWe": (candidate_power - reference_power) / 1.0e6,
            "raw_enthalpy_extraction_percent": candidate_extraction - reference_extraction,
            "estimated_total_plant_power_MWe": candidate_net - reference_net,
        },
        "dominates_reference_on_primary_metrics": bool(
            candidate_power > reference_power and candidate_extraction > reference_extraction
        ),
    }
