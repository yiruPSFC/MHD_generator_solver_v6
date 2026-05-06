"""Compatibility facade for the split v6_maingo_casadi implementation.

New code should prefer the focused submodules in this package. This module
intentionally re-exports the historical symbols so existing scripts that
import from ``v6_maingo_casadi.core`` keep working.
"""
from __future__ import annotations

from .constants import (
    B_FIELD,
    E_CHARGE,
    H_P,
    K_B,
    M_E,
    REPO_DIR,
    SIGMA_EP,
    OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION,
    OBJECTIVE_PROFILE_LAB_POC_V2,
    OBJECTIVE_PROFILES,
    WORKING_FLUID_PROFILE_ARGON_POTASSIUM,
    WORKING_FLUID_PROFILE_HELIUM_CESIUM,
    WORKING_FLUID_PROFILES,
    _A_IN,
    _AMU_KG,
    _DEFAULT_BASELINE_SUMMARY,
    _DELTA_MIN,
    _EPS,
    _FION_MAX,
    _FION_MIN,
    _G_HARD_MARGIN,
    _G_PENALTY_SCALE,
    _G_PENALTY_WEIGHT,
    _SAHA_K_MAX,
    _SAHA_K_MIN,
    _SAHA_LOG_K_MAX,
    _SAHA_LOG_K_MIN,
    _SAHA_PREFAC,
    _TP_MIN,
)
from .profiles import (
    WorkingFluidProfile,
    _DEFAULT_WORKING_FLUID_PROFILE,
    _WORKING_FLUID_ALIASES,
    _WORKING_FLUID_PROFILE_MAP,
    _augment_value_terms_with_hall_diagnostics,
    _design_value_weights_lab_poc_v2_objective,
    _normalize_objective_profile,
    _normalize_working_fluid_profile,
    _objective_profile_name,
    _value_profile_dict,
)
from .numerics import (
    _clip_range,
    _floored_pos,
    _json_load,
    _max_op,
    _min_op,
    _ops_for_casadi,
    _ops_for_maingo,
    _ops_for_numeric,
    _reduce_min,
    _safe_float,
    _safe_pos,
    _safe_signed_denom,
    _velikhov_margin_penalty,
)
from .geometry import (
    SplineAreaDesign,
    _evaluate_area_design_nodes,
    _evaluate_area_design_samples,
    _sample_area_reference,
)
from .models import BaselineSeed, CoarseProfileResult, HybridRunResult, InletDesign
from .physics import (
    _closure_state,
    _design_score_generic,
    _df_dbeta,
    _df_dz,
    _dynamic_system_terms,
    _evaluate_midpoint_closures,
    _f_beta_z,
    _implicit_step_residuals,
    _inlet_design_generic,
    _inlet_enthalpy_flux_generic,
    _rk4_rollout_generic,
    _saha_terms,
    _state_rhs,
    evaluate_inlet_design_numeric,
)
from .casadi_evaluator import CasadiCoarseEvaluator, _make_casadi_rollout_function
from .implicit import (
    ImplicitResidualScales,
    ImplicitTrajectoryScaling,
    ImplicitTrajectoryVariables,
    _ImplicitVariableLayout,
    _build_coarse_result_from_state_trajectory,
    _build_implicit_reference,
    _build_implicit_step_rootfinder,
    _constant_implicit_reference,
    _implicit_reference_is_reasonable,
    _interpolate_decision_vector,
    _interpolate_implicit_variables,
    _project_implicit_trajectory,
    _resample_profile_result,
    _restore_feasible_implicit_solution,
    _scaled_interval,
)
from .maingo_models import (
    _MAiNGOHybridImplicitModelBase,
    _MAiNGOHybridModelBase,
    _import_maingopy,
    _retcode_name,
    _safe_solver_metric,
)
from .workflow import _handoff_bounds_from_best, _write_json, run_hybrid_maingo_casadi

__all__ = [name for name in globals() if not name.startswith("__")]
