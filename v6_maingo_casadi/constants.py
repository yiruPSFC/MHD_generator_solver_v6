from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_core.local_algebraic_closure import (
    B_FIELD,
    E_CHARGE,
    H_P,
    K_B,
    M_E,
    SIGMA_EP,
)

_EPS = 1e-30


_TP_MIN = 1.0


_DELTA_MIN = 1e-12


_FION_MIN = 1e-12


_FION_MAX = 1.0 - 1e-12


_SAHA_K_MIN = 1e-100


_SAHA_K_MAX = 1e60


_SAHA_LOG_K_MIN = math.log(_SAHA_K_MIN)


_SAHA_LOG_K_MAX = math.log(_SAHA_K_MAX)


_SAHA_PREFAC = 2.0 * math.pi * M_E * K_B / (H_P * H_P)


_A_IN = 1.0


_G_HARD_MARGIN = 5e-7


_G_PENALTY_WEIGHT = 25.0


_G_PENALTY_SCALE = 1e-2


_DEFAULT_BASELINE_SUMMARY = (
    REPO_DIR
    / "v6_casadi_v2"
    / "outputs"
    / "continuation"
    / "baseline_release_from_v6_candidate_022_sigma_0p5"
    / "continuation_summary.json"
)


OBJECTIVE_PROFILE_LAB_POC_V2 = "lab_poc_v2"


OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION = "enthalpy_extraction"


OBJECTIVE_PROFILES = (
    OBJECTIVE_PROFILE_LAB_POC_V2,
    OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION,
)


WORKING_FLUID_PROFILE_ARGON_POTASSIUM = "ar_k"


WORKING_FLUID_PROFILE_HELIUM_CESIUM = "he_cs"


WORKING_FLUID_PROFILES = (
    WORKING_FLUID_PROFILE_ARGON_POTASSIUM,
    WORKING_FLUID_PROFILE_HELIUM_CESIUM,
)


_AMU_KG = 1.66053906660e-27
