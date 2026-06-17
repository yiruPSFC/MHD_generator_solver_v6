from __future__ import annotations

import math

B_FIELD = 0.02
E_CHARGE = 1.602176634e-19
K_B = 1.380649e-23
H_P = 6.62607015e-34
M_E = 9.10938356e-31
SIGMA_EP = 3.942573033087758e-21

_EPS = 1.0e-30
_TP_MIN = 1.0
_DELTA_MIN = 1.0e-12
_FION_MIN = 1.0e-12
_FION_MAX = 1.0 - 1.0e-12
_SAHA_K_MIN = 1.0e-100
_SAHA_K_MAX = 1.0e60
_SAHA_LOG_K_MIN = math.log(_SAHA_K_MIN)
_SAHA_LOG_K_MAX = math.log(_SAHA_K_MAX)
_SAHA_PREFAC = 2.0 * math.pi * M_E * K_B / (H_P * H_P)
