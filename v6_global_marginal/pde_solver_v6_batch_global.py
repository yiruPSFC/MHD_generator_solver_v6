from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    from numba import njit, prange
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

    def prange(*args):
        return range(*args)


from v6_core.local_algebraic_closure import (
    B_FIELD,
    E_CHARGE,
    E_I,
    K_B,
    M_P,
    SIGMA_EP,
    beta_from_np_te,
    ne_from_np_te,
    saha_K,
)
from v6_global_marginal.local_algebraic_closure_global import (
    compute_currents_fields_global,
    local_closure_global_with_partials,
)
from v6_batch.pde_solver_v6_batch import (
    EVENT_INLET_ERROR,
    EVENT_INVALID_STATE,
    EVENT_MACH_HIGH,
    EVENT_MACH_LOW,
    EVENT_NONE,
    _as_1d_array,
    _crossed_threshold,
    _inlet_velocity_from_eq8_prime,
    event_name_from_code as _base_event_name_from_code,
)


_EPS = 1e-300
_TMIN = 1.0

EVENT_NO_MARGINAL_SEED = 5
EVENT_SEED_OUT_OF_RANGE = 6


@dataclass
class BatchForwardResultV6Global:
    x: np.ndarray
    valid_points: np.ndarray
    success: np.ndarray
    reached_end: np.ndarray
    event_code: np.ndarray
    seed_fraction: np.ndarray
    n_p: np.ndarray
    T_e: np.ndarray
    T_p: np.ndarray
    A: np.ndarray
    v_p: np.ndarray
    n_e: np.ndarray
    beta: np.ndarray
    eta: np.ndarray
    Z: np.ndarray
    J_x: np.ndarray
    J_y: np.ndarray
    E_x: np.ndarray
    mach: np.ndarray
    velikhov_margin: np.ndarray
    step_size: float

    def event_names(self) -> List[Optional[str]]:
        return [event_name_from_code(int(code)) for code in self.event_code]


@dataclass
class BatchTerminalResultV6Global:
    x_end: np.ndarray
    valid_points: np.ndarray
    success: np.ndarray
    reached_end: np.ndarray
    event_code: np.ndarray
    seed_fraction: np.ndarray
    n_p: np.ndarray
    T_e: np.ndarray
    T_p: np.ndarray
    A: np.ndarray
    v_p: np.ndarray
    n_e: np.ndarray
    beta: np.ndarray
    eta: np.ndarray
    Z: np.ndarray
    J_x: np.ndarray
    J_y: np.ndarray
    E_x: np.ndarray
    mach: np.ndarray
    velikhov_margin: np.ndarray
    step_size: float

    def event_names(self) -> List[Optional[str]]:
        return [event_name_from_code(int(code)) for code in self.event_code]


@dataclass
class BatchInletMetricsV6Global:
    success: np.ndarray
    event_code: np.ndarray
    seed_fraction: np.ndarray
    dn_dx: np.ndarray
    dTe_dx: np.ndarray
    dA_dx: np.ndarray
    dTe_rel_grad: np.ndarray
    dA_rel_grad: np.ndarray
    n_p: np.ndarray
    T_e: np.ndarray
    T_p: np.ndarray
    A: np.ndarray
    v_p: np.ndarray
    n_e: np.ndarray
    beta: np.ndarray
    eta: np.ndarray
    Z: np.ndarray
    J_x: np.ndarray
    J_y: np.ndarray
    E_x: np.ndarray
    mach: np.ndarray
    inlet_velikhov_margin: np.ndarray

    def event_names(self) -> List[Optional[str]]:
        return [event_name_from_code(int(code)) for code in self.event_code]


def event_name_from_code(code: int) -> Optional[str]:
    if code == EVENT_NO_MARGINAL_SEED:
        return "no_marginal_seed"
    if code == EVENT_SEED_OUT_OF_RANGE:
        return "seed_out_of_range"
    return _base_event_name_from_code(code)


@njit(cache=True)
def _safe_pos(x: float, floor: float = _EPS) -> float:
    return x if x > floor else floor


@njit(cache=True)
def _safe_signed(x: float, floor: float = _EPS) -> float:
    if abs(x) >= floor:
        return x
    return floor if x >= 0.0 else -floor


@njit(cache=True)
def _invalid_eval_state():
    nan = math.nan
    return (
        nan, nan, nan,
        nan, nan, nan, nan, nan, nan,
        nan, nan, nan,
        nan, nan, EVENT_INVALID_STATE,
    )


@njit(cache=True)
def project_seed_fraction_to_marginal_inlet(
    n_p_in: float,
    T_p_in: float,
    T_e_in: float,
    B: float,
):
    if (
        (not math.isfinite(n_p_in))
        or (not math.isfinite(T_p_in))
        or (not math.isfinite(T_e_in))
        or n_p_in <= 0.0
        or T_p_in <= 0.0
        or T_e_in <= T_p_in
    ):
        return EVENT_INLET_ERROR, math.nan

    delta = T_e_in / T_p_in - 1.0
    if delta <= 0.0 or (not math.isfinite(delta)):
        return EVENT_INLET_ERROR, math.nan

    beta = beta_from_np_te(n_p_in, T_e_in, B=B, sigma_ep=SIGMA_EP)
    if (not math.isfinite(beta)) or beta < 0.0:
        return EVENT_INLET_ERROR, math.nan

    U = 2.0 + 1.0 / delta
    W = 1.0 + 1.0 / delta
    if U <= 0.0 or W <= 0.0:
        return EVENT_INLET_ERROR, math.nan

    rad = 1.0 + W * beta * beta / U
    if rad < 0.0 or (not math.isfinite(rad)):
        return EVENT_NO_MARGINAL_SEED, math.nan
    alpha_star = (math.sqrt(rad) - 1.0) / (2.0 * W)

    C = K_B * T_e_in / (2.0 * E_I)
    if C <= 0.0 or (not math.isfinite(C)):
        return EVENT_INLET_ERROR, math.nan
    r = alpha_star / C
    if r <= 2.0:
        return EVENT_NO_MARGINAL_SEED, math.nan

    denom = r - 1.0
    if abs(denom) < _EPS:
        return EVENT_NO_MARGINAL_SEED, math.nan
    f_I = (r - 2.0) / denom
    if (not math.isfinite(f_I)) or f_I <= 0.0 or f_I >= 1.0:
        return EVENT_NO_MARGINAL_SEED, math.nan

    K = saha_K(T_e_in)
    if K <= 0.0 or (not math.isfinite(K)):
        return EVENT_NO_MARGINAL_SEED, math.nan

    seed_fraction = K * (1.0 - f_I) / (n_p_in * f_I * f_I)
    if (not math.isfinite(seed_fraction)) or seed_fraction <= 0.0:
        return EVENT_NO_MARGINAL_SEED, math.nan
    if seed_fraction >= 1.0:
        return EVENT_SEED_OUT_OF_RANGE, seed_fraction

    return EVENT_NONE, seed_fraction


@njit(cache=True)
def _solve_3x3(
    a11: float, a12: float, a13: float,
    a21: float, a22: float, a23: float,
    a31: float, a32: float, a33: float,
    b1: float, b2: float, b3: float,
):
    m11, m12, m13, r1 = a11, a12, a13, b1
    m21, m22, m23, r2 = a21, a22, a23, b2
    m31, m32, m33, r3 = a31, a32, a33, b3

    # Partial pivoting for the first column.
    p1 = abs(m11)
    p2 = abs(m21)
    p3 = abs(m31)
    if p2 > p1 and p2 >= p3:
        m11, m12, m13, r1, m21, m22, m23, r2 = m21, m22, m23, r2, m11, m12, m13, r1
        p1 = abs(m11)
    elif p3 > p1 and p3 > p2:
        m11, m12, m13, r1, m31, m32, m33, r3 = m31, m32, m33, r3, m11, m12, m13, r1
        p1 = abs(m11)

    if p1 < _EPS or (not math.isfinite(m11)):
        return 0.0, 0.0, 0.0, False

    f21 = m21 / m11
    f31 = m31 / m11
    m21 = 0.0
    m22 = m22 - f21 * m12
    m23 = m23 - f21 * m13
    r2 = r2 - f21 * r1
    m31 = 0.0
    m32 = m32 - f31 * m12
    m33 = m33 - f31 * m13
    r3 = r3 - f31 * r1

    # Partial pivoting for the second column.
    if abs(m32) > abs(m22):
        m21, m22, m23, r2, m31, m32, m33, r3 = m31, m32, m33, r3, m21, m22, m23, r2

    if abs(m22) < _EPS or (not math.isfinite(m22)):
        return 0.0, 0.0, 0.0, False

    f32 = m32 / m22
    m32 = 0.0
    m33 = m33 - f32 * m23
    r3 = r3 - f32 * r2

    if abs(m33) < _EPS or (not math.isfinite(m33)):
        return 0.0, 0.0, 0.0, False

    x3 = r3 / m33
    x2 = (r2 - m23 * x3) / m22
    x1 = (r1 - m12 * x2 - m13 * x3) / m11
    ok = (
        math.isfinite(x1)
        and math.isfinite(x2)
        and math.isfinite(x3)
    )
    return x1, x2, x3, ok


@njit(cache=True)
def _prepare_inlet_constants_global(
    n_p_in: float,
    Z_in: float,
    T_p_in: float,
    T_e_in: float,
    A_in: float,
    B: float,
):
    if A_in <= 0.0:
        return EVENT_INLET_ERROR, 0.0, 0.0, math.nan

    seed_status, seed_fraction = project_seed_fraction_to_marginal_inlet(
        n_p_in=n_p_in,
        T_p_in=T_p_in,
        T_e_in=T_e_in,
        B=B,
    )
    if seed_status != EVENT_NONE:
        return seed_status, 0.0, 0.0, seed_fraction

    v_in, inlet_status = _inlet_velocity_from_eq8_prime(T_e_in, T_p_in, n_p_in, Z_in, B)
    if inlet_status != EVENT_NONE:
        return inlet_status, 0.0, 0.0, seed_fraction

    n_e_in = ne_from_np_te(n_p_in, T_e_in, seed_fraction)
    beta_in = beta_from_np_te(n_p_in, T_e_in, B=B, sigma_ep=SIGMA_EP)
    b2 = beta_in * beta_in
    den = _safe_signed(b2 + 1.0 + Z_in)
    J_x_in = b2 / den * E_CHARGE * n_e_in * v_in
    dot_N = n_p_in * v_in * A_in
    I_0 = J_x_in * A_in
    return EVENT_NONE, dot_N, I_0, seed_fraction


@njit(cache=True)
def _evaluate_state_global(
    n_p: float,
    T_e: float,
    A: float,
    dot_N: float,
    I_0: float,
    seed_fraction: float,
    B: float,
    sigma_ep: float,
):
    if (
        (not math.isfinite(n_p))
        or (not math.isfinite(T_e))
        or (not math.isfinite(A))
        or n_p <= 0.0
        or T_e <= 0.0
        or A <= 0.0
    ):
        return _invalid_eval_state()

    vals = local_closure_global_with_partials(
        n_p=n_p,
        T_e=T_e,
        A=A,
        dot_N=dot_N,
        I_0=I_0,
        seed_fraction=seed_fraction,
        B=B,
        sigma_ep=sigma_ep,
    )

    v_p = vals[0]
    n_e = vals[1]
    beta = vals[2]
    eta = vals[3]
    Z = vals[4]
    T_p = vals[5]
    dTp_dTe = vals[6]
    dTp_dnp = vals[7]
    dTp_dA = vals[8]
    G = vals[13]
    G_np = vals[14]
    G_Te = vals[15]
    G_A = vals[16]

    closure_ok = (
        math.isfinite(v_p)
        and math.isfinite(n_e)
        and math.isfinite(beta)
        and math.isfinite(eta)
        and math.isfinite(Z)
        and math.isfinite(T_p)
        and math.isfinite(dTp_dTe)
        and math.isfinite(dTp_dnp)
        and math.isfinite(dTp_dA)
        and math.isfinite(G)
        and math.isfinite(G_np)
        and math.isfinite(G_Te)
        and math.isfinite(G_A)
    )
    if (not closure_ok) or T_p <= 0.0 or v_p <= 0.0 or n_e < 0.0:
        return _invalid_eval_state()

    J_x, J_y, E_x, _ = compute_currents_fields_global(v_p, n_e, beta, eta, Z, I_0, A)
    nu_E = eta * 2.0 * E_CHARGE * E_CHARGE * n_e / M_P

    M11 = (-M_P * v_p * v_p + K_B * T_p) + K_B * n_p * dTp_dnp
    M12 = K_B * n_p * dTp_dTe
    M13 = K_B * n_p * dTp_dA - M_P * n_p * v_p * v_p / _safe_pos(A)

    E11 = -T_p + 1.5 * n_p * dTp_dnp
    E12 = 1.5 * n_p * dTp_dTe
    E13 = 1.5 * n_p * dTp_dA

    rhs_m = J_y * B
    rhs_e = 1.5 * nu_E * n_e * (T_e - T_p) / _safe_pos(v_p)

    dn_dx, dTe_dx, dA_dx, ok = _solve_3x3(
        M11, M12, M13,
        E11, E12, E13,
        G_np, G_Te, G_A,
        rhs_m, rhs_e, 0.0,
    )
    if not ok:
        return (
            0.0, 0.0, 0.0,
            T_p, v_p, n_e, beta, eta, Z,
            J_x, J_y, E_x,
            0.0, G, EVENT_INVALID_STATE,
        )

    c_s = math.sqrt((5.0 / 3.0) * K_B * _safe_pos(T_p) / M_P)
    mach = v_p / _safe_pos(c_s)

    finite_ok = (
        math.isfinite(dn_dx)
        and math.isfinite(dTe_dx)
        and math.isfinite(dA_dx)
        and math.isfinite(T_p)
        and math.isfinite(v_p)
        and math.isfinite(n_e)
        and math.isfinite(beta)
        and math.isfinite(eta)
        and math.isfinite(Z)
        and math.isfinite(J_x)
        and math.isfinite(J_y)
        and math.isfinite(E_x)
        and math.isfinite(mach)
        and math.isfinite(G)
    )
    status = EVENT_NONE
    if (not finite_ok) or T_p <= 0.0 or v_p <= 0.0 or n_e < 0.0 or A <= 0.0:
        status = EVENT_INVALID_STATE

    return (
        dn_dx, dTe_dx, dA_dx,
        T_p, v_p, n_e, beta, eta, Z,
        J_x, J_y, E_x,
        mach, G, status,
    )


@njit(cache=True)
def _rk4_step_global(
    n_p: float,
    T_e: float,
    A: float,
    dx: float,
    dot_N: float,
    I_0: float,
    seed_fraction: float,
    B: float,
    sigma_ep: float,
):
    k1 = _evaluate_state_global(n_p, T_e, A, dot_N, I_0, seed_fraction, B, sigma_ep)
    if k1[14] != EVENT_NONE:
        return n_p, T_e, A, EVENT_INVALID_STATE

    y2_np = n_p + 0.5 * dx * k1[0]
    y2_te = T_e + 0.5 * dx * k1[1]
    y2_A = A + 0.5 * dx * k1[2]
    k2 = _evaluate_state_global(y2_np, y2_te, y2_A, dot_N, I_0, seed_fraction, B, sigma_ep)
    if k2[14] != EVENT_NONE:
        return n_p, T_e, A, EVENT_INVALID_STATE

    y3_np = n_p + 0.5 * dx * k2[0]
    y3_te = T_e + 0.5 * dx * k2[1]
    y3_A = A + 0.5 * dx * k2[2]
    k3 = _evaluate_state_global(y3_np, y3_te, y3_A, dot_N, I_0, seed_fraction, B, sigma_ep)
    if k3[14] != EVENT_NONE:
        return n_p, T_e, A, EVENT_INVALID_STATE

    y4_np = n_p + dx * k3[0]
    y4_te = T_e + dx * k3[1]
    y4_A = A + dx * k3[2]
    k4 = _evaluate_state_global(y4_np, y4_te, y4_A, dot_N, I_0, seed_fraction, B, sigma_ep)
    if k4[14] != EVENT_NONE:
        return n_p, T_e, A, EVENT_INVALID_STATE

    n_next = n_p + dx / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
    te_next = T_e + dx / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
    A_next = A + dx / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
    if (not math.isfinite(n_next)) or (not math.isfinite(te_next)) or (not math.isfinite(A_next)):
        return n_p, T_e, A, EVENT_INVALID_STATE
    return n_next, te_next, A_next, EVENT_NONE


@njit(cache=True)
def _solve_one_final_global(
    n_p_in: float,
    Z_in: float,
    T_p_in: float,
    T_e_in: float,
    A_in: float,
    B: float,
    dx: float,
    n_steps: int,
    mach_low: float,
    mach_high: float,
):
    success = False
    reached_end = False
    event_code = EVENT_INLET_ERROR
    valid_points = 0
    x_end = 0.0
    seed_fraction_final = np.nan

    n_p_final = np.nan
    T_e_final = np.nan
    T_p_final = np.nan
    A_final = np.nan
    v_p_final = np.nan
    n_e_final = np.nan
    beta_final = np.nan
    eta_final = np.nan
    Z_final = np.nan
    J_x_final = np.nan
    J_y_final = np.nan
    E_x_final = np.nan
    mach_final = np.nan
    G_final = np.nan

    inlet_status, dot_N, I_0, seed_fraction = _prepare_inlet_constants_global(
        n_p_in,
        Z_in,
        T_p_in,
        T_e_in,
        A_in,
        B,
    )
    if inlet_status != EVENT_NONE:
        return (
            x_end, valid_points, success, reached_end, inlet_status,
            seed_fraction,
            n_p_final, T_e_final, T_p_final, A_final, v_p_final, n_e_final,
            beta_final, eta_final, Z_final, J_x_final, J_y_final, E_x_final,
            mach_final, G_final,
        )

    n_cur = n_p_in
    te_cur = T_e_in
    A_cur = A_in
    state0 = _evaluate_state_global(n_cur, te_cur, A_cur, dot_N, I_0, seed_fraction, B, SIGMA_EP)
    if state0[14] != EVENT_NONE:
        return (
            x_end, valid_points, success, reached_end, EVENT_INVALID_STATE,
            seed_fraction,
            n_p_final, T_e_final, T_p_final, A_final, v_p_final, n_e_final,
            beta_final, eta_final, Z_final, J_x_final, J_y_final, E_x_final,
            mach_final, G_final,
        )

    success = True
    event_code = EVENT_NONE
    valid_points = 1
    seed_fraction_final = seed_fraction
    prev_mach = state0[12]
    n_p_final = n_cur
    T_e_final = te_cur
    T_p_final = state0[3]
    A_final = A_cur
    v_p_final = state0[4]
    n_e_final = state0[5]
    beta_final = state0[6]
    eta_final = state0[7]
    Z_final = state0[8]
    J_x_final = state0[9]
    J_y_final = state0[10]
    E_x_final = state0[11]
    mach_final = state0[12]
    G_final = state0[13]

    for step in range(1, n_steps + 1):
        n_next, te_next, A_next, step_status = _rk4_step_global(
            n_cur, te_cur, A_cur, dx, dot_N, I_0, seed_fraction, B, SIGMA_EP
        )
        if step_status != EVENT_NONE:
            success = False
            event_code = EVENT_INVALID_STATE
            break

        state_next = _evaluate_state_global(n_next, te_next, A_next, dot_N, I_0, seed_fraction, B, SIGMA_EP)
        if state_next[14] != EVENT_NONE:
            success = False
            event_code = EVENT_INVALID_STATE
            break

        valid_points = step + 1
        x_end = step * dx
        n_p_final = n_next
        T_e_final = te_next
        T_p_final = state_next[3]
        A_final = A_next
        v_p_final = state_next[4]
        n_e_final = state_next[5]
        beta_final = state_next[6]
        eta_final = state_next[7]
        Z_final = state_next[8]
        J_x_final = state_next[9]
        J_y_final = state_next[10]
        E_x_final = state_next[11]
        mach_final = state_next[12]
        G_final = state_next[13]

        descending = mach_final < prev_mach
        if descending:
            if _crossed_threshold(prev_mach, mach_final, mach_high, True):
                event_code = EVENT_MACH_HIGH
                break
            if _crossed_threshold(prev_mach, mach_final, mach_low, True):
                event_code = EVENT_MACH_LOW
                break
        else:
            if _crossed_threshold(prev_mach, mach_final, mach_low, False):
                event_code = EVENT_MACH_LOW
                break
            if _crossed_threshold(prev_mach, mach_final, mach_high, False):
                event_code = EVENT_MACH_HIGH
                break

        n_cur = n_next
        te_cur = te_next
        A_cur = A_next
        prev_mach = mach_final

    if success and event_code == EVENT_NONE and valid_points == (n_steps + 1):
        reached_end = True
        x_end = n_steps * dx

    return (
        x_end, valid_points, success, reached_end, event_code,
        seed_fraction_final,
        n_p_final, T_e_final, T_p_final, A_final, v_p_final, n_e_final,
        beta_final, eta_final, Z_final, J_x_final, J_y_final, E_x_final,
        mach_final, G_final,
    )


@njit(cache=True, parallel=True)
def _solve_batch_terminal_kernel_global(
    n_p_in: np.ndarray,
    Z_in: np.ndarray,
    T_p_in: np.ndarray,
    T_e_in: np.ndarray,
    A_in: np.ndarray,
    B: float,
    dx: float,
    n_steps: int,
    mach_low: float,
    mach_high: float,
):
    n_batch = n_p_in.shape[0]

    x_end = np.zeros(n_batch, dtype=np.float64)
    valid_points = np.zeros(n_batch, dtype=np.int64)
    success = np.zeros(n_batch, dtype=np.bool_)
    reached_end = np.zeros(n_batch, dtype=np.bool_)
    event_code = np.full(n_batch, EVENT_INLET_ERROR, dtype=np.int64)
    seed_fraction = np.full(n_batch, np.nan, dtype=np.float64)
    n_p = np.full(n_batch, np.nan, dtype=np.float64)
    T_e = np.full(n_batch, np.nan, dtype=np.float64)
    T_p = np.full(n_batch, np.nan, dtype=np.float64)
    A = np.full(n_batch, np.nan, dtype=np.float64)
    v_p = np.full(n_batch, np.nan, dtype=np.float64)
    n_e = np.full(n_batch, np.nan, dtype=np.float64)
    beta = np.full(n_batch, np.nan, dtype=np.float64)
    eta = np.full(n_batch, np.nan, dtype=np.float64)
    Z = np.full(n_batch, np.nan, dtype=np.float64)
    J_x = np.full(n_batch, np.nan, dtype=np.float64)
    J_y = np.full(n_batch, np.nan, dtype=np.float64)
    E_x = np.full(n_batch, np.nan, dtype=np.float64)
    mach = np.full(n_batch, np.nan, dtype=np.float64)
    velikhov_margin = np.full(n_batch, np.nan, dtype=np.float64)

    for i in prange(n_batch):
        out = _solve_one_final_global(
            n_p_in[i], Z_in[i], T_p_in[i], T_e_in[i], A_in[i],
            B, dx, n_steps, mach_low, mach_high,
        )
        x_end[i] = out[0]
        valid_points[i] = out[1]
        success[i] = out[2]
        reached_end[i] = out[3]
        event_code[i] = out[4]
        seed_fraction[i] = out[5]
        n_p[i] = out[6]
        T_e[i] = out[7]
        T_p[i] = out[8]
        A[i] = out[9]
        v_p[i] = out[10]
        n_e[i] = out[11]
        beta[i] = out[12]
        eta[i] = out[13]
        Z[i] = out[14]
        J_x[i] = out[15]
        J_y[i] = out[16]
        E_x[i] = out[17]
        mach[i] = out[18]
        velikhov_margin[i] = out[19]

    return (
        x_end, valid_points, success, reached_end, event_code, seed_fraction,
        n_p, T_e, T_p, A, v_p, n_e, beta, eta, Z, J_x, J_y, E_x, mach, velikhov_margin,
    )


@njit(cache=True, parallel=True)
def _solve_batch_profiles_kernel_global(
    n_p_in: np.ndarray,
    Z_in: np.ndarray,
    T_p_in: np.ndarray,
    T_e_in: np.ndarray,
    A_in: np.ndarray,
    B: float,
    dx: float,
    n_steps: int,
    mach_low: float,
    mach_high: float,
):
    n_batch = n_p_in.shape[0]
    n_points = n_steps + 1

    valid_points = np.zeros(n_batch, dtype=np.int64)
    success = np.zeros(n_batch, dtype=np.bool_)
    reached_end = np.zeros(n_batch, dtype=np.bool_)
    event_code = np.full(n_batch, EVENT_INLET_ERROR, dtype=np.int64)
    seed_fraction = np.full(n_batch, np.nan, dtype=np.float64)
    n_p = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    T_e = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    T_p = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    A = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    v_p = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    n_e = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    beta = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    eta = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    Z = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    J_x = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    J_y = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    E_x = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    mach = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    velikhov_margin = np.full((n_batch, n_points), np.nan, dtype=np.float64)

    for i in prange(n_batch):
        inlet_status, dot_N, I_0, seed_fraction_i = _prepare_inlet_constants_global(
            n_p_in[i], Z_in[i], T_p_in[i], T_e_in[i], A_in[i], B,
        )
        if inlet_status != EVENT_NONE:
            event_code[i] = inlet_status
            seed_fraction[i] = seed_fraction_i
            continue
        seed_fraction[i] = seed_fraction_i

        n_cur = n_p_in[i]
        te_cur = T_e_in[i]
        A_cur = A_in[i]
        state0 = _evaluate_state_global(n_cur, te_cur, A_cur, dot_N, I_0, seed_fraction_i, B, SIGMA_EP)
        if state0[14] != EVENT_NONE:
            event_code[i] = EVENT_INVALID_STATE
            continue

        success[i] = True
        event_code[i] = EVENT_NONE
        valid_points[i] = 1

        n_p[i, 0] = n_cur
        T_e[i, 0] = te_cur
        T_p[i, 0] = state0[3]
        A[i, 0] = A_cur
        v_p[i, 0] = state0[4]
        n_e[i, 0] = state0[5]
        beta[i, 0] = state0[6]
        eta[i, 0] = state0[7]
        Z[i, 0] = state0[8]
        J_x[i, 0] = state0[9]
        J_y[i, 0] = state0[10]
        E_x[i, 0] = state0[11]
        mach[i, 0] = state0[12]
        velikhov_margin[i, 0] = state0[13]

        prev_mach = state0[12]

        for step in range(1, n_points):
            n_next, te_next, A_next, step_status = _rk4_step_global(
                n_cur, te_cur, A_cur, dx, dot_N, I_0, seed_fraction_i, B, SIGMA_EP
            )
            if step_status != EVENT_NONE:
                success[i] = False
                event_code[i] = EVENT_INVALID_STATE
                break

            state_next = _evaluate_state_global(n_next, te_next, A_next, dot_N, I_0, seed_fraction_i, B, SIGMA_EP)
            if state_next[14] != EVENT_NONE:
                success[i] = False
                event_code[i] = EVENT_INVALID_STATE
                break

            valid_points[i] = step + 1
            n_p[i, step] = n_next
            T_e[i, step] = te_next
            T_p[i, step] = state_next[3]
            A[i, step] = A_next
            v_p[i, step] = state_next[4]
            n_e[i, step] = state_next[5]
            beta[i, step] = state_next[6]
            eta[i, step] = state_next[7]
            Z[i, step] = state_next[8]
            J_x[i, step] = state_next[9]
            J_y[i, step] = state_next[10]
            E_x[i, step] = state_next[11]
            mach[i, step] = state_next[12]
            velikhov_margin[i, step] = state_next[13]

            descending = state_next[12] < prev_mach
            if descending:
                if _crossed_threshold(prev_mach, state_next[12], mach_high, True):
                    event_code[i] = EVENT_MACH_HIGH
                    break
                if _crossed_threshold(prev_mach, state_next[12], mach_low, True):
                    event_code[i] = EVENT_MACH_LOW
                    break
            else:
                if _crossed_threshold(prev_mach, state_next[12], mach_low, False):
                    event_code[i] = EVENT_MACH_LOW
                    break
                if _crossed_threshold(prev_mach, state_next[12], mach_high, False):
                    event_code[i] = EVENT_MACH_HIGH
                    break

            n_cur = n_next
            te_cur = te_next
            A_cur = A_next
            prev_mach = state_next[12]

        if success[i] and event_code[i] == EVENT_NONE and valid_points[i] == n_points:
            reached_end[i] = True

    return (
        valid_points, success, reached_end, event_code, seed_fraction,
        n_p, T_e, T_p, A, v_p, n_e, beta, eta, Z, J_x, J_y, E_x, mach, velikhov_margin,
    )


@njit(cache=True, parallel=True)
def _evaluate_batch_inlet_metrics_kernel_global(
    n_p_in: np.ndarray,
    Z_in: np.ndarray,
    T_p_in: np.ndarray,
    T_e_in: np.ndarray,
    A_in: np.ndarray,
    B: float,
):
    n_batch = n_p_in.shape[0]

    success = np.zeros(n_batch, dtype=np.bool_)
    event_code = np.full(n_batch, EVENT_INLET_ERROR, dtype=np.int64)
    seed_fraction = np.full(n_batch, np.nan, dtype=np.float64)
    dn_dx = np.full(n_batch, np.nan, dtype=np.float64)
    dTe_dx = np.full(n_batch, np.nan, dtype=np.float64)
    dA_dx = np.full(n_batch, np.nan, dtype=np.float64)
    dTe_rel_grad = np.full(n_batch, np.nan, dtype=np.float64)
    dA_rel_grad = np.full(n_batch, np.nan, dtype=np.float64)
    n_p = np.full(n_batch, np.nan, dtype=np.float64)
    T_e = np.full(n_batch, np.nan, dtype=np.float64)
    T_p = np.full(n_batch, np.nan, dtype=np.float64)
    A = np.full(n_batch, np.nan, dtype=np.float64)
    v_p = np.full(n_batch, np.nan, dtype=np.float64)
    n_e = np.full(n_batch, np.nan, dtype=np.float64)
    beta = np.full(n_batch, np.nan, dtype=np.float64)
    eta = np.full(n_batch, np.nan, dtype=np.float64)
    Z = np.full(n_batch, np.nan, dtype=np.float64)
    J_x = np.full(n_batch, np.nan, dtype=np.float64)
    J_y = np.full(n_batch, np.nan, dtype=np.float64)
    E_x = np.full(n_batch, np.nan, dtype=np.float64)
    mach = np.full(n_batch, np.nan, dtype=np.float64)
    inlet_velikhov_margin = np.full(n_batch, np.nan, dtype=np.float64)

    for i in prange(n_batch):
        inlet_status, dot_N, I_0, seed_fraction_i = _prepare_inlet_constants_global(
            n_p_in[i], Z_in[i], T_p_in[i], T_e_in[i], A_in[i], B,
        )
        seed_fraction[i] = seed_fraction_i
        if inlet_status != EVENT_NONE:
            event_code[i] = inlet_status
            continue

        state = _evaluate_state_global(n_p_in[i], T_e_in[i], A_in[i], dot_N, I_0, seed_fraction_i, B, SIGMA_EP)
        event_code[i] = state[14]
        if state[14] != EVENT_NONE:
            continue

        success[i] = True
        dn_dx[i] = state[0]
        dTe_dx[i] = state[1]
        dA_dx[i] = state[2]
        dTe_rel_grad[i] = state[1] / _safe_pos(T_e_in[i], _TMIN)
        dA_rel_grad[i] = state[2] / _safe_pos(A_in[i])
        n_p[i] = n_p_in[i]
        T_e[i] = T_e_in[i]
        T_p[i] = state[3]
        A[i] = A_in[i]
        v_p[i] = state[4]
        n_e[i] = state[5]
        beta[i] = state[6]
        eta[i] = state[7]
        Z[i] = state[8]
        J_x[i] = state[9]
        J_y[i] = state[10]
        E_x[i] = state[11]
        mach[i] = state[12]
        inlet_velikhov_margin[i] = state[13]

    return (
        success, event_code, seed_fraction, dn_dx, dTe_dx, dA_dx, dTe_rel_grad, dA_rel_grad,
        n_p, T_e, T_p, A, v_p, n_e, beta, eta, Z, J_x, J_y, E_x, mach, inlet_velikhov_margin,
    )


def _broadcast_batch_inputs_global(
    n_p_in,
    Z_in,
    T_p_in,
    T_e_in,
    A_in,
):
    arrays = {
        "n_p_in": _as_1d_array("n_p_in", n_p_in),
        "Z_in": _as_1d_array("Z_in", Z_in),
        "T_p_in": _as_1d_array("T_p_in", T_p_in),
        "T_e_in": _as_1d_array("T_e_in", T_e_in),
        "A_in": _as_1d_array("A_in", A_in),
    }

    sizes = [arr.size for arr in arrays.values()]
    n_batch = max(sizes)
    out = {}
    for name, arr in arrays.items():
        if arr.size == n_batch:
            out[name] = np.ascontiguousarray(arr.astype(float, copy=False))
            continue
        if arr.size == 1:
            out[name] = np.full(n_batch, float(arr[0]), dtype=float)
            continue
        raise ValueError(f"{name} has size {arr.size}, expected 1 or {n_batch}.")
    return out


class ForwardPDESolverV6BatchGlobal:
    def __init__(self, B: float = B_FIELD, length: float = 5.4):
        self.B = float(B)
        self.length = float(length)

    def solve_batch(
        self,
        n_p_in,
        Z_in,
        T_p_in,
        T_e_in,
        A_in,
        dx: float = 1e-3,
        x_span: tuple[float, float] | None = None,
        mach_low: float = 0.99,
        mach_high: float = 1.01,
        store_profiles: bool = True,
        parallel: bool = True,
    ) -> BatchForwardResultV6Global | BatchTerminalResultV6Global:
        del parallel  # current implementation always uses numba parallel kernels
        if dx <= 0.0:
            raise ValueError("dx must be positive.")

        arrays = _broadcast_batch_inputs_global(n_p_in, Z_in, T_p_in, T_e_in, A_in)
        x0, x1 = (0.0, self.length) if x_span is None else (float(x_span[0]), float(x_span[1]))
        if x1 <= x0:
            raise ValueError("x_span must satisfy x1 > x0.")

        length = x1 - x0
        n_steps = max(1, int(math.ceil(length / dx)))
        dx_eff = length / n_steps

        if store_profiles:
            x = x0 + dx_eff * np.arange(n_steps + 1, dtype=float)
            out = _solve_batch_profiles_kernel_global(
                arrays["n_p_in"],
                arrays["Z_in"],
                arrays["T_p_in"],
                arrays["T_e_in"],
                arrays["A_in"],
                self.B,
                dx_eff,
                n_steps,
                float(mach_low),
                float(mach_high),
            )
            return BatchForwardResultV6Global(
                x=x,
                valid_points=out[0],
                success=out[1],
                reached_end=out[2],
                event_code=out[3],
                seed_fraction=out[4],
                n_p=out[5],
                T_e=out[6],
                T_p=out[7],
                A=out[8],
                v_p=out[9],
                n_e=out[10],
                beta=out[11],
                eta=out[12],
                Z=out[13],
                J_x=out[14],
                J_y=out[15],
                E_x=out[16],
                mach=out[17],
                velikhov_margin=out[18],
                step_size=dx_eff,
            )

        out = _solve_batch_terminal_kernel_global(
            arrays["n_p_in"],
            arrays["Z_in"],
            arrays["T_p_in"],
            arrays["T_e_in"],
            arrays["A_in"],
            self.B,
            dx_eff,
            n_steps,
            float(mach_low),
            float(mach_high),
        )
        return BatchTerminalResultV6Global(
            x_end=out[0],
            valid_points=out[1],
            success=out[2],
            reached_end=out[3],
            event_code=out[4],
            seed_fraction=out[5],
            n_p=out[6],
            T_e=out[7],
            T_p=out[8],
            A=out[9],
            v_p=out[10],
            n_e=out[11],
            beta=out[12],
            eta=out[13],
            Z=out[14],
            J_x=out[15],
            J_y=out[16],
            E_x=out[17],
            mach=out[18],
            velikhov_margin=out[19],
            step_size=dx_eff,
        )

    def evaluate_inlet_batch(
        self,
        n_p_in,
        Z_in,
        T_p_in,
        T_e_in,
        A_in,
        parallel: bool = True,
    ) -> BatchInletMetricsV6Global:
        del parallel
        arrays = _broadcast_batch_inputs_global(n_p_in, Z_in, T_p_in, T_e_in, A_in)
        out = _evaluate_batch_inlet_metrics_kernel_global(
            arrays["n_p_in"],
            arrays["Z_in"],
            arrays["T_p_in"],
            arrays["T_e_in"],
            arrays["A_in"],
            self.B,
        )
        return BatchInletMetricsV6Global(
            success=out[0],
            event_code=out[1],
            seed_fraction=out[2],
            dn_dx=out[3],
            dTe_dx=out[4],
            dA_dx=out[5],
            dTe_rel_grad=out[6],
            dA_rel_grad=out[7],
            n_p=out[8],
            T_e=out[9],
            T_p=out[10],
            A=out[11],
            v_p=out[12],
            n_e=out[13],
            beta=out[14],
            eta=out[15],
            Z=out[16],
            J_x=out[17],
            J_y=out[18],
            E_x=out[19],
            mach=out[20],
            inlet_velikhov_margin=out[21],
        )
