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

from local_algebraic_closure import (
    B_FIELD,
    E_CHARGE,
    E_I,
    K_B,
    M_P,
    SIGMA_EP,
    beta_from_np_te,
    compute_currents_fields,
    local_closure_with_partials,
    ne_from_np_te,
)


EVENT_NONE = 0
EVENT_MACH_LOW = 1
EVENT_MACH_HIGH = 2
EVENT_INVALID_STATE = 3
EVENT_INLET_ERROR = 4

_EPS = 1e-300
_TMIN = 1.0


@dataclass
class BatchForwardResultV6:
    x: np.ndarray
    valid_points: np.ndarray
    success: np.ndarray
    reached_end: np.ndarray
    event_code: np.ndarray
    n_p: np.ndarray
    T_e: np.ndarray
    T_p: np.ndarray
    v_p: np.ndarray
    n_e: np.ndarray
    beta: np.ndarray
    eta: np.ndarray
    Z: np.ndarray
    J_x: np.ndarray
    J_y: np.ndarray
    E_x: np.ndarray
    mach: np.ndarray
    step_size: float

    def event_names(self) -> List[Optional[str]]:
        return [event_name_from_code(int(code)) for code in self.event_code]


@dataclass
class BatchTerminalResultV6:
    x_end: np.ndarray
    valid_points: np.ndarray
    success: np.ndarray
    reached_end: np.ndarray
    event_code: np.ndarray
    n_p: np.ndarray
    T_e: np.ndarray
    T_p: np.ndarray
    v_p: np.ndarray
    n_e: np.ndarray
    beta: np.ndarray
    eta: np.ndarray
    Z: np.ndarray
    J_x: np.ndarray
    J_y: np.ndarray
    E_x: np.ndarray
    mach: np.ndarray
    step_size: float

    def event_names(self) -> List[Optional[str]]:
        return [event_name_from_code(int(code)) for code in self.event_code]


@dataclass
class BatchInletMetricsV6:
    success: np.ndarray
    event_code: np.ndarray
    dn_dx: np.ndarray
    dTe_dx: np.ndarray
    dTe_rel_grad: np.ndarray
    n_p: np.ndarray
    T_e: np.ndarray
    T_p: np.ndarray
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
    if code == EVENT_MACH_LOW:
        return "mach_0p99"
    if code == EVENT_MACH_HIGH:
        return "mach_1p01"
    if code == EVENT_INVALID_STATE:
        return "invalid_state"
    if code == EVENT_INLET_ERROR:
        return "inlet_error"
    return None


@njit(cache=True)
def _safe_pos(x: float, floor: float = _EPS) -> float:
    return x if x > floor else floor


@njit(cache=True)
def _crossed_threshold(prev_mach: float, mach: float, threshold: float, descending: bool) -> bool:
    if descending:
        return mach <= threshold <= prev_mach
    return prev_mach <= threshold <= mach


@njit(cache=True)
def _inlet_velocity_from_eq8_prime(T_e_in: float, T_p_in: float, n_p_in: float, Z_in: float, B: float):
    if n_p_in <= 0.0 or T_e_in <= T_p_in:
        return 0.0, EVENT_INLET_ERROR

    beta_in = beta_from_np_te(n_p_in, T_e_in, B=B, sigma_ep=SIGMA_EP)
    b2 = beta_in * beta_in
    d = 1.0 + Z_in
    den = _safe_pos((b2 + d) * (b2 + d))
    F = b2 * (b2 + d * d) / den
    if F <= 0.0:
        return 0.0, EVENT_INLET_ERROR

    vp2 = 3.0 * K_B * (T_e_in - T_p_in) / (M_P * F)
    if vp2 <= 0.0 or (not math.isfinite(vp2)):
        return 0.0, EVENT_INLET_ERROR
    return math.sqrt(vp2), EVENT_NONE


@njit(cache=True)
def _prepare_inlet_constants(
    n_p_in: float,
    Z_in: float,
    T_p_in: float,
    T_e_in: float,
    seed_fraction: float,
    B: float,
):
    if seed_fraction <= 0.0 or seed_fraction > 1.0:
        return EVENT_INLET_ERROR, 0.0, 0.0

    v_in, inlet_status = _inlet_velocity_from_eq8_prime(T_e_in, T_p_in, n_p_in, Z_in, B)
    if inlet_status != EVENT_NONE:
        return inlet_status, 0.0, 0.0

    n_e_in = ne_from_np_te(n_p_in, T_e_in, seed_fraction)
    beta_in = beta_from_np_te(n_p_in, T_e_in, B=B, sigma_ep=SIGMA_EP)
    b2 = beta_in * beta_in
    den = b2 + 1.0 + Z_in
    if abs(den) < _EPS:
        den = _EPS if den >= 0.0 else -_EPS
    Jx0 = b2 / den * E_CHARGE * n_e_in * v_in
    m0 = n_p_in * v_in
    return EVENT_NONE, m0, Jx0


@njit(cache=True)
def _velikhov_margin_one(
    beta: float,
    T_e: float,
    T_p: float,
    n_e: float,
    n_p: float,
    seed_fraction: float,
):
    n_s = seed_fraction * n_p
    if n_s <= 0.0 or T_p <= 0.0:
        return -math.inf

    fI = n_e / _safe_pos(n_s)
    if fI < 1e-12:
        fI = 1e-12
    if fI > 1.0 - 1e-12:
        fI = 1.0 - 1e-12

    delta = T_e / _safe_pos(T_p) - 1.0
    if delta < 1e-12:
        delta = 1e-12

    alpha = (K_B * T_e / (2.0 * E_I)) * (2.0 - fI) / _safe_pos(1.0 - fI)
    rhs = 4.0 * alpha * (2.0 + 1.0 / delta) * (1.0 + alpha * (1.0 + 1.0 / delta))
    lhs = beta * beta
    return rhs - lhs


@njit(cache=True)
def _evaluate_state(
    n_p: float,
    T_e: float,
    m0: float,
    Jx0: float,
    seed_fraction: float,
    B: float,
    sigma_ep: float,
):
    n_p = max(n_p, 1e-20)
    T_e = max(T_e, _TMIN)

    vals = local_closure_with_partials(
        n_p=n_p,
        T_e=T_e,
        m_0=m0,
        J_x0=Jx0,
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

    J_x, J_y, E_x, _ = compute_currents_fields(v_p, n_e, beta, eta, Z)
    nu_E = eta * 2.0 * E_CHARGE * E_CHARGE * n_e / M_P

    A = -M_P * v_p * v_p + K_B * T_p
    Bcoef = K_B * n_p
    C = J_y * B
    D = -T_p
    Ecoef = 1.5 * n_p
    F_rhs = 1.5 * nu_E * n_e * (T_e - T_p) / _safe_pos(v_p)

    det = A * Ecoef - Bcoef * D
    if abs(det) < _EPS:
        det = _EPS if det >= 0.0 else -_EPS

    dn_dx = (C * Ecoef - Bcoef * F_rhs) / det
    if abs(dTp_dTe) < _EPS:
        dTp_dTe = _EPS if dTp_dTe >= 0.0 else -_EPS
    dTp_dx = (A * F_rhs - C * D) / det
    dTe_dx = (dTp_dx - dTp_dnp * dn_dx) / dTp_dTe

    c_s = math.sqrt((5.0 / 3.0) * K_B * _safe_pos(T_p) / M_P)
    mach = v_p / _safe_pos(c_s)

    status = EVENT_NONE
    finite_ok = (
        math.isfinite(dn_dx)
        and math.isfinite(dTe_dx)
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
    )
    if (not finite_ok) or T_p <= 0.0 or v_p <= 0.0 or n_e < 0.0:
        status = EVENT_INVALID_STATE

    return dn_dx, dTe_dx, T_p, v_p, n_e, beta, eta, Z, J_x, J_y, E_x, mach, status


@njit(cache=True)
def _rk4_step(
    n_p: float,
    T_e: float,
    dx: float,
    m0: float,
    Jx0: float,
    seed_fraction: float,
    B: float,
    sigma_ep: float,
):
    k1 = _evaluate_state(n_p, T_e, m0, Jx0, seed_fraction, B, sigma_ep)
    if k1[12] != EVENT_NONE:
        return n_p, T_e, EVENT_INVALID_STATE

    y2_np = n_p + 0.5 * dx * k1[0]
    y2_te = T_e + 0.5 * dx * k1[1]
    k2 = _evaluate_state(y2_np, y2_te, m0, Jx0, seed_fraction, B, sigma_ep)
    if k2[12] != EVENT_NONE:
        return n_p, T_e, EVENT_INVALID_STATE

    y3_np = n_p + 0.5 * dx * k2[0]
    y3_te = T_e + 0.5 * dx * k2[1]
    k3 = _evaluate_state(y3_np, y3_te, m0, Jx0, seed_fraction, B, sigma_ep)
    if k3[12] != EVENT_NONE:
        return n_p, T_e, EVENT_INVALID_STATE

    y4_np = n_p + dx * k3[0]
    y4_te = T_e + dx * k3[1]
    k4 = _evaluate_state(y4_np, y4_te, m0, Jx0, seed_fraction, B, sigma_ep)
    if k4[12] != EVENT_NONE:
        return n_p, T_e, EVENT_INVALID_STATE

    n_next = n_p + dx / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
    te_next = T_e + dx / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
    if (not math.isfinite(n_next)) or (not math.isfinite(te_next)):
        return n_p, T_e, EVENT_INVALID_STATE
    return n_next, te_next, EVENT_NONE


@njit(cache=True)
def _solve_one_final(
    n_p_in: float,
    Z_in: float,
    T_p_in: float,
    T_e_in: float,
    seed_fraction: float,
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

    n_p_final = np.nan
    T_e_final = np.nan
    T_p_final = np.nan
    v_p_final = np.nan
    n_e_final = np.nan
    beta_final = np.nan
    eta_final = np.nan
    Z_final = np.nan
    J_x_final = np.nan
    J_y_final = np.nan
    E_x_final = np.nan
    mach_final = np.nan

    inlet_status, m0, Jx0 = _prepare_inlet_constants(
        n_p_in,
        Z_in,
        T_p_in,
        T_e_in,
        seed_fraction,
        B,
    )
    if inlet_status != EVENT_NONE:
        return (
            x_end,
            valid_points,
            success,
            reached_end,
            event_code,
            n_p_final,
            T_e_final,
            T_p_final,
            v_p_final,
            n_e_final,
            beta_final,
            eta_final,
            Z_final,
            J_x_final,
            J_y_final,
            E_x_final,
            mach_final,
        )

    n_cur = n_p_in
    te_cur = T_e_in
    state0 = _evaluate_state(n_cur, te_cur, m0, Jx0, seed_fraction, B, SIGMA_EP)
    if state0[12] != EVENT_NONE:
        return (
            x_end,
            valid_points,
            success,
            reached_end,
            EVENT_INVALID_STATE,
            n_p_final,
            T_e_final,
            T_p_final,
            v_p_final,
            n_e_final,
            beta_final,
            eta_final,
            Z_final,
            J_x_final,
            J_y_final,
            E_x_final,
            mach_final,
        )

    success = True
    event_code = EVENT_NONE
    valid_points = 1
    prev_mach = state0[11]
    n_p_final = n_cur
    T_e_final = te_cur
    T_p_final = state0[2]
    v_p_final = state0[3]
    n_e_final = state0[4]
    beta_final = state0[5]
    eta_final = state0[6]
    Z_final = state0[7]
    J_x_final = state0[8]
    J_y_final = state0[9]
    E_x_final = state0[10]
    mach_final = state0[11]

    for step in range(1, n_steps + 1):
        n_next, te_next, step_status = _rk4_step(n_cur, te_cur, dx, m0, Jx0, seed_fraction, B, SIGMA_EP)
        if step_status != EVENT_NONE:
            success = False
            event_code = EVENT_INVALID_STATE
            break

        state_next = _evaluate_state(n_next, te_next, m0, Jx0, seed_fraction, B, SIGMA_EP)
        if state_next[12] != EVENT_NONE:
            success = False
            event_code = EVENT_INVALID_STATE
            break

        valid_points = step + 1
        x_end = step * dx
        n_p_final = n_next
        T_e_final = te_next
        T_p_final = state_next[2]
        v_p_final = state_next[3]
        n_e_final = state_next[4]
        beta_final = state_next[5]
        eta_final = state_next[6]
        Z_final = state_next[7]
        J_x_final = state_next[8]
        J_y_final = state_next[9]
        E_x_final = state_next[10]
        mach_final = state_next[11]

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
        prev_mach = mach_final

    if success and event_code == EVENT_NONE and valid_points == (n_steps + 1):
        reached_end = True
        x_end = n_steps * dx

    return (
        x_end,
        valid_points,
        success,
        reached_end,
        event_code,
        n_p_final,
        T_e_final,
        T_p_final,
        v_p_final,
        n_e_final,
        beta_final,
        eta_final,
        Z_final,
        J_x_final,
        J_y_final,
        E_x_final,
        mach_final,
    )


@njit(cache=True, parallel=True)
def _solve_batch_terminal_kernel(
    n_p_in: np.ndarray,
    Z_in: np.ndarray,
    T_p_in: np.ndarray,
    T_e_in: np.ndarray,
    seed_fraction: np.ndarray,
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
    n_p = np.full(n_batch, np.nan, dtype=np.float64)
    T_e = np.full(n_batch, np.nan, dtype=np.float64)
    T_p = np.full(n_batch, np.nan, dtype=np.float64)
    v_p = np.full(n_batch, np.nan, dtype=np.float64)
    n_e = np.full(n_batch, np.nan, dtype=np.float64)
    beta = np.full(n_batch, np.nan, dtype=np.float64)
    eta = np.full(n_batch, np.nan, dtype=np.float64)
    Z = np.full(n_batch, np.nan, dtype=np.float64)
    J_x = np.full(n_batch, np.nan, dtype=np.float64)
    J_y = np.full(n_batch, np.nan, dtype=np.float64)
    E_x = np.full(n_batch, np.nan, dtype=np.float64)
    mach = np.full(n_batch, np.nan, dtype=np.float64)

    for i in prange(n_batch):
        out = _solve_one_final(
            n_p_in[i],
            Z_in[i],
            T_p_in[i],
            T_e_in[i],
            seed_fraction[i],
            B,
            dx,
            n_steps,
            mach_low,
            mach_high,
        )
        x_end[i] = out[0]
        valid_points[i] = out[1]
        success[i] = out[2]
        reached_end[i] = out[3]
        event_code[i] = out[4]
        n_p[i] = out[5]
        T_e[i] = out[6]
        T_p[i] = out[7]
        v_p[i] = out[8]
        n_e[i] = out[9]
        beta[i] = out[10]
        eta[i] = out[11]
        Z[i] = out[12]
        J_x[i] = out[13]
        J_y[i] = out[14]
        E_x[i] = out[15]
        mach[i] = out[16]

    return x_end, valid_points, success, reached_end, event_code, n_p, T_e, T_p, v_p, n_e, beta, eta, Z, J_x, J_y, E_x, mach


@njit(cache=True)
def _solve_serial_terminal_kernel(
    n_p_in: np.ndarray,
    Z_in: np.ndarray,
    T_p_in: np.ndarray,
    T_e_in: np.ndarray,
    seed_fraction: np.ndarray,
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
    n_p = np.full(n_batch, np.nan, dtype=np.float64)
    T_e = np.full(n_batch, np.nan, dtype=np.float64)
    T_p = np.full(n_batch, np.nan, dtype=np.float64)
    v_p = np.full(n_batch, np.nan, dtype=np.float64)
    n_e = np.full(n_batch, np.nan, dtype=np.float64)
    beta = np.full(n_batch, np.nan, dtype=np.float64)
    eta = np.full(n_batch, np.nan, dtype=np.float64)
    Z = np.full(n_batch, np.nan, dtype=np.float64)
    J_x = np.full(n_batch, np.nan, dtype=np.float64)
    J_y = np.full(n_batch, np.nan, dtype=np.float64)
    E_x = np.full(n_batch, np.nan, dtype=np.float64)
    mach = np.full(n_batch, np.nan, dtype=np.float64)

    for i in range(n_batch):
        out = _solve_one_final(
            n_p_in[i],
            Z_in[i],
            T_p_in[i],
            T_e_in[i],
            seed_fraction[i],
            B,
            dx,
            n_steps,
            mach_low,
            mach_high,
        )
        x_end[i] = out[0]
        valid_points[i] = out[1]
        success[i] = out[2]
        reached_end[i] = out[3]
        event_code[i] = out[4]
        n_p[i] = out[5]
        T_e[i] = out[6]
        T_p[i] = out[7]
        v_p[i] = out[8]
        n_e[i] = out[9]
        beta[i] = out[10]
        eta[i] = out[11]
        Z[i] = out[12]
        J_x[i] = out[13]
        J_y[i] = out[14]
        E_x[i] = out[15]
        mach[i] = out[16]

    return x_end, valid_points, success, reached_end, event_code, n_p, T_e, T_p, v_p, n_e, beta, eta, Z, J_x, J_y, E_x, mach


@njit(cache=True, parallel=True)
def _solve_batch_profiles_kernel(
    n_p_in: np.ndarray,
    Z_in: np.ndarray,
    T_p_in: np.ndarray,
    T_e_in: np.ndarray,
    seed_fraction: np.ndarray,
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
    n_p = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    T_e = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    T_p = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    v_p = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    n_e = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    beta = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    eta = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    Z = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    J_x = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    J_y = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    E_x = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    mach = np.full((n_batch, n_points), np.nan, dtype=np.float64)

    for i in prange(n_batch):
        inlet_status, m0, Jx0 = _prepare_inlet_constants(
            n_p_in[i],
            Z_in[i],
            T_p_in[i],
            T_e_in[i],
            seed_fraction[i],
            B,
        )
        if inlet_status != EVENT_NONE:
            continue

        n_cur = n_p_in[i]
        te_cur = T_e_in[i]
        state0 = _evaluate_state(n_cur, te_cur, m0, Jx0, seed_fraction[i], B, SIGMA_EP)
        if state0[12] != EVENT_NONE:
            event_code[i] = EVENT_INVALID_STATE
            continue

        success[i] = True
        event_code[i] = EVENT_NONE
        valid_points[i] = 1

        n_p[i, 0] = n_cur
        T_e[i, 0] = te_cur
        T_p[i, 0] = state0[2]
        v_p[i, 0] = state0[3]
        n_e[i, 0] = state0[4]
        beta[i, 0] = state0[5]
        eta[i, 0] = state0[6]
        Z[i, 0] = state0[7]
        J_x[i, 0] = state0[8]
        J_y[i, 0] = state0[9]
        E_x[i, 0] = state0[10]
        mach[i, 0] = state0[11]

        prev_mach = state0[11]

        for step in range(1, n_points):
            n_next, te_next, step_status = _rk4_step(n_cur, te_cur, dx, m0, Jx0, seed_fraction[i], B, SIGMA_EP)
            if step_status != EVENT_NONE:
                success[i] = False
                event_code[i] = EVENT_INVALID_STATE
                break

            state_next = _evaluate_state(n_next, te_next, m0, Jx0, seed_fraction[i], B, SIGMA_EP)
            if state_next[12] != EVENT_NONE:
                success[i] = False
                event_code[i] = EVENT_INVALID_STATE
                break

            valid_points[i] = step + 1
            n_p[i, step] = n_next
            T_e[i, step] = te_next
            T_p[i, step] = state_next[2]
            v_p[i, step] = state_next[3]
            n_e[i, step] = state_next[4]
            beta[i, step] = state_next[5]
            eta[i, step] = state_next[6]
            Z[i, step] = state_next[7]
            J_x[i, step] = state_next[8]
            J_y[i, step] = state_next[9]
            E_x[i, step] = state_next[10]
            mach[i, step] = state_next[11]

            descending = state_next[11] < prev_mach
            if descending:
                if _crossed_threshold(prev_mach, state_next[11], mach_high, True):
                    event_code[i] = EVENT_MACH_HIGH
                    break
                if _crossed_threshold(prev_mach, state_next[11], mach_low, True):
                    event_code[i] = EVENT_MACH_LOW
                    break
            else:
                if _crossed_threshold(prev_mach, state_next[11], mach_low, False):
                    event_code[i] = EVENT_MACH_LOW
                    break
                if _crossed_threshold(prev_mach, state_next[11], mach_high, False):
                    event_code[i] = EVENT_MACH_HIGH
                    break

            n_cur = n_next
            te_cur = te_next
            prev_mach = state_next[11]

        if success[i] and event_code[i] == EVENT_NONE and valid_points[i] == n_points:
            reached_end[i] = True

    return valid_points, success, reached_end, event_code, n_p, T_e, T_p, v_p, n_e, beta, eta, Z, J_x, J_y, E_x, mach


@njit(cache=True, parallel=True)
def _evaluate_batch_inlet_metrics_kernel(
    n_p_in: np.ndarray,
    Z_in: np.ndarray,
    T_p_in: np.ndarray,
    T_e_in: np.ndarray,
    seed_fraction: np.ndarray,
    B: float,
):
    n_batch = n_p_in.shape[0]

    success = np.zeros(n_batch, dtype=np.bool_)
    event_code = np.full(n_batch, EVENT_INLET_ERROR, dtype=np.int64)
    dn_dx = np.full(n_batch, np.nan, dtype=np.float64)
    dTe_dx = np.full(n_batch, np.nan, dtype=np.float64)
    dTe_rel_grad = np.full(n_batch, np.nan, dtype=np.float64)
    n_p = np.full(n_batch, np.nan, dtype=np.float64)
    T_e = np.full(n_batch, np.nan, dtype=np.float64)
    T_p = np.full(n_batch, np.nan, dtype=np.float64)
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
        inlet_status, m0, Jx0 = _prepare_inlet_constants(
            n_p_in[i],
            Z_in[i],
            T_p_in[i],
            T_e_in[i],
            seed_fraction[i],
            B,
        )
        if inlet_status != EVENT_NONE:
            event_code[i] = inlet_status
            continue

        state = _evaluate_state(n_p_in[i], T_e_in[i], m0, Jx0, seed_fraction[i], B, SIGMA_EP)
        event_code[i] = state[12]
        if state[12] != EVENT_NONE:
            continue

        success[i] = True
        dn_dx[i] = state[0]
        dTe_dx[i] = state[1]
        dTe_rel_grad[i] = state[1] / _safe_pos(T_e_in[i], _TMIN)
        n_p[i] = n_p_in[i]
        T_e[i] = T_e_in[i]
        T_p[i] = state[2]
        v_p[i] = state[3]
        n_e[i] = state[4]
        beta[i] = state[5]
        eta[i] = state[6]
        Z[i] = state[7]
        J_x[i] = state[8]
        J_y[i] = state[9]
        E_x[i] = state[10]
        mach[i] = state[11]
        inlet_velikhov_margin[i] = _velikhov_margin_one(
            state[5],
            T_e_in[i],
            state[2],
            state[4],
            n_p_in[i],
            seed_fraction[i],
        )

    return (
        success,
        event_code,
        dn_dx,
        dTe_dx,
        dTe_rel_grad,
        n_p,
        T_e,
        T_p,
        v_p,
        n_e,
        beta,
        eta,
        Z,
        J_x,
        J_y,
        E_x,
        mach,
        inlet_velikhov_margin,
    )


@njit(cache=True)
def _evaluate_serial_inlet_metrics_kernel(
    n_p_in: np.ndarray,
    Z_in: np.ndarray,
    T_p_in: np.ndarray,
    T_e_in: np.ndarray,
    seed_fraction: np.ndarray,
    B: float,
):
    n_batch = n_p_in.shape[0]

    success = np.zeros(n_batch, dtype=np.bool_)
    event_code = np.full(n_batch, EVENT_INLET_ERROR, dtype=np.int64)
    dn_dx = np.full(n_batch, np.nan, dtype=np.float64)
    dTe_dx = np.full(n_batch, np.nan, dtype=np.float64)
    dTe_rel_grad = np.full(n_batch, np.nan, dtype=np.float64)
    n_p = np.full(n_batch, np.nan, dtype=np.float64)
    T_e = np.full(n_batch, np.nan, dtype=np.float64)
    T_p = np.full(n_batch, np.nan, dtype=np.float64)
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

    for i in range(n_batch):
        inlet_status, m0, Jx0 = _prepare_inlet_constants(
            n_p_in[i],
            Z_in[i],
            T_p_in[i],
            T_e_in[i],
            seed_fraction[i],
            B,
        )
        if inlet_status != EVENT_NONE:
            event_code[i] = inlet_status
            continue

        state = _evaluate_state(n_p_in[i], T_e_in[i], m0, Jx0, seed_fraction[i], B, SIGMA_EP)
        event_code[i] = state[12]
        if state[12] != EVENT_NONE:
            continue

        success[i] = True
        dn_dx[i] = state[0]
        dTe_dx[i] = state[1]
        dTe_rel_grad[i] = state[1] / _safe_pos(T_e_in[i], _TMIN)
        n_p[i] = n_p_in[i]
        T_e[i] = T_e_in[i]
        T_p[i] = state[2]
        v_p[i] = state[3]
        n_e[i] = state[4]
        beta[i] = state[5]
        eta[i] = state[6]
        Z[i] = state[7]
        J_x[i] = state[8]
        J_y[i] = state[9]
        E_x[i] = state[10]
        mach[i] = state[11]
        inlet_velikhov_margin[i] = _velikhov_margin_one(
            state[5],
            T_e_in[i],
            state[2],
            state[4],
            n_p_in[i],
            seed_fraction[i],
        )

    return (
        success,
        event_code,
        dn_dx,
        dTe_dx,
        dTe_rel_grad,
        n_p,
        T_e,
        T_p,
        v_p,
        n_e,
        beta,
        eta,
        Z,
        J_x,
        J_y,
        E_x,
        mach,
        inlet_velikhov_margin,
    )


def _as_1d_array(name: str, value) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be scalar or 1D array.")
    return arr


def _broadcast_batch_inputs(
    n_p_in,
    Z_in,
    T_p_in,
    T_e_in,
    seed_fraction,
):
    arrays = {
        "n_p_in": _as_1d_array("n_p_in", n_p_in),
        "Z_in": _as_1d_array("Z_in", Z_in),
        "T_p_in": _as_1d_array("T_p_in", T_p_in),
        "T_e_in": _as_1d_array("T_e_in", T_e_in),
        "seed_fraction": _as_1d_array("seed_fraction", seed_fraction),
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


class ForwardPDESolverV6Batch:
    def __init__(self, B: float = B_FIELD, length: float = 0.039):
        self.B = float(B)
        self.length = float(length)

    def solve_batch(
        self,
        n_p_in,
        Z_in,
        T_p_in,
        T_e_in,
        seed_fraction,
        dx: float = 1e-4,
        x_span: tuple[float, float] | None = None,
        mach_low: float = 0.99,
        mach_high: float = 1.01,
        store_profiles: bool = True,
        parallel: bool = True,
    ) -> BatchForwardResultV6 | BatchTerminalResultV6:
        if dx <= 0.0:
            raise ValueError("dx must be positive.")

        arrays = _broadcast_batch_inputs(n_p_in, Z_in, T_p_in, T_e_in, seed_fraction)
        x0, x1 = (0.0, self.length) if x_span is None else (float(x_span[0]), float(x_span[1]))
        if x1 <= x0:
            raise ValueError("x_span must satisfy x1 > x0.")

        length = x1 - x0
        n_steps = max(1, int(math.ceil(length / dx)))
        dx_eff = length / n_steps

        if store_profiles:
            x = x0 + dx_eff * np.arange(n_steps + 1, dtype=float)
            out = _solve_batch_profiles_kernel(
                arrays["n_p_in"],
                arrays["Z_in"],
                arrays["T_p_in"],
                arrays["T_e_in"],
                arrays["seed_fraction"],
                self.B,
                dx_eff,
                n_steps,
                float(mach_low),
                float(mach_high),
            )
            return BatchForwardResultV6(
                x=x,
                valid_points=out[0],
                success=out[1],
                reached_end=out[2],
                event_code=out[3],
                n_p=out[4],
                T_e=out[5],
                T_p=out[6],
                v_p=out[7],
                n_e=out[8],
                beta=out[9],
                eta=out[10],
                Z=out[11],
                J_x=out[12],
                J_y=out[13],
                E_x=out[14],
                mach=out[15],
                step_size=dx_eff,
            )

        if parallel:
            out = _solve_batch_terminal_kernel(
                arrays["n_p_in"],
                arrays["Z_in"],
                arrays["T_p_in"],
                arrays["T_e_in"],
                arrays["seed_fraction"],
                self.B,
                dx_eff,
                n_steps,
                float(mach_low),
                float(mach_high),
            )
        else:
            out = _solve_serial_terminal_kernel(
                arrays["n_p_in"],
                arrays["Z_in"],
                arrays["T_p_in"],
                arrays["T_e_in"],
                arrays["seed_fraction"],
                self.B,
                dx_eff,
                n_steps,
                float(mach_low),
                float(mach_high),
            )
        return BatchTerminalResultV6(
            x_end=x0 + out[0],
            valid_points=out[1],
            success=out[2],
            reached_end=out[3],
            event_code=out[4],
            n_p=out[5],
            T_e=out[6],
            T_p=out[7],
            v_p=out[8],
            n_e=out[9],
            beta=out[10],
            eta=out[11],
            Z=out[12],
            J_x=out[13],
            J_y=out[14],
            E_x=out[15],
            mach=out[16],
            step_size=dx_eff,
        )

    def evaluate_inlet_batch(
        self,
        n_p_in,
        Z_in,
        T_p_in,
        T_e_in,
        seed_fraction,
        parallel: bool = True,
    ) -> BatchInletMetricsV6:
        arrays = _broadcast_batch_inputs(n_p_in, Z_in, T_p_in, T_e_in, seed_fraction)

        if parallel:
            out = _evaluate_batch_inlet_metrics_kernel(
                arrays["n_p_in"],
                arrays["Z_in"],
                arrays["T_p_in"],
                arrays["T_e_in"],
                arrays["seed_fraction"],
                self.B,
            )
        else:
            out = _evaluate_serial_inlet_metrics_kernel(
                arrays["n_p_in"],
                arrays["Z_in"],
                arrays["T_p_in"],
                arrays["T_e_in"],
                arrays["seed_fraction"],
                self.B,
            )

        return BatchInletMetricsV6(
            success=out[0],
            event_code=out[1],
            dn_dx=out[2],
            dTe_dx=out[3],
            dTe_rel_grad=out[4],
            n_p=out[5],
            T_e=out[6],
            T_p=out[7],
            v_p=out[8],
            n_e=out[9],
            beta=out[10],
            eta=out[11],
            Z=out[12],
            J_x=out[13],
            J_y=out[14],
            E_x=out[15],
            mach=out[16],
            inlet_velikhov_margin=out[17],
        )
