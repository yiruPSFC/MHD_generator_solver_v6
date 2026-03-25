from __future__ import annotations

"""
V6 PDE forward solver for 1D partially-ionized MHD flow.

Core changes vs V5:
- State variables are y = [n_p, T_e]
- Local closure is 100% explicit algebraic (no brentq/fsolve)
- Jacobian transform from (dn_p/dx, dT_p/dx) -> (dn_p/dx, dT_e/dx) uses analytical partials
"""

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp

from local_algebraic_closure import (
    B_FIELD,
    E_CHARGE,
    K_B,
    M_P,
    SIGMA_EP,
    beta_from_np_te,
    compute_currents_fields,
    evaluate_local_closure,
    ne_from_np_te,
)


@dataclass
class InletConstants:
    m0: float
    Jx0: float
    v_in: float
    n_e_in: float
    beta_in: float


@dataclass
class ForwardResultV6:
    success: bool
    message: str
    status: int
    x: np.ndarray
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
    event_name: Optional[str]
    solver_stats: Dict[str, float]


def _safe_pos(x: float, floor: float = 1e-300) -> float:
    return x if x > floor else floor


def _inlet_velocity_from_eq8_prime(T_e_in: float, T_p_in: float, n_p_in: float, Z_in: float, B: float) -> float:
    if T_e_in <= T_p_in:
        raise ValueError("inlet requires T_e_in > T_p_in to invert Eq.(8').")

    beta_in = float(beta_from_np_te(n_p_in, T_e_in, B=B, sigma_ep=SIGMA_EP))
    b2 = beta_in * beta_in
    d = 1.0 + Z_in
    den = _safe_pos((b2 + d) * (b2 + d))
    F = b2 * (b2 + d * d) / den
    if F <= 0.0:
        raise ValueError("inlet Eq.(8') gives non-positive F(beta,Z).")

    vp2 = 3.0 * K_B * (T_e_in - T_p_in) / (M_P * F)
    if vp2 <= 0.0:
        raise ValueError("inlet Eq.(8') gives non-positive v_p^2.")
    return float(math.sqrt(vp2))


class ForwardPDESolverV6:
    def __init__(self, B: float = B_FIELD, length: float = 0.039):
        self.B = float(B)
        self.length = float(length)
        self._inlet: Optional[InletConstants] = None
        self._seed_fraction: Optional[float] = None
        self._rhs_calls = 0

    def initialize_from_inlet(
        self,
        n_p_in: float,
        Z_in: float,
        T_p_in: float,
        T_e_in: float,
        seed_fraction: float,
    ) -> InletConstants:
        n_p_in = float(n_p_in)
        Z_in = float(Z_in)
        T_p_in = float(T_p_in)
        T_e_in = float(T_e_in)
        seed_fraction = float(seed_fraction)

        if n_p_in <= 0.0:
            raise ValueError("n_p_in must be positive.")
        if seed_fraction <= 0.0 or seed_fraction > 1.0:
            raise ValueError("seed_fraction must be in (0, 1].")

        v_in = _inlet_velocity_from_eq8_prime(T_e_in, T_p_in, n_p_in, Z_in, self.B)
        n_e_in = float(ne_from_np_te(n_p_in, T_e_in, seed_fraction))
        beta_in = float(beta_from_np_te(n_p_in, T_e_in, B=self.B, sigma_ep=SIGMA_EP))

        b2 = beta_in * beta_in
        den = b2 + 1.0 + Z_in
        if abs(den) < 1e-300:
            den = 1e-300 if den >= 0.0 else -1e-300
        Jx0 = b2 / den * E_CHARGE * n_e_in * v_in

        m0 = n_p_in * v_in

        inlet = InletConstants(m0=float(m0), Jx0=float(Jx0), v_in=float(v_in), n_e_in=float(n_e_in), beta_in=float(beta_in))
        self._inlet = inlet
        self._seed_fraction = seed_fraction
        self._rhs_calls = 0
        return inlet

    def _rhs(self, x: float, y: np.ndarray) -> np.ndarray:
        del x
        self._rhs_calls += 1
        if self._inlet is None or self._seed_fraction is None:
            raise RuntimeError("solver not initialized. call initialize_from_inlet first.")

        n_p = max(float(y[0]), 1e-20)
        T_e = max(float(y[1]), 1.0)

        c = evaluate_local_closure(
            n_p=n_p,
            T_e=T_e,
            m_0=self._inlet.m0,
            J_x0=self._inlet.Jx0,
            seed_fraction=self._seed_fraction,
            B=self.B,
            sigma_ep=SIGMA_EP,
        )

        _, J_y, _, _ = compute_currents_fields(c.v_p, c.n_e, c.beta, c.eta, c.Z)
        nu_E = c.eta * 2.0 * E_CHARGE * E_CHARGE * c.n_e / M_P

        # Linear system from Eq.(3)(4):
        # A*dn + B*dTp = C
        # D*dn + E*dTp = F
        A = -M_P * c.v_p * c.v_p + K_B * c.T_p
        Bcoef = K_B * n_p
        C = J_y * self.B

        D = -c.T_p
        Ecoef = 1.5 * n_p
        F_rhs = 1.5 * nu_E * c.n_e * (T_e - c.T_p) / _safe_pos(c.v_p)

        det = A * Ecoef - Bcoef * D
        if abs(det) < 1e-300:
            det = 1e-300 if det >= 0.0 else -1e-300

        dn_dx = (C * Ecoef - Bcoef * F_rhs) / det
        dTp_dx = (A * F_rhs - C * D) / det

        dTp_dTe = c.dTp_dTe
        if abs(dTp_dTe) < 1e-300:
            dTp_dTe = 1e-300 if dTp_dTe >= 0.0 else -1e-300

        dTe_dx = (dTp_dx - c.dTp_dnp * dn_dx) / dTp_dTe

        return np.array([dn_dx, dTe_dx], dtype=float)

    def _mach_value(self, y: np.ndarray) -> float:
        if self._inlet is None or self._seed_fraction is None:
            raise RuntimeError("solver not initialized.")

        n_p = max(float(y[0]), 1e-20)
        T_e = max(float(y[1]), 1.0)
        c = evaluate_local_closure(
            n_p=n_p,
            T_e=T_e,
            m_0=self._inlet.m0,
            J_x0=self._inlet.Jx0,
            seed_fraction=self._seed_fraction,
            B=self.B,
            sigma_ep=SIGMA_EP,
        )
        c_s = math.sqrt((5.0 / 3.0) * K_B * _safe_pos(c.T_p) / M_P)
        return c.v_p / _safe_pos(c_s)

    def _event_mach_low(self, x: float, y: np.ndarray) -> float:
        del x
        return self._mach_value(y) - 0.99

    def _event_mach_high(self, x: float, y: np.ndarray) -> float:
        del x
        return self._mach_value(y) - 1.01

    def solve(
        self,
        n_p_in: float,
        Z_in: float,
        T_p_in: float,
        T_e_in: float,
        seed_fraction: float,
        x_span: tuple[float, float] | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_step: Optional[float] = None,
    ) -> ForwardResultV6:
        inlet = self.initialize_from_inlet(
            n_p_in=n_p_in,
            Z_in=Z_in,
            T_p_in=T_p_in,
            T_e_in=T_e_in,
            seed_fraction=seed_fraction,
        )

        x0, x1 = (0.0, self.length) if x_span is None else (float(x_span[0]), float(x_span[1]))
        y0 = np.array([n_p_in, T_e_in], dtype=float)

        def ev_low(x: float, y: np.ndarray) -> float:
            return self._event_mach_low(x, y)

        def ev_high(x: float, y: np.ndarray) -> float:
            return self._event_mach_high(x, y)

        ev_low.terminal = True
        ev_low.direction = 0
        ev_high.terminal = True
        ev_high.direction = 0

        ivp_kwargs = dict(
            fun=self._rhs,
            t_span=(x0, x1),
            y0=y0,
            method="LSODA",
            rtol=rtol,
            atol=atol,
            events=[ev_low, ev_high],
            dense_output=False,
        )
        if max_step is not None:
            ivp_kwargs["max_step"] = float(max_step)

        ivp = solve_ivp(**ivp_kwargs)

        x = ivp.t
        n_p = ivp.y[0]
        T_e = ivp.y[1]

        N = len(x)
        T_p = np.empty(N, dtype=float)
        v_p = np.empty(N, dtype=float)
        n_e = np.empty(N, dtype=float)
        beta = np.empty(N, dtype=float)
        eta = np.empty(N, dtype=float)
        Z = np.empty(N, dtype=float)
        J_x = np.empty(N, dtype=float)
        J_y = np.empty(N, dtype=float)
        E_x = np.empty(N, dtype=float)
        mach = np.empty(N, dtype=float)

        for i in range(N):
            c = evaluate_local_closure(
                n_p=float(max(n_p[i], 1e-20)),
                T_e=float(max(T_e[i], 1.0)),
                m_0=inlet.m0,
                J_x0=inlet.Jx0,
                seed_fraction=seed_fraction,
                B=self.B,
                sigma_ep=SIGMA_EP,
            )
            jx_i, jy_i, ex_i, _ = compute_currents_fields(c.v_p, c.n_e, c.beta, c.eta, c.Z)

            T_p[i] = c.T_p
            v_p[i] = c.v_p
            n_e[i] = c.n_e
            beta[i] = c.beta
            eta[i] = c.eta
            Z[i] = c.Z
            J_x[i] = jx_i
            J_y[i] = jy_i
            E_x[i] = ex_i

            c_s = math.sqrt((5.0 / 3.0) * K_B * _safe_pos(c.T_p) / M_P)
            mach[i] = c.v_p / _safe_pos(c_s)

        event_name = None
        if len(ivp.t_events) >= 1 and len(ivp.t_events[0]) > 0:
            event_name = "mach_0p99"
        if len(ivp.t_events) >= 2 and len(ivp.t_events[1]) > 0:
            event_name = "mach_1p01"

        stats = {
            "rhs_calls": float(self._rhs_calls),
            "n_steps": float(N),
            "x_end": float(x[-1]) if N > 0 else float(x0),
            "m0": float(inlet.m0),
            "Jx0": float(inlet.Jx0),
        }

        return ForwardResultV6(
            success=bool(ivp.success),
            message=str(ivp.message),
            status=int(ivp.status),
            x=x,
            n_p=n_p,
            T_e=T_e,
            T_p=T_p,
            v_p=v_p,
            n_e=n_e,
            beta=beta,
            eta=eta,
            Z=Z,
            J_x=J_x,
            J_y=J_y,
            E_x=E_x,
            mach=mach,
            event_name=event_name,
            solver_stats=stats,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="V6 explicit MHD PDE solver")
    p.add_argument("--n-pin", type=float, required=True, help="inlet ion density n_p [m^-3]")
    p.add_argument("--z-in", type=float, required=True, help="inlet load factor Z")
    p.add_argument("--tp-in", type=float, required=True, help="inlet ion temperature T_p [K]")
    p.add_argument("--te-in", type=float, required=True, help="inlet electron temperature T_e [K]")
    p.add_argument("--seed-fraction", type=float, default=0.01, help="n_s / n_p")
    p.add_argument("--B", type=float, default=B_FIELD, help="magnetic field [T]")
    p.add_argument("--L", type=float, default=0.039, help="channel length [m]")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--max-step", type=float, default=None)
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()

    solver = ForwardPDESolverV6(B=args.B, length=args.L)
    out = solver.solve(
        n_p_in=args.n_pin,
        Z_in=args.z_in,
        T_p_in=args.tp_in,
        T_e_in=args.te_in,
        seed_fraction=args.seed_fraction,
        rtol=args.rtol,
        atol=args.atol,
        max_step=args.max_step,
    )

    i_last = len(out.x) - 1
    print(f"success={out.success} status={out.status} msg={out.message}")
    print(f"event={out.event_name}")
    if i_last >= 0:
        print(
            "x_end={:.6e} n_p_end={:.6e} Te_end={:.6e} Tp_end={:.6e} Z_end={:.6e} M_end={:.6f}".format(
                out.x[i_last],
                out.n_p[i_last],
                out.T_e[i_last],
                out.T_p[i_last],
                out.Z[i_last],
                out.mach[i_last],
            )
        )
    print("stats:", out.solver_stats)
    return 0 if out.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
