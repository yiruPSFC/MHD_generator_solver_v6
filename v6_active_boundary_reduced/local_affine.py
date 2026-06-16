from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .numba_physics import closure_state_numba, dynamic_terms_numba, g_state_and_gradients_numba
from .policy_types import PhysicsParamsLike


@dataclass(frozen=True)
class ForwardAffineCoefficients:
    a0: float
    a1: float
    b0: float
    b1: float
    det_D: float
    phi_current: float
    G_current: float
    T_p_current: float
    A_current: float
    logA_current: float
    n_prime_a0: float
    n_prime_a1: float
    Te_prime_a0: float
    Te_prime_a1: float
    f0_momentum: float
    f0_energy: float
    f1_momentum: float
    f1_energy: float


def compute_forward_affine_coefficients(
    *,
    n_p: float,
    T_e: float,
    A: float,
    logA: float,
    params: PhysicsParamsLike,
    log_gradient_eps: float = 1.0e-5,
    g_gradient_mode: str = "finite_difference",
) -> ForwardAffineCoefficients:
    """Compute local physical-forward affine coefficients at the current state.

    The primitive dynamics have the local form

        D * [n_p', T_e'] = f0 + A_prime * f1

    with A_prime = A*sigma.  This helper keeps those physical-forward
    coefficients distinct from the reverse-step coefficients used by policy.py.
    """

    n = float(n_p)
    te = float(T_e)
    area = float(A)
    if n <= 0.0 or te <= 0.0 or area <= 0.0:
        raise ValueError("n_p, T_e, and A must be positive for local affine coefficients.")

    terms = dynamic_terms_numba(
        n,
        te,
        area,
        0.0,
        float(params.dot_N),
        float(params.I_0),
        float(params.seed_fraction),
        float(params.B),
        float(params.heavy_particle_mass_kg),
        float(params.seed_ionization_energy_J),
        float(params.sigma_ep),
    )
    M11 = float(terms[0])
    M12 = float(terms[1])
    M13 = float(terms[2])
    E11 = float(terms[3])
    E12 = float(terms[4])
    E13 = float(terms[5])
    f0_momentum = float(terms[7])
    f0_energy = float(terms[8])
    det_D = float(terms[9])

    D = np.array([[M11, M12], [E11, E12]], dtype=float)
    f0 = np.array([f0_momentum, f0_energy], dtype=float)
    f1 = np.array([-M13, -E13], dtype=float)
    try:
        y0 = np.linalg.solve(D, f0)
        y1 = np.linalg.solve(D, f1)
    except np.linalg.LinAlgError:
        # RISK: Singular D should have been routed to sonic left-null compatibility before affine endpoint use.
        y0 = np.array([np.nan, np.nan], dtype=float)
        y1 = np.array([np.nan, np.nan], dtype=float)

    closure = closure_state_numba(
        n,
        te,
        area,
        float(params.dot_N),
        float(params.I_0),
        float(params.seed_fraction),
        float(params.B),
        float(params.heavy_particle_mass_kg),
        float(params.seed_ionization_energy_J),
        float(params.sigma_ep),
    )
    T_p = float(closure[10])
    dTp_dTe = float(closure[11])
    dTp_dn = float(closure[12])
    dTp_dA = float(closure[13])
    G = float(closure[18])
    phi = float(te / max(T_p, 1.0e-300) - 1.0)

    inv_Tp2 = 1.0 / max(T_p * T_p, 1.0e-300)
    dPhi_dn = -te * inv_Tp2 * dTp_dn
    dPhi_dTe = 1.0 / max(T_p, 1.0e-300) - te * inv_Tp2 * dTp_dTe
    dPhi_dA = -te * inv_Tp2 * dTp_dA
    a0 = float(dPhi_dn * y0[0] + dPhi_dTe * y0[1])
    a1 = float(dPhi_dn * y1[0] + dPhi_dTe * y1[1] + dPhi_dA)

    mode = str(g_gradient_mode).strip().lower().replace("-", "_")
    if mode in {"analytic", "analytical"}:
        dG_dn, dG_dTe, dG_dA = _closure_G_gradients_analytic(n_p=n, T_e=te, A=area, params=params)
    elif mode in {"finite_difference", "central_difference", "centered_difference"}:
        dG_dn, dG_dTe, dG_dA = _closure_G_gradients(
            n_p=n,
            T_e=te,
            A=area,
            params=params,
            log_gradient_eps=float(log_gradient_eps),
        )
    else:
        raise ValueError("g_gradient_mode must be 'finite_difference' or 'analytic'.")
    b0 = float(dG_dn * y0[0] + dG_dTe * y0[1])
    b1 = float(dG_dn * y1[0] + dG_dTe * y1[1] + dG_dA)

    return ForwardAffineCoefficients(
        a0=a0,
        a1=a1,
        b0=b0,
        b1=b1,
        det_D=det_D,
        phi_current=phi,
        G_current=G,
        T_p_current=T_p,
        A_current=area,
        logA_current=float(logA),
        n_prime_a0=float(y0[0]),
        n_prime_a1=float(y1[0]),
        Te_prime_a0=float(y0[1]),
        Te_prime_a1=float(y1[1]),
        f0_momentum=f0_momentum,
        f0_energy=f0_energy,
        f1_momentum=float(f1[0]),
        f1_energy=float(f1[1]),
    )


def _closure_G_gradients(
    *,
    n_p: float,
    T_e: float,
    A: float,
    params: PhysicsParamsLike,
    log_gradient_eps: float,
) -> tuple[float, float, float]:
    eps = max(float(log_gradient_eps), 1.0e-8)
    log_n = float(np.log(max(float(n_p), 1.0e-300)))
    log_te = float(np.log(max(float(T_e), 1.0e-300)))
    log_A = float(np.log(max(float(A), 1.0e-300)))

    def G_at(log_n_value: float, log_te_value: float, log_A_value: float) -> float:
        return float(
            closure_state_numba(
                float(np.exp(np.clip(log_n_value, -700.0, 700.0))),
                float(np.exp(np.clip(log_te_value, -700.0, 700.0))),
                float(np.exp(np.clip(log_A_value, -700.0, 700.0))),
                float(params.dot_N),
                float(params.I_0),
                float(params.seed_fraction),
                float(params.B),
                float(params.heavy_particle_mass_kg),
                float(params.seed_ionization_energy_J),
                float(params.sigma_ep),
            )[18]
        )

    dG_dlogn = (G_at(log_n + eps, log_te, log_A) - G_at(log_n - eps, log_te, log_A)) / (2.0 * eps)
    dG_dlogte = (G_at(log_n, log_te + eps, log_A) - G_at(log_n, log_te - eps, log_A)) / (2.0 * eps)
    dG_dlogA = (G_at(log_n, log_te, log_A + eps) - G_at(log_n, log_te, log_A - eps)) / (2.0 * eps)
    return (
        float(dG_dlogn / max(float(n_p), 1.0e-300)),
        float(dG_dlogte / max(float(T_e), 1.0e-300)),
        float(dG_dlogA / max(float(A), 1.0e-300)),
    )


def _closure_G_gradients_analytic(
    *,
    n_p: float,
    T_e: float,
    A: float,
    params: PhysicsParamsLike,
) -> tuple[float, float, float]:
    _, dG_dn, dG_dTe, dG_dA = g_state_and_gradients_numba(
        float(n_p),
        float(T_e),
        float(A),
        float(params.dot_N),
        float(params.I_0),
        float(params.seed_fraction),
        float(params.B),
        float(params.heavy_particle_mass_kg),
        float(params.seed_ionization_energy_J),
        float(params.sigma_ep),
    )
    return float(dG_dn), float(dG_dTe), float(dG_dA)
