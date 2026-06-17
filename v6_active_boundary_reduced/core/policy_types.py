from __future__ import annotations

from typing import Protocol


class PhysicsParamsLike(Protocol):
    dot_N: float
    I_0: float
    seed_fraction: float
    B: float
    heavy_particle_mass_kg: float
    seed_ionization_energy_J: float
    sigma_ep: float
