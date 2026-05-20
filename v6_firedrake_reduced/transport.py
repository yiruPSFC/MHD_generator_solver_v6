from __future__ import annotations

from dataclasses import replace
from typing import Any

from .legacy_physics import normalize_working_fluid_profile


ELECTRON_TRANSPORT_E_HE = "e-He"
ELECTRON_TRANSPORT_E_ARGON = "e-Argon"
ELECTRON_TRANSPORT_MODELS = (ELECTRON_TRANSPORT_E_HE, ELECTRON_TRANSPORT_E_ARGON)

LXCAT_E_HE_MEDIAN_4300K_M2 = 6.450369478794445e-20
LXCAT_E_HE_MEDIAN_4300K_NOTE = (
    "LXCat e-He ELASTIC momentum-transfer median at 4300 K; "
    "used as the Firedrake Yamasaki He/Cs default"
)
LEGACY_E_ARGON_NOTE = "Legacy e-Argon-style electron-heavy momentum-transfer cross section."


def normalize_electron_transport(value: Any) -> str:
    text = str(value if value is not None else ELECTRON_TRANSPORT_E_HE).strip().lower()
    normalized = text.replace("_", "-").replace(" ", "")
    aliases = {
        "e-he": ELECTRON_TRANSPORT_E_HE,
        "ehe": ELECTRON_TRANSPORT_E_HE,
        "he": ELECTRON_TRANSPORT_E_HE,
        "lxcat-e-he-median-4300k": ELECTRON_TRANSPORT_E_HE,
        "e-argon": ELECTRON_TRANSPORT_E_ARGON,
        "eargon": ELECTRON_TRANSPORT_E_ARGON,
        "e-ar": ELECTRON_TRANSPORT_E_ARGON,
        "ear": ELECTRON_TRANSPORT_E_ARGON,
        "argon": ELECTRON_TRANSPORT_E_ARGON,
        "legacy": ELECTRON_TRANSPORT_E_ARGON,
    }
    if normalized not in aliases:
        raise ValueError(f"unknown electron_transport={value!r}; expected one of {ELECTRON_TRANSPORT_MODELS}")
    return aliases[normalized]


def working_fluid_for_config(config: Any):
    """Return the legacy working-fluid profile with the selected electron transport model."""

    fluid = normalize_working_fluid_profile(config.working_fluid_profile)
    metadata = dict(getattr(config, "metadata", {}) or {})
    legacy_keys = {"sigma_ep_override_m2", "sigma_ep_override_note", "sigma_ep_model"} & set(metadata)
    if legacy_keys:
        raise ValueError(
            "arbitrary sigma_ep metadata is no longer supported; set electron_transport to "
            f"{ELECTRON_TRANSPORT_E_HE!r} or {ELECTRON_TRANSPORT_E_ARGON!r}. "
            f"Unsupported keys: {sorted(legacy_keys)}"
        )

    electron_transport = normalize_electron_transport(metadata.get("electron_transport", ELECTRON_TRANSPORT_E_HE))
    if electron_transport == ELECTRON_TRANSPORT_E_HE:
        return replace(
            fluid,
            sigma_ep=LXCAT_E_HE_MEDIAN_4300K_M2,
            sigma_ep_note=LXCAT_E_HE_MEDIAN_4300K_NOTE,
        )
    return replace(fluid, sigma_ep=float(fluid.sigma_ep), sigma_ep_note=LEGACY_E_ARGON_NOTE)
