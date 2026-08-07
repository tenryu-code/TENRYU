"""Shared plotting style and labels."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


try:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


FIELD_META: dict[str, tuple[str, str, bool]] = {
    "rho": ("ρ", "g/cm³", True),
    "Te": ("Tₑ", "eV", False),
    "Ti": ("Tᵢ", "eV", False),
    "Tr": ("T_r", "eV", False),
    "u": ("u_r", "cm/s", False),
    "P": ("P", "dyn/cm²", True),
    "Pe": ("Pₑ", "dyn/cm²", True),
    "Pi": ("Pᵢ", "dyn/cm²", True),
    "Qvisc": ("Qvisc", "dyn/cm²", False),
    "hot_e_Q": ("Q_hot-e", "erg/cm³/s", True),
    "hot_e_eps_cum": ("ε_preheat", "erg/g", True),
    "burn_rate": ("burn_rate", "1/cm³/s", True),
    "burn_Q_e": ("Q_burn,e", "erg/cm³/s", True),
    "burn_Q_i": ("Q_burn,i", "erg/cm³/s", True),
    "burn_eps_cum": ("ε_burn", "erg/g", True),
    "burn_n_D": ("n_D", "1/cm³", True),
    "burn_n_T": ("n_T", "1/cm³", True),
    "burn_n_He3": ("n_He3", "1/cm³", True),
    "burn_n_He4": ("n_He4", "1/cm³", True),
    "burn_n_p": ("n_p", "1/cm³", True),
    "zbar": ("zbar", "—", False),
    "ee": ("eₑ", "erg/g", True),
    "ei": ("eᵢ", "erg/g", True),
    "laser_dep": ("laser_dep", "erg/cm³/s", False),
}


def ensure_matplotlib() -> Any:
    if plt is None:
        raise RuntimeError("matplotlib is required. Install dependency: pip install matplotlib")
    return plt


def xunit_scale(xunit: str) -> tuple[float, str]:
    if xunit == "cm":
        return 1.0, "cm"
    if xunit == "um":
        return 1.0e4, "um"
    raise ValueError(f"unsupported xunit: {xunit}")


def time_unit(t_max_s: float) -> tuple[float, str]:
    """Pick a display unit for times up to |t_max_s| seconds.

    Returns (scale, unit) with display_value = t_seconds * scale.
    """
    t = abs(float(t_max_s))
    if t >= 0.1:
        return 1.0, "s"
    if t >= 1.0e-4:
        return 1.0e3, "ms"
    if t >= 1.0e-7:
        return 1.0e6, "us"
    if t >= 1.0e-10:
        return 1.0e9, "ns"
    return 1.0e12, "ps"


def format_time(t_seconds: float) -> str:
    scale, unit = time_unit(t_seconds)
    return f"{t_seconds * scale:.6g} {unit}"


def axis_label(field: str) -> str:
    label, unit, _ = FIELD_META[field]
    if unit == "—":
        return label
    return f"{label} [{unit}]"


def save_figure(fig: Any, out_path: str | Path, dpi: int) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    return path
