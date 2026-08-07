"""Frozen RMtV reference table adapter.

Provenance: parses src/verification/rmtv_reference_table.hpp at runtime and
uses the same dimensional mapping as src/drivers/cmd_verify.cpp
measure_rmtv_checkpoint for the Reinicke & Meyer-ter-Vehn 1991 similarity
solution.

This is the frozen dimensional instance for the rmtv_1d verification deck only;
it is valid only for the deck parameters baked into the header. The certified
similarity band is XiCertifiedLo..XiFront; the C++ L2 gate uses xi in
[0.40, 1.90].
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[3]
_HEADER_PATH = _REPO_ROOT / "src" / "verification" / "rmtv_reference_table.hpp"
_TABLE: dict[str, object] | None = None


def constants() -> dict[str, float]:
    """Return a copy of the parsed RMtV constants."""
    return dict(_load_table()["constants"])


def similarity_radius(t: float) -> float:
    """Return r_sim(t) = Zeta * t**Alpha."""
    c = _load_table()["constants"]
    return c["Zeta"] * float(t) ** c["Alpha"]


def shock_radius(t: float) -> float:
    """Return the dimensional RMtV shock radius."""
    c = _load_table()["constants"]
    return similarity_radius(t) * c["XiShock"]


def front_radius(t: float) -> float:
    """Return the dimensional RMtV heat-front radius."""
    c = _load_table()["constants"]
    return similarity_radius(t) * c["XiFront"]


def t_center_expected(t: float) -> float:
    """Return the C++ verify gate center-temperature expectation."""
    c = _load_table()["constants"]
    return (
        c["Theta0"]
        * (c["Alpha"] * c["Zeta"]) ** 2
        * float(t) ** (2.0 * (c["Alpha"] - 1.0))
        / c["GammaGas"]
    )


def profile(r, t: float) -> dict[str, np.ndarray]:
    """Return RMtV rho, Te, and xi arrays for radius r and time t."""
    if t <= 0.0:
        raise ValueError("t must be > 0")
    table = _load_table()
    c = table["constants"]
    arrays = table["arrays"]
    r_arr = np.asarray(r, dtype=np.float64)
    xi = r_arr / similarity_radius(t)
    theta = np.interp(xi, arrays["Xi"], arrays["Theta"])
    g = np.interp(xi, arrays["Xi"], arrays["G"])
    valid_r = r_arr > 0.0
    inside_front = xi <= c["XiFront"]

    with np.errstate(divide="ignore", invalid="ignore"):
        ambient = c["G0"] * r_arr ** c["KappaRho"]
        te = theta * (c["Alpha"] * r_arr / t) ** 2 / c["GammaGas"]
        rho = np.where(inside_front, g * ambient, ambient)
    te = np.where(inside_front, te, np.nan)
    te = np.where(valid_r, te, np.nan)
    rho = np.where(valid_r, rho, np.nan)
    return {"rho": rho, "Te": te, "xi": xi}


def _load_table() -> dict[str, object]:
    global _TABLE
    if _TABLE is None:
        _TABLE = _parse_header()
    return _TABLE


def _parse_header() -> dict[str, object]:
    if not _HEADER_PATH.is_file():
        raise RuntimeError("rmtv reference table header not found (tenryu_plot must live in the repo)")
    text = _HEADER_PATH.read_text(encoding="utf-8")
    constants_map = _parse_constants(text)
    arrays = _parse_arrays(text)
    return {"constants": constants_map, "arrays": arrays}


def _parse_constants(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        match = re.search(r"inline constexpr double k(\w+) = ([^;]+);", line)
        if match is None:
            continue
        name, raw = match.groups()
        out[name] = _eval_constant(raw)
    return out


def _eval_constant(raw: str) -> float:
    expr = raw.split("//", 1)[0].strip()
    try:
        return float(expr)
    except ValueError:
        pass
    match = re.match(r"^\s*([0-9.eE+-]+)\s*/\s*([0-9.eE+-]+)\s*$", expr)
    if match is None:
        raise RuntimeError(f"unsupported RMtV constant expression: {raw.strip()}")
    num, den = match.groups()
    return float(num) / float(den)


def _parse_arrays(text: str) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in ("Xi", "U", "Theta", "G", "W"):
        match = re.search(rf"k{name} = \{{(.*?)\}};", text, flags=re.DOTALL)
        if match is None:
            raise RuntimeError(f"RMtV array k{name} not found")
        values = [float(part.strip()) for part in match.group(1).split(",") if part.strip()]
        if len(values) != 2000:
            raise RuntimeError(f"RMtV array k{name} has {len(values)} entries, expected 2000")
        arrays[name] = np.asarray(values, dtype=np.float64)
    if not np.all(np.diff(arrays["Xi"]) > 0.0):
        raise RuntimeError("RMtV kXi grid is not strictly increasing")
    return arrays
