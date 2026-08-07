"""Exact Noh reference solution."""

from __future__ import annotations

import numpy as np


_GEOMETRY_M = {"planar": 0, "cylindrical": 1, "spherical": 2}


def shock_radius(t: float, v0: float = 1.0, gamma: float = 5.0 / 3.0) -> float:
    """Noh shock radius."""
    return 0.5 * (gamma - 1.0) * v0 * t


def density_plateau(geometry: str, rho0: float = 1.0, gamma: float = 5.0 / 3.0) -> float:
    """Post-shock density plateau."""
    m = _geometry_m(geometry)
    compression = (gamma + 1.0) / (gamma - 1.0)
    return rho0 * compression ** (m + 1)


def profile(
    r,
    t: float,
    geometry: str,
    rho0: float = 1.0,
    v0: float = 1.0,
    gamma: float = 5.0 / 3.0,
) -> dict[str, np.ndarray]:
    """Exact Noh profile.

    Provenance: generalized from src/verification/noh_analytic.cpp and Noh
    1987, J. Comput. Phys. 72, 78.
    """
    r_arr = np.asarray(r, dtype=np.float64)
    m = _geometry_m(geometry)
    rs = shock_radius(t, v0, gamma)
    rho_post = density_plateau(geometry, rho0, gamma)
    inside = r_arr < rs

    rho = np.empty_like(r_arr, dtype=np.float64)
    u = np.empty_like(r_arr, dtype=np.float64)
    p = np.empty_like(r_arr, dtype=np.float64)

    rho[inside] = rho_post
    u[inside] = 0.0
    p[inside] = 0.5 * (gamma - 1.0) * rho_post * v0 * v0

    if m == 0:
        rho[~inside] = rho0
    else:
        ratio = np.divide(
            v0 * t,
            r_arr,
            out=np.full_like(r_arr, np.inf, dtype=np.float64),
            where=r_arr != 0.0,
        )
        rho[~inside] = rho0 * (1.0 + ratio[~inside]) ** m
    u[~inside] = -v0
    p[~inside] = 0.0
    return {"rho": rho, "u": u, "p": p}


def _geometry_m(geometry: str) -> int:
    try:
        return _GEOMETRY_M[geometry]
    except KeyError as exc:
        raise ValueError(f"unsupported Noh geometry: {geometry}") from exc
