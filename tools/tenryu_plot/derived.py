"""Derived quantities for standard TENRYU 1D plots."""

from __future__ import annotations

import numpy as np


# Mirror of src/core/constants.hpp, verified in the P0 design.
A_EV = 1.3720e2
C_LIGHT = 2.99792458e10
EV_TO_ERG = 1.6022e-12


def cell_centers(x_r: np.ndarray) -> np.ndarray:
    return 0.5 * (x_r[:-1] + x_r[1:])


def cell_dr(x_r: np.ndarray) -> np.ndarray:
    return np.diff(x_r)


def radiation_temperature(energy_density_2d: np.ndarray) -> np.ndarray:
    total_energy_density = np.sum(np.asarray(energy_density_2d, dtype=np.float64), axis=1)
    return (total_energy_density / A_EV) ** 0.25


def total_pressure(Pe: np.ndarray, Pi: np.ndarray) -> np.ndarray:
    return np.asarray(Pe, dtype=np.float64) + np.asarray(Pi, dtype=np.float64)


def shock_radius(snapshot, window: tuple[float, float] | None = None) -> float | None:
    qvisc = snapshot.field("Qvisc")
    if qvisc is None:
        return None
    q = np.asarray(qvisc, dtype=np.float64)
    r = snapshot.r_centers
    if window is not None:
        r_lo, r_hi = window
        mask = (r >= r_lo) & (r <= r_hi)
        if not np.any(mask):
            return None
        q = q[mask]
        r = r[mask]
    if q.size == 0:
        return None
    max_q = float(np.nanmax(q))
    if not np.isfinite(max_q) or max_q <= 0.0:
        return None
    return float(r[int(np.nanargmax(q))])


def bang_time(history) -> float | None:
    for name in ("implosion/rho_peak", "implosion/center_temperature"):
        series = history.get(name)
        if series is None:
            continue
        values = np.asarray(series, dtype=np.float64)
        if values.ndim > 1:
            values = values[:, 0]
        if values.size == 0 or not np.any(np.isfinite(values)):
            continue
        return float(history.t[int(np.nanargmax(values))])
    return None


def rhoR(rho: np.ndarray, x_r: np.ndarray) -> float:
    return float(np.sum(np.asarray(rho, dtype=np.float64) * cell_dr(x_r)))
