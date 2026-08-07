"""Reusable radiative-shock verification metrics for 2D RZ z-slab gates.

This module implements the Wave 12 shock-windowed, cell-averaged,
convolved-reference metric framework in a reusable form for I1-I7 2D RZ
production verification harnesses.  It operates on 1D z-profiles, typically
produced from 2D RZ snapshots by ``rz_profile_average.py``.

The validation directory is used as a flat Python path in TENRYU.  Import this
module with, for example:

    sys.path.insert(0, "tools/validation")
    from radiative_shock_metrics import shock_windowed_l2
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple, Sequence

import numpy as np


TINY = 1.0e-300

__all__ = [
    "ShockFront",
    "detect_shock_front",
    "measure_hydro_kernel",
    "cell_averaged_convolved_reference",
    "shock_windowed_l2",
    "richardson_d_infinity_fit",
    "convolved_peak_gap",
    "conservation_residual_closed",
    "conservation_residual_open",
    "no_escape_valve_check",
]


class ShockFront(NamedTuple):
    """Detected shock-front metadata.

    Attributes
    ----------
    z_shock : float
        Detected shock-front position in cm.
    idx_shock : int
        Index of the nearest input z cell center.
    peak_gradient : float
        Peak absolute electron-temperature gradient in eV/cm.
    """

    z_shock: float
    idx_shock: int
    peak_gradient: float


def _trapz(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def _trapz_float(y: np.ndarray, x: np.ndarray) -> float:
    return float(_trapz(np.asarray(y, dtype=float), np.asarray(x, dtype=float)))


def _as_1d_float(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr


def _finite_sorted_xy(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    x_arr = _as_1d_float(x, "x")
    y_arr = _as_1d_float(y, "y")
    if x_arr.size != y_arr.size:
        raise ValueError(f"x and y sizes differ: {x_arr.size} != {y_arr.size}")
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size == 0:
        return x_arr, y_arr
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    keep = np.concatenate(([True], np.diff(x_arr) > 0.0))
    return x_arr[keep], y_arr[keep]


def _uniform_kernel(half_width_cm: float) -> tuple[np.ndarray, np.ndarray]:
    width = max(abs(float(half_width_cm)), 1.0)
    x = np.asarray([-0.5 * width, 0.5 * width], dtype=float)
    k = np.full(2, 1.0 / width, dtype=float)
    return x, k


def _normalize_kernel(kernel_z: Any, kernel: Any) -> tuple[np.ndarray, np.ndarray]:
    z_arr, k_arr = _finite_sorted_xy(kernel_z, np.maximum(np.asarray(kernel, dtype=float), 0.0))
    if z_arr.size < 2:
        return z_arr, k_arr
    integral = _trapz_float(k_arr, z_arr)
    if not (math.isfinite(integral) and integral > 0.0):
        span = float(z_arr[-1] - z_arr[0])
        if span <= TINY:
            return z_arr[:1], np.asarray([1.0], dtype=float)
        k_arr = np.full_like(z_arr, 1.0 / span)
        integral = _trapz_float(k_arr, z_arr)
    return z_arr, k_arr / max(integral, TINY)


def _cell_edges_from_centers(z: np.ndarray) -> np.ndarray:
    if z.size == 0:
        return np.asarray([], dtype=float)
    if z.size == 1:
        return np.asarray([z[0] - 0.5, z[0] + 0.5], dtype=float)
    edges = np.empty(z.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (z[:-1] + z[1:])
    edges[0] = z[0] - 0.5 * (z[1] - z[0])
    edges[-1] = z[-1] + 0.5 * (z[-1] - z[-2])
    return edges


def _convolved_reference_at(
    ref_z: np.ndarray,
    ref_dtr: np.ndarray,
    kernel_z: np.ndarray,
    kernel: np.ndarray,
    query_z: np.ndarray,
) -> np.ndarray:
    query = np.asarray(query_z, dtype=float)
    out = np.full(query.shape, np.nan, dtype=float)
    ref_z_arr, ref_arr = _finite_sorted_xy(ref_z, ref_dtr)
    if ref_z_arr.size == 0:
        return out
    kernel_z_arr, kernel_arr = _normalize_kernel(kernel_z, kernel)
    if ref_z_arr.size < 2 or kernel_z_arr.size < 2:
        return np.interp(query, ref_z_arr, ref_arr, left=np.nan, right=np.nan)
    if float(kernel_z_arr[-1] - kernel_z_arr[0]) <= TINY:
        return np.interp(query, ref_z_arr, ref_arr, left=np.nan, right=np.nan)

    for start in range(0, query.size, 128):
        stop = min(start + 128, query.size)
        q = query[start:stop]
        offsets = q[:, None] - ref_z_arr[None, :]
        weights = np.interp(offsets, kernel_z_arr, kernel_arr, left=0.0, right=0.0)
        out[start:stop] = _trapz(ref_arr[None, :] * weights, ref_z_arr, axis=1)
    return out


def _eligible_internal_positions(size: int, boundary_exclusion_cells: int) -> np.ndarray:
    eligible = np.ones(size, dtype=bool)
    edge = max(int(boundary_exclusion_cells), 0)
    if edge > 0 and size > 2 * edge:
        eligible[:edge] = False
        eligible[-edge:] = False
    return eligible


def _dedupe_sorted_profiles(
    z: np.ndarray,
    profiles: Sequence[np.ndarray],
    finite: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    finite_indices = np.nonzero(finite)[0]
    if finite_indices.size == 0:
        empty_profiles = [np.asarray([], dtype=float) for _ in profiles]
        return finite_indices, np.asarray([], dtype=float), empty_profiles
    order = np.argsort(z[finite_indices])
    sorted_indices = finite_indices[order]
    z_sorted = z[sorted_indices]
    keep = np.concatenate(([True], np.diff(z_sorted) > 0.0))
    sorted_indices = sorted_indices[keep]
    return sorted_indices, z[sorted_indices], [profile[sorted_indices] for profile in profiles]


def detect_shock_front(
    Te_profile: np.ndarray,
    z_centers: np.ndarray,
    rho_profile: np.ndarray | None = None,
    uz_profile: np.ndarray | None = None,
    *,
    boundary_exclusion_cells: int = 0,
    co_location_cells: int = 3,
) -> ShockFront:
    """Locate the shock front from internal, optionally hydro-consistent jumps.

    Parameters
    ----------
    Te_profile : np.ndarray
        One-dimensional electron-temperature profile in eV.
    z_centers : np.ndarray
        One-dimensional cell-center coordinates in cm.
    rho_profile : np.ndarray, optional
        One-dimensional density profile.  When supplied with ``uz_profile``,
        the selected front must have Te, rho, and u_z gradient peaks co-located
        within ``co_location_cells``.
    uz_profile : np.ndarray, optional
        One-dimensional axial velocity profile.
    boundary_exclusion_cells : int
        Number of cells to exclude at each z boundary before ranking front
        candidates.
    co_location_cells : int
        Maximum index spread allowed among Te, rho, and u_z local gradient
        peaks for hydro-consistent detection.

    Returns
    -------
    ShockFront
        Position, nearest input index, and selected ``abs(dT_e/dz)`` in eV/cm.
    """
    Te = _as_1d_float(Te_profile, "Te_profile")
    z = _as_1d_float(z_centers, "z_centers")
    if Te.size != z.size:
        raise ValueError(f"Te_profile and z_centers sizes differ: {Te.size} != {z.size}")

    hydro_profiles: list[np.ndarray] = []
    if rho_profile is not None or uz_profile is not None:
        if rho_profile is None or uz_profile is None:
            raise ValueError("rho_profile and uz_profile must be supplied together")
        rho = _as_1d_float(rho_profile, "rho_profile")
        uz = _as_1d_float(uz_profile, "uz_profile")
        if rho.size != z.size or uz.size != z.size:
            raise ValueError("rho_profile, uz_profile, and z_centers must have the same size")
        hydro_profiles = [rho, uz]

    profiles = [Te] + hydro_profiles
    finite = np.isfinite(z)
    for profile in profiles:
        finite &= np.isfinite(profile)
    finite_indices, z_sorted, sorted_profiles = _dedupe_sorted_profiles(z, profiles, finite)
    if finite_indices.size == 0:
        return ShockFront(z_shock=math.nan, idx_shock=-1, peak_gradient=math.nan)

    if z_sorted.size < 3:
        idx = int(finite_indices[0])
        return ShockFront(z_shock=float(z[idx]), idx_shock=idx, peak_gradient=0.0)

    gradients = [np.abs(np.gradient(profile, z_sorted)) for profile in sorted_profiles]
    for grad in gradients:
        grad[~np.isfinite(grad)] = 0.0
    eligible = _eligible_internal_positions(z_sorted.size, boundary_exclusion_cells)

    if not hydro_profiles:
        grad = gradients[0].copy()
        grad[~eligible] = -math.inf
        local = int(np.nanargmax(grad))
        z_shock = float(z_sorted[local])
        idx = int(finite_indices[local])
        return ShockFront(z_shock=z_shock, idx_shock=idx, peak_gradient=float(gradients[0][local]))

    candidate_positions = np.nonzero(eligible)[0]
    if candidate_positions.size == 0:
        return ShockFront(z_shock=math.nan, idx_shock=-1, peak_gradient=math.nan)

    requested_edge = max(int(boundary_exclusion_cells), 0)
    edge = requested_edge if z_sorted.size > 2 * requested_edge else 0
    coloc = max(int(co_location_cells), 0)
    normalizers = [max(float(np.max(grad[eligible])), TINY) for grad in gradients]
    best: tuple[float, float, int, list[int]] | None = None
    for center in candidate_positions:
        lo = max(edge, int(center) - coloc)
        hi = min(z_sorted.size - edge, int(center) + coloc + 1)
        if hi <= lo:
            continue
        peak_positions: list[int] = []
        strengths: list[float] = []
        for grad, normalizer in zip(gradients, normalizers):
            local = grad[lo:hi]
            peak = lo + int(np.argmax(local))
            peak_positions.append(peak)
            strengths.append(float(grad[peak]) / normalizer)
        if max(peak_positions) - min(peak_positions) > coloc:
            continue
        score = min(strengths)
        mean_strength = float(sum(strengths) / len(strengths))
        better_score = score > best[0] if best is not None else True
        better_tie = (
            best is not None
            and score == best[0]
            and (mean_strength > best[1] or (mean_strength == best[1] and center < best[2]))
        )
        if best is None or better_score or better_tie:
            best = (score, mean_strength, int(center), peak_positions)

    if best is None:
        return ShockFront(z_shock=math.nan, idx_shock=-1, peak_gradient=math.nan)

    peak_positions = best[3]
    local = int(round(float(np.median(np.asarray(peak_positions, dtype=float)))))
    return ShockFront(
        z_shock=float(z_sorted[local]),
        idx_shock=int(finite_indices[local]),
        peak_gradient=float(gradients[0][peak_positions[0]]),
    )


def measure_hydro_kernel(
    Te_profile: np.ndarray,
    z_centers: np.ndarray,
    z_shock: float,
    half_width_cm: float = 1.0,
) -> dict[str, Any]:
    """Measure a normalized hydro shock kernel from ``abs(dT_e/dz)`` near the shock.

    Parameters
    ----------
    Te_profile : np.ndarray
        One-dimensional electron-temperature profile in eV.
    z_centers : np.ndarray
        One-dimensional cell-center coordinates in cm.
    z_shock : float
        Shock-front position in cm, usually from :func:`detect_shock_front`.
    half_width_cm : float
        Half-width of the extraction window around ``z_shock`` in cm.

    Returns
    -------
    dict[str, Any]
        JSON-compatible metadata containing relative ``kernel_z`` coordinates
        in cm, normalized ``kernel`` values in 1/cm, normalization ``integral``,
        and ``n_cells``.
    """
    Te = _as_1d_float(Te_profile, "Te_profile")
    z = _as_1d_float(z_centers, "z_centers")
    if Te.size != z.size:
        raise ValueError(f"Te_profile and z_centers sizes differ: {Te.size} != {z.size}")
    finite = np.isfinite(Te) & np.isfinite(z)
    if np.count_nonzero(finite) < 3:
        kernel_z, kernel = _uniform_kernel(half_width_cm)
        return {
            "kernel_z": kernel_z,
            "kernel": kernel,
            "z_shock": float(z_shock),
            "half_width_cm": float(half_width_cm),
            "integral": _trapz_float(kernel, kernel_z),
            "n_cells": int(kernel_z.size),
        }

    mask = finite & (np.abs(z - float(z_shock)) <= abs(float(half_width_cm)))
    if np.count_nonzero(mask) < 3:
        candidates = np.nonzero(finite)[0]
        nearest = candidates[np.argsort(np.abs(z[candidates] - float(z_shock)))[:3]]
        mask = np.zeros_like(finite, dtype=bool)
        mask[np.sort(nearest)] = True

    z_local, Te_local = _finite_sorted_xy(z[mask] - float(z_shock), Te[mask])
    if z_local.size < 3:
        kernel_z, kernel = _uniform_kernel(half_width_cm)
    else:
        kernel_z, kernel = _normalize_kernel(z_local, np.abs(np.gradient(Te_local, z_local)))
        if kernel_z.size < 2:
            kernel_z, kernel = _uniform_kernel(half_width_cm)
    integral = _trapz_float(kernel, kernel_z) if kernel_z.size >= 2 else 1.0
    return {
        "kernel_z": kernel_z,
        "kernel": kernel,
        "z_shock": float(z_shock),
        "half_width_cm": float(half_width_cm),
        "integral": integral,
        "n_cells": int(kernel_z.size),
    }


def cell_averaged_convolved_reference(
    ref_z: np.ndarray,
    ref_dTr: np.ndarray,
    kernel_z: np.ndarray,
    kernel: np.ndarray,
    target_z: np.ndarray,
) -> np.ndarray:
    """Convolve a steady 1D reference profile with a measured kernel and cell-average it.

    Parameters
    ----------
    ref_z : np.ndarray
        Reference coordinates in cm.
    ref_dTr : np.ndarray
        Reference ``T_rad / T_e - 1`` values at ``ref_z``.
    kernel_z : np.ndarray
        Relative kernel coordinates in cm, centered on zero.
    kernel : np.ndarray
        Normalized kernel values in 1/cm.
    target_z : np.ndarray
        Target TENRYU z cell centers in cm.

    Returns
    -------
    np.ndarray
        Convolved reference profile sampled as cell averages on ``target_z``.
    """
    target = _as_1d_float(target_z, "target_z")
    kernel_z_arr, kernel_arr = _normalize_kernel(kernel_z, kernel)
    if kernel_z_arr.size < 2 or (
        kernel_z_arr.size > 0 and float(kernel_z_arr[-1] - kernel_z_arr[0]) <= TINY
    ):
        ref_z_arr, ref_arr = _finite_sorted_xy(ref_z, ref_dTr)
        if ref_z_arr.size == 0:
            return np.full(target.shape, np.nan, dtype=float)
        return np.interp(target, ref_z_arr, ref_arr, left=np.nan, right=np.nan)
    if target.size < 2:
        return _convolved_reference_at(ref_z, ref_dTr, kernel_z_arr, kernel_arr, target)
    edges = _cell_edges_from_centers(target)
    left = _convolved_reference_at(ref_z, ref_dTr, kernel_z_arr, kernel_arr, edges[:-1])
    center = _convolved_reference_at(ref_z, ref_dTr, kernel_z_arr, kernel_arr, target)
    right = _convolved_reference_at(ref_z, ref_dTr, kernel_z_arr, kernel_arr, edges[1:])
    return (left + 4.0 * center + right) / 6.0


def shock_windowed_l2(
    code_dTr: np.ndarray,
    ref_dTr_conv: np.ndarray,
    z: np.ndarray,
    z_shock: float,
    mfp_post_cm: float,
    N: int = 10,
) -> dict[str, Any]:
    """Compute shock-windowed L2 metrics for code-minus-reference dTr profiles.

    Parameters
    ----------
    code_dTr : np.ndarray
        Code profile of ``T_rad / T_e - 1``.
    ref_dTr_conv : np.ndarray
        Convolved and cell-averaged reference ``dTr`` profile.
    z : np.ndarray
        Target z cell centers in cm.
    z_shock : float
        Shock-front position in cm.
    mfp_post_cm : float
        Post-shock mean free path in cm.
    N : int
        Half-window size in units of ``mfp_post_cm``.

    Returns
    -------
    dict[str, Any]
        JSON-compatible metrics containing absolute L2, peak-normalized L2,
        RMS-relative L2, window bounds, and cell counts.
    """
    code = _as_1d_float(code_dTr, "code_dTr")
    ref = _as_1d_float(ref_dTr_conv, "ref_dTr_conv")
    z_arr = _as_1d_float(z, "z")
    if code.size != ref.size or code.size != z_arr.size:
        raise ValueError("code_dTr, ref_dTr_conv, and z must have the same size")
    window_half = float(N) * float(mfp_post_cm)
    base = {
        "L2_abs": math.inf,
        "L2_norm": math.inf,
        "L2_rel_rms": math.inf,
        "max_dTr_in_window": 0.0,
        "ref_dTr_max_in_window": 0.0,
        "window_half_cm": window_half,
        "window_min_cm": float(z_shock) - window_half,
        "window_max_cm": float(z_shock) + window_half,
        "n_cells_in_window": 0,
    }
    if not (math.isfinite(window_half) and window_half > 0.0 and math.isfinite(float(z_shock))):
        return base
    mask = (
        np.isfinite(z_arr)
        & np.isfinite(code)
        & np.isfinite(ref)
        & (np.abs(z_arr - float(z_shock)) <= window_half)
    )
    count = int(np.count_nonzero(mask))
    if count < 3:
        base["n_cells_in_window"] = count
        return base

    diff = code[mask] - ref[mask]
    l2_abs = float(np.sqrt(np.mean(diff * diff)))
    ref_peak = float(np.max(np.abs(ref[mask])))
    ref_rms = float(np.sqrt(np.mean(ref[mask] * ref[mask])))
    code_peak = float(np.max(np.abs(code[mask])))
    return {
        "L2_abs": l2_abs,
        "L2_norm": 0.0 if l2_abs <= TINY and ref_peak <= TINY else l2_abs / max(ref_peak, TINY),
        "L2_rel_rms": 0.0 if l2_abs <= TINY and ref_rms <= TINY else l2_abs / max(ref_rms, TINY),
        "max_dTr_in_window": code_peak,
        "ref_dTr_max_in_window": ref_peak,
        "window_half_cm": window_half,
        "window_min_cm": float(z_shock) - window_half,
        "window_max_cm": float(z_shock) + window_half,
        "n_cells_in_window": count,
    }


def richardson_d_infinity_fit(
    d_values: Sequence[float],
    dx_values: Sequence[float],
    model: str = "rational",
) -> dict[str, Any]:
    """Fit ``D(dx) = D_infinity / (1 + C * dx)`` by a linear fit of ``1 / D``.

    Parameters
    ----------
    d_values : Sequence[float]
        Measured D values, ordered on any positive grid spacing sequence.
    dx_values : Sequence[float]
        Positive grid spacings in cm corresponding to ``d_values``.
    model : str
        Fit model name.  Only ``"rational"`` is supported.

    Returns
    -------
    dict[str, Any]
        JSON-compatible fit diagnostics including ``D_infinity``, ``C``,
        ``observed_order``, ``predicted_at_dx_max``, residual norm, and point
        count.
    """
    if model != "rational":
        raise ValueError(f"unsupported Richardson model: {model}")
    d_arr = _as_1d_float(d_values, "d_values")
    dx_arr = _as_1d_float(dx_values, "dx_values")
    if d_arr.size != dx_arr.size:
        raise ValueError(f"d_values and dx_values sizes differ: {d_arr.size} != {dx_arr.size}")
    mask = np.isfinite(d_arr) & np.isfinite(dx_arr) & (d_arr > 0.0) & (dx_arr > 0.0)
    if np.count_nonzero(mask) < 2:
        raise ValueError("at least two finite positive D(dx) points are required")
    d_fit = d_arr[mask]
    dx_fit = dx_arr[mask]
    slope, intercept = np.polyfit(dx_fit, 1.0 / d_fit, 1)
    if not (math.isfinite(float(intercept)) and intercept > 0.0):
        raise ValueError("Richardson fit produced a nonpositive 1/D_infinity intercept")
    d_infinity = float(1.0 / intercept)
    c_coeff = float(slope / intercept)
    predicted = d_infinity / (1.0 + c_coeff * dx_fit)
    residuals = d_fit - predicted
    errors = np.abs(d_fit - d_infinity)
    order_mask = np.isfinite(errors) & (errors > TINY) & np.isfinite(dx_fit) & (dx_fit > 0.0)
    if np.count_nonzero(order_mask) >= 2:
        observed_order = float(np.polyfit(np.log(dx_fit[order_mask]), np.log(errors[order_mask]), 1)[0])
    else:
        observed_order = math.nan
    dx_max = float(np.max(dx_fit))
    return {
        "model": model,
        "D_infinity": d_infinity,
        "C": c_coeff,
        "observed_order": observed_order,
        "predicted_at_dx_max": float(d_infinity / (1.0 + c_coeff * dx_max)),
        "fit_residual_linf": float(np.max(np.abs(residuals))),
        "n_points": int(np.count_nonzero(mask)),
    }


def convolved_peak_gap(code_dTr_max: float, ref_dTr_conv_max: float) -> float:
    """Return the relative convolved-peak gap.

    Parameters
    ----------
    code_dTr_max : float
        Maximum absolute code ``dTr`` in the comparison window.
    ref_dTr_conv_max : float
        Maximum absolute convolved-reference ``dTr`` in the comparison window.

    Returns
    -------
    float
        ``abs(code_dTr_max - ref_dTr_conv_max) / max(abs(ref_dTr_conv_max), TINY)``.
    """
    return float(abs(float(code_dTr_max) - float(ref_dTr_conv_max)) / max(abs(float(ref_dTr_conv_max)), TINY))


def _raw_snapshot_value(snapshot: Any, *names: str) -> Any:
    for name in names:
        if isinstance(snapshot, dict) and name in snapshot:
            return snapshot[name]
        if "/" not in name and hasattr(snapshot, name):
            return getattr(snapshot, name)
        try:
            if name in snapshot:
                return snapshot[name]
        except Exception:
            pass
        attrs = getattr(snapshot, "attrs", None)
        if attrs is not None and name in attrs:
            return attrs[name]
    raise KeyError(names[0])


def _array_value(snapshot: Any, *names: str) -> np.ndarray | None:
    try:
        value = _raw_snapshot_value(snapshot, *names)
    except KeyError:
        return None
    try:
        value = value[()]
    except Exception:
        pass
    return np.asarray(value, dtype=float)


def _sum_density_times_vol(density: np.ndarray | None, vol: np.ndarray | None) -> float | None:
    if density is None or vol is None:
        return None
    den = np.asarray(density, dtype=float)
    weights = np.asarray(vol, dtype=float)
    if den.ndim == weights.ndim + 1:
        den = np.sum(den, axis=-1)
    if den.shape != weights.shape:
        if den.size == weights.size:
            den = den.reshape(weights.shape)
        else:
            return None
    return float(np.sum(den * weights))


def _kinetic_energy(snapshot: Any) -> float | None:
    node_mass = _array_value(snapshot, "node_mass", "hydro/node_mass")
    v_r = _array_value(snapshot, "v_r_nodes", "mesh/v_r")
    v_z = _array_value(snapshot, "v_z_nodes", "mesh/v_z")
    if node_mass is None or v_r is None or v_z is None:
        return None
    if node_mass.shape != v_r.shape and node_mass.size == v_r.size:
        node_mass = node_mass.reshape(v_r.shape)
    if v_z.shape != v_r.shape and v_z.size == v_r.size:
        v_z = v_z.reshape(v_r.shape)
    if node_mass.shape != v_r.shape or v_z.shape != v_r.shape:
        return None
    return float(0.5 * np.sum(node_mass * (v_r * v_r + v_z * v_z)))


def _snapshot_total(snapshot: Any, quantity: str) -> float:
    quantity_key = quantity.lower()
    vol = _array_value(snapshot, "vol", "hydro/vol", "cell_volume", "cell_volumes")
    if quantity_key == "mass":
        rho = _array_value(snapshot, "rho", "hydro/rho")
        total = _sum_density_times_vol(rho, vol)
        if total is None:
            raise ValueError("mass residual requires rho and vol fields")
        return total

    direct_names = (
        quantity,
        f"{quantity}_total",
        f"total_{quantity}",
        "total_energy",
        "E_total",
        "total_energy_erg",
    )
    for name in direct_names:
        arr = _array_value(snapshot, name)
        if arr is not None and arr.size == 1:
            return float(arr.reshape(-1)[0])

    if quantity_key not in ("energy", "total_energy"):
        raise ValueError(f"unsupported conservation quantity: {quantity}")

    total = 0.0
    found = False
    rad_e = _array_value(snapshot, "rad_E", "E_rad", "radiation/energy_density")
    rad_total = _sum_density_times_vol(rad_e, vol)
    if rad_total is not None:
        total += rad_total
        found = True

    rho = _array_value(snapshot, "rho", "hydro/rho")
    ee = _array_value(snapshot, "ee", "hydro/ee")
    ei = _array_value(snapshot, "ei", "hydro/ei")
    if rho is not None and vol is not None and ee is not None and ei is not None:
        if rho.shape != vol.shape and rho.size == vol.size:
            rho = rho.reshape(vol.shape)
        if ee.shape != vol.shape and ee.size == vol.size:
            ee = ee.reshape(vol.shape)
        if ei.shape != vol.shape and ei.size == vol.size:
            ei = ei.reshape(vol.shape)
        if rho.shape == vol.shape and ee.shape == vol.shape and ei.shape == vol.shape:
            total += float(np.sum(rho * (ee + ei) * vol))
            found = True

    ke = _kinetic_energy(snapshot)
    if ke is not None:
        total += ke
        found = True
    if not found:
        raise ValueError("energy residual requires total_energy or compatible rho/ee/ei/rad_E/vol fields")
    return total


def _relative_residual(numerator: float, reference: float) -> float:
    return float(abs(float(numerator)) / max(abs(float(reference)), TINY))


def conservation_residual_closed(
    snapshot_initial: Any,
    snapshot_final: Any,
    quantity: str = "energy",
) -> dict[str, Any]:
    """Compute a closed-domain conservation residual for mass or total energy.

    Parameters
    ----------
    snapshot_initial : Any
        Initial snapshot object, dict, or HDF5 handle.
    snapshot_final : Any
        Final snapshot object, dict, or HDF5 handle.
    quantity : str
        Conserved quantity.  Supported values are ``"mass"``, ``"energy"``,
        and ``"total_energy"``.

    Returns
    -------
    dict[str, Any]
        JSON-compatible totals and residuals with
        ``residual_rel = abs(final - initial) / abs(initial)``.
    """
    initial = _snapshot_total(snapshot_initial, quantity)
    final = _snapshot_total(snapshot_final, quantity)
    delta = final - initial
    return {
        "quantity": quantity,
        "initial": float(initial),
        "final": float(final),
        "boundary_integral": 0.0,
        "residual_abs": float(abs(delta)),
        "residual_rel": _relative_residual(delta, initial),
    }


def _boundary_flux_integral(boundary_fluxes: Any) -> float:
    if boundary_fluxes is None:
        return 0.0
    if isinstance(boundary_fluxes, (int, float, np.number)):
        return float(boundary_fluxes)
    if isinstance(boundary_fluxes, dict):
        for key in (
            "boundary_integral",
            "integrated_boundary_flux",
            "net_boundary_integral",
            "net_flux_integral",
        ):
            if key in boundary_fluxes:
                return _boundary_flux_integral(boundary_fluxes[key])
        time = boundary_fluxes.get("time_s", boundary_fluxes.get("t"))
        flux = boundary_fluxes.get("flux", boundary_fluxes.get("net_flux", boundary_fluxes.get("boundary_flux")))
        if time is not None and flux is not None:
            return _trapz_float(np.asarray(flux, dtype=float), np.asarray(time, dtype=float))
        total = 0.0
        for value in boundary_fluxes.values():
            try:
                arr = np.asarray(value, dtype=float)
            except Exception:
                continue
            if arr.size == 1:
                total += float(arr.reshape(-1)[0])
        return total
    if isinstance(boundary_fluxes, (tuple, list)) and len(boundary_fluxes) == 2:
        first = np.asarray(boundary_fluxes[0], dtype=float)
        second = np.asarray(boundary_fluxes[1], dtype=float)
        if first.ndim == 1 and second.ndim == 1 and first.size == second.size and first.size > 1:
            return _trapz_float(second, first)
    arr = np.asarray(boundary_fluxes, dtype=float)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return float(np.sum(arr))


def conservation_residual_open(
    snapshot_initial: Any,
    snapshot_final: Any,
    boundary_fluxes: Any,
    quantity: str = "energy",
) -> dict[str, Any]:
    """Compute an open-domain conservation residual including boundary fluxes.

    Parameters
    ----------
    snapshot_initial : Any
        Initial snapshot object, dict, or HDF5 handle.
    snapshot_final : Any
        Final snapshot object, dict, or HDF5 handle.
    boundary_fluxes : Any
        Integrated net boundary flux, or time/flux data that can be integrated.
        Positive sign means net gain by the domain.
    quantity : str
        Conserved quantity.  Supported values are ``"mass"``, ``"energy"``,
        and ``"total_energy"``.

    Returns
    -------
    dict[str, Any]
        JSON-compatible totals and residuals with
        ``residual_rel = abs(final - initial - boundary_integral) / abs(initial)``.
    """
    initial = _snapshot_total(snapshot_initial, quantity)
    final = _snapshot_total(snapshot_final, quantity)
    boundary_integral = _boundary_flux_integral(boundary_fluxes)
    delta = final - initial - boundary_integral
    return {
        "quantity": quantity,
        "initial": float(initial),
        "final": float(final),
        "boundary_integral": float(boundary_integral),
        "residual_abs": float(abs(delta)),
        "residual_rel": _relative_residual(delta, initial),
    }


_ESCAPE_COUNTER_ALIASES: dict[str, tuple[str, ...]] = {
    "emergency_cell_deactivation_fired_count": (
        "emergency_cell_deactivation_fired_count",
        "emergency_cell_deactivation_count",
        "emergency_cell_deactivation",
    ),
    "thermal_subcycle_floor_hit_count": (
        "thermal_subcycle_floor_hit_count",
        "thermal_subcycle_floor_count",
        "thermal_subcycle_floor_hit",
    ),
    "axis_spike_floor_activation_count": (
        "axis_spike_floor_activation_count",
        "axis_spike_floor_count",
        "axis_spike_floor",
    ),
    "newton_invalid_count": (
        "newton_invalid_count",
        "newton_invalid_count_total",
        "fld_newton_invalid_count_total",
    ),
}


def _counter_from_summary(summary: dict[str, Any], aliases: tuple[str, ...]) -> int | None:
    for alias in aliases:
        if alias not in summary:
            continue
        value = summary[alias]
        if isinstance(value, bool):
            return int(value)
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        if arr.dtype == np.dtype("bool"):
            return int(np.count_nonzero(arr))
        try:
            return int(np.nanmax(np.asarray(value, dtype=float)))
        except Exception:
            return None
    return None


def no_escape_valve_check(snapshot_summary: dict[str, Any]) -> dict[str, Any]:
    """Check that strict-gate escape-valve counters are present and zero.

    Parameters
    ----------
    snapshot_summary : dict[str, Any]
        Flat summary dictionary containing the escape-valve counters or their
        accepted aliases.

    Returns
    -------
    dict[str, Any]
        JSON-compatible counter report.  ``all_zero`` is true only when every
        required counter is present and equal to zero.
    """
    out: dict[str, Any] = {}
    missing: list[str] = []
    nonzero: list[str] = []
    for canonical, aliases in _ESCAPE_COUNTER_ALIASES.items():
        value = _counter_from_summary(snapshot_summary, aliases)
        out[canonical] = value
        if value is None:
            missing.append(canonical)
        elif value != 0:
            nonzero.append(canonical)
    out["missing_counters"] = missing
    out["nonzero_counters"] = nonzero
    out["all_zero"] = not missing and not nonzero
    return out
