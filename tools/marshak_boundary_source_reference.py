#!/usr/bin/env python3
"""1D cgs/eV references for Marshak boundary-source verification.

The FLD reference solves the nonlinear LTE matter-radiation finite-volume
system with a fully implicit Newton iteration.  The S_N reference uses the
same first-order upwind Strang split structure as ``su_olson_sn_reference.py``,
with cgs scaling and Marshak/reflect boundary conditions adapted to the
``x = z_top - z`` depth coordinate.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np

try:
    import scipy.linalg as _la
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla
except ModuleNotFoundError:  # pragma: no cover - dense fallback for minimal Python envs.
    _la = None
    _sp = None
    _spla = None


A_EV = 1.3720e2
C_LIGHT = 2.99792458e10
E_FLOOR = 1.0e-300


def _validate_inputs(
    L_cm: float,
    n_z: int,
    sigma_a: float,
    rho: float,
    cv_e_volume: float,
    t_end_s: float,
    n_steps: int,
    T_init_eV: float,
) -> None:
    if L_cm <= 0.0:
        raise ValueError("L_cm must be positive")
    if n_z <= 1:
        raise ValueError("n_z must be greater than 1")
    if sigma_a < 0.0 or rho <= 0.0 or cv_e_volume <= 0.0:
        raise ValueError("sigma_a must be nonnegative and rho/cv_e_volume positive")
    if t_end_s < 0.0 or n_steps <= 0:
        raise ValueError("t_end_s must be nonnegative and n_steps positive")
    if T_init_eV < 0.0:
        raise ValueError("T_init_eV must be nonnegative")


def _cell_centers(L_cm: float, n_z: int) -> tuple[np.ndarray, float]:
    dx = L_cm / float(n_z)
    return (np.arange(n_z, dtype=np.float64) + 0.5) * dx, dx


def _fld_flux(E: np.ndarray, dx: float, D: float, F_inc: float) -> np.ndarray:
    n = E.size
    g = np.empty(n + 1, dtype=np.float64)
    g[0] = 0.0 if F_inc == 0.0 else C_LIGHT * E[0] / 4.0 - F_inc
    if n > 1:
        g[1:n] = D * (E[1:] - E[:-1]) / dx
    g[n] = 0.0
    return -g


def _solve_fld_step(
    E_old: np.ndarray,
    T_old: np.ndarray,
    dt: float,
    dx: float,
    sigma: float,
    cv_e_volume: float,
    F_inc: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    n = E_old.size
    if sigma <= 0.0:
        D = 0.0
    else:
        D = C_LIGHT / (3.0 * sigma)
    y = np.empty(2 * n, dtype=np.float64)
    y[0::2] = np.maximum(E_old, 0.0)
    y[1::2] = np.maximum(T_old, 0.0)
    alpha = C_LIGHT * sigma
    use_banded = _la is not None

    def residual_and_jacobian(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        E = v[0::2]
        T = np.maximum(v[1::2], 0.0)
        T3 = T * T * T
        T4 = T3 * T
        R = np.zeros(2 * n, dtype=np.float64)
        J = np.zeros((5, 2 * n), dtype=np.float64) if use_banded else np.zeros((2 * n, 2 * n), dtype=np.float64)

        def add_j(row: int, col: int, value: float) -> None:
            if use_banded:
                J[2 + row - col, col] += value
            else:
                J[row, col] += value

        g_left = 0.0 if F_inc == 0.0 else C_LIGHT * E[0] / 4.0 - F_inc
        for i in range(n):
            row = 2 * i
            e_col = 2 * i
            t_col = e_col + 1
            if i == 0:
                g_m = g_left
                dg_m_i = 0.0 if F_inc == 0.0 else C_LIGHT / 4.0
            else:
                g_m = D * (E[i] - E[i - 1]) / dx
                dg_m_i = D / dx
                add_j(row, 2 * (i - 1), -D / (dx * dx))

            if i == n - 1:
                g_p = 0.0
                dg_p_i = 0.0
            else:
                g_p = D * (E[i + 1] - E[i]) / dx
                dg_p_i = -D / dx
                add_j(row, 2 * (i + 1), -D / (dx * dx))

            R[row] = (E[i] - E_old[i]) / dt - (g_p - g_m) / dx - alpha * (A_EV * T4[i] - E[i])
            add_j(row, e_col, 1.0 / dt - (dg_p_i - dg_m_i) / dx + alpha)
            add_j(row, t_col, -alpha * 4.0 * A_EV * T3[i])

            mrow = row + 1
            R[mrow] = cv_e_volume * (T[i] - T_old[i]) / dt - alpha * (E[i] - A_EV * T4[i])
            add_j(mrow, e_col, -alpha)
            add_j(mrow, t_col, cv_e_volume / dt + alpha * 4.0 * A_EV * T3[i])
        return R, J

    norm = math.inf
    for it in range(1, 51):
        R, J = residual_and_jacobian(y)
        norm = float(np.linalg.norm(R, ord=np.inf))
        scale = max(float(np.linalg.norm(y, ord=np.inf)), 1.0)
        if norm <= max(1.0e-9 * scale, 1.0e-2):
            return np.maximum(y[0::2], 0.0), np.maximum(y[1::2], 0.0), it, norm
        delta = _la.solve_banded((2, 2), J, -R) if use_banded else np.linalg.solve(J, -R)
        lam = 1.0
        base = norm
        while lam > 1.0e-6:
            trial = y + lam * delta
            trial[1::2] = np.maximum(trial[1::2], 1.0e-12)
            trial[0::2] = np.maximum(trial[0::2], 0.0)
            R_trial, _ = residual_and_jacobian(trial)
            trial_norm = float(np.linalg.norm(R_trial, ord=np.inf))
            if math.isfinite(trial_norm) and trial_norm <= max(0.8 * base, 1.0e-30):
                y = trial
                break
            lam *= 0.5
        else:
            y += delta
            y[1::2] = np.maximum(y[1::2], 1.0e-12)
            y[0::2] = np.maximum(y[0::2], 0.0)
    return np.maximum(y[0::2], 0.0), np.maximum(y[1::2], 0.0), 50, norm


def solve_marshak_fld_1d(
    L_cm: float,
    n_z: int,
    sigma_a: float,
    rho: float,
    cv_e_volume: float,
    F_inc_erg_per_cm2_s: float,
    t_end_s: float,
    n_steps: int,
    T_init_eV: float,
) -> dict[str, Any]:
    """Solve the 1D FLD Marshak source problem in ``x = z_top - z``."""

    _validate_inputs(L_cm, n_z, sigma_a, rho, cv_e_volume, t_end_s, n_steps, T_init_eV)
    x, dx = _cell_centers(L_cm, n_z)
    E = np.full(n_z, A_EV * T_init_eV**4, dtype=np.float64)
    T = np.full(n_z, T_init_eV, dtype=np.float64)
    dt = t_end_s / float(n_steps) if n_steps > 0 else 0.0
    max_newton = 0
    final_residual = 0.0
    for _ in range(n_steps):
        if dt > 0.0:
            E, T, nit, final_residual = _solve_fld_step(
                E, T, dt, dx, float(sigma_a), float(cv_e_volume), float(F_inc_erg_per_cm2_s)
            )
            max_newton = max(max_newton, nit)
    D = 0.0 if sigma_a <= 0.0 else C_LIGHT / (3.0 * sigma_a)
    return {
        "x_centers": x,
        "E_r": E,
        "T_eV": T,
        "F_r": _fld_flux(E, dx, D, float(F_inc_erg_per_cm2_s)),
        "diagnostics": {
            "scheme": "fully_implicit_newton_fv",
            "a_eV": A_EV,
            "c_cgs": C_LIGHT,
            "dx_cm": dx,
            "dt_s": dt,
            "max_newton_iterations": max_newton,
            "final_newton_residual_inf": final_residual,
        },
    }


def _gauss_legendre_sn(order: int) -> tuple[np.ndarray, np.ndarray]:
    mu, w = np.polynomial.legendre.leggauss(order)
    return mu.astype(np.float64), w.astype(np.float64)


def _stream_upwind_marshak(
    I_in: np.ndarray,
    I_out: np.ndarray,
    mu: np.ndarray,
    pair: np.ndarray,
    dt: float,
    dx: float,
    psi_top_in: float,
) -> None:
    for n, mu_n in enumerate(mu):
        cfl = C_LIGHT * abs(float(mu_n)) * dt / dx
        if mu_n > 0.0:
            top_in = I_in[pair[n], 0] if psi_top_in == 0.0 else psi_top_in
            I_out[n, 0] = I_in[n, 0] - cfl * (I_in[n, 0] - top_in)
            I_out[n, 1:] = I_in[n, 1:] - cfl * (I_in[n, 1:] - I_in[n, :-1])
        else:
            reflected = I_in[pair[n], -1]
            I_out[n, :-1] = I_in[n, :-1] + cfl * (I_in[n, 1:] - I_in[n, :-1])
            I_out[n, -1] = I_in[n, -1] + cfl * (reflected - I_in[n, -1])


def _collision_sn_lte(
    I: np.ndarray,
    T: np.ndarray,
    w: np.ndarray,
    dt: float,
    sigma: float,
    cv_e_volume: float,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = C_LIGHT * sigma * dt
    if alpha <= 0.0:
        return I.copy(), T.copy()
    E_stream = np.sum(w[:, None] * I, axis=0) / C_LIGHT
    T_new = np.maximum(T.copy(), 1.0e-12)
    for i in range(T.size):
        Told = T[i]
        Es = E_stream[i]
        Ti = max(T_new[i], 1.0e-12)
        for _ in range(30):
            T4 = Ti**4
            B = A_EV * T4
            E_new = (Es + alpha * B) / (1.0 + alpha)
            R = cv_e_volume * (Ti - Told) - alpha * (E_new - B)
            dB = 4.0 * A_EV * Ti**3
            dE = alpha * dB / (1.0 + alpha)
            dR = cv_e_volume - alpha * (dE - dB)
            dT = -R / max(dR, E_FLOOR)
            Ti = max(Ti + dT, 1.0e-12)
            if abs(dT) <= 1.0e-11 * max(Ti, 1.0):
                break
        T_new[i] = Ti
    B_new = A_EV * T_new**4
    I_new = (I + alpha * (0.5 * C_LIGHT * B_new[None, :])) / (1.0 + alpha)
    return I_new, T_new


def solve_marshak_sn_1d(
    L_cm: float,
    n_z: int,
    sigma_a: float,
    rho: float,
    cv_e_volume: float,
    F_inc_erg_per_cm2_s: float,
    t_end_s: float,
    n_steps: int,
    T_init_eV: float,
    sn_order: int = 16,
) -> dict[str, Any]:
    """Solve a 1D S_N Marshak source proxy in ``x = z_top - z``.

    TENRYU's z-top incoming ordinates have ``mu_z < 0`` and
    ``psi_in = 2 F_inc``.  In this depth coordinate those same rays have
    ``mu_x > 0``.
    """

    _validate_inputs(L_cm, n_z, sigma_a, rho, cv_e_volume, t_end_s, n_steps, T_init_eV)
    x, dx = _cell_centers(L_cm, n_z)
    mu, w = _gauss_legendre_sn(sn_order)
    pair = np.arange(mu.size - 1, -1, -1, dtype=np.int64)
    E0 = A_EV * T_init_eV**4
    I = np.full((mu.size, n_z), 0.5 * C_LIGHT * E0, dtype=np.float64)
    T = np.full(n_z, T_init_eV, dtype=np.float64)
    dt_base = t_end_s / float(n_steps) if n_steps > 0 else 0.0
    cfl_limit = 0.45 * dx / (C_LIGHT * float(np.max(np.abs(mu))))
    subcycles = max(1, int(math.ceil(dt_base / cfl_limit))) if dt_base > 0.0 else 1
    dt = dt_base / float(subcycles)
    I_half = np.empty_like(I)
    I_coll = np.empty_like(I)
    I_next = np.empty_like(I)
    psi_in = 2.0 * max(float(F_inc_erg_per_cm2_s), 0.0)
    for _ in range(n_steps):
        for _ in range(subcycles):
            _stream_upwind_marshak(I, I_half, mu, pair, 0.5 * dt, dx, psi_in)
            I_coll, T = _collision_sn_lte(I_half, T, w, dt, float(sigma_a), float(cv_e_volume))
            _stream_upwind_marshak(I_coll, I_next, mu, pair, 0.5 * dt, dx, psi_in)
            I, I_next = I_next, I
    E = np.sum(w[:, None] * I, axis=0) / C_LIGHT
    F_cells = np.sum(w[:, None] * mu[:, None] * I, axis=0)
    F_faces = np.empty(n_z + 1, dtype=np.float64)
    F_faces[0] = np.sum(w[mu > 0.0] * mu[mu > 0.0] * psi_in) - np.sum(
        w[mu < 0.0] * abs(mu[mu < 0.0]) * I[mu < 0.0, 0]
    )
    if n_z > 1:
        F_faces[1:n_z] = 0.5 * (F_cells[:-1] + F_cells[1:])
    F_faces[n_z] = 0.0
    return {
        "x_centers": x,
        "E_r": E,
        "T_eV": T,
        "F_r": F_faces,
        "diagnostics": {
            "scheme": "strang_upwind_implicit_lte_collision",
            "a_eV": A_EV,
            "c_cgs": C_LIGHT,
            "dx_cm": dx,
            "dt_s": dt,
            "requested_steps": n_steps,
            "streaming_subcycles_per_step": subcycles,
            "sn_order": sn_order,
            "psi_in": psi_in,
        },
    }


def _rel_linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.nanmax(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), E_FLOOR)))


def _self_test() -> int:
    failures: list[str] = []
    eq = A_EV * 1.7**4
    fld0 = solve_marshak_fld_1d(1.0, 64, 1.0, 1.0, 5.488e2, 0.0, 1.0e-10, 10, 1.7)
    err0 = _rel_linf(fld0["E_r"], np.full(64, eq))
    if err0 > 1.0e-10:
        failures.append(f"FLD zero-flux equilibrium drift rel={err0:.3e}")

    fld = solve_marshak_fld_1d(0.1, 24, 10.0, 1.0, 1.0, 1.0e10, 1.0e-9, 20, 0.1)
    boundary_ref = 4.0e10 / C_LIGHT
    boundary_err = abs(float(fld["E_r"][0]) - boundary_ref) / max(boundary_ref, E_FLOOR)
    if boundary_err > 0.25:
        failures.append(f"FLD long-time boundary rel={boundary_err:.3e}")
    lte_err = _rel_linf(fld["E_r"][12:], A_EV * fld["T_eV"][12:] ** 4)
    if lte_err > 5.0e-2:
        failures.append(f"FLD far-interior LTE rel={lte_err:.3e}")

    sn0 = solve_marshak_sn_1d(1.0, 64, 1.0, 1.0, 5.488e2, 0.0, 1.0e-10, 10, 1.7)
    sn_err0 = _rel_linf(sn0["E_r"], np.full(64, eq))
    if sn_err0 > 1.0e-10:
        failures.append(f"SN zero-flux equilibrium drift rel={sn_err0:.3e}")
    sn = solve_marshak_sn_1d(0.1, 24, 10.0, 1.0, 1.0, 1.0e10, 1.0e-9, 20, 0.1, sn_order=16)
    sn_lte_err = _rel_linf(sn["E_r"][12:], A_EV * sn["T_eV"][12:] ** 4)
    if sn_lte_err > 8.0e-2:
        failures.append(f"SN far-interior LTE rel={sn_lte_err:.3e}")

    if failures:
        print("Marshak boundary-source reference self-test: FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("Marshak boundary-source reference self-test: PASS")
    print(f"  FLD zero-flux rel={err0:.3e}")
    print(f"  FLD boundary rel={boundary_err:.3e}")
    print(f"  FLD far-interior LTE rel={lte_err:.3e}")
    print(f"  SN zero-flux rel={sn_err0:.3e}")
    print(f"  SN far-interior LTE rel={sn_lte_err:.3e}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    parser.error("no action requested; use --self-test")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
