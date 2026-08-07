#!/usr/bin/env python3
"""Generate static-material grey S_N radiative-shock references.

This is the I3 W0 in-house reference from
``docs/design/i3_sn_equation_audit.md`` section 7.  It solves the steady
shock-frame, radiation-momentum-free system

    mu_l d psi_l / dz = sigma_a (0.5 c a T**4 - psi_l)

with TENRYU's azimuth-summed polar Gauss-Legendre normalization, sum(w)=2.
The default closure is S_N; ``--closure ced`` is a strict adapter to the
existing constant-Eddington generator and exists only for the self-identity
check in ``docs/design/i3_sn_radshock_spec.md`` section 4.1.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq


C_CGS = 2.99792458e10
A_EV = 1.3720e2
TINY = 1.0e-300
DEFAULT_SPAN_REL = 25.0
MAX_ANGULAR_POINTS = 201


@dataclass(frozen=True)
class SNConfig:
    mach: float = 2.0
    n_angles: int = 16
    kappa_scale: float = 1.0
    n_points: int = 1601
    span_rel: float = DEFAULT_SPAN_REL
    closure: str = "sn"
    with_rad_momentum: bool = False
    relax: float = 0.5
    max_iter: int = 500
    tol: float = 1.0e-12
    gamma: float = 5.0 / 3.0
    rho0_gcc: float = 2.0
    T0_eV: float = 30.0
    Cv_erg_per_g_eV: float = 2873691896536.808
    A_amu: float = 1.0
    zbar: float = 1.0

    @property
    def R_gas(self) -> float:
        return (self.gamma - 1.0) * self.Cv_erg_per_g_eV

    @property
    def cp(self) -> float:
        return self.gamma * self.Cv_erg_per_g_eV

    @property
    def kappa_a_cm2_g(self) -> float:
        return 0.5 * self.kappa_scale

    @property
    def upstream_sigma_cm_inv(self) -> float:
        return self.kappa_a_cm2_g * self.rho0_gcc

    @property
    def half_span_cm(self) -> float:
        return self.span_rel / self.upstream_sigma_cm_inv


@dataclass(frozen=True)
class Invariants:
    R_gas: float
    cp: float
    cv: float
    rho0: float
    T0: float
    u0: float
    p0: float
    mass_flux: float
    momentum_flux: float
    energy_flux: float
    T_sonic: float
    T_down: float
    with_rad_momentum: bool = False


@dataclass(frozen=True)
class TransportSolution:
    psi: np.ndarray
    E: np.ndarray
    F: np.ndarray
    P: np.ndarray
    chi: np.ndarray
    rho: np.ndarray
    u: np.ndarray
    sigma_a: np.ndarray
    source: np.ndarray


def safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def validate_config(cfg: SNConfig) -> None:
    if cfg.mach <= 1.0:
        raise ValueError("mach must be supersonic")
    if cfg.n_angles < 2 or cfg.n_angles % 2 != 0:
        raise ValueError("n_angles must be an even integer >= 2")
    if cfg.kappa_scale <= 0.0:
        raise ValueError("kappa_scale must be positive")
    if cfg.n_points < 101:
        raise ValueError("n_points must be >= 101")
    if cfg.span_rel <= 0.0:
        raise ValueError("span_rel must be positive")
    if cfg.closure not in {"sn", "ced"}:
        raise ValueError("closure must be 'sn' or 'ced'")
    if not 0.0 < cfg.relax <= 1.0:
        raise ValueError("relax must be in (0, 1]")
    if cfg.max_iter < 1 or cfg.tol <= 0.0:
        raise ValueError("max_iter and tol must be positive")


def default_ced_json_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "verification"
        / "data"
        / "fld_const_eddington_reference"
        / "fld_ced_M2.json"
    )


def load_default_config() -> SNConfig:
    """Load frozen I1-A defaults from the CED JSON when it is present."""
    path = default_ced_json_path()
    if not path.exists():
        # Fallback constants are the frozen fld_ced_M2.json parameters.
        return SNConfig()
    with path.open(encoding="utf-8") as stream:
        params = json.load(stream).get("parameters", {})
    return SNConfig(
        mach=float(params.get("M0", 2.0)),
        gamma=float(params.get("gamma", 5.0 / 3.0)),
        rho0_gcc=float(params.get("rho0_g_per_cc", 2.0)),
        T0_eV=float(params.get("T0_eV", 30.0)),
        Cv_erg_per_g_eV=float(params.get("Cv_erg_per_g_eV", 2873691896536.808)),
        A_amu=float(params.get("A_amu", 1.0)),
        zbar=float(params.get("zbar", 1.0)),
    )


def equilibrium_E(T_eV: np.ndarray | float) -> np.ndarray | float:
    return A_EV * np.asarray(T_eV) ** 4 if not isinstance(T_eV, (float, int)) else A_EV * T_eV**4


def build_invariants(cfg: SNConfig) -> Invariants:
    validate_config(cfg)
    R = cfg.R_gas
    cp = cfg.cp
    rho0 = cfg.rho0_gcc
    T0 = cfg.T0_eV
    p0 = rho0 * R * T0
    u0 = cfg.mach * math.sqrt(cfg.gamma * R * T0)
    J = rho0 * u0
    E0 = A_EV * T0**4
    P0 = E0 / 3.0
    Pi = J * u0 + p0 + (P0 if cfg.with_rad_momentum else 0.0)
    Phi = J * (cp * T0 + 0.5 * u0 * u0)
    if cfg.with_rad_momentum:
        Phi += (4.0 / 3.0) * u0 * E0

    if cfg.with_rad_momentum:
        T_down = solve_downstream_temperature_with_rad_momentum(cfg, J, Pi, Phi)
        # This is only an auxiliary study.  Use the gas sonic point as a stable
        # upper bracket for the branch update; the radiation-pressure term is
        # tiny for the I3 parameters.
        T_sonic = Pi * Pi / (4.0 * J * J * R)
    else:
        T_sonic = Pi * Pi / (4.0 * J * J * R)
        provisional = Invariants(R, cp, cfg.Cv_erg_per_g_eV, rho0, T0, u0, p0, J, Pi, Phi, T_sonic, T0)
        T_down = float(
            brentq(
                lambda T: gas_energy_flux(T, "downstream", provisional) - Phi,
                T0,
                T_sonic * (1.0 - 1.0e-12),
                xtol=1.0e-12,
                rtol=1.0e-12,
                maxiter=200,
            )
        )
    return Invariants(
        R,
        cp,
        cfg.Cv_erg_per_g_eV,
        rho0,
        T0,
        u0,
        p0,
        J,
        Pi,
        Phi,
        T_sonic,
        T_down,
        cfg.with_rad_momentum,
    )


def velocity_from_T(T_eV: float, branch: str, inv: Invariants) -> float:
    """Return the gas-only momentum-root velocity.

    Audit section 7 uses mass C1=rho*u and gas momentum
    C2=rho*u**2+p.  With p=rho*R*T and rho=C1/u,

        C2 = C1*u + C1*R*T/u.

    Multiplying by u gives C1*u**2 - C2*u + C1*R*T = 0, hence

        u = (C2 +/- sqrt(C2**2 - 4*C1**2*R*T)) / (2*C1).

    The plus root is continuous with the supersonic upstream state, and the
    minus root is the subsonic downstream branch.  The embedded subshock
    matching condition is enforced by evaluating both roots at the same
    continuous radiation flux F_z at z_s=0; then the gas energy flux is the
    same on both sides, so the local Hugoniot jump is satisfied.
    """
    disc = inv.momentum_flux * inv.momentum_flux - 4.0 * inv.mass_flux * inv.mass_flux * inv.R_gas * T_eV
    if disc < -1.0e-12 * inv.momentum_flux * inv.momentum_flux:
        raise FloatingPointError("momentum quadratic has no real root")
    root = math.sqrt(max(disc, 0.0))
    if branch == "upstream":
        return (inv.momentum_flux + root) / (2.0 * inv.mass_flux)
    if branch == "downstream":
        return (inv.momentum_flux - root) / (2.0 * inv.mass_flux)
    raise ValueError(f"unknown branch {branch!r}")


def velocity_array_from_T(
    T_eV: np.ndarray,
    upstream_mask: np.ndarray,
    inv: Invariants,
    P_rad: np.ndarray | None = None,
) -> np.ndarray:
    b = inv.momentum_flux if P_rad is None else inv.momentum_flux - P_rad
    disc = b * b - 4.0 * inv.mass_flux * inv.mass_flux * inv.R_gas * T_eV
    root = np.sqrt(np.maximum(disc, 0.0))
    return np.where(upstream_mask, (b + root) / (2.0 * inv.mass_flux), (b - root) / (2.0 * inv.mass_flux))


def gas_energy_flux(T_eV: float, branch: str, inv: Invariants) -> float:
    u = velocity_from_T(T_eV, branch, inv)
    return inv.mass_flux * (inv.cp * T_eV + 0.5 * u * u)


def radiation_flux_energy(T_eV: float, branch: str, inv: Invariants) -> float:
    return inv.energy_flux - gas_energy_flux(T_eV, branch, inv)


def solve_downstream_temperature_with_rad_momentum(cfg: SNConfig, J: float, Pi: float, Phi: float) -> float:
    R = cfg.R_gas
    cp = cfg.cp

    def u_down(T: float) -> float:
        E = A_EV * T**4
        b = Pi - E / 3.0
        disc = b * b - 4.0 * J * J * R * T
        return (b - math.sqrt(max(disc, 0.0))) / (2.0 * J)

    def residual(T: float) -> float:
        u = u_down(T)
        E = A_EV * T**4
        return J * (cp * T + 0.5 * u * u) + (4.0 / 3.0) * u * E - Phi

    lo = cfg.T0_eV
    hi = max(cfg.T0_eV * 2.5, cfg.T0_eV + 10.0)
    while residual(hi) < 0.0:
        hi *= 1.5
        if hi > 1.0e5 * cfg.T0_eV:
            raise RuntimeError("failed to bracket auxiliary radiation-momentum downstream state")
    return float(brentq(residual, lo, hi, xtol=1.0e-12, rtol=1.0e-12, maxiter=200))


def grid_for_config(cfg: SNConfig) -> np.ndarray:
    return np.linspace(-cfg.half_span_cm, cfg.half_span_cm, cfg.n_points)


def import_ced_generator() -> Any | None:
    try:
        import fld_const_eddington_reference_generator as ced  # type: ignore
    except ImportError:
        return None
    return ced


def ced_payload_for_config(cfg: SNConfig) -> dict[str, Any] | None:
    ced = import_ced_generator()
    if ced is None:
        return None
    ced_cfg = ced.FLDCEDConfig(
        mach=cfg.mach,
        gamma=cfg.gamma,
        rho0_gcc=cfg.rho0_gcc,
        T0_eV=cfg.T0_eV,
        kappa_R_cm2_g=cfg.kappa_a_cm2_g,
        A_amu=cfg.A_amu,
        zbar=cfg.zbar,
        n_points=max(cfg.n_points, 1000),
        target_domain_span_cm=2.0 * cfg.half_span_cm,
    )
    return ced.build_reference(ced_cfg)


def initial_temperature_profile(z: np.ndarray, cfg: SNConfig, inv: Invariants) -> np.ndarray:
    ced_payload = ced_payload_for_config(cfg)
    if ced_payload is not None:
        x_ced = np.asarray(ced_payload["table"]["x_cm"], dtype=float)
        T_ced = np.asarray(ced_payload["table"]["T_eV"], dtype=float)
        T = np.interp(z, x_ced, T_ced, left=inv.T0, right=inv.T_down)
    else:
        width = max(0.12 * cfg.half_span_cm, 1.0 / cfg.upstream_sigma_cm_inv)
        T = 0.5 * (inv.T0 + inv.T_down) + 0.5 * (inv.T_down - inv.T0) * np.tanh(z / width)
    T[0] = inv.T0
    T[-1] = inv.T_down
    return T


def one_minus_exp_neg(tau: float) -> float:
    if tau > 745.0:
        return 1.0
    return -math.expm1(-tau)


def integrating_factor_update(inflow: float, source: float, tau: float) -> float:
    """Exact constant-cell transport update.

    For a cell with constant sigma_a and source S=0.5*c*a*T**4, audit section
    7 gives d psi / ds = sigma_a (S - psi), where ds=dz/mu for mu>0 and
    ds=-dz/mu for mu<0.  The cell-exit value is

        psi_out = S + (psi_in - S) exp(-tau)
                = psi_in + (S - psi_in) (1 - exp(-tau)).

    The second form is evaluated with expm1 so optically thin cells do not
    lose the small relaxation term.
    """
    return inflow + (source - inflow) * one_minus_exp_neg(tau)


def moments_from_psi(mu: np.ndarray, w: np.ndarray, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weighted = w[:, None] * psi
    E = np.sum(weighted, axis=0) / C_CGS
    F = np.sum(w[:, None] * mu[:, None] * psi, axis=0)
    P = np.sum(w[:, None] * mu[:, None] ** 2 * psi, axis=0) / C_CGS
    chi = P / np.maximum(E, TINY)
    return E, F, P, chi


def solve_transport(
    z: np.ndarray,
    T: np.ndarray,
    upstream_mask: np.ndarray,
    mu: np.ndarray,
    w: np.ndarray,
    inv: Invariants,
    cfg: SNConfig,
    P_flow: np.ndarray | None = None,
) -> TransportSolution:
    u = velocity_array_from_T(T, upstream_mask, inv, P_rad=P_flow if inv.with_rad_momentum else None)
    rho = inv.mass_flux / np.maximum(u, TINY)
    sigma = cfg.kappa_a_cm2_g * rho
    source = 0.5 * C_CGS * A_EV * T**4
    psi = np.empty((len(mu), len(z)), dtype=float)
    dz = np.diff(z)
    sigma_cell = 0.5 * (sigma[:-1] + sigma[1:])
    source_cell = 0.5 * (source[:-1] + source[1:])
    psi_left = 0.5 * C_CGS * A_EV * inv.T0**4
    psi_right = 0.5 * C_CGS * A_EV * inv.T_down**4

    for l, mu_l in enumerate(mu):
        if mu_l > 0.0:
            psi[l, 0] = psi_left
            for i in range(len(z) - 1):
                tau = sigma_cell[i] * dz[i] / mu_l
                psi[l, i + 1] = integrating_factor_update(psi[l, i], source_cell[i], tau)
        else:
            abs_mu = -mu_l
            psi[l, -1] = psi_right
            for i in range(len(z) - 2, -1, -1):
                tau = sigma_cell[i] * dz[i] / abs_mu
                psi[l, i] = integrating_factor_update(psi[l, i + 1], source_cell[i], tau)

    E, F, P, chi = moments_from_psi(mu, w, psi)
    return TransportSolution(psi=psi, E=E, F=F, P=P, chi=chi, rho=rho, u=u, sigma_a=sigma, source=source)


def gas_energy_array(T: np.ndarray, upstream_mask: np.ndarray, inv: Invariants, P_rad: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    u = velocity_array_from_T(T, upstream_mask, inv, P_rad=P_rad if inv.with_rad_momentum else None)
    return inv.mass_flux * (inv.cp * T + 0.5 * u * u), u


def dgas_energy_dT(T: np.ndarray, upstream_mask: np.ndarray, inv: Invariants, P_rad: np.ndarray | None = None) -> np.ndarray:
    u = velocity_array_from_T(T, upstream_mask, inv, P_rad=P_rad if inv.with_rad_momentum else None)
    denom = 1.0 - inv.R_gas * T / np.maximum(u * u, TINY)
    denom = np.where(np.abs(denom) < 1.0e-10, np.copysign(1.0e-10, denom), denom)
    dudT = -(inv.R_gas / u) / denom
    deriv = inv.mass_flux * (inv.cp + u * dudT)
    if inv.with_rad_momentum and P_rad is not None:
        deriv += (4.0 / 3.0) * P_rad * 0.0
    return deriv


def update_temperature_from_flux(
    F: np.ndarray,
    upstream_mask: np.ndarray,
    previous_T: np.ndarray,
    inv: Invariants,
    E_rad: np.ndarray | None = None,
    P_rad: np.ndarray | None = None,
) -> np.ndarray:
    T = previous_T.copy()
    lo = np.where(upstream_mask, inv.T0, inv.T_down)
    hi = inv.T_sonic * (1.0 - 1.0e-11)
    if inv.with_rad_momentum:
        hi = np.maximum(hi, inv.T_down * 1.05)

    for _ in range(12):
        gas, u = gas_energy_array(T, upstream_mask, inv, P_rad=P_rad)
        residual = gas + F - inv.energy_flux
        deriv = dgas_energy_dT(T, upstream_mask, inv, P_rad=P_rad)
        if inv.with_rad_momentum and E_rad is not None:
            residual += (4.0 / 3.0) * u * E_rad
            denom = 1.0 - inv.R_gas * T / np.maximum(u * u, TINY)
            denom = np.where(np.abs(denom) < 1.0e-10, np.copysign(1.0e-10, denom), denom)
            dudT = -(inv.R_gas / u) / denom
            deriv += (4.0 / 3.0) * E_rad * dudT
        step = residual / np.where(np.abs(deriv) > TINY, deriv, np.sign(deriv) * TINY + TINY)
        T = np.clip(T - step, lo, hi)
    T[0] = inv.T0
    T[-1] = inv.T_down
    return T


def subshock_temperatures(F_s: float, inv: Invariants) -> tuple[float, float]:
    def solve(branch: str, lo: float) -> float:
        return float(
            brentq(
                lambda T: radiation_flux_energy(T, branch, inv) - F_s,
                lo,
                inv.T_sonic * (1.0 - 1.0e-10),
                xtol=1.0e-12,
                rtol=1.0e-12,
                maxiter=200,
            )
        )

    return solve("upstream", inv.T0), solve("downstream", inv.T_down)


def state_from_T_E_F_P(T_eV: float, E_rad: float, F_rad: float, P_rad: float, branch: str, inv: Invariants) -> dict[str, float]:
    u = velocity_from_T(T_eV, branch, inv)
    rho = inv.mass_flux / u
    return {
        "rho": rho,
        "u": u,
        "T": T_eV,
        "T_rad": (max(E_rad, 0.0) / A_EV) ** 0.25,
        "E_rad": E_rad,
        "F_rad": F_rad,
        "F_rad_lab": F_rad,
        "P_mat": rho * inv.R_gas * T_eV,
        "P_rad": P_rad,
    }


def table_from_solution(z: np.ndarray, T: np.ndarray, upstream_mask: np.ndarray, tr: TransportSolution, inv: Invariants) -> dict[str, list[Any]]:
    p_mat = tr.rho * inv.R_gas * T
    branches = np.where(upstream_mask, "upstream", "downstream")
    return {
        "x_cm": z.tolist(),
        "rho_g_per_cc": tr.rho.tolist(),
        "u_cm_per_s": tr.u.tolist(),
        "T_eV": T.tolist(),
        "T_rad_eV": ((np.maximum(tr.E, 0.0) / A_EV) ** 0.25).tolist(),
        "E_rad_erg_per_cm3": tr.E.tolist(),
        "F_rad_erg_per_cm2_s": tr.F.tolist(),
        "F_rad_energy_invariant_erg_per_cm2_s": tr.F.tolist(),
        "F_rad_lab_erg_per_cm2_s": tr.F.tolist(),
        "P_mat_dyne_per_cm2": p_mat.tolist(),
        "P_rad_dyne_per_cm2": tr.P.tolist(),
        "chi_z": tr.chi.tolist(),
        "branch": branches.tolist(),
    }


def select_angular_grid(z: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(MAX_ANGULAR_POINTS, len(z))
    indices = np.unique(np.rint(np.linspace(0, len(z) - 1, n)).astype(int))
    return z[indices], T[indices]


def exact_transport_identity_residual(
    z: np.ndarray,
    psi: np.ndarray,
    source: np.ndarray,
    sigma: np.ndarray,
    mu: np.ndarray,
) -> float:
    dz = np.diff(z)
    sigma_cell = 0.5 * (sigma[:-1] + sigma[1:])
    source_cell = 0.5 * (source[:-1] + source[1:])
    worst = 0.0
    for l, mu_l in enumerate(mu):
        if mu_l > 0.0:
            pairs = range(len(z) - 1)
            for i in pairs:
                expected = integrating_factor_update(psi[l, i], source_cell[i], sigma_cell[i] * dz[i] / mu_l)
                scale = max(abs(expected), abs(psi[l, i + 1]), TINY)
                worst = max(worst, abs(psi[l, i + 1] - expected) / scale)
        else:
            abs_mu = -mu_l
            for i in range(len(z) - 2, -1, -1):
                expected = integrating_factor_update(psi[l, i + 1], source_cell[i], sigma_cell[i] * dz[i] / abs_mu)
                scale = max(abs(expected), abs(psi[l, i]), TINY)
                worst = max(worst, abs(psi[l, i] - expected) / scale)
    return worst


def angular_block_from_solution(
    z: np.ndarray,
    T: np.ndarray,
    upstream_mask: np.ndarray,
    mu: np.ndarray,
    w: np.ndarray,
    inv: Invariants,
    cfg: SNConfig,
) -> tuple[dict[str, Any], float]:
    z_ang, T_ang = select_angular_grid(z, T)
    up_ang = z_ang < 0.0
    tr_ang = solve_transport(z_ang, T_ang, up_ang, mu, w, inv, cfg)
    residual = exact_transport_identity_residual(z_ang, tr_ang.psi, tr_ang.source, tr_ang.sigma_a, mu)
    block = {
        "x_cm": z_ang.tolist(),
        "mu": mu.tolist(),
        "w": w.tolist(),
        "psi_table": tr_ang.psi.tolist(),
        "E_rad_erg_per_cm3": tr_ang.E.tolist(),
        "F_rad_erg_per_cm2_s": tr_ang.F.tolist(),
        "P_rad_dyne_per_cm2": tr_ang.P.tolist(),
        "chi_z": tr_ang.chi.tolist(),
    }
    return block, residual


def rh_residuals(states: dict[str, dict[str, float]], inv: Invariants) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name in ("upstream", "downstream"):
        state = states[name]
        rho = state["rho"]
        u = state["u"]
        T = state["T"]
        mass = rho * u
        momentum = rho * u * u + rho * inv.R_gas * T
        energy = inv.mass_flux * (inv.cp * T + 0.5 * u * u) + state["F_rad"]
        out[name] = {
            "mass": abs(mass - inv.mass_flux) / max(abs(inv.mass_flux), TINY),
            "momentum": abs(momentum - inv.momentum_flux) / max(abs(inv.momentum_flux), TINY),
            "energy": abs(energy - inv.energy_flux) / max(abs(inv.energy_flux), TINY),
        }
    return out


def advection_fraction(z: np.ndarray, tr: TransportSolution, cfg: SNConfig) -> dict[str, float]:
    half_window = 10.0 / cfg.upstream_sigma_cm_inv
    mask = np.abs(z) <= half_window
    if np.count_nonzero(mask) == 0:
        mask = np.ones_like(z, dtype=bool)
    max_f = float(np.nanmax(np.abs(tr.F[mask])))
    denom = max(max_f, TINY)
    uE = np.abs(tr.u[mask] * tr.E[mask])
    return {
        "max_uE_over_maxFz_window": float(np.nanmax(uE) / denom),
        "max_4thirds_uE_over_maxFz_window": float(np.nanmax((4.0 / 3.0) * uE) / denom),
    }


def git_commit(path: Path | None = None) -> str | None:
    cmd = ["git"]
    if path is not None:
        cmd += ["-C", str(path)]
    cmd += ["rev-parse", "HEAD"]
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def build_sn_reference(cfg: SNConfig) -> dict[str, Any]:
    inv = build_invariants(cfg)
    z = grid_for_config(cfg)
    upstream_mask = z < 0.0
    mu, w = np.polynomial.legendre.leggauss(cfg.n_angles)
    T = initial_temperature_profile(z, cfg, inv)
    P_flow: np.ndarray | None = None
    residual = math.inf
    iterations = 0
    converged = False

    for iteration in range(1, cfg.max_iter + 1):
        tr = solve_transport(z, T, upstream_mask, mu, w, inv, cfg, P_flow=P_flow)
        T_candidate = update_temperature_from_flux(
            tr.F,
            upstream_mask,
            T,
            inv,
            E_rad=tr.E if inv.with_rad_momentum else None,
            P_rad=tr.P if inv.with_rad_momentum else None,
        )
        residual = float(np.nanmax(np.abs(T_candidate - T) / np.maximum(np.abs(T), TINY)))
        T = (1.0 - cfg.relax) * T + cfg.relax * T_candidate
        T[0] = inv.T0
        T[-1] = inv.T_down
        P_flow = tr.P if inv.with_rad_momentum else None
        iterations = iteration
        if residual <= cfg.tol:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            f"S_N fixed point did not converge: residual={residual:.6e} after {cfg.max_iter} iterations"
        )

    tr = solve_transport(z, T, upstream_mask, mu, w, inv, cfg, P_flow=P_flow)
    table = table_from_solution(z, T, upstream_mask, tr, inv)
    F_s = float(np.interp(0.0, z, tr.F))
    if inv.with_rad_momentum:
        T_pre = float(T[np.searchsorted(z, 0.0) - 1])
        T_post = float(T[np.searchsorted(z, 0.0)])
    else:
        T_pre, T_post = subshock_temperatures(F_s, inv)
    E_s = float(np.interp(0.0, z, tr.E))
    P_s = float(np.interp(0.0, z, tr.P))
    upstream = state_from_T_E_F_P(inv.T0, A_EV * inv.T0**4, 0.0, A_EV * inv.T0**4 / 3.0, "upstream", inv)
    downstream = state_from_T_E_F_P(
        inv.T_down,
        A_EV * inv.T_down**4,
        0.0,
        A_EV * inv.T_down**4 / 3.0,
        "downstream",
        inv,
    )
    states = {
        "upstream": upstream,
        "downstream": downstream,
        "shock_upstream": state_from_T_E_F_P(T_pre, E_s, F_s, P_s, "upstream", inv),
        "shock_downstream": state_from_T_E_F_P(T_post, E_s, F_s, P_s, "downstream", inv),
    }
    angular, transport_identity = angular_block_from_solution(z, T, upstream_mask, mu, w, inv, cfg)
    diagnostics = {
        "outer_iterations": iterations,
        "max_T_residual": residual,
        "transport_selfidentity_max_rel": transport_identity,
        "rh_residual_rel": rh_residuals(states, inv),
        "subshock": {
            "present": bool(T_post - T_pre > 1.0e-8 * max(inv.T_down, inv.T0)),
            "z_s_cm": 0.0,
            "T_gas_jump": T_post - T_pre,
        },
        "advection_fraction": advection_fraction(z, tr, cfg),
    }
    model = "static_material_grey_SN"
    if cfg.with_rad_momentum:
        model = "AUX_with_rad_momentum_NOT_A_REFERENCE"
    return payload_from_parts(cfg, inv, table, states, diagnostics, angular, model)


def ced_intensity_block(table: dict[str, Any], mu: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    x = np.asarray(table["x_cm"], dtype=float)
    n = min(MAX_ANGULAR_POINTS, len(x))
    idx = np.unique(np.rint(np.linspace(0, len(x) - 1, n)).astype(int))
    E = np.asarray(table["E_rad_erg_per_cm3"], dtype=float)[idx]
    F = np.asarray(table["F_rad_erg_per_cm2_s"], dtype=float)[idx]
    psi = 0.5 * C_CGS * E[None, :] + 1.5 * mu[:, None] * F[None, :]
    P = np.sum(w[:, None] * mu[:, None] ** 2 * psi, axis=0) / C_CGS
    return {
        "x_cm": x[idx].tolist(),
        "mu": mu.tolist(),
        "w": w.tolist(),
        "psi_table": psi.tolist(),
        "E_rad_erg_per_cm3": E.tolist(),
        "F_rad_erg_per_cm2_s": F.tolist(),
        "P_rad_dyne_per_cm2": P.tolist(),
        "chi_z": (P / np.maximum(E, TINY)).tolist(),
    }


def build_ced_closure_reference(cfg: SNConfig) -> dict[str, Any]:
    ced_payload = ced_payload_for_config(cfg)
    if ced_payload is None:
        raise RuntimeError("fld_const_eddington_reference_generator.py is not importable")
    inv = build_invariants(cfg)
    table_in = ced_payload["table"]
    table = {key: list(value) for key, value in table_in.items()}
    E = np.asarray(table["E_rad_erg_per_cm3"], dtype=float)
    table["P_rad_dyne_per_cm2"] = (E / 3.0).tolist()
    table["chi_z"] = np.full_like(E, 1.0 / 3.0).tolist()
    if "F_rad_lab_erg_per_cm2_s" not in table:
        table["F_rad_lab_erg_per_cm2_s"] = list(table["F_rad_erg_per_cm2_s"])
    states = {name: dict(state) for name, state in ced_payload["states"].items()}
    for state in states.values():
        state["P_rad"] = state["E_rad"] / 3.0
    mu, w = np.polynomial.legendre.leggauss(cfg.n_angles)
    angular = ced_intensity_block(table, mu, w)
    diagnostics = {
        "outer_iterations": 0,
        "max_T_residual": 0.0,
        "transport_selfidentity_max_rel": 0.0,
        "rh_residual_rel": rh_residuals(states, inv),
        "subshock": {
            "present": True,
            "z_s_cm": 0.0,
            "T_gas_jump": states["shock_downstream"]["T"] - states["shock_upstream"]["T"],
        },
        "advection_fraction": advection_fraction_from_table(table, cfg),
    }
    return payload_from_parts(cfg, inv, table, states, diagnostics, angular, "static_material_CED_identity")


def advection_fraction_from_table(table: dict[str, Any], cfg: SNConfig) -> dict[str, float]:
    x = np.asarray(table["x_cm"], dtype=float)
    u = np.asarray(table["u_cm_per_s"], dtype=float)
    E = np.asarray(table["E_rad_erg_per_cm3"], dtype=float)
    F = np.asarray(table["F_rad_erg_per_cm2_s"], dtype=float)
    mask = np.abs(x) <= 10.0 / cfg.upstream_sigma_cm_inv
    if np.count_nonzero(mask) == 0:
        mask = np.ones_like(x, dtype=bool)
    denom = max(float(np.nanmax(np.abs(F[mask]))), TINY)
    uE = np.abs(u[mask] * E[mask])
    return {
        "max_uE_over_maxFz_window": float(np.nanmax(uE) / denom),
        "max_4thirds_uE_over_maxFz_window": float(np.nanmax((4.0 / 3.0) * uE) / denom),
    }


def payload_from_parts(
    cfg: SNConfig,
    inv: Invariants,
    table: dict[str, Any],
    states: dict[str, dict[str, float]],
    diagnostics: dict[str, Any],
    angular: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "reference_model": model,
        "problem": "SN_static_material_radshock",
        "model": model,
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "tools/validation/sn_radshock_reference_generator.py",
            "git_commit": git_commit(),
            "references": [
                {
                    "label": "Ferguson-Morel-Lowrie 2017",
                    "doi": "10.1016/j.hedp.2017.02.010",
                },
                {
                    "label": "Lowrie 2007",
                    "doi": "10.1007/s00193-007-0081-2",
                },
                {
                    "label": "TENRYU I3 equation audit",
                    "path": "docs/design/i3_sn_equation_audit.md",
                },
            ],
            "flux_convention": (
                "F_rad_erg_per_cm2_s is F_z=sum_l w_l*mu_l*psi_l in the "
                "shock-frame z coordinate.  The static-material reference has "
                "no O(v/c) radiation advection in the governing equations."
            ),
            "shock_matching": (
                "The gas subshock is pinned at z_s=0.  Radiation intensity, "
                "E_rad, P_rad, and F_z are continuous; gas variables jump "
                "between the upstream and downstream momentum roots at the "
                "same F_z, satisfying the local Hugoniot condition."
            ),
        },
        "parameters": {
            "M0": cfg.mach,
            "gamma": cfg.gamma,
            "rho0_g_per_cc": cfg.rho0_gcc,
            "T0_eV": cfg.T0_eV,
            "Cv_erg_per_g_eV": cfg.Cv_erg_per_g_eV,
            "kappa_a_cm2_per_g": cfg.kappa_a_cm2_g,
            "kappa_scale": cfg.kappa_scale,
            "A_amu": cfg.A_amu,
            "zbar": cfg.zbar,
            "n_points": len(table["x_cm"]),
            "n_angles": cfg.n_angles,
            "span_rel": cfg.span_rel,
            "closure": cfg.closure,
            "with_rad_momentum": cfg.with_rad_momentum,
            "quadrature": "gauss_legendre",
            "weights_sum": 2.0,
            "weights_sum_note": "Gauss-Legendre polar weights are normalized on [-1,1], so sum(w)=2.",
            "a_eV_erg_cm3_eV4": A_EV,
            "c_cm_per_s": C_CGS,
        },
        "invariants": {
            "C1": inv.mass_flux,
            "C2": inv.momentum_flux,
            "C3": inv.energy_flux,
            "R_gas_erg_per_g_eV": inv.R_gas,
            "cp_erg_per_g_eV": inv.cp,
            "Cv_erg_per_g_eV": inv.cv,
            "mass_flux_g_per_cm2_s": inv.mass_flux,
            "momentum_flux_dyne_per_cm2": inv.momentum_flux,
            "energy_flux_erg_per_cm2_s": inv.energy_flux,
            "T_sonic_eV": inv.T_sonic,
            "T_downstream_eV": inv.T_down,
        },
        "states": states,
        "diagnostics": diagnostics,
        "angular": angular,
        "table": table,
    }


def build_reference(cfg: SNConfig) -> dict[str, Any]:
    validate_config(cfg)
    if cfg.closure == "ced":
        return build_ced_closure_reference(cfg)
    return build_sn_reference(cfg)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    defaults = load_default_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mach", type=float, default=defaults.mach)
    parser.add_argument("--n-angles", type=int, default=16)
    parser.add_argument("--kappa-scale", type=float, default=1.0)
    parser.add_argument("--n-points", type=int, default=1601)
    parser.add_argument("--span-rel", type=float, default=DEFAULT_SPAN_REL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--closure", choices=("sn", "ced"), default="sn")
    parser.add_argument("--with-rad-momentum", action="store_true")
    parser.add_argument("--relax", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1.0e-12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = load_default_config()
    cfg = SNConfig(
        mach=args.mach,
        n_angles=args.n_angles,
        kappa_scale=args.kappa_scale,
        n_points=args.n_points,
        span_rel=args.span_rel,
        closure=args.closure,
        with_rad_momentum=args.with_rad_momentum,
        relax=args.relax,
        max_iter=args.max_iter,
        tol=args.tol,
        gamma=base.gamma,
        rho0_gcc=base.rho0_gcc,
        T0_eV=base.T0_eV,
        Cv_erg_per_g_eV=base.Cv_erg_per_g_eV,
        A_amu=base.A_amu,
        zbar=base.zbar,
    )
    try:
        payload = build_reference(cfg)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2
    write_json(args.out, payload)
    diag = payload["diagnostics"]
    print(
        f"{args.out}: model={payload['model']} closure={cfg.closure} "
        f"n_points={len(payload['table']['x_cm'])} n_angles={cfg.n_angles} "
        f"outer_iterations={diag['outer_iterations']} "
        f"max_T_residual={diag['max_T_residual']:.3e} "
        f"transport_identity={diag['transport_selfidentity_max_rel']:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
