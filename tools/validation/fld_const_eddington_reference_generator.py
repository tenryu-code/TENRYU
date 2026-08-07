#!/usr/bin/env python3
"""Generate in-house constant-Eddington FLD nED grey-shock references.

The model is the TENRYU equation-matched, low-radiation-pressure nED shock:
gas-only mass/momentum invariants, gas enthalpy plus comoving FLD radiation
flux as the total-energy invariant, and no O(v/c) radiation source terms.
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
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


C_CGS = 2.99792458e10
A_EV = 1.3720e2
EV_TO_ERG = 1.6022e-12
PROTON_MASS_G = 1.6726219e-24
E_FLOOR = 1.0e-300
LOW_PRAD_HARD_GATE = 1.0e-3
IDENTITY_MEDIAN_GATE = 1.0e-3
IDENTITY_P95_GATE = 1.0e-2
TARGET_DOMAIN_SPAN_CM = 50.0
PADDING_POINTS_PER_SIDE = 8


@dataclass(frozen=True)
class FLDCEDConfig:
    mach: float
    gamma: float = 5.0 / 3.0
    rho0_gcc: float = 2.0
    T0_eV: float = 30.0
    kappa_R_cm2_g: float = 0.5
    A_amu: float = 1.0
    zbar: float = 1.0
    n_points: int = 1601
    endpoint_clip_fraction: float = 1.0e-6
    shock_clip_fraction: float = 1.0e-4
    target_domain_span_cm: float = TARGET_DOMAIN_SPAN_CM
    padding_points_per_side: int = PADDING_POINTS_PER_SIDE
    ode_rtol: float = 1.0e-9
    shooting_rtol: float = 1.0e-10


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


@dataclass(frozen=True)
class ShockMatch:
    F_shock: float
    T_upstream_shock: float
    T_downstream_shock: float
    E_shock: float
    T_rad_shock: float
    upstream_length_cm: float
    downstream_length_cm: float


def safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def parse_mach_list(value: str) -> list[float]:
    out = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("mach list is empty")
    return out


def linspace(a: float, b: float, n: int) -> list[float]:
    if n < 2:
        return [float(a)]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def gas_constant(cfg: FLDCEDConfig) -> float:
    return (1.0 + cfg.zbar) * EV_TO_ERG / (cfg.A_amu * PROTON_MASS_G)


def pmat(rho: float, T_eV: float, R_gas: float) -> float:
    return rho * R_gas * T_eV


def erad_eq(T_eV: float) -> float:
    return A_EV * T_eV**4


def prad_from_E(E_rad: float) -> float:
    return E_rad / 3.0


def validate_config(cfg: FLDCEDConfig) -> None:
    if cfg.mach <= 1.0:
        raise ValueError("mach must be supersonic")
    if cfg.gamma <= 1.0:
        raise ValueError("gamma must exceed 1")
    if cfg.rho0_gcc <= 0.0 or cfg.T0_eV <= 0.0 or cfg.kappa_R_cm2_g <= 0.0:
        raise ValueError("rho0, T0, and kappa_R must be positive")
    if cfg.A_amu <= 0.0 or cfg.zbar < 0.0:
        raise ValueError("A_amu must be positive and zbar nonnegative")
    if cfg.n_points < 1000:
        raise ValueError("n_points must be >= 1000 for frozen I1 references")
    if not 0.0 < cfg.endpoint_clip_fraction < 1.0e-2:
        raise ValueError("endpoint_clip_fraction must be in (0, 1e-2)")
    if not 0.0 <= cfg.shock_clip_fraction < 1.0e-2:
        raise ValueError("shock_clip_fraction must be in [0, 1e-2)")
    if cfg.target_domain_span_cm <= 0.0:
        raise ValueError("target_domain_span_cm must be positive")
    if cfg.padding_points_per_side < 1:
        raise ValueError("padding_points_per_side must be positive")


def build_invariants(cfg: FLDCEDConfig) -> Invariants:
    validate_config(cfg)
    R = gas_constant(cfg)
    cv = R / (cfg.gamma - 1.0)
    cp = cfg.gamma / (cfg.gamma - 1.0) * R
    rho0 = cfg.rho0_gcc
    T0 = cfg.T0_eV
    p0 = pmat(rho0, T0, R)
    u0 = cfg.mach * math.sqrt(cfg.gamma * p0 / rho0)
    J = rho0 * u0
    Pi = J * u0 + p0
    Phi = J * (cp * T0 + 0.5 * u0 * u0)

    def disc(T_eV: float) -> float:
        return Pi * Pi - 4.0 * J * J * R * T_eV

    lo = T0
    hi = max(T0 * 1.25, T0 + 1.0)
    while disc(hi) > 0.0:
        hi *= 1.25
        if hi > 1.0e7 * T0:
            raise RuntimeError("failed to bracket sonic temperature")
    for _ in range(160):
        mid = 0.5 * (lo + hi)
        if disc(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    T_sonic = 0.5 * (lo + hi)

    provisional = Invariants(R, cp, cv, rho0, T0, u0, p0, J, Pi, Phi, T_sonic, T0)
    lo = T0
    hi = T_sonic * (1.0 - 1.0e-12)

    def residual(T_eV: float) -> float:
        return gas_energy_flux(T_eV, "downstream", provisional) - Phi

    if residual(lo) >= 0.0 or residual(hi) <= 0.0:
        raise RuntimeError("failed to bracket downstream equilibrium temperature")
    for _ in range(160):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    T_down = 0.5 * (lo + hi)
    return Invariants(R, cp, cv, rho0, T0, u0, p0, J, Pi, Phi, T_sonic, T_down)


def velocity_from_T(T_eV: float, branch: str, inv: Invariants) -> float:
    b = inv.momentum_flux
    disc = b * b - 4.0 * inv.mass_flux * inv.mass_flux * inv.R_gas * T_eV
    if disc < -1.0e-12 * b * b:
        raise FloatingPointError("momentum quadratic has no real root")
    root = math.sqrt(max(disc, 0.0))
    if branch == "upstream":
        return (b + root) / (2.0 * inv.mass_flux)
    if branch == "downstream":
        return (b - root) / (2.0 * inv.mass_flux)
    raise ValueError(f"unknown branch {branch!r}")


def gas_energy_flux(T_eV: float, branch: str, inv: Invariants) -> float:
    u = velocity_from_T(T_eV, branch, inv)
    return inv.mass_flux * (inv.cp * T_eV + 0.5 * u * u)


def radiation_flux_energy(T_eV: float, branch: str, inv: Invariants) -> float:
    return inv.energy_flux - gas_energy_flux(T_eV, branch, inv)


def dF_dT(T_eV: float, branch: str, inv: Invariants) -> float:
    u = velocity_from_T(T_eV, branch, inv)
    denom = inv.mass_flux * (1.0 - inv.R_gas * T_eV / (u * u))
    dudT = -(inv.mass_flux * inv.R_gas / u) / denom
    d_gas_dT = inv.mass_flux * (inv.cp + u * dudT)
    return -d_gas_dT


def branch_far_temperature(branch: str, inv: Invariants) -> float:
    if branch == "upstream":
        return inv.T0
    if branch == "downstream":
        return inv.T_down
    raise ValueError(f"unknown branch {branch!r}")


def temperature_for_flux(F_target: float, branch: str, inv: Invariants) -> float:
    if branch == "upstream":
        lo = inv.T0
    elif branch == "downstream":
        lo = inv.T_down
    else:
        raise ValueError(f"unknown branch {branch!r}")
    hi = inv.T_sonic * (1.0 - 1.0e-10)
    return float(
        brentq(
            lambda T: radiation_flux_energy(T, branch, inv) - F_target,
            lo,
            hi,
            xtol=1.0e-12,
            rtol=1.0e-12,
            maxiter=200,
        )
    )


def far_asymptotic_W_slope(branch: str, inv: Invariants) -> float:
    T_far = branch_far_temperature(branch, inv)
    F_T = dF_dT(T_far, branch, inv)
    A_T = 4.0 * A_EV * T_far**3
    B = 3.0 * F_T * F_T / (C_CGS * C_CGS)
    if branch == "upstream":
        q_E = 0.5 * (A_T + math.sqrt(A_T * A_T + 4.0 * B))
    else:
        q_E = 0.5 * (A_T - math.sqrt(A_T * A_T + 4.0 * B))
    return q_E - A_T


def integrate_branch(
    cfg: FLDCEDConfig,
    inv: Invariants,
    F_shock: float,
    branch: str,
    *,
    dense: bool = False,
) -> Any:
    T_far = branch_far_temperature(branch, inv)
    T_shock = temperature_for_flux(F_shock, branch, inv)
    span = T_shock - T_far
    if not span > 0.0:
        raise RuntimeError(f"{branch} branch has nonpositive temperature span")
    T_start = T_far + max(cfg.endpoint_clip_fraction * span, 1.0e-10)
    W_start = far_asymptotic_W_slope(branch, inv) * (T_start - T_far)
    length_sign = 1.0 if branch == "upstream" else -1.0

    def rhs(T: float, y: Any) -> list[float]:
        W = float(y[0])
        F = radiation_flux_energy(T, branch, inv)
        F_T = dF_dT(T, branch, inv)
        u = velocity_from_T(T, branch, inv)
        sigma_R = cfg.kappa_R_cm2_g * inv.mass_flux / u
        dW_dT = 3.0 * F * F_T / (C_CGS * C_CGS * W) - 4.0 * A_EV * T**3
        dL_dT = -length_sign * F_T / (C_CGS * sigma_R * W)
        return [dW_dT, dL_dT]

    sol = solve_ivp(
        rhs,
        (T_start, T_shock),
        [W_start, 0.0],
        method="DOP853",
        rtol=cfg.ode_rtol,
        atol=[1.0e-8, 1.0e-13],
        max_step=max(abs(span) / 300.0, 1.0e-10),
        dense_output=dense,
    )
    if not sol.success:
        raise RuntimeError(f"{branch} ODE failed: {sol.message}")
    W_end = float(sol.y[0, -1])
    length = float(sol.y[1, -1])
    E_end = erad_eq(T_shock) + W_end
    if not length > 0.0:
        raise RuntimeError(f"{branch} branch length is nonpositive")
    return {
        "solution": sol,
        "T_far": T_far,
        "T_start": T_start,
        "T_shock": T_shock,
        "W_shock": W_end,
        "E_shock": E_end,
        "length_cm": length,
    }


def shock_mismatch(cfg: FLDCEDConfig, inv: Invariants, F_shock: float) -> float:
    up = integrate_branch(cfg, inv, F_shock, "upstream")
    dn = integrate_branch(cfg, inv, F_shock, "downstream")
    return float(up["E_shock"] - dn["E_shock"])


def solve_shock_match(cfg: FLDCEDConfig, inv: Invariants) -> ShockMatch:
    F_min_up = radiation_flux_energy(inv.T_sonic * (1.0 - 1.0e-8), "upstream", inv)
    F_min_dn = radiation_flux_energy(inv.T_sonic * (1.0 - 1.0e-8), "downstream", inv)
    F_min_common = max(F_min_up, F_min_dn)
    if not F_min_common < 0.0:
        raise RuntimeError("common shock-flux bracket is not negative")

    samples = [F_min_common * frac for frac in np.geomspace(1.0e-3, 0.75, 48)]
    bracket: tuple[float, float] | None = None
    previous: tuple[float, float] | None = None
    for F_value in samples:
        try:
            mismatch = shock_mismatch(cfg, inv, float(F_value))
        except (RuntimeError, FloatingPointError, ValueError):
            previous = None
            continue
        if previous is not None and previous[1] * mismatch < 0.0:
            bracket = (previous[0], float(F_value))
            break
        previous = (float(F_value), mismatch)
    if bracket is None:
        raise RuntimeError("failed to bracket continuous E_rad shock match")

    F_shock = float(
        brentq(
            lambda F: shock_mismatch(cfg, inv, F),
            bracket[0],
            bracket[1],
            xtol=max(abs(F_min_common) * cfg.shooting_rtol, 1.0e6),
            rtol=cfg.shooting_rtol,
            maxiter=100,
        )
    )
    up = integrate_branch(cfg, inv, F_shock, "upstream")
    dn = integrate_branch(cfg, inv, F_shock, "downstream")
    E_shock = 0.5 * (float(up["E_shock"]) + float(dn["E_shock"]))
    return ShockMatch(
        F_shock=F_shock,
        T_upstream_shock=float(up["T_shock"]),
        T_downstream_shock=float(dn["T_shock"]),
        E_shock=E_shock,
        T_rad_shock=(E_shock / A_EV) ** 0.25,
        upstream_length_cm=float(up["length_cm"]),
        downstream_length_cm=float(dn["length_cm"]),
    )


def state_from_T_E_F(T_eV: float, E_rad: float, F_rad: float, branch: str, inv: Invariants) -> dict[str, float]:
    u = velocity_from_T(T_eV, branch, inv)
    rho = inv.mass_flux / u
    return {
        "rho": rho,
        "u": u,
        "T": T_eV,
        "T_rad": (max(E_rad, 0.0) / A_EV) ** 0.25,
        "E_rad": E_rad,
        "P_mat": pmat(rho, T_eV, inv.R_gas),
        "P_rad": prad_from_E(E_rad),
        "F_rad": F_rad,
        "F_rad_lab": F_rad,
    }


def sample_branch(
    cfg: FLDCEDConfig,
    inv: Invariants,
    F_shock: float,
    branch: str,
    n_samples: int,
) -> list[dict[str, Any]]:
    integ = integrate_branch(cfg, inv, F_shock, branch, dense=True)
    sol = integ["solution"]
    T_far = float(integ["T_far"])
    T_start = float(integ["T_start"])
    T_shock = float(integ["T_shock"])
    length = float(integ["length_cm"])
    span = T_shock - T_far
    T_last = T_shock - cfg.shock_clip_fraction * span
    if T_last <= T_start:
        T_last = T_shock
    rows: list[dict[str, Any]] = []
    for T in linspace(T_start, T_last, n_samples):
        W, distance_from_far = sol.sol(T)
        W = float(W)
        distance_from_far = float(distance_from_far)
        E_rad = erad_eq(T) + W
        F_rad = radiation_flux_energy(T, branch, inv)
        if branch == "upstream":
            x = distance_from_far - length
        else:
            x = length - distance_from_far
        rows.append(
            {
                "x_cm": x,
                "rho_g_per_cc": inv.mass_flux / velocity_from_T(T, branch, inv),
                "u_cm_per_s": velocity_from_T(T, branch, inv),
                "T_eV": T,
                "T_rad_eV": (max(E_rad, 0.0) / A_EV) ** 0.25,
                "E_rad_erg_per_cm3": E_rad,
                "P_mat_dyne_per_cm2": pmat(
                    inv.mass_flux / velocity_from_T(T, branch, inv),
                    T,
                    inv.R_gas,
                ),
                "P_rad_dyne_per_cm2": prad_from_E(E_rad),
                "F_rad_erg_per_cm2_s": F_rad,
                "F_rad_energy_invariant_erg_per_cm2_s": F_rad,
                "F_rad_lab_erg_per_cm2_s": F_rad,
                "branch": branch,
            }
        )
    if branch == "downstream":
        rows.reverse()
    return rows


def build_table(cfg: FLDCEDConfig, inv: Invariants, match: ShockMatch) -> dict[str, list[Any]]:
    n_up = cfg.n_points // 2
    n_dn = cfg.n_points - n_up
    rows = sample_branch(cfg, inv, match.F_shock, "upstream", n_up)
    rows.extend(sample_branch(cfg, inv, match.F_shock, "downstream", n_dn))
    rows.sort(key=lambda row: float(row["x_cm"]))
    fields = (
        "x_cm",
        "rho_g_per_cc",
        "u_cm_per_s",
        "T_eV",
        "T_rad_eV",
        "E_rad_erg_per_cm3",
        "P_mat_dyne_per_cm2",
        "P_rad_dyne_per_cm2",
        "F_rad_erg_per_cm2_s",
        "F_rad_energy_invariant_erg_per_cm2_s",
        "F_rad_lab_erg_per_cm2_s",
        "branch",
    )
    return {field: [row[field] for row in rows] for field in fields}


def equilibrium_row(x_cm: float, state: dict[str, float], branch: str) -> dict[str, Any]:
    e_rad = erad_eq(state["T"])
    return {
        "x_cm": x_cm,
        "rho_g_per_cc": state["rho"],
        "u_cm_per_s": state["u"],
        "T_eV": state["T"],
        "T_rad_eV": state["T"],
        "E_rad_erg_per_cm3": e_rad,
        "P_mat_dyne_per_cm2": state["P_mat"],
        "P_rad_dyne_per_cm2": prad_from_E(e_rad),
        "F_rad_erg_per_cm2_s": 0.0,
        "F_rad_energy_invariant_erg_per_cm2_s": 0.0,
        "F_rad_lab_erg_per_cm2_s": 0.0,
        "branch": branch,
    }


def pad_table_to_domain(
    table: dict[str, list[Any]],
    upstream: dict[str, float],
    downstream: dict[str, float],
    cfg: FLDCEDConfig,
) -> dict[str, list[Any]]:
    x = table_array(table, "x_cm")
    current_span = float(x[-1] - x[0])
    if current_span >= cfg.target_domain_span_cm:
        return table
    extra = 0.5 * (cfg.target_domain_span_cm - current_span)
    x_min = float(x[0])
    x_max = float(x[-1])
    left = linspace(x_min - extra, x_min, cfg.padding_points_per_side + 1)[:-1]
    right = linspace(x_max, x_max + extra, cfg.padding_points_per_side + 1)[1:]
    rows = [equilibrium_row(value, upstream, "upstream") for value in left]
    rows.extend(
        {field: table[field][i] for field in table}
        for i in range(len(table["x_cm"]))
    )
    rows.extend(equilibrium_row(value, downstream, "downstream") for value in right)
    rows.sort(key=lambda row: float(row["x_cm"]))
    return {field: [row[field] for row in rows] for field in table}


def table_array(table: dict[str, list[Any]], key: str) -> np.ndarray:
    return np.asarray(table[key], dtype=float)


def branch_identity_stats(table: dict[str, list[Any]], cfg: FLDCEDConfig) -> dict[str, dict[str, float]]:
    x = table_array(table, "x_cm")
    rho = table_array(table, "rho_g_per_cc")
    e_rad = table_array(table, "E_rad_erg_per_cm3")
    f_ref = table_array(table, "F_rad_erg_per_cm2_s")
    branches = np.asarray(table["branch"], dtype=str)
    out: dict[str, dict[str, float]] = {}
    rels_all: list[np.ndarray] = []
    for branch in ("upstream", "downstream"):
        mask = branches == branch
        if np.count_nonzero(mask) < 3:
            values = np.asarray([math.inf], dtype=float)
        else:
            order = np.argsort(x[mask])
            x_b = x[mask][order]
            rho_b = rho[mask][order]
            e_b = e_rad[mask][order]
            f_b = f_ref[mask][order]
            f_grad = -C_CGS / (3.0 * cfg.kappa_R_cm2_g * np.maximum(rho_b, E_FLOOR)) * np.gradient(e_b, x_b)
            rel = np.abs((f_b - f_grad) / np.maximum(np.abs(f_b) + np.abs(f_grad), E_FLOOR))
            values = rel[10:-10] if rel.size > 20 else rel
        rels_all.append(values)
        out[branch] = {
            "median_rel": float(np.median(values)),
            "p95_rel": float(np.percentile(values, 95.0)),
        }
    total = np.concatenate(rels_all)
    out["total"] = {
        "median_rel": float(np.median(total)),
        "p95_rel": float(np.percentile(total, 95.0)),
    }
    return out


def max_gradient_ratio(table: dict[str, list[Any]]) -> float:
    x = table_array(table, "x_cm")
    p_mat = table_array(table, "P_mat_dyne_per_cm2")
    p_rad = table_array(table, "P_rad_dyne_per_cm2")
    branches = np.asarray(table["branch"], dtype=str)
    ratios: list[float] = []
    for branch in ("upstream", "downstream"):
        mask = branches == branch
        if np.count_nonzero(mask) < 3:
            return math.inf
        order = np.argsort(x[mask])
        x_b = x[mask][order]
        grad_mat = np.gradient(p_mat[mask][order], x_b)
        grad_rad = np.gradient(p_rad[mask][order], x_b)
        scale = float(np.nanmax(np.abs(grad_mat)))
        threshold = max(1.0e-12 * scale, E_FLOOR)
        local = np.isfinite(grad_mat) & np.isfinite(grad_rad) & (np.abs(grad_mat) >= threshold)
        ratios.extend(
            float(abs(gr) / max(abs(gm), E_FLOOR))
            for gm, gr in zip(grad_mat[local], grad_rad[local])
        )
    return max(ratios) if ratios else math.inf


def conservation_diagnostics(table: dict[str, list[Any]], cfg: FLDCEDConfig, inv: Invariants) -> dict[str, float]:
    rho = table_array(table, "rho_g_per_cc")
    u = table_array(table, "u_cm_per_s")
    T = table_array(table, "T_eV")
    p = table_array(table, "P_mat_dyne_per_cm2")
    pr = table_array(table, "P_rad_dyne_per_cm2")
    F = table_array(table, "F_rad_energy_invariant_erg_per_cm2_s")
    mass = rho * u
    mom = rho * u * u + p
    energy = inv.mass_flux * (inv.cp * T + 0.5 * u * u) + F

    def residual(values: np.ndarray, ref_value: float) -> float:
        return float(np.nanmax(np.abs(values - ref_value)) / max(abs(ref_value), E_FLOOR))

    prad_ratio = float(np.nanmax(np.abs(pr) / np.maximum(np.abs(p), E_FLOOR)))
    identity = branch_identity_stats(table, cfg)
    temp_sep = np.abs(table_array(table, "T_rad_eV") / np.maximum(table_array(table, "T_eV"), E_FLOOR) - 1.0)
    mass_res = residual(mass, inv.mass_flux)
    mom_res = residual(mom, inv.momentum_flux)
    energy_res = residual(energy, inv.energy_flux)
    return {
        "mass_flux_residual_rel_max": mass_res,
        "momentum_flux_residual_rel_max": mom_res,
        "energy_flux_residual_rel_max": energy_res,
        "conservation_residual_rel_max": max(mass_res, mom_res, energy_res),
        "max_P_rad_over_P_mat": prad_ratio,
        "max_abs_grad_P_rad_over_abs_grad_P_mat": max_gradient_ratio(table),
        "max_abs_T_rad_over_T_e_minus_1": float(np.nanmax(temp_sep)),
        "reference_identity_upstream_median_rel": identity["upstream"]["median_rel"],
        "reference_identity_upstream_p95_rel": identity["upstream"]["p95_rel"],
        "reference_identity_downstream_median_rel": identity["downstream"]["median_rel"],
        "reference_identity_downstream_p95_rel": identity["downstream"]["p95_rel"],
    }


def git_commit(path: Path | None = None) -> str | None:
    cmd = ["git"]
    if path is not None:
        cmd += ["-C", str(path)]
    cmd += ["rev-parse", "HEAD"]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def build_reference(cfg: FLDCEDConfig) -> dict[str, Any]:
    inv = build_invariants(cfg)
    match = solve_shock_match(cfg, inv)
    upstream = state_from_T_E_F(inv.T0, erad_eq(inv.T0), 0.0, "upstream", inv)
    downstream = state_from_T_E_F(inv.T_down, erad_eq(inv.T_down), 0.0, "downstream", inv)
    table = build_table(cfg, inv, match)
    table = pad_table_to_domain(table, upstream, downstream, cfg)
    diagnostics = conservation_diagnostics(table, cfg, inv)
    if diagnostics["max_P_rad_over_P_mat"] > LOW_PRAD_HARD_GATE:
        raise RuntimeError(
            "low-P_rad hard gate failed: "
            f"max_P_rad/P_mat={diagnostics['max_P_rad_over_P_mat']:.6e} > {LOW_PRAD_HARD_GATE:.1e}"
        )
    identity = branch_identity_stats(table, cfg)
    for branch in ("upstream", "downstream"):
        if (
            identity[branch]["median_rel"] >= IDENTITY_MEDIAN_GATE
            or identity[branch]["p95_rel"] >= IDENTITY_P95_GATE
        ):
            raise RuntimeError(
                f"{branch} diffusion identity failed: "
                f"median={identity[branch]['median_rel']:.3e}, "
                f"p95={identity[branch]['p95_rel']:.3e}"
            )
    payload: dict[str, Any] = {
        "schema_version": 3,
        "reference_model": "FLD_const_Eddington_nED_low_Prad",
        "problem": "FLD_const_Eddington",
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "tools/validation/fld_const_eddington_reference_generator.py",
            "git_commit": git_commit(),
            "reference_model": "FLD_const_Eddington_nED_low_Prad",
            "low_P_rad_hard_gate": LOW_PRAD_HARD_GATE,
            "low_P_rad_diagnostics": {
                "max_P_rad_over_P_mat": diagnostics["max_P_rad_over_P_mat"],
                "max_abs_grad_P_rad_over_abs_grad_P_mat": diagnostics[
                    "max_abs_grad_P_rad_over_abs_grad_P_mat"
                ],
            },
            "flux_convention": (
                "F_rad_erg_per_cm2_s is the comoving constant-Eddington FLD flux "
                "conjugate to monotone x_cm. Because this in-house model omits "
                "O(v/c) source terms, F_rad_energy_invariant_erg_per_cm2_s and "
                "F_rad_lab_erg_per_cm2_s are identical to the comoving flux."
            ),
            "shock_matching": (
                "The shooting variable is the common gas-shock radiation flux. "
                "Gas variables jump between the upstream and downstream hydro "
                "branches; E_rad and F_rad are continuous across the front."
            ),
            "references": [
                {
                    "label": "Mihalas and Mihalas",
                    "section": "97",
                    "note": "constant-Eddington diffusion-limit moment equations",
                },
                {
                    "label": "Lowrie 2007",
                    "section": "2",
                    "note": "steady radiative-shock ODE shooting technique",
                },
            ],
        },
        "parameters": {
            "M0": cfg.mach,
            "gamma": cfg.gamma,
            "rho0_g_per_cc": cfg.rho0_gcc,
            "T0_eV": cfg.T0_eV,
            "kappa_R_cm2_per_g": cfg.kappa_R_cm2_g,
            "A_amu": cfg.A_amu,
            "zbar": cfg.zbar,
            "n_points": len(table["x_cm"]),
            "endpoint_clip_fraction": cfg.endpoint_clip_fraction,
            "shock_clip_fraction": cfg.shock_clip_fraction,
            "ode_n_points": cfg.n_points,
            "target_domain_span_cm": cfg.target_domain_span_cm,
            "padding_points_per_side": cfg.padding_points_per_side,
            "problem": "FLD_const_Eddington",
            "reference_model": "FLD_const_Eddington_nED_low_Prad",
            "a_eV_erg_cm3_eV4": A_EV,
            "c_cm_per_s": C_CGS,
            "eV_to_erg": EV_TO_ERG,
            "proton_mass_g": PROTON_MASS_G,
            "Cv_erg_per_g_eV": inv.cv,
        },
        "invariants": {
            "R_gas_erg_per_g_eV": inv.R_gas,
            "cp_erg_per_g_eV": inv.cp,
            "Cv_erg_per_g_eV": inv.cv,
            "mass_flux_g_per_cm2_s": inv.mass_flux,
            "momentum_flux_dyne_per_cm2": inv.momentum_flux,
            "energy_flux_erg_per_cm2_s": inv.energy_flux,
            "T_sonic_eV": inv.T_sonic,
            "T_downstream_eV": inv.T_down,
            "F_shock_erg_per_cm2_s": match.F_shock,
            "T_upstream_shock_eV": match.T_upstream_shock,
            "T_downstream_shock_eV": match.T_downstream_shock,
            "E_rad_shock_erg_per_cm3": match.E_shock,
            "T_rad_shock_eV": match.T_rad_shock,
            "upstream_branch_length_cm": match.upstream_length_cm,
            "downstream_branch_length_cm": match.downstream_length_cm,
        },
        "states": {
            "upstream": upstream,
            "downstream": downstream,
            "shock_upstream": state_from_T_E_F(
                match.T_upstream_shock,
                match.E_shock,
                match.F_shock,
                "upstream",
                inv,
            ),
            "shock_downstream": state_from_T_E_F(
                match.T_downstream_shock,
                match.E_shock,
                match.F_shock,
                "downstream",
                inv,
            ),
        },
        "diagnostics": diagnostics,
        "reference_identity_branch_wise": identity,
        "table": table,
    }
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="tests/verification/data/fld_const_eddington_reference")
    parser.add_argument("--mach-list", type=parse_mach_list, default=parse_mach_list("1.5,2.0,3.0"))
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--rho0-gcc", type=float, default=2.0)
    parser.add_argument("--T0-eV", type=float, default=30.0)
    parser.add_argument("--kappa-R-cm2-g", type=float, default=0.5)
    parser.add_argument("--A-amu", type=float, default=1.0)
    parser.add_argument("--zbar", type=float, default=1.0)
    parser.add_argument("--n-points", type=int, default=1601)
    parser.add_argument("--endpoint-clip-fraction", type=float, default=1.0e-6)
    parser.add_argument("--shock-clip-fraction", type=float, default=1.0e-4)
    parser.add_argument("--target-domain-span-cm", type=float, default=TARGET_DOMAIN_SPAN_CM)
    parser.add_argument("--padding-points-per-side", type=int, default=PADDING_POINTS_PER_SIDE)
    parser.add_argument("--ode-rtol", type=float, default=1.0e-9)
    parser.add_argument("--shooting-rtol", type=float, default=1.0e-10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    rows: list[dict[str, Any]] = []
    for mach in args.mach_list:
        cfg = FLDCEDConfig(
            mach=mach,
            gamma=args.gamma,
            rho0_gcc=args.rho0_gcc,
            T0_eV=args.T0_eV,
            kappa_R_cm2_g=args.kappa_R_cm2_g,
            A_amu=args.A_amu,
            zbar=args.zbar,
            n_points=args.n_points,
            endpoint_clip_fraction=args.endpoint_clip_fraction,
            shock_clip_fraction=args.shock_clip_fraction,
            target_domain_span_cm=args.target_domain_span_cm,
            padding_points_per_side=args.padding_points_per_side,
            ode_rtol=args.ode_rtol,
            shooting_rtol=args.shooting_rtol,
        )
        payload = build_reference(cfg)
        filename = f"fld_ced_M{safe_float_token(mach)}.json"
        write_json(out_dir / filename, payload)
        diag = payload["diagnostics"]
        rows.append(
            {
                "file": filename,
                "M0": mach,
                "schema_version": payload["schema_version"],
                "problem": payload["problem"],
                "reference_model": payload["reference_model"],
                "n_points": len(payload["table"]["x_cm"]),
                "ode_n_points": cfg.n_points,
                "target_domain_span_cm": cfg.target_domain_span_cm,
                "domain_span_cm": float(payload["table"]["x_cm"][-1] - payload["table"]["x_cm"][0]),
                "max_P_rad_over_P_mat": diag["max_P_rad_over_P_mat"],
                "max_abs_grad_P_rad_over_abs_grad_P_mat": diag[
                    "max_abs_grad_P_rad_over_abs_grad_P_mat"
                ],
                "max_abs_T_rad_over_T_e_minus_1": diag["max_abs_T_rad_over_T_e_minus_1"],
                "conservation_residual_rel_max": diag["conservation_residual_rel_max"],
                "reference_identity_upstream_median_rel": diag["reference_identity_upstream_median_rel"],
                "reference_identity_upstream_p95_rel": diag["reference_identity_upstream_p95_rel"],
                "reference_identity_downstream_median_rel": diag["reference_identity_downstream_median_rel"],
                "reference_identity_downstream_p95_rel": diag["reference_identity_downstream_p95_rel"],
                "generated_utc": payload["provenance"]["generated_utc"],
            }
        )
        print(
            f"{filename}: domain_span={payload['table']['x_cm'][-1] - payload['table']['x_cm'][0]:.6e} "
            f"max_P_rad/P_mat={diag['max_P_rad_over_P_mat']:.6e} "
            f"max_grad_ratio={diag['max_abs_grad_P_rad_over_abs_grad_P_mat']:.6e} "
            f"Tsep={diag['max_abs_T_rad_over_T_e_minus_1']:.6e} "
            f"identity_up=({diag['reference_identity_upstream_median_rel']:.3e},"
            f"{diag['reference_identity_upstream_p95_rel']:.3e}) "
            f"identity_dn=({diag['reference_identity_downstream_median_rel']:.3e},"
            f"{diag['reference_identity_downstream_p95_rel']:.3e}) "
            f"conservation={diag['conservation_residual_rel_max']:.6e}"
        )
    manifest = {
        "schema_version": "tenryu.fld_const_eddington_reference_manifest.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/validation/fld_const_eddington_reference_generator.py",
        "git_commit": git_commit(),
        "rows": rows,
    }
    write_json(out_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
