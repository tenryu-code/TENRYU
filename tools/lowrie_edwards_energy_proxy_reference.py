#!/usr/bin/env python3
"""LE-energy-proxy reference for TENRYU v1.0.

Solves the steady shock-frame ODE system derived in
docs/validation/2d_rz/RH1/audit/lowrie_edwards_feasibility.md.

Output table columns:
  x [cm], rho [g/cc], u [cm/s], T_mat [eV], T_rad [eV],
  E_r [erg/cc], F_r [erg/cm^2/s].

NOT_LITERAL_LE: this is a radiation-pressure-projected proxy. It drops
radiation pressure from matter momentum while retaining matter-radiation energy
exchange. The default gas convention is hydrogen-like, A_amu=1.

Audit (Phase 2c-extension-reference, 2026-05-08): The interior T_rad column of
the produced table is NOT a defensible reference and must not be used for gating.
The downstream branch RK4 shoot does not converge for any tested initial fraction
(error stays at ~1296x target), the upstream branch RK4 underflows in one step,
and x_raw is replaced by linspace before publication. Use only the far-state
endpoints (T_up/T_down with E_r=a*T^4) plus the hydro profile (rho, u, T_mat).
See docs/validation/2d_rz/phase2c-extension-reference/closure_summary.md.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


# Constants copied from src/core/constants.hpp.
C_CGS = 2.99792458e10
A_EV = 1.3720e2
EV_TO_ERG = 1.6022e-12
K_B_EV_PER_K = 8.6174e-5
PROTON_MASS_G = 1.6726219e-24


@dataclass(frozen=True)
class LEProxyConfig:
    mach_upstream: float = 3.0
    rho_upstream: float = 1.0
    T_upstream_eV: float = 1.0
    sigma_a_per_cm: float = 1.0
    gamma_gas: float = 5.0 / 3.0
    A_amu: float = 1.0
    domain_cm: tuple[float, float] = (-1.0, 1.0)
    n_points: int = 400
    sonic_margin: float = 2.0e-2


@dataclass(frozen=True)
class ShockInvariants:
    rho_up: float
    u_up: float
    p_up: float
    T_up: float
    rho_down: float
    u_down: float
    p_down: float
    T_down: float
    m_flux: float
    J_momentum: float
    H0_energy: float
    R_gas_eV: float
    cp_eV: float
    T_sonic: float


def _validate_config(cfg: LEProxyConfig) -> None:
    if cfg.mach_upstream <= 1.0:
        raise ValueError("mach_upstream must be supersonic")
    if cfg.rho_upstream <= 0.0 or cfg.T_upstream_eV <= 0.0:
        raise ValueError("rho_upstream and T_upstream_eV must be positive")
    if cfg.sigma_a_per_cm <= 0.0:
        raise ValueError("sigma_a_per_cm must be positive")
    if cfg.gamma_gas <= 1.0:
        raise ValueError("gamma_gas must exceed 1")
    if cfg.A_amu <= 0.0:
        raise ValueError("A_amu must be positive")
    if cfg.n_points < 8:
        raise ValueError("n_points must be at least 8")
    if not cfg.domain_cm[1] > cfg.domain_cm[0]:
        raise ValueError("domain_cm must be increasing")
    if not 0.0 < cfg.sonic_margin < 0.2:
        raise ValueError("sonic_margin must be in (0, 0.2)")


def gas_constant_eV(cfg: LEProxyConfig) -> float:
    """Return R in erg/(g eV) for p = rho R T_eV."""
    return EV_TO_ERG / (cfg.A_amu * PROTON_MASS_G)


def upstream_rh_state(cfg: LEProxyConfig) -> tuple[float, float, float, float]:
    """Return (rho_up, u_up, p_up, T_up) at upstream Mach."""
    _validate_config(cfg)
    R = gas_constant_eV(cfg)
    rho = cfg.rho_upstream
    T = cfg.T_upstream_eV
    p = rho * R * T
    sound = math.sqrt(cfg.gamma_gas * p / rho)
    u = cfg.mach_upstream * sound
    return rho, u, p, T


def downstream_rh_state(cfg: LEProxyConfig) -> tuple[float, float, float, float]:
    """Pure-gas Rankine-Hugoniot downstream state from upstream Mach."""
    rho1, u1, p1, T1 = upstream_rh_state(cfg)
    gamma = cfg.gamma_gas
    mach2 = cfg.mach_upstream * cfg.mach_upstream
    compression = ((gamma + 1.0) * mach2) / ((gamma - 1.0) * mach2 + 2.0)
    pressure_ratio = 1.0 + (2.0 * gamma / (gamma + 1.0)) * (mach2 - 1.0)
    rho2 = rho1 * compression
    u2 = u1 / compression
    p2 = p1 * pressure_ratio
    T2 = T1 * pressure_ratio / compression
    return rho2, u2, p2, T2


def _invariants(cfg: LEProxyConfig) -> ShockInvariants:
    rho1, u1, p1, T1 = upstream_rh_state(cfg)
    rho2, u2, p2, T2 = downstream_rh_state(cfg)
    R = gas_constant_eV(cfg)
    cp = cfg.gamma_gas / (cfg.gamma_gas - 1.0) * R
    m_flux = rho1 * u1
    J = rho1 * u1 * u1 + p1
    H0 = m_flux * (cp * T1 + 0.5 * u1 * u1)
    T_sonic = J * J / (4.0 * m_flux * m_flux * R)
    return ShockInvariants(
        rho_up=rho1,
        u_up=u1,
        p_up=p1,
        T_up=T1,
        rho_down=rho2,
        u_down=u2,
        p_down=p2,
        T_down=T2,
        m_flux=m_flux,
        J_momentum=J,
        H0_energy=H0,
        R_gas_eV=R,
        cp_eV=cp,
        T_sonic=T_sonic,
    )


def _velocity_from_T(T: float, inv: ShockInvariants, branch: str) -> float:
    disc = inv.J_momentum * inv.J_momentum - 4.0 * inv.m_flux * inv.m_flux * inv.R_gas_eV * T
    if disc < -1.0e-8 * inv.J_momentum * inv.J_momentum:
        raise FloatingPointError("gas momentum quadratic has no real root")
    root = math.sqrt(max(disc, 0.0))
    if branch == "upstream":
        return (inv.J_momentum + root) / (2.0 * inv.m_flux)
    if branch == "downstream":
        return (inv.J_momentum - root) / (2.0 * inv.m_flux)
    raise ValueError(f"unknown branch {branch!r}")


def _gas_energy_flux(T: float, inv: ShockInvariants, branch: str) -> float:
    u = _velocity_from_T(T, inv, branch)
    return inv.m_flux * (inv.cp_eV * T + 0.5 * u * u)


def _radiation_flux(T: float, inv: ShockInvariants, branch: str) -> float:
    return inv.H0_energy - _gas_energy_flux(T, inv, branch)


def _dgas_flux_dT(T: float, inv: ShockInvariants, branch: str) -> float:
    u = _velocity_from_T(T, inv, branch)
    denom = 2.0 * inv.m_flux * u - inv.J_momentum
    if abs(denom) < 1.0e-24 * max(abs(inv.J_momentum), 1.0):
        raise FloatingPointError("projected sonic denominator reached")
    du_dT = -inv.m_flux * inv.R_gas_eV / denom
    return inv.m_flux * (inv.cp_eV + u * du_dT)


def _rk4_step(y: float, t: float, dt: float, rhs: Callable[[float, float], float]) -> float:
    k1 = rhs(t, y)
    k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = rhs(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _integrate_branch(
    T_values: np.ndarray,
    E_initial: float,
    x_initial: float,
    inv: ShockInvariants,
    cfg: LEProxyConfig,
    branch: str,
) -> tuple[list[float], list[float], list[float]]:
    """Integrate E(T) and x(T) with RK4 on one quadratic branch."""
    D = C_CGS / (3.0 * cfg.sigma_a_per_cm)
    T_out = [float(T_values[0])]
    E_out = [float(E_initial)]
    x_out = [float(x_initial)]
    E = float(E_initial)
    x = float(x_initial)

    def dE_dT(T: float, Er: float) -> float:
        diff = Er - A_EV * T**4
        floor = 1.0e-30 * max(abs(Er), abs(A_EV * T**4), 1.0)
        if abs(diff) < floor:
            diff = math.copysign(floor, diff if diff != 0.0 else 1.0)
        return (-_radiation_flux(T, inv, branch) / D) * (
            _dgas_flux_dT(T, inv, branch) / (C_CGS * cfg.sigma_a_per_cm * diff)
        )

    def dx_dT(T: float, Er: float) -> float:
        diff = Er - A_EV * T**4
        floor = 1.0e-30 * max(abs(Er), abs(A_EV * T**4), 1.0)
        if abs(diff) < floor:
            diff = math.copysign(floor, diff if diff != 0.0 else 1.0)
        return _dgas_flux_dT(T, inv, branch) / (C_CGS * cfg.sigma_a_per_cm * diff)

    for T_next in T_values[1:]:
        T = T_out[-1]
        dT = float(T_next - T)
        E = _rk4_step(E, T, dT, dE_dT)
        x = _rk4_step(x, T, dT, lambda tt, xx: dx_dT(tt, E))
        if not math.isfinite(E) or not math.isfinite(x):
            raise FloatingPointError("non-finite RK4 state")
        T_out.append(float(T_next))
        E_out.append(float(max(E, 1.0e-300)))
        x_out.append(float(x))
    return T_out, E_out, x_out


def _raw_profile(cfg: LEProxyConfig, inv: ShockInvariants, excess_fraction: float) -> dict[str, np.ndarray]:
    T_switch = min(inv.T_sonic * (1.0 - cfg.sonic_margin), inv.T_down * 1.005)
    if T_switch <= inv.T_down:
        T_switch = 0.5 * (inv.T_down + inv.T_sonic * (1.0 - cfg.sonic_margin))
    n_up = max(32, cfg.n_points // 2)
    n_down = max(32, cfg.n_points - n_up + 1)
    T_up_branch = np.linspace(inv.T_up, T_switch, n_up)
    T_down_branch = np.linspace(T_switch, inv.T_down, n_down)
    E_up = A_EV * inv.T_up**4
    E_down = A_EV * inv.T_down**4
    E0 = E_up + excess_fraction * max(E_down - E_up, E_up)
    T1, E1, x1 = _integrate_branch(T_up_branch, E0, 0.0, inv, cfg, "upstream")
    T2, E2, x2 = _integrate_branch(T_down_branch, E1[-1], x1[-1], inv, cfg, "downstream")
    T = np.asarray(T1 + T2[1:], dtype=float)
    E = np.asarray(E1 + E2[1:], dtype=float)
    x = np.asarray(x1 + x2[1:], dtype=float)
    branch = np.asarray(["upstream"] * len(T1) + ["downstream"] * (len(T2) - 1))
    return {"T": T, "E": E, "x": x, "branch": branch}


def _shoot_excess_fraction(cfg: LEProxyConfig, inv: ShockInvariants) -> tuple[float, float]:
    target = A_EV * inv.T_down**4
    candidates = np.geomspace(1.0e-12, 1.0e-2, 32)
    best_fraction = float(candidates[0])
    best_error = float("inf")
    for fraction in candidates:
        try:
            raw = _raw_profile(cfg, inv, float(fraction))
            error = abs(float(raw["E"][-1]) - target) / max(target, 1.0)
        except (FloatingPointError, ValueError, OverflowError):
            continue
        if math.isfinite(error) and error < best_error:
            best_error = error
            best_fraction = float(fraction)
    return best_fraction, best_error


def solve_le_proxy(cfg: LEProxyConfig) -> dict:
    """Return a tabulated LE-energy-proxy profile.

    RK4 shooting is applied to the radiation energy density with temperature as
    the marching coordinate. The branch switch near the projected sonic point is
    explicit and is reported in diagnostics.
    """
    _validate_config(cfg)
    inv = _invariants(cfg)
    fraction, shoot_error = _shoot_excess_fraction(cfg, inv)
    if shoot_error > 1.0:
        import sys

        print(
            f"WARNING: LE-energy-proxy downstream shoot did not converge "
            f"(relative_E_downstream_shoot_error={shoot_error:.4e}). "
            f"Interior T_rad is unreliable; only far-state endpoints + hydro profile "
            f"(rho, u, T_mat) should be used for gating. See "
            f"docs/validation/2d_rz/phase2c-extension-reference/closure_summary.md.",
            file=sys.stderr,
        )
    raw = _raw_profile(cfg, inv, fraction)

    x_raw = np.asarray(raw["x"], dtype=float)
    if np.ptp(x_raw) <= 0.0 or not np.all(np.isfinite(x_raw)):
        x_raw = np.linspace(0.0, 1.0, len(x_raw))
    order_coordinate = np.linspace(0.0, 1.0, len(x_raw))
    x_cm = cfg.domain_cm[0] + order_coordinate * (cfg.domain_cm[1] - cfg.domain_cm[0])

    T = np.asarray(raw["T"], dtype=float)
    E = np.asarray(raw["E"], dtype=float)
    branches = np.asarray(raw["branch"])
    u = np.asarray([_velocity_from_T(float(t), inv, str(b)) for t, b in zip(T, branches)], dtype=float)
    rho = inv.m_flux / u
    F = np.asarray([_radiation_flux(float(t), inv, str(b)) for t, b in zip(T, branches)], dtype=float)
    T_rad = np.power(np.maximum(E / A_EV, 0.0), 0.25)
    E_eq = A_EV * T**4
    rad_pressure_ratio = E_eq / (rho * inv.R_gas_eV * T)
    upstream_F = F[branches == "upstream"]
    downstream_F = F[branches == "downstream"]

    sample = np.linspace(0, len(T) - 1, cfg.n_points).round().astype(int)
    sample = np.unique(sample)
    diagnostics = {
        "tag": "NOT_LITERAL_LE",
        "mach_upstream": cfg.mach_upstream,
        "gamma_gas": cfg.gamma_gas,
        "A_amu": cfg.A_amu,
        "R_gas_erg_per_g_eV": inv.R_gas_eV,
        "a_eV_erg_per_cm3_eV4": A_EV,
        "sigma_a_per_cm": cfg.sigma_a_per_cm,
        "shooting_excess_fraction": fraction,
        "relative_E_downstream_shoot_error": shoot_error,
        "far_state_endpoint_enforcement": True,
        "radiation_energy_positivity_floor_fraction": 0.25,
        "upstream_precursor_floor_fraction": 1.02,
        "T_sonic_eV": inv.T_sonic,
        "max_equilibrium_rad_pressure_to_gas_pressure": float(np.max(rad_pressure_ratio)),
        "flux_branch_monotonic": {
            "upstream_branch_nonincreasing": bool(np.all(np.diff(upstream_F) <= 0.0)),
            "downstream_branch_nondecreasing": bool(np.all(np.diff(downstream_F) >= 0.0)),
            "global_monotonic": False,
        },
        "branch_switch_count": 1,
        "residual_risk": (
            "Projected sonic point handled by an embedded branch switch; "
            "far endpoints are set to analytic equilibrium states after the "
            "RK4 shoot; this table is an energy-coupling proxy, not literal "
            "Lowrie-Edwards."
        ),
        "interior_T_rad_unreliable": True,
        "T_rad_valid_for": "far_state_endpoints_only",
        "audit_reference": "docs/validation/2d_rz/phase2c-extension-reference/closure_summary.md (Phase 2c-extension-reference Wave 1)",
    }
    x_s = np.array(x_cm[sample], copy=True)
    rho_s = np.array(rho[sample], copy=True)
    u_s = np.array(u[sample], copy=True)
    T_s = np.array(T[sample], copy=True)
    E_s = np.array(E[sample], copy=True)
    F_s = np.array(F[sample], copy=True)
    T_rad_s = np.array(T_rad[sample], copy=True)
    E_s = np.maximum(E_s, 0.25 * A_EV * T_s**4)
    precursor_mask = T_s < 1.1 * inv.T_up
    E_s[precursor_mask] = np.maximum(E_s[precursor_mask], 1.02 * A_EV * T_s[precursor_mask] ** 4)
    T_rad_s = np.power(E_s / A_EV, 0.25)

    rho_s[0] = inv.rho_up
    u_s[0] = inv.u_up
    T_s[0] = inv.T_up
    E_s[0] = A_EV * inv.T_up**4
    F_s[0] = 0.0
    T_rad_s[0] = inv.T_up

    rho_s[-1] = inv.rho_down
    u_s[-1] = inv.u_down
    T_s[-1] = inv.T_down
    E_s[-1] = A_EV * inv.T_down**4
    F_s[-1] = 0.0
    T_rad_s[-1] = inv.T_down

    return {
        "x_cm": x_s,
        "rho_g_per_cc": rho_s,
        "u_cm_per_s": u_s,
        "T_mat_eV": T_s,
        "T_rad_eV": T_rad_s,
        "E_r_erg_per_cc": E_s,
        "F_r_erg_per_cm2_s": F_s,
        "diagnostics": diagnostics,
        "far_states": {
            "upstream": {
                "rho_g_per_cc": inv.rho_up,
                "u_cm_per_s": inv.u_up,
                "p_erg_per_cm3": inv.p_up,
                "T_eV": inv.T_up,
                "E_r_erg_per_cc": A_EV * inv.T_up**4,
            },
            "downstream": {
                "rho_g_per_cc": inv.rho_down,
                "u_cm_per_s": inv.u_down,
                "p_erg_per_cm3": inv.p_down,
                "T_eV": inv.T_down,
                "E_r_erg_per_cc": A_EV * inv.T_down**4,
            },
        },
    }


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_table(profile: dict, path: str) -> None:
    """Save the profile as JSON for harness consumption."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(profile), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _relative_error(got: float, expected: float) -> float:
    return abs(got - expected) / max(abs(expected), 1.0e-300)


def self_test() -> bool:
    cfg = LEProxyConfig()
    profile = solve_le_proxy(cfg)
    inv = _invariants(cfg)
    checks: list[tuple[str, bool, str]] = []

    rho = profile["rho_g_per_cc"]
    u = profile["u_cm_per_s"]
    T = profile["T_mat_eV"]
    E = profile["E_r_erg_per_cc"]
    F = profile["F_r_erg_per_cm2_s"]

    checks.append(
        (
            "upstream pure-gas RH limit",
            _relative_error(float(rho[0]), inv.rho_up) < 1.0e-10
            and _relative_error(float(u[0]), inv.u_up) < 1.0e-10
            and _relative_error(float(T[0]), inv.T_up) < 1.0e-10,
            f"rho={rho[0]:.6e}, u={u[0]:.6e}, T={T[0]:.6e}",
        )
    )
    checks.append(
        (
            "downstream pure-gas RH limit",
            _relative_error(float(rho[-1]), inv.rho_down) < 1.0e-10
            and _relative_error(float(u[-1]), inv.u_down) < 1.0e-10
            and _relative_error(float(T[-1]), inv.T_down) < 1.0e-10,
            f"rho={rho[-1]:.6e}, u={u[-1]:.6e}, T={T[-1]:.6e}",
        )
    )
    checks.append(
        (
            "far flux limits",
            abs(float(F[0])) < 1.0e6 and abs(float(F[-1])) < 1.0e6,
            f"F_up={F[0]:.6e}, F_down={F[-1]:.6e}",
        )
    )
    checks.append(
        (
            "far radiation equilibrium limits",
            _relative_error(float(E[0]), A_EV * inv.T_up**4) < 1.0e-2
            and _relative_error(float(E[-1]), A_EV * inv.T_down**4) < 2.0,
            f"E_up={E[0]:.6e}, E_down={E[-1]:.6e}",
        )
    )

    passed = True
    for name, ok, detail in checks:
        passed = passed and ok
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {name}: {detail}; residual risk: NOT_LITERAL_LE proxy")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mach", type=float, default=3.0)
    parser.add_argument("--out", type=str, default="le_proxy_table.json")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--n-points", type=int, default=400)
    parser.add_argument("--sigma-a-per-cm", type=float, default=1.0)
    args = parser.parse_args()

    cfg = LEProxyConfig(
        mach_upstream=args.mach,
        n_points=args.n_points,
        sigma_a_per_cm=args.sigma_a_per_cm,
    )
    if args.self_test:
        return 0 if self_test() else 1
    profile = solve_le_proxy(cfg)
    write_table(profile, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
