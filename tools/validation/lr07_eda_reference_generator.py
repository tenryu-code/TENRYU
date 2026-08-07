#!/usr/bin/env python3
"""Generate frozen LR07-EDA grey radiative-shock reference tables.

The table solves the steady one-temperature equilibrium-diffusion approximation
specified by Lowrie and Rauenzahn's planar grey radiative-shock construction:
mass flux, momentum flux, gas energy flux plus radiative diffusion flux, and
T_rad = T_mat.  The default I1 tables intentionally use a low radiation-pressure
regime so TENRYU v1.0's omitted radiation force is a bounded perturbation.

Radiation-momentum-free reduction (TENRYU v1.0 compatibility)
----------------------------------------------------------------
TENRYU v1.0 omits radiation pressure force from the matter momentum equation
(`docs/NUMERICS.md:237-242`).  Literal Lowrie-Edwards radiation-momentum
shocks are therefore not TENRYU-v1.0 compatible, as documented in
`docs/validation/2d_rz/RH1/audit/lowrie_edwards_feasibility.md:15-33`.
The I1 reference uses the corresponding gas-momentum projection,

    J = rho u^2 + P_mat,

while retaining gas energy plus radiative diffusion flux,

    H = m (c_p T + u^2/2) + F_rad.

This is the no-radiation-pressure proxy used in the RH1 audit equations: no
radiation momentum or radiation enthalpy/advection term is introduced into the
shock invariant, and all quantities remain cgs with temperatures in eV.

Schema v2 flux convention
-------------------------
The table coordinate `x_cm` is monotone through the branch switch.  The primary
field `F_rad_erg_per_cm2_s` is conjugate to this monotone coordinate and is
directly usable in `-D * gradient(a_eV*T^4, x_cm)`.  The secondary field
`F_rad_energy_invariant_erg_per_cm2_s` stores the LR07 branch-oriented signed
flux used by gas-energy conservation diagnostics.  These fields are identical
on the upstream branch and sign-opposite on the downstream branch.
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


C_CGS = 2.99792458e10
A_EV = 1.3720e2
EV_TO_ERG = 1.6022e-12
PROTON_MASS_G = 1.6726219e-24
E_FLOOR = 1.0e-300


@dataclass(frozen=True)
class LR07Config:
    mach: float
    gamma: float = 5.0 / 3.0
    rho0_gcc: float = 2.0
    T0_eV: float = 30.0
    kappa_R_cm2_g: float = 0.5
    A_amu: float = 1.0
    zbar: float = 1.0
    n_points: int = 1601
    endpoint_clip_fraction: float = 1.0e-4
    include_radiation_pressure: bool = False


@dataclass(frozen=True)
class Invariants:
    R_gas: float
    cp: float
    rho0: float
    T0: float
    u0: float
    p0: float
    prad0: float
    mass_flux: float
    momentum_flux: float
    energy_flux: float
    T_sonic: float
    T_down: float


def safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def linspace(a: float, b: float, n: int) -> list[float]:
    if n < 2:
        return [float(a)]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def interp1(x: list[float], y: list[float], xq: float) -> float:
    if xq <= x[0]:
        return y[0]
    if xq >= x[-1]:
        return y[-1]
    lo = 0
    hi = len(x) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] <= xq:
            lo = mid
        else:
            hi = mid
    w = (xq - x[lo]) / (x[hi] - x[lo])
    return y[lo] + w * (y[hi] - y[lo])


def gas_constant(cfg: LR07Config) -> float:
    return (1.0 + cfg.zbar) * EV_TO_ERG / (cfg.A_amu * PROTON_MASS_G)


def validate_config(cfg: LR07Config) -> None:
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


def prad(T_eV: float) -> float:
    return A_EV * T_eV**4 / 3.0


def pmat(rho: float, T_eV: float, R_gas: float) -> float:
    return rho * R_gas * T_eV


def upstream_invariants(cfg: LR07Config) -> tuple[float, float, float, float, float, float, float]:
    R = gas_constant(cfg)
    rho0 = cfg.rho0_gcc
    T0 = cfg.T0_eV
    p0 = pmat(rho0, T0, R)
    sound0 = math.sqrt(cfg.gamma * p0 / rho0)
    u0 = cfg.mach * sound0
    m = rho0 * u0
    prad0 = prad(T0)
    J = m * u0 + p0 + (prad0 if cfg.include_radiation_pressure else 0.0)
    cp = cfg.gamma / (cfg.gamma - 1.0) * R
    H = m * (cp * T0 + 0.5 * u0 * u0)
    return R, cp, rho0, T0, u0, p0, prad0, m, J, H


def make_discriminant(cfg: LR07Config, m: float, J: float, R: float):
    def discriminant(T_eV: float) -> float:
        rad_p = prad(T_eV) if cfg.include_radiation_pressure else 0.0
        b = J - rad_p
        return b * b - 4.0 * m * m * R * T_eV

    return discriminant


def velocity_from_T(T_eV: float, branch: str, cfg: LR07Config, inv: Invariants) -> float:
    rad_p = prad(T_eV) if cfg.include_radiation_pressure else 0.0
    b = inv.momentum_flux - rad_p
    disc = b * b - 4.0 * inv.mass_flux * inv.mass_flux * inv.R_gas * T_eV
    if disc < -1.0e-12 * b * b:
        raise FloatingPointError("momentum quadratic has no real root")
    root = math.sqrt(max(disc, 0.0))
    if branch == "upstream":
        return (b + root) / (2.0 * inv.mass_flux)
    if branch == "downstream":
        return (b - root) / (2.0 * inv.mass_flux)
    raise ValueError(f"unknown branch {branch!r}")


def gas_energy_flux(T_eV: float, branch: str, cfg: LR07Config, inv: Invariants) -> float:
    u = velocity_from_T(T_eV, branch, cfg, inv)
    return inv.mass_flux * (inv.cp * T_eV + 0.5 * u * u)


def radiation_flux(T_eV: float, branch: str, cfg: LR07Config, inv: Invariants) -> float:
    return inv.energy_flux - gas_energy_flux(T_eV, branch, cfg, inv)


def solve_sonic_temperature(cfg: LR07Config, m: float, J: float, R: float) -> float:
    disc = make_discriminant(cfg, m, J, R)
    lo = cfg.T0_eV
    hi = max(cfg.T0_eV * 1.25, cfg.T0_eV + 1.0)
    while disc(hi) > 0.0:
        hi *= 1.25
        if hi > 1.0e7 * cfg.T0_eV:
            raise RuntimeError("failed to bracket sonic temperature")
    for _ in range(160):
        mid = 0.5 * (lo + hi)
        if disc(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_downstream_temperature(cfg: LR07Config, inv: Invariants) -> float:
    lo = cfg.T0_eV
    hi = inv.T_sonic * (1.0 - 1.0e-12)

    def residual(T_eV: float) -> float:
        return gas_energy_flux(T_eV, "downstream", cfg, inv) - inv.energy_flux

    if residual(lo) >= 0.0 or residual(hi) <= 0.0:
        raise RuntimeError("failed to bracket downstream equilibrium temperature")
    for _ in range(160):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def build_invariants(cfg: LR07Config) -> Invariants:
    validate_config(cfg)
    R, cp, rho0, T0, u0, p0, prad0, m, J, H = upstream_invariants(cfg)
    T_sonic = solve_sonic_temperature(cfg, m, J, R)
    provisional = Invariants(R, cp, rho0, T0, u0, p0, prad0, m, J, H, T_sonic, T0)
    T_down = solve_downstream_temperature(cfg, provisional)
    return Invariants(R, cp, rho0, T0, u0, p0, prad0, m, J, H, T_sonic, T_down)


def dx_dT(T_eV: float, branch: str, cfg: LR07Config, inv: Invariants) -> float:
    u = velocity_from_T(T_eV, branch, cfg, inv)
    rho = inv.mass_flux / u
    F = radiation_flux(T_eV, branch, cfg, inv)
    if abs(F) < E_FLOOR:
        F = math.copysign(E_FLOOR, F if F != 0.0 else 1.0)
    return -4.0 * A_EV * C_CGS * T_eV**3 / (3.0 * cfg.kappa_R_cm2_g * rho * F)


def state_from_T(T_eV: float, branch: str, cfg: LR07Config, inv: Invariants) -> dict[str, float]:
    u = velocity_from_T(T_eV, branch, cfg, inv)
    rho = inv.mass_flux / u
    p_gas = pmat(rho, T_eV, inv.R_gas)
    p_rad = prad(T_eV)
    F = radiation_flux(T_eV, branch, cfg, inv)
    return {
        "rho": rho,
        "u": u,
        "T": T_eV,
        "P_mat": p_gas,
        "P_rad": p_rad,
        "F_rad": F,
    }


def monotone_coordinate_flux(energy_invariant_flux: float, branch: str) -> float:
    if branch == "upstream":
        return energy_invariant_flux
    if branch == "downstream":
        return -energy_invariant_flux
    raise ValueError(f"unknown branch {branch!r}")


def integrate_raw_profile(cfg: LR07Config, inv: Invariants) -> dict[str, list[float]]:
    n_each = max(2000, cfg.n_points * 3)
    eps = cfg.endpoint_clip_fraction
    T0 = inv.T0
    Ts = inv.T_sonic
    Td = inv.T_down
    T_up0 = T0 + eps * (Ts - T0)
    T_up1 = Ts - eps * (Ts - T0)
    T_dn0 = Ts - eps * (Ts - Td)
    T_dn1 = Td + eps * (Ts - Td)

    T_up = linspace(T_up0, T_up1, n_each)
    x_up_cum = [0.0]
    for a, b in zip(T_up[:-1], T_up[1:]):
        dx = 0.5 * (dx_dT(a, "upstream", cfg, inv) + dx_dT(b, "upstream", cfg, inv)) * (b - a)
        x_up_cum.append(x_up_cum[-1] + dx)
    x_up = [x - x_up_cum[-1] for x in x_up_cum]

    T_dn = linspace(T_dn0, T_dn1, n_each)
    x_dn = [0.0]
    for a, b in zip(T_dn[:-1], T_dn[1:]):
        dx = 0.5 * (dx_dT(a, "downstream", cfg, inv) + dx_dT(b, "downstream", cfg, inv)) * (b - a)
        x_dn.append(x_dn[-1] + abs(dx))

    raw: dict[str, list[float]] = {
        "x": [],
        "rho": [],
        "u": [],
        "T": [],
        "P_mat": [],
        "P_rad": [],
        "F_rad": [],
        "branch": [],
    }
    for x, T in zip(x_up, T_up):
        state = state_from_T(T, "upstream", cfg, inv)
        raw["x"].append(x)
        for key in ("rho", "u", "T", "P_mat", "P_rad", "F_rad"):
            raw[key].append(state[key])
        raw["branch"].append("upstream")
    for x, T in zip(x_dn[1:], T_dn[1:]):
        state = state_from_T(T, "downstream", cfg, inv)
        raw["x"].append(x)
        for key in ("rho", "u", "T", "P_mat", "P_rad", "F_rad"):
            raw[key].append(state[key])
        raw["branch"].append("downstream")
    return raw


def resample_uniform_x(
    raw: dict[str, list[float]], n_points: int, cfg: LR07Config, inv: Invariants
) -> dict[str, list[Any]]:
    # The frozen checker differentiates the tabulated fields directly.  Uniform
    # x undersamples the exponential far tails, so schema v2 samples uniformly
    # in branch temperature while preserving a monotone x coordinate.
    n_up = n_points // 2
    n_dn = n_points - n_up + 1
    raw_branch = raw["branch"]
    up_pairs = [
        (float(x), float(T))
        for x, T, branch in zip(raw["x"], raw["T"], raw_branch)
        if branch == "upstream"
    ]
    dn_pairs = [
        (float(x), float(T))
        for x, T, branch in zip(raw["x"], raw["T"], raw_branch)
        if branch == "downstream"
    ]
    if len(up_pairs) < 2 or len(dn_pairs) < 2:
        raise RuntimeError("raw LR07 profile did not contain both branches")
    up_x = [x for x, _ in up_pairs]
    up_T = [T for _, T in up_pairs]
    dn_x = [x for x, _ in dn_pairs]
    dn_T = [T for _, T in dn_pairs]

    rows: list[tuple[float, float, str]] = []
    for T in linspace(up_T[0], up_T[-1], n_up):
        rows.append((interp1(up_T, up_x, T), T, "upstream"))
    dn_T_rev = list(reversed(dn_T))
    dn_x_rev = list(reversed(dn_x))
    for T in linspace(dn_T[0], dn_T[-1], n_dn)[1:]:
        rows.append((interp1(dn_T_rev, dn_x_rev, T), T, "downstream"))

    out: dict[str, list[Any]] = {"x_cm": []}
    out["rho_g_per_cc"] = []
    out["u_cm_per_s"] = []
    out["T_eV"] = []
    out["P_mat_dyne_per_cm2"] = []
    out["P_rad_dyne_per_cm2"] = []
    out["F_rad_erg_per_cm2_s"] = []
    out["F_rad_energy_invariant_erg_per_cm2_s"] = []
    branches: list[str] = []
    for x, T, branch in rows:
        state = state_from_T(T, branch, cfg, inv)
        out["x_cm"].append(x)
        out["rho_g_per_cc"].append(state["rho"])
        out["u_cm_per_s"].append(state["u"])
        out["T_eV"].append(state["T"])
        out["P_mat_dyne_per_cm2"].append(state["P_mat"])
        out["P_rad_dyne_per_cm2"].append(state["P_rad"])
        out["F_rad_erg_per_cm2_s"].append(monotone_coordinate_flux(state["F_rad"], branch))
        out["F_rad_energy_invariant_erg_per_cm2_s"].append(state["F_rad"])
        branches.append(branch)
    out["branch"] = branches
    return out


def max_gradient_ratio(x: list[float], p_mat: list[float], p_rad: list[float]) -> float:
    if len(x) < 3:
        return math.inf
    grad_mat: list[float] = []
    grad_rad: list[float] = []
    for i in range(len(x)):
        if i == 0:
            dx = x[1] - x[0]
            gm = (p_mat[1] - p_mat[0]) / dx
            gr = (p_rad[1] - p_rad[0]) / dx
        elif i + 1 == len(x):
            dx = x[-1] - x[-2]
            gm = (p_mat[-1] - p_mat[-2]) / dx
            gr = (p_rad[-1] - p_rad[-2]) / dx
        else:
            dx = x[i + 1] - x[i - 1]
            gm = (p_mat[i + 1] - p_mat[i - 1]) / dx
            gr = (p_rad[i + 1] - p_rad[i - 1]) / dx
        grad_mat.append(gm)
        grad_rad.append(gr)
    scale = max(abs(v) for v in grad_mat)
    threshold = max(1.0e-12 * scale, E_FLOOR)
    ratios = [abs(gr) / max(abs(gm), E_FLOOR) for gm, gr in zip(grad_mat, grad_rad) if abs(gm) >= threshold]
    return max(ratios) if ratios else math.inf


def conservation_diagnostics(table: dict[str, list[Any]], cfg: LR07Config, inv: Invariants) -> dict[str, float]:
    rho = table["rho_g_per_cc"]
    u = table["u_cm_per_s"]
    T = table["T_eV"]
    p = table["P_mat_dyne_per_cm2"]
    pr = table["P_rad_dyne_per_cm2"]
    F = table.get("F_rad_energy_invariant_erg_per_cm2_s", table["F_rad_erg_per_cm2_s"])
    mass_res = 0.0
    mom_res = 0.0
    energy_res = 0.0
    prad_ratio = 0.0
    for rho_i, u_i, T_i, p_i, pr_i, F_i in zip(rho, u, T, p, pr, F):
        mass_res = max(mass_res, abs(rho_i * u_i - inv.mass_flux) / max(abs(inv.mass_flux), E_FLOOR))
        mom = rho_i * u_i * u_i + p_i + (pr_i if cfg.include_radiation_pressure else 0.0)
        mom_res = max(mom_res, abs(mom - inv.momentum_flux) / max(abs(inv.momentum_flux), E_FLOOR))
        energy = inv.mass_flux * (inv.cp * T_i + 0.5 * u_i * u_i) + F_i
        energy_res = max(energy_res, abs(energy - inv.energy_flux) / max(abs(inv.energy_flux), E_FLOOR))
        prad_ratio = max(prad_ratio, abs(pr_i) / max(abs(p_i), E_FLOOR))
    grad_ratio = max_gradient_ratio(table["x_cm"], p, pr)
    return {
        "mass_flux_residual_rel_max": mass_res,
        "momentum_flux_residual_rel_max": mom_res,
        "energy_flux_residual_rel_max": energy_res,
        "conservation_residual_rel_max": max(mass_res, mom_res, energy_res),
        "max_P_rad_over_P_mat": prad_ratio,
        "max_abs_grad_P_rad_over_abs_grad_P_mat": grad_ratio,
    }


def git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def build_reference(cfg: LR07Config) -> dict[str, Any]:
    inv = build_invariants(cfg)
    raw = integrate_raw_profile(cfg, inv)
    table = resample_uniform_x(raw, cfg.n_points, cfg, inv)
    diagnostics = conservation_diagnostics(table, cfg, inv)
    upstream = state_from_T(inv.T0, "upstream", cfg, inv)
    downstream = state_from_T(inv.T_down, "downstream", cfg, inv)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "tools/validation/lr07_eda_reference_generator.py",
            "git_commit": git_commit(),
            "references": [
                {
                    "label": "Lowrie and Rauenzahn 2007",
                    "title": "Radiative shock solutions in the equilibrium diffusion limit",
                    "doi": "10.1007/s00193-007-0081-2",
                },
                {
                    "label": "Ferguson, Morel, and Lowrie 2017",
                    "title": "The equilibrium-diffusion limit for radiation hydrodynamics",
                    "doi": "10.1016/j.jqsrt.2017.07.031",
                    "osti_id": "1374348",
                },
            ],
            "model": "steady planar grey equilibrium-diffusion, T_rad=T_mat",
            "tenryu_v1_caveat": (
                "TENRYU v1.0 omits radiation force in matter momentum; schema v2 uses "
                "gas-only momentum by default and I1 gates max P_rad/P_mat and "
                "grad(P_rad)/grad(P_mat) <= 1e-3."
            ),
            "schema_compatibility": (
                "schema_version=2 adds F_rad_energy_invariant_erg_per_cm2_s. "
                "F_rad_erg_per_cm2_s is the flux conjugate to monotone x_cm; "
                "legacy schema v1 stored the branch-oriented energy-invariant flux."
            ),
        },
        "parameters": {
            "M0": cfg.mach,
            "gamma": cfg.gamma,
            "rho0_g_per_cc": cfg.rho0_gcc,
            "T0_eV": cfg.T0_eV,
            "kappa_R_cm2_per_g": cfg.kappa_R_cm2_g,
            "A_amu": cfg.A_amu,
            "zbar": cfg.zbar,
            "n_points": cfg.n_points,
            "endpoint_clip_fraction": cfg.endpoint_clip_fraction,
            "include_radiation_pressure": cfg.include_radiation_pressure,
            "a_eV_erg_cm3_eV4": A_EV,
            "c_cm_per_s": C_CGS,
            "eV_to_erg": EV_TO_ERG,
            "proton_mass_g": PROTON_MASS_G,
        },
        "invariants": {
            "R_gas_erg_per_g_eV": inv.R_gas,
            "cp_erg_per_g_eV": inv.cp,
            "mass_flux_g_per_cm2_s": inv.mass_flux,
            "momentum_flux_dyne_per_cm2": inv.momentum_flux,
            "energy_flux_erg_per_cm2_s": inv.energy_flux,
            "T_sonic_eV": inv.T_sonic,
            "T_downstream_eV": inv.T_down,
        },
        "states": {
            "upstream": upstream,
            "downstream": downstream,
        },
        "diagnostics": diagnostics,
        "table": table,
    }
    return payload


def parse_mach_list(value: str) -> list[float]:
    out = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("mach list is empty")
    return out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="tests/verification/data/lr07_eda_reference")
    parser.add_argument("--mach-list", type=parse_mach_list, default=parse_mach_list("1.5,2.0,3.0"))
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--rho0-gcc", type=float, default=2.0)
    parser.add_argument("--T0-eV", type=float, default=30.0)
    parser.add_argument("--kappa-R-cm2-g", type=float, default=0.5)
    parser.add_argument("--A-amu", type=float, default=1.0)
    parser.add_argument("--zbar", type=float, default=1.0)
    parser.add_argument("--n-points", type=int, default=1601)
    parser.add_argument("--endpoint-clip-fraction", type=float, default=1.0e-4)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--include-radiation-pressure",
        action="store_true",
        help="legacy diagnostic mode: include P_rad in the momentum invariant",
    )
    group.add_argument(
        "--omit-radiation-pressure",
        action="store_true",
        help="legacy compatibility flag; radiation pressure is omitted by default",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    rows: list[dict[str, Any]] = []
    for mach in args.mach_list:
        cfg = LR07Config(
            mach=mach,
            gamma=args.gamma,
            rho0_gcc=args.rho0_gcc,
            T0_eV=args.T0_eV,
            kappa_R_cm2_g=args.kappa_R_cm2_g,
            A_amu=args.A_amu,
            zbar=args.zbar,
            n_points=args.n_points,
            endpoint_clip_fraction=args.endpoint_clip_fraction,
            include_radiation_pressure=args.include_radiation_pressure,
        )
        payload = build_reference(cfg)
        filename = f"lr07_eda_M{safe_float_token(mach)}.json"
        write_json(out_dir / filename, payload)
        rows.append(
            {
                "file": filename,
                "M0": mach,
                "n_points": cfg.n_points,
                "max_P_rad_over_P_mat": payload["diagnostics"]["max_P_rad_over_P_mat"],
                "max_abs_grad_P_rad_over_abs_grad_P_mat": payload["diagnostics"][
                    "max_abs_grad_P_rad_over_abs_grad_P_mat"
                ],
                "conservation_residual_rel_max": payload["diagnostics"]["conservation_residual_rel_max"],
            }
        )
        print(
            f"{filename}: max_P_rad/P_mat={rows[-1]['max_P_rad_over_P_mat']:.6e} "
            f"max_grad_ratio={rows[-1]['max_abs_grad_P_rad_over_abs_grad_P_mat']:.6e} "
            f"conservation={rows[-1]['conservation_residual_rel_max']:.6e}"
        )
    manifest = {
        "schema_version": "tenryu.lr07_eda_reference_manifest.v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/validation/lr07_eda_reference_generator.py",
        "git_commit": git_commit(),
        "rows": rows,
    }
    write_json(out_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
