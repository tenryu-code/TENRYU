#!/usr/bin/env python3
"""Generate in-house 2T (electron/ion) constant-Eddington FLD nED grey-shock
references with finite Spitzer Q_ei coupling.

The model uses gas-only mass/momentum invariants, gas enthalpy from both
species plus comoving FLD flux as the total-energy invariant, radiation
coupling to electrons only, AV/shock heating to ions (kernel
av_heat_to='ion'), an electron-adiabat subshock, and no O(v/c) terms.

Citations: docs/design/i5_2t_refgen_spec.md; src/materials/eos_device.cuh:111-181;
src/hydro/hydro_2d.cu:2104-2127; src/radiation/fld_2d_rz_gpu.cu:3664-3670;
NUMERICS section 1.1.3/1.1.4.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.linalg import eig
from scipy.integrate import solve_ivp
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import brentq, least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fld_const_eddington_reference_generator as one_t


C = one_t.C_CGS
A_EV = one_t.A_EV
EV = one_t.EV_TO_ERG
MP = one_t.PROTON_MASS_G
E_FLOOR = one_t.E_FLOOR

QEI_TAU_COEFF = 3.16e8
LNLAM_FLOOR = 2.0
TE_POW_CAP_EV = 1.0e8
LOW_PRAD_HARD_GATE = 1.0e-3
IDENTITY_MEDIAN_GATE = 1.0e-3
IDENTITY_P95_GATE = 1.0e-2
INVARIANT_RESIDUAL_GATE = 1.0e-12
JUMP_RESIDUAL_GATE = 1.0e-12
FAR_EQUILIBRATION_GATE = 1.0e-10
MATCH_RESIDUAL_GATE = 1.0e-9
SV2_MIN_ORDER = 0.8
SCHEMA_VERSION = "fld_ced_2t.v1"
PROBLEM = "FLD_const_Eddington_2T"
REFERENCE_MODEL = "FLD_const_Eddington_2T_nED_low_Prad_finite_Qei"
QEI_MODEL = "spitzer_nrl_lnlam_floor2"
REDUCED_SOLVER_EPS_MAX = 1.0e-6
REDUCED_ENDPOINT_CLIP_FRACTION = 1.0e-6
FULL_DOWNSTREAM_GROW_EFOLDS = 25.0
FULL_DOWNSTREAM_SLOW_EFOLDS = 4.0
FULL_DOWNSTREAM_MAX_LAMBDA_H = 2.0
FULL_DOWNSTREAM_MIN_INTERVALS = 4
FULL_DOWNSTREAM_MAX_INTERVALS = 192


@dataclass(frozen=True)
class FLDCED2TConfig:
    mach: float
    gamma: float = 5.0 / 3.0
    rho0_gcc: float = 2.0
    T0_eV: float = 30.0
    kappa_R_cm2_g: float = 0.5
    A_amu: float = 1.0
    zbar: float = 1.0
    qei_multiplier: float = 1.0
    mode: str = "2t"
    solver: str = "auto"
    regime_tag: str = "R1"
    n_points: int = 1601
    seed_amplitude_rel: float = 1.0e-6
    shock_clip_fraction: float = 1.0e-4
    target_domain_span_cm: float | None = None
    padding_points_per_side: int = 8
    ode_rtol: float = 1.0e-10
    ode_atol_T_eV: float = 1.0e-8
    ode_atol_W: float = 1.0e-8
    match_xtol: float = 1.0e-12
    continuation_ladder: tuple[float, ...] = (
        1.0e6,
        3.0e5,
        1.0e5,
        3.0e4,
        1.0e4,
        3.0e3,
        1.0e3,
        3.0e2,
        1.0e2,
        3.0e1,
        1.0e1,
        3.0,
        1.0,
    )


@dataclass(frozen=True)
class Invariants2T:
    base: Any
    R_i: float
    R_e: float
    cv_i: float
    cv_e: float
    cp_i: float
    cp_e: float


@dataclass(frozen=True)
class WarmStart:
    one_cfg: Any
    match: Any
    up_len_1t: float
    dn_len_1t: float
    T_up_shock_1t: float
    T_dn_shock_1t: float
    rho_ratio_1t: float


@dataclass(frozen=True)
class EigenSeeds:
    upstream_eigenvalues: list[float]
    downstream_eigenvalues: list[float]
    upstream_vector: np.ndarray
    upstream_fast_vector: np.ndarray
    upstream_left_fast_vector: np.ndarray
    downstream_slow_vector: np.ndarray
    downstream_fast_vector: np.ndarray
    downstream_left_fast_vector: np.ndarray
    downstream_left_grow_vector: np.ndarray
    downstream_lambda_slow: float
    downstream_lambda_fast: float
    downstream_lambda_grow: float
    upstream_lambda_grow: float
    upstream_slope_warning: dict[str, Any]


@dataclass(frozen=True)
class ShockSolution2T:
    cfg: FLDCED2TConfig
    inv2: Invariants2T
    warm: WarmStart
    seeds: EigenSeeds
    ell_up: float
    up_sol: Any
    bvp_sol: Any
    L_dn_span: float
    y_minus: np.ndarray
    y_plus: np.ndarray
    jump_diagnostics: dict[str, float]
    bc_residuals: list[float]
    continuation_diagnostics: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class DownstreamShootingSolution:
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    status: int
    message: str
    L_dn_span: float
    breakpoints_cm: np.ndarray
    segments: tuple[Any, ...]
    tenryu_theta_margin: float
    tenryu_newton_iterations: int
    tenryu_final_residual_inf: float
    tenryu_final_absolute_residual_inf: float
    tenryu_residual_breakdown: dict[str, Any]
    tenryu_projection_diagnostics: dict[str, Any]
    tenryu_converged_with_warning: bool = False
    tenryu_warning: str | None = None
    tenryu_method: str = "damped_newton_multiple_shooting"
    tenryu_sol_input_physical: bool = False
    nfev: int = 0
    njev: int = 0
    nlu: int = 0

    def sol(self, t: Any) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        scalar = t_arr.ndim == 0
        t_flat = t_arr.reshape(-1)
        out = np.empty((3, t_flat.size), dtype=float)
        for idx, value in enumerate(t_flat):
            if self.tenryu_sol_input_physical:
                s = min(max(float(value), 0.0), self.L_dn_span)
            else:
                s = min(max(float(value) * self.L_dn_span, 0.0), self.L_dn_span)
            if s <= self.breakpoints_cm[0]:
                out[:, idx] = self.y[:, 0]
                continue
            if s >= self.breakpoints_cm[-1]:
                out[:, idx] = self.y[:, -1]
                continue
            seg_idx = int(np.searchsorted(self.breakpoints_cm, s, side="right") - 1)
            seg_idx = min(max(seg_idx, 0), len(self.segments) - 1)
            out[:, idx] = np.asarray(self.segments[seg_idx].sol(s), dtype=float)
        if scalar:
            return out[:, 0]
        return out.reshape((3, *t_arr.shape))


@dataclass(frozen=True)
class DownstreamIntervalSolution:
    raw: Any
    s0: float
    s1: float
    scale: np.ndarray

    @property
    def y(self) -> np.ndarray:
        return self.scale[:, None] * self.raw.y

    @property
    def status(self) -> int:
        return int(self.raw.status)

    @property
    def success(self) -> bool:
        return bool(self.raw.success)

    @property
    def message(self) -> str:
        return str(self.raw.message)

    def sol(self, s: Any) -> np.ndarray:
        s_arr = np.asarray(s, dtype=float)
        u = (s_arr - self.s0) / (self.s1 - self.s0)
        return self.scale[:, None] * self.raw.sol(u) if np.asarray(u).ndim > 0 else self.scale * self.raw.sol(u)


@dataclass(frozen=True)
class ReducedBranchSolution:
    branch: str
    F_shock: float
    T_far: float
    T_start: float
    T_shock: float
    Z_shock: float
    E_shock: float
    length_cm: float
    solution: Any
    Z_abs_atol: float


@dataclass(frozen=True)
class ReducedShockSolution:
    cfg: FLDCED2TConfig
    inv2: Invariants2T
    warm: WarmStart
    ell_up: float
    up_branch: ReducedBranchSolution
    dn_branch: ReducedBranchSolution
    L_dn_span: float
    y_minus: np.ndarray
    y_plus: np.ndarray
    y_plus_raw: np.ndarray
    jump_diagnostics: dict[str, float]
    bc_residuals: list[float]
    reduced_diagnostics: dict[str, Any]
    continuation_diagnostics: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class FullEigenSeeds:
    upstream_eigenvalues: list[float]
    downstream_eigenvalues: list[float]
    upstream_vector: np.ndarray
    downstream_left_grow_vector: np.ndarray
    downstream_lambda_slow: float
    downstream_lambda_fast: float
    downstream_lambda_grow: float
    upstream_lambda_grow: float


@dataclass(frozen=True)
class FullBranchSolution:
    branch: str
    F_shock: float
    T_far: float
    T_start: float
    T_shock: float
    length_cm: float
    solution: Any


@dataclass(frozen=True)
class FullShockSolution2T:
    cfg: FLDCED2TConfig
    inv2: Invariants2T
    warm: WarmStart
    seeds: FullEigenSeeds
    ell_up: float
    up_branch: FullBranchSolution
    dn_branch: FullBranchSolution
    L_dn_span: float
    y_minus: np.ndarray
    y_plus: np.ndarray
    jump_diagnostics: dict[str, float]
    bc_residuals: list[float]
    full_diagnostics: dict[str, Any]
    continuation_diagnostics: tuple[dict[str, Any], ...] = ()


class FullDownstreamClassifiedFailure(RuntimeError):
    def __init__(self, classification: str, residual_sign: float, message: str) -> None:
        super().__init__(message)
        self.classification = classification
        self.residual_sign = residual_sign


def validate_config(cfg: FLDCED2TConfig) -> None:
    if cfg.mach <= 1.0:
        raise ValueError("mach must be supersonic")
    if cfg.gamma <= 1.0:
        raise ValueError("gamma must exceed 1")
    if cfg.rho0_gcc <= 0.0 or cfg.T0_eV <= 0.0 or cfg.kappa_R_cm2_g <= 0.0:
        raise ValueError("rho0, T0, and kappa_R must be positive")
    if cfg.A_amu <= 0.0 or cfg.zbar <= 0.0:
        raise ValueError("A_amu and zbar must be positive")
    if cfg.n_points < 1000:
        raise ValueError("n_points must be >= 1000")
    if cfg.mode not in {"2t", "equal_t"}:
        raise ValueError("mode must be '2t' or 'equal_t'")
    if cfg.solver not in {"auto", "reduced", "full"}:
        raise ValueError("solver must be 'auto', 'reduced', or 'full'")
    if cfg.qei_multiplier <= 0.0:
        raise ValueError("qei_multiplier must be positive")
    if not 0.0 < cfg.seed_amplitude_rel < 1.0e-3:
        raise ValueError("seed_amplitude_rel must be in (0, 1e-3)")
    if not 0.0 <= cfg.shock_clip_fraction < 1.0e-2:
        raise ValueError("shock_clip_fraction must be in [0, 1e-2)")
    if cfg.target_domain_span_cm is not None and cfg.target_domain_span_cm <= 0.0:
        raise ValueError("target_domain_span_cm must be positive when set")
    if cfg.padding_points_per_side < 1:
        raise ValueError("padding_points_per_side must be positive")
    if cfg.ode_rtol <= 0.0 or cfg.ode_atol_T_eV <= 0.0 or cfg.ode_atol_W <= 0.0:
        raise ValueError("ODE tolerances must be positive")
    if cfg.match_xtol <= 0.0:
        raise ValueError("match_xtol must be positive")


def one_t_config_from_2t(
    cfg: FLDCED2TConfig,
    *,
    target_domain_span_cm: float | None = None,
    shooting_rtol: float = 1.0e-10,
) -> Any:
    span = target_domain_span_cm
    if span is None:
        span = one_t.TARGET_DOMAIN_SPAN_CM
    return one_t.FLDCEDConfig(
        mach=cfg.mach,
        gamma=cfg.gamma,
        rho0_gcc=cfg.rho0_gcc,
        T0_eV=cfg.T0_eV,
        kappa_R_cm2_g=cfg.kappa_R_cm2_g,
        A_amu=cfg.A_amu,
        zbar=cfg.zbar,
        n_points=cfg.n_points,
        endpoint_clip_fraction=1.0e-6,
        shock_clip_fraction=cfg.shock_clip_fraction,
        target_domain_span_cm=float(span),
        padding_points_per_side=cfg.padding_points_per_side,
        ode_rtol=cfg.ode_rtol,
        shooting_rtol=shooting_rtol,
    )


def build_invariants_2t(cfg: FLDCED2TConfig) -> Invariants2T:
    validate_config(cfg)
    one_cfg = one_t.FLDCEDConfig(
        mach=cfg.mach,
        gamma=cfg.gamma,
        rho0_gcc=cfg.rho0_gcc,
        T0_eV=cfg.T0_eV,
        kappa_R_cm2_g=cfg.kappa_R_cm2_g,
        A_amu=cfg.A_amu,
        zbar=cfg.zbar,
    )
    base = one_t.build_invariants(one_cfg)
    R_i = EV / (cfg.A_amu * MP)
    R_e = cfg.zbar * EV / (cfg.A_amu * MP)
    gm1 = cfg.gamma - 1.0
    inv2 = Invariants2T(
        base=base,
        R_i=R_i,
        R_e=R_e,
        cv_i=R_i / gm1,
        cv_e=R_e / gm1,
        cp_i=cfg.gamma * R_i / gm1,
        cp_e=cfg.gamma * R_e / gm1,
    )
    if abs(R_i + R_e - base.R_gas) > 1.0e-13 * base.R_gas:
        raise RuntimeError("2T gas constants do not contract to the 1T gas constant")
    return inv2


def velocity_from_theta(theta: float, branch: str, inv2: Invariants2T) -> float:
    return one_t.velocity_from_T(theta / inv2.base.R_gas, branch, inv2.base)


def gas_energy_flux_2t(T_i: float, T_e: float, u: float, inv2: Invariants2T) -> float:
    return inv2.base.mass_flux * (inv2.cp_i * T_i + inv2.cp_e * T_e + 0.5 * u * u)


def radiation_flux_2t(T_i: float, T_e: float, u: float, inv2: Invariants2T) -> float:
    return inv2.base.energy_flux - gas_energy_flux_2t(T_i, T_e, u, inv2)


def _tau_eq(rho: float, T_e: float, cfg: FLDCED2TConfig) -> float:
    te = min(T_e, TE_POW_CAP_EV)
    if rho <= 0.0 or te <= 0.0 or not math.isfinite(rho) or not math.isfinite(te):
        raise ValueError("unphysical rho or Te in Qei tau_eq")
    n_i = rho / (cfg.A_amu * MP)
    n_e = cfg.zbar * n_i
    if n_i <= 0.0 or n_e <= 0.0:
        raise ValueError("unphysical ion/electron density in Qei tau_eq")
    if te >= 10.0 * cfg.zbar**2:
        lnlam_raw = 24.0 - 0.5 * math.log(n_e) + math.log(te)
    else:
        lnlam_raw = 23.0 - 0.5 * math.log(n_e) - math.log(cfg.zbar) + 1.5 * math.log(te)
    lnlam = max(LNLAM_FLOOR, lnlam_raw)
    tau_eq = QEI_TAU_COEFF * cfg.A_amu * te**1.5 / (cfg.zbar**2 * n_i * lnlam)
    if tau_eq <= 0.0 or not math.isfinite(tau_eq):
        raise ValueError("unphysical Qei tau_eq")
    return tau_eq


def qei_rate_per_gram(rho: float, T_e: float, T_i: float, cfg: FLDCED2TConfig, inv2: Invariants2T) -> float:
    if not math.isfinite(T_i):
        raise ValueError("unphysical Ti in Qei")
    te = min(T_e, TE_POW_CAP_EV)
    tau_eq = _tau_eq(rho, T_e, cfg)
    return cfg.qei_multiplier * inv2.cv_e * (te - T_i) / tau_eq


def tau_scales(cfg: FLDCED2TConfig, inv2: Invariants2T) -> dict[str, float]:
    T1 = inv2.base.T_down
    u1 = velocity_from_theta(inv2.base.R_gas * T1, "downstream", inv2)
    rho1 = inv2.base.mass_flux / u1
    tau_eq = _tau_eq(rho1, T1, cfg)
    tau_ei = (tau_eq / cfg.qei_multiplier) * inv2.cv_i / (inv2.cv_e + inv2.cv_i)
    tau_rad = inv2.cv_e / (4.0 * C * cfg.kappa_R_cm2_g * A_EV * T1**3)
    return {
        "tau_ei_s": tau_ei,
        "tau_rad_s": tau_rad,
        "tau_ratio": tau_ei / tau_rad,
        "l_ei_cm": u1 * tau_ei,
        "l_rad_cm": u1 * tau_rad,
        "rho1_gcc": rho1,
        "T1_eV": T1,
        "u1_cm_per_s": u1,
    }


def fixed_point(branch: str, inv2: Invariants2T) -> np.ndarray:
    T_far = inv2.base.T0 if branch == "upstream" else inv2.base.T_down
    return np.asarray([T_far, T_far, 0.0], dtype=float)


def rhs_2t(x: float, y: Any, branch: str, cfg: FLDCED2TConfig, inv2: Invariants2T) -> list[float]:
    del x
    T_i = float(y[0])
    T_e = float(y[1])
    W_e = float(y[2])
    T_far = inv2.base.T0 if branch == "upstream" else inv2.base.T_down
    if (
        abs(T_i - T_far) <= 1.0e-13 * max(abs(T_far), 1.0)
        and abs(T_e - T_far) <= 1.0e-13 * max(abs(T_far), 1.0)
        and abs(W_e) <= 1.0e-13 * max(A_EV * T_far**4, 1.0)
    ):
        return [0.0, 0.0, 0.0]
    theta = inv2.R_i * T_i + inv2.R_e * T_e
    u = velocity_from_theta(theta, branch, inv2)
    J = inv2.base.mass_flux
    rho = J / u
    p_i = rho * inv2.R_i * T_i
    p_e = rho * inv2.R_e * T_e
    denom = 2.0 * J * u - inv2.base.momentum_flux
    q = qei_rate_per_gram(rho, T_e, T_i, cfg, inv2)
    kap = cfg.kappa_R_cm2_g

    a11 = J * inv2.cv_i - p_i * J * inv2.R_i / denom
    a12 = -p_i * J * inv2.R_e / denom
    a21 = -p_e * J * inv2.R_i / denom
    a22 = J * inv2.cv_e - p_e * J * inv2.R_e / denom
    b1 = rho * q
    b2 = -rho * q + C * kap * rho * W_e
    det = a11 * a22 - a12 * a21
    scale = max(abs(a11 * a22), abs(a12 * a21), E_FLOOR)
    if abs(det) <= 1.0e-14 * scale:
        raise FloatingPointError("singular 2T species Jacobian near sonic locus")
    dTi = (a22 * b1 - a12 * b2) / det
    dTe = (-a21 * b1 + a11 * b2) / det
    F = radiation_flux_2t(T_i, T_e, u, inv2)
    dWe = -(3.0 * kap * rho / C) * F - 4.0 * A_EV * T_e**3 * dTe
    # Summing the species rows reproduces dF/dx = C*kappa*rho*(a Te^4 - E)
    # because p = Pi - J*u; SV4 checks this numerically rather than integrating it.
    return [dTi, dTe, dWe]


def _cv_total(inv2: Invariants2T) -> float:
    return inv2.cv_i + inv2.cv_e


def _beta_i(inv2: Invariants2T) -> float:
    return inv2.cv_i / _cv_total(inv2)


def _beta_e(inv2: Invariants2T) -> float:
    return inv2.cv_e / _cv_total(inv2)


def _tau_exchange_s(rho: float, T_e: float, cfg: FLDCED2TConfig) -> float:
    return _tau_eq(rho, T_e, cfg) / cfg.qei_multiplier


def _mixture_temperature_from_y(y: Any, inv2: Invariants2T) -> float:
    return (inv2.cv_i * float(y[0]) + inv2.cv_e * float(y[1])) / _cv_total(inv2)


def _reduced_Z_abs_atol(T_ref: float) -> float:
    return max(1.0e-8 * A_EV * T_ref**4, 1.0e-4)


def _reduced_coefficients(T: float, branch: str, cfg: FLDCED2TConfig, inv2: Invariants2T) -> dict[str, float]:
    u = one_t.velocity_from_T(T, branch, inv2.base)
    J = inv2.base.mass_flux
    rho = J / u
    A = 2.0 * J * u - inv2.base.momentum_flux
    cv_tot = _cv_total(inv2)
    R = inv2.base.R_gas
    M = J * cv_tot - rho * R * T * J * R / A
    tau_ex = _tau_exchange_s(rho, T, cfg)
    K = (tau_ex / (rho * inv2.cv_e)) * (J * inv2.cv_i - rho * inv2.R_i * T * J * R / A)
    denom = M + 4.0 * C * cfg.kappa_R_cm2_g * rho * A_EV * T**3 * _beta_i(inv2) * K
    F = one_t.radiation_flux_energy(T, branch, inv2.base)
    return {
        "u": u,
        "rho": rho,
        "A": A,
        "M": M,
        "K": K,
        "denom": denom,
        "F": F,
        "F_T": one_t.dF_dT(T, branch, inv2.base),
        "tau_exchange_s": tau_ex,
    }


def rhs_reduced_2t(x: float, y: Any, branch: str, cfg: FLDCED2TConfig, inv2: Invariants2T) -> list[float]:
    del x
    T = float(y[0])
    Z = float(y[1])
    coeff = _reduced_coefficients(T, branch, cfg, inv2)
    T_prime = C * cfg.kappa_R_cm2_g * coeff["rho"] * Z / coeff["denom"]
    Z_prime = -(3.0 * cfg.kappa_R_cm2_g * coeff["rho"] / C) * coeff["F"] - 4.0 * A_EV * T**3 * T_prime
    return [T_prime, Z_prime]


def _reduced_far_Z_slope(branch: str, cfg: FLDCED2TConfig, inv2: Invariants2T) -> float:
    T_far = inv2.base.T0 if branch == "upstream" else inv2.base.T_down
    coeff = _reduced_coefficients(T_far, branch, cfg, inv2)
    A_T = 4.0 * A_EV * T_far**3
    B = -3.0 * coeff["denom"] * coeff["F_T"] / (C * C)
    if B <= 0.0 or not math.isfinite(B):
        raise RuntimeError(f"reduced {branch} far-field slope has nonpositive quadratic drive: B={B:.16e}")
    root = math.sqrt(A_T * A_T + 4.0 * B)
    if branch == "upstream":
        return 0.5 * (-A_T + root)
    if branch == "downstream":
        return 0.5 * (-A_T - root)
    raise ValueError(f"unknown branch {branch!r}")


def _full_y_from_reduced(T: float, Z: float, branch: str, cfg: FLDCED2TConfig, inv2: Invariants2T) -> np.ndarray:
    T_prime = rhs_reduced_2t(0.0, [T, Z], branch, cfg, inv2)[0]
    delta_s = _reduced_coefficients(T, branch, cfg, inv2)["K"] * T_prime
    Ti = T - _beta_e(inv2) * delta_s
    Te = T + _beta_i(inv2) * delta_s
    E = A_EV * T**4 + Z
    W_e = E - A_EV * Te**4
    return np.asarray([Ti, Te, W_e], dtype=float)


def _reduced_state_from_raw_post_jump(y_raw: np.ndarray, inv2: Invariants2T) -> np.ndarray:
    T = _mixture_temperature_from_y(y_raw, inv2)
    E = A_EV * float(y_raw[1]) ** 4 + float(y_raw[2])
    return np.asarray([T, E - A_EV * T**4], dtype=float)


def _integrate_reduced_branch(
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
    F_shock: float,
    branch: str,
    *,
    T_shock: float | None = None,
    length_hint_cm: float | None = None,
    dense: bool = False,
) -> ReducedBranchSolution:
    T_far = inv2.base.T0 if branch == "upstream" else inv2.base.T_down
    if T_shock is None:
        T_shock = one_t.temperature_for_flux(F_shock, branch, inv2.base)
    span = float(T_shock - T_far)
    if not span > 0.0:
        raise RuntimeError(f"reduced {branch} branch has nonpositive temperature span")
    T_start = T_far + max(REDUCED_ENDPOINT_CLIP_FRACTION * span, 1.0e-10)
    Z_start = _reduced_far_Z_slope(branch, cfg, inv2) * (T_start - T_far)
    length_sign = 1.0 if branch == "upstream" else -1.0
    z_abs = _reduced_Z_abs_atol(max(abs(T_far), abs(float(T_shock))))
    denom_floor = 1.0e-10 * abs(inv2.base.momentum_flux)
    D_max = 5.0 * (length_hint_cm if length_hint_cm is not None else one_t.TARGET_DOMAIN_SPAN_CM)
    if D_max <= 0.0 or not math.isfinite(D_max):
        raise RuntimeError(f"reduced {branch} branch has invalid length hint")

    def rhs(D: float, y: Any) -> list[float]:
        del D
        T = float(y[0])
        Z = float(y[1])
        T_prime, Z_prime = rhs_reduced_2t(0.0, [T, Z], branch, cfg, inv2)
        return [length_sign * T_prime, length_sign * Z_prime]

    def shock_event(D: float, y: Any) -> float:
        del D
        return float(y[0]) - float(T_shock)

    shock_event.terminal = True  # type: ignore[attr-defined]
    shock_event.direction = 1.0  # type: ignore[attr-defined]

    def sonic_event(D: float, y: Any) -> float:
        del D
        return abs(_reduced_coefficients(float(y[0]), branch, cfg, inv2)["A"]) - denom_floor

    sonic_event.terminal = True  # type: ignore[attr-defined]
    sonic_event.direction = -1.0  # type: ignore[attr-defined]

    sol = solve_ivp(
        rhs,
        (0.0, D_max),
        [T_start, Z_start],
        method="DOP853",
        rtol=min(cfg.ode_rtol, 1.0e-10),
        atol=[1.0e-8, z_abs],
        max_step=max(D_max / 300.0, 1.0e-10),
        dense_output=dense,
        events=[shock_event, sonic_event],
    )
    if sol.t_events[1].size > 0:
        raise FloatingPointError(f"reduced {branch} branch hit sonic guard before the shock state")
    if not sol.success:
        raise RuntimeError(f"reduced {branch} ODE failed: {sol.message}")
    if sol.t_events[0].size == 0:
        raise RuntimeError(
            f"reduced {branch} branch did not reach shock temperature: "
            f"T_end={float(sol.y[0, -1]):.16e}, T_shock={float(T_shock):.16e}, D_max={D_max:.16e}"
        )
    length = float(sol.t_events[0][0])
    y_end = np.asarray(sol.y_events[0][0], dtype=float)
    Z_end = float(y_end[1])
    if not length > 0.0:
        raise RuntimeError(f"reduced {branch} branch length is nonpositive")
    return ReducedBranchSolution(
        branch=branch,
        F_shock=float(F_shock),
        T_far=float(T_far),
        T_start=float(T_start),
        T_shock=float(T_shock),
        Z_shock=Z_end,
        E_shock=A_EV * float(T_shock) ** 4 + Z_end,
        length_cm=length,
        solution=sol,
        Z_abs_atol=z_abs,
    )


def _reduced_shock_parts(
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
    warm: WarmStart,
    F_shock: float,
    *,
    dense: bool = False,
) -> tuple[float, ReducedBranchSolution, ReducedBranchSolution, np.ndarray, np.ndarray, np.ndarray]:
    up = _integrate_reduced_branch(cfg, inv2, F_shock, "upstream", length_hint_cm=warm.up_len_1t, dense=dense)
    y_minus = _full_y_from_reduced(up.T_shock, up.Z_shock, "upstream", cfg, inv2)
    y_raw = solve_post_jump_state(y_minus, cfg, inv2, warm)
    T_plus, Z_plus = _reduced_state_from_raw_post_jump(y_raw, inv2)
    dn = _integrate_reduced_branch(
        cfg,
        inv2,
        F_shock,
        "downstream",
        T_shock=float(T_plus),
        length_hint_cm=warm.dn_len_1t,
        dense=dense,
    )
    y_plus_outer = _full_y_from_reduced(float(T_plus), float(Z_plus), "downstream", cfg, inv2)
    return float(Z_plus - dn.Z_shock), up, dn, y_minus, y_raw, y_plus_outer


def _reduced_solution_stats(solution: ReducedShockSolution) -> dict[str, Any]:
    cfg = solution.cfg
    inv2 = solution.inv2
    deltas: list[float] = []
    t_primes: list[float] = []
    z_values: list[float] = []
    sonic_margins: list[float] = []
    for branch_sol in (solution.up_branch, solution.dn_branch):
        samples = np.linspace(0.0, branch_sol.length_cm, 512)
        for D in samples:
            T, Z = [float(v) for v in branch_sol.solution.sol(float(D))]
            coeff = _reduced_coefficients(float(T), branch_sol.branch, cfg, inv2)
            T_prime = rhs_reduced_2t(0.0, [float(T), Z], branch_sol.branch, cfg, inv2)[0]
            deltas.append(coeff["K"] * T_prime)
            t_primes.append(T_prime)
            z_values.append(Z)
            sonic_margins.append(abs(coeff["A"]))
    max_delta = float(np.max(np.abs(np.asarray(deltas, dtype=float)))) if deltas else 0.0
    tau = tau_scales(cfg, inv2)
    return {
        "method": "reduced_outer_T_Z_dop853_shooting",
        "branch": "reduced",
        "eps_l_ei_over_l_rad": float(tau["tau_ratio"]),
        "tau_exchange_downstream_s": float(_tau_eq(tau["rho1_gcc"], tau["T1_eV"], cfg) / cfg.qei_multiplier),
        "max_abs_Delta_s_eV": max_delta,
        "max_abs_delta_ei": float(max_delta / max(tau["T1_eV"], E_FLOOR)),
        "max_abs_T_prime_eV_per_cm": float(np.max(np.abs(np.asarray(t_primes, dtype=float)))) if t_primes else 0.0,
        "max_abs_Z_erg_per_cm3": float(np.max(np.abs(np.asarray(z_values, dtype=float)))) if z_values else 0.0,
        "min_abs_sonic_A": float(np.min(np.asarray(sonic_margins, dtype=float))) if sonic_margins else math.inf,
        "Z_abs_atol_upstream": float(solution.up_branch.Z_abs_atol),
        "Z_abs_atol_downstream": float(solution.dn_branch.Z_abs_atol),
    }


def _mdz_scale(T_ref: float) -> np.ndarray:
    return np.asarray([T_ref, T_ref, max(A_EV * T_ref**4, E_FLOOR)], dtype=float)


def _fixed_point_mdz(branch: str, inv2: Invariants2T) -> np.ndarray:
    T_far = inv2.base.T0 if branch == "upstream" else inv2.base.T_down
    return np.asarray([T_far, 0.0, 0.0], dtype=float)


def _species_y_from_mdz(y: Any, inv2: Invariants2T) -> np.ndarray:
    T = float(y[0])
    delta = float(y[1])
    Z = float(y[2])
    Ti = T - _beta_e(inv2) * delta
    Te = T + _beta_i(inv2) * delta
    E = A_EV * T**4 + Z
    W_e = E - A_EV * Te**4
    return np.asarray([Ti, Te, W_e], dtype=float)


def _mdz_from_species_y(y: Any, inv2: Invariants2T) -> np.ndarray:
    Ti = float(y[0])
    Te = float(y[1])
    W_e = float(y[2])
    T = _mixture_temperature_from_y(y, inv2)
    E = A_EV * Te**4 + W_e
    return np.asarray([T, Te - Ti, E - A_EV * T**4], dtype=float)


def _lnlam_derivatives(rho: float, T_e: float, cfg: FLDCED2TConfig) -> tuple[float, float, float, float]:
    te = min(T_e, TE_POW_CAP_EV)
    if rho <= 0.0 or te <= 0.0 or not math.isfinite(rho) or not math.isfinite(te):
        raise ValueError("unphysical rho or Te in Qei tau derivative")
    n_i = rho / (cfg.A_amu * MP)
    n_e = cfg.zbar * n_i
    if n_i <= 0.0 or n_e <= 0.0:
        raise ValueError("unphysical ion/electron density in Qei tau derivative")
    if te >= 10.0 * cfg.zbar**2:
        raw = 24.0 - 0.5 * math.log(n_e) + math.log(te)
        d_raw_dte = 1.0 / te
    else:
        raw = 23.0 - 0.5 * math.log(n_e) - math.log(cfg.zbar) + 1.5 * math.log(te)
        d_raw_dte = 1.5 / te
    d_raw_drho = -0.5 / rho
    lnlam = max(LNLAM_FLOOR, raw)
    if raw <= LNLAM_FLOOR:
        return lnlam, 0.0, 0.0, te
    dte_dTe = 0.0 if T_e > TE_POW_CAP_EV else 1.0
    return lnlam, d_raw_drho, d_raw_dte * dte_dTe, te


def _tau_exchange_derivatives_mdz(
    rho: float,
    rho_T: float,
    T_e: float,
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
) -> tuple[float, float, float]:
    tau = _tau_exchange_s(rho, T_e, cfg)
    lnlam, dln_drho, dln_dTe, te = _lnlam_derivatives(rho, T_e, cfg)
    dte_dTe = 0.0 if T_e > TE_POW_CAP_EV else 1.0
    beta_i = _beta_i(inv2)
    dlogtau_dT = 1.5 * dte_dTe / te - rho_T / rho - (dln_drho * rho_T + dln_dTe) / lnlam
    dlogtau_dD = 1.5 * dte_dTe * beta_i / te - dln_dTe * beta_i / lnlam
    return tau, tau * dlogtau_dT, tau * dlogtau_dD


def _full_mdz_coefficients(T: float, branch: str, inv2: Invariants2T) -> dict[str, float]:
    u = one_t.velocity_from_T(T, branch, inv2.base)
    J = inv2.base.mass_flux
    R = inv2.base.R_gas
    rho = J / u
    A = 2.0 * J * u - inv2.base.momentum_flux
    u_T = -J * R / A
    rho_T = -rho * u_T / u
    A_T = 2.0 * J * u_T
    G = J * R * R
    M_term_T = G * (rho_T * T / A + rho / A - rho * T * A_T / (A * A))
    M = J * _cv_total(inv2) - rho * R * T * J * R / A
    M_T = -M_term_T
    F = one_t.radiation_flux_energy(T, branch, inv2.base)
    F_T = one_t.dF_dT(T, branch, inv2.base)
    return {
        "u": u,
        "rho": rho,
        "A": A,
        "u_T": u_T,
        "rho_T": rho_T,
        "A_T": A_T,
        "M": M,
        "M_T": M_T,
        "F": F,
        "F_T": F_T,
    }


def rhs_full_mdz(x: float, y: Any, branch: str, cfg: FLDCED2TConfig, inv2: Invariants2T) -> list[float]:
    del x
    T = float(y[0])
    delta = float(y[1])
    Z = float(y[2])
    coeff = _full_mdz_coefficients(T, branch, inv2)
    beta_i = _beta_i(inv2)
    beta_e = _beta_e(inv2)
    Te = T + beta_i * delta
    Ti = T - beta_e * delta
    if Ti <= 0.0 or Te <= 0.0:
        raise FloatingPointError("nonpositive full-branch species temperature")
    W_e = Z + A_EV * T**4 - A_EV * Te**4
    J = inv2.base.mass_flux
    T_prime = C * cfg.kappa_R_cm2_g * coeff["rho"] * W_e / coeff["M"]
    u_prime = coeff["u_T"] * T_prime
    p_i = coeff["rho"] * inv2.R_i * Ti
    tau = _tau_exchange_s(coeff["rho"], Te, cfg)
    delta_prime = (
        J * inv2.cv_i * T_prime
        + p_i * u_prime
        - coeff["rho"] * inv2.cv_e * delta / tau
    ) / (J * inv2.cv_i * beta_e)
    Z_prime = -(3.0 * cfg.kappa_R_cm2_g * coeff["rho"] / C) * coeff["F"] - 4.0 * A_EV * T**3 * T_prime
    return [T_prime, delta_prime, Z_prime]


def jac_full_mdz(x: float, y: Any, branch: str, cfg: FLDCED2TConfig, inv2: Invariants2T) -> np.ndarray:
    del x
    T = float(y[0])
    delta = float(y[1])
    Z = float(y[2])
    coeff = _full_mdz_coefficients(T, branch, inv2)
    beta_i = _beta_i(inv2)
    beta_e = _beta_e(inv2)
    Te = T + beta_i * delta
    Ti = T - beta_e * delta
    if Ti <= 0.0 or Te <= 0.0:
        raise FloatingPointError("nonpositive full-branch species temperature")
    rho = coeff["rho"]
    rho_T = coeff["rho_T"]
    M = coeff["M"]
    M_T = coeff["M_T"]
    W_e = Z + A_EV * T**4 - A_EV * Te**4
    W_T = 4.0 * A_EV * (T**3 - Te**3)
    W_D = -4.0 * A_EV * Te**3 * beta_i
    S = C * cfg.kappa_R_cm2_g * rho / M
    S_T = S * (rho_T / rho - M_T / M)
    T_p = S * W_e
    T_p_T = S_T * W_e + S * W_T
    T_p_D = S * W_D
    T_p_Z = S

    u_T = coeff["u_T"]
    u_TT = inv2.base.mass_flux * inv2.base.R_gas * coeff["A_T"] / (coeff["A"] * coeff["A"])
    u_p = u_T * T_p
    u_p_T = u_TT * T_p + u_T * T_p_T
    u_p_D = u_T * T_p_D
    u_p_Z = u_T * T_p_Z

    p_i = rho * inv2.R_i * Ti
    p_i_T = inv2.R_i * (rho_T * Ti + rho)
    p_i_D = -rho * inv2.R_i * beta_e
    tau, tau_T, tau_D = _tau_exchange_derivatives_mdz(rho, rho_T, Te, cfg, inv2)
    qfac_T = inv2.cv_e * delta * (rho_T / tau - rho * tau_T / (tau * tau))
    qfac_D = inv2.cv_e * (rho / tau - rho * delta * tau_D / (tau * tau))

    J = inv2.base.mass_flux
    denom = J * inv2.cv_i * beta_e
    N_T = J * inv2.cv_i * T_p_T + p_i_T * u_p + p_i * u_p_T - qfac_T
    N_D = J * inv2.cv_i * T_p_D + p_i_D * u_p + p_i * u_p_D - qfac_D
    N_Z = J * inv2.cv_i * T_p_Z + p_i * u_p_Z

    Z_source_T = -(3.0 * cfg.kappa_R_cm2_g / C) * (rho_T * coeff["F"] + rho * coeff["F_T"])
    Z_p_T = Z_source_T - 4.0 * A_EV * (3.0 * T * T * T_p + T**3 * T_p_T)
    Z_p_D = -4.0 * A_EV * T**3 * T_p_D
    Z_p_Z = -4.0 * A_EV * T**3 * T_p_Z

    return np.asarray(
        [
            [T_p_T, T_p_D, T_p_Z],
            [N_T / denom, N_D / denom, N_Z / denom],
            [Z_p_T, Z_p_D, Z_p_Z],
        ],
        dtype=float,
    )


def _full_eigen_seed(cfg: FLDCED2TConfig, inv2: Invariants2T) -> FullEigenSeeds:
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for branch in ("upstream", "downstream"):
        y_far = _fixed_point_mdz(branch, inv2)
        Jmat = jac_full_mdz(0.0, y_far, branch, cfg, inv2)
        values_c, vectors_c = eig(Jmat)
        real_scale = max(float(np.max(np.abs(np.real(values_c)))), 1.0)
        if float(np.max(np.abs(np.imag(values_c)))) > 1.0e-8 * real_scale:
            raise RuntimeError(f"{branch} full fixed-point spectrum has complex pairs: {values_c.tolist()}")
        spectra[branch] = (np.real(values_c), np.real(vectors_c))
    lam_up, vec_up = spectra["upstream"]
    lam_dn, vec_dn = spectra["downstream"]
    grow_up = [i for i, val in enumerate(lam_up) if val > 0.0]
    decay_up = [i for i, val in enumerate(lam_up) if val < 0.0]
    grow_dn = [i for i, val in enumerate(lam_dn) if val > 0.0]
    decay_dn = [i for i, val in enumerate(lam_dn) if val < 0.0]
    if len(grow_up) != 1 or len(decay_up) != 2:
        raise RuntimeError(_spectrum_error("full upstream", lam_up, vec_up))
    if len(grow_dn) != 1 or len(decay_dn) != 2:
        raise RuntimeError(_spectrum_error("full downstream", lam_dn, vec_dn))
    v_up = np.asarray(vec_up[:, grow_up[0]], dtype=float)
    if v_up[0] < 0.0:
        v_up = -v_up
    if abs(float(v_up[0])) <= np.finfo(float).tiny:
        raise RuntimeError("full upstream growing eigenvector has zero T component")
    decay_dn_sorted = sorted(decay_dn, key=lambda idx: abs(float(lam_dn[idx])))

    J_dn = jac_full_mdz(0.0, _fixed_point_mdz("downstream", inv2), "downstream", cfg, inv2)
    lam_left_c, vec_left_c = eig(J_dn.T)
    lam_grow = float(lam_dn[grow_dn[0]])
    left_grow = _left_eigen_index(lam_left_c, lam_grow, "full downstream growing")
    scale_dn = _mdz_scale(inv2.base.T_down)
    w_grow = _normalized_left_eigenvector(vec_left_c[:, left_grow], scale_dn, "full downstream growing")
    return FullEigenSeeds(
        upstream_eigenvalues=[float(v) for v in lam_up],
        downstream_eigenvalues=[float(v) for v in lam_dn],
        upstream_vector=v_up,
        downstream_left_grow_vector=w_grow,
        downstream_lambda_slow=float(lam_dn[decay_dn_sorted[0]]),
        downstream_lambda_fast=float(lam_dn[decay_dn_sorted[1]]),
        downstream_lambda_grow=lam_grow,
        upstream_lambda_grow=float(lam_up[grow_up[0]]),
    )


def _denom_from_y(y: Any, branch: str, inv2: Invariants2T) -> float:
    T_i = float(y[0])
    T_e = float(y[1])
    theta = inv2.R_i * T_i + inv2.R_e * T_e
    u = velocity_from_theta(theta, branch, inv2)
    return 2.0 * inv2.base.mass_flux * u - inv2.base.momentum_flux


def jacobian_at(y_far: np.ndarray, branch: str, cfg: FLDCED2TConfig, inv2: Invariants2T) -> np.ndarray:
    T_far = float(y_far[0])
    steps = np.asarray(
        [1.0e-6 * T_far, 1.0e-6 * T_far, 1.0e-6 * A_EV * T_far**4],
        dtype=float,
    )
    J = np.zeros((3, 3), dtype=float)
    for k, step in enumerate(steps):
        y_p = np.array(y_far, dtype=float)
        y_m = np.array(y_far, dtype=float)
        y_p[k] += step
        y_m[k] -= step
        f_p = np.asarray(rhs_2t(0.0, y_p, branch, cfg, inv2), dtype=float)
        f_m = np.asarray(rhs_2t(0.0, y_m, branch, cfg, inv2), dtype=float)
        J[:, k] = (f_p - f_m) / (2.0 * step)
    return J


def _normalized_eigenvector(vec: np.ndarray, T_far: float) -> np.ndarray:
    scale = np.asarray([T_far, T_far, A_EV * T_far**4], dtype=float)
    dimless = np.real(vec) / scale
    norm = float(np.max(np.abs(dimless)))
    if norm <= 0.0 or not math.isfinite(norm):
        raise RuntimeError("zero or nonfinite eigenvector")
    return dimless / norm


def _left_eigen_index(lam_left_c: np.ndarray, target: float, label: str) -> int:
    lam_left = np.real(lam_left_c)
    distances = np.abs(lam_left - target)
    order = np.argsort(distances)
    best = int(order[0])
    best_dist = float(distances[best])
    match_tol = 1.0e-3 * max(abs(target), 1.0)
    if best_dist > match_tol:
        raise RuntimeError(
            f"left/right {label} eigenvalue pairing failed: "
            f"target={target:.16e}, left_spectrum={lam_left_c.tolist()}"
        )
    if order.size > 1 and float(distances[int(order[1])]) <= match_tol:
        raise RuntimeError(
            f"left/right {label} eigenvalue pairing is ambiguous: "
            f"target={target:.16e}, left_spectrum={lam_left_c.tolist()}"
        )
    return best


def _normalized_left_eigenvector(vec: np.ndarray, scale: np.ndarray, label: str) -> np.ndarray:
    w_hat = np.real(vec) * scale
    w_norm = float(np.linalg.norm(w_hat))
    if w_norm <= 0.0 or not math.isfinite(w_norm):
        raise RuntimeError(f"zero or nonfinite left {label} eigenvector")
    return w_hat / w_norm


def _spectrum_error(branch: str, values: np.ndarray, vectors: np.ndarray) -> str:
    return (
        f"{branch} eigenstructure violates the 2T matching topology: "
        f"eigenvalues={values.tolist()}, eigenvectors={vectors.tolist()}"
    )


def _spectra_context(seeds: EigenSeeds) -> str:
    return f"upstream_spectrum={seeds.upstream_eigenvalues}, downstream_spectrum={seeds.downstream_eigenvalues}"


def eigen_seed(cfg: FLDCED2TConfig, inv2: Invariants2T) -> EigenSeeds:
    y_up = fixed_point("upstream", inv2)
    y_dn = fixed_point("downstream", inv2)
    J_up = jacobian_at(y_up, "upstream", cfg, inv2)
    J_dn = jacobian_at(y_dn, "downstream", cfg, inv2)
    lam_up_c, vec_up_c = eig(J_up)
    lam_dn_c, vec_dn_c = eig(J_dn)
    for branch, lam_c in (("upstream", lam_up_c), ("downstream", lam_dn_c)):
        real_scale = max(float(np.max(np.abs(np.real(lam_c)))), 1.0)
        if float(np.max(np.abs(np.imag(lam_c)))) > 1.0e-8 * real_scale:
            raise RuntimeError(f"{branch} fixed-point spectrum has complex pairs: {lam_c.tolist()}")
    lam_up = np.real(lam_up_c)
    lam_dn = np.real(lam_dn_c)
    grow_up = [i for i, val in enumerate(lam_up) if val > 0.0]
    decay_up = [i for i, val in enumerate(lam_up) if val < 0.0]
    grow_dn = [i for i, val in enumerate(lam_dn) if val > 0.0]
    decay_dn = [i for i, val in enumerate(lam_dn) if val < 0.0]
    if len(grow_up) != 1 or len(decay_up) != 2:
        raise RuntimeError(_spectrum_error("upstream", lam_up, np.real(vec_up_c)))
    if len(grow_dn) != 1 or len(decay_dn) != 2:
        raise RuntimeError(_spectrum_error("downstream", lam_dn, np.real(vec_dn_c)))

    decay_up_sorted = sorted(decay_up, key=lambda idx: abs(lam_up[idx]))
    v_up = _normalized_eigenvector(vec_up_c[:, grow_up[0]], inv2.base.T0)
    if v_up[2] < 0.0:
        v_up = -v_up
    v_up_fast = _normalized_eigenvector(vec_up_c[:, decay_up_sorted[1]], inv2.base.T0)

    decay_sorted = sorted(decay_dn, key=lambda idx: abs(lam_dn[idx]))
    v_slow = _normalized_eigenvector(vec_dn_c[:, decay_sorted[0]], inv2.base.T_down)
    v_fast = _normalized_eigenvector(vec_dn_c[:, decay_sorted[1]], inv2.base.T_down)
    if v_slow[2] > 0.0:
        v_slow = -v_slow
    if v_fast[1] - v_fast[0] < 0.0:
        v_fast = -v_fast

    lam_left_up_c, vec_left_up_c = eig(J_up.T)
    scale_up = np.asarray([inv2.base.T0, inv2.base.T0, A_EV * inv2.base.T0**4], dtype=float)
    left_up_fast = _left_eigen_index(lam_left_up_c, float(lam_up[decay_up_sorted[1]]), "upstream fast")
    w_up_fast = _normalized_left_eigenvector(vec_left_up_c[:, left_up_fast], scale_up, "upstream fast")

    lam_left_c, vec_left_c = eig(J_dn.T)
    lam_grow_dn = float(lam_dn[grow_dn[0]])
    scale_dn = np.asarray([inv2.base.T_down, inv2.base.T_down, A_EV * inv2.base.T_down**4], dtype=float)
    left_grow = _left_eigen_index(lam_left_c, lam_grow_dn, "growing")
    left_fast = _left_eigen_index(lam_left_c, float(lam_dn[decay_sorted[1]]), "fast")
    if left_grow == left_fast:
        raise RuntimeError(
            "downstream left growing/fast eigenvectors paired to the same mode: "
            f"right_spectrum={lam_dn.tolist()}, left_spectrum={lam_left_c.tolist()}"
        )
    w_hat = _normalized_left_eigenvector(vec_left_c[:, left_grow], scale_dn, "growing")
    w_fast = _normalized_left_eigenvector(vec_left_c[:, left_fast], scale_dn, "fast")

    actual_v_up = scale_up * v_up
    q_E = one_t.far_asymptotic_W_slope("upstream", inv2.base)
    temp_component = 0.5 * (actual_v_up[0] + actual_v_up[1])
    measured = math.inf if abs(temp_component) <= E_FLOOR else actual_v_up[2] / temp_component
    rel = abs(measured - q_E) / max(abs(q_E), E_FLOOR)
    warning = {
        "checked": cfg.qei_multiplier >= 1.0e6,
        "passed": (rel <= 0.2) if cfg.qei_multiplier >= 1.0e6 else None,
        "q_E_1T": q_E,
        "q_E_from_2T_eigenvector": measured,
        "relative_difference": rel,
    }
    return EigenSeeds(
        upstream_eigenvalues=[float(v) for v in lam_up],
        downstream_eigenvalues=[float(v) for v in lam_dn],
        upstream_vector=v_up,
        upstream_fast_vector=v_up_fast,
        upstream_left_fast_vector=w_up_fast,
        downstream_slow_vector=v_slow,
        downstream_fast_vector=v_fast,
        downstream_left_fast_vector=w_fast,
        downstream_left_grow_vector=w_hat,
        downstream_lambda_slow=float(lam_dn[decay_sorted[0]]),
        downstream_lambda_fast=float(lam_dn[decay_sorted[1]]),
        downstream_lambda_grow=lam_grow_dn,
        upstream_lambda_grow=float(lam_up[grow_up[0]]),
        upstream_slope_warning=warning,
    )


def _seed_state(y_far: np.ndarray, vec: np.ndarray, eps: float) -> np.ndarray:
    T_far = float(y_far[0])
    scale = np.asarray([T_far, T_far, A_EV * T_far**4], dtype=float)
    return y_far + eps * scale * vec


def integrate_upstream_once(
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
    seeds: EigenSeeds,
    X_max: float,
) -> tuple[Any, float]:
    y_far = fixed_point("upstream", inv2)
    y0 = _seed_state(y_far, seeds.upstream_vector, cfg.seed_amplitude_rel)
    scale = np.asarray([inv2.base.T0, inv2.base.T0, A_EV * inv2.base.T0**4], dtype=float)
    # High-Qei upstream starts contain an unphysical fast layer only through roundoff;
    # remove it before BDF chooses its startup history.
    y0, upstream_fast_amplitude_removed = _upstream_slow_manifold_projection(
        y0,
        y_far,
        scale,
        seeds.upstream_fast_vector,
        seeds.upstream_left_fast_vector,
    )

    def rhs(x: float, y: Any) -> list[float]:
        return rhs_2t(x, y, "upstream", cfg, inv2)

    def sonic_event(x: float, y: Any) -> float:
        del x
        return _denom_from_y(y, "upstream", inv2) - 1.0e-10 * inv2.base.momentum_flux

    sonic_event.terminal = True  # type: ignore[attr-defined]
    sonic_event.direction = -1.0  # type: ignore[attr-defined]

    def negT_event(x: float, y: Any) -> float:
        del x
        return min(float(y[0]), float(y[1])) - 1.0e-6 * cfg.T0_eV

    negT_event.terminal = True  # type: ignore[attr-defined]
    negT_event.direction = -1.0  # type: ignore[attr-defined]

    def w_cap_event(x: float, y: Any) -> float:
        del x
        return 1.5 * A_EV * inv2.base.T_sonic**4 - float(y[2])

    w_cap_event.terminal = True  # type: ignore[attr-defined]
    w_cap_event.direction = -1.0  # type: ignore[attr-defined]

    sol = solve_ivp(
        rhs,
        (0.0, X_max),
        y0,
        method="BDF",
        rtol=cfg.ode_rtol,
        atol=[cfg.ode_atol_T_eV, cfg.ode_atol_T_eV, cfg.ode_atol_W],
        dense_output=True,
        events=[sonic_event, negT_event, w_cap_event],
    )
    if not sol.success and sol.t.size <= 1:
        raise RuntimeError(f"upstream ODE failed immediately: {sol.message}")
    if sol.status == -1:
        # status -1 with progress = controller crash in the unphysical overshoot zone past the W-cap; reach remains valid
        pass
    x_reach = float(sol.t[-1])
    if x_reach < 1.0e-3 * X_max:
        raise RuntimeError(f"upstream seed immediately invalid: x_reach={x_reach:.6e}, X_max={X_max:.6e}")
    sol.tenryu_upstream_projection_diagnostics = {
        "projection_applied": True,
        "projection_amplitude_removed": float(upstream_fast_amplitude_removed),
    }
    return sol, x_reach


DOWNSTREAM_NEWTON_TOL = 1.0e-11
DOWNSTREAM_NEWTON_STAGNATION_TOL = 1.0e-9
DOWNSTREAM_NEWTON_MAX_ITER = 50
DOWNSTREAM_NEWTON_MAX_HALVINGS = 8


def _downstream_state_scale(inv2: Invariants2T) -> np.ndarray:
    return np.asarray([inv2.base.T_down, inv2.base.T_down, A_EV * inv2.base.T_down**4], dtype=float)


def _downstream_continuity_residual_scale(y_far: np.ndarray) -> np.ndarray:
    T1 = float(y_far[0])
    # Fixed once from the downstream far state: Ti/Te use T1, and W_e uses
    # the far-field W amplitude with a blackbody floor so residuals are not
    # judged on an absolute erg/cm3 scale.
    return np.asarray([T1, T1, max(abs(float(y_far[2])), A_EV * T1**4 * 1.0e-6)], dtype=float)


def _fast_mode_projection(
    y_start: np.ndarray,
    y_far: np.ndarray,
    scale: np.ndarray,
    v_fast: np.ndarray,
    w_fast: np.ndarray,
    label: str,
) -> tuple[np.ndarray, float]:
    denom = float(np.dot(w_fast, v_fast))
    if abs(denom) <= np.finfo(float).tiny or not math.isfinite(denom):
        raise RuntimeError(f"{label} fast-mode projection has zero left/right pairing")
    delta_dimless = (y_start - y_far) / scale
    amplitude = float(np.dot(w_fast, delta_dimless) / denom)
    y0 = y_start - scale * amplitude * v_fast
    if not np.all(np.isfinite(y0)):
        raise FloatingPointError(f"{label} fast-mode projection produced nonfinite state")
    return y0, amplitude


def _downstream_slow_manifold_projection(
    y_jump: np.ndarray,
    y_far: np.ndarray,
    scale: np.ndarray,
    v_fast: np.ndarray,
    w_fast: np.ndarray,
) -> tuple[np.ndarray, float]:
    return _fast_mode_projection(y_jump, y_far, scale, v_fast, w_fast, "downstream")


def _upstream_slow_manifold_projection(
    y_seed: np.ndarray,
    y_far: np.ndarray,
    scale: np.ndarray,
    v_fast: np.ndarray,
    w_fast: np.ndarray,
) -> tuple[np.ndarray, float]:
    return _fast_mode_projection(y_seed, y_far, scale, v_fast, w_fast, "upstream")


def _downstream_shooting_breakpoints(L_dn: float, lam_fast: float) -> np.ndarray:
    if L_dn <= 0.0 or lam_fast <= 0.0:
        raise ValueError("downstream shooting lengths must be positive")
    first = min(2.0 / lam_fast, 0.5 * L_dn)
    max_width = L_dn / 8.0
    growth = 10.0
    widths_list: list[float] = []
    width = first
    filled = 0.0
    while filled + width < L_dn and width < max_width:
        widths_list.append(width)
        filled += width
        width *= growth
    remaining = L_dn - filled
    if remaining > 0.0:
        n_tail = max(1, int(math.ceil(remaining / max_width)))
        widths_list.extend([remaining / n_tail] * n_tail)
    widths = np.asarray(widths_list, dtype=float)
    nodes = np.concatenate(([0.0], np.cumsum(widths)))
    nodes[-1] = L_dn
    if np.any(np.diff(nodes) <= 0.0):
        raise RuntimeError(f"downstream shooting breakpoints are not strictly increasing: {nodes.tolist()}")
    return nodes


def _pack_downstream_unknowns(Y: np.ndarray, ell_up: float) -> np.ndarray:
    return np.concatenate((Y[:, 1:].T.reshape(-1), np.asarray([ell_up], dtype=float)))


def _unpack_downstream_unknowns(z: np.ndarray, n_breakpoints: int) -> tuple[np.ndarray, float]:
    states = z[:-1].reshape((n_breakpoints - 1, 3)).T
    Y = np.empty((3, n_breakpoints), dtype=float)
    Y[:, 1:] = states
    return Y, float(z[-1])


def _downstream_residual_breakdown(
    residual: np.ndarray,
    residual_scale: np.ndarray,
    Y: np.ndarray,
    ell_up: float,
    breakpoints: np.ndarray,
) -> dict[str, Any]:
    scaled = residual / residual_scale
    n_blocks = (residual.size - 1) // 3
    continuity = residual[:-1].reshape((n_blocks, 3))
    continuity_scaled = scaled[:-1].reshape((n_blocks, 3))
    block_inf = np.max(np.abs(continuity), axis=1) if n_blocks else np.asarray([], dtype=float)
    block_scaled_inf = np.max(np.abs(continuity_scaled), axis=1) if n_blocks else np.asarray([], dtype=float)
    return {
        "residual_inf": float(np.max(np.abs(residual))),
        "residual_l2": float(np.linalg.norm(residual)),
        "scaled_residual_inf": float(np.max(np.abs(scaled))),
        "scaled_residual_l2": float(np.linalg.norm(scaled)),
        "continuity_block_inf": [float(v) for v in block_inf],
        "continuity_scaled_block_inf": [float(v) for v in block_scaled_inf],
        "continuity_components": continuity.tolist(),
        "continuity_scaled_components": continuity_scaled.tolist(),
        "far_projection": float(residual[-1]),
        "far_projection_scaled": float(scaled[-1]),
        "residual_scales": [float(v) for v in residual_scale],
        "max_continuity_block": int(np.argmax(block_scaled_inf)) if n_blocks else None,
        "ell_up_cm": float(ell_up),
        "breakpoints_cm": [float(v) for v in breakpoints],
        "last_state": [float(v) for v in Y[:, -1]],
    }


def _integrate_downstream_interval(
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
    y0: np.ndarray,
    s0: float,
    s1: float,
    *,
    dense_output: bool,
) -> Any:
    if not np.all(np.isfinite(y0)):
        raise FloatingPointError("nonfinite downstream shooting state")
    if float(np.min(y0[0:2])) <= 0.0:
        raise FloatingPointError("nonpositive downstream shooting temperature")
    width = float(s1 - s0)
    if width <= 0.0:
        raise ValueError("downstream shooting interval width must be positive")
    state_scale = _downstream_state_scale(inv2)
    z0 = np.asarray(y0, dtype=float) / state_scale

    denom_floor = 1.0e-10 * abs(inv2.base.momentum_flux)

    def rhs(u: float, z: Any) -> list[float]:
        del u
        y = state_scale * np.asarray(z, dtype=float)
        return (width * np.asarray(rhs_2t(0.0, y, "downstream", cfg, inv2), dtype=float) / state_scale).tolist()

    def jac(u: float, z: Any) -> np.ndarray:
        del u
        z_arr = np.asarray(z, dtype=float)
        steps = np.asarray(
            [
                1.0e-7 * max(abs(float(z_arr[0])), 1.0),
                1.0e-7 * max(abs(float(z_arr[1])), 1.0),
                1.0e-7 * max(abs(float(z_arr[2])), 1.0),
            ],
            dtype=float,
        )
        out = np.empty((3, 3), dtype=float)
        for k, step in enumerate(steps):
            z_p = np.array(z_arr, dtype=float)
            z_m = np.array(z_arr, dtype=float)
            z_p[k] += step
            z_m[k] -= step
            f_p = np.asarray(rhs_2t(0.0, state_scale * z_p, "downstream", cfg, inv2), dtype=float) / state_scale
            f_m = np.asarray(rhs_2t(0.0, state_scale * z_m, "downstream", cfg, inv2), dtype=float) / state_scale
            out[:, k] = width * (f_p - f_m) / (2.0 * step)
        return out

    def sonic_event(u: float, z: Any) -> float:
        del u
        y = state_scale * np.asarray(z, dtype=float)
        return abs(_denom_from_y(y, "downstream", inv2)) - denom_floor

    sonic_event.terminal = True  # type: ignore[attr-defined]
    sonic_event.direction = -1.0  # type: ignore[attr-defined]

    def negT_event(u: float, z: Any) -> float:
        del u
        y = state_scale * np.asarray(z, dtype=float)
        return min(float(y[0]), float(y[1])) - 1.0e-12 * inv2.base.T_down

    negT_event.terminal = True  # type: ignore[attr-defined]
    negT_event.direction = -1.0  # type: ignore[attr-defined]

    sol = solve_ivp(
        rhs,
        (0.0, 1.0),
        z0,
        method="Radau",
        rtol=max(cfg.ode_rtol, 1.0e-7),
        atol=[
            max(cfg.ode_atol_T_eV / state_scale[0], 1.0e-10),
            max(cfg.ode_atol_T_eV / state_scale[1], 1.0e-10),
            max(cfg.ode_atol_W / state_scale[2], 1.0e-10),
        ],
        max_step=0.1,
        dense_output=dense_output,
        events=[sonic_event, negT_event],
    )
    if sol.status == 1:
        raise FloatingPointError(
            "downstream shooting interval hit sonic/temperature guard: "
            f"s0={s0:.16e}, s1={s1:.16e}, y0={np.asarray(y0, dtype=float).tolist()}"
        )
    if not sol.success:
        raise RuntimeError(
            "downstream shooting interval failed: "
            f"message={sol.message!r}, s0={s0:.16e}, s1={s1:.16e}, y0={np.asarray(y0, dtype=float).tolist()}"
        )
    if not np.all(np.isfinite(sol.y[:, -1])):
        raise FloatingPointError("downstream shooting interval ended nonfinite")
    return DownstreamIntervalSolution(raw=sol, s0=float(s0), s1=float(s1), scale=state_scale)


def solve_downstream_multiple_shooting(
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
    seeds: EigenSeeds,
    up_sol: Any,
    up_reach: float,
    warm: WarmStart,
    ell_up_guess: float,
    prev: Any = None,
) -> tuple[Any, float, float]:
    """Experimental full-branch downstream solve used only behind ``--solver full``."""
    lam_slow = abs(seeds.downstream_lambda_slow)
    lam_fast = abs(seeds.downstream_lambda_fast)
    L_dn = 30.0 / lam_slow
    y_far = fixed_point("downstream", inv2)
    scale = _downstream_state_scale(inv2)
    continuity_residual_scale = _downstream_continuity_residual_scale(y_far)
    far_projection_weights = seeds.downstream_left_grow_vector / scale
    far_projection_scale = abs(float(np.dot(far_projection_weights, y_far)))
    if far_projection_scale <= 0.0 or not math.isfinite(far_projection_scale):
        far_projection_scale = 1.0
    theta_limit = inv2.base.momentum_flux**2 / (4.0 * inv2.base.mass_flux**2)
    T_floor = 1.0e-9 * inv2.base.T_down
    ell_lo = 0.02 * warm.up_len_1t
    ell_hi = 0.98 * up_reach
    if not ell_lo < ell_hi:
        raise RuntimeError(
            "downstream shooting ell_up interval is empty: "
            f"lower={ell_lo:.16e}, upper={ell_hi:.16e}, up_reach={up_reach:.16e}"
        )

    s_nodes = _downstream_shooting_breakpoints(L_dn, lam_fast)
    h1 = float(s_nodes[1] - s_nodes[0])
    lam_fast_h1 = float(lam_fast * h1)
    lam_fast_over_slow = float(lam_fast / lam_slow)
    projection_applied = lam_fast_over_slow > 100.0

    def downstream_start_state(y_jump: np.ndarray) -> tuple[np.ndarray, float]:
        if not projection_applied:
            return np.asarray(y_jump, dtype=float), 0.0
        # The dropped fast-layer contribution is O(exp(-lam_fast*l_layer)) and
        # identically unrepresentable; this is the asymptotic layer limit.
        y0, amplitude = _downstream_slow_manifold_projection(
            np.asarray(y_jump, dtype=float),
            y_far,
            scale,
            seeds.downstream_fast_vector,
            seeds.downstream_left_fast_vector,
        )
        if float(np.min(y0[0:2])) <= 0.0:
            raise RuntimeError("downstream slow-manifold projection produced nonpositive temperature")
        return y0, amplitude

    ell_up_guess_eval = min(max(float(ell_up_guess), ell_lo), ell_hi)
    y_minus0 = np.asarray(up_sol.sol(ell_up_guess_eval), dtype=float)
    y_plus0 = solve_post_jump_state(y_minus0, cfg, inv2, warm)
    y_start0, _ = downstream_start_state(y_plus0)
    if prev is not None:
        prev_L = float(getattr(prev, "L_dn_span", L_dn))
        y_init = np.empty((3, s_nodes.size), dtype=float)
        y_init[:, 0] = y_start0
        for j, s in enumerate(s_nodes[1:], start=1):
            y_init[:, j] = np.asarray(prev.sol(min(float(s) / prev_L, 1.0)), dtype=float)
        y_init[0:2, :] = np.maximum(y_init[0:2, :], 1.0e-12 * inv2.base.T_down)
    else:
        basis = np.column_stack((seeds.downstream_slow_vector, seeds.downstream_fast_vector))
        dimless_jump = (y_start0 - y_far) / scale
        amps = np.linalg.lstsq(basis, dimless_jump, rcond=None)[0]
        slow = amps[0] * np.exp(seeds.downstream_lambda_slow * s_nodes)
        fast = amps[1] * np.exp(seeds.downstream_lambda_fast * s_nodes)
        y_init = y_far[:, None] + scale[:, None] * (
            seeds.downstream_slow_vector[:, None] * slow[None, :]
            + seeds.downstream_fast_vector[:, None] * fast[None, :]
        )
        y_init[:, 0] = y_start0
        if float(np.min(y_init[0:2, :])) <= 0.0:
            raise RuntimeError("downstream shooting eigenmode initial guess produced nonpositive temperature")
        tail_scaled = np.max(np.abs((y_init - y_far[:, None]) / scale[:, None]), axis=0)
        tail_mask = tail_scaled < 1.0e-3
        tail_mask[0] = False
        y_init[:, tail_mask] = y_far[:, None]

    z = _pack_downstream_unknowns(y_init, ell_up_guess_eval)
    unknown_scale = np.concatenate((np.tile(scale, s_nodes.size - 1), np.asarray([max(up_reach, warm.up_len_1t, 1.0e-30)])))
    residual_scale = np.concatenate(
        (
            np.tile(continuity_residual_scale, s_nodes.size - 1),
            np.asarray([far_projection_scale], dtype=float),
        )
    )

    def interval_residual(Y: np.ndarray, k: int, *, dense_output: bool = False) -> tuple[np.ndarray, Any]:
        seg = _integrate_downstream_interval(
            cfg,
            inv2,
            Y[:, k],
            float(s_nodes[k]),
            float(s_nodes[k + 1]),
            dense_output=dense_output,
        )
        return np.asarray(seg.y[:, -1], dtype=float) - Y[:, k + 1], seg

    def evaluate(z_in: np.ndarray, *, dense_output: bool = False) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
        Y, ell_up = _unpack_downstream_unknowns(z_in, s_nodes.size)
        if not ell_lo < ell_up < ell_hi:
            raise FloatingPointError(
                f"ell_up outside allowed branch interval: ell_up={ell_up:.16e}, "
                f"lower={ell_lo:.16e}, upper={ell_hi:.16e}"
            )
        y_minus = np.asarray(up_sol.sol(ell_up), dtype=float)
        Y[:, 0], _ = downstream_start_state(solve_post_jump_state(y_minus, cfg, inv2, warm))
        residuals: list[float] = []
        segments: list[Any] = []
        for k in range(s_nodes.size - 1):
            block, seg = interval_residual(Y, k, dense_output=dense_output)
            residuals.extend(block.tolist())
            if dense_output:
                segments.append(seg)
        residuals.append(float(np.dot(far_projection_weights, Y[:, -1] - y_far)))
        res = np.asarray(residuals, dtype=float)
        if not np.all(np.isfinite(res)):
            raise FloatingPointError("downstream shooting residual is nonfinite")
        return res, Y, tuple(segments)

    def scaled_residual(residual: np.ndarray) -> np.ndarray:
        return residual / residual_scale

    def residual_norm(residual: np.ndarray) -> float:
        return float(np.max(np.abs(scaled_residual(residual))))

    def try_residual(z_in: np.ndarray) -> np.ndarray | None:
        try:
            return evaluate(z_in)[0]
        except (FloatingPointError, RuntimeError, ValueError):
            return None

    def finite_difference_jacobian(z_base: np.ndarray, residual_base: np.ndarray, Y_base: np.ndarray) -> np.ndarray:
        sqrt_eps = math.sqrt(np.finfo(float).eps)
        J = np.empty((residual_base.size, z_base.size), dtype=float)
        J[:, :] = 0.0
        n_nodes = s_nodes.size
        for node in range(1, n_nodes):
            for component in range(3):
                col = 3 * (node - 1) + component
                step = sqrt_eps * max(abs(float(Y_base[component, node])), float(unknown_scale[col]), 1.0)
                prev_rows = slice(3 * (node - 1), 3 * node)
                J[prev_rows, col] = 0.0
                J[3 * (node - 1) + component, col] = -1.0
                if node < n_nodes - 1:
                    Y_p = np.array(Y_base, dtype=float)
                    Y_p[component, node] += step
                    block_p, _ = interval_residual(Y_p, node)
                    rows = slice(3 * node, 3 * (node + 1))
                    J[rows, col] = (block_p - residual_base[rows]) / step
                else:
                    J[-1, col] = far_projection_weights[component]

        ell_col = z_base.size - 1
        ell_step = sqrt_eps * max(abs(float(z_base[-1])), float(unknown_scale[-1]), 1.0e-30)
        ell_step = min(ell_step, 0.25 * (ell_hi - float(z_base[-1])), 0.25 * (float(z_base[-1]) - ell_lo))
        if ell_step <= 0.0 or not math.isfinite(ell_step):
            raise RuntimeError("cannot finite-difference ell_up at the branch interval boundary")
        Y_p = np.array(Y_base, dtype=float)
        y_minus_p = np.asarray(up_sol.sol(float(z_base[-1]) + ell_step), dtype=float)
        Y_p[:, 0], _ = downstream_start_state(solve_post_jump_state(y_minus_p, cfg, inv2, warm))
        block_p, _ = interval_residual(Y_p, 0)
        J[0:3, ell_col] = (block_p - residual_base[0:3]) / ell_step
        return J

    residual, Y_current, _ = evaluate(z)
    norm = residual_norm(residual)
    iteration = 0
    convergence_warning: str | None = None
    last_line_search_trial_norm: float | None = None
    newton_tol = DOWNSTREAM_NEWTON_TOL
    stagnation_tol = max(DOWNSTREAM_NEWTON_STAGNATION_TOL, 100.0 * cfg.ode_rtol, 1.0e-7)
    for iteration in range(DOWNSTREAM_NEWTON_MAX_ITER + 1):
        if norm <= newton_tol:
            break
        if iteration == DOWNSTREAM_NEWTON_MAX_ITER:
            breakdown = _downstream_residual_breakdown(residual, residual_scale, Y_current, float(z[-1]), s_nodes)
            raise RuntimeError(
                "downstream multiple-shooting Newton reached iteration limit: "
                f"mult={cfg.qei_multiplier:.12g}, residual_breakdown={breakdown}"
            )
        J = finite_difference_jacobian(z, residual, Y_current)
        scaled_r = scaled_residual(residual)
        scaled_J = J / residual_scale[:, None]
        try:
            lu, piv = lu_factor(scaled_J)
            delta = lu_solve((lu, piv), -scaled_r)
        except Exception as exc:
            breakdown = _downstream_residual_breakdown(residual, residual_scale, Y_current, float(z[-1]), s_nodes)
            raise RuntimeError(
                "downstream multiple-shooting LU solve failed: "
                f"mult={cfg.qei_multiplier:.12g}, residual_breakdown={breakdown}"
            ) from exc
        if not np.all(np.isfinite(delta)):
            breakdown = _downstream_residual_breakdown(residual, residual_scale, Y_current, float(z[-1]), s_nodes)
            raise RuntimeError(
                "downstream multiple-shooting Newton step is nonfinite: "
                f"mult={cfg.qei_multiplier:.12g}, residual_breakdown={breakdown}"
            )

        accepted = False
        last_trial_norm: float | None = None
        for halving in range(DOWNSTREAM_NEWTON_MAX_HALVINGS + 1):
            damping = 0.5**halving
            z_trial = z + damping * delta
            r_trial = try_residual(z_trial)
            if r_trial is None:
                continue
            trial_norm = residual_norm(r_trial)
            last_trial_norm = trial_norm
            if trial_norm < norm or trial_norm <= newton_tol:
                z = z_trial
                residual = r_trial
                norm = trial_norm
                _, Y_current, _ = evaluate(z)
                accepted = True
                break
        if not accepted:
            if norm <= stagnation_tol:
                convergence_warning = (
                    "line search could not improve scaled residual; accepted current iterate "
                    f"at scaled_residual_inf={norm:.6e}"
                )
                last_line_search_trial_norm = last_trial_norm
                break
            breakdown = _downstream_residual_breakdown(residual, residual_scale, Y_current, float(z[-1]), s_nodes)
            if last_trial_norm is not None:
                breakdown["last_trial_scaled_residual_inf"] = float(last_trial_norm)
            raise RuntimeError(
                "downstream multiple-shooting Newton stagnated during line search: "
                f"mult={cfg.qei_multiplier:.12g}, residual_breakdown={breakdown}"
            )

    residual, Y_final, segments = evaluate(z, dense_output=True)
    norm = residual_norm(residual)
    if norm > newton_tol and convergence_warning is None:
        breakdown = _downstream_residual_breakdown(residual, residual_scale, Y_final, float(z[-1]), s_nodes)
        raise RuntimeError(
            "downstream multiple-shooting Newton failed final residual gate: "
            f"mult={cfg.qei_multiplier:.12g}, residual_breakdown={breakdown}"
        )
    if norm > stagnation_tol:
        breakdown = _downstream_residual_breakdown(residual, residual_scale, Y_final, float(z[-1]), s_nodes)
        raise RuntimeError(
            "downstream multiple-shooting Newton failed final residual gate: "
            f"mult={cfg.qei_multiplier:.12g}, residual_breakdown={breakdown}"
        )
    theta_nodes = inv2.R_i * Y_final[0, :] + inv2.R_e * Y_final[1, :]
    margin = float(np.max(theta_nodes)) / theta_limit
    if margin > 0.999:
        raise RuntimeError(f"downstream shooting solution touches the sonic wedge: max theta/theta_limit={margin:.6f}")
    if float(np.min(Y_final[0:2, :])) <= 100.0 * T_floor:
        raise RuntimeError("downstream shooting solution touches temperature floor")
    far_clean = float(np.max(np.abs((Y_final[:, -1] - y_far) / scale)))
    if far_clean > 1.0e-8:
        raise RuntimeError(f"downstream shooting far cleanliness failed: scaled={far_clean:.6e}")
    ell_up_star = float(z[-1])
    if not (ell_lo < ell_up_star < ell_hi):
        raise RuntimeError(
            "downstream shooting ell_up outside allowed branch interval: "
            f"ell_up={ell_up_star:.16e}, lower={ell_lo:.16e}, upper={ell_hi:.16e}"
        )
    y_minus_star = np.asarray(up_sol.sol(ell_up_star), dtype=float)
    y_jump_star = solve_post_jump_state(y_minus_star, cfg, inv2, warm)
    _, projection_amplitude_removed = downstream_start_state(y_jump_star)
    projection_diagnostics = {
        "lambda_fast_times_h1": lam_fast_h1,
        "lambda_fast_over_slow": lam_fast_over_slow,
        "projection_applied": projection_applied,
        "projection_amplitude_removed": float(projection_amplitude_removed),
    }
    residual_breakdown = _downstream_residual_breakdown(residual, residual_scale, Y_final, ell_up_star, s_nodes)
    if last_line_search_trial_norm is not None:
        residual_breakdown["last_trial_scaled_residual_inf"] = float(last_line_search_trial_norm)
    sol = DownstreamShootingSolution(
        x=s_nodes / L_dn,
        y=Y_final,
        p=np.asarray([ell_up_star], dtype=float),
        status=0,
        message="converged" if convergence_warning is None else f"converged_with_warning: {convergence_warning}",
        L_dn_span=L_dn,
        breakpoints_cm=s_nodes,
        segments=segments,
        tenryu_theta_margin=margin,
        tenryu_newton_iterations=iteration,
        tenryu_final_residual_inf=norm,
        tenryu_final_absolute_residual_inf=float(np.max(np.abs(residual))),
        tenryu_residual_breakdown=residual_breakdown,
        tenryu_projection_diagnostics=projection_diagnostics,
        tenryu_converged_with_warning=convergence_warning is not None,
        tenryu_warning=convergence_warning,
    )
    return sol, ell_up_star, L_dn


def one_t_warm_start(cfg: FLDCED2TConfig, inv2: Invariants2T) -> WarmStart:
    one_cfg = one_t.FLDCEDConfig(
        mach=cfg.mach,
        gamma=cfg.gamma,
        rho0_gcc=cfg.rho0_gcc,
        T0_eV=cfg.T0_eV,
        kappa_R_cm2_g=cfg.kappa_R_cm2_g,
        A_amu=cfg.A_amu,
        zbar=cfg.zbar,
    )
    match = one_t.solve_shock_match(one_cfg, inv2.base)
    u_minus = one_t.velocity_from_T(match.T_upstream_shock, "upstream", inv2.base)
    u_plus = one_t.velocity_from_T(match.T_downstream_shock, "downstream", inv2.base)
    return WarmStart(
        one_cfg=one_cfg,
        match=match,
        up_len_1t=float(match.upstream_length_cm),
        dn_len_1t=float(match.downstream_length_cm),
        T_up_shock_1t=float(match.T_upstream_shock),
        T_dn_shock_1t=float(match.T_downstream_shock),
        rho_ratio_1t=u_minus / u_plus,
    )


def _jump_residuals_from_states(
    y_minus: np.ndarray,
    y_plus: np.ndarray,
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
) -> dict[str, float]:
    T1 = inv2.base.T_down
    Ti_m, Te_m, W_m = [float(v) for v in y_minus]
    Ti_p, Te_p, W_p = [float(v) for v in y_plus]
    theta_m = inv2.R_i * Ti_m + inv2.R_e * Te_m
    theta_p = inv2.R_i * Ti_p + inv2.R_e * Te_p
    u_m = velocity_from_theta(theta_m, "upstream", inv2)
    u_p = velocity_from_theta(theta_p, "downstream", inv2)
    E_m = A_EV * Te_m**4 + W_m
    E_p = A_EV * Te_p**4 + W_p
    F_m = radiation_flux_2t(Ti_m, Te_m, u_m, inv2)
    F_p = radiation_flux_2t(Ti_p, Te_p, u_p, inv2)
    electron_target = Te_m * (u_m / u_p) ** (cfg.gamma - 1.0)
    gas_m = inv2.cp_i * Ti_m + inv2.cp_e * Te_m + 0.5 * u_m * u_m
    gas_p = inv2.cp_i * Ti_p + inv2.cp_e * Te_p + 0.5 * u_p * u_p
    return {
        "electron_adiabat_residual": (Te_p - electron_target) / max(T1, E_FLOOR),
        "gas_flux_continuity_residual": (gas_p - gas_m) / max(inv2.cp_i * T1, E_FLOOR),
        "E_continuity_residual": (E_p - E_m) / max(abs(E_m), A_EV * T1**4, E_FLOOR),
        "F_continuity_residual": (F_p - F_m) / max(abs(F_m), abs(inv2.base.energy_flux), E_FLOOR),
        "E_continuity_abs": E_p - E_m,
        "F_continuity_abs": F_p - F_m,
    }


def solve_post_jump_state(
    y_minus: np.ndarray,
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
    warm: WarmStart,
) -> np.ndarray:
    T1 = inv2.base.T_down
    Ti_m, Te_m, W_m = [float(v) for v in y_minus]
    theta_m = inv2.R_i * Ti_m + inv2.R_e * Te_m
    u_m = velocity_from_theta(theta_m, "upstream", inv2)
    E_m = A_EV * Te_m**4 + W_m
    Te_seed = max(1.0e-9 * T1, Te_m * warm.rho_ratio_1t ** (cfg.gamma - 1.0))
    Ti_seed = max(1.0e-9 * T1, warm.T_dn_shock_1t)

    def residual(z: Any) -> np.ndarray:
        Ti_p = float(z[0])
        Te_p = float(z[1])
        if Ti_p <= 0.0 or Te_p <= 0.0:
            return np.asarray([1.0e6, 1.0e6], dtype=float)
        try:
            theta_p = inv2.R_i * Ti_p + inv2.R_e * Te_p
            u_p = velocity_from_theta(theta_p, "downstream", inv2)
        except (FloatingPointError, ValueError):
            return np.asarray([1.0e6, 1.0e6], dtype=float)
        r1 = (Te_p - Te_m * (u_m / u_p) ** (cfg.gamma - 1.0)) / max(T1, E_FLOOR)
        gas_m = inv2.cp_i * Ti_m + inv2.cp_e * Te_m + 0.5 * u_m * u_m
        gas_p = inv2.cp_i * Ti_p + inv2.cp_e * Te_p + 0.5 * u_p * u_p
        r2 = (gas_p - gas_m) / max(inv2.cp_i * T1, E_FLOOR)
        return np.asarray([r1, r2], dtype=float)

    res = least_squares(
        residual,
        x0=np.asarray([Ti_seed, Te_seed], dtype=float),
        bounds=([1.0e-12 * T1, 1.0e-12 * T1], [np.inf, np.inf]),
        method="trf",
        x_scale=(T1, T1),
        xtol=1.0e-15,
        ftol=1.0e-15,
        gtol=1.0e-15,
        max_nfev=200,
    )
    scaled = residual(res.x)
    if float(np.max(np.abs(scaled))) > JUMP_RESIDUAL_GATE:
        raise RuntimeError(
            "post-jump solve failed: "
            f"y_minus={y_minus.tolist()}, x={res.x.tolist()}, residuals={scaled.tolist()}"
        )
    Ti_p = float(res.x[0])
    Te_p = float(res.x[1])
    return np.asarray([Ti_p, Te_p, E_m - A_EV * Te_p**4], dtype=float)


def _full_integrate_upstream_branch(
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
    seeds: FullEigenSeeds,
    warm: WarmStart,
    F_shock: float,
    *,
    dense: bool = False,
) -> FullBranchSolution:
    branch = "upstream"
    T_far = inv2.base.T0
    T_shock = one_t.temperature_for_flux(F_shock, branch, inv2.base)
    span = float(T_shock - T_far)
    if not span > 0.0:
        raise RuntimeError("full upstream branch has nonpositive temperature span")
    T_start = T_far + max(REDUCED_ENDPOINT_CLIP_FRACTION * span, 1.0e-10)
    y_far = _fixed_point_mdz(branch, inv2)
    alpha = (T_start - T_far) / float(seeds.upstream_vector[0])
    y0 = y_far + alpha * seeds.upstream_vector
    y0[0] = T_start
    scale = _mdz_scale(max(abs(T_far), abs(T_shock)))
    D_max = max(5.0 * warm.up_len_1t, 2.0 / max(abs(seeds.upstream_lambda_grow), E_FLOOR))
    denom_floor = 1.0e-10 * abs(inv2.base.momentum_flux)

    def rhs(D: float, y: Any) -> list[float]:
        del D
        return rhs_full_mdz(0.0, y, branch, cfg, inv2)

    def jac(D: float, y: Any) -> np.ndarray:
        del D
        return jac_full_mdz(0.0, y, branch, cfg, inv2)

    def shock_event(D: float, y: Any) -> float:
        del D
        return float(y[0]) - float(T_shock)

    shock_event.terminal = True  # type: ignore[attr-defined]
    shock_event.direction = 1.0  # type: ignore[attr-defined]

    def sonic_event(D: float, y: Any) -> float:
        del D
        return abs(_full_mdz_coefficients(float(y[0]), branch, inv2)["A"]) - denom_floor

    sonic_event.terminal = True  # type: ignore[attr-defined]
    sonic_event.direction = -1.0  # type: ignore[attr-defined]

    def negT_event(D: float, y: Any) -> float:
        del D
        species = _species_y_from_mdz(y, inv2)
        return min(float(species[0]), float(species[1])) - 1.0e-12 * T_far

    negT_event.terminal = True  # type: ignore[attr-defined]
    negT_event.direction = -1.0  # type: ignore[attr-defined]

    sol = solve_ivp(
        rhs,
        (0.0, D_max),
        y0,
        method="Radau",
        jac=jac,
        rtol=cfg.ode_rtol,
        atol=[cfg.ode_atol_T_eV, cfg.ode_atol_T_eV, _reduced_Z_abs_atol(max(abs(T_far), abs(T_shock)))],
        max_step=max(D_max / 500.0, 1.0e-30),
        dense_output=dense,
        events=[shock_event, sonic_event, negT_event],
    )
    if sol.t_events[1].size > 0 or sol.t_events[2].size > 0:
        raise FloatingPointError("full upstream branch hit sonic/temperature guard before the shock state")
    if not sol.success:
        raise RuntimeError(f"full upstream ODE failed: {sol.message}")
    if sol.t_events[0].size == 0:
        raise RuntimeError(
            f"full upstream branch did not reach shock temperature: "
            f"T_end={float(sol.y[0, -1]):.16e}, T_shock={T_shock:.16e}, D_max={D_max:.16e}"
        )
    length = float(sol.t_events[0][0])
    if not length > 0.0:
        raise RuntimeError("full upstream branch length is nonpositive")
    del scale
    return FullBranchSolution(
        branch=branch,
        F_shock=float(F_shock),
        T_far=float(T_far),
        T_start=float(T_start),
        T_shock=float(T_shock),
        length_cm=length,
        solution=sol,
    )


def _full_integrate_upstream_once(
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
    seeds: FullEigenSeeds,
    warm: WarmStart,
) -> tuple[Any, float]:
    branch = "upstream"
    T_far = inv2.base.T0
    span_guess = max(abs(float(warm.T_up_shock_1t) - T_far), cfg.seed_amplitude_rel * T_far)
    T_start = T_far + max(REDUCED_ENDPOINT_CLIP_FRACTION * span_guess, 1.0e-10)
    y_far = _fixed_point_mdz(branch, inv2)
    if abs(float(seeds.upstream_vector[0])) <= np.finfo(float).tiny:
        raise RuntimeError("full upstream growing eigenvector has zero T component")
    alpha = (T_start - T_far) / float(seeds.upstream_vector[0])
    y0 = y_far + alpha * seeds.upstream_vector
    y0[0] = T_start
    D_max = max(2.5 * warm.up_len_1t, 12.0 / max(abs(seeds.upstream_lambda_grow), E_FLOOR))
    denom_floor = 1.0e-10 * abs(inv2.base.momentum_flux)

    def rhs(D: float, y: Any) -> list[float]:
        del D
        return rhs_full_mdz(0.0, y, branch, cfg, inv2)

    def jac(D: float, y: Any) -> np.ndarray:
        del D
        return jac_full_mdz(0.0, y, branch, cfg, inv2)

    def sonic_event(D: float, y: Any) -> float:
        del D
        return abs(_full_mdz_coefficients(float(y[0]), branch, inv2)["A"]) - denom_floor

    sonic_event.terminal = True  # type: ignore[attr-defined]
    sonic_event.direction = -1.0  # type: ignore[attr-defined]

    def sonic_temperature_event(D: float, y: Any) -> float:
        del D
        return inv2.base.T_sonic * (1.0 - 1.0e-8) - float(y[0])

    sonic_temperature_event.terminal = True  # type: ignore[attr-defined]
    sonic_temperature_event.direction = -1.0  # type: ignore[attr-defined]

    def negT_event(D: float, y: Any) -> float:
        del D
        species = _species_y_from_mdz(y, inv2)
        return min(float(species[0]), float(species[1])) - 1.0e-12 * T_far

    negT_event.terminal = True  # type: ignore[attr-defined]
    negT_event.direction = -1.0  # type: ignore[attr-defined]

    sol = solve_ivp(
        rhs,
        (0.0, D_max),
        y0,
        method="Radau",
        jac=jac,
        rtol=cfg.ode_rtol,
        atol=[cfg.ode_atol_T_eV, cfg.ode_atol_T_eV, _reduced_Z_abs_atol(max(abs(T_far), abs(warm.T_up_shock_1t)))],
        max_step=max(D_max / 500.0, 1.0e-30),
        dense_output=True,
        events=[sonic_event, sonic_temperature_event, negT_event],
    )
    if sol.t_events[2].size > 0:
        raise FloatingPointError("full upstream branch hit temperature guard")
    if not sol.success and sol.t.size <= 1:
        raise RuntimeError(f"full upstream ODE failed immediately: {sol.message}")
    up_reach = float(sol.t[-1])
    if up_reach < 0.5 * warm.up_len_1t:
        raise RuntimeError(f"full upstream integration ended before the shock window: reach={up_reach:.16e}")
    sol.tenryu_method = "full_T_Delta_Z_upstream_radau"
    return sol, up_reach


def _full_downstream_span(seeds: FullEigenSeeds, warm: WarmStart, tau: dict[str, float]) -> float:
    lam_slow = abs(float(seeds.downstream_lambda_slow))
    lam_grow = abs(float(seeds.downstream_lambda_grow))
    slow_span = FULL_DOWNSTREAM_SLOW_EFOLDS / lam_slow if lam_slow > 0.0 else 0.0
    grow_span = FULL_DOWNSTREAM_GROW_EFOLDS / lam_grow if lam_grow > 0.0 else 0.0
    physical = 4.0 * max(float(tau["l_ei_cm"]), float(tau["l_rad_cm"]), E_FLOOR)
    return max(slow_span, grow_span, physical, 2.0 * warm.dn_len_1t)


def _full_downstream_shooting_breakpoints(L_dn: float, seeds: FullEigenSeeds) -> np.ndarray:
    if L_dn <= 0.0:
        raise ValueError("full downstream shooting span must be positive")
    lam_max = max(abs(float(value)) for value in seeds.downstream_eigenvalues)
    if lam_max <= 0.0 or not math.isfinite(lam_max):
        raise ValueError("full downstream fixed-point spectrum has invalid max lambda")
    n_intervals = int(math.ceil(lam_max * L_dn / FULL_DOWNSTREAM_MAX_LAMBDA_H))
    n_intervals = min(max(n_intervals, FULL_DOWNSTREAM_MIN_INTERVALS), FULL_DOWNSTREAM_MAX_INTERVALS)
    nodes = np.linspace(0.0, L_dn, n_intervals + 1)
    if np.any(np.diff(nodes) <= 0.0):
        raise RuntimeError(f"full downstream shooting breakpoints are not strictly increasing: {nodes.tolist()}")
    return nodes


def _integrate_full_downstream_interval(
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
    y0: np.ndarray,
    s0: float,
    s1: float,
    state_scale: np.ndarray,
    *,
    dense_output: bool,
) -> Any:
    branch = "downstream"
    if not np.all(np.isfinite(y0)):
        raise FloatingPointError("nonfinite full downstream shooting state")
    if float(np.min(_species_y_from_mdz(y0, inv2)[0:2])) <= 0.0:
        raise FloatingPointError("nonpositive full downstream shooting species temperature")
    width = float(s1 - s0)
    if width <= 0.0:
        raise ValueError("full downstream shooting interval width must be positive")
    z0 = np.asarray(y0, dtype=float) / state_scale
    denom_floor = max(1.0e-10 * abs(inv2.base.momentum_flux), E_FLOOR)

    def rhs(u: float, z: Any) -> list[float]:
        del u
        y = state_scale * np.asarray(z, dtype=float)
        return (width * np.asarray(rhs_full_mdz(0.0, y, branch, cfg, inv2), dtype=float) / state_scale).tolist()

    def jac(u: float, z: Any) -> np.ndarray:
        del u
        y = state_scale * np.asarray(z, dtype=float)
        return width * (jac_full_mdz(0.0, y, branch, cfg, inv2) * state_scale[None, :]) / state_scale[:, None]

    def sonic_event(u: float, z: Any) -> float:
        del u
        y = state_scale * np.asarray(z, dtype=float)
        return abs(_full_mdz_coefficients(float(y[0]), branch, inv2)["A"]) - denom_floor

    sonic_event.terminal = True  # type: ignore[attr-defined]
    sonic_event.direction = -1.0  # type: ignore[attr-defined]

    def negT_event(u: float, z: Any) -> float:
        del u
        y = state_scale * np.asarray(z, dtype=float)
        species = _species_y_from_mdz(y, inv2)
        return min(float(species[0]), float(species[1])) - 1.0e-12 * T_far

    negT_event.terminal = True  # type: ignore[attr-defined]
    negT_event.direction = -1.0  # type: ignore[attr-defined]

    T_far = inv2.base.T_down
    sol = solve_ivp(
        rhs,
        (0.0, 1.0),
        z0,
        method="Radau",
        jac=jac,
        rtol=cfg.ode_rtol,
        atol=[
            max(cfg.ode_atol_T_eV / state_scale[0], 1.0e-11),
            max(cfg.ode_atol_T_eV / state_scale[1], 1.0e-11),
            max(_reduced_Z_abs_atol(T_far) / state_scale[2], 1.0e-11),
        ],
        max_step=0.25,
        dense_output=dense_output,
        events=[sonic_event, negT_event],
    )
    if sol.status == 1:
        raise FloatingPointError(
            "full downstream shooting interval hit sonic/temperature guard: "
            f"s0={s0:.16e}, s1={s1:.16e}, y0={np.asarray(y0, dtype=float).tolist()}"
        )
    if not sol.success:
        raise RuntimeError(
            "full downstream shooting interval failed: "
            f"message={sol.message!r}, s0={s0:.16e}, s1={s1:.16e}, y0={np.asarray(y0, dtype=float).tolist()}"
        )
    if not np.all(np.isfinite(sol.y[:, -1])):
        raise FloatingPointError("full downstream shooting interval ended nonfinite")
    return DownstreamIntervalSolution(raw=sol, s0=float(s0), s1=float(s1), scale=state_scale)


def solve_full_downstream_multiple_shooting(
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
    seeds: FullEigenSeeds,
    up_sol: Any,
    up_reach: float,
    warm: WarmStart,
    ell_up_guess: float,
    prev: Any = None,
) -> tuple[Any, float, float]:
    tau = tau_scales(cfg, inv2)
    L_dn = _full_downstream_span(seeds, warm, tau)
    y_far = _fixed_point_mdz("downstream", inv2)
    scale = _mdz_scale(inv2.base.T_down)
    continuity_residual_scale = _downstream_continuity_residual_scale(y_far)
    far_projection_weights = seeds.downstream_left_grow_vector / scale
    ell_lo = 0.02 * warm.up_len_1t
    ell_hi = 0.98 * up_reach
    if not ell_lo < ell_hi:
        raise RuntimeError(
            "full downstream shooting ell_up interval is empty: "
            f"lower={ell_lo:.16e}, upper={ell_hi:.16e}, up_reach={up_reach:.16e}"
        )

    s_nodes = _full_downstream_shooting_breakpoints(L_dn, seeds)
    n_nodes = s_nodes.size
    n_interior = n_nodes - 2
    unknown_scale = np.concatenate(
        (
            np.tile(scale, n_interior),
            np.asarray([max(up_reach, warm.up_len_1t, 1.0e-30)]),
        )
    )
    residual_scale = np.concatenate(
        (
            np.tile(continuity_residual_scale, n_interior),
            np.asarray([1.0], dtype=float),
        )
    )

    def downstream_start_state(ell_up: float) -> np.ndarray:
        y_minus_mdz = np.asarray(up_sol.sol(float(ell_up)), dtype=float)
        y_minus = _species_y_from_mdz(y_minus_mdz, inv2)
        return _mdz_from_species_y(solve_post_jump_state(y_minus, cfg, inv2, warm), inv2)

    def pack(Y: np.ndarray, ell_up: float) -> np.ndarray:
        return np.concatenate((Y[:, 1:-1].T.reshape(-1), np.asarray([ell_up], dtype=float)))

    def unpack(z_in: np.ndarray, y_start: np.ndarray) -> tuple[np.ndarray, float]:
        ell_up = float(z_in[-1])
        Y = np.empty((3, n_nodes), dtype=float)
        Y[:, 0] = y_start
        Y[:, -1] = y_far
        if n_interior > 0:
            Y[:, 1:-1] = z_in[:-1].reshape((n_interior, 3)).T
        return Y, ell_up

    ell_up_guess_eval = min(max(float(ell_up_guess), ell_lo), ell_hi)
    y_start0 = downstream_start_state(ell_up_guess_eval)
    y_init = np.empty((3, n_nodes), dtype=float)
    y_init[:, 0] = y_start0
    y_init[:, -1] = y_far
    if prev is not None:
        prev_L = float(getattr(prev, "L_dn_span", L_dn))
        for j, s in enumerate(s_nodes[1:-1], start=1):
            y_init[:, j] = np.asarray(prev.sol(min(float(s), prev_L)), dtype=float)
    else:
        for j, s in enumerate(s_nodes[1:-1], start=1):
            frac = float(s) / L_dn
            y_init[:, j] = (1.0 - frac) * y_start0 + frac * y_far
    for j in range(n_nodes):
        if float(np.min(_species_y_from_mdz(y_init[:, j], inv2)[0:2])) <= 0.0:
            raise RuntimeError("full downstream shooting initial guess produced nonpositive temperature")

    z = pack(y_init, ell_up_guess_eval)

    def interval_residual(Y: np.ndarray, k: int, *, dense_output: bool = False) -> tuple[np.ndarray, Any]:
        seg = _integrate_full_downstream_interval(
            cfg,
            inv2,
            Y[:, k],
            float(s_nodes[k]),
            float(s_nodes[k + 1]),
            scale,
            dense_output=dense_output,
        )
        return np.asarray(seg.y[:, -1], dtype=float) - Y[:, k + 1], seg

    def final_projection(Y: np.ndarray, *, dense_output: bool = False) -> tuple[float, np.ndarray, Any]:
        seg = _integrate_full_downstream_interval(
            cfg,
            inv2,
            Y[:, -2],
            float(s_nodes[-2]),
            float(s_nodes[-1]),
            scale,
            dense_output=dense_output,
        )
        y_end = np.asarray(seg.y[:, -1], dtype=float)
        return float(np.dot(far_projection_weights, y_end - y_far)), y_end, seg

    def evaluate(z_in: np.ndarray, *, dense_output: bool = False) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
        ell_up = float(z_in[-1])
        if not ell_lo < ell_up < ell_hi:
            raise FloatingPointError(
                f"ell_up outside allowed branch interval: ell_up={ell_up:.16e}, "
                f"lower={ell_lo:.16e}, upper={ell_hi:.16e}"
            )
        Y, _ = unpack(z_in, downstream_start_state(ell_up))
        residuals: list[float] = []
        segments: list[Any] = []
        for k in range(n_interior):
            block, seg = interval_residual(Y, k, dense_output=dense_output)
            residuals.extend(block.tolist())
            if dense_output:
                segments.append(seg)
        far, y_end, seg = final_projection(Y, dense_output=dense_output)
        Y[:, -1] = y_end
        residuals.append(far)
        if dense_output:
            segments.append(seg)
        res = np.asarray(residuals, dtype=float)
        if not np.all(np.isfinite(res)):
            raise FloatingPointError("full downstream shooting residual is nonfinite")
        return res, Y, tuple(segments)

    def scaled_residual(residual: np.ndarray) -> np.ndarray:
        return residual / residual_scale

    def residual_norm(residual: np.ndarray) -> float:
        return float(np.max(np.abs(scaled_residual(residual))))

    def try_residual(z_in: np.ndarray) -> np.ndarray | None:
        try:
            return evaluate(z_in)[0]
        except (FloatingPointError, RuntimeError, ValueError):
            return None

    def finite_difference_jacobian(z_base: np.ndarray, residual_base: np.ndarray, Y_base: np.ndarray) -> np.ndarray:
        sqrt_eps = math.sqrt(np.finfo(float).eps)
        J = np.zeros((residual_base.size, z_base.size), dtype=float)
        for node in range(1, n_nodes - 1):
            for component in range(3):
                col = 3 * (node - 1) + component
                step = sqrt_eps * max(abs(float(Y_base[component, node])), float(unknown_scale[col]), 1.0)
                prev_rows = slice(3 * (node - 1), 3 * node)
                J[prev_rows, col] = 0.0
                J[3 * (node - 1) + component, col] = -1.0
                Y_p = np.array(Y_base, dtype=float)
                Y_p[component, node] += step
                if node < n_nodes - 2:
                    block_p, _ = interval_residual(Y_p, node)
                    rows = slice(3 * node, 3 * (node + 1))
                    J[rows, col] = (block_p - residual_base[rows]) / step
                else:
                    far_p, _, _ = final_projection(Y_p)
                    J[-1, col] = (far_p - residual_base[-1]) / step

        ell_col = z_base.size - 1
        ell_step = sqrt_eps * max(abs(float(z_base[-1])), float(unknown_scale[-1]), 1.0e-30)
        ell_step = min(ell_step, 0.25 * (ell_hi - float(z_base[-1])), 0.25 * (float(z_base[-1]) - ell_lo))
        if ell_step <= 0.0 or not math.isfinite(ell_step):
            raise RuntimeError("cannot finite-difference ell_up at the full branch interval boundary")
        Y_p, _ = unpack(z_base, downstream_start_state(float(z_base[-1]) + ell_step))
        block_p, _ = interval_residual(Y_p, 0)
        J[0:3, ell_col] = (block_p - residual_base[0:3]) / ell_step
        return J

    residual, Y_current, _ = evaluate(z)
    norm = residual_norm(residual)
    iteration = 0
    convergence_warning: str | None = None
    last_line_search_trial_norm: float | None = None
    newton_tol = DOWNSTREAM_NEWTON_TOL
    stagnation_tol = max(DOWNSTREAM_NEWTON_STAGNATION_TOL, 100.0 * cfg.ode_rtol, 1.0e-7)
    for iteration in range(DOWNSTREAM_NEWTON_MAX_ITER + 1):
        if norm <= newton_tol:
            break
        if iteration == DOWNSTREAM_NEWTON_MAX_ITER:
            breakdown = _downstream_residual_breakdown(residual, residual_scale, Y_current, float(z[-1]), s_nodes)
            raise RuntimeError(
                "full downstream multiple-shooting Newton reached iteration limit: "
                f"mult={cfg.qei_multiplier:.12g}, residual_breakdown={breakdown}"
            )
        J = finite_difference_jacobian(z, residual, Y_current)
        scaled_r = scaled_residual(residual)
        scaled_J = J / residual_scale[:, None]
        try:
            lu, piv = lu_factor(scaled_J)
            delta = lu_solve((lu, piv), -scaled_r)
        except Exception as exc:
            breakdown = _downstream_residual_breakdown(residual, residual_scale, Y_current, float(z[-1]), s_nodes)
            raise RuntimeError(
                "full downstream multiple-shooting LU solve failed: "
                f"mult={cfg.qei_multiplier:.12g}, residual_breakdown={breakdown}"
            ) from exc
        if not np.all(np.isfinite(delta)):
            breakdown = _downstream_residual_breakdown(residual, residual_scale, Y_current, float(z[-1]), s_nodes)
            raise RuntimeError(
                "full downstream multiple-shooting Newton step is nonfinite: "
                f"mult={cfg.qei_multiplier:.12g}, residual_breakdown={breakdown}"
            )

        accepted = False
        last_trial_norm: float | None = None
        for halving in range(DOWNSTREAM_NEWTON_MAX_HALVINGS + 1):
            damping = 0.5**halving
            z_trial = z + damping * delta
            r_trial = try_residual(z_trial)
            if r_trial is None:
                continue
            trial_norm = residual_norm(r_trial)
            last_trial_norm = trial_norm
            if trial_norm < norm or trial_norm <= newton_tol:
                z = z_trial
                residual = r_trial
                norm = trial_norm
                _, Y_current, _ = evaluate(z)
                accepted = True
                break
        if not accepted:
            if norm <= stagnation_tol:
                convergence_warning = (
                    "line search could not improve scaled residual; accepted current iterate "
                    f"at scaled_residual_inf={norm:.6e}"
                )
                last_line_search_trial_norm = last_trial_norm
                break
            breakdown = _downstream_residual_breakdown(residual, residual_scale, Y_current, float(z[-1]), s_nodes)
            if last_trial_norm is not None:
                breakdown["last_trial_scaled_residual_inf"] = float(last_trial_norm)
            raise RuntimeError(
                "full downstream multiple-shooting Newton stagnated during line search: "
                f"mult={cfg.qei_multiplier:.12g}, residual_breakdown={breakdown}"
            )

    residual, Y_final, segments = evaluate(z, dense_output=True)
    norm = residual_norm(residual)
    if norm > newton_tol and convergence_warning is None:
        breakdown = _downstream_residual_breakdown(residual, residual_scale, Y_final, float(z[-1]), s_nodes)
        raise RuntimeError(
            "full downstream multiple-shooting Newton failed final residual gate: "
            f"mult={cfg.qei_multiplier:.12g}, residual_breakdown={breakdown}"
        )
    if norm > stagnation_tol:
        breakdown = _downstream_residual_breakdown(residual, residual_scale, Y_final, float(z[-1]), s_nodes)
        raise RuntimeError(
            "full downstream multiple-shooting Newton failed final residual gate: "
            f"mult={cfg.qei_multiplier:.12g}, residual_breakdown={breakdown}"
        )
    for j in range(n_nodes):
        if float(np.min(_species_y_from_mdz(Y_final[:, j], inv2)[0:2])) <= 0.0:
            raise RuntimeError("full downstream shooting solution touches temperature floor")
    sonic_margins = [abs(_full_mdz_coefficients(float(Y_final[0, j]), "downstream", inv2)["A"]) for j in range(n_nodes)]
    if min(sonic_margins) <= 1.0e-10 * abs(inv2.base.momentum_flux):
        raise RuntimeError("full downstream shooting solution touches the sonic wedge")
    ell_up_star = float(z[-1])
    residual_breakdown = _downstream_residual_breakdown(residual, residual_scale, Y_final, ell_up_star, s_nodes)
    if last_line_search_trial_norm is not None:
        residual_breakdown["last_trial_scaled_residual_inf"] = float(last_line_search_trial_norm)
    nfev = int(sum(int(getattr(seg.raw, "nfev", 0)) for seg in segments))
    njev = int(sum(int(getattr(seg.raw, "njev", 0)) for seg in segments))
    nlu = int(sum(int(getattr(seg.raw, "nlu", 0)) for seg in segments))
    sol = DownstreamShootingSolution(
        x=s_nodes / L_dn,
        y=Y_final,
        p=np.asarray([ell_up_star], dtype=float),
        status=0,
        message="converged" if convergence_warning is None else f"converged_with_warning: {convergence_warning}",
        L_dn_span=L_dn,
        breakpoints_cm=s_nodes,
        segments=segments,
        tenryu_theta_margin=float(min(sonic_margins) / max(abs(inv2.base.momentum_flux), E_FLOOR)),
        tenryu_newton_iterations=iteration,
        tenryu_final_residual_inf=norm,
        tenryu_final_absolute_residual_inf=float(np.max(np.abs(residual))),
        tenryu_residual_breakdown=residual_breakdown,
        tenryu_projection_diagnostics={
            "lambda_max_times_h_max": float(max(abs(v) for v in seeds.downstream_eigenvalues) * np.max(np.diff(s_nodes))),
            "n_intervals": int(s_nodes.size - 1),
            "projection_applied": False,
            "projection_amplitude_removed": 0.0,
        },
        tenryu_converged_with_warning=convergence_warning is not None,
        tenryu_warning=convergence_warning,
        tenryu_method="full_T_Delta_Z_radau_multiple_shooting",
        tenryu_sol_input_physical=True,
        nfev=nfev,
        njev=njev,
        nlu=nlu,
    )
    return sol, ell_up_star, L_dn


def _full_downstream_far_residual(branch: FullBranchSolution, seeds: FullEigenSeeds, inv2: Invariants2T) -> float:
    y_far = _fixed_point_mdz("downstream", inv2)
    scale = _mdz_scale(inv2.base.T_down)
    y_end = np.asarray(branch.solution.y[:, -1], dtype=float)
    return float(np.dot(seeds.downstream_left_grow_vector, (y_end - y_far) / scale))


def _full_solution_stats(solution: FullShockSolution2T) -> dict[str, Any]:
    cfg = solution.cfg
    inv2 = solution.inv2
    deltas: list[float] = []
    sonic_margins: list[float] = []
    for branch_sol in (solution.up_branch, solution.dn_branch):
        samples = np.linspace(0.0, branch_sol.length_cm, 512)
        for value in samples:
            y = np.asarray(branch_sol.solution.sol(float(value)), dtype=float)
            deltas.append(float(y[1]))
            sonic_margins.append(abs(_full_mdz_coefficients(float(y[0]), branch_sol.branch, inv2)["A"]))
    tau = tau_scales(cfg, inv2)
    max_delta = float(np.max(np.abs(np.asarray(deltas, dtype=float)))) if deltas else 0.0
    return {
        "method": "full_T_Delta_Z_radau_multiple_shooting",
        "branch": "full",
        "eps_l_ei_over_l_rad": float(tau["tau_ratio"]),
        "tau_exchange_downstream_s": float(_tau_eq(tau["rho1_gcc"], tau["T1_eV"], cfg) / cfg.qei_multiplier),
        "max_abs_Delta_eV": max_delta,
        "max_abs_delta_ei": float(max_delta / max(tau["T1_eV"], E_FLOOR)),
        "min_abs_sonic_A": float(np.min(np.asarray(sonic_margins, dtype=float))) if sonic_margins else math.inf,
        "upstream_nfev": int(getattr(solution.up_branch.solution, "nfev", -1)),
        "upstream_njev": int(getattr(solution.up_branch.solution, "njev", -1)),
        "upstream_nlu": int(getattr(solution.up_branch.solution, "nlu", -1)),
        "downstream_nfev": int(getattr(solution.dn_branch.solution, "nfev", -1)),
        "downstream_njev": int(getattr(solution.dn_branch.solution, "njev", -1)),
        "downstream_nlu": int(getattr(solution.dn_branch.solution, "nlu", -1)),
    }


def _solve_one_multiplier_full_mdz(
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
    warm: WarmStart,
    ell_up_guess: float | None = None,
    prev_dn: Any = None,
) -> FullShockSolution2T:
    seeds = _full_eigen_seed(cfg, inv2)
    up_sol, up_reach = _full_integrate_upstream_once(cfg, inv2, seeds, warm)
    bvp_sol, ell_up, L_dn = solve_full_downstream_multiple_shooting(
        cfg,
        inv2,
        seeds,
        up_sol,
        up_reach,
        warm,
        warm.up_len_1t if ell_up_guess is None else ell_up_guess,
        prev=prev_dn,
    )
    y_minus_mdz = np.asarray(up_sol.sol(ell_up), dtype=float)
    y_minus = _species_y_from_mdz(y_minus_mdz, inv2)
    y_raw = solve_post_jump_state(y_minus, cfg, inv2, warm)
    y_raw_mdz = _mdz_from_species_y(y_raw, inv2)
    jump_diag = _jump_residuals_from_states(y_minus, y_raw, cfg, inv2)
    F_shock = _state_payload_from_y(y_minus, "upstream", inv2)["F_rad"]
    up = FullBranchSolution(
        branch="upstream",
        F_shock=float(F_shock),
        T_far=float(inv2.base.T0),
        T_start=float(up_sol.y[0, 0]),
        T_shock=float(y_minus_mdz[0]),
        length_cm=float(ell_up),
        solution=up_sol,
    )
    dn = FullBranchSolution(
        branch="downstream",
        F_shock=float(F_shock),
        T_far=float(inv2.base.T_down),
        T_start=float(y_raw_mdz[0]),
        T_shock=float(y_raw_mdz[0]),
        length_cm=float(L_dn),
        solution=bvp_sol,
    )
    scale_dn = _mdz_scale(inv2.base.T_down)
    y_far_dn = _fixed_point_mdz("downstream", inv2)
    r_jump = (bvp_sol.y[:, 0] - y_raw_mdz) / scale_dn
    r_far = float(np.dot(seeds.downstream_left_grow_vector, (bvp_sol.y[:, -1] - y_far_dn) / scale_dn))
    provisional = FullShockSolution2T(
        cfg=cfg,
        inv2=inv2,
        warm=warm,
        seeds=seeds,
        ell_up=float(ell_up),
        up_branch=up,
        dn_branch=dn,
        L_dn_span=float(dn.length_cm),
        y_minus=y_minus,
        y_plus=y_raw,
        jump_diagnostics=jump_diag,
        bc_residuals=[float(r_jump[0]), float(r_jump[1]), float(r_jump[2]), r_far],
        full_diagnostics={},
    )
    stats = _full_solution_stats(provisional)
    stats["far_projection_residual"] = r_far
    stats["shooting_iterations"] = int(getattr(bvp_sol, "tenryu_newton_iterations"))
    stats["shooting_function_calls"] = int(getattr(bvp_sol, "nfev", -1))
    stats["shooting_breakpoints"] = [float(v) for v in getattr(bvp_sol, "breakpoints_cm")]
    stats["shooting_residual_breakdown"] = getattr(bvp_sol, "tenryu_residual_breakdown")
    return dataclasses.replace(provisional, full_diagnostics=stats)


def _solve_one_multiplier_reduced(
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
    warm: WarmStart,
) -> ReducedShockSolution:
    tau = tau_scales(cfg, inv2)
    eps = float(tau["tau_ratio"])
    if eps >= REDUCED_SOLVER_EPS_MAX:
        raise RuntimeError(
            "reduced outer 2T solver requires eps=l_ei/l_rad < "
            f"{REDUCED_SOLVER_EPS_MAX:.1e}; got {eps:.6e} at qei_multiplier={cfg.qei_multiplier:.12g}. "
            "Use --solver full only as the experimental diagnostic branch for this parameter set."
        )
    F_min_up = one_t.radiation_flux_energy(inv2.base.T_sonic * (1.0 - 1.0e-8), "upstream", inv2.base)
    F_min_dn = one_t.radiation_flux_energy(inv2.base.T_sonic * (1.0 - 1.0e-8), "downstream", inv2.base)
    F_min_common = max(F_min_up, F_min_dn)
    if not F_min_common < 0.0:
        raise RuntimeError("reduced common shock-flux bracket is not negative")

    samples = [F_min_common * frac for frac in np.geomspace(1.0e-3, 0.75, 48)]
    bracket: tuple[float, float] | None = None
    previous: tuple[float, float] | None = None
    for F_value in samples:
        try:
            mismatch = _reduced_shock_parts(cfg, inv2, warm, float(F_value))[0]
        except (RuntimeError, FloatingPointError, ValueError):
            previous = None
            continue
        if previous is not None and previous[1] * mismatch < 0.0:
            bracket = (previous[0], float(F_value))
            break
        previous = (float(F_value), mismatch)
    if bracket is None:
        raise RuntimeError("reduced solver failed to bracket continuous outer E_rad shock match")

    F_shock = float(
        brentq(
            lambda F: _reduced_shock_parts(cfg, inv2, warm, float(F))[0],
            bracket[0],
            bracket[1],
            xtol=max(abs(F_min_common) * cfg.match_xtol, 1.0e6),
            rtol=cfg.match_xtol,
            maxiter=100,
        )
    )
    mismatch, up, dn, y_minus, y_raw, y_plus_outer = _reduced_shock_parts(
        cfg,
        inv2,
        warm,
        F_shock,
        dense=True,
    )
    jump_diag = _jump_residuals_from_states(y_minus, y_raw, cfg, inv2)
    E_scale = max(abs(A_EV * float(y_raw[1]) ** 4 + float(y_raw[2])), A_EV * inv2.base.T_down**4, E_FLOOR)
    provisional = ReducedShockSolution(
        cfg=cfg,
        inv2=inv2,
        warm=warm,
        ell_up=float(up.length_cm),
        up_branch=up,
        dn_branch=dn,
        L_dn_span=float(dn.length_cm),
        y_minus=y_minus,
        y_plus=y_plus_outer,
        y_plus_raw=y_raw,
        jump_diagnostics=jump_diag,
        bc_residuals=[float(mismatch / E_scale)],
        reduced_diagnostics={},
    )
    stats = _reduced_solution_stats(provisional)
    stats["outer_match_Z_residual_erg_per_cm3"] = float(mismatch)
    stats["outer_match_scaled_residual"] = float(mismatch / E_scale)
    return dataclasses.replace(provisional, reduced_diagnostics=stats)


def solve_shock_match_reduced_2t(
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
    *,
    cache_mults: set[float] | None = None,
) -> tuple[ReducedShockSolution, dict[float, ReducedShockSolution]]:
    warm = one_t_warm_start(cfg, inv2)
    cache: dict[float, ReducedShockSolution] = {}
    ladder_diagnostics: list[dict[str, Any]] = []
    final_solution: ReducedShockSolution | None = None
    if cache_mults is None:
        cache_mults = set()
    for mult in _continuation_ladder(cfg):
        cfg_m = dataclasses.replace(cfg, qei_multiplier=mult)
        solution = _solve_one_multiplier_reduced(cfg_m, inv2, warm)
        diag = dict(solution.reduced_diagnostics)
        diag.update(
            {
                "qei_multiplier": float(mult),
                "solver": "reduced",
                "ell_up_cm": float(solution.ell_up),
                "ell_dn_cm": float(solution.L_dn_span),
                "F_shock_erg_per_cm2_s": float(solution.up_branch.F_shock),
            }
        )
        ladder_diagnostics.append(diag)
        cache[float(mult)] = solution
        if any(math.isclose(mult, want, rel_tol=0.0, abs_tol=max(abs(want), 1.0) * 1.0e-14) for want in cache_mults):
            cache[float(mult)] = solution
        if math.isclose(mult, cfg.qei_multiplier, rel_tol=0.0, abs_tol=max(abs(cfg.qei_multiplier), 1.0) * 1.0e-14):
            final_solution = solution
    if final_solution is None:
        raise RuntimeError("reduced continuation did not produce the requested multiplier")
    final_solution = dataclasses.replace(final_solution, continuation_diagnostics=tuple(ladder_diagnostics))
    cache[float(final_solution.cfg.qei_multiplier)] = final_solution
    return final_solution, cache


def _solve_one_multiplier(
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
    warm: WarmStart,
    ell_up_guess: float,
    prev_bvp: Any = None,
) -> ShockSolution2T:
    seeds = eigen_seed(cfg, inv2)
    X_max = 2.5 * warm.up_len_1t
    up_sol, up_reach = integrate_upstream_once(cfg, inv2, seeds, X_max)
    bvp_sol, ell_up, L_dn = solve_downstream_multiple_shooting(
        cfg,
        inv2,
        seeds,
        up_sol,
        up_reach,
        warm,
        ell_up_guess,
        prev=prev_bvp,
    )
    y_minus = np.asarray(up_sol.sol(ell_up), dtype=float)
    y_plus = solve_post_jump_state(y_minus, cfg, inv2, warm)
    jump_diag = _jump_residuals_from_states(y_minus, y_plus, cfg, inv2)
    scale = _downstream_state_scale(inv2)
    y_far = fixed_point("downstream", inv2)
    projection_diag = getattr(bvp_sol, "tenryu_projection_diagnostics", {})
    if projection_diag.get("projection_applied", False):
        y_start, _ = _downstream_slow_manifold_projection(
            y_plus,
            y_far,
            scale,
            seeds.downstream_fast_vector,
            seeds.downstream_left_fast_vector,
        )
    else:
        y_start = y_plus
    r_jump = (bvp_sol.y[:, 0] - y_start) / scale
    r_far = float(np.dot(seeds.downstream_left_grow_vector, (bvp_sol.y[:, -1] - y_far) / scale))
    bc_residuals = [float(r_jump[0]), float(r_jump[1]), float(r_jump[2]), r_far]
    return ShockSolution2T(
        cfg=cfg,
        inv2=inv2,
        warm=warm,
        seeds=seeds,
        ell_up=ell_up,
        up_sol=up_sol,
        bvp_sol=bvp_sol,
        L_dn_span=L_dn,
        y_minus=y_minus,
        y_plus=y_plus,
        jump_diagnostics=jump_diag,
        bc_residuals=bc_residuals,
    )


def _continuation_ladder(cfg: FLDCED2TConfig) -> list[float]:
    if cfg.qei_multiplier >= 1.0e6:
        return [cfg.qei_multiplier]
    ladder = [float(m) for m in cfg.continuation_ladder if m > cfg.qei_multiplier]
    ladder.append(float(cfg.qei_multiplier))
    out: list[float] = []
    for mult in ladder:
        if not out or mult != out[-1]:
            out.append(mult)
    return out


def _full_multiplier_ladder(cfg: FLDCED2TConfig, cache_mults: set[float]) -> list[float]:
    ladder = [float(cfg.qei_multiplier)]
    for mult in sorted(cache_mults):
        if math.isclose(mult, cfg.qei_multiplier, rel_tol=0.0, abs_tol=max(abs(cfg.qei_multiplier), 1.0) * 1.0e-14):
            continue
        ladder.append(float(mult))
    return ladder


def solve_shock_match_2t(
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
    *,
    cache_mults: set[float] | None = None,
) -> tuple[ShockSolution2T, dict[float, ShockSolution2T]]:
    """Compatibility wrapper for the retired full ``(T_i,T_e,W_e)`` branch."""
    validate_config(cfg)
    warm = one_t_warm_start(cfg, inv2)
    ell_up_guess = warm.up_len_1t
    prev_bvp: Any = None
    cache: dict[float, ShockSolution2T] = {}
    if cache_mults is None:
        cache_mults = set()
    final_solution: ShockSolution2T | None = None
    ladder_diagnostics: list[dict[str, Any]] = []
    for mult in _continuation_ladder(cfg):
        cfg_m = dataclasses.replace(cfg, qei_multiplier=mult)
        solution = _solve_one_multiplier(cfg_m, inv2, warm, ell_up_guess, prev_bvp=prev_bvp)
        ell_up_guess = solution.ell_up
        prev_bvp = solution.bvp_sol
        projection_diag = getattr(solution.bvp_sol, "tenryu_projection_diagnostics")
        ladder_diagnostics.append(
            {
                "qei_multiplier": float(mult),
                "newton_iterations": int(getattr(solution.bvp_sol, "tenryu_newton_iterations")),
                "final_residual_inf": float(getattr(solution.bvp_sol, "tenryu_final_residual_inf")),
                "final_scaled_residual_inf": float(getattr(solution.bvp_sol, "tenryu_final_residual_inf")),
                "final_absolute_residual_inf": float(getattr(solution.bvp_sol, "tenryu_final_absolute_residual_inf")),
                "converged_with_warning": bool(getattr(solution.bvp_sol, "tenryu_converged_with_warning", False)),
                "warning": getattr(solution.bvp_sol, "tenryu_warning", None),
                "ell_up_cm": float(solution.ell_up),
                "L_dn_span_cm": float(solution.L_dn_span),
                "n_breakpoints": int(solution.bvp_sol.x.size),
                "lambda_fast_times_h1": float(projection_diag["lambda_fast_times_h1"]),
                "lambda_fast_over_slow": float(projection_diag["lambda_fast_over_slow"]),
                "projection_applied": bool(projection_diag["projection_applied"]),
                "projection_amplitude_removed": float(projection_diag["projection_amplitude_removed"]),
            }
        )
        cache[float(mult)] = solution
        if any(math.isclose(mult, want, rel_tol=0.0, abs_tol=max(abs(want), 1.0) * 1.0e-14) for want in cache_mults):
            cache[float(mult)] = solution
        if math.isclose(mult, cfg.qei_multiplier, rel_tol=0.0, abs_tol=max(abs(cfg.qei_multiplier), 1.0) * 1.0e-14):
            final_solution = solution
    if final_solution is None:
        raise RuntimeError("continuation did not produce the requested multiplier")
    final_solution = dataclasses.replace(final_solution, continuation_diagnostics=tuple(ladder_diagnostics))
    cache[float(final_solution.cfg.qei_multiplier)] = final_solution
    return final_solution, cache


def _select_solver_mode(cfg: FLDCED2TConfig, inv2: Invariants2T) -> str:
    if cfg.solver == "auto":
        eps = float(tau_scales(cfg, inv2)["tau_ratio"])
        return "reduced" if eps < REDUCED_SOLVER_EPS_MAX else "full"
    return cfg.solver


def solve_shock_match_full_mdz_2t(
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
    *,
    cache_mults: set[float] | None = None,
) -> tuple[FullShockSolution2T, dict[float, FullShockSolution2T]]:
    validate_config(cfg)
    warm = one_t_warm_start(cfg, inv2)
    cache: dict[float, FullShockSolution2T] = {}
    if cache_mults is None:
        cache_mults = set()
    final_solution: FullShockSolution2T | None = None
    ladder_diagnostics: list[dict[str, Any]] = []
    ell_up_guess = warm.up_len_1t
    prev_dn: Any = None
    for mult in _full_multiplier_ladder(cfg, cache_mults):
        cfg_m = dataclasses.replace(cfg, qei_multiplier=mult, solver="full")
        solution = _solve_one_multiplier_full_mdz(cfg_m, inv2, warm, ell_up_guess=ell_up_guess, prev_dn=prev_dn)
        ell_up_guess = solution.ell_up
        prev_dn = solution.dn_branch.solution
        diag = dict(solution.full_diagnostics)
        diag.update(
            {
                "qei_multiplier": float(mult),
                "solver": "full",
                "ell_up_cm": float(solution.ell_up),
                "ell_dn_cm": float(solution.L_dn_span),
                "F_shock_erg_per_cm2_s": float(solution.up_branch.F_shock),
            }
        )
        ladder_diagnostics.append(diag)
        cache[float(mult)] = solution
        if any(math.isclose(mult, want, rel_tol=0.0, abs_tol=max(abs(want), 1.0) * 1.0e-14) for want in cache_mults):
            cache[float(mult)] = solution
        if math.isclose(mult, cfg.qei_multiplier, rel_tol=0.0, abs_tol=max(abs(cfg.qei_multiplier), 1.0) * 1.0e-14):
            final_solution = solution
    if final_solution is None:
        raise RuntimeError("full continuation did not produce the requested multiplier")
    final_solution = dataclasses.replace(final_solution, continuation_diagnostics=tuple(ladder_diagnostics))
    cache[float(final_solution.cfg.qei_multiplier)] = final_solution
    return final_solution, cache


def solve_shock_match_auto_2t(
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
    *,
    cache_mults: set[float] | None = None,
) -> tuple[Any, dict[float, Any]]:
    validate_config(cfg)
    warm = one_t_warm_start(cfg, inv2)
    cache: dict[float, Any] = {}
    if cache_mults is None:
        cache_mults = set()
    final_solution: Any | None = None
    ladder_diagnostics: list[dict[str, Any]] = []
    full_ell_up_guess = warm.up_len_1t
    full_prev_dn: Any = None
    target_mode = _select_solver_mode(cfg, inv2)
    multiplier_ladder = _full_multiplier_ladder(cfg, cache_mults) if target_mode == "full" else _continuation_ladder(cfg)
    for mult in multiplier_ladder:
        cfg_m_requested = dataclasses.replace(cfg, qei_multiplier=mult)
        mode = _select_solver_mode(cfg_m_requested, inv2)
        cfg_m = dataclasses.replace(cfg_m_requested, solver=mode)
        if mode == "reduced":
            solution = _solve_one_multiplier_reduced(cfg_m, inv2, warm)
            diag = dict(solution.reduced_diagnostics)
        elif mode == "full":
            solution = _solve_one_multiplier_full_mdz(
                cfg_m,
                inv2,
                warm,
                ell_up_guess=full_ell_up_guess,
                prev_dn=full_prev_dn,
            )
            full_ell_up_guess = solution.ell_up
            full_prev_dn = solution.dn_branch.solution
            diag = dict(solution.full_diagnostics)
        else:
            raise RuntimeError(f"unknown selected solver mode {mode!r}")
        diag.update(
            {
                "qei_multiplier": float(mult),
                "solver": mode,
                "branch": mode,
                "ell_up_cm": float(solution.ell_up),
                "ell_dn_cm": float(solution.L_dn_span),
                "F_shock_erg_per_cm2_s": float(solution.up_branch.F_shock),
            }
        )
        ladder_diagnostics.append(diag)
        cache[float(mult)] = solution
        if any(math.isclose(mult, want, rel_tol=0.0, abs_tol=max(abs(want), 1.0) * 1.0e-14) for want in cache_mults):
            cache[float(mult)] = solution
        if math.isclose(mult, cfg.qei_multiplier, rel_tol=0.0, abs_tol=max(abs(cfg.qei_multiplier), 1.0) * 1.0e-14):
            final_solution = solution
    if final_solution is None:
        raise RuntimeError("auto continuation did not produce the requested multiplier")
    final_solution = dataclasses.replace(final_solution, continuation_diagnostics=tuple(ladder_diagnostics))
    cache[float(final_solution.cfg.qei_multiplier)] = final_solution
    return final_solution, cache


def _state_fields_from_y(y: Any, branch: str, inv2: Invariants2T) -> dict[str, float]:
    Ti = float(y[0])
    Te = float(y[1])
    W = float(y[2])
    E = A_EV * Te**4 + W
    theta = inv2.R_i * Ti + inv2.R_e * Te
    u = velocity_from_theta(theta, branch, inv2)
    rho = inv2.base.mass_flux / u
    p_i = rho * inv2.R_i * Ti
    p_e = rho * inv2.R_e * Te
    F = radiation_flux_2t(Ti, Te, u, inv2)
    return {
        "rho": rho,
        "u": u,
        "T_e": Te,
        "T_i": Ti,
        "T_rad": (max(E, 0.0) / A_EV) ** 0.25,
        "E_rad": E,
        "P_e": p_e,
        "P_i": p_i,
        "P_mat": p_e + p_i,
        "P_rad": E / 3.0,
        "F_rad": F,
        "F_rad_lab": F,
        "delta_ei": Ti / Te - 1.0,
    }


def _state_fields_from_reduced(T: float, Z: float, branch: str, cfg: FLDCED2TConfig, inv2: Invariants2T) -> dict[str, float]:
    y = _full_y_from_reduced(T, Z, branch, cfg, inv2)
    state = _state_fields_from_y(y, branch, inv2)
    state["T_mix"] = T
    state["Z_rad"] = Z
    state["Delta_s"] = float(y[1] - y[0])
    state["W_e"] = float(y[2])
    return state


def _state_fields_from_mdz(y_mdz: Any, branch: str, inv2: Invariants2T) -> dict[str, float]:
    y = _species_y_from_mdz(y_mdz, inv2)
    state = _state_fields_from_y(y, branch, inv2)
    state["T_mix"] = float(y_mdz[0])
    state["Delta"] = float(y_mdz[1])
    state["Z_rad"] = float(y_mdz[2])
    state["W_e"] = float(y[2])
    return state


def _row_from_y(x_cm: float, y: Any, branch: str, inv2: Invariants2T) -> dict[str, Any]:
    state = _state_fields_from_y(y, branch, inv2)
    return {
        "x_cm": x_cm,
        "rho_g_per_cc": state["rho"],
        "u_cm_per_s": state["u"],
        "Te_eV": state["T_e"],
        "Ti_eV": state["T_i"],
        "T_rad_eV": state["T_rad"],
        "E_rad_erg_per_cm3": state["E_rad"],
        "P_e_dyne_per_cm2": state["P_e"],
        "P_i_dyne_per_cm2": state["P_i"],
        "P_mat_dyne_per_cm2": state["P_mat"],
        "P_rad_dyne_per_cm2": state["P_rad"],
        "F_rad_erg_per_cm2_s": state["F_rad"],
        "F_rad_energy_invariant_erg_per_cm2_s": state["F_rad"],
        "F_rad_lab_erg_per_cm2_s": state["F_rad"],
        "delta_ei": state["delta_ei"],
        "branch": branch,
    }


def _row_from_reduced(
    x_cm: float,
    T: float,
    Z: float,
    branch: str,
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
) -> dict[str, Any]:
    state = _state_fields_from_reduced(T, Z, branch, cfg, inv2)
    return {
        "x_cm": x_cm,
        "rho_g_per_cc": state["rho"],
        "u_cm_per_s": state["u"],
        "Te_eV": state["T_e"],
        "Ti_eV": state["T_i"],
        "T_rad_eV": state["T_rad"],
        "E_rad_erg_per_cm3": state["E_rad"],
        "P_e_dyne_per_cm2": state["P_e"],
        "P_i_dyne_per_cm2": state["P_i"],
        "P_mat_dyne_per_cm2": state["P_mat"],
        "P_rad_dyne_per_cm2": state["P_rad"],
        "F_rad_erg_per_cm2_s": state["F_rad"],
        "F_rad_energy_invariant_erg_per_cm2_s": state["F_rad"],
        "F_rad_lab_erg_per_cm2_s": state["F_rad"],
        "delta_ei": state["delta_ei"],
        "branch": branch,
    }


def _row_from_mdz(x_cm: float, y_mdz: Any, branch: str, inv2: Invariants2T) -> dict[str, Any]:
    state = _state_fields_from_mdz(y_mdz, branch, inv2)
    return {
        "x_cm": x_cm,
        "rho_g_per_cc": state["rho"],
        "u_cm_per_s": state["u"],
        "Te_eV": state["T_e"],
        "Ti_eV": state["T_i"],
        "T_rad_eV": state["T_rad"],
        "E_rad_erg_per_cm3": state["E_rad"],
        "P_e_dyne_per_cm2": state["P_e"],
        "P_i_dyne_per_cm2": state["P_i"],
        "P_mat_dyne_per_cm2": state["P_mat"],
        "P_rad_dyne_per_cm2": state["P_rad"],
        "F_rad_erg_per_cm2_s": state["F_rad"],
        "F_rad_energy_invariant_erg_per_cm2_s": state["F_rad"],
        "F_rad_lab_erg_per_cm2_s": state["F_rad"],
        "delta_ei": state["delta_ei"],
        "branch": branch,
    }


def _equilibrium_y(T: float) -> np.ndarray:
    return np.asarray([T, T, 0.0], dtype=float)


def equilibrium_row_2t(x_cm: float, state: dict[str, float], branch: str) -> dict[str, Any]:
    Te = float(state["T_e"])
    Ti = float(state["T_i"])
    E = A_EV * Te**4
    return {
        "x_cm": x_cm,
        "rho_g_per_cc": state["rho"],
        "u_cm_per_s": state["u"],
        "Te_eV": Te,
        "Ti_eV": Ti,
        "T_rad_eV": Te,
        "E_rad_erg_per_cm3": E,
        "P_e_dyne_per_cm2": state["P_e"],
        "P_i_dyne_per_cm2": state["P_i"],
        "P_mat_dyne_per_cm2": state["P_mat"],
        "P_rad_dyne_per_cm2": E / 3.0,
        "F_rad_erg_per_cm2_s": 0.0,
        "F_rad_energy_invariant_erg_per_cm2_s": 0.0,
        "F_rad_lab_erg_per_cm2_s": 0.0,
        "delta_ei": 0.0,
        "branch": branch,
    }


def _table_from_rows(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    fields = (
        "x_cm",
        "rho_g_per_cc",
        "u_cm_per_s",
        "Te_eV",
        "Ti_eV",
        "T_rad_eV",
        "E_rad_erg_per_cm3",
        "P_e_dyne_per_cm2",
        "P_i_dyne_per_cm2",
        "P_mat_dyne_per_cm2",
        "P_rad_dyne_per_cm2",
        "F_rad_erg_per_cm2_s",
        "F_rad_energy_invariant_erg_per_cm2_s",
        "F_rad_lab_erg_per_cm2_s",
        "delta_ei",
        "branch",
    )
    return {field: [row[field] for row in rows] for field in fields}


def _downstream_table_span(solution: ShockSolution2T) -> float:
    if hasattr(solution, "dn_branch"):
        return float(solution.L_dn_span)
    lam_slow = abs(solution.seeds.downstream_lambda_slow)
    return min(-math.log(solution.cfg.seed_amplitude_rel) / lam_slow, solution.L_dn_span)


def _sample_reduced_branch_rows(
    solution: ReducedShockSolution,
    branch_sol: ReducedBranchSolution,
    n_samples: int,
) -> list[dict[str, Any]]:
    cfg = solution.cfg
    inv2 = solution.inv2
    span = branch_sol.T_shock - branch_sol.T_far
    T_last = branch_sol.T_shock - cfg.shock_clip_fraction * span
    if T_last <= branch_sol.T_start:
        T_last = branch_sol.T_shock
    rows: list[dict[str, Any]] = []
    if branch_sol.branch == "upstream":
        rows.append(_row_from_y(-branch_sol.length_cm, _equilibrium_y(inv2.base.T0), "upstream", inv2))
    del T_last
    sample_values = one_t.linspace(0.0, branch_sol.length_cm * (1.0 - cfg.shock_clip_fraction), n_samples)
    for distance_from_far in sample_values[1:]:
        T, Z = branch_sol.solution.sol(float(distance_from_far))
        if branch_sol.branch == "upstream":
            x = float(distance_from_far) - branch_sol.length_cm
        else:
            x = branch_sol.length_cm - float(distance_from_far)
        rows.append(_row_from_reduced(x, float(T), float(Z), branch_sol.branch, cfg, inv2))
    if branch_sol.branch == "downstream":
        rows.append(_row_from_y(branch_sol.length_cm, _equilibrium_y(inv2.base.T_down), "downstream", inv2))
    return rows


def _sample_full_branch_rows(
    solution: FullShockSolution2T,
    branch_sol: FullBranchSolution,
    n_samples: int,
) -> list[dict[str, Any]]:
    cfg = solution.cfg
    inv2 = solution.inv2
    rows: list[dict[str, Any]] = []
    if branch_sol.branch == "upstream":
        rows.append(_row_from_y(-branch_sol.length_cm, _equilibrium_y(inv2.base.T0), "upstream", inv2))
        sample_values = one_t.linspace(0.0, branch_sol.length_cm * (1.0 - cfg.shock_clip_fraction), n_samples)
        for distance_from_far in sample_values[1:]:
            x = float(distance_from_far) - branch_sol.length_cm
            rows.append(_row_from_mdz(x, branch_sol.solution.sol(float(distance_from_far)), "upstream", inv2))
        return rows

    lam_fast = abs(solution.seeds.downstream_lambda_fast)
    d_min = max(cfg.shock_clip_fraction / max(lam_fast, E_FLOOR), 1.0e-12 * branch_sol.length_cm)
    d_max = branch_sol.length_cm * (1.0 - 1.0e-12)
    if d_min >= d_max:
        d_min = max(1.0e-12 * branch_sol.length_cm, np.nextafter(0.0, 1.0))
    distances = np.geomspace(d_min, d_max, n_samples - 1)
    for distance in distances:
        rows.append(_row_from_mdz(float(distance), branch_sol.solution.sol(float(distance)), "downstream", inv2))
    rows.append(_row_from_y(branch_sol.length_cm, _equilibrium_y(inv2.base.T_down), "downstream", inv2))
    return rows


def build_unpadded_table_2t(solution: Any) -> dict[str, list[Any]]:
    cfg = solution.cfg
    inv2 = solution.inv2
    if isinstance(solution, ReducedShockSolution):
        n_up = cfg.n_points // 2
        n_dn = cfg.n_points - n_up
        if n_up < 2 or n_dn < 2:
            raise RuntimeError("n_points must leave at least two samples per branch")
        rows = _sample_reduced_branch_rows(solution, solution.up_branch, n_up)
        rows.extend(_sample_reduced_branch_rows(solution, solution.dn_branch, n_dn))
        rows.sort(key=lambda row: float(row["x_cm"]))
        return _table_from_rows(rows)

    if isinstance(solution, FullShockSolution2T):
        n_up = cfg.n_points // 2
        n_dn = cfg.n_points - n_up
        if n_up < 2 or n_dn < 2:
            raise RuntimeError("n_points must leave at least two samples per branch")
        rows = _sample_full_branch_rows(solution, solution.up_branch, n_up)
        rows.extend(_sample_full_branch_rows(solution, solution.dn_branch, n_dn))
        rows.sort(key=lambda row: float(row["x_cm"]))
        return _table_from_rows(rows)

    ell_up = float(solution.ell_up)
    ell_dn = _downstream_table_span(solution)
    lam_fast = abs(solution.seeds.downstream_lambda_fast)
    n_up = cfg.n_points // 2
    n_dn = cfg.n_points - n_up
    rows: list[dict[str, Any]] = []

    if n_up < 2 or n_dn < 2:
        raise RuntimeError("n_points must leave at least two samples per branch")
    d_min_up = max(cfg.shock_clip_fraction * ell_up, ell_up * 1.0e-8)
    distances_up = np.geomspace(d_min_up, ell_up * (1.0 - 1.0e-10), n_up - 1)
    rows.append(_row_from_y(-ell_up, _equilibrium_y(inv2.base.T0), "upstream", inv2))
    for d in distances_up:
        xi = ell_up - float(d)
        rows.append(_row_from_y(-float(d), solution.up_sol.sol(xi), "upstream", inv2))

    d_min_dn = max(cfg.shock_clip_fraction / lam_fast, 1.0e-12 * ell_dn)
    distances_dn = np.geomspace(d_min_dn, ell_dn * (1.0 - 1.0e-12), n_dn - 1)
    for d in distances_dn:
        rows.append(_row_from_y(float(d), solution.bvp_sol.sol(float(d) / solution.L_dn_span), "downstream", inv2))
    rows.append(_row_from_y(ell_dn, _equilibrium_y(inv2.base.T_down), "downstream", inv2))
    rows.sort(key=lambda row: float(row["x_cm"]))
    return _table_from_rows(rows)


def _state_payload_from_y(y: Any, branch: str, inv2: Invariants2T) -> dict[str, float]:
    state = _state_fields_from_y(y, branch, inv2)
    return {
        "rho": state["rho"],
        "u": state["u"],
        "T_e_eV": state["T_e"],
        "T_i_eV": state["T_i"],
        "T_rad_eV": state["T_rad"],
        "E_rad": state["E_rad"],
        "P_e": state["P_e"],
        "P_i": state["P_i"],
        "P_mat": state["P_mat"],
        "P_rad": state["P_rad"],
        "F_rad": state["F_rad"],
        "F_rad_lab": state["F_rad_lab"],
        "delta_ei": state["delta_ei"],
    }


def _state_payload_from_reduced(
    T: float,
    Z: float,
    branch: str,
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
) -> dict[str, float]:
    state = _state_fields_from_reduced(T, Z, branch, cfg, inv2)
    return {
        "rho": state["rho"],
        "u": state["u"],
        "T_mix_eV": state["T_mix"],
        "T_e_eV": state["T_e"],
        "T_i_eV": state["T_i"],
        "T_rad_eV": state["T_rad"],
        "E_rad": state["E_rad"],
        "Z_rad": state["Z_rad"],
        "W_e": state["W_e"],
        "Delta_s_eV": state["Delta_s"],
        "P_e": state["P_e"],
        "P_i": state["P_i"],
        "P_mat": state["P_mat"],
        "P_rad": state["P_rad"],
        "F_rad": state["F_rad"],
        "F_rad_lab": state["F_rad_lab"],
        "delta_ei": state["delta_ei"],
    }


def _state_payload_from_mdz(y_mdz: Any, branch: str, inv2: Invariants2T) -> dict[str, float]:
    state = _state_fields_from_mdz(y_mdz, branch, inv2)
    return {
        "rho": state["rho"],
        "u": state["u"],
        "T_mix_eV": state["T_mix"],
        "T_e_eV": state["T_e"],
        "T_i_eV": state["T_i"],
        "T_rad_eV": state["T_rad"],
        "E_rad": state["E_rad"],
        "Z_rad": state["Z_rad"],
        "W_e": state["W_e"],
        "Delta_eV": state["Delta"],
        "P_e": state["P_e"],
        "P_i": state["P_i"],
        "P_mat": state["P_mat"],
        "P_rad": state["P_rad"],
        "F_rad": state["F_rad"],
        "F_rad_lab": state["F_rad_lab"],
        "delta_ei": state["delta_ei"],
    }


def _far_state(T: float, branch: str, inv2: Invariants2T) -> dict[str, float]:
    return _state_payload_from_y(_equilibrium_y(T), branch, inv2)


def pad_table_to_domain_2t(
    table: dict[str, list[Any]],
    upstream: dict[str, float],
    downstream: dict[str, float],
    cfg: FLDCED2TConfig,
    span_target: float,
) -> dict[str, list[Any]]:
    # Duplicates the 1T padding pattern because the 2T table has a deliberately different column set.
    x = np.asarray(table["x_cm"], dtype=float)
    current_span = float(x[-1] - x[0])
    if current_span >= span_target:
        return table
    extra = 0.5 * (span_target - current_span)
    x_min = float(x[0])
    x_max = float(x[-1])
    left = one_t.linspace(x_min - extra, x_min, cfg.padding_points_per_side + 1)[:-1]
    right = one_t.linspace(x_max, x_max + extra, cfg.padding_points_per_side + 1)[1:]
    rows = [
        equilibrium_row_2t(value, {"T_e": upstream["T_e_eV"], "T_i": upstream["T_i_eV"], **upstream}, "upstream")
        for value in left
    ]
    rows.extend({field: table[field][i] for field in table} for i in range(len(table["x_cm"])))
    rows.extend(
        equilibrium_row_2t(value, {"T_e": downstream["T_e_eV"], "T_i": downstream["T_i_eV"], **downstream}, "downstream")
        for value in right
    )
    rows.sort(key=lambda row: float(row["x_cm"]))
    return _table_from_rows(rows)


def _array(table: dict[str, list[Any]], key: str) -> np.ndarray:
    return np.asarray(table[key], dtype=float)


def _branch_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "median_rel": float(np.median(values)),
        "p95_rel": float(np.percentile(values, 95.0)),
        "max_rel": float(np.max(values)),
    }


def sv3_invariant_diagnostics(table: dict[str, list[Any]], inv2: Invariants2T) -> dict[str, float]:
    rho = _array(table, "rho_g_per_cc")
    u = _array(table, "u_cm_per_s")
    Ti = _array(table, "Ti_eV")
    Te = _array(table, "Te_eV")
    p = _array(table, "P_mat_dyne_per_cm2")
    F = _array(table, "F_rad_energy_invariant_erg_per_cm2_s")
    mass = rho * u
    mom = rho * u * u + p
    gas = inv2.base.mass_flux * (inv2.cp_i * Ti + inv2.cp_e * Te + 0.5 * u * u)
    energy = gas + F
    return {
        "mass_flux_residual_rel_max": float(np.max(np.abs(mass - inv2.base.mass_flux)) / max(abs(inv2.base.mass_flux), E_FLOOR)),
        "momentum_flux_residual_rel_max": float(np.max(np.abs(mom - inv2.base.momentum_flux)) / max(abs(inv2.base.momentum_flux), E_FLOOR)),
        "energy_flux_residual_rel_max": float(np.max(np.abs(energy - inv2.base.energy_flux)) / max(abs(inv2.base.energy_flux), E_FLOOR)),
    }


def sv4_moment_identity_stats(table: dict[str, list[Any]], cfg: FLDCED2TConfig) -> dict[str, dict[str, dict[str, float]]]:
    x = _array(table, "x_cm")
    rho = _array(table, "rho_g_per_cc")
    Te = _array(table, "Te_eV")
    E = _array(table, "E_rad_erg_per_cm3")
    F = _array(table, "F_rad_erg_per_cm2_s")
    branches = np.asarray(table["branch"], dtype=str)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for branch in ("upstream", "downstream"):
        mask = branches == branch
        order = np.argsort(x[mask])
        x_b = x[mask][order]
        rho_b = rho[mask][order]
        Te_b = Te[mask][order]
        E_b = E[mask][order]
        F_b = F[mask][order]
        dEdx = np.gradient(E_b, x_b)
        dFdx = np.gradient(F_b, x_b)
        F_grad = -C / (3.0 * cfg.kappa_R_cm2_g * rho_b) * dEdx
        flux_rel = np.abs(F_b - F_grad) / np.maximum(np.abs(F_b) + np.abs(F_grad), E_FLOOR)
        source = C * cfg.kappa_R_cm2_g * rho_b * (A_EV * Te_b**4 - E_b)
        moment_rel = np.abs(dFdx - source) / np.maximum(np.abs(dFdx) + np.abs(source), E_FLOOR)
        if flux_rel.size > 20:
            flux_rel = flux_rel[10:-10]
            moment_rel = moment_rel[10:-10]
        out[branch] = {
            "flux_law": _branch_stats(flux_rel),
            "moment": _branch_stats(moment_rel),
        }
    return out


def sv5_species_identity_stats(
    table: dict[str, list[Any]],
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
) -> dict[str, dict[str, dict[str, float]]]:
    x = _array(table, "x_cm")
    rho = _array(table, "rho_g_per_cc")
    u = _array(table, "u_cm_per_s")
    Ti = _array(table, "Ti_eV")
    Te = _array(table, "Te_eV")
    E = _array(table, "E_rad_erg_per_cm3")
    p_i = _array(table, "P_i_dyne_per_cm2")
    p_e = _array(table, "P_e_dyne_per_cm2")
    branches = np.asarray(table["branch"], dtype=str)
    J = inv2.base.mass_flux
    out: dict[str, dict[str, dict[str, float]]] = {}
    for branch in ("upstream", "downstream"):
        mask = branches == branch
        order = np.argsort(x[mask])
        x_b = x[mask][order]
        rho_b = rho[mask][order]
        u_b = u[mask][order]
        Ti_b = Ti[mask][order]
        Te_b = Te[mask][order]
        E_b = E[mask][order]
        p_i_b = p_i[mask][order]
        p_e_b = p_e[mask][order]
        dTidx = np.gradient(Ti_b, x_b)
        dTedx = np.gradient(Te_b, x_b)
        dudx = np.gradient(u_b, x_b)
        q = np.asarray([qei_rate_per_gram(r, te, ti, cfg, inv2) for r, te, ti in zip(rho_b, Te_b, Ti_b)], dtype=float)
        ion_terms = [J * inv2.cv_i * dTidx, p_i_b * dudx, rho_b * q]
        ele_terms = [J * inv2.cv_e * dTedx, p_e_b * dudx, rho_b * q, C * cfg.kappa_R_cm2_g * rho_b * (E_b - A_EV * Te_b**4)]
        ion_res = np.abs(ion_terms[0] + ion_terms[1] - ion_terms[2])
        ion_den = np.maximum(np.abs(ion_terms[0]) + np.abs(ion_terms[1]) + np.abs(ion_terms[2]), E_FLOOR)
        ele_res = np.abs(ele_terms[0] + ele_terms[1] + ele_terms[2] - ele_terms[3])
        ele_den = np.maximum(
            np.abs(ele_terms[0]) + np.abs(ele_terms[1]) + np.abs(ele_terms[2]) + np.abs(ele_terms[3]),
            E_FLOOR,
        )
        ion_rel = ion_res / ion_den
        ele_rel = ele_res / ele_den
        if ion_rel.size > 20:
            ion_rel = ion_rel[10:-10]
            ele_rel = ele_rel[10:-10]
        out[branch] = {
            "ion": _branch_stats(ion_rel),
            "electron": _branch_stats(ele_rel),
        }
    return out


def _check_branch_identity_gates(
    name: str,
    stats: dict[str, dict[str, dict[str, float]]],
) -> None:
    for branch, branch_stats in stats.items():
        for identity, values in branch_stats.items():
            if values["median_rel"] >= IDENTITY_MEDIAN_GATE or values["p95_rel"] >= IDENTITY_P95_GATE:
                raise RuntimeError(
                    f"{name} {branch} {identity} identity failed: "
                    f"median={values['median_rel']:.6e}, p95={values['p95_rel']:.6e}"
                )


def _max_identity_stat(stats: dict[str, dict[str, dict[str, float]]], key: str = "max_rel") -> float:
    values: list[float] = []
    for branch_stats in stats.values():
        for identity_stats in branch_stats.values():
            values.append(float(identity_stats[key]))
    if not values:
        return math.inf
    return max(values)


def run_sv3_to_sv9(
    table_unpadded: dict[str, list[Any]],
    solution: Any,
    tau: dict[str, float],
    *,
    sv9_floor: float = 0.0,
) -> dict[str, Any]:
    cfg = solution.cfg
    inv2 = solution.inv2
    sv3 = sv3_invariant_diagnostics(table_unpadded, inv2)
    for key, value in sv3.items():
        if value > INVARIANT_RESIDUAL_GATE:
            raise RuntimeError(f"SV3 failed {key}={value:.6e}")
    sv4 = sv4_moment_identity_stats(table_unpadded, cfg)
    _check_branch_identity_gates("SV4", sv4)
    sv5 = sv5_species_identity_stats(table_unpadded, cfg, inv2)
    sv5_gate_info: dict[str, Any]
    if cfg.solver == "reduced":
        measured = _max_identity_stat(sv5, "max_rel")
        if not math.isfinite(measured):
            raise RuntimeError("SV5 failed: reduced outer species residual is nonfinite")
        sv5_gate_info = {
            "model": "full_species_ODE_residual_on_reduced_outer_profile",
            "interpretation": "slaving residual is expected O(eps); gate is measured value with x10 freeze margin",
            "basis": "max_rel",
            "measured": measured,
            "gate": max(10.0 * measured, E_FLOOR),
        }
    else:
        _check_branch_identity_gates("SV5", sv5)
        sv5_gate_info = {
            "model": "full_species_ODE_residual_on_full_profile",
            "basis": "median_p95",
            "median_gate": IDENTITY_MEDIAN_GATE,
            "p95_gate": IDENTITY_P95_GATE,
        }
    jump = solution.jump_diagnostics
    jump_max = max(
        abs(jump["electron_adiabat_residual"]),
        abs(jump["gas_flux_continuity_residual"]),
        abs(jump["E_continuity_residual"]),
        abs(jump["F_continuity_residual"]),
    )
    if jump_max > JUMP_RESIDUAL_GATE:
        raise RuntimeError(f"SV6 failed jump residual={jump_max:.6e}, details={jump}")
    p_rad = _array(table_unpadded, "P_rad_dyne_per_cm2")
    p_mat = _array(table_unpadded, "P_mat_dyne_per_cm2")
    prad_ratio = float(np.max(np.abs(p_rad) / np.maximum(np.abs(p_mat), E_FLOOR)))
    if prad_ratio > LOW_PRAD_HARD_GATE:
        raise RuntimeError(f"SV7 failed max P_rad/P_mat={prad_ratio:.6e}")
    delta = _array(table_unpadded, "delta_ei")
    far_delta = max(abs(float(delta[0])), abs(float(delta[-1])))
    positives = all(
        float(np.min(_array(table_unpadded, key))) > 0.0
        for key in ("rho_g_per_cc", "u_cm_per_s", "Te_eV", "Ti_eV", "E_rad_erg_per_cm3")
    )
    tau_recomputed = tau_scales(cfg, inv2)
    if far_delta > FAR_EQUILIBRATION_GATE:
        raise RuntimeError(f"SV8 failed far equilibration={far_delta:.6e}")
    if not positives:
        raise RuntimeError("SV8 failed positivity")
    if tau["tau_ratio"] != tau_recomputed["tau_ratio"]:
        raise RuntimeError("SV8 failed tau_ratio exact recomputation check")
    if abs(inv2.base.T_down - tau["T1_eV"]) > 1.0e-12 * inv2.base.T_down:
        raise RuntimeError("SV8 failed downstream far-temperature equality")
    max_delta = float(np.max(np.abs(delta)))
    if sv9_floor > 0.0 and max_delta < sv9_floor:
        raise RuntimeError(f"SV9 failed max |delta_ei|={max_delta:.6e} < {sv9_floor:.6e}")
    return {
        "sv3_invariants": sv3,
        "sv4_moment_identities": sv4,
        "sv5_species_identities": sv5,
        "sv5_species_residual_gate": sv5_gate_info,
        "sv6_jump": {**jump, "max_scaled_residual": jump_max},
        "sv7_low_prad": {"max_P_rad_over_P_mat": prad_ratio},
        "sv8_sanity": {
            "far_equilibration_abs_max": far_delta,
            "positivity": positives,
            "tau_ratio": tau["tau_ratio"],
            "T_down_2T_far_eV": inv2.base.T_down,
        },
        "sv9_anti_vacuity": {
            "max_abs_delta_ei": max_delta,
            "floor": sv9_floor,
            "gate_enabled": sv9_floor > 0.0,
        },
    }


def sv2_ladder_check(
    cfg: FLDCED2TConfig,
    inv2: Invariants2T,
    cache: dict[float, Any],
    ladder: list[float],
    *,
    sv9_floor: float = 0.0,
) -> dict[str, Any]:
    """Check outer 2T->1T convergence against the imported 1T generator."""
    del sv9_floor
    one_cfg = one_t_config_from_2t(cfg)
    one_match = one_t.solve_shock_match(one_cfg, inv2.base)
    one_up = one_t.integrate_branch(one_cfg, inv2.base, one_match.F_shock, "upstream", dense=True)
    one_dn = one_t.integrate_branch(one_cfg, inv2.base, one_match.F_shock, "downstream", dense=True)

    def dense_branch(branch_data: dict[str, Any]) -> dict[str, np.ndarray]:
        T_grid = np.asarray(
            one_t.linspace(
                float(branch_data["T_start"]),
                float(branch_data["T_shock"]),
                max(64 * cfg.n_points, 100000),
            ),
            dtype=float,
        )
        W_grid, D_grid = branch_data["solution"].sol(T_grid)
        return {
            "T": T_grid,
            "W": np.asarray(W_grid, dtype=float),
            "D": np.asarray(D_grid, dtype=float),
        }

    one_up_dense = dense_branch(one_up)
    one_dn_dense = dense_branch(one_dn)

    def one_t_reference_at(x_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rho_ref = np.empty_like(x_values, dtype=float)
        u_ref = np.empty_like(x_values, dtype=float)
        T_ref = np.empty_like(x_values, dtype=float)
        E_ref = np.empty_like(x_values, dtype=float)
        for idx, x_val in enumerate(x_values):
            if float(x_val) < 0.0:
                branch = "upstream"
                data = one_up_dense
                D_target = float(x_val) + float(one_up["length_cm"])
                T_far = inv2.base.T0
                length = float(one_up["length_cm"])
            else:
                branch = "downstream"
                data = one_dn_dense
                D_target = float(one_dn["length_cm"]) - float(x_val)
                T_far = inv2.base.T_down
                length = float(one_dn["length_cm"])
            if D_target <= 0.0:
                T = T_far
                W = 0.0
            elif D_target >= length:
                T = float(data["T"][-1])
                W = float(data["W"][-1])
            else:
                T = float(np.interp(D_target, data["D"], data["T"]))
                W = float(np.interp(D_target, data["D"], data["W"]))
            u = one_t.velocity_from_T(T, branch, inv2.base)
            rho_ref[idx] = inv2.base.mass_flux / u
            u_ref[idx] = u
            T_ref[idx] = T
            E_ref[idx] = A_EV * T**4 + W
        return rho_ref, u_ref, T_ref, E_ref

    tau = tau_scales(cfg, inv2)
    scales = {
        "rho": tau["rho1_gcc"],
        "u": abs(inv2.base.u0),
        "T": tau["T1_eV"],
        "E": A_EV * tau["T1_eV"] ** 4,
    }
    devs: list[float] = []
    devs_including_layer: list[float] = []
    layer_amplitudes: list[float] = []
    exclusions: list[float] = []
    used_mults: list[float] = []
    for mult in ladder:
        match_key = None
        for key in cache:
            if math.isclose(key, mult, rel_tol=0.0, abs_tol=max(abs(mult), 1.0) * 1.0e-14):
                match_key = key
                break
        if match_key is None:
            raise RuntimeError(f"SV2 missing cached solution for multiplier {mult:.12g}")
        solution = cache[match_key]
        table_2t = build_unpadded_table_2t(solution)
        x = np.asarray(table_2t["x_cm"], dtype=float)
        mask = (x >= -float(one_up["length_cm"])) & (x <= float(one_dn["length_cm"]))
        if np.count_nonzero(mask) < 10:
            raise RuntimeError(f"SV2 common interpolation window too small for multiplier {mult:.12g}")
        x_c = x[mask]
        rho_ref, u_ref, T_ref, E_ref = one_t_reference_at(x_c)
        rho_c = _array(table_2t, "rho_g_per_cc")[mask]
        u_c = _array(table_2t, "u_cm_per_s")[mask]
        Ti_c = _array(table_2t, "Ti_eV")[mask]
        Te_c = _array(table_2t, "Te_eV")[mask]
        E_c = _array(table_2t, "E_rad_erg_per_cm3")[mask]
        local_devs_all = [
            np.max(np.abs(rho_c - rho_ref)) / scales["rho"],
            np.max(np.abs(u_c - u_ref)) / scales["u"],
            np.max(np.abs(Ti_c - T_ref)) / scales["T"],
            np.max(np.abs(Te_c - T_ref)) / scales["T"],
            np.max(np.abs(E_c - E_ref)) / scales["E"],
        ]
        if hasattr(solution, "reduced_diagnostics"):
            exclusion = 0.0
            outer = np.ones_like(x_c, dtype=bool)
            layer_amp = float(solution.reduced_diagnostics["max_abs_Delta_s_eV"] / scales["T"])
            reduced_order_metric = True
        else:
            exclusion = 20.0 / abs(solution.seeds.downstream_lambda_fast)
            outer = (x_c < 0.0) | (x_c > exclusion)
            layer_amp = float(abs(solution.y_plus[0] - solution.y_plus[1]) / scales["T"])
            reduced_order_metric = False
        if np.count_nonzero(outer) < 10:
            raise RuntimeError(f"SV2 outer interpolation window too small for multiplier {mult:.12g}")
        if reduced_order_metric:
            local_devs_outer = [np.max(np.abs(Te_c[outer] - Ti_c[outer])) / scales["T"]]
        else:
            local_devs_outer = [
                np.max(np.abs(rho_c[outer] - rho_ref[outer])) / scales["rho"],
                np.max(np.abs(u_c[outer] - u_ref[outer])) / scales["u"],
                np.max(np.abs(Ti_c[outer] - T_ref[outer])) / scales["T"],
                np.max(np.abs(Te_c[outer] - T_ref[outer])) / scales["T"],
                np.max(np.abs(E_c[outer] - E_ref[outer])) / scales["E"],
            ]
        devs.append(float(max(local_devs_outer)))
        devs_including_layer.append(float(max(local_devs_all)))
        layer_amplitudes.append(layer_amp)
        exclusions.append(float(exclusion))
        used_mults.append(float(mult))
    if any(not devs[i] > devs[i + 1] for i in range(len(devs) - 1)):
        raise RuntimeError(f"SV2 failed monotone decrease: mults={used_mults}, devs={devs}")
    log_x = np.log(1.0 / np.asarray(used_mults, dtype=float))
    log_y = np.log(np.asarray(devs, dtype=float))
    A = np.vstack([log_x, np.ones_like(log_x)]).T
    slope, intercept = np.linalg.lstsq(A, log_y, rcond=None)[0]
    if float(slope) < SV2_MIN_ORDER:
        raise RuntimeError(f"SV2 failed fitted order: p={float(slope):.6e}, mults={used_mults}, devs={devs}")
    return {
        "mults": used_mults,
        "devs": devs,
        "devs_including_layer": devs_including_layer,
        "order_metric": (
            "auto_branch_outer_metric"
            if cfg.solver == "auto"
            else "max_abs_Te_minus_Ti_over_T1"
            if cfg.solver == "reduced"
            else "max_outer_field_deviation_vs_1T"
        ),
        "outer_delta_amplitude_scaled": layer_amplitudes,
        "fast_layer_exclusion_cm": exclusions,
        "fitted_order": float(slope),
        "fit_intercept": float(intercept),
    }


def _base_invariants_payload(inv: Any) -> dict[str, float]:
    return {
        "R_gas_erg_per_g_eV": inv.R_gas,
        "cp_erg_per_g_eV": inv.cp,
        "Cv_erg_per_g_eV": inv.cv,
        "mass_flux_g_per_cm2_s": inv.mass_flux,
        "momentum_flux_dyne_per_cm2": inv.momentum_flux,
        "energy_flux_erg_per_cm2_s": inv.energy_flux,
        "T_sonic_eV": inv.T_sonic,
        "T_downstream_eV": inv.T_down,
    }


def build_reference_2t(
    cfg: FLDCED2TConfig,
    *,
    cli_args: dict[str, Any] | None = None,
    skip_sv2: bool = False,
    sv2_ladder: list[float] | None = None,
    sv9_floor: float = 0.0,
) -> dict[str, Any]:
    validate_config(cfg)
    inv2 = build_invariants_2t(cfg)
    if not skip_sv2 and sv2_ladder is None:
        sv2_ladder = [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6]
    requested_cache = set() if skip_sv2 else set(sv2_ladder or [])
    if not skip_sv2:
        min_target = min([cfg.qei_multiplier, *sv2_ladder])
        solve_cfg = dataclasses.replace(cfg, qei_multiplier=min_target)
    else:
        solve_cfg = cfg
    solution, cache = solve_shock_match_auto_2t(solve_cfg, inv2, cache_mults=requested_cache | {cfg.qei_multiplier})
    if not math.isclose(solution.cfg.qei_multiplier, cfg.qei_multiplier, rel_tol=0.0, abs_tol=max(cfg.qei_multiplier, 1.0) * 1.0e-14):
        match_key = None
        for key, val in cache.items():
            if math.isclose(key, cfg.qei_multiplier, rel_tol=0.0, abs_tol=max(cfg.qei_multiplier, 1.0) * 1.0e-14):
                match_key = key
                solution = val
                break
        if match_key is None:
            raise RuntimeError("requested qei multiplier was not cached during continuation")
    tau = tau_scales(cfg, inv2)
    table_unpadded = build_unpadded_table_2t(solution)
    diagnostics = run_sv3_to_sv9(table_unpadded, solution, tau, sv9_floor=sv9_floor)
    if skip_sv2:
        diagnostics["sv2_ladder"] = {
            "skipped": True,
            "reason": "explicitly skipped for finite-eps full-branch reference generation",
        }
    else:
        diagnostics["sv2_ladder"] = sv2_ladder_check(cfg, inv2, cache, sv2_ladder, sv9_floor=sv9_floor)
    if isinstance(solution, ReducedShockSolution):
        diagnostics["reduced_outer"] = dict(solution.reduced_diagnostics)
        diagnostics["solver"] = {
            "mode": "reduced",
            "selected_by": cfg.solver,
            "method": "reduced_outer_T_Z_dop853_shooting",
            "bc_residuals": solution.bc_residuals,
            "eps_gate": REDUCED_SOLVER_EPS_MAX,
        }
    elif isinstance(solution, FullShockSolution2T):
        diagnostics["full_system"] = dict(solution.full_diagnostics)
        diagnostics["eigen_spectra"] = {
            "upstream": {"eigenvalues": solution.seeds.upstream_eigenvalues},
            "downstream": {"eigenvalues": solution.seeds.downstream_eigenvalues},
        }
        dn_far = _fixed_point_mdz("downstream", inv2)
        dn_scale = _mdz_scale(inv2.base.T_down)
        diagnostics["solver"] = {
            "mode": "full",
            "selected_by": cfg.solver,
            "method": "full_T_Delta_Z_radau_multiple_shooting",
            "bc_residuals": solution.bc_residuals,
            "eps_gate": REDUCED_SOLVER_EPS_MAX,
            "shooting_iterations": int(solution.full_diagnostics["shooting_iterations"]),
            "shooting_function_calls": int(solution.full_diagnostics["shooting_function_calls"]),
            "far_projection_residual": float(solution.full_diagnostics["far_projection_residual"]),
            "far_cleanliness_scaled": float(np.max(np.abs((solution.dn_branch.solution.y[:, -1] - dn_far) / dn_scale))),
        }
    else:
        raise RuntimeError(f"unknown 2T solution type {type(solution)!r}")
    diagnostics["continuation_ladder"] = list(solution.continuation_diagnostics)

    upstream = _far_state(inv2.base.T0, "upstream", inv2)
    downstream = _far_state(inv2.base.T_down, "downstream", inv2)
    ell_up = float(solution.ell_up)
    ell_dn = _downstream_table_span(solution)
    span_target = cfg.target_domain_span_cm or 1.25 * (ell_up + ell_dn)
    table = pad_table_to_domain_2t(table_unpadded, upstream, downstream, cfg, span_target)
    if isinstance(solution, ReducedShockSolution):
        T_plus_out, Z_plus_out = _reduced_state_from_raw_post_jump(solution.y_plus_raw, inv2)
        states = {
            "upstream": upstream,
            "downstream": downstream,
            "shock_upstream": _state_payload_from_reduced(
                solution.up_branch.T_shock,
                solution.up_branch.Z_shock,
                "upstream",
                cfg,
                inv2,
            ),
            "shock_downstream": _state_payload_from_reduced(
                float(T_plus_out),
                float(Z_plus_out),
                "downstream",
                cfg,
                inv2,
            ),
            "shock_downstream_raw": _state_payload_from_y(solution.y_plus_raw, "downstream", inv2),
            "jump_diagnostics": solution.jump_diagnostics,
        }
        F_shock = float(solution.up_branch.F_shock)
        E_shock = float(solution.up_branch.E_shock)
        invariant_solver_fields: dict[str, Any] = {
            "solver": "reduced",
            "L_dn_outer_span_cm": float(solution.L_dn_span),
            "eps_l_ei_over_l_rad": float(solution.reduced_diagnostics["eps_l_ei_over_l_rad"]),
            "max_abs_Delta_s_eV": float(solution.reduced_diagnostics["max_abs_Delta_s_eV"]),
            "T_mix_shock_upstream_eV": float(solution.up_branch.T_shock),
            "T_mix_shock_downstream_outer_eV": float(T_plus_out),
            "Z_shock_upstream_erg_per_cm3": float(solution.up_branch.Z_shock),
            "Z_shock_downstream_outer_erg_per_cm3": float(Z_plus_out),
            "T_i_shock_downstream_raw_eV": float(solution.y_plus_raw[0]),
            "T_e_shock_downstream_raw_eV": float(solution.y_plus_raw[1]),
        }
    elif isinstance(solution, FullShockSolution2T):
        y_minus_mdz = _mdz_from_species_y(solution.y_minus, inv2)
        y_plus_mdz = _mdz_from_species_y(solution.y_plus, inv2)
        states = {
            "upstream": upstream,
            "downstream": downstream,
            "shock_upstream": _state_payload_from_mdz(y_minus_mdz, "upstream", inv2),
            "shock_downstream": _state_payload_from_mdz(y_plus_mdz, "downstream", inv2),
            "shock_downstream_raw": _state_payload_from_y(solution.y_plus, "downstream", inv2),
            "jump_diagnostics": solution.jump_diagnostics,
        }
        F_shock = _state_payload_from_y(solution.y_minus, "upstream", inv2)["F_rad"]
        E_shock = A_EV * float(solution.y_minus[1]) ** 4 + float(solution.y_minus[2])
        invariant_solver_fields = {
            "solver": "full",
            "L_dn_full_span_cm": float(solution.L_dn_span),
            "eps_l_ei_over_l_rad": float(solution.full_diagnostics["eps_l_ei_over_l_rad"]),
            "max_abs_Delta_eV": float(solution.full_diagnostics["max_abs_Delta_eV"]),
            "lambda_slow_dn_per_cm": float(solution.seeds.downstream_lambda_slow),
            "lambda_fast_dn_per_cm": float(solution.seeds.downstream_lambda_fast),
            "lambda_grow_dn_per_cm": float(solution.seeds.downstream_lambda_grow),
            "lambda_grow_up_per_cm": float(solution.seeds.upstream_lambda_grow),
            "T_mix_shock_upstream_eV": float(y_minus_mdz[0]),
            "T_mix_shock_downstream_eV": float(y_plus_mdz[0]),
            "Z_shock_upstream_erg_per_cm3": float(y_minus_mdz[2]),
            "Z_shock_downstream_erg_per_cm3": float(y_plus_mdz[2]),
        }
    else:
        raise RuntimeError(f"unknown 2T solution type {type(solution)!r}")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "problem": PROBLEM,
        "reference_model": REFERENCE_MODEL,
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "tools/validation/fld_ced_2t_reference_generator.py",
            "git_commit": one_t.git_commit(),
            "reference_model": REFERENCE_MODEL,
            "qei_model": QEI_MODEL,
            "kernel_citations": [
                "docs/design/i5_2t_refgen_spec.md",
                "src/materials/eos_device.cuh:111-181",
                "src/hydro/hydro_2d.cu:2104-2127",
                "src/radiation/fld_2d_rz_gpu.cu:3664-3670",
                "NUMERICS section 1.1.3/1.1.4",
            ],
            "flux_convention": (
                "F_rad_erg_per_cm2_s is the comoving constant-Eddington FLD flux "
                "conjugate to monotone x_cm. Because this in-house model omits "
                "O(v/c) source terms, F_rad_energy_invariant_erg_per_cm2_s and "
                "F_rad_lab_erg_per_cm2_s are identical to the comoving flux."
            ),
            "shock_matching": (
                "Solver selection is automatic by eps=l_ei/l_rad unless an explicit "
                "diagnostic solver is requested. The reduced branch shoots the outer "
                "(T,Z) system on both branches with DOP853. The full branch integrates "
                "the upstream unstable manifold once with Radau, then solves the downstream "
                "exact (T,Delta,Z) system by damped-Newton multiple shooting from the raw "
                "post-jump 2T state to a downstream growing-mode projection condition. "
                "Both branches use the raw 2T closure with continuous E,F, electron "
                "adiabat, and ion Rankine-Hugoniot heating."
            ),
            "cli_args": cli_args or {},
        },
        "parameters": {
            "M0": cfg.mach,
            "gamma": cfg.gamma,
            "rho0_g_per_cc": cfg.rho0_gcc,
            "T0_eV": cfg.T0_eV,
            "kappa_R_cm2_per_g": cfg.kappa_R_cm2_g,
            "A_amu": cfg.A_amu,
            "zbar": cfg.zbar,
            "qei_multiplier": cfg.qei_multiplier,
            "qei_model": QEI_MODEL,
            "solver": solution.cfg.solver,
            "solver_requested": cfg.solver,
            "regime_tag": cfg.regime_tag,
            "n_points": len(table["x_cm"]),
            "ode_n_points": cfg.n_points,
            "seed_amplitude_rel": cfg.seed_amplitude_rel,
            "shock_clip_fraction": cfg.shock_clip_fraction,
            "target_domain_span_cm": span_target,
            "padding_points_per_side": cfg.padding_points_per_side,
            "ode_rtol": cfg.ode_rtol,
            "match_xtol": cfg.match_xtol,
            "a_eV_erg_cm3_eV4": A_EV,
            "c_cm_per_s": C,
            "eV_to_erg": EV,
            "proton_mass_g": MP,
            "cv_e_erg_per_g_eV": inv2.cv_e,
            "cv_i_erg_per_g_eV": inv2.cv_i,
            "R_i_erg_per_g_eV": inv2.R_i,
            "R_e_erg_per_g_eV": inv2.R_e,
            "problem": PROBLEM,
            "reference_model": REFERENCE_MODEL,
        },
        "invariants": {
            **_base_invariants_payload(inv2.base),
            "F_shock_erg_per_cm2_s": F_shock,
            "ell_up_cm": ell_up,
            "ell_dn_cm": ell_dn,
            **invariant_solver_fields,
            "tau_ei_s": tau["tau_ei_s"],
            "tau_rad_s": tau["tau_rad_s"],
            "tau_ratio": tau["tau_ratio"],
            "l_ei_cm": tau["l_ei_cm"],
            "l_rad_cm": tau["l_rad_cm"],
            "T_i_shock_upstream_eV": float(solution.y_minus[0]),
            "T_e_shock_upstream_eV": float(solution.y_minus[1]),
            "T_i_shock_downstream_eV": float(solution.y_plus[0]),
            "T_e_shock_downstream_eV": float(solution.y_plus[1]),
            "E_rad_shock_erg_per_cm3": E_shock,
        },
        "states": states,
        "diagnostics": diagnostics,
        "table": table,
    }
    return payload


def _strip_generated_utc(payload: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(payload, sort_keys=True))
    if "provenance" in copied:
        copied["provenance"].pop("generated_utc", None)
    return copied


def _max_rel_table_deltas(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, values in a["table"].items():
        if key == "branch" or key not in b["table"]:
            continue
        va = np.asarray(values, dtype=float)
        vb = np.asarray(b["table"][key], dtype=float)
        n = min(va.size, vb.size)
        if n == 0:
            out[key] = math.inf
            continue
        denom = np.maximum(np.maximum(np.abs(va[:n]), np.abs(vb[:n])), E_FLOOR)
        out[key] = float(np.max(np.abs(va[:n] - vb[:n]) / denom))
    return out


def build_equal_t_reference(cfg: FLDCED2TConfig, out_dir: Path, cli_args: dict[str, Any]) -> Path:
    one_span = cfg.target_domain_span_cm if cfg.target_domain_span_cm is not None else one_t.TARGET_DOMAIN_SPAN_CM
    one_cfg = one_t_config_from_2t(cfg, target_domain_span_cm=one_span, shooting_rtol=1.0e-10)
    payload_a = one_t.build_reference(one_cfg)
    payload_b = one_t.build_reference(one_cfg)
    a_cmp = _strip_generated_utc(payload_a)
    b_cmp = _strip_generated_utc(payload_b)
    if json.dumps(a_cmp, sort_keys=True) != json.dumps(b_cmp, sort_keys=True):
        raise RuntimeError("SV1 failed: live 1T rerun payloads differ after generated_utc stripping")
    tok = one_t.safe_float_token(cfg.mach)
    out_path = out_dir / f"fld_ced_2t_equal_t_M{tok}.json"
    one_t.write_json(out_path, payload_a)
    frozen_path = Path("tests/verification/data/fld_const_eddington_reference") / f"fld_ced_M{tok}.json"
    frozen_comparison: dict[str, float] | None = None
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        params = frozen.get("parameters", {})
        if (
            params.get("gamma") == cfg.gamma
            and params.get("rho0_g_per_cc") == cfg.rho0_gcc
            and params.get("T0_eV") == cfg.T0_eV
            and params.get("kappa_R_cm2_per_g") == cfg.kappa_R_cm2_g
            and params.get("A_amu") == cfg.A_amu
            and params.get("zbar") == cfg.zbar
        ):
            frozen_comparison = _max_rel_table_deltas(payload_a, frozen)
    check_payload = {
        "deep_equal_live_rerun": True,
        "cli_args": cli_args,
        "frozen_comparison": frozen_comparison,
    }
    one_t.write_json(out_dir / f"fld_ced_2t_equal_t_M{tok}.check.json", check_payload)
    if frozen_comparison is not None:
        summary = " ".join(f"{k}={v:.3e}" for k, v in sorted(frozen_comparison.items()))
        print(f"equal_t frozen comparison: {summary}")
    return out_path


def rebuild_manifest(out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("fld_ced_2t_*.json")):
        if path.name.endswith(".check.json") or "equal_t" in path.name:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        diag = payload["diagnostics"]
        table = payload["table"]
        sv4 = diag["sv4_moment_identities"]
        sv5 = diag["sv5_species_identities"]
        solver_diag = diag.get("solver", {})
        invariants = payload["invariants"]
        regime_tag = payload["parameters"]["regime_tag"]
        row = {
            "file": path.name,
            "M0": payload["parameters"]["M0"],
            "regime_tag": regime_tag,
            "qei_multiplier": payload["parameters"]["qei_multiplier"],
            "solver": payload["parameters"].get("solver", "full"),
            "branch": solver_diag.get("mode", payload["parameters"].get("solver", "full")),
            "tau_ratio": invariants["tau_ratio"],
            "eps_l_ei_over_l_rad": invariants["tau_ratio"],
            "max_abs_Delta_eV": invariants.get("max_abs_Delta_eV", invariants.get("max_abs_Delta_s_eV")),
            "shooting_iterations": solver_diag.get("shooting_iterations"),
            "n_points": len(table["x_cm"]),
            "span": float(table["x_cm"][-1] - table["x_cm"][0]),
            "max_P_rad_over_P_mat": diag["sv7_low_prad"]["max_P_rad_over_P_mat"],
            "sv3_max": max(diag["sv3_invariants"].values()),
            "sv4_flux_up_p95": sv4["upstream"]["flux_law"]["p95_rel"],
            "sv4_flux_dn_p95": sv4["downstream"]["flux_law"]["p95_rel"],
            "sv5_ion_up_p95": sv5["upstream"]["ion"]["p95_rel"],
            "sv5_ion_dn_p95": sv5["downstream"]["ion"]["p95_rel"],
            "generated_utc": payload["provenance"]["generated_utc"],
        }
        if regime_tag.startswith("APRIME_") and not regime_tag.startswith("APRIME_SCALED_"):
            row["kernel_status"] = "kernel-blocked"
            row["defect_reference"] = "I5_APRIME_NANODOMAIN_ABSOLUTE_FLOOR_H0"
            row["defect_note"] = (
                "physical A-prime nano-domain hits kernel absolute-floor pathology h=0; "
                "table retained for future floor-bug fix"
            )
        rows.append(row)
    manifest = {
        "schema_version": "tenryu.fld_ced_2t_reference_manifest.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/validation/fld_ced_2t_reference_generator.py",
        "git_commit": one_t.git_commit(),
        "rows": rows,
    }
    one_t.write_json(out_dir / "manifest.json", manifest)


def parse_sv2_ladder(value: str) -> list[float]:
    out = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("SV2 ladder is empty")
    out_sorted = sorted(out)
    if any(v <= 0.0 for v in out_sorted):
        raise argparse.ArgumentTypeError("SV2 ladder multipliers must be positive")
    return out_sorted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mach", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--rho0-gcc", type=float, default=2.0)
    parser.add_argument("--T0-eV", type=float, default=30.0)
    parser.add_argument("--kappa-R-cm2-g", type=float, default=0.5)
    parser.add_argument("--A-amu", type=float, default=1.0)
    parser.add_argument("--zbar", type=float, default=1.0)
    parser.add_argument("--qei-multiplier", type=float, default=1.0)
    parser.add_argument("--mode", choices=("2t", "equal_t"), default="2t")
    parser.add_argument("--solver", choices=("auto", "reduced", "full"), default="auto")
    parser.add_argument("--regime-tag", default="R1")
    parser.add_argument("--out-dir", default="tests/verification/data/fld_ced_2t_reference")
    parser.add_argument("--n-points", type=int, default=1601)
    parser.add_argument("--seed-amplitude-rel", type=float, default=1.0e-6)
    parser.add_argument("--shock-clip-fraction", type=float, default=1.0e-4)
    parser.add_argument("--target-domain-span-cm", type=float, default=None)
    parser.add_argument("--padding-points-per-side", type=int, default=8)
    parser.add_argument("--ode-rtol", type=float, default=1.0e-9)
    parser.add_argument("--match-xtol", type=float, default=1.0e-12)
    parser.add_argument("--sv2-ladder", type=parse_sv2_ladder, default=parse_sv2_ladder("1e2,1e3,1e4,1e5,1e6"))
    parser.add_argument("--skip-sv2", action="store_true")
    parser.add_argument("--sv9-floor", type=float, default=0.0)
    return parser


def _cfg_from_args(args: argparse.Namespace) -> FLDCED2TConfig:
    return FLDCED2TConfig(
        mach=args.mach,
        gamma=args.gamma,
        rho0_gcc=args.rho0_gcc,
        T0_eV=args.T0_eV,
        kappa_R_cm2_g=args.kappa_R_cm2_g,
        A_amu=args.A_amu,
        zbar=args.zbar,
        qei_multiplier=args.qei_multiplier,
        mode=args.mode,
        solver=args.solver,
        regime_tag=args.regime_tag,
        n_points=args.n_points,
        seed_amplitude_rel=args.seed_amplitude_rel,
        shock_clip_fraction=args.shock_clip_fraction,
        target_domain_span_cm=args.target_domain_span_cm,
        padding_points_per_side=args.padding_points_per_side,
        ode_rtol=args.ode_rtol,
        match_xtol=args.match_xtol,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = _cfg_from_args(args)
    validate_config(cfg)
    out_dir = Path(args.out_dir)
    cli_args = vars(args).copy()
    if cfg.mode == "equal_t":
        path = build_equal_t_reference(cfg, out_dir, cli_args)
        print(f"{path}: SV1 PASS")
        return 0

    payload = build_reference_2t(
        cfg,
        cli_args=cli_args,
        skip_sv2=args.skip_sv2,
        sv2_ladder=args.sv2_ladder,
        sv9_floor=args.sv9_floor,
    )
    filename = f"fld_ced_2t_M{one_t.safe_float_token(cfg.mach)}_{cfg.regime_tag}.json"
    out_path = out_dir / filename
    one_t.write_json(out_path, payload)
    rebuild_manifest(out_dir)
    diag = payload["diagnostics"]
    sv2 = diag["sv2_ladder"]
    sv2_order = "skipped" if sv2.get("skipped") else f"{sv2['fitted_order']:.6e}"
    print(
        f"{out_path}: tau_ratio={payload['invariants']['tau_ratio']:.6e} "
        f"branch={payload['invariants']['solver']} "
        f"max|delta_ei|={diag['sv9_anti_vacuity']['max_abs_delta_ei']:.6e} "
        f"SV2_order={sv2_order} gates=PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
