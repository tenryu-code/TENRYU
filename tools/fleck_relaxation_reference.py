#!/usr/bin/env python3
"""Zero-dimensional radiation-matter relaxation reference checker."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import expit


# Exact values from src/core/constants.hpp.
C_LIGHT = 2.99792458e10
A_EV = 1.3720e2
EV_TO_ERG = 1.6022e-12
PROTON_MASS = 1.6726219e-24

CASE_NAME = "fleck_relaxation_0d"
# beta_sec G-S2 softstep feature constants — must match the "softstep" branch
# of examples/verification/fleck_relaxation_0d.py verbatim.
BUMP_STEP_D = 2.0e12
BUMP_STEP_TC = 1240.0
BUMP_STEP_W = 100.0
REQUIRED_ROLES = (
    "rate_dt0",
    "rate_dt1",
    "rate_dt2",
    "co_tab_dt0",
    "co_tab_dt1",
    "co_tab_dt2",
    "co_leg_dt0",
    "co_leg_dt1",
    "co_leg_dt2",
    "tau",
    "eqhold",
    "fled_table",
    "fled_legacy",
    "bump_tan_dt0",
    "bump_tan_dt1",
    "bump_tan_dt2",
    "bump_tan_dt3",
    "bump_sec_dt0",
    "bump_sec_dt1",
    "bump_sec_dt2",
    "bump_sec_dt3",
    "bump_exp_dt0",
    "bump_exp_dt1",
    "bump_exp_dt2",
    "bump_exp_dt3",
    "eff_tan",
    "eff_sec",
    "eff_grd",
    "bump_qxp_dt0",
    "bump_qxp_dt1",
    "bump_qxp_dt2",
    "bump_qxp_dt3",
    "bump_qmg_dt0",
    "bump_qmg_dt1",
    "bump_qmg_dt2",
    "bump_qmg_dt3",
)
# (fleck_cv_source mode, init, dt, t_end, ion A, eos variant, fleck_beta,
#  max_outer_iterations, fleck_form, source_integrator, groups)
ROLE_SPEC = {
    "rate_dt0": ("table", "hotmat", 1.0e-13, 3.0e-13, 1.0e5, "power", "tangent", 60, "be", "fleck", 1),
    "rate_dt1": ("table", "hotmat", 5.0e-14, 3.0e-13, 1.0e5, "power", "tangent", 60, "be", "fleck", 1),
    "rate_dt2": ("table", "hotmat", 2.5e-14, 3.0e-13, 1.0e5, "power", "tangent", 60, "be", "fleck", 1),
    "co_tab_dt0": ("table", "hotmat", 1.0e-13, 3.0e-13, 1.0, "power", "tangent", 60, "be", "fleck", 1),
    "co_tab_dt1": ("table", "hotmat", 5.0e-14, 3.0e-13, 1.0, "power", "tangent", 60, "be", "fleck", 1),
    "co_tab_dt2": ("table", "hotmat", 2.5e-14, 3.0e-13, 1.0, "power", "tangent", 60, "be", "fleck", 1),
    "co_leg_dt0": ("legacy", "hotmat", 1.0e-13, 3.0e-13, 1.0, "power", "tangent", 60, "be", "fleck", 1),
    "co_leg_dt1": ("legacy", "hotmat", 5.0e-14, 3.0e-13, 1.0, "power", "tangent", 60, "be", "fleck", 1),
    "co_leg_dt2": ("legacy", "hotmat", 2.5e-14, 3.0e-13, 1.0, "power", "tangent", 60, "be", "fleck", 1),
    "tau": ("table", "linpert", 5.0e-15, 1.0e-12, 1.0e5, "power", "tangent", 60, "be", "fleck", 1),
    "eqhold": ("table", "eqhold", 1.0e-13, 2.0e-12, 1.0e5, "power", "tangent", 60, "be", "fleck", 1),
    "fled_table": ("table", "hotmat", 1.0e-13, 2.0e-13, 1.0e5, "power", "tangent", 60, "be", "fleck", 1),
    "fled_legacy": ("legacy", "hotmat", 1.0e-13, 2.0e-13, 1.0e5, "power", "tangent", 60, "be", "fleck", 1),
    "bump_tan_dt0": ("table", "bumphot", 8.0e-15, 8.0e-15, 1.0e5, "softstep", "tangent", 1, "be", "fleck", 1),
    "bump_tan_dt1": ("table", "bumphot", 4.0e-15, 4.0e-15, 1.0e5, "softstep", "tangent", 1, "be", "fleck", 1),
    "bump_tan_dt2": ("table", "bumphot", 2.0e-15, 2.0e-15, 1.0e5, "softstep", "tangent", 1, "be", "fleck", 1),
    "bump_tan_dt3": ("table", "bumphot", 1.0e-15, 1.0e-15, 1.0e5, "softstep", "tangent", 1, "be", "fleck", 1),
    "bump_sec_dt0": ("table", "bumphot", 8.0e-15, 8.0e-15, 1.0e5, "softstep", "secant", 1, "be", "fleck", 1),
    "bump_sec_dt1": ("table", "bumphot", 4.0e-15, 4.0e-15, 1.0e5, "softstep", "secant", 1, "be", "fleck", 1),
    "bump_sec_dt2": ("table", "bumphot", 2.0e-15, 2.0e-15, 1.0e5, "softstep", "secant", 1, "be", "fleck", 1),
    "bump_sec_dt3": ("table", "bumphot", 1.0e-15, 1.0e-15, 1.0e5, "softstep", "secant", 1, "be", "fleck", 1),
    "bump_exp_dt0": ("table", "bumphot", 8.0e-15, 8.0e-15, 1.0e5, "softstep", "tangent", 1, "exp_phi1", "fleck", 1),
    "bump_exp_dt1": ("table", "bumphot", 4.0e-15, 4.0e-15, 1.0e5, "softstep", "tangent", 1, "exp_phi1", "fleck", 1),
    "bump_exp_dt2": ("table", "bumphot", 2.0e-15, 2.0e-15, 1.0e5, "softstep", "tangent", 1, "exp_phi1", "fleck", 1),
    "bump_exp_dt3": ("table", "bumphot", 1.0e-15, 1.0e-15, 1.0e5, "softstep", "tangent", 1, "exp_phi1", "fleck", 1),
    "eff_tan": ("table", "bumphot", 1.0e-13, 3.0e-13, 1.0e5, "softstep", "tangent", 60, "be", "fleck", 1),
    "eff_sec": ("table", "bumphot", 1.0e-13, 3.0e-13, 1.0e5, "softstep", "secant", 60, "be", "fleck", 1),
    "eff_grd": ("table", "bumphot", 1.0e-13, 3.0e-13, 1.0e5, "softstep", "guard", 60, "be", "fleck", 1),
    "bump_qxp_dt0": ("table", "bumphot", 8.0e-15, 8.0e-15, 1.0e5, "softstep", "tangent", 1, "be", "exp_rosenbrock", 1),
    "bump_qxp_dt1": ("table", "bumphot", 4.0e-15, 4.0e-15, 1.0e5, "softstep", "tangent", 1, "be", "exp_rosenbrock", 1),
    "bump_qxp_dt2": ("table", "bumphot", 2.0e-15, 2.0e-15, 1.0e5, "softstep", "tangent", 1, "be", "exp_rosenbrock", 1),
    "bump_qxp_dt3": ("table", "bumphot", 1.0e-15, 1.0e-15, 1.0e5, "softstep", "tangent", 1, "be", "exp_rosenbrock", 1),
    "bump_qmg_dt0": ("table", "bumphot", 8.0e-15, 8.0e-15, 1.0e5, "softstep", "tangent", 1, "be", "exp_rosenbrock", 4),
    "bump_qmg_dt1": ("table", "bumphot", 4.0e-15, 4.0e-15, 1.0e5, "softstep", "tangent", 1, "be", "exp_rosenbrock", 4),
    "bump_qmg_dt2": ("table", "bumphot", 2.0e-15, 2.0e-15, 1.0e5, "softstep", "tangent", 1, "be", "exp_rosenbrock", 4),
    "bump_qmg_dt3": ("table", "bumphot", 1.0e-15, 1.0e-15, 1.0e5, "softstep", "tangent", 1, "be", "exp_rosenbrock", 4),
}
INIT_SPEC = {
    "hotmat": (200.0, 0.0),
    "hotrad": (100.0, A_EV * 200.0**4),
    "eqhold": (150.0, A_EV * 150.0**4),
    "linpert": (100.0, A_EV * 105.0**4),
    "bumphot": (1000.0, A_EV * 1500.0**4),
}
HISTORY_DATASETS = (
    "t",
    "energy/radiation_field",
    "energy/internal_electron",
    "energy/internal_ion",
    "radiation/fld_outer_iterations",
    "radiation/fld_outer_converged",
)


class StopCheck(RuntimeError):
    pass


def stop(message: str) -> None:
    raise StopCheck(message)


def list_datasets(handle: h5py.File) -> list[str]:
    names: list[str] = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            names.append(name)

    handle.visititems(visit)
    return sorted(names)


def require_dataset(handle: h5py.File, path: str, source: Path) -> np.ndarray:
    if path not in handle:
        available = "\n  ".join(list_datasets(handle))
        stop(f"{source}: required dataset '{path}' not found. Available datasets:\n  {available}")
    return np.asarray(handle[path], dtype=float)


def require_key(mapping: dict, path: tuple[str, ...], source: Path):
    value = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            stop(f"{source}: frozen config lacks required field '{'.'.join(path)}'")
        value = value[key]
    return value


def require_float(mapping: dict, path: tuple[str, ...], source: Path) -> float:
    value = require_key(mapping, path, source)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        stop(f"{source}: frozen field '{'.'.join(path)}' is not numeric")
    value = float(value)
    if not math.isfinite(value):
        stop(f"{source}: frozen field '{'.'.join(path)}' is not finite")
    return value


def close(a: float, b: float, rtol: float = 1.0e-12) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1.0e-300)


def softplus(x):
    # Stable softplus, identical form to the W2a table generator
    # (src/materials/eos_table.cpp): max(x, 0) + log1p(exp(-|x|)).
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def eos_energy_per_g(config: FrozenConfig, temp):
    # e_e(rho, T) per gram: power_law_te plus the optional c_v softstep.
    base = config.eos_f * np.power(temp, config.eos_beta) * config.rho ** (-config.eos_mu)
    if config.step_D > 0.0:
        base = base + config.step_D * config.step_w * softplus(
            (temp - config.step_Tc) / config.step_w
        )
    return base


def invert_te(config: FrozenConfig, e_per_g: np.ndarray) -> np.ndarray:
    # Numeric inversion of the softstep EOS (monotone in T).
    values = np.asarray(e_per_g, dtype=float)
    out = np.empty_like(values)
    for k, target in enumerate(values):
        def residual(temp, target=target):
            return float(eos_energy_per_g(config, temp)) - target
        hi = 200.0
        while residual(hi) < 0.0:
            hi *= 2.0
            if hi > 1.0e6:
                stop("softstep Te inversion failed to bracket the target energy")
        out[k] = brentq(residual, 1.0e-6, hi, rtol=1.0e-13)
    return out


@dataclass
class FrozenConfig:
    path: Path
    name: str
    t_end: float
    nr: int
    r_min: float
    r_max: float
    dt: float
    mode: str
    radiation_field: str
    kappa_a: float
    kappa_s: float
    material_A: float
    material_Z: float
    gamma_eff: float
    zbar: float
    eos_f: float
    eos_beta: float
    eos_mu: float
    rho: float
    rho_source: str
    fleck_beta: str
    fleck_form: str
    source_integrator: str
    groups: int
    step_D: float
    step_Tc: float
    step_w: float
    max_outer: int


@dataclass
class History:
    path: Path
    t: np.ndarray
    erad: np.ndarray
    eint_e: np.ndarray
    eint_i: np.ndarray
    outer_iters: np.ndarray
    outer_converged: np.ndarray


@dataclass
class RunData:
    role: str
    run_dir: Path
    init: str
    config: FrozenConfig
    history: History
    volume: float = math.nan
    te: np.ndarray | None = None
    ed: np.ndarray | None = None


def frozen_config_path(run_dir: Path) -> Path:
    path = run_dir / "config" / f"{CASE_NAME}_frozen.json"
    if not path.is_file():
        stop(f"{run_dir}: missing confirmed frozen config path {path}")
    return path


def read_frozen_config(run_dir: Path, role: str, rho_default: float) -> FrozenConfig:
    path = frozen_config_path(run_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        stop(f"{path}: cannot read frozen config: {exc}")

    materials = require_key(payload, ("materials", "materials"), path)
    if not isinstance(materials, list) or len(materials) != 1 or not isinstance(materials[0], dict):
        stop(f"{path}: expected exactly one material object")
    mat = materials[0]
    if require_key(mat, ("eos_model",), path) != "power_law_te":
        stop(f"{path}: material eos_model is not 'power_law_te'")
    if require_key(mat, ("opacity_model",), path) != "constant":
        stop(f"{path}: material opacity_model is not 'constant'")

    geometry_rho = require_key(payload, ("geometry", "rho"), path)
    if isinstance(geometry_rho, (int, float)) and not isinstance(geometry_rho, bool):
        rho = float(geometry_rho)
        rho_source = "frozen config geometry.rho"
    else:
        if not isinstance(geometry_rho, dict) or geometry_rho.get("_type") != "callable":
            stop(f"{path}: geometry.rho is neither numeric nor a frozen callable")
        rho = rho_default
        rho_source = f"--rho default/fallback ({rho_default:.8g} g/cm^3)"

    config = FrozenConfig(
        path=path,
        name=str(require_key(payload, ("main", "name"), path)),
        t_end=require_float(payload, ("main", "t_end"), path),
        nr=int(require_float(payload, ("mesh", "nr"), path)),
        r_min=require_float(payload, ("mesh", "r_min"), path),
        r_max=require_float(payload, ("mesh", "r_max"), path),
        dt=require_float(payload, ("numerics", "dt", "initial_s"), path),
        mode=str(
            require_key(
                payload,
                ("radiation", "multigroup_diffusion", "fleck_cv_source"),
                path,
            )
        ),
        radiation_field=str(require_key(payload, ("geometry", "radiation_field"), path)),
        kappa_a=require_float(mat, ("kappa_a_constant",), path),
        kappa_s=require_float(mat, ("kappa_s_constant",), path),
        material_A=require_float(mat, ("A",), path),
        material_Z=require_float(mat, ("Z",), path),
        gamma_eff=require_float(mat, ("ideal_gas_gamma",), path),
        zbar=require_float(payload, ("materials", "zbar", "fixed_value"), path),
        eos_f=require_float(mat, ("eos_power_law_f_erg_g",), path),
        eos_beta=require_float(mat, ("eos_power_law_beta",), path),
        eos_mu=require_float(mat, ("eos_power_law_mu_rho",), path),
        rho=rho,
        rho_source=rho_source,
        fleck_beta=str(
            require_key(
                payload,
                ("radiation", "multigroup_diffusion", "fleck_beta"),
                path,
            )
        ),
        fleck_form=str(
            require_key(
                payload,
                ("radiation", "multigroup_diffusion", "fleck_form"),
                path,
            )
        ),
        source_integrator=str(
            require_key(
                payload,
                ("radiation", "multigroup_diffusion", "source_integrator"),
                path,
            )
        ),
        groups=int(require_float(payload, ("radiation", "groups"), path)),
        step_D=require_float(mat, ("eos_power_law_step_D_erg_g_eV",), path),
        step_Tc=require_float(mat, ("eos_power_law_step_Tc_eV",), path),
        step_w=require_float(mat, ("eos_power_law_step_w_eV",), path),
        max_outer=int(
            require_float(
                payload,
                ("radiation", "multigroup_diffusion", "max_outer_iterations"),
                path,
            )
        ),
    )

    (
        expected_mode,
        init,
        expected_dt,
        expected_t_end,
        expected_A,
        expected_eos,
        expected_beta,
        expected_outer,
        expected_form,
        expected_src,
        expected_groups,
    ) = ROLE_SPEC[role]
    expected_field = {
        "hotmat": "zero",
        "hotrad": "planck",
        "eqhold": "equilibrium",
        "linpert": "planck",
        "bumphot": "planck",
    }[init]
    expected = (
        (config.name == CASE_NAME, f"main.name={config.name!r}"),
        (config.mode == expected_mode, f"fleck_cv_source={config.mode!r}"),
        (config.radiation_field == expected_field, f"radiation_field={config.radiation_field!r}"),
        (close(config.dt, expected_dt), f"dt={config.dt:.8e}"),
        (close(config.t_end, expected_t_end), f"t_end={config.t_end:.8e}"),
        (close(config.kappa_a, 1667.0), f"kappa_a={config.kappa_a:.8e}"),
        (config.kappa_s == 0.0, f"kappa_s={config.kappa_s:.8e}"),
        (close(config.material_A, expected_A), f"A={config.material_A:.8e}"),
        (close(config.material_Z, 1.0), f"Z={config.material_Z:.8e}"),
        (close(config.eos_f, 5.7e8), f"eos f={config.eos_f:.8e}"),
        (close(config.eos_beta, 1.6), f"eos beta={config.eos_beta:.8e}"),
        (config.eos_mu == 0.0, f"eos mu_rho={config.eos_mu:.8e}"),
        (config.nr == 4, f"nr={config.nr}"),
        (close(config.r_min, 0.0), f"r_min={config.r_min:.8e}"),
        (close(config.r_max, 0.4), f"r_max={config.r_max:.8e}"),
        (config.fleck_beta == expected_beta, f"fleck_beta={config.fleck_beta!r}"),
        (config.fleck_form == expected_form, f"fleck_form={config.fleck_form!r}"),
        (config.source_integrator == expected_src, f"source_integrator={config.source_integrator!r}"),
        (config.groups == expected_groups, f"groups={config.groups}"),
        (config.max_outer == expected_outer, f"max_outer_iterations={config.max_outer}"),
    )
    if expected_eos == "power":
        expected = expected + (
            (config.step_D == 0.0, f"step_D={config.step_D:.8e}"),
        )
    else:
        expected = expected + (
            (close(config.step_D, BUMP_STEP_D), f"step_D={config.step_D:.8e}"),
            (close(config.step_Tc, BUMP_STEP_TC), f"step_Tc={config.step_Tc:.8e}"),
            (close(config.step_w, BUMP_STEP_W), f"step_w={config.step_w:.8e}"),
        )
    bad = [detail for ok, detail in expected if not ok]
    if bad:
        stop(f"{path}: role {role} does not match the specified deck: {', '.join(bad)}")
    return config


def read_history(run_dir: Path, config: FrozenConfig) -> History:
    path = run_dir / "results" / f"{config.name}_history.h5"
    if not path.is_file():
        stop(f"{run_dir}: missing confirmed history path {path}")
    with h5py.File(path, "r") as handle:
        arrays = {name: require_dataset(handle, name, path) for name in HISTORY_DATASETS}
    for name, array in arrays.items():
        if array.ndim != 1:
            stop(f"{path}: history dataset '{name}' is not rank 1")
    n = arrays["t"].size
    if n == 0:
        stop(f"{path}: empty history")
    if any(array.size != n for array in arrays.values()):
        stop(f"{path}: required history datasets have different lengths")
    t = arrays["t"]
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
        stop(f"{path}: history time is not finite and strictly increasing")
    if not t[0] > 0.0:
        stop(f"{path}: expected history sample 0 after the first step (t[0] > 0)")
    if not close(float(t[-1]), config.t_end, rtol=1.0e-10):
        stop(f"{path}: final history time {t[-1]:.16e} does not equal t_end {config.t_end:.16e}")
    if any(not np.all(np.isfinite(array)) for array in arrays.values()):
        stop(f"{path}: a required history dataset contains non-finite values")
    return History(
        path=path,
        t=t,
        erad=arrays["energy/radiation_field"],
        eint_e=arrays["energy/internal_electron"],
        eint_i=arrays["energy/internal_ion"],
        outer_iters=arrays["radiation/fld_outer_iterations"],
        outer_converged=arrays["radiation/fld_outer_converged"],
    )


def parse_legs(values: list[str]) -> dict[str, Path]:
    legs: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            stop(f"--leg expects role=path, got {raw!r}")
        role, raw_path = raw.split("=", 1)
        if role not in ROLE_SPEC:
            stop(f"unknown --leg role {role!r}; expected one of {', '.join(REQUIRED_ROLES)}")
        if role in legs:
            stop(f"duplicate --leg role {role!r}")
        if not raw_path:
            stop(f"--leg {role}: empty path")
        legs[role] = Path(raw_path)
    missing = [role for role in REQUIRED_ROLES if role not in legs]
    if missing:
        stop(f"missing required --leg roles: {', '.join(missing)}")
    return legs


def equilibrium_temperature(t0: float, ed0: float, config: FrozenConfig) -> float:
    total = ed0 + config.rho * float(eos_energy_per_g(config, t0))

    def residual(temp):
        return A_EV * temp**4 + config.rho * float(eos_energy_per_g(config, temp)) - total

    hi = max(t0, (ed0 / A_EV) ** 0.25 if ed0 > 0.0 else t0, 1.0)
    while residual(hi) < 0.0:
        hi *= 2.0
    return float(brentq(residual, 0.0, hi, rtol=1.0e-13))


def integrate_reference(run: RunData) -> tuple[np.ndarray, np.ndarray]:
    config = run.config
    t0, ed0 = INIT_SPEC[run.init]
    sigma = config.kappa_a * config.rho
    rho_factor = config.rho ** (1.0 - config.eos_mu)

    def rhs(_time, state):
        ed, temp = state
        exchange = C_LIGHT * sigma * (A_EV * temp**4 - ed)
        rho_cv = rho_factor * config.eos_f * config.eos_beta * temp ** (config.eos_beta - 1.0)
        if config.step_D > 0.0:
            rho_cv = rho_cv + config.rho * config.step_D * expit(
                (temp - config.step_Tc) / config.step_w
            )
        return (exchange, -exchange / rho_cv)

    ed_scale = max(abs(ed0), A_EV * t0**4, 1.0)
    result = solve_ivp(
        rhs,
        (0.0, float(run.history.t[-1])),
        (ed0, t0),
        method="Radau",
        t_eval=run.history.t,
        rtol=1.0e-11,
        atol=np.asarray((ed_scale, t0), dtype=float) * 1.0e-12,
    )
    if not result.success or result.y.shape != (2, run.history.t.size):
        stop(f"{run.role}: reference ODE integration failed: {result.message}")
    return result.y[0], result.y[1]


def prepare_runs(leg_paths: dict[str, Path], rho_default: float) -> dict[str, RunData]:
    runs: dict[str, RunData] = {}
    for role in REQUIRED_ROLES:
        run_dir = leg_paths[role]
        config = read_frozen_config(run_dir, role, rho_default)
        history = read_history(run_dir, config)
        runs[role] = RunData(
            role=role,
            run_dir=run_dir,
            init=ROLE_SPEC[role][1],
            config=config,
            history=history,
        )

    eqhold = runs["eqhold"]
    volume = float(eqhold.history.erad[0] / (A_EV * 150.0**4))
    if not math.isfinite(volume) or volume <= 0.0:
        stop("eqhold: radiation energy did not yield a positive finite total volume")
    for run in runs.values():
        if not close(run.config.rho, eqhold.config.rho):
            stop(f"{run.role}: rho differs from eqhold")
        run.volume = volume
        denom = volume * run.config.rho ** (1.0 - run.config.eos_mu) * run.config.eos_f
        if denom <= 0.0 or np.any(run.history.eint_e <= 0.0):
            stop(f"{run.role}: cannot invert non-positive electron internal energy")
        if run.config.step_D > 0.0:
            run.te = invert_te(
                run.config, run.history.eint_e / (volume * run.config.rho)
            )
        else:
            run.te = np.power(run.history.eint_e / denom, 1.0 / run.config.eos_beta)
        run.ed = run.history.erad / volume
    return runs


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def gate_rate(runs: dict[str, RunData]) -> tuple[bool, dict[str, float]]:
    errors: list[float] = []
    spans: list[float] = []
    for role in ("rate_dt0", "rate_dt1", "rate_dt2"):
        run = runs[role]
        _, te_ref = integrate_reference(run)
        t0, ed0 = INIT_SPEC[run.init]
        teq = equilibrium_temperature(t0, ed0, run.config)
        span = abs(t0 - teq)
        if span == 0.0:
            stop(f"{role}: zero initial-to-equilibrium temperature span")
        errors.append(abs(float(run.te[-1]) - float(te_ref[-1])) / span)
        spans.append(span)
    if not (close(spans[0], spans[1]) and close(spans[1], spans[2])):
        stop("rate legs have inconsistent initial-to-equilibrium spans")
    order = math.log2(errors[1] / errors[2]) if errors[1] > 0.0 and errors[2] > 0.0 else math.inf
    fine_ok = errors[2] < 0.04
    monotone_ok = errors[0] > errors[1] > errors[2]
    # The outer-iteration fleck recompute yields between-first-and-second-order convergence; the band must admit both.
    order_ok = 0.6 <= order <= 2.4
    ok = fine_ok and monotone_ok and order_ok
    print("(a) Rate fidelity + convergence")
    print(
        f"  errors dt,dt/2,dt/4 = {errors[0]:.8e}, {errors[1]:.8e}, {errors[2]:.8e}; "
        f"err_2 < 0.04: {status(fine_ok)}"
    )
    print(f"  monotone err_0 > err_1 > err_2: {status(monotone_ok)}")
    print(f"  observed order p = {order:.8f} in [0.6, 2.4]: {status(order_ok)}")
    print(f"  gate (a): {status(ok)}")
    return ok, {"err0": errors[0], "err1": errors[1], "err2": errors[2], "span": spans[0]}


def gate_bump(runs: dict[str, RunData]) -> bool:
    # Coefficient-isolation stress gate (design doc section 7 / external
    # verdict section 10.1): ONE Fleck step per leg (t_end == dt,
    # max_outer_iterations == 1) across a deliberately strong, table-resolved
    # c_v softstep, measured as the absolute one-step endpoint temperature
    # error against the exact Radau reference. Reduced-model pre-run forecast
    # of the ratios: ~0.215 / 0.238 / 0.413 / 0.633 (coarse -> fine).
    errors: dict[str, list[float]] = {"tan": [], "sec": []}
    diag: list[str] = []
    for family in ("tan", "sec"):
        for rung in range(4):
            run = runs[f"bump_{family}_dt{rung}"]
            ed_ref, te_ref = integrate_reference(run)
            # Radiation-side endpoint: with max_outer=1 the one-pass step is
            # non-conservative (matter gains ~2x the radiation loss, measured
            # 2026-07-15), so T reflects that surplus; E follows the coupled
            # Fleck map and isolates the beta choice.
            err = abs(float(run.ed[-1]) - float(ed_ref[-1]))
            errors[family].append(err)
            if family == "tan":
                cfg = run.config
                t0, _ = INIT_SPEC[run.init]
                sigma = cfg.kappa_a * cfg.rho
                cv0 = float(
                    cfg.eos_f * cfg.eos_beta * t0 ** (cfg.eos_beta - 1.0)
                    * cfg.rho ** (-cfg.eos_mu)
                ) + cfg.step_D / (1.0 + math.exp(-(t0 - cfg.step_Tc) / cfg.step_w))
                beta_tan = 4.0 * A_EV * t0**3 / (cfg.rho * cv0)
                t_end_ref = float(te_ref[-1])
                du = cfg.rho * (
                    float(eos_energy_per_g(cfg, t_end_ref))
                    - float(eos_energy_per_g(cfg, t0))
                )
                db = A_EV * (t_end_ref**4 - t0**4)
                beta_ch = db / du if du != 0.0 else float("nan")
                h = C_LIGHT * sigma * cfg.dt
                diag.append(
                    f"  dt={cfg.dt:.1e}: beta_tan={beta_tan:.4f} beta_ch_ref={beta_ch:.4f} "
                    f"f_tan={1.0 / (1.0 + beta_tan * h):.4f} f_ch={1.0 / (1.0 + beta_ch * h):.4f} "
                    f"T_ref={t_end_ref:.3f} eV"
                )
    ratios = [errors["sec"][k] / errors["tan"][k] for k in range(4)]
    monotone_ok = all(
        errors[f][0] > errors[f][1] > errors[f][2] > errors[f][3] for f in ("tan", "sec")
    )
    positive_ok = all(err > 0.0 for f in ("tan", "sec") for err in errors[f])
    coarse_ok = all(r <= 0.5 for r in ratios[:3])
    all_ok = all(r <= 0.7 for r in ratios)
    ok = monotone_ok and positive_ok and coarse_ok and all_ok
    print("(f) Coefficient-isolation secant stress gate [one step, max_outer=1]")
    for line in diag:
        print(line)
    for family in ("tan", "sec"):
        errs = errors[family]
        print(
            f"  {family}: |E_rad endpoint err| dt0..dt3 = "
            f"{errs[0]:.6e}, {errs[1]:.6e}, {errs[2]:.6e}, {errs[3]:.6e} erg/cc"
        )
    print(
        f"  err_sec/err_tan per rung = {ratios[0]:.4f}, {ratios[1]:.4f}, "
        f"{ratios[2]:.4f}, {ratios[3]:.4f}"
    )
    print(f"  coarse rungs <= 0.5: {status(coarse_ok)}; all rungs <= 0.7: {status(all_ok)}")
    print(f"  monotone within each family: {status(monotone_ok)}")
    print(f"  gate (f): {status(ok)}")
    return ok


def gate_form(runs: dict[str, RunData]) -> bool:
    # phi_1 form stress gate (design doc docs/design/fleck_exp_source_20260716.md
    # section 2 W2): same one-step E_rad endpoint metric as gate (f), families
    # tan (fleck_form="be" baseline, reusing the bump_tan legs) vs exp
    # (fleck_form="exp_phi1", tangent beta) — isolates the time-shape form.
    # Nonlinear Radau pre-run forecast of err_exp/err_tan: 0.508 / 0.367 /
    # 0.261 / 0.186 (coarse -> fine; scratch forecast 2026-07-16).
    errors: dict[str, list[float]] = {"tan": [], "exp": []}
    for family in ("tan", "exp"):
        for rung in range(4):
            run = runs[f"bump_{family}_dt{rung}"]
            ed_ref, _ = integrate_reference(run)
            errors[family].append(abs(float(run.ed[-1]) - float(ed_ref[-1])))
    ratios = [errors["exp"][k] / errors["tan"][k] for k in range(4)]
    monotone_ok = all(errors["exp"][k] > errors["exp"][k + 1] for k in range(3))
    positive_ok = all(err > 0.0 for family in ("tan", "exp") for err in errors[family])
    all_ok = all(r <= 0.65 for r in ratios)
    fine_ok = ratios[3] <= 0.30
    ok = monotone_ok and positive_ok and all_ok and fine_ok
    print("(g) fleck_form exp_phi1 stress gate [one step, max_outer=1]")
    for family in ("tan", "exp"):
        errs = errors[family]
        print(
            f"  {family}: |E_rad endpoint err| dt0..dt3 = "
            f"{errs[0]:.6e}, {errs[1]:.6e}, {errs[2]:.6e}, {errs[3]:.6e} erg/cc"
        )
    print(
        f"  err_exp/err_tan per rung = {ratios[0]:.4f}, {ratios[1]:.4f}, "
        f"{ratios[2]:.4f}, {ratios[3]:.4f}"
    )
    print(f"  all rungs <= 0.65: {status(all_ok)}; fine rung <= 0.30: {status(fine_ok)}")
    print(f"  monotone exp errors: {status(monotone_ok)}; positive errors: {status(positive_ok)}")
    print(f"  gate (g): {status(ok)}")
    return ok


def gate_outer_efficiency(runs: dict[str, RunData]) -> bool:
    # Outer-iteration efficiency measurement (verdict section 10.3; design doc
    # fleck_exp_source_20260716.md section 4). Measurement-first: no
    # iteration-count band on the first data set — the binding checks are
    # (1) every step of every family converged within max_outer, and
    # (2) the converged endpoints agree across families to a LOOSE band:
    # the converged fleck_cummings step retains f(beta), so different beta
    # choices legitimately converge to O(h*df)-different discrete solutions
    # (measured ~1.5e-3 of span at h~1, 2026-07-16); the 1e-2 band only
    # catches gross defects.
    fams = ("tan", "sec", "grd")
    stats: dict[str, tuple[float, int]] = {}
    converged_ok = True
    for fam in fams:
        run = runs[f"eff_{fam}"]
        iters = np.asarray(run.history.outer_iters, dtype=float)
        conv = np.asarray(run.history.outer_converged, dtype=float)
        if iters.size == 0 or np.any(iters < 1.0):
            stop(f"eff_{fam}: outer iteration history is empty or non-positive")
        if np.any(conv < 1.0):
            converged_ok = False
        stats[fam] = (float(iters.mean()), int(iters.max()))
    te_end = {fam: float(runs[f"eff_{fam}"].te[-1]) for fam in fams}
    t0, ed0 = INIT_SPEC[runs["eff_tan"].init]
    teq = equilibrium_temperature(t0, ed0, runs["eff_tan"].config)
    span = abs(t0 - teq)
    if span == 0.0:
        stop("eff legs: zero initial-to-equilibrium temperature span")
    dev_sec = abs(te_end["sec"] - te_end["tan"]) / span
    dev_grd = abs(te_end["grd"] - te_end["tan"]) / span
    endpoint_ok = dev_sec <= 1.0e-2 and dev_grd <= 1.0e-2
    ok = converged_ok and endpoint_ok
    print("(i) outer-iteration efficiency [multi-step softstep, max_outer=60]")
    for fam in fams:
        mean_it, max_it = stats[fam]
        print(f"  {fam}: outer iters mean {mean_it:.3f}, max {max_it}")
    print(
        f"  mean ratios sec/tan = {stats['sec'][0] / stats['tan'][0]:.4f}, "
        f"grd/tan = {stats['grd'][0] / stats['tan'][0]:.4f} (measurement — no band yet)"
    )
    print(f"  all steps converged in all families: {status(converged_ok)}")
    print(
        f"  beta-independent fixed point: |dTe_end|/span sec = {dev_sec:.3e}, "
        f"grd = {dev_grd:.3e} <= 1e-2: {status(endpoint_ok)}"
    )
    print(f"  gate (i): {status(ok)}")
    return ok


def gate_qexp(runs: dict[str, RunData]) -> bool:
    # gate (j): exp_rosenbrock direct-transfer stress gate (design doc
    # fleck_exp_source_20260716.md section 3 W3). Same one-step E_rad
    # endpoint metric as gates (f)/(g), family qxp vs the bump_tan BE
    # baseline. Pilot-derived forecast qxp/tan: 0.489/0.323/0.203/0.118.
    # Second binding signal: the direct transfer is exactly conservative
    # (unlike the fleck one-pass step, measured 2.1-2.3x non-conservation) —
    # the one-step total-energy drift must sit at machine level.
    errors: dict[str, list[float]] = {"tan": [], "qxp": []}
    drift_ok = True
    drift_max = 0.0
    for family in ("tan", "qxp"):
        for rung in range(4):
            run = runs[f"bump_{family}_dt{rung}"]
            ed_ref, _ = integrate_reference(run)
            errors[family].append(abs(float(run.ed[-1]) - float(ed_ref[-1])))
            if family == "qxp":
                e_tot = (
                    np.asarray(run.history.erad, dtype=float)
                    + np.asarray(run.history.eint_e, dtype=float)
                    + np.asarray(run.history.eint_i, dtype=float)
                )
                scale = float(np.max(np.abs(e_tot)))
                rel = float(np.max(np.abs(e_tot - e_tot[0]))) / max(scale, 1.0e-300)
                drift_max = max(drift_max, rel)
                if rel > 1.0e-10:
                    drift_ok = False
    ratios = [errors["qxp"][k] / errors["tan"][k] for k in range(4)]
    monotone_ok = all(errors["qxp"][k] > errors["qxp"][k + 1] for k in range(3))
    positive_ok = all(err > 0.0 for family in errors for err in errors[family])
    all_ok = all(r <= 0.60 for r in ratios)
    fine_ok = ratios[3] <= 0.20
    ok = monotone_ok and positive_ok and all_ok and fine_ok and drift_ok
    print("(j) exp_rosenbrock direct-transfer stress gate [one step]")
    for family in ("tan", "qxp"):
        errs = errors[family]
        print(
            f"  {family}: |E_rad endpoint err| dt0..dt3 = "
            f"{errs[0]:.6e}, {errs[1]:.6e}, {errs[2]:.6e}, {errs[3]:.6e} erg/cc"
        )
    print(
        f"  err_qxp/err_tan per rung = {ratios[0]:.4f}, {ratios[1]:.4f}, "
        f"{ratios[2]:.4f}, {ratios[3]:.4f}"
    )
    print(f"  all rungs <= 0.60: {status(all_ok)}; fine rung <= 0.20: {status(fine_ok)}")
    print(f"  monotone qxp errors: {status(monotone_ok)}; positive: {status(positive_ok)}")
    print(
        f"  exact-conservation drift max {drift_max:.3e} <= 1e-10: {status(drift_ok)}"
    )
    print(f"  gate (j): {status(ok)}")
    return ok


def gate_qmg(runs: dict[str, RunData]) -> bool:
    # gate (k): multigroup exp_rosenbrock stress gate (design doc
    # exp_mg_phi1_20260717.md section 3 W3). Substrate: G=4, EQUAL sigma_g,
    # sum b_g = 1, sum db_g/dT = 0 — the frozen mg system marginalizes
    # EXACTLY to the frozen grey system, so the mg TOTAL radiation endpoint
    # must match the grey exp legs (bump_qxp) to near machine precision:
    # a sharp, b_g-model-free device identity that catches rank-1/deflation
    # bugs. Secondary: exact conservation and the second-order slope vs the
    # grey Radau reference (valid for the total under marginalization).
    ident_ok = True
    drift_ok = True
    errors: list[float] = []
    ident_max = 0.0
    drift_max = 0.0
    for rung in range(4):
        mg = runs[f"bump_qmg_dt{rung}"]
        gx = runs[f"bump_qxp_dt{rung}"]
        e_mg = float(mg.ed[-1])
        e_gx = float(gx.ed[-1])
        rel = abs(e_mg - e_gx) / max(abs(e_gx), 1.0e-300)
        ident_max = max(ident_max, rel)
        if rel > 1.0e-9:
            ident_ok = False
        e_tot = (
            np.asarray(mg.history.erad, dtype=float)
            + np.asarray(mg.history.eint_e, dtype=float)
            + np.asarray(mg.history.eint_i, dtype=float)
        )
        scale = float(np.max(np.abs(e_tot)))
        drel = float(np.max(np.abs(e_tot - e_tot[0]))) / max(scale, 1.0e-300)
        drift_max = max(drift_max, drel)
        if drel > 1.0e-10:
            drift_ok = False
        ed_ref, _ = integrate_reference(mg)
        errors.append(abs(float(mg.ed[-1]) - float(ed_ref[-1])))
    slope = errors[0] / max(errors[3], 1.0e-300)
    monotone_ok = errors[0] > errors[1] > errors[2] > errors[3]
    slope_ok = slope >= 60.0  # 2nd order over 8x refinement ~ 100x; 1st ~ 8x
    ok = ident_ok and drift_ok and monotone_ok and slope_ok
    print("(k) multigroup exp_rosenbrock stress gate [G=4 equal-sigma, one step]")
    print(
        f"  E_tot vs grey-exp identity: max rel {ident_max:.3e} <= 1e-9: {status(ident_ok)}"
    )
    print(f"  exact-conservation drift max {drift_max:.3e} <= 1e-10: {status(drift_ok)}")
    print(
        f"  |E_tot endpoint err| vs grey Radau dt0..dt3 = "
        f"{errors[0]:.6e}, {errors[1]:.6e}, {errors[2]:.6e}, {errors[3]:.6e} erg/cc"
    )
    print(
        f"  monotone: {status(monotone_ok)}; slope err(dt0)/err(dt3) = {slope:.1f} >= 60: {status(slope_ok)}"
    )
    print(f"  gate (k): {status(ok)}")
    return ok


def gate_tau(runs: dict[str, RunData]) -> bool:
    run = runs["tau"]
    teq = float(run.te[-1])
    ed_inf = A_EV * teq**4
    delta = np.abs(run.ed - ed_inf)
    if not np.all(np.isfinite(delta)) or delta[0] <= 0.0:
        stop("tau: invalid initial radiation-energy decay amplitude")
    threshold = delta[0] * 1.0e-2
    crossings = np.flatnonzero(delta <= threshold)
    if crossings.size == 0:
        stop("tau: radiation-energy decay does not span two decades")
    fit_indices = np.arange(0, int(crossings[0]) + 1)
    if fit_indices.size < 3:
        stop("tau: fewer than three history samples in the first two decay decades")
    if np.any(delta[fit_indices] <= 0.0):
        stop("tau: non-positive decay amplitude inside the first two decades")
    slope, _ = np.polyfit(run.history.t[fit_indices], np.log(delta[fit_indices]), 1)
    fitted = -float(slope)
    cv_mass = run.config.eos_f * run.config.eos_beta * teq ** (run.config.eos_beta - 1.0)
    beta_id = 4.0 * A_EV * teq**3 / (run.config.rho * cv_mass)
    expected = C_LIGHT * run.config.kappa_a * run.config.rho * (1.0 + beta_id)
    rel = abs(fitted - expected) / abs(expected)
    ok = rel <= 0.05
    print("(b) Linear-perturbation relaxation time")
    print(
        f"  T_eq(late)={teq:.8e} eV, samples={fit_indices.size}, "
        f"lambda_fit={fitted:.8e} 1/s, lambda_expected={expected:.8e} 1/s, "
        f"rel_diff={rel:.8e} <= 5e-2: {status(ok)}"
    )
    print(f"  gate (b): {status(ok)}")
    return ok


def checkpoint_paths(run: RunData) -> list[Path]:
    directory = run.run_dir / "checkpoints"
    paths = sorted(directory.glob(f"{run.config.name}_ckpt_*.h5"))
    if len(paths) != 2:
        names = "\n  ".join(str(path) for path in paths) if paths else "(none)"
        stop(f"{run.role}: expected exactly two retained checkpoints in {directory}; found:\n  {names}")
    return paths


def checkpoint_state(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {
            "vol": require_dataset(handle, "hydro/vol", path),
            "te": require_dataset(handle, "hydro/Te", path),
            "ed": require_dataset(handle, "radiation/energy_density", path),
            "emit": require_dataset(handle, "radiation/rad_emit", path),
        }


def ledger_samples(
    run: RunData,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    paths = checkpoint_paths(run)
    states = [checkpoint_state(path) for path in paths]
    nr = run.config.nr
    for path, state in zip(paths, states):
        if state["vol"].shape != (nr,) or state["te"].shape != (nr,):
            stop(f"{path}: hydro/vol or hydro/Te shape does not equal ({nr},)")
        if state["ed"].shape != (nr, 1) or state["emit"].shape != (nr, 1):
            stop(f"{path}: radiation datasets do not have shape ({nr}, 1)")
    if not np.allclose(states[0]["vol"], states[1]["vol"], rtol=0.0, atol=0.0):
        stop(f"{run.role}: cell volumes changed between the two checkpoints")
    if np.any(states[0]["vol"] <= 0.0):
        stop(f"{run.role}: checkpoint contains non-positive cell volume")

    sigma = run.config.kappa_a * run.config.rho
    te_start = [np.full(nr, 200.0), states[0]["te"]]
    te_end = [states[0]["te"], states[1]["te"]]
    ed_start = [np.zeros(nr), states[0]["ed"][:, 0]]
    ledger: list[np.ndarray] = []
    expected_start: list[np.ndarray] = []
    expected_end: list[np.ndarray] = []

    def fleck_expected(temp: np.ndarray) -> np.ndarray:
        if run.config.mode == "legacy":
            cv_mass = (
                run.config.zbar
                * EV_TO_ERG
                / (run.config.material_A * PROTON_MASS * (run.config.gamma_eff - 1.0))
            )
        else:
            cv_mass = (
                run.config.eos_f
                * run.config.eos_beta
                * temp ** (run.config.eos_beta - 1.0)
            )
        beta_fleck = 4.0 * A_EV * temp**3 / (run.config.rho * cv_mass)
        return np.asarray(
            1.0 / (1.0 + beta_fleck * C_LIGHT * sigma * run.config.dt),
            dtype=float,
        )

    for step in range(2):
        # The final booked f and emission temperature are both near the end-of-step state.
        denom = A_EV * te_end[step] ** 4 - ed_start[step]
        if np.any(denom == 0.0):
            stop(f"{run.role}: zero ledger denominator at step {step + 1}")
        f_ledger = (
            states[step]["emit"][:, 0]
            / (run.config.dt * states[step]["vol"] * C_LIGHT * sigma)
            - ed_start[step]
        ) / denom
        ledger.append(f_ledger)
        expected_start.append(fleck_expected(te_start[step]))
        expected_end.append(fleck_expected(te_end[step]))
    return ledger, expected_start, expected_end


def gate_ledger(runs: dict[str, RunData]) -> bool:
    table_ledger, table_expected_start, table_expected_end = ledger_samples(runs["fled_table"])
    legacy_ledger, legacy_expected_start, legacy_expected_end = ledger_samples(
        runs["fled_legacy"]
    )

    print("(c) Fleck ledger back-solve")
    for role, ledger, expected_start, expected_end in (
        ("fled_table", table_ledger, table_expected_start, table_expected_end),
        ("fled_legacy", legacy_ledger, legacy_expected_start, legacy_expected_end),
    ):
        for step in range(2):
            print(
                f"  {role} step{step + 1}: "
                f"f_ledger={np.array2string(ledger[step], precision=8)}, "
                f"f_exp_start={np.array2string(expected_start[step], precision=8)}, "
                f"f_exp_end={np.array2string(expected_end[step], precision=8)}"
            )

    b_eff = legacy_ledger[0] / legacy_expected_end[0]
    b_eff_ok = bool(np.all((0.995 <= b_eff) & (b_eff <= 1.0)))
    legacy_rel = float(
        np.max(np.abs(legacy_ledger[1] / (b_eff * legacy_expected_end[1]) - 1.0))
    )
    table_steps = [
        float(np.max(np.abs(ledger / (b_eff * expected) - 1.0)))
        for ledger, expected in zip(table_ledger, table_expected_end)
    ]
    legacy_ok = legacy_rel <= 1.0e-6
    table_ok = max(table_steps) <= 2.0e-2
    magnitude_ok = float(np.min(table_ledger[0])) > 10.0 * float(np.max(legacy_ledger[0]))
    ok = b_eff_ok and table_ok and legacy_ok and magnitude_ok
    print(f"  b_eff={np.array2string(b_eff, precision=8)} in [0.995, 1.0]: {status(b_eff_ok)}")
    print(
        f"  table max rel diff step1/step2 = {table_steps[0]:.8e}, {table_steps[1]:.8e}; "
        f"max <= 2e-2: {status(table_ok)}"
    )
    print(
        f"  legacy step2 rel diff = {legacy_rel:.8e} <= 1e-6: {status(legacy_ok)}"
    )
    print(
        f"  step1 min(f_table)={np.min(table_ledger[0]):.8e}, "
        f"max(f_legacy)={np.max(legacy_ledger[0]):.8e}, table > 10*legacy: {status(magnitude_ok)}"
    )
    print(f"  gate (c): {status(ok)}")
    return ok


def gate_coalescence(runs: dict[str, RunData]) -> bool:
    span_run = runs["co_tab_dt0"]
    t0, ed0 = INIT_SPEC[span_run.init]
    teq = equilibrium_temperature(t0, ed0, span_run.config)
    span = abs(t0 - teq)
    if span == 0.0:
        stop("co_tab_dt0: zero initial-to-equilibrium temperature span")
    differences = []
    for table_role, legacy_role in zip(
        ("co_tab_dt0", "co_tab_dt1", "co_tab_dt2"),
        ("co_leg_dt0", "co_leg_dt1", "co_leg_dt2"),
    ):
        differences.append(abs(float(runs[legacy_role].te[-1]) - float(runs[table_role].te[-1])) / span)
    monotone_ok = differences[0] > differences[1] > differences[2]
    reduction_ok = differences[2] < 0.5 * differences[0]
    fine_ok = differences[2] < 0.05
    ok = monotone_ok and reduction_ok and fine_ok
    print("(d) Legacy/table coalescence")
    print(
        f"  d_0,d_1,d_2 = {differences[0]:.8e}, {differences[1]:.8e}, {differences[2]:.8e}; "
        f"monotone: {status(monotone_ok)}"
    )
    print(
        f"  d_2={differences[2]:.8e} < 0.5*d_0={0.5 * differences[0]:.8e}: "
        f"{status(reduction_ok)}"
    )
    print(f"  d_2={differences[2]:.8e} < 0.05: {status(fine_ok)}")
    print(f"  gate (d): {status(ok)}")
    return ok


def gate_eqhold(runs: dict[str, RunData]) -> bool:
    run = runs["eqhold"]
    erad_rel = float(np.max(np.abs(run.history.erad - run.history.erad[0])) / abs(run.history.erad[0]))
    eint_rel = float(np.max(np.abs(run.history.eint_e - run.history.eint_e[0])) / abs(run.history.eint_e[0]))
    erad_ok = erad_rel <= 1.0e-10
    eint_ok = eint_rel <= 1.0e-10
    closure = {}
    closure_ok = {}
    closure_limit = {"eqhold": 1.0e-11, "rate_dt0": 2.0e-9}
    for role in ("eqhold", "rate_dt0"):
        closure_run = runs[role]
        # qei is suppressed by A=1e5 but not zero — the transient leg leaks O(1e-6) of exchanged energy into ions, which is a real reservoir, not a conservation defect; the 3-term sum is the true closed-box ledger. The remaining rate_dt0 closure gap is the one-pass FLD outer-iteration truncation residual, not a booking defect (measured: outer_tol 1e-5 -> closure 5.4e-6; outer_tol 1e-9 -> closure 3.4e-10).
        total = closure_run.history.erad + closure_run.history.eint_e + closure_run.history.eint_i
        closure[role] = float(np.max(np.abs(total - total[0])) / total[0])
        closure_ok[role] = closure[role] <= closure_limit[role]
    ok = erad_ok and eint_ok and all(closure_ok.values())
    print("(e) Equilibrium fixed point")
    print(f"  radiation max relative drift={erad_rel:.8e} <= 1e-10: {status(erad_ok)}")
    print(f"  electron max relative drift={eint_rel:.8e} <= 1e-10: {status(eint_ok)}")
    print("  replaced energy/conservation_error because it divides by E_in ~= 0 in a closed box")
    for role in ("eqhold", "rate_dt0"):
        print(
            f"  {role} direct (E_rad + E_int_e + E_int_i) closure={closure[role]:.8e} "
            f"<= {closure_limit[role]:g}: "
            f"{status(closure_ok[role])}"
        )
    print(f"  gate (e): {status(ok)}")
    return ok


def run_check(args) -> int:
    leg_paths = parse_legs(args.leg)
    runs = prepare_runs(leg_paths, args.rho)
    print("0-D radiation-matter relaxation reference")
    print(f"a_eV={A_EV:.8e} erg/cm^3/eV^4; V_total(eqhold)={runs['eqhold'].volume:.8e} cm^3")
    print(f"rho source: {runs['eqhold'].config.rho_source}")
    print("History time dataset: t (sample 0 is post-first-step)")
    print()
    rate_ok, rate = gate_rate(runs)
    print()
    tau_ok = gate_tau(runs)
    print()
    ledger_ok = gate_ledger(runs)
    print()
    coalescence_ok = gate_coalescence(runs)
    print()
    eqhold_ok = gate_eqhold(runs)
    print()
    bump_ok = gate_bump(runs)
    print()
    form_ok = gate_form(runs)
    print()
    eff_ok = gate_outer_efficiency(runs)
    print()
    qexp_ok = gate_qexp(runs)
    print()
    qmg_ok = gate_qmg(runs)
    all_pass = rate_ok and tau_ok and ledger_ok and coalescence_ok and eqhold_ok and bump_ok and form_ok and eff_ok and qexp_ok and qmg_ok
    print()
    print(f"Overall: {status(all_pass)}")
    return 0 if all_pass else 1


def selftest() -> int:
    t0 = 100.0
    e0 = A_EV * 105.0**4
    sigma = 333.0
    rho_cv = 2.873e11
    total = e0 + rho_cv * t0

    def equilibrium_residual(temp):
        return A_EV * temp**4 + rho_cv * temp - total

    teq = float(brentq(equilibrium_residual, 0.0, 200.0, rtol=1.0e-14))
    beta = 4.0 * A_EV * teq**3 / rho_cv
    expected = C_LIGHT * sigma * (1.0 + beta)

    def rhs(_time, state):
        ed, temp = state
        exchange = C_LIGHT * sigma * (A_EV * temp**4 - ed)
        return (exchange, -exchange / rho_cv)

    times = np.linspace(0.0, 0.5 / expected, 101)
    scales = np.asarray((max(e0, A_EV * t0**4), t0), dtype=float)
    result = solve_ivp(
        rhs,
        (0.0, float(times[-1])),
        (e0, t0),
        method="Radau",
        t_eval=times,
        rtol=1.0e-11,
        atol=1.0e-3 * scales,
    )
    if not result.success:
        print(f"SELFTEST FAIL: Radau integration failed: {result.message}")
        return 1
    delta = result.y[0] - A_EV * result.y[1] ** 4
    if np.any(delta == 0.0):
        print("SELFTEST FAIL: zero delta in early-time fit")
        return 1
    slope, _ = np.polyfit(times, np.log(np.abs(delta)), 1)
    fitted = -float(slope)
    rel = abs(fitted - expected) / abs(expected)
    ok = rel <= 0.01
    print("0-D constant-cv two-ODE selftest")
    print(
        f"T_eq={teq:.8e} eV, lambda_fit={fitted:.8e} 1/s, "
        f"lambda_linear={expected:.8e} 1/s, rel_diff={rel:.8e}"
    )
    print(f"SELFTEST {status(ok)}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument(
        "--leg",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="run root tagged with one of the required roles (repeatable)",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.2,
        help="uniform rho fallback when frozen geometry.rho is a callable [g/cm^3]",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.selftest:
            return selftest()
        if not math.isfinite(args.rho) or args.rho <= 0.0:
            stop("--rho must be positive and finite")
        return run_check(args)
    except StopCheck as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
