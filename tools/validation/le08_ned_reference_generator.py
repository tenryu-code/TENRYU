#!/usr/bin/env python3
"""Generate frozen LE08 nonequilibrium-diffusion grey shock references.

ExactPack is an optional regeneration dependency only.  The frozen JSON tables
emitted by this tool are the verification dependency used by ctest; TENRYU
runtime and the ctest harness do not import ExactPack.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


C_CGS = 2.99792458e10
A_EV = 1.3720e2
EV_TO_ERG = 1.6022e-12
PROTON_MASS_G = 1.6726219e-24
E_FLOOR = 1.0e-300
LOW_PRAD_HARD_GATE = 1.0e-3
TARGET_DOMAIN_SPAN_CM = 50.0
PADDING_POINTS_PER_SIDE = 8


@dataclass(frozen=True)
class LE08Config:
    mach: float
    gamma: float = 5.0 / 3.0
    rho0_gcc: float = 2.0
    T0_eV: float = 30.0
    kappa_R_cm2_g: float = 0.5
    A_amu: float = 1.0
    zbar: float = 1.0
    n_points: int = 1601
    target_domain_span_cm: float = TARGET_DOMAIN_SPAN_CM
    padding_points_per_side: int = PADDING_POINTS_PER_SIDE
    problem: str = "LM_nED"
    epsilon: float = 1.0
    int_tol: float = 1.0e-10


def safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def gas_cv(cfg: LE08Config) -> float:
    r_gas = (1.0 + cfg.zbar) * EV_TO_ERG / (cfg.A_amu * PROTON_MASS_G)
    return r_gas / (cfg.gamma - 1.0)


def validate_config(cfg: LE08Config) -> None:
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
    if cfg.target_domain_span_cm <= 0.0:
        raise ValueError("target_domain_span_cm must be positive")
    if cfg.padding_points_per_side < 1:
        raise ValueError("padding_points_per_side must be positive")
    if cfg.problem == "nED":
        return
    if cfg.problem != "LM_nED":
        raise ValueError("LE08 generation requires problem='LM_nED' (or explicit FML problem='nED')")


def reference_model(problem: str) -> str:
    if problem == "LM_nED":
        return "LE08_nED_LM_low_Prad"
    if problem == "nED":
        return "FML_nED"
    return f"UNKNOWN_{problem}"


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


def exactpack_git_commit(module_file: str | None) -> str | None:
    if not module_file:
        return None
    path = Path(module_file).resolve()
    for parent in [path.parent, *path.parents]:
        if (parent / ".git").exists():
            return git_commit(parent)
    return None


def instantiate_exactpack(cfg: LE08Config) -> tuple[Any, dict[str, Any]]:
    try:
        import exactpack
        from exactpack.solvers.radshocks import nED_Solver
    except ImportError as exc:
        raise RuntimeError(
            "ExactPack is required only to regenerate LE08 tables; install with "
            "pip install git+https://github.com/lanl/ExactPack.git@master"
        ) from exc

    if cfg.problem == "LM_nED":
        params = getattr(nED_Solver, "parameters", {})
        problem_doc = str(params.get("problem", ""))
        if "LM_nED" not in problem_doc:
            raise RuntimeError(
                "ExactPack nED_Solver does not advertise problem='LM_nED'; "
                "refusing to fall back to problem='nED' because that is FML, not LE08."
            )

    kwargs = {
        "M0": cfg.mach,
        "rho0": cfg.rho0_gcc,
        "Tref": cfg.T0_eV,
        "Cv": gas_cv(cfg),
        "gamma": cfg.gamma,
        "sigA": cfg.kappa_R_cm2_g * cfg.rho0_gcc,
        "sigS": 0.0,
        "expDensity_abs": 1.0,
        "expTemp_abs": 0.0,
        "expDensity_scat": 1.0,
        "expTemp_scat": 0.0,
        "problem": cfg.problem,
        "epsilon": cfg.epsilon,
        "int_tol": cfg.int_tol,
    }
    stdout = io.StringIO()
    stderr = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            solver = nED_Solver(**kwargs)
    if cfg.problem == "LM_nED" and getattr(solver, "problem", None) != "LM_nED":
        raise RuntimeError(
            f"ExactPack returned problem={getattr(solver, 'problem', None)!r}; "
            "refusing to label this output as LE08."
        )
    meta = {
        "exactpack_version": getattr(exactpack, "__version__", None),
        "exactpack_module": getattr(exactpack, "__file__", None),
        "exactpack_git_commit": exactpack_git_commit(getattr(exactpack, "__file__", None)),
        "exactpack_kwargs": kwargs,
        "integrator_warnings": [str(item.message) for item in caught],
        "solver_stdout": stdout.getvalue()[-4000:],
        "solver_stderr": stderr.getvalue()[-4000:],
    }
    return solver, meta


def unique_sorted_solver_arrays(solver: Any) -> dict[str, np.ndarray]:
    required = ("x", "Tm", "Tr", "Fr", "Density", "Speed", "Pressure", "RADE")
    arrays: dict[str, np.ndarray] = {}
    for name in required:
        if not hasattr(solver, name):
            raise RuntimeError(f"ExactPack nED_Solver missing required attribute {name}")
        arrays[name] = np.asarray(getattr(solver, name), dtype=float)
    order = np.argsort(arrays["x"])
    arrays = {key: value[order] for key, value in arrays.items()}
    x = arrays["x"]
    unique = np.concatenate(([True], np.diff(x) > 1.0e-10))
    return {key: value[unique] for key, value in arrays.items()}


def sample_branch_indices(count: int, n_samples: int) -> np.ndarray:
    if count <= 0:
        raise RuntimeError("empty ExactPack branch")
    return np.unique(np.round(np.linspace(0, count - 1, n_samples)).astype(int))


def exactpack_sound_speed(solver: Any) -> float:
    if hasattr(solver, "Sound_Speed"):
        sound = np.asarray(getattr(solver, "Sound_Speed"), dtype=float)
        finite = sound[np.isfinite(sound) & (sound > 0.0)]
        if finite.size:
            return float(finite[0])
    density = np.asarray(getattr(solver, "Density"), dtype=float)
    speed = np.asarray(getattr(solver, "Speed"), dtype=float)
    finite = (np.isfinite(density) & np.isfinite(speed) & (density > 0.0))
    mass_flux = density[finite] * speed[finite]
    if mass_flux.size:
        return float(np.median(mass_flux) / getattr(solver, "M0"))
    raise RuntimeError("could not infer ExactPack upstream sound speed")


def table_from_solver(solver: Any, cfg: LE08Config) -> dict[str, list[Any]]:
    arrays = unique_sorted_solver_arrays(solver)
    x_all = arrays["x"]
    n_up = cfg.n_points // 2
    n_dn = cfg.n_points - n_up
    sound = exactpack_sound_speed(solver)
    rows: list[tuple[str, int]] = []
    for branch, mask, n_branch in (
        ("upstream", x_all < 0.0, n_up),
        ("downstream", x_all > 0.0, n_dn),
    ):
        branch_indices = np.nonzero(mask)[0]
        selected = branch_indices[sample_branch_indices(branch_indices.size, n_branch)]
        rows.extend((branch, int(index)) for index in selected)
    rows.sort(key=lambda item: float(x_all[item[1]]))

    table: dict[str, list[Any]] = {
        "x_cm": [],
        "rho_g_per_cc": [],
        "u_cm_per_s": [],
        "T_eV": [],
        "P_mat_dyne_per_cm2": [],
        "P_rad_dyne_per_cm2": [],
        "F_rad_erg_per_cm2_s": [],
        "F_rad_energy_invariant_erg_per_cm2_s": [],
        "T_rad_eV": [],
        "E_rad_erg_per_cm3": [],
        "F_rad_lab_erg_per_cm2_s": [],
        "branch": [],
    }
    for branch, index in rows:
        rho = float(arrays["Density"][index])
        u = float(arrays["Speed"][index])
        e_rad = float(arrays["RADE"][index])
        f_lab = float(arrays["Fr"][index]) * sound
        # ExactPack's Fr is the lab-frame radiation energy flux divided by the
        # sound speed.  The FLD identity field is the comoving diffusion flux.
        f_diff = f_lab - (4.0 / 3.0) * u * e_rad
        table["x_cm"].append(float(arrays["x"][index]))
        table["rho_g_per_cc"].append(rho)
        table["u_cm_per_s"].append(u)
        table["T_eV"].append(float(arrays["Tm"][index]))
        table["P_mat_dyne_per_cm2"].append(float(arrays["Pressure"][index]))
        table["P_rad_dyne_per_cm2"].append(e_rad / 3.0)
        table["F_rad_erg_per_cm2_s"].append(f_diff)
        table["F_rad_energy_invariant_erg_per_cm2_s"].append(f_diff if branch == "upstream" else -f_diff)
        table["T_rad_eV"].append(float(arrays["Tr"][index]))
        table["E_rad_erg_per_cm3"].append(e_rad)
        table["F_rad_lab_erg_per_cm2_s"].append(f_lab)
        table["branch"].append(branch)
    return table


def table_array(table: dict[str, list[Any]], key: str) -> np.ndarray:
    return np.asarray(table[key], dtype=float)


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
        grad_mat = np.gradient(p_mat[mask], x[mask])
        grad_rad = np.gradient(p_rad[mask], x[mask])
        scale = float(np.nanmax(np.abs(grad_mat)))
        threshold = max(1.0e-12 * scale, E_FLOOR)
        local = np.isfinite(grad_mat) & np.isfinite(grad_rad) & (np.abs(grad_mat) >= threshold)
        ratios.extend(
            float(abs(gr) / max(abs(gm), E_FLOOR))
            for gm, gr in zip(grad_mat[local], grad_rad[local])
        )
    return max(ratios) if ratios else math.inf


def branch_identity_stats(table: dict[str, list[Any]], cfg: LE08Config) -> dict[str, dict[str, float]]:
    x = table_array(table, "x_cm")
    rho = table_array(table, "rho_g_per_cc")
    e_rad = table_array(table, "E_rad_erg_per_cm3")
    f_ref = table_array(table, "F_rad_erg_per_cm2_s")
    branches = np.asarray(table["branch"], dtype=str)
    out: dict[str, dict[str, float]] = {}
    all_rels: list[np.ndarray] = []
    for branch in ("upstream", "downstream"):
        mask = branches == branch
        order = np.argsort(x[mask])
        x_b = x[mask][order]
        rho_b = rho[mask][order]
        e_b = e_rad[mask][order]
        f_b = f_ref[mask][order]
        d_e = np.gradient(e_b, x_b)
        f_grad = -C_CGS / (3.0 * cfg.kappa_R_cm2_g * np.maximum(rho_b, E_FLOOR)) * d_e
        rel = np.abs((f_b - f_grad) / np.maximum(np.abs(f_b) + np.abs(f_grad), E_FLOOR))
        interior = rel[10:-10] if rel.size > 20 else rel
        all_rels.append(interior)
        out[branch] = {
            "median_rel": float(np.median(interior)),
            "p95_rel": float(np.percentile(interior, 95.0)),
        }
    total = np.concatenate(all_rels)
    out["total"] = {
        "median_rel": float(np.median(total)),
        "p95_rel": float(np.percentile(total, 95.0)),
    }
    return out


def conservation_diagnostics(table: dict[str, list[Any]], cfg: LE08Config) -> dict[str, float]:
    rho = table_array(table, "rho_g_per_cc")
    u = table_array(table, "u_cm_per_s")
    p = table_array(table, "P_mat_dyne_per_cm2")
    pr = table_array(table, "P_rad_dyne_per_cm2")
    f_lab = table_array(table, "F_rad_lab_erg_per_cm2_s")
    e_int = p / np.maximum(rho, E_FLOOR) / (cfg.gamma - 1.0)
    mass = rho * u
    mom = rho * u * u + p + pr
    energy = rho * u * (e_int + 0.5 * u * u + p / np.maximum(rho, E_FLOOR)) + f_lab

    def residual(values: np.ndarray) -> float:
        ref = float(values[0])
        return float(np.nanmax(np.abs(values - ref)) / max(abs(ref), E_FLOOR))

    prad_ratio = float(np.nanmax(np.abs(pr) / np.maximum(np.abs(p), E_FLOOR)))
    grad_ratio = max_gradient_ratio(table)
    identity = branch_identity_stats(table, cfg)
    temp_sep = np.abs(table_array(table, "T_rad_eV") / np.maximum(table_array(table, "T_eV"), E_FLOOR) - 1.0)
    mass_res = residual(mass)
    mom_res = residual(mom)
    energy_res = residual(energy)
    return {
        "mass_flux_residual_rel_max": mass_res,
        "momentum_flux_residual_rel_max": mom_res,
        "energy_flux_residual_rel_max": energy_res,
        "conservation_residual_rel_max": max(mass_res, mom_res, energy_res),
        "max_P_rad_over_P_mat": prad_ratio,
        "max_abs_grad_P_rad_over_abs_grad_P_mat": grad_ratio,
        "max_abs_T_rad_over_T_e_minus_1": float(np.nanmax(temp_sep)),
        "reference_identity_upstream_median_rel": identity["upstream"]["median_rel"],
        "reference_identity_upstream_p95_rel": identity["upstream"]["p95_rel"],
        "reference_identity_downstream_median_rel": identity["downstream"]["median_rel"],
        "reference_identity_downstream_p95_rel": identity["downstream"]["p95_rel"],
    }


def state_from_table(table: dict[str, list[Any]], index: int) -> dict[str, float]:
    return {
        "rho": float(table["rho_g_per_cc"][index]),
        "u": float(table["u_cm_per_s"][index]),
        "T": float(table["T_eV"][index]),
        "T_rad": float(table["T_rad_eV"][index]),
        "E_rad": float(table["E_rad_erg_per_cm3"][index]),
        "P_mat": float(table["P_mat_dyne_per_cm2"][index]),
        "P_rad": float(table["P_rad_dyne_per_cm2"][index]),
        "F_rad": float(table["F_rad_erg_per_cm2_s"][index]),
        "F_rad_lab": float(table["F_rad_lab_erg_per_cm2_s"][index]),
    }


def linspace(a: float, b: float, n: int) -> list[float]:
    if n < 2:
        return [float(a)]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def equilibrium_row(x_cm: float, state: dict[str, float], branch: str) -> dict[str, Any]:
    f_energy = state["F_rad"] if branch == "upstream" else -state["F_rad"]
    return {
        "x_cm": x_cm,
        "rho_g_per_cc": state["rho"],
        "u_cm_per_s": state["u"],
        "T_eV": state["T"],
        "P_mat_dyne_per_cm2": state["P_mat"],
        "P_rad_dyne_per_cm2": state["P_rad"],
        "F_rad_erg_per_cm2_s": state["F_rad"],
        "F_rad_energy_invariant_erg_per_cm2_s": f_energy,
        "T_rad_eV": state["T_rad"],
        "E_rad_erg_per_cm3": state["E_rad"],
        "F_rad_lab_erg_per_cm2_s": state["F_rad_lab"],
        "branch": branch,
    }


def pad_table_to_domain(
    table: dict[str, list[Any]],
    upstream: dict[str, float],
    downstream: dict[str, float],
    cfg: LE08Config,
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


def build_reference(cfg: LE08Config) -> dict[str, Any]:
    validate_config(cfg)
    solver, exactpack_meta = instantiate_exactpack(cfg)
    table = table_from_solver(solver, cfg)
    upstream = state_from_table(table, 0)
    downstream = state_from_table(table, len(table["x_cm"]) - 1)
    table = pad_table_to_domain(table, upstream, downstream, cfg)
    diagnostics = conservation_diagnostics(table, cfg)
    if diagnostics["max_P_rad_over_P_mat"] > LOW_PRAD_HARD_GATE:
        raise RuntimeError(
            "low-P_rad hard gate failed: "
            f"max_P_rad/P_mat={diagnostics['max_P_rad_over_P_mat']:.6e} > {LOW_PRAD_HARD_GATE:.1e}"
        )
    payload: dict[str, Any] = {
        "schema_version": 3,
        "reference_model": reference_model(cfg.problem),
        "problem": cfg.problem,
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "tools/validation/le08_ned_reference_generator.py",
            "git_commit": git_commit(),
            "exactpack_version": exactpack_meta["exactpack_version"],
            "exactpack_git_commit": exactpack_meta["exactpack_git_commit"],
            "exactpack_module": exactpack_meta["exactpack_module"],
            "exactpack_problem": cfg.problem,
            "reference_model": reference_model(cfg.problem),
            "low_P_rad_hard_gate": LOW_PRAD_HARD_GATE,
            "low_P_rad_diagnostics": {
                "max_P_rad_over_P_mat": diagnostics["max_P_rad_over_P_mat"],
                "max_abs_grad_P_rad_over_abs_grad_P_mat": diagnostics[
                    "max_abs_grad_P_rad_over_abs_grad_P_mat"
                ],
            },
            "integrator_warnings": exactpack_meta["integrator_warnings"],
            "exactpack_stdout_tail": exactpack_meta["solver_stdout"],
            "exactpack_stderr_tail": exactpack_meta["solver_stderr"],
            "flux_convention": (
                "F_rad_erg_per_cm2_s is the comoving diffusion flux "
                "ExactPack_Fr*sound-(4/3)uE_rad conjugate to monotone x_cm. "
                "F_rad_lab_erg_per_cm2_s preserves ExactPack_Fr*sound for "
                "full radiation-energy conservation diagnostics."
            ),
            "references": [
                {
                    "label": "Lowrie and Edwards 2008",
                    "title": "Radiative shock solutions with grey nonequilibrium diffusion",
                    "doi": "10.1007/s00193-008-0143-0",
                }
            ],
        },
        "parameters": {
            "M0": cfg.mach,
            "gamma": cfg.gamma,
            "rho0_g_per_cc": cfg.rho0_gcc,
            "T0_eV": cfg.T0_eV,
            "kappa_R_cm2_per_g": cfg.kappa_R_cm2_g,
            "exactpack_sigA_cm_inv_at_rho0": cfg.kappa_R_cm2_g * cfg.rho0_gcc,
            "A_amu": cfg.A_amu,
            "zbar": cfg.zbar,
            "n_points": len(table["x_cm"]),
            "solver_sample_points": cfg.n_points,
            "target_domain_span_cm": cfg.target_domain_span_cm,
            "padding_points_per_side": cfg.padding_points_per_side,
            "problem": cfg.problem,
            "reference_model": reference_model(cfg.problem),
            "epsilon": cfg.epsilon,
            "int_tol": cfg.int_tol,
            "a_eV_erg_cm3_eV4": A_EV,
            "c_cm_per_s": C_CGS,
            "eV_to_erg": EV_TO_ERG,
            "proton_mass_g": PROTON_MASS_G,
            "Cv_erg_per_g_eV": gas_cv(cfg),
            "C0": float(getattr(solver, "C0")),
            "P0": float(getattr(solver, "P0")),
        },
        "states": {
            "upstream": state_from_table(table, 0),
            "downstream": state_from_table(table, len(table["x_cm"]) - 1),
        },
        "diagnostics": diagnostics,
        "reference_identity_branch_wise": branch_identity_stats(table, cfg),
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
    parser.add_argument("--out-dir", default="tests/verification/data/le08_ned_reference")
    parser.add_argument("--mach-list", type=parse_mach_list, default=parse_mach_list("1.5,2.0,3.0"))
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--rho0-gcc", type=float, default=2.0)
    parser.add_argument("--T0-eV", type=float, default=30.0)
    parser.add_argument("--kappa-R-cm2-g", type=float, default=0.5)
    parser.add_argument("--A-amu", type=float, default=1.0)
    parser.add_argument("--zbar", type=float, default=1.0)
    parser.add_argument("--n-points", type=int, default=1601)
    parser.add_argument("--target-domain-span-cm", type=float, default=TARGET_DOMAIN_SPAN_CM)
    parser.add_argument("--padding-points-per-side", type=int, default=PADDING_POINTS_PER_SIDE)
    parser.add_argument("--problem", default="LM_nED", help="ExactPack nED problem label; LE08 requires LM_nED")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--int-tol", type=float, default=1.0e-10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    rows: list[dict[str, Any]] = []
    for mach in args.mach_list:
        cfg = LE08Config(
            mach=mach,
            gamma=args.gamma,
            rho0_gcc=args.rho0_gcc,
            T0_eV=args.T0_eV,
            kappa_R_cm2_g=args.kappa_R_cm2_g,
            A_amu=args.A_amu,
            zbar=args.zbar,
            n_points=args.n_points,
            target_domain_span_cm=args.target_domain_span_cm,
            padding_points_per_side=args.padding_points_per_side,
            problem=args.problem,
            epsilon=args.epsilon,
            int_tol=args.int_tol,
        )
        payload = build_reference(cfg)
        filename = f"le08_nED_M{safe_float_token(mach)}.json"
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
                "solver_sample_points": cfg.n_points,
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
                "exactpack_version": payload["provenance"]["exactpack_version"],
                "generated_utc": payload["provenance"]["generated_utc"],
            }
        )
        print(
            f"{filename}: problem={payload['problem']} "
            f"domain_span={payload['table']['x_cm'][-1] - payload['table']['x_cm'][0]:.6e} "
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
        "schema_version": "tenryu.le08_ned_reference_manifest.v3",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/validation/le08_ned_reference_generator.py",
        "git_commit": git_commit(),
        "rows": rows,
    }
    write_json(out_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
