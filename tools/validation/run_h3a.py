#!/usr/bin/env python3
"""Run Stage H3-A 2D RZ cylindrical Sedov line-blast validation."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np


STAGE_ID = "H3"
STAGE_LABEL = "H3A"

GAMMA = 5.0 / 3.0
RHO_0_GCC = 1.0
E_PER_CM_DEFAULT = 1.0e15
BETA_2 = 1.0

R_MIN_CM = 0.0
R_MAX_CM = 1.0
Z_MIN_CM = -0.02
Z_MAX_CM = 0.02
P_AMBIENT_ERG_PER_CC = 1.0e6
BLAST_CELLS_DEFAULT = 8

BETA_OBSERVED_MIN = 0.9
BETA_OBSERVED_MAX = 1.3
ENERGY_ERR_PCT_MAX = 0.5
RHO_POST_ERR_PCT_MAX = 30.0
T_ALPHA_ERR_MAX = 0.10


def parse_mesh_list(value: str) -> list[tuple[int, int]]:
    pairs = []
    for part in value.split(";"):
        fields = [item.strip() for item in part.split(",")]
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise argparse.ArgumentTypeError(f"invalid mesh entry: {part!r}")
        nr = int(fields[0])
        nz = int(fields[1])
        if nr <= 0 or nz <= 0:
            raise argparse.ArgumentTypeError("mesh dimensions must be positive")
        pairs.append((nr, nz))
    if not pairs:
        raise argparse.ArgumentTypeError("mesh list is empty")
    return pairs


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("seed list is empty")
    return seeds


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("float list is empty")
    return values


def parse_ale_every_n_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("ALE every_n list is empty")
    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("ALE every_n entries must be non-negative")
    return values


def float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p")


def case_name(nr: int, nz: int, seed: int, av_c2: float, cfl: float, ale_every: int) -> str:
    return (
        f"2d_rz_h3a_sedov_cyl_nr{nr}_nz{nz}_seed{seed}"
        f"_C2{float_token(av_c2)}_cfl{float_token(cfl)}_ale{ale_every}"
    )


def run_id_for(nr: int, nz: int, seed: int, av_c2: float, cfl: float, ale_every: int) -> str:
    return (
        f"h3a_nr{nr}_nz{nz}_C2{float_token(av_c2)}"
        f"_cfl{float_token(cfl)}_ale{ale_every}_seed{seed}"
    )


def output_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def read_scalar(handle: h5py.File, name: str) -> float:
    return float(np.asarray(handle[name][()]))


def read_int_scalar(handle: h5py.File, name: str) -> int:
    return int(np.asarray(handle[name][()]))


def discover_results_dir(outdir: Path) -> Path:
    def _has_numbered_plot(d: Path) -> bool:
        return any(len(p.stem) >= 5 and p.stem[-4:].isdigit() and p.stem[-5] == "_" for p in d.glob("*.h5"))

    results_dir = outdir / "results"
    if _has_numbered_plot(results_dir):
        return results_dir

    candidates = []
    for path in outdir.parent.glob(f"{outdir.name}_*"):
        candidate = path / "results"
        if _has_numbered_plot(candidate):
            candidates.append(candidate)
    if candidates:
        return max(candidates, key=lambda path: path.parent.stat().st_mtime)
    raise FileNotFoundError(f"no plot files found under {outdir} or {outdir}_*/results")


def locate_plot_files(results_dir: Path, name: str) -> list[Path]:
    def plot_index(path: Path) -> int:
        suffix = path.stem.removeprefix(f"{name}_")
        return int(suffix)

    return sorted(
        (p for p in results_dir.glob(f"{name}_*.h5") if p.stem.removeprefix(f"{name}_").isdigit()),
        key=plot_index,
    )


def sedov_shock_radius_cm(
    t_s: float,
    E_per_cm: float = E_PER_CM_DEFAULT,
    rho_0: float = RHO_0_GCC,
    beta: float = BETA_2,
) -> float:
    return beta * (E_per_cm / rho_0) ** 0.25 * t_s**0.5


def sedov_shock_velocity(
    t_s: float,
    E_per_cm: float = E_PER_CM_DEFAULT,
    rho_0: float = RHO_0_GCC,
    beta: float = BETA_2,
) -> float:
    R_s = sedov_shock_radius_cm(t_s, E_per_cm, rho_0, beta)
    return 0.5 * R_s / t_s


def sedov_post_shock_density() -> float:
    return RHO_0_GCC * (GAMMA + 1.0) / (GAMMA - 1.0)


def sedov_post_shock_pressure(
    t_s: float,
    E_per_cm: float = E_PER_CM_DEFAULT,
    rho_0: float = RHO_0_GCC,
    beta: float = BETA_2,
) -> float:
    Rdot = sedov_shock_velocity(t_s, E_per_cm, rho_0, beta)
    return (2.0 / (GAMMA + 1.0)) * rho_0 * Rdot * Rdot


def as_cell_field(handle: h5py.File, name: str, nr: int, nz: int) -> np.ndarray:
    data = np.asarray(handle[name][()])
    if data.shape == (nr, nz):
        return data.astype(float, copy=False)
    if data.size == nr * nz:
        return data.reshape((nr, nz)).astype(float, copy=False)
    raise ValueError(f"{name} has shape {data.shape}, expected {(nr, nz)} or flat {nr * nz}")


def as_node_field(handle: h5py.File, name: str, nr: int, nz: int) -> np.ndarray:
    data = np.asarray(handle[name][()])
    if data.shape == (nr + 1, nz + 1):
        return data.astype(float, copy=False)
    if data.size == (nr + 1) * (nz + 1):
        return data.reshape((nr + 1, nz + 1)).astype(float, copy=False)
    raise ValueError(f"{name} has shape {data.shape}, expected {(nr + 1, nz + 1)}")


def field_to_cells(handle: h5py.File, name: str, nr: int, nz: int) -> np.ndarray:
    data = np.asarray(handle[name][()])
    if data.shape == (nr, nz) or data.size == nr * nz:
        return as_cell_field(handle, name, nr, nz)
    if data.shape == (nr + 1, nz + 1) or data.size == (nr + 1) * (nz + 1):
        nodes = as_node_field(handle, name, nr, nz)
        return 0.25 * (nodes[:-1, :-1] + nodes[1:, :-1] + nodes[:-1, 1:] + nodes[1:, 1:])
    raise ValueError(f"{name} has shape {data.shape}, cannot map to cells")


def read_pressure(handle: h5py.File, nr: int, nz: int) -> np.ndarray:
    if "hydro/P" in handle:
        return as_cell_field(handle, "hydro/P", nr, nz)
    if "hydro/Pi+Pe" in handle:
        return as_cell_field(handle, "hydro/Pi+Pe", nr, nz)
    if "hydro/Pi" in handle and "hydro/Pe" in handle:
        return as_cell_field(handle, "hydro/Pi", nr, nz) + as_cell_field(handle, "hydro/Pe", nr, nz)
    raise KeyError("no pressure field found; expected hydro/P, hydro/Pi+Pe, or hydro/Pi plus hydro/Pe")


def cell_centers(handle: h5py.File, nr: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    x_r = as_node_field(handle, "mesh/x_r", nr, nz)
    x_z = as_node_field(handle, "mesh/x_z", nr, nz)
    r_cell = 0.25 * (x_r[:-1, :-1] + x_r[1:, :-1] + x_r[:-1, 1:] + x_r[1:, 1:])
    z_cell = 0.25 * (x_z[:-1, :-1] + x_z[1:, :-1] + x_z[:-1, 1:] + x_z[1:, 1:])
    return r_cell, z_cell


def blast_radius_cm(nr: int, blast_cells: int = BLAST_CELLS_DEFAULT) -> float:
    return blast_cells * (R_MAX_CM - R_MIN_CM) / nr


def box_volume_cm3() -> float:
    return math.pi * (R_MAX_CM * R_MAX_CM - R_MIN_CM * R_MIN_CM) * (Z_MAX_CM - Z_MIN_CM)


def initial_energy_analytic_erg(E_per_cm: float = E_PER_CM_DEFAULT) -> float:
    e_ambient = P_AMBIENT_ERG_PER_CC / (GAMMA - 1.0) * box_volume_cm3()
    e_blast = E_per_cm * (Z_MAX_CM - Z_MIN_CM)
    return e_ambient + e_blast


def shock_radius(r_axis: np.ndarray, rho_radial: np.ndarray, r_blast_cm: float) -> float:
    if r_axis.size < 2:
        raise ValueError("at least two radial cells are required to locate the shock")
    grad = np.abs(np.gradient(rho_radial, r_axis))
    mask = (r_axis >= r_blast_cm) & (r_axis <= 0.7)
    if not np.any(mask):
        mask = (r_axis >= 0.0) & (r_axis <= 0.7)
    if not np.any(mask):
        mask = np.ones_like(r_axis, dtype=bool)
    indices = np.nonzero(mask)[0]
    return float(r_axis[indices[int(np.argmax(grad[indices]))]])


def postshock_annulus_mean(
    r_axis: np.ndarray,
    values: np.ndarray,
    shock_radius_sim_cm: float,
    nr: int,
    r_blast_cm: float,
) -> float:
    dR = (R_MAX_CM - R_MIN_CM) / nr
    lo = shock_radius_sim_cm - 4.0 * dR
    hi = shock_radius_sim_cm - 1.0 * dR
    mask = (r_axis >= lo) & (r_axis <= hi)
    if not np.any(mask):
        mask = (r_axis >= max(r_blast_cm, shock_radius_sim_cm - 6.0 * dR)) & (
            r_axis < shock_radius_sim_cm
        )
    if not np.any(mask):
        raise RuntimeError("could not identify post-shock annulus cells")
    return float(np.mean(values[mask]))


def percent_error(sim: float, analytic: float) -> float:
    return abs(sim - analytic) / max(abs(analytic), 1.0e-300) * 100.0


def total_energy_from_plot(handle: h5py.File, nr: int, nz: int) -> tuple[float, float, float]:
    mass = as_cell_field(handle, "hydro/mass", nr, nz)
    ee = as_cell_field(handle, "hydro/ee", nr, nz)
    ei = as_cell_field(handle, "hydro/ei", nr, nz)

    if "hydro/node_mass" in handle:
        v_r = as_node_field(handle, "mesh/v_r", nr, nz)
        v_z = as_node_field(handle, "mesh/v_z", nr, nz)
        node_mass = as_node_field(handle, "hydro/node_mass", nr, nz)
        kinetic = float(0.5 * np.sum(node_mass * (v_r * v_r + v_z * v_z), dtype=np.float64))
    else:
        warnings.warn(
            "hydro/node_mass missing; using legacy cell-averaged kinetic energy",
            RuntimeWarning,
        )
        v_r_cell = field_to_cells(handle, "mesh/v_r", nr, nz)
        v_z_cell = field_to_cells(handle, "mesh/v_z", nr, nz)
        kinetic = float(np.sum(0.5 * mass * (v_r_cell * v_r_cell + v_z_cell * v_z_cell)))
    internal = float(np.sum(mass * (ee + ei)))
    return kinetic + internal, kinetic, internal


def radial_profiles_from_plot(path: Path, nr: int, nz: int) -> tuple[float, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        r_cell, _ = cell_centers(handle, nr, nz)
        rho = as_cell_field(handle, "hydro/rho", nr, nz)
        t_s = read_scalar(handle, "time_state/t")
    return t_s, r_cell.mean(axis=1), rho.mean(axis=1)


def shock_history_alpha(plot_files: list[Path], nr: int, nz: int) -> float | None:
    r_blast_cm = blast_radius_cm(nr)
    self_similar_R_threshold = 2.0 * r_blast_cm
    times = []
    radii = []
    for path in plot_files:
        try:
            t_s, r_axis, rho_radial = radial_profiles_from_plot(path, nr, nz)
            if t_s <= 0.0:
                continue
            r_s = shock_radius(r_axis, rho_radial, r_blast_cm)
        except (OSError, KeyError, ValueError, RuntimeError):
            continue
        if r_s <= self_similar_R_threshold:
            continue
        if r_s > 0.0 and math.isfinite(r_s):
            times.append(t_s)
            radii.append(r_s)
    if len(times) < 2:
        return None
    coeff = np.polyfit(np.log(np.asarray(times)), np.log(np.asarray(radii)), 1)
    return float(coeff[0])


def analyze_plot(path: Path, nr: int, nz: int, ale_every: int) -> dict[str, float | int | None]:
    r_blast_cm = blast_radius_cm(nr)
    with h5py.File(path, "r") as handle:
        r_cell, _ = cell_centers(handle, nr, nz)
        rho_cell = as_cell_field(handle, "hydro/rho", nr, nz)
        pressure_cell = read_pressure(handle, nr, nz)
        Te_cell = as_cell_field(handle, "hydro/Te", nr, nz)
        energy_total_sim, energy_kinetic_sim, energy_internal_sim = total_energy_from_plot(handle, nr, nz)
        t_final_s = read_scalar(handle, "time_state/t")
        step_final = read_int_scalar(handle, "time_state/step")

    r_axis = r_cell.mean(axis=1)
    rho_radial = rho_cell.mean(axis=1)
    pressure_radial = pressure_cell.mean(axis=1)

    shock_radius_analytic = sedov_shock_radius_cm(t_final_s)
    shock_radius_sim = shock_radius(r_axis, rho_radial, r_blast_cm)
    dR = (R_MAX_CM - R_MIN_CM) / nr
    shock_radius_err_cells = abs(shock_radius_sim - shock_radius_analytic) / max(dR, 1.0e-300)
    beta_observed = shock_radius_sim / max(
        (E_PER_CM_DEFAULT / RHO_0_GCC) ** 0.25 * t_final_s**0.5,
        1.0e-30,
    )

    rho_post_sim = postshock_annulus_mean(r_axis, rho_radial, shock_radius_sim, nr, r_blast_cm)
    pressure_post_sim = postshock_annulus_mean(
        r_axis,
        pressure_radial,
        shock_radius_sim,
        nr,
        r_blast_cm,
    )
    rho_post_analytic = sedov_post_shock_density()
    pressure_post_analytic = sedov_post_shock_pressure(t_final_s)
    energy_initial_analytic = initial_energy_analytic_erg()

    return {
        "shock_radius_sim_cm": shock_radius_sim,
        "shock_radius_analytic_cm": shock_radius_analytic,
        "shock_radius_err_cells": float(shock_radius_err_cells),
        "beta_observed": float(beta_observed),
        "rho_post_sim_gcc": rho_post_sim,
        "rho_post_analytic_gcc": rho_post_analytic,
        "rho_post_err_pct": percent_error(rho_post_sim, rho_post_analytic),
        "P_post_sim_erg_per_cc": pressure_post_sim,
        "P_post_analytic_erg_per_cc": pressure_post_analytic,
        "P_post_err_pct": percent_error(pressure_post_sim, pressure_post_analytic),
        "energy_total_sim_erg": energy_total_sim,
        "energy_kinetic_sim_erg": energy_kinetic_sim,
        "energy_internal_sim_erg": energy_internal_sim,
        "energy_total_initial_analytic_erg": energy_initial_analytic,
        "energy_err_pct": percent_error(energy_total_sim, energy_initial_analytic),
        "r_blast_cm": r_blast_cm,
        "Te_axis_max_eV": float(np.max(Te_cell[: min(2, nr), :])),
        "ale_every_n_steps": ale_every,
        "ale_event_count": None,
        "t_final_s": t_final_s,
        "step_final": step_final,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_run(args: argparse.Namespace, run: dict[str, object]) -> int:
    cmd = [
        sys.executable,
        "tools/validation/append_run.py",
        "--stage-id",
        STAGE_ID,
        "--run-id",
        str(run["run_id"]),
        "--ladder-row",
        str(run["ladder_row"]),
        "--seed",
        str(run["seed"]),
        "--deck-path",
        args.deck,
        "--gpu-model",
        args.gpu_model,
        "--cuda-version",
        args.cuda_version,
        "--run-profile",
        args.run_profile,
        "--profile-class",
        args.profile_class,
        "--evidence-role",
        args.evidence_role,
        "--result",
        str(run["result"]),
        "--exit-code",
        str(run["exit_code"]),
        "--wall-time-s",
        f"{run['wall_time_s']:.9g}",
        "--output-bytes",
        str(run["output_bytes"]),
        "--root",
        args.root,
    ]
    failed_gates = run.get("failed_gates") or []
    if failed_gates:
        cmd.extend(["--failed-gates", ",".join(str(gate) for gate in failed_gates)])
    metrics_ref = run.get("metrics_ref")
    if metrics_ref:
        cmd.extend(["--metrics-ref", str(metrics_ref)])
    return subprocess.run(cmd).returncode


def empty_metrics(wall_time_s: float, bytes_out: int, ale_every: int, nr: int) -> dict[str, object]:
    return {
        "shock_radius_sim_cm": None,
        "shock_radius_analytic_cm": None,
        "shock_radius_err_cells": None,
        "beta_observed": None,
        "rho_post_sim_gcc": None,
        "rho_post_analytic_gcc": sedov_post_shock_density(),
        "rho_post_err_pct": None,
        "P_post_sim_erg_per_cc": None,
        "P_post_analytic_erg_per_cc": None,
        "P_post_err_pct": None,
        "energy_total_sim_erg": None,
        "energy_kinetic_sim_erg": None,
        "energy_internal_sim_erg": None,
        "energy_total_initial_analytic_erg": initial_energy_analytic_erg(),
        "energy_err_pct": None,
        "r_blast_cm": blast_radius_cm(nr),
        "Te_axis_max_eV": None,
        "t_alpha": None,
        "t_alpha_err_abs": None,
        "ale_every_n_steps": ale_every,
        "ale_event_count": None,
        "t_final_s": None,
        "step_final": None,
        "wall_time_s": wall_time_s,
        "output_bytes": bytes_out,
    }


def thresholds_payload() -> dict[str, float]:
    return {
        "beta_observed_min": BETA_OBSERVED_MIN,
        "beta_observed_max": BETA_OBSERVED_MAX,
        "energy_err_pct_max": ENERGY_ERR_PCT_MAX,
        "rho_post_err_pct_max": RHO_POST_ERR_PCT_MAX,
        "t_alpha_err_abs_max": T_ALPHA_ERR_MAX,
        "sedov_beta_2_reference": BETA_2,
    }


def run_one(
    args: argparse.Namespace,
    nr: int,
    nz: int,
    seed: int,
    av_c2: float,
    cfl: float,
    ale_every: int,
) -> dict[str, object]:
    run_id = run_id_for(nr, nz, seed, av_c2, cfl, ale_every)
    ladder_row = f"nr{nr}_nz{nz}"
    outdir = Path(args.out_root) / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "TENRYU_H3A_NR": str(nr),
            "TENRYU_H3A_NZ": str(nz),
            "TENRYU_H3A_SEED": str(seed),
            "TENRYU_H3A_OUTDIR": str(outdir),
            "TENRYU_H3A_AV_C2": f"{av_c2:.12g}",
            "TENRYU_H3A_CFL": f"{cfl:.12g}",
            "TENRYU_H3A_T_END_S": f"{args.t_end_s:.12g}",
            "TENRYU_H3A_E_PER_CM": f"{E_PER_CM_DEFAULT:.12g}",
            "TENRYU_H3A_BLAST_CELLS": str(BLAST_CELLS_DEFAULT),
            "TENRYU_H3A_ALE_EVERY_N_STEPS": str(ale_every),
        }
    )

    log_path = outdir / "run.log"
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([args.tenryu_bin, "run", args.deck], env=env, stdout=log, stderr=log)
    wall_time_s = time.monotonic() - start

    result = "PASS"
    failed_gates: list[str] = []
    metrics = empty_metrics(wall_time_s, output_bytes(outdir), ale_every, nr)

    if proc.returncode != 0:
        result = "CRASHED"
        failed_gates = ["run_failure"]
    else:
        try:
            results_dir = discover_results_dir(outdir)
            effective_outdir = results_dir.parent
            metrics["output_bytes"] = output_bytes(effective_outdir)
            if effective_outdir != outdir:
                metrics["output_bytes"] += output_bytes(outdir)

            plot_files = locate_plot_files(results_dir, case_name(nr, nz, seed, av_c2, cfl, ale_every))
            if not plot_files:
                raise RuntimeError("expected at least 1 plot file, found 0")

            metrics.update(analyze_plot(plot_files[-1], nr, nz, ale_every))
            t_alpha = shock_history_alpha(plot_files, nr, nz)
            metrics["t_alpha"] = t_alpha
            metrics["t_alpha_err_abs"] = None if t_alpha is None else abs(t_alpha - 0.5)
            metrics["wall_time_s"] = wall_time_s

            beta_obs = float(metrics["beta_observed"])
            if beta_obs < BETA_OBSERVED_MIN or beta_obs > BETA_OBSERVED_MAX:
                failed_gates.append("beta_observed")
            if float(metrics["energy_err_pct"]) > ENERGY_ERR_PCT_MAX:
                failed_gates.append("energy_err_pct")
            if float(metrics["rho_post_err_pct"]) > RHO_POST_ERR_PCT_MAX:
                failed_gates.append("rho_post_err_pct")
            if metrics["t_alpha_err_abs"] is None or float(metrics["t_alpha_err_abs"]) > T_ALPHA_ERR_MAX:
                failed_gates.append("t_alpha")
            if failed_gates:
                result = "FAIL"
        except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            result = "FAIL"
            failed_gates = ["analysis_failure"]
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[run_h3a] analysis failed: {exc}\n")

    rel_metrics_ref = f"docs/validation/2d_rz/{STAGE_ID}/metrics/{run_id}.json"
    metrics_path = Path(rel_metrics_ref)
    payload = {
        "stage_id": STAGE_ID,
        "stage_label": STAGE_LABEL,
        "run_id": run_id,
        "ladder_row": ladder_row,
        "seed": seed,
        "metrics": metrics,
        "thresholds": thresholds_payload(),
        "result": result,
    }
    write_json(metrics_path, payload)

    run = {
        "run_id": run_id,
        "ladder_row": ladder_row,
        "seed": seed,
        "av_c2": av_c2,
        "cfl": cfl,
        "ale_every": ale_every,
        "nr": nr,
        "nz": nz,
        "result": result,
        "exit_code": proc.returncode,
        "failed_gates": failed_gates,
        "metrics_ref": rel_metrics_ref,
        "wall_time_s": wall_time_s,
        "output_bytes": metrics["output_bytes"],
        "shock_radius_sim_cm": metrics["shock_radius_sim_cm"],
        "shock_radius_err_cells": metrics["shock_radius_err_cells"],
        "beta_observed": metrics["beta_observed"],
        "energy_err_pct": metrics["energy_err_pct"],
        "rho_post_err_pct": metrics["rho_post_err_pct"],
        "P_post_err_pct": metrics["P_post_err_pct"],
        "t_alpha": metrics["t_alpha"],
    }
    append_code = append_run(args, run)
    if append_code != 0 and result == "PASS":
        run["result"] = "FAIL"
        run["failed_gates"] = ["append_run_failure"]
    return run


def format_metric(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_summary(rows: list[dict[str, object]]) -> None:
    print(
        "Stage H3A thresholds: 0.9 <= beta_observed <= 1.3, energy_err_pct <= 0.5, "
        "rho_post_err_pct <= 30, |t_alpha - 0.5| <= 0.10"
    )
    print("ladder_row seed C2 cfl ale result beta_obs energy_pct rho_post_pct P_post_pct t_alpha wall_time_s")
    for row in rows:
        print(
            "{ladder_row} {seed} {av_c2} {cfl} {ale_every} {result} {beta_observed} "
            "{energy_err_pct} {rho_post_err_pct} {P_post_err_pct} {t_alpha} {wall_time_s:.3f}".format(
                ladder_row=row["ladder_row"],
                seed=row["seed"],
                av_c2=float_token(float(row["av_c2"])),
                cfl=float_token(float(row["cfl"])),
                ale_every=row["ale_every"],
                result=row["result"],
                beta_observed=format_metric(row["beta_observed"]),
                energy_err_pct=format_metric(row["energy_err_pct"]),
                rho_post_err_pct=format_metric(row["rho_post_err_pct"]),
                P_post_err_pct=format_metric(row["P_post_err_pct"]),
                t_alpha=format_metric(row["t_alpha"]),
                wall_time_s=float(row["wall_time_s"]),
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/2d_rz_h3a_sedov_cyl.py")
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    parser.add_argument("--gpu-model", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--cuda-version", default="12.0")
    parser.add_argument("--profile-class", default="ci_short", choices=("ci_short", "paper_full", "exploratory"))
    parser.add_argument("--evidence-role", default="ci_gate", choices=("ci_gate", "paper_figure", "exploratory"))
    parser.add_argument("--mesh-list", type=parse_mesh_list, default=parse_mesh_list("256,32;512,32"))
    parser.add_argument("--seed-list", type=parse_seed_list, default=parse_seed_list("12345,12346,12347,12348,12349"))
    parser.add_argument("--av-c2-list", type=parse_float_list, default=parse_float_list("1.5"))
    parser.add_argument("--cfl-list", type=parse_float_list, default=parse_float_list("0.1"))
    parser.add_argument("--ale-every-n-list", type=parse_ale_every_n_list, default=parse_ale_every_n_list("0"))
    parser.add_argument("--t-end-s", type=float, default=8.0e-9)
    parser.add_argument("--out-root", default="./build/h3a_runs")
    parser.add_argument("--run-profile", default="H3A cylindrical Sedov line-blast mesh ladder seed sweep")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mesh_list = list(args.mesh_list)
    if args.quick:
        mesh_list = mesh_list[:1]
    ale_every_n_list = parse_ale_every_n_list("0") if args.quick else list(args.ale_every_n_list)

    matrix = [
        (nr, nz, seed, av_c2, cfl, ale_every)
        for nr, nz in mesh_list
        for seed in args.seed_list
        for av_c2 in args.av_c2_list
        for cfl in args.cfl_list
        for ale_every in ale_every_n_list
    ]

    if args.dry_run:
        label = "quick " if args.quick else ""
        print(f"Planned Stage H3A {label}runs:")
        for nr, nz, seed, av_c2, cfl, ale_every in matrix:
            run_id = run_id_for(nr, nz, seed, av_c2, cfl, ale_every)
            print(
                f"{run_id} nr{nr}_nz{nz} seed={seed} "
                f"C2={float_token(av_c2)} cfl={float_token(cfl)} ale={ale_every}"
            )
        print(f"total_runs={len(matrix)}")
        return 0

    rows = [run_one(args, nr, nz, seed, av_c2, cfl, ale_every) for nr, nz, seed, av_c2, cfl, ale_every in matrix]
    print_summary(rows)
    return 0 if all(row["result"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
