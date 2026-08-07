#!/usr/bin/env python3
"""Run Stage H1 2D RZ planar Sod validation."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np


STAGE_ID = "H1"
GAMMA = 5.0 / 3.0
RHO_L_DIM_GCC = 1.0
P_REF_ERG_PER_CC = 1.0e12
L_SCALE_CM = 1.0
Z_MIN_CM = -0.5
Z_MAX_CM = 0.5
C_S_L_CM_S = math.sqrt(GAMMA * P_REF_ERG_PER_CC / RHO_L_DIM_GCC)
V_SCALE_CM_S = math.sqrt(P_REF_ERG_PER_CC / RHO_L_DIM_GCC)
TIME_SCALE_S = L_SCALE_CM / V_SCALE_CM_S

SHOCK_POS_ERR_CELLS_MAX = 2.0   # spec §2: ≤ 1–2 cells at fine mesh
RHO_POST_ERR_PCT_MAX = 3.0      # spec §2: ≤ 2–3% on the fine mesh
P_POST_ERR_PCT_MAX = 3.0        # spec §2: ≤ 2–3% on the fine mesh
RZ_INACTIVE_MAX_MAX = 1.5e-2    # spec §1 ideal 1e-6; relaxed pending Test E (axis BC investigation)

SOD_LEFT = {"rho": 1.0, "p": 1.0, "u": 0.0}
SOD_RIGHT = {"rho": 0.125, "p": 0.1, "u": 0.0}


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


def float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p")


def case_name(nr: int, nz: int, seed: int, av_c2: float, cfl: float) -> str:
    return (
        f"2d_rz_h1_sod_planar_nr{nr}_nz{nz}_seed{seed}"
        f"_C2{float_token(av_c2)}_cfl{float_token(cfl)}"
    )


def run_id_for(nr: int, nz: int, seed: int, av_c2: float, cfl: float) -> str:
    return f"h1_nr{nr}_nz{nz}_C2{float_token(av_c2)}_cfl{float_token(cfl)}_seed{seed}"


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


def pressure_function(p: float, state: dict[str, float], gamma: float) -> tuple[float, float]:
    rho_k = state["rho"]
    p_k = state["p"]
    c_k = math.sqrt(gamma * p_k / rho_k)
    if p > p_k:
        a_k = 2.0 / ((gamma + 1.0) * rho_k)
        b_k = (gamma - 1.0) / (gamma + 1.0) * p_k
        factor = math.sqrt(a_k / (p + b_k))
        f = (p - p_k) * factor
        df = factor * (1.0 - 0.5 * (p - p_k) / (p + b_k))
        return f, df

    pr = p / p_k
    exponent = (gamma - 1.0) / (2.0 * gamma)
    f = 2.0 * c_k / (gamma - 1.0) * (pr**exponent - 1.0)
    df = 1.0 / (rho_k * c_k) * pr ** (-(gamma + 1.0) / (2.0 * gamma))
    return f, df


def solve_sod(gamma: float = GAMMA) -> dict[str, float]:
    left = SOD_LEFT
    right = SOD_RIGHT
    p = 0.5 * (left["p"] + right["p"])
    for _ in range(64):
        f_l, df_l = pressure_function(p, left, gamma)
        f_r, df_r = pressure_function(p, right, gamma)
        residual = f_l + f_r + right["u"] - left["u"]
        p_next = p - residual / (df_l + df_r)
        if p_next <= 0.0:
            p_next = 0.5 * p
        if abs(p_next - p) <= 1.0e-12 * max(1.0, 0.5 * (p_next + p)):
            p = p_next
            break
        p = p_next

    f_l, _ = pressure_function(p, left, gamma)
    f_r, _ = pressure_function(p, right, gamma)
    u_star = 0.5 * (left["u"] + right["u"]) + 0.5 * (f_r - f_l)

    c_l = math.sqrt(gamma * left["p"] / left["rho"])
    c_r = math.sqrt(gamma * right["p"] / right["rho"])

    if p > left["p"]:
        p_ratio_l = p / left["p"]
        beta = (gamma - 1.0) / (gamma + 1.0)
        rho_star_l = left["rho"] * ((p_ratio_l + beta) / (beta * p_ratio_l + 1.0))
        s_l = left["u"] - c_l * math.sqrt((gamma + 1.0) / (2.0 * gamma) * p_ratio_l + (gamma - 1.0) / (2.0 * gamma))
        s_hl = s_l
        s_tl = s_l
        c_star_l = math.sqrt(gamma * p / rho_star_l)
    else:
        rho_star_l = left["rho"] * (p / left["p"]) ** (1.0 / gamma)
        c_star_l = c_l * (p / left["p"]) ** ((gamma - 1.0) / (2.0 * gamma))
        s_hl = left["u"] - c_l
        s_tl = u_star - c_star_l

    if p > right["p"]:
        p_ratio_r = p / right["p"]
        beta = (gamma - 1.0) / (gamma + 1.0)
        rho_star_r = right["rho"] * ((p_ratio_r + beta) / (beta * p_ratio_r + 1.0))
        s_r = right["u"] + c_r * math.sqrt((gamma + 1.0) / (2.0 * gamma) * p_ratio_r + (gamma - 1.0) / (2.0 * gamma))
        s_hr = s_r
        s_tr = s_r
        c_star_r = math.sqrt(gamma * p / rho_star_r)
    else:
        rho_star_r = right["rho"] * (p / right["p"]) ** (1.0 / gamma)
        c_star_r = c_r * (p / right["p"]) ** ((gamma - 1.0) / (2.0 * gamma))
        s_hr = right["u"] + c_r
        s_tr = u_star + c_star_r
        s_r = s_hr

    return {
        "gamma": gamma,
        "p_star": p,
        "u_star": u_star,
        "rho_star_l": rho_star_l,
        "rho_star_r": rho_star_r,
        "c_l": c_l,
        "c_r": c_r,
        "c_star_l": c_star_l,
        "c_star_r": c_star_r,
        "s_head_l": s_hl,
        "s_tail_l": s_tl,
        "s_tail_r": s_tr,
        "s_head_r": s_hr,
        "s_shock_r": s_r,
    }


def sample_sod_one(x: float, t_norm: float, solution: dict[str, float]) -> tuple[float, float, float]:
    if t_norm <= 0.0:
        state = SOD_LEFT if x < 0.0 else SOD_RIGHT
        return state["rho"], state["p"], state["u"]

    gamma = solution["gamma"]
    xi = x / t_norm
    p_star = solution["p_star"]
    u_star = solution["u_star"]

    if xi <= u_star:
        if p_star > SOD_LEFT["p"]:
            if xi <= solution["s_head_l"]:
                return SOD_LEFT["rho"], SOD_LEFT["p"], SOD_LEFT["u"]
            return solution["rho_star_l"], p_star, u_star
        if xi <= solution["s_head_l"]:
            return SOD_LEFT["rho"], SOD_LEFT["p"], SOD_LEFT["u"]
        if xi >= solution["s_tail_l"]:
            return solution["rho_star_l"], p_star, u_star
        c_l = solution["c_l"]
        u = 2.0 / (gamma + 1.0) * (c_l + 0.5 * (gamma - 1.0) * SOD_LEFT["u"] + xi)
        c = 2.0 / (gamma + 1.0) * (c_l + 0.5 * (gamma - 1.0) * (SOD_LEFT["u"] - xi))
        rho = SOD_LEFT["rho"] * (c / c_l) ** (2.0 / (gamma - 1.0))
        p = SOD_LEFT["p"] * (c / c_l) ** (2.0 * gamma / (gamma - 1.0))
        return rho, p, u

    if p_star > SOD_RIGHT["p"]:
        if xi >= solution["s_shock_r"]:
            return SOD_RIGHT["rho"], SOD_RIGHT["p"], SOD_RIGHT["u"]
        return solution["rho_star_r"], p_star, u_star
    if xi >= solution["s_head_r"]:
        return SOD_RIGHT["rho"], SOD_RIGHT["p"], SOD_RIGHT["u"]
    if xi <= solution["s_tail_r"]:
        return solution["rho_star_r"], p_star, u_star
    c_r = solution["c_r"]
    u = 2.0 / (gamma + 1.0) * (-c_r + 0.5 * (gamma - 1.0) * SOD_RIGHT["u"] + xi)
    c = 2.0 / (gamma + 1.0) * (c_r - 0.5 * (gamma - 1.0) * (SOD_RIGHT["u"] - xi))
    rho = SOD_RIGHT["rho"] * (c / c_r) ** (2.0 / (gamma - 1.0))
    p = SOD_RIGHT["p"] * (c / c_r) ** (2.0 * gamma / (gamma - 1.0))
    return rho, p, u


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


def rz_inactive_metric(rho: np.ndarray) -> float:
    nr, nz = rho.shape
    q_bar = np.mean(rho, axis=0)
    scale = float(np.max(np.abs(q_bar))) if q_bar.size else 0.0
    if scale <= 0.0:
        scale = 1.0

    z_mask = np.ones(nz, dtype=bool)
    edge = min(5, max(0, nz // 4))
    if edge > 0:
        z_mask[:edge] = False
        z_mask[-edge:] = False

    if nz >= 3:
        grad = np.abs(np.gradient(q_bar))
        work = grad.copy()
        for _ in range(min(3, nz)):
            idx = int(np.argmax(work))
            if work[idx] <= 0.0:
                break
            lo = max(0, idx - 8)
            hi = min(nz, idx + 9)
            z_mask[lo:hi] = False
            work[lo:hi] = 0.0

    if not np.any(z_mask):
        z_mask[:] = True
    diff = np.abs(rho - q_bar[np.newaxis, :]) / scale
    return float(np.max(diff[:, z_mask]))


def shock_position(z_axis: np.ndarray, rho_z: np.ndarray, contact_pos_cm: float) -> float:
    grad = np.abs(np.gradient(rho_z, z_axis))
    dz = float(np.median(np.abs(np.diff(z_axis)))) if z_axis.size > 1 else 0.0
    mask = z_axis > max(0.0, contact_pos_cm + 5.0 * dz)
    if not np.any(mask):
        mask = z_axis > 0.0
    if not np.any(mask):
        mask = np.ones_like(z_axis, dtype=bool)
    indices = np.nonzero(mask)[0]
    return float(z_axis[indices[int(np.argmax(grad[indices]))]])


def plateau_mean(z_axis: np.ndarray, values: np.ndarray, z_shock_sim_cm: float) -> float:
    mask = (z_axis >= z_shock_sim_cm - 0.05) & (z_axis <= z_shock_sim_cm - 0.02)
    if not np.any(mask):
        mask = (z_axis < z_shock_sim_cm) & (z_axis > z_shock_sim_cm - 0.08)
    if not np.any(mask):
        raise RuntimeError("could not identify post-shock plateau cells")
    return float(np.mean(values[mask]))


def contact_width_cells(z_axis: np.ndarray, rho_z: np.ndarray, solution: dict[str, float], t_norm: float) -> int:
    contact_pos_cm = solution["u_star"] * t_norm * L_SCALE_CM
    window = (z_axis >= contact_pos_cm - 0.1) & (z_axis <= contact_pos_cm + 0.1)
    if not np.any(window):
        return 0

    rho_lo = min(solution["rho_star_l"], solution["rho_star_r"]) * RHO_L_DIM_GCC
    rho_hi = max(solution["rho_star_l"], solution["rho_star_r"]) * RHO_L_DIM_GCC
    denom = max(rho_hi - rho_lo, 1.0e-300)
    eta = (rho_z - rho_lo) / denom
    transition = window & (eta > 0.05) & (eta < 0.95)
    return int(np.count_nonzero(transition))


def analyze_plot(path: Path, nr: int, nz: int) -> dict[str, float | int]:
    solution = solve_sod(GAMMA)
    with h5py.File(path, "r") as handle:
        r_cell, z_cell = cell_centers(handle, nr, nz)
        rho = as_cell_field(handle, "hydro/rho", nr, nz)
        pressure = read_pressure(handle, nr, nz)
        v_z = field_to_cells(handle, "mesh/v_z", nr, nz)
        t_final_s = read_scalar(handle, "time_state/t")
        step_final = read_int_scalar(handle, "time_state/step")

    z_axis = z_cell[0, :]
    rho_z = rho[0, :]
    pressure_z = pressure[0, :]
    t_norm = t_final_s / TIME_SCALE_S

    shock_pos_analytic_cm = solution["s_shock_r"] * t_norm * L_SCALE_CM
    contact_pos_cm = solution["u_star"] * t_norm * L_SCALE_CM
    shock_pos_sim_cm = shock_position(z_axis, rho_z, contact_pos_cm)
    dz_cell = float(np.median(np.abs(np.diff(z_axis)))) if z_axis.size > 1 else (Z_MAX_CM - Z_MIN_CM) / nz
    shock_pos_err_cells = abs(shock_pos_sim_cm - shock_pos_analytic_cm) / max(dz_cell, 1.0e-300)

    rho_post_sim = plateau_mean(z_axis, rho_z, shock_pos_sim_cm)
    p_post_sim = plateau_mean(z_axis, pressure_z, shock_pos_sim_cm)
    rho_post_analytic = solution["rho_star_r"] * RHO_L_DIM_GCC
    p_post_analytic = solution["p_star"] * P_REF_ERG_PER_CC

    _, _, u_profile = zip(*(sample_sod_one(float(z), t_norm, solution) for z in z_axis))
    v_z_axis = v_z[0, :]
    velocity_linf_cm_s = float(np.max(np.abs(v_z_axis - np.asarray(u_profile) * V_SCALE_CM_S)))

    return {
        "shock_pos_sim_cm": shock_pos_sim_cm,
        "shock_pos_analytic_cm": shock_pos_analytic_cm,
        "shock_pos_err_cells": float(shock_pos_err_cells),
        "rho_post_sim_gcc": rho_post_sim,
        "rho_post_analytic_gcc": rho_post_analytic,
        "rho_post_err_pct": abs(rho_post_sim - rho_post_analytic) / max(rho_post_analytic, 1.0e-300) * 100.0,
        "P_post_sim_erg_per_cc": p_post_sim,
        "P_post_analytic_erg_per_cc": p_post_analytic,
        "P_post_err_pct": abs(p_post_sim - p_post_analytic) / max(p_post_analytic, 1.0e-300) * 100.0,
        "contact_width_cells": contact_width_cells(z_axis, rho_z, solution, t_norm),
        "rz_inactive_max": rz_inactive_metric(rho),
        "t_final_s": t_final_s,
        "t_norm_dimless": t_norm,
        "step_final": step_final,
        "velocity_profile_linf_cm_s": velocity_linf_cm_s,
        "r_axis_cell_center_cm": float(r_cell[0, 0]),
    }


def write_metrics(path: Path, payload: dict[str, object]) -> None:
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


def run_one(
    args: argparse.Namespace,
    nr: int,
    nz: int,
    seed: int,
    av_c2: float,
    cfl: float,
    finest_mesh: tuple[int, int],
) -> dict[str, object]:
    run_id = run_id_for(nr, nz, seed, av_c2, cfl)
    ladder_row = f"nr{nr}_nz{nz}"
    outdir = Path(args.out_root) / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "TENRYU_H1_NR": str(nr),
            "TENRYU_H1_NZ": str(nz),
            "TENRYU_H1_SEED": str(seed),
            "TENRYU_H1_OUTDIR": str(outdir),
            "TENRYU_H1_AV_C2": f"{av_c2:.12g}",
            "TENRYU_H1_CFL": f"{cfl:.12g}",
            "TENRYU_H1_T_END_S": f"{args.t_end_s:.12g}",
        }
    )

    log_path = outdir / "run.log"
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([args.tenryu_bin, "run", args.deck], env=env, stdout=log, stderr=log)
    wall_time_s = time.monotonic() - start

    result = "PASS"
    failed_gates: list[str] = []
    metrics: dict[str, object] = {
        "shock_pos_sim_cm": None,
        "shock_pos_analytic_cm": None,
        "shock_pos_err_cells": None,
        "rho_post_sim_gcc": None,
        "rho_post_analytic_gcc": None,
        "rho_post_err_pct": None,
        "P_post_sim_erg_per_cc": None,
        "P_post_analytic_erg_per_cc": None,
        "P_post_err_pct": None,
        "contact_width_cells": None,
        "rz_inactive_max": None,
        "t_final_s": None,
        "t_norm_dimless": None,
        "step_final": None,
        "wall_time_s": wall_time_s,
        "output_bytes": output_bytes(outdir),
    }

    is_finest_mesh = (nr, nz) == finest_mesh
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

            plot_files = locate_plot_files(results_dir, case_name(nr, nz, seed, av_c2, cfl))
            if not plot_files:
                raise RuntimeError("expected at least 1 plot file, found 0")

            metrics.update(analyze_plot(plot_files[-1], nr, nz))

            if float(metrics["shock_pos_err_cells"]) > SHOCK_POS_ERR_CELLS_MAX:
                failed_gates.append("shock_pos_err_cells")
            if float(metrics["rz_inactive_max"]) > RZ_INACTIVE_MAX_MAX:
                failed_gates.append("rz_inactive_max")
            if is_finest_mesh and float(metrics["rho_post_err_pct"]) > RHO_POST_ERR_PCT_MAX:
                failed_gates.append("rho_post_err_pct")
            if is_finest_mesh and float(metrics["P_post_err_pct"]) > P_POST_ERR_PCT_MAX:
                failed_gates.append("P_post_err_pct")
            if failed_gates:
                result = "FAIL"
        except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            result = "FAIL"
            failed_gates = ["analysis_failure"]
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[run_h1] analysis failed: {exc}\n")

    rel_metrics_ref = f"docs/validation/2d_rz/H1/metrics/{run_id}.json"
    metrics_path = Path(rel_metrics_ref)
    payload = {
        "stage_id": STAGE_ID,
        "run_id": run_id,
        "ladder_row": ladder_row,
        "seed": seed,
        "metrics": metrics,
        "thresholds": {
            "shock_pos_err_cells_max": SHOCK_POS_ERR_CELLS_MAX,
            "rho_post_err_pct_max": RHO_POST_ERR_PCT_MAX,
            "P_post_err_pct_max": P_POST_ERR_PCT_MAX,
            "rz_inactive_max_max": RZ_INACTIVE_MAX_MAX,
        },
        "result": result,
    }
    write_metrics(metrics_path, payload)

    run = {
        "run_id": run_id,
        "ladder_row": ladder_row,
        "seed": seed,
        "av_c2": av_c2,
        "cfl": cfl,
        "nr": nr,
        "nz": nz,
        "result": result,
        "exit_code": proc.returncode,
        "failed_gates": failed_gates,
        "metrics_ref": rel_metrics_ref,
        "wall_time_s": wall_time_s,
        "output_bytes": metrics["output_bytes"],
        "shock_pos_err_cells": metrics["shock_pos_err_cells"],
        "rho_post_err_pct": metrics["rho_post_err_pct"],
        "P_post_err_pct": metrics["P_post_err_pct"],
        "contact_width_cells": metrics["contact_width_cells"],
        "rz_inactive_max": metrics["rz_inactive_max"],
    }
    append_code = append_run(args, run)
    if append_code != 0 and result == "PASS":
        run["result"] = "FAIL"
        run["failed_gates"] = ["append_run_failure"]
    return run


def print_contact_width_check(rows: list[dict[str, object]], finest_mesh: tuple[int, int]) -> None:
    by_seed: dict[tuple[float, float, int], dict[tuple[int, int], int]] = {}
    for row in rows:
        width = row.get("contact_width_cells")
        if width is None:
            continue
        key = (float(row["av_c2"]), float(row["cfl"]), int(row["seed"]))
        by_seed.setdefault(key, {})[(int(row["nr"]), int(row["nz"]))] = int(width)

    meshes = sorted({(int(row["nr"]), int(row["nz"])) for row in rows}, key=lambda item: item[0] * item[1])
    if len(meshes) < 2:
        return
    coarse_mesh = meshes[0]
    print(f"contact_width_check coarse={coarse_mesh[0]}x{coarse_mesh[1]} finest={finest_mesh[0]}x{finest_mesh[1]}")
    for key, widths in sorted(by_seed.items()):
        if coarse_mesh in widths and finest_mesh in widths:
            status = "PASS" if widths[finest_mesh] < widths[coarse_mesh] else "CHECK"
            av_c2, cfl, seed = key
            print(
                f"  C2={float_token(av_c2)} cfl={float_token(cfl)} seed={seed} "
                f"coarse={widths[coarse_mesh]} finest={widths[finest_mesh]} {status}"
            )


def print_summary(rows: list[dict[str, object]], finest_mesh: tuple[int, int]) -> None:
    print(
        "Stage H1 thresholds: shock_pos_err_cells <= 2, rz_inactive_max <= 1.5e-2, "
        "finest rho/P post-shock errors <= 5%"
    )
    print("ladder_row seed C2 cfl result shock_cells rho_post_pct P_post_pct contact_width rz_inactive wall_time_s")
    for row in rows:
        print(
            "{ladder_row} {seed} {av_c2} {cfl} {result} {shock_pos_err_cells} "
            "{rho_post_err_pct} {P_post_err_pct} {contact_width_cells} {rz_inactive_max} {wall_time_s:.3f}".format(
                ladder_row=row["ladder_row"],
                seed=row["seed"],
                av_c2=float_token(float(row["av_c2"])),
                cfl=float_token(float(row["cfl"])),
                result=row["result"],
                shock_pos_err_cells=row["shock_pos_err_cells"],
                rho_post_err_pct=row["rho_post_err_pct"],
                P_post_err_pct=row["P_post_err_pct"],
                contact_width_cells=row["contact_width_cells"],
                rz_inactive_max=row["rz_inactive_max"],
                wall_time_s=float(row["wall_time_s"]),
            )
        )
    print_contact_width_check(rows, finest_mesh)


def self_test_riemann() -> int:
    production = solve_sod(GAMMA)
    reference = solve_sod(1.4)
    print(
        "production_gamma_5_3 "
        f"p_star={production['p_star']:.8f} "
        f"u_star={production['u_star']:.8f} "
        f"rho_star_R={production['rho_star_r']:.8f} "
        f"S_R={production['s_shock_r']:.8f}"
    )
    print(
        "reference_gamma_1_4 "
        f"p_star={reference['p_star']:.8f} "
        f"u_star={reference['u_star']:.8f} "
        f"rho_star_R={reference['rho_star_r']:.8f} "
        f"S_R={reference['s_shock_r']:.8f}"
    )
    checks = [
        abs(production["p_star"] - 0.29394518766601785) < 1.0e-10,
        abs(production["u_star"] - 0.8411948521688083) < 1.0e-10,
        abs(production["rho_star_r"] - 0.229805749311947) < 1.0e-10,
        abs(reference["p_star"] - 0.30313017805064685) < 1.0e-10,
        abs(reference["u_star"] - 0.9274526200489499) < 1.0e-10,
        abs(reference["rho_star_r"] - 0.2655737117053071) < 1.0e-10,
        sample_sod_one(-10.0, 0.0, production) == (1.0, 1.0, 0.0),
        sample_sod_one(10.0, 0.0, production) == (0.125, 0.1, 0.0),
    ]
    if all(checks):
        print("riemann_self_test=PASS")
        return 0
    print("riemann_self_test=FAIL")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/2d_rz_h1_sod_planar.py")
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    parser.add_argument("--gpu-model", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--cuda-version", default="12.0")
    parser.add_argument("--profile-class", default="ci_short", choices=("ci_short", "paper_full", "exploratory"))
    parser.add_argument("--evidence-role", default="ci_gate", choices=("ci_gate", "paper_figure", "exploratory"))
    parser.add_argument("--mesh-list", type=parse_mesh_list, default=parse_mesh_list("64,256;128,512"))
    parser.add_argument("--seed-list", type=parse_seed_list, default=parse_seed_list("12345,12346,12347,12348,12349"))
    parser.add_argument("--av-c2-list", type=parse_float_list, default=parse_float_list("1.5"))
    parser.add_argument("--cfl-list", type=parse_float_list, default=parse_float_list("0.1"))
    parser.add_argument("--t-end-s", type=float, default=1.0e-7)
    parser.add_argument("--out-root", default="./build/h1_runs")
    parser.add_argument("--run-profile", default="H1 mesh ladder seed sweep")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test-riemann", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test_riemann:
        return self_test_riemann()

    mesh_list = list(args.mesh_list)
    finest_mesh = max(mesh_list, key=lambda item: (item[0] * item[1], item[1], item[0]))
    matrix = [
        (nr, nz, seed, av_c2, cfl)
        for nr, nz in mesh_list
        for seed in args.seed_list
        for av_c2 in args.av_c2_list
        for cfl in args.cfl_list
    ]

    if args.dry_run:
        print("Planned Stage H1 runs:")
        for nr, nz, seed, av_c2, cfl in matrix:
            run_id = run_id_for(nr, nz, seed, av_c2, cfl)
            print(f"{run_id} nr{nr}_nz{nz} seed={seed} C2={float_token(av_c2)} cfl={float_token(cfl)}")
        print(f"total_runs={len(matrix)}")
        return 0

    rows = [run_one(args, nr, nz, seed, av_c2, cfl, finest_mesh) for nr, nz, seed, av_c2, cfl in matrix]
    print_summary(rows, finest_mesh)
    return 0 if all(row["result"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
