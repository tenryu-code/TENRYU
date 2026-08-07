#!/usr/bin/env python3
"""Run Stage H3-B 2D RZ spherical Sedov blast validation."""

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
STAGE_LABEL = "H3B"

GAMMA = 5.0 / 3.0
RHO_0_GCC = 1.0
E_TOTAL_DEFAULT = 1.0e13
BETA_3 = 1.15

R_MIN_CM = 0.0
R_MAX_CM = 1.2
Z_MIN_CM = -1.2
Z_MAX_CM = 1.2
P_AMBIENT_ERG_PER_CC = 1.0e6
BLAST_CELLS_DEFAULT = 4
N_SHELLS_DEFAULT = 50
# Resolve the smeared shock profile at least as finely as the R mesh.
SHELLS_PER_R_CELL = 8

# Finite-source initialization and moderate t_end shift the observed order-unity Sedov prefactor.
BETA_OBSERVED_MIN = 0.85
BETA_OBSERVED_MAX = 1.5
ENERGY_ERR_PCT_MAX = 1.0
RHO_PEAK_ERR_PCT_MAX = 25.0
T_ALPHA_ERR_MAX = 0.10
A_RHO_MAX_VALID_MAX = 0.10
A_P_MAX_VALID_INFORMATIONAL_REFERENCE = 0.10
ANISOTROPY_VALID_RHO_FRACTION = 0.30
QUADRUPOLE_DISTORTION_MAX = 0.05
CONS_ERR_MAX_MAX = 5.15e-5
SNAPSHOT_MAX_METRIC_KEYS = [
    "A_rho_max_valid",
    "A_rho_max_raw_all_shells",
    "A_rho_max_cavity_excluded_valid",
    "A_rho_max_all_shells",
    "A_p_max_valid",
    "A_p_max_raw_all_shells",
    "A_p_max_cavity_excluded_valid",
    "A_p_max_all_shells",
    "quadrupole_distortion",
]
SNAPSHOT_MAX_QUADRUPOLE_COMPANION_KEYS = [
    "quadrupole_a0_cm",
    "quadrupole_a2_cm",
    "quadrupole_a2_amplitude_cm",
    "quadrupole_a2_over_a0",
    "quadrupole_ray_count",
]

PHASE_9_11_T31_ENV = {
    "TENRYU_H3B_AXIS_Z_MOTION": "fixed",
    "TENRYU_H3B_AXIS_REPAIR_MODE": "axis_spine_only",
    "TENRYU_H3B_REMAP_DAMAGE_GATE": "1",
    "TENRYU_H3B_REMAP_DAMAGE_DMAX": "0.05",
    "TENRYU_H3B_REMAP_DAMAGE_AXIS_ETA": "0.02",
    "TENRYU_H3B_REMAP_DAMAGE_AXIS_BUDGET": "1",
    "TENRYU_H3B_REMAP_DAMAGE_AXIS_BUDGET_FACTOR": "2.0",
    "TENRYU_H3B_REMAP_SCHEME": "ms2_moments",
    "TENRYU_H3B_REMAP_MS2_LIMITER": "van_leer",
    "TENRYU_H3B_KE_CLOSURE": "1",
    "TENRYU_H3B_KE_CLOSURE_REDISTRIBUTE_FLOOR": "1",
    "TENRYU_H3B_KE_CLOSURE_AUDIT": "0",
    "TENRYU_H3B_PHASE_RESOLVED_ENERGY": "0",
    "TENRYU_H3B_AXIS_MOTION_FLOOR_FRACTION": "0.0",
    "TENRYU_H3B_AXIS_MARGIN_DT_FLOOR_FRACTION": "0.0",
    "TENRYU_H3B_PREVENTIVE_AXIS_GUARD_FRACTION": "0.1",
}

LONG_TIME_T_END_LADDER_S = [9.0e-9, 1.2e-8, 1.5e-8, 1.8e-8]
MULTIGRID_LADDER_MESH_LIST = "128,256;256,512;512,1024"


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
        f"2d_rz_h3b_sedov_sph_nr{nr}_nz{nz}_seed{seed}"
        f"_C2{float_token(av_c2)}_cfl{float_token(cfl)}_ale{ale_every}"
    )


def run_id_for(nr: int, nz: int, seed: int, av_c2: float, cfl: float, ale_every: int) -> str:
    return (
        f"h3b_nr{nr}_nz{nz}_C2{float_token(av_c2)}"
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


def latest_history_file(results_dir: Path) -> Path | None:
    histories = sorted(results_dir.glob("*_history.h5"), key=lambda path: path.stat().st_mtime)
    return histories[-1] if histories else None


def sedov_3d_shock_radius_cm(
    t_s: float,
    E: float = E_TOTAL_DEFAULT,
    rho: float = RHO_0_GCC,
    beta: float = BETA_3,
) -> float:
    return beta * (E / rho) ** 0.2 * t_s**0.4


def sedov_3d_shock_velocity(
    t_s: float,
    E: float = E_TOTAL_DEFAULT,
    rho: float = RHO_0_GCC,
    beta: float = BETA_3,
) -> float:
    R_s = sedov_3d_shock_radius_cm(t_s, E, rho, beta)
    return (2.0 / 5.0) * R_s / t_s


def sedov_post_shock_density() -> float:
    return RHO_0_GCC * (GAMMA + 1.0) / (GAMMA - 1.0)


def sedov_post_shock_pressure(
    t_s: float,
    E: float = E_TOTAL_DEFAULT,
    rho: float = RHO_0_GCC,
    beta: float = BETA_3,
) -> float:
    Rdot = sedov_3d_shock_velocity(t_s, E, rho, beta)
    return (2.0 / (GAMMA + 1.0)) * rho * Rdot * Rdot


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


def cell_volumes(handle: h5py.File, nr: int, nz: int) -> np.ndarray:
    x_r = as_node_field(handle, "mesh/x_r", nr, nz)
    x_z = as_node_field(handle, "mesh/x_z", nr, nz)
    r_inner = 0.5 * (x_r[:-1, :-1] + x_r[:-1, 1:])
    r_outer = 0.5 * (x_r[1:, :-1] + x_r[1:, 1:])
    z_bottom = 0.5 * (x_z[:-1, :-1] + x_z[1:, :-1])
    z_top = 0.5 * (x_z[:-1, 1:] + x_z[1:, 1:])
    return math.pi * np.abs(r_outer * r_outer - r_inner * r_inner) * np.abs(z_top - z_bottom)


def blast_radius_cm(nr: int, blast_cells: int = BLAST_CELLS_DEFAULT) -> float:
    return blast_cells * (R_MAX_CM - R_MIN_CM) / nr


def box_volume_cm3() -> float:
    return math.pi * (R_MAX_CM * R_MAX_CM - R_MIN_CM * R_MIN_CM) * (Z_MAX_CM - Z_MIN_CM)


def initial_energy_analytic_erg(E: float = E_TOTAL_DEFAULT) -> float:
    e_ambient = P_AMBIENT_ERG_PER_CC / (GAMMA - 1.0) * box_volume_cm3()
    return e_ambient + E


def percent_error(sim: float, analytic: float) -> float:
    return abs(sim - analytic) / max(abs(analytic), 1.0e-300) * 100.0


def analysis_shell_count(nr: int) -> int:
    return max(N_SHELLS_DEFAULT, SHELLS_PER_R_CELL * nr)


def json_float_array(values: np.ndarray) -> list[float | None]:
    return [float(value) if math.isfinite(float(value)) else None for value in values]


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


def shell_profiles(
    r_cell: np.ndarray,
    z_cell: np.ndarray,
    rho_cell: np.ndarray,
    pressure_cell: np.ndarray,
    volume_cell: np.ndarray,
    n_shells: int = N_SHELLS_DEFAULT,
) -> dict[str, np.ndarray]:
    s_cell = np.hypot(r_cell, z_cell)
    s_max = float(np.max(s_cell))
    edges = np.linspace(0.0, s_max, n_shells + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    s_flat = s_cell.ravel()
    volume_flat = volume_cell.ravel()
    rho_flat = rho_cell.ravel()
    pressure_flat = pressure_cell.ravel()
    bin_index = np.searchsorted(edges, s_flat, side="right") - 1
    bin_index = np.clip(bin_index, 0, n_shells - 1)

    sum_w = np.bincount(bin_index, weights=volume_flat, minlength=n_shells)
    sum_rho = np.bincount(bin_index, weights=volume_flat * rho_flat, minlength=n_shells)
    sum_p = np.bincount(bin_index, weights=volume_flat * pressure_flat, minlength=n_shells)
    sum_rho2 = np.bincount(bin_index, weights=volume_flat * rho_flat * rho_flat, minlength=n_shells)
    sum_p2 = np.bincount(bin_index, weights=volume_flat * pressure_flat * pressure_flat, minlength=n_shells)

    rho_bar = np.full(n_shells, np.nan)
    p_bar = np.full(n_shells, np.nan)
    sigma_rho = np.full(n_shells, np.nan)
    sigma_p = np.full(n_shells, np.nan)
    A_rho = np.full(n_shells, np.nan)
    A_p = np.full(n_shells, np.nan)

    valid = sum_w > 0.0
    rho_bar[valid] = sum_rho[valid] / sum_w[valid]
    p_bar[valid] = sum_p[valid] / sum_w[valid]
    rho_var = np.maximum(sum_rho2[valid] / sum_w[valid] - rho_bar[valid] * rho_bar[valid], 0.0)
    p_var = np.maximum(sum_p2[valid] / sum_w[valid] - p_bar[valid] * p_bar[valid], 0.0)
    sigma_rho[valid] = np.sqrt(rho_var)
    sigma_p[valid] = np.sqrt(p_var)
    A_rho[valid] = sigma_rho[valid] / np.maximum(np.abs(rho_bar[valid]), 1.0e-300)
    A_p[valid] = sigma_p[valid] / np.maximum(np.abs(p_bar[valid]), 1.0e-300)

    return {
        "s_cm": centers,
        "shell_volume_cm3": sum_w,
        "rho_bar": rho_bar,
        "P_bar": p_bar,
        "sigma_rho": sigma_rho,
        "sigma_P": sigma_p,
        "A_rho": A_rho,
        "A_p": A_p,
    }


def shock_radius_from_shells(shells: dict[str, np.ndarray], r_blast_cm: float) -> float:
    s = shells["s_cm"]
    rho_bar = shells["rho_bar"]
    finite = np.isfinite(s) & np.isfinite(rho_bar)
    if np.count_nonzero(finite) < 2:
        raise RuntimeError("at least two populated spherical shells are required to locate the shock")

    s_valid = s[finite]
    rho_valid = rho_bar[finite]
    grad = np.abs(np.gradient(rho_valid, s_valid))
    mask = (s_valid >= r_blast_cm) & (s_valid <= 0.95 * s_valid[-1])
    if not np.any(mask):
        mask = s_valid >= 0.0
    indices = np.nonzero(mask)[0]
    return float(s_valid[indices[int(np.argmax(grad[indices]))]])


def postshock_shell_mean(
    shells: dict[str, np.ndarray],
    field_name: str,
    shock_radius_sim_cm: float,
    nr: int,
    r_blast_cm: float,
) -> float:
    s = shells["s_cm"]
    values = shells[field_name]
    dR = (R_MAX_CM - R_MIN_CM) / nr
    lo = shock_radius_sim_cm - 4.0 * dR
    hi = shock_radius_sim_cm - 1.0 * dR
    mask = (s >= lo) & (s <= hi) & np.isfinite(values)
    if not np.any(mask):
        mask = (s >= max(r_blast_cm, shock_radius_sim_cm - 6.0 * dR)) & (
            s < shock_radius_sim_cm
        ) & np.isfinite(values)
    if not np.any(mask):
        raise RuntimeError("could not identify post-shock shell cells")
    return float(np.mean(values[mask]))


def peak_density_near_shock(
    shells: dict[str, np.ndarray],
    shock_radius_sim_cm: float,
    nr: int,
) -> float:
    s = shells["s_cm"]
    rho_bar = shells["rho_bar"]
    dR = (R_MAX_CM - R_MIN_CM) / nr
    lo = shock_radius_sim_cm - 4.0 * dR
    hi = shock_radius_sim_cm + 1.0 * dR
    finite = np.isfinite(s) & np.isfinite(rho_bar)
    mask = finite & (s >= lo) & (s <= hi)
    if not np.any(mask):
        mask = finite & (s >= lo)
    if not np.any(mask):
        raise RuntimeError("could not identify near-shock shells for density peak")
    return float(np.max(rho_bar[mask]))


def shell_profile_normalized(
    shells: dict[str, np.ndarray],
    shock_radius_sim_cm: float,
) -> dict[str, list[float | None]]:
    return {
        "s_over_R_s": json_float_array(shells["s_cm"] / max(abs(shock_radius_sim_cm), 1.0e-300)),
        "rho_bar_gcc": json_float_array(shells["rho_bar"]),
    }


def shell_anisotropy_metrics(
    shells: dict[str, np.ndarray],
    shock_radius_sim_cm: float,
    r_blast_cm: float,
    rho_fraction_min: float = ANISOTROPY_VALID_RHO_FRACTION,
) -> dict[str, float | int]:
    s = shells["s_cm"]
    rho_bar = shells["rho_bar"]
    A_rho = shells["A_rho"]
    A_p = shells["A_p"]
    finite_rho = np.isfinite(rho_bar)
    if not np.any(finite_rho):
        raise RuntimeError("no populated shells available for density peak")
    rho_peak_in_shells = float(np.max(rho_bar[finite_rho]))
    rho_valid_threshold = rho_fraction_min * rho_peak_in_shells

    all_mask = np.isfinite(s) & np.isfinite(rho_bar) & np.isfinite(A_rho) & np.isfinite(A_p)
    if not np.any(all_mask):
        raise RuntimeError("no populated shells available for anisotropy metric")
    domain_mask = all_mask & (s > 2.0 * r_blast_cm) & (s < shock_radius_sim_cm)
    valid_mask = domain_mask & (rho_bar > rho_valid_threshold)
    if not np.any(valid_mask):
        raise RuntimeError("no density-valid shells available for anisotropy metric")

    combined_all = np.maximum(A_rho[all_mask], A_p[all_mask])
    combined_valid = np.maximum(A_rho[valid_mask], A_p[valid_mask])
    A_rho_raw_max = float(np.max(A_rho[all_mask]))
    A_p_raw_max = float(np.max(A_p[all_mask]))
    A_rho_valid_max = float(np.max(A_rho[valid_mask]))
    A_p_valid_max = float(np.max(A_p[valid_mask]))
    return {
        "A_rho_max_raw_all_shells": A_rho_raw_max,
        "A_rho_max_raw_all_shells_at_s_cm": float(s[all_mask][int(np.argmax(A_rho[all_mask]))]),
        "A_p_max_raw_all_shells": A_p_raw_max,
        "A_p_max_raw_all_shells_at_s_cm": float(s[all_mask][int(np.argmax(A_p[all_mask]))]),
        "A_rho_max_cavity_excluded_valid": A_rho_valid_max,
        "A_rho_max_cavity_excluded_valid_at_s_cm": float(s[valid_mask][int(np.argmax(A_rho[valid_mask]))]),
        "A_p_max_cavity_excluded_valid": A_p_valid_max,
        "A_p_max_cavity_excluded_valid_at_s_cm": float(s[valid_mask][int(np.argmax(A_p[valid_mask]))]),
        "A_rho_max_all_shells": A_rho_raw_max,
        "A_p_max_all_shells": A_p_raw_max,
        "shells_anisotropy_max_all_shells_at_s_cm": float(s[all_mask][int(np.argmax(combined_all))]),
        "A_rho_max_valid": A_rho_valid_max,
        "A_p_max_valid": A_p_valid_max,
        "shells_anisotropy_max_valid_at_s_cm": float(s[valid_mask][int(np.argmax(combined_valid))]),
        "rho_peak_shells_gcc": rho_peak_in_shells,
        "rho_valid_shell_threshold_gcc": rho_valid_threshold,
        "n_anisotropy_populated_shells": int(np.count_nonzero(all_mask)),
        "n_anisotropy_domain_shells": int(np.count_nonzero(domain_mask)),
        "n_anisotropy_cavity_excluded_shells": int(np.count_nonzero(valid_mask)),
        "n_anisotropy_valid_shells": int(np.count_nonzero(valid_mask)),
    }


def density_anisotropy_gate_failed(metrics: dict[str, object]) -> bool:
    value = finite_metric(metrics, "A_rho_max_valid")
    return value is None or value > A_RHO_MAX_VALID_MAX


def quadrupole_gate_failed(metrics: dict[str, object]) -> bool:
    value = finite_metric(metrics, "quadrupole_distortion")
    return value is None or value > QUADRUPOLE_DISTORTION_MAX


def finite_metric(metrics: dict[str, object], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    value_f = float(value)
    return value_f if math.isfinite(value_f) else None


def symmetry_gate_status(metrics: dict[str, object]) -> str:
    raw_a_rho = finite_metric(metrics, "A_rho_max_raw_all_shells")
    if raw_a_rho is None:
        raw_a_rho = finite_metric(metrics, "A_rho_max_all_shells")
    valid_a_rho = finite_metric(metrics, "A_rho_max_valid")
    quadrupole = finite_metric(metrics, "quadrupole_distortion")
    quadrupole_snapshot_max = finite_metric(metrics, "quadrupole_distortion_snapshot_max")
    quadrupole_snapshot_max_t = finite_metric(metrics, "quadrupole_distortion_snapshot_max_at_t_s")

    if quadrupole is None:
        return "not_evaluated"
    if quadrupole > QUADRUPOLE_DISTORTION_MAX:
        return "true_low_order_distortion"
    if valid_a_rho is None:
        return "anisotropy_not_evaluated"
    if valid_a_rho > A_RHO_MAX_VALID_MAX:
        return "valid_shell_anisotropy"
    if (
        quadrupole_snapshot_max is not None
        and quadrupole_snapshot_max > 0.10
        and quadrupole_snapshot_max_t is not None
        and quadrupole_snapshot_max_t < 1.0e-12
    ):
        return "early_blast_quadrupole"
    if raw_a_rho is not None and raw_a_rho > A_RHO_MAX_VALID_MAX:
        return "cavity_artifact"
    return "symmetric"


def shock_radius_along_ray(
    theta: float,
    s_cell: np.ndarray,
    theta_cell: np.ndarray,
    rho_cell: np.ndarray,
    volume_cell: np.ndarray,
    r_blast_cm: float,
) -> float:
    half_width = math.pi / 48.0
    sector_mask = np.abs(theta_cell - theta) <= half_width
    while np.count_nonzero(sector_mask) < 12 and half_width < math.pi / 8.0:
        half_width *= 1.5
        sector_mask = np.abs(theta_cell - theta) <= half_width
    if not np.any(sector_mask):
        raise RuntimeError("no cells available in angular sector for shock detection")

    s_sector = s_cell[sector_mask]
    rho_sector = rho_cell[sector_mask]
    volume_sector = volume_cell[sector_mask]
    s_max = float(np.max(s_sector))
    if s_max <= 2.0 * r_blast_cm:
        raise RuntimeError("angular sector is too short to locate shock")

    n_bins = 100
    edges = np.linspace(0.0, s_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_index = np.searchsorted(edges, s_sector, side="right") - 1
    bin_index = np.clip(bin_index, 0, n_bins - 1)
    sum_w = np.bincount(bin_index, weights=volume_sector, minlength=n_bins)
    sum_rho = np.bincount(bin_index, weights=volume_sector * rho_sector, minlength=n_bins)
    rho_bar = np.full(n_bins, np.nan)
    valid = sum_w > 0.0
    rho_bar[valid] = sum_rho[valid] / sum_w[valid]
    finite = np.isfinite(rho_bar)
    if np.count_nonzero(finite) < 2:
        raise RuntimeError("not enough populated sector bins for shock detection")

    s_valid = centers[finite]
    rho_valid = rho_bar[finite]
    grad = np.abs(np.gradient(rho_valid, s_valid))
    mask = (s_valid >= r_blast_cm) & (s_valid <= 0.95 * s_max)
    indices = np.nonzero(mask)[0]
    if indices.size == 0:
        raise RuntimeError("no sector bins available for shock detection")
    return float(s_valid[indices[int(np.argmax(grad[indices]))]])


def quadrupole_mode_metrics(
    r_cell: np.ndarray,
    z_cell: np.ndarray,
    rho_cell: np.ndarray,
    volume_cell: np.ndarray,
    r_blast_cm: float,
) -> dict[str, object]:
    s_cell = np.hypot(r_cell, z_cell)
    theta_cell = np.arctan2(r_cell, z_cell)
    angles = np.linspace(0.0, math.pi, 9)
    shock_radii = np.asarray(
        [
            shock_radius_along_ray(
                float(theta),
                s_cell,
                theta_cell,
                rho_cell,
                volume_cell,
                r_blast_cm,
            )
            for theta in angles
        ]
    )
    weights = np.ones_like(angles)
    weights[0] = 0.5
    weights[-1] = 0.5
    weight_sum = float(np.sum(weights))
    a0 = float(np.sum(weights * shock_radii) / weight_sum)
    a2 = float(np.sum(weights * shock_radii * np.cos(2.0 * angles)) / weight_sum)
    a2_over_a0 = a2 / max(abs(a0), 1.0e-300)
    return {
        "quadrupole_distortion": abs(a2_over_a0),
        "quadrupole_a0_cm": a0,
        "quadrupole_a2_cm": a2,
        "quadrupole_a2_amplitude_cm": abs(a2),
        "quadrupole_a2_over_a0": a2_over_a0,
        "quadrupole_ray_count": int(angles.size),
    }


def quadrupole_distortion(
    r_cell: np.ndarray,
    z_cell: np.ndarray,
    rho_cell: np.ndarray,
    volume_cell: np.ndarray,
    r_blast_cm: float,
) -> float:
    return float(
        quadrupole_mode_metrics(r_cell, z_cell, rho_cell, volume_cell, r_blast_cm)[
            "quadrupole_distortion"
        ]
    )


def shape_metrics_from_plot(
    path: Path,
    nr: int,
    nz: int,
) -> dict[str, object]:
    r_blast_cm = blast_radius_cm(nr)
    with h5py.File(path, "r") as handle:
        r_cell, z_cell = cell_centers(handle, nr, nz)
        rho_cell = as_cell_field(handle, "hydro/rho", nr, nz)
        pressure_cell = read_pressure(handle, nr, nz)
        volume_cell = cell_volumes(handle, nr, nz)
        t_s = read_scalar(handle, "time_state/t")

    shells = shell_profiles(r_cell, z_cell, rho_cell, pressure_cell, volume_cell, analysis_shell_count(nr))
    shock_radius_sim = shock_radius_from_shells(shells, r_blast_cm)
    quadrupole = quadrupole_mode_metrics(r_cell, z_cell, rho_cell, volume_cell, r_blast_cm)
    payload: dict[str, object] = {
        "t_snapshot_s": t_s,
        "anisotropy_analysis_error": None,
    }
    payload.update(quadrupole)
    try:
        anisotropy = shell_anisotropy_metrics(shells, shock_radius_sim, r_blast_cm)
    except RuntimeError as exc:
        payload["anisotropy_analysis_error"] = str(exc)
        return payload
    payload.update(
        {
            "A_rho_max_valid": float(anisotropy["A_rho_max_valid"]),
            "A_rho_max_raw_all_shells": float(anisotropy["A_rho_max_raw_all_shells"]),
            "A_rho_max_cavity_excluded_valid": float(anisotropy["A_rho_max_cavity_excluded_valid"]),
            "A_rho_max_all_shells": float(anisotropy["A_rho_max_all_shells"]),
            "A_p_max_valid": float(anisotropy["A_p_max_valid"]),
            "A_p_max_raw_all_shells": float(anisotropy["A_p_max_raw_all_shells"]),
            "A_p_max_cavity_excluded_valid": float(anisotropy["A_p_max_cavity_excluded_valid"]),
            "A_p_max_all_shells": float(anisotropy["A_p_max_all_shells"]),
        }
    )
    return payload


def max_shape_metrics_over_snapshots(plot_files: list[Path], nr: int, nz: int) -> dict[str, object]:
    maxima: dict[str, float] = {}
    times: dict[str, float] = {}
    companion_metrics: dict[str, object] = {}
    used = 0
    skipped = 0
    for path in plot_files:
        try:
            snapshot = shape_metrics_from_plot(path, nr, nz)
        except (OSError, KeyError, ValueError, RuntimeError):
            skipped += 1
            continue
        used += 1
        t_snapshot_s = float(snapshot["t_snapshot_s"])
        for key in SNAPSHOT_MAX_METRIC_KEYS:
            value = snapshot.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                continue
            if key not in maxima or float(value) > maxima[key]:
                maxima[key] = float(value)
                times[f"{key}_at_t_s"] = t_snapshot_s
                if key == "quadrupole_distortion":
                    for companion_key in SNAPSHOT_MAX_QUADRUPOLE_COMPANION_KEYS:
                        companion_value = snapshot.get(companion_key)
                        if companion_value is not None:
                            companion_metrics[companion_key] = companion_value

    if used == 0:
        raise RuntimeError("no snapshots were usable for Sedov shape metrics")

    payload: dict[str, object] = {
        "shape_metrics_source": "max_across_snapshots",
        "shape_snapshot_count": used,
        "shape_snapshot_skipped_count": skipped,
    }
    payload.update(maxima)
    payload.update(companion_metrics)
    payload.update(times)
    return payload


def snapshot_max_key(key: str) -> str:
    if key == "shape_metrics_source":
        return "shape_snapshot_max_source"
    if key == "shape_snapshot_count":
        return "shape_snapshot_max_count"
    if key == "shape_snapshot_skipped_count":
        return "shape_snapshot_max_skipped_count"
    if key.endswith("_at_t_s"):
        return f"{key[:-len('_at_t_s')]}_snapshot_max_at_t_s"
    return f"{key}_snapshot_max"


def snapshot_max_informational(plot_files: list[Path], nr: int, nz: int) -> dict[str, object]:
    snapshot_max = max_shape_metrics_over_snapshots(plot_files, nr, nz)
    return {snapshot_max_key(key): value for key, value in snapshot_max.items()}


def empty_snapshot_max_metrics() -> dict[str, object]:
    metrics: dict[str, object] = {
        "shape_snapshot_max_source": None,
        "shape_snapshot_max_count": None,
        "shape_snapshot_max_skipped_count": None,
    }
    for key in SNAPSHOT_MAX_METRIC_KEYS:
        metrics[f"{key}_snapshot_max"] = None
        metrics[f"{key}_snapshot_max_at_t_s"] = None
    for key in SNAPSHOT_MAX_QUADRUPOLE_COMPANION_KEYS:
        metrics[f"{key}_snapshot_max"] = None
    return metrics


def history_metrics(results_dir: Path) -> dict[str, object]:
    history = latest_history_file(results_dir)
    if history is None:
        return {
            "cons_err_max": None,
            "conservation_error_max": None,
            "ale_event_count": None,
        }

    metrics: dict[str, object] = {
        "cons_err_max": None,
        "conservation_error_max": None,
        "ale_event_count": None,
    }
    with h5py.File(history, "r") as handle:
        if "energy/conservation_error" in handle:
            values = np.asarray(handle["energy/conservation_error"][()]).astype(float, copy=False)
            finite = values[np.isfinite(values)]
            if finite.size > 0:
                cons_err_max = float(np.max(np.abs(finite)))
                metrics["cons_err_max"] = cons_err_max
                metrics["conservation_error_max"] = cons_err_max
        if "mesh/ale_rezone_invocations" in handle:
            values = np.asarray(handle["mesh/ale_rezone_invocations"][()])
            if values.size > 0:
                metrics["ale_event_count"] = int(values.reshape(-1)[-1])
            else:
                metrics["ale_event_count"] = 0
    return metrics


def shell_profiles_from_plot(
    path: Path,
    nr: int,
    nz: int,
) -> tuple[float, dict[str, np.ndarray]]:
    with h5py.File(path, "r") as handle:
        r_cell, z_cell = cell_centers(handle, nr, nz)
        rho_cell = as_cell_field(handle, "hydro/rho", nr, nz)
        pressure_cell = read_pressure(handle, nr, nz)
        volume_cell = cell_volumes(handle, nr, nz)
        t_s = read_scalar(handle, "time_state/t")
    return t_s, shell_profiles(r_cell, z_cell, rho_cell, pressure_cell, volume_cell, analysis_shell_count(nr))


def shock_history_alpha(plot_files: list[Path], nr: int, nz: int) -> float | None:
    r_blast_cm = blast_radius_cm(nr)
    self_similar_R_threshold = 2.0 * r_blast_cm
    times = []
    radii = []
    for path in plot_files:
        try:
            t_s, shells = shell_profiles_from_plot(path, nr, nz)
            if t_s <= 0.0:
                continue
            r_s = shock_radius_from_shells(shells, r_blast_cm)
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


def analyze_plot(path: Path, nr: int, nz: int, ale_every: int) -> dict[str, object]:
    r_blast_cm = blast_radius_cm(nr)
    with h5py.File(path, "r") as handle:
        r_cell, z_cell = cell_centers(handle, nr, nz)
        rho_cell = as_cell_field(handle, "hydro/rho", nr, nz)
        pressure_cell = read_pressure(handle, nr, nz)
        Te_cell = as_cell_field(handle, "hydro/Te", nr, nz)
        volume_cell = cell_volumes(handle, nr, nz)
        energy_total_sim, energy_kinetic_sim, energy_internal_sim = total_energy_from_plot(handle, nr, nz)
        t_final_s = read_scalar(handle, "time_state/t")
        step_final = read_int_scalar(handle, "time_state/step")

    n_shells = analysis_shell_count(nr)
    shells = shell_profiles(r_cell, z_cell, rho_cell, pressure_cell, volume_cell, n_shells)
    shock_radius_analytic = sedov_3d_shock_radius_cm(t_final_s)
    shock_radius_sim = shock_radius_from_shells(shells, r_blast_cm)
    dR = (R_MAX_CM - R_MIN_CM) / nr
    shock_radius_err_cells = abs(shock_radius_sim - shock_radius_analytic) / max(dR, 1.0e-300)
    beta_observed = shock_radius_sim / max(
        (E_TOTAL_DEFAULT / RHO_0_GCC) ** 0.2 * t_final_s**0.4,
        1.0e-30,
    )

    rho_post_sim = postshock_shell_mean(shells, "rho_bar", shock_radius_sim, nr, r_blast_cm)
    pressure_post_sim = postshock_shell_mean(shells, "P_bar", shock_radius_sim, nr, r_blast_cm)
    rho_post_analytic = sedov_post_shock_density()
    rho_peak_sim = peak_density_near_shock(shells, shock_radius_sim, nr)
    rho_peak_analytic = sedov_post_shock_density()
    pressure_post_analytic = sedov_post_shock_pressure(t_final_s)
    energy_initial_analytic = initial_energy_analytic_erg()
    quadrupole = quadrupole_mode_metrics(r_cell, z_cell, rho_cell, volume_cell, r_blast_cm)
    try:
        anisotropy = shell_anisotropy_metrics(shells, shock_radius_sim, r_blast_cm)
        anisotropy_analysis_error = None
    except RuntimeError as exc:
        anisotropy = {}
        anisotropy_analysis_error = str(exc)

    metrics: dict[str, object] = {
        "shock_radius_sim_cm": shock_radius_sim,
        "shock_radius_analytic_cm": shock_radius_analytic,
        "shock_radius_err_cells": float(shock_radius_err_cells),
        "beta_observed": float(beta_observed),
        "rho_peak_sim_gcc": rho_peak_sim,
        "rho_peak_analytic_gcc": rho_peak_analytic,
        "rho_peak_err_pct": percent_error(rho_peak_sim, rho_peak_analytic),
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
        "anisotropy_analysis_error": anisotropy_analysis_error,
        "n_shells": n_shells,
        "shell_profile_normalized": shell_profile_normalized(shells, shock_radius_sim),
        "r_blast_cm": r_blast_cm,
        "Te_axis_max_eV": float(np.max(Te_cell[: min(2, nr), :])),
        "ale_every_n_steps": ale_every,
        "ale_event_count": None,
        "cons_err_max": None,
        "conservation_error_max": None,
        "t_final_s": t_final_s,
        "step_final": step_final,
    }
    metrics.update(anisotropy)
    metrics.update(quadrupole)
    return metrics


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
    metrics: dict[str, object] = {
        "shock_radius_sim_cm": None,
        "shock_radius_analytic_cm": None,
        "shock_radius_err_cells": None,
        "beta_observed": None,
        "rho_peak_sim_gcc": None,
        "rho_peak_analytic_gcc": sedov_post_shock_density(),
        "rho_peak_err_pct": None,
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
        "A_rho_max_valid": None,
        "A_p_max_valid": None,
        "A_rho_max_raw_all_shells": None,
        "A_rho_max_raw_all_shells_at_s_cm": None,
        "A_p_max_raw_all_shells": None,
        "A_p_max_raw_all_shells_at_s_cm": None,
        "A_rho_max_cavity_excluded_valid": None,
        "A_rho_max_cavity_excluded_valid_at_s_cm": None,
        "A_p_max_cavity_excluded_valid": None,
        "A_p_max_cavity_excluded_valid_at_s_cm": None,
        "A_rho_max_all_shells": None,
        "A_p_max_all_shells": None,
        "quadrupole_distortion": None,
        "quadrupole_a0_cm": None,
        "quadrupole_a2_cm": None,
        "quadrupole_a2_amplitude_cm": None,
        "quadrupole_a2_over_a0": None,
        "quadrupole_ray_count": None,
        "anisotropy_analysis_error": None,
        "symmetry_gate_status": "not_evaluated",
        "n_shells": analysis_shell_count(nr),
        "shells_anisotropy_max_valid_at_s_cm": None,
        "shells_anisotropy_max_all_shells_at_s_cm": None,
        "rho_peak_shells_gcc": None,
        "rho_valid_shell_threshold_gcc": None,
        "n_anisotropy_populated_shells": None,
        "n_anisotropy_domain_shells": None,
        "n_anisotropy_cavity_excluded_shells": None,
        "n_anisotropy_valid_shells": None,
        "shell_profile_normalized": None,
        "r_blast_cm": blast_radius_cm(nr),
        "Te_axis_max_eV": None,
        "t_alpha": None,
        "t_alpha_err_abs": None,
        "ale_every_n_steps": ale_every,
        "ale_event_count": None,
        "cons_err_max": None,
        "conservation_error_max": None,
        "t_final_s": None,
        "step_final": None,
        "wall_time_s": wall_time_s,
        "output_bytes": bytes_out,
    }
    metrics.update(empty_snapshot_max_metrics())
    return metrics


def thresholds_payload() -> dict[str, float]:
    return {
        "beta_observed_min": BETA_OBSERVED_MIN,
        "beta_observed_max": BETA_OBSERVED_MAX,
        "energy_err_pct_max": ENERGY_ERR_PCT_MAX,
        "rho_peak_err_pct_max": RHO_PEAK_ERR_PCT_MAX,
        "t_alpha_err_abs_max": T_ALPHA_ERR_MAX,
        "A_rho_max_valid_max": A_RHO_MAX_VALID_MAX,
        "A_p_max_valid_informational_reference": A_P_MAX_VALID_INFORMATIONAL_REFERENCE,
        "anisotropy_valid_rho_fraction_of_peak": ANISOTROPY_VALID_RHO_FRACTION,
        "quadrupole_distortion_max": QUADRUPOLE_DISTORTION_MAX,
        "cons_err_max_max": CONS_ERR_MAX_MAX,
        "sedov_beta_3_reference": BETA_3,
        "n_shells_min": float(N_SHELLS_DEFAULT),
        "shells_per_r_cell": float(SHELLS_PER_R_CELL),
    }


def active_env_overrides(args: argparse.Namespace) -> dict[str, str]:
    if args.phase_9_11_t31:
        env = dict(PHASE_9_11_T31_ENV)
        if args.shape_multiseed_winslow:
            env["TENRYU_H3B_AXIS_REPAIR_MODE"] = "full_winslow"
        return env
    return {}


def is_shape_multiseed_preset(args: argparse.Namespace) -> bool:
    return bool(args.shape_multiseed or args.shape_multiseed_winslow)


def run_id_suffix(args: argparse.Namespace) -> str:
    suffix = args.run_id_suffix
    if args.shape_multiseed_winslow and not suffix:
        suffix = "shape_p9_11_t31_winslow"
    if args.shape_multiseed and not suffix:
        suffix = "shape_p9_11_t31"
    if args.long_time_sweep and not suffix:
        suffix = "long_time_p9_11_t31"
    if args.multigrid_ladder and not suffix:
        suffix = "multigrid_ladder_p9_11_t31"
    if not suffix:
        return ""
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in suffix)
    return f"_{sanitized.lstrip('_')}"


def t_end_run_id_suffix(args: argparse.Namespace, t_end_s: float) -> str:
    if not args.long_time_sweep:
        return ""
    return f"_tend{float_token(t_end_s)}"


def run_one(
    args: argparse.Namespace,
    nr: int,
    nz: int,
    seed: int,
    av_c2: float,
    cfl: float,
    ale_every: int,
    t_end_s: float,
) -> dict[str, object]:
    run_id = (
        run_id_for(nr, nz, seed, av_c2, cfl, ale_every)
        + t_end_run_id_suffix(args, t_end_s)
        + run_id_suffix(args)
    )
    ladder_row = f"nr{nr}_nz{nz}"
    if args.long_time_sweep:
        ladder_row = f"{ladder_row}_tend{float_token(t_end_s)}"
    outdir = Path(args.out_root) / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "TENRYU_H3B_NR": str(nr),
            "TENRYU_H3B_NZ": str(nz),
            "TENRYU_H3B_SEED": str(seed),
            "TENRYU_H3B_OUTDIR": str(outdir),
            "TENRYU_H3B_AV_C2": f"{av_c2:.12g}",
            "TENRYU_H3B_CFL": f"{cfl:.12g}",
            "TENRYU_H3B_T_END_S": f"{t_end_s:.12g}",
            "TENRYU_H3B_E_TOTAL": f"{E_TOTAL_DEFAULT:.12g}",
            "TENRYU_H3B_BLAST_CELLS": str(BLAST_CELLS_DEFAULT),
            "TENRYU_H3B_ALE_EVERY_N_STEPS": str(ale_every),
        }
    )
    env.update(active_env_overrides(args))

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
            metrics.update(snapshot_max_informational(plot_files, nr, nz))
            metrics.update(history_metrics(results_dir))
            t_alpha = shock_history_alpha(plot_files, nr, nz)
            metrics["t_alpha"] = t_alpha
            metrics["t_alpha_err_abs"] = None if t_alpha is None else abs(t_alpha - 0.4)
            metrics["wall_time_s"] = wall_time_s

            beta_obs = float(metrics["beta_observed"])
            if beta_obs < BETA_OBSERVED_MIN or beta_obs > BETA_OBSERVED_MAX:
                failed_gates.append("beta_observed")
            if float(metrics["energy_err_pct"]) > ENERGY_ERR_PCT_MAX:
                failed_gates.append("energy_err_pct")
            if metrics["rho_peak_err_pct"] is None or float(metrics["rho_peak_err_pct"]) > RHO_PEAK_ERR_PCT_MAX:
                failed_gates.append("rho_peak_err_pct")
            if metrics["t_alpha_err_abs"] is None or float(metrics["t_alpha_err_abs"]) > T_ALPHA_ERR_MAX:
                failed_gates.append("t_alpha")
            if density_anisotropy_gate_failed(metrics):
                failed_gates.append("A_rho_max_valid")
            if args.cons_err_gate:
                cons_err_max = finite_metric(metrics, "cons_err_max")
                if cons_err_max is None or cons_err_max > CONS_ERR_MAX_MAX:
                    failed_gates.append("cons_err_max")
            if quadrupole_gate_failed(metrics):
                failed_gates.append("quadrupole_distortion")
            metrics["symmetry_gate_status"] = symmetry_gate_status(metrics)
            if failed_gates:
                result = "FAIL"
        except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            result = "FAIL"
            failed_gates = ["analysis_failure"]
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[run_h3b] analysis failed: {exc}\n")

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
        "env_overrides": active_env_overrides(args),
        "shape_multiseed": is_shape_multiseed_preset(args),
        "run_profile": args.run_profile,
        "result": result,
        "failed_gates": failed_gates,
    }
    if args.shape_multiseed_winslow:
        payload["axis_repair_mode_used"] = "full_winslow"
    if args.long_time_sweep:
        payload["long_time_sweep"] = True
        payload["t_end_s"] = t_end_s
    if args.multigrid_ladder:
        payload["multigrid_ladder"] = True
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
        "rho_peak_err_pct": metrics["rho_peak_err_pct"],
        "rho_post_err_pct": metrics["rho_post_err_pct"],
        "P_post_err_pct": metrics["P_post_err_pct"],
        "t_alpha": metrics["t_alpha"],
        "A_rho_max_valid": metrics["A_rho_max_valid"],
        "A_p_max_valid": metrics["A_p_max_valid"],
        "A_rho_max_raw_all_shells": metrics["A_rho_max_raw_all_shells"],
        "A_rho_max_all_shells": metrics["A_rho_max_all_shells"],
        "A_p_max_all_shells": metrics["A_p_max_all_shells"],
        "quadrupole_distortion": metrics["quadrupole_distortion"],
        "quadrupole_a2_over_a0": metrics["quadrupole_a2_over_a0"],
        "quadrupole_a2_amplitude_cm": metrics["quadrupole_a2_amplitude_cm"],
        "A_rho_max_valid_snapshot_max": metrics["A_rho_max_valid_snapshot_max"],
        "A_rho_max_raw_all_shells_snapshot_max": metrics["A_rho_max_raw_all_shells_snapshot_max"],
        "quadrupole_distortion_snapshot_max": metrics["quadrupole_distortion_snapshot_max"],
        "symmetry_gate_status": metrics["symmetry_gate_status"],
        "cons_err_max": metrics["cons_err_max"],
    }
    if args.long_time_sweep:
        run["t_end_s"] = t_end_s
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
        "Stage H3B thresholds: 0.85 <= beta_observed <= 1.5, energy_err_pct <= 1.0, "
        "rho_peak_err_pct <= 25, |t_alpha - 0.4| <= 0.10, "
        "A_rho_max_valid <= 0.10 (cavity-excluded density anisotropy), "
        "quadrupole_distortion <= 0.05 on all grids, "
        "cons_err_max <= 5.15e-5 when enabled"
    )
    print(
        "ladder_row seed C2 cfl ale result beta_obs energy_pct rho_peak_pct P_post_pct "
        "t_alpha A_rho_valid A_rho_raw A_p_valid q2 symmetry_status cons_err_max wall_time_s"
    )
    for row in rows:
        print(
            "{ladder_row} {seed} {av_c2} {cfl} {ale_every} {result} {beta_observed} "
            "{energy_err_pct} {rho_peak_err_pct} {P_post_err_pct} {t_alpha} "
            "{A_rho_max_valid} {A_rho_max_raw_all_shells} {A_p_max_valid} "
            "{quadrupole_distortion} {symmetry_gate_status} {cons_err_max} {wall_time_s:.3f}".format(
                ladder_row=row["ladder_row"],
                seed=row["seed"],
                av_c2=float_token(float(row["av_c2"])),
                cfl=float_token(float(row["cfl"])),
                ale_every=row["ale_every"],
                result=row["result"],
                beta_observed=format_metric(row["beta_observed"]),
                energy_err_pct=format_metric(row["energy_err_pct"]),
                rho_peak_err_pct=format_metric(row["rho_peak_err_pct"]),
                P_post_err_pct=format_metric(row["P_post_err_pct"]),
                t_alpha=format_metric(row["t_alpha"]),
                A_rho_max_valid=format_metric(row["A_rho_max_valid"]),
                A_rho_max_raw_all_shells=format_metric(row["A_rho_max_raw_all_shells"]),
                A_p_max_valid=format_metric(row["A_p_max_valid"]),
                quadrupole_distortion=format_metric(row["quadrupole_distortion"]),
                symmetry_gate_status=format_metric(row["symmetry_gate_status"]),
                cons_err_max=format_metric(row["cons_err_max"]),
                wall_time_s=float(row["wall_time_s"]),
            )
        )


def sample_std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def aggregate_metric(rows: list[dict[str, object]], metric: str) -> dict[str, object]:
    values = [float(row[metric]) for row in rows if isinstance(row.get(metric), (int, float))]
    if not values:
        return {"n": 0, "mean": None, "std": None}
    mean = sum(values) / len(values)
    return {
        "n": len(values),
        "mean": mean,
        "std": sample_std(values, mean),
        "min": min(values),
        "max": max(values),
    }


def write_shape_summary(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    if not args.summary_json:
        return
    metrics = [
        "t_alpha",
        "beta_observed",
        "A_rho_max_valid",
        "A_rho_max_raw_all_shells",
        "A_rho_max_valid_snapshot_max",
        "A_rho_max_raw_all_shells_snapshot_max",
        "quadrupole_distortion",
        "quadrupole_distortion_snapshot_max",
        "cons_err_max",
    ]
    payload = {
        "stage_id": STAGE_ID,
        "stage_label": STAGE_LABEL,
        "summary_id": Path(args.summary_json).stem,
        "shape_multiseed": is_shape_multiseed_preset(args),
        "run_profile": args.run_profile,
        "result": "PASS" if all(row["result"] == "PASS" for row in rows) else "FAIL",
        "n_runs": len(rows),
        "env_overrides": active_env_overrides(args),
        "thresholds": thresholds_payload(),
        "runs": rows,
        "aggregate": {metric: aggregate_metric(rows, metric) for metric in metrics},
    }
    write_json(Path(args.summary_json), payload)


def write_long_time_summary(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    if not args.summary_json:
        return
    summary_rows = [
        {
            "run_id": row["run_id"],
            "result": row["result"],
            "failed_gates": row["failed_gates"],
            "t_end_s": row["t_end_s"],
            "cons_err_max": row["cons_err_max"],
            "A_rho_max_valid_final": row["A_rho_max_valid"],
            "quadrupole_distortion_final": row["quadrupole_distortion"],
            "wall_time_s": row["wall_time_s"],
        }
        for row in rows
    ]
    payload = {
        "stage_id": STAGE_ID,
        "stage_label": STAGE_LABEL,
        "summary_id": Path(args.summary_json).stem,
        "long_time_sweep": True,
        "run_profile": args.run_profile,
        "result": "PASS" if all(row["result"] == "PASS" for row in rows) else "FAIL",
        "n_runs": len(rows),
        "env_overrides": active_env_overrides(args),
        "thresholds": thresholds_payload(),
        "runs": summary_rows,
    }
    write_json(Path(args.summary_json), payload)


def write_multigrid_ladder_summary(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    if not args.summary_json:
        return
    per_grid = [
        {
            "mesh": f"{row['nr']}x{row['nz']}",
            "nr": row["nr"],
            "nz": row["nz"],
            "run_id": row["run_id"],
            "result": row["result"],
            "failed_gates": row["failed_gates"],
            "seed": row["seed"],
            "t_alpha": row["t_alpha"],
            "beta_observed": row["beta_observed"],
            "A_rho_max_valid": row["A_rho_max_valid"],
            "quadrupole_distortion": row["quadrupole_distortion"],
            "cons_err_max": row["cons_err_max"],
            "wall_time_s": row["wall_time_s"],
        }
        for row in rows
    ]
    payload = {
        "stage_id": STAGE_ID,
        "stage_label": STAGE_LABEL,
        "summary_id": Path(args.summary_json).stem,
        "multigrid_ladder": True,
        "run_profile": args.run_profile,
        "result": "PASS" if all(row["result"] == "PASS" for row in rows) else "FAIL",
        "n_runs": len(rows),
        "env_overrides": active_env_overrides(args),
        "thresholds": thresholds_payload(),
        "per_grid": per_grid,
    }
    write_json(Path(args.summary_json), payload)


def write_summary(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    if args.long_time_sweep:
        write_long_time_summary(args, rows)
    elif args.multigrid_ladder:
        write_multigrid_ladder_summary(args, rows)
    else:
        write_shape_summary(args, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/2d_rz_h3b_sedov_sph.py")
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    parser.add_argument("--gpu-model", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--cuda-version", default="12.0")
    parser.add_argument("--profile-class", default="ci_short", choices=("ci_short", "paper_full", "exploratory"))
    parser.add_argument("--evidence-role", default="ci_gate", choices=("ci_gate", "paper_figure", "exploratory"))
    parser.add_argument("--mesh-list", type=parse_mesh_list, default=parse_mesh_list("128,256;256,512"))
    parser.add_argument("--seed-list", type=parse_seed_list, default=parse_seed_list("12345,12346,12347,12348,12349"))
    parser.add_argument("--av-c2-list", type=parse_float_list, default=parse_float_list("1.5"))
    parser.add_argument("--cfl-list", type=parse_float_list, default=parse_float_list("0.1"))
    parser.add_argument("--ale-every-n-list", type=parse_ale_every_n_list, default=parse_ale_every_n_list("0"))
    parser.add_argument("--t-end-s", type=float, default=6.0e-9)
    parser.add_argument("--out-root", default="./build/h3b_runs")
    parser.add_argument("--run-profile", default="H3B spherical Sedov RZ mesh ladder seed sweep")
    parser.add_argument("--run-id-suffix", default="")
    parser.add_argument(
        "--summary-json",
        default="",
        help="optional path for an aggregate JSON summary with mean/std across runs",
    )
    parser.add_argument(
        "--phase-9-11-t31",
        action="store_true",
        help="enable Phase 9-11/T31 H3-B deck environment overrides",
    )
    parser.add_argument(
        "--cons-err-gate",
        action="store_true",
        help="gate energy/history cons_err_max against the Phase 9-11/T31 production threshold",
    )
    shape_group = parser.add_mutually_exclusive_group()
    shape_group.add_argument(
        "--shape-multiseed",
        action="store_true",
        help="run the 256x512 Phase 9-11/T31 Sedov-shape 5-seed production harness",
    )
    shape_group.add_argument(
        "--shape-multiseed-winslow",
        action="store_true",
        help="run the 256x512 Phase 9-11/T31 Sedov-shape 5-seed harness with full_winslow axis repair",
    )
    shape_group.add_argument(
        "--long-time-sweep",
        action="store_true",
        help="run the 256x512 seed=12345 Phase 9-11/T31 long-time drift t_end ladder",
    )
    shape_group.add_argument(
        "--multigrid-ladder",
        action="store_true",
        help="run the 128x256, 256x512, 512x1024 seed=12345 Phase 9-11/T31 mesh ladder",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if is_shape_multiseed_preset(args):
        args.phase_9_11_t31 = True
        args.cons_err_gate = True
        args.mesh_list = parse_mesh_list("256,512")
        args.seed_list = parse_seed_list("12345,12346,12347,12348,12349")
        args.av_c2_list = parse_float_list("1.5")
        args.cfl_list = parse_float_list("0.1")
        args.ale_every_n_list = parse_ale_every_n_list("5")
        args.t_end_s = 6.0e-9
        args.out_root = (
            "./build/h3b_shape_multiseed_winslow"
            if args.shape_multiseed_winslow
            else "./build/h3b_shape_multiseed"
        )
        if args.run_profile == "H3B spherical Sedov RZ mesh ladder seed sweep":
            args.run_profile = (
                "H3B 256x512 Phase 9-11/T31 Sedov shape multi-seed full_winslow sweep"
                if args.shape_multiseed_winslow
                else "H3B 256x512 Phase 9-11/T31 Sedov shape multi-seed sweep"
            )
        if not args.summary_json:
            args.summary_json = (
                "docs/validation/2d_rz/H3/metrics/h3b_shape_multiseed_winslow_summary.json"
                if args.shape_multiseed_winslow
                else "docs/validation/2d_rz/H3/metrics/h3b_shape_multiseed_summary.json"
            )
    if args.long_time_sweep:
        args.phase_9_11_t31 = True
        args.cons_err_gate = True
        args.mesh_list = parse_mesh_list("256,512")
        args.seed_list = parse_seed_list("12345")
        args.av_c2_list = parse_float_list("1.5")
        args.cfl_list = parse_float_list("0.1")
        args.ale_every_n_list = parse_ale_every_n_list("5")
        args.out_root = "./build/h3b_long_time_drift"
        if args.run_profile == "H3B spherical Sedov RZ mesh ladder seed sweep":
            args.run_profile = "H3B 256x512 Phase 9-11/T31 long-time drift sweep"
        if not args.summary_json:
            args.summary_json = "docs/validation/2d_rz/H3/metrics/h3b_long_time_drift_summary.json"
    if args.multigrid_ladder:
        args.phase_9_11_t31 = True
        args.cons_err_gate = True
        args.mesh_list = parse_mesh_list(MULTIGRID_LADDER_MESH_LIST)
        args.seed_list = parse_seed_list("12345")
        args.av_c2_list = parse_float_list("1.5")
        args.cfl_list = parse_float_list("0.1")
        args.ale_every_n_list = parse_ale_every_n_list("5")
        args.t_end_s = 6.0e-9
        args.out_root = "./build/h3b_multigrid_ladder"
        if args.run_profile == "H3B spherical Sedov RZ mesh ladder seed sweep":
            args.run_profile = "H3B Phase 9-11/T31 multigrid ladder sweep"
        if not args.summary_json:
            args.summary_json = "docs/validation/2d_rz/H3/metrics/h3b_multigrid_ladder_summary.json"

    mesh_list = list(args.mesh_list)
    if args.quick:
        mesh_list = mesh_list[:1]
    seed_list = list(args.seed_list)
    if args.quick and is_shape_multiseed_preset(args):
        seed_list = seed_list[:1]
    ale_every_n_list = (
        parse_ale_every_n_list("0")
        if args.quick
        and not (is_shape_multiseed_preset(args) or args.long_time_sweep or args.multigrid_ladder)
        else list(args.ale_every_n_list)
    )
    t_end_list = list(LONG_TIME_T_END_LADDER_S) if args.long_time_sweep else [args.t_end_s]
    if args.quick and args.long_time_sweep:
        t_end_list = t_end_list[:1]

    matrix = [
        (nr, nz, seed, av_c2, cfl, ale_every, t_end_s)
        for nr, nz in mesh_list
        for seed in seed_list
        for av_c2 in args.av_c2_list
        for cfl in args.cfl_list
        for ale_every in ale_every_n_list
        for t_end_s in t_end_list
    ]

    if args.dry_run:
        label = "quick " if args.quick else ""
        print(f"Planned Stage H3B {label}runs:")
        if args.phase_9_11_t31:
            print("Phase 9-11/T31 env overrides:")
            for key, value in sorted(active_env_overrides(args).items()):
                print(f"  {key}={value}")
        for nr, nz, seed, av_c2, cfl, ale_every, t_end_s in matrix:
            run_id = (
                run_id_for(nr, nz, seed, av_c2, cfl, ale_every)
                + t_end_run_id_suffix(args, t_end_s)
                + run_id_suffix(args)
            )
            t_end_label = f" t_end_s={t_end_s:.12g}" if args.long_time_sweep else ""
            print(
                f"{run_id} nr{nr}_nz{nz} seed={seed} "
                f"C2={float_token(av_c2)} cfl={float_token(cfl)} ale={ale_every}{t_end_label}"
            )
        print(f"total_runs={len(matrix)}")
        if args.summary_json:
            print(f"summary_json={args.summary_json}")
        return 0

    rows = [
        run_one(args, nr, nz, seed, av_c2, cfl, ale_every, t_end_s)
        for nr, nz, seed, av_c2, cfl, ale_every, t_end_s in matrix
    ]
    print_summary(rows)
    write_summary(args, rows)
    return 0 if all(row["result"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
