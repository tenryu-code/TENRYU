#!/usr/bin/env python3
"""Run Stage RH2 2D RZ hydro+S_N validation sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE_ID = "RH2"
MODES = (
    "shock_tube_grey_sn",
    "cylindrical_blast_sn",
    "sn_vs_fld_thick",
    "optically_thin_precursor_sn",
)
FULL_MESHES_BY_MODE: dict[str, tuple[tuple[int, int], ...]] = {mode: ((32, 64),) for mode in MODES}
SMOKE_MESHES_BY_MODE = FULL_MESHES_BY_MODE

NUMBER_RE = r"[-+0-9.eE]+"
CLAMP_RE = re.compile(r"clamp_count=(?P<count>\d+)")
SN_TIMING_RE = re.compile(r"\[sn_2d_rz_timing\].*?outer_iters=(?P<outer>\d+)")
DECK_LINE_RE = re.compile(
    r"\[deck:2d_rz_rh2_hydro_sn\].*?radiation_mode=(?P<rad_mode>\w+).*?"
    r"groups=(?P<groups>\d+).*?rho_gcc=(?P<rho>{n}).*?"
    r"kappa_a_cm2_g=(?P<kappa>{n}).*?t_end_s=(?P<t_end>{n})".format(n=NUMBER_RE)
)

A_EV = 1.3720e2
E_FLOOR = 1.0e-300
ENERGY_TOL = 1.0e-3
THICK_TE_SHELL_AVG_REL_TOL = 5.0e-2


@dataclass(frozen=True)
class Mesh:
    nr: int
    nz: int
    r_nodes: Any
    z_nodes: Any
    vol: Any


def parse_mesh_list(value: str) -> list[tuple[int, int]]:
    out = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        sep = "x" if "x" in item.lower() else ","
        parts = [part.strip() for part in item.lower().split(sep)]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"invalid mesh entry: {item!r}")
        nr = int(parts[0])
        nz = int(parts[1])
        if nr <= 0 or nz <= 0:
            raise argparse.ArgumentTypeError("mesh dimensions must be positive")
        out.append((nr, nz))
    if not out:
        raise argparse.ArgumentTypeError("mesh list is empty")
    return out


def parse_csv(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    out = [part.strip().lower().replace("-", "_") for part in value.split(",") if part.strip()]
    bad = [part for part in out if part not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(f"invalid {label}: {', '.join(bad)}")
    if not out:
        raise argparse.ArgumentTypeError(f"{label} list is empty")
    return out


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("seed list is empty")
    return seeds


def safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def mode_rho_default(mode: str) -> float:
    if mode == "cylindrical_blast_sn":
        return 1.0e-3
    if mode == "optically_thin_precursor_sn":
        return 1.0e-5
    return 1.0


def mode_kappa_default(mode: str) -> float:
    if mode == "sn_vs_fld_thick":
        return 1.0e4
    if mode == "optically_thin_precursor_sn":
        return 0.01
    return 10.0


def default_max_steps(mode: str) -> int:
    return 5 if mode == "sn_vs_fld_thick" else 3


def case_name(mode: str, rad_mode: str, nr: int, nz: int, rho_gcc: float, kappa_a: float, seed: int) -> str:
    return (
        f"rh2_{mode}_{rad_mode}_nr{nr}_nz{nz}_rho{safe_float_token(rho_gcc)}"
        f"_ka{safe_float_token(kappa_a)}_g1_seed{seed}"
    )


def output_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=False) + "\n")


def discover_results_dir(outdir: Path) -> Path:
    def has_numbered_plot(path: Path) -> bool:
        return path.exists() and any(
            len(p.stem) >= 5 and p.stem[-4:].isdigit() and p.stem[-5] == "_"
            for p in path.glob("*.h5")
        )

    direct = outdir / "results"
    if has_numbered_plot(direct):
        return direct
    candidates = []
    for path in outdir.parent.glob(f"{outdir.name}_*"):
        candidate = path / "results"
        if has_numbered_plot(candidate):
            candidates.append(candidate)
    if candidates:
        return max(candidates, key=lambda path: path.parent.stat().st_mtime)
    raise FileNotFoundError(f"no plot files found under {outdir} or {outdir}_*/results")


def locate_plot_files(results_dir: Path, run_id: str) -> list[Path]:
    def plot_index(path: Path) -> int:
        return int(path.stem.rsplit("_", 1)[1])

    files = sorted(
        (p for p in results_dir.glob(f"{run_id}_*.h5") if p.stem.removeprefix(f"{run_id}_").isdigit()),
        key=plot_index,
    )
    if files:
        return files
    return sorted((p for p in results_dir.glob("*_*.h5") if p.stem.rsplit("_", 1)[-1].isdigit()), key=plot_index)


def reshape_cells(values: Any, nr: int, nz: int, name: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == nr * nz:
        return arr.reshape((nr, nz))
    if arr.shape == (nr, nz):
        return arr
    raise ValueError(f"{name} has shape {arr.shape}, expected {(nr, nz)} or flat {nr * nz}")


def reshape_nodes(values: Any, nr: int, nz: int, name: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == (nr + 1) * (nz + 1):
        return arr.reshape((nr + 1, nz + 1))
    if arr.shape == (nr + 1, nz + 1):
        return arr
    raise ValueError(f"{name} has shape {arr.shape}, expected {(nr + 1, nz + 1)}")


def scalar_dataset(handle: Any, path: str, default: float | None = None) -> float | None:
    import numpy as np

    if path not in handle:
        return default
    arr = np.asarray(handle[path][()], dtype=float).reshape(-1)
    if arr.size == 0:
        return default
    return float(arr[-1])


def read_snapshot(path: Path, nr: int, nz: int) -> dict[str, Any]:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        t_value = float(handle.attrs.get("t", 0.0))
        if "time_state/t" in handle:
            t_value = float(np.asarray(handle["time_state/t"][()]).reshape(-1)[0])
        r_node_path = "mesh/x_r" if "mesh/x_r" in handle else "mesh/v_r"
        z_node_path = "mesh/x_z" if "mesh/x_z" in handle else "mesh/v_z"
        rad_E = reshape_cells(handle["radiation/energy_density"][()], nr, nz, "radiation/energy_density")
        return {
            "path": str(path),
            "t": t_value,
            "rho": reshape_cells(handle["hydro/rho"][()], nr, nz, "hydro/rho"),
            "Te": reshape_cells(handle["hydro/Te"][()], nr, nz, "hydro/Te"),
            "Ti": reshape_cells(handle["hydro/Ti"][()], nr, nz, "hydro/Ti"),
            "ee": reshape_cells(handle["hydro/ee"][()], nr, nz, "hydro/ee"),
            "ei": reshape_cells(handle["hydro/ei"][()], nr, nz, "hydro/ei"),
            "vol": reshape_cells(handle["hydro/vol"][()], nr, nz, "hydro/vol"),
            "rad_E": rad_E,
            "r_nodes": reshape_nodes(handle[r_node_path][()], nr, nz, r_node_path),
            "z_nodes": reshape_nodes(handle[z_node_path][()], nr, nz, z_node_path),
            "v_r_nodes": reshape_nodes(handle["mesh/v_r"][()], nr, nz, "mesh/v_r") if "mesh/v_r" in handle else None,
            "v_z_nodes": reshape_nodes(handle["mesh/v_z"][()], nr, nz, "mesh/v_z") if "mesh/v_z" in handle else None,
            "node_mass": reshape_nodes(handle["hydro/node_mass"][()], nr, nz, "hydro/node_mass")
            if "hydro/node_mass" in handle
            else None,
            "E_rad_escaped_state": scalar_dataset(handle, "time_state/E_rad_escaped", 0.0),
        }


def read_snapshots(results_dir: Path, run_id: str, nr: int, nz: int) -> list[dict[str, Any]]:
    files = locate_plot_files(results_dir, run_id)
    if len(files) < 2:
        raise RuntimeError("expected at least initial and final plot snapshots")
    return [read_snapshot(path, nr, nz) for path in files]


def parse_log_metadata(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    out: dict[str, Any] = {
        "n_clamp": sum(int(match.group("count")) for match in CLAMP_RE.finditer(text)),
        "sn_outer_iterations_max": None,
        "sn_outer_iterations_series": [],
    }
    deck_match = DECK_LINE_RE.search(text)
    if deck_match:
        out["deck_radiation_mode"] = deck_match.group("rad_mode")
        out["deck_groups"] = int(deck_match.group("groups"))
        out["deck_rho_gcc"] = float(deck_match.group("rho"))
        out["deck_kappa_a_cm2_g"] = float(deck_match.group("kappa"))
        out["deck_t_end_s"] = float(deck_match.group("t_end"))
    outer = [int(match.group("outer")) for match in SN_TIMING_RE.finditer(text)]
    if outer:
        out["sn_outer_iterations_series"] = outer
        out["sn_outer_iterations_max"] = max(outer)
    return out


def compute_E_rad_total(rad_E: Any, mesh: Mesh) -> float:
    import numpy as np

    return float(np.sum(np.asarray(rad_E, dtype=float) * np.asarray(mesh.vol, dtype=float)))


def compute_lte_residual(rad_E: Any, Te: Any, mask: Any) -> dict[str, Any]:
    import numpy as np

    rad = np.asarray(rad_E, dtype=float)
    lte = A_EV * np.asarray(Te, dtype=float) ** 4
    denom = np.maximum(np.maximum(np.abs(rad), np.abs(lte)), E_FLOOR)
    residual = np.abs(rad - lte) / denom
    masked = residual[np.asarray(mask, dtype=bool)]
    if masked.size == 0:
        return {"lte_residual_max": None, "lte_residual_mean": None, "interior_cell_count": 0}
    return {
        "lte_residual_max": float(np.nanmax(masked)),
        "lte_residual_mean": float(np.nanmean(masked)),
        "interior_cell_count": int(masked.size),
    }


def compute_thick_lte_mask(
    final_snapshot: dict, mesh: Mesh, kappa_a_cm2_g: float, rho_gcc: float, n_boundary_cells: int = 1
) -> Any:
    """Boolean (nr, nz) mask: True where κ·ρ·dx > 1 in BOTH r and z directions, boundaries stripped.

    For sn_vs_fld_thick (κ=1e4, ρ=1, dx≈0.03125), τ_cell ≈ 312 → mask covers entire interior.
    """
    import numpy as np

    nr, nz = mesh.nr, mesh.nz
    r_nodes = np.asarray(mesh.r_nodes, dtype=float)
    z_nodes = np.asarray(mesh.z_nodes, dtype=float)
    dr = float(np.median(np.diff(r_nodes, axis=0))) if r_nodes.shape[0] >= 2 else 0.0
    dz = float(np.median(np.diff(z_nodes, axis=1))) if z_nodes.shape[1] >= 2 else 0.0
    rho_arr = np.asarray(final_snapshot["rho"], dtype=float)
    tau_r = kappa_a_cm2_g * rho_arr * dr
    tau_z = kappa_a_cm2_g * rho_arr * dz
    mask = (tau_r > 1.0) & (tau_z > 1.0)
    if n_boundary_cells > 0:
        mask[:n_boundary_cells, :] = False
        mask[max(nr - n_boundary_cells, 0):, :] = False
        mask[:, :n_boundary_cells] = False
        mask[:, max(nz - n_boundary_cells, 0):] = False
    return mask


def compute_U_internal_total(rho: Any, ee: Any, ei: Any, mesh: Mesh) -> float:
    import numpy as np

    return float(np.sum(np.asarray(rho) * (np.asarray(ee) + np.asarray(ei)) * np.asarray(mesh.vol)))


def compute_K_node_total(snapshot: dict[str, Any]) -> float:
    import numpy as np

    if snapshot.get("node_mass") is None or snapshot.get("v_r_nodes") is None or snapshot.get("v_z_nodes") is None:
        return 0.0
    return float(0.5 * np.sum(snapshot["node_mass"] * (snapshot["v_r_nodes"] ** 2 + snapshot["v_z_nodes"] ** 2)))


def rel_delta(a: float, b: float) -> float:
    return abs(b - a) / max(abs(a), E_FLOOR)


def mesh_from_snapshot(snapshot: dict[str, Any], nr: int, nz: int) -> Mesh:
    return Mesh(nr=nr, nz=nz, r_nodes=snapshot["r_nodes"], z_nodes=snapshot["z_nodes"], vol=snapshot["vol"])


def z_shell_avg(field: Any, vol: Any) -> list[float]:
    import numpy as np

    arr = np.asarray(field, dtype=float)
    weights = np.asarray(vol, dtype=float)
    denom = np.maximum(np.sum(weights, axis=0), E_FLOOR)
    return [float(x) for x in np.sum(arr * weights, axis=0) / denom]


def common_metrics(
    mode: str,
    nr: int,
    nz: int,
    kappa_a_cm2_g: float,
    rho_gcc: float,
    snapshots: list[dict[str, Any]],
    log_metrics: dict[str, Any],
    wall_time_s: float,
    bytes_out: int,
) -> dict[str, Any]:
    import numpy as np

    first = snapshots[0]
    final = snapshots[-1]
    first_mesh = mesh_from_snapshot(first, nr, nz)
    final_mesh = mesh_from_snapshot(final, nr, nz)
    rad0 = compute_E_rad_total(first["rad_E"], first_mesh)
    rad1 = compute_E_rad_total(final["rad_E"], final_mesh)
    u0 = compute_U_internal_total(first["rho"], first["ee"], first["ei"], first_mesh)
    u1 = compute_U_internal_total(final["rho"], final["ee"], final["ei"], final_mesh)
    k0 = compute_K_node_total(first)
    k1 = compute_K_node_total(final)
    escaped = float(final.get("E_rad_escaped_state") or 0.0)
    metrics: dict[str, Any] = {
        "a_eV_erg_cm3_eV4": A_EV,
        "groups": 1,
        "rho_min": float(np.nanmin([np.nanmin(s["rho"]) for s in snapshots])),
        "rho_max": float(np.nanmax([np.nanmax(s["rho"]) for s in snapshots])),
        "Te_min_eV": float(np.nanmin([np.nanmin(s["Te"]) for s in snapshots])),
        "Te_max_eV": float(np.nanmax([np.nanmax(s["Te"]) for s in snapshots])),
        "rad_E_min": float(np.nanmin([np.nanmin(s["rad_E"]) for s in snapshots])),
        "rad_E_max": float(np.nanmax([np.nanmax(s["rad_E"]) for s in snapshots])),
        "rad_E_nan_count": int(sum(np.count_nonzero(~np.isfinite(s["rad_E"])) for s in snapshots)),
        "rad_E_negative_count": int(sum(np.count_nonzero(np.asarray(s["rad_E"]) < 0.0) for s in snapshots)),
        "U_internal_initial": u0,
        "U_internal_final": u1,
        "dU_internal_rel": rel_delta(u0, u1),
        "K_node_initial": k0,
        "K_node_final": k1,
        "E_rad_initial": rad0,
        "E_rad_final": rad1,
        "dE_rad_rel": rel_delta(rad0, rad1),
        "E_rad_escaped": escaped,
        "E_balance_rel": abs((u1 + k1 + rad1 + escaped) - (u0 + k0 + rad0)) / max(abs(u0 + k0 + rad0), E_FLOOR),
        "Te_shell_avg_final": z_shell_avg(final["Te"], final["vol"]),
        "wall_time_s": wall_time_s,
        "output_bytes": bytes_out,
    }
    metrics.update(log_metrics)
    if mode == "sn_vs_fld_thick":
        thick_mask = compute_thick_lte_mask(final, final_mesh, kappa_a_cm2_g, rho_gcc, n_boundary_cells=1)
        lte_metrics = compute_lte_residual(final["rad_E"], final["Te"], thick_mask)
        metrics.update(lte_metrics)
        metrics["lte_residual_gate_status"] = (
            "skipped_empty_interior_mask"
            if lte_metrics.get("interior_cell_count", 0) == 0
            else "PASS"
            if lte_metrics.get("lte_residual_max", float("inf")) <= 1.0e-3
            else "FAIL"
        )
        metrics["lte_residual_tolerance"] = 1.0e-3
    return metrics


def evaluate_gates(mode: str, rad_mode: str, metrics: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    if metrics.get("rad_E_nan_count", 0) != 0:
        failed.append("rh2_rad_E_finite")
    if metrics.get("rho_min", 0.0) <= 0.0:
        failed.append("rh2_positive_density")
    if metrics.get("Te_min_eV", 0.0) <= 0.0:
        failed.append("rh2_positive_temperature")
    if metrics.get("rad_E_negative_count", 0) != 0:
        failed.append("rh2_rad_E_nonnegative")
    if mode in ("shock_tube_grey_sn", "cylindrical_blast_sn", "optically_thin_precursor_sn"):
        if metrics.get("E_balance_rel", math.inf) > ENERGY_TOL:
            failed.append(f"rh2_{mode}_energy_balance")
    if mode == "sn_vs_fld_thick":
        if metrics.get("interior_cell_count", 0) > 0 and metrics.get("lte_residual_max", math.inf) > 1.0e-3:
            failed.append("rh2_thick_interior_lte_residual")
    return failed


def active_env(
    args: argparse.Namespace,
    mode: str,
    rad_mode: str,
    nr: int,
    nz: int,
    seed: int,
    outdir: Path,
) -> dict[str, str]:
    env = {
        "TENRYU_RH2_MODE": mode,
        "TENRYU_RH2_RADIATION_MODE_OVERRIDE": rad_mode,
        "TENRYU_RH2_NR": str(nr),
        "TENRYU_RH2_NZ": str(nz),
        "TENRYU_RH2_SEED": str(seed),
        "TENRYU_RH2_OUTDIR": str(outdir),
        "TENRYU_RH2_MAX_STEPS": str(args.max_steps if args.max_steps is not None else default_max_steps(mode)),
        "TENRYU_RH2_VERBOSITY": args.verbosity,
    }
    if args.t_end_s is not None:
        env["TENRYU_RH2_T_END_S"] = str(args.t_end_s)
    if args.dt_initial_s is not None:
        env["TENRYU_RH2_DT_INITIAL_S"] = str(args.dt_initial_s)
    if args.dt_max_s is not None:
        env["TENRYU_RH2_DT_MAX_S"] = str(args.dt_max_s)
    if args.rho_gcc is not None:
        env["TENRYU_RH2_RHO_GCC"] = str(args.rho_gcc)
    if args.kappa_a_cm2_g is not None:
        env["TENRYU_RH2_KAPPA_A_CM2_G"] = str(args.kappa_a_cm2_g)
    return env


def effective_rho(args: argparse.Namespace, mode: str) -> float:
    return float(args.rho_gcc if args.rho_gcc is not None else mode_rho_default(mode))


def effective_kappa(args: argparse.Namespace, mode: str) -> float:
    return float(args.kappa_a_cm2_g if args.kappa_a_cm2_g is not None else mode_kappa_default(mode))


def run_radiation_case(
    args: argparse.Namespace,
    mode: str,
    rad_mode: str,
    nr: int,
    nz: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, str], int, float]:
    rho_gcc = effective_rho(args, mode)
    kappa_a = effective_kappa(args, mode)
    run_id = case_name(mode, rad_mode, nr, nz, rho_gcc, kappa_a, seed)
    outdir = Path(args.out_root) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    env_overrides = active_env(args, mode, rad_mode, nr, nz, seed, outdir)
    env = os.environ.copy()
    env.update(env_overrides)
    log_path = outdir / "run.log"

    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([args.tenryu_bin, "run", args.deck], env=env, stdout=log, stderr=log)
    wall_time_s = time.monotonic() - start

    log_metrics = parse_log_metadata(log_path)
    if proc.returncode != 0:
        return (
            {**log_metrics, "wall_time_s": wall_time_s, "output_bytes": output_bytes(outdir)},
            env_overrides,
            proc.returncode,
            wall_time_s,
        )
    results_dir = discover_results_dir(outdir)
    effective_outdir = results_dir.parent
    bytes_out = output_bytes(effective_outdir)
    if effective_outdir != outdir:
        bytes_out += output_bytes(outdir)
    snapshots = read_snapshots(results_dir, run_id, nr, nz)
    metrics = common_metrics(mode, nr, nz, kappa_a, rho_gcc, snapshots, log_metrics, wall_time_s, bytes_out)
    return metrics, env_overrides, proc.returncode, wall_time_s


def thick_compare(sn_metrics: dict[str, Any], fld_metrics: dict[str, Any]) -> tuple[float | None, list[str]]:
    import numpy as np

    failed: list[str] = []
    sn = np.asarray(sn_metrics.get("Te_shell_avg_final", []), dtype=float)
    fld = np.asarray(fld_metrics.get("Te_shell_avg_final", []), dtype=float)
    if sn.size == 0 or fld.size == 0 or sn.shape != fld.shape:
        return None, ["rh2_thick_shell_avg_unavailable"]
    rel = float(np.max(np.abs(sn - fld) / np.maximum(np.abs(fld), E_FLOOR)))
    if rel > THICK_TE_SHELL_AVG_REL_TOL:
        failed.append("rh2_thick_sn_fld_Te_shell_avg")
    return rel, failed


def run_one(args: argparse.Namespace, mode: str, nr: int, nz: int, seed: int) -> dict[str, Any]:
    rho_gcc = effective_rho(args, mode)
    kappa_a = effective_kappa(args, mode)
    primary_run_id = case_name(mode, "sn_transport", nr, nz, rho_gcc, kappa_a, seed)
    ladder_row = f"{mode}_nr{nr}_nz{nz}"
    metrics_ref = f"{args.root.rstrip('/')}/{STAGE_ID}/metrics/{primary_run_id}.json"

    result = "PASS"
    failed_gates: list[str] = []
    failure_class: str | None = None
    env_overrides: dict[str, Any] = {}
    exit_code = 0
    wall_time_s = 0.0
    metrics: dict[str, Any]
    aux_metrics: dict[str, Any] = {}

    try:
        metrics, sn_env, exit_code, wall_time_s = run_radiation_case(args, mode, "sn_transport", nr, nz, seed)
        env_overrides["sn_transport"] = sn_env
        failed_gates = ["run_failure"] if exit_code != 0 else evaluate_gates(mode, "sn_transport", metrics)
        if mode == "sn_vs_fld_thick" and exit_code == 0:
            fld_metrics, fld_env, fld_exit_code, fld_wall = run_radiation_case(
                args, mode, "multigroup_diffusion", nr, nz, seed
            )
            env_overrides["multigroup_diffusion"] = fld_env
            aux_metrics["multigroup_diffusion"] = fld_metrics
            exit_code = exit_code or fld_exit_code
            wall_time_s += fld_wall
            if fld_exit_code != 0:
                failed_gates.append("rh2_thick_fld_reference_run_failure")
            else:
                failed_gates.extend(evaluate_gates(mode, "multigroup_diffusion", fld_metrics))
                rel, compare_failed = thick_compare(metrics, fld_metrics)
                metrics["thick_Te_shell_avg_rel_linf"] = rel
                metrics["thick_Te_shell_avg_rel_tol"] = THICK_TE_SHELL_AVG_REL_TOL
                failed_gates.extend(compare_failed)
    except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        metrics = {"analysis_error": str(exc), "wall_time_s": wall_time_s}
        failed_gates = ["analysis_failure"]
        failure_class = "analysis_failure"

    if failed_gates:
        result = "CRASHED" if failed_gates == ["run_failure"] else "FAIL"
        failure_class = failure_class or ("run_failure" if "run_failure" in failed_gates else "gate_failure")

    payload = {
        "run_id": primary_run_id,
        "stage_id": STAGE_ID,
        "ladder_row": ladder_row,
        "seed": seed,
        "deck_path": args.deck,
        "mode": mode,
        "nr": nr,
        "nz": nz,
        "groups": 1,
        "rho_gcc": rho_gcc,
        "kappa_a_cm2_g": kappa_a,
        "result": result,
        "failure_class": failure_class,
        "failed_gates": failed_gates,
        "exit_code": exit_code,
        "wall_time_s": wall_time_s,
        "metrics": metrics,
        "aux_metrics": aux_metrics,
        "env_overrides": env_overrides,
    }
    write_json(Path(metrics_ref), payload)
    return row_from_payload(payload, metrics_ref)


def row_from_payload(payload: dict[str, Any], metrics_ref: str) -> dict[str, Any]:
    metrics = payload["metrics"]
    return {
        "run_id": payload["run_id"],
        "stage_id": payload["stage_id"],
        "ladder_row": payload["ladder_row"],
        "seed": payload["seed"],
        "deck_path": payload["deck_path"],
        "mode": payload["mode"],
        "nr": payload["nr"],
        "nz": payload["nz"],
        "groups": payload["groups"],
        "rho_gcc": payload["rho_gcc"],
        "kappa_a_cm2_g": payload["kappa_a_cm2_g"],
        "result": payload["result"],
        "failure_class": payload["failure_class"],
        "failed_gates": payload["failed_gates"],
        "exit_code": payload["exit_code"],
        "metrics_ref": metrics_ref,
        "wall_time_s": payload["wall_time_s"],
        "output_bytes": metrics.get("output_bytes"),
        "E_balance_rel": metrics.get("E_balance_rel"),
        "rad_E_min": metrics.get("rad_E_min"),
        "rad_E_max": metrics.get("rad_E_max"),
        "sn_outer_iterations_max": metrics.get("sn_outer_iterations_max"),
        "thick_Te_shell_avg_rel_linf": metrics.get("thick_Te_shell_avg_rel_linf"),
        "lte_residual_max": metrics.get("lte_residual_max"),
        "lte_residual_mean": metrics.get("lte_residual_mean"),
        "lte_residual_gate_status": metrics.get("lte_residual_gate_status"),
        "interior_cell_count": metrics.get("interior_cell_count"),
        "n_clamp": metrics.get("n_clamp"),
    }


def matrix_rows(args: argparse.Namespace) -> list[tuple[str, int, int, int]]:
    rows = []
    for mode in args.mode_list:
        meshes = tuple(args.mesh_list) if args.mesh_list is not None else (
            SMOKE_MESHES_BY_MODE[mode] if args.smoke else FULL_MESHES_BY_MODE[mode]
        )
        for nr, nz in meshes:
            for seed in args.seed_list:
                rows.append((mode, nr, nz, seed))
    return rows


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def write_summary(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    aggregate = {
        "total_runs": len(rows),
        "passing_runs": sum(1 for row in rows if row["result"] == "PASS"),
        "crashed_runs": sum(1 for row in rows if row["result"] == "CRASHED"),
        "failed_runs": sum(1 for row in rows if row["result"] == "FAIL"),
        "a_eV_erg_cm3_eV4": A_EV,
    }
    for key in (
        "wall_time_s",
        "E_balance_rel",
        "rad_E_max",
        "sn_outer_iterations_max",
        "thick_Te_shell_avg_rel_linf",
        "lte_residual_max",
        "n_clamp",
    ):
        values = finite_values(rows, key)
        aggregate[f"{key}_max"] = max(values) if values else None
    payload = {
        "stage_id": STAGE_ID,
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "matrix": rows,
        "aggregate": aggregate,
    }
    write_json(Path(args.summary_json), payload)


def update_stage_json(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    stage_path = Path(args.root) / STAGE_ID / "stage.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8")) if stage_path.exists() else {"stage_id": STAGE_ID}
    stage["attempted_runs"] = int(stage.get("attempted_runs", 0)) + len(rows)
    stage["passing_runs"] = int(stage.get("passing_runs", 0)) + sum(1 for row in rows if row["result"] == "PASS")
    stage["last_attempt_status"] = "PASS" if rows and all(row["result"] == "PASS" for row in rows) else "FAIL"
    if rows:
        stage["status"] = "PASSED" if all(row["result"] == "PASS" for row in rows) else "IN_PROGRESS"
    write_json(stage_path, stage)


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("Stage RH2 hydro+S_N sweep summary")
    print(
        "ladder_row seed result E_balance_rel sn_outer_iterations_max thick_Te_shell_avg_rel_linf "
        "lte_residual_max failed_gates"
    )
    for row in rows:
        print(
            "{ladder_row} {seed} {result} {E_balance_rel} {sn_outer_iterations_max} "
            "{thick_Te_shell_avg_rel_linf} {lte_residual_max} {failed_gates}".format(**row)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/2d_rz_rh2_hydro_sn.py")
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    parser.add_argument("--out-root", default="./build/rh2_runs")
    parser.add_argument("--modes", "--mode-list", dest="mode_list", type=lambda v: parse_csv(v, MODES, "mode"), default=list(MODES))
    parser.add_argument("--mesh-list", type=parse_mesh_list, default=None)
    parser.add_argument("--seed-list", type=parse_seed_list, default=parse_seed_list("12345"))
    parser.add_argument("--smoke", action="store_true", help="use each mode's smallest grid only")
    parser.add_argument("--rho-gcc", type=float, default=None)
    parser.add_argument("--kappa-a-cm2-g", type=float, default=None)
    parser.add_argument("--t-end-s", type=float, default=None)
    parser.add_argument("--dt-initial-s", type=float, default=None)
    parser.add_argument("--dt-max-s", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--verbosity", choices=("quiet", "normal", "verbose"), default="verbose")
    parser.add_argument("--summary-json", default="docs/validation/2d_rz/RH2/stats.json")
    parser.add_argument("--runs-jsonl", default="docs/validation/2d_rz/RH2/runs.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = [run_one(args, mode, nr, nz, seed) for mode, nr, nz, seed in matrix_rows(args)]
    print_summary(rows)
    write_summary(args, rows)
    append_jsonl(Path(args.runs_jsonl), rows)
    update_stage_json(args, rows)
    return 0 if rows and all(row["result"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
