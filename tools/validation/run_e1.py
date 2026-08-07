#!/usr/bin/env python3
"""Run Stage E1 2D RZ TMAT EOS dry-run validation sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE_ID = "E1"
MODES = (
    "static_cd_target",
    "static_cd_uniform",
    "isentropic",
    "noh_weak",
    "ale_remap",
    "cold_corner_sentinel",
    "cold_corner_guard_negative",
)
SMOKE_MESHES = ((64, 128),)
FULL_MESHES = ((64, 128), (128, 256))

NUMBER_RE = r"[-+0-9.eE]+"
CLAMP_RE = re.compile(r"clamp_count=(?P<count>\d+)")
REMAP_DELTA_RE = re.compile(
    rf"\[ale-stats\] remap_delta .*?n_applied=(?P<n>\d+) "
    rf".*?dM=(?P<dM>{NUMBER_RE}) .*?dM_rel=(?P<dM_rel>{NUMBER_RE}) "
    rf".*?dE=(?P<dE>{NUMBER_RE}) .*?dE_rel=(?P<dE_rel>{NUMBER_RE})"
)
TABLE_BOUNDARY_RE = re.compile(
    r"(table[-_ ]?boundary|boundary excursion|outside.*table|table.*out of range)",
    re.IGNORECASE,
)

EV_TO_ERG = 1.6022e-12
M_P = 1.6726e-24
A_CD = 7.0
GAMMA = 5.0 / 3.0
CS_CD_10EV = 1.0e6
COLD_RHO_GCC = 4.4e-5
COLD_T_EV = 0.24
CV_FLOOR_THRESHOLD = 1.0e-3
STATIC_DELTA_TOL = 1.0e-12
STATIC_UNIFORM_DRIFT_TOL = 1.0e-6
STATIC_UNIFORM_ENTROPY_TOL = 1.0e-6
STATIC_UNIFORM_RHO_SPREAD_TOL = 1.0e-6
STATIC_UNIFORM_TE_SPREAD_TOL = 5.0e-2
ISENTROPIC_ENERGY_DRIFT_TOL = 1.0e-3


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


def case_name(mode: str, nr: int, nz: int, seed: int, rho_label: float) -> str:
    return f"e1_{mode}_nr{nr}_nz{nz}_rho{safe_float_token(rho_label)}_seed{seed}"


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
        return any(
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


def read_snapshot(path: Path, nr: int, nz: int) -> dict[str, Any]:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        t_value = float(handle.attrs.get("t", 0.0))
        if "time_state/t" in handle:
            t_value = float(np.asarray(handle["time_state/t"][()]).reshape(-1)[0])
        data = {
            "path": str(path),
            "t": t_value,
            "rho": reshape_cells(handle["hydro/rho"][()], nr, nz, "hydro/rho"),
            "Te": reshape_cells(handle["hydro/Te"][()], nr, nz, "hydro/Te"),
            "Ti": reshape_cells(handle["hydro/Ti"][()], nr, nz, "hydro/Ti"),
            "ee": reshape_cells(handle["hydro/ee"][()], nr, nz, "hydro/ee"),
            "ei": reshape_cells(handle["hydro/ei"][()], nr, nz, "hydro/ei"),
            "Pe": reshape_cells(handle["hydro/Pe"][()], nr, nz, "hydro/Pe"),
            "Pi": reshape_cells(handle["hydro/Pi"][()], nr, nz, "hydro/Pi"),
            "mass": reshape_cells(handle["hydro/mass"][()], nr, nz, "hydro/mass"),
            "vol": reshape_cells(handle["hydro/vol"][()], nr, nz, "hydro/vol"),
            "zbar": reshape_cells(handle["hydro/zbar"][()], nr, nz, "hydro/zbar"),
            "v_r": reshape_nodes(handle["mesh/v_r"][()], nr, nz, "mesh/v_r"),
            "v_z": reshape_nodes(handle["mesh/v_z"][()], nr, nz, "mesh/v_z"),
        }
        if "hydro/volFrac" in handle:
            vf = np.asarray(handle["hydro/volFrac"][()], dtype=float)
            if vf.shape == (nr * nz, 1):
                data["volfrac"] = vf[:, 0].reshape((nr, nz))
            elif vf.shape == (nr, nz, 1):
                data["volfrac"] = vf[:, :, 0]
            elif vf.size == nr * nz:
                data["volfrac"] = vf.reshape((nr, nz))
            else:
                data["volfrac"] = vf
        return data


def read_snapshots(results_dir: Path, run_id: str, nr: int, nz: int) -> list[dict[str, Any]]:
    files = locate_plot_files(results_dir, run_id)
    if len(files) < 2:
        raise RuntimeError("expected at least initial and final plot snapshots")
    return [read_snapshot(path, nr, nz) for path in files]


def parse_clamp_count(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return sum(int(match.group("count")) for match in CLAMP_RE.finditer(text))


def parse_ale_stats(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    rows = []
    for match in REMAP_DELTA_RE.finditer(text):
        rows.append(
            {
                "n": int(match.group("n")),
                "dM": float(match.group("dM")),
                "dM_rel": float(match.group("dM_rel")),
                "dE": float(match.group("dE")),
                "dE_rel": float(match.group("dE_rel")),
            }
        )
    return {
        "n_remaps_applied": max((row["n"] for row in rows), default=0),
        "remap_energy_delta_max_rel": max((abs(row["dE_rel"]) for row in rows), default=None),
        "remap_mass_delta_max_rel": max((abs(row["dM_rel"]) for row in rows), default=None),
        "remap_rows": rows,
    }


def has_table_boundary_warning(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    return bool(TABLE_BOUNDARY_RE.search(log_path.read_text(encoding="utf-8", errors="replace")))


def cell_center_speed(snapshot: dict[str, Any]) -> Any:
    import numpy as np

    vr = np.asarray(snapshot["v_r"], dtype=float)
    vz = np.asarray(snapshot["v_z"], dtype=float)
    vc_r = 0.25 * (vr[:-1, :-1] + vr[1:, :-1] + vr[:-1, 1:] + vr[1:, 1:])
    vc_z = 0.25 * (vz[:-1, :-1] + vz[1:, :-1] + vz[:-1, 1:] + vz[1:, 1:])
    return np.sqrt(vc_r * vc_r + vc_z * vc_z)


def total_energy(snapshot: dict[str, Any]) -> float:
    import numpy as np

    mass = np.asarray(snapshot["mass"], dtype=float)
    internal = np.sum(mass * (np.asarray(snapshot["ee"]) + np.asarray(snapshot["ei"])))
    kinetic = np.sum(0.5 * mass * cell_center_speed(snapshot) ** 2)
    return float(internal + kinetic)


def laplacian_smoothness(field: Any) -> float:
    import numpy as np

    arr = np.asarray(field, dtype=float)
    if min(arr.shape) < 3:
        return 0.0
    interior = arr[1:-1, 1:-1]
    lap = arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:] - 4.0 * interior
    denom = max(abs(float(np.nanmean(interior))), 1.0e-300)
    return float(np.nanmax(np.abs(lap)) / denom)


def max_state_delta(snapshots: list[dict[str, Any]], mask: Any | None) -> float:
    import numpy as np

    fields = ("rho", "Te", "Ti", "ee", "ei", "Pe", "Pi", "zbar")
    first = snapshots[0]
    max_delta = 0.0
    for snap in snapshots[1:]:
        for field in fields:
            delta = np.abs(np.asarray(snap[field]) - np.asarray(first[field]))
            if mask is not None:
                delta = delta[mask]
            if delta.size:
                max_delta = max(max_delta, float(np.nanmax(delta)))
    return max_delta


def cold_corner_mask(snapshot: dict[str, Any]) -> Any:
    import numpy as np

    rho = np.asarray(snapshot["rho"], dtype=float)
    te = np.asarray(snapshot["Te"], dtype=float)
    ti = np.asarray(snapshot["Ti"], dtype=float)
    return (rho <= COLD_RHO_GCC) & ((te <= COLD_T_EV) | (ti <= COLD_T_EV))


def entropy_proxy(snapshot: dict[str, Any]) -> float:
    import numpy as np

    rho = np.maximum(np.asarray(snapshot["rho"], dtype=float), 1.0e-300)
    p = np.maximum(np.asarray(snapshot["Pe"], dtype=float) + np.asarray(snapshot["Pi"], dtype=float), 1.0e-300)
    mass = np.asarray(snapshot["mass"], dtype=float)
    return float(np.sum(mass * np.log(p / rho**GAMMA)))


def common_metrics(snapshots: list[dict[str, Any]], log_path: Path, wall_time_s: float, bytes_out: int) -> dict[str, Any]:
    import numpy as np

    first = snapshots[0]
    final = snapshots[-1]
    e0 = total_energy(first)
    e1 = total_energy(final)
    p_final = np.asarray(final["Pe"]) + np.asarray(final["Pi"])
    t_final = 0.5 * (np.asarray(final["Te"]) + np.asarray(final["Ti"]))
    s0 = entropy_proxy(first)
    s1 = entropy_proxy(final)
    cold_counts = [int(np.count_nonzero(cold_corner_mask(snap))) for snap in snapshots]
    speed_max = max(float(np.nanmax(cell_center_speed(snap))) for snap in snapshots)
    out = {
        "rho_min": float(np.nanmin([np.nanmin(s["rho"]) for s in snapshots])),
        "rho_max": float(np.nanmax([np.nanmax(s["rho"]) for s in snapshots])),
        "Te_min_eV": float(np.nanmin([np.nanmin(s["Te"]) for s in snapshots])),
        "Te_max_eV": float(np.nanmax([np.nanmax(s["Te"]) for s in snapshots])),
        "Ti_min_eV": float(np.nanmin([np.nanmin(s["Ti"]) for s in snapshots])),
        "Ti_max_eV": float(np.nanmax([np.nanmax(s["Ti"]) for s in snapshots])),
        "T_min": float(np.nanmin([min(np.nanmin(s["Te"]), np.nanmin(s["Ti"])) for s in snapshots])),
        "T_max": float(np.nanmax([max(np.nanmax(s["Te"]), np.nanmax(s["Ti"])) for s in snapshots])),
        "P_min": float(np.nanmin(p_final)),
        "P_max": float(np.nanmax(p_final)),
        "total_E_initial": e0,
        "total_E_final": e1,
        "dE_total_rel": abs(e1 - e0) / max(abs(e0), 1.0e-300),
        "max_v_cm_s": speed_max,
        "max_v_cs_ratio": speed_max / CS_CD_10EV,
        "laplacian_p_norm": laplacian_smoothness(p_final),
        "laplacian_T_norm": laplacian_smoothness(t_final),
        "entropy_initial": s0,
        "entropy_final": s1,
        "entropy_drift_rel": abs(s1 - s0) / max(abs(s0), 1.0e-300),
        "cold_corner_any": any(count > 0 for count in cold_counts),
        "cold_corner_count_max": max(cold_counts) if cold_counts else 0,
        "hdf5_sample_count": len(snapshots),
        "n_clamp": parse_clamp_count(log_path),
        "wall_time_s": wall_time_s,
        "output_bytes": bytes_out,
    }
    out.update(parse_ale_stats(log_path))
    out["table_boundary_warning"] = has_table_boundary_warning(log_path)
    return out


def collect_metrics(
    mode: str,
    nr: int,
    nz: int,
    results_dir: Path,
    run_id: str,
    log_path: Path,
    wall_time_s: float,
    bytes_out: int,
) -> dict[str, Any]:
    import numpy as np

    snapshots = read_snapshots(results_dir, run_id, nr, nz)
    metrics = common_metrics(snapshots, log_path, wall_time_s, bytes_out)
    first = snapshots[0]

    if mode == "static_cd_target":
        if "volfrac" not in first:
            raise RuntimeError("static_cd_target requires hydro/volFrac output")
        vf = np.asarray(first["volfrac"], dtype=float)
        rho = np.asarray(first["rho"], dtype=float)
        inside = rho > 0.5
        outside = ~inside
        z_inside = np.asarray(first["zbar"])[inside]
        z_outside = np.asarray(first["zbar"])[outside]
        metrics.update(
            {
                "inside_cell_count": int(np.count_nonzero(inside)),
                "outside_cell_count": int(np.count_nonzero(outside)),
                "volfrac_inside_min": float(np.nanmin(vf[inside])) if np.any(inside) else None,
                "volfrac_inside_max": float(np.nanmax(vf[inside])) if np.any(inside) else None,
                "volfrac_outside_min": float(np.nanmin(vf[outside])) if np.any(outside) else None,
                "volfrac_outside_max": float(np.nanmax(vf[outside])) if np.any(outside) else None,
                "zbar_inside_mean": float(np.nanmean(z_inside)) if z_inside.size else None,
                "zbar_inside_min": float(np.nanmin(z_inside)) if z_inside.size else None,
                "zbar_inside_max": float(np.nanmax(z_inside)) if z_inside.size else None,
                "zbar_outside_max": float(np.nanmax(z_outside)) if z_outside.size else None,
                "max_state_delta_inside": max_state_delta(snapshots, inside),
            }
        )

    if mode == "static_cd_uniform":
        metrics["max_state_delta_all"] = max_state_delta(snapshots, None)

    if mode in ("cold_corner_sentinel", "cold_corner_guard_negative"):
        per_snapshot = [int(np.count_nonzero(cold_corner_mask(snap))) for snap in snapshots]
        n_cells = int(np.asarray(first["rho"]).size)
        metrics.update(
            {
                "cold_corner_count_series": per_snapshot,
                "cold_corner_all_cells": bool(per_snapshot and all(count == n_cells for count in per_snapshot)),
                "cell_count": n_cells,
            }
        )
    return metrics


def evaluate_gates(mode: str, metrics: dict[str, Any]) -> tuple[list[str], str | None]:
    failed = []
    guard_status: str | None = None

    if mode == "cold_corner_sentinel":
        guard_status = "expected_trip" if metrics.get("cold_corner_any", False) else "no_trip"
        if guard_status != "expected_trip":
            failed.append("e1_cold_corner_expected_trip")
        if not metrics.get("cold_corner_all_cells", False):
            failed.append("e1_cold_corner_all_cells")
    elif mode == "cold_corner_guard_negative":
        guard_status = "trip" if metrics.get("cold_corner_any", False) else "no_spurious_trip"
        if guard_status != "no_spurious_trip":
            failed.append("e1_cold_corner_no_spurious_trip")
    else:
        guard_status = "trip" if metrics.get("cold_corner_any", False) else "no_spurious_trip"
        if metrics.get("cold_corner_any", False):
            failed.append("e1_cold_corner_runtime_guard")

    if mode == "static_cd_target":
        # CD fills every cell; the target shape is represented by dense and
        # floor-density regions so the outside exercises low-density fallback.
        if metrics.get("inside_cell_count", 0) <= 0 or metrics.get("outside_cell_count", 0) <= 0:
            failed.append("e1_static_cd_target_density_partition")
        if (
            abs(float(metrics.get("volfrac_inside_min") or 0.0) - 1.0) > 1.0e-12
            or abs(float(metrics.get("volfrac_inside_max") or 0.0) - 1.0) > 1.0e-12
            or abs(float(metrics.get("volfrac_outside_min") or 0.0) - 1.0) > 1.0e-12
            or abs(float(metrics.get("volfrac_outside_max") or 0.0) - 1.0) > 1.0e-12
        ):
            failed.append("e1_static_cd_target_full_cd_volfrac")
        if metrics.get("dE_total_rel", math.inf) > STATIC_DELTA_TOL or metrics.get("max_v_cm_s", math.inf) > STATIC_DELTA_TOL:
            failed.append("e1_static_cd_target_static")
        zbar_inside = float(metrics.get("zbar_inside_mean") or math.nan)
        zbar_outside = float(metrics.get("zbar_outside_max") or math.nan)
        if not math.isfinite(zbar_inside) or not math.isfinite(zbar_outside) or zbar_inside <= zbar_outside:
            failed.append("e1_static_cd_target_tmat_fallback")
    elif mode == "static_cd_uniform":
        # Hydro is disabled for this uniform static mode; TMAT EOS reclosure
        # should be idempotent on a spatially uniform state.
        rho_spread = metrics.get("rho_max", math.inf) / max(metrics.get("rho_min", 0.0), 1.0e-300) - 1.0
        te_spread = metrics.get("Te_max_eV", math.inf) / max(metrics.get("Te_min_eV", 0.0), 1.0e-300) - 1.0
        if metrics.get("dE_total_rel", math.inf) > STATIC_UNIFORM_DRIFT_TOL:
            failed.append("e1_static_cd_uniform_energy")
        if metrics.get("entropy_drift_rel", math.inf) > STATIC_UNIFORM_ENTROPY_TOL:
            failed.append("e1_static_cd_uniform_entropy")
        if rho_spread > STATIC_UNIFORM_RHO_SPREAD_TOL:
            failed.append("e1_static_cd_uniform_rho_spread")
        if te_spread > STATIC_UNIFORM_TE_SPREAD_TOL:
            failed.append("e1_static_cd_uniform_Te_spread")
    elif mode == "isentropic":
        if metrics.get("max_v_cs_ratio", math.inf) >= 0.5:
            failed.append("e1_isentropic_subsonic")
        # AV dissipates compression energy; entropy drift remains diagnostic.
        if metrics.get("dE_total_rel", math.inf) > ISENTROPIC_ENERGY_DRIFT_TOL:
            failed.append("e1_isentropic_energy")
        if metrics.get("laplacian_p_norm", math.inf) > 1.0 or metrics.get("laplacian_T_norm", math.inf) > 1.0:
            failed.append("e1_isentropic_no_checkerboard")
    elif mode == "noh_weak":
        if metrics.get("T_max", math.inf) >= 100.0:
            failed.append("e1_noh_weak_Tmax")
        if metrics.get("rho_max", math.inf) >= 10.0:
            failed.append("e1_noh_weak_rhomax")
        if metrics.get("table_boundary_warning", False):
            failed.append("e1_noh_weak_no_table_boundary_warning")
    elif mode == "ale_remap":
        remap_rel = metrics.get("remap_energy_delta_max_rel")
        if remap_rel is None or remap_rel > 1.0e-6:
            failed.append("e1_ale_remap_energy")
        if metrics.get("laplacian_p_norm", math.inf) > 1.0:
            failed.append("e1_ale_remap_pressure_smoothness")
    return failed, guard_status


def rho_label_for_mode(mode: str, rho_gcc: float) -> float:
    if mode == "cold_corner_sentinel":
        return 1.0e-5
    if mode == "cold_corner_guard_negative":
        return 1.0
    return rho_gcc


def active_env(args: argparse.Namespace, mode: str, nr: int, nz: int, seed: int, outdir: Path) -> dict[str, str]:
    return {
        "TENRYU_E1_MODE": mode,
        "TENRYU_E1_NR": str(nr),
        "TENRYU_E1_NZ": str(nz),
        "TENRYU_E1_SEED": str(seed),
        "TENRYU_E1_OUTDIR": str(outdir),
        "TENRYU_E1_RHO_GCC": str(args.rho_gcc),
        "TENRYU_E1_MAX_STEPS": str(args.max_steps),
    }


def run_one(args: argparse.Namespace, mode: str, nr: int, nz: int, seed: int) -> dict[str, Any]:
    rho_label = rho_label_for_mode(mode, args.rho_gcc)
    run_id = case_name(mode, nr, nz, seed, rho_label)
    ladder_row = f"{mode}_nr{nr}_nz{nz}"
    outdir = Path(args.out_root) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    env_overrides = active_env(args, mode, nr, nz, seed, outdir)
    env = os.environ.copy()
    env.update(env_overrides)
    log_path = outdir / "run.log"

    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([args.tenryu_bin, "run", args.deck], env=env, stdout=log, stderr=log)
    wall_time_s = time.monotonic() - start

    metrics: dict[str, Any] = {"n_clamp": parse_clamp_count(log_path), "wall_time_s": wall_time_s, "output_bytes": output_bytes(outdir)}
    result = "PASS"
    failed_gates: list[str] = []
    failure_class: str | None = None
    guard_status: str | None = None
    if proc.returncode != 0:
        result = "CRASHED"
        failed_gates = ["run_failure"]
        failure_class = "run_failure"
    else:
        try:
            results_dir = discover_results_dir(outdir)
            effective_outdir = results_dir.parent
            bytes_out = output_bytes(effective_outdir)
            if effective_outdir != outdir:
                bytes_out += output_bytes(outdir)
            metrics = collect_metrics(mode, nr, nz, results_dir, run_id, log_path, wall_time_s, bytes_out)
            failed_gates, guard_status = evaluate_gates(mode, metrics)
            result = "PASS" if not failed_gates else "FAIL"
            failure_class = None if not failed_gates else "gate_failure"
        except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            result = "FAIL"
            failed_gates = ["analysis_failure"]
            failure_class = "analysis_failure"
            metrics["analysis_error"] = str(exc)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[run_e1] analysis failed: {exc}\n")

    rel_metrics_ref = f"{args.root.rstrip('/')}/{STAGE_ID}/metrics/{run_id}.json"
    payload = {
        "run_id": run_id,
        "stage_id": STAGE_ID,
        "ladder_row": ladder_row,
        "seed": seed,
        "deck_path": args.deck,
        "mode": mode,
        "nr": nr,
        "nz": nz,
        "result": result,
        "failure_class": failure_class,
        "failed_gates": failed_gates,
        "guard_status": guard_status,
        "exit_code": proc.returncode,
        "wall_time_s": wall_time_s,
        "metrics": metrics,
        "env_overrides": env_overrides,
    }
    write_json(Path(rel_metrics_ref), payload)

    return {
        "run_id": run_id,
        "stage_id": STAGE_ID,
        "ladder_row": ladder_row,
        "seed": seed,
        "deck_path": args.deck,
        "mode": mode,
        "nr": nr,
        "nz": nz,
        "result": result,
        "failure_class": failure_class,
        "failed_gates": failed_gates,
        "guard_status": guard_status,
        "exit_code": proc.returncode,
        "metrics_ref": rel_metrics_ref,
        "wall_time_s": wall_time_s,
        "output_bytes": metrics.get("output_bytes", output_bytes(outdir)),
        "rho_min": metrics.get("rho_min"),
        "rho_max": metrics.get("rho_max"),
        "T_min": metrics.get("T_min"),
        "T_max": metrics.get("T_max"),
        "dE_total_rel": metrics.get("dE_total_rel"),
        "max_v_cs_ratio": metrics.get("max_v_cs_ratio"),
        "n_clamp": metrics.get("n_clamp"),
    }


def region_bounds(mask: Any, rho_grid: Any, t_lo: Any, t_hi: Any) -> dict[str, float | None]:
    import numpy as np

    if not np.any(mask):
        return {"rho_min": None, "rho_max": None, "T_min": None, "T_max": None}
    rho2 = np.broadcast_to(np.asarray(rho_grid)[:, None], mask.shape)
    tlo2 = np.broadcast_to(np.asarray(t_lo)[None, :], mask.shape)
    thi2 = np.broadcast_to(np.asarray(t_hi)[None, :], mask.shape)
    return {
        "rho_min": float(np.nanmin(rho2[mask])),
        "rho_max": float(np.nanmax(rho2[mask])),
        "T_min": float(np.nanmin(tlo2[mask])),
        "T_max": float(np.nanmax(thi2[mask])),
    }


def run_c6_audit(args: argparse.Namespace) -> dict[str, Any]:
    import h5py
    import numpy as np

    table_path = Path(args.tmat_path)
    with h5py.File(table_path, "r") as handle:
        e_e = np.asarray(handle["eos/fields/e_e"][()], dtype=float)
        temp = np.asarray(handle["eos/grid/temperature_eV"][()], dtype=float)
        ni = np.asarray(handle["eos/grid/ni_cm3"][()], dtype=float)
        abar = float(np.asarray(handle["material/Abar_ion_amu"][()]))

    rho = ni * abar * M_P
    raw_cv = (e_e[:, 1:] - e_e[:, :-1]) / (temp[1:] - temp[:-1])[None, :]
    t_lo = temp[:-1]
    t_hi = temp[1:]
    validated = raw_cv > CV_FLOOR_THRESHOLD
    flat = raw_cv <= 0.0
    floor = (raw_cv > 0.0) & (raw_cv <= CV_FLOOR_THRESHOLD)
    cold = (rho[:, None] <= COLD_RHO_GCC) & (t_lo[None, :] <= COLD_T_EV)
    flat_cold = flat & cold
    floor_cold = floor & cold

    regions = {
        "validated": region_bounds(validated, rho, t_lo, t_hi),
        "flat": region_bounds(flat, rho, t_lo, t_hi),
        "floor": region_bounds(floor, rho, t_lo, t_hi),
        "cold_flat": region_bounds(flat_cold, rho, t_lo, t_hi),
        "cold_floor": region_bounds(floor_cold, rho, t_lo, t_hi),
    }
    summary = {
        "stage_id": STAGE_ID,
        "table_path": str(table_path),
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "raw_cv_threshold_erg_g_ev": CV_FLOOR_THRESHOLD,
        "n_rho": int(rho.size),
        "n_T": int(temp.size),
        "n_cv_cells": int(raw_cv.size),
        "n_validated": int(np.count_nonzero(validated)),
        "n_flat": int(np.count_nonzero(flat)),
        "n_floor": int(np.count_nonzero(floor)),
        "n_cold_flat": int(np.count_nonzero(flat_cold)),
        "n_cold_floor": int(np.count_nonzero(floor_cold)),
        "n_cold_overlap": int(np.count_nonzero(flat_cold) + np.count_nonzero(floor_cold)),
        "regions": regions,
    }
    write_json(Path(args.c6_json), summary)
    write_c6_markdown(Path(args.c6_md), summary)
    return summary


def write_c6_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("VALIDATED raw_cv > 1e-3", "n_validated", "validated"),
        ("flat/inverted raw_cv <= 0", "n_flat", "flat"),
        ("below floor 0 < raw_cv <= 1e-3", "n_floor", "floor"),
    ]
    lines = [
        "# E1 C6 Offline CD TMAT Table Audit",
        "",
        f"- Table: `{summary['table_path']}`",
        f"- Generated: `{summary['generated_iso']}`",
        f"- Forward-difference cells: `{summary['n_cv_cells']}`",
        f"- Cold corner: rho <= {COLD_RHO_GCC:.6g} g/cm3 and T <= {COLD_T_EV:.6g} eV",
        "",
        "| region | count | rho_min | rho_max | T_min | T_max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, count_key, region_key in rows:
        bounds = summary["regions"][region_key]
        fmt = lambda value: "none" if value is None else f"{float(value):.6g}"
        lines.append(
            "| {label} | {count} | {rho_min} | {rho_max} | {t_min} | {t_max} |".format(
                label=label,
                count=summary[count_key],
                rho_min=fmt(bounds["rho_min"]),
                rho_max=fmt(bounds["rho_max"]),
                t_min=fmt(bounds["T_min"]),
                t_max=fmt(bounds["T_max"]),
            )
        )
    lines.append("")
    lines.append(f"Cold corner overlap: flat={summary['n_cold_flat']}, floor={summary['n_cold_floor']}, total={summary['n_cold_overlap']}.")
    lines.extend(
        [
            "",
            "The validated region is the part of the raw table whose forward-difference electron "
            "specific heat is above the artificial runtime floor threshold. Flat, inverted, and "
            "below-floor regions are excluded from E1 production trajectories by the runtime "
            "cold-corner guard.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def matrix_rows(args: argparse.Namespace) -> list[tuple[str, int, int, int]]:
    meshes = list(args.mesh_list or (SMOKE_MESHES if args.smoke else FULL_MESHES))
    rows = []
    for mode in args.mode_list:
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


def write_summary(args: argparse.Namespace, rows: list[dict[str, Any]], audit: dict[str, Any] | None) -> None:
    aggregate = {
        "total_runs": len(rows),
        "passing_runs": sum(1 for row in rows if row["result"] == "PASS"),
        "crashed_runs": sum(1 for row in rows if row["result"] == "CRASHED"),
        "failed_runs": sum(1 for row in rows if row["result"] == "FAIL"),
    }
    for key in ("wall_time_s", "dE_total_rel", "max_v_cs_ratio", "n_clamp", "T_max", "rho_max"):
        values = finite_values(rows, key)
        aggregate[f"{key}_max"] = max(values) if values else None
    payload = {
        "stage_id": STAGE_ID,
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "matrix": rows,
        "aggregate": aggregate,
        "c6_audit_ref": args.c6_json,
        "c6_audit": audit,
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
    print("Stage E1 TMAT dry-run sweep summary")
    print("ladder_row seed result guard_status dE_total_rel max_v_cs_ratio n_clamp wall_time_s failed_gates")
    for row in rows:
        print(
            "{ladder_row} {seed} {result} {guard_status} {dE_total_rel} {max_v_cs_ratio} {n_clamp} "
            "{wall_time_s:.3f} {failed_gates}".format(**row)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/2d_rz_e1_tmat_dry.py")
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    parser.add_argument("--out-root", default="./build/e1_runs")
    parser.add_argument("--modes", "--mode-list", dest="mode_list", type=lambda v: parse_csv(v, MODES, "mode"), default=list(MODES))
    parser.add_argument("--mesh-list", type=parse_mesh_list, default=None)
    parser.add_argument("--seed-list", type=parse_seed_list, default=parse_seed_list("12345"))
    matrix = parser.add_mutually_exclusive_group()
    matrix.add_argument("--smoke", action="store_true", help="use 64x128 grid only")
    parser.add_argument("--rho-gcc", type=float, default=1.05)
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument("--summary-json", default="docs/validation/2d_rz/E1/stats.json")
    parser.add_argument("--runs-jsonl", default="docs/validation/2d_rz/E1/runs.jsonl")
    parser.add_argument("--tmat-path", default="TMAT-H5/CD.tmat.h5")
    parser.add_argument("--c6-md", default="docs/validation/2d_rz/E1/audit/c6_offline_table_audit.md")
    parser.add_argument("--c6-json", default="docs/validation/2d_rz/E1/metrics/c6_table_audit.json")
    parser.add_argument("--audit-c6-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.audit_c6_only:
        summary = run_c6_audit(args)
        print(
            "C6 audit: n_validated={n_validated} n_flat={n_flat} n_floor={n_floor} "
            "cold_overlap={n_cold_overlap}".format(**summary)
        )
        return 0

    rows_spec = matrix_rows(args)
    audit = run_c6_audit(args)
    rows = [run_one(args, mode, nr, nz, seed) for mode, nr, nz, seed in rows_spec]
    print_summary(rows)
    write_summary(args, rows, audit)
    append_jsonl(Path(args.runs_jsonl), rows)
    update_stage_json(args, rows)
    return 0 if all(row["result"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
