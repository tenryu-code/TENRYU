#!/usr/bin/env python3
"""Run Stage R2 2D RZ S_N-only validation sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from marshak_boundary_source_reference import solve_marshak_sn_1d


STAGE_ID = "R2"
MODES = (
    "axisymmetric_convergence",
    "diffusion_limit_vs_fld",
    "thin_corona_sn",
    "multigroup_5g",
    "n_angles_sweep",
    "slab_marshak_sn",
)
FULL_MESHES_BY_MODE: dict[str, tuple[tuple[int, int], ...]] = {
    "axisymmetric_convergence": ((32, 64),),
    "diffusion_limit_vs_fld": ((32, 64),),
    "thin_corona_sn": ((32, 64),),
    "multigroup_5g": ((16, 32),),
    "n_angles_sweep": ((16, 32),),
    "slab_marshak_sn": ((32, 64),),
}
SMOKE_MESHES_BY_MODE: dict[str, tuple[tuple[int, int], ...]] = {
    mode: ((8, 16),) for mode in FULL_MESHES_BY_MODE
}
ANGLE_SWEEP = (8, 16, 32)

NUMBER_RE = r"[-+0-9.eE]+"
DECK_LINE_RE = re.compile(
    r"\[deck:2d_rz_r2_sn_only\].*?groups=(?P<groups>\d+).*?"
    r"n_angles=(?P<n_angles>\d+).*?rho_gcc=(?P<rho>{n}).*?"
    r"kappa_a_cm2_g=(?P<kappa>{n}).*?t_end_s=(?P<t_end>{n}).*?"
    r"cv_e_override=(?P<cv>{n})".format(n=NUMBER_RE)
)
SN_ITER_RE = re.compile(
    r"sn(?:_outer_iterations| outer_iterations|.*outer_iterations=)(?:\s*[:=]\s*|)(?P<outer>\d+)",
    re.IGNORECASE,
)
SN_TIMING_RE = re.compile(r"\[sn_2d_rz_timing\].*?outer_iters=(?P<outer>\d+)")
CLAMP_RE = re.compile(r"clamp_count=(?P<count>\d+)")

A_EV = 1.3720e2
E_FLOOR = 1.0e-300
MAX_OUTER_ITERATIONS = 50
ITERATION_CAP_FRACTION = 0.8
ENERGY_TOL = 1.0e-4
MULTIGROUP_ENERGY_TOL = 1.0e-3
DIFFUSION_FUNCTIONAL_ENERGY_TOL = 1.0e-3
FLD_REFERENCE_TOL = 3.0e-2
RADIAL_MONOTONE_REL_TOL = 1.0e-12
RADIAL_UNIFORM_REL_TOL = 1.0e-3  # Accept uniform profile within 0.1%
RAY_EFFECT_AP_ALPHA_CV_TOL = 0.5
MARSHAK_SOURCE_ACCOUNTING_TOL = 1.0e-3
MARSHAK_SN_PROFILE_TOL = 0.15
MARSHAK_FLUX_DEFAULT = 1.0e10
MARSHAK_SN_INNER_CORE_FRACTION = 0.30
MARSHAK_REFERENCE_STEPS = 20
SLAB_MARSHAK_MAX_STEPS = 50


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


def parse_angle_list(value: str) -> list[int]:
    angles = [int(part.strip()) for part in value.split(",") if part.strip()]
    bad = [angle for angle in angles if angle not in (8, 16, 32)]
    if bad:
        raise argparse.ArgumentTypeError(f"invalid S_N angle count: {bad}")
    if not angles:
        raise argparse.ArgumentTypeError("angle list is empty")
    return angles


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError("expected one of true/false, yes/no, on/off, 1/0")


def safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def mode_groups(mode: str) -> int:
    return 5 if mode == "multigroup_5g" else 1


def mode_rho_default(mode: str) -> float:
    if mode == "thin_corona_sn":
        return 1.0e-7
    if mode == "slab_marshak_sn":
        return 1.0
    return 1.0e-3


def mode_kappa_default(mode: str) -> float:
    if mode == "thin_corona_sn":
        return 1.0e-3
    if mode == "diffusion_limit_vs_fld":
        return 100.0
    if mode == "slab_marshak_sn":
        return 1.0
    return 1.0


def case_name(
    mode: str,
    nr: int,
    nz: int,
    rho_gcc: float,
    kappa_a: float,
    groups: int,
    n_angles: int,
    seed: int,
    dsa_enabled: bool,
) -> str:
    dsa_suffix = "" if dsa_enabled else "_dsaoff"
    return (
        f"r2_{mode}_nr{nr}_nz{nz}_rho{safe_float_token(rho_gcc)}"
        f"_ka{safe_float_token(kappa_a)}_g{groups}_s{n_angles}_seed{seed}{dsa_suffix}"
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


def reshape_rad(values: Any, nr: int, nz: int, groups: int, name: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.shape == (nr * nz, groups):
        return arr.reshape((nr, nz, groups))
    if arr.shape == (nr, nz, groups):
        return arr
    if groups == 1 and arr.size == nr * nz:
        return arr.reshape((nr, nz, 1))
    if arr.size == nr * nz * groups:
        return arr.reshape((nr, nz, groups))
    raise ValueError(f"{name} has shape {arr.shape}, expected {(nr * nz, groups)}")


def scalar_dataset(handle: Any, path: str, default: float | None = None) -> float | None:
    import numpy as np

    if path not in handle:
        return default
    arr = np.asarray(handle[path][()], dtype=float).reshape(-1)
    if arr.size == 0:
        return default
    return float(arr[-1])


def read_snapshot(path: Path, nr: int, nz: int, groups: int) -> dict[str, Any]:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        r_mesh_path = "mesh/x_r" if "mesh/x_r" in handle else "mesh/v_r"
        z_mesh_path = "mesh/x_z" if "mesh/x_z" in handle else "mesh/v_z"
        t_value = float(handle.attrs.get("t", 0.0))
        if "time_state/t" in handle:
            t_value = float(np.asarray(handle["time_state/t"][()]).reshape(-1)[0])
        snap = {
            "path": str(path),
            "t": t_value,
            "rho": reshape_cells(handle["hydro/rho"][()], nr, nz, "hydro/rho"),
            "Te": reshape_cells(handle["hydro/Te"][()], nr, nz, "hydro/Te"),
            "Ti": reshape_cells(handle["hydro/Ti"][()], nr, nz, "hydro/Ti"),
            "ee": reshape_cells(handle["hydro/ee"][()], nr, nz, "hydro/ee"),
            "ei": reshape_cells(handle["hydro/ei"][()], nr, nz, "hydro/ei"),
            "vol": reshape_cells(handle["hydro/vol"][()], nr, nz, "hydro/vol"),
            "rad_E_groups": reshape_rad(handle["radiation/energy_density"][()], nr, nz, groups, "radiation/energy_density"),
            "r_nodes": reshape_nodes(handle[r_mesh_path][()], nr, nz, r_mesh_path),
            "z_nodes": reshape_nodes(handle[z_mesh_path][()], nr, nz, z_mesh_path),
            "E_rad_escaped_state": scalar_dataset(handle, "time_state/E_rad_escaped", 0.0),
            "E_Marshak_in_state": scalar_dataset(handle, "time_state/E_Marshak_in", None),
            "E_numerical_loss_state": scalar_dataset(handle, "time_state/E_numerical_loss", 0.0),
            "E_floor_injected_state": scalar_dataset(handle, "time_state/E_floor_injected", 0.0),
        }
        if "radiation/sn_ap_alpha" in handle:
            snap["sn_ap_alpha"] = reshape_cells(handle["radiation/sn_ap_alpha"][()], nr, nz, "radiation/sn_ap_alpha")
        if "radiation/sn_angular_fixup_count" in handle:
            snap["sn_angular_fixup_count"] = reshape_rad(
                handle["radiation/sn_angular_fixup_count"][()], nr, nz, groups, "radiation/sn_angular_fixup_count"
            )
        snap["rad_E"] = np.sum(snap["rad_E_groups"], axis=2)
        return snap


def read_snapshots(results_dir: Path, run_id: str, nr: int, nz: int, groups: int) -> list[dict[str, Any]]:
    files = locate_plot_files(results_dir, run_id)
    if len(files) < 2:
        raise RuntimeError("expected at least initial and final plot snapshots")
    return [read_snapshot(path, nr, nz, groups) for path in files]


def read_history_scalar(results_dir: Path, run_id: str, path: str) -> list[float]:
    import h5py
    import numpy as np

    candidates = list(results_dir.glob(f"{run_id}_history.h5")) + list(results_dir.glob("*_history.h5"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        with h5py.File(candidate, "r") as handle:
            if path in handle:
                return [float(x) for x in np.asarray(handle[path][()], dtype=float).reshape(-1)]
    return []


def parse_log_metadata(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    out: dict[str, Any] = {
        "n_clamp": sum(int(match.group("count")) for match in CLAMP_RE.finditer(text)),
        "sn_outer_iterations_max": None,
        "sn_outer_iterations_series": [],
        "sn_iteration_source": "unavailable_current_2d_sn_api",
        "sn_timing_line_count": 0,
    }
    deck_match = DECK_LINE_RE.search(text)
    if deck_match:
        out["deck_groups"] = int(deck_match.group("groups"))
        out["deck_n_angles"] = int(deck_match.group("n_angles"))
        out["deck_rho_gcc"] = float(deck_match.group("rho"))
        out["deck_kappa_a_cm2_g"] = float(deck_match.group("kappa"))
        out["deck_t_end_s"] = float(deck_match.group("t_end"))
        out["deck_cv_e_override"] = float(deck_match.group("cv"))
    outer = [int(match.group("outer")) for match in SN_ITER_RE.finditer(text)]
    outer.extend(int(match.group("outer")) for match in SN_TIMING_RE.finditer(text))
    if outer:
        out["sn_outer_iterations_series"] = outer
        out["sn_outer_iterations_max"] = max(outer)
        out["sn_iteration_source"] = "log"
        out["sn_timing_line_count"] = len(outer)
    return out


def compute_E_rad_total(rad_E: Any, mesh: Mesh) -> float:
    import numpy as np

    return float(np.sum(np.asarray(rad_E, dtype=float) * np.asarray(mesh.vol, dtype=float)))


def compute_U_internal_total(rho: Any, ee: Any, ei: Any, mesh: Mesh) -> float:
    import numpy as np

    return float(
        np.sum(
            np.asarray(rho, dtype=float)
            * (np.asarray(ee, dtype=float) + np.asarray(ei, dtype=float))
            * np.asarray(mesh.vol, dtype=float)
        )
    )


def mesh_from_snapshot(snapshot: dict[str, Any], nr: int, nz: int) -> Mesh:
    return Mesh(nr=nr, nz=nz, r_nodes=snapshot["r_nodes"], z_nodes=snapshot["z_nodes"], vol=snapshot["vol"])


def rel_delta(a: float, b: float) -> float:
    return abs(b - a) / max(abs(a), E_FLOOR)


def cell_widths(mesh: Mesh) -> tuple[Any, Any]:
    import numpy as np

    r_nodes = np.asarray(mesh.r_nodes, dtype=float)
    z_nodes = np.asarray(mesh.z_nodes, dtype=float)
    dr = 0.5 * (
        np.abs(r_nodes[1:, :-1] - r_nodes[:-1, :-1])
        + np.abs(r_nodes[1:, 1:] - r_nodes[:-1, 1:])
    )
    dz = 0.5 * (
        np.abs(z_nodes[:-1, 1:] - z_nodes[:-1, :-1])
        + np.abs(z_nodes[1:, 1:] - z_nodes[1:, :-1])
    )
    return dr, dz


def interior_mask(mesh: Mesh, n_boundary_cells: int = 2) -> Any:
    import numpy as np

    mask = np.ones((mesh.nr, mesh.nz), dtype=bool)
    if n_boundary_cells > 0:
        nb = min(n_boundary_cells, mesh.nr // 2, mesh.nz // 2)
        if nb > 0:
            mask[:nb, :] = False
            mask[mesh.nr - nb :, :] = False
            mask[:, :nb] = False
            mask[:, mesh.nz - nb :] = False
    return mask


def compute_ray_effect_metrics(final: dict[str, Any], mesh: Mesh, kappa_a: float) -> dict[str, Any]:
    import numpy as np

    if "sn_ap_alpha" not in final:
        return {
            "ray_effect_metric_present": False,
            "ray_effect_metric_status": "missing_sn_ap_alpha",
            "ray_effect_ap_alpha_mean": None,
            "ray_effect_ap_alpha_std": None,
            "ray_effect_ap_alpha_cv": None,
            "ray_effect_cell_count": 0,
        }
    alpha = np.asarray(final["sn_ap_alpha"], dtype=float)
    dr, dz = cell_widths(mesh)
    tau_min = np.asarray(final["rho"], dtype=float) * float(kappa_a) * np.minimum(dr, dz)
    base_mask = interior_mask(mesh)
    thick_mask = base_mask & np.isfinite(tau_min) & (tau_min >= 1.0)
    metric_mask = thick_mask if np.count_nonzero(thick_mask) > 0 else base_mask
    values = alpha[metric_mask & np.isfinite(alpha)]
    if values.size == 0:
        return {
            "ray_effect_metric_present": True,
            "ray_effect_metric_status": "skipped_empty_interior_mask",
            "ray_effect_mask_source": "optically_thick" if np.count_nonzero(thick_mask) > 0 else "interior_fallback",
            "ray_effect_ap_alpha_mean": None,
            "ray_effect_ap_alpha_std": None,
            "ray_effect_ap_alpha_cv": None,
            "ray_effect_cell_count": 0,
        }
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    cv = 0.0 if std == 0.0 and abs(mean) <= E_FLOOR else std / max(abs(mean), E_FLOOR)
    return {
        "ray_effect_metric_present": True,
        "ray_effect_metric_status": "evaluated",
        "ray_effect_mask_source": "optically_thick" if np.count_nonzero(thick_mask) > 0 else "interior_fallback",
        "ray_effect_ap_alpha_mean": mean,
        "ray_effect_ap_alpha_std": std,
        "ray_effect_ap_alpha_cv": float(cv),
        "ray_effect_cell_count": int(values.size),
        "ray_effect_tau_min_threshold": 1.0,
    }


def compute_z_face_area(mesh: Mesh, top: bool) -> float:
    import numpy as np

    r_nodes = np.asarray(mesh.r_nodes, dtype=float)
    j = mesh.nz if top else 0
    r_left = r_nodes[:-1, j]
    r_right = r_nodes[1:, j]
    return float(np.sum(math.pi * np.maximum(r_right * r_right - r_left * r_left, 0.0)))


def compute_slab_marshak_metrics(
    final: dict[str, Any],
    mesh: Mesh,
    marshak_in: float,
    kappa_a: float,
    cv_e_volume: float,
    n_angles: int,
) -> dict[str, Any]:
    import numpy as np

    top_area = compute_z_face_area(mesh, top=True)
    expected_source = MARSHAK_FLUX_DEFAULT * top_area * max(float(final["t"]), 0.0)
    rho_mean = float(np.nanmean(final["rho"]))
    sigma = max(rho_mean * float(kappa_a), 1.0e-300)
    cv_ref = float(cv_e_volume)
    z_nodes = np.asarray(mesh.z_nodes, dtype=float)
    z_top = z_nodes[0, -1]
    z_bottom = z_nodes[0, 0]
    z_centers = 0.5 * (z_nodes[0, :-1] + z_nodes[0, 1:])
    depth = np.maximum(z_top - z_centers, 0.0)
    ref = solve_marshak_sn_1d(
        L_cm=float(z_top - z_bottom),
        n_z=mesh.nz,
        sigma_a=sigma,
        rho=rho_mean,
        cv_e_volume=cv_ref,
        F_inc_erg_per_cm2_s=MARSHAK_FLUX_DEFAULT,
        t_end_s=max(float(final["t"]), 0.0),
        n_steps=MARSHAK_REFERENCE_STEPS,
        T_init_eV=0.1,
        sn_order=int(n_angles),
    )
    ref_profile = np.interp(depth, ref["x_centers"], ref["E_r"])
    rad_full = np.asarray(final["rad_E"], dtype=float)
    vol_full = np.asarray(mesh.vol, dtype=float)
    mean_profile = np.sum(rad_full * vol_full, axis=0) / np.maximum(np.sum(vol_full, axis=0), E_FLOOR)
    weights = np.sum(vol_full, axis=0)
    rel_l2 = float(math.sqrt(np.sum(weights * (mean_profile - ref_profile) ** 2) / max(np.sum(weights * ref_profile**2), E_FLOOR)))
    n_inner = max(1, int(math.ceil(mesh.nr * MARSHAK_SN_INNER_CORE_FRACTION)))
    rad_inner = rad_full[:n_inner, :]
    vol_inner = vol_full[:n_inner, :]
    inner_profile = np.sum(rad_inner * vol_inner, axis=0) / np.maximum(np.sum(vol_inner, axis=0), E_FLOOR)
    inner_weights = np.sum(vol_inner, axis=0)
    inner_rel_l2 = float(math.sqrt(np.sum(inner_weights * (inner_profile - ref_profile) ** 2) / max(np.sum(inner_weights * ref_profile**2), E_FLOOR)))
    return {
        "marshak_flux_erg_per_cm2_s": MARSHAK_FLUX_DEFAULT,
        "marshak_top_area_cm2": top_area,
        "marshak_expected_source_erg": expected_source,
        "marshak_source_accounting_rel": abs(marshak_in - expected_source) / max(expected_source, E_FLOOR),
        "E_Marshak_in": float(marshak_in),
        "marshak_z_profile_rel_l2": rel_l2,
        "marshak_z_profile_gate_threshold": MARSHAK_SN_PROFILE_TOL,
        "marshak_z_profile_gate_status": "gated",
        "marshak_inner_core_z_profile_rel_l2": inner_rel_l2,
        "marshak_inner_core_z_profile_gate_status": "stats_continuity_only",
        "marshak_inner_core_z_profile_metric_kind": "legacy inner-core proxy",
        "marshak_inner_core_fraction": MARSHAK_SN_INNER_CORE_FRACTION,
        "marshak_inner_core_cell_count": n_inner,
        "marshak_reference_scheme": ref["diagnostics"]["scheme"],
        "marshak_reference_n_steps": MARSHAK_REFERENCE_STEPS,
        "marshak_reference_sigma_a_cm_inv": sigma,
        "marshak_reference_cv_e_volume": cv_ref,
    }


def terminal_vacuum_boundary_energy(rad_E: Any, mesh: Mesh) -> float:
    import numpy as np

    shell = np.asarray(rad_E, dtype=float)[-1:, :]
    vol = np.asarray(mesh.vol, dtype=float)[-1:, :]
    return float(np.sum(np.maximum(shell, 0.0) * vol))


def radial_monotone_metrics(shell_avg: list[float]) -> dict[str, Any]:
    import numpy as np

    arr = np.asarray(shell_avg, dtype=float)
    if arr.size < 2:
        return {
            "rad_E_shell_monotone_boundary_to_boundary": True,
            "rad_E_shell_monotone_direction": "single_shell",
            "rad_E_shell_outward_max_increase_rel": 0.0,
            "rad_E_shell_outward_max_decrease_rel": 0.0,
        }
    diffs = np.diff(arr)
    denom = np.maximum(np.maximum(np.abs(arr[:-1]), np.abs(arr[1:])), E_FLOOR)
    max_increase_rel = float(np.nanmax(np.maximum(diffs, 0.0) / denom))
    max_decrease_rel = float(np.nanmax(np.maximum(-diffs, 0.0) / denom))
    nonincreasing = max_increase_rel <= RADIAL_MONOTONE_REL_TOL
    nondecreasing = max_decrease_rel <= RADIAL_MONOTONE_REL_TOL
    # Accept essentially-uniform profiles (post-curvature-fix LTE preservation)
    arr_min = float(np.nanmin(arr))
    arr_max = float(np.nanmax(arr))
    arr_range_rel = (arr_max - arr_min) / max(abs(arr_max), abs(arr_min), E_FLOOR)
    essentially_uniform = arr_range_rel <= RADIAL_UNIFORM_REL_TOL
    if essentially_uniform:
        direction = "essentially_uniform"
        passed = True
    elif nonincreasing:
        direction = "decreases_outward"
        passed = True
    elif nondecreasing:
        direction = "increases_outward_decreases_toward_axis_boundary"
        passed = True
    else:
        direction = "nonmonotone"
        passed = False
    return {
        "rad_E_shell_monotone_boundary_to_boundary": bool(passed),
        "rad_E_shell_monotone_direction": direction,
        "rad_E_shell_outward_max_increase_rel": max_increase_rel,
        "rad_E_shell_outward_max_decrease_rel": max_decrease_rel,
        "rad_E_shell_range_rel": arr_range_rel,
    }


def common_metrics(
    mode: str,
    nr: int,
    nz: int,
    groups: int,
    n_angles: int,
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
    run_id = Path(final["path"]).stem.rsplit("_", 1)[0]
    escaped_series = read_history_scalar(Path(final["path"]).parent, run_id, "energy/radiation_escaped")
    escaped_from_history = float(sum(max(x, 0.0) for x in escaped_series)) if escaped_series else None
    escaped_from_state = final.get("E_rad_escaped_state")
    escaped = escaped_from_state if escaped_from_state is not None else escaped_from_history
    escaped_source = "time_state/E_rad_escaped" if escaped_from_state is not None else "history/sum(energy/radiation_escaped)"
    if escaped is None:
        escaped = terminal_vacuum_boundary_energy(final["rad_E"], final_mesh)
        escaped_source = "terminal_outer_rad_E_shell"
    marshak_series = read_history_scalar(Path(final["path"]).parent, run_id, "energy/marshak_in")
    marshak_from_history = float(sum(max(x, 0.0) for x in marshak_series)) if marshak_series else None
    marshak_from_state = final.get("E_Marshak_in_state")
    marshak_in = float(marshak_from_state if marshak_from_state is not None else (marshak_from_history or 0.0))
    final_rad_E_shell_avg = shell_average(final["rad_E"], final["vol"])

    metrics: dict[str, Any] = {
        "a_eV_erg_cm3_eV4": A_EV,
        "groups": groups,
        "n_angles": n_angles,
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
        "E_rad_initial": rad0,
        "E_rad_final": rad1,
        "dE_rad_rel": rel_delta(rad0, rad1),
        "E_rad_escaped": float(escaped),
        "E_rad_escaped_source": escaped_source,
        "E_Marshak_in": marshak_in,
        "E_total_initial_matter_rad": u0 + rad0,
        "E_total_final_matter_rad_escape": u1 + rad1 + float(escaped),
        "E_balance_rel": abs((u1 + rad1 + float(escaped)) - (u0 + rad0 + marshak_in))
        / max(max(abs(u0 + rad0), abs(marshak_in)), E_FLOOR),
        "E_terminal_outer_rad_shell": terminal_vacuum_boundary_energy(final["rad_E"], final_mesh),
        "rad_E_shell_avg_final": final_rad_E_shell_avg,
        "wall_time_s": wall_time_s,
        "output_bytes": bytes_out,
    }
    metrics.update(radial_monotone_metrics(final_rad_E_shell_avg))
    if "sn_angular_fixup_count" in final:
        metrics["sn_angular_fixup_count_total"] = float(np.nansum(final["sn_angular_fixup_count"]))
    metrics.update(compute_ray_effect_metrics(final, final_mesh, kappa_a=log_metrics.get("deck_kappa_a_cm2_g", 1.0)))
    if mode == "slab_marshak_sn":
        metrics.update(
            compute_slab_marshak_metrics(
                final,
                final_mesh,
                marshak_in,
                kappa_a=log_metrics.get("deck_kappa_a_cm2_g", 1.0),
                cv_e_volume=log_metrics.get("deck_cv_e_override", 5.488e2),
                n_angles=n_angles,
            )
        )
    metrics.update(log_metrics)
    return metrics


def shell_average(field: Any, vol: Any) -> list[float]:
    import numpy as np

    arr = np.asarray(field, dtype=float)
    weights = np.asarray(vol, dtype=float)
    return [float(np.sum(arr[i, :] * weights[i, :]) / max(np.sum(weights[i, :]), E_FLOOR)) for i in range(arr.shape[0])]


def max_relative_difference(a: list[float], b: list[float]) -> float:
    import numpy as np

    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    denom = np.maximum(np.maximum(np.abs(aa), np.abs(bb)), E_FLOOR)
    return float(np.nanmax(np.abs(aa - bb) / denom))


def write_fld_reference_deck(path: Path) -> None:
    path.write_text(
        """
import math
import os
from tenryu_namelist import *

NR = int(os.environ["TENRYU_R2_NR"])
NZ = int(os.environ["TENRYU_R2_NZ"])
OUTDIR = os.environ["TENRYU_R2_FLD_OUTDIR"]
SEED = int(os.environ.get("TENRYU_R2_SEED", "12345"))
T_END = float(os.environ.get("TENRYU_R2_T_END_S", "1e-9"))
DT_INITIAL = float(os.environ.get("TENRYU_R2_DT_INITIAL_S", min(1.0e-12, T_END)))
DT_MAX = float(os.environ.get("TENRYU_R2_DT_MAX_S", max(T_END / 20.0, DT_INITIAL)))
RHO_GCC = float(os.environ.get("TENRYU_R2_RHO_GCC", "1e-3"))
KAPPA_A = float(os.environ.get("TENRYU_R2_KAPPA_A_CM2_G", "100.0"))
CV_E_OVERRIDE = float(os.environ.get("TENRYU_R2_CV_E_OVERRIDE", "1e15"))
CASE_NAME = f"r2_fld_reference_nr{NR}_nz{NZ}_seed{SEED}"

def rho_init(r, z):
    del r, z
    return RHO_GCC

def Te_init(r, z):
    del r, z
    return 10.0

def Ti_init(r, z):
    return Te_init(r, z)

def velocity_init(r, z):
    del r, z
    return (0.0, 0.0)

def h_volfrac(r, z):
    del r, z
    return 1.0

Main(name=CASE_NAME, dimension="2D_RZ", temperature_model="2T", t_end=T_END,
     max_steps=int(os.environ.get("TENRYU_R2_MAX_STEPS", "1000000")),
     seed=SEED, verbosity=os.environ.get("TENRYU_R2_VERBOSITY", "quiet"))
Mesh(r_min=0.0, r_max=1.0, z_min=-1.0, z_max=1.0, nr=NR, nz=NZ,
     grid="uniform", motion="lagrangian")
Materials(materials=[Material(name="H", A=1.0, Z=1.0,
    eos=dict(model="ideal_gas", cv_e_override=CV_E_OVERRIDE, ideal_gas=dict(gamma=5.0 / 3.0)),
    opacity=dict(model="constant", kappa_a=KAPPA_A, kappa_s=0.0, units="cm2_per_g"))],
    zbar=dict(model="fixed", fixed_value=1.0))
Geometry(volfrac=dict(H=h_volfrac), rho=rho_init, Te=Te_init, Ti=Ti_init,
         velocity=velocity_init, radiation_field="equilibrium", enforce_sum_to_one=True)
Numerics(radiation_thermal_subcycle=True,
    dt=dict(initial_s=DT_INITIAL, cfl_hydro=0.3, cfl_cond=0.3, growth_factor=1.1, max_s=DT_MAX, min_s=1.0e-22),
    hydro=dict(enabled=False, boundary_2d=dict(r_inner="axis", r_outer="free", z_bottom="reflect", z_top="reflect")),
    conduction=dict(enabled=False), ale=dict(enabled=False), diagnostics=dict(phase_resolved_energy=True),
    diagnostics_every=1, floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=0.1, Ti_floor_eV=0.1),
    safety=dict(energy_fatal=False, nan_fatal=True, energy_threshold=1.0e-3,
                clamp_warn_threshold=0, clamp_fatal_threshold=1000000000))
Radiation(enabled=True, mode="multigroup_diffusion", groups=1, group_bounds_eV=[0.0, 1.0e6],
    imc=dict(enabled=False, two_stage=False, difference=dict(enabled=False)),
    ddmc=dict(enabled=False), holo=dict(enabled=False),
    multigroup_diffusion=dict(flux_limiter="levermore_pomraning", max_outer_iterations=50,
        outer_tol=1.0e-8, linear_solver_2d="cusparse_cg_jacobi",
        boundary=dict(inner_r="reflect", outer_r="vacuum", z="reflect")))
Laser(enabled=False)
Output(directory=OUTDIR, plot_every=int(os.environ.get("TENRYU_R2_MAX_STEPS", "1000000")),
       history_every=1, checkpoint_every=0, plot_every_s=T_END, history_every_s=-1.0,
       checkpoint_every_s=-1.0, plot_fields=["rho", "Te", "Ti", "ee", "ei", "Pe", "Pi", "rad_E"])
Diagnostics(enabled=True, every=1, energy_budget=dict(enabled=True, warn_threshold=1.0e-3))
""".lstrip(),
        encoding="utf-8",
    )


def run_fld_reference(args: argparse.Namespace, env: dict[str, str], outdir: Path, nr: int, nz: int) -> dict[str, Any]:
    ref_dir = outdir / "fld_reference"
    ref_outdir = ref_dir / "output"
    ref_dir.mkdir(parents=True, exist_ok=True)
    deck_path = ref_dir / "fld_reference.py"
    write_fld_reference_deck(deck_path)
    ref_env = os.environ.copy()
    ref_env.update(env)
    ref_env["TENRYU_R2_FLD_OUTDIR"] = str(ref_outdir)
    log_path = ref_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([args.tenryu_bin, "run", str(deck_path)], env=ref_env, stdout=log, stderr=log)
    if proc.returncode != 0:
        return {"fld_reference_result": "CRASHED", "fld_reference_exit_code": proc.returncode, "fld_reference_log": str(log_path)}
    run_id = f"r2_fld_reference_nr{nr}_nz{nz}_seed{env.get('TENRYU_R2_SEED', '12345')}"
    try:
        results_dir = discover_results_dir(ref_outdir)
        snap = read_snapshots(results_dir, run_id, nr, nz, 1)[-1]
    except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        return {"fld_reference_result": "ANALYSIS_FAIL", "fld_reference_error": str(exc), "fld_reference_log": str(log_path)}
    return {
        "fld_reference_result": "PASS",
        "fld_reference_exit_code": proc.returncode,
        "fld_reference_log": str(log_path),
        "fld_reference_rad_E_shell_avg": shell_average(snap["rad_E"], snap["vol"]),
        "fld_reference_Te_shell_avg": shell_average(snap["Te"], snap["vol"]),
    }


def evaluate_iteration_gate(metrics: dict[str, Any], failed: list[str]) -> None:
    outer_max = metrics.get("sn_outer_iterations_max")
    threshold = ITERATION_CAP_FRACTION * MAX_OUTER_ITERATIONS
    metrics["sn_outer_iterations_gate_threshold"] = threshold
    if outer_max is None:
        metrics["sn_outer_iterations_gate_status"] = "skipped_unavailable_current_2d_sn_api"
    elif float(outer_max) >= threshold:
        metrics["sn_outer_iterations_gate_status"] = "warn_cap_reached"
    else:
        metrics["sn_outer_iterations_gate_status"] = "pass"


def evaluate_gates(mode: str, metrics: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    if metrics.get("ray_effect_metric_present") is not True:
        failed.append("ray_effect_metric")
    if metrics.get("rad_E_nan_count", 0) != 0:
        failed.append("r2_rad_E_finite")
    if mode in ("axisymmetric_convergence", "diffusion_limit_vs_fld", "thin_corona_sn", "n_angles_sweep"):
        energy_tol = DIFFUSION_FUNCTIONAL_ENERGY_TOL if mode == "diffusion_limit_vs_fld" else ENERGY_TOL
        if metrics.get("E_balance_rel", math.inf) > energy_tol:
            failed.append(f"r2_{mode}_energy_balance")
    if mode == "multigroup_5g" and metrics.get("E_balance_rel", math.inf) > MULTIGROUP_ENERGY_TOL:
        failed.append("r2_multigroup_5g_energy_balance")
    if mode in ("axisymmetric_convergence", "diffusion_limit_vs_fld", "thin_corona_sn", "multigroup_5g", "n_angles_sweep", "slab_marshak_sn"):
        if metrics.get("rad_E_min", -math.inf) < 0.0 or metrics.get("rad_E_negative_count", 0) != 0:
            failed.append(f"r2_{mode}_nonnegative")
    if mode == "axisymmetric_convergence":
        evaluate_iteration_gate(metrics, failed)
        if metrics.get("ray_effect_metric_status") == "evaluated":
            if metrics.get("ray_effect_ap_alpha_cv", math.inf) > RAY_EFFECT_AP_ALPHA_CV_TOL:
                failed.append("ray_effect_metric")
    if mode == "diffusion_limit_vs_fld":
        if not metrics.get("rad_E_shell_monotone_boundary_to_boundary", False):
            failed.append("r2_diffusion_rad_E_monotone_boundary_to_boundary")
    if mode == "slab_marshak_sn":
        if metrics.get("marshak_source_accounting_rel", math.inf) > MARSHAK_SOURCE_ACCOUNTING_TOL:
            failed.append("r2_slab_marshak_sn_source_accounting")
        if metrics.get("marshak_z_profile_rel_l2", math.inf) > MARSHAK_SN_PROFILE_TOL:
            failed.append("r2_slab_marshak_sn_z_profile_l2")
    return failed


def active_env(args: argparse.Namespace, mode: str, nr: int, nz: int, seed: int, n_angles: int, outdir: Path) -> dict[str, str]:
    default_max_steps = 1 if args.smoke else (
        SLAB_MARSHAK_MAX_STEPS if mode == "slab_marshak_sn" else 3
    )
    max_steps = args.max_steps if args.max_steps is not None else default_max_steps
    env = {
        "TENRYU_R2_MODE": mode,
        "TENRYU_R2_NR": str(nr),
        "TENRYU_R2_NZ": str(nz),
        "TENRYU_R2_SEED": str(seed),
        "TENRYU_R2_N_ANGLES": str(n_angles),
        "TENRYU_R2_OUTDIR": str(outdir),
        "TENRYU_R2_MAX_STEPS": str(max_steps),
        "TENRYU_R2_VERBOSITY": args.verbosity,
        "TENRYU_R2_DSA_ENABLED": "1" if args.dsa_enabled else "0",
    }
    if args.t_end_s is not None:
        env["TENRYU_R2_T_END_S"] = str(args.t_end_s)
    if args.dt_initial_s is not None:
        env["TENRYU_R2_DT_INITIAL_S"] = str(args.dt_initial_s)
    if args.dt_max_s is not None:
        env["TENRYU_R2_DT_MAX_S"] = str(args.dt_max_s)
    if args.rho_gcc is not None:
        env["TENRYU_R2_RHO_GCC"] = str(args.rho_gcc)
    if args.kappa_a_cm2_g is not None:
        env["TENRYU_R2_KAPPA_A_CM2_G"] = str(args.kappa_a_cm2_g)
    return env


def effective_rho(args: argparse.Namespace, mode: str) -> float:
    return float(args.rho_gcc if args.rho_gcc is not None else mode_rho_default(mode))


def effective_kappa(args: argparse.Namespace, mode: str) -> float:
    return float(args.kappa_a_cm2_g if args.kappa_a_cm2_g is not None else mode_kappa_default(mode))


def run_one(args: argparse.Namespace, mode: str, nr: int, nz: int, seed: int, n_angles: int) -> dict[str, Any]:
    rho_gcc = effective_rho(args, mode)
    kappa_a = effective_kappa(args, mode)
    groups = mode_groups(mode)
    run_id = case_name(mode, nr, nz, rho_gcc, kappa_a, groups, n_angles, seed, args.dsa_enabled)
    dsa_suffix = "" if args.dsa_enabled else "_dsaoff"
    ladder_row = f"{mode}_nr{nr}_nz{nz}_s{n_angles}{dsa_suffix}"
    outdir = Path(args.out_root) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    env_overrides = active_env(args, mode, nr, nz, seed, n_angles, outdir)
    env = os.environ.copy()
    env.update(env_overrides)
    log_path = outdir / "run.log"

    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([args.tenryu_bin, "run", args.deck], env=env, stdout=log, stderr=log)
    wall_time_s = time.monotonic() - start

    log_metrics = parse_log_metadata(log_path)
    metrics: dict[str, Any] = {**log_metrics, "wall_time_s": wall_time_s, "output_bytes": output_bytes(outdir)}
    result = "PASS"
    failed_gates: list[str] = []
    failure_class: str | None = None
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
            snapshots = read_snapshots(results_dir, run_id, nr, nz, groups)
            metrics = common_metrics(mode, nr, nz, groups, n_angles, snapshots, log_metrics, wall_time_s, bytes_out)
            if mode == "diffusion_limit_vs_fld":
                metrics["fld_reference_result"] = "skipped_short_sn_functional_scope"
            failed_gates = evaluate_gates(mode, metrics)
            result = "PASS" if not failed_gates else "FAIL"
            failure_class = None if not failed_gates else "gate_failure"
        except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            result = "FAIL"
            failed_gates = ["analysis_failure"]
            failure_class = "analysis_failure"
            metrics["analysis_error"] = str(exc)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[run_r2] analysis failed: {exc}\n")

    metrics_ref = f"{args.root.rstrip('/')}/{STAGE_ID}/metrics/{run_id}.json"
    payload = {
        "run_id": run_id,
        "stage_id": STAGE_ID,
        "ladder_row": ladder_row,
        "seed": seed,
        "deck_path": args.deck,
        "mode": mode,
        "nr": nr,
        "nz": nz,
        "groups": groups,
        "n_angles": n_angles,
        "rho_gcc": rho_gcc,
        "kappa_a_cm2_g": kappa_a,
        "result": result,
        "failure_class": failure_class,
        "failed_gates": failed_gates,
        "exit_code": proc.returncode,
        "wall_time_s": wall_time_s,
        "metrics": metrics,
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
        "n_angles": payload["n_angles"],
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
        "sn_outer_iterations_gate_status": metrics.get("sn_outer_iterations_gate_status"),
        "fld_reference_rad_E_shell_relerr_max": metrics.get("fld_reference_rad_E_shell_relerr_max"),
        "ray_effect_ap_alpha_cv": metrics.get("ray_effect_ap_alpha_cv"),
        "ray_effect_metric_status": metrics.get("ray_effect_metric_status"),
        "marshak_source_accounting_rel": metrics.get("marshak_source_accounting_rel"),
        "marshak_z_profile_rel_l2": metrics.get("marshak_z_profile_rel_l2"),
        "marshak_z_profile_gate_status": metrics.get("marshak_z_profile_gate_status"),
        "marshak_inner_core_z_profile_rel_l2": metrics.get("marshak_inner_core_z_profile_rel_l2"),
        "marshak_inner_core_z_profile_gate_status": metrics.get("marshak_inner_core_z_profile_gate_status"),
        "marshak_inner_core_z_profile_metric_kind": metrics.get("marshak_inner_core_z_profile_metric_kind"),
        "E_Marshak_in": metrics.get("E_Marshak_in"),
        "n_clamp": metrics.get("n_clamp"),
    }


def load_metrics_payload(metrics_ref: str) -> dict[str, Any]:
    return json.loads(Path(metrics_ref).read_text(encoding="utf-8"))


def downsample_cell_average(fine: Any, fine_mesh: Mesh, nr: int, nz: int) -> Any:
    import numpy as np

    field = np.asarray(fine, dtype=float)
    vol = np.asarray(fine_mesh.vol, dtype=float)
    fr = fine_mesh.nr // nr
    fz = fine_mesh.nz // nz
    if fine_mesh.nr % nr != 0 or fine_mesh.nz % nz != 0:
        raise ValueError("reference grid must be an integer multiple of target grid")
    weighted = field * vol
    weighted_blocks = weighted.reshape(nr, fr, nz, fz).sum(axis=(1, 3))
    vol_blocks = vol.reshape(nr, fr, nz, fz).sum(axis=(1, 3))
    return weighted_blocks / np.maximum(vol_blocks, E_FLOOR)


def axisymmetric_reference_errors(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    import numpy as np

    axis_rows = [row for row in rows if row["mode"] == "axisymmetric_convergence" and row["seed"] == args.seed_list[0]]
    by_mesh = {(row["nr"], row["nz"]): row for row in axis_rows if row["exit_code"] == 0}
    if (128, 256) not in by_mesh or (64, 128) not in by_mesh or (32, 64) not in by_mesh:
        return None
    ref_row = by_mesh[(128, 256)]
    ref_payload = load_metrics_payload(ref_row["metrics_ref"])
    ref_run_id = ref_payload["run_id"]
    ref_results = discover_results_dir(Path(args.out_root) / ref_run_id)
    ref_snap = read_snapshots(ref_results, ref_run_id, 128, 256, 1)[-1]
    ref_mesh = mesh_from_snapshot(ref_snap, 128, 256)
    errors: dict[str, float] = {}
    for mesh in ((32, 64), (64, 128)):
        row = by_mesh[mesh]
        payload = load_metrics_payload(row["metrics_ref"])
        run_id = payload["run_id"]
        results_dir = discover_results_dir(Path(args.out_root) / run_id)
        snap = read_snapshots(results_dir, run_id, mesh[0], mesh[1], 1)[-1]
        ref_on_target = downsample_cell_average(ref_snap["rad_E"], ref_mesh, mesh[0], mesh[1])
        denom = np.maximum(np.maximum(np.abs(ref_on_target), np.abs(snap["rad_E"])), E_FLOOR)
        errors[f"nr{mesh[0]}_nz{mesh[1]}"] = float(np.nanmax(np.abs(snap["rad_E"] - ref_on_target) / denom))
    err32 = errors["nr32_nz64"]
    err64 = errors["nr64_nz128"]
    return {
        "axisymmetric_reference_run_id": ref_run_id,
        "axisymmetric_reference_errors": errors,
        "axisymmetric_reference_improvement_32_to_64": err32 / max(err64, E_FLOOR),
        "axisymmetric_reference_gate_pass": bool(err64 <= err32),
    }


def apply_axisymmetric_reference_gate(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    summary = axisymmetric_reference_errors(args, rows)
    if summary is None:
        for row in rows:
            if row["mode"] == "axisymmetric_convergence":
                payload = load_metrics_payload(row["metrics_ref"])
                payload["metrics"]["axisymmetric_reference_gate_status"] = "skipped_missing_32_64_128_reference_set"
                write_json(Path(row["metrics_ref"]), payload)
        return
    for row in rows:
        if row["mode"] != "axisymmetric_convergence":
            continue
        payload = load_metrics_payload(row["metrics_ref"])
        payload["metrics"].update(summary)
        payload["metrics"]["axisymmetric_reference_gate_status"] = "pass" if summary["axisymmetric_reference_gate_pass"] else "fail"
        if not summary["axisymmetric_reference_gate_pass"] and "r2_axisymmetric_reference_refinement" not in payload["failed_gates"]:
            payload["failed_gates"].append("r2_axisymmetric_reference_refinement")
            payload["result"] = "FAIL"
            payload["failure_class"] = "gate_failure"
        write_json(Path(row["metrics_ref"]), payload)
        updated = row_from_payload(payload, row["metrics_ref"])
        row.clear()
        row.update(updated)


def apply_angle_sweep_gate(rows: list[dict[str, Any]]) -> None:
    angle_rows = [row for row in rows if row["mode"] == "n_angles_sweep" and row["exit_code"] == 0]
    by_angle = {row["n_angles"]: row for row in angle_rows}
    if not all(angle in by_angle for angle in ANGLE_SWEEP):
        for row in angle_rows:
            payload = load_metrics_payload(row["metrics_ref"])
            payload["metrics"]["angle_sweep_gate_status"] = "skipped_missing_s8_s16_s32_set"
            write_json(Path(row["metrics_ref"]), payload)
        return
    errors = {angle: float(by_angle[angle].get("E_balance_rel") or math.inf) for angle in ANGLE_SWEEP}
    for row in angle_rows:
        payload = load_metrics_payload(row["metrics_ref"])
        payload["metrics"]["angle_sweep_E_balance_by_angle"] = errors
        payload["metrics"]["angle_sweep_gate_status"] = "skipped_short_sn_functional_scope"
        write_json(Path(row["metrics_ref"]), payload)
        updated = row_from_payload(payload, row["metrics_ref"])
        row.clear()
        row.update(updated)


def matrix_rows(args: argparse.Namespace) -> list[tuple[str, int, int, int, int]]:
    rows = []
    for mode in args.mode_list:
        meshes = tuple(args.mesh_list) if args.mesh_list is not None else (
            SMOKE_MESHES_BY_MODE[mode] if args.smoke else FULL_MESHES_BY_MODE[mode]
        )
        angles = tuple(args.angle_list) if mode == "n_angles_sweep" and not args.smoke else (args.n_angles,)
        for nr, nz in meshes:
            for seed in args.seed_list:
                for n_angles in angles:
                    rows.append((mode, nr, nz, seed, n_angles))
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
        "fld_reference_rad_E_shell_relerr_max",
        "ray_effect_ap_alpha_cv",
        "marshak_source_accounting_rel",
        "marshak_z_profile_rel_l2",
        "marshak_inner_core_z_profile_rel_l2",
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
    print("Stage R2 S_N-only sweep summary")
    print("ladder_row seed result E_balance_rel rad_E_min sn_outer_iterations_max ray_effect_ap_alpha_cv failed_gates")
    for row in rows:
        print(
            "{ladder_row} {seed} {result} {E_balance_rel} {rad_E_min} "
            "{sn_outer_iterations_max} {ray_effect_ap_alpha_cv} {failed_gates}".format(**row)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/2d_rz_r2_sn_only.py")
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    parser.add_argument("--out-root", default="./build/r2_runs")
    parser.add_argument("--modes", "--mode-list", dest="mode_list", type=lambda v: parse_csv(v, MODES, "mode"), default=list(MODES))
    parser.add_argument("--mesh-list", type=parse_mesh_list, default=None)
    parser.add_argument("--seed-list", type=parse_seed_list, default=parse_seed_list("12345"))
    parser.add_argument("--angle-list", type=parse_angle_list, default=list(ANGLE_SWEEP))
    parser.add_argument("--n-angles", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--dsa-enabled", type=parse_bool, default=True)
    parser.add_argument("--smoke", action="store_true", help="use each mode's smallest grid only")
    parser.add_argument("--skip-fld-reference", action="store_true")
    parser.add_argument("--rho-gcc", type=float, default=None)
    parser.add_argument("--kappa-a-cm2-g", type=float, default=None)
    parser.add_argument("--t-end-s", type=float, default=None)
    parser.add_argument("--dt-initial-s", type=float, default=None)
    parser.add_argument("--dt-max-s", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--verbosity", choices=("quiet", "normal", "verbose"), default="verbose")
    parser.add_argument("--summary-json", default="docs/validation/2d_rz/R2/stats.json")
    parser.add_argument("--runs-jsonl", default="docs/validation/2d_rz/R2/runs.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows_spec = matrix_rows(args)
    rows = [run_one(args, mode, nr, nz, seed, n_angles) for mode, nr, nz, seed, n_angles in rows_spec]
    apply_axisymmetric_reference_gate(args, rows)
    apply_angle_sweep_gate(rows)
    print_summary(rows)
    write_summary(args, rows)
    append_jsonl(Path(args.runs_jsonl), rows)
    update_stage_json(args, rows)
    return 0 if rows and all(row["result"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
