#!/usr/bin/env python3
"""Run the I1B-GEO 2D RZ mesh-geometric-admissibility gate.

I1B-GEO is a hydro-only, pre-accelerated capsule geometry-stress harness.  The
primary admissibility signal is the all-run committed post-ALE
``/diagnostics/mesh_quality_min/v1`` history group.  Sparse plot snapshots are
used for drive-validity diagnostics and mesh-quality corroboration only.
"""

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

import run_i1b_lite_2d_rz as lite


STAGE_ID = "I1B-GEO"
MODE = "i1b_geo_2d_rz_mesh_gate"
R_OUTER_DEFAULT_CM = 0.07
DEFAULT_V_PK_CMS = 3.5e7
DEFAULT_C_STAR_GRID = (4.0, 6.0, 8.0, 10.0, 12.0, 14.0)
DEFAULT_A0_GRID = (1.0e-3, 3.0e-3, 1.0e-2)
DEFAULT_CI_C_STAR = 8.0
DEFAULT_CI_A0 = 3.0e-3
DEFAULT_L_MIN = 2
DEFAULT_L_MAX = 10
DEFAULT_L_EFF = 10.0
DEFAULT_NR = 64
DEFAULT_NZ = 128
DEFAULT_DT_S = 2.0e-13
DEFAULT_PLOT_EVERY_STEPS = 50
DEFAULT_HISTORY_EVERY_STEPS = 1
DEFAULT_MAX_STEPS = 1_000_000
DRIVE_C_FRACTION_FLOOR = 0.9
INITIAL_E_KIN_FLOOR_ERG = 1.0e11
MESH_QUALITY_GROUP = "/diagnostics/mesh_quality_min/v1"
ALE_PROVENANCE_GROUP = "/diagnostics/ale_provenance/v1"


@dataclass(frozen=True)
class GeoCase:
    c_star: float
    a0: float
    v_pk_cms: float
    l_min: int
    l_max: int
    l_eff: float


@dataclass(frozen=True)
class DriveSnapshot:
    path: Path
    t: float
    step: int
    r_nodes: Any
    z_nodes: Any
    v_r_nodes: Any
    v_z_nodes: Any
    rho: Any
    vol: Any
    material_id: Any | None


def parse_float_list(value: str) -> list[float]:
    out = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one float")
    if any(not math.isfinite(item) or item <= 0.0 for item in out):
        raise argparse.ArgumentTypeError("values must be finite and positive")
    return sorted(dict.fromkeys(out))


def json_sanitize(value: Any) -> Any:
    return lite.json_sanitize(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    lite.write_json(path, payload)


def signal(value: Any, passed: bool, source: str, required: bool = True) -> dict[str, Any]:
    return lite.signal(value, passed, source, required)


def safe_float_token(value: float) -> str:
    return lite.safe_float_token(value)


def x_geo(case: GeoCase) -> float:
    return case.c_star * case.l_eff * case.a0


def expected_t_end_s(case: GeoCase, r_outer_cm: float = R_OUTER_DEFAULT_CM) -> float:
    return (r_outer_cm / case.v_pk_cms) * (1.0 - 1.0 / case.c_star)


def velocity_km_s(value_cms: float) -> float:
    return value_cms / 1.0e5


def compact_case_label(case: GeoCase) -> str:
    c_token = safe_float_token(case.c_star)
    a_token = f"{case.a0:.0e}".replace("+", "").replace("-", "m")
    v_token = f"{velocity_km_s(case.v_pk_cms):.0f}"
    return f"C{c_token}_A{a_token}_V{v_token}"


def planned_cases(args: argparse.Namespace) -> list[GeoCase]:
    if args.mode == "ci":
        return [
            GeoCase(
                c_star=float(args.ci_c_star),
                a0=float(args.ci_a0),
                v_pk_cms=float(args.v_pk_cms),
                l_min=int(args.l_min),
                l_max=int(args.l_max),
                l_eff=float(args.l_eff),
            )
        ]
    return [
        GeoCase(
            c_star=float(c_star),
            a0=float(a0),
            v_pk_cms=float(args.v_pk_cms),
            l_min=int(args.l_min),
            l_max=int(args.l_max),
            l_eff=float(args.l_eff),
        )
        for c_star in args.c_star_grid
        for a0 in args.a0_grid
    ]


def active_env(args: argparse.Namespace, outdir: Path, case: GeoCase) -> dict[str, str]:
    env = {
        "TENRYU_I1CAPSULE_OUTDIR": str(outdir),
        "TENRYU_I1CAPSULE_I1B_GEO_MODE": "true",
        "TENRYU_I1CAPSULE_IMPLOSION_V_PK_CMS": f"{case.v_pk_cms:.17g}",
        "TENRYU_I1CAPSULE_GEO_C_STAR": f"{case.c_star:.17g}",
        "TENRYU_I1CAPSULE_LEG_AMPLITUDE": f"{case.a0:.17g}",
        "TENRYU_I1CAPSULE_LEG_L_MIN": str(case.l_min),
        "TENRYU_I1CAPSULE_LEG_L_MAX": str(case.l_max),
        "TENRYU_I1CAPSULE_NR": str(args.nr),
        "TENRYU_I1CAPSULE_NZ": str(args.nz),
        "TENRYU_I1CAPSULE_RAYS_PER_BEAM": str(args.rays_per_beam),
        "TENRYU_I1CAPSULE_SEED": str(args.seed),
        "TENRYU_I1CAPSULE_DT_INITIAL_S": f"{args.dt_initial_s:.17g}",
        "TENRYU_I1CAPSULE_DT_MAX_S": f"{args.dt_max_s:.17g}",
        "TENRYU_I1CAPSULE_PLOT_EVERY_STEPS": str(args.plot_every_steps),
        "TENRYU_I1CAPSULE_HISTORY_EVERY_STEPS": str(args.history_every_steps),
        "TENRYU_I1CAPSULE_MAX_STEPS": str(args.max_steps),
        "TENRYU_I1CAPSULE_VERBOSITY": args.verbosity,
        "TENRYU_I1CAPSULE_PROFILE_ENABLED": "true",
        "TENRYU_I1CAPSULE_PROFILE_ENFORCE": "true",
        "TENRYU_I1CAPSULE_ALE_PROVENANCE_DIAGNOSTICS": "true",
        "TENRYU_I1CAPSULE_ICF_DIAGNOSTICS": "true",
        "TENRYU_I1CAPSULE_CONSERVATION_DIAGNOSTICS": "true",
        "TENRYU_I1CAPSULE_MESH_QUALITY_MIN_DIAG": "true",
        "TENRYU_I1CAPSULE_IN_HYDRO_GAUSS_J_GUARD": "true",
        "TENRYU_I1CAPSULE_IN_HYDRO_RZ_VOLUME_GUARD": "true",
        "TENRYU_I1CAPSULE_IN_HYDRO_GAUSS_J_FLOOR_REL": f"{lite.ADMISSIBILITY_FLOOR_REL:.17g}",
        "TENRYU_I1CAPSULE_IN_HYDRO_RZ_VOLUME_FLOOR_REL": f"{lite.ADMISSIBILITY_FLOOR_REL:.17g}",
        "TENRYU_I1CAPSULE_REJECT_ZERO_GAUSS_J": "true",
        "TENRYU_I1CAPSULE_ZERO_GAUSS_J_FLOOR_REL": f"{lite.ADMISSIBILITY_FLOOR_REL:.17g}",
    }
    if args.t_end_s is not None:
        env["TENRYU_I1CAPSULE_T_END_S"] = f"{args.t_end_s:.17g}"
    return env


def parse_deck_kv(log_path: Path) -> dict[str, str]:
    text = lite.read_text(log_path)
    for line in text.splitlines():
        if not line.startswith("[deck:2d_rz_i1_capsule]"):
            continue
        return {match.group(1): match.group(2) for match in re.finditer(r"([A-Za-z0-9_]+)=([^ \t]+)", line)}
    return {}


def bool_token(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    return None


def maybe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def maybe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_geo_log_metadata(log_path: Path) -> dict[str, Any]:
    out = lite.parse_log_metadata(log_path)
    kv = parse_deck_kv(log_path)
    float_keys = {
        "r_outer_cm": "deck_r_outer_cm",
        "leg_amplitude": "deck_leg_amplitude",
        "i1b_geo_v_pk_cms": "i1b_geo_v_pk_cms",
        "i1b_geo_c_star": "i1b_geo_c_star",
        "i1b_geo_a0": "i1b_geo_a0",
        "i1b_geo_t_end_s": "i1b_geo_t_end_s",
    }
    int_keys = {
        "leg_l_min": "deck_leg_l_min",
        "leg_l_max": "deck_leg_l_max",
    }
    for key, out_key in float_keys.items():
        value = maybe_float(kv.get(key))
        if value is not None:
            out[out_key] = value
    for key, out_key in int_keys.items():
        value = maybe_int(kv.get(key))
        if value is not None:
            out[out_key] = value
    geo_mode = bool_token(kv.get("i1b_geo_mode"))
    if geo_mode is not None:
        out["i1b_geo_mode"] = geo_mode
    if "i1b_geo_operator_semantics" in kv:
        out["i1b_geo_operator_semantics"] = kv["i1b_geo_operator_semantics"]
    return out


def reshape_cells(values: Any, nr: int, nz: int, name: str, dtype: Any = float) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=dtype)
    if arr.shape == (nr, nz):
        return arr
    if arr.size == nr * nz:
        return arr.reshape((nr, nz))
    raise ValueError(f"{name} has shape {arr.shape}, expected {(nr, nz)} or flat {nr * nz}")


def reshape_nodes(values: Any, nr: int, nz: int, name: str) -> Any:
    return lite.reshape_mesh_nodes(values, nr, nz, name)


def read_drive_snapshot(path: Path, nr: int, nz: int) -> DriveSnapshot:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        t_value = float(handle.attrs.get("t", 0.0))
        if "time_state/t" in handle:
            t_value = lite.scalar_dataset(handle, "time_state/t", t_value)
        step = int(lite.scalar_dataset(handle, "time_state/step", float(handle.attrs.get("cycle", 0))))
        material_id = None
        if "mesh/cell_material_id" in handle:
            material_id = reshape_cells(handle["mesh/cell_material_id"][()], nr, nz, "mesh/cell_material_id", int)
        return DriveSnapshot(
            path=path,
            t=t_value,
            step=step,
            r_nodes=reshape_nodes(handle["mesh/x_r"][()], nr, nz, "mesh/x_r"),
            z_nodes=reshape_nodes(handle["mesh/x_z"][()], nr, nz, "mesh/x_z"),
            v_r_nodes=reshape_nodes(handle["mesh/v_r"][()], nr, nz, "mesh/v_r"),
            v_z_nodes=reshape_nodes(handle["mesh/v_z"][()], nr, nz, "mesh/v_z"),
            rho=reshape_cells(handle["hydro/rho"][()], nr, nz, "hydro/rho"),
            vol=reshape_cells(handle["hydro/vol"][()], nr, nz, "hydro/vol"),
            material_id=material_id,
        )


def cell_centers(snapshot: DriveSnapshot) -> tuple[Any, Any]:
    r = snapshot.r_nodes
    z = snapshot.z_nodes
    return (
        0.25 * (r[:-1, :-1] + r[1:, :-1] + r[:-1, 1:] + r[1:, 1:]),
        0.25 * (z[:-1, :-1] + z[1:, :-1] + z[:-1, 1:] + z[1:, 1:]),
    )


def nodal_velocity_to_cells(snapshot: DriveSnapshot) -> tuple[Any, Any]:
    vr = snapshot.v_r_nodes
    vz = snapshot.v_z_nodes
    return (
        0.25 * (vr[:-1, :-1] + vr[1:, :-1] + vr[:-1, 1:] + vr[1:, 1:]),
        0.25 * (vz[:-1, :-1] + vz[1:, :-1] + vz[:-1, 1:] + vz[1:, 1:]),
    )


def kinetic_energy_erg(snapshot: DriveSnapshot) -> float:
    import numpy as np

    vr, vz = nodal_velocity_to_cells(snapshot)
    ke = 0.5 * snapshot.rho * snapshot.vol * (vr * vr + vz * vz)
    finite = ke[np.isfinite(ke)]
    return float(np.sum(finite)) if finite.size else math.nan


def infer_material_classes(snapshot: DriveSnapshot) -> dict[str, Any]:
    import numpy as np

    if snapshot.material_id is None:
        positive = snapshot.rho[np.isfinite(snapshot.rho) & (snapshot.rho > 0.0)]
        threshold = None
        if positive.size:
            threshold = float(math.sqrt(float(np.min(positive)) * float(np.max(positive))))
        return {
            "available": False,
            "source": "density_contour",
            "outer_density_threshold_gcc": threshold,
        }

    medians: list[tuple[float, int, int]] = []
    for material_id in sorted(int(item) for item in np.unique(snapshot.material_id)):
        mask = snapshot.material_id == material_id
        values = snapshot.rho[mask & np.isfinite(snapshot.rho)]
        if values.size:
            medians.append((float(np.median(values)), material_id, int(values.size)))
    medians.sort(key=lambda item: item[0])
    classes: dict[str, Any] = {
        "available": bool(medians),
        "source": "mesh/cell_material_id_with_initial_density_order",
        "material_density_medians_gcc": [
            {"material_id": material_id, "median_rho_gcc": median, "cell_count": count}
            for median, material_id, count in medians
        ],
    }
    if medians:
        classes["corona_material_id"] = medians[0][1]
    if len(medians) >= 2:
        classes["dt_gas_material_id"] = medians[1][1]
    if len(medians) >= 3:
        classes["dt_ice_material_id"] = medians[2][1]
    if len(medians) >= 4:
        classes["cd_material_id"] = medians[-1][1]
    return classes


def density_outer_threshold(snapshot: DriveSnapshot) -> float | None:
    import numpy as np

    positive = snapshot.rho[np.isfinite(snapshot.rho) & (snapshot.rho > 0.0)]
    if positive.size == 0:
        return None
    return float(math.sqrt(float(np.min(positive)) * float(np.max(positive))))


def legendre_fit(mu: Any, radius: Any, l_min: int, l_max: int) -> dict[str, Any]:
    import numpy as np

    mu_arr = np.asarray(mu, dtype=float)
    radius_arr = np.asarray(radius, dtype=float)
    valid = np.isfinite(mu_arr) & np.isfinite(radius_arr) & (radius_arr > 0.0)
    mu_arr = mu_arr[valid]
    radius_arr = radius_arr[valid]
    mode_count = 1 + max(0, l_max - l_min + 1)
    if mu_arr.size < mode_count:
        return {
            "available": False,
            "sample_count": int(mu_arr.size),
            "missing": ["insufficient_interface_samples_for_legendre_fit"],
        }
    vander = np.polynomial.legendre.legvander(mu_arr, l_max)
    ells = list(range(l_min, l_max + 1))
    design = np.column_stack([vander[:, 0]] + [vander[:, ell] for ell in ells])
    coeff, residuals, rank, singular_values = np.linalg.lstsq(design, radius_arr, rcond=None)
    rbar = float(coeff[0])
    if not math.isfinite(rbar) or rbar <= 0.0:
        return {
            "available": False,
            "sample_count": int(mu_arr.size),
            "missing": ["nonpositive_legendre_fit_rbar"],
        }
    a_l = {str(ell): float(coeff[idx + 1] / rbar) for idx, ell in enumerate(ells)}
    l_times_a_l = {str(ell): float(ell * a_l[str(ell)]) for ell in ells}
    denom = sum(value * value for value in a_l.values())
    spectral_rms_l = None
    if denom > 0.0:
        spectral_rms_l = math.sqrt(
            sum(float(ell * (ell + 1)) * a_l[str(ell)] * a_l[str(ell)] for ell in ells) / denom
        )
    fit_radius = design @ coeff
    residual = radius_arr - fit_radius
    rms_rel = float(np.sqrt(np.mean((residual / rbar) ** 2))) if residual.size else None
    return {
        "available": True,
        "sample_count": int(mu_arr.size),
        "rbar_cm": rbar,
        "a_l": a_l,
        "l_times_a_l": l_times_a_l,
        "spectral_rms_l": spectral_rms_l,
        "fit_rms_rel": rms_rel,
        "fit_rank": int(rank),
        "fit_singular_values": [float(item) for item in singular_values],
    }


def boundary_samples_from_mask(snapshot: DriveSnapshot, mask: Any, n_bins: int) -> tuple[Any, Any]:
    import numpy as np

    r_center, z_center = cell_centers(snapshot)
    radius = np.sqrt(r_center * r_center + z_center * z_center)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(radius > 0.0, z_center / radius, np.nan)
    valid = mask & np.isfinite(radius) & np.isfinite(mu) & (radius > 0.0)
    mu_samples: list[float] = []
    radius_samples: list[float] = []
    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    for idx in range(n_bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        in_bin = valid & (mu >= lo) & ((mu < hi) if idx + 1 < n_bins else (mu <= hi))
        if not np.any(in_bin):
            continue
        radii = np.where(in_bin, radius, -np.inf)
        flat_idx = int(np.argmax(radii))
        i, j = np.unravel_index(flat_idx, radius.shape)
        mu_samples.append(float(mu[i, j]))
        radius_samples.append(float(radius[i, j]))
    return np.asarray(mu_samples, dtype=float), np.asarray(radius_samples, dtype=float)


def interface_masks(snapshot: DriveSnapshot, material_classes: dict[str, Any], fallback_density_threshold: float | None) -> dict[str, Any]:
    import numpy as np

    masks: dict[str, Any] = {}
    if snapshot.material_id is not None and "corona_material_id" in material_classes:
        mat = snapshot.material_id
        corona = int(material_classes["corona_material_id"])
        masks["outer_shell_corona"] = mat != corona
        if "cd_material_id" in material_classes:
            cd = int(material_classes["cd_material_id"])
            masks["ablator_inner"] = (mat != corona) & (mat != cd)
        if "dt_gas_material_id" in material_classes:
            gas = int(material_classes["dt_gas_material_id"])
            masks["dt_gas_outer"] = mat == gas
    if "outer_shell_corona" not in masks and fallback_density_threshold is not None:
        masks["outer_density_contour"] = np.isfinite(snapshot.rho) & (snapshot.rho >= fallback_density_threshold)
    return masks


def modal_interfaces(
    snapshot: DriveSnapshot,
    material_classes: dict[str, Any],
    fallback_density_threshold: float | None,
    l_min: int,
    l_max: int,
    n_bins: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, mask in interface_masks(snapshot, material_classes, fallback_density_threshold).items():
        mu, radius = boundary_samples_from_mask(snapshot, mask, n_bins)
        fit = legendre_fit(mu, radius, l_min, l_max)
        fit.update({"source": name})
        out[name] = fit
    return out


def choose_outer_interface(interfaces: dict[str, Any]) -> dict[str, Any] | None:
    for name in ("outer_shell_corona", "outer_density_contour"):
        item = interfaces.get(name)
        if isinstance(item, dict) and item.get("available"):
            return item
    return None


def sampled_corner_j_stats(snapshots: list[DriveSnapshot]) -> dict[str, Any]:
    import numpy as np

    if len(snapshots) < 2:
        return {
            "available": False,
            "missing": ["at_least_two_snapshots_for_sampled_j_rel"],
        }
    initial = snapshots[0]
    initial_j = lite.corner_jacobians(initial.r_nodes, initial.z_nodes)
    initial_v = lite.rz_quad_volume_exact(initial.r_nodes, initial.z_nodes)
    ratios_all: list[Any] = []
    volume_ratios_all: list[Any] = []
    worst: dict[str, Any] | None = None
    worst_value = math.inf
    for snap in snapshots:
        current_j = lite.corner_jacobians(snap.r_nodes, snap.z_nodes)
        stacked = []
        for old, new in zip(initial_j, current_j):
            with np.errstate(divide="ignore", invalid="ignore"):
                stacked.append(np.where(old != 0.0, new / old, np.nan))
        ratio = np.stack(stacked, axis=0)
        finite_ratio = ratio[np.isfinite(ratio)]
        if finite_ratio.size:
            ratios_all.append(finite_ratio)
            local_min = float(np.min(finite_ratio))
            if local_min < worst_value:
                flat_idx = int(np.nanargmin(ratio))
                corner_idx, i, j = np.unravel_index(flat_idx, ratio.shape)
                r_center, z_center = cell_centers(snap)
                mat = None
                if snap.material_id is not None:
                    mat = int(snap.material_id[i, j])
                worst_value = local_min
                worst = {
                    "value": local_min,
                    "snapshot_path": str(snap.path),
                    "step": snap.step,
                    "time_s": snap.t,
                    "corner_index": int(corner_idx),
                    "cell_i": int(i),
                    "cell_j": int(j),
                    "R_cm": float(r_center[i, j]),
                    "Z_cm": float(z_center[i, j]),
                    "axis_distance_cm": float(abs(r_center[i, j])),
                    "material_id": mat,
                }
        current_v = lite.rz_quad_volume_exact(snap.r_nodes, snap.z_nodes)
        with np.errstate(divide="ignore", invalid="ignore"):
            v_ratio = np.where(initial_v != 0.0, current_v / initial_v, np.nan)
        finite_v = v_ratio[np.isfinite(v_ratio)]
        if finite_v.size:
            volume_ratios_all.append(finite_v)
    if not ratios_all:
        return {
            "available": False,
            "missing": ["finite_sampled_corner_j_rel"],
        }
    all_corner = np.concatenate(ratios_all)
    out = {
        "available": True,
        "source": "sparse_plot_snapshots_vs_initial",
        "snapshot_count": len(snapshots),
        "min_corner_j_rel": float(np.min(all_corner)),
        "p0p1_corner_j_rel": float(np.percentile(all_corner, 0.1)),
        "p1_corner_j_rel": float(np.percentile(all_corner, 1.0)),
        "worst_cell": worst,
    }
    if volume_ratios_all:
        all_vol = np.concatenate(volume_ratios_all)
        out["min_rz_volume_rel"] = float(np.min(all_vol))
        out["p0p1_rz_volume_rel"] = float(np.percentile(all_vol, 0.1))
        out["p1_rz_volume_rel"] = float(np.percentile(all_vol, 1.0))
    return out


def evaluate_drive_validity(
    results_dir: Path | None,
    *,
    nr: int,
    nz: int,
    c_star: float,
    r_outer_cm: float,
    l_min: int,
    l_max: int,
    n_interface_bins: int,
) -> dict[str, Any]:
    missing: list[str] = []
    failed: list[str] = []
    analysis_errors: list[str] = []
    if results_dir is None:
        missing.append("results_dir")
        return {
            "available": False,
            "passed": False,
            "missing": missing,
            "failed": failed,
            "analysis_errors": analysis_errors,
            "signals": {},
        }

    plot_files = lite.locate_plot_files(results_dir)
    if len(plot_files) < 2:
        missing.append("at_least_two_plot_snapshots")
        return {
            "available": False,
            "passed": False,
            "missing": missing,
            "failed": failed,
            "analysis_errors": analysis_errors,
            "plot_file_count": len(plot_files),
            "signals": {},
        }

    try:
        snapshots = [read_drive_snapshot(path, nr, nz) for path in plot_files]
        material_classes = infer_material_classes(snapshots[0])
        fallback_threshold = material_classes.get("outer_density_threshold_gcc")
        if fallback_threshold is None:
            fallback_threshold = density_outer_threshold(snapshots[0])
        interfaces_by_snapshot = [
            {
                "path": str(snap.path),
                "step": snap.step,
                "time_s": snap.t,
                "interfaces": modal_interfaces(
                    snap,
                    material_classes,
                    fallback_threshold,
                    l_min,
                    l_max,
                    n_interface_bins,
                ),
            }
            for snap in snapshots
        ]
        outer_by_snapshot = [
            choose_outer_interface(item["interfaces"]) for item in interfaces_by_snapshot
        ]
        if any(item is None for item in outer_by_snapshot):
            missing.append("outer_interface_modal_fit")
        compression_series = []
        for item, outer in zip(interfaces_by_snapshot, outer_by_snapshot):
            if outer is None:
                continue
            rbar = float(outer["rbar_cm"])
            compression_series.append(
                {
                    "step": item["step"],
                    "time_s": item["time_s"],
                    "rbar_outer_cm": rbar,
                    "C_meas": r_outer_cm / rbar if rbar > 0.0 else None,
                    "outer_source": outer.get("source"),
                    "outer_spectral_rms_l": outer.get("spectral_rms_l"),
                }
            )
        e_kin0 = kinetic_energy_erg(snapshots[0])
        j_stats = sampled_corner_j_stats(snapshots)
        if not j_stats.get("available"):
            missing.extend(j_stats.get("missing", ["sampled_corner_j_stats"]))
    except (OSError, KeyError, ValueError, RuntimeError, FloatingPointError) as exc:
        analysis_errors.append(str(exc))
        return {
            "available": False,
            "passed": False,
            "missing": missing,
            "failed": failed,
            "analysis_errors": analysis_errors,
            "source_path": str(results_dir),
            "signals": {},
        }

    final_c = None
    if compression_series:
        final_c = compression_series[-1].get("C_meas")
    c_required = DRIVE_C_FRACTION_FLOOR * c_star
    signals = {
        "C_meas_final": signal(
            final_c,
            isinstance(final_c, (int, float)) and math.isfinite(float(final_c)) and float(final_c) >= c_required,
            str(plot_files[-1]),
        ),
        "initial_kinetic_energy_erg": signal(
            e_kin0,
            math.isfinite(e_kin0) and e_kin0 > INITIAL_E_KIN_FLOOR_ERG,
            str(plot_files[0]),
        ),
        "outer_modal_fit_available": signal(
            not any(item is None for item in outer_by_snapshot),
            not any(item is None for item in outer_by_snapshot),
            str(results_dir),
        ),
        "sampled_corner_j_stats_available": signal(
            bool(j_stats.get("available")),
            bool(j_stats.get("available")),
            str(results_dir),
        ),
    }
    for name, item in signals.items():
        if item["required"] and item["value"] is None:
            missing.append(name)
        if item["required"] and not item["passed"]:
            failed.append(name)
    missing = sorted(set(missing))
    failed = sorted(set(failed))
    return {
        "available": not missing and not analysis_errors,
        "passed": not missing and not failed and not analysis_errors,
        "missing": missing,
        "failed": failed,
        "analysis_errors": analysis_errors,
        "source_path": str(results_dir),
        "plot_file_count": len(plot_files),
        "first_snapshot": {
            "path": str(snapshots[0].path),
            "step": snapshots[0].step,
            "time_s": snapshots[0].t,
        },
        "last_snapshot": {
            "path": str(snapshots[-1].path),
            "step": snapshots[-1].step,
            "time_s": snapshots[-1].t,
        },
        "material_boundary_model": material_classes,
        "fallback_outer_density_threshold_gcc": fallback_threshold,
        "compression_required": {
            "C_star": c_star,
            "fraction": DRIVE_C_FRACTION_FLOOR,
            "C_required": c_required,
            "R0_outer_cm": r_outer_cm,
        },
        "compression_series": compression_series,
        "initial_kinetic_energy_erg": e_kin0,
        "modal_interfaces": interfaces_by_snapshot,
        "sampled_mesh_quality_corrob": j_stats,
        "signals": signals,
    }


def read_required_series(handle: Any, dataset: str, missing: list[str]) -> Any | None:
    import numpy as np

    key = dataset[1:] if dataset.startswith("/") else dataset
    if key not in handle:
        missing.append(dataset)
        return None
    arr = np.asarray(handle[key][()]).reshape(-1)
    if arr.size == 0:
        missing.append(f"{dataset}:empty")
        return None
    return arr


def read_mesh_quality_min_history(history: Path | None) -> dict[str, Any]:
    if history is None:
        return {
            "available": False,
            "source": MESH_QUALITY_GROUP,
            "source_path": None,
            "missing": ["history_hdf5"],
        }
    import h5py
    import numpy as np

    missing: list[str] = []
    analysis_errors: list[str] = []
    try:
        with h5py.File(history, "r") as handle:
            paths = {
                "time_s": f"{MESH_QUALITY_GROUP}/time_s",
                "step": f"{MESH_QUALITY_GROUP}/step",
                "achieved_min_corner_j_rel": f"{MESH_QUALITY_GROUP}/achieved_min_corner_j_rel",
                "achieved_min_gauss_j_rel": f"{MESH_QUALITY_GROUP}/achieved_min_gauss_j_rel",
                "achieved_min_rz_volume_rel": f"{MESH_QUALITY_GROUP}/achieved_min_rz_volume_rel",
                "negative_rz_volume_count_total": f"{MESH_QUALITY_GROUP}/negative_rz_volume_count_total",
            }
            series = {name: read_required_series(handle, path, missing) for name, path in paths.items()}
    except OSError as exc:
        analysis_errors.append(str(exc))
        return {
            "available": False,
            "source": MESH_QUALITY_GROUP,
            "source_path": str(history),
            "missing": missing,
            "analysis_errors": analysis_errors,
        }
    if missing:
        return {
            "available": False,
            "source": MESH_QUALITY_GROUP,
            "source_path": str(history),
            "missing": sorted(set(missing)),
            "analysis_errors": analysis_errors,
        }
    corner = np.asarray(series["achieved_min_corner_j_rel"], dtype=float)
    gauss = np.asarray(series["achieved_min_gauss_j_rel"], dtype=float)
    rz_volume = np.asarray(series["achieved_min_rz_volume_rel"], dtype=float)
    neg = np.asarray(series["negative_rz_volume_count_total"], dtype=float)
    finite_corner = corner[np.isfinite(corner)]
    finite_gauss = gauss[np.isfinite(gauss)]
    finite_rz = rz_volume[np.isfinite(rz_volume)]
    finite_neg = neg[np.isfinite(neg)]
    if finite_corner.size == 0:
        missing.append("finite_achieved_min_corner_j_rel")
    if finite_gauss.size == 0:
        missing.append("finite_achieved_min_gauss_j_rel")
    if finite_rz.size == 0:
        missing.append("finite_achieved_min_rz_volume_rel")
    if finite_neg.size == 0:
        missing.append("finite_negative_rz_volume_count_total")
    if missing:
        return {
            "available": False,
            "source": MESH_QUALITY_GROUP,
            "source_path": str(history),
            "missing": sorted(set(missing)),
            "analysis_errors": analysis_errors,
        }
    step = np.asarray(series["step"], dtype=float)
    time_s = np.asarray(series["time_s"], dtype=float)
    return {
        "available": True,
        "source": MESH_QUALITY_GROUP,
        "source_path": str(history),
        "row_count": int(min(len(corner), len(gauss), len(rz_volume), len(neg))),
        "first_step": int(step[0]) if step.size else None,
        "last_step": int(step[-1]) if step.size else None,
        "first_t_s": float(time_s[0]) if time_s.size else None,
        "last_t_s": float(time_s[-1]) if time_s.size else None,
        "min_corner_j_rel": float(np.min(finite_corner)),
        "min_gauss_j_rel": float(np.min(finite_gauss)),
        "min_rz_volume_rel": float(np.min(finite_rz)),
        "negative_rz_volume_count_total_max": int(np.max(finite_neg)),
        "final_running_min_corner_j_rel": float(corner[-1]),
        "final_running_min_gauss_j_rel": float(gauss[-1]),
        "final_running_min_rz_volume_rel": float(rz_volume[-1]),
        "final_negative_rz_volume_count_total": int(neg[-1]),
        "missing": [],
        "analysis_errors": analysis_errors,
    }


def read_run_info(path: Path | None, errors: list[str]) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"run_info parse failed: {exc}")
        return {}


def evaluate_completion_escape_and_fail_closed(
    *,
    exit_code: int,
    timed_out: bool,
    outdir: Path,
    log_path: Path,
    results_dir: Path | None,
    history: Path | None,
    frozen_config: Path | None,
    run_info_path: Path | None,
    t_end_s: float,
) -> dict[str, Any]:
    analysis_errors: list[str] = []
    run_info = read_run_info(run_info_path, analysis_errors)
    actual_outdir = results_dir.parent if results_dir is not None else outdir
    audit = lite.run_audit_summary(run_info_path, history, frozen_config, outdir)
    no_escape = lite.no_escape_valves_from_sources(log_path, history, actual_outdir, audit)
    empty_repair_polytope_count = lite.count_empty_repair_polytope(lite.read_text(log_path))
    final_t = lite.finite_float(run_info.get("t"))
    termination_reason = str(run_info.get("termination_reason", "missing"))
    reached_t_end = (
        exit_code == 0
        and not timed_out
        and termination_reason == "t_end_reached"
        and final_t is not None
        and final_t >= 0.999 * t_end_s
    )
    mesh_geometry_failures = lite.hdf5_series_max(
        history, f"{ALE_PROVENANCE_GROUP}/mesh_geometry_failures_observed"
    )
    signals = {
        "reached_t_end": signal(
            reached_t_end,
            reached_t_end,
            str(run_info_path) if run_info_path is not None else "missing run_info.json",
        ),
        "emergency_cell_deactivation_fired_count": signal(
            no_escape.get("emergency_cell_deactivation_fired_count"),
            no_escape.get("emergency_cell_deactivation_fired_count") == 0,
            str(no_escape["sources"].get("history_hdf5") or no_escape["sources"].get("escape_valve_events_jsonl")),
        ),
        "empty_repair_polytope_count": signal(
            empty_repair_polytope_count,
            empty_repair_polytope_count == 0,
            str(log_path),
        ),
        "escape_valve_counters_all_zero": signal(
            no_escape.get("nonzero_counters"),
            bool(no_escape.get("all_zero")),
            str(no_escape["sources"].get("history_hdf5") or log_path),
        ),
        "mesh_geometry_failures_observed_count": signal(
            mesh_geometry_failures,
            mesh_geometry_failures == 0,
            str(history) if history is not None else "missing history_hdf5",
        ),
    }
    missing = []
    if not run_info:
        missing.append("run_info")
    missing.extend(no_escape.get("missing_counters", []))
    for name, item in signals.items():
        if item["required"] and item["value"] is None:
            missing.append(name)
    failed = [name for name, item in signals.items() if item["required"] and not item["passed"]]
    return {
        "passed": not timed_out and exit_code == 0 and not analysis_errors and not missing and not failed,
        "missing_signals": sorted(set(missing)),
        "failed_signals": failed,
        "signals": signals,
        "no_escape_valves": no_escape,
        "audit_summary": audit,
        "run_info": run_info,
        "analysis_errors": analysis_errors,
        "actual_outdir": str(actual_outdir),
        "termination_reason": termination_reason,
        "final_t_s": final_t,
        "t_end_required_s": t_end_s,
    }


def mesh_quality_geometry_signals(mesh_quality: dict[str, Any]) -> dict[str, Any]:
    return {
        "min_corner_j_rel": signal(
            mesh_quality.get("min_corner_j_rel"),
            isinstance(mesh_quality.get("min_corner_j_rel"), (int, float))
            and math.isfinite(float(mesh_quality["min_corner_j_rel"]))
            and float(mesh_quality["min_corner_j_rel"]) > lite.ADMISSIBILITY_FLOOR_REL,
            str(mesh_quality.get("source_path") or "missing mesh_quality_min history"),
        ),
        "min_gauss_j_rel": signal(
            mesh_quality.get("min_gauss_j_rel"),
            isinstance(mesh_quality.get("min_gauss_j_rel"), (int, float))
            and math.isfinite(float(mesh_quality["min_gauss_j_rel"]))
            and float(mesh_quality["min_gauss_j_rel"]) > lite.ADMISSIBILITY_FLOOR_REL,
            str(mesh_quality.get("source_path") or "missing mesh_quality_min history"),
        ),
        "min_rz_volume_rel": signal(
            mesh_quality.get("min_rz_volume_rel"),
            isinstance(mesh_quality.get("min_rz_volume_rel"), (int, float))
            and math.isfinite(float(mesh_quality["min_rz_volume_rel"]))
            and float(mesh_quality["min_rz_volume_rel"]) > 0.0,
            str(mesh_quality.get("source_path") or "missing mesh_quality_min history"),
        ),
        "negative_rz_volume_count_total": signal(
            mesh_quality.get("negative_rz_volume_count_total_max"),
            mesh_quality.get("negative_rz_volume_count_total_max") == 0,
            str(mesh_quality.get("source_path") or "missing mesh_quality_min history"),
        ),
    }


def evaluate_predicate(
    *,
    exit_code: int,
    timed_out: bool,
    outdir: Path,
    log_path: Path,
    args: argparse.Namespace,
    case: GeoCase,
) -> dict[str, Any]:
    metadata = parse_geo_log_metadata(log_path)
    analysis_errors: list[str] = []
    results_dir: Path | None = None
    history: Path | None = None
    run_info_path: Path | None = None
    frozen_config: Path | None = None
    try:
        results_dir = lite.discover_results_dir(outdir)
        history = lite.locate_history_file(results_dir)
        run_info_path = lite.locate_run_info(outdir, results_dir)
        frozen_config = lite.locate_frozen_config(outdir, results_dir)
    except (OSError, RuntimeError, FileNotFoundError) as exc:
        analysis_errors.append(str(exc))
        run_info_path = lite.locate_run_info(outdir, results_dir)

    nr = int(metadata.get("deck_nr", args.nr))
    nz = int(metadata.get("deck_nz", args.nz))
    r_outer_cm = float(metadata.get("deck_r_outer_cm", R_OUTER_DEFAULT_CM))
    t_end_s = float(
        metadata.get(
            "i1b_geo_t_end_s",
            metadata.get("deck_t_end_s", args.t_end_s if args.t_end_s is not None else expected_t_end_s(case, r_outer_cm)),
        )
    )
    mesh_quality = read_mesh_quality_min_history(history)
    completion = evaluate_completion_escape_and_fail_closed(
        exit_code=exit_code,
        timed_out=timed_out,
        outdir=outdir,
        log_path=log_path,
        results_dir=results_dir,
        history=history,
        frozen_config=frozen_config,
        run_info_path=run_info_path,
        t_end_s=t_end_s,
    )
    drive = evaluate_drive_validity(
        results_dir,
        nr=nr,
        nz=nz,
        c_star=case.c_star,
        r_outer_cm=r_outer_cm,
        l_min=case.l_min,
        l_max=case.l_max,
        n_interface_bins=args.interface_bins,
    )

    geometry_signals = mesh_quality_geometry_signals(mesh_quality)
    signals = {
        **completion.get("signals", {}),
        **geometry_signals,
        **{f"drive_{name}": value for name, value in drive.get("signals", {}).items()},
    }
    missing_signals = []
    missing_signals.extend(mesh_quality.get("missing", []))
    missing_signals.extend(completion.get("missing_signals", []))
    missing_signals.extend(f"drive:{name}" for name in drive.get("missing", []))
    for name, item in signals.items():
        if item["required"] and item["value"] is None:
            missing_signals.append(name)
    failed_signals = [name for name, item in signals.items() if item["required"] and not item["passed"]]
    missing_signals = sorted(set(missing_signals))
    all_analysis_errors = (
        analysis_errors
        + mesh_quality.get("analysis_errors", [])
        + completion.get("analysis_errors", [])
        + drive.get("analysis_errors", [])
    )
    predicate_passed = (
        not timed_out
        and exit_code == 0
        and not all_analysis_errors
        and not missing_signals
        and not failed_signals
    )
    geometry_admissible = all(
        item["passed"] for item in geometry_signals.values()
    ) and mesh_quality.get("available")
    drive_valid = bool(drive.get("passed"))
    completion_valid = bool(completion.get("passed"))
    return {
        "passed": predicate_passed,
        "geometry_admissible": bool(geometry_admissible),
        "completion_escape_fail_closed_passed": completion_valid,
        "drive_validity_passed": drive_valid,
        "missing_signals": missing_signals,
        "failed_signals": failed_signals,
        "signals": signals,
        "thresholds": {
            "corner_j_rel_floor": lite.ADMISSIBILITY_FLOOR_REL,
            "gauss_j_rel_floor": lite.ADMISSIBILITY_FLOOR_REL,
            "rz_volume_rel_floor": 0.0,
            "negative_rz_volume_count_total": 0,
            "C_meas_final_floor": DRIVE_C_FRACTION_FLOOR * case.c_star,
            "initial_kinetic_energy_floor_erg": INITIAL_E_KIN_FLOOR_ERG,
        },
        "mesh_quality_min": mesh_quality,
        "completion_escape_fail_closed": completion,
        "drive_validity": drive,
        "metadata": metadata,
        "analysis_errors": all_analysis_errors,
        "paths": {
            "outdir": str(outdir),
            "actual_outdir": completion.get("actual_outdir"),
            "log_path": str(log_path),
            "results_dir": str(results_dir) if results_dir is not None else None,
            "history_hdf5": str(history) if history is not None else None,
            "run_info": str(run_info_path) if run_info_path is not None else None,
            "frozen_config": str(frozen_config) if frozen_config is not None else None,
        },
    }


def row_from_payload(payload: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    predicate = payload["predicate"]
    signals = predicate.get("signals", {})
    drive = predicate.get("drive_validity", {})
    compression = drive.get("compression_series", [])
    final_compression = compression[-1] if compression else {}
    sampled = drive.get("sampled_mesh_quality_corrob", {})
    missing_signals = predicate.get("missing_signals", [])
    analysis_errors = predicate.get("analysis_errors", [])
    harness_clean = (not bool(payload["timed_out"])) and not missing_signals and not analysis_errors
    return {
        "run_id": payload["run_id"],
        "case_label": payload["case_label"],
        "x_geo": payload["x_geo"],
        "c_star": payload["c_star"],
        "a0": payload["a0"],
        "l_eff": payload["l_eff"],
        "v_pk_cms": payload["v_pk_cms"],
        "v_pk_km_s": payload["v_pk_km_s"],
        "erg_equivalent": None,
        "erg_equivalent_note": "N/A (geometric stress, not laser energy)",
        "result": payload["result"],
        "admissible": predicate["passed"],
        "geometry_admissible": predicate.get("geometry_admissible"),
        "drive_validity_passed": predicate.get("drive_validity_passed"),
        "completion_escape_fail_closed_passed": predicate.get("completion_escape_fail_closed_passed"),
        "exit_code": payload["exit_code"],
        "timed_out": payload["timed_out"],
        "wall_time_s": payload["wall_time_s"],
        "harness_clean": harness_clean,
        "predicate_summary_json": str(summary_path),
        "missing_signals": missing_signals,
        "failed_signals": predicate.get("failed_signals", []),
        "analysis_errors": analysis_errors,
        "final_t_s": predicate.get("completion_escape_fail_closed", {}).get("final_t_s"),
        "termination_reason": predicate.get("completion_escape_fail_closed", {}).get("termination_reason"),
        "min_corner_j_rel": signals.get("min_corner_j_rel", {}).get("value"),
        "min_gauss_j_rel": signals.get("min_gauss_j_rel", {}).get("value"),
        "min_rz_volume_rel": signals.get("min_rz_volume_rel", {}).get("value"),
        "negative_rz_volume_count_total": signals.get("negative_rz_volume_count_total", {}).get("value"),
        "mesh_geometry_failures_observed_count": signals.get("mesh_geometry_failures_observed_count", {}).get("value"),
        "escape_valve_counters_all_zero": signals.get("escape_valve_counters_all_zero", {}).get("passed"),
        "C_meas_final": final_compression.get("C_meas"),
        "C_required": drive.get("compression_required", {}).get("C_required"),
        "initial_kinetic_energy_erg": drive.get("initial_kinetic_energy_erg"),
        "outer_spectral_rms_l_final": final_compression.get("outer_spectral_rms_l"),
        "sampled_corner_j_rel_min": sampled.get("min_corner_j_rel"),
        "sampled_corner_j_rel_p0p1": sampled.get("p0p1_corner_j_rel"),
        "sampled_corner_j_rel_p1": sampled.get("p1_corner_j_rel"),
        "worst_cell": sampled.get("worst_cell"),
    }


def run_one(args: argparse.Namespace, case: GeoCase, label: str) -> dict[str, Any]:
    stem = (
        f"{label}_{compact_case_label(case)}_nr{args.nr}_nz{args.nz}"
        f"_seed{args.seed}"
    )
    run_id, outdir = lite.allocate_run_dir(Path(args.out_root), stem)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "run.log"
    env_overrides = active_env(args, outdir, case)
    env = os.environ.copy()
    env.update(env_overrides)
    start = time.monotonic()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                [args.tenryu_bin, "run", args.deck],
                env=env,
                stdout=log,
                stderr=log,
                timeout=args.wall_time_s,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            proc = subprocess.CompletedProcess(args=[args.tenryu_bin, "run", args.deck], returncode=-1)
    wall_time_s = time.monotonic() - start
    predicate = evaluate_predicate(
        exit_code=proc.returncode,
        timed_out=timed_out,
        outdir=outdir,
        log_path=log_path,
        args=args,
        case=case,
    )
    result = "ADMISSIBLE" if predicate["passed"] else "INVALID"
    if timed_out:
        result = "WALL_TIME_TIMEOUT"
    payload = {
        "schema_version": "tenryu.i1b_geo_2d_rz_run_predicate.v1",
        "run_id": run_id,
        "case_label": compact_case_label(case),
        "stage_id": STAGE_ID,
        "mode": MODE,
        "x_geo": x_geo(case),
        "x_geo_definition": "X_geo = C_star * l_eff * a0",
        "c_star": case.c_star,
        "a0": case.a0,
        "l_min": case.l_min,
        "l_max": case.l_max,
        "l_eff": case.l_eff,
        "v_pk_cms": case.v_pk_cms,
        "v_pk_km_s": velocity_km_s(case.v_pk_cms),
        "expected_t_end_s_if_deck_auto": expected_t_end_s(case),
        "deck_path": args.deck,
        "result": result,
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "wall_time_s": wall_time_s,
        "output_bytes": lite.output_bytes(outdir),
        "predicate": predicate,
        "env_overrides": env_overrides,
    }
    summary_path = outdir / "predicate_summary.json"
    write_json(summary_path, payload)
    return row_from_payload(payload, summary_path)


def characterization_status(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no_runs"
    admissible = [row for row in rows if bool(row.get("admissible"))]
    inadmissible = [row for row in rows if not bool(row.get("admissible"))]
    if not admissible:
        return "none_admissible"
    if not inadmissible:
        return "all_admissible"
    x_geom = max(float(row["x_geo"]) for row in admissible)
    if any(float(row["x_geo"]) > x_geom for row in inadmissible):
        return "bracketed"
    return "nonmonotone_unbracketed"


def row_matches_ci_gate(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        abs(float(row["c_star"]) - float(args.ci_c_star)) < 1.0e-15
        and abs(float(row["a0"]) - float(args.ci_a0)) < 1.0e-15
        and abs(float(row["v_pk_cms"]) - float(args.v_pk_cms)) < 1.0e-6
    )


def characterize(args: argparse.Namespace) -> dict[str, Any]:
    rows = [run_one(args, case, "ladder") for case in planned_cases(args)]
    admissible = [row for row in rows if bool(row.get("admissible"))]
    inadmissible = [row for row in rows if not bool(row.get("admissible"))]
    x_geom = max((float(row["x_geo"]) for row in admissible), default=None)
    x_gate_range = None
    x_gate = None
    overdrive_failure = None
    if x_geom is not None:
        x_gate_range = {"low": 0.7 * x_geom, "high": 0.8 * x_geom}
        x_gate = 0.75 * x_geom
        candidates = [
            row
            for row in inadmissible
            if float(row["x_geo"]) > x_geom and not row_matches_ci_gate(row, args)
        ]
        if candidates:
            overdrive_failure = min(candidates, key=lambda row: float(row["x_geo"]))
    status = characterization_status(rows)
    clean_result = all(bool(row.get("harness_clean")) for row in rows)
    summary = {
        "schema_version": "tenryu.i1b_geo_2d_rz_characterization.v1",
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "stage_id": STAGE_ID,
        "mode": MODE,
        "deck_path": args.deck,
        "tenryu_bin": args.tenryu_bin,
        "characterization_status": status,
        "clean_result": clean_result,
        "x_mapping": {
            "definition": "X_geo = C_star * l_eff * a0",
            "l_eff": args.l_eff,
            "spectral_rms_l": "reported per run from the fitted outer-interface modes as secondary diagnostic",
            "erg_equivalent": "N/A (geometric stress, not laser energy)",
        },
        "predicate_definition": {
            "geometry_source": MESH_QUALITY_GROUP,
            "min_corner_j_rel": f"> {lite.ADMISSIBILITY_FLOOR_REL:.1e}",
            "min_gauss_j_rel": f"> {lite.ADMISSIBILITY_FLOOR_REL:.1e}",
            "min_rz_volume_rel": "> 0.0",
            "negative_rz_volume_count_total": 0,
            "reached_t_end": True,
            "termination_reason": "t_end_reached",
            "final_t_margin": "final_t >= 0.999 * t_end",
            "escape_valve_counters_all_zero": True,
            "mesh_geometry_failures_observed_count": 0,
            "C_meas_final": ">= 0.9 * C_star",
            "initial_kinetic_energy_erg": f"> {INITIAL_E_KIN_FLOOR_ERG:.1e}",
            "missing_required_signal_policy": "fail_closed_INVALID",
        },
        "budget": {
            "nr": args.nr,
            "nz": args.nz,
            "v_pk_cms": args.v_pk_cms,
            "v_pk_km_s": velocity_km_s(args.v_pk_cms),
            "dt_initial_s": args.dt_initial_s,
            "dt_max_s": args.dt_max_s,
            "plot_every_steps": args.plot_every_steps,
            "history_every_steps": args.history_every_steps,
            "max_steps": args.max_steps,
            "t_end_s": args.t_end_s,
            "t_end_policy": "deck auto from R_OUTER/V_pk*(1-1/C_star) unless --t-end-s is set",
        },
        "ladder": {
            "c_star_grid": list(args.c_star_grid),
            "a0_grid": list(args.a0_grid),
            "run_count": len(rows),
        },
        "x_geom": x_geom,
        "x_gate_proposed": x_gate,
        "x_gate_proposed_factor": 0.75 if x_gate is not None else None,
        "x_gate_recommended_range": x_gate_range,
        "overdrive_failure_outside_ci": overdrive_failure,
        "runs": sorted(rows, key=lambda row: (float(row["x_geo"]), float(row["c_star"]), float(row["a0"]))),
    }
    write_json(Path(args.summary_json), summary)
    return summary


def ci(args: argparse.Namespace) -> dict[str, Any]:
    case = planned_cases(args)[0]
    row = run_one(args, case, "ci")
    summary = {
        "schema_version": "tenryu.i1b_geo_2d_rz_ci.v1",
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "stage_id": STAGE_ID,
        "mode": MODE,
        "deck_path": args.deck,
        "tenryu_bin": args.tenryu_bin,
        "ci_status": "passed" if bool(row.get("admissible")) else "failed",
        "clean_result": bool(row.get("harness_clean")) and bool(row.get("admissible")),
        "gate_point": {
            "name": "I1B-GEO-C8-A3e-3-V350"
            if abs(case.c_star - 8.0) < 1.0e-15 and abs(case.a0 - 3.0e-3) < 1.0e-15
            else compact_case_label(case),
            "x_geo": x_geo(case),
            "c_star": case.c_star,
            "a0": case.a0,
            "l_min": case.l_min,
            "l_max": case.l_max,
            "l_eff": case.l_eff,
            "v_pk_cms": case.v_pk_cms,
            "v_pk_km_s": velocity_km_s(case.v_pk_cms),
            "expected_t_end_s_if_deck_auto": expected_t_end_s(case),
        },
        "run": row,
    }
    write_json(Path(args.summary_json), summary)
    return summary


def dry_run_summary(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for case in planned_cases(args):
        planned_outdir = Path(args.out_root) / (
            f"planned_{compact_case_label(case)}_nr{args.nr}_nz{args.nz}_seed{args.seed}"
        )
        rows.append(
            {
                "case_label": compact_case_label(case),
                "x_geo": x_geo(case),
                "c_star": case.c_star,
                "a0": case.a0,
                "l_min": case.l_min,
                "l_max": case.l_max,
                "l_eff": case.l_eff,
                "v_pk_cms": case.v_pk_cms,
                "v_pk_km_s": velocity_km_s(case.v_pk_cms),
                "expected_t_end_s_if_deck_auto": expected_t_end_s(case),
                "erg_equivalent": None,
                "erg_equivalent_note": "N/A (geometric stress, not laser energy)",
                "planned_outdir": str(planned_outdir),
                "env_overrides": active_env(args, planned_outdir, case),
                "predicate_result": "not_evaluated_dry_run",
                "planned": True,
            }
        )
    return {
        "schema_version": "tenryu.i1b_geo_2d_rz_dry_run.v1",
        "mode": args.mode,
        "stage_id": STAGE_ID,
        "deck_path": args.deck,
        "x_mapping": {
            "definition": "X_geo = C_star * l_eff * a0",
            "l_eff": args.l_eff,
            "erg_equivalent": "N/A",
        },
        "budget": {
            "nr": args.nr,
            "nz": args.nz,
            "dt_initial_s": args.dt_initial_s,
            "dt_max_s": args.dt_max_s,
            "plot_every_steps": args.plot_every_steps,
            "history_every_steps": args.history_every_steps,
            "max_steps": args.max_steps,
            "t_end_s": args.t_end_s,
            "t_end_policy": "deck auto unless --t-end-s is set",
        },
        "planned_runs": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("characterize", "ci"), default="characterize")
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/2d_rz_i1_capsule.py")
    parser.add_argument("--out-root", default="./build/i1b_geo_2d_rz_runs")
    parser.add_argument("--summary-json", default="./build/i1b_geo_2d_rz_summary.json")
    parser.add_argument("--c-star-grid", type=parse_float_list, default=list(DEFAULT_C_STAR_GRID))
    parser.add_argument("--a0-grid", type=parse_float_list, default=list(DEFAULT_A0_GRID))
    parser.add_argument("--ci-c-star", type=float, default=DEFAULT_CI_C_STAR)
    parser.add_argument("--ci-a0", type=float, default=DEFAULT_CI_A0)
    parser.add_argument("--v-pk-cms", type=float, default=DEFAULT_V_PK_CMS)
    parser.add_argument("--l-min", type=int, default=DEFAULT_L_MIN)
    parser.add_argument("--l-max", type=int, default=DEFAULT_L_MAX)
    parser.add_argument("--l-eff", type=float, default=DEFAULT_L_EFF)
    parser.add_argument("--nr", type=int, default=DEFAULT_NR)
    parser.add_argument("--nz", type=int, default=DEFAULT_NZ)
    parser.add_argument("--rays-per-beam", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--t-end-s", type=float, default=None)
    parser.add_argument("--dt-initial-s", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--dt-max-s", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--plot-every-steps", type=int, default=DEFAULT_PLOT_EVERY_STEPS)
    parser.add_argument("--history-every-steps", type=int, default=DEFAULT_HISTORY_EVERY_STEPS)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--interface-bins", type=int, default=64)
    parser.add_argument("--verbosity", choices=("quiet", "normal", "verbose"), default="verbose")
    parser.add_argument("--wall-time-s", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="print planned runs without invoking TENRYU")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.nr <= 0 or args.nz <= 0:
        raise ValueError("--nr and --nz must be positive")
    if args.rays_per_beam <= 0:
        raise ValueError("--rays-per-beam must be positive")
    if args.v_pk_cms <= 0.0 or not math.isfinite(args.v_pk_cms):
        raise ValueError("--v-pk-cms must be finite and positive")
    if args.ci_c_star <= 1.0 or not math.isfinite(args.ci_c_star):
        raise ValueError("--ci-c-star must be finite and greater than 1")
    if args.ci_a0 <= 0.0 or not math.isfinite(args.ci_a0):
        raise ValueError("--ci-a0 must be finite and positive")
    if any(value <= 1.0 for value in args.c_star_grid):
        raise ValueError("--c-star-grid values must be greater than 1")
    if args.l_min < 0 or args.l_max < args.l_min:
        raise ValueError("--l-min/--l-max must satisfy 0 <= l_min <= l_max")
    if args.l_eff <= 0.0 or not math.isfinite(args.l_eff):
        raise ValueError("--l-eff must be finite and positive")
    if args.t_end_s is not None and (args.t_end_s <= 0.0 or not math.isfinite(args.t_end_s)):
        raise ValueError("--t-end-s must be finite and positive when set")
    if args.dt_initial_s <= 0.0 or args.dt_max_s <= 0.0:
        raise ValueError("time-step arguments must be positive")
    if args.plot_every_steps <= 0 or args.history_every_steps <= 0:
        raise ValueError("--plot-every-steps and --history-every-steps must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.interface_bins < args.l_max + 1:
        raise ValueError("--interface-bins must be at least --l-max + 1")
    if args.wall_time_s is not None and args.wall_time_s <= 0.0:
        raise ValueError("--wall-time-s must be positive when set")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if args.dry_run:
        print(json.dumps(dry_run_summary(args), indent=2, sort_keys=True))
        return 0
    summary = ci(args) if args.mode == "ci" else characterize(args)
    print(json.dumps(json_sanitize(summary), indent=2, sort_keys=True))
    return 0 if summary.get("clean_result") else 2


if __name__ == "__main__":
    raise SystemExit(main())
