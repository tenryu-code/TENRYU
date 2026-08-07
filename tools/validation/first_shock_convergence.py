"""TENRYU first-shock convergence analyzer (Phase 2cd Wave 2 deliverable).

Computes informational metrics for ICF first-shock convergence on TENRYU 2D RZ
HDF5 dumps. Locked spec from CC ⇄ Codex MCP plan-consensus thread
019e02d1-45c4-7a33-931f-a2c417a9184e (3 rounds, Q1-Q13 CONSENSUS), inheriting
Round 3 Q4 from earlier thread 019e0114-b868-7a20-bb05-2c1aa8ac792c.

Metrics per snapshot:
- core_rho_volume_weighted_mean / core_rho_p95_volume_weighted (5 µm core default)
- core_pressure_volume_weighted_mean
- converging_flow_frac (volume fraction with u·r̂_sph < 0, INSIDE sphere only)
- front_r_sph_axis_plus_cm / axis_minus_cm / equator_cm via signed-gradient peak
  with inward-shock validity gate: compression_ok AND (pressure_ok OR inflow_ok)
- anisotropy_A_shock = (max - min) / mean over valid-front rays
- convergence_fraction = 1 - mean(front_r_sph_valid_rays) / sphere_radius_cm

All metrics emit either a finite value OR an explicit {field}_unavailable_reason
string per Q11 harness-quality contract.

Trajectory mode adds summary record with rho_init from first snapshot,
core_compression_max, core_compression_t_frac_at_2x, etc.

CLI:
  python tools/validation/first_shock_convergence.py \\
    --results-dir DIR --run-id ID --output PATH.jsonl \\
    [--sphere-radius-cm 0.01] [--core-radius-cm 5e-4]

  python tools/validation/first_shock_convergence.py \\
    --single-snapshot DUMP.h5 --output PATH.jsonl [--rho-init G_PER_CC]

NOT a strict gate. All output is informational only (Q11).
"""

from __future__ import annotations

import argparse, json, math, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


CORE_RADIUS_DEFAULT_CM = 5e-4
SPHERE_RADIUS_DEFAULT_CM = 0.01
INFLOW_THRESHOLD_CM_S = 1.0e3
DELTA_RHO = 0.05
DELTA_P = 0.05
ANISOTROPY_R_FRONT_FRAC_THRESHOLD = 0.1
MIN_CORE_CELLS = 8
WINDOW_DENOM = 16
EPS = 1e-12


@dataclass(frozen=True)
class Snapshot:
    path: Path
    t: float
    step: int
    nr: int
    nz: int
    rho: np.ndarray
    Pe: np.ndarray
    Pi: np.ndarray
    vol: np.ndarray
    r_nodes: np.ndarray
    z_nodes: np.ndarray
    v_r_nodes: np.ndarray
    v_z_nodes: np.ndarray


def _require_dataset(h: h5py.File, name: str) -> h5py.Dataset:
    if name not in h:
        raise KeyError(f"required HDF5 dataset missing: {name}")
    obj = h[name]
    if not isinstance(obj, h5py.Dataset):
        raise KeyError(f"required HDF5 dataset path is not a dataset: {name}")
    return obj


def reshape_cells(arr: np.ndarray, nr: int, nz: int, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.size == nr * nz:
        return arr.reshape((nr, nz))
    if arr.shape == (nr, nz):
        return arr
    raise ValueError(f"{name}: expected ({nr},{nz}) or flat {nr*nz}, got {arr.shape}")


def reshape_nodes(arr: np.ndarray, nr: int, nz: int, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    target = (nr + 1, nz + 1)
    if arr.size == target[0] * target[1]:
        return arr.reshape(target)
    if arr.shape == target:
        return arr
    raise ValueError(f"{name}: expected {target} or flat {target[0]*target[1]}, got {arr.shape}")


def read_snapshot(path: Path) -> Snapshot:
    import h5py

    with h5py.File(path, "r") as h:
        required = {
            "hydro/rho": _require_dataset(h, "hydro/rho"),
            "hydro/Pe": _require_dataset(h, "hydro/Pe"),
            "hydro/Pi": _require_dataset(h, "hydro/Pi"),
            "hydro/vol": _require_dataset(h, "hydro/vol"),
            "mesh/x_r": _require_dataset(h, "mesh/x_r"),
            "mesh/x_z": _require_dataset(h, "mesh/x_z"),
            "mesh/v_r": _require_dataset(h, "mesh/v_r"),
            "mesh/v_z": _require_dataset(h, "mesh/v_z"),
        }
        nr = int(h.attrs.get("nr", 0)) or required["mesh/x_r"].shape[0] - 1
        nz = int(h.attrs.get("nz", 0)) or required["mesh/x_r"].shape[1] - 1
        if "time_state/t" in h:
            t = float(np.asarray(h["time_state/t"][()]).reshape(-1)[0])
        else:
            t = float(h.attrs.get("t", 0.0))
        if "time_state/step" in h:
            step = int(np.asarray(h["time_state/step"][()]).reshape(-1)[0])
        else:
            step = int(h.attrs.get("step", h.attrs.get("cycle", 0)))
        return Snapshot(
            path=path,
            t=t,
            step=step,
            nr=nr,
            nz=nz,
            rho=reshape_cells(required["hydro/rho"][()], nr, nz, "hydro/rho"),
            Pe=reshape_cells(required["hydro/Pe"][()], nr, nz, "hydro/Pe"),
            Pi=reshape_cells(required["hydro/Pi"][()], nr, nz, "hydro/Pi"),
            vol=reshape_cells(required["hydro/vol"][()], nr, nz, "hydro/vol"),
            r_nodes=reshape_nodes(required["mesh/x_r"][()], nr, nz, "mesh/x_r"),
            z_nodes=reshape_nodes(required["mesh/x_z"][()], nr, nz, "mesh/x_z"),
            v_r_nodes=reshape_nodes(required["mesh/v_r"][()], nr, nz, "mesh/v_r"),
            v_z_nodes=reshape_nodes(required["mesh/v_z"][()], nr, nz, "mesh/v_z"),
        )


def cell_centers(snap: Snapshot) -> tuple[np.ndarray, np.ndarray]:
    rn, zn = snap.r_nodes, snap.z_nodes
    rc = 0.25 * (rn[:-1, :-1] + rn[1:, :-1] + rn[:-1, 1:] + rn[1:, 1:])
    zc = 0.25 * (zn[:-1, :-1] + zn[1:, :-1] + zn[:-1, 1:] + zn[1:, 1:])
    return rc, zc


def cell_centered_velocity(snap: Snapshot) -> tuple[np.ndarray, np.ndarray]:
    vr_n, vz_n = snap.v_r_nodes, snap.v_z_nodes
    ur = 0.25 * (vr_n[:-1, :-1] + vr_n[1:, :-1] + vr_n[:-1, 1:] + vr_n[1:, 1:])
    uz = 0.25 * (vz_n[:-1, :-1] + vz_n[1:, :-1] + vz_n[:-1, 1:] + vz_n[1:, 1:])
    return ur, uz


def r_sph_cells(snap: Snapshot) -> np.ndarray:
    rc, zc = cell_centers(snap)
    return np.sqrt(rc * rc + zc * zc)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Volume-weighted quantile via cumulative-weight method.
    values, weights are 1D arrays of equal length. q in [0,1].
    Uses sorted cumulative weight; returns the value at the smallest cumulative weight >= q.
    """
    if values.size == 0:
        return float("nan")
    sort_idx = np.argsort(values)
    v_s = values[sort_idx]
    w_s = weights[sort_idx]
    total = float(np.sum(w_s))
    if total <= 0.0:
        return float("nan")
    cum = np.cumsum(w_s) / total
    idx = int(np.searchsorted(cum, q, side="left"))
    idx = max(0, min(idx, v_s.size - 1))
    return float(v_s[idx])


def core_metrics(snap: Snapshot, sphere_radius_cm: float, core_radius_cm: float) -> dict[str, Any]:
    r_sph = r_sph_cells(snap)
    mask = r_sph <= core_radius_cm
    n_core = int(np.sum(mask))
    if n_core < MIN_CORE_CELLS:
        return {
            "core_cell_count": n_core,
            "core_rho_volume_weighted_mean_g_per_cc": None,
            "core_rho_p95_volume_weighted_g_per_cc": None,
            "core_pressure_volume_weighted_mean_dyne_per_cm2": None,
            "core_metrics_unavailable_reason": "core_cell_count_below_threshold",
        }
    rho_core = snap.rho[mask].ravel()
    vol_core = snap.vol[mask].ravel()
    p_core = (snap.Pe + snap.Pi)[mask].ravel()
    total_vol = float(np.sum(vol_core))
    if total_vol <= 0.0:
        return {
            "core_cell_count": n_core,
            "core_rho_volume_weighted_mean_g_per_cc": None,
            "core_rho_p95_volume_weighted_g_per_cc": None,
            "core_pressure_volume_weighted_mean_dyne_per_cm2": None,
            "core_metrics_unavailable_reason": "core_total_volume_nonpositive",
        }
    rho_mean = float(np.average(rho_core, weights=vol_core))
    rho_p95 = weighted_quantile(rho_core, vol_core, 0.95)
    p_mean = float(np.average(p_core, weights=vol_core))
    return {
        "core_cell_count": n_core,
        "core_rho_volume_weighted_mean_g_per_cc": rho_mean,
        "core_rho_p95_volume_weighted_g_per_cc": rho_p95,
        "core_pressure_volume_weighted_mean_dyne_per_cm2": p_mean,
        "core_metrics_unavailable_reason": None,
    }


def converging_flow_frac(snap: Snapshot, sphere_radius_cm: float) -> dict[str, Any]:
    rc, zc = cell_centers(snap)
    ur, uz = cell_centered_velocity(snap)
    r_sph = np.sqrt(rc * rc + zc * zc)
    inside_sphere = r_sph <= sphere_radius_cm
    away_from_axis = r_sph > EPS
    safe = inside_sphere & away_from_axis
    if not np.any(safe):
        return {"converging_flow_frac": None, "converging_flow_unavailable_reason": "no_safe_cells_inside_sphere"}
    u_r_sph = (ur * rc + uz * zc) / np.maximum(r_sph, EPS)
    converging = (u_r_sph < 0.0) & safe
    vol_safe = float(np.sum(snap.vol[safe]))
    vol_inflow = float(np.sum(snap.vol[converging]))
    if vol_safe <= 0.0:
        return {"converging_flow_frac": None, "converging_flow_unavailable_reason": "zero_total_volume_inside_sphere"}
    return {"converging_flow_frac": vol_inflow / vol_safe, "converging_flow_unavailable_reason": None}


def find_front_along_ray(rho_ray: np.ndarray, p_ray: np.ndarray, u_r_sph_ray: np.ndarray, r_sph_ray: np.ndarray) -> dict[str, Any]:
    """Find inward first-shock front along a 1D ray.
    All inputs are 1D arrays of equal length, ALREADY sorted by increasing r_sph.
    Returns {"front_r_sph_cm": float|None, "unavailable_reason": str|None}.
    """
    n = rho_ray.size
    if n < 4:
        return {"front_r_sph_cm": None, "unavailable_reason": "ray_too_short"}
    dr = np.diff(r_sph_ray)
    drho = np.diff(rho_ray)
    valid_dr = dr > EPS
    if not np.any(valid_dr):
        return {"front_r_sph_cm": None, "unavailable_reason": "ray_r_sph_degenerate"}
    grad = np.where(valid_dr, drho / np.where(valid_dr, dr, 1.0), -np.inf)
    positive_idx = np.where(grad > 0)[0]
    if positive_idx.size == 0:
        return {"front_r_sph_cm": None, "unavailable_reason": "no_positive_d_rho_d_r_sph"}
    i_star = int(positive_idx[np.argmax(grad[positive_idx])])
    front_r_sph = 0.5 * (float(r_sph_ray[i_star]) + float(r_sph_ray[i_star + 1]))

    W = max(2, n // WINDOW_DENOM)
    inner_lo = max(0, i_star - W)
    inner_hi = i_star
    outer_lo = i_star + 1
    outer_hi = min(n, i_star + 1 + W)
    if (inner_hi - inner_lo) < 2 or (outer_hi - outer_lo) < 2:
        return {"front_r_sph_cm": None, "unavailable_reason": "window_too_short"}

    inner_rho = rho_ray[inner_lo:inner_hi]
    outer_rho = rho_ray[outer_lo:outer_hi]
    inner_p = p_ray[inner_lo:inner_hi]
    outer_p = p_ray[outer_lo:outer_hi]
    inner_u = u_r_sph_ray[inner_lo:inner_hi]
    outer_u = u_r_sph_ray[outer_lo:outer_hi]

    inner_rho_mean = float(np.mean(inner_rho))
    outer_rho_mean = float(np.mean(outer_rho))
    inner_p_mean = float(np.mean(inner_p))
    outer_p_mean = float(np.mean(outer_p))
    combined_u_mean = float(np.mean(np.concatenate([inner_u, outer_u])))

    compression_ok = outer_rho_mean > (1.0 + DELTA_RHO) * inner_rho_mean
    pressure_ok = outer_p_mean > (1.0 + DELTA_P) * inner_p_mean
    inflow_ok = combined_u_mean < -INFLOW_THRESHOLD_CM_S

    if not compression_ok:
        return {"front_r_sph_cm": None, "unavailable_reason": "failed_compression"}
    if not (pressure_ok or inflow_ok):
        return {"front_r_sph_cm": None, "unavailable_reason": "failed_pressure_and_inflow"}
    return {"front_r_sph_cm": float(front_r_sph), "unavailable_reason": None}


def extract_ray(snap: Snapshot, ur: np.ndarray, uz: np.ndarray, ray: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract 1D arrays along a ray, sorted by increasing r_sph.
    Returns (rho, p_total, u_r_sph, r_sph) all 1D, all length matching cells along the ray.
    """
    rc, zc = cell_centers(snap)
    if ray in ("axis_plus", "axis_minus"):
        idx_i = 0
        zc_row = zc[idx_i, :]
        if ray == "axis_plus":
            mask = zc_row > 0.0
        else:
            mask = zc_row < 0.0
        if not np.any(mask):
            return None
        rho = snap.rho[idx_i, mask]
        p = (snap.Pe + snap.Pi)[idx_i, mask]
        ur_c = ur[idx_i, mask]
        uz_c = uz[idx_i, mask]
        rc_c = rc[idx_i, mask]
        zc_c = zc_row[mask]
        rsph = np.sqrt(rc_c * rc_c + zc_c * zc_c)
        u_r_sph = (ur_c * rc_c + uz_c * zc_c) / np.maximum(rsph, EPS)
    elif ray == "equator":
        zc_col = np.mean(zc, axis=0)
        j_mid = int(np.argmin(np.abs(zc_col)))
        rho = snap.rho[:, j_mid]
        p = (snap.Pe + snap.Pi)[:, j_mid]
        ur_c = ur[:, j_mid]
        uz_c = uz[:, j_mid]
        rc_c = rc[:, j_mid]
        zc_c = zc[:, j_mid]
        rsph = np.sqrt(rc_c * rc_c + zc_c * zc_c)
        u_r_sph = (ur_c * rc_c + uz_c * zc_c) / np.maximum(rsph, EPS)
    else:
        raise ValueError(f"unknown ray: {ray}")

    sort_idx = np.argsort(rsph)
    return rho[sort_idx], p[sort_idx], u_r_sph[sort_idx], rsph[sort_idx]


def compute_metrics_for_snapshot(
    snap: Snapshot, *, sphere_radius_cm: float, core_radius_cm: float, rho_init: float | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {"t": float(snap.t), "step": int(snap.step), "snapshot_path": str(snap.path)}
    out.update(core_metrics(snap, sphere_radius_cm, core_radius_cm))
    out.update(converging_flow_frac(snap, sphere_radius_cm))
    ur, uz = cell_centered_velocity(snap)
    front_r_per_ray: dict[str, float] = {}
    for ray in ("axis_plus", "axis_minus", "equator"):
        extracted = extract_ray(snap, ur, uz, ray)
        if extracted is None:
            out[f"front_r_sph_{ray}_cm"] = None
            out[f"front_r_sph_{ray}_unavailable_reason"] = "ray_no_cells"
            continue
        rho_ray, p_ray, u_r_sph_ray, r_sph_ray = extracted
        result = find_front_along_ray(rho_ray, p_ray, u_r_sph_ray, r_sph_ray)
        out[f"front_r_sph_{ray}_cm"] = result["front_r_sph_cm"]
        out[f"front_r_sph_{ray}_unavailable_reason"] = result["unavailable_reason"]
        if result["front_r_sph_cm"] is not None:
            front_r_per_ray[ray] = result["front_r_sph_cm"]

    if not front_r_per_ray:
        out["convergence_fraction"] = None
        out["convergence_fraction_unavailable_reason"] = "no_validated_shock_front_on_any_ray"
        out["anisotropy_A_shock"] = None
        out["anisotropy_unavailable_reason"] = "no_validated_shock_front_on_any_ray"
    else:
        r_values = np.array(list(front_r_per_ray.values()), dtype=float)
        mean_r = float(np.mean(r_values))
        out["convergence_fraction"] = float(1.0 - mean_r / sphere_radius_cm)
        out["convergence_fraction_unavailable_reason"] = None
        if mean_r < ANISOTROPY_R_FRONT_FRAC_THRESHOLD * sphere_radius_cm:
            out["anisotropy_A_shock"] = None
            out["anisotropy_unavailable_reason"] = "mean_r_front_below_threshold"
        elif r_values.size < 2:
            out["anisotropy_A_shock"] = None
            out["anisotropy_unavailable_reason"] = "single_valid_ray"
        else:
            out["anisotropy_A_shock"] = float((np.max(r_values) - np.min(r_values)) / mean_r)
            out["anisotropy_unavailable_reason"] = None
        out["valid_front_rays"] = list(front_r_per_ray.keys())

    rho_mean = out.get("core_rho_volume_weighted_mean_g_per_cc")
    rho_p95 = out.get("core_rho_p95_volume_weighted_g_per_cc")
    if rho_init is not None and rho_init > 0.0 and rho_mean is not None:
        out["core_compression"] = float(rho_mean / rho_init)
        out["core_compression_p95"] = float(rho_p95 / rho_init) if rho_p95 is not None else None
        out["core_compression_unavailable_reason"] = None
    else:
        out["core_compression"] = None
        out["core_compression_p95"] = None
        out["core_compression_unavailable_reason"] = (
            "rho_init_unavailable_in_single_snapshot" if rho_init is None else "core_metrics_unavailable_or_zero_init"
        )

    return out


def compute_trajectory(
    snapshots: list[Snapshot], *, sphere_radius_cm: float, core_radius_cm: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not snapshots:
        return [], {"summary": True, "n_snapshots": 0, "unavailable_reason": "no_snapshots"}
    snapshots = sorted(snapshots, key=lambda s: s.t)
    initial = compute_metrics_for_snapshot(snapshots[0], sphere_radius_cm=sphere_radius_cm, core_radius_cm=core_radius_cm)
    rho_init = initial.get("core_rho_volume_weighted_mean_g_per_cc")
    rho_init_source_t_s = float(snapshots[0].t)
    rows = []
    for snap in snapshots:
        row = compute_metrics_for_snapshot(snap, sphere_radius_cm=sphere_radius_cm, core_radius_cm=core_radius_cm, rho_init=rho_init)
        rows.append(row)
    cf_finite = [r["convergence_fraction"] for r in rows if r["convergence_fraction"] is not None]
    cc_finite = [r["core_compression"] for r in rows if r["core_compression"] is not None]
    aniso_finite = [r["anisotropy_A_shock"] for r in rows if r["anisotropy_A_shock"] is not None]
    cff_finite = [r["converging_flow_frac"] for r in rows if r["converging_flow_frac"] is not None]
    t_end = max(s.t for s in snapshots) or 1.0
    t_frac_2x = None
    for r in rows:
        cc = r.get("core_compression")
        if cc is not None and cc >= 2.0:
            t_frac_2x = float(r["t"]) / t_end if t_end > 0 else None
            break
    cc_at_t_end = rows[-1].get("core_compression")
    summary = {
        "summary": True,
        "n_snapshots": len(rows),
        "t_min_s": float(rows[0]["t"]),
        "t_max_s": float(rows[-1]["t"]),
        "sphere_radius_cm": sphere_radius_cm,
        "core_radius_cm": core_radius_cm,
        "rho_init_source_t_s": rho_init_source_t_s,
        "rho_init_g_per_cc": rho_init,
        "convergence_fraction_max": max(cf_finite) if cf_finite else None,
        "core_compression_max": max(cc_finite) if cc_finite else None,
        "core_compression_t_frac_at_2x": t_frac_2x,
        "core_compression_gate_achievement_at_t_end": cc_at_t_end,
        "anisotropy_A_shock_max": max(aniso_finite) if aniso_finite else None,
        "converging_flow_frac_max": max(cff_finite) if cff_finite else None,
        "n_snapshots_with_validated_front": sum(1 for r in rows if r.get("convergence_fraction") is not None),
    }
    return rows, summary


def discover_snapshot_files(results_dir: Path, run_id: str) -> list[Path]:
    """Mirror run_i1.py / run_l2.py logic: numbered _XXXX.h5 files, sorted by index."""
    files = []
    for p in sorted(results_dir.glob(f"{run_id}_*.h5")):
        stem = p.stem.removeprefix(f"{run_id}_")
        if stem.isdigit():
            files.append((int(stem), p))
    if not files:
        for p in sorted(results_dir.glob("*_*.h5")):
            tail = p.stem.rsplit("_", 1)[-1]
            if tail.isdigit():
                files.append((int(tail), p))
    return [p for _, p in sorted(files)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TENRYU first-shock convergence analyzer (Phase 2cd Wave 2)")
    parser.add_argument("--results-dir", type=Path, help="Directory of HDF5 snapshots")
    parser.add_argument("--run-id", type=str, help="Run id prefix for snapshot filenames")
    parser.add_argument("--single-snapshot", type=Path, help="Process a single snapshot instead of trajectory")
    parser.add_argument("--sphere-radius-cm", type=float, default=SPHERE_RADIUS_DEFAULT_CM)
    parser.add_argument("--core-radius-cm", type=float, default=CORE_RADIUS_DEFAULT_CM)
    parser.add_argument("--rho-init", type=float, default=None, help="Required for single-snapshot mode if compression ratios desired")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path (no default; will not overwrite without explicit path)")
    args = parser.parse_args(argv)

    sphere_source = "cli" if abs(args.sphere_radius_cm - SPHERE_RADIUS_DEFAULT_CM) > 0.0 else "default"

    if args.single_snapshot is not None:
        snap = read_snapshot(args.single_snapshot)
        row = compute_metrics_for_snapshot(snap, sphere_radius_cm=args.sphere_radius_cm, core_radius_cm=args.core_radius_cm, rho_init=args.rho_init)
        row["sphere_radius_cm"] = args.sphere_radius_cm
        row["sphere_radius_source"] = sphere_source
        row["mode"] = "single_snapshot"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=False) + "\n")
        return 0

    if args.results_dir is None or args.run_id is None:
        parser.error("--results-dir and --run-id required (or use --single-snapshot)")
    files = discover_snapshot_files(args.results_dir, args.run_id)
    if not files:
        parser.error(f"no numbered HDF5 snapshots found in {args.results_dir} for run_id {args.run_id}")
    snapshots = [read_snapshot(p) for p in files]
    rows, summary = compute_trajectory(snapshots, sphere_radius_cm=args.sphere_radius_cm, core_radius_cm=args.core_radius_cm)
    summary["sphere_radius_source"] = sphere_source
    summary["run_id"] = args.run_id
    summary["results_dir"] = str(args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=False) + "\n")
        f.write(json.dumps(summary, sort_keys=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
