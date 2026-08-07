#!/usr/bin/env python3
"""Re-process existing H1/H2 HDF5 plots to compute hardened metrics.

P3.F (H1): rz_inactive metric with fixed-cm mask + L1, L2, max norms + max-loc plot.
P3.6 (H2): alternate shock detector (density-crossing) + interval preshock L1.

Does NOT re-run simulations. Writes diagnostic supplements to:
  tmp/diagnostics/h1_metric_hardening/results.json
  tmp/diagnostics/h2_metric_hardening/results.json
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import h5py
import numpy as np


def discover_results_dir(outdir: Path) -> Path:
    def has_numbered(d: Path) -> bool:
        return any(len(p.stem) >= 5 and p.stem[-4:].isdigit() and p.stem[-5] == "_" for p in d.glob("*.h5"))
    rd = outdir / "results"
    if has_numbered(rd):
        return rd
    candidates = []
    for path in outdir.parent.glob(f"{outdir.name}_*"):
        cd = path / "results"
        if has_numbered(cd):
            candidates.append(cd)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.parent.stat().st_mtime)


def numbered_plots(results_dir: Path, name: str) -> list[Path]:
    return sorted(
        (p for p in results_dir.glob(f"{name}_*.h5") if p.stem.removeprefix(f"{name}_").isdigit()),
        key=lambda p: int(p.stem.removeprefix(f"{name}_")),
    )


def harden_h1(run_id: str, mesh_pair: tuple[int, int], runs_root: Path) -> dict:
    nr, nz = mesh_pair
    outdir = runs_root / run_id
    rd = discover_results_dir(outdir)
    if rd is None:
        return {"run_id": run_id, "status": "no_results_dir"}
    case_name = f"2d_rz_h1_sod_planar_nr{nr}_nz{nz}_seed{run_id.split('_seed')[-1]}_C21p5_cfl0p1"
    plots = numbered_plots(rd, case_name)
    if not plots:
        return {"run_id": run_id, "status": "no_plots"}
    with h5py.File(plots[-1], "r") as h:
        rho = np.asarray(h["hydro/rho"]).reshape(nr, nz)
        x_z = np.asarray(h["mesh/x_z"]).reshape(nr+1, nz+1)
        t = float(h["time_state/t"][()])

    z_cell = 0.25 * (x_z[:-1,:-1] + x_z[1:,:-1] + x_z[:-1,1:] + x_z[1:,1:])
    z_axis = z_cell[0, :]

    # Cylindrical-volume-weighted radial mean (Codex-recommended)
    # V_ij ∝ R_{i+1}^2 - R_i^2 ; for uniform R-grid with r_min=0, r_max=R_OUT:
    R_OUT = 0.1  # r_max per H1 deck
    r_edges = np.linspace(0, R_OUT, nr + 1)
    V_r = np.pi * (r_edges[1:]**2 - r_edges[:-1]**2)  # per-i radial volume weight
    V_total_r = V_r.sum()
    rho_bar_z_volweighted = (rho * V_r[:, None]).sum(axis=0) / V_total_r

    # rho deviation field
    diff = rho - rho_bar_z_volweighted[None, :]
    scale = float(np.max(np.abs(rho_bar_z_volweighted)))

    # Fixed-cm mask: exclude regions within 0.04 cm of largest gradients
    grad = np.abs(np.gradient(rho_bar_z_volweighted))
    # Find shock and contact (top 2 peaks)
    peak_indices = []
    work = grad.copy()
    for _ in range(2):
        if work.max() <= 0:
            break
        idx = int(np.argmax(work))
        peak_indices.append(idx)
        # zero out 0.04 cm around this peak
        z_peak = z_axis[idx]
        in_band = np.abs(z_axis - z_peak) <= 0.04
        work[in_band] = 0.0
    z_mask = np.ones(nz, dtype=bool)
    for idx in peak_indices:
        z_peak = z_axis[idx]
        in_band = np.abs(z_axis - z_peak) <= 0.04
        z_mask[in_band] = False
    # Also exclude 5 cells from each Z boundary
    edge = 5
    z_mask[:edge] = False
    z_mask[-edge:] = False

    if not z_mask.any():
        z_mask[:] = True

    diff_masked = diff[:, z_mask] / max(scale, 1.0)
    abs_diff = np.abs(diff_masked)
    cell_volumes_per_zcol = V_r  # per-i (not per-cell, since uniform Z)
    # weight matrix:
    weights = np.broadcast_to(V_r[:, None], abs_diff.shape)
    Vsum = weights.sum()
    L1 = float((abs_diff * weights).sum() / Vsum)
    L2 = float(np.sqrt(((abs_diff**2) * weights).sum() / Vsum))
    Linf = float(abs_diff.max())
    max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    j_indices = np.where(z_mask)[0]
    R_max = (max_idx[0] + 0.5) * R_OUT / nr  # cell center R
    Z_max = float(z_axis[j_indices[max_idx[1]]])

    return {
        "run_id": run_id,
        "status": "OK",
        "nr": nr, "nz": nz,
        "t_final_s": t,
        "rz_inactive_max_norm_volweighted": Linf,
        "rz_inactive_L1_volweighted": L1,
        "rz_inactive_L2_volweighted": L2,
        "max_at_R_cm": R_max,
        "max_at_Z_cm": Z_max,
        "shock_contact_peak_z_cm": [float(z_axis[i]) for i in peak_indices],
    }


def harden_h2(run_id: str, mesh_pair: tuple[int, int], runs_root: Path) -> dict:
    nr, nz = mesh_pair
    outdir = runs_root / run_id
    rd = discover_results_dir(outdir)
    if rd is None:
        return {"run_id": run_id, "status": "no_results_dir"}
    case_name_match = re.match(r"h2_(nr\d+_nz\d+)_(C2[\dp]+)_(cfl[\dp]+)_(ale\d+)_(seed\d+)", run_id)
    if not case_name_match:
        return {"run_id": run_id, "status": f"could_not_parse_run_id"}
    nr_nz, c2, cfl, ale, seed = case_name_match.groups()
    case_name = f"2d_rz_h2_noh_cyl_{nr_nz}_{seed}_{c2}_{cfl}_{ale}"
    plots = numbered_plots(rd, case_name)
    if not plots:
        return {"run_id": run_id, "status": "no_plots", "expected_name": case_name}
    with h5py.File(plots[-1], "r") as h:
        rho = np.asarray(h["hydro/rho"]).reshape(nr, nz)
        x_r = np.asarray(h["mesh/x_r"]).reshape(nr+1, nz+1)
        t = float(h["time_state/t"][()])
    r_cell = 0.25 * (x_r[:-1,:-1] + x_r[1:,:-1] + x_r[:-1,1:] + x_r[1:,1:])
    r_axis_radial = r_cell.mean(axis=1)
    rho_radial = rho.mean(axis=1)

    V_SCALE = 1e6
    R_s_analytic = (V_SCALE/3.0) * t

    # Detector A: max |drho/dr|
    grad = np.abs(np.gradient(rho_radial, r_axis_radial))
    mask = (r_axis_radial >= 0.05) & (r_axis_radial <= 0.5)
    R_s_grad = float(r_axis_radial[np.where(mask)[0][int(np.argmax(grad[mask]))]])

    # Detector B: density crossing at half of (rho_post + rho_pre), i.e. (16+rho_pre_at_R)/2
    # rho_pre at the shock face approximately = rho_0 * (1 + u0*t/R_s), but use observed pre-shock far from shock
    # Take rho at R = 1.2 * R_s_analytic if inside mesh, else use outermost cell rho
    target_R_pre = 1.2 * R_s_analytic
    if target_R_pre < r_axis_radial.max():
        idx_pre = int(np.argmin(np.abs(r_axis_radial - target_R_pre)))
        rho_pre = rho_radial[idx_pre]
    else:
        rho_pre = rho_radial[-1]
    rho_post_target = (16.0 + rho_pre) / 2.0
    # find first R (going outward from axis) where rho descends below rho_post_target
    interior = (r_axis_radial >= 0.02) & (r_axis_radial <= r_axis_radial.max())
    R_s_cross = None
    for i in range(len(r_axis_radial) - 1):
        if not interior[i] or not interior[i+1]:
            continue
        if rho_radial[i] >= rho_post_target and rho_radial[i+1] < rho_post_target:
            # linear interpolation
            f = (rho_radial[i] - rho_post_target) / (rho_radial[i] - rho_radial[i+1])
            R_s_cross = float(r_axis_radial[i] + f * (r_axis_radial[i+1] - r_axis_radial[i]))
            break

    dR = (1.0 - 0.0) / nr
    err_grad_cells = abs(R_s_grad - R_s_analytic) / dR
    err_cross_cells = abs(R_s_cross - R_s_analytic) / dR if R_s_cross else None

    # Interval preshock metric: L1 over [R_s + 8*dR, R_outer - 8*dR]
    R_lo_pre = R_s_analytic + 8 * dR
    R_hi_pre = r_axis_radial.max() - 8 * dR
    pre_mask = (r_axis_radial >= R_lo_pre) & (r_axis_radial <= R_hi_pre)
    if pre_mask.any():
        rho_analytic_pre = 1.0 * (1.0 + V_SCALE * t / r_axis_radial[pre_mask])
        L1_pre = float(np.abs(rho_radial[pre_mask] - rho_analytic_pre).mean() / np.maximum(np.abs(rho_analytic_pre).mean(), 1e-30) * 100.0)
    else:
        L1_pre = None

    return {
        "run_id": run_id,
        "status": "OK",
        "nr": nr, "nz": nz,
        "t_final_s": t,
        "shock_radius_analytic_cm": R_s_analytic,
        "shock_radius_grad_cm": R_s_grad,
        "shock_radius_cross_cm": R_s_cross,
        "shock_err_cells_grad": err_grad_cells,
        "shock_err_cells_cross": err_cross_cells,
        "shock_detector_consistency_cells": abs(R_s_grad - R_s_cross) / dR if R_s_cross else None,
        "preshock_L1_pct_interval": L1_pre,
        "preshock_interval_R_lo_cm": R_lo_pre,
        "preshock_interval_R_hi_cm": R_hi_pre,
    }


def main():
    out_h1 = Path("./tmp/diagnostics/h1_metric_hardening")
    out_h1.mkdir(parents=True, exist_ok=True)
    out_h2 = Path("./tmp/diagnostics/h2_metric_hardening")
    out_h2.mkdir(parents=True, exist_ok=True)

    h1_runs_root = Path("./build/h1_runs")
    h1_results = []
    for run_id, mesh in [
        ("h1_nr64_nz256_C21p5_cfl0p1_seed12345", (64, 256)),
        ("h1_nr128_nz512_C21p5_cfl0p1_seed12345", (128, 512)),
    ]:
        h1_results.append(harden_h1(run_id, mesh, h1_runs_root))
    (out_h1 / "results.json").write_text(json.dumps(h1_results, indent=2) + "\n")
    print("=== H1 metric hardening (volume-weighted, fixed-cm mask, L1/L2/Linf) ===")
    for r in h1_results:
        if r.get("status") == "OK":
            print(f"  {r['run_id']}:")
            print(f"    Linf = {r['rz_inactive_max_norm_volweighted']:.4e}  L1 = {r['rz_inactive_L1_volweighted']:.4e}  L2 = {r['rz_inactive_L2_volweighted']:.4e}")
            print(f"    max-loc: R={r['max_at_R_cm']:.4f} cm, Z={r['max_at_Z_cm']:.4f} cm")
        else:
            print(f"  {r['run_id']}: {r.get('status')}")

    h2_runs_root = Path("./build/h2_runs")
    h2_results = []
    for run_id, mesh in [
        ("h2_nr128_nz16_C21p5_cfl0p1_ale0_seed12345", (128, 16)),
        ("h2_nr256_nz16_C21p5_cfl0p1_ale0_seed12345", (256, 16)),
    ]:
        h2_results.append(harden_h2(run_id, mesh, h2_runs_root))
    (out_h2 / "results.json").write_text(json.dumps(h2_results, indent=2) + "\n")
    print("\n=== H2 metric hardening (alternate shock detector, interval preshock) ===")
    for r in h2_results:
        if r.get("status") == "OK":
            print(f"  {r['run_id']}:")
            print(f"    R_s analytic = {r['shock_radius_analytic_cm']:.5f} cm")
            print(f"    R_s grad-detector  = {r['shock_radius_grad_cm']:.5f} cm  ({r['shock_err_cells_grad']:.3f} cells err)")
            cross_str = f"{r['shock_radius_cross_cm']:.5f} cm  ({r['shock_err_cells_cross']:.3f} cells err)" if r.get('shock_radius_cross_cm') else "None"
            print(f"    R_s cross-detector = {cross_str}")
            print(f"    detector agreement: {r.get('shock_detector_consistency_cells', 'N/A')} cells")
            print(f"    preshock L1 (interval [{r['preshock_interval_R_lo_cm']:.3f}, {r['preshock_interval_R_hi_cm']:.3f}] cm): {r.get('preshock_L1_pct_interval', 'N/A')}%")
        else:
            print(f"  {r['run_id']}: {r.get('status')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
