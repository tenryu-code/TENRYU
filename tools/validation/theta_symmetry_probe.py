#!/usr/bin/env python3
"""Theta-symmetry probe for polar multiblock snapshots.

For each selected snapshot, reports the median over shell rows of
theta-RMS(rho)/mean (the watercolor metric) and the dm/dOmega Legendre
amplitudes (deformation-proof mass channel). The spherical (a0)
acceptance target is the LSB noise band (~1e-10 class, the 2026-08-26
estimator-band baseline); percent-level values indicate maintenance
contamination.

Usage:
  theta_symmetry_probe.py --results <dir> [--count 8] [--rows 20:50]
"""

from __future__ import annotations

import argparse
import glob
import os

import h5py
import numpy as np
from numpy.polynomial.legendre import legvander


def snapshot_time(path: str) -> float:
    with h5py.File(path, "r") as f:
        return float(np.asarray(f["time_state/t"][()]).reshape(-1)[0])


def probe(path: str, row_lo: int, row_hi: int):
    with h5py.File(path, "r") as f:
        t = float(np.asarray(f["time_state/t"][()]).reshape(-1)[0])
        rho = np.asarray(f["hydro/rho"][()], float).reshape(-1)
        mass = np.asarray(f["hydro/mass"][()], float).reshape(-1)
        xr = np.asarray(f["mesh/x_r"][()], float).reshape(-1)
        xz = np.asarray(f["mesh/x_z"][()], float).reshape(-1)
        off = np.asarray(f["mesh/topology/v3/cell_node_csr_offsets"][()])
        idx = np.asarray(f["mesh/topology/v3/cell_node_csr_indices"][()])
        bcb = np.asarray(f["mesh/topology/v3/block_cell_begin"][()])
        bcc = np.asarray(f["mesh/topology/v3/block_cell_count"][()])
        bni = np.asarray(f["mesh/topology/v3/block_n_i_cells"][()])
        bnj = np.asarray(f["mesh/topology/v3/block_n_j_cells"][()])
        roles = np.asarray(f["mesh/topology/v3/block_role"][()])

    # shell block = role 5
    shell = int(np.where(roles == 5)[0][0])
    nrows, ncols = int(bni[shell]), int(bnj[shell])
    sh = rho[bcb[shell]:bcb[shell] + bcc[shell]].reshape(nrows, ncols)
    r_hi = min(row_hi, nrows)
    vals = [sh[r].std() / max(sh[r].mean(), 1e-300)
            for r in range(row_lo, r_hi)]
    rms = float(np.median(vals)) if vals else float("nan")

    # high-pass split: physical drive modes live at low theta harmonics;
    # watercolor is grid-scale column noise. Per row, project rho(theta)
    # onto the first n_low cosine harmonics and report the residual RMS.
    n_low = 8
    hp_vals = []
    k = np.arange(ncols)
    basis = [np.ones(ncols)]
    for m in range(1, n_low + 1):
        basis.append(np.cos(np.pi * m * (k + 0.5) / ncols))
    B = np.vstack(basis).T
    BtB_inv_Bt = np.linalg.pinv(B)
    for r in range(row_lo, r_hi):
        row = sh[r]
        resid = row - B @ (BtB_inv_Bt @ row)
        hp_vals.append(resid.std() / max(row.mean(), 1e-300))
    hp = float(np.median(hp_vals)) if hp_vals else float("nan")

    # equatorial parity split: mirror column j <-> ncols-1-j (uniform
    # theta columns). For an even (z-symmetric) drive the odd part is
    # pure numerical contamination.
    od_vals = []
    for r in range(row_lo, r_hi):
        row = sh[r]
        odd = 0.5 * (row - row[::-1])
        od_vals.append(np.sqrt((odd * odd).mean()) / max(row.mean(), 1e-300))
    od = float(np.median(od_vals)) if od_vals else float("nan")

    # dm/dOmega over active blocks (roles 5, 6, 1)
    dm = np.zeros(ncols)
    th_sum = np.zeros(ncols)
    th_n = np.zeros(ncols)
    for b in range(len(roles)):
        if roles[b] not in (5, 6, 1):
            continue
        for c in range(bcb[b], bcb[b] + bcc[b]):
            col = (c - bcb[b]) % ncols
            nd = idx[off[c]:off[c + 1]]
            nd = nd[nd >= 0]
            r = xr[nd].mean()
            z = xz[nd].mean()
            dm[col] += mass[c]
            th_sum[col] += np.arctan2(max(r, 0.0), z)
            th_n[col] += 1
    th = th_sum / np.maximum(th_n, 1)
    mu = np.cos(th)
    order = np.argsort(mu)
    mu = mu[order]
    q = dm[order]
    w = np.gradient(mu)
    q = q / np.maximum(np.abs(w), 1e-300)
    V = legvander(mu, 8)
    a = np.array([(2 * l + 1) / 2 * np.sum(w * q * V[:, l])
                  for l in range(9)])
    rel = np.abs(a / a[0]) if a[0] != 0 else np.full(9, np.nan)
    return t, rms, hp, od, rel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--rows", default="20:50")
    args = ap.parse_args()

    row_lo, row_hi = (int(x) for x in args.rows.split(":"))
    paths = sorted(glob.glob(os.path.join(args.results, "*_0*.h5")))
    paths = [p for p in paths if "history" not in p]
    if not paths:
        print("no snapshots found")
        return 1
    sel = [paths[i] for i in
           np.linspace(0, len(paths) - 1, args.count, dtype=int)]
    for p in sel:
        try:
            t, rms, hp, od, rel = probe(p, row_lo, row_hi)
        except Exception as err:  # noqa: BLE001 - diagnostic tool
            print(f"{os.path.basename(p)}: unreadable ({err})")
            continue
        print("t_ps=%8.2f  thetaRMS=%.3e  hiRMS=%.3e  oddRMS=%.3e  "
              "|a_l/a0| l1=%.1e l2=%.1e l3=%.1e l4=%.1e"
              % (t * 1e12, rms, hp, od,
                 rel[1], rel[2], rel[3], rel[4]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
