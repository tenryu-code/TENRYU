#!/usr/bin/env python3
"""Per-arm mini-gate report for the v2 maintenance battery.

Pages: (1) density maps at selected times (full-page Agg raster to avoid
the mixed-renderer chunk drop), (2) theta-RMS / high-pass RMS time
series against the spherical LSB band, (3) dm/dOmega Legendre amplitude
evolution, (4) run summary text.

Usage:
  make_minigate_v2_report.py --results build/mini_a0/results \
      --out mini_a0_report.pdf --title "mini a0 (v2)" \
      [--death-ps 298.3] [--note "..."]
"""

from __future__ import annotations

import argparse
import glob
import io
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection

import theta_symmetry_probe as probe_mod


def load_mesh_rho(path):
    with h5py.File(path, "r") as f:
        t = float(np.asarray(f["time_state/t"][()]).reshape(-1)[0])
        rho = np.asarray(f["hydro/rho"][()], float).reshape(-1)
        xr = np.asarray(f["mesh/x_r"][()], float).reshape(-1)
        xz = np.asarray(f["mesh/x_z"][()], float).reshape(-1)
        off = np.asarray(f["mesh/topology/v3/cell_node_csr_offsets"][()])
        idx = np.asarray(f["mesh/topology/v3/cell_node_csr_indices"][()])
    return t, rho, xr, xz, off, idx


def field_page(pdf, paths, title):
    fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.0))
    for ax, path in zip(axes.ravel(), paths):
        t, rho, xr, xz, off, idx = load_mesh_rho(path)
        polys = []
        vals = []
        for c in range(len(off) - 1):
            nd = idx[off[c]:off[c + 1]]
            nd = nd[nd >= 0]
            if len(nd) < 3:
                continue
            polys.append(np.column_stack([xz[nd], xr[nd]]))
            vals.append(rho[c])
        vals = np.array(vals)
        lo = np.percentile(vals[vals > 0], 1) if (vals > 0).any() else 1e-6
        pc = PolyCollection(
            polys, array=np.log10(np.maximum(vals, lo)),
            cmap="viridis", edgecolors="none", rasterized=True)
        ax.add_collection(pc)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_title(f"t = {t * 1e12:.1f} ps", fontsize=9)
        ax.tick_params(labelsize=7)
        fig.colorbar(pc, ax=ax, shrink=0.7).set_label(
            "log10 rho [g/cc]", fontsize=7)
    fig.suptitle(f"{title} — density (log10)", fontsize=11)
    # full-page Agg raster: render to PNG, embed as image page
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    plt.close(fig)
    buf.seek(0)
    img = plt.imread(buf)
    fig2 = plt.figure(figsize=(8.27, 11.0))
    ax2 = fig2.add_axes([0, 0, 1, 1])
    ax2.imshow(img)
    ax2.axis("off")
    pdf.savefig(fig2)
    plt.close(fig2)


def series_pages(pdf, snaps, title, death_ps, note):
    rows = []
    for p in snaps:
        try:
            t, rms, hp, rel = probe_mod.probe(p, 20, 50)
            rows.append((t * 1e12, rms, hp, rel))
        except Exception:
            continue
    ts = np.array([r[0] for r in rows])
    rms = np.array([r[1] for r in rows])
    hp = np.array([r[2] for r in rows])
    rel = np.array([r[3] for r in rows])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.0))
    ax1.semilogy(ts, np.maximum(rms, 1e-17), "o-", label="theta-RMS(rho)/mean")
    ax1.semilogy(ts, np.maximum(hp, 1e-17), "s-",
                 label="high-pass (m>8) RMS")
    ax1.axhspan(1e-16, 1e-12, color="green", alpha=0.15,
                label="LSB noise band (spherical target)")
    if death_ps:
        ax1.axvline(death_ps, color="red", ls="--", lw=1,
                    label=f"chain end {death_ps:.0f} ps")
    ax1.set_xlabel("t [ps]")
    ax1.set_ylabel("median shell-row relative RMS")
    ax1.legend(fontsize=8)
    ax1.set_title(f"{title} — theta symmetry metrics", fontsize=11)
    ax1.grid(alpha=0.3)

    for l, mk in ((2, "o"), (4, "s"), (6, "^"), (8, "v")):
        ax2.plot(ts, rel[:, l], mk + "-", label=f"|a{l}/a0|", ms=3)
    if death_ps:
        ax2.axvline(death_ps, color="red", ls="--", lw=1)
    ax2.set_xlabel("t [ps]")
    ax2.set_ylabel("dm/dOmega Legendre amplitude (rel)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_title("column-integrated mass channel "
                  "(static offsets are estimator geometry)", fontsize=9)
    fig.text(0.08, 0.02, note, fontsize=7, wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--death-ps", type=float, default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results, "*_0*.h5")))
    paths = [p for p in paths if "history" not in p]
    if not paths:
        print("no snapshots")
        return 1
    sel = [paths[i] for i in
           np.linspace(0, len(paths) - 1, 4, dtype=int)]
    with PdfPages(args.out) as pdf:
        field_page(pdf, sel, args.title)
        series_pages(pdf, paths, args.title, args.death_ps, args.note)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
