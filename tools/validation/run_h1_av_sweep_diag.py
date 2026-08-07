#!/usr/bin/env python3
"""Diagnostic AV C2 sweep for H1 (Test C from peer review).

Purpose: determine whether rz_inactive_max scales with the VNR quadratic
coefficient C2, or whether it persists at C2=0 (which would point to
non-VNR axis-row pathology, complementing Test E findings).

Does NOT touch docs/validation/2d_rz/H1/. Writes diagnostic results to
tmp/diagnostics/h1_av_sweep/.

Run via:
    python3 tools/validation/run_h1_av_sweep_diag.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

NR = 64
NZ = 256
SEED = 12345
T_END_S = 1.0e-7
CFL = 0.1
DECK = "examples/verification/2d_rz_h1_sod_planar.py"
TENRYU_BIN = "./build/tenryu"

C2_VALUES = [0.0, 0.5, 1.5, 3.0]
OUT_ROOT = Path("./build/h1_av_sweep_diag")
DIAG_ROOT = Path("./tmp/diagnostics/h1_av_sweep")


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
        raise FileNotFoundError(f"no plot files under {outdir} or {outdir}_*")
    return max(candidates, key=lambda p: p.parent.stat().st_mtime)


def numbered_plots(results_dir: Path, name: str) -> list[Path]:
    return sorted(
        (p for p in results_dir.glob(f"{name}_*.h5") if p.stem.removeprefix(f"{name}_").isdigit()),
        key=lambda p: int(p.stem.removeprefix(f"{name}_")),
    )


def safe_token(v: float) -> str:
    return f"{v:.12g}".replace(".", "p")


def run_one(c2: float) -> dict:
    case_id = f"av_c2{safe_token(c2)}"
    outdir = OUT_ROOT / case_id
    outdir.mkdir(parents=True, exist_ok=True)
    case_name = f"2d_rz_h1_sod_planar_nr{NR}_nz{NZ}_seed{SEED}_C2{safe_token(c2)}_cfl{safe_token(CFL)}"

    env = os.environ.copy()
    env.update({
        "TENRYU_H1_NR": str(NR),
        "TENRYU_H1_NZ": str(NZ),
        "TENRYU_H1_SEED": str(SEED),
        "TENRYU_H1_OUTDIR": str(outdir),
        "TENRYU_H1_AV_C2": f"{c2:.12g}",
        "TENRYU_H1_CFL": f"{CFL:.12g}",
        "TENRYU_H1_T_END_S": f"{T_END_S:.12g}",
    })
    log = outdir / "run.log"
    print(f"  [c2={c2}] launching tenryu run...", flush=True)
    t0 = time.monotonic()
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run([TENRYU_BIN, "run", DECK], env=env, stdout=f, stderr=f)
    wall = time.monotonic() - t0
    if proc.returncode != 0:
        print(f"  [c2={c2}] FAILED (exit {proc.returncode})")
        return {"c2": c2, "status": "CRASHED", "exit_code": proc.returncode, "wall_s": wall}

    rd = discover_results_dir(outdir)
    plots = numbered_plots(rd, case_name)
    if not plots:
        return {"c2": c2, "status": "NO_PLOTS", "wall_s": wall}
    last = plots[-1]
    with h5py.File(last, "r") as h:
        rho = np.asarray(h["hydro/rho"]).reshape(NR, NZ)
        x_z = np.asarray(h["mesh/x_z"]).reshape(NR + 1, NZ + 1)
        v_z = np.asarray(h["mesh/v_z"]).reshape(NR + 1, NZ + 1) if h["mesh/v_z"].size == (NR+1)*(NZ+1) else np.asarray(h["mesh/v_z"])
        t_final = float(h["time_state/t"][()])
        step_final = int(h["time_state/step"][()])

    # rz_inactive metric (max norm, simple version)
    q_bar = rho.mean(axis=0)
    scale = float(np.max(np.abs(q_bar)))
    rz_max = float(np.max(np.abs(rho - q_bar[None, :]) / max(scale, 1.0)))

    # vertex 128 lag at axis i=0 vs at i=N/2
    if x_z.shape == (NR+1, NZ+1):
        v128_axis_z = float(x_z[0, 128])
        v128_bulk_z = float(x_z[NR // 2, 128])
        analytic_contact = 0.84119485 * (t_final * 1e6 / 1.0)  # u_star * t_norm * L_scale
        axis_lag = (analytic_contact - v128_axis_z) / max(abs(analytic_contact), 1e-30) * 100.0
        bulk_lag = (analytic_contact - v128_bulk_z) / max(abs(analytic_contact), 1e-30) * 100.0
    else:
        axis_lag = bulk_lag = None
        v128_axis_z = v128_bulk_z = None

    return {
        "c2": c2,
        "status": "OK",
        "wall_s": wall,
        "step_final": step_final,
        "t_final_s": t_final,
        "rz_inactive_max": rz_max,
        "vertex128_axis_z_cm": v128_axis_z,
        "vertex128_bulk_z_cm": v128_bulk_z,
        "axis_contact_lag_pct": axis_lag,
        "bulk_contact_lag_pct": bulk_lag,
        "outdir": str(outdir),
    }


def main():
    DIAG_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"H1 AV C2 sweep: nr={NR} nz={NZ} seed={SEED} t_end={T_END_S}")
    print(f"  C2 values: {C2_VALUES}")
    print(f"  output: {OUT_ROOT}/<case>/")
    rows = []
    for c2 in C2_VALUES:
        rows.append(run_one(c2))
    out = {
        "stage": "H1_av_sweep_diagnostic",
        "purpose": "Test C: determine whether rz_inactive_max scales with VNR C2",
        "config": {
            "nr": NR, "nz": NZ, "seed": SEED, "cfl": CFL, "t_end_s": T_END_S,
            "c2_values": C2_VALUES,
        },
        "results": rows,
    }
    out_path = DIAG_ROOT / "results.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nResults written to {out_path}")
    print(f"\n{'C2':>6s}  {'rz_inactive_max':>18s}  {'axis_lag_pct':>14s}  {'bulk_lag_pct':>14s}  {'wall_s':>8s}")
    for r in rows:
        if r["status"] == "OK":
            print(f"  {r['c2']:>4.2f}  {r['rz_inactive_max']:>18.6e}  {r.get('axis_contact_lag_pct',0):>13.2f}%  {r.get('bulk_contact_lag_pct',0):>13.2f}%  {r['wall_s']:>7.1f}")
        else:
            print(f"  {r['c2']:>4.2f}  {r['status']:>18s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
