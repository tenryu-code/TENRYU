#!/usr/bin/env python3
"""Grid convergence gate of the 1D FLD outer boundary closure (NUMERICS §6.7 1D BC, the internal verification record §23).

Runs examples/verification/fld_1d_outer_closure_slab.py with a Marshak face and with a vacuum face on a ladder of grids
(default 64, 128, 256 and 2048 cells; the outer cell's optical depth is 0.78 at 64 cells) and compares the net
radiation energy through the outer face at 0.2 ns (the history's cumulative energy/marshak_in - energy/radiation_escaped)
with the finest grid's. Before 2026-09-29 the face value was the outer cell's centre value and the coarse grids erred
at first order (+18.7 %, +10.4 %, +5.1 % with the Marshak face; +27 %, +14 %, +6.7 % in the escape with the vacuum
face); with the half-cell diffusion resistance they converge at second order or better.

Gates, per mode: the relative errors of the three coarse grids within the bounds below; the observed order between
successive coarse grids at least 1.8; every grid's energy conservation error (history energy/conservation_error)
below 1e-8. Exits non-zero when a gate fails or a run does not reach t_end.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

T_END_S = 2.0e-10
BOUNDS = {
    "marshak": {64: 0.06, 128: 0.015, 256: 0.004},
    "vacuum": {64: 0.08, 128: 0.02, 256: 0.004},
}
MIN_ORDER = 1.8
MAX_CONSERVATION_ERROR = 1.0e-8


def run_one(binary: Path, deck: Path, out_root: Path, mode: str, nr: int) -> Path:
    out_dir = out_root / f"{mode}_nr{nr}"
    env = dict(os.environ, FLD_CLOSURE_MODE=mode, FLD_CLOSURE_NR=str(nr), FLD_CLOSURE_OUTDIR=str(out_dir))
    log = out_root / f"{mode}_nr{nr}.log"
    with open(log, "w", encoding="utf-8") as stream:
        rc = subprocess.run([str(binary), "run", str(deck)], env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"{mode} nr={nr}: tenryu exited with {rc} (see {log})")
    return out_dir


def read_run(out_dir: Path) -> dict:
    histories = sorted((out_dir / "results").glob("*_history.h5"))
    if not histories:
        raise RuntimeError(f"{out_dir}: no history file")
    with h5py.File(histories[0], "r") as h:
        for name in ("energy/marshak_in_step", "energy/radiation_escaped_step"):
            if name not in h:
                # the cumulative ledgers carry *_step siblings since 2026-09-23; an older history cannot be read here
                raise RuntimeError(f"{histories[0]}: {name} missing (the ledgers are not the cumulative ones)")
        t = np.asarray(h["t"][()], dtype=float).reshape(-1)
        marshak_in = float(np.asarray(h["energy/marshak_in"][()], dtype=float).reshape(-1)[-1])
        escaped = float(np.asarray(h["energy/radiation_escaped"][()], dtype=float).reshape(-1)[-1])
        conservation = np.asarray(h["energy/conservation_error"][()], dtype=float).reshape(-1)
    return {
        "t_end": float(t[-1]),
        "marshak_in": marshak_in,
        "escaped": escaped,
        "net_in": marshak_in - escaped,
        "max_conservation_error": float(np.max(np.abs(conservation))) if conservation.size else math.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/fld_1d_outer_closure_slab.py")
    parser.add_argument("--out-root", default="./build/output_fld_1d_outer_closure")
    parser.add_argument("--modes", default="marshak,vacuum")
    parser.add_argument("--reference-nr", type=int, default=2048)
    parser.add_argument("--jobs", type=int, default=4, help="runs launched at once")
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    binary = Path(args.tenryu_bin).resolve()
    deck = Path(args.deck).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    modes = [m for m in args.modes.split(",") if m]
    coarse = sorted(BOUNDS["marshak"])
    grids = coarse + [args.reference_nr]

    jobs = [(mode, nr) for mode in modes for nr in grids]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {job: pool.submit(run_one, binary, deck, out_root, *job) for job in jobs}
        results = {job: read_run(fut.result()) for job, fut in futures.items()}

    ok = True
    summary = {}
    for mode in modes:
        ref = results[(mode, args.reference_nr)]
        errors = {}
        print(f"{mode}: net energy through the outer face at {T_END_S:.1e} s, against {args.reference_nr} cells "
              f"({ref['net_in']:.6e} erg/cm^2)")
        for nr in grids:
            r = results[(mode, nr)]
            err = (r["net_in"] - ref["net_in"]) / abs(ref["net_in"])
            errors[nr] = err
            reached = r["t_end"] >= T_END_S * (1.0 - 1.0e-9)
            cons_ok = r["max_conservation_error"] <= MAX_CONSERVATION_ERROR
            bound = BOUNDS[mode].get(nr)
            err_ok = True if bound is None else abs(err) <= bound
            status = "PASS" if (reached and cons_ok and err_ok) else "FAIL"
            ok = ok and status == "PASS"
            print(f"  nr {nr:5d}  net_in {r['net_in']:.6e}  rel_err {100.0 * err:+8.3f} %"
                  f"{'' if bound is None else f' (bound {100.0 * bound:.1f} %)'}  t_end {r['t_end']:.3e}"
                  f"  max|conservation error| {r['max_conservation_error']:.2e}  {status}")
        orders = []
        for a, b in zip(coarse[:-1], coarse[1:]):
            ea, eb = abs(errors[a]), abs(errors[b])
            order = math.log(ea / eb) / math.log(b / a) if ea > 0.0 and eb > 0.0 else math.inf
            orders.append(order)
            status = "PASS" if order >= MIN_ORDER else "FAIL"
            ok = ok and status == "PASS"
            print(f"  observed order {a} -> {b}: {order:.2f} (at least {MIN_ORDER})  {status}")
        summary[mode] = {"errors": {str(k): v for k, v in errors.items()}, "orders": orders,
                         "runs": {str(nr): results[(mode, nr)] for nr in grids}}
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Overall:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
