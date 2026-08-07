#!/usr/bin/env python3
"""Generate the G=4 Kirchhoff-consistent IONMIX fixture for the multigroup
FLD table_nlte verification gates (fld_1d_mg_nlte_*).

One-shot reproducible generation (run from anywhere):

    python3 tools/generate_ionmix_mg_fld.py

writes tests/data/ionmix_mg_fld_g4.cn4.

Design (docs/VERIFICATION.md §7.5): bounds {1, 50, 150, 400, 1500} eV chosen
so every group carries b_g >= 3e-2 at the gate plateau T = 50 eV; per-group
kappa {30, 100, 300, 1000} cm^2/g CONSTANT over the whole (rho, T) grid so
the bilinear log-log table interpolation is exact at any state point;
kappa_PA == kappa_PE == kappa_R (Kirchhoff/LTE-consistent) so the multigroup
fixed point is E_g = b_g a T^4. Grid ranges follow the existing fixtures
(tools/generate_ionmix_nlte.py defaults): T in [1e-2, 1e4] eV, n_i in
[1e18, 1e24] cm^-3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_ionmix_nlte import write_ionmix_cn4  # noqa: E402


def main() -> None:
    ntemp = 10
    ndens = 5
    temps_eV = np.logspace(-2.0, 4.0, ntemp, dtype=np.float64)
    numdens_cm3 = np.logspace(18.0, 24.0, ndens, dtype=np.float64)
    bounds_eV = np.array([1.0, 50.0, 150.0, 400.0, 1500.0], dtype=np.float64)
    kappas = [30.0, 100.0, 300.0, 1000.0]
    ngroups = len(kappas)
    assert bounds_eV.size == ngroups + 1

    table = np.empty((ngroups, ndens, ntemp), dtype=np.float64)
    for g, kappa in enumerate(kappas):
        table[g, :, :] = kappa

    out = Path(__file__).resolve().parents[1] / "tests/data/ionmix_mg_fld_g4.cn4"
    write_ionmix_cn4(
        output_path=out,
        temps_eV=temps_eV,
        numdens_cm3=numdens_cm3,
        bounds_eV=bounds_eV,
        kappa_R=table,
        kappa_PA=table,
        kappa_PE=table,
    )
    print(f"Generated: {out}")


if __name__ == "__main__":
    main()
