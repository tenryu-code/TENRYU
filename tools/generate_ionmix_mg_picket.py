#!/usr/bin/env python3
"""Generate the G=2 picket-fence IONMIX fixtures for the Su & Olson (1999)
multigroup FLD benchmark gate (fld_1d_mg_picket_fence_suolson).

    python3 tools/generate_ionmix_mg_picket.py

writes tests/data/ionmix_mg_picket_{b,b_swap,c,c_swap}.cn4.

kappa_PA == kappa_PE == kappa_R == w_n [cm^2/g], constant over the whole
(rho, T) grid (exact bilinear interpolation; rho = 1 g/cc in the gate so
sigma_n = w_n kbar with kbar = 1/cm). The *_swap fixtures carry the two
bands in reversed order for the source-superposition runs (the FLD volume
source is deposited into group 0 only; the benchmark's p_n-split source is
recovered by linearity: Sol(p1 Q e1 + p2 Q e2) = p1 Sol(Q e1) + p2 Sol(Q e2),
with Sol(Q e2) obtained from the band-swapped medium by relabeling).
Group bounds are placeholders — the gate drives b_g == p_n through
PlanckTable::build_constant_fractions, so the frequency bounds never enter
the physics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_ionmix_nlte import write_ionmix_cn4  # noqa: E402


def write_pair(stem: str, kappas: list[float]) -> None:
    ntemp = 10
    ndens = 5
    temps_eV = np.logspace(-2.0, 4.0, ntemp, dtype=np.float64)
    numdens_cm3 = np.logspace(18.0, 24.0, ndens, dtype=np.float64)
    bounds_eV = np.array([1.0, 100.0, 10000.0], dtype=np.float64)
    out_dir = Path(__file__).resolve().parents[1] / "tests/data"
    for suffix, kaps in ((f"{stem}.cn4", kappas),
                         (f"{stem}_swap.cn4", list(reversed(kappas)))):
        table = np.empty((2, ndens, ntemp), dtype=np.float64)
        for g, kappa in enumerate(kaps):
            table[g, :, :] = kappa
        write_ionmix_cn4(
            output_path=out_dir / suffix,
            temps_eV=temps_eV,
            numdens_cm3=numdens_cm3,
            bounds_eV=bounds_eV,
            kappa_R=table,
            kappa_PA=table,
            kappa_PE=table,
        )
        print(f"Generated: {out_dir / suffix}")


def main() -> None:
    write_pair("ionmix_mg_picket_b", [2.0 / 11.0, 20.0 / 11.0])
    write_pair("ionmix_mg_picket_c", [2.0 / 101.0, 200.0 / 101.0])


if __name__ == "__main__":
    main()
