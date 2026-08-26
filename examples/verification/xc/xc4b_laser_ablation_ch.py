# xc4b_laser_ablation_ch — XC-4b: XC-4 with the REAL converted CH tables (W5 leg).
# Same-tables directive: TENRYU runs tmp/xc/tmat/CH_multi.h5 (converted from
# MULTI's bundled tables/Carbon — eeos 75901 / ieos 75902 / planck 50003000 /
# ross 50004000, Z=3.5 fixed, LTE); MULTI runs tables/Carbon directly.
# 20-group FLD at the TMAT bounds == MULTI ng=20. Single material, so the
# multi-material closure pitfall does not apply.
# MULTI pair: ops/xc/cases/xc4_laser_ablation/multi.input (CPC88-heritage numbers:
# Al-like foil, 3e14 W/cm^2, 300 ps FWHM sin^2, 0.44 um — the 2015CPC example
# reduced to a single synthetic ideal-gas material).
# Design: docs/design/xc_tenryu_multiife_1d_comparison_design_20260710.md §XC-4.
#
# Quasi-planar shell at R0 = 10 cm (tier-0 finding: TENRYU's planar
# radial_absorption_1d power convention is ill-defined; the spherical
# convention I = P/(4 pi R0^2) is exact — curvature over 2 um is 2e-5).
# Rear (inner node) is pinned by TENRYU (NUMERICS §3) — MULTI mirrors with
# ileft=1; outer face free. Radiation ON, grey, small constant kappa=10
# (tau_foil ~ 5e-3: laser-dominated energetics). f-scan via TENRYU_XC4_F.
#
# v2: LOW-DENSITY SEED CORONA in front of the foil (identical staircase in
# both codes) — v1 started from bare solid and TENRYU deposited EXACTLY
# nothing (ledger laser_deposited = 0.0): with every ray meeting n_e = 45 n_c
# at the first cell, the delta=0 convention (critical residual reflects,
# Addendum 1.D) reflects the whole pulse. The design's "low-density start
# layer per MULTI's WKB needs" is in fact TENRYU's need. Seed: 0.4/0.1/0.02/
# 0.004 g/cc x 2/3/5/10 um (outermost 0.2 n_c — under-critical entry);
# uniform dx = 50 nm both codes (MULTI mirrors with per-layer nc).
import bisect
import math
import os

from tenryu_namelist import *

NR = int(os.environ.get("TENRYU_XC4B_NR", "440"))
F_LIM = float(os.environ.get("TENRYU_XC4B_F", "0.1"))
outdir = os.environ.get(
    "TENRYU_XC4B_OUTDIR",
    f"tmp/xc/xc4b_laser_ablation_ch/tenryu_n{NR}_f{F_LIM:g}".replace(".", "p"))

R0 = 10.0                # cm
THICK = 2.0e-4           # cm (2 um foil)
I_W_CM2 = 3.0e14
P0_W = I_W_CM2 * 4.0 * math.pi * R0 * R0    # 3.7699e17 W
FWHM_S = 300.0e-12       # sin^2: P = P0 sin^2(pi t / (2 FWHM)), duration 2*FWHM

# seed staircase, foil surface outward: (width_cm, rho_g_cc)
SEED = [(2.0e-4, 0.4), (3.0e-4, 0.1), (5.0e-4, 0.02), (1.0e-3, 0.004)]
EDGES = [R0 + THICK]
RHOS = [2.7]
for w, rho_v in SEED:
    EDGES.append(EDGES[-1] + w)
    RHOS.append(rho_v)
R_MAX = EDGES[-1]        # 10.0 + 22 um


def rho_profile(r):
    if r < R0 + THICK:
        return 1.05
    i = bisect.bisect_right(EDGES, r) - 1
    i = max(0, min(i, len(RHOS) - 1))
    return RHOS[i]

Main(
    name=f"xc4b_laser_ablation_ch_n{NR}",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=610.0e-12,
    seed=12345,
    verbosity="quiet",
)

Mesh(r_min=R0, r_max=R_MAX, nr=NR, grid="uniform", geometry_1d="spherical")

Materials(
    materials=[
        Material(
            name="ch",
            A=6.51,
            Z=3.5,
            eos=dict(model="tmat", file="tmp/xc/tmat/CH_multi.h5"),
            opacity=dict(model="tmat", file="tmp/xc/tmat/CH_multi.h5"),
        ),
    ],
    zbar=dict(model="fixed", fixed_value=3.5),
)

Geometry(
    volfrac=dict(ch=lambda r: 1.0),
    rho=rho_profile,  # solid 1.05 g/cc + the xc4 seed staircase (n_c(CH)=1.78e-2 similar)
    Te=lambda r: 0.03,
    Ti=lambda r: 0.03,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

Numerics(
    dt=dict(initial_s=1.0e-16, max_s=2.0e-13, growth_factor=1.2, min_s=1.0e-21),
    hydro=dict(enabled=True, boundary_1d="free"),
    conduction=dict(enabled=True, f_lim=F_LIM),
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=20,
    group_bounds_eV=[1.0, 10.0, 50.0, 100.0, 150.0, 250.0, 400.0, 550.0, 700.0,
                     900.0, 1200.0, 1600.0, 2000.0, 2200.0, 2400.0, 2600.0,
                     2800.0, 3100.0, 3500.0, 4000.0, 5000.0],
    compute_T_range_eV=[0.1, 1.0e5],
    multigroup_diffusion=dict(
        opacity_floor=0.0,
        opacity_cap=1.0e30,
        fleck_cv_source="table",
    ),
)

Laser(
    enabled=True,
    wavelength_nm=440.0,
    mode="radial_absorption_1d",
    beams=[
        LaserBeam(
            name="beam_00",
            direction=(0.0, 0.0, -1.0),
            power=lambda t: (P0_W * math.sin(math.pi * t / (2.0 * FWHM_S)) ** 2
                             if t < 2.0 * FWHM_S else 0.0),
            profile=dict(model="super_gaussian", w0_um=1000.0, m=2),
        )
    ],
    cbet=dict(enable=False),
    hot_electron=dict(enable=False),
)
Burn(enabled=False)

Output(
    directory=outdir,
    plot_every=0,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=2.0e-11,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
