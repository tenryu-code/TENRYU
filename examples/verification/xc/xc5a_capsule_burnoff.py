# xc5a_capsule_burnoff — XC-5a indirect-drive capsule implosion, burn OFF (TENRYU side).
# MULTI pair: ops/xc/cases/xc5a_capsule_burnoff/multi.input.
# Design: docs/design/xc_tenryu_multiife_1d_comparison_design_20260710.md §XC-5a.
#
# Synthetic 3-layer capsule with 2015IND-derived dimensions (doped layer merged
# into the ablator): DT vapor fill 980 um @ 3e-4 | DT fuel 50 um @ 0.22053 |
# Be-like ablator 160 um... no — 16 um @ 1.86 (Z=4, A=9). Drive = the 2015IND
# Tr(t) table verbatim (80 -> 250 eV over 14.2 ns, then hold; piecewise linear),
# outer Marshak, inner reflect (TENRYU node-0 pin == MULTI ileft=1). Constant
# grey kappa per material: DT 5.0, ablator 2e3 cm^2/g (pin refined on the first
# run). Conduction f=0.1 both; 2T live exchange both (hydro ON); burn OFF.
# Interpretation carries Addendum 5 (MULTI two-stream D=c/4kr => shallower
# radiation preheat) and Addendum 9 (delta convention) as declared budgets —
# radiation-driven so the delta budget is inactive here (no laser).
import bisect
import os

from tenryu_namelist import *

outdir = os.environ.get("TENRYU_XC5A_OUTDIR", "tmp/xc/xc5a_capsule_burnoff/tenryu")

R_FILL = 0.098           # cm
R_FUEL = R_FILL + 0.005
R_OUT = R_FUEL + 0.016   # doped layer merged into the ablator
RHO = [3.0e-4, 0.22053, 1.86]
EDGES = [0.0, R_FILL, R_FUEL, R_OUT]

# 2015IND drive table (ns, eV), piecewise linear; hold last value to t_end
TTAB = [0.0, 3.3, 4.6625, 6.025, 7.3875, 8.75, 10.1125, 11.475, 14.2]
PTAB = [80.0, 80.0, 87.0, 96.0, 111.0, 130.0, 159.0, 186.0, 250.0]


def tr_drive_eV(t_s):
    t_ns = t_s * 1.0e9
    if t_ns <= TTAB[0]:
        return PTAB[0]
    if t_ns >= TTAB[-1]:
        return PTAB[-1]
    i = bisect.bisect_right(TTAB, t_ns) - 1
    w = (t_ns - TTAB[i]) / (TTAB[i + 1] - TTAB[i])
    return PTAB[i] + w * (PTAB[i + 1] - PTAB[i])


def region(r):
    i = bisect.bisect_right(EDGES, r) - 1
    return max(0, min(i, 2))


Main(
    name="xc5a_capsule_burnoff",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=30.0e-9,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=R_OUT,
    grid=dict(type="graded", segments=[
        {"r_start": 0.0, "r_end": R_FILL, "nr": 40},
        {"r_start": R_FILL, "r_end": R_FUEL, "nr": 60},
        {"r_start": R_FUEL, "r_end": R_OUT, "nr": 100},
    ]),
    geometry_1d="spherical",
)

Materials(
    materials=[
        Material(
            name="dt",
            A=2.5,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=5.0, kappa_s=0.0,
                         units="cm2_per_g"),
        ),
        Material(
            name="ablator",
            A=9.0,
            Z=4.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=2.0e3, kappa_s=0.0,
                         units="cm2_per_g"),
        ),
    ],
    zbar=dict(model="fixed"),  # fixed = each material's own Z (DT 1, ablator 4)
)

Geometry(
    volfrac=dict(
        dt=lambda r: 1.0 if region(r) < 2 else 0.0,
        ablator=lambda r: 1.0 if region(r) == 2 else 0.0,
    ),
    rho=lambda r: RHO[region(r)],
    Te=lambda r: 1.0e-3,
    Ti=lambda r: 1.0e-3,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

Numerics(
    dt=dict(initial_s=1.0e-13, max_s=5.0e-12, growth_factor=1.2, min_s=1.0e-20),
    hydro=dict(enabled=True, boundary_1d="free"),
    conduction=dict(enabled=True, f_lim=0.1),
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=1,
    group_bounds_eV=[0.1, 1.0e5],
    compute_T_range_eV=[0.1, 1.0e5],
    multigroup_diffusion=dict(
        opacity_floor=0.0,
        opacity_cap=1.0e8,
        fleck_cv_source="table",
        boundary=dict(inner_r="reflect", outer_r="marshak"),
    ),
    boundary=dict(
        marshak_Tr=tr_drive_eV,
    ),
)

Laser(enabled=False, cbet=dict(enable=False), hot_electron=dict(enable=False))
Burn(enabled=False)

Output(
    directory=outdir,
    plot_every=0,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=3.75e-10,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
