# xc3_radiative_ablation — XC-3 subsonic radiative ablation (TENRYU side).
# MULTI pair: ops/xc/cases/xc3_rad_ablation/multi.input.
# Design: docs/design/xc_tenryu_multiife_1d_comparison_design_20260710.md §XC-3.
#
# Constant Tr = 150 eV Marshak drive onto a solid Be-like ideal-gas slab
# (Z=4 fixed, A=9, rho=1.85, gamma=5/3), constant grey kappa = 3e3 cm^2/g
# (pinned for a clearly subsonic heat front: v_front ~ 2.4e6 cm/s vs
# c_s(150 eV) ~ 1.15e7 — Mach ~ 0.2). Slab 100 um at [0.05, 0.06] cm
# (offset from the origin so the free rear face is away from r=0), free
# boundaries both ends (MULTI ileft=iright=0): rear breakout is physical.
# Coupled rad+hydro+conduction (f=0.1, both codes; TENRYU limiter is the
# BUG-19-fixed Kirchhoff-consistent one), 2T with live exchange both codes
# (hydro ON => TENRYU Q_ei alive — the xc0a asymmetry does not apply).
# No analytic reference: code-to-code with N-ladders (TENRYU_XC3_NR).
import os

from tenryu_namelist import *

NR = int(os.environ.get("TENRYU_XC3_NR", "200"))
outdir = os.environ.get("TENRYU_XC3_OUTDIR", f"tmp/xc/xc3_rad_ablation/tenryu_n{NR}")

RHO0 = 1.85
T0 = 1.0          # eV both channels (P0 ~ 1 Mbar ~ 2% of the ablation pressure)

Main(
    name=f"xc3_rad_ablation_n{NR}",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=3.0e-9,
    seed=12345,
    verbosity="quiet",
)

Mesh(r_min=0.05, r_max=0.06, nr=NR, grid="uniform", geometry_1d="planar")

Materials(
    materials=[
        Material(
            name="be_like",
            A=9.0,
            Z=4.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=3.0e3, kappa_s=0.0,
                         units="cm2_per_g"),
        ),
    ],
    zbar=dict(model="fixed", fixed_value=4.0),
)

Geometry(
    volfrac=dict(be_like=lambda r: 1.0),
    rho=lambda r: RHO0,
    Te=lambda r: T0,
    Ti=lambda r: T0,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

Numerics(
    dt=dict(initial_s=1.0e-15, max_s=5.0e-13, growth_factor=1.2, min_s=1.0e-20),
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
        marshak_Tr=lambda t: 150.0,
    ),
)

Laser(enabled=False, cbet=dict(enable=False), hot_electron=dict(enable=False))
Burn(enabled=False)

Output(
    directory=outdir,
    plot_every=0,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=1.0e-10,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
