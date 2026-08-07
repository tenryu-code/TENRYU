# MPI 2D burn rank-agreement gate deck (M18d d2). Hot DT disk (5 keV,
# r<0.25 cm, |z|<0.15 cm) in a 1 keV bath, radiation OFF, Burn
# scheme="diffusion" (Corman alpha diffusion — exercises the distributed
# CG: owned-masked dots + Allreduce + search-direction ghost exchange)
# with the owned-window network stage. Owned-slab dumps from P in {2,4}
# vs P=1 (tools/validation/compare_owned_dump.py); tiers decided from
# measurement (design doc §6k).
#
# Env knobs:
#   TENRYU_PBURN2D_SCHEME    Burn scheme (default "diffusion"; "local")
#   TENRYU_PBURN2D_MAXSTEPS  max_steps (default 25)
#   TENRYU_PBURN2D_OUTDIR    output directory

import os

from tenryu_namelist import *

_SCHEME = os.environ.get("TENRYU_PBURN2D_SCHEME", "diffusion")
_MAX_STEPS = int(os.environ.get("TENRYU_PBURN2D_MAXSTEPS", "25"))
_OUTDIR = os.environ.get(
    "TENRYU_PBURN2D_OUTDIR", "./output_parallel_burn_2d"
)

Main(
    name="parallel_burn_2d",
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=1.0e-9,
    max_steps=_MAX_STEPS,
    seed=12345,
)

Mesh(
    r_min=0.0,
    r_max=1.0,
    z_min=-0.5,
    z_max=0.5,
    nr=64,
    nz=32,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="DT",
            A=2.5,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(
                model="constant",
                kappa_a=0.0,
                kappa_s=0.0,
                units="cm2_per_g",
            ),
        ),
    ],
)

_T_HOT_EV = 5.0e3
_T_COLD_EV = 1.0e3
_R_CORE = 0.25
_Z_CORE = 0.15


def _te_init(r_cm, z_cm):
    if r_cm < _R_CORE and abs(z_cm) < _Z_CORE:
        return _T_HOT_EV
    return _T_COLD_EV


Geometry(
    volfrac=dict(DT=lambda r_cm, z_cm: 1.0),
    rho=lambda r_cm, z_cm: 1.0,
    Te=_te_init,
    Ti=_te_init,
    velocity=None,
    radiation_field="zero",
)

Numerics(
    conduction=dict(enabled=False),
    dt=dict(
        initial_s=2.0e-13,
        max_s=2.0e-13,
        min_s=1.0e-22,
        growth_factor=1.0,
    ),
    hydro=dict(
        enabled=True,
        boundary_2d=dict(
            r_inner="axis", r_outer="free", z_bottom="free", z_top="free"
        ),
    ),
)

Output(
    directory=_OUTDIR,
    plot_every=1000000,
)

Radiation(enabled=False)

Burn(
    enabled=True,
    scheme=_SCHEME,
    fuels=["DT"],
    fuel_materials=["DT"],
)

Laser(enabled=False)
