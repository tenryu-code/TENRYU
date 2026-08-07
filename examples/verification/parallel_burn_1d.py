# MPI 1D burn replication gate deck (M18d d2). Same planar-like 1D_SPH
# shell as parallel_rad_1d.py, but with a DT fuel material, a 5 keV hot
# core (DT reactions active) in a 1 keV bath, radiation OFF — isolates
# the replicated burn stage (network + tallies + inventories). Under P>1
# the burn inputs are line-gathered and the host stage runs the full
# line on every rank, so owned-slab dumps must be BITWISE identical to
# P=1 (tools/validation/compare_owned_dump.py --rtol 0).
# Design doc docs/design/mpi_m18_20_20260717.md §6k; spec
# docs/design/mpi_m18d_laser_burn_spec.md §2 d2.
#
# Env knobs:
#   TENRYU_PBURN1D_SCHEME    Burn scheme (default "fraley"; "mc" probes
#                            the 1D mc dispatch)
#   TENRYU_PBURN1D_MAXSTEPS  max_steps (default 25)
#   TENRYU_PBURN1D_OUTDIR    output directory

import os

from tenryu_namelist import *

_SCHEME = os.environ.get("TENRYU_PBURN1D_SCHEME", "fraley")
_MAX_STEPS = int(os.environ.get("TENRYU_PBURN1D_MAXSTEPS", "25"))
_OUTDIR = os.environ.get(
    "TENRYU_PBURN1D_OUTDIR", "./output_parallel_burn_1d"
)

Main(
    name="parallel_burn_1d",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=1.0e-9,
    max_steps=_MAX_STEPS,
    seed=12345,
)

Mesh(
    r_min=100.0,
    r_max=100.6,
    nr=192,
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
_R_HOT = 100.1


def _te_init(r_cm):
    return _T_HOT_EV if r_cm < _R_HOT else _T_COLD_EV


Geometry(
    volfrac=dict(DT=lambda r: 1.0),
    rho=lambda r: 1.0,
    Te=_te_init,
    Ti=_te_init,
    velocity=lambda r: 0.0,
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
    hydro=dict(enabled=True, boundary_1d="reflect"),
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
