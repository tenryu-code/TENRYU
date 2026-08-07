# MPI 1D deterministic-radiation replication gate deck (VERIFICATION
# §16.4, M18c). 1D spherical thin shell at large radius (r in [100.0,
# 100.6] cm — planar-like), three regions: hot wall layer (300 eV, r <
# 100.1) | mid layer | cold bath (10 eV), 2T, hydro on, constant opacity.
# Under P>1 the deterministic 1D radiation path (FLD or SN) runs the FULL
# line redundantly on every rank from Allgatherv'd matter inputs (driver
# allgather_1d_radiation_inputs), so owned-slab dumps must be BITWISE
# identical to the P=1 run (tools/validation/compare_owned_dump.py
# --rtol 0).
# With TENRYU_PRAD1D_GAP=1 the mid layer r in [100.2, 100.4) becomes a
# transparent gap (rho=1e-6 -> tau ~ 1e-5): radiation streams across rank
# interfaces inside the gap (192 cells: P=2 interface = cell 96 and P=4
# interface = cell 96 both sit inside the gap cells 64..127). True-void
# contracts stay covered by the P=1 ctest gate
# sn_1d_planar_transparent_gap; this deck gates rank-count invariance.
# Design doc docs/design/mpi_m18_20_20260717.md §6h.
#
# Env knobs:
#   TENRYU_PRAD1D_MODE      "multigroup_diffusion" (default) |
#                           "sn_transport"
#   TENRYU_PRAD1D_GAP       "1" -> transparent mid layer (default "0")
#   TENRYU_PRAD1D_MAXSTEPS  max_steps (default 25)
#   TENRYU_PRAD1D_OUTDIR    output directory

import os

from tenryu_namelist import *

_MODE = os.environ.get("TENRYU_PRAD1D_MODE", "multigroup_diffusion")
_GAP = os.environ.get("TENRYU_PRAD1D_GAP", "0") == "1"
_MAX_STEPS = int(os.environ.get("TENRYU_PRAD1D_MAXSTEPS", "25"))
_OUTDIR = os.environ.get(
    "TENRYU_PRAD1D_OUTDIR", "./output_parallel_rad_1d"
)

Main(
    name="parallel_rad_1d",
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
            name="gas",
            A=2.0,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(
                model="constant",
                kappa_a=50.0,
                kappa_s=0.0,
                units="cm2_per_g",
            ),
        ),
    ],
)

_R_HOT = 100.1
_R_GAP_LO = 100.2
_R_GAP_HI = 100.4
_RHO_GAP = 1.0e-6 if _GAP else 1.0


def _rho_init(r_cm):
    if _R_GAP_LO <= r_cm < _R_GAP_HI:
        return _RHO_GAP
    return 1.0


def _te_init(r_cm):
    return 300.0 if r_cm < _R_HOT else 10.0


Geometry(
    volfrac=dict(gas=lambda r: 1.0),
    rho=_rho_init,
    Te=_te_init,
    Ti=_te_init,
    velocity=lambda r: 0.0,
    radiation_field="equilibrium",
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

_RAD_COMMON = dict(
    enabled=True,
    groups=4,
    group_bounds_eV=[0.0, 100.0, 300.0, 1000.0, 1.0e6],
    imc=dict(enabled=False, two_stage=False, difference=dict(enabled=False)),
    ddmc=dict(enabled=False),
    holo=dict(enabled=False),
    boundary=dict(inner_r="reflect", outer_r="vacuum"),
)

if _MODE == "sn_transport":
    Radiation(
        mode="sn_transport",
        sn_transport=dict(
            n_angles=8,
            max_outer_iterations=20,
            outer_tol=1.0e-8,
            dsa_enabled=True,
            boundary=dict(inner_r="reflect_parity", outer_r="vacuum"),
        ),
        **_RAD_COMMON,
    )
else:
    Radiation(
        mode="multigroup_diffusion",
        multigroup_diffusion=dict(
            flux_limiter="levermore_pomraning",
            max_outer_iterations=50,
            outer_tol=1.0e-8,
            outer_accel="anderson",
            boundary=dict(inner_r="reflect", outer_r="vacuum"),
        ),
        **_RAD_COMMON,
    )

Laser(enabled=False)
