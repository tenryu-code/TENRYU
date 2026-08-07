# MPI rank-count agreement deck for coupled SN-2D (M18c slice-3c,
# INFORMATIONAL companion to VERIFICATION §16.5 — the §16.5 binding gates
# are the STATIC transport decks driven via 2d_rz_r2_sn_only.py; this deck
# adds 2T matter coupling + hydro so the KBA pipeline is exercised on an
# EVOLVING medium. Same geometry/coupling class as parallel_fld_2d.py:
# hot disk (300 eV, r<0.25 cm, |z|<0.15 cm) in a 10 eV bath, rho=1,
# kappa_a=50 cm2/g, 64x32, 4 groups, dt=2e-13 fixed. Owned-slab dumps
# from P in {2,4} vs P=1 (tools/validation/compare_owned_dump.py).
# Design doc docs/design/mpi_m18_20_20260717.md §6h; KBA spec
# docs/design/mpi_m18c_sn2d_kba_spec.md.
#
# Env knobs:
#   TENRYU_PSN2D_SCHEME    sn spatial_scheme (default
#                          "linear_characteristic"; "diamond_difference")
#   TENRYU_PSN2D_DSA       "1" (default) / "0" -> dsa_enabled
#   TENRYU_PSN2D_MAXSTEPS  max_steps (default 25; tier-1 probes use 1)
#   TENRYU_PSN2D_OUTDIR    output directory

import os

from tenryu_namelist import *

_SCHEME = os.environ.get("TENRYU_PSN2D_SCHEME", "linear_characteristic")
_DSA = os.environ.get("TENRYU_PSN2D_DSA", "1") == "1"
_MAX_STEPS = int(os.environ.get("TENRYU_PSN2D_MAXSTEPS", "25"))
_OUTDIR = os.environ.get(
    "TENRYU_PSN2D_OUTDIR", "./output_parallel_sn_2d"
)

Main(
    name="parallel_sn_2d",
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

_T_HOT_EV = 300.0
_T_COLD_EV = 10.0
_R_CORE = 0.25  # cm — the P=2 interface sits at r=0.5 (i=32 of 64)
_Z_CORE = 0.15  # cm


def _te_init(r_cm, z_cm):
    if r_cm < _R_CORE and abs(z_cm) < _Z_CORE:
        return _T_HOT_EV
    return _T_COLD_EV


Geometry(
    volfrac=dict(gas=lambda r_cm, z_cm: 1.0),
    rho=lambda r_cm, z_cm: 1.0,
    Te=_te_init,
    Ti=_te_init,
    velocity=None,
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

Radiation(
    enabled=True,
    mode="sn_transport",
    groups=4,
    group_bounds_eV=[0.0, 100.0, 300.0, 1000.0, 1.0e6],
    imc=dict(enabled=False, two_stage=False, difference=dict(enabled=False)),
    ddmc=dict(enabled=False),
    holo=dict(enabled=False),
    sn_transport=dict(
        spatial_scheme=_SCHEME,
        n_angles=8,
        max_outer_iterations=50,
        outer_tol=1.0e-8,
        dsa_enabled=_DSA,
        boundary=dict(
            inner_r="reflect_parity", outer_r="vacuum", z="vacuum"
        ),
    ),
    boundary=dict(
        inner_r="reflect", outer_r="vacuum", bottom_z="vacuum", top_z="vacuum"
    ),
)

Laser(enabled=False)
