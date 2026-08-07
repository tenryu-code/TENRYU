# MPI rank-count agreement gate deck (VERIFICATION §16.2 class, M18 W2).
# 1D spherical Sedov-type blast, hydro-only, deterministic: owned-slab dumps
# from P=1 and P>1 runs must agree BITWISE
# (tools/validation/compare_owned_dump.py; design doc
# docs/design/mpi_m18_20_20260717.md §5.4).

from tenryu_namelist import *

Main(
    name="parallel_sedov_1d",
    dimension="1D_SPH",
    t_end=2.0e-7,
    max_steps=400,
    seed=12345,
)

Mesh(
    r_min=0.0,
    r_max=1.0,
    nr=200,
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
                kappa_a=0.0,
                kappa_s=0.0,
                units="cm2_per_g",
            ),
        ),
    ],
)

_E0_EV = 3.0e6  # hot-core temperature spike [eV]
_R_CORE = 0.02  # cm


def _te_init(r_cm):
    return _E0_EV if r_cm < _R_CORE else 1.0


Geometry(
    volfrac=dict(gas=lambda r_cm: 1.0),
    rho=lambda r_cm: 1.0,
    Te=_te_init,
    Ti=_te_init,
    velocity=None,
    radiation_field="zero",
)

Numerics(
    conduction=dict(enabled=False),
)

Output(
    directory="./output_parallel_sedov_1d",
    plot_every=1000000,
)

Radiation(enabled=False)
Laser(enabled=False)
