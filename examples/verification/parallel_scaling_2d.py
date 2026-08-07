# MPI M20 scaling deck (VERIFICATION SS16.11): the parallel_fld_2d gate
# physics (hydro + FLD diffusion-limit) on a scalable mesh. Weak scaling
# grows NR with the rank count (r_max grows with it so the cell size and
# time step stay fixed); strong scaling fixes the global mesh.
#
# Knobs:
#   TENRYU_PSCALE_NR        radial cells (default 512)
#   TENRYU_PSCALE_NZ        axial cells (default 512)
#   TENRYU_PSCALE_MAXSTEPS  max_steps (default 25)
#   TENRYU_PSCALE_OUTDIR    output directory

import os

from tenryu_namelist import *

_NR = int(os.environ.get("TENRYU_PSCALE_NR", "512"))
_NZ = int(os.environ.get("TENRYU_PSCALE_NZ", "512"))
_MAX_STEPS = int(os.environ.get("TENRYU_PSCALE_MAXSTEPS", "25"))
_OUTDIR = os.environ.get(
    "TENRYU_PSCALE_OUTDIR", "./output_parallel_scaling_2d"
)

# keep dr identical across weak-scaling sizes: r_max tracks NR
_R_MAX = 1.0 * (_NR / 512.0)

Main(
    name="parallel_scaling_2d",
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=1.0e-9,
    max_steps=_MAX_STEPS,
    seed=12345,
)

Mesh(
    r_min=0.0,
    r_max=_R_MAX,
    z_min=-0.5,
    z_max=0.5,
    nr=_NR,
    nz=_NZ,
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
_R_CORE = 0.25  # cm — inner hot disk, present at every weak-scaling size
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
    mode="multigroup_diffusion",
    groups=4,
    group_bounds_eV=[0.0, 100.0, 300.0, 1000.0, 1.0e6],
    imc=dict(enabled=False, two_stage=False, difference=dict(enabled=False)),
    ddmc=dict(enabled=False),
    holo=dict(enabled=False),
    multigroup_diffusion=dict(
        flux_limiter="levermore_pomraning",
        max_outer_iterations=50,
        outer_tol=1.0e-8,
        outer_accel="anderson",
        linear_solver_2d="cusparse_cg_zline",
        boundary=dict(inner_r="reflect", outer_r="vacuum", z="vacuum"),
    ),
    boundary=dict(
        inner_r="reflect", outer_r="vacuum", bottom_z="vacuum", top_z="vacuum"
    ),
)

Laser(enabled=False)
