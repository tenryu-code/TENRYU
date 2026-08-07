# MPI LaserMesh gather / partition-of-unity gate deck (VERIFICATION
# §16.6, M18d). 2D RZ raytrace_3d: one axial beam (e_z, F/3, 1 TW) onto
# a CH sphere (R_out = 0.025 cm, rho = 1.05 g/cm3) in a thin background.
# Hydro 200x400 over R in [0, 0.05] cm, Z in [-0.05, 0.05] cm; LaserMesh
# 256x128 (approved §16.6 text). Under P>1 each rank contributes only
# its owned hydro cells to the LM projection (partition-of-unity,
# NUMERICS §12.4.2) and the Allreduce(SUM) assembles the identical
# owner-true medium on every rank, so:
#   - TENRYU_LM_DEBUG_DUMP=<prefix> files must be machine-exact across
#     ranks and vs the P=1 run;
#   - the raytrace and the HydroMesh deposit follow bit-for-bit
#     (owned-slab dumps via TENRYU_MPI_DUMP_OWNED, compare rtol 0).
# Spec docs/design/mpi_m18d_laser_burn_spec.md §2 d1.
#
# Env knobs:
#   TENRYU_PLM2D_NR / TENRYU_PLM2D_NZ  hydro mesh (default 200 x 400)
#   TENRYU_PLM2D_RAYS                  rays per beam (default 64)
#   TENRYU_PLM2D_MAXSTEPS              max_steps (default 3)
#   TENRYU_PLM2D_OUTDIR                output directory

import math
import os

from tenryu_namelist import *

NR = int(os.environ.get("TENRYU_PLM2D_NR", "200"))
NZ = int(os.environ.get("TENRYU_PLM2D_NZ", "400"))
RAYS = int(os.environ.get("TENRYU_PLM2D_RAYS", "64"))
MAX_STEPS = int(os.environ.get("TENRYU_PLM2D_MAXSTEPS", "3"))
OUTDIR = os.environ.get(
    "TENRYU_PLM2D_OUTDIR", "./output_parallel_lasermesh_2d"
)

R_MIN, R_MAX = 0.0, 0.05
Z_MIN, Z_MAX = -0.05, 0.05

_A_CH = 6.5
_Z_CH = 3.5
_GAMMA = 5.0 / 3.0
_RHO_SHELL = 1.05
_RHO_BG = 1.0e-4
_R_OUT = 0.025
_T_INIT_EV = 5.0

Main(
    name="parallel_lasermesh_2d",
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=1.0e-9,
    max_steps=MAX_STEPS,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=R_MIN,
    r_max=R_MAX,
    z_min=Z_MIN,
    z_max=Z_MAX,
    nr=NR,
    nz=NZ,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="ch",
            A=_A_CH,
            Z=_Z_CH,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=_GAMMA)),
            opacity=dict(
                model="constant", kappa_a=0.0, kappa_s=0.0,
                units="cm2_per_g",
            ),
        )
    ],
    zbar=dict(model="fixed", fixed_value=_Z_CH),
)


def _rho_init(r_cm, z_cm):
    radius = math.hypot(r_cm, z_cm)
    return _RHO_SHELL if radius < _R_OUT else _RHO_BG


Geometry(
    volfrac=dict(ch=lambda r, z: 1.0),
    rho=_rho_init,
    Te=lambda r, z: _T_INIT_EV,
    Ti=lambda r, z: _T_INIT_EV,
    velocity=lambda r, z: (0.0, 0.0),
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=1.0e-13,
        max_s=1.0e-13,
        min_s=1.0e-20,
        growth_factor=1.0,
    ),
    hydro=dict(
        boundary_2d=dict(
            r_inner="axis", r_outer="free", z_bottom="free", z_top="free"
        ),
    ),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-8, Te_floor_eV=1.0e-3,
                Ti_floor_eV=1.0e-3),
)

Radiation(enabled=False)


def _laser_power(t_s):
    return 1.0e12  # 1 TW, constant


Laser(
    enabled=True,
    wavelength_nm=351.0,
    mode="raytrace_3d",
    rays_per_beam=RAYS,
    absorption=dict(eps_n=1.0e-4, coulomb_log_floor=2.0),
    lasermesh=dict(
        nr=256, nz=128, r_max_factor=1.2, z_span_factor=1.2,
        critical_margin=0.9999,
    ),
    raytrace=dict(
        cfl_ray=0.8, eps_crit=1.0e-4, intensity_cutoff=1.0e-6,
        max_steps=50000,
    ),
    raytrace_skip_config=dict(enabled=False, threshold=0.01,
                              max_consecutive=10),
    deposit=dict(deposit_smooth_passes=1, deposit_smooth_alpha=0.25),
    beams=[
        LaserBeam(
            name="axial_top",
            direction=[0.0, 0.0, -1.0],
            focus=[0.0, 0.0, 0.0],
            f_number=3.0,
            profile=dict(model="super_gaussian", w0_um=150.0, m=4),
            power=_laser_power,
        )
    ],
)

Output(
    directory=OUTDIR,
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)
