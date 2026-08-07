# MPI rank-count agreement gate deck (VERIFICATION §16.3, M18c).
# 2D RZ multigroup FLD: a hot disk at the domain center diffuses radiation
# radially ACROSS the r-slab rank interface (and in z), with 2T matter
# coupling active. Owned-slab dumps from P in {2,4} and P=1 runs are
# compared field-max-normalized (tools/validation/compare_owned_dump.py
# --rtol). Two deck classes (VERIFICATION §16.3):
#   diffusion-limit (default, kappa_a=50): stiff hot-disk edge; gate =
#     two-tier — steps=1 <=1e-8 (defect detector) and steps=25 <=1e-4
#     (trajectory band; CG dot-reorder seeds ~1e-12, the stiff Newton
#     coupling amplifies to ~1e-6 saturation — design doc §6h).
#   thin-corona (TENRYU_PFLD2D_KAPPA=0.05): well-conditioned; single tier
#     <=1e-8 at both steps=1 and steps=25.
# NOTE: max_outer_iterations=50 with cap-exit policy=warn is a DESIGN
# FEATURE of this deck: the early stiff steps cap out identically on all
# ranks, keeping the outer operation sequence rank-uniform; raising the
# cap makes iteration counts data-dependent and the 25-step comparison
# chaos-dominated (measured: cap=400 -> 1e-4..1e-2). Do not "fix" the cap.
# Tier numbers are provisional pending user ratification — design doc
# docs/design/mpi_m18_20_20260717.md §6h; spec mpi_m18c_fld2d_cg_spec.md.
#
# Env knobs:
#   TENRYU_PFLD2D_SOLVER    linear_solver_2d (default cusparse_cg_zline;
#                           jacobi / cusparse_cg_rgmg for policy probes —
#                           rgmg under P>1 falls back to zline by design)
#   TENRYU_PFLD2D_ACCEL     outer_accel (default anderson — exercises the
#                           masked Gram dots)
#   TENRYU_PFLD2D_KAPPA     kappa_a in cm2/g (default 50.0 diffusion-limit
#                           class; 0.05 = thin-corona class)
#   TENRYU_PFLD2D_MAXSTEPS  max_steps (default 25; tier-1 uses 1)
#   TENRYU_PFLD2D_OUTDIR    output directory

import os

from tenryu_namelist import *

_SOLVER = os.environ.get("TENRYU_PFLD2D_SOLVER", "cusparse_cg_zline")
_ACCEL = os.environ.get("TENRYU_PFLD2D_ACCEL", "anderson")
_OUTDIR = os.environ.get(
    "TENRYU_PFLD2D_OUTDIR", "./output_parallel_fld_2d"
)
_MAX_STEPS = int(os.environ.get("TENRYU_PFLD2D_MAXSTEPS", "25"))
_KAPPA = float(os.environ.get("TENRYU_PFLD2D_KAPPA", "50.0"))

Main(
    name="parallel_fld_2d",
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
                kappa_a=_KAPPA,
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
        outer_accel=_ACCEL,
        linear_solver_2d=_SOLVER,
        boundary=dict(inner_r="reflect", outer_r="vacuum", z="vacuum"),
    ),
    boundary=dict(
        inner_r="reflect", outer_r="vacuum", bottom_z="vacuum", top_z="vacuum"
    ),
)

Laser(enabled=False)
