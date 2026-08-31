from tenryu_namelist import *

# EX-06: Cylindrical CD-liner compression of a D2 column (pure hydro).
# Expected sequence: the converging shock reaches the axis at about
# 1.5-2.7 ns, liner stagnation occurs at 2.5-4.0 ns with convergence
# ratio 4-8, and the compressed column subsequently rebounds.

try:
    import numpy as np
except ImportError:
    import warnings

    warnings.warn("numpy not found; using scalar fallback")

    class _NumpyFallback:
        @staticmethod
        def where(cond, a, b):
            return a if cond else b

    np = _NumpyFallback()


um = 1.0e-4
ns = 1.0e-9

R_D2 = 150.0 * um
R_LINER = 180.0 * um
R_MAX = 300.0 * um

mat_d2 = Material(
    name="D2",
    A=2.014,
    Z=1.0,
    eos=dict(
        model="sesame",
        file="SESAME/xsesame_short",
        sesame_material_id=5263,
    ),
    opacity=dict(
        model="constant",
        kappa_a=0.0,
        kappa_s=0.0,
    ),
)

mat_cd = Material(
    name="CD",
    A=7.0,
    Z=3.5,
    eos=dict(
        model="tmat",
        file="TMAT-H5/CD.tmat.h5",
    ),
    opacity=dict(
        model="tmat",
        file="TMAT-H5/CD.tmat.h5",
        lambda_method="finite_difference",
        lambda_fd_delta_rel=1.0e-4,
        lambda_fd_abs_min=1.0e-6,
        f_min=1.0e-4,
    ),
)

mat_void = Material(
    name="VOID",
    A=1.0,
    Z=1.0,
    is_void=True,
)

Main(
    name="lp1d_ex06_cyl_liner",
    dimension="1D_CYL",
    temperature_model="2T",
    t_end=4.5 * ns,
    seed=12345,
    max_steps=10_000_000,
    verbosity="normal",
)

# Six pin-delimited intervals with 150 cells each produce 450 uniform
# D2 cells, 300 uniform liner cells, and 150 uniform exterior cells.
# Width jumps are allowed only at the two material interfaces.
Mesh(
    r_min=0.0,
    r_max=R_MAX,
    zoning_intent=dict(
        n_cells=900,
        measure="width",
        pins=[
            {"r": 50.0 * um, "ratio_jump_allowed": False},
            {"r": 100.0 * um, "ratio_jump_allowed": False},
            {"r": R_D2, "ratio_jump_allowed": True},
            {"r": 165.0 * um, "ratio_jump_allowed": False},
            {"r": R_LINER, "ratio_jump_allowed": True},
        ],
        preferred_ratio=1.3,
        ratio_hard_max=2.0,
        min_cells_per_segment=150,
    ),
    motion="lagrangian",
    floors=dict(
        rho_floor_gcc=1.0e-9,
        Te_floor_eV=0.1,
        Ti_floor_eV=0.1,
    ),
)

Materials(
    materials=[mat_d2, mat_cd, mat_void],
    zbar=dict(
        model="fixed",
        fixed_value=1.75,
    ),
    void_config=dict(
        rho=1.0e-9,
        Te=1.0,
        Ti=1.0,
    ),
)


def vf_d2(r_cm, z_cm=0.0):
    return np.where(r_cm < R_D2, 1.0, 0.0)


def vf_cd(r_cm, z_cm=0.0):
    return np.where(
        r_cm < R_D2,
        0.0,
        np.where(r_cm <= R_LINER, 1.0, 0.0),
    )


def vf_void(r_cm, z_cm=0.0):
    return np.where(r_cm > R_LINER, 1.0, 0.0)


def rho_profile(r_cm, z_cm=0.0):
    if r_cm < R_D2:
        return 0.0050
    if r_cm <= R_LINER:
        return 1.05
    return 1.0e-9


def Te_profile(r_cm, z_cm=0.0):
    return 1.0


def Ti_profile(r_cm, z_cm=0.0):
    return 1.0


def velocity_profile(r_cm, z_cm=0.0):
    if r_cm < R_D2:
        return 0.0
    if r_cm <= R_LINER:
        return -5.0e6
    return 0.0


Geometry(
    rho=rho_profile,
    Te=Te_profile,
    Ti=Ti_profile,
    velocity=velocity_profile,
    volfrac=dict(
        D2=vf_d2,
        CD=vf_cd,
        VOID=vf_void,
    ),
    enforce_sum_to_one=True,
)

Radiation(
    enabled=False,
)

Laser(
    enabled=False,
)

Numerics(
    dt=dict(
        initial_s=1.0e-13,
        max_s=1.0e-10,
        cfl_hydro=0.3,
        cfl_cond=0.25,
    ),
    hydro=dict(
        boundary_1d="free",
        driver_full_step_retry_enabled=True,
    ),
    conduction=dict(
        enabled=False,
    ),
    positivity=dict(
        clamp=True,
    ),
    safety=dict(
        nan_fatal=True,
    ),
    diagnostics_every=100,
)

Output(
    directory="outputs/lp1d_ex06_cyl_liner",
    format="hdf5",
    plot_every_s=2.0e-11,
    history_every_s=1.0e-9,
    checkpoint_every=5000,
    checkpoint_keep_last=2,
    save_namelist_copy=True,
    save_frozen_config=True,
)

Diagnostics(
    enabled=True,
    every=1,
    energy_budget=dict(
        enabled=True,
        warn_threshold=1.0e-3,
    ),
)
