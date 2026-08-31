from tenryu_namelist import *

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
TW = 1.0e12

R_CD = 250.0 * um
R_MAX = 450.0 * um

# EX-07 expected observables:
# Converging shock focuses at 2.8-4.2 ns, peak density 8-30 g/cc
# (3-cell average), followed by a reflected shock after focus.

mat_cd = Material(
    name="CD",
    A=7.0,
    Z=3.5,
    eos=dict(model="tmat", file="TMAT-H5/CD.tmat.h5"),
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
    name="lp1d_ex07_solid_sphere",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=5.0e-9,
    seed=12345,
    max_steps=10_000_000,
    verbosity="normal",
)

# Two uniform material segments: 720 CD cells over [0, 250 um]
# (dx = 0.347222 um) and 180 void cells over [250, 450 um]
# (dx = 1.111111 um), with the material interface on a cell face.
MESH_SEGMENTS = [
    {"r_start": 0.0, "r_end": R_CD, "nr": 720},
    {"r_start": R_CD, "r_end": R_MAX, "nr": 180},
]

Mesh(
    r_min=0.0,
    r_max=R_MAX,
    geometry_1d="spherical",
    grid=dict(
        type="graded",
        segments=MESH_SEGMENTS,
        grading=dict(
            edge_ratio=0.9999999999999999,
            sg_order=4,
            sg_sigma=0.2,
        ),
    ),
    motion="lagrangian",
    floors=dict(
        rho_floor_gcc=1.0e-9,
        Te_floor_eV=0.1,
        Ti_floor_eV=0.1,
    ),
)

Materials(
    materials=[mat_cd, mat_void],
    opacity_mix_rule="linear_mass",
    zbar=dict(model="fixed", fixed_value=3.5),
    void_config=dict(rho=1.0e-9, Te=0.1, Ti=0.1),
)


def vf_cd(r_cm, z_cm=0.0):
    return np.where(r_cm <= R_CD, 1.0, 0.0)


def vf_void(r_cm, z_cm=0.0):
    return np.where(r_cm > R_CD, 1.0, 0.0)


def rho_profile(r_cm, z_cm=0.0):
    if r_cm <= R_CD:
        return 1.05
    return 1.0e-9


def Te_profile(r_cm, z_cm=0.0):
    if r_cm <= R_CD:
        return 1.0
    return 0.1


def Ti_profile(r_cm, z_cm=0.0):
    if r_cm <= R_CD:
        return 1.0
    return 0.1


Geometry(
    rho=rho_profile,
    Te=Te_profile,
    Ti=Ti_profile,
    volfrac=dict(
        CD=vf_cd,
        VOID=vf_void,
    ),
    radiation_field="zero",
    enforce_sum_to_one=True,
)

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    group_repack_hard_xray=True,
    multigroup_diffusion=dict(
        hydro_coupling="none",
        flux_limiter="levermore_pomraning",
        max_outer_iterations=20,
        outer_tol=1.0e-5,
        boundary=dict(
            inner_r="reflect",
            outer_r="vacuum",
        ),
    ),
    boundary=dict(
        inner_r="reflect",
        outer_r="vacuum",
    ),
)

# Piecewise-linear 351 nm pulse; total laser energy is 2.82 kJ.
def laser_power(t_s):
    t_ns = t_s / ns

    if t_ns <= 0.0:
        return 0.0
    if t_ns < 0.20:
        return TW * (0.40 * t_ns / 0.20)
    if t_ns < 0.80:
        return TW * 0.40
    if t_ns < 1.00:
        return TW * (
            0.40
            + (2.50 - 0.40) * (t_ns - 0.80) / 0.20
        )
    if t_ns < 1.80:
        return TW * 2.50
    if t_ns < 2.00:
        return TW * (2.50 * (2.00 - t_ns) / 0.20)
    return 0.0


Laser(
    enabled=True,
    wavelength_nm=351.0,
    mode="raytrace_2d",
    rays_per_beam=8000,
    ray_output_count=200,
    ray_output_trajectory=True,
    lasermesh=dict(
        mesh_factor=0.1,
        rmax_n_hat_threshold=0.001,
        ghost_corona=dict(
            enabled=True,
            n_out=12,
            ne_min_frac=0.03,
            ne_max_frac=0.99,
            Te_min_eV=50.0,
            zbar_min=1.0,
            zbar_max=4.0,
            handoff_cells=6,
            handoff_decay=2.0,
            transition_enabled=True,
            transition_resolved_nhat=0.9,
            transition_resolved_cells=3,
            transition_density_exponent=1.0,
        ),
    ),
    raytrace=dict(
        ds_adapt_g_target=0.05,
        ds_adapt_tau_target=0.05,
        ds_adapt_max_factor=2.0,
    ),
    deposit=dict(
        deposit_smooth_passes=3,
        deposit_smooth_alpha=0.25,
    ),
    beams=[
        LaserBeam(
            name="beam_00",
            direction=(0.0, 0.0, -1.0),
            power=laser_power,
            f_number=3.0,
            focus=(0.0, 0.0, -750.0 * um),
            profile=dict(
                model="super_gaussian",
                w0_um=250.0,
                m=4,
            ),
        )
    ],
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
        av_heat_C=0.5,
        av_heat_to="ion",
        odd_even_damping_C=1.0,
        ee_odd_even_C=0.15,
        driver_full_step_retry_enabled=True,
        driver_full_step_retry_max_attempts=8,
    ),
    conduction=dict(
        enabled=True,
        solver="implicit",
        f_lim=0.06,
    ),
    positivity=dict(clamp=True),
    safety=dict(nan_fatal=True),
    diagnostics_every=100,
)

Output(
    directory="outputs/lp1d_ex07_solid_sphere",
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
    areal_density=dict(
        enabled=True,
        angles_deg=[0.0],
    ),
    sphericity=dict(
        enabled=True,
        modes=[0, 2, 4],
    ),
    laser_pattern=dict(
        enabled=True,
        per_beam=True,
    ),
)
