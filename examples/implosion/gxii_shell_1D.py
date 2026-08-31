from tenryu_namelist import *

from math import sqrt, exp, log, pi

try:
    import numpy as np
except ImportError:
    import warnings

    warnings.warn("numpy not found; using scalar fallback")

    class _NumpyFallback:
        @staticmethod
        def sqrt(x):
            return sqrt(x)

        @staticmethod
        def where(cond, a, b):
            return a if cond else b

    np = _NumpyFallback()

um = 1.0e-4
ns = 1.0e-9

R_OUT = 250.0 * um
SHELL_THICKNESS = 7.0 * um
R_IN = R_OUT - SHELL_THICKNESS
R_MAX = 400.0 * um

# EOS and opacity: TMAT-H5 tables
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
mat_void = Material(name="VOID", A=1.0, Z=1.0, is_void=True)

Main(
    name="gxii_shell_1D",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=3.5 * ns,
    seed=12345,
    max_steps=10_000_000,
    verbosity="normal",
)

MESH_SEGMENTS = [
    {"r_start": 0.0,               "r_end": 200.0 * um,        "nr": 30},
    {"r_start": 200.0 * um,        "r_end": R_IN,              "nr": 50},
    {"r_start": R_IN,              "r_end": R_OUT - 2.0 * um,  "nr": 30},
    {"r_start": R_OUT - 2.0 * um,  "r_end": R_OUT,             "nr": 40},
    {"r_start": R_OUT,             "r_end": R_OUT + 20.0 * um, "nr": 60},
    {"r_start": R_OUT + 20.0 * um, "r_end": R_OUT + 60.0 * um, "nr": 50},
    {"r_start": R_OUT + 60.0 * um, "r_end": R_MAX,             "nr": 40},
]

AUTO_REGIONS = [
    {"r_end": R_IN,              "nz": 30, "rho_ref": 0.02},
    {"r_end": R_OUT - 2.0 * um,  "nz": 40, "rho_ref": 1.05, "material_group": "shell"},
    {"r_end": R_OUT,             "nz": 40, "rho_ref": 1.05, "material_group": "shell"},
    {"r_end": R_MAX,             "nz": 40, "rho_ref": 1.0e-10, "is_void": True},
]

Mesh(
    r_min=0.0,
    r_max=R_MAX,
    auto_regions=AUTO_REGIONS,
    auto_zone=dict(mass_ratio_max=1.3, dr_min=1.0e-6),
    motion="lagrangian",
    floors=dict(rho_floor_gcc=1.0e-9, Te_floor_eV=0.1, Ti_floor_eV=0.1),
)

Materials(
    materials=[mat_cd, mat_void],
    opacity_mix_rule="linear_mass",
    zbar=dict(model="fixed", fixed_value=3.5),
    void_config=dict(rho=1.0e-9, Te=0.1, Ti=0.1),
)


def vf_cd(r_cm, z_cm=0.0):
    return np.where(r_cm <= R_OUT, 1.0, 0.0)


def vf_void(r_cm, z_cm=0.0):
    return np.where(r_cm > R_OUT, 1.0, 0.0)


def rho_profile(r_cm, z_cm=0.0):
    if r_cm <= R_IN:
        return 0.02
    if r_cm <= R_OUT:
        return 1.05
    return 1.0e-10


def Te_profile(r_cm, z_cm=0.0):
    return 0.2


def Ti_profile(r_cm, z_cm=0.0):
    return 0.2


Geometry(
    rho=rho_profile,
    Te=Te_profile,
    Ti=Ti_profile,
    volfrac=dict(CD=vf_cd, VOID=vf_void),
    radiation_field="equilibrium",
    enforce_sum_to_one=True,
)

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    group_repack_hard_xray=True,
    multigroup_diffusion=dict(
        hydro_coupling="none",  # explicit opt-out: compatible_energy x gamma_r_43 unsupported (v1)
        flux_limiter="levermore_pomraning",
        max_outer_iterations=20,
        outer_tol=1.0e-5,
        boundary=dict(inner_r="reflect", outer_r="vacuum"),
    ),
    boundary=dict(inner_r="reflect", outer_r="vacuum"),
)


# Gaussian pulse: FWHM = 1.3 ns, total energy = 300 J x 12 = 3600 J
FWHM = 1.3 * ns
T_CENTER = 1.3 * ns
E_TOTAL = 300.0 * 12  # J (total laser energy)
# P_peak such that integral of Gaussian = E_TOTAL
P_PEAK = E_TOTAL / (FWHM * sqrt(pi / (4.0 * log(2.0))))


def power_gaussian(t_s):
    return P_PEAK * exp(-4.0 * log(2.0) * ((t_s - T_CENTER) / FWHM) ** 2)


Laser(
    enabled=True,
    wavelength_nm=527.0,
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
    deposit=dict(deposit_smooth_passes=3, deposit_smooth_alpha=0.25),
    beams=[
        LaserBeam(
            name="beam_00",
            direction=(0.0, 0.0, -1.0),
            power=power_gaussian,
            f_number=3.0,
            focus=(0.0, 0.0, -750.0 * um),
            profile=dict(model="super_gaussian", w0_um=250.0, m=4),
        )
    ],
)

Numerics(
    dt=dict(
        initial_s=1.0e-12,
        cfl_hydro=0.3,
        cfl_cond=0.25,
        f_min_fleck=0.01,
        max_s=1.0e-10,
    ),
    hydro=dict(
        boundary_1d="free",
        av_heat_C=0.5,
        av_heat_to="ion",
        odd_even_damping_C=1.0,
        ee_odd_even_C=0.15,
    ),
    conduction=dict(
        enabled=True,
        solver="sts",
        f_lim=0.06,
    ),
    positivity=dict(clamp=True),
    safety=dict(energy_threshold=1.0e-3, nan_fatal=True),
    diagnostics_every=100,
)

Output(
    directory="outputs/gxii_shell_1D",
    format="hdf5",
    plot_every_s=50.0e-12,
    history_every_s=1.0e-9,
    checkpoint_every=2000,
    checkpoint_keep_last=2,
    save_namelist_copy=True,
    save_frozen_config=True,
)

Diagnostics(
    enabled=True,
    every=1,
    energy_budget=dict(enabled=True, warn_threshold=1.0e-3),
    areal_density=dict(enabled=True, angles_deg=[0.0]),
    sphericity=dict(enabled=True, modes=[0, 2, 4]),
    laser_pattern=dict(enabled=True, per_beam=True),
    mc_stats=dict(enabled=True),
)
