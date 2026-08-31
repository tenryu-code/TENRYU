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
TW = 1.0e12

R_D2 = 405.0 * um
R_SHELL_OUT = 430.0 * um
R_MAX = 650.0 * um

# EX-08 broad commissioning windows:
# - Three shocks merge in the gas at 2.0-3.0 ns.
# - Peak implosion velocity is 2.0e7-3.5e7 cm/s.
# - Stagnation is at 3.6-4.5 ns with convergence 10-20.
# - Peak rhoR is 0.08-0.20 g/cm^2.

mat_d2 = Material(
    name="D2",
    A=2.014,
    Z=1.0,
    eos=dict(model="tmat", file="TMAT-H5/D2.tmat.h5"),
    opacity=dict(model="constant", kappa_a=1.0, kappa_s=0.0, units="cm2_per_g"),
)

mat_cd = Material(
    name="CD",
    A=7.0,
    Z=3.5,
    eos=dict(model="tmat", file="TMAT-H5/CD_lte.tmat.h5"),
    # CD kappa_a: Planck mean of the CD table at 1.05 g/cc, 100-150 eV (surface-absorption regime).
    opacity=dict(model="constant", kappa_a=5.0e3, kappa_s=0.0, units="cm2_per_g"),
)

mat_void = Material(name="VOID", A=1.0, Z=1.0, is_void=True)

Main(
    name="lp1d_ex08_d2_shell_implosion",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=5.0 * ns,
    seed=12345,
    max_steps=10_000_000,
    verbosity="normal",
)

# The 360-cell CD shell uses 80 cells in each 5 um interface band
# and 200 cells in the central 15 um. The exterior VOID region is
# one auto-zone region, so all 160 of its cells have uniform width.
AUTO_REGIONS = [
    {
        "r_end": R_D2,
        "nz": 400,
        "rho_ref": 0.0016,
        "material_group": "target",
    },
    {
        "r_end": R_D2 + 5.0 * um,
        "nz": 80,
        "rho_ref": 1.05,
        "material_group": "target",
    },
    {
        "r_end": R_SHELL_OUT - 5.0 * um,
        "nz": 200,
        "rho_ref": 1.05,
        "material_group": "target",
    },
    {
        "r_end": R_SHELL_OUT,
        "nz": 80,
        "rho_ref": 1.05,
        "material_group": "target",
    },
    {
        "r_end": R_MAX,
        "nz": 160,
        "rho_ref": 1.0e-9,
        "is_void": True,
    },
]

# Fine shell-interface zoning is paired with the full-step snapshot
# and dt/2 retry path enabled in Numerics.hydro below.
Mesh(
    r_min=0.0,
    r_max=R_MAX,
    geometry_1d="spherical",
    auto_regions=AUTO_REGIONS,
    auto_zone=dict(mass_ratio_max=1.3, dr_min=5.0e-7),
    motion="lagrangian",
    floors=dict(
        rho_floor_gcc=1.0e-9,
        Te_floor_eV=0.1,
        Ti_floor_eV=0.1,
    ),
)

Materials(
    materials=[mat_d2, mat_cd, mat_void],
    opacity_mix_rule="linear_mass",
    zbar=dict(model="fixed", fixed_value=3.5),
    void_config=dict(rho=1.0e-9, Te=0.1, Ti=0.1),
)


def vf_d2(r_cm, z_cm=0.0):
    return np.where(r_cm < R_D2, 1.0, 0.0)


def vf_cd(r_cm, z_cm=0.0):
    return np.where(
        r_cm < R_D2,
        0.0,
        np.where(r_cm < R_SHELL_OUT, 1.0, 0.0),
    )


def vf_void(r_cm, z_cm=0.0):
    return np.where(r_cm < R_SHELL_OUT, 0.0, 1.0)


def rho_profile(r_cm, z_cm=0.0):
    return np.where(
        r_cm < R_D2,
        0.0016,
        np.where(r_cm < R_SHELL_OUT, 1.05, 1.0e-9),
    )


def Te_profile(r_cm, z_cm=0.0):
    return np.where(r_cm < R_SHELL_OUT, 1.0, 0.1)


def Ti_profile(r_cm, z_cm=0.0):
    return np.where(r_cm < R_SHELL_OUT, 1.0, 0.1)


Geometry(
    rho=rho_profile,
    Te=Te_profile,
    Ti=Ti_profile,
    volfrac=dict(D2=vf_d2, CD=vf_cd, VOID=vf_void),
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
        boundary=dict(inner_r="reflect", outer_r="vacuum"),
    ),
    boundary=dict(inner_r="reflect", outer_r="vacuum"),
)

# Piecewise-linear drive; trapezoidal integral = 15.9775 kJ (~16.0 kJ).
# Picket powers retuned 2026-08-29 (x0.25) after the per-material opacity fix
# (48bce971b): the original 4.7/3.8/3.8 TW pickets were auto-commissioned under
# the transparent-shell bug and over-drive the shell once the CD ablator
# absorbs coronal re-emission.
LASER_PULSE = [
    (0.00 * ns, 0.00 * TW),
    (0.10 * ns, 1.20 * TW),
    (0.25 * ns, 0.00 * TW),
    (0.40 * ns, 0.00 * TW),
    (0.55 * ns, 0.95 * TW),
    (0.70 * ns, 0.00 * TW),
    (0.85 * ns, 0.00 * TW),
    (1.00 * ns, 0.95 * TW),
    (1.15 * ns, 0.00 * TW),
    (1.30 * ns, 0.00 * TW),
    (1.55 * ns, 7.50 * TW),
    (1.80 * ns, 9.90 * TW),
    (2.80 * ns, 9.90 * TW),
    (3.05 * ns, 0.00 * TW),
]


def power_piecewise_linear(t_s):
    if t_s < LASER_PULSE[0][0] or t_s > LASER_PULSE[-1][0]:
        return 0.0

    for i in range(len(LASER_PULSE) - 1):
        t0, p0 = LASER_PULSE[i]
        t1, p1 = LASER_PULSE[i + 1]
        if t_s <= t1:
            fraction = (t_s - t0) / (t1 - t0)
            return p0 + fraction * (p1 - p0)

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
            power=power_piecewise_linear,
            f_number=3.0,
            focus=(0.0, 0.0, -3.0 * R_SHELL_OUT),
            profile=dict(
                model="super_gaussian",
                w0_um=430.0,
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
        f_lim=0.10,
    ),
    positivity=dict(clamp=True),
    safety=dict(nan_fatal=True),
    diagnostics_every=100,
)

Output(
    directory="outputs/lp1d_ex08_d2_shell_implosion",
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
