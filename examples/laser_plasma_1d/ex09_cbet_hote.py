from tenryu_namelist import *
from math import sqrt

# EX-09 full-physics arm: the foot builds a >= 100 um scale-length corona;
# CBET redistributes absorption during the main drive; the 3.1-3.4 ns spike
# exceeds TPD/SRS thresholds and deposits hot-electron preheat. The A/B
# companions (CBET off and hot-e off) are separate decks.

um = 1.0e-4
ns = 1.0e-9
TW = 1.0e12

R_TARGET = 500.0 * um
R_MAX = 1000.0 * um

# EOS and opacity: TMAT-H5 CD table; GXII-style transparent void.
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
    name="lp1d_ex09_cbet_hote",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=6.0 * ns,
    seed=12345,
    max_steps=10_000_000,
    verbosity="normal",
)

# 430 + 470 = 900 equal-mass CD cells. The outer CD region has modestly
# smaller cell mass; the is_void region is exactly 400 uniform-width cells.
AUTO_REGIONS = [
    {
        "r_end": 400.0 * um,
        "nz": 430,
        "rho_ref": 1.05,
        "material_group": "CD",
    },
    {
        "r_end": R_TARGET,
        "nz": 470,
        "rho_ref": 1.05,
        "material_group": "CD",
    },
    {
        "r_end": R_MAX,
        "nz": 400,
        "rho_ref": 1.0e-9,
        "is_void": True,
    },
]

Mesh(
    r_min=0.0,
    r_max=R_MAX,
    geometry_1d="spherical",
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
    return 1.0 if r_cm <= R_TARGET else 0.0


def vf_void(r_cm, z_cm=0.0):
    return 1.0 if r_cm > R_TARGET else 0.0


def rho_profile(r_cm, z_cm=0.0):
    return 1.05 if r_cm <= R_TARGET else 1.0e-9


def Te_profile(r_cm, z_cm=0.0):
    return 1.0 if r_cm <= R_TARGET else 0.1


def Ti_profile(r_cm, z_cm=0.0):
    return 1.0 if r_cm <= R_TARGET else 0.1


Geometry(
    rho=rho_profile,
    Te=Te_profile,
    Ti=Ti_profile,
    volfrac=dict(CD=vf_cd, VOID=vf_void),
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

# Total piecewise-linear laser power. Its integral is 25.65 kJ.
PULSE_POINTS = (
    (0.00 * ns, 0.00 * TW),
    (0.20 * ns, 1.00 * TW),
    (1.20 * ns, 1.00 * TW),
    (1.50 * ns, 8.00 * TW),
    (2.70 * ns, 8.00 * TW),
    (2.90 * ns, 0.00 * TW),
    (3.00 * ns, 0.00 * TW),
    (3.10 * ns, 32.0 * TW),
    (3.40 * ns, 32.0 * TW),
    (3.50 * ns, 0.00 * TW),
)


def laser_power_total(t_s):
    if t_s < PULSE_POINTS[0][0] or t_s > PULSE_POINTS[-1][0]:
        return 0.0
    for index in range(len(PULSE_POINTS) - 1):
        t0, p0 = PULSE_POINTS[index]
        t1, p1 = PULSE_POINTS[index + 1]
        if t_s <= t1:
            fraction = (t_s - t0) / (t1 - t0)
            return p0 + fraction * (p1 - p0)
    return 0.0


B_INNER = 0.2
B_MIDDLE = 0.6
B_OUTER = 0.9

Laser(
    enabled=True,
    wavelength_nm=351.0,
    mode="raytrace_2d",
    rays_per_beam=2000,
    ray_output_count=200,
    ray_output_trajectory=True,
    hot_electron=dict(
        enable=True,
        angular_model="cone",
        eta_mode="model",
        eta_model=dict(eta_total_cap=0.03),
        subtract_from_laser=True,
        sources=[
            dict(
                mechanism="srs",
                T_hot_eV=4.5e4,
                capture_nc_fraction=0.18,
            ),
            dict(
                mechanism="tpd",
                T_hot_eV=6.0e4,
                capture_nc_fraction=0.25,
            ),
        ],
    ),
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
    cbet=dict(
        enable=True,
        geometry_mode="port_section",
        n_section_phi=16,
        n_impact_bins=16,
    ),
    # One geometric prototype supplies the common focus, f-number, profile,
    # and total P(t). Port weights produce the 0.25/0.50/0.25 power split.
    beams=[
        LaserBeam(
            name="beam_prototype",
            direction=(0.0, 0.0, -1.0),
            power=laser_power_total,
            f_number=3.0,
            focus=(0.0, 0.0, -2000.0 * um),
            profile=dict(model="super_gaussian", w0_um=500.0, m=4),
        )
    ],
    port_configuration=dict(
        normalization="sum_weights_one",
        ports=[
            # Nominal b/R = 0.2 group.
            dict(
                port_id=0,
                direction=(
                    B_INNER,
                    0.0,
                    -sqrt(1.0 - B_INNER * B_INNER),
                ),
                power_weight=0.25,
            ),
            # Nominal b/R = 0.6 group, azimuthally separated by 120 degrees.
            dict(
                port_id=1,
                direction=(
                    -0.5 * B_MIDDLE,
                    0.5 * sqrt(3.0) * B_MIDDLE,
                    -sqrt(1.0 - B_MIDDLE * B_MIDDLE),
                ),
                power_weight=0.50,
            ),
            # Nominal b/R = 0.9 group, azimuthally separated by 240 degrees.
            dict(
                port_id=2,
                direction=(
                    -0.5 * B_OUTER,
                    -0.5 * sqrt(3.0) * B_OUTER,
                    -sqrt(1.0 - B_OUTER * B_OUTER),
                ),
                power_weight=0.25,
            ),
        ],
    ),
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
    directory="outputs/lp1d_ex09_cbet_hote",
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
    energy_budget=dict(enabled=True, warn_threshold=1.0e-3),
    areal_density=dict(enabled=True, angles_deg=[0.0]),
    sphericity=dict(enabled=True, modes=[0, 2, 4]),
    laser_pattern=dict(enabled=True, per_beam=True),
)
