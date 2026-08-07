"""GXII solid CD sphere, 1D_SPH, with mechanism-resolved hot-electron channels.

Demonstrates Laser.hot_electron.sources (NUMERICS §5.11.1; design
docs/design/hote_directional_sources_20260710.md): a TPD ring channel
captured at quarter critical plus a forward SRS channel captured in the
sub-quarter-critical band. eta / T_hot / angle knobs are CALIBRATION inputs;
the values below are literature-plausible placeholders, not predictions.

References: Rosenberg 2018 PRL 120.055001 (SRS 0.15-0.22 nc, T_h 45-55 keV);
Rosenberg 2020 PoP 27.042705; Solodov 2020 PoP 27.052706 (f_hot 1-5%,
T_h 37-48 keV); Vu 2012 PoP 19.102703 + Rovere 2023 PoP 30.042104 (TPD
T_h 60-100 keV; the >50 keV tail is forward +-25-30 deg in kinetic runs --
reproduce that operating point with tpd_theta_deg=15, tpd_delta_deg=15).

Pulse: Gaussian, FWHM=1.3 ns, peak at t=1.3 ns, integrated energy
  E_TOTAL = 300 J x 12 beams = 3600 J, 1 effective beam in 1D_SPH.

Radiation: HYDRA-aligned discrete-ordinates S_N transport (NUMERICS §6.8).
"""
from tenryu_namelist import *
from math import sqrt, exp, log

um = 1.0e-4
ns = 1.0e-9

R_SPHERE = 100.0 * um  # 200 um diameter solid CD sphere

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

Main(
    name="hot_electron_tpd_srs_1d",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=2.5 * ns,
    seed=12345,
    max_steps=10_000_000,
    verbosity="normal",
)

Mesh(
    r_min=0.0,
    r_max=R_SPHERE,
    grid=dict(
        type="graded",
        segments=[{"r_start": 0.0, "r_end": R_SPHERE, "nr": 300}],
        grading=dict(edge_ratio=0.25),
    ),
    floors=dict(rho_floor_gcc=1.0e-9, Te_floor_eV=0.1, Ti_floor_eV=0.1),
)

Materials(
    materials=[mat_cd],
    opacity_mix_rule="linear_mass",
    zbar=dict(model="tabular"),
)


def rho_profile(r_cm, z_cm=0.0):
    return 1.05


def Te_profile(r_cm, z_cm=0.0):
    return 1.0


def Ti_profile(r_cm, z_cm=0.0):
    return 1.0


Geometry(
    rho=rho_profile,
    Te=Te_profile,
    Ti=Ti_profile,
    volfrac=dict(CD=lambda r, z=0.0: 1.0),
    radiation_field="equilibrium",
    enforce_sum_to_one=True,
)

Radiation(
    enabled=True,
    mode="sn_transport",
    group_repack_hard_xray=True,
    sn_transport=dict(
        n_angles=16,
        max_outer_iterations=20,
        max_inner_iterations=100,
        outer_tol=1.0e-5,
        inner_tol=1.0e-6,
        dsa_enabled=True,
        boundary=dict(inner_r="reflect_parity", outer_r="vacuum"),
    ),
    boundary=dict(inner_r="reflect", outer_r="vacuum"),
)


FWHM = 1.3 * ns
T_CENTER = 1.3 * ns
E_TOTAL = 300.0 * 12  # J
P_PEAK = E_TOTAL / (FWHM * sqrt(3.141592653589793 / (4.0 * log(2.0))))


def power_gaussian(t_s):
    return P_PEAK * exp(-4.0 * log(2.0) * ((t_s - T_CENTER) / FWHM) ** 2)


Laser(
    enabled=True,
    wavelength_nm=527.0,
    mode="raytrace_2d",
    rays_per_beam=2000,
    ray_output_count=0,
    ray_output_trajectory=False,
    # Mechanism channels (Laser.hot_electron.sources, NUMERICS §5.11.1):
    # each channel captures at its own n_e = capture_nc_fraction * n_crit
    # surface with its own eta / T_hot / angular band. Channels fire in
    # ascending capture-fraction order with sequential (1-eta) ray-power
    # depletion. The scalar v1 keys remain the single-cone shorthand and
    # cannot be combined with sources.
    # QoI: snapshots write hydro/hot_e_eps_cum [erg/g]; histories write the
    # laser/hot_e_* totals plus per-channel laser/hot_e_ch{i}_* series.
    hot_electron=dict(
        enable=True,
        angular_model="cone",
        sources=[
            dict(
                mechanism="srs",       # forward cone, default 20 deg
                eta=0.01,              # calibration input
                T_hot_eV=4.5e4,        # Solodov 2020 / Rosenberg 2018 band
                capture_nc_fraction=0.18,
            ),
            dict(
                mechanism="tpd",       # ring 45+-10 deg around the ray
                eta=0.01,              # calibration input
                T_hot_eV=6.0e4,        # Vu 2012 / Rovere 2023 band
                # Rovere 2023 kinetic-tail alternative:
                # tpd_theta_deg=15.0, tpd_delta_deg=15.0,
            ),
        ],
    ),
    lasermesh=dict(
        mesh_factor=0.1,
        rmax_n_hat_threshold=0.001,
        ghost_corona=dict(
            enabled=True, n_out=12, ne_min_frac=0.03, ne_max_frac=0.99,
            Te_min_eV=50.0, zbar_min=1.0, zbar_max=4.0,
            handoff_cells=6, handoff_decay=2.0,
            transition_enabled=True, transition_resolved_nhat=0.9,
            transition_resolved_cells=3, transition_density_exponent=1.0,
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
            focus=(0.0, 0.0, -500.0 * um),
            profile=dict(model="super_gaussian", w0_um=250.0, m=4),
        )
    ],
)

Numerics(
    dt=dict(
        initial_s=1.0e-13, cfl_hydro=0.3, cfl_cond=0.25,
        f_min_fleck=0.01, max_s=1.0e-10, min_s=1.0e-22,
    ),
    hydro=dict(
        boundary_1d="free",
        compatible_energy=True,
        av_type="csw",
        csw_shock_limiter_floor=0.0,
        av_heat_C=0.5, av_heat_to="ion",
        odd_even_damping_C=1.0, ee_odd_even_C=0.5,
    ),
    conduction=dict(enabled=True, solver="implicit", f_lim=0.06),
    positivity=dict(clamp=True),
    safety=dict(energy_threshold=1.0, nan_fatal=True),
    radiation_thermal_subcycle=True,
    diagnostics_every=100,
)

Output(
    directory="./build/hot_electron_tpd_srs_1d/",
    format="hdf5",
    plot_every_s=5.0e-11,
    history_every_s=1.0e-9,
    checkpoint_every=2000,
    checkpoint_keep_last=2,
    save_namelist_copy=True,
    save_frozen_config=True,
)

Diagnostics(
    enabled=True, every=1,
    energy_budget=dict(enabled=True, warn_threshold=1.0e-3),
    areal_density=dict(enabled=True, angles_deg=[0.0]),
    sphericity=dict(enabled=True, modes=[0, 2, 4]),
    laser_pattern=dict(enabled=True, per_beam=True),
    mc_stats=dict(enabled=True),
)
