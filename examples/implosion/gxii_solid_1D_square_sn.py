"""GXII solid CD sphere, 1D_SPH, square pulse 120J / 6ns, S_N radiation transport.

Validated reference configuration (CD R=100μm, ρ₀=1.05 g/cc).

Radiation: HYDRA-aligned discrete-ordinates S_N transport (NUMERICS §6.8).
- 80 groups (1 eV - 30 keV, hard-X-ray repacked)
- S_16 Gauss-Legendre angular quadrature (default; S_8 allowed via gate test)
- Pure-SN Fleck bypass (f=1, σ_s_eff=0; raw σ^PA absorption, raw σ^PE emission)
- Morel-Adams angular redistribution + metric scaling (Cut-FIX-2/3)
- DSA acceleration (cuSPARSE tridiag)
- PA/PE-consistent matter Newton with TMAT-aware electron EOS

Hydro: pure Lagrangian (1D ALE removed in ALE-FIX-1).

Reference run: GXII solid 1D_SPH square 120J/6ns SN ALE-off matched TMAT Hugoniot
within 0.4% on shock front compression (peak ρ₂/ρ₁ ≈ 19.5× at t=4.10 ns,
T_e_center ≈ 66.2 eV at peak compression).
"""
from tenryu_namelist import *

um = 1.0e-4
ns = 1.0e-9

R_SPHERE = 100.0 * um  # 200 μm diameter solid CD sphere

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

Main(
    name="gxii_solid_1D_square_sn",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=8.0 * ns,
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

# Square pulse: constant power for 6 ns, integrated to 120 J
PULSE_DURATION = 6.0 * ns
E_TOTAL = 120.0
P_CONST = E_TOTAL / PULSE_DURATION


def power_square(t_s):
    return P_CONST if 0.0 <= t_s <= PULSE_DURATION else 0.0


# HYDRA-aligned discrete-ordinates S_N transport (NUMERICS §6.8)
# Pure SN: IMC/DDMC/HOLO/difference all disabled
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
            enabled=True, n_out=12, ne_min_frac=0.03, ne_max_frac=0.99,
            Te_min_eV=50.0, zbar_min=1.0, zbar_max=4.0,
            handoff_cells=6, handoff_decay=2.0,
            transition_enabled=True, transition_resolved_nhat=0.9,
            transition_resolved_cells=3, transition_density_exponent=1.0,
        ),
    ),
    raytrace=dict(
        ds_adapt_g_target=0.01,
        ds_adapt_tau_target=0.01,
        ds_adapt_max_factor=2.0,
    ),
    deposit=dict(deposit_smooth_passes=1, deposit_smooth_alpha=0.25),
    beams=[
        LaserBeam(
            name="beam_00",
            direction=(0.0, 0.0, -1.0),
            power=power_square,
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
    directory="outputs/gxii_solid_1D_square_sn",
    format="hdf5",
    plot_every_s=25.0e-12,
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
