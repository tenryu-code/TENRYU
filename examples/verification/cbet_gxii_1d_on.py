"""CBET G3b tendency gate deck (CBET ON, no detuning): GXII-like CD sphere, 2 beams,
short window, radiation disabled (laser+hydro+conduction only). Derived from
gxii_square_120J_6ns_fld_nr600.py; goldens untouched (new deck, gate-only)."""
from tenryu_namelist import *

um = 1.0e-4
ns = 1.0e-9

R_SPHERE = 100.0 * um

CBET_ENABLE = True
DELTA_LAMBDA_NM = 0.0

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
    name="cbet_gxii_1d_on",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=1.2 * ns,
    seed=12345,
    max_steps=10_000_000,
    verbosity="normal",
)

Mesh(
    r_min=0.0,
    r_max=R_SPHERE,
    grid=dict(
        type="graded",
        segments=[{"r_start": 0.0, "r_end": R_SPHERE, "nr": 400}],
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

PULSE_DURATION = 6.0 * ns
E_TOTAL = 120.0
P_CONST = E_TOTAL / PULSE_DURATION


def power_half(t_s):
    return 0.5 * P_CONST if 0.0 <= t_s <= PULSE_DURATION else 0.0


Radiation(enabled=False)

Laser(
    enabled=True,
    wavelength_nm=527.0,
    mode="raytrace_2d",
    rays_per_beam=4000,
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
    cbet=dict(
        enable=CBET_ENABLE,
        n_impact_bins=16,
        n_phi=8,
        tol=1.0e-3,
        max_iters=50,
    ),
    beams=[
        LaserBeam(
            name="beam_00",
            direction=(0.0, 0.0, -1.0),
            power=power_half,
            f_number=3.0,
            focus=(0.0, 0.0, -500.0 * um),
            profile=dict(model="super_gaussian", w0_um=250.0, m=4),
            delta_lambda_nm=+0.5 * DELTA_LAMBDA_NM,
        ),
        LaserBeam(
            name="beam_01",
            direction=(0.0, 0.0, -1.0),
            power=power_half,
            f_number=3.0,
            focus=(0.0, 0.0, -500.0 * um),
            profile=dict(model="super_gaussian", w0_um=250.0, m=4),
            delta_lambda_nm=-0.5 * DELTA_LAMBDA_NM,
        ),
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
    diagnostics_every=100,
)

Output(
    directory="outputs/cbet_gxii_1d_on",
    format="hdf5",
    plot_every_s=1.0e-9,
    history_every_s=50.0e-12,
    checkpoint_every=0,
    save_namelist_copy=False,
    save_frozen_config=True,
)

Diagnostics(
    enabled=True, every=1,
    energy_budget=dict(enabled=True, warn_threshold=1.0e-3),
    laser_pattern=dict(enabled=True, per_beam=True),
)
