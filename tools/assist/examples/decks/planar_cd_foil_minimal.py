from tenryu_namelist import *

# Minimal complete 1D_SPH deck: a 25 um CD foil driven by a 351 nm square pulse
# (1e14 W/cm^2, 100 ps rise, 1.2 ns), planar 1 cm^2 column, radiation off.
# The mesh block is a plain uniform grid (0.1 um cells, interface on a node);
# production decks take the mesh from the tenryu-mesh-1d skill (recommend-mesh).
# Table paths resolve from the launch directory: run from the repository root,
# which holds TMAT-H5/CD.tmat.h5. The inner node (x = 0) is a fixed wall in every
# geometry: the shock arrives at the rear face at the right time and reflects there.

um = 1.0e-4
ns = 1.0e-9

X_FOIL = 25.0 * um   # rear face on the fixed inner wall at x = 0, laser-facing face at X_FOIL
X_MAX = 125.0 * um   # 100 um of void padding on the laser side

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
    name="planar_cd_foil_minimal",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=1.2 * ns,
    seed=12345,
    max_steps=10_000_000,
    verbosity="normal",
)

Mesh(
    r_min=0.0,
    r_max=X_MAX,
    geometry_1d="planar",
    nr=1250,
    motion="lagrangian",
    floors=dict(rho_floor_gcc=1.0e-9, Te_floor_eV=0.1, Ti_floor_eV=0.1),
)

Materials(
    materials=[mat_cd, mat_void],
    opacity_mix_rule="linear_mass",
    zbar=dict(model="fixed", fixed_value=3.5),
    void_config=dict(rho=1.0e-9, Te=1.0, Ti=1.0),
)


def vf_cd(x_cm, z_cm=0.0):
    return 1.0 if x_cm < X_FOIL else 0.0


def vf_void(x_cm, z_cm=0.0):
    return 0.0 if x_cm < X_FOIL else 1.0


def rho_profile(x_cm, z_cm=0.0):
    return 1.05 if x_cm < X_FOIL else 1.0e-9


Geometry(
    rho=rho_profile,
    Te=lambda x_cm, z_cm=0.0: 1.0,
    Ti=lambda x_cm, z_cm=0.0: 1.0,
    volfrac=dict(CD=vf_cd, VOID=vf_void),
    enforce_sum_to_one=True,
)

Radiation(enabled=False)


def laser_power(t_s):
    # Planar 1 cm^2 convention: P[W] = I[W/cm^2] * 1 cm^2.
    peak_power_W = 1.0e14
    if t_s <= 0.0:
        return 0.0
    if t_s < 0.10 * ns:
        return peak_power_W * t_s / (0.10 * ns)
    if t_s <= 1.20 * ns:
        return peak_power_W
    return 0.0


Laser(
    enabled=True,
    wavelength_nm=351.0,
    mode="radial_absorption_1d",
    rays_per_beam=8000,
    absorption=dict(model="inverse_bremsstrahlung"),
    # The foil starts as bare solid, far above the critical density: the radial
    # integration stops at the first critical cell, so without this laser-only ghost
    # corona the laser would deposit nothing (NUMERICS §5.4a, §5.7.5). Same settings as
    # examples/laser_plasma_1d/ex01_cd_foil_breakout.py.
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
            direction=(-1.0, 0.0, 0.0),
            power=laser_power,
            f_number=3.0,
            focus=(0.0, 0.0, 0.0),
            profile=dict(model="super_gaussian", w0_um=250.0, m=4),
        )
    ],
    cbet=dict(enable=False),
    hot_electron=dict(enable=False),
)

Burn(enabled=False)

Numerics(
    dt=dict(initial_s=1.0e-13, max_s=1.0e-10, cfl_hydro=0.3, cfl_cond=0.25),
    hydro=dict(boundary_1d="free", driver_full_step_retry_enabled=True),
    conduction=dict(enabled=True, solver="implicit", f_lim=0.06),
    positivity=dict(clamp=True),
    safety=dict(nan_fatal=True),
    diagnostics_every=100,
)

Output(
    directory="outputs/planar_cd_foil_minimal",
    format="hdf5",
    plot_every_s=1.0e-11,
    history_every_s=1.0e-11,
    checkpoint_every=5000,
    checkpoint_keep_last=2,
    save_namelist_copy=True,
    save_frozen_config=True,
)

Diagnostics(enabled=True, every=1, energy_budget=dict(enabled=True, warn_threshold=1.0e-3))
