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

# Scientific target: density-impedance matching in a CD piston/foam/witness stack.
# Expected fast shock through 0.10 g/cc foam at 100-180 um/ns.
# Expected transmitted slower shock and reflected compression at the
# foam/witness interface.
# Expected witness breakout at t ~ 1.8-2.8 ns.
# Target interface pressure continuity is better than 5%.

um = 1.0e-4
ns = 1.0e-9

X_WITNESS_OUTER_FACE = 15.0 * um
X_FOAM_OUTER_FACE = 115.0 * um
X_PISTON_OUTER_FACE = 155.0 * um
X_MAX = 205.0 * um

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
    name="lp1d_ex03_impedance_match",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=3.5 * ns,
    seed=12345,
    max_steps=10_000_000,
    verbosity="normal",
)

# The four auto-zoned regions preserve the requested 850-cell allocation:
# 240 witness, 400 foam, 150 piston, and 60 uniform outer-void cells.
# Every material and density interface is an exact auto-region boundary.
Mesh(
    r_min=0.0,
    r_max=X_MAX,
    geometry_1d="planar",
    auto_regions=[
        {
            "r_end": X_WITNESS_OUTER_FACE,
            "nz": 240,
            "rho_ref": 1.05,
            "is_void": False,
            "material_group": "CD",
        },
        {
            "r_end": X_FOAM_OUTER_FACE,
            "nz": 400,
            "rho_ref": 0.10,
            "is_void": False,
            "material_group": "CD",
        },
        {
            "r_end": X_PISTON_OUTER_FACE,
            "nz": 150,
            "rho_ref": 1.05,
            "is_void": False,
            "material_group": "CD",
        },
        {
            "r_end": X_MAX,
            "nz": 60,
            "rho_ref": 1.0e-9,
            "is_void": True,
            "material_group": "VOID",
        },
    ],
    auto_zone=dict(
        mass_ratio_max=1.3,
        dr_min=1.0e-8,
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
    void_config=dict(
        rho=1.0e-9,
        Te=0.1,
        Ti=0.1,
    ),
)


def vf_void(x_cm, z_cm=0.0):
    return np.where(x_cm < X_PISTON_OUTER_FACE, 0.0, 1.0)


def vf_cd(x_cm, z_cm=0.0):
    return np.where(x_cm < X_PISTON_OUTER_FACE, 1.0, 0.0)


def rho_profile(x_cm, z_cm=0.0):
    if x_cm < X_WITNESS_OUTER_FACE:
        return 1.05
    if x_cm < X_FOAM_OUTER_FACE:
        return 0.10
    if x_cm < X_PISTON_OUTER_FACE:
        return 1.05
    return 1.0e-9


def Te_profile(x_cm, z_cm=0.0):
    if x_cm >= X_PISTON_OUTER_FACE:
        return 0.1
    return 1.0


def Ti_profile(x_cm, z_cm=0.0):
    if x_cm >= X_PISTON_OUTER_FACE:
        return 0.1
    return 1.0


Geometry(
    rho=rho_profile,
    Te=Te_profile,
    Ti=Ti_profile,
    volfrac=dict(CD=vf_cd, VOID=vf_void),
    enforce_sum_to_one=True,
)

Radiation(enabled=False)


def laser_power(t_s):
    peak_power_W = 3.0e14
    if t_s <= 0.0:
        return 0.0
    if t_s < 0.10 * ns:
        return peak_power_W * t_s / (0.10 * ns)
    if t_s <= 0.90 * ns:
        return peak_power_W
    if t_s < 1.00 * ns:
        return peak_power_W * (1.00 * ns - t_s) / (0.10 * ns)
    return 0.0


# Planar 1 cm^2 convention: P[W] = I[W/cm^2] * 1 cm^2;
# the resulting fluence is 2.70e5 J/cm^2.
# The beam points along -x from the outer boundary toward the target.
Laser(
    enabled=True,
    wavelength_nm=351.0,
    mode="radial_absorption_1d",
    rays_per_beam=8000,
    ray_output_count=200,
    ray_output_trajectory=True,
    absorption=dict(model="inverse_bremsstrahlung"),
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
            profile=dict(
                model="super_gaussian",
                w0_um=250.0,
                m=4,
            ),
        )
    ],
    cbet=dict(enable=False),
    hot_electron=dict(enable=False),
)

Burn(enabled=False)

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
        enabled=True,
        solver="implicit",
        f_lim=0.06,
    ),
    positivity=dict(clamp=True),
    safety=dict(nan_fatal=True),
    diagnostics_every=100,
)

Output(
    directory="outputs/lp1d_ex03_impedance_match",
    format="hdf5",
    plot_every_s=1.0e-11,
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
