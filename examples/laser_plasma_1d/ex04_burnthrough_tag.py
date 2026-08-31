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

# Scientific target: quasi-steady ablation front reaches the buried CD tag
# at 0.6-1.1 ns for a mass-ablation rate of 0.7-1.4 mg/cm^2/ns.
# Afterwards, the target enters exploding-foil expansion.

um = 1.0e-4
ns = 1.0e-9

X_BACKING_TAG_FACE = 4.0 * um
X_TAG_ABLATOR_FACE = 4.1 * um
X_ABLATOR_OUTER_FACE = 12.1 * um
X_MAX = 162.1 * um

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

mat_cd_tag = Material(
    name="CD_tag",
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
    name="lp1d_ex04_burnthrough_tag",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=2.0 * ns,
    seed=12345,
    max_steps=10_000_000,
    verbosity="normal",
)

# The auto mesh contains 344 cells:
# 80 backing cells, 4 tag cells, 160 ablator cells, and 100 uniform
# outer-void cells. The backing begins at the free boundary x=0.
# Every material interface is an exact auto-region boundary. The three
# identical-density CD regions share a material_group so the mesh generator
# can smoothly match cell masses while retaining both pinned tag faces.
AUTO_REGIONS = [
    {
        "r_end": X_BACKING_TAG_FACE,
        "nz": 80,
        "rho_ref": 1.05,
        "is_void": False,
        "material_group": "CD",
    },
    {
        "r_end": X_TAG_ABLATOR_FACE,
        "nz": 4,
        "rho_ref": 1.05,
        "is_void": False,
        "material_group": "CD",
    },
    {
        "r_end": X_ABLATOR_OUTER_FACE,
        "nz": 160,
        "rho_ref": 1.05,
        "is_void": False,
        "material_group": "CD",
    },
    {
        "r_end": X_MAX,
        "nz": 100,
        "rho_ref": 1.0e-9,
        "is_void": True,
        "material_group": "VOID",
    },
]

Mesh(
    r_min=0.0,
    r_max=X_MAX,
    geometry_1d="planar",
    auto_regions=AUTO_REGIONS,
    auto_zone=dict(
        mass_ratio_max=1.3,
    ),
    motion="lagrangian",
    floors=dict(
        rho_floor_gcc=1.0e-9,
        Te_floor_eV=0.1,
        Ti_floor_eV=0.1,
    ),
)

Materials(
    materials=[mat_cd, mat_cd_tag, mat_void],
    opacity_mix_rule="linear_mass",
    zbar=dict(model="fixed", fixed_value=3.5),
    void_config=dict(rho=1.0e-9, Te=0.1, Ti=0.1),
)


def vf_void(x_cm, z_cm=0.0):
    return np.where(x_cm < X_ABLATOR_OUTER_FACE, 0.0, 1.0)


def vf_cd(x_cm, z_cm=0.0):
    return np.where(
        x_cm < X_BACKING_TAG_FACE,
        1.0,
        np.where(
            x_cm < X_TAG_ABLATOR_FACE,
            0.0,
            np.where(x_cm < X_ABLATOR_OUTER_FACE, 1.0, 0.0),
        ),
    )


def vf_cd_tag(x_cm, z_cm=0.0):
    return np.where(
        x_cm < X_BACKING_TAG_FACE,
        0.0,
        np.where(x_cm < X_TAG_ABLATOR_FACE, 1.0, 0.0),
    )


def rho_profile(x_cm, z_cm=0.0):
    if x_cm < X_ABLATOR_OUTER_FACE:
        return 1.05
    return 1.0e-9


def Te_profile(x_cm, z_cm=0.0):
    if x_cm < X_ABLATOR_OUTER_FACE:
        return 1.0
    return 0.1


def Ti_profile(x_cm, z_cm=0.0):
    if x_cm < X_ABLATOR_OUTER_FACE:
        return 1.0
    return 0.1


Geometry(
    rho=rho_profile,
    Te=Te_profile,
    Ti=Ti_profile,
    volfrac=dict(
        CD=vf_cd,
        CD_tag=vf_cd_tag,
        VOID=vf_void,
    ),
    enforce_sum_to_one=True,
)

Radiation(enabled=False)


def laser_power(t_s):
    peak_power_W = 5.0e14
    if t_s <= 0.0:
        return 0.0
    if t_s < 0.08 * ns:
        return peak_power_W * t_s / (0.08 * ns)
    if t_s <= 0.72 * ns:
        return peak_power_W
    if t_s < 0.80 * ns:
        return peak_power_W * (0.80 * ns - t_s) / (0.08 * ns)
    return 0.0


# Planar 1 cm^2 convention: P[W] = I[W/cm^2] * 1 cm^2.
# The resulting fluence is 3.60e5 J/cm^2.
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
        f_lim=0.10,
    ),
    positivity=dict(clamp=True),
    safety=dict(nan_fatal=True),
    diagnostics_every=100,
)

Output(
    directory="outputs/lp1d_ex04_burnthrough_tag",
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
