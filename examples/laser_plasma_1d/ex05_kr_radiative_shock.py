from tenryu_namelist import *

# Expected phenomena:
# - The piston is shocked at 0.1-1.2 ns.
# - The material shock enters Kr at 0.5-1.5 ns.
# - A radiative precursor forms ahead of the density jump at 1-3 ns,
#   with length 100-300 um and precursor Te 5-30 eV.
# - The shock reaches the rear witness at 8-13 ns, marked by the first
#   sharp pressure rise in the witness; a reflected shock follows.
# - Commissioning material-shock speed window: 70-120 um/ns.

um = 1.0e-4
ns = 1.0e-9

X_CD_WITNESS_OUT = 20.0 * um
X_KR_OUT = 820.0 * um
X_CD_PISTON_OUT = 850.0 * um
X_MAX = 1050.0 * um

mat_cd = Material(
    name="CD",
    A=7.0,
    Z=3.5,
    eos=dict(
        model="tmat",
        file="TMAT-H5/CD.tmat.h5",
    ),
    opacity=dict(
        model="constant",
        kappa_a=5.0e3,  # CD Planck mean (table, 1.05 g/cc, 100-150 eV): surface-absorption regime
        kappa_s=0.0,
        units="cm2_per_g",
    ),
)

# TENRYU v1 restricts opacity.model="power_law" to grey transport, so the
# 24-group FLD uses a constant gray kappa for Kr. Supervisor choice: the
# geometric mean of the PrOpacEOS table Planck (9.1e3) and Rosseland (1.4e3)
# means at 18.7 eV / 0.044 g/cc -> 3.5e3 cm2/g (absorption length ~71 um at
# 0.040 g/cc), sized so the radiative precursor (100-300 um) is resolvable
# while the 800 um column stays optically thick (tau ~ 11).
mat_kr = Material(
    name="Kr",
    A=83.798,
    Z=36.0,
    eos=dict(
        model="tmat",
        file="TMAT-H5/KR_lte.tmat.h5",
    ),
    opacity=dict(
        model="constant",
        kappa_a=3.5e3,
        kappa_s=0.0,
        units="cm2_per_g",
    ),
)

mat_void = Material(
    name="VOID",
    A=1.0,
    Z=1.0,
    is_void=True,
)

Main(
    name="lp1d_ex05_kr_radiative_shock",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=18.0 * ns,
    seed=12345,
    max_steps=10_000_000,
    verbosity="normal",
)

DX_CD_WITNESS = X_CD_WITNESS_OUT / 80.0
DX_KR = (X_KR_OUT - X_CD_WITNESS_OUT) / 900.0
DX_VOID = (X_MAX - X_CD_PISTON_OUT) / 200.0

# This preferred-width profile gives the pin-delimited regions
# 80/900/160/200 cells. The piston width decreases monotonically from
# about 0.49 um to about 0.09 um at its laser-facing outer surface.
PROFILE_EPS = 1.0e-3 * um
PREFERRED_WIDTH_PROFILE = [
    {
        "r": 0.0,
        "w": DX_CD_WITNESS,
    },
    {
        "r": X_CD_WITNESS_OUT - PROFILE_EPS,
        "w": DX_CD_WITNESS,
    },
    {
        "r": X_CD_WITNESS_OUT + PROFILE_EPS,
        "w": DX_KR,
    },
    {
        "r": X_KR_OUT - PROFILE_EPS,
        "w": DX_KR,
    },
    {
        "r": X_KR_OUT + PROFILE_EPS,
        "w": 0.495 * um,
    },
    {
        "r": X_CD_PISTON_OUT - PROFILE_EPS,
        "w": 0.09 * um,
    },
    {
        "r": X_CD_PISTON_OUT + PROFILE_EPS,
        "w": DX_VOID,
    },
    {
        "r": X_MAX,
        "w": DX_VOID,
    },
]

# Narrow gaps prevent roundoff-only overlap between adjacent bands.
# Every witness, Kr, and VOID cell still overlaps its region's band,
# fixing those three regions to uniform cell widths within roundoff.
BAND_GAP = 1.0e-3 * um
UNIFORM_WIDTH_BANDS = [
    {
        "measure_frac_begin": 0.0,
        "measure_frac_end": (X_CD_WITNESS_OUT - BAND_GAP) / X_MAX,
        "cell_measure_min": DX_CD_WITNESS * (1.0 - 1.0e-12),
        "cell_measure_max": DX_CD_WITNESS,
    },
    {
        "measure_frac_begin": (X_CD_WITNESS_OUT + BAND_GAP) / X_MAX,
        "measure_frac_end": (X_KR_OUT - BAND_GAP) / X_MAX,
        "cell_measure_min": DX_KR,
        "cell_measure_max": DX_KR * (1.0 + 1.0e-12),
    },
    {
        "measure_frac_begin": (X_CD_PISTON_OUT + BAND_GAP) / X_MAX,
        "measure_frac_end": 1.0,
        "cell_measure_min": DX_VOID,
        "cell_measure_max": DX_VOID * (1.0 + 1.0e-12),
    },
]

Mesh(
    r_min=0.0,
    r_max=X_MAX,
    geometry_1d="planar",
    zoning_intent=dict(
        n_cells=1340,
        measure="width",
        pins=[
            {
                "r": X_CD_WITNESS_OUT,
                "ratio_jump_allowed": True,
            },
            {
                "r": X_KR_OUT,
                "ratio_jump_allowed": True,
            },
            {
                "r": X_CD_PISTON_OUT,
                "ratio_jump_allowed": True,
            },
        ],
        profile=PREFERRED_WIDTH_PROFILE,
        bands=UNIFORM_WIDTH_BANDS,
        dr_min=0.05 * um,
        preferred_ratio=1.3,
        ratio_hard_max=2.0,
        min_cells_per_segment=1,
    ),
    motion="lagrangian",
    floors=dict(
        rho_floor_gcc=1.0e-9,
        Te_floor_eV=0.1,
        Ti_floor_eV=0.1,
    ),
)

Materials(
    materials=[mat_cd, mat_kr, mat_void],
    opacity_mix_rule="linear_mass",
    zbar=dict(
        model="thomas_fermi",
    ),
    void_config=dict(
        rho=1.0e-9,
        Te=0.1,
        Ti=0.1,
    ),
)


def vf_cd(x_cm, z_cm=0.0):
    if x_cm < X_CD_WITNESS_OUT:
        return 1.0
    if X_KR_OUT <= x_cm < X_CD_PISTON_OUT:
        return 1.0
    return 0.0


def vf_kr(x_cm, z_cm=0.0):
    if X_CD_WITNESS_OUT <= x_cm < X_KR_OUT:
        return 1.0
    return 0.0


def vf_void(x_cm, z_cm=0.0):
    if x_cm >= X_CD_PISTON_OUT:
        return 1.0
    return 0.0


def rho_profile(x_cm, z_cm=0.0):
    if x_cm < X_CD_WITNESS_OUT:
        return 1.05
    if x_cm < X_KR_OUT:
        return 0.040
    if x_cm < X_CD_PISTON_OUT:
        return 1.05
    return 1.0e-9


def Te_profile(x_cm, z_cm=0.0):
    if x_cm < X_CD_PISTON_OUT:
        return 1.0
    return 0.1


def Ti_profile(x_cm, z_cm=0.0):
    if x_cm < X_CD_PISTON_OUT:
        return 1.0
    return 0.1


Geometry(
    rho=rho_profile,
    Te=Te_profile,
    Ti=Ti_profile,
    volfrac=dict(
        CD=vf_cd,
        Kr=vf_kr,
        VOID=vf_void,
    ),
    radiation_field="zero",
    enforce_sum_to_one=True,
)

GROUPS = 24
GROUP_BOUNDS_EV = [
    1.0e-3 * (5.0e3 / 1.0e-3) ** (i / GROUPS)
    for i in range(GROUPS + 1)
]

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=GROUPS,
    group_bounds_eV=GROUP_BOUNDS_EV,
    multigroup_diffusion=dict(
        flux_limiter="levermore_pomraning",
    ),
)

LASER_PULSE = [
    (0.00 * ns, 0.0),
    (0.10 * ns, 5.0e13),
    (0.90 * ns, 5.0e13),
    (1.00 * ns, 0.0),
]


def laser_power(t_s):
    if t_s < LASER_PULSE[0][0] or t_s > LASER_PULSE[-1][0]:
        return 0.0

    for i in range(len(LASER_PULSE) - 1):
        t0, p0 = LASER_PULSE[i]
        t1, p1 = LASER_PULSE[i + 1]
        if t_s <= t1:
            fraction = (t_s - t0) / (t1 - t0)
            return p0 + fraction * (p1 - p0)

    return 0.0


# Planar 1 cm^2 convention: P[W] = I[W/cm^2] * 1 cm^2.
# The resulting fluence is 4.5e4 J/cm^2.
# The beam points along -x from the outer boundary toward the target.
Laser(
    enabled=True,
    wavelength_nm=351.0,
    mode="radial_absorption_1d",
    rays_per_beam=8000,
    ray_output_count=200,
    ray_output_trajectory=True,
    absorption=dict(
        model="inverse_bremsstrahlung",
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
    cbet=dict(
        enable=False,
    ),
    hot_electron=dict(
        enable=False,
    ),
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
        driver_full_step_retry_max_attempts=8,
        av_heat_C=0.5,
        av_heat_to="ion",
        odd_even_damping_C=1.0,
        ee_odd_even_C=0.15,
    ),
    conduction=dict(
        enabled=True,
        solver="implicit",
        f_lim=0.06,
    ),
    positivity=dict(
        clamp=True,
    ),
    safety=dict(
        nan_fatal=True,
    ),
    diagnostics_every=100,
)

Output(
    directory="outputs/lp1d_ex05_kr_radiative_shock",
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
)
