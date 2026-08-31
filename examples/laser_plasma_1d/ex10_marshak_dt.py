from tenryu_namelist import *

# EX-10: Marshak-driven DT exploding-pusher analogue with fusion yield.
# Radiative ablation drives shocks through the thin shell at 0.5-2.0 ns.
# Stagnation is expected at 3.5-5.0 ns and bang time at 4.0-5.5 ns.
# Neutron-yield screening window: 1.0e8-1.0e11; alpha self-heating is
# absent by construction.

um = 1.0e-4
ns = 1.0e-9

R_DT = 350.0 * um
R_SHELL_OUT = 360.0 * um
R_MAX = 550.0 * um

mat_dt = Material(
    name="DT",
    A=2.515,
    Z=1.0,
    eos=dict(
        model="tmat",
        file="TMAT-H5/DT.tmat.h5",
    ),
    opacity=dict(model="constant", kappa_a=1.0, kappa_s=0.0, units="cm2_per_g"),
)

mat_cd = Material(
    name="CD",
    A=7.0,
    Z=3.5,
    eos=dict(
        model="tmat",
        file="TMAT-H5/CD_lte.tmat.h5",
    ),
    # CD kappa_a: Planck mean of the CD table at 1.05 g/cc, 100-150 eV (surface-absorption regime).
    opacity=dict(model="constant", kappa_a=5.0e3, kappa_s=0.0, units="cm2_per_g"),
)

mat_void = Material(name="VOID", A=1.0, Z=1.0, is_void=True)

Main(
    name="lp1d_ex10_marshak_dt",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=7.0e-9,
    seed=12345,
    max_steps=10_000_000,
    verbosity="normal",
)

# Forty pins form 41 intervals. With 820 cells and a 20-cell minimum per
# interval, the zoning solver must assign exactly 20 cells to every interval:
# 25 intervals in DT, eight in CD, and eight in VOID. This gives 500, 160,
# and 160 uniform-width cells, respectively. Width jumps are allowed only at
# the material interfaces. The constrained mesh is paired with the full-step
# snapshot and dt/2 retry path enabled in Numerics.hydro below.
MESH_PINS = (
    [
        {
            "r": i * R_DT / 25.0,
            "ratio_jump_allowed": False,
        }
        for i in range(1, 25)
    ]
    + [
        {
            "r": R_DT,
            "ratio_jump_allowed": True,
        }
    ]
    + [
        {
            "r": R_DT + i * (R_SHELL_OUT - R_DT) / 8.0,
            "ratio_jump_allowed": False,
        }
        for i in range(1, 8)
    ]
    + [
        {
            "r": R_SHELL_OUT,
            "ratio_jump_allowed": True,
        }
    ]
    + [
        {
            "r": R_SHELL_OUT + i * (R_MAX - R_SHELL_OUT) / 8.0,
            "ratio_jump_allowed": False,
        }
        for i in range(1, 8)
    ]
)

Mesh(
    r_min=0.0,
    r_max=R_MAX,
    geometry_1d="spherical",
    zoning_intent=dict(
        n_cells=820,
        measure="width",
        pins=MESH_PINS,
        preferred_ratio=1.3,
        ratio_hard_max=2.0,
        min_cells_per_segment=20,
    ),
    motion="lagrangian",
    floors=dict(
        rho_floor_gcc=1.0e-9,
        Te_floor_eV=0.1,
        Ti_floor_eV=0.1,
    ),
)

Materials(
    materials=[mat_dt, mat_cd, mat_void],
    zbar=dict(
        model="fixed",
        fixed_value=3.5,
    ),
    void_config=dict(
        rho=1.0e-9,
        Te=0.1,
        Ti=0.1,
    ),
)


def vf_dt(r_cm):
    return 1.0 if r_cm < R_DT else 0.0


def vf_cd(r_cm):
    return 1.0 if R_DT <= r_cm < R_SHELL_OUT else 0.0


def vf_void(r_cm):
    return 1.0 if r_cm >= R_SHELL_OUT else 0.0


def rho_profile(r_cm):
    if r_cm < R_DT:
        return 0.0040
    if r_cm < R_SHELL_OUT:
        return 1.05
    return 1.0e-9


def Te_profile(r_cm):
    return 1.0 if r_cm < R_SHELL_OUT else 0.1


def Ti_profile(r_cm):
    return 1.0 if r_cm < R_SHELL_OUT else 0.1


Geometry(
    volfrac=dict(
        DT=vf_dt,
        CD=vf_cd,
        VOID=vf_void,
    ),
    rho=rho_profile,
    Te=Te_profile,
    Ti=Ti_profile,
    radiation_field="zero",
    enforce_sum_to_one=True,
)

TR_POINTS = [
    (0.00 * ns, 0.0),
    (0.20 * ns, 60.0),
    (0.80 * ns, 60.0),
    (1.00 * ns, 120.0),
    (1.80 * ns, 120.0),
    (2.20 * ns, 180.0),
    (3.20 * ns, 180.0),
    (3.50 * ns, 0.0),
]


def marshak_temperature(t_s):
    if t_s < TR_POINTS[0][0] or t_s > TR_POINTS[-1][0]:
        return 0.0

    for i in range(len(TR_POINTS) - 1):
        t0, Tr0 = TR_POINTS[i]
        t1, Tr1 = TR_POINTS[i + 1]
        if t_s <= t1:
            fraction = (t_s - t0) / (t1 - t0)
            return Tr0 + fraction * (Tr1 - Tr0)

    return 0.0


Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=8,
    group_bounds_eV=[
        1.0,
        3.0,
        10.0,
        30.0,
        100.0,
        300.0,
        1000.0,
        3000.0,
        10000.0,
    ],
    multigroup_diffusion=dict(
        boundary=dict(
            inner_r="reflect",
            outer_r="marshak",
        ),
    ),
    boundary=dict(
        marshak_Tr=marshak_temperature,
    ),
)

Laser(enabled=False)

Burn(
    enabled=True,
    fuels=["DT"],
    scheme="fraley",
    partition="fraley",
    fuel_materials=["DT"],
    x_D=0.5,
    x_T=0.5,
    x_He3=0.0,
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
    directory="outputs/lp1d_ex10_marshak_dt",
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
