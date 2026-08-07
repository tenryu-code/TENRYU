import os

from tenryu_namelist import *


N_C = int(os.environ.get("TENRYU_COMPAT_FORCE_SMOKE_N_C", "4"))
NZ_POLAR = 4 * N_C
NR_SHELL = int(os.environ.get("TENRYU_COMPAT_FORCE_SMOKE_NR_SHELL", "8"))
SEED = int(os.environ.get("TENRYU_COMPAT_FORCE_SMOKE_SEED", "12345"))
S_MAX = float(os.environ.get("TENRYU_COMPAT_FORCE_SMOKE_S_MAX_CM", "0.07"))
OUTDIR = os.environ.get(
    "TENRYU_COMPAT_FORCE_SMOKE_OUTDIR",
    "./build/output_verify_2d_rz_multiblock_compatible_force_smoke",
)

EV_TO_ERG = 1.6022e-12
M_P = 1.6726219e-24
A = 1.0
ZBAR = 1.0
GAMMA = 5.0 / 3.0
RHO0_GCC = 1.0e-3
T0_EV = 1.0
R_C = S_MAX / 12.0
R_MATCH = 2.0 * R_C
BRIDGE_LAYERS = max(4, round(N_C / 8.0))
DT = 2.5e-11
T_END = 2.5e-10

CASE_NAME = "2d_rz_multiblock_compatible_force_smoke"

print(
    "[deck:2d_rz_multiblock_compatible_force_smoke] "
    f"n_c={N_C} nz_polar={NZ_POLAR} nr_shell={NR_SHELL} "
    f"bridge_layers={BRIDGE_LAYERS} seed={SEED} S_max_cm={S_MAX} "
    f"r_match_cm={R_MATCH} dt_s={DT} t_end_s={T_END} outdir={OUTDIR}"
)


def rho_init(r, z):
    return RHO0_GCC


def Te_init(r, z):
    return T0_EV


def Ti_init(r, z):
    return T0_EV


def velocity_init(r, z):
    if r > 0.5 * R_MATCH:
        return (1.0e-3, 1.0e-3)
    return (0.0, 0.0)


Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=T_END,
    max_steps=10,
    seed=SEED,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=S_MAX,
    z_min=-S_MAX,
    z_max=S_MAX,
    nr=NR_SHELL,
    nz=NZ_POLAR,
    grid="uniform",
    motion="lagrangian",
    logical_mesh_2d="spherical_polar_halfplane",
    spherical_polar_s_max=S_MAX,
    topology_scheme="multiblock_cart_core_polar_shell",
    multiblock_cart_core_n_c=N_C,
    multiblock_cart_core_bridge_layers=BRIDGE_LAYERS,
    multiblock_cart_core_r_c=R_C,
    multiblock_cart_core_r_match=R_MATCH,
)

Materials(
    materials=[
        Material(
            name="gas",
            A=A,
            Z=ZBAR,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=ZBAR),
)

Geometry(
    volfrac=dict(gas=lambda r, z: 1.0),
    rho=rho_init,
    Te=Te_init,
    Ti=Ti_init,
    velocity=velocity_init,
    radiation_field="zero",
)

Numerics(
    radiation_thermal_subcycle=False,
    dt=dict(
        initial_s=DT,
        cfl_hydro=0.2,
        growth_factor=1.0,
        max_s=DT,
        min_s=1.0e-22,
    ),
    hydro=dict(
        enabled=True,
        boundary_2d=dict(
            r_inner="axis",
            r_outer="reflect",
            z_bottom="reflect",
            z_top="reflect",
        ),
        av_model="csw_edge",
        subzonal_pressure_enabled=True,
        trial_volume_cfl_enabled=False,
        mesh_quality_dt_cfl_enabled=False,
    ),
    ale=dict(enabled=False),
    conduction=dict(enabled=False),
    plic=dict(enabled=False),
    diagnostics=dict(
        phase_resolved_energy=True,
        conservation=dict(enabled=True),
        mesh_quality_min=dict(enabled=True),
    ),
    diagnostics_every=1,
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-4, Ti_floor_eV=1.0e-4),
    safety=dict(energy_fatal=False, nan_fatal=True, energy_threshold=1.0),
)

Output(
    directory=OUTDIR,
    format="hdf5",
    plot_every=1,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    save_namelist_copy=True,
    save_frozen_config=True,
    plot_fields=["rho", "Te", "Ti", "ee", "ei", "Pe", "Pi", "volfrac"],
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=True, every=1, energy_budget=dict(enabled=True, warn_threshold=1.0))
