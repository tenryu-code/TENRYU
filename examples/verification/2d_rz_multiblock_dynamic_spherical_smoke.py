import math
import os

from tenryu_namelist import *


N_C = int(os.environ.get("TENRYU_G4_N_C", "8"))
SEED = int(os.environ.get("TENRYU_G4_SEED", "12345"))
S_MAX = 0.07
NZ = 4 * N_C
BRIDGE_LAYERS = max(4, round(N_C / 8.0))
R_C = S_MAX / 12.0
R_MATCH = 2.0 * R_C
OUTDIR = os.environ.get(
    "TENRYU_G4_OUTDIR",
    "./build/output_verify_2d_rz_multiblock_dynamic_spherical_smoke_g4",
)

EV_TO_ERG = 1.6022e-12
M_P = 1.6726219e-24
A = 1.0
ZBAR = 1.0
GAMMA = 5.0 / 3.0
RHO_AMBIENT_GCC = 1.0
PE_AMBIENT_DYN_PER_CM2 = 1.0e10
PI_AMBIENT_DYN_PER_CM2 = 1.0e10
P_TOTAL_AMBIENT_DYN_PER_CM2 = PE_AMBIENT_DYN_PER_CM2 + PI_AMBIENT_DYN_PER_CM2
CS_AMBIENT_CM_PER_S = math.sqrt(
    GAMMA * P_TOTAL_AMBIENT_DYN_PER_CM2 / RHO_AMBIENT_GCC
)
T_END = 0.7 * S_MAX / CS_AMBIENT_CM_PER_S
DT_INITIAL = 1.0e-9
DT_MAX = 1.0e-8
CFL_HYDRO = 0.5
MAX_STEPS = 20000

CV_EV_PER_G_ION = EV_TO_ERG / (A * M_P * (GAMMA - 1.0))
CV_EV_PER_G_ELECTRON = ZBAR * EV_TO_ERG / (A * M_P * (GAMMA - 1.0))
TI_AMBIENT_EV = PI_AMBIENT_DYN_PER_CM2 / (
    RHO_AMBIENT_GCC * (GAMMA - 1.0) * CV_EV_PER_G_ION
)
TE_AMBIENT_EV = PE_AMBIENT_DYN_PER_CM2 / (
    RHO_AMBIENT_GCC * (GAMMA - 1.0) * CV_EV_PER_G_ELECTRON
)
PULSE_RADIUS_CM = S_MAX / 20.0
TAPER_RADIUS_CM = S_MAX / 10.0

CASE_NAME = "multiblock_dynamic_spherical_smoke_g4"

print(
    "[deck:2d_rz_multiblock_dynamic_spherical_smoke] "
    f"n_c={N_C} nz={NZ} bridge_layers={BRIDGE_LAYERS} seed={SEED} "
    f"S_max_cm={S_MAX} r_c_cm={R_C} r_match_cm={R_MATCH} "
    f"pulse_radius_cm={PULSE_RADIUS_CM} taper_radius_cm={TAPER_RADIUS_CM} "
    f"t_end_s={T_END} dt_initial_s={DT_INITIAL} dt_max_s={DT_MAX} "
    f"cfl_hydro={CFL_HYDRO} max_steps={MAX_STEPS} outdir={OUTDIR} "
    f"rho_gcc={RHO_AMBIENT_GCC} Pe_ambient_dyn_cm2={PE_AMBIENT_DYN_PER_CM2} "
    f"Pi_ambient_dyn_cm2={PI_AMBIENT_DYN_PER_CM2}"
)


def pressure_multiplier(r, z):
    s = math.hypot(r, z)
    if s <= PULSE_RADIUS_CM:
        return 2.0
    if s <= TAPER_RADIUS_CM:
        x = (s - PULSE_RADIUS_CM) / (TAPER_RADIUS_CM - PULSE_RADIUS_CM)
        return 1.0 + math.cos(0.5 * math.pi * x) ** 2
    return 1.0


def rho_init(r, z):
    return RHO_AMBIENT_GCC


def Te_init(r, z):
    return TE_AMBIENT_EV * pressure_multiplier(r, z)


def Ti_init(r, z):
    return TI_AMBIENT_EV * pressure_multiplier(r, z)


def velocity_init(r, z):
    return (0.0, 0.0)


Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=S_MAX,
    z_min=-S_MAX,
    z_max=S_MAX,
    nr=N_C,
    nz=NZ,
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
        initial_s=DT_INITIAL,
        cfl_hydro=CFL_HYDRO,
        growth_factor=1.1,
        max_s=DT_MAX,
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
        av_C1=0.1,
        av_C2=1.5,
        trial_volume_cfl_enabled=False,
        mesh_quality_dt_cfl_enabled=False,
    ),
    ale=dict(enabled=False),
    conduction=dict(enabled=False),
    plic=dict(enabled=False),
    diagnostics=dict(
        phase_resolved_energy=True,
        mesh_quality_min=dict(enabled=True),
    ),
    diagnostics_every=1,
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-4, Ti_floor_eV=1.0e-4),
)

Output(
    directory=OUTDIR,
    format="hdf5",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=T_END,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    save_namelist_copy=True,
    save_frozen_config=True,
    plot_fields=["rho", "Te", "Ti", "ee", "ei", "Pe", "Pi", "volfrac"],
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=True, every=1, energy_budget=dict(enabled=True, warn_threshold=1.0))
