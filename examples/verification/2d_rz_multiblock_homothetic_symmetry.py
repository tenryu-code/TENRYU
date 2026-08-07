import os

from tenryu_namelist import *


N_C = int(os.environ.get("TENRYU_MULTIBLOCK_HOMOTHETIC_G3_N_C", "8"))
BRIDGE_LAYERS = int(
    os.environ.get("TENRYU_MULTIBLOCK_HOMOTHETIC_G3_BRIDGE_LAYERS", "4")
)
SEED = int(os.environ.get("TENRYU_MULTIBLOCK_HOMOTHETIC_G3_SEED", "12345"))
S_MAX = float(os.environ.get("TENRYU_MULTIBLOCK_HOMOTHETIC_G3_S_MAX_CM", "0.07"))
T_END = float(os.environ.get("TENRYU_MULTIBLOCK_HOMOTHETIC_G3_T_END_S", "5.0e-7"))
DT_INITIAL = float(
    os.environ.get("TENRYU_MULTIBLOCK_HOMOTHETIC_G3_DT_INITIAL_S", "1.0e-9")
)
DT_MAX = float(os.environ.get("TENRYU_MULTIBLOCK_HOMOTHETIC_G3_DT_MAX_S", "5.0e-9"))
CFL_HYDRO = float(
    os.environ.get("TENRYU_MULTIBLOCK_HOMOTHETIC_G3_CFL_HYDRO", "0.5")
)
MAX_STEPS = int(os.environ.get("TENRYU_MULTIBLOCK_HOMOTHETIC_G3_MAX_STEPS", "20000"))
OUTDIR = os.environ.get(
    "TENRYU_MULTIBLOCK_HOMOTHETIC_G3_OUTDIR",
    "./build/output_verify_2d_rz_multiblock_homothetic_g3",
)

EV_TO_ERG = 1.6022e-12
M_P = 1.6726219e-24
A = 1.0
ZBAR = 1.0
GAMMA = 5.0 / 3.0
RHO0_GCC = 1.0
P0_DYN_PER_CM2 = 1.0e10
ALPHA_INV_S = -1.0e5
CV_EV_PER_G_ION = EV_TO_ERG / (A * M_P * (GAMMA - 1.0))
CV_EV_PER_G_ELECTRON = ZBAR * EV_TO_ERG / (A * M_P * (GAMMA - 1.0))
TI0_EV = P0_DYN_PER_CM2 / (RHO0_GCC * (GAMMA - 1.0) * CV_EV_PER_G_ION)
TE0_EV = P0_DYN_PER_CM2 / (RHO0_GCC * (GAMMA - 1.0) * CV_EV_PER_G_ELECTRON)
R_C = S_MAX / 12.0
R_MATCH = 2.0 * R_C

CASE_NAME = "multiblock_homothetic_symmetry_g3"

print(
    "[deck:2d_rz_multiblock_homothetic_symmetry] "
    f"n_c={N_C} bridge_layers={BRIDGE_LAYERS} seed={SEED} "
    f"S_max_cm={S_MAX} r_c_cm={R_C} r_match_cm={R_MATCH} "
    f"t_end_s={T_END} dt_initial_s={DT_INITIAL} dt_max_s={DT_MAX} "
    f"cfl_hydro={CFL_HYDRO} max_steps={MAX_STEPS} outdir={OUTDIR} "
    f"rho_gcc={RHO0_GCC} Pe_dyn_cm2={P0_DYN_PER_CM2} "
    f"Pi_dyn_cm2={P0_DYN_PER_CM2} alpha_inv_s={ALPHA_INV_S}"
)


def rho_init(r, z):
    return RHO0_GCC


def Te_init(r, z):
    return TE0_EV


def Ti_init(r, z):
    return TI0_EV


def velocity_init(r, z):
    return (ALPHA_INV_S * r, ALPHA_INV_S * z)


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
    nz=4 * N_C,
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
    plot_every_s=T_END / 4.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    save_namelist_copy=True,
    save_frozen_config=True,
    plot_fields=["rho", "Te", "Ti", "ee", "ei", "Pe", "Pi", "volfrac"],
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=True, every=1, energy_budget=dict(enabled=True, warn_threshold=1.0))
