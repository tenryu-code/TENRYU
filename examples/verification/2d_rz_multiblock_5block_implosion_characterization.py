import os

from tenryu_namelist import *


N_C = int(os.environ.get("TENRYU_IMPLO_N_C", "8"))
FAN_LAYERS = max(4, round(N_C / 8.0))
NR_SHELL = N_C
SEED = int(os.environ.get("TENRYU_IMPLO_SEED", "12345"))
S_MAX = float(os.environ.get("TENRYU_IMPLO_S_MAX_CM", "0.07"))
T_END = float(os.environ.get("TENRYU_IMPLO_T_END_S", "5.0e-9"))
P_EXT = float(os.environ.get("TENRYU_IMPLO_P_EXT_DYN_CM2", "3.0e9"))
OUTDIR = os.environ.get(
    "TENRYU_IMPLO_OUTDIR",
    "./build/output_verify_2d_rz_multiblock_5block_implosion_characterization",
)

EV_TO_ERG = 1.6022e-12
M_P = 1.6726219e-24
A = 1.0
ZBAR = 1.0
GAMMA = 5.0 / 3.0
RHO0_GCC = float(os.environ.get("TENRYU_IMPLO_RHO0_GCC", "1.0e-3"))
T0_EV = 1.0
R_C = S_MAX / 12.0
R_MATCH = 2.0 * R_C
NZ = 4 * N_C
DT_INITIAL = float(os.environ.get("TENRYU_IMPLO_DT_INITIAL_S", "2.5e-11"))
DT_MAX = float(os.environ.get("TENRYU_IMPLO_DT_MAX_S", "5.0e-11"))
CFL_HYDRO = float(os.environ.get("TENRYU_IMPLO_CFL_HYDRO", "0.2"))
MAX_STEPS = int(os.environ.get("TENRYU_IMPLO_MAX_STEPS", "20000"))
MOTION = os.environ.get("TENRYU_IMPLO_MOTION", "ale")
if MOTION not in {"ale", "lagrangian"}:
    raise ValueError(f"Unknown TENRYU_IMPLO_MOTION: {MOTION}")
REZONE_MODE = os.environ.get("TENRYU_IMPLO_REZONE_MODE", "axis")
if MOTION == "ale" and REZONE_MODE not in {"axis", "scaled", "both", "none", "m1"}:
    raise ValueError(f"Unknown TENRYU_IMPLO_REZONE_MODE: {REZONE_MODE}")
AXIS_REZONE_ENABLED = MOTION == "ale" and REZONE_MODE in {"axis", "both"}
SCALED_REFERENCE_ENABLED = MOTION == "ale" and REZONE_MODE in {"scaled", "both"}
M1_ENABLED = MOTION == "ale" and REZONE_MODE == "m1"
KILLTEST = os.environ.get("TENRYU_IMPLO_KILLTEST", "0") == "1"
TRACE = os.environ.get("TENRYU_IMPLO_TRACE", "0") == "1"

CASE_NAME = "2d_rz_multiblock_5block_implosion_characterization"

print(
    "[deck:2d_rz_multiblock_5block_implosion_characterization] "
    f"n_c={N_C} fan_layers={FAN_LAYERS} nr_shell={NR_SHELL} nz={NZ} "
    f"seed={SEED} S_max_cm={S_MAX} r_c_cm={R_C} r_match_cm={R_MATCH} "
    f"P_ext_dyn_cm2={P_EXT} rho_gcc={RHO0_GCC} T0_eV={T0_EV} "
    f"t_end_s={T_END} dt_initial_s={DT_INITIAL} dt_max_s={DT_MAX} "
    f"cfl_hydro={CFL_HYDRO} max_steps={MAX_STEPS} axis_rezone={AXIS_REZONE_ENABLED} "
    f"rezone_mode={REZONE_MODE} motion={MOTION} killtest={KILLTEST} outdir={OUTDIR}"
)


def boundary_pressure(t):
    return P_EXT


def rho_init(r, z):
    return RHO0_GCC


def Te_init(r, z):
    return T0_EV


def Ti_init(r, z):
    return T0_EV


def velocity_init(r, z):
    return (0.0, 0.0)


HYDRO = dict(
    enabled=True,
    boundary_2d=dict(
        r_inner="axis",
        r_outer="pressure",
        z_bottom="reflect",
        z_top="reflect",
    ),
    boundary_pressure=boundary_pressure,
    av_model="csw_edge",
    subzonal_pressure_enabled=True,
    av_C1=0.1,
    av_C2=1.5,
    trial_volume_cfl_enabled=True,
    mesh_quality_dt_cfl_enabled=True,
)
if KILLTEST:
    HYDRO["driver_full_step_retry_enabled"] = True

ALE = dict(enabled=False)
if MOTION == "ale":
    ALE = dict(
        enabled=True,
        relaxation=0.0,
    )
if MOTION == "ale" and AXIS_REZONE_ENABLED:
    ALE["axis_rezone_enabled"] = True
if MOTION == "ale" and SCALED_REFERENCE_ENABLED:
    ALE["multiblock_scaled_reference_enabled"] = True
if MOTION == "ale" and M1_ENABLED:
    ALE["rezone_solver"] = "m1_tmop"
    ALE["m1_theta_reg"] = float(os.environ.get("TENRYU_IMPLO_M1_THETA_REG", "1.0"))
    ALE["m1_sweeps"] = int(os.environ.get("TENRYU_IMPLO_M1_SWEEPS", "8"))
    ALE["m1_lambda_tether"] = float(
        os.environ.get("TENRYU_IMPLO_M1_LAMBDA_TETHER", "0.1")
    )
if MOTION == "ale" and KILLTEST:
    ALE["reference_barrier_enabled"] = True
    ALE["driver_retry_reference_barrier_enabled"] = True
    ALE["multiblock_scaled_reference_enabled"] = True


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
    nr=NR_SHELL,
    nz=NZ,
    grid="uniform",
    motion=MOTION,
    logical_mesh_2d="spherical_polar_halfplane",
    spherical_polar_s_max=S_MAX,
    topology_scheme="multiblock_half_butterfly_5block",
    multiblock_cart_core_n_c=N_C,
    multiblock_cart_core_bridge_layers=FAN_LAYERS,
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
        growth_factor=1.05,
        max_s=DT_MAX,
        min_s=1.0e-22,
    ),
    hydro=HYDRO,
    ale=ALE,
    conduction=dict(enabled=False),
    plic=dict(enabled=False),
    diagnostics=dict(
        phase_resolved_energy=True,
        conservation=dict(enabled=True),
        mesh_quality_min=dict(enabled=True),
        production_audit=dict(
            enabled=True,
            tier="A",
            gcl=dict(enabled=True),
            positivity=dict(enabled=True, fatal_on_neg=True),
            audit_json_path=os.path.join(OUTDIR, "audit_summary.json"),
        ),
    ),
    diagnostics_every=1,
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-4, Ti_floor_eV=1.0e-4),
    safety=dict(energy_fatal=False, nan_fatal=True, energy_threshold=1.0),
    **(
        dict(debug=dict(trace_mesh_motion=True, trace_max_steps=MAX_STEPS))
        if TRACE
        else {}
    ),
)

Output(
    directory=OUTDIR,
    format="hdf5",
    plot_every=0,
    history_every=1,
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
