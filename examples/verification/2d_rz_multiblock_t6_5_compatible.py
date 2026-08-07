"""Stage A-F T8 T6.5 compatible strong-drive acceptance deck."""

import os

from tenryu_namelist import *


N_C = int(os.environ.get("TENRYU_T6_5_COMPAT_N_C", "8"))
NZ_POLAR = int(os.environ.get("TENRYU_T6_5_COMPAT_NZ_POLAR", str(4 * N_C)))
NR_SHELL = int(os.environ.get("TENRYU_T6_5_COMPAT_NR_SHELL", "24"))
SEED = int(os.environ.get("TENRYU_T6_5_COMPAT_SEED", "12345"))
S_MAX = float(os.environ.get("TENRYU_T6_5_COMPAT_S_MAX_CM", "0.07"))
T_END = float(os.environ.get("TENRYU_T6_5_COMPAT_T_END_S", "2.0e-8"))
OUTDIR = os.environ.get(
    "TENRYU_T6_5_COMPAT_OUTDIR",
    "./build/output_verify_2d_rz_multiblock_t6_5_compatible",
)

A = 1.0
ZBAR = 1.0
GAMMA = 5.0 / 3.0
RHO0_GCC = 1.0e-3
T0_EV = 1.0
EV_TO_ERG = 1.6022e-12
M_P = 1.6726219e-24  # must match core::constants::proton_mass
P_INITIAL = RHO0_GCC * (1.0 + ZBAR) * EV_TO_ERG * T0_EV / (A * M_P)
P_FOOT = float(os.environ.get("TENRYU_T6_5_COMPAT_P_FOOT_DYN_CM2", "5.0e9"))
P_PEAK = float(os.environ.get("TENRYU_T6_5_COMPAT_P_PEAK_DYN_CM2", "1.5e11"))
TAU_FOOT = float(os.environ.get("TENRYU_T6_5_COMPAT_TAU_FOOT_S", "0.5e-9"))
TAU_HOLD = float(os.environ.get("TENRYU_T6_5_COMPAT_TAU_HOLD_S", "0.5e-9"))
TAU_MAIN = float(os.environ.get("TENRYU_T6_5_COMPAT_TAU_MAIN_S", "3.0e-9"))
R_C = S_MAX / 12.0
R_MATCH = 2.0 * R_C
BRIDGE_LAYERS = int(os.environ.get("TENRYU_T6_5_COMPAT_BRIDGE_LAYERS", "8"))
DT_INITIAL = 2.5e-11
DT_MAX = 5.0e-11
CFL_HYDRO = 0.2
MAX_STEPS = 50000

CASE_NAME = "2d_rz_multiblock_t6_5_compatible"

print(
    "[deck:2d_rz_multiblock_t6_5_compatible] "
    f"n_c={N_C} nz_polar={NZ_POLAR} nr_shell={NR_SHELL} "
    f"bridge_layers={BRIDGE_LAYERS} seed={SEED} S_max_cm={S_MAX} "
    f"r_c_cm={R_C} r_match_cm={R_MATCH} t_end_s={T_END} "
    f"dt_initial_s={DT_INITIAL} dt_max_s={DT_MAX} cfl_hydro={CFL_HYDRO} "
    f"P_initial_dyn_cm2={P_INITIAL} P_foot_dyn_cm2={P_FOOT} "
    f"P_peak_dyn_cm2={P_PEAK} tau_foot_s={TAU_FOOT} tau_hold_s={TAU_HOLD} "
    f"tau_main_s={TAU_MAIN} max_steps={MAX_STEPS} "
    f"rho_gcc={RHO0_GCC} T0_eV={T0_EV} outdir={OUTDIR}"
)


def smoothstep5(x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


def boundary_pressure(t):
    # Total outward pressure BC: start at ideal-gas p0, then smooth foot/hold/main
    # drive; smoothstep saturation holds P_PEAK after the main rise completes.
    foot = smoothstep5(t / TAU_FOOT)
    main = smoothstep5((t - TAU_FOOT - TAU_HOLD) / TAU_MAIN)
    return P_INITIAL + (P_FOOT - P_INITIAL) * foot + (P_PEAK - P_FOOT) * main


def rho_init(r, z):
    return RHO0_GCC


def Te_init(r, z):
    return T0_EV


def Ti_init(r, z):
    return T0_EV


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
    nr=NR_SHELL,
    nz=NZ_POLAR,
    grid="uniform",
    motion="ale",
    logical_mesh_2d="spherical_polar_halfplane",
    spherical_polar_s_max=S_MAX,
    topology_scheme="multiblock_cart_core_polar_shell",
    multiblock_cart_core_n_c=N_C,
    multiblock_cart_core_bridge_layers=BRIDGE_LAYERS,
    multiblock_cart_core_r_c=R_C,
    multiblock_cart_core_r_match=R_MATCH,
    multiblock_transition_scheme="rounded_core_seam",
    multiblock_cap_p=6.0,
    multiblock_bridge_elliptic_sweeps=4,
    multiblock_bridge_elliptic_omega=0.5,
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
    debug=dict(
        trace_mesh_motion=os.environ.get("TENRYU_T6_5_COMPAT_TRACE", "0") == "1",
        trace_max_steps=int(
            os.environ.get("TENRYU_T6_5_COMPAT_TRACE_MAX_STEPS", str(MAX_STEPS))
        ),
    ),
    hydro=dict(
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
        trial_volume_cfl_enabled=False,
        mesh_quality_dt_cfl_enabled=False,
        driver_full_step_retry_enabled=True,
    ),
    ale=dict(
        enabled=True,
        every_n_steps=1,
        multiblock_cross_seam_rezone_enabled=True,
        multiblock_scaled_reference_enabled=True,
        multiblock_path_admissibility_enabled=True,
        path_admissibility_floor=0.01,
        dt_rejection_factor=0.5,
        max_dt_rejections=8,
        conservative_remap_enabled=True,
        conservative_remap_order="second_order_van_leer",
        conservative_remap_target="reference",
    ),
    conduction=dict(enabled=False),
    plic=dict(enabled=False),
    diagnostics=dict(
        phase_resolved_energy=True,
        conservation=dict(enabled=True),
        mesh_quality_min=dict(enabled=True),
        production_audit=dict(
            enabled=True,
            tier="A",
            gcl=dict(enabled=False),
            positivity=dict(enabled=False),
            audit_json_path=os.path.join(OUTDIR, "audit_summary.json"),
        ),
    ),
    diagnostics_every=1,
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-4, Ti_floor_eV=1.0e-4),
    safety=dict(energy_fatal=False, nan_fatal=True, energy_threshold=1.0),
)

Output(
    directory=OUTDIR,
    format="hdf5",
    plot_every=0,
    history_every=1,
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
