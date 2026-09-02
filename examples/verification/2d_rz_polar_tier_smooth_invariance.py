import math
import os

from tenryu_namelist import *


S_MAX = 0.010
R_DISK = 1.196e-3
NR = 288
NTHETA = 192
SEED = 12345

EV_TO_ERG = 1.6022e-12
M_P = 1.6726219e-24
A = 1.0
ZBAR = 1.0
GAMMA = 5.0 / 3.0
RHO0 = 1.0
PE0 = 1.0e10
PI0 = 1.0e10
P0 = PE0 + PI0
CS = math.sqrt(GAMMA * P0 / RHO0)

V0 = 500.0
H = 5.0e4
DT = 5.0e-10
MAX_STEPS = 400
T_END = 1.0

CV_EV_PER_G_ION = EV_TO_ERG / (A * M_P * (GAMMA - 1.0))
CV_EV_PER_G_ELECTRON = ZBAR * EV_TO_ERG / (A * M_P * (GAMMA - 1.0))
TI0 = PI0 / (RHO0 * (GAMMA - 1.0) * CV_EV_PER_G_ION)
TE0 = PE0 / (RHO0 * (GAMMA - 1.0) * CV_EV_PER_G_ELECTRON)

MODE = os.environ.get("TENRYU_P3_MODE", "static").strip().lower()
if MODE not in ("static", "translate", "homologous"):
    raise ValueError(
        "TENRYU_P3_MODE must be one of {'static', 'translate', 'homologous'}"
    )
OUTDIR = os.environ.get(
    "TENRYU_P3_OUTDIR",
    f"./build/output_p3_smooth_{MODE}",
)
CASE_NAME = f"p3_polar_tier_smooth_invariance_{MODE}"

print(
    "[deck:p3_smooth] "
    f"mode={MODE} nr={NR} ntheta={NTHETA} S_max_cm={S_MAX} "
    f"R_disk_cm={R_DISK} rho_gcc={RHO0} Pe_dyn_cm2={PE0} "
    f"Pi_dyn_cm2={PI0} cs_cm_s={CS} V0_cm_s={V0} H_inv_s={H} "
    f"dt_s={DT} max_steps={MAX_STEPS} outdir={OUTDIR}"
)


def boundary_pressure(t):
    return P0


def rho_init(r, z):
    return RHO0


def Te_init(r, z):
    return TE0


def Ti_init(r, z):
    return TI0


def velocity_init(r, z):
    if MODE == "translate":
        return (0.0, V0)
    if MODE == "homologous":
        return (-H * r, -H * z)
    return (0.0, 0.0)


Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    temperature_model="1T",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity="quiet",
)

Mesh(
    logical_mesh_2d="spherical_polar_halfplane",
    topology_scheme="multiblock_polar_tier",
    spherical_polar_s_max=S_MAX,
    multiblock_cart_core_r_match=0.0027083333333333334,
    polar_tier_hydro_enabled=True,
    polar_tier_dendrite_enabled=True,
    polar_tier_dendrite_s_theta_rows_below=5,
    polar_tier_native_pentagon=True,
    polar_tier_pole_cap_m=0,
    polar_tier_pole_cap_alpha=0.0,
    nr=NR,
    r_min=0.0,
    r_max=S_MAX,
    z_min=-S_MAX,
    z_max=S_MAX,
    nz=NTHETA,
    grid="uniform",
    motion="ale",
)

Materials(
    materials=[
        Material(
            name="gas",
            A=A,
            Z=ZBAR,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=dict(
                model="constant",
                kappa_a=0.0,
                kappa_s=0.0,
                units="cm2_per_g",
            ),
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
        cfl_hydro=0.15,
        growth_factor=1.02,
        max_s=DT,
        min_s=1.0e-22,
        cfl_cond=0.10,
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
        av_C1=0.1,
        av_C2=1.5,
        av_model="csw_edge_csw98",
        csw_polar_slaving_enabled=True,
        csw_polar_slaving_min_columns=96,
        csw_polar_slaving_full_columns=4,
        csw_polar_slaving_outer_columns=6,
        csw_polar_slaving_chi_on=0.08,
        csw_polar_slaving_chi_full=0.20,
        csw_polar_slaving_strength=1.0,
        csw_polar_slaving_av_stiffness_cfl_enabled=True,
        csw_polar_slaving_av_stiffness_sigma=0.8,
        rz_momentum_scheme="area_weighted_symmetric",
        subzonal_pressure_enabled=True,
        av_qcap_over_p=0.0,
        subzonal_dt_limiter_enabled=False,
        aw_compatible_force_work=True,
        axis_node_mass_convention="corner_subzonal",
        tri_fan_center_cfl_enabled=False,
        tri_fan_center_cfl_safety=0.5,
        tri_fan_center_cfl_band_radial_index=3,
    ),
    ale=dict(
        enabled=True,
        every_n_steps=1,
        max_iterations=100,
        convergence_tol=1.0e-9,
        max_displacement_fraction=0.5,
        axis_repair_mode="full_winslow",
        rezone_solver="legacy_winslow",
        euler_window=dict(
            enabled=True,
            shape="spherical_disk",
            role="axis_survival_core",
            cr=0.0,
            cz=0.0,
            rad_in=0.0,
            rad_out=R_DISK,
            transition_width=6.0e-5,
            feather_min_layers=3,
            guard_layers=1,
            axis_core_transaction_mode="static",
            t_on_s=0.0,
            t_off_s=-1.0,
        ),
        conservative_remap_enabled=True,
    ),
    conduction=dict(enabled=False),
    plic=dict(enabled=False),
    diagnostics=dict(
        phase_resolved_energy=True,
        mesh_quality_min=dict(enabled=True),
    ),
    diagnostics_every=50,
    floors=dict(
        rho_floor_gcc=1.0e-12,
        Te_floor_eV=1.0e-4,
        Ti_floor_eV=1.0e-4,
    ),
)

Output(
    directory=OUTDIR,
    format="hdf5",
    plot_every=MAX_STEPS,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    save_namelist_copy=True,
    save_frozen_config=True,
    plot_fields=["rho", "Te", "Ti"],
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(
    enabled=True,
    every=50,
    energy_budget=dict(enabled=True, warn_threshold=1.0),
)
