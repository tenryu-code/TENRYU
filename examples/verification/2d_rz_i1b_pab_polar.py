"""I1B-PAB-POLAR Stage-1 a0=0 hydro-only pressure-drive deck."""

import os

from tenryu_namelist import *

from i1b_polar_common import (
    DEFAULT_C_STAR,
    DEFAULT_GAMMA,
    DEFAULT_NS_CELLS,
    DEFAULT_NTHETA_CELLS,
    DEFAULT_P0_DYN_CM2,
    DEFAULT_R_OUTER_CM,
    DEFAULT_RHO0_GCC,
    DEFAULT_T0_EV,
    DEFAULT_T_END_S,
    DEFAULT_T_HOLD_END_S,
    DEFAULT_T_RAMP_DOWN_S,
    DEFAULT_T_RAMP_UP_S,
    ablation_pressure_ramp,
    make_gamma_law_material,
)


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


PREFIX = "TENRYU_I1B_PAB_POLAR_"

NR = _env_int(PREFIX + "NR", DEFAULT_NS_CELLS)
NZ = _env_int(PREFIX + "NZ", DEFAULT_NTHETA_CELLS)
SEED = _env_int(PREFIX + "SEED", 12345)
MAX_STEPS = _env_int(PREFIX + "MAX_STEPS", 1000000)

R_OUTER = _env_float(PREFIX + "R_OUTER_CM", DEFAULT_R_OUTER_CM)
S_MAX = R_OUTER
GAMMA = DEFAULT_GAMMA
A_AMU = 1.0
ZBAR = 1.0
RHO0 = _env_float(PREFIX + "RHO0_GCC", DEFAULT_RHO0_GCC)
T0 = _env_float(PREFIX + "T0_EV", DEFAULT_T0_EV)

P0 = _env_float(PREFIX + "P0_DYN_CM2", DEFAULT_P0_DYN_CM2)
T_R = _env_float(PREFIX + "T_RAMP_UP_S", DEFAULT_T_RAMP_UP_S)
T_H = _env_float(PREFIX + "T_HOLD_END_S", DEFAULT_T_HOLD_END_S)
T_D = _env_float(PREFIX + "T_RAMP_DOWN_S", DEFAULT_T_RAMP_DOWN_S)
C_STAR = _env_float(PREFIX + "C_STAR", DEFAULT_C_STAR)
T_END = _env_float(PREFIX + "T_END_S", DEFAULT_T_END_S)

DT_INITIAL = _env_float(PREFIX + "DT_INITIAL_S", 2.0e-13)
DT_MAX = _env_float(PREFIX + "DT_MAX_S", 2.0e-13)
DT_GROWTH_FACTOR = _env_float(PREFIX + "DT_GROWTH_FACTOR", 1.02)
CFL_HYDRO = _env_float(PREFIX + "CFL_HYDRO", 0.15)
ALE_EVERY_N_STEPS = _env_int(PREFIX + "ALE_EVERY_N_STEPS", 5)
MESH_QUALITY_MIN_DIAG = _env_bool(PREFIX + "MESH_QUALITY_MIN_DIAG", True)
POLAR_CENTER_TREATMENT = os.environ.get(PREFIX + "POLAR_CENTER_TREATMENT", "annular")
POLAR_EQUAL_MU_ZONING = _env_bool(PREFIX + "POLAR_EQUAL_MU_ZONING", False)
CENTER_BUTTON_OUTER_NODE_RING = _env_int(PREFIX + "CENTER_BUTTON_OUTER_NODE_RING", 2)

PLOT_EVERY_STEPS = _env_int(PREFIX + "PLOT_EVERY_STEPS", 0)
HISTORY_EVERY_STEPS = _env_int(PREFIX + "HISTORY_EVERY_STEPS", 1)

if NR < 4 or NZ < 4:
    raise ValueError("TENRYU_I1B_PAB_POLAR_NR/NZ must both be >= 4")
if R_OUTER <= 0.0:
    raise ValueError("TENRYU_I1B_PAB_POLAR_R_OUTER_CM must be > 0")
if RHO0 <= 0.0 or T0 <= 0.0:
    raise ValueError("TENRYU_I1B_PAB_POLAR_RHO0_GCC and T0_EV must be > 0")
if P0 < 0.0:
    raise ValueError("TENRYU_I1B_PAB_POLAR_P0_DYN_CM2 must be >= 0")
if not (T_R > 0.0 and T_H >= T_R and T_D > 0.0):
    raise ValueError(
        "TENRYU_I1B_PAB_POLAR ramp times must satisfy "
        "T_RAMP_UP_S>0, T_HOLD_END_S>=T_RAMP_UP_S, T_RAMP_DOWN_S>0"
    )
if not (C_STAR > 1.0):
    raise ValueError("TENRYU_I1B_PAB_POLAR_C_STAR must be > 1")
if not (T_END > 0.0 and DT_INITIAL > 0.0 and DT_MAX > 0.0):
    raise ValueError("TENRYU_I1B_PAB_POLAR_T_END_S/DT_INITIAL_S/DT_MAX_S must be > 0")
if POLAR_CENTER_TREATMENT not in ("annular", "tri_fan", "button"):
    raise ValueError(
        "TENRYU_I1B_PAB_POLAR_POLAR_CENTER_TREATMENT must be annular, tri_fan, or button"
    )
if POLAR_CENTER_TREATMENT == "button" and CENTER_BUTTON_OUTER_NODE_RING < 1:
    raise ValueError(
        "TENRYU_I1B_PAB_POLAR_CENTER_BUTTON_OUTER_NODE_RING must be >= 1 when button is enabled"
    )

CASE_NAME = (
    f"i1b_pab_polar_a0_nr{NR}_nz{NZ}"
    f"_P{_safe_float_token(P0)}_C{_safe_float_token(C_STAR)}_seed{SEED}"
)
OUTDIR = os.environ.get(PREFIX + "OUTDIR", f"./build/output_verify_{CASE_NAME}")

CONSERVATIVE_REMAP_ENABLED = _env_bool(PREFIX + "CONSERVATIVE_REMAP_ENABLED", False)
CONSERVATIVE_REMAP_LAGRANGIAN_BULK_ENABLED = _env_bool(
    PREFIX + "CONSERVATIVE_REMAP_LAGRANGIAN_BULK_ENABLED", False
)
CONSERVATIVE_REMAP_LAGRANGIAN_BULK_CENTER_NODE_RING_MAX = _env_int(
    PREFIX + "CONSERVATIVE_REMAP_LAGRANGIAN_BULK_CENTER_NODE_RING_MAX", 4
)
AV_QCAP_OVER_P = _env_float(PREFIX + "AV_QCAP_OVER_P", 0.0)
AV_QCAP_CENTER_BAND_ONLY = _env_bool(PREFIX + "AV_QCAP_CENTER_BAND_ONLY", False)
TRI_FAN_CENTER_CFL_ENABLED = _env_bool(PREFIX + "TRI_FAN_CENTER_CFL_ENABLED", False)
TRI_FAN_CENTER_CFL_SAFETY = _env_float(PREFIX + "TRI_FAN_CENTER_CFL_SAFETY", 0.5)
TRI_FAN_CENTER_CFL_BAND_RADIAL_INDEX = _env_int(
    PREFIX + "TRI_FAN_CENTER_CFL_BAND_RADIAL_INDEX", 3
)
TRI_FAN_CENTER_PERTURBATION_DIAG_ENABLED = _env_bool(
    PREFIX + "TRI_FAN_CENTER_PERTURBATION_DIAG_ENABLED", False
)
TRI_FAN_TRACKING_REFERENCE_ENABLED = _env_bool(
    PREFIX + "TRI_FAN_TRACKING_REFERENCE_ENABLED", False
)
TRI_FAN_TRACKING_REFERENCE_MODE = os.environ.get(
    PREFIX + "TRI_FAN_TRACKING_REFERENCE_MODE", "seamless_converging"
)
TRI_FAN_TRACKING_REFERENCE_OMEGA = _env_float(
    PREFIX + "TRI_FAN_TRACKING_REFERENCE_OMEGA", 0.2
)
TRI_FAN_TRACKING_REFERENCE_G0 = _env_float(
    PREFIX + "TRI_FAN_TRACKING_REFERENCE_G0", 0.03
)
TRI_FAN_TRACKING_REFERENCE_NU = _env_float(
    PREFIX + "TRI_FAN_TRACKING_REFERENCE_NU", 0.15
)
TRI_FAN_TRACKING_REFERENCE_EPS_V = _env_float(
    PREFIX + "TRI_FAN_TRACKING_REFERENCE_EPS_V", 0.05
)
OPERATOR_SEMANTICS = "laser_off_radiation_off_conduction_off_pressure_drive"
BUTTON_HEADER = (
    f"center_button_outer_node_ring={CENTER_BUTTON_OUTER_NODE_RING} "
    if POLAR_CENTER_TREATMENT == "button"
    else ""
)

if TRI_FAN_TRACKING_REFERENCE_MODE not in ("legacy_lagging", "seamless_converging"):
    raise ValueError(
        "TENRYU_I1B_PAB_POLAR_TRI_FAN_TRACKING_REFERENCE_MODE must be "
        "legacy_lagging or seamless_converging"
    )
if not (0.0 <= TRI_FAN_TRACKING_REFERENCE_OMEGA <= 1.0):
    raise ValueError(
        "TENRYU_I1B_PAB_POLAR_TRI_FAN_TRACKING_REFERENCE_OMEGA must be in [0, 1]"
    )
if TRI_FAN_TRACKING_REFERENCE_G0 <= 0.0:
    raise ValueError(
        "TENRYU_I1B_PAB_POLAR_TRI_FAN_TRACKING_REFERENCE_G0 must be > 0"
    )
if TRI_FAN_TRACKING_REFERENCE_NU < 0.0:
    raise ValueError(
        "TENRYU_I1B_PAB_POLAR_TRI_FAN_TRACKING_REFERENCE_NU must be >= 0"
    )
if TRI_FAN_TRACKING_REFERENCE_EPS_V <= 0.0:
    raise ValueError(
        "TENRYU_I1B_PAB_POLAR_TRI_FAN_TRACKING_REFERENCE_EPS_V must be > 0"
    )
if CONSERVATIVE_REMAP_LAGRANGIAN_BULK_CENTER_NODE_RING_MAX < 0:
    raise ValueError(
        "TENRYU_I1B_PAB_POLAR_CONSERVATIVE_REMAP_LAGRANGIAN_BULK_CENTER_NODE_RING_MAX "
        "must be >= 0"
    )

print(
    "[deck:2d_rz_i1b_pab_polar] "
    "i1b_pab_polar_mode=True "
    f"a0=0 P0={P0} P0_units=dyn_per_cm2 "
    f"t_r_s={T_R} t_h_s={T_H} t_d_s={T_D} C_star={C_STAR} "
    f"S_max={S_MAX} S_max_units=cm gamma={GAMMA} rho0_gcc={RHO0} T0_eV={T0} "
    f"t_end_s={T_END} nr={NR} nz={NZ} seed={SEED} "
    f"av_model={os.environ.get(PREFIX + 'AV_MODEL', 'scalar_vnr_legacy')} "
    f"av_C1={_env_float(PREFIX + 'AV_C1', 0.1)} "
    f"av_C2={_env_float(PREFIX + 'AV_C2', 1.5)} "
    f"subzonal_pressure_enabled={_env_bool(PREFIX + 'SUBZONAL', False)} "
    f"outdir={OUTDIR} "
    f"ale_every_n_steps={ALE_EVERY_N_STEPS} "
    f"mesh_quality_min_diag={MESH_QUALITY_MIN_DIAG} "
    f"polar_center_treatment={POLAR_CENTER_TREATMENT} "
    f"{BUTTON_HEADER}"
    f"polar_equal_mu_zoning={POLAR_EQUAL_MU_ZONING} "
    f"conservative_remap_enabled={CONSERVATIVE_REMAP_ENABLED} "
    "conservative_remap_lagrangian_bulk_enabled="
    f"{CONSERVATIVE_REMAP_LAGRANGIAN_BULK_ENABLED} "
    "conservative_remap_lagrangian_bulk_center_node_ring_max="
    f"{CONSERVATIVE_REMAP_LAGRANGIAN_BULK_CENTER_NODE_RING_MAX} "
    f"tri_fan_tracking_reference_enabled={TRI_FAN_TRACKING_REFERENCE_ENABLED} "
    f"tri_fan_tracking_reference_mode={TRI_FAN_TRACKING_REFERENCE_MODE} "
    f"tri_fan_tracking_reference_g0={TRI_FAN_TRACKING_REFERENCE_G0} "
    f"tri_fan_tracking_reference_eps_v={TRI_FAN_TRACKING_REFERENCE_EPS_V} "
    f"operator_semantics={OPERATOR_SEMANTICS}"
)


def boundary_pressure(t):
    return ablation_pressure_ramp(t, P0, T_R, T_H, T_D)


def rho_init(r, z):
    return RHO0


def Te_init(r, z):
    return T0


def Ti_init(r, z):
    return T0


def velocity_init(r, z):
    return (0.0, 0.0)


Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity="quiet",
    temperature_model="2T",
)

mesh_kwargs = dict(
    r_min=0.0,
    r_max=S_MAX,
    z_min=-S_MAX,
    z_max=S_MAX,
    nr=NR,
    nz=NZ,
    grid="uniform",
    motion="ale",
    logical_mesh_2d="spherical_polar_halfplane",
    spherical_polar_s_max=S_MAX,
    polar_center_treatment=POLAR_CENTER_TREATMENT,
    polar_equal_mu_zoning=POLAR_EQUAL_MU_ZONING,
)
if POLAR_CENTER_TREATMENT == "button":
    mesh_kwargs["center_button_outer_node_ring"] = CENTER_BUTTON_OUTER_NODE_RING

Mesh(**mesh_kwargs)

Materials(
    materials=[
        make_gamma_law_material(
            Material,
            name="i1b_pab_polar_gas",
            gamma=GAMMA,
            A=A_AMU,
            Z=ZBAR,
        )
    ],
    zbar=dict(model="fixed", fixed_value=ZBAR),
)

Geometry(
    volfrac=dict(i1b_pab_polar_gas=lambda r, z: 1.0),
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
        max_s=DT_MAX,
        min_s=1.0e-22,
        growth_factor=DT_GROWTH_FACTOR,
        cfl_hydro=CFL_HYDRO,
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
        av_model=os.environ.get(PREFIX + "AV_MODEL", "scalar_vnr_legacy"),
        av_C1=_env_float(PREFIX + "AV_C1", 0.1),
        av_C2=_env_float(PREFIX + "AV_C2", 1.5),
        subzonal_pressure_enabled=_env_bool(PREFIX + "SUBZONAL", False),
        av_qcap_over_p=AV_QCAP_OVER_P,
        av_qcap_center_band_only=AV_QCAP_CENTER_BAND_ONLY,
        tri_fan_center_cfl_enabled=TRI_FAN_CENTER_CFL_ENABLED,
        tri_fan_center_cfl_safety=TRI_FAN_CENTER_CFL_SAFETY,
        tri_fan_center_cfl_band_radial_index=TRI_FAN_CENTER_CFL_BAND_RADIAL_INDEX,
        tri_fan_center_perturbation_diag_enabled=TRI_FAN_CENTER_PERTURBATION_DIAG_ENABLED,
    ),
    conduction=dict(enabled=False),
    ale=dict(
        enabled=True,
        every_n_steps=ALE_EVERY_N_STEPS,
        max_iterations=100,
        convergence_tol=1.0e-9,
        max_displacement_fraction=0.5,
        axis_repair_mode="full_winslow",
        rezone_solver="legacy_winslow",
        conservative_remap_enabled=CONSERVATIVE_REMAP_ENABLED,
        conservative_remap_lagrangian_bulk_enabled=(
            CONSERVATIVE_REMAP_LAGRANGIAN_BULK_ENABLED
        ),
        conservative_remap_lagrangian_bulk_center_node_ring_max=(
            CONSERVATIVE_REMAP_LAGRANGIAN_BULK_CENTER_NODE_RING_MAX
        ),
        tri_fan_tracking_reference_enabled=TRI_FAN_TRACKING_REFERENCE_ENABLED,
        tri_fan_tracking_reference_mode=TRI_FAN_TRACKING_REFERENCE_MODE,
        tri_fan_tracking_reference_omega=TRI_FAN_TRACKING_REFERENCE_OMEGA,
        tri_fan_tracking_reference_g0=TRI_FAN_TRACKING_REFERENCE_G0,
        tri_fan_tracking_reference_nu=TRI_FAN_TRACKING_REFERENCE_NU,
        tri_fan_tracking_reference_eps_v=TRI_FAN_TRACKING_REFERENCE_EPS_V,
    ),
    diagnostics=dict(
        phase_resolved_energy=True,
        mesh_quality_min=dict(enabled=MESH_QUALITY_MIN_DIAG),
        production_audit=dict(
            enabled=_env_bool(PREFIX + "PRODUCTION_AUDIT", False),
            tier="A",
            audit_json_path=os.path.join(OUTDIR, "audit_summary.json"),
        ),
    ),
    diagnostics_every=1,
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
    positivity=dict(clamp=True),
    safety=dict(energy_fatal=False, nan_fatal=True, energy_threshold=1.0),
)

Output(
    directory=OUTDIR,
    format="hdf5",
    plot_every=PLOT_EVERY_STEPS,
    history_every=HISTORY_EVERY_STEPS,
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
