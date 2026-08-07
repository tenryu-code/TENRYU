import os
import math

from tenryu_namelist import *


def _env_bool(name, default="0"):
    return os.environ.get(name, default) == "1"


NR = int(os.environ.get("TENRYU_A1_NR", "16"))
NZ = int(os.environ.get("TENRYU_A1_NZ", "16"))
SEED = int(os.environ.get("TENRYU_A1_SEED", "12345"))
OUTDIR = os.environ.get("TENRYU_A1_OUTDIR", "./build/output_verify_2d_rz_aux_a1")
T_END = float(os.environ.get("TENRYU_A1_T_END_S", "1.0e-9"))
MODE = os.environ.get("TENRYU_A1_MODE", "default")
ANALYTIC_FIELD = os.environ.get("TENRYU_A1_ANALYTIC_FIELD", "gaussian")
DISTORTION = float(os.environ.get("TENRYU_A1_DISTORTION", "0.35"))
AXIS_DISTORTION = float(os.environ.get("TENRYU_A1_AXIS_DISTORTION", "0.49"))
QUALITY_THRESHOLD = float(os.environ.get("TENRYU_A1_QUALITY_THRESHOLD", "0.2"))
MAX_ITERATIONS = int(os.environ.get("TENRYU_A1_ALE_MAX_ITERATIONS", "80"))
AXIS_Z_MOTION = os.environ.get("TENRYU_A1_AXIS_Z_MOTION", "fixed")
AXIS_REPAIR_MODE = os.environ.get("TENRYU_A1_AXIS_REPAIR_MODE", "full_winslow")
REMAP_DAMAGE_GATE = _env_bool("TENRYU_A1_REMAP_DAMAGE_GATE")
REMAP_DAMAGE_AXIS_BUDGET = _env_bool("TENRYU_A1_REMAP_DAMAGE_AXIS_BUDGET")
REMAP_DAMAGE_DMAX = float(os.environ.get("TENRYU_A1_REMAP_DAMAGE_DMAX", "0.05"))
REMAP_DAMAGE_AXIS_ETA = float(os.environ.get("TENRYU_A1_REMAP_DAMAGE_AXIS_ETA", "0.02"))
REMAP_DAMAGE_AXIS_BUDGET_FACTOR = float(
    os.environ.get("TENRYU_A1_REMAP_DAMAGE_AXIS_BUDGET_FACTOR", "2.0")
)
REMAP_SCHEME = os.environ.get("TENRYU_A1_REMAP_SCHEME", "legacy_split")
REMAP_MS2_LIMITER = os.environ.get("TENRYU_A1_REMAP_MS2_LIMITER", "van_leer")
KE_CONSERVATION_CLOSURE = _env_bool("TENRYU_A1_KE_CLOSURE")
KE_CONSERVATION_CLOSURE_AUDIT = _env_bool("TENRYU_A1_KE_CLOSURE_AUDIT")
KE_CLOSURE_REDISTRIBUTE_FLOOR = _env_bool("TENRYU_A1_KE_CLOSURE_REDISTRIBUTE_FLOOR")
TOTAL_ENERGY_REMAP = _env_bool("TENRYU_A1_TOTAL_ENERGY_REMAP")

R_MIN, R_MAX = 0.0, 1.0
Z_MIN, Z_MAX = -0.5, 0.5
DR = (R_MAX - R_MIN) / NR
DZ = (Z_MAX - Z_MIN) / NZ

if MODE not in ("default", "identity", "analytic_smooth", "forced_distortion"):
    raise ValueError(
        "TENRYU_A1_MODE must be one of: default, identity, analytic_smooth, forced_distortion"
    )
if MODE == "analytic_smooth" and ANALYTIC_FIELD not in ("gaussian", "linear", "quadratic"):
    raise ValueError("TENRYU_A1_ANALYTIC_FIELD must be one of: gaussian, linear, quadratic")

CASE_NAME = f"2d_rz_aux_a1_ale_forced_nr{NR}_nz{NZ}_seed{SEED}"
if MODE != "default":
    CASE_NAME = f"{CASE_NAME}_{MODE}"

if MODE == "default":
    print(
        "[deck:2d_rz_aux_a1_ale_forced] "
        f"nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
        f"distortion={DISTORTION} axis_distortion={AXIS_DISTORTION} "
        f"quality_threshold={QUALITY_THRESHOLD} axis_repair_mode={AXIS_REPAIR_MODE} "
        f"remap_scheme={REMAP_SCHEME} ke_closure={KE_CONSERVATION_CLOSURE} "
        f"total_energy_remap={TOTAL_ENERGY_REMAP}"
    )
else:
    print(
        "[deck:2d_rz_aux_a1_ale_forced] "
        f"mode={MODE} analytic_field={ANALYTIC_FIELD} "
        f"nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
        f"distortion={DISTORTION} axis_distortion={AXIS_DISTORTION} "
        f"quality_threshold={QUALITY_THRESHOLD} axis_repair_mode={AXIS_REPAIR_MODE} "
        f"remap_scheme={REMAP_SCHEME} ke_closure={KE_CONSERVATION_CLOSURE} "
        f"total_energy_remap={TOTAL_ENERGY_REMAP}"
    )


def _node_index(value, lower, spacing):
    return int(round((value - lower) / spacing))


def rho_init(r, z):
    if MODE != "analytic_smooth":
        return 1.0
    z01 = (z - Z_MIN) / (Z_MAX - Z_MIN)
    if ANALYTIC_FIELD == "linear":
        return 1.0 + 0.20 * r + 0.10 * z01
    if ANALYTIC_FIELD == "quadratic":
        return 1.0 + 0.16 * r + 0.08 * z01 + 0.06 * r * r + 0.04 * z01 * z01
    return 1.0 + 0.25 * math.exp(-(((r - 0.55) / 0.20) ** 2 + ((z - 0.03) / 0.24) ** 2))


def velocity_init(r, z):
    if MODE == "identity":
        return (0.0, 0.0)
    i = _node_index(r, R_MIN, DR)
    j = _node_index(z, Z_MIN, DZ)
    if MODE == "analytic_smooth":
        if 0 < i < NR and 0 < j < NZ:
            xi = i / NR
            eta = j / NZ
            return (
                0.20 * DISTORTION * DR * math.sin(math.pi * xi) * math.sin(2.0 * math.pi * eta)
                / T_END,
                0.20 * DISTORTION * DZ * math.sin(2.0 * math.pi * xi) * math.sin(math.pi * eta)
                / T_END,
            )
        return (0.0, 0.0)
    if AXIS_REPAIR_MODE == "axis_spine_only":
        if i == 0 and 0 < j < NZ:
            z_sign = 1.0 if (j % 2) == 0 else -1.0
            return (0.0, AXIS_DISTORTION * DZ * z_sign / T_END)
        return (0.0, 0.0)
    if 0 < i < NR and 0 < j < NZ:
        rz_sign = 1.0 if ((i + j) % 2) == 0 else -1.0
        z_sign = 1.0 if (i % 2) == 0 else -1.0
        return (
            DISTORTION * DR * rz_sign / T_END,
            DISTORTION * DZ * z_sign / T_END,
        )
    return (0.0, 0.0)


Main(name=CASE_NAME, dimension="2D_RZ", t_end=T_END, max_steps=4, seed=SEED, verbosity="quiet")

Mesh(r_min=R_MIN, r_max=R_MAX, z_min=Z_MIN, z_max=Z_MAX, nr=NR, nz=NZ, grid="uniform", motion="ale")

Materials(
    materials=[
        Material(
            name="fuel",
            A=1.0,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=1.0),
)

Geometry(
    volfrac=dict(fuel=lambda r, z: 1.0),
    rho=rho_init,
    Te=lambda r, z: 1.0,
    Ti=lambda r, z: 1.0,
    velocity=velocity_init,
    radiation_field="zero",
)

Numerics(
    dt=dict(initial_s=T_END, cfl_hydro=0.5, growth_factor=1.0, max_s=T_END, min_s=1.0e-20),
    hydro=dict(
        boundary_2d=dict(r_inner="axis", r_outer="free", z_bottom="free", z_top="free"),
        av_C1=0.1,
        av_C2=1.5,
        total_energy_remap_2d_rz=TOTAL_ENERGY_REMAP,
    ),
    conduction=dict(enabled=False),
    ale=dict(
        enabled=True,
        every_n_steps=1,
        quality_threshold=QUALITY_THRESHOLD,
        max_iterations=MAX_ITERATIONS,
        convergence_tol=1.0e-9,
        max_displacement_fraction=0.5,
        axis_z_motion=AXIS_Z_MOTION,
        axis_repair_mode=AXIS_REPAIR_MODE,
        remap_damage_gate_enabled=REMAP_DAMAGE_GATE,
        remap_damage_dmax=REMAP_DAMAGE_DMAX,
        remap_damage_axis_eta=REMAP_DAMAGE_AXIS_ETA,
        remap_damage_axis_budget_enabled=REMAP_DAMAGE_AXIS_BUDGET,
        remap_damage_axis_budget_factor=REMAP_DAMAGE_AXIS_BUDGET_FACTOR,
        remap_scheme=REMAP_SCHEME,
        remap_ms2_limiter=REMAP_MS2_LIMITER,
        ke_conservation_closure=KE_CONSERVATION_CLOSURE,
        ke_conservation_closure_audit=KE_CONSERVATION_CLOSURE_AUDIT,
        ke_closure_redistribute_floor=KE_CLOSURE_REDISTRIBUTE_FLOOR,
    ),
    diagnostics=dict(phase_resolved_energy=True),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Output(
    directory=OUTDIR,
    plot_every=0,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=True)
