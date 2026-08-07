import math
import os

from tenryu_namelist import *


def _env_bool(name, default="0"):
    return os.environ.get(name, default) == "1"


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p")


def _parse_grid(value):
    if not value:
        return None
    sep = "x" if "x" in value else ","
    parts = [part.strip() for part in value.lower().split(sep)]
    if len(parts) != 2:
        raise ValueError("TENRYU_A1_GRID must be formatted as NRxNZ or NR,NZ")
    return int(parts[0]), int(parts[1])


IC = os.environ.get("TENRYU_A1_IC", "sod").lower()
CADENCE = os.environ.get("TENRYU_A1_CADENCE", "quality_only").lower().replace("-", "_")
grid_env = _parse_grid(os.environ.get("TENRYU_A1_GRID", ""))
NR = int(os.environ.get("TENRYU_A1_NR", str(grid_env[0] if grid_env else 64)))
NZ = int(os.environ.get("TENRYU_A1_NZ", str(grid_env[1] if grid_env else 128)))
SEED = int(os.environ.get("TENRYU_A1_SEED", "12345"))
OUTDIR = os.environ.get("TENRYU_A1_OUTDIR", "./build/output_verify_2d_rz_a1")
DEBUG_PER_REMAP_LOG = _env_bool("TENRYU_A1_DEBUG_PER_REMAP_LOG", "1")
PRODUCTION_AUDIT_TIER = os.environ.get("TENRYU_A1_PRODUCTION_AUDIT_TIER", "none")
SWEPT_SIGN = os.environ.get("TENRYU_A1_SWEPT_SIGN", "")
CORNER_MASS = os.environ.get("TENRYU_A1_CORNER_MASS", "")
TIME_INTEGRATION = os.environ.get("TENRYU_A1_TIME_INTEGRATION", "")

if IC not in ("sod", "noh", "sedov"):
    raise ValueError("TENRYU_A1_IC must be one of: sod, noh, sedov")
if CADENCE not in ("quality_only", "every20", "every5", "every1"):
    raise ValueError("TENRYU_A1_CADENCE must be one of: quality_only, every20, every5, every1")
if PRODUCTION_AUDIT_TIER not in ("none", "A", "B"):
    raise ValueError('TENRYU_A1_PRODUCTION_AUDIT_TIER must be one of: "none", "A", "B"')
if SWEPT_SIGN not in ("", "0", "1"):
    raise ValueError('TENRYU_A1_SWEPT_SIGN must be one of: "", "0", "1"')
if CORNER_MASS not in ("", "bbsw_radial_v0", "kinematic_basis_rz_v1"):
    raise ValueError(
        'TENRYU_A1_CORNER_MASS must be one of: "", "bbsw_radial_v0", '
        '"kinematic_basis_rz_v1"'
    )
if TIME_INTEGRATION not in ("", "pc_v0", "midpoint_v1"):
    raise ValueError(
        'TENRYU_A1_TIME_INTEGRATION must be one of: "", "pc_v0", "midpoint_v1"'
    )
if NR <= 0 or NZ <= 0:
    raise ValueError("TENRYU_A1_NR and TENRYU_A1_NZ must be positive")

CADENCE_EVERY = {
    "quality_only": 1,
    "every20": 20,
    "every5": 5,
    "every1": 1,
}
DEFAULT_QUALITY_THRESHOLD = 0.2 if CADENCE == "quality_only" else 1.0
ALE_EVERY_N_STEPS = int(os.environ.get("TENRYU_A1_ALE_EVERY_N_STEPS", str(CADENCE_EVERY[CADENCE])))
QUALITY_THRESHOLD = float(
    os.environ.get("TENRYU_A1_QUALITY_THRESHOLD", str(DEFAULT_QUALITY_THRESHOLD))
)

AV_C2 = float(os.environ.get("TENRYU_A1_AV_C2", "1.5"))
CFL = float(os.environ.get("TENRYU_A1_CFL", "0.1"))
MAX_STEPS_DEFAULT = "60" if CADENCE == "every1" else "300"
MAX_STEPS = int(os.environ.get("TENRYU_A1_MAX_STEPS", MAX_STEPS_DEFAULT))
T_END_DEFAULT = {
    "sod": "2.0e-8",
    "noh": "1.0e-7",
    "sedov": "2.0e-9",
}[IC]
T_END = float(os.environ.get("TENRYU_A1_T_END_S", T_END_DEFAULT))

AXIS_Z_MOTION = os.environ.get("TENRYU_A1_AXIS_Z_MOTION", "fixed")
AXIS_REPAIR_MODE = os.environ.get("TENRYU_A1_AXIS_REPAIR_MODE", "axis_spine_only")
PREVENTIVE_AXIS_GUARD_FRACTION = float(
    os.environ.get("TENRYU_A1_PREVENTIVE_AXIS_GUARD_FRACTION", "0.1")
)
REMAP_DAMAGE_GATE = _env_bool("TENRYU_A1_REMAP_DAMAGE_GATE", "0")
REMAP_DAMAGE_DMAX = float(os.environ.get("TENRYU_A1_REMAP_DAMAGE_DMAX", "0.05"))
REMAP_DAMAGE_AXIS_ETA = float(os.environ.get("TENRYU_A1_REMAP_DAMAGE_AXIS_ETA", "0.02"))
REMAP_DAMAGE_AXIS_BUDGET = _env_bool("TENRYU_A1_REMAP_DAMAGE_AXIS_BUDGET", "0")
REMAP_DAMAGE_AXIS_BUDGET_FACTOR = float(
    os.environ.get("TENRYU_A1_REMAP_DAMAGE_AXIS_BUDGET_FACTOR", "2.0")
)
REMAP_SCHEME = os.environ.get("TENRYU_A1_REMAP_SCHEME", "ms2_moments")
REMAP_MS2_LIMITER = os.environ.get("TENRYU_A1_REMAP_MS2_LIMITER", "van_leer")
KE_CONSERVATION_CLOSURE = _env_bool("TENRYU_A1_KE_CLOSURE", "1")
KE_CONSERVATION_CLOSURE_AUDIT = _env_bool("TENRYU_A1_KE_CLOSURE_AUDIT", "0")
KE_CLOSURE_REDISTRIBUTE_FLOOR = _env_bool("TENRYU_A1_KE_CLOSURE_REDISTRIBUTE_FLOOR", "1")
PRODUCTION_AUDIT = dict(enabled=False, tier="none")
if PRODUCTION_AUDIT_TIER in ("A", "B"):
    PRODUCTION_AUDIT = dict(
        enabled=True,
        tier=PRODUCTION_AUDIT_TIER,
        gcl=dict(enabled=True),
        positivity=dict(enabled=True),
        audit_json_path=os.path.join(OUTDIR, "audit_summary.json"),
    )

EV_TO_ERG = 1.6022e-12
M_P = 1.6726219e-24  # must match core::constants::proton_mass
A = 1.0
GAMMA = 5.0 / 3.0
ZBAR = 1.0

CASE_NAME = f"a1_{IC}_{CADENCE}_nr{NR}_nz{NZ}_seed{SEED}"

print(
    "[deck:2d_rz_a1_ale_sweep] "
    f"ic={IC} cadence={CADENCE} nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"every_n_steps={ALE_EVERY_N_STEPS} quality_threshold={QUALITY_THRESHOLD} "
    f"debug_per_remap_log={DEBUG_PER_REMAP_LOG} remap_scheme={REMAP_SCHEME}"
)


cv_i = EV_TO_ERG / (A * M_P * (GAMMA - 1.0))
cv_e = ZBAR * EV_TO_ERG / (A * M_P * (GAMMA - 1.0))


def _temperature_from_pressure(p_erg_per_cc, rho_gcc):
    return p_erg_per_cc / (rho_gcc * (GAMMA - 1.0) * (cv_i + cv_e))


if IC == "sod":
    R_MIN, R_MAX = 0.0, 0.1
    Z_MIN, Z_MAX = -0.5, 0.5
    RHO_L_GCC = 1.0
    RHO_R_GCC = 0.125
    P_REF = 1.0e12
    P_L = P_REF
    P_R = 0.1 * P_REF
    T_L = _temperature_from_pressure(P_L, RHO_L_GCC)
    T_R = _temperature_from_pressure(P_R, RHO_R_GCC)
    TEMP_MODEL = "2T"

    def rho_init(r, z):
        del r
        return RHO_L_GCC if z < 0.0 else RHO_R_GCC

    def Te_init(r, z):
        del r
        return T_L if z < 0.0 else T_R

    def Ti_init(r, z):
        return Te_init(r, z)

    def velocity_init(r, z):
        del r, z
        return (0.0, 0.0)

    BOUNDARY_2D = dict(r_inner="axis", r_outer="reflect", z_bottom="reflect", z_top="reflect")
    DT_INITIAL = min(1.0e-11, T_END)
    DT_MAX = max(T_END / 20.0, DT_INITIAL)

elif IC == "noh":
    R_MIN, R_MAX = 0.0, 1.0
    Z_MIN, Z_MAX = -0.02, 0.02
    RHO_0_GCC = 1.0
    T_INIT = 1.0e-3
    U0 = 1.0e6
    TEMP_MODEL = "2T"

    def rho_init(r, z):
        del r, z
        return RHO_0_GCC

    def Te_init(r, z):
        del r, z
        return T_INIT

    def Ti_init(r, z):
        return Te_init(r, z)

    def velocity_init(r, z):
        del z
        return (-U0 if r > 0.0 else 0.0, 0.0)

    BOUNDARY_2D = dict(r_inner="axis", r_outer="free", z_bottom="reflect", z_top="reflect")
    DT_INITIAL = min(1.0e-11, T_END)
    DT_MAX = max(T_END / 20.0, DT_INITIAL)

else:
    R_MIN, R_MAX = 0.0, 1.2
    Z_MIN, Z_MAX = -1.2, 1.2
    RHO_0_GCC = 1.0
    E_TOTAL = float(os.environ.get("TENRYU_A1_SEDOV_E_TOTAL", "1.0e13"))
    BLAST_CELLS = int(os.environ.get("TENRYU_A1_SEDOV_BLAST_CELLS", "4"))
    P_AMBIENT = 1.0e6
    DR = (R_MAX - R_MIN) / NR
    DZ = (Z_MAX - Z_MIN) / NZ
    R_BLAST = BLAST_CELLS * DR
    T_AMBIENT = _temperature_from_pressure(P_AMBIENT, RHO_0_GCC)
    TEMP_MODEL = "2T"

    def _inside_blast(r, z):
        return math.hypot(r, z) < R_BLAST

    def _selected_blast_volume_cm3():
        volume = 0.0
        for i in range(NR):
            r_inner = R_MIN + i * DR
            r_outer = R_MIN + (i + 1) * DR
            r_cell = 0.5 * (r_inner + r_outer)
            for j in range(NZ):
                z_bottom = Z_MIN + j * DZ
                z_top = Z_MIN + (j + 1) * DZ
                z_cell = 0.5 * (z_bottom + z_top)
                if _inside_blast(r_cell, z_cell):
                    volume += math.pi * (r_outer * r_outer - r_inner * r_inner) * (
                        z_top - z_bottom
                    )
        return volume

    V_PATCH = _selected_blast_volume_cm3()
    if V_PATCH <= 0.0:
        raise ValueError("A1 Sedov blast patch selected no cells")
    T_BLAST = (E_TOTAL / (RHO_0_GCC * V_PATCH)) / (cv_i + cv_e)

    def rho_init(r, z):
        del r, z
        return RHO_0_GCC

    def Te_init(r, z):
        return T_BLAST if _inside_blast(r, z) else T_AMBIENT

    def Ti_init(r, z):
        return Te_init(r, z)

    def velocity_init(r, z):
        del r, z
        return (0.0, 0.0)

    BOUNDARY_2D = dict(r_inner="axis", r_outer="free", z_bottom="free", z_top="free")
    DT_INITIAL = min(1.0e-13, T_END)
    DT_MAX = max(T_END / 20.0, DT_INITIAL)


Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    temperature_model=TEMP_MODEL,
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity="quiet",
)

Mesh(
    r_min=R_MIN,
    r_max=R_MAX,
    z_min=Z_MIN,
    z_max=Z_MAX,
    nr=NR,
    nz=NZ,
    grid="uniform",
    motion="ale",
)

Materials(
    materials=[
        Material(
            name="fuel",
            A=A,
            Z=ZBAR,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=ZBAR),
)

Geometry(
    volfrac=dict(fuel=lambda r, z: 1.0),
    rho=rho_init,
    Te=Te_init,
    Ti=Ti_init,
    velocity=velocity_init,
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=DT_INITIAL,
        cfl_hydro=CFL,
        growth_factor=1.1,
        max_s=DT_MAX,
        min_s=1.0e-22,
    ),
    hydro=dict(
        boundary_2d=BOUNDARY_2D,
        av_C1=0.1,
        av_C2=AV_C2,
        **(
            {"corner_mass_convention": CORNER_MASS}
            if CORNER_MASS in ("bbsw_radial_v0", "kinematic_basis_rz_v1")
            else {}
        ),
        **(
            {"time_integration": TIME_INTEGRATION}
            if TIME_INTEGRATION in ("pc_v0", "midpoint_v1")
            else {}
        ),
    ),
    conduction=dict(enabled=False),
    ale=dict(
        enabled=True,
        every_n_steps=ALE_EVERY_N_STEPS,
        quality_threshold=QUALITY_THRESHOLD,
        max_iterations=int(os.environ.get("TENRYU_A1_ALE_MAX_ITERATIONS", "100")),
        convergence_tol=float(os.environ.get("TENRYU_A1_ALE_CONVERGENCE_TOL", "1.0e-9")),
        max_displacement_fraction=float(
            os.environ.get("TENRYU_A1_ALE_MAX_DISPLACEMENT_FRACTION", "0.5")
        ),
        preventive_axis_guard_fraction=PREVENTIVE_AXIS_GUARD_FRACTION,
        axis_z_motion=AXIS_Z_MOTION,
        axis_repair_mode=AXIS_REPAIR_MODE,
        remap_damage_gate_enabled=REMAP_DAMAGE_GATE,
        remap_damage_dmax=REMAP_DAMAGE_DMAX,
        remap_damage_axis_eta=REMAP_DAMAGE_AXIS_ETA,
        remap_damage_axis_budget_enabled=REMAP_DAMAGE_AXIS_BUDGET,
        remap_damage_axis_budget_factor=REMAP_DAMAGE_AXIS_BUDGET_FACTOR,
        remap_scheme=REMAP_SCHEME,
        remap_ms2_limiter=REMAP_MS2_LIMITER,
        **({"swept_volume_sign_fixed": SWEPT_SIGN == "1"} if SWEPT_SIGN in ("0", "1") else {}),
        ke_conservation_closure=KE_CONSERVATION_CLOSURE,
        ke_conservation_closure_audit=KE_CONSERVATION_CLOSURE_AUDIT,
        ke_closure_redistribute_floor=KE_CLOSURE_REDISTRIBUTE_FLOOR,
        debug_per_remap_log=DEBUG_PER_REMAP_LOG,
    ),
    diagnostics=dict(phase_resolved_energy=True, production_audit=PRODUCTION_AUDIT),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Output(
    directory=OUTDIR,
    plot_every=0,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=T_END / 4.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=True)
