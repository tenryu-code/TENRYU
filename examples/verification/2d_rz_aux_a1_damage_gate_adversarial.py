import math
import os

from tenryu_namelist import *


def _env_bool(name, default="0"):
    return os.environ.get(name, default) == "1"


CASE = os.environ.get("TENRYU_A1_DAMAGE_CASE", "below").lower()
if CASE not in ("below", "above"):
    raise ValueError("TENRYU_A1_DAMAGE_CASE must be 'below' or 'above'")

NR = int(os.environ.get("TENRYU_A1_NR", "16"))
NZ = int(os.environ.get("TENRYU_A1_NZ", "16"))
SEED = int(os.environ.get("TENRYU_A1_SEED", "12345"))
OUTDIR = os.environ.get(
    "TENRYU_A1_OUTDIR", f"./build/output_verify_2d_rz_aux_a1_damage_gate_{CASE}"
)
T_END = float(os.environ.get("TENRYU_A1_T_END_S", "1.0e-8"))
MAX_STEPS = int(os.environ.get("TENRYU_A1_MAX_STEPS", "50"))
DISTORTION = float(os.environ.get("TENRYU_A1_DISTORTION", "0.60"))
QUALITY_THRESHOLD = float(os.environ.get("TENRYU_A1_QUALITY_THRESHOLD", "0.99"))
MAX_ITERATIONS = int(os.environ.get("TENRYU_A1_ALE_MAX_ITERATIONS", "80"))
RHO_LOW = float(os.environ.get("TENRYU_A1_DAMAGE_RHO_LOW", "1.0"))
RHO_HIGH = float(os.environ.get("TENRYU_A1_DAMAGE_RHO_HIGH", "19.0"))

R_MIN, R_MAX = 0.0, 1.0
Z_MIN, Z_MAX = -0.5, 0.5
DR = (R_MAX - R_MIN) / NR
DZ = (Z_MAX - Z_MIN) / NZ

RHO_CONTRAST = abs(RHO_HIGH - RHO_LOW) / max(RHO_HIGH + RHO_LOW, 1.0e-300)
EXPECTED_D_RHO = DISTORTION * RHO_CONTRAST
DEFAULT_DMAX = EXPECTED_D_RHO * (0.65 if CASE == "below" else 0.45)

REMAP_DAMAGE_GATE = _env_bool("TENRYU_A1_REMAP_DAMAGE_GATE", "1")
REMAP_DAMAGE_DMAX = float(os.environ.get("TENRYU_A1_REMAP_DAMAGE_DMAX", str(DEFAULT_DMAX)))
REMAP_DAMAGE_AXIS_ETA = float(os.environ.get("TENRYU_A1_REMAP_DAMAGE_AXIS_ETA", "1.0"))
REMAP_DAMAGE_AXIS_BUDGET = _env_bool("TENRYU_A1_REMAP_DAMAGE_AXIS_BUDGET", "0")
REMAP_DAMAGE_AXIS_BUDGET_FACTOR = float(
    os.environ.get("TENRYU_A1_REMAP_DAMAGE_AXIS_BUDGET_FACTOR", "2.0")
)
AXIS_REPAIR_MODE = os.environ.get("TENRYU_A1_AXIS_REPAIR_MODE", "full_winslow")
REMAP_SCHEME = os.environ.get("TENRYU_A1_REMAP_SCHEME", "legacy_split")
REMAP_MS2_LIMITER = os.environ.get("TENRYU_A1_REMAP_MS2_LIMITER", "van_leer")

CASE_NAME = f"2d_rz_aux_a1_damage_gate_{CASE}_nr{NR}_nz{NZ}_seed{SEED}"
EXPECTED_FIRE = CASE == "above"

print(
    "[deck:2d_rz_aux_a1_damage_gate_adversarial] "
    f"case={CASE} nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"t_end_s={T_END} max_steps={MAX_STEPS} distortion={DISTORTION} "
    f"quality_threshold={QUALITY_THRESHOLD} rho_low={RHO_LOW} rho_high={RHO_HIGH} "
    f"expected_D_rho_heuristic={EXPECTED_D_RHO} remap_damage_dmax={REMAP_DAMAGE_DMAX} "
    f"expected_gate_fire={EXPECTED_FIRE}"
)


def _node_index(value, lower, spacing):
    return int(round((value - lower) / spacing))


def rho_init(r, z):
    del z
    return RHO_LOW if r < 0.5 * (R_MIN + R_MAX) else RHO_HIGH


def temperature_init(r, z):
    return 1.0 / rho_init(r, z)


def velocity_init(r, z):
    i = _node_index(r, R_MIN, DR)
    j = _node_index(z, Z_MIN, DZ)
    if i <= 0 or i >= NR or j <= 0 or j >= NZ:
        return (0.0, 0.0)

    xi = i / NR
    eta = j / NZ
    radial_envelope = math.exp(-((xi - 0.5) / 0.18) ** 2)
    z_envelope = math.sin(math.pi * eta)
    checker = 1.0 if ((i + j) % 2) == 0 else -1.0
    vr = DISTORTION * DR * radial_envelope * z_envelope * checker / T_END
    vz = 0.25 * DISTORTION * DZ * math.sin(2.0 * math.pi * xi) * z_envelope / T_END
    return (vr, vz)


Main(name=CASE_NAME, dimension="2D_RZ", t_end=T_END, max_steps=MAX_STEPS, seed=SEED, verbosity="quiet")

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
    Te=temperature_init,
    Ti=temperature_init,
    velocity=velocity_init,
    radiation_field="zero",
)

Numerics(
    dt=dict(initial_s=T_END, cfl_hydro=0.5, growth_factor=1.0, max_s=T_END, min_s=1.0e-20),
    hydro=dict(
        boundary_2d=dict(r_inner="axis", r_outer="free", z_bottom="free", z_top="free"),
        av_C1=0.1,
        av_C2=1.5,
    ),
    conduction=dict(enabled=False),
    ale=dict(
        enabled=True,
        every_n_steps=1,
        quality_threshold=QUALITY_THRESHOLD,
        max_iterations=MAX_ITERATIONS,
        convergence_tol=1.0e-9,
        max_displacement_fraction=0.5,
        axis_repair_mode=AXIS_REPAIR_MODE,
        remap_damage_gate_enabled=REMAP_DAMAGE_GATE,
        remap_damage_dmax=REMAP_DAMAGE_DMAX,
        remap_damage_axis_eta=REMAP_DAMAGE_AXIS_ETA,
        remap_damage_axis_budget_enabled=REMAP_DAMAGE_AXIS_BUDGET,
        remap_damage_axis_budget_factor=REMAP_DAMAGE_AXIS_BUDGET_FACTOR,
        remap_scheme=REMAP_SCHEME,
        remap_ms2_limiter=REMAP_MS2_LIMITER,
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
