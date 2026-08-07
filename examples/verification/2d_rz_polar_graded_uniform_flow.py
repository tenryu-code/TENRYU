import math
import os

from tenryu_namelist import *


S_MAX = 0.07
RADIAL_ZONING = os.environ.get("TENRYU_G1_RADIAL", "auto").strip().lower()
THETA_ZONING = os.environ.get("TENRYU_G1_THETA", "uniform").strip().lower()
CENTER_TREATMENT = os.environ.get("TENRYU_G1_CENTER", "tri_fan").strip().lower()
NTHETA = int(os.environ.get("TENRYU_G1_NTHETA", "32"))
R_OUTER = os.environ.get("TENRYU_G1_R_OUTER", "pressure").strip().lower()
OUTDIR = os.environ.get(
    "TENRYU_G1_OUTDIR",
    "./build/output_g1_polar_graded",
)

if RADIAL_ZONING not in ("auto", "uniform"):
    raise ValueError("TENRYU_G1_RADIAL must be one of {'auto', 'uniform'}")
if THETA_ZONING not in ("uniform", "graded"):
    raise ValueError("TENRYU_G1_THETA must be one of {'uniform', 'graded'}")
if CENTER_TREATMENT not in ("tri_fan", "button"):
    raise ValueError("TENRYU_G1_CENTER must be one of {'tri_fan', 'button'}")
if NTHETA <= 0:
    raise ValueError("TENRYU_G1_NTHETA must be positive")
if R_OUTER not in ("pressure", "reflect", "free"):
    raise ValueError("TENRYU_G1_R_OUTER must be one of {'pressure', 'reflect', 'free'}")

SEED = 12345
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
T_END = 0.35 * S_MAX / CS_AMBIENT_CM_PER_S
PLOT_EVERY_S = float(os.environ.get("TENRYU_G1_PLOT_EVERY_S", str(T_END)))
# The auto radial ladder's equal-mass zoning gives dr_min ~5e-4 cm at the region
# interface and the graded-theta first ring ~6e-4 cm tangential: CFL ~9e-10 s at
# the doubled-pressure center. The 1e-9 value marginally blew up (measured GG
# inversion at steps 2-3, 2026-07-17); 1e-10 lets reactive dt control take over.
DT_INITIAL = 1.0e-10
# The reflect arc-pin excites a wedge boundary-layer secular instability against
# uniform ambient pressure (deterministic inversion ~step 60 at dt_max 1e-9,
# cells i=37-38 near-pole, 2026-07-17; root cause = open-contour reaction missing
# between position commit and boundary projection; ledgered for its own arc).
# The constant external pressure P0=ambient force-closes the contour exactly
# (I1-B PAB production pattern), leaving the interior blast symmetric and the
# box effectively closed.
DT_MAX = 1.0e-9
CFL_HYDRO = 0.5
MAX_STEPS = 4000

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

RESOLVED_NR = 40
RESOLVED_NTHETA = NTHETA if THETA_ZONING == "uniform" else 32
CASE_NAME = "g1_polar_graded_uniform_flow"

print(
    "[deck:g1_polar_graded] "
    f"radial={RADIAL_ZONING} theta={THETA_ZONING} "
    f"center={CENTER_TREATMENT} r_outer={R_OUTER} "
    f"nr={RESOLVED_NR} ntheta={RESOLVED_NTHETA} "
    f"seed={SEED} S_max_cm={S_MAX} pulse_radius_cm={PULSE_RADIUS_CM} "
    f"taper_radius_cm={TAPER_RADIUS_CM} t_end_s={T_END} "
    f"dt_initial_s={DT_INITIAL} dt_max_s={DT_MAX} cfl_hydro={CFL_HYDRO} "
    f"max_steps={MAX_STEPS} outdir={OUTDIR} rho_gcc={RHO_AMBIENT_GCC} "
    f"Pe_ambient_dyn_cm2={PE_AMBIENT_DYN_PER_CM2} "
    f"Pi_ambient_dyn_cm2={PI_AMBIENT_DYN_PER_CM2}"
)


def boundary_pressure(t):
    return P_TOTAL_AMBIENT_DYN_PER_CM2


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

mesh_kwargs = dict(
    r_min=0.0,
    r_max=S_MAX,
    z_min=-S_MAX,
    z_max=S_MAX,
    grid="uniform",
    motion="lagrangian",
    logical_mesh_2d="spherical_polar_halfplane",
    spherical_polar_s_max=S_MAX,
    polar_center_treatment=CENTER_TREATMENT,
    topology_scheme="single_block",
)
if RADIAL_ZONING == "auto":
    mesh_kwargs["auto_regions"] = [
        dict(r_end=S_MAX / 3.0, nz=16, rho_ref=1.0),
        dict(r_end=S_MAX, nz=24, rho_ref=0.05),
    ]
else:
    mesh_kwargs["nr"] = 40

if THETA_ZONING == "graded":
    # The shared grading default (edge_ratio 0.1) is the 1D refinement semantic; G1 tests the BAND structure, so segments are uniform inside.
    mesh_kwargs["grid_theta"] = dict(
        type="graded",
        grading=dict(edge_ratio=0.9999999999999999),
        segments=[
            dict(r_start=0.0, r_end=1.0471975511965976, nr=8),
            dict(r_start=1.0471975511965976, r_end=2.0943951023931953, nr=16),
            dict(r_start=2.0943951023931953, r_end=3.141592653589793, nr=8),
        ],
    )
else:
    mesh_kwargs["nz"] = NTHETA

Mesh(**mesh_kwargs)

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

hydro_kwargs = dict(
    enabled=True,
    # The pole rays are structural (axis-pinned); only the outer arc needed closing for G1.
    boundary_2d=dict(
        r_inner="axis",
        r_outer=R_OUTER,
        z_bottom="reflect",
        z_top="reflect",
    ),
    av_C1=0.1,
    av_C2=1.5,
    trial_volume_cfl_enabled=False,
    mesh_quality_dt_cfl_enabled=False,
)
if R_OUTER == "pressure":
    hydro_kwargs["boundary_pressure"] = boundary_pressure

Numerics(
    radiation_thermal_subcycle=False,
    dt=dict(
        initial_s=DT_INITIAL,
        cfl_hydro=CFL_HYDRO,
        growth_factor=1.1,
        max_s=DT_MAX,
        min_s=1.0e-22,
    ),
    hydro=hydro_kwargs,
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

# Node velocities are HDF5 mesh datasets, not supported plot_fields entries.
# The calibration runner therefore measures the cell fields rho, Te, and Ti.
Output(
    directory=OUTDIR,
    format="hdf5",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=PLOT_EVERY_S,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    save_namelist_copy=True,
    save_frozen_config=True,
    plot_fields=["rho", "Te", "Ti"],
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=True, every=1, energy_budget=dict(enabled=True, warn_threshold=1.0))
