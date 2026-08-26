import math
import os

from tenryu_namelist import *


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


MODE = os.environ.get("TENRYU_L1_MODE", "vacuum_transparent").lower().replace("-", "_")
VALID_MODES = (
    "vacuum_transparent",
    "uniform_absorbing_cylinder",
    "super_gaussian",
    "cd_planar_slab",
    "cd_sphere_weak_pulse",
    "l1_conduction_submode",
    "l1_qei_submode",
    "l1_multibeam",
)
if MODE not in VALID_MODES:
    raise ValueError("TENRYU_L1_MODE must be one of: " + ", ".join(VALID_MODES))

MODE_DEFAULTS = {
    "vacuum_transparent": dict(
        t_end_s=1.0e-10,
        dt_s=1.0e-12,
        pulse_duration_s=1.0e-10,
        plot_every_s=1.0e-10,
        rho_cd_gcc=1.05,
        te0_ev=1.0,
        ti0_ev=1.0,
        laser_peak_w=1.0e10,
        spot_w0_um=3500.0,
    ),
    "uniform_absorbing_cylinder": dict(
        t_end_s=1.0e-10,
        dt_s=1.0e-12,
        pulse_duration_s=1.0e-10,
        plot_every_s=1.0e-10,
        rho_cd_gcc=1.05,
        te0_ev=1.0,
        ti0_ev=1.0,
        laser_peak_w=1.0e10,
        spot_w0_um=3500.0,
    ),
    "super_gaussian": dict(
        t_end_s=1.0e-10,
        dt_s=1.0e-12,
        pulse_duration_s=1.0e-10,
        plot_every_s=1.0e-10,
        rho_cd_gcc=1.0e-5,
        te0_ev=1.0,
        ti0_ev=1.0,
        laser_peak_w=1.0e10,
        spot_w0_um=3000.0,
    ),
    "cd_planar_slab": dict(
        t_end_s=1.0e-10,
        dt_s=1.0e-12,
        pulse_duration_s=1.0e-10,
        plot_every_s=1.0e-10,
        rho_cd_gcc=1.05,
        te0_ev=1.0,
        ti0_ev=1.0,
        laser_peak_w=1.0e10,
        spot_w0_um=3500.0,
    ),
    "cd_sphere_weak_pulse": dict(
        t_end_s=1.0e-10,
        dt_s=1.0e-12,
        pulse_duration_s=1.0e-10,
        plot_every_s=1.0e-10,
        rho_cd_gcc=1.05,
        te0_ev=1.0,
        ti0_ev=1.0,
        laser_peak_w=3.0e9,
        spot_w0_um=200.0,
    ),
    "l1_conduction_submode": dict(
        t_end_s=1.0e-9,
        dt_s=1.0e-12,
        pulse_duration_s=1.0e-10,
        plot_every_s=1.0e-10,
        rho_cd_gcc=1.05,
        te0_ev=1.0,
        ti0_ev=1.0,
        laser_peak_w=1.0e10,
        spot_w0_um=3500.0,
    ),
    "l1_qei_submode": dict(
        t_end_s=2.8e-8,
        dt_s=1.0e-11,
        pulse_duration_s=1.0e-10,
        plot_every_s=2.8e-8,
        rho_cd_gcc=1.0e-5,
        te0_ev=10.0,
        ti0_ev=1.0,
        laser_peak_w=1.0e10,
        spot_w0_um=3500.0,
    ),
    "l1_multibeam": dict(
        t_end_s=1.0e-10,
        dt_s=1.0e-12,
        pulse_duration_s=1.0e-10,
        plot_every_s=1.0e-10,
        rho_cd_gcc=1.05,
        te0_ev=1.0,
        ti0_ev=1.0,
        laser_peak_w=1.0e10,
        spot_w0_um=3500.0,
    ),
}
mode_defaults = MODE_DEFAULTS[MODE]

NR = int(os.environ.get("TENRYU_L1_NR", "64"))
NZ = int(os.environ.get("TENRYU_L1_NZ", "128"))
SEED = int(os.environ.get("TENRYU_L1_SEED", "12345"))
RAYS_PER_AXIS = int(os.environ.get("TENRYU_L1_N_RAYS_PER_AXIS", "72"))
N_BEAMS = int(os.environ.get("TENRYU_L1_NBEAMS", "2" if MODE == "l1_multibeam" else "1"))
OUTDIR = os.environ.get("TENRYU_L1_OUTDIR", "./build/output_verify_2d_rz_l1_laser_dep")
T_END = _env_float("TENRYU_L1_T_END_S", mode_defaults["t_end_s"])
DT_S = _env_float("TENRYU_L1_DT_S", mode_defaults["dt_s"])
PULSE_DURATION_DEFAULT = mode_defaults["pulse_duration_s"] if MODE in ("l1_conduction_submode", "l1_qei_submode") else T_END
PLOT_EVERY_S_DEFAULT = mode_defaults["plot_every_s"] if MODE == "l1_conduction_submode" else T_END
PULSE_DURATION = _env_float("TENRYU_L1_PULSE_DURATION_S", PULSE_DURATION_DEFAULT)
PLOT_EVERY_S = _env_float("TENRYU_L1_PLOT_EVERY_S", PLOT_EVERY_S_DEFAULT)
MAX_STEPS = int(os.environ.get("TENRYU_L1_MAX_STEPS", "1000000"))
P_PEAK_W = _env_float("TENRYU_L1_LASER_PEAK_W", mode_defaults["laser_peak_w"])
SPOT_W0_UM = _env_float("TENRYU_L1_LASER_W0_UM", mode_defaults["spot_w0_um"])
SPOT_M_DEFAULT = "2" if MODE == "super_gaussian" else "8"
SPOT_M = int(os.environ.get("TENRYU_L1_M", os.environ.get("TENRYU_L1_LASER_M", SPOT_M_DEFAULT)))

if N_BEAMS < 1:
    raise ValueError("TENRYU_L1_NBEAMS must be >= 1")
if MODE != "l1_multibeam" and N_BEAMS != 1:
    raise ValueError("TENRYU_L1_NBEAMS is only supported for TENRYU_L1_MODE=l1_multibeam")
if SPOT_M < 1:
    raise ValueError("TENRYU_L1_M/TENRYU_L1_LASER_M must be >= 1")

R_MIN = 0.0
R_MAX = 1.0
Z_MIN = -1.0
Z_MAX = 1.0

A_CD = 7.0
Z_CD = 3.5
RHO_CD = _env_float("TENRYU_L1_RHO_CD_GCC", mode_defaults["rho_cd_gcc"])
RHO_VAC = 1.0e-9
RHO_FLOOR = 1.0e-12
TE0_EV = _env_float("TENRYU_L1_TE0_EV", mode_defaults["te0_ev"])
TI0_EV = _env_float("TENRYU_L1_TI0_EV", mode_defaults["ti0_ev"])
TE_FLOOR = 0.1
TI_FLOOR = 0.1

CYLINDER_RADIUS = 0.5
CYLINDER_Z_HALF_WIDTH = 0.1
SLAB_Z_HALF_WIDTH = 0.1
SPHERE_RADIUS = 0.1
SUPER_GAUSSIAN_FOCUS_DISTANCE_CM = 100.0
SUPER_GAUSSIAN_TEST_KAPPA_CM_INV = 10.0
LASERMESH_R_MAX_FACTOR = 1.1
LASERMESH_Z_SPAN_FACTOR = 1.0 if MODE == "super_gaussian" else 1.1

CASE_NAME = (
    f"l1_{MODE}_nr{NR}_nz{NZ}_rays{RAYS_PER_AXIS}"
    f"_p{_safe_float_token(P_PEAK_W)}_seed{SEED}"
)
CONDUCTION_ENABLED = MODE in ("l1_conduction_submode", "l1_qei_submode")
RADIATION_ENABLED = MODE == "l1_qei_submode"
RADIATION_THERMAL_SUBCYCLE = MODE == "l1_qei_submode"

print(
    "[deck:2d_rz_l1_laser_dep] "
    f"mode={MODE} nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"t_end_s={T_END} dt_s={DT_S} rays_per_axis={RAYS_PER_AXIS} n_beams={N_BEAMS} "
    f"laser_power_W={P_PEAK_W} spot_w0_um={SPOT_W0_UM} spot_m={SPOT_M} hydro=False "
    f"radiation={RADIATION_ENABLED} conduction={CONDUCTION_ENABLED} "
    "ale=False laser_mode=raytrace_3d"
)


def is_cd(r, z):
    if MODE == "vacuum_transparent":
        return False
    if MODE == "uniform_absorbing_cylinder":
        return r <= CYLINDER_RADIUS and abs(z) <= CYLINDER_Z_HALF_WIDTH
    if MODE == "super_gaussian":
        del z
        return r <= CYLINDER_RADIUS
    if MODE in ("cd_planar_slab", "l1_conduction_submode", "l1_qei_submode", "l1_multibeam"):
        del r
        return abs(z) <= SLAB_Z_HALF_WIDTH
    if MODE == "cd_sphere_weak_pulse":
        return math.hypot(r, z) <= SPHERE_RADIUS
    return False


def rho_init(r, z):
    return RHO_CD if is_cd(r, z) else RHO_VAC


def super_gaussian_shape(r):
    w0_cm = max(SPOT_W0_UM * 1.0e-4, 1.0e-30)
    exponent = 2.0 * float(SPOT_M)
    r0_cm = w0_cm / math.pow(2.0, 1.0 / exponent)
    return math.exp(-math.pow(abs(r) / r0_cm, exponent))


def profiled_temperature(base_ev, floor_ev, r):
    if MODE != "super_gaussian":
        return base_ev
    return max(floor_ev, base_ev * super_gaussian_shape(r))


def Te_init(r, z):
    del z
    return profiled_temperature(TE0_EV, TE_FLOOR, r)


def Ti_init(r, z):
    del z
    return profiled_temperature(TI0_EV, TI_FLOOR, r)


def velocity_init(r, z):
    del r, z
    return (0.0, 0.0)


def cd_volfrac(r, z):
    del r, z
    return 1.0


def laser_power(t_s):
    if t_s < 0.0 or t_s > PULSE_DURATION:
        return 0.0
    return P_PEAK_W


conduction_config = dict(enabled=CONDUCTION_ENABLED)
if CONDUCTION_ENABLED:
    conduction_config.update(
        solver="sts",
        f_lim=0.06,
        sts_damping=0.01,
        sts_max_stages=40,
    )
    if MODE == "l1_conduction_submode":
        # Spitzer conduction with MFP limiter (MFP-limiter pattern from the L2 deck).
        # test_kappa=10 is too weak to drive observable smoothing in the
        # L1 pure-conduction (no hydro) 1 ns / 1 cm regime.
        conduction_config["mfp_limiter_C"] = 0.1
    elif MODE == "l1_qei_submode":
        # qei sub-mode keeps test_kappa=10 because the gate measures
        # |Te-Ti| relaxation (Qei time scale), not conduction smoothing.
        conduction_config["test_kappa"] = 10.0


def make_axial_beam(name, direction_z):
    focus_z = direction_z * SUPER_GAUSSIAN_FOCUS_DISTANCE_CM if MODE == "super_gaussian" else 0.0
    z_entry = LASERMESH_Z_SPAN_FACTOR if direction_z < 0.0 else -LASERMESH_Z_SPAN_FACTOR
    aperture_radius = CYLINDER_RADIUS if MODE == "super_gaussian" else None
    f_number = (
        abs(z_entry - focus_z) / (2.0 * aperture_radius)
        if aperture_radius is not None
        else 4.0
    )
    return LaserBeam(
        name=name,
        direction=[0.0, 0.0, direction_z],
        focus=[0.0, 0.0, focus_z],
        f_number=f_number,
        profile=dict(model="super_gaussian", w0_um=SPOT_W0_UM, m=SPOT_M),
        power=laser_power,
    )


if MODE == "l1_multibeam":
    beams = [
        make_axial_beam(
            f"axial_{'top' if i % 2 == 0 else 'bottom'}_{i}",
            -1.0 if i % 2 == 0 else 1.0,
        )
        for i in range(N_BEAMS)
    ]
else:
    beams = [make_axial_beam("axial_top", -1.0)]


Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity=os.environ.get("TENRYU_L1_VERBOSITY", "quiet"),
    temperature_model="2T",
)

Mesh(
    r_min=R_MIN,
    r_max=R_MAX,
    z_min=Z_MIN,
    z_max=Z_MAX,
    nr=NR,
    nz=NZ,
    grid="uniform",
    motion="lagrangian",
)

Materials(
    low_density_extrapolation=True,
    materials=[
        Material(
            name="CD",
            A=A_CD,
            Z=Z_CD,
            eos=dict(model="tmat", file="TMAT-H5/CD.tmat.h5"),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="tabular"),
)

Geometry(
    volfrac=dict(CD=cd_volfrac),
    rho=rho_init,
    Te=Te_init,
    Ti=Ti_init,
    velocity=velocity_init,
    radiation_field="zero",
)

Numerics(
    radiation_thermal_subcycle=RADIATION_THERMAL_SUBCYCLE,
    dt=dict(
        initial_s=DT_S,
        max_s=DT_S,
        min_s=1.0e-20,
        growth_factor=1.0,
        cfl_hydro=0.25,
        cfl_cond=0.25,
    ),
    hydro=dict(
        enabled=False,
        boundary_2d=dict(r_inner="axis", r_outer="free", z_bottom="free", z_top="free"),
        av_C1=0.1,
        av_C2=1.5,
    ),
    conduction=conduction_config,
    ale=dict(enabled=False),
    diagnostics=dict(phase_resolved_energy=True),
    diagnostics_every=1,
    floors=dict(rho_floor_gcc=RHO_FLOOR, Te_floor_eV=TE_FLOOR, Ti_floor_eV=TI_FLOOR),
    safety=dict(
        energy_fatal=False,
        nan_fatal=True,
        energy_threshold=1.0,
        clamp_warn_threshold=0,
        clamp_fatal_threshold=1000000000,
    ),
)

if MODE == "l1_qei_submode":
    Radiation(
        enabled=True,
        mode="multigroup_diffusion",
        groups=1,
        group_bounds_eV=[0.0, 1.0e6],
        imc=dict(enabled=False, two_stage=False, difference=dict(enabled=False)),
        ddmc=dict(enabled=False),
        holo=dict(enabled=False),
        multigroup_diffusion=dict(
            flux_limiter="levermore_pomraning",
            max_outer_iterations=5,
            outer_tol=1.0e-8,
            linear_solver_2d="auto",
            boundary=dict(inner_r="reflect", outer_r="vacuum", z="reflect"),
        ),
    )
else:
    Radiation(enabled=False)

raytrace_config = dict(cfl_ray=0.8, eps_crit=1.0e-4, intensity_cutoff=1.0e-8, max_steps=50000)
if MODE == "super_gaussian":
    raytrace_config["test_kappa"] = SUPER_GAUSSIAN_TEST_KAPPA_CM_INV

Laser(
    enabled=True,
    wavelength_nm=351.0,
    mode="raytrace_3d",
    rays_per_beam=RAYS_PER_AXIS,
    absorption=dict(eps_n=1.0e-4, coulomb_log_floor=2.0),
    lasermesh=dict(
        nr=64,
        nz=128,
        r_max_factor=LASERMESH_R_MAX_FACTOR,
        z_span_factor=LASERMESH_Z_SPAN_FACTOR,
        critical_margin=0.9999,
    ),
    raytrace=raytrace_config,
    raytrace_skip_config=dict(enabled=False, threshold=0.01, max_consecutive=10),
    deposit=dict(deposit_smooth_passes=1, deposit_smooth_alpha=0.25),
    beams=beams,
)

Output(
    directory=OUTDIR,
    plot_every=MAX_STEPS,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=PLOT_EVERY_S,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    plot_fields=["rho", "Te", "Ti", "ee", "ei", "Pe", "Pi", "zbar", "volfrac"],
)

Diagnostics(
    enabled=True,
    every=1,
    energy_budget=dict(enabled=True, warn_threshold=1.0),
    laser_pattern=dict(enabled=True, absorbed_power_profile=True, critical_surface=True, per_beam=False),
)
