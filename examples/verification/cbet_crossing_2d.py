"""CBET 2D_RZ crossing-slab gate deck.

Small two-beam underdense corona slab for G2-2D conservation and G3-2D
detuning tendency. The wavelength ladder is controlled by
TENRYU_CBET_SLAB_2D_DELTA_LAMBDA_NM on beam 0.
"""
import math
import os

from tenryu_namelist import *


um = 1.0e-4
ps = 1.0e-12


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


NR = _env_int("TENRYU_CBET_SLAB_2D_NR", 48)
NZ = _env_int("TENRYU_CBET_SLAB_2D_NZ", 96)
SEED = _env_int("TENRYU_CBET_SLAB_2D_SEED", 12345)
RAYS_PER_BEAM = _env_int("TENRYU_CBET_SLAB_2D_RAYS_PER_BEAM", 24)
DELTA_LAMBDA_NM = _env_float("TENRYU_CBET_SLAB_2D_DELTA_LAMBDA_NM", 0.0)
CBET_ENABLE = _env_bool("TENRYU_CBET_SLAB_2D_ENABLE", True)
N_IMPACT_BINS = _env_int("TENRYU_CBET_SLAB_2D_N_IMPACT_BINS", 8)

R_MIN = 0.0
R_MAX = _env_float("TENRYU_CBET_SLAB_2D_R_MAX_CM", 160.0 * um)
Z_HALF = _env_float("TENRYU_CBET_SLAB_2D_Z_HALF_CM", 240.0 * um)
Z_MIN = -Z_HALF
Z_MAX = Z_HALF

WAVELENGTH_NM = 351.0
LAMBDA_UM = WAVELENGTH_NM * 1.0e-3
N_CRIT_CM3 = 1.115e21 / (LAMBDA_UM * LAMBDA_UM)
AVOGADRO = 6.02214076e23
A_CD = 7.0
Z_CD = 3.5
GAMMA = 5.0 / 3.0

TE_EV = _env_float("TENRYU_CBET_SLAB_2D_TE_EV", 1500.0)
TI_EV = _env_float("TENRYU_CBET_SLAB_2D_TI_EV", 750.0)
VZ0 = _env_float("TENRYU_CBET_SLAB_2D_VZ0_CM_S", 1.5e7)
VZ1 = _env_float("TENRYU_CBET_SLAB_2D_VZ1_CM_S", 0.5e7)

DT = _env_float("TENRYU_CBET_SLAB_2D_DT_S", 2.0e-13)
MAX_STEPS = _env_int("TENRYU_CBET_SLAB_2D_MAX_STEPS", 5)
T_END = _env_float("TENRYU_CBET_SLAB_2D_T_END_S", MAX_STEPS * DT)
PULSE_DURATION = _env_float("TENRYU_CBET_SLAB_2D_PULSE_DURATION_S", 10.0 * ps)
P_BEAM_W = _env_float("TENRYU_CBET_SLAB_2D_P_BEAM_W", 1.2e11)
SPOT_W0_UM = _env_float("TENRYU_CBET_SLAB_2D_SPOT_W0_UM", 120.0)

CASE_NAME = (
    "cbet_crossing_2d"
    f"_dl{_safe_float_token(DELTA_LAMBDA_NM)}"
    f"_bins{N_IMPACT_BINS}"
)
OUTDIR = os.environ.get("TENRYU_CBET_SLAB_2D_OUTDIR", f"outputs/{CASE_NAME}")

print(
    "[deck:cbet_crossing_2d] "
    f"nr={NR} nz={NZ} rays_per_beam={RAYS_PER_BEAM} "
    f"delta_lambda_nm={DELTA_LAMBDA_NM} cbet_enable={CBET_ENABLE} "
    f"n_impact_bins={N_IMPACT_BINS} outdir={OUTDIR}"
)


def _smooth_step(x):
    return 0.5 * (1.0 + math.tanh(x))


def ne_ncrit_profile(r, z):
    axial = 0.14 + 0.14 * _smooth_step((z + 0.15 * Z_HALF) / (0.35 * Z_HALF))
    radial = 1.0 - 0.06 * min((r / max(R_MAX, 1.0e-30)) ** 2, 1.0)
    return max(0.08, min(0.28, axial * radial))


def rho_profile(r, z):
    ne_cm3 = ne_ncrit_profile(r, z) * N_CRIT_CM3
    return ne_cm3 * A_CD / (Z_CD * AVOGADRO)


def Te_profile(r, z):
    del r, z
    return TE_EV


def Ti_profile(r, z):
    del r, z
    return TI_EV


def velocity_profile(r, z):
    del r
    return (0.0, VZ0 + VZ1 * math.tanh(z / (0.4 * Z_HALF)))


def vf_cd(r, z):
    del r, z
    return 1.0


def laser_power(t_s):
    if 0.0 <= t_s <= PULSE_DURATION:
        return P_BEAM_W
    return 0.0


def beam_direction(theta_deg):
    theta = math.radians(theta_deg)
    return (math.sin(theta), 0.0, -math.cos(theta))


Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity=os.environ.get("TENRYU_CBET_SLAB_2D_VERBOSITY", "quiet"),
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
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=Z_CD),
)

Geometry(
    volfrac=dict(CD=vf_cd),
    rho=rho_profile,
    Te=Te_profile,
    Ti=Ti_profile,
    velocity=velocity_profile,
    radiation_field="zero",
    enforce_sum_to_one=True,
)

Radiation(enabled=False)

Laser(
    enabled=True,
    wavelength_nm=WAVELENGTH_NM,
    mode="raytrace_3d",
    rays_per_beam=RAYS_PER_BEAM,
    ray_output_count=0,
    ray_output_trajectory=False,
    absorption=dict(eps_n=1.0e-4, coulomb_log_floor=2.0),
    lasermesh=dict(nr=64, nz=128, r_max_factor=1.15, z_span_factor=1.15, critical_margin=0.9999),
    raytrace=dict(cfl_ray=0.8, eps_crit=1.0e-4, intensity_cutoff=1.0e-8, max_steps=50000),
    raytrace_skip_config=dict(enabled=False, threshold=0.01, max_consecutive=10),
    deposit=dict(deposit_smooth_passes=1, deposit_smooth_alpha=0.25),
    cbet=dict(
        enable=CBET_ENABLE,
        n_impact_bins=N_IMPACT_BINS,
        n_phi=8,
        tol=1.0e-4,
        max_iters=80,
        theta_cap=0.2,
    ),
    beams=[
        LaserBeam(
            name="beam_theta20",
            direction=beam_direction(20.0),
            focus=(0.0, 0.0, 0.0),
            f_number=2.0,
            profile=dict(model="super_gaussian", w0_um=SPOT_W0_UM, m=4),
            power=laser_power,
            delta_lambda_nm=DELTA_LAMBDA_NM,
        ),
        LaserBeam(
            name="beam_theta40",
            direction=beam_direction(40.0),
            focus=(0.0, 0.0, 0.0),
            f_number=2.0,
            profile=dict(model="super_gaussian", w0_um=SPOT_W0_UM, m=4),
            power=laser_power,
            delta_lambda_nm=0.0,
        ),
    ],
)

Numerics(
    dt=dict(
        initial_s=DT,
        max_s=DT,
        min_s=1.0e-22,
        growth_factor=1.0,
        cfl_hydro=0.25,
        cfl_cond=0.25,
    ),
    hydro=dict(
        enabled=True,
        boundary_2d=dict(r_inner="axis", r_outer="free", z_bottom="free", z_top="free"),
        av_C1=0.1,
        av_C2=1.5,
    ),
    conduction=dict(enabled=False),
    ale=dict(enabled=False),
    diagnostics_every=1,
    floors=dict(rho_floor_gcc=1.0e-9, Te_floor_eV=0.1, Ti_floor_eV=0.1),
    positivity=dict(clamp=True),
    safety=dict(
        energy_fatal=False,
        nan_fatal=True,
        energy_threshold=1.0,
        clamp_warn_threshold=0,
        clamp_fatal_threshold=1000000000,
    ),
)

Output(
    directory=OUTDIR,
    format="hdf5",
    plot_every=0,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    save_namelist_copy=False,
    save_frozen_config=True,
    plot_fields=["rho", "Te", "Ti", "ee", "ei", "Pe", "Pi", "zbar", "volfrac"],
)

Diagnostics(
    enabled=True,
    every=1,
    energy_budget=dict(enabled=True, warn_threshold=1.0),
    laser_pattern=dict(enabled=True, absorbed_power_profile=True, critical_surface=True, per_beam=True),
)
