import os

from tenryu_namelist import *


um = 1.0e-4
ns = 1.0e-9


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


GXII_REFERENCE_E_ERG = 3600.0 * 1.0e7
LASER_INPUT_ERG = _env_float("TENRYU_I1_RADIAL_LASER_INPUT_ERG", GXII_REFERENCE_E_ERG * 0.05)
PULSE_DURATION = _env_float("TENRYU_I1_RADIAL_PULSE_DURATION_S", 1.0 * ns)
P_PEAK_W = _env_float(
    "TENRYU_I1_RADIAL_LASER_PEAK_W",
    LASER_INPUT_ERG / 1.0e7 / max(PULSE_DURATION, 1.0e-300),
)

R_MIN = 0.0
R_MAX = _env_float("TENRYU_I1_RADIAL_R_MAX_CM", 200.0 * um)
N_R = _env_int("TENRYU_I1_RADIAL_NR", 256)
SEED = _env_int("TENRYU_I1_RADIAL_SEED", 12345)
RAYS_PER_BEAM = _env_int("TENRYU_I1_RADIAL_RAYS_PER_BEAM", 128)
MAX_STEPS = _env_int("TENRYU_I1_RADIAL_MAX_STEPS", 1000000)
OUTDIR = os.environ.get(
    "TENRYU_I1_RADIAL_OUTDIR", "./build/output_verify_i1_radial_5pct"
)

GAMMA = _env_float("TENRYU_I1_RADIAL_GAMMA", 5.0 / 3.0)
A_CD = 7.0
Z_CD = 3.5
KAPPA_A = _env_float("TENRYU_I1_RADIAL_KAPPA_A_CM2_G", 10.0)
RHO_CAPSULE = _env_float("TENRYU_I1_RADIAL_RHO_CAPSULE_GCC", 1.05)
RHO_BACKGROUND = _env_float("TENRYU_I1_RADIAL_RHO_BACKGROUND_GCC", 1.0e-3)
T_INITIAL_EV = _env_float("TENRYU_I1_RADIAL_T_INITIAL_EV", 1.0)
SHELL_R_OUTER = _env_float("TENRYU_I1_RADIAL_SHELL_R_OUTER_CM", 50.0 * um)
DT_INITIAL = _env_float("TENRYU_I1_RADIAL_DT_INITIAL_S", 5.0e-14)
DT_MAX = _env_float("TENRYU_I1_RADIAL_DT_MAX_S", 5.0e-14)
DT_GROWTH_FACTOR = _env_float("TENRYU_I1_RADIAL_DT_GROWTH_FACTOR", 1.0)
T_END = _env_float("TENRYU_I1_RADIAL_T_END_S", 5.0e-11)
PLOT_EVERY_STEPS = _env_int("TENRYU_I1_RADIAL_PLOT_EVERY_STEPS", 10)
HISTORY_EVERY_STEPS = _env_int("TENRYU_I1_RADIAL_HISTORY_EVERY_STEPS", 1)

F_NUMBER = _env_float("TENRYU_I1_RADIAL_F_NUMBER", 4.0)
SPOT_W0_UM = _env_float("TENRYU_I1_RADIAL_LASER_W0_UM", 200.0)
FOCUS_Z = _env_float("TENRYU_I1_RADIAL_FOCUS_Z_CM", 0.0)

CASE_NAME = (
    f"i1_radial_1d_5pct_grey_fld_nr{N_R}_rays{RAYS_PER_BEAM}"
    f"_e{_safe_float_token(LASER_INPUT_ERG)}_ka{_safe_float_token(KAPPA_A)}_seed{SEED}"
)

print(
    "[deck:i1_radial_1d_5pct_grey_fld] "
    f"nr={N_R} seed={SEED} outdir={OUTDIR} r_max_cm={R_MAX} "
    f"t_end_s={T_END} dt_initial_s={DT_INITIAL} dt_max_s={DT_MAX} "
    f"dt_growth_factor={DT_GROWTH_FACTOR} laser_input_erg={LASER_INPUT_ERG} "
    f"laser_peak_W={P_PEAK_W} pulse_duration_s={PULSE_DURATION} "
    f"rays_per_beam={RAYS_PER_BEAM} kappa_a_cm2_g={KAPPA_A} "
    "hydro=True conduction=True qei=True grey_fld=True dimension=1D_SPH"
)


def rho_profile(r_cm):
    return RHO_CAPSULE if r_cm < SHELL_R_OUTER else RHO_BACKGROUND


def Te_profile(r_cm):
    del r_cm
    return T_INITIAL_EV


def Ti_profile(r_cm):
    del r_cm
    return T_INITIAL_EV


def laser_power(t_s):
    return P_PEAK_W if 0.0 <= t_s <= PULSE_DURATION else 0.0


Main(
    name=CASE_NAME,
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity=os.environ.get("TENRYU_I1_RADIAL_VERBOSITY", "quiet"),
)

Mesh(
    r_min=R_MIN,
    r_max=R_MAX,
    nr=N_R,
    grid="uniform",
    motion="lagrangian",
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=0.1, Ti_floor_eV=0.1),
)

Materials(
    low_density_extrapolation=True,
    materials=[
        Material(
            name="CD",
            A=A_CD,
            Z=Z_CD,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=dict(model="constant", kappa_a=KAPPA_A, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=Z_CD),
)

Geometry(
    volfrac=dict(CD=lambda r: 1.0),
    rho=rho_profile,
    Te=Te_profile,
    Ti=Ti_profile,
    velocity=lambda r: 0.0,
    radiation_field="equilibrium",
    enforce_sum_to_one=True,
)

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=1,
    group_bounds_eV=[0.0, 1.0e6],
    imc=dict(enabled=False, two_stage=False, difference=dict(enabled=False)),
    ddmc=dict(enabled=False),
    holo=dict(enabled=False),
    multigroup_diffusion=dict(
        hydro_coupling="none",  # explicit opt-out: compatible_energy x gamma_r_43 unsupported (v1)
        flux_limiter="levermore_pomraning",
        max_outer_iterations=50,
        outer_tol=1.0e-8,
        boundary=dict(inner_r="reflect", outer_r="vacuum"),
    ),
    boundary=dict(inner_r="reflect", outer_r="vacuum"),
)

Laser(
    enabled=True,
    wavelength_nm=351.0,
    mode="raytrace_2d",
    rays_per_beam=RAYS_PER_BEAM,
    ray_output_count=0,
    ray_output_trajectory=False,
    absorption=dict(eps_n=1.0e-4, coulomb_log_floor=2.0),
    lasermesh=dict(
        mesh_factor=0.1,
        rmax_n_hat_threshold=0.001,
        critical_margin=0.9999,
        ghost_corona=dict(
            enabled=True,
            n_out=12,
            ne_min_frac=0.03,
            ne_max_frac=0.99,
            Te_min_eV=50.0,
            zbar_min=1.0,
            zbar_max=4.0,
            handoff_cells=6,
            handoff_decay=2.0,
            transition_enabled=True,
            transition_resolved_nhat=0.9,
            transition_resolved_cells=3,
            transition_density_exponent=1.0,
        ),
    ),
    raytrace=dict(
        ds_adapt_g_target=0.01,
        ds_adapt_tau_target=0.01,
        ds_adapt_max_factor=2.0,
        cfl_ray=0.8,
        eps_crit=1.0e-4,
        intensity_cutoff=1.0e-8,
        max_steps=50000,
    ),
    raytrace_skip_config=dict(enabled=False, threshold=0.01, max_consecutive=10),
    deposit=dict(deposit_smooth_passes=1, deposit_smooth_alpha=0.25),
    beams=[
        LaserBeam(
            name="radial_in",
            direction=(0.0, 0.0, -1.0),
            focus=(0.0, 0.0, FOCUS_Z),
            f_number=F_NUMBER,
            profile=dict(model="super_gaussian", w0_um=SPOT_W0_UM, m=8),
            power=laser_power,
        )
    ],
)

Numerics(
    radiation_thermal_subcycle=True,
    dt=dict(
        initial_s=DT_INITIAL,
        max_s=DT_MAX,
        min_s=1.0e-22,
        growth_factor=DT_GROWTH_FACTOR,
        cfl_hydro=0.15,
        cfl_cond=0.1,
        f_min_fleck=0.01,
    ),
    hydro=dict(
        enabled=True,
        boundary_1d="free",
        compatible_energy=True,
        av_C1=0.1,
        av_C2=1.5,
    ),
    conduction=dict(
        enabled=True,
        solver=os.environ.get("TENRYU_I1_RADIAL_COND_SOLVER", "sts"),
        f_lim=0.06,
        test_kappa=10.0,
        sts_damping=0.01,
        sts_max_stages=40,
    ),
    diagnostics_every=1,
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
    plot_every=PLOT_EVERY_STEPS,
    history_every=HISTORY_EVERY_STEPS,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(
    enabled=True,
    every=1,
    energy_budget=dict(enabled=True, warn_threshold=3.0e-3),
    areal_density=dict(enabled=True, angles_deg=[0.0]),
    sphericity=dict(enabled=True, rho_threshold=10.0, modes=[0, 2, 4]),
    laser_pattern=dict(enabled=True, per_beam=True),
)
