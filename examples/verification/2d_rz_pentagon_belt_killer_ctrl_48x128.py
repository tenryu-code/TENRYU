import math
import os

from tenryu_namelist import *


CASE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTDIR = f"./output_{CASE_NAME}"

NR = 48
NTHETA = 128

GAMMA = 5.0 / 3.0
A = 1.0
ZBAR = 1.0
RHO0_GCC = 1.0
T0_EV = 1.0e-3
S_MAX_CM = 1.0
KAPPA = 4.0

EV_TO_ERG = 1.6022e-12
PROTON_MASS_G = 1.6726219e-24
SPECIFIC_INTERNAL_ENERGY_ERG_G = (
    (1.0 + ZBAR)
    * EV_TO_ERG
    * T0_EV
    / (A * PROTON_MASS_G * (GAMMA - 1.0))
)
P0_DYN_CM2 = (GAMMA - 1.0) * RHO0_GCC * SPECIFIC_INTERNAL_ENERGY_ERG_G
CS0_CM_S = math.sqrt(
    GAMMA * (GAMMA - 1.0) * SPECIFIC_INTERNAL_ENERGY_ERG_G
)
# Rigid-inner-wall radial pileup terminates both variants past crossing; measure angular squeeze through belt crossing before wall terminality.
V0_CM_S = 1.0 * CS0_CM_S

# For gamma=5/3 and upstream Mach 1.5, the planar reflected-shock
# Rankine-Hugoniot estimate is D/cs=sqrt(2)-1/2.
SHOCK_SPEED_CM_S = (math.sqrt(2.0) - 0.5) * CS0_CM_S
DELTA_S_CM = S_MAX_CM / (NR + KAPPA)
INNER_RING_RADIUS_CM = KAPPA * DELTA_S_CM
MATCHED_BELT_RADIUS_CM = (NR // 2 + KAPPA + 0.5) * DELTA_S_CM
CROSSING_TIME_ESTIMATE_S = (
    MATCHED_BELT_RADIUS_CM - INNER_RING_RADIUS_CM
) / SHOCK_SPEED_CM_S
# Pre-collapse healthy window; campaign metric = collapse onset + mesh quality at fixed t; full crossing campaign returns after P3 angular regularization.
T_END_S = 1.2e-6

print(
    f"[deck:{CASE_NAME}] nr={NR} ntheta={NTHETA} topology=single_block "
    f"rho0_gcc={RHO0_GCC} P0_dyn_cm2={P0_DYN_CM2} "
    f"cs0_cm_s={CS0_CM_S} v0_cm_s={V0_CM_S} "
    f"crossing_time_estimate_s={CROSSING_TIME_ESTIMATE_S} "
    f"t_end_s={T_END_S} outdir={OUTDIR}"
)


def rho_init(r, z):
    return RHO0_GCC


def temperature_init(r, z):
    return T0_EV


def velocity_init(r, z):
    radius = math.hypot(r, z)
    if radius == 0.0:
        return (0.0, 0.0)
    return (-V0_CM_S * r / radius, -V0_CM_S * z / radius)


Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    temperature_model="1T",
    t_end=T_END_S,
    seed=12345,
)

Mesh(
    r_min=0.0,
    r_max=S_MAX_CM,
    z_min=-S_MAX_CM,
    z_max=S_MAX_CM,
    nr=NR,
    nz=NTHETA,
    grid="uniform",
    motion="lagrangian",
    logical_mesh_2d="spherical_polar_halfplane",
    polar_center_treatment="annular",
    spherical_polar_s_max=S_MAX_CM,
    spherical_polar_kappa=KAPPA,
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
    Te=temperature_init,
    Ti=temperature_init,
    velocity=velocity_init,
    radiation_field="zero",
    enforce_sum_to_one=True,
)

Numerics(
    radiation_thermal_subcycle=False,
    dt=dict(
        initial_s=1e-15,
        # Structured-only guards cannot run on belt (P2-3 followup: belt-capable guard port); CFL 0.1 is the empirically-surviving dt scale (dense-output diagnostic completed).
        cfl_hydro=0.1,
        growth_factor=1.2,
        max_s=T_END_S,
        min_s=1.0e-20,
    ),
    hydro=dict(
        enabled=True,
        boundary_2d=dict(
            # Rigid no-slip inner wall; the axis-idiom projection leaves near-pole tangential slide free (followup: discrete cause of pole-ward slide).
            r_inner="axis",
            r_outer="free",
            z_bottom="reflect",
            z_top="reflect",
        ),
        av_model="csw_edge_csw98",
        av_C1=0.1,
        av_C2=1.5,
        subzonal_mass_enabled=False,
        subzonal_pressure_enabled=False,
        hllc_z_flux_2d_rz=False,
        mesh_quality_dt_cfl_enabled=False,
    ),
    conduction=dict(enabled=False),
    ale=dict(enabled=False, conservative_remap_enabled=False),
    plic=dict(enabled=False),
    diagnostics=dict(
        phase_resolved_energy=True,
        conservation=dict(enabled=True),
        mesh_quality_min=dict(enabled=True),
        production_audit=dict(
            enabled=True,
            tier="A",
            gcl=dict(enabled=True),
            positivity=dict(enabled=True, fatal_on_neg=True),
            audit_json_path=os.path.join(OUTDIR, "audit_summary.json"),
        ),
    ),
    diagnostics_every=1,
    floors=dict(
        rho_floor_gcc=1.0e-12,
        Te_floor_eV=T0_EV/100.0,
        Ti_floor_eV=T0_EV/100.0,
    ),
    safety=dict(energy_fatal=False, nan_fatal=True, energy_threshold=1.0),
)

Output(
    directory=OUTDIR,
    format="hdf5",
    plot_every=0,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=T_END_S / 16.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    save_namelist_copy=True,
    save_frozen_config=True,
    plot_fields=["rho", "Te", "Ti", "ee", "ei", "Pe", "Pi", "volfrac"],
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(
    enabled=True,
    every=1,
    energy_budget=dict(enabled=True, warn_threshold=1.0),
)
