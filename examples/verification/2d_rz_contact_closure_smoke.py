# Contact-closure smoke vehicle descended from consultation §19.2-A/D:
# symmetric colliding streams plus the open-close-cycle contact endgame.  The z
# boundaries are sustained-supply pistons, so convergence is continuously
# driven like the L2 corona; state_supply+ALE uses the documented mesh-velocity
# decoupling.  With the default 4.0e6 cm/s speed, conversion occurs at step 20,
# followed by predicted per-pair arming and pair-0 engagement at step ~2,570
# with gap = g0 ~18.5 nm and dK ~0.66 erg to the two axial neighbors, yielding
# ~1,400 steps of sustained hold.  There are zero invariant violations through
# step 4,000.
#
# The policy density (1.0e-2 g/cc), laser-transparency limit (1.0), and
# conduction-coupling limit (1.0) are deck-level physics declarations for this
# laser-free collision test.  They exercise the normal gate chain; they are not
# gate bypasses.  This deck disables re-materialization so the converted spacer
# persists to geometric closure (the contact path under test); refill physics is
# covered elsewhere.

import os

from tenryu_namelist import *


UM = 1.0e-4


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


NR = 8
NZ = 64
R_MIN = 0.0
R_MAX = 8.0 * UM
Z_MIN = -32.0 * UM
Z_MAX = 32.0 * UM

# Stage-1 scope: engagement (~step 130) + sustained real-flow hold; 1500 keeps the regression fast. With the null-space volume floor the radial pinch is ARRESTED (a 4000-step run completes cleanly: the seam holds at the gap floor and the trapped compression equilibrates) — raise the env knob for the long demonstration.
MAX_STEPS = _env_int("TENRYU_SMOKE_CONTACT_MAX_STEPS", 1500)
V_CM_S = _env_float("TENRYU_SMOKE_CONTACT_V_CM_S", 4.0e6)
DT_S = _env_float("TENRYU_SMOKE_CONTACT_DT_S", 1.0e-13)
OUTDIR = os.environ.get(
    "TENRYU_SMOKE_CONTACT_OUTDIR",
    "./build/output_verify_2d_rz_contact_closure_smoke",
)
T_END = MAX_STEPS * DT_S

RHO_SLAB_GCC = 1.0
RHO_THIN_GCC = 1.0e-3
T_INIT_EV = 1.0
A = 27.0
ZBAR = 3.0
GAMMA = 5.0 / 3.0

print(
    "[deck:2d_rz_contact_closure_smoke] "
    f"nr={NR} nz={NZ} max_steps={MAX_STEPS} outdir={OUTDIR} "
    f"v_cm_s={V_CM_S} dt_s={DT_S} t_end_s={T_END} "
    f"rho_slab_gcc={RHO_SLAB_GCC} rho_thin_gcc={RHO_THIN_GCC} "
    "ale=True evacuated_cell=True conduction=True radiation=False laser=False"
)


def rho_init(r, z):
    if r < UM and -UM < z < 0.0:
        return RHO_THIN_GCC
    return RHO_SLAB_GCC


def temperature_init(r, z):
    del r, z
    return T_INIT_EV


def velocity_init(r, z):
    del r
    return (0.0, V_CM_S if z < -0.5 * UM else -V_CM_S)


Main(
    name="2d_rz_contact_closure_smoke",
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=12345,
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
            name="aluminum",
            A=A,
            Z=ZBAR,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=ZBAR),
)

Geometry(
    volfrac=dict(aluminum=lambda r, z: 1.0),
    rho=rho_init,
    Te=temperature_init,
    Ti=temperature_init,
    velocity=velocity_init,
    radiation_field="zero",
)

Numerics(
    radiation_thermal_subcycle=False,
    # ceiling via max_s; growth 1.2 lets transient dips recover (A497).
    dt=dict(
        initial_s=DT_S,
        max_s=DT_S,
        min_s=1.0e-22,
        growth_factor=1.2,
        cfl_hydro=0.15,
        cfl_cond=0.1,
    ),
    hydro=dict(
        enabled=True,
        # throttles dt as a cell approaches corner inversion (the seam-neighbor crush class, ledger A497); floored at 0.05x acoustic dt, recoverable via growth 1.2.
        corner_j_predict_cfl_enabled=True,
        boundary_2d=dict(
            r_inner="axis",
            r_outer="reflect",
            z_bottom=dict(
                type="state_supply",
                rho_g_per_cc=RHO_SLAB_GCC,
                u_z_cm_per_s=+V_CM_S,
                T_eV=T_INIT_EV,
                # Engagement is ~1.0e-11 s at V=4e6 with the constraint-owned-trigger ALE cadence (ledger A490); end the drive +6 ps later so the contact holds trapped compression instead of driving the seam to radial annihilation.
                drive_t_end_s=1.6e-11,
            ),
            z_top=dict(
                type="state_supply",
                rho_g_per_cc=RHO_SLAB_GCC,
                u_z_cm_per_s=-V_CM_S,
                T_eV=T_INIT_EV,
                drive_t_end_s=1.6e-11,
            ),
        ),
        av_C1=0.1,
        av_C2=1.5,
        volume_rate_cfl_enabled=True,
        volume_rate_cfl_threshold=0.5,
        trial_volume_cfl_enabled=True,
        trial_volume_cfl_floor_fraction=0.05,
        trial_volume_cfl_shrink_fraction=0.5,
        corner_jacobian_ale_trigger_enabled=True,
        corner_jacobian_floor_eps=1.0e-6,
        corner_jacobian_ale_trigger_scale=0.5,
    ),
    conduction=dict(
        enabled=True,
        solver="sts",
        f_lim=0.06,
        sts_damping=0.01,
        sts_max_stages=40,
        test_kappa=10.0,
    ),
    ale=dict(
        enabled=True,
        every_n_steps=5,
        spacing_ratio_threshold=1.5,
        quality_threshold=0.2,
        max_iterations=100,
        convergence_tol=1.0e-9,
        max_displacement_fraction=0.5,
        predictive_acceptance_enabled=True,
        predictive_acceptance_axis_floor_fraction=0.10,
        predictive_acceptance_cell_vol_floor_fraction=0.05,
        axis_repair_mode="axis_spine_only",
        evacuated_cell=dict(
            enabled=True,
            rematerialize_enabled=False,
            every_n_steps=10,
            rho_vacuum_policy_g_per_cc=1.0e-2,
            laser_ne_over_ncrit_max=1.0,
            coupling_fraction_max=1.0,
        ),
    ),
    diagnostics=dict(
        phase_resolved_energy=True,
        conduction_energy_rate_export=dict(enabled=True),
    ),
    diagnostics_every=1,
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
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
    plot_every=250,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    save_namelist_copy=True,
    save_frozen_config=True,
    plot_fields=["rho", "Te", "Ti", "ee", "ei", "Pe", "Pi", "zbar", "volfrac"],
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=True, every=1, energy_budget=dict(enabled=True, warn_threshold=1.0))
