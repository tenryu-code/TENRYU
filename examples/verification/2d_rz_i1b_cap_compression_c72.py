"""S3 cap C>=7.2 homologous-compression verification deck.

This gate certifies that the pinned TRI-fan cap topology carries an idealized
homologous Noh-class convergent implosion to ICF-class peak compression
C>=7.2 with positive 2D RZ cell volumes throughout, exact mass conservation,
and <=10% total-energy drift through the compression phase at N_C=8 on the
compatible staggered-Lagrangian pure-Lagrangian path with CSW-edge AV and
subzonal pressure.

It does not certify a physical PAB/ablation drive, post-stagnation/rebound
energy conservation, or fine-grid apex convergence where N_C refinement also
refines apex angular resolution and tightens dt scaling.
"""

import os

from tenryu_namelist import *


CASE_NAME = "2d_rz_i1b_cap_compression_c72"
OUTDIR = "./build/output_verify_2d_rz_i1b_cap_compression_c72"

N_C = 8
BRIDGE_LAYERS = 4
NR_SHELL = 8
NZ = 32
SEED = 12345
S_MAX = 0.07
R_C = S_MAX / 12.0
R_MATCH = 2.0 * R_C

A = 1.0
ZBAR = 1.0
GAMMA = 5.0 / 3.0
RHO0_GCC = 1.0e-3
T0_EV = 1.0
A_CONV = 8.0e8
P_EXT = 0.0

T_END = 5.0e-9
DT_INITIAL = 2.5e-11
DT_MAX = 5.0e-11
CFL_HYDRO = 0.2
MAX_STEPS = 8000
PLOT_EVERY_STEPS = 8
HISTORY_EVERY_STEPS = 1

print(
    "[deck:2d_rz_i1b_cap_compression_c72] "
    f"topology_scheme=multiblock_half_butterfly_trifan_cap_5block "
    f"n_c={N_C} fan_layers={BRIDGE_LAYERS} nr_shell={NR_SHELL} nz={NZ} "
    f"seed={SEED} S_max_cm={S_MAX} r_c_cm={R_C} r_match_cm={R_MATCH} "
    f"multiblock_cap_p=6.0 P_ext_dyn_cm2={P_EXT} rho_gcc={RHO0_GCC} "
    f"T0_eV={T0_EV} gamma={GAMMA} A={A} Z={ZBAR} A_conv_s_inv={A_CONV} "
    f"t_end_s={T_END} dt_initial_s={DT_INITIAL} dt_max_s={DT_MAX} "
    f"cfl_hydro={CFL_HYDRO} max_steps={MAX_STEPS} "
    f"plot_every_steps={PLOT_EVERY_STEPS} history_every_steps={HISTORY_EVERY_STEPS} "
    f"motion=lagrangian ale_enabled=False av_model=csw_edge "
    f"subzonal_pressure_enabled=True outdir={OUTDIR}"
)


def boundary_pressure(t):
    del t
    return P_EXT


def rho_init(r, z):
    del r, z
    return RHO0_GCC


def Te_init(r, z):
    del r, z
    return T0_EV


def Ti_init(r, z):
    del r, z
    return T0_EV


def velocity_init(r, z):
    return (-A_CONV * r, -A_CONV * z)


Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=S_MAX,
    z_min=-S_MAX,
    z_max=S_MAX,
    nr=NR_SHELL,
    nz=NZ,
    grid="uniform",
    motion="lagrangian",
    logical_mesh_2d="spherical_polar_halfplane",
    spherical_polar_s_max=S_MAX,
    topology_scheme="multiblock_half_butterfly_trifan_cap_5block",
    multiblock_cap_p=6.0,
    multiblock_cart_core_n_c=N_C,
    multiblock_cart_core_bridge_layers=BRIDGE_LAYERS,
    multiblock_cart_core_r_c=R_C,
    multiblock_cart_core_r_match=R_MATCH,
)

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

Numerics(
    radiation_thermal_subcycle=False,
    dt=dict(
        initial_s=DT_INITIAL,
        cfl_hydro=CFL_HYDRO,
        growth_factor=1.05,
        max_s=DT_MAX,
        min_s=1.0e-22,
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
        av_model="csw_edge",
        subzonal_pressure_enabled=True,
        av_C1=0.1,
        av_C2=1.5,
        trial_volume_cfl_enabled=True,
        mesh_quality_dt_cfl_enabled=True,
    ),
    ale=dict(enabled=False),
    conduction=dict(enabled=False),
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
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-4, Ti_floor_eV=1.0e-4),
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
