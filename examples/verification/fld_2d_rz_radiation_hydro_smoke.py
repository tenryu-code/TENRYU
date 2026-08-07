import os

from tenryu_namelist import *


AXIS_REPAIR_MODE = os.environ.get("TENRYU_FLD_AXIS_REPAIR_MODE", "full_winslow")
REMAP_DAMAGE_GATE = os.environ.get("TENRYU_FLD_REMAP_DAMAGE_GATE", "0") == "1"
REMAP_DAMAGE_AXIS_BUDGET = os.environ.get("TENRYU_FLD_REMAP_DAMAGE_AXIS_BUDGET", "0") == "1"
REMAP_DAMAGE_DMAX = float(os.environ.get("TENRYU_FLD_REMAP_DAMAGE_DMAX", "0.05"))
REMAP_DAMAGE_AXIS_ETA = float(os.environ.get("TENRYU_FLD_REMAP_DAMAGE_AXIS_ETA", "0.02"))
REMAP_SCHEME = os.environ.get("TENRYU_FLD_REMAP_SCHEME", "legacy_split")
REMAP_MS2_LIMITER = os.environ.get("TENRYU_FLD_REMAP_MS2_LIMITER", "van_leer")
KE_CONSERVATION_CLOSURE = os.environ.get("TENRYU_FLD_KE_CLOSURE", "0") == "1"
KE_CLOSURE_REDISTRIBUTE_FLOOR = os.environ.get("TENRYU_FLD_KE_CLOSURE_REDISTRIBUTE_FLOOR", "0") == "1"

Main(
    name="fld_2d_rz_radiation_hydro_smoke",
    dimension="2D_RZ",
    t_end=5.0e-15,
    max_steps=5,
    seed=24680,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=0.1,
    z_min=-0.1,
    z_max=0.1,
    nr=8,
    nz=8,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="smoke_mat",
            A=1.0,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=1.0e-30, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=1.0),
)

Geometry(
    volfrac=dict(smoke_mat=lambda r, z: 1.0),
    rho=lambda r, z: 1.0e-3,
    Te=lambda r, z: 1.0,
    Ti=lambda r, z: 1.0,
    velocity=lambda r, z: (0.0, 0.0),
    radiation_field="zero",
)

Numerics(
    dt=dict(initial_s=1.0e-15, max_s=1.0e-15, min_s=1.0e-20, growth_factor=1.0),
    hydro=dict(
        enabled=True,
        boundary_2d=dict(r_inner="axis", r_outer="reflect", z_bottom="reflect", z_top="reflect"),
    ),
    conduction=dict(enabled=False),
    ale=dict(
        enabled=False,
        axis_repair_mode=AXIS_REPAIR_MODE,
        remap_damage_gate_enabled=REMAP_DAMAGE_GATE,
        remap_damage_axis_budget_enabled=REMAP_DAMAGE_AXIS_BUDGET,
        remap_damage_dmax=REMAP_DAMAGE_DMAX,
        remap_damage_axis_eta=REMAP_DAMAGE_AXIS_ETA,
        remap_scheme=REMAP_SCHEME,
        remap_ms2_limiter=REMAP_MS2_LIMITER,
        ke_conservation_closure=KE_CONSERVATION_CLOSURE,
        ke_closure_redistribute_floor=KE_CLOSURE_REDISTRIBUTE_FLOOR,
    ),
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=1,
    group_bounds_eV=[0.0, 1.0e4],
    multigroup_diffusion=dict(
        flux_limiter="levermore_pomraning",
        max_outer_iterations=5,
        outer_tol=1.0e-8,
        linear_solver_2d="cusparse_cg_jacobi",
        boundary=dict(inner_r="reflect", outer_r="vacuum", z="reflect"),
    ),
)

Laser(enabled=False)

Output(
    directory="./build/output_verify_fld_2d_rz_radiation_hydro_smoke",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
