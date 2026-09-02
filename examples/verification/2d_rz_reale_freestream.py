from tenryu_namelist import *


Main(
    name="2d_rz_reale_freestream",
    dimension="2D_RZ",
    temperature_model="1T",
    t_end=6.0e-13,
    max_steps=40,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    logical_mesh_2d="spherical_polar_halfplane",
    topology_scheme="multiblock_polar_tier",
    spherical_polar_s_max=0.010,
    multiblock_cart_core_r_match=0.0027083333333333334,
    polar_tier_hydro_enabled=True,
    polar_tier_dendrite_enabled=True,
    polar_tier_native_pentagon=True,
    polar_tier_dendrite_s_theta_rows_below=5,
    nr=96,
    nz=64,
    r_min=0.0,
    r_max=0.010,
    z_min=-0.010,
    z_max=0.010,
    grid="uniform",
    motion="ale",
)

Materials(
    materials=[
        Material(
            name="gas",
            A=1.0,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(
                model="constant",
                kappa_a=0.0,
                kappa_s=0.0,
                units="cm2_per_g",
            ),
        )
    ],
    zbar=dict(model="fixed", fixed_value=1.0),
)

Geometry(
    volfrac=dict(gas=lambda r, z: 1.0),
    rho=lambda r, z: 0.25,
    Te=lambda r, z: 1.0,
    Ti=lambda r, z: 1.0,
    velocity=lambda r, z: (0.0, 0.0),
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=1.0e-14,
        cfl_hydro=0.15,
        growth_factor=1.02,
        max_s=2.0e-13,
        min_s=1.0e-22,
        cfl_cond=0.10,
    ),
    hydro=dict(
        enabled=True,
        boundary_2d=dict(
            r_inner="axis",
            r_outer="reflect",
            z_bottom="reflect",
            z_top="reflect",
        ),
        av_C1=0.1,
        av_C2=1.5,
        av_model="csw_edge_csw98",
        csw98_damper_impulse_beta=0.15,
        rz_momentum_scheme="area_weighted_symmetric",
        aw_compatible_force_work=True,
        subzonal_pressure_enabled=True,
        axis_node_mass_convention="corner_subzonal",
    ),
    ale=dict(
        mesh_mode="reale_v2",
        enabled=False,
        force_rezone_every_n_steps=5,
        every_n_steps=5,
        max_iterations=100,
        convergence_tol=1.0e-9,
        max_displacement_fraction=0.5,
        axis_repair_mode="full_winslow",
        rezone_solver="legacy_winslow",
    ),
    conduction=dict(enabled=False),
    plic=dict(enabled=False),
    diagnostics=dict(phase_resolved_energy=True),
    floors=dict(
        rho_floor_gcc=1.0e-12,
        Te_floor_eV=1.0e-4,
        Ti_floor_eV=1.0e-4,
    ),
)

Output(
    directory="./build/output_2d_rz_reale_freestream",
    format="hdf5",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    save_namelist_copy=True,
    save_frozen_config=True,
)
