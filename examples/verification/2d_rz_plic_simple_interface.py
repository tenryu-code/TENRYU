from tenryu_namelist import *

R_IFACE = 5.1e-2

Main(
    name="plic_simple_interface",
    dimension="2D_RZ",
    t_end=1.0e-8,
    max_steps=1,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=1.0e-1,
    z_min=0.0,
    z_max=2.0e-2,
    nr=32,
    nz=8,
    grid="uniform",
    motion="ale",
)

Materials(
    materials=[
        Material(
            name="CD",
            A=6.0,
            Z=6.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        ),
        Material(
            name="H_CORONA",
            A=1.0,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        ),
    ],
    zbar=dict(model="fixed", fixed_value=1.0),
)

Geometry(
    volfrac=dict(
        CD=lambda r, z: 1.0 if r < R_IFACE else 0.0,
        H_CORONA=lambda r, z: 0.0 if r < R_IFACE else 1.0,
    ),
    rho=lambda r, z: 1.0,
    Te=lambda r, z: 5.0,
    Ti=lambda r, z: 5.0,
    velocity=lambda r, z: (0.0, 0.0),
    radiation_field="zero",
)

Numerics(
    T_start_eV=1.0e9,
    dt=dict(
        initial_s=1.0e-8,
        cfl_hydro=0.2,
        growth_factor=1.0,
        max_s=1.0e-8,
        min_s=1.0e-20,
    ),
    hydro=dict(
        boundary_2d=dict(r_inner="axis", r_outer="free", z_bottom="reflect", z_top="reflect"),
        av_C1=0.1,
        av_C2=1.5,
    ),
    conduction=dict(enabled=False),
    ale=dict(
        enabled=True,
        every_n_steps=1,
        quality_threshold=0.99,
        max_iterations=20,
        convergence_tol=1.0e-6,
        max_displacement_fraction=0.25,
    ),
    plic=dict(enabled=True),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Output(
    directory="./build/output_verify_plic_simple_interface",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=True)
