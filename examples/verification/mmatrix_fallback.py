from tenryu_namelist import *

Main(
    name="mmatrix_fallback",
    dimension="1D_SPH",
    t_end=1.0e-12,
    max_steps=40,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=1.0e4,
    r_max=1.0001e4,
    nr=10,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="ddmc_mat",
            A=12.0,
            Z=6.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=100.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=6.0),
)

Geometry(
    volfrac=dict(ddmc_mat=lambda r: 1.0),
    rho=lambda r: 1.0,
    Te=lambda r: 10.0,
    Ti=lambda r: 10.0,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

Numerics(
    dt=dict(initial_s=1.0e-13, max_s=1.0e-13, min_s=1.0e-20, growth_factor=1.0),
    hydro=dict(enabled=False),
    conduction=dict(enabled=False),
)

Radiation(
    enabled=True,
    groups=1,
    group_bounds_eV=[0.0, 1.0e6],
    imc=dict(
        particles_per_cell_group=100,
        implicit_capture=True,
        cutoff_fraction=0.0,
        inelastic_scatter=True,
        weight_cutoff=1.0e-10,
        roulette_survival=0.1,
    ),
    ddmc=dict(enabled=True, tau_ddmc=3.0, omega_ddmc=0.9, leak_stencil="4"),
    boundary=dict(inner_r="vacuum", outer_r="vacuum"),
)

Laser(enabled=False)

Output(
    directory="./build/output_verify_mmatrix_fallback",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
