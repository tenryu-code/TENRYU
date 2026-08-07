from tenryu_namelist import *
from math import exp, sqrt

Main(
    name="radiation_symmetry_2d",
    dimension="2D_RZ",
    t_end=5.0e-11,
    max_steps=50,
    seed=42,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=2.0,
    z_min=-2.0,
    z_max=2.0,
    nr=50,
    nz=100,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="fuel",
            A=12.0,
            Z=6.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=1.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=6.0),
)

Geometry(
    volfrac=dict(fuel=lambda r, z: 1.0),
    rho=lambda r, z: 1.0,
    Te=lambda r, z: 100.0 * exp(-(sqrt(r * r + z * z) / 0.5) ** 2),
    Ti=lambda r, z: 100.0 * exp(-(sqrt(r * r + z * z) / 0.5) ** 2),
    velocity=lambda r, z: (0.0, 0.0),
    radiation_field="zero",
)

Numerics(
    dt=dict(initial_s=1.0e-12, max_s=1.0e-12, min_s=1.0e-20, growth_factor=1.0, f_min_fleck=0.01),
    hydro=dict(enabled=False, boundary_2d=dict(r_inner="axis", r_outer="free", z_bottom="free", z_top="free")),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Radiation(
    enabled=True,
    groups=1,
    group_bounds_eV=[0.0, 1.0e6],
    imc=dict(
        alpha=1.0,
        f_max=1.0,
        particles_per_cell_group=100,
        implicit_capture=True,
        cutoff_fraction=1.0e-4,
        inelastic_scatter=True,
        weight_cutoff=1.0e-10,
        roulette_survival=0.1,
        linearized_planck=False,
    ),
    ddmc=dict(enabled=False),
    boundary=dict(inner_r="reflect", outer_r="vacuum", bottom_z="vacuum", top_z="vacuum"),
)

Laser(enabled=False)

Output(
    directory="./build/output_verify_radiation_symmetry_2d",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
