from tenryu_namelist import *

Main(
    name="P2_imc_thin",
    dimension="2D_RZ",
    t_end=1.0e-8,
    max_steps=100,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=1.0,
    z_min=0.0,
    z_max=0.5,
    nr=500,
    nz=250,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="mat",
            A=1.0,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=0.1, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=1.0),
)

Geometry(
    volfrac=dict(mat=lambda r, z: 1.0),
    rho=lambda r, z: 1.0,
    Te=lambda r, z: 10.0,
    Ti=lambda r, z: 10.0,
    velocity=lambda r, z: (0.0, 0.0),
    radiation_field="zero",
)

Radiation(
    enabled=True,
    mode="imc_ddmc",
    groups=16,
    group_bounds_eV=[10.0 ** (-2.0 + 4.0 * i / 16.0) for i in range(17)],
    imc=dict(alpha=1.0, f_max=1.0, particles_per_cell_group=1),
    ddmc=dict(enabled=True, tau_ddmc=3.0, omega_ddmc=0.9, leak_stencil="9_kershaw"),
    boundary=dict(inner_r="marshak", outer_r="vacuum", bottom_z="reflect", top_z="reflect",
                  marshak_Tr_eV=100.0, marshak_particles=1000000),
)

Laser(enabled=False)

Numerics(
    dt=dict(initial_s=1.0e-10, max_s=1.0e-10, min_s=1.0e-20, growth_factor=1.0),
    hydro=dict(enabled=False),
    conduction=dict(enabled=False),
)

Output(
    directory="./output_perf_P2",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
)

Diagnostics(enabled=False)
