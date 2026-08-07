from tenryu_namelist import *

# Electron-viscosity regime gate deck (design doc section 12, V-d): static
# uniform DT plasma (Z=1, A=2.5, Te=Ti=100 eV, n_i=5e21 cm^-3): R ~ 1.0e-2
# (ion-dominant).

Main(
    name="braginskii_regime_idom",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=1.0e-12,
    max_steps=100,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=1.0,
    nr=32,
    grid="uniform",
    geometry_1d="planar",
)

Materials(
    materials=[
        Material(
            name="gas",
            A=2.5,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=1.6666666666666667)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ]
)

Geometry(
    volfrac=dict(gas=lambda r: 1.0),
    rho=lambda r: 0.020908,  # n_i = 5e21 cm^-3 * 2.5 * m_p
    Te=lambda r: 100.0,
    Ti=lambda r: 100.0,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=1.0e-13,
        cfl_hydro=0.3,
        growth_factor=1.2,
        max_s=1.0e-13,
        min_s=1.0e-20,
    ),
    hydro=dict(
        boundary_1d="reflect",
        plasma_viscosity=dict(enabled=True, model="braginskii", species="both"),
    ),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Output(
    directory="./build/output_verify_braginskii_regime_idom",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=2.5e-13,
    checkpoint_every_s=-1.0,
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=True)
