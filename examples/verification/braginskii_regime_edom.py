from tenryu_namelist import *

# Electron-viscosity regime gate deck (design doc section 12, V-d): static
# uniform Au-like plasma (Z=50, A=197, Te=Ti=1 keV, n_i=1e21 cm^-3), 2T,
# braginskii model, species="both". Expected R = eta_e/eta_i ~ 1.6e2
# (electron-dominant; lnLambda_ii floors at 2). Mean free paths are ~1e-6 cm
# << dr = 3.1e-2 cm, so the mfp cap (C=20) stays inactive and R is the pure
# coefficient ratio.

Main(
    name="braginskii_regime_edom",
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
            A=197.0,
            Z=50.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=1.6666666666666667)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ]
)

Geometry(
    volfrac=dict(gas=lambda r: 1.0),
    rho=lambda r: 0.32951,  # n_i = 1e21 cm^-3 * 197 * m_p
    Te=lambda r: 1000.0,
    Ti=lambda r: 1000.0,
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
    directory="./build/output_verify_braginskii_regime_edom",
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
