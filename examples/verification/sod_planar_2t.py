# DIAGNOSTIC VARIANT of sod_planar.py: temperature_model="2T" forces the
# two-temperature path, which disables the 1T energy renormalization
# (hydro_1d apply_1t_energy_renorm requires !use_two_temp). With Z=0 there
# are no electrons, so the dynamics are identical to the 1T deck; only the
# global ee renorm smear is absent. Used via TENRYU_SOD_DIAG_DECK for the
# clean convergence ladder (VERIFICATION.md 3.2s). Not wired as a default gate.
from tenryu_namelist import *

Main(
    name="sod_planar_2t",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=2.0e-7,
    max_steps=200000,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=1.0,
    nr=200,
    grid="uniform",
    geometry_1d="planar",
)

Materials(
    materials=[
        Material(
            name="gas",
            A=1.0,
            Z=0.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=1.4)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ]
)

Geometry(
    volfrac=dict(gas=lambda r: 1.0),
    rho=lambda r: 1.0 if r < 0.5 else 0.125,
    Te=lambda r: 1.0 if r < 0.5 else 0.8,
    Ti=lambda r: 1.0 if r < 0.5 else 0.8,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=1.0e-10,
        cfl_hydro=0.3,
        growth_factor=1.2,
        max_s=1.0e-7,
        min_s=1.0e-20,
    ),
    # Outer wall: "free" treats the exterior as vacuum and drives an inward
    # expansion rarefaction from x=1 (c_R*t ~ 0.21 cm) that corrupts the
    # right state; a reflecting wall is in equilibrium with the static
    # right state, so no wave is generated within the Sod window.
    hydro=dict(boundary_1d="reflect", av_C1=0.5, av_C2=1.5),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Output(
    directory="./build/output_verify_sod_planar_2t",
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
