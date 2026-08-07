from tenryu_namelist import *
import math

# W-H B2 gate deck: planar standing sound wave, AV off, conduction off,
# ideal gas gamma=5/3 with cs = 1e5 cm/s (e = 9e9 erg/g at Te=Ti below).
# The gate overlays a constant-eta Braginskii run on a control run and
# compares the differential decay rate to gamma = (2/3)(eta/rho) k^2.
# Sampling is by FRESH loads run to increasing t_end: driver.run in-memory
# re-entry is unsupported (start-up on an evolved state pumps total energy
# ~x1.9 per call — measured 2026-07-04; single-run E conserved to 4e-16).

Main(
    name="braginskii_wave",
    dimension="1D_SPH",
    t_end=8.0e-5,
    max_steps=2000000,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=1.0,
    nr=96,
    grid="uniform",
    geometry_1d="planar",
)

Materials(
    materials=[
        Material(
            name="gas",
            A=1.0,
            Z=0.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=1.6666666666666667)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ]
)

Geometry(
    volfrac=dict(gas=lambda r: 1.0),
    rho=lambda r: 1.0,
    Te=lambda r: 6.2637e-3,
    Ti=lambda r: 6.2637e-3,
    velocity=lambda r: 100.0 * math.sin(math.pi * r),
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=1.0e-9,
        cfl_hydro=0.3,
        growth_factor=1.2,
        max_s=1.0e-8,
        min_s=1.0e-20,
    ),
    # av_C1=0: the LINEAR AV term damps linear sound waves and would bias
    # the decay-rate fit. av_C2=1.5 (default-scale quadratic): inert for
    # smooth waves (q/p ~ 1e-9 at Mach 1e-3) but suppresses the odd-even
    # sawtooth instability of the AV-less staggered scheme; its residual
    # damping cancels in the viscous-minus-inviscid differential.
    hydro=dict(boundary_1d="reflect", av_C1=0.0, av_C2=1.5),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Output(
    directory="./build/output_verify_braginskii_wave",
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
