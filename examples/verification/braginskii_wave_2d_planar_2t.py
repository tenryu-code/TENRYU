from tenryu_namelist import *
import math

# W2 V3 gate deck: ionized 2T copy of the planar-z standing-wave deck.

Main(
    name="braginskii_wave_2d_planar_2t",
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=2.0e-7,
    max_steps=1000,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=0.1,
    z_min=0.0,
    z_max=1.0,
    nr=4,
    nz=64,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="gas",
            A=1.0,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=1.6666666666666667)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ]
)

Geometry(
    volfrac=dict(gas=lambda r, z: 1.0),
    rho=lambda r, z: 1.0,
    Te=lambda r, z: 6.2637e-3,
    Ti=lambda r, z: 6.2637e-3,
    velocity=lambda r, z: (0.0, 100.0 * math.sin(math.pi * z)),
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=1.0e-11,
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
    hydro=dict(
        boundary_2d=dict(
            r_inner="axis",
            r_outer="reflect",
            z_bottom="reflect",
            z_top="reflect",
        ),
        av_C1=0.0,
        av_C2=1.5,
        # The loader requires >0; 1e-30 makes e-i transfer negligible over 2e-7 s.
        qei_multiplier=1.0e-30,
    ),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Output(
    directory="./build/output_verify_braginskii_2d_planar_2t",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=False)
