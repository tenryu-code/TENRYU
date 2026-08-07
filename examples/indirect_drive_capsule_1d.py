# Indirect-drive capsule example: a hohlraum-equivalent radiation temperature waveform Tr(t) drives a CH-like
# shell around a low-density fuel-like fill; the ablation front launches the compression. Input = Tr(t) via
# Radiation.boundary.marshak_Tr (frozen to a table at init); see
# docs/design/indirect_drive_tr_boundary_20260709.md.
from tenryu_namelist import *

Main(
    name="indirect_drive_capsule_1d",
    dimension="1D_SPH",
    t_end=6.0e-9,
    max_steps=500000,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=0.033,
    nr=200,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="fuel",
            A=2.5,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=1.0, kappa_s=0.0, units="cm2_per_g"),
        ),
        Material(
            name="ablator",
            A=6.5,
            Z=3.5,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=200.0, kappa_s=0.0, units="cm2_per_g"),
        ),
    ]
)

Geometry(
    volfrac=dict(
        fuel=lambda r: 1.0 if r < 0.030 else 0.0,
        ablator=lambda r: 0.0 if r < 0.030 else 1.0,
    ),
    rho=lambda r: 0.01 if r < 0.030 else 1.05,
    Te=lambda r: 1.0e-3,
    Ti=lambda r: 1.0e-3,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

Numerics(
    dt=dict(initial_s=1.0e-15, max_s=2.0e-12, min_s=1.0e-22, growth_factor=1.2),
    hydro=dict(boundary_1d="free", av_C1=0.5, av_C2=1.5),
    conduction=dict(enabled=True),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Radiation(
    enabled=True,
    groups=8,
    group_bounds_eV=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0],
    multigroup_diffusion=dict(boundary=dict(inner_r="reflect", outer_r="marshak")),
    boundary=dict(
        inner_r="reflect",
        outer_r="vacuum",
        marshak_Tr=lambda t: 150.0 * min(t / 5.0e-10, 1.0),
    ),
)

Laser(enabled=False)

Output(
    directory="./build/indirect_drive_capsule_1d",
    plot_every=0,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=2.0e-10,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
