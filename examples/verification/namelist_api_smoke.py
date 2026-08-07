from tenryu_namelist import *

Main(name="smoke", dimension="1D_SPH", t_end=1.0e-9, seed=42)
Mesh(r_min=0.0, r_max=0.05, nr=100)
Materials(
    materials=[
        Material(
            name="test",
            A=1.0,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=1.0, units="cm2_per_g"),
        ),
    ],
    opacity_mix_rule="linear_mass",
)
Geometry(
    volfrac=dict(test=lambda r: 1.0),
    rho=lambda r: 1.0,
    Te=lambda r: 1.0,
    Ti=lambda r: 1.0,
)
Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=dict(bounds_eV=[0.1, 1.0, 10.0]),
    imc=dict(enabled=False, alpha=1.0),
    ddmc=dict(enabled=False),
    holo=dict(enabled=False),
)
Laser(enabled=False)
Numerics(dt=dict(initial_s=1e-15))
Output(directory="./test_output")
Diagnostics(enabled=True)
Parallel()
Burn(enabled=False)

m = Material(
    name="x",
    A=1.0,
    Z=1.0,
    eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
    opacity=dict(model="constant", kappa_a=0.0, units="cm2_per_g"),
)
assert isinstance(m, dict), "Material() must return dict"

b = LaserBeam(
    name="b0",
    direction=(0.0, 0.0, -1.0),
    power=lambda t: 1.0,
    f_number=8.0,
)
assert isinstance(b, dict), "LaserBeam() must return dict"
