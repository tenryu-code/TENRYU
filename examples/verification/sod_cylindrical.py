from tenryu_namelist import *

# W-G2 sod_cylindrical gate deck — file-parameterized at NAMELIST TIME only
# (the geometry lambdas are tabulated/frozen at initialization; no runtime
# Python). The verify gate writes `build/sod_cylindrical_variant.txt` per
# variant and reloads fresh (driver.run in-memory re-entry is unsupported).
import os

# Parameters arrive via a file (NOT env: the embedded interpreter's os.environ
# is a snapshot from process start; the gate's in-process setenv is invisible).
_geom, _r0, _nr = "cylindrical", 0.05, 200
_pf = "./build/sod_cylindrical_variant.txt"
if os.path.exists(_pf):
    with open(_pf) as _f:
        _tok = _f.read().split()
    _geom, _r0, _nr = _tok[0], float(_tok[1]), int(_tok[2])
_L = 1.0
_xj = _r0 + 0.5 * _L

Main(
    name="sod_cylindrical",
    dimension="1D_SPH",
    t_end=2.0e-7,
    max_steps=200000,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=_r0,
    r_max=_r0 + _L,
    nr=_nr,
    grid="uniform",
    geometry_1d=_geom,
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
    rho=lambda r: 1.0 if r < _xj else 0.125,
    Te=lambda r: 1.0 if r < _xj else 0.8,
    Ti=lambda r: 1.0 if r < _xj else 0.8,
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
    # Both walls reflect: static far-field states stay untouched within the
    # Sod window (W-G1 lesson: "free" is an external vacuum).
    hydro=dict(boundary_1d="reflect", av_C1=0.5, av_C2=1.5),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Output(
    directory="./build/output_verify_sod_cylindrical",
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
