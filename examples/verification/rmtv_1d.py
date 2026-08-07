from tenryu_namelist import *

import os

_t_end = 2.0e-9
_pf = "./build/rmtv_1d_t_end.txt"
if os.path.exists(_pf):
    with open(_pf) as _f:
        _tok = _f.read().split()
    if _tok:
        _t_end = float(_tok[0])

_nr = 400
_pf_nr = "./build/rmtv_1d_nr.txt"
if os.path.exists(_pf_nr):
    with open(_pf_nr) as _f:
        _tok = _f.read().split()
    if _tok:
        _nr = int(_tok[0])

Main(
    name="rmtv_1d",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=_t_end,
    max_steps=500000,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=0.25,
    nr=_nr,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="gas",
            A=1.0,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=1.25)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ]
)

Geometry(
    volfrac=dict(gas=lambda r: 1.0),
    rho=lambda r: 1.0e-3 * max(r, 1.0e-4) ** (-19.0 / 9.0),
    Te=lambda r: 1.0e-3,
    Ti=lambda r: 1.0e-3,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=1.0e-16,
        cfl_hydro=0.3,
        growth_factor=1.2,
        max_s=1.0e-11,
        min_s=1.0e-22,
    ),
    hydro=dict(boundary_1d="reflect", av_C1=0.1, av_C2=1.5),
    conduction=dict(
        enabled=True,
        solver="sts",
        f_lim=0.06,
        sts_damping=0.01,
        sts_max_stages=40,
        test_kappa=1.402400e7,
    ),
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-4, Ti_floor_eV=1.0e-4),
)

Output(
    directory="./build/output_verify_rmtv_1d",
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
