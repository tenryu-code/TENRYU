from tenryu_namelist import *

import os


T0_KEV = float(os.environ.get("TENRYU_BURN_T0_KEV", "3.0"))
RHO = float(os.environ.get("TENRYU_BURN_RHO", "10.0"))
R_CM = float(os.environ.get("TENRYU_BURN_R_CM", "0.02"))
T_END_S = float(os.environ.get("TENRYU_BURN_TEND_S", "0.5e-9"))
PARTITION = os.environ.get("TENRYU_BURN_PARTITION", "fraley")
BURN_SCHEME = os.environ.get("TENRYU_BURN_SCHEME", "fraley")
BURN_GROUPS = int(os.environ.get("TENRYU_BURN_GROUPS", "30"))
BURN_MC_SEED = os.environ.get("TENRYU_BURN_MC_SEED")
OUTDIR = os.environ.get("TENRYU_BURN_OUTDIR", "./out_burn_micro")


mat_dt = Material(
    name="DT",
    A=2.5,
    Z=1.0,
    eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
    opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
)

main_kwargs = dict(
    name="burn_microsphere_1d",
    dimension="1D_SPH",
    t_end=T_END_S,
)
if BURN_MC_SEED is not None:
    main_kwargs["seed"] = int(BURN_MC_SEED)
Main(**main_kwargs)

Mesh(
    r_min=0.0,
    r_max=R_CM,
    nr=200,
    grid="uniform",
    geometry_1d="spherical",
)

Materials(
    materials=[mat_dt],
    zbar=dict(model="fixed", fixed_value=1.0),
)

Geometry(
    volfrac=dict(DT=lambda r: 1.0),
    rho=lambda r: RHO,
    Te=lambda r: T0_KEV * 1.0e3,
    Ti=lambda r: T0_KEV * 1.0e3,
)

Radiation(enabled=False)

kwargs = dict(
    enabled=True,
    fuels=["DT"],
    scheme=BURN_SCHEME,
    partition=PARTITION,
    fuel_materials=["DT"],
    x_D=0.5,
    x_T=0.5,
    x_He3=0.0,
)
if BURN_SCHEME == 'diffusion':
    kwargs['diffusion_groups'] = BURN_GROUPS
Burn(**kwargs)

Numerics(
    dt=dict(initial_s=1.0e-15),
    hydro=dict(boundary_1d="free", av_C1=0.5, av_C2=1.5),
    conduction=dict(enabled=False),
)

Output(
    directory=OUTDIR,
    format="hdf5",
    plot_every=0,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=1.0e-11,
    checkpoint_every_s=-1.0,
)
