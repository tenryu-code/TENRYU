# Zero-dimensional radiation-matter relaxation deck.
#
# This closed, reflecting planar box certifies the FLD Fleck-factor rate against
# the exact coupled radiation-matter ODE.  Large-A legs isolate the
# electron-radiation system; A=1 legs are used by the coalescence gate where
# legacy and table modes must share identical physics.
# Design: docs/design/fleck_cv_default_flip_20260711.md section 5 W2.
import os

from tenryu_namelist import *


MODE = os.environ.get("TENRYU_FRLX_MODE", "table")
INIT = os.environ.get("TENRYU_FRLX_INIT", "hotmat")
KAPPA = float(os.environ.get("TENRYU_FRLX_KAPPA", "1667.0"))
DT = float(os.environ.get("TENRYU_FRLX_DT", "1.0e-13"))
A_ION = float(os.environ.get("TENRYU_FRLX_A", "1.0e5"))
TEND = float(os.environ.get("TENRYU_FRLX_TEND", "2.0e-12"))
CKPT = int(os.environ.get("TENRYU_FRLX_CKPT", "0"))
EOS = os.environ.get("TENRYU_FRLX_EOS", "power")
BETA = os.environ.get("TENRYU_FRLX_BETA", "tangent")
OUTER = int(os.environ.get("TENRYU_FRLX_OUTER", "60"))
FORM = os.environ.get("TENRYU_FRLX_FORM", "be")
SRC = os.environ.get("TENRYU_FRLX_SRC", "fleck")
GROUPS = int(os.environ.get("TENRYU_FRLX_GROUPS", "1"))
OUTDIR = os.environ.get(
    "TENRYU_FRLX_OUTDIR", "outputs/fleck_relaxation_0d"
)

if INIT == "hotmat":
    T_INIT = 200.0
    RADIATION_FIELD = "zero"
    RADIATION_FIELD_TR_EV = None
elif INIT == "hotrad":
    T_INIT = 100.0
    RADIATION_FIELD = "planck"
    RADIATION_FIELD_TR_EV = 200.0
elif INIT == "eqhold":
    T_INIT = 150.0
    RADIATION_FIELD = "equilibrium"
    RADIATION_FIELD_TR_EV = None
elif INIT == "linpert":
    T_INIT = 100.0
    RADIATION_FIELD = "planck"
    RADIATION_FIELD_TR_EV = 105.0
elif INIT == "bumphot":
    # beta_sec G-S2 regime: radiation-dominated heating (beta_tan ~ 48,
    # z = beta c sigma dt ~ 0.9-3.6 across the ladder) so the Fleck factor
    # is load-bearing and the beta choice has leverage.
    T_INIT = 1000.0
    RADIATION_FIELD = "planck"
    RADIATION_FIELD_TR_EV = 1500.0
else:
    raise ValueError(
        "TENRYU_FRLX_INIT must be 'hotmat', 'hotrad', 'eqhold', 'linpert',"
        " or 'bumphot'"
    )

if EOS == "power":
    EOS_STEP = {}
elif EOS == "softstep":
    # beta_sec coefficient-isolation feature (external verdict, design doc
    # section 7): strong c_v softstep (~x30 at the plateau; w ~ 3.9 table
    # intervals at 1240 eV) so the tangent-vs-chord mismatch is order unity
    # across a single coarse step. Constants are frozen here and
    # cross-checked by tools/fleck_relaxation_reference.py (BUMP_STEP_*).
    EOS_STEP = dict(
        step_D_erg_g_eV=2.0e12,
        step_Tc_eV=1240.0,
        step_w_eV=100.0,
    )
else:
    raise ValueError("TENRYU_FRLX_EOS must be 'power' or 'softstep'")

if BETA not in ("tangent", "secant", "guard"):
    raise ValueError("TENRYU_FRLX_BETA must be 'tangent', 'secant', or 'guard'")

if FORM not in ("be", "exp_phi1"):
    raise ValueError("TENRYU_FRLX_FORM must be 'be' or 'exp_phi1'")

if SRC not in ("fleck", "exp_rosenbrock"):
    raise ValueError("TENRYU_FRLX_SRC must be 'fleck' or 'exp_rosenbrock'")

if GROUPS not in (1, 4):
    raise ValueError("TENRYU_FRLX_GROUPS must be 1 or 4 (4 uses the fixed test bounds)")


Main(
    name="fleck_relaxation_0d",
    dimension="1D_SPH",
    t_end=TEND,
    seed=314159,
    max_steps=500000,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=0.4,
    nr=4,
    grid="uniform",
    geometry_1d="planar",
)

Materials(
    materials=[
        Material(
            name="frlx",
            A=A_ION,
            Z=1.0,
            eos=dict(
                model="power_law_te",
                f_erg_g=5.7e8,
                beta=1.6,
                mu_rho=0.0,
                **EOS_STEP,
            ),
            opacity=dict(model="constant", kappa_a=KAPPA, kappa_s=0.0),
        )
    ],
    zbar=dict(model="fixed", fixed_value=1.0),
)

geometry_kwargs = dict(
    volfrac=dict(frlx=lambda r: 1.0),
    rho=lambda r: 0.2,
    Te=lambda r: T_INIT,
    Ti=lambda r: T_INIT,
    velocity=lambda r: 0.0,
    radiation_field=RADIATION_FIELD,
)
if RADIATION_FIELD_TR_EV is not None:
    geometry_kwargs["radiation_field_Tr_eV"] = RADIATION_FIELD_TR_EV
Geometry(**geometry_kwargs)

Numerics(
    dt=dict(
        initial_s=DT,
        max_s=DT,
        min_s=1.0e-22,
        growth_factor=1.0,
    ),
    hydro=dict(enabled=False, boundary_1d="reflect", av_C1=0.1, av_C2=1.5),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=0.1, Ti_floor_eV=0.1),
)

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=GROUPS,
    group_bounds_eV=[0.1, 1.0e5] if GROUPS == 1 else [0.1, 300.0, 1000.0, 3000.0, 1.0e5],
    compute_T_range_eV=[0.1, 1.0e5],
    multigroup_diffusion=dict(
        opacity_floor=0.0,
        opacity_cap=1.0e8,
        fleck_cv_source=MODE,
        fleck_beta=BETA,
        fleck_form=FORM,
        source_integrator=SRC,
        # verification runs the scheme at tight outer tolerance — the closed-box closure gap equals the outer-iteration truncation residual (measured: closure tracks outer_tol; 1e-5 -> 5.4e-6, 1e-9 -> 3.4e-10)
        outer_tol=1.0e-9,
        max_outer_iterations=OUTER,
        boundary=dict(inner_r="reflect", outer_r="reflect"),
    ),
)

Laser(enabled=False)

Output(
    directory=OUTDIR,
    plot_every=0,
    history_every=1,
    checkpoint_every=(1 if CKPT else 0),
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
