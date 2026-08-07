# One-dimensional Marshak feature-crossing verification deck.
import os

from tenryu_namelist import *


DT = float(os.environ.get("TENRYU_MFEAT_DT", "1.0e-15"))
TEND = float(os.environ.get("TENRYU_MFEAT_TEND", "4.0e-12"))
NX = int(os.environ.get("TENRYU_MFEAT_NX", "512"))
SRC = os.environ.get("TENRYU_MFEAT_SRC", "fleck")
CLOSURE = os.environ.get("TENRYU_MFEAT_CLOSURE", "energy_authoritative")
OUTER = int(os.environ.get("TENRYU_MFEAT_OUTER", "60"))
PLOT_EVERY = float(os.environ.get("TENRYU_MFEAT_PLOT", "1.0e-13"))
OUTDIR = os.environ.get(
    "TENRYU_MFEAT_OUTDIR", "outputs/marshak_feature_1d"
)

if SRC not in ("fleck", "exp_rosenbrock"):
    raise ValueError("TENRYU_MFEAT_SRC must be 'fleck' or 'exp_rosenbrock'")
if CLOSURE not in ("legacy", "energy_authoritative"):
    raise ValueError("TENRYU_MFEAT_CLOSURE must be 'legacy' or 'energy_authoritative'")


Main(
    name="marshak_feature_1d",
    dimension="1D_SPH",
    t_end=TEND,
    seed=314159,
    max_steps=500000,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=0.04,
    nr=NX,
    grid="uniform",
    geometry_1d="planar",
)

Materials(
    materials=[
        Material(
            name="mfeat",
            A=1.0e5,
            Z=1.0,
            eos=dict(
                model="power_law_te",
                f_erg_g=5.7e8,
                beta=1.6,
                mu_rho=0.0,
                step_D_erg_g_eV=2.0e12,
                step_Tc_eV=1240.0,
                step_w_eV=100.0,
            ),
            opacity=dict(model="constant", kappa_a=1667.0, kappa_s=0.0),
        )
    ],
    zbar=dict(model="fixed", fixed_value=1.0),
)

Geometry(
    volfrac=dict(mfeat=lambda r: 1.0),
    rho=lambda r: 0.2,
    Te=lambda r: 1000.0,
    Ti=lambda r: 1000.0,
    velocity=lambda r: 0.0,
    radiation_field="equilibrium",
)

Numerics(
    dt=dict(
        initial_s=DT,
        max_s=DT,
        min_s=DT,
        growth_factor=1.0,
    ),
    # BUG-26 closure (2026-07-20): the 2026-07-18 "EA x exp front offset" freeze
    # traced to the 1T init ignoring the material EOS table (ideal-gas ee, healed
    # silently by the legacy closure at step 1, propagated by energy_authoritative).
    # With the init fixed the closures agree bit-tightly on this deck; the gate now
    # certifies under the production-default closure. TENRYU_MFEAT_CLOSURE remains
    # for A/B (e.g. "legacy").
    hydro=dict(enabled=False, eos_closure_mode=CLOSURE, boundary_1d="reflect",
               av_C1=0.1, av_C2=1.5),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=0.1, Ti_floor_eV=0.1),
)

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=1,
    group_bounds_eV=[0.1, 1.0e5],
    compute_T_range_eV=[0.1, 1.0e5],
    multigroup_diffusion=dict(
        opacity_floor=0.0,
        opacity_cap=1.0e8,
        fleck_cv_source="table",
        source_integrator=SRC,
        outer_tol=1.0e-9,
        max_outer_iterations=OUTER,
        boundary=dict(inner_r="reflect", outer_r="marshak"),
    ),
    boundary=dict(marshak_Tr_eV=1500.0),
)

Laser(enabled=False)

Output(
    directory=OUTDIR,
    plot_every=0,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=PLOT_EVERY,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
