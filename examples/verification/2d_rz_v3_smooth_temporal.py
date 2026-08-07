import math
import os

from tenryu_namelist import *


T_END = 5.0e-8
DT_MAX = float(os.environ.get("TENRYU_V3_DT_MAX", "1.0e-12"))
OUTDIR = os.environ.get(
    "TENRYU_V3_OUTDIR", "./build/output_verify_2d_rz_v3_smooth_temporal"
)
SWEPT_SIGN = os.environ.get("TENRYU_V3_SWEPT_SIGN", "")
CORNER_MASS = os.environ.get("TENRYU_V3_CORNER_MASS", "")
TIME_INTEGRATION = os.environ.get("TENRYU_V3_TIME_INTEGRATION", "")

if DT_MAX <= 0.0:
    raise ValueError("TENRYU_V3_DT_MAX must be positive")
if SWEPT_SIGN not in ("", "0", "1"):
    raise ValueError('TENRYU_V3_SWEPT_SIGN must be one of: "", "0", "1"')
if CORNER_MASS not in ("", "bbsw_radial_v0", "kinematic_basis_rz_v1"):
    raise ValueError(
        'TENRYU_V3_CORNER_MASS must be one of: "", "bbsw_radial_v0", '
        '"kinematic_basis_rz_v1"'
    )
if TIME_INTEGRATION not in ("", "pc_v0", "midpoint_v1"):
    raise ValueError(
        'TENRYU_V3_TIME_INTEGRATION must be one of: "", "pc_v0", "midpoint_v1"'
    )


def rho_init(r, z):
    del r, z
    return 1.0


def temperature_init(r, z):
    bump = math.exp(-(((r - 1.0) ** 2 + z**2) / (0.15**2)))
    return 100.0 * (1.0 + 0.05 * bump)


def velocity_init(r, z):
    del r, z
    return (0.0, 0.0)


def gas_volfrac(r, z):
    del r, z
    return 1.0


print(
    "[deck:2d_rz_v3_smooth_temporal] "
    f"outdir={OUTDIR} t_end_s={T_END} dt_initial_s={DT_MAX} dt_max_s={DT_MAX}"
)

Main(
    name="v3_smooth_temporal",
    dimension="2D_RZ",
    temperature_model="1T",
    t_end=T_END,
    max_steps=100000,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.5,
    r_max=1.5,
    z_min=-0.5,
    z_max=0.5,
    nr=32,
    nz=32,
    grid="uniform",
    motion="lagrangian",
)

Materials(
    materials=[
        Material(
            name="gas",
            A=1.0,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(
                model="constant",
                kappa_a=0.0,
                kappa_s=0.0,
                units="cm2_per_g",
            ),
        )
    ],
    zbar=dict(model="fixed", fixed_value=1.0),
)

Geometry(
    volfrac=dict(gas=gas_volfrac),
    rho=rho_init,
    Te=temperature_init,
    Ti=temperature_init,
    velocity=velocity_init,
    radiation_field="zero",
    enforce_sum_to_one=True,
)

Numerics(
    radiation_thermal_subcycle=False,
    dt=dict(
        initial_s=DT_MAX,
        cfl_hydro=0.3,
        growth_factor=1.0,
        max_s=DT_MAX,
        min_s=1.0e-22,
    ),
    hydro=dict(
        enabled=True,
        boundary_2d=dict(
            r_inner="reflect",
            r_outer="reflect",
            z_bottom="reflect",
            z_top="reflect",
        ),
        av_linear=0.0,
        av_quadratic=0.0,
        hourglass=dict(enabled=False),
        **(
            {"corner_mass_convention": CORNER_MASS}
            if CORNER_MASS in ("bbsw_radial_v0", "kinematic_basis_rz_v1")
            else {}
        ),
        **(
            {"time_integration": TIME_INTEGRATION}
            if TIME_INTEGRATION in ("pc_v0", "midpoint_v1")
            else {}
        ),
    ),
    conduction=dict(enabled=False),
    ale=dict(
        enabled=False,
        **({"swept_volume_sign_fixed": SWEPT_SIGN == "1"} if SWEPT_SIGN in ("0", "1") else {}),
    ),
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=0.1, Ti_floor_eV=0.1),
)

Radiation(enabled=False)
Laser(enabled=False)

Output(
    directory=OUTDIR,
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=T_END,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    plot_fields=["rho", "ee", "ei"],
)

Diagnostics(enabled=False)
