"""I4a 2D RZ layered Marshak FLD slab deck."""

import os

from tenryu_namelist import *


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


# Keep in sync with tools/validation/layered_marshak_reference_generator.py.
C_LIGHT = 2.99792458e10
A_EV = 1.3720e2
DEFAULT_F_INC = C_LIGHT * A_EV * 100.0**4 / 4.0

NR = _env_int("TENRYU_I4A_NR", 8)
NZ = _env_int("TENRYU_I4A_NZ", 256)
L_CM = _env_float("TENRYU_I4A_L_CM", 1.0)
KAPPA1 = _env_float("TENRYU_I4A_KAPPA1", 30.0)
KAPPA2 = _env_float("TENRYU_I4A_KAPPA2", 10.0)
T_END_S = _env_float("TENRYU_I4A_T_END_S", 3.0e-10)
DT_S = _env_float("TENRYU_I4A_DT_S", 3.33e-12)
DT_INITIAL = _env_float("TENRYU_I4A_DT_INITIAL_S", 1.0e-15)
F_INC = _env_float("TENRYU_I4A_F_INC", DEFAULT_F_INC)
SINGLE_MAT = os.environ.get("TENRYU_I4A_SINGLE_MAT", "0") == "1"
OUTDIR = os.environ.get(
    "TENRYU_I4A_OUTDIR",
    "./build/output_verify_i4a_layered_marshak",
)

PLOT_EVERY_S = T_END_S / 5.0
CASE_NAME = (
    "i4a_layered_marshak_2d_rz_slab"
    f"_nr{NR}_nz{NZ}"
    f"_k1{_safe_float_token(KAPPA1)}_k2{_safe_float_token(KAPPA2)}"
)

print(
    "[deck:i4a_layered_marshak_2d_rz_slab] "
    f"nr={NR} nz={NZ} outdir={OUTDIR} "
    f"kappa1={KAPPA1} kappa2={KAPPA2} "
    f"t_end_s={T_END_S} dt_initial_s={DT_INITIAL} dt_s={DT_S} f_inc={F_INC} L_cm={L_CM} "
    "hydro=False radiation=multigroup_diffusion conduction=False laser=False "
    f"dimension=2D_RZ single_mat={int(SINGLE_MAT)}"
)


def rho_init(r, z):
    del r, z
    return 1.0


def Te_init(r, z):
    del r, z
    return 1.0e-3


def Ti_init(r, z):
    del r, z
    return 1.0e-3


def velocity_init(r, z):
    del r, z
    return (0.0, 0.0)


def volfrac_layer1(r, z):
    del r
    return 1.0 if z < 0.5*L_CM else 0.0


def volfrac_layer2(r, z):
    return 1.0 - volfrac_layer1(r, z)


Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=T_END_S,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=0.125,
    z_min=0.0,
    z_max=L_CM,
    nr=NR,
    nz=NZ,
    grid="uniform",
    motion="lagrangian",
)

Materials(
    materials=[
        Material(
            name="layer1",
            A=1.0,
            Z=1.0,
            eos=dict(
                model="ideal_gas",
                cv_e_override=5.488e2,
                ideal_gas=dict(gamma=5.0 / 3.0),
            ),
            opacity=dict(
                model="constant",
                kappa_a=KAPPA1,
                kappa_s=0.0,
                units="cm2_per_g",
            ),
        ),
        *(
            []
            if SINGLE_MAT
            else [
                Material(
                    name="layer2",
                    A=1.0,
                    Z=1.0,
                    eos=dict(
                        model="ideal_gas",
                        cv_e_override=5.488e2,
                        ideal_gas=dict(gamma=5.0 / 3.0),
                    ),
                    opacity=dict(
                        model="constant",
                        kappa_a=KAPPA2,
                        kappa_s=0.0,
                        units="cm2_per_g",
                    ),
                ),
            ]
        ),
    ],
    zbar=dict(model="fixed", fixed_value=1.0),
)

Geometry(
    volfrac=(
        dict(layer1=lambda r, z: 1.0)
        if SINGLE_MAT
        else dict(layer1=volfrac_layer1, layer2=volfrac_layer2)
    ),
    rho=rho_init,
    Te=Te_init,
    Ti=Ti_init,
    velocity=velocity_init,
    radiation_field="zero",
    enforce_sum_to_one=True,
)

Numerics(
    dt=dict(
        initial_s=DT_INITIAL,
        max_s=DT_S,
        min_s=1.0e-22,
        growth_factor=1.1,
    ),
    hydro=dict(
        enabled=False,
        boundary_2d=dict(
            r_inner="axis",
            r_outer="reflect",
            z_bottom="reflect",
            z_top="reflect",
        ),
    ),
    conduction=dict(enabled=False),
    ale=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=1,
    group_bounds_eV=[0.0, 1.0e6],
    imc=dict(enabled=False, two_stage=False, difference=dict(enabled=False), **({"alpha": _env_float("TENRYU_I4A_IMC_ALPHA", -1.0)} if os.environ.get("TENRYU_I4A_IMC_ALPHA") else {})),
    ddmc=dict(enabled=False),
    holo=dict(enabled=False),
    multigroup_diffusion=dict(
        flux_limiter="none",
        linear_solver_2d=os.environ.get("TENRYU_I4A_LINEAR_SOLVER_2D", "auto"),
        outer_tol=1.0e-8,
        max_outer_iterations=60,
        boundary=dict(
            inner_r="reflect",
            outer_r="reflect",
            z_bottom="marshak",
            z_top="reflect",
        ),
        marshak=dict(flux_erg_per_cm2_s=F_INC),
    ),
)

Laser(enabled=False)

Output(
    directory=OUTDIR,
    format="hdf5",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=PLOT_EVERY_S,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    plot_fields=["rho", "Te", "Ti", "ee", "ei", "rad_E", "volfrac"],
)

Diagnostics(enabled=True, every=1, energy_budget=dict(enabled=True, warn_threshold=1.0e-3))
