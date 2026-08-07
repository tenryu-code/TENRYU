from tenryu_namelist import *
import math

A = 12.0
Z_BAR = 6.0
RHO0 = 1.0
TE0 = 100.0
TI0 = 10.0
GAMMA = 5.0 / 3.0

m_p = 1.6726219e-24  # must match core::constants::proton_mass
n_i = RHO0 / (A * m_p)
n_e = Z_BAR * n_i
if TE0 >= 10.0 * Z_BAR * Z_BAR:
    ln_lambda_raw = 24.0 - 0.5 * math.log(n_e) + math.log(TE0)
else:
    ln_lambda_raw = 23.0 - 0.5 * math.log(n_e) - math.log(Z_BAR) + 1.5 * math.log(TE0)
ln_lambda = max(2.0, ln_lambda_raw)
tau_eq = 3.16e8 * A * (TE0 ** 1.5) / (Z_BAR * Z_BAR * n_i * ln_lambda)

Main(
    name="ei_relaxation",
    dimension="1D_SPH",
    t_end=5.0 * tau_eq,
    max_steps=1_000_000,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=1.0,
    nr=100,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="carbon",
            A=A,
            Z=Z_BAR,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=Z_BAR),
)

Geometry(
    volfrac=dict(carbon=lambda r: 1.0),
    rho=lambda r: RHO0,
    Te=lambda r: TE0,
    Ti=lambda r: TI0,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=0.01 * tau_eq,
        max_s=0.01 * tau_eq,
        growth_factor=1.0,
        min_s=1.0e-20,
    ),
    hydro=dict(boundary_1d="reflect", av_C1=0.1, av_C2=1.5),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Radiation(enabled=False)
Laser(enabled=False)

Output(
    directory="./build/output_verify_ei_relaxation",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
