from tenryu_namelist import *
import math

# W-K gate B: planar standing wave; with the coupling ON and a uniform
# seeded E_r0, c_eff^2 = (gamma_g p_g + (4/9) E_r0)/rho. Gate measures the
# modal frequency ratio ON/OFF (fresh-load sampling).

Main(
    name="rad_gamma_ceff",
    dimension="1D_SPH",
    t_end=4.0e-5,
    max_steps=2000000,
    seed=12345,
    verbosity="quiet",
)

Mesh(r_min=0.0, r_max=1.0, nr=96, grid="uniform", geometry_1d="planar")

Materials(
    materials=[
        Material(
            name="gas",
            A=1.0,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=1.6666666666666667)),
            # kappa_a strictly 0 is a never-productioned singular limit (the
            # FLD matter solve killed a seeded field to ~1e-110 within ~35
            # steps in that mode); 1e-20 cm^2/g gives sigma*c ~ 3e-10 /s —
            # absorption over the whole gate window ~7e-16, six decades
            # below the law tolerance, while avoiding any 1/sigma pathology.
            opacity=dict(model="constant", kappa_a=1.0e-20, kappa_s=0.0, units="cm2_per_g"),
        )
    ]
)

Geometry(
    volfrac=dict(gas=lambda r: 1.0),
    rho=lambda r: 1.0,
    Te=lambda r: 6.2637e-3,
    Ti=lambda r: 6.2637e-3,
    velocity=lambda r: 100.0 * math.sin(math.pi * r),
    radiation_field="zero",
)

Numerics(
    dt=dict(initial_s=1.0e-9, cfl_hydro=0.3, growth_factor=1.2, max_s=1.0e-8, min_s=1.0e-20),
    hydro=dict(boundary_1d="reflect", av_C1=0.0, av_C2=1.5),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Output(
    directory="./build/output_verify_rad_gamma_ceff",
    plot_every=0, history_every=0, checkpoint_every=0,
    plot_every_s=-1.0, history_every_s=-1.0, checkpoint_every_s=-1.0,
)

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=1,
    group_bounds_eV=[0.1, 10000.0],
    compute_T_range_eV=[0.01, 10000.0],
    multigroup_diffusion=dict(
        hydro_coupling="none",  # explicit OFF baseline: the verify runner turns the coupling ON per leg via env
        flux_limiter="levermore_pomraning",
        max_outer_iterations=20,
        outer_tol=1.0e-5,
        # The FLD 1D solver honors THIS boundary struct
        # (multigroup_diffusion.boundary), not Radiation.boundary.
        boundary=dict(inner_r="reflect", outer_r="reflect"),
    ),
)
Laser(enabled=False)
Diagnostics(enabled=True)
