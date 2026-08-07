# micro-probe: does material INDEX 1 lose its transport opacity in FLD?
# planar box: mat0 = thin gas (20 cells), mat1 = THICK wall (30 cells) at the
# OUTER marshak-driven side. If kappa_R(mat1) is honored, E_r stays in the wall;
# if lost, the whole box floods within ~ps.
from tenryu_namelist import *

Main(name="probe_mat1_opacity", dimension="1D_SPH", temperature_model="2T",
     t_end=1.0e-10, seed=1, verbosity="quiet")
Mesh(r_min=0.0, r_max=0.1, nr=50, grid="uniform", geometry_1d="planar")
Materials(
    materials=[
        Material(name="gas", A=1.0, Z=1.0,
                 eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0/3.0)),
                 opacity=dict(model="constant", kappa_a=1.0e-3, kappa_s=0.0,
                              units="cm2_per_g")),
        Material(name="wall", A=9.0, Z=4.0,
                 eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0/3.0)),
                 opacity=dict(model="constant", kappa_a=1.0e3, kappa_s=0.0,
                              units="cm2_per_g")),
    ],
    zbar=dict(model="fixed"),
)
Geometry(
    volfrac=dict(gas=lambda r: 1.0 if r < 0.04 else 0.0,
                 wall=lambda r: 0.0 if r < 0.04 else 1.0),
    rho=lambda r: 1.0e-4 if r < 0.04 else 1.0,
    Te=lambda r: 1.0e-3, Ti=lambda r: 1.0e-3, velocity=lambda r: 0.0,
    radiation_field="zero",
)
Numerics(
    dt=dict(initial_s=1.0e-15, max_s=5.0e-13, growth_factor=1.2, min_s=1.0e-21),
    hydro=dict(enabled=False, boundary_1d="reflect"),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)
Radiation(
    enabled=True, mode="multigroup_diffusion", groups=1,
    group_bounds_eV=[0.1, 1.0e5], compute_T_range_eV=[0.1, 1.0e5],
    multigroup_diffusion=dict(opacity_floor=0.0, opacity_cap=1.0e8,
                              fleck_cv_source="table",
                              boundary=dict(inner_r="reflect", outer_r="marshak")),
    boundary=dict(marshak_Tr=lambda t: 200.0),
)
Laser(enabled=False, cbet=dict(enable=False), hot_electron=dict(enable=False))
Burn(enabled=False)
Output(directory="tmp/xc/probe_mat1", plot_every=0, history_every=0,
       checkpoint_every=0, plot_every_s=2.0e-11, history_every_s=-1.0,
       checkpoint_every_s=-1.0)
Diagnostics(enabled=True)
