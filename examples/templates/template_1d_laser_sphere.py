from tenryu_namelist import *

um = 1.0e-4
ns = 1.0e-9

R_SPHERE = 500.0 * um
PULSE_DURATION = 2.0 * ns
P_LASER_W = 1.0e12


def power_square(t_s):
    return P_LASER_W if 0.0 <= t_s <= PULSE_DURATION else 0.0


# Main ブロック
# 何を設定するか: 1D 球対称、2T、終了時刻、乱数 seed、最大ステップ数。
# 単位: t_end は s、temperature_model="2T" は電子/イオン温度を分ける指定。
# 初学者が変えて良い knob と典型値域: t_end=5e-10 から 5e-9 s、name は任意。
# 触らないほうがよい物: dimension="1D_SPH" は radial_absorption_1d の前提なので変更しない。
Main(
    name="template_1d_laser_sphere",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=2.0 * ns,
    seed=12345,
    max_steps=500000,
    verbosity="quiet",
)

# Mesh ブロック
# 何を設定するか: 半径 500 um の 1D 球メッシュ。
# 単位: r_min/r_max は cm、nr はセル数、um=1e-4 cm。
# 初学者が変えて良い knob と典型値域: nr=100 から 800、R_SPHERE=100 から 1000 um。
# 触らないほうがよい物: r_min=0 は球中心を表すため変更しない。
Mesh(
    r_min=0.0,
    r_max=R_SPHERE,
    nr=300,
    grid="uniform",
)

# Materials ブロック
# 何を設定するか: CD 風の ideal gas と constant opacity。
# 単位: A/Z は無次元、kappa_a/kappa_s は cm2/g。
# 初学者が変えて良い knob と典型値域: kappa_a=1 から 200 cm2/g、A/Z は材料に合わせる。
# 触らないほうがよい物: model は外部テーブルなしで validate できるよう固定する。
Materials(
    materials=[
        Material(
            name="CD",
            A=7.0,
            Z=3.5,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=100.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    opacity_mix_rule="linear_mass",
    zbar=dict(model="fixed", fixed_value=3.5),
)

# Geometry ブロック
# 何を設定するか: 一様な初期密度、温度、速度、材料体積率。
# 単位: rho は g/cm3、Te/Ti は eV、velocity は cm/s。
# 初学者が変えて良い knob と典型値域: rho=0.01 から 1.2 g/cm3、Te/Ti=0.1 から 10 eV。
# 触らないほうがよい物: enforce_sum_to_one=True は材料体積率の整合性確認のため残す。
Geometry(
    volfrac=dict(CD=lambda r: 1.0),
    rho=lambda r: 1.05,
    Te=lambda r: 1.0,
    Ti=lambda r: 1.0,
    velocity=lambda r: 0.0,
    radiation_field="equilibrium",
    enforce_sum_to_one=True,
)

# Radiation ブロック
# 何を設定するか: grey FLD を真空境界で有効化。
# 単位: group_bounds_eV/compute_T_range_eV は eV。
# 初学者が変えて良い knob と典型値域: group_bounds_eV の上限=1e4 から 1e5 eV。
# 触らないほうがよい物: mode/groups は灰色 FLD 入門例として固定する。
Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=1,
    group_bounds_eV=[0.1, 1.0e5],
    compute_T_range_eV=[0.1, 1.0e5],
    multigroup_diffusion=dict(
        hydro_coupling="none",
        boundary=dict(inner_r="reflect", outer_r="vacuum"),
    ),
)

# Laser ブロック
# 何を設定するか: radial_absorption_1d の 351 nm、1 TW square 波形、単一ビーム。
# 単位: wavelength_nm は nm、power callable は W、profile w0_um は um。
# 初学者が変えて良い knob と典型値域: P_LASER_W=1e11 から 1e13 W、PULSE_DURATION=0.5 から 5 ns。
# 触らないほうがよい物: mode="radial_absorption_1d" は 1D 球の直接照射用なので変更しない。
Laser(
    enabled=True,
    wavelength_nm=351.0,
    mode="radial_absorption_1d",
    beams=[
        LaserBeam(
            name="beam_00",
            direction=(0.0, 0.0, -1.0),
            power=power_square,
            profile=dict(model="super_gaussian", w0_um=500.0, m=4),
        )
    ],
    cbet=dict(enable=False),
    hot_electron=dict(enable=False),
)

# Numerics ブロック
# 何を設定するか: 時間刻み、hydro、Spitzer conduction、floor。
# 単位: dt は s、floor は g/cm3 と eV、f_lim は無次元。
# 初学者が変えて良い knob と典型値域: dt.max_s=1e-14 から 2e-13 s、f_lim=0.03 から 0.1。
# 触らないほうがよい物: conduction solver は標準 Spitzer 経路を使うため既定値を保つ。
Numerics(
    dt=dict(
        initial_s=5.0e-14,
        max_s=5.0e-14,
        min_s=1.0e-22,
        growth_factor=1.0,
        cfl_hydro=0.3,
        cfl_cond=0.25,
        f_min_fleck=0.01,
    ),
    hydro=dict(boundary_1d="free"),
    conduction=dict(enabled=True, f_lim=0.06),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=0.1, Ti_floor_eV=0.1),
)

# Burn ブロック
# 何を設定するか: 核燃焼を使わないことを明示。
# 単位: enabled は bool。
# 初学者が変えて良い knob と典型値域: このテンプレートでは変更しない。
# 触らないほうがよい物: enabled=False はレーザー直接照射の基礎だけを見るため固定する。
Burn(enabled=False)

# Output ブロック
# 何を設定するか: 出力先とスナップショット間隔。
# 単位: plot_every_s/history_every_s/checkpoint_every_s は s。
# 初学者が変えて良い knob と典型値域: plot_every_s=1e-11 から 1e-10 s。
# 触らないほうがよい物: checkpoint_every=0 は入門例を軽く保つため固定する。
Output(
    directory="outputs/template_1d_laser_sphere",
    plot_every=0,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=5.0e-11,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

# Diagnostics ブロック
# 何を設定するか: 標準 diagnostics の有効化。
# 単位: enabled は bool。
# 初学者が変えて良い knob と典型値域: enabled=True/False。
# 触らないほうがよい物: enabled=True はレーザー吸収確認に使うため推奨する。
Diagnostics(enabled=True)
