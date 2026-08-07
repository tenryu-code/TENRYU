from tenryu_namelist import *

ns = 1.0e-9

# Main ブロック
# 何を設定するか: ケース名、次元、終了時刻、乱数 seed、最大ステップ数。
# 単位: t_end は s。dimension は 1D_SPH にし、Mesh.geometry_1d で planar を選ぶ。
# 初学者が変えて良い knob と典型値域: t_end=1e-10 から 1e-8 s、name は任意。
# 触らないほうがよい物: seed と units は再現性と単位系固定のため変更しない。
Main(
    name="template_1d_slab_radiation",
    dimension="1D_SPH",
    t_end=1.0 * ns,
    seed=314159,
    max_steps=500000,
    verbosity="quiet",
)

# Mesh ブロック
# 何を設定するか: 1D 平面スラブの計算範囲とセル数。
# 単位: r_min/r_max は cm、nr はセル数、geometry_1d="planar" は平面幾何。
# 初学者が変えて良い knob と典型値域: nr=100 から 800、r_max=0.01 から 0.1 cm。
# 触らないほうがよい物: grid="uniform" は入門例の解釈を簡単にするため固定する。
Mesh(
    r_min=0.0,
    r_max=0.06,
    nr=200,
    grid="uniform",
    geometry_1d="planar",
)

# Materials ブロック
# 何を設定するか: 材質の原子量・電離度と物性モデル。
# 単位: A/Z は無次元、kappa_a/kappa_s は cm2/g、cv_e_override は erg/(g*eV)。
# 初学者が変えて良い knob と典型値域: A=6.5/Z=3.5 は CH プラスチック相当の代表値、kappa_a=10 から 1000 cm2/g。
# 触らないほうがよい物: eos と opacity の model は外部テーブル不要の入門例にするため変更しない。
Materials(
    materials=[
        Material(
            name="ch_slab",
            A=6.5,
            Z=3.5,
            eos=dict(
                model="ideal_gas",
                ideal_gas=dict(gamma=5.0 / 3.0),
                cv_e_override=8.68e11,
            ),
            opacity=dict(model="constant", kappa_a=100.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=1.0),
)

# Geometry ブロック
# 何を設定するか: 初期密度、電子/イオン温度、速度、材料体積率。
# 単位: rho は g/cm3、Te/Ti は eV、velocity は cm/s。
# 初学者が変えて良い knob と典型値域: rho=0.01 から 1.0 g/cm3、Te/Ti=0.1 から 10 eV。
# 触らないほうがよい物: radiation_field="zero" は Marshak 波の初期条件を明確にするため固定する。
Geometry(
    volfrac=dict(ch_slab=lambda r: 1.0),
    rho=lambda r: 0.2,
    Te=lambda r: 1.0,
    Ti=lambda r: 1.0,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

# Numerics ブロック
# 何を設定するか: 時間刻み、hydro OFF、conduction OFF、温度/密度 floor。
# 単位: dt は s、rho_floor_gcc は g/cm3、Te/Ti floor は eV。
# 初学者が変えて良い knob と典型値域: dt.max_s=1e-13 から 5e-12 s。
# 触らないほうがよい物: hydro/conduction OFF は純粋な放射伝播入門にするため固定する。
Numerics(
    dt=dict(
        initial_s=1.0e-14,
        max_s=2.0e-12,
        min_s=1.0e-22,
        growth_factor=1.2,
    ),
    hydro=dict(enabled=False, boundary_1d="reflect", av_C1=0.1, av_C2=1.5),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0, Ti_floor_eV=1.0),
)

# Radiation ブロック
# 何を設定するか: grey FLD と外側 Marshak 境界の 120 eV 定数駆動。
# 単位: group_bounds_eV と marshak_Tr_eV は eV、境界名は文字列。
# 初学者が変えて良い knob と典型値域: marshak_Tr_eV=50 から 300 eV、outer_r は marshak/vacuum。
# 触らないほうがよい物: mode と groups は grey FLD 入門例として固定する。
Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=1,
    group_bounds_eV=[0.1, 1.0e5],
    compute_T_range_eV=[0.1, 1.0e5],
    multigroup_diffusion=dict(
        opacity_floor=0.0,
        opacity_cap=1.0e8,
        boundary=dict(inner_r="reflect", outer_r="marshak"),
    ),
    boundary=dict(
        marshak_Tr_eV=120.0,
    ),
)

# Laser ブロック
# 何を設定するか: レーザーを使わないことを明示。
# 単位: enabled は bool。
# 初学者が変えて良い knob と典型値域: このテンプレートでは変更しない。
# 触らないほうがよい物: enabled=False は放射境界だけを見るため固定する。
Laser(enabled=False)

# Burn ブロック
# 何を設定するか: 核燃焼を使わないことを明示。
# 単位: enabled は bool。
# 初学者が変えて良い knob と典型値域: このテンプレートでは変更しない。
# 触らないほうがよい物: enabled=False は放射伝播だけを分離するため固定する。
Burn(enabled=False)

# Output ブロック
# 何を設定するか: 出力先とスナップショット間隔。
# 単位: plot_every_s/history_every_s/checkpoint_every_s は s、plot_every はステップ数。
# 初学者が変えて良い knob と典型値域: plot_every_s=1e-11 から 1e-10 s。
# 触らないほうがよい物: checkpoint_every=0 は入門例を軽く保つため固定する。
Output(
    directory="outputs/template_1d_slab_radiation",
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
# 触らないほうがよい物: enabled=True は入門時の確認情報を残すため推奨する。
Diagnostics(enabled=True)
