from tenryu_namelist import *

ns = 1.0e-9


def marshak_Tr_drive_eV(t_s):
    if t_s < 1.0 * ns:
        return 120.0
    if t_s < 2.0 * ns:
        return 120.0 + 80.0 * ((t_s - 1.0 * ns) / (1.0 * ns))
    return 200.0


# Main ブロック
# 何を設定するか: 1D 球対称の Tr 駆動ケース名、終了時刻、乱数 seed。
# 単位: t_end は s。
# 初学者が変えて良い knob と典型値域: t_end=1e-9 から 5e-9 s、name は任意。
# 触らないほうがよい物: dimension="1D_SPH" は球対称間接照射の前提なので変更しない。
Main(
    name="template_1d_indirect_tr",
    dimension="1D_SPH",
    t_end=3.0 * ns,
    max_steps=500000,
    seed=12345,
    verbosity="quiet",
)

# Mesh ブロック
# 何を設定するか: 燃料とアブレータを含む 1D 球メッシュ。
# 単位: r_min/r_max は cm、nr はセル数。
# 初学者が変えて良い knob と典型値域: nr=100 から 600、r_max=0.02 から 0.06 cm。
# 触らないほうがよい物: r_min=0 は球中心を表すため変更しない。
Mesh(
    r_min=0.0,
    r_max=0.033,
    nr=200,
    grid="uniform",
)

# Materials ブロック
# 何を設定するか: fuel と ablator の ideal gas EOS と constant opacity。
# 単位: A/Z は無次元、kappa_a/kappa_s は cm2/g。
# 初学者が変えて良い knob と典型値域: ablator kappa_a=50 から 500 cm2/g、fuel rho は Geometry で調整。
# 触らないほうがよい物: model は外部テーブルなしで validate できるよう固定する。
Materials(
    materials=[
        Material(
            name="fuel",
            A=2.5,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=1.0, kappa_s=0.0, units="cm2_per_g"),
        ),
        Material(
            name="ablator",
            A=6.5,
            Z=3.5,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=200.0, kappa_s=0.0, units="cm2_per_g"),
        ),
    ]
)

# Geometry ブロック
# 何を設定するか: fuel/ablator の配置、初期密度、温度、速度。
# 単位: rho は g/cm3、Te/Ti は eV、velocity は cm/s、半径しきい値は cm。
# 初学者が変えて良い knob と典型値域: fuel 半径=0.01 から 0.03 cm、ablator rho=0.5 から 1.2 g/cm3。
# 触らないほうがよい物: radiation_field="zero" は外部 Tr 駆動の効果を分けるため固定する。
Geometry(
    volfrac=dict(
        fuel=lambda r: 1.0 if r < 0.030 else 0.0,
        ablator=lambda r: 0.0 if r < 0.030 else 1.0,
    ),
    rho=lambda r: 0.01 if r < 0.030 else 1.05,
    Te=lambda r: 1.0e-3,
    Ti=lambda r: 1.0e-3,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

# Numerics ブロック
# 何を設定するか: 時間刻み、hydro、conduction、floor。
# 単位: dt は s、rho_floor_gcc は g/cm3、Te/Ti floor は eV。
# 初学者が変えて良い knob と典型値域: dt.max_s=5e-13 から 5e-12 s。
# 触らないほうがよい物: floor は低密度燃料の数値安定性のため極端に下げない。
Numerics(
    dt=dict(initial_s=1.0e-15, max_s=2.0e-12, min_s=1.0e-22, growth_factor=1.2),
    hydro=dict(boundary_1d="free", av_C1=0.5, av_C2=1.5),
    conduction=dict(enabled=True),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

# Radiation ブロック
# 何を設定するか: 外側 Marshak 境界に Python callable の Tr(t) を与える。
# 単位: marshak_Tr_drive_eV の戻り値は eV、t_s は s、group_bounds_eV は eV。
# 初学者が変えて良い knob と典型値域: Tr plateau=100 から 300 eV、ramp 時刻=0.5 から 3 ns。
# 触らないほうがよい物: callable は初期化時に凍結テーブル化されるため、実行中に外部状態へ依存させない。
Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=1,
    group_bounds_eV=[0.1, 1.0e5],
    compute_T_range_eV=[0.1, 1.0e5],
    multigroup_diffusion=dict(
        boundary=dict(inner_r="reflect", outer_r="marshak"),
    ),
    boundary=dict(
        marshak_Tr=marshak_Tr_drive_eV,
    ),
)

# Laser ブロック
# 何を設定するか: レーザーを使わないことを明示。
# 単位: enabled は bool。
# 初学者が変えて良い knob と典型値域: このテンプレートでは変更しない。
# 触らないほうがよい物: enabled=False は Tr 駆動だけを見るため固定する。
Laser(enabled=False)

# Burn ブロック
# 何を設定するか: 核燃焼を使わないことを明示。
# 単位: enabled は bool。
# 初学者が変えて良い knob と典型値域: このテンプレートでは変更しない。
# 触らないほうがよい物: enabled=False は間接照射の基礎だけを見るため固定する。
Burn(enabled=False)

# Output ブロック
# 何を設定するか: 出力先とスナップショット間隔。
# 単位: plot_every_s/history_every_s/checkpoint_every_s は s。
# 初学者が変えて良い knob と典型値域: plot_every_s=5e-11 から 2e-10 s。
# 触らないほうがよい物: checkpoint_every=0 は入門例を軽く保つため固定する。
Output(
    directory="outputs/template_1d_indirect_tr",
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
# 触らないほうがよい物: enabled=True は Tr 駆動応答の確認に使うため推奨する。
Diagnostics(enabled=True)
