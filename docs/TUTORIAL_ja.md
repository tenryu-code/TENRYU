# TENRYU チュートリアル（初学者向け・日本語）

本書は **初めて TENRYU を使う人**のための入門書です。ビルド → 最初の計算 →
namelist（入力ファイル）の書き方 → 結果の見方 → よくあるエラー、の順に進みます。
網羅的なリファレンスは `docs/USER_MANUAL.md`（使い方全般）と
`docs/SPECIFICATION.md` §6.4（全キーの正典）にあります。本書は「最短で動かして
仕組みを理解する」ことに絞ります。

---

## 1. TENRYU とは（1 分）

ICF（慣性核融合）向けの **輻射流体コード**です。1D（球/円筒/平面）と 2D RZ で、

- Lagrangian/ALE **2 温度流体**（電子温度 Te とイオン温度 Ti を別々に解く）
- **決定論的多群輻射輸送**（FLD = 流束制限拡散が既定、S_N = 離散座標も選択可）
- **レーザー**（レイトレース + 逆制動輻射吸収。CBET・ホット電子は opt-in）
- 電子熱伝導（Spitzer–Härm + flux limiter。非局所 SNB は opt-in）
- 核燃焼（DT/DD、opt-in）

を NVIDIA GPU (CUDA) で解きます。**単位系は cgs + eV で固定**です
（長さ cm・時間 s・密度 g/cm³・温度 eV・エネルギー erg）。

---

## 2. ビルド（初回のみ）

```bash
cmake -S . -B build -GNinja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DTENRYU_ENABLE_MPI=ON \
  -DTENRYU_ENABLE_HDF5=ON \
  -DTENRYU_ENABLE_PYTHON=ON \
  -DTENRYU_ENABLE_NVTX=ON
ninja -C build
```

成功すると `build/tenryu` が実行ファイルです。詳細は `.claude/commands.md`。

---

## 3. 5 分で最初の計算

テンプレート（§6 参照）を使います。**run の前に必ず validate** する癖をつけて
ください — 設定ミスは実行前にほぼ全部見つかります。

```bash
# ① 入力ファイルの検査（実行しない。数秒で終わる）
./build/tenryu validate examples/templates/template_1d_slab_radiation.py

# ② 実行
./build/tenryu run examples/templates/template_1d_slab_radiation.py

# ③ 結果の場所（deck の Output(directory=...) で指定した場所）
ls outputs/template_1d_slab_radiation/results/
```

`validate` は設定の要約（pre-flight summary: 次元・格子・材料・物理モジュールの
ON/OFF・出力先など）を表示します。**要約を目で確認してから run** — これが最も
安上がりなデバッグです。

---

## 4. namelist（入力ファイル）の考え方

TENRYU の入力は **1 個の Python ファイル**です（Smilei 方式）。中身は
「設定ブロック＝関数呼び出し」の列です:

```python
from tenryu_namelist import *

Main(name="my_first_run", dimension="1D_SPH", t_end=1.0e-9)   # 全体設定
Mesh(nr=200, r_min=0.0, r_max=0.05)                            # 格子 [cm]
Materials(materials=[mat])                                     # 物性（EOS/opacity）
Geometry(volfrac=..., rho=..., Te=..., Ti=...)                 # 初期分布
Radiation(enabled=True, groups=1, ...)                         # 輻射
Laser(enabled=False)                                           # レーザー
Numerics(dt=dict(initial_s=1e-13, max_s=1e-12), ...)           # 数値制御
Output(directory="outputs/my_first_run", plot_every=5e-11)     # 出力
```

初学者が押さえるべき原則は 4 つだけです:

1. **単位は常に cgs + eV**。長さを µm で書きたいときは自分で換算します
   （例: `um = 1.0e-4` を定義して `r_max=500*um`）。
2. **空間分布は「関数」で与える**（callable プロファイル）。例えば密度なら
   `def rho(r_cm): return 1.0 if r_cm < 0.02 else 0.001` のように、半径 [cm] を
   受け取って値を返す Python 関数を渡します。**この関数は初期化時に一度だけ
   評価されて凍結され、実行中に Python は呼ばれません**（時間依存の波形
   — レーザーパワーや駆動温度 — も同様に「テーブル化して凍結」されます）。
3. **書かなかったキーは既定値**になります。既定値の一覧は SPECIFICATION §9.1。
   迷ったら書かない → validate の要約で実効値を確認、が安全です。
4. **キー名を打ち間違えるとエラーで止まります**（黙って無視はしません）。
   その際 `did you mean 'kappa_a'?` のような**修正候補**が表示されます。

---

## 5. ブロック別・最小限の書き方

ここでは「初学者がまず触るキー」だけを載せます。全キーは SPECIFICATION §6.4。

### Main — 何を・どの座標系で・いつまで
```python
Main(name="run名", dimension="1D_SPH", t_end=2.0e-9)  # t_end [s]
```
- `dimension`: `"1D_SPH"`（1 次元）か `"2D_RZ"`。1D の幾何（球/円筒/平面）は
  Mesh 側の `geometry_1d` で選びます（既定は球対称）。

### Mesh — 格子
```python
Mesh(nr=300, r_min=0.0, r_max=0.05)                    # [cm]
Mesh(nr=200, r_min=0.0, r_max=0.06, geometry_1d="planar")  # 平面 1D の例
```
- 解像度を上げる＝ nr を増やす。まず粗く動かし、後から倍々で上げて
  結果が変わらないこと（収束）を確認するのが定石です。

### Materials — 物性
```python
mat = Material(
    name="fuel", A=2.5, Z=1.0,                          # 質量数・原子番号
    eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0/3.0)),
    opacity=dict(model="constant", kappa_a=100.0, kappa_s=0.0),  # [cm²/g]
)
Materials(materials=[mat])
```
- 入門は `ideal_gas` + `constant` で十分。実験解析では SESAME/IONMIX の
  テーブルを使います（USER_MANUAL §4）。
- 多材料（シェル+燃料など）は materials のリストに並べ、Geometry の
  volfrac で空間配置します。

### Geometry — 初期分布（関数で与える）
```python
def rho(r_cm):   return 1.0                            # [g/cm³]
def T0(r_cm):    return 1.0                            # [eV]
Geometry(volfrac=dict(fuel=lambda r: 1.0), rho=rho, Te=T0, Ti=T0,
         radiation_field="zero")                       # 放射場の初期値
```

### Radiation — 輻射輸送
```python
Radiation(
    enabled=True,
    groups=1, group_bounds_eV=[0.1, 1.0e5],            # 1 群（灰色）
    boundary=dict(inner_r="reflect", outer_r="vacuum"),
)
```
- 輸送モードは書かなければ **FLD（既定）**。S_N を使うときだけ
  `mode="sn_transport"` を足します。
- 多群にするなら `groups=dict(bounds_eV=[...])` に境界列 [eV] を渡します。
- **間接照射（放射温度駆動）**: 境界を `outer_r="marshak"` にし、
  `Radiation.boundary.marshak_Tr` に時間の関数 [s]→[eV] を渡します
  （テンプレ③参照。初期化時に凍結テーブル化されます）。

### Laser — レーザー
```python
def pulse(t_s):  return 1.0e12 if t_s < 1.0e-9 else 0.0   # [W] = 1 TW（power は W 指定、内部で erg/s に変換）
Laser(enabled=True, wavelength_nm=351.0,
      mode="radial_absorption_1d",                     # 1D の最簡モード
      beams=[LaserBeam(name="beam_00",
                       direction=(0.0, 0.0, -1.0),
                       power=pulse)])                  # power は時間 [s] の関数
```
- 1D 入門は `radial_absorption_1d`（ビーム形状に依存しない径方向吸収）が
  最も簡単。レイトレースが要るときは `raytrace_2d`（USER_MANUAL §7）。
- CBET・ホット電子プリヒートは opt-in（`Laser.cbet` / `Laser.hot_electron`、
  SPECIFICATION §6.4 参照）。まずは OFF のままで。

### Numerics — 時間刻みと物理スイッチ
```python
Numerics(
    dt=dict(initial_s=1.0e-14, max_s=2.0e-12),
    hydro=dict(enabled=True),
    conduction=dict(enabled=True),                     # 電子熱伝導
)
```
- 放射だけのテスト問題では `hydro=dict(enabled=False)` にして流体を凍結
  できます（テンプレ①がこの形）。
- dt は自動制御です。`initial_s` は小さめ、`max_s` は問題の時間スケールの
  1/1000 程度から始めるのが安全です。

### Output — 出力
```python
Output(directory="outputs/my_run",
       plot_every_s=5.0e-11,        # スナップショット間隔 [秒]
       history_every_s=1.0e-12)     # 時系列（スカラー）間隔 [秒]
```
- `plot_every`（`_s` なし）は「N ステップごと」の**整数**版です。時間で
  指定したいときは必ず `_s` 付きを使ってください（単位事故の定番ポイント）。
- **注意**: 同じ directory へ再実行すると、上書きせず `_001` のような
  連番ディレクトリに書かれます。「結果が変わらない…」と思ったら
  まず出力先を確認してください（実際に踏みがちな罠です）。

---

## 6. テンプレート（ここから始める）

`examples/templates/` に、全ブロックへ日本語コメントを付けた雛形があります。
**コピーして名前を変えて改造**してください。

| テンプレ | 内容 | 学べること |
|---|---|---|
| `template_1d_slab_radiation.py` | 平面 1D・灰色 FLD・Marshak 境界駆動・流体 OFF | 放射伝播の最小構成、validate→run→可視化の流れ |
| `template_1d_laser_sphere.py` | 球 1D・レーザー直接照射・2T・伝導 ON | レーザー波形の書き方、2 温度、複合物理 |
| `template_1d_indirect_tr.py` | 球 1D・Tr(t) 間接照射駆動 | 時間依存 callable → 凍結テーブルの流儀 |

改造の定番:
- 駆動温度・レーザーパワーを変える（callable の返り値）
- nr を倍にして収束を見る
- `Radiation(mode="sn_transport")` にして FLD と比べる

---

## 7. 結果の見方

出力ディレクトリの構造:
```
outputs/<name>/
├── config/    ← 凍結された全設定 (frozen JSON)。「実際に効いた値」の記録
├── results/   ← スナップショット *.h5 と history (*_history.h5)
├── log/       ← 実行ログ
└── run_info.json
```

- スナップショット HDF5 の主なデータ: `hydro/Te`, `hydro/Ti`, `hydro/rho`,
  `radiation/energy_density`, `mesh/x_r` など（全一覧は docs/OUTPUT_SCHEMA.md）。
- 最短の可視化は付属の **tenryu_plot**（サブコマンド: `profile` / `history` /
  `spacetime` / `summary` / `compare` / `spectrum` など）:
  ```bash
  python3 -m tools.tenryu_plot --help                  # 一覧
  python3 -m tools.tenryu_plot profile outputs/<name>  # 空間プロファイル
  python3 -m tools.tenryu_plot history outputs/<name>  # 時系列
  ```
  各サブコマンドの引数は `... profile --help` で確認できます。
- Python で直接読むなら:
  ```python
  import h5py
  f = h5py.File("outputs/<name>/results/<name>_0010.h5")
  Te = f["hydro/Te"][:]; r = f["mesh/x_r"][:]
  ```

**「設定が効いたか怪しい」ときは `config/*_frozen.json` を見る** — 凍結された
実効値が全部入っています（デバッグの一次資料）。

---

## 8. よくあるエラーと対処

| 症状 | 原因と対処 |
|---|---|
| `ConfigError: ... is not a supported key (did you mean 'X'?)` | キー名の打ち間違い。候補が出るのでそのまま直す |
| `ConfigError: eos.file is required for model=sesame` | テーブル EOS はファイル必須。入門は `ideal_gas` に |
| `ConfigError: opacity.model="power_law" is grey-only` | 冪乗 opacity は 1 群限定。多群にするなら constant/table へ |
| `Radiation.mode="imc_ddmc"` でエラー | 旧 Monte Carlo は退役済み（1D では選択不可）。書かない（=FLD 既定）か `sn_transport` に |
| `imc=dict(...)` / `ddmc=dict(enabled=True)` でエラー | 退役モード専用の互換キー。現行 FLD/S_N では書かない |
| 実行したのに結果が古いまま | 出力 dir 衝突で `_001` へ書かれている（§5 Output 注意参照） |
| 温度が上がらない/波が進まない | まず validate の要約で enabled 群を確認。次に frozen JSON で境界・opacity の実効値を確認 |
| dt がどんどん小さくなり進まない | ログの dt limiter 名（`hydro`/`conduction`/`braginskii` 等）を確認し、該当物理の設定（床値・格子）を見直す |

---

## 9. 物理モジュール ON/OFF 早見表（全て namelist で制御）

| モジュール | 既定 | 有効化キー |
|---|---|---|
| 輻射 FLD（多群） | **ON**（mode 省略時） | `Radiation(enabled=True)` |
| 輻射 S_N | OFF | `Radiation(mode="sn_transport")` |
| 電子熱伝導 | ON | `Numerics.conduction.enabled` |
| 非局所伝導 (SNB) | OFF | `Numerics.conduction.nonlocal_model="snb"` |
| レーザー | OFF | `Laser(enabled=True, ...)` |
| CBET | OFF | `Laser.cbet.enable=True`（1D_SPH） |
| ホット電子プリヒート | OFF | `Laser.hot_electron.enable=True` |
| イオン物理粘性 (Braginskii) | OFF | `Numerics.hydro.plasma_viscosity.enabled=True`（1D） |
| 核燃焼 (DT/DD) | OFF | `Burn(enabled=True, fuels=..., scheme=...)` |

opt-in 群は「書かなければ完全に不活性（結果はビット単位で不変）」という契約で
実装されています。安心して 1 個ずつ足してください。

---

## 10. 次のステップ

- リファレンス: `docs/USER_MANUAL.md`（運用全般）→ `docs/SPECIFICATION.md` §6.4（全キー）
- 物理と離散化の正典: `docs/NUMERICS.md`
- 検証済みの例: `examples/verification/`（各 gate の実物 deck）
- うまくいかないとき: ①validate ②frozen JSON ③`log/` の warning、の順に見る
