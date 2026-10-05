# TMAT-H5: TENRYU 物質テーブル HDF5 フォーマット — v1.0 仕様書

## 1. 概要

TMAT-H5 は、TENRYU 輻射流体コード用の状態方程式（EOS）および多群不透明度テーブルを格納するための、
自己記述型 HDF5 ベースフォーマットである。

| 項目 | 値 |
|---|---|
| フォーマット識別子 | `tenryu.material_table.hdf5` |
| ファイル拡張子 | `.tmat.h5` |
| スキーマバージョン | セマンティックバージョニング (`1.0.0`) |
| 単位系 | `cgs_eV`（固定） |
| 数値精度 | `float64` のみ（v1） |
| 文字列エンコーディング | 可変長 UTF-8 |

### 1.1 動機

IONMIX4 (.cn4) Fortran 非整形バイナリおよび SESAME ASCII フォーマットを以下で置換する:
- **自己記述型メタデータ** — 単位、元素組成、来歴情報
- **EOS/不透明度の独立グリッド** — グリッド共有の強制なし
- **バージョン管理されたスキーマ** — 前方互換性のある拡張性
- **標準 HDF5** — ポータブル、エンディアン問題なし、圧縮対応

### 1.2 既存フォーマットとの比較

| 特徴 | IONMIX4 | SESAME | TMAT-H5 |
|---|---|---|---|
| フォーマット | Fortran バイナリ | ASCII 80桁 | HDF5 |
| 自己記述型 | なし | なし | あり |
| 多群不透明度 | あり | なし（グレーのみ） | あり |
| EOS/不透明度の独立グリッド | なし（一致必須） | 該当なし | あり |
| 密度軸 | n_i [cm^-3] | rho [g/cm^3] | n_i [cm^-3] |
| 単位 | SI系（J/g, J/cm^3） | 混在 | cgs+eV（ネイティブ） |
| EOS単独 / 不透明度単独 | 不可 | 部分的 | 可 |
| バージョニング | なし | なし | セマンティックバージョニング |
| 来歴情報 | なし | なし | 完全 |

## 2. HDF5 グループ階層

```
/
  @format_id              = "tenryu.material_table.hdf5"
  @schema_version         = "1.0.0"
  @units_system           = "cgs_eV"
  @required_features      （任意、CSVトークン）
  @default_interpolation  （任意、推奨値 "bilinear_loglog_clamp"）

  /material               （必須）
    name                  string スカラー
    Z                     int32  [n_species]
    A_amu                 float64 [n_species]
    mass_fraction         float64 [n_species]
    Abar_ion_amu          float64 スカラー
    number_fraction       float64 [n_species]  （任意）
    species_name          string  [n_species]  （任意）

  /eos                    （EOSを含む場合は必須）
    @axis_order           = "D,T"
    @primary_density_axis = "ni_cm3"
    /grid
      ni_cm3              float64 [nD_eos]
      temperature_eV      float64 [nT_eos]
    /fields
      zbar                float64 [nD_eos, nT_eos]
      P_i                 float64 [nD_eos, nT_eos]
      P_e                 float64 [nD_eos, nT_eos]
      e_i                 float64 [nD_eos, nT_eos]
      e_e                 float64 [nD_eos, nT_eos]
      cv_i                float64 [nD_eos, nT_eos]  （任意）
      cv_e                float64 [nD_eos, nT_eos]  （任意）

  /opacity                （不透明度を含む場合は必須）
    @axis_order           = "G,D,T"
    @primary_density_axis = "ni_cm3"
    @is_lte               int32 スカラー (0 または 1)
    /grid
      ni_cm3              float64 [nD_op]
      temperature_eV      float64 [nT_op]
      group_bounds_eV     float64 [nG+1]
    /fields
      kappa_R             float64 [nG, nD_op, nT_op]
      kappa_PA            float64 [nG, nD_op, nT_op]
      kappa_PE            float64 [nG, nD_op, nT_op]

  /provenance             （推奨）
    source_format         string スカラー
    source_files          string [n_src]
    source_sha256         string [n_src]  （任意）
    generator_name        string スカラー
    generator_version     string スカラー
    generator_git_commit  string スカラー  （任意）
    command_line          string スカラー  （任意）
    notes                 string スカラー  （任意）

  /extensions             （任意）
    /<feature_name>/v<major>/...
```

## 3. メモリレイアウト

全ての多次元配列は **C 行優先** 順序（HDF5 デフォルト）を使用する。
最後のインデックスが最速で変化する。

| テーブル | 形状 | インデックス式 | 最速次元 |
|---|---|---|---|
| EOS フィールド | `[nD, nT]` | `idx = d * nT + t` | 温度 |
| 不透明度フィールド | `[nG, nD, nT]` | `idx = g * nD * nT + d * nT + t` | 温度 |

これは TENRYU の補間アクセスパターン（温度ブラケットを先に検索）と一致する。

**注意**: TENRYU 内部の `EOSTable` は `[T, rho]` レイアウト（`idx = t * nRho + rho`）を使用する。
TMAT リーダーは `EOSTable` への変換時に `[D,T] -> [T,D]` 転置を行う。
不透明度データは `[G,D,T]` をそのまま使用する（転置不要）。

## 4. 単位規約

全ての値は TENRYU のネイティブ `cgs + eV` 単位で格納される。リーダーによる
ランタイム単位変換は行わない。

| 物理量 | 単位 | データセット接尾辞 |
|---|---|---|
| 温度 | eV | `temperature_eV` |
| イオン数密度 | cm^-3 | `ni_cm3` |
| 圧力 (P_i, P_e) | dyne/cm^2 | |
| 比エネルギー (e_i, e_e) | erg/g | |
| 比熱 (cv_i, cv_e) | erg/(g*eV) | |
| 質量不透明度 (kappa_*) | cm^2/g | |
| エネルギー群境界 | eV | `group_bounds_eV` |

各数値データセットには、可読性のために `units` 属性を付与すべきである（SHOULD）。
リーダーはランタイム変換に `units` 属性を使用してはならない（MUST NOT）
（スキーマバージョンがセマンティクスを定義する）。

## 5. 厳密 v1 バリデーションチェックリスト

### 5.1 規範レベル

- **MUST** — 必須。違反時はロード失敗。
- **SHOULD** — 推奨。存在する場合の不正は致命的、不在は許容。
- **MAY** — 任意の拡張。
- **MUST*** — 含有グループが存在する場合のみ必須。

### 5.2 ルート属性

| パス | レベル | 型 | 制約 | 失敗条件 |
|---|---|---|---|---|
| `/@format_id` | MUST | UTF-8 文字列 | 正確に `"tenryu.material_table.hdf5"` | 欠落または不一致 |
| `/@schema_version` | MUST | UTF-8 文字列 | semver, major=1 | 欠落、不正形式、非対応メジャーバージョン |
| `/@units_system` | MUST | UTF-8 文字列 | 正確に `"cgs_eV"` | 欠落または不一致 |
| `/@required_features` | MAY | UTF-8 文字列 | CSVトークン `<name>_vN` | 不明トークン |
| `/@default_interpolation` | SHOULD | UTF-8 文字列 | 推奨値 | 失敗なし |

### 5.3 物質グループ

| パス | レベル | 型 | 形状 | 制約 | 失敗条件 |
|---|---|---|---|---|---|
| `/material` | MUST | グループ | — | 存在 | 欠落 |
| `/material/name` | SHOULD | UTF-8 文字列 | スカラー | 非空 | 失敗なし |
| `/material/Z` | MUST | int32 | `[n_species]` | n_species >= 1, 各値 >= 1 | 欠落/型/ランク/値 |
| `/material/A_amu` | MUST | float64 | `[n_species]` | 有限, > 0 | 欠落/型/形状/値 |
| `/material/mass_fraction` | MUST | float64 | `[n_species]` | 有限, >= 0, |sum-1| <= 1e-8 | 欠落/型/形状/値 |
| `/material/Abar_ion_amu` | MUST | float64 | スカラー | 有限, > 0 | 欠落/型/値 |
| `/material/number_fraction` | SHOULD | float64 | `[n_species]` | 有限, >= 0, sum ~ 1 | 存在するが不正の場合失敗 |
| `/material/species_name` | MAY | UTF-8 文字列 | `[n_species]` | — | 存在する場合形状不一致で失敗 |

### 5.4 EOS グループ（条件付き）

| パス | レベル | 型 | 形状 | 制約 | 失敗条件 |
|---|---|---|---|---|---|
| `/eos` | MUST* | グループ | — | — | — |
| `/eos/@axis_order` | MUST* | UTF-8 文字列 | スカラー | 正確に `"D,T"` | 欠落/不一致 |
| `/eos/@primary_density_axis` | MUST* | UTF-8 文字列 | スカラー | 正確に `"ni_cm3"` | 欠落/不一致 |
| `/eos/grid/ni_cm3` | MUST* | float64 | `[nD_eos]` | 有限, 狭義単調増加, > 0 | 型/ランク/値 |
| `/eos/grid/temperature_eV` | MUST* | float64 | `[nT_eos]` | 有限, 狭義単調増加, > 0 | 型/ランク/値 |
| `/eos/fields/zbar` | MUST* | float64 | `[nD_eos, nT_eos]` | 有限, >= 0 | 型/形状/値 |
| `/eos/fields/P_i` | MUST* | float64 | `[nD_eos, nT_eos]` | 有限 | 型/形状/非有限 |
| `/eos/fields/P_e` | MUST* | float64 | `[nD_eos, nT_eos]` | 有限 | 型/形状/非有限 |
| `/eos/fields/e_i` | MUST* | float64 | `[nD_eos, nT_eos]` | 有限 | 型/形状/非有限 |
| `/eos/fields/e_e` | MUST* | float64 | `[nD_eos, nT_eos]` | 有限 | 型/形状/非有限 |
| `/eos/fields/cv_i` | SHOULD* | float64 | `[nD_eos, nT_eos]` | 有限, > 0 | 存在するが不正の場合失敗 |
| `/eos/fields/cv_e` | SHOULD* | float64 | `[nD_eos, nT_eos]` | 有限, > 0 | 存在するが不正の場合失敗 |

### 5.5 不透明度グループ（条件付き）

| パス | レベル | 型 | 形状 | 制約 | 失敗条件 |
|---|---|---|---|---|---|
| `/opacity` | MUST* | グループ | — | — | — |
| `/opacity/@axis_order` | MUST* | UTF-8 文字列 | スカラー | 正確に `"G,D,T"` | 欠落/不一致 |
| `/opacity/@primary_density_axis` | MUST* | UTF-8 文字列 | スカラー | 正確に `"ni_cm3"` | 欠落/不一致 |
| `/opacity/@is_lte` | MUST* | int32 | スカラー | 0 または 1 | 欠落または不正 |
| `/opacity/grid/ni_cm3` | MUST* | float64 | `[nD_op]` | 有限, 狭義単調増加, > 0 | 型/ランク/値 |
| `/opacity/grid/temperature_eV` | MUST* | float64 | `[nT_op]` | 有限, 狭義単調増加, > 0 | 型/ランク/値 |
| `/opacity/grid/group_bounds_eV` | MUST* | float64 | `[nG+1]` | nG >= 1, 有限, 狭義単調増加, 先頭 >= 0 | 型/ランク/値 |
| `/opacity/fields/kappa_R` | MUST* | float64 | `[nG, nD_op, nT_op]` | 有限, >= 0 | 型/形状/値 |
| `/opacity/fields/kappa_PA` | MUST* | float64 | `[nG, nD_op, nT_op]` | 有限, >= 0 | 型/形状/値 |
| `/opacity/fields/kappa_PE` | MUST* | float64 | `[nG, nD_op, nT_op]` | 有限, >= 0 | 型/形状/値 |

### 5.6 来歴グループ

| パス | レベル | 型 | 制約 | 失敗条件 |
|---|---|---|---|---|
| `/provenance` | SHOULD | グループ | — | 失敗なし |
| `/provenance/source_format` | SHOULD | UTF-8 文字列 | 例: `IONMIX4`, `SESAME`, `PROPACEOS` | 失敗なし |
| `/provenance/source_files` | SHOULD | UTF-8 文字列配列 | 存在する場合 n_src >= 1 | 不正形式のみ |
| `/provenance/source_sha256` | MAY | UTF-8 文字列配列 | source_files と長さ一致 | 不一致 |
| `/provenance/generator_name` | SHOULD | UTF-8 文字列 | — | 失敗なし |
| `/provenance/generator_version` | SHOULD | UTF-8 文字列 | — | 失敗なし |
| `/provenance/generator_git_commit` | MAY | UTF-8 文字列 | — | 失敗なし |
| `/provenance/command_line` | MAY | UTF-8 文字列 | — | 失敗なし |
| `/provenance/notes` | MAY | UTF-8 文字列 | — | 失敗なし |

### 5.7 拡張グループ

- `/extensions` は MAY（任意）。
- `/@required_features` にトークン `foo_v1` が含まれる場合、`/extensions/foo/v1` が存在しなければならない（MUST）。
- 不明な必須機能トークンはロード失敗をトリガーする（フェイルファスト）。

### 5.8 フィールド間交差バリデーション（MUST）

1. `/eos` または `/opacity` の少なくとも一方が存在しなければならない（MUST）。
2. `n_species` は `Z`、`A_amu`、`mass_fraction` 間で整合しなければならない。
3. EOS フィールドの形状は `[nD_eos, nT_eos]` に正確に一致しなければならない。
4. 不透明度フィールドの形状は `[nG, nD_op, nT_op]` に正確に一致しなければならない。
5. グレー不透明度 (`nG=1`) は形状 `[1, nD_op, nT_op]` で有効。
6. 全てのコア数値データセットは `float64` でなければならない。

## 6. ロードエラーコード

| コード | 説明 |
|---|---|
| `TMAT_E001` | 必須パスが欠落 |
| `TMAT_E002` | 非対応のスキーマメジャーバージョン |
| `TMAT_E003` | 単位系の不一致 |
| `TMAT_E004` | 不明な必須機能トークン |
| `TMAT_E005` | HDF5 型の不一致 |
| `TMAT_E006` | ランクまたは形状の不一致 |
| `TMAT_E007` | 非有限値を検出 |
| `TMAT_E008` | 単調性違反（グリッドが狭義単調増加でない） |
| `TMAT_E009` | 不正な数値範囲（負の不透明度、非正の軸値） |
| `TMAT_E010` | 質量分率の制約違反（sum != 1） |
| `TMAT_E011` | 文字列エンコーディング失敗（不正な UTF-8） |
| `TMAT_E012` | 非対応の HDF5 フィルター/圧縮 |
| `TMAT_E013` | コアペイロードなし（/eos と /opacity の両方が不在） |

## 7. スキーマバージョニング

セマンティックバージョニング (`MAJOR.MINOR.PATCH`):

| 変更種別 | バージョンバンプ | リーダーの動作 |
|---|---|---|
| メタデータのみの明確化 | PATCH | 受容 |
| グループ/データセットの追加 | MINOR | `required_features` が既知であれば受容 |
| 破壊的な名前変更/型/セマンティクス変更 | MAJOR | major != 1 の場合は拒否 |

リーダールール: 不明なメジャーバージョンは拒否、全ての `required_features` が対応済みであれば新しいマイナーバージョンは受容。

## 8. ペイロード存在ルール

| ファイル種別 | `/eos` | `/opacity` | ユースケース |
|---|---|---|---|
| 完全 | あり | あり | EOS + 不透明度の統合テーブル |
| EOS のみ | あり | なし | EOS は TMAT から、不透明度は他ソースから |
| 不透明度のみ | なし | あり | 不透明度は TMAT から、EOS は他ソースから |

少なくとも一方が存在しなければならない（MUST）。両方とも不在の場合は `TMAT_E013` をトリガーする。

## 9. `is_lte` ポリシー

- **ライター**: `/opacity/@is_lte` を明示的に設定しなければならない（MUST）（ソースデータ解析に基づく）。
- **リーダー（厳密モード）**: 属性が存在しなければならない（MUST）。
- **リーダー（互換モード）**: 不在の場合、以下で自動検出し、1回警告を出す:
  `max_rel(|kappa_PA - kappa_PE| / max(kappa_PA, kappa_PE, 1e-30)) <= 1e-6`

## 10. 圧縮ポリシー

- コンバーターデフォルト: gzip deflate レベル6、チャンクデータセット。
- リーダー: 圧縮・非圧縮の両方のファイルを透過的に受け入れなければならない（MUST）
  （標準 HDF5 の動作 — リーダー側のロジック不要）。

## 11. TENRYU との統合

### 11.1 ネームリスト設定

新しいモデル名（明示的、自動検出なし）:
```python
Material(
    eos=dict(model="tmat", file="material.tmat.h5"),
    opacity=dict(model="tmat", file="material.tmat.h5"),
)
```

既存モデル（`sesame`、`ionmix`、`table_nlte` 等）は変更なし。
混合使用も可能: `eos.model="sesame"` + `opacity.model="tmat"`。

### 11.2 C++ リーダー API

```cpp
namespace materials {

struct TmatEOSData {
    int ndens = 0, ntemp = 0;
    std::vector<double> rho_grid;     // [nD] cm^-3 (n_i)
    std::vector<double> T_grid_eV;    // [nT] eV
    std::vector<double> zbar;         // [nD * nT] idx = d*nT+t
    std::vector<double> P_i;          // [nD * nT] dyne/cm^2
    std::vector<double> P_e;          // [nD * nT] dyne/cm^2
    std::vector<double> e_i;          // [nD * nT] erg/g
    std::vector<double> e_e;          // [nD * nT] erg/g
    std::optional<std::vector<double>> cv_i;  // [nD * nT] erg/(g*eV)
    std::optional<std::vector<double>> cv_e;  // [nD * nT] erg/(g*eV)
};

struct TmatOpacityData {
    int ngroups = 0, ndens = 0, ntemp = 0;
    std::vector<double> rho_grid;     // [nD] cm^-3 (n_i)
    std::vector<double> T_grid_eV;    // [nT] eV
    std::vector<double> bounds_eV;    // [nG+1] eV
    std::vector<double> kappa_R;      // [nG * nD * nT] cm^2/g
    std::vector<double> kappa_PA;     // [nG * nD * nT] cm^2/g
    std::vector<double> kappa_PE;     // [nG * nD * nT] cm^2/g
    bool is_lte = false;
};

struct TmatMaterialInfo {
    std::string name;
    std::vector<int> Z;
    std::vector<double> A_amu;
    std::vector<double> mass_fraction;
    double Abar_ion_amu = 0.0;
};

struct TmatFile {
    std::string schema_version;
    std::string units_system;
    std::vector<std::string> required_features;
    TmatMaterialInfo material;
    std::optional<TmatEOSData> eos;
    std::optional<TmatOpacityData> opacity;
};

enum class TmatLoadMode { Strict, Compatible };

// TMAT-H5 ファイルをロード
TmatFile load_tmat(const std::string& filepath,
                   TmatLoadMode mode = TmatLoadMode::Compatible);

// TENRYU 内部 EOSTable に変換（D,T -> T,D 転置）
EOSTablePair tmat_eos_to_table_pair(const TmatEOSData& eos);

} // namespace materials
```

### 11.3 データフロー

```
[.tmat.h5 ファイル]
     |
     v
load_tmat()  ──> TmatFile { eos, opacity, material }
     |                |
     |                v
     |         tmat_eos_to_table_pair()  ──> EOSTablePair (D,T->T,D 転置)
     |
     v
TmatOpacityData を compute_nlte_coefficients() で直接使用
（IonmixOpacityData と同じフラット [G,D,T] レイアウト）
```

## 12. コンバータースイート

全てのコンバーターは `h5py` を使用する Python スクリプトである。

### 12.1 `ionmix_to_tmat.py`
- 入力: IONMIX4/6 `.cn4` バイナリ
- 出力: `.tmat.h5`
- 密度軸: n_i [cm⁻³] をそのまま保存（変換不要）
- 単位変換: J/g -> erg/g (×1e7), J/cm^3 -> dyne/cm^2 (×1e7)
- ラウンドトリップ検証: ソースと読み戻しの max_rel <= 1e-12
- `is_lte` を kappa_PA vs kappa_PE の比較に基づいて設定

### 12.2 `propaceos_to_tmat.py`
- 入力: PROPACEOS `.prp` ASCII
- 出力: `.tmat.h5`
- `Propaceos_to_IONMIX4.py` の後継
- EOS/不透明度の独立グリッドを許容（グリッド共有の強制なし）
- 単位変換: dyne/cm^2 -> cgs（変換不要）, erg/g -> cgs（変換不要）

### 12.3 `sesame_to_tmat.py`
- 入力: xSESAME ASCII
- 出力: `.tmat.h5`
- グレー不透明度は `nG=1`、形状 `[1, nD, nT]`
- `kappa_PE = kappa_PA`（LTE）、`is_lte = 1`
- CLI オプションで群境界を指定（デフォルト: 全スペクトル単一群）

### 12.4 `validate_tmat.py`
- 完全 v1 チェックリスト（第5章）を実装するスタンドアロンバリデーター
- 全コンバーターが最終ゲートとして使用
- 全違反をエラーコード（第6章）付きで報告
- 終了コード 0 = 有効、非ゼロ = 違反あり
