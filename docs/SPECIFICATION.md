# TENRYU — SPECIFICATION.md
**TENRYU (Total Energetic Numerical Radiation-hYdrodynamic Unit)**  
ICF向け 輻射流体（rad‑hydro）コード：多群 **決定論的輻射輸送（FLD 拡散 + S_N 離散座標）** を軸に、Lagrangian/ALE 2T 流体・レーザー（レイトレース + IB + CBET + ホット電子プリヒート）・電子熱伝導を **NVIDIA GPU（CUDAのみ）** で解く。

> **輻射輸送の現行ステータス（2026-07-10 truth-restoration）**
> 本番・検証対象は `Radiation.mode="multigroup_diffusion"`（FLD、**既定**）と `"sn_transport"`（決定論 S_N）。
> 初期設計の **IMC–PGRW–DDMC** Monte Carlo ハイブリッド（`mode="imc_ddmc"`）は**退役**した：
> コードはツリー内に残存し 2D_RZ でのみ選択可能だが、検証 gate は退役済みで**本番使用不可**（1D_SPH では `ConfigError`）。
> 本書の一部の節は歴史的経緯として IMC 前提の記述を含む — 該当箇所には退役注記を付す（§3、§5.2–5.3、§6.4.5 参照）。

## 0. 本書の位置付け
- 本書は **TENRYUの全体仕様（機能・入出力・物理モデル・制約・既定値）** を定義する。
- 数式と離散化の“唯一の真実”は `NUMERICS.md`。
- モジュール境界・依存方向は `ARCHITECTURE.md`。
- 検証（収束/MMS/参照解/許容誤差）は `VERIFICATION.md`。
- 性能（KPI/測定/プロファイル）は `PERFORMANCE.md`。
- LLM（Claude Code等）運用ルールは `CLAUDE.md`。
- 本仕様で未確定事項を残さない。**“選択肢”は仕様としては残さず、既定を固定**し、ユーザは namelist で上書きできる形にする。

---

## 1. 目的（Mission）
### 1.1 科学目的
高出力レーザーによる **球殻カプセル（薄シェル + 液体燃料）** の直接照射爆縮を輻射流体シミュレーションで再現し、以下を定量評価できること：
1. 到達密度（ρ_peak、ρR、圧縮率、中心温度）
2. 最適レーザー波形（pulse shaping：立上り・フット・メインの最適化）
3. 流体的歪み（RZではLegendreモード Pℓ、将来3DでYℓm）と不安定性傾向（殻厚変調、密度勾配、界面擾乱の増幅）
4. 照射パターン（ビーム配置・スポット・パワーバランス）が圧縮対称性に与える影響
5. 物理モデル切替（FLD↔S_N 輻射輸送、熱伝導フラックス制限、EOS/opacityモデル、CBET/ホット電子 ON/OFF）に対する感度

### 1.2 工学目的
- **決定論的 FLD/S_N 輻射輸送**（implicit 線形解法・加速つき、Fleck 線形化物質結合）により光学的厚領域を含む全域で **GPUでのスループットを確保**する。（初期設計の IMC–PGRW–DDMC による“局所イベント処理”方針は退役 — 冒頭ステータス参照。）
- 並列は **MPI + NVIDIA GPU（CUDA）** のみを公式サポートとする（1 rank / 1 GPU を基本）。
- “最適化・大量スイープ”を外部Pythonから回せるよう、入力と出力を機械可読にし、再現性を担保する。

---

## 2. スコープ / 非スコープ
### 2.1 スコープ（v1.0必須：現状は全て存在する前提）
- 次元：
  - **1D球対称（1D_SPH）**
  - **2D RZ軸対称（2D_RZ）**
  - 3Dは将来追加（本仕様では非対応）
- 流体：
  - 単一流体（one‑fluid）+ **2温度（T_i, T_e）**
  - 人工粘性（von Neumann–Richtmyer）
  - 1DはLagrangian、2DはALE（rezoning + remap含む）
- 伝導：
  - 電子熱伝導（Spitzer‑Härm + flux limiter）
  - イオン伝導はオプション（既定OFF）
- 輻射：
  - 多群（multigroup）放射輸送：**FLD（flux-limited diffusion、既定）+ S_N（離散座標、決定論）**（§5.3）
  - 退役互換経路として IMC–PGRW–DDMC ハイブリッド（`mode="imc_ddmc"`、2D_RZ のみ・本番使用不可 — 冒頭ステータス参照）
- レーザー：
  - 幾何光学（Geometric Optics）レイトレース + 逆制動輻射吸収（IB）
  - **外部レイトレース連携は行わない**（TENRYU内部のみ）
  - レイトレースの“思想”（LaserMesh分離、保守的沈着、性能）だけ **xRAGE** を参考にする
- 物性：
  - **SESAME を既定 EOS**。xSESAME ASCII 形式をサポート
  - **IONMIX を代替オプション**（EOS + opacity）
  - SESAME EOS + IONMIX opacity の混合構成を推奨（SESAME opacity（502/505）は grey のみ）
  - **理想気体EOSモード**（検証/MMS/簡易ラン用）を必須で持つ
- GPU：
  - **NVIDIA CUDA のみ**（HIP/他GPUバックエンドの公式サポート無し）
- Build-time diagnostics controls:
  - `TENRYU_RFA_V2_MODE: Literal["OFF","STUB","DUMMY_BUFFER","FULL"]`（CMake option；既定 `"FULL"`）。`OFF` は radial Fourier audit v2 implementation body をプリプロセッサで除外する。`STUB` は v2 API を残すが host-visible capture と GPU kernel launch を実行しない。`DUMMY_BUFFER` は v2 fixed-mode CUDA kernel を実行し一時 GPU buffer にのみ書く（HDF5 v2 schema へは append しない）。`FULL` は PR G2-A の通常動作で、`per_operator_radial_fourier_complex_enabled=True` 時に `/diagnostics/radial_fourier_audit_v2/v1/` を出力する。

### 2.2 非スコープ（現段階では実装しない）
- LPI の波動レベル第一原理計算（SRS/SBS/TPD の分散関係・成長率解計算）：非スコープ。ただし **CBET（Marozas 型 pairwise ray 交換、1D_SPH opt-in・既定OFF、NUMERICS §5.10）と ホット電子プリヒート（処方源 η_hot + 多群 CSDA 輸送、1D、NUMERICS §5.11）は実装済み**。TPD/SRS 由来の指向性ホット電子源モデル等は将来拡張。
- 磁場・MHD、非理想MHD。
- 核燃焼（DT燃焼・α自己加熱）：将来拡張（v1.0はOFF）。
- Comoving frameの **O(v/c)** 輸送項（ドップラー/放射圧仕事など）：v1.0は無視（明示；詳細はNUMERICS）。
- **Voidセル**：`is_void: True` フラグ付きの材料として定義される真空領域をサポートする。Void材料は理想気体EOS（低密度）を使用し、opacity=0、radiation/laser/conduction couplingをスキップする。§6.4.3 Materials 参照。

### 2.3 輻射モデルの適用範囲（v1.0）
v1.0の輻射モデルは以下の前提に基づく。これらの制限はユーザが意識すべき適用範囲を定める。

- **LTE（局所熱平衡）前提**：放射源は黒体放射 \(aT_e^4\) と Planck分率 \(b_g(T_e)\) に基づく。非LTE（準位population、非平衡イオン化）、散乱支配領域、線スペクトル支配系には適用不可。
- **物理散乱なし（v1.0既定）**：\(\sigma_{s,phys} = 0\)。散乱はFleck factor由来の実効散乱のみ。Thomson/Compton散乱が重要な高温低Zプラズマでは透過・スペクトル誤差が生じうる。
- **O(v/c)項の無視**：ドップラーシフト、放射圧仕事、Compton heating/cooling を含まない（§5.5参照）。放射力が流体力学に匹敵する条件（radiation-dominated flow）では精度が不足する。
- **放射エネルギーの電子結合**：`rad_dep` は電子内部エネルギー \(e_e\) にのみ加算される（ARCHITECTURE §4.5、NUMERICS §10.2）。検証問題（Su-Olson等）での「物質温度」は \(T_e\) を指す。

---

## 3. 主要要件（High‑level Requirements）
### 3.1 物理・数値の健全性
- 質量保存（閉系/適切境界で）：相対誤差 ≤ 1e‑10（機械誤差レベル目標）
- エネルギー保存（流体 + 輻射 + レーザー入射）：
  - 「投入エネルギー」＝「系内エネルギー増分 + 境界流出 + 数値散逸（人工粘性）」
- 決定論輸送（FLD/S_N）では保存台帳を **gate ごとの機械精度級許容**（VERIFICATION 参照）で検査する。時間平均保存誤差の定義は全ステップの相対保存誤差 \(\varepsilon_n = |E_{in}^n - E_{out}^n - \Delta E_{int}^n| / E_{in}^n\) の**算術平均** \(\bar{\varepsilon} = \frac{1}{N}\sum_{n=1}^{N}\varepsilon_n\)（§6.4.9 `energy_budget` の `conservation_error` に対応）。（退役 imc_ddmc モードでは MC 統計誤差込みで \(\bar{\varepsilon}\le 0.1\%\) を基準としていた — 歴史的要件。）
- 参照コード（DRACO相当設定）との主要スカラー量比較で整合：
  - ρ_peak, ρR, shock timing, 吸収エネルギー、放射損失

### 3.2 性能
- 実測 KPI・プロファイル・ホスト側オーバーヘッドの現況は `PERFORMANCE.md` を正とする（決定論輸送化後の主律速はホスト側 CUDA driver 経路 — W-F 系列で追跡）
- （歴史的目標：厚領域支配問題で純IMCに対し ≥5× のスループット改善 — 退役 imc_ddmc 時代の KPI）
- 弱スケーリング：GPU数を増やし 1GPU当たり計算量一定で **80%効率**（目標）
- 強スケーリング：典型2D問題で 8→64 GPU で **≥6×**（目標）
- 通信オーバーヘッド：MPI通信（ハロー交換＋粒子移動＋Reduction）が
  総wall-clock時間に占める割合 **≤15%**（目標；NUMERICS §12参照）

### 3.3 再現性（Reproducibility）
- **決定論輸送（FLD/S_N）+ 1D Lagrangian 経路**：同一GPU・同一構成で **run-to-run bit 恒等**を検証 gate で確認する（既知の例外は文書化する：1D の一部 host 集計 ledger の ~1e-15 帯 replica 揺らぎ、2D_RZ の atomicAdd 順序由来 LSB 帯 — VERIFICATION の noise-band gate 参照）。
- **退役 imc_ddmc（Monte Carlo）モードのみ** bitwise 再現を要求せず、同一GPU・同一rank数・同一seedでの**統計的再現**（主要量の平均・分散が一致、CV ≤ 0.1%）を基準としていた（歴史的要件として保持；CLAUDE.md §0.3 の RNG 分割規則はこのモードに紐づく）。
- **namelistファイル本体**と、内部で生成された"凍結設定（frozen config）"を出力へ保存し、再現を可能にする（6章）。

---

## 4. 単位系（Laser‑plasma標準）
TENRYUは **cgs + eV** を既定とし、I/Oも同系で扱う（内部でSIへ変換しない）。

- 長さ：cm（µm入力許容 → cmへ変換）
- 時間：s（ns入力許容）
- 密度：g/cm³
- 温度：eV（keV入力許容）
- 圧力：dyne/cm²（出力でMbar併記可）
- エネルギー：erg（J入力許容 → ergへ変換）
- パワー：erg/s（W入力許容 → erg/sへ変換）
- 放射群境界：eV（既定）。群代表値は幾何平均。
- 重要定数：c, k_B, h, a_rad, e, m_e, m_p … は `constants.toml`（内部）で管理（単位はcgs整合）

### 物理定数 (constants.toml, CODATA 2018 準拠)

| 定数 | 記号 | 値 (cgs + eV) | 単位 |
|------|------|---------------|------|
| 光速 | c | 2.99792458 × 10¹⁰ | cm/s |
| Boltzmann 定数 | k_B | 1.380649 × 10⁻¹⁶ | erg/K |
|  | k_B_eV | 8.617333262 × 10⁻⁵ | eV/K |
| Planck 定数 | h | 6.62607015 × 10⁻²⁷ | erg·s |
| 放射定数 | a_rad | 7.5657236 × 10⁻¹⁵ | erg/(cm³·K⁴) |
| 放射定数(eV) | a_eV = a_rad/(k_B_eV)⁴ | 1.3720 × 10⁺² | erg/(cm³·eV⁴) |
| Stefan-Boltzmann | σ_SB | 5.670374419 × 10⁻⁵ | erg/(cm²·s·K⁴) |
| 電子質量 | m_e | 9.1093837015 × 10⁻²⁸ | g |
| 陽子質量 | m_p | 1.67262192369 × 10⁻²⁴ | g |
| 素電荷 | e | 4.80320427 × 10⁻¹⁰ | esu (statcoulomb) |
| 電子ボルト | 1 eV | 1.602176634 × 10⁻¹² | erg |
| 古典電子半径 | r_e | 2.8179403262 × 10⁻¹³ | cm |

注: a_rad = 4σ_SB/c, k_B_eV = k_B / (1 eV)。
eV 単位の温度: T[eV] = k_B_eV × T[K]。

> **k_B 記号の規約**（NUMERICS §0.1 準拠）：本仕様書および NUMERICS.md の式中で温度 T [eV] と組み合わせる場合、
> k_B は **eV_to_erg = 1.602176634 × 10⁻¹² erg/eV**（= 1 eV の行）を意味する。
> 上表の k_B = 1.380649 × 10⁻¹⁶ erg/K は Kelvin 系の Boltzmann 定数であり、
> T [eV] と組み合わせる文脈では使用しない。
> 例：理想気体 EOS `P_i = ρ k_B T_i / (A m_p)` で T_i [eV] の場合、k_B = eV_to_erg。

---

## 5. 物理モデル（概要：詳細はNUMERICS）
### 5.1 流体（ALE/Lagrangian、2T）
- 既定：**ALE**（Arbitrary Lagrangian–Eulerian）
  - Lagrangianステップ → rezoning → 保存的remap
- 1D球：セル境界が半径方向に移動
- 2D RZ：四辺形セル（回転対称）でセル歪みを許容

**2温度**：
- 電子：熱伝導、レーザー吸収、輻射との交換を受ける
- イオン：PdV仕事、人工粘性Qによるショック加熱、e‑i緩和を受ける
  - v1.0既定：人工粘性による加熱は **イオンへ全量** 入れる（明示；変更はnamelistで）

**DRACO準拠（Lagrangian Hydro）**：
- スタガード格子：ノード中心に速度 \((v_r, v_z)\)、セル中心に熱力学量 \((\rho, e_i, e_e, T_i, T_e, P, Q)\)
- Wilkins型面積重みコーナー力による圧力勾配離散化
- ALE既定：Winslow equipotential rezoning + 保存的flux-based remap
- 詳細：NUMERICS.md §3.2–3.3

**多材料セルのリマップ量**：

ALE rezoneに伴うリマップでは、以下の量を材料別に保存的に移送する：

| 量 | 次元 | リマップ方式 |
|----|------|-------------|
| ρ_α × V（材料別部分質量）| [g] | 保存的フラックス積分 |
| f_{vol,α}（体積分率）| [-] | 部分質量から逆算：f_{vol,α} = (ρ_α V_α) / (ρ_total V_new) |
| ρ_α e_{e,α}（材料別電子エネルギー密度）| [erg/cm³] | 保存的フラックス積分 |
| ρ_α e_{i,α}（材料別イオンエネルギー密度）| [erg/cm³] | 保存的フラックス積分 |
| ρ_α v（材料別運動量密度）| [g/(cm²·s)] | 保存的フラックス積分 |

**v1.0 簡略化**：single-state仮定（NUMERICS §1.1.5(c)）により、全材料が同一 (T_e, T_i) を共有。
リマップ後の温度は以下で復元：
T_e^{new} = (Σ_α ρ_α e_{e,α} V_α)^{remap} / (Σ_α C_{v,e,α} V_α)^{new}

**部分密度の定義**：ρ_α = f_{vol,α} × ρ_total（体積分率 × 総密度）

### 5.2 物性（EOS/Opacity）
- EOS/Opacityは **Materials APIのみ**から参照され、Hydro/Rad/Laserはテーブルへ直接アクセスしない（設計要件）。
- SESAME/IONMIX の opacity は **質量不透明度 κ [cm²/g]** を基本とする。
- 輸送方程式へ渡す減衰係数は **長さ不透明度 σ [1/cm]**：
  - **σ = ρ κ**（この変換を仕様として固定）
- ソルバ別の平均不透明度の使い分け（致命的事故防止のため固定）：
  - **FLD（現行既定）**：拡散係数・face flux は **Rosseland mean（群別） κ_R,g**（face 値は Kirchhoff face-κ 合成が既定）、吸収/放射の物質結合（Kirchhoff整合）は **Planck mean（群別） κ_P,g**（Fleck 線形化）
  - **S_N（現行）**：群別吸収断面と Fleck 実効散乱で輸送・物質結合（NUMERICS §6.8/§8）
  - （退役 imc_ddmc：IMC 吸収/放射 = Planck mean、PGRW = collapsed \(\bar{\sigma}_t\), \(\bar{\sigma}_a\), \(D\) の毎 step 再構成、DDMC 拡散係数 = Rosseland mean）
  - いずれも内部では σ = ρ κ に変換して使用（詳細はNUMERICS 6–7章）

#### 5.2.1 Non-LTE Opacity Model（M17: IONMIX テーブル駆動 Phase 1）

> 注記（2026-07-10）：本節の separate-emissivity（\(\sigma^{PA}/\sigma^{PE}\)）の数理は現行 FLD/S_N の NLTE 経路でも共有される。一方、本節中の **IMC 粒子・DDMC・rad_lite・same-step reinjection に関する規定は退役 `mode="imc_ddmc"` 専用**の歴史的仕様である（冒頭ステータス参照）。

Non-LTE では放射源が Kirchhoff 則（\(j_\nu = \sigma_{a,\nu} B_\nu\)）に従わず、
吸収不透明度 \(\sigma^{PA}_g\) と放射不透明度 \(\sigma^{PE}_g\) が独立に供給される。
IONMIX4/6 フォーマットはこの区別をネイティブにサポートする（§ IONMIX ファイルフォーマット参照）。

- **不透明度の3種分離**: IONMIX4/6 は群ごとに Rosseland \(\kappa_R\)、Planck absorption \(\kappa^{PA}\)、Planck emission \(\kappa^{PE}\) の3種を独立に格納する。LTE テーブルでは \(\kappa^{PA} = \kappa^{PE}\) であるが、Non-LTE テーブルでは一般に \(\kappa^{PA} \neq \kappa^{PE}\)
- **群放射率の導出**: 実装の `eta_g` は角度積分済み emissivity \(\widehat{\eta}_g = \sigma^{PE}_g \cdot c \cdot a_{eV} T_e^4 \cdot b_g(T_e)\) [erg/(cm³·s)]
  - \(\sigma^{PE}_g = \rho \kappa^{PE}_g\) [cm\(^{-1}\)]。テーブルの Planck emission opacity から構成
  - LTE 回帰: \(\kappa^{PE} = \kappa^{PA}\) のとき \(\widehat{\eta}_g = c \sigma^{PA}_g a_{eV} T_e^4 b_g(T_e)\)（Kirchhoff 則と一致）
- **Jayenne separate emissivity**: \(\sigma_{p,abs}=\sum_g b_g\sigma^{PA}_g\)、\(\sigma_{p,em}=\sum_g\widehat{\eta}_g/(a_{eV}cT_e^4)\)、\(s_g=\widehat{\eta}_g/\sum_i\widehat{\eta}_i\) を組み立て、Fleck factor は既定で \(f = 1/(1+\alpha\Delta t\,\beta c \sigma_{p,em})\)、`Radiation.imc.corrected_fleck=True` では Cleveland & Wollaber (2018) の修正 \(f = 1/(1+\alpha\Delta t\,\beta c \sigma_{p,em}(1+\xi))\) を用いる（NUMERICS §6.1.1）
- **禁止事項**: raw emissivity derivative から Fleck factor を作らない。`table_nlte` / `tmat` は `lambda_method` に依らず同一の separate-emissivity path を使う
- **LTE 回帰**: テーブルの \(\kappa^{PE} = \kappa^{PA}\) の場合、\(\sigma_{p,em}=\sigma_{p,abs}\)、\(s_g=b_g\sigma^{PA}_g/\sigma_{p,abs}\)、\(\gamma_{diag}=1\) となり、旧 LTE multigroup IMC kernel を回復する
- **DDMC/rad_lite 制約**: 真の NLTE テーブル（`is_lte=false`）でも same-group-grid DDMC は使用可。空間 diffusion / leakage と IMC⇄DDMC 境界変換は従来通り群ごとの Rosseland ベースで行い、局所 effective scattering のみ \(\sigma_{s,eff,g}=(1-f)\sigma^{PA}_g\)、outgoing group law \(s_g\) へ差し替える
- **same-step reinjection**: true NLTE DDMC の local redistribution が IMC-only 群へ飛んだ場合、その粒子は同一セル内一様位置・等方方向で IMC へ降格し、残余時間を同じ step の tail IMC phase で追跡する。tail IMC では IMC→DDMC 再昇格は行わない
- **rad_lite 制約**: `rad_lite_mesh` は 1D IMC overlay として true NLTE でも使用可。coarse rad cell に単一 `eta_cdf` は持たせず、散乱時は粒子位置から member hydro cell を再構成してその hydro-cell `eta_cdf` を参照する。2D_RZ や coarse-DDMC donor split は現 scope 外
- **Phase 1 スコープ**: IONMIX .cn4 / TMAT-H5 opacity 読込・補間 + start-of-step \(J_g\) 再構成。inline CR ソルバ、outer Picard は将来拡張

#### 5.2.2 多材料セル混合（伝導・ソース結合）

多材料セルでは電子・イオン温度はセル内平衡（共有 \(T_e, T_i\)）とし、
体積分率 \(f_m\)（\(f_m \ge 0,\ \sum_m f_m = 1\)）から以下を構成する（NUMERICS §1.1.5a）：
\[
A_{eff} = \left(\sum_m \frac{f_m}{A_m}\right)^{-1},\qquad
\gamma_{eff} = \sum_m f_m\,\gamma_m
\]
\[
n_e = \frac{\rho\,\bar{Z}}{A_{eff}m_p},\qquad
c_{v,e} = \frac{\bar{Z}k_B}{A_{eff}m_p(\gamma_{eff}-1)},\qquad
c_{v,i} = \frac{k_B}{A_{eff}m_p(\gamma_{eff}-1)}
\]

このセル実効量を以下で使用する：
- 電子熱伝導（Spitzer, flux limiter, STS の \(D_{eff}\)）
- ソース結合（`inject_radiation_source_terms`, `inject_laser_source_terms` の \(e_e \leftrightarrow T_e\) クロージャ）

単一材料では \(A_{eff}=A_0,\ \gamma_{eff}=\gamma_0\) となり既存式へ厳密に退化する。

### 5.3 輻射輸送（多群：FLD / S_N 決定論 — 現行）
本番輸送は決定論ソルバ（`Radiation.mode` 既定 `"multigroup_diffusion"`、§6.4.5 を正とする）：
- **FLD（multigroup flux-limited diffusion、既定）**：群別 Rosseland 拡散 + **face 平均 E で評価する flux limiter**（Levermore–Pomraning 系の face 中心化、厚極限で古典拡散に一致）、**Fleck 線形化**の物質結合（Planck mean 吸収/放射、Kirchhoff face-κ 既定）、implicit 線形解法（CG 系；AMGX はビルドオプション）。1D（球/円筒/平面）+ 2D_RZ。詳細 NUMERICS §6.7。
- **S_N（離散座標）輸送**：保存形 FV sweep（1D 球面は \(\theta(\tau)\) + Mark–Miller 加重 diamond + 内向き開始方向で機械精度の一様黒体不動点を保証）、DSA 加速（1D）、Fleck 実効散乱 + material Newton 結合、多群。1D + 2D_RZ。詳細 NUMERICS §6.8/§8。
- 両ソルバとも境界駆動（Marshak/Tr(t) テーブル）と保存台帳検査（VERIFICATION）をサポートする。多材料セルの opacity 混合は §5.2.2 を参照。

#### 5.3.1 退役：IMC–PGRW–DDMC ハイブリッド（`mode="imc_ddmc"`、legacy）
初期設計の Monte Carlo ハイブリッド。**検証 gate は退役済み・本番使用不可**（2D_RZ でのみ選択可能な互換経路として存置、1D_SPH は `ConfigError`）。以下は歴史的仕様：
- **IMC**：Fleck–Cummings型暗黙化で強結合を安定化
- **PGRW**：optically thick だが DDMC 条件を満たさない 1D_SPH の IMC 粒子に対し、`imc_transport_persistent` 内の partially-gray random walk branch で cell-local diffusion を解析近似加速する
- **DDMC**：拡散離散式の近傍結合をリークイベントへ変換し、線形ソルバを不要化
- セル×群の transport mode map はステップ冒頭に **IMC/DDMC** で構築する。hybrid diffusion セルは粒子輸送 kernel へは IMC として渡すが、`state.ddmc_mode_map` には `TransportMode::Diffusion=3` を記録し、post-radiation source injection の再適用を抑止する
  - DDMC：ω ≥ ω_DDMC（既定 0.9）かつ τ ≥ τ_DDMC（既定 4.0）、M‑matrix条件、変換確率制約を満たす
  - それ以外は IMC（NUMERICS §7.1 参照）
- `tau_rw > 0` のときは、IMC kernel 内で 1D_SPH 専用の PGRW eligibility 判定を追加する。PGRW は独立 mode を作らず、DDMC でない diffusive group の IMC 粒子だけに適用される（NUMERICS §7.4.2 参照）
- IMC⇄DDMC境界：**asymptotic diffusion-limit境界条件**に基づく方向依存変換確率 P(μ) を使用
  - Marshak境界条件では異方入射時に不正確（Densmore 2007で実証）
  - P(μ)で変換されない分はIMC側へ反射
- DDMCのリーク係数：**面温度で評価**した面不透明度を使用（伝搬停止回避）
- DDMC運動量沈着：面フラックス経由の推定量を診断出力（NUMERICS §7.8参照）

### 5.4 レーザー（内部Ray tracing + IB）
- 幾何光学：屈折率  
  \(n_{refr}=\sqrt{\max(\varepsilon_n, 1-n_e/n_{crit})}\)
- **臨界近傍の数値発散回避を仕様で固定**：
  - \(n_{refr}\) に下限 \(\varepsilon_n\) を導入（既定 1e‑4）
  - さらに \(n_e/n_{crit}\ge 1-\varepsilon_{crit}\) でレイを終了（既定 \(\varepsilon_{crit}=1e‑4\)）
  - 残存強度は “未吸収（反射/損失）” としてエネルギー収支へ計上
- レイトレースは **LaserMesh（独立格子）上で屈折・吸収を計算 → 流体メッシュへ沈着をマップ**
  - 流体セルが void または近似セル中心 \(\hat{n}_c = \rho_c \bar{Z}_c /(A_{\mathrm{eff},c} m_p n_{\mathrm{crit}}) \ge 1\) の場合、そのセルへは沈着しない
  - 1D_SPH では void 側パワーをハンドオフステンシルで亜臨界セルへ再分配し、supercritical 実セルへ落ちた分は外側の亜臨界実セルへ付け替える
  - 1D_SPH では臨界面に隣接する最外の超臨界実セル 1 個を例外受け皿として許可し、臨界面が外側の亜臨界セル内部にある場合はその内側分の沈着をこのセルへ割り当てる
  - 1D_SPH で resolved-corona が不足している間は、critical-adjacent な亜臨界セルの沈着もベース指数重みの inward stencil で近臨界帯へ広げる
  - 遷移モデルの密度バイアスは void 側ハンドオフにのみ適用し、実表面セルの再配分には使わない
  - それでも受け皿が存在しないブロック済みパワーは Hydro へ結合せず、未吸収エネルギーとして収支へ戻す
- 吸収：逆制動輻射（IB）
  \(dI/ds=-\kappa_{IB} I\)（モデル詳細はNUMERICS）
- `laser.mode="radial_absorption_1d"` では、1D_SPH の全ビームパワーを
  \(P_{\mathrm{total}}=\sum_b P_b(t)\) として合算し、球殻面に垂直な inward radial flux を
  外側セルから内側セルへ1本積分する。レイ軌跡出力は持たず、ビーム方向・F値・焦点・プロファイル・レイ本数は吸収分布に影響しない
- **Non-goals（現況注記つき）**：共鳴吸収、LPI の波動レベル計算（SRS/SBS/TPD）、非局所輸送、自己集束/自己位相変調、磁場効果、偏光依存吸収は扱わない（NUMERICS §5.2.1）。**CBET は実装済み**（Marozas 型 pairwise ray energy exchange、1D_SPH opt-in・既定OFF — NUMERICS §5.10、§6.4 Laser.cbet）。**ホット電子プリヒートは処方源モデルとして実装済み**（quarter-critical 捕獲 + 多群 CSDA 沈着、1D — NUMERICS §5.11、§6.4 Laser.hot_electron；機構別指向性チャネル TPD/SRS は `sources` で処方 — NUMERICS §5.11.1。LPI の波動レベル自己無撞着計算は依然 non-goal）。2D RZ 輸送（ray_trace_3d 捕獲 + hydro 格子弦輸送）は本 branch で配線済み — NUMERICS §5.11 の 2D 小節参照。
- **未吸収エネルギー**は「反射/散乱で外へ逃げたエネルギー」としてエネルギー収支に計上する（入射 = 吸収 + 未吸収）

#### 5.4.1 1D_SPHにおける2D運動学 + 1D radial lookup
1D_SPH次元であっても、レーザー吸収計算の **レイ運動学は 2D RZ座標系** で行う。
ただし、**場参照と沈着は 1D radial profile** に短縮する。
レイトレースは **1ビーム分のみ実行** し、複数ビームの結果は重ね合わせで得る（NUMERICS §5.6.4）。

- 1DのRHD出力（\(\rho(r), T_e(r), \bar Z(r)\)）を球対称と仮定し、ビームローカル円筒座標 \((R,Z)\) 上に \(\rho(\sqrt{R^2+Z^2})\) 等としてマッピング
- 代表ビーム（1本目）の軸をZ軸とする **LaserMesh（2D RZ構造格子）を1つ** 生成する
  - LaserMeshは **臨界密度以下の領域のみ** をカバーする（\(n_e < n_{crit}\)）
  - ストレッチ格子を許容し、密度勾配が大きい領域でメッシュを自動的に細かくする
  - 1D_SPHでは同時に midplane \(Z=0\) から radial profile
    \((r,\hat n,\hat n_{raw},A_{smooth},d\hat n/dr)\) を抽出し、ray trace の lookup 用に保持する
- レイはビーム軸対称の **1D配列**（R方向のみ）で初期化する。F値と集光位置からレイの初期位置・方向を決定
- レイ位置・速度は \((R,Z,v_R,v_Z)\) のまま leapfrog 更新するが、場参照は
  \(r=\sqrt{R^2+Z^2}\) に対する **1D線形補間** で行う
- 吸収パワーは LaserMesh の 2D node 配列へは積まず、Hydro の 1D球対称セルへ **直接** 蓄積する
- 直接沈着後も、1D_SPH の blocked cell / ghost handoff / transition blend の再配分は従来どおり適用する
- ビーム毎にF値・プロファイルが異なる場合は、パラメータが同一のグループ毎に1回ずつレイトレースし、結果を重ね合わせる（NUMERICS §5.6.4）
- **1D_SPHの球対称性**により、ビーム方向が異なっても各ビームのRZ吸収パターンは同一であることが保証される
- **1D_SPH グループ化基準**（validate で検査）：同一グループの条件は `f_number`、`profile.model`、`profile` の全パラメータ（`w0_um`, `m`, `radius_um`）が一致すること。`direction` はグループ化に影響しない（球対称）。グループ数 = レイトレース実行回数

**等方平均（P0）モデル**：
- 全ビーム同一条件の場合に等方平均モデルを **オプション** として残す（`laser.mode="spherical_average"`）
- 既定は **2D運動学 + 1D radial lookup**（`laser.mode="raytrace_2d"`）

#### 5.4.1a 1D_SPH radial absorption integral
`laser.mode="radial_absorption_1d"` は 1D_SPH 専用の代替レーザー吸収モードである。
R-Z レイ運動学を解かず、Hydro 1Dセルの外側から中心へ向かう radial flux を逐次積分する。

- `P_total(t) = Σ_b P_b(t)` を1本の inward radial flux として扱う
- `direction`, `f_number`, `focus`, `defocus`, `profile`, `rays_per_beam` は入力として受理するが吸収分布には使わない
- IB吸収係数・単位系・未吸収エネルギー計上は §5.4 の既存規約に従う
- 2D_RZ で指定した場合は `ConfigError`

#### 5.4.2 2D_RZにおける3Dレイトレース
2D_RZ流体計算では、レーザーレイトレースを **3D空間** で行う（`laser.mode="raytrace_3d"`）。

- 流体場は軸対称 \(\rho(R,Z)\) だが、各ビームは **任意の3D方向** から入射する
- レイの位置・速度を3Dベクトル \((x,y,z)\) で追跡する（NUMERICS §5.3.4）
- 場の参照は \(R=\sqrt{x^2+y^2}\) として2D LaserMeshから取得（3Dメッシュ不要）
- 密度勾配は2D→3D変換：\(\partial\hat n/\partial x = (\partial\hat n/\partial R)(x/R)\) 等
- LaserMeshは **流体対称軸に沿う** 2D RZ格子（ビームローカルではない）
- レイ初期化は **ビーム軸直交平面上の2D配列**（1D_SPHの1D配列と異なる）
- 沈着は3D位置 \((x,y,z)\) → LaserMesh \((R,Z)=(\sqrt{x^2+y^2},z)\) → HydroMeshへ直接マッピング
- 多ビーム重ね合わせ：ビームの **極角θ** とパラメータ（F値、プロファイル）でグループ化（NUMERICS §5.6.4）
  - 軸対称ターゲットでは同一極角θのビームが同一吸収パターンを生成
  - グループ毎に1回の3Dレイトレースを実行し、パワースケーリングで重ね合わせ

### 5.5 v/c項の扱い（明示）
- v1.0では、放射輸送は基本的に“静止媒質（lab frame）”形で、**O(v/c)** の仕事項・ドップラー項を無視する。  
  ICF典型で u/c≪1 を根拠とするが、差分要因として出力メタデータに明示する（NUMERICS参照）。

---

## 6. 入力仕様：**単一ファイル Python namelist（.py）**
TENRYUは **すべてのシミュレーション条件を1つのPythonファイル**に記述する。  
このファイルを **namelist** と呼ぶ（Smilei方式に寄せる）。

### 6.1 実行CLI
- 通常実行：
  - `tenryu run <namelist.py>`
  - `tenryu run <namelist.py> --restart <checkpoint_prefix>`（任意。`Main.restart_from` の実行時オーバーライド）
  - `tenryu run <namelist.py> --output-dir <dir>`（任意、2026-07-12 追加 — GUI Studio M3 報告の解消。`Output.directory` の実行時オーバーライド。**新規 run 専用**: `--restart` または `Main.restart_from` との併用は `ConfigError`（restart は元の出力レイアウトを継続するため）。上書き後の値は frozen config に自己記述される）
- 検証：
  - `tenryu verify all`
  - `tenryu verify <case>`
- 入力検査（実行せずパースのみ）：
  - `tenryu validate <namelist.py>`
  - `tenryu validate <namelist.py> --n-ranks <P>`（M16以降：rank依存の並列整合性チェックを有効化）
  - 検証成功時は**プリフライト要約**を出力する（2026-07-11 追加）: main（name/次元/幾何/t_end）、mesh（セル数・範囲）、材料ごとの EOS/opacity モデル、radiation（mode/群数/境界）、laser（mode/beams/cbet/hot_electron）、hydro（plasma_viscosity）、conduction（nonlocal_model）、burn（scheme/screening/fuels）、dt、output。実行前の設定ミス発見を目的とする人間可読出力であり、機械可読の正典は frozen JSON（§6.2）
  - 初学者向け入門は `docs/TUTORIAL_ja.md`、雛形 deck は `examples/templates/`（ctest `validate_template_*` で常時パース検証）
- 凍結設定の出力（再現性用）：
  - `tenryu freeze <namelist.py> -o frozen_case.json`  
    （※`run`時にも自動で frozen を出力へ保存する）

### 6.2 namelistの基本構造（Smileiライク）
namelistは **TENRYUが提供するPython API**（`tenryu_namelist`）を import して、ブロック関数を呼ぶ形式にする：

```python
from tenryu_namelist import *

Main(...)
Mesh(...)
Materials(...)
Geometry(...)
Radiation(...)
Laser(...)
Numerics(...)
Output(...)
Diagnostics(...)
Parallel(...)
```

- 各ブロックは **1回だけ**呼ぶ。`Material()` はブロックではなく辞書を返すヘルパーであり、**`Materials(materials=[Material(...), ...])`** のリスト要素としてのみ使用する。リスト内の順序が材料インデックス（0始まり）を決定する。
- ブロック呼び出し順は原則自由（TENRYU側で整合性検査する）が、推奨は上記。
- **ファイルパス解決規則**：namelist 内で指定するファイルパス（EOS テーブル、opacity テーブル等）は、**相対パスの場合は namelist ファイルの所在ディレクトリを基準に解決** する。絶対パスはそのまま使用する。`~` はホームディレクトリに展開する。環境変数の展開は行わない。`Main.restart_from` も同一規則（namelist ディレクトリ基準）。`Output.directory` の相対パスは **実行時カレントディレクトリ（CWD）基準** で解決する（出力先はワークフロー依存のため。CLI `--output-dir` 指定時はこの値を上書き）。
- **ユーザ定義関数**（密度・温度・波形など）は namelist 内で自由に定義して良い。
- Python↔C++バインディングには **pybind11** を使用する。
  - 選定理由：型安全なC++↔Pythonバインディング、科学計算コミュニティでの広い採用実績、ヘッダオンリー（追加ライブラリ不要）
  - CPython C API の直接使用は行わない（保守性・移植性のため）
- 実行時、TENRYUは
  1) CPythonインタプリタを組み込み起動し、pybind11経由でnamelistを実行
  2) ブロック引数を検証
  3) 物理量を内部単位（cgs+eV）へ変換
  4) 初期場を生成（geometry関数を格子上で評価）
  5) 波形・群境界などを"凍結"してC++側へ渡す
  という流れを取る。

**呼び出し規約：グローバルレジストリ方式**

各ブロック関数（`Main()`, `Mesh()`, `Materials()` 等）はグローバルシングルトンに設定を登録する。
戻り値は使用しない（副作用ベース）。

```python
# tenryu_namelist モジュールの内部実装概要：
_registry = {}

def Main(**kwargs):
    _registry['main'] = kwargs

def Mesh(**kwargs):
    _registry['mesh'] = kwargs

def Materials(**kwargs):
    # 特殊：materials キーにリストを格納
    _registry['materials'] = kwargs.get('materials', [])

def Material(**kwargs):
    # Material() は Materials(materials=[...]) 内でのみ使用
    return kwargs  # 辞書を返す（登録はしない）

# C++側は Py_Finalize 前に _registry を読み取り Config を構築
```

**ユーザー側の使用例**：
```python
from tenryu_namelist import *

Main(name="test", dimension="1D_SPH", t_end=1e-9, seed=42)
Mesh(r_min=0.0, r_max=0.05, nr=200)
Materials(materials=[
    Material(name="DT", A=2.5, Z=1.0, ...),
])
# ... 他のブロック
```

**制約**：
- 各ブロック関数は1回だけ呼ぶこと（2回目は上書き、警告を出力）
- `Material()` は `Materials(materials=[...])` のリスト要素としてのみ使用
- `from tenryu_namelist import *` は pybind11 経由で C++ から注入されるモジュール

**値の型規約**（Python namelist 固有）：
- **bool**：Python の `True` / `False` のみ。`1`/`0`, `"yes"`/`"no"` は不可（型エラー）
- **数値**：Python の `int` / `float` リテラル（例：`1e-9`, `0.3`, `200`）
- **文字列**：Python の `str`（`"vacuum"`, `'reflect'` 等。引用符必須）
- **リスト**：Python の `list`（例：`[0.01, 0.1, 1.0, 10.0]`）
- **辞書**：Python の `dict`（例：`{"type": "graded", ...}`）
- **callable**：Python の関数オブジェクト（`def` で定義、または `lambda`）
- **同一パラメータの重複指定**：同一ブロック内で同じキーワード引数を重複して渡した場合は Python の標準動作（`SyntaxError`）に従う

### 6.3 "関数で与える"プロファイル仕様（最重要）
#### 6.3.1 座標系と関数シグネチャ
- 1D_SPH：セル中心半径 `r_cm` を引数に取る
  - `rho(r_cm)`, `Te(r_cm)`, `Ti(r_cm)` など
- 2D_RZ：セル中心座標 `(r_cm, z_cm)` を引数に取る
  - `rho(r_cm, z_cm)`, `Te(r_cm, z_cm)` …

**禁止**：実行中ステップごとにPython関数を呼ぶ設計（性能と再現性が崩れる）。  
→ TENRYUは初期化時に関数を評価して **配列に固定**する。

#### 6.3.2 ベクトル化（性能推奨）
関数は **NumPy配列入力**にも対応する実装を推奨する：
- 入力：`r_cm` が `numpy.ndarray`
- 出力：同shapeの `numpy.ndarray`

TENRYUは初期化時に座標配列を渡して一括評価する（Python呼び出し回数を最小化）。

#### 6.3.3 多材料セルの指定（体積分率関数）
Geometryは、点（セル中心）ごとに **材料体積分率（または質量分率）**を返す関数で指定する。

既定仕様（体積分率）：
- `volfrac(material_name, coords...) -> [0..1]`
- 全材料の和が1になるようにする（TENRYUが正規化も行うが、ユーザ関数側で満たすのが推奨）

例：球殻（shell）と燃料（fuel）
```python
def vf_shell(r):
    return 1.0 if (R_in < r < R_out) else 0.0

def vf_fuel(r):
    return 1.0 if (r <= R_in) else 0.0
```

#### 6.3.4 物性（EOS/opacity）の指定場所
**材料定義（Materialsブロック）**の中で、EOS/opacityのモデルと入力ファイル（パス）を指定する。  
Geometryと同じnamelist内にあるため、ユーザは “どの領域にどの材料テーブルを使うか” を1ファイルで完結できる。

### 6.4 ブロック仕様（型・既定値）
> ここで列挙するキーは v1.0で必ず解釈できる“固定API”。  
> 将来拡張で引数は増やせるが、既存引数の意味を変えない（後方互換）。

#### 6.4.1 Main(...)
- `name: Optional[str]`（既定：namelistファイル名（拡張子なし）。§7.1参照）：ケース名。有効文字: `[a-zA-Z0-9_-]`。最大長: 64。空文字列不可
- `dimension: Literal["1D_SPH","1D_CYL","2D_RZ"]`
  - `"1D_CYL"`：1D cylindrical per-unit-length radial hydro（B1; wave-1 scope pure hydro core — radiation/laser/conduction/ale1d are rejected at validation; NUMERICS 1D geometry note; verification gate H3-RADIAL-CYL）
- `temperature_model: Literal["1T","2T","auto"]`（既定 `"auto"`）
  - `"1T"`：単温度モード（電子・イオンを単一温度で扱う）
  - `"2T"`：二温度モード（既存挙動）
  - `"auto"`：初期化時に `"2T"` へ解決（後方互換のための既定）。実行時に WARNING を出力
- `t_end: float`：終了時刻 [s]（有効範囲：`> 0`）
- `seed: int`：64-bit seed（Philox）（既定 `12345`；有効範囲：`0 ≤ seed ≤ 2^64-1`）
- `restart_from: Optional[str]`：checkpointプレフィックス（既定 `None`）。形式：`<dir>/<case>_ckpt_NNNNNN`（ランクサフィックス `_rNNNN.h5` は省略）。指定されたプレフィックスに一致するファイルが存在しない場合は `ConfigError("checkpoint files not found: {prefix}_r*.h5")`。§7.4のリスタート手順に従う。CLI `--restart <checkpoint_prefix>` 指定時はこの値を実行時に上書きする
- `units: Literal["cgs_eV"]`：既定 `"cgs_eV"`（変更禁止）
- `max_steps: int`（既定 `10_000_000`）— 最大ステップ数。有効範囲: `[1, 16_777_215]`（= 2²⁴ − 1。global_id = step × N_max_per_step(2⁴⁰) の uint64 オーバーフロー回避、NUMERICS §12.7.1）
- **停止条件の優先順位**：シミュレーションは `t >= t_end` **または** `step >= max_steps` のいずれか **先に到達した方** で終了する。終了理由は history 出力の `attrs: termination_reason`（`"t_end"` or `"max_steps"`）に記録される
- `verbosity: Literal["quiet", "normal", "verbose", "debug"]`（既定 `"normal"`）

#### ログ出力レベル

| レベル | 出力内容 |
|--------|---------|
| quiet  | ERROR, FATAL のみ |
| normal | ERROR, FATAL, WARNING, step 進捗 (100 step 毎) |
| verbose | 上記 + INFO (全 step 進捗、エネルギー収支、粒子統計) |
| debug  | 上記 + DEBUG (カーネル起動パラメータ、MPI通信量、メモリ使用量) |

CLI フラグ `--verbose` は `verbosity="verbose"` と等価。
`--quiet` は `verbosity="quiet"` と等価。
CLI フラグは namelist 設定を上書きする。

#### 6.4.2 Mesh(...)
- 1D_SPH：
  - `r_min: float` [cm]（通常0；有効範囲：`≥ 0`）
  - `r_max: float` [cm]（有効範囲：`> r_min`）
  - `nr: int`（有効範囲：`≥ 4`；Kershawステンシル＋ゴースト層に4セル以上必要）
  - `grid: Literal["graded"] | dict`（既定 `"graded"`）
    - 1D_SPH の初期メッシュは常に graded として生成する
    - `grid="graded"` の場合は `nr`, `r_min`, `r_max` から単一区間の graded メッシュを構成する
    - `grid=dict(type="graded", segments=[...], grading={...})` の場合は下記 `graded` 指定を用いる
  - `grid_type_r/grid_r` は後方互換のため受理するが、1D_SPH では値に関わらず内部で `"graded"` に固定する
  - `geometry_1d: Literal["spherical","cylindrical","planar"]`（既定 `"spherical"`；W-G。1D の座標幾何を選ぶ: 面積 \(4\pi r^2\) / \(2\pi r\)（単位長） / \(1\)（単位面積）、体積はそれぞれの殻体積。**1D_SPH 専用**（2D_RZ で指定すると `ConfigError`）。非球面の制約: `mode="imc_ddmc"` 不可、`"cylindrical"` は `sn_transport` 不可（W-G3 で製品化予定 — 1D 円筒 S_N は積求積が必要）、`Laser.enabled=True` は `mode="radial_absorption_1d"` 必須（raytrace_2d の beam-local mesh は球面専用）。既定 `"spherical"` は W-G 以前と bitwise 同一（NUMERICS §3 の不変契約参照））
- 2D_RZ：
  - `r_min,r_max,z_min,z_max: float` [cm]（有効範囲：`r_min ≥ 0`, `r_max > r_min`, `z_max > z_min`。**v1.0制約**：`r_min = 0` 必須（R=0対称軸を前提）。`r_min > 0`（円筒空洞）は将来版で対応予定（非軸内壁BCの定義が必要）。`r_min > 0` を指定した場合は `ConfigError("r_min > 0 (cylindrical cavity) is not supported in v1.0")`。放射境界は `Radiation.boundary.r_inner`（§6.4.5）で指定）
    - 2D_RZ `rectangular_rz` の場合、`r_min`, `r_max`, `z_min`, `z_max` は全て **必須パラメータ**（デフォルト値なし）であり、省略時は `ConfigError("2D_RZ requires explicit r_min, r_max, z_min, z_max")` を送出する一方、`polar_in_box` では bounding box を `r_min=0`, `r_max=box_r_max`, `z_min=box_z_min`, `z_max=box_z_max` として派生し、これらを明示する場合は相対許容差 `1e-12` 以内で一致必須。
  - `nr,nz: int`（有効範囲：各 `≥ 4`）
  - `grid_type_r/grid_type_z: Literal["uniform","graded"]`（既定 `"uniform"`）。2026-07-17 stage A/A' より 2D_RZ でも radial（r または polar の s）と z（rectangular_rz のみ）の graded を受理する。`grid_type_z="graded"` は `grid_segments_z`（下記 `grid_z` dict）必須で、`spherical_polar_halfplane`（z index は角度）では `ConfigError`。
  - `grid_r/grid_z: str | dict` — 文字列形は `grid_type_*` の別名（後方互換）。dict 形 `grid_r=dict(type="graded", segments=[...], grading={...})` / `grid_z=dict(...)` は方向別に segments を与える（`grid_r` dict → 径方向、`grid_z` dict → z 方向）。`grading` は共有（`grid`/`grid_r`/`grid_z` のうち **1 箇所のみ**で指定可、複数指定は `ConfigError`）。dict 形と対応する `grid_type_*` キーの併用は `ConfigError`（曖昧）。segment 境界ノードは**厳密に pin** される（2D 経路は `build_graded_nodes(pin_segment_boundaries=true)` — 材料界面がリング/格子面に exact に載る body-fit 契約。1D は歴史的挙動を bitwise 維持）。
  - `grid: Literal["uniform","graded"] | dict` — 2D でも受理。dict 形は r と z の両方に同じ type を設定するため、2D で r のみ graded にする場合は `grid=dict(type="graded", segments=[...])` に `grid_type_z="uniform"` を併記する（z gate のエラーメッセージが誘導する）。
  - multiblock topology（`topology_scheme != "single_block"`）での `grid` segments（2026-07-17 stage B2）: 径方向 segments は **polar shell の半径分布**を駆動する。被覆は `[multiblock_cart_core_r_match, spherical_polar_s_max]` 必須（`1e-12` 相対以内 → 端は exact snap）、zone 合計 = `nr`（shell 径方向数）。シェル内の材料界面（capsule ablator リング等）が shell リングに**厳密 pin**される。segments 省略時は歴史的一様分布（**bitwise 不変**）。multiblock + `explicit_nodes`/`auto_regions` は `ConfigError`（auto は繰延）。診断は `[mesh-graded] multiblock shell radial nodes` 行。
  - `grid_theta: dict`（省略可、**dict 形のみ**。2026-07-17 stage A''）。`spherical_polar_halfplane` 専用の θ 方向 segments: `grid_theta=dict(type="graded", segments=[dict(r_start,r_end,nr), ...][, grading={...}])`。θ は **radian** で、segments は `[0, π]` を被覆必須（先頭 `r_start==0.0` 厳密、末尾 `r_end` は π と `1e-12` 以内 → 極ノードは内部で exact π に snap; GridSegment のフィールド名 r_start/r_end/nr を θ の意味で再利用）。segment 境界は θ 面に**厳密 pin** — コーン壁 `θ_cone` やストーク wedge 境界を格子面に exact に載せる（cone/stalk sector primitive）。`nz`（角度 index）は segment 合計から自動設定（`nz==1` は未設定扱い、明示値と食い違うと `ConfigError`）。`polar_equal_mu_zoning` と排他。`rectangular_rz`・1D deck では `ConfigError`。`grading` の 1 箇所ルールは `grid`/`grid_r`/`grid_z`/`grid_theta` の 4 者で共有。診断は `[mesh-graded] spherical_polar theta nodes` 行（`max_adjacent_dtheta_ratio` 含む）。
  - `explicit_nodes: list[float]`（省略可）。有限値の狭義単調増加ノード列で、`nr=len(explicit_nodes)-1` を設定する（明示 `nr` と不一致なら `ValueError`）。同じ r 方向の `auto_regions` / `grid_r` と排他。single_block only（multiblock uses shell grid segments）。
  - `explicit_nodes_z: list[float]`（省略可）。有限値の狭義単調増加ノード列で、`nz=len(explicit_nodes_z)-1` を設定する（`nz==1` は未設定扱い、明示値と不一致なら `ValueError`）。同じ z 方向の `auto_regions` / `grid_z` と排他。single_block only（multiblock uses shell grid segments）。
  - `explicit_nodes_theta: list[float]`（省略可）。有限値の狭義単調増加 θ ノード列 [radian] で、`nz=len(explicit_nodes_theta)-1` を設定する。`grid_theta` と排他で、`spherical_polar_halfplane` の `[0, π]` を被覆必須。single_block only（multiblock uses shell grid segments）。
  - `auto_regions: list[dict]`（省略可）。**自動ゾーニング**（stage A、docs/design/mesh2d_graded_bodyfit_20260717.md）: 各領域 `dict(r_end [cm 必須・狭義単調増加], nz [int>0 必須], rho_ref [g/cc ≥0 必須], is_void [bool 省略可], material_group [str 省略可])`。`core/auto_zone` エンジン（質量比制約 + bridge 帯）でノード列を計算し `explicit_nodes` に格納、`nr`（axis="z" では `nz`）を自動設定する（明示値と食い違うと `ConfigError`; `nz` は 1 を未設定と見なす）。最終領域の `r_end` は外側境界（rect/1D: `r_max`、polar: `spherical_polar_s_max`、axis="z": `z_max`）と一致必須。`grid` segments と排他。polar `annular` とは併用不可（segments を使う）。診断は `[mesh-autozone]` 行（質量比 min/mean/max・違反数・警告）。
  - `auto_regions_axis: Literal["r","z"]`（既定 `"r"`）。`"z"` は 2D_RZ `rectangular_rz` 専用（planar 幾何で z 方向に自動ゾーニング — 平板積層ターゲット用）。
  - `auto_zone: dict`（省略可）。auto_regions のアルゴリズム調整: `mass_ratio_max`（既定 1.3、>1）、`n_bridge_min/max`（2/10）、`bridge_frac_max`（0.25）、`rho_void_cut`（1e-6 g/cc）、`dr_min`（1e-8 cm）、`mass_ratio_hard_max`（2.0）、`max_iter`（30）、`bulk_mass_tol`（1e-3）。
  - `topology_scheme: Literal["single_block","multiblock_cart_core_polar_shell","multiblock_half_butterfly_5block","multiblock_half_butterfly_trifan_cap_5block","pentagon_belt_shell"]`（既定 `"single_block"`；単位なし）。`"single_block"` は既存の単一ブロック topology。`"multiblock_cart_core_polar_shell"` は I1-B γ MVP の Cartesian core + Hermite bridge + polar shell topology で、equation details are in NUMERICS §3.2 and the runtime ownership/CSR layout is in ARCHITECTURE §4.2. `"multiblock_half_butterfly_5block"` は B-S1 5-block R-Z half-plane half-butterfly (central half-rect core + north/east/south angular fan blocks + polar shell); removes the square-core diagonal-corner rank-loss by replacing it with finite-valence multiblock vertices; opt-in; reuses `multiblock_cart_core_*` sizing params. B-S1 builds the in-memory block/index/node-CSR structure, production central+Coons/TFI-fan+polar-shell coordinates, face adjacency/seam tags, and additive `/mesh/topology/v3` checkpoint metadata. B-S2 retargets hydro consumers to block-role metadata, CSR lookup, and `cell_orientation_sign`, so this topology is hydro-runnable for the accepted gentle closed at-rest smoke. B-S3 seam-flux-under-gradient and B-S4 compression/PAB acceptance remain required before claiming the production compression gate. `"multiblock_half_butterfly_trifan_cap_5block"` is an S3 default-off opt-in value for the pinned tri-fan cap graft. It uses \(N_{\rm cap}=N_c\), cap cells \(4N_cN_{\rm cap}=4N_c^2\), cap nodes \(1+N_{\rm cap}(4N_c+1)\), one pinned apex at \((R,Z)=(0,0)\), first-row triangular cap cells (`cell_nverts=3`), a shared outer cap ring whose node IDs are the fan inner seam IDs, and the existing fan/shell counts. S3 T0-T5 retarget the runtime machinery to the active-slot `cell_nverts` contract: CSR remap/GCL, hydro corner/node mass, pressure force and compatible force-work, CSW AV, subzonal pressure, CFL, ALE barrier/axis/full-patch driver, and pinned-apex projection are cap-aware. The topology remains opt-in and default-off. `"pentagon_belt_shell"`（staged: plumbing and counts are present, but mesh construction lands in ALE P2-2b-2 and currently fails loud. ALE design P2 pentagon belt rings）。
  - `pentagon_belt_layers: list[int]` (default `[]`; valid only with `topology_scheme="pentagon_belt_shell"`). The list contains node-ring indices \(b_j\), must be non-empty, strictly increasing, contain at most four entries, and satisfy \(1\le b_j\le nr-2\). For \(K=\mathrm{len}(\texttt{pentagon_belt_layers})\), `nz` must be divisible by \(2^K\) and `nz >> K >= 4`. The belt scheme requires `logical_mesh_2d="spherical_polar_halfplane"` and `polar_center_treatment="annular"`. Uniform-\(\theta\) and `polar_equal_mu_zoning` ladders are accepted; `explicit_nodes_theta`, `grid_segments_theta`, and Radiation `mode="sn_transport"` are staged and rejected.
  - `multiblock_cart_core_r_c: float` [cm]（既定 \(S_{\max}/12\)；有効範囲：`>0` かつ `sqrt(2)*r_c < multiblock_cart_core_r_match`）。Cartesian half-core の半幅。`topology_scheme="multiblock_cart_core_polar_shell"`、`"multiblock_half_butterfly_5block"`、または `"multiblock_half_butterfly_trifan_cap_5block"` の場合のみ有効で、`single_block` または省略時に指定すると `ConfigError`。
  - `multiblock_cart_core_r_match: float` [cm]（既定 `2*multiblock_cart_core_r_c`；有効範囲：`sqrt(2)*r_c < r_match < spherical_polar_s_max`）。Hermite bridge/fan と polar shell の接続半径。`topology_scheme="multiblock_cart_core_polar_shell"`、`"multiblock_half_butterfly_5block"`、または `"multiblock_half_butterfly_trifan_cap_5block"` の場合のみ有効。
  - `multiblock_cart_core_n_c: int`（既定 `nz/4`；有効範囲：`>=4` かつ `nz=4*n_c`；単位なし）。Cartesian half-core の片側分割数で、half-square/fan boundary は \(4n_c\) segments を持つ。`topology_scheme="multiblock_cart_core_polar_shell"`、`"multiblock_half_butterfly_5block"`、または `"multiblock_half_butterfly_trifan_cap_5block"` の場合のみ有効。
  - `multiblock_cart_core_bridge_layers: int`（既定 `max(4, round(n_c/8))`；有効範囲：`>=1`；単位なし）。Cartesian half-core と polar shell の間の Hermite bridge 層数、または 5-block fan radial layer count。`topology_scheme="multiblock_cart_core_polar_shell"`、`"multiblock_half_butterfly_5block"`、または `"multiblock_half_butterfly_trifan_cap_5block"` の場合のみ有効。
  - `multiblock_cart_core_bridge_grading: Literal["uniform","quintic_log"]`（既定 `"uniform"`；単位なし）。`topology_scheme="multiblock_cart_core_polar_shell"` の場合のみ指定可能。`"quintic_log"` は `multiblock_transition_scheme="hermite_bridge"` を必須とし、core spacing から最初の shell spacing まで bridge layer を quintic-log grading する。
  - `multiblock_transition_scheme: Literal["hermite_bridge","rounded_half_butterfly","rounded_core_seam"]`（既定 `"hermite_bridge"`；単位なし）。`"hermite_bridge"` は既存の I1-B γ-MVP Hermite square→circle bridge を保持する。`"rounded_half_butterfly"` は S5 rounded half-butterfly/O-grid transition generation 用の opt-in で、`topology_scheme="multiblock_cart_core_polar_shell"` の場合のみ有効。`"rounded_core_seam"` rounds the shared l=0 core/bridge seam (full superellipse, \(a=r_c\cdot2^{1/p}\)) and regenerates the Cartesian core interior; opt-in; requires `multiblock_cart_core_polar_shell` topology. It is retained as a falsified A1 diagnostic/superseded path, not the B-S1 topology.
  - `multiblock_cap_p: float`（既定 `6.0`；有効範囲：`>2`；単位なし、dimensionless）。Rounded/squircle cap exponent \(p\). For S5 rounded-core transition generation it defines the superellipse \(s_p(\theta)=a(|\sin\theta|^p+|\cos\theta|^p)^{-1/p}\) when `multiblock_transition_scheme="rounded_half_butterfly"` or `"rounded_core_seam"`. For `topology_scheme="multiblock_half_butterfly_trifan_cap_5block"`, it is the pinned-cap squircle exponent used by cap geometry generation. The key is frozen-config elided at its default when unused, so existing decks remain byte-identical.
  - `multiblock_bridge_elliptic_sweeps: int`（既定 `0`；有効範囲：`>=0`；単位なし）。S5 transition generation の elliptic/Winslow smoothing sweeps。
  - `multiblock_bridge_elliptic_omega: float`（既定 `0.5`；有効範囲：`0 < omega < 2`；単位なし）。S5 transition generation の elliptic smoothing relaxation。
  - `multiblock_outer_svec_tangent_balance: bool`（既定 `true`）。multiblock outer-shell の各 non-pole node で、隣接する corner Svec pair の接線方向和を geometry recompute ごとに射影して I1-B S3-T3 G1 constant-state seam-GCL closure を保つ。BUG-25: strong drive では outer arc の接線復元力を毎 step 消去して pole-adjacent outer cells を不安定化するため、boundary-acceleration projection への置換が入るまでは drive deck で `false` にすること。
  - `logical_mesh_2d: Literal["rectangular_rz","spherical_polar_halfplane","polar_in_box","cone_shell"]`（既定 `"rectangular_rz"`）。`"rectangular_rz"` は従来の一様RZ格子を生成する。`"spherical_polar_halfplane"` は Phase 6 opt-in の半平面球座標論理格子で、`nr` を \(N_s\)、`nz` を \(N_\theta\) と解釈し、\(\theta\in[0,\pi]\), \(r=s\sin\theta\), \(z=s\cos\theta\) の節点を生成する。`"polar_in_box"` は `explicit_nodes` で固定した exact polar prefix から rectangular box へ遷移する単一論理ブロックを選択する。`"cone_shell"` は `WALL`, `OUTER_NF`, `INNER_NF`, `END_NF`, `TIP_CORNER_O`, `TIP_CORNER_I`, `CAVITY_CORE`, `EXTERIOR_CORE`, `TIP_FILL_WEST`, `TIP_FILL_MID`, `TIP_FILL_EAST` の固定11-block watertight standalone assembly を生成し、wall 周囲を gas fill で埋め、全外部境界 edge を axis または box 上に置く。内部で `topology_scheme="cone_shell_spine"` を派生設定する（deck からの同 scheme 明示は禁止）。cone_shell と polar_in_box の runtime scope は mesh build/init/step-0 output のみで、最初の time-step 試行は `ConfigError` になる。Phase 6-minimum では `"spherical_polar_halfplane"` のみ hydro-only scope とし、`Radiation.enabled=True`、`Laser.enabled=True`、または `Numerics.conduction.enabled=True` は deck parse の final validation で hard-fail する。**境界（2026-07-17）**: `Numerics.hydro.boundary_2d` は wedge では `r_outer ∈ {"free","reflect","pressure"}` のみ受理（reflect = 球弧法線速度消去+半径 `s_max` pin; pressure = PAB drive 経路が力を供給し本関数では拘束なし; それ以外は `ConfigError` — 従来は**全文字列が黙って無視**されていた）。`z_bottom/z_top` は構造的極線（axis-pin）で `"free"`/legacy `"reflect"` のみ、`r_inner` は `"axis"` 必須（annular 内弧 BC は繰延）。既知の注意: reflect 弧 pin は一様外圧を支える配置で境界層永年不安定を起こしうる（決定論再現あり — 閉箱には `pressure` 閉包を推奨、docs/design/mesh2d_graded_bodyfit_20260717.md 発見 2）。
  - `polar_center_treatment: Literal["annular","tri_fan","button"]` (default `"annular"`). This key is used only with polar-family `logical_mesh_2d` values (`"spherical_polar_halfplane"` and `"polar_in_box"`); `polar_in_box` Stage 1 の一般RZ四角形初期化は `"annular"` のみ受理し、`tri_fan`/`button` は後続stageまで `ConfigError` とする。`"annular"` preserves the existing finite inner spherical core. `"tri_fan"` keeps the structured `(nr+1)*(nz+1)` node array but sets the `i=0` node row to the origin and derives three-corner center cells in memory. `"button"` replaces the covered inner radial rings with one active central seam-node polygon cell `c=0`; the covered structured cells are dormant/void and zeroed. `tri_fan` supports pure-Lagrangian hydro and first-order conservative reference remap; center-aware rezone smoothing, anti-hourglass, HLLC z-flux, precise RZ geometric CFL, and total-energy remap remain guarded/deferred. `button` supports button-aware Lagrangian hydro, conservative remap, acoustic/geometric CFL, scalar VNR artificial viscosity, and additive HDF5 cell flags; quad anti-hourglass, HLLC z-flux, precise RZ geometric CFL, and total-energy remap remain guarded/deferred.
  - `center_button_outer_node_ring: int` (default `2`; valid `1 <= center_button_outer_node_ring < nr`). Used only when `polar_center_treatment="button"`. It selects the outer node ring \(I_{\rm btn}\) whose seam nodes define the central button polygon; structured cells with radial index \(i<I_{\rm btn}\), except `c=0`, are dormant/void.
  - `polar_equal_mu_zoning: bool` (default `False`). Opt-in angular node zoning for polar-family meshes with `polar_center_treatment="tri_fan"`. `False` preserves the legacy uniform-\(\theta\) nodes \(\theta_j=j\pi/N_\theta\). `True` uses equal-solid-angle nodes \(\theta_j=\arccos(1-2j/N_\theta)\), with the arccos argument clamped to \([-1,1]\).
  - `polar_theta_min: float` [rad] (default `0.0`; valid finite `0 <= polar_theta_min < 2.6`). `0.0` preserves the full `[0, pi]` `spherical_polar_halfplane` mesh bitwise. A positive value truncates the angular ladder to `[polar_theta_min, pi]`, requires `logical_mesh_2d="spherical_polar_halfplane"` and `polar_center_treatment="tri_fan"`, tags the `j=0` side as `PolarCutFace`, and treats that standalone cut ray as a reflecting mirror boundary.
  - `spherical_polar_s_max: float` [cm]（既定 `1.0`；有効範囲：`>0`）。`logical_mesh_2d="spherical_polar_halfplane"` の外半径、または `"polar_in_box"` の exact polar prefix 外半径 \(s_b\)。後者では最後の `explicit_nodes` と一致必須。
  - `box_r_max: float` [cm]（`polar_in_box` では必須；有効範囲：`>0` かつ `> spherical_polar_s_max`）。rectangular box の右端半径。
  - `box_z_min: float` [cm]（`polar_in_box` では必須）。rectangular box の下端で、`box_z_min < box_center_z - spherical_polar_s_max` を満たす。
  - `box_z_max: float` [cm]（`polar_in_box` では必須；`> box_z_min`）。rectangular box の上端で、`box_z_max > box_center_z + spherical_polar_s_max` を満たす。
  - `box_center_z: float` [cm]（既定 `0.0`；`box_z_min < box_center_z < box_z_max`）。axis 上の polar prefix 中心位置。
  - `cone_shell_alpha: float` [rad]（既定は未設定；`cone_shell` では必須；有効範囲：有限かつ `0 < alpha < pi/2`、corner conditioning は `abs(cos(alpha)) >= 0.2`）。truncated cone の axis から測る half-angle。
  - `cone_shell_wall_thickness: float` [cm]（既定は未設定；`cone_shell` では必須；有効範囲：有限かつ `>0`）。wall face 間を法線方向に測った物理 shell 厚さ \(t_w\)。
  - `cone_shell_tip_radius: float` [cm]（既定は未設定；`cone_shell` では必須；有効範囲：有限かつ `>0`、kind 変換後の inner-face truncation radius も `>0` 必須）。tip truncation radius。
  - `cone_shell_tip_radius_kind: Literal["inner_face","mid_surface"]`（既定 `"inner_face"`）。`cone_shell_tip_radius` を truncation plane 上の inner face または wall mid-surface のどちらで解釈するかを選ぶ。
  - `cone_shell_tip_z: float` [cm]（既定は未設定；`cone_shell` では必須；有効範囲：有限）。planar tip truncation の axial coordinate \(z_t\)。
  - `cone_shell_wall_length: float` [cm]（既定は未設定；`cone_shell` では必須；有効範囲：有限かつ `>0`）。truncation から base までの wall mid-surface arclength \(L_w\)。
  - `cone_shell_axis_sign: int`（既定 `+1`；許可値 `{-1,+1}`）。tip から base へ進む axis 方向 \(\sigma_z\)。
  - `cone_shell_n_cells: int`（既定 `10`；有効範囲：偶数かつ `>=6`）。symmetric through-wall ladder の総 cell 数 \(N_n\)。
  - `cone_shell_n_growth: float`（既定 `1.25`；有効範囲：有限かつ `(1,2]`）。through-wall ladder 各半分の face-to-center 幾何成長率。
  - `cone_shell_tip_size_factor: float`（既定 `3.0`；有効範囲：有限かつ `[1,64]`）。tip-side along-wall target width \(h_t=f_t h_{n,0}\) の倍率 \(f_t\)。
  - `cone_shell_base_size_factor: float`（既定 `8.0`；有効範囲：有限、`cone_shell_tip_size_factor` 以上かつ `<=64`）。body/base-side along-wall target width \(h_b=f_b h_{n,0}\) の倍率 \(f_b\)。
  - `cone_shell_tip_hold: float` [cm]（既定は未設定、builder が `4*cone_shell_wall_thickness` を派生；有効範囲：有限かつ `>=0`）。fine tip spacing を保持する arclength \(L_h\)。
  - `cone_shell_grading_length: float` [cm]（既定は未設定、builder が `min(0.35*cone_shell_wall_length,20*cone_shell_wall_thickness)` を派生；有効範囲：有限かつ `>0`）。tip から body spacing への quintic-log grading 長 \(L_g\)。
  - `cone_shell_l_ratio_max: float`（既定 `1.12`；有効範囲：有限かつ `(1,1.5]`）。along-wall neighbor cell-size ratio repair の上限。
  - `cone_shell_tip_rotation_length: float` [cm]（既定は未設定、builder が `3*cone_shell_wall_thickness` を派生；有効範囲：有限かつ `>0`）。tip facet を planar にする Q5 end-shear rotation 長 \(L_t\)；analytic wall-Jacobian margin が `>=0.25` でなければ `ConfigError`。
  - `cone_shell_base_cut: Literal["planar","wall_normal"]`（既定 `"planar"`）。base facet の cut condition。`"planar"` では tip/base rotation lengths に `L_t + L_b <= L_w` を要求する。
  - `cone_shell_base_rotation_length: float` [cm]（既定は未設定、builder が `cone_shell_tip_rotation_length` を派生；有効範囲：有限かつ `>0`）。planar base cut の Q5 end-shear rotation 長 \(L_b\)；上記 length-sum と wall-Jacobian margin gate の対象。
  - `cone_shell_farfield_target_measure: Literal["station_uniform","wall_phi"]`（既定 `"station_uniform"`）。CAVITY_CORE の axis target と EXTERIOR_CORE の box-face target の分布測度。`"station_uniform"` は station index で等分布（遠方格子幅 ~ span/N_q、列は扇状に開く）、`"wall_phi"` は v1 の Φ 線形輸送（tip-hold の細分が遠方まで保存される — 既定変更前の挙動の A/B 用）。
  - `cone_shell_outer_vac_first_factor: float`（既定 `0.8`；有効範囲：有限かつ `(0,4]`）、`cone_shell_outer_vac_layers: int`（既定 `10`；`[2,64]`）、`cone_shell_outer_vac_growth: float`（既定 `1.18`；有限かつ `(1,2]`）。OUTER_NF strip の初幅 \(h_{v,0}=\text{factor}\,h_{n,0}\)、層数、幾何成長率。
  - `cone_shell_inner_vac_first_factor: float`（既定 `1.0`；有効範囲：有限かつ `(0,4]`）、`cone_shell_inner_vac_layers: int`（既定 `10`；`[2,64]`）、`cone_shell_inner_vac_growth: float`（既定 `1.15`；有限かつ `(1,2]`）。INNER_NF strip の対応パラメータ。
  - `cone_shell_end_vac_first_factor: float`（既定 `1.0`；有効範囲：有限かつ `(0,4]`）、`cone_shell_end_vac_layers: int`（既定 `10`；`[2,64]`）、`cone_shell_end_vac_growth: float`（既定 `1.15`；有限かつ `(1,2]`）。END_NF strip の対応パラメータ。C3/C4 corner closure は outer/inner/end の layer 数がすべて等しいことを要求し、不一致は `ConfigError`。
  - `cone_shell_cavity_cells: int`（`MeshConfig` storage 既定 `0`；deck key ではなく指定不可；builder 派生後は `[6,32]`）。CAVITY_CORE Hermite column の共通 cell 数。\(K=\operatorname{clamp}(\lceil\log(1+0.35d_{\max}/h_{\rm start})/\log1.35\rceil,6,32)\) とし、short-station width が `0.5*h_start` 未満、または必要 growth が `1.70` 超なら `ConfigError`。
  - `cone_shell_exterior_cells: int`（`MeshConfig` storage 既定 `0`；deck key ではなく指定不可；builder 派生後は `[6,32]`）。EXTERIOR_CORE Hermite column の共通 cell 数。CAVITY_CORE と同じ mixed-ladder count/error 条件を outer-strip の \(h_{\rm start},d_{\max}\) に適用する。
  - `cone_shell_tip_fill_layers: int`（`MeshConfig` storage 既定 `0`；deck key ではなく指定不可；builder 派生後は `[6,48]`）。tip-side box face までの TIP_FILL WEST/EAST layer 数で、\(\operatorname{clamp}(\lceil |z_{\rm box,tip}-z_t|/(2h_t)\rceil,6,48)\)；MID はこの値より1行少ない。
  - `cone_theta_wall: float` [rad]（既定は未設定；有効範囲：有限かつ `0 < cone_theta_wall < pi`）。`logical_mesh_2d="polar_in_box"` の exterior-only cone conformance を有効化する。このキーを指定した場合、以下の cone keys の全制約と angular capacity budget が適用される。
  - `cone_tip_radius: float` [cm]（既定は未設定）。`cone_theta_wall` 指定時は有限かつ polar lock radius `explicit_nodes.back()` より大きくなければならない。
  - `cone_activation_radius: float` [cm]（既定は未設定）。`cone_theta_wall` 指定時は有限かつ `explicit_nodes.back() <= cone_activation_radius < cone_tip_radius`。
  - `cone_fine_cells_minus: int`（既定 `4`；単位なし）。`theta < cone_theta_wall` 側の fine-band cell 数で、`cone_theta_wall` 指定時は `>=1`。
  - `cone_fine_cells_plus: int`（既定 `4`；単位なし）。`theta > cone_theta_wall` 側の fine-band cell 数で、`cone_theta_wall` 指定時は `>=1`。Angular capacity は `cone_fine_cells_minus + cone_fine_cells_plus + 2*4 + 2*2 + 2*1 <= nz`（各側の最小 transition、corner separation、axis neighborhood）を必須とする。
  - `cone_angular_growth_max: float`（既定 `1.25`；有効範囲：有限かつ `1.0 < cone_angular_growth_max <= 2.0`；単位なし）。cone fine band から baseline angle spacing への遷移に用いる上限。
  - `cone_tip_style: Literal["single_line"]`（既定 `"single_line"`；単位なし）。`cone_theta_wall` 指定時、v1 は zero-thickness wall の `"single_line"` のみ受理する。
  - `polar_prefix_nr: int`（`polar_in_box` の派生値；既定 `-1`）。`len(explicit_nodes)-1` を保存し、総 radial zone 数は `polar_prefix_nr + morph_rings + collar_rings` とする。
  - `morph_rings: int`（既定 `16`；有効範囲：`>=4`）。polar prefix と rectangular collar の間の morph ring 数。
  - `collar_rings: int`（既定 `6`；有効範囲：`[2,32]`）。box 境界までの nested rectangular collar ring 数。
  - `morph_growth_max: float`（既定 `1.20`；有効範囲：`1.0 < morph_growth_max <= 2.0`）。vacuum morph 内の角度別 radial geometric-ratio 上限。
  - `spherical_polar_kappa: float`（既定 `0.5`；有効範囲：`>0`）。内核正則化係数で、\(\Delta s=S_{\max}/(N_s+\kappa)\)、\(s_{\min}=\kappa\Delta s\)。When `polar_center_treatment="tri_fan"`, this value is ignored/deprecated and the radial nodes are \(s_0=0,\ s_i=iS_{\max}/N_s\).

**S2 runtime feature gate:**

When `mesh.topology_scheme` selects `multiblock_cart_core_polar_shell` or
`multiblock_half_butterfly_5block`, the production runtime path is restricted
to a hydro-only subset with either Lagrangian motion or S4-T1-next ALE remap.
`tenryu run` rejects any deck that combines multiblock topology with any
unsupported condition in the following gate entries:

- `main.dimension != "2D_RZ"`
- `numerics.hydro.enabled = false`
- `mesh.motion not in {"lagrangian", "ale"}`
- `numerics.ale.enabled = true` | S4-T1-next T6/T7 lift | accepted for
  multiblock. With `mesh.motion="ale"`, the driver enters the scheduled
  `apply_ale` path and `apply_multiblock_csr_ale_step` enforces
  `every_n_steps`. With
  `numerics.ale.multiblock_cross_seam_rezone_enabled=false` (default), the
  production driver runs the CSR per-block Winslow path and CSR conservative
  remap while skipping cross-seam smoothing. With
  `multiblock_cross_seam_rezone_enabled=true`, the driver additionally moves
  seam-shared nodes with CSR cross-seam Winslow smoothing. Both modes avoid
  structured Winslow/local-repair code.
- `numerics.conduction.enabled = true`
- `radiation.enabled = true`
- `laser.enabled = true`
- `numerics.plic.enabled = true`
- `numerics.hydro.boundary_2d.r_outer = "state_supply"` (r-face
  state-supply remains unsupported; current state-supply is z-face only)

These restrictions reflect the S2 scope documented in NUMERICS §3.2 and
ARCHITECTURE §4.2. S4-T1-next enables the multiblock ALE remap path under
`mesh.motion="ale"`; conduction, radiation, laser, PLIC source terms, and
r-face state-supply for multiblock remain deferred to stages S3+ (seam GCL gate
plus production extension). Rejection is loud at `tenryu run` startup.
S4-T7 lifted the `trial_volume_cfl_enabled` and `mesh_quality_dt_cfl_enabled`
multiblock guards after S3 seam GCL gates G1+G2 PASS demonstrated multiblock CFL
aggregation correctness (S3-T4 ctest #654). The existing structured
trial-volume and mesh-quality pre-commit limiter bodies are admissible no-ops on
multiblock until CSR trajectory predicates are implemented.
S4-T6 lifted the outer-shell pressure-BC rejection. Multiblock dispatches
`launch_multiblock_polar_shell_pressure_forces`, which targets the polar-shell
block's outer radial ring exclusively. Core and bridge nodes receive zero
contribution from this BC by construction. The same outer radial ring is tagged
`NODE_OUTER_PHYSICAL_BOUNDARY` and uses an explicit `r_outer` motion-constraint
dispatch before seam `NODE_BOUNDARY` handling: `"fixed"` zeros both mesh-vector
components; `"reflect"` removes the spherical-normal component; `"free"` and
`"pressure"` apply no mesh-vector constraint; `"state_supply"` and unknown
values fall back to reflect semantics. This preserves G1/G3/G4 reflect gates
while allowing S4-T6 pressure-drive normal motion.

B-S2 adds no namelist keys. The five-block hydro smoke deck exercises existing
`numerics.ale.*` controls, `diagnostics.mesh_quality_min`, and
`diagnostics.production_audit.gcl` on
`mesh.topology_scheme="multiblock_half_butterfly_5block"`. Its acceptance scope
is hydro-runnable at rest with conservation/GCL/HDF5/mesh-quality checks; it is
not a seam-flux-under-gradient or strong-drive compression acceptance.

S3 keeps the default-off
`mesh.topology_scheme="multiblock_half_butterfly_trifan_cap_5block"` opt-in
for the pinned tri-fan cap. It generates the cap node coordinates and
in-memory cell/face CSR topology: one shared apex at \((R,Z)=(0,0)\),
\(N_{\rm cap}=N_c\), first-row triangular cap cells with `cell_nverts=3`,
outer cap ring node IDs shared exactly with the north/east/south fan inner
seams, and unchanged fan/shell counts. S3 T0-T5 also retarget the hydro/remap
runtime machinery listed in ARCHITECTURE §4.2.1 to the active-slot contract.
Production ICF-class dynamic certification remains a verification status, not
a namelist acceptance implied by selecting the topology.

S3 adds no namelist keys. The multiblock seam GCL, symmetry, and conservation
verification gates G1 through G4 constrain the existing
`multiblock_cart_core_polar_shell` topology. Their thresholds and empirical
γ MVP architectural floors are specified in NUMERICS §3.2.x; the runtime
dispatch contract remains ARCHITECTURE §4.2.1, this §6.4.2, and the unchanged
hydro defaults in §9.1.

**`grid.type = "graded"`**：スーパーガウシアン質量分布を持つ1D区分格子
```python
Mesh(
    grid=dict(
        type="graded",
        segments=[
            {"r_start": 0.0,    "r_end": 60e-4,  "nr": 20},
            {"r_start": 60e-4,  "r_end": 95e-4,  "nr": 70},
            {"r_start": 95e-4,  "r_end": 100e-4, "nr": 30},
            {"r_start": 100e-4, "r_end": 120e-4, "nr": 50},
            {"r_start": 120e-4, "r_end": 300e-4, "nr": 40},
        ],
        grading=dict(edge_ratio=0.05, sg_order=4, sg_sigma=0.7),
    ),
)
```
- `segments`：区間リスト（`r_start`, `r_end`, `nr` の辞書）
- `grading: dict`（任意、既定値あり）
  - `edge_ratio: float`（既定 `0.1`；有効範囲：`0 < edge_ratio < 1`）— セグメント端 / 中央のセル質量比
  - `mapping: Literal["legacy_estimated_radius","exact_measure_v2"]`（既定 `"legacy_estimated_radius"`；1D 専用（2D_RZ で `"exact_measure_v2"` は `ConfigError`）。legacy は推定中心 1/(r_est^2+r_ref^2) 薄殻近似（bitwise 保存）。`"exact_measure_v2"` は累積重み分率を厳密殻測度 \(r^d\)（球3/円筒2/平面1）で反転し、定密度セル質量を丸め誤差内で \(w_k\) 比に一致させる（原点区間含む；2026-07-26 追加、AI review k01 P0-1）。NUMERICS §3.1.0 参照）
  - `sg_order: int`（既定 `4`；有効範囲：偶数かつ `≥ 2`）— スーパーガウシアン次数
  - `sg_sigma: float`（既定 `0.7`；有効範囲：`0 < sg_sigma < 1`）— スーパーガウシアン幅
- 各セグメント内ではセル中心 \(\xi_k=(k+0.5)/n\) からスーパーガウシアン重み \(w_k\) を作り、球対称質量 \(m_k \propto r_k^2 \Delta r_k\) がその形状に従うよう \(\Delta r_k\) を決める（NUMERICS §3.1.0）
- 隣接セグメント境界では末尾セル幅と先頭セル幅の幾何平均を境界幅ターゲットとし、総セグメント長を保つよう残りセルを再スケーリングする。これによりセル幅は境界で連続
- `r_start = 0` を含む区間では、質量換算のゼロ割りを避けるため半径ガード \(r_{\mathrm{guard}} = 10^{-20}\) cm を用いる
- 最終ノードは `segments[-1].r_end` に固定し、丸め誤差でのドリフトを防ぐ
- 1D_SPH 専用。2D_RZ で指定した場合はエラー
- **graded 格子の検証規則**：
  - **単調性**：各区間は `r_start < r_end` かつ `nr > 0`
  - **境界連続性**：隣接区間の境界は `|r_end[k] - r_start[k+1]| < 1e-14 * r_max` 以内で一致すること
  - **パラメータ範囲**：`edge_ratio ∈ (0,1)`, `sg_sigma ∈ (0,1)`, `sg_order` は偶数かつ `≥ 2`
  - **可解性**：ある区間が両隣境界の目標セル幅を同時に満たせないほど少ないセル数しか持たない場合はエラー

**グリッド型の次元別対応表**：

| 次元 | 有効な grid 型 | 説明 |
|------|--------------|------|
| 1D_SPH | graded | 常に graded メッシュを用いる。`grid_type_r` は内部で `"graded"` に固定 |
| 2D_RZ | uniform | r/z とも uniform のみ対応 |

- 共通：
  - `motion: Literal["lagrangian","ale"]`（既定：1D_SPH=`"lagrangian"`、2D_RZ=`"ale"`。1D_SPH で `"ale"` を指定した場合は `ConfigError("ALE not supported for 1D_SPH")`。NUMERICS §3.1（Lagrangian）、§3.3（2D ALE）、§3.4（1D pure Lagrangian）参照）
  - `rezoning` / `numerics.ale: dict`
    - `enabled: bool`（既定 True；2D_RZ ALE 用。1D_SPH では使用しない）
    - `ale_identity_mode: bool`（既定 False；2D_RZ ALE 診断専用。True では Hydro2D の Lagrangian update 後、ALE の rezone/remap/velocity projection/boundary reapply/EOS reclosure/geometry rewrite/reference-state update をすべて skip し、ALE branch を物理的 no-op にする。`enabled=True` の ALE 実行が pure Lagrangian と一致する identity limit の検査に使う。False では完全に inert で、frozen-config には出力しない。NUMERICS §3.3.3）
    - `ale_mover_diag: bool`（既定 False；2D_RZ ALE 診断専用。True では post-hydro/post-projection の carried velocity と Lagrangian mesh motion の整合性、および identity field hashes を JSONL `[ale_mover_diag]`/`[ale_identity_diag]` として出力する。False では完全に inert で、frozen-config には出力しない。NUMERICS §3.3.3）
    - `ale_preserve_lagrangian_velocity_carry: bool`（既定 False；2D_RZ ALE CSR 診断専用。True では通常の ALE rezone/remap/projection/total-energy closure を実行した後、次ステップへ carried velocity として渡す `state.v` だけを post-hydro Lagrangian velocity に戻す。速度と内部エネルギーの整合性を意図的に破るため production では使用しない。False では完全に inert で、frozen-config には出力しない。NUMERICS §3.3.3）
    - `align_diagnostics: dict`（Stage 0 の single-block post-Lagrange host 診断。既定 `{"enabled": False, "every_n_steps": 0, "c_q_threshold": 0.2, "w_rho": 1.0, "w_p": 1.0, "floor_rel": 1.0e-12}`。rezone/remap/state を変更せず `[ale-align-diag]` を log-only 出力する。0 cadence は最初と、`t_end` または `max_steps` から判定可能な最終ステップ。正値は `step % every_n_steps == 0`。既定値ブロックは frozen-config から省略され、明示的な有効化または非既定値は全フィールドを round-trip する。NUMERICS §3.3.3）
    - `every_n_steps: int`（既定 5；有効範囲：`≥ 1`）
    - `force_rezone_every_n_steps: int`（既定 0；有効範囲：`≥ 0`；0 で無効。正値の場合、2D_RZ ALE が有効な通常 ALE driver path で `state.step % force_rezone_every_n_steps == 0` のステップに既存の forced-rezone/remap path を診断目的で起動する。単位なし [steps]）
    - `warmup_steps: int`（既定 0；有効範囲：`≥ 0`）
    - `relaxation: float`（既定 0.2；有効範囲：`[0, 1]`）
    - `spacing_ratio_threshold: float`（既定 1.5；有効範囲：`≥ 1`）
    - `quality_threshold: float`（既定 0.2；有効範囲：`(0, 1]`；2D ALE mesh quality trigger）
    - `max_iterations: int`（既定 20；有効範囲：`≥ 1`；2D Winslow 平滑化の最大反復回数。NUMERICS §3.3.3）
    - `max_displacement_fraction: float`（既定 0.5；有効範囲：`(0, 1]`；rezone 1回あたりのノード最大変位量）
    - `convergence_tol: float`（既定 `1e-6`；有効範囲：`> 0`；2D Winslow 平滑化の収束許容誤差）
    - `swept_volume_sign_fixed: bool`（既定 True；True では post-2026-05-11 の corrected ALE swept-volume sign convention を swept-volume primitive at source に適用し、donor/flux/intermediate-volume/MS2 moment/axis-band remap の符号を統一する。False では pre-fix legacy convention を bit-exact に保持する。legacy behavior requires `Numerics.profile.legacy_regression@2026-07-27` or an explicit False。旧 `donor_sign_fixed` は deprecated alias として受理し WARNING を出す。CSR total-energy remap では effective convention が `swept_volume_sign_fixed || total_energy_remap_2d_rz` となり、fixed convention と CSR hydro mass-positivity limiter が有効になる。物理的解釈は NUMERICS.md §3.3.4 "Swept-volume convention (post-2026-05-11 fix)" 参照。Experimental migration gate）
    - `remap_limiter: Literal["van_leer","minmod"]`（既定 `"van_leer"`）
    - `remap_ms_midpoint: bool`（既定 False）
    - `remap_ms_post_check: bool`（既定 False）
    - `remap_ms_post_max_iter: int`（既定 3；有効範囲：`≥ 1`）
    - `remap_ms_rescale_floor: float`（既定 0.01；有効範囲：`[0,1]`）
    - `ke_fixup: bool`（既定 True）
    - `ke_conservation_closure: bool`（既定 False；2D_RZ ALE cell-to-node velocity projection の kinetic-energy defect を内部エネルギーへ deposit して閉じる opt-in closure。1T では \(e_e\) に全量、2T では raw remapped internal-energy fraction \(e_e/(e_e+e_i)\) で \(e_e/e_i\) に分配。NUMERICS §3.3.5）
    - `ke_conservation_closure_audit: bool`（既定 False；`ke_conservation_closure=True` 時のみ、ALE kinetic-energy closure の保存恒等式・positivity floor・diagnostic mismatch を `energy/ale_closure_audit/*` history と WARNING ログへ出す opt-in 計測。False では audit kernel と history dataset は作成しない）
    - `ke_closure_redistribute_floor: bool`（既定 False；`ke_conservation_closure=True` 時のみ有効。closure deposit で負の比内部エネルギーを生じるセルの positivity deficit を、正の内部エネルギー容量を持つセルから比例配分で差し引く。容量不足時のみ未解決分を `energy/redistribution_unresolved` に記録する。NUMERICS §3.3.5）
    - `debug_per_remap_log: bool`（既定 False；2D_RZ ALE の accepted remap ごとに WARNING ログ `[ale-stats] remap_delta step=N n_applied=K dM=... dM_rel=... dE=... dE_rel=...` を出す。history/HDF5 schema は変更しない。A1 harness はこのログから accepted remap 数と per-remap conservation metrics を集計する）
    - `shock_sensor_guard_cells: int`（既定 2；有効範囲：`≥ 0`）
    - `density_jump_threshold: float`（既定 0.1；有効範囲：`≥ 0`）
    - `Te_jump_threshold: float`（既定 0.2；有効範囲：`≥ 0`）
    - `preventive_axis_guard_fraction: float`（既定 0.1；有効範囲：`≥ 0`；0 で無効。2D_RZ axis margin が初期値のこの割合を下回ると cadence 前でも ALE rezone を起動）
    - `axis_z_motion: str`（既定 `"fixed"`；許可値 `{"fixed", "winslow", "lagrangian", "lagrangian_tangential"}`；2D_RZ axisymmetric BC で axis nodes の Z motion を制御。`"fixed"`=legacy 凍結、`"winslow"`=mirrored-axis Winslow update で Z motion 許可、`"lagrangian_tangential"`=hydro step 中の axis-Z tangential motion、`"lagrangian"`=未実装。NUMERICS §3.3.5）
    - `winslow_axis_kappa: float`（既定 0.7；有効範囲：`(0, 1]`；`axis_z_motion="winslow"` 時、axis-adjacent \(i=1\) nodes の試行半径に \(r_{1,j}^{trial}\ge\kappa r_{1,j}^{Lag}\) を課す。NUMERICS §3.3.5）
    - `button_morph: dict` (default `{"enabled": False, "t_start_s": 0.0, "t_end_s": 0.0, "max_step_fraction": 0.05, "every_n_steps": 1}`; read-write shock-ahead button reorientation scheduled over the absolute simulation-time window `[t_start_s, t_end_s]`. `max_step_fraction` caps each transaction's node displacement against the local minimum incident edge, and `every_n_steps` sets the transaction cadence. Valid ranges are finite `t_start_s>=0`, finite `t_end_s>=0`, `0<max_step_fraction<=0.5`, and `every_n_steps>=1`; when enabled, `t_end_s>t_start_s`. It moves core/bridge nodes; conservative remap preserves mass/energy budgets. Runs inside the multiblock CSR ALE step; requires Numerics.ale.enabled=True and mesh motion="ale"; inert on non-button topologies (warning once). reference_barrier_enabled may stay False: the morph drives the barrier transaction engine directly and the barrier's own triggers remain governed by reference_barrier_enabled.)
    - `reference_barrier_enabled: bool`（既定 False；Phase 9 reference-barrier ALE infrastructure を有効化。production deck activation は別途検証後）
    - `reference_target: str`（既定 `"none"`；許可値 `{"none", "eulerian_initial", "spherical_equal_angle"}`；reference-barrier target mesh）
    - `reference_blend_default: float`（既定 1.0；有効範囲 `[0, 1]`；reference target への初期 blend 係数）
    - `reference_volume_floor_rel: float`（既定 1e-8；有効範囲 `≥ 0`；A0 admissibility の RZ volume relative floor。When enabled, the relative floor compares against the per-cell REFERENCE-mesh Jacobian (x_*_reference) where available (non-ratcheting); modes without reference coordinates fall back to the previous-mesh baseline.）
    - `reference_corner_j_floor_rel: float`（既定 1e-8；有効範囲 `≥ 0`；A0 admissibility の corner-J relative floor。When enabled, the relative floor compares against the per-cell REFERENCE-mesh Jacobian (x_*_reference) where available (non-ratcheting); modes without reference coordinates fall back to the previous-mesh baseline.）
    - `reference_gauss_j_floor_rel: float`（既定 1e-8；有効範囲 `≥ 0`；A0 admissibility の Gauss-J relative floor。When enabled, the relative floor compares against the per-cell REFERENCE-mesh Jacobian (x_*_reference) where available (non-ratcheting); modes without reference coordinates fall back to the previous-mesh baseline.）
    - `reference_linesearch_max_iters: int`（既定 24；有効範囲 `[0, 60]`；reference blend line search の最大反復）
    - `reference_force_engage_every_step: bool`（既定 False；debug 用。True の場合 reference-barrier ALE trigger predicates をバイパスして毎ステップ engage）
    - `reference_trigger_axis_margin_enabled: bool`（既定 True；axis-margin trigger を有効化）
    - `reference_trigger_axis_margin_threshold: float`（既定 1e-2；有効範囲 `≥ 0`；axis-margin trigger 閾値）
    - `reference_trigger_corner_j_ratio_enabled: bool`（既定 True；corner-J ratio trigger を有効化）
    - `reference_trigger_corner_j_ratio_threshold: float`（既定 0.5；有効範囲 `≥ 0`；corner-J ratio trigger 閾値）
    - `dgcl_commit_gate: bool`（既定 False；Opt-in per-cell discrete volume-closure gate on remap transactions: a candidate whose per-face swept volumes fail \(|\Sigma dV - \Delta V| \le \mathrm{rtol}\cdot V\) is discarded before commit (first offender logged).）
    - `transaction_failure_inject_point: int`（既定 0；DEBUG-ONLY failure injection for
      mesh-transaction batteries (Layer-T V10). 0 = off. K >= 1 forces the managed
      axis-band remap attempt at band width K to be discarded through the transactional
      rollback path even when the remap succeeded (logged as `[axis_band_inject]`).
      Production decks must leave this at 0; values outside [0, 10] are rejected at
      validation.）
    - `dgcl_commit_rtol: double`（既定 1.0e-11；Relative tolerance for `dgcl_commit_gate`.）
    - `remap_damage_gate_enabled: bool`（既定 False；Phase 9 remap-damage gate を ALE backtracking で有効化。NUMERICS §3.3.5）
    - `remap_damage_dmax: float`（既定 0.05；有効範囲：`≥ 0`；Phase 9 の \(D_{\rho,\max}\) 閾値。これを超える候補を reject）
    - `remap_damage_axis_eta: float`（既定 0.02；有効範囲：`≥ 0`；Phase 9 の axis-specific \(A_{0,j}\) 閾値。これを超える候補を reject）
    - `remap_damage_axis_budget_enabled: bool`（既定 False；累積 axis inflow budget tracking を有効化。`remap_damage_gate_enabled=true` が必須）
    - `remap_damage_axis_budget_factor: float`（既定 2.0；有効範囲：`≥ 0`；budget cap multiplier: \(B_{0,j}^{max}=\text{factor}\cdot M_{0,j}^{init}\)）
	    - `predictive_acceptance_enabled: bool`（default False; opt-in ALE backtracking gate that predicts one frozen-velocity hydro step from the accepted candidate. Skipped when the caller supplies `dt_hydro_used <= 0`. NUMERICS §3.3.5 Phase 9b）
	    - `predictive_acceptance_axis_floor_fraction: float`（default 0.0; valid range `[0,1]`; required fraction of the candidate minimum analytic axis margin retained by the frozen-velocity look-ahead）
	    - `predictive_acceptance_cell_vol_floor_fraction: float`（default 0.0; valid range `[0,1]`; per-cell fraction of candidate direct RZ volume retained by the frozen-velocity look-ahead）
	    - `corner_cell_aspect_protection_enabled: bool`（default True; enables the in-kernel RZ full-metric Winslow corner-cell area floor for annular state-supply z-face corners. Inactive for physical-axis RZ and for z faces without `state_supply`. NUMERICS §3.3.3）
	    - `corner_cell_aspect_eta: float`（default 0.5; valid range `[0,1]`; minimum protected corner-cell planar area as a fraction of the initial cell area）
	    - `safe_backtrack_enabled: bool`（default False; default-off ALE backtracking replacement. False preserves the legacy six-lambda schedule bit-exactly. True validates \(\lambda=0\), searches powers of two down to `2^-safe_backtrack_min_exp`, then binary-refines upward. NUMERICS §3.3.5 Phase 3.2）
    - `safe_backtrack_min_exp: int`（default 20; valid range `[0,60]`; adaptive backtracking minimum positive search value \(\lambda_{\min}=2^{-N}\)）
    - `safe_backtrack_binary_iters: int`（default 8; valid range `[0,60]`; number of upward binary-refinement iterations after the first passing power-of-two lambda）
    - `rezone_solver: str`（default `"legacy_winslow"`; allowed `{"legacy_winslow", "rz_full_metric_winslow", "m1_tmop"}`. The default preserves the legacy Cartesian/orthogonal Winslow kernel bit-exactly. `"rz_full_metric_winslow"` selects the opt-in 9-point full R-Z metric Winslow kernel with mixed derivative, R-weighted candidate blend, weighted-Laplacian singular-stencil fallback, and optional local admissibility line search. `"m1_tmop"` selects the opt-in M1 objective solver described in NUMERICS §3.3.3; M1 v1 is staged for QUAD polar-family logical meshes, and validation rejects `rectangular_rz` and `cone_shell`.）
    - `m1_gamma_align: float`（default `0.0`; valid range `>= 0`; alignment contribution in the M1 cell metric）
    - `m1_lambda_tether: float`（default `0.0`; valid range `>= 0`; M1 Lagrangian-position tether weight）
    - `m1_theta_reg: float`（default `0.0`; valid range `>= 0`; neighbor-aware same-ring spacing regularization weight）
    - `m1_sweeps: int`（default `8`; valid range `>= 1`; number of M1 damped Gauss-Newton sweeps per rezone attempt）
    - `m1_min_j_dec_rel: float`（default `0.0` = legacy strict-decrease acceptance; valid range finite `>= 0`; minimum RELATIVE objective decrease `(J_before - J_after)/|J_before|` required to accept an M1 rezone. A positive value rejects rezones below the relative-decrease threshold (tuning knob for noise-acceptance pathologies). Default 0.0 preserves legacy behavior: measured ctrl/implosion M1 gains come from thousands of small accepted decreases (mean ΔJ/J ≈ 2e-5 at N_C=64, ≈5e-5 at N_C=192 — a 1e-4 threshold would suppress 92-100% of them), so any positive default is harmful there; the once-motivating belt inversion proved to be run-to-run chaos-band noise（2026-07-29 adjudication）)
    - `m1_barrier_beta: float`（default `1e-3`; valid range `>= 0`; M1 positive-Jacobian log-barrier weight）
    - `euler_window.enabled: bool`（default False; ALE P6B. Eulerian-window remap target on the multiblock CSR orchestrator: node targets blend the Lagrangian coordinates with the frozen reference (Eulerian) grid by a C1 window weight; the blended target passes the candidate-mesh admissibility oracle and then reanchors the reference and runs the conservative remap, exactly like an accepted M1 rezone. Mutually exclusive with `rezone_solver="m1_tmop"`; requires `conservative_remap_enabled=True` and a multiblock mesh）
    - `euler_window.shape: str`（default `"rectangle"`; allowed `{"rectangle", "annulus"}`）
    - `euler_window.r0/r1/z0/z1: float`（rectangle bounds, cm; required `r1>r0`, `z1>z0` when enabled with shape rectangle）
    - `euler_window.cr/cz/rad_in/rad_out: float`（annulus center and radii, cm; required `rad_out>rad_in>=0` when enabled with shape annulus）
    - `euler_window.transition_width: float`（cm; required `> 0` when enabled; C1 smoothstep band outside the window）
    - `rezone_local_admissibility_linesearch: bool`（default False; enables per-node local incident-cell admissibility line search inside `rezone_solver="rz_full_metric_winslow"`. Inactive for `legacy_winslow`）
    - `rezone_local_j_floor_rel: float`（default `1e-8`; valid range `>= 0`; relative Gauss-J floor used by the local full-metric incident-cell admissibility check）
    - `rezone_local_linesearch_max_halves: int`（default 8; valid range `[0,32]`; maximum number of binary halvings in the full-metric local line search）
    - `reject_zero_gauss_j: bool`（default False; False preserves legacy Gauss-J tangle behavior bit-exactly. True rejects any ALE backtracking trial mesh whose 2x2 Gauss-point Jacobian satisfies \(J \le \max(k_J, \texttt{zero\_gauss\_j\_floor\_rel}\,J_{\max,\mathrm{eff}})\). NUMERICS §3.3.2）
    - `zero_gauss_j_floor_rel: float`（default `1e-8`; valid range `(0,∞)`; relative floor for `reject_zero_gauss_j` near-zero Gauss-J detection）
    - `lambda_sweep_diagnostic_enabled: bool`（default False; default-off ALE backtrack rejection diagnostic. When enabled and the configured target cell is the first Gauss/corner-J rejection, TENRYU logs a λ table and emits optional `/diagnostics/ale_lambda_sweep/v1/` HDF5 sidecar data. False is bit-exact inert. NUMERICS §3.3.5）
    - `lambda_sweep_target_cell_c: int`（default -1; target cell linear index. `>=0` selects this cell and takes precedence over `(i,j)`）
    - `lambda_sweep_target_cell_i: int`, `lambda_sweep_target_cell_j: int`（default -1; alternative target coordinates. Both must be set or both left at -1）
    - `lambda_sweep_max_exp: int`（default 20; valid range `[0,1022]`; includes \(2^{-N}\) in the diagnostic λ list）
    - `corner_jacobian_post_tangle_enabled: bool`（default True; ALE backtracking の post-rezone admissibility で active 2D cell の signed corner-J を評価し、任意 corner が非正または非有限なら候補 mesh を reject する。False では legacy Gauss-point post-tangle のみ。NUMERICS §3.3.5 Phase 9a）
    - `corner_post_tangle_strict_floor_enabled: bool`（default False; `corner_jacobian_post_tangle_enabled=True` 時の opt-in strict floor。True では active 2D cell ごとに \(J_{\mathrm{floor},c}=\max(k_J, corner\_jacobian\_floor\_eps \max_k |J_{c,k}^{corner}|)\) を作り、positive-but-below-floor corner も ALE candidate reject とする。False では legacy \(J>0\) post-tangle predicate を保持する。NUMERICS §3.3.5 Phase 9a）
    - `local_boundary_repair_enabled: bool`（default False; Phase 9c emergency fallback。corner-J post-tangle で全 backtrack lambda が reject された後、top/bottom z boundary または outer-r boundary の非corner node に限り、boundary line を保った 1D tangential repositioning を試行する。accepted candidate は通常の damage/remap/predictive gates を通す。NUMERICS §3.3.5 Phase 9c）
    - `multi_node_boundary_repair_enabled: bool`（default False; `local_boundary_repair_enabled=True` 時の Phase 9d opt-in。1D boundary repair の feasible interval が空の場合、apex + 1-ring nodes の coordinated motion を linearized active-set solve で試行し、global Gauss/corner-J validation 後に通常の acceptance gates へ戻す。NUMERICS §3.3.5 Phase 9d）
    - `multi_node_interior_repair_enabled: bool` (default False; driver-requested interior patch projection opt-in. It is used only when the retry selector explicitly dispatches `InteriorMultiNodeProjection`; otherwise interior CD failures first use the CD-local Winslow mode. NUMERICS §3.3.5 Phase 9f)
    - `axis_variational_projection_enabled: bool` (default False; Stage 23 Wave 1 axis-band escalation opt-in. When AxisSpinePlusLocal fails, the retry ladder may dispatch `AxisVariationalProjection`, a deterministic projection-style half-space feasibility operator, before `InteriorMultiNodeProjection` and `FullWinslow`. NUMERICS_03 §3.3.5a; ARCHITECTURE_06-10 §6.1)
    - `multiblock_cross_seam_rezone_enabled: bool` (default False; S4-T1-next T5b/T6 opt-in for multiblock CSR Winslow smoothing that moves seam-shared nodes using face-adjacent neighbors from all incident seam cells. False still enables the multiblock ALE production path but limits rezone to T5a per-block smoothing before CSR conservative remap. NUMERICS §3.3.3)
    - `multiblock_scaled_reference_enabled: bool` (default False; Phase 1 opt-in for multiblock conservative-reference remap and reference-barrier targets. When True, the γ-MVP target is rebuilt as \(X_i^R(t)=\alpha(t)X_i^0\), where \(\alpha(t)\) is the mean current outer-ring radius divided by initial `Mesh.spherical_polar_s_max`; target cell volumes scale by \(\alpha^3\). False preserves the static IC reference target. NUMERICS §3.3.4)
    - `multiblock_differential_reference_enabled: bool` (default False; opt-in multiblock 2D_RZ conservative-remap reference. When True, and `Mesh.topology_scheme` is multiblock, the remap call installs the Lagrangian-close differential converging reference before the γ-MVP scaled-reference fallback. It conflicts with `multiblock_scaled_reference_enabled=True`. Units: dimensionless boolean. NUMERICS §3.3.4)
    - `multiblock_differential_reference_band_count: int` (default 64; valid range `[8,4096]`; number of radial ξ bands used to compute robust median Lagrangian projected radii. Units: band count. NUMERICS §3.3.4)
    - `multiblock_differential_reference_smoothing_g0: float` (default 0.03; valid range `(0,1]`; dimensionless shock-aware smoothing gradient scale in the ξ-band correction weights. NUMERICS §3.3.4)
    - `multiblock_differential_reference_nu: float` (default 0.10; valid range `(0,1]`; dimensionless limiter fraction used for ξ-band correction caps and per-node displacement caps. NUMERICS §3.3.4)
    - `multiblock_differential_reference_eps_v: float` (default 0.03; valid range `(0,1]`; dimensionless relative edge-spacing floor used by the band monotonicity/admissibility check. NUMERICS §3.3.4)
    - `multiblock_differential_reference_s_cap_min_rel: float` (default `1e-3`; valid range `(0,1]`; dimensionless center/cap floor relative to the initial maximum radius \(s^0_{\max}\). NUMERICS §3.3.4)
    - `multiblock_differential_reference_xi_seam_tol: float` (default `1e-9`; valid range `(0,1e-3]`; dimensionless tolerance for validating equal ξ on seam-equivalent multiblock nodes. NUMERICS §3.3.4)
    - `multiblock_differential_reference_sigma_warn_floor: float` (default 0.5; valid range `(0,1]`; dimensionless warning threshold reserved for differential-reference σ-health diagnostics. Current reference acceptance is still controlled by the CSR line search and `reference_*_floor_rel` settings. NUMERICS §3.3.4)
    - `multiblock_lagrangian_bulk_center_patch_reference_enabled: bool` (default False; opt-in multiblock 2D_RZ conservative-remap reference mode. When True, it conflicts with `multiblock_scaled_reference_enabled=True` and `multiblock_differential_reference_enabled=True`, requires `conservative_remap_enabled=True` and `conservative_remap_target="reference"`, and keeps the bulk reference Lagrangian while permitting only a center/quality patch to rezone. CP1 is config/docs only; False is behavior-inert. NUMERICS §3.3.4)
    - `multiblock_center_patch_ring_max: int` (default 4; valid range `>=0`; permanent center-patch cell-ring limit / tri-fan cap inclusion for the multiblock Lagrangian-bulk center-patch reference. NUMERICS §3.3.4)
    - `multiblock_center_patch_xi_center: float` (default 0.0; dimensionless ξ cutoff for adding cells with `ref_xi_cell < xi_center` to the permanent patch; 0 selects ring-only patching. NUMERICS §3.3.4)
    - `multiblock_center_patch_halo_layers: int` (default 2; valid range `>=0`; face-adjacency halo layers dilated around center/quality-triggered patch cells. NUMERICS §3.3.4)
    - `multiblock_center_patch_vol_on`, `multiblock_center_patch_vol_off: float` (defaults 0.05, 0.10; valid range `(0,1)` with `on < off`; hysteresis thresholds for adding/removing quality-patch cells based on \(V_c/V_c^0\). NUMERICS §3.3.4)
    - `multiblock_center_patch_cornerj_on`, `multiblock_center_patch_cornerj_off: float` (defaults 0.03, 0.08; valid range `(0,1)` with `on < off`; hysteresis thresholds for signed corner-J quality patching. NUMERICS §3.3.4)
    - `multiblock_center_patch_gaussj_on`, `multiblock_center_patch_gaussj_off: float` (defaults 0.03, 0.08; valid range `(0,1)` with `on < off`; hysteresis thresholds for 2x2 Gauss-J quality patching. NUMERICS §3.3.4)
    - `multiblock_path_admissibility_enabled: bool` (default False; opt-in for multiblock Hydro2D path-admissibility rejection. When True, the predictor and corrector Lagrangian mesh paths from \(X^n\) to their candidate positions are checked analytically before position commit for corner-J, 2x2 Gauss-J, and signed-area floor crossings. Failure requests a driver full-step retry with smaller dt. NUMERICS §3.2.13f)
    - `path_admissibility_floor: float` (default `0.01`; valid range `> 0`; relative floor \(J_\mathrm{floor}=path\_admissibility\_floor\,|J(0)|\) used by the multiblock path-admissibility predicate. When enabled, the relative floor compares against the per-cell REFERENCE-mesh Jacobian (x_*_reference) where available (non-ratcheting); modes without reference coordinates fall back to the previous-mesh baseline.)
    - `dt_rejection_factor: float` (default `0.5`; valid range `(0,1)`; safety multiplier in `suggested_dt = dt_rejection_factor * lambda_first * dt` when `multiblock_path_admissibility_enabled=True` rejects a hydro path)
    - `max_dt_rejections: int` (default 8; valid range `>= 1`; maximum full-step dt reductions for the Phase 2 multiblock path-admissibility reason before the driver exits through the retry-exhaustion diagnostic)
    - `axis_band_managed_remap_enabled: bool` (default False; Stage 24 Wave 2 managed axis-band controller opt-in. False preserves the legacy ALE/retry path.)
    - `axis_band_managed_remap_width: int` (default 3; valid range `>= 1`; initial row-K band width for controller evaluation and retry-time repair.)
    - `axis_band_managed_remap_max_width: int` (default 6; valid range `>= axis_band_managed_remap_width` and `<= 32`; maximum K fallback width.)
    - `axis_band_managed_remap_every_hydro_half_step: bool` (default True; enables post-Strang-hydro half-step controller checks.)
    - `axis_band_managed_remap_margin_trigger: float` (default `1e-4`; valid range `> 0`; minimum healthy row-K margin for K selection.)
    - `axis_band_managed_remap_equal_volume: bool` (default True; selects equal-volume axis-band target geometry.)
    - `axis_band_managed_remap_include_radiation_groups: bool` (default True; includes radiation group energy fields in axis-band snapshot/remap bookkeeping.)
    - `axis_rezone_enabled: bool` (default False; enables the repeatable target-only full-axis ALE rezone only for `mesh.topology_scheme="multiblock_half_butterfly_5block"` with five multiblock blocks. False is inert for all topologies.)
    - `axis_rezone_trigger_edge_fraction: float` (default `0.1`; units: dimensionless ratio; valid range `(0,1]`; fires when the current minimum full-axis edge length falls below this fraction of the first activated reference value.)
    - `axis_rezone_trigger_min_altitude_fraction: float` (default `0.1`; units: dimensionless ratio; valid range `(0,1]`; fires when the current minimum incident-cell quad altitude along the full-axis chain falls below this fraction of the first activated reference value.)
    - `axis_rezone_eta_floor: float` (default `1e-2`; units: dimensionless ratio; valid range `(0,1)`; spacing-floor fraction passed to the target-only weighted PAVA projection.)
    - `core_freeze_enabled: bool` (default False; opt-in gas-core pure-Lagrangian ALE target freeze. False is inert and emits no `[core_freeze]` diagnostics. NUMERICS §3.3.3)
    - `core_freeze_source: str` (default `"gas_tracer"`; allowed values in S1: `"gas_tracer"` only. Reserved source names such as `"block_role"` and `"radius"` are not accepted yet.)
    - `core_freeze_tracer_cut: float` (default `0.5`; units: dimensionless gas tracer mass fraction; valid range `[0,1]`; cells with `gas_tracer_Y >= cut` seed the frozen region.)
    - `core_freeze_halo_layers: int` (default `1`; valid range `>= 0`; number of face-adjacency dilation layers applied to the seed frozen cell set.)
    - `core_freeze_apply_to_axis_rezone: bool` (default True; when False, the full-axis target-only rezone ignores the core-freeze mask while the other ALE rezone paths remain gated.)
    - `core_freeze_skip_velocity_projection: bool` (default True; active only when `core_freeze_enabled=True`; frozen nodes keep their post-hydro Lagrangian velocity instead of being overwritten by the CSR cell-to-node velocity projection. Set False only for S1/S1b A/B diagnostics.)
    - `emergency_cell_deactivation_enabled: bool`（default False; `local_boundary_repair_enabled=True` and `multi_node_boundary_repair_enabled=True` 時の Phase 9e opt-in。全 backtrack lambda が corner-J gate で reject され、1D repair と 1-ring multi-node repair がともに infeasible の場合のみ、failing active cell の mass/internal energy を隣接 active cell へ移し、failing cell を `cell_is_void=1, hydro_active=0` として以後の active-cell geometry checks/hydro から外す。NUMERICS §3.3.5 Phase 9e）
    - `axis_repair_mode: str`（既定 `"full_winslow"`；許可値 `{"full_winslow", "axis_spine_only", "axis_z_winslow"}`。`"axis_spine_only"` は Phase 10 minimal repair、`"axis_z_winslow"` は Phase 8a 予約値で現状は WARNING の上 `"full_winslow"` に fallback）
    - `remap_scheme: str`（既定 `"legacy_split"`；許可値 `{"legacy_split", "ms2_moments"}`。`"ms2_moments"` は Phase 11 MS2 second-order monotonic conservative remap）
    - `remap_ms2_limiter: str`（既定 `"van_leer"`；許可値 `{"van_leer", "barth_jespersen"}`。`remap_scheme="ms2_moments"` のときのみ使用）
    - `conservative_remap_enabled: bool`（既定 False；2D_RZ shock-frame 用の conservative reference remap を Hydro Lagrangian update 直後に実行する。`polar_center_treatment="tri_fan"` では Stage 3 first-order remap がサポート対象。False では従来 ALE path と bit-exact）
    - `conservative_remap_target: str`（既定 `"reference"`；許可値は現状 `"reference"` のみ。target mesh は通常 IC reference mesh `State.x_r_reference/x_z_reference`。`Numerics.ale.multiblock_lagrangian_bulk_center_patch_reference_enabled=True` かつ multiblock の場合は bulk を Lagrangian reference として freeze し center/quality patch のみ rezone する reference mode を選ぶ（CP1 では config/docs のみ）。そうでなく `Numerics.ale.multiblock_differential_reference_enabled=True` かつ multiblock の場合は Lagrangian-close differential converging reference をインストールし、accepted reference coordinates から `State.cell_vol_initial` を再計算する。そうでなく `Numerics.ale.multiblock_scaled_reference_enabled=True` かつ multiblock の場合は current outer ring から \(\alpha(t)\) を計算し、\(\alpha(t)X^0\) と \(\alpha^3 V^0\) を reference target とする）
    - `conservative_remap_radiation_enabled: bool`（既定 True；`conservative_remap_enabled=True` 時、radiation group energy density \(E_g\) も \(E_gV\) として保存的に remap）
    - `conservative_remap_order: str`（既定 `"first_order_donor"`；許可値 `{"first_order_donor", "second_order_van_leer"}`。`"first_order_donor"` は PR B donor-cell flux を bit-exact に保持し、`"second_order_van_leer"` は PR C の Van Leer slope + Barth-Jespersen bounded face reconstruction を使用。tri_fan の fan-touching faces は Stage 3 では first-order donor に fallback し、pure quad-to-quad faces のみ second-order reconstruction を使う）
	    - `conservative_remap_lagrangian_bulk_enabled: bool`（既定 False；single-block `Mesh.logical_mesh_2d="spherical_polar_halfplane"`, `Mesh.polar_center_treatment="tri_fan"`, `Numerics.hydro.boundary_2d.r_outer="pressure"`, `conservative_remap_enabled=True`, `conservative_remap_target="reference"` の opt-in。True では local center band の tracking/rezone target を保ち、bulk node rings \(i>M\) は post-Lagrange coordinates を reference として保存的 remap の bulk swept volume をゼロにする。False では既存 reference target を変更しない。NUMERICS §3.3.4）
	    - `conservative_remap_lagrangian_bulk_center_node_ring_max: int`（既定 4；有効範囲 `>= 0`；上記 opt-in で tracking/rezone target を保つ local center node ring 上限 \(M\)。bulk rings は \(i>M\)。bulk corner-J warning が出る場合は、この値を増やして劣化セルを center band に含める。NUMERICS §3.3.4）
	    - `central_pseudo_core_enabled: bool` (default False; Exp1 default-off virtual agglomeration for the 2D_RZ five-block multiblock center topology. It builds a fixed ring-conforming central member set from structured block/ring indices using initial node radii, treats those cells as one pseudo control volume for hydro operators, and leaves CSR topology unchanged. No face-adjacent halo is added. Requires compatible pressure-force/work mode, rejects per-material conservation, and has no de-agglomeration in Exp1. NUMERICS §3.3.5)
	    - `central_pseudo_core_s_c: float` (default 0.0 cm; required `>0` when `central_pseudo_core_enabled=True`. This is the cgs cutoff for selecting the largest complete ring set whose maximum initial node radius is `<= s_c`; the pseudo-core volume is the sum of member RZ cell volumes, not a prescribed analytic sphere.)
	    - `central_pseudo_core_ring_absorption_enabled: bool`（既定 False；中央 macro cell の動的完全リング吸収。吸収 DEPTH は cap リング → fan 層（3 fan の union）→ 純ガス polar-shell 行の ladder。4 トリガ（volume / center-patch failure / dt-floor rescue / macro-boundary rescue）が単一の実行経路を共有する。NUMERICS Exp1 吸収段落参照。歴史的 env `TENRYU_I1B_RING_ABSORB` が SET されている場合は env が優先される。）
	    - `central_pseudo_core_ring_absorption_tau: float`（既定 0.05；有効範囲 (0,1)；volume トリガ閾値 — 監視中ユニットの最小 RZ 体積が arming 時最小値の τ 倍を割ると吸収。env override: `TENRYU_I1B_RING_ABSORB_TAU`）
	    - `central_pseudo_core_ring_absorption_max_rings: int`（既定 0 = 材料/topology ガードまで無制限；`>0` で任意の深さ上限。env override: `TENRYU_I1B_RING_ABSORB_MAX_RINGS`）
	    - `central_pseudo_core_ring_absorption_gas_tracer_min: float`（既定 0.99；有効範囲 (0,1)；材料ガード — shell 行の質量加重ガス tracer 分率 Σm·Y/Σm がこの床を満たす連続 prefix のみ吸収可能。env override: `TENRYU_I1B_RING_ABSORB_GAS_TRACER_MIN`）
		    - `central_pseudo_core_ring_absorption_gas_tracer_cell_min: float`（既定 0.5；有効範囲 (0,1)；per-cell ハード下限 — 最悪セルが過半 shell 材料の行は吸収しない。env override: `TENRYU_I1B_RING_ABSORB_GAS_TRACER_CELL_MIN`）
	    - `conv_rezone_enabled: bool`（既定 False；収束追随 global rezone（認証 I1-B スタック構成員、NUMERICS §13 反証格子）。env override: `TENRYU_I1B_CONV_REZONE`。companion 調整 env 群（`_EVERY`/`_START_T`/`_COOLDOWN_T` 等）は experimental knob として env のまま — C++ 既定が認証値）
	    - `central_pseudo_core_core1d_enabled: bool`（既定 False；成層 1D 球対称 Lagrangian core サブモデル（NUMERICS §13）の master switch。macro CV の動力学/計量権限を pooled 0-D 閉包から置換する。env override: `TENRYU_I1B_CORE_1D_SUBMODEL`）
	    - `central_pseudo_core_core1d_build_shells: int`（既定 48；有効範囲 [4,4096]；初期 build 時の目標殻数。env override: `TENRYU_I1B_CORE_1D_BUILD_SHELLS`）
	    - `central_pseudo_core_core1d_split_append: int`（既定 0 = off；有効範囲 [0,1024]；巨大 append の等質量サブ殻分割数上限（NUMERICS §13.2）。env override: `TENRYU_I1B_CORE_1D_SPLIT_APPEND`）
	    - `central_pseudo_core_core1d_av_c1: float`（既定 0.5；`>0`；サブモデル VNR 線形 AV 係数。env override: `TENRYU_I1B_CORE_1D_AV_C1`）
	    - `central_pseudo_core_core1d_av_c2: float`（既定 4.0；`>0`；サブモデル VNR 二次 AV 係数。env override: `TENRYU_I1B_CORE_1D_AV_C2`）
	    - `central_pseudo_core_core1d_cfl: float`（既定 0.25；有効範囲 (0,1)；サブモデル subcycle CFL。env override: `TENRYU_I1B_CORE_1D_CFL`）
	    - `central_pseudo_core_core1d_piston_cap: float`（既定 10.0；`>0`；運動学 piston 速度上限の信号速度倍率。env override: `TENRYU_I1B_CORE_1D_PISTON_CAP`）
	    - `central_pseudo_core_core1d_max_substeps: int`（既定 20000；`>=1`；2D step あたり subcycle 上限。env override: `TENRYU_I1B_CORE_1D_MAX_SUBSTEPS`）
	    - `central_pseudo_core_spherical_absorb_gasfront: bool`（既定 False；球対称吸収スケジュールの gas-front mode。env override: `TENRYU_I1B_SPHERICAL_ABSORB_GASFRONT`）
	    - `central_pseudo_core_spherical_absorb_alpha: float`（既定 0.0 = off；0 以外の有効範囲 (0,1)；球対称吸収スケジュールの半径収束トリガ。env override: `TENRYU_I1B_SPHERICAL_ABSORB_ALPHA`）
	    - `central_pseudo_core_spherical_absorb_pjump: float`（既定 0.0 = off；0 以外の有効範囲 `>1`；球対称吸収スケジュールの圧力 jump トリガ。env override: `TENRYU_I1B_SPHERICAL_ABSORB_PJUMP`）
	    - `central_pseudo_core_mixed_absorb_enabled: bool`（既定 False；gas prefix 枯渇後の emergency mixed-row absorption。env override: `TENRYU_I1B_MIXED_ABSORB`）
	    - `central_pseudo_core_absorb_watch_rows: int`（既定 1；有効範囲 [1,8]；ring absorption volume trigger が監視する active unit 行数。env override: `TENRYU_I1B_ABSORB_WATCH_ROWS`）
	    - `central_pseudo_core_terminal_absorb_enabled: bool`（既定 False；terminal endgame absorption master switch。env override: `TENRYU_I1B_TERMINAL_ABSORB`）
	    - `central_pseudo_core_terminal_rebound_factor: float`（既定 1.02；有効範囲 `>1`；terminal absorption の gas-volume rebound 判定倍率。env override: `TENRYU_I1B_TERMINAL_REBOUND_FACTOR`）
	    - `central_pseudo_core_terminal_tail_dt_s: float`（既定 1.0e-12 s；有効範囲 `>0`；terminal core1d tail integration chunk dt。env override: `TENRYU_I1B_TERMINAL_TAIL_DT`）
	    - `remap_mass_closure_reject_tol: float`（既定 0.0 = off；有効範囲 `>=0`；remap total-mass closure violation の full-step retry rejection threshold。default-on regression sweep 完了までは既定 off；certified I1-B deck は 1e-8 を明示する。env override: `TENRYU_I1B_REMAP_CLOSURE_REJECT_TOL`）
	    - `rezone_closure_cooldown_steps: int`（既定 50；有効範囲 `>=1`；remap closure rejection 後に rezone initiator が stand down する cooldown steps。env override: `TENRYU_I1B_REZONE_CLOSURE_COOLDOWN_STEPS`）
	    - `csr_optionb_coherent_enabled: bool`（既定 False；Option-B coherent-lite bookkeeping/projection path。env override: `TENRYU_I1B_OPTIONB_COHERENT`）
	    - `csr_optionb_velocity_remap_enabled: bool`（既定 False；CSR Option-B velocity remap authority。env override: `TENRYU_I1B_OPTIONB_VELREMAP`）
	    - `pole_axis_bbsw_enabled: bool`（既定 False；pole-axis BBSW closure。env override: `TENRYU_I1B_POLE_AXIS_BBSW`）
	    - `axis_contact_guard_enabled: bool`（既定 False；axis contact position guard。env override: `TENRYU_I1B_AXIS_CONTACT_GUARD`）
	    - `mass_floor_absorb_enabled: bool`（既定 False；remap mass-floor sentinel absorption guard。env override: `TENRYU_I1B_MASS_FLOOR_ABSORB`）
	    - `interior_patch_remap_enabled: bool`（既定 False；Ring7 interior-patch remap generalization。env override: `TENRYU_I1B_INTERIOR_PATCH_REMAP`）
	    - runtime env `TENRYU_I1B_POLAR_SHELL_ANGULAR_DEREFINE`（既定 unset/OFF；SET non-empty/non-zero で I1-B Stage-1 dynamic polar-shell angular de-refinement を有効化。POLAR_SHELL の両 pole band に dyadic 1-radial-row angular macro を構築し、accepted geometry updates 後に \(r\Delta\theta\ge\chi\Delta r\) 基準で monotonic に span を拡張し、covered fine children を hydro/remap/CFL/path から除外する。ON は `TENRYU_I1B_PATH_PREDICATE_HARDEN` 相当の hardened path predicate も有効化する。namelist/frozen-config/checkpoint/HDF5 schema には状態を追加しない。NUMERICS §3.3.5 I1-B Stage 1）
		    - `pole_sector_rezone_enabled: bool`（既定 False；pole-sector 角度 rezone — axis rezone 発火時に、各 pole の最初の m_theta 本の off-axis ノード列を、行の a=m_theta 列に anchor した参照角度 ladder へ球半径保存で retarget する（圧縮 AV が見えない接線方向 null mode の de-shearing）。target は axis-chain rezone と同一の transactional guard + conservative remap に乗る。**注意: 既定 off であり、nr16 Case B の判別 run（commit 5df5635e）で経験的に NET-NEGATIVE** — pole 近傍の remap 活動が純ガス行 tracer budget（ring-absorption の生存資源）を de-shearing の利得より速く消費する。利用可能な robustness lever であって推奨設定ではない。歴史的 env `TENRYU_I1B_POLE_REZONE` が SET（非空）の場合は env が優先される。NUMERICS の *Pole-sector angular rezone* 段落参照）
	    - `pole_sector_rezone_m_theta: int`（既定 4；有効範囲 `>= 2`；pole ごとに retarget する off-axis ノード列数。runtime はさらに ntheta/4 で cap。env override: `TENRYU_I1B_POLE_REZONE_M`）
	    - `pole_sector_rezone_lambda: float`（既定 0.5；有効範囲 (0,1]；fire ごとの参照 ladder への blend 率。env override: `TENRYU_I1B_POLE_REZONE_LAMBDA`）
	    - `pole_sector_rezone_mode: str`（既定 "uniform"；許容 {"uniform","equal_mu"}；参照 ladder — "uniform" は初期 uniform-theta zoning の復元で健全 mesh 上では identity、"equal_mu" は axisymmetric-volume-fraction ladder（初期 zoning からの大規模 restructuring で、連続適用時の remap churn が大きい）。env override: `TENRYU_I1B_POLE_REZONE_MODE`）
	    - `pole_sector_rezone_deadband_frac: float`（既定 0.0 = deadband なし；有効範囲 [0,1)；anchor 角 delta_M 比の per-node deadband — 参照角との差がこの帯内のノードは触らず、健全な pole sector では identity target になる。既定 0.0 は歴史的 env-unset 挙動と一致（5df5635e コミットメッセージの "default 0.05" は同コミットの deadband run で用いた推奨量であり、env-unset 時の値ではない）。env override: `TENRYU_I1B_POLE_REZONE_DEADBAND_FRAC`）
	    - 旧 1D 専用 ALE キー群は ALE-FIX-1 で削除済み。新しい 1D solution-adaptive ALE は `Numerics.ale1d` で制御し、既定では無効。
  - `floors: dict`
    - `rho_floor_gcc: float`（既定 `1e-10`；密度下限 [g/cm³]。有効範囲：`> 0`。NUMERICS §1.1.7参照）
    - `Te_floor_eV: float`（既定 `1e-3`；電子温度下限 [eV]。有効範囲：`> 0`。唯一の権威ソース。`Numerics.positivity.Te_min_eV`（非推奨エイリアス）も指定された場合は `max(Te_floor_eV, Te_min_eV)` を適用）
    - `Ti_floor_eV: float`（既定 `1e-3`；イオン温度下限 [eV]。有効範囲：`> 0`）

#### 6.4.3 Materials(...)
Materialsは `Material(...)` の配列を受け取る。

- `materials: list[Material]`
- `mixture: dict`
  - `fractions: Literal["volume","mass"]`（既定 `"volume"`；Geometry の `volfrac` 関数が返す値の解釈を制御。`"volume"` = 体積分率、`"mass"` = 質量分率。NUMERICS §1.1.5参照）
  - `opacity_mix_rule: Literal["linear_mass","harmonic_mass_R","max"]`（既定 `"linear_mass"`）
    - **混合則の数式**（NUMERICS §1.1.5 (c) に対応）：
      - `"linear_mass"`（Planck用）：\(\kappa_P = \sum_\alpha f_{m,\alpha}\,\kappa_{P,\alpha}\)
      - `"harmonic_mass_R"`（Rosseland用）：\(1/\kappa_R = \sum_\alpha f_{m,\alpha}/\kappa_{R,\alpha}\)
      - `"max"`：\(\kappa = \max_\alpha(\kappa_\alpha)\)
    - ここで \(f_{m,\alpha}\) は材料 \(\alpha\) の質量分率、\(\kappa\) は質量不透明度 [cm²/g]
    - **Non-LTE 混合則**（M17: `opacity.model="table_nlte"` 時）：\(\kappa^{PA}\) と \(\kappa^{PE}\) はそれぞれ `linear_mass` を適用（吸収・放射とも additive）、\(\kappa_R\) は `harmonic_mass_R` を適用（ARCHITECTURE §4.3.3 参照）
  - `eos_mix_rule: Literal["mass_weighted_same_state"]`（既定：全材料が同一(ρ,Te,Ti)を共有し、質量分率でe,Pを加重平均。NUMERICS §1.1.5 (c) 参照）
    - `"mass_weighted_same_state"`：\(P_{mix} = \sum_\alpha f_{m,\alpha}\,P_\alpha(\rho, T)\)。v1.0唯一の選択肢。`"pressure_equilibrium"` は将来版で追加予定
  - **多材料エネルギー混合則**：\(e_{k,mix} = \sum_\alpha f_{m,\alpha}\,e_{k,\alpha}(\rho, T_k)\)、\(C_{v,mix} = \sum_\alpha f_{m,\alpha}\,C_{v,\alpha}\)（\(k = e, i\)）
  - **伝導/ソース結合のセル実効量（固定仕様、追加キーなし）**（NUMERICS §1.1.5a）：
    - \(A_{eff} = (\sum_\alpha f_\alpha/A_\alpha)^{-1}\)（体積分率調和平均）
    - \(\gamma_{eff} = \sum_\alpha f_\alpha \gamma_\alpha\)（体積分率線形平均）
    - \(n_e = \rho \bar{Z}/(A_{eff}m_p)\)
    - \(c_{v,e} = \bar{Z}k_B/(A_{eff}m_p(\gamma_{eff}-1))\)、\(c_{v,i} = k_B/(A_{eff}m_p(\gamma_{eff}-1))\)
    - `n_mat == 1` では \(A_{eff}=A_0,\ \gamma_{eff}=\gamma_0\)
- `zbar: dict`（平均電離度 \(\bar{Z}\) モデル。NUMERICS §1.1.4参照）
  - `model: Literal["fixed","thomas_fermi","tabular"]`（既定 `"fixed"`）
  - `"fixed"`：\(\bar{Z} = Z\)（各Materialの `Z` パラメータを使用。完全電離仮定。
    多材料時は材料別 Z̄_α = Z_α を混合平均して Z̄_eff を算出、CUDA_KERNELS §7.6 参照）
  - `"thomas_fermi"`：More et al. (1988) フィッティング公式により \(\bar{Z}(\rho, T_e)\) を動的計算
  - `"tabular"`：IONMIXテーブルから \(\bar{Z}(\rho, T_e)\) を補間取得。IONMIXソースの優先順位: (1) `eos.model="ionmix"` の `eos.file`、(2) `opacity.model` が `"table_nlte"` または `"ionmix"` の `opacity.file`、(3) `Materials.zbar.table_file`
  - `table_file: Optional[str]`（`model="tabular"` 時のフォールバック用IONMIXファイルパス。`eos.model="ionmix"` や `opacity.model="table_nlte"/"ionmix"` が設定されていない場合に使用。NUMERICS §1.1.5 (b) 参照）
- `void_config: dict`（void材料のデフォルト物理量。`is_void=True` の材料を持つセルに適用）
  - `rho: float`（既定 `1e-10`；void セル密度 [g/cm³]）
  - `Te: float`（既定 `1e-3`；void セル電子温度 [eV]）
  - `Ti: float`（既定 `1e-3`；void セルイオン温度 [eV]）

`Material(...)`：
- `name: str`（例 `"CH"`, `"DD"`）。Material 名は一意でなければならない。重複時は `ConfigError("Duplicate material name: {name}")` を送出。
- `A: float`（平均原子量 [amu]；有効範囲：`> 0`。例：DT=2.5、CH=6.5）
- `Z: float`（平均原子番号 or 代表値 [無次元]；有効範囲：`> 0`、`Z ≤ A` 推奨。`zbar.model="fixed"` の場合にそのまま \(\bar{Z}\) として使用。`zbar.model="thomas_fermi"` または `"tabular"` の場合は初期推定値のみ）
- `is_void: bool`（既定 `False`；`True` の場合、この材料は真空（void）として扱われる。EOS/opacity テーブルの読み込みをスキップし、`eos.model="ideal_gas"`, `opacity.model="constant"`, `kappa_a=0`, `kappa_s=0`, `A=1`, `Z=0` がデフォルトで適用される。void材料は `materials[0]`（先頭）に配置不可（`ConfigError`）。少なくとも1つの非void材料が必須。§6.4.4 `cell_is_void` 参照）
- `eos: dict`
  - `model: Literal["sesame","ionmix","tmat","ideal_gas","power_law_te"]`（既定 `"ideal_gas"`）
    - `"sesame"`: xSESAME ASCII table EOS
    - `"ionmix"`: IONMIX `.cn4` table EOS
    - `"tmat"`: TMAT-H5 `/eos` table EOS。**任意ブロック `/ionization`（2026-07-30 追加）**: 元素別電離段分率 `fields/fractions [nS,nD,nT]` + `stage_element`/`stage_charge [nS]` + `grid/{ni_cm3,temperature_eV}`（`/eos` 格子と一致必須、相対 1e-12）。元素ごと各格子点で |Σf−1|≤0.2 を要求し読み込み時に再正規化（超過は `TMAT_IONIZATION_NORMALIZATION`）。**ブロック不在のファイルは従来と同一に読める**（後方互換）。存在時は \(Z_{\rm eff}/\bar Z\) 表へ縮約され `Laser.ib.zeff_model="table"`（§6.4.6）が利用可能になる
    - `"ideal_gas"`: analytic 2T ideal-gas EOS
    - `"power_law_te"`（2026-07-10 導入）: \(e_e=\)`f_erg_g`\(\cdot T_e^{\beta}\rho^{-\mu}\) [erg/g]、\(P_e=(\gamma_p-1)\rho e_e\)。**初期化時に 64×512 log 格子へ tabulate** され、既存 table-EOS 経路（逆写像・cv 含む）で消費される（凍結テーブル哲学）。キー: `f_erg_g: float`（必須 >0）/ `beta: float`（必須 >0）/ `mu_rho: float`（既定 0）/ `gamma_p: float`（既定 5/3、>1）。ion 側は ideal-gas 充填。`ideal_gas` subdict 併用は `ConfigError`。`cv_e_override` との併用は未定義（設定しないこと）。**optional c_v softstep（2026-07-14 導入、beta_sec G-S2 判別器基盤）**: `step_D_erg_g_eV: float`（既定 0 = 不在・厳密退化。>0 で \(e_e \mathrel{+}= D\,w\,\mathrm{softplus}((T_e-T_c)/w)\)、c_v に高さ \(D\) の滑らかな段差）/ `step_Tc_eV: float`（段差中心 [eV]）/ `step_w_eV: float`（幅 [eV]）。`step_D_erg_g_eV>0` は `step_Tc_eV>0` かつ `step_w_eV>0` を要求（`ConfigError`）。softplus は生成器と参照 tool で同一の安定形 \(\max(x,0)+\log1p(e^{-|x|})\)（設計 = docs/design/fleck_beta_secant_20260714.md §5-6）
  - `file: Optional[str]`（テーブルファイルパス。`model in {"sesame","ionmix","tmat"}` の場合は必須。未指定時は `ConfigError("eos.file is required for model={model}")`）
  - `sesame_material_id: Optional[int]`（SESAME 材料番号。`model="sesame"` の場合は必須。例：CH=7593, DT=5265。xSESAME ファイル内の材料 ID と一致すること。不一致時は `ConfigError("SESAME material ID {id} not found in {file}")`）
  - `sesame_format: Literal["ascii"]`（既定 `"ascii"`。xSESAME ASCII 形式。v1.0 では唯一のオプション）
  - `sesame_table_total: int`（既定 301。SESAME total EOS テーブル番号）
  - `sesame_table_electron: int`（既定 304。SESAME 電子 EOS テーブル番号。-1 を指定した場合は 1T フォールバック：\(P_e = P_{total} \times \bar{Z}/(1+\bar{Z})\)）
  - `hydro_backend: Literal["legacy","helmholtz_spline","helmholtz_jet","exact_ideal_gas","rho_e_table","mie_gruneisen"]`（既定 `"legacy"`。Hydro 専用 EOS backend 選択。`"legacy"` は既存の raw `P/e/c_v` テーブルを \((\log\rho,\log T)\) 双線形補間して `e→T` を単調探索で反転する経路、`"helmholtz_spline"` は**互換 key 名**で、実装は raw `total` EOS の `P(\rho,T), e(\rho,T)` を **shape-preserving C¹ tensor-product bicubic Hermite** surrogate に変換し、hydro に効く total pressure / total energy / sound speed をその解析的導関数から評価する。`"helmholtz_jet"` は raw `total` EOS から \(\phi=F/T\) の局所 projected jet を構築し、各セルを biquintic Hermite patch で補間する backend で、nodal \((\phi,\phi_x,\phi_y,\phi_{xx},\phi_{yy},\phi_{xy})\) を用いて total pressure / total energy / sound speed を導出する。`"exact_ideal_gas"` は **1D_SPH 限定の診断 backend** で、raw table は upload したまま 1D hydro kernel 内の EOS closure / sound speed のみを解析 ideal gas
    \[
    T = e/c_v,\qquad P = (\gamma-1)\rho e,\qquad c_s^2 = \gamma P/\rho
    \]
    に置き換える。`"rho_e_table"` は **1D_SPH 限定** の hydro-only backend で、初期化時に raw `total` EOS の `P(\rho,T), e(\rho,T)` から、既定では同じ \(\log\rho\) 軸と 200 点の一様 \(\log e\) 軸、`Numerics.hydro.rho_e_linear_grid=True` では元の `rho_grid` と 200 点の一様線形 \(e\) 軸を持つ `P(\rho,e_{total}), T(\rho,e_{total})` テーブルを構築し、さらに各 field に対して選択された座標系の C² tensor-product natural cubic spline 用 2 階導関数を事前計算する。Hydro では total EOS の `e→T→P` 反転 chain を経ずにこの spline から total pressure / total temperature / sound speed を直接評価する。sound speed は
    \[
    c_s^2 = \left.\frac{\partial P}{\partial \rho}\right|_e + \frac{P}{\rho^2}\left.\frac{\partial P}{\partial e}\right|_\rho
    \]
    を同じ `P(\rho,e)` テーブルから評価する。`"mie_gruneisen"` は **1D_SPH + 2T 限定** の hydro-only backend で、ion/electron 各枝に対して
    \[
    P_k(\rho,e_k)=P_{k,ref}(\rho)+\Gamma_k(\rho)\rho\left(e_k-e_{k,ref}(\rho)\right),\qquad k\in\{i,e\}
    \]
    を使う。`P_{k,ref}, e_{k,ref}, \Gamma_k` は raw ion/electron table の固定参照温度 `mg_T_ref_eV` と局所有限差分幅 `mg_dT_rel` から CPU 初期化時に構築し、\(\log\rho\) 上の monotone Hermite 補間係数として GPU に upload する。Hydro predictor/corrector の EOS closure / sound speed はこの affine closure だけを使い、raw table への `e→T` 反転・hydro-side writeback は行わない。代わりに source-step 側で raw table を用いて `T_i,T_e,c_{v,i},c_{v,e}` を再同期する。`exact_ideal_gas` では、cold-curve offset を hydro state へ持ち込まないため、初期状態および conduction / source-term 後の `Te/Ti → ee/ei/Pe/Pi` 再閉包も同じ ideal-gas 関係式を使う。`rho_e_table` / `helmholtz_spline` / `helmholtz_jet` では 2T の \(T_i,T_e,e_i,e_e,c_{v,i},c_{v,e}\) の個別 bookkeeping は従来の raw ion/electron table path を維持し、`rho_e_table` の 2T 圧力分割は raw \((P_i,P_e)\) 比を保ったまま total \(P(\rho,e_{total})\) に再正規化する。**Hydro のみ**が切替対象で、radiation/opacity の \((\rho,T)\) テーブル補間は従来の raw table path を維持する。NUMERICS §1.1.5(b) 参照）
  - `mg_T_ref_eV: float`（既定 `10.0` eV；`eos.hydro_backend="mie_gruneisen"` のときの参照温度。`P_ref(\rho), e_ref(\rho)` の抽出点として使う。`>0` 必須）
  - `mg_dT_rel: float`（既定 `0.1`；`eos.hydro_backend="mie_gruneisen"` のときの \(\Gamma(\rho)\) 評価用の相対差分幅。`dT = mg_dT_rel \times mg_T_ref_eV` を基準に raw table から \(\partial P/\partial e|_\rho\) を取る。`>0` 必須）
  - `ideal_gas: dict`（model=="ideal_gas"のとき必須。model!="ideal_gas" で指定された場合は無視して WARNING）
    - `gamma: float`（既定 5/3；断熱指数 \(\gamma_e = \gamma_i = \gamma\)。有効範囲：`(1, 3]`。NUMERICS §1.1.5 (a) 参照）
    - `cv_e_override: Optional[float]`（既定 None；指定時は電子比熱を定数 \(C_{v,e}\) [erg/(cm³·eV)] として固定する。**検証テスト用**：Marshak wave や Su-Olson 等の解析解が定数 \(C_v\) を要求する場合に使用。指定時は `gamma` による \(C_{v,e}\) 計算を上書きし、温度依存性なし。圧力は \(P_e = (\gamma-1)\,\rho\,e_e\) をそのまま使用。NUMERICS §6.1 の \(\beta_c = 4\,a_{eV}\,T_{e,c}^3 / C_{v,e}\) に直接適用される）
    - 2T理想気体EOS（NUMERICS §1.1.5 (a)）：
      - イオン：\(P_i = \rho k_B T_i / (A\,m_p)\)、\(e_i = \frac{3}{2} k_B T_i / (A\,m_p)\)
      - 電子：\(P_e = \bar{Z}\,\rho k_B T_e / (A\,m_p)\)、\(e_e = \frac{3}{2}\bar{Z}\,k_B T_e / (A\,m_p)\)
      - \(\bar{Z}\) は `zbar` モデルに従う
      - `cv_e_override` 指定時は \(C_{v,e}\) を上記公式の代わりに定数値を使用する
  - Runtime effects when `eos.model` is tabular (`sesame` / `ionmix` / `tmat`):
    - Hydro closure (`enforce_1t_closure_kernel` / `enforce_2t_closure_kernel`) uses the GPU hydro EOS backend selected by `eos.hydro_backend`. `exact_ideal_gas` では同じ 1D table-hydro kernel を通しつつ、EOS closure / sound speed だけを解析 ideal-gas 式へ置換し、電子比熱はセルごとの `state.zbar` と `cv_e_override` から再評価する。table backend (`legacy`, `helmholtz_spline`, `helmholtz_jet`, `rho_e_table`) では closure 中に `Te/Ti`, `Pe/Pi`, `cv_e/cv_i` を table / surrogate から閉じ、既定 `Numerics.hydro.eos_writeback=True` では closure 後に table / surrogate の clamped energy を state へ writeback して旧来の毎 step re-projection を行う。`eos_writeback=False` では Hydro が更新した `ee/ei` を保持し、NaN / Inf / 負の energy だけを repair として table / surrogate の clamped energy へ戻す。`rho_e_table` では 1T の total closure / sound speed を precomputed `P(\rho,e),T(\rho,e)` table へ切り替え、2T では total pressure / sound speed のみを `rho_e_table` に置き換えつつ `T_i,T_e,e_i,e_e,c_{v,i},c_{v,e}` は従来の raw ion/electron table path を維持する。`mie_gruneisen` では 2T hydro closure / sound speed は affine branch pressure だけを使い、hydro 中には `e→T` 反転も `ee/ei` writeback も行わない。table からの `T_i,T_e,c_{v,i},c_{v,e}` 再同期は driver/source-step 側で行う。さらに `Numerics.hydro.exact_override` が `pressure` / `sound_speed` / `temperature` のいずれかなら、table backend (`legacy`, `helmholtz_spline`, `helmholtz_jet`, `rho_e_table`) の 1D hydro closure 後にその 1 量だけを \(\gamma=5/3\), \(A\), \(Z\) の ideal-gas 式で diagnostic 上書きする。legacy diagnostic value `exact_override="no_writeback"` は引き続き受理され、`eos_writeback=True` 指定時でも 1D table path の writeback を強制的に無効化する。
    - Conduction (`compute_spitzer_deff_*`) consumes `state.cv_e` produced by the active hydro-EOS path; `mie_gruneisen` では hydro 外の raw-table thermo refresh がこれを供給する。
    - IMC Fleck factor (`compute_fleck_kernel`) consumes `state.cv_e` for \(C_{v,e}\). FLD grey/constant-opacity Fleck uses the dedicated `compute_fleck_for_fld` kernel, distinct from the IMC-shared kernel because FLD stiff-cell behavior requires no \(\beta\le1\) cap and no `f_min_fleck` floor.
    - \(Q_{ei}\) exchange uses `compute_qei_term_with_cv` when `cv_e/cv_i` are available from the active hydro-EOS path / source-side thermo refresh.
- `opacity: dict`
  - `model: Literal["ionmix","sesame","constant","table_nlte"]`（既定 `"ionmix"`。**推奨構成**: `eos.model="sesame"` + `opacity.model="ionmix"`。SESAME opacity（テーブル502/505）は grey のみのため、多群輸送には IONMIX opacity を推奨。Non-LTE は `"table_nlte"` を使用）
  - `"ionmix"`：IonMixテーブルファイルから読み込み（多群 κ_P, κ_R）
  - `"sesame"`：SESAME テーブル 502（Rosseland mean）/ 505（Planck mean）から読み込み。grey のみ（全群同一値）。xSESAME ファイルに 502/505 テーブルが含まれない場合は `ConfigError("SESAME opacity tables (502/505) not found")`
  - `"constant"`：一定不透明度（検証テスト用）
    - `kappa_a: float` -- 吸収不透明度 [cm²/g]（`model="constant"` の場合は必須）
    - `kappa_planck: Optional[float]` -- Planck-mean constant override [cm²/g] (default: unset; when unset the Planck constant follows `kappa_a`; Rosseland stays `kappa_a`)
    - `kappa_s: float` -- 散乱不透明度 [cm²/g]（既定 0.0）
    - 群依存性なし（全群同一値）
    - Rosseland/Planck 区別なし（κ_R = κ_P = kappa_a）
  - `"power_law"`（2026-07-10 導入、Hammer–Rosen gate 用の解析冪乗 opacity）:
    \(\kappa_P=\kappa_R=\)`kappa0_cm2_g`\(\cdot(T_e/\)`T_ref_eV`\()^{-\texttt{alpha\_T}}\cdot(\rho/\)`rho_ref_g_cc`\()^{\texttt{lambda\_rho}}\)
    - `kappa0_cm2_g: float`（必須、>0 [cm²/g]）/ `alpha_T: float`（既定 0）/ `lambda_rho: float`（既定 0）/ `T_ref_eV: float`（既定 1.0、>0）/ `rho_ref_g_cc: float`（既定 1.0、>0）
    - **grey 専用（v1）**: 放射群数 ≠ 1 は `ConfigError`。評価は FLD/S_N の opacity floor/cap（§6.4.5）でクランプ。Fleck 機構上は constant と同格（NUMERICS §6.7）
  - `"table_nlte"`（M17）：Non-LTE テーブル駆動。IONMIX .cn4 ファイルの3種不透明度（Rosseland, Planck absorption, Planck emission）を読み込み、\(\kappa^{PA} \neq \kappa^{PE}\) によるnon-LTE放射輸送を実現（§5.2.1、NUMERICS §6.1.1 参照）
    - `file: str`（必須。IONMIX .cn4 ファイルパス。最低2テーブル（Rosseland + Planck absorption）必須。Planck emission テーブル不在時は LTE フォールバック — §6.4.3 IONMIX 参照）
    - `lambda_method: Literal["finite_difference", "freeze_opacity"]`（既定 `"finite_difference"`。**後方互換用の no-op**。separate-emissivity 実装では runtime で無視される）
    - `lambda_fd_delta_rel: float`（既定 1e-4。**後方互換用の no-op**）
    - `lambda_fd_abs_min: float`（既定 1e-6 eV。**後方互換用の no-op**）
    - `f_min: float`（既定 1e-4。**後方互換用の no-op**。`table_nlte` / `tmat` の Fleck は \(\sigma_{p,em}\) から一意に決まる）
  - **非推奨（DEPRECATED）**：旧名称 `"none"` は `"constant"` + `kappa_a=0` と等価。`"none"` はv1.0では**非推奨**であり、将来バージョンで削除予定。
    `"none"` は v1.0 で受理されるが `WARNING: opacity.model="none" is deprecated, use model="constant" with kappa_a=0` を出力する。
    内部的に `{"model": "constant", "kappa_a": 0.0, "kappa_s": 0.0}` に変換される。
    全不透明度がゼロの場合、放射輸送は実質無効となる（光子は自由に streaming）。
    DDMC では τ_cell = 0 < τ_DDMC により全セルが IMC モードとなるため、
    ゼロ除算は発生しない。
  - `file: Optional[str]`（IONMIX ファイルは EOS と不透明度データの両方を含む。`eos.model="ionmix"` かつ `opacity.model="ionmix"` のとき、`opacity.file` を省略可能（`eos.file` から自動読み取り）。ファイルは1回のみ読み込まれ、EOS テーブルと不透明度テーブルの両方が抽出される）
  - `units: Literal["cm2_per_g"]`（既定 `"cm2_per_g"` [cm²/g]；v1.0では変更禁止。`"table_nlte"` 時は未使用。他の値を指定すると `ConfigError`）

#### SESAME ファイルフォーマット (xSESAME ASCII)

TENRYU は xSESAME ASCII 形式（80 文字固定幅、5E15.8）をデフォルト EOS として使用する。

**レコード構造**：
- 各レコードは **レコードヘッダ**（matId, tableId, nWords）で始まる
- データは 5E15.8 フォーマット（1行に最大5個の倍精度浮動小数点数、各15文字幅、指数8桁）
- 妥当性フラグ（validation flags）は読み込み時にスキップ

**EOS テーブル ID**：
| テーブル ID | 内容 | 単位（SESAME ネイティブ） | TENRYU 内部単位 |
|------------|------|------------------------|----------------|
| 201 | 補足情報（コメント） | — | — |
| 301 | Total EOS（\(P, e\) vs \(\rho, T\)） | GPa, MJ/kg, K | dyne/cm², erg/g, eV |
| 304 | Electron EOS（\(P_e, e_e\) vs \(\rho, T_e\)） | GPa, MJ/kg, K | dyne/cm², erg/g, eV |
| 305 | Ion EOS（データ構造は 304 同等） | GPa, MJ/kg, K | dyne/cm², erg/g, eV |
| 306 | \(\bar{Z}\) テーブル | — | — |
| 401 | Hugoniot 情報 | — | 読み込み不要 |

**単位変換**（NUMERICS §0.1 定数表準拠）：
- 温度：\(T_{\text{eV}} = T_K \times k_{B,eV/K}\)（\(k_{B,eV/K} = 8.6174 \times 10^{-5}\) eV/K）
- 圧力：\(P_{\text{dyne/cm}^2} = P_{\text{GPa}} \times 10^{10}\)
- 比エネルギー：\(e_{\text{erg/g}} = e_{\text{MJ/kg}} \times 10^{10}\)
- 密度：変換不要（g/cm³）

**2T 分離アルゴリズム**（CRITICAL）：
SESAME テーブル 301 と 304 は **グリッドサイズが異なる**場合がある
（例：Polystyrene mat 7593 では 301=73×41, 304=63×33）。
このため要素ごとの直接減算は不可能であり、以下の設計を採用する：
- `eos_total[mat]`: テーブル 301 の固有グリッドで EOSTable を構築
- `eos_e[mat]`: テーブル 304 の固有グリッドで EOSTable を構築
- イオン EOS は**クエリ時に差分算出**：\(P_i(\rho,T) = \text{interp}(\text{eos\_total},\rho,T) - \text{interp}(\text{eos\_e},\rho,T)\)
- これにより各テーブルの固有グリッド解像度が保持される

**304 不在時の 1T フォールバック**（例：Deuterium mat 5265 は 304 を持たない）：
- `sesame_table_electron = -1` を指定（または自動検出）
- 等価分割：\(P_e = P_{total} \times \bar{Z}/(1+\bar{Z})\)、\(e_e = e_{total} \times \bar{Z}/(1+\bar{Z})\)
- イオン：\(P_i = P_{total} - P_e\)

**Opacity テーブル**（`opacity.model="sesame"` 時）：
| テーブル ID | 内容 |
|------------|------|
| 502 | Rosseland mean opacity \(\kappa_R(\rho, T)\) [cm²/g] |
| 505 | Planck mean opacity \(\kappa_P(\rho, T)\) [cm²/g] |

SESAME opacity は **grey（全群同一値）** のみ。多群輸送には IONMIX opacity を推奨。

#### IONMIX ファイルフォーマット (v4/v6, .cn4 バイナリ)

TENRYU は FLASH/DRACO 互換の IONMIX4 バイナリフォーマット（`.cn4`）を使用する。
IONMIX6 は IONMIX4 の拡張であり、電子比エントロピーブロックが追加される（TENRYU はスキップする）。

> **参考実装**: opacplot2 ライブラリ（FLASH Center）がリファレンスリーダー/ライターを提供する。

**ファイル形式**: Fortran **unformatted sequential** バイナリ。各レコードは 4バイト整数の
レコード長マーカーで前後を挟む（`[len][data][len]`）。倍精度浮動小数点（8 bytes/value）。

##### ヘッダセクション

| レコード | 内容 | 型 | 備考 |
|---------|------|-----|------|
| 1 | `ntemp` | int | 温度グリッド点数 |
| 2 | `ndens` | int | **イオン数密度**グリッド点数 |
| 3 | 組成原子番号リスト `Z[n_elem]` | float | TENRYU は使用しない（Material.Z を優先） |
| 4 | 組成数密度比リスト `frac[n_elem]` | float | TENRYU は使用しない |
| 5 | `ngroups` | int | エネルギー群数 |

##### 軸グリッド

| レコード | 内容 | 単位 | 備考 |
|---------|------|------|------|
| 6 | `temps[ntemp]` | eV | 温度グリッド（昇順） |
| 7 | `numDens[ndens]` | **cm⁻³** | **イオン数密度**グリッド（昇順） |

> **密度軸の注意（CRITICAL）**: IONMIX の密度軸は**イオン数密度** \(n_i\) [cm⁻³] であり、
> 質量密度 \(\rho\) [g/cm³] ではない。TENRYU 内部で使用する質量密度との変換は：
> \[
> n_i = \frac{\rho}{A \cdot m_p}
> \]
> ここで \(A\) は平均原子量 [amu]、\(m_p = 1.6726 \times 10^{-24}\) g（陽子質量、§0.1 定数表）。
> テーブル補間時はまず \(\rho \to n_i\) に変換してから log 空間で補間する。

##### EOS 2D テーブル（2温度 12ブロック構成）

各ブロックは `ndens × ntemp` 要素の 2D 配列。
**配列順序**: 温度が**最速**（inner loop）、密度が**中間**。

| ブロック | 内容 | IONMIX 単位 | cgs 変換 |
|---------|------|------------|----------|
| 1 | \(\bar{Z}(n_i, T)\) 平均電離度 | 無次元 | 不要 |
| 2 | \(d\bar{Z}/dT_e\) | [eV⁻¹] | TENRYU はスキップ |
| 3 | \(P_i(n_i, T)\) イオン圧力 | **J/cm³** | ×10⁷ → erg/cm³ = dyne/cm² |
| 4 | \(P_e(n_i, T)\) 電子圧力 | **J/cm³** | ×10⁷ → erg/cm³ = dyne/cm² |
| 5 | \(dP_i/dT\) | [J/(cm³·eV)] | TENRYU はスキップ |
| 6 | \(dP_e/dT\) | [J/(cm³·eV)] | TENRYU はスキップ |
| 7 | \(e_i(n_i, T)\) イオン比内部エネルギー | **J/g** | ×10⁷ → erg/g |
| 8 | \(e_e(n_i, T)\) 電子比内部エネルギー | **J/g** | ×10⁷ → erg/g |
| 9 | \(de_i/dT\) イオン比熱 | [J/(g·eV)] | TENRYU はスキップ |
| 10 | \(de_e/dT\) 電子比熱 | [J/(g·eV)] | TENRYU はスキップ |
| 11 | \(de_i/dn_i\) | | TENRYU はスキップ |
| 12 | \(de_e/dn_i\) | | TENRYU はスキップ |

> **IONMIX6 追加**: ブロック 13 に電子比エントロピー \(s_e\) [J/(g·eV)] が追加される。
> TENRYU はこのブロックをスキップする（存在判定は `ngroups` レコード位置で自動検出）。

**TENRYU が使用する EOS ブロック**: 1（Z̄）, 3（P_i）, 4（P_e）, 7（e_i）, 8（e_e）の5ブロック。
その他の導関数ブロック（2, 5, 6, 9–12）は読み飛ばす。

**単位変換**:
- 圧力：\(P_{\text{dyne/cm}^2} = P_{\text{J/cm}^3} \times 10^{7}\)
- 比エネルギー：\(e_{\text{erg/g}} = e_{\text{J/g}} \times 10^{7}\)
- 温度：変換不要（eV）
- 密度：補間はイオン数密度 \(n_i\) [cm⁻³] で実行

##### エネルギー群境界

| レコード | 内容 | 単位 |
|---------|------|------|
| 次 | `bounds[ngroups+1]` | eV |

群境界は \(G+1\) 個の単調増加配列（\(G\) = `ngroups`）。

##### 多群不透明度 3D テーブル（3種分離）

各テーブルは `ngroups × ndens × ntemp` 要素の 3D 配列。
**配列順序**: 温度が**最速**（innermost）、密度が**中間**、群が**最外**（outermost）。

```
for (ig = 0; ig < ngroups; ig++)
  for (id = 0; id < ndens; id++)
    for (it = 0; it < ntemp; it++)
      read κ[ig][id][it]
```

| テーブル | 内容 | 単位 | TENRYU での用途 |
|---------|------|------|----------------|
| 1 | \(\kappa_R(n_i, T, g)\) Rosseland 不透明度 | cm²/g | DDMC リーク係数 |
| 2 | \(\kappa^{PA}(n_i, T, g)\) Planck **absorption** 不透明度 | cm²/g | IMC 吸収・有効断面積 |
| 3 | \(\kappa^{PE}(n_i, T, g)\) Planck **emission** 不透明度 | cm²/g | Non-LTE η_g 導出 |

> **LTE テーブル**: \(\kappa^{PA} = \kappa^{PE}\) が成立する。TENRYU は読み込み時に
> 全群・全グリッド点で \(|\kappa^{PA} - \kappa^{PE}| / \max(\kappa^{PA}, \epsilon)\) を検査し、
> 最大相対差が \(10^{-6}\) 以下の場合に LTE テーブルと判定する（INFO メッセージ出力）。
>
> **Non-LTE テーブル**: \(\kappa^{PA} \neq \kappa^{PE}\) により、
> 放射率 \(\eta_g = \sigma^{PE}_g \cdot c \cdot a_{eV} T_e^4 \cdot b_g(T_e)\) が
> 吸収係数 \(\sigma^{PA}_g\) と独立に決定される。
>
> **Planck emission 不在時**: 一部の古い IONMIX ファイルでは 3番目の不透明度テーブル（Planck emission）が
> 存在しない場合がある。この場合、\(\kappa^{PE} = \kappa^{PA}\) と仮定する（LTE フォールバック、WARNING 出力）。

EOS と不透明度を同一ファイルから読み取る場合、`opacity.file` を省略可能
（`eos.file` から自動読み取り）。

**IONMIX 読み込み時のエラー処理**：
- ファイルが存在しない場合：`ConfigError("IONMIX file not found: {path}")`
- ヘッダの `ntemp` または `ndens` が 0 以下の場合：`ConfigError("Invalid IONMIX header: ntemp={ntemp}, ndens={ndens}")`
- EOS 2D ブロック数が不足する場合（最低5ブロック：Z̄, P_i, P_e, e_i, e_e）：`ConfigError("IONMIX file has insufficient EOS blocks: expected ≥5, found {found}")`
- 不透明度テーブル数が不足する場合（最低2テーブル：Rosseland + Planck absorption）：`ConfigError("IONMIX file has insufficient opacity tables: expected ≥2, found {found}")`
- `opacity.model="ionmix"` でファイルの群数 `ngroups` が `Radiation.groups` の \(G\) と一致しない場合：`ConfigError("IONMIX opacity group count mismatch: file has {ngroups}, namelist specifies {G_user}")`。一致する場合は群境界の整合性を検査し、相対誤差 > 1e-3 の場合は WARNING を出力
- `opacity.model="table_nlte"` では最初に読み込んだ IONMIX ファイルの群数・群境界を採用し、後続の `table_nlte` 材料はそれとの整合性を検査する。群数不一致は `ConfigError`、群境界の相対誤差 > 1e-6 も `ConfigError`（WARNING ではない。Non-LTE ではη_g の群構造がIMCソースに直結するため、群境界不整合はエラー扱い）
- テーブル内に NaN / Inf / 負の不透明度が含まれる場合：`ConfigError("Invalid opacity value in IONMIX file at (T={T}, n_i={n_i}): {value}")`
- Fortran レコード長マーカーの不整合が検出された場合：`ConfigError("IONMIX binary record marker mismatch at offset {offset}")`

#### TMAT-H5 ファイルフォーマット (v1.0, .tmat.h5)

TENRYU は self-describing HDF5 形式の TMAT-H5（`.tmat.h5`）をサポートする。
IONMIX4/SESAME の代替フォーマットとして、EOS と opacity を独立グリッドで保持できる。

- 形式識別子：`/@format_id = "tenryu.material_table.hdf5"`
- 単位系：`/@units_system = "cgs_eV"`（TENRYU 内部単位と一致。ロード時の単位変換なし）
- EOS 軸順序：`/eos/@axis_order = "D,T"`（`/eos/grid/ni_cm3`, `/eos/grid/temperature_eV`）
- opacity 軸順序：`/opacity/@axis_order = "G,D,T"`（`/opacity/grid/group_bounds_eV` を含む）
- 密度軸：`ni_cm3` [cm⁻³]（IONMIX と同じイオン数密度軸）

**Namelist モデル値（明示指定）**：
- `eos.model="tmat"`：`Material.eos.file` に `.tmat.h5` を指定して EOS を読み込む
- `opacity.model="tmat"`：`Material.opacity.file` に `.tmat.h5` を指定して多群不透明度を読み込む
- EOS-only / opacity-only ファイルも許可（少なくとも一方は必須）

**設定例**：

```python
Material(
    name="CH",
    A=6.5,
    Z=3.5,
    eos=dict(model="tmat", file="CH.tmat.h5"),
    opacity=dict(model="tmat", file="CH.tmat.h5"),
)
```

**混在構成**：
- `eos.model="sesame"` + `opacity.model="tmat"` など、EOS と opacity のモデル混在を許可する。

#### 6.4.4 Geometry(...)
- `volfrac: dict[str, callable]`（必須）
  - key＝material名（`Materials` ブロックで定義済みの名前と一致必須。未登録名は `ConfigError("Unknown material: {name}")`）、value＝体積分率関数（戻り値 [無次元、0-1]）
  - **初期部分密度**：\(\rho_\alpha = f_{vol,\alpha} \times \rho_{total}\)。single-state 仮定では質量分率 = 体積分率
- `rho: callable`（必須）：総密度プロファイル関数。シグネチャ：1D_SPH `rho(r_cm) -> float` [g/cm³]、2D_RZ `rho(r_cm, z_cm) -> float` [g/cm³]。戻り値 `> 0` 必須（フロア以下は初期化時にクランプ）
- `Te: callable`（必須）：初期電子温度関数。シグネチャ：同上、戻り値 [eV]。`≥ Te_floor_eV` 必須
- `Ti: callable`（必須）：初期イオン温度関数。シグネチャ：同上、戻り値 [eV]。`≥ Ti_floor_eV` 必須
- `enforce_sum_to_one: bool`（既定 True；False の場合：正規化せず WARNING を出力。\(\sum f_{vol} < 1\) の場合、残余は void 扱い。\(\sum f_{vol} > 1\) の場合は `ConfigError`）
- **`cell_is_void` マスク導出**：Geometry 評価後、各セルについて非void材料の体積分率合計 \(\sum_{\alpha \notin void} f_{vol,\alpha}\) を計算し、\(\le 10^{-12}\) のセルを `cell_is_void=1`（voidセル）と判定する。voidセルでは `void_config` の `rho`, `Te`, `Ti` で初期化し、Zbar=0 を強制する。ALE remap 後にマスクを再計算する。mixed cell（例: 50% void + 50% material）はvoid扱いしない
- `velocity: Optional[callable]`（初期速度プロファイル [cm/s]；既定 `None`（= ゼロ速度）。1D_SPH: `v(r) -> v_r`、2D_RZ: `v(R,Z) -> (v_R, v_Z)`）
- `radiation_field: Literal["equilibrium","zero","planck"]`（既定 `"equilibrium"`；初期放射場の設定）
  - `"equilibrium"`：非リスタート初期化で、geometry 評価後に局所電子温度の Planck 場を設定する。各セル \(c\)、群 \(g\) で
    \[
    E_{c,g}=a_{eV}\,T_{e,c}^4\,b_g(T_{e,c})
    \]
    とし、backward-Euler 履歴 `rad_E_old[c,g]` も同じ値にする。
  - `"zero"`：`rad_E=0`、`rad_E_old=0` のまま保持する（真空/冷放射場 start の検証用）。
  - `"planck"`（2026-07-11 導入）：一様な放射温度 `radiation_field_Tr_eV` の Planck 場 \(E_{c,g}=a_{eV}\,T_r^4\,b_g(T_r)\) を全セルに設定する（`rad_E_old` も同値）。物質温度と独立に \(E \ne aT_e^4\) を初期化でき、放射→物質方向の緩和検証（§7.8a fleck 0-D gate 等）に使う。
  - `Radiation.enabled=False` の場合は初期化カーネルを呼ばず、allocation のゼロ値を保持する。restart 実行では checkpoint の `rad_E`/`rad_E_old` をそのまま使う。
- `radiation_field_Tr_eV: float`（既定 `-1.0`；`radiation_field="planck"` のとき必須 \(> 0\)、それ以外では無視される。`"planck"` 指定で正値がなければ `ConfigError`）

> 仕様上：ρ,Te,Ti は「セルの初期状態」であり、材料毎に別プロファイルを持ちたい場合は将来拡張（v1.0は総量として与える）。
> ただし、材料を分けた密度（部分密度）を与えたい場合は `rho` を省略して `rho_partial[mat]` を与える拡張を将来追加可能（v1.0は非対応）。

#### 6.4.5 Radiation(...)
- `enabled: bool`（既定 True；False の場合：transport 不実行、`rad_dep=0`、`dt_rad=∞`、全群 \(E_g=0\) 固定。DDMC/IMC 設定は無視される）
- `mode: Literal["imc_ddmc","multigroup_diffusion","sn_transport"]`（既定 `"multigroup_diffusion"`；DEFAULT-FLD: FREEZE-1D-RAD/FLD-FIX-1 後、既定は deterministic FLD に変更した。`"imc_ddmc"` は従来の IMC/DDMC/PGRW/HOLO/difference 経路。`"multigroup_diffusion"` は 1D_SPH/2D_RZ CUDA FLD 経路。`"sn_transport"` は 1D_SPH/2D_RZ CUDA pure \(S_N\) 経路。FLD/S_N mode は IMC/DDMC/HOLO/difference を完全に bypass し、`imc.enabled=False`, `ddmc.enabled=False`, `holo.enabled=False`, `imc.difference.enabled=False` が必須で、違反時は `ConfigError`。`Main.dimension` により 1D_SPH と 2D_RZ の dispatch を自動選択する）
  - `mode` 省略時の DEFAULT-FLD では、省略された `imc` / `ddmc` subblock は無効として解釈する。従来 IMC/DDMC を使う namelist は `mode="imc_ddmc"` を明示する。
  - **1D_SPH 制限**: `Main.dimension="1D_SPH"` の場合、`mode` は
    `"multigroup_diffusion"` または `"sn_transport"` のみ受理する。`"imc_ddmc"`
    を 1D_SPH と組み合わせると `ConfigError`。2D_RZ では全モードが利用可能。
  - `Radiation.imc` / `.ddmc` / `.holo` は internal/test-only mode 用の互換 subblock である。production namelist ではこれらを設定せず、`multigroup_diffusion` または `sn_transport` の mode 専用 subblock を使う。
- `origin_parity_only: bool`（既定 False；1D_SPH \(S_N\) 原点境界の調査用互換フラグ。現行 CPU/GPU \(S_N\) sweep は既に \(r=0\) parity 境界 \(\psi(0,+|\mu|)=\psi(0,-|\mu|)\) として実装されているため、このフラグは輸送式を変更しない。NUMERICS §8 参照）
- `group_repack_hard_xray: bool`（既定 False；True の場合、既存群数を維持したまま runtime の群境界を hard-X-ray 用に再配置する。80群では 2--5 keV 帯に 20 群以上を割り当て、table opacity は新しい群代表エネルギーへ再標本化される。False では従来通り input table または user 指定の群境界を用いる。NUMERICS §0.3 参照）
- `diagnose_hard_xray_opacity: bool`（既定 False；True の場合、起動時に一度だけ CD material の \(\kappa^{PA}\) hard-X-ray audit を `[hard_xray_opacity_diag]` として出力する。診断のみで opacity table は変更しない。NUMERICS §0.3 参照）
- `volume_source_rate: float`（既定 `0.0` [erg/(cm³ s)]；外部輻射体積線源（Su-Olson 級）の一定注入率。IMC では particle source として、1D_SPH `multigroup_diffusion` では W-B 実装の FV RHS source として消費する。1D_SPH FLD では `groups=1` 限定（多群は `ConfigError`）。注入エネルギーは step energy budget の `volume_in`（FLD: `fld_volume_source_in_step`）に計上。NUMERICS §6.7 1D_SPH 体積線源参照）
- `volume_source_x_max: float`（既定 `-1.0` [cm]；線源適用域の上限。セル中心 \(r_c \le x_{max}\)（IMC 平面系では \(x_c\)）のセルに適用。`volume_source_rate > 0` の場合は `> 0` 必須（`ConfigError`））
- `groups: dict`
  - `bounds_eV: list[float]`（単調増加 [eV]。長さ \(G+1\)（\(1 \le G \le 80\) 群）。有効範囲：各要素 `> 0`、`bounds_eV[i+1] > bounds_eV[i]`。最低2要素（1群）必須。例：`[0.01, 0.1, 1.0, 10.0, 100.0]` = 4群）。`opacity.model="table_nlte"` の材料が存在する場合、`bounds_eV` は IONMIX ファイルから自動導出され、明示指定は不要。明示指定した場合も IONMIX の値で上書きされる。
  - `representative: Literal["geometric_mean"]`（既定。群代表エネルギー \(E_g = \sqrt{E_{g-1/2} \cdot E_{g+1/2}}\)。NUMERICS §0.3参照）
  - `planck_fraction: dict`
    - `method: Literal["compute","tabulate"]`（既定 `"compute"`；`"compute"` = 初期化時に `bounds_eV` と温度グリッドから自動計算。`"tabulate"` = ユーザ提供のテーブルを使用（`T_grid_eV` と `b_g` が必須）。NUMERICS §0.3、ARCHITECTURE §4.5参照）
    - `compute_N_T: int`（既定 200；有効範囲：`[10, 10000]`；`method="compute"` 時の自動温度グリッド点数。対数等間隔で \(T_{min}\) ～ \(T_{max}\) を生成。`method="tabulate"` の場合は無視）
    - `compute_T_range_eV: Optional[list[float,float]]`（既定は未指定；有効範囲：`T_min > 0, T_max > T_min`；自動温度グリッドの下限・上限 [eV]。未指定時は active material の EOS table 電子温度範囲から自動導出し、EOS table がない場合は opacity table 範囲、最後に `[0.01, 1000.0]` eV へフォールバックする。`method="tabulate"` の場合は無視）
    - `T_grid_eV: list[float]`（`"tabulate"`時に必須；単調増加の温度グリッド [eV]、長さ \(N_T \ge 2\)）
    - `b_g: list[list[float]]`（`"tabulate"`時に必須；形状 \([G][N_T]\)、`b_g[g][k]` = 群gの温度 `T_grid_eV[k]` におけるPlanck分率。各kで \(\sum_g b_g[g][k]=1\) を満たすこと。許容誤差 \(|1-\sum_g|<10^{-12}\) を超える場合は再正規化して警告。再正規化前の欠損率 \(\delta(T_k) = 1 - \sum_g b_g^{raw}\) が \(10^{-3}\) を超える温度点がある場合は WARNING（群境界がPlanck分布の尾部を十分にカバーしていない；NUMERICS §0.3参照）。群設計は \(E_0 \ll T_{min}\)、\(E_G \gg T_{max}\) を推奨）
    - 補間：実行時は \(b_g(T)\) を温度方向に線形補間（NUMERICS §0.3準拠）
- `imc: dict`
  - `enabled: bool`（既定 False；internal/test-only の `mode="imc_ddmc"` 従来経路で legacy IMC transport を有効にする互換フラグ。production namelist では `imc` subblock を設定しない。`mode="multigroup_diffusion"` / `"sn_transport"` では True のままなら `ConfigError`）
  - `alpha: float`（time-centering；既定 1.0 [無次元]。有効範囲：`[0.5, 1.0]`；0.5=Crank-Nicolson、1.0=fully implicit。NUMERICS §6.1参照）
  - `f_max: float`（Fleck factor上限 [無次元]；既定 1.0（制限なし＝文献準拠）。有効範囲：`(0, 1.0]`。NUMERICS §6.1参照。0.5等で冷領域の拡散改善。`f_min_fleck`（§6.4.7）との関係：`f_max < f_min_fleck` は無効（`ConfigError`））
  - `corrected_fleck: bool`（既定 False；Cleveland & Wollaber (2018) の Modified Fleck / Corrected IMC を有効化。True のとき \(f = 1/(1+\alpha\beta c\Delta t\,\sigma_P(1+\xi))\) を用い、\(\xi = \frac{1}{4}\partial\ln\sigma_P/\partial\ln T\) を table opacity の有限差分で評価する。`table_nlte` / `tmat` の separate-emissivity path では \(\sigma_P=\sigma_{p,em}\) を使う。安全策：\(\sigma_P < 10^{-30}\) では \(\xi=0\)、\(1+\xi\) は \([0.1, 10]\) に clamp。NUMERICS §6.1参照）
  - `particles_per_cell_group: int`（既定 50（テスト用）。有効範囲：`≥ 1`。本番計算では 200 以上推奨（§8 GXII基準問題参照）。NUMERICS §6.3参照）
  - `implicit_capture: bool`（既定 True；False の場合：\(f=1\)、\(\sigma_{s,eff}=0\)、analog capture を使用。検証テスト用。`implicit_capture=False` の場合 `cutoff_fraction` と `weight_cutoff` は無効（粒子はanalog absorptionで消滅）。NUMERICS §6.2参照）
  - `cutoff_fraction: float`（既定 0.0 [無次元]（無効）；birth energyの割合、0.01等で有効化。有効範囲：`[0, 1)`。0.0 = 使用しない（粒子エネルギーがゼロになるまで追跡）。NUMERICS §6.3.4参照）
  - `inelastic_scatter: bool`（既定 True；非弾性実効散乱による群再サンプリング。False の場合：群変更なし（\(g_{new}=g_{old}\)）、方向のみ再サンプル。灰色（1群）の場合は効果なし。NUMERICS §6.2参照）
  - `weight_cutoff: float`（既定 1e-10 [無次元]；Russian roulette閾値（E_avg比：\(E < w_{cutoff} \times E_{avg}\) で判定）。有効範囲：`(0, 1)`。NUMERICS §6.3.4参照）
  - `roulette_survival: float`（既定 0.1 [無次元]；Russian roulette生存確率。有効範囲：`(0, 1)`。`roulette_survival ≥ weight_cutoff` でなければ `ConfigError`。NUMERICS §6.3.4参照）
  - `weight_split: float`（既定 1e+2 [無次元]；粒子分裂閾値（E_avg比）。**将来拡張（v1.0未実装）**。v1.0では設定値を保持するが分裂判定は実行しない。`≤ 0` で分裂無効の設定を先行定義。NUMERICS §6.3.4参照）
  - `max_split: int`（既定 8；1回の分裂での最大娘粒子数。**将来拡張（v1.0未実装）**。有効範囲：`[2, 64]`（予約）。NUMERICS §6.3.4参照）
  - `max_pool_size: int`（既定 `100_000_000`（10⁸）；PhotonPool最大容量 [粒子数]。超過時は緊急Russian rouletteを発動して粒子数を低減する。有効範囲：`≥ particles_per_cell_group × n_cells × n_groups` かつ `≤ 2,000,000,000`（2×10⁹、int32カウンタ安全上限。CUB API が int 引数を使用するため INT_MAX 以下が必須）。NUMERICS §6.4参照）
  - `source_tilting: bool`（既定 False；LD-IMC Phase-1 の thermal source tilting を有効化。`True` のとき 1D_SPH thermal emission のセル内位置サンプリングを \(T_e^4\) 勾配に応じて傾け、2D_RZ thermal emission では双線形写像 + \(R/R_{max}\) 棄却重みに \(T_e^4\) 勾配由来の real-space tilt bias を掛ける。放出エネルギー総量 \(S^{emit}_{i,g}V_i\Delta t\) と粒子重みは変更しない。1D tilt 係数は interior cell で \(\delta_i = \operatorname{clamp}\!\left[\frac{\Delta r_i}{2\max(U_i,U_{floor})}\frac{U_{i+1}-U_{i-1}}{r_{c,i+1}-r_{c,i-1}}, -1, 1\right]\)、\(U_i=a_{eV}T_{e,i}^4\) とし、境界セルまたは self/neighbor が void の場合は \(\delta_i=0\)。uniform mesh では \((U_{i+1}-U_{i-1})/(2U_i)\) に退化。2D_RZ では R/Z 方向の中心差分 tilt を L1 正規化して用いる。NUMERICS §6.2, §6.3参照）
  - `source_localization: bool`（既定 False；1D_SPH thermal source の emit 位置を前ステップ吸収位置のセル別 finite-width PDF へ局所化する。輸送で midpoint accumulators \(W_i^{abs}=\sum r_{mid}\Delta E_{p}^{abs}\)、\(Q_i^{abs}=\sum r_{mid}^2\Delta E_{p}^{abs}\)、\(E_i^{abs}=\sum \Delta E_{p}^{abs}\) を集計し、次ステップは \(\mu_i=\mathrm{EMA}(W_i^{abs}/E_i^{abs})\)、\(\sigma_i=\mathrm{clamp}(\sqrt{Q_i^{abs}/E_i^{abs}-\mu_i^2})\)、\(\alpha_{E,i}=\min(1,E_i^{abs}/(E_i^{abs}+E_{gate}))\) を作る。さらに Rosseland proxy optical depth \(\tau_i=\max_g(\sigma_{R,i,g}\Delta r_i)\) から \(w_{\tau,i}=\tau_i/(\tau_i+\tau_{ref})\) を作り、\(\mu_i\leftarrow r_{c,i}+w_{\tau,i}(\mu_i-r_{c,i})\)、\(\sigma_i\leftarrow \sigma_{uni,i}+w_{\tau,i}(\sigma_i-\sigma_{uni,i})\)、\(\alpha_i=\alpha_{E,i}w_{\tau,i}\) で薄いセルの局所化を弱める。`xi_mix < alpha_i` のとき \(r_p=\operatorname{clamp}(\mu_i+\sigma_i z,r_{i-1/2},r_{i+1/2})\)（\(z\) は Box-Muller の標準正規乱数）、それ以外は `source_tilting=True` なら tilted sampling、無効なら体積一様 sampling を使う。低吸収セル（\(E_i^{abs}\le 0.1E_{gate}\)）は \(\alpha_i=0\) で局所化を止め、セル中心を保持する。放出エネルギー総量 \(S^{emit}_{i,g}V_i\Delta t\) と粒子重みは変更しない。Phase-1 では 1D_SPH のみ対応し、2D_RZ では WARNING を出して従来 sampling へフォールバックする。localized branch 自体は `source_tilting` より優先するが、energy/tau gate で localized branch に入らない粒子は tilted/uniform fallback を使う。NUMERICS §6.2, §6.3参照）
  - `sloc_ema_beta: float`（既定 0.4 [無次元]；`source_localization` の mean radius に対する時間方向 EMA 係数。有効範囲：`[0,1]`。`0` = 完全に前ステップ値を保持、`1` = raw mean をそのまま採用）
  - `sloc_sigma_floor: float`（既定 0.1 [無次元]；`source_localization` の emit 幅下限をセル幅 \(\Delta r\) に対する比で与える。有効範囲：`> 0` かつ `<= sloc_sigma_cap`）
  - `sloc_sigma_cap: float`（既定 0.5 [無次元]；`source_localization` の emit 幅上限をセル幅 \(\Delta r\) に対する比で与える。有効範囲：`> 0` かつ `>= sloc_sigma_floor`）
  - `sloc_tau_ref: float`（既定 1.0 [無次元]；`source_localization` の optical-depth gate 基準。\(w_{\tau,i}=\tau_i/(\tau_i+\tau_{ref})\)、\(\tau_i=\max_g(\sigma_{R,i,g}\Delta r_i)\) として薄いセルの localized branch を連続的に弱める。有効範囲：`> 0`）
  - `spectral_bias_eta: float`（既定 0.0 [無次元]；thermal emission の spectral biasing 強度。有効範囲：`[0, 1]`。`0` = 無効、`1` = 完全 Rosseland importance。cell-local に \(p^{P}_{i,g}\propto b_g(T_{e,i})\)、\(\Delta T_i=\max(0.01\,T_{e,i},10^{-3}\,\mathrm{eV})\)、\(\left.\frac{\partial b_g}{\partial T}\right|_i \approx \frac{b_g(T_{e,i}+\Delta T_i)-b_g(\max(T_{e,i}-\Delta T_i,10^{-6}\,\mathrm{eV}))}{2\Delta T_i}\)、\(I_{i,g}=\max(\partial b_g/\partial T,0)/\max(\sigma_{R,i,g},\sigma_{floor})\)、\(q_{i,g}=(1-\eta)p^{P}_{i,g}+\eta I_{i,g}/\sum_h I_{i,h}\) を構築し、**thermal emission の粒子数配分のみ**を \(q_{i,g}\) ベースへ置換する。`source_E[i,g]`、`rad_emit[i,g]`、群積分放出エネルギーは不変で、1粒子エネルギー \(E_p = source_E[i,g]/N_{p,i,g}\) が自動補償する。Stage 1 / Phase-1 では effective-scatter regrouping と PGRW upscatter には適用しない。推奨値は `0.3-0.5`。NUMERICS §6.2参照）
  - `opacity_predictor: bool`（既定 False；true NLTE (`opacity.model in {"table_nlte","tmat"}`) の係数評価専用の半ステップ温度予測。True のとき、前ステップのセル別 `delta_E_rad_prev`（前ステップで電子エネルギーへ実際に適用した net radiation source）を用いて \(T_e^{pred} = T_e^n + \theta \Delta E_{rad}^{prev}/(\rho c_{v,e} V)\)（\(\theta = 0.5\) 固定）を作り、NLTE 係数評価の評価温度だけを置き換える。クランプ：`Te_floor` 以上、かつ \(|T_e^{pred} - T_e^n| / T_e^n \le 0.5\)。輸送・ソース注入・エネルギー更新は常に実状態 \(T_e^n\) を使うため、エネルギー保存は変更しない。初回ステップや restart 直後で履歴がない場合はそのステップだけ無効。NUMERICS §6.1参照）
  - `two_stage: bool`（既定 False；True のとき `Radiation` 演算子 \(\mathcal{R}(\Delta t)\) を2つの半ステージへ分割し，`transport_step(Δt/2) → source_injection(Δt/2)` を2回実行する。1段目の source injection 後に EOS 再クロージャで \(T_e,e_e,P_e,C_v\) を同期し，\(\bar{Z}\) を更新してから2段目を実行するため，2段目の opacity / Fleck / NLTE 係数は更新済み状態で再評価される。generic subcycling と異なり，stage 間で物質状態を更新して係数を再構成する。`state.rad_dep`，`state.rad_emit`，`delta_E_rad_prev` は \(\mathcal{R}(\Delta t)\) 終了時に2段の合計を保持し，`delta_E_rad_prev` は各段で電子エネルギーへ適用した net radiation source の合計を表す。NUMERICS §2.1, §6.3 参照）
  - `difference: dict`（既定 `{"enabled": False, "W_max": 1.0, "tau0": 3.0, "chi0": 1.0, "face_transport": True}`；full difference formulation の段階導入用設定。`enabled=True` のとき、1D_SPH および `face_transport=False` の 2D_RZ LTE nonlinear thermal source を signed residual source \(Q'=c\sigma_{a,eff}(B-E^{ref})V\Delta t\) として transport し、物理 `rad_emit` は \(c\sigma_{a,eff}BV\Delta t\) のまま保持し、reference absorption \(c\sigma_{a,eff}E^{ref}V\Delta t\) を `rad_dep` に preseed する。step 冒頭の census は previous-reference reservoir と signed residual 粒子へ再分割する。1D_SPH の `face_transport=True` では AP-limited deterministic reference face transport を separate reservoir buffers に適用し、境界 reference leakage は radiation escape accounting へ加えるが `rad_dep` へは加えない。`face_transport=False` は PR5 と同じ \(U^{ref,end}=U^{ref,start}\) の挙動であり、2D_RZ ではこの設定が必須。2D_RZ の optical-depth gate は \(\tau_i=\min(\bar\sigma_i h_{R,i},\bar\sigma_i h_{Z,i})\) を使い、empty-bin/census residual 粒子は2D thermal source と同じ双線形写像 + \(R/R_{max}\) 棄却サンプリングで配置する。`rad_E` は \(\bar{E}^{ref}+\mathrm{signed\_rad\_E\_tally}/(Vc\Delta t)\) として再構成し、clamp は final physical `rad_E` のみに適用する。loaded table-opacity path は source/census residualization を適用し、`linearized_planck` path は diagnostics-only。有効範囲：`0 <= W_max <= 1`, `tau0 > 0`, `chi0 > 0`。PR9 gate 完了まで production 推奨は無効で、既定OFFを維持する。NUMERICS §6.1.2参照）
    - `enabled: bool`（既定 False；difference reference/source split の有効化フラグ）
    - `W_max: float`（既定 1.0 [無次元]；reference weight の上限）
    - `tau0: float`（既定 3.0 [無次元]；\(W_i\) の optical-depth gate 基準）
    - `chi0: float`（既定 1.0 [無次元]；\(W_i\) の radiation-matter mismatch gate 基準）
    - `face_transport: bool`（既定 True；True で 1D_SPH AP reference face transport を有効化。2D_RZ では True は `ConfigError` で、`difference.enabled=True` と併用する場合は False が必須。False で reference reservoir は step start のまま保持）
  - `net_e_source_smoothing: dict`（既定 `{"enabled": False, "alpha": 0.2, "tau_threshold": 4.0, "passes": 1, "grad_Te_scale": 0.3, "grad_rho_scale": 0.5, "gradient_adaptive": False}`；Radiation source injection の前に、`H_raw = Σ_g(rad_dep - rad_emit)` に対して 1D_SPH/2D_RZ face-based conservative smoothing を適用する。`enabled=True` のとき、`imc.cpp` が保持する `sigma_R_max[c] = max_g sigma_R[c,g]` と cell mass / node geometry / dominant material / void mask を用いて face flux \(F_{ab}^{(p)}=\alpha_{ab} \lambda_{ab} m_{ab}(H_a^{(p)}/m_a-H_b^{(p)}/m_b)\) を `passes` 回 Jacobi 形式で計算し、\(H^{(passes)}\) を \(e_e\) へ適用する。1D_SPH は左右 face、2D_RZ は R/Z 4-face stencil を使い、2D の optical-depth gate は R face で \(\sigma_{R,\max}h_R\)、Z face で \(\sigma_{R,\max}h_Z\) を使う（area weight は使わない）。`difference.enabled=True` では reference weight \(W_i\ge0.5\) のセルも barrier とし、face は両隣セルが \(W<0.5\) の場合だけ smoothing 対象になる（0.5 は namelist knob ではない v1 互換性定数）。`gradient_adaptive=False` では \(\alpha_{ab}=\alpha\)。`gradient_adaptive=True` では source injection 前の \(T_e,\rho\) から \(G_T=\exp[-(|\Delta\ln T_e|/\texttt{grad_Te_scale})^2]\)、\(G_\rho=\exp[-(|\Delta\ln\rho|/\texttt{grad_rho_scale})^2]\) を作り、barrier 通過 face で \(\alpha_{ab}=\alpha G_T G_\rho\) とする。mask \(\lambda\) は `tau_threshold` と void/material interface で決まり、各 pass の総和 \(\sum H\) は厳密保存。`passes=0` で無効。`alpha` の有効範囲は 1D_SPH または smoothing 無効時 `[0, 0.25]`、2D_RZ で `enabled=True` のとき `[0, 0.125]`。NUMERICS §2.1参照）
    - `enabled: bool`（既定 False；source smoothing の有効化フラグ）
    - `alpha: float`（既定 0.2 [無次元]；1-pass conservative exchange の係数。有効範囲：1D_SPH または smoothing 無効時 `[0, 0.25]`、2D_RZ で `enabled=True` のとき `[0, 0.125]`）
    - `tau_threshold: float`（既定 4.0 [無次元]；face smoothing を許可する optical-depth gate。1D_SPH は各セルで \(\tau_c=\max_g(\sigma_{R,c,g}\Delta r_c)\) を計算し、2D_RZ は R face で \(\tau_{R,c}=\sigma_{R,\max,c}h_{R,c}\)、Z face で \(\tau_{Z,c}=\sigma_{R,\max,c}h_{Z,c}\) を使う。face では `min(tau_left, tau_right) >= tau_threshold` を要求する。有効範囲：`> 0`）
    - `passes: int`（既定 1；conservative exchange の pass 数。有効範囲：`>= 0`。`0` は smoothing 無効）
    - `grad_Te_scale: float`（既定 0.3 [無次元]；`gradient_adaptive=True` 時の \(|\Delta\ln T_e|\) Gaussian suppression scale。有効範囲：`> 0`）
    - `grad_rho_scale: float`（既定 0.5 [無次元]；`gradient_adaptive=True` 時の \(|\Delta\ln\rho|\) Gaussian suppression scale。有効範囲：`> 0`）
    - `gradient_adaptive: bool`（既定 False；True で face ごとに \(\alpha_{c+1/2}=\alpha G_TG_\rho\) を使い、smooth interior では最大 \(\alpha\)、大きな \(T_e,\rho\) 勾配では連続的に 0 へ近づける。difference 併用時の \(W<0.5\) gate は `gradient_adaptive` の値にかかわらず適用される）
  - `particle_budget: int`（既定 -1；総粒子数上限（ソース+census合計）。-1で無効（従来動作）、>0で有効。有効時、各ステップのソース生成粒子数を適応的に制御する（NUMERICS §6.2.2参照）。有効範囲：`-1` または `≥ particles_per_cell_group`。`0 < particle_budget < particles_per_cell_group` の場合は `ConfigError`）
  - `census_comb: dict`（Census Combing 設定。NUMERICS §6.4.1参照）
    - `enabled: bool`（既定 False；census粒子の個体数制御を有効化。True時、輸送後にcensus粒子数がトリガー条件を超えた場合に重要度重み付きリサンプリングを実行する。True時は予測的個体数コントローラ（NUMERICS §6.2）も自動的に有効化される）
    - `max_particles: int`（既定 `1_000_000`（10⁶）；census combing のターゲット粒子数上限。有効範囲：`≥ 1000`）
    - `min_per_bin: int`（既定 1；ビンあたりの最小保持粒子数。有効範囲：`≥ 1`）
    - `trigger_ratio: float`（既定 1.0 [無次元]；`n_alive > max_particles × trigger_ratio` で発動。有効範囲：`(0, 2.0]`）**[deprecated: 予測コントローラ導入により未使用。後方互換性のためパースは継続]**
    - `target_fraction: float`（既定 0.8 [無次元]；combing後のターゲット粒子数 = `max_particles × target_fraction`。有効範囲：`(0, 1.0]`。`target_fraction ≥ trigger_ratio` の場合は WARNING（combing が粒子を増やすことになる））**[deprecated: 予測コントローラ導入により未使用。後方互換性のためパースは継続]**
    - `mode_weight_imc: float`（既定 1.0 [無次元]；IMCビンの重要度重み）
    - `mode_weight_ddmc: float`（既定 0.5 [無次元]；DDMCビンの重要度重み。DDMCは位置非依存のため、IMCより低い重みが合理的）
    - `adaptive_trigger: bool`（既定 True；プール利用率に応じてtrigger_ratioを動的調整）**[deprecated: 予測コントローラ導入により未使用。後方互換性のためパースは継続]**
    - `adaptive_util_start: float`（既定 0.70 [無次元]；適応トリガーの開始利用率。有効範囲：`(0, 1)`）**[deprecated: 予測コントローラ導入により未使用。後方互換性のためパースは継続]**
    - `adaptive_util_end: float`（既定 0.95 [無次元]；適応トリガーの終了利用率。有効範囲：`(adaptive_util_start, 1]`）**[deprecated: 予測コントローラ導入により未使用。後方互換性のためパースは継続]**
    - `trigger_ratio_floor: float`（既定 0.85 [無次元]；適応トリガーの下限trigger_ratio。有効範囲：`(0, trigger_ratio]`）**[deprecated: 予測コントローラ導入により未使用。後方互換性のためパースは継続]**
    - `trigger_hysteresis: float`（既定 0.05 [無次元]；発動/停止間のヒステリシス幅。有効範囲：`[0, 0.5]`）**[deprecated: 予測コントローラ導入により未使用。後方互換性のためパースは継続]**
    - `ess_floor_enabled: bool`（既定 False；hard-trigger combing の直前に、Rosseland importance 上位群の low-ESS bin を split して粒子統計を底上げする）
    - `ess_min_tier0: float`（既定 16.0 [粒子数]；Rosseland importance 累積 50% 以内の tier-0 群に要求する最小 ESS。有効範囲：`> 0`）
    - `ess_min_tier1: float`（既定 8.0 [粒子数]；Rosseland importance 累積 90% 以内の tier-1 群に要求する最小 ESS。有効範囲：`> 0`）
    - `max_split_factor: int`（既定 4；ESS floor が 1 粒子を複製できる最大分割数。有効範囲：`≥ 1`）
  - `rad_lite_mesh: dict`（放射メッシュ粗視化設定。1D_SPH専用）
    - `enabled: bool`（既定 False；放射メッシュ粗視化の有効化）
    - `sigma_ratio_max: float`（既定 2.0 [無次元]；隣接セルの不透明度比閾値。有効範囲：`> 1.0`）
    - `nlte_auto: bool`（既定 False；`opacity.model in {"table_nlte","tmat"}` のとき step 単位で RadLite overlay を自動有効化する。`enabled=False` でも auto 条件を満たせば有効。自動有効化時は `sigma_ratio_max = max(user_value, 3.0)` を使い、既存 merge criterion のままより積極的にセル併合を許す。`nlte_auto=False` では従来どおり `enabled` のみで制御される。NUMERICS §7.5.2参照）
- `multigroup_diffusion: dict`（`mode="multigroup_diffusion"` で使用。NUMERICS §6.7 参照）
  - `flux_limiter: Literal["levermore_pomraning","larsen","none"]`（既定 `"levermore_pomraning"`；Levermore-Pomraning FLD limiter、Larsen limiter、または bare diffusion）
  - `max_outer_iterations: int`（既定 20；matter-radiation Picard 最大反復数。有効範囲：`>= 1`）
  - `outer_tol: float`（既定 `1.0e-5`；\(\max|\Delta T_e|/T_e\) 収束許容値。有効範囲：`> 0`）
  - `fleck_mode: str`（既定 `"fleck_cummings"`；物質 emission の時間線形化。`"fleck_cummings"` = f=1/(1+z) ブレンド、`"afi"` = Almost Fully Implicit（Larsen, Kumar & Morel, JCP 238 (2013)）— f ブレンド不使用、outer 反復が完全陰的 emission を収束（`max_outer_iterations >= 40` 推奨）。許容値：`"fleck_cummings"` / `"afi"`）
  - `hydro_coupling: Literal["none","gamma_r_43"]`（既定 `"gamma_r_43"`；`"gamma_r_43"` は W-K gamma_r=4/3 radiation compression on the Lagrangian mesh を有効化し、v1 scope は 1D + deterministic FLD のみ。`"none"` は frozen-density historic behavior への明示 opt-out。`compatible_energy` 併用は unsupported-assert。\(W_r\) は existing `E_rad_mesh_advection` tally で \(\Delta(E_r V)\) として booked。SnTransport では活性化しない（mode=="multigroup_diffusion" を driver で enforce）。compatible_energy=True の既存 GXII 系 deck は明示 hydro_coupling="none" で opt-out（compatible 対応は v2））
    - 2026-07-06: flip → 同日 revert（v1 創出 +10%）→ v3（力仕事共役支払い + rad_E_old snapshot 修正）で解決後、新 A/B（deposited +1.65% / bang −2.28% / ρ_peak +4.75%）に基づき同日深夜に再採用（R2-1=A）。
  - `state_supply_boundary_policy: Literal["local_D_current","harmonic_ghost_D_test","radial_mean_D_test"]`（既定 `"local_D_current"`；2D_RZ 灰色1群 FLD の `"state_supply"` Z 面における境界拡散係数 closure。`"local_D_current"` が production 既定で、境界セルの現行 \(D\) を使う。`"harmonic_ghost_D_test"` と `"radial_mean_D_test"` は Round 13 candidate-A isolation 用の **DIAGNOSTIC-ONLY** policy であり、production 既定ではない。`"radial_mean_D_test"` は slab 以外の幾何では安全な物理 closure と見なさない。NUMERICS §6.7.3 参照）
  - `diagnostic_radial_fourier_substage_enabled: bool`（既定 `False`；True のとき FLD 内部 substage の radial Fourier audit を `/diagnostics/fld_substage_audit/v1/` へ出力する。diagnostic-only で、無効時は group を作成しない）
  - `cg_inner_tol: float`（既定 `1.0e-10`；2D_RZ FLD CG solve の内部相対許容値。有効範囲：`> 0`。既定値は従来挙動を保持する）
  - `cg_tol_norm: string`（既定 `"r0"`；許容値 `{"r0","rhs"}`；CG convergence normalization: 'r0' = historic initial-residual-relative; 'rhs' = RHS-norm-relative (fixed solve quality irrespective of warm start; opt-in).）
  - `outer_accel: string`（既定 `"none"`；許容値 `{"none","anderson"}`；opt-in Anderson acceleration of the FLD outer fixed-point iteration; the converged exit is always the raw Newton output）
  - `anderson_m: int`（既定 `2`；有効範囲：`[1,4]`；Anderson history depth）
  - `anderson_beta: float`（既定 `1.0`；有効範囲：`(0,1]`；Anderson damping）
  - `cg_max_iter: int`（既定 `500`；2D_RZ FLD CG solve の最大反復 cap。実際の反復上限は `max(50, min(cg_max_iter, n_rows))`。単位なし。有効範囲：`>= 1`。既定値は従来の hardcoded cap を保持する）
  - `cap_exit_policy: Literal["warn","fail"]`（既定 `"warn"`；2D_RZ FLD の CG cap exit または outer cap exit が許容値未達の場合の扱い。`"warn"` は従来どおり計算を継続し、`run_info.json` の `cg_cap_exit_unconverged` / `newton_cap_exit_unconverged` を増やして rate-limited warning を出す。`"fail"` は同じ counter を増やした後、step・group・residual を含む fatal error で run を停止する）
  - `rgmg_smoother_omega: float`（既定 `0.67`；`linear_solver_2d="cusparse_cg_rgmg"` の damped z-line block-Jacobi smoother の減衰係数。他 solver では未使用。有効範囲：`> 0`。SPD preconditioner としては実用上 \(\omega \le 2/\lambda_{\max}(M_z^{-1}A)\) が必要）
  - `linear_solver_1d: Literal["cusparse_tridiag"]`（既定 `"cusparse_tridiag"`；1D group-batched tridiagonal solve。これ以外は `ConfigError`）
  - `linear_solver_2d: Literal["auto","amgx_cg","jacobi","cusparse_cg_jacobi","cusparse_cg_zline","cusparse_cg_rgmg"]`（既定 `"auto"`；2D_RZ CSR solve。`"auto"` は **namelist validate 時**に格子から解決する: `nr` が 2 の冪かつ `nz>=3` → `"cusparse_cg_rgmg"`、それ以外で `nz>=3` → `"cusparse_cg_zline"`、どちらも不成立 → `"cusparse_cg_jacobi"`（解決結果は 1 回 INFO ログ、frozen config には解決後の値が入る）。solver 整合性契約（外部 AI 裁定前提①）: deck が値を明示したかを追跡し、`linear_solver_2d_requested`（deck 指定値、未指定なら既定）と `linear_solver_2d_resolved`（実際に使う solver）を run_info + HDF5 metadata に記録する — 黙った能力置換は無い。**AmgX が link されていない build で `"amgx_cg"` を明示指定すると ConfigError（fatal）**。`"jacobi"` は `"cusparse_cg_jacobi"` と同じ debug fallback CG の別名。`"cusparse_cg_zline"` は z-line block-Jacobi preconditioned CG（radial line ごとの z-tridiagonal solve、cuSPARSE `cusparseDgtsv2`）。`"cusparse_cg_rgmg"` は r-semi-coarsened geometric multigrid を SPD preconditioner として使う MG-PCG（r 方向のみ半粗化、z-line smoother、Galerkin coarse operator；`nr` 2 冪必須）。既定 flip の検証 battery は VERIFICATION §9.5.6（SPD/直接解/contrast sweep/真残差/tol ladder）を参照）
  - `amgx_config: dict`
    - `preset: Literal["AGGREGATION_JACOBI"]`（既定 `"AGGREGATION_JACOBI"`；同梱 `resources/amgx_fld_config.json` の設定名）
  - `opacity_floor: float`（既定 `1.0e-100` [cm\(^{-1}\)]；FLD/opacity 評価の下限。有効範囲：`>= 0`）
  - `opacity_cap: float`（既定 `1.0e20` [cm\(^{-1}\)]；FLD opacity 評価の上限。有効範囲：`> opacity_floor`）
  - `fleck_cv_source: Literal["legacy","table"]`（**既定 `"table"`**、2026-07-10 導入・**2026-07-11 既定フリップ（外部AI裁定、docs/design/fleck_cv_default_flip_20260711.md）**）: FLD Fleck 因子の \(z\) 評価に使う電子比熱の出所。`"table"`（既定） = 電子 EOS テーブルが存在すれば現在 \(T_e\) の table cv を最優先（matter Newton が前進させるエネルギー関数と同一の \(\partial U_e/\partial T_e\) — Fleck–Cummings 1971 の整合要件、NUMERICS §6.7）; テーブル不在時は legacy チェーンに落ちる（matter 側も非テーブル分岐のため整合）。`"legacy"` = 旧チェーン（cv_e_override → state cv_e → ideal-gas fallback）を凍結保存する明示互換モード — 旧 golden の bit 再現・A/B 比較用。table-EOS 材料で legacy を使うと Fleck 線形化と matter 更新が別の熱力学モデルになり、transient 交換率が最大 \(q=C_{legacy}/C_{table}\) 倍歪む（HR-W1c 発見; 0-D 緩和 gate verify_fleck_relaxation_0d が率忠実度を常設検証）。フリップの既存 golden への影響は無し — 本 branch の table-EOS FLD gate 群（Hammer–Rosen / 0-D 緩和）は mode を deck 内で明示 pin 済み、GXII FLD regression は ideal_gas EOS で knob 構造的不活性（table 既定下の golden 再生成が旧 golden と全 6 指標 bit 同一、VERIFICATION §4.z3）。tmat+FLD の無 pin deck（XC probe、2D lane）はフリップで補正物理を継承する
  - `fleck_beta: Literal["tangent","secant","guard"]`（既定 `"tangent"`、2026-07-14 導入 — flip verdict 繰延 item 1）: Fleck β の線形化点。`"tangent"`（既定・bit 保存） = β = 4a_eV T³/C_v（従来）。`"secant"` = grey 弦 β = ΔB/ΔU_e を 0-D 局所予測子（f_tan·cσ_P(E−B)Δt/C_v、trust region ±0.5T、退化時 tangent へ fallback）の張る区間で評価 — table-EOS セルのみ・**1D FLD のみ**（2D の fleck kernel は独立実装で tangent 固定）。`"guard"` = 片側単調性リミッタ β_used = max(β_tan, β_sec) ⇒ f_used = min(f_tan, f_sec)（裁定 2026-07-15 §9.2；secant と同じ table-EOS/1D 制約・同じ予測子、選択のみ max）。設計・gate 計画 = docs/design/fleck_beta_secant_20260714.md
  - `fleck_form: Literal["be","exp_phi1"]`（既定 `"be"`、2026-07-16 導入 — β_sec 後続裁定 §12 の係数レバー）: Fleck 因子の時間形状。`"be"`（既定・bit 保存） = f = 1/(1+z)（従来の backward-Euler 形）。`"exp_phi1"` = f = (1−e^{−z})/z = φ₁(−z) — 固定輻射スカラー緩和の厳密保持率（0<f≤1、stiff 極限 z·f→1 で交換維持）。fleck_beta と直交（z の β には tangent/secant がそのまま入る）。**1D FLD のみ**（persistent path 含む; 2D fleck kernel は独立実装で be 固定）。設計 = docs/design/fleck_exp_source_20260716.md
  - `source_integrator: Literal["fleck","exp_rosenbrock"]`（既定 `"fleck"`、2026-07-16 導入 — rung-2、docs/design/fleck_exp_source_20260716.md §3）: 1D FLD の物質–輻射ソース積分器。`"fleck"`（既定・bit 保存） = 従来のモノリシック半陰的 outer ループ。`"exp_rosenbrock"`（opt-in） = Lie 分割 step — 凍結係数の厳密直接移送 q=hφ₁(−(1+β)h)(E−aT⁴) を両側対称適用（局所保存厳密）→ 交換項を除いた純拡散陰解、outer Picard なし（max_outer_iterations は無視・documented）。**制約**: `Radiation.groups <= 96`（多群は G×G rank-1 縮約の固定極 φ₁ = O(pG)/セル、2026-07-17 導入・docs/design/exp_mg_phi1_20260717.md；灰色 G==1 は従来のスカラー kernel を bit 不変で維持、γ_g=d(b_gB)/dU で可変 Planck 分率でも 2 次）・`fleck_mode="afi"` と非互換・`fleck_form="exp_phi1"` と併用不可（dead knob 防止）・persistent path は非対応（multi-kernel fallback）・**1D FLD のみ**（2D は独立実装で fleck 固定）。fleck_beta は v1 tangent 限定（secant/guard 併用は `ConfigError` — 予測子は認証済み Fleck kernel 内にあり、複製/refactor いずれもリスク）。0-D gate (j) で 2 次実証 (比 0.430/0.293/0.186/0.109・保存 drift 厳密 0)；**W4 1-D Marshak feature gate PASS (2026-07-17、分割バイアス無し・fleck と同一連続極限) — 事前登録の生産適格基準を充足** (verify_marshak_feature_1d 常設)
  - `z_boundary: Literal["vacuum","reflect","marshak","state_supply"]`（既定 `"vacuum"`；2D_RZ Z端境界。`boundary.z` と同義の両Z端共通指定。`boundary.z_bottom` / `boundary.z_top` を指定した場合はそれらが各面の実効値になる。`"state_supply"` は両Z端を state-supply にするため、両 hydro z-face も `type="state_supply"` でなければ `ConfigError`）
  - `boundary: dict`
    - `inner_r: Literal["reflect"]`（既定 `"reflect"`；1D_SPH 原点対称 / 2D_RZ R軸反射。他値は `ConfigError`）
    - `outer_r: Literal["vacuum","reflect","marshak"]`（既定 `"vacuum"`；`"vacuum"` は Marshak-like \(F=cE/2\) escape、`"reflect"` は face flux 0。`"marshak"` は 1D_SPH のみ（W-B）: \(F_{out}=(c/4)E-F_{inc}\)。入射駆動は排他的二択 — `Radiation.boundary.marshak_Tr_eV > 0`（黒体、multigroup 可）または `marshak.flux_erg_per_cm2_s > 0`（灰色 `groups=1` 限定）。両方指定・両方ゼロは `ConfigError`。NUMERICS §6.7 1D_SPH BC 参照）
    - `z: Literal["vacuum","reflect","marshak","state_supply"]`（既定 `"vacuum"`；2D_RZ 両Z端境界の共通既定値）
    - `z_bottom: Literal["vacuum","reflect","marshak","state_supply"]`（既定は `boundary.z`；`z=z_min` 面）
    - `z_top: Literal["vacuum","reflect","marshak","state_supply"]`（既定は `boundary.z`；`z=z_max` 面）
    - `"state_supply"` は 2D_RZ 灰色1群 FLD のみ対応し、同じ面の `Numerics.hydro.boundary_2d.{z_bottom,z_top}` が dict-form `type="state_supply"` でなければ `ConfigError`。Dirichlet 面値は hydro supply field の `T_eV` から \(E_b=a_{eV}T_s^4\) として導出し、radiation 側の独立温度パラメータは持たない。`Radiation.groups != 1` では `ConfigError`。
    - 1D_SPH（W-B）: `z` / `z_bottom` / `z_top` は 2D_RZ 専用のため、非既定値（`"vacuum"` 以外）を指定すると `ConfigError`（黙殺しない）。`inner_r` は `"reflect"` 固定。
  - `marshak: dict`
    - `flux_erg_per_cm2_s: float`（既定 `0.0` [erg/(cm² s)]；2D_RZ で `boundary.z_bottom` または `boundary.z_top` が `"marshak"` の場合、駆動は排他的二択 — (i) この灰色定常入射 flux（`Radiation.groups != 1` は `ConfigError`、`flux_pulse_duration_s` 矩形パルス対応）、または (ii) `Radiation.boundary` の Tr(t) 源（`marshak_Tr_eV` 定数 / `marshak_Tr` 時間 callable / `marshak_Tr_map` 面別 dict — indirect-drive 2026-07-11; per-group Planck 重み \(b_g(T_r)\) を供給するため**多群可**）。両方指定・両方ゼロは `ConfigError`。解決規約（面テーブル [正準 `bottom_z`/`top_z`、alias `z_bottom`/`z_top`] ▸ 定数 ▸ スカラーテーブル、solve-entry 評価）と RHS post-pass 実装は NUMERICS §6.7 の 2D Tr(t) 段落参照）
    - `flux_pulse_duration_s: float`（既定 `-1.0` [s]；`-1` は全stepで有効、`>=0` は `t < flux_pulse_duration_s` のstepで有効。任意波形は未実装）
- `sn_transport: dict`（`mode="sn_transport"` で使用。NUMERICS §6.8 参照）
  - `n_angles: Literal[2,4,8,16,32]`（既定 `16`；1D \(S_N\) Gauss-Legendre 角度数、および 2D_RZ level-symmetric product quadrature の代表次数。`8` は高速設定として許容するが、production では S8/S16 angular error gate を確認すること。**W-G3 (2026-07-04): `Mesh.geometry_1d="cylindrical"` では `n_angles` は積求積の総 ordinate 数 \(2L^2\)（L=極 level 数、方位角 M=2L/level）と解釈され、\(2L^2\)（8, 18, 32, 50, …、L≥2）以外は driver が assert で拒否する — NUMERICS §6.8.3。DSA は cylindrical で自動無効（警告 1 回、加速のみで収束解不変）**）
  - `angular_quadrature: str`（既定 `"level_symmetric_16"`；2D_RZ 角度集合名。`level_symmetric_N` の N は `n_angles` と同期）
  - `spatial_scheme: Literal["diamond_difference","linear_characteristic"]`（既定 `"linear_characteristic"`；1D_SPH radial sweep および 2D_RZ R-Z sweep closure。`"diamond_difference"` は deprecated regression option として DD 経路の後方比較にのみ残す）
  - `max_outer_iterations: int`（既定 `20`；material Picard 最大反復数。有効範囲：`>= 1`）
  - `max_inner_iterations: int`（既定 `100`；source iteration 最大反復数。有効範囲：`>= 1`）
  - `outer_tol: float`（既定 `1.0e-4`；\(\max|\Delta T_e|/T_e\) Picard 収束許容値。有効範囲：`> 0`）
  - `outer_tol_stagnation_factor: float`（既定 `0.5`；5回目以降の Picard 反復で \(r_k/r_{k-2} >\) この値なら停滞として収束扱い。有効範囲：`> 0`）
  - `outer_tol_hydro_error_scale: float`（既定 `1.0e-5`；Picard 有効許容値を \(\max(\text{outer_tol},10\,\text{outer_tol_hydro_error_scale})\) に緩和する hydrodynamic time-error scale。有効範囲：`>= 0`）
  - `inner_tol: float`（既定 `1.0e-6`；\(\max|\Delta\phi|/\phi\) 収束許容値。有効範囲：`> 0`）
  - `inner_graph_unroll: int`（既定 `5`；1D_SPH source iteration CUDA graph の unroll 数。残差はこの回数ごとに確認する。`1` は graph capture を無効化する。有効範囲：`>= 1`）
  - `dsa_enabled: bool`（既定 `True`；DSA 加速を有効化する。`False` では source iteration の DSA correction を実行しない。比較検証・収束効果測定用）
  - `diffusion_fallback_mode: Literal["none","per_group_hysteresis"]`（既定 `"none"`；`"per_group_hysteresis"` は Cut-2+ 予約。Cut-1b production では `"none"` 必須）
  - `tau_diffusion_on: float`（既定 `10.0`；予約値。有効範囲：`>= 0`）
  - `tau_diffusion_off: float`（既定 `5.0`；予約値。有効範囲：`>= 0`）
  - `opacity_floor: float`（既定 `1.0e-100` [cm\(^{-1}\)]；opacity 評価の下限。有効範囲：`>= 0`）
  - `opacity_cap: float`（既定 `1.0e20` [cm\(^{-1}\)]；opacity 評価の上限。有効範囲：`> opacity_floor`）
  - `timing_enabled: bool`（既定 `False`；True のとき 1D_SPH \(S_N\) step が 100 radiation steps ごと（環境変数 `TENRYU_SN_TIMING_WINDOW` で変更可）に `[sn_timing]` per-block timing summary を stdout へ出力し、2D_RZ \(S_N\) step は `[sn_2d_rz_timing] outer_iters=...` の反復診断を stdout へ出力する。テスト影響回避のため既定OFF）
  - 1D_SPH `linear_characteristic` sweep は角度平均と outgoing intensity を保存し、\(\phi\), \(F\), \(P_{rr}\), raw face flux を決定的 reduction kernel で後段計算する
  - `z_boundary: Literal["vacuum","reflect","marshak"]`（既定 `"vacuum"`；2D_RZ Z端入射境界。`boundary.z` と同義で、両方指定時は一致必須。`boundary.z_bottom` / `boundary.z_top` を指定した場合はそれらが各面の実効値になる）
  - `boundary: dict`
    - `inner_r: Literal["reflect_parity"]`（既定 `"reflect_parity"`；\(r=0\) parity \(\psi(0,\mu<0)=\psi(0,-\mu)\)。他値は `ConfigError`）
    - `outer_r: Literal["vacuum","reflect","marshak"]`（既定 `"vacuum"`；`"vacuum"` は outer incoming intensity zero、`"reflect"` は 2D_RZ の閉円筒/1D-z reduction 用。1D_SPH では `"reflect"` は `ConfigError`。`"marshak"` は 1D_SPH（W-B2）: 内向き半区間へ \(\psi^-=2F_{inc,g}\) を入射（`spatial_scheme="linear_characteristic"` 必須）。駆動は `Radiation.boundary.marshak_Tr_eV > 0`（黒体・multigroup 可）または `sn_transport.marshak.flux_erg_per_cm2_s > 0`（灰色 `groups=1`、`flux_pulse_duration_s` 対応）の排他的二択、両方指定・両方ゼロは `ConfigError`。NUMERICS §6.8 1D_SPH Marshak 参照）
    - `z: Literal["vacuum","reflect","marshak"]`（既定 `"vacuum"`；2D_RZ 両Z端境界の共通既定値）
    - `z_bottom: Literal["vacuum","reflect","marshak"]`（既定は `boundary.z`；`z=z_min` 面）
    - `z_top: Literal["vacuum","reflect","marshak"]`（既定は `boundary.z`；`z=z_max` 面）
  - `marshak: dict`
    - `flux_erg_per_cm2_s: float`（既定 `0.0` [erg/(cm² s)]；2D_RZ で `boundary.z_bottom` または `boundary.z_top` が `"marshak"` の場合、駆動は排他的二択 — (i) この灰色定常入射 flux、または (ii) `Radiation.boundary` の Tr(t) 源（indirect-drive 2026-07-11、\(F_{inc}=(c/4)a_{eV}T_r^4\) を既存スカラー slot へ供給）。**SN 2D はどちらの経路も `Radiation.groups != 1` で `ConfigError`**（z 面注入 \(\psi^-=2F_{inc}\) が構造的に灰色のため; 多群 spectral 注入は将来 wave）。Tr 経路で両 z 面 marshak + `marshak_Tr_map` 面別テーブルは `ConfigError`（単一スカラー共有; 定数/スカラー callable 源は両面共通で可）。両方指定・両方ゼロは `ConfigError`。NUMERICS §6.7 の 2D Tr(t) 段落参照）
- `ddmc: dict`
  - `enabled: bool`（既定 False；internal/test-only の `mode="imc_ddmc"` 従来経路で DDMC を有効にする互換フラグ。False の場合：全セルが IMC mode map に留まる。`tau_ddmc`, `omega_ddmc`, `leak_stencil` など DDMC 専用パラメータは無視される。`tau_rw` は internal PGRW threshold として引き続き有効。production namelist では `ddmc` subblock を設定しない。実装上は cold-start warm-up として global step `0..9` でも DDMC entry を無効化し、全セルが IMC に留まる）
  - `implicit_diffusion: bool`（既定 False；HIMCD Phase-1 の切替。True かつ **1D / LTE / Marshak境界なし / volume_source_rate=0** の場合、DDMC セル×群は particle DDMC の代わりに host-side backward Euler implicit diffusion solve で更新する。Phase-1 では DDMC-IMC 界面は zero-flux とし、未対応条件では WARNING を出して従来の particle DDMC にフォールバックする。NUMERICS §7.4.1 参照）
  - `tau_ddmc: float`（既定 4.0 [無次元]；有効範囲：`≥ 1.0`；DDMC 切替の光学厚下限閾値。NUMERICS §7.1 参照）
  - `tau_rw: float`（既定 0.0 [無次元]；有効範囲：`≥ 0`；internal PGRW threshold。`tau_rw=0` で PGRW 無効。Phase-1 では 1D_SPH のみ対応し、`rad_lite_mesh` 無効かつ IMC 粒子が per-cell diffusive group cutoff の内側にあり、RW sphere optical depth が `tau_rw` 以上のとき `imc_transport_persistent` 内で PGRW branch を実行する。独立の `TransportMode::RW` は生成しない。NUMERICS §7.4.2 参照）
  - `omega_ddmc: float`（既定 0.9 [無次元]；有効範囲：`[0, 1)`；DDMC entry に用いる scattering-ratio gate。`0` で無効化できる。NUMERICS §7.1参照）
  - `leak_stencil: Literal["4","9_kershaw"]`（既定 `"9_kershaw"`；`"4"` = 直接隣接4面のみ（Densmore近似、直交格子向け、NUMERICS §7.3.4）、`"9_kershaw"` = Kershaw 9点差分（歪格子対応、NUMERICS §7.3.5, Appendix A参照））
  - `interface_method: Literal["asymptotic_diffusion_limit","marshak"]`（既定 `"asymptotic_diffusion_limit"`；IMC⇄DDMC境界変換方式。`"marshak"` は精度が低いが安定（Densmore 2007参照）。将来拡張: `"cleveland_gentile"` 2D RZ幾何拡張。NUMERICS §7.7参照）
  - `emissivity_preserving: bool`（既定 True；Densmore 2006 の \(\hat{P}\) 補正を使用。Falseで標準P、NUMERICS §7.7.3参照）
  - `interface_exit_distribution: Literal["cosine","half_isotropic"]`（既定 `"cosine"`；DDMC→IMCリーク時の角度分布。`"cosine"`: \(P(\mu) = 2\mu\) (\(\mu \in [0,1]\))（物理的に正しい拡散流束分布）。`"half_isotropic"`: \(P(\mu) = 1\) (\(\mu \in [0,1]\))（簡略化）。NUMERICS §7.7参照）
  - `rz_face_r_weight: bool`（既定 True；2D_RZでDDMC→IMCリーク面位置サンプリングにR重み付けを使用。Falseで一様サンプル。1D_SPHでは無視（面は球面のため一意）。NUMERICS §7.7.2参照）
  - `face_opacity_temperature: Literal["radiative_mean","arithmetic_mean"]`（既定 `"radiative_mean"`；DDMCリーク係数算出時の面温度規約。`"radiative_mean"` = 放射温度による重み平均、`"arithmetic_mean"` = 隣接セル温度の算術平均。NUMERICS §7.3.2参照）
  - `m_matrix_check: bool`（既定 True；True の場合、M-matrix 条件不合格セルは DDMC ではなく IMC モードにフォールバック。False の場合は §7.1 の M-matrix 条件をスキップ（検証テスト用、歪格子で不安定になりうる））
  - `momentum_deposition: bool`（既定 True；診断のみ（output-only）。hydro 運動量へのフィードバックなし。HDF5 に `radiation/momentum_dep` として出力）
  - `tau_ddmc_off: float`（既定 -1.0 [無次元]；DDMC脱出τ閾値。`< 0` で `tau_ddmc` と同値。有効時 `0.5 ≤ tau_ddmc_off ≤ tau_ddmc`。NUMERICS §7.1.3参照）
  - `omega_ddmc_off: float`（既定 -1.0 [無次元]；DDMC脱出ω閾値。`< 0` で `omega_ddmc` と同値。NUMERICS §7.1.3参照）
  - `mode_hold: int`（既定 0；IMC→DDMC遷移前の最小滞留ステップ数。0でヒステリシスなし。有効範囲：`[0, 100]`。NUMERICS §7.1.3参照）
  - `rate_max: float`（既定 1e30 [無次元]；|Δτ/τ|の最大許容変化率。1e30で事実上無制限。有効範囲：`> 0`。NUMERICS §7.1.3参照）
- `diffusion: dict`
  - `enabled: bool`（既定 False；1D_SPH の diffusion-cell 分類、entry/exit energy conversion、cell-local matter-radiation source solve、RKL2 空間 diffusion step を有効化する。分類セルと guard セルは DDMC から除外される。2D_RZ では無視。NUMERICS §7.1.2a-d参照）
  - `tau_on: float`（既定 5.0 [無次元]；diffusion entry の Rosseland 光学厚閾値。有効範囲：`tau_on >= tau_off > 0`）
  - `tau_off: float`（既定 3.0 [無次元]；diffusion exit の Rosseland 光学厚閾値）
  - `reduced_flux_on: float`（既定 0.15 [無次元]；entry の reduced flux 上限。有効範囲：`0 <= reduced_flux_on <= reduced_flux_off <= 1`）
  - `reduced_flux_off: float`（既定 0.25 [無次元]；exit の reduced flux 上限）
  - `mode_hold: int`（既定 0；entry 条件を満たしてから diffusion へ入るまでの保持ステップ数。0で即時 entry）
  - `rate_max: float`（既定 1e30 [無次元]；entry 時の \(|\Delta\tau_R/\tau_R|\) 上限。1e30で事実上無制限）
  - `mode_update_interval: int`（既定 10；新規 diffusion entry と island filter を評価する step 間隔。`1` で毎 step 評価。既存 diffusion セルの hard exit（void/radiation floor/\(\tau_R<\tau_{off}\)/\(R_F>R_{F,off}\)）は毎 step 評価し、この間隔を待たない。有効範囲：`>= 1`）
  - `min_diffusion_island_cells: int`（既定 5；1D global cell index で連続する diffusion 候補 island がこのセル数未満なら diffusion entry を禁止する。guard セルは island size に数えず、island filter 後に生成する。`1` で island filter を実質無効化。有効範囲：`>= 1`）
  - `imc_guard_cells: int`（既定 1；diffusion セルの周囲で DDMC を禁止し IMC とする guard セル幅。有効範囲：`>= 1`）
  - `sts_max_stages: int`（既定 0；1D_SPH RKL2 diffusion の最大 stage 数。0 は上限なし、正値では超過時に subcycle。必要 subcycle 数が 10 を超える場合はその step の diffusion mask を IMC へ戻す。NUMERICS §7.1.2d参照）
  - `sts_damping: float`（既定 0.05 [無次元]；RKL2 Legendre 引数の damping。有効範囲：`(0, 1)`）
  - `sts_subcycle_eta: float`（既定 0.8 [無次元]；RKL2 stage 数見積もりと max-stage subcycle の安全係数。有効範囲：`(0, 1]`）
  - `interface_particles_per_face_group: int`（既定 32；diffusion-IMC interface の outgoing face-current \(J^{out}_{f,g}\) を IMC 粒子へ変換する際の face×group あたり生成粒子数。有効範囲：`>= 1`）
  - `exit_particles_per_cell_group: int`（既定 32；diffusion exit 時に cell×group あたり生成する IMC 粒子数。有効範囲：`>= 1`）
  - `lte_entry_initialization: bool`（既定 False；将来の LTE entry 初期化用予約値）
  - `lte_entry_energy_fraction_cap: float`（既定 0.01 [無次元]；将来の LTE entry 初期化で物質エネルギーから移す上限割合。有効範囲：`>= 0`）
- `holo: dict`（High-Order Low-Order 放射輸送の internal/test-only 設定。production namelist では `holo` subblock を設定しない。`enabled=False` では既存 IMC/DDMC/PGRW/hybrid diffusion の runtime 経路を変更しない。NUMERICS §7.1.2g 参照）
  - `enabled: bool`（既定 False；HOLO 有効化フラグ。v1 の LO solver は `Main.dimension="1D_SPH"` のみ対応。`2D_RZ` で True を指定した場合は、`sn_material_coupling=True` の deterministic GPU \(S_N\) material coupling 経路だけを有効とし、それ以外は WARNING を出して False に無効化し、凍結設定にも False として記録する）
  - `region: Literal["shell"]`（既定 `"shell"`；v1 は shell 領域のみ。将来 PR で材料グループや質量座標 interval へ拡張予定）
  - `material_group: Literal["shell"]`（既定 `"shell"`；v1 は shell material coupling mask のみ）
  - `coupling_tau: float`（既定 5.0 [無次元]；LO material-coupling mask の Rosseland 光学厚閾値。有効範囲：`>= 0`）
  - `guard_cells: int`（既定 3；`coupling_tau` で選ばれた cell mask の膨張 cell 半幅。有効範囲：`>= 0`）
  - `solver: Literal["implicit_1d","quasidiffusion_1d"]`（既定 `"implicit_1d"`；v1 の low-order solver 名。`quasidiffusion_1d` では high-order \(P_{rr}/E\) closure を使用する）
  - `closure: Literal["diffusion"]`（既定 `"diffusion"`；v1 closure）
  - `closure_relax: float`（既定 0.2 [無次元]；`quasidiffusion_1d` の filtered closure temporal relaxation weight。有効範囲：`[0, 1]`）
  - `closure_smooth_passes: int`（既定 1；`quasidiffusion_1d` の \(P_{rr}/E\) closure に適用する same-material spatial smoothing pass 数。有効範囲：`>= 0`）
  - `closure_smooth_alpha: float`（既定 0.5 [無次元]；各 spatial smoothing pass の neighbor mixing weight。有効範囲：`[0, 1]`）
  - `consistency_alpha: float`（既定 1.0 [無次元]；same-step predictor-corrector consistency source を LO corrector RHS へ入れる緩和係数。有効範囲：`[0, 1]`。互換用 key `gamma_alpha` も同じ値として受理する）
  - `boundary_flux: Literal["physical"]`（既定 `"physical"`；global LO solve は inner reflect / outer vacuum の物理境界だけを使う）
  - `p_rr_tally: bool`（既定 True；HOLO 有効時に passive radial pressure moment \(P_{rr}\) track-length tally と coverage/chi 診断を出力する。`enabled=False` では runtime 経路に影響しない）
  - `sn_closure: bool`（既定 True；`solver="quasidiffusion_1d"` の QD closure に 1D spherical \(S_N\) deterministic closure を使う。`solver="implicit_1d"` では runtime 経路に影響しない。False の場合は従来の passive MC \(P_{rr}/E\) tally closure を使う）
  - `sn_n_angles: int`（既定 8；\(S_N\) angular quadrature order。偶数かつ `>= 2`）
  - `sn_material_coupling: bool` (default False; when True, run the GPU \(S_N\) solve once per radiation step. In 1D_SPH with `solver="quasidiffusion_1d"`, the GPU \(S_N\) solve is used as a chi-only HO closure, and the existing QD LO solver updates `State.holo_E_LO`, `State.holo_F_LO`, `State.Te`, `State.ee`, and `State.Pe`. In 2D_RZ, the path publishes the deterministic material source as `State.holo_rad_dep - State.holo_rad_emit`, and material source injection applies this deterministic source instead of particle `rad_dep/rad_emit`. MC transport continues for diagnostics/validation but does not own material energy. Requires `sn_closure=True`; supports 1D_SPH and 2D_RZ.)
  - `residual_particles_per_cell_group: int`（既定 4；将来の residual particle 診断/制御用予約値。有効範囲：`>= 1`）
  - 互換用 deprecated keys: `q_min`, `q_max`, `tau_on`, `tau_off`, `reduced_flux_on`, `reduced_flux_off`, `update_interval`, `hold_on`, `min_dwell_steps`, `min_island_cells`, `core_margin_cells` は parse/freeze されるが、v1 の LO material-coupling mask では使用しない。
- `boundary: dict`
  - `type: Literal["vacuum","reflect","marshak"]`（既定 `"vacuum"`）
  - `inner: Optional[Literal["vacuum","reflect","marshak"]]`（1D_SPH内側境界；既定 `"reflect"`（r=0対称））
  - `outer: Optional[Literal["vacuum","reflect","marshak"]]`（1D_SPH外側境界；既定は `type` の値）
  - `marshak_Tr_eV: Optional[Union[callable, dict[str, callable]]]`（境界放射温度関数 \(T_{r,f}(t)\) [eV]、検証用。1D_SPH: 単一 `callable`（外側境界に適用）。2D_RZ: `dict[str, callable]`（面ごとに指定、例: `{"r_outer": T_func}`、§6.4.5 参照）。各callableのシグネチャ：`T(t_s: float) -> float` [eV]。初期化時に FrozenTable1D 化（サンプリング: \(t \in [0, t_{end}]\) を10000等間隔点で評価し、線形補間テーブルとして凍結）。**端点外挿規則**：\(t < 0\) および \(t > t_{end}\) では \(T_r = 0\) とする（Marshakソース停止と等価））
  - `marshak_Tr: Optional[callable]`（Time callable T_r(t) [eV] for the marshak boundary drive; frozen to a table at init. Consumed by the 2D IMC emitter AND (since 2026-07-09) the deterministic 1D FLD/SN marshak outer boundaries (indirect-drive mode). A positive `marshak_Tr_eV` takes precedence. For 1D, exactly one of {`marshak_Tr_eV`, `marshak_Tr`, grey `marshak.flux_erg_per_cm2_s`} must be set when `outer_r="marshak"`.）
  - `marshak_particles: int`（既定 1000；有効範囲：`≥ n_marshak_faces`（Marshak境界面数。最低1粒子/面保証のため）；Marshak境界のソース粒子数/ステップ。`Radiation.enabled=False` の場合は無視される。`marshak_particles < n_marshak_faces` の場合は `ConfigError("marshak_particles must be >= number of Marshak boundary faces")`。NUMERICS §8.2参照）
  - **境界タイプ別の必須パラメータ**：`"vacuum"` / `"reflect"` は追加パラメータ不要。`"marshak"` は `marshak_Tr_eV`（callable: \(T_{r,f}(t)\) [eV]）が必須。未指定の場合は `ConfigError`
  - **境界型の優先順位 (1D_SPH)**：
    - `type` は `inner` と `outer` の両方のデフォルト値を設定する
    - `inner` と `outer` が個別に指定された場合、`type` を上書きする
    - `inner` のデフォルトは常に `"reflect"` (1D_SPH の原点対称性)。
      `inner` に `"reflect"` 以外が指定された場合は `ConfigError("1D_SPH inner boundary must be reflect (r=0 symmetry)")`
    - 全三つが指定された場合: `inner` と `outer` の個別値を使用し、`type` は無視
    - 例: `type="vacuum", inner="reflect", outer="marshak"` → inner=reflect, outer=marshak

**2D_RZ 境界指定**：

2D_RZ では4辺の境界を個別に指定する：

```python
# 放射境界（Radiation block内）
Radiation(
    boundary={
        "r_inner": "reflect",       # r=0 対称軸（既定、変更不可）
        "r_outer": "vacuum",        # r=r_max（既定 "vacuum"）
        "z_bottom": "vacuum",       # z=z_min（既定 "vacuum"）
        "z_top": "vacuum",          # z=z_max（既定 "vacuum"）
    },
)
```
有効値：`"vacuum"`, `"reflect"`, `"marshak"`
- `r_inner` は常に `"reflect"`（対称軸のため変更不可）
- Marshak境界の場合、`marshak_Tr_eV` を面ごとに指定可能（旧名称 `marshak_T` は非推奨、`marshak_Tr_eV` に統一）：
  `marshak_Tr_eV: dict[str, callable]` = `{"r_outer": callable, "z_bottom": callable}`
  各値の署名: `T(t_s: float) -> float` [eV]。初期化時に FrozenTable1D 化（サンプリング: \(t \in [0, t_{end}]\) を10000等間隔点で評価し、線形補間テーブルとして凍結）。**端点外挿規則**：\(t < 0\) および \(t > t_{end}\) では \(T_r = 0\) とする（Marshakソース停止と等価）。

```python
# 流体境界（Numerics block内）
Numerics(
    hydro={
        "boundary": {
            "r_inner": "axis",         # r=0（既定、変更不可：v_r=0 強制）
            "r_outer": "free",         # （既定 "free"）
            "z_bottom": "free",        # （既定 "free"）
            "z_top": "free",           # （既定 "free"）
        },
    },
)
```
有効値：`"free"`, `"fixed"`, `"reflect"`, `"pressure"`
- `"free"`：ゴースト圧力 P = 0、速度は外挿
- `"fixed"`：境界ノード速度 \(v=0\) 固定。ノード位置も固定（静止壁）
- `"reflect"`：スリップ壁（法線速度 \(v_n = 0\)、接線速度は自由）
- `"pressure"`：P_ghost = `boundary_pressure(t)` で指定
- `r_inner` は常に `"axis"`（v_r=0、v_z自由）

**命名統一**: hydro と radiation の両方で `'reflect'` を使用する。
- hydro `'reflect'`：スリップ壁（\(v_n = 0\)、\(v_t\) 自由）
- radiation `'reflect'`：光子方向の鏡面反射
- v1.0 では後方互換のため `'reflective'` も受け入れ、`'reflect'` に変換し WARNING 出力。

#### 6.4.6 Laser(...)
- `enabled: bool`（既定 True；False の場合：レイトレース不実行、`laser_dep=0`、LaserMesh は確保されない。レーザーは独立Δt制約を持たないため（NUMERICS §2.2(d)）、無効化によるΔt変化はない。laser_pattern 診断は自動無効化。LaserBeam 定義は無視される）
- `wavelength_nm: float`（例 351.0 [nm]；有効範囲：`> 0`。内部で cm に変換（`λ_cm = wavelength_nm × 1e-7`）。ICF典型値：351 nm（3ω）、527 nm（2ω）、1053 nm（1ω）。NUMERICS §5.4参照）
- `mode: Literal["raytrace_2d","raytrace_3d","spherical_average","radial_absorption_1d"]`（1D_SPH既定 `"raytrace_2d"`、2D_RZ既定 `"raytrace_3d"`；1D_SPH では `"raytrace_2d"` または `"radial_absorption_1d"` を受理し、`"spherical_average"` は全ビーム同一条件の場合の検証・回帰テスト用互換モードとして受理する。2D_RZ で `"radial_absorption_1d"` または `"raytrace_2d"` を指定した場合は `ConfigError`。1D_SPH で `"raytrace_3d"` を指定した場合は `ConfigError`。§5.4.1、§5.4.1a、§5.4.2参照）
- `beams: list[LaserBeam]`（12本など；有効範囲：`≥ 1`。`Laser.enabled=True` の場合は必須（空リスト不可）。各ビームは `LaserBeam(...)` で定義）
- `rays_per_beam: int`（既定：1D_SPH=1000、2D_RZ=128；有効範囲：`≥ 10`。1D_SPHではビーム軸対称1D配列のレイ本数、2D_RZではビーム断面2D配列の1辺あたりのレイ本数（総数は \(\sim \pi/4 \times N^2\)））
- `ray_output_count: int`（既定 0；有効範囲：`≥ 0`。HDF5 snapshot `laser/rays/` に出力するレイ本数。`0` で無効。`Laser.enabled=True` の場合は `ray_output_count <= rays_per_beam` が必須。1D_SPH は代表ビームごと、2D_RZ は beam_group ごとに先頭 `ray_output_count` 本の初期状態を記録。`radial_absorption_1d` ではレイが存在しないため `laser/rays/` は空/無効）
- `ray_output_trajectory: bool`（既定 False）レイ軌跡出力の有効化。True にすると、スナップショット出力時にレイの各積分ステップの位置と残存パワーを `laser/rays/trajectory/` に書き出す。`ray_output_count > 0` が必須。`radial_absorption_1d` では無効
- `ray_output_max_steps: int`（既定 10000；有効範囲：`[1, 100000]`）レイ1本あたりの最大記録ステップ数。超過した場合は打ち切り（最大ステップまでの軌跡のみ記録）
- `absorption: dict`
  - `model: Literal["inverse_bremsstrahlung"]`（既定 `"inverse_bremsstrahlung"`；v1.0では唯一のオプション。NUMERICS §5.4参照）
  - `critical_handling: dict`
    - `eps_n: float`（既定 1e‑4 [無次元]；有効範囲：`(0, 0.1]`；屈折率下限 \(n_{refr} \ge \varepsilon_n\)。NUMERICS §5.3参照）
    - `eps_crit: float`（既定 1e‑4 [無次元]；有効範囲：`(0, 0.1]`；臨界密度終了判定。`critical_margin < 1 - eps_crit` の場合は **ConfigError**（WARNINGではなくエラー）。NUMERICS §5.2参照）
    - `terminate: bool`（既定 True；v1.0では **True固定**。False は将来版用（臨界面付近の全反射モデル）。v1.0で False を指定した場合は `ConfigError("terminate=False is not supported in v1.0")`。NUMERICS §5.2参照）
  - `coulomb_log_floor: float`（既定 2.0 [無次元]；有効範囲：`[1.0, 30.0]`；IB吸収固有のクーロン対数 lnΛ 下限。未指定時は `Numerics.coulomb_log_floor` にフォールバック。NUMERICS §5.4参照）
  - `debug_dump_lasermesh: bool`（既定 False；デバッグ用。True の場合、2D_RZ `raytrace_3d` で LaserMesh の \(T_e\), \(Zbar\), \(\hat n\), raw \(\hat n\), `smooth_kappa_factor` の一回限りの統計ダンプを出力する。物理・沈着結果には影響しない）
- `ib: dict`（拡張 IB 物理。`zeff_model` と `langdon_model` は既定 `"auto"`（構築時解決）、`coulomb_log_model` と `ra` は既定 OFF。v1 は 1D_SPH 限定。非対応の `"auto"` はエラーなく `"off"` に解決されるため、非 1D deck は影響を受けない。NUMERICS §5.4.5 参照。2026-07-30 追加）
  - `zeff_model: Literal["auto","off","sequential_strip","table"]`（既定 `"auto"` = 材料が TMAT `/ionization` を提供していれば `"table"`、なければ `"off"` に構築時解決（解決時に INFO ログ 1 行）。表なし run は従来とビット同一；混合プラズマ衝突電荷 \(Z_{\rm eff}=\langle Z^2\rangle/\langle Z\rangle\) の評価法。`"sequential_strip"` は `species` とセル \(\bar Z\) から核電荷昇順の逐次剥離クロージャで評価（完全電離極限で厳密）。`"table"` は TMAT 材料の `/ionization` ブロックから縮約した \(Z_{\rm eff}/\bar Z\) 表（log-log 双線形・端クランプ）で評価 — `/ionization` を持つ tmat 材料がちょうど 1 つ必要（無い場合 ConfigError）、かつ表の (nᵢ, T) 両格子は log 一様必須）
  - `species: list[list[float]]`（既定 `[]`；`[[Z_nuc, x_number], ...]` 最大 4 種、Z_nuc 狭義昇順、x>0、Σx∈[0.99,1.01]。`zeff_model="sequential_strip"` で必須。Langdon の \(Z_{\rm coll}\) は species があればそれを使い、無ければ最外非 void セルの \(\bar Z\) で代用）
  - `coulomb_log_model: Literal["debye","laser_frequency"]`（既定 `"debye"` = 従来式（引数 ∝ n̂^{-1/2}）。`"laser_frequency"` は \(b_{max}\sim v_T/\omega_0\) 規約で引数から n̂^{-1/2} を除く。フロアは `coulomb_log_floor` を共用）
  - `langdon_model: Literal["auto","off","legacy_vacuum_map"]`（既定 `"auto"` = 1D_SPH のレイトレース構成（laser 有効・`radial_absorption_1d` 以外・全ビームが共通の gaussian/super_gaussian/flat_top プロファイル）なら `"legacy_vacuum_map"`、それ以外は `"off"` に構築時解決（解決時に INFO ログ 1 行）。明示 `"legacy_vacuum_map"` は非対応構成で ConfigError。`"legacy_vacuum_map"` は Langdon 1980 の f_L(α) を、全ビーム合計パワーの共通プロファイル真空強度（gaussian / super_gaussian / flat_top 対応）でレイ現在 R において評価して適用）
  - `langdon_te_min_eV: float`（既定 100.0 [eV]；有効範囲 [0, 1e5]；この Te 未満では f_L=1（適用域制限 — 高衝突の冷物質では超ガウス歪みが成立しない））
- `ra: dict`（転回点共鳴吸収イベント。既定 OFF。NUMERICS §5.4.5(d) 参照。2026-07-30 追加）
  - `enable: bool`（既定 False；レイの球半径最小通過で 1 回、Ginzburg/Colaitis 縮約曲線 \(f_p(\eta)\)、\(\eta=(k_0L_n)^{2/3}(b/r_c)^2\) の分率を除去し臨界隣接亜臨界セルへ堆積。診断 `laser/E_ra`・`energy/laser_ra_deposited` に累積）
  - `chi_p: float`（既定 0.5；有効範囲 [0,1]；p 偏光分率（球面方位平均 = 0.5））
  - `c_ra: float`（既定 1.0；有効範囲 [0,10]；全体乗数（感度試験用））
- `profile: dict`（既定：ビーム個別設定で上書き可；全ビーム共通のビーム強度プロファイル）
  - `model: Literal["gaussian","super_gaussian","flat_top","custom"]`（既定 `"gaussian"`）
  - `w0_um: float`（1/eビームウェスト半径 [µm]；`model="gaussian"` または `"super_gaussian"` の場合は必須。他のモデルでは無視。有効範囲：`> 0`）
  - `m: int`（super-Gaussian指数；`model="super_gaussian"` の場合のみ使用、既定 2。有効範囲：`≥ 1`。m=1 は gaussian と等価）
  - `radius_um: float`（flat_topビーム半径 [µm]；`model="flat_top"` の場合は必須。有効範囲：`> 0`）
  - `func: Optional[callable]`（`model="custom"` の場合は必須；シグネチャ：`I(r_transverse_cm: float) -> float`（相対強度、無次元）。初期化時にテーブル化される。`model` が `"custom"` 以外の場合は無視）
- `spot: dict`（**非推奨（deprecated）**：内部で `profile` に自動変換される。後方互換のため残す。変換規則は `LaserBeam.spot` と同一（下記参照）。常に deprecation WARNING を出力）
- `lasermesh: dict`
  - `enabled: bool`（既定 True）
  - `nr: int`（既定 128；有効範囲：`≥ 16`；R方向メッシュ数。1D_SPHでは動的サイジングにより上書きされる（§5.7.1参照）。2D_RZでは静的指定値を使用）
  - `nz: int`（既定 256；有効範囲：`≥ 16`；Z方向メッシュ数。1D_SPHでは `nz = 2 × nr`（動的）。2D_RZでは静的指定値を使用）
  - `r_max: float`（R方向の最大値 [cm]；既定 \(1.5 \times R_{target}\)。2D_RZのみ使用）
  - `z_min: float`（Z方向の最小値 [cm]；既定 \(Z_{center} - 1.5 \times R_{target}\)。2D_RZのみ使用）
  - `z_max: float`（Z方向の最大値 [cm]；既定 \(Z_{center} + 1.5 \times R_{target}\)。2D_RZのみ使用）
  - **R_target / Z_center の既定値**：R_target = Mesh.r_max (1D_SPH) or max{r : ρ(r) > ρ_floor × 10} (2D_RZ)。Z_center = (Mesh.z_min + Mesh.z_max) / 2 (2D_RZ) or 0 (1D_SPH)。
  - `r_max_factor: float`（既定 1.5；有効範囲：`[1.0, 5.0]`；動的メッシュサイジング時の R_max スケール係数。1D_SPHで使用。NUMERICS §5.7.1参照）
  - `mesh_factor: float`（既定 0.5；有効範囲：`(0, 2.0]`；1D_SPH の動的gradedメッシュで最小セル幅 \(dR_{fine}\) を設定する係数。\(dR_{fine}=\) `mesh_factor` × 臨界面近傍の局所最小流体セル幅（§5.7.2）。小さいほど高解像度だがコスト増）
  - `rmax_n_hat_threshold: float`（既定 0.001；有効範囲：`(0, 1.0)`；R_max を決定する正規化電子密度閾値。\(\hat n \ge\) この値を満たす最外セルの外側エッジからR_maxを決定する。1D_SPHの動的サイジングで使用。NUMERICS §5.7.1参照）
  - `nr_max: int`（既定 4096；有効範囲：`>= 4`；1D_SPH の動的gradedメッシュにおける R方向セル数のハード上限。GPU OOM 安全策。`nz=2*nr`）
  - `stretch: dict`（格子ストレッチ設定。**v1.0では実質legacy**：1D_SPHでは臨界面中心のpiecewise-geometric gradedメッシュを使用し、`stretch` パラメータはメッシュ生成に影響しない）
    - `enabled: bool`（既定 True）
    - `method: Literal["density_gradient"]`（既定 `"density_gradient"`；後方互換のため受理。1D_SPH の実装では内部固定の graded 生成（`g_core=1.08`, `g_corona=1.05`）を使用）
    - `min_ratio: float`（既定 0.2；有効範囲：`(0, 1]`；後方互換パラメータ。1D_SPH 動的gradedメッシュでは未使用）
  - `critical_clip: bool`（既定 True；臨界密度以下の領域のみをカバーする）
  - `critical_margin: float`（既定 `1 - eps_crit`（= 0.9999 when eps\_crit=1e‑4）；有効範囲：`(0, 1)`；\(n_e/n_{crit}\) がこの値以下の領域をカバー。**整合性要件**（validate で **ConfigError**）: `critical_margin ≥ 1 - eps_crit` でなければならない。既定値は `eps_crit` に追随するため、ユーザが `eps_crit` のみ変更すれば自動整合する。明示指定時は整合性を検証する。NUMERICS §5.7.1参照）
  - `ghost_corona: dict`（ゴーストコロナ設定。1D_SPH シャープギャップ対応。NUMERICS §5.7.5参照）
    - `enabled: bool`（既定 False；True にすると LaserMesh 上に合成亜臨界コロナプロファイルを構築し、void 側吸収パワーを内向きステンシルで再分配する。1D_SPH 専用）
    - `n_out: int`（既定 12；有効範囲：`≥ 1`；ゴーストセル数）
    - `ne_min_frac: float`（既定 0.03；有効範囲：`(0, 1)`；ゴースト密度下限比 \(\hat{n}_{\min}\)）
    - `ne_max_frac: float`（既定 0.99；有効範囲：`(ne_min_frac, 1)`；ゴースト密度上限比 \(\hat{n}_{\max}\)）
    - `Te_min_eV: float`（既定 50.0；有効範囲：`> 0`；ゴースト電子温度下限 [eV]）
    - `zbar_min: float`（既定 1.0；有効範囲：`> 0`；ゴースト Z̄ 下限）
    - `zbar_max: float`（既定 4.0；有効範囲：`≥ zbar_min`；ゴースト Z̄ 上限）
    - `handoff_cells: int`（既定 4；有効範囲：`≥ 1`；内向きハンドオフステンシルの深さ。NUMERICS §5.7.5.3参照）
    - `handoff_decay: float`（既定 1.5；有効範囲：`> 0`；ステンシル指数関数減衰長。NUMERICS §5.7.5.3参照）
    - `transition_enabled: bool`（既定 False；True にするとブローオフ遷移モデルを有効化。NUMERICS §5.7.5.4参照）
    - `transition_resolved_nhat: float`（既定 0.9；有効範囲：`> 0`；resolved-corona 判定閾値 \(\hat{n}_{\text{resolved}}\)。`transition_enabled=True` 時のみ使用）
    - `transition_resolved_cells: int`（既定 3；有効範囲：`≥ 1`；ベースライン復帰に必要な連続亜臨界セル数 \(N_{\text{required}}\)。`transition_enabled=True` 時のみ使用）
    - `transition_density_exponent: float`（既定 1.0；有効範囲：`≥ 0`；密度バイアス指数 \(\alpha_\rho\)。0 で密度バイアス無効。`transition_enabled=True` 時のみ使用）
- `raytrace: dict`
  - `integrator: Literal["leapfrog","rk2","rk4"]`（既定 `"leapfrog"`；v1.0では **leapfrog固定**。`"rk2"`/`"rk4"` は将来版予約で、v1.0では `ConfigError("raytrace.integrator must be 'leapfrog' in v1.0")`。NUMERICS §5.3.4参照）
  - `cfl_ray: float`（既定 0.8 [無次元]；有効範囲：`(0, 1]`；レイステップ制約 \(C_{ray} = \Delta s / \Delta x \le\) cfl_ray。NUMERICS §5.3.4 (d) 参照）
  - `gradient_interpolation: Literal["bilinear"]`（既定 `"bilinear"`；節点4点双線形補間。v1.0では唯一のオプション。NUMERICS §5.3.4参照）
  - `intensity_cutoff: float`（既定 1e-6 [無次元]；レイの最小強度カットオフ（初期パワー比）。`I < intensity_cutoff * I_0` で終了。`0` で無効。NUMERICS §5.2参照）
  - `ds_adapt_g_target: float`（既定 0.05 [無次元]；有効範囲：`(0, 1]`；適応ステップ幅制御の屈折角変化目標。1ステップで \(|\hat\nabla\hat n| \times \Delta s_{base}\) がこの値以下になるようステップ幅を拡大する。NUMERICS §5.3.2参照）
  - `ds_adapt_tau_target: float`（既定 0.05 [無次元]；有効範囲：`(0, 1]`；適応ステップ幅制御の光学的深さ目標。1ステップで \(\kappa \times \Delta s_{base}\) がこの値以下になるようステップ幅を拡大する。NUMERICS §5.3.2参照）
  - `ds_adapt_theta_target: float`（既定 0.04 [rad/step]；`<= 0` で無効；転回弧角度制御。1ステップのレイ回転角 \(|\hat\nabla\hat n|\Delta s/(2(1-\hat n))\) がこの値以下になるようステップ幅を制限する。g/τ target と異なり**縮小（\(m<1\)、床 \(10^{-4}\)）を許す**唯一のメトリクスで、転回弧のステップ数を格子非依存の \(\sim\pi/\theta_{target}\) 本に規格化する。2026-07-31 新設。NUMERICS §5.3.2参照）
  - `ds_adapt_max_factor: float`（既定 4.0 [無次元]；有効範囲：`[1.0, 100.0]`；適応ステップ幅の最大拡大倍率。`1.0` で適応無効化（全ステップが基準幅）。NUMERICS §5.3.2参照）
  - `debug_one_ray: bool`（既定 False；デバッグ用。True の場合、2D_RZ `raytrace_3d` の ray index 0 について各シミュレーションステップ先頭64セグメントまで、IB吸収計算の区間値を device `printf` で出力する。物理・沈着結果には影響しない）
- `raytrace_skip: dict`（レイトレースSkip最適化；プラズマ条件がゆっくり変化する場合に前回結果をパワースケーリングして再利用。NUMERICS §5.9参照）
  - `enabled: bool`（既定 True）
  - `threshold: float`（既定 0.01 [無次元]；有効範囲：`(0, 1)`；最大相対変化量δがこの値未満ならスキップ。小さいほど精度向上するがスキップ頻度が低下）
  - `max_consecutive: int`（既定 10；有効範囲：`[1, 100]`；強制再計算までの最大連続スキップ数。大きいほど性能向上するが蓄積誤差のリスク増）
  - `norm: Literal["max_relative","l2_relative"]`（既定 `"max_relative"`；変化メトリクスのノルム選択。`"max_relative"` = 最大セルの相対変化（保守的）、`"l2_relative"` = L2ノルム（緩い）。NUMERICS §5.9参照）
  - `crit_guard: float`（既定 0.01 [無次元]；有効範囲：`[0, 0.1]`；臨界近傍ガード閾値。いずれかのLaserMeshセルで n̂ > n̂\_margin − crit\_guard の場合は強制再計算。臨界面近傍ではn̂の微小変化がterminate有無を左右するためSkipは安全でない。0 に設定するとガード無効。NUMERICS §5.9.4参照）
  - **キャッシュ無効化条件**（いずれか1つでも満たされた場合、全ビームを再トレースする）：
    (a) \(|P_{total}(t) - P_{total}(t_{cached})| / \max(P_{total}(t), P_{total}(t_{cached}), P_{floor}) > 0.01\)
        ここで \(P_{floor} = 10^{-30}\) [erg/s]（両方ゼロ時のゼロ除算防止。両方ゼロの場合は沈着もゼロであり無効化不要）
    (a') **per-group 条件**（n\_groups > 1 の場合）：いずれかのグループ g で \(P_g(t_{cached}) = 0 \wedge P_g(t) > 0\) または \(P_g(t_{cached}) > 0 \wedge P_g(t) = 0\) が成立した場合は再トレース（グループ間パワー比変化による空間パターン不整合防止、NUMERICS §5.9.4準拠）
    (b) いずれかのビームの direction ベクトルの L2 ノルム変化 \(> 10^{-10}\)
    (c) 現ステップで ALE rezone が実行された
- `cbet: dict`（Cross-Beam Energy Transfer、Marozas/DRACO 型 v1。**1D_SPH + mode="raytrace_2d" または 2D_RZ + mode="raytrace_3d" 専用、単一 MPI rank 必須**。NUMERICS §5.10 参照）
  - `enable: bool`（既定 False；CBET 有効化。False で全既存挙動と bit 恒等。True は `(Main.dimension="1D_SPH" and Laser.mode="raytrace_2d")` または `(Main.dimension="2D_RZ" and Laser.mode="raytrace_3d")` のみ受理）
  - `f_cbet: float`（既定 1.0；有効範囲：`> 0`；偏光/較正係数 \(\eta_{pol}=f_{cbet}[1+(\hat k_q\cdot\hat k_p)^2]/4\)）
  - `alpha_iaw: float`（既定 0.2；有効範囲：`> 0`；無次元イオン音波減衰率 \(\nu_a/(|k_a|c_a)\)）
  - `theta_cap: float`（既定 0.3；有効範囲：`(0, 1)`；donor cap — 1 交差あたり相対損失上限）
  - `tol: float`（既定 1e-3；有効範囲：`> 0`；固定点反復の L1 収束判定 \(\sum|L^{(m)}-L^{(m-1)}|/\sum L^{(m-1)}\)）
  - `max_iters: int`（既定 50；有効範囲：`≥ 1`；反復上限。未収束でも保存的な最終 pass を実行し警告）
  - `n_impact_bins: int`（既定 16；有効範囲：`≥ 1`；impact parameter ビン数/枝/ビーム。1D グループ数 G = n_beams×2×n_impact_bins。2D では theta-group ごと、quadrant branch ごとの aperture-radius bin 数で、G = n_theta_groups×4×n_impact_bins）
  - `n_phi: int`（既定 8；有効範囲：`≥ 1`；1D 方位角平均の中点求積節点数。2D_RZ CBET では Z2 mirror closure を使うため inert）
  - `ne_frac_cutoff: float`（既定 0.95；有効範囲：`(0, 1]`；\(\hat n_{raw}\ge\) cutoff のセルは CBET 評価から除外）
  - `k_a_floor: float`（既定 1e-6；有効範囲：`> 0`；\(|k_a|/\bar k\) がこの値未満の求積節点は寄与ゼロ）
  - `max_segments_per_ray: int`（既定 0；有効範囲：`≥ 0`；経路記録容量/ray。0 で自動（1D: 2×n_cells+64、2D: 4×(lasermesh.nr+lasermesh.nz)+64）。溢れた ray は CBET 交換から外れ計数される）
  - `test_chi: float`（既定 -1.0；TEST HOOK — 正値で全ペア結合係数 χ̄ [cm·s/erg] を定数上書き。物理結果は無効）
  - `geometry_mode: Literal["legacy","port_section"]`（既定 `"legacy"`（現行経路と bit 恒等）；`"port_section"` = 単一トレース＋実ポート配置の sector/section 位相空間 CBET（設計 docs/design/multibeam_1d_superposition_20260727.md、Follett PoP 32,022709 (2025)）。**S2 で物理有効化済み（NUMERICS §5.10.8）**。要件: `Main.dimension="1D_SPH"`・`cbet.enable=True`・`Laser.port_configuration.ports` 非空・`len(Laser.beams)==1`（単一 prototype trace））
  - `n_section_phi: int`（既定 16；有効範囲：`[4, 64]`；`port_section` の方位 section 求積数。V5 収束 gate は 8/16/32）
- `port_configuration: dict`（既定 absent；多ビーム・ポート表 — `geometry_mode="port_section"` および hot-e overlap 縮約の入力。設計 §5）
  - `normalization: Literal["sum_weights_one"]`（既定・唯一値；`power_weight` 総和 = 1 を検証（|Σ−1| ≤ 1e-6））
  - `ports: list[dict]`（1..192。各 dict）:
    - `port_id: int`（必須；`≥ 0`・一意。正準順序はこの id の昇順 — 入力順の置換は結果 bitwise 不変）
    - `direction: list[float]`（必須；長さ 3・非零。parse 時に単位ベクトルへ正規化）
    - `roll_deg: float`（既定 0.0；ビーム軸周り回転）
    - `power_weight: float`（必須；`> 0`；時間波形は beams[0] の波形 × weight）
    - `delta_lambda_nm: float`（既定 0.0；ポート別デチューニング）
    - `beam_class: str`（既定 ""；同一クラス圧縮ラベル）
    - `polarization: Literal["unpolarized"]`（既定・v1 唯一値；Jones/Stokes は将来拡張）
  - **2D_RZ notes**：`cbet.enable=True` のとき theta-group fold key は `delta_lambda_nm` も含む。2D CBET では raytrace skip cache を v1 で無効化する（2D cache は group power を total row に潰すため）。CbetWorkspace は projected allocation が 8 GiB を超える場合に停止し、調整レバーは `rays_per_beam`, `n_impact_bins`, `max_segments_per_ray`, `lasermesh.nr/nz`。
- `hot_electron: dict`（1D hot-electron preheat。NUMERICS §5.11 参照）
  - `enable: bool`（既定 False；hot-electron preheat 有効化）
  - `source_nc_fraction: float`（既定 0.25；有効範囲：`(0, 1]`；source は \(n_e=\) fraction × \(n_{crit}\)）
  - `eta_hot: float`（既定 0.0；有効範囲：`[0, 0.95]`；capture power fraction）
  - `eta_hot_table: callable`（既定 absent；`t[s] -> eta`。初期化時に freeze され、検出時は `eta_hot` より優先）
  - `eta_mode: Literal["legacy","model"]`（既定 `"legacy"`；`"model"` で η(t) 物理モデル（NUMERICS §5.11.3）を有効化。`"legacy"` は constant/table 優先の従来経路と bitwise 恒等）
  - `eta_model: dict`（既定 absent；`eta_mode="model"` の共有 knob）：
    - `ln_filter_tau_s: float`（既定 5.0e-12；有効範囲：`> 0`；\(1/L_n\) EMA 時定数 [s]）
    - `eta_total_cap: float`（既定 0.08；有効範囲：`(0, 0.5]`；チャネル合計 η の上限（比例縮小））
  - `T_hot_eV: float`（既定 5.0e4；有効範囲：`> 0`；hot-electron temperature [eV]。keV ではない）
  - `n_energy_groups: int`（既定 30；有効範囲：`≥ 1`）
  - `E_min_over_Th: float`（既定 0.2；有効範囲：`> 0`）
  - `E_max_over_Th: float`（既定 8.0；有効範囲：`> E_min_over_Th`）
  - `angular_model: Literal["cone","radial"]`（既定 `"cone"`）
  - `theta_div_deg: float`（既定 60.0；有効範囲：`[0, 90]`；cone half-angle [deg]）
  - `n_mu: int`（既定 6；有効範囲：`≥ 1`；cone \(\mu\) Gauss-Legendre nodes）
  - `n_phi: int`（既定 8；有効範囲：`≥ 1`；cone azimuthal nodes）
  - `subtract_from_laser: bool`（既定 True；True で capture power を laser ray power から差し引く。multi-channel では sequential depletion — NUMERICS §5.11.1）
  - `inner_bc: Literal["deposit_residual","escape"]`（既定 `"deposit_residual"`；`angular_model="radial"` のみ）
  - `explicit_source_limit: float`（既定 0.2；有効範囲：`> 0`；hot-electron dt limiter safety factor）
  - `sources: list[dict]`（既定 absent；**機構別チャネル列**（1..4）。NUMERICS §5.11.1。上記スカラー shorthand 群（`source_nc_fraction`/`eta_hot`/`eta_hot_table`/`T_hot_eV`/`n_energy_groups`/`E_min_over_Th`/`E_max_over_Th`/`theta_div_deg`/`n_mu`/`n_phi`）との併用は parse エラー；shorthand は単一 `cone` チャネルと厳密等価（出力 bitwise））。各チャネル dict：
    - `mechanism: Literal["cone","tpd","srs"]`（既定 `"cone"`）
    - `capture_nc_fraction: float`（既定 cone/tpd 0.25、srs 0.18；有効範囲：`(0, 1]`；チャネル捕獲面）
    - `eta: float`（既定 0.0；有効範囲：`[0, 0.95]`）
    - `eta_table: callable`（既定 absent；`t[s] -> eta`。初期化時に freeze（tables key `laser.hot_e_eta_ch<i>`）、検出時は `eta` より優先）
    - **`eta_mode="model"` 専用チャネル knob**（NUMERICS §5.11.3；sentinel −1/空文字は parse 時に機構既定へ解決 — tpd: C=1.0/η∞=0.01/η_hard=0.03/`"vu2012"`、srs: C=8.0/η∞=0.08/η_hard=0.08/`"fixed"`）：
      - `eval_nc_fraction: float`（既定 `capture_nc_fraction`；有効範囲：`(0, 1)`；閾値パラメータ（T_e, L_n）評価面）
      - `threshold_multiplier: float`（既定 機構依存；有効範囲：`> 0`；閾値係数 \(C_k\)）
      - `eta_inf: float`（既定 機構依存；有効範囲：`(0, 1)`；漸近効率 \(\eta_\infty\)）
      - `eta_hard_cap: float`（既定 機構依存；有効範囲：`(0, 1)`；チャネル個別上限）
      - `shape_coefficient: float`（既定 1.0；有効範囲：`> 0`；飽和形状係数 \(a_k\)）
      - `relaxation_model: Literal["vu2012","fixed"]`（既定 機構依存；緩和時定数則）
      - `relaxation_tau_s: float`（既定 6.0e-12；有効範囲：`> 0`；`"fixed"` の τ [s]）
      - `relaxation_tau_min_s: float` / `relaxation_tau_max_s: float`（既定 3.0e-12 / 1.0e-11；`0 < min ≤ max`；`"vu2012"` の clip 範囲 [s]）
  - `tpd_overlap_mode: Literal["single_beam","common_wave_cluster"]`（既定 `"single_beam"`；`"common_wave_cluster"` = Michel 2013 Eq.(1) 等角クラスタの overlap 強度（設計 §4/§11、**S3 で物理有効化**）。要件: `port_configuration.ports` 非空・`eta_mode="model"`・`common_wave_delta_theta_deg` 指定）
  - `srs_overlap_mode: Literal["per_beam_class"]`（既定・v1 唯一値；SRS はクラス別駆動 — 全ビーム和を単一 pump にしない）
  - `illumination_metric: Literal["fixed","equivalent_area"]`（既定 `"fixed"`（f_illum=1 現行）；`"equivalent_area"` = \(f_{\rm illum}^{(2)}=(\int I\,d\Omega)^2/(4\pi\int I^2 d\Omega)\)。要件: ports 非空・`eta_mode="model"`）
  - `common_wave_delta_theta_deg: float`（既定 -1（未指定）；`common_wave_cluster` 時必須・有効範囲 `(0, 90]`。**普遍既定値なし（設計 §4）— 較正必須+half/nominal/double 感度解析**）
    - `T_hot_eV: float`（既定 cone 5.0e4、tpd 6.0e4、srs 4.5e4；有効範囲：`> 0`）
    - `n_energy_groups: int` / `E_min_over_Th: float` / `E_max_over_Th: float`（既定 30 / 0.2 / 8.0；スカラー版と同一の有効範囲）
    - `theta_div_deg: float`（既定 cone 60.0、srs 20.0；有効範囲：`[0, 90]`；**cone/srs のみ** — tpd チャネルに与えると parse エラー）
    - `tpd_theta_deg: float`（既定 45.0；有効範囲：`(0, 90]`；**tpd のみ**；環中心極角）
    - `tpd_delta_deg: float`（既定 10.0；有効範囲：`[0, 90]`；**tpd のみ**；環半幅。\(\theta_c-\Delta<0\) は前方 cap へ折返し）
    - `n_mu: int` / `n_phi: int`（既定 6 / 8；`≥ 1`）
  - **制約**：`enable=True` は `Main.dimension="1D_SPH"`（`Laser.mode in {"radial_absorption_1d","raytrace_2d"}`）または `"2D_RZ"`（`Laser.mode="raytrace_3d"`、単一 MPI rank）を要求する。`Laser.cbet.enable` と相互排他（**例外: `cbet.geometry_mode=\"port_section\"` かつ `eta_mode=\"model\"` のみ併用可 — capture は CBET 後 power で実行、NUMERICS §5.10.8**）。1D では `angular_model="cone"` は `Mesh.geometry_1d="cylindrical"` で拒否される。2D_RZ では `angular_model="radial"` と明示 `inner_bc`（既定以外）は拒否される（1D 専用概念）。multiblock 格子は hote 層で対応済み（2026-07-17 — MeshView2D multiblock builder、三角セル含む全 scheme、unit-gated；NUMERICS §5.11.2）— ただし multiblock の載る `logical_mesh_2d="spherical_polar_halfplane"` が現在 hydro-only（Laser 自体を validate で拒否）のため、deck 到達性は上流 laser-on-spherical-polar フェーズ待ち。hot-e 有効時は raytrace-skip cache が無効化される。persistent kernel は `hot_electron` 有効時に拒否する。`sources` は最大 4 チャネル・空リスト不可；機構外 key（tpd 系 knob を cone/srs へ等）は parse エラー。frozen config には `sources` 使用時のみ `sources` ブロックが出力される（機構に応じた key 集合；未使用 deck の frozen 出力は byte 恒等）。**`eta_mode="model"` の追加制約**：`Main.dimension="1D_SPH"` のみ（2D_RZ は capture が segment-start power のため設計文書 §17.1 により拒否）；`sources` 必須（scalar shorthand 不可）；全チャネル `mechanism in {"tpd","srs"}`（`"cone"` 不可）；`subtract_from_laser=True` 必須；チャネル `eta`/`eta_table` の指定は拒否（model が η を所有）；frozen config には `eta_mode` と `eta_model` ブロック・チャネル model key 群は `eta_mode="model"` のときのみ出力（legacy frozen 出力は byte 恒等）。
- `deposit: dict`
  - `map: Literal["bilinear_node","conservative_overlap"]`（既定 `"bilinear_node"`；近傍4節点への双線形分配）
    - v1.0 で `"conservative_overlap"` を指定した場合: `ConfigError("conservative_overlap is not available until v1.1")` を送出。
  - `deposit_smooth_passes: int`（既定 0；有効範囲：`>= 0`。1D_SPH/2D_RZ の Hydro cell への転写時に適用する診断用の保存的 smoothing パス数。1D_SPH は mass-weighted、2D_RZ は 4 近傍 Jacobi smoothing。`0` で無効。NUMERICS §5.8.1 参照）
  - `deposit_smooth_alpha: float`（既定 0.25 [無次元]；有効範囲：`[0, 0.5]`。診断用 deposit smoothing の 1 パスあたり重み。`0` で無効。void/blocked 隣接セルでは自動的に抑制される。NUMERICS §5.8.1 参照）

`LaserBeam(...)`：
- `name: str`（任意；既定 `"beam_0"`, `"beam_1"`, ... 自動付番。有効文字: `[a-zA-Z0-9_-]`、最大長: 64）
- `direction: tuple[float,float,float]`（入射方向単位ベクトル (dx, dy, dz)；ビーム軸の方向。**必須**。初期化時にユニット長に正規化。\(|\mathbf{direction}| < 10^{-10}\) の場合は `ConfigError`。内部表現では極角 θ / 方位角 φ [deg] に変換して ARCHITECTURE §4.1 Config::LaserConfig の BeamDef.theta / BeamDef.phi に格納）
- `focus: tuple[float,float,float]`（焦点座標 [cm]；ビーム軸上の集光位置。`focus` と `defocus` が両方指定された場合は **`focus` が優先**）
- `f_number: float`（F値 [無次元]；**必須**。すべてのビームで指定が必要。focus/defocus 設定に関わらず、ビーム初期化と幾何計算に使用される。有効範囲：`[1.0, 50.0]`。レンズ焦点距離とレンズ径の比 \(f/D_{lens}\)。ICF典型値：1.0-10.0。NUMERICS §5.6参照）
- `defocus: Optional[float]`（デフォーカスパラメータ D/R（無次元）；`focus` 未指定時に使用。D = 集光位置とターゲット中心のビーム軸方向符号付き距離、R = ターゲット外半径。**符号規約**: D/R < 0：ターゲット手前に集光（over-focused、実験で最も一般的）、D/R > 0：ターゲット奥に集光（under-focused）、D/R = 0：ターゲット中心に集光。NUMERICS §5.6.5参照）
- `power: callable`（時間波形。**必須**。シグネチャ：`P_W(t_s: float) -> float`（入力 [s]、出力 [W]）。初期化時に FrozenTable1D に変換される。サンプリング: t ∈ [0, Main.t_end] を 10000 等間隔点で評価し、線形補間テーブルとして凍結。boundary_pressure と同じ方式（§6.4.7 参照））
  - **2026-07-12 doc-truth 訂正**: 旧記載の dict 形式 `{"type":"piecewise_linear","t_s":[...],"P_W":[...]}` は実装に存在しない（builder は callable のみ受理 — GUI Studio golden gate が SPEC/実装乖離を検出、M4 報告）。区分線形波形は callable で表現する（GUI Studio は同等の `_gui_pwl` callable を生成して deck に埋め込む）。dict 形式の実装追加は未計画（要望が出た時点で判断）
  - **端点外挿規則**：凍結範囲 \([0, t_{end}]\) 外は \(P = 0\) とする
- `energy_J: Optional[float]`（既定 `None`；与えた場合は `power` の波形形状を保ったまま積分エネルギーが `energy_J` に一致するようにリスケールする [J]。有効範囲：`> 0`。`power` は波形形状の指定として必須であり、`energy_J` と `power` の同時指定は正当な使用法）
  - energy_J が指定された場合のスケーリング手順:
    1. FrozenTable1D の 10000 点等間隔グリッド上で P(t) を台形公式で積分: E_computed = ∫₀^{t_end} P(t) dt [erg]
    2. スケール係数: s = energy_J × 10⁷ / E_computed  (J → erg 変換)
    3. P_scaled(t) = s × P(t)
    4. E_computed < 10⁻³⁰ erg の場合: ConfigError("zero-integral waveform cannot be normalized")
- `profile: Optional[dict]`（ビーム強度プロファイル；未指定時はLaser全体の `profile` を使用）
  - `model: Literal["gaussian","super_gaussian","flat_top","table","custom"]`
  - `w0_um: float`（1/eビームウェスト半径 [µm]）
  - `m: int`（super-Gaussian指数）
  - `r_um: list[float]`（`model="table"` 専用・必須；半径格子 [µm]、狭義単調増加・先頭 >= 0）
  - `I_rel: list[float]`（`model="table"` 専用・必須；相対強度 >= 0（少なくとも 1 点 > 0）。区分線形補間、r < r_um[0] は I_rel[0]、r > r_um[-1] は 0（有限ビーム径）。正規化は不要 — 解析モデル同様 ray 初期化側で規格化される。2026-07-17 導入、docs/design/laser_profile_table_20260717.md）
  - `radius_um: float`（flat_topビーム半径 [µm]）
  - `func: Optional[callable]`（custom用；`I(r_transverse_cm) -> I_relative`）
- `spot: Optional[dict]`（**非推奨（deprecated）**：内部で `profile` に自動変換。常に deprecation WARNING を出力）
  - **spot → profile 変換規則**：
    - `spot` と `profile` の両方が指定された場合: `ConfigError("spot and profile are mutually exclusive; use profile")`
    - `spot.model='gaussian'` → `profile = {'model': 'gaussian', 'w0_um': spot.radius_um}`
    - `spot.model='flat_top'` → `profile = {'model': 'flat_top', 'radius_um': spot.radius_um}`
    - `spot.model='super_gaussian'` → `profile = {'model': 'super_gaussian', 'w0_um': spot.radius_um, 'm': spot.m}`
- `rays_per_beam: Optional[int]`（ビーム個別のレイ本数；未指定時はLaser全体の `rays_per_beam` を使用）
- `delta_lambda_nm: float`（既定 0.0 [nm]；ビーム毎の波長 detuning \(\Delta\lambda_b\)。\(\omega_b=2\pi c/(\lambda_0+\Delta\lambda_b)\) として CBET の g にのみ入る（\(|k|\)・\(n_{crit}\) は共通 \(\lambda_0\) で凍結）。CBET 無効時は無視。NUMERICS §5.10.1 参照）
- `polarization: Optional[str]`（`"s"` / `"p"` / `"circular"` / `"unpolarized"`；将来実装用。v1.0 で指定した場合、値を frozen config に保存し無視（INFO メッセージ出力）。既定 `None`）
- `pointing_error: Optional[float]`（[μm]；将来実装用。v1.0 で指定した場合、値を frozen config に保存し無視（INFO メッセージ出力）。既定 `None`）
- **必須フィールドまとめ**：`power`、`direction`、`f_number`。`name`、`rays_per_beam` は任意

#### 6.4.7 Numerics(...)
- `dt: dict`
  - `initial_s: Optional[float]`（既定 `1e-15`；初期タイムステップ [s]。float指定時の有効範囲：`> 0`。ICF爆縮では初期の急速加熱で小さな値が必要。`None` 指定時は自動計算：\(\Delta t_0 = 0.1 \times \min_c(\Delta l_c / c_{s,c})\)（CFL安定性に基づく自動推定）。成長率 1.2 は内部定数）
  - `max_s: float`（既定 `1e-9`；タイムステップ上限 [s]。有効範囲：`≥ initial_s`）
  - `min_s: float`（既定 `1e-20`；タイムステップ下限 [s]。有効範囲：`> 0`。\(\Delta t < \text{min\_s}\) で FATAL 停止（ストーリング防止）。NUMERICS §2.2(e) 参照）
  - `growth_factor: float`（既定 `1.2` [無次元]；有効範囲：`(1.0, 2.0]`；連続ステップ間のΔt成長率上限 \(\Delta t^{n+1} \le growth\_factor \times \Delta t^n\)。NUMERICS §2.2参照）
  - `floor_stall_max_consecutive_steps: int`（既定 `0`；有効範囲：`>= 0`。0 で無効。正値では、commit 済み step の \(\Delta t\) が \((1+10^{-6})\text{min\_s}\) 以下に連続して張り付き、残り時刻が \(10^3\text{min\_s}\) より大きく、limiter が `output` / `t_end` でない場合に D4-class floor-stall として FATAL 停止する。NUMERICS §2.2(e) 参照）
  - `cfl_hydro: float`（既定 0.3 [無次元]；有効範囲：`(0, 1]`；流体CFL数。NUMERICS §2.2 (a) 参照）
  - `cfl_cond: float`（既定 0.25 [無次元]；有効範囲：`(0, 1]`；伝導CFL数。STSの内部明示的CFL限界 \(\Delta t_{exp}\) の係数として使用。グローバルΔtには \(s_{max}(s_{max}+1)/2\) 倍に緩和された \(\Delta t_{cond,sts}\) が寄与する。`conduction.enabled=False` の場合は無視。NUMERICS §2.2 (b), §4.2.1 参照）
  - `f_min_fleck: float`（既定 `0.01` [無次元]；有効範囲：`(0, 1]`；IMC側 Fleck factor下限によるΔt_rad制約。FLD側 Fleck は stiff-cell 極限を保つためこの下限を適用しない。`Radiation.enabled=False` の場合は無視。NUMERICS §2.2 (c) 参照）
  - **相互整合チェック（Phase 2）**：`dt.min_s >= dt.max_s` → `ConfigError("dt.min_s must be < dt.max_s (got min_s={min_s}, max_s={max_s})")`
- `persistent_loop: dict`（Persistent-loop v1 dispatch scaffolding。Phase C までは opt-in decision と scope refusal のみを行い、multi-kernel path を実行する）
  - `enabled: bool`（既定 False；persistent-loop v1 dispatch opt-in。False で全既存挙動と bit 恒等）
  - `chunk_steps: int`（既定 128；有効範囲：`≥ 1`；persistent-kernel launch あたり step 数 \(K\)）
- `debug: dict`
  - `trace_mesh_motion: bool`（既定 `False`；Phase 3 mesh-freeze diagnostic。True では first `trace_max_steps` hydro/ALE steps の mesh-motion trace を stderr に `[mesh-trace step=N stage=name] key=value ...` 形式で出力する。物理状態、kernel force assembly、HDF5 schema には影響しない）
  - `trace_mesh_node_selector: str`（既定 `"outer_equator"`；現状唯一の許可値。trace node は host 側で `argmax hypot(r,z) - 0.01*abs(z)` として選ぶ）
  - `trace_mesh_cell: int`（既定 `7`；bridge/canary and compatible-force diagnostic cell index。`>= 0`）
  - `trace_max_steps: int`（既定 `5`；trace 出力の最大 step 数。`>= 0`）
- `radiation_thermal_subcycle: bool`（既定 `False`；True のとき、`Radiation.imc.two_stage=False` の Radiation 演算子で compressed cell が \(T_{e,floor}+0.5\) eV 以内まで落ちた場合に、熱力学状態を復元して \(n_{sub}=2,4,8\) の順に再試行する。NUMERICS §2.1, §11.2 参照）
- `materials: dict`（Stage 32a Wave A; top-level `Materials(...)` とは別名前空間）
  - `per_material_conservation_enabled: bool`（既定 `False`；Wave F per-material conservation arrays の master switch。False では `State.mass_per_material`, `Ee_per_material`, `Ei_per_material`, `Te_per_material`, `Ti_per_material` と valid flags は size 0 のまま）
  - `presence_threshold_volfrac: float`（既定 `1.0e-10`；有効範囲 `> 0`；材料存在判定の体積分率閾値）
  - `presence_threshold_mass_density_g_per_cc: float`（既定 `1.0e-12` [g/cm³]；有効範囲 `> 0`；材料存在判定の密度下限）
  - `eos_table_validity_lower_bound_g_per_cc: dict[str,float]`（既定 `{}`；key は `Materials.materials[i].name` と一致。未指定 material は実装既定の generic floor `1.0e-7` [g/cm³] を用いる。値の有効範囲 `> 0`）
  - `lazy_cache_te_m_enabled: bool`（既定 `False`；derived `Te_m/Ti_m` lazy cache を確保する。cache は権威状態ではなく、`Ee_per_material/Ei_per_material` 更新で invalidation される）
  - `hdf5_emit_derived_per_material: bool`（既定 `False`；V22 derived per-material datasets 出力の予約 knob。Wave A では dataset は出力しない）
  - `deposit_redistribute_fallback_enabled: bool`（既定 `False`；radiation/source/laser 向け explicit fallback 予約 knob。`Q_ei` は fallback 対象外）
  - `deposit_redistribute_provenance_label: str`（既定 `"TENRYU_EXTENDED_ALE_WAVE_F_DEPOSIT_REDISTRIBUTE_FALLBACK"`；有効範囲：非空文字列）
  - `conservation_residual_warn_threshold_rel: float`（既定 `1.0e-12`；有効範囲 `> 0`）
  - `conservation_residual_hard_warning_threshold_rel: float`（既定 `1.0e-10`；有効範囲 `>= conservation_residual_warn_threshold_rel`）
- `hydro: dict`
  - `T_start_eV: float`（既定 `0.0` [eV]；有効範囲：`≥ 0`；Hydro開始温度閾値。セルごとに `T_e >= T_start_eV` を判定し、閾値未満のセルではHydro（メッシュ移動）をスキップする。0.0 = 全セル常時有効。一度活性化したセルは以降判定をスキップ（一方向スイッチ）。リスタート時は `hydro_active` フラグから復元（§7.4参照）。NUMERICS §2.1.1参照）
  - `axis_motion_floor_fraction: float`（既定 `0.0` [無次元]；有効範囲：`[0, 1]`；2D_RZ 専用の Lagrangian axis-row preflight。`0` で無効。`>0` では Predictor/Corrector の位置 commit 前に \(i=1\) row の radial motion を縮小し、各 axis-row cell の analytic margin が commit 前 margin のこの割合以上に残るようにする。NUMERICS §3.2.12a 参照）
  - `axis_margin_dt_floor_fraction: float`（既定 `0.0` [無次元]；有効範囲：`[0, 1]`；2D_RZ 専用の hydro dt limiter。`0` で無効。`>0` では候補 hydro dt の trial position で axis-row margin を評価し、各 axis-row cell の analytic margin が現在 margin のこの割合以上に残るように dt を縮小する。NUMERICS §3.2.13 参照）
  - `volume_rate_cfl_enabled: bool`（既定 `False`；2D_RZ 専用の post-hoc volume-rate CFL を有効化する。False では既存 hydro dt と bitwise 同一の default-off path。True では前回完了 hydro step の `vol_old`, current `vol_new`, `dt_used` から \(|\Delta V|/\max(V_{old},10^{-30})/dt_used\) を測定し、次 step の hydro dt を縮小する。初回 step または履歴なし sentinel では no-op。Accepted ALE rezone/remap unconditionally resets `vol_old` to current `vol_new` before the next hydro CFL evaluation, without changing `dt_used` and without a namelist control。NUMERICS §2.2, §3.2.13 参照）
  - `volume_rate_cfl_threshold: float`（既定 `0.5` [無次元]；有効範囲：`> 0`；`volume_rate_cfl_enabled=True` 時の許容 fractional volume change per hydro timestep。既定 0.5 は 50% volume change per dt を上限とする）
  - `rz_geometric_cfl_enabled: bool`（既定 `False`；2D_RZ 専用の predictive geometric hydro CFL。False では kernel を呼ばず既存 hydro dt と bitwise 同一。True では候補 hydro dt の \(\mathbf{x}(\tau)=\mathbf{x}^n+\tau\mathbf{u}^{1/2}\) 上で RZ cell volume と radial floor を評価し、全 cell が \(V(\tau)\ge\eta_V V^n\) を満たす最大 \(\tau\) へ dt を縮小する。NUMERICS §3.2.13 参照）
  - `rz_geometric_cfl_etaV: float`（既定 `0.5` [無次元]；有効範囲：`(0, 1]`；`rz_geometric_cfl_enabled=True` 時の predictive RZ-volume floor \(V_{floor}=\eta_V V^n\)）
  - `rz_geometric_cfl_r_floor: float`（既定 `1e-10` [cm]；有効範囲：`>= 0`；`rz_geometric_cfl_enabled=True` 時に projected node radius が下回ってはならない radial floor）
  - `rz_geometric_cfl_cumulative_protection_enabled: bool`（既定 `True`；`rz_geometric_cfl_enabled=True` 時に initial cell volume floor \(V_{floor,cum}=\eta_{V0}V_c^{initial}\) を geometric CFL に追加する。False では従来の per-step floor のみ）
  - `rz_geometric_cfl_v_initial_floor: float`（既定 `0.1` [無次元]；有効範囲：`[0, 1]`；cumulative protection の \(\eta_{V0}\)。既定 0.1 は cell volume が IC 時体積の 10% 未満へ進む timestep を縮小する）
  - `rz_geometric_cfl_precise_u_half_enabled: bool`（既定 `False`；True では geometric CFL 前に current pressure+AV force から \(a_n\) を再計算し、\(\mathbf{u}^{1/2}=\mathbf{u}^n+0.5\Delta t\,\mathbf{a}_n\) を geometric CFL へ渡す。False では current `state.v_r/v_z` を displacement velocity とする default path）
  - `trial_volume_cfl_enabled: bool`（既定 `False`；2D_RZ 専用の pre-corrector trial-volume CFL diagnostic を有効化する。False では既存 Hydro2D path と bitwise 同一の default-off path。True では corrector node velocity 更新と axis-motion preflight 後、位置 commit 前に trial cell volume を評価し、`trial_volume_cfl_floor_fraction` 未満の shrink をログする。`driver_full_step_retry_enabled=True` では同じ failure が driver full-step retry を要求する。初回 step または履歴なし sentinel では no-op。NUMERICS §2.2, §3.2.13, §3.2.13a 参照）
  - `trial_volume_cfl_floor_fraction: float`（既定 `0.05` [無次元]；有効範囲：`(0, 1]`；`trial_volume_cfl_enabled=True` 時に要求する \(V^{trial}/V^n\) 下限。既定 0.05 は trial volume が step 開始時 volume の 5% 未満になる cell を reject 診断として報告する）
  - `trial_volume_cfl_shrink_fraction: float`（既定 `0.5` [無次元]；有効範囲：`(0, 1)`；trial-volume diagnostic が failure を検出した場合に報告する suggested retry timestep 係数 \(\Delta t_{suggested}=f_{shrink}\Delta t\)。Phase 2d-extension 時点ではログ専用）
  - `corner_jacobian_ale_trigger_enabled: bool`（既定 `False`；2D_RZ 専用の signed corner-J pre-hydro ALE trigger と pre-commit diagnostic を有効化する。False では既存 hydro dt / Hydro2D path と bitwise 同一の default-off path。True では候補 hydro dt の trial node position \(\mathbf{x}^{trial}=\mathbf{x}+\Delta t\mathbf{v}\) で全 cell corner の signed Jacobian が現在値の `corner_jacobian_floor_eps` 倍未満になる場合の admissible scale を評価し、scale が `corner_jacobian_ale_trigger_scale` 未満なら hydro 前に ALE rezone を強制してから dt を再計算する。Hydro2D corrector の最終位置 commit 前にも同じ predicate を評価し、`driver_full_step_retry_enabled=True` では failure が driver full-step retry を要求する。NUMERICS §2.2, §3.2.13, §3.2.13a 参照）
  - `corner_jacobian_floor_eps: float`（既定 `1e-6` [無次元]；有効範囲：`[0, 1)`；`corner_jacobian_ale_trigger_enabled=True` 時の floor \(J_{floor,k}=corner\_jacobian\_floor\_eps \times J_k^n\)。既定値は signed corner-J の sign-preservation に小さい相対余裕を加える）
  - `corner_jacobian_ale_trigger_scale: float`（既定 `0.5` [無次元]；有効範囲：`(0, 1]`；`corner_jacobian_ale_trigger_enabled=True` 時に pre-hydro ALE を強制する admissible scale 閾値。候補 hydro dt 全体を許容できる場合は scale=1、corner-J floor 到達が候補 dt の 50% 未満で予測される場合に既定で ALE を強制する）
  - `in_hydro_corner_j_guard_enabled: bool`（既定 `False`；2D_RZ 専用の in-hydro candidate-mesh corner-J guard。False では既存 Hydro2D predictor/corrector path と bitwise 同一の default-off path。True では predictor/corrector candidate node positions と hydro-stage-start positions の間の analytic \(J(\sigma)=J_0+\sigma J_1+\sigma^2J_2\) を評価し、`refresh_geometry_and_density` の前に typed `HydroStepResult` と suggested dt scale を返す。NUMERICS §3.2.13a 参照）
  - `in_hydro_gauss_j_guard_enabled: bool`（既定 `False`；2D_RZ in-hydro candidate-mesh guard で、4 Gauss-Legendre 点の analytic \(J_g(\sigma)\) を candidate trajectory 上で評価する追加 predicate。False では corner-J guard の既存結果を変更しない。NUMERICS §3.2.13a 参照）
  - `in_hydro_rz_volume_guard_enabled: bool`（既定 `False`；2D_RZ in-hydro candidate-mesh guard で、axisymmetric signed RZ volume \(V_{RZ}(\sigma)\) を candidate trajectory 上で評価する追加 predicate。False では corner-J guard の既存結果を変更しない。NUMERICS §3.2.13a 参照）
  - `in_hydro_gauss_j_floor_rel: float`（既定 `1e-8` [無次元]；有効範囲：`> 0`；`in_hydro_gauss_j_guard_enabled=True` 時の floor \(J_{g,floor}=in\_hydro\_gauss\_j\_floor\_rel \times J_g(0)\)）
  - `in_hydro_rz_volume_floor_rel: float`（既定 `1e-8` [無次元]；有効範囲：`> 0`；`in_hydro_rz_volume_guard_enabled=True` 時の floor \(V_{floor}=in\_hydro\_rz\_volume\_floor\_rel \times V_{RZ}(0)\)）
  - `mesh_quality_dt_cfl_enabled: bool`（既定 `False`；2D_RZ Lagrangian predictor/corrector の位置 commit 前に mesh-quality dt CFL を評価する opt-in master switch。False では既存 Hydro2D path と bitwise 同一。True では \(\mathbf{x}(\sigma)=\mathbf{x}^n+\sigma\Delta t_s\mathbf{u}^{1/2}\)（predictor は \(\Delta t_s=0.5\Delta t\)、corrector は \(\Delta t_s=\Delta t\)）上の first admissible \(\sigma\) を corner-J / Gauss-J / RZ-volume / axis-margin predicates の最小で求め、failure 時は `HydroStepResult.suggested_dt` を返して full-step retry に渡す。NUMERICS §3.2.13a 参照）
  - `mesh_quality_dt_safety_alpha: float`（既定 `0.5` [無次元]；有効範囲：`(0, 1]`；failure 時の suggested dt safety factor。`suggested_dt = mesh_quality_dt_safety_alpha * sigma_safe * dt`）
  - `mesh_quality_dt_corner_j_enabled: bool`（既定 `True`；`mesh_quality_dt_cfl_enabled=True` の時、corner-J quadratic trajectory predicate を有効化）
  - `mesh_quality_dt_gauss_j_enabled: bool`（既定 `True`；`mesh_quality_dt_cfl_enabled=True` の時、4 Gauss-point \(J_g(\sigma)\) predicate を有効化）
  - `mesh_quality_dt_rz_volume_enabled: bool`（既定 `True`；`mesh_quality_dt_cfl_enabled=True` の時、signed RZ-volume cubic trajectory predicate を有効化）
  - `mesh_quality_dt_axis_margin_additive: bool`（既定 `True`；`mesh_quality_dt_cfl_enabled=True` の時、axis-face cells に axis-margin predicate を追加で AND 評価する。corner-J / Gauss-J / RZ-volume の replacement にはしない）
  - `mesh_quality_dt_corner_j_floor_rel: float`（既定 `1e-8` [無次元]；有効範囲：`> 0`；mesh-quality dt CFL の corner-J floor \(J_{floor}=mesh\_quality\_dt\_corner\_j\_floor\_rel \times J(0)\)。axis-margin additive predicate の relative floor にも使用）
  - `mesh_quality_dt_gauss_j_floor_rel: float`（既定 `1e-8` [無次元]；有効範囲：`> 0`；mesh-quality dt CFL の Gauss-J floor \(J_{g,floor}=mesh\_quality\_dt\_gauss\_j\_floor\_rel \times J_g(0)\)）
  - `mesh_quality_dt_rz_volume_floor_rel: float`（既定 `1e-8` [無次元]；有効範囲：`> 0`；mesh-quality dt CFL の RZ-volume floor \(V_{floor}=mesh\_quality\_dt\_rz\_volume\_floor\_rel \times V_{RZ}(0)\)）
  - `ring7_quotient_enabled: bool`（既定 `False`；I1-B Ring7OuterSeamQuotientRemap。False では Hydro2D path と bitwise 同一。runtime env `TENRYU_I1B_RING7_QUOTIENT` が SET されている場合は env が優先される。`TENRYU_I1B_RING7_QUOTIENT_DIAG=1` の時は `[ring7_seam]` diagnostic scan を出力し、`TENRYU_I1B_RING7_QUOTIENT_DIAG_EVERY` は診断 cadence（既定 1 step）。True では StepStart seam transaction も有効化されるが、one-shot request がない場合は seam allocation / metric evaluation / dt reduction / state mutation の前に return する。driver full-step retry が production `mesh_quality_rz_volume` または `multiblock_path_admissibility` rejection を Ring7 seam-patch cell で検出した場合は、復元した次 attempt に single-use Ring7 repair/oracle request（failing cell 付き）を渡して同じ `dt` で再試行する。accepted-step path-oracle minimum margin が Ring7 seam-patch cell に属し `TENRYU_I1B_RING7_PATH_MARGIN_TRIGGER=0.05`（既定）未満の場合は、次 step 開始時の同じ seam transaction に request を arm する。`TENRYU_I1B_RING7_REZONE_COOLDOWN_STEPS=3`（既定）は proactive request の再 arm を抑制する。Interior seam failures can use the dedicated `[ring7_seam_packet]` / `[ring7_seam_remap]` packet path. Driven-pole cap failures use `[ring7_pole_cap]` production-limiter option logs and, in Increment 4b, `[ring7_pole_cap_remap]` conservative ALE packet commit/validation logs. env tunables `TENRYU_I1B_RING7_QUOTIENT_Q_TARGET`（既定 `3*mesh_quality_dt_corner_j_floor_rel` 相当）, `TENRYU_I1B_RING7_QUOTIENT_ETA_TARGET=2.0`, `TENRYU_I1B_RING7_QUOTIENT_SWEEP_CAP=0.20`, `TENRYU_I1B_RING7_QUOTIENT_MOVE_SCALE=0.35`, `TENRYU_I1B_RING7_QUOTIENT_MAX_SUBMOVES=4`, `TENRYU_I1B_RING7_QUOTIENT_BACKTRACKS=8`, `TENRYU_I1B_RING7_QUOTIENT_DT_SHRINK=0.5`, `TENRYU_I1B_RING7_PATH_MARGIN_TRIGGER=0.05`, `TENRYU_I1B_RING7_REZONE_COOLDOWN_STEPS=3`, `TENRYU_I1B_RING7_PACKET_SUBCYCLE_KMAX=8`, `TENRYU_I1B_RING7_PACKET_POS_EPS=1e-12`, `TENRYU_I1B_RING7_PACKET_LAMBDA_BACKTRACKS=5`, `TENRYU_I1B_RING7_PACKET_LAMBDA_MIN=0.1`, `TENRYU_I1B_POLE_CAP_SIGMA_TRIGGER_MAX=1e-2`, `TENRYU_I1B_POLE_CAP_LAYERS_MIN=1`, `TENRYU_I1B_POLE_CAP_LAYERS_MAX=2`, `TENRYU_I1B_POLE_CAP_ALPHA_CENTER=0.0`, `TENRYU_I1B_POLE_CAP_ALPHA_SECOND=0.25`, `TENRYU_I1B_POLE_CAP_ETA_ACCEPT=1.0`, `TENRYU_I1B_POLE_CAP_PACKET_SUBCYCLE_KMAX=8`, `TENRYU_I1B_POLE_CAP_PACKET_POS_EPS=1e-12` を読む。追加 namelist field はない）
  - `regime_aware_corner_j_guard_enabled: bool` (default `False`; 2D_RZ Wave 3 per-cell mesh regime metadata for the pre-hydro corner-J trigger. False allocates no regime buffer and preserves the legacy global threshold path. True classifies active cells after StepStart geometry refresh and lets the pre-hydro trigger read per-cell `CellRegime.trigger_scale_threshold`. NUMERICS §3.2.13a)
  - `axis_margin_guard_enabled: bool` (default `False`; 2D_RZ Wave 3 axis-face replacement predicate. False uses the generic corner-J check for all cells. True makes `i=0` cells use the five-condition axis-margin predicate while non-axis cells keep the generic corner-J predicate. NUMERICS §3.2.13a)
  - `axis_margin_additive_in_action8_enabled: bool` (default `False`; PR4 completion gate for the in-hydro candidate guard. False preserves the legacy `axis_margin_guard_enabled=True` replacement behavior. True records axis-margin as an additional axis-face failure slot while still evaluating corner-J / Gauss-J / RZ-volume, with composite sigma/location reduction selecting the first predicate to fail. NUMERICS §3.2.13a)
  - `axis_guard_band_cells: int` (default `2`; valid range `>=0`; cells with \(i \in [1, axis_guard_band_cells]\) are tagged `AxisBand`, while \(i=0\) is `AxisFace`. NUMERICS §3.2.13a)
  - `driver_full_step_retry_enabled: bool`（既定 `False`；inadmissible Hydro2D corrector trial に対する opt-in driver-level full-step retry。False では既存 Hydro2D path と bitwise 同一の default-off path。True では outer step 開始時の State snapshot へ復元し、dt を厳密に半減して split operators 全体を再試行する。FLD/SN deterministic radiation modes 専用で、`radiation.mode == ImcDdmc` では driver entry で fatal。**1D_SPH（W-B）**: Hydro1D の非正セル体積（圧縮駆動ノード交差、`reason="non_positive_volume_1d"`）と SN material Newton の timestep 拒否（NUMERICS §6.8）も同じ snapshot+dt/2 経路・同じ予算で retry する（2D の repair-plan 機構は経由しない）。retry 無効時の 1D 非正体積は hydro 直後の hard assert）
  - `driver_full_step_retry_max_attempts: int`（既定 `3`；初回 failed attempt 後に許可する retry 回数。各 retry で dt は `0.5 * previous_attempt_dt`。上限到達後も inadmissible の場合は first failing cell/corner と metric を含む fatal）
  - `driver_retry_reference_barrier_enabled: bool`（既定 `False`；B-prime bounded kill-test。2D_RZ ALE + driver full-step retry の mesh-quality AxisBand failure、または `POLAR_SHELL` block の active fine child `multiblock_path_admissibility` / `edge_cross` failure に対し、snapshot 復元後・次 hydro attempt 前に reference-barrier ALE conservative remap を実行する。NUMERICS Phase 9b 参照）
  - `driver_retry_reference_barrier_K_axis: int`（既定 `4`；`first_failing_i <= K_axis` を AxisBand と分類）
  - `driver_retry_reference_barrier_eta_axis: float`（既定 `0.05`；cell centroid の \(r/\sqrt{r^2+z^2}\) がこの値未満なら AxisBand と分類）
  - `driver_retry_reference_barrier_max_attempts: int`（既定 `6`；1 driver step 内の B-prime reference-barrier retry attempt 上限）
  - `driver_retry_reference_barrier_same_sig_max: int`（既定 `3`；同一 `(i,j,stage,reason)` signature の連続上限）
  - `driver_retry_reference_barrier_cell_window: int`（既定 `2`；same-signature 判定で許容する cell-index window）
  - `driver_retry_reference_barrier_dt_collapse_rel: float`（既定 `1.0e-3`；retry dt が failed dt のこの割合未満なら dt-collapse abort）
  - `driver_retry_reference_barrier_lambda_collapse_threshold: float`（既定 `1.0e-3`；accepted reference-barrier \(\lambda\) collapse 閾値）
  - `driver_retry_reference_barrier_lambda_collapse_count: int`（既定 `2`；lambda-collapse abort の連続イベント数）
  - `driver_retry_reference_barrier_quality_progress_factor: float`（既定 `1.25`；reference-barrier 後 quality が前回からこの倍率以上改善しない場合を stagnation と数える）
  - `driver_retry_reference_barrier_quality_progress_count: int`（既定 `2`；quality-stagnation abort の連続イベント数）
  - `driver_retry_reference_barrier_rezone_freq_warn_fraction: float`（既定 `0.20`；recent window 内の B-prime rezone 頻度 warning 閾値）
  - `driver_retry_reference_barrier_rezone_freq_window: int`（既定 `200`；B-prime rezone frequency tracking window）
  - `driver_retry_reference_barrier_chi: float`（既定 `0.8`；`sigma_safe` に掛ける retry dt safety factor）
  - `driver_retry_reference_barrier_q_retry: float`（既定 `0.5`；`sigma_safe` fallback/上限の retry dt ratio）
  - `driver_retry_active_mesh_repair_enabled: bool`（既定 `False`；2D_RZ ALE + driver-level retry 専用の retry-only active mesh repair を有効化する。False では新しい corner-balance kernel と active repair ALE invocation は実行されない。True かつ `driver_full_step_retry_enabled=True` では各 attempt の current corner-J balance を診断し、`retry_attempts > 0` かつ predicate failure の場合だけ `apply_ale(..., force_rezone=true)` を hydro 前に強制する。NUMERICS §3.2.13b 参照）
  - `driver_retry_corner_balance_threshold: float`（既定 `0.01` [無次元]；有効範囲：`(0, 1)`；active mesh repair の \(q_{bal}=\min_k J_{c,k}/\max_k J_{c,k}\) 下限。既定は same-cell corner-J imbalance 100:1 を failure とする。非正または非有限 corner-J は閾値に関係なく predicate failure）
  - `cascade_on_hydro_retry_enabled: bool`（既定 `False`；診断用。True の場合のみ、`driver_retry_active_mesh_repair_enabled` による retry-only forced ALE 呼び出しへ Hydro2D `corner_j` soft-failure の cell/corner と corner-balance failure context を渡し、ALE backtrack が accept しても Phase 9c--9e cascade gate を `gate_reason=hydro_retry_bad_corner_balance` として開く。False では non-retry path と既存 backtrack-exhausted cascade gate は不変。NUMERICS §3.2.13b, Phase 9c 参照）
  - `driver_retry_use_suggested_dt_enabled: bool` (default `False`; for non-geometric typed hydro failures that supply positive `suggested_dt` and `trial_scale`, driver retry uses `min(0.5*failed_dt, suggested_dt)` instead of unconditional half-step retry. Geometric failures (`mesh_quality_*`, `in_hydro_*`) honor finite positive `suggested_dt < failed_dt` strictly regardless of this flag. False otherwise preserves the legacy halving path. NUMERICS §3.2.13b)
  - `geometric_retry_stagnation: dict`（既定 `enabled=False`；driver full-step retry の default-off fail-fast diagnostic。False では retry policy・物理状態・bitwise path は既存通り。True では同一 `(reason, cell, corner/slot)` tuple 内で stage 別の consecutive count と sigma band を保持し、tuple が変わった場合だけ全 state を reset する。`dt_drop_factor` 以下まで dt が低下した上で、任意 stage が `same_cell_count_threshold` 回以上かつ `sigma_rel_tol` 内の sigma band を保つか、stage 合計 count が `2*same_cell_count_threshold` 以上になった場合、`terminal_geometric_stagnation` として `Numerics.dt.min_s` 到達待ちを打ち切る。`force_diagnostic_dump=True` では検出時に `mesh_degeneracy_forensics` dump を `max_dumps_per_run` cap 外で1回試行する）
    - `enabled: bool`（既定 `False`）
    - `same_cell_count_threshold: int`（既定 `3`; validation `>=1`）
    - `sigma_rel_tol: float`（既定 `0.25`; validation `(0,1]`）
    - `dt_drop_factor: float`（既定 `1.0e-4`; validation `(0,1)`）
    - `force_diagnostic_dump: bool`（既定 `True`）
  - `dispatcher_state_sensitive_bypass_enabled: bool` (default `False`; enables the Stage 24 retry dispatcher classifier for typed axis failures. State-sensitive axis failures request same-dt ALE repair, dt-sensitive axis failures use the dt-star/halving path, and False preserves the existing retry selector. NUMERICS §3.2.13b)
  - `dispatcher_state_sensitive_repair_cap_per_step: int` (default `3`; valid range `>=1`; maximum number of state-sensitive same-dt repair generations per outer step before falling through to the legacy retry selector. NUMERICS §3.2.13b)
  - `strategy_first_retry_enabled: bool` (default `False`; enables driver retry selection to try state-sensitive repair hints at the failed timestep before dt halving. State-sensitive hints are `ForceAxisSpinePlusLocalAle`, `ForceBoundaryPatchRepair`, `ForceCdLocalRezone`, `ForceInteriorMultiNodeRepair`, first `ForceFullWinslow`, and `RepairOnly`; `ReduceDtOnly` and `None` remain hydro-inversion/fallback hints and always halve. False preserves legacy retry dt selection.)
  - `strategy_first_max_same_dt_attempts: int` (default `2`; valid range `>=0`; maximum consecutive same-dt strategy-first retry plans before falling back to legacy `compute_retry_dt`; `0` effectively disables same-dt strategy-first retry even when `strategy_first_retry_enabled=True`.)
  - `av_qcap_over_p: float` (default `0.0`; valid range `>=0`; units: dimensionless; VNR q-cap factor \(k\). When `>0`, \(q_{capped}=\min(q,k\max(P_e+P_i,0))\) at every AV producer site; force and work consume the same `Qvisc` storage, so compatible-work conservation is by construction. `0` disables the cap and is byte-identical to the pre-T1 baseline. NUMERICS §3.2.9)
  - `av_qcap_center_band_only: bool` (default `False`; valid values `{True, False}`; units: none; legacy compatibility key. When `av_qcap_scope` is omitted, `True` maps to `av_qcap_scope="tri_fan_radial_index"`; explicit conflicting legacy/scope settings are rejected. Runtime q-cap dispatch consumes `av_qcap_scope`.)
  - `tri_fan_center_cfl_enabled: bool` (default `False`; valid values `{True, False}`; units: none; when true, replaces the tri_fan early-return in `compute_axis_margin_cfl_dt` with a gated center-band stiffness CFL \(\Delta t=\mathrm{safety}\,s_{mid}\Delta\theta/\sqrt{(P+q)/\rho}\) over \(i\in[0,\mathrm{band}], j\in[0,n_z-1]\), including pole-adjacent cells. False preserves tri_fan early-return behavior byte-identical.)
  - `tri_fan_center_cfl_safety: float` (default `0.5`; valid range `>0`; units: dimensionless; safety factor for the center-band stiffness CFL. Smaller values are more conservative and reduce throughput.)
  - `tri_fan_center_cfl_band_radial_index: int` (default `3`; valid range `>=0`; units: radial-cell-index; radial extent of the tri_fan center band, \(i\in[0,\text{this value}]\). `3` covers the center triangles plus first three quad rings. Also used by `av_qcap_scope="tri_fan_radial_index"`.)
  - `corner_j_predict_cfl_enabled: bool` (default `False`; valid values `{True, False}`; units: none; enables the S-D predictive corner-Jacobian timestep limiter on the multiblock Cartesian-core/bridge/button band and the configured number of shell rings. Other topologies are inert with a warning.)
  - `corner_j_predict_cfl_safety: float` (default `0.5`; valid range `(0,1]`; units: dimensionless; multiplies the minimum exact corner-J quadratic root over the scoped cells.)
  - `corner_j_predict_floor_frac: float` (default `0.05`; valid range `(0,1]`; units: dimensionless; clamps the predictive limiter from below at this fraction of the same-step acoustic CFL timestep to prevent Zeno freeze.)
  - `corner_j_predict_max_shrink: float` (default `0.25`; valid range `(0,1)` exclusive; units: dimensionless; maximum permitted one-step relative shrink of each initially positive corner Jacobian under the current node velocities.)
  - `corner_j_predict_shell_rings: int` (default `4`; valid range `>=0`; units: shell-ring-count; extends the prediction scope from all core and bridge cells through this many shell rings past the seam.)
  - `tri_fan_center_perturbation_diag_enabled: bool` (default `False`; valid values `{True, False}`; units: none; when true, emits the additive HDF5 history group `/diagnostics/tri_fan_center_perturbation/v1` per output step with center-band perturbation-energy scalars (`Edot_prime_P`, `Edot_prime_Q`, `Eprime_k`, `max_q_over_p`, `min_corner_J`, `min_cell_volume`). Diagnostic-only; no production logic change.)
  - `av_qcap_scope: str` (default `"global"`; valid values `{ "global", "tri_fan_radial_index", "centroid_r_le_r_match" }`; units: none; topology-aware q-cap scope. Legacy `av_qcap_center_band_only=True` maps to `"tri_fan_radial_index"` when this key is omitted; explicit conflicting legacy/scope settings are rejected. NUMERICS §3.2.9)
  - `center_cfl_scope: str` (default `"disabled"`; valid values `{ "disabled", "tri_fan_radial_index", "centroid_r_le_r_match" }`; units: none; topology-aware center-CFL scope. Legacy `tri_fan_center_cfl_enabled=True` maps to `"tri_fan_radial_index"` when this key is omitted. NUMERICS §3.2.13)
  - `center_perturbation_diag_scope: str` (default `"disabled"`; valid values `{ "disabled", "tri_fan_first_ring", "centroid_r_innermost_bins" }`; units: none; topology-aware perturbation diagnostic scope. Legacy `tri_fan_center_perturbation_diag_enabled=True` maps to `"tri_fan_first_ring"` when this key is omitted. NUMERICS §3.2.13d.1)
  - `center_perturbation_diag_radial_bins: int` (default `2`; valid range `>=1` for `topology_scheme="multiblock_cart_core_polar_shell"`; units: radial-bin-count; active only when `center_perturbation_diag_scope!="disabled"`. NUMERICS §3.2.13d.1)
  - `mesh_geometry_soft_fail_enabled: bool`（既定 `False`；2D_RZ Hydro geometry refresh で非正または非有限の mesh geometry を typed soft failure として driver retry 経路へ返す。False では既存の hard-assert geometry path と bitwise 同一。control flow only であり、物理式・離散化・RNG・単位系は変更しない。NUMERICS §3.2.13a 参照）
  - `qei_evaluate_at_t_n: bool`（既定 `True`；2D_RZ non-per-material 2T Hydro corrector の \(Q_{ei}\) 評価温度を Predictor entry の \(T_e^n,T_i^n\) snapshot に固定する。True は NUMERICS §3.2.12 準拠。False は legacy compatibility mode として Predictor 後の EOS 再クロージャ済み `Te/Ti` を使う）
  - `qei_multiplier: float`（既定 `1.0`；有効範囲 `>0`；2T electron-ion coupling の有限 \(\Delta t\) relaxation exponent を \(m_{ei}\Delta t/\tau_{eff}\) にする無次元倍率。`1.0` は従来の物理 coupling と同一。I5 scaled A-prime deck は選択 table JSON の `parameters.qei_multiplier` を読み、`1.0e-6` を設定する。NUMERICS §3.1.5, §4.1.2 参照）
  - `total_energy_remap_2d_rz: bool`（既定 `False`；2D_RZ conservative ALE remap の total-material-energy recovery path。True では electron/ion internal energy を別々に remap せず、corner-mass nodal kinetic energy と material internal energy の和を conservative scalar として remap し、post-remap node projection 後に同じ corner-mass nodal kinetic energy を差し引いて `e_e/e_i` を bounded `Y_e^int` tracer から復元する。CSR multiblock `topology_scheme="multiblock_half_butterfly_trifan_cap_5block"` では RZ corner-mass-consistent velocity projection、CSR hydro mass-positivity limiter、KE-realizability nodal velocity limiter、report-first `E_floor_injected` accounting を使う。False では legacy `e_e/e_i` separate remap path を選ぶ。structured `polar_center_treatment="tri_fan"` および未検証 CSR topology では True は ConfigError）
	  - `work_split_audit_2d_rz: bool`（既定 `False`；2D_RZ Hydro 2T energy update の diagnostic-only \(P\,dV/Q\,dV\) split CSV `${Output.directory}/diag/work_split_audit_2d_rz_rank%04d.csv` を出力する。True でも hydro state は変更しない。NUMERICS §3.3.5）
	  - `work_split_audit_cell_every_n_steps: int`（既定 `0`；有効範囲 `>=0`；`0` では per-cell row は plot/final cadence のみ、`>0` では指定 step 間隔でも出力）
	  - `work_split_audit_all_rows: bool`（既定 `False`；False では per-cell row は mid-radius row のみ、True では全 radial row を出力。per-window summary は常に mid-radius と radial-summed を出力）
	  - `hllc_z_flux_2d_rz: bool`（既定 `False`；I1-A `I1A_2D_RZ_FLD_CED_PLANAR_Z_SHOCK_HLLC` 用の 2D_RZ quasi-1D z-normal shock experimental Eulerian z-face HLLC conservative flux。True では legacy staggered z Lagrange + conservative reference-remap z transport を bypass し、cell-centered authoritative `hllc_mom_z_cell` と total-material-energy update を使う。`total_energy_remap_2d_rz=True` が必須。A2 z-HLLC submode の code-verification であり、default VNR/ALE radiative-shock validation ではない）
	  - `hllc_z_flux_audit_2d_rz: bool`（既定 `False`；`${Output.directory}/diag/hllc_z_flux_2d_rz_rank%04d.csv` に Te shock width, rho width, HLLC/HLLE fallback counts, flux and projection diagnostics を出力する）
	  - `hllc_z_flux_hlle_fallback: bool`（既定 `True`；HLLC star state が非有限または非正の場合に HLLE flux へ fallback する）
	  - `hllc_z_flux_strict_quasi_1d: bool`（既定 `False`；experimental flag。現実装では quasi-1D 制約は audit/warning 用であり、production 2D shock 捕獲を主張しない）
	  - `bbs_axis_policy_enabled: bool`（既定 `False`；2D_RZ 専用。True では five-block half-butterfly の \(R=0\) compatible-acceleration mass-floor fallback に Barlow-Burton-Shashkov RZ subzonal volume mass \(m_{c,k}=\Delta M_c V_{RZ}(Q_{c,k})/\sum_jV_{RZ}(Q_{c,j})\) を使い、planar \(1/4\) control-volume fallback を使わない。False では既存 path と bitwise 同一。HDF5 schema 変更なし。NUMERICS §3.2.4）
	  - `subzonal_mass_enabled: bool`（既定 `False`；2D_RZ 専用。True では Caramana-Shashkov 型 corner/subzone mass を runtime state として確保し、Lagrangian step 中は frozen、ALE remap acceptance 後だけ remapped cell mass と current mesh から再初期化する。`tri_fan` では ConfigError。HDF5 schema 変更なし。NUMERICS §3.2.9b）
	  - `subzonal_mass_lagrangian_invariant_enabled: bool`（既定 `False`；2D_RZ 専用。True では `state.corner_mass[c*4+k]` を IC-time initialization 後の pure-Lagrange invariant として扱い、Hydro2D step 内の geometry-driven recompute を禁止する。`subzonal_mass_enabled=True` または `hourglass.enabled=True` では runtime effective flag が自動的に True になる。compatible multiblock ALE remap 後は Option-2-a subzonal-aware remap が corner-mass fractions を passive mass-weighted scalars として transport し、\(\sum_k m_{c,k}=M_c\) に closure する。compatible-off path は従来通り warning 付き再初期化に戻る。HDF5 schema 変更なし。NUMERICS §3.2.9b）
	  - `anti_hourglass_kappa: float`（既定 `0.05` [無次元]；有効範囲 `>0`；`subzonal_mass_enabled=True` path の anti-hourglass force 係数。legacy `hourglass.enabled=True` path では `hourglass.scale` を保持する）
	  - `subzonal_pressure_mode: Literal["uniform_cell"]`（既定 `"uniform_cell"`；Phase 3 MVP は cell pressure を subzone pressure coefficient として使う。`"caramana_shashkov"` は将来の EOS-based subzonal pressure 用予約値で現実装では ConfigError）
	  - `subzonal_band_mode: Literal["off","bridge_feather"]` (default `"off"`; `"bridge_feather"` scales compatible subzonal-pressure corner forces by a topology-derived per-cell weight that is 1 on multiblock bridge cells, has a quintic face-adjacency feather into the core and shell, and is 0 beyond the feather. Requires the multiblock button topology.)
	  - `subzonal_band_feather_layers: int` (default `2`; valid range `>=1`; number of face-adjacency layers on each side of the bridge band that receive the quintic feather.)
	  - `av_model: Literal["scalar_vnr_legacy","csw_edge","csw_edge_csw98","csw_edge_plus_tensor_limited"]`（既定 `"scalar_vnr_legacy"` for all topologies; no topology-default magic. `"csw_edge"` selects the Phase 4 DRACO/HYDRA-class edge AV surface and must be paired with `subzonal_pressure_enabled=True`. `"csw_edge_csw98"` selects the CSW-1998-faithful median-mesh edge AV (I1-B Stage-G W1; NUMERICS §3.2.9c): median-mesh S per Eq. 16, strict compression switch, logical-line continuation-edge limiter (Eq. 18), force-coupled AV CFL. It runs the same compatible force/work path and is valid with OR without `subzonal_pressure_enabled` (column A/B isolation). `"csw_edge_plus_tensor_limited"` is a Stage G placeholder and is rejected at validation with `ConfigError("Stage G tensor AV is not yet implemented; use csw_edge")`）
	  - `subzonal_dt_limiter_enabled: bool` (default `True`; gates the subzonal-pressure frequency dt bound (an upper-bound stiffness estimate from corner masses/volumes). `False` removes only the dt limiter — the subzonal force/work physics is untouched. Escape hatch for venues where the bound is spuriously tight; champion-era reference binaries predate this limiter entirely (2026-07-29).)
	  - `corner_mass_convention: string` (default `"kinematic_basis_rz_v1"`; Selects the corner-mass partition of cell mass onto its four corners for the compatible discretization (node mass/KE/projection/energy-audit share it). bbsw_radial_v0: the logical-index radial-linear BBSW lump (exact R-weighted Q1 lump on i-aligned radial rectangles; degenerates to an equal split on axis-column cells). kinematic_basis_rz_v1: m_k = M·int(N_k R J)/int(R J) by 2x2 Gauss — the exact R-weighted lump of the Q1 kinematic basis for ANY bilinear quad. P1 triangle-degenerate cells use the exact P1 closed form w_k=(r_k+Σr)/(4Σr) under kinematic_basis_rz_v1. Default kinematic_basis_rz_v1 as of the Wave-2D G4 epoch (2026-07-27); bbsw_radial_v0 remains selectable and is the permanent resolution for legacy frozen configs that predate the key. The multiblock exact-subpolygon path is not governed by this knob.)
	  - `time_integration: Literal["pc_v0","midpoint_v1"]` (default `"midpoint_v1"`; 2D_RZ Hydro time integration. `"pc_v0"` is the retained first-order predictor-corrector whose V3 smooth-temporal measured orders are \(p_\rho=1.00072\) and \(p_e=1.00071\). `"midpoint_v1"` selects the F-08 fixed-one-corrector midpoint scheme: half-step compatible work, exact half-geometry density \(M/V(x^h)\), one shared midpoint corner-force family for momentum/work/audit, one provisional full update, and exactly one deterministic force re-evaluation. Default midpoint_v1 as of the Wave-2E G5 epoch (2026-07-27; measured V3 orders p_rho=2.0000, p_e=1.9999). pc_v0 remains selectable and is the permanent resolution for legacy frozen configs predating the key. Multiblock/CSR, HLLC, and button-center integrations remain pc_v0 regardless of the knob (step-0 note).)
	  - `total_energy_identity_check: bool` (default `False`; opt-in Hydro debug assertion in both time-integration modes. After a step it checks \(R=\Delta K+\Delta U-W_{\rm ext}-E_{\rm floor}\) with \(|R|\le10^{-11}\max(E_{\rm kin}^n+E_{\rm int}^n,|W_{\rm ext}|,10^{-300})\), and reports all terms on failure. False disables the assertion; `"pc_v0"` then performs no associated energy-ledger reduction, while `"midpoint_v1"` retains its required staged \(W_{\rm ext}\) capture.)
	  - `rz_momentum_scheme: Literal["volume_weighted","area_weighted_symmetric"]`（既定 `"volume_weighted"`；`"area_weighted_symmetric"` は FIX2-W1/W2 の structured single-block および all-quad multiblock CSR 2D_RZ scalar-VNR momentum operator。`Main.dimension="2D_RZ"`, `av_model="scalar_vnr_legacy"`, `subzonal_pressure_enabled=False` が必須で、違反は `ConfigError`。multiblock tri-fan-cap topology は v1 では `ConfigError`、CSR launch でも全 cell が quad であることを assert する。内部エネルギーは既存の geometric `P dV` 非compatible 更新を保持する。NUMERICS §3.2.5）
	  - `axis_node_mass_convention: Literal["corner_subzonal","equal_split","equal_split_all"]`（既定 `"corner_subzonal"`；multiblock CSR の軸ノード質量規約。`"corner_subzonal"` は exact R-weighted subzonal corner mass を合算し variational endpoint drive split 下で 4/3 pole impedance seed を生じる。`"equal_split"` は軸ノードで incident-cell の m/4（quad）または m/3（triangle）を合算して structured polar convention に一致させる。`"equal_split_all"` は全ノードで equal-split share を使用する full structured-convention mass distribution であり、`"equal_split"` の diagnostic superset である。corner-mass cache は不変）
	  - `av_cfl_coefficient: float`（既定 `0.25` [無次元]；有効範囲 `>0`；`av_model="csw_edge"` の edge-relative velocity CFL safety coefficient \(C_{avs}\), \(\Delta t\le C_{avs}|\Delta x_e|/|\Delta u_e|\)。legacy `scalar_vnr_legacy` では未使用。`csw_edge_csw98` では実力結合形 \(\Delta t\le C_{avs}|\Delta x_e|/(|\Delta u_e|(1-\psi_e)|\hat{u}_e\cdot\hat{S}_e|)\) を用い、力がゼロの辺は dt を拘束しない。NUMERICS §3.2.9b/§3.2.9c）
	  - `subzonal_pressure_enabled: bool`（既定 `False`；Phase 4 subzonal pressure surface. Validation: `av_model="csw_edge"` requires `subzonal_pressure_enabled=True`; `subzonal_pressure_enabled=True` rejects `av_model="scalar_vnr_legacy"`; `"csw_edge_csw98"` itself may run with subzonal on or off）
	  - `subzonal_merit_mode: Literal["caramana_auto","constant","off"]`（既定 `"caramana_auto"`；Caramana-Shashkov subzonal merit coefficient mode for the Phase 4 pressure surface）
	  - `subzonal_alpha1: float`（既定 `1.4142135623730951` [無次元]；Caramana-Shashkov merit coefficient \(\alpha_1\)）
	  - `subzonal_alpha2: float`（既定 `0.1` [無次元]；Caramana-Shashkov merit coefficient \(\alpha_2\)）
	  - `subzonal_merit_power: int`（既定 `2` [無次元]；Caramana-Shashkov merit exponent）
	  - `subzonal_merit_constant: float`（既定 `1.0` [無次元]；constant-mode merit coefficient）
	  - I1-A strict verification harness metric（namelist ではない）: `tools/validation/run_i1_2d_rz_fld_ced.py` は finest grid \(64\times1024\) の `D=T_r/T_e-1` を固定 upstream window \(W_u=[z_s-8\ell,z_s-\delta]\), \(\delta=\max(2\Delta z,0.25\ell)\), \(\ell=1/(\sqrt{3}\kappa_R\rho_{up})\) で評価し、reference は candidate kernel ではなく exact cell-average projection で比較する。pass/fail は \(D_{\max}\) 10%, integrated positive precursor 10%, shape \(L_2\le0.08\), and independent 10--90% matter-shock thickness \(W_{shock}/\ell\le1.0\) or \(N_{shock}\le3\)。by-grid diagnostics も summary JSON に出力する。legacy `shock_windowed_l2_abs` and `convolved_peak_gap` は diagnostic-only。`reference_table` IC の t0-admissibility keys は `mach_match`, `pressure_ratio_match`, `profile_match`, and `reference_table_radiation_equilibrium_admissibility`。`pressure_ratio_match` は code-vs-reference \(P_{rad}/P_{mat}\) edge+shock-ratio relative error \(\le\) `T0_PRAD_RATIO_REL_GATE`; `reference_table_radiation_equilibrium_admissibility` は t0 \(T_{rad}=T_e\) の relative \(L_\infty\le\) `T0_TRAD_TE_REL_GATE` を要求する。deck は `radiation_field=equilibrium` IC を使うため FLD precursor は pre-loaded ではなく時間発展で形成される。これは smooth reference-table IC に不適用な two-state nED check を置換する。`TENRYU_I1_2D_RZ_VERBOSE_DIAG` は gate-default lean diagnostics を verbose smoke/debug diagnostics に戻す environment gate であり、physics state には影響しない。Newton H3 line は lean mode でも `production_audit` 経由で出力される。
	  - I1-B polar pole diagnostic（namelist ではない）: `TENRYU_I1B_POLAR_POLE_DIAG=1` enables default-off JSONL diagnostics under `tmp/diagnostics/`. Each record covers the outer theta-pole cell windows and includes step/time, cell indices, exact RZ volume ratio, Gauss-J ratio, corner-J minimum/ratio, pole-node `u_r/u_z`, and ALE trigger/forced-rezone counters. Unset means no output, no HDF5/restart schema change, and no physics-path change.
	  - I1-B PAB polar deck button knob（namelist ではない）: `TENRYU_I1B_PAB_POLAR_CENTER_BUTTON_OUTER_NODE_RING` (default `2`) is read by `examples/verification/2d_rz_i1b_pab_polar.py` only when `TENRYU_I1B_PAB_POLAR_POLAR_CENTER_TREATMENT=button`; it forwards to `Mesh.center_button_outer_node_ring` and is printed in the deck header only in button mode. `annular` and `tri_fan` deck namelists remain unchanged when button is off.
	  - I1-B pole angular coarsen/motion pilots（namelist ではない）: `TENRYU_I1B_POLE_COARSEN_PILOT=1` enables the default-off Hydro2D path-admissibility quotient pilot. `TENRYU_I1B_POLE_MOTION_PILOT=1` enables the default-off coherent mesh-position-velocity pilot, reusing the same q-band and dyadic macro controls but leaving fine-cell path checks enabled. `TENRYU_I1B_PATH_PREDICATE_HARDEN=1` separately enables default-off Stage-0 path predicate hardening; unset preserves the legacy live predicate path. Optional env bounds are `TENRYU_I1B_POLE_COARSEN_Q_MIN`/`Q_MAX` (defaults `7`/`7`) and `TENRYU_I1B_POLE_COARSEN_LEVEL_MAX` (default `3`, dyadic spans 2/4/8). The motion pilot's inward taper uses `TENRYU_I1B_POLE_MOTION_TRANSITION_ROWS` (default `4`) and `TENRYU_I1B_POLE_MOTION_PROFILE` (default `smoothstep`). These add no namelist key, frozen-config field, checkpoint field, HDF5 group, or restart migration.
	  - Tier-A butterfly-center authority harness（namelist ではない）: opt-in is the CTest target `test_butterfly_authority_tier_a` (filter with `ctest -R butterfly_authority_tier_a`). It constructs only geometry on `multiblock_half_butterfly_5block`, emits `[tier_a_butterfly_authority]` curves for \(C=\{1,1.25,1.5,2,3,4,6,8\}\), and has no production namelist key, HDF5 schema effect, restart state, hydro/remap/BBS path, or default runtime behavior.
	  - `hourglass: dict`（2D_RZ 専用の default-off Caramana-Shashkov subzonal pressure anti-hourglass force。`enabled=False` では state 配列を確保せず、既存 Hydro2D path と bitwise 同一。NUMERICS §3.2.9b 参照）
    - `enabled: bool`（既定 `False`；2D_RZ 以外で True は `ConfigError`）
    - `scale: float`（既定 `0.05` [無次元]；有効範囲 `> 0`；subzonal pressure force の \(C_{hg}\)）
    - `compatible_work_enabled: bool`（既定 `True`；True では hourglass force work を同じ stage の internal energy increment へ coupled accounting する）
    - `activation_corner_j_ratio_threshold: float`（既定 `0.5` [無次元]；有効範囲 `(0, 1]`；\(\min J/\max J\) がこの値未満の cell だけ active）
    - `activation_hourglass_amplitude_threshold: float`（既定 `0.01` [無次元]；有効範囲 `(0, 1]`；\(\|\mathbf{a}_{\xi\eta}\|/\sqrt A\) がこの値以上の cell だけ active）
    - `subzonal_pressure_model: Literal["linearized"]`（既定 `"linearized"`；`"eos_lookup"` は予約値だが v1 実装では `ConfigError`）
    - `max_force_per_node_fraction: float`（既定 `0.2` [無次元]；有効範囲 `> 0`；\(\|\mathbf{F}^{hg}_n\|/(m_n\|\mathbf{a}^{P}_n\|)\) の cap）
  - `av_type: Literal["vnr","riemann","riemann_compatible","csw"]`（既定: **1D_SPH の raw deck で未指定なら builder が `"csw"` に解決**（2026-08-03、AV 近代化 Stage 1A — post-shock plateau ripple 3.43%→1.97% rms・吸収/バンタイム パリティ実測）。構造体既定は `"vnr"` のまま（`"csw"` は 1D_SPH 専用・2D_RZ は `av_model` を使用）で、明示指定（`"vnr"` 含む）は常に尊重される。frozen config は av_type を常時明示出力するため旧 frozen config の再生は不変。1D_SPH の shock support 形式。`"vnr"` は既存の Christensen-limited von Neumann-Richtmyer cell-centered AV、`"riemann"` は各セル境界で midpoint cell velocity の minmod reconstruction と nonlinear impedance \(Z^{eff}=\rho(c_s+\alpha\Delta u^+)\), \(\alpha=(\Gamma_1+1)/4\) を用いる acoustic Riemann solver から \(Q_{i+1/2}=\max(0,P^\*-\tfrac12(P_L+P_R))\) を求め、それを `Q_i = 0.5(Q_{i-1/2}+Q_{i+1/2})` で cell center に平均して既存の momentum / PdV 更新へ渡す。`"riemann_compatible"`（v1.1 2026-08-03）は Morgan (2014) 型 staggered Godunov-like 縮約: nodal→cell-center 射影（nodal secant 勾配 + Barth–Jespersen \(\beta=0.5\)）で得た制限付き corner 状態の full jump \(\Delta u^c\) と対称一次 impedance \(\mu=\rho(c_s+\tfrac12 b_1\Delta u^c)\)、\(b_1=(\Gamma_1+1)/2\) から \(Q_i=(\mu/2)\Delta u^c\) を作り、体積圧縮ゲート（\(\dot V<0\) のみ活性）付きで既存の momentum / PdV compatible-work 経路へ渡す opt-in 形式（\(c_s=0\) の冷セルでも有効）。`"csw"` は node velocity の monotone reconstruction から \(\chi^{lim}\) を作る Caramana-Shashkov-Whalen 型 AV。`"riemann"`、`"riemann_compatible"` と `"csw"` は VNR shock-support gate と mild-compression branch を使用しない。`"riemann"`、`"riemann_compatible"` と `"csw"` は 1D_SPH 専用で、2D_RZ では `ConfigError`。NUMERICS §3.1.6 参照）
  - `av_C1: float`（既定 0.1 [無次元]；有効範囲：`[0, 10]`；`av_type="vnr"` のときの人工粘性線形係数。2D_RZ `av_model="csw_edge"` / `"csw_edge_csw98"` では CSW edge AV の \(C_1\) としても使い、namelist で `av_C1` / `av_linear` が省略された場合は CSW published default `1.0` に切り替える。0=線形粘性無効。`av_type="riemann"` / `"riemann_compatible"` / 1D `av_type="csw"` では未使用。NUMERICS §3.1.6, §3.2.9b/§3.2.9c参照）
  - `av_C2: float`（既定 1.5 [無次元]；有効範囲：`[0, 10]`；`av_type="vnr"` のときの人工粘性2次係数。2D_RZ `av_model="csw_edge"` / `"csw_edge_csw98"` では CSW edge AV の \(C_2\) としても使い、namelist で `av_C2` / `av_quadratic` が省略された場合は CSW published default `1.0` に切り替える。0=2次粘性無効。典型値 1.0-2.0。`av_type="riemann"` / `"riemann_compatible"` / 1D `av_type="csw"` では未使用。NUMERICS §3.1.6, §3.2.9b/§3.2.9c参照）
  - `csw_C1: float`（既定 0.5 [無次元]；有効範囲：`[0, +∞)`；`av_type="csw"` 専用の線形 AV 係数。`av_C1` / `av_linear` とは分離し、VNR 既定値を変えない。NUMERICS §3.1.6 参照）
  - `csw_C2: float`（既定 2.0 [無次元]；有効範囲：`[0, +∞)`；`av_type="csw"` 専用の2次 AV 係数。NUMERICS §3.1.6 参照）
  - `csw_limiter: Literal["van_leer","bj"]`（既定 `"van_leer"`；`av_type="csw"` の node slope limiter。`"bj"` は Barth-Jespersen limiter。NUMERICS §3.1.6 参照）
  - `csw_limiter_enabled: bool`（既定 `True`；2D_RZ `av_model="csw_edge"` / `"csw_edge_csw98"` の edge limiter debug switch（csw_edge: Christensen-Caramana 形; csw_edge_csw98: CSW98 Eq. 18 continuation-edge 形）。False では \(\psi_e=0\) とし、より強い edge AV dissipation を発生させる。1D `av_type="csw"` では未使用。NUMERICS §3.2.9b/§3.2.9c）
  - `csw_shock_limiter_floor: float`（既定 0.65 [無次元]；有効範囲：`[0, 1]`；CSW reconstruction が圧縮を検出した shock-like セルで \(\chi^{lim}\) の下限を \(\chi^{raw}\) のこの割合に制限する。一様圧縮で \(\chi^{rec}=0\) の場合は適用しない）
  - `csw_zero_uniform_compression: bool`（既定 `True`；True で monotone reconstruction により滑らかな線形/相似圧縮の CSW AV を 0 にする。False では \(\chi^{raw}\) をそのまま用いる診断モード）
  - `csw_diagnostics: bool`（既定 `False`；CSW 用診断予約フラグ。PR03 では config/freeze のみで、追加 diagnostic field は出力しない）
  - `av_limiter_J: float`（既定 1.0 [無次元]；有効範囲：`[0, +∞)`；`av_type="vnr"` のときの 1D球対称 Christensen 速度リミタ係数。`J=1` が標準、`J=0` で制限傾きがゼロとなり、AV/CFL は未制限のセル端速度差を用いる。`av_type="riemann"` / `"riemann_compatible"` / `"csw"` では未使用。2D_RZ では未使用。NUMERICS §3.1.6, §3.1.9 参照）
  - `av_heat_C: float`（既定 0.0 [無次元]；有効範囲：`[0, +∞)`；`av_type="vnr"` / `"csw"` のときの 1D球対称人工熱流束係数 \(C_H\)。各 AV の compression sensor \(\chi\) とイオン（1Tでは全熱）比内部エネルギー勾配から post-shock 温度プロファイルを平滑化する。`0` で無効。`av_type="riemann"` / `"riemann_compatible"` では使用不可で、`>0` は `ConfigError`。2D_RZ では未使用。NUMERICS §3.1.6 参照）
  - `ion_art_heat_C: float`（既定 0.0 [無次元]；有効範囲：`[0, +∞)`；1D_SPH + 2T + `av_type="vnr"` 専用。quadratic AV 成分 \(q^{(2)}\)、compression sensor、contact sensor から作る explicit ion-only artificial heat conduction の強度係数。main/compatible work 更新後、EOS 再閉包前にイオン比内部エネルギー `ei` だけへ保存形で加える。`0` で無効。2D_RZ、1T hydro、`av_type="riemann"` / `"riemann_compatible"` / `"csw"` では `>0` は `ConfigError`。v1 の推奨 sweep は `0.25, 0.5, 1.0`。NUMERICS §3.1.5, §3.1.6 参照）
  - `post_shock_heat: bool`（既定 `False`；1D_SPH 専用。True で recently-shocked セルだけに短メモリの局所人工熱流束を有効化する。shock history は `Qvisc > 0.01 * max(Qvisc)` を満たしたセルで更新し、\(\psi=\exp(-(t-t_{shock})/\tau_{decay})\) で減衰する。2D_RZ では未使用。NUMERICS §3.1.6, §3.1.9 参照）
  - `post_shock_heat_C: float`（既定 `0.1` [無次元]；有効範囲：`[0, +∞)`；localized post-shock artificial heat flux の強度係数。`post_shock_heat=False` のとき無視。2T では total internal energy に適用した後、セルごとに `Pe/(Pe+Pi)` 比で電子・イオンへ分配する。2D_RZ では未使用。NUMERICS §3.1.5, §3.1.6 参照）
  - `post_shock_heat_decay: float`（既定 `3.0` [cell crossing times]；有効範囲：`> 0`；shock history の減衰時間 \(\tau_{decay} = post\_shock\_heat\_decay \times \Delta r / c_s\)。`post_shock_heat=False` のとき無視。2D_RZ では未使用。NUMERICS §3.1.6, §3.1.9 参照）
  - `post_shock_velocity_damping_C: float`（既定 `0.0` [無次元]；有効範囲：`[0, +∞)`；1D_SPH 専用。既存の odd-even nodal damping 係数 \(\mu_i\) に recently-shocked weight \(C_{psv}\psi_i^{eff}\) を追加し、shock 通過直後の velocity ringing を直接減衰する。shock history は `post_shock_heat` と共有し、`Qvisc > 0.2 * max(Qvisc)` の shock core では \(\psi_i^{eff}=0\) として main shock structure を保護する。`0` で無効。2D_RZ では未使用。NUMERICS §3.1.4, §3.1.6 参照）
  - `bulk_viscosity_C: float`（既定 `0.0` [無次元]；有効範囲：`[0, +∞)`；1D_SPH 専用。cell-centered viscous pressure \(Q_{bulk}=C_{bulk}\rho c_s |(u_{j+1}-u_j)/\Delta r_i| \Delta r_i\) を既存 `Qvisc` に加算する。**圧縮セルのみ**（\(du<0\)）で作用し、post-shock 圧縮域の音響振動を直接減衰する（2026-07-26 是正: 旧仕様の膨張側適用は反散逸＝エントロピー減少であり撤廃 — AI review k01 P0-5 / k03 F-03）。`0` で無効。2D_RZ では未使用。NUMERICS §3.1.6 参照）
  - `crossing_dt_safety: float`（既定 `0.5` [無次元]；有効範囲：`[0, +∞)`；1D 節点交差 timestep ガード。1 step で 1 セル面が現セル幅の本割合を超えて閉じないよう \(\Delta t_{cross} = C_{cross}\min_i \Delta r_i/\max(u_i-u_{i+1},0)\) を hydro dt の min に加える（生の幾何制約、`cfl_hydro` 倍なし）。acoustic+AV 側が有効な領域では発火せず、limiter が \(\chi\) を消す冷たい滑らかな収縮のみを保護する。`0` で無効。NUMERICS §3.1.9 参照）
  - `time_integrator: Literal["legacy_pc","midpoint_v2"]`（既定 `"legacy_pc"`；1D 専用（2D_RZ で `"midpoint_v2"` は `ConfigError`）。`"legacy_pc"` は歴史的 predictor–corrector（predictor はエネルギーを進めず、legacy PdV は \((P^n+P^{n+1/2})/2\) 平均 — 時間 1 次のエネルギー更新、bitwise 保存）。`"midpoint_v2"` は predictor 段で \(e_e,e_i\) も half step へ stage 更新し、corrector エネルギーに stage \(P^{n+1/2},Q^{n+1/2}\) を直接使う時間 2 次の中点法。NUMERICS §3.1.10 参照）
  - `adaptive_av: dict`（既定 `{"enabled": False, "base": {"c1": 0.10, "c2": 1.50, "heat_C": 0.50, "Cpsv": 0.00, "cbulk": 0.00}, "primary": {"c1": 0.50, "c2": 1.50, "heat_C": 0.80, "Cpsv": 0.75, "cbulk": 0.25}, "rebound": {"c1": 0.18, "c2": 1.50, "heat_C": 0.30, "Cpsv": 0.00, "cbulk": 0.00}, "taper_r_start": 0.25, "taper_r_end": 0.05, "hysteresis_w": 0.3, "hysteresis_tau": 0.0, "support_ahead": 1, "support_behind": 10}`；1D_SPH + `av_type="vnr"` 専用。base VNR probe から leading shock cluster を検出し、履歴 gate \(g_i=(1-w)g_i^{old}+w g_i^{target}\) でセルごとの \(C_1,C_2,C_{heat},C_{psv},C_{bulk}\) を mode 係数へ補間する。`base/primary/rebound` の各係数は `c1,c2,heat_C,Cpsv,cbulk` を持ち、全て `>=0`。`taper_r_start > taper_r_end >= 0`、`hysteresis_w in [0,1]`、`hysteresis_tau >= 0` [s]（`>0` で gate blend を物理時定数形 \(w(\Delta t)=1-\exp(-\Delta t/\tau_g)\) に切替 — timestep refinement 不変な履歴緩和。`0` は legacy 固定 `hysteresis_w`；2026-07-26 追加、AI review k01 §8.1）、`support_ahead/support_behind >=0`。`av_type!="vnr"` では `ConfigError`。`Cpsv` は Stage 1 では既存 post-shock nodal damping へ per-cell 係数として渡し、`compatible_energy=True` では compatible update を無効化する。NUMERICS §3.1.6 参照）
  - `plasma_viscosity` (dict, default disabled) — Braginskii plasma shear
    viscosity (ion + electron channels) for 1D (all geometries) and 2D RZ
    (NUMERICS §3.1.13):
    - `enabled` (bool, default `False`) — master switch. Disabled runs are
      bit-identical to pre-W-H behavior.
    - `model` (str, `"braginskii"` | `"constant"`, default `"braginskii"`) —
      `"braginskii"`: η₀ = 0.96 n_i kT_i τ_i with NRL τ_i and ion-ion Coulomb
      log (floor 2), per-cell A_eff/Z̄; `"constant"`: uniform `eta_const`
      (verification gates).
    - `species` (str, `"ion"` | `"electron"` | `"both"`, default `"ion"`) —
      which Braginskii channel(s) feed π = −η_eff W. `"ion"`: W-H legacy
      (bit-preserving default, 1D and 2D). `"electron"`: η_e = η₀₀^e(Z) n_e
      kT_e τ_e with the Whitney Z-dependent coefficient (0.733 at Z=1, →
      1.81 as Z→∞). `"both"`: additive η_eff = η_i + η_e — the production
      recommendation; the dominance regimes (electron-dominant / ion-
      dominant / mixed) emerge automatically from R = η_e/η_i ∝
      (T_e/T_i)^{5/2} Z³/√A (NUMERICS §3.1.13). Under `model="constant"`
      η_eff = `eta_const` for every species value (ion: all-ion, electron:
      all-electron, both: half/half) — only the heat routing changes.
    - `eta_const` (float ≥ 0, poise, default 0) — constant-model viscosity.
    - `eta0_scale` (float > 0, default 1) — multiplier on η₀ (transport-model
      uncertainty knob; first-principles models spread ~5×, Haines 2024).
    - `mfp_cap_cells` (float ≥ 0, default 20) — cap λ_ii ≤ C·Δr and λ_e ≤ C·Δr (Mason 2014; shared C for both channels);
      0 disables (NOT recommended: uncapped η ∝ T_i^{5/2} collapses dt in hot
      dilute cells).
    - `lnlambda_fixed` (float ≥ 0, default 0) — 0: NRL lnΛ_ii / lnΛ_ei; >0: fixed value applied to both.
    - `dt_safety` (float > 0, default 0.3) — viscous CFL fraction in
      dt ≤ dt_safety·ρΔr²/((8/3)η).
    Viscous heating deposits per channel: Q_i → ion energy, Q_e → electron
    energy (both → the single material energy in 1T mode). When enabled, a
    /diagnostics/plasma_viscosity_history/ group (η_i/η_e/η_eff maxima,
    R = η_e/η_i statistics, regime cell counts, per-channel heat rates) is
    appended at history cadence.
    Artificial viscosity is unaffected and remains on (physical viscosity is
    not a shock-capturing replacement, Manheimer & Colombant 2007).
    - `eta0_scale` (float > 0, default 1) — multiplier on both channels
      (transport-model uncertainty knob; first-principles models spread ~5×,
      Haines 2024).
    - `mfp_cap_cells` (float ≥ 0, default 20) — cap λ_ii ≤ C·Δr and
      λ_e ≤ C·Δr (Mason 2014; shared C for both channels); 0 disables (NOT
      recommended: uncapped η ∝ T^{5/2} collapses dt in hot dilute cells).
    - `lnlambda_fixed` (float ≥ 0, default 0) — 0: NRL lnΛ_ii / lnΛ_ei;
      >0: fixed value applied to both.
    - `dt_safety` (float > 0, default 0.3) — viscous CFL fraction in
      dt ≤ dt_safety·ρΔr²/((8/3)η_eff), shared by both channels.
    Viscous heating deposits per channel: Q_i → ion energy, Q_e → electron
    energy (both → the single material energy in 1T mode; in 2D this
    per-channel routing is hard-wired on both the legacy and compatible
    energy paths). When enabled, a /diagnostics/plasma_viscosity_history/
    group (η_i/η_e/η_eff maxima, R = η_e/η_i statistics, regime cell
    counts, per-channel heat rates) is appended at history cadence (1D and
    2D). Artificial viscosity is unaffected and remains on (physical
    viscosity is not a shock-capturing replacement, Manheimer & Colombant
    2007). In 2D RZ, enabling plasma viscosity with
    `Numerics.materials.per_material_conservation_enabled=True` raises `ConfigError`.
  - `av_eos_aware: bool`（既定 `False`；`av_type="vnr"` かつ table EOS が active なセルで局所 \(\Gamma_1 = \rho c_s^2 / (P_e+P_i)\) に応じて人工粘性係数を増幅する。True のとき \(B_i = \min(B_{max}, \max(1, \Gamma_{1,ref}/\Gamma_{1,i}))\) を AV 内部でのみ適用し、`C1_eff=C1 B_i`, `C2_eff=C2 B_i` とする。ideal-gas path では `B_i=1`。`av_type="riemann"` / `"riemann_compatible"` / `"csw"` では未使用。NUMERICS §3.1.6, §3.2.9参照）
  - `av_eos_gamma1_ref: float`（既定 `5/3` [無次元]；有効範囲：`> 0`；`av_type="vnr"` かつ `av_eos_aware=True` のときの EOS-aware AV 基準 \(\Gamma_1\)。それ以外では無視）
  - `av_eos_boost_max: float`（既定 `3.0` [無次元]；有効範囲：`[1, +∞)`；`av_type="vnr"` かつ `av_eos_aware=True` のときの EOS-aware AV 最大増幅倍率 \(B_{max}\)。それ以外では無視）
  - `odd_even_damping_C: float`（既定 0.0 [無次元]；有効範囲：`[0, +∞)`；1D_SPH 専用の odd-even 抑制係数 \(C_{oe}\)。`C_{oe}>0` で、(i) total pressure `pq = Pe + Pi + Qvisc` の 5-cell `pq`+`rho` checkerboard を加速度計算前に弱く平滑化する filter（\(\beta_{eff}=0.15\min(C_{oe},1)\)、shock-supporting cell `Qvisc>0` は保護）と、(ii) 比体積 \(s=1/\rho\) の checkerboard に対する保存的 nodal damping force \(F^{oe}\) を有効化する。legacy PdV path では \(H^{oe}\) を別 heat として加え、`compatible_energy=True` の exact path では \(F^{oe}\) の force work を compatible work に含めて \(H^{oe}\) は別途加えない。`0` で両方とも無効。2D_RZ では未使用。NUMERICS §3.1.4 参照）
  - `anti_hourglass_C: float`（既定 `0.0` [無次元]；有効範囲：`[0, +∞)`；1D_SPH 専用。shell specific volume \(s=\log(V/\Delta M)\) の曲率から edge-based restoring force \(F^\pi\) を圧力/AV 加速度に追加する。compression speed scale \(U_c=\Delta r\max(0,-\nabla\cdot u)\) を用い、force は pressure/AV nodal force の 5% に cap する。same-force work \(\Delta E_i^\pi\) は signed internal energy increment として加え、`compatible_energy=True` の exact path では corrector stage の material energy 増分へ含める。`0` で無効。NUMERICS §3.1.4, §3.1.5 参照）
  - `ee_odd_even_C: float`（既定 `0.0` [無次元]；有効範囲：`[0, +∞)`；1D_SPH + 2T 専用。hydro の main energy/heating 更新後、EOS による `Te/Pe` 再閉包の直前に、電子比内部エネルギー `ee` に対する保存的 odd-even face filter を適用する係数 \(C_{ee}^{oe}\)。検出は pre-EOS の `ee` checkerboard を proxy に使い、void/inactive セルと first/last non-void cell は保護する。`0` で無効。推奨試験範囲は `0.05-0.15`。NUMERICS §3.1.5 参照）
  - `hk_velocity_damper_C: float`（既定 `0.0` [無次元]；有効範囲：`[0, 1]`；1D_SPH 専用。Corrector の節点速度更新後、最終密度再計算前に、3-node local linear fit で抽出した high-k velocity residual に保存的 pair impulse を適用する。失われた kinetic energy は exact pairwise formula で測定し、2T では常に ion energy `ei` へ、1T では total material energy `ee` へ加える。shock/ablation-front/material-interface/void では guard-cell 展開 front mask と secondary sensor により無効。`0` で無効。NUMERICS §3.1.4 参照）
  - `hk_velocity_damper_tau_min: float`（既定 `8.0` [無次元]；有効範囲：`[0, +∞)`；high-k velocity damper を有効にする最小 node optical depth。cell \(\theta_c=\sigma_{R,\max,c}\Delta r_c\) から隣接 cell の最大値を node \(\tau_j\) とし、pair 両端の最小値で判定する。\(\sigma_{R,\max}\) は直近の radiation stage で評価された group 最大 Rosseland opacity を使うため、Strang 分割の先頭 hydro half-step では1 radiation stage 古い値になる。`0` では optical-depth gate を無効化）
  - `hk_velocity_damper_grad_Te_max: float`（既定 `0.2` [無次元]；有効範囲：`[0, +∞)`；隣接セルの \(|\Delta\ln T_e|\) がこの値を超える face の両側 cell を front cell として扱い、さらに immediate pair stencil の secondary guard でも high-k velocity damper を無効化する）
  - `hk_velocity_damper_grad_rho_max: float`（既定 `0.3` [無次元]；有効範囲：`[0, +∞)`；隣接セルの \(|\Delta\ln\rho|\) がこの値を超える face の両側 cell を front cell として扱い、さらに immediate pair stencil の secondary guard でも high-k velocity damper を無効化する）
  - `hk_velocity_damper_guard_cells: int`（既定 `25` [cells]；有効範囲：`[0, +∞)`；high-k velocity damper の front mask を展開する cell 半幅。front cell は `Qvisc>0`、void cell、または上記 \(T_e/\rho\) log jump 閾値超過 face の両側 cell。pair \((i,i+1)\) は `near_front[i]` が真なら無効）
  - `av_heat_to: Literal["ion","electron","split"]`（既定 `"ion"`；legacy PdV path の人工粘性仕事、および odd-even damping heat/work の配分先。`"ion"` = イオンへ全量（v1.0既定）、`"electron"` = 電子へ全量。`"split"` = v1.0未実装 → `ConfigError`。`compatible_energy=True` の exact 1D_SPH 2T path では、force-derived \(Q\) work は v1 既定として ion energy へ含め、odd-even damping work はこの設定で選ぶ。NUMERICS §3.1.5 参照）
  - `compatible_energy: bool`（既定 `False`；1D_SPH 専用。True かつ ideal-gas hydro EOS path、または `eos.model="tmat"` かつ `eos.hydro_backend="legacy"` の TMAT table path では、Corrector 加速度に使った filter 後 total pressure \(p_q=P_e+P_i+Q\)、half-step node area、\(\bar u=(u^n+u^{n+1})/2\)、および同じ boundary ghost convention から cell ごとの exact force work を計算し、legacy \(V^{n+1}-V^n\) PdV/Q 更新を置き換える。`odd_even_damping_C>0` または `post_shock_velocity_damping_C>0` の implicit damping increment、および `anti_hourglass_C>0` の restoring force work increment は material energy 増分へ含め、別経路の \(H^{oe}\), \(H^\pi\) heat は加えない。2T では pressure/Q work を \(P_e\) と \(P_i+Q\) の非負正規化比で電子・イオンへ分配し、\(Q\) work は ion 側へ入れる。TMAT 2T では \(e_e\to T_e\), \(e_i\to T_i\) の独立 monotone inverse reclosure で \(P_e,P_i,c_v\) を再評価し、compatible work 後の `ee/ei` は table 範囲外 clamp または repair 時以外 writeback しない。damping/restoring work は `av_heat_to` で選ばれた熱容量場へ入れる。IONMIX / SESAME / Helmholtz / rho-e table / Mie-Gruneisen hydro EOS path は現実装では `ConfigError`。False で既存の体積差分 PdV に戻る。2D_RZ では未使用。NUMERICS §3.1.5 参照）
  - `rho_e_linear_grid: bool`（既定 `False`；`eos.hydro_backend="rho_e_table"` のときだけ有効。False では \((\log\rho,\log e)\) spline、True では元の `rho_grid` と一様線形 \(e\) grid による \((\rho,e)\) spline を使う。1D 診断専用で、production の既定経路は False）
  - `eos_writeback: bool`（既定 `True`；table backend (`legacy`, `helmholtz_spline`, `helmholtz_jet`, `rho_e_table`) の EOS closure で `e \leftarrow e(\rho,T)` re-projection を行うかどうか。既定 `True` では closure 後に table / surrogate の clamped energy を state へ writebackして旧来の毎 step re-projection を行う。`False` では Hydro が更新した `ee/ei` を保持し、NaN / Inf / 負の energy だけを repair として table / surrogate の clamped energy へ戻す。`compatible_energy=True` の TMAT reclosure 直後だけは compatible work の保存量を優先し、table 範囲内なら `eos_writeback=True` でも `ee/ei` を保持する）
  - `eos_closure_mode: str`（既定 `"energy_authoritative"`；許容値 `"legacy" | "energy_authoritative"`。EOS 閉包が table 逆算の lower/upper clamp（table 端で目標エネルギーが表域外）に遭ったときのエネルギー方針。`"legacy"` は pre-BUG-24 の再現用モードで、従来どおり内部エネルギーを clamp 後の表値で上書きする（BUG-24: blowoff コロナで超過エネルギーを毎ステップ無記帳で破棄しうる）。`"energy_authoritative"` は進化させた内部エネルギーを権威量として保持し、T/P のみ clamp 逆算値を用いる。非有限入力の修復と bracket 失敗時の安全修復は両モードで同一。1D 閉包に適用（2D は追って）。docs/design/bug24_hydro_entry_eos_projection_20260718.md 参照）
  - `exact_override: Literal["none","pressure","sound_speed","temperature"]`（既定 `"none"`；1D の table backend (`legacy`, `helmholtz_spline`, `helmholtz_jet`, `rho_e_table`) に対する診断用 A/B 分離スイッチ。`"pressure"` は EOS closure 後の `Pe/Pi` のみ、`"sound_speed"` は sound speed のみ、`"temperature"` は `Te/Ti` のみを \(\gamma=5/3\), \(c_{v,i}=1.5\,k_B/(A\,m_p)\), \(c_{v,e}=1.5\,Z\,k_B/(A\,m_p)\) の ideal-gas 式で上書きする。既定 `"none"` は無効。legacy diagnostic value `"no_writeback"` も引き続き受理され、この場合は `eos_writeback=True` 指定時でも 1D table path の energy writeback を無効化する。TMAT 物理とは互換でなく、合成 ideal-gas table の診断以外には使わない）
  - 1D球対称の shock support は `av_type` で切り替わる。`"vnr"` では Christensen 圧縮センサ \(\chi_i\) に加えて
    \(\phi_i = W_{shock,i}\max(0.25,\;W_{comp,i}W_{osc,i})\) を内部的に掛ける。
    `"riemann"` では face Riemann pressure correction をそのまま cell center へ平均した \(Q_i\) を既存経路へ渡し、
    VNR の jump-based `W_shock` / mild-compression gate / `W_osc` は適用しない。
    `"riemann_compatible"` では cell midpoint velocity の minmod reconstruction と各節点速度との差から左右の nonlinear impedance pressure \(q_L,q_R\) を求め、\(Q_i=0.5(q_L+q_R)\) を既存の momentum / PdV compatible-work 経路へ渡す。
    この形式にも VNR の jump-based `W_shock` / mild-compression gate / `W_osc` と `av_eos_aware` は適用しない。
    `"csw"` では node velocity の van Leer または Barth-Jespersen limited reconstruction から
    \(\chi^{lim}\) を作り、専用係数 `csw_C1`, `csw_C2` で \(Q_i\) を計算する。
    CSW でも VNR の jump-based gates と `av_eos_aware` は適用しない。
    VNR の `W_shock` は total pressure \(P=P_e+P_i\) と密度 jump に基づく developed shock /
    pressure-dominated precursor の2分岐、`W_comp` は圧縮Mach数、`W_osc` は Quirk 型 odd-even 抑制である。
    閾値 `kShockPressureJumpThreshold=0.3`,
    `kShockDensityJumpThreshold=0.05`,
    `kShockRhConsistencyThreshold=0.5`,
    `kCompMachScale=0.05`,
    `kOscillationThreshold=0.2`,
    `kShockSupportFloor=0.25`
    は `src/hydro/artificial_viscosity.cu` の内部定数であり、v1.0 では namelist 非公開とする。
    これは Sedov blast の強 shock support を保ちながら、GXII shell の source-heated front を
    人工粘性で過剰加熱しないための実装上の固定値である（NUMERICS §3.1.6）。
  - `boundary`：次元に応じて型が異なる（ARCHITECTURE §4.1 Config::HydroConfig 参照）。
    - **1D_SPH**：`boundary: Literal["free","fixed","reflect","pressure"]`（既定 `"free"`；外側境界条件。`"free"` = P_ext=0（ICF標準）、`"fixed"` = 速度固定壁（v=0）、`"reflect"` = スリップ壁（v_n=0、v_t自由）、`"pressure"` = 外部駆動圧力。NUMERICS §8.1参照）
    - **2D_RZ**：`boundary: dict`（per-face 指定）
      - `r_inner: Literal["axis"]`（既定 `"axis"`、変更不可：R=0 対称軸、v_r=0 強制。ARCHITECTURE §4.1 Config::HydroConfig）
      - `r_outer: Literal["free","fixed","reflect","pressure"]`（既定 `"free"`）。Multiblock physical outer-shell mesh-vector constraints use `"fixed"` = both components zero, `"reflect"` = spherical-normal component removed, and `"free"`/`"pressure"` = no normal-motion clamp.
      - `z_bottom: Literal["free","fixed","reflect"] | dict`（既定 `"free"`；`"pressure"` は `r_outer` のみサポート。`z_bottom`/`z_top` で `"pressure"` 指定は `ConfigError`）
      - `z_top: Literal["free","fixed","reflect"] | dict`（既定 `"free"`；同上）
      - `mesh_tangential_target: Literal["lagrangian","reference"]`（既定 `"lagrangian"`）。2D_RZ の clamped boundary face で tangential mesh coordinate を既存 Lagrangian 位置のままにするか、IC 時 reference mesh 座標へ戻すかを指定する。`"reference"` では r-face の z 座標と z-face の r 座標を `State.x_*_reference` に合わせる。
      - `state_supply_donor_mode: Literal["interior_per_i","interior_radial_average"]`（既定 `"interior_per_i"`）。state_supply z-face の ALE open-flow remap で outflow が interior donor を使う場合の donor 選択を指定する。`"interior_radial_average"` は境界 row の interior donor state を i 方向算術平均し、全 i に同じ donor を用いる。
      - `z_bottom`/`z_top` の dict 形式は v1.0 Wave 1 では `{"type":"state_supply", "rho_g_per_cc": float, "u_z_cm_per_s": float, "T_eV": float}` のみをサポートし、`rho_g_per_cc`, `u_z_cm_per_s`, `T_eV` はすべて必須である。z-face 境界セルの `rho/mass/Te/Ti` を供給値へ戻し、`Te=Ti=T_eV` として既存 EOS closure により `ee/ei/Pe/Pi` を再計算する。境界セルの material `v_z` は各 step の boundary override 後に `u_z_cm_per_s` へ復元・保持される。ALE 有効時に mesh normal を z_min/z_max に固定し mesh node `v_z` をゼロにしても、これは mesh anchoring のみであり material `v_z` をゼロ化しない。`state_supply` は r-face/1D 境界では `ConfigError` とする。文字列 `"state_supply"` は曖昧さ回避のため不可で、dict 形式を要求する。ALE open-flow remap と reservoir/ghost tally はこの supplied material `u_z_cm_per_s` を用いる。
  - `boundary_pressure: Optional[callable]`（`boundary="pressure"` 時の駆動圧力 \(P_{drive}(t)\) [dyne/cm²]。シグネチャ：`P(t_s: float) -> float` [dyne/cm²]。初期化時に FrozenTable1D 化（サンプリング: \(t \in [0, t_{end}]\) を10000等間隔点で評価し、線形補間テーブルとして凍結）。**端点外挿規則**：\(t < 0\) および \(t > t_{end}\) では \(P = 0\) とする（free境界と等価）。2D_RZ では `r_outer` 境界のみに適用。`boundary="pressure"` で未提供の場合は `ConfigError`）
- `conduction: dict`
  - `enabled: bool`（既定 True；電子熱伝導の有効化。False の場合：電子熱伝導を完全にスキップ（`dt_cond=∞`）。`ion_conduction` および `f_lim` は無視される。NUMERICS §4参照）
  - `solver: Literal["sts","implicit","hypre"]`（既定 `"sts"`；伝導ソルバの選択。`"sts"` = Super-Time-Stepping（明示的、Chebyshev加速）。`"implicit"` = 1D_SPH 専用の backward Euler + 三重対角直接解法（GPU `cusparseDgtsv2`、伝導CFL制約なし）。`"hypre"` = 2D_RZ 向け Hypre AMG+PCG（陰的、伝導CFL制約なし）。`"hypre"` は `-DTENRYU_ENABLE_HYPRE=ON` ビルド時のみ使用可能。未ビルド時に `"hypre"` 指定 → `ConfigError`。`"implicit"` を非対応形状またはMPI分割1Dで指定した場合は実行時に STS へフォールバックする。NUMERICS §4.2.1（STS）, §4.2.3（陰的）参照）
  - `sts_floor_limiter: Literal["net","donor"]`（既定 `"net"`；`"net"` は従来の正味出力制限、`"donor"` は流出和と donor 側係数による STS 温度下限保証を使用する。NUMERICS §4.2.1参照）
  - `ion_conduction: bool`（既定 False；イオン熱伝導。v1.0スコープ内だが既定OFF）
    - **v1.0スコープ内のオプション機能**：有効化時は電子伝導と同じ Kershaw 9点ソルバをイオン熱伝導度 κ_i で使用（NUMERICS §4参照）
    - 既定OFFの理由：典型的ICFプラズマでは κ_i ≪ κ_e（質量比 \(\sqrt{m_e/m_i}\) のオーダー）であり、電子伝導が支配的。高密度燃料コアなど κ_i が無視できない条件ではユーザが明示的に有効化する
  - `f_lim: float`（既定 0.06 [無次元]；有効範囲：`(0, 1]`；flux limiter係数。0.06=DRACO標準。1.0=制限なし（自由ストリーミング許容）。NUMERICS §4.1参照）
  - `spitzer_z_correction`（**deprecated no-op（2026-07-30）**: \(\gamma_0(Z)\) は基本式に常時組込み。`"auto"`/`"epperlein_short"` は受理して INFO のみ、`"off"` は ConfigError（固定係数経路は削除済み）。NUMERICS §4.1）
  - `mfp_limiter_C: float`（既定 0.0 [無次元]；有効範囲：`[0, +∞)`；電子平均自由行程制限係数。`>0` の場合、\(\lambda_{ei}\) とセル代表長 \(\Delta l\) から \(\kappa \leftarrow \kappa_{SH}\min(1, C\Delta l/\lambda_{ei})\) を適用し、非衝突領域でSpitzer伝導率を減衰。`0` は無効。NUMERICS §4.1参照）
  - `sts_damping: float`（既定 0.01 [無次元]；有効範囲：`(0, 1)`；Super-Time-Stepping (STS) のダンピングパラメータ ν。`solver="sts"` 時のみ使用。小さいほど加速率が高いが安定性マージンが減少する。0.01 は実用上十分な安定性と高加速率を両立。NUMERICS §4.2.1参照）
  - `sts_max_stages: int`（既定 40；有効範囲：`[1, 200]`；STS最大ステージ数 \(s_{max}\)。`solver="sts"` 時のみ使用。グローバルΔtは \(s_{max}(s_{max}+1)/2 \times \Delta t_{exp}\) まで伝導制約を緩和。40の場合、素朴法比で最大820倍のΔt緩和。NUMERICS §2.2(b), §4.2.1参照）
  - `sts_subcycle_eta: float`（既定 0.9 [無次元]；有効範囲：`(0, 1]`；`solver="sts"` 時のみ使用。`sts_max_stages` 上限で STS 安定性を維持するためのサブサイクル安全率。\( \Delta t_{sts,max} = \eta \cdot \frac{s_{max}(s_{max}+1)}{2}\Delta t_{exp} \) を超える場合、\(\Delta t\) を \(n_{sub}=\lceil \Delta t/\Delta t_{sts,max}\rceil\) に分割して各サブステップで STS を実行する）
  - `sts_total_stages_max: int`（既定 200000；有効範囲：`>= 0`、0 = 上限なし（legacy）；1 回の伝導適用が起動する総ステージ数 \(n_{sub}\times s\) の liveness 上限。潰れセルの \(\Delta t_{exp}\) 崩壊で \(n_{sub}\) が非有界化して実質ハングに至るのを防ぐ。超過時は state 未変更のまま driver full-step retry（dt/2）を要求し、予算枯渇時は診断付き abort。NUMERICS §4.2.1 参照）
  - `hypre_rtol: float`（既定 1e-8 [無次元]；有効範囲：`(0, 1)`；`solver="hypre"` 時のみ使用。PCG相対収束判定 \(\|r_k\|/\|r_0\| \le rtol\)。NUMERICS §4.2.3参照）
  - `hypre_max_iter: int`（既定 50；有効範囲：`[1, 500]`；`solver="hypre"` 時のみ使用。PCG最大反復数。NUMERICS §4.2.3参照）
  - `hypre_amg_coarsen: int`（既定 10；`solver="hypre"` 時のみ使用。BoomerAMG粗視化タイプ。10=HMIS（GPU向き）。NUMERICS §4.2.3参照）
  - `hypre_amg_relax: int`（既定 18；`solver="hypre"` 時のみ使用。BoomerAMG緩和タイプ。18=l1-Jacobi（GPU向き、atomic不要）。NUMERICS §4.2.3参照）
  - `hypre_amg_interp: int`（既定 6；`solver="hypre"` 時のみ使用。BoomerAMG補間タイプ。6=ext+i（Kershaw行列の対角優位性に適合）。NUMERICS §4.2.3参照）
  - `hypre_amg_levels: int`（既定 25；有効範囲：`[2, 50]`；`solver="hypre"` 時のみ使用。BoomerAMG最大レベル数。通常変更不要。NUMERICS §4.2.3参照）
  - `halo_strategy: Literal["every","adaptive"]`（既定 `"every"`；`solver="sts"` 時のSTS各ステージ間のハロー交換戦略。`"every"` = 毎ステージ交換（安全優先、v1.0推奨）、`"adaptive"` = s≤4で毎回、s>4で条件付き（|ΔT/T|>0.1超過時のみ追加交換）。NUMERICS §12.2.3参照）
  - `face_kappa_policy | string | "kirchhoff_same_material" | 面伝導率閉包: "kirchhoff_same_material"（同材料滑面 S_{5/2} 割線、界面/void/κ₀ 跳び >10x は調和 fallback）/ "harmonic"（歴史閉包; NUMERICS §4.1）`
    - Default flipped 2026-07-06 per user decision (rebaseline sprint).
  - `nonlocal_model: Literal["none","snb"]`（既定 `"none"`；SNB (Schurtz–Nicolaï–Busquet) 型非局所電子熱輸送の opt-in。`"none"` は従来の局所 Spitzer-Härm + flux limiter 経路を bit 恒等で保存。`"snb"` は群別拡散 H_g 方程式（cusparse 群バッチ三重対角）で SH 流束を非局所補正し、既存 `f_lim` cap を外側安全 cap として合成する（cap 発火は診断計上）。**1D_SPH（planar/cylindrical/spherical）+ `Main.temperature_model="2T"` + `conduction.enabled=True` 専用**。`per_material_conservation_enabled=True` との併用、2D_RZ、1T は `ConfigError`。設計 doc `docs/design/snb_nonlocal_1d_20260710.md`、NUMERICS §4.4 参照）
  - `snb_n_groups: int`（既定 24；有効範囲：`[2, ∞)`；`nonlocal_model="snb"` 時のみ使用。SNB エネルギー群数。群端は毎伝導ステップ β̂=E/(k_B max T_e) の幾何級数 [0.1, snb_E_max_over_Te]（+ [0, 0.1] の最下群）で再構築。24 群幾何で連続カーネル比 ≤0.8%（設計 doc §7C）。NUMERICS §4.4 参照）
  - `snb_E_max_over_Te: float`（既定 20.0；有効範囲：`(1, ∞)`；`nonlocal_model="snb"` 時のみ使用。群構造の最大エネルギー E_max/(k_B max T_e)。20 で未収容 tail ≤1.7e-5（未収容分は純 SH のまま扱われ保存を破らない）。NUMERICS §4.4 参照）
  - `snb_mfp: Literal["geometric_r2","original"]`（既定 `"geometric_r2"`；`nonlocal_model="snb"` 時のみ使用。群 mfp 変種。`"geometric_r2"` = Sherlock 2017 Eq(1) の幾何平均 φ 補正形 λ_g=2√2 β̄_g² λ_0、λ_0=T_e²/(4πn_e√(Z̄φ)e⁴lnΛ)、φ=(Z̄+4.2)/(Z̄+0.24)（Brodrick r=2 等価、高 Z で VFP 一致改善）。`"original"` = Schurtz 2000 Eq(3)+(23) の λ_g=2β̄_g²λ_B/√(Z̄+1)。NUMERICS §4.4 参照）
  - `snb_efield: Literal["none","local"]`（既定 `"none"`；`nonlocal_model="snb"` 時のみ使用。電場停止長による輸送 mfp 補正（Schurtz Eq 32–35）。既定 `"none"`：`geometric_r2` の r=2 較正は電場抑制を織込み済みで、重ねると二重計上になる（Sherlock 2017 §VII の裁定）。`"original"` 変種と併用する場合の推奨は `"local"`。NUMERICS §4.4 参照）
  - `snb_picard_max_iters: int`（既定 8；有効範囲：`[2, ∞)`；`nonlocal_model="snb"` 時のみ使用。iSNB Picard（Cao 2015 §IV）反復上限。収束せず上限到達時は warn + 診断記録の上で最終反復を採用（沈黙しない）。NUMERICS §4.4 参照）
  - `snb_picard_rtol: float`（既定 0.01；有効範囲：`(0, ∞)`；`nonlocal_model="snb"` 時のみ使用。Picard 収束判定（max-norm の δq 変化 ≤ rtol × max(|q_sh|+|δq|)、Cao の α 判定に相当）。NUMERICS §4.4 参照）
  - `nonlocal_model: Literal["none","snb"]`（既定 `"none"`；SNB (Schurtz–Nicolaï–Busquet) 型非局所電子熱輸送の opt-in。`"none"` は従来の局所 Spitzer-Härm + flux limiter 経路を bit 恒等で保存。`"snb"` は群別拡散 H_g 方程式（1D: cusparse 群バッチ三重対角 / 2D: Kershaw 9-point 対称化 CSR + 群バッチ Jacobi-PCG）で SH 流束を非局所補正し、既存 `f_lim` cap を外側安全 cap として合成する（cap 発火は診断計上）。**この tree では 2D_RZ + `Main.temperature_model="2T"` + `conduction.enabled=True` + `conduction.solver="sts"` 専用**（1D_SPH 実装は 1d-brushup 系譜 — merge train で次元ガードが union になる）。2D は Kershaw 9-point 対称化ステンシル CSR + 群バッチ Jacobi-PCG（cusparse 三重対角の 2D 対応物）。`per_material_conservation_enabled=True` との併用、1D_SPH、1T、`solver!="sts"` は `ConfigError`。`snb_efield="local"` は 2D v1 で `ConfigError`（fail-closed）。設計 doc `docs/design/2d_snb_port_spec.md`（1D: `docs/design/snb_nonlocal_1d_20260710.md`）、NUMERICS §4.5 参照）
  - `snb_n_groups: int`（既定 24；有効範囲：`[2, ∞)`；`nonlocal_model="snb"` 時のみ使用。SNB エネルギー群数。群端は毎伝導ステップ β̂=E/(k_B max T_e) の幾何級数 [0.1, snb_E_max_over_Te]（+ [0, 0.1] の最下群）で再構築。24 群幾何で連続カーネル比 ≤0.8%（設計 doc §7C）。NUMERICS §4.5（2D_RZ；1D は merge 後の §4.4）参照）
  - `snb_E_max_over_Te: float`（既定 20.0；有効範囲：`(1, ∞)`；`nonlocal_model="snb"` 時のみ使用。群構造の最大エネルギー E_max/(k_B max T_e)。20 で未収容 tail ≤1.7e-5（未収容分は純 SH のまま扱われ保存を破らない）。NUMERICS §4.5（2D_RZ；1D は merge 後の §4.4）参照）
  - `snb_mfp: Literal["geometric_r2","original"]`（既定 `"geometric_r2"`；`nonlocal_model="snb"` 時のみ使用。群 mfp 変種。`"geometric_r2"` = Sherlock 2017 Eq(1) の幾何平均 φ 補正形 λ_g=2√2 β̄_g² λ_0、λ_0=T_e²/(4πn_e√(Z̄φ)e⁴lnΛ)、φ=(Z̄+4.2)/(Z̄+0.24)（Brodrick r=2 等価、高 Z で VFP 一致改善）。`"original"` = Schurtz 2000 Eq(3)+(23) の λ_g=2β̄_g²λ_B/√(Z̄+1)。NUMERICS §4.5（2D_RZ；1D は merge 後の §4.4）参照）
  - `snb_efield: Literal["none","local"]`（既定 `"none"`；`nonlocal_model="snb"` 時のみ使用。電場停止長による輸送 mfp 補正（Schurtz Eq 32–35）。既定 `"none"`：`geometric_r2` の r=2 較正は電場抑制を織込み済みで、重ねると二重計上になる（Sherlock 2017 §VII の裁定）。`"original"` 変種と併用する場合の推奨は `"local"`。NUMERICS §4.5（2D_RZ；1D は merge 後の §4.4）参照。**2D_RZ v1 では "local" は ConfigError（fail-closed）**）
  - `snb_picard_max_iters: int`（既定 8；有効範囲：`[2, ∞)`；`nonlocal_model="snb"` 時のみ使用。iSNB Picard（Cao 2015 §IV）反復上限。収束せず上限到達時は warn + 診断記録の上で最終反復を採用（沈黙しない）。NUMERICS §4.5（2D_RZ；1D は merge 後の §4.4）参照）
  - `snb_picard_rtol: float`（既定 0.01；有効範囲：`(0, ∞)`；`nonlocal_model="snb"` 時のみ使用。Picard 収束判定（max-norm の δq 変化 ≤ rtol × max(|q_sh|+|δq|)、Cao の α 判定に相当）。NUMERICS §4.5（2D_RZ；1D は merge 後の §4.4）参照）
- `ale1d: dict`（1D_SPH solution-adaptive ALE V3。既定 `enabled=False`。Week 6 では GPU sensors、feature list API、monitor、common node mask、scratch candidate rezone、MUSCL/minmod remap scratch API、velocity projection scratch API、scratch diagnostics、および hard-tolerance gated two-phase commit driver を提供する。**動作境界（2026-07-26 fail-closed 化、AI review k16 C3/C7/C8）**: `enabled=True` は 単一材料・`eos_model="ideal_gas"`・radiation は off または `multigroup_diffusion`・`Burn.enabled=False` を要求し、違反は `ConfigError`（`sn_transport` は角度状態 ψ が、burn は在庫場が remap 随伴されないため。post-remap EOS reclosure/音速は ideal-γ 式のため table EOS も不可）。V3 は default-off の研究 prototype であり、この境界外での使用は certify されていない）
  - `enabled: bool`（実験的。既定 `False`。NUMERICS §3.4 の localized moving feature cases に限って opt-in 推奨。True は `Main.dimension="1D_SPH"` かつ deterministic radiation `mode in {"multigroup_diffusion","sn_transport"}` のみ有効。`mode="imc_ddmc"` は `ConfigError`）
  - trigger: `every_n_steps=100`, `min_steps_between_ale=50`, `enable_benefit_gate=True`, `benefit_min_dt_gain=1.5`, `candidate_dt_penalty_max=1.25`, `emergency_enabled=True`
  - eligibility: `min_cells=256`, `protected_fraction_max=0.25`, `min_movable_segment_warn=24`, `min_movable_segment_hard=8`, `max_node_displacement_fraction_mu=0.35`, `max_node_displacement_fraction_r=0.35`
  - tolerances are `{soft, hard}` dictionaries: `total_mass_tol={1e-12,1e-9}`, `material_mass_tol={1e-11,1e-8}`, `radiation_group_energy_tol={1e-8,1e-5}`, `material_internal_energy_tol={1e-8,1e-5}`, `total_material_energy_tol={1e-7,1e-5}`, `global_total_energy_tol={1e-6,1e-4}`, `kinetic_energy_drift_tol={1e-7,1e-5}`. For all tolerances `soft <= hard` is required.
  - diagnostics: `diagnostics_enabled=True`, `diagnostics_log_every_n_steps=100`, `diagnostics_collect_step_result=True`, `diagnostics_fail_on_unexpected_apply=False`
  - `laser_sensor: dict`: `enabled=True`, `target_cells_fraction=0.060`, `sigma_min_cells=4`, `sigma_max_cells=16`, `peak_fraction=0.35`, `conf_low=0.10`, `conf_high=0.40`
  - `ablation_sensor: dict`: `enabled=True`, `target_cells_fraction=0.080`, `sigma_min_cells=3`, `sigma_max_cells=14`, `peak_fraction=0.40`, `reference_density_gcc=1.05`, `rho_gate_frac=0.07`, `rho_gate_width=0.02`, `te_gate_low_eV=0.5`, `te_gate_high_eV=2.0`, `conf_low=0.10`, `conf_high=0.35`
  - `shock_sensor: dict`: `enabled=True`, `target_cells_fraction=0.040`, `sigma_min_cells=2`, `sigma_max_cells=8`, `peak_fraction=0.35`, `qvisc_conf_low=0.03`, `qvisc_conf_high=0.10`, `du_cs_conf_low=0.03`, `du_cs_conf_high=0.15`
  - `interface_sensor: dict`: `enabled=True`, `target_cells_fraction=0.033`, `target_cells_cap_fraction=0.067`, `max_features=8`, `min_separation_cells=4`, `jump_low=0.05`, `jump_high=0.25`, `sigma_min_cells=2`, `sigma_max_cells=4`, `pin_interfaces=True`
  - `center_sensor: dict`: `enabled=True`, `target_cells_fraction=0.053`, `sigma_min_cells=6`, `sigma_max_cells=20`, `search_x=0.12`
  - `rezone: dict`: `monitor_floor=1.0`, `monitor_wmax_ratio=50.0`, `monitor_smoothing_iterations=2`, `monitor_smooth_across_protected_faces=False`, `min_floor_fraction=0.55`, `gaussian_truncation_sigma=3.0`, `spatial_monitor_enabled=True`, `spatial_target_cells_fraction=0.067`, `spatial_power=2.0`, laser spatial \(\Delta r\) clip `2.5e-5..2.0e-4` cm, ablation `1.5e-5..1.2e-4` cm, shock `1.0e-5..8.0e-5` cm.
  - `remap: dict`: `reject_multicell_sweeps=True`, `high_order_enabled=True`, `limiter_theta=1.5`, `high_order_ramp_cells=2`, `radiation_high_order_ramp_cells=2`, `fallback_to_first_order_on_bounds_fail=True`, `reject_strict_zero_flux_on_moving_protected_face=True`. `high_order_enabled=False` keeps the Week 4 first-order donor remap path.
  - validation: `protected_fraction_max` and both displacement fractions must be in `(0,0.5)`, `min_movable_segment_hard >= 4`, and `min_movable_segment_warn >= min_movable_segment_hard`. Rezone monitor floor, truncation, spatial power, and spatial \(\Delta r\) bounds must be positive; `monitor_wmax_ratio >= 1`, `min_floor_fraction in (0,1)`, and `spatial_target_cells_fraction in [0,1)`. Remap `limiter_theta` must be positive, and remap ramp cell counts must be nonnegative.
- `coulomb_log_floor: float`（既定 `2.0` [無次元]；有効範囲：`[1.0, 30.0]`；クーロン対数 \(\ln\Lambda\) の全体的な下限値。電子-イオン緩和（NUMERICS §1.1.3）、熱伝導（NUMERICS §4.1）、レーザーIB吸収（NUMERICS §5.4）の全てに適用。`Laser.absorption.coulomb_log_floor` が別途指定された場合は IB 吸収のみをその値で上書きする。未指定の場合は `Numerics.coulomb_log_floor` にフォールバック）
- `splitting: dict`
  - `order: Literal["strang"]`（既定 `"strang"`；`L(Δt) → H(Δt/2) → C(Δt) → R(Δt) → H(Δt/2)`。Laser はステップ先頭で full-step 外部ソースとして1回のみ適用。Hydro↔(C-R)結合は2次精度、Laser↔(H-C-R) および C-R 相互間は1次精度（Lie splitting）。NUMERICS §2.1参照）
- `positivity: dict`
  - `clamp: bool`（既定 True；温度・密度フロアへのクランプを有効化。False の場合は負温度・負密度が発生しうる（デバッグ用のみ推奨）。NUMERICS §1.1.7参照）
  - `Te_min_eV: float`（既定 `1e-3`；有効範囲：`> 0`；`Mesh.floors.Te_floor_eV` のエイリアス。**v1.0 で非推奨（deprecated）**。使用時は WARNING 出力。実効値は `max(Mesh.floors.Te_floor_eV, Numerics.positivity.Te_min_eV)`。`Mesh.floors.Te_floor_eV` が唯一の権威ソース。推奨は `Mesh.floors.Te_floor_eV` のみを使用すること）
- `safety: dict`
  - `energy_fatal: bool`（既定 False；エネルギー収支 \(\varepsilon_{budget}\) が閾値を超えた場合に停止するか。NUMERICS §11.1参照）
  - `nan_fatal: bool`（既定 True；NaN/Inf検出時に fatal 停止するか。`energy_fatal` とは独立に制御。NaN伝播は数値破綻の兆候であり、通常は True を推奨）
  - `energy_threshold: float`（既定 `1e-3` [無次元]；有効範囲：`(0, 0.1]`；\(\varepsilon_{budget,max}\)。`energy_fatal=False` の場合も WARNING は出力。`Diagnostics.energy_budget.warn_threshold` と同値を推奨。NUMERICS §11.1参照）
  - `overshoot_warn: float`（既定 `0.01` [無次元]；有効範囲：`(0, 1.0]`；最大原理違反率 \(\delta_{overshoot}\) がこの値を超えた場合に WARNING。NUMERICS §11.8参照）
  - `overshoot_fatal: float`（既定 `0.10` [無次元]；有効範囲：`(0, 1.0]`；最大原理違反率がこの値を超えた場合に FATAL。`safety.overshoot_fatal_enabled=False`（既定）時は無視。NUMERICS §11.8参照）
  - `overshoot_fatal_enabled: bool`（既定 False；True で `overshoot_fatal` 閾値超過時に FATAL 停止。NUMERICS §11.8参照）
  - `clamp_warn_threshold: int`（既定 100；1ステップ内のフロアクランプ回数がこの閾値を超えた場合 WARNING 出力。ARCHITECTURE §4.1.2 SafetyConfig 準拠）
  - `clamp_fatal_threshold: int`（既定 10000；1ステップ内のフロアクランプ回数がこの閾値を超えた場合 FATAL 停止。ARCHITECTURE §4.1.2 SafetyConfig 準拠）
  - `opacity_floor: float`（既定 `1e-20` [cm²/g]；不透明度の下限。ゼロ不透明度による0除算を防止）
  - `opacity_cap: float`（既定 `1e20` [cm²/g]；不透明度の上限。数値オーバーフロー防止）
  - **相互整合チェック（Phase 2）**：`opacity_floor >= opacity_cap` → `ConfigError("safety.opacity_floor must be < safety.opacity_cap (got floor={floor}, cap={cap})")`
- `diagnostics_every: int`（既定 1；有効範囲：`≥ 1`；Numericsレベルの診断計算頻度 [サイクル]。`Diagnostics.every` と独立に設定可能。`diagnostics_every` は内部安全チェック（フロアカウンタ等）の頻度、`Diagnostics.every` は出力頻度を制御）
- `diagnostics: dict`
  - `phase_resolved_energy: bool`（既定 False；True の場合のみ history に `energy/phase_diagnostic/*` を追加し、hydro 前・hydro 後/ALE 前・ALE 後の物質全エネルギー、内部エネルギー、運動エネルギー、および hydro/ALE 各相の差分を記録する。False では後方互換性のため当該 HDF5 dataset は作成しない）
  - `r_momentum_source_audit: bool`（既定 False；2D_RZ only. When enabled, accumulates the discrete r-impulse of the completed nodal force array each hydro step (device reduction after compatible corner-force scatter, Braginskii addition, boundary/outer-pressure, and mirror-force assembly) and reports a standalone line, independently of `production_audit`, at the existing conservation-totals cadence: the source-balanced R-momentum residual ΔP_r − I_r, where ΔP_r is the change of the node-mass R-momentum Σ m_node·u_r (node masses built by the same kernel the acceleration divides by), and secondary consistency value S_hoop = 2π·Σ_c p_c A_c. Here p_c=Pe_c+Pi_c is total thermodynamic pressure only and A_c is planar cell area; AV, subzonal, and viscous stresses are excluded from S_hoop. Log-only diagnostic (no gate, no abort); unsupported force paths, multiblock, and MPI multi-rank (node-sum ownership) emit one warning and disable it; OFF path is bitwise-identical to previous behavior.）
  - `dt_breakdown_history_enabled: bool`（既定 True；True の場合、history HDF5 に `/diagnostics/dt_breakdown_history/`, `/diagnostics/cfl_winner/`, `/diagnostics/per_row_mass/`, `/diagnostics/av_max/` を追加する。診断専用で、CFL・AV・境界条件・状態更新の数値経路は変更しない。`Output.history_every=0` でも本フラグが True なら history HDF5 は診断出力のため作成される）
  - `icf: dict`（既定 `{"enabled": False, "rho_inner_threshold_g_per_cc": 0.0, "rho_outer_threshold_g_per_cc": 0.0}`；True または `Numerics.profile.icf_standard_ale.enabled=True` の場合のみ `/diagnostics/icf/v1/` history に shell radius, thickness, IFAR, CR を出力する。threshold 0 は自動設定で、inner は `0.5*rho_peak`、outer は `0.1*rho_peak`。有効範囲：threshold は `>= 0`）
  - `hotspot_gas: dict`（既定 `{"enabled": False, "R_g_cm": 0.0, "mass_drift_warn_rel": 1e-10}`；True の場合のみ、初期セル重心 `sqrt(r^2+z^2)<R_g_cm` を gas tracer `Y_g=1` としてタグし、CSR conservative remap では `Q_g=mY_g` を hydro mass flux と同じ donor/limiter で輸送する。history HDF5 `/diagnostics/hotspot_gas/v1/` と snapshot HDF5 `/diagnostics/hotspot_gas/v1/` に gas tracer mass drift, gas-mass quantile CR50/90/95/99, normalized `C_R50_norm`, invariant volume-density compression `CR_V`, density-median compression `CR_rho50`, `rho_bar_initial`, CRrms, density/pressure/adiabat means and percentiles, hotspot gas-mass-weighted `hotspot_Te_*` (`mean,p10,p50,p90`) と 2T のみ `hotspot_Ti_*`, `hotspot_energy_*`, explicitly named `hotspot_work_proxy_*` を出力する。`hotspot_work_proxy_total_erg` は tracer-defined control volume の `sum(Y_g*m*(ee+ei+0.5*|u_cell|^2))-initial` であり、曖昧な tracer 境界の compatible pressure-work 積分ではない。これらは path-versioned diagnostic group への additive dataset で、namelist/checkpoint/HDF5 root schema は変更しない。測定専用で hydro state への feed-back はない。有効範囲：enabled 時 `R_g_cm>0`, `mass_drift_warn_rel>=0`）
  - `ale_velcoherence: dict`（既定 `{"enabled": False, "every_n_steps": 1}`；`enabled=True` または環境変数 `TENRYU_I1B_DISC_ALE_VELCOHERENCE=1` の場合のみ、multiblock CSR ALE step 内で `[ale_velcoherence]` 行を `s0_post_hydro`, `s1_post_rezone`, `s2_post_remap`, `s3_post_velproj` の各 checkpoint に出力する。各行は gas region の `M_gas`, mass-weighted radial velocity `mw_ur`, coherent radial kinetic energy `rad_ke`, total gas kinetic energy `tot_ke` を報告する測定専用診断で、数値経路・状態更新・HDF5 schema は変更しない。有効範囲：`every_n_steps>=1`）
  - `conservation: dict`（既定 `{"enabled": False}`；True または `Numerics.profile.icf_standard_ale.enabled=True` の場合のみ、clean boundary と明示的 `delta_E_ext` を持つ演算子について `/diagnostics/conservation/v1/` に per-operator energy residual を出力する）
  - `ale_provenance_emission: dict`（既定 `{"enabled": False}`；True または `Numerics.profile.icf_standard_ale.enabled=True` の場合のみ `/diagnostics/ale_provenance/v1/` に ALE provenance counters と run-end attributes を出力する）
  - `mesh_quality_min: dict`（既定 `{"enabled": False}`；True の場合のみ accepted committed post-ALE mesh の all-run achieved minimum diagnostics を計算し、history HDF5 に `/diagnostics/mesh_quality_min/v1/` を additive/backward-compatible group として出力する。The additive field set includes corner/Gauss/RZ-volume minima, negative RZ-volume count, and mesh-conditioning edge-length/altitude/condition-number diagnostics. Multiblock evaluation uses CSR cell-node lookup and `cell_orientation_sign` for committed RZ-volume sign. False では per-step device-to-host mesh copy と host loop を実行せず、group も作成しない）
  - `shock_approach: dict` (default `{"enabled": False, "every": 50, "target_radius_cm": 0.0, "bins": 192, "h_cell_cm": 0.0}`; read-only shock-ahead diagnostic. When `enabled=True`, every `every` steps it theta-averages cell pressure radially to detect the `|dp/ds|` ridge and writes the fitted inward speed, predicted arrival time at `target_radius_cm`, and crossing-time lead to the `[shock_approach]` log. It changes neither the physical state nor the HDF5 schema. Valid range: `every>=1`, `bins>=16`, finite `h_cell_cm>=0`, and, when enabled, finite `target_radius_cm>0`. `h_cell_cm=0` uses `s_max/bins`.)
  - `mesh_attribution: dict`
    - `enabled: bool`（既定 False；2D_RZ mesh failure の per-source attribution JSONL diagnostics を有効化する master flag。False では buffer allocation と JSONL write は行わず、既存 path は bitwise default-off）
    - `record_node_displacements: bool`（既定 False；True かつ `enabled=True` の場合のみ per-source device displacement buffers を確保し、Hydro2D invocation ごとに direct mesh coordinate displacement を記録する）
    - `dump_on_failure_only: bool`（既定 True；True では mesh failure 時のみ `mesh_failure_attribution.jsonl` へ出力する。False では各 Hydro2D invocation 成功時にも aggregate record を出力する）
    - `enable_leave_one_out_replay: bool`（既定 False；Wave 5 では API stub のみで shadow replay は未実装）
  - `mesh_degeneracy_forensics: dict`
    - `enabled: bool`（既定 False；repeated pre-commit `mesh_quality_*` / `in_hydro_*` rejection の Phase A JSONL forensic dump を有効化する master flag。False では sample capture と JSONL write は行わず、既存 path は bitwise default-off）
    - `corner_j_source_budget_enabled: bool`（既定 False；True の場合のみ Phase D-1 corner-J source budget を forensic JSONL に追加する。pressure / scalar Q / anti-hourglass / predictor / corrector / ALE rezone / remap の線形化 \(\Delta J_q\) attribution を出力する。False では追加 snapshot と budget 計算を行わない）
    - `corner_j_source_budget_include_1_ring: bool`（既定 False；Phase D-1 予約フラグ。True は対象 cell の 1-ring も budget 対象にする意図を frozen config に保存する。現実装の JSONL emission は既存 forensic trigger cell を対象にする）
    - `velocity_history_enabled: bool`（既定 False；Phase D-3 multi-step velocity history JSONL を有効化する master flag。False では追加 sample capture と JSONL write は行わず、既存 path は bitwise default-off）
    - `velocity_history_target_cell_c: int`（既定 -1；`>=0` なら対象 cell linear index。`-1` では最初の eligible mesh-degeneracy failure cell を観測後に自動選択する）
    - `velocity_history_sample_every_n_steps: int`（既定 1；有効範囲 `>=1`；driver phase sampling stride [cycles]）
    - `velocity_history_include_1_ring: bool`（既定 True；True では対象 cell に加えて valid な 8-neighbor 1-ring cell も記録する）
    - `velocity_history_max_records: int`（既定 5000；run あたり velocity-history JSONL record 上限。0 は dump 無効。有効範囲：`>=0`）
    - `same_cell_count: int`（既定 3；同一 `(cell, corner, stage)` の連続 failure がこの回数以上で dump 対象。有効範囲：`>= 1`）
    - `sigma_threshold: float`（既定 0.5；`sigma_safe < threshold` なら連続回数に関係なく dump 対象。有効範囲：`0 < threshold <= 1`）
    - `max_dumps_per_run: int`（既定 100；run あたり JSONL dump 上限。0 は dump 無効。有効範囲：`>= 0`）
    - `output_dir: str`（既定 `""`；空なら `Output.directory`、非空ならその directory に `mesh_degeneracy_forensics.jsonl` を追記）
  - `production_audit: dict`（Wave 0 production-audit umbrella; commits cc62bad5, f56f8612, 547d894c, f08646b8）
    - `enabled: bool`（既定 `False`；all Wave 0 production-audit infrastructure の master gate）
    - `tier: str`（既定 `"none"`；有効値 `"A"`, `"B"`, `"none"`。`"A"` は verification-production、`"B"` は engineering production、`"none"` は production-audit tier claim なし）
    - `audit_json_path: str`（既定 `"<output_dir>/audit_summary.json"`；`tools/validation/audit_summary.py` が生成する `.audit.json` postprocess 出力先）
    - `escape_valve_budget: dict`
      - `mass_max: double`（既定 `0.0` [g]；有効範囲 `>= 0.0`。Tier-B cumulative mass-delta budget）
      - `energy_max: double`（既定 `0.0` [erg]；有効範囲 `>= 0.0`。Tier-B cumulative energy-delta budget）
    - `region_of_interest: list[dict]`（既定 `[]`；各要素は `{i_min:int, i_max:int, j_min:int, j_max:int}`。有効範囲：各 index `>= 0` かつ `i_min <= i_max`, `j_min <= j_max`。Tier-B で escape-valve firing を禁止する cell range）
    - `gcl: dict`
      - `enabled: bool`（既定 `False`；volume-closure residual hook。`gcl` は互換性のため保持する legacy config key）
    - `positivity: dict`
      - `enabled: bool`（既定 `False`；positivity scanner）
      - `fatal_on_neg: bool`（既定 `False`；negative event が1件でもあれば abort）
    - **Tier-A parse-time validation**：`production_audit.tier == "A"` かつ 6 escape-valve flags のいずれか（`Numerics.ale.emergency_cell_deactivation_enabled`, `Numerics.ale.multi_node_boundary_repair_enabled`, `Numerics.ale.multi_node_interior_repair_enabled`, `Numerics.ale.axis_variational_projection_enabled`, `Numerics.ale.local_boundary_repair_enabled`, `Numerics.hydro.driver_retry_active_mesh_repair_enabled`）が enabled の場合、builder validation は parse time に fatal config error を送出する。
- `profile: dict`
  - `icf_standard_ale: dict`（既定 `enabled=False`；public-baseline ALE characterization profile。False では既存 path と bitwise 同一。True では profile validator が allowed/forbidden policy を検査し、2D mesh geometry hard failure sites は profile observability のため typed soft-fail predicate を有効化する）
    - `enabled: bool`（既定 `False`）
    - `enforce: bool`（既定 `True`；True では policy mismatch を `ConfigError`、False では documented nonstandard run として WARNING）
    - `claim_level: str`（既定 `"characterization"`；`"characterization"`, `"pre_plic_smoke"`, `"production_comparable"` のいずれか）
    - `allowed_when_enabled: dict`：`ale_enabled_required_value=True`, `ale_axis_repair_mode_required_value="full_winslow"`, `ale_remap_scheme_allowed_values=["legacy_split","ms2_moments"]`, `ale_donor_sign_fixed_allowed_values: list<bool>`（既定 `[]` = both allowed；legacy profile key name, constrains `Numerics.ale.swept_volume_sign_fixed`）, `hydro_driver_full_step_retry_enabled_required_value=True`
    - `forbidden_when_enabled: dict`：`hydro_dispatcher_state_sensitive_bypass_enabled_forbidden_value=True`, `ale_local_boundary_repair_enabled_forbidden_value=True`, `ale_multi_node_boundary_repair_enabled_forbidden_value=True`, `ale_multi_node_interior_repair_enabled_forbidden_value=True`, `ale_axis_variational_projection_enabled_forbidden_value=True`, `ale_emergency_cell_deactivation_enabled_forbidden_value=True`, `hydro_driver_retry_active_mesh_repair_enabled_forbidden_value=True`
    - `escape_valves: dict`：`allow_nonstandard_mesh_rescue=False`, `require_deck_reason=True`, `mark_run_nonstandard=True`
  - `legacy_regression: dict`（既定 `enabled=False, revision="2026-07-27"`；bit-changing 数値規約を凍結する versioned regression profile。この revision は `Numerics.ale.swept_volume_sign_fixed=False`（legacy donor convention）を pin し、`icf_standard_ale` との併用を禁止する。有効化時は CI/regression 専用の warning を出力する。False では既存 path と bitwise 同一）
- `cell_search: dict`
  - `max_rings: int`（既定 3；有効範囲：`[1, 10]`；stencil walk失敗時のring expansion最大半径。値が大きいほどフォールバック成功率が上がるが計算コスト増。NUMERICS §9.4参照）
  - `fatal: bool`（既定 True；global fallbackでも未発見の場合に `FATAL` エラーで `MPI_Abort`。False の場合は粒子を消滅させ WARNING を出力）

#### 6.4.8 Output(...)
- `directory: str`（既定 `"./output"`。最大長: 256文字。ディレクトリが存在しない場合は自動作成（`mkdir -p` 相当）。既存ファイルがある場合は WARNING を出力。パス内の `~` はホームディレクトリに展開）
- `format: Literal["hdf5"]`（既定 `"hdf5"`；v1.0では唯一のオプション。§7参照）
- `plot_every: int`（既定 100；有効範囲：`≥ 1`；スナップショット出力間隔 [サイクル]。§7.2参照）
- `history_every: int`（既定 1；有効範囲：`≥ 1`；時系列出力間隔 [サイクル]。§7.3参照）
- `checkpoint_every: int`（既定 1000；有効範囲：`≥ 1`；チェックポイント出力間隔 [サイクル]。§7.4参照）
- `plot_every_s: float`（既定 -1.0 [s]；`> 0.0` で有効、`-1.0` で無効、`0.0` は `ConfigError`。有効時、Δtが出力時刻に整合される（NUMERICS §2.2 (f)）。`plot_every` とのOR論理で評価）
- `history_every_s: float`（既定 -1.0 [s]；`> 0.0` で有効、`-1.0` で無効、`0.0` は `ConfigError`。有効時、Δtが出力時刻に整合される（NUMERICS §2.2 (f)）。`history_every` とのOR論理で評価）
- `checkpoint_every_s: float`（既定 -1.0 [s]；`> 0.0` で有効、`-1.0` で無効、`0.0` は `ConfigError`。`X_every_s > t_end - t_current` の場合 WARNING（`t_current` は初回実行時=0、リスタート時=checkpoint時刻）。`checkpoint_every` とのOR論理で評価）
- `checkpoint_keep_last: int`（既定 2；有効範囲：`≥ 1`；保持するチェックポイント数。古いものから自動削除）
- `compression: Literal["none","gzip"]`（既定 `"gzip"`；HDF5データセット圧縮方式。§7.5参照）
- `compression_level: int`（既定 4；有効範囲：`[0, 9]`；gzip圧縮レベル。0=無圧縮（gzipヘッダのみ）、9=最大圧縮。`compression="none"` の場合は無視）
- `save_namelist_copy: bool`（既定 True；出力ディレクトリにnamelistソースファイルのコピーを保存。再現性用）
- `save_frozen_config: bool`（既定 True；出力ディレクトリに凍結設定（JSON）を保存。§3.3再現性要件に対応）

> **出力頻度の指定方式**：ステップ数ベース（`plot_every` 等）と時間間隔ベース（`plot_every_s` 等 [s]）を
> 同時に有効化可能（OR論理：いずれかの条件が成立すれば出力）。時間間隔ベースではΔtが出力時刻に整合される（NUMERICS §2.2 (f)）。
> ステップ0と最終ステップは設定に関わらず常に出力。ファイル命名は時間ベース出力でもサイクル番号（`NNNNNN`）を使用する。

#### 6.4.9 Diagnostics(...)
- `enabled: bool`（既定 True；False で全診断無効化）
- `every: int`（既定 1；有効範囲：`≥ 1`；diagnosticsの出力頻度。`every=10` で10ステップに1回計算・出力）
- `areal_density: dict`（ρR 面密度の線積分診断）
  - `enabled: bool`（既定 True）
  - `angles_deg: list[float]`（既定 `[0, 45, 90]`；対称軸からの角度 [度]。有効範囲：各要素 `[0, 180]`。1D_SPHでは`[0]`のみ有効（他の値は無視され INFO 出力）。各角度方向にρの線積分 \(\rho R = \int \rho\, dr\) を計算）
  - `r_range: Literal["full","shell"]`（既定 `"shell"`；`"full"` = 原点から外側境界まで全域積分、`"shell"` = 動的殻領域 \(\rho > 0.1\times\rho_{max}\) のみ積分。ARCHITECTURE §4.8 DiagnosticsConfig::ArealDensity に対応）
  - `Numerics.diagnostics.hotspot_gas.enabled=True` かつ gas tracer が初期化済みの場合、同じ角度列で tracer-masked \(\rho R_{hotspot,tracer}=\int\rho Y_g ds\) も出力する。これは `r_range="shell"` の殻マスクとは独立で、one-material pilot の hotspot-gas column density を表す。distinct fuel tracer は現行 state にないため `rhoR_fuel_tracer` は未生成；将来の multi-material deck は同じ versioned group に material-resolved hotspot/fuel/shell rhoR を追加する。
- `sphericity: dict`（RZ形状のLegendreモード分解。2D_RZ時のみ有効、1D_SPHでは自動的にスキップ）
  - `enabled: bool`（既定 True）
  - `modes: list[int]`（既定 `[0, 2, 4]`；評価するLegendreモード次数 \(P_\ell\)。有効範囲：各要素 `≥ 0`。偶数モードのみ物理的に有意（奇数も指定可能だが軸対称では理論上ゼロ））
  - `surface: Literal["isodensity","material_interface"]`（既定 `"isodensity"`；モード分解を行う表面の定義。`"isodensity"` = 指定密度の等値面、`"material_interface"` = 最外殻の材料界面）
  - `rho_threshold: float | str`（既定 10.0；有効範囲：`> 0` （float 指定時）または `'auto'`；`"isodensity"` 選択時の等値面密度 [g/cm³]。爆縮フェーズに応じて調整が必要。密度基準を満たすセルが存在しない場合、全モード振幅と shell_radius_mean に NaN を出力。動的閾値が必要な場合は `rho_threshold = 'auto'` を指定可能（内部で \(0.1 \times \rho_{max}\) を使用））
- `energy_budget: dict`（エネルギー収支の診断出力）
  - `enabled: bool`（既定 True）
  - `components: list[str]`（既定 `["laser_incident", "laser_deposited", "laser_escaped", "radiation_escaped", "marshak_in", "pdv_boundary", "numerical_loss", "kinetic", "internal_electron", "internal_ion", "radiation_field"]`；NUMERICS §10.2 恒等式に基づくエネルギー収支成分。保存誤差 = |ΔE_total - (E_in - E_out)| / E_in。有効なコンポーネント名は上記の固定セット + 診断専用: `"floor_injected"`, `"safety_injected"`, `"redistribution_unresolved"`, `"solver_residual"`。無効な名前は `ConfigError`。サブセット指定可能。保存誤差は物理成分のみから計算、診断専用成分は分子に含めない（§10.2参照））
    - `E_floor` / `E_safety` / `E_redistribution_unresolved` は安全機構の診断専用成分として常時出力するが、物理保存誤差（`conservation_error`）の分子には含めない（NUMERICS §10.2）
  - `warn_threshold: float`（既定 1e-3；有効範囲：`(0, 0.1]`；保存誤差がこの値を超えた場合に警告出力。IMC/DDMCの統計ノイズにより 1e-6 レベルの保存は困難なため、§3.1 の物理要件（時間平均保存誤差 ≤ 0.1%）と整合する 1e-3 を既定とする。検証テスト（解析解比較等）では 1e-6 等へ縮小可能）
- `laser_pattern: dict`（レーザー照射パターン診断。laser.enabled=True時のみ有効）
  - `enabled: bool`（既定 True）
  - `absorbed_power_profile: bool`（既定 True；臨界面近傍の吸収パワー密度分布を出力）
  - `critical_surface: bool`（既定 True；臨界面位置 \(R_{crit}(\theta)\) を出力（2D_RZ時は角度分解、1D_SPHではスカラー））
  - `per_beam: bool`（既定 False；True でビーム毎の吸収分率を個別出力。既定はグループ合計のみ）
- `mc_stats: dict`（モンテカルロ粒子統計）
  - `enabled: bool`（既定 True）
  - `particle_counts: bool`（既定 True；IMC/DDMC/census/absorbed/escaped/leaked の粒子数をステップ毎に出力）
  - `weight_stats: bool`（既定 True；粒子重みの min/mean/max をステップ毎に出力）
  - `cell_particle_density: bool`（既定 False；True でセル毎の粒子数分布をsnapshot出力に含める。大規模計算ではストレージ増加に注意）
  - `ddmc_fraction: bool`（既定 True；全粒子中のDDMC粒子割合を出力）
- `fleck_diag: dict`（1D_SPH 向けの Fleck/NLTE ステップ診断。標準出力へ構造化1行ログを出す）
  - `enabled: bool`（既定 False；`Main.verbosity="verbose"` でも有効化可能）
  - `every: int`（既定 10；有効範囲：`≥ 1`；`[fleck_diag]` ログの出力頻度 [step]）
  - `cells: list[int]`（既定 `[]`；監視するローカルセル番号の固定リスト）
  - `r_min_cm: float`（既定 -1.0；`r_max_cm` と対で指定。セル中心半径がこの下限以上のセルを監視）
  - `r_max_cm: float`（既定 -1.0；`r_min_cm` と対で指定。セル中心半径がこの上限以下のセルを監視）
  - `cells` と半径窓は併用可能で、選択セルの和集合を監視する。半径指定は cgs 長さ単位 [cm] を用いる（例：55–58 μm は `5.5e-3`–`5.8e-3` cm）
  - 出力形式：
    ```text
    [fleck_diag] step=N cell=C Te=X rho=X cv_e=X sigma_p_em=X beta=X f=X eta_tot=X E_emit=X dep_sum=X delta_E=X
    ```
    ここで `Te`, `rho`, `cv_e`, `beta`, `sigma_p_em`, `f`, `eta_tot` はその放射ステップで係数評価に使用した値、`E_emit`, `dep_sum`, `delta_E` は同ステップのセル別放射エネルギー収支
- `per_operator_radial_fourier_enabled: bool`（既定 False；2D_RZ の Strang-stage 境界で radial-null-mode Fourier audit を実行する。診断専用で状態更新には書き込まない）
- `radial_fourier_window_t_start_s: float`（既定 `1.35e-5` [s]；audit 有効時間窓の開始、inclusive）
- `radial_fourier_window_t_end_s: float`（既定 `1.70e-5` [s]；audit 有効時間窓の終了、exclusive。有効範囲：`>= radial_fourier_window_t_start_s`）
- `radial_fourier_max_mode: int`（既定 `-1`；`-1` は Nyquist まで全 radial mode、`>=0` は監査する最大 radial mode index）
- `per_operator_radial_fourier_complex_enabled: bool`（既定 False；PR G2-A 固定 `(m,j)` complex-coefficient radial Fourier audit を有効化する。既定 OFF で v1 `A_max` audit と physics update path は変更しない）
- `per_operator_radial_fourier_complex_m_targets: list[int]`（既定 `[14,15,16]`；固定 coefficient を出力する radial mode index。実行時 `0 <= m <= floor(nr/2)` の範囲外は無視する）
- `per_operator_radial_fourier_complex_j_targets: list[int]`（既定 `[507,508,509,510,511]`；固定 coefficient を出力する z-index。実行時 mesh 外の index は無視する）
- `per_operator_radial_fourier_complex_fields: list[str]`（既定 `["rho","M","V","M_over_V","P_r","P_z","u_r","u_z","E_e","E_i","E_rad","T_e","T_i","x_r","x_z","A_r","A_z","Q_visc","f_Fleck"]`；未公開または cell-wise に利用できない field は `NOT_AVAILABLE` として silently skipped。現在の deferred fields: `dV_swept`, `lambda_FLD`, `R_FLD`, `kappa_eff`, `newton_iters`, `newton_residual`）
  - Build-time `TENRYU_RFA_V2_MODE` further gates this v2 path: `OFF` and `STUB` must not launch v2 kernels or append v2 HDF5 rows; `DUMMY_BUFFER` launches kernels but suppresses HDF5 append; `FULL` preserves the schema and behavior above.
- `overshoot_monitor: bool`（既定 True；放射演算子後の温度最大原理違反を監視。NUMERICS §11.8参照）

**HDF5出力パスマッピング**：`energy_budget` → `energy/*`、`mc_stats` → `mc_stats/*`（= `mc/*` in history）、`areal_density` → snapshot `/diagnostics/areal_density/v1/{rhoR,rhoR_hotspot_tracer}`（history `implosion/rho_R`, optional `implosion/rho_R_hotspot_tracer`）。各診断は `every` ステップごとに出力。

#### 6.4.10 Parallel(...)
MPI並列計算の設定。数理詳細は NUMERICS §12、モジュール設計は ARCHITECTURE §7 参照。

**v1 実装状態（M18、2026-07；NUMERICS §12.1.4a Option C）**：
- 実装済み・検証済み：r-slab 分割（1D/2D）、ハロー交換（g=2）、`halo.gpu_aware_mpi`、
  決定論ソルバ（FLD/SN/burn/laser）の rank 結合（VERIFICATION §16 gate 群）。
- 2D の z 分割・cartesian 分割はハロー機構としては動作するが物理 gate 未検証、
  かつレーザー/deposit の全域 gather は r-slab を FATAL assert で要求する。
- `migration.*` / `particle_balance.*` は IMC 退役により **LEGACY-inactive**
  （光子粒子は存在しない。NUMERICS §12.3 の LEGACY 注記参照）。
- `reproducibility.mode="statistical"` の記述は §16 の tier 体系
  （T-bit / T-sum / T-noise、VERIFICATION §16）に置き換えられた：決定論経路は
  P 間 bitwise（T-bit）を gate で保証する。

- `decomposition: dict`（領域分割設定）
  - `method: Literal["slab","cartesian"]`（**次元依存既定値**: 1D_SPH → `"slab"`（`"cartesian"` 指定時は `ConfigError`）、2D_RZ → `"cartesian"`（`"slab"` も許可されるが非推奨、WARNING出力）。NUMERICS §12.1参照）
  - `dims: Optional[list[int]]`（2Dカート分割の次元 `[P_r, P_z]`；既定 `None`（自動決定）。手動指定時は `P_r × P_z = N_ranks` 必須（不一致は `ConfigError`）。1D_SPH では無視。NUMERICS §12.1.2の通信面積最小化基準で自動決定）
    - `N_ranks` の解決規則：
      - `run`/`freeze`：`MPI_Comm_size` を使用（M16実装）
      - `validate --n-ranks <P>`：`P` を使用
      - `validate`（`--n-ranks` 省略）：`P_r × P_z = N_ranks` は判定保留（INFOを出力し、実行時チェックへ委譲）
  - `min_cells_per_rank: int`（既定 8；有効範囲：`[4, nr]`；1rankあたりの最小セル数。Kershawステンシル＋ゴースト1層の安全余裕。`N_cells_total / N_ranks < min_cells_per_rank` の場合は `ConfigError("Too many ranks for mesh size")`。`validate` では `--n-ranks` 指定時のみ厳密チェック。NUMERICS §12.1参照）
- `halo: dict`（ハロー交換設定）
  - `gpu_aware_mpi: Literal["auto","force","disable"]`（既定 `"auto"`；`"auto"`はCMake検出結果（`TENRYU_GPU_AWARE_MPI`）に従う、`"force"`は強制使用（非対応環境ではセグフォルト可能性あり）、`"disable"`はhost-staging強制（安全だが低速）。ARCHITECTURE §7.3参照）
- `migration: dict`（粒子移動設定）
  - `method: Literal["batch"]`（既定 `"batch"`；サブステップ末バッチ送信。v1.0では唯一のオプション。NUMERICS §12.3.1参照）
  - `max_substeps: int`（既定 32；有効範囲：`≥ 1`；バッチ間の最大サブステップ数。超過時：未配送粒子のエネルギーを **`E_numerical_loss`** として計上し消滅（物理的境界流出 `E_escaped` とは区別する）。`E_numerical_loss` は history 出力の `energy/numerical_loss` に記録される。全粒子の 0.1% を超える場合は `FATAL`（verify モード）または `ERROR`（通常モード））
  - `emigrant_threshold: int`（既定 1000；有効範囲：`≥ 1`；1回のバッチでこの数を超える場合に WARNING を出力。チューニング指標：閾値を頻繁に超える場合は `dt.max_s` の縮小か `max_substeps` の増加を検討。ARCHITECTURE §7.1参照）
  - `initial_capacity: int`（既定 10000；有効範囲：`≥ 1024`；rankごとの emigration バッファ初期容量）
  - `growth_factor: float`（既定 1.5；有効範囲：`(1.0, 4.0]`；容量不足時の拡張倍率）
- `laser_parallel: dict`（LaserMesh並列戦略）
  - `strategy: Literal["replicated"]`（既定 `"replicated"`；全rank複製方式。v1.0では唯一のオプション。将来 `"distributed"` 追加予定。NUMERICS §12.4.2参照）
- `particle_balance: dict`（粒子負荷分散）
  - `enabled: bool`（既定 False；work-stealing方式の有効化。単一GPU（1 rank）では無効化推奨（オーバーヘッドのみ））
  - `imbalance_threshold: float`（既定 1.5 [無次元]；有効範囲：`(1.0, 10.0]`；N_max / N_mean がこの値を超えた場合に発動。`particle_balance.enabled=False` の場合は無視。NUMERICS §12.6.2参照）
  - `method: Literal["work_stealing"]`（既定 `"work_stealing"`；v1.0では唯一のオプション。`particle_balance.enabled=False` の場合は無視。NUMERICS §12.6.2参照）
- `reproducibility: dict`（並列再現性設定）
  - `mode: Literal["statistical"]`（既定 `"statistical"`；統計的再現（主要量の平均・分散が一致）を保証。v1.0では唯一のオプション。将来 `"bitwise"` 追加予定。NUMERICS §12.7参照）
  - `sort_after_migration: bool`（既定 False；移動後のglobal_idソート。セルソート（`gpu_optimization.particle_sort_by_cell`）が局所性を担保するため、再現性目的のソートは不要。True にするとデバッグ時に粒子追跡が容易になるが計算コスト増。NUMERICS §12.3参照）
- `gpu_optimization: dict`（GPU性能最適化設定。Phase A/B最適化の制御）
  - `particle_sort_by_cell: bool`（既定 True；輻射演算子冒頭での Composite Key Sort（NUMERICS §6.5）。合成キーソートによりセルソート + dead compaction + IMC/DDMC分離を単一パスに融合し、粒子管理オーバーヘッドを~60%削減。False でフォールバック（CUB個別呼び出し：mode sync + NaN化 + CompactFlag + Partition + R7b resample。デバッグ用。mode sync/NaN化/R7b は NaN sentinel 不変条件の保証に必須）。**注意**: `tally_mode="warp"` と不可分 — `particle_sort_by_cell=False` かつ `tally_mode="warp"` は `ConfigError`）
  - `tally_mode: Literal["global","warp"]`（既定 `"warp"`；タリー集約方式。NUMERICS §10.3）
    - `"global"`：Stage 3のみ（global atomicAdd直接）。CC 7.0未満のフォールバック、デバッグ用
    - `"warp"`：Stage 1+3（warp-level `__match_any_sync` 集約 → global atomicAdd）。v1.0既定。CC 7.0+（Volta以降）必須。CC 7.0未満のGPUで `tally_mode="warp"` が指定された場合は自動的に `"global"` に降格し `WARNING("tally_mode='warp' requires CC>=7.0, falling back to 'global'")` を出力。セルソート（`particle_sort_by_cell=True`）と不可分
    - ~~`"warp_block"`~~：将来拡張（Persistent Warp との設計上の緊張あり、NUMERICS §10.3.4参照）。v1.0では選択不可
  - `compute_comm_overlap: bool`（既定 False；計算-通信オーバーラップ。NUMERICS §12.5.5参照。True で内部セル計算とハロー交換を非同期並列化。`gpu_aware_mpi="disable"` との併用は非推奨（host-staging がオーバーラップのメリットを大幅に減少させるため WARNING 出力）。ARCHITECTURE §5.6.2参照）

#### 6.4.11 Burn(...)
核燃焼カーネル（1D_SPH v1）。数理は NUMERICS §14、設計記録は
`docs/design/burn_kernel_1d_v1_design_20260710.md`。既定 OFF（bit 恒等契約）。

- `enabled: bool`（既定 `False`。`True` は `Main.dimension="1D_SPH"`（`Mesh.geometry_1d="spherical"` 必須）
  または `"2D_RZ"`。persistent path は自動拒否（multi-kernel 実行））
- `fuels: list[str]`（既定 `["DT","DD"]`；許容 `{"DT","DD","D3He"}` の空でない部分集合、
  重複不可。`"DD"` は D(d,p)T / D(d,n)³He 両分岐を有効化）
- `scheme: Literal["fraley","diffusion","mc"]`（既定 `"fraley"`。`"diffusion"` =
  Corman 1975 多群拡散（NUMERICS §14.7）、`"mc"` = 直線 CSDA Monte Carlo
  （NUMERICS §14.9、統計モード — bitwise 非適用・§0.3 MC 条項）。2D_RZ は
  `"local"`｜`"diffusion"` のみ（`"fraley"` は 1D 球面固有、`"mc"` は未移植で
  ConfigError）。`"local"` = 全量出生セル沈着（partition 適用）。）
- `mc_particles_per_cell: int`（既定 16；[1,4096]。scheme="mc" の slot・セル・
  ステップ毎サンプル数）
- `diffusion_groups: int`（既定 30；[4,512]。scheme="diffusion" のエネルギー群数）
- `diffusion_E_min_keV: float`（既定 20.0；(1,100]。熱化下端（群格子下限））
- `partition: Literal["li_petrasso","fraley"]`（既定 `"li_petrasso"`；`"fraley"`
  （Eq.4、DT-α 限定）は `fuels` に DD/D³He を含むと `ConfigError`）
- `screening: Literal["none","salpeter","chugunov_dewitt"]`（既定 `"none"`
  （補正なし、v1 と bit 恒等）。反応率の遮蔽増強 F=e^h：`"salpeter"` = 弱遮蔽
  （電子込み 2T Debye、有効域 h≪1）、`"chugunov_dewitt"` = CD09 A4 補間
  （イオン遮蔽・剛体電子背景、弱〜強結合）。NUMERICS §14.1 参照。v2 追加、additive）
- `fuel_materials: list[str]`（既定 `["DT"]`；宣言済み非 void Material 名。不一致/void は
  `ConfigError`）
- `x_D, x_T, x_He3: float`（既定 0.5/0.5/0.0；各 ≥0、総和 = 1±1e-6（非既定混合は
  3 つとも明示指定））
- `T_floor_keV: float`（既定 0.2；>0。反応率床（Bosch-Hale fit 有効域下端））
- `explicit_source_limit: float`（既定 0.2；(0,1]。dt 制限係数、hot_electron 意味論）
- `eps_deplete: float`（既定 0.1；(0,0.5]。substep あたり最大相対在庫変化）
- `subcycle_max: int`（既定 64；[1,4096]）
- `vf_threshold: float`（既定 1e-3；(0,1)。燃料域検出の volFrac 閾値）
- `neutron_heating: bool`（既定 False；v2-E。DT-n 14.049 / DD-n 2.449 MeV の
  2 線群・単一飛行 first-collision 加熱。燃料 D/T 在庫でのみ減衰（shell 材
  kerma は v3）。凍結断面積と設計は
  docs/design/burn_kernel_v2_20260710.md §E）
- `neutron_heating_n_mu: int`（既定 16；偶数、[2,64]。等方放出の
  Gauss-Legendre μ 求積次数）

**2D_RZ + ALE**：`Numerics.ale.conservative_remap_enabled=True` かつ single_block のみ許可
（種と飛行中スペクトルが CSR/構造格子 swept remap に質量整合で随伴）。
per_material_conservation / total_energy_remap_2d_rz / axis-band /
multiblock / hllc_z_flux / force_rezone / reference_barrier との併用は
ConfigError（fail-closed、NUMERICS §14.10）。

出力（すべて additive、schema version 不変、burn 無効時は一切出力されない）：
snapshot `hydro/burn_{rate,Q_e,Q_i,eps_cum,n_D,n_T,n_He3,n_He4,n_p}`、
checkpoint `time_state/{E_burn_released,E_burn_dep_e,E_burn_dep_i,E_burn_esc_charged,E_burn_esc_neutron,N_burn_neutrons_dt,N_burn_neutrons_dd}`、
history `burn/*`（step 5 種＋累積 7 種＋dt_limit；v2-E で
`burn/neutron_{dep_e,dep_i,degraded,escaped}` step 4 種を追加 — degraded は
散乱後未追跡中性子の残エネルギーで esc_neutron 内数）。burn 有効 run の restart は
burn 有効 checkpoint 必須（`hydro/burn_n_*` 欠損は起動時 `runtime_error`）。

### 6.5 入力検証とエラー処理

TENRYUは入力パラメータを3段階で検証する。全パラメータは **パース時**（Phase 1 + Phase 2）に検証される。物理的な相互整合性（例：メッシュ vs 材料互換性）は **init 時**（Phase 2）に検査される。実行時クランプ（Phase 3）はシミュレーション中に適用される。

**エラーメッセージ形式**：全エラーメッセージは以下の統一形式に従う：
```
TENRYU {LEVEL} [{module}]: {description} (got {value}, expected {constraint})
```
- `{LEVEL}`：`FATAL`, `ERROR`, `WARNING`, `INFO` のいずれか
- `{module}`：エラー発生モジュール名（`Config`, `Mesh`, `Materials`, `Radiation`, `Laser`, `Hydro`, `IO` 等）
- `{description}`：人間可読なエラー内容
- `(got ..., expected ...)`：具体値と制約を併記（省略可能な場合もある）
- 例：`TENRYU ERROR [Config]: Mesh.nr out of range (got -1, expected >= 4)`
- 例：`TENRYU WARNING [Materials]: EOS table extrapolation clamped (got T=0.001 eV, expected [0.01, 100.0] eV)`

**Phase 1: Python評価時（`tenryu validate` または `tenryu run`）**
- 型チェック：Python の型アノテーションに基づく
- 範囲チェック：§6.4 の各パラメータの有効範囲に基づく
- 未知キーチェック：各ブロック（`Main`, `Mesh`, `Materials` 等）の `**kwargs` に対し、§6.4 で定義されていないキーが含まれる場合は `ConfigError("Unknown parameter '{key}' in {block}. Did you mean '{closest_match}'?")` を送出する。Levenshtein距離最小の候補を提示
- 範囲外の場合：`ValueError` を送出し、パラメータ名・指定値・有効範囲を表示
- 例：`nr=-1` → `ValueError: Mesh.nr must be >= 4, got -1`

**Phase 2: 整合性検査（Config構築時）**
- 相互依存パラメータの検査：
  - `dimension="1D_SPH"` なのに `nz > 1` → エラー
  - `opacity.model="ionmix"` なのに `opacity.file` 未指定、かつ `eos.model` が `"ionmix"` でない（`eos.file` からのフォールバック不可）→ エラー（§6.4.3 `opacity.file` の条件付き省略規則参照）
  - `zbar.model="tabular"` かつ IONMIX ソースが不在（`eos.model≠"ionmix"` かつ `opacity.model` が `"table_nlte"`/`"ionmix"` でない かつ `zbar.table_file` 未指定）→ `ConfigError("zbar.model='tabular' requires IONMIX source: eos.model='ionmix', opacity.model in {'ionmix','table_nlte'}, or zbar.table_file")`
  - `safety.opacity_floor >= safety.opacity_cap` → `ConfigError`
- `radiation.enabled=False` なのに `ddmc.enabled=True` → 警告（DDMCは無視される）
- ファイル存在チェック：EOS/opacityテーブルファイルのパスを検証
- FrozenTable1D callable validation: 全ての callable（`LaserBeam.power`, `marshak_Tr_eV`, `boundary_pressure`）について、FrozenTable1D 構築時に全サンプル点で `isfinite` を検証。`NaN`/`Inf` が検出された場合は `ConfigError("Callable {name} returned non-finite value at t={t_sample}")` を送出。さらに `LaserBeam.power` と `marshak_Tr_eV` は非負を要求し、負値は `ConfigError("Callable {name} returned negative value at t={t_sample}")` を送出
- 整合性エラー：`ConfigError` を送出

> **開発マイルストーン注記**：M01 実装では `validate` を m1-profile（段階実装プロファイル）として運用し、
> EOS/opacity ファイル存在チェックおよび callable シグネチャ検証を延期してよい。
> これらは M05（ファイル存在）/M02（callable シグネチャ）で有効化し、最終的に本節の標準挙動へ収束させる。

**Phase 3: 実行時クランプ（シミュレーション中）**
- EOS テーブル範囲外：入力 (ρ, T) をテーブル境界にクランプ + 1回限りの警告
- 温度/密度フロア：NUMERICS §1.1.7 のフロア値を適用 + 診断カウンタを増加
- エネルギー保存違反：`safety.energy_threshold` 超過時に警告、`energy_fatal=True` なら停止

**エラーレベル**：
| レベル | 動作 |
|-------|------|
| FATAL | 即座に MPI_Abort（修復不能） |
| ERROR | チェックポイント書き出し後に終了 |
| WARNING | メッセージ出力、実行継続（同一メッセージは最大10回） |
| INFO | 冗長モード時のみ出力 |

---

## 7. 出力仕様（Output）

### 7.1 ファイル構成
1つのシミュレーション実行は以下の3種のHDF5ファイルを生成する：

| ファイル | 命名規則 | 出力頻度 | 用途 |
|---------|---------|---------|------|
| スナップショット | `<case>_NNNNNN.h5` | `plot_every` ステップ毎 または `plot_every_s` 秒毎（OR論理） | 場の可視化・後処理 |
| 時系列 | `<case>_history.h5` | `history_every` ステップ毎 または `history_every_s` 秒毎（OR論理、追記） | スカラー診断量の時間変化 |
| チェックポイント | `<case>_ckpt_NNNNNN_rNNNN.h5` | `checkpoint_every` ステップ毎 または `checkpoint_every_s` 秒毎（OR論理） | リスタート（ランク別ファイル） |

- `NNNNNN`：6桁ゼロ詰めのサイクル番号。時間間隔ベース出力で生成されたファイルもサイクル番号で命名する（物理時刻はルート属性 `t` に格納）
- **並列IO方式**：スナップショットは並列HDF5（MPI-IO）で単一ファイルに書き込み。時系列（history）は rank 0 のみが書き込み。チェックポイントはランク別ファイル（`<case>_ckpt_NNNNNN_rNNNN.h5`）

**スナップショット IO 仕様**：
- H5FD_MPIO ドライバ（MPI-IO バックエンド）
- collective write（`H5Pset_dxpl_mpio(H5FD_MPIO_COLLECTIVE)`）
- チャンクサイズ: `[N_cells_per_rank, ...]`（1D）or `[N_cells_r_per_rank, N_cells_z_per_rank, ...]`（2D）
- ファイルシステム固有のヒントはハードコードしない
- ユーザーは環境変数（`ROMIO_HINTS` 等）で制御可能
- `<case>`：`Main.name` を優先。未指定（`name` 省略）の場合は namelist ファイル名（拡張子なし）をフォールバックとして使用。全ファイルは `Output.directory` 下に格納
- チェックポイントは `Output.checkpoint_keep_last` 個を保持し、古いものを自動削除

### 7.2 スナップショットHDF5レイアウト（`<case>_NNNNNN.h5`）

```
/                                   [root group]
│  attrs: t (float64, s)
│  attrs: cycle (int64)
│  attrs: geometry ("1D_SPH" | "2D_RZ")
│  attrs: n_cells (int64)
│  attrs: n_nodes (int64)
│  attrs: n_groups (int64)
│  attrs: n_materials (int64)
│
├── metadata/
│   ├── namelist_source             string      namelist Pythonソース全文
│   ├── frozen_config               string      JSON形式の凍結設定
│   │  attrs: git_hash (string)
│   │  attrs: build_type (string)
│   │  attrs: cuda_arch (string)
│   │  attrs: gpu_name (string)
│   │  attrs: n_ranks (int32)
│   │  attrs: rng_seed (uint64)
│   └── group_bounds_eV             float64[G+1]  放射群境界
│
├── mesh/                           NodeField（節点中心量。ARCHITECTURE §5.2 State 準拠）
│   ├── x_r                         float64[N_r+1] or [N_r+1, N_z+1]   節点R座標 [cm]（State.x_r）
│   ├── x_z                         float64[N_r+1, N_z+1]              節点Z座標 [cm]（2D_RZのみ。State.x_z）
│   ├── v_r                         float64[N_node]                     節点R速度 [cm/s]（State.v_r）
│   ├── v_z                         float64[N_node]                     節点Z速度 [cm/s]（2D_RZのみ。State.v_z）
│   ├── cell_material_id            int32[N_cell]                       材料インデックス（0始まり）。多材料セル：体積分率最大の材料インデックスを格納。同率の場合は小さいインデックスを優先
│   ├── topology/v2/                optional。`MULTIBLOCK_CART_CORE_POLAR_SHELL`（3-block）または `CONE_SHELL_SPINE`（Stage C2 は 4-block）の multiblock topology。single_block ではグループ全体を省略
│   │   ├── block_count             int32 scalar（scheme の実 block count）
│   │   ├── cell_block_id           int32[N_cell]
│   │   ├── cell_id_stable          int32[N_cell]
│   │   ├── cell_node_csr_offsets   int32[N_cell+1]
│   │   ├── cell_node_csr_indices   int32[4*N_cell]
│   │   ├── face_adj_csr_offsets    int32[N_cell+1]
│   │   ├── face_adj_csr_indices    int32[4*N_cell]（外部境界は -1）
│   │   └── face_bc_tags            int32[4*N_cell]
│   ├── topology/v3/                optional。`MULTIBLOCK_HALF_BUTTERFLY_5BLOCK` 用の variable-block multiblock topology。reader は v3 を優先し、無ければ v2、さらに無ければ v1 single_block として扱う。HDF5 root `schema_version` は 1 のまま path-versioning で管理
│       ├── block_count             int32 scalar（現在の half-butterfly では 5）
│       ├── block_id                int32[block_count]（dense 0-based）
│       ├── block_role              int32[block_count]（central, north fan, east fan, south fan, polar shell）
│       ├── block_n_i_cells         int32[block_count]
│       ├── block_n_j_cells         int32[block_count]
│       ├── block_cell_begin        int32[block_count]
│       ├── block_cell_count        int32[block_count]
│       ├── block_owned_node_begin  int32[block_count]
│       ├── block_owned_node_count  int32[block_count]
│       ├── seam_block_a            int32[N_seam]
│       ├── seam_side_a             int32[N_seam]
│       ├── seam_block_b            int32[N_seam]
│       ├── seam_side_b             int32[N_seam]
│       ├── seam_orientation        int32[N_seam]（+1/-1）
│       ├── seam_index_begin        int32[N_seam]
│       ├── seam_index_count        int32[N_seam]
│       ├── cell_block_id           int32[N_cell]
│       ├── cell_id_stable          int32[N_cell]
│       ├── cell_orientation_sign   int32[N_cell]（central +1; fans and shell -1）
│       ├── cell_node_csr_offsets   int32[N_cell+1]
│       ├── cell_node_csr_indices   int32[4*N_cell]
│       ├── face_adj_csr_offsets    int32[N_cell+1]
│       ├── face_adj_csr_indices    int32[4*N_cell]（外部境界は -1）
│       └── face_bc_tags            int32[4*N_cell]
│   └── topology/v4/                optional。`PENTAGON_BELT_SHELL` の variable-vertex corner storage metadata
│       attrs: corner_stride (int32、現在は 8)
│       └── cell_nverts             int8[N_cell]（セルごとの有効頂点数）
│
├── hydro/                          CellField（セル中心量。ARCHITECTURE §5.2 State 準拠）
│   ├── rho                         float64[N_cell]       質量密度 [g/cm³]（State.rho）
│   ├── Te                          float64[N_cell]       電子温度 [eV]（State.Te）
│   ├── Ti                          float64[N_cell]       イオン温度 [eV]（State.Ti）
│   ├── ee                          float64[N_cell]       電子比内部エネルギー [erg/g]（State.ee）
│   ├── ei                          float64[N_cell]       イオン比内部エネルギー [erg/g]（State.ei）
│   ├── Pe                          float64[N_cell]       電子圧力 [dyne/cm²]（State.Pe）
│   ├── Pi                          float64[N_cell]       イオン圧力 [dyne/cm²]（State.Pi）
│   ├── Qvisc                       float64[N_cell]       人工粘性圧 [dyne/cm²]（State.Qvisc。NUMERICS §3.1.6, §3.2.9）
│   ├── shock_time                  float64[N_cell]       optional。最近 shock を検知した時刻 [s]（State.shock_time。`post_shock_heat=True`、`post_shock_velocity_damping_C>0`、または adaptive AV の `Cpsv>0` 時。NUMERICS §3.1.6）
│   ├── adaptive_av_gate            float64[N_cell]       optional。adaptive AV gate 履歴 \(g_i\) [-]（State.adaptive_av_gate。`adaptive_av.enabled=True` 時。NUMERICS §3.1.6）
│   ├── adaptive_av_mode_cell       int8[N_cell]          optional。adaptive AV gate が有効なセル（\(g_i>10^{-6}\)）を 1、それ以外を 0 とする可視化用 mask（`adaptive_av.enabled=True` 時）
│   ├── mass                        float64[N_cell]       セル質量 [g]（State.mass = ρ×V）
│   ├── corner_mass                 float64[N_cell, corner_stride] optional。2D_RZの Lagrangian-invariant subzonal corner mass [g]（State.corner_mass。legacy および topology/v4 不在時の corner_stride は 4、topology/v4 がある場合は同 group の attr 値。旧checkpointで不在の場合は restart 後の初回 hydro step で再計算）
│   ├── vol                         float64[N_cell]       セル体積 [cm³]（State.vol）
│   ├── zbar                        float64[N_cell]       平均電離度 Z̄ [-]（State.zbar。NUMERICS §1.1.4）
│   ├── eta_compatible              float64[N_cell]       optional。legacy volume-form compatible 診断の体積差不整合 \(\eta=(V^{n+1}-V^n)-\Delta V_{comp}\) [cm³]（State.eta_compatible；exact force-work path の保存機構には未使用）
│   ├── volFrac                     float64[N_cell, N_mat] 体積分率（多材料時。State.volFrac）
│   └── per_material/v1/            optional。`Numerics.materials.per_material_conservation_enabled=True` の時だけ作成。Wave A は dataset なしの schema skeleton で、disabled 時は group 全体を省略する
│       attrs: enabled (uint8 bool), n_mat (int32), eos_method ("table"|"ideal_gas"), schema_version=1,
│              conserved_basis="extensive", layout="cell_major_ncells_nmat",
│              material_names (string[N_mat]), material_ids (int32[N_mat])
│
├── diagnostics/
│   ├── areal_density/v1/           optional; `Diagnostics.areal_density.enabled=True`
│   │   ├── angles_deg              float64[N_angle]      ray angle list [deg]
│   │   ├── rhoR                    float64[N_angle]      existing total/shell rhoR according to `r_range` [g/cm²]
│   │   ├── rhoR_hotspot_tracer     float64[N_angle] optional; \(\int\rho Y_g ds\) [g/cm²]
│   │   └── rhoR_fuel_tracer        float64[N_angle] optional; reserved for future distinct fuel tracer/material mask [g/cm²]
│   └── hotspot_gas/v1/             optional; `Numerics.diagnostics.hotspot_gas.enabled=True`
│       ├── hotspot_Te_mean_eV, hotspot_Te_p10_eV, hotspot_Te_p50_eV, hotspot_Te_p90_eV float64 scalar
│       ├── hotspot_Ti_valid        int32 scalar; 1 for 2T, 0 for 1T
│       ├── hotspot_Ti_mean_eV, hotspot_Ti_p10_eV, hotspot_Ti_p50_eV, hotspot_Ti_p90_eV float64 scalar optional; 2T only
│       ├── hotspot_energy_internal_erg, hotspot_energy_kinetic_erg, hotspot_energy_total_erg float64 scalar
│       ├── hotspot_energy_internal_initial_erg, hotspot_energy_kinetic_initial_erg, hotspot_energy_total_initial_erg float64 scalar
│       ├── hotspot_work_proxy_internal_erg, hotspot_work_proxy_kinetic_erg, hotspot_work_proxy_total_erg float64 scalar
│       └── hotspot_work_definition string
│
├── radiation/
│   ├── energy_density              float64[N_cell, G]    E_g [erg/cm³]（legacy particle cell では track-length estimator、difference cell では \(\bar{E}^{ref}\)+signed residual estimator、hybrid diffusion cell では deterministic \(E^D_{i,g}\)）
│   ├── rad_dep                     float64[N_cell, G]    放射-物質交換エネルギー [erg]（当該ステップ累積；IMC/DDMC の gross absorption tally。PGRW は IMC kernel 内で同じ tally に加算する。NUMERICS §10.2参照）
│   ├── rad_emit                    float64[N_cell, G]    particle emission diagnostic [erg]（当該ステップ累積）
│   ├── deposited_power             float64[N_cell, G]    吸収パワー密度 [erg/cm³/s]（= rad_dep / (V × Δt)；可視化・解析用）
│   ├── fleck_factor                float64[N_cell]       2D_RZ FLD only。FLD material coupling で使った per-cell Fleck factor \(f_i\) [dimensionless]。現行の診断意味は gray \(G=1\) で有効；multigroup FLD の per-group Fleck index 修正は別作業。
│   ├── sn_tau_R                    float64[N_cell]       2D_RZ \(S_N\) only。AP face_blend 診断用 cell-local Rosseland optical-depth proxy（群・セル幅の最小）[dimensionless]
│   ├── sn_reduced_flux             float64[N_cell]       2D_RZ \(S_N\) only。AP face_blend 診断用 reduced flux proxy（内部隣接面・群の最大 \(|F|/(cE)\)）[dimensionless]
│   ├── sn_ap_alpha                 float64[N_cell]       2D_RZ \(S_N\) only。AP face_blend weight \(\alpha\) のセル隣接面・群最大 [dimensionless]
│   ├── diag_rad_E_pre              float64[N_cell, G]    1D S_N diagnostic: radiation-step entry E_g [erg/cm³]
│   ├── diag_rad_E_post             float64[N_cell, G]    1D S_N diagnostic: post material Newton/Phase B E_g [erg/cm³]
│   ├── diag_rad_emission_at_Tn     float64[N_cell, G]    1D S_N diagnostic: Δt c σ_PE(T_old) B(T_old) sweep RHS term [erg/cm³]
│   ├── diag_rad_emission_at_Tnp1   float64[N_cell, G]    1D S_N diagnostic: Δt c σ_PE(T_old) B(T_new) Newton observer term [erg/cm³]
│   ├── diag_rad_absorption         float64[N_cell, G]    1D S_N diagnostic: Δt c σ_PA(T_old) E_sweep before Phase B [erg/cm³]
│   ├── diag_clip_energy            float64[N_cell, G]    1D S_N diagnostic: Phase B negative pre-source clip contribution [erg/cm³]
│   ├── diag_clip_full_deficit      float64[N_cell, G]    1D S_N diagnostic: full Phase B negative pre-source clip deficit [erg/cm³]
│   ├── diag_chi_opacity            float64[N_cell, G]    1D S_N diagnostic: post-Newton opacity lag max relative change [dimensionless]
│   ├── diag_F_first_moment         float64[N_cell, G]    1D S_N diagnostic: angular first moment Σ_n w_n μ_n ψ_n [erg/cm²/s]
│   ├── diag_E_star_flux            float64[N_cell, G]    1D S_N diagnostic: face-flux \(E^*\) override [erg/cm³]
│   ├── diag_stream_theta           float64[N_cell, G]    1D S_N diagnostic: donor-theta limiter weight [dimensionless]
│   ├── diag_ap_alpha_face          float64[N_face, G]    1D S_N diagnostic: AP face_blend weight [dimensionless]
│   ├── ddmc_flag                   int8[N_cell, G]       0=IMC, 1=DDMC, 2=RW, 3=Diffusion（現行 PGRW 実装は 2 を生成しない）
│   ├── boundary_flux               float64[G]            群別境界流出 [erg/s]
│   └── momentum_dep               float64[N_cell, 2]    運動量沈着 [dyne·s/cm³] (R,Z成分；診断のみ、hydro非結合)。1D_SPH: [N_cell, 1]（径方向のみ）
│
├── holo/                            （Radiation.holo.enabled=True かつ selector state 有効時のみ）
│   ├── E_LO                        float64[N_cell, G]    global low-order 物理フレーム放射エネルギー密度 \(E^{LO}_{i,g}\) [erg/cm³]。step 間で persistent、初期/resize 時は 0
│   ├── consistency_source          float64[N_cell, G]    same-step LO corrector RHS source \(R^{cons}_{i,g}\) [erg/s]
│   ├── rad_dep_LO                  float64[N_cell, G]    LO gross absorption diagnostic [erg]
│   ├── rad_emit_LO                 float64[N_cell, G]    LO gross emission diagnostic [erg]
│   ├── Prr_HO                      float64[N_cell, G]    passive high-order radial pressure moment \(P^{HO}_{rr,i,g}\) [erg/cm³]
│   ├── chi                         float64[N_cell, G]    \(P^{HO}_{rr,i,g}/\max(E^{HO}_{i,g},E_{floor})\) [dimensionless]
│   ├── Prr_coverage                float64[N_cell, G]    \(P_{rr}\) covered track-length fraction [dimensionless]
│   ├── core_mask                   uint8[N_cell]         現ステップ LO material-coupling mask（legacy dataset名）
│   ├── prev_core_mask              uint8[N_cell]         selector 更新前の LO material-coupling mask（legacy dataset名）
│   ├── hold_count                  int32[N_cell]         entry hysteresis hold counter
│   ├── dwell_count                 int32[N_cell]         exit dwell counter
│   ├── tau_R                       float64[N_cell]       selector Rosseland optical-depth proxy [dimensionless]
│   ├── reduced_flux                float64[N_cell]       selector reduced-flux proxy [dimensionless]
│   └── mass_q                      float64[N_cell]       shell mass coordinate \(q_i\) [dimensionless]
│
├── difference/                      （Radiation.imc.difference.enabled=True時のみ）
│   ├── W                           float64[N_cell]       reference weight \(W_i\) [dimensionless]
│   ├── E_ref                       float64[N_cell, G]    `rad_E` reconstruction に使った time-average reference density \(\bar{E}^{ref}_{i,g}\) [erg/cm³]
│   └── residual_energy_density     float64[N_cell, G]    signed residual estimator \(\mathrm{rad\_E\_tally}_{i,g}/(V_i c\Delta t)\) [erg/cm³]
│
└── laser/                          （laser.enabled=True時のみ）
    ├── deposited_power             float64[N_cell]       レーザー沈着パワー密度 [erg/cm³/s]
    ├── A_Q_r_shells                float64[N_r]          2D_RZ laser only。各 \(r_i\) shell の raw `laser_dep` [erg] に対する \((q_{\max}-q_{\min})/(q_{\max}+q_{\min})\) [dimensionless]。空 shell は 0。
    ├── A_Q_z_shells                float64[N_z]          2D_RZ laser only。各 \(z_j\) shell の raw `laser_dep` [erg] に対する \((q_{\max}-q_{\min})/(q_{\max}+q_{\min})\) [dimensionless]。空 shell は 0。
    ├── absorption_fraction         float64               全体吸収率（absorbed/incident）
    ├── ray_density                 float64[N_cell]       レイ初期配置密度（セルに配置されたレイ数 / セル体積）[1/cm³]
    ├── mesh/                       （ray_output_trajectory=True かつ非skipステップ時）
    │   ├── n_nodes_r               int32                 メッシュR方向ノード数
    │   ├── n_nodes_z               int32                 メッシュZ方向ノード数
    │   ├── n_crit                  float64               臨界電子密度 [1/cm³]
    │   ├── node_R                  float64[n_nodes_r]    Rノード座標 [cm]
    │   ├── node_Z                  float64[n_nodes_z]    Zノード座標 [cm]
    │   ├── n_e_hat                 float64[n_nodes_r*n_nodes_z]  正規化電子密度
    │   ├── T_e                     float64[n_nodes_r*n_nodes_z]  電子温度 [eV]
    │   ├── Zbar                    float64[n_nodes_r*n_nodes_z]  平均電離度
    │   ├── grad_n_hat_R            float64[n_nodes_r*n_nodes_z]  密度勾配R成分 [1/cm]
    │   └── grad_n_hat_Z            float64[n_nodes_r*n_nodes_z]  密度勾配Z成分 [1/cm]
    └── rays/                       （ray_output_count>0 かつ非skipステップ時）
        ├── n_rays                  int32                 実際に出力されたレイ数
        ├── beam_id                 int32[n_rays]         レイのビームID（2D_RZは代表beam_groupのID）
        ├── R0 / Z0 / vR0 / vZ0     float64[n_rays]       1D_SPH（raytrace_2d）初期位置・方向
        ├── x0 / y0 / z0            float64[n_rays]       2D_RZ（raytrace_3d）初期位置
        ├── vx0 / vy0 / vz0         float64[n_rays]       2D_RZ（raytrace_3d）初期方向
        ├── power0                  float64[n_rays]       初期レイパワー [erg/s]
        └── trajectory/             （ray_output_trajectory=True かつ非skipステップ時）
            ├── n_rays              int32                 出力レイ本数
            ├── offsets             int64[n_rays+1]       CSRオフセット
            ├── step_count          int32[n_rays]         レイ毎のステップ数
            ├── beam_id             int32[n_rays]         ビームID
            ├── pos_R / pos_x       float64[total_steps]  位置 [cm]
            ├── pos_Z / pos_y       float64[total_steps]  位置 [cm]
            ├── pos_z               float64[total_steps]  位置 [cm]（3Dのみ）
            └── power               float64[total_steps]  残存パワー [erg/s]
```

`laser.mode="radial_absorption_1d"` では `laser/deposited_power` と `laser/absorption_fraction` は通常どおり出力するが、
物理的なレイ軌跡がないため `laser/rays/` と `laser/rays/trajectory/` は空または未作成とする。

**データセット命名規約**：`mesh/` と `hydro/` のデータセット名は `ARCHITECTURE.md §5.2` の `State` 構造体フィールド名と **1:1対応** する。これによりI/Oコードで名前マッピングが不要となる。全数値データセットは `units` 属性（string）を持つ（例：`"g/cm3"`, `"eV"`, `"dyne/cm2"`）。`radiation/` と `laser/` は `CellFieldG` / raw配列に対応し、独自の記述的命名を使用する。

**1D_SPH時の簡略化**：
- `mesh/x_z`, `mesh/v_z` は省略
- 配列形状は1Dフラット（例：`x_r` は `float64[N+1]`）

### 7.3 時系列HDF5レイアウト（`<case>_history.h5`）

時系列ファイルは **extensible dataset**（HDF5 chunked + unlimited dimension）を使用し、各ステップで末尾に追記する。

```
/
│  attrs: termination_reason        string                停止理由（"t_end" | "max_steps"）。シミュレーション完了時に書き込み（§6.1 停止条件参照）
├── t                               float64[N_step]       シミュレーション時間 [s]
├── cycle                           int64[N_step]         サイクル番号
├── dt                              float64[N_step]       タイムステップ幅 [s]
│
├── energy/
│   ├── laser_incident              float64[N_step]       レーザー入射 E_laser_in [erg]（NUMERICS §10.2）
│   ├── laser_deposited             float64[N_step]       レーザー吸収累積 E_laser_deposited [erg]
│   ├── laser_escaped               float64[N_step]       レーザー未吸収累積 E_laser_escaped [erg]
│   ├── radiation_escaped           float64[N_step]       放射境界流出累積 E_rad_escaped [erg]（E_escape[G]の全群合算）
│   ├── marshak_in                  float64[N_step]       Marshak境界入射累積 E_Marshak_in [erg]（§8.2）
│   ├── pdv_boundary                float64[N_step]       境界PdV仕事累積 E_pdV_bdry [erg]（§10.2）
│   ├── kinetic                     float64[N_step]       運動エネルギー合計 E_kin [erg]
│   ├── internal_electron           float64[N_step]       電子内部エネルギー合計 E_int_e [erg]
│   ├── internal_ion                float64[N_step]       イオン内部エネルギー合計 E_int_i [erg]
│   ├── radiation_field             float64[N_step]       放射場エネルギー E_rad [erg]（通常は census粒子エネルギー合計。difference path では deterministic reference reservoir + signed residual census）
│   ├── numerical_loss              float64[N_step]       数値的喪失エネルギー累積 E_numerical_loss [erg]（粒子移送失敗等）
│   ├── floor_injected              float64[N_step]       フロア補正注入累積 E_floor_injected [erg]（診断専用、保存誤差の分子に含めない）
│   ├── safety_injected             float64[N_step]       安全補正注入累積 E_safety [erg]（診断専用、保存誤差の分子に含めない）
│   ├── redistribution_unresolved   float64[N_step]       ALE positivity redistribution 未解決分 E_redistribution_unresolved [erg]（診断専用、通常0）
│   ├── solver_residual             float64[N_step]       Hypre残差エネルギー累積 E_solver [erg]（v1.0=0、診断専用）
│   ├── conservation_error          float64[N_step]       相対保存誤差 ε_budget（NUMERICS §10.2 恒等式に基づく）
│   ├── E_volume_in                 float64[N_step]       境界体積仕事入力累積（内部診断）[erg]
│   ├── E_total                     float64[N_step]       全エネルギー（内部診断）[erg]
│   ├── dE_total                    float64[N_step]       全エネルギー差分（内部診断）[erg]
│   ├── E_denom                     float64[N_step]       保存誤差分母（内部診断）[erg]
│   ├── phase_diagnostic/           optional; `Numerics.diagnostics.phase_resolved_energy=True` の場合のみ作成
│       ├── E_pre_hydro             float64[N_step]       hydro 前の物質全エネルギー [erg]
│       ├── E_post_hydro            float64[N_step]       hydro 後/ALE 前の物質全エネルギー [erg]
│       ├── E_post_ale              float64[N_step]       ALE 後の物質全エネルギー [erg]
│       ├── dE_hydro                float64[N_step]       E_post_hydro - E_pre_hydro [erg]
│       ├── dE_ale                  float64[N_step]       E_post_ale - E_post_hydro [erg]
│       ├── internal_*              float64[N_step]       pre_hydro/post_hydro/post_ale の内部エネルギー [erg]
│       ├── kinetic_*               float64[N_step]       pre_hydro/post_hydro/post_ale の運動エネルギー [erg]
│       ├── dE_hydro_internal       float64[N_step]       hydro 相の内部エネルギー差分 [erg]
│       ├── dE_ale_internal         float64[N_step]       ALE 相の内部エネルギー差分 [erg]
│       ├── dE_hydro_kinetic        float64[N_step]       hydro 相の運動エネルギー差分 [erg]
│       └── dE_ale_kinetic          float64[N_step]       ALE 相の運動エネルギー差分 [erg]
│   └── ale_closure_audit/          optional; `Numerics.ale.ke_conservation_closure_audit=True` の ALE 呼び出しごとに作成
│       ├── cycle, t                int64/float64[N_ale]  audit 対象サイクルと時刻 [s]
│       ├── K0_cellcorner, K0_node_from_corner, K0_budget, I0, K0_scalar_total
│       ├── K_remap_total, I_raw, K_cellmom, K_node_preBC, K_node_postBC, K_post_budget
│       ├── sum_dI_raw, sum_dI_after_floor, F_floor
│       ├── residual_*              float64[N_ale]        K0/Remap/closure target/ΔE 分解 residual [erg]
│       └── n_cells_*, min_tentative_*, mechanism_code, flag_*  floor/分類 counters
│
├── mesh/
│   └── ale_rezone_invocations      int64[N_step]         累積ALE rezone発動回数 [count]。旧historyで不在の場合は unknown と扱う
│
├── plasma/
│   ├── Zbar_mean                   float64[N_step]       質量重み平均電離度 Z̄_mean [dimensionless]。算出：\(\bar{Z}_{mean} = \sum M_c \bar{Z}_c / \sum M_c\)
│   └── Zbar_max                    float64[N_step]       最大電離度 Z̄_max [dimensionless]
│
├── implosion/
│   ├── rho_peak                    float64[N_step]       ピーク密度 [g/cm³]
│   ├── rho_R                       float64[N_step, N_angle] 面密度ρR [g/cm²]
│   │  attrs: angles_deg (float64[N_angle])
│   ├── rho_R_hotspot_tracer        float64[N_step, N_angle] optional; \(\int\rho Y_g ds\) hotspot gas tracer 面密度 [g/cm²]
│   │  attrs: angles_deg (float64[N_angle])
│   ├── shell_radius_mean           float64[N_step]       シェル平均半径 [cm]。算出：\(\rho_c > 0.1 \times \rho_{max}\) のセルを質量重み平均 \(R_{mean} = \sum M_c r_c / \sum M_c\)
│
├── diagnostics/
│   ├── icf/v1/                     optional; `Numerics.diagnostics.icf.enabled=True` または ICF ALE profile 有効時のみ作成。HDF5 root `schema_version` は 1 のまま path-versioning で管理
│   │   ├── time_s, step             float64/int32[N]      history cadence の時刻・ステップ
│   │   ├── shell_radius_cm          float64[N]            \(R_{shell,mean}\) [cm]
│   │   ├── shell_thickness_cm       float64[N]            \(R_{outer}-R_{inner}\) [cm]
│   │   ├── IFAR, CR                 float64[N]            in-flight aspect ratio / convergence ratio
│   │   └── attrs: R_initial_cm      float64               最初の有効ICF診断サンプルの半径 [cm]
│   ├── hotspot_gas/v1/             optional; `Numerics.diagnostics.hotspot_gas.enabled=True` の場合のみ作成
│   │   ├── time_s, step             float64/int32[N]      history cadence の時刻・ステップ
│   │   ├── gas_mass_initial_g, gas_mass_g, gas_mass_rel_drift float64[N]
│   │   ├── R50_cm, R90_cm, R95_cm, R99_cm, CR50, CR90, CR95, CR99, C_R50_norm, Rrms_cm, CRrms, CR_V, CR_rho50 float64[N]
│   │   ├── rho_*_gcc, p*_dyn_cm2, K* float64[N]           tracer-gas density/pressure/adiabat summaries
│   │   ├── hotspot_Te_mean_eV, hotspot_Te_p10_eV, hotspot_Te_p50_eV, hotspot_Te_p90_eV float64[N]
│   │   ├── hotspot_Ti_valid         int32[N]              1 for 2T runs, 0 for 1T runs
│   │   ├── hotspot_Ti_mean_eV, hotspot_Ti_p10_eV, hotspot_Ti_p50_eV, hotspot_Ti_p90_eV float64[N] optional; 2T runs only
│   │   ├── hotspot_energy_internal_erg, hotspot_energy_kinetic_erg, hotspot_energy_total_erg float64[N]
│   │   ├── hotspot_energy_internal_initial_erg, hotspot_energy_kinetic_initial_erg, hotspot_energy_total_initial_erg float64[N]
│   │   ├── hotspot_work_proxy_internal_erg, hotspot_work_proxy_kinetic_erg, hotspot_work_proxy_total_erg float64[N]
│   │   └── attrs: R_g_cm, hotspot_work_definition
│   ├── ale_provenance/v1/           optional; `Numerics.diagnostics.ale_provenance_emission.enabled=True` または ICF ALE profile 有効時のみ作成
│   │   ├── time_s, step             float64/int32[N]      history cadence + run-end final sample
│   │   ├── forbidden_config_violations, escape_valve_activations, class_c_runtime_fires, mesh_geometry_failures_observed, public_baseline_terminal_failures int32[N]
│   │   ├── emergency_cell_deactivation_fired, last_failure_kind uint8[N]
│   │   ├── last_failing_cell, last_failing_i, last_failing_j int32[N]
│   │   ├── last_min_cell_vol, last_min_corner_j float64[N]
│   │   ├── escape_valve_events/split_phase, operator_inserted string[N_event]
│   │   ├── escape_valve_events/order_degraded uint8[N_event]
│   │   ├── escape_valve_events/E_thermal_before, E_thermal_after, E_kinetic_before, E_kinetic_after float64[N_event] [erg]
│   │   ├── escape_valve_events/mass_transfer, momentum_transfer_R, momentum_transfer_Z, internal_energy_transfer float64[N_event] (currently zero unless rescue source-term transfer data is exposed)
│   │   ├── escape_valve_events/step, time int32/float64[N_event]
│   │   └── attrs: final_provenance, final_claim_level, reached_t_end, profile_enabled
│   ├── mesh_quality_min/v1/         optional; `Numerics.diagnostics.mesh_quality_min.enabled=True` の場合のみ作成。診断専用の additive/backward-compatible group
│   │   ├── time_s, step             float64/int32[N]      history cadence + run-end final sample
│   │   ├── achieved_min_corner_j_rel, achieved_min_gauss_j_rel, achieved_min_rz_volume_rel float64[N] all accepted committed post-ALE meshes over the run, normalized by each cell's initial geometry
│   │   ├── negative_rz_volume_count_total int64[N]        accepted committed post-ALE meshesで検出した non-finite または <=0 RZ volume セル数の累積
│   │   ├── achieved_min_edge_length_rel, achieved_min_altitude_rel float64[N] all accepted committed post-ALE meshes over the run, normalized by each cell's initial edge-length/altitude
│   │   └── achieved_max_condition_number float64[N]       max accepted committed post-ALE 2x2 Gauss-point cell-Jacobian singular-value ratio
│   ├── ale_lambda_sweep/v1/        optional; `Numerics.ale.lambda_sweep_diagnostic_enabled=True` かつ target cell が最初の Gauss/corner-J backtrack rejection と一致した場合のみ作成
│   │   ├── lambda                  float64[N_lambda]      sampled backtrack fraction [dimensionless]
│   │   ├── min_gauss_J             float64[N_lambda]      target+1-ring active cells の最小 Gauss-J [cm²]
│   │   ├── min_corner_J            float64[N_lambda]      target+1-ring active cells の最小 signed corner-J [cm²]
│   │   ├── min_V_RZ                float64[N_lambda]      target+1-ring active cells の最小 signed RZ volume [cm³]
│   │   ├── admissible              uint8[N_lambda]        all three minima are finite and positive
│   │   └── attrs: schema_version=1, target_cell_c, target_cell_i, target_cell_j, classification
│   ├── radial_fourier_audit/v1/    optional; `Diagnostics.per_operator_radial_fourier_enabled=True` かつ configured time window 内の per-stage sample
│   │   ├── cycle                   uint64[N]              audit 対象サイクル
│   │   ├── t_s                     float64[N]             audit 時刻 [s]
│   │   ├── stage_id                uint8[N]               0=hydro_lag, 1=AV, 2=winslow, 3=remap, 4=bc_fill, 5=FLD_solve, 6=Newton_src, 7=positivity, 8=dt_ctrl
│   │   ├── stage_phase             uint8[N]               0=before, 1=after
│   │   ├── field_id                uint8[N]               0=rho, 1=Te, 2=Ti, 3=u_r, 4=u_z, 5=E_rad
│   │   ├── A_max                   float64[N]             max over radial mode and z-index of normalized Fourier amplitude [dimensionless]
│   │   ├── m_max                   uint16[N]              radial mode index at A_max
│   │   └── j_max                   uint16[N]              z-index at A_max
│   ├── radial_fourier_audit_v2/v1/ optional; `Diagnostics.per_operator_radial_fourier_complex_enabled=True` かつ configured time window 内の fixed `(m,j)` complex coefficient sample
│   │   ├── cycle                   uint64[N]              audit 対象サイクル
│   │   ├── t_s                     float64[N]             audit 時刻 [s]
│   │   ├── dt_cycle                float64[N]             cycle time step used for stage gain normalization [s]
│   │   ├── stage_id                uint8[N]               v1 と同じ stage enum
│   │   ├── stage_phase             uint8[N]               0=before, 1=after
│   │   ├── field_id                uint8[N]               0=rho, 1=M, 2=V, 3=M_over_V, 4=P_r, 5=P_z, 6=u_r, 7=u_z, 8=E_e, 9=E_i, 10=E_rad, 11=T_e, 12=T_i, 13=x_r, 14=x_z, 15=A_r, 16=A_z, 18=Q_visc, 22=f_Fleck
│   │   ├── m, j                    uint16[N]              fixed radial mode and z-index
│   │   ├── mean_unw, mean_vol      float64[N]             unweighted and volume-weighted radial means
│   │   ├── cre_unw, cim_unw        float64[N]             unweighted complex coefficient real/imag
│   │   ├── amp_unw, phase_unw      float64[N]             unweighted coefficient amplitude/phase [phase rad]
│   │   ├── cre_vol, cim_vol        float64[N]             volume-weighted complex coefficient real/imag
│   │   ├── amp_vol, phase_vol      float64[N]             volume-weighted coefficient amplitude/phase [phase rad]
│   │   ├── q_min_j, q_max_j        float64[N]             radial min/max of field q at z-index j
│   │   └── wsum_vol                float64[N]             Σ_i V_{ij} [cm3]
│   ├── dt_breakdown_history/       optional; `Numerics.diagnostics.dt_breakdown_history_enabled=True` の場合に毎ステップ作成
│   │   ├── cycle                   int64[N]               `[dt_breakdown] step` と同じ pre-step cycle
│   │   ├── t_s                     float64[N]             pre-step 時刻 [s]
│   │   ├── dt_chosen, dt_hydro, dt_rad, dt_cond, dt_post_shock, dt_growth, dt_max, dt_output, dt_remaining float64[N] [s]
│   │   ├── dt_hydro_acoustic, dt_hydro_axis_margin, dt_hydro_volume_rate float64[N] [s]
│   │   └── dt_winner, dt_winner_code string/int32[N]      `hydro/rad/growth/max/cond/post_shock/volume_rate/output/t_end/init/driver_retry/other`
│   ├── cfl_winner/                 optional; dt_breakdown history と同じ行数
│   │   ├── cycle, t_s              int64/float64[N]       pre-step cycle/time
│   │   ├── cell_id, i, j           int32[N]               2D hydro acoustic CFL winner cell（非2Dまたは非hydroは -1）
│   │   └── dt_at_cell, dl_at_cell, cs_at_cell, rho_at_cell, u_z_at_cell float64[N]
│   ├── corner_bc_audit/v1/         optional; cfl_winner が r_outer-reflect ∩ z_top-state_supply corner halo 内にある step のみ作成
│   │   ├── cycle, t_s, cell_id, i, j int64/float64/int32[N] audited CFL winner
│   │   ├── interior_rho, interior_u_r, interior_u_z, interior_e, interior_E_r float64[N]
│   │   ├── r_outer_ghost_rho, r_outer_ghost_u_r, r_outer_ghost_u_z float64[N]
│   │   ├── z_top_ghost_rho, z_top_ghost_u_r, z_top_ghost_u_z float64[N]
│   │   ├── diagonal_corner_ghost_rho, diagonal_corner_ghost_u_r, diagonal_corner_ghost_u_z float64[N]
│   │   └── dt_at_cell, cs_at_cell, q_visc_at_cell float64[N]
│   ├── fld_substage_audit/v1/      optional; `Radiation.multigroup_diffusion.diagnostic_radial_fourier_substage_enabled=True` の場合のみ作成
│   │   ├── cycle, t_s, dt_cycle    uint64/float64/float64[N]
│   │   ├── substage_id, field_id, normalization_kind uint8[N]
│   │   ├── m, j                    uint16[N]             audited radial mode / axial row
│   │   ├── group, outer_iter, nr, nz int32[N]
│   │   └── Re, Im, amplitude, phase, mean, min, max, normalization, solver_residual_l2_rel, solver_residual_max float64[N]
│   ├── per_row_mass/               optional; dt_breakdown history 有効時、最初の committed step と以後50 committed steps ごとに作成
│   │   ├── cycle, t_s              int64/float64[N_row]   post-step cycle/time
│   │   ├── z_row_idx               int32[N_z]             z-row index
│   │   └── mass_per_z_row, rho_max_per_z_row, rho_at_axis_per_z_row, rho_at_outer_per_z_row, cs_max_per_z_row float64[N_row, N_z]
│   ├── av_max/                     optional; dt_breakdown history 有効時、2D hydro step ごとに作成
│   │   ├── cycle, t_s              int64/float64[N]       post-step cycle/time
│   │   ├── cell_id, i, j           int32[N]               maximum Qvisc cell
│   │   └── q_visc_max, rho_at_max, cs_at_max, delta_u_at_max float64[N]
│   └── conservation/v1/             optional; `Numerics.diagnostics.conservation.enabled=True` または ICF ALE profile 有効時のみ作成
│       ├── time_s, step             float64/int32[N]      history cadence
│       └── eps_E_<op>, E_before_<op>, E_after_<op>, delta_E_ext_<op> float64[N_op] clean boundary を持つ演算子のみ
│   ├── shell_radius_min            float64[N_step]       シェル最小半径 [cm]（2D_RZ）。算出：\(R_{min} = \min\{r_c : \rho_c > 0.1 \times \rho_{max}\}\)。\(r_c\) はセル中心の R 座標。2D_RZ では密度基準を満たす全セルの最小値。1D_SPH では shell_radius_mean と同値
│   └── center_temperature          float64[N_step]       中心電子温度 [eV]
│
├── modes/                          （2D_RZのみ）
│   ├── P_ell                       float64[N_step, N_mode] Legendreモード振幅
│   └── ell_values                  int32[N_mode]           モード次数
│
├── laser/
│   ├── critical_surface_r          float64[N_step]       臨界面半径 [cm]
│   ├── absorbed_total              float64[N_step]       当該 step のレーザー吸収エネルギー総量 [erg]
│   ├── absorbed_power_total        float64[N_step]       当該 step のレーザー吸収パワー総量 [erg/s]（= absorbed_total / dt）
│   ├── commanded_power_total       float64[N_step]       当該 step のレーザー入射パワー総量 [erg/s]
│   ├── trace_unabsorbed_power_total float64[N_step]      raytrace/skip 直後の未吸収パワー総量 [erg/s]
│   ├── unabsorbed_power_total      float64[N_step]       当該 step のレーザー未吸収パワー総量 [erg/s]
│   ├── trace_absorption_efficiency_total float64[N_step] 当該 step の raytrace/skip 直後の総吸収効率 [dimensionless]
│   ├── absorption_efficiency_total float64[N_step]       当該 step の総吸収効率 [dimensionless]（= absorbed_power_total / commanded_power_total）
│   ├── transfer_blocked_power_total float64[N_step]      transfer 段で受け皿がなく捨てられたパワー総量 [erg/s]
│   ├── tail_closure_count          int64[N_step]         tail closure で終了した ray 数 [count]
│   ├── tail_closure_absorbed_power_total float64[N_step] tail closure で吸収されたパワー総量 [erg/s]
│   ├── critical_surface_hit_count  int64[N_step]         臨界面で打ち切られた ray 数 [count]
│   ├── cbet_exchanged_power_total  float64[N_step]       当該 step の CBET 交換パワー総量 Σ|ΔP|/2 [erg/s]（CBET 無効時 0）
│   ├── cbet_ledger_residual_rel    float64[N_step]       CBET pairwise 台帳残差 |ΣdQ|/Σ|dQ| [dimensionless]
│   ├── cbet_iterations             int64[N_step]         CBET 固定点反復数 [count]
│   ├── cbet_clamp_count            int64[N_step]         CBET 正値性 clamp 発生数 [count]
│   └── absorbed_fraction_beam_<i>  float64[N_step]       ビームiへの吸収配分重み [dimensionless]（単一ビームでは常に 1）
│
├── radiation/
│   ├── fld_outer_iterations        int64[N_step]         FLD outer (Picard) iterations used this step [count]（FLD 無効時 0；2026-07-16 additive）
│   ├── fld_outer_residual          float64[N_step]       FLD outer residual at exit [dimensionless]
│   ├── fld_outer_converged         int64[N_step]         1 = outer_tol reached, 0 = hit max_outer_iterations
│
├── mc/
│   ├── n_total                     int64[N_step]         全粒子数
│   ├── n_imc                       int64[N_step]         IMC粒子数
│   ├── n_ddmc                      int64[N_step]         DDMC粒子数
│   ├── n_census                    int64[N_step]         census粒子数
│   ├── n_absorbed                  int64[N_step]         吸収粒子数（ステップ内）
│   ├── n_escaped                   int64[N_step]         境界流出粒子数
│   ├── n_leaked                    int64[N_step]         DDMC→IMCリーク数
│   ├── ddmc_fraction               float64[N_step]       DDMC粒子割合
│   ├── weight_min                  float64[N_step]       粒子重み最小値
│   ├── weight_mean                 float64[N_step]       粒子重み平均値
│   ├── weight_max                  float64[N_step]       粒子重み最大値
│   ├── overshoot_count             int64[N_step]         最大原理違反セル数
│   ├── overshoot_max               float64[N_step]       最大超過率 \(\delta_{max}\)
│   ├── ddmc_mode_count             int64[N_step]         DDMCモードセル数（互換診断）
│   ├── imc_mode_count              int64[N_step]         IMCモードセル数（互換診断）
│   ├── mmatrix_violations          int64[N_step]         M-matrix違反検出数（互換診断）
│   ├── mmatrix_fallback_count      int64[N_step]         fallback発生回数（互換診断）
│   ├── omega_below_threshold       int64[N_step]         ωしきい値未満セル数（互換診断）
│   ├── interface_transitions       int64[N_step]         IMC/DDMC境界遷移数（互換診断）
│   ├── interface_reflections       int64[N_step]         IMC/DDMC境界反射数（互換診断）
│   ├── conversion_prob_violations  int64[N_step]         変換確率制約違反数（互換診断）
│   ├── ddmc_to_imc_conversions     int64[N_step]         DDMC→IMC変換数（互換診断）
│   └── rad_momentum_deposition     float64[N_step]       放射運動量沈着（互換診断）[g*cm/s]
│
├── difference/
│   ├── reference_valid             int64[N_step]         reference diagnostics 有効フラグ [count]
│   ├── eligible_cells              int64[N_step]         非void対象セル数 [cells]
│   ├── active_cells                int64[N_step]         \(W_i>0\) セル数 [cells]
│   ├── strong_cells                int64[N_step]         \(W_i\ge 0.5\) セル数 [cells]
│   ├── hybrid_suppressed_cells     int64[N_step]         hybrid diffusion mask で \(W_i=0\) にしたセル数 [cells]
│   ├── W_min                       float64[N_step]       reference weight 最小値 [dimensionless]
│   ├── W_mean                      float64[N_step]       reference weight 平均値 [dimensionless]
│   ├── W_max                       float64[N_step]       reference weight 最大値 [dimensionless]
│   ├── tau_min                     float64[N_step]       Rosseland optical depth 最小値 [dimensionless]
│   ├── tau_mean                    float64[N_step]       Rosseland optical depth 平均値 [dimensionless]
│   ├── tau_max                     float64[N_step]       Rosseland optical depth 最大値 [dimensionless]
│   ├── chi_mean                    float64[N_step]       radiation-matter mismatch 平均値 [dimensionless]
│   ├── chi_max                     float64[N_step]       radiation-matter mismatch 最大値 [dimensionless]
│   ├── reduced_flux_max            float64[N_step]       face reduced-flux proxy 最大値 [dimensionless]
│   ├── knudsen_max                 float64[N_step]       face Knudsen proxy 最大値 [dimensionless]
│   ├── front_grad_Te_max           float64[N_step]       \(|\Delta\ln T_e|\) 最大値 [dimensionless]
│   ├── front_grad_rho_max          float64[N_step]       \(|\Delta\ln\rho|\) 最大値 [dimensionless]
│   └── E_ref_total                 float64[N_step]       \(\sum_i V_i\sum_g E^{ref}_{i,g}\) [erg]
│
├── holo/
│   ├── n_core_cells                int64[N_step]         LO material-coupled cell 数 [cells]（legacy metric名）
│   ├── n_entered                   int64[N_step]         当該 selector 更新で LO coupling mask に入った cell 数 [cells]
│   ├── n_exited                    int64[N_step]         当該 selector 更新で LO coupling mask から出た cell 数 [cells]
│   ├── n_hard_exited               int64[N_step]         hard-exit した cell 数 [cells]
│   ├── n_island_rejected           int64[N_step]         island filter で拒否された cell 数 [cells]
│   ├── tau_R_min                   float64[N_step]       selector \(\tau_R\) 最小値 [dimensionless]
│   ├── tau_R_max                   float64[N_step]       selector \(\tau_R\) 最大値 [dimensionless]
│   ├── reduced_flux_max            float64[N_step]       selector reduced-flux 最大値 [dimensionless]
│   ├── E_LO_total                  float64[N_step]       \(\sum_i V_i\sum_g E^{LO}_{i,g}\) [erg]（global LO domain）
│   ├── E_LO_boundary_in            float64[N_step]       physical inner reflect boundary の LO 境界流入エネルギー [erg]（v1 は 0）
│   ├── E_LO_boundary_out           float64[N_step]       physical outer vacuum boundary の LO 境界流出エネルギー [erg]
│   ├── matter_delta                float64[N_step]       LO source solve が物質へ与えたエネルギー [erg]
│   ├── source_balance_error        float64[N_step]       LO source/boundary energy balance residual [erg]
│   ├── particle_net_source_core    float64[N_step]       LO-coupled cell 内の particle diagnostic net source \(\sum(\mathrm{rad\_dep}-\mathrm{rad\_emit})\) [erg]
│   ├── lo_particle_source_mismatch float64[N_step]       `matter_delta - particle_net_source_core`。LO-owned source と particle diagnostic source の差 [erg]
│   ├── Prr_coverage                float64[N_step]       LO-coupled cell×group の \(P_{rr}\) coverage 平均 [dimensionless]
│   ├── chi_min                     float64[N_step]       LO-coupled cell×group の \(P_{rr}/E\) 最小値 [dimensionless]
│   ├── chi_mean                    float64[N_step]       LO-coupled cell×group の \(P_{rr}/E\) 平均値 [dimensionless]
│   └── chi_max                     float64[N_step]       LO-coupled cell×group の \(P_{rr}/E\) 最大値 [dimensionless]
│
└── safety/
    └── clamp_count                 int64[N_step]         温度/エネルギークランプ発生回数 [count]
```

補足（実装互換）：
- 旧履歴ファイル追記時は `energy/E_*` 系の旧キー（例: `energy/E_int_e`, `energy/E_rad_esc`, `energy/epsilon_budget`）へフォールバックする場合がある。
- 後方互換として `implosion/rhoR_deg_<angle_mdeg>`（角度別スカラー）と `modes/a<ell>`（モード別スカラー）も併記される。hotspot tracer rhoR が有効な場合は `implosion/rhoR_hotspot_tracer_deg_<angle_mdeg>` も併記される。
- `difference/*` は PR3 で追加された optional history group である。旧履歴ファイルには存在しない場合があり、reader は欠損を disabled/zero diagnostics として扱う。旧ファイルへ追記する場合は追加後の step から dataset が作成され、長さ不一致は migration warning として扱う。
- `radiation/rad_emit` は particle emission diagnostic である。旧 checkpoint に存在しない場合、restart reader は 0 で補完する。
- `radiation/diag_*` は 1D S_N plateau investigation 用の output-only diagnostics である。restart reader は solver state として要求せず、旧 snapshot/checkpoint で欠損しても物理状態復元には影響しない。
- `holo/*` は HOLO selector/LO state 用の optional group である。checkpoint では `holo/E_LO` と `holo/consistency_source` に加えて LO gross source diagnostics `holo/rad_dep_LO` と `holo/rad_emit_LO`、passive closure diagnostics `holo/Prr_HO`、`holo/chi`、`holo/Prr_coverage` を保存できる。旧 snapshot/checkpoint にこれらが存在しない場合、restart reader は対応する `State.holo_*` field を 0 で補完する。旧 `holo/gamma` は前 step defect なので same-step corrector では restart 入力に使わず、欠損/存在のどちらでも `State.holo_consistency_source` は 0 から再生成する。旧履歴ファイルに HOLO 診断（`particle_net_source_core` / `lo_particle_source_mismatch` を含む）が存在しない場合は 0 diagnostics として扱い、追記時の長さ不一致は migration warning として扱う。
- `polar_center_treatment="button"` snapshots/checkpoints add compatible per-cell topology flags: `/hydro_flags/hydro_active`, `/hydro_flags/cell_is_void`, and `/mesh/topology/v1/cell_nverts`. The button cell reports the seam-polygon vertex count, dormant cells report `cell_is_void=1` and `cell_nverts=0`, and non-button output omits these additions to preserve existing deck output.

### 7.4 チェックポイントHDF5レイアウト（`<case>_ckpt_NNNNNN_rNNNN.h5`）

チェックポイントファイルは **スナップショットの全データ** に加え、粒子プールとRNG状態を含む。

```
/
├── [スナップショットと同一の mesh/, hydro/, radiation/, laser/ グループ]
│
├── hydro_flags/                     Hydro制御フラグ
│   ├── hydro_active                int8[N_cells]   セル単位活性フラグ（NUMERICS §2.1.1の一方向スイッチ）
│   └── cell_is_void                uint8[N_cells]  button dormant/void flag（button出力で追加）
│
├── particles/                       粒子プール（SoA形式）
│   │  attrs: n_particles (int64)
│   ├── pool_capacity               int64          プール容量（PhotonPool.capacity。再開時メタデータ復元に使用）
│   ├── pos_r                       float64[N_p]    R座標 [cm]（1D: r座標）
│   ├── pos_z                       float64[N_p]    Z座標 [cm]（1D: 3D方向追跡で使用）
│   ├── dir_r                       float64[N_p]    方向ベクトル Ω_r
│   ├── dir_z                       float64[N_p]    方向ベクトル Ω_z
│   ├── dir_phi                     float64[N_p]    方向ベクトル Ω_φ
│   │  注：内部表現（R,Z,φ）で保存。ARCHITECTURE §5.3 PhotonPool SoA フィールド名と1:1対応。
│   ├── energy                      float64[N_p]    粒子エネルギー [erg]
│   ├── birth_energy                float64[N_p]    生成時エネルギー [erg]
│   ├── sign                        int8[N_p]       粒子符号（+1 or -1）。旧checkpointで不在の場合は +1 として復元
│   ├── group_id                    uint16[N_p]     所属群インデックス（ARCHITECTURE §5.3 PhotonPool の uint16_t 準拠）
│   ├── cell_id                     int32[N_p]      所属セル **グローバル** インデックス。ランク数変更リスタート時に新パーティションへの再配布に必要。DDMC粒子は pos=NaN のため位置ベース再同定不可。
│   │                                               **1D_SPH**: `global = cell_id_local + cell_offset`
│   │                                               **2D_RZ**: `local(i,j) = (cell_id/nz_local, cell_id%nz_local)` →
│   │                                                 `global = (i + ir_start) * nz_global + (j + jz_start)`
│   │                                               （単純な +offset は nz_local ≠ nz_global 時にストライド不整合。ARCHITECTURE §5.3 参照）
│   ├── mode                        uint8[N_p]      0=IMC, 1=DDMC, 2=RW（legacy enum値。現行 PGRW 実装は 2 を生成しない）
│   ├── global_id                   uint64[N_p]     大域一意ID（再現性用）
│   ├── weight                      float64[N_p]    統計的重み [無次元]
│   └── time_remain                 float64[N_p]    残存時間 [s]（t^{n+1} - t_p）
│
├── rng/                             RNG状態（Philox4x32-10）
│   ├── rng_counter                 uint32[N_p]     粒子ごとの乱数消費カウンタ（PhotonPool.rng_counter と1:1対応）
│   └── global_id                   uint64[N_p]     大域一意ID（curand_init(seed=global_id ^ user_seed, subsequence=step_number, offset=rng_counter) で復元。user_seed = Main.seed。NUMERICS §12.7.1準拠）
│
├── output_state/                    出力タイミング状態（時間間隔ベース出力用）
│   ├── t_next_plot              float64     次回plot出力時刻 [s]（-1.0=無効）
│   ├── t_next_history           float64     次回history出力時刻 [s]（-1.0=無効）
│   └── t_next_checkpoint        float64     次回checkpoint出力時刻 [s]（-1.0=無効）
│
└── time_state/                      時間管理・累積診断状態
    ├── t                        float64     現在時刻 [s]
    ├── step                     int32       **最後に完了した**ステップ番号（0始まり。restart 時は `step + 1` から再開。
    │                                                    RNG: `curand_init(subsequence = step + 1)` で次ステップのストリームを開始。NUMERICS §12.7.1）
    ├── dt                       float64     現在タイムステップ幅 [s]（restart時の Δt 成長制限 ≤1.2×dt に必要、NUMERICS §2.2）
    ├── ale_last_applied_step    int32       1D V3 ALE が最後に commit した step（旧checkpointでは -1）
    ├── axis_margin_initial      float64     2D_RZ ALE Phase 7 軸feasibility guard基準値 [cm²]（State.axis_margin_initial。旧checkpointでは -1.0 として扱い、初回ALE時に現在形状からlazy-init）
    ├── axis_mass_initial        float64[nz] M_{0,j}^init for Phase 9 budget gate（旧checkpointで不在の場合は empty として扱い、必要時に lazy-init）
    ├── axis_inflow_budget       float64[nz] B_{0,j}^n cumulative inflow budget（旧checkpointで不在の場合は empty として扱い、必要時に lazy-init）
    ├── E_safety                 float64     伝導安全補正累積値 [erg]（NUMERICS §4.2.2, §10.2）
    ├── E_numerical_loss         float64     退化セル注入不能エネルギー累積値 [erg]（NUMERICS §10.2）
    ├── E_laser_deposited        float64     レーザー沈着エネルギー累積値 [erg]
    ├── E_laser_escaped          float64     レーザー脱出エネルギー累積値 [erg]
    ├── E_rad_escaped            float64     放射脱出エネルギー累積値 [erg]
    ├── E_floor_injected         float64     フロア注入エネルギー累積値 [erg]
    ├── E_pdV_bdry               float64     境界PdV仕事累積値 [erg]（NUMERICS §10.2）
    ├── E_Marshak_in             float64     Marshak境界入射エネルギー累積値 [erg]（NUMERICS §10.2）
    ├── E_solver                 float64     Hypre残差エネルギー累積値 [erg]（v1.0=0、NUMERICS §10.2）
    ├── adaptive_av_r0           float64     adaptive AV の latch 済み初期半径 [cm]（旧checkpointでは0）
    ├── adaptive_av_last_rs      float64     adaptive AV tracker の前回 shock 半径 [cm]（旧checkpointでは0）
    ├── adaptive_av_last_us      float64     adaptive AV tracker の前回 shock 速度 [cm/s]（旧checkpointでは0）
    ├── adaptive_av_rs_min       float64     adaptive AV tracker の最小 shock 半径 [cm]（旧checkpointでは∞）
    ├── adaptive_av_tracker_steps int32      adaptive AV tracker の valid step 数（bounce warm-up 用。旧checkpointでは0）
    ├── adaptive_av_mode         int32       adaptive AV mode（0=BASE, 1=PRIMARY_FULL, 2=PRIMARY_TAPER, 3=REBOUND。旧checkpointでは0）
    ├── adaptive_av_tracker_valid int32      adaptive AV tracker 履歴が有効なら1（旧checkpointでは0）
    ├── adaptive_av_bounce_seen  int32       adaptive AV bounce latch が立っていれば1（旧checkpointでは0）
    └── user_seed                uint64      Main.seed の値（restart時の一致検証に使用。ARCHITECTURE §8.4）
```

**リスタート手順**：
1. チェックポイント読み込み → mesh/hydro/radiation の全場を復元
2. hydro_flags/ から `hydro_active` フラグを復元（一方向スイッチ状態の保持に必須）
3. particles/ から粒子プールを再構築（SoA形式のまま）。`particles/sign` が不在の旧チェックポイントは全粒子 `sign=+1` として補完する。**DDMC NaN sentinel 検証**: 復元後、`mode==DDMC` の粒子に対し `pos_r/pos_z/dir_r/dir_z/dir_phi` が NaN であることをアサートする。NaN でない場合は強制的に NaN を設定し WARNING を出力する（旧チェックポイントとの後方互換を維持しつつ、NaN 不変条件を保証。CUDA_KERNELS §6.0d サブステップ1 参照）。**メタデータ再構築**: N_total はHDF5データセット次元から取得、チェックポイントは alive 粒子のみ保存するため復元後に alive[i]=1 を全粒子に設定し n_alive = N_total、n_census = n_alive（再開直後は全alive粒子がcensus扱い）、capacity はチェックポイントの particles/pool_capacity から復元
4. rng/ から rng_counter + global_id を復元し、`curand_init(seed=global_id ^ user_seed, subsequence=step+1, offset=rng_counter)` で各粒子のRNGストリームを再構築。**注: subsequence は checkpoint の step（最後に完了したステップ）+ 1** — 再開ステップのストリームを開始する（step そのままだとチェックポイント直前のステップと同一ストリームになり、乱数が重複する）。user_seed = time_state/user_seed（チェックポイント保存値）。現在の namelist の Main.seed と不一致の場合は `ConfigError`（RNGストリーム連続性の破壊を防止。ARCHITECTURE §8.4、NUMERICS §12.7.1準拠）
5. output_state/ から `t_next_plot`, `t_next_history`, `t_next_checkpoint` を復元。グループ不在（旧checkpoint）の場合は `t + X_every_s` で再初期化。output パラメータが変更された場合（「調整可」）は `t_next_X = t + new_X_every_s` で再計算
6. time_state/ から `t`, `step`, `dt`, `ale_last_applied_step`, `axis_mass_initial`, `axis_inflow_budget`, `E_safety`, `E_numerical_loss`, `E_laser_deposited`, `E_laser_escaped`, `E_rad_escaped`, `E_floor_injected`, `E_pdV_bdry`, `E_Marshak_in`, `E_solver` と adaptive AV tracker scalars を復元。dt は NUMERICS §2.2 の成長制限（≤1.2×dt）を尊重するために必須。`ale_last_applied_step` は 1D V3 ALE の min-step gate continuity を保証し、旧チェックポイントでは -1 として扱う。Phase 9 の `axis_mass_initial` / `axis_inflow_budget` は旧チェックポイントで不在なら empty として扱い、budget gate が必要になった時点で lazy-init する。累積診断値はエネルギー収支の連続性を保証。旧チェックポイント（E_pdV_bdry、adaptive AV tracker scalars 等が不在）の場合は 0.0 で初期化（後方互換）
7. `laser_cache_valid = false` に設定し、`laser_dep_frac` をクリア（リスタート直後のステップで必ず full raytrace を実行。ARCHITECTURE §4.6、CUDA_KERNELS §9 Phase 3 準拠）
8. 凍結設定を現在のnamelistと比較（凍結項目の不一致は `ConfigError`、調整可能項目は INFO/WARNING）

`eos_signature` checkpoint validation (implementation note):
- Checkpoints store per-material EOS signatures at `/metadata/eos/eos_signature` (`uint64` array, hash over EOS model/path/grid metadata).
- On restart, if `/metadata/eos/eos_signature` exists and either side uses table EOS, each material signature must match (`ConfigError` on mismatch). If the path is missing (legacy checkpoint), EOS-signature validation is skipped with warning.
- Checkpoints also store `/metadata/frozen_config`; if it is canonical JSON, restart requires semantic equivalence with current canonical JSON (`ConfigError` on mismatch). Legacy non-JSON `frozen_config` is compared as warning-only compatibility check.

**リスタート時のパラメータ変更制約**：

| 区分 | パラメータ | 変更可否 |
|------|-----------|---------|
| 凍結（ConfigError）| dimension, mesh, materials, n_groups, group_bounds_eV, Main.seed | 変更不可 |
| 調整可（INFO）| t_end, dt, output, diagnostics | 変更可 |
| 調整可（WARNING）| radiation, laser, numerics | 変更可（物理的整合性に注意） |

メッシュ変更は `ConfigError` で禁止。変更が必要な場合は新規シミュレーションとして開始すること。
`group_bounds_eV` は各境界で
\(|old-new| \le \max(10^{-12}\max(|old|,|new|),\;10^{-14}\,\mathrm{eV})\)
を満たさなければ `ConfigError("group_bounds_eV mismatch: checkpoint has {old}, namelist has {new}")` を送出。

**ランク数変更時のリスタート**：rank 0 が全データを読み込み → 再分割 → 再配置 → 粒子再配布。ランク数変更時も統計的再現を保証（Persistent Warp モデルにより、同一ランク数であってもbitwise再現は保証しない）

### 7.5 出力の互換性規約
- **スキーマバージョン**：ルートグループの `attrs: schema_version` に整数を格納（v1.0は `1`）
- **後方互換**：データセットの追加は許可するが、既存データセットの名前変更・型変更・削除は禁止
- **スキーマ変更ログ**：ALE-FIX-1 で 1D ALE diagnostics の
  `/diagnostics/ale_center/` は出力対象から削除された。Legacy output に存在する場合は
  解析側で optional group として扱う。
- **読込時のバージョン規約**：
  - `schema_version == current`：通常読込
  - `schema_version < current` または属性欠落（v0）：後方互換リーダーで既定値補完し WARNING
  - `schema_version > current`：`ConfigError` で拒否
- **圧縮**：gzip level 4 を既定とする（`Output.compression: Literal["none","gzip"]`（既定 `"gzip"`）、`Output.compression_level: int`（既定 4、0-9の範囲））
- **単位属性**：全数値データセットに `units` string属性を付与する。単位文字列は §4（cgs + eV）に準拠

---

## 8. 基準問題：GXII（正十二面体12ビーム）による500 µmカプセル爆縮
TENRYUの実アプリ基準として、GEKKO XII（GXII）同等の **12ビーム直接照射**爆縮を標準ケースにする。

### 8.1 ターゲット（与条件）
- 外径：**500 µm**（外半径 \(R_{out}=250\,\mu m = 0.025\) cm）
- シェル：**CH（ポリスチレン）**
  - 厚さ：**20 µm**（内半径 \(R_{in}=230\,\mu m\)）
  - 初期密度：\(\rho_{CH} = 1.05\) g/cm³
  - 組成：C₁H₁（A = 6.5、Z_eff = 3.5）
- 充填ガス：**DT（重水素-三重水素）**
  - 初期密度：\(\rho_{DT} = 0.010\) g/cm³（10 mg/cc）
  - 組成：D₀.₅T₀.₅（A = 2.5、Z = 1）
- 外部：真空（数値的には floor = \(\rho_{floor}=10^{-10}\) g/cm³ を設定。実装上は is_void 材料マスク（§6.4.3）で表現し、伝導・EOS から除外する）
- **プレプラズマ・シード（1D_SPH FLD 回帰の既定）**：冷間開始の密度ステップでは 351 nm 光線が臨界面で IB 光路長ゼロのまま全反射し、コロナが自己形成できない（実測吸収率 4×10⁻⁴/1 ns）。このため基準デッキはシェル外面に指数ランプ \(\rho = 0.05\,e^{-(r-R_{out})/2\,\mu m}\) g/cm³（下限 \(10^{-7}\)、外端 \(R_{out}+20\,\mu m\)、付加質量 <1%）のシード・コロナを置き、臨界面（CH@351nm: 0.028 g/cm³）を分解されたランプ内に配置する。直接照射コードの標準的な初期化であり、golden はこのシードを含む定義で取得する。
- 初期温度：\(T_e = T_i = 0.025\) eV（室温相当、全領域均一）
- 初期速度：ゼロ
- EOS：**理想気体（γ = 5/3）** を既定ベンチマーク条件とする。テーブル EOS 使用時は **SESAME**（CH=mat 7593, DT=mat 5265）を推奨。IONMIX テーブル切替でも同一ターゲットを使用可
- Opacity：IONMIX テーブル（`CH.ionmix`, `DT.ionmix`）。理想気体ベンチマーク条件では `opacity.model="constant"` とし、放射は固定不透明度（\(\kappa_P = \kappa_R = 100\) cm²/g）を使用
- \(\bar{Z}\) モデル：`fixed`（完全電離仮定）

> 本パラメータは GXII 同等の直接照射爆縮ベンチマークとして設計。
> テーブル EOS 使用時は SESAME（CH=mat 7593, DT=mat 5265）+ IONMIX opacity を推奨。IONMIX EOS への切替も可能。

### 8.2 レーザー（与条件）
- 波長：**3ω（351 nm）**、\(\lambda = 0.351\,\mu m\)
- 総エネルギー：**10 kJ**（= \(10^{11}\) erg）
- パルス形状：**1 ns 矩形パルス（square pulse）**
  - 立ち上がり時刻 \(t_{on} = 0\) s、立ち下がり時刻 \(t_{off} = 1 \times 10^{-9}\) s
  - パワー：\(P_{total}(t) = E_{total}/\Delta t_{pulse} = 10^{13}\) W（= \(10^{20}\) erg/s）
  - namelist表記：callable（10 ps の有限幅ランプで正則化した台形波形 — 同時刻の重複点はテーブル線形補間でゼロ除算を生じうるため。実装例は `examples/verification/gxii_1d_fld_regression.py` の `pulse_power`。旧記載の dict 形式は実装に存在しない — §6.4.6 の 2026-07-12 doc-truth 訂正参照）
- ビーム配置：**正十二面体の面に均等に12本**（GXII同等）
  - 実装上は「面法線＝正二十面体（icosahedron）の12頂点方向」を単位ベクトルとして保持
  - 各ビームのパワー：\(P_{beam} = P_{total}/12\)
- ビームパラメータ：
  - F値：f/3（f_number = 3.0）
  - プロファイル：super-Gaussian（m = 4）、\(w_0 = 200\,\mu m\)
  - デフォーカス：D/R = 0（ターゲット中心に集光）
- レーザー物理：幾何光学 + IB のみ（LPI/CBETはOFF）

### 8.2.1 数値パラメータ（ベンチマーク既定）
- グリッド（1D_SPH）：\(N_r = 200\) セル、\(R \in [0, 0.05]\) cm（= [0, 500 µm]）
- グリッド（2D_RZ）：\(N_r \times N_z = 200 \times 400\)
- 放射群数：\(G = 20\)、群境界：\([0.01, 100]\) eV を対数等間隔
- IMC粒子数：50 / cell / group（テスト用）、本番は 200 以上推奨
- タイムステップ：\(\Delta t_{init} = 10^{-15}\) s、成長率 1.2、\(\Delta t_{max} = 10^{-11}\) s
- 総シミュレーション時間：\(t_{end} = 2 \times 10^{-9}\) s（パルス終了後1 ns）
- レイ数/ビーム：1000（1D_SPH）、50×50（2D_RZ）

### 8.3 次元ごとの取り扱い（重要）
- **1D_SPH**（既定：2Dレイトレース `laser.mode="raytrace_2d"`）：
  - **1ビーム分のみレイトレース** を実行し、複数ビームの結果はパワースケーリングで重ね合わせ（NUMERICS §5.6.4）
  - 代表ビーム（1本目）のRZ円筒座標系で2Dレイトレースを実施
  - 1D球対称プロファイル \(\rho(r)\) を \(\rho(\sqrt{R^2+Z^2})\) として2Dマッピング
  - レイはビーム軸対称の1D配列（R方向のみ）で生成
  - 正規化吸収分率を全ビームのパワー合計でスケーリングし、1D球座標へ射影
  - F値・プロファイルがビーム間で異なる場合はパラメータグループ毎にレイトレース
  - LaserMeshは臨界密度以下の領域のみカバーし、密度勾配が大きい領域で自動的にメッシュを細かくする
  - オプション：`laser.mode="radial_absorption_1d"` で全ビームパワーを1本の inward radial flux として積分する
  - オプション：`laser.mode="spherical_average"` で等方平均（P0）入射（検証・回帰テスト用）
- **2D_RZ**（既定：3Dレイトレース `laser.mode="raytrace_3d"`）：
  - 流体場は軸対称 \(\rho(R,Z)\) だが、各ビームを **3D空間で追跡** する（NUMERICS §5.3.4）
  - LaserMeshは流体対称軸に沿う2D RZ格子（§5.4.2）
  - レイ初期化はビーム軸直交平面上の2D断面配列（NUMERICS §5.6.3 (b)）
  - 場の参照は \(R=\sqrt{x^2+y^2}\) で2D LaserMeshから取得（3Dメッシュ不要）
  - 沈着は3D→\((R,Z)\) 写像でLaserMesh→HydroMeshへ直接マッピング（1D球座標転写は不要）
  - 多ビーム重ね合わせ：極角θ＋パラメータ（F値、プロファイル）でグループ化し、グループ毎に1回ずつ3Dレイトレース
  - GXII 12ビーム（正十二面体配置）では極角θの対称性により数グループで完結

### 8.4 この基準問題で評価する量（診断）
- 爆縮指標：\(\rho_{peak}(t)\)、\(\rho R(t)\)、shell radius（平均/モード）
- エネルギー収支：入射/吸収/未吸収（臨界反射含む）/放射流出/熱伝導損失
- 流体歪み：RZの \(P_2,P_4\) モード振幅、シェル厚変動
- 照射パターン：吸収パワー角度分布、時間変化、臨界面位置
- 解析出力：中心線プロファイル、等密度面形状、レーザー吸収位置分布

### 8.5 ゴールデン値の取得方法（自己収束）
基準問題のゴールデン値（参照解）は **自己収束（格子独立解）** で取得する。外部コードへの依存を避け、TENRYUの独立検証を担保する。

- **方法**：メッシュ解像度を \(N,\,2N,\,4N\) と倍増し、Richardson外挿で収束解を推定する
  - 1D_SPH：\(N_r = 200,\,400,\,800\)
  - 2D_RZ：\(N_r \times N_z = 200{\times}400,\,400{\times}800,\,800{\times}1600\)
  - 粒子数は各解像度で十分な統計（≥ 200/cell/group）を確保し、統計誤差がメッシュ誤差を支配しないようにする
- **判定基準**：隣接解像度間の主要スカラー量（\(\rho_{peak},\,\rho R,\,E_{absorbed}\) 等）の相対変化が \(\varepsilon_{conv}\)（既定 \(10^{-3}\)）以下で収束と判断する
- **外部コード参照解との関係**：DRACO/LILAC等の参照解が入手可能な場合は **cross-validation** として比較するが、primary criterionはあくまで自己収束とする。これにより外部コードアクセスへの依存を排除する

---

## 9. 安全な既定値（Defaults）
### 9.1 既定値（推奨）

**Mesh / floors**：
- geometry_1d="spherical"（W-G。非球面は 1D_SPH 専用: imc_ddmc 不可 / cylindrical+sn_transport 不可 / laser は radial_absorption_1d 限定）
- floors：rho_floor_gcc=1e-10, Te_floor_eV=1e-3, Ti_floor_eV=1e-3（NUMERICS §1.1.7準拠）
- motion：lagrangian（1D）、ale（2D）
- graded grid：edge_ratio=`0.1`, sg_order=`4`, sg_sigma=`0.7`
- explicit node lists：`explicit_nodes: list[float]=[]`, `explicit_nodes_z: list[float]=[]`, `explicit_nodes_theta: list[float]=[]`（指定時は有限値の狭義単調増加列）
- mesh topology defaults: topology_scheme=`"single_block"`; pentagon_belt_layers=`[]`; multiblock_cart_core_r_c=`s_max/12`; multiblock_cart_core_r_match=`2*r_c`; multiblock_cart_core_n_c=`nz/4` with `nz=4*n_c` required for multiblock; multiblock_cart_core_bridge_layers=`max(4,n_c/8)`; multiblock_cart_core_bridge_grading=`"uniform"`; multiblock_transition_scheme=`"hermite_bridge"`; multiblock_cap_p=`6.0` (dimensionless rounded/squircle cap exponent); multiblock_bridge_elliptic_sweeps=`0`; multiblock_bridge_elliptic_omega=`0.5`. `pentagon_belt_layers` is frozen only for `pentagon_belt_shell`; this scheme does not derive or freeze `multiblock_cart_core_*` defaults. `multiblock_cart_core_*` fields are unset/frozen-config elided unless topology_scheme is explicitly `multiblock_cart_core_polar_shell`, `multiblock_half_butterfly_5block`, or `multiblock_half_butterfly_trifan_cap_5block`. The tri-fan cap topology is default-off and opt-in only; old implicit `single_block` frozen configs still elide `topology_scheme`. S5 transition fields and `multiblock_cap_p` are frozen-config elided whenever they equal defaults and are unused, preserving legacy frozen-config byte identity.
- mesh topology defaults: topology_scheme=`"single_block"`; multiblock_cart_core_r_c=`s_max/12`; multiblock_cart_core_r_match=`2*r_c`; multiblock_cart_core_n_c=`nz/4` with `nz=4*n_c` required for multiblock; multiblock_cart_core_bridge_layers=`max(4,n_c/8)`; multiblock_cart_core_bridge_grading=`"uniform"`; multiblock_transition_scheme=`"hermite_bridge"`; multiblock_cap_p=`6.0` (dimensionless rounded/squircle cap exponent); multiblock_bridge_elliptic_sweeps=`0`; multiblock_bridge_elliptic_omega=`0.5`; multiblock_outer_svec_tangent_balance=`true`. `multiblock_cart_core_*` fields are unset/frozen-config elided unless topology_scheme is explicitly `multiblock_cart_core_polar_shell`, `multiblock_half_butterfly_5block`, or `multiblock_half_butterfly_trifan_cap_5block`. The tri-fan cap topology is default-off and opt-in only; old implicit `single_block` frozen configs still elide `topology_scheme`. S5 transition fields, `multiblock_cap_p`, and `multiblock_outer_svec_tangent_balance` are frozen-config elided whenever they equal defaults and are unused, preserving legacy frozen-config byte identity.
- logical_mesh_2d=`"rectangular_rz"`, polar_center_treatment=`"annular"`, center_button_outer_node_ring=`2`, polar_equal_mu_zoning=`False`, polar_theta_min=`0.0`, spherical_polar_s_max=`1.0`, box_r_max/box_z_min/box_z_max=unset（`polar_in_box` では必須）, box_center_z=`0.0`, cone_theta_wall/cone_tip_radius/cone_activation_radius=unset, cone_fine_cells_minus=`4`, cone_fine_cells_plus=`4`, cone_angular_growth_max=`1.25`, cone_tip_style=`"single_line"`, polar_prefix_nr=`-1`（`polar_in_box` では `len(explicit_nodes)-1` の派生値）, morph_rings=`16`, collar_rings=`6`, morph_growth_max=`1.20`, spherical_polar_kappa=`0.5`
- cone_shell wall/map defaults: `cone_shell_alpha`/`cone_shell_wall_thickness`/`cone_shell_tip_radius`/`cone_shell_tip_z`/`cone_shell_wall_length` は unset、`cone_shell_tip_radius_kind="inner_face"`, `cone_shell_axis_sign=1`, `cone_shell_n_cells=10`, `cone_shell_n_growth=1.25`, `cone_shell_tip_size_factor=3.0`, `cone_shell_base_size_factor=8.0`, `cone_shell_tip_hold=unset -> 4*t_w`, `cone_shell_grading_length=unset -> min(0.35*L_w,20*t_w)`, `cone_shell_l_ratio_max=1.12`, `cone_shell_tip_rotation_length=unset -> 3*t_w`, `cone_shell_base_cut="planar"`, `cone_shell_base_rotation_length=unset -> cone_shell_tip_rotation_length`, `cone_shell_farfield_target_measure="station_uniform"`。
- cone_shell strip defaults: outer `(first_factor,layers,growth)=(0.8,10,1.18)`; inner `(1.0,10,1.15)`; end `(1.0,10,1.15)`。factor は wall face width \(h_{n,0}\) に対する無次元比。
- cone_shell C4 builder-derived storage defaults: `cone_shell_cavity_cells=0`, `cone_shell_exterior_cells=0`, `cone_shell_tip_fill_layers=0`（いずれも deck key ではなく frozen config には出力しない。builder がそれぞれ `[6,32]`, `[6,32]`, `[6,48]` の topology count へ派生）。
	- rezoning / ale：enabled=True（2D_RZ ALE 用）, ale_identity_mode=False, ale_mover_diag=False, ale_preserve_lagrangian_velocity_carry=False, every_n_steps=5, force_rezone_every_n_steps=0, warmup_steps=0, relaxation=0.2, spacing_ratio_threshold=1.5, quality_threshold=0.2, max_iterations=20, max_displacement_fraction=0.5, remap_limiter=`"van_leer"`, remap_ms_midpoint=False, remap_ms_post_check=False, remap_ms_post_max_iter=3, remap_ms_rescale_floor=0.01, ke_fixup=True, ke_conservation_closure=False, ke_conservation_closure_audit=False, ke_closure_redistribute_floor=False, debug_per_remap_log=False, shock_sensor_guard_cells=2, density_jump_threshold=0.1, Te_jump_threshold=0.2, preventive_axis_guard_fraction=0.1, axis_z_motion=`"fixed"`, winslow_axis_kappa=0.7, button_morph={enabled=False, t_start_s=0.0, t_end_s=0.0, max_step_fraction=0.05, every_n_steps=1}, reference_barrier_enabled=False, reference_target=`"none"`, reference_blend_default=1.0, reference_volume_floor_rel=1e-8, reference_corner_j_floor_rel=1e-8, reference_gauss_j_floor_rel=1e-8, reference_linesearch_max_iters=24, reference_force_engage_every_step=False, reference_trigger_axis_margin_enabled=True, reference_trigger_axis_margin_threshold=1e-2, reference_trigger_corner_j_ratio_enabled=True, reference_trigger_corner_j_ratio_threshold=0.5, driver_retry_reference_barrier_enabled=False, driver_retry_reference_barrier_K_axis=4, driver_retry_reference_barrier_eta_axis=0.05, driver_retry_reference_barrier_max_attempts=6, driver_retry_reference_barrier_same_sig_max=3, driver_retry_reference_barrier_cell_window=2, driver_retry_reference_barrier_dt_collapse_rel=1e-3, driver_retry_reference_barrier_lambda_collapse_threshold=1e-3, driver_retry_reference_barrier_lambda_collapse_count=2, driver_retry_reference_barrier_quality_progress_factor=1.25, driver_retry_reference_barrier_quality_progress_count=2, driver_retry_reference_barrier_rezone_freq_warn_fraction=0.20, driver_retry_reference_barrier_rezone_freq_window=200, driver_retry_reference_barrier_chi=0.8, driver_retry_reference_barrier_q_retry=0.5, remap_damage_gate_enabled=False, remap_damage_dmax=0.05, remap_damage_axis_eta=0.02, remap_damage_axis_budget_enabled=False, remap_damage_axis_budget_factor=2.0, predictive_acceptance_enabled=False, predictive_acceptance_axis_floor_fraction=0.0, predictive_acceptance_cell_vol_floor_fraction=0.0, safe_backtrack_enabled=False, safe_backtrack_min_exp=20, safe_backtrack_binary_iters=8, euler_window.enabled=False, euler_window.shape=`"rectangle"`, euler_window.transition_width=0.0, rezone_solver=`"legacy_winslow"`, m1_gamma_align=0.0, m1_lambda_tether=0.0, m1_theta_reg=0.0, m1_sweeps=8, m1_min_j_dec_rel=0.0, m1_barrier_beta=1e-3, rezone_local_admissibility_linesearch=False, rezone_local_j_floor_rel=1e-8, rezone_local_linesearch_max_halves=8, reject_zero_gauss_j=False, zero_gauss_j_floor_rel=1e-8, lambda_sweep_diagnostic_enabled=False, lambda_sweep_target_cell_c=-1, lambda_sweep_target_cell_i=-1, lambda_sweep_target_cell_j=-1, lambda_sweep_max_exp=20, corner_jacobian_post_tangle_enabled=True, corner_post_tangle_strict_floor_enabled=False, local_boundary_repair_enabled=False, multi_node_boundary_repair_enabled=False, multi_node_interior_repair_enabled=False, axis_variational_projection_enabled=False, multiblock_cross_seam_rezone_enabled=False, multiblock_scaled_reference_enabled=False, multiblock_differential_reference_enabled=False, multiblock_differential_reference_band_count=64, multiblock_differential_reference_smoothing_g0=0.03, multiblock_differential_reference_nu=0.10, multiblock_differential_reference_eps_v=0.03, multiblock_differential_reference_s_cap_min_rel=1e-3, multiblock_differential_reference_xi_seam_tol=1e-9, multiblock_differential_reference_sigma_warn_floor=0.5, multiblock_lagrangian_bulk_center_patch_reference_enabled=False, multiblock_center_patch_ring_max=4, multiblock_center_patch_xi_center=0.0, multiblock_center_patch_halo_layers=2, multiblock_center_patch_vol_on=0.05, multiblock_center_patch_vol_off=0.10, multiblock_center_patch_cornerj_on=0.03, multiblock_center_patch_cornerj_off=0.08, multiblock_center_patch_gaussj_on=0.03, multiblock_center_patch_gaussj_off=0.08, multiblock_path_admissibility_enabled=False, path_admissibility_floor=0.01, dt_rejection_factor=0.5, max_dt_rejections=8, axis_band_managed_remap_enabled=False, axis_band_managed_remap_width=3, axis_band_managed_remap_max_width=6, axis_band_managed_remap_every_hydro_half_step=True, axis_band_managed_remap_margin_trigger=1e-4, axis_band_managed_remap_equal_volume=True, axis_band_managed_remap_include_radiation_groups=True, axis_rezone_enabled=False, axis_rezone_trigger_edge_fraction=0.1, axis_rezone_trigger_min_altitude_fraction=0.1, axis_rezone_eta_floor=1e-2, emergency_cell_deactivation_enabled=False, axis_repair_mode=`"full_winslow"`, remap_scheme=`"legacy_split"`, remap_ms2_limiter=`"van_leer"`, swept_volume_sign_fixed=True, convergence_tol=1e-6; the six M1 keys are emitted by frozen-config serialization only when `rezone_solver="m1_tmop"`; legacy behavior requires `Numerics.profile.legacy_regression@2026-07-27` or an explicit False.
		- axis ALE rezone defaults: `axis_rezone_enabled=False`; `axis_rezone_trigger_edge_fraction=0.1`, dimensionless ratio, validation `(0,1]`; `axis_rezone_trigger_min_altitude_fraction=0.1`, dimensionless ratio, validation `(0,1]`; `axis_rezone_eta_floor=1e-2`, dimensionless ratio, validation `(0,1)`.
		- ALE core-freeze defaults: `core_freeze_enabled=False`; `core_freeze_source="gas_tracer"`; `core_freeze_tracer_cut=0.5`, validation `[0,1]`; `core_freeze_halo_layers=1`, validation `>=0`; `core_freeze_apply_to_axis_rezone=True`; `core_freeze_skip_velocity_projection=True`. Frozen-config serialization emits these fields only when `core_freeze_enabled=True`, preserving default-off byte identity.
		- ALE central pseudo-core defaults: `central_pseudo_core_enabled=False`, `central_pseudo_core_s_c=0.0` cm; enabled membership is ring-conforming with no halo. Frozen-config serialization emits these fields only when enabled or non-default, preserving default-off byte identity.
		- ALE ring-absorption defaults: `central_pseudo_core_ring_absorption_enabled=False`, `tau=0.05`, `max_rings=0` (unlimited up to the material/topology guard), `gas_tracer_min=0.99` (mass-weighted row fraction), `gas_tracer_cell_min=0.5`. The historical `TENRYU_I1B_RING_ABSORB*` environment variables override the namelist when SET; frozen-config emits the block only when enabled or non-default.
		- ALE core1d sub-model defaults (promotion Wave A): `central_pseudo_core_core1d_enabled=False`, `build_shells=48` `[4,4096]`, `split_append=0` (off) `[0,1024]`, `av_c1=0.5`, `av_c2=4.0`, `cfl=0.25` `(0,1)`, `piston_cap=10.0`, `max_substeps=20000`. Values are injected once into the core-header-free sub-model (`core1d::set_params`) at `ensure_built`; the historical `TENRYU_I1B_CORE_1D_*` envs override when SET. Frozen-config emits the block only when enabled or non-default. NUMERICS §13.
		- ALE absorption-schedule defaults (promotion Wave B): `central_pseudo_core_spherical_absorb_gasfront=False`, `central_pseudo_core_spherical_absorb_alpha=0.0` (off; otherwise `(0,1)`), `central_pseudo_core_spherical_absorb_pjump=0.0` (off; otherwise `>1`), `central_pseudo_core_mixed_absorb_enabled=False`, `central_pseudo_core_absorb_watch_rows=1` `[1,8]`. The historical `TENRYU_I1B_SPHERICAL_ABSORB_*`, `TENRYU_I1B_MIXED_ABSORB`, and `TENRYU_I1B_ABSORB_WATCH_ROWS` envs override when SET; frozen-config emits the block only when enabled or non-default.
		- ALE terminal endgame defaults (promotion Wave C): `central_pseudo_core_terminal_absorb_enabled=False`, `central_pseudo_core_terminal_rebound_factor=1.02` (`>1`), `central_pseudo_core_terminal_tail_dt_s=1.0e-12` s (`>0`). The historical `TENRYU_I1B_TERMINAL_ABSORB`, `TENRYU_I1B_TERMINAL_REBOUND_FACTOR`, and `TENRYU_I1B_TERMINAL_TAIL_DT` envs override when SET; `TENRYU_I1B_TERMINAL_TAIL_LOG_EVERY` remains env-only. Frozen-config emits the block only when enabled or non-default.
		- ALE remap conservation gate defaults (promotion Wave D): `remap_mass_closure_reject_tol=0.0` (`>=0`, 0=off) and `rezone_closure_cooldown_steps=50` (`>=1`). Default remains off until the default-on regression sweep passes; the certified I1-B deck sets `remap_mass_closure_reject_tol=1e-8` explicitly. The historical `TENRYU_I1B_REMAP_CLOSURE_REJECT_TOL` and `TENRYU_I1B_REZONE_CLOSURE_COOLDOWN_STEPS` envs override when SET; frozen-config emits the block only when non-default.
		- ALE certified transport/robustness flag defaults (promotion Wave E): `csr_optionb_coherent_enabled=False`, `csr_optionb_velocity_remap_enabled=False`, `pole_axis_bbsw_enabled=False`, `axis_contact_guard_enabled=False`, `mass_floor_absorb_enabled=False`, `interior_patch_remap_enabled=False`. The historical `TENRYU_I1B_OPTIONB_COHERENT`, `TENRYU_I1B_OPTIONB_VELREMAP`, `TENRYU_I1B_POLE_AXIS_BBSW`, `TENRYU_I1B_AXIS_CONTACT_GUARD`, `TENRYU_I1B_MASS_FLOOR_ABSORB`, and `TENRYU_I1B_INTERIOR_PATCH_REMAP` envs override when SET; frozen-config emits the block only when enabled.
		- ALE pole-sector rezone defaults: `pole_sector_rezone_enabled=False`, `m_theta=4` (validation `>= 2`), `lambda=0.5` (validation `(0,1]`), `mode="uniform"` (allowed `{"uniform","equal_mu"}`), `deadband_frac=0.0` (validation `[0,1)`; 0 = no deadband, matching the historical env-unset behavior). The historical `TENRYU_I1B_POLE_REZONE*` environment variables override the namelist when SET non-empty; frozen-config emits the block only when enabled or non-default. Default-off and empirically NET-NEGATIVE at nr16 Case B (commit 5df5635e) — an available robustness lever, not a recommendation.

**Laser 拡張吸収物理（2026-07-30 追加；§6.4.6 / NUMERICS §5.4.5）**：
- ib：zeff_model=`"auto"`, species=`[]`, coulomb_log_model=`"debye"`, langdon_model=`"auto"`
- ra：enable=`False`, chi_p=`0.5`, c_ra=`1.0`
- raytrace 適応ステップ（2026-07-31 追加分）：ds_adapt_theta_target=`0.04`（rad/step 転回弧角度制御、`<=0` 無効、縮小可・床 1e-4。§6.4 / NUMERICS §5.3.2。既存 g/τ/max_factor 既定は不変）
- 導入時（2026-07-30）は全既定 OFF で、生産経路は導入前とビット同一（GXII 二重バイナリ比較で 235/235 共通データセット byte 一致を確認）。2026-08-10 以降は `zeff_model`/`langdon_model` の既定が `"auto"`（構築時解決）となり、1D レイトレース構成では Langdon が既定有効（GXII 回帰は全指標が許容帯内で PASS、吸収エネルギー −3.2%）。有効化は v1 では 1D_SPH 限定。
		- ALE velocity-coherence diagnostic defaults: `Numerics.diagnostics.ale_velcoherence.enabled=False`; `every_n_steps=1`; env `TENRYU_I1B_DISC_ALE_VELCOHERENCE=1` is an equivalent runtime enable. Frozen-config serialization emits this block only when enabled or when `every_n_steps != 1`, preserving default-off byte identity. No HDF5 schema change.
		- ALE gradient-alignment Stage 0 diagnostic defaults: `Numerics.ale.align_diagnostics={"enabled":False,"every_n_steps":0,"c_q_threshold":0.2,"w_rho":1.0,"w_p":1.0,"floor_rel":1.0e-12}`. Frozen-config serialization emits the block only when enabled or non-default, preserving default-off byte identity. Output is log-only; no HDF5 or run_info schema change.
		- Pole angular coarsen/motion pilot defaults: no namelist state; env `TENRYU_I1B_POLE_COARSEN_PILOT` unset disables the path-quotient pilot, env `TENRYU_I1B_POLE_MOTION_PILOT` unset disables the coherent mesh-position-velocity pilot, and env `TENRYU_I1B_PATH_PREDICATE_HARDEN` unset disables the stricter Stage-0 predicate path. When either pilot is enabled, env defaults are `TENRYU_I1B_POLE_COARSEN_LEVEL_MAX=3`, `TENRYU_I1B_POLE_COARSEN_Q_MIN=7`, and `TENRYU_I1B_POLE_COARSEN_Q_MAX=7`; the motion taper defaults are `TENRYU_I1B_POLE_MOTION_TRANSITION_ROWS=4` and `TENRYU_I1B_POLE_MOTION_PROFILE=smoothstep`. No frozen-config, checkpoint, or HDF5 schema change.
		- Polar-shell angular de-refinement default: no namelist state; env `TENRYU_I1B_POLAR_SHELL_ANGULAR_DEREFINE` unset disables the Stage-1 dynamic dyadic one-row polar-shell angular overlay and leaves OFF runs byte-identical. When SET non-empty/non-zero it also enables the Stage-0 hardened path predicate behavior; enabled overlays grow spans monotonically after accepted geometry updates using the NUMERICS §3.3.5 criterion. No frozen-config, checkpoint, or HDF5 schema change.
		- Tier-A butterfly-center authority harness default: no runtime/default namelist state. It is available only through the opt-in CTest target `test_butterfly_authority_tier_a`; normal single-block, three-block, five-block hydro, and tri_fan runs do not enter it.
	- hydro I1-B tri_fan center-stability defaults: av_qcap_over_p=0.0, av_qcap_center_band_only=False, tri_fan_center_cfl_enabled=False, tri_fan_center_cfl_safety=0.5, tri_fan_center_cfl_band_radial_index=3, tri_fan_center_perturbation_diag_enabled=False, av_qcap_scope="global", center_cfl_scope="disabled", center_perturbation_diag_scope="disabled", center_perturbation_diag_radial_bins=2
	- `Numerics.ale.corner_cell_aspect_protection_enabled=True`, `Numerics.ale.corner_cell_aspect_eta=0.5`
	- `Numerics.ale.swept_volume_sign_fixed = True`; legacy behavior requires `Numerics.profile.legacy_regression@2026-07-27` or an explicit False.
	- Wave-1 P0 additive gate defaults: `Numerics.ale.dgcl_commit_gate=False`, `Numerics.ale.dgcl_commit_rtol=1.0e-11`.
	- `Numerics.ale.transaction_failure_inject_point = 0`.
- ale1d：enabled=False, every_n_steps=100, min_steps_between_ale=50, benefit_min_dt_gain=1.5, candidate_dt_penalty_max=1.25, min_cells=256, protected_fraction_max=0.25, min_movable_segment_warn=24, min_movable_segment_hard=8, max_node_displacement_fraction_mu=0.35, max_node_displacement_fraction_r=0.35, diagnostics_enabled=True。Sensor defaults: laser target=0.060N sigma=4..16 cells peak=0.35 conf=0.10..0.40; ablation target=0.080N sigma=3..14 peak=0.40 ρ_ref=1.05 ρ_gate=0.07±0.02 Te_gate=0.5..2.0eV conf=0.10..0.35; shock target=0.040N sigma=2..8 peak=0.35 Q_conf=0.03..0.10 χ_conf=0.03..0.15; interface target=0.033N total cap=0.067N max_features=8 separation=4 jump=0.05..0.25 sigma=2..4 pin=True; center target=0.053N sigma=6..20 search_x=0.12. Rezone defaults: W0=1, Wmax=50W0, smoothing=2, floor fraction=0.55, Gaussian truncation=3σ, spatial monitor enabled with target=0.067N, p=2, laser/ablation/shock spatial Δr clips of 2.5e-5..2.0e-4 / 1.5e-5..1.2e-4 / 1.0e-5..8.0e-5 cm. Remap defaults: reject multicell sweeps=True, high_order_enabled=True, limiter_theta=1.5, high_order_ramp_cells=2, radiation_high_order_ramp_cells=2, fallback_to_first_order_on_bounds_fail=True, reject_strict_zero_flux_on_moving_protected_face=True. min_width_floor defaults (wave-5, experimental-incomplete): enabled=False, floor_cm=0.0, target_factor=1.25, relief_halfwidth_cells=3, max_growth_factor=1.8, retrigger_cooldown_steps=0 (validation: floor_cm>0, target_factor>1, relief_halfwidth_cells>=1, 1<max_growth_factor<=2, retrigger_cooldown_steps>=0 when enabled).
- Diagnostics：`per_operator_radial_fourier_enabled=False`, `radial_fourier_window_t_start_s=1.35e-5`, `radial_fourier_window_t_end_s=1.70e-5`, `radial_fourier_max_mode=-1`, `per_operator_radial_fourier_complex_enabled=False`, `per_operator_radial_fourier_complex_m_targets=[14,15,16]`, `per_operator_radial_fourier_complex_j_targets=[507,508,509,510,511]`。The HDF5 groups `/diagnostics/radial_fourier_audit/v1/` and `/diagnostics/radial_fourier_audit_v2/v1/` are additive and omitted unless the corresponding diagnostic is enabled. CMake `TENRYU_RFA_V2_MODE` defaults to `FULL`; non-`FULL` modes are verification/debug builds and do not change namelist defaults.
- Radiation.multigroup_diffusion Round 13 fields: `state_supply_boundary_policy="local_D_current"` (valid values: `"local_D_current"`, `"harmonic_ghost_D_test"`, `"radial_mean_D_test"`), `diagnostic_radial_fourier_substage_enabled=False`, `cg_inner_tol=1.0e-10` (validation `> 0`), `cg_max_iter=500` (validation `>= 1`). `/diagnostics/fld_substage_audit/v1/` is additive and omitted unless `diagnostic_radial_fourier_substage_enabled=True`; absence is the default reader state, so no existing-reader migration is required.
- schema change log：ALE-FIX-1 で旧 1D ALE namelist と `/diagnostics/ale_center/` を削除。V3 で新しい `Numerics.ale1d` を追加し、既定は無効。Week 6 で checkpoint `time_state/ale_last_applied_step` を追加し、旧 checkpoint は -1 で補完するため migration は不要。T23 で `Numerics.ale.ke_conservation_closure` を追加し、T29 で `Numerics.ale.ke_conservation_closure_audit` を追加した。どちらも既定 False を frozen-config schema 8 migration で補完する。T31 で `Numerics.ale.ke_closure_redistribute_floor` を追加し、既定 False を frozen-config schema 9 migration で補完する。T34 で history `mesh/ale_rezone_invocations` を追加した。A1 で `Numerics.ale.debug_per_remap_log` を追加し、既定 False を legacy frozen-config default 補完で扱う。Phase 2d adds `Numerics.ale.predictive_acceptance_*` fields; they are default-off and are added to legacy frozen-config default completion with no HDF5 schema change. Phase 2d volume-rate CFL adds `Numerics.hydro.volume_rate_cfl_*`; frozen-config schema 10 fills the default-off values for legacy checkpoint comparison, with no HDF5 schema change. Phase 2d-extension trial-volume CFL adds `Numerics.hydro.trial_volume_cfl_*`; frozen-config schema 11 fills the default-off values for legacy checkpoint comparison, with no HDF5 schema change. Phase 2d-extension v2 corner-J ALE trigger adds `Numerics.hydro.corner_jacobian_*`; frozen-config schema 12 fills the default-off values for legacy checkpoint comparison, with no HDF5 schema change. Phase 2d-extension v5 active mesh repair adds `Numerics.hydro.driver_retry_active_mesh_repair_enabled` and `Numerics.hydro.driver_retry_corner_balance_threshold`; frozen-config schema 13 fills the default-off values for legacy checkpoint comparison, with no HDF5 schema change. Phase 2d-extension v5 Wave 3 adds `Numerics.ale.corner_jacobian_post_tangle_enabled`; frozen-config schema 14 fills the default-on value for legacy checkpoint comparison, with no HDF5 schema change. Wave 3 Step 3.4 adds `Numerics.ale.local_boundary_repair_enabled`; frozen-config schema 15 fills the default-off value for legacy checkpoint comparison, with no HDF5 schema change. Wave 5 adds `Numerics.ale.multi_node_boundary_repair_enabled`; frozen-config schema 16 fills the default-off value for legacy checkpoint comparison, with no HDF5 schema change. Wave 7 adds `Numerics.ale.emergency_cell_deactivation_enabled`; frozen-config schema 17 fills the default-off value for legacy checkpoint comparison, with no HDF5 schema change. Phase 2 Mesh Stability Wave 1 adds `Numerics.hydro.mesh_geometry_soft_fail_enabled`; legacy frozen-config default completion fills the default False value, with no HDF5 schema change. Phase 2 Mesh Stability Wave 2 adds `Numerics.hydro.in_hydro_corner_j_guard_enabled`; legacy frozen-config default completion fills the default False value, with no HDF5 schema change. Phase 2 Mesh Stability Wave 3 adds `Numerics.hydro.regime_aware_corner_j_guard_enabled`, `Numerics.hydro.axis_margin_guard_enabled`, and `Numerics.hydro.axis_guard_band_cells`; legacy frozen-config default completion fills False, False, and 2 respectively, with no HDF5 schema change. Phase 2 Mesh Stability Wave 4 adds `Numerics.ale.multi_node_interior_repair_enabled` and `Numerics.hydro.driver_retry_use_suggested_dt_enabled`; legacy frozen-config default completion fills both default False values, with no HDF5 schema change. Phase 2 Mesh Stability Wave 5 adds default-off `Numerics.diagnostics.mesh_attribution.*` and the separate JSONL stream `mesh_failure_attribution.jsonl`; legacy frozen-config default completion fills the default values, with no HDF5 schema or dt_lineage format change. Stage 22 Wave 2 adds `Numerics.dt.floor_stall_max_consecutive_steps` (int, default 0, validation >=0); legacy frozen-config default completion fills 0, with no HDF5 schema change. Stage 23 Wave 1 adds `Numerics.ale.axis_variational_projection_enabled`; frozen-config schema V18 fills the default False value for legacy V17 checkpoints, with no HDF5 schema change. Stage 24 Wave 1 adds `Numerics.hydro.dispatcher_state_sensitive_bypass_enabled` (bool, default False) and `Numerics.hydro.dispatcher_state_sensitive_repair_cap_per_step` (int, default 3, validation >=1); legacy frozen-config default completion fills both default values, with no HDF5 schema change. Stage 24 Wave 2 adds 7 new namelist keys under [Numerics.ale]: axis_band_managed_remap_enabled (bool, default False), axis_band_managed_remap_width (int, default 3, validation >= 1), axis_band_managed_remap_max_width (int, default 6, validation >= width), axis_band_managed_remap_every_hydro_half_step (bool, default True), axis_band_managed_remap_margin_trigger (double, default 1e-4, validation > 0), axis_band_managed_remap_equal_volume (bool, default True), axis_band_managed_remap_include_radiation_groups (bool, default True); legacy frozen-config default completion fills all defaults, with no HDF5 schema change. Stage 27 Wave B3 extends `Numerics.profile.icf_standard_ale.forbidden_when_enabled` with `hydro_driver_retry_active_mesh_repair_enabled_forbidden_value=True`; no HDF5 schema change. Stage 27 Wave C adds 5 new keys under [Numerics.diagnostics] (icf.enabled, icf.rho_inner_threshold_g_per_cc, icf.rho_outer_threshold_g_per_cc, conservation.enabled, ale_provenance_emission.enabled); legacy frozen-config default completion fills False/0/0 with no HDF5 schema change. Stage 27 Wave D adds `Numerics.profile.icf_standard_ale.claim_level` (string, default `"characterization"`, one of `{characterization, pre_plic_smoke, production_comparable}`); legacy frozen-config V19 default completion fills `"characterization"` with no HDF5 schema change. Q2.4 adds `Numerics.hydro.strategy_first_retry_enabled` (bool, default False) and `Numerics.hydro.strategy_first_max_same_dt_attempts` (int, default 2, validation >=0); legacy frozen-config default completion fills both defaults, with no HDF5 schema change. Phase A adds default-off `Numerics.diagnostics.mesh_degeneracy_forensics.*` and the separate JSONL stream `mesh_degeneracy_forensics.jsonl`; legacy frozen-config default completion fills the defaults, with no HDF5 schema change. Phase D-1 adds default-off `Numerics.diagnostics.mesh_degeneracy_forensics.corner_j_source_budget_*`; legacy frozen-config default completion fills False, with no HDF5 schema change. AI-review-1D wave (2026-07-26) adds `Numerics.hydro.crossing_dt_safety` (float, default 0.5), `Numerics.hydro.time_integrator` (string, default "legacy_pc"), `Numerics.hydro.adaptive_av.hysteresis_tau` (float, default 0.0), and `Mesh.grid.grading.mapping` (string, default "legacy_estimated_radius"); legacy frozen-config default completion fills all four defaults, with no HDF5 schema change. 旧historyで不在の場合は unknown として扱い、既存readerのmigrationは不要。Laser march wave (2026-07-31) adds `Laser.raytrace.ds_adapt_theta_target` (float, default 0.04, `<=0` disables); legacy frozen-config default completion fills the default, with no HDF5 schema change. AV wave (2026-08-03): raw 1D_SPH decks omitting `Numerics.hydro.av_type` now resolve to "csw" in the builder (struct default stays "vnr"; frozen configs always emit av_type explicitly, so legacy frozen-config replay is unchanged; no HDF5 schema change). Perf wave-5 (2026-08-07) adds `Numerics.ale1d.min_width_floor` (nested dict: enabled bool default False, floor_cm double default 0.0, target_factor double default 1.25, relief_halfwidth_cells int default 3, max_growth_factor double default 1.8); legacy frozen-config default completion fills the defaults, no HDF5 schema change. Perf wave-5 also flips `Laser.raytrace_skip_config.enabled` default True→False (the old True default was universally inert because the pre-fix crit guard vetoed every call on overdense targets — behavior-compatible) and changes the crit guard semantics to a cache-relative band-crossing test (NUMERICS §5.9.4); frozen configs emit the fields explicitly so legacy replay is unchanged, no HDF5 schema change. Perf wave-10 (2026-08-10) adds `Numerics.ale1d.min_width_floor.retrigger_cooldown_steps` (int, default 0, validation >=0): after a floor-triggered rezone attempt that is not applied (any reason), the floor-trigger evaluation is skipped for that many steps, and any applied rezone resets the cooldown; legacy frozen-config default completion fills 0, the in-memory cooldown counter is not checkpointed (a restart re-evaluates the floor once and re-arms on rejection), no HDF5 schema change.
Langdon default-on wave (2026-08-10): `Laser.ib.langdon_model` gains `"auto"` (new default; resolves to `"legacy_vacuum_map"` for enabled 1D_SPH raytrace configs whose beams share one gaussian/super_gaussian/flat_top profile, else `"off"`), the species requirement narrows to `zeff_model="sequential_strip"` only, and the flat_top vacuum-map radius is corrected to w0; frozen configs emit the resolved value, so legacy frozen replay is unchanged.
- add Numerics.diagnostics.r_momentum_source_audit (log-only F-06 source-balanced R-momentum audit; default false, bit-neutral)
- add Numerics.hydro.corner_mass_convention (KINEMATIC_BASIS_RZ_V1 implementation behind a knob; default bbsw_radial_v0, bit-neutral)
- Wave 2D stage 2 adds the exact P1 triangle-degenerate corner-mass closed form w_k=(r_k+Σr)/(4Σr) under kinematic_basis_rz_v1.
- G4 epoch: corner_mass_convention default bbsw_radial_v0 -> kinematic_basis_rz_v1 (golden epoch; legacy frozen configs pin to bbsw_radial_v0).
- add Numerics.hydro.time_integration and Numerics.hydro.total_energy_identity_check (Wave 2E stage 1 F-08; defaults pc_v0/false; legacy frozen-config default completion; no HDF5 schema change)
- G5 epoch: time_integration default pc_v0 -> midpoint_v1 (golden epoch; legacy frozen configs pin to pc_v0).
  AI-review Wave 2A-1 (2026-07-27) adds Numerics.profile.legacy_regression.enabled (bool, default False) and Numerics.profile.legacy_regression.revision (string, default "2026-07-27", known values {"2026-07-27"}); legacy frozen-config default completion fills both defaults, with no HDF5 schema change. AI-review Wave 2A-2 (2026-07-27) flips the default of Numerics.ale.swept_volume_sign_fixed to True (corrected oriented donor convention, golden epoch G1); legacy behavior requires Numerics.profile.legacy_regression@2026-07-27 or an explicit False, and restart files enforce their recorded resolved swept-volume contract (metadata/swept_volume_contract).
- AI-review Wave-1 P0 additive gate adds `Numerics.ale.dgcl_commit_gate` (bool, default False) and `Numerics.ale.dgcl_commit_rtol` (double, default 1.0e-11); legacy frozen-config default completion fills both defaults, and frozen-config emits them only when enabled or non-default. No HDF5 schema change.
- AI-review Wave-1 P0 renames the production-audit log labels and internal identifiers from `gcl` to `vol_closure`; log labels only; HDF5 unchanged.
- Stage 30 Wave E adds `Numerics.plic.rho_material_aware_donor` (bool, default False); legacy frozen-config default completion fills False, with no HDF5 schema change.
- **Burn**（核燃焼 1D v1、2026-07-10）：enabled=False, fuels=["DT","DD"], scheme="fraley", partition="li_petrasso", fuel_materials=["DT"], x_D=0.5, x_T=0.5, x_He3=0.0, T_floor_keV=0.2, explicit_source_limit=0.2, eps_deplete=0.1, subcycle_max=64, vf_threshold=1e-3, neutron_heating=False, neutron_heating_n_mu=16（v2-E 2026-07-14）。frozen-config には `burn` block が常時 emit される（additive；旧 config との差分は本 block のみ — W5 A/B で field/history bit 恒等を確認済み）。HDF5 追加は §6.4.11 参照（すべて additive、schema version 不変）。2D_RZ は scheme 明示必須（既定 "fraley" は 2D で拒否）
- T3 adds default-off `Numerics.ale.axis_rezone_*` keys for the 5-block full-axis target-only ALE rezone; legacy frozen-config default completion fills False/0.1/0.1/1e-2 with no HDF5 schema change.
- I1-B S1 adds default-off `Numerics.ale.core_freeze_*` keys for gas-tracer-gated pure-Lagrangian ALE core freeze. S1b adds `core_freeze_skip_velocity_projection=True`, so enabled core-freeze also preserves frozen-node Lagrangian velocity through the CSR projection unless the A/B diagnostic flag is set False. The fields are serialized only when enabled; there is no HDF5 schema change and no migration requirement for default-off frozen configs.
- I1-B PR2-S0 bumps frozen-config schema V24 -> V25 and adds default-off diagnostic key `Numerics.ale.ale_preserve_lagrangian_velocity_carry` for the 2D_RZ multiblock CSR ALE carried-velocity discriminator. The field is serialized only when true, so default-off frozen configs still omit it; there is no HDF5 schema change.
- I1-B PR1 bumps frozen-config schema V23 -> V24 and adds default-off diagnostic keys `Numerics.ale.ale_identity_mode` and `Numerics.ale.ale_mover_diag` for the 2D_RZ ALE Lagrangian-identity limit and mover-coupling JSONL diagnostics. The fields are serialized only when true, so default-off frozen configs still omit them; there is no HDF5 schema change.
- I1-B diagnostic adds default-off `Numerics.diagnostics.ale_velcoherence.{enabled,every_n_steps}` and env gate `TENRYU_I1B_DISC_ALE_VELCOHERENCE=1` for per-sub-operation multiblock CSR ALE velocity-coherence logging. Legacy frozen-config default completion fills `False` and `1`; the block is serialized only when enabled or non-default cadence is requested. No HDF5 schema change.
- ALE gradient-alignment Stage 0 adds default-off `Numerics.ale.align_diagnostics.{enabled,every_n_steps,c_q_threshold,w_rho,w_p,floor_rel}`. Legacy frozen-config default completion fills `False,0,0.2,1.0,1.0,1.0e-12`; the block is serialized only when enabled or non-default. No run_info or HDF5 schema change.
- I1-B Ring7 scaffold adds `Numerics.hydro.ring7_quotient_enabled` (bool, default False), env gate `TENRYU_I1B_RING7_QUOTIENT`, diagnostic env `TENRYU_I1B_RING7_QUOTIENT_DIAG` / `_EVERY`, and env-only seam/pole transaction tunables documented in §6.4, including the path-margin trigger `TENRYU_I1B_RING7_PATH_MARGIN_TRIGGER` and cooldown `TENRYU_I1B_RING7_REZONE_COOLDOWN_STEPS`. Increment 3b adds the `[ring7_seam_remap]` runtime log for successful/rejected dedicated packet commits; Increment 4b adds `[ring7_pole_cap]` production-limiter option logs and `[ring7_pole_cap_remap]` conservative pole-cap commit/validation logs. No checkpoint/history/HDF5 datasets or migration are added. Legacy frozen-config default completion fills False.
- Tier-A butterfly-center authority harness adds no namelist key, no frozen-config field, no checkpoint/history dataset, and no migration. It is a ctest-only geometry authority emitter.
- Phase C adds `Numerics.hydro.geometric_retry_stagnation.*` (default-off fail-fast retry diagnostic); legacy frozen-config default completion fills the defaults, with no HDF5 schema change.
- I1B-GEO adds default-off `Numerics.diagnostics.mesh_quality_min.enabled` and optional history group `/diagnostics/mesh_quality_min/v1/`; legacy frozen-config default completion fills `False`. The group is additive/backward-compatible, old readers can ignore absence/presence, and no migration is required.
- I1-B Tier 2 Phase A adds no namelist keys and no HDF5 root schema change. It additively extends existing hotspot/areal-density diagnostics with optional snapshot/history datasets: `/diagnostics/hotspot_gas/v1/hotspot_Te_*`, 2T-only `hotspot_Ti_*`, `hotspot_energy_*`, `hotspot_work_proxy_*`, snapshot `/diagnostics/areal_density/v1/rhoR_hotspot_tracer`, and history `implosion/rho_R_hotspot_tracer` plus per-angle scalar aliases. Absence means the diagnostic was disabled or the gas tracer was unavailable; old readers may ignore presence.
- Phase 2d-extension v6 Wave 3 adds `Numerics.hydro.cascade_on_hydro_retry_enabled` (bool, default False); legacy frozen-config default completion fills False, with no HDF5 schema change.
- AI #7 diagnostic adds `Numerics.ale.force_rezone_every_n_steps` (int, default 0, validation >=0); legacy frozen-config default completion fills 0, with no HDF5 schema change.
- Phase 2d-extension v6 Wave 4 adds `Numerics.ale.corner_post_tangle_strict_floor_enabled` (bool, default False); legacy frozen-config default completion fills False, with no HDF5 schema change.
- I1-B TRI-fan center-stability stack adds `Numerics.hydro.av_qcap_over_p`, `av_qcap_center_band_only`, `tri_fan_center_cfl_enabled`, `tri_fan_center_cfl_safety`, `tri_fan_center_cfl_band_radial_index`, and `tri_fan_center_perturbation_diag_enabled`; legacy frozen-config default completion fills `0.0`, `False`, `False`, `0.5`, `3`, and `False`. `/diagnostics/dt_breakdown_history/tri_fan_center_cfl/` and `/diagnostics/tri_fan_center_perturbation/v1/` are additive optional groups; no existing-reader migration is required.
- I1-B S2 T1 adds topology-aware scope keys `Numerics.hydro.av_qcap_scope`, `center_cfl_scope`, `center_perturbation_diag_scope`, and `center_perturbation_diag_radial_bins`; defaults are `"global"`, `"disabled"`, `"disabled"`, and `2`. Default values are elided from frozen config for byte-identical single-block decks; no HDF5 schema change is required.
- S4-T1-next T5b adds `Numerics.ale.multiblock_cross_seam_rezone_enabled` (bool, default False); legacy frozen-config default completion fills `False`, preserving G1-G4 static-seam behavior unless the deck opts in. No HDF5 schema change.
- Phase 1 scaled γ-MVP target adds `Numerics.ale.multiblock_scaled_reference_enabled` (bool, default False); legacy frozen-config default completion fills `False`, preserving static IC reference targets unless the deck opts in. No HDF5 schema change.
- T7 differential converging reference adds default-off `Numerics.ale.multiblock_differential_reference_*` keys for the multiblock 2D_RZ Lagrangian-close conservative-remap reference; legacy frozen-config default completion fills `False`, `64`, `0.03`, `0.10`, `0.03`, `1e-3`, `1e-9`, and `0.5`. The fields are emitted only when enabled or non-default, and there is no HDF5 schema change.
- CP1 adds default-off `Numerics.ale.multiblock_lagrangian_bulk_center_patch_reference_enabled` and `multiblock_center_patch_*` keys for the multiblock Lagrangian-bulk center/quality-patch reference; legacy frozen-config default completion fills `False`, `4`, `0.0`, `2`, `0.05`, `0.10`, `0.03`, `0.08`, `0.03`, and `0.08`. The fields are emitted only when enabled or non-default, and there is no HDF5 schema change.
- R3 Exp1 adds default-off `Numerics.ale.central_pseudo_core_enabled` and `central_pseudo_core_s_c` for the virtual central pseudo-core diagnostic; membership is a ring-conforming fixed cell set with no halo. Legacy frozen-config default completion fills `False` and `0.0`; the fields are emitted only when enabled or non-default, and there is no HDF5 schema change.
- Phase 2 multiblock path-admissibility dt rejection adds `Numerics.ale.multiblock_path_admissibility_enabled` (bool, default False), `path_admissibility_floor` (double, default 0.01), `dt_rejection_factor` (double, default 0.5), and `max_dt_rejections` (int, default 8); legacy frozen-config default completion fills these values. No HDF5 schema change.
- A-09: relative-J admissibility floors rebased to reference-mesh J when enabled (default-off; no default-path change)
- Stage 32a Wave A adds `Numerics.materials.*` per-material conservation controls and V22 additive `/hydro/per_material/v1/` skeleton. HDF5 `kSchemaVersion` remains 1; frozen-config schema V21 fills default-disabled values for legacy checkpoint comparison.
- PR3 Qei temperature alignment adds `Numerics.hydro.qei_evaluate_at_t_n` (bool, default True); frozen-config schema V22 fills the default for legacy checkpoint comparison, with no HDF5 schema change.
- I5 scaled A-prime support adds `Numerics.hydro.qei_multiplier` (float, default 1.0, validation `>0`); frozen-config schema V22 fills the default for legacy checkpoint comparison, with no HDF5 schema change.
- I1 Phase 1 total-energy remap adds default-off `Numerics.hydro.total_energy_remap_2d_rz`; legacy frozen-config default completion fills `False`, with no HDF5 schema change. T4 extends this existing flag to the multiblock trifan-cap CSR conservative-remap path; it adds no namelist key and no HDF5 schema field.
- I1 Fix B discriminator adds default-off diagnostic `Numerics.hydro.work_split_audit_2d_rz`, `work_split_audit_cell_every_n_steps`, and `work_split_audit_all_rows`; legacy frozen-config default completion fills `False`, `0`, and `False`, with no HDF5 schema change.
- I1-A Fix A z-HLLC adds default-off `Numerics.hydro.hllc_z_flux_2d_rz`, `hllc_z_flux_audit_2d_rz`, `hllc_z_flux_hlle_fallback`, and `hllc_z_flux_strict_quasi_1d`; legacy frozen-config default completion fills `False`, `False`, `True`, and `False`. Optional checkpoint dataset `/hydro/hllc_mom_z_cell` is backward-compatible; old checkpoints reconstruct the field from nodal velocity on first HLLC step. This supports the `I1A_2D_RZ_FLD_CED_PLANAR_Z_SHOCK_HLLC` code-verification row only; it does not change the global default hydro mode.
- Stage A-F Plan T1 adds explicit `Numerics.hydro.av_model`, `subzonal_pressure_enabled`, and subzonal merit constants; legacy frozen-config default completion fills `"scalar_vnr_legacy"`, `False`, `"caramana_auto"`, `1.4142135623730951`, `0.1`, `2`, and `1.0`. No HDF5 schema change.
- I1-B Stage B S1 T7 adds additive `/mesh/topology/v3/` for `MULTIBLOCK_HALF_BUTTERFLY_5BLOCK` checkpoints. v1 single_block (no topology group) and v2 3-block `/mesh/topology/v2/` remain readable and writer-compatible; readers prefer v3, then v2, then v1. HDF5 root `schema_version` remains 1 because the schema is path-versioned and backward-compatible.
- I1-B Stage B S2 retargets existing multiblock hydro/ALE diagnostics and the five-block smoke to `MULTIBLOCK_HALF_BUTTERFLY_5BLOCK`; it adds no namelist keys, frozen-config fields, HDF5 groups, checkpoint fields, or migration requirement.
- I1-A P-metric hardening changes only the validation harness and ctest gate: no namelist, checkpoint, or HDF5 schema change. The pass/fail metric is now the fixed-scale precursor amplitude/area/shape plus independent matter-shock-thickness gate documented in NUMERICS §6.7.3; legacy shock-windowed L2 and convolved peak gap remain diagnostic-only. The reference-table t0-admissibility contract and `TENRYU_I1_2D_RZ_VERBOSE_DIAG` lean-diagnostics environment gate are harness/deck diagnostics only and add no namelist, checkpoint, or HDF5 schema field.
- I1-B polar pole diagnostic adds only the environment gate `TENRYU_I1B_POLAR_POLE_DIAG=1` and JSONL files under `tmp/diagnostics/`; it adds no namelist key, checkpoint field, HDF5 group, or restart migration.
- I1-B PAB polar button deck knob adds only `TENRYU_I1B_PAB_POLAR_CENTER_BUTTON_OUTER_NODE_RING` (default `2`) for forwarding to existing `Mesh.center_button_outer_node_ring` when the deck selects `polar_center_treatment="button"`. Button snapshots add compatible `/hydro_flags/cell_is_void` and `/mesh/topology/v1/cell_nverts` datasets; non-button deck output is unchanged.
- I1-B pole angular coarsen/motion pilots add only default-off environment gates `TENRYU_I1B_POLE_COARSEN_PILOT`, `TENRYU_I1B_POLE_MOTION_PILOT`, `TENRYU_I1B_PATH_PREDICATE_HARDEN`, `TENRYU_I1B_POLE_COARSEN_LEVEL_MAX`, `TENRYU_I1B_POLE_COARSEN_Q_MIN`, `TENRYU_I1B_POLE_COARSEN_Q_MAX`, `TENRYU_I1B_POLE_MOTION_TRANSITION_ROWS`, and `TENRYU_I1B_POLE_MOTION_PROFILE`; they add no namelist key, frozen-config field, checkpoint field, HDF5 group, or restart migration.
- 2026-05-11 `Numerics.ale.donor_sign_fixed` was renamed to `Numerics.ale.swept_volume_sign_fixed`; default `False` is filled by `apply_legacy_numerics_defaults` for backward-compatible checkpoint loading, and the old key remains a deprecated parsing/checkpoint alias. No HDF5 schema change.
- 2026-05-11 `safe_backtrack_enabled`, `safe_backtrack_min_exp`, and `safe_backtrack_binary_iters` added to `Numerics.ale`; defaults are filled by legacy frozen-config default completion. No HDF5 schema change.
	- 2026-05-11 `rezone_solver`, `rezone_local_admissibility_linesearch`, `rezone_local_j_floor_rel`, and `rezone_local_linesearch_max_halves` added to `Numerics.ale`; defaults select the legacy Winslow kernel and disable local line search, and are filled by legacy frozen-config default completion. No HDF5 schema change.
	- 2026-07-28 `m1_gamma_align`, `m1_lambda_tether`, `m1_theta_reg`, `m1_sweeps`, and `m1_barrier_beta` added to `Numerics.ale`; legacy frozen-config default completion fills their defaults, while serialization emits them only for `rezone_solver="m1_tmop"` (era-additive, following the pentagon precedent). No HDF5 schema change.
	- 2026-07-29 `m1_min_j_dec_rel` added to `Numerics.ale` (minimum relative J-decrease acceptance gate for M1 rezones); default revised 1e-4 -> 0.0 the same day after the knob-sweep distribution analysis showed a positive default suppresses the measured ctrl/implosion M1 mechanism (small-ΔJ acceptances) while its motivating belt evidence was adjudicated as chaos-band noise. Same era-additive serialization rule as the other M1 keys. No HDF5 schema change.
	- 2026-05-16 `corner_cell_aspect_protection_enabled` and `corner_cell_aspect_eta` added to `Numerics.ale`; legacy frozen-config default completion fills `True` and `0.5`. No HDF5 schema change.
	- 2026-05-11 `reject_zero_gauss_j` and `zero_gauss_j_floor_rel` added to `Numerics.ale`; defaults are filled by legacy frozen-config default completion. No HDF5 schema change.
- 2026-05-17 `conservative_remap_order` added to `Numerics.ale`; default `"first_order_donor"` preserves PR B behavior and legacy frozen-config default completion fills the key. No HDF5 schema change.
- 2026-05-17 PR E changes state-supply conservative-remap z-face boundary fluxes to use upwind donors and adds optional diagnostic-only `/diagnostics/corner_bc_audit/v1/` history rows when the CFL winner is in the r_outer/z_top corner halo. HDF5 root `schema_version` is unchanged.
- 2026-05-13 `reference_barrier_*` keys added to `Numerics.ale`; defaults keep the path disabled and are filled by legacy frozen-config default completion. No HDF5 schema change.
- 2026-05-13 B-prime `driver_retry_reference_barrier_*` keys added to `Numerics.ale`; defaults keep the retry primitive disabled and are filled by legacy frozen-config default completion. No HDF5 schema change.
- 2026-05-11 `Numerics.hydro.in_hydro_gauss_j_guard_enabled`, `Numerics.hydro.in_hydro_rz_volume_guard_enabled`, `Numerics.hydro.in_hydro_gauss_j_floor_rel`, and `Numerics.hydro.in_hydro_rz_volume_floor_rel` added for default-off in-hydro Gauss-J / RZ-volume candidate guards; legacy frozen-config default completion fills `False`, `False`, `1e-8`, and `1e-8`. No HDF5 schema change.
- 2026-05-11 `Numerics.hydro.mesh_quality_dt_cfl_enabled`, `mesh_quality_dt_safety_alpha`, `mesh_quality_dt_corner_j_enabled`, `mesh_quality_dt_gauss_j_enabled`, `mesh_quality_dt_rz_volume_enabled`, `mesh_quality_dt_axis_margin_additive`, `mesh_quality_dt_corner_j_floor_rel`, `mesh_quality_dt_gauss_j_floor_rel`, and `mesh_quality_dt_rz_volume_floor_rel` added for default-off pre-commit mesh-quality dt CFL; legacy frozen-config default completion fills `False`, `0.5`, `True`, `True`, `True`, `True`, `1e-8`, `1e-8`, and `1e-8`. No HDF5 schema change.
- 2026-05-16 `Numerics.hydro.rz_geometric_cfl_enabled`, `rz_geometric_cfl_etaV`, and `rz_geometric_cfl_r_floor` added for default-off predictive 2D_RZ geometric hydro CFL; legacy frozen-config default completion fills `False`, `0.5`, and `1e-10`. No HDF5 schema change.
- 2026-05-17 `Numerics.hydro.rz_geometric_cfl_cumulative_protection_enabled`, `rz_geometric_cfl_v_initial_floor`, and `rz_geometric_cfl_precise_u_half_enabled` added for 2D_RZ geometric hydro CFL initial-volume protection and opt-in force-predicted half-step velocity; legacy frozen-config default completion fills `True`, `0.1`, and `False`. No HDF5 schema change.
- 2026-05-11 `Numerics.hydro.axis_margin_additive_in_action8_enabled` added for default-off additive axis-margin evaluation in the in-hydro candidate guard; legacy frozen-config default completion fills `False`. No HDF5 schema change.
- 2026-05-11 `lambda_sweep_diagnostic_enabled`, `lambda_sweep_target_cell_c/i/j`, and `lambda_sweep_max_exp` added to `Numerics.ale`; defaults are filled by legacy frozen-config default completion. V23 additive optional sidecar `/diagnostics/ale_lambda_sweep/v1/` is written only when the diagnostic fires; root HDF5 `schema_version` remains 1.
- 2026-05-28 Phase 3 adds default-off `Numerics.hydro.subzonal_mass_enabled`, `subzonal_mass_lagrangian_invariant_enabled`, `anti_hourglass_kappa`, and `subzonal_pressure_mode` for multiblock Caramana-Shashkov subzonal/corner masses plus anti-hourglass force. Legacy frozen-config default completion fills `False`, `False`, `0.05`, and `"uniform_cell"`. No HDF5 schema change; `state.corner_mass`, `state.corner_volume`, and `state.subzonal_mass_corner{0,1,2,3}` remain runtime-only.
- 2026-06-03 S2-T1 adds default-off `Numerics.hydro.bbs_axis_policy_enabled` for the five-block \(R=0\) BBS RZ-compatible axis mass fallback. Legacy frozen-config default completion fills `False`. No HDF5 schema change.
- 2026-05-28 Phase 3 mesh-freeze diagnostics add `Numerics.debug.trace_mesh_motion`, `trace_mesh_node_selector`, `trace_mesh_cell`, and `trace_max_steps`; legacy frozen-config default completion fills `False`, `"outer_equator"`, `7`, and `5`. Trace output is stderr-only and diagnostic-only. No HDF5 schema change.
- 2026-05-12 `Numerics.hydro.hourglass.*` added for default-off 2D_RZ Caramana-Shashkov subzonal pressure anti-hourglass force; legacy frozen-config default completion fills `enabled=False`, `scale=0.05`, `compatible_work_enabled=True`, `activation_corner_j_ratio_threshold=0.5`, `activation_hourglass_amplitude_threshold=0.01`, `subzonal_pressure_model="linearized"`, and `max_force_per_node_fraction=0.2`. No HDF5 schema change; `state.subzonal_mass_corner{0,1,2,3}` are runtime state arrays.
- 2026-05-12 Phase D-3 adds default-off `Numerics.diagnostics.mesh_degeneracy_forensics.velocity_history_*` and the separate JSONL stream `mesh_degeneracy_velocity_history_<i>_<j>.jsonl`; legacy frozen-config default completion fills `False`, `-1`, `1`, `True`, and `5000`. No HDF5 schema change.
- Wave 0 Phase 2-kernel-production-level infrastructure (commits cc62bad5, f56f8612, 547d894c, f08646b8) bumps frozen-config schema V22 -> V23. V23 adds default-off `Numerics.diagnostics.production_audit.*` fields: `enabled=False`, `tier="none"`, `audit_json_path="<output_dir>/audit_summary.json"`, `escape_valve_budget.mass_max=0.0`, `escape_valve_budget.energy_max=0.0`, `region_of_interest=[]`, `gcl.enabled=False`, `positivity.enabled=False`, and `positivity.fatal_on_neg=False`.
- PR 4.10 adds `Numerics.diagnostics.dt_breakdown_history_enabled=True` and optional history groups `/diagnostics/dt_breakdown_history/`, `/diagnostics/cfl_winner/`, `/diagnostics/per_row_mass/`, `/diagnostics/av_max/`. Legacy frozen-config default completion fills True; checkpoint/read migration is unnecessary because the groups are diagnostics-only and readers ignore them when absent.
- I1B-GEO T2 adds default-off `Numerics.diagnostics.mesh_quality_min.enabled` and optional `/diagnostics/mesh_quality_min/v1/`. The group is emitted only when that dedicated flag is true, is additive, backward-compatible, and diagnostic-only; old readers ignore it when absent or unknown, so no migration is required.
- I1B-GEO T3 extends `/diagnostics/mesh_quality_min/v1/` with additive `achieved_min_edge_length_rel`, `achieved_min_altitude_rel`, and `achieved_max_condition_number` datasets under the existing default-off flag and cadence. Existing datasets keep their names and meanings; old readers can ignore the new datasets, so no migration is required.

**Wave 0 production-audit output schema**：
- `/diagnostics/escape_valve_audit/v1` HDF5 group（per-step fixed-column datasets; no variable-length maps, intentionally preserving HDF5 reader compatibility）:
  - `emergency_cell_deactivation_count: int[N_step]`
  - `multi_node_boundary_repair_count: int[N_step]`
  - `multi_node_interior_repair_count: int[N_step]`
  - `axis_variational_projection_count: int[N_step]`
  - `local_boundary_repair_count: int[N_step]`
  - `retry_active_mesh_repair_count: int[N_step]`
- `/diagnostics/positivity/v1` HDF5 group（per-step minima）:
  - `rho_min: float64[N_step]`
  - `p_min: float64[N_step]`
  - `Te_min: float64[N_step]`
  - `Ti_min: float64[N_step]`
  - `Er_min: float64[N_step]`
  - `kappa_min: float64[N_step]`
- `<output_dir>/escape_valve_events.jsonl`: one JSON object per firing with fields `cell_id`, `time`, `flag_name`, `reason`, `mass_delta`, `momentum_delta_r`, `momentum_delta_z`, `energy_delta`, `before_state{rho,p,Te,Ti}`, and `after_state{rho,p,Te,Ti}`.
- `<output_dir>/audit_summary.json`（produced by `tools/validation/audit_summary.py` postprocessor）:
  - `audit_status: str`（`"PASS"` | `"TIER_A_PASS"` | `"TIER_A_FAIL_ESCAPE_VALVE_FIRED"` | `"TIER_A_FAIL_RUN_ABORTED"` | `"TIER_A_FAIL_POSITIVITY_VIOLATION"` | `"TIER_A_FAIL_MISSING_DATA"`）
  - `termination_reason: str`（from existing `run_info.json`）
  - `conservation_residual_max: dict`（mass / R-momentum / Z-momentum / energy）
  - `positivity_min: dict`（rho / p / Te / Ti / Er / kappa）
  - `solver_stats: dict`（per-physics linear/nonlinear residuals + iterations）
  - `escape_valve_firings: dict`（6-flag counts）
  - `profile_l2_error: dict`（optional; per-field L2 error vs reference）
  - `symmetry_residual: dict`（optional; per-mode if applicable）
  - `gcl_residual: float|null`（Tier-A required when `production_audit.gcl.enabled=True`; max per-step dimensionless ALE volume-closure residual。`gcl_residual` は互換性のため保持する legacy HDF5/summary field name）
  - `reproducibility: dict`（same_arch_bitwise: string, cross_arch_status: string, cross_arch_metadata: dict）

**Materials**：
- eos.model in {`sesame`, `ionmix`, `tmat`, `ideal_gas`}（既定 `ideal_gas`）
- eos.hydro_backend in {`legacy`, `helmholtz_spline`, `helmholtz_jet`, `exact_ideal_gas`, `rho_e_table`, `mie_gruneisen`}（既定 `legacy`；tabular EOS の hydro-only backend。`exact_ideal_gas` と `rho_e_table` は 1D_SPH 専用、`mie_gruneisen` は 1D_SPH + 2T 専用）
- sesame settings（when `eos.model=sesame`）：sesame_format=ascii、sesame_table_total=301、sesame_table_electron=304
- opacity.model=ionmix（既定。推奨構成: SESAME EOS + IONMIX opacity）
- opacity.kappa_planck is unset by default (Planck constant follows `kappa_a`; Rosseland stays `kappa_a`)
- zbar：model=fixed（完全電離仮定。多材料時は材料別 Z̄_α = Z_α を混合平均。thomas_fermi, tabularも選択可）
- opacity_mix_rule=linear_mass, eos_mix_rule=mass_weighted_same_state

**Materials（TMAT-H5 追加）**：
- eos.model=tmat（`.tmat.h5` から EOS を読み込む。既存 `sesame` / `ionmix` / `ideal_gas` と選択）
- opacity.model=tmat（`.tmat.h5` から多群 opacity を読み込む。既存 `ionmix` / `sesame` / `constant` / `table_nlte` と選択）
- opacity.tmat_skip_lte_repair: bool（既定 False。False = non-LTE tables get the PE:=PA repair on flagged nodes；True = load raw PE、CRE A/B decks 用）

**Materials（Void）**：
- is_void=False（既定）。is_void=True 時：eos_model=ideal_gas, opacity_model=constant, kappa_a=0, kappa_s=0, A=1, Z=0

**Materials（power-law 解析モデル、2026-07-10 追加 — Hammer–Rosen gate 系）**：
- opacity.model=`power_law`（opt-in）: alpha_T=0, lambda_rho=0, T_ref_eV=1.0, rho_ref_g_cc=1.0（kappa0_cm2_g は必須）。grey 専用（群数 1 必須）
- eos.model=`power_law_te`（opt-in）: mu_rho=0, gamma_p=5/3（f_erg_g, beta は必須）。初期化時 tabulate → table-EOS 経路。optional softstep: step_D_erg_g_eV=0（>0 で step_Tc_eV>0 と step_w_eV>0 必須 — c_v 段差、beta_sec G-S2 判別器基盤 §6.4.3）
- Radiation.multigroup_diffusion.fleck_cv_source=`table`（既定、2026-07-11 フリップ — 外部AI裁定）。`legacy` は旧 golden bit 再現用の明示互換モード（§6.4.5）
- void_config：rho=1e-10 [g/cm³], Te=1e-3 [eV], Ti=1e-3 [eV]
- cell_is_void判定：非void材料の体積分率合計 ≤ 1e-12 でvoidセルと判定

**Geometry**：
- velocity=None（ゼロ速度）, radiation_field=equilibrium, radiation_field_Tr_eV=-1.0（planck 時のみ必須 >0）, enforce_sum_to_one=True

**Numerics / dt**：
- dt：initial_s=1e-15, max_s=1e-9, min_s=1e-20, growth_factor=1.2, cfl_hydro=0.3, cfl_cond=0.25, f_min_fleck=0.01, floor_stall_max_consecutive_steps=0（爆縮典型値。ユーザ調整必須。floor stall detector は 0 で無効）
- `Numerics.dt.floor_stall_max_consecutive_steps`: int, default 0, validation `>= 0`; 32 is the Stage 22 production-policy value decks may opt into.
- numerics persistent_loop：enabled=False（**既定 OFF・bit 恒等**）, chunk_steps=128
- debug：trace_mesh_motion=False, trace_mesh_node_selector=`"outer_equator"`, trace_mesh_cell=7, trace_max_steps=5
- hydro T_start_eV=0.0（常時有効；NUMERICS §2.1.1準拠）
- hydro corner_mass_convention = "kinematic_basis_rz_v1"
- hydro time_integration = "midpoint_v1", total_energy_identity_check = False
- hydro av_type="vnr", av_C1=0.1, av_C2=1.5, av_cfl_coefficient=0.25, axis_motion_floor_fraction=0.0, axis_margin_dt_floor_fraction=0.0, volume_rate_cfl_enabled=False, volume_rate_cfl_threshold=0.5, rz_geometric_cfl_enabled=False, rz_geometric_cfl_etaV=0.5, rz_geometric_cfl_r_floor=1e-10, rz_geometric_cfl_cumulative_protection_enabled=True, rz_geometric_cfl_v_initial_floor=0.1, rz_geometric_cfl_precise_u_half_enabled=False, trial_volume_cfl_enabled=False, trial_volume_cfl_floor_fraction=0.05, trial_volume_cfl_shrink_fraction=0.5, corner_jacobian_ale_trigger_enabled=False, corner_jacobian_floor_eps=1e-6, corner_jacobian_ale_trigger_scale=0.5, in_hydro_corner_j_guard_enabled=False, in_hydro_gauss_j_guard_enabled=False, in_hydro_rz_volume_guard_enabled=False, in_hydro_gauss_j_floor_rel=1e-8, in_hydro_rz_volume_floor_rel=1e-8, mesh_quality_dt_cfl_enabled=False, mesh_quality_dt_safety_alpha=0.5, mesh_quality_dt_corner_j_enabled=True, mesh_quality_dt_gauss_j_enabled=True, mesh_quality_dt_rz_volume_enabled=True, mesh_quality_dt_axis_margin_additive=True, mesh_quality_dt_corner_j_floor_rel=1e-8, mesh_quality_dt_gauss_j_floor_rel=1e-8, mesh_quality_dt_rz_volume_floor_rel=1e-8, ring7_quotient_enabled=False, regime_aware_corner_j_guard_enabled=False, axis_margin_guard_enabled=False, axis_margin_additive_in_action8_enabled=False, axis_guard_band_cells=2, driver_full_step_retry_enabled=False, driver_full_step_retry_max_attempts=3, driver_retry_active_mesh_repair_enabled=False, driver_retry_corner_balance_threshold=0.01, cascade_on_hydro_retry_enabled=False, driver_retry_use_suggested_dt_enabled=False, geometric_retry_stagnation={enabled=False, same_cell_count_threshold=3, sigma_rel_tol=0.25, dt_drop_factor=1e-4, force_diagnostic_dump=True}, dispatcher_state_sensitive_bypass_enabled=False, dispatcher_state_sensitive_repair_cap_per_step=3, strategy_first_retry_enabled=False, strategy_first_max_same_dt_attempts=2, mesh_geometry_soft_fail_enabled=False, qei_evaluate_at_t_n=True, qei_multiplier=1.0, total_energy_remap_2d_rz=False, work_split_audit_2d_rz=False, work_split_audit_cell_every_n_steps=0, work_split_audit_all_rows=False, hllc_z_flux_2d_rz=False, hllc_z_flux_audit_2d_rz=False, hllc_z_flux_hlle_fallback=True, hllc_z_flux_strict_quasi_1d=False, bbs_axis_policy_enabled=False, subzonal_mass_enabled=False, subzonal_mass_lagrangian_invariant_enabled=False, anti_hourglass_kappa=0.05, subzonal_pressure_mode="uniform_cell", hourglass.enabled=False, hourglass.scale=0.05, hourglass.compatible_work_enabled=True, hourglass.activation_corner_j_ratio_threshold=0.5, hourglass.activation_hourglass_amplitude_threshold=0.01, hourglass.subzonal_pressure_model="linearized", hourglass.max_force_per_node_fraction=0.2, csw_C1=0.5, csw_C2=2.0, csw_limiter="van_leer", csw_limiter_enabled=True, csw_shock_limiter_floor=0.65, csw_zero_uniform_compression=True, csw_diagnostics=False, av_limiter_J=1.0, av_heat_C=0.0, ion_art_heat_C=0.0, post_shock_heat=False, post_shock_heat_C=0.1, post_shock_heat_decay=3.0, post_shock_velocity_damping_C=0.0, bulk_viscosity_C=0.0, crossing_dt_safety=0.5, time_integrator="legacy_pc", adaptive_av.enabled=False, adaptive_av.hysteresis_tau=0.0, av_eos_aware=False, av_eos_gamma1_ref=5/3, av_eos_boost_max=3.0, odd_even_damping_C=0.0, anti_hourglass_C=0.0, ee_odd_even_C=0.0, hk_velocity_damper_C=0.0, hk_velocity_damper_tau_min=8.0, hk_velocity_damper_grad_Te_max=0.2, hk_velocity_damper_grad_rho_max=0.3, hk_velocity_damper_guard_cells=25, compatible_energy=False, rho_e_linear_grid=False, eos_writeback=True, eos_closure_mode="energy_authoritative", exact_override="none"（NUMERICS §3.1.4, §3.1.5, §3.1.6, §3.1.9, §3.2.9, §3.2.9b, §3.2.9c, §3.2.12a, §3.2.13, §3.2.13a, §3.2.13b準拠）
- hydro av_type="vnr", rz_momentum_scheme="volume_weighted", axis_node_mass_convention="corner_subzonal", av_C1=0.1, av_C2=1.5, av_cfl_coefficient=0.25, axis_motion_floor_fraction=0.0, axis_margin_dt_floor_fraction=0.0, volume_rate_cfl_enabled=False, volume_rate_cfl_threshold=0.5, corner_j_predict_cfl_enabled=False, corner_j_predict_cfl_safety=0.5, corner_j_predict_floor_frac=0.05, corner_j_predict_max_shrink=0.25, corner_j_predict_shell_rings=4, rz_geometric_cfl_enabled=False, rz_geometric_cfl_etaV=0.5, rz_geometric_cfl_r_floor=1e-10, rz_geometric_cfl_cumulative_protection_enabled=True, rz_geometric_cfl_v_initial_floor=0.1, rz_geometric_cfl_precise_u_half_enabled=False, trial_volume_cfl_enabled=False, trial_volume_cfl_floor_fraction=0.05, trial_volume_cfl_shrink_fraction=0.5, corner_jacobian_ale_trigger_enabled=False, corner_jacobian_floor_eps=1e-6, corner_jacobian_ale_trigger_scale=0.5, in_hydro_corner_j_guard_enabled=False, in_hydro_gauss_j_guard_enabled=False, in_hydro_rz_volume_guard_enabled=False, in_hydro_gauss_j_floor_rel=1e-8, in_hydro_rz_volume_floor_rel=1e-8, mesh_quality_dt_cfl_enabled=False, mesh_quality_dt_safety_alpha=0.5, mesh_quality_dt_corner_j_enabled=True, mesh_quality_dt_gauss_j_enabled=True, mesh_quality_dt_rz_volume_enabled=True, mesh_quality_dt_axis_margin_additive=True, mesh_quality_dt_corner_j_floor_rel=1e-8, mesh_quality_dt_gauss_j_floor_rel=1e-8, mesh_quality_dt_rz_volume_floor_rel=1e-8, ring7_quotient_enabled=False, regime_aware_corner_j_guard_enabled=False, axis_margin_guard_enabled=False, axis_margin_additive_in_action8_enabled=False, axis_guard_band_cells=2, driver_full_step_retry_enabled=False, driver_full_step_retry_max_attempts=3, driver_retry_active_mesh_repair_enabled=False, driver_retry_corner_balance_threshold=0.01, cascade_on_hydro_retry_enabled=False, driver_retry_use_suggested_dt_enabled=False, geometric_retry_stagnation={enabled=False, same_cell_count_threshold=3, sigma_rel_tol=0.25, dt_drop_factor=1e-4, force_diagnostic_dump=True}, dispatcher_state_sensitive_bypass_enabled=False, dispatcher_state_sensitive_repair_cap_per_step=3, strategy_first_retry_enabled=False, strategy_first_max_same_dt_attempts=2, mesh_geometry_soft_fail_enabled=False, qei_evaluate_at_t_n=True, qei_multiplier=1.0, total_energy_remap_2d_rz=False, work_split_audit_2d_rz=False, work_split_audit_cell_every_n_steps=0, work_split_audit_all_rows=False, hllc_z_flux_2d_rz=False, hllc_z_flux_audit_2d_rz=False, hllc_z_flux_hlle_fallback=True, hllc_z_flux_strict_quasi_1d=False, bbs_axis_policy_enabled=False, subzonal_mass_enabled=False, subzonal_mass_lagrangian_invariant_enabled=False, anti_hourglass_kappa=0.05, subzonal_pressure_mode="uniform_cell", hourglass.enabled=False, hourglass.scale=0.05, hourglass.compatible_work_enabled=True, hourglass.activation_corner_j_ratio_threshold=0.5, hourglass.activation_hourglass_amplitude_threshold=0.01, hourglass.subzonal_pressure_model="linearized", hourglass.max_force_per_node_fraction=0.2, csw_C1=0.5, csw_C2=2.0, csw_limiter="van_leer", csw_limiter_enabled=True, csw_shock_limiter_floor=0.65, csw_zero_uniform_compression=True, csw_diagnostics=False, av_limiter_J=1.0, av_heat_C=0.0, ion_art_heat_C=0.0, post_shock_heat=False, post_shock_heat_C=0.1, post_shock_heat_decay=3.0, post_shock_velocity_damping_C=0.0, bulk_viscosity_C=0.0, adaptive_av.enabled=False, av_eos_aware=False, av_eos_gamma1_ref=5/3, av_eos_boost_max=3.0, odd_even_damping_C=0.0, anti_hourglass_C=0.0, ee_odd_even_C=0.0, hk_velocity_damper_C=0.0, hk_velocity_damper_tau_min=8.0, hk_velocity_damper_grad_Te_max=0.2, hk_velocity_damper_grad_rho_max=0.3, hk_velocity_damper_guard_cells=25, compatible_energy=False, rho_e_linear_grid=False, eos_writeback=True, exact_override="none"（NUMERICS §3.1.4, §3.1.5, §3.1.6, §3.1.9, §3.2.5, §3.2.9, §3.2.9b, §3.2.9c, §3.2.12a, §3.2.13, §3.2.13a, §3.2.13b準拠）
- `Numerics.hydro.plasma_viscosity = {enabled: False, model: "braginskii", species: "ion", eta_const: 0.0, eta0_scale: 1.0, mfp_cap_cells: 20.0, lnlambda_fixed: 0.0, dt_safety: 0.3}` (scope: 1D (all geometries) and 2D RZ)
- Phase 4 AV/subzonal defaults: `Numerics.hydro.av_model="scalar_vnr_legacy"` and `Numerics.hydro.rz_momentum_scheme="volume_weighted"` for every topology, `subzonal_pressure_enabled=False`, `subzonal_band_mode="off"`, `subzonal_band_feather_layers=2`, `av_cfl_coefficient=0.25`, `csw_limiter_enabled=True`, `subzonal_merit_mode="caramana_auto"`, `subzonal_alpha1=1.4142135623730951`, `subzonal_alpha2=0.1`, `subzonal_merit_power=2`, and `subzonal_merit_constant=1.0`. Frozen-config default completion fills `"volume_weighted"` for legacy input. `csw_edge` requires explicit `subzonal_pressure_enabled=True`; no topology default enables it implicitly. If `csw_edge` is selected and `av_C1`/`av_C2` are omitted, their namelist defaults are `1.0/1.0` for the edge AV package while legacy VNR defaults remain `0.1/1.5`. The same conditional `1.0/1.0` defaulting applies to `csw_edge_csw98` (identical Kuropatenko kernel). `csw_edge_csw98` is NOT paired-validated with subzonal pressure (runs with either setting); central pseudo-core accepts either edge mode but still requires subzonal.
- plic：enabled=False, normal_estimator=`"youngs_seeded_LVIRA"`, t0_volume_cut_method=`"adaptive_subdivision_2x2"`, in_run_disabled=False, rho_material_aware_donor=False
- ale conservative reference remap：conservative_remap_enabled=False, conservative_remap_target=`"reference"`, conservative_remap_radiation_enabled=True, conservative_remap_order=`"first_order_donor"`, conservative_remap_lagrangian_bulk_enabled=False, conservative_remap_lagrangian_bulk_center_node_ring_max=4（valid range `>= 0`; the lagrangian-bulk keys are scoped to opt-in single-block spherical-polar-halfplane `tri_fan` pressure-driven conservative remap; I1 2D_RZ shock-frame deck may opt in explicitly to `"second_order_van_leer"`; tri_fan fan-touching faces fall back to first-order donor）
- ale central pseudo-core：central_pseudo_core_enabled=False, central_pseudo_core_s_c=0.0 cm（Exp1 2D_RZ five-block multiblock center diagnostic; ring-conforming membership with no halo; default-off and serialized only when enabled/non-default）
- materials：per_material_conservation_enabled=False, presence_threshold_volfrac=1.0e-10, presence_threshold_mass_density_g_per_cc=1.0e-12, eos_table_validity_lower_bound_g_per_cc={}, lazy_cache_te_m_enabled=False, hdf5_emit_derived_per_material=False, deposit_redistribute_fallback_enabled=False, deposit_redistribute_provenance_label=`"TENRYU_EXTENDED_ALE_WAVE_F_DEPOSIT_REDISTRIBUTE_FALLBACK"`, conservation_residual_warn_threshold_rel=1.0e-12, conservation_residual_hard_warning_threshold_rel=1.0e-10
- profile.icf_standard_ale：enabled=False, enforce=True, claim_level="characterization", allowed_when_enabled={ale_enabled_required_value=True, ale_axis_repair_mode_required_value="full_winslow", ale_remap_scheme_allowed_values=["legacy_split","ms2_moments"], hydro_driver_full_step_retry_enabled_required_value=True}, forbidden_when_enabled={hydro_dispatcher_state_sensitive_bypass_enabled_forbidden_value=True, ale_local_boundary_repair_enabled_forbidden_value=True, ale_multi_node_boundary_repair_enabled_forbidden_value=True, ale_multi_node_interior_repair_enabled_forbidden_value=True, ale_axis_variational_projection_enabled_forbidden_value=True, ale_emergency_cell_deactivation_enabled_forbidden_value=True, hydro_driver_retry_active_mesh_repair_enabled_forbidden_value=True}, escape_valves={allow_nonstandard_mesh_rescue=False, require_deck_reason=True, mark_run_nonstandard=True}
- profile.legacy_regression：enabled=False, revision="2026-07-27"
- hydro av_heat_to=ion（legacy PdV path の人工粘性仕事と人工熱流束はイオンへ。exact compatible-energy path の force-derived \(Q\) work も v1 ではイオンへ）
- 1D shock sensor 閾値は内部固定：
  `kShockPressureJumpThreshold=0.3`,
  `kShockDensityJumpThreshold=0.05`,
  `kShockRhConsistencyThreshold=0.5`,
  `kCompMachScale=0.05`,
  `kOscillationThreshold=0.2`,
  `kShockSupportFloor=0.25`
- hydro boundary=free（P_ext=0；2D z-face の `state_supply` dict は既定OFF；2D_RZ `boundary_2d.mesh_tangential_target="lagrangian"`、`state_supply_donor_mode="interior_per_i"`）
- conduction：enabled=True, solver=sts, sts_floor_limiter=net, ion_conduction=False, f_lim=0.06, mfp_limiter_C=0.0, sts_damping=0.01, sts_max_stages=40, sts_subcycle_eta=0.9, sts_total_stages_max=200000, halo_strategy=every, hypre_rtol=1e-8, hypre_max_iter=50, hypre_amg_coarsen=10, hypre_amg_relax=18, hypre_amg_interp=6, hypre_amg_levels=25, face_kappa_policy = "kirchhoff_same_material", nonlocal_model="none", snb_n_groups=24, snb_E_max_over_Te=20.0, snb_mfp="geometric_r2", snb_efield="none", snb_picard_max_iters=8, snb_picard_rtol=0.01
- coulomb_log_floor=2.0
- radiation_thermal_subcycle=False
- splitting order=strang
- positivity：clamp=True
- floors：rho_floor_gcc=1e-10, Te_floor_eV=1e-3, Ti_floor_eV=1e-3
- safety：energy_fatal=False, nan_fatal=True, energy_threshold=1e-3, overshoot_warn=0.01, overshoot_fatal=0.10, overshoot_fatal_enabled=False, clamp_warn_threshold=100, clamp_fatal_threshold=10000, opacity_floor=1e-20, opacity_cap=1e20
- cell search：max_rings=3, fatal=True（NUMERICS §9参照）
- diagnostics_every=1
- diagnostics.phase_resolved_energy=False
- diagnostics.r_momentum_source_audit=False
- diagnostics.dt_breakdown_history_enabled=True
- diagnostics.mesh_attribution: enabled=False, record_node_displacements=False, dump_on_failure_only=True, enable_leave_one_out_replay=False
- diagnostics.mesh_degeneracy_forensics: enabled=False, corner_j_source_budget_enabled=False, corner_j_source_budget_include_1_ring=False, velocity_history_enabled=False, velocity_history_target_cell_c=-1, velocity_history_sample_every_n_steps=1, velocity_history_include_1_ring=True, velocity_history_max_records=5000, same_cell_count=3, sigma_threshold=0.5, max_dumps_per_run=100, output_dir=`""`
- diagnostics.icf: enabled=False, rho_inner_threshold_g_per_cc=0.0, rho_outer_threshold_g_per_cc=0.0
- diagnostics.hotspot_gas: enabled=False, R_g_cm=0.0, mass_drift_warn_rel=1e-10
- diagnostics.ale_velcoherence: enabled=False, every_n_steps=1
- diagnostics.shock_approach: enabled=False, every=50, target_radius_cm=0.0, bins=192, h_cell_cm=0.0
- diagnostics.conservation.enabled=False
- diagnostics.ale_provenance_emission.enabled=False

**Radiation / IMC**：
- radiation enabled=True
- radiation mode=`"multigroup_diffusion"`（DEFAULT-FLD: FREEZE-1D-RAD/FLD-FIX-1 後の既定。1D_SPH/2D_RZ production radiation は `"multigroup_diffusion"` と `"sn_transport"` のみ受理する。`"imc_ddmc"`（従来 IMC/DDMC/PGRW/HOLO/difference 経路）は 1D_SPH/2D_RZ production namelist では `ConfigError`。`"sn_transport"` は 1D_SPH/2D_RZ CUDA pure \(S_N\) で、FLD/S_N は IMC/DDMC/HOLO/difference 有効時に `ConfigError`）
- origin_parity_only=False, group_repack_hard_xray=False, diagnose_hard_xray_opacity=False（legacy 既定はすべてOFF）
- volume_source_rate=0.0 [erg/(cm³ s)], volume_source_x_max=-1.0 [cm]（外部体積線源 OFF。1D_SPH FLD は W-B 実装、`rate > 0` は `groups=1` かつ `x_max > 0` 必須）
- planck_fraction：method=compute, compute_N_T=200, compute_T_range_eV は未指定時に EOS table 温度範囲から自動導出
- IMC enabled=False（internal/test-only。`mode="imc_ddmc"` の legacy 経路で必要な場合は明示的に有効化する。production namelist では `imc` subblock を設定しない）
- IMC time-centering α=1.0（fully implicit）
- IMC f_max=1.0（Fleck factor上限、制限なし）
- IMC corrected_fleck=False（既定。True で Cleveland & Wollaber (2018) の修正 Fleck factor）
- IMC emission particles：50 / cell / group（テスト用；本番は増やす）
- IMC implicit_capture=True, inelastic_scatter=True
- IMC cutoff_fraction=0.0（無効）
- IMC weight_cutoff=1e-10, roulette_survival=0.1, weight_split=1e+2, max_split=8（weight_split/max_split は v1.0 では予約）
- IMC max_pool_size=100,000,000（1億粒子、GPUメモリ60%上限との min で制約）
- IMC source_tilting=False（既定OFF。True で 1D_SPH/2D_RZ thermal source の emit 位置だけを \(T_e^4\) 勾配で傾ける）
- IMC source_localization=False（既定OFF。True で前ステップ吸収 midpoint の mean/variance から 1D_SPH thermal source を finite-width PDF へ局所化する）
- IMC sloc_ema_beta=0.4（source_localization の mean radius に対する時間方向 EMA）
- IMC sloc_sigma_floor=0.1, sloc_sigma_cap=0.5（source_localization の emit 幅の cell-width 比 floor/cap）
- IMC sloc_tau_ref=1.0（source_localization の optical-depth gate 基準）
- IMC spectral_bias_eta=0.0（既定OFF。`0.3-0.5` で thermal emission の window-group biasing を有効化）
- IMC opacity_predictor=False（既定。true NLTE の係数評価だけで半ステップ温度予測を使う）
- IMC two_stage=False（既定。True で `R(Δt)` を `Δt/2 + Δt/2` へ分割し，中間で EOS / Zbar を再同期）
- IMC difference={enabled=False, W_max=1.0, tau0=3.0, chi0=1.0, face_transport=True}（既定OFF。1D_SPH および `face_transport=False` の 2D_RZ LTE nonlinear thermal source を signed residual source に置換し、物理 `rad_emit` と reference absorption preseed を保持し、step 冒頭の census を previous-reference reservoir と signed residual 粒子へ再分割する。`face_transport=True` は 1D_SPH のみ AP reference face transport で \(U^{ref,end}\) を更新し、2D_RZ では ConfigError。`rad_E` は reference average と signed residual estimator の和として再構成する。PR9 gate 完了まで production 推奨は行わない）
- IMC net_e_source_smoothing={enabled=False, alpha=0.2, tau_threshold=4.0, passes=1, grad_Te_scale=0.3, grad_rho_scale=0.5, gradient_adaptive=False}（既定OFF。1D_SPH/2D_RZ の optically thick かつ非interface face で net electron source を conservative に平滑化。2D_RZ で enabled=True の alpha 上限は 0.125、その他は 0.25。difference 併用時は \(W\ge0.5\) セルを smoothing barrier とする。gradient_adaptive=True では \(T_e,\rho\) の対数勾配で face 係数を連続的に弱める）
- IMC particle_budget=-1（無効。>0で有効化。検証テストでは使用しない）
- IMC census_comb：enabled=False, max_particles=1,000,000, min_per_bin=1, trigger_ratio=1.0, target_fraction=0.8, mode_weight_imc=1.0, mode_weight_ddmc=0.5, adaptive_trigger=True, adaptive_util_start=0.70, adaptive_util_end=0.95, trigger_ratio_floor=0.85, trigger_hysteresis=0.05, ess_floor_enabled=False, ess_min_tier0=16.0, ess_min_tier1=8.0, max_split_factor=4
- IMC rad_lite_mesh：enabled=False, sigma_ratio_max=2.0, nlte_auto=False（1D_SPH専用）
- FLD multigroup_diffusion：flux_limiter=`"levermore_pomraning"`, max_outer_iterations=20, outer_tol=1e-5, fleck_mode=`"fleck_cummings"`（allowed: `"afi"`）, hydro_coupling=`"gamma_r_43"`（allowed: `"none"`）, state_supply_boundary_policy=`"local_D_current"`, diagnostic_radial_fourier_substage_enabled=False, cg_inner_tol=1.0e-10, cg_tol_norm=`"r0"`, outer_accel=`"none"`, anderson_m=2, anderson_beta=1.0, cg_max_iter=500, cap_exit_policy=`"warn"`（allowed: `"fail"`）, rgmg_smoother_omega=0.67, linear_solver_1d=`"cusparse_tridiag"`, linear_solver_2d=`"auto"`（allowed: `"auto"`, `"amgx_cg"`, `"jacobi"`, `"cusparse_cg_jacobi"`, `"cusparse_cg_zline"`, `"cusparse_cg_rgmg"`；`"auto"` は nr 2 冪かつ nz>=3 → rgmg / nz>=3 → zline / else jacobi に validate 時解決。**明示 `"amgx_cg"` の AmgX 未 link build は ConfigError（fatal）**。requested/resolved を run_info+HDF5 metadata に記録）, amgx_config.preset=`"AGGREGATION_JACOBI"`, opacity_floor=1e-100, opacity_cap=1e20, z_boundary=`"vacuum"`（`"state_supply"` は default-off）, boundary.inner_r=`"reflect"`, boundary.outer_r=`"vacuum"`, boundary.z=`"vacuum"`, boundary.z_bottom=`"vacuum"`, boundary.z_top=`"vacuum"`, marshak.flux_erg_per_cm2_s=0, marshak.flux_pulse_duration_s=-1（`mode="multigroup_diffusion"` で使用）
- S_N sn_transport：n_angles=16, angular_quadrature=`"level_symmetric_16"`, spatial_scheme=`"linear_characteristic"`, max_outer_iterations=20, max_inner_iterations=100, outer_tol=1e-4, outer_tol_stagnation_factor=0.5, outer_tol_hydro_error_scale=1e-5, inner_tol=1e-6, inner_graph_unroll=5, dsa_enabled=True, diffusion_fallback_mode=`"none"`, tau_diffusion_on=10.0, tau_diffusion_off=5.0, opacity_floor=1e-100, opacity_cap=1e20, timing_enabled=False, z_boundary=`"vacuum"`, boundary.inner_r=`"reflect_parity"`, boundary.outer_r=`"vacuum"`, boundary.z=`"vacuum"`, boundary.z_bottom=`"vacuum"`, boundary.z_top=`"vacuum"`, marshak.flux_erg_per_cm2_s=0（`mode="sn_transport"` で使用。1D_SPH/2D_RZ closure は conservative active set + face flux + donor theta + AP face blend に固定。`"diamond_difference"` は deprecated regression option）
- radiation boundary：type=vacuum, inner=reflect（1D_SPH）, marshak_particles=1000

**Radiation / DDMC**：
- DDMC enabled=False（internal/test-only。`mode="imc_ddmc"` の legacy 経路で必要な場合は明示的に有効化する。production namelist では `ddmc` subblock を設定しない）
- DDMC implicit_diffusion=False（既定OFF。HIMCD Phase-1 は 1D LTE 限定）
- tau_ddmc=4.0, tau_rw=0.0, omega_ddmc=0.9
- DDMC leak_stencil=9_kershaw（Kershaw 9点差分）
- DDMC interface_method=asymptotic_diffusion_limit
- DDMC emissivity_preserving=True（Densmore 2006 \(\hat{P}\) 補正）
- DDMC interface_exit_distribution=cosine（μ比例）
- DDMC face_opacity_temperature=radiative_mean
- DDMC m_matrix_check=True
- DDMC momentum_deposition=True
- DDMC rz_face_r_weight=True（2D_RZ面位置サンプリングのR重み付け）
- DDMC tau_ddmc_off=-1.0、omega_ddmc_off=-1.0
- DDMC mode_hold=0（ヒステリシスなし）、rate_max=1e30（無制限）

**Radiation / Diffusion**：
- diffusion enabled=False
- tau_on=5.0, tau_off=3.0
- reduced_flux_on=0.15, reduced_flux_off=0.25
- mode_hold=0, rate_max=1e30
- mode_update_interval=10, min_diffusion_island_cells=5
- imc_guard_cells=1
- sts_max_stages=0, sts_damping=0.05, sts_subcycle_eta=0.8
- interface_particles_per_face_group=32, exit_particles_per_cell_group=32
- lte_entry_initialization=False, lte_entry_energy_fraction_cap=0.01

**Radiation / HOLO**：
- holo enabled=False（internal/test-only。production namelist では `holo` subblock を設定しない。既存 physics runtime 経路を変更しない）
- region=shell, material_group=shell
- coupling_tau=5.0, guard_cells=3
- deprecated compatibility: tau_on=5.0, tau_off=3.0, reduced_flux_on=0.15, reduced_flux_off=0.25
- deprecated compatibility: update_interval=10, min_dwell_steps=20, min_island_cells=5, core_margin_cells=3
- solver=implicit_1d, closure=diffusion, closure_relax=0.2, closure_smooth_passes=1, closure_smooth_alpha=0.5, consistency_alpha=1.0（`gamma_alpha` は互換 alias）
- boundary_flux=physical
- p_rr_tally=True
- sn_closure=True, sn_n_angles=8, sn_material_coupling=False
- residual_particles_per_cell_group=4

**Laser**：
- laser mode：raytrace_2d（1D_SPH）、raytrace_3d（2D_RZ）
- laser mode option：radial_absorption_1d（1D_SPH専用、全ビームパワーを inward radial flux として合算）
- laser rays_per_beam=1000（1D_SPH）、128（2D_RZ）
- laser ray_output_count=0, ray_output_trajectory=False, ray_output_max_steps=10000
- laser profile：model=gaussian
- laser critical：eps_n=1e‑4, eps_crit=1e‑4, terminate=True
- laser coulomb_log_floor=2.0, absorption.debug_dump_lasermesh=False
- laser raytrace：integrator=leapfrog（v1.0固定）, cfl_ray=0.8, gradient_interpolation=bilinear, intensity_cutoff=1e-6, debug_one_ray=False
- laser mesh（2D_RZ）：enabled=True, nr=128, nz=256, r_max=1.5×R_target, z_min/z_max=Z_center±1.5×R_target
- laser mesh（1D_SPH動的）：mesh_factor=0.5, rmax_n_hat_threshold=0.001, nr_max=4096, R_crit中心piecewise-geometric graded（g_core=1.08, g_corona=1.05）, nz=2×nr
- laser mesh：critical_clip=True, critical_margin=1-eps_crit（= 0.9999）
- laser mesh ghost_corona：enabled=False, n_out=12, ne_min_frac=0.03, ne_max_frac=0.99, Te_min_eV=50, zbar_min=1.0, zbar_max=4.0, handoff_cells=4, handoff_decay=1.5, transition_enabled=False, transition_resolved_nhat=0.9, transition_resolved_cells=3, transition_density_exponent=1.0
- laser deposit：bilinear_node（近傍4節点双線形分配）, deposit_smooth_passes=0（1D_SPH/2D_RZ とも既定で無効）, deposit_smooth_alpha=0.25
- laser raytrace_skip：enabled=False (wave-5 2026-08-07 で既定 OFF 化 — 旧既定 True は crit ガード不発により全デッキで実質不活性だったため挙動互換), threshold=0.01, max_consecutive=10, norm=max_relative, crit_guard=0.01（crit ガードは wave-5 でキャッシュ相対の帯域横断判定に改定 — NUMERICS §5.9.4）
- laser cbet：enable=False（**既定 OFF・bit 恒等**）, f_cbet=1.0, alpha_iaw=0.2, theta_cap=0.3, tol=1e-3, max_iters=50, n_impact_bins=16, n_phi=8, ne_frac_cutoff=0.95, k_a_floor=1e-6, max_segments_per_ray=0(auto), test_chi=-1(off)；beam 毎 delta_lambda_nm=0.0, geometry_mode=legacy, n_section_phi=16
- laser port_configuration：absent（既定）；normalization=sum_weights_one, port 毎 roll_deg=0.0, delta_lambda_nm=0.0, beam_class="", polarization=unpolarized（port_id/direction/power_weight は必須）
- laser hot_electron（多ビーム overlap）：tpd_overlap_mode=single_beam, srs_overlap_mode=per_beam_class, illumination_metric=fixed, common_wave_delta_theta_deg=-1（cluster 時必須）
- laser hot_electron：enable=False, source_nc_fraction=0.25, eta_hot=0.0, eta_hot_table=absent, eta_mode=legacy, eta_model={ln_filter_tau_s=5.0e-12, eta_total_cap=0.08}, T_hot_eV=5.0e4（eV）, n_energy_groups=30, E_min_over_Th=0.2, E_max_over_Th=8.0, angular_model=cone, theta_div_deg=60.0, n_mu=6, n_phi=8, subtract_from_laser=True, inner_bc=deposit_residual, explicit_source_limit=0.2, sources=absent
- laser hot_electron.sources[i]（機構別チャネル；mechanism 依存の parse 時既定）：mechanism=cone, eta=0.0, eta_table=absent, n_energy_groups=30, E_min_over_Th=0.2, E_max_over_Th=8.0, n_mu=6, n_phi=8；capture_nc_fraction=0.25（srs のみ 0.18）, T_hot_eV=5.0e4（tpd 6.0e4 / srs 4.5e4）, theta_div_deg=60.0（srs 20.0；cone/srs のみ）, tpd_theta_deg=45.0, tpd_delta_deg=10.0（tpd のみ）；eta_mode=model のみ（parse 時に sentinel 解決）：eval_nc_fraction=capture_nc_fraction, threshold_multiplier=1.0（srs 8.0）, eta_inf=0.01（srs 0.08）, eta_hard_cap=0.03（srs 0.08）, shape_coefficient=1.0, relaxation_model=vu2012（srs fixed）, relaxation_tau_s=6.0e-12, relaxation_tau_min_s=3.0e-12, relaxation_tau_max_s=1.0e-11

最小 1D radial absorption 例：

```python
Laser(
    enabled=True,
    mode="radial_absorption_1d",
    beams=[
        dict(
            name="b0",
            direction=[0.0, 0.0, -1.0],
            f_number=8.0,
            power=lambda t: 1.0e13,
        )
    ],
)
```

`direction` と `f_number` は LaserBeam schema のために指定するが、このモードの吸収分布には影響しない。

**Parallel**：
- parallel decomposition：method=slab(1D)/cartesian(2D), dims=None（自動決定）, min_cells_per_rank=8
- parallel halo：gpu_aware_mpi=auto
- parallel migration：method=batch, max_substeps=32, emigrant_threshold=1000
- parallel laser_parallel：strategy=replicated
- parallel particle_balance：enabled=False, imbalance_threshold=1.5, method=work_stealing
- parallel reproducibility：mode=statistical, sort_after_migration=False
- parallel gpu_optimization：particle_sort_by_cell=True, tally_mode=warp, compute_comm_overlap=False

**Output**：
- output directory="./output", format=hdf5, plot_every=100, history_every=1, checkpoint_every=1000, checkpoint_keep_last=2
- output plot_every_s=-1.0, history_every_s=-1.0, checkpoint_every_s=-1.0
- output compression=gzip, compression_level=4
- save_namelist_copy=True, save_frozen_config=True
- restart compatibility checks include both `/metadata/eos/eos_signature` and `/metadata/frozen_config`

**Diagnostics**：
- diagnostics：enabled=True, every=1
- diagnostics areal_density：enabled=True, angles_deg=[0,45,90], r_range=shell
- diagnostics sphericity：enabled=True, modes=[0,2,4], surface=isodensity, rho_threshold=10.0
- diagnostics energy_budget：enabled=True, warn_threshold=1e-3
- diagnostics laser_pattern：enabled=True, absorbed_power_profile=True, critical_surface=True, per_beam=False
- diagnostics mc_stats：enabled=True, particle_counts=True, weight_stats=True, cell_particle_density=False, ddmc_fraction=True
- diagnostics fleck_diag：enabled=False, every=10, cells=[], r_min_cm=-1.0, r_max_cm=-1.0
- diagnostics overshoot_monitor=True

**Main**：
- verbosity=normal, seed=12345, max_steps=10_000_000, units=cgs_eV

**RNG**：
- Philox4x32‑10、Main.seed=12345（既定；§6.4.1参照）

### 9.2 ユーザが調整すべき重要パラメータ
- 物性：EOS/opacityテーブル、Zbarモデル、混合則
- 放射多群：群境界、群数
- 粒子数：統計誤差と速度のトレードオフ
- rezoning頻度・品質閾値
- レーザーパルス形状（最適化対象）
- ビーム配置・スポット（最適化対象）

### 6.4 Addendum: `numerics.plic.*` (Stage 30 Wave A)

`numerics.plic` is default-disabled. With `enabled=False`, the fields below are
serialized and validated but not consulted by runtime physics paths.

| key | default | valid values / range |
|---|---:|---|
| `enabled` | `False` | bool |
| `normal_estimator` | `"youngs_seeded_LVIRA"` | `"youngs"`, `"LVIRA"`, `"youngs_seeded_LVIRA"` |
| `t0_volume_cut_method` | `"adaptive_subdivision_2x2"` | `"centroid_only_legacy"`, `"adaptive_subdivision_2x2"`, `"adaptive_subdivision_3x3"` |
| `t0_volume_cut_max_depth` | `6` | integer `[4, 16]` |
| `t0_volume_cut_volfrac_tol` | `1e-10` | finite `> 0` |
| `fast_path_threshold_min` | `1e-10` | finite `> 0` |
| `fast_path_threshold_max` | `0.9999999999` | finite `> 0`; `< 1` when enabled |
| `fast_path_halo_radius_cells` | `1` | integer `>= 1` |
| `alpha_solver_max_iter` | `50` | integer `>= 1` |
| `alpha_tolerance_rel` | `1e-12` | finite `> 0` |
| `thermodynamic_error_soft_threshold` | `0.05` | finite `> 0`; calibration-pending |
| `thermodynamic_error_hard_threshold` | `0.10` | finite `> 0`; calibration-pending |
| `class_d_dense_fraction_threshold` | `0.01` | finite `> 0`; calibration-pending |
| `material_interface_per_cell_state` | `"off"` | `"off"`, `"sparse_on_degradation"`, `"dense_debug"` |
| `production_comparable_gate_strict` | `True` | bool |
| `drift_sensor_max_relative` | `0.1` | finite `> 0`; cell-width fraction |
| `drift_sensor_max_swept_fraction` | `0.05` | finite `> 0` |
| `prev_normal_freshness_volfrac_threshold` | `0.05` | finite `> 0` |
| `plic_per_step_cost_target_fraction` | `0.5` | finite `> 0`; PLIC cost target |
| `in_run_disabled` | `False` | bool |
| `rho_material_aware_donor` | `False` | bool; CF6 Wave F preview |

`t0_volume_cut_max_depth` default 6 (was 12). Empirically: max_depth=6 is
sufficient to resolve smooth Legendre-perturbed shell interfaces in I1 capsule
deck within ~5 sec/cell. Higher depths (>=12) cause exponential blowup against
hard step functions and are not recommended for production. Wave A range was
[8, 16]; Wave E expanded to [4, 16] to allow faster init for empirical runs.

Wave C runtime semantics are intentionally narrower than the namelist surface:
`enabled=False` is inert and allocates no PLIC remap scratch; `in_run_disabled`
is a per-run switch that forces the scalar ALE material-fraction remap even
when `enabled=True`; and a per-run sticky fallback may set the same effective
runtime-disabled state after repeated drift triggers.  The drift defaults remain
`drift_sensor_max_relative=0.1` and
`drift_sensor_max_swept_fraction=0.05`.  The relative drift is measured from
the reconstructed interface centroid in each multi-material cell, not from cell
corner motion.

Wave E adds `rho_material_aware_donor=False` by default.  When enabled, the
runtime PLIC remapper may remap density with a material-aware donor density at
interface donor cells.  It does not add per-material density arrays to `State`;
failed or disabled PLIC remap keeps density on the scalar ALE remapper.

Wave C PLIC remap is serial-only.  A configuration with `part.n_ranks > 1` and
`numerics.plic.enabled=True` raises
`ConfigError("Wave C PLIC remap not validated under MPI; deferred to Stage 31")`
at ALE entry.  This guard does not change disabled PLIC runs.

### 7.3 Addendum: `/diagnostics/material_interface/v1/`

Stage 30 adds a path-versioned history group
`/diagnostics/material_interface/v1/`. `kSchemaVersion` remains 1 because this
is additive. When PLIC is disabled, the group is omitted. Wave A creates the
group only when PLIC is enabled and writes identification attributes:
`plic_reconstruction_engine_version`, `plic_normal_estimator`,
`t0_volume_cut_method`, `plic_enabled`, `plic_schema_version`, and
`plic_reconstruction_method`.

Wave D fills the group with per-sample scalars (`time_s`, `step`,
`interface_cells_observed`, reconstruction attempt/success counts,
`plic_max_eta_E_observed`, `plic_max_volume_fraction_residual_observed`, and
`plic_min_grad_F_observed`), a `[N,3,3]`
`class_d_runtime_fires_matrix`, `/plic_events/`, and optional `/per_cell_state/`
when `material_interface_per_cell_state` requests it.  The event stream is
rate-limited: more than 10000 newly emitted events in one history call are
written as aggregate rows under `/plic_events_summary/` instead of individual
`/plic_events/` rows for that call.

Wave C also exposes the in-memory `plic_remap_fallback_engaged` observability
flag for Wave D attribute wiring.  The HDF5 schema version remains
`kSchemaVersion == 1`; the material-interface diagnostics group is additive.

### 9.1 Addendum: PLIC `production_comparable` gate

The PLIC production-comparable gate is code-enforced, not an HDF5 cross-group
constraint. It is enabled only when `numerics.plic.enabled=True`. Wave D
requires the run to reach `t_end`, retain `PUBLIC_BASELINE` ALE provenance,
avoid hard or dense class-(d) aggregate events, and maintain reconstruction
success rate at least 0.999 after excluding axis-exempt cells from the
denominator.  Wave C supplies the remap fallback signal and the maximum
material-volume residual needed by that evaluation.  If the measured
PLIC per-step cost exceeds `plic_per_step_cost_target_fraction` times the
scalar ALE remap baseline, Wave C completion is still allowed but Wave D/E must
deny a `production_comparable` claim until calibration or optimization closes
the gap.  On failure, Wave D downgrades the run claim rather than allowing a
`production_comparable` assertion. When the original claim was
`production_comparable`, the evaluated `plic_gate_status` is also duplicated
under `/diagnostics/ale_provenance/v1/`. Wave E empirical calibration is
required before the claim can be accepted.

---

## 10. 互換性と参照方針（DRACO/xRAGE）
- DRACO：以下の設計・手法を踏襲する:
  - **IMC基本構造**: Fleck factor, census, 粒子プール, implicit capture
  - **Lagrangian hydro**: 構造四辺形RZメッシュ上のスタガード格子、Wilkins型面積重みコーナー力
    （NUMERICS §3.2参照）
  - **Kershaw 9点差分**: 歪格子上の拡散離散化。電子熱伝導（NUMERICS §4.3）および
    DDMCリーク係数（§7.3）で使用（NUMERICS Appendix A参照）
- xRAGE：レーザーメッシュ分離と沈着写像の設計思想のみ参考（外部ツール連携はしない）。
- ただし、TENRYUは v1.0で **CUDAのみ**、**入力はPython namelist** を前提とする点で異なる。

---

## 11. 制限事項（v1.0で明示しておくこと）
- 3D非対称（m≠0）は扱えない（2D_RZではPℓのみ）。
- 放射のO(v/c)項は無視（将来拡張）。
- LPIは未実装（将来拡張）。
- SESAME/IONMIXテーブルの適用範囲外（密度/温度外）は **外挿を禁止**し、
  - clamp（範囲端に固定）を既定
  - 逸脱をdiagnosticsで警告
  とする（物理破綻防止）。
- `conservative_overlap`（多ビームレーザー重なり領域での保存的沈着補正）は **v1.1へ延期**。v1.0では **加算重ね合わせ（additive superposition）** を使用する（§5.4.1, §5.4.2の多ビーム重ね合わせ参照）。

---
