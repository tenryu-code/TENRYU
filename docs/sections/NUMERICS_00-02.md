<!-- 分割元: docs/NUMERICS.md | このファイルは参照用です。原本（docs/NUMERICS.md）が権威です。 -->
# TENRYU — NUMERICS.md
本書はTENRYUの数値仕様（支配方程式、離散化、多群輻射輸送（現行: FLD §6.7 / S_N §6.8、退役: IMC–PGRW–DDMC §6.3–6.6・§7）、レーザーレイトレース（CBET §5.10・ホット電子 §5.11 含む）、推定量、安定化・安全策）を定義する。
**本書に書かれた規約が“唯一の真実”**であり、実装は必ず一致させる。

---

## 0. 規約・単位・記号（必読）
### 0.1 単位（cgs + eV）
- 長さ：cm、時間：s、密度：g/cm³、温度：eV
- エネルギー：erg、圧力：dyne/cm²
- 放射群境界：eV（光子エネルギー）

**基本物理定数・変換因子（実装定数）**

| 記号 | 名称 | 値 | 単位 |
|------|------|------|------|
| \(m_e\) | 電子質量 | \(9.1094 \times 10^{-28}\) | g |
| \(m_p\) | 陽子質量 | \(1.6726 \times 10^{-24}\) | g |
| \(c\) | 光速 | \(2.9979 \times 10^{10}\) | cm/s |
| \(e\) | 素電荷（esu） | \(4.8032 \times 10^{-10}\) | esu |
| \(\sigma_{SB}\) | Stefan–Boltzmann定数 | \(5.6704 \times 10^{-5}\) | erg cm\(^{-2}\) s\(^{-1}\) K\(^{-4}\) |
| \(a_{cgs}\) | 放射定数（Kelvin系） | \(7.5657 \times 10^{-15}\) | erg cm\(^{-3}\) K\(^{-4}\) |
| \(\text{eV\_to\_erg}\) | eV→erg変換 | \(1.6022 \times 10^{-12}\) | erg/eV |
| \(k_{B,eV/K}\) | Boltzmann定数（eV/K） | \(8.6174 \times 10^{-5}\) | eV/K |
| \(a_{eV}\) | 放射定数（eV系） | \(1.3720 \times 10^{+2}\) | erg cm\(^{-3}\) eV\(^{-4}\) |

導出関係：
- \(a_{cgs} = 4\sigma_{SB}/c\)
- \(a_{eV} = a_{cgs} / k_{B,eV/K}^4\)（§6.1で詳述）
- \(\text{eV\_to\_erg} = k_{B,cgs} / k_{B,eV/K}\)（= 1 eV のエネルギー [erg]）

> **k_B 記号の規約**：本書の式中で温度 \(T\) [eV] と組み合わせる場合、
> \(k_B\) は **eV\_to\_erg** = \(1.6022 \times 10^{-12}\) erg/eV を意味する
> （例：\(P = nk_B T\) → \(n \times \text{eV\_to\_erg} \times T[\text{eV}]\) [dyne/cm²]）。
> Kelvin系の Boltzmann 定数 \(k_{B,cgs} = 1.3807 \times 10^{-16}\) erg/K とは異なる値であるため、
> 明示的に数値を記す箇所では **eV\_to\_erg** または **\(k_{B,eV/K}\)** の名称を用いて混同を防ぐ。

### 0.1.1 比熱の記号規約
本書では2種類のC_vを使い分ける。混同防止のため以下のルールに従う：

- **c_v** [erg/(g·eV)]：質量比熱（EOS テーブル出力値、ARCHITECTURE EOSTable::Cv_table）
- **C_v = ρ c_v** [erg/(cm³·eV)]：体積比熱（離散化式中で使用）

> **CUDA_KERNELS カーネル引数**: カーネル引数名 `Cv_e`, `Cv_i` は **質量比熱 c_v** [erg/(g·eV)] を渡す（EOS テーブル出力値と同一）。カーネル内部で必要に応じ `C_v = ρ × c_v` を計算する。名前の `C` は歴史的慣習による。

式中で c_v と C_v を混同すると ρ倍のズレが生じる。SPECIFICATION §6.4.3 の `cv_e_override` は体積比熱 [erg/(cm³·eV)]。

### 0.2 不透明度（致命的事故防止）
TENRYUは **質量不透明度 κ** と **長さ不透明度 σ** を厳密に区別する。

- テーブル入力（SESAME/IONMIX）が返す：
  - **κ_P,g(ρ,T)**：Planck mean（群別）質量不透明度 \([cm^2/g]\)  
  - **κ_R,g(ρ,T)**：Rosseland mean（群別）質量不透明度 \([cm^2/g]\)
- 輸送/拡散で使う内部表現：  
  - **σ_a,g = ρ κ_P,g** \([1/cm]\)（IMC吸収・放射に使用）  
  - **σ_R,g = ρ κ_R,g** \([1/cm]\)（DDMC拡散係数に使用）

> 重要：σ と κ を混同すると吸収長が **ρ倍ズレる**。  
> 実装では `Opacity::kappa_*` と `Opacity::sigma_*` を別名で保持し、暗黙の変換を禁止する。

### 0.3 放射の群（multigroup）
群境界（eV）：\([E_0, E_1, ..., E_G]\)。群 g は \([E_{g-1},E_g)\)。

- 群代表エネルギー（既定）：幾何平均  
  \(E_g^{rep}=\sqrt{E_{g-1}E_g}\)
- 温度T（eV）に対する **Planck分率**（群内黒体エネルギーの割合）：
\[
b_g(T)=\frac{\int_{E_{g-1}}^{E_g}\frac{E^3}{\exp(E/T)-1}dE}{\int_{0}^{\infty}\frac{E^3}{\exp(E/T)-1}dE}
\]
ここで分母 \(\int_0^\infty \frac{x^3}{e^x-1}dx=\frac{\pi^4}{15}\) を用いると
\[
b_g(T)=\frac{15}{\pi^4}\int_{x_{g-1}}^{x_g}\frac{x^3}{\exp(x)-1}dx,\quad x=E/T
\]
- \(\sum_g b_g(T)=1\) を数値的に保証する（補正正規化を実装で入れて良い）。

**群境界の被覆と尾部欠損**：
上記の定義で分母は \([0,\infty)\) の積分であるが、群境界 \([E_0, E_G]\) が有限の場合、
物理的には \(\sum_g b_g^{raw}(T) < 1\) となりうる（高エネルギー尾部や低エネルギー側の寄与が群外にある）。
再正規化により \(\sum_g b_g = 1\) とするが、欠損率が大きい場合はスペクトルが系統的に歪む。

- **欠損率の定義**：\(\delta(T) \equiv 1 - \sum_g b_g^{raw}(T)\)
- **許容欠損率**：\(\delta < 10^{-3}\)（0.1%）。超過時は WARNING を出力。
- **診断出力**：初期化時に全温度グリッド点での \(\max_k \delta(T_k)\) を出力し、群設計の妥当性を検査する。
- **群境界の推奨設計**：\(E_0 \ll T_{min}\)、\(E_G \gg T_{max}\) とし、
  Planck関数の尾部を十分にカバーすること。
  実用上、\(E_0 \le 10^{-3}\) eV、\(E_G \ge 10 \times T_{max}\) eV を推奨する。

**計算方法（v1.0既定）**
- namelistで `planck_fraction.method="compute"` のとき：
  - 初期化時に温度グリッド \(\{T_k\}\) を自動生成し、各Tで数値積分して \(b_g(T_k)\) をテーブル化
  - 実行中は \(b_g(T)\) を線形補間（TはeV）
- `method="tabulate"` のときはユーザテーブルを読む（検証用途）

**数値積分の安定評価規約**（被積分関数 \(f(x) = x^3/(\exp(x)-1)\), \(x = E/T\)）：
- \(x > 500\)：\(f(x) \approx x^3 \exp(-x)\) を使用（`exp(x)` は `x > 709` で double overflow）
- \(x < 10^{-3}\)：\(f(x) \approx x^2 - x^3/2 + \ldots\)（`expm1(x)` を使用してゼロ除算回避）
- それ以外：\(f(x) = x^3 / \text{expm1}(x)\) を使用（`exp(x)-1` の精度損失を `expm1` で回避）
- 再正規化後の検証：\(|\sum_g b_g - 1| < 10^{-12}\)（double精度）を assert する

**テーブル温度範囲と範囲外処理**

- `compute_T_range_eV` が未指定の場合、Planck table の温度範囲は active material の
  EOS table 電子温度範囲から自動導出する。EOS table 範囲が得られない場合は opacity table
  温度範囲、最後に `[0.01, 1000.0]` eV へフォールバックする（SPECIFICATION §6.4.5）。
  検証ケース等で解析解の温度範囲を固定したい場合は `compute_T_range_eV` を明示指定する。
- 実行時に \(T > T_{max}\) または \(T < T_{min}\) の参照が発生した場合、
  \(b_g\) は端点の値に **クランプ**（外挿しない）される。
- **範囲外アクセス診断**：各ステップで範囲外参照の回数 \(N_{OOR}\) と
  最大超過率 \(\max(|T - T_{max}|/T_{max},\, |T_{min} - T|/T_{min})\) を記録し、
  \(N_{OOR} > 0\) のとき WARNING を出力する。
  この診断により「テーブル範囲不足による静かな精度劣化」を防止する。
- **推奨範囲**：\(T_{min}^{tab} \le \min(E_0)/10\)、\(T_{max}^{tab} \ge \max(E_G) \times 10\)
  （群境界の1/10〜10倍）。Marshak/Su-Olson等の高温検証では境界温度を必ずカバーすること。
  **記号注意（2026-07-26 明確化, AI review k12 §3.3）**：ここでの
  \(T_{min}^{tab}, T_{max}^{tab}\) は **`compute_T_range_eV` = \(b_g(T)\) テーブルの温度軸範囲**であり、
  上の群境界設計則（\(E_0 \ll T_{min}^{plasma}\)、\(E_G \gg T_{max}^{plasma}\)）に現れる
  **プラズマ温度範囲とは別物**。同じ記号で読むと二つの規則が逆向きに見えるが矛盾ではない
  （テーブル温度軸は群境界の外側まで、群境界はプラズマ温度の外側まで、が正しい包含関係）。

**Hard-X-ray 群再配置（任意）**：
`Radiation.group_repack_hard_xray=True` の場合、既存の群数を変えずに群境界だけを再配置する。
80群構成では、明示または自動導出された `compute_T_range_eV=[T_min,T_max]` の範囲内で
\([T_{min},10]\), \([10,200]\), \([200,1000]\), \([1000,2000]\),
\([2000,5000]\), \([5000,T_{max}]\) eV の各区間を対数分割し、群数を
\(15,20,5,5,20,15\) に割り当てる。これにより 200--5000 eV 帯に 30 群、
2--5 keV 帯に 20 群を置く。
実装（`repack_radiation_group_bounds_for_hard_xray`）は固定エッジ
\(\{10,200,1000,2000,5000\}\) のうち \((T_{min},T_{max})\) の**内側にあるものだけ**を採用する —
\(T_{max} < 5000\) 等でも逆転区間は生成されない（2026-07-26 明記, AI review k12 §3.4 対応）。
既知の設計制約：photon domain の上端が `compute_T_range_eV` の \(T_{max}\) で切られるため、
\(T\sim T_{max}\) の黒体尾部は群外になる（\(b_g\) 再正規化が保存は維持する）。
domain を \(15\times T_{max}\) 級へ拡張する再設計は P1 相当の follow-up。
入力 opacity table は同じ群数のまま、新しい群代表エネルギーへ
\(\kappa_R,\kappa^{PA},\kappa^{PE}\) をエネルギー方向の log-linear 補間で再標本化する。
既定は False であり、既存 table の群境界をそのまま用いる。

**Hard-X-ray opacity 診断（任意）**：
`Radiation.diagnose_hard_xray_opacity=True` の場合、初期化時に一度だけ
CD material（存在しなければ最初の non-void material）の
\(\kappa^{PA}\) を \(\rho=\{0.2,1,3,6\}\) g/cm\(^3\)、
\(T=\{10,30,80,150\}\) eV、\(h\nu=\{1,2,3,4,5,6\}\) keV で評価して
`[hard_xray_opacity_diag]` として 24 行出力する。出力には NIST polystyrene 参照値
\(\kappa(2,3,4,5\mathrm{keV})=(278,83,34,17)\) cm\(^2\)/g を併記する。
この診断は opacity table を変更しない。

> `a_g` のような"温度非依存の群放射定数"は **定義しない**。  
> 黒体スペクトルの群分配は温度依存であり、必ず \(b_g(T)\) を介して扱う。

### 0.4 2D_RZにおけるIMC/DDMC粒子の幾何（3D粒子・2Dフィールド） [RETIRED — legacy IMC/DDMC; 現行輻射は §6.7 FLD / §6.8 \(S_N\)]

2D_RZモードでは、IMC/DDMC粒子はレーザーレイトレース（§5.5）と同一の
**「3D粒子・2Dフィールド」方式** を採用する。

**位置**：フル3D Cartesian座標 \((x, y, z)\)。
メッシュ参照は \(R=\sqrt{x^2+y^2}\) でRZ平面へ射影し、セル \((R, Z)\) を同定する。

**方向**：フル3D単位ベクトル \(\hat\Omega=(\Omega_x,\Omega_y,\Omega_z)\)。
等方サンプリング（§6.2）は3D全立体角に対して行う。

**直線輸送**（IMC）：
\[
\mathbf{r}(s)=\mathbf{r}_0 + s\,\hat\Omega \quad\text{（3D直線）}
\]
各ステップで \(R(s)=\sqrt{x(s)^2+y(s)^2}\) を計算し、2D RZフィールドを参照する。

**セル境界交差**（IMC追跡、§6.3.2(b)との整合）：
RZメッシュの面 \(k\)（頂点 \(\mathbf{V}_k(R_k,Z_k)\) と \(\mathbf{V}_{k+1}(R_{k+1},Z_{k+1})\) を結ぶ）は
3D空間では**円錐台面**（conical frustum）を形成する。
3D粒子軌跡とRZ面の交差は、以下の手順で計算する：

1. **R面**（r座標が変化する辺）：面がR方向に傾斜する場合、面のR(Z)を線形補間し、
   \(R(s)^2 = R_{face}(Z(s))^2\) を満たす \(s\) を二次方程式で解く
2. **Z面**（一定Z）：\(z_0 + s\,\Omega_z = Z_{face}\) から \(s=(Z_{face}-z_0)/\Omega_z\)
3. 各面の \(s>0\) の最小値を \(s_{bdry}\) とする

> **§6.3.2(b)の2D平面交差との関係**：
> §6.3.2(b)のパラメトリック交差式は、3D軌跡を \((R(s), Z(s))\) 平面に射影した
> 近似として成立する。RZ面が円錐台であることによる補正は、
> \(R(s)\) の非線形性（\(\sqrt{(x_0+s\Omega_x)^2+(y_0+s\Omega_y)^2}\)）に起因するが、
> セル内の移動距離が十分小さい場合（CFL制約下で保証）、
> \(R(s) \approx R_0 + s(\Omega_R)_{local}\) の線形近似で十分な精度が得られる。
> ここで \((\Omega_R)_{local} = (x\Omega_x + y\Omega_y)/R\) はRZ平面への射影方向余弦。
>
> v1.0では**線形近似**（§6.3.2(b)のパラメトリック交差）を使用し、
> 精密な二次方程式解法は将来の拡張とする。
> 線形近似の誤差は \(O((\Delta R/R)^2)\) であり、r=0付近を除き十分小さい。
> r=0付近では \(R_{floor} = 10^{-15}\) cm（幾何フロア）により \(\Omega_R = (x\Omega_x + y\Omega_y)/R\) の
> 0除算特異性を回避する。\(R < R_{floor}\) の場合は \(R = R_{floor}\) にクランプする。

**DDMCセル**：DDMCは方向を持たない（§7.5）ため、3D方向は不要。
リーク確率はRZ面に対して定義される（§7.3）。
DDMC→IMC変換（§7.7.2）時に3D出射方向をサンプルする：
- 面法線に対する天頂角をP(μ)分布（§7.7.1）からサンプル
- 方位角φを \([0,2\pi)\) で一様サンプル
- 3D方向ベクトルに変換（面法線座標系→Cartesian座標系）

> **根拠**：この方式はDRACO（fdm1290）の2D RZ IMC実装と同一であり、
> 軸対称幾何における回転体積要素 \(2\pi R\) を自然に処理する。
>
> **データ表現とARCHITECTURE §5.3 PhotonPoolとの対応**：
> - **1D_SPH**：位置はスカラー \(r\)（`pos_r` に格納、`pos_z = 0`）。
>   方向は3D単位ベクトル \((\Omega_r, \Omega_z, \Omega_\phi)\)（`dir_r, dir_z, dir_phi`）。
> - **2D_RZ**：位置は \((R, Z)\)（`pos_r, pos_z`）。
>   方向は3D単位ベクトル \((\Omega_r, \Omega_z, \Omega_\phi)\)（`dir_r, dir_z, dir_phi`）。
>   方位角 \(\phi\) は軸対称性により暗黙的であり、位置には格納しない。
>   §0.4冒頭のフル3D Cartesian \((x,y,z)\) は概念的な記述であり、
>   実装上の位置格納は \((R,Z)\) の2成分のみ。3D直線輸送の計算時に
>   \(R(s) = \sqrt{x(s)^2+y(s)^2}\) の評価が必要な場合は、
>   \(\phi\) を一時変数として保持する（PhotonPoolには格納しない）。
> - 転送フォーマット（§12.3.4 `ParticleEmigrant`）の `position[3]` は
>   `(pos_r, pos_z, 0)` としてパックされる。

**φ（方位角）の時間発展**：2D\_RZ モードでは粒子の3D位置を \((x, y, z) = (R \cos\phi, R \sin\phi, Z)\) で追跡する。\(\phi\) は以下の式で陰的に更新される：
\[
\phi^{new} = \text{atan2}(y + \Omega_y \Delta s,\; x + \Omega_x \Delta s)
\]
ただし直接 \(\phi\) を格納・更新するのではなく、\((x, y)\) を内部的に \((R, \phi)\) に射影して \(R = \sqrt{x^2+y^2}\) を保持する。
1ステップ内で複数セグメントを横断する場合、各セグメント終了時に \(R\) を再計算する。
\(\phi\) 自体は PhotonPool に格納しない（\(R\) と方向ベクトルから復元可能）。

---

### 0.5 v/c項の扱い（明示）
v1.0の放射輸送は **静止媒質（lab frame）** 形で、以下のO(v/c)項を無視する：
- 放射圧仕事 \(\mathbf{P}_r:\nabla\mathbf{u}\)
- ドップラー（周波数シフト）・アバレーション
- 速度依存不透明度

ICF典型で \(u/c\ll 1\) を根拠とする。差分要因としてメタデータに記録する。

---

## 1. 支配方程式（連続系）
### 1.1 流体（one‑fluid, 2T, ALE）
未知量：
- ρ：質量密度
- \(\mathbf{u}\)：流速
- \(e_i,e_e\)：比内部エネルギー（ion/electron）
- \(T_i,T_e\)：温度（EOSが \(e_k \leftrightarrow T_k\) を与える）
- \(P_i(\rho,T_i), P_e(\rho,T_e)\)
- Q：人工粘性圧

#### 1.1.1 質量保存（ラグランジュ形式）
\[
\frac{D\rho}{Dt} = -\rho \nabla\cdot \mathbf{u}
\]

#### 1.1.2 運動量
\[
\rho \frac{D\mathbf{u}}{Dt} = -\nabla(P_i + P_e + Q) + \mathbf{f}_{geom}
\]
> **v1.0**: 放射力（radiation pressure force）は含めない（§0.5 の O(v/c) 無視方針と整合）。
> 放射運動量沈着は診断出力のみ（§10.1.1）。

#### 1.1.3 エネルギー（2T）
**符号規約**：\(Q_{ei}>0\) を「電子→イオンへ流れるエネルギー（イオン加熱）」とする。

イオン：
\[
\rho\frac{De_i}{Dt} = -(P_i + Q)\nabla\cdot\mathbf{u} + Q_{ei} + S_i
\]
電子：
\[
\rho\frac{De_e}{Dt} = -P_e\nabla\cdot\mathbf{u} - \nabla\cdot\mathbf{q}_e - Q_{ei} + S_e + S_L + S_r
\]
- \(\mathbf{q}_e\)：電子熱流束 [erg/(cm²·s)]（§4参照）
- \(S_L\)：レーザー吸収 [erg/(cm³·s)]（電子へ、§5参照）
- \(S_r\)：輻射との交換 [erg/(cm³·s)]（IMC/DDMCの沈着として計上、§6–§7参照）
- \(S_i,S_e\)：追加源 [erg/(cm³·s)]（v1.0は0）

**電子–イオンエネルギー交換 \(Q_{ei}\)**（Spitzer–Braginskii）：
\[
Q_{ei} = \frac{C_{v,e}\,(T_e - T_i)}{\tau_{eq}} \quad [\text{erg/cm}^3/\text{s}]
\]
\[
\tau_{eq} = \frac{A\,m_p}{2\,m_e}\,\tau_e \quad [\text{s}]
\]
ここで \(\tau_e\) はNRL Plasma Formularyに基づく電子衝突時間：
\[
\tau_e = \frac{3.44\times 10^5\;T_e[\text{eV}]^{3/2}}{n_e[\text{cm}^{-3}]\;\bar{Z}\;\ln\Lambda_{ei}} \quad [\text{s}]
\]

数値的な平衡化時間（cgs+eV便利形式）：
\[
\tau_{eq} = \frac{3.16\times 10^8\;A\;T_e[\text{eV}]^{3/2}}{\bar{Z}^2\;n_i[\text{cm}^{-3}]\;\ln\Lambda_{ei}} \quad [\text{s}]
\]

**Constant-coefficient peer-review target (C2 feature-gap sentinel)**:
For a hypothetical verification-only mode with constant \(C_{v,e}\),
constant \(C_{v,i}\), and constant electron-ion coupling coefficient
\(\Lambda_{ei}\),
\[
\frac{dT_e}{dt}=-\frac{\Lambda_{ei}}{C_{v,e}}(T_e-T_i),\quad
\frac{dT_i}{dt}=+\frac{\Lambda_{ei}}{C_{v,i}}(T_e-T_i),
\]
the exact reference is
\[
\Delta T(t)=\Delta T(0)\exp\left[-\frac{C_{v,e}+C_{v,i}}
{C_{v,e}C_{v,i}}\Lambda_{ei}t\right],
\quad
T_\infty=\frac{C_{v,e}T_e(0)+C_{v,i}T_i(0)}
{C_{v,e}+C_{v,i}}.
\]
The finite-\(\Delta t\) conservative transfer is
\[
\Delta e_e =
\frac{C_{v,e}C_{v,i}}{C_{v,e}+C_{v,i}}(T_e-T_i)
\left(1-\exp[-\Delta t/\tau_{\rm eff}]\right),
\quad
\tau_{\rm eff}=\frac{C_{v,e}C_{v,i}}
{(C_{v,e}+C_{v,i})\Lambda_{ei}}.
\]
Current TENRYU production \(Q_{ei}\) does not expose this constant
\(\Lambda_{ei}\) mode; it uses the Spitzer/NRL temperature-dependent
\(\tau_{eq}(T_e,\rho,\bar Z,A)\) above. Therefore the C2 peer-review tests
`test_c2_exact_2t_relaxation`,
`test_c2_total_eit_conservation_stiff_coupling`, and
`test_c2_large_dt_implicit_positivity` are registered as feature-gap sentinels
until a constant-\(\Lambda_{ei}\) verification override is added.

> **符号検査**：\(T_e > T_i\) のとき \(Q_{ei}>0\)（電子→イオン）。✓
>
> **注意**：上式は \(T_e \gg (m_e/m_i)\,T_i\)（ICF典型条件）の近似。
> 一般の場合は \(\tau_{eq}\) の分子の \(T_e^{3/2}\) を
> \(\bigl(T_e/m_e + T_i/(A\,m_p)\bigr)^{3/2}\,m_e^{3/2}\) で置換する
> （Braginskii 1965）。


#### 1.1.3a 混合プラズマ電荷モーメント補正（TMAT ionization fractions、自動、2026-07-30）

単一有効種の \(\bar Z\) は多種イオン混合の衝突モーメントを系統的に誤る
（等モル CD 完全電離: \(\langle Z^2\rangle/\bar Z^2=1.51\)、
\(\langle Z^4\rangle/\bar Z^4=4.32\)）。TMAT 材料が `/ionization`
（元素別電離段分率）を持つ場合、config 構築時に 2 つの比表
\(r_2=\langle Z^2\rangle/\bar Z^2\)（クランプ [1,10]）と
\(r_4=\langle Z^4\rangle/\bar Z^4\)（クランプ [1,100]）へ縮約し、
毎 1D ステップ頭で per-cell 場 `zmom_r2`/`zmom_r4` を
\((n_i=\rho/(A_{eff}m_p),\ T_e)\) の log-log 双線形（端クランプ）で充填する。
**表が無ければ何も起きない**（場は未充填・カーネルは legacy 実体 —
false テンプレート実体はテキスト同一で bit 不変）。消費先と置換:
(i) §1.1.3 の \(\tau_{eq}\): \(\bar Z^2\to\bar Z^2 r_2\)、
(ii) §4.1 Spitzer: \(\bar Z\to\bar Z r_2\)（\(=Z_{\rm eff}\)）、
(iii) SNB の Z 補間因子: 同上、(iv) §3.1.13 Braginskii イオン粘性:
\(Z^4\to\bar Z^4 r_4\)。クーロン対数は全消費先で legacy 単一種形を保持
（対数的に弱い）。レーザー IB は §5.4.5(a) の `zeff_model`
（既定 "auto" = 表があれば table、なければ off）。制約 (v1): 1D_SPH 限定・
単一材料構成必須（volFrac 混合セルに per-material 帰属が無いため）・
提供材料はちょうど 1 つ（すべて config 構築時に検証、表が無い run は無検証）。
CBET の IAW 減衰は多イオン種の行列問題でモーメント置換の対象外（将来課題）、
hot-e 停止能は \(n_e\) 支配で対象外、輻射不透明度はテーブル由来で構成的に正しい。

#### 1.1.4 プラズマ基本量（\(n_e\), \(\bar{Z}\), クーロン対数）

**イオン数密度**：
\[
n_i = \frac{\rho}{A\,m_p} \quad [\text{cm}^{-3}]
\]

> **A の契約（2026-07-26 明文化, AI review k12 §5.6）**：`Materials.materials[*].A` は
> **イオン 1 個あたりの平均質量 [amu]** である（分子質量ではない）。化合物では
> number fraction \(x_j\) による平均 \(\bar A = \sum_j x_j A_j\) を与えること —
> 等原子数 CH は \(\bar A = 6.5\)（13 ではない）、CD は 7、等モル DT は 2.5、
> D\(_2\) は解離後の deuteron で 2（4 ではない）。IONMIX opacity の密度軸
> （総イオン数密度 \(n_i\)）との整合はこの契約に依存する。誤って分子質量を与えると
> \(n_i\) が整数倍ずれ、opacity/EOS のテーブル参照が系統的に誤る。
> TMAT-H5 は `/material/Abar_ion_amu` をファイル側で持つため parser が検証できるが、
> IONMIX 経路は deck 側の責任（parser は検証不能）。

**平均電離度 \(\bar{Z}\)**：v1.0では3モデルを提供する。
1. **fixed**（既定）：\(\bar{Z} = Z\)（ユーザ指定の原子番号 or 有効電荷。完全電離を仮定）
2. **thomas\_fermi**：More et al. (1988) のフィッティング公式
3. **tabular**：IONMIXテーブルから \(\bar{Z}(\rho, T_e)\) を補間取得

**Thomas–Fermiモデル**（More, Warren, Young & Zimmerman, Phys. Fluids 31, 3059, 1988）：

水素等価変数による無次元化：
\[
\rho^H = \rho/A,\quad T_e^H = T_e/Z^{4/3}
\]
圧力電離（冷）：
\[
\bar{Z}_0^H = \frac{\eta}{1+\eta},\quad \eta = \left(\frac{\rho^H}{0.148\;\text{g/cm}^3}\right)^{1/2}
\]
熱電離：
\[
\bar{Z}_{th}^H = (1-\bar{Z}_0^H)\,Y,\quad
Y = \left[1 + \left(\frac{T_0^H}{T_e^H}\right)^{1/2}\right]^{-1}
\]
\[
T_0^H = 0.0327\,\exp\!\bigl(6.98\,(\rho^H)^{0.075}\bigr) \quad [\text{eV}]
\]
合計：
\[
\bar{Z} = Z\,\bigl(\bar{Z}_0^H + \bar{Z}_{th}^H\bigr)
\]

> **適用範囲**：More et al. (1988) フィッティングの妥当性は
> \(\rho \in [10^{-3},\, 10^{4}]\) g/cm\(^3\)、\(T_e \in [0.01,\, 10^4]\) eV の範囲で確認されている。
> 範囲外では \(\bar{Z}\) をクランプする：\(\rho < 10^{-3}\) では \(\rho^H = 10^{-3}/A\)、
> \(\rho > 10^4\) では \(\rho^H = 10^4/A\)、\(T_e\) も同様に \([0.01, 10^4]\) eV でクランプ。
> \(Z\) はユーザ指定の **単一種の原子番号** である（混合材料セルではEOS混合則 §1.1.5(c) を適用）。
> 精度は～10%。生産計算ではIONMIXテーブルの \(\bar{Z}\) を推奨する。

**電子数密度**：
\[
n_e = \bar{Z}\,n_i = \frac{\bar{Z}\,\rho}{A\,m_p} \quad [\text{cm}^{-3}]
\]

**クーロン対数 \(\ln\Lambda_{ei}\)**（NRL Plasma Formulary準拠）：

\(T_e \ge 10\,\bar{Z}^2\) eV の場合（ICF典型）：
\[
\ln\Lambda_{ei} = 24 - \ln\!\left(\frac{\sqrt{n_e[\text{cm}^{-3}]}}{T_e[\text{eV}]}\right)
\]
\(T_e < 10\,\bar{Z}^2\) eV の場合：
\[
\ln\Lambda_{ei} = 23 - \ln\!\left(\sqrt{n_e[\text{cm}^{-3}]}\;\bar{Z}\;T_e[\text{eV}]^{-3/2}\right)
\]

下限値：
\[
\ln\Lambda_{ei} \ge \ln\Lambda_{\min} = 2 \quad (\text{SPECIFICATION §6.4.7 coulomb\_log\_floor})
\]

#### 1.1.5 状態方程式（EOS）

TENRYUは 2温度EOS（電子とイオンで独立の状態方程式）を持つ。

**(a) 理想気体（2T ideal gas）**

イオン：
\[
P_i = n_i\,k_B\,T_i = \frac{\rho\,k_B\,T_i}{A\,m_p},\quad
e_i = \frac{3}{2}\frac{k_B\,T_i}{A\,m_p},\quad
c_{v,i} = \frac{3}{2}\frac{k_B}{A\,m_p}
\]

電子：
\[
P_e = n_e\,k_B\,T_e = \frac{\bar{Z}\,\rho\,k_B\,T_e}{A\,m_p},\quad
e_e = \frac{3}{2}\frac{\bar{Z}\,k_B\,T_e}{A\,m_p},\quad
c_{v,e} = \frac{3}{2}\frac{\bar{Z}\,k_B}{A\,m_p}
\]

> **注**：\(\bar{Z}\) が温度依存（Thomas–Fermiモデル）の場合、
> \(c_{v,e} = \partial e_e/\partial T_e = (3k_B/2A m_p)(\bar{Z}+T_e\,d\bar{Z}/dT_e)\)
> とする。`fixed` \(\bar{Z}\) では上式で十分。
>
> **\(d\bar{Z}/dT_e\) の数値微分**：Thomas–Fermi モデル（§1.1.4）の \(\bar{Z}(\rho, T_e)\) は
> 解析的微分が煩雑であるため、中心差分で数値微分する：
> \[
> \frac{d\bar{Z}}{dT_e} \approx \frac{\bar{Z}(\rho,\, T_e + \delta T_e) - \bar{Z}(\rho,\, T_e - \delta T_e)}{2\,\delta T_e}
> \]
> ステップ幅：\(\delta T_e = \max(\varepsilon_{fd} \cdot T_e,\; T_{floor})\)、\(\varepsilon_{fd} = 10^{-4}\)。
> \(T_{floor}\) は §1.1.7 の温度フロア値（既定 \(10^{-3}\) eV）。
> テーブル端での片側差分は §1.1.6 の音速数値微分と同一の規則を適用する。
> `tabular` モデル（IONMIX \(\bar{Z}\) テーブル）でも同じ数値微分手法を用いる。

断熱指数：単原子理想気体として \(\gamma_e = \gamma_i = 5/3\)（無次元）。多原子分子や電離効果は \(\gamma\) に反映しない（v1.0の制約）。

**(b) テーブルEOS（SESAME / IONMIX / TMAT）**

TENRYU は **SESAME を既定テーブル EOS** として使用する。IONMIX/TMAT は代替オプション。
いずれの場合も GPU 上では同一の EOSTable 構造体で補間コードを共有する（SPECIFICATION §6.4.3 参照）。

**SESAME（既定）**：xSESAME ASCII 形式（80 文字固定幅、5E15.8）。
テーブル 301（total EOS）と テーブル 304（electron EOS）を読み込む。

単位変換（§0.1 定数表準拠）：
- \(T_{\text{eV}} = T_K \times k_{B,eV/K}\)（\(k_{B,eV/K} = 8.6174 \times 10^{-5}\) eV/K）
- \(P_{\text{dyne/cm}^2} = P_{\text{GPa}} \times 10^{10}\)
- \(e_{\text{erg/g}} = e_{\text{MJ/kg}} \times 10^{10}\)

2T 分離：テーブル 301 と 304 は**グリッドサイズが異なる**場合がある（例：Polystyrene 301=73×41, 304=63×33）。
形状の異なる配列の要素ごとの減算は不可（旧実装は 301 サイズの添字で 304 配列を読む
out-of-bounds だった — 2026-07-26 修正, AI review k11 C4）。実装（**build 時差分**）：
- `eos_total[mat]`: テーブル 301 の固有グリッドで EOSTable を構築
- `eos_e[mat]`: テーブル 304 の固有グリッドで EOSTable を構築（runtime の electron 参照はこのまま）
- イオン EOS は `build_sesame_ion_table` が **301 グリッド上で build 時に差分算出**：
\[
P_i(\rho_i,T_j) = P_{301}(\rho_i,T_j) - \text{interp}_{\log}(\text{eos\_e},\rho_i,T_j)
\]
\[
e_i(\rho_i,T_j) = e_{301}(\rho_i,T_j) - \text{interp}_{\log}(\text{eos\_e},\rho_i,T_j)
\]
  （\(\text{interp}_{\log}\) は runtime と同じ bilinear-log 補間・域外は端 clamp。
  301/304 が同一グリッド（サイズ・ノード値一致）のファイルは node-wise 直差分の
  fast path を通り、旧来挙動と bitwise 同値。）

> **負値ノードの扱い（2026-07-26 変更）**：旧設計の `max(…, 0)` positivity guard と
> `DeviceErrorFlags::eos_ion_negative` は**未実装のまま廃止**（当該 flag は実装に
> 存在しない）。負のイオン圧力/エネルギーのノードは clamp せずそのまま保持する —
> 比内部エネルギーの零点は 301/304 で異なりうるため負値はそれ自体では非物理でなく、
> clamp は人工プラトーを作って行単調性認証（`supports_rho_e_reclosure`、非単調行は
> table inverse を無効化し ideal 閉包へ fallback）と同一グリッド時の split 恒等式
> \(P_i+P_e=P_{total}\) を壊す。負値ノード数は build 時に 1 回 WARNING でログする。

304 不在時の 1T フォールバック（例：Deuterium mat 5265）：
\[
P_e = P_{total} \times \frac{\bar{Z}}{1+\bar{Z}},\quad e_e = e_{total} \times \frac{\bar{Z}}{1+\bar{Z}}
\]
\[
P_i = P_{total} - P_e,\quad e_i = e_{total} - e_e
\]
> 注：1T フォールバックでは減算が正確（P_e + P_i = P_total が代数的に保証される）ため、
> positivity guard は不要。

**xSESAME ASCII リーダー擬似コード**：
```
function read_sesame(filename, target_mat_id):
    open filename
    while not EOF:
        read record_header: (matId, tableId, nWords)
        if matId != target_mat_id:
            skip nWords values
            continue
        data = read nWords values in 5E15.8 format
        switch tableId:
            case 301:  // Total EOS
                (n_rho, n_T) = (int(data[0]), int(data[1]))
                rho_grid = data[2 : 2+n_rho]           // [g/cm³]
                T_grid_K = data[2+n_rho : 2+n_rho+n_T]  // [K]
                T_grid   = T_grid_K * k_{B,eV/K}        // → [eV]
                // xSESAME record layout after grids:
                //   offset_P = 2 + n_rho + n_T
                //   P_total[n_rho * n_T] at data[offset_P .. offset_P + n_rho*n_T)
                //   offset_e = offset_P + n_rho * n_T
                //   e_total[n_rho * n_T] at data[offset_e .. offset_e + n_rho*n_T)
                // Storage order: T-major (outer loop T, inner loop rho)
                //   flat index for (rho_i, T_j) = j * n_rho + i
                offset_P = 2 + n_rho + n_T
                P_total  = reshape(data[offset_P : offset_P + n_rho*n_T],
                                   (n_T, n_rho)) * 1e10       // GPa → dyne/cm²
                offset_e = offset_P + n_rho * n_T
                e_total  = reshape(data[offset_e : offset_e + n_rho*n_T],
                                   (n_T, n_rho)) * 1e10       // MJ/kg → erg/g
                // Validate: nWords == 2 + n_rho + n_T + 2*n_rho*n_T
                assert nWords == 2 + n_rho + n_T + 2*n_rho*n_T
            case 304:  // Electron EOS (grids may differ from 301)
                // Same layout as 301: (n_rho_e, n_T_e, rho_grid_e, T_grid_e, P_e, e_e)
                // n_rho_e, n_T_e may differ from 301's n_rho, n_T
                (n_rho_e, n_T_e) = (int(data[0]), int(data[1]))
                rho_grid_e = data[2 : 2+n_rho_e]
                T_grid_e   = data[2+n_rho_e : 2+n_rho_e+n_T_e] * k_{B,eV/K}
                offset_Pe = 2 + n_rho_e + n_T_e
                P_e = reshape(data[offset_Pe : offset_Pe + n_rho_e*n_T_e],
                              (n_T_e, n_rho_e)) * 1e10
                offset_ee = offset_Pe + n_rho_e * n_T_e
                e_e = reshape(data[offset_ee : offset_ee + n_rho_e*n_T_e],
                              (n_T_e, n_rho_e)) * 1e10
            case 502:  // Rosseland opacity (optional)
            case 505:  // Planck opacity (optional)
    return (eos_total, eos_e, opacity_R, opacity_P)
```

**IONMIX（代替）**：IONMIX v4/6 バイナリ .cn4 フォーマット（Fortran unformatted sequential）。

ファイル構造：
1. ヘッダスカラー: `ntemp` (float64), `ndens` (float64)
2. 組成配列: `Z_values[]` (float64), `fractions[]` (float64) — 可変長
3. ヘッダスカラー: `ngroups` (float64)
4. 温度グリッド: `temps_eV[ntemp]` (float64) [eV]
5. 密度グリッド: `numdens_cm3[ndens]` (float64) — **イオン数密度** \(n_i\) [cm\(^{-3}\)]
6. 12個の EOS 2Dブロック: 各 `[ndens × ntemp]`（密度メジャー、温度最速）
   - Block 1: \(\bar{Z}\)（平均電離度、無次元）
   - Block 2: \(d\bar{Z}/dT\)（未使用）
   - Block 3: \(P_i\) [J/cm³] → \(\times 10^7\) → [dyne/cm²]
   - Block 4: \(P_e\) [J/cm³] → \(\times 10^7\) → [dyne/cm²]
   - Block 5–6: 導関数（未使用）
   - Block 7: \(e_i\) [J/g] → \(\times 10^7\) → [erg/g]
   - Block 8: \(e_e\) [J/g] → \(\times 10^7\) → [erg/g]
   - Block 9–12: 導関数（未使用）
7. 群境界: `bounds_eV[ngroups+1]` (float64) [eV]（IONMIX6形式では追加エントロピーブロックが挿入される場合あり）
8. 不透明度テーブル: `kappa_R`, `kappa_PA` 各 `[ngroups × ndens × ntemp]` (float64) [cm²/g]
9. Planck emission（任意）: `kappa_PE[ngroups × ndens × ntemp]` (float64) [cm²/g]（不在時は LTE フォールバック: \(\kappa^{PE} = \kappa^{PA}\)）

密度グリッド変換：EOSTable の密度軸は質量密度 \(\rho\) [g/cm³] を使用するため、
\[
\rho_i = n_{i,i} \times A \times m_p
\]
で変換する。

インデックス転置：IONMIX は密度メジャー \(\text{idx} = d \times n_T + t\) で格納するが、
EOSTable は温度メジャー \(\text{flat\_index}(i_\rho, j_T) = j_T \times n_\rho + i_\rho\) を使用する。
ロード時に転置を行う。

IONMIX/TMAT では電子・イオンを直接構築した 3 テーブルを生成する：
- `eos_ion`: \(P = P_i\)、\(e = e_i\)
- `eos_e`: \(P = P_e\)、\(e = e_e\)
- `eos_total`: \(P = P_i + P_e\)、\(e = e_i + e_e\)

SESAME は total + electron のみを提供するため、ion は差分で構築する：
\[
P_i = P_{total} - P_e,\quad e_i = e_{total} - e_e
\]

> **IONMIX v4 ASCII フォーマット**（レガシー）：トークンベースの ASCII リーダーも存在するが、
> バイナリ .cn4 リーダーを推奨する。ASCII リーダーは将来削除予定。

**TMAT-H5（v1.0）**：
- `/eos/@primary_density_axis` と `/opacity/@primary_density_axis` は `"ni_cm3"` を使用する。
- 密度グリッドは `/eos/grid/ni_cm3` と `/opacity/grid/ni_cm3` に格納される。
- `tmat_eos_to_table_triplet()` と `tmat_eos_to_zbar_table()` では `/eos/grid/ni_cm3` を
  \(\rho = n_i A m_p\) で質量密度 [g/cm³] に変換してから EOS/Zbar テーブルを構築する。
- `tmat_to_ionmix_opacity()` では `ni_cm3` を `IonmixOpacityData.numdens_cm3` に直接コピーし、`A m_p` による変換は行わない。

**共通仕様**：

テーブルから \((\rho, T_k)\) を引数に：
- \(e_k(\rho, T_k)\)：比内部エネルギー [erg/g]
- \(P_k(\rho, T_k)\)：圧力 [dyne/cm²]
- \(c_{v,k}(\rho, T_k) = \partial e_k/\partial T_k\big|_\rho\)：比熱 [erg/(g·eV)]

補間：\((\log\rho, \log T)\) 空間で双線形補間。テーブル範囲外はフロア値（§1.1.7）にクランプ。

テーブルEOSから得た \(c_v\) が非正（\(c_{v,k} \le 0\)）の場合は \(c_{v,floor} = 10^{-3}\) erg/(g·eV) にクランプし WARNING を出力する。非正の \(c_v\) は \(Q_{ei}\)（§1.1.3）や音速（§1.1.6）で 0除算や符号反転を引き起こすため、フロア適用は必須である。

> テーブルは Materials API（ARCHITECTURE §4.3）のみがアクセスし、
> 他モジュールは \(e_k, P_k, c_{v,k}\) を受け取る（設計制約）。

**E1 production closure verification (2026-05-13)**:
- `test_e1_manufactured_tmat_tables.cu` verifies CPU `EOSTable` bilinear \((\log\rho,\log T)\) interpolation against manufactured analytic \(P,e,c_v\) tables at grid nodes (≤ \(10^{-13}\) relative) and cell log-midpoints (≤ \(10^{-10}\) relative). The opacity portion uses a positive manufactured `IonmixOpacityData` table and verifies `interpolate_kappa()` at the same tolerances; production synthetic TMAT opacity contains floor-aware zero entries, so it is not used as the double-precision manufactured oracle.
- `test_e1_thermodynamic_consistency_cv_de_dT.cu` checks production CD TMAT ion/electron/total tables against \(c_v \approx \partial e/\partial T|_\rho\) at interior \((\rho,T)\) samples with a 5% relative gate. Flat low-temperature energy samples where the production table stores the positive \(c_v\) floor are reported as a documented table-construction caveat.
- `test_e1_ionization_multi_phase_transitions.cu` sweeps production CD \(\bar Z(\rho,T)\) through L-shell and K-shell transition ranges, requiring finite values, \(0\le\bar Z\le Z_{\max}\), monotone non-decrease in \(T\), finite EOS \(P/e/c_s\), and ≥0.5 rise across each transition band. Solid/liquid/gas phase boundaries are not exposed in the current CD TMAT schema and remain a feature gap outside this ionization gate.
- `test_e1_cgs_ev_unit_conversion_regression.cu` pins TENRYU's production cgs+eV constants `eV_to_erg`, `k_B_eV_per_K`, `proton_mass`, `c_light`, `a_eV` as a regression sentinel (bit-exact silent-change detection per CLAUDE.md §2). Any change to these values requires an explicit NUMERICS.md update + PR per §0 rule 1. CODATA-precision values are documented but NOT enforced here; tightening to full CODATA is a separate physics-constant PR.

**Current GPU table-EOS implementation (DeviceEOSTableView)**:
- Interpolation is bilinear in \((\log \rho, \log T)\), with bracket search on each axis and clamping to table bounds.
- `find_rho_bracket(tab, rho)` computes \((i_0,i_1,w_\rho)\) once per cell; the same `RhoBracket` is reused for:
  - \(T(\rho,e)\) inversion,
  - \(P(\rho,T)\), \(e(\rho,T)\), and \(c_v(\rho,T)\) evaluation.
- This `RhoBracket` reuse removes redundant density-axis searches in closure/sound-speed kernels.

**Zero-iteration \(T(\rho,e)\) inversion (monotone search)**:
- Instead of Newton iteration, the device path performs a monotone binary search in the temperature index on the mixed row energy
  \[
  e_j(\rho) = (1-w_\rho)e_{j,i_0} + w_\rho e_{j,i_1}.
  \]
- It finds \(j_0,j_1\) such that \(e_{j_0} \le e_{target} < e_{j_1}\), linearly interpolates in \(\log T\), and returns \(T=\exp(\log T)\).
- Out-of-range \(e_{target}\) is clamped to table-end temperatures. This is a zero-Newton-iteration inverse map.

**Low-density analytic extrapolation policy (opt-in)**:
- The predicate `eos_use_low_density_extrapolation(tab, flag_enabled, rho)` is true only when the caller enables the flag and
  the positive density is below the table lower bound in log space, with a roundoff margin at the boundary:
  \[
  \log\rho < \log\rho_{\min}.
  \]
  Therefore \(\rho = \rho_{\min}\) remains in the normal table-clamped path.
- Forward device wrappers `device_eos_pressure_extrap`, `device_eos_energy_extrap`, and `device_eos_cv_extrap` replace the table
  lookup below \(\rho_{\min}\) with the nondegenerate ideal-electron EOS used in §1.1.5(a). With
  \(Z_{\rm eff}=\bar{Z}\) if \(\bar{Z}>0\) and \(1\) otherwise, \(A_{\rm eff}=A\) if \(A>0\) and \(1\) otherwise, and
  \(T_{\rm eff}=\max(T,10^{-30}\,\mathrm{eV})\), the device formulas are:
  \[
  c_{v,e} = \frac{3}{2}\frac{Z_{\rm eff}\,eV_{\rm erg}}{A_{\rm eff}m_p},\qquad
  e_e = c_{v,e}T_{\rm eff},\qquad
  P_e = \frac{2}{3}\rho e_e.
  \]
- `device_inverse_reclose` is unchanged: it always uses the table monotone inverse and table-clamped forward reclosure.
- Callers that explicitly need the same low-density analytic policy for inverse reclosure use
  `device_inverse_reclose_with_low_density_extrap(tab, rho, e_target, T_floor, Zbar, A_amu, flag_enabled)`. When the predicate
  is false, this helper returns `device_inverse_reclose` exactly. When the predicate is true, it inverts the same analytic
  ideal-electron EOS:
  \[
  T = \max\left(\frac{e_{target}}{c_{v,e}},\,T_{floor},\,10^{-30}\,\mathrm{eV}\right),
  \]
  then recomputes \(e_e\), \(P_e\), and \(c_{v,e}\) using the forward extrapolation formulas above. The helper is opt-in;
  2D_RZ ALE post-remap reclosure uses it when `HydroEOSContext` provides table-backed EOS.

**EOS 閉包のエネルギー方針（`Numerics.hydro.eos_closure_mode`、2026-07-18 BUG-24）**:
1D の hydro 入口/出口 EOS 閉包（1T/2T の table 分岐と persistent 経路）は、table 逆算
`device_inverse_reclose` が lower/upper clamp（目標 \(e\) が当該 \(\rho\) 行の表域外）を報告した
セルについて、従来（`"legacy"`、bit 凍結・旧既定）は内部エネルギーを clamp 後の表値
\(e_{\rm tab}\) で無条件に上書きしていた。laser 加熱で table 温度天井を超えた blowoff コロナ
セルではこれが**超過エネルギーの毎ステップ無記帳破棄**になる（GXII 級 solid FLD 実測で累積
−362 J = 吸収の 21%、docs/design/bug24_hydro_entry_eos_projection_20260718.md）。
`"energy_authoritative"`（2026-07-18 ユーザー裁定以降の**既定**、production 再基準化済み —
`config.hpp` の default と一致）は進化させた \(e\) を権威量として保持し、clamp セルでは
`eos_writeback` の roundtrip 上書きも含め \(e\) を書き換えない（\(T,P,c_v\) は clamp 逆算値 =
表端評価のまま）。非有限入力の repair と bracket 失敗の安全修復は両モードで不変。
最終状態（2026-07-18、ファミリ閉鎖）: 同ノブ下で table 温度天井の単調 ideal-tail 拡張
（低密度 extrap の温度版）を hydro 閉包の逆算・FLD matter Newton（template
\(\langle\)EOS_TAIL\(\rangle\)）・exp source kernel・SN matter Newton・伝導 handoff
（増分 \(e{+}{=}c_v\Delta T\) + tail 逆算 = 保存形）へ一貫適用し、full-run 未計上損失は
**fleck 0.9% / exp_rosenbrock 0.3% / SN 2.9%（対吸収; 修正前は各 ~20%）**。exp kernel は
入口を状態読み（\(e_n=e_e\)）・出口を増分形とし再基底化注入を除去。fleck の増分形出口のみ
dt 推定暴走のため射影形を維持（+17.5 J/2.5 ns の文書化残差）。OFF-bit 再認証 10/10
（GXII golden・fleck 0-D a–k・SN×4・Marshak feature・namelist）。QA 標準: 累積保存は
per-step `dE_total` の総和ではなく端点閉包式 \((E_0+\sum\text{src}-E_{\rm end})-\sum\text{esc}\)
で評価する（per-step 系列の cumsum は縫い目二重計上で 5–10× 過大）。2D_RZ の閉包は
別構造のため本ノブの対象外（BUG-24 doc の relay 節参照）。

> **追補（2026-07-26, AI review k11 C2）**: 1D ALE の post-remap EOS reclosure
> （`ale_1d_driver.cu::eos_reclosure_kernel`）は上記ファミリ閉鎖から漏れており、
> closure mode にかかわらず常に表射影を行っていた（super-ceiling 超過の無記帳破棄・
> 表下端 clamp の無記帳注入）。修正済み: `energy_authoritative` では hydro 閉包と同じ
> clamp-veto（進化 \(e\) を保持、非有限/負の raw 入力の repair は veto を迂回）+
> 温度天井の ideal-tail 逆算を適用し、`legacy` は従来挙動を bit 保存する。
> T-floor 注入は両モードで `State::E_floor_injected`（retry rollback 対象の正典 ledger）
> へ計上する — 従来この kernel の ledger 引数は唯一の呼び出し点で `nullptr` に
> 配線されており死んでいた。2D_RZ 側の同型経路（`ale_axis_band_controller`）は
> 未修正（2D lane への relay）。

[BUG-26, 2026-07-20] The 1T branch of the driver EOS initialization computed
\(e_e\) from the ideal-gas cv even for table-EOS materials (power_law_te /
TMAT), while the 2T branch consults the table; the conduction-side
`sync_ee_from_Te_table` never runs at init when conduction is disabled. Every
1T table deck therefore started EOS-inconsistent (measured: \((c_v^e+c_v^i)T_e
= 2.874\times10^{10}\) vs table \(5.34\times10^{13}\) erg/g in the
Marshak-feature deck). The legacy closure silently rewrote \(e_e\) to the
table value during the first radiation step (an invisible, budget-baseline-
predating phantom injection), while `eos_closure_mode="energy_authoritative"`
propagated the wrong init — \(T_e\) collapsing to the table inverse of the
ideal \(e_e\) (1000 eV \(\to\) 11.36 eV) and retarding the Marshak-feature
front by the dt-independent \(\sim\)3.75 cells recorded at the 2026-07-18
freeze. The 1T init now uses the same total-table accessors as the runtime
sync (the analytic `eos_T_ref_eV`+`cv_e_override` precedence and the
ideal-gas fallback are unchanged). Verification: the exp\(\times\)EA front
snapped back to the legacy/fleck position (131.50 vs 127.50 cells), all
certified gates re-passed (GXII golden, Hammer–Rosen, RMtV, freq-dep
relaxation, fleck_relaxation_0d 36 legs, verify_marshak_feature_1d), and the
Marshak-feature gate now certifies under the production-default closure.

**Hydro-only exact ideal-gas diagnostic backend (`eos.hydro_backend="exact_ideal_gas"`)**:
- 1D_SPH 限定の診断 backend。raw ion/electron/total table を upload したまま、Hydro kernel 内の EOS closure / sound speed のみを解析 ideal gas に置き換える。
- 2T closure では material 定数 \(A, \gamma\) とセルごとの \(\bar{Z}\) を用いて
  \[
  c_{v,i} = \frac{k_B}{A\,m_p\,(\gamma-1)}, \qquad
  c_{v,e} =
  \begin{cases}
    \text{cv\_e\_override}/\rho & (\text{cv\_e\_override} > 0) \\
    \dfrac{\bar{Z}\,k_B}{A\,m_p\,(\gamma-1)} & (\text{otherwise})
  \end{cases}
  \]
  \[
  T_i = e_i/c_{v,i},\qquad T_e = e_e/c_{v,e},\qquad
  P_i = (\gamma-1)\rho e_i,\qquad P_e = (\gamma-1)\rho e_e
  \]
  を直接評価する。1T closure では \(e=e_e+e_i\), \(T=e/(c_{v,i}+c_{v,e})\), \(P=(\gamma-1)\rho e\) を使う。
- sound speed も table 導関数ではなく
  \[
  c_s^2 = \gamma\frac{P_e+P_i}{\rho}
  \]
  （1T では \(c_s^2 = \gamma(\gamma-1)e\)）を直接使う。
- Hydro state を table cold-curve energy に引き戻さないよう、初期 `Te/Ti → ee/ei/Pe/Pi` 生成と、
  conduction / source-term 後の `Te/Ti → ee/ei/Pe/Pi` 再閉包も同じ ideal-gas 関係式へ投影する。
- 目的は、table-hydro kernel 本体や周辺 operator は変えずに、EOS closure のみを厳密 ideal gas へ置換する決定的診断である。

**Hydro-only rho-e table backend (`eos.hydro_backend="rho_e_table"`)**:
- 1D_SPH 限定の hydro-only backend。初期化時に raw `total` EOS `P(\rho,T), e(\rho,T)` から
  hydro 専用の `P(\rho,e_{total}), T(\rho,e_{total})` table を CPU で事前構築する。
- 既定 (`Numerics.hydro.rho_e_linear_grid=False`) では、\(\rho\) 軸は raw table と同じ
  \(\log\rho\) grid を使い、\(e\) 軸は
  \[
  e_{min} = \min_i e(\rho_i, T_{min}),\qquad
  e_{max} = \max_i e(\rho_i, T_{max})
  \]
  で定めた全 row union を 200 点の一様 \(\log e\) grid に再標本化する。
- 診断用に `Numerics.hydro.rho_e_linear_grid=True` を指定すると、
  \(\rho\) 軸は raw table の元の `rho_grid` をそのまま線形 \(\rho\) 座標として使い、
  \(e\) 軸は同じ \([e_{min}, e_{max}]\) を 200 点の一様線形 \(e\) grid に再標本化する。
- 各 \((\rho_i,e_j)\) node では、raw `total` table に対して \(e(\rho_i,T)=e_j\) を
  monotone bisection で反転し、その \(T\) を使って \(P(\rho_i,T)\) を取得する。
- その後、`P(\rho,e)` と `T(\rho,e)` の各 field について
  `rho_e_linear_grid=False` では \((\log\rho,\log e)\) 面、
  `rho_e_linear_grid=True` では \((\rho,e)\) 面の
  **C² tensor-product natural cubic spline** を構築する。
  CPU では row/column の natural cubic solve から
  \[
  (P_{xx}, P_{yy}, P_{xxyy}), \qquad (T_{xx}, T_{yy}, T_{xxyy})
  \]
  を事前計算して GPU へ転送する。
- Hydro の total EOS は runtime で `e→T→P` chain を行わず、
  選択された座標系の natural cubic spline で
  \[
  P = P_{rhoe}(\rho,e), \qquad T = T_{rhoe}(\rho,e)
  \]
  を直接評価する。
- 1T closure では `total` の \(P,T,c_v,c_s\) をこの table から直接取得する。2T closure では
  \(T_i,T_e,e_i,e_e,c_{v,i},c_{v,e}\) は従来どおり raw ion/electron table の inverse map を使い、
  hydro に効く total pressure と sound speed のみを `rho_e_table` に置き換える。
- 既定では `Numerics.hydro.eos_writeback=True` のため、1T total closure の
  \(T(\rho,e)\) / \(P(\rho,e)\) 評価後は `rho_e_table` の clamped \(e\) を state へ writeback し、
  旧来の `e \leftarrow e_{table}(\rho,T)` を行う。`eos_writeback=False` では Hydro が更新した
  total \(e\) を保持し、NaN / Inf / 負の \(e\) だけを repair として clamped \(e\) へ戻す。
- 2T の圧力分割は raw path の \((P_i^{raw},P_e^{raw})\) 比を保持し、
  \[
  P_i + P_e = P_{rhoe,total}(\rho,e_i+e_e)
  \]
  となるよう scale を掛けて再正規化する。raw 合圧が極小/非有限なら
  \(P_i=0,\;P_e=P_{rhoe,total}\) へフォールバックする。
- 数値安全のため、spline 評価で \(P \le 0\) または \(c_s^2 \le 0\) になった場合は
  それぞれ微小正値へクランプする。さらに \(\partial T/\partial e|_\rho \le 0\) または
  \(\partial P/\partial e|_\rho \le 0\) の場合は \(c_v\) を floor 値へ落とす。
- `Numerics.hydro.exact_override` は 1D 診断専用の A/B 分離スイッチで、table backend
  (`legacy`, `helmholtz_spline`, `helmholtz_jet`, `rho_e_table`) の EOS closure 後に
  `pressure`, `sound_speed`, `temperature` のいずれか 1 量だけを
  \(\gamma=5/3\), \(c_{v,i}=1.5\,k_B/(A\,m_p)\), \(c_{v,e}=1.5\,Z\,k_B/(A\,m_p)\)
  の ideal-gas 式で上書きする。既定 `none` では無効。TMAT 物理との整合性はなく、
  合成 ideal-gas table の診断以外には使わない。

**Hydro-only Helmholtz spline backend (`eos.hydro_backend="helmholtz_spline"`)**:
- backend 名は互換性のため `helmholtz_spline` のままだが、実装は hydro 用の
  `total` EOS に対する \(P(\log\rho,\log T)\), \(e(\log\rho,\log T)\) の
  **shape-preserving C¹ tensor-product bicubic Hermite** surrogate である。
- GPU には各 field のノード量
  \[
  (P,\ P_x,\ P_y,\ P_{xy}), \qquad
  (e,\ e_x,\ e_y,\ e_{xy})
  \]
  を転送する。
- Hydro が参照する total 熱力学量は spline から一括導出する：
  \[
  P = P_{spline}, \qquad
  e = e_{spline}, \qquad
  c_v = \frac{1}{T}\,\frac{\partial e_{spline}}{\partial \ln T}
  \]
  \[
  \left.\frac{\partial P}{\partial \rho}\right|_T
  = \frac{1}{\rho}\,\frac{\partial P_{spline}}{\partial \ln\rho}, \qquad
  \left.\frac{\partial P}{\partial T}\right|_\rho
  = \frac{1}{T}\,\frac{\partial P_{spline}}{\partial \ln T}
  \]
- 1T path では `total` surrogate をそのまま用いる。2T path では
  \(T_i,T_e,e_i,e_e,c_{v,i},c_{v,e}\) の個別 bookkeeping は legacy raw-table EOS を維持し、
  hydro に効く total pressure \(P_e+P_i\) と sound speed のみを `total` surrogate へ置き換える。
- 既定では `Numerics.hydro.eos_writeback=True` のため、1T total closure 後は
  surrogate 側の clamped \(e(\rho,T)\) を state へ writeback する。`eos_writeback=False` では
  Hydro が更新した total \(e\) を保持し、NaN / Inf / 負の \(e\) だけを repair として
  clamped \(e(\rho,T)\) へ戻す。2T の個別 \(e_i,e_e\) も raw inverse map で \(T_i,T_e\) を
  閉じるが、`eos_writeback=False` のときのみ re-projection を行わず、NaN / Inf / 負のときだけ
  repair する。
- 2T closure では、legacy raw-table から得た \((P_i^{raw}, P_e^{raw})\) の比を保持したまま
  scale を掛け、\(
  P_i + P_e = P_{spline,total}
  \) を満たすように再正規化する。raw-table 合圧が極小/非有限の場合は
  \(P_i=0,\;P_e=P_{spline,total}\) へフォールバックする。
- **適用範囲**：Hydro closure / sound speed のみ。Radiation / opacity / Zbar の raw-table path は変更しない。
- 1D 導関数は Fritsch-Carlson monotone Hermite limiter で構築し、ゼロクロスや低温端のオーバーシュートを抑制する。したがって \(C^2\) ではなく \(C^1\) 連続である。

**Hydro-only Helmholtz jet backend (`eos.hydro_backend="helmholtz_jet"`)**:
- `total` EOS の raw table から \(\phi = F/T\) の **local projected jet** を各ノードで構築し、
  各セルを **biquintic Hermite patch** で補間する hydro 専用 backend である。
- ノードジェットは
  \[
  \phi_x = \frac{P}{\rho T}, \qquad \phi_y = -\frac{e}{T}
  \]
  を raw table から exact に設定し、\(\phi_{xx}, \phi_{yy}, \phi_{xy}\) は
  \((\log\rho,\log T)\) 面での有限差分から与える。
- 局所投影で
  \[
  c_v = -(\phi_y + \phi_{yy}) > 0, \qquad
  \left.\frac{\partial P}{\partial \rho}\right|_T = T(\phi_x+\phi_{xx}) > 0
  \]
  をノードごとに満たすよう \(\phi_{yy}, \phi_{xx}\) を最小限 clamp する。
- \(\phi\) 自体は \(\phi_x,\phi_y\) の x/y 積分を 2 経路で行い、その平均で gauge を固定する。
- 各セルでは 4 隅の
  \[
  (\phi,\phi_x,\phi_y,\phi_{xx},\phi_{yy},\phi_{xy})
  \]
  から tensor-product biquintic Hermite patch を組み、未拘束の高次混合項は 0 とする。
- Hydro が使う量は patch から直接導出する：
  \[
  P = \rho T \phi_x,\qquad
  e = -T\phi_y,\qquad
  c_v = -(\phi_y+\phi_{yy})
  \]
  \[
  \left.\frac{\partial P}{\partial \rho}\right|_T = T(\phi_x+\phi_{xx}),\qquad
  \left.\frac{\partial P}{\partial T}\right|_\rho = \rho(\phi_x+\phi_{xy})
  \]
- 1T path では `total` jet surrogate をそのまま用いる。2T path では
  \(T_i,T_e,e_i,e_e,c_{v,i},c_{v,e}\) の個別 bookkeeping は legacy raw ion/electron table を維持し、
  hydro に効く total pressure \(P_e+P_i\) と sound speed のみを jet surrogate へ置き換える。
- 既定では `Numerics.hydro.eos_writeback=True` のため、1T total closure 後は
  jet surrogate 側の clamped \(e(\rho,T)\) を state へ writeback する。`eos_writeback=False`
  では Hydro が更新した total \(e\) を保持し、NaN / Inf / 負の \(e\) だけを repair として
  clamped \(e(\rho,T)\) へ戻す。2T の個別 \(e_i,e_e\) も raw inverse map で \(T_i,T_e\) を
  閉じるが、`eos_writeback=False` のときのみ re-projection を行わず、NaN / Inf / 負のときだけ
  repair する。
- 2T closure の pressure split は `helmholtz_spline` backend と同じで、
  raw \((P_i^{raw},P_e^{raw})\) 比を保持したまま scale を掛けて
  \(P_i+P_e=P_{jet,total}\) を満たす。raw 合圧が極小/非有限なら
  \(P_i=0,\;P_e=P_{jet,total}\) にフォールバックする。
- **適用範囲**：Hydro closure / sound speed のみ。Radiation / opacity / Zbar の raw-table path は変更しない。

**Experimental CPU Helmholtz B-spline fit (Phase 1, not runtime-active)**:
- `src/materials/helmholtz_bspline.cpp` は raw `total` EOS table に対して
  \[
  \phi(x,y) = \frac{F(\rho,T)}{T}, \qquad x=\ln\rho,\ y=\ln T
  \]
  の quintic tensor-product B-spline least-squares fit を構築する CPU-only utility である。
- fit で用いる定義式は
  \[
  P = \rho T \phi_x,\qquad e = -T\phi_y,\qquad
  c_v = -(\phi_y+\phi_{yy})
  \]
  \[
  \left.\frac{\partial P}{\partial \rho}\right|_T = T(\phi_x+\phi_{xx}),\qquad
  \left.\frac{\partial P}{\partial T}\right|_\rho = \rho(\phi_x+\phi_{xy})
  \]
  であり、そこから
  \[
  c_s^2 =
  \left.\frac{\partial P}{\partial \rho}\right|_T +
  T\frac{\left(\partial P/\partial T|_\rho\right)^2}{\rho^2 c_v}
  \]
  を評価する。
- 行列は raw `P,e` データの相対重み付き normal equations を局所 support から直接組み立てる。
  knot stride は option 化されており、現 default は full raw grid (`1,1`) である。
  必要なら reduced knot grid へ間引ける。
  さらに \(\psi=\phi/\phi_{scale}\) with
  \[
  \phi_{scale} = \max\!\left(1,\ \max\left|\frac{P}{\rho T}\right|,\ \max\left|\frac{e}{T}\right|\right)
  \]
  により内部 unknown を無次元化する。目的関数は raw `P,e` fit に加えて、
  monotone Hermite reference surrogate から得た
  \[
  \left.\frac{\partial P}{\partial \rho}\right|_T,\qquad c_v
  \]
  を weak reference rows として与え、さらに係数二次差分 smoothing と 1 個の gauge constraint を加える。
- `c_v > 0` および \(\partial P/\partial \rho|_T > 0\) は iterative penalty ではなく
  active-set QP 近似で課す。まず不等式制約なしの LS 解を求め、その後
  \[
  c_v \ge \varepsilon_v,\qquad \left.\frac{\partial P}{\partial \rho}\right|_T \ge \varepsilon_\rho
  \]
  を最も強く破っているノードを 1 つずつ active set に追加し、対応する等式制約を
  KKT 系
  \[
  \begin{bmatrix}
    H & G_A^T \\
    G_A & 0
  \end{bmatrix}
  \begin{bmatrix}
    c \\ \lambda
  \end{bmatrix}
  =
  \begin{bmatrix}
    b \\ h_A
  \end{bmatrix}
  \]
  として解く。負の Lagrange multiplier を持つ active 制約は解放し、双対可行性と
  原始可行性がそろうまで反復する。
- 現段階では runtime hydro backend へは未接続で、fit diagnostics のみを提供する。

**(c) 混合材料セルのEOS**

多材料セルでは全材料が同一 \((\rho, T_e, T_i)\) を共有する（single-state仮定）。
各材料 \(\alpha\) は独自の \((A_\alpha, Z_\alpha)\) を持ち、EOS量 \(e_\alpha, P_\alpha, C_{v,\alpha}\) は
それぞれの材料パラメータで評価する（理想気体では §1.1.5(a) の式に \(A_\alpha, \bar{Z}_\alpha\) を適用、
テーブルでは材料ごとのIONMIXファイルから取得）。
合成量は質量分率 \(f_{m,\alpha}\) による加重平均：
\[
e = \sum_\alpha f_{m,\alpha}\,e_\alpha,\quad
P = \sum_\alpha f_{m,\alpha}\,P_\alpha,\quad
C_v = \sum_\alpha f_{m,\alpha}\,C_{v,\alpha}
\]

**混合材料セルにおけるプラズマ基本量**（§1.1.4 の拡張）：

多材料セルでは、イオン数密度・電子数密度・平均電離度を以下のように材料加重で計算する：
\[
n_i = \sum_\alpha \frac{f_{m,\alpha}\,\rho}{A_\alpha\,m_p},\qquad
n_e = \sum_\alpha \frac{\bar{Z}_\alpha\,f_{m,\alpha}\,\rho}{A_\alpha\,m_p}
\]
セルの有効平均電離度：
\[
\bar{Z}_{eff} = \frac{n_e}{n_i} = \frac{\sum_\alpha \bar{Z}_\alpha\,f_{m,\alpha}/A_\alpha}{\sum_\alpha f_{m,\alpha}/A_\alpha}
\]
セルの有効原子量（調和平均）：
\[
\frac{1}{A_{eff}} = \sum_\alpha \frac{f_{m,\alpha}}{A_\alpha},\qquad
A_{eff} = \left(\sum_\alpha \frac{f_{m,\alpha}}{A_\alpha}\right)^{-1}
\]
これらの \(n_i, n_e, \bar{Z}_{eff}, A_{eff}\) は EOS 混合量（\(e,P,C_v\)）と
プラズマ基本量の導出に用いる。
電子熱伝導とソース結合で用いるセル実効量（\(A_{eff}, \gamma_{eff}, n_e, c_{v,e}, c_{v,i}\)）は
§1.1.5a の体積分率混合則を用いる。
単一材料セルでは \(f_{m,1}=1\) であり、上式は §1.1.4 の定義に帰着する。

**Void 材料の混合則除外**：`is_void = true` の材料は \(\bar{Z}_{eff}\)、\(A_{eff}\)、
EOS 混合平均（\(e, P, C_v\)）、および質量分率 \(f_{m,\alpha}\) の計算から除外する。
Void 材料の体積分率は `cell_is_void` マスクの導出にのみ使用される。
`cell_is_void = 1` のセルでは \(\bar{Z} = 0\) を強制し、Thomas-Fermi モデルの
評価をスキップする（実装: `geometry_eval.cpp` の Zbar 計算ループ）。

**不透明度の混合則**（SPECIFICATION §6.4.3 `opacity_mix_rule` 参照）：

- **Planck不透明度**（吸収評価用）：質量加重平均
\[
\kappa_{P,mix} = \sum_\alpha f_{m,\alpha}\,\kappa_{P,\alpha}
\]
- **Rosseland不透明度**（拡散/リーク評価用）：調和平均
\[
\frac{1}{\kappa_{R,mix}} = \sum_\alpha \frac{f_{m,\alpha}}{\kappa_{R,\alpha}}
\]
  調和平均では \(\kappa_{R,\alpha} \to 0\) のとき \(1/\kappa_{R,\alpha} \to \infty\) となり除算が発散する。
  これを防止するため、調和平均の前に各成分にフロアを適用する：
  \(\kappa_{R,\alpha} \ge \kappa_{floor} = 10^{-20}\) cm\(^2\)/g。
  物理的に、ほぼ透明な材料（\(\kappa_R \approx 0\)）はRosseland平均において
  フラックスの大半を担うため、有限の下限値を設けても拡散フラックスへの影響は無視できる。

> **根拠**：Planck平均は放射吸収の線形重畳に基づき、質量加重が自然な混合則である。
> Rosseland平均はフラックス（拡散）支配の輸送であり、光学的に厚い領域での
> 調和平均が物理的に正しい（光子は最も透過しやすい成分を選好する）。
> v1.0では上記2種を固定する（SPECIFICATION §6.4.3: Planck平均 = `"linear_mass"`、Rosseland平均 = `"harmonic_mass_R"`）。

**質量分率 \(f_{m,\alpha}\) の時間発展**：ラグランジュステップでは各セルの \(f_{m,\alpha}\) は不変（セル境界を越える物質移動なし）。ALE remap（§3.3.4）により体積分率 \(f_\alpha V\) がセル間で輸送され、remap 後に \(f_\alpha = (f_\alpha V)' / V'\) として正規化し、\(f_{m,\alpha} = f_\alpha\,\rho_\alpha / \sum_\beta f_\beta\,\rho_\beta\) として質量分率を再計算する（ARCHITECTURE §4.4 remap リスト参照）。v1.0 では材料界面追跡を行わず、remap が唯一の \(f_{m,\alpha}\) 更新機構である。

> xRAGEの3T Z-splitting法（\(P_e = Z/(Z+1)\,P_t\) 等）はv1.0では使用しない。
> 将来的にはテーブルEOS分離の精度向上に導入を検討する。

#### 1.1.5a Multi-Material Cell Mixing Model（Conduction / Source Coupling）

多材料セルの伝導・ソース結合では、セル内は温度平衡（single-state）を仮定し、
全材料が同一の \(T_e, T_i\) を共有する。

体積分率 \(f_m\) は
\[
f_m \ge 0,\qquad \sum_m f_m = 1
\]
を満たす（`Geometry.volfrac` を正規化した値）。

**有効原子量（調和平均）**：
\[
A_{eff} = \left(\sum_m \frac{f_m}{A_m}\right)^{-1}
\]
この定義は LaserMesh 実装（`src/laser/laser_mesh.cu`）の \(A_{eff}\) と一致する。

**有効断熱指数（体積分率線形平均）**：
\[
\gamma_{eff} = \sum_m f_m\,\gamma_m
\]

**電子数密度**（\(\bar{Z}\) はセル既知値）：
\[
n_e = \frac{\rho\,\bar{Z}}{A_{eff}\,m_p}
\]

**比熱**（質量比熱）：
\[
c_{v,e} = \frac{\bar{Z}\,k_B}{A_{eff}\,m_p\,(\gamma_{eff}-1)},\qquad
c_{v,i} = \frac{k_B}{A_{eff}\,m_p\,(\gamma_{eff}-1)}
\]

**伝導混合則**：
- Spitzer 伝導率はセルごとの \(n_e(A_{eff})\)、\(c_{v,e}(A_{eff},\gamma_{eff})\) を用いて評価する
- Flux limiter の \(q_{max}\) はセル（面）ごとの \(n_e(A_{eff})\) で評価する
- STS の \(\Delta t_{exp}\) は混合伝導率から得た \(D_{eff}\) を用いて評価する

**ソース結合混合則**：
- `inject_radiation_source_terms` / `inject_laser_source_terms` はセルごとの
  \(A_{eff}, \gamma_{eff}\) を用いて \(e_e \leftrightarrow T_e\) クロージャを行う
- Te \(\leftrightarrow\) ee の変換は上式の \(c_{v,e}(A_{eff},\gamma_{eff})\) を用いる

**単一材料フォールバック**：
\[
n_{mat}=1 \Rightarrow A_{eff}=A_0,\ \gamma_{eff}=\gamma_0
\]
となり、既存の単一材料式に厳密に一致する。

#### 1.1.6 音速

2T系の音速は **総圧力** から：
\[
c_s = \sqrt{\frac{\gamma_e\,P_e + \gamma_i\,P_i}{\rho}}
\]

理想気体で \(\gamma_e=\gamma_i=5/3\) のとき：
\[
c_s = \sqrt{\frac{5}{3}\frac{(1+\bar{Z})\,k_B\,T_{eff}}{A\,m_p}},\quad
T_{eff} = \frac{T_i + \bar{Z}\,T_e}{1+\bar{Z}}
\]
ここで \(T_{eff} = (T_i + \bar{Z}\,T_e)/(1+\bar{Z})\) は圧力重み平均温度
（\(P = P_e + P_i = n_i k_B(T_i + \bar{Z} T_e)\)）を使った等価な書き換えである。

> **訂正（2026-07-26, AI review k11 §2.2）**：\(c_s^2=(\gamma_e P_e+\gamma_i P_i)/\rho\) は
> 「\(T_e \simeq T_i\) のときのみ有効な 1T 等価近似」ではない。電子・イオン成分が加法的
> （\(P=\sum_k P_k(\rho,T_k)\)）で、音響摂動中に各成分が独立に断熱圧縮される
> （e–i 交換は source operator に分離済み — TENRYU の operator splitting と整合）とき、
> これは任意の \(T_e, T_i\) に対して厳密な **frozen-2T 音速**である：
> \[
> c_{s,\mathrm{fr}}^2
> = \sum_{k=e,i}\left[
> \left.\frac{\partial P_k}{\partial\rho}\right|_{T_k}
> + \frac{T_k}{\rho^2 c_{v,k}}
> \left(\left.\frac{\partial P_k}{\partial T_k}\right|_\rho\right)^2
> \right]
> \qquad(\text{理想気体では } \textstyle\sum_k \gamma_k P_k/\rho).
> \]
> e–i 緩和が音響時間より十分速い極限の equilibrium-1T 音速（共通 \(T\) での全微分、
> 交差項を含む）は別物であり、将来の拡張として予約する。hydro CFL と
> characteristic speed には frozen-2T 音速を使うのが離散方程式と整合する。

テーブルEOSでは音速を **等エントロピー偏微分** から評価する。1T（total table）では：
\[
c_s^2 = \left.\frac{\partial P}{\partial \rho}\right|_T
+ \frac{T}{\rho\, C_v}\left(\left.\frac{\partial P}{\partial T}\right|_\rho\right)^2
\]
（\(C_v = \rho\,c_v\)：体積比熱）。2T raw-table backend は 1T 等価温度を**使わず**、
ion / electron それぞれの table を自分の温度 \(T_k\)・比熱 \(c_{v,k}\) で評価した
frozen 成分を二乗和で合成する（下記 Current GPU path 参照）：
\(c_{s,\text{cell}}^2 = c_{s,i}^2 + c_{s,e}^2\)（上の frozen-2T 和と一致）。
`rho_e_table` backend では
\[
c_s^2 = \left.\frac{\partial P}{\partial \rho}\right|_e
+ \frac{P}{\rho^2}\left.\frac{\partial P}{\partial e}\right|_\rho
\]
を \((\log\rho,\log e)\) の C² tensor-product natural cubic spline から直接評価し、
\[
\left.\frac{\partial P}{\partial \rho}\right|_e
= \frac{1}{\rho}\,\frac{\partial P_{rhoe}}{\partial \ln\rho}, \qquad
\left.\frac{\partial P}{\partial e}\right|_\rho
= \frac{1}{e}\,\frac{\partial P_{rhoe}}{\partial \ln e}
\]
\[
c_v = \left(\left.\frac{\partial T}{\partial e}\right|_\rho\right)^{-1}
= \left(\frac{1}{e}\,\frac{\partial T_{rhoe}}{\partial \ln e}\right)^{-1}
\]
を同じ `T(\rho,e)` table の \(e\) 方向導関数から得る。`helmholtz_spline` backend では
\[
\left.\frac{\partial P}{\partial \rho}\right|_T
= \frac{1}{\rho}\,\frac{\partial P_{spline}}{\partial \ln\rho}, \qquad
\left.\frac{\partial P}{\partial T}\right|_\rho
= \frac{1}{T}\,\frac{\partial P_{spline}}{\partial \ln T}
\]
を spline の解析的導関数から直接評価し、\(C_v=\rho\,c_v\) には同じ energy spline 由来の
\[
c_v = \frac{1}{T}\,\frac{\partial e_{spline}}{\partial \ln T}
\]
を用いる。
`exact_ideal_gas` backend では table 導関数は使わず、
\[
c_s^2 = \gamma\frac{P_e+P_i}{\rho}
\]
をそのまま使う。

> **注意**：\(\partial P/\partial\rho\big|_T\) のみでは等温音速であり、断熱圧縮の寄与を欠く。
> 上式の第2項（熱圧力補正項）は高温プラズマで支配的になりうる。
>
> **数値安全**：テーブルEOSの数値微分誤差（補間ノイズ等）により \(c_s^2 < 0\) となる場合がある。
> 実装では \(c_s^2 = \max(c_s^2,\; 0)\) とクランプし、クランプ発生時は
> `DeviceErrorFlags::sound_speed_negative` を設定する（WARNING）。
> \(c_s = 0\) のセルはCFL計算で \(\Delta t_{hydro} \to \infty\) 相当となり、他セルのCFLで律速される。

> **廃止注記（2026-07-26）**：旧設計の \(\delta\rho\)/\(\delta T\) 対称差分
> （\(\delta\rho = \max(10^{-4}\rho,\,10^{-10})\)、テーブル端の片側差分規則を含む）に
> 基づく数値微分は**実装されていない**（対応コードはどの backend にも存在しない）。
> raw-table backend の微分は下記 Current GPU path の
> **補間関数の解析導関数**が正である。グリッドが1点のみの方向は
> \(\partial P/\partial x = 0\) とする（実装と一致）。

断熱指数の後方互換定義（無次元）：
\[
\gamma_k = \rho\, c_s^2 / P \quad (\text{テーブル直接提供がない場合})
\]
ここで \(P = P_e + P_i\)（総圧力）。

代替方式（v1.0既定）：テーブルEOSが \(\gamma_k\) を直接提供する場合はそれを使用する。
IONMIX形式のテーブルは \(\gamma_k\) フィールドを含むことが多い。
テーブルに \(\gamma_k\) がない場合は上記の数値微分を使用する。

**Current GPU sound-speed path (Gamma1 form)**:
For each table (total in 1T, ion/electron separately in 2T), the implementation computes
\[
\Gamma_1 =
\frac{\rho}{P}\left.\frac{\partial P}{\partial \rho}\right|_T
 + \frac{T}{\rho P c_v}\left(\left.\frac{\partial P}{\partial T}\right|_\rho\right)^2,
\qquad
c_s^2 = \Gamma_1 \frac{P}{\rho}.
\]
Derivatives are the **analytic local derivatives of the bilinear-in-\((\ln\rho,\ln T)\)
interpolant**, evaluated at the query point clamped into the enclosing cell
(2026-07-26 fix, AI review k11 §3.2 — previously cell-wide linear secants
\((P(\rho_1,T)-P(\rho_0,T))/(\rho_1-\rho_0)\), which give the mean slope near the
cell's logarithmic mean rather than the slope at the query point; the distortion
factor spans \(\Delta x/(e^{\Delta x}-1)\) to \(\Delta x\,e^{\Delta x}/(e^{\Delta x}-1)\)
across a cell of log-width \(\Delta x\)):
\[
\left.\frac{\partial P}{\partial \rho}\right|_T
= \frac{(1-w_T)\,(P_{10}-P_{00}) + w_T\,(P_{11}-P_{01})}{\rho\,\Delta\ln\rho},
\qquad
\left.\frac{\partial P}{\partial T}\right|_\rho
= \frac{(1-w_\rho)\,(P_{01}-P_{00}) + w_\rho\,(P_{11}-P_{10})}{T\,\Delta\ln T},
\]
where \(w_\rho, w_T\) are the bilinear weights of the query point inside the cell
and \(\rho, T\) are clamped into the cell/table domain (out-of-range queries use the
edge-cell derivative evaluated at the table edge, continuous with the interior).
In 2T table mode, \(c_{s,\text{cell}} = \sqrt{c_{s,i}^2 + c_{s,e}^2}\).
If \(c_s^2 \le 0\) or non-finite, the implementation falls back to \(\sqrt{(5/3)P/\rho}\) for that table.

> CFL条件（§3.1.9, §3.2.13）および人工粘性（§3.1.6, §3.2.9）で使用。

#### 1.1.7 フロア値

数値安全のため、以下の下限値を強制する（SPECIFICATION §6.4.2準拠）：

| 量 | フロア値（既定） | 単位 |
|---|---|---|
| 密度 \(\rho\) | \(10^{-10}\) | g/cm³ |
| 電子温度 \(T_e\) | \(10^{-3}\) | eV |
| イオン温度 \(T_i\) | \(10^{-3}\) | eV |

- フロア適用は各演算子（Hydro, Conduction, Laser, Radiation）の温度更新後に都度実施（§11.2参照）
- 適用回数を診断に出力（物理破綻の早期検出）
- テーブルEOS参照時もフロア値未満にクランプしてから補間

**フロア適用によるエネルギー会計**：フロア適用による注入エネルギー
\[
\Delta E_{floor} = \sum_{c:\,T_c^{computed}<T_{floor}} \rho_c\,c_{v,c}(T_{floor})\,(T_{floor} - T_c^{computed})\,V_c
\]
をエネルギー収支（§10.2）の \(E_{floor}\) 項として記録する。タイミングは§11.2に従い、各演算子（Hydro, Conduction, Laser, Radiation）の温度更新後にそれぞれクランプを適用し、全クランプ分の \(\Delta E_{floor}\) を累積して計上する。

---

### 1.2 輻射輸送（多群、静止媒質）
群 g の放射強度 \(I_g(\mathbf{r},\mathbf{\Omega},t)\)。

\[
\frac{1}{c}\frac{\partial I_g}{\partial t} + \mathbf{\Omega}\cdot\nabla I_g
= \sigma_{a,g} B_g(T_e) - \sigma_{t,g} I_g + \mathcal{S}_{sca}
\]
- \(\sigma_{a,g}\)：吸収係数 \([1/cm]\)（定義は0.2）
- \(\sigma_{t,g}=\sigma_{a,g}+\sigma_{s,g}\)
- \(B_g(T)\)：群積分したPlanck関数（黒体源）
- \(\mathcal{S}_{sca}\)：散乱源。v1.0 では物理散乱を実装しない：\(\sigma_{s,g} = 0\)（全セル・全群）。実効散乱 \((1-f)\sigma_{a,g}\)（§6.1）は IMC の暗黙化手法であり、物理散乱とは異なる。将来版で Thomson 散乱等を追加する場合は \(\sigma_{s,g}\) テーブルを導入する

物質（電子）への交換（連続系の形式として）：
\[
S_r = \sum_g c\,\sigma_{a,g}\left(E_g - a_{eV} T_e^4\,b_g(T_e)\right)
\]
ただし実装では、IMC/DDMCが **沈着エネルギーを直接タリー**して \(S_r\) を構成する（10章）。

---

## 2. 時間積分（operator splitting）
1ステップ \(t^n \to t^{n+1}=t^n+\Delta t\)。

### 2.1 既定：Strang-type splitting
\[
\mathcal{U}^{n+1} =
\mathcal{H}_{\Delta t/2}\circ
\mathcal{R}_{\Delta t}\circ
\mathcal{C}_{\Delta t}\circ
\mathcal{H}_{\Delta t/2}\circ
\mathcal{L}_{\Delta t}(\mathcal{U}^{n})
\]
- \(\mathcal{H}\)：Hydro（Lagrangian step + BC）— 2D_RZ の ALE rezone/remap は2回目の \(\mathcal{H}(\Delta t/2)\) 後にのみ条件付き実行（§3.3）。1D_SPH は pure Lagrangian（§3.4）
- \(\mathcal{C}\)：電子熱伝導
- \(\mathcal{L}\)：Laser（レイトレース→沈着）
- \(\mathcal{R}\)：Radiation（IMC–PGRW–DDMC）

**既定順序**（直ドライブを想定）：
1. Laser full
2. Hydro half
3. Conduction full
4. Radiation full
5. Hydro half

> **精度に関する注意**：古典的Strang splitting \(\mathcal{A}_{\Delta t/2}\circ\mathcal{B}_{\Delta t}\circ\mathcal{A}_{\Delta t/2}\)
> は **2演算子** の場合にのみ \(O(\Delta t^2)\) を保証する。
> 本コードの4演算子構成では、\(\mathcal{H}\) のみが \(\mathcal{C}\text{-}\mathcal{R}\) ブロックに対して
> Strangサンドイッチ（half–half）を形成する。一方、\(\mathcal{L}\) はレイトレースに基づく外部ソースとして
> ステップ先頭で \(\Delta t\) 全体を1回だけ適用する。
> したがって、**\(\mathcal{H}\) と \(\mathcal{C}\text{-}\mathcal{R}\) の結合は2次精度** だが、
> **\(\mathcal{L}\) と \(\mathcal{H}\text{-}\mathcal{C}\text{-}\mathcal{R}\) の結合** および
> **\(\mathcal{C}\text{-}\mathcal{R}\) 相互間は1次精度**（Lie splitting）である。
> ICFシミュレーションにおいて \(\mathcal{L}\) と \(\mathcal{H}\text{-}\mathcal{C}\text{-}\mathcal{R}\) 間、
> および \(\mathcal{C}\text{-}\mathcal{R}\) 間の結合は
> operator splitting 誤差の主要因ではなく（各演算子がΔtで解く物理量が異なるため）、
> Δt制御（§2.2）で実用上十分な精度を確保する。
> 完全な2次精度が必要な場合は、3段以上のStrang分割
> \(\mathcal{H}_{/2}\circ\mathcal{C}_{/2}\circ\mathcal{L}_{/2}\circ\mathcal{R}\circ\mathcal{L}_{/2}\circ\mathcal{C}_{/2}\circ\mathcal{H}_{/2}\)
> を将来オプションとして検討する。

**\(\mathcal{H}(\Delta t/2)\) の具体的手続き**：\(\mathcal{H}(\Delta t/2)\) は \(\Delta t/2\) を \(\Delta t\) として §3.1.10（1D）/§3.2.12（2D）の完全な Predictor-Corrector サイクルを実行する。すなわち Predictor(\(\Delta t/2\)) → 幾何更新 → Corrector(\(\Delta t/2\), \(Q_{ei}\) 含む) の全工程を踏む。Conduction \(\mathcal{C}(\Delta t)\) は Corrector 完了後の \(T_e\)（\(Q_{ei}\) 適用済み）を初期値として受け取る。

**演算子間 EOS 再クロージャ（必須）**：各演算子が \(T_e\) または \(e_e\) を更新した後、後続演算子が正しい熱力学量を参照できるよう EOS 同期を行う。この再クロージャを省略すると、後続フェーズが古い値を参照し、物理的に不正な結果を生む（CUDA_KERNELS §9 参照）。

| 演算子 | 更新対象 | 後処理 | 理由 |
|--------|---------|--------|------|
| H(Δt/2) | \(e_e, e_i\) via PdV | H14(\(e \to T\)) → H13(\(T \to P, C_v\)) → U2(floor) | Corrector内に含む |
| C(Δt) | \(T_e\) 直接 | U2(floor) → H13(\(T_e \to e_e, P_e, C_v\)) | STS が Te のみ更新、ee/Pe/Cv_e 未同期（§4.2.1） |
| L(Δt) | \(e_e\) via source_injection | H14(\(e_e \to T_e\)) → H13(\(T_e \to P_e, C_v\)) → U2(floor) | 1回目 H(Δt/2) が最新 Te, Pe を必要 |
| R(Δt) | \(e_e\) via source_injection | H14(\(e_e \to T_e\)) → H13(\(T_e \to P_e, C_v\)) → U2(floor) | 2回目 H(Δt/2) が最新 Te, Pe を必要 |

**ソース注入プロトコル（U1: source_injection）**：レーザー沈着とIMC/DDMC沈着は、統一カーネル U1 を **フェーズ別に2回** 呼び出すことで注入する（ARCHITECTURE §4.7、CUDA_KERNELS §7.1 参照）：
- Laser演算子後：`source_injection(laser_dep, rad_dep=nullptr)` — レーザー沈着のみ
- Radiation演算子後：`source_injection(laser_dep=nullptr, rad_dep)` — 輻射沈着のみ

現行実装では PGRW は `imc_transport_persistent` 内の IMC branch として処理され、
吸収減衰は通常 IMC と同じ `rad_dep` tally に入る。
したがって source injection は従来どおり
\(\Delta E = \sum_g \texttt{rad\_dep} - \sum_g \texttt{rad\_emit}\)
を用いる。

Hybrid diffusion セル（§7.1.2c）は例外である。diffusion セルでは Radiation
演算子内の cell-local source solve が `Te`, `ee`, `Pe`, `diff_E` を直接更新するため、
その `rad_dep` / `rad_emit` は診断専用であり、U1 では再注入しない。実装上は
`TransportMode::Diffusion` として記録されたセルの
\(H_c^{raw}=\sum_g(\texttt{rad\_dep}_{c,g}-\texttt{rad\_emit}_{c,g})\) を
smoothing 前に 0 とし、face smoothing の barrier としても扱う。

**Radiation thermal microcycling（optional）**：
`Numerics.radiation_thermal_subcycle=True` のとき、single-stage Radiation 演算子
（`Radiation.imc.two_stage=False`）だけを対象に、放射 source injection 後の
compressed cell floor hit を検出して同じ Radiation 演算子を細分化して再試行する。
Hydro / Laser / IMC transport kernel の離散化は変更しない。Conduction は同じ
`conduction_step` を使うが、standalone Strang 位置では呼ばず、thermal substep 内で
`Radiation -> Qei -> Conduction` として評価する。

Retry 判定は各 radiation substep の `source_injection` 完了後に行う。
セル \(c\) が
\[
\rho_c > 5.0\ {\rm g\,cm^{-3}}
\quad\land\quad
T_{e,c} \le T_{e,floor} + 0.5\ {\rm eV}
\]
を満たす場合、compressed floor hit とする。ここで \(T_{e,floor}\) は
`Numerics.floors.Te`（`Mesh.floors.Te_floor_eV` から設定）である。

Algorithm:
1. Radiation 演算子開始時に `ee`, `ei`, `Te`, `Ti`, `Pe`, `Pi` に加え、放射
   prognostic state `rad_E`, `rad_E_old`, `sn_psi_prev`（mode 依存で空の場合は
   no-op）を GPU 上の backup field へ device-to-device copy し、step energy
   ledger accumulator（escaped / marshak_in / volume_source_in /
   numerical_loss / HOLO LO 4 量）の現在値を host snapshot する
   （2026-07-26、AI review k15 C-4: 従来は熱力学 6 field のみで、retry が
   half-advanced な放射場から再開し、失敗 attempt の境界流が step budget に
   二重計上されていた）。
2. 前ステップ以降かつ cell 数が正の場合、compressed cell
   \((\rho>5.0\,{\rm g\,cm^{-3}})\) の \(T_e\) が
   \(T_{e,floor}+1.0\,{\rm eV}\) guard に近いかを調べ、guard から
   5 eV 未満の cell について
   \[
   r_c = \frac{T_{e,c}-(T_{e,floor}+1.0\,{\rm eV})}{\max(T_{e,c},1.0\,{\rm eV})}
   \]
   の最小値から \(n_{sub}\) の初期値を予測する：
   \[
   n_{pred}=\left\lceil\frac{1}{0.2\,\min_c r_c}\right\rceil
   \]
   を \([1,16]\) に clamp し、次の2冪へ切り上げる。該当 cell がなければ
   \(n_{sub}=1\) とする。
3. \(\Delta t_{sub}=\Delta t_R/n_{sub}\) として、各 substep で
   `run_radiation_stage(Δt_sub)`、cell-local ion-electron Coulomb coupling
   \(Q_{ei}(\Delta t_{sub})\)、electron heat conduction
   `conduction_step(Δt_sub)` を順に実行する。Qei は
   \(e_e \leftarrow e_e-q_{ei}\)、\(e_i \leftarrow e_i+q_{ei}\) の後に
   \(e_e,e_i \to T_e,T_i,P_e,P_i\) を再クロージャする。Conduction 後も通常の
   conduction callback と同じ \(T_e\to e_e,P_e\) 同期を行う。
4. 任意の substep 後に compressed floor hit が見つかり、かつ \(n_{sub}<16\) なら、
   pre-radiation の熱力学+放射状態を backup から復元し、ledger accumulator を
   snapshot 値へ巻き戻して、\(n_{sub}\leftarrow 2n_{sub}\)
   として再試行する。
5. \(n_{sub}=16\) でも floor hit が残る場合は、それ以上の retry は行わず結果を受理する。

Prototype limitation（IMC 系。deterministic FLD/\(S_N\) は 2026-07-26 の
transactional 化で解消）: IMC photon pool / census state は復元しない。
そのため IMC での retry 後 Monte Carlo history は初回試行と bitwise には一致せず、
`imc.save_census_snapshot()` / restore 相当の導入が必要である（IMC は退役済み）。
deterministic lane の残存既知事項: 失敗 attempt が書く診断（fld substage audit
history 行、conduction step counter）は巻き戻さない（診断専用・prognostic 影響なし）。

`Radiation.imc.net_e_source_smoothing.enabled = true` の場合、Radiation 演算子の
source injection は \(H_c^{raw}=\sum_g(\texttt{rad\_dep}_{c,g}-\texttt{rad\_emit}_{c,g})\)
をまずセル別に集計し、EOS 再クロージャ前に face-based conservative exchange を
\(N_{\mathrm{pass}}\)（`Radiation.imc.net_e_source_smoothing.passes`）回適用する。
\(H_c^{(0)}=H_c^{raw}\) とし、隣接セル \(a,b\) を結ぶ oriented face で
\[
F_{ab}^{(p)} = \alpha_{ab}\,\lambda_{ab}\,m_{ab}
\left(\frac{H_a^{(p)}}{m_a} - \frac{H_b^{(p)}}{m_b}\right)
\]
を計算する。ここで \(m_{ab}\) は隣接セル質量の harmonic mean であり、2D_RZ でも
face area weight は掛けない。1D_SPH では
\[
H_c^{(p+1)} = H_c^{(p)} + F_{c-1,c}^{(p)} - F_{c,c+1}^{(p)}
\]
を使う。2D_RZ ではセル \(c=(i,j)\) に対して
\[
\begin{aligned}
H_{i,j}^{(p+1)} = H_{i,j}^{(p)}
&+ F_{i-1,j\to i,j}^{(p)} - F_{i,j\to i+1,j}^{(p)} \\
&+ F_{i,j-1\to i,j}^{(p)} - F_{i,j\to i,j+1}^{(p)}
\end{aligned}
\]
を使う。存在しない境界 face（R 軸側を含む）は flux 0 とし、Jacobi 形式（pass ごとに別
buffer へ書き出し）で更新してから
\(e_{e,c} \mathrel{+}= H_c^{(N_{\mathrm{pass}})}/m_c\) とする。

\(\lambda\) は `(tau_threshold, void/material interface)` マスクであり、
`sigma_R_max[c]=max_g sigma_R[c,g]` は IMC opacity assembly 後に保持した cache を使う。
1D_SPH では \(\tau_c=\max_g(\sigma_{R,c,g}\Delta r_c)\) を使う。2D_RZ では
`compute_cell_widths_2d` の代表幅を用い、R face の gate は
\(\tau_{R,c}=\texttt{sigma\_R\_max}[c]\,h_{R,c}\)、Z face の gate は
\(\tau_{Z,c}=\texttt{sigma\_R\_max}[c]\,h_{Z,c}\) とし、各 face で
\(\min(\tau_{\mathrm{left}},\tau_{\mathrm{right}})\ge\texttt{tau\_threshold}\)
を要求する。2D_RZ で `enabled=true` のとき \(\alpha\) の上限は 4-neighbor Jacobi
stability のため 0.125 とし、1D_SPH および smoothing 無効時は従来どおり 0.25 とする。

`Radiation.imc.difference.enabled=true` で difference reference weight \(W_c\) が有効な
場合、実装定数 `active_W_for_smoothing=0.5` として \(W_c\ge0.5\) のセルも
smoothing barrier に加える。したがって difference 併用時の face は、既存 gate に加えて
両隣セルが \(W<0.5\) を満たす場合だけ active になる。この cutoff は smoothing 互換性の
ためだけに使い、reference weight \(W\) の定義には入れない。
既定では `gradient_adaptive=false` で \(\alpha_{ab}=\alpha\) とし、従来と同じ
一様係数を使う。`gradient_adaptive=true` の場合のみ、barrier を通過した face で
\[
\alpha_{ab} = \alpha\,G^T_{ab}\,G^\rho_{ab},
\quad
G^T_{ab} =
\exp\!\left[-\left(\frac{|\Delta\ln T_e|}{s_T}\right)^2\right],
\quad
G^\rho_{ab} =
\exp\!\left[-\left(\frac{|\Delta\ln \rho|}{s_\rho}\right)^2\right],
\]
\[
|\Delta\ln T_e| =
\left|\ln\frac{\max(T_{e,b},10^{-30})}{\max(T_{e,a},10^{-30})}\right|,
\quad
|\Delta\ln\rho| =
\left|\ln\frac{\max(\rho_b,10^{-30})}{\max(\rho_a,10^{-30})}\right|
\]
を用いる。ここで \(s_T\) は `grad_Te_scale`、\(s_\rho\) は `grad_rho_scale` であり、
\(\alpha\) は最大 smoothing 係数として扱う。\(T_e,\rho\) は source injection 前の
状態から読み、multi-pass 中は \(H\) のみを更新するため全 pass で同じ
\(\alpha_{ab}\) を再計算する。1D_SPH の `gradient_adaptive=true` 診断では各 source
injection で `[net_e_smooth] step=N faces_total=M faces_active=A alpha_mean=X alpha_min=Y`
をログ出力し、平均と最小は active face 上の \(\alpha_{ab}\) で定義する。
face flux は反対称なので
\(\sum_c H_c^{(p+1)} = \sum_c H_c^{(p)}\)、従って
\(\sum_c H_c^{(N_{\mathrm{pass}})} = \sum_c H_c^{raw}\) が厳密に成り立つ。
`passes=0` または `alpha=0` では smoothing は無効である。

この `nullptr` ゲーティングにより、各ソースが \(e_e\) に **正確に1回** 注入される不変量を保証する。

**動的Zbar再評価**：`zbar.model = "thomas_fermi"` または `"tabular"` の場合、
\(\mathcal{C}(\Delta t)\) 冒頭と \(\mathcal{R}(\Delta t)\) 冒頭で
現在の \((\rho, T_e)\) から \(\bar{Z}\) を再評価する。
これにより、1回目の \(\mathcal{H}(\Delta t/2)\) による圧縮/加熱後の電離度が伝導係数へ反映され、
\(\mathcal{C}(\Delta t)\) 後の \(T_e\) 変化が NLTE/Fleck closure に反映される。

#### 2.1.1 Hydro開始温度条件（セル単位）

Hydro演算子 \(\mathcal{H}\) の適用をセル単位で制御する。
ナムリストパラメータ `hydro.T_start_eV`（既定 0.0 = 常時有効）が正の場合、
各セルに**一方向フラグ** `hydro_active_c` を持たせる：

\[
\text{hydro\_active}_c^{n} =
\begin{cases}
\text{true}  & \text{if } \text{hydro\_active}_c^{n-1} = \text{true} \quad\text{（判定スキップ）} \\
\text{true}  & \text{if } T_{e,c}^{n} \ge T_{\text{start}} \\
\text{false} & \text{otherwise}
\end{cases}
\]

- **一方向性**：一度 `hydro_active_c = true` になったセルは以降判定を行わない。温度比較の計算コストを省き、揺動による断続的ON/OFFも回避する。
- **初期条件**：`hydro_active_c⁰ = (T_start_eV == 0.0)` — 閾値 0 なら全セルが最初から有効。
- **ノードへの伝播**：ノード \(j\) は、隣接セルのうち少なくとも1つが `hydro_active` であれば移動対象とする。
  \[
  \text{node\_active}_j = \bigvee_{c \in \mathcal{N}(j)} \text{hydro\_active}_c
  \]
  ここで \(\mathcal{N}(j)\) はノード \(j\) に隣接するセル集合。
- **非活性セルの処理**：`hydro_active_c = false` のセルは圧力・人工粘性による力の寄与をゼロとする。座標・速度は固定。
- **Δt への影響**：CFL条件（§2.2 (a)）は活性セルのみを対象とする。全セルが非活性の場合は \(\Delta t_{hydro} = \infty\)（Δt制御から除外）。
- **GPU実装**：フラグ更新は単純なCUDAカーネル（1スレッド/セル）で行い、非活性セルは `if (!hydro_active[c]) { if (Te[c] >= T_start) hydro_active[c] = 1; }` のみ。活性セルはカーネル内で即座に `return` する。
- **使用制約（重要）**：`hydro_active` は**低密度フィル領域（真空代替）を静的に保持する数値マスク**であり、高密度物質の運動を温度条件で抑制する目的には使用してはならない。非活性セルは力の寄与がゼロであるため、隣接する活性セルのノードが OR 伝播（node_active）で移動する際に、非活性セルは「動かされるが押し返さない」片務的な力学状態になる。これはフィル領域（ρ ≪ ρ_shell）では物理的に許容されるが、高密度物質に適用するとエネルギー・運動量の非物理的な注入/消失を引き起こす。
- **活性化境界でのエネルギー保存**：非活性→活性の遷移時に、そのセルの座標・体積は隣接活性ノードの移動によって既に変化している可能性がある。活性化直後の最初のステップで PdV 仕事が不連続にならないよう、活性化時に ρ, e を現在の V から再計算する（質量保存からの ρ = ΔM/V による自動更新で対応）。

### 2.2 Δt制御
\[
\Delta t = \min(\Delta t_{hydro},\; \Delta t_{cond},\; \Delta t_{rad},\; \Delta t_{user},\; \Delta t_{output})
\]

> 全セルが `hydro_active_c = false`（§2.1.1）の場合、\(\Delta t_{hydro} = \infty\) として上式から実質的に除外される。
> 一部セルのみ活性の場合、CFL計算は活性セルのみを対象とする。

**(a) Hydro CFL**（§3.1.9, §3.2.13参照）：
\[
\Delta t_{hydro} = C_{CFL}\cdot\min_c\!\left(\frac{\Delta l_c}{|\mathbf{u}_c| + c_{s,c}}\right),\quad C_{CFL}=0.3\;(\text{既定})
\]
ここで \(\Delta l_c\) はセル代表長 [cm]（1D: \(\Delta r_i\)、2D: \(\sqrt{A_c}\)、§3.1.9/§3.2.13参照）。
1D_SPH で `post_shock_heat=True` の場合は、上式で得た acoustic/AV 制約に加えて
§3.1.9 の明示的 post-shock heat flux 制約 \(\Delta t_{ps}\) を評価し、
\(\Delta t_{hydro} = \min(\Delta t_{acoustic+AV}, \Delta t_{ps})\) とする。
2D_RZ で `Numerics.hydro.axis_margin_dt_floor_fraction > 0` の場合は、
§3.2.13 の axis-margin Δt limiter も hydro CFL の縮小側制約として
\(\Delta t_{hydro}\) に含める。既定 `0.0` では無効であり、既存の
hydro Δt と bitwise 同一である。
No vacuum-aware acoustic-CFL guard or sound-speed cap is added here.  Bounded
\(c_s\) for CSR remap cells raised to \(m_{floor}=\rho_{floor}V\) comes from the
post-remap thermodynamic floor closure in §3.3.4, not from a CFL override.

For 2D_RZ, `Numerics.hydro.corner_jacobian_ale_trigger_enabled=True` adds an
opt-in signed-corner-J pre-hydro ALE trigger evaluated after the ordinary
hydro dt is selected.  For each active cell corner \(k\), the signed
corner Jacobian \(J_{c,k}\) is the 2-D cross product of the two incident
edge vectors in the ordered quadrilateral.  For the current candidate
\(\Delta t_*\), TENRYU evaluates
\[
\mathbf{x}^{trial}(\tau)=\mathbf{x}^n+\tau\mathbf{v}^n,\qquad
0\le\tau\le\Delta t_*
\]
and finds by bisection the largest \(\tau\) for which every positive current
corner remains above
\[
J_{floor,c,k}=\epsilon_J J^n_{c,k},
\quad
\epsilon_J=\texttt{Numerics.hydro.corner\_jacobian\_floor\_eps}.
\]
If the admissible scale \(\tau/\Delta t_*\) is less than
`Numerics.hydro.corner_jacobian_ale_trigger_scale`, TENRYU forces an ALE rezone
before applying the split operators and then recomputes the timestep.  The
trigger is default-off and does not directly clamp hydro dt.

For 2D_RZ driver-level retry, `Numerics.hydro.driver_retry_active_mesh_repair_enabled=True`
adds a retry-only static mesh repair predicate.  At each retry epoch attempt,
TENRYU evaluates the current signed corner Jacobians at \(\tau=0\).  For an
active cell with four positive finite corner Jacobians, define the dimensionless
corner balance
\[
q_{bal,c}=\frac{\min_k J_{c,k}}{\max_k J_{c,k}} .
\]
If any active corner has \(J_{c,k}\le0\), any active corner is non-finite, or
\(q_{bal,c}<\) `Numerics.hydro.driver_retry_corner_balance_threshold`, the
static predicate is inadmissible.  The predicate is evaluated passively on
attempt 0 for diagnostics, but it forces `apply_ale(..., force_rezone=true)`
only when `retry_attempts > 0`; the existing retry exhaustion path remains the
only hard failure point.  This predicate is independent of \(\Delta t\), uses no
initial-volume or last-rezone normalizer, and has no units.

For 2D_RZ, `Numerics.hydro.volume_rate_cfl_enabled=True` adds an opt-in
post-hoc volume-rate limiter after the acoustic/AV and axis-margin hydro
limiters.  For the hydro step that produced the current state, TENRYU stores
the previous hydro volume \(V_c^{old}\), current volume \(V_c^{new}\), and the
used hydro timestep \(\Delta t_{used}\).  The fractional volume change and rate
are
\[
f_{V,c}=\frac{|V_c^{new}-V_c^{old}|}{\max(V_c^{old},10^{-30})},\qquad
r_{V,c}=\frac{f_{V,c}}{\Delta t_{used}}.
\]
For threshold \(f_{V,max}=\)
`Numerics.hydro.volume_rate_cfl_threshold`, the next hydro candidate is capped by
\[
\Delta t_{hydro}^{n+1}\le
\min_c\frac{f_{V,max}}{r_{V,c}}
= \min_c\frac{f_{V,max}\Delta t_{used}}{f_{V,c}} .
\]
The limiter is late by one step: an impulsive volume change in the current step
is measured after that step and can only reduce the following step.  It is a
next-step safety net for source-driven volume changes that the standard
\(|v|/\Delta x\) CFL can underestimate.  The default is off, so existing decks
do not enter this path.

After an accepted ALE rezone/remap (`out.applied == true` on the successful
tail path), TENRYU resets `state.vol_prev_hydro` to the current `state.vol`
before the next hydro CFL evaluation.  ALE displacement is non-Lagrangian mesh
motion, while this limiter measures Lagrangian hydro volume change; without the
reset, the immediately following volume-rate sample would treat ALE motion as a
one-step hydro flow signal.  The reset skips only that polluted post-ALE
measurement.  It does not change `state.dt_prev_hydro`, so trial-volume
diagnostics that require a positive previous hydro timestep keep their history.
This behavior closes the Phase 2d-ext v3 L2 256x512 ALE-on dt-collapse root
cause, where accepted ALE motion polluted the volume-rate CFL baseline.

For 2D_RZ, `Numerics.hydro.trial_volume_cfl_enabled=True` adds an opt-in
pre-corrector trial-volume diagnostic in the Lagrangian corrector path.  After
the corrector node velocities are updated and axis-motion preflight has applied
any radial scaling, but before the final node-position commit, TENRYU evaluates
trial final cell volumes \(V_c^{trial}\) from the same node positions that the
corrector would commit.  The diagnostic requires
\[
V_c^{trial} \ge f_{trial} V_c^n
\]
for every cell, where \(f_{trial}=\)
`Numerics.hydro.trial_volume_cfl_floor_fraction` and \(V_c^n\) is the cell
volume at the beginning of the hydro step.  If any cell violates the floor,
TENRYU reports the minimum \(V_c^{trial}/V_c^n\), the first local failing cell,
and a suggested retry step
\[
\Delta t_{suggested}=f_{shrink}\Delta t
\]
with \(f_{shrink}\) given by
`Numerics.hydro.trial_volume_cfl_shrink_fraction`.
The current driver has no full-step retry/restore path for split hydro,
laser, and radiation state, so this Phase 2d-extension implementation is
diagnostic-only: it warns before the existing geometry refresh/volume assert
path rather than re-running the step.  The first hydro step is bypassed while
the previous-step sentinel \(\Delta t_{prev}\le0\) is present.  The default is
off, so existing decks do not enter this path.

**(b) 伝導 Δt**（ソルバ依存、§4.2.1/§4.2.3参照）：

**STS ソルバ**（`conduction.solver="sts"`、既定）：
\[
\Delta t_{cond} = \Delta t_{cond,sts} = \frac{s_{max}(s_{max}+1)}{2}\;\Delta t_{exp},\quad
\Delta t_{exp} = C_{cond}\cdot\min_c\!\left(\frac{(\Delta l_c)^2}{D_{eff,c}}\right),\quad
C_{cond}=0.25\;(\text{既定})
\]
ここで \(s_{max}\) はSTS最大ステージ数（既定40、SPECIFICATION §6.4.7）、
\(D_{eff}\) は実効拡散係数（§4.3）。
伝導演算子は自身のCFL制約をSTSで内部処理するため（§4.2.1）、
グローバルΔtには \(\Delta t_{exp}\) ではなく
\(s_{max}(s_{max}+1)/2\) 倍に緩和された \(\Delta t_{cond,sts}\) のみが寄与する。

**陰的ソルバ**（`conduction.solver="implicit"` for 1D_SPH, `conduction.solver="hypre"` for 2D_RZ）：
\[
\Delta t_{cond} = \infty \quad\text{（伝導CFL制約なし）}
\]
陰的スキーム（backward Euler）は無条件安定であるため、
伝導がグローバルΔtを制約することはない。§4.2.3参照。

> **設計根拠（STS）**：従来の \(\Delta t_{cond} = \Delta t_{exp}\) をグローバルmin に含める設計では、
> \(N_{sub} = \lceil\Delta t / \Delta t_{exp}\rceil = 1\) が常に成立し、
> サブサイクリングが事実上無効化される矛盾があった。
> STS導入により、伝導演算子は \(s\) ステージ（\(s \le s_{max}\)）で
> \(O(s^2)\) 倍の安定領域を確保するため、
> グローバルΔtを \(s_{max}(s_{max}+1)/2\) 倍まで緩和でき、他演算子のΔtに追従できる。
>
> **設計根拠（陰的）**：STSの \(s_{max}\) 到達時や、Δtが hydro/radiation に
> 律速されず伝導のみに律速される状況で、陰的ソルバが有効。
> 1D_SPH では三重対角直接解法、2D_RZ では Hypre を用いることで、
> いずれもグローバルΔtから伝導CFL制約を外せる。

**床保護スロットル（BUG-15 修正、2026-07-08、対称 pair-min 形）**：
各ステージの床保護は二段構成である。第1パスがセル毎に
\(\alpha_c = \min\!\bigl(1,\ \rho c_v V (T_e - T_{floor}) / (|P_{net,c}|\,\Delta t_{sub})\bigr)\)
（流出 \(P_{net,c}<0\) のセルのみ、他は 1）を評価し、第2パスは**各 face/pair の
flux 寄与を両側共通の \(\min(\alpha_c, \alpha_n)\) で縮約**する。歴史的な
セル毎スケーリング（流出側のみ \(\alpha\) 倍）は pair 反対称性を破り
\((1-\alpha)|P|\Delta t\) を無から創出していた（結合レーザー・アブレーション系の
blow-off tip で実測 eps 2.17 → 修正後 6.1e-7）。\(\alpha \equiv 1\)（勾配が床保護を
要さない全ての認証構成）では \(\times\min(1,1)\) の挿入のみで演算列は恒等。
床クランプと \(E_{floor}\) 記帳は最終ガードとして不変（監査上 artificial 計上）。
全6変種（1D harmonic/Kirchhoff/secant・1D per-material・2D Kershaw plain/per-material）
に適用。

**(c) 輻射 Δt**（Fleck factor制約）：

IMCの暗黙化パラメータ（Fleck factor）が極端に小さくならないようΔtを制限する。
Fleck factor \(f_c\)（§6.1）は：
\[
f_c = \frac{1}{1 + \alpha\,c\,\beta_c\,\sigma_{P,c}\,\Delta t}
\]
ここで \(\beta_c = 4\,a_{eV}\,T_{e,c}^3 / C_{v,e,c}\)（§6.1 参照）、\(\sigma_{P,c} = \rho_c\,\kappa_{P,c}\)。

\(f_c\) が小さすぎると IMC の実効散乱が増大し分散が悪化するため、下限 \(f_{\min}\) を設ける：
\[
\Delta t_{rad} = \min_c\!\left(\frac{1-f_{\min}}{f_{\min}}\cdot\frac{1}{\alpha\,c\,\beta_c\,\sigma_{P,c}}\right),\quad f_{\min}=0.01\;(\text{既定})
\]

> \(f_{\min}\) が小さいほど制約は緩い。\(f_{\max}\)（SPECIFICATION §6.4.5、Fleck factor上限）とは独立のパラメータ。
> この下限は IMC 側 Fleck にのみ適用する。FLD 側 Fleck は stiff-cell 極限を保つため下限を使わない。

> **σ_P 陳腐化に関する注意**：\(\sigma_{P,c}\) は Phase 4（Radiation演算子冒頭）で計算される。
> Phase 5（Hydro 半ステップ後半）で \(T_e\) が変化するため、Phase 6 の \(\Delta t_{rad}\) 計算時には
> \(\sigma_P\) が陳腐化している。この誤差は成長率制限（\(g_{dt} = 1.2\)）により、1ステップで
> \(\Delta t\) が急変しないことで実用上吸収される。Phase 5 での \(T_e\) 変化が大きい場合は
> 次ステップの \(\Delta t_{rad}\) が自動的に小さくなり自己修正される。

**(d) レーザー Δt**：独立の CFL 制約は不要（レーザー吸収は hydro Δt のサブステップで処理されるため）。
v1.0 では `Δt_laser` を独立に算出せず、`Δt_hydro` に包含する。上式の min 項にも含めない。

**(e) ユーザ指定**：`dt.initial_s`（初期Δt）、`dt.max_s`（上限Δt）、`dt.min_s`（下限Δt、既定 \(10^{-20}\) s）を SPECIFICATION §6.4.7 で設定。
`dt.initial_s` は **step 0 のみ** 使用される（step 0 では前ステップが存在せず成長制限が適用不能なため、ユーザ指定値を初期 Δt とする）。
step ≥ 1 では `dt.initial_s` は無視され、通常の CFL + 成長制限が適用される。
リスタート時はチェックポイントの dt を初期値として使用し、`dt.initial_s` は適用しない。
\(\Delta t < \Delta t_{min}\) となった場合はシミュレーションを FATAL 停止する（ストーリング防止）。

**dt floor-stall detector（opt-in）**：`Numerics.dt.floor_stall_max_consecutive_steps=N`
（既定 0 = disabled）を正値にした場合、driver は成功 commit された step だけを対象に
床張り付き状態を連続カウントする。判定は
\[
\Delta t_{committed} \le (1+10^{-6})\,\Delta t_{min},\qquad
t_{end}-t > 10^3\,\Delta t_{min},\qquad
\text{limiter}\notin\{\texttt{output},\texttt{t\_end}\}
\]
をすべて満たす場合に true とする。非 stall step でカウンタは 0 に戻る。カウンタが
`floor_stall_max_consecutive_steps` を超えると、D4-class の retry-cascade
dt 床張り付きとして FATAL 停止し、`dt.min_s`、現在の \(\Delta t\)、残り時刻、
limiter、および disable 方法を診断に出力する。既定値 0 ではカウンタは常に無効で、
baseline の bit-exact 挙動を保つ。

**Δt成長制限**：急激なΔt増大を防止するため、前ステップ比で制限する：
\[
\Delta t^{n+1} \le \min\!\bigl(\Delta t_{all},\; g_{dt}\,\Delta t^n\bigr),\quad g_{dt} = \text{growth\_factor}\;(\text{既定 }1.2,\;\text{SPECIFICATION §6.4.7})
\]

**(f) 出力時刻整合**（SPECIFICATION §6.4.8 `X_every_s` パラメータ）：

時間間隔ベース出力が有効（`X_every_s > 0`）の場合、タイムステップを出力時刻に正確に到達させる：
\[
\Delta t_{output} = \min\!\bigl(t_{next,plot} - t,\; t_{next,history} - t,\; t_{next,checkpoint} - t\bigr)
\]
ただし各項は対応する `X_every_s > 0` の場合のみ min に含める。全て無効（`-1.0`）の場合は \(\Delta t_{output} = \infty\)。

**浮動小数点許容差**：出力判定に浮動小数点許容差 \(\varepsilon\) を使用する：
\[
\varepsilon = 10^{-14} \times \max(|t|,\; X\_every\_s)
\]
出力条件は \(t \ge t_{next,X} - \varepsilon\) で判定する。

**出力後の \(t_{next}\) 更新**：出力が実行された後、次の出力時刻を前進させる：
```
while t_next_X <= t + ε:
    t_next_X += X_every_s
```
while ループにより、Δtが `X_every_s` より大きい場合（極端な設定時）にも正しく動作する。

**初期化**：`t_{next,X} = t_{start} + X\_every\_s`。リスタート時はチェックポイントの `output_state/` から復元（SPECIFICATION §7.4 リスタート手順 ステップ5）。

**MPI同期不要**：\(t\), \(t_{next,X}\), `X_every_s` は全rankで同一値を保持するため、\(\Delta t_{output}\) の計算に追加の `MPI_Allreduce` は不要。

**安定性への影響**：\(\Delta t_{output}\) は常にΔt縮小方向に作用するため、CFL安定性条件を緩和することはなく、安全である。

**マルチrank同期**（§12参照）：各CFL条件の \(\min_c\) はローカルセルに対して計算される。
マルチGPU実行時は各rankがローカル \(\Delta t\) を計算した後、
`MPI_Allreduce(MPI_MIN)` でグローバル最小 \(\Delta t\) を全rankで共有する：
\[
\Delta t_{global} = \min_{p=0}^{P-1} \Delta t_p
\]
ここで \(\Delta t_p\) は rank \(p\) のローカル \(\min(\Delta t_{hydro,p},\, \Delta t_{cond,p},\, \Delta t_{rad,p},\, \Delta t_{user})\)。
全CFL条件を単一の Allreduce で処理し（各rankがローカル min を取った後に1回の Allreduce）、
通信レイテンシを最小化する。成長制限 \(1.2\,\Delta t^n\) はグローバル \(\Delta t_{global}\) に対して適用する。
\(\Delta t_{output}\)（(f) 出力時刻整合）はホスト側で適用し、全rankで同一値のため追加 Allreduce は不要。

---

