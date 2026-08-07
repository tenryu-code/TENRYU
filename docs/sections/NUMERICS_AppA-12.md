<!-- 分割元: docs/NUMERICS.md | このファイルは参照用です。原本（docs/NUMERICS.md）が権威です。 -->
## Appendix A. Kershaw 9点差分法（歪四辺形RZメッシュ上の拡散離散化）

### A.1 目的・適用範囲

Kershaw 9点差分法は、歪四辺形RZメッシュ上で拡散方程式を離散化する統一手法である。
TENRYUでは以下の2つのモジュールで共通基盤として使用する：

- **電子熱伝導**（§4.3）：拡散係数 \(D = \kappa_{SH}/(\rho\, c_v)\)
- **DDMCリーク係数**（§7.3）：拡散係数 \(D_g = 1/(3\sigma_{R,g})\)

> **参照**: Kershaw (1981), "Differencing of the diffusion equation in Lagrangian hydrodynamic codes",
> *J. Comput. Phys.* 39, 375–395.

### A.2 メッシュ記法

論理座標 \((i,j)\) に対して物理座標 \((r_{i,j}, z_{i,j})\) をノード位置とする。

- **セル中心** \((i+1/2, j+1/2)\)：4頂点の算術平均
\[
r_{i+1/2,j+1/2} = \frac{1}{4}(r_{i,j}+r_{i+1,j}+r_{i+1,j+1}+r_{i,j+1})
\]
\[
z_{i+1/2,j+1/2} = \frac{1}{4}(z_{i,j}+z_{i+1,j}+z_{i+1,j+1}+z_{i,j+1})
\]

- **辺中点** \((i+1/2, j)\)：2ノードの算術平均
\[
r_{i+1/2,j} = \frac{1}{2}(r_{i,j}+r_{i+1,j}),\quad
z_{i+1/2,j} = \frac{1}{2}(z_{i,j}+z_{i+1,j})
\]
\((i, j+1/2)\) も同様。

- セル中心に未知量 \(\phi_{i+1/2,j+1/2}\) を配置する（温度、放射エネルギー密度等）。

### A.3 幾何ベクトル（A, B）とヤコビアン

ノード \((i,j)\) における幾何ベクトル：

- **A ベクトル**（j方向に沿う）：
\[
\mathbf{A}_{i,j} = \begin{pmatrix} r_{i,j+1/2} - r_{i,j-1/2} \\ z_{i,j+1/2} - z_{i,j-1/2} \end{pmatrix}
\]

- **B ベクトル**（i方向に沿う）：
\[
\mathbf{B}_{i,j} = \begin{pmatrix} r_{i+1/2,j} - r_{i-1/2,j} \\ z_{i+1/2,j} - z_{i-1/2,j} \end{pmatrix}
\]

- **ヤコビアン**（外積のz成分）：
\[
J_{i,j} = \mathbf{A}_{i,j} \times \mathbf{B}_{i,j} = A_r B_z - A_z B_r \quad [\text{cm}^2]
\]

\(J_{i,j} > 0\) はメッシュが正の向き（反時計回り）であることを表す。

**ヤコビアン \(J\) のフロア処理**：
\[
J_{i,j} = \max(J_{i,j}^{computed},\; J_{floor}), \quad J_{floor} = 10^{-30}\;\text{cm}^2
\]
\(J_{i,j} < J_{tol} = 10^{-20}\) cm\(^2\) の場合：当該ノードのKershaw勾配演算子 **B** をゼロに設定し、
隣接セルへの拡散フラックスを打ち切る（縮退セルからの非物理的な熱流を防止）。
爆縮末期に内殻セルが極度に圧縮される場合に発生しうる。

### A.4 B-演算子（セル中心値→頂点勾配への写像）

ノード \((i,j)\) を共有する4つのセルを以下のように定義する：
- **NW**: \((i-1/2, j+1/2)\)
- **NE**: \((i+1/2, j+1/2)\)
- **SE**: \((i+1/2, j-1/2)\)
- **SW**: \((i-1/2, j-1/2)\)

Kershaw (1981) Eqs.10–13 に基づく4つのB-演算子 \(B^1, B^2, B^3, B^4\) は、
4セル中心値 \((\phi_{NW}, \phi_{NE}, \phi_{SE}, \phi_{SW})\) からノード \((i,j)\) での勾配成分を構成する：

\[
B^1_{i,j}(\phi) = B_z(\phi_{NW} - \phi_{SE}) - A_z(\phi_{NE} - \phi_{SW})
\]
\[
B^2_{i,j}(\phi) = A_z(\phi_{NW} - \phi_{SE}) - B_z(\phi_{NE} - \phi_{SW})
\]
\[
B^3_{i,j}(\phi) = A_r(\phi_{NW} - \phi_{SE}) - B_r(\phi_{NE} - \phi_{SW})
\]
\[
B^4_{i,j}(\phi) = B_r(\phi_{NW} - \phi_{SE}) - A_r(\phi_{NE} - \phi_{SW})
\]

ここで \(A_r, A_z, B_r, B_z\) はすべてノード \((i,j)\) での値。

ノード \((i,j)\) における勾配ベクトル：
\[
\nabla\phi\big|_{i,j} = \frac{1}{2J_{i,j}}
\begin{pmatrix}
B^1_{i,j}(\phi) \\
B^4_{i,j}(\phi)
\end{pmatrix}
\]

### A.5 拡散係数の面平均

セル中心に定義された拡散係数 \(D_{i+1/2,j+1/2}\) を、面（辺中点）で調和平均する。

- **i-面**（ノード \((i,j)\) と \((i,j+1)\) を結ぶ辺の中点）：
\[
\sigma_{i,j+1/2} = \frac{2\, D_{i-1/2,j+1/2}\, D_{i+1/2,j+1/2}}{D_{i-1/2,j+1/2} + D_{i+1/2,j+1/2}}
\]

- **j-面**（ノード \((i,j)\) と \((i+1,j)\) を結ぶ辺の中点）：
\[
\lambda_{i+1/2,j} = \frac{2\, D_{i+1/2,j-1/2}\, D_{i+1/2,j+1/2}}{D_{i+1/2,j-1/2} + D_{i+1/2,j+1/2}}
\]

> 調和平均は物質界面での拡散係数不連続に対して正しい透過率（interface conductance）を再現する。
> 算術平均を使うと界面での流束が過大評価される。

**\(D_L + D_R = 0\) の場合**：\(\sigma = 0\)（断熱壁として扱い、面を通過するフラックスをゼロとする）。実装：`if (D_L + D_R < 1e-30) σ = 0;`。\(\lambda\) についても同様。

### A.6 9点ステンシル組み立て

拡散方程式 \(-\nabla\cdot(D\nabla\phi) = S\) の離散化により、各セル中心 \(i\) に対して
9点ステンシルを構成する。

> **導出概要**：これらの係数は拡散方程式の弱形式をセル体積で積分し、
> 発散定理を適用（体積積分→面積分）、B 演算子（A.4）で勾配を置換、
> 4つのコーナー寄与を平均化して得られる（Kershaw 1981, Eqs. 15--22 に対応）。
> 具体的には、セル \(i\) の体積積分 \(\int_{V_i} \nabla\cdot(D\nabla\phi)\,dV = \oint_{\partial V_i} D\nabla\phi\cdot\hat{n}\,dA\)
> を4辺の面積分に分解し、各面での \(\nabla\phi\) を隣接ノードの B 演算子で近似する。
> 面積分の離散化に面拡散係数（A.5 の \(\sigma, \lambda\)）を用い、
> 1つのノードを共有する2辺の寄与を平均して1/4 因子が現れる。
\[
\sum_{j'\in\mathcal{N}_9(i)} a_{i,j'}\, \phi_{j'} = b_i
\]

ステンシルは以下の9つの係数で構成される：
- **直接隣接4点**（N, E, S, W）：\(\sigma, \lambda\) 項から構成
- **対角隣接4点**（NE, NW, SE, SW）：メッシュ非直交性から生じる \(\rho^{1-4}\) 交差項
- **対角（自身）**：\(a_{ii} = -\sum_{j'\ne i} a_{i,j'}\)（行和=0、保存性保証）

セル \((i+1/2, j+1/2)\) に対する明示的な9点係数
（Kershaw (1981) の記法に基づき、添字 \((i,j)\) のノードが
セルの左下コーナーに対応）：

**直接隣接（i方向）**：
\[
a_E = \frac{1}{4}\left[\sigma_{i+1,j+1/2}\frac{(A_r^2+A_z^2)_{i+1,j}}{J_{i+1,j}}
+ \sigma_{i+1,j+1/2}\frac{(A_r^2+A_z^2)_{i+1,j+1}}{J_{i+1,j+1}}\right]
\]
\[
a_W = \frac{1}{4}\left[\sigma_{i,j+1/2}\frac{(A_r^2+A_z^2)_{i,j}}{J_{i,j}}
+ \sigma_{i,j+1/2}\frac{(A_r^2+A_z^2)_{i,j+1}}{J_{i,j+1}}\right]
\]

**直接隣接（j方向）**：
\[
a_N = \frac{1}{4}\left[\lambda_{i+1/2,j+1}\frac{(B_r^2+B_z^2)_{i,j+1}}{J_{i,j+1}}
+ \lambda_{i+1/2,j+1}\frac{(B_r^2+B_z^2)_{i+1,j+1}}{J_{i+1,j+1}}\right]
\]
\[
a_S = \frac{1}{4}\left[\lambda_{i+1/2,j}\frac{(B_r^2+B_z^2)_{i,j}}{J_{i,j}}
+ \lambda_{i+1/2,j}\frac{(B_r^2+B_z^2)_{i+1,j}}{J_{i+1,j}}\right]
\]

**対角隣接**（メッシュ非直交交差項）：
\[
\rho^1_{i,j} = -\frac{1}{4}\frac{(\mathbf{A}\cdot\mathbf{B})_{i,j}}{J_{i,j}},\quad
\rho^2_{i,j} = \frac{1}{4}\frac{(\mathbf{A}\cdot\mathbf{B})_{i,j}}{J_{i,j}}
\]
\[
a_{NE} = \sigma_{i+1,j+1/2}\,\rho^1_{i+1,j+1} + \lambda_{i+1/2,j+1}\,\rho^1_{i+1,j+1}
\]
\[
a_{NW} = \sigma_{i,j+1/2}\,\rho^2_{i,j+1} + \lambda_{i+1/2,j+1}\,\rho^2_{i,j+1}
\]
\[
a_{SE} = \sigma_{i+1,j+1/2}\,\rho^2_{i+1,j} + \lambda_{i+1/2,j}\,\rho^2_{i+1,j}
\]
\[
a_{SW} = \sigma_{i,j+1/2}\,\rho^1_{i,j} + \lambda_{i+1/2,j}\,\rho^1_{i,j}
\]

**対角（自身）**：
\[
a_C = -(a_E + a_W + a_N + a_S + a_{NE} + a_{NW} + a_{SE} + a_{SW})
\]

行和がゼロ（\(\sum_{j'} a_{i,j'} = 0\)）であり、一様場 \(\phi=\text{const}\) で \(-\nabla\cdot(D\nabla\phi)=0\) が離散的に保証される（保存性）。

> **M-matrix性**：直交格子上では交差項がゼロ（\(\mathbf{A}\cdot\mathbf{B}=0\)）となり、
> ステンシルは古典的5点差分に退化する。歪格子上では交差項が非零となるが、
> 歪みが過大でなければオフ対角 \(a_{j'}\le 0\)（\(j'\ne i\)）のM-matrix性が保たれる。
> M-matrix性はDDMCのリーク率の非負性に必須であり（§7.3.1）、
> 違反するセルではDDMCを禁止してIMCへフォールバックする（§7.3.3）。

**正のオフ対角係数の処理（M-matrix修復）**：
歪格子上で対角隣接係数 \(a_{NE}, a_{NW}, a_{SE}, a_{SW}\) が正になる場合
（メッシュ非直交性が過大、すなわち \(\mathbf{A}\cdot\mathbf{B}\) の符号が交差項を正にする場合）、
以下の Kershaw (1981) §4 に基づく修復処理を適用する：

1. **正のオフ対角クランプ**：\(a_{corner} > 0\) の場合、\(a_{corner} \leftarrow 0\) に設定
2. **対角再計算**：\(a_C = -\sum_{j'\ne i} a_{i,j'}\) を再計算して行和ゼロを維持
3. **保存性保証**：行和ゼロの再計算により、一様場でのフラックスゼロは保証される

この修復は **保存性を保つ** が、精度は1次に低下する（非直交交差項の寄与を切り捨てるため）。
DDMC用途では §7.3.3 のM-matrix判定で修復前の係数を検査し、
正のオフ対角が検出されたセルではDDMCを禁止する（修復された係数での拡散は使用しない）。
電子熱伝導（§4）用途では修復後の係数を使用して explicit 更新を安定に行う。

**診断**：M-matrix修復が適用されたセル数を1ステップあたりで集計し、
`diagnostics/kershaw_mmatrix_fix_count` に記録する。
修復セル数 \(> 0.1 \times N_{cells}\) の場合は WARNING（メッシュ品質劣化の兆候）。

**GPU実装**：ノード並列 gather 方式。各ノードスレッドが隣接4セルの拡散係数を読み取り、9点ステンシル係数を計算する。scatter（atomic）は不使用（gather のみで完結）。`block_size = 256`（CUDA_KERNELS.md §10.3 参照）。

### A.7 RZ幾何因子

2D RZ（軸対称回転体）では、各面積・体積積分に \(2\pi r\) 因子が入る。

ステンシル係数の修正：
- 各ノード \((i,j)\) の面中点座標 \(r\) で重み付けを行う
- i-面の \(\sigma\) 項に面中点の \(r_{i,j+1/2}\) を乗じる
- j-面の \(\lambda\) 項に面中点の \(r_{i+1/2,j}\) を乗じる
- 右辺の体積項（ソース項）にはセル中心の \(r_{i+1/2,j+1/2}\) を含むRZ体積で重み付け

具体的には、A.6の各係数に以下の置換を適用する：
\[
\sigma_{i,j+1/2} \to \sigma_{i,j+1/2}\cdot r_{i,j+1/2},\quad
\lambda_{i+1/2,j} \to \lambda_{i+1/2,j}\cdot r_{i+1/2,j}
\]

**\(r = 0\) 軸上のノード処理**：
\(r = 0\) に位置するノード \((i=0, j)\) では面中点座標 \(r_{0,j+1/2} = 0\) となり、
上記置換で i-面の \(\sigma\) 項がゼロになる。これは物理的に正しい：
軸対称条件 \(\partial\phi/\partial r\big|_{r=0} = 0\)（A.8.2）により
\(r = 0\) を横切るR方向の拡散フラックスはゼロである。
具体的には：
- i-面（R方向）の \(\sigma\) 項：\(\sigma_{0,j+1/2} \cdot r_{0,j+1/2} = 0\)（軸上面の寄与はゼロ）
- j-面（Z方向）の \(\lambda\) 項：\(r_{1/2,j}\) は軸近傍の正の値を持つため非零
- 交差項 \(\rho^{1-4}\) もR方向の \(\sigma\) 項を含むため、軸上ノードではゼロ化される
- **L'Hopital則の不要性**：Kershaw定式では \(1/r\) 因子が陽に現れないため、
  特異点処理（L'Hopital則等）は不要である。\(r\) 因子は面積重みとして乗じられるだけであり、
  \(r=0\) でのゼロ化が自然に反射境界条件を実現する

### A.8 境界条件

#### A.8.1 Dirichlet境界
ゴーストセル（メッシュ外仮想セル）に規定値 \(\phi_{ghost} = \phi_{BC}\) を設定する。
ステンシル係数のうちゴーストセルに対応する項を右辺に移動する。

#### A.8.2 反射（Neumann零流束）
ゴーストセルの値を内部セルの値に設定する：\(\phi_{ghost} = \phi_{interior}\)。

r=0軸の対称条件：
- r=0上のノードでは反射境界条件を適用する
- これにより \(\partial\phi/\partial r\big|_{r=0} = 0\) が離散的に保証される

#### A.8.3 真空（Robin境界、DDMC用）
DDMC領域の外部境界では、外挿長を用いたRobin境界条件を適用する：
\[
\phi + d_{ext}\,\hat{\mathbf{n}}\cdot\nabla\phi = 0
\]
外挿長：\(d_{ext} = 0.7104/\sigma_{tr}\)（Milne問題の漸近値、§7.4参照）。

ゴーストセルの値は外挿により設定する：
\[
\phi_{ghost} = -\phi_{interior}\,\frac{d_{ext} - \Delta x/2}{d_{ext} + \Delta x/2}
\]

**\(\Delta x\) の定義**：\(\Delta x = V_{cell} / A_{face}\)（§7.7.4 の \(\Delta x_m\) と同一定義）。2D_RZ: \(\Delta x = V_{cell} / (2\pi R_{face}\, L_{face})\)。

**\(\sigma_{tr}\) の定義**：\(\sigma_{tr} = \sigma_{a,g} + \sigma_{s,g}\)（v1.0: 散乱なしのため \(\sigma_{tr} = \sigma_{a,g}\)）。群依存。
本節の Robin BC は **電子熱伝導** と **DDMC** の両方で使用されるが、DDMC 用途では §7.1.1 に従い \(\sigma_{tr} = \sigma_{R,g}\)（Rosseland）を使用する。電子熱伝導用途では不透明度は関与しない（\(d_{ext}\) は §4 では使用しない）。
\(\sigma_{tr} \approx 0\) の場合 \(d_{ext} \to \infty\) となり \(\phi_{ghost} \to -\phi_{interior}\)（ゼロ入射フラックス漸近：真空側からの入射がゼロ）。

### A.9 時間依存拡散方程式への適用

Kershawステンシル（A.6）は空間離散化のみを定義する。
時間依存の拡散方程式 \(\partial\phi/\partial t = \nabla\cdot(D\nabla\phi) + S\) への適用時の
時間積分方式を定める。

**電子熱伝導**（§4.2）：**前進Euler**（explicit）＋サブサイクリング
\[
\phi^{(m+1)} = \phi^{(m)} + \Delta t_{sub}\left[\frac{1}{V_i}\sum_{j'\in\mathcal{N}_9(i)} a_{i,j'}\,\phi_{j'}^{(m)} + S_i^{(m)}\right]
\]
安定性条件 \(\Delta t_{sub} \le C_{cond}\cdot\min_i(V_i / \sum_{j'\ne i}|a_{i,j'}|)\)（§2.2(b)参照）。

**DDMCリーク係数**（§7.3）：Kershawステンシルは **イベント率の定義** にのみ使用する。
時間積分はモンテカルロのイベント駆動（§7.5の指数待ち時間サンプリング、temporally continuous方式）
で行われるため、行列方程式を陽に解く必要はない。

> **行列ソルバー不要の根拠**：DDMCのリーク確率はステンシル係数から直接定義（§7.3.2）され、
> 粒子ベースのランダムウォークとして時間積分される。これはモンテカルロ法の特長であり、
> 決定論的拡散コードのCG/GMRES等とは根本的に異なる。

### A.10 DDMCリーク係数との接続

Kershawステンシル（A.6）で構成される行列 \(A_{ij}\) から、DDMCリーク係数（§7.3.2）を導出する。
既定（`m_matrix_check=True`）では：
\[
\Sigma^{leak}_{i\to j,g} = \frac{-A_{ij,g}}{V_i}
\]
を用い、\(A_{ij,g}>0\)（\(j\ne i\)）が検出されたセル×群はDDMC禁止（IMCへフォールバック）とする。

`m_matrix_check=False`（検証用）では安全クランプを許可し：
\[
\Sigma^{leak}_{i\to j,g} = \frac{\max(0, -A_{ij,g})}{V_i}
\]
とする。

> **§7.3.2との関係**：§7.3.2で禁止している `max(0, A_ij)`（正のオフ対角を拾う操作）とは
> 異なる。本式の `max(0, -A_ij)` は `m_matrix_check=False` 時の保護措置であり、
> 既定運用ではDDMC継続に使わない。

**1D特殊化**：1D球対称では Kershawステンシルは3点トリダイアゴナルに退化し、
§7.3.4で定義した1D DDMCリーク係数と一致する。これにより
1D・2Dの拡散離散化が統一的な枠組みで記述される。

---

## 12. 並列分割と通信（MPI + CUDA）

本節では、TENRYUのマルチGPU並列化に必要な領域分割、ハロー交換、
粒子（光子）移動、LaserMesh分散戦略、各演算子の越境処理、負荷分散、
および再現性保証の数理・プロトコルを定義する。
基本方針：**1 MPI rank = 1 GPU**、空間メッシュの領域分割に基づく。

**単一GPU（n_ranks==1）規約**：n_ranks==1 の場合、以下の通信は no-op またはスキップする：
- ハロー交換（§12.2）: ゴーストセルは存在しない（n_ghost=0 として扱う）。pack/send/recv/unpack 全体をスキップ
- 粒子移送（§12.3）: P5/MPI/P6 全体をスキップ（emigrant は存在しない。CUDA_KERNELS §9 参照）
- MPI_Exscan（§12.7.1）: スキップし rank_offset=0 を設定
- MPI_Allreduce（§2.2 Δt, §10.2 エネルギー収支等）: 呼び出してもよい（MPI_COMM_SELF で no-op 同等）が、コスト回避のためスキップも可。実装方針は通信ラッパーで一元管理する
- LaserMesh 同期（§12.4）: MPI_Allgatherv/Allreduce をスキップ（全データがローカルに存在）

---

### 12.1 領域分割

#### 12.1.1 1D_SPH：動径スラブ分割

1D球対称メッシュ（§3.1）をP個のMPI rankへ分割する。
セル番号 \(i=0,\dots,N_r-1\) を連続スラブとして各rankへ割り当てる。

**分割規則**：
- rank \(p\) はセル \([i_p^{start}, i_p^{end})\) を所有する
- 各rankの所有セル数 \(n_p = i_p^{end} - i_p^{start}\)
- **最小セル数制約**：\(n_p \ge n_{min}\)（既定 \(n_{min}=8\)）
  - Kershawステンシル（3点）＋ゴースト1層＋安全余裕
  - **制約違反時**：\(N_r < P \times n_{min}\) の場合、初期化時に FATAL エラーで停止する。エラーメッセージに \(N_r, P, n_{min}\) を表示し、rank数の削減を促す
- 分配：\(n_p = \lfloor N_r/P \rfloor + (p < N_r \bmod P)\)（余りは先頭rankへ）
- rank 0 は \(r=0\)（中心）境界を含む
- rank \(P-1\) は外側境界を含む

**隣接関係**：
- rank \(p\) の左隣：\(p-1\)（\(p=0\) は左隣なし＝反射/中心境界）
- rank \(p\) の右隣：\(p+1\)（\(p=P-1\) は右隣なし＝外側境界）

#### 12.1.2 2D_RZ：2次元デカルト分割

2D RZメッシュ（§3.2）を \(P_r \times P_z\) のデカルトグリッドで分割する。
総rank数 \(P = P_r \times P_z\)。

**分割次元の決定**：
- ユーザ指定 `dims=[P_r, P_z]` がある場合はそれを使用
- 未指定時は通信面積最小化で自動決定：
\[
(P_r^*, P_z^*) = \arg\min_{P_r P_z = P} \left( P_r \cdot N_z + P_z \cdot N_r \right)
\]
ただし \(N_r, N_z\) はグローバルメッシュのセル数。

**r=0軸の制約**：
- r=0軸上のセルは幾何特異性を持つため、r方向に分割する際の
  最内スラブ（\(p_r=0\)）は r=0 を含む全セルを所有する
- rank配置は `MPI_Cart_create(ndims=2, dims=[P_r,P_z], periods=[false,false])`

**各rank \((p_r, p_z)\) の所有セル範囲**：
\[
r: [i_r^{start}(p_r),\, i_r^{end}(p_r)), \quad
z: [j_z^{start}(p_z),\, j_z^{end}(p_z))
\]

**\(P\) が素数の場合の処理**：
\(P\) が素数の場合、分解は \((1, P)\) または \((P, 1)\) の2通りのみとなる。
argmin の結果、アスペクト比がより等方的な配置を選択する。
\(P\) が素数のとき WARNING を出力：
`"WARNING: prime rank count P={P} results in 1D decomposition. Recommend composite rank count for better load balance."`

#### 12.1.3 節点所有権と共有境界

分割境界上の節点は複数rankで共有される。
**所有権規則**：共有節点は **最小rank番号** のrankが所有する。

**コーナー力の加算プロトコル**（§3.2.6 Wilkins型コーナー力との接続）：

現行実装（M18 前）は単一GPU運用を前提とし、物理境界では H16 相当の境界条件を
境界ノードへ直接適用する（ゴーストセル層なし）。H4 はこの境界ノード状態を用いて
コーナー力を計算する。

MPI 実装（M18）では **ゴーストセル方式（Approach B）** を採用する：
1. ステップ/Phase 冒頭のセルハロー交換（§12.2.2 Hydro行：ρ, Te, Ti, Pe, Pi, Q, hydro_active）で
   ゴーストセルの圧力・密度を最新化する
2. 各rankは H4（compute_corner_force）で **所有セル＋ゴーストセル** の全寄与を計算する
3. \(n_{ghost}=1\) により、分割境界上の共有節点に寄与する全セル（所有＋隣接1層）が
   ローカルに可視であり、完全なコーナー力が局所的に得られる
4. 別途のコーナー力交換 MPI 通信は**不要**

この方式（M18実装後）により Predictor-Corrector 4回の H4 呼び出し全てで
追加通信なしに正しいコーナー力が得られる。H5 後の節点速度は両rankで同一値に
更新され、Phase 末の節点ハロー交換で整合性を確保する。

> **注**: 所有セルのみでコーナー力を計算し MPI 交換で合算する方式（Approach A）も等価だが、
> 通信回数が増えるため M18 ではゴーストセル方式を採用する。

#### 12.1.4 分割メタデータ：PartitionInfo

各rankが保持するメタデータ構造体：

```
PartitionInfo:
  rank:            int          # MPI rank番号
  n_ranks:         int          # 総rank数
  cart_coords:     int[2]       # 2Dカート座標 (1Dでは[p,0])
  cart_dims:       int[2]       # カートトポロジ [P_r, P_z]
  local_cell_range: int[2][2]   # [[ir_start,ir_end],[jz_start,jz_end]]
  local_node_range: int[2][2]   # セル範囲+1（節点）
  ghost_layers:    int          # ゴーストレイヤー数 (=1)（ARCHITECTURE §7.1.1 PartitionInfo 準拠）
  n_ghost_cells:   int          # ゴーストセル総数（derived: 2D_RZ で 2*(nr_local+nz_local)+4）
  neighbor_ranks:  int[8]       # [left,right,bottom,top,NE,NW,SE,SW] (-1=境界)
  global_to_local: offset[2]    # global_id - offset = local_id
```

**local↔global写像**：
- global cell ID \((i,j)\) → local cell ID \((i - i_r^{start}, j - j_z^{start})\)
- local配列サイズ：\((n_r^{local} + 2 n_{ghost}) \times (n_z^{local} + 2 n_{ghost})\)

#### 12.1.4a Option C 実装レイアウト（v1 正規、M18 実装 2026-07）

v1 実装は上記のローカル圧縮配列ではなく **Option C（global-size 配列）** を採用する
（設計記録: docs/design/mpi_m18_20_20260717.md）。本項が §12.1.4 のローカル配列
記述に優先する。

- **全 rank が global サイズの配列を保持**する。セルは flat 添字 \(c = i\,n_z + j\)
  （2D_RZ、r-slab では所有域が連続区間になる）。添字変換は存在しない
  （serial と同一の添字空間）。
- **所有窓（owned window）**: 各 rank は自分の所有スラブ
  \([c_{begin}, c_{end})\) のみを更新する。実装 API は
  `State::owned_cell_window(n)`（cw）と、ghost 帯を含む
  `State::owned_cell_window_ghost(n, stride)`（fw）。serial（n_ranks==1）では
  いずれも \([0, n)\) に退化し、**bit 恒等**が定義により成立する。
- **ghost 帯**は所有スラブに隣接する自然な global 添字位置にあり
  （別領域のコピーではない）、ハロー交換で owner の bit をそのまま受け取る。
- **far 領域**（所有+ghost の外）は *stale-but-finite* 不変条件に従う：
  値は古くてよいが有限でなければならず、いかなる所有域の結果にも
  影響してはならない（「far は不可視」）。全域走査カーネル（例: mesh 幾何
  再計算）は far で garbage-finite を生成してよい。
- **節点**は cell-tile 整列で分割し、共有 i-plane は両隣接 rank が重複計算
  （同一入力・同一式 → bit 同一）した上で、交換は owner-overwrite で権威を確定する。
- 縮約（dot/min/max/sum）は **所有窓の部分縮約 + MPI_Allreduce** を正規とし、
  bit 再現が要る係数（例: レーザー保存リスケール §12.4）は rank0 の正準順
  全和を exact-zero Allreduce で配布する。

---

### 12.2 ハロー（ゴーストセル）交換

#### 12.2.1 ハロー層数

**\(n_{ghost} = 2\)**（v1 実装、全演算子で統一。M18 改訂 2026-07）。

根拠（最広ステンシルが 2 層を要求する）：
- 1D checkerboard PQ フィルタ（奇偶減衰）：\(\rho\)/active を \(i\pm2\) 参照 → 2層
- 電子系奇偶 flux：セル \(j-2 \dots j+1\) 参照 → 2層
- Kershaw 9点（Appendix A）・Wilkins コーナー力（§3.2.6）・Winslow（§3.3.3）は
  1層で充足（2層は上位互換）
- （laser deposit 平滑化は owner-true 全域 gather 上で走るため ghost 幅に非依存 §12.5.0）

（旧規定 \(n_{ghost}=1\) は M18 前の設計値。）

#### 12.2.2 交換フィールド一覧

各計算フェーズで交換するフィールドを下表に示す。

| フェーズ | 交換フィールド | データ型 | 備考 |
|----------|---------------|----------|------|
| **Hydro** | ρ, Te, Ti, Pe, Pi, Q, hydro_active | cell-centered, double×6 + int8×1 | ステップ初期＋中間。Te/Ti: EOS(H13)用、Q（≡ Qvisc、ARCHITECTURE §5.2 CellField 名）: 人工粘性(H4)用、hydro_activeはノードOR判定（§2.1.1）に必要 |
| **Hydro（節点）** | 節点座標 r,z、節点速度 | node-centered, double×4 | H(Δt/2) 後（次フェーズの H4 がゴーストノード座標を参照する前提を保証） |
| **Conduction** | T_e | cell-centered, double×1 | STS サブステージ毎（v1.0: Te のみ。伝導は電子 Te を直接更新し、Ti は変更しない。κ_e は C1 が ghost 含め局所計算） |
| **ALE rezone** | 節点座標 r,z | node-centered, double×2 | Jacobi反復毎 |
| **ALE remap** | ρ, Te, Ti, Pe, Pi | cell-centered, double×5 | remap＋EOS reclosure後1回（A3→H7→H8→H14→H13完了後の原始変数を交換。CUDA_KERNELS §9 Phase 5参照） |
| **Radiation（前処理1）** | T_e, ρ, Z̄, volFrac, vol, face_area, ell_ddmc | cell-centered, double×(3+1+1+n_faces+1) | 放射ステップ前（U9 がghost cellのopacityを計算、R2/R3 がghost cellのDDMC判定に幾何量を参照） |
| **Radiation（前処理2）** | f_fleck | cell-centered, double×1 | R1 → R2 間（R2 がghost cellの Fleck factor を参照するため） |
| **Radiation（後処理）** | （交換不要） | — | 粒子移送は P5/P6 で別途処理。mesh場の post-radiation 交換は不要 |

> **Radiation ハロー交換の設計根拠**：opacity（σ_a, σ_R, σ_t）は U9 カーネルが Te, ρ, Z̄ から
> 各セル（ghost cell 含む）で局所計算するため、群別 opacity の直接交換は不要。
> これにより放射ハロー交換のデータ量を double×(1+3G) → double×(3+幾何量) に削減（G=16 で 392 → ~64 bytes/cell）。
> 幾何量（vol, face_area, ell_ddmc）は H7 が owned cells のみ計算するためハロー交換で ghost cells に供給する。
> f_fleck は R1 が owned cells のみ計算するため R1→R2 間の小ハロー交換（double×1）で供給する。

#### 12.2.3 Strang splitting内の交換タイミング

**v1 実装正規（M18、2026-07）— staged intra-step 交換**：以下の固定表に代えて、
「**所有窓を書いた直後・最初の ghost 消費者の前**」に段階交換を置くことを正規とする。
実装上の不変条件（違反は §6m/§6n 級の界面欠陥を生む；設計記録 §6m 参照）：

1. **Hydro 入口**: cell 交換 {ee, ei, mass, Q, ρ, Pe, Pi, Te, Ti, z̄}（10 field。
   ρ/Pe/Pi/Te/Ti/z̄ は ghost の EOS/密度再計算（fw 窓）を owner-bit に一致させる
   ための前提。z̄ は step 内 producer を持たず、この交換が唯一の ghost 更新）。
2. **position commit（predictor/corrector 各段）直後**: node 交換 {x, v}。
   半歩幾何 refresh・mesh cache upload・work 段より**前**でなければならない
   （commit は所有 node のみを動かすため、交換前の隣接列幾何は 1-motion stale）。
3. **step 末尾の AV 再計算の前**: 最終 cell 交換（ρ, vol, Te, Ti, Pe, Pi, cs）+
   node 交換 {x, v}。翌 step の pq が消費する保存 Q の界面セル値を owner-true にする。
4. **演算子内交換**は phase タグ付きで演算子が自弁する（FLD-2D CG の探索方向
   p を反復毎に交換、SN-2D KBA の plane pipeline、レーザー入口の全域 gather 等。
   §12.5 参照）。

以下の固定タイミング表は M18 前の設計案であり、参考として残す（正規ではない）：

§2.1のStrang分割（L(Δt) → H(Δt/2) → C(Δt) → R(Δt) → H(Δt/2)）
における交換タイミング（1物理ステップあたり）：

```
 1. Laserフルステップ前      : MPI_Allreduce(MAX, δ_skip) [1回]  // skip判定を全rank統一（§5.9）
 2. Laserフルステップ後      : （交換不要：沈着は局所HydroMeshのみ） [0回]
 3. Hydro半ステップ前       : cell交換 (ρ,Te,Ti,Pe,Pi,Q,hydro_active) [1回]  // §12.2.2 Hydro行
 4. Hydro半ステップ後       : node交換 (x_r,x_z,v_r,v_z)      [1回]  // Lagrangianメッシュ移動後
 5. Conductionフルステップ前 : cell交換 (T_e)                  [1回]
 6. Conductionフルステップ後 : cell交換 (更新T_e)              [1回]
 7. Radiation前             : cell交換 (T_e, ρ, Z̄, volFrac, vol, face_area, ell_ddmc)  [1回]
                              // U9: ghost cell opacity計算、R2/R3: ghost cell DDMC判定に幾何量が必要
 7b. R1→R2間              : cell交換 (f_fleck)              [1回]
                              // R2 がghost cellの Fleck factor を参照するため
 8. Radiation後             : （粒子移送はP5/P6で別途処理、mesh場の交換不要） [0回]
 9. Hydro半ステップ前       : cell交換 (ρ,Te,Ti,Pe,Pi,Q,hydro_active) [1回]  // Laser/Hydro/Conduction/Radiation で更新されたため再交換
10. Hydro半ステップ後       : node交換 (x_r,x_z,v_r,v_z)      [1回]
11. ALE rezone（発動時）     : node交換 ×K回 (Jacobi反復)     [K回]
12. ALE remap（発動時）      : cell交換                        [1回]
```

> **Laserの交換について**：LaserMeshは全rank複製方式（§12.4.2）であり、
> 各rankが独立にレイトレースを実行する。沈着結果は自rank所有HydroMeshセルのみに
> 適用するため、沈着後のrank間通信は不要（§12.4.3参照）。
> ただし、**skip判定**（§5.9）は全rankで一致させる必要があるため、
> δ_local の MPI_Allreduce(MAX) → δ_global を Laser ステップ前に1回実行する。
> skip判定が不一致の場合、laser_mesh_sync の MPI集団通信でデッドロックが発生する。

**合計**：約8回/ステップ（ALE発動なし時は8回、発動時は8+K+1回（rezone node交換K回 + remap cell交換1回））。
Kは典型的に10–50回（Jacobi反復）だが、通信量は小さい（node座標のみ）。

**伝導 STS ステージ中のハロー交換**：
伝導演算子が STS で \(s\) ステージを実行する場合（§4.2.1）、各ステージで新しいゴーストセル温度が必要。

方式：
- \(s \le 4\)：ステージ毎にハロー交換を実施（正確だが通信コストが高い）
- \(s > 4\)：以下の最適化を許可
  - 1回目のステージ前にハロー交換
  - 以降は内部セルのみ更新し、境界セル温度はラグ付き値を使用
  - ただし、境界セルの温度変化率が \(|\Delta T/T| > 0.1\) を超えた場合は追加ハロー交換を挿入
  - **MPI同期必須**：追加交換の発動判定は `MPI_Allreduce(MAX, need_extra)` で全rank統一すること。ローカル判定のみでは交換回数不一致によるデッドロックが発生する

v1.0 既定：毎ステージ交換（安全優先）。
`conduction.halo_strategy = "every" | "adaptive"`（既定 `"every"`）

> **注意**：ハロー交換の省略は領域境界での伝導精度を劣化させ、
> \(s\) が大きい場合（強い伝導領域）には不安定性を引き起こす可能性がある。
> STS 導入により \(s\) は素朴法の \(N_{sub}\) の \(\sqrt{}\) 程度に削減されるため、
> 毎ステージ交換のコストも大幅に低減される。

#### 12.2.4 GPU-aware MPI実装

各ハロー交換の実装手順：

```
1. Pack   : GPU上でハローバッファへ pack（CUDAカーネル）
2. Isend  : GPU-aware MPI_Isend（デバイスポインタ直接）
3. Irecv  : GPU-aware MPI_Irecv（デバイスポインタ直接）
4. Wait   : MPI_Waitall
5. Unpack : GPU上でゴーストセルへ unpack（CUDAカーネル）
```

**フォールバック（GPU-aware MPIが利用不可の場合）**：
```
1. Pack   : GPU上でハローバッファへ pack
2. D2H    : cudaMemcpyAsync (Device→Host staging buffer)
3. Isend  : MPI_Isend (ホストバッファ)
4. Irecv  : MPI_Irecv (ホストバッファ)
5. Wait   : MPI_Waitall
6. H2D    : cudaMemcpyAsync (Host→Device)
7. Unpack : GPU上でゴーストセルへ unpack
```

GPU-aware MPIの検出はCMake時に行い、`TENRYU_GPU_AWARE_MPI` マクロで分岐する。

#### 12.2.5 コーナーゴーストセル交換プロトコル（2D RZ、9点Kershawステンシル対応）

Kershaw 9点ステンシルは対角隣接セルの値を参照するため、コーナーゴーストセルの交換が必須。
2段階方式を採用：

**Phase 1: 面方向ハロー交換（4方向）**
- LEFT/RIGHT 方向：n_ghost × n_z_local セルを交換
- BOTTOM/TOP 方向：n_r_local × n_ghost セルを交換
- MPI_Isend/Irecv × 4方向 → MPI_Waitall

**Phase 2: コーナー方向ハロー交換（4方向）**
- Phase 1 完了後に実行。対角ランクとの直接コーナー交換
- BL/BR/TL/TR 方向：各 n_ghost × n_ghost セルを交換
- 交換先：対角ランク（存在しない場合は送受信を行わず、物理境界条件でゴースト値を設定）
- MPI_Isend/Irecv × 4方向 → MPI_Waitall

**混合コーナー（物理境界×並列境界）の処理**：
対角ランクが存在しないが、1方向は並列境界・もう1方向は物理境界の場合
（例：r=0 軸上のrank の SW/NW コーナー）、Phase 2 で以下の手順を適用する：
1. Phase 1 の面方向交換で並列方向のゴーストセルが取得済みであることを前提とする
2. 物理境界方向の反射/ゼロ勾配を、Phase 1 で受信したゴーストセルに適用する
   例：r=0 軸（i=-1）の SW コーナーゴースト (-1, jz_start-1):
   Phase 1 BOTTOM 交換で (0, jz_start-1) が取得済み → r=0 反射で (-1, jz_start-1) = mirror(0, jz_start-1)
3. **Phase 1 のデータに依存する** ため、Phase 1 完了後に実行する（上記の順序保証が必須）

**タグ割り当て**：
```
tag = phase_id * 1000 + direction * 100 + field_id
```
- phase_id: 1=cell_hydro, 2=node_hydro, 3=cell_conduction, 4=cell_radiation, 5=cell_radiation_f_fleck, 6=cell_ALE, 7=emigrant, 8=ddmc_leak
- direction: LEFT=0, RIGHT=1, BOTTOM=2, TOP=3, NE=4, NW=5, SE=6, SW=7（ARCHITECTURE §7.1.1 準拠）
- field_id: v1.0 ではパック交換方式（1方向1メッセージ）のため常に **field_id=0**。
  将来のフィールド個別交換に備え、以下の field_id を予約する：
  0=T_e, 1=T_i, 2=rho, 3=Pe, 4=Pi, 5=ee, 6=ei, 7=Cv_e, 8=Cv_i, 9=zbar,
  10=vol, 11=mass, 12=Q, 13=hydro_active(int8), 14=volFrac, 15=face_area,
  16=ell_ddmc, 17=f_fleck（最大 field_id ≤ 99、桁衝突なし）

> **旧式 `phase*100 + dir*10 + field_id` は field_id ≥ 10 でタグ衝突するため廃止**。
- `neighbor_ranks[8]` との対応：
  - face: LEFT=0, RIGHT=1, BOTTOM=2, TOP=3
  - corner: NE=4 (=TR), NW=5 (=TL), SE=6 (=BR), SW=7 (=BL)

**パッキング**：各方向のフィールドを単一バッファにセルmajorでパック。`send_buf[i * n_fields + f]`（1スレッドが1ゴーストセルの全フィールドをパック。ARCHITECTURE §7.1.2 準拠）。

---

### 12.3 粒子（光子）移動

**[LEGACY — v1 実装対象外（M18 改訂 2026-07）]** IMC/DDMC の退役（正典輻射 =
FLD §6.7 / SN §6.8 の決定論ソルバ）により、光子粒子の rank 間移動は v1 MPI の
実装対象外である。本節のプロトコルは将来の MC モード復活に備えた設計記録として
保持する（`src/parallel/particle_migration.cu` の基盤は存置）。なお 1D burn
scheme="mc" は MPI 以前に driver assert で停止する既知の死路（本線へ報告済み）。

IMC/DDMCの光子粒子がセル境界を越え、その先が他rankの領域である場合の
移動（migration）プロトコルを定義する。

#### 12.3.1 バッチ移動戦略

**EMIGRANT バッファ容量**：`Parallel.migration.initial_capacity`（既定 10000）を初期容量とする。超過時は P5 が OVERFLOW 粒子のエネルギーを `E_numerical_loss` に計上し、当該粒子を alive=2（OVERFLOW）に設定する（R7 で除去）。`DeviceErrorFlags::emigrant_overflow` に報告する（`particle_overflow` は PhotonPool 容量超過用、ARCHITECTURE §10.1 参照）。ステップ終了後に `capacity = ceil(capacity × growth_factor)`（`growth_factor` 既定 1.5）で拡張する。

**方針**：粒子がrank境界を越えた時点では即時送信せず、
サブステップ末にまとめてバッチ送信する。

**理由**：
1. 即時送信は小メッセージが大量発生しMPIオーバーヘッドが支配的になる
2. GPUカーネル実行中にMPI通信を挿入すると、カーネル中断が必要
3. バッチ化によりメッセージサイズが大きくなり帯域効率が上がる

**サブステップの定義**：
- IMC：全alive粒子の1回の追跡バッチ（R8/R9カーネルの1回の実行完了）
- **v1.0設計**：1ステップ = 1サブステップ。Persistent Warp (R8) は `time_remain` 消費で1回のカーネル実行内に完結する（CUDA_KERNELS §9 参照）。将来版でサブステップ分割を導入する場合は、R8/R9内にサブステップ境界を設ける
- 最大サブステップ数 \(N_{sub}^{max}\)（既定32）を超えた場合は強制送信。
  \(N_{sub}^{max}\) 回の送受信サイクル後も未配送の粒子は消滅させ、そのエネルギーを **\(E_{numerical\_loss}\)** として計上する（物理的境界流出 \(E_{escaped}\) とは別計上）。全粒子の 0.1% を超える場合は verify モードで FATAL、通常モードで ERROR。\(E_{numerical\_loss}\) は history 出力の `energy/numerical_loss` に記録される

**Emigrant バッファオーバーフロー処理**：
1. 各スレッドは emigrant 追加前に atomic counter を検査：
   `slot = atomicAdd(&n_emigrant, 1)`
   `if (slot >= capacity)`: 粒子の alive フラグを OVERFLOW (= 2) に設定
2. OVERFLOW 粒子のエネルギーは `atomicAdd(&E_numerical_loss, energy)` で損失計上（エネルギー保存の追跡）
3. OVERFLOW 粒子は以降の Composite Key Sort（§6.5, R7）で dead として除去（comp_key = 0xFFFFFFFF）
4. `DeviceErrorFlags::emigrant_overflow` に報告（P5 内で設定、CUDA_KERNELS §8.2 準拠）
5. ステップ終了後、ホスト側で `capacity = ceil(capacity × growth_factor)` に拡張し再確保
6. 回復手順：w_survive を2倍、N_p を50%削減。最大3ステップで未解消なら ERROR（ARCHITECTURE §10.1）
7. overflow 発生時は WARNING を出力し、`parallel.migration.initial_capacity` と `parallel.migration.growth_factor` の見直しを促す

#### 12.3.2 IMC越境プロトコル

IMC粒子が追跡中にrank境界面を越えた場合：

1. **EMIGRANT検出**：粒子のセル交差ルーチンで、移動先セルがゴーストセル
   （他rank所有）であることを検出
2. **射出情報記録**：粒子の状態（位置、方向、エネルギー、群、残存時間、
   global_id、RNGカウンタ状態）をEMIGRANTバッファへ書き込み
3. **粒子の停止**：元のrank上ではこの粒子をdead扱い（compactionで除去）
4. **サブステップ末に一括送信**：
   - EMIGRANTバッファを宛先rank別にソート
   - 各宛先rankへ `MPI_Isend`
5. **受信rank側で追跡再開**：
   - 受信した粒子を alive プールに追加
   - 次のサブステップから追跡を再開
   - **受信粒子の位置**：実位置をそのまま保持する。受信rank の §9 セル探索で再同定する。面射影は行わない（track length 正確性維持のため）。

**粒子移送サブステップ同期プロトコル**：
1. 各トラッキングサブステップ終了後、全ランクが隣接ランクとの非同期交換を実行：
   - `MPI_Isend`（emigrant buffer → 隣接ランク）× 最大8方向
   - `MPI_Irecv`（隣接ランクから受信）× 最大8方向
   - emigrant 数がゼロでも空メッセージを送信（デッドロック回避）
2. `MPI_Waitall` 完了後、受信粒子を alive pool に追加
3. 次のサブステップ開始
4. 全ランクが参加（emigrant なしのランクも空交換を実行）

#### 12.3.3 DDMC越境リークプロトコル

DDMCの粒子がリーク（§7.3.2）でrank境界面を越える場合：

1. **リーク先判定**：リーク先セルが他rank所有のゴーストセルであることを検出
2. **モード判定の延期**：リーク先セルでのDDMC/IMCモード判定（§7.7）は
   **受信rank側で実行**する
   - 理由：モード判定にはリーク先セルの局所幾何情報（体積、面積）と
     不透明度が必要であり、これらは受信rankのみが正確に保持する
3. **EMIGRANT記録**：粒子状態＋リーク元セルID＋リーク面方向を記録
4. **受信rank側での処理**：
   - DDMC継続の場合：リーク先セルのDDMC粒子として追跡再開
   - IMC変換の場合：§7.7.2（DDMC→IMC）の変換規則に従い、IMC粒子として出射方向・
     位置をサンプリングし追跡開始

#### 12.3.4 粒子データ転送フォーマット

1粒子あたりの転送データ（SoA→AoSパック→転送→SoA展開）：

```
ParticleEmigrant:
  position:     double[3]     # (r, z, 0) or (x, y, z)    24 bytes
  direction:    double[3]     # (Ω_r, Ω_z, Ω_φ)          24 bytes
  energy:       double        # 粒子エネルギー [erg]        8 bytes
  weight:       double        # 統計重み                    8 bytes
  time_remain:  double        # 残存時間 [s]                8 bytes
  birth_energy: double        # 生成時エネルギー [erg]      8 bytes
                              # （Russian roulette §6.3.4 で受信rank側参照）
  global_id:    uint64        # グローバル粒子ID            8 bytes
  rng_counter:  uint32        # Philoxカウンタ状態          4 bytes
  cell_id_src:  int32         # 送信元ローカルセルID        4 bytes
  group:        uint16        # 群番号                      2 bytes
  sign:         int8          # 粒子符号（+1 or -1）         1 byte
  leak_face:    int8          # リーク面方向 0-3（CUDA_KERNELS §6.4.3規約、IMC/DDMC共通）  1 byte
  mode:         uint8         # ParticleMode: 0=IMC, 1=DDMC, 2=RW（P6 セル再同定で mode 依存分岐に必須。CUDA_KERNELS §8.3 参照）  1 byte
  padding:      uint8[3]      # 8-byte alignment           3 bytes
  ---
  合計: 104 bytes/particle（8-byte aligned、sizeof = 104 = 13×8）
```

> 注：80 bytesは設計初期の見積もり。birth_energy、sign、アライメントを含め104 bytes。rng_counter は uint32（4 bytes）で格納し、curand_init 呼び出し時に uint64 へ暗黙昇格される。cell_id_src は送信元rankでのローカルセルIDであり、デバッグおよび粒子移動先の候補セル推定に使用する。birth_energy は受信rank側で Russian roulette（§6.3.4）の R12 カーネルが参照する。

#### 12.3.5 Census粒子の管理

**Census粒子**（§6.3、時間ステップ末に生存している粒子）は、
静的分割では追跡終了時点のセルが所在rankに属するため、
**rank間移動は不要**。

移動粒子がcensusとなった場合（残存時間=0で他rank領域にいる場合）は、
12.3.2/12.3.3のEMIGRANTプロトコルで受信rankへ送り、
受信rank側でcensusプールに入れる。

---

### 12.4 LaserMesh並列戦略

#### 12.4.1 1D_SPH：Allgatherv方式

1D球対称でのLaserMesh（§5）並列化：

1. 各rankが自身の所有セルの動径プロファイル（\(\rho(r), T_e(r), n_e(r)\)）を準備
2. `MPI_Allgatherv` で全rankのプロファイルを収集（全rankが全動径プロファイルを持つ）
3. `raytrace_2d` では各rankが独立に **1ビーム分の** レイトレースを実行する（§5.6.4）。
   `radial_absorption_1d` では rank 0 のみが §5.4a の serial kernel を実行する
4. `raytrace_2d` では正規化吸収分率を全ビームパワー合計でスケーリングする。
   `radial_absorption_1d` では rank 0 が得た 1D 沈着パワー配列を既存 Allgatherv 経路で集約する
5. 局所セル範囲のみ自身のHydroMeshへ適用する

**通信量見積もり**：
- 1D_SPH、\(N_r = 1000\) セル、3変数（ρ, T_e, n_e）：
  \(N_r \times 3 \times 8\,\text{bytes} = 24\,\text{KB}\)
- Allgatherv総量：\(\approx 24\,\text{KB} \times P\)（P << 100で無視可能）

#### 12.4.2 2D_RZ：全rank複製方式（v1.0）

2D RZでのLaserMesh並列化（v1.0）：

**戦略**：LaserMeshを**全rankで複製**する。

**理由**：
- LaserMesh（§5.1）はHydroMeshとは独立な構造格子であり、
  HydroMeshに比べてサイズが小さい（典型 \(N_r^{LM} \times N_z^{LM} \approx 100 \times 200\)）
- 分散レイトレースは光線がrank境界を横断する際の通信が複雑
- v1.0では全rank複製で十分なスケーラビリティが得られる

**手順**：
1. 各rankが自身の所有セルから LaserMesh への場の値（\(\rho, T_e, n_e, Z^*\)）を
   局所的に射影（bilinear）
2. `MPI_Allreduce(SUM)` で全rankの寄与を合算 → 全rankで同一LaserMeshを構成
3. 各rankが独立に **各極角グループの代表ビーム** でレイトレースを実行（§5.6.4に従い重ね合わせ）
4. 吸収エネルギーをHydroMeshへ写像（§12.4.3）

**通信量見積もり**：
- LaserMesh \(100 \times 200\) セル、4変数：
  \(100 \times 200 \times 4 \times 8\,\text{bytes} = 640\,\text{KB}\)
- Allreduce総量：\(\approx 640\,\text{KB}\)（rank数に依存せず一定）

**重複寄与処理**：各ランクは所有 HydroMesh セルの寄与のみを計算する。`MPI_Allreduce(SUM)` 後の重み合計は双線形分割の一体性（partition of unity）により 1.0 となる。追加の正規化は不要。診断：\(|w_{total} - 1.0| > 10^{-10}\) で WARNING を出力する。

#### 12.4.3 沈着写像：分割HydroMeshへのscatter-add（v1.0）

v1.0（`laser_parallel.strategy="replicated"`）では、沈着写像も次元依存で扱う。

- **1D_SPH**：
  レイトレース結果は Hydro 1Dセル吸収パワー [erg/s] として直接得る。
  rank 間ではこの 1D 配列を `Allgatherv` で同期し、その後に各rankが
  blocked cell / ghost handoff / transition blend を適用して
  自rank所有セルの `laser_dep` へ反映する。
- **2D_RZ**：
  レイトレース結果（LaserMeshノード上の吸収パワー [erg/s]）を各rankが保持し、
  **自rank所有セルにのみ** scatter-add する。
  1. 各rankが同一 LaserMesh 吸収場を保持（§12.4.2 の MPI_Allreduce(SUM) 後）
  2. 各 HydroMesh セルの中心 \((r_c, z_c)\) を LaserMesh 上で逆引きし、
     近傍4ノードの deposit から bilinear 補間で沈着パワーを計算（CUDA_KERNELS L5 準拠）
  3. 所有セル判定（`PartitionInfo`）で自rank所有セルのみ加算
  4. パワー→エネルギー変換：`laser_dep[c] *= Δt` [erg]（§5.8.1 準拠、Δt乗算は転写段で1回のみ）
  5. 他rank所有セルへの加算は行わない（通信しない）

いずれの経路でも、各セルへの加算責務が単一rankに一意化されるため、
重複加算と取りこぼしを同時に防げる。

---

### 12.5 演算子固有の越境処理

各物理演算子における分割境界の処理を定義する。

#### 12.5.0 v1 実装正規（Option C、M18 実装 2026-07）

以下が実装された正規プロトコルであり、後続の各小節の M18 前設計記述に優先する
（実装詳細・測定は docs/design/mpi_m18_20_20260717.md §6g–§6n）：

- **Hydro 2D（コーナー力）**: 力の producer/scatter は ghost 含み窓（fw）で
  全 rank が両側寄与を積む（コーナー力専用 MPI 交換なし）。前提となる ghost
  入力（pq = Pe+Pi+Q、svec、活性 mask、EOS closure 出力、ρ）は §12.2.3 の
  staged 交換 + ghost 再計算（同一式・交換済み入力 → owner-bit 一致）で供給。
  position commit 後の node 交換順序不変条件（§12.2.3 項 2–3）が
  界面コーナーの bit 相殺を保証する。
- **FLD-2D 分散 CG**: dot は所有窓（連続 span）の部分和 + Allreduce(SUM) で
  α/β/収束判定を rank 一様化。探索方向 p は反復毎に ghost 帯交換。
  解 x の ghost は交換により厳密（x-ghost exactness 不変条件）。
  CUDA Graph は MPI 時に無効化。外側 Picard 残差・材料 Newton の retry 判定は
  所有窓 + Allreduce(MAX) で rank 一様（rank-local 判定は P>1 deadlock を生む）。
- **SN-2D KBA**: 波面カーネルを oriented diagonal clamp で所有 i-範囲に窓化し、
  方向符号で分割起動（sign-split）。(m, sign) 毎の plane-hop pipeline で
  上流 rank の界面 r-face flux を blocking 交換。界面 unique-face の和完成は
  disjoint writer 集合の sendrecv-add。moment 層は phi_sweep / rad_E / Te の
  ghost 交換 + 近界面 face-flux plane 転写（donor-θ の ghost credit）。
  global-dot 並べ替えが無いため P 間 ~bitwise。
- **Laser（raytrace + LaserMesh）**: LM 射影は partition-of-unity（LM 節点を
  locate(c00) で一意所有 → owner 計算 + exact-zero Allreduce 組立 = rank 間
  machine-exact）。ray は rank 分割せず全 rank が複製 trace するため、trace 入力
  {ρ, Te, z̄, volFrac, x} は呼び出し毎に owner-true へ全域 gather（1D: 全線
  Allgatherv、2D: r-slab 連続 span Allgatherv）。deposit は所有窓 mask で適用し、
  保存リスケール係数は gather 済み全配列の正準順全和を rank0 から exact-zero
  Allreduce で配布（rank 数不変 bit）。IMC 期の deposit Allreduce（複製 trace に
  ×N_ranks を掛ける phantom）は撤去済み。
- **Burn**: 1D は全線複製（入力 Allgatherv + rank0 tally share）で P 間 bitwise。
  2D Corman α拡散は FLD-2D CG パターンの鏡写し（所有窓 dot + Allreduce、
  x0/p の交換）。
- **v1.1 繰延 guard**: SNB / CBET / hot-electron / Braginskii 粘性 /
  per-material conservation は MPI 時に FATAL（fail-loud、黙走禁止）。

#### 12.5.1 Hydro：コーナー力の分割境界処理

Wilkins型コーナー力（§3.2.6）における分割境界処理：

現行実装（M18 前）は単一GPU運用で、物理境界では H16 相当の境界条件を
境界ノードへ直接適用する（ゴーストセル層なし）。H4 はその境界ノード状態を用いる。

MPI 実装（M18）では **ゴーストセル方式**（§12.1.3 Approach B）を採用する：
1. Hydro フェーズ冒頭のセルハロー交換（§12.2.2）でゴーストセルの P, Q, ρ 等を最新化
2. H4（compute_corner_force）が所有セル＋ゴーストセルの全寄与を計算
3. \(n_{ghost}=1\) により、共有境界節点に寄与する全セルがローカルに可視
4. **コーナー力専用のMPI交換は不要**

Predictor-Corrector（PC）内では P/Q が局所更新されるため、Predictor 後のゴーストセル圧力は
ステップ冒頭の値から stale になるが、§3.2.12 の時間中心化 \(P^{n+1/2}=(P^n+P^{pred})/2\) により
O(Δt²) の精度が保たれる。ゴーストセルの stale P は境界 1 層のみの影響であり、
物理的に無視可能である（CUDA_KERNELS §9 stale window 注記参照）。

Phase 末の節点ハロー交換（x_r, x_z, v_r, v_z）で共有節点の速度・位置を再同期する。

**`hydro_active` のrank境界処理**（§2.1.1）：
- `hydro_active` フラグはハロー交換でゴーストセルにも反映する（§12.2.2）
- rank境界の共有節点における `node_active` のOR判定は、隣接rankのゴーストセルの
  `hydro_active` を参照して計算する
- 非活性ゴーストセルからのコーナー力寄与はゼロ（フラグに基づき送信側でマスク）

#### 12.5.2 Conduction：Kershawステンシル

Kershaw 9点差分（Appendix A）での分割境界処理：

- 9点ステンシルは隣接1セルのみを参照（Appendix A.2）
- **\(n_{ghost}=1\) のハロー交換で完全に対応可能**
- Conductionサブステップ毎にハロー交換を実行（§12.2.3、項目3,4）

特別な処理は不要：ハロー交換後はローカルの Kershaw ルーチンがそのまま動作する。

#### 12.5.3 ALE：Winslow Jacobi反復

ALE rezone（§3.3.3 Winslow equipotential）での分割境界処理：

- Winslow Jacobi反復は隣接節点座標のみを参照
- **各反復でハロー交換が必要**（節点座標の更新を隣接rankへ伝搬）
- K回の反復 → K回のハロー交換

**通信頻度の注意**：K=50の場合、rezone 1回あたり50回のハロー交換。
ただし交換データは節点座標（double×2）のみであり、通信量は小さい。

ALE remap（§3.3.4）は rezone後に1回のcellフィールド交換で対応。

#### 12.5.4 Radiation：エネルギー収支の全rank還元

放射輸送ステップ後のグローバルエネルギー収支計算：

- 各rankの局所エネルギー量（吸収、放射、流出）を `MPI_Allreduce(SUM)` で集約
- 対象量：
  - \(E_{abs}^{total}\)：全セル吸収エネルギー合計
  - \(E_{emit}^{total}\)：全セル放射エネルギー合計
  - \(E_{escape}^{total}\)：境界脱出エネルギー合計
  - \(E_{census}^{total}\)：census粒子エネルギー合計
  - \(N_{particles}^{total}\)：全粒子数
  - \(N_{mode\_switch}^{total}\)：モード変換回数
- 収支チェック：§10.2 の \(\varepsilon_{budget}\) 定義に従い、分母に \(E_{denom} = \max(E_{total}^n, E_{source}, 10^{-20})\) を使用する（§11.1参照）

**エネルギー収支 MPI_Allreduce のタイミング**：

エネルギー収支の `MPI_Allreduce` は放射ステップの最終粒子移送交換完了**後**に実行する。
この時点で：
- 全 emigrant 粒子は受信側ランクで処理済み
- \(E_{census}^{total}\) は各粒子の現在所在ランクで計上
- \(E_{escape}\) は各ランクのローカル集計を `Allreduce(SUM)` で合算

タイミング（正規 reduction リストは CUDA_KERNELS §9 Phase 6 が規範的）：
```
R8/R9 transport → R12 roulette → [MPI] final emigrant exchange →
[MPI] Allreduce(SUM): E_kin, E_int_e, E_int_i, E_rad, E_escape[G],
                      E_numerical_loss, E_floor_injected, E_safety,
                      step_E_pdV_bdry, step_E_Marshak_in, step_E_solver,
                      step_laser_dep_total →
// step_laser_escaped = E_laser_incident - step_laser_dep_total (Allreduce後に算出)
[MPI] Allreduce(MIN): dt_hydro, dt_cond, dt_rad →
[MPI] Allreduce(MAX): error_flags →
energy_budget check
```

#### 12.5.4.1 Tier-A conservation residual diagnostics

For production-audit Tier A, the driver emits per-step dimensionless residuals
to `/diagnostics/conservation/v1/*` in the history file.  The postprocessor
takes the maximum absolute value over all emitted steps.

Mass:
\[
\varepsilon_M^n = \frac{|M^n - M^0|}{|M^0|}, \qquad
M^n = \sum_c \rho_c^n V_c^n .
\]

R/Z momentum:
\[
\varepsilon_{P_k}^n =
\begin{cases}
|P_k^n-P_k^0|/|P_k^0|, & |P_k^0| > 10^{-30},\\
0, & |P_k^n-P_k^0|\le 10^{-30}\ {\rm and}\ M^0 v_{\rm th}^0\le 10^{-30},\\
|P_k^n-P_k^0|/\max(M^0 v_{\rm th}^0,10^{-30}),
  & |P_k^0| \le 10^{-30},
\end{cases}
\quad
P_k^n = \sum_c \rho_c^n V_c^n v_{k,c}^n ,
\]
where \(k \in \{R,Z\}\), \(M^0=\sum_c\rho_c^0V_c^0\),
\(v_{\rm th}^0\) is the initial material thermal velocity scale derived from
the EOS-closed state, and the 2D RZ cell-centered velocity is the arithmetic
mean of the active corner-node velocities.  For legacy quads this is the four
corner-node mean; for cap triangles it is the three active-node mean.  For
multiblock meshes this average uses CSR cell-node lookup, not structured
`(i,j)` addressing.  The
physical-scale zero-reference branch is used for near-at-rest audit decks where
the exact component momentum can be zero or wall-balanced; it prevents
roundoff-level numerator noise from being divided by a collapsed reference.
If both the numerator and the physical momentum scale are negligible, the
relative momentum drift is reported as zero.  Absolute conservation residuals
remain the primary momentum check.
Mass and energy diagnostics keep the legacy absolute fallback for zero
reference values.

GCL:
\[
\varepsilon_{\mathrm{GCL}}^n =
\max(\varepsilon_M^n,
     \varepsilon_{E_{\mathrm{int}}}^n,
     \varepsilon_{P_R}^n,
     \varepsilon_{P_Z}^n)
\]
using the existing ALE GCL reference/residual evaluator captured immediately
before the ALE operator and evaluated immediately after accepted rezone/remap.
This is the conservative ALE-operator fallback accepted for the Tier-A audit
path when explicit swept-volume face accounting is not emitted.

#### 12.5.5 計算-通信オーバーラップ

マルチGPU実行時、各演算子のハロー交換中にGPUがアイドルになる時間を削減するため、
**内部セル計算と境界通信を非同期で重畳**する。

**v1.0既定**：オーバーラップ **無効**（逐次実行）。有効化は `Parallel.gpu_optimization.compute_comm_overlap=True`。

##### (a) 内部/境界セルの分類

各rankのローカルセルを2種に分類する：

- **境界セル（halo-dependent）**：ゴーストセル（§12.2）の値に依存するセル。
  Kershaw 9点ステンシル（§A.1）は近接1層が必要なため（n_ghost=1、ARCHITECTURE §7.1.1 PartitionInfo 準拠）、
  \(i = 0\) または \(i = n_r-1\) または \(j = 0\) または \(j = n_z-1\) のセルが該当。
  - 1D_SPH：先頭1セル＋末尾1セル
  - 2D_RZ：外周1層のセル帯（各辺で1セル幅、9点ステンシルの角近傍を含む）
- **内部セル（halo-independent）**：上記以外。ゴーストセルの値に依存しない。

> **分類の事前計算**：初期化時に `uint8_t cell_zone[n_cells]` を生成する。
> `cell_zone[c] = 0`（内部）or `1`（境界）。セルインデックスの算術判定のみで O(N)。

##### (b) オーバーラップ実行モデル

各演算子のハロー交換を含むフェーズで、以下の手順で実行する：

```
Stream compute:  カーネル起動（内部セルのみ）
Stream comm:     halo_pack → cudaStreamSynchronize(comm) → MPI_Isend/Irecv
                → MPI_Waitall → halo_unpack
cudaStreamSynchronize(compute)
cudaStreamSynchronize(comm)
Stream compute:  カーネル起動（境界セルのみ）
cudaStreamSynchronize(compute)
```

1. compute ストリームで **内部セル**のカーネルを起動（ゴーストセル非参照のため即実行可能）
2. 同時に comm ストリームで **ハローパック** → **MPI通信** → **ハローアンパック** を実行
3. 両ストリーム同期後、compute ストリームで **境界セル**のカーネルを起動

##### (c) 適用可能な演算子

| 演算子 | オーバーラップ可否 | 理由 |
|:-------|:----------------:|:-----|
| Hydro（§3） | ○ | コーナー力計算（H4）で近接セルを参照するが、内部セルはゴースト不要 |
| Conduction（§4） | ○ | Kershaw 9点ステンシル（C2, C3）は近接1層。内部セルはゴースト不要 |
| Radiation（§6, §7） | △ | 粒子輸送は任意セルを参照するため、セル分割は適用困難。ただしタリーfinalizeとFleck計算はセルベースで適用可能 |
| Laser（§5） | × | LaserMesh は全rank複製（§12.4.2）のためハロー交換なし |

##### (d) 正当性の保証

- **内部セルの独立性**：構造格子上のステンシル演算（Kershaw, Hydro corner force）は近接 \(n_{ghost}=1\) 層のみを参照する。
  内部セルの計算はゴーストセルの値を使用しないため、ハロー交換の完了を待たずに正しく計算できる。
- **境界セルの依存性**：境界セルはゴーストセルを参照するため、ハロー交換の完了後にのみ計算する。
- **粒子輸送の例外**：IMC/DDMC粒子は輸送中に任意のセルを横断する可能性があり、
  境界近傍の粒子はゴーストセル情報を必要とする。よって粒子輸送カーネル自体には内部/境界分割を適用しない。
  ただしFleck計算（R1）、DDMC mode judge（R2）、タリーfinalize（R10）等のセルベースカーネルには適用可能。

##### (e) 性能モデル

オーバーラップによる1演算子あたりの時間削減：

\[
\Delta T_{overlap} = \min(T_{interior}, T_{comm}) - T_{overhead}
\]

ここで：
- \(T_{interior}\)：内部セルカーネルの実行時間
- \(T_{comm}\)：halo pack + MPI通信 + halo unpack の合計時間
- \(T_{overhead}\)：2回のカーネル起動オーバーヘッド（~10 μs）

**典型値**（2D_RZ 500×250、4 GPU）：
- \(T_{interior} \approx 0.8 \times T_{total}\)（内部セルが ~80%）
- \(T_{comm} \approx 0.5\) ms（ハロー交換）
- Hydro半ステップでの削減：~0.4 ms（~25% のハロー待ち時間を隠蔽）
- ステップ全体（Hydro×2 + Conduction）での削減：~1.2 ms（~2–3% of 50 ms/step）

4 GPU以上のスケーリングでは \(T_{comm}\) が増大するため、オーバーラップの効果が相対的に増加する。

---

### 12.6 負荷分散

#### 12.6.1 静的分割（v1.0既定）

v1.0では**静的分割**を既定とする。

- 初期化時に各rankのセル範囲を決定し、シミュレーション中は変更しない
- セル数ベースの均等分割（§12.1.1、§12.1.2）

**静的分割の限界**：
- IMC/DDMC粒子は光学的に厚い領域に集中する傾向があり、
  粒子数（計算負荷）がセル数に比例しない
- 爆縮問題では中心部が高密度化し、中心を持つrankに負荷が偏る

#### 12.6.2 粒子レベル負荷分散（既定OFF）

粒子数の偏りを軽減するための work-stealing 方式：

- **imbalance指標**：\(\alpha = N_{max} / \bar{N}\)（最大粒子数 / 平均粒子数）
- \(\alpha > \alpha_{threshold}\)（既定 1.5）の場合に発動
- 過負荷rankの census 粒子の一部を隣接rankへ移動
- **制約**：移動先rankのゴーストセル内に位置する粒子のみ（遠距離移動は禁止）

> v1.0では既定OFF。有効化は `parallel.particle_balance.enabled=true`。

#### 12.6.3 動的再分割（将来拡張）

将来の検討事項として記録する（v1.0ではスコープ外）：
- ParMETIS等のグラフ分割ライブラリによる動的再分割
- 再分割時のセル・粒子・checkpoint整合性の保証が大きな設計課題
- ALE rezone後のメッシュ品質変化に連動する再分割

---

### 12.7 再現性（Reproducibility）

モンテカルロ法の性質上、TENRYUは **システム全体としてbitwise再現を要求しない**。
**統計的再現**（同一seed・同一構成で主要物理量の平均・分散が一致）のみを保証する。
ただし、**決定論的演算子**（Hydro、Conduction）は同一入力で bitwise 一致が得られる。
放射フィールド（IMC/DDMC）は Persistent Warp の非決定性および atomicAdd 順序により
bitwise 一致は保証されず、統計的一致（VERIFICATION §16.8 の CV ≤ 0.1% 基準）を要求する。

#### 12.7.1 RNG分割：cuRAND device API による Philox4x32-10

**cuRAND device API**（`curand_kernel.h`、CUDA Toolkit 同梱）の
`curandStatePhilox4_32_10_t` を使用する。Philox4x32-10 はカウンタベースRNGであり、
任意のストリーム位置に O(1) でアクセスできる。TENRYUでは以下の
**唯一の権威的マッピング**を固定する（ARCHITECTURE §4.1 と整合）。

**cuRAND 初期化マッピング**：
```cuda
curandStatePhilox4_32_10_t rng_state;
curand_init(
    /*seed=*/        global_id ^ user_seed, // uint64: Philox key（粒子ID × ユーザーseed）
    /*subsequence=*/ step_number,           // uint64: ストリーム分離（ステップ間）
    /*offset=*/      rng_counter,           // uint64: 消費済み乱数位置（O(1) skipahead）
    &rng_state
);
```
ここで `user_seed` は `Main.seed`（SPECIFICATION §6.4.1、既定 12345）から取得する uint64 値である。

- **seed = global_id ^ user_seed**：粒子固有の 64ビット識別子とユーザー指定seedのXOR → 内部的に Philox key を導出。`Main.seed` を変更すると全粒子の乱数ストリームが変化し、独立な統計サンプルを生成できる
- **subsequence = step_number**：タイムステップ番号 → 同一 global_id でもステップごとに独立ストリーム
- **offset = rng_counter**：当該ステップ内で消費済みの乱数個数。
  新規生成粒子（R6/R13）は rng_counter = 0 で開始。census 粒子は前ステップの rng_counter を引き継ぐが、subsequence = step_number が変化するため独立ストリームとなり問題ない。
  Philox はカウンタベースのため skipahead は O(1)（カウンタ加算のみ、逐次スキップ不要）

**乱数生成**：
```cuda
double xi = curand_uniform_double(&rng_state);  // U(0,1]（0は除外、1は含む）
```
`curand_uniform_double(curandStatePhilox4_32_10_t*)` は `curand(state)` の
uint32 出力を1語消費し、`(x + 1) / 2^32` に写像して倍精度 \(U(0,1]\) を返す
（`curand_uniform.h` 実装準拠）。

> **cuRAND 採用の理由**：
> (1) NVIDIA による統計品質検証済み（TestU01 BigCrush 全合格）、
> (2) PTX intrinsics による最適化された10ラウンド乗算、
> (3) CPU 実装（PhiloxCpu）と同一の1語消費マッピングで CPU/GPU 再現性を確保、
> (4) `curand_kernel.h` はヘッダオンリーであり `libcurand.so` リンク不要。
> ライセンスは NVIDIA EULA（LICENSE.md 参照）。

**global_idの構成**（step_base によるステップ間衝突回避を含む、§下記 Census 回避方式参照）：
- \(\text{global\_id} = \text{step\_base} + \text{rank\_offset} + \text{local\_id}\)
- \(\text{step\_base} = \text{step} \times N_{max\_per\_step}\)（\(N_{max\_per\_step} = 2^{40}\)）
- \(\text{rank\_offset} = \sum_{p'<p} N_p^{emit}(p')\)（rank \(p\) より前の
  全rankの射出粒子数累積、`MPI_Exscan` で計算）
- diffusion exit 粒子は低位 source emission 範囲を使わず、local_id の高位予約範囲
  \([2^{39},2^{39}+2^{38})\) を rank 数で等分した subrange を使う（§7.1.2b）
- diffusion-interface spawn 粒子は local_id の高位予約範囲
  \([2^{39}+2^{38},2^{40})\) を rank 数で等分した subrange を使う（§7.1.2e）
- census粒子は前ステップの `global_id` を引き継ぐ（step番号の変化で新しいストリームとなる）

**`MPI_Exscan` の実行タイミング**：`radiation_step` 冒頭、ソース生成カーネル前。各ランクの \(N_p^{emit}\) を計算した後に `MPI_Exscan` を実行し、`global_id = step_base + rank_offset + k`（\(k = 0, \dots, N_p^{emit}-1\)）を決定する。ここで **\(N_p^{emit}\) は R6（体積ソース）と R13（Marshak境界ソース）の合計**である。R6 粒子は \(k = 0, \dots, N_{R6}-1\)、R13 粒子は \(k = N_{R6}, \dots, N_p^{emit}-1\) の区間を使用する。diffusion exit 粒子と diffusion-interface spawn 粒子は high local-id subrange を使うため、この `MPI_Exscan` の対象外である。census 粒子は既存の `global_id` を維持し、`MPI_Exscan` の対象外とする。

**Census粒子との global_id 衝突回避**：
新規粒子の global_id 空間は、census粒子の既存 global_id と衝突しないよう設計する。

方式：offset + local_index
- step n での新規粒子：global_id = step_base + rank_offset + local_index
  - step_base = step × N_max_per_step（N_max_per_step = 2^40、ステップあたり最大粒子数）
  - rank_offset: `if (n_ranks==1) rank_offset=0; else { MPI_Exscan(N_emit_local, SUM, &rank_offset); if (rank==0) rank_offset=0; }`
    （MPI_Exscan は rank 0 の recvbuf が MPI 規格上未定義のため、rank 0 では明示的に 0 を設定する）
  - local_index = 0, 1, 2, ..., N_emit_local - 1
- diffusion exit 粒子：global_id = step_base + \(2^{39}\) + rank × floor(\(2^{38}/n_{ranks}\)) + local_index
- diffusion-interface spawn 粒子：global_id = step_base + \(2^{39}+2^{38}\) + rank × floor(\(2^{38}/n_{ranks}\)) + local_index
- census粒子：前ステップで割り当てられた global_id をそのまま引き継ぐ
- step_base のステップ依存性により、異なるステップで生成された粒子の global_id は必ず異なる
- N_max_per_step は uint64 空間（2^64）に対して十分小さいため、
  2^24 ステップ（= 16,777,216 ステップ）までオーバーフローしない
- **ランタイム検査（必須）**：タイムループ冒頭で `step < 2^24` を検証し、
  超過時は FATAL エラーとする。ICF典型計算（~10^4–10^6 step）では到達しないが、
  安全ネットとして実装する

**非衝突の証明**：census 粒子は creation_step \(< n\) で生成されたため、
そのglobal_id は \(\text{global\_id} \in [\text{creation\_step} \times N_{\max\_per\_step},\; (\text{creation\_step}+1) \times N_{\max\_per\_step})\) の区間に属する。
ステップ \(n\) の新規粒子は \(\text{global\_id} \in [n \times N_{\max\_per\_step},\; (n+1) \times N_{\max\_per\_step})\) の区間に属する。
\(\text{creation\_step} < n\)（全 census 粒子について成立）であるため、
これらの区間は互いに素（disjoint）であり、global_id の衝突は発生しない。
Philox のカウンタ空間でも、counter[1] = step_number が異なるため、
同一 global_id を持つ粒子が仮に存在しても（存在しないが）異なるストリームとなる。

**cuRAND マッピングの数学的対応**：
cuRAND の `curand_init(seed, subsequence, offset)` は内部で seed → Philox key、
subsequence → ストリーム分離（key の修飾）、offset → カウンタ位置 に写像する。
具体的な Philox key/counter への内部写像は NVIDIA 実装に委ねるが、
以下の保証が cuRAND API 仕様で成立する：
- 異なる `(seed, subsequence)` ペア → 統計的に独立なストリーム
- 同一 `(seed, subsequence)` に対する offset → O(1) skipahead（カウンタ加算）
- 各粒子のRNGストリームはrank分割に依存しない
  （同一 global_id × step_number で同一乱数列を生成する）

**cuRAND state のライフサイクル**：
- cuRAND state（`curandStatePhilox4_32_10_t`、44バイト）はカーネル実行中に
  **レジスタ常駐**する。PhotonPool（グローバルメモリ）には保存しない
- PhotonPool 格納は `rng_counter`（uint32）のみ。カーネル終了時に消費済み乱数数を書き戻す
- **inter-kernel continuity**：IMC→DDMC遷移時、DDMCカーネルが
  `curand_init(global_id ^ user_seed, step_number, rng_counter, &state)` で O(1) 復元する

**粒子ごとのRNGストリーム割り当て**：

cuRAND Philox4x32-10 のカウンタベース設計により、各粒子は独立したRNGストリームを持つ。

- **seed** = global_id ^ user_seed（uint64、粒子固有ID × Main.seed、§12.7.1 準拠）
- **subsequence** = step_number（uint64、ステップごとに変化 → 新ストリーム）
- **offset** = rng_counter（uint32→uint64、ステップ内乱数消費位置。新規粒子は0から開始、census粒子は前ステップの値を引き継ぐ）

`curand_uniform_double()` は1回の呼び出しで1つの倍精度 \(U(0,1]\) を返す。
内部で消費する uint32 は1個である（Philox実装）。
rng_counter は `curand_uniform_double()` の呼び出し回数を追跡する。

**各イベントの乱数消費数**：

| イベント | 消費数 | 用途 |
|---------|--------|------|
| IMC方向サンプリング | 2 | μ, φ |
| IMC位置サンプリング | 1-2 | r (1D) or (t_face, φ) (2D) |
| 散乱距離 | 1 | -ln(ξ)/σ |
| 散乱群再サンプリング | 1 | CDF逆変換 |
| Russian roulette | 1 | 生存判定 |
| DDMC イベント時間 | 1 | -ln(ξ)/(cΣ) |
| DDMC イベント種別 | 1 | 吸収/リーク選択 |
| DDMC→IMC 位置 | 1-2 | 面上位置 |
| DDMC→IMC 方向 | 2 | μ, φ |

**IMC/DDMCモード切替時**：
同一カウンタストリームを継続使用（モード切替で rng_counter はリセットしない）。

**Census粒子**：
次ステップでは subsequence = new_step_number に更新（独立ストリーム生成）。
offset = rng_counter は前ステップの値を**引き継ぐ**（リセットしない）。
key は不変（global_id が同一のため）。
注: subsequence の変化により、offset 値に関わらず独立なストリームが保証される（Philox カウンタベース設計）。
rng_counter の引き継ぎはストリーム位置の「ずらし」に過ぎず、統計的独立性に影響しない。

**Persistent Warp実行モデルとRNG**：
Persistent Warp（§6.6, CUDA_KERNELS §6.4）では、各スレッドが複数の粒子を順次処理する。
粒子切替時にRNG状態（rng_counter）はSoAに保存・復元される。
global_idベースのキー構成により、どのスレッドが処理しても同一粒子は同一RNGストリームを使用する。
ただし粒子のスレッドへの割り当て順序は非決定的であり、タリーへのatomicAdd順序も非決定的となるため、
bitwise再現は保証されない。統計的再現は各粒子のRNG独立性により保証される。

#### 12.7.2 粒子順序：移動後のソート

rank間移動後の粒子ソートは**オプション**（既定OFF）。

- セルソート（§6.5）がキャッシュ局所性を担保するため、global_idソートは再現性目的では不要
- デバッグ時に粒子追跡を容易にするため、`sort_after_migration=True` に設定できる
- ソートコスト：\(O(N)\)（基数ソート）だが、移動粒子数は通常少数のため無視可能

#### 12.7.3 MPI Reduction

`MPI_Allreduce`（`MPI_SUM`）を標準APIで使用する。
浮動小数点加算の結合順序はMPI実装依存であり、rank数変更時に最下位ビットが変動し得るが、
統計的再現には影響しない。

> **旧設計との変更点**：bitwise再現のために採用していた Gather + Root逐次加算プロトコルは廃止。
> 標準 `MPI_Allreduce` は O(log P) レイテンシであり、Gather + Bcast の O(P) より効率的。
> P ≤ 64 の規模では差は小さいが、将来のスケーリングを考慮して標準APIを採用する。

**適用箇所**：
- エネルギー収支の全ランク合計
- LaserMesh の Allreduce
- 粒子統計の全ランク集約

#### 12.7.4 Persistent chunk cooperative-grid reductions

The persistent chunk kernel uses fixed-order two-stage grid reductions when it
is launched as a CUDA cooperative grid. Each block first emits one P9
fixed-order partial, then block 0 reduces the partial slab with the same padded
power-of-two halving tree. This is deterministic for a fixed cooperative grid
size. Changing the cooperative grid size changes the reduction order and is a
tolerance-class change, like changing `blockDim`.

---

