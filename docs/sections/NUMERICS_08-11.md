<!-- 分割元: docs/NUMERICS.md | このファイルは参照用です。原本（docs/NUMERICS.md）が権威です。 -->
## 8. 境界条件
### 8.1 流体

**Lagrangianメッシュの流体境界条件**：

Lagrangian法では質量がセル境界を横切らないため、
境界条件は主に **運動量**（圧力/速度）と **エネルギー**（熱流束）に対して適用する。

**(a) 外側境界**（1D_SPH: \(r=r_N\)、2D_RZ: 外周面）：

| 種別 | 速度条件 | 圧力条件 | 熱流束条件 | 用途 |
|------|---------|---------|-----------|------|
| `free`（既定） | 自由（運動方程式で決定） | \(P_{ext}=0\) | \(\partial T/\partial n = 0\)（断熱） | ICF標準 |
| `fixed` | \(v=0\)（全速度固定） | 壁圧力（反力） | \(\partial T/\partial n = 0\) | 剛体壁 |
| `reflect` | \(v_n = 0\)（法線固定、接線自由） | 壁圧力（反力） | \(\partial T/\partial n = 0\) | スリップ壁 |
| `pressure` | 自由 | \(P_{ext} = P_{drive}(t)\) | \(\partial T/\partial n = 0\) | 外部駆動 |

ゴーストセル（仮想セル）を用いた実装：
- `free`：ゴーストセルの圧力 \(P_{ghost} = 0\)、密度 \(\rho_{ghost} = \rho_{boundary}\)
- `fixed`：ゴーストセルの速度 \(v_{ghost} = -v_{boundary}\)（全速度成分反転で \(v=0\) を実現）
- `reflect`：ゴーストセルの法線速度 \(v_{n,ghost} = -v_{n,boundary}\)、接線速度 \(v_{t,ghost} = v_{t,boundary}\)（法線のみ反転でスリップ壁を実現）
- `pressure`：ゴーストセルの圧力 \(P_{ghost} = P_{drive}(t)\)（§3.1.4/§3.2.5の加速度式がcell-center圧力差を直接使うため、face平均ではなく直接代入）

圧力境界条件のゴーストセル値（`pressure` 詳細）：
- \(P_{ghost} = P_{drive}(t)\) — 駆動圧力（FrozenTable1D）
- \(\rho_{ghost} = \rho_{interior}\) — 最近接内部セルの密度をコピー
- \(u_{ghost} = u_{interior}\)（法線成分）— ゼロ勾配外挿（速度自由；§8.1表の「自由」に対応）
- \(T_{e,ghost} = T_{e,interior}\) — 温度は内部セルをコピー
- \(T_{i,ghost} = T_{i,interior}\)

**(b) 中心境界**（1D_SPH: \(r=0\)）：

- \(u_r(r=0) = 0\)（対称条件）
- \(\partial T / \partial r\big|_{r=0} = 0\)（ゼロ流束、§4.3の伝導BCと統合）
- 実装：最内セルの左隣に仮想セルを配置し、物理量を鏡像反射

**(c) RZ軸**（2D_RZ: \(R=0\)）：

- \(v_R(R=0) = 0\)（軸対称条件）
- \(v_Z\) は自由（軸方向運動を許容）
- \(\partial Q / \partial R\big|_{R=0} = 0\)（全スカラー量のゼロ勾配）
- 実装：R=0上のノードの R方向速度成分を強制的にゼロに設定

**(d) 2D_RZゴーストセルの配置と値の設定**：

2D RZ構造格子では、4つの境界にそれぞれ1層のゴーストセルを配置する：

| 境界 | ゴーストセル範囲 | スカラー量 | 速度 |
|------|-----------------|-----------|------|
| R=0軸（\(i=-1\)） | 鏡像反射 \(\phi_{-1,j}=\phi_{0,j}\) | ゼロ勾配 | \(v_{R,ghost}=-v_{R,interior}\), \(v_{Z,ghost}=v_{Z,interior}\) |
| R=R\(_{max}\)（外周） | 条件依存（下記） | 条件依存 | 条件依存 |
| Z=Z\(_{min}\) | 条件依存 | 条件依存 | 条件依存 |
| Z=Z\(_{max}\) | 条件依存 | 条件依存 | 条件依存 |

R\(_{max}\), Z\(_{min}\), Z\(_{max}\) の各面は種別を面ごとに独立に指定できる。
- **R\(_{max}\)**：`free`/`fixed`/`reflect`/`pressure` のいずれか
- **Z\(_{min}\)**, **Z\(_{max}\)**：`free`/`fixed`/`reflect` のいずれか（`pressure` は R\(_{max}\) のみサポート。SPECIFICATION §6.4.1 参照）

ゴーストセル値は (a) と同じ規則で設定する。

**実装注記（現行 v1.0, H16）**：
現行実装は境界条件を物理境界ノードへ直接適用する（ゴーストセル層なし）。
`boundary_2d.cu` の境界処理（H16相当）は境界ノード速度/座標を直接拘束する。
ゴーストセルバッファ方式は MPI 領域分割（M18）で導入する。

**マルチrank境界**（§12参照）：rank間境界ではゴーストセルの値は
ハロー交換（§12.2）で隣接rankの内部セルから取得する。
上記のゴーストセル設定は物理境界面にのみ適用する。

### 8.2 輻射

**vacuum**：境界到達で粒子消滅、流出エネルギーに計上（`E_escape += E_p`）。
ここで \(E_p\) は **境界面到達時点の粒子エネルギー**（最終セグメントの連続吸収適用後の値）であり、
重み調整（Russian roulette等）が先行して適用されている場合はその結果を反映した値である。
タリー順序：(1) 最終セグメントの吸収沈着 \(\Delta E_{dep}\) を `rad_dep` に計上、
(2) 吸収後の残余エネルギー \(E_p = E_{in} - \Delta E_{dep}\) を `E_escape` に計上、
(3) 粒子を消滅させる。これにより \(\Delta E_{dep} + E_{escape} = E_{in}\) が恒等的に成立する。
DDMCの真空境界リーク率は§7.4で定義。

**reflect**：鏡面反射（IMC粒子の方向ベクトルの法線成分を反転）。
\[
\hat\Omega' = \hat\Omega - 2(\hat\Omega \cdot \hat{\mathbf{n}})\,\hat{\mathbf{n}}
\]
エネルギー・時刻は保持。

**Marshak**（入射放射源、検証用）：
各Marshak境界面の放射温度 \(T_{r,f}(t)\)（面 \(f\) ごとに独立指定、SPECIFICATION §6.4.5 `marshak_Tr_eV: dict[str, callable]`）に基づき、境界面からIMC粒子を生成する。以下のアルゴリズムは各Marshak面に対して独立に適用する。

**Marshak境界ソース粒子の生成アルゴリズム**：

1. **各面の入射エネルギー**（半空間Marshak条件、fdm1290 Eq.22準拠）：
\[
E_{Marshak,f} = \frac{a_{eV}\,c}{4}\, T_{r,f}(t)^4 \cdot A_f \cdot \Delta t \quad [\text{erg}]
\]
ここで \(a_{eV}\) は放射定数（eV単位系、§6.1）、\(T_{r,f}\) [eV] は面 \(f\) の境界放射温度、\(A_f\) は面 \(f\) の面積。全Marshak面の合計：\(E_{Marshak,in} = \sum_f E_{Marshak,f}\)。

2. **粒子数**：\(N_{total}\) = `radiation.boundary.marshak_particles`（既定 1000/step、全Marshak面合計）。各面への配分は面積比：\(N_f = \text{round}(N_{total} \cdot A_f / \sum_{f'} A_{f'})\)
   **並列時の面積合計**（§12 準拠）：\(\sum_{f'} A_{f'}\) は **全rank の全Marshak面** の面積合計であり、`MPI_Allreduce(SUM)` で取得する（CUDA_KERNELS §9 Phase 4 Marshak配分プロトコル参照）。各rankは自身の所有 Marshak 面に対して \(N_f\) を計算し、\(n_{marshak,local} = \sum_{owned\ f} N_f\) 粒子を R13 で生成する。端数調整は最大面積を持つ面（グローバル一意に決定）に加減する。

3. **各粒子のエネルギー**（面 \(f\) に属する粒子）：
\[
E_p = \frac{E_{Marshak,f}}{N_f}
\]

4. **位置**：境界面上で一様サンプル
   - 1D_SPH（外側球面 \(r=r_N\)）：等方球面サンプリング
   - 2D_RZ（外周面）：面上で一様（R重み付き棄却法）

5. **方向**：内向き半空間のcosine分布（\(\mu = \sqrt{\xi}\)、\(\mu > 0\) は内向き法線方向）。
   Interface source angular distribution uses cosine-weighted half-space sampling: \(\mu = \sqrt{\xi}\), consistent with Lambert's cosine law for surface emission.

6. **群**：面 \(f\) の Planck 分布 \(b_g(T_{r,f})\) に比例した確率でサンプル

7. **時刻**：ステップ内一様 \(t = t^n + \xi\,\Delta t\)

**Marshak BC 粒子の面別配分**：
- 各 Marshak 境界面 \(f\) について、面面積 \(A_f\) に比例して粒子数を配分：
\[
N_{p,f} = \text{round}\!\left(N_{marshak\_total} \times \frac{A_f}{\sum_f A_f}\right)
\]
- 最低 1 粒子/面を保証（端数調整は最大面積の面に加減）。この保証のため \(N_{total} \ge n_{Marshak\_faces}\) が必須（SPECIFICATION §6.4.5 で validate 時に検証）。\(N_f = 0\) は発生しない前提で \(E_p = E_{Marshak,f}/N_f\) の 0 除算を防止する
- 2D RZ：\(A_f = 2\pi\, R_{mid} \times L_f\)（面中点 R 座標 × 面長さ）
- 面上の位置：面上で一様ランダム（1 RNG draw for face-local coordinate）
- 方向：cos-weighted half-space（2 RNG draws、NUMERICS §6.2）

> **注意**：Marshak BCはMarshak wave検証問題（VERIFICATION §7.2）で使用される。
> 本番ICFシミュレーションでは通常 vacuum BC を使用する。

### 8.3 レーザー
- 入射面：ターゲット周囲の球面（1D）または円筒/外周（2D）で定義
- LaserMesh外へ出たレイは終了（未吸収として診断）
- 臨界終了は5.2の規約

---


## 9. 移動メッシュと粒子セル再同定
Lagrangian/ALEでメッシュが動くため、粒子の空間座標は固定でも cellId が変わりうる。

### 9c. Spherical-Polar Halfplane Logical Mesh (Phase 6 Foundation)

Phase 6-1 adds a default-off 2D logical mesh mode:

```cpp
LogicalMesh2D::RectangularRZ
LogicalMesh2D::SphericalPolarHalfplane
```

The default remains `rectangular_rz` and preserves the existing uniform Cartesian RZ mesh. The opt-in value `spherical_polar_halfplane` is valid only for `Main.dimension="2D_RZ"` and reinterprets `Mesh.nr` as \(N_s\) radial shells and `Mesh.nz` as \(N_\theta\) angular zones.

For \(S_{\max}=\) `Mesh.spherical_polar_s_max` and \(\kappa=\) `Mesh.spherical_polar_kappa`,

\[
\Delta s = {S_{\max}\over N_s+\kappa},\qquad
s_i=(i+\kappa)\Delta s,\quad i=0,\ldots,N_s,
\]

\[
\theta_j={j\pi\over N_\theta},\quad j=0,\ldots,N_\theta.
\]

For `Mesh.polar_equal_mu_zoning=True` with
`Mesh.polar_center_treatment="tri_fan"`, the angular nodes instead use
equal-solid-angle spacing,
\[
\mu_j=\cos\theta_j=1-{2j\over N_\theta},\qquad
\theta_j=\arccos\!\left(\mathrm{clamp}(\mu_j,-1,1)\right),
\quad j=0,\ldots,N_\theta.
\]
The default `False` path keeps the uniform-\(\theta\) formula above.

Physical RZ node coordinates are

\[
r_{ij}=s_i\sin\theta_j,\qquad z_{ij}=s_i\cos\theta_j.
\]

Thus the logical halfplane covers \(\theta\in[0,\pi]\) with \(r\ge0\). The inner boundary is a finite regularized core at \(s_{\min}=\kappa S_{\max}/(N_s+\kappa)\); the outer boundary is \(s=S_{\max}\).

**Exact axis snap (2026-07-27).** Axis-end nodes are assigned \(r=0\) exactly by logical index (\(j=0\) and \(j=N_\theta\)), never by evaluating \(s_i\sin\theta_j\) (which yields \(r\approx1.2\times10^{-16}s_i\) at \(\theta=\pi\)). The snap is applied at node construction, before any volume/area/S-vector/corner-mass derivation, uniformly across the single-block spherical-polar center treatments (annular, tri\_fan, button, polar\_in\_box prefix) and the polar-in-box collar previous-ring reconstruction. Interior nodes and non-axis-reaching wedge builders are untouched. `NODE_AXIS` classification retains its relative-tolerance machinery, which the snap now satisfies trivially.

**Equator mirror-and-snap (2026-07-27).** For symmetric-by-construction \(\theta\) ladders (uniform and `polar_equal_mu_zoning`), only the northern half \(j\le N_\theta/2\) is evaluated from \(\sin/\cos\); the equator node (even \(N_\theta\), \(j=N_\theta/2\)) is assigned \(z=\text{center}_z\) exactly, and every southern node is the bitwise mirror of its northern partner: \(r_j=r_{N_\theta-j}\) (bit copy), \(z_j=\text{center}_z-(z_{N_\theta-j}-\text{center}_z)\) (sign-flipped offset). The southern axis node thereby inherits \(r=0\) exactly and \(z=\text{center}_z-s\) exactly. User-specified ladders (`explicit_nodes_theta`, `grid_segments_theta`) and the box-anchored `polar_in_box` paths keep direct per-node evaluation — the equator has no privileged status in user geometry.

For a spherical-polar logical cell \([s_i,s_{i+1}]\times[\theta_j,\theta_{j+1}]\), the exact swept volume is obtained from \(dV=2\pi r\,dr\,dz\). With \(r=s\sin\theta\), \(z=s\cos\theta\), and \(|\partial(r,z)/\partial(s,\theta)|=s\),

\[
dV = 2\pi s^2\sin\theta\,ds\,d\theta.
\]

Integrating gives

\[
V_{ij}={2\pi\over3}\left(s_{i+1}^3-s_i^3\right)
\left(\cos\theta_j-\cos\theta_{j+1}\right).
\]

The corresponding exact spherical face areas used for Phase 6 foundation checks are

\[
A_s(s;\theta_j,\theta_{j+1})=
2\pi s^2\left(\cos\theta_j-\cos\theta_{j+1}\right),
\]

\[
A_\theta(\theta;s_i,s_{i+1})=
\pi\sin\theta\left(s_{i+1}^2-s_i^2\right).
\]

Boundary edge tags are assigned in logical-edge space:

| Logical edge | BoundaryKind |
|---|---|
| \(\theta=0\) | `RZAxisTheta0` |
| \(\theta=\pi\) | `RZAxisThetaPi` |
| \(s=s_{\min}\) | `SphericalInnerCore` |
| \(s=S_{\max}\) | `SphericalOuterFree` |
| all other logical edges | `Interior` |

Rectangular RZ keeps its existing behavior and receives rectangular location tags (`RectRInnerAxis`, `RectROuter`, `RectZBottom`, `RectZTop`) for mesh metadata.

Phase 6-2 adds hydro-only boundary dispatch for `spherical_polar_halfplane`.
`boundary_2d.cu` branches on `Mesh.logical` before the rectangular RZ path, so
`rectangular_rz` keeps the existing `Numerics.hydro.boundary_2d` dispatch and
node constraints. In spherical-polar mode, boundary conditions are selected from
the logical-edge `BoundaryKind` tags:

- `RZAxisTheta0` and `RZAxisThetaPi`: cylindrical-axis symmetry. Boundary nodes
  are constrained to \(r=0\) with \(u_r=0\).
- `SphericalInnerCore`: initial hydro reflecting inner core. The spherical normal
  velocity is projected out at \(s=s_{\min}\):
  \[
  u_s = u_r {r\over s} + u_z {z\over s},\qquad
  \mathbf{u}\leftarrow\mathbf{u}-u_s\left({r\over s},{z\over s}\right).
  \]
  A movable inner boundary is deferred.
- `SphericalOuterFree`: free/outflow. No direct node constraint is applied,
  matching the existing `free` behavior.

`logical_mesh_2d="spherical_polar_halfplane"` is Phase 6-minimum hydro-only
scope. Namelist validation hard-fails if `Radiation.enabled`,
`Laser.enabled`, or `Numerics.conduction.enabled` is true, covering
FLD/IMC/DDMC, laser, and conduction before runtime initialization.

Stage I1-B adds the opt-in
`Mesh.polar_center_treatment="tri_fan"` center treatment for standalone
geometry/admissibility tests only. The structured node array is unchanged:
`node(i,j)=i*(N_theta+1)+j`, and the full `(N_s+1)*(N_theta+1)` storage remains
allocated. In `tri_fan`, the center row is pinned at the origin,
\(s_0=0\), and \(s_i=iS_{\max}/N_s\) for \(i\ge1\); the legacy
`spherical_polar_kappa` inner-core offset is ignored in this mode.

Each center cell \((0,j)\) is evaluated as the three-corner polygon
\(\{node(0,j), node(1,j), node(1,j+1)\}\). The structured slot
`node(0,j+1)` remains allocated but is inactive for that cell. All other cells
remain four-corner cells. The derived per-cell vertex count is not persisted to
HDF5 or restart files.

The `tri_fan` RZ volume predicate uses the signed exact polygon formula
\[
V_\mathrm{raw}={\pi\over3}\sum_k (r_k+r_{k+1})
  (r_k z_{k+1}-r_{k+1}z_k).
\]
The spherical-polar slot order is clockwise in \((R,Z)\), so the canonical
admissible orientation is \(V=-V_\mathrm{raw}>0\). Collapse and inversion are
detected by signed same-orientation and absolute floors; `fabs(V)` is not used
as an acceptance value because it would mask collapse-through-zero. Candidate
mesh admissibility and ALE post-rezone quality predicates use the same
canonical sign convention for both three-corner center cells and four-corner
tri_fan ring cells. The inactive fourth slot of center cells has zero cached
`cell_Svec` entries; this is an out-of-bounds/NaN safety net, not a hydro
mass/force model.

The default-off S1 multiblock γ MVP topology is specified with the 2D RZ
geometry in §3.2.0a.

**H3-B Phase 6 ctest:**

`tests/verification/test_h3_sedov_polar.cu` is the Phase 6 production-gate
integration test for the spherical-polar H3-B Sedov path. It runs
`examples/verification/2d_rz_h3b_sedov_sph_polar.py` at \(N_s\times
N_\theta=\{128\times256,256\times512,512\times1024\}\), with radiation, laser,
and conduction disabled. The deck uses the 2T ideal-gas initialization with
fixed \(\bar Z=1\), and places the blast outside the finite inner core over the
first four radial shells, \(R_\mathrm{blast}=s_\min+4\Delta s\). The test checks
completion to \(t=6.0\times10^{-9}\) s, Sedov exponent \(\alpha=0.4\pm0.06\),
\(\beta\) median in `[0.92, 1.20]`,
\(\mathrm{CV}(\beta)\le0.06\), refinement-dependent similarity-collapse RMSE,
optional HDF5 total-energy conservation \(\le10^{-3}\), optional
Hydro2D run-log energy-correction relative maximum \(\le10^{-2}\), and final
angular shock anisotropy \(\le5\%\). The rectangular H3 Sedov radius-similarity
ctest applies the same run-log Hydro2D energy-correction gate when those entries
are present.

**再同定の必要条件**：
- **2D ALE rezone 後（§3.3）**：rezoneはメッシュを非Lagrangian的に移動させるため、**IMC粒子**の cellId を再同定する（U7: `cell_search_after_rezone`、CUDA_KERNELS §9 Phase 5）。DDMC粒子は位置座標が NaN sentinel（§7.7.3）であり空間探索が不可能なためスキップする。DDMC粒子の cellId は R7 composite key sort（§6.5）がセルモードテーブルから決定するため、U7 での再同定は不要。
- **Lagrangian ステップ後（§3.2）**：1DではIMC transport stepの冒頭、source emission・DDMC/RW partition・RadLite remapより前に、current mesh node \(x_r\) に対してcensus粒子の cellId をGPU上で再同定する。圧縮セルでは粒子位置が保存済み cellId の境界外へ出る場合があり、古い cellId のまま吸収・放出係数を参照すると非物理的なエネルギー授受を生む。2D RZでのLagrangian後再同定は未実装であり、既存の輸送中面交差追跡に従う。

### 9.1 Point-in-cell判定（2D RZ構造四辺形メッシュ）

セル \((i,j)\) の4頂点を反時計回りに \(\mathbf{V}_0, \mathbf{V}_1, \mathbf{V}_2, \mathbf{V}_3\) とする（各頂点は \((R,Z)\) 座標）。
点 \(\mathbf{P}=(R_p, Z_p)\) がセル内にあるかを **外積符号判定** で決定する：

辺 \(k \to k{+}1\)（\(\mathrm{mod}\, 4\)）に対し：
\[
C_k = (\mathbf{V}_{k+1} - \mathbf{V}_k) \times (\mathbf{P} - \mathbf{V}_k)
= (R_{k+1}-R_k)(Z_p-Z_k) - (Z_{k+1}-Z_k)(R_p-R_k) \quad [\text{cm}^2]
\]

判定：
- 全 \(C_k \ge 0\)（反時計回り規約）⇒ 内部（辺上を含む）
- 1つでも \(C_k < 0\) ⇒ 外部

> **凸性の保証**：Lagrangianメッシュの四辺形セルは時間発展中に凸性を失う場合がある。
> 凸性が崩れた場合（外積符号が混在する場合）は、セルを2三角形に分割して各三角形で判定する。
> 三角形分割は対角線 \(\mathbf{V}_0\)–\(\mathbf{V}_2\) を使用する。

計算量：1セルあたり O(1)（外積4回）。

**点包含判定の詳細**（2D 四角形）：4辺外積テスト。頂点は反時計回り前提。
\[
\text{cross}_k = (\mathbf{V}_{k+1} - \mathbf{V}_k) \times (\mathbf{P} - \mathbf{V}_k)
\]
全 \(k\) で \(\text{cross}_k \ge -\varepsilon_{geom}\) なら内部と判定する。
退化セル（\(J \approx 0\)）では \(\varepsilon_{geom}\) を \(\max(\varepsilon_{geom},\, 10^{-6}\sqrt{V_c})\) に緩和し、数値ゼロ近傍での誤判定を回避する。

**\(\varepsilon_{geom}\) のデフォルト値**：
\[
\varepsilon_{geom} = 10^{-12} \times \min(\Delta r_{min},\, \Delta z_{min})
\]
ここで \(\Delta r_{min}, \Delta z_{min}\) は各方向の最小セル幅。
メッシュ初期化時に一度計算し、ALE rezone 後に再計算する。

**非凸セルの判定**：
v1.0 の構造格子では、ALE rezone 後もセルは凸四辺形を維持する
（Winslow smoothing が凸性を保存、§3.3.3 の Jacobian 正値性検査で保証）。
Jacobian が負になった場合は FATAL エラーで停止する（§11 Safety）。
したがって、v1.0 では非凸セルの point-in-cell 処理は不要。

### 9.2 Point-in-cell判定（1D球対称メッシュ）

セル \(i\) は同心球殻 \(r_{i-1} \le r < r_i\) で定義される（\(r_0, r_1, ..., r_N\) は単調増加の節点半径）。
粒子の動径座標 \(r_p\) に対し：

- **局所判定**：\(r_{i-1} \le r_p < r_i\) を直接比較（O(1)）
- **二分探索（フォールバック）**：ソート済み節点配列 \(\{r_i\}\) に対する二分探索（O(log N)）

### 9.3 局所探索アルゴリズム（stencil walk）

CFL制約により、1タイムステップでのメッシュ変位は1セル幅未満である。
したがって、粒子の新セルは元セルの近傍に必ず存在する。

**アルゴリズム**（1粒子あたり）：

```
入力：粒子位置 P, 前ステップの cellId = (i₀, j₀)
出力：新しい cellId

1. point_in_cell(P, i₀, j₀) を判定
   → True ⇒ return (i₀, j₀)  // 変化なし

2. 直接隣接4セルを判定（i₀±1, j₀), (i₀, j₀±1）
   → True のセルがあれば return

3. 対角隣接4セルを判定（i₀±1, j₀±1）
   → True のセルがあれば return

4. リング拡張探索（§9.4）へ移行
```

1D_SPHの場合はステップ2を `(i₀-1)` と `(i₀+1)` のみで判定する。

### 9.4 リング拡張探索

stencil walk（§9.3）で失敗した場合、探索半径を拡大する：

```
for ring = 2 to max_rings:
  // ring-k の近傍 = チェビシェフ距離（L∞ノルム） == k のセル集合
  for (di, dj) where max(|di|,|dj|) == ring:
    if point_in_cell(P, i₀+di, j₀+dj):
      return (i₀+di, j₀+dj)
```

- `max_rings`（既定 3）：リング拡張の最大ステップ数
- ring-k の候補セル数：\(8k\)（ring-2: 16、ring-3: 24）
- 境界セルは配列範囲外のためスキップ

### 9.5 大域フォールバック（background grid）

リング拡張でも見つからない場合（ALE rezoneで大きな変位が生じた場合など）：

**背景格子ハッシュ**（空間ハッシュ構造）：

**構築アルゴリズム**：
1. 計算領域の AABB（axis-aligned bounding box）を決定：
   \((R_{min}, Z_{min})\) ～ \((R_{max}, Z_{max})\)（全節点座標の最小/最大）
2. 一様直交格子（\(M_R \times M_Z\) セル）を構築
   - セル幅：\(\Delta R_{bg} = (R_{max}-R_{min})/M_R\)、\(\Delta Z_{bg} = (Z_{max}-Z_{min})/M_Z\)
3. 各メッシュセル \(c\) のAABBを計算し、重なる背景セルのインデックスを決定：
   - メッシュセルのAABB：4頂点の座標から \((R_{min}^c, R_{max}^c, Z_{min}^c, Z_{max}^c)\) を算出
   - 重なる背景セル範囲：
     \(p_0 = \lfloor(R_{min}^c-R_{min})/\Delta R_{bg}\rfloor\) ～ \(p_1 = \lfloor(R_{max}^c-R_{min})/\Delta R_{bg}\rfloor\)
     \(q_0 = \lfloor(Z_{min}^c-Z_{min})/\Delta Z_{bg}\rfloor\) ～ \(q_1 = \lfloor(Z_{max}^c-Z_{min})/\Delta Z_{bg}\rfloor\)
   - 背景セル \((p,q)\) の候補リストにメッシュセル \(c\) を追加
4. **フラット配列への格納**（GPU向け）：
   - `offsets[M_R × M_Z + 1]`：各背景セルの候補リスト開始位置
   - `indices[total_entries]`：全候補のメッシュセルインデックス
   - 背景セル \((p,q)\) の候補は `indices[offsets[p*M_Z+q] .. offsets[p*M_Z+q+1])` で参照

**検索手順**（1粒子あたり O(1)期待値）：
1. 粒子位置 \(\mathbf{P}=(R_p, Z_p)\) から背景セルインデックスを計算：
   \(p = \lfloor(R_p-R_{min})/\Delta R_{bg}\rfloor\)、\(q = \lfloor(Z_p-Z_{min})/\Delta Z_{bg}\rfloor\)
2. 背景セル \((p,q)\) の候補リスト内のメッシュセルのみを point_in_cell 判定

背景格子パラメータ：
- **背景セル幅の下限**：\(\text{cell\_size} = \max(\Delta r_{max}, \Delta z_{max}) \times 1.1\)。各メッシュセルが少なくとも1つの背景セルに収まるよう、最大メッシュセル幅にマージン1.1を乗じる
- **グリッド解像度**：\(\Delta R_{bg} \ge \text{cell\_size}\) を満たす最小の \(M_R, M_Z\) を使用する：
  \[
  M_R = \min\!\left(\lceil\sqrt{N_{cell}}\rceil,\;\max\!\left(1,\;\left\lfloor\frac{R_{max}-R_{min}}{\text{cell\_size}}\right\rfloor\right)\right),\quad
  M_Z = \min\!\left(\lceil\sqrt{N_{cell}}\rceil,\;\max\!\left(1,\;\left\lfloor\frac{Z_{max}-Z_{min}}{\text{cell\_size}}\right\rfloor\right)\right)
  \]
  これにより \(\Delta R_{bg} = (R_{max}-R_{min})/M_R \ge \text{cell\_size}\) が保証される
- 候補リストの平均長さ：\(N_{cell}/(M_R \times M_Z) \approx 1\)
- ALE rezone 後は背景格子の再構築時に `cell_size` と \(M_R, M_Z\) を再計算する

**注**：上記の CSR 形式（`offsets/indices`）は直接グリッドインデックス \((p,q)\) でアクセスするため、ハッシュ衝突は発生しない。各背景セルの候補リスト長は平均 \(\approx 1\)（候補重複を含めても \(\le 4\) 程度）であり、v1.0 では CSR 形式を唯一の実装として使用する。

- **構築タイミング**：
  - 初回構築：初期化時（step=0の前）
  - 再構築：ALE rezone実行時のみ（Lagrangianステップではメッシュ変形が小さく不要）
  - 再構築コスト：\(O(N_{cell})\)（全セルのAABBを再計算、背景セルリストを再生成）

**フォールバック探索**：hash grid + neighbor walk（stencil walk §9.3 → リング拡張 §9.4 → background grid §9.5）で `cellId` 未確定（`walk_max=20` 回以内に見つからない）の場合、brute-force 点包含判定（全セル走査、\(O(N_{cells})\)）を実行する。発生率は通常 < 0.1%。brute-force でも未発見の場合は**領域外逸脱**と判定し、当該粒子のエネルギーを数値損失に計上（\(E_{numerical\_loss}\) に加算。物理的脱出ではなく数値的損失のため \(E_{rad,esc}\) ではない）して粒子を消滅させる。

**セル探索失敗時の物理的根拠**：
粒子が全探索（stencil walk → ring expansion → hash grid → brute-force）で見つからない場合、
粒子が計算領域外に逸脱したと判定する。これは以下の状況で発生しうる：
- ALE rezone による大きなメッシュ変形で粒子がメッシュ外へ
- Lagrangian ステップでのメッシュ移動が粒子位置を超過
- 浮動小数点丸め誤差による微小な領域外判定

**エネルギー保存への影響**：消滅粒子のエネルギーは \(E_{numerical\_loss}\)（§10.2 エネルギー収支）に計上し、
グローバルエネルギー保存を維持する。セル探索失敗は数値的損失であり、物理的境界脱出 \(E_{rad,esc}\) とは区別する。消滅粒子数はステップごとに `n_cell_search_lost` として記録し、
diagnostics に出力する。

**動作モード**（`cell_search.fatal`（既定 True、SPECIFICATION §6.4.7））：
- `True`（デバッグモード）：brute-force 含む全探索失敗時に **ERROR**（checkpoint 書き出し後に終了、ARCHITECTURE §10.2 準拠）
- `False`（本番モード）：粒子を消滅させ、エネルギーを \(E_{numerical\_loss}\) に計上、消滅粒子数を WARNING 出力

### 9.6 メッシュ移動時のバッチ処理

全粒子のセル再同定はGPU上でバッチ処理する：

1. **1D Lagrangian mesh 後**：IMC transport step 冒頭に
   `src/radiation/particle_reid.cu` の `reidentify_finite_position_particles_1d_cuda`
   を呼ぶ。各 CUDA thread が1粒子を担当し、`alive==ALIVE` かつ `mode!=DDMC`
   かつ finite `pos_r` の粒子だけを処理する。DDMC 粒子と NaN sentinel 粒子は
   cellId を変更しない。
2. **1D cell search**：まず保存済み `old_cellId` が current node
   \([x_{j},x_{j+1}]\) を含むかを判定する。外れている場合は
   `old_cellId±1, ±2` の近傍 walk を試し、それでも失敗した粒子だけ
   monotone node 配列に対して \(O(\log N)\) の二分探索を行う。領域外の有限位置は
   最近傍端セルへ clamp する。
3. **partition invalidation**：mesh 再同定で finite-position 粒子が存在する場合、
   `State::particle_sort_cache_invalidated=true` とする。次の radiation operator は
   DDMC/RW partition が必要な場合は通常の composite sort で再分離し、final census
   compaction/sort 後に invalidation flag を clear する。
4. **2D ALE 後**：2D_RZ の ALE rezone/remap 後は §9.1 の point-in-cell 判定と
   §9.4 の探索で finite-position 粒子を再同定する。

### 9.7 呼び出しタイミング

| タイミング | 探索種別 | 理由 |
|-----------|---------|------|
| Lagrangianステップ後 | 1D: IMC transport step冒頭で二分探索再同定 / 2D: 未実装 | 圧縮セルで保存済み cellId が current mesh 境界外になる場合がある |
| ALE rezone後 | 1D: old cell check → ±2 walk → binary search / 2D: stencil walk → リング拡張（§9.4） | ノード位置が大きく変わりうる |
| 粒子生成時 | 不要（生成セルが既知） | セル内一様サンプリングで生成 |
| DDMC→IMCリーク時 | 不要（リーク先セルが既知） | 隣接セルIDは事前計算済み |
| 並列ドメイン間移動時 | 受信先で局所探索（§12.3） | 移動先rankのメッシュで再同定 |

---

## 10. 推定量とエネルギー収支（rad_E, budgets）
### 10.1 IMCのtrack‑length推定量（rad_E）
IMC粒子のセル内セグメント長 \(\Delta s\) に対し、滞在時間 \(\Delta t=\Delta s/c\)。
時間平均エネルギー密度推定：
\[
\hat E_{i,g} = \frac{1}{V_i \Delta t}\sum_{segments} E_{mid}\,\Delta t
= \frac{1}{V_i c \Delta t}\sum_{segments} E_{mid}\,\Delta s
\]

**体積 \(V_i\) の時間レベル**：\(V_i = V_i^{*}\)（最初のHydro半ステップ H(\(\Delta t/2\)) 後の体積）を使用する。
放射輸送（R演算子）は Strang分割（§2.1）において L(Δt) → H(Δt/2) → C(Δt) → **R(Δt)** → H(Δt/2) の順で実行され、
ALE rezone/remap は2回目の H(\(\Delta t/2\)) 後にのみ実行されるため（§2.1, §3.3）、
R演算子開始時点では ALE 未実行であり、メッシュ体積は最初の半ステップ後の \(V_i^{*}\) である。
タリー集約（`rad_dep`, `rad_E_tally`）はこの体積を用いて正規化する。

**タリー集約の時間区間**：`rad_dep[i,g]`、`rad_E_tally[i,g]`、および `laser_dep[i]` は
物理時間 \([t^n, t^{n+1}]\)（＝ \(\Delta t\)）全体にわたる寄与を累積する。
ゼロ初期化タイミングは使用フェーズ直前に設定する（CUDA_KERNELS §9 参照）：
- `rad_dep`：ステップ冒頭（Laser注入での `rad_dep=0` 前提を保証）**および** Radiation 冒頭（R8/R9 atomicAdd の累積先として再初期化）
- `rad_E_tally`：Radiation 冒頭のみ（R8/R9 が使用する Radiation ローカル配列）
- `laser_dep`：Radiation 冒頭のみ（Laser で L5/reconstruct_laser_dep が書き込み、Laser後 U1 で消費済み。Radiation 後 U1 での二重計上防止のためゼロ化）

`rad_dep`/`rad_E_tally` は radiation operator 全体にわたり累積される。
`laser_dep` はレーザーオペレータ（§5.8）完了後に確定する。

- \(E_{mid}\) はセグメント中の代表エネルギー。連続吸収（§6.3.3）では指数減衰を考慮し、
  セグメント入口エネルギー \(E_{in}\) と出口エネルギー \(E_{out}=E_{in}e^{-\sigma_{a,eff}\,\Delta s}\) の
  **track-length平均**（解析積分）を使用する：
\[
E_{mid} = \frac{1}{\Delta s}\int_0^{\Delta s} E_{in}\,e^{-\sigma_{a,eff}\,s}\,ds
= \frac{E_{in}(1 - e^{-\sigma_{a,eff}\,\Delta s})}{\sigma_{a,eff}\,\Delta s}
= \frac{\Delta E_{dep}}{\sigma_{a,eff}\,\Delta s}
\]
ここで \(\Delta E_{dep} = E_{in} - E_{out}\)。
**\(\sigma_{a,eff}\,\Delta s\) のスイッチ閾値**（0/0 回避）：
\(\tau = \sigma_{a,eff} \times \Delta s\) として、
- \(\tau \geq 10^{-6}\)：標準公式 \(\Delta E_{dep} = E_{in}(1 - e^{-\tau})\)、\(E_{mid} = \Delta E_{dep} / \tau\)
- \(\tau < 10^{-6}\)：Taylor展開 \(\Delta E_{dep} \approx E_{in}\,\tau(1 - \tau/2)\)、\(E_{mid} \approx E_{in}(1 - \tau/2)\)
- \(\sigma_{a,eff} = 0\) かつ \(\Delta s = 0\)（静止粒子）：\(\Delta E_{dep} = 0\)、\(E_{mid} = E_{in}\)

#### 10.1.1 IMC運動量沈着推定量（診断）

IMC粒子の各セグメントにおける運動量沈着の寄与（track-length estimator）：
\[
\Delta \mathbf{p}_{i,seg} = \frac{\sigma_{t,i}}{c}\, E_{mid}\,\Delta s\, \hat{\Omega}
\]
ここで \(\sigma_{t,i} = \sigma_{a,eff,g} + \sigma_{s,eff,g}\) はIMCの全相互作用不透明度（§6.3.1）、
\(\hat{\Omega}\) は粒子の方向ベクトル。セル i の運動量沈着（1D_SPH: 径方向成分、2D_RZ: R,Z成分）：
\[
\text{rad\_mom\_dep}_{i} = \frac{1}{V_i} \sum_{segments \in i} \Delta \mathbf{p}_{i,seg}
\quad [\text{dyne}\cdot\text{s/cm}^3]
\]
DDMCの運動量沈着（§7.8.2）と合算して出力する。

> **v1.0での扱い**：運動量沈着は **診断出力のみ**（SPECIFICATION §6.4.5
> `momentum_deposition=True`）。hydro 運動量方程式へのフィードバックは行わない
> （O(v/c)項を無視する設計方針と整合）。
> 統計誤差が大きいため（Densmore 2007）、定量的な利用には分散低減が必要。

### 10.2 エネルギー保存のタリー
各ステップで以下を必ず積算し、差分を診断出力する。

**エネルギー収支の恒等式**（保存チェック）：
\[
\Delta E_{total} = \Delta E_{int,e} + \Delta E_{int,i} + \Delta E_{kin} + \Delta E_{rad}
\]
\[
= E_{laser,in} - E_{laser,esc} + E_{Marshak,in} - E_{rad,esc} - E_{pdV}^{boundary} - E_{numerical\_loss} + E_{floor} + E_{safety} + E_{redistribution\_unresolved} + E_{solver}
\]

各項の定義（全項 [erg] 単位）：

| 記号 | 名称 | タリー方法 |
|------|------|-----------|
| \(\Delta E_{int,e}\) | 電子内部エネルギー変化 | \(\sum_i \rho_i\,\Delta e_{e,i}\,V_i\) |
| \(\Delta E_{int,i}\) | イオン内部エネルギー変化 | \(\sum_i \rho_i\,\Delta e_{i,i}\,V_i\)（人工粘性散逸を含む） |
| \(\Delta E_{kin}\) | 運動エネルギー変化 | 1D は \(\sum_i \frac{1}{2}m_i u_i^2\)（\(u_i=\frac{1}{2}(v_i+v_{i+1})\)）の差分。2D RZ は §3.3 Phase 12 と同じ corner-mass nodal kinetic energy \(\sum_c K_c^{diag}\) の差分 |
| \(\Delta E_{rad}\) | 放射場エネルギー変化 | \(E_{census}^{n+1} - E_{census}^n\)（通常は census粒子エネルギーの差分、\(E_{census}=\sum_p E_p\)。difference path では \(E_{census}=\sum_{i,g}U^{ref}_{i,g}+\sum_p s_pE_p\)） |
| \(E_{laser,in}\) | レーザー入射エネルギー | \(\sum_b P_b(t)\,\Delta t\) |
| \(E_{laser,esc}\) | レーザー未吸収流出 | 臨界終了 + LaserMesh外終了のレイエネルギー合計 |
| \(E_{Marshak,in}\) | Marshak境界入射エネルギー | \(\sum_f \frac{a_{eV}\,c}{4}\,T_{r,f}^4\,A_f\,\Delta t\)（§8.2 per-face合計） |
| \(E_{rad,esc}\) | 放射境界流出 | vacuum/Marshak境界で脱出した粒子のエネルギー合計（`E_escape[g]` の全群合算）|
| \(E_{pdV}^{boundary}\) | 境界PdV仕事 | 外側境界面での \(P\,dV\) |
| \(E_{floor}\) | フロア補正注入（実装名: `E_floor_injected`） | \(\sum_c \Delta E_{floor,c}\)（§1.1.7 温度・密度フロア + §11.7 安全検査で注入） |
| \(E_{safety}\) | 伝導安全補正 | §4.2.2 の負温度clampで注入されたエネルギー |
| \(E_{redistribution\_unresolved}\) | ALE positivity redistribution 未解決分 | `Numerics.ale.ke_closure_redistribute_floor=true` で \(C_{tot}<D_{tot}\) のときの \(D_{tot}-C_{tot}\)。通常は0 |
| \(E_{solver}\) | Hypre残差エネルギー | \(\sum_c C_{v,c}(T_{e,c}^{n+1}-T_{e,c}^n)V_c - \Delta t\sum_c(\nabla\cdot q)_c V_c\)（§4.2.3 参照。Hypre無効時は0） |

> **人工補正項の分離**：\(E_{floor}\)、\(E_{safety}\)、\(E_{redistribution\_unresolved}\)、\(E_{solver}\) は物理的なエネルギー源ではなく数値安全策・ソルバ残差による注入量である。
> 保存誤差の診断時にはこれらを明示的に分離し、\(\varepsilon_{budget}\) の分子には含めない（下記参照）。
> これにより「物理的保存」と「人工補正を含めた総収支」の両方を独立に監視できる。

**\(E_{pdV}^{boundary}\) の計算**：
\[
E_{pdV}^{boundary} = \sum_{f \in \partial\Omega} P_f \,(A_f\, v_{n,f})\,\Delta t
\]
正値 = 膨張（系から外部へのエネルギー流出）。1D_SPH: \(f\) = 最外ノード。2D_RZ: §3.2.14 の4境界面（r=0軸、外周、上下端）。

**エネルギー保存誤差の定義**：
\[
\varepsilon_{budget} = \frac{|(E_{total}^{n+1} - E_{total}^n - E_{artificial}) - (E_{source} - E_{sink})|}
                          {E_{denom}}
\]
ここで \(E_{source} = E_{laser,in} + E_{Marshak,in} + E_{volume,in} + \max(E_{solver},0)\)、\(E_{sink} = E_{laser,esc} + E_{rad,esc} + E_{numerical\_loss} + E_{pdV}^{boundary} + \max(-E_{solver},0)\)。
\[
E_{artificial}=E_{floor}+\max(E_{safety}-E_{floor},0)+E_{redistribution\_unresolved}.
\]
\(E_{numerical\_loss}\) は粒子移送失敗等のアルゴリズム限界による数値的喪失エネルギー（§12.3.1参照）。物理的境界流出 \(E_{rad,esc}\) とは明確に区別する。
分母のフロア：\(E_{denom} = \max(E_{total}^n,\, E_{source},\, 10^{-20}\;\text{erg})\)。コールドスタート（全エネルギー≈0）での0除算と偽 FATAL を防止する。

> **注意**：\(\varepsilon_{budget}\) の分子は **物理的保存誤差**であり、\(E_{floor}\)、\(E_{safety}\)、\(E_{redistribution\_unresolved}\) は人工補正として取り除く。\(E_{solver}\) は符号に応じて \(E_{source}\) または \(E_{sink}\) に入れる。
> これらは総収支式に陽に入るため、正しく計上すれば恒等式を満たす。
> 別途 \(E_{floor}/E_{denom}\)、\(E_{safety}/E_{denom}\)、\(E_{redistribution\_unresolved}/E_{denom}\)、\(E_{solver}/E_{denom}\) を独立に監視し、
> 数値安全策・ソルバ残差の寄与が支配的になっていないか（例：\(> 10^{-3}\)）を検出する。

**出力**：各ステップで \(\varepsilon_{budget}\)、\(E_{floor}\)、\(E_{safety}\)、\(E_{redistribution\_unresolved}\)、\(E_{solver}\)、\(E_{numerical\_loss}\) を history ファイルに記録する。

**放射サブシステムのエネルギー恒等式**（per-operator保存チェック）：
放射輸送ステップ（Phase 4）単独でのエネルギー保存を独立に検証する：
\[
E_{emit} + E_{census}^{n} = E_{abs} + E_{esc} + E_{census}^{n+1} + E_{numerical\_loss}
\]
ここで：
- \(E_{emit}\)：ステップ中に放出された粒子エネルギー合計（census放出 + Marshak境界放出）
- \(E_{census}^{n}\)：Phase 4 開始時の census 粒子エネルギー合計
- \(E_{abs} = \sum_{i,g} \text{rad\_dep}[i,g]\)：物質との放射交換エネルギー合計
  （IMC/DDMC の gross absorption tally）
- \(E_{esc}\)：vacuum 境界から流出したエネルギー合計
- \(E_{census}^{n+1}\)：Phase 4 終了時の census 粒子エネルギー合計
- \(E_{numerical\_loss}\)：移送失敗（R8 MAX\_EVENTS 超過）・退化セル未注入（U1 `ρV < 10^{-30}`）・粒子喪失（P6）等による数値的喪失。**注**: Russian roulette（R8 step 7 / R12）で消滅した粒子のエネルギーは `rad_dep` に沈着され \(E_{abs}\) に含まれるため、\(E_{numerical\_loss}\) には計上しない

放射サブシステム保存誤差：\(\varepsilon_{rad} = |LHS - RHS| / \max(E_{emit} + E_{census}^{n},\, 10^{-20})\)。
各ステップで \(\varepsilon_{rad}\) を history に記録し、VERIFICATION §2.3 の閾値（1ステップ \(10^{-6}\)）を適用する。

**rad_dep と deposited_power の変換規約**：
内部タリー量 `rad_dep[i,g]` は **当該ステップ \(\Delta t\) 中に群 g でセル i が受け取った放射交換エネルギー [erg]** である。
`rad_dep` は IMC/DDMC の粒子 absorption/exchange tally を保持し、internal PGRW branch も
IMC と同じ `rad_dep` に寄与する。legacy all-positive path では gross absorption、
signed residual path では net signed exchange である。
signed residual 粒子では、粒子の非負エネルギー大きさ \(E_p\) に immutable な
\(s_p \in \{-1,+1\}\) を掛けた寄与を `rad_dep`、`rad_E_tally`、`E_escape`、
および数値的喪失 tally に加算する。すべての粒子が \(s_p=+1\) の場合は従来の
unsigned tally と一致する。
HDF5スナップショット出力時に以下の変換を行う：
\[
\text{deposited\_power}_{i,g} = \frac{\text{rad\_dep}_{i,g}}{V_i^{*} \times \Delta t} \quad [\text{erg/cm}^3/\text{s}]
\]
ここで \(V_i^{*}\) は §10.1 と同一の体積（最初の H(\(\Delta t/2\)) 後、ALE rezone 前）である。
エネルギー保存検証（上記の \(\varepsilon_{budget}\)）には `rad_dep` [erg] をそのまま使用する。
`deposited_power` は可視化・解析用の派生量であり、保存検証には使用しない。

### 10.3 タリー集約のGPU階層並列化

§10.1, §10.2 のタリー（`rad_dep`, `rad_E_tally`, `E_escape`）は、粒子カーネル内で
`atomicAdd(double*)` により全粒子からの寄与を累積する。GPU上では数千〜数百万のスレッドが
同時に書き込むため、atomic競合がボトルネックとなりうる。

本節では3段階の階層的集約アルゴリズムを定義する。
CUDAカーネル実装の詳細は CUDA_KERNELS.md §6.4 を参照。

#### 10.3.1 設計原理

タリー集約の数学的要件：
- **加法性**：各粒子の寄与 \(\Delta E_p\) は独立であり、セル×群ごとの合計 \(\sum_p \Delta E_p\) を求める
- **交換法則**：加算順序は結果の期待値に影響しない
- **IEEE 754 注意**：浮動小数点加算は結合法則を満たさないため、集約順序により最下位ビットが変動する。
  モンテカルロ法の統計的性質上、bitwise再現は要求しない。`tally_mode` の選択は性能のみに影響し、
  統計的結果（平均・分散）には影響しない。

**ΔE_tl の定義**：track-length推定量への寄与は以下で定義する。
- \(\sigma_{a,eff,g} > 0\) の場合：\(\Delta E_{tl} \equiv E_{mid} \times \Delta s = \Delta E_{dep} / \sigma_{a,eff,g}\)
- \(\sigma_{a,eff,g} = 0\) の場合：\(\Delta E_{tl} = E_{in} \times \Delta s\)（粒子エネルギー不変）

ここで \(E_{mid}\) は §10.1 で定義されたtrack-length平均エネルギーである。

#### 10.3.2 Stage 3: Global atomicAdd

最も単純な方式。各スレッドが直接 global memory の `rad_dep[cell*G+g]` に `atomicAdd` する。

```
atomicAdd(&rad_dep[cell_id * G + group_id], ΔE_dep);   // 吸収沈着
atomicAdd(&rad_E_tally[cell_id * G + group_id], ΔE_tl); // track-length推定量
```

- **性能特性**：CC 6.0+（Pascal）で `atomicAdd(double*)` がハードウェアサポート。
  L2キャッシュ上のatomic操作は ~100 ns/op。同一アドレスへの連続atomicはシリアライズされる。
- **競合度**：セルあたりの粒子数を \(N_{p/c}\) とすると、同一アドレスへの同時書き込み頻度は
  \(\sim N_{p/c} / (\text{warp数} \times \text{SM数})\)。セルソート（§6.5）なしでは空間的に分散するが、
  ソートありでは同一warp内で同一セルに集中し、Stage 1 の前提条件を作る。

#### 10.3.3 Stage 1: Warp-level集約（v1.0既定）

**前提条件**：粒子がセルソート済み（§6.5）。同一warp内の粒子の大半が同一 `cell_id` を持つ。

**アルゴリズム**：CUDA `__match_any_sync`（CC 7.0+, Volta以降）を用いたwarp内ピアグループ集約。

```
mask    ← __activemask()
key     ← cell_id × G + group_id
peers   ← __match_any_sync(mask, key)     // 同一keyのレーンのビットマスク
leader  ← __ffs(peers) − 1                // グループ内最小レーン番号

// ピアグループ内 segmented reduction（全ピアレーンが __shfl_down_sync に参加 — 必須）
// 注意：リーダーのみが __shfl_sync を呼ぶパターンは CUDA 仕様上の未定義動作（§B.15）。
// 全ピアレーンが同一 mask で __shfl_down_sync に参加する以下のパターンを使用する。
// （CUDA_KERNELS §6.4 と整合）
sum_dep ← ΔE_dep
sum_tl  ← ΔE_tl
for offset = 16, 8, 4, 2, 1:              // warp-width/2 からハーフィング
    tmp_dep ← __shfl_down_sync(peers, sum_dep, offset)
    tmp_tl  ← __shfl_down_sync(peers, sum_tl,  offset)
    src_lane ← lane + offset
    if (src_lane < 32 AND (peers >> src_lane) & 1):  // 範囲内かつピアメンバーのみ加算
        sum_dep += tmp_dep
        sum_tl  += tmp_tl
// リーダーのみが集約結果を書き出す
if (lane == leader):
    // Stage 2 有効時は shared mem へ、無効時は global へ
    if (Stage 2 有効):
        block_tally_accumulate(key, sum_dep, sum_tl)   // → shared mem（§10.3.4）
    else:
        atomicAdd(&rad_dep[key], sum_dep)              // → global
        atomicAdd(&rad_E_tally[key], sum_tl)
    // 運動量沈着も同一ピアグループで集約（CUDA_KERNELS §6.4 Stage 1 準拠）
    // rad_mom_dep は [n_cells × dim] のため key ではなく cell_id で書き出す
    if (sum_mom_r != 0) atomicAdd(&rad_mom_dep[cell_id*dim + 0], sum_mom_r)
    if (sum_mom_z != 0) atomicAdd(&rad_mom_dep[cell_id*dim + 1], sum_mom_z)
```

- **削減率**：セルソート済みなら1ワープ内のピアグループサイズは典型的に 28–32。
  global atomicAdd の回数を最大 **32分の1** に削減する。
- **レジスタ増加**：~8レジスタ（peers, leader, offset, tmp×4: dep, tl, mom_r, mom_z）。imc_transport_persistent の ~60 レジスタに対し微小。
- **不活性レーン処理**：`__match_any_sync` の `mask` 引数に `__activemask()` を渡すことで、ブロック末尾の不完全ワープ等で不活性なレーンを自動除外する。`__shfl_sync` も同一 `mask` を使用する。
- **適用カーネル**：`imc_transport_persistent`（R8）、`ddmc_event_loop`（R9）の両方に適用。

> **v1.0既定**：セルソート（§6.5）と合わせて v1.0 baseline として採用する。
> セルソートとStage 1は不可分であり（§6.5「不可分性」参照）、`particle_sort_by_cell=True`
> のとき自動的にStage 1が有効化される。Stage 3 のみの `"global"` モードは CC 7.0 未満
> のフォールバックおよびデバッグ用として残す。

#### 10.3.4 Stage 2: Block-level共有メモリ集約（将来拡張）

Stage 1 の出力（ワープリーダーの集約値）をさらにブロック内で共有メモリ上に蓄積し、
ブロック末尾で一括して global へ書き出す。

**アルゴリズム**：ブロック内ビンヒストグラム方式（open addressing + atomicCAS、レースコンディション対策版）。

```
// スロットの確保とキー書き込みをatomicCASで一体化
__shared__ int smem_keys[N_BINS];      // -1 で初期化
__shared__ double smem_dep[N_BINS];    // 0 で初期化
__shared__ double smem_tl[N_BINS];     // 0 で初期化
__shared__ int smem_n_bins;                // 0 で初期化

// 初期化（ブロック内全スレッドで分担）
for (int s = threadIdx.x; s < N_BINS; s += blockDim.x) {
    smem_keys[s] = -1;
    smem_dep[s] = 0.0;
    smem_tl[s] = 0.0;
}
if (threadIdx.x == 0) smem_n_bins = 0;
__syncthreads();

// キー検索＋挿入（open addressing, linear probing）
int slot = -1;
int hash = key % N_BINS;
for (int probe = 0; probe < N_BINS; probe++) {
    int idx = (hash + probe) % N_BINS;
    int old = atomicCAS(&smem_keys[idx], -1, key);
    if (old == -1 || old == key) {
        slot = idx;
        break;
    }
}
if (slot == -1) {
    // オーバーフロー：グローバルメモリにフォールバック
    atomicAdd(&global_dep[key], sum_dep);
    atomicAdd(&global_tl[key], sum_tl);
} else {
    atomicAdd(&smem_dep[slot], sum_dep);
    atomicAdd(&smem_tl[slot], sum_tl);
}
__syncthreads();

// Stage 3: 共有メモリ→グローバルメモリ
for (int s = threadIdx.x; s < N_BINS; s += blockDim.x) {
    if (smem_keys[s] != -1) {
        atomicAdd(&global_dep[smem_keys[s]], smem_dep[s]);
        atomicAdd(&global_tl[smem_keys[s]], smem_tl[s]);
    }
}
```

**旧実装からの変更点**：
- `atomicAdd(&smem_n_bins, 1)` による逐次スロット確保を廃止
- `atomicCAS` によるopen addressing + linear probing方式に変更
- キー書き込みとスロット確保が原子的に実行されるため、重複ビンが発生しない
- オーバーフロー時はグローバルメモリに直接フォールバック

- **`N_BINS`**：`N_BINS_MAX = 128`（`block_size = 128` と整合）。セルソート済みならブロック内ユニークセル数は典型的に 1–4 で、
  ビン数は \(\le 4 \times G = 64\)。overflow はほぼ発生しない。
- **共有メモリ使用量**：\(128 \times (8 + 8 + 4) = 2560\) bytes/block（`smem_dep` + `smem_tl` + `smem_keys`）。
  A100の164 KB/SM に対し、8 blocks/SM で 20 KB（12%）。occupancy への影響は無視可能。
- **削減率**：Stage 1 出力の ~4 atomicAdd/warp × 4 warps/block = ~16 atomicAdd/block が、
  ~4 cells × G = ~64 atomicAdd/block（flush）に集約される。
  セル数が少ない場合は Stage 1→2 で追加 ~4倍の削減。

> **v1.0では実装しない**：Persistent Warp（§6.6）との設計上の緊張があるため、v1.0ではStage 2を実装せず、
> 将来拡張とする。Persistent Warpではブロックがカーネル全生存期間にわたって存続し、
> 数百セルを処理するため、N\_BINS=128のヒストグラムは約8ユニークセル（128 ÷ G ≈ 128 ÷ 16 = 8）で飽和し、
> 大半の書き込みがoverflowしてglobal atomicAddにフォールバックする。
> 定期的フラッシュには `__syncthreads()` が必要だが、これはPersistent Warpの
> ワープ独立進行モデルと根本的に矛盾する。History-based DDMC（R9）では
> この問題は発生しないが、v1.0ではIMC/DDMC間のタリーモード統一性を優先し、
> 全カーネルで Stage 1 + Stage 3 方式を採用する。

#### 10.3.5 段階の選択（namelist制御）

| `tally_mode` | 使用段階 | 適用場面 |
|:-------------|:---------|:---------|
| `"global"` | Stage 3 のみ | デバッグ用、CC 7.0 未満のフォールバック |
| `"warp"`（既定） | Stage 1 → Stage 3 | v1.0 baseline。CC 7.0+（Volta以降）必須 |
| `"warp_block"` | Stage 1 → Stage 2 → Stage 3 | 将来拡張（Persistent Warpとの設計上の緊張あり、§10.3.4参照） |

> **namelist制御**：`Parallel.gpu_optimization.tally_mode`（既定 `"warp"`）。
> CC 7.0 未満のGPUでは自動的に `"global"` にフォールバックする。
> `"warp_block"` は将来拡張であり、v1.0では選択不可。

**`tally_mode` と再現性**：
全モード（`"global"`, `"warp"`, `"warp_block"`）で統計的再現を保証する。
`tally_mode` の選択は性能（atomicAdd競合の削減率）のみに影響し、
物理結果の期待値・分散には影響しない。bitwise再現は要求しない。

### 10.4 イベントカウンタ（診断）

IMC輸送カーネル（R8）内で以下の6種類のイベントカウンタをスレッドローカルに蓄積し、
粒子の追跡終了時にグローバルカウンタへ `atomicAdd` で集約する。
カウンタは `unsigned long long`（64ビット）で管理し、オーバーフローの心配はない。

**6種類のイベントカウンタ**：

| カウンタ名 | カウント対象 |
|-----------|-------------|
| `cnt_boundary` | 境界交差イベント数（セル間遷移・反射・脱出を含む全境界交差） |
| `cnt_scatter` | 散乱イベント数（\(\tau_{scatter,remain} \le 0\) での方向再サンプル） |
| `cnt_census` | census終了数（\(t_{remain} \le 0\) でのステップ間保存） |
| `cnt_absorb_kill` | エネルギー枯渇による粒子消滅数（\(E \le 0\) で dead化） |
| `cnt_absorb_survive` | 吸収沈着（alive継続）イベント数（連続吸収 \(\Delta E_{dep} > 0\) の回数） |
| `cnt_roulette_kill` | Russian roulette消滅数（\(\xi \ge p_s\) で dead化） |

**スレッドローカル蓄積**：各カウンタは `uint32_t` のスレッドローカル変数として保持し、
粒子の追跡終了時（dead化・census・MAX\_EVENTS超過）に1回のみ `atomicAdd` で
グローバルカウンタに書き出す。これにより atomic 圧力を粒子あたり6回に抑制する
（各イベントごとの atomicAdd に比べ ~10倍の削減）。

**Fleck 統計**（CPU集計）：
- \(f_{min}\)：全セルのFleck factorの最小値
- \(f_{mean}\)：Fleck factorの算術平均
- \(f_{p95}\)：Fleck factorの95パーセンタイル
- D2H転送後にCPU上でソート・集計する

**光学厚ヒストグラム**（CPU集計）：
- \(\tau > 1, 2, 3, 4\) を満たすセル×群の数をカウント
- DDMC対象領域の分布を把握するための診断情報

**出力条件**：`verbosity="verbose"` 時のみログ出力する。
`[imc_events]` プレフィックスで境界/散乱/census/absorb\_kill/absorb\_survive/roulette\_kill を出力。
`[imc_fleck]` プレフィックスで f\_min/f\_mean/f\_p95 を出力。
`Radiation.imc.difference.enabled=true` の場合は `verbosity!="quiet"` で
`[difference_ref]` プレフィックスを出力し、\(W\) の min/mean/max、active/strong cell 数、
reduced flux、Knudsen proxy、front-gradient indicators、\(E_{ref,total}\) を記録する。
PR9 の `gxii_1d_regression` はこの runtime 診断とは別に
`ablation_multishock_metric` と compressed-shell `shell_dep_noise_cv` を verification
ログへ出力する。これらは production gate の比較指標であり、輸送状態へフィードバックしない。

---

## 11. 数値的"安全策"まとめ（NaN/破綻回避）

### 11.1 エネルギー保存監視
- **エネルギー収支閾値**：\(\varepsilon_{budget} \le \varepsilon_{budget,max}\)（既定 \(\varepsilon_{budget,max} = 10^{-3}\)）
  - \(\varepsilon_{budget}\) の定義は §10.2 参照
  - 超過時の処理：**警告出力**（diagnostics ログ）。停止はしない（`safety.energy_fatal=False` で制御）
  - `safety.energy_fatal=True` に設定すると \(\varepsilon_{budget} > \varepsilon_{budget,max}\) で fatal error
  - IMC/DDMCは粒子法であり、統計ノイズにより \(10^{-6}\) レベルの保存は困難。SPECIFICATION §3.1 の物理要件（平均誤差 ≤ 0.1%）と整合する \(10^{-3}\) を既定とする。検証テスト（解析解比較等、確定論的問題）では \(10^{-6}\) 等へ縮小可能

### 11.2 温度フロア
- **温度クランプ**：\(T_e \ge T_{e,floor}\)、\(T_i \ge T_{i,floor}\)（既定 \(T_{e,floor} = T_{i,floor} = 10^{-3}\) eV、§1.1.7参照。SPECIFICATION §6.4.2 `Te_floor_eV`/`Ti_floor_eV` で個別設定可能）
  - 各演算子（Hydro, Conduction, Laser, Radiation）の温度更新後にクランプを適用
  - クランプ発生時は `clamp_count` を加算し、diagnostics へ出力
  - クランプにより注入されたエネルギーは \(\Delta E_{clamp} = \rho\,c_v\,(T_{floor}-T_{computed})\,V\) で計上し、
    エネルギー収支に含める

- **IMC解析放出とフロア注入の整合（`E_floor_injected`）**：
  IMCの解析的放出デビットはセル・群ごとに
  \[
  \Delta E_{emit,analytic} = \sigma_a \, c \, a \, T^4 \, b_g \, V \, \Delta t
  \]
  で評価される。一方、実際に系外へ放出されるのは離散化されたMC粒子エネルギーであり、
  有限粒子数では \(\Delta E_{emit,analytic}\) と厳密一致しない。
  この差により更新後に \(T_e < T_{e,floor}\) となったセルでは floor clamp が補償注入を行い、
  注入分を `E_floor_injected` に計上する。
  したがってエネルギー保存監視（§10.2 の verify 系収支）では `E_floor_injected` を `E_source`
  に含めて会計する。

- **Radiation thermal microcycling**：
  `Numerics.radiation_thermal_subcycle=True` では、single-stage Radiation 演算子内で
  compressed cell（\(\rho>5.0\) g/cm³）が `Te_floor + 0.5 eV` 以内まで落ちた場合に、
  pre-radiation の熱力学状態へ戻して最大 \(n_{sub}=16\) まで再試行する（§2.1）。
  Retry 前には \(T_e\) の floor guard 近接度から初期 \(n_{sub}\) を予測し、
  floor hit 後の再試行回数を減らす。
  この mode では standalone Strang conduction は二重適用を避けるため skip し、
  各 radiation thermal substep 内で `Radiation -> Qei -> Conduction` を評価する。
  Qei は ion-electron energy exchange 後に \(T_e,T_i,P_e,P_i\) を再クロージャし、
  conduction は既存の `conduction_step` と同じ EOS 同期を実行する。
  これは放射 source injection と温度フロアが結合して pressure support を失う collapse を
  抑えるための radiation callback 側の安全策であり、Hydro 離散化は変更しない。

`clamp_count` はタイムステップ開始時にゼロリセットする。
各 Hydro/Conduction/Laser/Radiation 演算子がフロアクランプを行う度にインクリメント。

閾値：
- \(\text{clamp\_count} > \text{clamp\_warn\_threshold}\)（default 100/step）：WARNING
- \(\text{clamp\_count} > \text{clamp\_fatal\_threshold}\)（default 10000/step）：FATAL

### 11.3 不透明度・断面積
- opacity：κ[cm²/g]→σ[1/cm] 変換を明示し、単位混同を防止（§0.2）
- テーブル範囲外：外挿禁止、clamp＋警告
- **不透明度テーブル範囲外処理**：入力クランプ方式。\((\rho, T)\) をテーブルの定義域 \([\rho_{min}, \rho_{max}] \times [T_{min}, T_{max}]\) にクランプしてから補間を行う。クランプ発生時は WARNING を出力する（1ステップあたり最大10回まで出力し、超過分はカウントのみ記録）。

不透明度クランプ WARNING はランクあたり・ステップあたり最大 10 件出力する。
11件目以降はカウントのみ。ステップ終了時、total > 10 の場合：
`"WARNING: opacity clamp triggered N times this step (first 10 shown)"`
カウントは Planck / Rosseland 区別なく合算。ステップ毎にリセット。

### 11.4 DDMC安全策
- M‑matrix診断に違反するセル×群はDDMC禁止（IMCへ）（§7.3.3）
- diffusion criterion（ω≥0.9 かつ τ≥4 かつ P(μ)制約）を満たさないセル×群はIMCへ（§7.1.2）
- 面Rosseland不透明度は面温度 \(T_{j+1/2}=((T_j^4+T_{j+1}^4)/2)^{1/4}\) で評価し、伝搬停止を回避（§7.3.4）
- リーク/拡散にはRosseland、吸収/放射にはPlanckを使用（grey文献からの多群拡張、§7.3.4参照）
- IMC→DDMC変換確率 P(μ) が 0≤P≤1 を満たさない場合はDDMC禁止（§7.7.1）
- 境界セル（インターフェースセル）のσ_{L,1}は修正版（§7.3.5）を使用。標準のσ_Lを使うとasymptotic BCとの整合が崩れる
- REFLECT/AXIS面のリーク係数は定義上ゼロ（§7.4）。CDF走査でこれらの面が選択されることは到達不能であるが、万一選択された場合は `DeviceErrorFlags::ddmc_reflect_leak` を設定し WARNING を出力する（CUDA_KERNELS §6.5, ARCHITECTURE §10.1）

### 11.5 レーザー安全策
- \(\varepsilon_n\) 下限 + クリティカル終了でIB発散を抑止（§5.2）
- LaserMeshは臨界密度以下の領域のみカバー（`critical_clip`）し、臨界面付近のレイ発散を未然に防止（§5.7.1）
- Leapfrog安定性条件 \(C_{ray} = c\Delta t_{ray}/\Delta x \le\) `cfl_ray`（既定0.8）（§5.3.2）
- クーロン対数 \(\ln\Lambda\) に下限2を設定し非物理値を回避（§5.4.3）
- エネルギー沈着の空間分配は次元依存（1D_SPH: Hydro 1Dセル direct、2D_RZ: 4-node bilinear）（§5.5）
- 1D_SPH は direct deposit→再配分、2D_RZ は transfer 前後のエネルギー保存を検証（§5.8.2）

### 11.6 伝導安全策
- 負温度防止clamp（§11.2の温度フロアと統合）、clamp発生を診断へ
- flux limiter \(f_{lim}\)（§4.1）による非物理的大流束の抑制

### 11.7 流体力学的安全検査

各タイムステップの Hydro 演算子適用後に以下を検査：

1. **密度フロア**：\(\rho_c < \rho_{floor}\)（default \(10^{-10}\) g/cm³）のセルは \(\rho_c = \rho_{floor}\) にクランプ。
   エネルギー注入量 \(\Delta E = (\rho_{floor} - \rho_{old}) \times e_{specific,c} \times V_c\) [erg] を \(E_{floor\_injected}\) に計上。
   ここで \(e_{specific,c} = e_{e,c} + e_{i,c}\) [erg/g] はクランプ前のセル総比内部エネルギー（電子＋イオン）であり、
   \(e_{e,c} = e_e(\rho_{old}, T_{e,c})\)、\(e_{i,c} = e_i(\rho_{old}, T_{i,c})\) は当該セルの現在の温度からEOSで評価する。
   密度増加分 \((\rho_{floor} - \rho_{old})\) の質量が同一温度の物質として注入されたとみなす。
   速度は不変とし、運動エネルギー変化 \(\frac{1}{2}(\rho_{floor}-\rho_{old})|u_c|^2 V_c\) も \(E_{floor\_injected}\) に含める。

2. **内部エネルギー正値性**：\(e_k < 0\) の場合、\(e_k = c_{v,k} \times T_{floor}\) にリセット。
   \(\Delta E\) を \(E_{floor\_injected}\) に計上。

3. **速度制限**：\(|u| > c\)（光速）の場合は WARNING を出力し、
   \(u = u \times (c / |u|)\) にスケーリング（相対論的効果は v1.0 では非対応だが、
   数値的安全のため光速上限を設ける）。

4. **clamp_count**：ステップ開始時にゼロリセット。
   \(\text{clamp\_count} > 100\)（WARNING）、\(> 10000\)（FATAL）。

5. **体積フロア**：\(V_c < V_{floor}\)（\(V_{floor} = 10^{-30}\) cm³）のセルは退化セルとして処理。
   - H8（compute_rho）: \(\rho_c = \rho_{floor}\) にクランプ（ゼロ除算防止）
   - H9（compute_dVdt）: \(dV/dt = 0\), \(\nabla\cdot\mathbf{u} = 0\) に設定
   - R10（normalize_rad_E）: \(E_{rad,c} = 0\) に設定
   - U1（inject_sources）: \(\rho V < 10^{-30}\) の場合は注入をスキップし、未注入エネルギーを \(E_{numerical\_loss}\) に計上
   - 全退化ガードは「clamp+continue」方式（エネルギー収支計上あり）を採用。fatal 停止は行わない。
   - ALE rezone（§3.3）直後に体積がほぼゼロのセルが生じうるため、本ガードが発動する主な場面である。

### 11.8 最大原理監視（Temperature Maximum Principle）

IMCのFleck linearization（§6.1）は暗黙的時間離散化の副作用として、
物質温度が物理的上限を超える「温度オーバーシュート」を生じ得る
（Fleck & Cummings 1971; Larsen & Mercier 1987）。

**最大原理の定義**：
\[
T_{max}^{n} \equiv \max\!\left(\max_i T_{e,i}^n,\; T_{boundary}\right)
\]
ここで \(T_{boundary}\) は Marshak BC の駆動温度（§8.2）。
境界条件が真空または反射のみの場合は \(T_{boundary} = 0\)。

**検出**：放射演算子（R(\(\Delta t\)））適用後に全セルで以下を検査：
\[
\delta_{overshoot,i} = \frac{T_{e,i}^{n+1} - T_{max}^{n}}{T_{max}^{n}}
\]
- \(\delta_{overshoot,i} > 0\) のセルをオーバーシュート違反としてカウント

**診断出力**（各ステップ）：
- `overshoot_count`：違反セル数
- `overshoot_max`：\(\max_i \delta_{overshoot,i}\)（最大超過率）

**閾値**：
- \(\text{overshoot\_count} > 0\) かつ \(\max_i \delta_{overshoot,i} > \varepsilon_{overshoot,warn}\)
  （既定 0.01 = 1%）：WARNING
- \(\max_i \delta_{overshoot,i} > \varepsilon_{overshoot,fatal}\)
  （既定 0.10 = 10%）：FATAL（`safety.overshoot_fatal_enabled=True` 時のみ。既定 False）

> **注意**：最大原理違反は IMC の離散化誤差であり、\(\Delta t\) を小さくすれば緩和される。
> FATAL 停止よりも \(\Delta t\) 制御による自動緩和が望ましい場合が多いため、
> `overshoot_fatal_enabled` の既定は False とする。

> **クランプしない理由**：温度フロア（§11.2）と異なり、オーバーシュートは
> エネルギー保存を維持したまま発生する（余剰エネルギーは放射場から来ている）。
> 人為的にクランプするとエネルギー保存が崩れるため、検出・記録のみ行い
> クランプは行わない。

> **適用タイミング**：放射演算子（R(\(\Delta t\)））の直後のみ検査する。
> Hydro/Conduction/Laser 演算子では最大原理違反は発生しない
> （陽的/STS 更新は安定性条件により単調性を保証する）。

---
