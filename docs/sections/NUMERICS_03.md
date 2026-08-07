<!-- 分割元: docs/NUMERICS.md | このファイルは参照用です。原本（docs/NUMERICS.md）が権威です。 -->
## 3. 幾何と離散化
### 3.1 1D球対称（Lagrangian）

#### 3.1.0a 1D 座標幾何の一般化（W-G, 2026-07-03）

`Mesh.geometry_1d`（既定 `"spherical"`）で 1D の座標幾何を選ぶ。面積・殻体積は
`src/mesh/geometry_1d.cuh` の共有ヘルパで一元化する:
\[
A(r) = 4\pi r^2 \;|\; 2\pi r \;|\; 1, \qquad
V(r_0,r_1) = \tfrac{4\pi}{3}\Delta(r^3) \;|\; \pi\Delta(r^2) \;|\; \Delta r
\]
（円筒は単位軸長、平面は単位断面積あたり）。**bitwise 不変契約**: 球面分岐は
W-G 以前のリテラル式と同一の浮動小数点演算順で評価する（既存 golden 全部が
回帰ゲート）。球面の殻体積は歴史的に 2 綴り
（因数分解形 \((4\pi/3)(r_1-r_0)(r_1^2+r_1r_0+r_0^2)\) と立方差形
\((4\pi/3)(r_1^3-r_0^3)\)）が共存し丸めが異なるため、ヘルパも両形を提供し
各呼び出し箇所は歴史的綴りを維持する。非球面の対応範囲と ConfigError 制約は
SPECIFICATION §6.4 Mesh（imc_ddmc 不可 / cylindrical+S_N は W-G3 まで不可 /
laser は radial_absorption_1d 限定）。W-G1=平面（全現行物理; 平面 S_N は
\(\alpha\equiv0\) で角度再配分が恒等的に消え、BUG-8 の球面 flux dip と
無縁）、W-G2=円筒（FLD + 電子熱伝導 — 伝導カーネルは §4.1a の
template\<int GEOM\> 化で 3 幾何対応、2026-07-04）、W-G3=円筒 S_N
（積求積、別設計）。

#### 3.1.0 Gradedメッシュ生成（1D_SPH）

`Mesh.grid.type="graded"` は区間列
\([r_{s,\mathrm{start}}, r_{s,\mathrm{end}}]\) を使うが、各区間内のセル質量を
スーパーガウシアン形状にする。区間 \(s\) の長さを
\[
L_s = r_{s,\mathrm{end}} - r_{s,\mathrm{start}},
\]
セル数を \(n_s\) とすると、セル \(k=0,\dots,n_s-1\) の正規化中心は
\[
\xi_k = \frac{k+1/2}{n_s}, \qquad
u_k = |2\xi_k - 1|
\]
であり、質量重みは
\[
w_k = \epsilon_{\mathrm{edge}} +
\left(1-\epsilon_{\mathrm{edge}}\right)
\exp\!\left[-\left(\frac{u_k}{\sigma_{\mathrm{sg}}}\right)^{m_{\mathrm{sg}}}\right]
\]
とする。ここで `edge_ratio = \(\epsilon_{\mathrm{edge}}\)`, `sg_sigma = \(\sigma_{\mathrm{sg}}\)`,
`sg_order = \(m_{\mathrm{sg}}\)` であり、\(0 < \epsilon_{\mathrm{edge}} < 1\),
\(0 < \sigma_{\mathrm{sg}} < 1\), \(m_{\mathrm{sg}}\) は偶数かつ \(m_{\mathrm{sg}} \ge 2\)。

球対称ではセル質量が
\[
m_k \propto r_k^2 \Delta r_k
\]
に比例するため、セル幅そのものではなく質量分布が \(w_k\) に従うように
\(\Delta r_k\) を決める。まず等間隔中心推定
\[
r_k^{\mathrm{est}} = r_{s,\mathrm{start}} + L_s \xi_k
\]
を用い、原点区間（\(r_{s,\mathrm{start}} < 10^{-12}\)）の特異性を正則化する
参照長
\[
r_{\mathrm{ref}} =
\begin{cases}
L_s / \sqrt{n_s} & (r_{s,\mathrm{start}} < 10^{-12}\ \mathrm{cm}) \\
0 & (\text{それ以外})
\end{cases}
\]
を導入して
\[
q_k = \frac{w_k}{(r_k^{\mathrm{est}})^2 + r_{\mathrm{ref}}^2}, \qquad
\Delta r_k = L_s \frac{q_k}{\sum_{j=0}^{n_s-1} q_j}
\]
を定義する（**2026-07-26 スペック是正**: 旧記載の
\(r_{\mathrm{guard}}=10^{-20}\,\mathrm{cm}\) の max クリップは実装に存在
しない。実装は上記の \(r_{\mathrm{ref}}\) 加算正則化であり、これが無いと
原点区間では \(N \to \infty\) でも第 1 セルが区間長の \(8/\pi^2 \approx 81\%\)
・一様密度質量の約 53% を占める定性的破綻になる — AI review 2026-07-26
k01 P0-1 の指摘は旧スペック式に対して正しい。\(r_{\mathrm{ref}}\) 付きの
実装では第 1 セル幅は \((2/\pi)L_s/\sqrt{n_s}\) に整い、質量比は
\(O(n_s^{-3/2})\) で消える）。これにより
\[
m_k \propto r_k^2 \Delta r_k \approx w_k
\]
となり、区間中央のセル質量が大きく、区間端では小さくなる。ただし推定中心
\(r^{\mathrm{est}}\) は生成後の実セル中心と自己無撞着でないため、原点近傍
では質量分布は \(w_k\) 比から有界にずれる（中心側へ過剰細分化する側）。

**`Mesh.grid.grading.mapping="exact_measure_v2"`（opt-in、1D 専用；k01
P0-1 是正）**：推定半径近似を使わず、累積重み分率
\[
W_k = \frac{\sum_{\ell=0}^{k} w_\ell}{\sum_{\ell=0}^{n_s-1} w_\ell}
\]
を厳密な殻測度で反転してノードを直接生成する：
\[
r_{k+1} = \left[r_{s,\mathrm{start}}^{\,d} +
W_k\left(r_{s,\mathrm{end}}^{\,d} - r_{s,\mathrm{start}}^{\,d}\right)\right]^{1/d},
\qquad
d = \begin{cases} 3 & \text{spherical} \\ 2 & \text{cylindrical} \\ 1 & \text{planar} \end{cases}
\]
定密度区間のセル質量は丸めを除き厳密に \(m_k \propto w_k\) となる
（原点区間含む）。\(\epsilon_{\mathrm{edge}} \to 1\) の極限は等質量ゾーニング
（§3.1.12a と同形）であり、legacy の一様「幅」ショートカットとは意味が
異なることに注意。区間境界の幾何平均幅補正（下記）は v2 でも同一に適用
されるため、多区間 deck の境界帯では厳密性が局所的に緩む。既定は
`"legacy_estimated_radius"`（bitwise 保存）。2D_RZ での指定は ConfigError。

隣接区間 \(s\) と \(s+1\) の境界では、左区間末尾セル幅
\(\Delta r_{s,-}\) と右区間先頭セル幅 \(\Delta r_{s+1,+}\) から
\[
\Delta r_{*,s+1/2} = \sqrt{\Delta r_{s,-}\,\Delta r_{s+1,+}}
\]
を境界目標幅とする。複数境界を持つ中間区間では、境界ごとに順次補正すると先に合わせた幅を
後段補正が壊すため、左右境界目標幅を先に確定し、残りセルだけを一括スケールする。
区間 \(s\) が左右両側に内部境界を持つとき、
\[
\Delta r_{s,0} = \Delta r_{*,s-1/2}, \qquad
\Delta r_{s,n_s-1} = \Delta r_{*,s+1/2},
\]
\[
\Delta r_{s,k} \leftarrow
\Delta r_{s,k}
\frac{L_s - \Delta r_{*,s-1/2} - \Delta r_{*,s+1/2}}
{\sum_{j=1}^{n_s-2} \Delta r_{s,j}},
\qquad k=1,\dots,n_s-2.
\]
最初と最後の区間は片側境界だけを固定し、残りセルを同様に再スケールする。
セル数が少なすぎて境界目標幅と区間長を両立できない場合は、その構成は不正とする。

> **優先順位の正典化（2026-07-26 AI カーネルレビュー k17 Z-01）**: 本構成は
> 「区間総長さの厳密保存」と「界面近傍セル幅の連続（幾何平均目標）」を厳密に
> 満たし、その代償として「セル質量が \(w_k\) に厳密比例」は近似に留める
> （\(q_k\) は一回評価の推定半径に基づき、境界幅補正はさらに \(w_k\) から
> ずらす）。厳密 mass-coordinate 生成（\(r_i=[a^3+(b^3-a^3)W_i/W]^{1/3}\)）は
> 未実装の将来 opt-in 候補（レビュー Z-02、ユーザー裁定待ち）。
>
> **実行可能条件の enforcement（同 Z-03; 実装 `graded_grid_apply_boundary_targets`）**:
> 「不正とする」の実体は fail-loud assert 群 — \(n_s=1\): 目標幅=区間長（両側
> 目標時は一致も要求）、\(n_s=2\)（両側目標）: \(h_L+h_R=L_s\)、\(n_s\ge3\):
> 補正前後の内部幅和 \(>0\)（\(L_s-h_L-h_R>0\) と等価）。補正後は全幅の
> 有限性・正値・幅和=区間長（tol \(10^{-12}L\)）・ノード狭義単調を再検査する。

ノードは
\[
r_0 = r_{0,\mathrm{start}}, \qquad
r_{i+1} = r_i + \Delta r_i
\]
で累積生成し、最後のノードだけは丸め誤差の蓄積を避けるため
\[
r_N = r_{\mathrm{max}} = r_{S-1,\mathrm{end}}
\]
に固定する。区間境界の連続性は
\(|r_{s,\mathrm{end}} - r_{s+1,\mathrm{start}}| \le 10^{-14} r_{\max}\) で検査する。

**適用範囲（2026-07-17 stage A/A'/A''/B2 拡張）**: 歴史的に `graded` は
1D_SPH 専用だったが、現在は 2D_RZ の全方向に拡張されている。2D 経路は
`build_graded_nodes(pin_segment_boundaries=true)` を用い、**各区間境界
ノードを区間端値に厳密固定**する（上記の最終ノードのみの固定に加えて全
内部区間境界も固定 — 材料界面がリング/格子面/θ 面に exact に載る body-fit
契約）。1D 経路は `pin_segment_boundaries=false` で歴史的挙動（累積生成
+ 最終ノード固定のみ）を bitwise 維持する。方向別の受理面:

- **radial（1D r / polar s / rect r）**: `Mesh.grid`/`grid_r` segments、
  または `auto_regions`（`core/auto_zone` — 隣接セル質量比 ≤ 1.3 契約の
  自動ゾーニング、void 帯・bridge 帯付き）。
- **z（rectangular_rz）**: `grid_z` segments / `auto_regions_axis="z"`
  （planar 質量測度）。polar では z index は角度のため不可。
- **θ（spherical_polar_halfplane）**: `grid_theta` segments（radian、
  \([0,\pi]\) 被覆、末尾ノードは許容 \(10^{-12}\) 検査後に \(\pi\) へ exact
  snap; `polar_equal_mu_zoning` と排他）。コーン壁 \(\theta_{\rm cone}\)
  等のセクタ境界を θ 面に厳密整合させる。
- **multiblock polar shell**: `grid` segments が shell 半径分布
  \([r_{\rm match}, s_{\max}]\) を駆動（被覆端は exact snap、Σ zones =
  `nr`）。既定（segments 無し）は歴史的一様分布と bitwise 恒等。

デフォルト（新キー未使用）の全経路は bitwise 不変が硬い契約であり、
uniform/equal-μ/一様 shell の歴史分岐は逐語保持される。宣言的な複合
ジオメトリ（球+コーン+スロープ等）から各方向 segments を合成する deck
前処理は `tools/mesh_planner.py`（runtime 非依存）が担う。

#### 3.1.1 スタガード格子配置

1D球対称メッシュのスタガード配置（2D RZ §3.2と同じ思想）。

- **ノード** \(j\)（\(j=0,\ldots,N\)）：セル境界半径 \(r_j\)、速度 \(u_j\)
- **セル** \(i\)（\(i=0,\ldots,N-1\)）：\(i\) 番セルは \([r_j, r_{j+1}]\)（添字対応：セル \(i\) の左ノードは \(j=i\)、右ノードは \(j=i+1\)）
  - 但し文献では \(i = j+1/2\) 表記も一般的。本書では \(i\)＝セル、\(j\)＝ノードで統一

**ノード中心量**：\(r_j,\; u_j\)

**セル中心量**：\(\rho_i,\; e_{i,i},\; e_{e,i},\; T_{i,i},\; T_{e,i},\; P_{i,i},\; P_{e,i},\; Q_i\)

#### 3.1.2 セル体積・質量・密度

セル体積：
\[
V_i = \frac{4\pi}{3}\left(r_{j+1}^3 - r_j^3\right)
= \frac{4\pi}{3}(r_{j+1}-r_j)(r_{j+1}^2 + r_{j+1} r_j + r_j^2)
\]

> **実装注意（桁落ち回避）**：右辺の因数分解形を使用すること。
> 左辺の直接差分 \(r_{j+1}^3 - r_j^3\) は薄殻（\(r_{j+1}/r_j \approx 1\)）で
> 桁落ちが発生する（例：\(r=10\) cm, \(\Delta r = 10^{-3}\) cm で約6桁損失）。

セル質量：
\[
\Delta M_i = \rho_i\,V_i
\]

純Lagrangian（rezone/remapなし）では \(\Delta M_i\) は時間不変である。一方、
2D RZ ALE remapでは conservative remap によりセルごとの質量は変化し得る。
1D solution-adaptive ALE（V3）でも、有効化時は同様にセルごとの質量が変化し得る。
したがって保存保証は、純Lagrangianでのセルごとの不変性から、ALE remap下での
大域量 \(\sum_i M_i\) の保存（V3仕様に従う場ごとの許容誤差内）へ移る。

密度更新：
\[
\rho_i = \frac{\Delta M_i}{V_i}
\]

セル界面面積：
\[
A_j = 4\pi\,r_j^2
\]

`Main.dimension="1D_CYL"` selects cylindrical per-unit-length geometry:
\[
V_i = \pi\left(r_{j+1}^2-r_j^2\right), \qquad
A(r) = 2\pi r,
\]
and the two-face area average used by 1D node/cell diagnostics is
\(0.5\,2\pi(r_a+r_b)\).  Implementation is by compile-time `Geometry1D`
template forks of the 1D kernels; spherical branches keep the historical
arithmetic verbatim (bitwise regime), with 1D_SPH bit-neutrality anchored by
`tests/hydro/test_1d_sph_bitwise_golden.cu`.  Wave-1 scope is the pure hydro
core: radiation, laser, conduction, and ale1d are rejected at validation.  The
boundary-PdV energy-audit diagnostic is geometry-aware.  The Sedov gate is
H3-RADIAL-CYL multi-grid exponent \(1/2\).

#### 3.1.3 ノード質量

ノード \(j\) の質量は隣接2セルからの寄与：
\[
\Delta M_j = \frac{1}{2}\left(\Delta M_{i-1} + \Delta M_i\right)
\]
- \(j=0\)（中心）：\(\Delta M_0 = \frac{1}{2}\Delta M_0^{cell}\)
- \(j=N\)（外側境界）：\(\Delta M_N = \frac{1}{2}\Delta M_{N-1}\)

#### 3.1.4 運動量方程式

\[
\Delta M_j\,\frac{du_j}{dt} = -A_j\left[(P+Q)_{i} - (P+Q)_{i-1}\right]
\]
ここで \(P_i = P_{i,i} + P_{e,i}\)（イオン圧力＋電子圧力）、\(Q_i\) は人工粘性（§3.1.6）。

加速度：
\[
a_j = -\frac{A_j}{\Delta M_j}\left[(P+Q)_{i} - (P+Q)_{i-1}\right]
\]

**境界条件**：
- 中心（\(j=0\)）：\(u_0 = 0\)（対称性）、\(r_0 = 0\) を維持
- 外側（\(j=N\)）：自由境界（\(P_{ext}=0\)）or 固定壁（\(u_N=0\)）

**1D odd-even suppression**：
1D_SPH では、節点力に使うセル圧力和
\[
(P+Q)_i = P_{e,i} + P_{i,i} + Q_i
\]
を構成したあと、`Numerics.hydro.odd_even_damping_C = C_{oe} > 0` の場合だけ
checkerboard filter を加速度経路へ入れる。内部 active セル
\(2 \le i \le N_{cell}-3\) で \(Q_i \le 0\) かつ、\(i-1\), \(i\), \(i+1\) の各セルが
\((P+Q)\) と \(\rho\) の両方で極値
\[
\delta^- \phi_k = \phi_k - \phi_{k-1},\qquad
\delta^+ \phi_k = \phi_{k+1} - \phi_k,\qquad
\delta^- \phi_k\,\delta^+ \phi_k < 0
\]
を満たすとき、5-cell checkerboard とみなして
\[
W^{cb}_{\phi,k} =
\frac{\min(|\delta^- \phi_k|,|\delta^+ \phi_k|)}
     {\max(|\delta^- \phi_k|,|\delta^+ \phi_k|,\varepsilon_{cb})},
\qquad
W_{cb,i} = \min\left(
W^{cb}_{P+Q,i-1}, W^{cb}_{P+Q,i}, W^{cb}_{P+Q,i+1},
W^{cb}_{\rho,i-1}, W^{cb}_{\rho,i}, W^{cb}_{\rho,i+1}\right)
\]
を定義する。ここで \(\varepsilon_{cb}=10^{-30}\) である。加速度に使う filtered pressure は
\[
\widetilde{(P+Q)}_i =
(P+Q)_i + \beta_{eff}\,W_{cb,i}\,
\frac{(P+Q)_{i-1} - 2(P+Q)_i + (P+Q)_{i+1}}{2},
\qquad
\beta_{eff} = 0.15\,\min(C_{oe}, 1)
\]
で与える。境界セル、inactive セル、および \(Q_i > 0\) の shock-supporting cell は
raw 値を保持する。一方で、隣接セルの \(Q=0\) 領域は filter 対象に残し、shock foot 直後の
odd-even mode を抑える。

その上で odd-even モード専用の保存的 damping force を節点へ加える。検出は圧力ではなく比体積
\[
s_i = \frac{V_i}{\Delta M_i} = \frac{1}{\rho_i}
\]
に対して行う。内部セル \(1 \le i \le N_{cell}-2\) で
\[
\Delta s_L = s_i - s_{i-1},\qquad
\Delta s_R = s_{i+1} - s_i
\]
を定義し、
\[
\Delta s_L\Delta s_R < 0
\]
を満たすとき odd-even 極値と判定する。このとき
\[
W_{oe,i} =
\frac{\min(|\Delta s_L|,|\Delta s_R|)}{\max(|\Delta s_L|,|\Delta s_R|)}
\in [0,1]
\]
とする。境界セル \(i=0\), \(i=N_{cell}-1\) および不成立セルでは
\(W_{oe,i}=0\) とする。

セル \(i\) の damping 係数は
\[
A_i^\ast = \frac{4\pi}{2}\left(r_i^2 + r_{i+1}^2\right),\qquad
\mu_i = \rho_i\,c_{s,i}\,A_i^\ast
\left(C_{oe}\,W_{oe,i} + C_{psv}\,\psi_i^{eff}\right)
\]
で定義する。ここで \(C_{oe}\) は namelist パラメータ
`Numerics.hydro.odd_even_damping_C`、\(C_{psv}\) は
`Numerics.hydro.post_shock_velocity_damping_C` であり、既定値はいずれも `0.0`（無効）である。
recent-shock weight は §3.1.6 の shock history を流用して
\[
\tau_i = C_{decay}\,\frac{\Delta r_i}{\max(c_{s,i},\varepsilon_{ps})},\qquad
\psi_i = \exp\!\left(-\frac{t - t_i^{shock}}{\tau_i}\right)
\]
とし、velocity damping では shock core を保護するため
\[
\psi_i^{eff} =
\begin{cases}
0, & Q_i > f_{core}\,Q^{max} \\
\psi_i, & \text{otherwise}
\end{cases},
\qquad f_{core}=0.2
\]
を使う。ここで \(Q^{max}\) は当該加速度評価に使う AV 場
（Predictor では \(Q^n\)、Corrector では \(Q^{n+1/2}\)）の active セル最大値であり、
\(C_{decay}\) は `post_shock_heat_decay` である。

節点 \(j\) の追加力は
\[
F^{oe}_j = \mu_j(u_{j+1}-u_j) - \mu_{j-1}(u_j-u_{j-1})
\]
で与える。したがって 1D_SPH の運動量方程式は
\[
\Delta M_j\,\frac{du_j}{dt} =
-A_j\left[\widetilde{(P+Q)}_j - \widetilde{(P+Q)}_{j-1}\right] + F^{oe}_j
\]
となる。`C_{oe}=0` では \(\widetilde{(P+Q)}=(P+Q)\) に戻る。内部節点では左右セルの
寄与が差分形で入るため、odd-even damping は線形運動量保存の形で作用する。

legacy PdV path で用いる対応 heat はセルごとに
\[
H^{oe}_i = \mu_i (u_{i+1} - u_i)^2 \ge 0
\]
と定義し、full step のエネルギー更新後に既存の人工熱流束 \(H\) と同じ経路で
比内部エネルギーへ加える。これにより PdV 仕事と人工粘性仕事の定義は変更せず、
odd-even damping による運動エネルギー散逸だけを非負の heat source として戻す。
exact compatible path ではこの separate heat は加えず、Corrector 加速度に入った
cell-pair force \(F^{oe}_{i,L}=+\mu_i(u_{i+1}-u_i)\),
\(F^{oe}_{i,R}=-\mu_i(u_{i+1}-u_i)\) の work を §3.1.5 の compatible work へ直接含める。
checkerboard filter 自体には追加の compatible heat を定義しない。

**1D high-k nodal velocity damper**：
`Numerics.hydro.hk_velocity_damper_C = C_{hk} > 0` のとき、Corrector の節点速度更新と
1D境界条件適用の後、最終密度再計算の前に、2–4 cell wavelength の節点速度ノイズだけを
対象にした保存的 impulse damper を1回適用する。既定 \(C_{hk}=0\) では完全に無効である。

各節点 \(j\) で、隣接3節点 \((j-1,j,j+1)\) の局所線形最小二乗
\[
u(r) = a_j + b_j(r-r_j)
\]
を解き、smooth 成分 \(\bar{u}_j=a_j\)、high-k 残差 \(\delta u_j=u_j-\bar{u}_j\) を定義する。
正規方程式は \(x_k=r_k-r_j\) として
\[
a_j =
\frac{\left(\sum u_k\right)\left(\sum x_k^2\right)
      -\left(\sum u_k x_k\right)\left(\sum x_k\right)}
     {3\sum x_k^2-\left(\sum x_k\right)^2}
\]
である。境界節点、inactive/void に接する節点、material interface に接する節点、
または退化した幾何では \(\delta u_j=0\) とする。このため定数速度および
線形速度場は非一様 Lagrange mesh 上でも厳密に保存される。

セル \(i\) の両端節点 pair \((i,i+1)\) に対して、red-black 順に
\[
q_i = u_i-u_{i+1},\qquad q_i^{hk}=\delta u_i-\delta u_{i+1},\qquad
\mu_i^n=\left(\frac{1}{M_i^n}+\frac{1}{M_{i+1}^n}\right)^{-1}
\]
を構成する。ここで \(M_j^n\) は節点質量である。noise sensor は
\[
S_i^{noise} =
\frac{(q_i^{hk})^2}{(q_i^{hk})^2+\left[(u_i-\delta u_i)-(u_{i+1}-\delta u_{i+1})\right]^2+\epsilon}
\]
で、\(S_i^{noise}<0.5\) の pair は damping しない。

さらに以下の mask がすべて成立する pair のみを対象にする。red-black impulse loop の前に
cell-centered front mask \(F_c\) を作る。front cell は
\[
Q_c^{half}>0,\quad \texttt{cell\_is\_void}_c,\quad
|\ln(T_{e,c}/T_{e,c-1})|>\texttt{hk\_velocity\_damper\_grad\_Te\_max},\quad
|\ln(\rho_c/\rho_{c-1})|>\texttt{hk\_velocity\_damper\_grad\_rho\_max}
\]
のいずれかを満たす cell（gradient 条件では face 両側の cell）である。
\(F_c\) を `hk_velocity_damper_guard_cells` cell 半幅（既定 25）で展開して
`near_front` を作り、pair \((i,i+1)\) は `near_front[i]` が真なら無効化する。
gradient 閾値の既定は \(T_e\) が 0.2、\(\rho\) が 0.3 である。
- half-step 粘性 \(Q_c^{half}=(Q_c^n+Q_c^{n+1/2})/2\) について
  \(Q_{i-1}^{half},Q_i^{half},Q_{i+1}^{half}=0\)（shock/AV active stencil では secondary guard として無効）
- cell optical depth \(\theta_c=\max(\sigma_{R,\max,c},0)\Delta r_c\) から
  node optical depth \(\tau_j=\max(\theta_{j-1},\theta_j)\) を作り、
  pair の両端で \(\min(\tau_i,\tau_{i+1})\ge
  \texttt{hk\_velocity\_damper\_tau\_min}\)
- 隣接セル jump が
  \(|\Delta\ln T_e|\le\texttt{hk\_velocity\_damper\_grad\_Te\_max}\),
  \(|\Delta\ln\rho|\le\texttt{hk\_velocity\_damper\_grad\_rho\_max}\)
  （buffered front mask に加え、immediate pair stencil の secondary guard として残す）
- void cell と material interface を含まない

ここで \(\sigma_{R,\max}\) は直近の radiation stage が保持した
\(\max_g\sigma_{R,g}\) であり、Strang 分割の先頭 hydro half-step では1 radiation stage 古い値を
用いる。まだ値がない場合、\(\texttt{hk\_velocity\_damper\_tau\_min}>0\) では damper を無効にする。

impulse は小減衰近似
\[
I_i = C_{hk} S_i^{noise}\mu_i^n q_i^{hk}
\]
で与える。ただし \(I_i q_i\le0\) のときは適用せず、相対速度の符号を反転しない範囲に
\(|I_i|\) を制限する。節点速度更新は
\[
u_i \leftarrow u_i - \frac{I_i}{M_i^n},\qquad
u_{i+1} \leftarrow u_{i+1} + \frac{I_i}{M_{i+1}^n}
\]
であり、pair ごとの線形運動量を保存する。実際に失われた kinetic energy は
\[
\Delta K_i =
I_i q_i - \frac{I_i^2}{2}\left(\frac{1}{M_i^n}+\frac{1}{M_{i+1}^n}\right) \ge 0
\]
を用いて測定し、full-step の main energy update 後にセル \(i\) のイオン比内部エネルギーへ
\(\Delta e_{i,i}^{hk}=\Delta K_i/\Delta M_i\) として加える。1T では全内部エネルギー
\(e_i\) へ加える。2T では `av_heat_to` に依存せず常に ion energy へ加える。
debug diagnostic は step ごとに
`[hk_vdamp] step=N active_pairs=M front_blocked=K tau_blocked=T KE_removed=X`
を出力し、`front_blocked` は buffered front mask と immediate secondary front/shock
guard で無効化された pair 数を表す。

#### 3.1.5 エネルギー方程式（2T）

§1.1.3の2Tモデルを1D球対称で離散化する。体積変化率：
\[
\frac{dV_i}{dt} = A_{j+1}\,u_{j+1} - A_j\,u_j
\]

速度発散：
\[
(\nabla\cdot\mathbf{u})_i = \frac{1}{V_i}\frac{dV_i}{dt}
\]

**Corrector のエネルギー離散化**：
- 既定（`Numerics.hydro.compatible_energy = false`）では legacy PdV 更新として
  \[
  \Delta V_i = V_i^{n+1} - V_i^n
  \]
  を用いる。
- 1D_SPH 専用オプション（`compatible_energy = true`）かつ ideal-gas hydro EOS path、
  または TMAT legacy table EOS path では、
  Corrector の節点加速度に使ったものと同じ cell-centered
  \(p_{q,i}^{n+1/2}=P_{e,i}^{n+1/2}+P_{i,i}^{n+1/2}+Q_i^{n+1/2}\)
  （odd-even pressure filter が有効な場合は filter 後の値）から、節点運動量更新と
  work-conjugate な内部エネルギー更新を行う。TMAT では backend が
  `eos.hydro_backend="legacy"` で、ion/electron table が \(T\) 方向に
  monotone な \(T(\rho,e)\) inverse reclosure を持つ場合だけ許可する。
  IONMIX / SESAME / Helmholtz / rho-e table / Mie-Gruneisen hydro EOS path は
  現実装では未対応であり、`compatible_energy=true` と同時指定すると
  namelist validation または EOS context validation で `ConfigError` とする。
  post-shock velocity damping force が有効な場合は v1 では legacy PdV 更新へ fall back し、
  step 0 で warning を出す。
  \[
  r_{j}^{n+1/2} = \frac{r_j^n + r_j^{n+1}}{2},\qquad
  A_j^{n+1/2} = 4\pi \left(r_j^{n+1/2}\right)^2
  \]
  節点 \(j\) に対する cell \(i\) の力寄与は、既存の
  `compute_acceleration_1d_kernel` と同じ符号で
  \[
  F_{i\to i}^{n+1/2} = -A_i^{n+1/2}p_{q,i}^{n+1/2},\qquad
  F_{i\to i+1}^{n+1/2} = +A_{i+1}^{n+1/2}p_{q,i}^{n+1/2}
  \]
  とする。外側 free boundary では `pressure_ghost_1d` と同じ ghost convention を使い、
  現行実装どおり thermal ghost pressure は 0、ghost \(Q\) は境界セルから copy されるため、
  外側 face の material force work は \(p_{q,N-1}^{n+1/2}-p_{q,ghost}^{n+1/2}\) を用いる。
  速度の work-conjugate 平均
  \[
  \bar u_j = \frac{u_j^n+u_j^{n+1}}{2}
  \]
  を用い、cell \(i\) の material internal energy 変化は
  \[
  \Delta E_i^{comp} = -\Delta t\left(
  F_{i\to i}^{n+1/2}\bar u_i + F_{i\to i+1}^{n+1/2}\bar u_{i+1}\right)
  \]
  を pressure/viscosity force work として計算する。`odd_even_damping_C>0` の場合は、
  同じ Corrector 加速度に使った pair force
  \[
  F^{oe}_{i,L}=+\mu_i^{n+1/2}(u_{i+1}^{n+1/2}-u_i^{n+1/2}),\qquad
  F^{oe}_{i,R}=-F^{oe}_{i,L}
  \]
  から
  \[
  \Delta E_i^{oe}=-\Delta t\,F^{oe}_{i,L}\left(\bar u_i-\bar u_{i+1}\right)
  \]
  を加える。この compatible path では separate \(H_i^{oe}\) heat は加えない。
  このとき閉境界では
  \(\sum_i\Delta E_i^{comp}+\sum_j\Delta K_j=O(\mathrm{roundoff})\) となる。
  ideal-gas exact path と TMAT inverse-reclosure path では `energy_update_with_old_volume_*` の
  \(V^{n+1}-V^n\) 型 PdV/Q 更新は実行しない。
  high-k velocity damper は Corrector 後の別 impulse として扱い、この compatible work には
  含めず、既存の測定済み \(\Delta K\) heat deposition を使う。
- 既存診断 `eta_compatible` は幾何学的な volume mismatch 診断であり、exact compatible path の
  conservation mechanism ではない。legacy fallback では従来どおり
  \[
  \Delta V_{i,\mathrm{comp}} =
  \Delta t\left(A_{i+1}^{n+1/2}u_{i+1}^{n+1/2} - A_i^{n+1/2}u_i^{n+1/2}\right)
  \]
  と \(V^{n+1}-V^n\) の差を保持する。

**イオンエネルギー**：
\[
\Delta M_i\,\frac{de_{i,i}}{dt} =
-(P_{i,i} + Q_i)\frac{dV_i}{dt} + Q_{ei,i}\,V_i - (\nabla\cdot\mathbf{H})_i\,V_i
\]

**電子エネルギー**：
\[
\Delta M_i\,\frac{de_{e,i}}{dt} = -P_{e,i}\frac{dV_i}{dt} - Q_{ei,i}\,V_i - (\nabla\cdot\mathbf{q}_e)_i\,V_i + (S_L + S_r)_i\,V_i
\]

- \(Q_{ei,i}\)：§1.1.3のSpitzer e-i緩和
- legacy PdV path では人工粘性 \(Q_i\) による仕事は `av_heat_to` で選ばれた熱容量場へ加える
  （v1.0既定はイオン）
- exact compatible path では総 work \(\Delta E_i^{comp}\) を
  \[
  \tilde f_e=\frac{\max(P_{e,i}^{n+1/2},0)}
  {\max(|p_{q,i}^{n+1/2}|,\epsilon)},\qquad
  \tilde f_{iQ}=\frac{\max(P_{i,i}^{n+1/2}+Q_i^{n+1/2},0)}
  {\max(|p_{q,i}^{n+1/2}|,\epsilon)}
  \]
  から、**明示的な後段正規化**
  \[
  f_e = \frac{\tilde f_e}{\tilde f_e + \tilde f_{iQ}},\qquad
  f_{iQ} = \frac{\tilde f_{iQ}}{\tilde f_e + \tilde f_{iQ}},\qquad
  f_e + f_{iQ} = 1
  \]
  を経た比で分配する（**2026-07-26 スペック是正**: 旧記載は \(\tilde f\) の
  表示式を「正規化した比」と呼ぶのみで正規化式を欠いており、字義どおりでは
  \(\tilde f_e + \tilde f_{iQ} \ne 1\) の場合（例: 負の cold pressure）に
  総 work の過大・過小配分＝保存則違反を意味した。実装は当初からこの
  後段正規化を持つ — AI review 2026-07-26 k01 P0-6）。
  \(\tilde f_e+\tilde f_{iQ}=0\)（または非有限）の退化セルでは 1/2–1/2 とする。
  v1 では artificial-viscosity pressure work \(Q_i^{n+1/2}\) は常に ion energy へ含める。
  TMAT 2T path では compatible work と \(Q_{ei}\) transfer は分離し、
  \(Q_{ei}\) には half-step \(T_e,T_i\) と table 由来の
  \(c_{v,e},c_{v,i}\) を使う。compatible work 後の reclosure は
  各成分を独立に
  \[
  T_{e,i}^{n+1}=T_e(\rho_i^{n+1},e_{e,i}^{n+1}),\qquad
  T_{i,i}^{n+1}=T_i(\rho_i^{n+1},e_{i,i}^{n+1})
  \]
  と monotone bracketed inverse で求め、\(P_e,P_i,c_v\) を table から再評価する。
  compatible path では table inverse が範囲内にある限り `ee/ei` は writeback せず、
  NaN/Inf/負 energy または table floor/ceiling clamp 時だけ table boundary energy へ repair する。
  compatible TMAT reclosure の floor/ceiling clamp と bracket failure は device counter で集計し、
  非ゼロなら hydro warning として出力する。
- `eta_compatible` 診断量
  \[
  \eta_i = (V_i^{n+1} - V_i^n) - \Delta V_{i,\mathrm{comp}}
  \]
  は legacy volume-form compatible 診断として定義され、exact force-work path の保存則には使わない。
- 人工熱流束 \(\mathbf{H}\) は §3.1.6 の 1D_SPH 専用 shock-smoothing 項であり、
  2T ではイオン方程式にのみ、1T では全比内部エネルギー \(e\) に同じ離散項を適用する
- `Numerics.hydro.ion_art_heat_C > 0` のときは、§3.1.6 の
  ion-only artificial heat conduction \(H^{ion}\) を、main/compatible work 更新後かつ
  EOS 再閉包前にイオン比内部エネルギー \(e_i\) へ保存形で加える。
  この項は compatible force-work 分解には含めない
- `post_shock_heat=True` のときは、別経路の局所熱流束 \(\mathbf{H}^{ps}\) を
  全比内部エネルギー \(e_{tot}=e_e+e_i\) に対して保存形で加える。
  2T ではセルの net \(\Delta e_{tot}^{ps}\) を
  \(P_e^{n+1/2}/(P_e^{n+1/2}+P_i^{n+1/2})\) と
  \(P_i^{n+1/2}/(P_e^{n+1/2}+P_i^{n+1/2})\) の比で電子・イオンへ分配する。
  分母が 0 の退化セルでは 1/2–1/2 とする
- odd-even damping の legacy compatible heat \(H^{oe}\) も `av_heat_to` で選ばれた同じ熱容量場へ加える。
  exact compatible path では \(H^{oe}\) を別途加えず、force work \(\Delta E_i^{oe}\) を
  同じ熱容量場へ入れる
- `Numerics.hydro.ee_odd_even_C = C_{ee}^{oe} > 0`（1D_SPH + 2T 専用）のときは、
  hydro の main energy update、人工熱流束 \(H\)、post-shock heat \(H^{ps}\) の後、
  EOS による \(T_e,P_e\) 再閉包の前に、電子比内部エネルギー \(e_{e,i}\) へ
  checkerboard 専用の保存的 face filter を追加する。pre-EOS では \(T_e\) がまだ
  再構成されていないため、検出は \(e_e\) を \(T_e\) の単調 proxy として行う
- セル重みは
  \[
  \Delta^- e_{e,i} = e_{e,i} - e_{e,i-1},\qquad
  \Delta^+ e_{e,i} = e_{e,i+1} - e_{e,i}
  \]
  に対して
  \[
  \Delta^- e_{e,i}\,\Delta^+ e_{e,i} < 0,\qquad
  \left|e_{e,i+1} - 2e_{e,i} + e_{e,i-1}\right|
  > 0.1 \cdot \frac{|e_{e,i+1} - e_{e,i-1}|}{2} + \varepsilon_{ee},
  \quad \varepsilon_{ee}=10^{-30}
  \]
  を同時に満たすときだけ
  \[
  W^{ee}_i =
  \frac{\min(|\Delta^- e_{e,i}|, |\Delta^+ e_{e,i}|)}
       {\max(|\Delta^- e_{e,i}|, |\Delta^+ e_{e,i}|, \varepsilon_{ee})}
  \]
  と定義し、それ以外では \(W^{ee}_i=0\) とする
- face \(i+\tfrac12\) では、左右セルがともに \(W^{ee}>0\) を持ち、かつ隣接 2 セルの
  曲率が反転して
  \[
  \delta^2 e_{e,i} = e_{e,i+1} - 2e_{e,i} + e_{e,i-1},\qquad
  \delta^2 e_{e,i+1} = e_{e,i+2} - 2e_{e,i+1} + e_{e,i},
  \qquad
  \delta^2 e_{e,i}\,\delta^2 e_{e,i+1} < 0
  \]
  を満たすときだけ flux を立てる。first/last non-void cell、inactive cell、
  void cellに隣接する face では \(F^{ee}_{i+1/2}=0\) とする
- 交換質量は体積重み付き
  \[
  V_{i+1/2} = \frac{2V_iV_{i+1}}{V_i + V_{i+1}},\qquad
  \rho_{i+1/2} = \frac{\rho_i + \rho_{i+1}}{2},\qquad
  M_{i+1/2}^{ee} = \rho_{i+1/2}V_{i+1/2}
  \]
  とし、face flux は
  \[
  F^{ee,raw}_{i+1/2} =
  C_{ee}^{oe}\,\min(W^{ee}_i, W^{ee}_{i+1})\,
  M_{i+1/2}^{ee}\,(e_{e,i+1} - e_{e,i})
  \]
  に **hull（donor）cap**
  \[
  \mu_{i+1/2} = \frac{\Delta M_i\,\Delta M_{i+1}}{\Delta M_i + \Delta M_{i+1}},
  \qquad
  |F^{ee}_{i+1/2}| \le \mu_{i+1/2}\,|e_{e,i+1} - e_{e,i}|
  \]
  を課して与える（符号は raw のまま；**2026-07-26 追加、AI review k01
  §4.1**: 強い質量 grading では \(M^{ee}_{i+1/2} = \rho_f V_{harm}\) が軽い
  セル自身の質量を超え、cap 無しの一発交換が軽セルの \(e_e\) を相手側を
  越えて負値まで押し込み得た。reduced-mass cap は両セルを
  \([\min(e_L,e_R),\max(e_L,e_R)]\) hull 内に保つ（正値性・単調性）。
  近一様質量では \(C_{ee}^{oe}\,\min(W) \le 1/2\) の限り cap は発火しない）
- 更新は divergence 形
  \[
  e_{e,i}^{n+1} \leftarrow e_{e,i}^{n+1} +
  \frac{F^{ee}_{i+1/2} - F^{ee}_{i-1/2}}{\Delta M_i}
  \]
  で行う。したがって
  \[
  \sum_i \Delta M_i\,\Delta e_{e,i}^{oe} =
  \sum_i \left(F^{ee}_{i+1/2} - F^{ee}_{i-1/2}\right) = 0
  \]
  となり、電子 internal energy の総和は保存される。filter は電子側にのみ作用し、
  \(e_i\)（ion specific internal energy）と momentum equation は変更しない
- `C_{ee}^{oe}=0` ではこの経路は完全に無効。推奨試験値は `0.05-0.15`
- 熱伝導 \((\nabla\cdot\mathbf{q}_e)_i\)：§4参照

**e-i緩和の有限\(\Delta t\)離散化（実装規約）**：
\[
\Delta T = T_e - T_i,\quad
\tau_{eff} = \tau_{eq}\frac{c_{v,i}}{c_{v,e}+c_{v,i}},\quad
m_{ei}=\texttt{Numerics.hydro.qei\_multiplier},\quad
f_{relax}=1-\exp\!\left(-\frac{m_{ei}\Delta t}{\tau_{eff}}\right)
\]
\[
q_{ei}^{step} =
\frac{c_{v,e}c_{v,i}}{c_{v,e}+c_{v,i}}\,
\Delta T\,f_{relax}
\quad [\mathrm{erg/g}]
\]
\[
\Delta e_i \mathrel{+}= q_{ei}^{step},\qquad
\Delta e_e \mathrel{-}= q_{ei}^{step}
\]
ここで \(c_{v,e}, c_{v,i}\) は質量比熱 [erg/(g·eV)]。この更新は
\(\Delta e_i + \Delta e_e = 0\) を厳密に満たし、\(\Delta t \gg \tau_{eq}\) でも
過渡オーバーシュートを起こさない。既定 \(m_{ei}=1\) は従来の物理
coupling と同一である。さらに \(m_{ei}\Delta t \ll \tau_{eff}\) では
\(f_{relax}\approx m_{ei}\Delta t/\tau_{eff}\) となり、
\(q_{ei}^{step}\approx m_{ei}Q_{ei}\Delta t/\rho\) に一致する。

> **床との整合（2026-07-26、AI review k15 1.4/P0-4）**: 実装は shared transfer を
> 許容区間 \(q\in[-\max(e_i,0),\ \max(e_e,0)]\) へ bracket してから
> \(e_e\mathrel{-}=q\)、\(e_i\mathrel{+}=q\) を一つの適用量で更新する
> （1D: `qei_coupling_substep_kernel`・`apply_qei_transfer_2t_kernel`・
> per-material 変種）。旧実装の両側独立 `fmax(·,0)` floor は片側が床に当たると
> 上記の厳密対保存を破っていた（table EOS で frozen-cv 移送が貯蔵エネルギーを
> 超過するセルで到達可能）。bracket は床が発火しない限り bit 恒等であり、
> 発火時は移送を admissible 区間端でクリップする（エネルギー生成なし、
> hydro 側は clip を `clamp_count` で計数）。\(f_{relax}\) の数値評価は
> `-expm1(-x)`（\(x\ll1\) の桁落ち回避、旧 `1-exp(-x)` は相対 ~eps/x 損失）。
> 2D hydro corrector 内の同型独立 clamp は別レーン所掌（cross-lane 項）。

**温度更新**：EOS（§1.1.5）を用いて
\[
T_{k,i}^{new} = T_{k,i}^{old} + \frac{\Delta e_{k,i}}{c_{v,k,i}} \quad (k=e,i)
\]

**テーブルEOSでの非線形逆変換**：テーブルEOSで \(C_v(T)\) の変化が大きい領域（イオン化フロント等）では、上記の線形更新が不正確になりうる。v1.0 では backend ごとに以下の EOS 逆変換を使用する：

1. \(e_{target} = e_{old} + \Delta e_k\) を計算
2. `legacy` raw-table backend：
   - 固定 \(\rho\) の mixed-row energy
     \[
     e_j(\rho) = (1-w_\rho)e_{j,i_0} + w_\rho e_{j,i_1}
     \]
     に対して温度インデックス方向の単調探索を行い、\(e_{j_0} \le e_{target} < e_{j_1}\) を満たす区間で \(\log T\) 線形補間する。
   - Newton は使わず、テーブル端を超えた \(e_{target}\) は table-end temperature へクランプする。
   - Hydro closure はこの inverse map で \(T\), \(P\), \(c_v\) を閉じるが、既定では
     `Numerics.hydro.eos_writeback=True` のため、table-end / floor 側の clamped energy を
     state へ writeback して旧来の毎 step re-projection \(e \leftarrow e(\rho,T)\) を行う。
     `eos_writeback=False` では Hydro が更新した \(e_{target}\) 自体は保持し、
     \(e_{target}\) が NaN / Inf / 負のときだけ repair step とみなして clamped energy を戻す。
3. `exact_ideal_gas` backend（1D 診断）：
   - 非線形逆変換は使わず、定数比熱から
     \[
     T_i = e_i/c_{v,i}, \qquad T_e = e_e/c_{v,e}
     \]
     を直接計算する。1T では \(T=e/(c_{v,i}+c_{v,e})\) を使う。
4. `rho_e_table` backend：
   - 1T total closure では runtime の \(e\to T\) root solve を行わず、事前構築した
     `total` の `P(\rho,e), T(\rho,e)` table を `rho_e_linear_grid=False` では
     \((\log\rho,\log e)\)、`rho_e_linear_grid=True` では \((\rho,e)\) の
     natural cubic spline で直接参照する。
   - 2T では total \(P(\rho,e_i+e_e)\) と \(c_s\) のみを `rho_e_table` から取得し、
     \(T_i,T_e,e_i,e_e,c_{v,i},c_{v,e}\) の個別 inverse map は legacy raw ion/electron table を維持する。
5. `helmholtz_spline` backend：
   - \(y=\ln T\) とし、
     \[
     g(y) = e_{spline}(\ln\rho, y) - e_{target}
     \]
     を解く。
   - 導関数は
     \[
     g'(y) = \frac{\partial e_{spline}}{\partial \ln T} = T\,c_v
     \]
     なので、table temperature nodes から bracket を見つけた上で safeguarded Newton を実行する。
   - Newton 提案値が bracket 外に出る、または \(g'\) が極小/非有限の場合は二分法へフォールバックする。
6. `helmholtz_jet` backend：
   - \(y=\ln T\) とし、
     \[
     g(y) = e_{jet}(\ln\rho, y) - e_{target}
     \]
     を table temperature nodes で bracket した後、safeguarded Newton で解く。
   - 導関数は
     \[
     g'(y) = T\,c_v
     \]
     であり、Newton が bracket 外へ出る、または \(g'\) が極小/非有限なら二分法に戻す。
6. **フロアクランプ**：最終的な \(T^{final}\) は \(T_{floor}\)（§1.1.7）以上にクランプする。

理想気体EOS（\(c_v = \text{const}\)）では反復なしで 1 回目で厳密に収束する。

#### 3.1.6 人工粘性（1D球対称）

`Numerics.hydro.av_type` で 1D_SPH の shock support 形式を選択する。
config 構造体の既定は `"vnr"` だが、namelist 経由の 1D_SPH deck では
`av_type` 未指定のとき builder finalize が `"csw"` を選択する（2026-08-03
Stage 1A 採択。2D_RZ は `av_model` 系で本キーの対象外。freeze 再現 deck は
常に `av_type` を明示出力するため既存 freeze deck の意味は変わらない）。
`av_type="vnr"` は von Neumann–Richtmyer 型（2D §3.2.9と同形式）を基底とするが、1D球対称では
Christensen型の速度リミタでセル端速度差を制限する。セル \(i\) はノード \(j=i\),
\(j+1=i+1\) に挟まれるものとし、各ノードで制限傾き \(\sigma_j\) を構成する：
\[
S_{L,j} = J\,\frac{u_j-u_{j-1}}{r_j-r_{j-1}}, \qquad
S_{R,j} = J\,\frac{u_{j+1}-u_j}{r_{j+1}-r_j}
\]
\[
S_{C,j} =
\frac{\frac{r_{j+1}-r_j}{r_j-r_{j-1}}(u_j-u_{j-1})
      +
      \frac{r_j-r_{j-1}}{r_{j+1}-r_j}(u_{j+1}-u_j)}
     {r_{j+1}-r_{j-1}}
\]
\[
\sigma_j =
\begin{cases}
\operatorname{sign}(S_{L,j})\,
\min\!\left(|S_{L,j}|,\;|S_{C,j}|,\;|S_{R,j}|\right) & S_{L,j}S_{R,j} > 0 \\
0 & S_{L,j}S_{R,j} \le 0
\end{cases}
\]
\[
u_{L,i}^{*} = u_j + \frac{\Delta r_i}{2}\sigma_j, \qquad
u_{R,i}^{*} = u_{j+1} - \frac{\Delta r_i}{2}\sigma_{j+1}
\]
\[
\chi_i = \max\!\left(0,\;-\frac{u_{R,i}^{*} - u_{L,i}^{*}}{\Delta r_i}\right)
\]
\[
Q_i =
\begin{cases}
Q_i^{shock} & W_{shock,i} > 0 \\
Q_i^{mild} & W_{shock,i} = 0
\end{cases}
\]
\[
Q_i^{shock} = \phi_i\,
\rho_i\left((C_2 B_i)^2\,(\Delta r_i)^2\,\chi_i^2 + (C_1 B_i)\,\Delta r_i\,c_{s,i}\,\chi_i\right)
\]
\[
\phi_i = W_{shock,i}\,\max\!\left(\phi_{floor},\;W_{comp,i}\,W_{osc,i}\right),
\qquad \phi_{floor}=0.25
\]
\[
W_{shock,i} = \max\!\left(w_{i-\frac{1}{2}},\;w_{i+\frac{1}{2}}\right)
\]
\[
w_{j+\frac{1}{2}} =
\begin{cases}
1 & \text{境界側に隣接セルが存在しないとき} \\
1 & \Delta P\,\Delta\rho > 0,\ J_p \ge 0.3,\ J_\rho \ge 0.05,\ Z \ge 0.5 \\
1 & J_p \ge 0.3,\ J_\rho < 0.05,\ Z \ge 0.5 \\
0 & \text{otherwise}
\end{cases}
\]
\[
J_p = \frac{|\Delta P|}{\min(P_L,\;P_R)}, \qquad
J_\rho = \frac{|\Delta\rho|}{\min(\rho_L,\;\rho_R)}
\]
\[
Z = \frac{|\Delta P|}
{0.5\,(c_{s,L}^2 + c_{s,R}^2)\,|\Delta\rho|}, \qquad
\Delta P = P_R - P_L,\ \Delta\rho = \rho_R - \rho_L,\qquad
P = P_e + P_i
\]
\[
M_{comp,i} = \frac{\Delta r_i\,\chi_i}{c_{s,i}}, \qquad
W_{comp,i} = \min\!\left(1,\;\max\!\left(0,\;\frac{M_{comp,i}}{0.05}\right)\right)
\]
\[
B_i =
\begin{cases}
\min\!\left(B_{max},\;\max\!\left(1,\;\dfrac{\Gamma_{1,ref}}{\Gamma_{1,i}}\right)\right)
& \texttt{av\_eos\_aware=True} \ \land\ \text{table EOS active} \\
1 & \text{otherwise}
\end{cases}
\qquad
\Gamma_{1,i} = \frac{\rho_i\,c_{s,i}^2}{P_{e,i}+P_{i,i}}
\]
\[
Q_i^{mild} = \phi_{mild,i}\,\rho_i\left((C_1 B_i)\,\Delta r_i\,c_{s,i}\,\chi_i\right)
\]
\[
\phi_{mild,i} = \alpha_{mild}\,W_{mild,i}\,W_{comp,i}\,W_{halo,i},
\qquad \alpha_{mild}=0.25
\]
\[
W_{mild,i} = \max\!\left(\widetilde{w}_{i-\frac{1}{2}},\;\widetilde{w}_{i+\frac{1}{2}}\right)
\]
\[
W_{halo,i} =
\begin{cases}
0 & \widetilde{w}^{shock}_{i-\frac{3}{2}} > 0 \ \text{or}\ \widetilde{w}^{shock}_{i+\frac{3}{2}} > 0 \\
1 & \text{otherwise}
\end{cases}
\]
\[
\widetilde{w}_{j+\frac{1}{2}} =
\begin{cases}
\min\!\left(1,\;\max\!\left(0,\;\dfrac{J_p}{0.3}\right)\right)
& \Delta P\,\Delta\rho > 0,\ Z \ge 0.5 \\
0 & \text{otherwise}
\end{cases}
\]
\[
\widetilde{w}^{shock}_{j+\frac{1}{2}} =
\begin{cases}
1 & \Delta P\,\Delta\rho > 0,\ J_p \ge 0.3,\ J_\rho \ge 0.05,\ Z \ge 0.5 \\
1 & J_p \ge 0.3,\ J_\rho < 0.05,\ Z \ge 0.5 \\
0 & \text{otherwise}
\end{cases}
\]
\[
\mathrm{osc}_i =
\begin{cases}
\dfrac{\min(|\Delta P_{L,i}|,\ |\Delta P_{R,i}|)}
      {\max(|\Delta P_{L,i}|,\ |\Delta P_{R,i}|)}
& \Delta P_{L,i}\Delta P_{R,i} < 0 \\
0 & \text{otherwise}
\end{cases}
\]
\[
\Delta P_{L,i} = P_i - P_{i-1}, \qquad
\Delta P_{R,i} = P_{i+1} - P_i
\]
\[
W_{osc,i} =
\begin{cases}
0 & \mathrm{osc}_i > 0.2 \ \text{かつ両側interfaceが developed shock ではない} \\
1 & \text{otherwise}
\end{cases}
\]

- 特性セル長：\(\Delta r_i = r_{j+1} - r_j\) [cm]（添字対応 \(j=i\)）
- 音速 \(c_{s,i}\) [cm/s]：§1.1.6
- 既定：\(C_1=0.1,\; C_2=1.5,\; J=1.0\)（SPECIFICATION §6.4.7準拠）
- `av_eos_aware=True` かつ table EOS が active なときは、AV 内部でのみ
  \(B_i\) を用いて局所 \(\Gamma_1\) 低下に応じた追加 damping を掛ける。
  ideal-gas path では \(B_i=1\) に退化する。
- エントロピーゲートとして \((\nabla\cdot\mathbf{u})_i < 0\) は維持する。速度リミタは
  粘性の大きさ \(\chi_i\) を与え、その後に pressure-density jump, 圧縮Mach数,
  odd-even 抑制から成る強shock重み \(\phi_i\) を掛ける。
- `W_shock=0` であっても、\(\Delta P\,\Delta\rho > 0\) かつ \(Z \ge 0.5\) を満たす
  Rankine-Hugoniot整合的な弱い圧縮波では、soft pressure-jump ramp
  \(W_{mild}\) により線形項のみの damping \(Q_i^{mild}\) を与える。
- ただし、mild branch は強shock の 1 セルhalo では無効化する。すなわち、外側隣接interface
  \(i-\frac{3}{2}\) または \(i+\frac{3}{2}\) に shock-support が立つセルでは
  \(W_{halo,i}=0\) とし、pre-/post-shock の前方散逸を避ける。
- 弱圧縮波 branch では quadratic 項を使わず、odd-even 抑制重み \(W_{osc}\) も
  掛けない。目的は checkerboard 検出時に damping を殺すことではなく、
  laser-driven implosion の圧縮域で弱い振動を減衰させることにある。
- 実装定数（`src/hydro/artificial_viscosity.cu`）：
  `kShockPressureJumpThreshold = 0.3`,
  `kShockDensityJumpThreshold = 0.05`,
  `kShockRhConsistencyThreshold = 0.5`,
  `kCompMachScale = 0.05`,
  `kOscillationThreshold = 0.2`,
  `kShockSupportFloor = 0.25`,
  `kMildCompressionAlpha = 0.25`,
  `kSensorEps = 1.0e-30`
- \(J_\rho < 0.05\) の precursor branch は、Sedov型 blast launch の初期に見られる
  「左セル rarefaction / 右セル compression」で \(\Delta P\Delta\rho < 0\) となる段階でも、
  圧力駆動 shock-support を残すために符号条件を課さない。
- \(\phi_{floor}\) は `W_shock=1` のセルで AV を完全に殺さないための safety floor であり、
  Sedov/Noh のような強 shock 問題で post-shock oscillation によるセル圧潰を防ぐ。
- pressure-density jump を導入する物理的動機は、GXII shell のような
  source-heated front（高温だが \(\Delta P\) と \(\Delta \rho\) が shock 的に整合しない前線）と、
  実 shock を分離することにある。
- 弱圧縮波 branch は hard threshold \(J_p \ge 0.3\) に届かない laser-driven compression wave
  を対象とし、source-heated front と区別するために \(\Delta P\,\Delta\rho > 0\) と
  \(Z \ge 0.5\) の整合条件は維持する。
- 閾値は `verify_gxii_1d_regression` で source-heated front を suppress しつつ、
  Sedov blast の pressure-dominated launch と Noh/Sedov の shock-support で
  `W_shock=1` を維持するよう固定している。
- 中心ノードでは対称ゴースト \(r_{-1}=-r_1,\;u_{-1}=-u_1\) を用いる。
  外側境界ノードでは線形外挿ゴースト
  \(r_{N+1}=2r_N-r_{N-1},\;u_{N+1}=2u_N-u_{N-1}\) を用いる。
- 一様または相似圧縮のような滑らかな流れでは \(\chi_i \to 0\) となり、真の速度不連続でのみ
  人工粘性が活性化する。

> 2次項の係数が \(C_2^2\) であることに注意（§3.2.9と同じ規約）。

`av_type="riemann"` では、セル境界 \(f=i+1/2\) で局所 Lagrangian Riemann
pressure correction を評価し、cell-centered \(Q_i\) へ平均する。VNR 専用の
Christensen limiter、shock-support gate、mild-compression branch、
`av_C1`、`av_C2`、`av_limiter_J`、`av_eos_aware` は使用しない。

セル速度状態は v1 では midpoint 量から構成する：
\[
\bar r_i = \frac{r_i+r_{i+1}}{2}, \qquad
\bar u_i = \frac{u_i+u_{i+1}}{2}.
\]
active な近傍セルの \((\bar r,\bar u)\) から minmod slope \(s_i\) を作る。
両側近傍がある場合は
\[
s_i=\operatorname{minmod}\left(
\frac{\bar u_i-\bar u_{i-1}}{\bar r_i-\bar r_{i-1}},
\frac{\bar u_{i+1}-\bar u_i}{\bar r_{i+1}-\bar r_i}
\right),
\]
片側だけがある境界近傍ではその one-sided slope、近傍がない場合は \(s_i=0\) とする。
interior face では
\[
u_L^f = \bar u_i+s_i(r_{i+1}-\bar r_i),\qquad
u_R^f = \bar u_{i+1}+s_{i+1}(r_{i+1}-\bar r_{i+1}),
\]
\[
\Delta u_f^+=\max(0,\;u_L^f-u_R^f).
\]
smooth linear velocity field では \(u_L^f=u_R^f\) となるため \(Q_f=0\) である。
中心 face は cell 0 の mirror state を使い、外側 face は v1 では \(Q_{N+1/2}=0\)
とする。

pressure と nonlinear impedance は
\[
P_k=P_{e,k}+P_{i,k},\qquad
\Gamma_{1,k}=\frac{\rho_k c_{s,k}^2}{\max(P_k,\epsilon)},
\]
\[
\alpha_k=\frac{\Gamma_{1,k}+1}{4},\qquad
Z_k^{eff}=\rho_k\left(c_{s,k}+\alpha_k\Delta u_f^+\right)
\]
で与える。非有限または非正の \(\Gamma_1\) では ideal-gas fallback
\(\alpha=2/3\) を用いる。Riemann pressure は
\[
P^* =
\frac{Z_R^{eff}P_L+Z_L^{eff}P_R+Z_L^{eff}Z_R^{eff}(u_L^f-u_R^f)}
{Z_L^{eff}+Z_R^{eff}}
\]
であり、face viscosity は
\[
Q_f =
\begin{cases}
\max\left(0,\;P^*-\frac{P_L+P_R}{2}\right) & \Delta u_f^+ > 0\\
0 & \Delta u_f^+ = 0
\end{cases}
\]
とする。cell-centered viscosity は
\[
Q_i=\frac{1}{2}\left(Q_{i-1/2}+Q_{i+1/2}\right)
\]
で既存の momentum / energy 経路へ渡す。

`av_type="csw"` では、Caramana-Shashkov-Whalen 型の monotone artificial
viscosity を 1D_SPH に限定して用いる。VNR の
`av_C1`/`av_C2`（alias: `av_linear`/`av_quadratic`）、
`av_limiter_J`、`av_eos_aware` は使用せず、専用係数
`csw_C1`, `csw_C2` を用いる。セル \(i\) で
\[
\Delta r_i = r_{i+1}-r_i,\qquad
\chi_i^{raw}=\max\left(0,\;-\frac{u_{i+1}-u_i}{\Delta r_i}\right)
\]
をまず評価する。ノード \(j\) の左右傾きは
\[
d_j^-=\frac{u_j-u_{j-1}}{r_j-r_{j-1}},\qquad
d_j^+=\frac{u_{j+1}-u_j}{r_{j+1}-r_j}
\]
であり、van Leer limiter では
\[
\phi(r)=\frac{r+|r|}{1+|r|},\qquad r=\frac{d_j^-}{d_j^+},\qquad
s_j=
\begin{cases}
\phi(r)d_j^+ & d_j^-d_j^+>0\\
0 & d_j^-d_j^+\le 0
\end{cases}
\]
とする。`csw_limiter="bj"` では中心傾き \((d_j^-+d_j^+)/2\) を
Barth-Jespersen の局所 extrema bound で縮小する。境界ノードでは one-sided slope を用いる。

セル端へ extrapolate した速度
\[
u_{L,i}^* = u_i+\frac{1}{2}\Delta r_i s_i,\qquad
u_{R,i}^* = u_{i+1}-\frac{1}{2}\Delta r_i s_{i+1}
\]
から
\[
\chi_i^{rec}=\max\left(0,\;-\frac{u_{R,i}^*-u_{L,i}^*}{\Delta r_i}\right),
\qquad
\chi_i^{lim}=\min(\chi_i^{raw},\chi_i^{rec})
\]
を作る。`csw_shock_limiter_floor=f` は \(\chi_i^{rec}>0\) のときだけ
\[
\chi_i^{lim}\leftarrow
\min\left(\chi_i^{raw},\;\max(\chi_i^{lim},\;f\chi_i^{raw})\right)
\]
を適用し、滑らかな一様圧縮（\(\chi_i^{rec}=0\)）では \(Q_i=0\) を保つ。
最終的な粘性圧は
\[
Q_i=\rho_i\left(C_2^2\Delta r_i^2(\chi_i^{lim})^2+
                 C_1\Delta r_i c_{s,i}\chi_i^{lim}\right),
\quad C_1=\texttt{csw\_C1},\ C_2=\texttt{csw\_C2}.
\]

CSW の既定値は `csw_C1=0.5`, `csw_C2=2.0`,
`csw_limiter="van_leer"`, `csw_shock_limiter_floor=0.65`,
`csw_zero_uniform_compression=True` である。VNR 既定
\(C_1=0.1,\ C_2=1.5\) は変更しない。単調 tail を \(N=3\) cell、
圧縮Mach \(M_s=2\)、floor \(f=0.65\) と見積もると、線形粘性相当係数は
\[
C_{1,eq}\simeq fC_1 + f^2 C_2^2\frac{M_s}{N}
\simeq 0.65(0.5)+0.65^2(2.0)^2\frac{2}{3}\simeq 1.45
\]
となり、単調 tail 目安 \(0.228NM_s\simeq1.37\) をわずかに上回る。
CSW は shock profile 内の過不足な AV を抑えるための置換である。
`adaptive_av.enabled=True` は VNR legacy 専用であり、`av_type="csw"` との
同時指定は `ConfigError` とする。

`av_type="riemann_compatible"`（Stage 1C v1.1、2026-08-03、opt-in）は
Morgan et al. (2014) の staggered Godunov-like corner 構成を 1D の
cell-centered \(Q\) slot へ縮約した形式である。`riemann` と同じく VNR
専用機構（Christensen limiter、shock-support gate、mild branch、
`av_C1`/`av_C2`、`av_limiter_J`、`av_eos_aware`）も `csw_C1`/`csw_C2` も
使用しない（係数レス — \(c_s\) と \(b_1\) のみ）。構成（実装は
`compute_q_riemann_compatible_1d_kernel`）:

1. **nodal→cell-center 射影**: セル中心 \(x^c_i=(r_i+r_{i+1})/2\)、セル
   平均速度 \(\bar u_i=(u_i+u_{i+1})/2\)。ノード \(j\) の勾配は隣接セル
   平均の secant
   \(g_j=(\bar u_j-\bar u_{j-1})/(x^c_j-x^c_{j-1})\)
   （内側境界は odd-velocity 反射状態で
   \(g_0=(\bar u_0-u_0)/(x^c_0-r_0)\)、外側境界は最後の 2 セル平均の
   one-sided secant）。
2. **1D Barth–Jespersen limiter**（\(\beta_{BJ}=0.5\)、ノードごとに単一
   \(\phi_j=\min_i\Phi_{j\to i}\)、extrema はノード速度
   \(\{u_{j-1},u_j,u_{j+1}\}\)、内側境界は反射値 \(2u_j-u_{j+1}\) を追加）。
3. **制限付き corner 状態と full 射影 jump**:
   \[
   u^{c}_{L,i}=u_i+\phi_i\,(x^c_i-r_i)\,g_i,\qquad
   u^{c}_{R,i}=u_{i+1}+\phi_{i+1}\,(x^c_i-r_{i+1})\,g_{i+1},
   \]
   \[
   \Delta u^c_i=\max(0,\ u^{c}_{L,i}-u^{c}_{R,i}).
   \]
4. **体積圧縮ゲート**: \(\dot V_i = A_{i+1}u_{i+1}-A_i u_i\)（\(A\) は
   `geometry_1d` に応じた面積 — compute_Q_1d の geom_code）。
   \(\Delta u^c_i=0\) または \(\dot V_i\ge 0\) なら \(Q_i=0\) —
   compatible work \(-Q_i\dot V_i\ge 0\) が構成的に保証される。
5. **symmetric full-projected-jump impedance**（相談 20260803-1922 §3.1
   で追認。両 kinematic impedance の算術平均と等価 —
   \(u^c_R\le\bar u\le u^c_L\) のとき \((\mu_L+\mu_R)/2=\mu\)）:
   \[
   \mu_i=\rho_i\left(c_{s,i}+\tfrac{1}{2}b_{1,i}\,\Delta u^c_i\right),\qquad
   b_{1,i}=\frac{\Gamma_{1,i}+1}{2},\qquad
   Q_i=\frac{\mu_i}{2}\,\Delta u^c_i
   \]
   （ideal-gas fallback \(b_1=4/3\)。一次極限で Morgan の
   \(Q=\tfrac12\rho c_s\Delta U+\tfrac{b_1}{4}\rho\Delta U^2\) に一致）。
   \(c_{s,i}=0\) の冷たいセルは**有効**（guard は \(c_s\ge 0\) のみ —
   二次項は音速を要しない。v1 の \(c_s>0\) guard は冷気体の粘性を全滅させ
   Noh 壁セルが \(t=\Delta r/|u|\) で自由落下閉塞する実測欠陥だった）。
   同じく v1.1 で退けた形: (a) v1 の midpoint-minmod 再構成は
   \(\Delta u_L\equiv\Delta u_R\)（corner 分割が幻影）かつ二次係数が
   Morgan の半分、(b) 片側 \(|u^c-\bar u|\) インピーダンスは制限射影が
   セル平均に一致すると \(\mu\to\rho c_s\to 0\) に退化する。
速度差のみを用いるため Galilean 不変、一様・線形速度場では
\(Q_i=0\)。既存の momentum / PdV compatible 経路（P+Q pairing）は不変。
1D_SPH 専用（2D_RZ では `ConfigError`）。検証: 単体テストに一次厳密値
（step 場で \(Q=7/3\)（\(c_s=1\)）/ \(4/3\)（\(c_s=0\)））を含む。

**昇格ステータス（2026-08-04、相談 20260803-1922 裁定）**: qualified
opt-in（1D 生産既定は csw のまま）。Phase-2 全ゲート: Noh front 有界・
外側境界正常・ledger 機械精度・決定論 bit 同一（263 frame 物理データ
セット/run 反復/cross-build csw 不変）・laser ladder 獲得維持（ripple
全レベル優位: 1.63/1.42/1.09% vs csw 2.32/1.97/1.43%、\(\rho_{peak}\)
メッシュ感度勾配 ≈ csw の 55%、吸収/バング パリティ）・12-run HELIOS
バッテリ全完走（abort 0）。**bounded-defect waiver**: 2T Noh 壁 1–2
セル startup 双極子（第1セル +37.5% / 第2セル −19.75% vs csw 第1セル
−33.75%）と post-shock \(L_1\) 比 ~2.35（2T）は明示 waiver として記録
— 壁局所 \(\phi=0\) 等のパッチは裁定により禁止（secular 成長なし・
1–2 セル限定・修復前に §2.5 の startup work audit が必須）。csw→rc
default flip には Noh 収束 ladder（\(I_{wall}=O(\Delta x^p)\)）・smooth
-flow 不変性（Kidder 等）・広い衝撃族・実 EOS/冷状態網羅（\(b_1=4/3\)
fallback の ideal-gas 明示限定化を含む）等の追加バーが定義済み
（response.md §1.3）。

同じ shock sensor \(\chi_i\) を用いて、面 \(j=i\)（セル \(i-1\) と \(i\) の境界）に
人工熱流束を構成する：
\[
H_j = -C_H\,\rho_j\,l_j^2\,\chi_{H,j}\,
\frac{e_i - e_{i-1}}{r_{c,i} - r_{c,i-1}}
\]
\[
\rho_j = \frac{\rho_{i-1} + \rho_i}{2}, \qquad
l_j = \frac{\Delta r_{i-1} + \Delta r_i}{2}, \qquad
\chi_{H,j} = \max(\chi_{i-1}, \chi_i)
\]
\[
(\nabla\cdot\mathbf{H})_i =
\frac{1}{V_i}\left[A_{j+1}H_{j+1} - A_j H_j\right], \qquad
A_j = 4\pi r_j^2
\]
\[
\Delta e_{H,i} =
-\frac{\left(A_{j+1}H_{j+1} - A_j H_j\right)\Delta t}{\Delta M_i}
\]

- セル中心半径：\(r_{c,i} = (r_j + r_{j+1})/2\)
- 境界条件：\(H_0 = H_N = 0\)。`hydro_active` の非活性界面でも \(H=0\) とする
- 既定：\(C_H = 0.0\)。`av_heat_C = 0` で人工熱流束を無効化する
- **AV type 別の \(\chi\) と有効性（2026-07-26 明確化、AI review k03 F-13）**：
  \(\chi_i\) は有効な AV type 自身の limited compression を用いる —
  `vnr` は Christensen limiter 後の \(\chi_i\)、`csw` は §3.1.6 CSW の
  \(\chi^{lim}_i\)。`av_type="riemann"` および `"riemann_compatible"` では
  \(H\) は**強制無効**
  （実装は `av_heat_C` を 0 に上書き — cell-centered \(\chi\) が定義されない
  ため）。宛先は 1T が \(e_{tot}\)、2T は source/target とも \(e_i\)（ion）
  固定 — `av_heat_to` は AV 圧力 work \(Q\,dV\) の宛先のみを選び、\(H\) には
  作用しない。post-shock heat の \(P_e/P_{tot}\) 分配とも別規約
- 実装では \(e=e_i\)（2T）または \(e=e_{tot}\)（1T）を用い、PdV/Q 更新の後に
  保存形のセル熱流束寄与 \(\Delta e_H\) を加算する

`ion_art_heat_C = C_{Hi} > 0` のときは、compatible-energy が除去する数値散逸を
補うための ion-only artificial heat conduction を別項として追加する。これは
1D_SPH + 2T + VNR AV 専用であり、電子エネルギー \(e_e\) には沈着しない。
面 \(j=i\)（セル \(L=i-1\), \(R=i\) の境界）における face power は、正値を
左から右への熱流として
\[
P^{ion}_j =
-A_j\,\kappa^{ion}_j\,\frac{T_{i,R}-T_{i,L}}{\Delta n_j}
\]
と定義する。したがって保存形のセル更新は
\[
H^{ion}_i = P^{ion}_{i} - P^{ion}_{i+1},\qquad
\Delta e^{ion}_{i}=\frac{\Delta t\,H^{ion}_i}{\Delta M_i}
\]
である。この符号規約では \(T_{i,R}>T_{i,L}\) のとき \(P^{ion}_j<0\) となり、
左セルが加熱、右セルが冷却される。

face 幾何と平均量は
\[
\Delta n_j=\frac{\Delta r_L+\Delta r_R}{2},\quad
\rho_j=\frac{\rho_L+\rho_R}{2},\quad
c_{s,j}=\frac{c_{s,L}+c_{s,R}}{2},
\]
\[
c_{s,j}^2=\frac{c_{s,L}^2+c_{s,R}^2}{2},\quad
c_{v,i,j}=\frac{c_{v,i,L}+c_{v,i,R}}{2}
\]
である。**face 音速の 2 定義の使い分け（2026-07-26 明確化、AI review k01
§4.5）**: 算術平均 \(c_{s,j}\) は compression Mach sensor
\(M_c = \Delta n_j\theta_j/c_{s,j}\) にのみ、二乗平均 \(c_{s,j}^2\) は
contact sensor の分母 \(c_{s,j}^2|\Delta\rho|\) にのみ使う（それぞれ速度
スケール・音響インピーダンススケールとしての用途別であり、同一式内で
混用しない）。EOS が `state.cv_i` を持たない ideal-gas path では解析的なイオン
\(c_v\) を用いる。**係数の名称について（2026-07-26 明確化、AI review k03
F-14）**: TENRYU の EOS state は \(c_p\) を保持せず、本演算子の熱容量は
一貫して mass-specific \(c_{v,i}\) [erg g\(^{-1}\) eV\(^{-1}\)] である。
expert specification 由来の記号 \(c_p\) は本仕様では使用しない（\(c_p\) を
入力とする式に \(c_v\) を代入しているのではなく、演算子定義そのものが
\(c_v\) 基底 — ideal gas なら \(\gamma\) 倍の差があるため、この区別は
係数校正に効く）。

VNR AV kernel からは raw spherical divergence と quadratic AV 成分を出力する：
\[
(\nabla\cdot u)_i =
\frac{A_{i+1/2}u_{i+1/2}-A_{i-1/2}u_{i-1/2}}{V_i},
\]
\[
q^{(2)}_i =
\rho_i\,(C_{2,eff})^2\,\Delta r_i^2\,\chi_i^2.
\]
ここで \(\chi_i\) は Christensen limiter 後の圧縮であり、\(C_{2,eff}\) は
`av_eos_aware` が有効な場合の boost を含む。すなわち \(q^{(2)}\) は
raw \(-\nabla\cdot u\) ではなく、実際に AV 圧力へ入る limited compression から
定義する。一方、下記の compression sensor は raw \(\nabla\cdot u\) を使う。

face 値は
\[
q^{(2)}_j=\max(q^{(2)}_L,q^{(2)}_R),\qquad
\theta_j=\max\left(\max(0,-\nabla\cdot u_L),
                  \max(0,-\nabla\cdot u_R)\right)
\]
とする。センサは
\[
M_{c,j}=\frac{\Delta n_j\theta_j}{\max(c_{s,j},\epsilon)},\qquad
S_{comp,j}=
\mathrm{clamp}\left(\frac{M_{c,j}-0.05}{0.10},0,1\right)
\]
\[
p_{th}=P_e+P_i,
\qquad
S_{contact,j} =
\frac{|p_{th,R}-p_{th,L}|}
{|p_{th,R}-p_{th,L}| + c_{s,j}^2|\rho_R-\rho_L| + p_{floor}}
\]
で与える。contact sensor では人工粘性 \(Q\) を pressure に含めない。

人工拡散係数は
\[
\beta_j=\frac{q^{(2)}_j}{\max(\theta_j,\epsilon)},\qquad
\kappa^{raw}_j=C_{Hi}S_{comp,j}S_{contact,j}\beta_j c_{v,i,j}
\]
とし、明示安定性 cap を**ペア熱容量コンダクタンス束縛**
\[
C_{L/R} = \Delta M_{L/R}\,c_{v,i,L/R},\qquad
\kappa^{cap}_j =
0.25\,\frac{\Delta n_j}{A_j\,\Delta t}\,
\frac{C_L C_R}{C_L + C_R}
\]
で
\[
\kappa^{ion}_j=\min(\kappa^{raw}_j,\kappa^{cap}_j)
\]
とする（**2026-07-26 是正、AI review k01 §4.4 / k03 F-14**: 旧 slab 型 cap
\(0.25\rho_j c_{v,i,j}\Delta n_j^2/\Delta t\) は面面積と軽い側セルの熱容量を
無視しており、graded/球面メッシュで軽セル側の明示安定限界を超え得た。
新 cap は face 交換の陽的更新
\(\Delta t\,(A_j\kappa/\Delta n_j)(1/C_L + 1/C_R) \le 0.25\) を厳密に保証する。
一様平面格子では旧 cap の 1/2 に相当）。単位は
\(\kappa^{ion}\) [erg cm\(^{-1}\) s\(^{-1}\) eV\(^{-1}\)]、
\(P^{ion}\) [erg s\(^{-1}\)] である。境界 face は v1 ではゼロ熱流とし、
セル 0 は右 face のみ、セル \(N-1\) は左 face のみを用いる。
`hydro_active` 非活性 face でも \(P^{ion}=0\) とする。

`post_shock_heat=True` または `post_shock_velocity_damping_C > 0` のときは、
shock history 時刻 \(t^{shock}_i\) を保持する。Predictor 後の AV 評価で
\[
Q^{max}_{1/2} = \max_{k \in \mathcal{A}} Q_k^{n+1/2}
\]
\[
t^{shock}_i \leftarrow t^{n+1/2}
\qquad \text{if} \qquad
Q_i^{n+1/2} > 10^{-2} Q^{max}_{1/2}
\]
と更新する。ここで \(\mathcal{A}\) は active セル集合である。flux 評価時の
cell sensor は
\[
\tau_i = C_{decay}\,\frac{\Delta r_i}{\max(c_{s,i},\varepsilon_{ps})},
\qquad
\psi_i = \exp\!\left(-\frac{t^{n+1/2} - t^{shock}_i}{\tau_i}\right),
\qquad
\varepsilon_{ps}=10^{-30}
\]
とする。`post_shock_heat=True` のときは、\(\chi\)-sensor の \(\mathbf{H}\) とは独立に、
recently-shocked セルだけへ局所人工熱流束 \(\mathbf{H}^{ps}\) を加える。面 \(j=i\)
（セル \(i-1\) と \(i\) の境界）では
\[
\psi_j = \frac{\psi_{i-1} + \psi_i}{2},
\qquad
\rho_j = \frac{\rho_{i-1} + \rho_i}{2},
\qquad
c_{s,j} = \frac{c_{s,i-1} + c_{s,i}}{2}
\]
\[
H^{ps}_j = -C_{ps}\,\rho_j\,c_{s,j}\,l_j\,\psi_j\,
\frac{e_{tot,i} - e_{tot,i-1}}{r_{c,i} - r_{c,i-1}},
\qquad
l_j = \frac{\Delta r_{i-1} + \Delta r_i}{2}
\]
と定義する。保存形のセル更新は
\[
\Delta e^{ps}_i =
-\frac{\left(A_{j+1}H^{ps}_{j+1} - A_j H^{ps}_j\right)\Delta t}{\Delta M_i}
\]
で与える。

- 既定：`post_shock_heat=False`, `post_shock_heat_C = 0.1`,
  `post_shock_heat_decay = 3.0`, `post_shock_velocity_damping_C = 0.0`
- `post_shock_heat=False` または \(C_{ps}=0\) で \(\mathbf{H}^{ps}=0\)
- `post_shock_velocity_damping_C > 0` のときは、§3.1.4 の
  `odd_even` nodal damping が \(\psi_i^{eff}\) を介して recently-shocked セルへも作用する
- \(t^{shock}_i \ll t\) のセルでは \(\psi_i \to 0\) となり、自動的に無効化される
- v1.0 の 1D_SPH は単一材料前提なので、material interface cut-off は不要

さらに `bulk_viscosity_C = C_{bulk} > 0` のとき、圧縮セル
（\((u_{j+1}-u_j)/\Delta r_i < 0\)）に限り cell-centered viscous pressure に
\[
Q_i \leftarrow Q_i + Q_i^{bulk}
\]
\[
Q_i^{bulk} =
C_{bulk}\,\rho_i\,c_{s,i}\,
\left|\frac{u_{j+1}-u_j}{\max(\Delta r_i,\varepsilon_{bulk})}\right|\,
\Delta r_i,
\qquad
\varepsilon_{bulk}=10^{-30}\ \mathrm{cm}
\]
を加える。ここで \(j=i\)、\(\Delta r_i=r_{j+1}-r_j\) である。

- **圧縮限定（compression-only）**：膨張セル（\(du \ge 0\)）では \(Q_i^{bulk}=0\)。
  膨張中に正の scalar pressure を残すと粘性仕事 \(-Q\,dV/dt\) が負になり、
  内部エネルギーを運動エネルギーへ戻す反散逸（エントロピー減少）となるため
  （Caramana–Shashkov–Whalen 1998 の「人工粘性は膨張でゼロ」要件；
  AI review 2026-07-26 k01 P0-5 / k03 F-03 で是正。旧仕様は圧縮・膨張の
  両方で作用としていたが、これは熱力学的に誤りであった）
- 適用対象は active な圧縮セル全体であり、shock front では通常の AV が支配的、post-shock 圧縮域では \(Q_i^{bulk}\) が残留速度振動を減衰する
- 既定：`bulk_viscosity_C = 0.0`（無効）
- CFL は §3.1.9 の既存 acoustic+AV 推定をそのまま用い、`bulk_viscosity_C` に対する追加制約は導入しない

##### Adaptive Sensor-Gated Artificial Viscosity

`Numerics.hydro.adaptive_av.enabled=True` のとき、1D_SPH + `av_type="vnr"` に限り、
leading compressive shock に追従する局所 gate \(g_i\in[0,1]\) で
\((C_1,C_2,C_{heat},C_{psv},C_{bulk})\) をセルごとに補間する。既定は無効であり、
無効時は従来の scalar 係数だけを用いる。`av_type!="vnr"` では
`ConfigError` とする。

各 hydro step の先頭で base VNR 係数による probe \(Q_i^{probe}\) と
\(\nabla\cdot u_i\) を作り、以下を満たす contiguous cluster を shock 候補とする：

\[
\nabla\cdot u_i < 0,\qquad
Q_i^{probe} > 0.01\max_k Q_k^{probe},\qquad
J_P > 0.02 \;\lor\; J_\rho > 0.01
\]

（**2026-07-26 記法是正、AI review k03 F-05**: 旧記法
\(\max(J_P,J_\rho) > (0.02,0.01)\) はスカラーと順序対の比較で数学的に
未定義だった。実装は上記の論理和である）

leading cluster は \(\sum_i Q_i^{probe}\) 最大で選ぶ。5% 以内の tie では、
bounce 前は小さい \(r_s\)、bounce 後は大きい \(r_s\) を選ぶ。cluster 中心は
\[
r_s = \frac{\sum_i Q_i^{probe} r_i}{\sum_i Q_i^{probe}},\qquad
U_s =
\begin{cases}
(r_s-r_s^{old})/\Delta t & \text{前回 tracker が有効なとき}\\
\sum_i Q_i^{probe} u_i/\sum_i Q_i^{probe} & \text{初回}
\end{cases}
\]
で求める。初期半径 \(R_0\) は最初の hydro 呼び出しで
`hydro_active && !cell_is_void` の最外 node 半径を latch し、fallback は mesh outer node とする。

mode は次で決める：

\[
\begin{array}{ll}
!bounce \land U_s<0 \land r_s>0.25R_0 & PRIMARY\_FULL\\
!bounce \land U_s<0 \land 0.05R_0<r_s\le0.25R_0 & PRIMARY\_TAPER\\
bounce \land U_s>0 \land \langle\nabla\cdot u\rangle_{shell}>0 & REBOUND\\
\text{otherwise} & BASE
\end{array}
\]

`bounce_seen` は tracker が 50 valid steps 以上 warm-up した後に限り latch する。
条件は \(r_s<0.02R_0\)、または tracker 履歴の最小半径 \(r_{s,\min}<0.3R_0\) まで
深く圧縮した後に \(r_s>1.2r_{s,\min}\) かつ \(U_s>1\,\mathrm{km/s}\) となる
genuine rebound である。valid でない状態から最初に valid へ入った step は
bounce 判定に使わない。PRIMARY_TAPER では \(r_s/R_0=0.25\to0.05\) に沿って
PRIMARY_FULL から BASE へ smoothstep で係数を戻す。

target gate は shock 中心から見た
\(\xi_i=(r_i-r_s)/\Delta r_s\) に対し、shock 進行方向の ahead 側
`support_ahead` cell、behind 側 `support_behind` cell の非対称 smooth hat とする。
既定は `support_ahead=1`, `support_behind=10` で、primary shock 通過直後の
post-shock tail を広めに覆う。
履歴 gate は
\[
g_i^{n+1} = (1-w)g_i^n + w\,g_i^{target}
\]
で更新する。blend 重みは既定で \(w=\)`hysteresis_w`（step 毎固定）だが、
`hysteresis_tau > 0` [s] のときは物理時定数形
\[
w(\Delta t) = 1 - \exp(-\Delta t/\tau_g),\qquad \tau_g = \texttt{hysteresis\_tau}
\]
を用いる（**2026-07-26 追加、AI review k01 §8.1 / k03 F-05**: 固定 \(w\) は
gate 緩和率が step 数依存になり \(\Delta t\) refinement で AV 履歴が収束
しない。\(\tau_g\) 形は同一物理時間で同一の緩和を与える。既定
`hysteresis_tau=0` は legacy 固定 \(w\) を bitwise 保存）。係数は
\[
C_{1,i}=C_{1,base}+g_i(C_{1,mode}-C_{1,base})
\]
とし、\(C_2,C_{heat},C_{psv},C_{bulk}\) も同様に補間する。`Cpsv` は Stage 1
では既存の post-shock nodal damping へ per-cell \(C_{psv,i}\) として渡す。
`Cbulk` は adaptive path では compression-only に限定する。`compatible_energy=True`
で adaptive \(C_{psv}\) が有効な場合は、既存の global post-shock velocity damping と同様に
compatible energy update を無効化する。
Hydro CFL は adaptive AV が有効な場合、`base/primary/rebound` の最大 \(C_1,C_2\) を
scalar AV 係数と比較した上限値で評価する。

#### 3.1.7 1D熱伝導の離散化

球対称の電子熱伝導（§4.1の1D版）：
\[
(\nabla\cdot\mathbf{q}_e)_i = \frac{1}{V_i}\left[A_{j+1}\,q_{j+1} - A_j\,q_j\right]
\]
\[
q_j = -\kappa_{eff,j}\,\frac{T_{e,i} - T_{e,i-1}}{r_{c,i} - r_{c,i-1}}
\]
ここで \(r_{c,i} = (r_j + r_{j+1})/2\)（セル中心半径）、
\(\kappa_{eff,j}\) は隣接2セルの伝導率の調和平均にflux limiterを適用したもの（§4.1）。

**具体的な評価順序**：(1) 各セル \(i\) の Spitzer 伝導率 \(\kappa_{SH,i}\) を計算、(2) 面 \(j\) での調和平均 \(\tilde{\kappa}_j = 2\kappa_{SH,L}\kappa_{SH,R}/(\kappa_{SH,L}+\kappa_{SH,R})\)、(3) 面 \(j\) でのフラックスリミタ適用 \(\kappa_{eff,j} = \tilde{\kappa}_j / (1 + |\tilde{\kappa}_j \nabla T_j| / q_{max,j})\)。すなわち「先に調和平均、後にフラックスリミタ」の順序とする。\(q_{max,j}\) は面両側の物理量の算術平均（\(n_{e,j} = (n_{e,L}+n_{e,R})/2\)、\(T_{e,j} = (T_{e,L}+T_{e,R})/2\)）で評価する。

#### 3.1.8 幾何項 \(\mathbf{f}_{geom}\)

§1.1.2の幾何力は球座標の人工項ではなく、離散運動量方程式そのものに
自然に組み込まれている。実装される節点方程式は
\[
M_j\,\frac{du_j}{dt} = -A_j\,(p_{q,i} - p_{q,i-1}),\qquad A_j = 4\pi r_j^2
\]
であり、**一様圧力場では \(p_{q,i} = p_{q,i-1}\) から全内部節点で
\(du_j/dt = 0\)**（uniform-pressure null）。これは連続極限
\(-\rho^{-1}\partial P/\partial r = 0\) と整合する。

> **2026-07-26 スペック是正（AI review k01 §2.12）**: 旧記載は
> \(A_{j+1}P - A_jP = 4\pi P(r_{j+1}^2 - r_j^2) \ne 0\) を「球殻の幾何加速」
> として掲げていたが、これは誤りである。この面積差項は球殻表面の radial
> スカラー射影だけを抜き出したもので、側面（横方向）圧力の radial 成分と
> 厳密に相殺する。節点力を「面積×圧力の差」\(A_{j+1}P_{i} - A_jP_{i-1}\)
> 型で実装したり、面積差を追加の幾何力として加えると二重計上になる。
> 実装式（面積×圧力**差**）が正であり、幾何効果は \(A_j\) が半径に依存する
> ことと殻体積式 \(V(r_L,r_R)\) を通じてのみ入る。

#### 3.1.9 CFL条件（1D）

\[
\Delta t_{acoustic+AV} = C_{CFL}\cdot\min_{i \in \mathcal{A}}\!\left(
\frac{\Delta r_i}{c_{s,i} + C_1\,c_{s,i} + C_2\,\Delta r_i\,\chi_i}
\right)
\]
- \(\mathcal{A} = \{i \mid \text{hydro\_active}_i = \text{true}\}\)：活性セル集合（§2.1.1）。\(\mathcal{A} = \emptyset\) のとき \(\Delta t_{hydro} = \infty\)。
- \(C_1,\; C_2\)：人工粘性係数（§3.1.6）
- \(\chi_i\) [1/s] は §3.1.6 の Christensen速度リミタで定義した圧縮率をそのまま用いる。
  CFL分母では圧縮速度 \(\Delta r_i\chi_i\) [cm/s] を人工粘性の補正項として加える。
- `av_type="riemann"` では VNR 係数を CFL に使用せず、§3.1.6 の Riemann face
  reconstruction から
  \[
  \Delta t_{acoustic+AV}^{riemann}
  = C_{CFL}\cdot\min_{i \in \mathcal{A}}\left(
  \frac{\Delta r_i}{c_{s,i}+
  \max(\Delta u_{i-1/2}^+,\Delta u_{i+1/2}^+)}
  \right)
  \]
  を用いる。
- `av_type="csw"` では、CFL 分母の AV 補正係数 \(C_1, C_2\) に
  `csw_C1`, `csw_C2` を用いる（AI review 2026-07-26 k03 §10 是正 —
  旧実装は `av_linear`/`av_quadratic` を流用しており、CSW 係数の方が
  大きい既定（0.5/2.0 vs 0.1/1.5）では AV 剛性を過小評価していた）。
- `av_type="riemann_compatible"`（2026-08-03、k03 §10 同型是正を実装時に
  適用）でも VNR 係数を CFL に使用せず、raw nodal 圧縮 jump
  \(\max(0,\ u_i-u_{i+1})\) を CFL 分母へ加算する（BJ 制限付き射影 jump の
  上位スケール。\(b_1\Delta u\) でなく \(\Delta u\) を加える点も保守側）。
  acoustic estimator・argmin 診断は `riemann` と同じ扱い
  （`av_linear` のまま — 主 CFL カーネルが AV 剛性を担う）。
- **節点交差ガード（k01 P0-4, AI review 2026-07-26）**：acoustic+AV 分母は
  冷たい滑らかな圧縮（\(c_s \to 0\), limiter により \(\chi_i \to 0\)）で
  消失しうるが、セル面は \(u_i - u_{i+1} > 0\) の速度で幾何学的に閉じ続ける。
  これを防ぐため
  \[
  \Delta t_{cross} =
  C_{cross}
  \min_{i \in \mathcal{A},\; u_i - u_{i+1} > 0}
  \frac{\Delta r_i}{u_i - u_{i+1}}
  \]
  を追加評価する。\(C_{cross}\) = `Numerics.hydro.crossing_dt_safety`
  （既定 0.5、0 で無効）。この量は生の幾何制約であり \(C_{CFL}\) 倍は
  掛けない。健全な衝撃波領域では \(\chi_i\) が生の圧縮率に一致し
  acoustic+AV 側が \(C_{CFL}/C_2 \le 0.5\) 倍以下の閉率で先に律速するため、
  このガードは limiter が \(\chi\) を消す滑らかな冷収縮のみで発火する。
  面加速度による step 内の閉率増加は本ガードでは捕捉されない（W-B の
  非正体積検出 + driver full-step retry がバックストップ）。
- `post_shock_heat=True` のときは、post-shock 熱流束演算子
  （flux \(= -C_{ps}\rho_f c_{s,f}\psi_f (e_R - e_L)\)、face 長は打ち消える）
  の**行和コンダクタンス束縛**を追加評価する（AI review 2026-07-26 k01
  §4.3 是正 — 旧式 \(0.5\Delta r_i/(C_{ps}\psi_i c_{s,i})\) は graded mesh
  と密度ジャンプ面で拡散率を過小評価していた）：
  \[
  G_f = A_f\,C_{ps}\,\rho_f\,c_{s,f}\,\psi_f,\qquad
  \Delta t_{ps} =
  \min_{i \in \mathcal{A}}
  \frac{0.5\,m_i}{\sum_{f \in \partial i} G_f}
  \]
  ここで \(\rho_f, c_{s,f}, \psi_f\) は演算子と同じ算術平均 face 値、
  \(m_i = \rho_i V_i\)。実際の hydro 制約は
  \[
  \Delta t_{hydro} = \min\!\left(\Delta t_{acoustic+AV},\ \Delta t_{cross},\ \Delta t_{ps},\ \Delta t_{H}\right)
  \]
  とする。ここで \(\psi_i\) は §3.1.6 の shock-history sensor である
- `av_heat_C > 0`（人工熱流束 \(H\)、`av_type="vnr"` のみ）のときは、同形の
  行和束縛
  \[
  G_f^{H} = A_f\,C_H\,\rho_f\,l_f\,\chi_f,\qquad
  \Delta t_{H} =
  \min_{i \in \mathcal{A}}
  \frac{0.5\,m_i}{\sum_{f \in \partial i} G_f^{H}}
  \]
  を追加評価する（\(l_f = (\Delta r_L + \Delta r_R)/2\)、
  \(\chi_f = \max(\chi_L, \chi_R)\)；k01 §4.2 — 従来 \(H\) には専用の
  timestep 制約が無かった）。`av_type="csw"` の \(H\)（\(\chi^{lim}\) 使用）
  は本束縛の対象外（従来マージン運用のまま）。
  いずれの \(\psi, \chi\) も CFL 時点（\(t^n\) 状態）の値で評価するため、
  step 内で新たに shock 化するセルは 0.5 の安全率と retry が吸収する。
- 既定：\(C_{CFL}=0.3\)

#### 3.1.10 時間積分（Predictor–Corrector、2次）

§3.2.12と同一スキームを1D球対称に適用する（ILESTA, Takabe 2008 と同形式）。

> **注**：`T_start_eV > 0` の場合、非活性セル（`hydro_active_c = false`）は力寄与がゼロとなり、
> 結果として座標・速度が更新されない（§2.1.1）。以下の式は活性セルに対して適用される。

**Predictor（半ステップ）**：
1. 加速度：\(a_j^n\) を \(P^n, Q^n\) から算出（§3.1.4）
2. 速度半更新：\(u_j^{n+1/2} = u_j^n + \frac{\Delta t}{2}\,a_j^n\)
3. 位置半更新：\(r_j^{n+1/2} = r_j^n + \frac{\Delta t}{2}\,u_j^{n+1/2}\)
4. \(V^{n+1/2}, \rho^{n+1/2}\) 再計算
5. EOS → \(P^{n+1/2}\)、AV → \(Q^{n+1/2}\)

**Corrector（全ステップ）**：
1. 加速度再計算：\(a_j^{n+1/2}\) を \(P^{n+1/2}, Q^{n+1/2}\) から算出
2. 速度全更新：\(u_j^{n+1} = u_j^n + \Delta t\,a_j^{n+1/2}\)
3. 位置全更新：\(r_j^{n+1} = r_j^n + \Delta t\,u_j^{n+1/2}\)
4. \(V^{n+1}, \rho^{n+1}\) 再計算
5. エネルギー更新（§3.1.5。legacy path では \(P_{half}, Q_{half}\) と
   \(V^{n+1}-V^n\) を使用。`compatible_energy=true` かつ ideal-gas 1D_SPH exact path では
   Corrector 加速度と同じ \(p_q^{n+1/2}\)、\(r^{n+1/2}\)、\(\bar u\) から force work を用いる）
   → 人工熱流束 \(H\) の適用（`av_heat_C > 0` のとき、§3.1.6）
   → ion-only artificial heat conduction \(H^{ion}\) の適用
      （`ion_art_heat_C > 0` のとき、§3.1.6）
   → localized post-shock heat flux \(H^{ps}\) の適用
      （`post_shock_heat=True` のとき、§3.1.6）
   → EOS → \(T^{n+1}, P^{n+1}\)

> legacy Corrector の PdV 更新で用いる時間中心値は
> \(P_{half} = (P^n + P^{n+1/2})/2,\; Q_{half} = (Q^n + Q^{n+1/2})/2\) とする。
> exact compatible path では Corrector 加速度に使った
> \(P_e^{n+1/2}, P_i^{n+1/2}, Q^{n+1/2}\) と filter 後の \(p_q^{n+1/2}\) をそのまま使う。

> **既知の時間精度制限（legacy_pc；AI review 2026-07-26 k01 P0-2/P0-3）**：
> Predictor は \(e_e, e_i\) を half step へ進めないため、EOS の返す
> \(P^{n+1/2} = P(\rho^{n+1/2}, e^n)\) は断熱圧縮でエネルギー寄与分
> \(O(\Delta t)\) だけ真の midpoint 圧力から外れる（理想気体断熱で
> \(d\ln P_{stale}/dt = -\theta\) vs 真値 \(-\gamma\theta\)）。さらに legacy
> PdV の平均 \((P^n + P^{n+1/2})/2 = P^{n+1/4} + O(\Delta t^2)\) は実効
> quarter-step 圧力であり、両者を合わせて legacy path のエネルギー更新は
> 時間 1 次精度（圧縮加熱の系統的過小評価）に留まる。歴史的挙動として
> bitwise 保存するため legacy_pc 既定は据え置く。

**`Numerics.hydro.time_integrator="midpoint_v2"`（opt-in、1D 専用；k01
P0-2/P0-3 是正）**：Predictor 段で全熱力学状態を half step へ進める。

1. Predictor の座標半更新・\(\rho^{n+1/2}\) 再計算の後、EOS closure の前に
   stage エネルギー更新を行う：
   \[
   e_{e}^{\star} = e_e^n - P_e^n\,\frac{\Delta V^{n\to n+1/2}}{\Delta M},\qquad
   e_{i}^{\star} = e_i^n - \left(P_i^n + [Q^n]\right)\frac{\Delta V^{n\to n+1/2}}{\Delta M}
   \]
   （\(\Delta V^{n\to n+1/2} = V^{n+1/2} - V^n\) は predictor の幾何半更新の
   体積差、\(Q^n\) work は `av_heat_to` の宛先へ、1T は総和形。\(Q_{ei}\)
   等の source は stage には含めない — Strang 分割の別演算子）。
2. EOS closure は \(P^{n+1/2} = P(\rho^{n+1/2}, e^{\star})\) を返す
   （真の時間中心圧力）。AV も同状態で再評価。
3. Corrector のエネルギー更新は \(P_{half} = P^{n+1/2}\)、
   \(Q_{half} = Q^{n+1/2}\) を**直接**使う（\(P^n\) との再平均はしない）。
4. stage 値 \(e^{\star}\) は最終保存則を決めない：Corrector の全エネルギー
   経路（legacy 1T/2T・compatible）は \(e^n\)（`e_old`）を基点に全ステップ
   更新するため、stage は midpoint force/圧力を作るためだけに使われ破棄
   される（stage の \(\max(\cdot,0)\) クランプも transient で台帳に載せない）。
   人工熱流束 \(H\) の \(e^{n+1/2}\) 参照（§3.1.6）は本モードで初めて字義
   どおりになる。

これにより滑らかな断熱問題の時間精度が 2 次に回復する（検証:
`hydro_1d_time_order` ctest — 一様断熱圧縮の \(\Delta t\) 半減比較で
収束次数 \(p \approx 2\) を要求。legacy_pc は \(p \approx 1\)）。
> 人工熱流束 \(H\) は Predictor 状態の \(\chi^{n+1/2}\)、\(\rho^{n+1/2}\)、\(e^{n+1/2}\)
> から構成し、同じ Corrector 内で保存形の別更新として適用する。
> `post_shock_heat` の sensor 更新、`post_shock_velocity_damping_C` の
> recent-shock weight 評価、および \(\mathbf{H}^{ps}\) 評価は
> \(Q^{n+1/2}\) と \(t^{n+1/2}\) を用いる。

> **注**：位置更新(3)に \(u^{n+1/2}\)（半ステップ速度）を使用する。
> これは Leapfrog 形式であり、\(u^{n+1}\)（全ステップ速度）を使用するとエネルギー保存が劣化する。

#### 3.1.11 境界条件（1D）

| 境界 | ノード | 速度 | 圧力 |
|------|--------|------|------|
| 中心 (\(j=0\)) | \(r_0=0\) 固定 | \(u_0=0\)（対称性） | —（ゴーストセル不要） |
| 外側 (\(j=N\)) 自由 | 自由移動 | 自由 | \(P_{ext}=0\)（真空） |
| 外側 (\(j=N\)) 固定壁 | \(r_N\) 固定 | \(u_N=0\) | 反射 |

**1D自由境界のゴーストセル値**：外側境界（\(j=N\)）に仮想セル \(i=N\) を配置する。
ゴーストセルの全スカラー量は **最近接内部セルのコピー**（ゼロ勾配外挿）とする：
- \(\rho_{ghost} = \rho_{N-1}\)
- \(T_{e,ghost} = T_{e,N-1}\)、\(T_{i,ghost} = T_{i,N-1}\)
- \(P_{ghost} = 0\)（自由境界）または \(P_{ghost} = P_{drive}(t)\)（駆動圧境界）
- \(Q_{ghost} = Q_{N-1}\)（人工粘性はゼロ勾配コピー — 外側節点の運動量式と
  compatible work の両方で同じ \(p_q - p_{q,ghost}\) を使うため、境界面の
  \(Q\) は力・仕事の双方から一貫して打ち消える）

**節点量のゴースト規約（スカラー量とは別系統；2026-07-26 明確化、AI review
k01 §2.11）**：AV の slope 再構成（§3.1.6 の \(\sigma_j\)）が参照する仮想
節点は次で定義する：
- 中心（\(j=0\)）：鏡映 \(r_{-1} = -r_1\)、\(u_{-1} = -u_1\)
- 外側（\(j=N\)）：線形外挿 \(r_{N+1} = 2r_N - r_{N-1}\)、
  \(u_{N+1} = 2u_N - u_{N-1}\)（全境界種別で共通）

固定壁では速度拘束 \(u_N = 0\) を課すため、上の線形外挿は自動的に
\(u_{N+1} = -u_{N-1}\)（壁鏡映）になる。旧記載の「固定壁ゴースト速度
\(u_{ghost} = -u_N\)」は \(u_N = 0\) の下では恒等的に 0 となり反射条件を
表さないため削除（実装は上記の線形外挿＋壁拘束で正しい鏡映を実現している）。
§8.1 の統合ゴーストセル規則と整合する。

> 爆縮計算の典型設定：中心＝対称、外側＝自由（真空）。

#### 3.1.12 自動等質量ゾーニング（Auto-Zone）

ICFカプセルのように密度が数桁変化する構成で、Lagrangian計算に適した初期メッシュを自動生成する。
HELIOSの自動ゾーニングに相当する機能であり、1D_SPH専用。

##### 3.1.12a 基本概念：球殻等質量分割

密度 \(\rho_{ref}\) のリージョン \([r_{in}, r_{out}]\) を \(N\) ゾーンに等質量分割する場合、
各ゾーン質量は：
\[
\Delta M = \frac{M_{total}}{N} = \frac{\rho_{ref}}{N}\frac{4\pi}{3}(r_{out}^3 - r_{in}^3)
\]

\(k\) 番目のノード位置（\(k=0,\ldots,N\)）：
\[
r_k = \left(r_{in}^3 + \frac{k}{N}(r_{out}^3 - r_{in}^3)\right)^{1/3}
\]

球対称では体積は \(r^3\) に比例するため、等質量条件は \(r^3\) の等間隔分割に帰着する。

##### 3.1.12b 界面ブリッジ（Asymmetric Geometric Bridging）

隣接する非voidリージョン間で \(\rho_{ref}\) が異なる場合、等質量条件のみでは界面で
ゾーン質量が不連続に変化する。これを解消するため、界面近傍のゾーンを幾何級数で
再分配する「ブリッジ」を生成する。

**設計思想**：各界面に「非対称ブリッジ」を構築する。左リージョンから \(n_L\) ゾーン、
右リージョンから \(n_R\) ゾーンを取り、界面に向かって質量が幾何級数で変化する
ブリッジゾーンとする。ブリッジ外のゾーン（「バルク」）は等質量を維持する。

ブリッジ公比 \(q\) の定義：
- 左側ブリッジゾーン質量：\(m_{bulk,L} \times q,\; m_{bulk,L} \times q^2,\; \ldots,\; m_{bulk,L} \times q^{n_L}\)（界面に向かって増加）
- 右側ブリッジゾーン質量：\(m_{bulk,R} \times q^{-n_R},\; \ldots,\; m_{bulk,R} \times q^{-2},\; m_{bulk,R} \times q^{-1}\)（界面から減少）
- 界面でのマッチング条件：\(m_{bulk,L} \times q^{n_L} = m_{bulk,R} \times q^{-n_R}\)

##### 3.1.12c 二分法による公比 \(q\) の決定

公比 \(q\) はリージョン質量保存から一意に決定される。左右のリージョン総質量を
\(M_L, M_R\)、バルクゾーン数を \(N_{bulk,L}, N_{bulk,R}\) とすると、質量保存式は：

\[
m_{bulk,L} = \frac{M_L}{N_{bulk,L} + S_L^{inc}(q) + \Sigma_{other,L}}
\]
\[
m_{bulk,R} = \frac{M_R}{N_{bulk,R} + S_R^{dec}(q) + \Sigma_{other,R}}
\]

ここで \(S_L^{inc}(q) = q + q^2 + \cdots + q^{n_L}\)（増加側幾何和）、
\(S_R^{dec}(q) = q^{-1} + q^{-2} + \cdots + q^{-n_R}\)（減少側幾何和）、
\(\Sigma_{other}\) は他の界面ブリッジからの寄与。

界面マッチング条件を残差関数として定義：
\[
g(q) = q^{n_L + n_R} \cdot M_L \cdot (N_{bulk,R} + S_R^{dec}(q) + \Sigma_{other,R})
     - M_R \cdot (N_{bulk,L} + S_L^{inc}(q) + \Sigma_{other,L})
\]

\(g(q) = 0\) を **二分法**（60反復、精度 \(\sim 10^{-18}\)）で求解する。
\(q > 1\) が解となり、\(M_L/N_{bulk,L} < M_R/N_{bulk,R}\) のとき \(q\) はブリッジが
左の小質量ゾーンから右の大質量ゾーンへ遷移することを反映する。

##### 3.1.12d ブリッジゾーン数の決定と制約調整

**Phase 1（初期計画）**：

各界面で必要なブリッジゾーン数を質量比から推定する：
\[
n_{required} = \lceil \log(ratio) / \log(\alpha) \rceil
\]
ここで \(ratio = \max(m_{bulk,L}, m_{bulk,R}) / \min(m_{bulk,L}, m_{bulk,R})\)、
\(\alpha\) は目標質量比（`mass_ratio_max` から開始し、必要に応じて `mass_ratio_hard_max` まで緩和）。

ブリッジゾーン数の上限：
\[
cap_i = \min(n_{bridge\_max},\; \lfloor bridge\_frac\_max \times nz_i \rfloor)
\]

**Phase 2（制約調整、最大20反復）**：

二分法で \(q\) を求めた後、以下の制約を反復的に調整する：

1. **ハード制約（`dr_min`）**：界面ゾーン幅が `dr_min` 未満の場合、ブリッジゾーン数を1つ減らす（左右交互）
2. **ソフト制約（`mass_ratio_max`）**：\(q > \text{mass\_ratio\_max}\) の場合、ブリッジゾーン数を1つ増やす（キャップ制約内で）

ハード制約はソフト制約に優先する。制約違反が解消できない場合は警告を発し、最善の結果を採用する。

##### 3.1.12e 多界面結合とファイナライゼーション

リージョンが3つ以上の場合、左右のブリッジが同一リージョン内で重複しうる。
この結合を解消するため、**3回の結合パス**を実行する。各パスで全界面の
\(q\) を再求解し、\(\Sigma_{other}\) を更新する。

**ファイナライゼーション**：全ブリッジ係数が確定した後、各リージョンのバルク質量を
最終的な係数和から厳密に再計算する：
\[
m_{bulk,i} = \frac{M_i}{N_{bulk,i} + \sum c_{left} + \sum c_{right}}
\]

これにより、リージョン質量の厳密な保存（相対誤差 \(\le 10^{-12}\)）を保証する。

##### 3.1.12f ノード生成

確定した各ゾーン質量 \(\Delta M_k\) からノード位置を逆算する：

非voidゾーン（球殻の質量逆変換）：
\[
r_{k+1} = \left(r_k^3 + \frac{3\,\Delta M_k}{4\pi\,\rho_{ref}}\right)^{1/3}
\]

voidゾーン：等半径間隔 \(\Delta r = (r_{out} - r_{in}) / N\)。

リージョン末端ノードは \(r_{end}\) に強制設定する（丸め誤差の蓄積を防止）。

##### 3.1.12g 診断

自動ゾーニング完了後、以下の診断量をログ出力する：
- `nr`：総ゾーン数（= \(\sum nz_i\)）
- `mass_ratio[min,max,mean]`：非voidゾーン間の隣接質量比統計
- `dr_min_actual`：実際の最小ゾーン幅 [cm]
- `violations`：`mass_ratio_max` 超過のゾーン数（void境界は除外）
- 警告メッセージ：`dr_min binding`（ハード制約発動）、`mass-ratio relaxation`（ソフト制約緩和）

#### 3.1.13 Braginskii プラズマ粘性（W-H イオン 2026-07-04 / 電子 channel 2026-07-12 / 2D species port 2026-07-17）

Physical-viscosity module adding unmagnetized Braginskii shear viscosity (ion + electron channels, single-fluid \(V_e=V_i\)) to the 1D (all geometries) and 2D RZ Lagrangian steps (**default OFF** — namelist `Numerics.hydro.plasma_viscosity`; diagnostic env hooks `TENRYU_BRAG_{ENABLE,MODEL,SPECIES,ETA_CONST,ETA0_SCALE,MFP_CAP_CELLS,LNLAMBDA_FIXED,DT_SAFETY}` remain available; all unset leaves the module inactive and bit-identical. `species="ion"` (default) is bit-identical to the pre-electron W-H trajectories in BOTH dims — the kernels are SPECIES-templated with a source-identical ion branch). Implementation: `src/hydro/braginskii_viscosity.{cuh,cu}`, `src/hydro/braginskii_viscosity_device.cuh` (shared coefficient device functions), and `src/hydro/braginskii_viscosity_2d.cu`; designs: `docs/design/wh_braginskii_viscosity_design.md` (ion), `docs/design/2d_visc_port_spec.md` (2D RZ), `docs/design/electron_viscosity_1d_20260712.md` (electron channel + regime adjudication, landed on feature/1d-brushup), `docs/design/visc_2d_parity_20260717.md` (this branch's port + 2D species extension); literature: Braginskii 1965（原典照合 2026-07-12: τ_e=Eq.(2.5e)、η₀^e=Eq.(2.25) "(Z=1)" 明記、η₀^i=Eq.(2.22)）/ Whitney PoP 6, 816 (1999)（η₀₀^e(Z)、一次文献は調達依頼中）/ Velikovich, Whitney & Thornhill PoP 8, 4524 (2001)（電子粘性 shock 加熱の物理; η₀₀^e(Z) 転写元 Eq.(3)）/ Hunana ApJS (2022)（η₀₀^e=0.73094 近代追認）/ Vold et al. PoP 22, 112708 (2015) (1D spherical reference implementation) / Manheimer & Colombant LPB 25, 541 (2007) (coefficient transcription) / Mason et al. PoP 21, 022705 (2014) (mfp cap) / Miller CF 210, 104672 (2020) / Haines PoP 31, 050501 (2024).

**係数（cgs+eV 凍結系）**:

\[ \eta_0 = 0.96\, n_i (k T_i)\, \tau_i \cdot \mathrm{eta0\_scale}, \qquad \tau_i = k_{\tau i0}\,\frac{\sqrt{A}\;T_i[\mathrm{eV}]^{3/2}}{n_i\, Z^4\, \ln\Lambda_{ii}}\ [\mathrm{s}] \]

\( k_{\tau i0} = 3\sqrt{m_p}\,(k_{\mathrm{eV}})^{3/2}/(4\sqrt{\pi} e^4) = 2.0852\times 10^{7} \)（NRL 表記 2.09e7 と 0.2% 一致; MC07 Eq.(2) は \(n_e = Z n_i\) 換え同型）。\( \ln\Lambda_{ii} = \max(2,\ 23 - \ln[Z^3\sqrt{2 n_i}/T_i^{3/2}]) \)（NRL 単一種; `lnlambda_fixed>0` で固定値、Vold は L≈5 固定を使用）。合成形 \( \eta_0 \approx 3.21\times10^{-5}\,\sqrt{A}\,T_i^{5/2}/(Z^4\ln\Lambda_{ii}) \) poise — **密度非依存**（古典結果、Mason 2014 が明言）。prefactor 0.96 は Braginskii 単一種値（イオン自己衝突なので Z 非依存、Z は τ_i 経由のみ; MC07 Eq.(1)、Miller/Whitney Eq.(17) の独立転写と <1% 一致を確認済み）。混合セルは per-cell 有効種（`state.A_eff`, `state.zbar`）の単一種近似。TMAT ionization fractions があるときは \(Z^4\to\bar Z^4 r_4\)（§1.1.3a、自動・1D）。

**電子 channel 係数（2026-07-12）**:

\[ \eta_e = \eta_{00}^e(Z)\, n_e (k T_e)\, \tau_e \cdot \mathrm{eta0\_scale}, \qquad \tau_e = k_{\tau e0}\,\frac{T_e[\mathrm{eV}]^{3/2}}{n_e\, Z\, \ln\Lambda_{ei}}\ [\mathrm{s}], \qquad n_e = Z n_i \]

\( k_{\tau e0} = 3\sqrt{m_e}\,(k_{\mathrm{eV}})^{3/2}/(4\sqrt{2\pi} e^4) = 3.4409\times 10^{5} \)（NRL ν_e=2.91e-6 の逆数と 0.13% 一致 — `conduction.cu` の ν_ei 係数と同族; Braginskii (2.5e) 実用形 3.5×10⁵ の精密値; 恒等式 \(k_{\tau e0}/k_{\tau i0} = \sqrt{m_e/m_p}/\sqrt{2}\) を満たすことを確認済み）。\( \ln\Lambda_{ei} \) は NRL 電子-イオン形（`conduction.cu::coulomb_log_formula` の module-local 双子 `braginskii_coulomb_log_ei`（共有 device header に配置、両 dim の TU が使用）、floor 2; `lnlambda_fixed>0` は lnΛ_ii と lnΛ_ei の両方を固定する）。Z 依存無次元係数は Whitney 1999（Velikovich 2001 Eq. (3) 転写）:

\[ \eta_{00}^e(Z) = \frac{1.81\,Z\,(Z^2+2.82Z+1.343)}{Z^3+4.434Z^2+5.534Z+1.78}, \qquad \eta_{00}^e(1)=0.733\ (=\text{Braginskii Eq. 2.25}),\quad \eta_{00}^e(\infty)=1.81 \]

合成形 \( \eta_e \approx \eta_{00}^e(Z)\times 5.513\times10^{-7}\, T_e^{5/2}/(Z \ln\Lambda_{ei}) \) poise — イオン channel 同様**密度非依存**（n_e は lnΛ_ei と mfp cap 経由のみ）。Z = fmax(zbar, 1)（イオン channel と同一床; Whitney fit は完全電離 Z≥1 域）、T_e 床 = `Numerics.floors.Te`。

**species 合成（user 指示 2026-07-12 の 3-regime 要求）**: `species = "ion"`（既定 = W-H bit 恒等; kernel は SPECIES template + `if constexpr` でイオン分岐 source 恒等）| `"electron"` | `"both"`。`"both"` は**加法合成** \( \eta_\mathrm{eff} = \eta_i + \eta_e \)（単一流体運動量方程式が含む π_i+π_e そのもの）。比

\[ R \equiv \eta_e/\eta_i = 0.01719\;\eta_{00}^e(Z)\,(T_e/T_i)^{5/2}\,\frac{Z^3}{\sqrt{A}}\cdot\frac{\ln\Lambda_{ii}}{\ln\Lambda_{ei}} \]

は (T_e/T_i)^{5/2} Z³ で生産条件横断 4+ 桁を掃引する（等温: H 0.013 / DT 0.008 / CH 0.36 / Al ≈12 / Au(Z̄≈50) ≈270; DT は T_e/T_i≈7 で R≈1）。従って電子支配域では η_eff≒η_e（イオン寄与 <1%）、イオン支配域では η_eff≒η_i、中間帯 R∈~[0.1,10] では両方が寄与 — **「支配側のみ・中間は両方」の regime 挙動が閾値スイッチなしに漸近的に実現される**。明示閾値切替は (i) η 不連続（力・dt の跳び）、(ii) 物理根拠のない r_lo/r_hi ノブ、(iii) 中間帯での実在応力の棄却、の三重欠陥で棄却（電子設計 doc §3）。交差軌跡 \((T_e/T_i)_{R=1} = [58.2\sqrt{A}/(\eta_{00}^e(Z) Z^3)]^{2/5}\)。`model="constant"` は species 不変の η_eff=eta_const（ion→全量イオン / electron→全量電子 / both→半々 — 運動量物理は species 不変、熱配分のみ変わる）。regime 状態は history 診断（下記）で常時可観測。

**mfp キャップ（Mason 2014、既定 ON C=20）**: \( \tau_{\mathrm{eff}} = \min(\tau_i,\ C\,\Delta r / v_{th,i}) \)、\( v_{th,i} = \sqrt{kT_i/m_i} \)（1D 熱速度規約; √3 の任意性は C に吸収）。高温希薄セル（void・ホットスポット縁）の \(T_i^{5/2}\) 暴走と dt 崩壊を防ぐ。λ_ii ≫ Δr は流体近似の域外（MC07 の粘性/イオン圧比 \(R_{visc}=1.28\,\tau_{ii}\,\partial_r v \gtrsim 1\) 警告域）。`mfp_cap_cells=0` で無効（非推奨）。電子版キャップも同一ノブ C を共有: \( \tau_{e,\mathrm{eff}} = \min(\tau_e,\ C\,\Delta r/v_{th,e}) \)、\( v_{th,e}=\sqrt{kT_e/m_e} \)（同一 1D 熱速度規約）。電子は \(v_{th,e}/v_{th,i}\approx 68\sqrt{T_e/T_i}\)（DT）だけ T^{5/2} 暴走に敏感で、電子熱伝導の flux limiter と同根の措置。

**応力（trace-free、1D 全幾何統一; `model="constant"` は η=eta_const 一様の検証モード）**: セル c で \( e = \Delta u/\Delta r,\ h = \bar u/\bar r = (u_L+u_R)/(r_L+r_R) \)、\( \nabla\!\cdot\!v = e + \alpha h \)（α = 2/1/0 = 球/円筒/平面、平面は h≡0）:

\[ W_{rr} = 2e - \tfrac{2}{3}\nabla\!\cdot\!v,\quad W_{tt} = 2h - \tfrac{2}{3}\nabla\!\cdot\!v,\quad (\text{円筒のみ } W_{zz} = -\tfrac{2}{3}\nabla\!\cdot\!v),\qquad \pi = -\eta_0 W \]

運動量発散（trace-free 恒等式）:

\[ (\nabla\cdot\pi)_r = \frac{\partial \pi_{rr}}{\partial r} + \frac{\alpha\,(\pi_{rr}-\pi_{tt})}{r} \]

ノード力 = 面積重み flux 差 \( A_j(\pi_{rr,R}-\pi_{rr,L}) \)（圧力核 §3.1.4 と同型）+ hoop 項 \( \bar V_j\,\alpha(\bar\pi_{rr}-\bar\pi_{tt})/r_j \)（ノード半セル体積重み）。**スカラー pq 融合は不可**（等方応力にのみ正しい離散化で、球面では +3π_rr/r の hoop が欠落する）ため専用加算カーネル。時間中心化は圧力・AV と同一: predictor は n 状態、corrector は half 状態の応力、いずれも damping filter 群の**前**に accel へ加算。境界: ノード 0 スキップ（中心対称）、外側ノードは FIXED/REFLECT でスキップ・FREE では外側応力ゼロ（stress-free 面）。

**粘性加熱**: \( \dot Q_i = \tfrac{\eta_i}{2}\sum W_{\alpha\beta}^2,\ \dot Q_e = \tfrac{\eta_e}{2}\sum W_{\alpha\beta}^2 \ \ge 0 \)（half 状態で評価、セル電力を `apply_artificial_heat_kernel` 経路で堆積; \(\dot Q_i+\dot Q_e = \tfrac{\eta_\mathrm{eff}}{2}W\!:\!W\)）→ **2T では \(\dot Q_i\) がイオン内部エネルギー（W-H 規約不変・AV の `av_heat_to="ion"` 既定と同規約）、\(\dot Q_e\) が電子内部エネルギー**（Velikovich 2001 §II–III: 電子粘性散逸は電子を直接加熱 — 高 Z 衝撃波の T_e/T_i 構造・K-shell 収量への一次効果）; 1T では両方とも全体エネルギー（ee）。正定値形を選択（compatible force-work 形は離散的に符号不定になり得る）; force-work との O(Δ²) 不整合は Vold 2015 と同じ受容で、B2 gate の全エネルギードリフト差分が常時監視する。

**恒等式検証**（`tests/hydro/test_braginskii_viscosity.cu` B1 電池 + gates B2/B3）: (i) 球 homologous \(u\propto r\) → \(W\equiv 0\)（等方 3D 膨張; B3 gate は dt 経路固定で粘性 ON/OFF 軌道一致 ≤1e-12 + dt 解放で limiter 発火 = step 数増を別建てで確認）; (ii) div-free 流（球 \(u\propto 1/r^2\)、円筒 \(u\propto 1/r\)）→ \((\nabla\cdot\pi)_r \equiv 0\) — flux と hoop の相殺で **hoop 係数を単独検定**（円筒は一様格子で離散的にも厳密 = roundoff-null、球は O(Δr²) 収束比 ≈4）; (iii) 平面定在音波の減衰率 \( \gamma = \tfrac{1}{2}\nu_L k^2 = \tfrac{2}{3}(\eta/\rho)k^2 \)（\(\nu_L = \tfrac43 \eta/\rho\)）を inviscid 対照との差分で ±5%（B2 gate、定数 η・AV off・reflect 壁・波エネルギー KE+音響 PE の対数線形フィット）。電子 channel gates（2026-07-12 レーン着地、2026-07-17 本 branch へ移植）: B1 電池に電子版を追加（η_e pin: DT 級 A=2.5, Z=1, T_e=1 keV, n_i=5e22 で η_e=2.676 P・lnΛ_ei=4.775; 電子 mfp cap pin 0.8856 P; Au 級 Z=50 で η₀₀^e 経由 R≈162 と加法恒等 η_both=η_i+η_e ≤1e-12; species 熱分配; dt 加法等式）+ `braginskii_electron_wave_decay`（B2 planar deck を species=electron / both で再走: 同一 γ_th ±5%・E-drift ≤5e-9・γ の species 不変性 ≤1e-6 γ_th）+ `braginskii_regime_map`（静的 2T 3 deck edom/idom/mixed: R 帯 [>50 / <0.05 / 0.2–5]・regime count=active・history H5 round-trip）。

**dt 制限（陽的）**: \( \Delta t \le \mathrm{dt\_safety}\cdot \rho\,\Delta r^2 / (\tfrac{8}{3}\eta_\mathrm{eff}) \)（縦拡散率 \(\nu_L\) の 1D 陽的安定条件）。`compute_dt_lineage`（driver.cpp）に `dt_visc` フィールドと limiter ラベル `"braginskii"` を追加済み。cap 活性域では \( \Delta t \gtrsim \mathrm{safety}\cdot\Delta r/(2.56\,C\,v_{th,i}) \)（音響 CFL の定数分の一）に下支えされる。生産 deck で恒常束縛が観測された場合の escalation path は conduction 同型の STS 部分循環（設計 doc §5; v1 未実装 — 黙った cap 強化での回避は禁止）。電子 channel は同一の縦拡散演算子に入るため `dt_safety` は共有（species 別ノブなし; \(1/\Delta t_\mathrm{both} = 1/\Delta t_i + 1/\Delta t_e\) が厳密に成り立つ — B1 電池で等式検証）。

**Gershgorin 安定性監査（fail-closed、2026-07-26 AI カーネルレビュー k13 F-05/F-09）**: 上記スカラー式は縦拡散のみをステップ開始状態で束縛する。球/円筒の hoop 剛性 \(\alpha(\pi_{rr}-\pi_{tt})/r\)（原点近傍で縦成分の最大 ~2.5 倍）と、ステップ内の \(\eta\propto T^{5/2}\) 成長（predictor 衝撃加熱の T 倍化で corrector 剛性 ~5.7 倍）は見えない。このため 1D Lagrangian step の終端で、組立済みノード加速度演算子（`braginskii_add_accel_1d_kernel` の凍結 η 線形化 — 一様平面格子で \(2/\lambda_G\) がスカラー式の安定限界に厳密一致）の Gershgorin 行和 \(\lambda_G\) を評価し、\(\Delta t \le 2/\lambda_G\) を監査する（`compute_viscous_gershgorin_lambda_1d`）。既定 `dt_safety=0.3` では常に合格（bit 中立）。違反時は non-positive-volume ガードと同型の soft retry（`driver_full_step_retry_enabled` 時、`suggested_dt = dt_safety · 2/\lambda_G`）または fail-fast。**入力検証（F-01）**: `TENRYU_BRAG_*` env と namelist の両起点で strict parse（完全 parse・有限性）+ 範囲検証（負の `eta0_scale` は反拡散、`dt_safety<=0` は dt 無効化）+ 未知 model/species の reject を `validate_params` が enabled 時に強制する（fail-open の黙認廃止）。**診断の物理値純化（F-10）**: history 診断の `eta_{i,e}_phys` は uncapped・unscaled・NRL-log の古典値に固定（cap 既定 20 cells が regime 地図へ格子依存を持ち込まないため）。構成値 channel（`eta_eff`・heat rate）は従来どおり run の knob を反映する。scratch-pool 化（W-F 規約準拠、旧 per-call cudaMalloc/Free の除去）も同時に実施。

**history 診断（診断バンドル原則、2026-07-12; dim==2 は 2026-07-17）**: enabled 時、history cadence で `/diagnostics/plasma_viscosity_history/` に {cycle, t_s, eta_i_max, eta_e_max, eta_eff_max, ratio_{min,geomean_masswt,max}, n_cells_{e_dom,i_dom,mixed,active}, heat_rate_{i,e}_tot} を追記する。eta_i/eta_e と R 統計は**物理 channel 値**（model/species 非依存 — legacy ion run でも regime 地図が見える）、eta_eff と heat rate は構成値。regime 分類（R≥10 で e_dom / R≤0.1 で i_dom）は診断専用定数で物理への feedback なし。還元は device per-cell 配列の host 逐次集約（atomics 不使用 — bitwise 再現性域に非決定を持ち込まない）。積算器は持たない（瞬時 rate のみ — Strang retry 安全）。dim==2 は同一場のセマンティクスで、歪み演算子は 2D 応力 kernel の shoelace-gradient + hoop 分解、cap 長は dt kernel と同じ最小 active 辺長（`compute_history_diagnostics_2d`、structured/multiblock 両対応）。

**2D RZ port**:

For a cell \(c\) with cyclic corners \(k\), signed planar shoelace area \(A_c\), coordinates \((r_k,z_k)\), and velocities \(\mathbf v_k=(v_{r,k},v_{z,k})\), define

\[
a_{r,k}=\frac{z_{k+1}-z_{k-1}}{2},\qquad
a_{z,k}=\frac{r_{k-1}-r_{k+1}}{2},\qquad
G=\frac{1}{A_c}\sum_k \mathbf v_k\otimes\mathbf a_k,\qquad
h=\frac{\sum_k v_{r,k}}{\sum_k r_k}.
\]

The polygon identity \(\sum_k\mathbf x_k\otimes\mathbf a_k=A_c I\) makes \(G\) exact for linear fields. Reversing the cyclic corner orientation changes the signs of both \(\mathbf a_k\) and \(A_c\), so \(G\) and the force ratios below are invariant; either cyclic orientation is accepted, while \(|A_c|=0\) is rejected. With \(D=G_{rr}+G_{zz}+h\), the trace-free components are

\[
W_{rr}=2G_{rr}-\frac{2}{3}D,\qquad
W_{zz}=2G_{zz}-\frac{2}{3}D,\qquad
W_{\phi\phi}=2h-\frac{2}{3}D,\qquad
W_{rz}=G_{rz}+G_{zr},\qquad \pi_{ab}=-\eta_\mathrm{eff}W_{ab}.
\]

The work-conjugate corner force uses \(\beta_k=(\sum_j r_j)^{-1}\):

\[
F_{r,k}=V_c\left[\frac{\pi_{rr}a_{r,k}+\pi_{rz}a_{z,k}}{A_c}+\pi_{\phi\phi}\beta_k\right],\qquad
F_{z,k}=V_c\left[\frac{\pi_{rz}a_{r,k}+\pi_{zz}a_{z,k}}{A_c}\right].
\]

These formulas give the exact per-cell identities

\[
\sum_kF_{z,k}=0,\qquad
-\sum_k\mathbf F_k\!\cdot\!\mathbf v_k
=\frac{V_c\eta_\mathrm{eff}}{2}\left(W_{rr}^2+W_{zz}^2+W_{\phi\phi}^2+2W_{rz}^2\right),\qquad
\sum_kF_{r,k}=\frac{V_c\pi_{\phi\phi}}{\bar r},
\]

where \(\bar r=(\sum_k r_k)/n_{verts}\). On the compatible force/work path, `work_visc_per_cell` tallies \(-\sum_k\mathbf F_k\cdot\mathbf v_k\); the disabled arm retains the byte-identical original expression. On the legacy path, the positive-definite \(V_c\eta W{:}W/2\) heat rates are deposited immediately after the PdV update. The 2D heating routing is hard-wired per channel in 2T and deliberately does not follow `av_heat_to` (the 1D path is hard-wired the same way — `H_brag` to `ei` in 2T / `ee` in 1T and `H_brag_e` to `ee`, `hydro_1d.cu`; `av_heat_to` governs ONLY the artificial-viscosity heat. An earlier revision of this sentence claimed the 1D physical-viscosity heat followed `av_heat_to` — stale text corrected 2026-07-26, AI kernel review k13 F-13): the ion-channel heat goes to `ei` and — since the 2026-07-17 species port — the electron-channel heat goes to `ee` (1T deposits everything into the single matter energy). Concretely, on the legacy path a species-gated second `apply_visc_heat_kernel` launch deposits `visc_heat_rate_e_per_cell` with the `use_two_temp=0` routing (= `ee` in both modes); on the compatible path the kernel is templated on `VISC_SPLIT` — the split arm routes \(f_e\,W_{visc}\) to `ee` and \((1-f_e)W_{visc}\) to `ei` with \(f_e = \dot Q_e/(\dot Q_i+\dot Q_e)\) from the corrector-stage heat-rate buffers (equal to \(\eta_e/\eta_\mathrm{eff}\) exactly in real arithmetic; \(W{:}W=0\) zeroes the work itself, so the \(f_e=0\) guard routes nothing), while the `VISC_SPLIT=0` arm (OFF and `species="ion"`) keeps the pre-electron arithmetic source-identical. The exact-adjointness total-energy closure is untouched — the split only routes it.

For 2D, let \(L_c=\min_k|\mathbf x_{k+1}-\mathbf x_k|\). The explicit limit and mfp cap use the same length (both channels):

\[
\Delta t_{visc}=\mathrm{dt\_safety}\min_c\frac{\rho_cL_c^2}{(8/3)\eta_{\mathrm{eff},c}+\mathrm{tiny}},\qquad
\tau_{eff}=\min\left(\tau_s,\frac{\mathrm{mfp\_cap\_cells}\,L_c}{v_{th,s}}\right),\quad s\in\{i,e\}.
\]

**2D species extension (2026-07-17, `docs/design/visc_2d_parity_20260717.md`)**: the 2D corner-force and dt kernels are `SPECIES`-templated with fully separated `if constexpr` branches exactly like the 1D kernels — SPECIES=0 keeps the ion-only load order and expressions source-identical (bitwise contract in both dims), SPECIES=1 evaluates \(\eta_e(T_e)\), SPECIES=2 composes \(\eta_\mathrm{eff}=\eta_i+\eta_e\) additively (`model="constant"` splits `eta_const` half/half, the 1D convention). Per-channel heat rates land in `visc_heat_rate_per_cell` (ion) and `visc_heat_rate_e_per_cell` (electron; allocated only when `species != "ion"`). The corner-force expression itself is built from \(\eta_\mathrm{eff}\) in every branch, so the adjointness / per-cell z-momentum / null identities hold for every species by construction.

The v1 exclusions are fail-closed: 2D per-material conservation with viscosity enabled raises `ConfigError`; `button_center`, `central_pseudo_core`, and `pole_angular_derefine` are guarded by runtime assertions. The `wj` mesh-forensics decomposition excludes viscosity (the force remains in the total), and reflect walls use one-sided viscous stress with no mirror-cell flux.

**適用限界（v1、設計 doc §10）**: 単一有効イオン種（Vold 型多種混合則なし）/ 電子 channel は古典 Braginskii/Whitney（完全電離 Z≥1 fit・非縮退）: **WDM/縮退域**（\(T_e \lesssim T_F = 7.9\,\mathrm{eV}\,(n_e/10^{23}\,\mathrm{cm^{-3}})^{2/3}\) や強結合）では古典 τ_e・lnΛ_ei が破れ η_e の温度感度を過大評価する — v1 は註記のみ（縮退補正なし）。単一流体 V_e=V_i（電流駆動の電子歪みなし）/ 非磁化 η₀ のみ / **AV は不変・併用が既定**（捕捉衝撃波は \(\Delta x/\lambda_{ii}\sim 10^2\text{-}10^3\) で物理粘性は shock capturing の代替にならない [MC07]; Kn≳1 では粘性を陽に入れても実測衝撃波構造は再現されない [Haines 2024 §III]）。係数の系統不確かさは一級モデル間 ~5×（Stanek 2024 / Haines 2024）— `eta0_scale` が感度ノブ。

### 3.2 2D RZ（四辺形セル、回転対称）— DRACO準拠 Lagrangian Hydro

#### 3.2.0 Polar-in-box Stage-1 algebraic mesh

For `Mesh.logical_mesh_2d="polar_in_box"`, let
$N_P=\texttt{polar\_prefix\_nr}=\operatorname{len}(\texttt{explicit\_nodes})-1$,
$N_M=\texttt{morph\_rings}$, and $N_C=\texttt{collar\_rings}$. The total
radial cell count is $N_r=N_P+N_M+N_C$. Nodes through $i=N_P$ reuse the
existing polar coordinate construction,

\[
(R_{ij},Z_{ij})=(s_i\sin\theta_j, z_c+s_i\cos\theta_j),
\]

with the two axis rows overwritten to $R=0$ exactly. The final-ring corner
indices are the theta-ladder indices nearest

\[
\theta_{TR}=\operatorname{atan2}(R_b,Z_t-z_c),\qquad
\theta_{BR}=\pi-\operatorname{atan2}(R_b,z_c-Z_b).
\]

Within each of the top, right, and bottom index bands, the side arclength
coordinate is proportional to the cumulative theta *values*. For example,

\[
u_{top,j}={\theta_j-\theta_0\over\theta_{j_{TR}}-\theta_0},
\]

with the analogous normalization on the other two bands. Thus a graded theta
ladder remains proportionally graded on each box side. The two anchor nodes are
assigned to the rectangle corners exactly.

The morph uses

\[
\phi_{kj}=\theta_j+A(k/N_M)(\phi_{Mj}-\theta_j),\qquad
A=10u^3-15u^4+6u^5,\qquad
u=\operatorname{clamp}{(k/N_M-0.20)\over0.80},
\]

and the per-line radial widths

\[
\Delta R_{kj}=\Delta_b q_j^k,qquad
D_M(\phi_{Mj})-s_b
=\Delta_b\sum_{k=0}^{N_M-1}q_j^k.
\]

The positive root is found by deterministic bisection. A required
$q_j>\texttt{morph\_growth\_max}$ is a configuration error. Strict angular
ordering is checked at every generated morph ring.

For the collar, $q_C=\min(\texttt{morph\_growth\_max},1.10)$. Let $h_0$ be
the median over angular nodes of the final morph cell projected onto its
assigned box-side normal. The first inset rectangle is defined by

\[
m_{in}=h_0\sum_{k=0}^{N_C-1}q_C^k.
\]

Because the morph endpoint itself depends on $m_{in}$, this scalar
self-consistency equation is solved deterministically. Collar side-normal
increments $h_0q_C^k$ are scaled once by $m_{in}/\sum h_k$; the final ring
is then assigned to the exact box rather than obtained by accumulation.

These clockwise logical quads use the general RZ quadrilateral geometry with
orientation factor $-1$, not the spherical-wedge or tri-fan kernels. Stage 1
supports mesh build, initialization, and step-0 output only; the first attempted
time step fails with `ConfigError` until the Stage-4 boundary refactor lands.

#### Graded button center (multiblock cart-core sizing)

For `center_mode="graded_button"`, the mesh planner searches the existing
SmoothZoning radial faces deterministically. Let \(s_k\) be an interior face,
\(h_{r,k}=s_{k+1}-s_k\), \(h_s\) the unfeathered natural spacing of the
innermost material region, and
\(\Delta\theta_{\min},\Delta\theta_{\max}\) the extrema of the angular ladder.
The face is valid only when

\[
A_k=\max\!\left(
 {h_{r,k}\over s_k\Delta\theta_{\min}},
 {s_k\Delta\theta_{\max}\over h_{r,k}}
\right)\le1.5,
\qquad
\min(h_{r,k},s_k\Delta\theta_{\min})\ge {h_s\over1.5},
\qquad
s_k+4h_s\le R_{\rm region}.
\]

Here \(R_{\rm region}\) is the outer interface of that single-material region.
Among valid faces, the planner selects the smallest face satisfying
\(s_k\ge1.15\sqrt{2}\,n_c h_s\); if none does, it selects the largest valid
face. The topology fixes \(n_c=N_\theta/4\), and the emitted Cartesian
half-core size is

\[
r_c=\min\!\left({s_b\over1.15\sqrt{2}},\ n_c h_s\right).
\]

Thus a uniform angular ladder requires the compatibility capacity
\(R_{\rm region}\ge(N_\theta/\pi+4)h_s\). The planner constants \(1.5\),
\(4\), and \(1.15\) are **PROVISIONAL**.

With `multiblock_cart_core_bridge_grading="quintic_log"`, let \(N_b\) be the
bridge layer count, \(h_{\rm core}=r_c/n_c\), and \(h_{\rm shell}\) the first
shell spacing. The implemented interior blend levels are generated from

\[
t_\ell={\ell+1/2\over N_b},\qquad
S(t)=6t^5-15t^4+10t^3,
\]
\[
\sigma_\ell=\exp\!\left[
 \log h_{\rm core}+
 (\log h_{\rm shell}-\log h_{\rm core})S(t_\ell)
\right],\qquad \ell=0,\ldots,N_b-1,
\]
\[
w_0=0,\qquad
w_\ell={\sum_{m=0}^{\ell-1}\sigma_m\over
             \sum_{m=0}^{N_b-1}\sigma_m}
\quad(1\le\ell<N_b),\qquad w_{N_b}=1.
\]

#### Exterior-only cone conformance (polar_in_box)

Cone generation is a post-pass over a completed no-cone `polar_in_box` mesh.
The polar prefix (and, for cart-core topology, its core, bridge, shell nodes,
connectivity, and angular ladder) is frozen and never rewritten; only rings
strictly outside the lock ring are replaced. Prefix byte identity is therefore
architectural rather than a consequence of evaluating a zero blend.

The wall ray \(j_w\) is the interior ray whose baseline **box-ring** angle
\(\phi^0_{\rm box,j}\) is nearest `cone_theta_wall`; its target angle is then
snapped to that wall angle. At the first baseline wall ring reaching the tip,
with radial spacing \(h_{\rm wall}\) and radius \(R_{\rm wall}\), the fine-band
spacing is clamped to

\[
\Delta\phi_w=\min\!\left[
 {h_{\rm wall}\over R_{\rm wall}},
 {1\over2}\min\left(
  \phi^0_{\rm box,j_w+1}-\phi^0_{\rm box,j_w},
  \phi^0_{\rm box,j_w}-\phi^0_{\rm box,j_w-1}
 \right)
\right].
\]

All moved rays share the wall-ray baseline radial activation

\[
\beta(s)=S\!\left(\operatorname{clamp}
 {s-s_a\over s_{\rm tip}-s_a},0,1\right),
\qquad S(t)=6t^5-15t^4+10t^3,
\]

which is exactly \(1.0\) at and beyond the tip ring. Wall and fine-band rays
are fully overridden. For transition ray \(r=1,\ldots,N_t\), counted outward
from a fine-band edge, the radial reconstruction uses that ray's own baseline
box angle and its exterior nodes are blended with the baseline as

\[
X_{p,r}=(1-w_r)X^0_{p,r}+w_rX^{\rm radial}_{p,r},\qquad
w_r=S\!\left(1-{r\over N_t+1}\right).
\]

The next, non-overridden ray is the exact \(w=0\) baseline join.

Generation gates require strict angular ordering on every exterior ring,
strict radial nesting on every fully overridden ray, positive discrete signed
cell volumes, and wall collinearity at and beyond the tip:

\[
\left|R_p\,(Z_{p+1}-z_c)-(Z_p-z_c)\,R_{p+1}\right|
\le8\epsilon_{\rm mach}\,
\lVert X_p-C\rVert\,\lVert X_{p+1}-C\rVert,
\qquad C=(0,z_c),
\]

i.e. the cross product and the norms are taken about the polar center
\(C=(0,\,\texttt{box\_center\_z})\), matching the implementation (which
subtracts `box_center_z` from every \(z\) before forming the cross product
and the ulp bound). [2026-07-26: the previous canon wrote the global-origin
form \(|R_pZ_{p+1}-Z_pR_{p+1}|\), which is wrong whenever \(z_c\ne0\) —
k17 AI-review §5.3 (M-14/E-03); the implementation has always been
center-relative, canon corrected to match.]

The tip-length gate is

\[
{1.875\,s_{\rm tip}|\Delta\phi_{\rm shift}|\over L_{\rm tip}}
\le\tan20^\circ,
\qquad
\Delta\phi_{\rm shift}=\texttt{cone\_theta\_wall}-\theta_{j_w},
\qquad L_{\rm tip}=s_{\rm tip}-s_a,
\]

with the corresponding discrete per-ring skew check through the activation
interval. v1 is limited to one cone, a zero-thickness single-line wall,
uniform-\(\theta\) zoning when combined with the cart-core topology, and cone
material tagging through the deck's density callable.

#### 3.2.0b2 Cone-shell wall map and ladders (Stage C1)

The truncated material shell is one mapped wall block. With
\(\tau=(\sin\alpha,\sigma_z\cos\alpha)\) and
\(\nu=(\cos\alpha,-\sigma_z\sin\alpha)\), its material-spine map is

\[
x(q,n)=c_t+s(q,n)\tau+n\nu,\qquad
s(q,n)=q+n\tan\alpha\,E(q),
\]

where \(0\le q\le L_w\), \(-t_w/2\le n\le t_w/2\), and
\(Q_5(u)=10u^3-15u^4+6u^5\). The implemented C² end-shear blend is

\[
E(q)=E_t(q)+E_b(q),\qquad
E_t(q)=
\begin{cases}1-Q_5(q/L_t),&q<L_t,\\0,&q\ge L_t,\end{cases}
\]

with, for the planar base cut only,

\[
E_b(q)=
\begin{cases}
0,&q\le L_w-L_b,\\
Q_5((q-L_w+L_b)/L_b),&q>L_w-L_b.
\end{cases}
\]

Thus \(E_t(0)=1\) makes every tip-facet node lie exactly on \(z=z_t\),
while \(E_b(L_w)=1\) makes every base-facet node lie exactly on
\(z=z_{\rm base}=z_t+\sigma_zL_w\cos\alpha\) when
`cone_shell_base_cut="planar"`. Away from the end rotations, \(E=0\) and
the coordinate is exactly wall-normal. The analytic positivity condition is

\[
1-{t_w\over2}\tan\alpha\,\lVert E'\rVert_\infty>0.
\]

The builder uses \(\lVert E'\rVert_\infty\le
\max(1.875/L_t,1.875/L_b)\) for a planar base and requires the resulting
margin to be at least \(0.25\); violation raises `ConfigError` with
`cone_shell analytic wall-Jacobian margin must be >= 0.25`.

The through-wall ladder has even \(N_n\) and mirrors two geometric halves.
For \(m=N_n/2\), growth \(g\), and the first face width \(h_{n,0}\),

\[
h_{n,0}={t_w\over2}{g-1\over g^m-1}.
\]

The endpoints \(n=\pm t_w/2\) and the midpoint \(n=0\) are pinned exactly.
Along the wall, the core landmark stations are
\(\{0,L_t,L_h,L_h+L_g,L_w\}\); a planar base additionally pins
\(L_w-L_b\). With \(h_t=f_t h_{n,0}\) and \(h_b=f_b h_{n,0}\), the inverse
target size is constant at \(1/h_t\) through the tip hold, changes as
\(\exp[-\log(h_b/h_t)Q_5((q-L_h)/L_g)]/h_t\) through the grading interval,
and is \(1/h_b\) thereafter. Each landmark segment is equidistributed in
this measure. A deterministic repair pass increments the segment containing
the larger cell at the first adjacent-size violation until
\(\max(h_i,h_{i+1})/\min(h_i,h_{i+1})\le\)
`cone_shell_l_ratio_max` (default \(1.12\)), or the eight-increment gate fails.

Because \(\det(\tau,\nu)=-\sigma_z\), the logical \(j\) direction uses the
normal ladder unchanged for \(\sigma_z=-1\) and reversed for
\(\sigma_z=+1\); the standard cell-node order therefore remains
counterclockwise. See `docs/design/cone_shell_multiblock_20260719.md` for the
full derivation and staged topology design.

#### 3.2.0c Cone-shell Stage C2 near-face strips

For `Mesh.logical_mesh_2d="cone_shell"`, the C1 wall coordinates and wall-node
ordering remain unchanged. Stage C2 appends three conforming vacuum strips in
the fixed block order `WALL`, `OUTER_NF`, `INNER_NF`, `END_NF`; wall faces own
all layer-0 node IDs. For \(h_{n,0}\) from the common symmetric through-wall
ladder, strip widths and cumulative distances are

\[
h_k=f h_{n,0}g^k,\qquad d_0=0,\qquad
d_k=\sum_{p=0}^{k-1}h_p.
\]

Let \(\Gamma_o(\xi)\), \(\Gamma_i(\xi)\), and \(\Gamma_t(n)\) be the
wall outer face, inner face, and tip facet. Let \(E(\xi)\) be the local
end-rotation envelope of §3.2.0b2 (zero on the wall body; the planar base
adds its base envelope). The through-wall direction is the normalized C1
transverse map derivative

\[
m(\xi)={\nu+\tan\alpha\,E(\xi)\,\tau\over
        \|\nu+\tan\alpha\,E(\xi)\,\tau\|},
\qquad \nu=(\cos\alpha,-\sigma_z\sin\alpha),
\]

so \(m=\nu\) wherever \(E=0\), and the strips stay continuous in \(\xi\)
and aligned with \(\partial x/\partial n\) of the C1 wall map inside the
rotation zones. The appended nodes are

\[
x_o(\xi_i,k)=\Gamma_o(\xi_i)+d_k m(\xi_i),\qquad
x_i(\xi_i,k)=\Gamma_i(\xi_i)-d_k m(\xi_i),\qquad
x_t(n_j,k)=\Gamma_t(n_j)-\sigma_z d_k e_z.
\]

[2026-07-26: the previous canonical text wrote a constant \(\pm\nu\)
offset at every station; the implementation has always used the rotated
normalized direction inside the rotation zones, which is the
C1-consistent choice — k17 AI-review §6.2 (M-09/CS-03); canon updated to
the implementation.]

Interior strip nodes are layer-major and station-minor. Cell numbering is
wall first, followed by OUTER_NF, INNER_NF, and END_NF. Connectivity is generic
four-corner CSR; shared wall-strip edges have two cell incidences and
`Interior` tags. The corner, cavity, exterior, and tip closures in
§3.2.0c2--§3.2.0c3 now close the former wall-base, strip-far-edge, and
strip-side seams, so the standalone assembly is watertight. Every remaining
one-incidence edge is tagged `RectRInnerAxis`, `RectROuter`, `RectZBottom`, or
`RectZTop` according to its axis or box location; boundary nodes carry
`NODE_BOUNDARY` together with `NODE_AXIS` or
`NODE_OUTER_PHYSICAL_BOUNDARY`. These tags are init/output topology metadata
only and do not enable time stepping.

The face-resolution gate is the identity

\[
\chi={d_1-d_0\over h_{n,0}}=f
\]

within \(4\epsilon_{\rm mach}\) relative. Every strip cell additionally
requires positive four-corner Jacobians and positive exact RZ revolved volume,
and the global edge audit requires incidence one or two with no edge above two.

#### 3.2.0c2 Cone-shell tip-corner closure and TIP_PORT (Stage C3)

The two truncated-tip corners use full-depth offset-face templates. At layer
\(k\), let \(A_k\) and \(B_k\) be the existing side nodes of the two meeting
strips, with cumulative offsets \(d_{a,k}\), \(d_{b,k}\) and outward normals
\(m_a\), \(m_b\). The new corner family is the deterministic \(2\times2\)
intersection

\[
C_k=P+\delta_k,\qquad
m_a\mathbin{\cdot}\delta_k=d_{a,k},\qquad
m_b\mathbin{\cdot}\delta_k=d_{b,k}.
\]

The normal matrix must satisfy \(|\det(m_a,m_b)|\ge0.2\), corresponding to
an included angle of at least \(11.5^\circ\); otherwise the builder raises the
corner-normal conditioning `ConfigError`. The template cells reference the
existing \(A_k\) and \(B_k\) strip node IDs directly, so the corner and strip
layers are conforming by construction rather than joined by duplicated nodes.

`TIP_PORT` records the ordered inner-corner--END_NF--outer-corner C-shaped
seam reserved for the future Stage C7 sphere-adapter attachment. In v1 it is
owner-stable topology bookkeeping only: `TIP_PORT` itself generates no cells.

#### 3.2.0c3 Cone-shell cavity, exterior, and tip closure (Stage C4) — watertight assembly

`CAVITY_CORE` emits cubic-Hermite columns from the INNER_NF far edge to the
axis. Define the normalized along-wall measure

\[
\Phi(q)={\int_0^q h(\xi)^{-1}\,d\xi\over
              \int_0^{L_w}h(\xi)^{-1}\,d\xi},
\]

where \(h^{-1}\) is the tip-hold/quintic-log/body size function in §3.2.0b2.
At station \(k\), the strictly monotone selected target measure is
\(M_k=k/N_q\) by default
(`cone_shell_farfield_target_measure="station_uniform"`), or
\(M_k=\Phi(q_k)\) for the legacy v1 transport selected by `"wall_phi"`.
The axis target is \(A(q)=(+0.0,z_A)\), with
\(z_A=z_t+M_k(z_{\rm base}-z_t)\), hence it is monotone and linear in
\(M\). If \(m\) is the outward through-wall unit direction and \(d\) the
near-to-target chord, the near tangent is
\(T_{\rm near}=-m\lambda\),
\(\lambda=\min(d,8h_{\rm start})\); the arrival tangent is the inward radial
\(T_{\rm arr}=-d\,e_r\), so the column meets the axis with \(dr/du<0\)
(approach from \(r>0\)).

`EXTERIOR_CORE` similarly joins the OUTER_NF far edge to targets
\((\texttt{box\_r\_max},z_E)\), with \(z_E\) proportional to the same selected
measure \(M\) and strictly monotone in \(\sigma_z z\). v1 uses the radial box face
only. Physical ray/box hits that would fold back through a box corner are
excluded by this monotone prescription. For `EXTERIOR_CORE` the near tangent
is \(+m\lambda\) and the arrival tangent is the outward radial
\(T_{\rm arr}=+d\,e_r\), so each column approaches \(r=\texttt{box\_r\_max}\)
from inside the box; emitted exterior nodes are asserted to satisfy
\(0<r\le\texttt{box\_r\_max}\). [2026-07-26: the arrival tangent was
previously emitted as \(-d\,e_r\), which makes
\(H(1-\varepsilon)=P_1-\varepsilon m_1\) overshoot the box face just before
arrival — k17 AI-review §6.3 (C-01); fixed together with the containment
assert.]

For either column family, let \(d_{\min}\), \(d_{\max}\) be its chord extrema
and let \(h_{\rm start}\) be the final near-strip width. The common cell count
is

\[
K=\operatorname{clamp}\!\left(
\left\lceil{\log(1+0.35d_{\max}/h_{\rm start})\over\log1.35}\right\rceil,
6,32\right).
\]

Each station's ladder spans the curve arclength
\(L=\int_0^1|H'(u)|\,du\ge d\), evaluated by composite 8-point
Gauss--Legendre quadrature on 64 fixed subintervals (the frozen cell count
\(K\) is not affected by this step). A short station with
\(Kh_{\rm start}\ge L\) uses uniform width \(L/K\), subject to
\(L/K\ge0.5h_{\rm start}\). A long station starts with \(h_{\rm start}\) and
solves \(h_{\rm start}\sum_{p=0}^{K-1}g^p=L\) for \(1<g\le1.70\); failure of
either hard band raises `ConfigError` before emission. Interior node
parameters invert the arclength: each cumulative ladder distance \(s_k\)
solves \(S(u_k)=s_k\) (same fixed quadrature) by a 64-iteration bisection on
\([0,1]\); \(u_0=0\) and \(u_K=1\) are pinned exactly, the \(u_k\) are
asserted strictly increasing, and the first emitted physical segment must
lie in the \([0.5,1.5]\) band of the first ladder width. [2026-07-26:
previously the ladder spanned the chord and \(\eta_k=s_k/d\) fed the
Hermite parameter directly; with near-parametric speed
\(\lambda=\min(d,8h_{\rm start})\) this compressed the first physical width
to \(\approx8h_{\rm start}^2/d\) on long stations — k17 AI-review §6.4
(M-08); the arclength inversion restores the requested widths.]

Below the tip, the final `TIP_FILL_WEST`, `TIP_FILL_MID`, and
`TIP_FILL_EAST` decomposition reaches the tip-side box face. The full-depth
corner edges \([A_K\to C_K]\) are block sides: node 0 is the existing
\(A_K\), node 1 is the existing \(C_K\), and both are pinned by node ID.
WEST and EAST own these shared sub-\(C\) columns; MID reuses their node IDs and
has \(K_f-1\) rows when WEST/EAST have \(K_f\) rows.

The final watertightness gate requires every one-incidence boundary edge to
lie either on \(r=0\) or on a box face. The gas fill therefore closes the
standalone simulation domain. This closure contract binds for
`cone_shell_base_cut="planar"`. With `"wall_normal"` the strip base is cut
normal to the wall and the C4a cavity/exterior blends follow it, so the base is
an intentionally OPEN curved surface (axis to box-r) until the
conforming-assembly stages land its closure; the gate tags those base edges
`SphericalOuterFree` (init-only tags — audits and HDF5, no boundary-condition
or hydro semantics) and requires them to equal, as a set, the generator's
expected base chain — the single simple path at the base station running
box radial face → exterior column → outer strip column → wall base column →
inner strip column → cavity column → axis. Every tagged edge must be a
member and the count must match, so seams, cracks, or holes anywhere else
still fail loudly; only the census mode downgrades to logging.
[2026-07-26: k17 AI-review §6.6 (C-05) — previously ANY unclassified
one-incidence edge was accepted as free in `wall_normal` mode.]
`TENRYU_CONE_C4B_CENSUS=1` lists every such edge with coordinates. [2026-07-20:
this restored the `wall_normal` analytic-twin gate (#479), which the C4b gate
had broken 90 minutes after C4a introduced it.] The fixed append-only block
registry is:

| ID | Block |
|---:|:------|
| 0 | `WALL` |
| 1 | `OUTER_NF` |
| 2 | `INNER_NF` |
| 3 | `END_NF` |
| 4 | `TIP_CORNER_O` |
| 5 | `TIP_CORNER_I` |
| 6 | `CAVITY_CORE` |
| 7 | `EXTERIOR_CORE` |
| 8 | `TIP_FILL_WEST` |
| 9 | `TIP_FILL_MID` |
| 10 | `TIP_FILL_EAST` |

HDF5 v2 validates stored `block_count` against a per-scheme expected table:
`cone_shell_spine` expects 11 while the cart-core scheme stays 3. This is a
constraint generalization, not a format change.

#### 3.2.0a S1 γ MVP multiblock topology (default OFF)

`Mesh.topology_scheme="multiblock_cart_core_polar_shell"` defines the S1
mesh-only γ MVP topology: a Cartesian half-core, a transition bridge, and a
polar shell over the RZ half-plane. The default transition is the legacy
Hermite bridge. With
`topology_scheme="single_block"` (default), the rectangular RZ,
spherical-polar annular, and spherical-polar `tri_fan` paths keep their legacy
allocation and generated coordinates byte-identical.

Let \(N_c=\) `Mesh.multiblock_cart_core_n_c`,
\(N_b=\) `Mesh.multiblock_cart_core_bridge_layers`, \(N_s=\) `Mesh.nr`,
\(r_c=\) `Mesh.multiblock_cart_core_r_c`, and
\(r_\mathrm{match}=\) `Mesh.multiblock_cart_core_r_match`. The Cartesian
half-core spans \(0\le R\le r_c\), \(-r_c\le Z\le r_c\), with
\(N_c\times 2N_c\) cells. The bridge has \(N_b\) radial layers and \(4N_c\)
cells per layer. The polar shell has \(N_s\) radial layers and \(4N_c\)
angular segments.

The multiblock sizing contract is
\[
N_\mathrm{cell} = 2N_c^2 + 4N_cN_b + N_s(4N_c),
\]
\[
N_\mathrm{node} =
  (N_c+1)(2N_c+1) + (4N_c+1)(N_b-1) + (N_s+1)(4N_c+1).
\]
The first node term is the Cartesian core, the second counts bridge interior
radial layers including both half-plane axis endpoints, and the third counts
polar-shell rings including the bridge-shell seam ring.

The bridge boundary parameter \(t\in[0,1]\) follows the half-square
counter-clockwise from the north axis to the south axis:
\[
I(t)=
\begin{cases}
(4r_ct,\ r_c), & 0\le t\le 1/4,\\
(r_c,\ r_c-4r_c(t-1/4)), & 1/4 < t\le 3/4,\\
(r_c-4r_c(t-3/4),\ -r_c), & 3/4 < t\le 1,
\end{cases}
\]
and the matching polar half-circle is
\[
O(t)=(r_\mathrm{match}\sin(\pi t),\ r_\mathrm{match}\cos(\pi t)).
\]
For bridge coordinate \(\eta\in[0,1]\), the S1 mapping is the cubic Hermite
blend
\[
X(t,\eta)=(1-w)I(t)+wO(t),\qquad w=3\eta^2-2\eta^3 .
\]
When `Mesh.topology_scheme="multiblock_half_butterfly_5block"`, the Cartesian
core and polar shell coordinates above are unchanged, but the transition is
split into three fan blocks: north \(\theta\in[0,\pi/4]\), east
\(\theta\in[\pi/4,3\pi/4]\), and south
\(\theta\in[3\pi/4,\pi]\). With
\(e(\theta)=(\sin\theta,\cos\theta)\), \(r_0=r_c\),
\(r_m=r_\mathrm{match}\), radial coordinate \(\rho\in[0,1]\), and fan
angular coordinate \(\eta\in[0,1]\), each fan node is the Coons/TFI patch
\[
X(\rho,\eta)=(1-\rho)C(\eta)+\rho O(\eta)
 +(1-\eta)S_0(\rho)+\eta S_1(\rho)-B(\rho,\eta),
\]
where \(B\) is the bilinear interpolation of the four corner points. The fan
edges are
\[
C_N=(r_0\eta,r_0),\quad
O_N=r_m e((\pi/4)\eta),\quad
S_{N0}=(0,(1-\rho)r_0+\rho r_m),\quad
S_{N1}=(1-\rho)(r_0,r_0)+\rho r_m e(\pi/4),
\]
\[
C_E=(r_0,r_0(1-2\eta)),\quad
O_E=r_m e(\pi/4+(\pi/2)\eta),
\]
with \(S_{E0}=S_{N1}\) and
\(S_{E1}=(1-\rho)(r_0,-r_0)+\rho r_m e(3\pi/4)\), and
\[
C_S=(r_0(1-\eta),-r_0),\quad
O_S=r_m e(3\pi/4+(\pi/4)\eta),
\]
with \(S_{S0}=S_{E1}\) and
\(S_{S1}=(0,-(1-\rho)r_0-\rho r_m)\). The central block owns the
\(\rho=0\) seam nodes, the polar shell owns the \(\rho=1\) ring, north owns
the \(\theta=\pi/4\) diagonal interior, and east owns the
\(\theta=3\pi/4\) diagonal interior. Therefore shared seams have one node id
and one coordinate. The half-plane axis edges are written with exactly
\(R=0\). Optional `multiblock_bridge_elliptic_sweeps` Jacobi smoothing moves
only strict fan-interior nodes and holds all fan edges fixed.

The half-butterfly is intentionally not the A1 `rounded_core_seam`. A1 kept a
single Cartesian core and rounded the \(l=0\) seam; at the diagonal logical
corner of that core, both incident logical edges become tangent to the same
smooth seam, so \(\det[X_i,X_j]\to0\). That rank loss is homology invariant for
the one-block corner and empirically flipped under about \(C\simeq1.1\)
compression. The five-block layout moves the same physical region into
finite-valence multiblock vertices: the central/fan/shell seams meet through
shared node ids, but each block keeps a nonsingular logical corner. The
canonical signed-volume orientation is `cell_orientation_sign=+1` for the
central block and `-1` for all three fan blocks plus the polar shell; the
geometry predicate tests `cell_orientation_sign * V_raw > 0`.

B-S1 verifies this mesh/schema layer only. The acceptance set is: five-block
coordinate build with positive cells, full face adjacency and unique-face
metadata, `/mesh/topology/v3` round trip, homologous \(C=10\) admissibility,
and an \(N_c\)-refinement contrast showing the A1 rounded-core rank-loss corner
persists while the five-block finite-valence vertex remains admissible. B-S2
then retargets the hydro consumers that previously inferred core/non-core
orientation from `cell < n_cells_core`: ALE/remap swept volumes,
scaled-reference target volumes, pressure-BC shell dispatch,
path-admissibility diagnostics, Fourier/radial diagnostics, committed
mesh-quality observation, and the GCL audit all consume block-role metadata,
CSR connectivity, or `cell_orientation_sign` instead. B-S2 makes the five-block
mesh hydro-runnable and validates it with a gentle uniform at-rest smoke. Seam
flux closure under gradients and the decisive \(C_\mathrm{meas}\ge7.2\) PAB
compression gate remain B-S3 and B-S4 respectively.

Status as of S3 T5: the pinned-origin tri-fan cap machinery is complete and
default-off byte-identical when unused. Regressions 250/251 are green; the
single red #495 G4 five-block dynamic smoke is the pre-existing central-keystone
known-fail that the cap replaces. Dynamically, the cap removes the central
keystone fold: a gentle convergent implosion runs with the apex pinned where
the Cartesian core folded. ICF-class \(C_\mathrm{meas}\ge7.2\) certification is
IN PROGRESS and NOT yet achieved. Under strong convergent drive, current
evidence shows a near-axis fan/shell mesh tangle and a runaway compatible-energy
residual; a high-AI consultation on that failure mode is in flight. This status
does not redefine \(C_\mathrm{meas}\), does not claim \(C\ge7.2\), and does not
claim "I1-B passed".

For
`Mesh.topology_scheme="multiblock_half_butterfly_trifan_cap_5block"`, S3 T2
replaces only the central Cartesian half-core with a pinned tri-fan cap. Let
\(N_{\rm cap}=N_c\), \(K=4N_c\), and \(I(k)\), \(0\le k\le K\), be the existing
half-butterfly fan inner boundary above. The cap owns one apex node
\(O=(0,0)\) with `NODE_CENTER|NODE_AXIS|NODE_BOUNDARY`. Ring nodes are
\[
X_{m,k}=\frac{m}{N_{\rm cap}} I(k),\qquad
1\le m\le N_{\rm cap},\quad 0\le k\le K .
\]
Rings \(m<N_{\rm cap}\) are private cap nodes, including their \(R=0\) axis
endpoints. Ring \(m=N_{\rm cap}\) is not duplicated: its node IDs are exactly
the fan inner seam IDs over the north, east, and south subranges
\([0,N_c]\), \([N_c,3N_c]\), and \([3N_c,4N_c]\). The first cap radial row is
the \(K\) triangles
\[
(O,\ X_{1,k},\ X_{1,k+1}),\qquad 0\le k<K,
\]
with `cell_nverts=3` and storage slot 3 inactive. Rows
\(1\le m<N_{\rm cap}\) are quads
\[
(X_{m,k},\ X_{m+1,k},\ X_{m+1,k+1},\ X_{m,k+1}).
\]
All cap cells use `cell_orientation_sign=-1`, matching the clockwise
half-plane angular ordering used by the fan and shell blocks. The existing
north/east/south core-to-fan seam records are retained as compatibility labels
for the three cap outer-ring subranges; actual face ownership and duplicate
suppression are through the active-slot CSR and unique-oriented-face lists.
S3 T0-T5 retarget the runtime machinery to these active slots. A cap triangle
has active local faces \(1,2,3\), mapped by
`mesh_topo_active_local_face_corners` to storage-corner edges
\((1,2)\), \((0,1)\), and \((2,0)\) respectively; local face 0 is inactive and
contributes zero swept volume.  The cap/fan interface uses the ordinary
bilateral `unique_internal_faces` CSR path because the cap outer-ring node IDs
are the fan inner-seam node IDs. There is no separate cap-seam remap flux.
CSR remap/GCL, reverse CSR, hydro corner mass and node mass, \(\dot V\),
pressure force and compatible force-work, CSW edge AV, subzonal pressure,
CFL, ALE reference barriers, axis rezone, full-patch targets, and boundary
projection consume `cell_nverts` rather than assuming four active corners.
The cap apex \(O\) remains pinned at \((R,Z)=(0,0)\).

For `Mesh.topology_scheme="multiblock_cart_core_polar_shell"` with
`Mesh.multiblock_transition_scheme="rounded_half_butterfly"`, the same
three-block count, bridge \((\ell,k)\) indexing, cell-node CSR, reverse CSR,
seam tags, and shared seam nodes are retained.  Only bridge interior
coordinates \(0<\eta<1\) are replaced: the inner seam remains \(I(t)\), the
outer seam remains \(O(t)\), and each interior half-plane angular region
\([0,\pi/4]\), \([\pi/4,3\pi/4]\), and \([3\pi/4,\pi]\) is generated by a
Coons/TFI patch between \(O\) and the rounded superellipse cap
\[
S_p(\theta)=a\left(|\sin\theta|^p+|\cos\theta|^p\right)^{-1/p},\qquad
a=r_c\,2^{1/p}.
\]
The optional `multiblock_bridge_elliptic_sweeps` performs mesh-local Jacobi
elliptic smoothing on bridge interior nodes while holding the core seam,
shell seam, and half-plane axis nodes fixed.

When `Mesh.multiblock_transition_scheme="rounded_core_seam"`, the same
cell/node counts, core node ordering, bridge \((\ell,k)\) indexing,
cell-node CSR, reverse CSR, and seam tags are retained, but the shared
core-bridge seam itself is \(S_p(\theta)\). The exposed core edges use the
exact inverse of the seam node map: top edge \((i,2N_c)\) uses \(k=i\),
right edge \((N_c,j)\) uses \(k=3N_c-j\), and bottom edge \((i,0)\) uses
\(k=4N_c-i\). The axis edge remains exactly \(R=0\) with
\[
Z_j = {j-N_c\over N_c}\,a,\qquad j=0,\ldots,2N_c .
\]
Strict interior core nodes are initialized by a Coons/TFI patch from the
axis edge and the three rounded exposed edges, then the same optional Jacobi
elliptic smoother moves only strict-interior core nodes. The rounded bridge
grid is also used with its \(\ell=0\) anchor set to \(S_p\). The
`rounded_half_butterfly` scheme keeps the square inner seam \(I(t)\) and
therefore preserves its previous coordinates. Since \(p>2\),
\[
a=r_c2^{1/p}<\sqrt{2}\,r_c<r_\mathrm{match},
\]
so the rounded cap stays strictly inside the bridge-shell match radius.

Seam nodes at the core-bridge and bridge-shell interfaces are shared once, not
duplicated. Coordinate identity is checked with tolerance
\[
\max(10^{-14}\max(1,|\mathrm{coord}|),\ 10^{-30})
\]
per coordinate. Mesh construction asserts positive RZ volume for every cell.
Verification tests check refinement-convergent total volume against the
analytic spherical volume for the shell outer radius; golden updates require
the verification process in VERIFICATION.md.

#### 3.2.0b Multiblock geometry and CSR connectivity

For `topology_scheme="single_block"` (default), the 2D_RZ hydro discretization
uses the structured cell id \(c=iN_z+j\), structured nodes
\((N_r+1)(N_z+1)\), and the legacy corner order documented in §3.2.1.
This is the byte-identical default path.  For
`topology_scheme="multiblock_cart_core_polar_shell"`, mesh construction builds
the same RZ coordinates and volumes (§3.2.2) on a three-block conforming
topology, then exposes cell-to-node connectivity through CSR:
`cell_node_csr_offsets[c] : cell_node_csr_offsets[c+1]` indexes the four
corner node ids of cell \(c\).  The device mirrors are
`state.mesh.multiblock_cell_node_csr_offsets/indices`; the reverse node-to-cell
CSR used by mass and force assembly is
`state.mesh.multiblock_reverse_csr_node_*`.
For `topology_scheme="multiblock_half_butterfly_5block"`, the same CSR
contract is populated from the five block tables and the v3 topology metadata;
the central block, three fans, and polar shell use single-owner shared nodes on
all seams and the per-cell `cell_orientation_sign` described in §3.2.0a.

All geometry remains cgs RZ geometry: coordinates are [cm], volumes are
[cm\(^3\)], areas are [cm\(^2\)], and pressures are [dyne/cm\(^2\)].
Hydro consumers dispatch on `mesh_topo_is_multiblock(cfg)`: single-block uses
structured indexing kernels, while multiblock uses CSR-aware variants.  The
namelist keys and validation contract are in SPECIFICATION §6.4.2, and the
runtime ownership/serialization layout is in ARCHITECTURE §4.2.

#### 3.2.1 スタガード格子配置

構造四辺形メッシュ上のスタガード格子（ノード中心に速度、セル中心に熱力学量）。

- **セル** \((i,j)\)：\(i=0,\ldots,N_r-1\)、\(j=0,\ldots,N_z-1\)
- **ノード** \((i,j)\)：\(i=0,\ldots,N_r\)、\(j=0,\ldots,N_z\)
- セル \((i,j)\) のコーナーノード：\((i,j),\,(i+1,j),\,(i+1,j+1),\,(i,j+1)\)（反時計回り、添字 \(k=1,2,3,4\)）

**セル中心量**：\(\rho,\, e_i,\, e_e,\, T_i,\, T_e,\, P_i,\, P_e,\, Q\)

**ノード中心量**：\(r,\, z,\, v_r,\, v_z\)

#### 3.2.2 セル体積（RZ回転体）

4頂点 \((r_k, z_k)\)（\(k=1,2,3,4\)、反時計回り、循環添字 \(5\to 1\)）のセル体積：
\[
V_{cell} = \frac{\pi}{3}\sum_{k=1}^{4}(r_k z_{k+1} - r_{k+1} z_k)(r_k + r_{k+1})
\]

> この式は回転体の体積公式を多角形断面に適用したもの（Pappusの定理の離散版）。

> **実装注意（桁落ち回避）**：ALE変形後の薄い・歪んだセルでは、Shoelace型の交差積
> \(r_k z_{k+1} - r_{k+1} z_k\) が強い桁落ちを起こす。実装では各頂点をセル重心
> \((\bar r, \bar z)\) 基準に平行移動してから計算すること（translated coordinates）。
> すなわち \(\tilde r_k = r_k - \bar r\), \(\tilde z_k = z_k - \bar z\) で置き換えてから
> \(V_{cell} = (\pi/3)\sum_{k}(\tilde r_k \tilde z_{k+1} - \tilde r_{k+1} \tilde z_k)(\tilde r_k + \tilde r_{k+1} + 2\bar r)\) とする。
> 断面積 \(A_{cell}\)（§3.2.3）も同様の重心シフトを適用する。

> **A0 candidate-mesh evaluator (library only)**: `src/mesh/candidate_mesh_admissibility.*`
> provides a pure candidate-quality evaluator that is not wired into hydro/ALE
> callers.  For candidate nodes
> \(\mathbf{x}_n(\sigma)=\mathbf{x}_n^{old}+\sigma\Delta\mathbf{x}_n\),
> it evaluates the same signed exact RZ quadrilateral volume above, four
> signed corner Jacobians from the incident edge cross products, and the
> arithmetic mean Gauss-J surrogate.  Spherical-polar `tri_fan` and `button`
> mesh overloads use the canonical clockwise polar sign convention before these
> predicates are applied; the button center cell uses the signed centroid-fan
> subpolygons of its outer node ring for corner-J and Gauss-J.  A candidate
> passes only when the new volume magnitude with unchanged orientation, corner-J
> magnitudes with unchanged orientation, and Gauss-J magnitude exceed
> \(\max(\epsilon_{abs},\epsilon_{rel}\,\text{old})\) for their respective floors.
> Structured overloads use \(c=iN_z+j\) node indexing; CSR overloads read the four
> corner nodes from `cell_node_csr_offsets/indices` and report the first rejected
> cell through `cell_id_stable` for deterministic diagnostics.
> The helper also provides halving line searches for the largest admissible
> \(\sigma\); ALE rezone/tracking paths now call these evaluators (the
> "library only" framing above is historical), so preserving production bit
> behavior rests on the default-off contract fields below.
>
> [2026-07-26, k17 AI-review §8] The evaluator gained opt-in contract fields
> on `CandidateMeshAdmissibilityFloors` (all default-off; existing callers
> keep their endpoint-only bit behavior): `continuous_path` also minimises
> each corner Jacobian (quadratic in the path parameter) and the revolved RZ
> volume (cubic) over the whole path \([0,\sigma]\) with deterministic
> closed-form stationary points, rejecting transient interior inversions
> that endpoint checks miss (§8.7); `require_nonnegative_r` rejects any
> candidate node with \(r<0\) — \(r\) is linear along the path, so endpoint
> checks suffice (§8.6, new failure kind `NegativeRadialCoordinate`);
> `n_nodes`/`n_csr_indices` bounds-check CSR node ids and offsets before any
> coordinate dereference (§8.5, `TopologyInvariantViolation`). The host
> macro-boundary predicates replaced the shared absolute `1e-14` tolerance
> with scale-aware bounds — \(64\,\epsilon_{\rm mach}\) times the 1-norm
> segment/point scales for length bounds and times their product for the
> orient2d test — removing the dimensional inconsistency that misclassified
> submicron cells (§8.10). The halving line search is documented as dyadic:
> it returns the first accepted \(\sigma_{\rm init}/2^k\), not the supremum,
> and an exact `sigma_min` below the last halving is never evaluated (§8.9).

#### 3.2.2a 2D_RZセル幅（放射・ソース平滑化用）

DF、source smoothing、deposit smoothing など、2D_RZセルに対してR方向/Z方向の代表幅が必要な処理では、
セル体積を対応する平均面積で割った幅を用いる。セルインデックスは \(c=iN_z+j\)、ノードインデックスは
\(n=i(N_z+1)+j\)。セル頂点を
\((R_{00},Z_{00})\)、\((R_{10},Z_{10})\)、\((R_{11},Z_{11})\)、\((R_{01},Z_{01})\) とする。

R方向の左右面は回転側面積：
\[
A_{R,L}=2\pi\,\frac{R_{01}+R_{00}}{2}
\sqrt{(R_{01}-R_{00})^2+(Z_{01}-Z_{00})^2},
\]
\[
A_{R,R}=2\pi\,\frac{R_{10}+R_{11}}{2}
\sqrt{(R_{11}-R_{10})^2+(Z_{11}-Z_{10})^2}.
\]

Z方向の上下面は annular ring 面積：
\[
A_{Z,B}=\pi |R_{10}^2-R_{00}^2|,\qquad
A_{Z,T}=\pi |R_{01}^2-R_{11}^2|.
\]

代表セル幅は
\[
h_R=\frac{V_c}{\frac{1}{2}(A_{R,L}+A_{R,R})},\qquad
h_Z=\frac{V_c}{\frac{1}{2}(A_{Z,B}+A_{Z,T})}.
\]
分母が非正または非有限の場合は、それぞれセル4頂点の bbox 幅
\(\max R-\min R\)、\(\max Z-\min Z\) へフォールバックする。
bbox 幅も非正の場合は \(10^{-30}\) cm を用いる。

#### 3.2.3 セル断面積

Shoelace公式によるセル断面積（人工粘性の特性長に使用）：
\[
A_{cell} = \frac{1}{2}\left|\sum_{k=1}^{4}(r_k z_{k+1} - r_{k+1} z_k)\right|
\]

#### 3.2.4 ノード質量（subcell gathering）

ノード \(n\) の質量は、そのノードを共有する全セル \(c\in\mathcal{C}(n)\) からの寄与を集約する：
\[
m_n = \sum_{c\in\mathcal{C}(n)} w_{c,n}\,\Delta M_c,\quad
\Delta M_c = \rho_c V_c
\]

RZ 2D では、セルの内側 R 面と外側 R 面の面平均半径を
\[
R_L = \frac{1}{2}(R_{00}+R_{01}),\qquad
R_R = \frac{1}{2}(R_{10}+R_{11})
\]
とし、Barlow-Burton-Shashkov-Wendroff (LA-UR 09-08130) の
R-weighted subzonal mass quadrature を用いる：
\[
w_L = \frac{2R_L + R_R}{6(R_L+R_R)},\qquad
w_R = \frac{R_L + 2R_R}{6(R_L+R_R)}.
\]
内側コーナー \(n_{00}, n_{01}\) はそれぞれ \(w_L\Delta M_c\)、外側コーナー
\(n_{10}, n_{11}\) はそれぞれ \(w_R\Delta M_c\) を受け取る。したがって
\[
2w_L + 2w_R = 1
\]
であり、各セル質量は4コーナーへの分配で保存される。この分配は
§3.2.5 の R-weighted \(S_{z}\) コーナー力と整合するため、軸行でも
z 方向加速度の column invariance を保つ。

境界ノードでは共有セル数が減るが、各共有セルからは同じ R-weighted
corner mass を受け取る。実装上、退化した不正 RZ セルで
\(R_L+R_R\le0\) となる場合のみ defensive fallback として一様 1/4 分配を用いる。

**KINEMATIC_BASIS_RZ_V1 (G4 epoch, 2026-07-27).** The production corner-mass partition is
the exact R-weighted lump of the Q1 kinematic basis: \(m_{c,k} = M_c\,\int N_k R\,J\,
d\xi d\eta / \int R\,J\,d\xi d\eta\) on the bilinear reference cell (2\(\times\)2 Gauss,
exact at bilinear-quad degree 3), computed at initialization and after each remap and
Lagrangian-invariant between remaps. On i-aligned radial rectangles this reduces exactly to
the previous BBSW radial-linear lump \((2R_L+R_R)/(6(R_L+R_R))\); on axis-column cells it
replaces BBSW's equal-split degeneration with the exact \([1/6,1/6,1/3,1/3]\) distribution;
triangle-degenerate cells use the exact P1 closed form \(w_k=(r_k+\Sigma r)/(4\Sigma r)\).
The legacy lump remains available as corner_mass_convention=bbsw_radial_v0 (and is the
permanent resolution of frozen configs predating the knob); the multiblock exact-subpolygon
ownership masses are a separate contract and are unchanged.

For `topology_scheme="multiblock_cart_core_polar_shell"`, all S1 cells are
four-corner quads but bridge/shell cells are not generally structured
rectangular RZ cells. The multiblock corner mass therefore uses exact RZ
subpolygon volume fractions. For a quad with corners
\((00,10,11,01)\), define the cell centroid
\[
\mathbf{x}_c=\frac{\mathbf{x}_{00}+\mathbf{x}_{10}+
                  \mathbf{x}_{11}+\mathbf{x}_{01}}{4}
\]
and the four edge midpoints \(\mathbf{x}_{01m},\mathbf{x}_{12m},
\mathbf{x}_{23m},\mathbf{x}_{30m}\). The four corner-owned subpolygons are
\[
\begin{aligned}
Q_0&=[\mathbf{x}_{00},\mathbf{x}_{01m},\mathbf{x}_c,\mathbf{x}_{30m}],\\
Q_1&=[\mathbf{x}_{01m},\mathbf{x}_{10},\mathbf{x}_{12m},\mathbf{x}_c],\\
Q_2&=[\mathbf{x}_c,\mathbf{x}_{12m},\mathbf{x}_{11},\mathbf{x}_{23m}],\\
Q_3&=[\mathbf{x}_{30m},\mathbf{x}_c,\mathbf{x}_{23m},\mathbf{x}_{01}].
\end{aligned}
\]
For compatible/invariant multiblock corner masses, the assigned corner masses are
\[
m_{c,k}=\Delta M_c\frac{V_{RZ}(Q_k)}{\sum_{j=0}^3 V_{RZ}(Q_j)},\qquad k=0,1,2,3.
\]
The denominator is the computed subpolygon-volume sum, not a separately
recomputed parent volume, so every active multiblock cell satisfies
\(\sum_k m_{c,k}=\Delta M_c\) to roundoff even on mixed-orientation seams. If
the subpolygon-volume sum is non-positive or non-finite, the implementation
falls back to a defensive uniform \(1/4\) split. On rectangular RZ cells the
exact subpolygon fractions do NOT coincide with the BBSW quadrature weights
above: the exact fractions are \(w_L^{sub}=(3R_L+R_R)/(8(R_L+R_R))\),
\(w_R^{sub}=(R_L+3R_R)/(8(R_L+R_R))\), differing from BBSW by
\(\pm(R_L-R_R)/(24(R_L+R_R))\) — equal only at \(R_L=R_R\) (both \(1/4\)),
largest at the axis (\(R_L=0\): \(1/8\) vs \(1/6\) per inner corner). The
corner-mass partition is therefore topology-dependent by construction:
single-block and tri_fan use BBSW, multiblock uses exact subpolygon
fractions, each used self-consistently for inertia, kinetic energy, and work
(difference locked by tests/hydro/test_rz_svec_exact_gradient.cu;
unification across topologies is a pending design decision — 2026-07-26
audit k02 F-07/§15-6 corrected the earlier "reduces to BBSW up to roundoff"
claim, which was algebraically false). Single-block and tri_fan paths
continue to use their pre-existing dispatch.
The legacy scalar-AV multiblock path keeps its pre-existing parent-volume
normalizer to preserve the byte-identical baseline when
`av_model="scalar_vnr_legacy"` and `subzonal_pressure_enabled=False`.

For `polar_center_treatment="tri_fan"`, center cells have active slots
`0,1,2` and inactive slot `3`. The four-corner BBSW formula is used unchanged
for all quad cells. For a center triangle with active vertices
\(\mathbf{x}_0=n_{00}\), \(\mathbf{x}_1=n_{10}\), \(\mathbf{x}_2=n_{11}\),
define the exact signed RZ polygon volume
\[
V_{RZ}(P)=\frac{\pi}{3}\sum_k
(R_k Z_{k+1}-R_{k+1} Z_k)(R_k+R_{k+1}).
\]
Let \(\mathbf{x}_c=(\mathbf{x}_0+\mathbf{x}_1+\mathbf{x}_2)/3\). Corner
\(k\in\{0,1,2\}\) owns the subpolygon
\[
Q_k = [\mathbf{x}_k,\;(\mathbf{x}_k+\mathbf{x}_{k+1})/2,\;
       \mathbf{x}_c,\;(\mathbf{x}_{k-1}+\mathbf{x}_k)/2],
\]
with cyclic indices in the same signed orientation as the parent triangle.
Stage-2 tri_fan hydro assigns
\[
m_k=\Delta M_c\frac{V_{RZ}(Q_k)}{V_{RZ}(T)},\quad k=0,1,2,\qquad m_3=0.
\]
The \(Q_k\) tile \(T\), and the axisymmetric volume integrand is linear in
\(R\), so the signed sub-volumes sum to \(V_{RZ}(T)\). Thus the three active
corner masses sum to \(\Delta M_c\) up to roundoff and are nonnegative for
valid tri_fan center geometry. Node-mass gathering skips inactive slot 3.

**Lagrangian invariance of subzonal corner masses (compatible-energy theorem):**

The R-weighted subzonal corner masses must be Lagrangian-invariant during the
hydro step (\(dm_c^p/dt = 0\)) for the Caramana-Burton-Shashkov-Whalen
compatible energy theorem to hold. Corner masses are therefore:

- Computed once at initialization from the initial Lagrangian geometry:
  \(m_c^p = w_L^0 \Delta M_c\) for inner corners or
  \(m_c^p = w_R^0 \Delta M_c\) for outer corners, where \(w_L^0,w_R^0\)
  are R-weights derived from the initial node radii.
- Stored in `state.corner_mass` with length \(4N_{cell}\), layout
  `c*4+{0,1,2,3}` for \((n_{00},n_{10},n_{11},n_{01})\).
- For `single_block` and `tri_fan`, used by `compute_node_mass_2d_kernel`
  via direct atomic accumulation into `node_mass`; the kernel does not
  recompute weights from current \(R\).
- For multiblock topology, node mass uses reverse cell-node CSR and a node-loop
  deterministic gather. Each node accumulates its incident corner masses in the
  fixed `ReverseCellNodeCSR` order sorted by `(cell_id_stable, corner_index)`,
  then assigns `node_mass[n]`; no cell-to-node atomic scatter is used.
  The gather is in grams [g] and is deterministic on GPU because every node
  owns a fixed incident-corner interval and accumulation order.
  `Numerics.hydro.axis_node_mass_convention` selects the multiblock CSR axis
  convention: the default `"corner_subzonal"` sums the exact R-weighted
  subzonal corner masses, while `"equal_split"` assembles only \(R=0\) node
  masses from incident-cell shares (\(m/4\) for quads, \(m/3\) for triangles),
  and `"equal_split_all"` applies those equal-split shares to every node as the
  full structured-convention distribution, the diagnostic superset of `"equal_split"`.
  Under the variational endpoint drive split, the R-weighted convention gives
  the exact 4/3 axis-node over-acceleration that seeds the pole impedance;
  `"equal_split"` matches the structured polar convention without changing
  non-axis node masses or any corner-mass cache. See
  `docs/design/bug25_csr_pole_axis_node_dynamics_20260720.md`.
  After CSR ALE remap, finite active multiblock node sums at the compatible
  acceleration mass floor are positivity-floored as a roundoff guard.  For the
  five-block half-butterfly central Cartesian core only, a roundoff-scale
  non-positive reverse-CSR sum may occur at an axis/core node after compatible
  corner-mass remap.  If the cancellation is no more negative than
  \(10^{-8}\) of the incident-cell control-volume mass, the nodal denominator is
  replaced by \(\frac{1}{4}\sum_{c\ni n}\Delta M_c\); larger negative sums remain
  non-positive and are rejected by the compatible pre-acceleration validator.

  With `Numerics.hydro.bbs_axis_policy_enabled=True`, the five-block
  \(R=0\) axis fallback is Barlow-Burton-Shashkov RZ-compatible instead of
  planar.  For an axis node \(n\) whose incident corner-mass sum reaches the
  compatible acceleration floor, the fallback denominator is
  \[
  m_n^{\mathrm{BBS}} =
    \sum_{(c,k)\in\mathcal{I}(n)} m_{c,k}^{\mathrm{BBS}},
  \qquad
  m_{c,k}^{\mathrm{BBS}} =
    \Delta M_c\frac{V_{RZ}(Q_{c,k})}
                     {\sum_{j=0}^3 V_{RZ}(Q_{c,j})}.
  \]
  Here \(Q_{c,k}\) is the same corner-owned subpolygon used by the compatible
  multiblock corner-mass partition, and
  \[
  V_{RZ}(Q)=2\pi A(Q)\bar R(Q)
  =\frac{\pi}{3}\left|
    \sum_{a\in Q}(R_aZ_{a+1}-R_{a+1}Z_a)(R_a+R_{a+1})\right|.
  \]
  Therefore a subzone whose centroid radius tends to the axis has
  \(V_{RZ}\to0\) and \(m_{c,k}^{\mathrm{BBS}}\to0\).  If the subpolygon-volume
  sum is non-positive or non-finite, this opt-in branch leaves the denominator
  non-positive; it does not use the planar \(1/4\) fallback.

  The compatible axis contract uses this same corner/subzonal mass partition
  for inertia and energy accounting:
  \[
  K=\sum_c\sum_{k=0}^3\frac{1}{2}m_{c,k}
    |\mathbf{u}_{n(c,k)}|^2,\qquad
  E=\sum_c\Delta M_c(e_{e,c}+e_{i,c})+K.
  \]
  Compatible pressure, subzonal-pressure, and edge-AV forces are assembled on
  the same cell-corner indices \(k\).  Their work update uses the matching
  corner velocities,
  \[
  \dot E_{\mathrm{int},c}
    =-\sum_k(\mathbf{F}^{p}_{c,k}+\mathbf{F}^{sub}_{c,k})
          \cdot\mathbf{u}_{n(c,k)}
      +\dot E^{AV}_c,
  \]
  while nodal acceleration uses
  \(\mathbf{a}_n=\sum_{(c,k)\in\mathcal{I}(n)}\mathbf{F}_{c,k}/m_n\) with
  \(m_n=\sum_{(c,k)\in\mathcal{I}(n)}m_{c,k}\), replaced only at the opt-in
  \(R=0\) floor by \(m_n^{\mathrm{BBS}}\) above.  Thus \(K\), force-derived
  acceleration, force work, and compatible total-energy accounting reference
  one subzonal mass partition at the axis.

For pure Lagrangian motion this preserves the column-invariance fix (axis-row
z-acceleration symmetry) while keeping \(dm_c^p/dt=0\). For radial motion, the
previous current-geometry recomputation introduced a missing
\(\frac{1}{2}\sum_p \dot m_p |\mathbf{v}_p|^2\) energy term; cached corner
masses remove that defect. The H3-A energy regression observed before this fix
was `energy_err_pct = 0.18%`; after this fix the ALE-off validation diagnostic
should be limited by the remaining discrete update and artificial-viscosity
terms rather than by mass redistribution.

Option-2-a subzonal-aware remap is used for compatible multiblock ALE
(`compatible_force_work_enabled(cfg) && mesh_topo_is_multiblock(cfg.mesh)`),
corner masses are remapped by mass-conserving subcell reconstruction instead of
post-remap geometry recomputation.  Before the CSR remap, each active cell forms
dimensionless corner fractions
\[
f_{c,k}^{n}=\frac{m_{c,k}^{n}}{M_c^{n}},\qquad k\in A_c,
\]
where \(A_c=\{0,\ldots,\texttt{cell\_nverts}_c-1\}\).  For legacy quads
\(|A_c|=4\); for cap triangles \(|A_c|=3\) and storage slot 3 is inactive.
The remap carries the passive numerator \(S_{c,k}=M_c f_{c,k}\) through the
same CSR donor, swept-volume, and limiter machinery used by the cell mass
transport.  For a face with donor \(d\) and signed swept volume \(\Delta V_f\),
the first-order contribution is
\[
F^{S}_{f,k}=f_{d,k}^{n}\,\rho_d^{n}\,\Delta V_f,
\]
and the second-order path uses the same limited reconstructed mass density
\(\rho_f^{lim}\) with a separately limited reconstructed fraction
\(f_{f,k}^{lim}\):
\[
F^{S}_{f,k}=f_{f,k}^{lim}\,\rho_f^{lim}\,\Delta V_f.
\]
After the conservative mass remap gives \(M_c^{n+1}\), TENRYU reconstructs
\[
\tilde f_{c,k}^{n+1}=\frac{S_{c,k}^{n+1}}{M_c^{n+1}},\qquad
g_{c,k}^{n+1}=
\begin{cases}
\tilde f_{c,k}^{n+1}/\sum_{j\in A_c}\tilde f_{c,j}^{n+1},&
\sum_{j\in A_c}\tilde f_{c,j}^{n+1}>\epsilon_f,\\
1/|A_c|,&\text{otherwise},
\end{cases}
\]
with \(\epsilon_f=10^{-300}\), then writes
\[
m_{c,k}^{n+1}=M_c^{n+1}g_{c,k}^{n+1}.
\]
The last active corner is written as the residual after the previous active
corners so \(\sum_{k\in A_c} m_{c,k}^{n+1}=M_c^{n+1}\) to roundoff by
construction; inactive slots are zero.  Legacy single-block, tri_fan, and
compatible-off multiblock ALE retain the existing post-remap geometry
recomputation path.

The Option-B test-invoked velocity-momentum packet remap has an opt-in
flux-corrected form.  The default Stage-2 entry remains the pure high-order
packet.  For a swept packet of mass \(dm_q\) at \(x_q^*\), the high-order
packet debits donor corner \(a\) by \(dm_q\lambda_a^D\) with donor nodal
momentum and credits receiver corner \(b\) by
\[
dm_q\lambda_b^R\left(u_D(x_b^R)+
u_D(x_q^*)-\sum_j\lambda_j^R u_D(x_j^R)\right).
\]
The low-order packet uses the same mass debit/credit and credits every receiver
corner with the bounded donor value \(u_D(x_q^*)\).  The applied packet is
\[
F_q=F_q^{LO}+\alpha_q(F_q^{HO}-F_q^{LO}),\qquad 0\le\alpha_q\le1,
\]
with one synchronized \(\alpha_q\) for mass, \(R\)-momentum, and \(Z\)-momentum.
Stage 3b bounds receiver corner velocities by the donor cell's own projected
nodal min/max for each component; the expanded donor-stencil bound is deferred
to the later wiring stage.  Both \(F^{LO}\) and \(F^{HO}\) conserve packet mass
and momentum, so the synchronized blend conserves them exactly up to floating
point roundoff.  For affine velocity fields whose packet is admissible, the
limiter returns \(\alpha_q=1\), leaving the high-order convergent mode
unchanged.

For tri_fan Stage 2, the same exact three-corner partition is also used by
ALE mass/kinetic/projection accounting, output `hydro/node_mass` recompute,
and 2D energy-budget kinetic accounting. Stage 3 additionally uses three active
nodes for center-cell velocity averaging during conservative reference remap.

tri_fan explicitly rejects four-corner adjuncts that still require separate
center-cell design: anti-hourglass
`Numerics.hydro.hourglass.enabled`, HLLC z-flux
`Numerics.hydro.hllc_z_flux_2d_rz`, precise RZ geometric CFL
`Numerics.hydro.rz_geometric_cfl_precise_u_half_enabled`, and total-material
energy recovery `Numerics.hydro.total_energy_remap_2d_rz`. VNR artificial
viscosity and first-order conservative reference remap are supported I1-B paths.

The default-off 2D_RZ anti-hourglass force (§3.2.9b) uses separate
`state.subzonal_mass_corner{0,1,2,3}` fields.  They are initialized from the
same R-weighted Pappus quadrature at the first enabled 2D hydro step and are
held fixed through pure Lagrangian motion.  An accepted ALE rezone/remap is the
only in-run operation that recomputes these anti-hourglass subzonal masses,
using the remapped cell mass and current post-remap mesh.

#### 3.2.5 運動量方程式（compatible コーナー力）

\[
m_n \frac{d\mathbf{v}_n}{dt} = \sum_{c\in\mathcal{C}(n)} \mathbf{F}_{c\to n}
\]

**コーナー力**（exact 体積勾配 — variational 契約）：
\[
\mathbf{F}_{c\to n_k} = +(P_c + Q_c)\,\mathbf{S}_{c,k},
\qquad
\mathbf{S}_{c,k}=\frac{\partial V_c}{\partial\mathbf{x}_{c,k}}\ （\text{§3.2.6}）
\]
ここで \(P_c = P_{i,c} + P_{e,c}\)（イオン圧力＋電子圧力）、\(Q_c\) は人工粘性。
\(\mathbf{S}\) は体積増加方向を向くので圧力力は \(+\) 符号で外向きに働き、
\(\sum_n\mathbf{u}_n\cdot\mathbf{F}_n=+\sum_c(P_c+Q_c)\dot V_c\)、セル内部エネルギー仕事は
\(-\sum_k\mathbf{F}_{c,k}\cdot\mathbf{u}_{n_k}=-(P_c+Q_c)\dot V_c\)（圧縮で加熱）である。
（2026-07-26 スペック監査 k02 F-03/§15-3,4: 旧記述 \(-(P_c+Q_c)\mathbf{S}_{c,k}\)（Wilkins
内向き規約）は実装と不一致だった。実装は全力経路で \(+p_q\,\mathbf{S}_{c,k}\)。）

**FIX2-W1/W2 area-weighted symmetric RZ momentum (v1, default off).**
`Numerics.hydro.rz_momentum_scheme="area_weighted_symmetric"` selects the
structured single-block or all-quad multiblock-CSR scalar-VNR momentum
operator.  The default
`"volume_weighted"` retains the existing cached exact-RZ `Svec` statements;
template dispatch and `if constexpr` keep that kernel path byte-identical.
For a quad with physical-outward planar corner vectors,
\[
 \mathbf{S}^{A}_{c,k}={1\over2}
 \begin{pmatrix}z_{k+1}-z_{k-1}\\r_{k-1}-r_{k+1}\end{pmatrix},
 \qquad \sum_k\mathbf{S}^{A}_{c,k}=0.
\]
Let \(A_{41},A_{12},A_{23},A_{34}\) be the four triangles formed by each
quad edge and the arithmetic quad centroid.  The planar subzonal areas use
the Barlow--Burton--Shashkov--Wendroff Fig. 5 quadrature,
\[
 A_c^1={5A_{41}+5A_{12}+A_{23}+A_{34}\over12},\qquad
 A_c^2={A_{41}+5A_{12}+5A_{23}+A_{34}\over12},
\]
with the corresponding cyclic formulas for corners 3 and 4.  Version 1 uses
the cell density for every subzone, \(\rho_c^k=\rho_c\), so
\[
 \langle\rho A\rangle_n=\sum_{c\in\mathcal C(n)}\rho_c A_c^n,
 \qquad
 {d\mathbf v_n\over dt}=
 {\sum_{c\in\mathcal C(n)}(P_c+Q_c)\mathbf S^A_{c,n}
  \over\langle\rho A\rangle_n}.
\]
The 12-weight rule satisfies
\(\sum_k r_k A_c^k=V_c^{RZ}/(2\pi)\), and the planar nodal mass remains
strictly positive at \(r=0\); no axis density-copy special case is used.
Boundary pressure and reflect-mirror forces use the same planar half-edge
convention under this scheme, so the drive scales correctly at any radius.
On the CSR path, acceleration divides by the planar nodal mass, so
`axis_node_mass_convention` affects only the true nodal mass retained for
kinetic energy and diagnostics, not this acceleration.  The
`multiblock_outer_svec_tangent_balance` pass changes only cached RZ `Svec`;
scheme 1 forms planar corner vectors from node coordinates on the fly, so the
pass does not affect its force path.  Morph and barrier machinery remains
unchanged at the geometry level.

This v1 momentum operator remains paired with the existing non-compatible
geometric \(P\,dV\) internal-energy update, where \(dV\) is the true RZ cell
volume change.  It is therefore not exactly total-energy compatible and has
the same conservation class as the legacy non-compatible path.  An exactly
compatible area-weighted force/work pairing is future work with the CSW
compatible-path port.  Version 1 rejects dimensions other than 2D_RZ,
non-`scalar_vnr_legacy` AV, and enabled subzonal pressure.  The CSR port
accepts all-quad multiblock meshes; tri-fan-cap topology is rejected at
namelist validation, and a launch guard asserts that every CSR cell has four
active vertices.
KE/momentum diagnostics, timestep estimates, and remap continue to use the
true RZ `node_mass`.

For tri_fan center cells, Stage-1 geometry writes
\(\mathbf{S}_{c,3}=0\). Therefore the inactive `n01` slot receives zero force
and contributes zero to the geometric volume-rate sum; force and dVdt kernels
are not rewritten in Stage 2. The active apex slot 0 has nonzero
\(\mathbf{S}_{c,0}\), so every `NODE_CENTER` node is pinned in hydro:
acceleration is zeroed, velocity is zeroed after predictor and corrector
updates, and committed position is written exactly \((R,Z)=(0,0)\) after both
predictor and corrector position commits.

For `logical_mesh_2d="spherical_polar_halfplane"`, the theta endpoints
\(\theta=0,\pi\) are the cylindrical \(Z\)-axis, not rectangular \(Z\)-planes.
Nodes marked `NODE_POLE_AXIS` therefore first use pole-axis velocity
constraints: `REFLECT` and `STATE_SUPPLY` zero only \(u_R\) and leave \(u_Z\)
free for pure pole-axis nodes, while `FIXED` zeros both components. In the
multiblock dispatch, nodes tagged as the physical outer shell receive the
`r_outer` branch described below. Internal block-seam nodes carry no
`NODE_BOUNDARY` flag in any current builder (multiblock flags mark only the
cylindrical axis, the cap center, and the outer physical shell) and receive
no vector constraint — conforming shared-node topology alone couples the
blocks (2026-07-26 audit k02 F-05). `NODE_CENTER`
pins both components at the tri_fan origin and takes priority. The same
ordering is used after ALE cell-to-node velocity projection and
reference/conservative remap projection.

`Mesh.polar_theta_min` generalizes the single-block tri-fan angular ladder to
\([\theta_{\min},\pi]\). The default `0.0` executes the historical full-span
construction bitwise unchanged. For a positive value, the `j=0` nodes are not
marked `NODE_POLE_AXIS`; their edges carry `PolarCutFace`, while the
\(\theta=\pi\) side keeps its pole-axis role. In W3a-1, standalone truncated
meshes treat the cut ray as a mirror plane; the FIREX compound replaces it with
the collar interface. Version 1 accepts this option only for
`logical_mesh_2d="spherical_polar_halfplane"` with the tri-fan center.

For S1/S2 multiblock velocity constraints, `node_flags` are applied in one
kernel with combined ordering: `NODE_CENTER` zeros \((u_R,u_Z)\) and returns;
`NODE_AXIS|NODE_POLE_AXIS` zeros only \(u_R\). Nodes carrying
`NODE_OUTER_PHYSICAL_BOUNDARY` are then dispatched by
`Numerics.hydro.boundary_2d.r_outer`; non-outer axis nodes return with
\(u_Z\) unconstrained:

| `r_outer` | physical outer-shell velocity/acceleration constraint |
|---|---|
| `"fixed"` | zero both components |
| `"reflect"` | remove the spherical-normal component and preserve tangent motion |
| `"free"` | no vector constraint |
| `"pressure"` | no vector constraint, allowing pressure-BC normal drive |
| `"state_supply"` or unknown | remove the spherical-normal component as the conservative fallback |

The reflect/fallback projection uses the local spherical normal
\(\hat n=(R/s,Z/s)\), \(s=\sqrt{R^2+Z^2}\):
\[
u_n = u_R {R\over s}+u_Z {Z\over s},\qquad
\mathbf{u}\leftarrow\mathbf{u}-u_n\hat n .
\]
The projection is guarded by \(s>0\). No further branch follows: non-outer
axis nodes keep their axial velocity on the cylindrical \(Z\)-axis, and no
seam node carries `NODE_BOUNDARY`, so no seam projection exists (the
historical unreachable seam tangent-projection fallthrough was removed —
2026-07-26 audit k02 F-05). This dispatch is only in the multiblock boundary
path; `single_block` and `tri_fan` paths keep their existing boundary
kernels.

The multiblock acceleration boundary constraint uses the same outward normal
and combined ordering before the velocity update. `NODE_CENTER` zeros
\((a_R,a_Z)\) and returns; `NODE_AXIS|NODE_POLE_AXIS` zero \(a_R\);
`NODE_OUTER_PHYSICAL_BOUNDARY` nodes use the same five-branch `r_outer`
dispatch above and return. Nothing else is constrained: pure axis nodes
preserve their axial component, pole-outer corner nodes receive both the
axis constraint and the `r_outer` physical-shell dispatch, and internal
seam nodes are untouched (2026-07-26 audit k02 F-05).

For multiblock topology, the pressure and AV force assembly is the same
compatible work discretization on CSR connectivity:
\[
\mathbf{F}_{c\to n_k}=+(P_c+Q_c)\mathbf{S}_{c,k},
\qquad
\mathbf{S}_{c,k}=\frac{\partial V_c}{\partial\mathbf{x}_{c,k}},
\qquad
\dot V_c=\sum_k\mathbf{u}_{n_k}\cdot\mathbf{S}_{c,k}.
\]
Node-loop reverse CSR accumulates the incident corner forces in sorted
`(cell_id_stable, corner_index)` order, so no `atomicAdd` is used.  With
cell-centered \(p_q=P_e+P_i+Q_{visc}\), the discrete identity
\[
\sum_n\mathbf{u}_n\cdot\mathbf{F}_n
=\sum_c p_{q,c}\dot V_c
\]
matches the pressure/AV work term consumed by the energy update.  Units are
force [dyne], velocity [cm/s], and work rate [erg/s].
For active-slot multiblock cells, all compatible force/work sums use
\(A_c=\{0,\ldots,n_c-1\}\) with \(n_c=\texttt{cell\_nverts}[c]\) when the
array is present, otherwise \(n_c=4\).  Cap triangles have \(n_c=3\);
storage slot 3 and inactive local face 0 are not part of the CSR
force/work sums.  Local faces \(1,2,3\) use the canonical active triangle
edges \((1,2),(0,1),(2,0)\).  Quad cells retain the legacy four-corner
local-face ordering used by the compatible edge-force and work buffers.

For the S1/S2 multiblock spherical outer shell, the raw Pappus-polygon
\(\mathbf{S}_{c,k}\) has a non-zero uniform-pressure tangential residual at
non-pole outer-boundary nodes because the circular shell is represented by
straight RZ chords.  After the raw multiblock Svec computation, each non-pole
outer node \(n=(q=N_r,k)\) balances the two incident outer-shell corner vectors
\(\mathbf{S}_L=\mathbf{S}_{(N_r-1,k-1),2}\) and
\(\mathbf{S}_R=\mathbf{S}_{(N_r-1,k),1}\).  With
\(\hat t=(Z/s,-R/s)\), \(s=\sqrt{R^2+Z^2}\), define
\(\tau=(\mathbf{S}_L+\mathbf{S}_R)\cdot\hat t\), then update
\[
\mathbf{S}_L \leftarrow \mathbf{S}_L-\frac{\tau}{2}\hat t,\qquad
\mathbf{S}_R \leftarrow \mathbf{S}_R-\frac{\tau}{2}\hat t .
\]
This leaves the pair normal sum unchanged and preserves the antisymmetric
pressure-gradient part, while enforcing
\((\mathbf{S}_L+\mathbf{S}_R)\cdot\hat t=0\) for uniform pressure.  The same
post-balance Svec arrays are used by force assembly and \(\dot V_c\), so the
compatible-work identity above remains exact for the implemented
discretization.  This correction is multiblock-only; ordinary annular
`single_block` and `tri_fan` Svec paths are unchanged.
The tangent-balance closure is applied on every multiblock geometry recompute
when `Mesh.multiblock_outer_svec_tangent_balance=true` (the default), and is
statically load-bearing for the I1-B S3-T3 G1 constant-state seam-GCL gates.
Under strong drive it deletes tangential restoring forces on the outer arc
every step and destabilizes the pole-adjacent outer cells.  Drive decks should
therefore set the knob to `false` until the planned boundary-acceleration
projection replaces this geometry correction.

For the opt-in single-block `polar_center_treatment="button"` topology, the
synthetic button polygon and active structured shell cells use the same
clockwise canonical spherical-polar sign, \(-1\), for their Hydro Svec arrays.
Dormant inner-ring cells keep zero Svec.  This matches the
candidate-admissibility sign convention and makes the button seam and
`r_outer` pressure-boundary force pairs opposite under uniform pressure.

#### 3.2.6 面積ベクトル（RZ幾何 — exact Pappus 体積勾配）

セル \(c\) のノード \(k\)（循環添字）に対する面積ベクトルは、Pappus 体積公式
（§3.2.2）の**正確な偏微分**である：
\[
\mathbf{S}_{c,k} \equiv \frac{\partial V_c}{\partial \mathbf{x}_{c,k}},
\qquad
\frac{\partial V_c}{\partial r_k} = \frac{\pi}{3}\bigl[2r_k(z_{k+1}-z_{k-1}) + r_{k+1}(z_{k+1}-z_k) + r_{k-1}(z_k-z_{k-1})\bigr],
\]
\[
\frac{\partial V_c}{\partial z_k} = \frac{\pi}{3}\bigl[r_{k-1}(r_{k-1}+r_k) - r_{k+1}(r_k+r_{k+1})\bigr].
\]

> **実装（桁落ち回避）**：§3.2.2 と同じくセル重心 \((\bar r_c,\bar z_c)\) への平行移動
> 座標で評価する。\(\tilde r_k = r_k-\bar r_c\)、\(\tilde z_k = z_k-\bar z_c\) として
> \[
> S_{r,k} = \frac{\pi}{3}\Bigl[2\tilde r_k(\tilde z_{k+1}-\tilde z_{k-1}) + \tilde r_{k+1}(\tilde z_{k+1}-\tilde z_k)
>   + \tilde r_{k-1}(\tilde z_k-\tilde z_{k-1}) + 3\bar r_c(\tilde z_{k+1}-\tilde z_{k-1})\Bigr],
> \]
> \[
> S_{z,k} = \frac{\pi}{3}\Bigl[\tilde r_{k-1}(\tilde r_{k-1}+\tilde r_k) - \tilde r_{k+1}(\tilde r_k+\tilde r_{k+1})
>   + 3\bar r_c(\tilde r_{k-1}-\tilde r_{k+1})\Bigr].
> \]
> 展開すると上の非平行移動式と厳密に等しい（\(z\) の差分が \(\bar z_c\) を消去し、
> \(r\) 二次項の \(\bar r_c\) 依存が \(3\bar r_c\) 項に集約される）。任意の平行移動
> 基準点で恒等（重心の選択は丸め誤差制御のみ）。全 Svec 生成経路 — structured
> （`recompute_geometry_2d_kernel`）/ polar / tri_fan / CSR multiblock / button —
> が同一のこの式を共有する（`rz_polygon_svec`、`rz_polygon_svec_exact`）。

**離散恒等式**（exact-gradient 契約が満たすもの・満たさないもの）：

- \(\sum_k S_{z,k} = 0\)（厳密：\(z\) 一様平行移動は回転体体積を変えない）。
- \(\sum_k S_{r,k} = 2\pi A_c\)（RZ 断面積。**ゼロではない** — \(r\) 一様シフトは
  回転体体積を増やす。これが円筒座標の幾何ソース（hoop）項の離散対応であり、
  1D_SPH の \(A_{j+1}\neq A_j\) と同じ役割を 2D_RZ で果たす）。
- 一様圧力の力の釣り合いは**セル単位ではなく内部ノード単位**で成立する：内部
  ノード \(n\) の変位は隣接セル群の総体積を変えないから
  \(\sum_{c\ni n}\mathbf{S}_{c,k(n)} = \partial\bigl(\sum_{c\ni n}V_c\bigr)/\partial\mathbf{x}_n = \mathbf{0}\)。
  G2 gate はこのノード単位 cancellation を検査する。旧記述の「\(\sum_k \mathbf{S}_{c,k}
  =\mathbf{0}\) を機械精度で満たす」はセル単位の主張であり、exact-gradient 契約では
  **成立しないし要求もしない**（2026-07-26 スペック監査 k02 F-01/§15-1,2 で訂正。
  contract lock: tests/hydro/test_rz_svec_exact_gradient.cu）。
- \(\dot V_c = \sum_k \mathbf{u}_{n_k}\cdot\mathbf{S}_{c,k}\) は chain rule により厳密（§3.2.7、GCL）。

> **設計注記（履歴・保存量・境界）**：v1.0 初期の Wilkins 面積重み
> （\(\sum_k\mathbf{S}=\mathbf{0}\) 型。exact 勾配への移行を C-06 として延期していた）
> は退役済みで、現行実装は上記 exact 勾配に統一されている（C-06 完了）。
> この離散化では**スカラー radial 運動量 \(\sum_a m_a u_{r,a}\) は保存量ではない**：
> \(dP_r/dt = \sum_c p_c\,2\pi A_c + （境界・拘束項）\) の hoop ソースを持つ
> （axisymmetric Euler の \(\partial_t(\rho u_r)+\cdots = p/r\) に対応）。診断の
> `R_momentum_residual` は静的力平衡問題でのみ自明にゼロとなる監査量であり、
> 動的 RZ 流れの保存則ゲートには使えない（2026-07-26 監査 k02 F-06/§15-11）。
> 境界力：spherical-polar 外殻の圧力境界は exact RZ endpoint traction
> \(\mathbf{G}_a=\frac{\pi}{3}(2r_a+r_b)(\Delta z,-\Delta r)\)、
> \(\mathbf{G}_b=\frac{\pi}{3}(r_a+2r_b)(\Delta z,-\Delta r)\) を常時使用する
> （`rz_exact_endpoint`：`logical_mesh_2d="spherical_polar_halfplane"` で有効。
> 旧 C-07「境界 \(2\pi r\) 係数欠落の延期」はこの経路では解消済み。矩形 RZ
> 境界経路は従来のまま — 2026-07-26 監査 k02 §7.3/§15-8 で記述を統一）。
> 輻射輸送の面幾何（§6.3.2, §7.3.2）は独立に修正する（Hydro Svec に依存しない）。

#### 3.2.7 体積変化率

セル体積の時間変化率（運動学的関係）：
\[
\frac{dV_c}{dt} = \sum_{k=1}^{4} \mathbf{v}_k \cdot \mathbf{S}_{c,k}
\]

#### 3.2.8 速度発散

\[
\nabla\cdot\mathbf{u}_c = \frac{1}{V_c}\frac{dV_c}{dt}
\]

#### 3.2.9 人工粘性（2D拡張）

von Neumann–Richtmyer型（§3.1と同一の形式を2Dに拡張）：
\[
Q_c = \begin{cases}
\rho_c\left(C_2^2(\Delta l_c)^2(\nabla\cdot\mathbf{u}_c)^2 + C_1\,\Delta l_c\,c_{s,c}\,|\nabla\cdot\mathbf{u}_c|\right) & \nabla\cdot\mathbf{u}_c<0\\
0 & \text{otherwise}
\end{cases}
\]

- 特性セル長：\(\Delta l_c = \sqrt{A_c}\)（§3.2.3の断面積から）
- \(\nabla\cdot\mathbf{u}_c\) は§3.2.8で評価
- 既定：\(C_1=0.1,\; C_2=1.5\)

> **注**：2次項の係数が \(C_2^2\) であることに注意。一部の文献では \(C_2\) を直接使う記法もあるが、
> TENRYUは \(C_2^2\) 形式を採用する（Wilkins型の慣用に従う）。

**Topology-aware q-cap dispatch.**
`Numerics.hydro.av_qcap_over_p = k > 0` のとき、VNR artificial viscosity
producer は cell-centered storage へ書き込む直前に
\[
Q_{c}^{cap}=\min\!\left(Q_{c}^{VNR},\; k\max(P_{e,c}+P_{i,c},0)\right)
\]
を適用する。\(k\le0\)（既定）では \(Q_c^{cap}=Q_c^{VNR}\) で、cap kernel は
起動しないため default-off path は byte-identical である。\(P_{e,c}+P_{i,c}\le0\)
では \(Q_c^{cap}=0\) とし、vacuum/cold cell には capped AV を入れない。
対象 cell は `av_qcap_scope` で選ぶ。`"global"` では全 active cell、
`"tri_fan_radial_index"` では `logical_mesh_2d="spherical_polar_halfplane"` かつ
`polar_center_treatment="tri_fan"` の center-band cell
\(i\le\)`tri_fan_center_cfl_band_radial_index`、`"centroid_r_le_r_match"` では
cell centroid \(R_c\le\)`multiblock_cart_core_r_match` の cell のみ対象にする。
tri_fan radial-index scope を tri_fan 以外の topology で指定した場合、cap は
legacy compatibility として no-op である。legacy `av_qcap_center_band_only=true`
は namelist builder で `"tri_fan_radial_index"` に写像され、runtime は
`av_qcap_scope` のみを参照する。Scope keys and legacy-bool compatibility are
cataloged in SPECIFICATION §6.4.2 and §9.1.

Cap は `state.Qvisc` storage point で一度だけ適用され、`hydro_2d.cu` の
4つの AV producer call site（`compute_Q_2d` /
`compute_artificial_viscosity_2d`; implementation sites around
`src/hydro/hydro_2d.cu`:3044, 3501, 3859, 4598）の直後に同じ保存値を作る。以後の
force assembly (`build_cell_pq`), Q-only acceleration (`build_cell_q`),
half-step copy (`Q_half` memcpy), energy work
(`energy_update_with_old_volume_2t_kernel`), `update_av_max_diagnostic`, and
`work_split` audit は同じ \(Q_c^{cap}\) を読む。したがって
\[
F_Q=-Q_c^{cap}{\partial V_c\over\partial x},\qquad
W_Q=-Q_c^{cap}\dot V_c
\]
であり、discrete compatible-hydro identity
\(F\cdot u=Q_c^{cap}\dot V_c=W_Q\) は同一 scalar storage により保たれる。
force と work が異なる \(Q\) を読む非対称性は構造上発生しない。
`per_material_hydro` では material scratch `Qvisc_per_material[c,m]` も
\[
f_c={Q_c^{cap}\over\max(Q_c^{raw},10^{-300})}
\]
で同率に scale し、aggregate force/work consistency を保つ。

#### 3.2.9a Hourglassモード制御方針

2D四角形Lagrangianメッシュではゼロエネルギーモード（hourglassモード）が発生しうる。
既定では `Numerics.hydro.hourglass.enabled=False` とし、既存検証の
bitwise 経路を変更しない。  2D_RZ の bulk-cell static mesh degeneracy
対策として、opt-in の Caramana-Shashkov 型 subzonal pressure force
（§3.2.9b）を用意する。

#### 3.2.9b Anti-hourglass subzonal pressure force (Caramana-Shashkov)

本項は Caramana, Burton, Shashkov, Whalen, J. Comput. Phys. 146, 227-262
(1998) の compatible energy 形式、および Caramana and Shashkov,
J. Comput. Phys. 142, 521-561 (1998) の subzonal masses and pressures
hourglass 抑制に従う。  これは速度フィルタではなく、セル内 subzone
体積変化に反応する pressure perturbation force である。

When `Numerics.hydro.subzonal_band_mode="bridge_feather"`, a single cell
weight multiplies every compatible subzonal-pressure corner force: bridge
cells with `cell_block_id=1` have \(w=1\); at face-adjacency BFS layer
\(\ell=1,\ldots,L\), \(u=1-\ell/(L+1)\) and
\(w=6u^5-15u^4+10u^3\); and \(w=0\) for \(\ell>L\), where
\(L=\texttt{subzonal\_band\_feather\_layers}\). Applying one scalar to all
corner forces of a cell preserves their zero-net-force identity, and the
compatible-work path consumes the same scaled force arrays. The default
`"off"` mode builds no weight cache and retains the byte-identical global path.

##### S-D2 predictive corner-Jacobian timestep limiter

The default-off `Numerics.hydro.corner_j_predict_cfl_enabled` limiter
extrapolates each scoped quadrilateral corner with the current node velocity.
For corner edges \(e_1,e_2\) and velocity differences \(d_1,d_2\),
\[
J(\Delta t)=\operatorname{cross}(e_1+\Delta t d_1,
                                  e_2+\Delta t d_2)
 =J_0+b\Delta t+a\Delta t^2,
\]
where \(b=\operatorname{cross}(e_1,d_2)+
\operatorname{cross}(d_1,e_2)\) and
\(a=\operatorname{cross}(d_1,d_2)\). It requires
\(J(\Delta t)\ge(1-s)J_0\) for each initially positive \(J_0\), with
\(s=\texttt{corner\_j\_predict\_max\_shrink}\). Thus the corner limit is the
smallest positive root of
\(g(\Delta t)=a\Delta t^2+b\Delta t+sJ_0\). The quadratic solve uses
\(q=-[b+\operatorname{sign}(b)\sqrt{b^2-4asJ_0}]/2\) and roots
\(q/a\), \(sJ_0/q\); \(|a|\le10^{-300}\) uses the linear root when
\(b<0\). Corners without a positive finite root impose no limit.

For `multiblock_cart_core_polar_shell`, the host reduction covers cell IDs
\(c<c_{shell}+N_{shell}\,N_\theta\): all Cartesian-core and bridge cells plus
`corner_j_predict_shell_rings` shell rings. The global bound is
`corner_j_predict_cfl_safety` times the minimum corner root, clamped from below
by `corner_j_predict_floor_frac` times the acoustic CFL dt of the same step
(anti-Zeno: a corner whose linear extrapolation demands a smaller step is
accepted at the floor — transient closures that rebound under the true forces
must not freeze the run). Other topologies are inert with a warning, and the
default-off path does not invoke the prediction.

セル \(c\) の反時計回り4ノード位置を
\(\mathbf{x}_{c,k}=(r_{c,k},z_{c,k})\), \(k=0,1,2,3\) とする。
hourglass modal amplitude は bilinear fit の checkerboard 係数：
\[
\mathbf{a}_{\xi\eta,c}
 = \frac{\mathbf{x}_{c,0}-\mathbf{x}_{c,1}
          +\mathbf{x}_{c,2}-\mathbf{x}_{c,3}}{4}.
\]
force は次の2条件を同時に満たす cell だけで active になる：
\[
\frac{\min_k J_{c,k}}{\max_k J_{c,k}}
< q_\mathrm{hg},\qquad
\frac{\|\mathbf{a}_{\xi\eta,c}\|}{\sqrt{A_c}} > a_\mathrm{hg},
\]
ここで \(q_\mathrm{hg}\) は
`activation_corner_j_ratio_threshold`（既定 0.5）、
\(a_\mathrm{hg}\) は `activation_hourglass_amplitude_threshold`
（既定 0.01）である。  inactive cell では force/work ともゼロ。

For `polar_center_treatment="button"`, the central `c=0` cell is an
\(N_\theta+1\)-gon, not a logical quadrilateral, so it is excluded from the
quad anti-hourglass restoring force.  Dormant button-covered cells are also
excluded and their compatible hourglass work is zero.  Active structured shell
cells remain eligible for the ordinary quad hourglass predicate if the
otherwise guarded hourglass option is enabled.  Shock-capturing artificial
viscosity for button decks remains the scalar VNR \(Q\) computed through the
standard divergence/pressure path; no quad hourglass mode is assigned to the
button polygon.

For multiblock cap cells with `cell_nverts[c]=3`, the Phase 3/compatible
anti-hourglass restoring force is also a no-op: a true triangle has no
bilinear checkerboard/hourglass degree of freedom.  The associated
hourglass-compatible work is therefore zero for those cells.  The subzonal
mass and volume scratch arrays still store the three active triangle subzones
and zero inactive slot 3, so node-mass and compatible bookkeeping use the same
active-slot partition without applying a quad hourglass force. With
`subzonal_pressure_enabled=True`, the cap triangle uses those three active
corner subzones for the Caramana-Shashkov pressure perturbation and contributes
compatible force/work from the pressure subzones; only the bilinear
anti-hourglass mode is absent.

各 corner subzone は corner node、隣接2辺の中点、cell center からなる
四辺形 \(s=(c,k)\) とし、RZ 体積は Pappus quadrature で
\[
V_{c,k}^{sub}=2\pi\,\bar r_{c,k}^{sub} A_{c,k}^{sub}
\]
を評価する（実装は §3.2.2 と同じ重心シフト形を使う）。  Phase 3
multiblock path では
\[
m_{c,k}^{sub} = \rho_c^0 V_{c,k}^{sub,0},\qquad
\sum_{k=0}^{3} V_{c,k}^{sub}=V_c,\qquad
\sum_{k=0}^{3} m_{c,k}^{sub}=m_c
\]
を初期化 invariant とする。  single-block legacy `hourglass.enabled`
path も同じ exact-subpolygon partition を使う。Lagrangian step 中は
\[
\frac{d m_{c,k}^{sub}}{dt}=0
\]
とし、`state.corner_mass[c*4+k]` は bitwise に保持する。一方で
`state.corner_volume[c*4+k]` は current mesh の exact-subpolygon volume として
geometry refresh 後に更新されるため、canonical CS subzonal density は
\[
\rho_{c,k}^{sub,n} = m_{c,k}^{sub,0}/V_{c,k}^{sub,n}
\]
を評価できる。  compatible multiblock ALE で rezone/remap を accepted した時は、
§3.2.4 の mass-conserving subcell reconstruction により
`state.corner_mass[c*4+k]` を remapped cell mass に閉じるよう再構築する。
compatible-off path は従来通り post-remap geometry から再初期化する。

`Numerics.hydro.subzonal_mass_enabled=True` は multiblock
Caramana-Shashkov Phase 3 gate である。既定 False では既存 Hydro2D
path と HDF5 schema を変更しない。True では flat runtime fields
`state.corner_mass[c*4+k]` and `state.corner_volume[c*4+k]` を確保し、
reverse cell-node CSR で nodal mass
\[
m_n = \sum_{(c,k):\,n(c,k)=n} m_{c,k}^{sub}
\]
を assemble する。`anti_hourglass_kappa`（既定 0.05）はこの gate の
\(C_{hg}\) として使う。`subzonal_pressure_enabled=True` では Phase 3
anti-hourglass kernel は呼ばず、後述の canonical Caramana-Shashkov
quadrilateral subzonal pressure を compatible force buffer に入れる。
`Numerics.hydro.subzonal_mass_lagrangian_invariant_enabled=True` は同じ
corner-mass invariant を明示的に有効化する default-off compatibility key
であり、`subzonal_mass_enabled=True`、legacy `hourglass.enabled=True`、
または `subzonal_pressure_enabled=True` では runtime effective flag が
自動的に True になる。
Canonical `subzonal_pressure_enabled=True` の production initialization
では、single-block quad も
`compute_quad_corner_masses_partitioned_subpolygon` により
initial median subpolygon volumes と同じ partition から fixed corner
masses を構築する。したがって deck の
`corner_mass_convention` が kinematic/BBSW basis を選んでいても、この
canonical path の初期 subdensity は cell density と一致し、その masses
は Lagrangian step 中に再構築しない。

Stage A compatible force/work path は `av_model="csw_edge"` かつ
`subzonal_pressure_enabled=True`、または `av_model="csw_edge_csw98"`
（subzonal pressure の有効・無効によらず常時 — I1-B column A/B isolation）で
有効である（`compatible_force_work_enabled`）。edge AV の compatible work は
subzonal pressure の enable 状態に依存しない（2026-07-26 監査 k03 F-02/§11-2,3
で dispatch 記述を実装に一致させた）。legacy
`av_model="scalar_vnr_legacy"` かつ `subzonal_pressure_enabled=False` では
従来の scalar \(p_q=P_e+P_i+Q\) nodal force assembly をそのまま使う。
compatible path では cell-corner pressure force、cell-corner subzonal
pressure force、unique-edge AV force を別バッファに分ける：
\[
\mathbf{F}^{p}_{c,k}=p_c\,\mathbf{S}_{c,k},\qquad
\mathbf{F}^{sub}_{c,k}=\delta\mathbf{f}^{sub}_{c,k},\qquad
\mathbf{F}^{av}_{e}=\sum_{c\supset e}\mathbf{f}^{av}_{c,e}.
\]
ここで \(\mathbf{S}_{c,k}\) は既存 Hydro2D の Pappus Svec と同じ符号規約を持つ
authoritative mesh area vector である。multiblock outer-shell Svec balance
後の値を使うため、corner pressure force を node に gather した和は既存の
\(p_c\mathbf{S}_{c,k}\) scalar pressure force と同じである。T4 は
subzonal pressure buffer を populate する。T3 は CSW edge AV force を
edge buffer に、cell work を `state.work_av_per_cell` に populate する。

For each cell edge \(e=(a,b)\), with the edge orientation used by the
edge-force scatter,
\[
\Delta\mathbf{x}_e=\mathbf{x}_a-\mathbf{x}_b,\qquad
\Delta\mathbf{u}_e=\mathbf{u}_a-\mathbf{u}_b,\qquad
\hat{\mathbf{u}}_e={\Delta\mathbf{u}_e\over|\Delta\mathbf{u}_e|}.
\]
\(\hat{\mathbf{u}}_e=0\) when \(|\Delta\mathbf{u}_e|=0\).  The RZ median
edge vector is derived transiently from current corner positions using the
same full-\(2\pi\) Pappus convention as `cell_Svec`:
\[
\mathbf{S}^{RZ}_{c,e}
=\pi(r_0+r_1)\left(z_1-z_0,\;-(r_1-r_0)\right),
\]
where \((0,1)\) is the cell-local outward edge orientation.  (Naming note:
despite the historical label, this vector is the face's own
lateral-revolution area vector, NOT the CSW98 Eq. 16 median-mesh vector;
for logically-grid-aligned compression it is perpendicular to
\(\Delta\mathbf{u}_e\) and the force is structurally zero -- see
docs/design/i1b_csw_edge_av_structural_zero_defect.md. The mode is kept
bit-identical for certification continuity; `av_model="csw_edge_csw98"`
(§3.2.9c) is the corrected formulation.)  The edge is
compressive for that cell iff
\[
\Delta\mathbf{u}_e\cdot\mathbf{S}_{c,e}\le0.
\]
No mesh-distortion gate, corner-J gate, or hourglass-amplitude gate activates
this AV; smooth-flow suppression comes only from the compression test and
the limiter below.

For compressive edges, TENRYU uses the Caramana-Shashkov-Whalen edge form
\[
\mathbf{f}^{av}_{c,e} =
\rho_e\left[
C_2{\gamma+1\over4}|\Delta\mathbf{u}_e|
+\sqrt{\left(C_2{\gamma+1\over4}\right)^2|\Delta\mathbf{u}_e|^2
+ C_1^2 c_{s,e}^2}
\right]
(1-\psi_e)
(\Delta\mathbf{u}_e\cdot\mathbf{S}_{c,e})
\hat{\mathbf{u}}_e .
\]
For T3, \(C_1=\texttt{av\_C1}\) and \(C_2=\texttt{av\_C2}\); when the
namelist selects `av_model="csw_edge"` and omits these keys, the CSW edge
defaults are \(C_1=C_2=1\).  The edge density and sound speed are harmonic
averages of incident-cell values,
\[
\rho_e={2\rho_c\rho_n\over\rho_c+\rho_n},\qquad
c_{s,e}={2c_{s,c}c_{s,n}\over c_{s,c}+c_{s,n}},
\]
with the current cell value used on physical boundaries.  This is the T3
simplification; corner/subzonal density replaces it in the later subzonal
stage.

The Christensen-Caramana limiter is
\[
\psi_e=\max\left(0,\min\left(1,2r_L,2r_R,{r_L+r_R\over2}\right)\right),
\]
where \(r_L,r_R\) compare projected neighboring edge velocity gradients along
the local edge line:
\[
r={(\Delta\mathbf{u}_{nbr}\cdot\hat{\mathbf{u}}_e)/
(\Delta\mathbf{x}_{nbr}\cdot\hat{\mathbf{x}}_e)
\over |\Delta\mathbf{u}_e|/|\Delta\mathbf{x}_e|}.
\]
Roundoff-small denominators set that ratio to 1.  Physical boundaries and
multiblock seam entries whose `face_adj_csr` neighbor is the sentinel `-1`
are treated as missing neighbors and also use ratio 1.
When `csw_limiter_enabled=false`, the debug path sets \(\psi_e=0\) and keeps
the same edge force/work/CFL surfaces.

The CSW AV work scratch is computed from the same cell-edge force,
\[
W^{av}_c=-\sum_{e\in\partial c}
\mathbf{f}^{av}_{c,e}\cdot\Delta\bar{\mathbf{u}}_e ,
\qquad
\bar{\mathbf{u}}={\mathbf{u}^n+\mathbf{u}^{n+1}\over 2},
\]
where the corrector's compatible-work recompute supplies the exact
time-centered velocities.  Because the AV force is velocity-dependent
while the momentum update changes the kinetic energy at the rate
\(\mathbf{F}\cdot\bar{\mathbf{u}}\), per-edge deposits are SIGNED: a small
negative deposit at the averaged velocities is the true KE pairing, and
clamping it non-negative biases the total-energy closure by a
truncation-ordered residual (G4 5-block: 5e-10 at \(N_c=16\), vs
3.5e-16 signed).  Signed zeros and non-finite contributions deposit an
explicit \(+0.0\) (the legacy normalization at zero-force
dormant-boundary edges).  T3 does not update internal energy; the later
compatible-energy stage deposits this AV work to ion energy in 2T.
With the signed pairing the compatible stack closes total energy to
machine round-off INCLUDING the CSW edge AV.

For `av_model="csw_edge"`, the edge-relative AV CFL component is
\[
\Delta t_{AV}=\min_{\Delta\mathbf{u}_e\cdot\mathbf{S}_{c,e}\le0}
\texttt{av\_cfl\_coefficient}\,
{|\Delta\mathbf{x}_e|\over|\Delta\mathbf{u}_e|},
\]
and participates in the global hydro timestep reduction.

#### 3.2.9c csw_edge_csw98: CSW-1998-faithful median-mesh edge AV (I1-B Stage-G W1)

`av_model="csw_edge_csw98"` replaces the projection geometry of §3.2.9b
with the actual CSW98 (Caramana-Shashkov-Whalen, JCP 144 (1998) 70, §4)
median-mesh construction; the Kuropatenko kernel, harmonic edge
density/sound speed, edge-force buffers, sum_edge_forces deposit convention
(-F at n0, +F at n1) and the signed compatible work recompute are shared
unchanged. The old mode forks at the host launcher; no old-mode kernel
changes (bitwise regression anchor:
tests/hydro/test_csw98_bit_identity_old_mode.cu).

Side vectors (C2 form, decision 2026-07-04,
docs/design/i1b_csw98_rz_eq16_decision.md): with the exact
revolution-volume corner gradients \(a_k=dV/dx_k\) (same polynomial family
as `cell_Svec`) and \(b_k=a_k-\mathrm{mean}(a)\), the cyclic system
\[
S_{k-1}-S_k=b_k,\qquad \sum_k S_k=0
\]
defines the unique RZ side vectors; they satisfy the deformational Eq. 16
identity
\[
\sum_k S_k\cdot d\mathbf{v}_k
=\sum_k b_k\cdot\mathbf{v}_k
=dV/dt-2\pi A\bar v_r
\]
to roundoff (<= 1e-12 rel, BINDING;
tests/hydro/test_csw98_eq16_identity.cu, quads and tri cells). No
edge-difference form can carry the r-translation part
(\(\sum_k dV/dx_k=(2\pi A,0)\) is invisible to \(d\mathbf{v}\)) -- the
deformational rate is the exactly-representable and Galilean-invariant
compression measure. In the planar (Cartesian) measure the same construction
reduces exactly to the CSW98 median-mesh vectors
\(S=\mathrm{orient\mbox{-}fix}[R(-90^\circ)(c-m)]\) (equivalence asserted
in-test); the Cartesian median identity \(\sum_k S_k\cdot d\mathbf{v}_k=dA/dt\)
is roundoff-exact. The spec's original \(\pi(r_c+r_m)\) median-segment
weighting (C1) remains measured in-tree as a cross-reference: its
deformational residual is first order \(O(dr/r)\) (2.13e-2 at \(r/dr=3\)
down to 7.17e-4 at \(r/dr=100\)), which C2 removes exactly.

Switch and force (Eq. 17/19/20; strict inequality):
\[
\mathrm{fire}\iff d\mathbf{v}_e\cdot S_e<0,
\]
\[
f_e=\rho_e W_e(1-\psi_e)\,
(d\mathbf{v}_e\cdot S_e)\,\hat{d\mathbf{v}}_e,
\]
with \(W_e\) the §3.2.9b Kuropatenko wave speed (\(C_1/C_2\) defaults
1.0/1.0 when omitted, as for `csw_edge`). Dimensions:
\([\rho_e W_e]=\mathrm{g\,cm^{-2}\,s^{-1}}\),
\([d\mathbf{v}_e\cdot S_e]=\mathrm{cm^3\,s^{-1}}\), so \([f_e]=\mathrm{dyn}\)
— the CSW98 Eq. 20 form (equivalently
\(f_e=q_{Kur,e}(1-\psi_e)(\hat{d\mathbf{v}}_e\cdot S_e)\hat{d\mathbf{v}}_e\)
with \(q_{Kur,e}=\rho_e W_e|d\mathbf{v}_e|\)). 旧版は
\((d\mathbf{v}_e\cdot S_e)/|d\mathbf{v}_e|\) と正規化済み \(\hat{d\mathbf{v}}_e\)
を併記した速度因子一つ不足の誤記（実装 `compatible_av_csw.cu` は当初から本式 —
2026-07-26 監査 k03 F-01 で訂正）。Zero-force continuity holds:
\(f_e\to0\) as \(d\mathbf{v}_e\cdot S_e\to0^-\).

Limiter (Eq. 12 + Eq. 18):
\[
\psi_e=\max\left(0,\min\left(1,2r_L,2r_R,{r_L+r_R\over2}\right)\right)
\]
with \(r_L/r_R\) taken from the CONTINUATION edges along the same logical line
through the edge endpoints (structured: the same local face of the \(+/-1\)
neighbor cells along the edge direction; multiblock: a node-membership search
-- step across the perpendicular face at each endpoint and take the neighbor's
face containing that node other than the shared face -- which resolves rotated
seam frames; the ratio is orientation-free since traversal reversal flips
numerator and denominator together).
\[
r={[(d\mathbf{v}_{nbr}\cdot\hat{d\mathbf{v}}_e)|d\mathbf{x}_e|]
\over
[(d\mathbf{x}_{nbr}\cdot\hat{d\mathbf{x}}_e)|d\mathbf{v}_e|]} .
\]
Missing members (physical boundary, inactive neighbor, unresolvable seam
continuation, near-perpendicular continuation = CSW98 right-angle exclusion)
use the CSW98 boundary closure ratio := 1 (the diagnostic counter includes
physical-boundary members). Unresolved-member counts are logged once when
`TENRYU_CSW98_LIMITER_DIAG` is set. Self-similar motion invariance (uniform
compression, rigid rotation) and linear grid shear (\(v_z\) proportional to
\(r\)) all yield Eq. 18 ratios that are EXACTLY 1.0 in FP (identical commuted
products), so \(\psi=1\) and the AV vanishes identically. Note a C2-specific
property: on rectangular cells the C2 side vectors carry an \(O(dr/r)\)
z-component from the RZ hoop asymmetry, so a divergence-free shear can trip
the per-SIDE switch (the per-cell side terms cancel exactly, \(dV/dt=0\) --
the CSW98 Fig. 3 per-edge-switch property); invariance is then enforced by
the limiter, not the switch -- in contrast to §3.2.9b, which fires finite
shear forces that only its (tautological) limiter masks.

AV CFL (force-coupled; replaces the §3.2.9b \(|dx|/|du|\) form for this mode):
\[
\Delta t_{AV}=\min_{\mathrm{firing}\ c,e}
{\texttt{av\_cfl\_coefficient}\,|d\mathbf{x}_e|
\over
|d\mathbf{v}_e|(1-\psi_e)|\hat{d\mathbf{v}}_e\cdot\hat S_e|},
\]
so edges with zero force (no fire, \(\psi=1\), or perpendicular projection)
contribute NO constraint -- eliminating the §3.2.9b failure mode where dt was
limited by an AV doing no work (column-test dt-floor abort). The winner-edge
trace diagnostic is not implemented for this mode in W1.

Work closure, negative-work clipping of the predictor deposit, corrector
signed recompute, 2T ion-energy routing, buffer sizing and the pole tangential
damper are inherited from §3.2.9b unchanged.

canonical CS subzonal pressure は毎 Lagrange cycle で常に評価する
（threshold gate は置かない）。corner density と EOS pressure は
\[
\rho_{c,k}^{sub}={m_{c,k}^{sub}\over V_{c,k}^{sub}(t)},\qquad
p_{c,k}^{sub}=p_\mathrm{EOS}(\rho_{c,k}^{sub},e_c),\qquad
\delta p_{c,k}=p_{c,k}^{sub}-(P_{e,c}+P_{i,c})
\]
である。2T では zone-centered \(e_{e,c},e_{i,c}\) を保持したまま
\(p_{c,k}^{sub}=p_e(\rho_{c,k}^{sub},e_{e,c})+
p_i(\rho_{c,k}^{sub},e_{i,c})\) と評価し、corner energy は導入しない。
四角形 corner \(k\) の force は Caramana-Shashkov Eq. 18 を2倍した形：
\[
\delta\mathbf{f}^{sub}_{c,k}=M_f\left[
2\delta p_{c,k}(\mathbf{a}_{k,L}+\mathbf{a}_{k,R})
+(\delta p_{c,k}-\delta p_{c,k-1})\mathbf{S}_{k-1}^{med}
+(\delta p_{c,k+1}-\delta p_{c,k})\mathbf{S}_{k}^{med}
\right].
\]
実装は等価な telescoping 形
\[
\delta\mathbf{f}^{sub}_{c,k}=M_f\left[
(\delta p_{c,k}+\delta p_{c,k+1})\mathbf{S}_{k}^{med}
-(\delta p_{c,k}+\delta p_{c,k-1})\mathbf{S}_{k-1}^{med}
\right]
\]
を用い、最後に roundoff 分の cell 平均 force を差し引いて
\(\sum_k\delta\mathbf{f}^{sub}_{c,k}=0\) を保証する。
\(\mathbf{S}_{k}^{med}\) は cell center から edge midpoint \(k\to k+1\)
への RZ Pappus median vector である。
For multiblock cells, \(\mathbf{S}_{k}^{med}\) carries
`cell_orientation_sign[c]`, matching the orientation-signed `cell_Svec` used by
the direct cell-centered pressure force.

For cap triangles the same construction is evaluated over the three active
subzones \(k\in A_c=\{0,1,2\}\).  The cell center is the arithmetic mean of the
three active vertices, the median vectors use the cyclic triangle edges
\(k\to k+1\), and the telescoping force is
\[
\delta\mathbf{f}^{sub}_{c,k}=M_f\left[
(\delta p_{c,k}+\delta p_{c,k+1})\mathbf{S}_{k}^{med}
-(\delta p_{c,k}+\delta p_{c,k-1})\mathbf{S}_{k-1}^{med}
\right],
\qquad k\in\{0,1,2\},
\]
with indices modulo 3.  The cell-mean force over the three active corners is
subtracted, and inactive slot 3 remains zero.  This is a three-subzone
compatible pressure path, not a skip; it preserves the force/work identity
because the same active corner set is used for force assembly and for
\(W^{sub}_c=-\sum_{k\in A_c}\mathbf{F}^{sub}_{c,k}\cdot\mathbf{u}_{n(c,k)}\).

merit factor は
\[
x_c={\max_k\rho_{c,k}^{sub}-\rho_c\over \rho_c},\qquad
M_f=
\begin{cases}
\left[{\alpha_1\over2}
\left(1-\cos{\pi x_c\over2\alpha_2}\right)\right]^n,
& x_c\le2\alpha_2,\\
\alpha_1^n,&x_c>2\alpha_2,
\end{cases}
\]
with defaults \(\alpha_1=\sqrt2,\alpha_2=0.1,n=2\).
`subzonal_merit_mode="constant"` uses `subzonal_merit_constant`;
`"off"` sets \(M_f=1\).

Canonical subzonal pressure の explicit stiffness は、同じ median
geometry と fixed corner mass から cell-local spectral upper bound を
構築する。Subzone \(j\) の pressure が nodal force に掛ける geometry
vector を
\[
\mathbf b_{j,j}=\mathbf S_j^{med}-\mathbf S_{j-1}^{med},\qquad
\mathbf b_{j,j-1}=\mathbf S_{j-1}^{med},\qquad
\mathbf b_{j,j+1}=-\mathbf S_j^{med}
\]
（その他はゼロ、indices は active corner set 上で cyclic）とすると、
mass-normalized positive pressure tangent の trace bound は
\[
\omega_c^2 \le
\sum_{j\in A_c} M_f
{\rho_{c,j}^{sub} c_{s,c}^2\over V_{c,j}^{sub}}
\sum_{\ell\in A_c}{\|\mathbf b_{j,\ell}\|^2\over m_{c,\ell}^{sub}} .
\]
ここで \(c_{s,c}^2=(\partial p/\partial\rho)_s\) は current EOS closure
の sound-speed tangent であり、fixed subcell inertia を用いるため、
incident-cell masses を足した実際の nodal inertia に対して保守的な
local upper bound になる。各 active cell は
\[
\Delta t_c^{sub}={C_\mathrm{CFL}\over\sqrt{\omega_c^2}}
\]
を与え、`HydroDtDiagnostics.subzonal_pressure_dt` はその最小値、
global hydro timestep はこれを既存 limiter 群との min に含める。
\(M_f=0\) または zero stiffness の cell は \(+\infty\) を与える。

同じ force バッファから compatible work scratch を計算する：
\[
W^p_c=-\sum_{k\in A_c}\mathbf{F}^{p}_{c,k}\cdot\mathbf{u}_{n(c,k)},\qquad
W^{sub}_c=-\sum_{k\in A_c}\mathbf{F}^{sub}_{c,k}\cdot\mathbf{u}_{n(c,k)},
\]
\[
W^{av}_c=-\sum_{e\in\partial c}
\mathbf{f}^{av}_{c,e}\cdot
(\mathbf{u}_{n_1(e)}-\mathbf{u}_{n_0(e)}).
\]
T5 wires `work_p_per_cell`, `work_sub_per_cell`, and `work_av_per_cell` into
the Lagrangian cell internal-energy update whenever the compatible
force/work path is enabled (`av_model="csw_edge"` with
`subzonal_pressure_enabled=True`, or `av_model="csw_edge_csw98"` regardless
of the subzonal setting — k03 F-02 correction).  The corrector
first recomputes pressure, CSW edge AV, and subzonal pressure forces at the
half-step geometry/state.  After the final nodal velocity update, the work
buffers are refreshed from those stored half-step forces using
\[
\bar{\mathbf{u}}_n={1\over2}(\mathbf{u}_n^n+\mathbf{u}_n^{n+1}),
\]
so the cell work is the negative of the kinetic work done by the same
compatible forces.  In compatible mode the legacy scalar
\(-(P+Q)\Delta V/m\) pressure-work term is bypassed; pressure work is counted
exactly once through \(W^p_c\):
\[
\Delta e_{e,c}^{comp}
={\Delta t\over m_c} f_{e,c}(W^p_c+W^{sub}_c),
\]
\[
\Delta e_{i,c}^{comp}
={\Delta t\over m_c}\left[f_{i,c}(W^p_c+W^{sub}_c)+W^{av}_c\right],
\]
where \(f_{e,c}=P_{e,c}/(P_{e,c}+P_{i,c})\) and
\(f_{i,c}=P_{i,c}/(P_{e,c}+P_{i,c})\).  If
\(P_{e,c}+P_{i,c}\le10^{-20}\,\mathrm{dyn\,cm^{-2}}\), the split falls back
to \(f_e=f_i=1/2\) and emits a debug log.  CSW AV work is shock dissipation and
is deposited entirely in ion energy.  In 1T mode the same compatible work sum
updates `ee` as \(\Delta e_c=\Delta t(W^p_c+W^{sub}_c+W^{av}_c)/m_c\).
The legacy `scalar_vnr_legacy` path and `subzonal_pressure_enabled=False` path
continue to use the pre-existing volume-work update.

Step-0 `[config_fingerprint]` は effective fixed-mass flag、explicit mass
flags、merit mode、\(\alpha_1,\alpha_2,n\)、constant value を記録する。
`[progress]` の既存 cadence（verbose は各 step、normal は 100 step ごと）
では、MPI-global の `subzonal_max_merit`,
`subzonal_nonzero_force_cells`, `subzonal_max_abs_force`,
`subzonal_max_abs_work` を記録する。

Phase 3 MVP の `subzonal_mass_enabled=True` path では corner pressure は
cell pressure を共有し、force は checkerboard 係数へ直接戻す lumped
stiffness として
\[
\mathbf{F}_{c,k}^{hg,raw}
= -\kappa\,m_{c,k}^{sub}\,s_k\,
  \frac{\mathbf{a}_{\xi\eta,c}}{\Delta t^2},\qquad
s_k=\{+1,-1,+1,-1\}
\]
を用いる。cell 平均 force を差し引くため、net artificial translation
はゼロである。

legacy `hourglass.enabled=True` path の linearized subzonal pressure
model は
\[
\rho_{c,k}^{sub}=\frac{m_{c,k}^{sub}}{V_{c,k}^{sub}},\qquad
\delta p_{c,k}
= \left(\frac{\partial p}{\partial \rho}\right)_e
  \left(\rho_{c,k}^{sub}-\rho_c\right).
\]
現実装の `subzonal_pressure_model="linearized"` では ideal-gas
hydro closure と整合する近似として
\[
\left(\frac{\partial p}{\partial \rho}\right)_e \simeq \frac{p_c}{\rho_c}
\]
を用いる。  `eos_lookup` は namelist 予約値であり、v1 実装では
`ConfigError` とする。

legacy linearized path の corner force は subzone corner の Pappus area vector
\(\mathbf{S}_{c,k}^{sub}=\partial V_{c,k}^{sub}/\partial \mathbf{x}_{c,k}\)
から
\[
\mathbf{F}_{c,k}^{hg,raw}
= C_{hg}\,\delta p_{c,k}\,\mathbf{S}_{c,k}^{sub}
\]
とする。  セル正味の人工並進を避けるため
\[
\mathbf{F}_{c,k}^{hg}
= \mathbf{F}_{c,k}^{hg,raw}
-\frac{1}{4}\sum_{\ell=0}^{3}\mathbf{F}_{c,\ell}^{hg,raw}
\]
を scatter-add し、実装上の area-vector 符号規約に対して
\(\sum_k \mathbf{F}_{c,k}^{hg}\cdot
((-1)^k\mathbf{a}_{\xi\eta,c}) \le 0\) となる向きだけを採用する。
これにより純並進、線形伸縮、線形 shear では \(\mathbf{a}_{\xi\eta}=0\)
なので force はゼロであり、checkerboard 成分だけを抑える。

nodal force は pressure/AV 加速度計算後、predictor velocity 更新前と
corrector velocity 更新前に加算する。  safety limiter は nodal mass
\(m_n\) と既存 pressure acceleration \(\mathbf{a}^{P}_n\) から
\[
\|\mathbf{F}^{hg}_n\|
\le f_\mathrm{max}\,m_n\,\|\mathbf{a}^{P}_n\|
\]
を課す（`max_force_per_node_fraction` 既定 0.2）。  axis node では
radial component を常にゼロにし、既存 2D boundary type の fixed/reflect
制約も適用する。

compatible work が有効な場合（既定 `compatible_work_enabled=True`）、
corrector stage の同一 hourglass acceleration と node velocity から cell
internal energy increment を
\[
\Delta E_{c}^{hg}
=-\Delta t\sum_{k=0}^{3}
  m_{c,k}^{sub}\,
  \mathbf{a}_{n(c,k)}^{hg}\cdot\mathbf{v}_{n(c,k)}
\]
として加える。  1T では `ee`、2T では `av_heat_to` と同じ規約
（既定 ion）で `ei` または `ee` に入れる。  これは force work と
internal energy work を同じ離散力から作る compatible accounting であり、
total energy drift を roundoff に抑えるための項である。

Winding orientation (BINDING, 2026-07-26). The corner gradients \(a_k\) are
shoelace-family expressions whose sign flips with the cell winding, so they are
multiplied by `orientation = sign(signed planar area)` -- the same correction
the pressure path applies in `rz_area_weighted.cuh`. Without it the side vectors
of clockwise-wound cells point inward and the `dv.S < 0` compression switch
selects expansion: the viscosity then fires in rarefactions and is silent at
shocks. Every cell of the spherical-polar logical meshes is clockwise-wound
under the kernel node ordering, so those meshes ran with an inverted AV switch
until this correction; meshes with positive winding (the RZ box family, and
every pre-existing certification case) are bit-identical because the factor is
exactly 1.0 there. Regression: tests/hydro/test_csw98_crush_fires.cu, case
"winding-invariant". Orientation evaluation hard-rejects zero/near-zero signed
areas (\(\lvert 2A\rvert \le 64\,\epsilon\sum_k(r_k^2+z_k^2)\)); a degenerate
cell must be rejected by mesh admissibility before reaching the AV path.

#### 3.2.10 エネルギー方程式（保存形）

セル質量 \(\Delta M_c\) に対するエネルギー保存：
\[
\Delta M_c \frac{de_c}{dt} = -(P_c + Q_c)\frac{dV_c}{dt}
\]

**2Tモデルの分離**（§1.1.3参照）：

イオンエネルギー：
\[
\Delta M_c \frac{de_{i,c}}{dt} = -(P_{i,c} + Q_c)\frac{dV_c}{dt} + Q_{ei,c}\,V_c
\]
- PdV仕事（イオン圧力 \(P_{i,c}\) 分）
- Q加熱：全量イオンへ（v1.0既定）
- e-i緩和（\(Q_{ei}>0\)：電子→イオン）

電子エネルギー：
\[
\Delta M_c \frac{de_{e,c}}{dt} = -P_{e,c}\frac{dV_c}{dt} - Q_{ei,c}\,V_c - (\nabla\cdot\mathbf{q}_e)_c\,V_c + (S_L + S_r)_c\,V_c
\]
- PdV仕事（電子圧力 \(P_{e,c}\) のみ）
- 熱伝導（§4参照）
- レーザー吸収（\(S_L\)）
- 輻射との交換（\(S_r\)、IMC/DDMCの沈着として計上）
- e-i緩和（符号反転）

e-i緩和の時間離散化は §3.1.5 の有限\(\Delta t\)解析更新
\(q_{ei}^{step}\) をそのまま用いる（1D/2Dで共通）。

#### 3.2.11 密度更新

Lagrangian形式では質量 \(\Delta M_c\) が一定のため：
\[
\rho_c = \frac{\Delta M_c}{V_c}
\]

#### 3.2.12 時間積分（Predictor-Corrector、2次）

1D（§3.1）と同じスキームを2D RZに拡張する。

> **注**：`T_start_eV > 0` の場合、非活性セル（`hydro_active_c = false`）からのコーナー力寄与はゼロ。
> 非活性セルのみに隣接するノードは移動しない（§2.1.1のOR判定による）。

**Predictor（半ステップ）**：
1. 力の計算：\(\mathbf{F}_{c\to n}^n\) を現時刻の \(P^n, Q^n\) から算出
2. 速度の半ステップ更新：\(\mathbf{v}^{n+1/2} = \mathbf{v}^n + \frac{\Delta t}{2}\frac{\mathbf{F}^n}{m_n}\)
3. 位置の半ステップ更新：\(\mathbf{r}^{n+1/2} = \mathbf{r}^n + \frac{\Delta t}{2}\mathbf{v}^{n+1/2}\)
4. 半ステップでの \(V, \rho, P, Q\) 再計算

**Corrector（全ステップ）**：
1. 力の再計算：\(\mathbf{F}^{n+1/2}\) を半ステップの \(P^{n+1/2}, Q^{n+1/2}\) から算出
2. 速度の全ステップ更新：\(\mathbf{v}^{n+1} = \mathbf{v}^n + \Delta t\,\frac{\mathbf{F}^{n+1/2}}{m_n}\)
3. 位置の全ステップ更新：\(\mathbf{r}^{n+1} = \mathbf{r}^n + \Delta t\,\mathbf{v}^{n+1/2}\)
4. 全量更新：\(V, \rho, e, T, P, Q\)

> **注**：位置更新(3)に \(\mathbf{v}^{n+1/2}\)（半ステップ速度）を使用する。
> これはLeapfrog形式であり、\(\mathbf{v}^{n+1}\) を使用するとエネルギー保存が劣化する（§3.1.10と同じ）。

> **注（Q_{ei}の適用タイミング）**：電子–イオンエネルギー交換 \(Q_{ei}\)（§1.1.3, §3.2.10）は
> **Correctorの全量更新（ステップ4）でのみ適用** される。Predictor半ステップでは
> 力学量（\(V, \rho, P, Q\)）のみを再計算し、エネルギー方程式（\(e_i, e_e\)）は
> 更新しない。したがって \(Q_{ei}\) はタイムステップあたり1回の適用であり、
> Predictor段階での半ステップエネルギー更新は行わない。
> これは1D（§3.1.10）でも同様である。

**Corrector ステップ4 の離散エネルギー更新式**：
\[
e_{k,c}^{n+1} = e_{k,c}^{n} - \left(P_{k,c}^{n+1/2} + \delta_{k,i}\,Q_{visc,c}^{n+1/2}\right)\frac{V_c^{n+1} - V_c^{n}}{\Delta M_c} + s_k\frac{Q_{ei,c}^{n+1/2}\,\Delta t}{\rho_c^{n+1}}
\]
ここで：
- \(P_{k,c}^{n+1/2} = (P_{k,c}^n + P_{k,c}^{pred})/2\)：\(P_{k,c}^{pred} = P_k(\rho^{n+1/2}, T_k^n)\) は Predictor 半ステップの密度と時刻 \(n\) の温度で評価した種別圧力（\(k=i\)：イオン圧力、\(k=e\)：電子圧力）
- \(Q_{visc,c}^{n+1/2}\)：§3.2.9 の人工粘性圧力を Predictor の \(\nabla\cdot\mathbf{u}\)（= \((V^{n+1/2} - V^n)/(\Delta t/2 \cdot V^{n+1/2})\) で近似した体積変化率）で評価した値
- \(\delta_{k,i}\) はKroneckerデルタ：\(\delta_{i,i}=1\)（イオン）、\(\delta_{e,i}=0\)（電子）。§3.2.10 の「Q加熱：全量イオンへ」の規約により、人工粘性仕事は **イオンのみ** に適用される
- イオン（\(k=i\)）の粘性仕事は \(-Q_{visc,c}^{n+1/2} \times (V_c^{n+1} - V_c^n) / \Delta M_c\)（符号注意：圧縮時 \(\Delta V < 0\) で正の加熱）。旧版の式は \(Q_{visc}\) 項を \(+\delta_{k,i}\,Q_{visc}\,\Delta t/\Delta M_c\)（次元不整合）と誤記していた — 実装は当初から \(-Q\,\Delta V/\Delta M\) 形（2026-07-26 スペック監査 k02 F-04/§15-5 で訂正）
- 電子（\(k=e\)）の PdV 仕事は \(-P_{e,c}^{n+1/2} \times (V_c^{n+1} - V_c^n) / \Delta M_c\) のみ（Q項なし）
- \(Q_{ei,c}^{n+1/2}\) は §1.1.3 の式を \(T_e^n, T_i^n\)（時刻 \(n\) の温度）で評価し、有限 \(\Delta t\) 更新の指数に `Numerics.hydro.qei_multiplier` を掛ける
- 実装は Predictor entry の EOS 再クロージャ直後に `Te/Ti` を `T^n` snapshot として保存し、既定 `Numerics.hydro.qei_evaluate_at_t_n=True` ではこの snapshot を Qei 評価に用いる。`False` は legacy compatibility mode で、Predictor 後の再クロージャ済み `Te/Ti` を使う。
- \(s_k\) は種別符号：\(s_i = +1\)（イオン）、\(s_e = -1\)（電子）。§1.1.3 の符号規約と整合
- PdV 仕事は体積差分 \((V^{n+1} - V^n)\) を使用する（速度形式ではない）
- 温度はEOS逆変換（§3.1.5の Newton 法）で更新する

> **\(Q_{ei}\) の評価温度に関する注記**：Predictorは力学量（\(V, \rho, P, Q\)）のみを
> 更新し、エネルギー方程式（\(e_i, e_e\)）は更新しない（上記注参照）。
> したがって Predictor 完了後の温度は \(T_e^{pred} = T_e^n\)、\(T_i^{pred} = T_i^n\)
> であり、\(Q_{ei}^{n+1/2}\) は実質的に時刻 \(n\) の温度で評価される。
> これは Predictor-Corrector 法の標準的な規約であり、温度の暗黙的更新は
> Corrector ステップ4（本式）で初めて行われる。

**Midpoint time integration (F-08, G5 epoch, 2026-07-27).** The production single-block integrator is the fixed-one-corrector midpoint scheme: a predictor with old-time forces builds \(u^h, x^h\) and the compatible half-step internal energy \(e^h_c = e^n_c - \tfrac{\Delta t}{2M_c}\sum_p \mathbf f^n_{c,p}\cdot\bar{\mathbf u}^{n\to h}_p\); the midpoint closure evaluates \(\rho^h = M_c/V(x^h)\) and the EOS at \((\rho^h, e^h)\), producing one midpoint corner-force family shared by momentum, work, and audit; a provisional full update and exactly one deterministic re-evaluation (\(x^{h,\mathrm{corr}} = (x^n + x^{n+1,*})/2\), half energy recomputed from the provisional midpoint forces) precede the final update from \(t^n\) with \(\bar{\mathbf u} = (u^n + u^{n+1})/2\), preserving the discrete kinetic–internal exchange identity. Measured smooth-flow temporal orders: \(p_\rho = 2.0000\), \(p_e = 1.9999\) (V3 harness), versus \(1.0007\) for the retained pc_v0 scheme. Work ownership per force family, \(Q_{ei}\) operator position, floors, and boundary-work ledgering are unchanged from pc_v0; stage-local trial states never write back to tables or persistent ledgers.

#### 3.2.12a Axis-row Lagrangian motion preflight

Axis-row predicates in this section are active only when
`Numerics.has_physical_rz_axis=True`, i.e. 2D_RZ, `Mesh.r_min` is within the
configured zero-axis tolerance, and the inner radial hydro boundary is
axis-like.  For annular 2D_RZ meshes (`Mesh.r_min > 0`), `i=0` is the inner
radial boundary row, not the physical axis; axis-margin, axis-spine, and
`AxisFace`/`AxisBand` special handling are skipped.

2D_RZ の axis-row cell \((i=0,j)\) について、axis nodes
\((0,j),(0,j+1)\) と外側 row nodes \((1,j),(1,j+1)\) から analytic
axis margin を定義する：
\[
s = z_{0,j+1}-z_{0,j},
\qquad
Q = r_{1,j}(z_{1,j+1}-z_{0,j+1}) - r_{1,j+1}(z_{1,j}-z_{0,j}),
\]
\[
M_j = s\min(r_{1,j},r_{1,j+1}) + \min(Q,0).
\]
\(s>0\), \(r_{1,j}>0\), \(r_{1,j+1}>0\), \(M_j>0\) は axis-row cell が
bilinear reference cell 全体で非反転であるための closed-form margin として用いる。
\(M_j\) の単位は cm\(^2\)。

`Numerics.hydro.axis_motion_floor_fraction = f` は既定 \(f=0\) で無効。
\(f>0\) のとき、Predictor と Corrector の位置 commit 前に preflight を実行する。
各 axis-row cell について、trial 位置の margin \(M_j(\lambda)\) を、axis row と
\(i=1\) row の \(z\) motion はそのまま、\(i=1\) row の radial motion だけ
\(\lambda\in[0,1]\) で縮小して評価する。full motion が
\[
M_j(1) < f M_j(0)
\]
を満たす場合、bisection で最大の \(\lambda\) such that
\[
M_j(\lambda) \ge f M_j(0)
\]
を求める。各 \(i=1\) node の scale は隣接 axis-row cells からの最小値を atomic
minimum で集約し、その node の radial position velocity に適用する。Corrector では
同じ scale を最終 radial velocity にも適用する。これにより既定 \(f=0\) では既存
Lagrangian update と bitwise 同一の no-op になり、opt-in 時だけ axis 近傍の過大な
inward radial motion を制限する。

#### 3.2.13 CFL条件（2D）

\[
\Delta t_{hydro} = C_{CFL}\cdot\min_{c \in \mathcal{A}}\left(\frac{\Delta l_c}{|\mathbf{u}_c| + c_{s,c}}\right)
\]
- \(\mathcal{A} = \{c \mid \text{hydro\_active}_c = \text{true}\}\)：活性セル集合（§2.1.1）。\(\mathcal{A} = \emptyset\) のとき \(\Delta t_{hydro} = \infty\)。
- \(\Delta l_c = \sqrt{A_c}\)
- \(c_{s,c}\)：セル中心の音速
- \(|\mathbf{u}_c|\)：セル平均速度の大きさ
- 既定：\(C_{CFL}=0.3\)（SPECIFICATION.md §9.1準拠）

For single-block spherical-polar `polar_center_treatment="button"`, dormant
cells are excluded from the acoustic CFL reduction even if their degenerate
geometry is present in the structured cell array.  The active button cell
`c=0` is treated as the convex seam-node \(N_\theta+1\)-gon rather than as a
logical quadrilateral.  Its acoustic timestep contribution is
\[
\Delta t_{\rm btn}=C_{CFL}{h_{\rm btn}\over c_{s,\rm btn}},
\qquad
h_{\rm btn}=\min\left({2A_{\rm btn}\over P_{\rm btn}},
                      h_{\triangle,\min}\right),
\]
where \(A_{\rm btn}\) and \(P_{\rm btn}\) are the planar R-Z area and perimeter
of the button polygon and \(h_{\triangle,\min}\) is the minimum altitude over
the centroid fan triangles \((\mathbf{x}_c,\mathbf{x}_k,\mathbf{x}_{k+1})\).
This mirrors the multiblock center-CFL robust length convention while avoiding
the historical apex zero-length winner.  Structured active shell cells keep the
legacy \(\sqrt{A_c}/(c_s(1+C_1))\) acoustic+linear-AV estimate.

For multiblock cap triangles in the centroid-r center CFL path, the center-cell
length uses the same active-polygon form:
\[
h_c=\min\left({2A_c\over P_c}, h_{\perp,c}\right),
\]
where \(A_c\) and \(P_c\) are the planar R-Z area and perimeter over the three
active vertices and \(h_{\perp,c}\) is the minimum altitude to an active
triangle edge.  The cell velocity sample is the mean over the three active
nodes.  Quad cells keep the legacy quadrilateral \(h_c\) and four-node mean.

`Numerics.hydro.center_cfl_scope` selects the topology-aware center-CFL
limiter.  The default `"disabled"` returns no center-CFL restriction and
preserves the existing default-off path.  Legacy namelists that set only
`tri_fan_center_cfl_enabled=True` are mapped by the builder to
`center_cfl_scope="tri_fan_radial_index"`.

For `logical_mesh_2d="spherical_polar_halfplane"` with
`polar_center_treatment="tri_fan"`, `"tri_fan_radial_index"` applies the
legacy center-band stiffness limiter over
\[
i\in[0,\texttt{tri\_fan\_center\_cfl\_band\_radial\_index}],\qquad
j\in[0,n_z-1],
\]
including the pole-adjacent \(j=0\) and \(j=n_z-1\) cells.  For each active
band cell,
\[
\Delta t_c =
\texttt{tri\_fan\_center\_cfl\_safety}\,
{h_{ij}\over c_{\mathrm{eff},c}},
\qquad
c_{\mathrm{eff},c}=
\sqrt{{P_{e,c}+P_{i,c}+Q_{visc,c}\over\max(\rho_c,\rho_{floor})}},
\]
\[
h_{ij}=s_{mid,i}\Delta\theta_j,\qquad
s_{mid,i}={1\over2}(s_{node,i}+s_{node,i+1}),\qquad
\Delta\theta_j=\theta_{node,j+1}-\theta_{node,j}.
\]
\(\rho_{floor}=10^{-30}\) g/cm\(^3\), and \(c_{\mathrm{eff}}\) is floored at
\(10^{-30}\) cm/s before division.  \(P_e\), \(P_i\), and \(Q_{visc}\) are in
dyn/cm\(^2\); \(h_{ij}\) is in cm; \(\Delta t_c\) is in s.  For tri_fan
\(i=0\) triangles, \(s_{mid,0}=0.5s_{node,1}>0\); the pinned origin
\(s_{node,0}=0\) is not used as a zero cell length, avoiding an apex
\(\Delta t=0\) trap.  \(\Delta\theta_j\) is computed from the actual node angles
via `atan2`, so non-uniform theta decks use their real angular spacing.

For `topology_scheme="multiblock_cart_core_polar_shell"`,
`center_cfl_scope="centroid_r_le_r_match"` applies the center limiter to
quad cells whose planar centroid radius satisfies
\[
R_c\le r_{\mathrm{match}}
=\texttt{Mesh.multiblock\_cart\_core\_r\_match}.
\]
For each scoped active polygon, including cap triangles with
`cell_nverts=3`,
\[
\Delta t_c =
\texttt{tri\_fan\_center\_cfl\_safety}\,
{h_c\over
\max(c_{s,c}, |\bar u_{r,c}|+|\bar u_{z,c}|, 10^{-30})},
\]
where \(c_{s,c}\) is the same cell sound speed used by the acoustic hydro CFL
when available (falling back to the pressure-based ideal-gas estimate
\(\sqrt{\gamma(P_e+P_i)/\rho}\) for standalone center-CFL evaluation).  The
active-polygon length scale is
\[
h_c=\min\left({2A_c\over P_c}, h_{\perp,c}\right),
\]
with \(A_c={1\over2}\left|\sum_k(r_k z_{k+1}-r_{k+1}z_k)\right|\) over the
active vertices, \(P_c=\sum_k\sqrt{(r_{k+1}-r_k)^2+(z_{k+1}-z_k)^2}\) the
planar perimeter, and
\[
h_{\perp,c}=\min_k
{ |(r_k-r_{k+1})(z_{k+2}-z_{k+1})
   -(z_k-z_{k+1})(r_{k+2}-r_{k+1})|
 \over
 \sqrt{(r_{k+2}-r_{k+1})^2+(z_{k+2}-z_{k+1})^2}}
\]
with indices modulo `cell_nverts[c]`. Thus \(h_{\perp,c}\) is the minimum
altitude to the opposite edge for a triangle and the corresponding local
perpendicular distance for a quad. This uses planar area, not Pappus RZ volume,
so the triangle cap CFL is \(h_c=\min(2A/P,\min h_{\rm altitude})\).

The reduction over the center band is deterministic: candidate cells are swept
in monotonic cell id order \(c=i n_z+j\), and equal-\(\Delta t\) ties choose the
lowest cell id.  The implementation uses a fixed-order host or single-thread
device sweep and no atomics.  If this limiter binds the global step, dt lineage
reports the canonical winner as `tri_fan_center`; the HDF5 history group
`/diagnostics/dt_breakdown_history/tri_fan_center_cfl/` records `dt`,
`binding_cell_id`, `binding_i`, `binding_j`, `h`, `c_eff`, `q_over_p`, and
`dt_global_over_dt_center`.

The limiter reads post-cap `Qvisc`.  With q-cap \(Q\le kP\), it sees bounded
\(c_{\mathrm{eff}}=\sqrt{(1+k)P/\rho}\) instead of runaway
\(\sqrt{(1+Q/P)P/\rho}\).  Under homothetic adiabatic contraction
\(s\propto C^{-1}\), \(\rho\propto C^3\), \(P\propto C^{2\gamma-1}=C^5\) for
\(\gamma=5/3\), and \(c_s\propto C\), so the pressure-only stable timestep
scales as
\[
\Delta t_{stable}\propto {s\Delta\theta\over c_s}\propto C^{-2}.
\]
This is a finite penalty (about \(64\times\) at \(C=8\)).  The I1-B baseline
catastrophic factor came from \(Q/P\approx2500\), which further reduced dt by
\(\sqrt{P/(P+Q)}\approx1/50\); q-cap changes that factor to
\(\sqrt{1/(1+k)}\), e.g. \(1/2.24\) for \(k=4\).

For multiblock acoustic CFL aggregation, the reduction range is the minimum
over all blocks and all hydro-active cells in the topology total
(`state.rho.size()`), not `nr*nz`.  The winner diagnostic reports the stable
cell id as primary; structured `(i,j)` is reported as `(-1,-1)` for multiblock
because no single global structured indexing exists.  The default I1-B gamma
MVP core has a smaller length scale than the polar shell, so core cells can
dominate the global hydro CFL; this is by design rather than a topology
aggregation error.

When `center_cfl_scope="disabled"` (default), the existing tri_fan early-return
in `compute_axis_margin_cfl_dt` is preserved exactly, so the default-off path
remains byte-identical.

`Numerics.hydro.axis_margin_dt_floor_fraction = f_{axis,dt}` は既定
\(f_{axis,dt}=0\) で無効な 2D_RZ 専用の追加 dt limiter である。
\(f_{axis,dt}>0\) のとき、上式と post-shock 制約から得た候補
\(\Delta t_*\) に対して、axis-row cell \((i=0,j)\) の trial 位置
\[
(r,z)^{trial}(\tau) = (r,z)^n + \tau (v_r,v_z)^n,\qquad
0 \le \tau \le \Delta t_*
\]
で §3.2.12a と同じ analytic axis margin \(M_j(\tau)\) を評価する。
現在ステップ開始時の margin を \(M_j^n=M_j(0)\) とし、floor を
\[
M_{floor,j}=f_{axis,dt}M_j^n
\]
と定義する。axis-margin limiter は
\[
\Delta t_{axis} =
\max\{\tau \in [0,\Delta t_*] \mid
M_j(\tau) \ge M_{floor,j}\ \forall j\}
\]
を二分探索で求め、最終的に
\[
\Delta t_{hydro} = \min(\Delta t_*,\Delta t_{axis})
\]
とする。この limiter は hydro 方程式・離散化・単位系を変更せず、
axis-row cell が opt-in floor 未満になる timestep だけを縮小する。
\(M_j\) と \(M_{floor,j}\) の単位はいずれも cm\(^2\) である。

`Numerics.hydro.corner_jacobian_ale_trigger_enabled` is a default-off 2D_RZ
signed-corner-J pre-hydro ALE trigger.  For cell \(c\) with ordered quadrilateral nodes
\((\mathbf{x}_0,\mathbf{x}_1,\mathbf{x}_2,\mathbf{x}_3)\), the corner
Jacobian at corner \(k\) is
\[
J_{c,k} =
(\mathbf{x}_{k+1}-\mathbf{x}_k)\times
(\mathbf{x}_{k-1}-\mathbf{x}_k),
\]
with indices modulo 4 and the 2-D cross product interpreted in the
\((r,z)\) plane.  The same definition is used by the mesh diagnostic dump.
Given the current candidate \(\Delta t_*\), the trigger evaluates
\[
\mathbf{x}_n(\tau)=\mathbf{x}_n^0+\tau\mathbf{v}_n^0,\qquad
0\le\tau\le\Delta t_* .
\]
For every active cell corner with \(J^0_{c,k}>0\), the floor is
\[
J_{floor,c,k} = \epsilon_J J^0_{c,k},
\qquad
\epsilon_J=\texttt{Numerics.hydro.corner\_jacobian\_floor\_eps}.
\]
If \(J_{c,k}(\Delta t_*) < J_{floor,c,k}\), a bisection search on
\([0,\Delta t_*]\) computes the largest admissible \(\tau\) for that corner.
The global admissible scale is the minimum over all cell corners:
\[
\alpha_{cornerJ} = \frac{1}{\Delta t_*}
\min_{c,k}\max\{\tau\in[0,\Delta t_*]\mid
J_{c,k}(\tau)\ge J_{floor,c,k}\}.
\]
If \(\alpha_{cornerJ}\) is below
`Numerics.hydro.corner_jacobian_ale_trigger_scale`, the driver invokes ALE
with `force_rezone=true` before laser / hydro / conduction / radiation split
operators, then recomputes the global timestep from the repaired mesh.  The
current implementation also evaluates the same predicate at Hydro2D corrector
pre-commit and logs an inadmissible trial, but full split-operator step retry
is intentionally not part of this trigger.

`Numerics.hydro.volume_rate_cfl_enabled` is a default-off 2D_RZ post-hoc
hydro dt limiter.  Let \(V_c^{old}\) be the cell volume at the beginning of the
previous completed hydro step, \(V_c^{new}\) the current volume, and
\(\Delta t_{used}\) the timestep used by that previous hydro step.  The stored
history is skipped when \(\Delta t_{used}\le0\), which covers the first step.
For each cell,
\[
f_{V,c}=\frac{|V_c^{new}-V_c^{old}|}{\max(V_c^{old},10^{-30})},\qquad
r_{V,c}=\frac{f_{V,c}}{\Delta t_{used}}.
\]
With threshold \(f_{V,max}\), the limiter computes
\[
\Delta t_{vol}=\min_c \frac{f_{V,max}}{r_{V,c}}
=\min_c\frac{f_{V,max}\Delta t_{used}}{f_{V,c}},
\qquad
\Delta t_{hydro}=\min(\Delta t_{hydro},\Delta t_{vol}).
\]
This limiter does not modify the Lagrangian hydro update or the axis-margin
limiter.  It only reduces the proposed next timestep after a measured large
volume change, so the response is intentionally late by one hydro step.
Accepted ALE rezone/remap resets \(V_c^{old}\) to the post-ALE current volume
before the next hydro CFL evaluation, because ALE mesh motion is not a
Lagrangian hydro volume-rate signal.

`Numerics.hydro.rz_geometric_cfl_enabled` is a default-off 2D_RZ predictive
hydro dt limiter for the Lagrangian half-step.  For each structured
quadrilateral cell with vertices ordered \((i,j),(i+1,j),(i+1,j+1),(i,j+1)\),
TENRYU evaluates the axisymmetric polygon volume
\[
V_{RZ}(\mathbf{x})={\pi\over3}\sum_{k=0}^{3}
(r_k+r_{k+1})(r_k z_{k+1}-r_{k+1}z_k),
\qquad k+1\equiv0\pmod4,
\]
using the same full cgs volume convention as `Mesh::recompute_geometry`.
Given the current node coordinates \(\mathbf{x}^n\), half-step node velocity
\(\mathbf{u}^{1/2}\), and candidate hydro timestep \(\Delta t\), the trial path
is
\[
\mathbf{x}(\tau)=\mathbf{x}^n+\tau\mathbf{u}^{1/2},\qquad
0\le\tau\le\Delta t.
\]
The default displacement velocity is the current `state.v_r/v_z` field.  When
`Numerics.hydro.rz_geometric_cfl_precise_u_half_enabled=True`, the limiter first
recomputes the current pressure+artificial-viscosity nodal force, forms
\[
\mathbf{u}^{1/2}_{CFL}=\mathbf{u}^n + {1\over2}\Delta t\,\mathbf{a}_n,
\]
and passes that velocity to the geometric predicate.  This path is opt-in and
does not alter the acoustic or artificial-viscosity CFL terms.
A cell is admissible at \(\tau\) when every projected radial coordinate is at
least `rz_geometric_cfl_r_floor` and
\[
V_{RZ}(\mathbf{x}(\tau))\ge
\max\left(\eta_V V_{RZ}(\mathbf{x}^n),\eta_{V0}V_c^{initial}\right),\qquad
\eta_V=\texttt{rz\_geometric\_cfl\_etaV}.
\]
The cumulative term is active when
`Numerics.hydro.rz_geometric_cfl_cumulative_protection_enabled=True`, with
\(\eta_{V0}=\texttt{rz\_geometric\_cfl\_v\_initial\_floor}\).  When disabled,
\(\eta_{V0}=0\) and the criterion reduces to the original per-step floor.
`State.cell_vol_initial` is populated from the IC mesh volume and remains fixed
through Lagrangian motion and ALE remap.
If the full candidate \(\Delta t\) is admissible for all cells, the limiter
returns \(+\infty\) to leave the existing hydro dt bitwise unchanged.  If any
cell would violate the floor, a device bisection finds the largest admissible
\(\tau\in[0,\Delta t]\) for that cell and the hydro dt is clamped to the global
minimum.  A non-positive or non-finite current cell volume returns a zero
limit, reporting an already invalid mesh.

In button-center topology, dormant cells are inert for this limiter and do not
enter the min reduction.  The button cell `c=0` uses the seam-node polygon RZ
volume
\[
V_{\rm btn}(\tau) =
s_{\rm btn}{\pi\over3}\sum_{k=0}^{N_\theta}
(r_k(\tau)+r_{k+1}(\tau))
(r_k(\tau)z_{k+1}(\tau)-r_{k+1}(\tau)z_k(\tau)),
\quad s_{\rm btn}=-1,
\]
with \(k+1\) wrapping around the closed seam polygon and
\(\mathbf{x}_k(\tau)=\mathbf{x}_k^n+\tau\mathbf{u}_k^{1/2}\).  It applies the
same per-step and cumulative volume floors as structured cells.  The radial
floor predicate remains the structured-cell predicate; the button polygon
branch is a volume-positivity check because its seam includes physical polar
axis endpoints.

`Numerics.hydro.trial_volume_cfl_enabled` is a default-off 2D_RZ pre-corrector
trial-volume diagnostic.  Let \(\mathbf{x}_n^0\) be the node coordinates at the
beginning of the hydro step and \(\mathbf{u}_n^{corr}\) the corrector
displacement velocity after axis-motion preflight.  The trial position used by
the diagnostic is
\[
\mathbf{x}_n^{trial} = \mathbf{x}_n^0 + \Delta t\,\mathbf{u}_n^{corr},
\]
matching the final corrector position commit.  Cell volumes are recomputed from
the trial quadrilateral nodes with the same 2D_RZ volume formula used by mesh
geometry refresh.  A cell is admissible when
\[
\frac{V_c^{trial}}{\max(V_c^n,10^{-300})} \ge f_{trial}.
\]
On failure the diagnostic returns `admissible=False`,
`min_trial_vol_ratio`, `first_failing_cell`, and
\(\Delta t_{suggested}=f_{shrink}\Delta t\).  With driver retry disabled this
result is diagnostic-only and the existing non-positive-volume guard remains
authoritative.  With driver retry enabled, the same predicate can reject the
current split-operator attempt before the final position commit.

#### 3.2.13a Driver-level full-step retry on inadmissible corrector (Phase 2d-extension v4 Wave 1)

`Numerics.hydro.driver_full_step_retry_enabled` is a default-off driver policy
for recovering from an inadmissible 2D_RZ Hydro corrector trial.  At the top of
each outer driver step, before any split operator runs, the driver captures a
rollback snapshot of `tenryu::core::State`.  Hydro2D then evaluates the
post-corrector pre-commit admissibility predicates for trial cell volume and
signed corner-J.  If either predicate fails and retry is enabled, Hydro returns
a soft failure result instead of committing the final node positions.

On a soft Hydro failure, the driver restores the pre-step snapshot, halves the
attempted outer-step timestep,
\[
\Delta t_{retry} = \frac{1}{2}\Delta t_{failed},
\]
and reruns the full split-operator step.  Geometric failures from the
mesh-quality dt CFL or in-hydro candidate guards instead honor finite positive
`suggested_dt < dt_failed` strictly:
\[
\Delta t_{retry}=\max(\Delta t_{min},\Delta t_{suggested}).
\]
For non-geometric failures, `driver_retry_use_suggested_dt_enabled=True` keeps
the legacy opt-in behavior \(\min(0.5\Delta t_{failed},\Delta t_{suggested})\);
otherwise the driver keeps strict halving.  The retry loop is bounded by
`Numerics.hydro.driver_full_step_retry_max_attempts` (default 3).  Exhaustion is
a fatal error that reports the step, time, final attempted timestep, first
failing cell/corner, minimum metric, and failure reason.

The snapshot covers all deterministic `State` fields needed by FLD, SN, HOLO,
hydro, conduction, laser deposition, cumulative energy scalars, ALE/adaptive-AV
state, and per-step radiation diagnostics.  It intentionally excludes the IMC
particle pool and IMC class-owned mutable counters.  Therefore driver retry is
fatal-disabled at driver entry when `radiation.mode == ImcDdmc`; Wave 1 scope is
deterministic FLD/SN modes.

When `driver_full_step_retry_enabled=False`, which is the default, Hydro2D keeps
the existing diagnostic-only behavior and does not return early.  This preserves
the baseline update path bit-for-bit.  Accepted retry-enabled steps reach
history, output, checkpoint, and cumulative-energy aggregation only after a
successful attempt.  Rejected attempts are restored before those post-step
paths run.

**1D_SPH soft failure (W-B, 2026-07-03)** — `Hydro1D::lagrangian_step` は
Lagrangian ノード移動後に非正セル体積（圧縮駆動のノード交差）を検出すると、
`driver_full_step_retry_enabled=True` の場合に soft failure
（`reason="non_positive_volume_1d"`、first_failing_cell/failing_value 付き）を
返す。driver は 2D の repair-plan 機構（ALE rung / axis regime）を経由せず、
snapshot 復元 + 厳密 dt/2 の単純経路で全 split step を再試行する（予算・
枯渇時 fatal は 2D と同一）。retry 無効時は従来どおり hard assert だが、
検出点が hydro 直後になったため下流の laser/診断の geometry recompute で
遅れて abort する旧挙動より文脈情報が正確になった。SN material Newton の
timestep 拒否（§6.8）と同じ driver 予算を共有する。

`Numerics.hydro.mesh_geometry_soft_fail_enabled` is a default-off Wave 1 mesh
geometry control-flow option.  When it is false, Hydro2D geometry refresh uses
the legacy `Mesh::recompute_geometry()` hard-assert path and the existing
post-refresh host cell-volume guard unchanged.  When it is true, the StepStart,
PostPredictor, and PostCorrector geometry refreshes call the typed mesh geometry
check and return a `HydroStepResult` before density refresh if a non-finite node
coordinate, non-finite cell geometry, or non-positive cell geometry is detected.
Retry snapshot restore validation recomputes only the mesh geometry caches and
uses a controlled fatal path if the restored mesh is invalid.  This is control
flow only -- no physics/equation/discretization/RNG/unit change.

`Numerics.hydro.in_hydro_corner_j_guard_enabled` is a default-off Wave 2
candidate-mesh control-flow option.  When it is false, Hydro2D does not allocate
new buffers, launch the candidate corner-J kernel, or change the predictor /
corrector refresh path.  When it is true, Hydro2D checks the predictor and
corrector candidate node positions before the corresponding geometry refresh.
For each active 2D_RZ cell corner, the guard compares the hydro-stage-start edge
vectors \(a_0,b_0\) with the candidate edge-vector changes \(\Delta a,\Delta b\)
along
\[
J(\sigma)=J_0+\sigma J_1+\sigma^2J_2,\qquad 0\le\sigma\le1,
\]
where
\[
J_0=\det(a_0,b_0),\quad
J_1=\det(\Delta a,b_0)+\det(a_0,\Delta b),\quad
J_2=\det(\Delta a,\Delta b).
\]
The floor is the existing dimensionless
\[
J_{floor,c,k}=\texttt{Numerics.hydro.corner\_jacobian\_floor\_eps}\,J_{0,c,k}.
\]
If the minimum of \(J(\sigma)\) on \([0,1]\) is below this floor, the guard solves
for the smallest root \(\sigma_{fail}\in(0,1]\), reports
\[
\Delta t_{suggested}=0.7\,\sigma_{fail}\,\Delta t,
\]
and returns a typed `HydroStepResult` with `retry_action=ReduceDtOnly` before
`refresh_geometry_and_density` is called.  If the start corner is already
non-positive or non-finite, the reported scale is zero.  This is an admissibility
predicate on the candidate mesh only; it does not change units, equations,
discretization, or accepted-step physics.

Q2.5 adds two default-off predicates to the same in-hydro candidate trajectory.
When `Numerics.hydro.in_hydro_gauss_j_guard_enabled=True`, each active cell also
evaluates \(J_g(\sigma)\) at the four \(2\times2\) Gauss-Legendre points
\((\xi,\eta)=(\pm1/\sqrt3,\pm1/\sqrt3)\).  The derivative vectors at each Gauss
point are linear in \(\sigma\), so
\[
J_g(\sigma)=J_{g,0}+\sigma J_{g,1}+\sigma^2J_{g,2},
\]
and the same endpoint-plus-vertex minimum and first floor-root solve are used.
The floor is
\[
J_{g,floor}=
\texttt{Numerics.hydro.in\_hydro\_gauss\_j\_floor\_rel}\,J_g(0),
\]
with default relative floor \(10^{-8}\).  For a bilinear quadrilateral,
\(J(\xi,\eta)\) is affine over the reference square at fixed \(\sigma\), so
positive corner-J at all four reference corners implies positive Gauss-J at that
same \(\sigma\); this guard is therefore a direct quadrature-point predicate and
a stricter/non-finite diagnostic rather than a replacement for the corner-J
trajectory.

When `Numerics.hydro.in_hydro_rz_volume_guard_enabled=True`, each active cell
also evaluates the existing signed axisymmetric quadrilateral volume
\[
V_{RZ}(\sigma)=\texttt{rz\_signed\_quad\_volume}(r_k(\sigma),z_k(\sigma)),
\qquad
(r_k,z_k)(\sigma)=(r_k,z_k)_0+\sigma[(r_k,z_k)_1-(r_k,z_k)_0].
\]
Because the Pappus-weighted polygon formula is cubic in the nodal coordinates,
\(V_{RZ}(\sigma)\) is treated as a cubic polynomial along the linear candidate
trajectory.  Its minimum on \([0,1]\) is evaluated from the endpoints and the
real stationary points of \(dV_{RZ}/d\sigma\); the first floor crossing is then
refined by bisection on the bracket ending at the first stationary point or
endpoint below floor.  The floor is
\[
V_{floor}=
\texttt{Numerics.hydro.in\_hydro\_rz\_volume\_floor\_rel}\,V_{RZ}(0),
\]
with default relative floor \(10^{-8}\).  This catches candidate meshes whose
Cartesian corner-J remains positive while the axisymmetric signed volume flips
because the radial coordinate trajectory crosses through the symmetry axis.

The accepted-step invariant is: if the trial-volume, signed corner-J, in-hydro
Gauss-J, or in-hydro RZ-volume pre-commit predicates are enabled, the accepted
Hydro corrector has passed the corresponding admissibility check before final
position commit.  This addresses
the Phase 2cd-coupled-ALE Wave 1 evidence item
`I1_mid_step_corner_J_corrector_inversion_bulk_cell`, where the pre-hydro ALE
trigger alone could miss an inversion that appears only after the corrector
velocity update.

`Numerics.hydro.mesh_quality_dt_cfl_enabled` is a default-off 2D_RZ
mesh-quality dt CFL for the Lagrangian predictor and corrector position
commits.  The predictor check is evaluated after the half-step velocity update
and axis-motion preflight, but before the half-step position commit
\[
\mathbf{x}^{n+1/2}=\mathbf{x}^{n}+{1\over2}\Delta t\,\mathbf{u}^{1/2}.
\]
The corrector check is evaluated after the corrector velocity update and
axis-motion preflight, but before the final position commit
\[
\mathbf{x}^{n+1}=\mathbf{x}^{n}+\Delta t\,\mathbf{u}^{1/2}.
\]
For each active cell, TENRYU evaluates the linear candidate trajectory
\[
\mathbf{x}(\sigma)=\mathbf{x}^{n}+\sigma\Delta t_s\,\mathbf{u}^{1/2},
\qquad 0\le\sigma\le1,
\]
where \(\Delta t_s={1\over2}\Delta t\) for the predictor and
\(\Delta t_s=\Delta t\) for the corrector,
and computes the first floor crossing over the enabled predicates:
corner-J quadratic roots, Gauss-point \(J_g(\sigma)\) quadratic roots, and
signed RZ-volume cubic roots.  The floors are relative to the \(\sigma=0\)
metric values:
\[
J_{floor}=\epsilon_{cJ}J(0),\quad
J_{g,floor}=\epsilon_{gJ}J_g(0),\quad
V_{floor}=\epsilon_{RZ}V_{RZ}(0),
\]
where the \(\epsilon\) values are
`mesh_quality_dt_corner_j_floor_rel`,
`mesh_quality_dt_gauss_j_floor_rel`, and
`mesh_quality_dt_rz_volume_floor_rel` respectively.  The root finders are the
same analytic quadratic / cubic helpers used by the in-hydro candidate guard;
endpoint-only checks are insufficient because both \(J(\sigma)\) and
\(V_{RZ}(\sigma)\) may have interior minima.

For multiblock 2D_RZ meshes, the RZ-volume predicate evaluates the same
oriented polygon volume used by the committed geometry refresh.  For cell \(c\),
the active vertices are gathered from `cell_node_csr_offsets/indices`, with
`cell_nverts[c]` selecting triangle versus quadrilateral storage, and
\[
V_{MB,c}(\sigma)=s_c\,
\texttt{rz\_polygon\_volume\_exact}(\mathbf r_c(\sigma),
                                    \mathbf z_c(\sigma), n_c),
\qquad s_c=\texttt{cell\_orientation\_sign}[c]\in\{-1,+1\}.
\]
The multiblock volume floor is
`mesh_quality_dt_rz_volume_floor_rel * state.vol[c]`.  After a geometry refresh,
`state.vol[c]` is the positive \(V_{MB,c}(0)\) for an admissible active cell, so
the \(\sigma=0\) path point uses the same physical positive-volume convention as
the committed mesh.  Active cells with non-positive current `state.vol[c]` are
not used by this limiter; that condition is treated as a pre-existing geometry
issue rather than a dt-CFL root.

For axis-face cells, `mesh_quality_dt_axis_margin_additive=True` adds the
analytic axis-margin predicate as an additional AND condition on the same
trajectory.  It does not replace corner-J, Gauss-J, or RZ-volume checks; all
enabled predicates contribute to
\[
\sigma_{safe} = \min_{c,m}\sigma_{first}(c,m).
\]
The reduction key is composite: the sortable 32-bit representation of
\(\sigma_{first}\) is reduced together with the failing cell/slot identity, so
the reported cell, corner/Gauss slot, and metric are the same predicate that set
\(\sigma_{safe}\).  If any predicate crosses its floor on the candidate path,
Hydro2D returns a typed soft failure before committing node positions, with
\[
\Delta t_{suggested}
= \alpha\,\sigma_{safe}\,\Delta t_s,\qquad
\alpha=\texttt{mesh\_quality\_dt\_safety\_alpha}.
\]
The `HydroStepResult` reason is one of `mesh_quality_corner_j`,
`mesh_quality_gauss_j`, `mesh_quality_rz_volume`, or
`mesh_quality_axis_margin`; `trial_scale` and `min_metric` carry
\(\sigma_{safe}\).  With the master switch disabled, no kernel is launched and
the accepted hydro path remains bitwise identical to the legacy path.

When `Numerics.debug.trace_mesh_motion=True`, the stderr-only mesh trace also
reports a diagnostic keystone metric for the current origin-adjacent
Cartesian-core cell 0.  Let the traced quad corners be
\(\mathbf{x}_k=(r_k,z_k)\), choose \(o=\arg\min_k\|\mathbf{x}_k\|_2\) as the
origin-adjacent corner, \(a=o+1\), \(d=o+2\), and \(b=o+3\) modulo four.  The
coordinate-free signed outboard-corner altitude is
\[
h_K =
-\operatorname{sign}\left(
{(\mathbf{x}_b-\mathbf{x}_a)\times(\mathbf{x}_o-\mathbf{x}_a)
\over \|\mathbf{x}_b-\mathbf{x}_a\|_2}\right)
{(\mathbf{x}_b-\mathbf{x}_a)\times(\mathbf{x}_d-\mathbf{x}_a)
\over \|\mathbf{x}_b-\mathbf{x}_a\|_2}.
\]
The trace field
\[
\texttt{keystone\_psi\_over\_b} =
{h_K\over {1\over2}\|\mathbf{x}_b-\mathbf{x}_a\|_2}
\]
reduces to \((2c-b)/b\) for the idealized cell
\((0,0),(b,0),(c,c),(0,b)\).  The same line reports per-corner signed
corner-J, signed RZ-plane area, minimum diagonal altitude, block/stable cell id,
and the participating corner/node ids.  The diagnostic is trace-gated only and
does not alter hydro, ALE, remap, mesh-quality predicates, or output schema.

`Numerics.diagnostics.mesh_quality_min.enabled` is a default-off, diagnostic-only
all-run observer evaluated at the committed post-ALE accepted-step site.  It does
not gate a step, change a floor, or modify solver state.  For each accepted 2D_RZ
cell \(c\), let the ordered corner coordinates be
\(\mathbf{x}_{c,k}=(r_{c,k},z_{c,k})\), \(k=0,\ldots,3\), with superscripts \(0\)
and \(n\) denoting the initial mesh and the current committed mesh.  The
edge-length diagnostic uses
\[
e_c^s = \min_k \left\|\mathbf{x}_{c,k+1}^s-\mathbf{x}_{c,k}^s\right\|_2,
\qquad s\in\{0,n\},
\]
and records the running minimum
\[
\texttt{achieved\_min\_edge\_length\_rel}
= \min_{accepted\ steps}\min_c {e_c^n\over e_c^0}.
\]
The altitude diagnostic uses the minimum point-to-opposite-edge-line distance
\[
h_c^s =
\min_k
{\left|(\mathbf{x}_{c,k+3}^s-\mathbf{x}_{c,k+2}^s)\times
(\mathbf{x}_{c,k}^s-\mathbf{x}_{c,k+2}^s)\right|
\over
\left\|\mathbf{x}_{c,k+3}^s-\mathbf{x}_{c,k+2}^s\right\|_2},
\]
with zero opposite-edge length treated as zero altitude, and records
\[
\texttt{achieved\_min\_altitude\_rel}
= \min_{accepted\ steps}\min_c {h_c^n\over h_c^0}.
\]
The conditioning diagnostic forms the bilinear mapping Jacobian matrix at the
four \(2\times2\) Gauss points,
\[
A_{c,q} =
\begin{bmatrix}
\partial r/\partial \xi & \partial r/\partial \eta\\
\partial z/\partial \xi & \partial z/\partial \eta
\end{bmatrix}_{c,q},
\]
computes
\[
\kappa(A_{c,q})={\sigma_{\max}(A_{c,q})\over\sigma_{\min}(A_{c,q})},
\]
and records
\[
\texttt{achieved\_max\_condition\_number}
= \max_{accepted\ steps}\max_{c,q}\kappa(A_{c,q}).
\]
The singular values are evaluated from the eigenvalues of \(A^TA\); if
\(\sigma_{\min}=0\), the diagnostic value is \(+\infty\).  The pre-existing
corner-J, Gauss-J, signed RZ-volume, and negative RZ-volume-count fields keep
their definitions and pass/fail semantics unchanged.  For multiblock meshes
the observer enumerates cells through CSR cell-node connectivity and applies
`cell_orientation_sign` to the raw RZ volume before classifying negative
committed volume, so central and fan/shell windings use the same physical
positive-volume convention.

When `mesh_quality_dt_cfl_enabled=True`, the older endpoint-only
`trial_volume_cfl_enabled` corrector diagnostic is bypassed.  The mesh-quality
envelope supersedes it because the envelope checks the full candidate path for
corner-J, Gauss-J, and signed RZ-volume floor crossings rather than shrinking
from a single endpoint volume ratio.

`Numerics.hydro.regime_aware_corner_j_guard_enabled` is a default-off Wave 3
extension of the signed corner-J guard.  When false, no regime buffer is
allocated and the pre-hydro comparison remains the legacy
`trial_scale < Numerics.hydro.corner_jacobian_ale_trigger_scale` path.  When
true, Hydro classifies active 2D_RZ cells into topology/regime metadata after
the StepStart geometry refresh, with a driver-owned `CellRegime` device cache
reused across the hydro step.  Inactive or void cells are tagged
`VoidOrInactive`.  Active-cell priority is
`AxisFace > AxisBand > DomainBoundary > InteriorCD > InteriorSmooth`; with
`axis_guard_band_cells=N`, `AxisBand` covers cells with
\(i \in [1,N]\), excluding the axis-face row \(i=0\).  If
`Numerics.has_physical_rz_axis=False`, `AxisFace` and `AxisBand` are not
assigned by the topology classifier.

The contact-discontinuity score is
\[
S_{CD,c}=S_{\nabla\rho,c}S_{\nabla\cdot u,c}.
\]
Cell-center coordinates are the arithmetic average of the four RZ nodes.  For
interior cells,
\[
D_r\log\rho_{i,j}=
\frac{\log\rho_{i+1,j}-\log\rho_{i-1,j}}
     {R_{i+1,j}-R_{i-1,j}},\qquad
D_z\log\rho_{i,j}=
\frac{\log\rho_{i,j+1}-\log\rho_{i,j-1}}
     {Z_{i,j+1}-Z_{i,j-1}},
\]
with the same formula using the adjacent cell-center pair as a one-sided
difference on domain boundaries.  The dimensionless gradient sensor is
\[
G_c=\sqrt{A_c}\sqrt{(D_r\log\rho)^2+(D_z\log\rho)^2}.
\]
The compression sensor uses the existing RZ volume-rate operator,
\[
\nabla\cdot u_c = \frac{1}{V_c}\sum_{k=0}^3
  \left(u_{r,k}S_{r,c,k}+u_{z,k}S_{z,c,k}\right),\qquad
C_c=\max(0,-\Delta t\nabla\cdot u_c).
\]
With
\[
\operatorname{sat}(x)=\min(1,\max(0,x)),
\]
the score terms are
\[
S_{\nabla\rho,c}=\operatorname{sat}\frac{G_c-0.1}{0.9-0.1},\qquad
S_{\nabla\cdot u,c}=\operatorname{sat}\frac{C_c-0.1}{0.9-0.1}.
\]
Interior cells enter `InteriorCD` only when \(S_{CD}>0.3\).  Equality is treated
as non-CD on entry.  A cell that was `InteriorCD` in the previous cached
classification remains CD while \(S_{CD}\ge0.2\) and exits below 0.2.  This
hysteresis state is the previous `CellRegime.primary_regime`; first allocation
initializes it to `Unknown`.

The per-cell trigger threshold is 0.5 for smooth/default regimes and 0.85 for
`InteriorCD`.  Because the trigger condition is `trial_scale < threshold`, the
0.85 CD threshold is a more conservative earlier trigger under the existing
comparison direction.  No additional namelist knobs expose the constants in
Wave 3.

`Numerics.hydro.axis_margin_guard_enabled` is a separate default-off axis
predicate.  When true, axis-face cells (\(i=0\)) skip the generic corner-J root
test and use a five-condition axis-margin predicate on the trial trajectory:
axis nodes remain on the axis, off-axis radial nodes stay positive and
monotone away from the axis, the axis z-edge preserves its ordering, signed RZ
area is positive above the corner-J floor scaling, and signed RZ volume is
positive above the same floor scaling.  Non-axis cells keep the generic
corner-J predicate.  This protects benign D4-style axis shears without
weakening non-axis L2 post-shock contact-discontinuity safety.
`Numerics.hydro.axis_margin_additive_in_action8_enabled` is a default-off PR4
completion gate for the in-hydro candidate guard.  When true, the axis-margin
predicate is recorded as an additional axis-face failure slot and the generic
corner-J, Gauss-J, and RZ-volume checks still run; the composite
\(\sigma\)/location reduction chooses the first predicate to fail.  When false,
the legacy replacement behavior above is preserved.

#### 3.2.13b Retry active mesh repair with corner-J balance (Phase 2d-extension v5)

`Numerics.hydro.driver_retry_active_mesh_repair_enabled` is a default-off
extension of the retry epoch in §3.2.13a, active only when driver-level retry
and 2D ALE are also enabled.  For every active 2D_RZ cell, the driver evaluates
current signed corner Jacobians at \(\tau=0\).  If any current corner-J is
non-finite or non-positive, the static predicate fails.  Otherwise the cell
corner balance is
\[
q_{bal,c} = \frac{\min_k J_{c,k}}{\max_k J_{c,k}},
\]
and the predicate fails when
\[
q_{bal,c} <
\texttt{Numerics.hydro.driver\_retry\_corner\_balance\_threshold}.
\]
The default threshold is \(0.01\), i.e. 100:1 same-cell corner-J imbalance.

Attempt 0 evaluates the predicate for diagnostics only.  On retry attempts
after snapshot restore, a failed static predicate forces a retry-only 2D ALE
repair, records post-ALE corner balance, recomputes the global timestep, and
then proceeds through the normal split operators.  A post-ALE failed static
predicate is diagnostic; the retry loop's existing max-attempt exhaustion
remains the only fatal outcome.  The predicate is independent of \(\Delta t\)
and deliberately avoids initial-volume, last-rezone-volume, neighbor-ratio, and
edge-length aspect-ratio normalizers.

`Numerics.hydro.cascade_on_hydro_retry_enabled` is a default-off diagnostic
extension to the retry active mesh repair path.  When it is `false`, retry ALE
calls do not pass hydro retry context into the ALE cascade gate.  When it is
`true`, a retry attempt whose Hydro2D soft-failure reason is `corner_j` carries
the original failing cell/corner and the current corner-balance failure into
`apply_ale(..., force_rezone=true)`.  If the ALE backtrack accepts a candidate
but this retry context reports bad corner balance, the Phase 9c--9e cascade may
enter through `gate_reason=hydro_retry_bad_corner_balance`.  The original
`gate_reason=ale_backtrack_exhausted` branch remains unchanged.

Wave 4 adds a host-side retry repair selector between typed hydro failure
metadata (§3.2.13a) and the retry-only ALE invocation.  The in-hydro guard first
records `RetryActionHint::ReduceDtOnly`, preserving the Wave 2 default.  Only
when the relevant default-off regime or axis guard is enabled may the driver
override the hint to a forced repair:

| Failure regime | Forced repair hint | ALE request |
|---|---|---|
| `AxisFace` / `AxisBand`, or real axis-margin failure | `ForceAxisSpinePlusLocalAle` | `AxisSpinePlusLocal` |
| `DomainBoundary` | `ForceBoundaryPatchRepair` | `BoundaryPatchProjection` |
| `InteriorCD` | `ForceCdLocalRezone` | `CdLocalWinslow` |
| `InteriorSmooth` / `Unknown` | `ForceFullWinslow` | `FullWinslow` |

If all Wave 4 selector inputs are default-off, the hint remains
`ReduceDtOnly` or `RepairOnly`, the legacy retry-halving behavior is preserved,
and no local repair request is emitted.  Explicit
`ForceInteriorMultiNodeRepair` requests run `InteriorMultiNodeProjection` only
when `Numerics.ale.multi_node_interior_repair_enabled=True`; otherwise they
fall back to full Winslow.

By default the retry timestep remains the legacy
\(\Delta t_{retry}=0.5\,\Delta t_{failed}\).  When
`Numerics.hydro.driver_retry_use_suggested_dt_enabled=True` and the typed
hydro failure reports both `suggested_dt > 0` and `trial_scale > 0`, the driver
instead uses
\[
\Delta t_{retry} =
\min(0.5\,\Delta t_{failed},\ \Delta t_{suggested})
=
\min(0.5\,\Delta t_{failed},\ 0.7\,\sigma\,\Delta t_{failed}),
\]
where \(\sigma=\texttt{trial\_scale}\) is the analytic corner-J admissible
scale from §3.2.13a.  The quantities are times in seconds; the scale and 0.7
safety factor are dimensionless.

Stage 24 adds a default-off retry dispatcher,
`Numerics.hydro.dispatcher_state_sensitive_bypass_enabled`.  The typed
corner-J guard classifies axis failures as state-sensitive when the
step-start same-cell relative corner-J margin
\(g_{\rm rel}=\min_k J_{c,k}/\max_k |J_{c,k}|\) is already below
\(\epsilon_{\rm rel}=10^{-6}\); otherwise failures whose trial margin crosses
that threshold with a positive admissible dt estimate are dt-sensitive.  In
`Driver::run`, the retry
state machine uses `RepairSameDt` for state-sensitive `AxisFace`/`AxisBand`
failures, forcing `AxisSpinePlusLocal` without halving dt; it uses
`HalveToDtStar` for dt-sensitive axis failures; all other cases fall through
to the existing retry selector.  Same-dt state repairs are capped per outer
step by `dispatcher_state_sensitive_repair_cap_per_step` (default 3, valid
range ≥1).  If a state-sensitive repair is followed by an accepted step, the
dt floor-stall counter is reset.  When a pending repair plan and retry-active
mesh repair are both present, the pending repair request supplies the ALE mode
and focus cell; the active-repair balance remains diagnostic for that attempt.

#### 3.2.13c Mesh-failure attribution diagnostics (Phase 2 Mesh Stability Wave 5)

`Numerics.diagnostics.mesh_attribution.*` is a default-off D6 investigation
path.  When both `enabled=True` and `record_node_displacements=True`, each
`Hydro2D::lagrangian_step` invocation snapshots the start-of-invocation node
positions and accumulates per-source nodal displacement deltas for
`KinematicCarry`, `HydroPressure`, and `ArtificialViscosity` along the direct
2D coordinate update path.  ALE rezone/remap entry points expose attribution
hooks as direct sources when an attribution workspace is active.  Laser,
conduction, FLD, S_N, and raytrace deposition callbacks are stub-only in this
wave: their mesh effect is attributed through the later hydro pressure/viscosity
coordinate update rather than as an immediate direct mesh displacement.

For a failing cell corner with start edge vectors \(a_0,b_0\), source nodal
delta edge vectors \(\delta a_s,\delta b_s\), and candidate corner Jacobian
\(J_1\), the linearized source contribution is
\[
\delta J_s =
\det(\delta a_s,b_0) + \det(a_0,\delta b_s).
\]
The dominant source is the source with the largest positive degradation
\(\max(0,-\delta J_s)\).  The diagnostic also records
\(\sum_s\delta J_s - (J_1-J_0)\) as a nonlinear residual, which captures
boundary constraints, axis preflight scaling, higher-order geometry terms, and
currently-unattributed direct motion.

Failure records are written to a separate JSONL stream,
`mesh_failure_attribution.jsonl`, using the same output-directory convention as
`dt_lineage.jsonl`.  Records include `step` and `t_simulation` for joining with
the corresponding `dt_lineage` step record; the dt-lineage schema is unchanged.
With the default `dump_on_failure_only=True`, records are emitted only on mesh
failure.  With `dump_on_failure_only=False`, one aggregate record is emitted per
Hydro2D invocation with `failing_cell=-1` and per-source values equal to the
sum over cells of positive degradation at each cell's most-degraded corner.

#### 3.2.13d ICF shell and operator energy diagnostics (Stage 27 Wave C)

`Numerics.diagnostics.icf.*` is default-off and also becomes effective when
`Numerics.profile.icf_standard_ale.enabled=True`.  The shell radius uses the
same cylindrical RZ mass-weighted convention as the implosion history:
cells with \(\rho_i \ge 0.1\rho_{\max}\) contribute
\[
R_{\mathrm{shell}} =
\frac{\sum_i R_i m_i}{\sum_i m_i}.
\]
The shell thickness is a threshold width, not a standard deviation:
\[
\Delta R_{\mathrm{shell}} = R_{\mathrm{outer}}(\rho>\rho_o) -
R_{\mathrm{inner}}(\rho>\rho_i).
\]
If the configured thresholds are zero, \(\rho_o=0.1\rho_{\max}\) and
\(\rho_i=0.5\rho_{\max}\).  The emitted diagnostics are
\[
\mathrm{IFAR} = \frac{R_{\mathrm{shell}}}{\Delta R_{\mathrm{shell}}},
\qquad
\mathrm{CR} = \frac{R_{\mathrm{initial}}}{R_{\mathrm{shell}}},
\]
with the first valid history sample capturing \(R_{\mathrm{initial}}\).  Rows
with non-positive or non-finite \(\Delta R_{\mathrm{shell}}\) are invalid and
are not used for IFAR/CR interpretation.

`Numerics.diagnostics.conservation.enabled` is also default-off and effective
under the ICF ALE profile.  For each operator with a clean before/after state
boundary and explicit external energy \(\Delta E_{\mathrm{ext}}\), the reported
relative residual is
\[
\epsilon_E =
\frac{E_{\mathrm{after}} - E_{\mathrm{before}} - \Delta E_{\mathrm{ext}}}
{\max(|E_{\mathrm{before}}|, |E_{\mathrm{after}}|,
      |\Delta E_{\mathrm{ext}}|, 1)}.
\]
Here \(E=E_{\mathrm{thermal}}+E_{\mathrm{kinetic}}\) is captured by the existing
diagnostic energy-total reductions.  Operators without a defensible
\(\Delta E_{\mathrm{ext}}\) are skipped rather than assigned a fabricated
residual.  Unit-test tolerance \(10^{-12}\) applies only to synthetic
host-controlled transitions; it is not a GPU integration-run acceptance
threshold.
Leave-one-out replay is API-stubbed only in this wave and returns `not_run`.

`TENRYU_I1B_ENERGY_AUDIT` enables a diagnostic-only global discrete audit for
I1-B runs.  The default is off; `TENRYU_I1B_ENERGY_AUDIT_EVERY` throttles log
emission when enabled.  The audit is read-only and uses the existing fields:
the hydro diagnostic energy totals for \(K,U_e,U_i\), `state.mass` for cell
mass, the 2D hydro `corner_mass` nodal-mass basis for momentum, and the
deterministic radiation-energy total \(\sum_{c,g}\max(E_{g,c},0)V_c\).  The
logged total is
\[
E_{\mathrm{tot}} = K + \sum_c m_c e_{i,c} + \sum_c m_c e_{e,c}
                  + E_{\mathrm{rad}}.
\]
The EOS specific internal energies are used as stored in `ee`/`ei`; no
additional ionization/excitation term is added outside the EOS writeback.
The mass and momenta are
\[
M=\sum_c m_c,\qquad
P_r=\sum_a m_a u_{r,a},\qquad
P_z=\sum_a m_a u_{z,a},
\]
where \(m_a\) is reconstructed from the same cached `corner_mass` and CSR
node-incidence rules used by the 2D hydro nodal-mass update.

For an outer pressure boundary the audit logs two independent drive-work
estimates.  Method A reproduces the existing RZ pressure-force endpoint area
vectors at \(t^{n+1/2}\) and forms
\[
W_A=\sum_{a\in\partial\Omega_p}\mathbf{F}_{p,a}\cdot
(\mathbf{x}^{n+1}_a-\mathbf{x}^n_a).
\]
Method B forms the swept quad \(Q_f=[A^n,B^n,B^{n+1},A^{n+1}]\) on each driven
face and uses the existing exact RZ polygon-volume routine,
\[
\Delta V_f=-V_{\mathrm{RZ}}(Q_f),\qquad
W_B=\sum_f -p_f^{n+1/2}\Delta V_f,
\]
so inward compression gives positive work.  Radiation loss \(L_{\mathrm{rad}}\)
uses the existing escaped-radiation step tallies plus laser escape; \(Q_{\rm ext}\)
uses existing Marshak, volume-source, and laser-input tallies.  The per-step
and cumulative residuals are
\[
R_n=(E_{\mathrm{tot}}^{n+1}-E_{\mathrm{tot}}^n)-W_A-Q_{\rm ext}+L_{\rm rad},
\qquad
\eta_E=\frac{|\sum_n R_n|}
{\max(|E_{\mathrm{tot}}^0|,|E_{\mathrm{tot}}^N|,
      \sum_n|W_A|,\sum_n|Q_{\rm ext}|,\sum_n|L_{\rm rad}|)}.
\]
The audit also force-logs on accepted steps where the center-patch reference is
the identity skip or an ALE axis/reference fallback was reported.

`Numerics.diagnostics.hotspot_gas.enabled` is a default-off, measurement-only
2D_RZ hotspot tracer diagnostic.  At initialization it tags the initial gas
region by cell centroid,
\[
Y_{g,c} =
\begin{cases}
1, & \sqrt{r_c^2+z_c^2} < R_g,\\
0, & \text{otherwise},
\end{cases}
\]
where \(R_g=\) `R_g_cm`.  The tracer is inert during the Lagrangian phase.  In
the CSR conservative ALE remap it transports the extensive gas tracer mass
\[
Q_{g,c}=m_cY_{g,c}
\]
with the same donor selection, swept-volume sign convention, and mass
positivity-limited fluxes as the hydro mass remap.  After remap
\(Y_g=Q_g/m\), clipped to \([0,1]\).  The diagnostic reports the relative drift
of \(\sum_c Q_{g,c}\) against the initialized gas tracer mass; it does not feed
back into density, velocity, energy, EOS, CFL, or rezone decisions.

Phase-1 hotspot compression metrics are geometry-light cell reductions using
only \(\{Y_g,m,V,\rho,p,r_c,z_c\}\).  Gas-mass quantile radii \(R_q\) are
weighted quantiles of \(\sqrt{r_c^2+z_c^2}\) with weights \(m_cY_{g,c}\), and
\(\mathrm{CR}_q=R_g/R_q\) for \(q=50,90,95,99\).  These centroid-radius
compressions are retained as legacy diagnostic fields; they are not
mesh-motion-invariant under indirect ALE.  The normalized legacy median-radius
metric is
\[
C_{R50,\mathrm{norm}}(t)=\mathrm{CR}_{50}(t)/\mathrm{CR}_{50}(0).
\]
The mesh-motion-invariant hotspot-gas compression gate is the tagged gas
volume-density metric
\[
M_g(t)=\sum_c m_c\,\mathrm{clamp}_{[0,1]}(Y_{g,c}),\qquad
V_g(t)=\sum_c V_c\,\mathrm{clamp}_{[0,1]}(Y_{g,c}),
\]
\[
\bar\rho_g(t)=M_g(t)/V_g(t),\qquad
\mathrm{CR}_V(t)=\left[\bar\rho_g(t)/\bar\rho_g(0)\right]^{1/3}.
\]
The mass-weighted density-median compression is
\[
\mathrm{CR}_{\rho50}(t)=
\left[\rho_{50,g}(t)/\rho_{50,g}(0)\right]^{1/3},
\]
where \(\rho_{50,g}\) is the weighted median of cell density with weights
\(m_cY_{g,c}\).  The RMS compression metric is
\[
R_{\mathrm{rms}} =
\sqrt{\frac{5}{3}\frac{\sum_c m_cY_{g,c}(r_c^2+z_c^2)}
                         {\sum_c m_cY_{g,c}}},
\qquad
\mathrm{CR}_{\mathrm{rms}} = R_g/R_{\mathrm{rms}} .
\]
The diagnostic also emits \(\bar\rho_g=\sum mY_g/\sum Y_gV\), the
gas-mass-weighted mean density, gas-mass percentiles of \(\rho\),
gas-mass-weighted pressure
\(p=P_e+P_i\), and \(K=p/\rho^\gamma\) mean/percentiles.

Stagnation temperature summaries are gas-mass weighted over the same tracer
control volume:
\[
\langle T_e\rangle_g =
{\sum_c m_cY_{g,c}T_{e,c}\over \sum_c m_cY_{g,c}},\qquad
T_{e,p}=\mathrm{wquantile}_p(T_{e,c};\,m_cY_{g,c}),
\]
with \(p=10,50,90\).  In two-temperature runs the same definitions are emitted
for \(T_i\).  In one-temperature runs the ion-temperature fields are not
written; the scalar `hotspot_Ti_valid` is 0 in history/snapshot diagnostics.

For Phase A stagnation work there is no unique boundary for the diffuse
tracer-defined hotspot control volume, so TENRYU does not fabricate a
pressure-work surface integral.  The emitted `hotspot_work_proxy_*` fields are
an explicitly named total-material-energy-change proxy:
\[
U_g(t)=\sum_c m_cY_{g,c}\left(e_{e,c}+e_{i,c}\right),\qquad
K_g(t)=\sum_c {1\over2}m_cY_{g,c}\,|\mathbf{u}_c|^2,
\]
\[
W^{proxy}_g(t)=\left[U_g(t)+K_g(t)\right]-
               \left[U_g(0)+K_g(0)\right].
\]
For 1D, \(\mathbf{u}_c\) is the average of the two radial node velocities.  For
2D_RZ, \(\mathbf{u}_c\) is the active cell-node/CSR average of
\((v_r,v_z)\).  While a central pseudo-core is active, its tracer-weighted
aggregate internal energy contributes to \(U_g\); no separate pseudo-core bulk
kinetic energy is invented.  This proxy is conservative-consistent because it
uses existing extensive material energies and velocities, but it includes any
thermal source/sink affecting the tracer gas and is not equal to compatible
boundary \(p\,dV\) work.

Angle-resolved hotspot tracer areal density is
\[
(\rho R)_{g,\theta}=\int_{\ell(\theta)} \rho(\mathbf{x})Y_g(\mathbf{x})\,ds,
\]
using the same ray quadrature as the existing `rho_R` diagnostic.  This
hotspot-tracer integral is independent of the existing `r_range="shell"` mask,
so shell-only total \(\rho R\) does not zero the gas column.  No distinct fuel
tracer exists in the current one-material finite-CR pilot deck; future
multi-material decks should populate parallel fuel/hotspot/shell material
\(\rho R\) fields under the same versioned diagnostic group rather than
redefining `rho_R_hotspot_tracer`.

#### 3.2.13d.1 Center perturbation-energy diagnostic

`Numerics.hydro.center_perturbation_diag_scope` controls a history-cadence
diagnostic.  The default `disabled` scope precedes topology checks,
allocation, device-to-host copies, and writer work; default decks therefore
emit no group.  The legacy
`Numerics.hydro.tri_fan_center_perturbation_diag_enabled=true` namelist key is
parse-time compatibility only: when the scope key is omitted, it maps to
`center_perturbation_diag_scope="tri_fan_first_ring"`.  Runtime dispatch uses
the enum scope.

For `tri_fan_first_ring` on `spherical_polar_halfplane` + `tri_fan`, first-ring
nodes \(i=1,\ j=0,\ldots,n_z\) form the ring-mean velocity
\[
\langle \mathbf{u}\rangle_{i=1}={1\over n_z+1}
\sum_{j=0}^{n_z}(u_R(1,j),u_Z(1,j)),
\qquad
\mathbf{u}'_n=\mathbf{u}_n-\langle \mathbf{u}\rangle_{i=1}.
\]

For `centroid_r_innermost_bins` on multiblock topologies, T7 uses a
deterministic block/layer bin selection rather than a continuous centroid sort.
Bin 0 is all `CENTRAL_CORE` cells.  Bins \(1,\ldots,N_b\) are transition
layers derived from the block table: the 3-block topology uses the `BRIDGE`
block's local layer index, and the 5-block half-butterfly topology unions the
same local layer index across the `NORTH_FAN`, `EAST_FAN`, and `SOUTH_FAN`
blocks.  `POLAR_SHELL` layers enter only for requested bin counts larger than
`1 + multiblock_cart_core_bridge_layers`.  Thus the default
`center_perturbation_diag_radial_bins=2` selects all core cells plus the first
transition layer.  The diagnostic node set is the unique union of corner nodes
incident on selected, hydro-active cells.

In the multiblock scope, TENRYU forms the mass-weighted band-mean velocity
\[
\langle \mathbf{u}\rangle_B={1\over \sum_{n\in B}m_n}
\sum_{n\in B} m_n (u_R(n),u_Z(n)),
\qquad
\mathbf{u}'_n=\mathbf{u}_n-\langle \mathbf{u}\rangle_B .
\]
Both scopes reuse the existing n-level source-split nodal accelerations:
\(\mathbf{a}^P_n\) from \(P_e+P_i\), and \(\mathbf{a}^Q_n\) from capped
`Qvisc`.  With nodal masses \(m_n\) [g], the emitted perturbation scalars are
\[
\dot E'_P=\sum_n m_n\,\mathbf{u}'_n\cdot\mathbf{a}^P_n
\quad [\mathrm{erg/s}],
\]
\[
\dot E'_Q=\sum_n m_n\,\mathbf{u}'_n\cdot\mathbf{a}^Q_n
\quad [\mathrm{erg/s}],
\qquad
E'_k={1\over2}\sum_n m_n|\mathbf{u}'_n|^2
\quad [\mathrm{erg}].
\]
For cells in the configured center band \(i\in[0,\texttt{band}]\), it also
reports `max_q_over_p` (dimensionless), `min_corner_J` (dimensionless), and
`min_cell_volume` [cm\(^3\)]; for multiblock these are evaluated over the same
selected cell band.  The additive HDF5 group is the historical
`/diagnostics/tri_fan_center_perturbation/v1/` path for both active scopes,
with datasets
`Edot_prime_P`, `Edot_prime_Q`, `Eprime_k`, `max_q_over_p`,
`min_corner_J`, `min_cell_volume`, `valid`, `cycle`, and `time_s`.

Interpretation is originator-vs-amplifier: \(\dot E'_Q>0\) dominating before
\(E'_k\) grows indicates AV-driven perturbation origin; \(\dot E'_Q\) growing
only after \(\mathbf{u}'\) grows indicates AV amplification of an existing mode.

#### 3.2.13d.2 S2 multiblock runtime feature gate

The S2 runtime-active multiblock hydro layer is limited to 2D_RZ hydro with
either Lagrangian motion or S4-T1-next ALE remap.  For
`Mesh.topology_scheme="multiblock_half_butterfly_5block"`, B-S2 retargets the
ALE/remap, pressure-BC, path-admissibility, and diagnostics consumers from
three-block structured assumptions to block-role, CSR, and
`cell_orientation_sign` metadata.  The equations in
§3.2.0a-§3.2.13d.1 document the in-tree
pre-T9 implementation; the production `tenryu run` feature gate is planned for
T9 and is specified in SPECIFICATION §6.4.2.  After T9, selecting
`Mesh.topology_scheme="multiblock_cart_core_polar_shell"` is intended to reject
decks that enable conduction, radiation, laser, PLIC, disabled hydro,
unsupported mesh motion, a dimension other than `2D_RZ`, or r-face state-supply.
S4-T1-next T7 allows `mesh.motion="ale"` and routes multiblock per-step ALE
through `apply_ale` to `apply_multiblock_csr_ale_step`, where `every_n_steps`
gates CSR Winslow rezone plus CSR conservative remap. S4-T7 lifted the
trial-volume and mesh-quality dt CFL knobs as
multiblock no-ops. S4-T6 lifted the pressure outer-shell drive by adding a
polar-shell-specific dispatch.

This gate is a scope boundary, not a new discretization.  Single-block remains
the default byte-identical path.  Multiblock runs that pass the S2 predicate use
the CSR geometry, exact corner masses, deterministic node mass, compatible
pressure/AV work, topology-aware q-cap, multiblock CFL, center-band diagnostic,
S4-T6 polar-shell pressure-BC drive, and the B-S2 five-block hydro retarget
described above.  The accepted B-S2 five-block smoke is a closed, uniform,
at-rest 2T state; it proves the five-block hydro and ALE/remap core reaches
`t_end` without inversion, dt-floor abort, path-admissibility rejection, or
conservation/GCL drift at rest.  Seam flux under gradients, source-term
coupling beyond the pressure drive, r-face state-supply, and production
compression closure are deferred to B-S3/B-S4 or later.

### §3.2.x Multiblock seam GCL + symmetry + conservation 4-gate framework

The S3 architecture pivot defines four verification gates for the γ MVP
multiblock topology, addressing distinct aspects of correctness. S3 adds no
configuration keys; the mesh contract remains SPECIFICATION §6.4.2 and hydro
defaults remain SPECIFICATION §9.1.

| Gate | Test | Threshold | Status |
|------|------|-----------|--------|
| G1 | Constant-state seam GCL | rho, Pe, Pi <= 1e-12 rel; v <= 1e-9 abs (FP floor); conservation <= 1e-13 | PASS |
| G2 | Uniform-pressure force balance (interior nodes) | max \|F_n\|/(P*max Svec) <= 1e-12 | PASS |
| G3 | Homothetic three-ring symmetry | t=0 strict 1e-10; 0<t<=2.5e-7 s <= 0.15 (γ MVP geometric floor) | PASS |
| G4 | Dynamic spherical smoke | 3-block: above 1e-4 floor A4_2N <= 0.5*A4_N; 5-block: compatible-path energy <= 1e-12, conservation <= 1e-13, Gauss-J/RZ-volume admissible; blast-profile W recorded as characterization only | PASS |

#### S4-T1-next T6 multiblock ALE production path

S4-T1-next T6 removes the S4-T1 passthrough/no-op marker path for multiblock
ALE. When `mesh_topo_is_multiblock(cfg.mesh)` and
`cfg.numerics.ale.enabled = true`, the production driver uses CSR mesh
admissibility, CSR Winslow rezone, and CSR conservative remap rather than any
structured-indexed ALE kernel.

With `Numerics.ale.multiblock_cross_seam_rezone_enabled=false` (default), the
rezone phase is T5a per-block CSR Winslow: seam-shared nodes are not smoothed
across block boundaries, but the ALE cycle still uses the CSR remap path. With
the flag set true, T5b cross-seam CSR Winslow also moves seam-shared nodes
using incident seam-cell neighbors. Reference-barrier ALE backtracking uses CSR
candidate admissibility and the same CSR conservative remap for multiblock
states.

#### 3.2.13f Phase 2 multiblock path-admissibility dt rejection

When `Numerics.ale.multiblock_path_admissibility_enabled=true`, Hydro2D checks
the multiblock Lagrangian path before committing predictor and corrector node
positions.  The predictor uses
\(\mathbf{x}^{n}+\lambda({1\over2}\Delta t\,\mathbf{u}^{1/2})\); the
corrector uses
\(\mathbf{x}^{n}+\lambda(\Delta t\,\mathbf{u}^{1/2})\).  This check is part of
the Lagrangian position update and is not gated by `mesh.motion="ale"` or by
whether a later ALE remap is enabled.  For each CSR quad cell and node
coordinate
\[
\mathbf{x}_i(\lambda)=\mathbf{x}_i^n+
\lambda\left(\mathbf{x}_i^{n+1}-\mathbf{x}_i^n\right),\qquad 0\le\lambda\le1,
\]
TENRYU evaluates signed corner Jacobians, 2x2 Gauss-point Jacobians, and the
signed polygon area along the full path.  Each predicate is quadratic in
\(\lambda\), because each coordinate is linear:
\[
J(\lambda)=J_0+\lambda J_1+\lambda^2J_2 .
\]
The implementation evaluates the exact minimum over \([0,1]\) by testing both
endpoints and the interior vertex when \(J_2>0\).  Orientation is normalized by
\(\mathrm{sign}(J_0)\), so clockwise and counter-clockwise cell orderings use
the same positive margin convention.  The acceptance margin for each metric is
\[
m=\min_{\lambda\in[0,1]}
\left[\mathrm{sign}(J_0)J(\lambda)\right]
- f_\mathrm{path}|J_0|,
\]
where \(f_\mathrm{path}\) is `path_admissibility_floor`.  This margin is
reported as the diagnostic minimum over all cells and all metrics.

**Reference-baseline floors (A-09, 2026-07-27).** When the relative floors are enabled, the
baseline is the reference-mesh value \(q_{\mathrm{ref}}\) (computed with the same discrete
formula on \(x^{\mathrm{ref}}\)) rather than the previous-mesh value \(q^{n}\):
\(q^{n+1}\ge f\,q_{\mathrm{ref}}\). This removes the ratcheting admitted by the
previous-mesh baseline (\(q^{n+1}\ge f\,q^{n}\) permits unbounded geometric decay across
accepted steps). Cells whose reference value is unavailable or degenerate (unpopulated
reference arrays, \(q_{\mathrm{ref}}\le 0\), non-finite, or orientation mismatch) fall back
to the previous-mesh baseline for that cell. Absolute floors and crossing checks are
unchanged.

Independently, TENRYU solves
\(\mathrm{sign}(J_0)J(\lambda)-f_\mathrm{path}|J_0|=0\) and records the
smallest root \(\lambda_*\in(0,1]\) over the four corner Jacobians, four
Gauss-point Jacobians, and signed polygon area.  A root at \(\lambda_*=1\) does
not reduce the timestep; a root with \(\lambda_*<1\) is a predict-before-commit
geometric rejection.

On failure, Hydro2D returns a soft failure with reason
`multiblock_path_admissibility`, retry action `ReduceDtOnly`, and suggested
\[
\Delta t'=\max(\Delta t_\min,\theta\lambda_*\Delta t),
\]
where \(\theta\) is `dt_rejection_factor`.  The coupling driver restores the
full step-start snapshot and re-executes the split operators and hydro
predictor / corrector at the smaller dt; pressure forces, artificial-viscosity work,
accelerations, velocities, positions, and subsequent ALE remap therefore remain
consistent with the accepted dt.  The retry reason uses
`max_dt_rejections` rather than the generic hydro retry limit.  Defaults keep
this path disabled, preserving single-block, tri_fan, and legacy multiblock
behavior unless the deck opts in.

Two I1-B Q2 pole diagnostics are environment-gated and default off. With
`TENRYU_I1B_PATH_ADMIS_ANATOMY=1`, the rejection record also carries the
metric kind that supplied the earliest rejecting root:
edge-cross, 2x2 Gauss-J, area/RZ-volume, or free-node R guard.  The same
record stores the winning cell's multiblock block id and block-local
\((i,j)\).  This is diagnostic-only and does not change the admissibility
floor, roots, or accepted path.

`TENRYU_I1B_MESH_FORECAST_DIAG` is a separate default-off Stage-A
mesh-survival forecast diagnostic.  It evaluates the same constrained
Lagrangian geometry that the Hydro2D predictor/corrector will commit: after
axis preflight, pole-motion overlay, and the BBSW/PAVA order constraint, the
position-velocity buffers \(w'_r,w'_z\) define
\[
\mathbf{x}_i(\tau)=\mathbf{x}_i^n+\tau\Delta t_s\mathbf{w}'_i,\qquad
0\le\tau\le1,
\]
with \(\Delta t_s={1\over2}\Delta t\) for the predictor and
\(\Delta t_s=\Delta t\) for the corrector.  For a candidate-coordinate check,
\(\mathbf{x}_i(\tau)=\mathbf{x}_i^n+\tau(\mathbf{x}_i^{trial}-\mathbf{x}_i^n)\).
For each active CSR cell, the diagnostic quality is
\[
q_c(\tau)=\min(q_J,q_V,q_{edge},q_R,q_\phi).
\]
\(q_J\) is the minimum oriented corner-Jacobian sine
\(\det(e_{1,k},e_{2,k})/(|e_{1,k}||e_{2,k}|)\).  \(q_V\) is the signed exact
RZ-volume ratio
\[
V_c^{RZ}={\pi\over3}\sum_k(r_k+r_{k+1})(r_kz_{k+1}-r_{k+1}z_k),
\qquad q_V={s_V V_c^{RZ}(\tau)\over
\max(f_\mathrm{path}|V_0|,10^{-300})}.
\]
Here \(s_V\) is the same orientation convention as the active
path-admissibility predicate, falling back to \(\mathrm{sign}(V_0)\) when
predicate hardening is not active.
\(q_{edge}\) uses the same edge-cross products as the path predicate,
normalized by the two segment lengths.  In the `POLAR_SHELL` block only,
\(q_R\) is the minimum radial-ring ordering ratio
\((R_{i+1,j}-R_{i,j})/(R_{i+1,j}^0-R_{i,j}^0)\), with
\(R=\sqrt{r^2+z^2}\), and \(q_\phi\) is the analogous angular ordering ratio
using \(\phi=\mathrm{atan2}(r,z)\).  Non-shell cells treat \(q_R,q_\phi\) as
not applicable.

The diagnostic returns/logs `endpoint_valid` and `tau_zero` from the existing
path-admissibility predicate, so it cannot change which steps fail.
`tau_warn` is the first sampled-and-bisected \(\tau\) where
\(q_c(\tau)\le q_\mathrm{warn}\), with `q_warn` set by
`TENRYU_I1B_MESH_FORECAST_Q_WARN` (default \(5\times10^{-2}\)); cadence is
`TENRYU_I1B_MESH_FORECAST_EVERY_N` or a numeric
`TENRYU_I1B_MESH_FORECAST_DIAG` value.  The emitted record includes
`q_min_now`, `q_min_end`, `q_min_path`, the first warning cell and block-local
\((i,j)\), component bits, seed count, per-component values at the warning
cell, and a five-point \(q(\tau)\) trace.  The seed mask is diagnostic-only and
does not trigger rezone, remap, dt rejection, or step acceptance.

`Numerics.hydro.ring7_quotient_enabled` is the default-off I1-B
Ring7OuterSeamQuotientRemap. With the flag off, Hydro2D returns before any
Ring7 allocation or mesh/state mutation. With the flag on, Hydro2D first
performs the Increment-1 diagnostic scan when
`TENRYU_I1B_RING7_QUOTIENT_DIAG=1`. Increment 2 also installs a zero-time seam
transaction after StepStart geometry refresh and before the ordinary Hydro2D
predictor setup, but the transaction returns before seam allocation, metric
evaluation, dt reduction, or state mutation unless a one-shot request is
already armed. If the ordinary predictor/corrector production
mesh-quality dt-CFL later rejects with `mesh_quality_rz_volume`, or the
multiblock path oracle rejects with `multiblock_path_admissibility`, on a cell
in the Ring7 seam patch, the driver restores the pre-step retry snapshot, sets
a single-use Ring7 repair request carrying the failing cell, and retries the
full step at the same `dt`. The same request is armed proactively for the next
step when the accepted HydroStepResult path-oracle minimum is owned by a Ring7
seam-patch cell and falls below `TENRYU_I1B_RING7_PATH_MARGIN_TRIGGER` (default
`0.05`). `TENRYU_I1B_RING7_REZONE_COOLDOWN_STEPS` (default `3`) suppresses
repeat proactive arming for the configured number of request steps; hard
production rejections still route immediately through the retry path. Interior
seam failures are routed to the seam packet remap. Driven pole-cap failures
(`POLAR_SHELL` ring 7, local `j=0`, domain boundary face; deterministically
cell 608) are not routed to the seam remap: that boundary collapse is a
mesh-motion problem, not an interior-seam conservative remap problem.
Only a requested transaction discovers the outer `POLAR_SHELL` ring patch and
applies the symmetry-paired tangential seam-node line search. If the requested
rezone cannot find an admissible target, control falls back to the existing
driver dt-reduction path. The hard RZ-volume barrier
uses `compute_multiblock_mesh_quality_rz_volume_cell_margins_with_volume()`,
which launches the same multiblock RZ-volume root kernel and helper functions
as the production `mesh_quality_rz_volume` dt limiter. Candidate node
coordinates are passed as the limiter start coordinates, candidate cell
volumes are passed as the floor-volume field, and the nominal velocity field is
checked at both \(\Delta t_\mathrm{nom}\) and
\(\eta_\star\Delta t_\mathrm{nom}\). If no path-admissible candidate satisfies
the swept-volume cap and \(\eta_V\ge\eta_\star\), Hydro2D returns a
`mesh_quality_rz_volume` dt-reduction result rather than falling through as if
the seam were fixed.

Increment 3a removes the Ring7 seam from the global ALE remap contract. The
accepted seam target is not written to `State.x_r/x_z`, not written to
`State.x_r_reference/x_z_reference`, and not passed to `ale_remap_2d_rz()`.
Instead, the StepStart request builds a dedicated seam-only packet-geometry
ledger over a closed `Ring7SeamPatch` and logs diagnostics before returning
without state or coordinate mutation. The driver then continues the same retry
attempt; because the mesh is unchanged, the production mesh-quality limiter is
still the only component that rejects the candidate geometry and selects the
dt-reduction fallback in Increment 3a.

The dedicated patch ledger promotes every cell whose multiblock
`cell_node_csr` node list references any moved seam target node into
`patch_cells`. On the I1-B mesh the `POLAR_SHELL`--butterfly seam uses shared
global node ids, so this direct incidence rule includes the butterfly-side
strip without a face-growth pass. Internal faces are then enumerated only from
the canonical `MultiBlockTopology::unique_internal_faces` physical-face list,
so each `POLAR_SHELL`--butterfly adjacency is counted once. Boundary faces
must have zero swept volume; nodes appearing on topology `boundary_faces` are
hard-frozen back to their source coordinates before packet diagnostics,
otherwise the diagnostic records an invariant violation and the offending face.
For each
owned internal face, the signed RZ swept volume is split into corner packets
using the existing
`State.corner_volume` edge-corner weights; packets are sorted by canonical
donor/acceptor/corner ids for deterministic future application, but 3a does
not apply mass, momentum, energy, tracer, or coordinate updates. The central
pseudo-core is only inspected through topology/state masks: patch cells and
moved patch nodes must not overlap central-core connectivity, and before/after
central-core hashes plus aggregate mass, total energy, and momentum sums are
logged unchanged because no mutation occurs.

Each diagnostic transaction logs `[ring7_seam_packet]` with patch-cell,
internal-face, boundary-face, moved-node, physical seam-face, and packet
counts; shared-vs-duplicate seam-node counts from the unique physical seam
faces; invariant-violation counts; boundary swept-volume sums; per-cell
geometry closure
\(V^\mathrm{target}_c-V^\mathrm{source}_c-\sum_p\Delta V_{p,c}\);
central-core hash/sum deltas; and the achieved target `min_q_J`, `min_q_edge`,
and `min_eta_V` from the optimizer. The seam motion remains paired under
\(\Pi:(r,z)\mapsto(r,-z)\) by computing the tangential displacement and cloning
\((\delta r,\delta z)\mapsto(\delta r,-\delta z)\) onto mirror nodes before
the packet geometry is built.
The RZ geometry primitives used by the remap are
\[
\mathbf{S}_f=\pi(r_0+r_1)\,(z_1-z_0,\,-(r_1-r_0))
\]
for the oriented axisymmetric area vector of a meridional face
\((r_0,z_0)\rightarrow(r_1,z_1)\), and
\[
\delta V(P)={\pi\over3}\sum_k(r_k+r_{k+1})
  (r_kz_{k+1}-r_{k+1}z_k)
\]
for the signed axisymmetric swept volume of an ordered meridional polygon.
Packet sums for quotient transactions use fixed-order compensated
accumulation. Increment 3a validates the dedicated packet ledger geometry;
Increment 3b applies it conservatively after the pre-commit gates pass.

Increment 3b enables the default-off commit path for the same closed packet
ledger. The transaction first requires the Increment-3a geometry gates
(`invariant_violations==0`, zero nonzero-swept boundary faces, and round-off
cell-volume closure). It then applies the sorted corner packets to host working
copies only: donor density is the current working cell mass divided by current
working cell volume, each packet transfers
\(\delta M=\rho_d\delta V\), \(\delta\mathbf{P}=\delta M\mathbf{u}_d\),
\(\delta E=\delta M(e_d+\tfrac12|\mathbf{u}_d|^2)\), electron-internal
fraction mass, gas-tracer mass when initialized, and \(Z\)-mass. Corner mass
and nodal momentum are updated in the same fixed order, with no unordered
atomics. A geometric positivity gate requires each donor corner's outgoing
swept volume for the current geometric submove to remain below its source
corner volume. If needed, the displacement
\(\mathbf{x}^{old}\rightarrow\mathbf{x}^{new}\) is split into \(K\) equal
coordinate segments; for each segment TENRYU rebuilds the closed-patch swept
packet geometry from \(\mathbf{x}^{(k-1)}\rightarrow\mathbf{x}^{(k)}\), checks
that submove's positivity, and applies that submove to the working state before
advancing to the next segment (`TENRYU_I1B_RING7_PACKET_SUBCYCLE_KMAX`, default
8; `TENRYU_I1B_RING7_PACKET_POS_EPS`, default \(10^{-12}\)). If all geometric
subcycle counts fail, the target displacement is retried with dyadic
\(\lambda\)-backoff down to `TENRYU_I1B_RING7_PACKET_LAMBDA_MIN` (default 0.1;
`TENRYU_I1B_RING7_PACKET_LAMBDA_BACKTRACKS`, default 5). No clipping or floor
injection is allowed: if mass, target density, tracer bounds, or internal
energy floors fail, the transaction rejects and returns the existing
dt-reduction fallback without any state write.

After packet accumulation, each remapped cell recovers internal energy from the
conserved total energy minus the final corner-mass/nodal-velocity kinetic
energy; the kinetic-energy mixing loss remains in the recovered internal
energy. Electron and ion specific energies are split by the transported
electron-internal fraction and the ideal-gas pressure/temperature closure is
refreshed for the remapped cells. Whole-domain mass, total energy, and paired
momentum deltas plus central-core hash/sum deltas are checked before commit.
Only if those checks pass are `State.x_r/x_z`, cell conservative fields,
node velocities, and transported corner masses copied back, followed by a mesh
geometry/`State.vol` refresh. The successful commit logs
`[ring7_seam_remap] fired=1 commit=1` with the committed subcycle count and
\(\lambda\); rejected attempts log the same tag with `commit=0` and fall back
to dt reduction.

Increment 4a adds a diagnostic-only driven-pole cap oracle for the cell-608
class of failures. When the production `mesh_quality_rz_volume` limiter rejects
a small-sigma (`TENRYU_I1B_POLE_CAP_SIGMA_TRIGGER_MAX`, default \(10^{-2}\))
driven-pole boundary cell, Hydro2D retains the exact limiter velocity field
used by the failed attempt and the driver retries the restored step once with a
single-use pole-cap oracle request. At StepStart, the oracle builds a small
logical pole cap around the failing shell cell (`TENRYU_I1B_POLE_CAP_LAYERS_MIN`
default 1, `TENRYU_I1B_POLE_CAP_LAYERS_MAX` default 2), constructs candidate
mesh velocities with \(w_z=\alpha u_z\) tapered from
`TENRYU_I1B_POLE_CAP_ALPHA_CENTER` (default 0.0) at the failing cell to
Lagrangian motion at the cap boundary, and also probes
`TENRYU_I1B_POLE_CAP_ALPHA_SECOND` (default 0.25). The tangential \(r\)
component remains Lagrangian in 4a. For each option, the diagnostic calls the
same production `compute_mesh_quality_dt_limit()` used by Hydro2D, first with
the retained failed velocity as the Lagrangian baseline and then with the
candidate cap mesh velocity. It also reports the cell-local RZ-volume
production margin for the failing cell. The `[ring7_pole_cap]` record is
diagnostic-only: it does not write coordinates, does not remap state, and does
not force a StepStart dt reduction. The unchanged retried step therefore falls
back to the existing production limiter rejection and dt-reduction path.

Increment 4b enables the selected driven-pole cap candidate. For the passing
option (default one logical layer and `TENRYU_I1B_POLE_CAP_ALPHA_CENTER=0`),
TENRYU forms the last-failed Lagrangian material geometry
\(\mathbf{x}_{lag}=\mathbf{x}^n+\Delta t\,\mathbf{u}^{1/2}\) and the cap mesh
geometry \(\mathbf{x}_{cap}=\mathbf{x}^n+\Delta t\,\mathbf{w}^{1/2}\) on cap
nodes, with \(\mathbf{x}_{cap}=\mathbf{x}_{lag}\) bit-for-bit outside the cap
so the relative ALE displacement is localized. A separate pole-cap packet
transaction remaps from
\(\mathbf{x}_{lag}\rightarrow\mathbf{x}_{cap}\) using the same deterministic
corner-extensive internal-face packets as the seam path. Cap cell volumes are
the exact RZ polygon volumes at `x_lag` and `x_cap`, and every active cap-face
swept contribution uses the swept-quad RZ polygon
\([x^\mathrm{lag}_a,x^\mathrm{lag}_b,x^\mathrm{cap}_b,x^\mathrm{cap}_a]\)
with the cell orientation applied. Thus domain-boundary swept volume in the
driven cap contributes to the target cell volume ledger but does not create an
off-domain mass/energy packet; the `[ring7_pole_cap_closure]` diagnostic dumps
the cell-608 exact volumes, the active polygon boundary ledger, and the older
edge-difference value side by side to localize any near-axis mismatch. Internal
cap faces transport mass, momentum, total energy, gas tracer,
electron-internal fraction, and \(Z\)-mass from donor to acceptor in fixed
sorted order. The domain-boundary volume ledger also applies pressure work
\(\Delta U_\mathrm{bdry}=-p_\mathrm{ext}\Delta V_\mathrm{bdry}\), using the
same `boundary_pressure` table value at `State.t` that drives the StepStart
pressure-boundary force path. The total-energy gate therefore compares the
post-transaction total energy against the pre-transaction total plus this
boundary work. The pole cap rejects before commit
if cap-boundary closure, donor-corner positivity, recovered internal-energy
floors, whole-domain mass/total-energy and paired-momentum deltas, or
central-core hash/sum isolation fail. Geometry subcycling is controlled by
`TENRYU_I1B_POLE_CAP_PACKET_SUBCYCLE_KMAX` (default 8) and
`TENRYU_I1B_POLE_CAP_PACKET_POS_EPS` (default \(10^{-12}\)). A successful
transaction commits only the cap-node coordinates from `x_cap` while leaving
off-cap coordinates at their StepStart values; the normal hydro step still owns
the full-mesh Lagrangian position update. It also commits the remapped
conservative fields, node velocities/corner masses, and refreshed mesh
geometry, then marks a pending `[ring7_pole_cap_remap]` validation. The next
Hydro2D production
limiter evaluation at the same trial dt logs whether the recomputed retry
velocity passes the RZ-volume limiter at the requested pole-cap cell; this log
does not alter the production limiter result.

`TENRYU_I1B_SHELL_SUBCYCLE=1` is a default-off Stage-C1 commit mode for the
same hydro-level replay-probe path used by the shell protected rezone
diagnostic.  Unlike the restore-only replay diagnostic, the subcycle mode is
not one-shot: each accepted `post_corrector_commit` forecast may attempt one
protected shell rezone after the Lagrangian mesh is already valid.  If the
post-commit forecast has `endpoint_valid=true`, `seed_count>0`, and
\[
q_\mathrm{floor}\le q_\mathrm{current}\le q_\mathrm{warn},
\]
where \(q_\mathrm{current}\) is the replay probe's post-commit current forecast
quality (`q_min_end` for the committed-coordinate check), the hydro probe
captures a rollback snapshot and applies the conservative
`shell_protected_rezone` to the current mesh using the forecast seed mask.  No
guarded partial step is run, and no path-admissibility failure is required; an
actual hydro failure continues through the existing full-step retry/dt-rejection
path.

After a candidate shell rezone, the hydro probe refreshes post-commit geometry
and density, refreshes hourglass subzonal masses, canonicalizes the Option-B
corner mass basis, resets volume-rate CFL history, and updates the void mask
before running hard gates on active-cell mass/volume/density positivity,
finite node velocity, finite hydro CFL, optional node-mass floor, and
post-versus-pre CFL/velocity regression.  A subcycle commit requires
\(q_\mathrm{after}>q_\mathrm{before}\), \(q_\mathrm{after}>0\), valid rezone
path, round-off mass/momentum/energy conservation, and no CFL or node-speed
regression beyond
`TENRYU_I1B_SHELL_SUBCYCLE_POST_REZONE_REL_TOL` (default \(3\times10^{-2}\)).
Restoring \(q_\mathrm{release}\) remains a logged metric but is not a commit
gate, so a partial positive quality repair can commit.  The optional node-mass
gate is `TENRYU_I1B_SHELL_SUBCYCLE_NODE_MASS_FLOOR_G` (default \(0\),
disabled), and the round-off conservation tolerance is
`TENRYU_I1B_SHELL_SUBCYCLE_CONSERVATION_TOL` (default \(10^{-10}\)).  Any
rezone, refresh, or audit failure restores the local pre-rezone snapshot and
allows the step to continue without the subcycle commit.  Each attempted
subcycle logs `[shell_subcycle]` with before/after \(q\), `committed=1/0`,
`gate=pass/rollback`, mass/momentum/energy deltas, pre/post CFL dt, pre/post
max node speed, and min node mass.  A passing subcycle rezone is committed
inside the hydro post-commit point and the driver skips the normal multiblock
ALE call for that step because the protected rezone already installed and
remapped the shell mesh.

`TENRYU_I1B_PATH_PREDICATE_HARDEN=1` is a separate default-off Stage-0
predicate hardening gate.  When unset, the live fine-cell path predicate keeps
the legacy sign-from-current-metric root path and does not copy
`cell_orientation_sign` into the path-admissibility kernels.  The scaled
coincidence tolerance, zero-length segment rejection, axis-coincident endpoint
guard, canonicalized loop nodes, trifan star-simple macro guard, and pole-macro
axis-closure handling are also inactive when this env is unset.  Genuine
single-apex cap loops still use the two-crossing star-simple guard.  A
one-radial-row pole-shell wedge instead keeps its two \(R=0\) pole-side nodes
distinct as a radial axis edge; its exact RZ polygon volume is evaluated on the
full boundary, and its edge-crossing predicate tests that \(R=0\) axis edge
like any other nonadjacent boundary edge.  Adjacent endpoint contacts remain
topologically allowed, but an off-axis nonadjacent edge crossing the axis-edge
interior is a real invalidity.  When the hardening is enabled, a
lambda-zero old-geometry degeneracy is not treated as a CFL condition.  A
genuine pole-axis radial-order inversion, identified by adjacent logical
polar-shell pole rows with \(|s_q|\not<|s_{q+1}|\), is reported separately as
`pole_axis_radial_order_inversion` with no suggested dt reduction so the driver
can run the conservative pole-axis radial-order repair. Other lambda-zero
macro-boundary degeneracies remain `macro_repair_required`, the angular/span
repair signal for the experimental de-ref path.

With `TENRYU_I1B_POLE_COARSEN_PILOT=1`, Hydro2D path checks install a
topology-only POLAR_SHELL pilot overlay for the shell radial band
\(q\in[\texttt{TENRYU_I1B_POLE_COARSEN_Q_MIN},
\texttt{TENRYU_I1B_POLE_COARSEN_Q_MAX}]\), default \(q=7\).  The angular
quotient starts with dyadic span-2 macros and may escalate simple-loop failures
to span 4 and then span 8 (`TENRYU_I1B_POLE_COARSEN_LEVEL_MAX=3` by default).
Each accepted macro boundary is the true exterior perimeter of its
\([q_\min,q_\max]\times[j_0,j_1]\) cell block: angular arcs contain only the
endpoint nodes \(j_0,j_1\), while radial sides retain intermediate q-row nodes
if the macro spans multiple shell rows.  Intermediate angular nodes are skipped
only when their fine child cells are covered by an accepted macro.

Accepted pole macros build a separate `pole_coarsen_inactive_fine_mask`; the
path checker ORs that mask with the central pseudo-core inactive-member mask
before launching the CUDA fine-cell scan.  The accepted macro boundary paths
are then evaluated on host and merged with the cell-scan result, so the macro
replaces, rather than supplements, its covered fine children.  State arrays,
material velocities, central pseudo-core masks, and later ALE/remap operators
are unchanged.  The overlay is not a conservation model.  A macro failure is
reported as `pole_coarsen_path_admissibility` with span, level, q-range,
j-range, skipped-node count, old/new macro volume, and simple-loop anatomy;
otherwise residual path failures are from cells not covered by accepted macros.

With `TENRYU_I1B_POLE_MOTION_PILOT=1`, Hydro2D uses the same POLAR_SHELL
q-band and dyadic macro acceptance, but controls only the mesh position
velocity \(w\), not the material velocity \(u\).  After the predictor
\(u^{1/2}\) is formed and axis preflight has run, and again after the
corrector velocity plus axis preflight, Hydro2D copies the candidate
position velocities into separate \(w_r,w_z\) buffers.  State-supply and
axis-tangential z-position constraints are applied to \(w_z\) as before.
For each accepted macro \([j_0,j_1]\), endpoint nodes keep their hydro
candidate motion.  Intermediate fine nodes on q-band node rows
\(i=q_{\min},\ldots,q_{\max}+1\) are reconstructed by linear interpolation of
the accepted endpoint candidate positions:
\[
X^c_{i,j}=(1-\eta)X^h_{i,j_0}+\eta X^h_{i,j_1},\qquad
\eta={j-j_0\over j_1-j_0},
\]
where \(X=(r,z)\) and \(X^h=X^n+\Delta t\,w^h\) is the unconstrained hydro
candidate position.  Inward transition node rows use a coherence weight
\(\alpha(q)\) set by `TENRYU_I1B_POLE_MOTION_PROFILE=smoothstep` and
`TENRYU_I1B_POLE_MOTION_TRANSITION_ROWS` (default 4).  Let
\(q_{\rm full}=q_{\min}\); the configured row count is the number of nonzero
partial rows, so the last fully free row is
\(q_{\rm free}=q_{\rm full}-N_{\rm trans}-1\).  Then
\[
\alpha(q)=S\left({q-q_{\rm free}\over q_{\rm full}-q_{\rm free}}\right),
\qquad S(s)=3s^2-2s^3,
\]
\(\alpha\) is clamped to 0 for \(q\le q_{\rm free}\) and to 1 for
\(q\ge q_{\rm full}\).  The transition-row candidate is
\[
X'_{q,j}=(1-\alpha(q))X^h_{q,j}+\alpha(q)X^c_{q,j}.
\]
For `Q_MIN=6` and `TRANSITION_ROWS=4`, the partial rows are \(q=2,3,4,5\)
with smoothstep weights about \(0.104,0.352,0.648,0.896\); \(q\le1\) is free
and \(q\ge6\) is fully coherent.  The committed position velocity is then
\(w'=(X'-X^n)/\Delta t\).  The same
\(w'_r,w'_z\) buffers feed mesh-quality dt limiting, path admissibility, and
the position commit, so the checked and committed mesh agree.  `state.v_*`
and compatible CCH force/work remain the material velocity path; this pilot is
a geometry/path de-risk gate and is not a conservation model.  The first
applied overwrite log records the profile, free/full q rows, and per-row
\(\alpha(q)\) values; with path-admissibility anatomy enabled, later
rejections report `metric_local_i`, exposing whether the rejection row moved
inward or vanished.

When the motion pilot supplies the pole overlay to path admissibility, the
overlay sets `skip_fine_child_paths=false`: accepted macro boundaries may still
be evaluated, but covered fine cells remain in the CUDA fine-cell scan.  Thus
the path check remains the inversion safeguard for all fine cells.

#### G1: Constant-state seam GCL

γ multiblock + uniform rho/Pe/Pi/v=0 + reflect BC + at least 5 Strang
half-steps must preserve the constant state. The gate asserts maximum relative
deviation per non-zero field <= 1e-12. For v=0 fields it uses <= 1e-9
absolute, reflecting the FP accumulation floor over N=18 steps plus
multi-block CSR aggregation:
machine_eps * max(F/m) * dt * sqrt(N_step) * N_corner.

Conservation residuals for mass, R-momentum, Z-momentum, and GCL must all be
machine-zero. This confirms that Newton's 3rd law remains globally preserved
across the seam aggregation.  On multiblock meshes the constant-state GCL audit
uses CSR cell-node lookup for node coordinates and velocity averaging rather
than structured `(i,j)` addressing, so the same audit covers three-block and
five-block topologies.  The S3 cap variant additionally checks the
`multiblock_half_butterfly_trifan_cap_5block` topology with the same field and
conservation thresholds, and verifies that cap/fan faces are ordinary
`unique_internal_faces` while triangle local face 0 is absent from active face
payloads.

#### G2: Uniform-pressure force balance

For uniform pressure, the per-node force at INTERIOR nodes, excluding
`NODE_BOUNDARY` and `NODE_CENTER`, should be zero by the compatible-work
identity. A single force-assembly call evaluates

```text
scale_n = max_{(c,k) in rcsr[n]} (|Svec_r[c,k]| + |Svec_z[c,k]|) * P_total
residual_n = |F_n| / scale_n
```

and the maximum residual over all interior nodes must be <= 1e-12, i.e. the
machine-precision floor. The gate is tested for both static v=0 and uniform
constant non-zero velocity (`v_r=2.5e4`, `v_z=-1.5e4`) to cover Galilean
invariance.

#### G3: Homothetic three-ring symmetry

The gate samples three block-local rings at carefully chosen physical radii,
not one shared sphere:

```text
s_core   = 0.75 * r_c
s_bridge = 0.5 * (sqrt(2)*r_c + r_match)
s_shell  = 0.5 * (r_match + s_max)
```

The γ MVP topology does not support a single sphere radius intersecting all
three blocks: core supports `s <= sqrt(2)*r_c ~= 1.41*r_c`, while shell
supports `s >= r_match = 2*r_c`. There is no overlap, so G3 uses three
independent rings, one per block.

For each ring, the homothetic radial velocity is `v = alpha*x` with
`alpha = -1e5 1/s`, a 5% contraction over `t_end=5e-7 s`. The gate samples
`4*N_c` angular bins, 32 at `N_c=8`, and computes cosine DFT modes
`m=0..2*N_c`. The mode ratio uses the analytical homothetic amplitude in the
denominator,
`|a_m| / |alpha * r_test * (1 - |alpha|*t)|`, to avoid `a_0` near-zero
crossing artifacts.

Thresholds are strict 1e-10 at `t=0` for initial-condition angular spectrum
purity, and 0.15 for `0 < t <= 2.5e-7 s`. The 0.15 floor is architectural:
the Hermite bridge maps a half-square, the Cartesian core outer boundary with
theta=pi/4 and 3*pi/4 corners, to a half-circle, the polar shell inner ring at
uniform `dtheta`. This four-fold angular discontinuity in the Jacobian
introduces inherent m=2 and m=4 Fourier modes of about 5-7% under non-uniform
flow. This is not treated as an implementation bug for γ MVP.

Late-time check times `t > 2.5e-7 s` are recorded in diagnostic JSON but not
enforced. Dynamic wave-like mesh-distortion accumulation produces
time-specific amplitude spikes, about 70% at core+bridge `t=3.75e-7 s` and
about 3% in shell at the same time, which require S4 production-scale
wave-dynamics investigation.

#### G4: Dynamic spherical smoke A4 convergence

The dynamic smoke gate is a two-resolution refinement check, `N_c=8` and 16.
It initializes an interior pressure pulse with `Pe=Pi=2x` ambient for
`s <= s_max/20`, using a cos^2 taper to ambient by `s_max/10`, and propagates
to `t_ref = 0.7*s_max/c_s ~= 2.68e-7 s`. The measured quantity is A4, the
cosine-DFT m=4 amplitude of rho, on reference ring
`s_ref = 0.7*s_max = 0.049 cm` with `4*N_c` samples.

Two-tier threshold:

- If `A4_{N_c=8} >= 1e-4`, require `A4_{N_c=16} <= 0.5*A4_{N_c=8}`, no
  m=1..8 amplitude growth under refinement, and no ratio in the `[0.5, 0.8]`
  plateau zone.
- If `A4_{N_c=8} < 1e-4`, ratio is N/A; verify `all_state_positive` and mode
  bounds within 2x floor.

The 1e-4 γ MVP architectural floor has the same root as G3: the Hermite bridge
four-fold angular structure injects about 1e-5 to 1e-4 architectural amplitude
into the rho field under dynamic flow. Refinement from `N_c=8` to 16 doubles
cells, but the bridge corners, four in the half-plane, are unchanged.
Empirical measurement at `N_c=8` gives `A4 = 7.14e-6`; at `N_c=16`,
`A4 = 4.26e-5`, a slight increase under refinement, so this is
architecture-limited rather than implementation-limited.

The earlier consensus 1e-8 floor was over-optimistic for γ MVP. Empirical
calibration at 1e-4 reflects the actual architectural noise. Per consensus
Round-2 self-disclosed weakest point: pulse amplitude is empirical; the floor
rule prevents false failure but may need calibration.

The five-block G4 extension uses the same gentle pressure pulse, pure
Lagrangian motion, and `t_ref`, but switches to
`topology_scheme="multiblock_half_butterfly_5block"` and checks `N_c=8` and
16 with the v3 CSR topology. It runs on the B-S4 compatible hydro path
(`av_model="csw_edge"` and `subzonal_pressure_enabled=True`) with the
production `mesh_quality_dt_cfl_enabled=True` guard. It does not inherit the
three-block Hermite-bridge architectural floor. The dynamic profile observable
is the centered angular profile L2 on the final `s_ref` ring for rho and
signed-radial velocity
\[
u_s = \frac{R v_R + Z v_Z}{\sqrt{R^2+Z^2}}.
\]
For rho the denominator is the sampled mean. For `u_s` the denominator is
`max(|mean(u_s)|, 0.01*c_s)`, where `c_s` is the ambient ideal-gas sound speed;
this is a physical velocity floor that prevents the denominator collapse seen
at radial-velocity reversal without asserting an architectural floor. Define
`W(N_c)=max(L2_rho,L2_u_s)` at `s_ref`. The five-block sidecar records
`W(8)`, `W(16)`, `W(16)/W(8)`, and observed order
`p=log2(W(8)/W(16))`, but this blast-profile metric is characterization-only:
the signed-radial profile is sensitive to the Cartesian-core grid imprint of
this core-pulse deck and is not a B-S4 readiness gate. The same test requires
strictly positive rho, Te, Ti, Pe, Pi, and committed volume at sampled
snapshots; positivity history with no negative scan; mesh admissibility by
positive 2x2 Gauss-J, positive RZ volume, and zero negative RZ-volume history
count, with corner-J warning-only; zero escape-valve count; and production
conservation/history residuals at the existing audit scale (`1e-13` for
mass/R-momentum/Z-momentum/GCL and `1e-12` for compatible force/work energy).

#### Multiblock outer/seam boundary constraint ordering (T1+T3-FIX2, S4-T6)

For each node `n` in the multiblock `apply_boundary_2d` dispatcher:

```cpp
if (node_flags[n] & NODE_CENTER) {
    v_r[n] = 0;
    v_z[n] = 0;
    return;  // highest priority: center pin
}

if (node_flags[n] & (NODE_AXIS | NODE_POLE_AXIS)) {
    v_r[n] = 0;
    // Fall through only to the physical outer-shell check.
}

if (node_flags[n] & NODE_OUTER_PHYSICAL_BOUNDARY) {
    switch (r_outer_type) {
    case FIXED:
        v_r[n] = 0;
        v_z[n] = 0;
        break;
    case REFLECT:
        remove_spherical_normal_component(v_r[n], v_z[n], R, Z);
        break;
    case FREE:
    case PRESSURE:
        break;  // pressure/free allow normal motion
    case STATE_SUPPLY:
    default:
        remove_spherical_normal_component(v_r[n], v_z[n], R, Z);
        break;
    }
    return;  // do not apply the seam tangent projection below
}

if (node_flags[n] & (NODE_AXIS | NODE_POLE_AXIS)) {
    return;  // interior Z-axis keeps v_z free
}

if (node_flags[n] & NODE_BOUNDARY) {
    s = sqrt(R*R + Z*Z);
    if (s > 0) {
        n_hat = (R/s, Z/s);
        v_n = dot(v, n_hat);
        v -= v_n * n_hat;  // tangent-only velocity preserved
    }
}
```

The fall-through from axis to physical outer-shell handling preserves the
outer pole constraint. At the pole-outer corner
`POLE_AXIS | NODE_OUTER_PHYSICAL_BOUNDARY`, where `R=0` and `Z=+-s_max`,
`r_outer="reflect"` gives `v=(0,0)`: the axis constraint zeros `v_r`, then the
physical-shell reflect branch projects against `n_hat ~= +-Z` and zeros `v_z`.
Interior axis nodes are different: even though they carry `NODE_BOUNDARY`,
they return before the seam tangent projection, because the physical RZ-axis
condition constrains only cylindrical-radial velocity. With
`r_outer="pressure"`, the outer physical node skips the tangent projection so
the pressure drive can move the shell normally.

The same sequence and `r_outer` dispatch table are applied to acceleration in
the multiblock force/accel update path by
`apply_boundary_accel_constraints_multiblock_kernel`: \(a_R\) is zeroed on the
axis, pole-outer nodes still receive the outer physical constraint, and
interior axis \(a_Z\) is free.

#### Outer-shell Svec tangent balance (T3-FIX3)

For multiblock-only outer-shell cells, the cell Svec discrete-Pappus
derivative is augmented with a tangent-balance correction at the outer-shell
boundary to eliminate spurious tangential force in the uniform-pressure null
mode. The correction is applied unconditionally as a pure-geometry post-pass
whenever the multiblock Svec arrays are recomputed — it is not gated on any
pressure-field detection, and the post-balance arrays feed both force
assembly and \(\dot V_c\), so the compatible-work identity is preserved by
construction (§3.2.5; the earlier "only when uniform-pressure detection is
active" wording matched no implementation — corrected per 2026-07-26 audit
k02 F-11/§15-7). Single-block and tri_fan paths are unchanged.

#### 3.2.13e Mesh-degeneracy forensics diagnostic (Phase A)

`Numerics.diagnostics.mesh_degeneracy_forensics.*` is a default-off diagnostic
only path for repeated pre-commit `mesh_quality_*` / `in_hydro_*` failures.
It does not change forces, coordinates, timestep selection, HDF5 output, or
accepted-step criteria.  When enabled, Hydro captures the failing cell's four
nodes at the rejected candidate state, and the driver retry path emits a JSONL
record when either the same `(cell, corner, stage)` has failed repeatedly or
the reported safe path scale is below the configured threshold.

For a failing corner \(q\), the diagnostic samples the same candidate path used
by the pre-commit guards:
\[
\mathbf{x}_k(\sigma)=\mathbf{x}_{k,0}
  +\sigma(\mathbf{x}_{k,1}-\mathbf{x}_{k,0}),\qquad 0\le\sigma\le1,
\]
\[
J_q(\sigma)=
\det\left(\mathbf{x}_{q+1}(\sigma)-\mathbf{x}_q(\sigma),\,
          \mathbf{x}_{q-1}(\sigma)-\mathbf{x}_q(\sigma)\right).
\]
The record stores \(J(0)\), \(J(1)\), \(J_\mathrm{floor}\), samples at
\(\sigma=\{0,0.1,\sigma_\mathrm{safe},0.5,1\}\), and a CPU recomputed first
floor crossing.  The finite-difference path velocity invariant is recorded as
\[
\frac{\Delta \mathbf{x}_k}{\Delta t}
=\frac{\mathbf{x}_{k,1}-\mathbf{x}_{k,0}}{\Delta t}.
\]
If the repeated failure is a true geometric near-degeneracy, the expected
signature is that \(\sigma_\mathrm{safe}\) remains bounded while both
\(J(0)-J_\mathrm{floor}\) and \(|J(1)-J(0)|\) shrink with the retried
\(\Delta t\).

The hourglass modal amplitude is extracted from the bilinear fit
\[
u(\xi,\eta)=a_0+a_\xi\xi+a_\eta\eta+a_{\xi\eta}\xi\eta,
\qquad
(\xi,\eta)\in\{(-1,-1),(1,-1),(1,1),(-1,1)\},
\]
so
\[
a_{\xi\eta}=\frac{u_0-u_1+u_2-u_3}{4}.
\]
The same coefficient is stored for candidate position and
\(\Delta \mathbf{x}/\Delta t\), along with ratios to the affine gradient
amplitude and local sound speed.  With `Numerics.hydro.hourglass.enabled=True`,
the nodal acceleration fields record the Phase B anti-hourglass contribution
from §3.2.9b; otherwise `a_hourglass=0` preserves the Phase A default-off
diagnostic path.

Phase D-1 adds an independent default-off corner-J source budget under
`Numerics.diagnostics.mesh_degeneracy_forensics.corner_j_source_budget_enabled`.
For each corner \(q\), the budget records a first-order projection
\[
\Delta J_q^{(s)} \simeq
\sum_n \frac{\partial J_q}{\partial r_n}\Delta r_n^{(s)}
      + \frac{\partial J_q}{\partial z_n}\Delta z_n^{(s)}
\]
for pressure, scalar VNR \(Q\), Phase B anti-hourglass force, predictor,
corrector, ALE rezone, and remap.  For
\[
J_q=(r_{q+1}-r_q)(z_{q-1}-z_q)
    -(z_{q+1}-z_q)(r_{q-1}-r_q),
\]
the nonzero derivatives are
\[
\begin{aligned}
\partial_{r_{q+1}}J_q&=z_{q-1}-z_q,&
\partial_{z_{q+1}}J_q&=-(r_{q-1}-r_q),\\
\partial_{r_{q-1}}J_q&=-(z_{q+1}-z_q),&
\partial_{z_{q-1}}J_q&=r_{q+1}-r_q,\\
\partial_{r_q}J_q&=z_{q+1}-z_{q-1},&
\partial_{z_q}J_q&=r_{q-1}-r_{q+1}.
\end{aligned}
\]
The hydro source terms use the same cgs position-displacement units as the
corner-J sensitivity: the predictor half-step projection uses
\(\Delta\mathbf{x}^{(s)}=\frac14\mathbf{a}^{(s)}\Delta t^2\), and the
corrector/full-step projection uses
\(\Delta\mathbf{x}^{(s)}=\frac12\mathbf{a}^{(s)}\Delta t^2\).  These
coefficients follow the implemented staggered Hydro2D update; they keep
\(\Delta\mathbf{x}\) in cm and leave the force update unchanged.  The
predictor and corrector budget arrays are total geometric displacements for
those substeps; the dominant-source classifier subtracts pressure/Q/hourglass
from those totals before assigning `predictor_other` or `corrector_other`.

When Phase D-1 is active, the driver may snapshot node coordinates at
`post_lagrange`, `post_ale_rezone`, and `post_remap` within the current retry
attempt/step.  ALE and remap \(\Delta J\) use these direct coordinate
differences; remap is normally zero because the implemented remap changes
cell-centered intensive/conservative quantities and not mesh coordinates.
No snapshot allocation or JSON fields are produced when
`corner_j_source_budget_enabled=False`.

The local work terms are diagnostic bookkeeping only:
\[
W_{PdV}=-(P_e+P_i)\Delta V,\qquad W_Q=-Q\Delta V,
\]
and
\[
W_{F\cdot v}\approx
\sum_{k\in cell} m_k(\mathbf{a}_{P,k}+\mathbf{a}_{Q,k})
\cdot\frac{\mathbf{u}_{k,old}+\mathbf{u}_{k,new}}{2}\Delta t.
\]
The reported residual is
\[
\Delta K_{4nodes}-W_{F\cdot v}.
\]
It is a local consistency indicator for later Phase B compatible-energy
comparison, not a global conservation criterion.

#### 3.2.14 2D RZ 境界条件

**(a) r = 0 対称軸**
- 速度：v_r(r=0, z) = 0（軸上ノードのr方向速度をゼロに固定）
- v_z は自由（対称軸上でも z 方向運動は許可）
- コーナー力：軸上ノード k に対し、F_{r,k} = 0 を強制（r方向の力を消去）
- 圧力ゴースト：不要（対称軸は面ではなく点 r=0 の正則化で処理）
- 熱伝導：∂T_e/∂r|_{r=0} = 0（Neumann）
- 放射：反射境界（§8.2）
- Annular 2D_RZ (`Mesh.r_min > Numerics.axis_eps_cm`) では inner R
  boundary は物理軸ではなく reflective wall である。`r_inner="reflect"`
  を推奨し、legacy `r_inner="axis"` は reflective semantics としてのみ
  許容する。この場合 `Numerics.has_physical_rz_axis=False` となり、
  axis-repair kernels は実行されない。
- `boundary_2d.mesh_tangential_target="reference"` では、clamped r-face
  boundary node の z 座標を IC reference mesh の z 座標へ戻す。
  既定 `"lagrangian"` では従来通り r-face tangential coordinate は
  material-following のまま保持する。

**(b) r = r_max 外側境界**
- 自由境界（`hydro.boundary` で指定）：
  - `"free"`：外側ゴーストセルの圧力 P_ghost = 0、速度は外挿
  - `"fixed"`：v_r = 0, v_z = 0（速度ゼロ壁）
  - `"pressure"`：P_ghost = P_drive(t)（時間依存駆動圧）。
    P_drive(t) はナムリストの Python callable から生成された **FrozenTable1D**（ARCHITECTURE §8 Step 3）として提供される。
    単位は dyne/cm\(^2\)。テーブル範囲外（\(t < 0\) または \(t > t_{end}\)）では \(P = 0\) とする（free境界と等価、SPECIFICATION §6.4.7 参照）。
- Pressure traction orientation and endpoint split: for an outer boundary edge
  \(a\to b\), with
  \(\Delta r_f=r_b-r_a\), \(\Delta z_f=z_b-z_a\), the legacy per-node RZ
  face-area vector is
  \[
  \mathbf{S}_f=\pi\bar r_f(\Delta z_f,\,-\Delta r_f),
  \]
  whose magnitude is one half of the swept face area
  \(A_f=2\pi\bar r_f|\Delta\mathbf{x}_f|\).  Before applying the pressure
  traction, \(\mathbf{S}_f\) is oriented with the adjacent interior cell:
  if \(\mathbf{S}_f\cdot(\mathbf{x}_{face}-\mathbf{x}_{cell})<0\), its sign is
  flipped.  Rectangular `r_outer` faces ordered in increasing \(z\) already
  have \(\mathbf{S}_f\) outward and are unchanged.  `spherical_polar_halfplane`
  outer faces ordered in increasing \(\theta\) produce the opposite raw
  orientation, so this check flips them to outward; the resulting pressure
  force points inward along \(-\hat{\mathbf e}_s\).

  For rectangular faces, the boundary force added to each edge node remains the
  legacy \(-P_{drive}(t)\mathbf{S}_{f,out}\).  For every
  `spherical_polar_halfplane` `r_outer` edge, the endpoint split uses the exact
  RZ vectors
  \[
  \mathbf{G}_a={\pi\over3}(2r_a+r_b)(\Delta z_f,\,-\Delta r_f),\qquad
  \mathbf{G}_b={\pi\over3}(r_a+2r_b)(\Delta z_f,\,-\Delta r_f),
  \]
  with the same outward sign flip as \(\mathbf{S}_f\).  The nodal boundary
  forces are \(-P_{drive}(t)\mathbf{G}_{a,out}\) and
  \(-P_{drive}(t)\mathbf{G}_{b,out}\).  This changes only the spherical-polar
  endpoint distribution: \(\mathbf{G}_a+\mathbf{G}_b
  =\pi(r_a+r_b)(\Delta z_f,\,-\Delta r_f)=2\mathbf{S}_f\), so the edge-total
  pressure traction is conserved relative to the legacy midpoint split.  The
  Hydro2D acceleration path, force/work audit diagnostic path, and precise RZ
  geometric-CFL half-velocity path use the same gated endpoint rule.
  For `Mesh.topology_scheme="multiblock_cart_core_polar_shell"`, S4-T6 uses
  `launch_multiblock_polar_shell_pressure_forces` instead of the structured
  `r_outer` launcher.  The shell-local node view begins at
  `MultiBlockTopology.n_nodes_core + n_nodes_bridge_interior`; within that view
  shell nodes retain the polar ordering \(q(N_\theta+1)+k\).  The multiblock
  launcher applies the same area-vector and endpoint rules only to
  \(q=N_r\), so core nodes, bridge nodes, and interior shell rings receive zero
  contribution from this boundary condition. The same \(q=N_r\) shell nodes are
  tagged `NODE_OUTER_PHYSICAL_BOUNDARY`; the multiblock velocity and
  acceleration constraint kernels dispatch these nodes by `r_outer` before the
  seam `NODE_BOUNDARY` projection. `r_outer="pressure"` and `"free"` leave the
  normal component unconstrained, `"reflect"` removes it, `"fixed"` zeros both
  components, and `"state_supply"`/unknown fall back to reflect semantics.
- コーナー力：外側ノード k に対し、ゴーストセル側の力寄与はゼロまたは P_ghost に基づく
- 熱伝導：∂T_e/∂r|_{r=r_max} = 0（断熱壁）またはユーザー指定
- 放射：ユーザー指定（§8、vacuum/Marshak/reflect）

**(c) z = z_min 下端境界**
- 自由境界（既定 `"free"`）：P_ghost = 0
- `"reflect"`：v_z(z=z_min) = 0、v_r は自由
- `"state_supply"`（dict 形式のみ）：`boundary_2d.cu::apply_state_supply_z_bottom_node_kernel` が下端ノードの z 座標を z_min に固定し、mesh node velocity としての v_z を 0 に設定する。これは mesh anchoring であり material velocity をゼロ化しない。`state_supply_bc.cu::override_state_supply_kernel` / `restore_state_supply_material_velocity` が境界 row cell の material `v_z` を `u_z_cm_per_s` へ復元し、この supplied material velocity を hydro flux、diagnostics、reservoir tally に用いる。既定 `boundary_2d.mesh_tangential_target="lagrangian"` では x_r/v_r は z-face 境界では変更しない。`"reference"` では ALE Winslow rezone 後に z-face boundary node の x_r を IC reference mesh 位置へ戻す。
- コーナー力：下端ノード k に対し、ゴーストセル側の力寄与を P_ghost で計算

**(d) z = z_max 上端境界**
- z_min と同様の選択肢。`"state_supply"` では `boundary_2d.cu::apply_state_supply_z_top_node_kernel` が上端ノードの z 座標を z_max に固定し、mesh node velocity としての v_z を 0 に設定する。material velocity は `state_supply_bc.cu` の override/restore path が `u_z_cm_per_s` へ復元する。`boundary_2d.mesh_tangential_target="reference"` では上端 node の x_r も IC reference mesh 位置へ戻す。

**(d2) z-face state-supply 境界セル**
- `Numerics.hydro.boundary.{z_bottom,z_top}` に `{"type":"state_supply", "rho_g_per_cc": rho_s, "u_z_cm_per_s": u_s, "T_eV": T_s}` を指定した場合、対応する境界セル row（下端 j=0、上端 j=nz-1）を reservoir-supplied cell として扱う。
- Lagrangian corrector 後、および radiation/Qei/conduction block 後に、境界セルの `rho` と `mass=rho_s*V_current`、`Te=Ti=T_s` を再設定し、既存 EOS closure で `ee/ei/Pe/Pi` を一貫して再計算する。Wave 1 では single-T 供給のみをサポートし、2T mode でも `Te=Ti=T_s` とする。
- この境界は保存形ではない。各 override で reservoir から注入/除去された `delta_mass`、`delta_E_total`、`delta_p_z` を step/cumulative tally として集計し、verbose energy budget で報告する。`delta_E_total` は override 前に保存した `mass,ee,ei,u_z` と、EOS closure 後の post state を用いて評価する。`pre_uz` には supply `u_z_cm_per_s` を使い、tally と restored material velocity の規約を一致させる。
- state-supply z-face の open-flow 保存フラックスは supplied material state \((\rho_s,u_{z,s},T_s)\) から作る。ALE remap kernel の signed face-speed convention に合わせた法線速度を \(u_{n,s}\) と書くと、advective mass flux は
  \[
  \Delta M_f=\rho_su_{n,s}A_f\Delta t .
  \]
  Rankine-Hugoniot flux audit uses the corresponding planar Euler/FLD invariants
  \[
  F_M=\rho_su_{z,s},\qquad
  F_P=\rho_su_{z,s}^2+P_{\mathrm{gas},s}+P_{\mathrm{rad},s},
  \]
  \[
  F_E=\rho_su_{z,s}\left(e_s+\frac{1}{2}u_{z,s}^2+
       \frac{P_{\mathrm{gas},s}}{\rho_s}\right)+F_{\mathrm{rad},s}.
  \]
  境界 mesh position が z_min/z_max に clamped されても \(u_{n,s}\) はゼロではなく、active state-supply face では conservative remap が projected node `v_z` ではなく `supply_u_z_cm_per_s` を boundary face speed として直接使う。
- ALE open-flow remap では、outflow 時の interior donor は既定 `boundary_2d.state_supply_donor_mode="interior_per_i"` で各 i の境界 row cell state \((\rho,u_z,e_e+e_i,E_{rad})\) を使う。`"interior_radial_average"` では remap flux pre-pass が同じ row の i 方向算術平均 donor を計算し、全 i の outflow donor に同じ平均値を用いる。inflow 時の reservoir donor はこの mode に依存しない。
- Wave 1 では z-face のみ有効であり、r-face/1D の `state_supply` は不正入力である。PR 4.6 以降は z-face state-supply と 2D_RZ ALE の併用を許可する。ALE rezone は interior nodes を移動できるが、state-supply z-boundary nodes は z_min/z_max に固定され、mesh velocity と material velocity は分離される。Hydro2D position update は temporary `predictor_pos_z` / `corrector_pos_z` buffers だけをゼロ化して clamped z-boundary node を動かさず、material `state.v_z` は supply velocity を保持する。PR A 以降、`boundary_2d.mesh_tangential_target="reference"` は clamped r-face node の z 座標と clamped z-face node の r 座標を `State.x_*_reference` へ戻す。T2 以降の conservative remap はこの open-flow flux contract（state-supply z-face で rho*(u_n-w_n)*A*dt）を active state-supply faces に適用し、boundary face speed には `supply_u_z_cm_per_s` を直接使う。
- RH invariant audit は snap 0001 で mass、z-momentum、energy face flux の相対誤差が \(10^{-3}\) 以下であること、および `supply_u_z_cm_per_s` の正符号が z_bottom/z_top の両方で保たれることを検査する。

**(e) コーナーノード（2境界の交点）**
- 両方の境界条件を同時に適用。例：r=0 かつ z=z_min のノードでは v_r = 0 かつ v_z = 0（反射の場合）
- 優先順位：r=0 対称軸の v_r=0 は常に適用（他の境界条件より優先）。`state_supply` は mesh-normal z 成分（x_z/v_z）を設定する。`mesh_tangential_target="reference"` のときだけ、BC-tangential 成分（z-face では x_r、r-face では x_z）を reference mesh 座標へ戻すため、corner nodes は幾何学的に固定されるが material-tied ではなくなる。

### §3.2.y Pentagon-belt shell topology — ALE P2 (2026-07-28)

Polar-shell mesh with K∈[1,4] static angular-coarsening rings ("belts"): pentagon
transition cells implement 2:1 zoning between angular bands, killing the near-axis
high-aspect killer cells identified by the Phase-0 forensics. Scheme
`Mesh.topology_scheme="pentagon_belt_shell"` (namelist contract: SPEC §6.4.2).

**Topology contract.** Fixed-width CSR storage with per-mesh slot stride
`corner_stride = corner_stride_for_scheme ∈ {4,8}` (8 for the belt scheme; all other
schemes keep 4 byte-identically). `cell_nverts ∈ [3,8]`; the canonical local-face map is
the legacy tri/quad branches verbatim plus identity for n≥5, with per-branch upper
guards (`local_face` beyond the active count → inactive; required so stride-8 padding
slots are never classified as active faces). Angular zone count per node ring i:
`ntheta_ring(i) = nz >> #{j : b_j ≥ i}` with belt layers b_1<…<b_K (strictly increasing,
separated by ≥1 quad layer, b_j ∈ [1, nr−2], nz divisible by 2^K, nz>>K ≥ 4).
Pentagon row at layer b (coarse count N): cell m has CCW-in-storage nodes
`[inner_m, outer_2m, outer_2m+1, outer_2m+2, inner_m+1]`; edge k connects node k→k+1
(e0/e3 radial, e1/e2 fine outer, e4 inner). All cells carry `cell_orientation_sign=-1`
(the polar (s,θ)→(r,z) node convention is clockwise; POLAR_SHELL precedent). Blocks:
2K+1 (K+1 quad bands + K pentagon rows), role `PENTAGON_BELT`. θ-ladders: uniform or
equal-μ only — coarse rings SUBSAMPLE the finest ladder (`θ_ring_i[j] =
θ_fine[j·nz/n(i)]`), making 2:1 nesting bitwise by construction; the southern half
mirrors northern coordinates exactly.

**Corner masses (n≥5).** Per-vertex RZ volume weights from the F4 star-P1 vertex
r-moments: `w_v = star_p1_vertex_r_moments(cell polygon)`, `m_v = m_cell · w_v / W`,
`W = Σ_v w_v` (fraction convention, matching the quad exact-subpolygon path; no
remainder shift). Degenerate fallback (`W ≤ 0` or non-finite): equal split `m_cell/n`
with the corner-mass fallback probe fired at stage HydroMultiblock, `orient = −3`
(distinguishes the polygon record from the quad's −2). Tri/quad branches are the
pre-existing code verbatim. Tail slots `[max(4,n), corner_stride)` are zeroed
unconditionally.

**Geometry and ring Svec tangent balance.** Cell volume/area/centroid/corner-Svec via
the generic-n RZ polygon forms (the multiblock CSR geometry kernel; capacity-8 vertex
buffers). At an open ring (free/rigid surface) node, the r-weighted discrete surface
vector of the two incident half-edges carries a TANGENTIAL residue (the pole-side and
equator-side half-edges have different r-weights), so a uniform-pressure static gas
would spuriously accelerate tangentially along the ring (measured ≈4%·c_s per step
before the cure). The cart-core outer-shell Svec tangent-balance postpass is
generalized (`balance_ring_svec_tangent`, template on the ring cell-id map; cart-core
call byte-identical) and applied to the belt OUTER ring (quad corner slots 1/2) and
INNER ring (slots 0/3). Interior node force closure is exact without any balance
(node-force audit: max |Σ_cells p·S_corner| ≤ 1e-11·p·s_max on the static belt mesh).

**Inner-ring boundary condition.** New node flag `NODE_INNER_PHYSICAL_BOUNDARY`
(1u<<5) on ring-0 nodes. For `Numerics.hydro.boundary_2d.r_inner="axis"` (the annular
idiom shared with the structured polar path) the multiblock velocity AND acceleration
BCs remove the spherical-normal component at inner-flagged nodes, composing with the
axis handling (a pole∩inner node ends fully pinned). Belt validation requires
r_inner="axis"; other inner BCs staged.

**Energy accounting.** The energy-budget CSR corner-mass kinetic path is
corner_stride/cell_nverts-general; belt meshes with initialized corner masses MUST use
it (fail-loud fence; stride-4 multiblock pre-init keeps the legacy structured
fallback). The 1T energy renormalization therefore operates correctly on belt meshes.

**Staged exclusions (config fail-loud).** Subzonal corner masses, HLLC z-flux, runtime
ALE, S_N transport, conduction (polar hydro-only guard), legacy CSW AV (`av_model=
"csw_edge"`: its opposite-face limiter pairing has no pentagon analog — use
`csw_edge_csw98`, whose multiblock path is nverts-general), explicit/graded θ ladders,
tri_fan/button center treatments. Restart: belt checkpoints write topology v3 + v4
(`corner_stride`, `cell_nverts`) and round-trip bitwise.

**Verification (tests/hydro/test_pentagon_belt_smoke.cu, tests/mesh/
test_pentagon_belt_builder.cu).** Builder counts/orientation/analytic shell volume/
bitwise θ-nesting/F1 oracle (orientation-normalized copy)/HDF5 round-trip; node-force
closure audit; static-gas smoke (25 steps: max|v| < 1e-8·c_s, mass 1e-12, energy
1e-10, corner-mass sums intact); P2-5 mini killer (κ=4, v0=1.5·c_s inward: the
convergent shock crosses the belt row with all cells positive-volume — the P2 exit
criterion at mini scale; production-scale Phase-0 battery and refinement convergence
remain open).

### 3.3 ALE Rezone/Remap（2D RZ）

#### 3.3.1 ALEサイクル

ALE（Arbitrary Lagrangian–Eulerian）はLagrangian計算のメッシュ歪みを緩和する。
各タイムステップでの処理順序：

1. **Lagrangianステップ**（§3.2）：通常のLagrangian hydro更新
2. **メッシュ品質チェック**（§3.3.2）：品質指標を評価
3. **Rezone**（§3.3.3）：品質不足時にメッシュを整形（ノード位置のみ変更）
4. **保存的Remap**（§3.3.4）：新メッシュへ物理量を写像

rezoneの実行頻度は `rezoning.every_n_steps`（既定5）ステップ毎。
`numerics.ale.force_rezone_every_n_steps`（既定0）は診断用の既存
`apply_ale(..., force_rezone=true)` invocation cadenceで、0では無効。

> **`hydro_active` との関係（§2.1.1）**：
> §2.1.1の「座標・速度は固定」は **Lagrangianステップ（§3.2）** において
> 非活性セルが力の寄与を受けず自然に静止することを意味する。
> ALE rezoneは物理演算ではなくメッシュ品質操作であり、`hydro_active` の状態に関わらず
> 全ノードを対象として平滑化を行う。これにより、活性/非活性セルの境界付近での
> メッシュ品質劣化を防止する。remap（§3.3.4）は全 active cell storage を対象とする。
> `polar_center_treatment="button"` では、button cell `c=0` と structured shell
> cells \(i\ge I_{\rm btn}\) が active remap set であり、inner structured storage
> cells \(c\ne0,\ i<I_{\rm btn}\) は dormant として remap donor/target から除外する。

#### 3.3.2 メッシュ品質指標

**ヤコビアン比**：セルの4 Gauss点でヤコビアンを評価し、比を品質指標とする。
参照要素 \([-1,1]^2\) 上の \(2 \times 2\) Gauss-Legendre 求積点：\((\xi_\alpha, \eta_\beta) = (\pm 1/\sqrt{3},\; \pm 1/\sqrt{3})\)。
セル \((i,j)\) の双線形写像：
\[
r(\xi,\eta) = \sum_{k=1}^{4} N_k(\xi,\eta)\, r_k, \quad
z(\xi,\eta) = \sum_{k=1}^{4} N_k(\xi,\eta)\, z_k
\]
ただし \(N_1 = (1-\xi)(1-\eta)/4\), \(N_2 = (1+\xi)(1-\eta)/4\), \(N_3 = (1+\xi)(1+\eta)/4\), \(N_4 = (1-\xi)(1+\eta)/4\)。
ヤコビアン \(J(\xi,\eta) = \partial r/\partial\xi \times \partial z/\partial\eta - \partial r/\partial\eta \times \partial z/\partial\xi\)。
既定の legacy mesh-quality/admissibility gate は bit-exact 互換性のため
\(J < 0\) のみを tangle とし、\(J_{\max} < k_{J,\mathrm{floor}}\) の場合は
退化セルとして扱う。`Numerics.ale.reject_zero_gauss_j=true` では
\[
J_{\mathrm{floor}} =
\max\left(k_{J,\mathrm{floor}},\;
\texttt{zero\_gauss\_j\_floor\_rel}\;J_{\max,\mathrm{eff}}\right),
\quad
J_{\max,\mathrm{eff}}=\max(J_{\max},k_{J,\mathrm{floor}})
\]
を用い、いずれかの Gauss 点で \(J \le J_{\mathrm{floor}}\) ならば
backtracking admissibility failure として trial mesh を reject する。
これにより \(J=0\) および相対的に near-zero の正値 \(J\) を持つ退化
trial mesh を受理しない。既定値は `reject_zero_gauss_j=false`,
`zero_gauss_j_floor_rel=1e-8`。

品質指標：
\[
q_c = \frac{J_{\min}}{J_{\max}}
\]
- \(q_c = 1\)：完全な長方形（または平行四辺形）
- \(q_c \to 0\)：退化セル（凹形状、ねじれ）

**rezoneトリガー**：
\[
\min_c(q_c) < q_{threshold}
\]
既定：\(q_{threshold} = 0.2\)（`rezoning.quality_threshold`）。

#### 3.3.3 Winslow equipotential rezoning

楕円型スムージング方程式により、論理座標 \((\xi, \eta)\) から物理座標 \((r, z)\) への写像を平滑化する：
\[
\frac{\partial}{\partial\xi}\left(\alpha\frac{\partial r}{\partial\xi}\right)
+ \frac{\partial}{\partial\eta}\left(\beta\frac{\partial r}{\partial\eta}\right) = 0
\]
\(z\) についても同形の方程式を解く。

計量係数：
\[
\alpha = \left(\frac{\partial r}{\partial\eta}\right)^2 + \left(\frac{\partial z}{\partial\eta}\right)^2,\quad
\beta = \left(\frac{\partial r}{\partial\xi}\right)^2 + \left(\frac{\partial z}{\partial\xi}\right)^2
\]

**メトリック偏微分の離散化**：中心差分を使用する。
\[
\frac{\partial r}{\partial\xi}\bigg|_{i,j} = \frac{r_{i+1,j} - r_{i-1,j}}{2}, \quad
\frac{\partial r}{\partial\eta}\bigg|_{i,j} = \frac{r_{i,j+1} - r_{i,j-1}}{2}
\]
\(\partial z/\partial\xi\), \(\partial z/\partial\eta\) も同様。
Jacobi 反復では前回反復 \(k\) の位置 \((r^{(k)}, z^{(k)})\) を用いてメトリック係数 \(\alpha, \beta\) を計算し、
新位置 \((r^{(k+1)}, z^{(k+1)})\) を得る（Gauss-Seidel ではなく Jacobi）。

> **交差微分項 \(\gamma\) の省略**：完全な Winslow 方程式には交差微分項
> \(\gamma = (\partial r/\partial\xi)(\partial r/\partial\eta) + (\partial z/\partial\xi)(\partial z/\partial\eta)\) が含まれ、
> \(-2\gamma\,\partial^2 r/(\partial\xi\,\partial\eta)\) 等の寄与がある。
> v1.0 では \(\gamma = 0\) とする **簡略化直交 Winslow** を採用する。
> この省略はメッシュの非直交性が中程度以下の場合（ICF爆縮の典型的な変形量）で許容される。
> メッシュ品質は §3.3.2 のヤコビアン正値性検査で保証される。
>
> **非線形性**：\(\alpha, \beta\) はノード位置に依存するため、方程式は非線形である。
> 各 Jacobi 反復で \(\alpha, \beta\) を現在の位置 \((r^{(k)}, z^{(k)})\) から再計算する
> （lagged evaluation）。これはピカール反復に相当し、十分な反復回数（§3.3.3の反復制御参照）で収束する。
>
> **単位論理グリッド間隔**：離散化では \(\Delta\xi = \Delta\eta = 1\) を仮定する
> （構造格子の論理インデックスが整数であるため）。

**離散化**（Jacobi反復）：
\[
r_{i,j}^{new} = \frac{\alpha\,(r_{i+1,j}+r_{i-1,j}) + \beta\,(r_{i,j+1}+r_{i,j-1})}{2(\alpha+\beta)}
\]
\(z\) も同様。

**Opt-in full R-Z metric Winslow solver (2026-05-11):**
`Numerics.ale.rezone_solver="legacy_winslow"` is the default and preserves the
legacy Cartesian/orthogonal update above bit-exactly.  The opt-in
`"rz_full_metric_winslow"` path uses the full inverse Winslow metric form for
\(\mathbf{x}=(r,z)\),
\[
\alpha \mathbf{x}_{\xi\xi} - 2\beta \mathbf{x}_{\xi\eta}
  + \gamma \mathbf{x}_{\eta\eta}=0,\qquad
\alpha=\mathbf{x}_\eta\cdot\mathbf{x}_\eta,\quad
\beta=\mathbf{x}_\xi\cdot\mathbf{x}_\eta,\quad
\gamma=\mathbf{x}_\xi\cdot\mathbf{x}_\xi .
\]
This is the metric-tensor form used by elliptic mesh generation and aligned
with Knupp (SIAM, 2001) and the TMOP lineage of Dobrev-Kolev-Mittal (2019).
With unit logical spacing,
\[
\mathbf{x}_{\xi\eta}
  =\frac{\mathbf{x}_{NE}-\mathbf{x}_{NW}
          -\mathbf{x}_{SE}+\mathbf{x}_{SW}}{4},
\]
so the mixed contribution in the Jacobi numerator is
\[
-2\beta\mathbf{x}_{\xi\eta}
=-\frac{\beta}{2}
(\mathbf{x}_{NE}-\mathbf{x}_{NW}-\mathbf{x}_{SE}+\mathbf{x}_{SW}).
\]

The full-metric candidate is blended 50/50 between the Cartesian full-metric
candidate and a conservative R-Z weighted candidate:
\[
\mathbf{x}^{cart} =
\frac{\alpha(\mathbf{x}_E+\mathbf{x}_W)+
      \gamma(\mathbf{x}_N+\mathbf{x}_S)-2\beta\mathbf{x}_{\xi\eta}}
     {2(\alpha+\gamma)},
\]
\[
\mathbf{x}^{rz} =
\frac{w_E\alpha\mathbf{x}_E+w_W\alpha\mathbf{x}_W+
      w_N\gamma\mathbf{x}_N+w_S\gamma\mathbf{x}_S
      -2r_C\beta\mathbf{x}_{\xi\eta}}
     {\alpha(w_E+w_W)+\gamma(w_N+w_S)},
\]
where \(w_E=\max((r_C+r_E)/2,\epsilon)\),
\(w_W=\max((r_C+r_W)/2,\epsilon)\), and \(w_N=w_S=r_C\).
The accepted candidate before displacement limiting is
\(\mathbf{x}^{cand}=0.5(\mathbf{x}^{cart}+\mathbf{x}^{rz})\).

At the axis the solver enforces \(r_{0,j}=0\).  When axis-Z Winslow motion is
enabled, symmetric derivatives use the ghost mirror
\(r_{-1,j}=-r_{1,j}\), \(z_{-1,j}=z_{1,j}\).  Unlike the legacy kernel, the
full-metric path does not freeze an interior node solely because the local raw
stencil \(J\le kJFloor\); it falls back to the same R-Z weighted Laplacian
average and then applies the normal displacement cap.  If
`Numerics.ale.rezone_local_admissibility_linesearch=true`, each movable node
then tries \(\lambda\in\{1,1/2,\ldots,2^{-N}\}\), where
\(N=\texttt{rezone\_local\_linesearch\_max\_halves}\), and accepts the first
trial whose up-to-four incident cells satisfy:
4-Gauss \(J>\max(\texttt{rezone\_local\_j\_floor\_rel}J_{max,eff},kJFloor)\),
positive corner-J, positive signed R-Z quad volume, and nonnegative non-axis
radii.  Only if all local trials reject does that node keep its old position.

**Corner-cell aspect protection (2026-05-16):**  The R-Z Winslow kernels apply
a default-on geometric constraint at annular state-supply z-face corners:
`Numerics.ale.corner_cell_aspect_protection_enabled=true`,
`Numerics.ale.corner_cell_aspect_eta=0.5`.  The predicate is true only when
all of the following hold:

- `has_physical_rz_axis=false` (annular RZ; physical-axis cases are inert).
- The movable node is the interior node of a radial/z corner cell:
  \(i\in\{1,n_r-1\}\) and \(j\in\{1,n_z-1\}\).
- The adjacent z face is `state_supply`: \(j=1\) requires z-bottom
  state-supply, and \(j=n_z-1\) requires z-top state-supply.

For the protected corner cell \(c\), planar cell area is measured by the
shoelace formula in cm\(^2\),
\[
A_c(\mathbf{x}) =
\frac{1}{2}\sum_{k=0}^{3}(r_k z_{k+1}-z_k r_{k+1}),
\qquad k+1\equiv 0 \pmod 4 .
\]
Let \(A_c^{init}=|A_c(\mathbf{x}^{init})|\), \(A_c^n=A_c(\mathbf{x}^n)\),
and \(A_c^{cand}=A_c(\mathbf{x}^{cand})\), where only the protected interior
node is replaced by the candidate position.  The floor is
\[
A_c^{floor}=\eta_{corner} A_c^{init},\qquad
\eta_{corner}=\texttt{Numerics.ale.corner\_cell\_aspect\_eta}.
\]
If \(A_c^{cand}<A_c^{floor}\le A_c^n\), the node displacement
\(\Delta\mathbf{x}=\mathbf{x}^{cand}-\mathbf{x}^n\) is first scaled by
\[
\lambda_A =
\frac{A_c^n-A_c^{floor}}{A_c^n-A_c^{cand}},
\qquad
\mathbf{x}^{new}=\mathbf{x}^n+\lambda_A\Delta\mathbf{x},
\]
with \(\lambda_A\) clamped to \([0,1]\).  If the scaled candidate is still
below the floor, the protected node is moved by the minimum correction along
the signed-area gradient of that cell so that \(A_c^{new}=A_c^{floor}\).  Thus
the guard also restores a cell that the preceding physical mesh move has
already brought below the floor before Winslow relaxation.  If multiple
protected corner cells share the same movable node on a degenerate small grid,
the constraints are applied sequentially.  On a scheduled ALE step, a protected
corner already below the floor is treated as a Winslow rezone-quality trigger so
that the in-kernel constraint can restore it immediately.  This is a continuous
Winslow constraint, not a repair or retry escape valve, and it does not change
CFL math or the cgs/eV unit system.

**反復制御**：
- 最大反復数：`rezoning.max_iterations`（既定 20）
- 収束判定（残差ノルム）：
\[
\delta_{rezone} = \max_n \sqrt{(r_n^{(k+1)}-r_n^{(k)})^2 + (z_n^{(k+1)}-z_n^{(k)})^2} < \varepsilon_{rezone}
\]
既定 \(\varepsilon_{rezone} = 10^{-6}\,\Delta l_{min}\)（メッシュ最小セル長の \(10^{-6}\) 倍）。
- 収束前に最大反復数に達した場合は最終反復の結果を使用し、警告を出力する
- 典型的なICF問題では5–15回で収束する

**境界条件**：
- 物理境界（外周）：ノード位置固定
- r=0軸：\(r=0\) を強制維持
- 物質界面ノード：v1.0では特別扱いしない（材料界面追跡は非スコープ）

**Multiblock CSR cross-seam Winslow (S4-T1-next T5b):**
For `topology_scheme="multiblock_cart_core_polar_shell"`, the CSR Winslow
path is opt-in via `Numerics.ale.multiblock_cross_seam_rezone_enabled=true`.
The default `false` preserves the S4 seam-GCL gates by running only the T5a
per-block smoother before CSR conservative remap. With the flag enabled, the
driver dispatches the CSR cross-seam smoother instead of the structured
`node_index(i,j,nz)` Winslow path.

A node is seam-shared when at least one incident cell has a face tag
`SEAM_CORE_BRIDGE` or `SEAM_BRIDGE_SHELL`.  For non-seam interior nodes, the
T5a per-block CSR update is unchanged.  For a movable seam-shared node \(n\),
the neighbor set is the unique set of face-adjacent corner nodes from all CSR
cells incident on \(n\), including cells from both blocks:
\[
\mathcal{N}_{seam}(n)=
\{m:\ m \text{ is an edge-adjacent corner of an incident cell containing } n\}.
\]
The Jacobi update is
\[
\mathbf{x}_n^{new}=(1-\omega)\mathbf{x}_n^{old}
 +\omega\,\frac{1}{|\mathcal{N}_{seam}(n)|}
  \sum_{m\in\mathcal{N}_{seam}(n)}\mathbf{x}_m^{old}.
\]
When reference coordinates are present, the same formula is applied to the
reference displacement \(\Delta\mathbf{x}=\mathbf{x}-\mathbf{x}^{ref}\) and the
candidate physical coordinates are reconstructed as
\(\mathbf{x}^{ref}+\Delta\mathbf{x}^{new}\).  Nodes with pure axis,
outer-physical-boundary, or center flags remain fixed; multiblock
`NODE_POLE_AXIS` nodes that are not center/outer keep \(R=0\) exactly and
receive a monotone tangential \(Z\)-only on-axis-neighbor Jacobi update with a
local reference-spacing floor.  For other movable interior nodes, the
neighbor-mean is a proposal: with all non-owned nodes held at the Jacobi
old state, each incident CSR quad is evaluated using oriented signed area and
the inverse Knupp corner condition quality
\[
q_K = \frac{2J_s}{\lVert e_+\rVert^2+\lVert e_-\rVert^2},
\]
where \(J_s\) is oriented by the current-iteration corner Jacobian.  The local
proposal must satisfy \(A_s^{cand}>10^{-8}|A^k|\) and
\(J_s^{cand}>10^{-8}|J^k|\) for every incident cell/corner; these are purely
relative cgs-safe floors with no absolute cm scale.  If the full
\(\omega\)-proposal violates a floor, a deterministic 24-step bisection
line-search accepts the largest admissible fraction.  For risky proposals
(near \(q_K<0.2\) or reducing the minimum incident signed area/corner-J), the
kernel also tests a signed-corner-J ascent direction for the worst affected
incident corner and a blend with the neighbor-mean proposal; the accepted
Jacobi write maximizes minimum \(q_K\), then the minimum corner-J ratio, then
the accepted line-search fraction.  After every Jacobi sweep, the full CSR
candidate mesh is checked by the S4-T1 candidate
admissibility floors (RZ volume, corner-J, and Gauss-J); if the candidate is
not admissible, \(\omega\) is halved and the sweep retried.  Seam agreement
requires no extra interpolation because both blocks reference the same flat
node index for a shared seam node.

**Env-gated shell protected rezone primitive (I1-B Stage B, default-off):**
`TENRYU_I1B_SHELL_REZONE_REPLAY=1` enables a one-shot replay probe, not a
production subcycle.  The primitive `shell_protected_rezone` is seeded from the
MeshForecast `seed_mask` for low-\(q\) `POLAR_SHELL` cells, grows
`TENRYU_I1B_SHELL_REZONE_PATCH_LAYERS` face-adjacency layers clamped to
\([2,4]\) (default 3), filters the patch back to active shell cells, and moves
only nodes whose complete active cell-star lies inside that shell patch.
Patch-boundary nodes remain fixed.

The frozen node set is
\[
P_{\rm frozen}=P_{\rm axis/center/pole/boundary}\cup P_{\rm core}
               \cup P_{\rm derefine},
\]
where \(P_{\rm core}\) is the central pseudo-core member/boundary node set and
\(P_{\rm derefine}\) is the polar angular de-refine inactive/boundary node set.
These nodes are removed from the active-node mask and are also passed to the
barrier optimizer as excluded nodes, so the pole/core/de-ref treatments already
solved by the surrounding algorithms are not displaced.

The target construction reuses the protected center-patch transaction shape:
masked CSR Winslow smoothing proposes active shell-interior motion in one-sweep
chunks and stops once the protected patch reaches
\[
q_{\rm stop}=\max(q_{\rm release},
                 \min(1, f_{\rm reserve}q_{\rm release})),
\]
where `TENRYU_I1B_SHELL_REZONE_Q_RESERVE` defaults to \(f_{\rm reserve}=1.25\).
The existing RZ/corner-\(J\) barrier optimizer projects the proposal into the
feasible incident-cell set, and the CSR global line search first finds the
largest \(\sigma\in[0,1]\) such that the affected path remains above
`Numerics.ale.path_admissibility_floor`.  If that endpoint exceeds
\(q_{\rm stop}\), a scalar bisection on the same displacement direction shrinks
\(\sigma\) to the first endpoint that reaches \(q_{\rm stop}\), avoiding a
maximal-smoothing target when a release-quality target is already viable.  The
endpoint is accepted only if the
MeshForecast patch metric satisfies
\[
\min_{c\in P_{\rm shell}} q_c \ge q_{\rm release},
\]
with \(q_{\rm release}=\max(0.10,2q_{\rm warn})\) by default and optional
diagnostic override `TENRYU_I1B_SHELL_REZONE_Q_RELEASE`.  The full rezone path
\(\mathbf{x}(s)=\mathbf{x}^{L}+s(\mathbf{x}^{R}-\mathbf{x}^{L})\),
\(s\in[0,1]\), is checked by the same multiblock path-admissibility predicate;
endpoint validity alone is insufficient.

If accepted, the primitive installs exactly one reference target and calls one
conservative `ale_remap_2d_rz` with `conservative_remap_target="reference"` and
fixed swept-volume sign.  This conservative replay requires the initialized
`State::corner_mass` basis; otherwise it aborts the transaction with
`missing_corner_mass_basis`.  The shell replay call additionally passes a local
default-off override that forces the CSR total-energy branch and the Option-B
corner/nodal momentum branch for this remap, forces the coherent Option-B basis
transport and \(P/M\) re-recovery, and permits that pair under the already-active
polar-shell angular de-refine mask.  It also passes the shell patch as a closed
support-closed transaction rather than one shared mask:
\[
M\rightarrow G=\mathrm{star}(M)\rightarrow A=\mathrm{verts}(G)
\rightarrow C_E=\mathrm{star}(A).
\]
Here \(M\) is the moved-node set, \(G\) is the scalar CSR hydro flux support,
\(A\) is the affected node set whose velocities are recovered, and \(C_E\) is
the total-energy recovery closure.  Option-B corner mass/momentum transport
uses a separate dual-flux support \(C_\pi\): start from \(G\) and iteratively
add any non-inactive cell in \(C_E\) that shares a nonzero swept internal dual
face with the current set.  The scalar primal remap remains masked to \(G\);
the corner transport receives on \(C_\pi\).  In this support-closed replay
branch, each unique moving internal face first materializes one limited mass
flux \(F_g^\mu\) and one momentum flux
\(F_g^\pi=F_g^\mu\hat{\mathbf u}_g\) into face-flux arrays.  A second kernel
gathers those arrays into each cell in fixed face-index order and applies the
same \(F_g^\pi\), with opposite sign, to the two adjacent corner ledgers.  No
cell independently recomputes the nominally opposite face flux.  Replay-only
target-mass reconciliation may rescale corner masses to the scalar target but
does not rescale the already gathered momentum, and the replay does not apply
the affine hourglass post-filter after this exact gather.  The
original \(G\) mask remains the target-cell-mass mask for scalar-remapped
cells; collar cells in \(C_\pi\setminus G\) keep the conservative Option-B
corner-mass sum as authoritative cell mass.  Cells outside \(G\) remain
inactive for scalar remap packets, so scalar faces crossing the support boundary
still have zero swept-volume exchange.  For nodes in
\(\mathrm{verts}(C_\pi)\), the replay assembles \(M_i\) and \(P_i\) over their
complete cell star:
transported corners from \(C_\pi\) plus unchanged collar corner contributions
from \(C_E\setminus C_\pi\).  The node velocity recovery is then
\(u_i^R=P_i^R/M_i^R\) on every node in \(\mathrm{verts}(C_\pi)\), including
shared geometrically fixed patch-boundary nodes, except that the shell replay
keeps the pre-replay velocity when \(M_i^R\) is below the near-massless floor
\(\max(10^{-300}\,\mathrm{g},\)
`TENRYU_I1B_SHELL_REZONE_MASSLESS_NODE_FLOOR_G`) to avoid an undefined
\(P/M\) ratio at fully ablated surface nodes.  This clamp is conservation
neutral at the floor because the node carries vanishing mass, momentum, and
kinetic energy.  Node positions outside \(M\) remain fixed; only node
velocities in \(\mathrm{verts}(C_\pi)\) are allowed to change; macro-boundary
repair, axis-trace repair, velocity copyback, and the post-copy boundary
constraints are masked/restored so nodes outside \(\mathrm{verts}(C_\pi)\)
retain their pre-replay velocity.
The closure is rejected with
`closure_violates_frozen` if \(C_E\) intersects central pseudo-core-owned
cells.  If \(A\) intersects hard-frozen pole/core nodes or polar-shell angular
de-refine velocity-owned nodes, the primitive first applies a
closure-respecting shrink of the move set: every moved node \(m\in M\) whose
support star pulls a hard-frozen node into \(A\) is demoted to the fixed patch
frame, then \(G\), \(A\), and \(C_E\) are rebuilt.  This repeats until
\(A\cap H=\emptyset\).  If the move set falls below the small movable-node
threshold while \(A\cap H\ne\emptyset\), or if no responsible \(m\) can be
identified, the transaction is skipped with `closure_unshrinkable` rather than
forcing a non-closed remap.  The replay probe logs the initial/final \(M\)
sizes, shrink iterations, removed-node count, final hard-frozen closure count,
and de-refine wedge samples with shell-grid coordinates and de-refine subtype.

The internal energy is recovered from conserved total energy over the full
energy closure
\[
U_c^R=E_c^R-\frac{1}{2}\sum_{a\in c}
m_{c,a}^{\mathrm{OB},R}|\mathbf{u}_a^R|^2,
\]
using transported Option-B corner masses for cells in \(C_\pi\) and unchanged
`State::corner_mass` collar masses for cells in \(C_E\setminus C_\pi\).  Cells
in \(C_\pi\setminus G\) receive conservative dual corner-mass and
corner-momentum fluxes but no scalar primal total-energy flux; their total
energy snapshot is
\(E_c^R=E_c^0\); their internal energy absorbs only the kinetic-energy change
from shared-node velocity recovery.  If any cell in \(C_E\) would fall below
the internal-energy floor, the replay first transfers total energy
deterministically from surplus cells inside \(C_E\) and then recomputes
\(U=E-K\); unresolved deficits reject the replay transaction with
`closure_floor_violation` rather than accepting a hard floor.
For the replay diagnostic the remapped corner masses are installed into
`State::corner_mass` for \(C_\pi\), and `State::mass`/`State::rho` are updated
from the Option-B corner-mass sum on the same support.  Collar cells outside
\(C_\pi\) keep their pre-replay basis.
The replay log reports the raw corner-momentum transport residual \(R_\pi\),
the corner-to-node assembly residual \(R_{\rm assm}\), the recovery residual
\(R_{\rm rec}\), and the total-energy residual \(R_E\).  The replay probe
defaults to
`TENRYU_I1B_SHELL_REZONE_REPLAY_STAGE=post_corrector_commit`, operates on the
current committed valid mesh, logs \(q_{\min}\) before/after, restored-release
status, recomputed full-step `tau_zero`, \(G/A/C_E\) sizes, collar samples,
frozen-node motion, and total mass/momentum/energy deltas, then restores the
pre-probe state so the run trajectory is unchanged.  The live shell subcycle is
gated by the separate `TENRYU_I1B_SHELL_SUBCYCLE` env and does not change the
restore-only behavior of `TENRYU_I1B_SHELL_REZONE_REPLAY`.  With both envs
unset, this path is not entered and existing runs remain byte-identical.

**Gas-tracer core freeze for ALE rezone (I1-B S1, opt-in):**
`Numerics.ale.core_freeze_enabled=true` makes the ALE rezone target spatially
selective while making the frozen region fully Lagrangian during CSR remap.  The
implemented source is `core_freeze_source="gas_tracer"`.  At each ALE step the
driver rebuilds the frozen cell set from the current conserved gas tracer:
\[
F_0 = \{c:\;Y_g(c)\ge \texttt{core\_freeze\_tracer\_cut}\}.
\]
This seed set is dilated by face adjacency for
`core_freeze_halo_layers` layers, so the active/frozen interface can be placed
inside the shell rather than exactly at the gas/shell contact.  A node is
frozen if any incident cell is frozen; no partial weights are used.

The target produced by the rezone smoother is restored after the smoother and
before the target is committed for remap:
\[
\mathbf{x}^{target}_n \leftarrow \mathbf{x}^{Lag}_n,\qquad n\in F_N .
\]
When `core_freeze_skip_velocity_projection=true` (default), the same frozen-node
mask is also passed to the CSR cell-to-node velocity projection.  Frozen nodes
skip the projection write,
\[
\mathbf{v}^{n+1}_n \leftarrow \mathbf{v}^{Lag}_n,\qquad n\in F_N ,
\]
so the gas core keeps its post-hydro Lagrangian velocity instead of receiving
the ALE-projected cell average.  Setting
`core_freeze_skip_velocity_projection=false` restores the S1 behavior for A/B
diagnostics while retaining the mesh-target freeze.
The hook is applied to the multiblock cross-seam Winslow CSR path, the
reference-barrier ALE paths, and, when
`core_freeze_apply_to_axis_rezone=true`, the full-axis target-only rezone path.
When a multiblock Winslow target is modified, target cell volumes are
recomputed from the restored target before CSR remap.

For any frozen cell, every active face has both endpoints restored to the
post-hydro Lagrangian coordinates, so its ALE swept volume is zero to
roundoff.  The CSR remap therefore becomes the identity in the frozen cell, and
retaining the Lagrangian nodal velocity is the conservative velocity update.
Across a frozen/active boundary, the face of the frozen cell is also a
zero-sweep face, so mass, momentum, energy, and the conserved gas tracer do not
leak out of the pure-Lagrangian gas core during remap.  The mask is rebuilt
from the current \(Y_g\) each ALE step; cells that no longer satisfy the
tracer-plus-halo predicate are unfrozen naturally and rezone with the shell.

When enabled, the driver emits `[core_freeze]` diagnostics with the frozen
cell count, frozen node count, the node count that will skip CSR velocity
projection, maximum frozen-node displacement after restore, and the sum of
absolute active-face swept volumes over frozen cells.  The displacement and
swept-volume quantities should be roundoff-scale for a correctly frozen core.

**ALE Lagrangian-identity and mover diagnostics (I1-B PR1, default-off):**
`Numerics.ale.ale_identity_mode=true` is a 2D_RZ diagnostic identity limit.  It
does not change Hydro2D.  Instead, after the Lagrangian hydro update, every ALE
entry point returns before rezone, remap, cell-to-node velocity projection,
boundary reapplication, EOS reclosure, geometry rewrite, reference-state
updates, counters, and ALE history changes.  The resulting ALE-enabled run
should therefore be field-identical to `ALE=0` at matched timesteps unless
there is hidden step-boundary state outside the guarded ALE post-processing.

When `Numerics.ale.ale_mover_diag=true`, TENRYU emits JSONL
`[ale_identity_diag]` records at `pre_hydro` and `post_lagrange` with
deterministic hashes and \(L_\infty\) norms for mesh, velocity, thermodynamic,
viscosity, mass, corner-mass, diagnostic node-mass, and hydro-history fields.
It also emits `[ale_mover_diag]` records at `post_hydro`,
`ale_pre_projection`, and `ale_post_projection`.  The mover diagnostic uses one
post-hydro snapshot of node mass, old coordinates, and Lagrangian coordinates,
so the only intended change between pre- and post-projection records is the
stored velocity field:
\[
C_{\rm stored} = -\sum_i M_i\,\mathbf{x}^{n+1,L}_i\cdot
                 \mathbf{u}^{\rm stored}_i,
\qquad
C_{\rm move} = -\sum_i M_i\,\mathbf{x}^{n+1,L}_i\cdot
               \frac{\mathbf{x}^{n+1,L}_i-\mathbf{x}^{n}_i}{\Delta t}.
\]
The reported velocity mismatch is
\[
\lVert \Delta u\rVert_\infty =
\max_i\left\lVert \mathbf{u}^{\rm stored}_i -
\frac{\mathbf{x}^{n+1,L}_i-\mathbf{x}^{n}_i}{\Delta t}\right\rVert_2.
\]
`Numerics.ale.ale_preserve_lagrangian_velocity_carry=true` is a default-off
2D_RZ multiblock CSR diagnostic for the carried-velocity failure mode.  The CSR
ALE remap snapshots the post-hydro Lagrangian nodal velocity before the normal
rezone/remap/projection/total-energy closure sequence mutates `state.v`, then
restores that snapshot before the `ale_post_projection` mover record and before
the next hydro step can snapshot `state.v` into `u_old`.  Scalar remap, total
energy remap, internal-energy recovery, EOS reclosure, geometry update, and
corner-mass remap still run normally.  This intentionally makes the restored
velocity and recovered internal energy inconsistent, so the switch is a
diagnostic discriminator only and is not a production ALE fix.
Records are emitted for all nodes plus gas/shell incident-cell bands.  The gas
band is selected by post-hydro cell-centroid radius
\(\sqrt{r_c^2+z_c^2}<R_g\) when
`Numerics.diagnostics.hotspot_gas.R_g_cm>0`, with the current gas tracer used
only as a fallback when the radius is not configured.  These ALE diagnostic
switches are measurement-only and default-off.

**Gradient-alignment Stage 0 diagnostic monitor (default-off):**
`Numerics.ale.align_diagnostics.enabled=true` evaluates a read-only, host-side
single-snapshot monitor at the single-block ALE driver's post-Lagrange,
pre-rezone entry. It does not change node coordinates, rezone targets, remap
state, or any acceptance predicate. For \(\phi\in\{\rho,p_e+p_i\}\), fixed-order
host maxima define \(\phi_f=\texttt{floor_rel}\max_c\phi_c\), followed by
\[
q_\phi=\log\left(1+\frac{\phi}{\phi_f}\right).
\]
A deterministic available-neighbor one-ring WLS reconstruction solves
\[
B_c g_{\phi,c}=b_{\phi,c},\quad
B_c=\sum_d w_{cd}\Delta x_{cd}\Delta x_{cd}^{T},\quad
b_{\phi,c}=\sum_d w_{cd}\Delta x_{cd}(q_{\phi,d}-q_{\phi,c}),
\]
with \(w_{cd}=(|\Delta x_{cd}|^2+\epsilon_h)^{-1}\). At least three
neighbors are required and \(|\det B_c|<10^{-30}\) is unavailable. A
volFrac-mixed cell (any fraction strictly between 0.01 and 0.99) forms a
stencil barrier: no WLS or smoothing pair may have a mixed endpoint. PLIC
interface masks are not used in Stage 0.

With the deterministic general-quadrilateral meridional area \(A_c\),
\(h_c=\sqrt{A_c}\), the dimensionless strength and sign-independent tensor are
\[
s_\phi=h_c|g_\phi|,\qquad
S_\phi=\frac{s_\phi^2}{s_\phi^2+s_{\phi,0}^2},\qquad
Q=w_\rho S_\rho\widehat g_\rho\widehat g_\rho^T+
  w_p S_p\widehat g_p\widehat g_p^T,
\]
where \(\widehat g_\phi=g_\phi/(|g_\phi|+\epsilon_g)\). One face-neighbor pass
uses \(Q_c\leftarrow0.75Q_c+0.25\,\mathrm{mean}_{d}Q_d\). Polar-family tensors
are provisionally smoothed as global \((r,z)\) components. Stage 0 provisional
formula-local values are \(s_{\rho,0}=s_{p,0}=1\) and
\(\epsilon_g=\epsilon_Q=10^{-30}\); \(\epsilon_h=10^{-30}h_c^2\).
Temporal relaxation (§5.6 of the adopted design consultation) is intentionally
absent for this single-snapshot diagnostic.

For eigenvalues \(\lambda_1\ge\lambda_2\) and principal director
\(\widehat g\),
\[
c_Q=\frac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2+\epsilon_Q},\qquad
e_A=1-(\widehat n_i\cdot\widehat g)^2,\qquad
\theta_A=\arccos|\widehat n_i\cdot\widehat g|.
\]
Here \(\widehat n_i\) is the normalized average of the cell's two unit
\(i\)-face normals. Coherence is only a threshold filter
(\(c_Q\ge\texttt{c_q_threshold}\)); it does not weight either diagnostic.
The log reports deterministic nearest-rank angle p50/p90/max and 90 one-degree
histogram bins split into bulk, Chebyshev-distance-\(\le2\) near-interface,
and vacuum-mask classes, plus mask/unavailable counts. Vacuum-flagged,
interface, zero-topology-weight, unavailable-WLS, and invalid-director cells
are excluded.

For `polar_in_box`, the provisional topology weight is one for
\(i<\texttt{polar_prefix_nr}\),
\(1-(i-\texttt{polar_prefix_nr}+0.5)/\texttt{morph_rings}\) (clamped) in the
morph, and zero in the collar. Multiblock topologies emit one skip notice.
`every_n_steps=0` emits at the first post-Lagrange point and at the hook's
available final signal: \(t+\Delta t\ge t_{\rm end}\) or
\(\texttt{step}+1\ge\texttt{max_steps}\). Positive values use
`step % every_n_steps == 0`. Output is one `[ale-align-diag]` log line;
there is no run_info or HDF5 field.

**Gradient-alignment Stage 1--2 host prototype (not integrated):**
`hydro/ale_align_rezone.{hpp,cpp}` implements the alignment-only rectangular
prototype and is not called by the ALE driver. `RezoneParams::geometry` selects
`Planar` (the default, preserving the Stage-1 path bitwise) or `Rz`. The solver
consumes the frozen Stage-0 cell director and coherence; an unavailable
director has zero effective coherence. With \(c_c\in[0,1]\),
\[
\kappa_{\xi,c}=1+\texttt{kappa\_gain}\,c_c,\qquad
\kappa_{\eta,c}=1+0.25\,\texttt{kappa\_gain}\,c_c,
\]
where the provisional defaults are \(\texttt{kappa\_gain}=2\) and
\(\omega_\eta=0.35\). The prototype fixes \(m_n=m_t=1\) and evaluates the Q1
energy with symmetric \(2\times2\) Gauss quadrature,
\[
I_h=\frac12\sum_c\sum_q w_q W_q J_q
\left[g_{\xi,q}^{T}A_{\xi,c}g_{\xi,q}
+\omega_\eta g_{\eta,q}^{T}A_{\eta,c}g_{\eta,q}\right],
\]
where \(W_q=1\) in planar mode and \(W_q=2\pi r_q\) in RZ mode, with
\(r_q=\sum_aN_a(q)r_a\). Thus the central finite-difference gradient includes
the explicit RZ-radius variation without a separate analytic-gradient path.
\[
A_\xi=\kappa_\xi^{-1/2}P_n+\kappa_\xi^{1/2}P_t,\qquad
A_\eta=\kappa_\eta^{-1/2}P_t+\kappa_\eta^{1/2}P_n.
\]
The reference term, Huang--Russell quality term, concentration, masks, and
production integration remain absent.

The prototype-only nodal gradient is a deterministic central finite difference
of the adjacent-cell energy patch with
\(h_{fd}=10^{-6}\sqrt{\min_{c\ni a}A_c}\). The fixed-iteration update is
\[
d_a=-\alpha\frac{g_a}{L_a+\epsilon_L},\qquad
L_a=\sum_{c\ni a}
\frac{\gamma_c(1+\texttt{kappa\_gain}\,c_c)}{A_c},\qquad
\gamma_c=\begin{cases}1&\text{planar},\\2\pi r_{\mathrm{center},c}&\text{RZ},\end{cases}
\]
with defaults \(\alpha=0.3\), \(\epsilon_L=10^{-30}\), and 40 iterations.
All four boundary sides use tangential slip on their initial straight lines and
the four corners are fixed. In RZ mode, every node whose initial radius is
exactly zero remains at `r=0.0` while its non-corner axial coordinate may slide.
One global line-search factor follows the fixed ladder
\(1,1/2,\ldots,2^{-8}\); a trial is accepted only when total energy strictly
decreases and every exact Q1 corner Jacobian satisfies
\[
J_{c,k}^{candidate}\ge
\min\!\left(0.05\,A_c^{initial},J_{c,k}^{current}\right).
\]
Thus a corner at or above the initial relative floor remains at or above it,
while a corner already below the floor is required only not to degrade.
RZ trials additionally require the exact signed cell volume
\(V_c=\sum_qw_q2\pi r_qJ_q\) to be finite and positive. The final per-cell
values are exposed as `RezoneResult::cell_rz_volumes` (empty in planar mode).
A fully rejected ladder collects the geometrically violating cells at its
smallest factor in ascending cell order, zeros both displacement components at
their four nodes, and repeats the complete ladder once. A second rejected
ladder skips that outer iteration. `RezoneResult` reports accepted iterations,
lambda-halving attempts, frozen retries, and skips separately. Loop and
reduction orders are fixed.
A monitor with no available positive-coherence director returns the input node
arrays bitwise unchanged.

+**ALE velocity-coherence diagnostic (I1-B, default-off):**
`Numerics.diagnostics.ale_velcoherence.enabled=true` or
`TENRYU_I1B_DISC_ALE_VELCOHERENCE=1` enables measurement-only logging in the
multiblock CSR ALE step.  It samples the state at four ordered checkpoints:
after the Lagrangian hydro substep (`s0_post_hydro`), after Winslow/rezone target
construction (`s1_post_rezone`), after conservative mass/scalar/momentum/energy
remap and before cell-to-node velocity projection (`s2_post_remap`), and at the
accepted ALE step end after projection (`s3_post_velproj`).  No checkpoint
reorders, skips, or modifies physics updates.

The gas region is selected by current gas tracer \(Y_g>0.5\) when the tracer is
available and initialized; otherwise the fallback mask is current cell centroid
radius \(\sqrt{r_c^2+z_c^2}<R_g\), with
\(R_g=\texttt{Numerics.diagnostics.hotspot\_gas.R\_g\_cm}\).  For each cell,
the node radial velocity is
\[
u_{r,n} = \frac{v_{r,n} r_n + v_{z,n} z_n}{\sqrt{r_n^2+z_n^2}},
\]
with zero contribution at the origin.  The cell radial velocity and cell vector
velocity are active-node CSR averages:
\[
u_{r,c}=\frac{1}{N_c}\sum_{n\in c}u_{r,n},\qquad
\mathbf{u}_c=\frac{1}{N_c}\sum_{n\in c}(v_{r,n},v_{z,n}).
\]
The emitted line reports
\[
M_{\rm gas}=\sum_{c\in G}m_c,\quad
\langle u_r\rangle_M=\frac{\sum_{c\in G}m_cu_{r,c}}{M_{\rm gas}},
\]
\[
K_{\rm rad}=\sum_{c\in G}\frac12 m_c u_{r,c}^2,\qquad
K_{\rm tot}=\sum_{c\in G}\frac12 m_c|\mathbf{u}_c|^2.
\]
The log format is
`[ale_velcoherence] step=<n> cp=<checkpoint> M_gas=<...> mw_ur=<...>
rad_ke=<...> tot_ke=<...>`.  The diagnostic does not write HDF5 state and does
not feed back into mesh, remap, EOS, CFL, or boundary conditions.

#### 3.3.4 保存的Remap

rezone後の新メッシュへ物理量を保存的に転写する。

**方式**：方向分離（directional split）flux-based remap + Van Leerスロープリミッタ

**保存量**（remapで厳密に保存する量）：
- 質量 \(\rho V\)
- 運動量 \(\rho u_r V\)、\(\rho u_z V\)（セル中心量としてremap）
- エネルギー \(\rho e_i V\)、\(\rho e_e V\)
- 物質体積分率 \(f_\alpha V\)
- `Numerics.ale.ke_conservation_closure=true` の場合のみ、診断互換 kinetic energy
  density \(q_K=K_c^{diag}/V_c\)（§3.3 Phase 12）

**方向分離**：2Dのremapをr方向とz方向に分離して逐次実行する。
ステップごとにスイープ順序を交替する（Strang-type、\(O(\Delta t^2)\) 精度維持）：
- 偶数ステップ：r-sweep → z-sweep
- 奇数ステップ：z-sweep → r-sweep

**1方向のflux-based remap**（r方向の例）：

> **z方向への拡張**：以下のr方向formulationにおいて、r→z、z→rの置換を行い、面インデックスを垂直面→水平面に読み替える。スイープ体積公式・donorセル判定・リミッタ構造は同一。

保存量 \(q\)（\(=\rho, \rho u_r, \rho e_e,\ldots\)）のセル体積積分
\(\bar q_i = q_i V_i\) に対し、面 \(i+1/2\) を横切るフラックスを計算する：

1. **掃引体積（swept volume）**：
   rezone前の面位置 \(r_{i+1/2}^{old}\) からrezone後の面位置 \(r_{i+1/2}^{new}\) への変位により
   掃引される体積：
   \[
   \delta V_{i+1/2} = V(r_{i+1/2}^{new}) - V(r_{i+1/2}^{old})
   \]
   2D RZでは、面の old/new 端点が作るR-Z平面上の swept polygon \(P_f\) を
   z軸まわりに回転した厳密体積で計算する。

   **2D RZ でのスイープ体積公式**：
   \[
   \Delta V_f = V(P_f)
     = \frac{\pi}{3}\sum_k (r_k+r_{k+1})
       (r_k z_{k+1} - r_{k+1} z_k)
   \]
   \(P_f\) の頂点順は legacy raw polygon orientation として保持する。
   `Numerics.ale.swept_volume_sign_fixed=true` では swept-volume primitive at
   source で \(\Delta V_f^{fixed}=-\Delta V_f^{raw}\) を用い、保存
   フラックス、中間体積、MS2 swept moments、axis-band remap の全 path が
   同じ符号規約を消費する。旧 `Numerics.ale.donor_sign_fixed` は deprecated
   alias として受理される。直交格子で
   R-face が純粋に外向きへ \(\Delta r\) 移動する場合、
   \[
   \Delta V_f = \pi(r_{new}^2-r_{old}^2)\Delta z
              = 2\pi r_{old}\Delta r\Delta z + \pi\Delta r^2\Delta z
   \]
   であり、Z-face が純粋に \(+\Delta z\) 移動する場合は
   \(\Delta V_f = \pi(r_1^2-r_0^2)\Delta z\) となる。

2. **供給セル（donor cell）の決定**：
   既定の `Numerics.ale.swept_volume_sign_fixed=false` では legacy convention として
   raw \(\Delta V_f\) を使い、pre-fix donor/flux/intermediate-volume behavior
   を bit-exact に保持する。corrected convention では下記の
   post-2026-05-11 規約に従う。

**Swept-volume convention (post-2026-05-11 fix)**

`Numerics.ale.swept_volume_sign_fixed=true` enables the corrected swept-volume
primitive. For an R-face between low-index cell \(i\) and high-index cell
\(i+1\), corrected \(\Delta V_f > 0\) means low-to-high transfer; the donor is
cell \(i\). Corrected \(\Delta V_f < 0\) means high-to-low transfer; the donor
is cell \(i+1\). A face moving outward in \(+r\) grows cell \(i\) and shrinks
cell \(i+1\), so it has corrected \(\Delta V_f < 0\) and transfers
high-index material into the growing low-index cell.

The stored conservative face flux is
\[
F_f = \int_{P_f^{fixed}} q_d\,dV,
\]
so the split update keeps the single algebra
\[
\bar q_i^{new} = \bar q_i^{old} - F_{i+1/2} + F_{i-1/2}.
\]
Intermediate split volumes use the same corrected swept volume:
\[
V_i^{mid}=V_i^{old}-\Delta V_{i+1/2}^{fixed}
          +\Delta V_{i-1/2}^{fixed}.
\]
MS2 moment remap applies the same `FixedSign` choice to swept-volume moments,
and Stage 24 axis-band remap uses the same fixed source sign rather than a
separate negated polygon convention.

`swept_volume_sign_fixed=false` (default) preserves the legacy pre-2026-05-11
convention bit-exactly. This path is retained only as a gated legacy path for
migration and bit-exact regression testing. Production decks should migrate to
`swept_volume_sign_fixed=true` once Phase 3 empirical validation confirms
physically correct behavior.

3. **勾配推定とスロープリミッタ**：
   供給セル \(d\) における保存量密度の勾配を Van Leer リミッタで制限する：
   \[
   (\nabla q)_d^{lim} = \text{VanLeer}(q_{d-1}, q_d, q_{d+1})
   \]
   **Van Leerリミッタ**：
   \[
   \phi(r) = \frac{r + |r|}{1 + |r|},\quad
   r = \frac{q_d - q_{d-1}}{q_{d+1} - q_d}
   \]
   **\(r < 0\) の場合**（局所極値）：\(r < 0\) は隣接セル間の勾配が逆符号（\(q_d\) が局所極大または極小）を意味する。このとき \(r + |r| = 0\) であるから \(\phi(r) = 0\) となり、勾配はゼロに制限される（donor cell に退化、1次精度）。これは TVD 条件を満たすために必須であり、局所極値付近でのオーバーシュートを防止する。

   **分母ガード**：\(|q_{d+1} - q_d| < \varepsilon_{VL}\)（\(\varepsilon_{VL} = 10^{-30}\)）の場合は \(r = 0\) とする（donor cell に退化）。これにより一様場（\(q_{d-1} = q_d = q_{d+1}\)）で NaN を回避し、\(\phi(0) = 0\) → 1次精度 remap となる。

   **メッシュ境界での勾配制限**：境界セル（\(d=0\) または \(d=N-1\)）では隣接セルが存在しない方向について \(q_{d-1} = q_d\)（ゼロ勾配外挿）とする。これにより \(r = 0\) → \(\phi = 0\) となり、境界では自動的に donor cell（1次精度）remap に退化する。

   制限付き勾配：
   \[
   (\nabla q)_d^{lim} = \phi(r)\,\frac{q_{d+1} - q_d}{\Delta x_d}
   \]

4. **フラックス計算**：
   \[
   F_{i+1/2} = \left(q_d + (\nabla q)_d^{lim} \cdot d_{cf}\right) \delta V_{i+1/2}
   \]
   ここで \(d_{cf}\) はドナーセル中心からスイープ面までの **符号付き距離** [cm]
   （スイープ方向に正）。legacy convention では
   \(d_{cf} = \text{sgn}(\delta V_{i+1/2}) \cdot \Delta x_d / 2\)。
   corrected convention では上記 **Donor convention (post-2026-05-11 fix)**
   の donor/flux sign に合わせて評価する。
   第2項が2次精度補正であり、\((\nabla q)_d^{lim}\) [量/cm] × \(d_{cf}\) [cm] = [量]（無次元補正）、
   全体に \(\delta V\) [cm\(^3\)] を乗じて [量·cm\(^3\)] の次元で整合する。

5. **保存量更新**：
   \[
   \bar q_i^{new} = \bar q_i^{old} - F_{i+1/2} + F_{i-1/2}
   \]

**密度更新**：質量 \(\bar\rho_i = \rho_i V_i\) のremap後に
\(\rho_i^{new} = \bar\rho_i^{new} / V_i^{new}\) で密度を復元する。

**速度のremap**：速度はセル中心量として \(\rho u_r\)、\(\rho u_z\) をremapし、
\(u_r^{new} = (\rho u_r)^{new}/\rho^{new}\)、\(u_z^{new} = (\rho u_z)^{new}/\rho^{new}\) で復元する。

> **節点速度への変換**：Lagrangianスキーム（§3.2）は節点速度を使用するため、
> セル中心速度から節点速度への射影が必要である。
> 既定（`ke_conservation_closure=false`）では節点 v の速度を、隣接セルの質量重み平均で計算する：
> \[
> \mathbf{u}_v = \frac{\sum_{c \in \mathcal{N}(v)} m_c\,\mathbf{u}_c}{\sum_{c \in \mathcal{N}(v)} m_c}
> \]
>
> **運動量保存モニタリング**：この質量重み投影は厳密な運動量保存を保証しない。
> 投影後の運動量誤差 \(\delta p = \sum_c (\rho u V)_c^{remap} - \sum_v M_v u_v^{proj}\) を計算し、
> \(|\delta p|/|p_{total}|\) をモニタリングする（典型的には \(O(10^{-10})\)）。
> v1.0 では補正を行わず、誤差がエネルギー収支閾値内に収まることを検証する。
> `ke_conservation_closure=true` では、同じ境界条件適用の後段を使うが、節点への戻しは
> corner mass momentum gather に切り替える：
> \[
> \mathbf{u}_v =
> \frac{\sum_{c\in\mathcal{N}(v)} m_{cv}\mathbf{u}_c^*}
>      {\sum_{c\in\mathcal{N}(v)} m_{cv}} .
> \]
> ここで \(m_{cv}\) は §3.2.4 の \((n_{00},n_{10},n_{11},n_{01})\)
> corner mass であり、projection 前後の全セル運動量和を同じ corner-mass
> 分配で評価する。
>
> **速度境界 projection mode**：2D_RZ ALE projection は face ごとの mode を使う。
> mode 0 は free、mode 1 は reflect、mode 2 は fixed、mode 3 は state_supply である。
> mode 3 では state-supply z-boundary の \(v_z\) を reflect-style にゼロ化せず、
> supplied/restored material \(u_z\) を保存する。

**Default-off CSR Option B corner velocity remap (Stages 4b/4c):**
`TENRYU_OPTIONB_CSR_CORNER_VELOCITY_REMAP=1` enables only a component-test
entry point; the production ALE remap does not call it in Stage 4b.  The path
uses the CSR unsplit face list and the existing mass-positivity limiter.  For
each active cell, the lagged nodal velocity is gathered into corner degrees of
freedom with the Option B first-moment corner masses:
\[
m_{c,a}=M_c\,\lambda_a(\bar{\mathbf{x}}_c),\qquad
\mathbf{p}_{c,a}=m_{c,a} A_a\mathbf{u}_a ,
\]
where \(A_a\) is `FREE`, `RZ_AXIS`, or `PINNED_OR_NODE_CENTER` according to
`node_flags` (`NODE_CENTER` pins both components; `NODE_AXIS`/`NODE_POLE_AXIS`
zero only \(u_R\)).  Internal faces are greedily colored on the fixed
`unique_internal_faces` cell-adjacency graph so that no same-color packet
touches the same cell.  Colors are launched serially; packets inside one color
can run in parallel because their live corner states are disjoint.

For a colored internal face, the swept packet volume is the same
`csr_face_swept_volume_outward`/`csr_face_swept_moments_outward` value used by
the scalar CSR remap, multiplied by
`mass_flux_scale[losing_cell]`.  The positive packet mass is
\[
\Delta m_q=\max(\rho_d^L,0)\,|\Delta V_f\,s_{\ell(f)}|,
\]
with the donor selected by the same CSR donor convention as the scalar mass
remap.  The packet location \(\mathbf{x}_q^*\) is the RZ centroid of the swept
quadrilateral from `csr_face_swept_moments_outward`.  The committed Option B
FCT packet remap is then applied using donor vertices on \(X^L\), receiver
vertices on \(X^R\), donor lagged nodal velocities, and live donor/receiver
corner mass-momentum buffers.  If the receiver vertices are outside the donor
cell hull, or if \(\mathbf{x}_q^*\) is outside only by a small normal distance to
a donor/receiver edge, the production-gated CSR component builds a positive
expanded donor stencil from the donor cell plus face-neighbor rings (currently
one then two rings).  Near-edge packet centroids use the design's edge endpoint
weights, clamped to the nearest edge, for donor/receiver packet weights.  The
source velocity at each receiver vertex is linearly interpolated from old nodal
velocities on the first stencil whose convex hull contains the query point; the
same \(C_R\) packet correction and FCT synchronization are then applied to the
single \(\Delta m_q\) packet.  Diagnostics separately count expanded requests,
one-ring successes, two-ring successes, centroid-out cases split into near-edge
and far-outside, expanded failures, and true conservative first-order
fallbacks.  Invalid input or an unresolved expanded request still falls back to
the conservative first-order packet: it debits the donor in proportion to live
donor corner mass, credits the receiver by target first-moment corner weights,
and uses the donor live cell-mean velocity for both debit and credit.

After all internal colors, boundary faces receive a one-sided first-order
treatment consistent with the scalar signed boundary \(\Delta V_f\): negative
boundary volume removes live cell-mean momentum proportionally to corner mass,
while positive boundary volume adds mass with the same cell-mean velocity and
target first-moment corner weights.  This is light Stage 4b coverage; full
boundary smoke coverage is deferred to the production wiring stages.  The
component then adds any \(\rho_{floor}V^R\) mass deficit as zero-momentum
corner mass, applies the affine-orthogonal Option B hourglass filter per cell,
and scatters to separate nodal output buffers with reverse CSR:
\[
\tilde{\mathbf{u}}_i^R =
A_i\frac{\sum_{(c,a)\in\mathcal{N}^{-1}(i)}\mathbf{p}_{c,a}}
          {\sum_{(c,a)\in\mathcal{N}^{-1}(i)}m_{c,a}} .
\]
The stored velocity \(\mathbf{u}_i^R\) is then made componentwise
bound-preserving against the projected source velocities on the source
incident-cell node stencil of \(i\) (including \(i\) itself):
\[
u_{i,\alpha}^R =
\operatorname{clip}\left(
\tilde u_{i,\alpha}^R,\,
\min_{j\in S_i}(A_j\mathbf u_j^L)_\alpha,\,
\max_{j\in S_i}(A_j\mathbf u_j^L)_\alpha\right).
\]
This scatter limiter is inactive for zero scatter mass, preserves affine fields
whose scattered value is inside the source stencil bounds to roundoff, and
prevents the RZ-axis \(2\pi r\to0\) mass denominator from emitting velocities
outside the physically adjacent source envelope.  The nodal momentum scratch is
updated to \(\mathbf P_i=M_i\mathbf u_i^R\) after clipping so the Option B/T6
kinetic-energy recovery reads the same bounded velocity.

Stage 4c adds an independent production driver gate,
`TENRYU_I1B_OPTIONB_VELREMAP=1`, default false.  When it is unset, the CSR
production path branches around all Option B allocations and launches and keeps
the legacy cell-to-node projection byte-identical.  When it is set in the
multiblock CSR remap path, the scalar remap is still completed first: mass,
material energy or total energy, `corner_fraction`, tracers, the
`mass_flux_scale`, and the finish/floor step produce the authoritative
\(M_c^R\).  The scalar path's effective swept-volume convention becomes
`Numerics.ale.swept_volume_sign_fixed || total_energy_remap_2d_rz ||
TENRYU_I1B_OPTIONB_VELREMAP`, so the packet velocity remap and scalar mass
state share the same mass limiter.

The Option B component then runs while the state arrays still hold \(X^L\),
\(\rho^L\), \(M^L\), and the post-Lagrangian nodal velocity, and it receives the
finished scalar `d_mass_new` as its target cell mass.  Before the hourglass
filter and reverse-CSR scatter, each cell's Option B corner masses are
reconciled to \(\sum_a m_{c,a}=M_c^R\): excess mass is removed by scaling corner
mass and momentum together, while a deficit is added as zero-momentum mass using
the target-cell first-moment corner weights.  The scattered nodal velocity is
copied to `State::v_r/v_z`; `csr_project_cell_velocity_to_nodes*` is skipped.
The normal boundary/axis constraints are applied afterward.

When the virtual central pseudo-core is active, member-cell faces remain scalar
no-flux in this Exp1 path: the inactive virtual members are not independent
donors or receivers in the CSR scalar remap.  The all-node scatter bound above is
therefore the axis-safe velocity limiter at pseudo-core boundary poles as well:
the inactive-member cells can contribute source-stencil velocity bounds, but no
scalar mass is fluxed through their faces.  The later pseudo-core boundary
velocity repair remains mask-gated and accepts an active-side expanded-stencil
reconstruction only if it stays inside the physically adjacent velocity envelope.

With `total_energy_remap_2d_rz=true`, the ordering is:
\[
\text{scalar total-energy remap}
\rightarrow \text{Option B corner velocity scatter}
\rightarrow \text{boundary constraints}
\rightarrow \text{KE realizability scale}
\rightarrow e = E - K .
\]
The KE scale and internal-energy recovery therefore read the Option B nodal
velocity (after boundary constraints), not the legacy cell-projected velocity.
Under `TENRYU_I1B_OPTIONB_VELREMAP=1`, the conserved scalar total energy is also
built with the same Option B first-moment corner masses used by the corner
momentum gather,
\[
K_c=\frac{1}{2}\sum_{a\in c} m_{c,a}^{\mathrm{OB}}|\mathbf{u}_a|^2 ,
\]
and recovery uses the post-remap Option B corner masses after scatter,
\[
U_c^R=E_c^R-\frac{1}{2}\sum_{a\in c}
m_{c,a}^{\mathrm{OB},R}|\mathbf{u}_a^R|^2 .
\]
The first-moment corner-mass basis is **internal to the remap/energy pair**: the
total-energy build computes it on the fly from \(X^L\) and the recovery uses the
transported Option B corner masses, so the pair conserves to roundoff without
touching `State::corner_mass`.  The hydro's subzonal corner masses
(`State::corner_mass`, the CBSW force/AV/subzonal-pressure basis) remain on
their Lagrangian-invariant lifecycle and are carried across remaps by the
corner-fraction transport.  Overwriting `State::corner_mass` with
geometry-recomputed first-moment masses (the original T6 "canonicalization" and
the post-remap install) is retained only behind the opt-in
`TENRYU_I1B_OPTIONB_CANONICALIZE=1` for A/B comparison: it violates subzonal
Lagrangian invariance and the subzonal-pressure \((m,V)\) equilibrium, injecting
spurious near-axis forces (largest where the first-moment and CBSW bases differ
most, \(r\to0\)).  With the velocity-remap flag unset, the legacy total-energy
basis and diagnostic path are unchanged.

The Stage-B shell replay is the narrow exception: its local remap override
forces the total-energy/coherent-Option-B pair, transports from the initialized
`State::corner_mass` basis, and now uses a support-closed replay transaction.
Scalar remap transport is restricted to \(G=\mathrm{star}(M)\), velocity
recovery uses the full-star \(P_i/M_i\) assembly on all affected nodes
\(A=\mathrm{verts}(G)\), retaining the old velocity for near-massless
surface nodes below the shell replay mass floor, and total-energy recovery extends to
\(C_E=\mathrm{star}(A)\).  Option-B corner mass/momentum transport receives on
the moving-face closure \(C_\pi\subseteq C_E\).  The replay computes each
unique moving internal face flux once, stores
\((F_g^\mu,F_g^\pi)\), and gathers the stored fluxes in fixed face-index order
with opposite signs on the two adjacent corner ledgers, so collar cells in
\(C_\pi\setminus G\) receive the opposite dual scatter that telescopes
\(\Sigma\pi\).  Replay-only target-mass reconciliation preserves the gathered
momentum.  Cells in \(C_E\setminus C_\pi\) keep their old corner-mass basis, and
nodes outside \(\mathrm{verts}(C_\pi)\) keep their pre-replay velocity after
recovery, repair, copyback, and boundary constraints.  Collar total energy
remains the pre-replay snapshot and internal energy is reconstructed as
\(U=E^0-K^R\) after shared-node velocity recovery.  Remapped corner masses and
matching cell mass/rho are installed for \(C_\pi\) during the replay
transaction, after which the probe restores the pre-probe state.  The
support-closed path is not production step-loop wiring.

**RZ axis regularity trace (Option B scatter, on-axis nodes).**  The physical RZ
nodal mass vanishes on the axis (\(m\sim2\pi r\)), so the scatter ratio
\(\sum\mathbf p/\sum m\) is singular there and amplifies any unpaired flux
residual.  For nodes with the `RZ_AXIS` projector the scattered velocity is
therefore replaced by the \(r\to0\) regularity trace of the adjacent off-axis
flow (Kenamond/Barlow-Shashkov axis treatment): a multiplicity-weighted affine
least-squares fit \(u_z(r,z)=a+br+c\,(z-z_n)\) over the off-axis `FREE` nodes of
the incident active cells and their face-adjacent active cells (normalized
\(3\times3\) Cramer solve; mean fallback on degenerate stencils; donors require
positive scatter mass), evaluated at \((0,z_n)\) so \(u_z^{axis}=a\) and
\(u_r^{axis}=0\).  Affine velocity fields are reproduced exactly.  The nodal
momentum scratch is updated to \(M_i u_i\) after the trace.  Opt-out:
`TENRYU_I1B_OPTIONB_AXIS_TRACE_DISABLE=1`.

**RZ幾何因子**：全ての体積計算に \(2\pi R\) 因子を含める（§3.2.3参照）。

**Multiblock differential converging reference (default OFF)**：
For the 5-block γ-MVP butterfly multiblock, `Numerics.ale.multiblock_differential_reference_enabled`
selects an opt-in reference target for conservative remap. The production default remains the
legacy γ-MVP reference path. When the differential path is enabled and the mesh topology is
multiblock 2D_RZ, it replaces the γ-MVP scaled-reference install point for that remap call;
otherwise the legacy/static or `multiblock_scaled_reference_enabled` path is used.

The γ-MVP scaled reference is initial-close:
\[
\mathbf{x}^{ref}_n = \alpha(t)\,\mathbf{x}^{0}_n,\qquad
V^{ref}_c=\alpha(t)^3 V^0_c ,
\]
with \(\alpha(t)\) taken from the current outer radius. This keeps the target homothetic to
the initial mesh and can hold the multiblock mesh open under a free outer surface. The
differential reference is Lagrangian-close: it is a correction to the post-hydro Lagrangian
mesh, not a replacement by a globally scaled IC mesh,
\[
\mathbf{x}^{ref}_n
= \mathbf{x}^{L}_n + \sigma\,\chi_n\,d(\xi_n)\,\hat{\mathbf{e}}_n,\qquad
d(\xi_i)=S_i-s^{raw}_i .
\]
Here \(\mathbf{x}^{L}\) is the current Lagrangian node coordinate, \(\hat{\mathbf{e}}_n\) is
the fixed initial director, \(\chi_n\in[0,1]\) is the per-node displacement cap, and
\(\sigma\in[0,1]\) is the accepted global admissibility scale.

The generalized layer coordinate is fixed at initialization from the seam-conforming initial
radius:
\[
s^0_n=\sqrt{(r^0_n)^2+(z^0_n)^2},\qquad
\xi_n=s^0_n/s^0_{\max}.
\]
The coordinate is monotone in \(s^0\), single-valued across seam-equivalent nodes within
`multiblock_differential_reference_xi_seam_tol`, and uses the initial director
\(\hat{\mathbf{e}}_n=(r^0_n,z^0_n)/s^0_n\) away from the center/axis degeneracy.

For each ξ band \(i\), the raw Lagrangian projected radius is the robust median of
\[
p_n=\mathbf{x}^{L}_n\cdot\hat{\mathbf{e}}_n
\]
over nodes in that band, with missing bands filled from neighboring finite bands. This gives
\(s^{raw}_i\). The corrected band scale \(S_i\) is the minimal accepted correction produced by:
1. checking raw monotone admissibility against a center floor
   `multiblock_differential_reference_s_cap_min_rel * s^0_max` and per-edge floor
   `multiblock_differential_reference_eps_v * h_i`;
2. if needed, applying eight shock-aware smoothing passes to \(s^{raw}_i\), with edge weight
   \(w_i=1/(1+((\alpha_{i+1}-\alpha_i)/g_0)^2)\) and
   \(g_0=\) `multiblock_differential_reference_smoothing_g0`, where
   \(\alpha_i=s^{raw}_i/s^0_i\) with the implementation's tiny-denominator guard;
3. limiting each band correction by \(\nu h_i\), where
   \(\nu=\) `multiblock_differential_reference_nu`;
4. projecting back to the same monotone/floor constraints.

The installed reference is admitted through three layers. First, the band sequence must satisfy
the monotone/floor constraints above. Second, each node displacement is capped by
\(\chi_n=\min(1,\nu h_n/|\Delta\mathbf{x}_n|)\). Third, an orientation-aware CSR line search
chooses the largest \(\sigma\) accepted by the candidate-mesh quality predicate on positive RZ
cell volume, signed corner-J, and 2x2 Gauss-J using the existing `reference_*_floor_rel` and
`reference_linesearch_max_iters` settings. The RZ volume used by this predicate and by remap
reference storage is the exact rotation volume
\[
V=\int 2\pi r\,dr\,dz ,
\]
with each cell's orientation sign applied. After \(\sigma\) is accepted, \(V^{ref}_c\) is
recomputed from the accepted reference coordinates and copied to `State.cell_vol_initial`.
It is not \(\alpha^3 V^0_c\), which preserves GCL consistency for the accepted target.

MVP limitations are explicit. The ξ coordinate is radial; harmonic or graph-Laplacian ξ fields
are out of scope. Equal-ξ bands can be multimodal across blocks, so the median is robust but
does not identify separate modal branches. At high compression ratio, the CSR line search can
collapse \(\sigma\) toward zero; diagnostics surface this condition rather than smoothing it
away silently. Empirical Case-B compression and σ-health characterization is a verification
result, not part of this scheme definition.

**Multiblock Lagrangian-bulk center/quality-patch reference (default OFF):**
`Numerics.ale.multiblock_lagrangian_bulk_center_patch_reference_enabled`
selects a third, mutually exclusive multiblock conservative-remap reference
mode. It is valid only for multiblock `2D_RZ` conservative remap with
`conservative_remap_target="reference"`. While disabled it changes no mesh
coordinates, swept volumes, or hydro fields.
When enabled, the center-patch driver forces the protected CSR swept-volume
convention for its remap handoff (`swept_volume_sign_fixed=true` in the copied
remap config), so the losing-cell outgoing-mass scale in the CSR remap is active
without requiring the deck to set the global compatibility knob.

The intended reference is Lagrangian in the bulk:
\[
\mathbf{x}^{ref}_n=\mathbf{x}^{L}_n
\]
for every node outside a small center/quality patch. Consequently the
bulk-to-reference swept volume is zero outside the patch, preserving
pure-Lagrangian compression there. The patch is the union of a permanent center
region and quality-triggered cells, dilated by
`multiblock_center_patch_halo_layers`. The permanent region includes cells
within `multiblock_center_patch_ring_max` / the tri-fan cap and, when
`multiblock_center_patch_xi_center > 0`, cells whose reference ξ cell value is
below that cutoff. Quality hysteresis uses dimensionless on/off thresholds for
\(V_c/V_c^0\), signed corner-J, and 2x2 Gauss-J:
`multiblock_center_patch_vol_on/off`,
`multiblock_center_patch_cornerj_on/off`, and
`multiblock_center_patch_gaussj_on/off`, each with `on < off` and values in
\((0,1)\).

When `TENRYU_I1B_DIFFREF_DIAG` is enabled, CSR remap also reports the
diagnostic-only contact swept-volume ratio
\(\eta_{\rm contact}^{step}=\sum_{f:(Y_{g,a}>0.5)\oplus(Y_{g,b}>0.5)}
|\Delta V_f^{lim}|/\sum_c Y_{g,c}V_c^L\). The numerator uses the same signed
RZ swept volume and mass-flux limiter as the hydro remap, so the diagnostic has
units cm^3/cm^3 and does not change cgs/eV state updates.

At runtime the driver saves the post-hydro Lagrangian coordinates
\(\mathbf{x}^{L}\), uploads the active-node and patch-boundary masks, and
applies the masked CSR Winslow operator as a seed coordinate smoother to the
current coordinates, not as a reference-displacement smoother:
\[
\mathbf{x}^{W} = W_{\mathrm{mask}}(\mathbf{x}^{L}),\qquad
\mathbf{x}^{W}_n=\mathbf{x}^{L}_n\ \mathrm{for}\ M_n=0 .
\]
The seed is screened by the ALE motion trigger before the expensive local
barrier is used. The center-patch branch enables only the corner-J trigger for
this decision: if the seed patch minimum corner-J ratio is below
`multiblock_center_patch_cornerj_off` (default 0.08), the seed is relaxed by the
local center-patch Phi-barrier optimizer on patch-interior active nodes:
\[
\mathbf{x}^{B}=\mathrm{GS}_{\Phi}(\mathbf{x}^{W};\mathbf{x}^{L},
\mathbf{x}^{0}),
\]
where frozen bulk, patch-boundary, axis, pole-axis, center, and cap-apex nodes
remain at their constrained Lagrangian coordinates. The barrier objective uses
cgs RZ coordinates, exact oriented RZ cell volumes, and signed corner-J
samples. Its relative floors are taken from the existing center-patch quality
thresholds,
\(V_{\mathrm{floor},c}=
\texttt{multiblock_center_patch_vol_on}\,|V^0_c|\) and
\(J_{\mathrm{floor},c,k}=
\texttt{multiblock_center_patch_cornerj_on}\,|J^0_{c,k}|\), using the
persistent reference geometry \(\mathbf{x}^0\). If the Winslow seed is outside
that barrier domain, the optimizer retries from the persistent reference seed.
When the trigger is off, \(\mathbf{x}^{B}\equiv\mathbf{x}^{W}\); in both cases
the CSR line search below remains the final admissibility filter.
The candidate displacement is
\[
\Delta\mathbf{x}_n=\mathbf{x}^{B}_n-\mathbf{x}^{L}_n ,
\]
which is exactly zero for bulk and patch-boundary nodes because those nodes are
masked inactive. An orientation-aware CSR line search chooses the largest
\(\sigma\in[0,1]\) accepted by the same positive exact-RZ-volume, signed
corner-J, and 2x2 Gauss-J floors used by the other reference builders. The
installed remap target is
\[
\mathbf{x}^{ref}_n=\mathbf{x}^{L}_n+\sigma\,\Delta\mathbf{x}_n .
\]
`State.cell_vol_initial` is recomputed from these accepted reference
coordinates with exact RZ quadrilateral volumes and exact cap-triangle polygon
volumes, multiplied by each multiblock cell orientation sign. It is not
\(\alpha^3 V^0_c\). The driver restores \(\mathbf{x}^{L}\) before CSR
conservative remap, and the remap path skips the differential/scaled reference
installers under this flag so they cannot overwrite the accepted center-patch
target.

**単調性保存**：Van Leerスロープリミッタにより、remap時に非物理的な振動（新しい極値の発生）を防止する。

**First-order conservative reference remap for shock-frame 2D RZ (PR B,
2026-05-17):** `Numerics.ale.conservative_remap_enabled=true` selects a
separate Lagrange-plus-remap closure used after each 2D RZ Lagrangian hydro
update.  The target mesh is restricted to
`Numerics.ale.conservative_remap_target="reference"`, i.e. the IC mesh stored
in `State.x_r_reference`, `State.x_z_reference`, and
`State.cell_vol_initial`.  The scheduled Winslow rezone block is bypassed in
this mode; driver-requested emergency ALE repairs remain available.

For multiblock meshes, `Numerics.ale.multiblock_scaled_reference_enabled=true`
replaces the static IC reference target with a homologously moving target.  Let
\(\mathcal{O}\) be the set of polar-shell outer-ring nodes tagged
`NODE_OUTER_PHYSICAL_BOUNDARY`; the instantaneous scale is
\[
\alpha(t)=
\frac{1}{|\mathcal{O}|}\sum_{i\in\mathcal{O}}
\frac{\sqrt{R_i(t)^2+Z_i(t)^2}}{s_{\max,0}} .
\]
The conservative-remap target and reference-barrier target are then
\[
\mathbf{x}_i^R(t)=\alpha(t)\,\mathbf{x}_{i,0},
\qquad
V_c^R(t)=\alpha(t)^3 V_{c,0}.
\]
For multiblock volume targets, the reference volume sign is taken from
`MultiBlockTopology::cell_orientation_sign[c]`; this replaces the older
three-block-only `cell < n_cells_core` core/non-core inference and covers both
the central/fan/shell five-block winding and the legacy three-block winding.
The flag is default-off; with it disabled, single-block, tri-fan, and
multiblock decks continue to use the static IC reference arrays bit-exactly.

Let \(X^L\) be the post-Lagrange mesh and \(X^R\) the reference target.  For a
face with Lagrange endpoints \((\mathbf{x}^L_0,\mathbf{x}^L_1)\) and reference
endpoints \((\mathbf{x}^R_0,\mathbf{x}^R_1)\), the swept RZ volume is the
axisymmetric volume of the quadrilateral
\((\mathbf{x}^L_0,\mathbf{x}^R_0,\mathbf{x}^R_1,\mathbf{x}^L_1)\):
\[
\Delta V_f =
-\frac{\pi}{3}\sum_{k=0}^{3}(r_k+r_{k+1})
(r_kz_{k+1}-r_{k+1}z_k),
\qquad k+1\equiv0\pmod4 .
\]
The sign is chosen so that, for a logical face between cell \(K\) and neighbor
\(K'\), \(\Delta V_f>0\) means transfer from \(K\) to \(K'\); the donor is
\(K\).  For \(\Delta V_f<0\), the donor is \(K'\).  The update is unsplit
first-order donor-cell:
\[
Q_K^R =
Q_K^L - \sum_{f\in\partial K}s_{Kf}\,\Delta V_f\,q_{d(f)}^L
       + Q_{K,\partial\Omega},
\]
where \(Q\) is an extensive conserved quantity, \(q=Q/V\) is its cell density,
and \(s_{Kf}=+1\) when the face sign is outward from \(K\).  The transported
extensives are mass \(\rho V\), momenta \(\rho u_rV,\rho u_zV\), electron and
ion material energies \(\rho e_eV,\rho e_iV\), and, when
`conservative_remap_radiation_enabled=true`, each radiation group energy
\(E_gV\).  Recovered primitives use \(V_K^R=V_K^{initial}\), with density and
energy floors applied only as positivity guards.

When the process environment flag `TENRYU_CAP_ENERGY_AUDIT` is enabled, the
driver emits a measurement-only cap energy audit.  It records
\(E^n\), \(E^{Lag}\), and \(E^R\) from the normal matter+radiation energy
reductions, and
\[
R_{Lag}=E^{Lag}-E^n-W_{ext}-S_{other},\qquad
R_{remap}=E^R-E^{Lag}.
\]
The pressure-boundary work uses the same `r_outer="pressure"` force launch as
the half-step momentum update:
\[
W_{ext}=\Delta t\sum_{p\in\partial\Omega_{r,out}}
{\bf F}_{p}^{n+1/2}\cdot{\bf u}_{p}^{n+1/2}.
\]
For CSR conservative remap, the same audit remaps an independent scalar
\(K_K^L/V_K^L\), where
\(K_K^L=\sum_{p\in P(K)}\frac12 m_{Kp}^L|{\bf u}_p^L|^2\), through the same
swept-volume fluxes and reconstruction order as the mass remap.  After the
normal node-velocity reconstruction and corner-mass update, it reports
\[
D_K=\sum_K\left(\widetilde K_K^{cons}
-\sum_{p\in P(K)}\frac12\widetilde m_{Kp}|{\bf u}_p^R|^2\right).
\]
This audit does not alter cell energies, node velocities, corner masses, or
any ALE acceptance/retry predicate.

When the process environment flag `TENRYU_I1B_SPURIOUS_SENSOR` is enabled, the
driver emits a measurement-only localization line for the I1-B near-axis
kinetic-energy diagnostics.  The default is off; when off, no sensor storage is
allocated and no sensor kernels are launched.

The hydro sensor is evaluated after the accepted 2D RZ hydro velocity update and
post-corrector geometry refresh.  For each active non-triangular cell, with
corner masses \(m_p\), positions \({\bf x}_p\), and nodal velocities
\({\bf u}_p\), it forms the mass-weighted means
\[
{\bf x}_c={\sum_p m_p{\bf x}_p\over \sum_p m_p},\qquad
{\bf u}_c={\sum_p m_p{\bf u}_p\over \sum_p m_p},
\]
then \(X_p={\bf x}_p-{\bf x}_c\), \(U_p={\bf u}_p-{\bf u}_c\),
\[
C=\sum_p m_p X_pX_p^T,\qquad D=\sum_p m_p U_pX_p^T,\qquad
B=D\left(C+10^{-12}\operatorname{tr}(C)I\right)^{-1}.
\]
The non-affine residual is \(h_p=U_p-BX_p\).  The reported residual energy is
\(\frac12\sum_p m_p|h_p|^2\), the affine reference energy is
\(\frac12\sum_p m_p|BX_p|^2\), and
\[
\eta^2={\sum_p m_p|h_p|^2\over \sum_p m_p|U_p|^2+10^{-300}}.
\]
The sensor keeps only per-block top entries on device and copies those reduced
records to the host for top-\(K\) selection.

The ALE sensor is evaluated across the velocity cell-to-node projection in the
2D RZ conservative remap.  For CSR remap it reuses the diagnostic scalar
kinetic-energy remap \(\widetilde K_K^{cons}\) above and compares it with the
post-projection corner kinetic energy \(\widetilde K_K^{node}\).  For
single-block remap it compares the remapped cell kinetic energy
\(\frac12 m_K^R|{\bf u}_K^R|^2\) with the post-projection corner kinetic
energy.  The reported cell residual is
\[
\Delta K_K=\widetilde K_K^{node}-\widetilde K_K^{before},
\]
with top-\(K\) selected by positive \(\Delta K_K\), plus rank-reduced sums and
maxima.  This sensor does not alter cell energies, node velocities, corner
masses, HDF5 output, or any ALE acceptance/retry predicate.

For a state-supply z boundary, the reference mesh is stationary in the normal
direction, \(w_z=0\), so PR E (2026-05-17) uses the adjacent interior cell
velocity \(v_f=u_{z,K}\) as the signed face speed and chooses the donor by
upwind direction.  At \(z=z_{min}\), \(v_f\ge0\) is inflow from the supply
reservoir and \(v_f<0\) is outflow with interior donor \(K=(i,0)\):
\[
\Delta M_{domain,f}=\rho_d v_f A_f\Delta t .
\]
At \(z=z_{max}\), \(v_f\ge0\) is outflow with interior donor \(K=(i,n_z-1)\)
and \(v_f<0\) is inflow from the supply reservoir:
\[
\Delta M_{domain,f}=-\rho_d v_f A_f\Delta t .
\]
The boundary momentum, material-energy, and radiation increments use the same
donor state,
\[
\Delta P_{z,f}=\Delta M_{domain,f} u_{z,d},\qquad
\Delta E_{gas,f}=\Delta M_{domain,f}(e_{e,d}+e_{i,d}),\qquad
\Delta E_{rad,g,f}=\pm E_{g,d}v_fA_f\Delta t ,
\]
with the plus sign at \(z_{min}\) and minus sign at \(z_{max}\).
\(A_f=\pi(r_{i+1}^2-r_i^2)\) is evaluated on the reference z face.
Closed-domain internal-face fluxes cancel pairwise because both cells use the
same swept-volume value and donor.  Open domains satisfy
\(\Delta M_{domain}=\sum_f\Delta M_f\) up to roundoff, apart from explicit
positivity floor injection.

For multiblock `Mesh.topology_scheme` values, S4-T1-next T3 uses the CSR
first-order donor path instead of structured `(i,j)` faces.  For a cell RZ
polygon edge \(a\to b\), define the Pappus edge contribution
\[
E(a,b)=(R_a+R_b)(R_aZ_b-R_bZ_a).
\]
The signed four-corner polygon volume is
\[
V_c=s_c\,{\pi\over3}\sum_{k=0}^{3}E(k,k+1),
\]
where \(s_c\) is `MultiBlockTopology::cell_orientation_sign[c]`, matching the
multiblock geometry winding convention in §3.2.2.  This orientation metadata,
not a `cell < n_cells_core` structured-id split, is the authoritative sign for
all CSR ALE/remap swept volumes.  For the three-block
cart-core/polar-shell mesh this reproduces the legacy signs exactly:
\(+1\) for core cells and \(-1\) for bridge/shell cells.  For the
half-butterfly five-block mesh it is \(+1\) for the central block and \(-1\)
for all fan/shell cells.  The outward swept volume for local face \(k\) is
\[
\Delta V_{c,k}=s_c\,{\pi\over3}
\left(E^{R}_{k,k+1}-E^{L}_{k,k+1}\right).
\]
Current CSR face slots use the topology order
`{inner, outer, lower, upper}`; the implementation maps these slots to polygon
edges `{3, 1, 0, 2}` before evaluating the formula above.  For every cell,
\[
\sum_{k=0}^{3}\Delta V_{c,k}=V_c^R-V_c^L
\]
to floating-point roundoff.  Internal faces are evaluated once from the
`unique_internal_faces` list and applied with equal/opposite signs to the two
adjacent cells.  The first-order donor rule is
donor \(=c\) when \(\Delta V_{c,k}>0\), otherwise the adjacent cell; boundary
faces use self-donor flux.  Thus a uniform \(\rho=1\) state satisfies
\(m_c^R=m_c^L+\sum_k\Delta V_{c,k}=V_c^R\).

For `polar_center_treatment="tri_fan"`, `ale_remap_2d_rz` uploads
`cell_nverts` only when the mesh contains center triangles; otherwise the device
pointer is null and the rectangular/annular arithmetic path is unchanged. In
tri_fan mode the polar cells use clockwise RZ winding, so the swept-volume
value above is multiplied by the same orientation sign \(-1\) used by
Stage-1 geometry. The center-cell inner origin face remains zero: it is skipped
by the existing `i>0` guard and any degenerate swept polygon also returns no
flux through the finite-nonzero check.

For a center cell `(0,j)`, with bottom spoke, top spoke, and outer radial face
as its three real faces, the signed GCL relation is
\[
-\Delta V_{\rm outer}-\Delta V_{\rm top}+\Delta V_{\rm bottom}
=V^R_{0,j}-V^L_{0,j}.
\]
The first-order donor update therefore preserves a constant state and conserves
mass, momentum, and separate electron/ion internal energies on closed tri_fan
meshes. `NODE_CENTER` node velocities are re-pinned to
\(v_r=v_z=0\) after the post-remap mass-weighted velocity projection.

For `polar_center_treatment="button"`, the conservative reference remap keeps
the same single reference target \(X^R\), line search, and reference-volume
recompute used by the ordinary structured pass.  The button cell is stored as
`c=0`; its real outer boundary is the node ring
\[
B_j=\texttt{node}(I_{\rm btn},j),\qquad j=0,\ldots,N_\theta .
\]
For each seam segment \(j=0,\ldots,N_\theta-1\), adjacent to shell cell
\((I_{\rm btn},j)\), the signed swept volume is
\[
\Delta V_{{\rm btn},j}
=s_{\rm btn}\left[-V(B_j^L,B_j^R,B_{j+1}^R,B_{j+1}^L)\right],
\qquad s_{\rm btn}=-1 .
\]
The sign convention is \(\Delta V_{{\rm btn},j}>0\) for button-to-shell
transfer.  A single donor-upwinded face flux
\[
F_j=q_d^L\Delta V_{{\rm btn},j}
\]
is applied pairwise as \(Q_{\rm button}^R\mathrel{-}=F_j\) and
\(Q_{I_{\rm btn},j}^R\mathrel{+}=F_j\).  Thus mass, \(R\)-momentum,
\(Z\)-momentum, material energy, and radiation energy close exactly pairwise
across the button seam up to floating-point roundoff.  The closing edge
\(B_{N_\theta}\to B_0\) lies on the \(R=0\) axis and contributes zero swept
volume.  The button cell recovers its density with the same discrete GCL target
volume used by the seam fluxes,
\[
V_{\rm btn}^{R,\mathrm{GCL}}=V_{\rm btn}^L-\sum_j\Delta V_{{\rm btn},j},
\]
rather than an independent button-polygon recomputation.  Active shell cells in
button mode use the same exact RZ polygon volume, area, centroid, and S-vector
formula with the polar orientation sign, including
`State.cell_vol_initial`.  Therefore the shell-side GCL target volume and the
structured/button swept-volume faces are the same discrete geometry.
Structured shell-shell faces \(i\ge I_{\rm btn}\) continue to use the ordinary
structured face loop with the button/polar orientation sign.  Dormant inner
structured cells \(c\ne0,\ i<I_{\rm btn}\) preserve their stored state and are
not valid donors or targets; ordinary faces touching them contribute zero flux,
their pre-remap volume storage is preserved, and post-remap EOS reclosure leaves
their stored thermodynamic fields unchanged.
When `"second_order_van_leer"` is selected, the button seam and structured faces
adjacent to the first active shell ring use the first-order donor flux so the
second-order reconstruction stencil never reads dormant storage.  Other active
shell-shell faces continue to use the second-order reconstruction.

For the pressure-driven single-block polar `tri_fan` case, the optional
`Numerics.ale.conservative_remap_lagrangian_bulk_enabled=true` finalizes the
conservative-remap reference target after the local center tracking reference
has been installed.  This path is default-off and scoped to
`conservative_remap_enabled=true`,
`conservative_remap_target="reference"`,
`Mesh.logical_mesh_2d="spherical_polar_halfplane"`,
`Mesh.polar_center_treatment="tri_fan"`, a single-block mesh, and
`Numerics.hydro.boundary_2d.r_outer="pressure"`.  Let
\(M=\texttt{Numerics.ale.conservative\_remap\_lagrangian\_bulk\_center\_node\_ring\_max}\).
For node rings \(i>M\), the finalized reference is set exactly to the current
post-Lagrange coordinates \(X^R_i=X^L_i\).  The swept volume on fully bulk
faces is therefore zero, so the conservative remap does not drain a compressed
bulk shell back to the initial-radius reference volume.  For the local center
band \(i\le M\), the reference remains the tri_fan tracking/rezone target so
the axis fan can still use the center stabilization target.

After this mixed target is assembled, `State.cell_vol_initial` is recomputed
from the finalized reference coordinates, including the center-band target and
bulk Lagrangian rings.  The GCL condition consumed by the remap is therefore
\(V_c^R=V_c[X^R]\) for the exact reference coordinates used in each swept
volume, avoiding a target-coordinate/reference-volume mismatch.  A diagnostic
warning is emitted if the minimum bulk corner-J ratio for cells with
\(c_i>M\) falls below the warning floor.  The warning is behavior-neutral; it
means the Lagrangian bulk mesh is approaching a tangle and the deck should
raise `conservative_remap_lagrangian_bulk_center_node_ring_max` if the
degraded cells need to be moved into the locally tracked center band.

`Numerics.ale.conservative_remap_order` selects the donor reconstruction used
by this reference remap.  The default
`"first_order_donor"` keeps the PR B cell-average donor flux above bit-exact.
`"second_order_van_leer"` (PR C, 2026-05-17) replaces the donor density
\(q_{d(f)}^L\) by a limited cell-centered reconstruction.  For each cell
\(K\), component slopes are built from neighbor center differences with a
Van Leer/minmod form,
\[
s_{r,K}={\rm VL}\!\left(
  {q_K-q_{i-1,j}\over r_K-r_{i-1,j}},
  {q_{i+1,j}-q_K\over r_{i+1,j}-r_K}\right),\quad
s_{z,K}={\rm VL}\!\left(
  {q_K-q_{i,j-1}\over z_K-z_{i,j-1}},
  {q_{i,j+1}-q_K\over z_{i,j+1}-z_K}\right),
\]
with one-sided slopes at physical boundaries and zero slope when no neighbor
exists.  Let \((\bar r_f,\bar z_f)\) be the swept-volume centroid used for the
single face flux.  The unlimited face value is
\[
q^*_f=q_K+s_{r,K}(\bar r_f-r_K)+s_{z,K}(\bar z_f-z_K).
\]
A Barth-Jespersen limiter scales each face increment by
\[
\psi_{Kf}=
\begin{cases}
\min\!\left(1,{q_{\max,K}-q_K\over q^*_f-q_K}\right), & q^*_f>q_K,\\
\min\!\left(1,{q_{\min,K}-q_K\over q^*_f-q_K}\right), & q^*_f<q_K,\\
1, & q^*_f=q_K,
\end{cases}
\]
where \(q_{\min,K},q_{\max,K}\) are the min/max over \(K\) and its face
neighbors.  The second-order flux is
\[
F_f^{(2)}=\Delta V_f
\left[q_K+\psi_{Kf}\{s_{r,K}(\bar r_f-r_K)+s_{z,K}(\bar z_f-z_K)\}\right].
\]
The same \(F_f^{(2)}\), donor, and swept-volume centroid are used by both
cells adjacent to an internal face, preserving pairwise conservation.  Density,
specific electron/ion energies, and radiation energy density are clamped
non-negative after limiting; final cell density and specific energies still use
the PR B positivity floors.

For tri_fan decks using `"second_order_van_leer"`, any face whose donor pair
contains a triangular center cell falls back to the first-order donor flux
above. Pure quad-to-quad faces continue to use the second-order reconstruction.
Fan-touching second-order reconstruction and `total_energy_remap_2d_rz=true`
remain deferred for tri_fan.

For button decks using `"second_order_van_leer"`, the button seam is always
first-order donor, and structured faces adjacent to the first active shell ring
\(i=I_{\rm btn}\) also fall back to first order.  This keeps the limiter and
neighbor extrema stencil over active storage only.  Shell-shell faces whose
reconstruction stencils contain only active shell cells continue to use the same
second-order path as the annular/rectangular structured remap.

For `topology_scheme="multiblock_cart_core_polar_shell"`, the CSR remap path
uses the same `conservative_remap_order` selector.  `"first_order_donor"` keeps
the S4-T1-next T3 unique-oriented face update.  `"second_order_van_leer"` builds
per-cell least-squares density gradients over the CSR face-adjacent neighbors
with weights \(w_{KL}=1/\lVert x_L-x_K\rVert^2\):
\[
\begin{bmatrix}g_R\\g_Z\end{bmatrix}_K =
\left(R_K^T W_K R_K\right)^{-1} R_K^T W_K
\left(q_L-q_K\right).
\]
Cells with fewer than two valid face neighbors, or a singular \(2\times2\)
normal matrix, use \(g_K=0\).  The gradient is then scaled by the
Barth-Jespersen cell limiter computed over all four CSR face centers, with a
roundoff tolerance \(10^{-14}\max(|q_{\max,K}-q_{\min,K}|,1)\) in the limiter
ratio decision, where \(q_{\min,K},q_{\max,K}\) are the bounds over \(K\) and
its valid face neighbors.  BJ limits the reconstructed face values on this
stencil; it is not a flux-corrected transport limiter and does not by itself
guarantee post-remap cell-average global extrema preservation.  For each unique
internal face \(f=(A,B)\), TENRYU uses the canonical `cell_a/local_a` face
center as \(x_f\) for both adjacent-cell reconstructions, then chooses the donor
from the T3 swept volume sign \(dV_A\); the mass flux is
\[
F_f = q_{d(f)}^{BJ}(x_f)\,dV_A ,
\]
and the two adjacent cells receive equal and opposite updates.  Boundary faces
use the owner cell as donor.  This preserves the T3 pairwise conservation and
constant-state GCL while adding bounded second-order density reconstruction on
the multiblock CSR topology.  When CSR radiation remap is enabled, each
radiation group uses the same LSQ/BJ scalar reconstruction for its volume
scalar flux.

For `topology_scheme="multiblock_cart_core_polar_shell"`, S4-T1-next T5a adds a
test-invoked, per-block CSR Winslow smoother.  It is not wired into the
production ALE driver.  For a movable node \(n\), the reverse cell-node CSR
gives the incident cells \(\mathcal{C}(n)\).  The node-neighbor set is the
deduplicated union of the other three corner nodes in each \(c\in\mathcal{C}(n)\):
\[
\mathcal{N}(n)=
\bigcup_{c\in\mathcal{C}(n)}
\{m:\,m\in\mathrm{nodes}(c),\,m\ne n\}.
\]
One Jacobi sweep with uniform weights is
\[
\mathbf{x}_n^{k+1}=(1-\omega)\mathbf{x}_n^k
+{\omega\over|\mathcal{N}(n)|}\sum_{m\in\mathcal{N}(n)}\mathbf{x}_m^k,
\qquad \mathbf{x}=(R,Z).
\]
The direct T5a driver applies this Jacobi operator to the displacement
\(\mathbf{d}_n=\mathbf{x}_n-\mathbf{x}_{n,\mathrm{ref}}\) and reconstructs
\(\mathbf{x}^{k+1}=\mathbf{x}_{\mathrm{ref}}+\mathbf{d}^{k+1}\), so the
generated gamma reference mesh is an exact fixed point.
Nodes tagged `NODE_AXIS`, `NODE_POLE_AXIS`, `NODE_OUTER_PHYSICAL_BOUNDARY`, or
`NODE_CENTER` are fixed.  T5a also fixes any node whose incident cells span more
than one block or whose incident cell has a face tagged `SEAM_CORE_BRIDGE` or
`SEAM_BRIDGE_SHELL`; cross-seam coordination is deferred to T5b.  After each
candidate sweep, TENRYU evaluates `evaluate_candidate_mesh_quality_csr` from the
pre-sweep coordinates and the proposed displacement.  If any CSR cell is
inadmissible, \(\omega\) is halved and the same sweep is retried; dropping below
\(\omega=0.01\) is a fatal diagnostic.  When the current coordinates are exactly
the stored reference coordinates, the direct T5a driver returns without a sweep,
preserving the static generated gamma mesh bitwise.  `single_block` and
`tri_fan` topologies bypass this CSR smoother.

**体積分率の再正規化**：各物質の体積分率 \(f_\alpha\) は個別にremapされるため、
浮動小数点誤差により \(\sum_\alpha f_\alpha \ne 1\) となりうる（典型的には \(O(10^{-15})\)/step）。
remap完了後に再正規化を行う：
\[
f_\alpha^{renorm} = \frac{f_\alpha^{remap}}{\sum_\beta f_\beta^{remap}}
\]
同時に非負性を保証する：\(f_\alpha = \max(f_\alpha, 0)\)（Van Leer で極小値近傍に微小負値が残る可能性への対策）。

> **退化ガード**：非負クランプ後に \(\sum_\beta f_\beta^{remap} < \varepsilon_f\)（\(\varepsilon_f = 10^{-30}\)）となる場合、
> 最大成分の添字 \(\alpha^* = \arg\max_\alpha f_\alpha^{remap}\) に対し \(f_{\alpha^*} = 1\)、
> 他を \(f_\alpha = 0\) に設定する（NaN/Inf 防止）。DeviceErrorFlags の `volfrac_degenerate` を設定する。

**Stage 30 Wave C PLIC material-volume remap**: when
`Numerics.plic.enabled=True`, `Numerics.plic.in_run_disabled=False`, and the
per-run sticky fallback is not engaged, ALE keeps the scalar remapper above for
\(\rho\), \(\rho u_r\), \(\rho u_z\), \(\rho e_e\), \(\rho e_i\), diagnostic
kinetic energy, and radiation energy.  Only the material volume integrals
\(f_m V\) branch to the PLIC material-volume remapper.

With the Stage 30 Wave E CF6 preview
`Numerics.plic.rho_material_aware_donor=True`, this invariant changes only for
\(\rho\): after successful PLIC face-volume construction, density is gathered
with \(\Delta V_f\rho^{\rm eff}_d\) at interface donor cells, where
\(\rho^{\rm eff}_d=\sum_m f_{d,m}\rho_{d,m}\).  The \(\rho_{d,m}\) values are
queried/estimated at runtime and are not stored in `State`.  If CF6 is false or
PLIC remap falls back, density remains on the scalar remapper.

For each R-face and Z-face, Wave C computes one signed swept volume
\(\Delta V_f\) with the same helper used by the scalar remapper.  PLIC then
computes per-material face volumes \(\Delta V_{f,m}\) from the donor-cell
interface reconstruction and enforces face closure exactly:
\[
\sum_m \Delta V_{f,m} = \Delta V_f .
\]
The stored face values are gathered antisymmetrically into the two adjacent
cells, so an internal face cancels as the same floating-point value rather than
two recomputed values.  After the gather,
\[
f'_{c,m} = \frac{f_{c,m}V_c-\sum_{f\in \partial c}s_{cf}\Delta V_{f,m}}
                {V'_c}
\]
is snapped by the deterministic residual rule
\[
r_c = 1-\sum_m f'_{c,m}, \qquad
m^* = \min\arg\max_m f'_{c,m}, \qquad
f'_{c,m^*}\leftarrow f'_{c,m^*}+r_c .
\]
The largest-material residual snap replaces the generic normalization kernel
for PLIC-active material-volume remap.  Any clamp, non-finite value, or
ill-conditioned reconstruction is a class-(d) PLIC repair.  Pure-material
states, no-interface states, disabled PLIC, `in_run_disabled=True`, and sticky
fallback all use the scalar material-fraction remap and the generic
normalization path above.

Wave C PLIC remap is serial-only.  If `part.n_ranks > 1` and
`Numerics.plic.enabled=True`, ALE rejects the step with
`ConfigError("Wave C PLIC remap not validated under MPI; deferred to Stage 31")`.

**質量保存精度**：機械精度（\(\le 10^{-14}\) 相対誤差）。

**Post-remap 熱力学的 reclosure**：
remap は保存量（\(\rho e_e V\)、\(\rho e_i V\) 等）を転写するが、原始変数（\(T_e, T_i, P_e, P_i, c_s\)）は更新しない。
remap 完了後、以下のシーケンスで原始変数を再構築する：
1. セル幾何（体積・面積・特性長）を新メッシュから再計算（H7）
2. 密度復元：\(\rho^{new} = m^{new} / V^{new}\)（H8）
3. 節点速度再投影：セル中心運動量→節点速度（質量重み投影、A4）
4. EOS 逆変換：`HydroEOSContext` に table-backed EOS がある成分は
   `device_inverse_reclose_with_low_density_extrap` で
   \((\rho^{new}, e_e) \to T_e\)、\((\rho^{new}, e_i) \to T_i\) を行う。
   table がない成分、または EOS context がない場合は従来の理想気体式を用いる（H14）。
5. EOS 順変換：table-backed 成分は同 helper の \(P,e,C_v\) を書き戻す。
   `Materials.low_density_extrapolation=True` かつ \(\rho < \rho_{\min}^{table}\) の場合は
   §1.1.5 の共有 low-density analytic policy を用い、False の場合は table edge clamp を維持する。
   ion table の low-density 分岐では同 helper に \(Z_{\rm eff}=1\) を渡し、従来の単原子イオン理想気体 \(C_v\) と整合させる。
   table がない成分は従来の
   \(P=(\gamma-1)\rho e\)、\(C_v \propto 1/(A m_p(\gamma-1))\) を用いる（H13）。
6. 音速：§1.1.6 準拠（H15。理想気体: \(c_s = \sqrt{(\gamma_e P_e + \gamma_i P_i)/\rho}\)、テーブルEOS: 等エントロピー偏微分）
7. 安全策（§11）：温度・密度フロアクランプ（U2）

CSR remap has one additional post-remap closure rule for evacuated cells.  If a
cell's remapped true mass satisfies \(m^{raw}<m_{floor}=\rho_{floor}V\), the
finish kernel sets \(m=m_{floor}\), \(\rho=\rho_{floor}\), and resets the
specific electron/ion energies to the temperature-floor values
\[
e_e=c_{v,e}T_{e,floor},\qquad e_i=c_{v,i}T_{i,floor}.
\]
The signed internal-energy adjustment
\[
\Delta E_{unresolved}=m(e_e+e_i)-(E_e^{raw}+E_i^{raw})
\]
is accumulated in `E_redistribution_unresolved`; the positive mass insertion is
accumulated in `mass_floor_delta`.  The driver reports this bounded closure via
`[ale-remap-floor-closure]`.  Cells with \(m^{raw}>m_{floor}\) keep the ordinary
specific-energy reclosure \(e_k=E_k^{raw}/m^{raw}\) followed by the existing
component floors.

この reclosure は §2.1 の Phase 5（2回目 H(Δt/2) 後の ALE）で実行される。CUDA カーネルは CUDA_KERNELS §9 Phase 5 参照。
温度 floor により追加された内部エネルギーは、従来どおり `E_floor` に加算する。

Known residual limitation: this closure removes only the floor-state
sound-speed explosion and the resulting timestep collapse.  It is not a
\(C\ge7.2\) completion.  The current convergent \(C\ge7.2\) ceiling is the FAN
mesh tangle, observed as non-positive cell volume / geometry inversion at a
block-3 fan cell around step 177 of the five-block \(P_{ext}=10^{10}\)
convergent run, plus the axis-rezone locality constraint documented in
§3.3.5.  Those items are separate from this thermodynamic floor closure.

#### 3.3.5 Rezone制約

- **ノード変位上限**：
\[
|\mathbf{r}_n^{new} - \mathbf{r}_n^{Lag}| \le \alpha_{rezone}\cdot\min_{c\in\mathcal{C}(n)} \sqrt{A_c}
\]
ここで \(\alpha_{rezone} = 0.5\)（既定、`rezoning.max_displacement_fraction`）、
\(A_c\) はセル断面積（§3.2.3）、\(\mathcal{C}(n)\) はノード \(n\) に隣接するセル集合。
超過した場合は変位方向を維持したまま上限でクランプする。

**軸近傍のcap floor (cap floor for axis-adjacent rows):**

軸近傍の \(i=1\) 行ノードでは、変位capの下限を
\(0.25 \times \min(dr_{\text{init}}, dz_{\text{init}})\) で押さえる:

\[
\text{max\_disp}_{i=1} = \alpha_{rezone}\cdot
\max(l_{\min}^{\text{current}},
  0.25\cdot\min(dr_{\text{init}}, dz_{\text{init}}))
\]

これは、軸近傍セルが極端に圧縮された場合（Sedov等で発生）に
\(l_{\min}^{\text{current}} \to 0\) となり、Winslow Jacobi反復が動けず
収束が停滞する病的ケースを回避するためである。\(i \neq 1\) では現状のcap式
\(\alpha_{rezone}\cdot l_{\min}^{\text{current}}\) を維持する。

軸ノード自身（\(i=0\)）は `NODE_AXIS` フラグで凍結されるため、cap floorは
その内側（\(i=1\)）のみに適用される。

**Phase 2: 軸近傍の特異 stencil に対する Laplacian fallback:**

軸近傍 (\(i=1\)) のノードにおいて、Jacobi stencil の Jacobian \(J \leq J_{\text{floor}}\)
となる特異セルでは、Winslow 更新の代わりに 4 近傍の uniform Laplacian fallback を適用する:

\[
\mathbf{x}^{\text{candidate}}_{i=1} = \frac{1}{4}\sum_{m \in \{i\pm,j\pm\}} \mathbf{x}_m
\]

更新量は Phase 1 と同じ cap (`max_disp = α_{rezone} \cdot \max(l_{\min}, 0.25 \cdot \min(dr_{\text{init}}, dz_{\text{init}}))`)
で制限される。

理由: Phase 1 (cap-floor) のみでは \(J \leq J_{\text{floor}}\) の特異 stencil で
Jacobi update が `x_new = x_old` の no-op に early-return し、
"stuck" 状態のノードが残存する。fallback は metric を介さない
(metric が特異だから fallback が必要) uniform 平均で、
近傍ノードへの幾何的な単純引き寄せを cap 内で行う。

\(i \neq 1\) では既存の no-op early-return 動作を維持する。
post-iteration の rollback (J<0 検出) ガードは不変であり、
fallback による over-shoot は最終的に許容されない。

**Phase 3.1: instrumentation (no physics change):**

To diagnose cases where Phase 2 fallback is insufficient (e.g., H3-B 256×512
ALE-on tangling at axis row in different j positions), per-iteration counters
are logged on non-convergence/rollback events. These counters do not affect
the rezoner output; they only provide diagnostic data for further design.
Counter fields: `j_floor_i1_fallback_hits`, `j_floor_ix_skip_hits`,
`regular_cap_i1_hits`, `regular_cap_ix_hits`, `fallback_cap_hits`,
`zero_or_tiny_motion_hits`, `local_linesearch_rejects`,
`weighted_laplacian_fallback_hits`.

**Phase 3.2: rezone displacement backtracking (line search):**

When the rezoner produces a candidate mesh that fails remap admissibility
(i.e., non-positive intermediate volume in a directional remap sweep), a
backtracking line search is applied. Let
\(\Delta_n = x^{\text{cand}}_n - x^{\text{old}}_n\) be the rezoner-proposed
displacement. With the default
`Numerics.ale.safe_backtrack_enabled=false`, the legacy bit-exact schedule is
unchanged: try \(\lambda \in \{1, 1/2, 1/4, 1/8, 1/16, 1/32\}\) and accept the
first (largest) trial mesh
\[
x^{\text{trial}}_n = x^{\text{old}}_n + \lambda\Delta_n
\]
that passes both:
- the 4-Gauss Jacobian admissibility gate from §3.3.2
- all-positive intermediate volumes in the first directional remap sweep


If no \(\lambda\) passes, the ALE step falls back to full rollback and preserves
the Lagrangian state. This converts a full rezone rollback (no mesh relief) into
a partial but remap-admissible rezone when a reduced displacement is valid.
Cases accepted with \(\lambda = 1\) preserve the previous full-candidate path
bit exactly for the mesh coordinates.

With `Numerics.ale.safe_backtrack_enabled=true`, the driver first validates the
pre-rezone mesh by evaluating the same admissibility checks at \(\lambda=0\).
Failure at \(\lambda=0\) is reported as `AleStatus::PreRezoneInvalid` and
distinguishes a mesh that is already inadmissible before applying the rezone
displacement. If \(\lambda=0\) passes, the driver searches
\[
\lambda_e = 2^{-e}, \qquad e=0,\ldots,N,
\]
where \(N=\texttt{safe\_backtrack\_min\_exp}\) (default 20). If no positive
\(\lambda_e\) passes, the result is `AleStatus::NoLambdaAdmissible` with
\(\lambda_{\min}=2^{-N}\). Otherwise the first passing \(\lambda_e\) defines
`lo`; `hi=min(1,2*lo)` brackets the next larger rejected power-of-two trial, and
`safe_backtrack_binary_iters` (default 8) bisection steps refine upward to the
largest admissible value within that bracket. Accepted adaptive lambdas update
the run-level `safe_backtrack_lambda_distribution[bin]`, with
\(\text{bin}=\lfloor-\log_2\lambda\rfloor\) clamped to
\([0,\texttt{safe\_backtrack\_min\_exp}]\), and the distribution is logged at
run end.

**Phase 4: exact signed R-Z swept polygon volume:**

The remap admissibility check and conservative remap use the exact
polygon-revolution formula for swept face volumes:

\[
V(P) = \frac{\pi}{3} \sum_k (r_k + r_{k+1})(r_k z_{k+1} - r_{k+1} z_k)
\]

This replaces the previous first-order approximation
\(V_R^{\text{first-order}} = 2\pi \cdot r_{\text{old}} \cdot z_{\text{len}} \cdot \Delta r\),
which was missing the second-order term \(\pi \Delta r^2 z_{\text{len}}\).
The missing term is non-negligible near the axis where \(r_{\text{old}}\)
is small. The new formula is consistent with the cell-volume calculation
in `src/mesh/mesh.cu` (which already uses polygon revolution).

For uniform-mesh interior cells where \(\Delta r \ll r_{\text{old}}\), the new
formula is numerically indistinguishable from the old; bit-exact preservation
holds for production cases where rezoner is inactive (H1, H2, H3-A,
H3-B 128x256). The new formula matters near the axis
(\(r_{\text{old}} \to 0\)) and for highly-distorted faces where face skew makes
the first-order approximation inaccurate.

**Phase 5: analytic axis-cell positivity check:**

The 4-Gauss Jacobian check is not a sufficient non-tangle proof for
bilinear quadrilaterals. For axis cells (\(i=0,j\)) where the inner edge is
pinned to \(r=0\), the closed-form full-cell positivity condition is:

\[
M_{\text{axis}}(j) =
s \cdot \min(\hat r_j, \hat r_{j+1}) + \min(Q, 0) > 0
\]

where \(s = z_{0,j+1} - z_{0,j}\) is the axial spacing of the axis edge,
\(\hat r_j, \hat r_{j+1}\) are the outer-row r-coordinates, and
\[
Q =
\hat r_j(\hat z_{j+1} - z_{0,j+1})
- \hat r_{j+1}(\hat z_j - z_{0,j})
\]
is the axial skew. The condition is derived from the bilinear Jacobian
\[
J(\xi,\eta) =
\frac{1}{8}
\left\{
s[(1-\eta)\hat r_j + (1+\eta)\hat r_{j+1}]
+ (1+\xi)Q
\right\},
\]
which is affine and separable in \((\xi,\eta)\), so the corner minimum gives
the full-cell minimum.

This margin is checked alongside the 4-Gauss check and the remap
admissibility check in the rezone backtracking loop. A trial mesh is accepted
only if \(M_{\text{axis}}(j) > 0\) for all \(j \in [0,n_z-1]\).

**Phase 6: sliding-axis Z motion fallback:**

When the legacy frozen-axis Winslow rezoner fails to produce a non-tangling
candidate (all backtracking trials rejected by post-tangle, remap admissibility,
or analytic axis margin), the ALE driver enters a sliding-axis fallback path:

- Constrain axis nodes to \(R=0\) exactly (axisymmetry preserved).
- Allow axis nodes to move in \(Z\) toward the \(i=1\) row Winslow candidate.
- Build axis spine target:
  \(z_{0,j}^{target} = z_{0,j}^{old} + \omega (z_{1,j}^{cand} - z_{0,j}^{old})\).
- Preserve domain-boundary axis nodes and `NODE_CENTER` nodes.
- Enforce monotonicity
  \(z_{0,j+1} - z_{0,j} \geq s_{\min}\) (no axis node crossover).
- Enforce symmetry \(z_{0,n_z-j} = -z_{0,j}\) when the initial axis spine is
  symmetric about zero.
- Try \(\omega \in \{1.0, 0.5, 0.25, 0.125\}\); accept the largest that passes
  all checks.
- If all \(\omega\) are rejected, fall back to the existing rollback
  (preserves the Lagrangian state).

This is a fallback: cases where the legacy path accepts (\(\lambda > 0\)) are
unaffected and remain bit-exact. The sliding-axis path activates only when the
legacy path would rollback. It gives the rezoner an additional geometric degree
of freedom (axis \(Z\) motion) to handle cases where blast-driven axis-row
distortion is too severe for frozen-axis rezoning alone.

Per the Barlow-Burton-Shashkov R-Z Lagrangian discretization, the correct
axisymmetric boundary condition is \(R=0\) and radial mesh velocity \(=0\).
Axial mesh velocity at the axis is not constrained by axisymmetry. Freezing
\(Z\) is an extra Dirichlet condition in the legacy formulation, removed here
for the fallback path only.

**Phase 7: preventive axis-feasibility guard:**

A per-cycle GPU axis-margin check uses the Phase 5 analytic margin. The
trigger condition is
\[
\min_j M_{\text{axis}}(j)
< \alpha_{\text{guard}} M_{\text{axis,initial}},
\]
where \(M_{\text{axis,initial}}\) is the cached initial margin and
\(\alpha_{\text{guard}}\) is
`Numerics.ale.preventive_axis_guard_fraction` (default 0.1).

When triggered, ALE fires preemptively even if the configured
`every_n_steps` cadence has not elapsed. The guard also forces the 2D rezone
path when the axis margin is at risk, rather than waiting for the generic mesh
quality trigger. Its cost is one \(O(n_z)\) GPU kernel and one reduction per
cycle, negligible compared with a hydro step.

Behavior:
- \(\alpha_{\text{guard}} = 0\): disabled; legacy cadence-only triggering is
  preserved.
- \(\alpha_{\text{guard}} > 0\): preemptive trigger when the axis margin drops
  below the configured fraction of its initial value.

For spherical-polar `polar_center_treatment="tri_fan"`, the `i=0` row is a
pinned point topology, not a rectangular RZ axis row with finite \(Z\)-spacing.
The rectangular analytic axis-margin minimum is therefore non-applicable for
that origin row and is excluded from the preventive guard and hydro
axis-margin CFL limiter using topology detection (`cell_nverts` containing
triangles or a `NODE_CENTER` origin row). Tri_fan cell admissibility remains
the responsibility of `candidate_mesh_admissibility`, which evaluates the
actual triangle/quad RZ volume, corner-J, and Gauss-J constraints.

For H3-B 256x512 spherical Sedov, the legacy cadence (every 5 steps) is
insufficient to keep up with blast-driven axis-row distortion; preemptive
triggering lets the rezoner fire when the axis approaches the feasible-set
boundary, not only on the round-robin schedule.

**Phase 8a: mirrored-axis Winslow update (axis Z motion):**

The legacy `NODE_AXIS` branch freezes both R and Z for axis nodes (\(i=0\)).
This is overconstrained per Barlow-Burton-Shashkov R-Z compatible
discretization: axisymmetric BC requires only \(R=0\) plus radial mesh
velocity \(=0\); axial mesh velocity at the axis is unconstrained.

Phase 8a (opt-in via `Numerics.ale.axis_z_motion = "winslow"`) lets the
axis Z move during Winslow Jacobi iteration via the mirrored-axis update:

\[
z_{0,j}^{new} =
\frac{2\alpha\, z_{1,j} + \gamma(z_{0,j+1} + z_{0,j-1})}{2(\alpha + \gamma)}
\]

with \(\alpha = ((z_{0,j+1} - z_{0,j-1})/2)^2\) and
\(\gamma = r_{1,j}^2\). Derivation: from the inverse Winslow equation
\(\alpha z_{\xi\xi} + \gamma z_{\eta\eta}=0\) with mirror symmetry across
\(r=0\) (\(r_{-1,j}=-r_{1,j}\), \(z_{-1,j}=z_{1,j}\)). \(r_{0,j}=0\) is
preserved exactly.

`NODE_BOUNDARY` and `NODE_CENTER` axis nodes remain frozen. Domain-end axis
nodes (\(j=0\), \(j=n_z\)) remain frozen.

After each winslow-mode Jacobi sweep, the candidate axis spine is projected
through the same symmetry and monotonicity constraints used by the sliding-axis
repair path. If the initial axis spine is symmetric, projection enforces
\(z_{0,n_z-j}^{new}=-z_{0,j}^{new}\); all projected candidates enforce strictly
positive axis-spine spacing with the §3.3.5 spacing floor while preserving fixed
axis nodes.

The axis-adjacent row also has a radial lower-bound acceptance condition:
\[
r_{1,j}^{trial} \ge
\kappa_{\rm axis}\,r_{1,j}^{Lag},
\qquad
\kappa_{\rm axis} = \texttt{Numerics.ale.winslow\_axis\_kappa}
\]
with default \(\kappa_{\rm axis}=0.7\). A winslow-mode sweep that violates this
bound is marked failed, and ALE backtracking rejects trial displacements that do
not satisfy the same bound so that a smaller \(\lambda\) is tried.

Behind opt-in flag (default `"fixed"`) for bit-exact preservation of legacy
passing cases (H1, H2, H3-A, H3-B 128x256).

**Phase 8b: Lagrangian tangential axis-Z update (hydro step):**

`Numerics.ale.axis_z_motion = "lagrangian_tangential"` lets axis nodes slide in
Z during the hydro Lagrangian predictor/corrector position update while keeping
\(R_{0,j}=0\). The unconstrained `"lagrangian"` spelling remains reserved for a
future full-axis-motion design; Phase 8b uses only the tangential axis
component.

For each hydro position update, the candidate axis displacement is
\(\Delta z_j=\Delta t\,u_{z,0,j}\). Before committing, a monotone spine limit is
computed:
\[
\sigma_{\rm mono} =
\min_{j:\Delta t(u_{z,0,j}-u_{z,0,j+1})>0}
(1-\epsilon_z)
\frac{z_{0,j+1}-z_{0,j}}
{\Delta t(u_{z,0,j}-u_{z,0,j+1})},
\qquad \sigma_{\rm mono}\le 1 .
\]
The full-mesh candidate uses \(\Delta r=0\) everywhere and \(\Delta z\) only on
axis nodes, then calls the A0 `linesearch_largest_admissible_sigma` check for
exact RZ volume, corner-J, and Gauss-J admissibility. The accepted axis
displacement is \(\sigma\,\Delta z_j\).

If \(\sigma < 10^{-3}\), Phase 8b is marked ineffective for that step and the
axis-Z displacement is skipped rather than collapsing the hydro timestep. The
state telemetry counters
`axis_lagrangian_tangential_engaged_count` and
`axis_lagrangian_tangential_ineffective_count` record use and ineffective
events; verbose logging emits `[axis-phase8b] step N: sigma collapsed ...`.
Fine-grid acceptance reports ineffective events as telemetry instead of making
the run fail solely because this optional axis-Z motion could not be applied.

**Phase 9: reference-barrier ALE infrastructure (default-off):**

Reference-barrier ALE is an opt-in rezone path for late-time 2D_RZ mesh
degeneracy near the axis. It is disabled by default via
`Numerics.ale.reference_barrier_enabled=false`; production decks do not activate
it until the follow-on empirical probes establish cost and remap-limiter
behavior.

Engagement is trigger based. When enabled, the driver evaluates an axis-margin
predicate over axis-adjacent cells and a corner-J ratio predicate. The default
triggers fire when `sin(theta)` near the axis drops below
`reference_trigger_axis_margin_threshold`, when the corner-J ratio drops below
`reference_trigger_corner_j_ratio_threshold`, or when the diagonal-corner
\(b_{\rm eff}\) length ratio drops below the same corner-J trigger threshold.
The \(b_{\rm eff}\) term is sampled with the diagonal \(Y_2\) corner so a low
\(\psi=J_2/b_{\rm eff}\) cannot be hidden by axis-length collapse. If no
trigger fires, the legacy ALE path is unchanged.
The default thresholds are `reference_trigger_axis_margin_threshold=1e-2` and
`reference_trigger_corner_j_ratio_threshold=0.5`, chosen to engage before
axis-adjacent mesh inversion in the Phase-9 H3 probes. The debug-only
`reference_force_engage_every_step=false` knob bypasses these predicates when
set true and is intended for wiring and conservation tests only.

The configured target mesh is selected by `Numerics.ale.reference_target`:

- `"none"`: no target displacement.
- `"eulerian_initial"`: target nodes are the initial node positions captured at
  mesh initialization.
- `"spherical_equal_angle"`: fixed initial shell radii \(s_i\) are combined with
  equal angular spokes, \(r_{ij}=s_i\sin\theta_j\) and
  \(z_{ij}=s_i\cos\theta_j\). Shock-following monitor shells are deferred.

The rezone displacement is
\[
\Delta r_{ij}=r^{target}_{ij}-r^n_{ij},\qquad
\Delta z_{ij}=z^{target}_{ij}-z^n_{ij}.
\]
The accepted blend \(\lambda\) starts from
`reference_blend_default` and is limited by the A0
`linesearch_largest_admissible_sigma` check using exact RZ volume, corner-J, and
Gauss-J floors:
\[
x^{n+}_{ij}=x^n_{ij}+\lambda\,\Delta x_{ij}.
\]
The conservative remap remains the existing ALE remap infrastructure; the
reference-barrier module recomputes geometry after the accepted blend, then
uses the legacy_split donor-cell conservative remap for density, conserved
internal-energy densities, momentum, material volume fractions, and radiation
group energies. This remap is naturally bound-preserving for donor-cell scalar
transport. MS2 remap requires the bound-preserving limiter C-prerequisite before
production activation.

**Button morph (shock-ahead reorientation, S-C):**

The button morph uses the Shirley-Chiu equal-volume core target and circular
bridge target defined by
`docs/design/shock_ahead_button_reorientation_20260720.md`.
Its scheduled blend is the C2 quintic smoothstep
\(s(u)=6u^5-15u^4+10u^3\), with \(u\) clamped to \([0,1]\) over the hard
absolute-time window `[t_start_s, t_end_s]`; it is inert outside that window.
Each transaction uniformly scales all node displacements so that
\(\max_i|\Delta\mathbf{x}_i|/h_{local,i}\) does not exceed
`max_step_fraction`, where \(h_{local,i}\) is the minimum incident current-mesh
edge length. The capped target executes through the reference-barrier
transactional rezone, quality line search, and conservative CSR remap; the
existing mass and energy ledgers are unchanged.
The transaction is scoped to the morph mandate: only core+bridge cells are
active in the remap, and velocity projection is frozen for every shell node.
It is therefore bitwise inert outside its region even while a shock transits
the shell.
In 1T runs every remap-side ion-temperature floor in the 2D-RZ remap TU (reclosure, unpack, recover, and flux kernels) is passed as zero — ei == 0 is the 1T convention and flooring it fabricates unledgered energy each remap pass (measured +5e-5/pass before the fix).
In 1T runs the reclosure derives the single temperature from the total internal energy (Te = Ti = e_tot/(cv_e+cv_i), Pi = 0).
Version 1 is restricted to
`cart_core_polar_shell`; non-mandate nodes (the shell family, including the
seam ring) are targeted at their current positions so the transaction is a
no-op for them, while core and bridge-interior targets remain anchored to the
frozen initial reference frame (reference-mutating ALE modes remain unsupported).

**Phase 9b: B-prime pre-hydro reference-barrier retry:**

B-prime wires the Phase-9 reference-barrier rezone into the existing
driver-level full-step retry path as a bounded kill-test primitive. It is
default-off via `Numerics.ale.driver_retry_reference_barrier_enabled=false`, so
existing runs keep the default retry behavior and bitwise path.

On a typed Hydro2D mesh-quality soft failure, the driver classifies the failure
as AxisBand when any of these predicates holds: `first_failing_i <=
driver_retry_reference_barrier_K_axis` (default 4), the cell centroid satisfies
\(r/\sqrt{r^2+z^2} < driver_retry_reference_barrier_eta_axis\) (default 0.05),
or \(|z| < 2\Delta z\) with `first_failing_i <= 8`. The same bounded retry also
applies to a `multiblock_path_admissibility` failure when its path source is an
active fine child in the `POLAR_SHELL` block and the failing path metric is
`edge_cross`. For such failures, the driver restores the top-of-step snapshot,
builds the reference target mesh (`eulerian_initial` when
`reference_target="none"`), applies `reference_barrier_ALE` with conservative
remap, reduces the next retry dt by `min(chi*sigma_safe, q_retry)`, and retries
hydro before any hydro commit can reintroduce the same inversion.

The B-prime abort criteria are deliberately bounded: per-step reference-barrier
attempt cap (`driver_retry_reference_barrier_max_attempts`, default 6),
same-signature cap over `(i,j,stage,reason)` with a configurable cell window,
dt collapse below `driver_retry_reference_barrier_dt_collapse_rel` of the
failed dt, consecutive accepted-\(\lambda\) collapse below
`driver_retry_reference_barrier_lambda_collapse_threshold`, and quality-progress
stagnation. State counters record attempts, successes, lambda-collapse events,
and each abort class.

This is a bounded kill-test framework. If H3-B nr=256 does not reach `t_end`
with bounded rezone count and clean Sedov metrics, Phase 6
(`rz_logical="spherical_polar_halfplane"`) is the next consensus round per the
AI Round 6 verdict.

**Tier-A butterfly-center authority harness (geometry only):**

`tests/hydro/test_butterfly_authority_tier_a.cu` is a default-off verification
harness for the five-block half-butterfly topology. It does not run
Lagrangian hydro, conservative remap, force/work assembly, or BBS. The harness
constructs the existing `MULTIBLOCK_HALF_BUTTERFLY_5BLOCK` mesh, imposes
central-region compression targets \(C=\{1,1.25,1.5,2,3,4,6,8\}\) with
\(s=1/C\), and emits geometry-only reference meshes for four configurations:
axis-only PAVA, coherent full-patch scaled target without a barrier, coherent
full-patch scaled target with the diagonal-corner reference barrier, and an
existing `PolarCenterTreatment::TriFan` cap-quality prototype.

The full-patch authority target is built from the central core, all three fan
blocks, and the first four polar-shell rings. The R=0 PAVA chain remains a
Dirichlet constraint. Off-axis core nodes are fixed to the scaled
\(\gamma\)-MVP target \(X^R=sX^0\); fan and shell transition nodes are relaxed
with the existing cross-seam Winslow-style neighbor averaging and a tapered
outer shell boundary. This is an extension of the axis-rezone patch target
machinery, not a new optimizer.

The load-bearing metric is the origin-adjacent Cartesian cell
\((Y_0,Y_1,Y_2,Y_3)\) with
\[
Y_0=(0,0),\quad Y_1=(b,0),\quad Y_2=(c,c),\quad Y_3=(0,b).
\]
At the diagonal corner \(Y_2\), define
\[
e_1=Y_3-Y_2,\qquad e_2=Y_1-Y_2,\qquad
J_2=\det(e_1,e_2)=b(2c-b).
\]
The harness samples this oriented \(J_2\), not cell area or RZ volume, because
cell area can remain finite while \(2c-b\) crosses zero. The effective length
and shear coordinate are
\[
b_{\rm eff}=\frac12\left(|Y_1-Y_0|+|Y_3-Y_0|\right),\qquad
\psi_{\rm eff}=J_2/b_{\rm eff}.
\]
With target \(\psi_{\rm tar}=s\psi_0\), the emitted curves are
\[
P_\psi=\psi_{\rm eff}/\psi_{\rm tar},\qquad
S_2=\sigma_2(A_2)/\sigma_2(W_2),
\]
\[
q_2=\frac{2J_2}{|e_1|^2+|e_2|^2},\qquad
H_2=\frac{h_{\min}(C)}{s\,h_{\min}(1)}.
\]
Here \(A_2=[e_1\ e_2]\), \(W_2\) is the same corner matrix on the scaled
target, and \(h_{\min}\) is the planar quad min-altitude. The harness also
emits the central-core argmin cell for the minimum of
\((P_\psi,S_2,q_2,H_2)\).

The config-3 barrier line-search samples the same oriented \(J_2\) at
axis-touching central-core diagonal corners and separately enforces a lower
bound on \(b_{\rm eff}\). The length term prevents a candidate from satisfying
\(\psi=J_2/b_{\rm eff}\) by collapsing \(b_{\rm eff}\) rather than preserving
the keystone corner. The barrier floor used by the harness is 0.25 for
\(P_\psi\), \(S_2\), \(q_2\), \(H_2\), and the scaled \(b_{\rm eff}\) bound;
the ctest records RED/YELLOW/GREEN data rather than turning those empirical
colors into final production thresholds.

For the TRI-fan cap prototype, the harness builds the existing single-block
`tri_fan` center geometry with apex \(O\) pinned at the origin and evaluates
the first-ring triangles \(T_j=[O,X_{1,j},X_{1,j+1}]\). It emits
\[
Q_j=\frac{4\sqrt{3}\,A_j}{\ell_0^2+\ell_1^2+\ell_2^2}
\]
with positive signed RZ half-plane area \(A_j\). This prototype measures cap
geometry only; it is not a multiblock hydro graft.

**Phase 9: remap-damage gate (opt-in):**

Phase 9 adds a physics-aware acceptance gate to ALE backtracking. It is disabled
by default (`Numerics.ale.remap_damage_gate_enabled=false`), so legacy runs keep
the previous bit-exact remap-admissibility behavior.

For each trial mesh, define the per-face density-damage factor
\[
D_{\rho,f} =
\frac{|\Delta V_f|}{\min(V_L,V_R)}
\frac{|\rho_R-\rho_L|}{\rho_L+\rho_R+\rho_{\rm floor}},
\]
where \(\Delta V_f\) is the exact signed swept volume from §3.3.4 and \(L,R\)
are the adjacent cells. The scalar gate metric is
\[
D_{\rho,\max}=\max_f D_{\rho,f}.
\]

For each axis cell row \(j\), define the inward axis excess-mass estimate
\[
I_{0,j}=\max(0,-\Delta V_{1/2,j})\max(0,\rho_{1,j}-\rho_{0,j}),
\]
where \(\Delta V_{1/2,j}\) is the swept volume of the face between axis cell
\((0,j)\) and its outer neighbor \((1,j)\). The axis-specific scalar is
\[
A_{0,j} =
\frac{I_{0,j}}{M_{0,j}^{\rm base}+\rho_{\rm floor}V_{0,j}},
\]
with \(M_{0,j}^{\rm base}=M_{0,j}^{\rm init}\) when the axis budget state is
available, otherwise \(M_{0,j}^{\rm base}=\rho_{0,j}V_{0,j}\).

When `Numerics.ale.remap_damage_axis_budget_enabled=true`, accepted events add
their \(I_{0,j}\) into the cumulative budget
\[
B_{0,j}^{n+1}=B_{0,j}^{n}+I_{0,j}^{\rm accepted}.
\]
The budget cap is
\[
B_{0,j}^{\max} =
\texttt{remap\_damage\_axis\_budget\_factor}\,M_{0,j}^{\rm init}.
\]

A candidate mesh is rejected if any enabled criterion fails:
- \(D_{\rho,\max} > \texttt{remap\_damage\_dmax}\).
- \(\max_j A_{0,j} > \texttt{remap\_damage\_axis\_eta}\).
- `remap_damage_axis_budget_enabled=true` and
  \(B_{0,j}^{n}+I_{0,j} > B_{0,j}^{\max}\) for any \(j\).

Implementation references: `src/hydro/ale_remap.cuh`
(`compute_remap_damage`, `remap_damage_axis_budget_exceeded`) and
`src/hydro/ale_driver.cu` (trial-mesh acceptance/backtracking).

**Phase 9a: corner-J post-rezone admissibility gate:**

Before the Phase 9 damage gate, ALE backtracking evaluates each trial mesh with
two geometric admissibility checks. The legacy check samples the bilinear
Jacobian at the four \(2\times2\) Gauss points and reports the existing
`post_tangle` / `trial_quality` diagnostics. Phase 9a adds a parallel signed
corner-J check over active 2D cells:
\[
J_{c,k}^{corner} =
(\mathbf{x}_{k+1}-\mathbf{x}_k)\times(\mathbf{x}_{k-1}-\mathbf{x}_k),
\qquad k\in\{0,1,2,3\}.
\]
The candidate is rejected when any active-cell corner has
\(J_{c,k}^{corner}\le0\) or a non-finite value. This aligns ALE acceptance with
the Hydro2D corner-J admissibility predicate used by the pre-commit corrector
guard and retry active mesh repair. The check is controlled by
`Numerics.ale.corner_jacobian_post_tangle_enabled` (default `true`). Disabling
it restores the pre-Phase-9a Gauss-only post-tangle gate for diagnostic
comparisons.

Phase 2d-extension v6 Wave 4 adds a default-off strict-floor refinement to the
same post-tangle gate. When
`Numerics.ale.corner_post_tangle_strict_floor_enabled=true`, each active cell
also forms
\[
J_{\mathrm{floor},c} =
\max\left(k_J,\,
\texttt{Numerics.hydro.corner\_jacobian\_floor\_eps}
\max_{k\in\{0,1,2,3\}}\left|J_{c,k}^{corner}\right|\right),
\]
where \(k_J=10^{-30}\,\mathrm{cm^2}\) is the internal absolute Jacobian floor.
The candidate is then rejected when any finite corner satisfies
\(J_{c,k}^{corner}\le J_{\mathrm{floor},c}\), in addition to the existing
non-finite / non-positive checks. This unifies ALE candidate acceptance with
the strict corner-J floor used by the Phase 9c--9d local boundary repair
acceptance tests while preserving the legacy \(J>0\) predicate bit-exactly when
the new flag is false.

Telemetry distinguishes the two geometry failure modes: JSONL
`ale_backtrack_iter` records `post_tangle`, `post_corner_tangle`,
`min_corner_J`, and the first corner-J failing cell/corner, while the
`[ale-stats] backtrack_lambda_accepted` line reports Gauss, corner-J, and other
backtrack rejection counts.

Phase 2d-extension v6 Wave 4 also lands the I1-B high-drive physics-stack
anchor (`I1B_1D_RADIAL_GXII_5PCT`) as a deck-only validation gate. It introduces
no new equations, units, discretization, constants, or namelist parameters: the
gate reuses existing 1D_SPH Lagrangian hydro, 1D-supported laser raytrace,
grey multigroup FLD, electron heat conduction, and Qei coupling. The production
deck fixes the GXII 5% drive at \(1.8\times10^9\) erg over a 1 ns square pulse
and evaluates the same global matter+radiation+laser/radiation-escape residual
used by the 2D I1 harness, with 1D cell-centered kinetic energy. Empirical
Wave-4 result: `t_end_reached`, step 1000,
`energy_residual=2.75396e-3 <= 3.0e-3`.

**λ-sweep rejection diagnostic (default off):**

When `Numerics.ale.lambda_sweep_diagnostic_enabled=true`, the first Gauss/corner-J
backtrack rejection whose failing cell matches the configured target cell is
replayed on the host over the target cell and its active 1-ring neighbors. For
each sample,
\[
\mathbf{x}(\lambda)=\mathbf{x}^{old}+\lambda
(\mathbf{x}^{cand}-\mathbf{x}^{old}).
\]
The default sample list is
\[
\lambda \in
\{0,2^{-20},2^{-15},2^{-10},2^{-7},2^{-5},2^{-4},2^{-3},
2^{-2},2^{-1},1\},
\]
with `lambda_sweep_max_exp=N` replacing the leading \(2^{-20}\) term by
\(2^{-N}\) when requested. The diagnostic reports the minimum bilinear
Gauss-point Jacobian \(J\) [cm²], minimum signed corner-J [cm²], and minimum
signed RZ cell volume \(V_{RZ}\) [cm³] over the active stencil. A sample is
admissible only when all three minima are finite and positive. The classifier
is `lambda0_invalid`, `small_lambda_admissible`, `standard_lambda_admissible`,
or `no_admissible`. This path performs no mesh acceptance decision and is
bit-exact inert while the flag is false.

**Phase 9c: local boundary repair fallback (opt-in):**

When every backtracking lambda is rejected by the Phase 9a corner-J gate, the
optional `Numerics.ale.local_boundary_repair_enabled` fallback may attempt a
single-node analytic repair before ALE rolls back.  The same Phase 9c--9e
cascade may also be entered by the default-off
`Numerics.hydro.cascade_on_hydro_retry_enabled` diagnostic path when driver
full-step retry reports `corner_j`, the retry corner-balance predicate is bad,
and ALE backtracking accepts a candidate rather than exhausting. The fallback is
deliberately limited to non-corner nodes on the top/bottom \(z\) boundary or the
outer-\(r\) boundary. It preserves the boundary line: top/bottom boundary nodes
only move in \(r\), and outer-\(r\) boundary nodes only move in \(z\). Axis nodes
and domain corner nodes are not handled by this fallback.

For the failing corner node, the adjacent active cells define affine constraints
\(J_{c,k}^{corner}(s)\ge J_\mathrm{repair}\) in the single allowed coordinate
\(s\), where \(J_\mathrm{repair}\) is
`Numerics.hydro.corner_jacobian_floor_eps` times the largest absolute corner-J
in the adjacent active cells, floored by the internal absolute Jacobian floor.
This strict positive floor prevents a zero-motion repair from accepting a corner
that lies exactly on the \(J=0\) degeneracy boundary. The fallback intersects
those constraints with boundary-line ordering constraints, selects the closest
feasible coordinate, and then re-runs the Gauss-point, corner-J, and local
corner-J-floor checks. A repaired mesh is not accepted directly; it must still
pass the same axis-margin, remap-damage, first-sweep remap-admissibility, and
predictive-acceptance gates as an ordinary Winslow candidate. Telemetry is
reported as `[ale-stats] local_boundary_repositioning ...`. The default is
`false` because this path changes the candidate mesh rather than merely
hardening an existing admissibility check.

**Phase 9d: multi-node boundary repair fallback (opt-in):**

When Phase 9c fires but the single boundary-node feasible interval is empty,
`Numerics.ale.multi_node_boundary_repair_enabled` may escalate to a coordinated
1-ring repair. The moving set is the failing boundary apex, its two boundary
neighbors along the boundary line, and the immediate interior neighbor normal to
the boundary; if that minimal ring is infeasible, the two adjacent interior
neighbors are added. Boundary nodes keep the boundary-normal coordinate fixed,
while interior nodes may move in both \(r\) and \(z\). The implementation caps
the repair at nine scalar degrees of freedom.

Affected active cells are all cells touching one of the moving nodes. Each
affected cell contributes all four signed corner-J inequalities. Because the
exact corner-J is nonlinear when several moving corners share a cell, Phase 9d
uses a deterministic Picard-linearized half-space solve: finite-difference the
corner-J gradients about the current candidate, project onto violated
half-spaces until the linearized system is feasible, then re-evaluate the exact
corner-J constraints against the same positive \(J_\mathrm{repair}\) floor and
repeat for a small fixed number of Picard passes. The linearized half-spaces
target \(J_\mathrm{repair}\) plus a roundoff-sized clearance so that projection
roundoff does not leave the exact candidate infinitesimally below the strict
floor; final acceptance is still based on the exact \(J>J_\mathrm{repair}\)
check.
Radial/axial node-ordering inequalities are included in the same half-space
system. A candidate that passes the affected-cell exact constraints is then
written to the mesh and revalidated globally with the legacy Gauss-point check
and the Phase 9a corner-J check. Accepted candidates still pass through the
ordinary axis-margin, remap-damage, first-sweep remap-admissibility, and
predictive-acceptance gates before remap. Telemetry is emitted as
`[ale-stats] multi_node_boundary_repair ...`. The default is `false`.

**Phase 9e: emergency cell deactivation fallback (opt-in):**

Phase 9e is a last-resort active-cell removal path for a chronic degeneracy that
cannot be repaired by the standard ALE backtrack, Phase 9c single-node boundary
repair, or Phase 9d 1-ring multi-node repair. It is controlled by
`Numerics.ale.emergency_cell_deactivation_enabled` (default `false`) and only
fires when Phase 9c and Phase 9d are already enabled and the multi-node repair
returns an infeasible ring.

Before applying the deactivation, the driver rolls the mesh coordinates back to
the pre-rezone mesh. The failing active cell \(c_f\) is then merged into a
directly adjacent active recipient \(c_n\). Recipient selection filters out
inactive/void cells, prefers non-boundary cells, and then minimizes the local
volume-ratio mismatch among the four face-neighbor candidates. Let
\(m_{floor}=\rho_{floor}V_{c_f}\). The mass and internal-energy transfer are
\[
\Delta m = \max(m_{c_f}-m_{floor},0),
\qquad
\Delta E_e = \max(m_{c_f}e_{e,c_f}-m_{floor}e_{e,floor},0),
\qquad
\Delta E_i = \max(m_{c_f}e_{i,c_f}-m_{floor}e_{i,floor},0).
\]
The recipient receives \(\Delta m,\Delta E_e,\Delta E_i\), while \(c_f\) is
left at density and temperature floors and marked
`cell_is_void=1, hydro_active=0`. Positive floor mass/energy insertion is
reported through the existing `mass_floor_delta` and `E_floor_injected`
accounting, so energy-budget diagnostics treat the event like other floor
operations. Material fractions in the recipient are mass-weighted with the
transferred mass; the deactivated cell is reset to the configured void material
when one exists, otherwise all non-void material fractions are zeroed so the
driver void-mask refresh preserves `cell_is_void=1`. The driver also preserves
this active/void topology mask across hydro full-step retries and across later
ALE remaps; remapped material fractions are re-zeroed for already void cells so
the cell remains ignored by later active-cell geometry checks.

The operation is not an ALE remap. No swept-volume flux is applied after this
fallback: if it succeeds, `apply_ale()` returns after the mesh rollback and cell
deactivation. The guard then re-evaluates active-cell Gauss-point and corner-J
admissibility globally. If any other active cell remains tangled, all modified
cell fields and active/void flags are restored and the fallback reports a
cascading degeneracy. Telemetry is emitted as
`[ale-stats] emergency_cell_deactivation ...` and includes the target cell's
pre-transfer mass, volume, density, mass floor, and post-helper active/void
flags to distinguish a real mass merge from a geometry-only floor-mass
deactivation.

**Phase 9f: driver-requested local retry operators (default-off dispatch):**

Driver retry may request a local ALE operator through `AleMode` instead of the
scheduled ALE cadence.  These requests are produced only from typed
`RetryActionHint::Force*` values (§3.2.13b); the ordinary scheduled Winslow,
axis-spine, and boundary repair operators keep their legacy entry points and
outputs when no request is present.

`AxisSpinePlusLocal` first applies the existing axis-spine repair policy to the
axis band, then projects a local patch with \(i \in [0,N]\), where
\(N=\max(1,\texttt{Numerics.hydro.axis\_guard\_band\_cells})\) unless the
request supplies a larger patch radius.  Axis nodes keep \(r=0\).  Off-axis
nodes are constrained to positive, radially monotone \(r\), and all affected
cells must retain positive signed corner-J, area, and RZ volume.  The local
projection is deterministic: propose a bounded move toward the patch reference
mesh, project violated linearized ordering constraints, re-evaluate exact
corner-J constraints, and stop when the patch is admissible or the fixed
iteration budget is exhausted.

`BoundaryPatchProjection` uses the same patch projection machinery for
non-axis boundary failures.  The patch is centered on the failing cell with the
requested \(i/j\) radii.  Patch-boundary nodes are frozen; domain-boundary
nodes keep their boundary-normal coordinate fixed (`z` on top/bottom,
`r` on outer-r), so the repair is boundary-condition aware.

`CdLocalWinslow` smooths a frozen-boundary patch around an `InteriorCD` failing
cell.  Its monitor is
\[
M_c = 1 + \alpha S_{CD,c} +
      \beta \max(0,-\Delta t\,\nabla\cdot u_c),
\qquad \alpha=\beta=1,
\]
with the dimensionless \(S_{CD,c}\) from §3.2.13a and the cgs divergence
operator already used by Hydro.  The fixed maximum is 20 Picard iterations.
If the CD-local solve cannot admit the patch and
`Numerics.ale.multi_node_interior_repair_enabled=True`, the driver may escalate
to `InteriorMultiNodeProjection`; otherwise the request falls back to full
Winslow.

`InteriorMultiNodeProjection` generalizes the Phase 9d constraint engine from
boundary apex motion to all mobile interior nodes in the requested patch.
Patch-boundary nodes remain frozen, every affected cell contributes the four
corner-J inequalities, and the final candidate is accepted only after exact
local validation.  This mode is opt-in through
`Numerics.ale.multi_node_interior_repair_enabled=False` by default.

**Stage 22 Wave 2 repair escalation ladder**：forced retry requests from
`AxisSpinePlusLocal` and `BoundaryPatchProjection` use a guarded ladder on local
non-convergence:
\[
\text{requested local mode}
\rightarrow \texttt{InteriorMultiNodeProjection}
\rightarrow \texttt{FullWinslow}.
\]
The interior rung is attempted only when
`Numerics.ale.multi_node_interior_repair_enabled=True`.  Before attempting it,
the driver restores the pre-request coordinates saved before the requested local
mode.  The interior result is accepted only if
\[
\texttt{triggered}\land\texttt{converged}\land\neg\texttt{mesh\_tangle}
\land
q_{min}^{interior}+10^{-12}\ge q_{min}^{pre-request},
\]
where \(q_{min}\) is `RezoneResult.min_quality`.  Thus the ladder cannot
regress mesh quality relative to the genuine pre-request local state.  On gate
failure, the driver restores the same original coordinates again and runs
FullWinslow.  `CdLocalWinslow` is excluded from this driver-level ladder because
its implementation already has an internal
`InteriorMultiNodeProjection` escalation.

### Axis-Band Managed ALE Controller

#### Row-K Margin

The row-K margin quantifies the structural health of the band-to-main-mesh
boundary at row \(K\). For each axial column \(j \in [0,n_z)\):
\[
V_{\mathrm{actual}}(K-1,j) =
\pi\Delta z_j\left(r_K^2-r_{K-1}^2\right),
\]
\[
V_{\mathrm{target}}(K-1,j) =
\frac{\pi R_K(j)^2}{K}\Delta z_j,
\qquad
\mathrm{margin}(K,j) =
\frac{V_{\mathrm{actual}}(K-1,j)}
     {V_{\mathrm{target}}(K-1,j)}.
\]
Here \(r_i\) denotes the node-row radius averaged across the cell's two
z-faces, and \(\Delta z_j\) is the top-minus-bottom face separation at the
row-\(K-1\) mid-radius. The implementation evaluates
\(V_{\mathrm{actual}}\) with the same centroid-based signed 2D_RZ quadrilateral
volume formula used by mesh geometry, which reduces to the expression above for
orthogonal annular RZ cells. \(R_K(j)\) is the row-\(K\) radius averaged
across the two z-faces of column \(j\).

The row-K margin is
\[
\min_j \mathrm{margin}(K,j).
\]
The controller considers the band healthy when this exceeds
`Numerics.ale.axis_band_managed_remap_margin_trigger` (default \(10^{-4}\)).

For the K-selection policy, the controller scans \(K' \in
[K_{\mathrm{initial}},K_{\max}]\) and selects the smallest \(K'\) for which the
row-\(K'\) margin clears the threshold. If no \(K'\) clears, the controller
defers to the legacy halve-dt path (Wave 1 dispatcher).

### Band-Only Swept-Volume Remap

For the axis-band region (cells \(i \in [0,K-1]\), \(j \in [0,n_z-1]\)), the
controller remaps from the current Lagrangian mesh to the equal-RZ-volume
target mesh:
\[
r_i^{\mathrm{target}}(j) = R_K(j)\sqrt{i/K}, \qquad i \in [0,K],
\]
\[
z_i^{\mathrm{target}}(j) = z_K^{\mathrm{node}}(j).
\]
The band \(z\)-coordinates therefore inherit row-\(K\) node coordinates.

The remap is conservative for mass, radial and axial momentum, electron and ion
internal energy, and radiation group energies. The per-cell material
volume-fraction field is transported as a cell-mass-weighted passive scalar
(\(m\,f_m\)) and renormalized by the remapped cell mass on writeback; this
keeps \(\sum_m f_m = 1\) but does **not** independently conserve per-material
masses when materials of differing density share the band (documented
limitation; the per-material conservative remap is a separate roadmap item).
It uses a directionally split swept-volume scheme: an \(R\)-sweep followed by a
\(Z\)-sweep, with a van-Leer limiter applied to donor-cell intensives. Both
intermediate post-\(R\)-sweep cell volumes and final post-\(Z\)-sweep cell
volumes must be positive; non-positivity is a hard rollback gate. Remapped cell
mass and remapped electron internal energy must also remain positive. In 2T
mode the remapped ion internal energy must also remain positive; in 1T mode the
ion channel is not independently evolved and is required only to remain finite
and non-negative. In addition, every radiation group extensive energy
\(E_{g,c}^{\rm rad}\) must remain non-negative after each sweep; a single
negative group is a hard rollback gate even when the group sum stays positive
(AI review k04 R13).

On successful commit, cell-centered momentum is projected back to band nodes
with the same corner-mass weighting used by the 2D_RZ ALE velocity projection:
\[
u_n = \frac{\sum_{c \in C(n)} m_{c,n} u_c}{\sum_{c \in C(n)} m_{c,n}}.
\]
Here \(m_{c,n}\) is the RZ corner mass assigned from the remapped cell mass.
This is the Wave-2 first-cut choice because it preserves translation-invariant
kinetic energy and is consistent with the existing 2D_RZ ALE projection
convention.

Conservation diagnostics (relative deltas for mass, electron internal energy,
ion internal energy, node-centric kinetic energy, and radiation energy, plus
physical-scale-normalized band cell-momentum deltas
\(\epsilon_P = \Delta P / \max(|P_R|, |P_Z|, M_{\rm band}\,v_{\max},
{\rm tiny})\)) are reported through `AxisBandRemapResult`. Unit tests enforce
\(10^{-12}\) relative conservation for mass/internal/radiation energies and
\(10^{-10}\) for translation-invariant kinetic energy. At runtime the residuals
for mass, internal energies, radiation energy (when the band held radiation
energy before the remap), and band cell momentum are a **hard rollback gate**
at \(10^{-8}\) (AI review k04 R14): the band remap is a serial host
transaction, so these residuals are deterministic and roundoff-level unless a
genuine defect is present. The node-projection kinetic-energy delta remains a
warning-level diagnostic only, because the mass-weighted cell-to-node
projection is not kinetic-energy-conserving by construction.

Cell volumes are computed with the exact signed-polygon RZ volume formula
\[
V = \frac{\pi}{3}\sum_{k}(r_k + r_{k+1})(r_k z_{k+1} - r_{k+1} z_k),
\]
implemented in centered form (vertex coordinates shifted by the vertex-average
point before the shoelace accumulation). The vertex-average shift is a pure
translation of the origin and cancels identically over the closed polygon, so
this is **not** a Pappus/centroid approximation — the same exact primitive
(identical centered form) is used by `src/mesh/mesh.cu` cell geometry, the
band swept-face volumes, and the row-\(K\) margin definition above.

### Controller Decision Policy and Trigger Sites

The Axis-Band Managed ALE Controller is invoked from three trigger sites
in `Driver::run`, each gated on
`Numerics.ale.axis_band_managed_remap_enabled` (default False):

1. Pre-step (after primary dt selection): `evaluate_axis_band_need`
   inspects the current row-K margin via `select_axis_band_K`. When
   needed and remap_succeeded, dt is recomputed via the
   `post_axis_band_managed_remap` dt-lineage phase to reflect the new
   geometry.

2. Post-Strang hydro half-step (when
   `axis_band_managed_remap_every_hydro_half_step=True`): same controller
   invocation after each successful hydro callback. Substep dt is fixed
   by the operator splitter; no in-flight dt recompute.

3. Retry-time (in `classify_retry`'s RepairSameDt branch when feature on):
   the controller call is SCHEDULED via `pending_axis_band_remap` and
   EXECUTED only after `restore_driver_retry_snapshot`, ensuring the remap
   sees the rolled-back state, not the failed partial-step state.

The controller's K-fallback policy is K_initial -> K_initial+1 -> ... ->
K_max. On every K, the controller captures a localized snapshot, attempts
the band-only swept-volume remap, and either commits (on positivity
success) or restores and increments K (on positivity failure).

When all K in [K_initial, K_max] fail (AllKFailed), the controller
returns without modifying state and the caller falls through to the
Wave 1 legacy halve-dt path (no Path gamma scope creep).

Conservation diagnostics (mass / E_int_e / E_int_i / E_kin / E_rad
relative deltas) are emitted via `AxisBandResult`; tolerances of 1e-12
relative are enforced in unit tests. At runtime the mass / internal /
radiation / band cell-momentum residuals are a hard rollback gate at 1e-8
inside `apply_axis_band_remap` (`ConservationViolation`), and any negative
radiation group after a sweep is a hard rollback gate
(`RemappedRadiationNegative`); only the projection kinetic-energy delta
remains warning-only (see the band-remap section above, AI review k04
R13/R14).

Telemetry: when the controller commits a remap, the driver's
`stage24_axis_variational_projection_engaged_via` is set to
`"axis_band_managed_remap"` (first-set per step), distinct from
`"primary"` and `"escalation"` introduced by Wave 1's effective_mode_executed
plumbing. The pre-Wave-2 dispatcher state machine remains active when
the namelist flag is off.

**Phase 9b: predictive next-hydro acceptance gate (opt-in):**

Phase 9b adds an optional frozen-velocity look-ahead to the ALE backtracking
acceptance loop. It is disabled by default
(`Numerics.ale.predictive_acceptance_enabled=false`). When disabled, or when
the caller passes `dt_hydro_used <= 0`, the check is skipped and the previous
acceptance path is unchanged.

For a candidate ALE mesh \(x^{\rm cand}=(r^{\rm cand},z^{\rm cand})\), the
look-ahead uses the most recently consumed hydro time step \(\Delta t_{\rm h}\)
as a pessimistic proxy for the next hydro step and freezes nodal velocity:
\[
x_i^\star = x_i^{\rm cand} + \Delta t_{\rm h} u_i .
\]
Coordinates remain in cm, velocities in cm s\(^{-1}\), and
\(\Delta t_{\rm h}\) in s, preserving the cgs/eV unit system.

The axis predicate reuses the Phase 5 analytic axis margin. Let
\[
M_{\min}^{\rm cand} = \min_j M_{\rm axis}^{\rm cand}(j), \qquad
M_{\min}^{\star} = \min_j M_{\rm axis}^{\star}(j).
\]
The accepted candidate must satisfy
\[
M_{\min}^{\star} >
f_{\rm axis,pred} M_{\min}^{\rm cand},
\]
where \(f_{\rm axis,pred} =
\texttt{Numerics.ale.predictive\_acceptance\_axis\_floor\_fraction}\).
The default \(f_{\rm axis,pred}=0\) means the look-ahead only rejects
non-positive predicted axis margin.

The cell-volume predicate computes direct RZ cell volumes from the candidate
geometry and from the look-ahead geometry:
\[
V_c^\star = V_c(x^\star), \qquad V_c^{\rm cand}=V_c(x^{\rm cand}).
\]
For every cell,
\[
V_c^\star >
f_{V,pred} V_c^{\rm cand},
\]
with \(f_{V,pred} =
\texttt{Numerics.ale.predictive\_acceptance\_cell\_vol\_floor\_fraction}\).
The default \(f_{V,pred}=0\) rejects only non-positive predicted cell volumes.

The gate runs after first-sweep remap admissibility succeeds and before a
backtracking trial is accepted. Rejections are counted separately as axis-margin
and cell-volume predictive rejects; logging is rate-limited and does not add
history or HDF5 schema fields.

**Phase 10: minimal axis-spine repair (opt-in):**

Phase 10 adds `Numerics.ale.axis_repair_mode="axis_spine_only"` as an
axis-only \(Z\)-projection alternative to the full Winslow path. The default
remains `"full_winslow"` for legacy behavior.

The minimal repair leaves \(R=0\) on the axis and updates only axis-node \(Z\)
coordinates toward the adjacent \(i=1\) row:
\[
z_{0,j}^{new} = \alpha z_{0,j} + (1-\alpha)z_{1,j}.
\]
It performs repeated sweeps with default \(\alpha=0.5\), stopping when
\[
\max_j |z_{0,j}^{new}-z_{0,j}| < 10^{-12}
\]
or after 10 sweeps. This is deliberately narrower than the full Winslow
rezoner: it repairs the axis spine without changing the non-axis mesh through a
global elliptic solve.

Implementation reference: `src/hydro/ale_rezone.cuh::minimal_axis_z_repair`.
`axis_repair_mode="axis_z_winslow"` is reserved for the Phase 8a axis-Z Winslow
path and currently falls back to `"full_winslow"` with a warning.

**Phase 11: MS2 monotonic conservative remap (opt-in):**

Phase 11 adds `Numerics.ale.remap_scheme="ms2_moments"` as an opt-in
second-order monotonic conservative remap following the Margolin-Shashkov
second-order moment remap idea (Margolin and Shashkov, JCP 2003). The default
remains `"legacy_split"`, the first-order-compatible split remap described in
§3.3.4.

For each cell-centered field, the MS2 path reconstructs a linear state
\[
q(\mathbf{x}) = q_L + \nabla q_L\cdot(\mathbf{x}-\mathbf{x}_L)
\]
with a 4-neighbor least-squares slope on the R-Z cell-center stencil. The slope
is limited by `Numerics.ale.remap_ms2_limiter`, either `"van_leer"` or
`"barth_jespersen"`, to prevent new local extrema.

For `"van_leer"`, each face midpoint \(f\) with neighbor cell \(N_f\) uses
\[
r_f = \frac{q_{N_f}-q_L}{q_f^{unlim}-q_L},\qquad
\psi_f = \frac{r_f+|r_f|}{1+|r_f|},
\]
with zero-gradient exterior data at mesh boundaries and \(\psi_f=1\) when
\(q_f^{unlim}=q_L\). The cell factor is
\(\psi_L=\min_f \psi_f\), clamped to \([0,1]\), and the limited slope is
\(\psi_L\nabla q_L\). The clamp prevents the limiter from amplifying the
least-squares slope; `"van_leer"` remains the default MS2 limiter.

For `"barth_jespersen"`, the local stencil extrema are
\[
q_{\min,L}=\min(q_L,q_N),\qquad q_{\max,L}=\max(q_L,q_N),
\]
over the four face-neighbor cells. Each unlimited face value is bounded by
\[
\psi_f =
\begin{cases}
(q_{\max,L}-q_L)/(q_f^{unlim}-q_L), & q_f^{unlim}>q_{\max,L},\\
(q_{\min,L}-q_L)/(q_f^{unlim}-q_L), & q_f^{unlim}<q_{\min,L},\\
1, & \text{otherwise},
\end{cases}
\]
and \(\psi_L=\min_f\psi_f\), clamped to \([0,1]\), multiplies the
least-squares slope.

For a swept polygon \(P_f\), the donor cell and stored face-flux sign follow
§3.3.4 **Donor convention (post-2026-05-11 fix)**. The donor reconstruction
integral used by the MS2 path is
\[
I_f = \rho_L |P_f| +
\nabla\rho_L\cdot\left(M_1(P_f)-\mathbf{x}_L |P_f|\right),
\]
where \(|P_f|\) is the R-Z axisymmetric swept volume and \(M_1(P_f)\) is its
exact first moment. The first moments use the T5 exact R-Z polygon routines
`rz_signed_quad_moment_r` and `rz_signed_quad_moment_z`, so the second-order
correction is geometrically consistent with the exact swept-volume formula.

Implementation reference: `src/hydro/ale_remap.cuh::launch_remap_strang_ms2`.

**Phase 12: ALE diagnostic-compatible kinetic-energy closure:**

The conservative ALE remap transports cell-centered momentum and internal
energy densities. Kinetic energy in the 2D Lagrangian update, however, is
diagnostic-compatible only when measured with nodal velocities and the
subzonal corner masses of §3.2.4. The 2D energy budget diagnostic always
reports kinetic energy with this corner-mass nodal measure. When
`Numerics.ale.ke_conservation_closure=true`, the ALE driver closes the
remap/projection kinetic-energy defect with that same measure, after node
projection and before EOS reclosure.

Before remap, each cell stores the diagnostic-compatible kinetic energy
\[
K_c^{diag,n} =
\frac{1}{2}\sum_{\alpha=0}^{3}
m_{c\alpha}^{n}|v_{n(c,\alpha)}^{n}|^2 ,
\]
where the corner layout is \((n_{00},n_{10},n_{11},n_{01})\). The remap scalar is
the energy density
\[
q_{K,c}^{n}=K_c^{diag,n}/V_c^{n},
\]
which is conservatively remapped by the same Strang/MS2 scalar remap used for
\(\rho e_e\) and \(\rho e_i\). After remap,
\[
K_c^{remap,*}=q_{K,c}^{*}V_c^{*}.
\]

With the closure enabled, the cell momentum prepared for remap is also built
from the old corner-mass nodal momentum,
\[
\mathbf{u}_c^{n} =
\frac{\sum_\alpha m_{c\alpha}^{n}\mathbf{v}_{n(c,\alpha)}^{n}}
     {\sum_\alpha m_{c\alpha}^{n}},
\qquad
(\rho\mathbf{u})_c^{n}=\rho_c^{n}\mathbf{u}_c^{n}.
\]
After momentum remap, the cell velocity \(\mathbf{u}_c^*\) is projected back to
nodes with the current-mesh corner masses and the existing velocity boundary
condition logic:
\[
\mathbf{v}_v^{n+1} =
\frac{\sum_{c\in\mathcal{N}(v)}m_{cv}^{*}\mathbf{u}_c^*}
     {\sum_{c\in\mathcal{N}(v)}m_{cv}^{*}} .
\]

The actual post-projection per-cell nodal kinetic energy is then
\[
K_c^{node,*} =
\frac{1}{2}\sum_{\alpha=0}^{3}
m_{c\alpha}^{*}|v_{n(c,\alpha)}^{n+1,postBC}|^2 .
\]
The local closure deposit is
\[
\Delta I_c = K_c^{remap,*}-K_c^{node,*}.
\]
It is added only to material internal energy; mass, remapped momentum, mesh
geometry, radiation energy, and volume fractions are not changed by the
deposit.

For 1T runs, \(\Delta I_c/m_c^*\) is added to the total material specific
energy stored in \(e_e\) (with \(e_i=0\) under the 1T closure). For 2T runs, the
split uses the raw remapped internal-energy fractions:
\[
f_e=\frac{e_{e,c}^{raw}}{e_{e,c}^{raw}+e_{i,c}^{raw}},\qquad
\Delta e_e=f_e\frac{\Delta I_c}{m_c^*},\qquad
\Delta e_i=(1-f_e)\frac{\Delta I_c}{m_c^*}.
\]
If the denominator is zero for a positive deposit, \(f_e=1/2\). For a negative
deposit, the default path clips removal to the available nonnegative internal
energy \(m_c^*(e_e^{raw}+e_i^{raw})\), preserving \(e_e,e_i\ge0\). This is the
legacy behavior used when `Numerics.ale.ke_closure_redistribute_floor=false`.

When `Numerics.ale.ke_closure_redistribute_floor=true`, the closure instead
uses a conservative two-pass positivity correction. First compute the
unclipped tentative species energies
\[
\tilde e_{e,c}=e_{e,c}^{raw}+f_e\frac{\Delta I_c}{m_c^*},\qquad
\tilde e_{i,c}=e_{i,c}^{raw}+(1-f_e)\frac{\Delta I_c}{m_c^*},
\]
with the 1T path using only the \(e_e\) equation. The closure positivity floor
for this redistribution is zero specific internal energy; the thermal
\(T_{floor}\) clamp remains owned by the following EOS reclosure. For each
active species \(k\in\{e,i\}\),
\[
D_{k,c}=m_c^*\max(0,-\tilde e_{k,c}),\qquad
C_{k,c}=m_c^*\max(0,\tilde e_{k,c}),
\]
and \(D_{tot}=\sum_{c,k}D_{k,c}\), \(C_{tot}=\sum_{c,k}C_{k,c}\) are reduced
globally across MPI ranks. If \(C_{tot}\ge D_{tot}\), set
\(a=D_{tot}/C_{tot}\) (or \(a=0\) when both totals are zero). Deficit species
are set to zero; capacity species are reduced by their proportional share:
\[
e_{k,c}^{n+1} =
\begin{cases}
0, & D_{k,c}>0,\\
\tilde e_{k,c}-a\,C_{k,c}/m_c^*, & D_{k,c}=0.
\end{cases}
\]
This preserves \(\sum_c m_c(e_{e,c}+e_{i,c})\) with the raw closure target
\(\sum_c\Delta I_c\) while enforcing \(e_e,e_i\ge0\). If \(C_{tot}<D_{tot}\),
all available capacity is removed (\(a=1\)); the remaining
\(E_{redistribution\_unresolved}=D_{tot}-C_{tot}\) is recorded as an artificial
budget term rather than silently appearing as floor injection. EOS reclosure
immediately follows the deposit, so \(T_e,T_i,P_e,P_i\) remain consistent with
the updated internal energies.

The flag defaults to false. Therefore existing ALE runs, including production
Phase 9--11 configurations, remain bit-exact unless the closure is explicitly
enabled.

**I1 Phase 1/T4: 2D_RZ total-material-energy remap (default-off):**

`Numerics.hydro.total_energy_remap_2d_rz=true` changes only the 2D_RZ
conservative ALE remap recovery path in `src/hydro/ale_remap_2d_rz.cu`. The
default value is false, which selects the legacy separate \(e_e/e_i\) remap
kernels. In the CSR multiblock implementation the flag is supported for
`topology_scheme="multiblock_half_butterfly_trifan_cap_5block"`; unsupported
CSR topologies still reject it.

Before remap, each cell forms the extensive total material energy
\[
E_{tot,c}^{mat}=m_c(e_{e,c}+e_{i,c})+K_c^{node},\qquad
K_c^{node}=\frac{1}{2}\sum_{\alpha=0}^{n_c-1}
m_{c\alpha}|v_{n(c,\alpha)}|^2 ,
\]
where \(n_c\in\{3,4\}\).  The corner order is the active topology order
\((n_{00},n_{10},n_{11},n_{01})\) for quads and slots \(\{0,1,2\}\) for cap
triangles.  The \(m_{c\alpha}\) are the exact RZ corner masses used by
`csr_corner_kinetic_for_cell`; they use Pappus/RZ subcell mass weighting and do
not divide by \(r\), so pinned apex corners carry zero kinetic energy when their
corner mass vanishes.  The remapped conserved hydro quantities are \(m\),
\(m u_r\), \(m u_z\), \(E_{tot}^{mat}\), and the bounded internal-energy split
mass \(mY_e^{int}\), with
\[
Y_e^{int}=\mathrm{clip}\left(\frac{e_e}{e_e+e_i},0,1\right).
\]
For the second-order van-Leer CSR remap, these hydro extensive quantities use
one donor face state and one limiter coefficient so that mass, momentum, total
energy, and split tracer are transported consistently.

The CSR donor convention is selected by the effective fixed-convention flag
`Numerics.ale.swept_volume_sign_fixed || total_energy_remap_2d_rz`, with the
multiblock center-patch driver forcing `swept_volume_sign_fixed=true` in the
temporary remap config for that opt-in path.  When the effective flag is false,
the legacy CSR donor/flux path is retained for bit-identical existing decks.
When the effective flag is true, an outgoing swept volume uses the losing cell as
donor.  A conservative positivity limiter then computes, for each losing cell, a
common scale on its outgoing hydro face fluxes so that the remapped mass cannot
fall below
\(m_{floor}=\rho_{floor}V^*\).  The same face scale is applied to the tied hydro
extensive fluxes \(m\), \(m u_r\), \(m u_z\), \(E_{tot}^{mat}\), and
\(mY_e^{int}\).  Internal faces still add equal-and-opposite fluxes to the two
adjacent cells, so the limiter preserves the global hydro conservation identity.
If `swept_volume_sign_fixed=true` is used without total-energy remap, the same
mass-flux scale applies to the legacy \(m e_e\) and \(m e_i\) extensive hydro
fluxes instead of \(E_{tot}^{mat}\) and \(mY_e^{int}\).

After remap, the cell momentum is converted to cell velocity and projected back
to nodes.  In the total-energy CSR branch this projection uses the same RZ
corner masses as \(K_c^{node}\), rather than the legacy equal
`mass/nverts` weights.  Boundary constraints are applied, and then a
KE-realizability limiter makes the nodal velocity field compatible with the
remapped total energy.  For each cell,
\[
K_{max,c}=E_{tot,c}^{mat,*}
          -m_c^*(e_{e,floor}+e_{i,floor}),
\qquad
s_c=\min\left(1,\sqrt{K_{max,c}/K_c^{node,*}}\right),
\]
with \(s_c=0\) if \(K_{max,c}\le0\) or non-finite.  Each node receives the
minimum \(s_c\) over incident cells from the reverse cell-node CSR, and its
velocity is scaled by that value before recovery.  The recovery then recomputes
the actual post-limiter \(K_c^{node,*}\) and sets
\[
e_{int,c}^{*}=\frac{E_{tot,c}^{mat,*}-K_c^{node,*}}{m_c^*},
\qquad
e_e^*=Y_e^{int,*}e_{int,c}^{*},\quad
e_i^*=(1-Y_e^{int,*})e_{int,c}^{*}.
\]
Thus removed projection kinetic energy remains in the cell internal-energy
balance instead of being hidden as floor injection.  Any remaining electron/ion
temperature-floor injection is still accumulated in `E_floor_injected` and
reported explicitly; mass-floor closure remains accounted separately through
`mass_floor_delta` and `E_redistribution_unresolved`.

Verification on the I1-B finite-CR shell discriminator
(`TENRYU_I1B_DISC_TOTAL_ENERGY_REMAP=1`, Case B, \(n_z=64\), 1500 steps)
showed no coherent kinetic-energy runaway, zero near-vacuum hits, negligible
total-energy recovery floor injection (\(2.18\times10^{-3}\) erg), and no
energy-budget-exceeded warnings.  A residual free-stream momentum-GCL drift was
characterized in this aggressive convergent deck (`max_vol_closure=76.9` at step 1500,
with mass-GCL \(9.6\times10^{-11}\)); the cap seam-GCL verification gate remains
passing, and this residual is not yet claimed resolved for production C72/ladder
decks.

**I1 Fix B discriminator: 2D_RZ work-split audit (default-off):**

`Numerics.hydro.work_split_audit_2d_rz=true` enables a diagnostic-only CSV
written from the 2D_RZ two-temperature Hydro energy update. It does not change
the pressure force, artificial viscosity, energy update, remap, EOS closure, or
state arrays. The CSV path is
`${Output.directory}/diag/work_split_audit_2d_rz_rank%04d.csv`.

For each cell at the live 2T energy-update site, the audit reconstructs the
work split
\[
W_{Pe,c}=-P_{e,c}^{1/2}\Delta V_c,\qquad
W_{Pi,c}=-P_{i,c}^{1/2}\Delta V_c,\qquad
W_{Q,c}=-Q_c^{1/2}\Delta V_c ,
\]
matching the implemented update
\[
\Delta e_i=-P_i^{1/2}\Delta V/m + Q_{ei} + W_Q/m,\qquad
\Delta e_e=-P_e^{1/2}\Delta V/m - Q_{ei}
\]
for the default `av_heat_to="ion"` route. The reported \(Q_{ei}\) is inferred
from the electron/ion energy deltas and the same \(P\,dV/Q\,dV\) terms, so floor
events remain visible as residuals.

The shock-window diagnostic is row-local in z. It forms
\[
\chi_c=\max(0,-\nabla\cdot u_c)\,H(\Delta\rho>0)\,H(\Delta P>0)\,H(\Delta T_e>0),
\]
selects the contiguous positive-\(\chi\) layer around the row maximum, excludes
end guard cells, marks the upstream half of that layer, and marks the first
1-3 high-density downstream target cells. Two non-mutating discriminators are
emitted:
\[
T_{e,noQ}=T_e-\frac{\sum^t W_Q}{m(c_{v,e}+c_{v,i})},
\qquad
T_{iso}=T_{up}\left(\frac{\rho}{\rho_{up}}\right)^{\gamma-1}.
\]
The upstream \(T_{up},\rho_{up}\) are inferred per row from cold cells adjacent
to the detected layer; the I1 constants \(T_{up}=30\,\mathrm{eV}\),
\(\rho_{up}=2\,\mathrm{g\,cm^{-3}}\) are also emitted as a comparison. Per-step
summary rows are written for the mid-radius row and the radial sum. Per-cell
rows are throttled to plot/final cadence by default, with
`work_split_audit_cell_every_n_steps` and `work_split_audit_all_rows` available
for focused runs.

**I1 Fix A: experimental 2D_RZ z-HLLC flux (default-off):**

`Numerics.hydro.hllc_z_flux_2d_rz=true` selects an experimental quasi-1D
Eulerian z operator for stationary z-normal shocks. It is not an added
viscosity. When enabled, the HLLC path owns z transport and the driver skips the
legacy staggered z Lagrange plus conservative reference-remap transport for that
hydro substep. The r direction is not promoted to a full 2D Godunov update in
this mode.

The conservative cell state advanced through logical z faces is
\[
U_c=\left(m_c,\; (m u_z)_c,\; E_c^{mat},\; m_cY_{e,c}^{int}\right),
\qquad
E_c^{mat}=m_c(e_{e,c}+e_{i,c})+K_c^{node}.
\]
The z momentum \((m u_z)_c\) is stored in the optional
`State.hllc_mom_z_cell` field and is authoritative while the flag is enabled.
It is initialized from corner-mass nodal velocities only when the field is
missing or a legacy checkpoint lacks `/hydro/hllc_mom_z_cell`; normal HLLC steps
do not reconstruct cell velocity from projected nodes. This avoids the
cell-to-node-to-cell averaging loop that would broaden a one-cell stationary
shock.

At each z face the code forms left/right primitive states from the authoritative
cell momentum and pressure \(P_e+P_i\), then evaluates the contact-resolving HLLC
flux with speeds \(S_L,S_M,S_R\). Non-finite or non-positive star states fall
back to HLLE only when `hllc_z_flux_hlle_fallback=true`, and fallback counts are
audited. The RZ z-face area is
\[
A_f=\pi |r_{i+1/2,outer}^2-r_{i+1/2,inner}^2|,
\]
so no additional \(2\pi r\) factor is applied.

After the conservative update, the cell momentum is projected one-way to nodal
`v_z` for compatibility with the staggered state, output, and the total-energy
recovery. The next HLLC primitive state still reads `hllc_mom_z_cell`, not this
projected nodal velocity. Internal energy recovery reuses the same
corner-mass/nodal-kinetic measure as `total_energy_remap_2d_rz`:
\[
e_{int,c}=\frac{E_c^{mat,*}-K_c^{node,*}}{m_c^*},\qquad
e_e^*=Y_e^{int,*}e_{int,c},\quad e_i^*=(1-Y_e^{int,*})e_{int,c}.
\]
The path requires `total_energy_remap_2d_rz=true` so this discrete energy measure
is explicit in the configuration.

`Numerics.hydro.hllc_z_flux_audit_2d_rz=true` writes
`${Output.directory}/diag/hllc_z_flux_2d_rz_rank%04d.csv` with Te shock width
(\(32\le T_e\le60\) eV, primary I1 metric), density 10-90 width, HLLC/HLLE
fallback counts, boundary flux samples, floor counts, and a cell-vs-node
momentum residual. The path is experimental and intended for I1/H1 quasi-1D
validation; H2/H3 flag-on runs are smoke tests until a full 2D Godunov design is
agreed.
The I1 production-grid closure for this opt-in path is documented in
`docs/validation/2d_rz/I1/closure_summary.md`; the legacy VNR/ALE default
remains byte-identical with the flag off.

**5-block half-butterfly axis ALE target primitive (target-only):**

For `topology_scheme="multiblock_half_butterfly_5block"`, the axis ALE target
primitive builds the full physical \(R=0\) node chain from the multiblock shared
node identity and `NODE_AXIS`/`NODE_POLE_AXIS` flags. The ordered chain
\(\mathcal{A}=(n_0,\ldots,n_{N-1})\) is sorted by increasing physical \(Z\) and
spans the south pole/shell axis, south fan axis, the central-core interior axis,
the shared core-fan seam nodes at \(Z=\pm r_c\), the north fan axis, and the
north shell/pole axis.  The seam entries are physical shared nodes, not
duplicated block-local endpoints.  The node mass used by the target solve is
the lumped incident corner-mass sum
\[
m_i=\sum_{(c,k):\,n(c,k)=n_i} m_{c,k},
\]
using the reverse cell-node CSR. Shared seam nodes therefore include both core
and fan incident corner-mass contributions. Single-block and the legacy
three-block multiblock topology return an inactive empty chain.

Given predicted post-Lagrangian axis coordinates \(\tilde Z_i\), initial edge
lengths \(\ell_i^0\), adjacent first-ring cell areas \(A_{\mathrm{adj},i}\),
and a caller-provided spacing-floor fraction \(\eta_{\rm floor}\), the minimum
allowed spacing is
\[
\ell_{\min,i}=\eta_{\rm floor}\min(\ell_i^0,\sqrt{A_{\mathrm{adj},i}}).
\]
\(\eta_{\rm floor}\) defines the enforced spacing floor. It is distinct from an
ALE trigger fraction, which is a driver policy and not part of this primitive.

The target-only projection solves
\[
\min_{Z^*}\sum_i {m_i\over 2\Delta t^2}(Z_i^*-\tilde Z_i)^2
\quad\text{subject to}\quad
Z_{i+1}^*-Z_i^*\ge\ell_{\min,i},\quad R_i^*=0.
\]
Let \(s_0=0\), \(s_i=\sum_{k<i}\ell_{\min,k}\), and
\(y_i=\tilde Z_i-s_i\). The constraints become ordinary monotonicity of
\(\hat y_i\), so TENRYU applies deterministic weighted PAVA with weights
\(m_i\) to \(y_i\), then back-substitutes
\[
Z_i^*=\hat y_i+s_i.
\]
This is exact for the stated one-dimensional quadratic projection and is
\(O(N)\). The active PAVA chain contains only finite positive-mass axis nodes.
The production driver first zeroes corner masses from void or hydro-inactive
cells and clips active-cell tiny negative roundoff to zero; active-cell
non-finite or materially negative corner masses remain assertion failures.
Axis nodes incident only to zeroed dormant cells have \(m_i=0\), are excluded
from the PAVA degree-of-freedom list, and retain their input \(Z_i=\tilde Z_i\)
as fixed no-motion axis entries for downstream patch construction. The
primitive returns \(R_i^*=0,Z_i^*\) for the full physical axis-node list, but
only positive-mass nodes participate in the quadratic solve. It does not modify
mesh coordinates, state, velocity, force, pressure, or energy fields. The
conservative remap consumer is responsible for installing any accepted target.
For the tri-fan-cap polar shell, the physical axis list is ordered
topologically rather than by the current \(Z\) coordinate: south-pole shell
rows run from the outer row inward, then the cap origin, then north-pole shell
rows run from the inner row outward.  This keeps the PAVA constraint aligned
with the logical polar-shell radial coordinate even if a current mesh has
already inverted two pole rows.  After the PAVA solve, the installed target is
projected on the two polar-shell pole columns from the outer physical boundary
inward so that
\[
|s^*_{q,pole}|+\ell_{\min,q}^{pole}\le |s^*_{q+1,pole}|,
\]
with the same spacing-area floor where the corresponding axis-chain edge is
available and a roundoff-scale positive fallback otherwise. This projection is
target-only; the conservative swept-volume and total-energy remap sees the
resulting node motion before any geometry commit.

Driver policy for `Numerics.ale.axis_rezone_enabled=True` is default-off and
restricted to five-block half-butterfly meshes:
`multiblock_half_butterfly_5block` and
`multiblock_half_butterfly_trifan_cap_5block`. On each multiblock ALE driver
opportunity after the Lagrangian corrector, TENRYU recomputes the full-axis
chain and the current minimum on-axis edge length \(\ell_{\min}^{axis}\) plus
the minimum active-polygon altitude over cells incident to the chain
\(h_{\min}^{axis}\), while skipping void or hydro-inactive cells in the
incident-cell metric scan. If zero-mass exclusions leave no active adjacent
cell for a consecutive positive-mass PAVA node pair, the edge's spacing-area
scale falls back to its own initial edge length squared. The first enabled
evaluation caches
\(\ell_{\min,0}^{axis}\), \(h_{\min,0}^{axis}\), the initial chain edge
lengths, and adjacent first-ring cell areas. A target-only axis rezone fires at
most once per hydro step when
\[
{\ell_{\min}^{axis}\over \ell_{\min,0}^{axis}} <
\texttt{axis\_rezone\_trigger\_edge\_fraction}
\quad\text{or}\quad
{h_{\min}^{axis}\over h_{\min,0}^{axis}} <
\texttt{axis\_rezone\_trigger\_min\_altitude\_fraction}.
\]
This trigger is repeatable across steps because the current ratios are
recomputed every driver opportunity, while the cached reference values remain
the first activated values. The driver installs \(R^*=0,Z^*\) only in the ALE
target field; all non-axis target nodes remain at their current Lagrangian
coordinates.

*Pole-sector angular rezone* (namelist
`Numerics.ale.pole_sector_rezone_enabled`, default off; the historical
`TENRYU_I1B_POLE_REZONE*` environment variables override the namelist keys
when SET non-empty). When the axis rezone fires, per pole the first
`pole_sector_rezone_m_theta` (default 4, validated \(\ge 2\), additionally
capped at \(n_\theta/4\); env `TENRYU_I1B_POLE_REZONE_M`) off-axis node
columns of every active polar-shell node row are additionally retargeted onto
a reference angular ladder anchored at the row's \(a=M\) column, preserving
each node's spherical radius — an ANGULAR rezone addressing the tangential
null mode the compression AV cannot see. Two ladders
(`pole_sector_rezone_mode`, default `"uniform"`; env
`TENRYU_I1B_POLE_REZONE_MODE`): `"uniform"`,
\(\delta_{\rm ref}(a)=(a/M)\,\delta_M\), restores the initial uniform-theta
pole zoning and is the identity on a healthy mesh; `"equal_mu"`,
\(\delta_{\rm ref}(a)=\arccos\!\big(1-(a/M)(1-\cos\delta_M)\big)\), the
axisymmetric-volume-fraction ladder — a large restructuring relative to the
initial zoning that churns the remap when applied continuously. Node targets
blend by `pole_sector_rezone_lambda` (default 0.5, range \((0,1]\); env
`TENRYU_I1B_POLE_REZONE_LAMBDA`); nodes within
`pole_sector_rezone_deadband_frac` (default 0.0 = no deadband, range
\([0,1)\); env `TENRYU_I1B_POLE_REZONE_DEADBAND_FRAC`) times \(\delta_M\) of
their reference angle are left untouched. Rows interior to the central macro
cell are skipped (virtual members). The pole-sector target rides the SAME
transactional guard and conservative remap as the axis-chain target. HONEST
STATUS: the nr16 Case B discriminating runs (commit 5df5635e) found this knob
NET-NEGATIVE in that regime — pole-local remap activity degrades the pure-gas
row tracer budget (the ring-absorption survival resource) faster than the
de-shearing pays back — so it stays default-off as an available robustness
lever, not a recommendation.

*Pole tangential damper* (experimental, env `TENRYU_I1B_POLE_DAMPER`, default
off). The pole-column angular-shear mode is a radially ALTERNATING tangential
slip of adjacent polar-shell node rows within the same near-pole angular
column; it is AV-blind (\(\nabla\!\cdot u\approx 0\) along the slip) and was
adjudicated locally self-sustaining in the dense shell (Q2 outer-freeze
discriminator). The damper adds, on every radial pole-column node pair
\((q,k)\)–\((q{+}1,k)\) with column \(a\in[1,M)\) of either pole
(`TENRYU_I1B_POLE_DAMPER_M`, default 4; \(a=0\) is the axis column itself,
excluded), the pairwise tangential drag
\[
F = -\,C_\theta\,\rho_e\,c_{s,e}\,|S_{\rm edge}|\,(\Delta u\cdot\hat t)\,\hat t,
\qquad \Delta u = u_{q+1}-u_q,
\]
with \(\hat t\) the in-plane unit normal of the radial edge,
\(\rho_e, c_{s,e}\) the harmonic means of the two angular-neighbor cells, and
\(|S_{\rm edge}|=\pi(r_0{+}r_1)\,h\) the SAME RZ edge-area convention as the
CSW AV svec (\(C_\theta\): `TENRYU_I1B_POLE_DAMPER_CTHETA`, default 0.05).
The force is accumulated into the CSW edge-AV force buffers in the canonical
cell-\(a\) face orientation, so the momentum kick (\(-F\) at \(n_0\), \(+F\)
at \(n_1\), pairwise momentum-conserving by construction) and the corrector's
SIGNED time-centered work recompute inherit the exact compatible total-energy
closure unchanged; the assembly-stage estimate deposits
\(w=+C_\theta\rho_e c_{s,e}|S|(\Delta u\cdot\hat t)^2\ge 0\) split evenly
between the two adjacent cells. Rows interior to the central macro cell are
excluded. Explicit-stability margin is \(\sim 1/C_\theta\) acoustic steps, so
no additional dt limiter is installed at the small default \(C_\theta\).

*Transactional rezone guard.* Before the axis-rezone target (including the
convergent-locality patch, the pole-sector angular-rezone patch, and
core-freeze restores) is committed, the driver
evaluates the SAME path-admissibility used for Lagrangian trials (all cells
via the CSR kernel, inactive macro members excluded, plus the macro-boundary
polygon volume and simple-loop checks) along the straight node paths from the
pre-rezone positions to the installed target. On failure the target is
blended toward the pre-rezone positions (\(x^{R}\!\leftarrow\!x^{old} +
\tfrac12(x^{R}-x^{old})\), up to four halvings); if no fraction is admissible
the rezone is skipped for that step and the persistent reference is restored
— an inadmissible rezone is rejected, never committed. The same
reject-not-die semantics applies to the multiblock center patch: when the
patch linesearch accepts \(\sigma=0\), the \(\sigma=0\) quality evaluation
fails its floors, and no legal ring-absorption repair remains (e.g. the
failing cell is in the material shell), the patch is skipped for that step
with the identity reference \(x_L\) instead of aborting; the hydro-side
gates (path admissibility, dt floors, absorption triggers) respond on the
evolved state.

For the S3 tri-fan cap variant, axis-rezone metrics, reverse cell-node CSR
rebuilds, core-quality triggers, patch target construction, and patch volume
guards use the active-slot topology contract from §3.2.0a: cap triangles read
only storage slots \(0,1,2\), inactive slot 3 contributes zero mass/velocity
projection weight, and inactive local face 0 is not sampled. Quad cells keep the
four-slot formulae. The diagonal-corner barrier is the half-butterfly quad
keystone metric; true cap triangles are not diagonal-barrier samples and remain
guarded by active-polygon corner/volume admissibility. The cap apex \(O\), the
only `NODE_CENTER|NODE_AXIS|NODE_BOUNDARY` node in the cap topology, is a fixed
origin anchor: it is excluded from the PAVA movable set, its axis target is
forced to \(R^*=0,Z^*=0\) in axis-only and full-patch target construction, and
boundary velocity projection enforces \(u_O=0\).

For the five-block half-butterfly topology, the axis target has an additional
convergent-locality patch target.  On each multiblock ALE driver opportunity,
independent of whether the ordinary axis-rezone trigger fires, the driver
evaluates the outer physical boundary scale
\[
\alpha={1\over N_o}\sum_{n\in\partial\Omega_o}
{|\mathbf{x}_n|\over |\mathbf{x}_{n,0}|},
\qquad
\dot s={1\over N_o}\sum_{n\in\partial\Omega_o}
{\mathbf{x}_n\cdot\mathbf{v}_n\over |\mathbf{x}_n|\,|\mathbf{x}_{n,0}|}.
\]
With \(\alpha_{\rm tol}=10^{-8}\) and \(\dot s_{\rm tol}=10^{-12}\,{\rm s}^{-1}\),
the convergence predicate is active when
\(\alpha<1-\alpha_{\rm tol}\) and
\(\dot s<-\dot s_{\rm tol}\).  Once active, it remains active while
\(\alpha<1-\alpha_{\rm tol}\).  If this convergence predicate is false, the
installed target is byte-identical to the axis-only target above: no patch is
built, no patch node is installed, and the convergent-locality engaged counter
remains zero.

The patch firing schedule is decoupled from the axis edge/altitude trigger.
The PAVA axis projection still fires only on the ordinary axis trigger above.
The non-axis patch fires when the convergence predicate is active and the
current core/seam quality trigger detects a developing core distortion:
\[
S_{\max}^{core}>S_{\rm trigger}=10^{-2},
\]
where \(S_{\max}^{core}\) is the maximum edge-scale-normalized hourglass shear
over `CENTRAL_CORE` cells and fan cells touching the shared core-fan seam. A
non-positive sampled core/seam corner-\(J\) also activates the same trigger as
a last admissibility signal.  When the core-quality trigger is inactive, the
patch is not built; if the axis trigger fired, the step remains axis-only. When
the core-quality trigger is active but the axis trigger is inactive, the patch
uses the current axis position as its Dirichlet axis target (\(R=0\), current
\(Z\)); PAVA is not called or installed on that step.

When the patch fires, TENRYU first installs either the PAVA axis target
\(R_i^*=0,Z_i^*\) on an axis-triggered step or the current axis \(Z\) target on
a patch-only step.  It then builds a patch-limited cross-seam CSR Winslow target
from the current Lagrangian mesh.  Let \(\mathbf{X}_L\) be the current node
coordinates and \(\mathbf{X}_W\) the patch Winslow solution.  The installed
non-axis patch target is
\[
\mathbf{X}_n^* =
\mathbf{X}_{L,n}+\beta_n(\mathbf{X}_{W,n}-\mathbf{X}_{L,n}).
\]
The patch is deterministic in sorted node id order.  Its smoothing domain is
all `CENTRAL_CORE` nodes plus the adjacent \(l=0\) fan-cell buffer.  Core and
shared core-fan seam nodes use \(\beta=1\).  The outer boundary of the fan
buffer uses \(\beta=0\), so those nodes are fixed during smoothing and install
their current Lagrangian coordinates.  This one-cell buffer is the discrete
\(\beta:1\to0\) taper.  Shared core-fan seam nodes are movable patch nodes and
use the same cross-seam CSR neighbor aggregation as the scheduled
multiblock Winslow smoother.  Axis nodes are forced Dirichlet during the patch
solve to the selected axis target, \(R=0, Z=Z_{\rm axis}^*\), overriding the
ordinary `NODE_POLE_AXIS` tangential Winslow behavior so the axis projection or
current-axis hold is preserved exactly.

The patch builder is target-only: it does not call the scheduled full smoother,
does not mutate `State.x_r/x_z`, and does not edit pressure, velocity, energy,
or density.  Before the remap transaction, TENRYU audits explicit target cell
113 (the default convergent diagnostic cell) by comparing the axis-only and
full-patch targets for hourglass shear, aspect ratio, and oriented minimum
corner-\(J\).  Shear reduction is required only when the axis-only target has
meaningful hourglass shear,
\[
S_{\rm axis}>S_{\rm audit}=10^{-2}.
\]
This threshold is dimensionless in edge-scale-normalized hourglass shear and is
about \(2.4\times\) the observed uniform-core baseline \(S\simeq4.2\times10^{-3}\);
therefore an already-quiescent core may accept an identity patch.  The
corner-\(J\) check is always active: the patch target must keep the oriented
minimum corner-\(J\) positive and not smaller than the axis-only target, up to a
\(10^{-12}\max(1,|J_{\rm axis}|)\) comparison tolerance.

A separate target-volume guard checks every core and fan-buffer cell with
\[
V_c^* \ge 0.5\,V_c^{\rm Lag}
\]
and checks the aggregate patch volume with
\[
\sum_c V_c^* \ge (1-10^{-10})\sum_c V_c^{\rm Lag}.
\]
The per-cell \(0.5\) floor allows Winslow redistribution of volume while still
blocking cell collapse.  The aggregate \(10^{-10}\) lower tolerance rejects
systematic material pre-compression while allowing roundoff-level conservation
noise.

Both guards are non-fatal.  If the volume guard fails, TENRYU logs
`axis_rezone_patch_volume_guard_fallback`; if the cell-113 audit fails, it logs
`axis_rezone_cell113_audit_fallback`.  In either case the driver skips only the
non-axis patch install and continues through the existing axis-only target and
conservative remap path when the axis trigger fired.  For a patch-only firing,
guard fallback returns without a remap because the remaining target is the
current mesh.  The convergent-locality engaged counter counts only steps where
the patch target is actually installed.

Energy and conservation contract: the axis rezone target is applied only
through the existing mass/momentum/energy-conserving multiblock CSR
conservative remap and its normal post-remap reclosure.  There is no unpaired
force, velocity, coordinate-commit, pressure, or internal-energy edit outside
that remap transaction; if the remap is not accepted, the pre-rezone mesh and
state are restored.

**Exp1 virtual central pseudo-core:**

`Numerics.ale.central_pseudo_core_enabled=True` activates a default-off
diagnostic for the 2D_RZ five-block multiblock center topology.  It addresses
the absolute-volume \(r\to0\) CFL collapse that can remain after the center
patch and \(\Phi\)-barrier have kept cell shape valid.  The \(\Phi\)-barrier is
a relative shape/admissibility operator; it does not impose a minimum absolute
control-volume size.

The pseudo-core is virtual: TENRYU does not remove CSR cells and does not add a
new topological cell.  At first use it builds one fixed member set from
structured block/ring indices, not from a cut through cell centroids.  For
`multiblock_half_butterfly_trifan_cap_5block`, `CENTRAL_CORE` is the pinned
tri-fan cap block.  TENRYU includes complete cap cell rings
\[
0\le i<K,\qquad 0\le j<n_\theta,
\]
where \(K\) is the largest count whose included rings have maximum initial node
radius no larger than \(s_c\):
\[
\max_{c\in C_K}\max_{n\in c}\sqrt{r_n^2+z_n^2}\le s_c.
\]
For the legacy non-cap five-block topology, the fallback is the analogous
axis-apex-centered rectangular ring in the central-core block,
\[
0\le i<K,\qquad j_{\rm mid}-K\le j<j_{\rm mid}+K.
\]
No face-adjacent halo is added.  The member/non-member face frontier is
therefore an existing mesh-edge loop and is converted to an ordered
boundary-node loop.  Exp1 fails closed if the boundary is not one closed loop.
Under the fixed Exp1 config there is no scheduled de-agglomeration.

*Dynamic complete-ring absorption* (namelist
`Numerics.ale.central_pseudo_core_ring_absorption_enabled`, default off; the
historical `TENRYU_I1B_RING_ABSORB*` environment variables override the
namelist keys when SET) can grow the member set during a run; membership
never shrinks.  The
absorption DEPTH \(D\) counts complete structural units along the ladder
*cap rings → fan layers → polar-shell rows*: \(D\le n_{\rm cap}\) absorbs the
first \(D\) cap rings; \(D=n_{\rm cap}+l\) additionally absorbs fan layers
\(0..l{-}1\) (each layer is the union of layer \(l\) across the NORTH/EAST/
SOUTH fan blocks, \(4n_c\) cells); \(D=n_{\rm cap}+n_b+q\) additionally
absorbs the first \(q\) complete polar-shell rows.  Three triggers share one
execution path:

1. *Volume trigger* (Lagrangian step start): the minimum RZ volume over a
   watched active unit (ring, fan layer, or shell row) falls below \(\tau\)
   (`central_pseudo_core_ring_absorption_tau`, default 0.05; env override
   `TENRYU_I1B_RING_ABSORB_TAU`) of that unit's birth minimum — the volume
   cached when the unit first entered the watch window, not the \(t=0\)
   volume. The watch spans the first `TENRYU_I1B_ABSORB_WATCH_ROWS` active
   units (default 1 = the historical single-ring watch). A wider window is
   the endgame PRE-trigger: a row beyond the first active unit can collapse
   to a committed-degenerate volume within one accepted step — observed as
   a hard non-positive-volume assert at \(t=2.78\) ns while the single-ring
   watch looked only at the first unit — and absorbing it at \(\tau\) of
   birth keeps the mesh out of the epsilon-margin regime where the
   path-metric and the exact volume recompute can disagree.
2. *Failure trigger*: the center-patch final admissibility check finds the
   post-Lagrangian mesh inadmissible (e.g. a one-step pole-cell fold) at an
   absorbable active cell.  A ring-absorption request is armed on
   `CentralPseudoCoreState`, the ALE step aborts with
   `AleStatus::CenterPatchInadmissibleAbsorbRetry`, and the driver full-step
   retry restores the pre-step snapshot.  The snapshot intentionally excludes
   `CentralPseudoCoreState`, so the request survives the restore and executes
   at the start of the retried attempt on the healthy pre-step state.
3. *dt-floor rescue*: when the dt lineage would abort below `dt.min_s` and
   the hydro argmin cell is absorbable, the request is armed and the step is
   clamped to `dt.min_s` for the absorption step instead of aborting.
4. *Macro-boundary rescue*: when the macro boundary ITSELF is
   path-inadmissible (`kCentralMacroCoreSentinelCell` failure) and the
   driver full-step dt retries are exhausted — a violation static in dt
   (boundary cells converged below the absolute margin floor) — the FIRST
   ACTIVE unit is absorbed so the boundary moves outward, and the step
   retries with a fresh budget. Bounded: the repeat guard refuses
   re-requests at an armed depth, so rescues ≤ absorbable units.

Execution rebuilds the member set with the requested depth (complete units
only).  Conservation reuses the activation machinery: the newly passive
interior nodes' kinetic energy is banked by the activation-KE projection and
deposited into \(U_e/U_i\) with the \(\chi_e\) split at the next
`hydro_step_start` sync aggregate, which also re-sums \(U_{e,C},U_{i,C}\) from
the absorbed unit's still-real member state before any mirror write
(\(M_C,M_{Y,C}\) are re-summed at every aggregate call).  Shell-row absorption
is bounded by the *material guard*: only the contiguous prefix of pure-gas
shell rows is absorbable, gated by the MASS-WEIGHTED row tracer fraction
\(\sum_j m_{q,j}Y_{q,j}/\sum_j m_{q,j}\ge\)
`central_pseudo_core_ring_absorption_gas_tracer_min` (default 0.99; env
override `TENRYU_I1B_RING_ABSORB_GAS_TRACER_MIN`) together with a loose
per-cell hard bound \(\min_j Y_{q,j}\ge\)
`central_pseudo_core_ring_absorption_gas_tracer_cell_min` (default 0.5; env
override `TENRYU_I1B_RING_ABSORB_GAS_TRACER_CELL_MIN`); no tracer field
→ zero rows.  (The weighted statistic is deliberate: a per-cell row minimum
is hostage to single pole-column cells whose tracer dips from remap
diffusion while the row is still physically gas, whereas a genuinely mixed
interface row fails the weighted gate by a wide margin.)  The gas/shell
material interface is therefore never absorbed into the macro CV; a topology
backstop additionally keeps at least one active shell row.  `central_pseudo_core_ring_absorption_max_rings` (env override
`TENRYU_I1B_RING_ABSORB_MAX_RINGS`) is an optional tighter cap (default 0 =
unlimited up to the material/topology guard).  A
repeated inadmissibility at a depth already requested, or a request beyond
the guard, refuses the request and falls through to the hard assert, so
absorption cannot mask a non-geometric defect or eat the material shell.

**I1-B Stage 1 dynamic polar-shell angular de-refinement:**

The runtime environment variable
`TENRYU_I1B_POLAR_SHELL_ANGULAR_DEREFINE=1` enables a default-off virtual
angular de-refinement over the `POLAR_SHELL` block pole bands.  OFF runs do not
build the overlay and remain byte-identical.  ON also enables the hardened path
predicate used by `TENRYU_I1B_PATH_PREDICATE_HARDEN`.

At first hydro setup the overlay builds dyadic angular macros at both pole
ends, with each macro covering exactly one radial shell row:
\[
i_{\rm end}=i_{\rm begin}+1,\qquad
j_{\rm end}-j_{\rm begin}=2^\ell .
\]
The selected span is the smallest dyadic span, capped by the Stage-1 level cap,
for which
\[
\ell_\theta = r\,\Delta\theta \ge \chi\,\Delta r,\qquad \chi=1 .
\]
Groups are admitted only if their fine children are pure material by the gas
tracer guard; the gas/shell interface is never crossed.  The radial gas-cell
count from pole to shell is therefore unchanged.  Only the near-pole angular
count participating in hydro/remap/CFL/path operators is reduced.

After each accepted Lagrangian energy/geometry update and after each accepted
ALE rezone+remap, the same criterion is re-evaluated on the current committed
`node_r,node_z` for every existing pole macro.  If the selected span is larger
than the current span, the macro grows monotonically to that selected span,
bounded by the same Stage-1 level cap; it never shrinks.  Growth is a descriptor
replacement, not a mesh-topology edit: `make_macro` builds the wider macro,
the existing `State.corner_mass` kinetic projection deposits the nonnegative
\(\delta K\) into member internal energy, the inactive/boundary masks and
geometry-exempt union are rebuilt, and the macro state is re-aggregated from
member mass, tracer mass, internal energy, and the exact new boundary volume.
Each accepted growth logs `[pole_angular_derefine_maintain]` with pole side,
q-row, old/new span, current \(r\Delta\theta/(\chi\Delta r)\), and the mass
delta measured across the replacement.  If the criterion is already satisfied
by the current span the maintenance pass is a no-op, so there is no per-step
span thrash.

The de-refinement is virtual.  CSR topology is unchanged.  Covered fine angular
children are marked inactive in one shared mask; ordinary hydro force/work,
CSW edge AV, acoustic and volume-rate CFL, path admissibility, ALE
donor/target/remap, corner-mass remap audits, KE fixup, and positivity/floor
paths consume the effective active or inactive mask.  A one-row de-ref pole
macro is not collapsed to a single apex.  Its pole side is the radial RZ-axis
edge between the two distinct shell nodes
\((i,j_{\rm pole})\) and \((i+1,j_{\rm pole})\), both with \(R=0\) but with
different \(s/z\).  The loop is oriented as a positive RZ polygon and uses the
four-node boundary
\[
(i,j_{\rm pole})\rightarrow(i,j_{\rm side})
\rightarrow(i+1,j_{\rm side})\rightarrow(i+1,j_{\rm pole}),
\]
where \(j_{\rm side}=j_1\) at the north pole and \(j_{\rm side}=j_0\) at the
south pole.  The macro volume is the exact finite RZ polygon volume over this
full boundary, including the \(R=0\) axis edge,
\[
V={\pi\over3}\sum_k (r_k+r_{k+1})(r_k z_{k+1}-r_{k+1}z_k),
\]
so the axis contributes with \(r=0\) and no \(1/r\) division.  A non-positive
axis-wedge volume is treated as a real macro inversion signal.  The pole macro
path predicate keeps the \(R=0\) axis edge as a valid closure and tests it for
nonadjacent crossings like every other boundary edge; only topologically
adjacent endpoint contacts are ignored.  Thus positive full-loop RZ volume is
necessary but not sufficient: a nonadjacent off-axis edge crossing the finite
axis-edge interior is a real macro self-intersection.  The macro boundary adds the
compatible pressure force
\[
F_a=(p+q)\,\frac{\partial V}{\partial x_a}
\]
on the ordered RZ polygon boundary and closes work by
\[
\dot U=-\sum_a u_a\cdot F_a .
\]
Internal child faces vanish from macro-owned force/work and edge-AV
accounting.

Activation uses `State.corner_mass` as the basis.  Child corner masses and
momenta are conservatively projected to the dyadic macro boundary basis with
nonnegative weights \(B_{Ii}\):
\[
\tilde m_I=\sum_i B_{Ii}m_i,\qquad
\tilde p_I=\sum_i B_{Ii}m_i u_i,\qquad
\tilde u_I=\tilde p_I/\tilde m_I .
\]
The new kinetic energy is
\[
\tilde K_I={|\tilde p_I|^2\over 2\tilde m_I},
\]
while the scattered old corner kinetic energy is
\[
\hat K_I=\sum_i B_{Ii}\,{1\over2}m_i|u_i|^2 .
\]
The nonnegative difference
\[
\delta K_I=\hat K_I-\tilde K_I\ge0
\]
is deposited into member-cell internal energy with positive mass weights and
the usual \(U_e/U_i\) split.  Total energy is not independently remapped and
there is no signed clamp or ledger.  When the polar overlay is active,
Option-B velocity authority and total-energy remap are rejected in the merged
region; this is the basis-redesign trigger if a deck still requires them.

Under the same env gate, Stage 1b hard assertions reject any inactive child
reported as a path winner, CFL winner, CSW edge-AV winner, or remap/closure
participant.  Per-step `[pole_angular_derefine_audit]` diagnostics report mass,
tracer mass, internal energy, and total-energy drift around remap and
compatible pressure work.  If a de-ref pole macro still reports
`macro_repair_required`, driver retry restores the pre-step snapshot and mutates
the macro before the next attempt: after removing any accidental zero-length
duplicate boundary edge, retry replaces the failed pole interval by the next
larger dyadic span on that q-row, rebuilds the inactive/boundary masks, and
re-aggregates the macro state.  If the path failure is instead
`pole_axis_radial_order_inversion`, retry restores the pre-step snapshot,
projects the pole-axis target to restore strict logical radial order, applies
the existing conservative swept-volume/total-energy remap through that target,
and retries at the same dt.  If an active fine child on the same q-row and pole
side outside the current macro becomes the path winner, retry re-evaluates the
same \(r\Delta\theta\ge\chi\Delta r\) span criterion on the current committed
node geometry and grows that pole macro far enough to include the child,
rounded to the next permitted span and capped by the Stage-1 level cap.  The
reactive path uses the same conservative descriptor replacement, kinetic
projection, mask rebuild, and aggregation sequence as proactive maintenance,
then retries at the same dt.  Each extension logs
`[pole_angular_derefine_extend]`; if the span grows beyond the pole-side bound
(\(n_j/4\) by default), `[pole_angular_derefine_ball_dominance_warn]` is emitted
and the run continues.  If the coarsest permitted span cannot include, repair,
or extend the failing child, retry stops with a terminal diagnostic instead of
replaying the unchanged geometry.

**I1-B Stage 2 pole-axis BBSW closure** is default-off under
`TENRYU_I1B_POLE_AXIS_BBSW=1`.  With the flag unset, Hydro2D uses the Stage-1
polar angular de-refinement path unchanged.  With the flag set, each polar-shell
pole column (`local_j=0` and `local_j=n_j`) forms a planar Barlow-Burton-
Shashkov-Wendroff nodal acceleration
\[
a^{\rm pc}_{z,q}={\sum f^{\rm pc}_{z,q}\over \sum \hat m^{\rm pc}_q}.
\]
The planar numerator includes active fine-cell pressure and scalar-AV corner
forces, the planarized CSW edge-AV contribution when `av_model="csw_edge"`, the
compatible angular de-ref macro boundary force on both radial macro-boundary
nodes, and the external r-outer pressure-drive boundary force with the same
inward sign as the true RZ pressure boundary.  The denominator uses the
corresponding planar corner/macro masses.  After the ordinary true-RZ
force-to-mass division and before boundary acceleration constraints, Hydro2D
overwrites only these pole-column axis nodes with `accel_r=0` and
`accel_z=a^{pc}_{z}`.

Before predictor and corrector Lagrangian position commit, the predicted
axis distances \( \hat s_q=s_q+\Delta t\,u_{s,q}\) are projected by weighted
PAVA onto \(s_{q+1}-s_q\ge\delta_q\), with weights
\(w_q=\hat m_q/\Delta t^2\) and
\(\delta_q=64\,\epsilon_{\rm mach}\max(|s_q|,|s_{q+1}|,10^{-300})\).  The
projected distance is converted back to the axis \(z\)-velocity and is used for
both the node position commit and compatible work.  In the corrector, the final
axis `state.v_z` is set so the old/new velocity average used by compatible
work equals the projected midpoint velocity.  Hydro2D then adds a compatible
work correction on the BBSW pole-column nodes:
\[
\dot W_{\rm corr,q}=-u^{n+1/2}_{z,q}
\left[m^{\rm RZ}_q{(u^{n+1}_{z,q}-u^n_{z,q})\over\Delta t}
-F^{\rm true,RZ}_{z,q}\right].
\]
This is distributed to incident active cells with the same planar corner-mass
weights used by the BBSW denominator.  Thus the axis-node kinetic energy in
`state.v_z`, the PAVA constraint impulse, and the compatible pressure-work
update use one effective constrained nodal impulse.  The moved nodes enter the
ordinary RZ swept-volume geometry refresh and any accepted ALE/remap path, so
mass follows the existing conservative cell-volume/remap accounting.  The
associated axis-contact timestep is
\[
\Delta t_{\rm axis}=0.5\min_q
{g_q-\delta_q\over \max(u_{s,q}-u_{s,q+1},0)},
\]
clamped at zero if a current roundoff gap is already exhausted, and is folded
into the hydro dt minimum.

*Emergency mixed-material absorption* (experimental, env
`TENRYU_I1B_MIXED_ABSORB`, default off; macro-boundary endgame verdict
direction (a), ideal-gas MVP). The terminal failure of the best-surviving
Case B runs is, by direct forensics, a SIMPLE-LOOP self-intersection of the
macro boundary at the pole after the pure-gas ladder is exhausted (the
volume floor passes by orders of magnitude). When enabled, a
failure-triggered absorption request whose target is the immediately-next
unit (\(D=\) member count, one row per request) may absorb a MIXED
dense-shell row — bypassing the gas-tracer guard but not the structural
backstop (the last shell row stays active) nor `max_rings` — gated on the
STATIC admissibility of the would-be new outer boundary loop at the current
committed node positions (positive RZ polygon volume + simple loop; the
loop ORDER is derived on the initial geometry, the same topological rule
the membership selector uses). The verdict's key point is implemented
literally: the admission test belongs on the proposed NEW boundary, because
absorbing the failed boundary is the rescue. Thermodynamic closure is
UNCHANGED in this MVP: with a single-\(\gamma\) ideal gas the pooled macro
closure \(p_C=(\gamma-1)U_C/V_C\) is exactly the two-material
pressure-equilibrium answer (\(p_C=\sum_m(\gamma_m-1)U_m/V_C\) degenerates
when all \(\gamma_m\) are equal), and the tracer already partitions the
macro mass by material (\(M_{Y,C}\) vs \(M_C-M_{Y,C}\)). Any future
per-material-EOS physics MUST replace this with a true multi-material macro
state (per-material \(M_m,U_m\), pressure-equilibrium solve, and the
compressibility-weighted work partition
\(\chi_m=(V_m/K_m)/\sum_j V_j/K_j\)). A run that performs at least one
emergency absorption is a *mixed-core endgame* run: the macro CV then
contains dense-shell material, and gas-region metrics that count the whole
macro volume (e.g. CR\(_V\) including the macro core) carry that caveat
from the first mixed absorption onward (`mixed_absorbed_row_count` in the
macro state and the granted-log line mark the transition).

*Corner-mass basis contract (experimental stack, all default-off).* The
dynamics' nodal masses and the energy-budget KE use the cached subzonal
corner-mass basis (`state.corner_mass`), while the Option-B velocity remap
conserves corner momenta in its transient first-moment basis; under heavy
remap activity (the mixed-core endgame) the basis product
\(\sum \tfrac12(m^{F}_a-m^{B}_a)\,\Delta(v^2)\) appears as a roll-dependent
budget drift (frozen-basis N=4 envelope \([-1.6\%,+23\%]\)). The landed
mitigation is the V-PAIRED INSTALL (`TENRYU_I1B_INSTALL_VPAIRED_CORNER_MASS`):
at each CSR remap the corner basis is rebuilt as \(m_a=\rho_c V_a\) with the
SAME exact-subpolygon corner volumes the subzonal-pressure closure divides
by (uniform states give \(\rho_a\equiv\rho_c\) and zero subzonal noise to
round-off; one multiplicative per-cell scale enforces
\(\sum_a m_a=m_c\) exactly), validated and abandoned-on-violation. This
narrows the drift envelope ×12 (N=4: \([+1.5\%,+3.5\%]\), all-positive,
monotone in the install count) and makes the endgame trajectory nearly
roll-deterministic, at the cost of a deterministically different (deeper,
earlier) absorption cascade. Related env-gated arms retained for
adjudication evidence only (NOT production candidates): per-step geometry
canonicalize and the raw transported install (both physics-breaking), the
frozen-basis TER measurement, the macro-band rezone taper + lifts, the
incremental KE-fixup deposit, the F-basis momentum projection, and the
Option-B subzonal-basis gather (catastrophic in composite). Full
adjudication: docs/design/20260612-i1b-corner-mass-basis-adjudication.md.

*Basis-coherent Option-B bookkeeping chain — "coherent-lite"
(`TENRYU_I1B_OPTIONB_COHERENT`, default-off; supersedes and hard-gates off
the subzonal-basis gather, the KE-fixup deposit, the F-basis projection,
the per-install KE compensation, the gap-form deposit, and the frozen-K
TER arm when set).* The adjudicated resolution of the energy-envelope
caveat. With the env set (production composition): the velocity remap's
transport internals stay FIRST-MOMENT (unchanged), and the coherence is
in the bookkeeping chain: (i) after the nodal scatter the transported
ledger is V-PAIR PROJECTED at the post-remap geometry
(\(m'_a=m_c V_a/\sum V\), the PR6-LO product computed on the component's
own cell sums; transiently degenerate cells keep their transported
partition renormalized to the cell sum, so a single bad cell no longer
abandons the whole install); (ii) the nodal bookkeeping is rebased
velocity-preservingly (node mass \(M'_n\) from the projected corners,
node momentum rewritten \(M'_n v_n\), velocities UNCHANGED — uniform flow
exact); (iii) the TER books its pre-remap K in the basis
(`state.corner_mass`) and its post-remap K in the projected ledger, so
all basis-swap KE lands in the existing TER internal-energy discrepancy
deposit — total energy closes by construction; (iv) the install writes
the component's EXPORTED projected ledger (not a recompute), so the
installed basis and the bookkeeping use identical numbers. Full-window
Case B (lite_r1, 2026-06-13): dE_rel = +2.7e-6 (frozen-basis arm +19.1%,
install-only [+1.5,+3.5]%), 69170/69170 installs, gas RESOLVED through
stagnation to t = 3.23 ns (frozen terminal-absorbed at 2.88 ns), resolved
CR50 at the 1D stagnation anchor 2.559 vs 2.57 (0.4%).
Two opt-in arms adjudicated PATHOLOGICAL and parked behind envs:
`TENRYU_I1B_OPTIONB_COHERENT_TRANSPORT=1` (basis gather seeding + fixed
σ-share donor distribution \(\sigma_a=m^{\rm basis}_a/\sum m^{\rm basis}\)
with the uniform velocity-content correction — unconditionally
positivity-safe and uniform-flow exact, but the rigid-shift donor
distribution delocalizes momentum removal under strong intra-cell
velocity gradients and crushes the gas-shell interface ~1 ns early), and
`TENRYU_I1B_OPTIONB_COHERENT_RERECOVER=1` (momentum-conserving re-recover
\(v'_n=P_n/M'_n\) — exact momentum in the installed basis, but its
per-install \(M/M'\) velocity ripple wrecks the converging-core mesh;
the projection trade momentum↔uniform-flow is irreducible because
V-pairing is not comoving under node motion). Adjudication record:
docs/design/20260612-i1b-optionb-basis-coherent-redesign.md.

The aggregate volume is the current sum of member RZ volumes,
\[
V_C=\sum_{i\in C}V_i.
\]
For member cell masses \(m_i=\rho_iV_i\), gas tracer \(Y_i\), and electron/ion
specific energies \(e_{e,i},e_{i,i}\), the conserved aggregate is
\[
M_C=\sum_{i\in C}m_i,\quad
M_{Y,C}=\sum_{i\in C}m_iY_i,\quad
U_{e,C}=\sum_{i\in C}m_ie_{e,i},\quad
U_{i,C}=\sum_{i\in C}m_ie_{i,i}.
\]
The projected intensives are
\[
\rho_C={M_C\over V_C},\quad
Y_C={M_{Y,C}\over M_C},\quad
e_{e,C}={U_{e,C}\over M_C},\quad
e_{i,C}={U_{i,C}\over M_C}.
\]
These intensives are written to all member cells at hydro step start, after the
predictor geometry/density refresh before EOS pressure closure, after accepted
ALE/remap changes, and after compatible energy updates.  The predictor
re-aggregation is required so the half-step pressure used by the pseudo-core
boundary force is the pressure of the same aggregate CV, not the pressure of a
freshly re-expanded representative storage cell.  For Exp1 the
non-representative members remain storage cells with their cell/corner masses
intact for nodal inertia, but their hydro pressure/subzonal/AV/dt operators are
passive.  Per-material conservation arrays are not supported in this diagnostic
path.

When `Numerics.hydro.total_energy_remap_2d_rz=true`, the CSR total-energy remap
seeds each active pseudo-core member from the aggregate mirror rather than any
independent member internal-energy history:
\[
E_{tot,i}^{mat}=m_i(e_{e,C}+e_{i,C})+K_i^{node},\qquad
Y_{e,i}^{int}={e_{e,C}\over e_{e,C}+e_{i,C}}.
\]
The kinetic term \(K_i^{node}\) is the same RZ corner-mass measure used by the
ordinary total-energy remap.  Thus the later recovery
\(e=(E_{tot}^{mat,*}-K^{node,*})/m^*\) sees the pseudo-core members as storage
mirrors of one aggregate control volume.  This override is local to the remap
seed and is skipped when either the pseudo-core or total-energy remap is
inactive.  The split fraction uses \(Y_{e,i}^{int}=1/2\) if the aggregate
internal-energy sum is non-positive or non-finite.

At activation this is a conservative representation change.  Let
\({\cal N}_C\) be the member-cell nodal set with nodal masses assembled from the
same RZ corner-mass partition used by the hydro node-mass operator, and let
\(\partial C\subset{\cal N}_C\) be the ordered pseudo-core boundary nodes.  Before
the interior pseudo nodes are made passive,
\[
K_C^{\rm old}=\sum_{n\in{\cal N}_C}{1\over2}M_{C,n}|\mathbf{u}_n|^2.
\]
After activation, only boundary nodes retain pseudo-zone active kinetic energy,
\[
K_C^{\rm new}=\sum_{n\in\partial C}{1\over2}M_{C,n}|\mathbf{u}_n|^2.
\]
TENRYU zeroes the non-boundary member-node velocities and banks
\[
\Delta U_C^{\rm act}=K_C^{\rm old}-K_C^{\rm new}
\]
into the pseudo-core internal energy before member mirrors are written.  The
increment is split with the same \(\chi_e+\chi_i=1\) rule as pressure work.  The
implementation uses the global energy-budget drop from zeroing as the banked
quantity, so the reported total energy is continuous across activation to
roundoff.  If a future run deactivates the pseudo-core after activation, the
inverse projection restores the saved passive-node velocities and removes the
restored kinetic energy from \(U_C\), failing closed if this would violate
internal-energy positivity.

The acoustic CFL and multiblock RZ-volume mesh-quality limiter skip every
pseudo-core member, including the representative.  TENRYU then adds one
pseudo-core acoustic limit
\[
\Delta t_C=C_{\rm CFL}{h_C\over c_C(1+C_{\rm av,1})},\qquad
h_C=\left({V_C\over 4\pi/3}\right)^{1/3},
\]
using the aggregate-state sound speed when available.  This removes the
pathological first-cell absolute-volume limiter while still letting the
agglomerated core bound the hydro step.

The compatible force/work coupling zeroes all pressure corner forces, subzonal
pressure forces, member-adjacent CSW edge-AV forces, and compatible work entries
for member cells before adding the pseudo-core boundary operator.  For each
ordered boundary node \(n_k\), the current one-loop member frontier gives the
exact gradient of the closed CV volume (equivalent to the member-volume sum for
a valid Exp1 member union) using the same discrete formula as
`rz_polygon_volume_exact`,
\[
\mathbf{S}_{C,k}={\partial V_C\over\partial\mathbf{x}_{n_k}},
\qquad
\dot V_C=\sum_{k\in\partial C}\mathbf{S}_{C,k}\cdot\mathbf{u}_{n_k}.
\]
The nodal force accumulator receives
\[
\mathbf{F}_{n_k}^{C}=p_C\,\mathbf{S}_{C,k}.
\]
The final-kick impulse stored for the pseudo-core boundary is
\[
\mathbf{I}_{C,k}=\Delta t\,\mathbf{F}_{n_k}^{C}.
\]
Its compatible internal-energy increment is evaluated only after the final
nodal velocity is available:
\[
\Delta U_C=-\sum_{k\in\partial C}\mathbf{I}_{C,k}\cdot
\mathbf{u}_{n_k}^{\sharp},\qquad
\mathbf{u}_{n_k}^{\sharp}={1\over2}
(\mathbf{u}_{n_k}^{n}+\mathbf{u}_{n_k}^{n+1}).
\]
This is the kinetic-energy-conjugate identity for
\(|\mathbf{u}^{n+1}|^2-|\mathbf{u}^{n}|^2\); using
\(\mathbf{u}^n\), \(\mathbf{u}^{n+1}\), or a predictor velocity is not
compatible with the final nodal momentum update.

While the pseudo-core is active, \(U_C\) is authoritative between explicit
synchronization points.  Member cells are storage mirrors.  The total
\(\Delta U_C\) is applied once and split
\[
\Delta U_{e,C}=\chi_e\Delta U_C,\qquad
\Delta U_{i,C}=(1-\chi_e)\Delta U_C,
\]
with \(\chi_e=U_{e,C}/(U_{e,C}+U_{i,C})\) when the two-temperature total is
positive and \(\chi_e=1/2\) otherwise.  The member mirrors are then written
from the updated aggregate totals.  This sign matches the existing compatible
work convention where positive nodal pressure force on the mesh corresponds to
negative pressure work on the CV.  Internal member/member faces and
pre-existing member subzonal forces are not counted.

Every post-energy aggregation emits a stderr audit line with total gas mass,
gas-tracer mass, electron internal energy, ion internal energy, and total energy
relative drift from the first aggregate baseline.  When
`TENRYU_I1B_PSEUDOCELL_WORK_PROBE` is set, a second stderr line prints
\(K_C^{\rm old}\), \(K_C^{\rm new}\), the activation bank
\(\Delta U_C^{\rm act}\), and the zeroed-node count.  The work probe also prints
\(W_{\rm old}=-\sum\mathbf{I}_{C,k}\cdot\mathbf{u}^n_k\),
\(W_{\rm new}=-\sum\mathbf{I}_{C,k}\cdot\mathbf{u}^{n+1}_k\),
\(W_{\rm mid}\), and
\(R_C=\Delta U_C+\sum\mathbf{I}_{C,k}\cdot\mathbf{u}^{\sharp}_k\).
The audit is diagnostic only and does not add HDF5 schema.

For a uniform field, the target-only axis rezone is free-stream preserving under
the same CSR conservative remap.  The forced-rezone uniform-field test applies
the full patch target and observes roundoff-level internal-energy GCL drift.  The
production `max_vol_closure` aggregate currently exposes the R-momentum
GCL-reference relative drift component; in nearly symmetric blast cases where
the net R-momentum reference is approximately zero, this can be a
collapsing-denominator metric artifact, analogous to the B-S2-T3 physical
momentum-floor issue, rather than a conservation failure.  The conservation
residuals for mass, R-momentum, and Z-momentum remain zero at the accepted
audit scale.

The primitive also reports first off-axis ring diagnostics over cells touching
the axis chain: the minimum current cell edge length and minimum current
quad-altitude. These diagnostics are report-first signals for possible
off-axis slivering; they do not trigger any two-dimensional band smoothing.

The convergent \(C\ge7.2\) interpretation has three outcomes.  Success means
cell 113 remains bounded with no new core/seam single-node mode beyond the
previous step-815 abort.  Relocation of the mode into the seam or fan is a
Fix-2 TRI-fan/topology signal, not license to widen this target patch.
Fallback to axis-only/current-coordinate target followed by a cell-113 failure
is a width-1-equivalent signal that the patch did not reach or help the failing
mode.  Abort means the
default-preserving diverging gates regressed or another non-guard fatal check
failed.

- **物質界面ノード**：v1.0では特別扱いしない（材料界面追跡は非スコープ）
- **r=0軸**：\(r=0\) を維持（軸対称の幾何的要件）

#### 3.3.6 輻射粒子との相互作用

rezone後、**IMC粒子のみ**の `cellId` を更新する必要がある（U7: `cell_search_after_rezone`）。
DDMC粒子は pos=NaN sentinel のため空間探索不可であり、cell_id をそのまま維持する
（ALE rezone はセル番号を変えないため再同定不要。§9.6 参照）。
§9のセル探索アルゴリズムを用いて再同定を行う。

- CFL制約によりrezoneのノード変位は小さいため、現在セル＋近傍の局所探索で十分
- 失敗時のみ拡張探索を行う

#### 3.3.13 Young/PLIC material-interface reconstruction (Stage 30 Wave B)

This section defines the default-disabled Stage 30 Wave B PLIC reconstruction
core.  Unless `Numerics.plic.enabled=True`, none of these paths are active:
initial geometry sampling remains the legacy centroid-only evaluation and
PLIC reconstruction is inert.

For a clipped RZ cell polygon \(P=\{(r_k,z_k)\}_{k=0}^{N-1}\), with vertices
counter-clockwise in the meridional \((r,z)\) plane, the revolved volume is
computed by the same Pappus identity used for quadrilateral cell volumes:
\[
V(P)=\frac{\pi}{3}\sum_{k=0}^{N-1}
(r_k+r_{k+1})(r_k z_{k+1}-r_{k+1} z_k),
\qquad k+1\equiv 0\pmod N .
\]
PLIC clips the convex quadrilateral cell by the half-plane
\[
\hat n\cdot x = n_r r+n_z z \le \alpha
\]
using Sutherland-Hodgman polygon clipping, then evaluates \(V(P)\) on the
clipped polygon.  The plane offset \(\alpha\) is found by bracketed bisection
over the four corner projections; Brent-style interpolation is not used.
The solve target is
\[
\frac{V_{\le\alpha}(\hat n)}{V_{\rm cell}} = f_m,
\]
with `alpha_solver_max_iter` and `alpha_tolerance_rel` controlling the
iteration.

The Youngs seed normal is the negative cell-centered material-fraction
gradient,
\[
\hat n_Y=-\frac{\nabla f_m}{\|\nabla f_m\|},
\]
with one-sided differences at physical boundaries.  Axis-near cells at
\(i=0\) are exempt from reconstruction and are counted separately.  If
\(h_{\rm eff}\|\nabla f_m\| < \max(10^{-8},10\epsilon_{vf}\sqrt{N_s})\),
where \(h_{\rm eff}=\sqrt{A_{RZ}}\), \(\epsilon_{vf}=10^{-10}\), and
\(N_s=9\) for the 3x3 stencil, the gradient is treated as ill-conditioned.

LVIRA refinement samples the fixed deterministic angle set
\[
\{0,30,45,60,90,120,135,150,180,210,225,240,270,300,315,330\}^{\circ}
\]
plus the Youngs seed when present.  For each candidate normal, the same plane
\((\hat n,\alpha)\) is applied to the neighbor cells and the weighted residual
\[
E(\hat n)=\left[
\frac{\sum_s 2\pi r_s\left(f^{PLIC}_s-f^{actual}_s\right)^2}
{\sum_s 2\pi r_s}
\right]^{1/2}
\]
is minimized.  The \(2\pi r_s\) weight is the RZ Pappus bias correction.

The fast path marks cells with any material fraction in
\([10^{-10},1-10^{-10}]\) as multi-material, then expands this mask by
`fast_path_halo_radius_cells` on the structured \((i,j)\) grid.  Single-material
cells outside the halo stay on the scalar path.

At initialization, `t0_volume_cut_method` selects the legacy centroid path or
adaptive volume-cut sampling.  The adaptive path recursively subdivides each
cell into four quads, applies 2x2 or 3x3 Gauss-Legendre quadrature to callable
rho/Te/Ti/material fractions, and accumulates volume averages with the RZ
volume element \(2\pi r\,dA\).  `t0_volume_cut_max_depth` defaults to 6
(was 12) and is tunable over [4, 16]. Empirically: max_depth=6 is sufficient to
resolve smooth Legendre-perturbed shell interfaces in I1 capsule deck within
~5 sec/cell. Higher depths (>=12) cause exponential blowup against hard step
functions and are not recommended for production. Wave A range was [8, 16];
Wave E expanded to [4, 16] to allow faster init for empirical runs. This path
is gated by `Numerics.plic.enabled=True`; setting `t0_volume_cut_method` alone
must not change disabled runs.

Wave B also defines three deterministic degradation detectors/helpers:
negative corner-\(J\) preflight (case 1), ill-conditioned normal fallback
chain previous-normal/LVIRA/facial jump/skip (case 2), and flux-weighted
thermodynamic closure mismatch
\[
\eta_E=
\frac{\sum_{f,m}|M_{fm}|\,|e_{\rm cell}-e^{closure}_{fm}|}
{\max(\sum_{f,m}|M_{fm}|\max(e_{\rm cell},e_{\rm floor}),E_{\rm floor})}
\]
(case 3).  Class-(d) PLIC events are separate from class-(c) ALE escape-valve
events and do not increment `class_c_runtime_fires`.

Stage 30 Wave C wires this reconstruction into ALE material-volume remap
without expanding the physical state contract.  There are no per-material
\(\rho_m\), \(e_m\), or momentum arrays.  The PLIC path transports only
\(f_m V\); all scalar conserved fields continue to use §3.3.4 scalar remap.
The GPU fast path builds an interface mask from
`fast_path_threshold_min <= f_m <= fast_path_threshold_max`, expands it by
`fast_path_halo_radius_cells`, reconstructs active cells on device, and leaves
cells outside the mask on the donor-fraction material-volume closure inside the
PLIC remapper.

Stage 32a Wave A adds a default-disabled per-material conservation state
contract for Wave F.  When
`Numerics.materials.per_material_conservation_enabled=false`, all new
per-material arrays remain size zero and the Stage 30 scalar remap semantics
above are unchanged.  When enabled, the authoritative per-material conserved
state is cell-major and extensive:
\[
M_{c,m}\ [g],\qquad E_{e,c,m}\ [erg],\qquad E_{i,c,m}\ [erg],
\]
stored as `mass_per_material`, `Ee_per_material`, and `Ei_per_material` with
index \(cN_{\rm mat}+m\).  Momentum is not persistent per material; later remap
waves derive per-material momentum from \(M_{c,m}\) and the shared cell
velocity.  `Te_per_material` and `Ti_per_material` are optional lazy caches of
derived EOS inversions, never authoritative thermodynamic state, and are size
zero unless `Numerics.materials.lazy_cache_te_m_enabled=true` while the master
switch is enabled.

Stage 30 Wave E adds a default-disabled CF6 preview,
`Numerics.plic.rho_material_aware_donor`.  When this flag is false, Wave C
semantics above are unchanged: density uses the scalar ALE remapper.  When it
is true and the PLIC remap succeeds, \(\rho\) is remapped through the PLIC face
volume fluxes with a material-aware donor density at PLIC interface donor
cells,
\[
\rho^{\rm eff}_d=\sum_m f_{d,m}\rho_{d,m}.
\]
The preview computes \(\rho_{d,m}\) at runtime without adding per-material
state arrays: table-backed EOS materials use a local GPU bisection in density
at the donor \(T_e,T_i,P_e+P_i\), ideal-gas materials use the corresponding
analytic inverse, and cells without a usable EOS pressure fall back to a
same-material 3x3 local density proxy.  Non-interface donor cells retain the
cell-mean donor density.  If PLIC reconstruction falls back, density falls
back to the scalar ALE remapper for that ALE call.

For an active interface cell, the interface-centroid drift sensor stores the
segment centroid implied by \((\hat n,\alpha)\).  On a later ALE check, the
relative drift is
\[
d_c = \frac{\|x^{\,new}_{I,c}-x^{\,last}_{I,c}\|}{h_c},
\]
where \(h_c\) is the maximum R/Z cell span.  A drift trigger occurs when
\(\max_c d_c >\) `drift_sensor_max_relative`; the swept-volume-fraction metric
is recorded with the same `drift_sensor_max_swept_fraction` threshold for
Wave D production-comparable policy wiring.  Out-of-cycle checks refresh the
stored centroids and log a warning, but they do not mutate
`numerics.ale.every_n_steps`.

More than five consecutive triggered ALE intervals engages a per-run sticky
fallback.  Once sticky fallback is set, `plic_remap_fallback_engaged=True`
persists for observability and all later material-fraction remap uses the
scalar path.  This PLIC fallback is orthogonal to class-(c) ALE escape valves:
both may fire in one step, but `class_c_runtime_fires` is controlled only by
the ALE escape-valve path.

#### 3.3.14 Per-material remap recovery rules (Stage 32a Wave B)

This section documents the per-material conservation array recovery rules
introduced in Stage 32a Wave B. These rules apply only when
`Numerics.materials.per_material_conservation_enabled=True`. When disabled,
legacy single-field scalar remap behavior is preserved.

##### 3.3.14.1 Whole-pass routing

Each ALE call chooses one per-material conserved remap route for the whole
pass: either the PLIC unified pass, where per-material face-volume fluxes
drive all five conserved-density slabs at every interface cell, or the scalar
5 x `n_mat` fallback, where each slab is handled independently. Mixed
face-by-face routing within one ALE call is not supported in Stage 32a.

PLIC is active for this route only when `Numerics.plic.enabled=True`, sticky
fallback is not engaged (`state.plic_remap_sticky_fallback=false`), no drift
fallback is triggered for the call, and PLIC reconstruction succeeds for all
non-axis interface cells. If PLIC is inactive or fallback is engaged, scalar
5 x `n_mat` per-material remap is used; each per-material slab is treated as a
separate density field under §3.3.4.

##### 3.3.14.2 Sum-then-divide momentum reduction

Per Stage 31 Q2.3, `recover_primitive_kernel_per_material` implements the
cell-mean velocity as
\[
v_{r,c}=\frac{\sum_m p_{r,c,m}}{\sum_m M_{c,m}},
\]
and analogously for \(v_z\). This preserves total cell momentum through
per-material storage:
\[
V_c\sum_m \eta_{\rho,m,c} v_{r,c}
=V_c\sum_m \eta_{\rho v_r,m,c}.
\]

##### 3.3.14.3 Dominant-material floor RECOVERY rule

When per-material mode is active and
\(\sum_m M_{c,m}<\rho_{\rm floor}V_c\) after remap, the deficit
\[
\Delta M=\rho_{\rm floor}V_c-\sum_m M_{c,m}
\]
is injected as numerical recovery into the dominant material, defined as the
largest post-clamp \(M_{c,m}\) with lowest material index as the tie break.

This is a recovery rule for floating-point invariant preservation, not a model
of physical material transfer. The injection is audited via `dm_floor`
(cumulative scalar) and `dm_floor_per_material[m]` (per-material breakdown).
The floor source is `cfg.numerics.floors.rho` only. Presence thresholds
(`Numerics.materials.presence_threshold_volfrac` and
`presence_threshold_mass_density_g_per_cc`) gate per-material ratio extraction
at donor/EOS query sites; they do not define physical mass injection.

##### 3.3.14.4 Snapshot-timing invariant for PLIC unified pass

PLIC unified per-material remap reads donor primitive ratios
\((\rho_m,e_{e,m},e_{i,m})\) from `_old` snapshots taken after per-material
pack and before any remap mutation. In `ale_driver.cu` this is the
device-to-device snapshot between `pack_conserved_kernel_per_material` and any
remap kernel launch.

A debug-build runtime assertion verifies that the `_old` snapshot belongs to
the current ALE call. This guards the silent failure mode where donor ratios
are accidentally read from post-remap state while some global conservation
checks still pass.

##### 3.3.14.5 CF6 rho_material_aware_donor bypass

When per-material mode is active, the Stage 30 Wave E CF6 preview
(`Numerics.plic.rho_material_aware_donor`) is redundant because \(\rho\) is
recovered as \(\sum_m M_{c,m}/V_c\). The CF6 code path is skipped by a host
guard. When per-material mode is disabled, the CF6 path remains unchanged.

##### 3.3.14.6 Four-clause floor discipline

Per Stage 31 Q2.5, four floor clauses apply:

- Numerical presence threshold: gates per-material ratio extraction only.
- Nonnegative conserved repairs: pack and recover clamp negative \(M_m\),
  \(E_{e,m}\), and \(E_{i,m}\) to zero with repair-count auditing.
- EOS-domain query clamps: gated at EOS query sites and reserved for Wave C.
- Non-positive cell volume: remains a remap failure and is not hidden by
  `fmax(vol, 1e-30)` in per-material mode.

##### 3.3.14.7 Per-material EOS inverse + cell-mean projection (Stage 32a Wave C)

When `numerics.materials.per_material_conservation_enabled = true`, the legacy
scalar material-0 EOS reclosure is replaced by a per-material refresh launcher.

For each cell \(c\) and present material \(m\)
\((\texttt{volfrac}[c,m] > \texttt{presence_threshold})\):

\[
\rho_m =
\frac{\texttt{mass\_per\_material}[c,m]}
     {\texttt{volfrac}[c,m]\,\texttt{vol}[c]},
\quad
e_{e,m} =
\frac{\texttt{Ee\_per\_material}[c,m]}
     {\texttt{mass\_per\_material}[c,m]},
\quad
e_{i,m} =
\frac{\texttt{Ei\_per\_material}[c,m]}
     {\texttt{mass\_per\_material}[c,m]} .
\]

- \(T_{e,m}, P_{e,m}, c_{v,e,m}, c_{s,e,m}\) are computed with
  `device_inverse_reclose_with_low_density_extrap(electron_view(m), rho_m,
  e_e_m, Te_floor, Zbar_m, A_m, low_density_extrap)`.
- \(T_{i,m}, P_{i,m}, c_{v,i,m}, c_{s,i,m}\) are computed with the ion
  EOS view using the same low-density extrapolation policy.
- The per-material sound speed projection input is
  \(c_{s,m}=\sqrt{c_{s,e,m}^2+c_{s,i,m}^2}\), where the electron and ion
  terms use `device_eos_sound_speed`.

Cell-mean projections are:

\[
\texttt{state.Te}[c] =
\frac{\sum_m \texttt{mass\_per\_material}[c,m]\,T_{e,m}}
     {\texttt{mass}[c]},
\quad
\texttt{state.Ti}[c] =
\frac{\sum_m \texttt{mass\_per\_material}[c,m]\,T_{i,m}}
     {\texttt{mass}[c]} .
\]

\[
\texttt{state.Pe}[c] = \sum_m \texttt{volfrac}[c,m]\,P_{e,m},
\quad
\texttt{state.Pi}[c] = \sum_m \texttt{volfrac}[c,m]\,P_{i,m}.
\]

\[
\texttt{state.cs}[c] = \max_{\text{present }m} c_{s,m}
\]

\[
\texttt{state.zbar}[c] =
\frac{\sum_m \texttt{mass\_per\_material}[c,m]\,\bar Z_m}
     {\texttt{mass}[c]},
\quad
\texttt{state.cv\_e}[c] =
\frac{\sum_m \texttt{mass\_per\_material}[c,m]\,c_{v,e,m}}
     {\texttt{mass}[c]},
\quad
\texttt{state.cv\_i}[c] =
\frac{\sum_m \texttt{mass\_per\_material}[c,m]\,c_{v,i,m}}
     {\texttt{mass}[c]} .
\]

\[
\texttt{state.ee}[c] =
\frac{\sum_m E_{e,c,m}}{\texttt{mass}[c]},
\quad
\texttt{state.ei}[c] =
\frac{\sum_m E_{i,c,m}}{\texttt{mass}[c]} .
\]

The \(T_e,T_i,\bar Z,c_v\) projections are mass-weighted. \(P_e,P_i\) are
volume-weighted intensive projections. The sound speed uses max-over-present
materials for CFL conservatism (Stage 32a Lock #19). The scalar `ee` and `ei`
fields are reductions of the authoritative extensive per-material energies,
not independent state in per-material mode. In Wave C, the device
projection sources \(\bar Z_m\) from fixed material-\(\bar Z\) configuration
when provided, otherwise from `materials.mat_defs[m].Z`; extending this path
to device-resident tabular/Thomas-Fermi per-material \(\bar Z\) is deferred to
the future operator that enables those models in per-material mode.

Lazy cache (`numerics.materials.lazy_cache_te_m_enabled`, default off):

- `Te_per_material[c,m]` / `Ti_per_material[c,m]` are memoized.
- `Te_per_material_valid[c,m]` / `Ti_per_material_valid[c,m]` flags are copied
  to temporary device mirrors for kernel read/write and copied back after a
  successful refresh.
- Cache entries are invalidated at every
  \(E_{e,m}\), \(E_{i,m}\), mass, volume-fraction, or volume mutation site.
- Cache values must be identical to recomputation. The cache is a pure
  performance optimization and never changes EOS semantics.
- Wave D Part 2 profiling (§4.1.4) keeps this default off because the
  baseline median EOS-refresh/step ratio did not exceed 10% in any measured
  regime (Q-XAI-1).

Wave D/F operators that mutate per-material energy, mass, volume fraction, or
volume must invalidate affected entries and invoke
`refresh_per_material_derived_cell_fields()` after their mutations to keep
cell-mean projections coherent.

##### 3.3.14.8 V22 restart/output contract and diagnostics (Stage 32a Wave E)

When `Numerics.materials.per_material_conservation_enabled=True`, HDF5 output
writes the authoritative Wave F conserved state under
`/hydro/per_material/v1/`:

- `mass[N_cell,N_mat]` in [g]
- `Ee[N_cell,N_mat]` in [erg]
- `Ei[N_cell,N_mat]` in [erg]

The layout is cell-major (`idx = c*N_mat + m`), `schema_version` remains 1,
and the group attributes record `conserved_basis="extensive"`,
`layout="cell_major_ncells_nmat"`, `material_names`, and `material_ids`.
Disabled mode omits `/hydro/per_material/v1/` entirely.  If
`Numerics.materials.hdf5_emit_derived_per_material=True`, diagnostic-only
derived arrays are also emitted:
`rho_derived`, `ee_derived`, `ei_derived`, `Te_derived`, `Ti_derived`, and
`Pe_derived`.  These arrays are not restart-authoritative; restart reads only
`mass`, `Ee`, and `Ei`, then rebuilds derived cell fields.

The cell-mean datasets `/hydro/{rho,ee,ei,Te,Ti,Pe,Pi,zbar,cv_e,cv_i,cs}` carry
`derived_projection_when_per_material_enabled=1` when Wave F mode is active.
This attribute tells readers that these cell means are projections of the
per-material extensive state rather than independent conserved quantities.

At output/history sample time the conservation residual diagnostic computes
cellwise maxima:

\[
\max_c \left|\sum_m M_{c,m} - M_c\right|,
\quad
\max_c \frac{\left|\sum_m M_{c,m} - M_c\right|}{|M_c|},
\]

and analogously for \(E_e\) against \(M_c e_{e,c}\) and \(E_i\) against
\(M_c e_{i,c}\).  The diagnostics are written under
`/diagnostics/conservation/v1/` as
`per_material_{mass,Ee,Ei}_max_{abs,rel}_residual`.  A relative residual above
`conservation_residual_warn_threshold_rel` (default \(10^{-12}\)) logs a
WARNING and increments `conservation_drift_warnings`; above
`conservation_residual_hard_warning_threshold_rel` (default \(10^{-10}\)) logs
a HARD WARNING.  No hard-fail threshold is defined in Stage 32a; empirical
calibration is deferred to Stage 33+.

Per-material event counters are cumulative and written under
`/diagnostics/per_material/v1/`: EOS table validity violations, absent-material
presence events, conservation drift warnings, lazy-cache invalidations, and
lazy-cache hit/miss counts.  `/metadata/dispatch_counters/` writes the
regression-hash inputs `per_material_kernel_call_count`,
`eos_inverse_call_count`, `mixture_projection_call_count`,
`lazy_cache_te_m_hit_count`, and `lazy_cache_te_m_miss_count`; disabled mode
forces all five persisted values to zero and hashes the all-zero vector.

The V22 reader applies monotonic chained migration.  V20/V21 files and V22
files without an enabled `/hydro/per_material/v1/` group are treated as
per-material disabled.  V22 enabled files load `mass/Ee/Ei` into
`State.mass_per_material`, `State.Ee_per_material`, and
`State.Ei_per_material`.  `Te_per_material_valid` and `Ti_per_material_valid`
are reset to false on restart; lazy cache values are never checkpoint
authoritative.

##### 3.3.14.9 production_comparable Wave F gate structure (Stage 32a Wave E)

The Stage 32b empirical rerun will classify Wave F outcomes with the following
seven `production_comparable` criteria:

1. PLIC enabled.
2. Wave F per-material conservation enabled.
3. The run reached `t_end`.
4. Final ALE provenance is `PUBLIC_BASELINE`.
5. Class-(d) aggregate is `none` or `soft_only`.
6. PLIC reconstruction success rate, excluding axis-exempt cells, is at least
   0.999.
7. The maximum per-material conservation relative residual is at most
   \(10^{-10}\).

The code-level enum is
`PASS`, `PARTIAL_A_CR_PROGRESSION`, `PARTIAL_B_PRODUCTION_RESIDUAL`,
`INCONCLUSIVE`, `FAIL`, and `DISABLED`.  Stage 32a Wave E wires the criteria
and enum only; Stage 32b supplies the multi-CF empirical data and applies the
final route.

##### 3.3.14.10 Wave F t=0 per-material initialization projection (Stage 32b Wave F)

For fresh deck initialization, or for V20/V21 legacy restart files that lack
`/hydro/per_material/v1/`, Wave F initializes the authoritative per-material
conserved arrays from the already-constructed cell-mean state by a volume-
fraction fan-out:
\[
M_{c,m}=f_{c,m}M_c,\qquad
E_{e,c,m}=f_{c,m}M_c e_{e,c},\qquad
E_{i,c,m}=f_{c,m}M_c e_{i,c}.
\]

With `Geometry(..., enforce_sum_to_one=True)`, the stored volume fractions
satisfy \(\sum_m f_{c,m}=1\), so the projection gives exact cellwise mass
closure,
\[
\sum_m M_{c,m}=M_c,
\]
with no additional normalization.  Pure cells therefore assign the full cell
mass and energy to the present material; empty or zero-mass cells keep zero
per-material conserved state.

In mixed cells this projection assumes the initial cell-mean \(\rho\),
\(e_e\), and \(e_i\) apply to every material at \(t=0\).  This is a deliberate
initialization approximation: material-specific evolution diverges after
startup through the per-material EOS, conduction, \(Q_{ei}\), artificial
viscosity, radiation, and remap operators.  A future Stage 33+ deck extension
may provide explicit \(\rho_m\), \(e_{e,m}\), and \(e_{i,m}\) callables to
replace this fan-out when physically resolved material thermodynamic profiles
are available.

V22 restarts with complete enabled per-material data do not apply this
projection; the checkpoint arrays remain authoritative.  A V22 checkpoint with
an enabled per-material group missing any of `mass`, `Ee`, or `Ei` is treated
as corrupt for Wave F enabled restart and must fail rather than silently
falling back to cell-mean fan-out.

### 3.4 1D_SPH pure Lagrangian hydro and optional V3 ALE

既定では 1D_SPH は pure Lagrangian であり、旧 `numerics.ale` による 1D rezone/remap は実行しない。球対称 1D mesh はセル tangling を起こさないため、従来の step 終端 mesh relaxation は不要である。一方、V3 の `numerics.ale1d` は solution-adaptive resolution を目的とする別経路で、既定無効、1D_SPH + deterministic radiation のみを対象にする。

#### 3.4.1 1D 方針

通常の 1D 球対称シミュレーションでは pure Lagrangian を使用する。旧 ALE rezoner は cell tangling のない 1D 球対称では不要であり、shock smearing の原因となるため削除された (ALE-FIX-1, 2026-04-26)。

したがって `numerics.ale.enabled` は 2D_RZ ALE のみを制御する。1D_SPH では `mesh.motion="lagrangian"` を用い、`mesh.motion="ale"` は許可しない。旧 1D 専用 ALE の設定面および実装面は削除された。

V3 solution-adaptive ALE は `numerics.ale1d.enabled` でのみ制御する。これは旧 1D ALE の復活ではなく、feature sensor、monitor equidistribution、conservative remap、velocity projection を二相 commit で行う独立機能である。Week 6 では `apply_ale_1d` が 21-step data flow を実装し、sensor/rezone/remap/projection/diagnostics は scratch にのみ書き、hard 許容誤差を満たした場合だけ mesh・保存量・速度を commit する。

> **運用注意**：`Ale1dConfig::enabled=false` / `numerics.ale1d.enabled=False` が既定であり、1D ALE V3 は実験的な opt-in 機能として、long-pulse ablation front penetration や multi-shock systems など localized moving feature がある場合に限って検討する。
> 典型的な GXII short-pulse cases では ALE off の pure Lagrangian を推奨する。120J/6ns FLD 評価では中央収束領域が広く局在 feature ではないため、ALE による speedup は確認されていない。

#### 3.4.2 2D ALE との境界

2D_RZ の ALE rezone/remap は §3.3 のまま維持する。2D production runs では `mesh.motion="ale"` かつ `numerics.ale.enabled=True` の場合に、2D Winslow rezone、保存的 remap、node velocity projection、EOS reclosure を実行する。この post-remap reclosure は `HydroEOSContext` を受け取り、table-backed EOS では `sn_material_newton_gpu` と同じ `Materials.low_density_extrapolation` policy（below-table analytic extrapolation / table-edge clamp）を用いる。EOS context がない、または該当成分に table がない場合は従来の理想気体 reclosure を維持する。

#### 3.4.3 V3 1D solution-adaptive ALE sensors

Week 2 の sensor は device resident な 1D cell/node fields から `Ale1dFeature` を作る。field 全体を host へコピーせず、max/sum/argmax と component 統計は GPU reduction で求め、host へ戻すのは feature 個数に比例する小さなメタデータのみである。半径は cm、温度は eV、密度は g/cc、圧力・人工粘性は erg/cc の cgs+eV 規約を維持する。

共通幾何：
\[
\Delta r_i = \max(r_{i+1}-r_i,\;10^{-14}\max(|r_N|,10^{-30}))
\]
\[
r_{c,i} = \frac{3}{4}\frac{r_{i+1}^4-r_i^4}{r_{i+1}^3-r_i^3}
\]
ただし分母が非正の場合は \((r_i+r_{i+1})/2\) を使う。cell mass coordinate は
\[
x_i = \frac{\sum_{k<i}m_k + m_i/2}{\sum_k m_k}, \qquad
\Delta x_i = \frac{m_i}{\sum_k m_k}.
\]
log gradient は \(D_r\ln f_i\) を cell centroid 上の centered stencil（境界は片側）で評価し、
\[
\ln f \leftarrow \ln\max(f,\max(f_{\rm abs}, f_{\rm rel} f_{\max}))
\]
を用いる。floors は \(\rho: (10^{-12},10^{-12})\), \(T_e,T_i:(5\times10^{-2},10^{-12})\), \(P_e+P_i:(10^{-30},10^{-12})\), \(q_{\rm visc}:(10^{-30},10^{-12})\), laser deposition rate \((10^{-99},10^{-30})\), volume fraction \(10^{-12}\)。

Laser absorption sensor uses deposited energy per step:
\[
\dot e_{L,i}=\max\left(\frac{\texttt{laser\_dep}_i}{V_i\Delta t},0\right),\quad
g_{L,i}=\min(1,\Delta r_i |D_r\ln \dot e_L|_i)
\]
\[
S_{L,i}=\sqrt{\frac{\dot e_{L,i}}{\dot e_{L,\max}+\epsilon}}
\left(0.5+0.5g_{L,i}\right).
\]
If \(\dot e_{L,\max}=0\), no laser feature is emitted. Otherwise the peak component threshold is \(0.35S_{L,\max}\). The raw width is \(\max(\sigma_{\rm mom,r},0.5L_L)\), clipped to \(4\Delta r_{\rm peak}\ldots16\Delta r_{\rm peak}\). Confidence uses
\[
N_{\rm eff,L}=\frac{(\sum_i\dot e_{L,i})^2}{\sum_i\dot e_{L,i}^2+\epsilon},\quad
c_L=\operatorname{smoothstep}\left(1-\frac{N_{\rm eff,L}}{N};0.10,0.40\right).
\]

Ablation/conduction-front sensor:
\[
H_{\rho,i}=\frac{1+\tanh((\rho_i/\rho_{\rm ref}-\rho_{\rm gate})/w_\rho)}{2},\quad
H_{T,i}=\operatorname{smoothstep}(T_{e,i};0.5,2.0)
\]
\[
S_{A,i}=H_{\rho,i}H_{T,i}
\sqrt{\frac{T_{e,i}}{T_{e,\max}+\epsilon}}
\min(1,4\Delta r_i |D_r\ln T_e|_i).
\]
The peak threshold is \(0.40S_{A,\max}\), width clips to \(3\Delta r_{\rm peak}\ldots14\Delta r_{\rm peak}\), and \(c_A=\operatorname{smoothstep}(S_{A,\max};0.10,0.35)\).

Shock sensor:
\[
\Delta u_i=v_{r,i+1}-v_{r,i},\quad
\chi_i=\frac{\max(0,-\Delta u_i)}{c_{s,i}+10^{-30}},
\]
\[
Q_i=\frac{q_{{\rm visc},i}}{\max(P_{e,i}+P_{i,i},0)+P_{\rm floor}},\quad
P_{\rm floor}=\max(10^{-30},10^{-12}P_{\max}),
\]
\[
S_{S,i}=\frac{Q_i}{Q_i+0.05}\frac{\chi_i}{\chi_i+0.05}.
\]
The peak threshold is \(0.35S_{S,\max}\), width clips to \(2\Delta r_{\rm peak}\ldots8\Delta r_{\rm peak}\), and
\[
c_S=\operatorname{smoothstep}(Q_{\max};0.03,0.10)
\operatorname{smoothstep}(\chi_{\max};0.03,0.15).
\]

Material interface sensor is face-based:
\[
J_j=\frac{1}{2}\sum_m |f_{j,m}-f_{j-1,m}|,\quad j=1,\ldots,N-1.
\]
Local maxima with \(J_j\ge0.05\) and separation at least 4 faces are emitted up to `max_features`. Width is \(2\Delta r_I\) clipped to the configured interface bounds; confidence is \(\operatorname{smoothstep}(J_j;0.05,0.25)\). If `pin_interfaces=True`, emitted interface features are pinned faces.

Center/hotspot sensor always emits when enabled, anchored at \(x=0,r=0\), with \(c_C=1\). Its default target width is \(\sigma_x=n_C/(3N)\) and \(\sigma_r\approx8\Delta r_0\), clipped to \(6\Delta r_0\ldots20\Delta r_0\). The diagnostic center activity signal is
\[
S_{C,i}=\exp[-(x_i/x_{\rm search})^2]\left[
0.5\frac{P_i}{P_{\max}+\epsilon}
+0.25\frac{T_{e,i}}{T_{e,\max}+\epsilon}
+0.25\frac{T_{i,i}}{T_{i,\max}+\epsilon}\right].
\]

#### 3.4.4 V3 1D monitor and scratch rezone

The rezone stage builds a cell monitor in normalized mass coordinate \(x\) and computes candidate node radii in scratch buffers only. For each emitted feature \(k\),
\[
G_{k,i} =
\begin{cases}
\exp[-\frac{1}{2}((x_i-x_k)/\sigma_{x,k})^2],
& |x_i-x_k|\le 3\sigma_{x,k},\\
0, & \text{otherwise},
\end{cases}
\qquad S_k=\sum_i G_{k,i}\Delta x_i.
\]
Features with \(S_k<10^{-14}\) are suppressed. Active budgets are
\[
\tilde n_k=c_k n_k,\qquad
B_{\max}=(1-f_{\rm floor,min})N,\qquad f_{\rm floor,min}=0.55,
\]
with a common cap \(\alpha_B=\min(1,B_{\max}/B_{\rm active})\). The floor budget is
\[
n_{\rm floor}=N-\sum_k \alpha_B\tilde n_k-n^*_{\rm sp}.
\]
For active feature kernels,
\[
A_k=\frac{\alpha_B\tilde n_k}{n_{\rm floor}S_k}.
\]

The optional spatial monitor applies only to laser absorption, ablation-front, and shock features:
\[
U_i \mathrel{+}= c_k
\exp[-\tfrac{1}{2}((x_i-x_k)/(2\sigma_{x,k}))^2]
\left(\max(1,\Delta r_i/\Delta r_{{\rm target},k})^p-1\right),
\qquad p=2.
\]
\(\Delta r_{{\rm target},k}\) is \(\sigma_{r,k}/3\) clipped to the feature-kind-specific cgs bounds in `Numerics.ale1d.rezone`. If \(S_{\rm sp}=\sum_iU_i\Delta x_i\ge10^{-14}\), the spatial monitor receives \(n^*_{\rm sp}\) from the same active-budget cap and contributes \(A_{\rm sp}U_i\), where \(A_{\rm sp}=n^*_{\rm sp}/(n_{\rm floor}S_{\rm sp})\).

The final monitor is
\[
W_i=W_0+\sum_k A_kG_{k,i}+A_{\rm sp}U_i,\qquad W_0=1,
\]
smoothed for two protected-aware passes:
\[
W_i^{new}=\frac{2W_i+l_iW_{i-1}+r_iW_{i+1}}{2+l_i+r_i}.
\]
\(l_i=0\) across pinned face \(i\), \(r_i=0\) across pinned face \(i+1\), unless explicitly overridden by `monitor_smooth_across_protected_faces`; material-interface pinned faces therefore block smoothing by default. The monitor is clipped to \(W_0\le W_i\le50W_0\). There is no \(W_{\min}<W_0\).

The common node mask pins node 0, pins node \(N\) only for `Numerics.hydro.boundary_1d="fixed"`, and pins the nearest face of each material-interface feature with `pinned_face=True`. The protected fraction is \(N_{\rm pinned}/(N+1)\).

Equidistribution is CPU-side and segment-wise between pinned nodes, with domain endpoints included as segment endpoints. For segment \([j_{\rm lo},j_{\rm hi}]\),
\[
I_W=\sum_{i=j_{\rm lo}}^{j_{\rm hi}-1}W_i\Delta x_i,
\]
and interior node \(j\) is placed where the cumulative monitor integral equals
\[
\frac{j-j_{\rm lo}}{j_{\rm hi}-j_{\rm lo}}I_W.
\]
The resulting mass-coordinate location is mapped back to radius by linear interpolation through the old \((x_j,r_j)\) node map. Candidate displacement is capped by scaling all node displacements if either normalized mass-coordinate or radius displacement exceeds the configured fractions. The rezone stage returns only `r_candidate`; conservative remap and commit are handled by later driver stages.

If the smoothed/clipped monitor is uniform to roundoff, the scratch rezone is the identity candidate after eligibility checks. This preserves the no-feature path exactly and avoids changing resolution when \(W_i=W_0\) everywhere.

#### 3.4.5 V3 1D MUSCL/minmod remap and velocity projection

The 1D_SPH ALE remap is conservative in spherical volume coordinate
\[
Y(r)=\frac{4\pi}{3}r^3,\qquad V_i=Y_{i+1}-Y_i.
\]
For a candidate node \(r_j^n\), the face swept volume is
\[
\delta Y_j=Y(r_j^n)-Y(r_j^o).
\]
If \(\delta Y_j>0\), the donor is cell \(j\); if \(\delta Y_j<0\), the donor is cell \(j-1\); and if \(\delta Y_j=0\), the face flux is zero. For any conserved extensive \(Q_i\) with old cell density \(q_i^o=Q_i^o/V_i^o\),
\[
F_j^{(1)}=\delta Y_j q_{\mathrm{donor}}^o,\qquad
Q_i^n=Q_i^o+F_{i+1}^{(1)}-F_i^{(1)}.
\]
Pinned faces require \(r_j^n=r_j^o\) and use \(F_j=0\). Multi-cell sweeps are rejected when `Numerics.ale1d.remap.reject_multicell_sweeps=True`: outward sweeps require \(Y_j^n\le Y_{j+1}^o\), and inward sweeps require \(Y_j^n\ge Y_{j-1}^o\). A violation returns `CandidateInvalid`; the caller discards scratch state.

For `Numerics.ale1d.remap.high_order_enabled=True` (default from Week 5), the second-order candidate reconstructs \(q\) in \(Y\). Cell centers are
\[
Y_{c,i}=\frac{1}{2}(Y_i+Y_{i+1}),
\]
and the generalized minmod slope with \(\theta=\) `limiter_theta` is
\[
s_i=\operatorname{minmod}\left(
\theta\frac{q_i-q_{i-1}}{Y_{c,i}-Y_{c,i-1}},
\frac{q_{i+1}-q_{i-1}}{Y_{c,i+1}-Y_{c,i-1}},
\theta\frac{q_{i+1}-q_i}{Y_{c,i+1}-Y_{c,i}}
\right).
\]
Boundary cells, cells adjacent to pinned/protected faces, and segment-edge cells use \(s_i=0\). The slope is then scaled so the reconstructed values at \(Y_i\) and \(Y_{i+1}\) lie inside \([\min(q_{i-1},q_i,q_{i+1}),\max(q_{i-1},q_i,q_{i+1})]\). For donor \(d\),
\[
\bar Y_j=\frac{1}{2}(Y_j^o+Y_j^n),\qquad
F_j^{(2)}=\delta Y_j\left(q_d+s_d(\bar Y_j-Y_{c,d})\right).
\]

Protected faces use a cosine taper. Let \(d_j\) be the integer distance from face \(j\) to the nearest protected face, including pinned/boundary faces. With \(N_{\rm ramp}=\) `high_order_ramp_cells`,
\[
\phi_j=\begin{cases}
0, & d_j=0,\\
\frac{1}{2}\left[1-\cos(\pi d_j/N_{\rm ramp})\right], & 0<d_j<N_{\rm ramp},\\
1, & d_j\ge N_{\rm ramp}.
\end{cases}
\]
The final flux is
\[
F_j=F_j^{(1)}+\phi_j(F_j^{(2)}-F_j^{(1)}).
\]
For \(N_{\rm ramp}=2\), \(\phi(0)=0\), \(\phi(1)=0.5\), and \(\phi(d\ge2)=1\). Radiation may use `radiation_high_order_ramp_cells`; the default is also 2. At protected faces \(\phi=0\) selects the first-order donor flux, not zero flux unless the face is pinned.

After the high-order attempt, each field is checked for positivity and local boundedness. Mass, material component mass, and radiation are checked as densities; electron/ion energies are checked as recovered specific energies after the conservative energy and mass remaps. If a field fails and `fallback_to_first_order_on_bounds_fail=True`, that field is recomputed with \(\phi=0\). For electron/ion energy fallback, the recovered specific energy uses a mass denominator from the same first-order donor remap; if the high-order mass candidate had passed its own bounds, mass is recomputed with \(\phi=0\) before the specific-energy fallback is accepted. If the fallback still fails, remap returns `ConservationRejected`. Scratch diagnostics record the cells/fields that required fallback. `high_order_enabled=False` directly selects the Week 4 first-order donor path.

The remapped conserved quantities are mass \(m_i\), material masses \(m_if_{i,m}\), electron and ion material energies \(m_ie_{e,i}\), \(m_ie_{i,i}\), and radiation group energies \(V_iE_{g,i}\). Scratch outputs store \(m_i^n\), \(V_i^n\), normalized \(f_{i,m}^n\), specific \(e_{e,i}^n,e_{i,i}^n\), and radiation energy densities \(E_{g,i}^n\). This function is a two-phase operation: it reads `State` and `r_candidate`, writes only `Ale1dRemapScratch`, and does not mutate mesh or physics state. The Week 6 driver then projects velocity into `Ale1dVelocityProjectScratch`, computes hard diagnostics, and only on acceptance commits `x_r`, mass, energies, radiation energy density, volume fractions, and nodal velocity, followed by boundary application, geometry/rho refresh, EOS reclosure, and transient source/viscosity invalidation.

Velocity projection is also two-phase. From old nodal velocities and old cell masses,
\[
u_i^o=\frac{1}{2}(v_i^o+v_{i+1}^o),\qquad p_i^o=m_i^o u_i^o.
\]
The already remapped mass \(m_i^n\) is reused. Cell momentum \(p_i^o\) is remapped with the same swept volumes, donors, and mass \(\phi_j\), then
\[
u_i^n=p_i^n/m_i^n.
\]
Interior node velocities are mass-weighted,
\[
v_j^n=\frac{m_{j-1}^n u_{j-1}^n+m_j^n u_j^n}{m_{j-1}^n+m_j^n},\qquad 1\le j<N,
\]
with \(v_0^n=0\) at the spherical center and \(v_N^n=u_{N-1}^n\) before the boundary kernel is applied. The diagnostic kinetic energy drift uses
\[
K=\sum_j\frac{1}{2}m_j^{node}v_j^2,\qquad
m_j^{node}\approx\frac{1}{2}(m_{j-1}+m_j),
\]
omitting missing neighbor masses at the domain ends, and reports \(|K^n-K^o|/K^o\).
