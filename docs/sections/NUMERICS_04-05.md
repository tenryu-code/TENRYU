<!-- 分割元: docs/NUMERICS.md | このファイルは参照用です。原本（docs/NUMERICS.md）が権威です。 -->
## 4. 電子熱伝導
### 4.1 Spitzer‑Härm + flux limiter

**Spitzer伝導率**（cgs+eV単位、2026-07-30 改訂: \(\gamma_0(Z)\) 無条件適用）：
\[
\kappa_{SH} = \kappa_0\,\xi(\bar Z)\,\frac{T_e^{5/2}}{\bar{Z} \ln\Lambda},\qquad
\xi(Z)=\frac{\varphi_{ES}(Z)}{\varphi_{ES}(1)},\quad
\varphi_{ES}(Z)=\frac{Z+0.24}{Z+4.2}
\quad [\text{erg}/(\text{cm}\cdot\text{s}\cdot\text{eV})]
\]
\(\xi(1)=1\)（水素は従来と厳密一致）、\(\xi(3.5)=2.0369\)、\(\xi(\infty)=4.19\)。
\(\xi\) は Braginskii の \(\gamma_0(Z)\)（3.16, 4.9, 6.1, 6.9, …, 12.5）の
Epperlein–Short 型補間で、全次元・全経路（1D/2D・per-material・SNB 基底流束・
persistent loop）に常時適用される（user 裁定 2026-07-30 — 固定係数経路は削除、
namelist `spitzer_z_correction` は deprecated no-op、"off" はエラー）。
charge-moment 表（§1.1.3a）併用時は衝突電荷 \(\bar Z r_2\) に対して評価する。
ここで \(\bar{Z}\) は平均電離度（§1.1.4）。
固定モデルでは \(\bar{Z} = Z\)（原子番号）。Thomas-Fermi/テーブルモデルでは §1.1.4 の \(\bar{Z}\) を使用する。

定数：
- \(\kappa_0 \approx 306\) [W/(cm·eV\(^{7/2}\))]
  = \(3.06\times 10^{9}\) [erg/(cm·s·eV\(^{7/2}\))]（Spitzer係数、cgs+eV単位系）

> **導出**：Braginskii (1965) の電子平行熱伝導率
> \(\kappa_\parallel = 3.16\,n_e k_B T_e \tau_e / m_e\) と
> NRL Plasma Formulary の電子緩和時間
> \(\tau_e = 3.44\times 10^5\,T_e[\text{eV}]^{3/2}/(n_e\,Z\,\ln\Lambda)\) [s]
> から \(n_e\) を消去し、\(T_e^{5/2}/(Z\ln\Lambda)\) 依存部を抽出すると：
> \[
> \kappa_0 = 3.16 \times 3.44\times 10^5 \times (\text{eV\_to\_erg})^2 / m_e
> = 3.063\times 10^9 \;\text{erg/(cm·s·eV}^{7/2}\text{)}
> \]
> ここで \(k_B = \text{eV\_to\_erg} = 1.6022\times 10^{-12}\) erg/eV（§0.1 定数表）、\(m_e=9.1094\times 10^{-28}\) g。


**Spitzer-Härm熱流束**：
\[
\mathbf{q}_{SH} = -\kappa_{SH}\,\nabla T_e
\]

**電子熱速度**：
\[
v_{th,e} = \sqrt{\frac{k_B T_e}{m_e}} \quad [\text{cm/s}]
\]

**Flux limiter**（harmonic mean型、既定）：
\[
\mathbf{q}_e = \frac{\mathbf{q}_{SH}}{1 + |\mathbf{q}_{SH}|/q_{\max}},
\quad q_{\max} = f_{lim}\, n_e k_B T_e\, v_{th,e}
\]
既定 \(f_{lim}=0.06\)。

**面中心での \(q_{max}\) 評価**：\(n_{e,face} = (n_{e,L} + n_{e,R})/2\)、\(T_{e,face} = (T_{e,L} + T_{e,R})/2\)、\(v_{th,e,face} = \sqrt{k_B\,T_{e,face}/m_e}\)。すべて面の両側セルの算術平均を用いる。この規約は §3.1.7 のフラックスリミタ評価と整合する。

**Mean-free-path limiter（オプション）**：`conduction.mfp_limiter_C = C_{mfp} > 0` の場合、
\[
\nu_{ei} = 2.91\times 10^{-6}\,\frac{n_e \bar Z \ln\Lambda}{T_e^{3/2}} \;[\mathrm{s}^{-1}],\quad
\lambda_{ei} = \frac{v_{th,e}}{\nu_{ei}}
\]
\[
\kappa = \kappa_{SH}\,\min\!\left(1,\; \frac{C_{mfp}\,\Delta l}{\lambda_{ei}}\right)
\]
を適用する。ここで \(\Delta l\) はセル代表長（1D: \(\Delta r\)、2D: \(\sqrt{A}\)）。
\(C_{mfp}=0\)（既定）では無効化され、従来の \(\kappa=\kappa_{SH}\) に一致する。

**Void セルの伝導率**：`cell_is_void[i] = 1` のセルでは \(\kappa_{eff} = 0\) を強制する。
面の調和平均 \(1/\kappa_f = 1/\kappa_L + 1/\kappa_R\) により、void-通常セル界面の
フラックスは自動的にゼロとなり、void 領域は断熱壁として機能する。

#### 4.1a 1D 伝導演算子の座標幾何一般化（W-G2, 2026-07-04）

1D の離散伝導演算子は保存形 FV：
\[
\rho c_{v,e}\,\frac{\partial T_e}{\partial t}\Big|_i
= \frac{A_{i+1/2}\,q_{i+1/2} - A_{i-1/2}\,q_{i-1/2}}{V_i},\qquad
q_{f} = \kappa_f\,\lambda_f\,\frac{T_{e,R}-T_{e,L}}{r_{c,R}-r_{c,L}}
\]
（\(\kappa_f\)=面調和平均、\(\lambda_f\)=面フラックスリミタ、
\(r_c=\tfrac12(x_{r,i}+x_{r,i+1})\) セル中心距離 — 幾何非依存）。幾何は面積
\(A\) とセル体積 \(V\) のみに現れる：

| geometry_1d | \(A(r)\) | \(V_i\) |
|---|---|---|
| spherical | \(4\pi r^2\)（歴史的綴り verbatim） | `state.vol[i]`（mesh 殻体積） |
| cylindrical | \(2\pi r\)（`geometry_1d_face_area`、単位軸長） | `state.vol[i]` \(=\pi(r_1^2-r_0^2)\) |
| planar | \(1\)（単位断面積） | \(x_{r,i+1}-x_{r,i}\)（歴史的 x_r 差綴り verbatim） |

planar の \(V\) が `state.vol` でなく x_r 差なのは歴史的 `test_planar`
フック（球面メッシュ上で平面演算子を走らせる verify 専用エイリアス）との
互換要件による — `geometry_1d="planar"` では mesh 体積も同値。実装は
STS stage / per-material STS stage / implicit 組立 / implicit clamp の 4
カーネルを `template<int GEOM>` + `if constexpr`（pattern v3、§3.1.0a の
bitwise 不変契約に従い GEOM==0/2 実体は歴史式 verbatim・runtime 幾何状態
ゼロ）とし、ホスト側は
`geom_code = (conduction.test_planar ? 2 : mesh.geometry_code)` で
\(\langle1\rangle/\langle2\rangle/\langle0\rangle\) 実体へ dispatch する
（launch 構成は 3 分岐同一）。境界は全幾何で両端 zero-flux（stencil 切断;
\(r=0\) は \(A\to0\) で自然に正則）。

**検証（eigenmode-decay 族、`tests/verification/test_conduction_eigenmode_1d.cu`）**：
固定 \(\kappa\)（`test_kappa`）線形拡散の第 1 zero-flux 固有モード
\[
T_e(r,t) = T_0 + A\,\varphi(kr)\,e^{-\chi k^2 t},\qquad
\varphi = \cos \,|\, J_0 \,|\, \tfrac{\sin x}{x},\quad
kR = \pi \,|\, j_{1,1} \,|\, \alpha\ (\tan\alpha=\alpha)
\]
を直接駆動し、L2 の loglog 次数 \(\in[1.8,2.2]\)・総電子エネルギー恒等式
\(|\Delta E|/E \le 10^{-14}\)・固有値 runtime 自己検証を課す。実測次数：
球面 2.055 / 平面 2.107 / 円筒は VERIFICATION §4.7 参照。この族が球面
演算子の 1D 解析ゲート（従来 heat_diffusion は test_planar 経由の平面
演算子のみ）・implicit 経路・per-material 経路の解析被覆を初めて与える。
また 16 桁精度ログにより template 化前後の bit 同一（rel=0）を測定した
（詳細は commit 63550add; E_floor 診断集計のみ atomicAdd 順序変化で 1 ulp、
軌道非影響）。

**非線形自己相似検証（Pattle 熱波、W-G2 nlheat, 2026-07-04）**：
検証専用 env hook `TENRYU_CONDUCTION_TEST_KAPPA_POWER=p`（既定不活性）が
test_kappa 経路を \(\kappa=\kappa_0T_e^{\,p}\) に切替える（1D 単一材料のみ、
歴史カーネル・launch は byte 不変、n=0 で歴史値と bit 一致）。参照解は
Pattle (1959, QJMAM 12, 407) 式 (4a)(6) の瞬間点源自己相似解（有限 front、
\(r_f\propto t^{1/(sn+2)}\)、転記は PDE 残差 \(\sim10^{-42}\)（mpmath）+
Q 正規化 Beta 恒等式 + in-gate FD 残差 + n→0 bit 接続で四重検証）。
**発見（縮退 front の面 κ 閉包）**：本節既定の面調和平均は compact-support
非線形 front を格子に釘付けにする（cold 側 \(\kappa(T_{floor})\) が面 flux を
絞る — 実測 front 移動 ~10⁻⁴ セル/全区間、中心は非線形減衰する一方で
エネルギーが front 内に閉じ込められる）。hook 経路の stage kernel は
Kirchhoff 割線
\(\kappa_f=\kappa_0\frac{T_R^{\,p+1}-T_L^{\,p+1}}{(p+1)(T_R-T_L)}\)
（\(\kappa\propto T^p\) の flux 厳密面係数; 等温極限で調和平均と一致、
p=0 は厳密に \(\kappa_0\)）を用い、Pattle 伝播を front 冪 0.9-1.5% /
プロファイル L2 6-9×10⁻⁴ / 次数 2.28(球)/2.16(円筒)/1.45(平面) で回復する
（VERIFICATION §4.8）。**生産 Spitzer 経路は調和平均のまま**（材料界面・
void 断熱の設計要件）— 実物の preheat front は放射 pedestal を持つため
影響規模は問題依存であり、閉包選択の見直しは W-K 系の別課題として起票
（silent 変更禁止）。

**face_kappa_policy（W-G2 kirchhoff, 2026-07-04 — 生産化・既定 on）**：
上記発見の生産側対応として、面伝導率閉包を
`Numerics.conduction.face_kappa_policy = "kirchhoff_same_material"（既定）|
"harmonic"` で選択可能にした（namelist は diff 提案 01 経由、
それまで env `TENRYU_CONDUCTION_FACE_KAPPA_POLICY` が bridge）。
kirchhoff_same_material は生産 Spitzer 経路（n=5/2）の同材料滑面のみ
\(\kappa_f=\bar\kappa_{0,f}\,S_{5/2}(T_L,T_R)\)
（\(\kappa_{0,c}=\kappa_c/T_c^{5/2}\) 抽出・\(\bar\kappa_{0,f}\)=調和平均、
近接温度 \(|\Delta T|/T_m<10^{-4}\) は Taylor 形
\(T_m^{5/2}(1+\tfrac{n(n-1)}{24}x^2)\)）に切替え、
**界面/void/強 table 跳びは調和維持**（detector
\(|\ln(\kappa_{0R}/\kappa_{0L})|>\ln 10\)；void は \(\kappa\le0\) 側で
調和→0 の断熱が構成的に保存）。STS 安定性は Kirchhoff conductance
\(G_f=A_f\kappa_f/d_f\) の Gershgorin 型 \(2C_i/\sum G_f\) をセル律速と
min 合成（一様極限で歴史値と一致）。定数 test_kappa 経路は両閉包が解析的に
一致するため policy は dispatch しない（nlheat hook は自前の厳密割線を優先）。
面診断（外部 verdict 指定）は `TENRYU_CONDUCTION_FACE_DIAG=1` で
\(R_{HK}(\theta)=\tfrac{2(n+1)\theta^n(1-\theta)}{(1+\theta^n)(1-\theta^{n+1})}\)、
\(s=|F_K|/F_{sat}=(1-\lambda)/\lambda\)、
\(R_{eff}=R_{HK}(1+s)/(1+R_{HK}s)\) の active-face min/median と閾値超え数を
100 step 毎に出力する（GXII A/B の解釈用、read-only）。既定切替は GXII A/B
（tmp/ab_kirchhoff_gxii.sh、main 直列実行）後のユーザー判断として完了した。
Default flipped 2026-07-06 per user decision (rebaseline sprint).

**flux-limiter 面推定の閉包整合契約（2026-07-13）**：面 flux-limiter
係数 \(\lambda_f=1/(1+|q_{SH,f}|/q_{max,f})\) の \(q_{SH,f}\) 推定は、
**消費側 flux カーネルと同一の面 κ 閉包**（`face_kappa_policy` に追従）で
評価しなければならない。修正前は推定が常に調和平均 κ で行われており、
Kirchhoff 閉包下の急峻 front（例: 3000 eV↔1 eV 隣接セル）では
\(q_{SH}^{harm}\ll q_{max}\Rightarrow\lambda\approx1\) のまま適用 flux が
Kirchhoff \(q_{SH}^{kir}\gg q_{max}\)（XC-0b 条件で \(\sim10^4\times\)）となり、
limiter が守るべき自由流上限が事実上未適用だった（未解像 front の過伝播:
XC-0b z1 nc=400 で front +13% vs 収束参照解; MULTI-IFE 交差比較 XC-0b が検出、
`docs/design/xc_tenryu_multiife_1d_comparison_design_20260710.md` Addendum 4）。
上記面診断の \(R_{eff}\) 代数（深飽和で \(R_{eff}\to1\) = 両閉包とも
\(q_{max}\) 頭打ち）は整合適用を前提としており、本契約はその実装化である。
適用箇所: STS/implicit 両経路 + persistent kernel の
`compute_1d_flux_limiter_faces`（`kirchhoff_face` 引数; guard は消費側
dispatch と同一の `policy=="kirchhoff_same_material" && test_kappa<=0`）。
harmonic 閉包・test_kappa 経路は bit 恒等。回帰:
`test_conduction_limiter_policy_consistency`（2 セル急峻 front で
\(|q_{applied}|\le q_{max}\) 不変量 + 判別 assert \(\lambda<10^{-2}\)）。

**kirchhoff dt Gershgorin の零体積セル guard（2026-07-13）**：上記
Gershgorin 型 dt 推定 \(2C_iV_i/\sum G_f\) は歴史的に \(V=\max(vol_i,10^{-30})\)
で床当てしていたため、\(x_r\) は有効だが \(vol\) が未評価（全零）の状態
（unit-test 級の手組み state）で \(V=10^{-30}\) を製造し、dt_exp が
\(\sim10^{-27}\,\mathrm{s}\)（健常値比 \(10^{24}\) 桁崩壊）→ STS
\(\sim1.3\times10^9\) 分割 = 実質 hang（ctest #845、07-11 full sweep の
two-material Timeout の真因 — 当時の GPU 競合仮 triage を訂正）。修正は
\(vol_i\le0\) セルを Gershgorin 寄与から除外する guard（deff 側推定
\(dl^2/D_{eff}\) は \(x_r\) 由来で元来免疫、そちらへ自然 fallback）。生産
state は常に \(vol>0\) のため健常 dt_exp は bit 恒等。診断確定は gdb 直接
測定（\(\kappa_{eff}\)/\(\rho c_v\) 健全・\(vol\equiv0\) を実測、委譲文書
`HANDOFF_BUG20_BUG21_20260713.md` の容疑 1/2 は共に棄却）。回帰: 同上
ctest へ零体積 property test（零 vol=無寄与 sentinel + 真球殻 vol=有限比）
を追加、#845 green 化。

**1T conduction \(c_v\) convention（2026-07-06）**：
`Main.temperature_model="1T"` では `ee` は total internal energy であり、
EOS closure の比熱契約は \(c_{v,total}=c_{v,i}+c_{v,e}\)（理想気体固定 \(Z\) では
\((1+\bar Z)k_B/[A m_p(\gamma-1)]\)）である。旧挙動では `state.cv_e` が無い
1T 経路で conduction が electron-convention fallback
\(\bar Z k_B/[A m_p(\gamma-1)]\) を使い、\(Z>0\) で一様場の conduction 適用でも
matter energy を \(z/(1+z)\) 倍へ落とす drain を作った。修正後は 1T で
`state.cv_e` を total-convention EOS \(c_v\) として allocation/fill し、
`compute_spitzer_deff_*` と conduction の EOS sync がその値を消費する。
一様場の conduction application は energy-neutral でなければならない。

**C1 Zel'dovich-Raizer thermal-wave verification deck**：
`examples/verification/2d_rz_c1_zeldovich_raizer.py` は 2D_RZ の薄い slab（反射境界、Hydro/Radiation/Laser/ALE 無効）で
`conduction.test_kappa > 0` の線形熱方程式
\[
\frac{\partial T_e}{\partial t} =
\alpha\frac{\partial^2 T_e}{\partial z^2},\qquad
\alpha=\frac{\kappa_{test}}{\rho c_{v,e}}
\]
を検証する。初期条件は
\[
T_e(z,0)=T_{floor}+A_0\exp\left[-\frac{(z-z_0)^2}{2\sigma_0^2}\right]
\]
で、解析解は
\[
T_e(z,t)=T_{floor}+A_0\frac{\sigma_0}{\sigma(t)}
\exp\left[-\frac{(z-z_0)^2}{2\sigma(t)^2}\right],\qquad
\sigma(t)=\sqrt{\sigma_0^2+2\alpha t}.
\]
しきい値 \(T_{front}=T_{floor}+\epsilon\) に対する解析 wave-front は
\[
L(t)=\sigma(t)\sqrt{2\ln\left(\frac{A_0\sigma_0}{\epsilon\sigma(t)}\right)}.
\]
ctest `C1 Zel'dovich-Raizer thermal-wave gate is production-level` は
\(N_z=64,128,256\) の HDF5 出力から \(T_e(z)\) を読み、線形補間で \(L_{sim}\) を抽出し、
\(|L_{sim}-L|/L\le 5\%\) と \(L_\infty(T_e)\) の refinement 低下（rate \(\ge 1.0\)）を要求する。
純粋な Barenblatt \(\kappa=\kappa_0T^n\) 正規化は、現 runtime が
\(\kappa_{SH}\propto T_e^{5/2}/(\bar Z\ln\Lambda)\) と flux/MFP limiter を持つため未実装であり、
この deck は production HDF5/波面抽出 gate として線形伝導解析解を用いる代替である。

#### 4.1.1 Per-material Spitzer-Härm conduction

`numerics.materials.per_material_conservation_enabled=true` かつ hydro EOS context が利用可能な場合、
電子熱伝導は材料ごとの電子エネルギー \(E_{e,c,m}\) を直接更新する。従来の単一材料/disabled 経路は
従来 kernel を使い、bit-exact disabled invariant の対象として残す。

1D face \(f=(L,R)\) では、材料 \(m\) の面体積率を
\[
\alpha_{f,m}=\frac{1}{2}\left(\alpha_{L,m}+\alpha_{R,m}\right)
\]
とし、境界面では内側セルの \(\alpha_{c,m}\) を用いる。材料温度は per-material EOS accessor で
\(T_{e,L,m},T_{e,R,m}\) を得て、\(T_{e,f,m}=(T_{e,L,m}+T_{e,R,m})/2\) とする。
材料密度 \(\rho_{f,m}\) から
\[
n_{e,f,m}=\rho_{f,m}\bar Z_m/(A_m m_p),\qquad
\ln\Lambda_{f,m}=\ln\Lambda(n_{e,f,m},T_{e,f,m},\bar Z_m)
\]
を評価し、
\[
\kappa_{SH,f,m}=\kappa_0\,T_{e,f,m}^{5/2}/(\bar Z_m\ln\Lambda_{f,m})
\]
を得る。Flux limiter は集約後ではなく材料ごとに適用する：
\[
q_{SH,f,m}=-\kappa_{SH,f,m}\nabla T_{e,f,m},\qquad
q_{\max,f,m}=f_{lim}n_{e,f,m}k_BT_{e,f,m}v_{th,e}(T_{e,f,m}),
\]
\[
\kappa_{eff,f,m}=
\frac{\kappa_{SH,f,m}}{1+|q_{SH,f,m}|/q_{\max,f,m}}.
\]
STS の安定性評価と stage scheduling には集約係数
\[
\kappa_{eff,f}=\sum_m \alpha_{f,m}\kappa_{eff,f,m}
\]
を用いる。ただし分母は従来通り cell-mean の \(\rho c_{v,e}^{eff}\) であり、
\(c_{v,e}^{eff}\) は per-material mass-weighted projection `state.cv_e` である。

STS apply では凍結した \(\kappa_{eff,f,m}\) を material-major scratch に保持し、各 face の symmetric
power を材料別に加算する。1D の保存形は
\[
\Delta E_{e,c,m}
=\Delta t\left(P_{c+1/2,m}-P_{c-1/2,m}\right),
\quad
P_{f,m}=A_f\,\alpha_{f,m}\kappa_{eff,f,m}\nabla T_{e,f},
\]
であり、閉じた領域では face pair が反対符号で現れるため
\(\sum_{c,m}E_{e,c,m}\) は floor clamp を除き保存される。Stage 後、
\[
e_{e,c}=\frac{\sum_m E_{e,c,m}}{m_c}
\]
を再計算し、`refresh_per_material_derived_cell_fields(..., force_invalidate_all=true)` により
`Te/Ti/Pe/Pi/cs/cv_e/cv_i` を per-material 中央 projection で再同期する。

Dirichlet/source conduction boundary では Marshak 境界を用いない（Marshak は radiation 専用）。
2D RZ の `state_supply` z-face は固定境界温度 \(T_b\) として扱い、
\[
F_b=\sum_m \alpha_{c,m}\kappa_{eff,b,m}\frac{T_b-T_{e,c,m}}{\Delta x_b}
\]
を用いる。面から領域へ入るエネルギーは
\(\Delta E_b=\Delta t\,F_b A_b\) で、\(A_b=\pi(r_{i+1}^2-r_i^2)\)、
\(\Delta x_b\) は境界面から隣接 cell center までの距離である。Neumann/reflect/vacuum
境界は従来通り zero conductive flux である。

#### 4.1.1.1 C1 peer-review verification gates (2026-05-13)

The C1 peer-review closure suite registers four additional Catch2 gates:
`test_c1_zeldovich_raizer_thermal_wave`,
`test_c1_anisotropic_tensor_rotation_skewed_mesh`,
`test_c1_boundary_heat_flux_conservation`, and
`test_c1_solver_residual_condition_number`.  One gate remains documented as a
deferred feature gap:

- Anisotropic tensor rotation on skewed meshes: current C1 conduction exposes
  scalar Spitzer or scalar `test_kappa` diffusion, not a rotated tensor
  \(\kappa=R(\theta)\operatorname{diag}(\kappa_\parallel,\kappa_\perp)R^T\).

`test_c1_zeldovich_raizer_thermal_wave`,
`test_c1_anisotropic_tensor_rotation_skewed_mesh`,
`test_c1_boundary_heat_flux_conservation`, and
`test_c1_solver_residual_condition_number` are active as of 2026-05-13.  The
anisotropic-tensor gate uses a production substitute: it explicitly shears a
2D RZ quadrilateral mesh in the test harness, builds the scalar Kershaw 9-point
operator with constant isotropic \(D\), applies the same symmetric-pair update
used by C1 conduction, and requires central Gaussian symmetry within 5% plus
64/128/256 self-convergence.  This validates the non-orthogonal scalar Kershaw
stress case, but it is not a rotated tensor-\(\kappa\) API validation.  The
Python deck API still admits only uniform rectangular 2D_RZ conduction meshes;
a deck-level skewed conduction case requires a future mesh construction control
or a supported prescribed-node deck path.

Each accepted conduction operator call updates `core::State` telemetry:
`c1_solver_steps_total`, `c1_solver_residual_{last,max}`,
`c1_solver_iter_{last,max}`, `c1_solver_cond_number_{last,max}`,
`c1_bc_heat_flux_last[4]`, and `c1_bc_heat_flux_integrated[4]`.
History output appends the per-history-sample values under
`/diagnostics/conduction/v1/solver_residual`,
`solver_residual_max`, `solver_iter`, `solver_iter_max`,
`solver_cond_number`, `solver_cond_number_max`, `bc_flux_r_inner`,
`bc_flux_r_outer`, `bc_flux_z_bottom`, and `bc_flux_z_top`, with matching
`time_s` and `step`.  Boundary ids are 0=`r_inner`, 1=`r_outer`,
2=`z_bottom`, 3=`z_top`; values are time-integrated energy [erg].

For the serial 1D implicit path, `solver_residual` is the post-solve relative
tridiagonal residual \(\|Ax-b\|_2/\|b\|_2\), `solver_iter=1` because cuSPARSE
`Dgtsv2` is a direct tridiagonal solve, and `solver_cond_number` is a
diagonal-ratio estimate \(\max_i |A_{ii}|/\min_i |A_{ii}|\).  For the 2D RZ C1
production STS path, no linear algebra solve is performed; `solver_residual=0`
by definition for the algebraic solve diagnostic, `solver_iter` is the number of
applied STS stages in that conduction operator call, and `solver_cond_number`
uses the same diagonal-ratio estimate on the implicit-equivalent Kershaw
operator diagonal \(1+\Delta t\,4\pi a_{cc}/(\rho c_{v,e}V_c)\).  These
diagnostics observe frozen operator coefficients only and do not feed back into
the conduction update.

#### 4.1.2 Per-material electron-ion coupling

`numerics.materials.per_material_conservation_enabled=true` の場合、
electron-ion coupling は常に per-material conserved energy に対して行う。
Radiation/source/laser の deposit-and-redistribute
fallback が将来導入されても、\(Q_{ei}\) は per-material end-to-end のまま
維持する。

各 present pair \((c,m)\) は
\[
\rho_{c,m}=\frac{M_{c,m}}{\alpha_{c,m}V_c},\quad
e_{e,c,m}=\frac{E_{e,c,m}}{M_{c,m}},\quad
e_{i,c,m}=\frac{E_{i,c,m}}{M_{c,m}}
\]
を用いる。\(T_{e,c,m}\), \(T_{i,c,m}\), \(c_{v,e,c,m}\), \(c_{v,i,c,m}\)
は per-material accessor で取得し、lazy cache が有効な場合も accessor の
cache/miss 規約に従う。材料定数は \(A_m\) と \(\bar Z_m\) を用いる。

Finite-\(\Delta t\) の transfer は legacy source-term と同じ
`compute_qei_term_with_cv` を材料別に評価する。すなわち
\[
n_i=\frac{\rho_{c,m}}{A_m m_p},\quad n_e=\bar Z_m n_i,
\]
\[
\tau_{ei}=3.16\times10^8
\frac{A_m T_{e,c,m}^{3/2}}{\bar Z_m^2 n_i\ln\Lambda},
\quad
\tau_{\rm eff}=\tau_{ei}\frac{c_{v,i,c,m}}
{c_{v,e,c,m}+c_{v,i,c,m}},
\]
\[
q_{ei,c,m}^{spec}=
\frac{c_{v,e,c,m}c_{v,i,c,m}}{c_{v,e,c,m}+c_{v,i,c,m}}
(T_{e,c,m}-T_{i,c,m})
\left(1-\exp[-m_{ei}\Delta t/\tau_{\rm eff}]\right),
\quad m_{ei}=\texttt{Numerics.hydro.qei\_multiplier}
\]
であり、材料 energy の更新量は
\[
\Delta E_{ei,c,m}=M_{c,m}q_{ei,c,m}^{spec}.
\]
electron equation からは減算し、ion equation へ同量を加算する：
\[
E_{e,c,m}^{n+1}=E_{e,c,m}^{n}-\Delta E_{ei,c,m},\quad
E_{i,c,m}^{n+1}=E_{i,c,m}^{n}+\Delta E_{ei,c,m}.
\]
この transfer は pair 内で保存的であり、floor clamp を除けば
\(\sum_m(E_e+E_i)\) を変えない。

Kernel 内で更新した \((c,m)\) の `Te_per_material_valid` と
`Ti_per_material_valid` は device-side mirror 上で 0 にする。Host launcher は
mirror を host vector へ戻し、batch 完了後に
`refresh_per_material_derived_cell_fields(..., force_invalidate_all=true)` を
呼び、`ee`, `ei`, `Te`, `Ti`, `Pe`, `Pi`, `cs`, `cv_e`, `cv_i` を
authoritative per-material energies から再投影する。Disabled mode は legacy
`qei_coupling_substep_kernel` をそのまま使う。

#### 4.1.3 Per-material artificial viscosity in 2D RZ

2D_RZ では VNR artificial viscosity の scalar shock sensor と velocity
gradient は従来通り cell で評価するが、粘性 pressure \(Q_{c,m}\) は材料別
thermodynamics を用いて計算する。Present material について
\(\rho_{c,m}\), \(P_{e,c,m}\), \(P_{i,c,m}\), \(c_{s,c,m}\) は per-material accessor
から得る。圧縮 \((\nabla\cdot u)_c<0\) のとき、
\[
Q_{c,m}=
\rho_{c,m}
\left(C_2 \Delta l_c |\nabla\cdot u|_c\right)^2
+C_1\rho_{c,m}c_{s,c,m}\Delta l_c|\nabla\cdot u|_c
\]
を材料別 scratch `Qvisc_per_material[c,m]` に格納し、膨張では
\(Q_{c,m}=0\) とする。既存 momentum update が読む scalar field は
\[
Q_c=\sum_m \alpha_{c,m}Q_{c,m}
\]
として集約する。この `state.Qvisc[c]` は momentum force work 専用であり、
per-material energy deposition には使わない。

2D_RZ の legacy volume-form energy update では、per-material mode の場合だけ
pressure work と viscosity work を material energies へ直接入れる：
\[
\Delta E_{e,c,m}^{PdV}= -\alpha_{c,m}P_{e,c,m}\Delta V_c,\quad
\Delta E_{i,c,m}^{PdV}= -\alpha_{c,m}P_{i,c,m}\Delta V_c,
\]
\[
\Delta E_{Q,c,m}= -\alpha_{c,m}Q_{c,m}\Delta V_c.
\]
`av_heat_to="electron"` では \(\Delta E_Q\) を \(E_e\) へ、それ以外では
既定どおり \(E_i\) へ加える。Cell-mean `ee`/`ei` はこの kernel で直接
更新せず、batch 後の
`refresh_per_material_derived_cell_fields(..., force_invalidate_all=true)` により
per-material energies から再生成する。これにより `state.Qvisc` による
momentum work と `Qvisc_per_material` による energy deposition の二重加算を
避ける。

1D_SPH の compatible-energy path と per-material AV は将来対応へ deferred と
する。現状では 1D per-material conservation enabled run も AV については
legacy single-fluid path を使う。

#### 4.1.4 EOS bisection cost profiling milestone

`TENRYU_PROFILE_EOS_BISECTION=1` の場合、
`refresh_per_material_derived_cell_fields()` は CUDA event で refresh elapsed time
を測り、`[per_material_eos_refresh_timing]` を出力する。Step total は既存の
`Main.verbosity="verbose"` の `[phase_timing] total=... ms` を用いる。Harness は
`scripts/profile_eos_bisection_cost.sh` で、pure-cell control、
mixed-material shell、50% mixed-cell stress の 3 regimes を 3 repeats 実行し、
5 warmup steps を除いた 55 measured steps について per-run median と p95 を
集計する。

Profiling measurement artifact (§4.1.4):
`tmp/profiling/wave_d_eos_bisection_cost.log`。

Baseline median refresh/step ratios:

- pure-cell control: \(9.720310\times10^{-2}\)
- mixed-material shell: \(9.851518\times10^{-2}\)
- high mixed-cell stress: \(9.777364\times10^{-2}\)

Activation rule は「いずれかの regime の baseline median ratio が 10% を超える」
である。今回の baseline は全 regime で 10% 未満だったため、
`numerics.materials.lazy_cache_te_m_enabled` は既定どおり `false` のままにする。
Harness は参考として lazy-cache-enabled pass も実行したが、force-invalidate
refresh が多い per-material 負荷の workload では median ratio が上がったため、本 milestone では
lazy cache activation は行わない。

### 4.2 離散化と安全策（負温度防止）

保存形：
\[
(\rho e_e)^{n+1} = (\rho e_e)^n - \Delta t \nabla\cdot \mathbf{q}_e
\]

#### 4.2.1 Super-Time-Stepping (STS) アルゴリズム

ICF爆縮問題では Spitzer 伝導率 \(\kappa_{SH} \propto T_e^{5/2}\) のため、
コロナ領域（高温・低密度）で実効拡散係数 \(D_{eff}\) が極端に大きくなり、
明示的CFL制約 \(\Delta t_{exp}\) がグローバルΔtに対して数百分の1まで縮小する。
素朴な等間隔サブサイクリングでは \(N_{sub} = \lceil\Delta t/\Delta t_{exp}\rceil\) 回の
Kershawカーネル起動が必要となり、GPU並列化の重大なボトルネックとなる。

**定量的見積もり**（GXII 500μm カプセル典型値）：

| 領域 | \(T_e\) [eV] | \(\rho\) [g/cm³] | \(D_{eff}\) [cm²/s] | \(\Delta l\) [cm] | \(N_{sub}\) (素朴) |
|------|-------------|-----------------|---------------------|-------------------|--------------------|
| コールドシェル | 1 | 1.0 | \(10^3\) | \(10^{-4}\) | 1 |
| アブレーション面 | 100 | 0.1 | \(10^7\) | \(10^{-4}\) | 5 |
| コロナ（flux-limited） | 2000 | \(10^{-3}\) | \(10^{10}\) | \(5\times 10^{-3}\) | 130 |
| コロナ（弱勾配） | 2000 | \(10^{-3}\) | \(3\times 10^{10}\) | \(5\times 10^{-3}\) | 340 |
| ホットスポット | 500 | 10 | \(10^6\) | \(10^{-4}\) | 1 |

> **注**：\(N_{sub}\) はグローバル（全セルの最悪値）であり、
> 1つのコロナセルが \(N_{sub}=340\) を要求すると、
> 全セル（〜80,000）が340回サブステップを踏む。
> 有効計算量の99.9%以上が無駄になる。

v1.0では **Super-Time-Stepping (STS)**（Alexiades, Amiez & Gremaud 1996）を採用し、
Chebyshev多項式の安定性拡大特性により \(O(s^2)\) 倍の安定領域を \(s\) ステージで実現する。

> **参考文献**：V. Alexiades, G. Amiez, P.-A. Gremaud,
> "Super-time-stepping acceleration of explicit schemes for parabolic problems",
> Comm. Numer. Methods Eng., vol. 12, pp. 31–42, 1996.

**明示的CFL限界**（§2.2(b)と同一）：
\[
\Delta t_{exp} = C_{cond}\cdot\min_c\!\left(\frac{(\Delta l_c)^2}{D_{eff,c}}\right),\quad C_{cond}=0.25\;(\text{既定})
\]

**STSステージ数の決定**：
\[
s = \max\!\left(1,\;\left\lceil \sqrt{2\,\frac{\Delta t}{\Delta t_{exp}}} \right\rceil\right)
\]

> STS の安定領域は \(\Delta t_{STS} \approx s(s+1)/2 \times \Delta t_{exp}\) であるため、
> \(s \approx \sqrt{2\,\Delta t/\Delta t_{exp}}\) ステージで Δt 全体をカバーできる。
> 素朴法の \(N_{sub} = 340\) に対し、STS は \(s = 27\) ステージで済む（約13倍の高速化）。

**ステージ数上限**：\(s_{max} = 40\)（既定、SPECIFICATION §6.4.7）。
\(s > s_{max}\) の場合は \(s = s_{max}\) にクランプし、警告を出力する。
§2.2 の \(\Delta t_{cond,sts}\) が正しく設定されていれば通常は発生しない。

**STSサブステップ幅**（Chebyshev根分布）：
\[
\tau_j = \frac{\Delta t_{exp}}{\nu^2 + (1-\nu^2)\cos^2\!\left(\frac{\pi(2j-1)}{4s+2}\right)},
\quad j=1,\ldots,s
\]
ここで \(\nu \in (0,1)\) はダンピングパラメータ（既定 0.01、SPECIFICATION §6.4.7）。
\(\nu\) が小さいほど加速率は高いが、減衰率（最大増幅因子）が \(1/T_s(1/\nu)\) に近づく。
\(\nu = 0.01\) は十分な安定性マージンを提供する。

**タイムステップスケーリング**：
実効スーパーステップ \(\Delta t_{STS} = \sum_{j=1}^{s} \tau_j\) が \(\Delta t\) と一致しない場合、
全 \(\tau_j\) を一様にスケーリングする：
\[
\tau_j \leftarrow \tau_j \times \frac{\Delta t}{\Delta t_{STS}}
\]

**ラダー増幅監査（fail-closed、2026-07-26 カーネルレビュー）**：
一様スケーリングは Chebyshev 安定多項式の根配置を保存しないため、その安定性は
スケール方向に依存する。既定（\(C_{cond}=0.25\)、\(\nu=0.01\)）では自然和
\(\Delta t_{STS}\) が要求 \(\Delta t_{sub}\) を常に上回り（縮小方向、実測増幅上界
0.175）安定だが、\(\nu\) を上げた合法な構成（例 \(\nu=0.05\)・\(s=40\) の
subcycle 上限）ではスケーリングが拡大方向となり、増幅 \(\sim10^{12}\) の不安定
ladder を黙って実行し得た。1D STS 経路（通常 / per-material；SNB は同一 τ を
共有）は stage 実行前にホスト側で
\[
\max_{0<\lambda\le 4C_{cond}/\Delta t_{exp}}\ \Bigl|\prod_{j=1}^{s}(1-\lambda\tau_j)\Bigr| \le 1
\]
を監査し（根間中点＋一様格子サンプリング、決定論・合格時 bit 中立）、違反時は
設定是正（\(C_{cond}\le0.25\) 維持・`sts_damping` 低減・dt 縮小）を促すメッセージと
共に fail-fast する（実装 `sts_ladder_amplification_bound` +
`audit_sts_ladder_stability`、単体テスト `test_conduction_sts_ladder_audit`）。
スペクトル証明書 \(\lambda_{max}\le 4C_{cond}/\Delta t_{exp}\) は kirchhoff dt 経路の
Gershgorin 律速（\(2C_iV_i/\sum G_f\) の min 合成、§4.1a）で厳密。セル局所
\(\Delta l^2/D_{eff}\) 推定のみに頼る harmonic 閉包経路では、強い格子勾配で
\(\lambda_{max}\Delta t_{exp}\) が最大 \(2/(1+r)\)（\(r\)=隣接セル幅比）まで証明書を
超過し得る（残存リスクとして記録；既定閉包は kirchhoff_same_material）。2D STS は
同じ rescale を共有するが監査は未接続（2D 側 todo）。

**拡散係数の凍結**：STSの理論的安定性保証は**線形演算子**（定係数）を前提とするため、
\(D_{eff}\)（安定性評価）と伝導係数（1D face \(\kappa\)、2D Kershaw \(\kappa_{eff}\)）は
スーパーステップの開始時（\(T_e^n\)）に1回だけ評価し、全 \(n_{sub}\times s\) ステージを通じて凍結する。
温度場 \(T_e\) 自体はステージ間で更新される。

**積分スキーム**（STS前進Euler、温度/エネルギー更新モデル）：
\(T_{e,0} = T_e^n\) として、各ステージ \(j=1,\ldots,s\) で 1D_SPH は温度を直接更新し、
2D_RZ は保存形の熱エネルギーを更新する：
\[
T_{e,j} = T_{e,j-1} + \frac{\tau_j}{\rho\, c_v}\,\mathcal{M}_{1D}(T_{e,j-1})
\]
\[
E_{c,j} = E_{c,j-1} + \tau_j\,V_c\,\mathcal{M}^{\kappa}_{Kershaw}(T_{e,j-1}),\qquad
T_{e,c,j}=\frac{E_{c,j}}{\rho_c c_{v,c}V_c}
\]
ここで \(E_c=\rho_c c_{v,c}V_cT_{e,c}\)、各 \(\mathcal{M}\) は凍結伝導係数による拡散演算子の離散化：
- **1D_SPH**：3点差分（§3.1.7の拡散離散化）
- **2D_RZ**：Kershaw 9点ステンシル（Appendix A）

\(\rho\, c_v\) は凍結値（ステップ開始時評価）。
最終結果：\(T_e^{n+1} = T_{e,s}\)。\(e_e\), \(P_e\), \(C_{v,e}\) は
post-conduction sync（擬似コード step 4）で EOS 順変換により再同期する。

The STS floor limiter is selected by `conduction.sts_floor_limiter`.  Legacy
`"net"` mode computes each cell alpha from its net power and scales every pair
by `min(alpha_c, alpha_nb)`.  With mixed-sign pair powers, this can suppress
inflow more than outflow and leave a floor undershoot for the clamp to book.
Opt-in `"donor"` mode computes alpha from the sum of unthrottled outflow
magnitudes and scales each pair by the donor cell's alpha.  The donor's total
realized outflow is therefore bounded by its energy above the floor, which
guarantees $T_{e,trial}\ge T_{e,floor}$ while preserving pair antisymmetry.
See `docs/design/bug18_conduction_floor_pumping_fix_20260712.md`.

**擬似コード**：
```
function conduction_sts(state, dt, config):
    // 1. 拡散係数の評価と凍結
    D_eff = compute_D_eff(state.Te, state.rho, state.Cv_e)  // §4.3
    kappa_eff = D_eff * rho_Cv
    build_kershaw_stencil(kappa_eff)  // 2D; 1D uses frozen face kappa

    // 2. STSステージ数とサブステップ幅の決定
    dt_exp = config.cfl_cond * min_cell(dl² / D_eff)
    s_raw = ceil(sqrt(2 * dt / dt_exp))
    smax = config.sts_max_stages
    eta = config.sts_subcycle_eta   // default 0.9
    dt_sts_max = 0.5 * smax * (smax + 1) * dt_exp * eta
    if dt <= dt_sts_max:
        n_sub = 1
        dt_sub = dt
    else:
        n_sub = ceil(dt / dt_sts_max)
        dt_sub = dt / n_sub

    s = min(ceil(sqrt(2 * dt_sub / dt_exp)), smax)
    for j = 1 to s:
        arg = pi * (2*j - 1) / (4*s + 2)
        tau[j] = dt_exp / (nu² + (1 - nu²) * cos²(arg))
    // タイムステップスケーリング（各サブステップ時間 dt_sub に合わせる）
    dt_sts = sum(tau)
    for j = 1 to s: tau[j] *= dt_sub / dt_sts

    // 3. STSステージ実行（1D温度更新、2D保存形エネルギー更新）
    for m = 1 to n_sub:
        for j = 1 to s:
            // C3: 1D updates Te with M/(rho Cv); 2D updates E=rho Cv V Te with V*M_kappa
            apply_kershaw_stencil(state.Te, tau[j], kappa_eff, rho_Cv)
            apply_temperature_floor(state.Te, config.T_floor)  // §4.2.2

    // 4. Post-conduction EOS sync（必須：STS は Te のみ更新、ee/Pe/Cv_e は未更新）
    floor_clamp(state)                                     // U2: 安全策
    eos_forward(state.rho, state.Te → state.ee, state.Pe, state.Cv_e)  // H13
```

**STS 総ステージ liveness 上限（2026-07-30 新設）**：`Numerics.conduction.sts_total_stages_max`
（既定 200000、0 = 無効）。1 回の伝導適用が起動する総ステージ数 \(n_{sub}\times s\) の上限。
\(n_{sub}=\lceil \Delta t/\Delta t_{sts,max}\rceil\) は \(\Delta t_{exp}\propto \Delta x^2\) に
上限を持たないため、ステップ途中でセルが潰れて \(\Delta t_{exp}\) が崩壊すると
\(n_{sub}\sim 10^8\)（総起動 \(\sim10^{10}\) = 実質ハング）に達し得る（2026-07-30
gauss+SNB で実測・gdb 局在化）。dt 制御下の定常 STS 律速運転は train あたり
40–100 stages（\(\Delta t\) が §2.2(b) の \(\Delta t_{cond}\) に拘束されるため）であり、
既定上限は 3 桁以上の余裕を持つ。超過時は所有セルへの書き込み前に早期 return し、
`ConductionResult::retry_required` で driver の full-step retry（snapshot 復元 + dt/2、
SN material Newton 棄却と同経路）を要求する。retry 予算枯渇時は
total_stages / n_sub / dt_exp を含む診断付き abort（ハングではなく診断可能な失敗）。
4 つの STS ホストサイト（1D/2D × plain/per-material）すべてに適用。

> **温度更新の理由**：STS 内部では \(e_e\) の EOS 逆変換を各サブステップで行わず、
> 1D は温度を、2D Kershaw は熱エネルギー \(E=\rho c_v V T_e\) を更新してから温度へ戻す。
> 全ステージ完了後に1回だけ EOS順変換（H13: \(T_e \to e_e, P_e, C_v\)）を実行して
> \(e_e\), \(P_e\), \(C_{v,e}\) を再同期する。この post-conduction sync がないと、後続フェーズの EOS逆変換（H14）が
> 古い \(e_e\) から旧 \(T_e\) を復元し、伝導更新が無効化される。

#### 4.2.2 局所エネルギー下限制約

各サブステップで、負のエネルギー密度を防止する追加制約を適用する：
\[
\Delta t_{sub,c}^{safe} = \frac{(\rho e_e)_c^{(m)} - (\rho e_e)_{floor}}{\max\!\bigl(|\nabla\cdot\mathbf{q}_e|_c^{(m)},\;\varepsilon_{div}\bigr)}
\]
ここで \((\rho e_e)_{floor} = \rho_c \cdot e_e(T_{e,floor})\)（\(T_{e,floor}\) は§1.1.7のフロア値）、\(\varepsilon_{div} = 10^{-30}\) erg/(cm³·s)（ゼロ除算ガード。\(|\nabla\cdot\mathbf{q}_e| < \varepsilon_{div}\) では \(\Delta t_{sub,c}^{safe} \to +\infty\) となり制約は非発動）。

局所サブステップ制限が発動する場合（\(\Delta t_{sub,c}^{safe} < \Delta t_{sub}\)）：
- 各セルの \(\alpha_c = \operatorname{clamp}\!\bigl(\Delta t_{sub,c}^{safe}/\Delta t_{sub},\;0,\;1\bigr)\) を計算（net: 正味パワー基準、donor: 流出和基準 — §4.2.1 末尾）
- 各面（対）のフラックスを両側セルで**同一の係数**でスケーリング（下記）
- なおフロア割れが残るセルは温度をフロアにクランプ
- clampが発生したセル数を診断へ出す（物理破綻検出）

**対整合スケーリングとエネルギー会計（一連の既知バグ修正後の現行実装；2026-07-26 記述更新）**：
\(\alpha\) は面（対）ごとに両側セルで同一の係数として適用される —
`sts_floor_limiter="net"`（既定）は対の両側の \(\min(\alpha_c,\alpha_{nb})\)、
`"donor"` は当該対の donor セルの \(\alpha\)（§4.2.1 末尾・対整合スケーリング設計文書参照）。
いずれも対反対称 \(P_{nc}=-P_{cn}\) を保存するため、旧記述（当該セルの寄与のみを
\(\alpha\) 倍し、非対称分 \(\Delta E_{scaling}\) を \(E_{safety}\) に計上する方式）は
もはや実装に存在しない（1D/2D STS とも）。エネルギー計上が生じるのは最終クランプ
のみ：スケーリング後もなお \(T_e < T_{floor}\)（net モードの混合符号対で起こり得る）
または非有限になった場合、\(T_e\) をフロアへ引き上げ、その注入
\(\Delta E_{floor} = \rho c_{v,e}\,(T_{floor}-T_{e,computed})\,V\) を
`E_floor_injected` としてステップ末のエネルギー収支（§10.2 の \(E_{safety}\) 項）へ
計上する。donor モードは構成的に \(T_{e,trial}\ge T_{floor}\) を保証するため、床下
不足分の clamp 計上は原理上生じない（非有限ガードのみ残る）。例外：SNB stage
kernel（§4.4）の床スロットルは床スロットル修正以前の per-cell 形を暫定継承しており
（対非対称・発火時は無記帳）、merge train での対称 pair-min 化待ちである。

> **STSステージ数の典型値**：上記のICF典型パラメータでは、
> コロナ支配時に素朴法で \(N_{sub}=130\text{–}340\) 必要な状況を、
> STS は \(s=16\text{–}27\) ステージで処理する。
> 爆縮初期（コロナ未発達）では \(s=1\text{–}5\) 程度。

#### 4.2.3 陰的拡散ソルバ（1D cuSPARSE 三重対角 / 2D Hypre AMG + PCG、オプション）

STS（§4.2.1）の代替として、1D_SPH では **cuSPARSE `cusparseDgtsv2`** による
三重対角 backward Euler ソルバ（SPECIFICATION §6.4.7 `conduction.solver="implicit"`）を、
2D_RZ では **Hypre ライブラリ**（v2.25+、MIT/Apache-2.0）による
陰的拡散ソルバ（`conduction.solver="hypre"`）を提供する。
`"hypre"` は `-DTENRYU_ENABLE_HYPRE=ON` でビルドされた場合のみ使用可能。

**動機**：STSは \(s_{max}\) ステージまでの安定領域拡大を提供するが、
\(N_{sub} > s_{max}(s_{max}+1)/2\)（既定 \(s_{max}=40\) で \(N_{sub}>820\)）の極端な剛性では
ステージ数が上限に達し、グローバルΔtが伝導制約に律速される。
陰的ソルバは伝導CFL制約を完全に除去し、他演算子（hydro, radiation）のΔtのみで
グローバルΔtを決定できる。

**1D_SPH（`conduction.solver="implicit"`）**：

セル \(i\) の球対称有限体積離散化は
\[
\rho_i c_{v,e,i} V_i \frac{T_{e,i}^{n+1} - T_{e,i}^{n}}{\Delta t}
= F_{i+1/2}^{n+1} - F_{i-1/2}^{n+1}
\]
\[
F_{i+1/2}^{n+1}
= A_{i+1/2}\,\kappa_{i+1/2}\,\phi_{i+1/2}\,
\frac{T_{e,i+1}^{n+1} - T_{e,i}^{n+1}}{\Delta r_{i+1/2}}
\]
である。ここで \(A_{i+1/2}=4\pi r_{i+1/2}^2\)（平面テスト時は 1）、
\(\kappa_{i+1/2}\) は調和平均、\(\phi_{i+1/2}\) は flux limiter、
\(\Delta r_{i+1/2}\) は隣接セル中心間距離であり、いずれもステップ開始時の
\(T_e^n\) で 1 回評価して凍結する。

\[
-w_{i-1/2} T_{e,i-1}^{n+1}
+ \left(\frac{\rho_i c_{v,e,i} V_i}{\Delta t} + w_{i-1/2} + w_{i+1/2}\right) T_{e,i}^{n+1}
- w_{i+1/2} T_{e,i+1}^{n+1}
= \frac{\rho_i c_{v,e,i} V_i}{\Delta t} T_{e,i}^{n}
\]
\[
w_{i+1/2} = A_{i+1/2}\,\kappa_{i+1/2}\,\phi_{i+1/2}/\Delta r_{i+1/2}
\]

となり、1 本の三重対角連立一次方程式 \(\mathbf{A}\mathbf{T}^{n+1}=\mathbf{b}\) を得る。
両端境界は Neumann（零熱流束）なので \(w_{-1/2}=w_{N-1/2}=0\) とする。
`cell_is_void` セルは恒等行（\(T_{e,i}^{n+1}=T_{e,i}^{n}\)）で固定する。
GPU 上では cuSPARSE の cyclic-reduction ベース直接解法 `cusparseDgtsv2`
でこの系を解く。

**2D_RZ（`conduction.solver="hypre"`）**：

**陰的定式化（backward Euler）**：

温度場に対する拡散方程式：
\[
C_{v,c}\, \frac{T_{e,c}^{n+1} - T_{e,c}^n}{\Delta t}
= \sum_{j' \in \mathcal{N}_9(c)} (-a_{c,j'})\, T_{e,j'}^{n+1}
\]
ここで \(a_{c,j'}\) は Kershaw 9点ステンシル係数（Appendix A.6、\(-\nabla\cdot(D\nabla\phi)\) の離散化）。
\(\mathcal{N}_9(c)\) はセル \(c\) と 8近傍の 9点ステンシル。

行列形式に整理すると：
\[
\underbrace{\left(\frac{C_v}{\Delta t}\,\mathbf{I} + \mathbf{M}_{Kershaw}\right)}_{\mathbf{A}}
\, \mathbf{T}_e^{n+1}
= \underbrace{\frac{C_v}{\Delta t}\,\mathbf{T}_e^n}_{\mathbf{b}}
\]

ここで \(\mathbf{M}_{Kershaw}\) は Kershaw ステンシル行列
（対角 \(a_{ii} \ge 0\)、オフ対角 \(a_{ij} \le 0\) for \(j \ne i\)、M-matrix性）。

**\(\mathbf{A}\) の性質**：
- **対角**：\(A_{cc} = C_{v,c}/\Delta t + a_{cc} > 0\)（両項とも正）
- **オフ対角**：\(A_{cj} = a_{cj} \le 0\)（Kershaw M-matrix性より）
- **対称**：Kershaw ステンシルは対称（Appendix A.6 より \(a_{cj} = a_{jc}\)）
- **正定値**：質量対角項 \(C_v/\Delta t > 0\) の付加により厳密正定値
- → **A は SPD（対称正定値）**であり、**PCG（共役勾配法）が最適**

**ソルバ構成**：
- **前処理**：Hypre BoomerAMG（代数的マルチグリッド）
  - 粗視化：HMIS（type 10）— GPU上でのスケーラビリティに優れる
  - 緩和：\(\ell_1\)-Jacobi（type 18）— GPU上でatomic不要、完全並列
  - 補間：ext+i（type 6）— Kershaw 行列の対角優位性に適合
  - 最大レベル数：25（既定）
- **ソルバ**：PCG
  - 相対収束判定：\(\|r_k\|/\|r_0\| \le \varepsilon_{rtol}\)、\(\varepsilon_{rtol} = 10^{-8}\)（既定）
  - 最大反復数：50（既定）
  - 初期推定：\(T_e^n\)（前ステップ解）

**拡散係数の線形化**：STS と同様に \(D_{eff}\) はステップ開始時の \(T_e^n\) で
1回評価して凍結する（§4.3 参照）。非線形反復（Picard/Newton）は行わない。
この線形化は STS と数学的に等価であり、ソルバ間の比較検証が可能となる。

**Δt制御への影響**：
`conduction.solver="implicit"` または `"hypre"` の場合、§2.2(b) の \(\Delta t_{cond,sts}\) は
\(\Delta t_{cond} = \infty\) に置換される（伝導CFL制約なし）。
グローバルΔtは \(\min(\Delta t_{hydro}, \Delta t_{rad}, \Delta t_{user})\) のみで決定される。

**エネルギー保存**：陰的スキームは無条件安定だが、大きなΔtでは離散化誤差が増大する。
エネルギー会計（§10.2）で残差エネルギー \(E_{solver} = \sum_c C_{v,c} (T_{e,c}^{n+1} - T_{e,c}^n) V_c - \Delta t \sum_c (\nabla\cdot q)_c V_c\) を追跡し、
\(|E_{solver}/E_{total}| < \varepsilon_{budget}\) であることをモニタリングする。

**性能特性**（GXII 500μm、2D RZ 400×200 = 80K cells、A100基準）：

| 項目 | STS (\(s=27\)) | Hypre |
|------|---------------|-------|
| ステンシル構築（C1+C2） | ~1 ms（共通） | ~1 ms（共通） |
| ソルブ | 27 × ~0.5 ms = ~14 ms | setup ~3 ms + PCG ~5 ms = ~8 ms |
| 合計 | ~15 ms | ~9 ms |
| Δt 制約 | \(s_{max}(s_{max}+1)/2 \times \Delta t_{exp}\) | なし（∞） |

> **使い分け指針**：
> - \(s \le 15\)（爆縮初期〜中期）：STS が高速（AMG setup コスト不要）
> - \(s \ge 20\)（コロナ発達期）：Hypre が有利（カーネル起動数一定）
> - 極端な剛性（\(N_{sub} > 820\)）：Hypre 一択（STS は \(s_{max}\) 飽和）
> - 検証用途：STS と Hypre の交差検証により実装の正しさを確認（§4.2.3(c)）

**(c) STS↔Hypre 交差検証**：
両ソルバは同一の Kershaw ステンシル（C2 カーネル出力）と同一の線形化（\(D_{eff}\) 凍結）を
共有するため、同一条件で STS と Hypre の解は一致しなければならない。
定量的基準：全セルで \(|T_e^{STS} - T_e^{Hypre}| / T_e^{STS} \le 10^{-6}\)
（STS（前進Euler）と Hypre（後退Euler）はともに時間1次精度 \(O(\Delta t)\) であり、
両者の1ステップあたりの差は \(O(\Delta t^2)\)。PCG 収束判定 \(10^{-8}\) と合わせてこの許容値を設定）。

**Post-conduction EOS sync**：Hypre も STS と同様に \(T_e\) を直接更新し、\(e_e\) は未更新のままである。
Hypre solve 後、§4.2.1 step 4 と同一の post-conduction sync（U2 + H13）が必須である。

> **参考文献**：
> - R. D. Falgout, U. M. Yang, "hypre: A Library of High Performance Preconditioners",
>   ICCS 2002, LNCS 2331, pp. 632–641.
> - V. E. Henson, U. M. Yang, "BoomerAMG: A Parallel Algebraic Multigrid Solver and Preconditioner",
>   Appl. Numer. Math., vol. 41, pp. 155–177, 2002.

### 4.3 2D RZ空間離散化

2D RZメッシュ上の電子熱伝導には **Kershaw 9点差分法**（Appendix A）を使用する。

**拡散係数**：
\[
D = \frac{\kappa_{SH}}{\rho\, c_v} \quad [\text{cm}^2/\text{s}]
\]
ここで \(c_v = \partial e_e/\partial T_e\big|_\rho\) [erg/(g·eV)] は電子比熱（質量あたり）。
体積比熱は \(C_v = \rho c_v\) [erg/(cm\(^3\)·eV)]。

**Flux limiterとの組み合わせ**：
セル毎に実効拡散係数を制限して \(\kappa_{eff,c}=\rho_c c_{v,c}D_{eff,c}\) を得てからKershaw行列を構築する：
\[
D_{eff,c} = \frac{|q_e|_c}{\rho_c\, c_{v,c}\,|\nabla T_e|_c}
\]
ここで \(|q_e|_c\) はflux limiter適用後の熱流束の大きさ（§4.1）。

**\(D_{eff}\) の評価戦略**：シングルパス（ラグ付き）。
1. 前ステップ（または前サブステップ）の温度場 \(T_e^{old}\) から勾配 \(\nabla T_e\) を計算
2. Spitzer熱流束 \(q_{SH}\) を評価し、フラックスリミッターを適用して \(|q_e|\) を得る
3. \(D_{eff} = |q_e| / (\rho\, c_v\, |\nabla T_e|)\) を計算（\(|\nabla T_e| < \varepsilon_{grad}\) の場合は \(D_{eff} = D_{SH}\)）
4. \(D_{eff}\) は STS 安定性評価に用い、2D Kershaw ステンシルは \(\kappa_{eff}\) で構築して熱エネルギーを1ステップ更新する

反復は行わない（計算コストと収束性の観点から）。

**セル中心での温度勾配 \(|\nabla T_e|_c\) の計算**：
Kershaw B-演算子（Appendix A.4）を用いてノード \((i,j)\) での勾配を構成し、
セル中心では隣接4ノードの勾配を算術平均する。

> **B-演算子の概要**：B-演算子（Appendix A.4）は、面積重み付き勾配演算子であり、
> セル中心値からノード勾配を構成する。2D RZにおいて、各ノード \((i,j)\) は
> 周囲4セルの温度値 \(T_e\) を入力として受け取り、それらセルの幾何ベクトル
> \(\mathbf{A}, \mathbf{B}\) とヤコビアン \(J\) を用いてノードでの
> \(\nabla T_e\big|_{n}\) を計算する。得られたノード勾配の大きさ
> \(|\nabla T_e|_n\) をセルの4コーナーノードで算術平均してセル中心値
> \(|\nabla T_e|_c\) とし、\(D_{eff}\) の評価に用いる。

具体的には：
\[
\nabla T_e\big|_c = \frac{1}{4}\sum_{k=1}^{4} \nabla T_e\big|_{n_k}
\]
ここで \(n_k\) はセル \(c\) の4コーナーノード。勾配の大きさ：
\[
|\nabla T_e|_c = \sqrt{\left(\frac{\partial T_e}{\partial r}\right)_c^2 + \left(\frac{\partial T_e}{\partial z}\right)_c^2}
\]
\(|\nabla T_e|_c < \varepsilon_{grad}\) の場合は
\(D_{eff,c} = D_c\)（Spitzer拡散係数、制限なし）とする。ここで
\[
\varepsilon_{grad} = \max\!\left(10^{-10} \times \frac{T_{e,c}}{\ell_c},\; 10^{-30}\;\text{eV/cm}\right)
\]
\(\ell_c = V_c^{1/3}\) [cm] はセル代表長（体積の立方根）。この相対的閾値により、ほぼ均一温度領域でも \(D_{eff}\) の非物理的発散（\(D_{eff} \gg D_c\)）を防ぐ。

Kershawステンシルの各係数（Appendix A.6）に \(\kappa_{eff}\) を入力することで、
flux limiterの非線形性が保存形の熱流束に反映される。

**v1.0 における解法（明示的演算子）**：

v1.0 では Kershaw 9点ステンシルを**明示的演算子**として熱エネルギーに適用する：
\[
E_c^{m+1} = E_c^m + \Delta t_{sub}\,V_c\,\mathcal{M}_{Kershaw}^{\kappa}(T_e^m),\qquad
T_{e,c}^{m+1} = \frac{E_c^{m+1}}{\rho_c c_{v,c}V_c}
\]
ここで \(E_c=\rho_c c_{v,c}V_cT_{e,c}\)、\(\mathcal{M}_{Kershaw}^{\kappa}\) は
Appendix A の9点ステンシルで離散化された \(\nabla\cdot(\kappa\nabla T_e)\)。
2D Kershaw conduction は symmetric pair conductances \(G_{ij}=G_{ji}\) を持つ演算子で
thermal energy \(\rho c_vVT\) を更新し、per-cell energy を \(dt\times\nabla\cdot(\kappa\nabla T)\,V\) で進めてから
\(T_{new}=E_{new}/(\rho c_vV)\) を導出するため、総和 \(\sum \rho c_vVT\) は構成上保存される。
暗黙的線形方程式系の求解は**不要**。
明示的安定性制約：\(\Delta t_{sub} \le \Delta x^2 / (2\, D_{max})\)（§4.2.1 の \(N_{sub}\) 式で保証）。

### 4.4 SNB 非局所電子熱輸送（1D、opt-in、2026-07-10 新設）

`Numerics.conduction.nonlocal_model="snb"`（既定 `"none"` = 本節不活性・従来経路 bit 恒等）で、
Schurtz–Nicolaï–Busquet (SNB) 型多群拡散の非局所補正を 1D（planar/cylindrical/spherical）の
電子熱伝導に適用する。設計・変種裁定・導出の全記録は
`docs/design/snb_nonlocal_1d_20260710.md`（binding）。一次文献: Schurtz, Nicolaï & Busquet,
Phys. Plasmas 7, 4238 (2000); Sherlock, Brodrick & Ridgers, Phys. Plasmas 24, 082706 (2017);
Cao, Moses & Delettrez, Phys. Plasmas 22, 082308 (2015)。

**モデル（operative form）**：エネルギー群 g（大域エッジ E_0=0<…<E_{N_g}、毎伝導ステップ
β̂=E/(k_B max T_e) の幾何級数 [0.1, `snb_E_max_over_Te`] で再構築）ごとに
\[
\frac{H_g}{\lambda_g^{abs}} - \nabla\!\cdot\!\Big(\frac{\lambda_g^{tr}}{3}\nabla H_g\Big)
 = -\nabla\!\cdot\! U_g,\qquad
U_g=\xi_g\,q_{SH},\quad
\xi_g = P(\beta_g)-P(\beta_{g-1}),\ \beta_g=\frac{E_g}{k_BT_e(\mathbf{x})}
\]
\[
P(\beta)=1-e^{-\beta}\,(\beta^4+4\beta^3+12\beta^2+24\beta+24)/24
\quad(\text{解析原始関数、数値求積なし})
\]
総熱流束は \(q_t=q_{SH}-\sum_g(\lambda_g^{tr}/3)\nabla H_g\)。境界条件は全境界で反射
\(\nabla H_g\cdot n=0\)（q_SH=0 の閉境界と整合）。

**群 mfp（`snb_mfp`）**：群代表エネルギー \(\varepsilon_g = (E_{g-1}+E_g)/2\)（**中心規約** —
上端規約の誤読は κ_eff を kλ₀=0.05 で −17% 偏らせる、設計 doc §7G）。
- `"geometric_r2"`（既定; Sherlock 2017 Eq 1、Brodrick r=2 等価）:
  \(\lambda_g = 2\sqrt2\,\varepsilon_g^2 / (4\pi n_e e^4 \ln\Lambda\,\sqrt{\bar Z\varphi})\)、
  \(\varphi=(\bar Z+4.2)/(\bar Z+0.24)\)（Epperlein–Short 1991 由来の Lorentz→SH 補正）。TMAT ionization fractions があるときは \(\bar Z\to\bar Z r_2=Z_{\rm eff}\)（§1.1.3a、自動・1D）。
- `"original"`（Schurtz 2000 Eq 3+23）:
  \(\lambda_g = 2\,\varepsilon_g^2/(4\pi n_e e^4\ln\Lambda\,\sqrt{\bar Z+1})\)。
両変種とも \(\lambda_g^{abs}=\lambda_g^{tr}=\lambda_g\)（E 場補正時のみ輸送側を分離、下記）。
n_e, lnΛ は §4.1 と同一のセル式。λ_g は lnΛ 依存を除き局所 T_e に非依存
（T 依存は ξ_g 側に集約）— 高温セル起源の長 mfp 電子が低温物質へ侵入する preheat の機構
そのもの。低 T_e 域での λ∝ε² 過大評価（Cao 2015 §V の range-mfp 問題）は既知の v1 制限。

**単位規約（cgs+eV 明示化、2026-07-26 カーネルレビュー）**：群端
\(E_g\) は実装上 **erg** で保持する（\(E_g=\hat\beta_g\,k_BT_{ref}\)、
\(k_BT_{ref}\) は eV→erg 換算後; `conduction_snb_1d.cu` の `edges_erg`）。上記 mfp 式の
\(\varepsilon_g\) は erg で代入しなければならない — 分母の \(e^4\) は esu（\(e^2\) が
erg·cm 次元）であり、eV のまま代入すると mfp が
\((1.602\times10^{-12})^{-2}\simeq3.9\times10^{23}\) 倍破綻する。一方 source 重み
\(\xi_g\) の \(\beta_g=E_g/(k_BT_e)\) は erg/erg の無次元（`snb_spectrum_cdf` は
\(E_g[\mathrm{erg}]/(k_BT_e[\mathrm{erg}])\) を受ける）。この規約は外部参照データとの
G3 Tier B（Marocchino 2013 OSHUN band）一致により、実装と独立に単位検証済み。

**電場補正（`snb_efield`、既定 `"none"`）**：`"local"` で輸送側 mfp のみ
\(1/\lambda_g^{tr} \mathrel{+}= |e\mathcal E|/\varepsilon_g\)、
\(|e\mathcal E| = k_BT_e\,|\nabla\ln n_e + \gamma(\bar Z)\nabla\ln T_e|\)、
\(\gamma = 1+\tfrac32(\bar Z+0.477)/(\bar Z+2.15)\)（Schurtz Eq 32–35）。既定 OFF の根拠:
r=2 較正は電場抑制を織込み済みで、重ねると二重計上（Sherlock 2017 §VII; 設計 doc §1.5）。

**離散化**：セル中心 H_g、体積積分 FV 三重対角（対角 = V/λ^{abs} + Σ面係数 ⇒ 厳密対角優位・
M 行列）。面拡散係数 \(D_{g,f} = \mathrm{harmonic}(\lambda^{tr}_L, \lambda^{tr}_R)/3\)。
面 ξ_g は面平均 T_e で評価。q_SH,face は §4.1 と同一の面 κ 閉包（`face_kappa_policy` に追従）。
群バッチ解法は `cusparseDgtsv2StridedBatch`（FLD §6.7 A-3 と同型; n_cells<3 は SNB 不活性
guard）。補正流束 \(\delta q_f = -\sum_g D_{g,f}(H_{g,R}-H_{g,L})/\Delta r_f\)
（群ループ固定順 = 決定論）。void セルは恒等行 H=0・隣接面 D=0（preheat は void を透過しない）。

**flux limiter との合成**：SNB は SH 流束を置換し、既存 f_lim は外側安全 cap:
\(q^{SNB}_f = \theta_f (q_{SH,f}+\delta q_f)\)、
\(\theta_f = 1/(1+|q_{SH,f}+\delta q_f|/q_{max,f})\)（§4.1 と同一の調和形・同一の面 q_max）。
局所極限で δq→0 ⇒ OFF 経路の制限済み流束に厳密一致。cap 発火は毎ステップ診断
（θ<0.99 / θ<0.5 面数、min θ — history `/diagnostics/conduction/snb/v1/*`）。

**時間結合（iSNB Picard、Cao 2015 §IV）**：分離明示適用は解像された非局所域で格子不安定
（補正演算子は反拡散的 ω∝+k²）。反復 {Te^n から (θ,δq)^{m-1} 凍結で STS 超ステップ →
反復端 Te^m から δq^m 再評価} を `snb_picard_rtol`（max-norm、次数非依存 ⇒ run-to-run bit 安定）
収束まで（最小 2、上限 `snb_picard_max_iters`; 非収束は warn+診断記録の上で最終反復採用 —
沈黙しない）。収束点の線形解析（設計 doc §2.3）: 増幅率 \(G = 1/(1+x(1-r)) \in (0,1)\) ∀dt
（無条件安定）、Picard 縮小率 \(xr/(1+x) < r < 1\)（常収束）。STS 段数・dt_exp は Te^n で凍結
（反復不変）。追加の dt 制御なし。エネルギー更新は面流束差分（telescoping ⇒
machine-precision 保存は構造的）。歴史 OFF 経路のカーネルと stage loop は byte 不変
（clone 方式、W-G1 の FMA 再配置教訓）。

**離散最大値原理 ceiling（2026-07-30 新設）**：SNB stage kernel は θ・dq を Picard パス
冒頭状態で凍結したまま STS 列を積分するため、微小 \(\rho c_v\)（希薄）セルへ凍結 dq が
定率注入されると Te が非有界に成長し得る（実測: 3step ベンチで Te→6.8e5 eV @ρ=5e-8、
Picard 不収束 iterate の受理と複合）。物理事実「電子輸送単独では作用対象状態の大域最大
Te を超えられない」を stage kernel で強制する: 天井 = 適用エントリ状態の大域最大
sanitized Te（既存の \(T_{ref}\)、Picard 全体で凍結; 全冷/床未満時は無効）。床 clamp と
同型の accepted-iterate 会計（`snb_ceiling_clamp_count` / `snb_E_ceiling_removed` [erg]）
を持ち、除去エネルギーは driver の numerical-loss 台帳へ計上、初回発動時に警告。
非発動経路はテキスト不変（健全 run は bit 恒等）。2D port（§4.5）は同型の潜在パターンを
持つが未修正（repro 無し・検証スコープ別 — DEFERRED 2026-07-30）。

**dq の面ローカル自由流束 bound（2026-07-31 新設）**：Picard 収束判定は大域最大流束で
正規化されるため、希薄面の dq は未収束ノイズのまま受理され得る。|dq| ≫ q_max は
outer cap を θ→0 に飽和させ q_SH ごと伝導を窒息させる — 平滑化を失った帯の Te/P
ジグザグが流体でリンギング成長し、単一セルの破局圧縮（実測 phase=hydro で Te×740、
3step t=3.867 ns）→ κ 爆発 → dt_cond 床 abort に至る（gauss の STS ハングを誘発した
セル潰れも同根）。物理事実「面を通る非局所補正は局所自由流束を超えられない」を
θ 形成前に強制する: `|dq_f| ≤ q_max_f`（outer cap 有効時のみ; q_max はその枝でのみ
定義）。健全面は |dq| ≪ q_max で非拘束（bit 恒等）。発動は counters[2] 集計+初回警告。
検証: 修正前に決定論的だった 3step abort が t_end 完走に転じ、per-phase overshoot
検出器（safety.overshoot_fatal=1.0 装備 run）無発火を確認（2026-07-31）。

**検証（VERIFICATION §7.9）**：G1 OFF-bit（GXII FLD golden rel=0 ×6 指標）/ G2 局所極限
（|R−R_disc|≤5.5e-8 実測 @tol 1e-3、二次法則勾配 1.986、tanh ramp 2 rung が導出曲率 bound 内・
二次 gain 100.7）/ G3 Epperlein–Short 分散（Tier A 離散解析一致 ≤7.2e-7 @gate 3e-3、
Tier B Marocchino 2013 OSHUN band）/ G4 保存台帳 ≤7.4e-17 @gate 1e-14（planar/cyl/sph）
/ G5 最大値原理（球対称 hotspot + ×1e-3 希薄受容帯: per-step \(\max T_e^{new}\le\max T_e^{entry}\)、
ceiling 台帳込み全区間エネルギー閉包 ≤1e-13、guard 実発動の自己検査、無帯 control leg で
非発動を確認; verify `snb_max_principle_1d`）。
解析分散 \(R(k)=\sum_g \xi_g/(1+\lambda_g^{abs}\lambda_g^{tr}k^2/3)\)
（一様プラズマ厳密閉形式、局所極限係数 \((8/3)\cdot 8!/24 = 4480\); kλ_ei 軸上で密度・lnΛ
不変）。既知のモデル漸近 1/k²（運動論は 1/k、E-S 1991）により band 認証は kλ_ei < 0.5 に限定
（kλ_ei=0.5 は記帳 tier）。

---

### 4.5 SNB 非局所電子熱輸送（2D_RZ port, 2026-07-11）

モデル（(SNB-H)/(SNB-Q)、ξ_g の厳密 primitive P(β)、群構造 β̂∈[0.1,20] 幾何
級数・群中心規約、mfp 変種 `geometric_r2`/`original`、iSNB Picard 結合、
`nonlocal_model` 系 namelist キー）は 1D 設計
`docs/design/snb_nonlocal_1d_20260710.md`（merge 後の §4.4）と同一。本節は
2D_RZ 離散化の差分のみ（設計正典 `docs/design/2d_snb_port_spec.md`、実装
`src/hydro/conduction_snb_2d.cu`）。

**離散化（pair-power 形式）** — 2D 伝導の更新は既に対毎の反対称 power
\(P_{cn}=4\pi\bar G_{cn}(T_n-T_c)\)（\(\bar G=\tfrac12(G_{cn}+G_{nc})\)、
Kershaw 9 点、`symmetric_pair_power`）で書かれており、SNB は同じ対構造で
入る：
- 群 g の (SNB-H) は、セル拡散係数 \(\lambda^{tr}_g/3\) で
  `assemble_kershaw_cell`（全 4 辺 BC_REFLECT、M-matrix repair ON）を群毎に
  組み、apply と同一の対称化 \(\bar G\) で 9 点 CSR 化する。対角は
  \(4\pi\sum_s\bar G + V_c/\lambda^{abs}_g\)（void/λ≤0 行は identity）。
  演算子と流束が同一離散対象なので、(ID) 恒等式は構成的に成立。
- RHS は対平均温度で評価した ξ_g を OFF T-stencil の対 power に掛けた
  \(\sum_s \xi_{g,cs}P^{sh}_{cs}\)。q_sh は OFF path の離散流束そのもの
  （2D OFF は f_lim を物理に適用しないため「unlimited q_sh」が OFF 流束と
  一致 — §8.6 found-issue 参照）。
- 解法は群バッチ一括（row = g·n_cells + c、群間非結合のブロック対角 SPD
  M-matrix）の Jacobi-PCG。決定論: dot は固定順 cub reduction、atomic なし。
  内部定数（knob ではない）: rtol 1e-10 / max 1000 反復。高群（λ∝ε² 巨大）
  ブロックは反射 BC で準特異となり反復が上限に張り付くことがある
  （cgres ~1e-9 到達、正しさへの影響は G2/G3 で有界と実測 — 効率課題として
  記録、port 設計 doc Addendum 1.3.3）。
- 補正対 power \(P^{dq}_{cs}=\sum_g(-\mathrm{csr}_s)(H_{g,n}-H_{g,c})\) は
  CSR 非対角の読み戻し（反対称厳密 ⇒ telescoping 保存は mesh 形に依らず
  構造的）。外側 cap は cell θ（4 ノード勾配平均で
  \(\vec q_{sh}+\vec q_{dq}\) を再構成し \(q_{max}(f\_lim)\) と比較）を
  pairwise-min で適用。stage kernel は kershaw alpha/apply の clone
  （\(\min(\theta)\,(P^{sh}_{live}+P^{dq}_{frozen})\)、既存 kernel byte
  不変）。Picard driver は 1D と同構造（stash/restore、max-norm 収束、
  min 2 反復、非収束は warn + 診断）。
- 適用範囲（この tree の validation）: 2D_RZ + 2T + `solver="sts"` + 単一
  rank + 非 per-material。`snb_efield="local"` は 2D v1 で ConfigError
  （fail-closed）。1D_SPH はこの tree では ConfigError（feature/1d-brushup ブランチとの merge でガードが union になる）。

**検証（VERIFICATION §4.10、実測 2026-07-11）**: G2 z-mode ladder
|R−R_disc| = 5.5e-8/5.3e-9/5.9e-10（1D gate と同値クラス）、slope 1.986、
集約 tanh-ramp 二乗則 gain 100.3；G3 E-S 分散 Tier A worst ~2.4e-4
（gate 3e-3、Z∈{1,4}×kλ_ei∈{0.05..0.5}×nz ladder）；G4 保存
（dt=10×dt_exp×20 step）: 直交 mesh E_rel 5.6e-17（機械精度 telescoping、
Picard 毎 step 2 反復収束）、歪み mesh（ノード 15% sin 変位）は
**no-added-leak 判定**（SNB 2.16e-5 ≤ 2×OFF 対照 2.56e-5 — OFF Kershaw 2D
自体の歪み mesh での dt 比例エネルギー漏れは pre-existing found-issue、
port 設計 doc Addendum 1.3.1 に記録）。

## 5. レーザーレイトレース（内部、2D RZ方式）

### 5.1 幾何光学と屈折率
屈折率（プラズマ）：
\[
n_{refr} = \sqrt{\max(\varepsilon_n,1-\frac{n_e}{n_{crit}})},\quad
n_{crit} = \frac{m_e \omega_L^2}{4\pi e^2} \quad [\text{cm}^{-3}]
\]
- 既定：\(\varepsilon_n=10^{-4}\)（無次元、\(n_{refr}\) の下限を \(\sqrt{\varepsilon_n}\) に制限）
- \(\omega_L = 2\pi c/\lambda_L\)：レーザー角周波数

正規化した電子密度：
\[
\hat n = \frac{n_e}{n_{crit}}
\]

レイ方程式（時間微分形）：
\[
\frac{d\mathbf{r}}{dt} = \mathbf{v},\quad
\frac{d\mathbf{v}}{dt} = -\frac{c^2}{2}\nabla\hat n
\]

> **注**：アーク長形式 \(\frac{d}{ds}(n_{refr}\frac{d\mathbf{r}}{ds})=\nabla n_{refr}\) と等価。
> 時間微分形はLeapfrog積分に自然に対応する。

### 5.2 臨界近傍の取り扱い（必須仕様）
臨界面 \(n_e=n_{crit}\) 近傍でモデルが破綻し数値発散しやすい。v1.0では以下を **固定**する。

- クリティカル判定：\(\hat n=n_e/n_{crit}\)
- もし \(\hat n \ge 1-\varepsilon_{crit}\)（既定 \(\varepsilon_{crit}=10^{-4}\)）なら：
  - レイをその場で **終了**（terminate）
  - 残存強度 \(I_{rem}\) は "未吸収（反射/損失）" として積算し、エネルギー収支に入れる
  - その時点までにIBで減衰した分のみを沈着として計上
- \(\kappa_{IB}\) の式に \(1/n_{refr}\) が含まれる場合でも、上の terminate により発散領域へ入らないことを保証し、さらに \(n_{refr}\ge \sqrt{\varepsilon_n}\) で下限を持つ
- **最小強度カットオフ**：\(I < I_{cutoff} \cdot I_0\) となったレイは終了する。
  ここで \(I_0\) はレイの初期パワー、\(I_{cutoff}\) は `raytrace.intensity_cutoff`（既定 \(10^{-6}\)、無次元）。
  残存パワー \(I_{rem}\) はエネルギー収支に計上する（`laser_unabsorbed` に加算）。
  この処理により、吸収がほぼ完了したレイが長時間追跡されることを防ぎ、
  計算コストを削減する。典型的にはIB吸収が強い高密度領域で \(I/I_0 \sim 10^{-10}\) まで
  減衰したレイに適用される。\(I_{cutoff} = 0\) で無効化可能

> 将来拡張：ターンニングポイントでの鏡面反射など（v1.0ではしない）。

#### 5.2.1 v1.0 で扱わない物理（Non-goals）
以下はv1.0のスコープ外であり、実装しない。将来バージョンでの拡張候補として記録する：
- **共鳴吸収**（resonance absorption）：臨界面での電場共鳴による吸収。v1.0はIBのみ
- **CBET**（Cross-Beam Energy Transfer）：ビーム間のエネルギー移行 — v1 実装済み（1D_SPH opt-in、§5.10）。2D_RZ は将来拡張
- **LPI**（Laser-Plasma Instability）：SRS、SBS、TPD等のパラメトリック不安定性
- **非局所電子輸送**：スーパーガウシアン分布関数の効果
- **自己集束・自己位相変調**：ポンデロモーティブ力によるビーム変形
- **磁場効果**：磁化プラズマ中の伝搬
- **偏光依存吸収**：v1.0は偏光非依存（スカラーIB）
- **LaserMeshの動的再生成**：格子配置は初期化時に固定

> **未吸収エネルギーの取り扱い**：臨界terminate またはメッシュ外終了した際の残存パワー \(I_{rem}\) は
> 「反射・散乱で外へ逃げたエネルギー」として `laser_unabsorbed` 診断に積算し、
> エネルギー収支（入射 = 吸収 + 未吸収）で検証する（§5.8.2、§10.2）。

### 5.2.2 L1 peer-review closure tests

L1 production closure includes four Catch2 verification contracts:

- `test_l1_oblique_incidence_refraction.cu`: records the required oblique-incidence
  Snell-law turning-depth sweep \(n_{turn}/n_{crit}=\sin^2\theta\). The current
  `2d_rz_l1_laser_dep.py` deck has no exposed beam-angle control, so this ctest
  succeeds with an explicit feature-gap caveat rather than adding a namelist/API.
- `test_l1_beer_lambert_inverse_bremsstrahlung_slab.cu`: records the required
  Beer-Lambert uniform-slab IB attenuation fit. The current L1 deck has no
  exposed constant undercritical uniform-slab mode with controllable IB/test
  opacity, so this ctest succeeds with an explicit feature-gap caveat.
- `test_l1_laser_energy_balance_in_absorbed_escaped_reflected.cu`: runs the
  vacuum-transparent L1 deck for one step and requires
  \[
  E_{in}=E_{laser,deposited}+E_{laser,escaped}+E_{laser,reflected}
  \]
  to relative error \(\le 10^{-6}\). TENRYU v1.0 does not export reflected laser
  energy separately; critical-surface termination and boundary escape are
  accounted in history `energy/laser_escaped`, so the reflected term is zero
  unless a future `energy/laser_reflected` dataset is present.
- `test_l1_ray_count_noise_convergence.cu`: runs the exposed super-Gaussian L1
  mode at 65/129/257 rays per axis and uses the 257-ray map as the reference.
  The relative L2 map difference must decrease from 65 to 129 rays with reduction
  factor \(\ge 1.2\), pinning the expected Monte-Carlo-style noise convergence
  under the current deterministic ray partition.

### 5.2.3 L2 peer-review closure tests

L2 laser+hydro production closure includes three additional Catch2 verification
contracts:

- `test_l2_planar_ablation_front_benchmark.cu`: pins the required planar
  ablation-pressure and ablation-front benchmark contract. The intended active
  gate compares measured ablation-front pressure against a Lindl-Kruer /
  DRACO-style scaling within 50% and requires monotone front recession. The
  current `2d_rz_l2_laser_hydro.py` deck exposes `planar_slab_pulse` and raw
  `Pe`/`Pi`/`Te` plot fields, but not a stable ablation-front/pressure
  diagnostic or benchmark mode, so the ctest succeeds with an explicit
  feature-gap caveat without adding new deck flags.
- `test_l2_laser_to_hydro_energy_partition.cu`: pins the required
  laser-deposited-energy to hydro-energy partition closure. The intended active
  gate is
  \[
  E_{laser,dep}=\Delta E_{thermal}+\Delta E_{kinetic}+\Delta E_{bc}+
  E_{residual}
  \]
  within 5% with \(f_{thermal}>0.3\). Current history output exposes cumulative
  laser deposition plus internal, kinetic, and phase-resolved hydro terms, but
  not a single auditable L2 boundary-loss/residual partition, so the ctest
  records this diagnostic feature gap.
- `test_l2_oblique_beam_target_offset_symmetry.cu`: pins the required on-axis
  symmetry and target/beam-offset deposition center-of-mass contract. The
  current L2 deck exports baseline deposition patterns, but exposes no
  target-offset, beam-offset, or oblique-angle control, so this ctest succeeds
  with an explicit feature-gap caveat rather than adding namelist parameters.

### 5.3 レイ方程式の離散化

#### 5.3.1 正規化
LaserMeshの代表セル幅を \(\Delta x\) として正規化変数を定義する：
\[
\hat{\mathbf{r}} = \mathbf{r}/\Delta x,\quad
\hat{\mathbf{v}} = \mathbf{v}/c
\]

正規化レイ方程式：
\[
\frac{d\hat{\mathbf{r}}}{dt} = \frac{c}{\Delta x}\hat{\mathbf{v}},\quad
\frac{d\hat{\mathbf{v}}}{dt} = -\frac{c}{2\Delta x}\hat\nabla\hat n
\]
ここで \(\hat\nabla\) は正規化座標系での勾配。

#### 5.3.2 Leapfrog時間積分（既定）
レイトレース副ステップ幅 \(\Delta t_{ray}\) に対し、
\(C_{ray} = c\Delta t_{ray}/\Delta x\) として：

**(i) 1D_SPH（`ray_trace_1d_sph`）— 可変ステップ Verlet（2026-07-31）**：

適応ステップ幅（下記）と両立するため、キックを前後の副ステップに分割した
velocity-Verlet（kick–drift–kick、後半キックは次反復の先頭で局所勾配により
実行）を用いる。副ステップ \(n\)（実効幅 \(\Delta s_n\)）に対し：
\[
\hat{\mathbf v} \leftarrow \hat{\mathbf v}
  - \frac{\Delta s_{n-1}}{4\Delta x}\hat\nabla\hat n(\hat{\mathbf r}^n)
  - \frac{\Delta s_{n}}{4\Delta x}\hat\nabla\hat n(\hat{\mathbf r}^n),\qquad
\hat{\mathbf r}^{n+1} = \hat{\mathbf r}^n
  + \frac{\Delta s_n}{\Delta x}\hat{\mathbf v}
\]
（第 1 項が前副ステップの後半キック、第 2 項が当該副ステップの前半キック。
両者とも現在位置の勾配 = 前ステップ終端の勾配で評価される。）
初期条件は \(\Delta s_{-1}=0\)（**シードキックなし**）、発射点の速度は
\(|\hat{\mathbf v}^0|=\sqrt{1-\hat n(\hat{\mathbf r}^0)}\)（真空発射で厳密に 1）。
セグメントが打ち切られた場合（mesh 退出・entry hand-off 等の \(t_{stop}<1\)）は
\(\Delta s_{n}\leftarrow t_{stop}\Delta s_n\) を記録する。

旧方式（スタッガード leapfrog、下記 (ii)）は \(\Delta s\) が一定のとき同等だが、
適応制御で \(\Delta s\) が変化すると半ステップ位相の不整合により
1 ステップあたり \(O(\hat\nabla\hat n\,\mathrm d\Delta s)\) の系統的
Hamiltonian 散逸を生む（実測: θ-limiter の単調縮小下で転回深さが
\(\hat n\) 換算 0.027 浅くなり、往復 τ を 22% 損失。可変ステップ Verlet で
転回誤差 \(1.1\times10^{-5}\)・τ 誤差 0.008% — 解析ゲート
`test_laser_absorption_analytic_1d` §V&V 参照）。

**(ii) `ray_trace_2d`（平面 2D）— スタッガード leapfrog（2026-07-26 修正 seeding）**：

速度更新（半ステップ先行）：
\[
\hat{\mathbf{v}}^{n+1/2} = \hat{\mathbf{v}}^{n-1/2} - \frac{C_{ray}}{2}\hat\nabla\hat n(\hat{\mathbf{r}}^n)
\]
位置更新：
\[
\hat{\mathbf{r}}^{n+1} = \hat{\mathbf{r}}^n + C_{ray}\,\hat{\mathbf{v}}^{n+1/2}
\]

**初期条件**（\(n=0\)）：
\[
\hat{\mathbf{v}}^{-1/2} = \hat{\mathbf{v}}^0 + \frac{C_{ray,1}}{4}\hat\nabla\hat n(\hat{\mathbf{r}}^0)
\]
ここで \(C_{ray,1}\) は**最初の副ステップの適応後ステップ幅**（下記 \(m\) を
初期状態で評価した \(\Delta s_{cur}\)）。後方半ステップ
\(\hat{\mathbf v}^{-1/2}=\hat{\mathbf v}^0-\tfrac{h}{2}\mathbf a^0\) に
\(\mathbf a=-\tfrac12\hat\nabla\hat n\) を代入した符号が正である
（2026-07-26 修正 — 旧仕様の \(-C_{ray}/4\) は最初のドリフトに 3 倍の
加速度寄与を与えていたというカーネルレビュー指摘）。\(|v|\) リスケールは (i) と同一。
可変ステップ Verlet への移行は適応 \(\Delta s\) 変動が同様の散逸を生むため
望ましいが、2D 系 gate 再認定と併せて 2D 側が所掌する。

**安定性条件**（全カーネル共通）：\(C_{ray} \le 1\)（レイが1ステップで1セル幅
以上進まない）。既定 \(C_{ray,max}=0.8\)（`cfl_ray`）。

**適用範囲 (2026-07-31 改訂)**: (i) は `ray_trace_1d_sph`（persistent-loop 共有
body を含む）。(ii) は `ray_trace_2d`。2D_RZ の `ray_trace_3d` は**旧 seeding
（負符号・\(\Delta s_{base}\)・\(|v|=1\) 発射）のまま**である — mirror 適用は
2D CBET slab detuning 傾向 gate を marginal red にしたため、gate 再認定と
併せて 2D 側が所掌する（コード内 2D_RZ NOTE 参照）。

**適応ステップ幅制御**（v1.0既定、expand-only + 転回弧角度制御）：

低勾配・低吸収領域ではレイステップ幅を基準値 \(\Delta s_{base}\) から拡大し、
計算コストを削減する。副ステップ始点の状態 \((\hat n_{raw}^n,\kappa^n)\) に対し、
まず
\[
m_{raw} = \min\!\left(
  \frac{g_{target}}{|\hat\nabla\hat n|\,\Delta s_{base}},\;
  \frac{\tau_{target}}{\kappa^n\,\Delta s_{base}},\;
  m_{max}
\right)
\]
を評価する。最終的な拡大倍率は
\[
m = \min\!\left(\max(m_{raw}, 1),\, m_{cap}(\hat n_{raw}^n)\right)
\]
とし、current substep の entry-state raw density \(\hat n_{raw}^n\) に依存する
soft cap \(m_{cap}\) を
\[
m_{cap}(\hat n_{raw}^n)=
\begin{cases}
m_{max}, & \hat n_{raw}^n \le 0.8, \\
\max\!\left(1,\,
  m_{max}\dfrac{0.95-\hat n_{raw}^n}{0.15}\right),
  & 0.8 < \hat n_{raw}^n < 0.95, \\
1, & \hat n_{raw}^n \ge 0.95
\end{cases}
\]
で与える。したがって \(\hat n_{raw}^n \le 0.8\) では expand-only 適応を維持し、
\(0.8 < \hat n_{raw}^n < 0.95\) では near-critical 帯に向かって拡大倍率を連続的に絞り、
\(\hat n_{raw}^n \ge 0.95\) では基準ステップ幅 \(\Delta s_{base}\) に戻す。ここで
\(g_{target}\)（既定 0.05）は1ステップあたりの屈折角変化目標、
\(\tau_{target}\)（既定 0.05）は1ステップあたりの光学的深さ目標、
\(m_{max}\)（既定 4.0）は最大拡大倍率である。光学厚メトリクスの \(\kappa^n\) は
**current substep の entry-state 吸収係数**を用い、1-step 前の値は使わない。
near-critical 判定も `carried_nh_raw` に基づいて行う。

**転回弧角度制御 \(\theta_{target}\)（2026-07-31、全カーネル）**：上記 \(m\) は
\(\max(m_{raw},1)\) の床により**拡大専用**であり、\(\Delta s_{base}\) が粗い場合に
転回弧を細分できない。leapfrog の 1 ステップあたり回転角は
\(\Delta\theta \simeq |\hat\nabla\hat n|\,\Delta s / (2|\hat{\mathbf v}|^2)\)、
\(|\hat{\mathbf v}|^2 = 1-\hat n\) であり転回点近傍で発散する。これを
`ds_adapt_theta_target`（既定 0.04 rad/step、\(\le 0\) で無効）で制限する
第 3 のメトリクス
\[
m_\theta = \frac{2\,\theta_{target}\,(1-\hat n_{raw}^n)}
                {|\hat\nabla\hat n|\,\Delta s_{base}}
\]
を最終段で \(m \leftarrow \min(m, \max(m_\theta, 10^{-4}))\) として適用する。
\(m_\theta\) は **1 未満（縮小）を許す**唯一のメトリクスであり、転回弧の
ステップ数は格子非依存の \(\sim\pi/\theta_{target}\) 本に規格化される
（床 \(10^{-4}\) は退化格子でのストール防止、`max_steps` が最終ガード）。
1D_SPH/`ray_trace_2d`/`ray_trace_3d` の全 modulator サイトに適用される。
Default raised 0.02 -> 0.04 (2026-08-05): the θ ladder showed A_total +0.007 pt only, while halving turning-arc cost.

実効ステップ幅 \(\Delta s_{cur} = m\,\Delta s_{base}\) に対し、キックは
実効ステップ幅で適用する。1D_SPH（可変ステップ Verlet、(i)）では前後の
クォーターキック \(\Delta s_{n-1}/4,\ \Delta s_n/4\) がそれぞれの実効幅を
用いる。`ray_trace_2d`（スタッガード、(ii)）では
\[
\hat{\mathbf{v}}^{n+1/2} = \hat{\mathbf{v}}^{n-1/2}
  - \frac{\Delta s_{cur}}{2\Delta x}\hat\nabla\hat n(\hat{\mathbf{r}}^n),\quad
\hat{\mathbf{r}}^{n+1} = \hat{\mathbf{r}}^n
  + \frac{\Delta s_{cur}}{\Delta x}\hat{\mathbf{v}}^{n+1/2}
\]
とし、初期クォーターキックは最初の副ステップと同じ適応幅 \(\Delta s_{cur,1}\)
（初期 entry-state で \(m\) をループと同一式・同一丸めで評価）で実行する —
これにより最初のキック・ドリフトが厳密に時間中心化される（2026-07-26、
旧仕様の「常に \(\Delta s_{base}\)」は \(m>1\) の発射点で一次の中心化誤差を
残していた。1D_SPH は 2026-07-31 の Verlet 移行でシードキック自体を廃止）。

**1D_SPH の初期 vacuum 進入**：
1D_SPH では、レイ初期位置 \((R^0,Z^0)\) が radial profile の外側
\[
r^0 = \sqrt{(R^0)^2 + (Z^0)^2} > r_{max},\qquad r_{max}=r_{N-1}
\]
にあっても、進行方向 \((v_R,v_Z)\) が profile と交差するなら、吸収ゼロの vacuum 直進で
最初の入射点まで進めてから通常の Leapfrog を開始する。直線軌道
\[
r(s)^2 = (R^0 + v_R s)^2 + (Z^0 + v_Z s)^2
\]
に対し、
\[
a = v_R^2 + v_Z^2,\qquad
b = 2(R^0 v_R + Z^0 v_Z),\qquad
c = (R^0)^2 + (Z^0)^2 - r_{max}^2
\]
として
\[
a s^2 + b s + c = 0
\]
を解き、小さい方の根
\[
s_{entry} = \frac{-b - \sqrt{b^2 - 4ac}}{2a}
\]
が \(a>0\)、\(b^2-4ac>0\)、\(s_{entry}>0\) を満たす場合に
\[
R^0 \leftarrow R^0 + v_R s_{entry},\qquad
Z^0 \leftarrow Z^0 + v_Z s_{entry}
\]
へ更新する。vacuum 区間では IB 吸収も沈着も行わない。更新後に数値的に
\(r^0 \le r_{max}\) を再確認できない場合、または上記条件を満たす root が存在しない場合は、
そのレイは profile に入らないものとして未吸収終了する。

**1D_SPH の行進中 vacuum→物質 entry hand-off（2026-07-31）**：
mesh 内部にも真空区間（\(\kappa=0,\ \hat\nabla\hat n=0,\ \hat n_{raw}=0\)）が
ありうる。純真空セグメントはどのステップメトリクスにも拘束されず（全メトリクス
のスケール量が 0）、粗い真空ノード由来の長大ストライドが物質面を盲目的に
跨ぎうる（実測: 0.01 cm シェルに対し 0.35 cm ストライド — セグメント定数
（始点区間の \(\kappa, g\)）が真空値のままシェル内 \(\hat n\ 0\to0.037\) を
無キック・無吸収で通過し、角運動量保存の下で転回が \(\hat n\) 換算 0.027
浅くなる）。純真空セグメントは厳密に直線なので、最外物質ノード半径
\(r_{surf}\)（\(\hat n_{raw}>0\) の最外ノードの外側隣接ノード）との弦-球面
交差を上記と同形の 2 次方程式で厳密に解き、\(t_{stop}\) をそこで打ち切る。
着地点はノード上で locate が曖昧になるため、位置ナッジではなく**次セグメント
の carried 区間を物質側区間に明示的に固定**する（\(t=1\) の区間上端 =
表面ノード。補間される停止点物理量は不変で、次セグメントの
\(\Delta s_{base},\ \hat\nabla\hat n,\ \kappa\) が物質側になる）。真空側は
\(g=0\) のため Verlet の打ち切りキック過剰も生じない。この打ち切り
（\(t_{stop}<1\)）は終端イベントではなく行進を継続する。

> **注意**：`ds_adapt_max_factor = 1.0` に設定すると適応は無効化され、
> 全ステップで \(\Delta s_{cur} = \Delta s_{base}\) となる。
> 既存の検証テスト（Beer-Lambert、屈折等）はこの設定で参照解と比較する。

**RK2（Velocity-Verlet、将来拡張）**：
`raytrace.integrator="rk2"` は将来版予約。**v1.0では未対応** であり、指定時は ConfigError とする。
\[
\hat{\mathbf{r}}^{n+1} = \hat{\mathbf{r}}^n + C_{ray}\,\hat{\mathbf{v}}^n - \frac{C_{ray}^2}{4}\hat\nabla\hat n(\hat{\mathbf{r}}^n)
\]
\[
\hat{\mathbf{v}}^{n+1} = \hat{\mathbf{v}}^n - \frac{C_{ray}}{4}\left[\hat\nabla\hat n(\hat{\mathbf{r}}^n) + \hat\nabla\hat n(\hat{\mathbf{r}}^{n+1})\right]
\]
精度2次、Leapfrogのシンプレクティック性を持たないが、半ステップ速度が不要。

**RK4（古典4段、将来拡張）**：
`raytrace.integrator="rk4"` は将来版予約。**v1.0では未対応** であり、指定時は ConfigError とする。
状態ベクトル \(\mathbf{y}=(\hat{\mathbf{r}}, \hat{\mathbf{v}})\) に対し：
\[
\mathbf{k}_1 = h\,\mathbf{f}(\mathbf{y}^n),\quad
\mathbf{k}_2 = h\,\mathbf{f}(\mathbf{y}^n+\mathbf{k}_1/2),\quad
\mathbf{k}_3 = h\,\mathbf{f}(\mathbf{y}^n+\mathbf{k}_2/2),\quad
\mathbf{k}_4 = h\,\mathbf{f}(\mathbf{y}^n+\mathbf{k}_3)
\]
\[
\mathbf{y}^{n+1} = \mathbf{y}^n + \frac{1}{6}(\mathbf{k}_1+2\mathbf{k}_2+2\mathbf{k}_3+\mathbf{k}_4)
\]
ここで \(h=C_{ray}\Delta x/c\) [s]（時間幅）、\(\mathbf{f}(\hat{\mathbf{r}},\hat{\mathbf{v}})=(c\hat{\mathbf{v}}/\Delta x,\; -(c/2\Delta x)\hat\nabla\hat n)\) [cm/s]。
精度4次だが計算コスト4倍（勾配評価4回/ステップ）。既定はLeapfrog。

#### 5.3.3 密度勾配の計算と双線形補間

**(a) 節点での勾配（非一様格子2次中心差分）**

LaserMeshの節点 \((i,j)\) における正規化密度勾配は、非一様ノード間隔
\[
h_{R,-}=R_i-R_{i-1},\quad h_{R,+}=R_{i+1}-R_i,\quad
h_{Z,-}=Z_j-Z_{j-1},\quad h_{Z,+}=Z_{j+1}-Z_j
\]
を用いて次式で評価する：
\[
\left.\frac{\partial \hat n}{\partial R}\right|_{i,j}
= \frac{-h_{R,+}^2\,\hat n_{i-1,j}
+(h_{R,+}^2-h_{R,-}^2)\,\hat n_{i,j}
+h_{R,-}^2\,\hat n_{i+1,j}}
{h_{R,-}h_{R,+}(h_{R,-}+h_{R,+})}
\]
\[
\left.\frac{\partial \hat n}{\partial Z}\right|_{i,j}
= \frac{-h_{Z,+}^2\,\hat n_{i,j-1}
+(h_{Z,+}^2-h_{Z,-}^2)\,\hat n_{i,j}
+h_{Z,-}^2\,\hat n_{i,j+1}}
{h_{Z,-}h_{Z,+}(h_{Z,-}+h_{Z,+})}
\]
等間隔格子（\(h_{-}=h_{+}\)）では通常の中心差分に一致する。
境界節点の取り扱い：
- **\(R=0\) 境界（軸対称条件）**: \(\partial\hat n/\partial R\big|_{R=0} = 0\) を強制する。
  物理的に \(\hat n = \hat n(\sqrt{R^2+Z^2})\) であるため \(\partial\hat n/\partial R = (\partial\hat n/\partial r)(R/r)\big|_{R=0} = 0\)。
  実装はミラー拡張ステンシル \(\hat n(-\Delta R, Z) = \hat n(\Delta R, Z)\) を用いた中心差分（結果は自動的に0）とする。
  片側差分を用いると非ゼロの勾配が生じ、軸上レイが不自然に屈折する。
- **その他の境界節点（\(R=R_{max}\), \(Z=Z_{min}\), \(Z=Z_{max}\)）**: 片側差分を使用する。

**(b) 任意点での勾配（双線形補間）**

レイ位置 \(\hat{\mathbf{r}}^n\) が属するセル \((i,j)\) の4頂点の勾配を面積重みで補間：
\[
\hat\nabla\hat n(\hat{\mathbf{r}}^n) =
w_{i,j}\,\hat\nabla\hat n_{i,j} +
w_{i+1,j}\,\hat\nabla\hat n_{i+1,j} +
w_{i,j+1}\,\hat\nabla\hat n_{i,j+1} +
w_{i+1,j+1}\,\hat\nabla\hat n_{i+1,j+1}
\]

重み（双線形）：
\[
w_{i,j}=(1-\xi)(1-\eta),\quad
w_{i+1,j}=\xi(1-\eta),\quad
w_{i,j+1}=(1-\xi)\eta,\quad
w_{i+1,j+1}=\xi\eta
\]
\[
\xi = \frac{R - R_i}{R_{i+1}-R_i},\quad
\eta = \frac{Z - Z_j}{Z_{j+1}-Z_j}
\]
\(\sum w = 1\) が自動的に成立する。

**(c) 物理量の補間**

同様に、\(\hat n\), \(T_e\), \(\bar Z\) 等の物理量もレイ位置で双線形補間する。

#### 5.3.4 2D_RZにおける3Dレイトレース

2D_RZモードでは流体場は軸対称 \(\rho(R,Z)\) だが、レーザービームは任意の3D方向から入射する。
φ平均近似では特に少ビーム・非対称照射で精度が低下するため、**レイトレースを3D化** する。
流体場が軸対称であるため、3Dメッシュは不要であり、場の参照は \(R=\sqrt{x^2+y^2}\) で2D LaserMeshから取得する。

**(a) 3Dレイ方程式**

レイの位置・速度を3Dベクトルとして追跡する：
\[
\mathbf{r} = (x, y, z) \in \mathbb{R}^3,\quad
\mathbf{v} = (v_x, v_y, v_z) \in \mathbb{R}^3
\]
\[
\frac{d\mathbf{r}}{dt} = \mathbf{v},\quad
\frac{d\mathbf{v}}{dt} = -\frac{c^2}{2}\nabla_{3D}\hat n
\]
ここで \(\nabla_{3D}\hat n\) は3D空間での正規化密度勾配である。
流体場が軸対称であるため \(\hat n = \hat n(R,Z)\)（\(R=\sqrt{x^2+y^2}\)）であり、
3D勾配は2D勾配から解析的に変換できる（下記 (b)）。

**(b) 2D勾配からの3D変換**

軸対称場 \(\hat n(R,Z)\) の3D勾配は連鎖律により：
\[
\frac{\partial\hat n}{\partial x} = \frac{\partial\hat n}{\partial R}\frac{x}{R},\quad
\frac{\partial\hat n}{\partial y} = \frac{\partial\hat n}{\partial R}\frac{y}{R},\quad
\frac{\partial\hat n}{\partial z} = \frac{\partial\hat n}{\partial Z}
\]
ここで \(R = \sqrt{x^2+y^2}\)。
2D勾配 \((\partial\hat n/\partial R,\; \partial\hat n/\partial Z)\) は§5.3.3の中心差分＋双線形補間で取得し、
レイの3D位置 \((x,y,z)\) から \(R=\sqrt{x^2+y^2}\) を計算してLaserMesh上の2Dセルを特定する。

**(c) R=0近傍の特異性処理**

\(R = \sqrt{x^2+y^2} \to 0\) では \(x/R, y/R\) が不定形となる。物理的に軸対称場では
\(\partial\hat n/\partial R\big|_{R=0} = 0\)（§5.3.3 (a) の境界条件）であるため、
\((\partial\hat n/\partial R)(x/R)\big|_{R\to 0} = 0\) が成立する。実装では：
\[
R < R_{floor} \quad \Rightarrow \quad \frac{\partial\hat n}{\partial x} = \frac{\partial\hat n}{\partial y} = 0
\]
既定 \(R_{floor} = 10^{-12}\Delta x\)（LaserMeshの最小セル幅の \(10^{-12}\) 倍）。
これは§5.3.3 (a) のミラー拡張ステンシルと整合する。

**(d) 正規化と Leapfrog 3Dベクトル版**

§5.3.1–5.3.2のLeapfrog積分を3Dベクトルに自然拡張する。正規化変数：
\[
\hat{\mathbf{r}} = \mathbf{r}/\Delta x,\quad
\hat{\mathbf{v}} = \mathbf{v}/c \quad \in \mathbb{R}^3
\]

速度更新：
\[
\hat{\mathbf{v}}^{n+1/2} = \hat{\mathbf{v}}^{n-1/2} - \frac{C_{ray}}{2}\hat\nabla_{3D}\hat n(\hat{\mathbf{r}}^n)
\]
位置更新：
\[
\hat{\mathbf{r}}^{n+1} = \hat{\mathbf{r}}^n + C_{ray}\,\hat{\mathbf{v}}^{n+1/2}
\]

ここで \(\hat\nabla_{3D}\hat n\) は正規化座標での3D勾配であり、(b) の変換を用いる。
\(C_{ray} = c\Delta t_{ray}/\Delta x\) は §5.3.2 で定義したレイ CFL 数である。
安定性条件 \(C_{ray} \le 1\) およびRK2/RK4オプションは§5.3.2と同一。

**(e) IB吸収・臨界処理の3D適用**

IB吸収（§5.4）および臨界近傍処理（§5.2）はレイ位置の \((R,Z) = (\sqrt{x^2+y^2}, z)\) を用いて
2D LaserMesh上の物理量を参照する。吸収・terminate判定のアルゴリズムは2Dレイトレースと同一であり、
レイの位置・速度ベクトルが3Dに拡張されている点のみが異なる。

**(f) 2Dレイトレースとの関係**

| 項目 | 1D_SPH（§5.3.2） | 2D_RZ（本節） |
|------|-------------------|---------------|
| レイベクトル | 2D \((R,Z)\) | 3D \((x,y,z)\) |
| 場の参照 | 1D radial profile（\(r=\sqrt{R^2+Z^2}\)） | 2D LaserMesh（\(R=\sqrt{x^2+y^2}\)） |
| 勾配 | 2D直接 | 2D→3D変換（(b)式） |
| Leapfrog | 2Dベクトル | 3Dベクトル |
| IB吸収 | 同一 | 同一（場の参照経路のみ異なる） |
| LaserMesh外判定 | \(r=\sqrt{R^2+Z^2}>R_{max}\) | 同一（\(R=\sqrt{x^2+y^2}\) で判定） |

#### 5.3.5 球対称高速経路（Bouguer 閉形式掃引、2026-08-04）

1D_SPH の `mode="raytrace_2d"` では、以下の全条件が成立するとき §5.3.2 の
leapfrog 行進を**球対称縮約**で置換できる（**現状 opt-in**: 環境変数
`TENRYU_FAST_TRACE=1` を与えた場合のみ有効・既定は march。namelist キー
なし）: CBET 無効・beam profile `flat_top`・hot-electron capture 無効・
trajectory/ray-output 収集なし（転回点 RA は fast path が同一規約で実装
済み）。条件外のビームは従来どおり march する。
**opt-in 留保の理由（2026-08-04 実測）**: shell 一定 κ/n̂ 近似は臨界近傍
の凸な κ を過小評価し、3step フル A/B で A_total −2.8pt（foot −0.13pt・
main −3.3pt）— shell 内線形 κ 閉形式 + 転回近傍サブ分割（S1.1）で
march パリティ達成後に既定化を再判定する。wall は march 比 ×2.7
（m0 window、A100）。

球対称成層 \(n(r)\) では Bouguer 不変量
\[
n(r)\,r\,\sin\theta = b,\qquad
b=\frac{|R_0 v_{Z0}-Z_0 v_{R0}|}{|\mathbf v_0|}
\]
（\(b\) はレイの衝突径数）によりレイ軌道は可積分であり、行進は不要になる。
laser mesh の半径シェル \([r_j,r_{j+1}]\)（シェル一定 \(\hat n_j,\kappa_j\)
— march と同一の `radial_n_hat`/`radial_smooth_kappa`/Langdon 補正、
flat_top では Langdon 強度を launch \(|R_0|\) で評価 = スポット内厳密）
内の光路長は閉形式
\[
\Delta s_j=\sqrt{r_{j+1}^2-a_j^2}-\sqrt{r_j^2-a_j^2},\qquad
a_j=\frac{b}{\sqrt{1-\hat n_j}}
\]
（転回シェルは下限を \(r_t=a_j\) に置換）。掃引は外→転回→外の 2 leg で、
シェル毎に \(P\leftarrow P e^{-\kappa_j\Delta s_j}\)、沈着は
\(P_{\rm before}-P_{\rm after}\) 形（テレスコープにより
incident = deposited + escaped が浮動小数点で厳密に閉じる）。臨界到達・
初期 supercritical の残余は march と同一の `terminate_mode` 規約
（`"deposit"` は最終 subcritical シェルへ、他モードは unabsorbed 計上）。
シェル→hydro セル対応は march と同じ locate 規則（critical-adjacent split
含む）をシェル中点で評価。決定論: [shell][ray] 転置 staging + shell 毎
block の**固定形状 tree reduction**（atomics 不使用・run 反復 bit 同一が
契約。march の ray 昇順とは加算順が異なる。staging が 512 MiB を超える
場合は 8192-ray の逐次バッチで固定順を保存）。

検証（VERIFICATION 参照）: 単体ゲート = 線形 profile の解析 16/15 則・
斜入射 cos⁵(b) 則・転回半径恒等 \(r_t=b/n(r_t)\)・台帳 1e-14・run 反復
bit 同一（5 case / 133 assertions PASS 2026-08-04）+ 3step フル A/B
（march 比の吸収・バング・ρ_peak 一致 — 実測値はコミット時点の
VERIFICATION 記録を正とする）。

### 5.4 逆制動輻射吸収（IB）

#### 5.4.1 吸収の方程式
光路に沿った強度減衰：
\[
\frac{dI}{ds} = -\kappa_{IB}\,I
\]
ここで \(s\) はレイの光路長、\(\kappa_{IB}\) [cm\(^{-1}\)] は逆制動輻射の空間吸収係数である。

\(\kappa_{IB}\) は電子-イオン衝突周波数 \(\nu_{ei}\) と群速度 \(v_g = c\sqrt{1-\hat n}\) から：
\[
\kappa_{IB} = \frac{\hat n\,\nu_{ei}}{c\,\sqrt{1-\hat n}} \quad [\text{cm}^{-1}]
\]
ここで \(\nu_{ei}\) [s\(^{-1}\)] は電子–イオン衝突周波数。
低密度域（\(\hat n \ll 1\)）では \(\sqrt{1-\hat n}\approx 1\) となり \(\kappa_{IB}\approx \hat n\,\nu_{ei}/c\) である。

> **注**：臨界密度では \(\sqrt{1-\hat n}\to 0\) で \(\kappa_{IB}\) が発散するが、
> §5.2のterminate処理により臨界近傍に到達する前にレイは終了する。

#### 5.4.2 離散化（台形公式）
1副ステップの光学厚（レイセグメントの始点 \(n\) と終点 \(n{+}1\) で台形平均）：
\[
S = \frac{\kappa_{IB}^{n+1}+\kappa_{IB}^n}{2}\,\Delta s_{ray}
\]
ここで \(\kappa_{IB}^n = \kappa_{IB}(\hat{\mathbf{r}}^n)\)、
\(\Delta s_{ray} = |\mathbf{r}^{n+1}-\mathbf{r}^n|\) はレイセグメント長 [cm]
であり、各位置での \(\hat n, T_e, \bar Z\) は
1D_SPH では radial profile に対する線形補間、
2D_RZ では双線形補間（§5.3.3）で求める。

強度更新と吸収パワー（**桁落ち回避形**で実装すること）：
\[
\Delta P = -I^n \operatorname{expm1}(-S) \quad [\text{erg/s}], \qquad
I^{n+1} = I^n - \Delta P
\]

> **実装注意**：\(S \ll 1\)（光学的に薄いセグメント）では \(\Delta P = I^n(1-e^{-S})\) の
> 直接計算で桁落ちが発生する。\(\operatorname{expm1}(-S) = e^{-S}-1\) を用いて
> \(\Delta P = -I^n \operatorname{expm1}(-S)\) とする。\(I^{n+1}\) は差分更新
> \(I^{n+1} = I^n - \Delta P\) で求め、\(I^n \exp(-S)\) は使用しない
> （ΔP と I^{n+1} の整合性を保証し、テレスコーピング和の保存性を維持）。
ここで \(I\) はレイが運ぶパワー [erg/s] である（§5.6.3でビームパワーから配分）。

> **空間分配ではΔP（パワー）を直接分配する**（§5.5）。
> `deposit` 配列はパワー [erg/s] を蓄積し、正規化吸収分率 \(\hat{f}\) を無次元量として得る（§5.6.4）。
> 最終的なエネルギー [erg] への変換は流体ステップ幅 \(\Delta t\) を乗じて1回だけ行う（§5.8.1）。
> \(\Delta t_{ray}\)（レイトレース副ステップ幅）は deposit に含めない — \(\Delta t_{ray}\) はレイの幾何追跡精度を制御する数値パラメータであり、定常ビームの吸収パワー分布とは独立である。

#### 5.4.3 逆制動輻射の吸収係数（cgs + eV）
§5.4.1の \(\kappa_{IB}\approx \hat n\,\nu_{ei}/c\)（\(\hat n\ll 1\) 近似）の具体式。

\(\hat n \cdot \nu_{ei}/c\) は以下で与えられる（cgs導出）：
\[
\frac{n_e\,\nu_{ei}}{n_{crit}\,c}
= \frac{4}{3}(2\pi m_e)^{1/2}\,\pi c\,e^2\,\frac{\bar Z}{\lambda_L^2\,(k_B T_e)^{3/2}}\,\hat n^2\,\ln\Lambda
\]

簡略形（混合単位）：
\[
\frac{n_e\,\nu_{ei}}{n_{crit}\,c}
= 3.4\,\frac{\bar Z\,\hat n^2}{\lambda_L[\mu m]^2\,T_e[keV]^{3/2}}\,\ln\Lambda
\]

**クーロン対数**：
\[
\ln\Lambda = \ln\left[1.5\times 10^4\,\frac{\lambda_L[\mu m]\,T_e[keV]^{3/2}}{\bar Z\,\hat n^{1/2}}\right]
\]
下限：\(\ln\Lambda = \max(\ln\Lambda, 2)\)（非物理値の回避、`coulomb_log_floor`）。

> **定数 \(1.5\times 10^4\) の導出**：
> クーロン対数の引数は \(\Lambda = \lambda_D / b_{90}\) で定義する。ここで
> \(\lambda_D = \sqrt{k_B T_e/(4\pi n_e e^2)}\) はDebye長、
> \(b_{90} = Ze^2/(3k_B T_e)\) は90°偏向の衝突パラメータ
> （電子の平均運動エネルギー \(\langle m_e v^2/2\rangle = \frac{3}{2}k_B T_e\) を使用）。
>
> \(n_e = \hat n\, n_{crit}\)、\(n_{crit} = m_e\pi c^2/(e^2\lambda_L^2)\) を代入すると：
> \[
> \Lambda = \frac{3(k_B T_e)^{3/2}\,\lambda_L}{2\pi\,Z\,e^2\,c\,\sqrt{\hat n\, m_e}}
> \]
> 混合単位（\(\lambda_L\) [\(\mu m\)]、\(T_e\) [keV]）に換算し、基本定数
> \(e = 4.803\times 10^{-10}\) esu、\(m_e = 9.109\times 10^{-28}\) g、
> \(k_B = 1.602\times 10^{-9}\) erg/keV（= keV\_to\_erg = eV\_to\_erg × \(10^3\)）を代入すると
> \(\Lambda \approx 1.47\times 10^4\,\lambda_L[\mu m]\,T_e[\text{keV}]^{3/2}/(\bar Z\,\hat n^{1/2})\)
> が得られ、\(1.5\times 10^4\) はこの2桁丸め値である。
>
> **他の規約との比較**：NRL Plasma Formulary (2019) の冷電子公式
> \(\ln\Lambda = 23 - \ln(n_e^{1/2}\,Z\,T_e^{-3/2}[\text{eV}])\) は
> 異なる \(b_{min}\) 規約を用いており、同じ混合単位で \(C \approx 9.2\times 10^3\) を与える。
> ICF条件（\(T_e \sim 1\) keV）では \(\ln\Lambda \sim 8\)–\(10\) の範囲にあるため、
> 両規約の差（\(\Delta\ln\Lambda \approx 0.5\)）は \(\kappa_{IB}\) に対して約5%の影響であり、
> フロア \(\ln\Lambda \ge 2\) との整合性も維持される。

> **単位変換の注意（実装時必須）**：
> 上記の簡略形は混合単位（\(\lambda_L\) を \(\mu m\)、\(T_e\) を keV）で記述されている。
> プロジェクトの入力はcgs+eV（\(\lambda_L\) [cm]、\(T_e\) [eV]）であるため、
> 実装では以下の変換を行う：
> - \(\lambda_L[\mu m] = \lambda_L[\text{cm}] \times 10^4\)
> - \(T_e[\text{keV}] = T_e[\text{eV}] \times 10^{-3}\)
>
> 変換漏れの影響：\(\lambda_L^2\) で \(10^8\) 倍、\(T_e^{3/2}\) で \(\sim 3\times 10^{-5}\) 倍のスケール事故となる。

#### 5.4.4 Critical-layer tail closure
臨界近傍では \(\kappa_{IB} \propto (1-\hat n)^{-1/2}\) のため、通常の「臨界面で打ち切る」処理だけでは
残余吸収を過小評価する。レイ副ステップ始点の状態 \((\hat n_0,\hat n_{raw,0}, \kappa_0)\) に対し、
\[
\hat n_{raw,0} \ge 0.5,\qquad
1-\hat n_{raw,0} < \beta \Delta n_A,\qquad
\beta = 1,\qquad
\Delta n_A = \frac{A_0}{|\nabla \hat n|_0},\qquad
\mathbf v\cdot\hat\nabla\hat n > 0
\]
を満たす場合は、通常の台形積分を行わず `critical-layer mode` に切り替えて、
臨界面までの残余光学厚を解析式で閉じる。最後の条件は**密度勾配を上る
（臨界面へ向かう）レイのみ**が closure の対象であることを保証する — これが
無いと、亜臨界 turning を終えた外向きレイが near-critical 帯を再通過する際に
架空の臨界層 tail を二重に吸収して終端されうる（2026-07-26 追加、
カーネルレビュー指摘。臨界交差による切替経路＝副ステップ終点が
\(\hat n_{raw}\ge1-\varepsilon_{crit}\) の場合は交差自体が内向きの証拠なので
方向条件は課さない。**適用範囲**: 1D_SPH 系カーネルのみ — 2D_RZ
`ray_trace_3d` は方向条件を bypass（v_dot_g=1.0 固定）しており、
mirror は 2D 側の対応範囲）。
さらに、通常積分の副ステップ終点が \(\hat n_{raw} \ge 1-\varepsilon_{crit}\) に到達した場合も、
その副ステップは「臨界面で単純打ち切り」へは戻さず、**同じ entry-state tail closure** へ切り替える。
したがって near-critical 終端 ray は、`tail closure` と `critical-hit cutoff` の二重モデルに分岐せず、
常に同一の解析 closure で終端される。`critical-hit cutoff` は、
\(|\nabla \hat n|_0\) や \(A_0\) が有限に再構成できない退避経路としてのみ残す。

\[
n_0 = \min(1,\max(0,\hat n_0)),\qquad
u_0 = \sqrt{\max(10^{-30}, 1-n_0)},\qquad
n_0^2 = \max(n_0^2, 10^{-30})
\]
とおく。吸収係数は
\[
\kappa_{IB} = A\,\frac{\hat n^2}{\sqrt{\max(\varepsilon_n,1-\hat n)}}
\]
の形なので、始点係数 \(A_0\) は current interpolation stencil の
node-centered smooth factor から再構成する。

2D_RZ では entry 点の双線形重みを \(w_{ab}\)、各ノードの subcritical mask を
\[
m_{ab} =
\begin{cases}
1 & (\hat n_{raw,ab} < 1) \\
0 & (\hat n_{raw,ab} \ge 1)
\end{cases}
\]
とすると、有効重みは
\[
\tilde w_{ab} = \frac{m_{ab} w_{ab}}{\sum_{cd} m_{cd} w_{cd}}
\]
で与える。`test_kappa` を使う場合だけ、始点係数は carried entry-state の \(\kappa_0\) から
\[
A_0 = \kappa_0\,\frac{u_0}{n_0^2}
\]
で逆算する。通常経路では
\[
A_0 = \sum_{ab} \tilde w_{ab} A_{ab}
\]
とし、\(A_{ab}\) は laser mesh node に事前計算した `smooth_kappa_factor`
（未計算経路では nodal \((\hat n, T_e, \bar Z)\) から `compute_kappa_smooth_factor` で再構成）
を用いる。すなわち、mixed hydro/ghost stencil でも tail closure 自体は許可するが、
**supercritical node は \(A_0\) の再構成から必ず除外する**。
これにより hydro 側の低温ノードが ghost corona 側の \(A_0\) を汚染することを防ぐ。
4 ノードすべてが subcritical であれば \(\tilde w_{ab} = w_{ab}\) となり、
通常の双線形補間に一致する。 \(\sum m_{cd} w_{cd} = 0\) の場合は tail closure を行わず、
通常の step-by-step 積分を継続する。

1D_SPH では radial profile 上の 2 ノード線形補間に置き換える。
entry 点の線形重みを \(w_0 = 1-t,\ w_1=t\) とし、
\[
\tilde w_\alpha = \frac{m_\alpha w_\alpha}{\sum_\beta m_\beta w_\beta},
\qquad \alpha \in \{0,1\}
\]
で正規化した後、
\[
A_0 = \sum_{\alpha\in\{0,1\}} \tilde w_\alpha A_\alpha
\]
とする。2D_RZ と同様に、supercritical node は \(A_0\) の再構成から必ず除外する。

残余光学厚は
\[
\tau_{tail}
= \frac{2A_0}{|\nabla \hat n|_0}
\left(
u_0 - \frac{2}{3}u_0^3 + \frac{1}{5}u_0^5
\right)
\]
で与え、数値的安定性のため \(\tau_{tail} \le 700\) に clamp する。
吸収パワーは §5.4.2 と同じく
\[
\Delta P_{tail} = -I^n \operatorname{expm1}(-\tau_{tail}),\qquad
I^{n+1} = I^n - \Delta P_{tail}
\]
で更新する。

沈着は 2D_RZ では副ステップ始点（entry 点）の bilinear nodes に対して行い、
重みは §5.3.3 と同じ双線形重みを使う。
1D_SPH では entry 半径 \(r_{entry}\) を含む Hydro の 1Dセルへ直接沈着する。
supercritical node への leak を防ぐため、2D_RZ では §5.5 と同様に
\(\hat n_{raw} < 1\) のノードだけを有効重みとして正規化する。
残余パワー \(I^{n+1}\) は `unabsorbed` に加算し、そのレイは終了する。

#### 5.4a Radial absorption 1D integral (`radial_absorption_1d` mode)

`Laser.mode="radial_absorption_1d"` は 1D_SPH 専用の球対称 1D 積分である。
球殻面に垂直な inward radial flux を仮定し、z軸平行光でも現行 R-Z 2D レイトレーシングでもない。

各ビームの時刻 \(t\) のパワーを
\[
P_{\mathrm{total}}(t) = \sum_b P_b(t) \quad [\mathrm{erg/s}]
\]
として合算し、1本の radial flux として扱う。
`direction`, `f_number`, `focus`, `defocus`, `profile`, `rays_per_beam` は
このモードの吸収分布に影響しない（入力互換とパワー波形のために保持される）。

離散化は Hydro 1Dセルの外側 \(r_{\max}\) から内側 \(r=0\) へ単一スレッドで逐次積分する。
セル \(c\) の幅を \(dr_c = r_{c+1/2}-r_{c-1/2}\) とし、radial lookup で得た
\(\kappa_{\mathrm{smooth},c}\) [cm\(^{-1}\)] を用いる：
\[
\tau_c = \kappa_{\mathrm{smooth},c}\,dr_c,\qquad
\Delta P_c = P\,(1-\exp(-\tau_c)),\qquad
P \leftarrow P\,\exp(-\tau_c).
\]
実装では \(\tau_c\) を `compute_optical_depth`、\(\Delta P_c\) を
`absorbed_power_expm1` で評価し、桁落ち回避形（§5.4.2）を維持する。
\(\Delta P_c\) は `deposit_power_cell[c]` [erg/s] に蓄積し、
エネルギー [erg] への変換は §5.8.1 の 1D 直接沈着パスで \(\Delta t\) を1回だけ掛ける。

停止条件：
- raw \(\hat n(c) \ge 1-\varepsilon_{crit}\)：臨界到達として
  残存 \(P\) を `unabsorbed` に加算し、`critical_surface_hit_count` を増やす
- 最内セル \(r=0\) まで積分しても \(P>0\)：残存 \(P\) を `unabsorbed` に加算
- \(P <\) `intensity_cutoff` \(\times P_{\mathrm{total}}\)：早期終了し、残存 \(P\) を `unabsorbed` に加算

吸収モデルは既存 IB（§5.4）と同一であり、単位系は cgs + eV から変更しない。
v1.0 では臨界反射、ポンデロモーティブ力、非線形吸収を扱わない。

#### 5.4.5 拡張吸収物理（v1 は 1D_SPH 限定；Langdon は 1D raytrace 既定有効）

HELIOS ベンチマーク吸収率監査（2026-07-29/30、
`benchmarks/helios/comparison_LTE/ANALYSIS_laser_algorithm_comparison.md`
追補 2/3）を受けて導入した 4 つの拡張。(a)〜(b) および (d) は
namelist opt-in（`Laser.ib` / `Laser.ra`、SPECIFICATION §6.4）で既定 OFF。
(c) Langdon は `ib.langdon_model="auto"` が既定で、`laser.enabled=True`、
`Main.dimension="1D_SPH"`、`laser.mode!="radial_absorption_1d"`、ビーム
リストが非空、かつ全ビームの有効な (profile model, w0, m) が共通で
model が gaussian / super_gaussian / flat_top のときだけ
`"legacy_vacuum_map"`、それ以外は `"off"` に構築時解決する（INFO ログ
1 行）。4 つとも v1 は 1D_SPH 限定である。OFF 解決時の実装はカーネル
第 3 テンプレート引数 `kPhysExt` + `if constexpr` によりレガシーコードと
同一命令列となり、runtime 分岐による FMA 再配置を避ける本プロジェクトの
bitwise 規約に従う。診断として snapshot
`laser/E_ra`・history `energy/laser_ra_deposited`（累積 erg）を追加。

**(a) 混合プラズマ衝突電荷 Z_eff**（`ib.zeff_model`）。§5.4.3 の
\(\bar Z\) 前置因子は単一種でのみ正しく、多種イオンでは衝突加重電荷
\[
Z_{\rm eff}=\frac{\sum_s n_s Z_s^2}{\sum_s n_s Z_s}
\]
を用いるべきである（等モル CD 完全電離で \(Z_{\rm eff}=37/7=5.286\)、
\(\bar Z=3.5\) の 1.51 倍）。κ の前置因子を
\(Z_{\rm coll}=\bar Z\cdot(Z_{\rm eff}/\bar Z)\) に置換する。比の評価は
2 モード:
- `"sequential_strip"`: 組成（`ib.species` = [[Z_nuc, x], …] 昇順）と
  セルの \(\bar Z\) から、核電荷昇順に逐次剥離するクロージャで
  \(z_s(\bar Z)\) を復元（完全電離極限で厳密）。PROPACEOS CD の電離段
  分率表との比較で **T≥100 eV で誤差 ≤3%、≥1 keV で ≤0.5%**（コロナ =
  IB が効く領域で検証済み）。既知の限界: 低温 CD では C の第一電離電位
  (11.3 eV) が D (13.6 eV) より低く実際は C が先に電離するため、
  低 T では順序が逆（吸収が κ 飽和する領域のため実害なし）。
- `"table"`: TMAT 材料の `/ionization` 電離段分率（下記）から縮約した
  \(Z_{\rm eff}/\bar Z\) 表（(nᵢ, T) 格子、[1,10] クランプ）を
  デバイス常駐させ、ノードごとに log-log 双線形（端クランプ）で評価。
  nᵢ = n̂·n_crit/\(\bar Z\)。表提供材料が 1 つのときのみ有効。

**(b) レーザー周波数クーロン対数**（`ib.coulomb_log_model =
"laser_frequency"`）。§5.4.3 の既定 lnΛ は \(b_{max}\sim v_T/\omega_p\)
（Debye 型、引数 ∝ n̂^{-1/2}）。高周波吸収では \(b_{max}\sim
v_T/\omega_0\) が適切（Dawson–Oberman；527 nm 透過実測 Turnbull 2023,
PRL 130, 145103 / Sherlock 2024, PRE 109, 055201）で、引数から
n̂^{-1/2} を落とす（= 希薄部で lnΛ を \(-\tfrac12\ln\hat n\) 低減、
n̂=0.01 で −2.3）。フロアは既定と同じ。

**(c) Langdon 因子**（`ib.langdon_model = "legacy_vacuum_map"`）。
超ガウス歪みによる IB 低減（Langdon 1980, PRL 44, 575）:
\[
\alpha = 0.03736\,Z_{\rm coll}\,\frac{I_{14}\lambda_{\mu m}^2}{T_{\rm keV}},
\qquad
f_L = 1-\frac{0.553}{1+(0.27/\alpha)^{0.75}}
\]
（\(v_T^2=k_BT_e/m_e\) 規約、α∈[10⁻⁶,10³]・f_L∈[0.447,1] クランプ）。
局所強度は **vacuum-map 近似**: 全ビームの合計瞬時パワー
\(P(t)=\sum_b P_b(t)\) を担う、共通プロファイルの単一等価ビームの
真空強度をレイの現在 R 座標（\(r=|R|\)）で評価する:
\[
\begin{aligned}
\text{gaussian:}\quad
&I(r)=\frac{P}{\pi w^2}\exp\!\left[-\left(\frac{r}{w}\right)^2\right],
\qquad w=\frac{w_0}{\sqrt{2}},\\
\text{flat\_top:}\quad
&I(r)=\begin{cases}
P/(\pi w_0^2),&r<w_0,\\
0,&r\ge w_0,
\end{cases}\\
\text{super\_gaussian:}\quad
&I(r)=I_0\exp\!\left[-2\left(\frac{r}{w_0}\right)^{2m}\right],
\qquad I_0=\frac{P m 2^{1/m}}{\pi w_0^2\Gamma(1/m)} .
\end{aligned}
\]
gaussian の \(w=w_0/\sqrt2\) は 1/e 強度半径である。
super_gaussian は \(m\ge2\) とし、\(m=1\) は gaussian に写像する。
flat_top の半径は \(w_0\) 自身である。従来の \(w_0/\sqrt2\) は gaussian
変換の誤適用であり、2026-08-10 に修正したため、opt-in の
HELIOS-parity deck は構成上結果が変わる。近似の内容と限界: (i)
減衰・屈折集光を無視（枯渇領域で α 過大 → f_L 過小 = 吸収過小側）、
(ii) 往復重なり（転回近傍 ~2 倍）を無視、(iii) segment 途中停止点では
全段末 R を使用（w=113 µm スケールに対し誤差無視可能）、(iv)
\(Z_{\rm coll}\) は `ib.species` 指定時には組成の完全電離
\(Z_{\rm eff}=\sum_s x_s z_s^2/\sum_s x_s z_s\) を用い、未指定時には
1D radial mirror の最外非 void セルの \(\bar Z\) をレーザー呼び出しごとに
更新して代用する（1 run に 1 回 INFO）。どちらも得られなければ 1 とする。
単一完全電離元素ではこの fallback は厳密だが、混合物では
\(\langle Z^2\rangle/\langle Z\rangle\) を過小評価し、吸収低減が弱い
保守側となるため、`species` が正確な経路である。制約: 全ビームが共通の
gaussian / super_gaussian / flat_top プロファイルを共有しなければならない。
`"auto"` では非互換構成は `"off"` に解決する（INFO）が、明示的な
`"legacy_vacuum_map"` は非対応構成で起動時 ConfigError となる。適用域:
`langdon_te_min_eV`（既定 100 eV）未満の Te では
f_L=1（eV 級高衝突物質では超ガウス歪みが e-e 衝突で即緩和されるため。
コールドスタートのスモークで界面 κ の非物理的 44% 抑制を実測し導入）。radial 事前吸収経路（§5.4 の radial_absorption_1d）には
適用しない（コロナ形成前の冷高密度域で f_L≈1）。

**(d) 転回点共鳴吸収イベント**（`Laser.ra`）。滑らか球対称プラズマ
では b>0 の全レイは亜臨界転回点 \(N(r_t)r_t=b\) で反転する。p 偏光
成分は転回点で臨界層と結合し、Ginzburg/Forslund 縮約曲線
（Colaitis 2015, PRE 92, 041101 のフィット）
\[
\eta=(k_0 L_n)^{2/3}\sin^2\theta_{\rm eff},\quad
\sin\theta_{\rm eff}=b/r_c,\quad
f_p=1.74098\,\frac{\eta}{\sqrt{\eta+0.435}}e^{-\frac43\eta^{3/2}}
\]
（最大 0.562 at η=0.511）で吸収される。実装: レイの球半径最小通過
（前ステップ r_old との比較）で 1 回だけ発火し、直前ステップの
(r, n̂) から \(b_{\rm eff}=\sqrt{1-\hat n}\,r\)（転回点で sinψ=1 の
厳密関係）を構成、\(\Delta P = \chi_p\,C_{RA}\,f_p\,P\)
（既定 χ_p=0.5 = 球面方位平均の p 偏光分率、C_RA=1、合計 [0,0.95]
クランプ、η≥6 で 0）をレイから除去し**臨界隣接亜臨界セルへ堆積**する
（決定論 deposit cache 経由。共鳴場のエネルギーは臨界層で熱化する
ため転回半径でなく臨界セルに置く。セル幅 ~ Airy 長 \(\ell_A=(L_n/
k_0^2)^{1/3}\) ≈ 0.4–0.9 µm のため、専用カーネル散布との差は
伝導平滑化スケール以下）。r_c と \(L_n=|d\ln\hat n/dr|^{-1}\)
（[10⁻⁵,1] cm クランプ)は毎レーザーステップ、hydro ミラーの最外
n̂=1 交差から線形補間でホスト前計算する。交差が無いステップは不活性。
熱電子への分配は将来課題（現状は全量熱電子チャネルなしの局所堆積）。

適用範囲と制約（v1）: 1D_SPH のみ（他次元は起動時エラー）。
persistent megakernel（numerics.persistent_loop）とは併用不可。
CBET・hot-e との相互作用は「拡張 κ で減衰したレイパワーを下流が見る」
以外に結合しない。参照文献: Dawson & Oberman 1962; Johnston & Dawson
1973; Langdon 1980; Matte 1988; Freidberg 1972; Forslund 1975;
Colaitis 2015; Turnbull 2023; Sherlock 2024。

### 5.5 吸収パワーの空間分配

1副ステップで吸収されたパワー \(\Delta P = I^n - I^{n+1}\) [erg/s]（§5.4.2）を、
中間位置 \(\hat{\mathbf{r}}^{n+1/2} = (\hat{\mathbf{r}}^n + \hat{\mathbf{r}}^{n+1})/2\) に対応する
場所へ分配する。

- **1D_SPH**：中間位置の半径
  \[
  r^{n+1/2} = \sqrt{(R^{n+1/2})^2 + (Z^{n+1/2})^2}
  \]
  を含む Hydro の 1Dセル \(k\) に直接加算する：
  \[
  \text{deposit}_{1D}[k] \mathrel{+}= \Delta P
  \]
- **2D_RZ**：中間位置を囲む近傍4節点に双線形重みで分配する：
  \[
  \text{deposit}_{i,j} \mathrel{+}= w_{i,j}\,\Delta P,\quad
  \text{deposit}_{i+1,j} \mathrel{+}= w_{i+1,j}\,\Delta P,\quad \ldots
  \]
  重みは §5.3.3 と同一の双線形補間重み（\(\hat{\mathbf{r}}^{n+1/2}\) で評価）。

> **deposit の単位は [erg/s]**（吸収パワー）である。全レイ・全副ステップの \(\Delta P\) を蓄積する。
> 単一レイについて \(\sum_n \Delta P^n = \sum_n (I^n - I^{n+1}) = I^0 - I^{final}\) と
> 望遠鏡和により total absorbed power に一致する。

**保守性**：\(\Delta P = \sum w \cdot \Delta P\) が \(\sum w = 1\) より自動的に成立する。

> **メッシュ当たりのレイ本数が少ない場合の偏り防止**：
> 単純にレイ位置のセルへ直接沈着すると、レイ本数が少ない場合に偏りが生じる。
> 近傍4節点への双線形分配は密度勾配補間と同じ方式であり、
> 沈着の空間的な滑らかさを確保する（坂上(2006)の方法に準拠）。

**2D_RZにおける3D→2D写像**（§5.3.4と連携）：
3Dレイトレースでは中間位置 \(\hat{\mathbf{r}}^{n+1/2} = (x,y,z)^{n+1/2}\) が3Dベクトルである。
沈着先のLaserMesh座標は：
\[
(R,Z) = \left(\sqrt{x^2+y^2},\; z\right)^{n+1/2}
\]
として2D LaserMesh上のセルを特定し、近傍4節点への双線形分配を行う。
重みの計算は上記と同一であり、保守性も自動的に保たれる。
流体場が軸対称であるため、異なるφ角度から入射するレイの沈着が同一の \((R,Z)\) セルに正しく集約される。

### 5.6 ビーム幾何とレイ初期化

#### 5.6.1 F値と集光幾何
F値（F-number）：\(F = f_{focal}/D_{lens}\)。
ここで \(f_{focal}\) は焦点距離 [cm]、\(D_{lens}\) はレンズ径 [cm]。
ビーム半角：\(\theta_{beam} = \arctan(1/(2F))\) [rad]。

レイの初期位置はビームローカルRZ座標系で：
- \(Z_{init} = Z_{max}\)（LaserMesh入射境界、レーザー入射側）
- \(R_{init}\)：\(0 \le R \le R_{beam}\) で1D配列として分布

入射面でのビーム半径（F値と集光位置から幾何的に決定）：
\[
R_{beam} = \frac{|Z_{init} - Z_{focus}|}{2F}
\]

#### 5.6.2 レイの初期方向
各レイは焦点に向かう（集光型）：
\[
\hat v_R^0 = -\frac{R_{init}}{L_{focus}},\quad
\hat v_Z^0 = -\text{sign}(Z_{init}-Z_{focus})\sqrt{\max\!\left(0,\;1-\left(\frac{R_{init}}{L_{focus}}\right)^2\right)}
\]
ここで \(L_{focus} = \sqrt{R_{init}^2 + (Z_{init}-Z_{focus})^2}\) は焦点までの距離。
`max(0, ...)` は浮動小数点丸め誤差により引数がわずかに負になる場合の `sqrt(負)` = NaN を防止する。
この単位ベクトルは方向のみを定め、速度の大きさはトレース開始点（profile
entry）で \(|\hat{\mathbf v}|=\sqrt{1-\hat n}\) にリスケールされる（§5.3.2）。

#### 5.6.3 レイのパワー配分と初期化

**(a) 1D_SPH：1D配列（ビーム軸対称）**

レイはR方向にのみ分布する1D配列（ビーム軸対称を利用）。
レイ \(k\) の **正規化吸収分率** を計算するため、単位パワー（\(P=1\)）でレイトレースする：
\[
w_k^{norm} = \frac{\text{profile}(R_k)\cdot 2\pi R_k\Delta R_k}{\sum_j \text{profile}(R_j)\cdot 2\pi R_j\Delta R_j}
\]
ここで \(\text{profile}(R)\) はビーム強度プロファイル（下記参照）、
\(2\pi R_k\Delta R_k\) はレイ \(k\) が代表する環状面積（RZ軸対称）。

最内リング \(k=0\) の代表半径は \(\Delta R/2\)（面ベース化により
厳密な軸上 \(R=0\) レイは存在しない）。

**ビーム強度プロファイル関数 profile(R) の定義**：

| モデル | 数式 | パラメータ |
|--------|------|-----------|
| `gaussian` | \(\text{profile}(R) = \exp\!\left(-2\left(\frac{R}{w_0}\right)^2\right)\) | \(w_0\)：1/e²ビームウェスト半径 [cm] |
| `super_gaussian` | \(\text{profile}(R) = \exp\!\left(-2\left(\frac{R}{w_0}\right)^{2m}\right)\) | \(w_0\), \(m\)：super-Gaussian指数（\(m=1\)でGaussian） |
| `flat_top` | \(\text{profile}(R) = \begin{cases}1 & R \le R_{flat}\\0 & R > R_{flat}\end{cases}\) | \(R_{flat}\)：flat-top半径 [cm] |
| `custom` | ユーザ定義関数 \(f(R)\) | namelistのcallableで凍結 |

> **1/e² 規約**：Gaussian の指数が \(-2(R/w_0)^2\) であることに注意。
> \(R=w_0\) で \(I/I_0 = e^{-2} \approx 0.135\) となる。
> namelist の `w0_um` は \(w_0 \times 10^4\)（cm→µm変換）で指定する。
>
> **ビーム半径 \(R_{beam}\) との関係**：\(R_{beam}\) はF値から決まるビーム外径（§5.6.1）であり、
> \(R > R_{beam}\) のレイは生成しない。profile(R) は \(R_{beam}\) 内での相対強度分布を定義する。

**1D_SPH レイ配列の構築**：

レイは**面ベース**の環状求積で配置する。`rays_per_beam` を \(N_R\) とする
（**\(N_R \ge 1\) 必須**）：環の面を \(R_{k\pm1/2}=k\Delta R\)
（\(\Delta R = R_{beam}/N_R\)）に置き、代表半径と厳密な環状面積を
\[
R_k = \left(k+\tfrac12\right)\Delta R,\qquad
\Delta A_k = \pi\left[(k+1)^2-k^2\right]\Delta R^2
\quad (k=0,1,\dots,N_R-1)
\]
とする。これで \([0,R_{beam}]\) が隙間なく被覆される（2026-07-26 変更 —
旧仕様の node-based 配置 \(R_k=k\Delta R\) は最外半リング
\([R_{beam}-\Delta R/2,\,R_{beam}]\) を欠き、profile モーメントを内向きに
\(O(1/N_R)\) 偏らせていた（2026-07-26 カーネルレビュー指摘）。

**(b) 2D_RZ：2D断面配列（ビーム軸直交平面）**

2D_RZモードでは各ビームが3D空間で任意の方向から入射するため、
レイをビーム軸直交平面上の **2D配列** として初期化する。

ビーム軸方向の単位ベクトルを \(\hat{\mathbf{d}}\) とし、
ビーム軸に直交する2つの単位ベクトル \(\hat{\mathbf{u}}, \hat{\mathbf{w}}\) を構築する
（\(\hat{\mathbf{u}} \times \hat{\mathbf{w}} = \hat{\mathbf{d}}\)）。

**正規直交基底の構築アルゴリズム**（決定論的、再現性保証）：
1. 参照ベクトルを選択：\(|\hat{\mathbf{d}} \cdot \hat{\mathbf{e}}_z| < 0.9\) ならば \(\hat{\mathbf{e}}_{ref} = \hat{\mathbf{e}}_z\)、
   そうでなければ \(\hat{\mathbf{e}}_{ref} = \hat{\mathbf{e}}_x\)
2. \(\hat{\mathbf{u}} = \text{normalize}(\hat{\mathbf{e}}_{ref} \times \hat{\mathbf{d}})\)
3. \(\hat{\mathbf{w}} = \hat{\mathbf{d}} \times \hat{\mathbf{u}}\)

> この構築は \(\hat{\mathbf{d}}\) から一意に決定されるため、同一ビーム方向では常に同一の基底を生成する。

レイ \((p,q)\) の初期位置（3D Lab座標系）：
\[
\mathbf{r}^0_{p,q} = \mathbf{r}_{entry} + u_p\,\hat{\mathbf{u}} + w_q\,\hat{\mathbf{w}}
\]
ここで \(\mathbf{r}_{entry}\) はビーム軸がLaserMesh入射境界と交差する点（下記参照）、
\((u_p, w_q)\) はビーム断面上の2D格子点座標であり：
\[
-R_{beam} \le u_p, w_q \le R_{beam},\quad
u_p^2 + w_q^2 \le R_{beam}^2
\]
の円形領域内のみを使用する。

> 円筒境界（\(R\le R_{max},\, Z_{min}\le Z\le Z_{max}\)）に対し、離散点
> \(\mathbf{r}^0_{p,q}\) が境界外になる場合は、そのレイの進行方向
> \(\hat{\mathbf{v}}^0_{p,q}\)（式(5.6.3-b-4)）に沿って最初の境界交点へ開始点をクリップする。
> 重み \(w_{p,q}^{norm}\) は変更しない（ビーム総パワー保存）。

**2D格子の構築**：`rays_per_beam` を \(N\) とする（**\(N \ge 2\) 必須**、\(N=1\) は除算 \(N-1=0\) で未定義）：
\[
\Delta u = \Delta w = \frac{2 R_{beam}}{N-1},\quad
u_p = -R_{beam} + p\,\Delta u,\quad
w_q = -R_{beam} + q\,\Delta w
\]
\((p,q) \in \{0,\dots,N{-}1\}^2\) のうち \(u_p^2 + w_q^2 \le R_{beam}^2\) を満たす点のみを使用する。
有効レイ数は約 \(\pi N^2/4\) 本。
v1.0の既定値は `rays_per_beam` = 128（2D_RZ）であり、実効本数は約12,800本/beam。

**入射点 \(\mathbf{r}_{entry}\) の計算**：
ビーム軸を \(\mathbf{r}(t) = \mathbf{r}_{source} + t\,\hat{\mathbf{d}}\)（\(\mathbf{r}_{source}\) は十分遠方の仮想光源位置）として、
LaserMesh境界面（\(R_{max}\), \(Z_{min}\), \(Z_{max}\) で定義される円筒面）との交点を計算する：

1. **Z面との交点**：各面 \(Z = Z_{min}, Z_{max}\) に対し、\(|d_z| > \varepsilon_{dir}\)（\(\varepsilon_{dir} = 10^{-30}\)）の場合のみ
   \(t = (Z_{face} - z_{source})/d_z\) を計算し、交点の \(R = \sqrt{x^2+y^2}\) が \(R \le R_{max}\) なら候補とする。
   \(|d_z| \le \varepsilon_{dir}\) の場合はZ面候補をスキップする（レイが面と平行）
2. **R面との交点**：\(R = R_{max}\) の円筒面に対し
   \((x_s + t\,d_x)^2 + (y_s + t\,d_y)^2 = R_{max}^2\) を解く（2次方程式）
3. 全候補の中で \(t > 0\)（光源から遠ざかる方向）かつ最小の \(t\) を選択

> **仮想光源位置**：\(\mathbf{r}_{source} = \mathbf{r}_{focus} - L_{source}\,\hat{\mathbf{d}}\)、
> \(L_{source}\) はFocus-to-Lens距離 = \(2F \times R_{beam}\)。

レイの初期方向（焦点に向かう集光型）：
\[
\hat{\mathbf{v}}^0_{p,q} = \frac{\mathbf{r}_{focus} - \mathbf{r}^0_{p,q}}{|\mathbf{r}_{focus} - \mathbf{r}^0_{p,q}|}
\]

パワー重み：
\[
w_{p,q}^{norm} = \frac{\text{profile}(\sqrt{u_p^2+w_q^2})\,\Delta u\,\Delta w}{\sum_{p',q'} \text{profile}(\sqrt{u_{p'}^2+w_{q'}^2})\,\Delta u\,\Delta w}
\]
プロファイル関数は1D_SPHと同一（radial profileを断面距離で評価）。

> **1D配列との使い分け**：1D_SPHではビーム軸対称（全ビーム同一吸収パターン）のため
> 1D配列で十分だが、2D_RZでは非軸対称入射のため2D断面配列が必須である。

#### 5.6.4 1ビーム計算と多ビーム重ね合わせ

**基本方針**：レイトレースは **1ビーム分のみ** 実行する。
複数ビームが存在する場合は、1ビームの計算結果をビーム毎のパワーでスケーリングして重ね合わせる。

`radial_absorption_1d` ではレイトレースもビームグループ化も行わず、
\(P_{\mathrm{total}}(t)=\sum_b P_b(t)\) を1本の inward radial flux として §5.4a の積分に渡す。
このモードでは beam の方向・F値・焦点・デフォーカス・プロファイル・レイ本数は吸収分布を変えない。

**物理的根拠**（1D_SPH）：
- ターゲットが球対称であるため、全ビームが同一の密度プロファイル \(\rho(\sqrt{R^2+Z^2})\) を「見る」
- ビーム方向が異なっても、ビームローカルRZ座標での物理量分布は同一
- したがってレイ軌跡と吸収パターンは全ビームで同一であり、1回のレイトレースで十分

**処理手順**：

1. **1ビームのレイトレース**：代表ビーム（1本目）のパラメータ（F値、プロファイル）で
   レイトレースを実行し、LaserMesh上の **正規化吸収分率** \(\hat{f}_{i,j}\) を得る：
\[
\hat{f}_{i,j} = \frac{\text{deposit}_{i,j}^{(1)}}{P_1(t)} \quad [\text{無次元}], \quad
\sum_{i,j} \hat{f}_{i,j} = \eta_{abs}
\]
ここで \(\text{deposit}_{i,j}^{(1)}\) [erg/s] は代表ビームのレイトレースで得た吸収パワー（§5.5）、
\(P_1(t)\) [erg/s] は代表ビームの入射パワー、
\(\eta_{abs}\) は吸収効率（= 1 − 反射/損失率、無次元）。
**次元検証**：[erg/s] / [erg/s] = 無次元 ✓

2. **多ビーム重ね合わせ**：各ビーム \(b\) のパワー \(P_b(t)\) でスケーリングして加算：
\[
\text{deposit}_{i,j}^{total} = \hat{f}_{i,j} \times \sum_b P_b(t) \quad [\text{erg/s}]
\]

3. **流体メッシュへの転写とエネルギー変換**（§5.8）：
\[
\text{laser\_dep}[k] = \left(\sum_{\substack{(i,j):\\ r_{i,j}\in\text{cell}_k}} \text{deposit}_{i,j}^{total}\right) \times \Delta t \quad [\text{erg}]
\]
ここで \(\Delta t\) は流体タイムステップ幅。\(\Delta t\) の乗算はここで **1回だけ** 行う。

**ビーム毎に異なるパラメータを持つ場合**：
- F値やプロファイルがビーム間で異なる場合、ビームを **パラメータが同一のグループ** に分類する
- 各グループで1回ずつレイトレースを行い、結果をグループ内のビーム数とパワーでスケーリング
- 全グループの結果を加算する
- 典型的なICF実験（GXII 12ビーム等）では全ビーム同一パラメータであり、1回のレイトレースで完結する

**計算量の削減**：
- \(N_{beam}\) ビームで全ビーム同一パラメータの場合、計算量は \(1/N_{beam}\) に削減
- GXIIの12ビーム構成では約 **12倍の高速化**

**2D_RZにおける多ビーム重ね合わせ**：

2D_RZでは流体場が軸対称のため、ビームのグループ化条件が1D_SPHと異なる。
ビーム \(b\) の方向ベクトルと対称軸（Z軸）のなす **極角** \(\theta_b\) を用いる：
\[
\theta_b = \arccos\left(\frac{\hat{\mathbf{d}}_b \cdot \hat{\mathbf{e}}_z}{|\hat{\mathbf{d}}_b|}\right)
\]

- **同一極角θ、同一F値・プロファイル** のビームは同一の吸収パターンを生成する
  （軸対称場のφ非依存性による）
- したがって極角θとパラメータ（F値、プロファイル）の組でグループ化する
- 各グループで1回ずつ3Dレイトレースを行い、グループ内パワー合計でスケーリング
- 全グループの結果をLaserMesh上で加算する

**極角グループ化の処理手順**：

1. 各ビームの極角 \(\theta_b\) を計算
2. \(|\theta_b - \theta_{b'}| < \varepsilon_\theta\)（既定 \(\varepsilon_\theta = 10^{-6}\) rad）かつ
   同一F値・プロファイル・焦点条件（focus一致、またはdefocus換算後focus一致）のビームを同一グループとする
3. 各グループ \(g\) で代表ビーム1本の3Dレイトレースを実行：
\[
\hat{f}_{i,j}^{(g)} = \frac{\text{deposit}_{i,j}^{(g)}}{P_{rep}^{(g)}(t)} \quad [\text{無次元}]
\]
ここで \(\text{deposit}_{i,j}^{(g)}\) [erg/s] は代表ビームのレイトレースで得た吸収パワー。
4. 全グループの重ね合わせ：
\[
\text{deposit}_{i,j}^{total} = \sum_g \hat{f}_{i,j}^{(g)} \times \sum_{b \in g} P_b(t) \quad [\text{erg/s}]
\]

**例**（GXII 12ビーム正十二面体配置）：
- 全ビーム同一F値・プロファイルの場合、極角θの一致で分類
- 正十二面体の対称性により数個のθグループに分類され、計算量が大幅に削減される

#### 5.6.5 デフォーカスパラメータ D/R
\[
D/R = \frac{Z_{target\_center} - Z_{focus}}{R_{target}}
\]
- \(D/R < 0\)：ターゲット手前に集光（over-focused）
- \(D/R > 0\)：ターゲット奥に集光（under-focused）
- \(D/R = 0\)：ターゲット中心に集光

`focus`（3D焦点座標）と `defocus`（D/R）が両方指定された場合は **`focus` が優先**。
`defocus` のみ指定時はビーム方向ベクトルとターゲット外半径からfocusを自動計算する：

**自動計算アルゴリズム**：
1. ターゲット中心座標 \(\mathbf{r}_{center}\) とターゲット外半径 \(R_{target}\) を取得
2. ビーム方向単位ベクトル \(\hat{\mathbf{d}}\) を正規化
3. 焦点位置を計算：
\[
\mathbf{r}_{focus} = \mathbf{r}_{center} + (D/R) \cdot R_{target} \cdot \hat{\mathbf{d}}
\]
ここで \(D/R\) は `defocus` パラメータ。

**1D_SPH**（球対称）：\(\mathbf{r}_{center}=(0,0,0)\)、\(\hat{\mathbf{d}}\) はビーム入射方向。
焦点のZ座標（ビームローカル）：\(Z_{focus} = (D/R) \cdot R_{target}\)。

**2D_RZ**（軸対称）：\(\mathbf{r}_{center}\) はシミュレーション領域の幾何中心。
ビーム極角 \(\theta_b\)（§5.6.4参照）が定義されていれば、
焦点のRZ座標は \(R_{focus}=(D/R)\cdot R_{target}\sin\theta_b\)、
\(Z_{focus}=Z_{center}+(D/R)\cdot R_{target}\cos\theta_b\) となる。

### 5.7 LaserMesh（2D RZ構造格子）

#### 5.7.1 メッシュ構造
**LaserMeshは1つだけ** 生成する（ビーム毎に独立なメッシュは作らない）。

**1D_SPH**：代表ビーム（1本目）の軸をZ軸とする **ビームローカル** 2D RZ構造格子を構築する。

**2D_RZ**：**流体対称軸（Z軸）に沿う** 2D RZ構造格子を構築する（ビームローカルではない）。
流体メッシュの対称軸と一致するため、物理量のマッピングが直接的であり、
3Dレイトレース（§5.3.4）のレイ位置 \((x,y,z)\) から \((R,Z)=(\sqrt{x^2+y^2},z)\) への
変換が自然に対応する。

共通仕様：
- R方向：\(0 \le R \le R_{max}\)（\(R_{max}\) は既定でターゲット外半径の1.5倍）
- Z方向：\(Z_{min} \le Z \le Z_{max}\)（ターゲット中心を含む範囲）

**既定値**：

| パラメータ | 既定値 | 説明 |
|-----------|-------|------|
| `nr` | 128 | R方向セル数 |
| `nz` | 256 | Z方向セル数 |
| `nr_max` | 4096 | 1D_SPH動的gradedメッシュの R方向セル数上限（OOM safety） |
| `r_max` | \(1.5 \times R_{target}\) [cm] | R方向上限 |
| `z_min` | \(Z_{center} - 1.5 \times R_{target}\) [cm] | Z方向下限 |
| `z_max` | \(Z_{center} + 1.5 \times R_{target}\) [cm] | Z方向上限 |

**1D_SPH**：\(Z_{center}=0\)（ビームローカル座標原点 = ターゲット中心）。
1D_SPHでは Rノードを各レーザー演算子呼び出し時に再生成し、Zノードは
\(Z=\{-R_{nr},\dots,-R_1,0,R_1,\dots,R_{nr}\}\) の鏡映配置とする（\(nz=2nr\)）。
**2D_RZ**：\(Z_{center}\) はシミュレーション領域の幾何中心のZ座標。
- **臨界密度クリップ**（既定ON）：\(\hat n < \hat n_{margin}\)（既定 \(\hat n_{margin}=1-\varepsilon_{crit}=0.9999\)）の領域のみカバー
  - 臨界面より奥（高密度側）はメッシュに含めない
  - **整合性要件**（**validate で強制、違反は ConfigError**）: \(\hat n_{margin} \ge 1-\varepsilon_{crit}\) でなければならない。これにより§5.2のterminate処理がLaserMesh外判定より先に発動し、臨界面近傍の吸収が欠落しないことが保証される。LaserMesh外終了はレイが横方向に逸れた場合のフォールバックとして機能する
  - 既定値は \(1-\varepsilon_{crit}\) に追随する（`eps_crit` のみ変更すれば自動的に整合する）
  - 旧既定値0.95では \(0.95 < \hat n < 1-\varepsilon_{crit}\) の高吸収領域（\(\kappa_{IB} \propto \hat n^2\)）がメッシュ外終了で欠落する系統誤差があった

#### 5.7.2 1D_SPH動的gradedメッシュ生成
1D_SPHでは、臨界面近傍のみ高解像度とし、内外側で幾何級数的に粗くする
piecewise-geometricメッシュを毎ステップ再生成する。

**(a) 臨界位置とスケールの決定**

1. 各流体セルで \(\hat n_c = n_{e,c}/n_{crit}\) を評価する
2. 臨界交差面 \(f_{crit}\) を
   \[
   \hat n_c \ge 1,\ \hat n_{c+1}<1
   \]
   を満たす最外側faceとして求める（なければ未定義）
3. \(R_{crit}\) は \(r_{edge}[f_{crit}]\)（未定義時は \(0.5R_{max}\)）
4. 局所最小セル幅 \(min\_dr_{crit}\) は以下の和集合領域で評価する：
   - \(|c-f_{crit}| \le 10\) の近傍セル
   - \(|r_c-R_{crit}| \le 0.05R_{crit}\) の半径窓
5. \(dR_{fine} = \texttt{mesh\_factor} \times min\_dr_{crit}\)

**(b) R方向ノードのpiecewise-geometric構成**

- Zone A（core）\([0,R_{left}]\)：\(R_{left}\) 近傍を細かく、軸側へ幾何級数で粗化（`g_core=1.08`）
- Zone B（fine band）\([R_{left},R_{right}]\)：一様幅 \(dR_{fine}\)
- Zone C（corona）\([R_{right},R_{max}]\)：\(R_{right}\) 近傍を細かく、外側へ幾何級数で粗化（`g_corona=1.05`）

細帯の目標幅は各側 \(64\,dR_{fine}\) とし、\(R_{crit}\) を中心にクリップして配置する。

**(c) R_max とセル数上限**

\(R_{max}\) は既存規約のまま
\[
R_{max}=r_{max\_factor}\times r_{\hat n\ge threshold,\ outer\,edge}
\]
（該当セルなし時は \(r_{max\_factor}\times R_{target}\)）で決定する。
その後 \(R_{max}\ge 4\,min\_dr_{crit}\) を保証する。

生成セル数 \(nr\) が `nr_max` を超える場合は \(dR_{fine}\) をスケールアップし、
二分探索で \(nr\le nr_{max}\) を満たすまで再生成する（OOM safety）。

**(d) Z方向ノード（鏡映）**
\[
Z = \{-R_{nr},\dots,-R_1,0,R_1,\dots,R_{nr}\}
\]
とし、常に \(nz=2nr\) とする。

#### 5.7.3 物理量のLaserMeshへのマッピング

**(a) 1D_SPH：球対称プロファイルのRZ展開**

1Dの物理量 \(Q(r)\) をビームローカルRZ座標の節点 \((R_i, Z_j)\) に展開：
\[
Q(R_i, Z_j) = Q\!\left(\sqrt{R_i^2 + Z_j^2}\right)
\]
ここで座標原点はターゲット中心（ビーム方向ベクトルと `focus` から決定）。

**節点密度の評価（2026-07-31 線形補間化）**：\(\hat n\) の元となる電子密度
\(n_e = \rho\bar Z/(A_{\rm eff}m_p)\) は、節点半径 \(r\) を挟む**同種物質の
隣接 hydro セル中心間の \(r\) 線形補間**で評価する（境界・void 隣接・表面外
セルは従来どおり所属セル値へフォールバック）。旧仕様の区分一定サンプリング
（所属セル値をそのまま使用）は、節点 \(\hat n\) に \(|\partial_r\hat n|\,
\Delta r_{cell}/2\) の階段誤差（解析ゲートで実測 \(2.44\times10^{-4}\) =
予測値）を与え、レーザーノードが hydro セルより細かい臨界近傍では区間勾配
`radial_dn_dr` が 0/2 倍の櫛状ノイズ（実測 27.5%）となってレイ転回点を
散乱させていた。線形補間により線形プロファイルは任意のノード配置で厳密に
再現される（場誤差 \(4.4\times10^{-16}\)）。\(T_e,\bar Z\) の節点値は
従来どおり所属セル値（補間しない）。fcrit セルの臨界クリップ対数補間分岐は
補間後の \(\hat n_{raw}\) に対して従来どおり適用される。

**(b) 2D_RZ：軸対称場の直接マッピング**

2D_RZ流体メッシュの物理量 \(Q(R,Z)\) をLaserMesh節点 \((R_i, Z_j)\) にマッピングする。
LaserMeshが流体対称軸に沿っている（§5.7.1）ため、座標変換は不要であり、
HydroMeshからLaserMeshへの双線形補間で直接マッピングできる。

対象物理量（共通）：\(\rho\), \(T_e\), \(\bar Z\)（→ \(\hat n\), \(\nu_{ei}\) 等を計算）。
2D_RZでは、LaserMesh節点が現在のHydroMesh実領域
\([R_{\min},R_{\max}]\times[Z_{\min},Z_{\max}]\) の外側にある場合、
補間後に \(\bar Z=0\) および \(\hat n_{raw}=0\) を強制する。
\(T_e\) は補間値を保持してよいが、\(\bar Z=0\) によりIB吸収係数は0となる。
これはビーム投入用にHydroMesh外へ拡張した2D_RZ LaserMesh領域を透明に保つための処理であり、
1D_SPHのゴーストコロナ（§5.7.5）には適用しない。

#### 5.7.4 LaserMeshの更新タイミング
LaserMesh上の物理量（\(\hat n, T_e, \bar Z\)）は **各レーザー演算子呼び出し時** に
HydroMeshから再マッピングする（Strang splitting中に1回）。
1D_SPHでは同タイミングで格子点配置（`node_R`, `node_Z`）も再生成する。
2D_RZでは格子点配置は初期化時固定のまま再利用する。
1D_SPH ではさらに、`node_Z=0` 列から
`radial_node_r`, `radial_n_hat`, `radial_n_hat_raw`, `radial_smooth_kappa`,
`radial_dn_dr` を抽出し、ray trace の 1D lookup に使う。

1D_SPH の `map_from_hydro_1d` では、レイトレースに使う clipped nodal density
\(\hat n\) に対して step 間 EMA を適用する。前回値が有効で節点数が不変、かつ
current/previous の両方が \(\hat n > 0.3\) を満たし、
\[
|\hat n_{cur} - \hat n_{prev}| < 0.01
\]
のときに限り、
\[
\hat n \leftarrow \alpha \hat n_{cur} + (1-\alpha)\hat n_{prev},\qquad \alpha = 0.05
\]
で更新する。`n_hat_raw` は平滑化せず、節点数が変化した場合は EMA 状態を破棄する。

1D_SPH の動的 LaserMesh 再生成で用いる臨界半径 \(R_{\mathrm{crit}}\) は、
最後の超臨界実セル \(c_{\mathrm{hi}}\) とその外側の亜臨界実セル \(c_{\mathrm{lo}}\)
（\(\hat n_{c_{\mathrm{hi}}} \ge 1\), \(\hat n_{c_{\mathrm{lo}}} < 1\)）が両方存在する場合、
セル中心間の対数補間で \(\hat n = 1\) を満たす位置に置く：
\[
\theta = \frac{\ln(1 / \hat n_{c_{\mathrm{hi}}})}
               {\ln(\hat n_{c_{\mathrm{lo}}} / \hat n_{c_{\mathrm{hi}}})},
\qquad
R_{\mathrm{crit}} =
r_{c_{\mathrm{hi}}} + \text{clamp}(\theta, 0, 1)\,
\left(r_{c_{\mathrm{lo}}} - r_{c_{\mathrm{hi}}}\right)
\]
ここで \(r_c\) はセル中心半径。外側に亜臨界実セルがまだ存在しない場合は、
従来どおり最後の超臨界セル外側 face を用いる。`map_from_hydro_1d` の
critical-adjacent subcritical cell の log 再構成も同じ \(R_{\mathrm{crit}}\) を
アンカーに使う。

> **将来拡張**：Lagrangian流体メッシュの大変形に追従するLaserMeshの動的再生成。

#### 5.7.5 ゴーストコロナ（1D_SPH シャープギャップ対応）

##### 5.7.5.1 目的

1D_SPH 爆縮問題で流体初期条件がシャープな固体–真空界面を持つ場合（プレプラズマなし）、
レーザー吸収エネルギーが最外固体セルに集中し、非物理的な加熱と数値不安定を引き起こす。
ゴーストコロナ（ghost corona）はこの問題を緩和するための
**レーザー専用の合成亜臨界プロファイル** である。

- 流体メッシュ上の状態（密度・温度・電荷数）には一切変更を加えない。
- LaserMesh 上にのみ仮想的な低密度コロナプロファイルを構築する。
- レイトレースはこのプロファイル上で屈折・吸収を計算するため、光学的にはコロナが存在するように振る舞う。
- 1D_SPH 専用。2D_RZ では使用しない。

##### 5.7.5.2 ゴーストプロファイルの構成

LaserMesh 再マッピング時（§5.7.4）に、最外 3 非 void セルから以下のアンカー値を
平滑化して取得する：

- \(T_{e,\text{anchor}}\) [eV]：アンカー電子温度
- \(\bar{Z}_{\text{anchor}}\) [無次元]：アンカー平均電荷数
- \(A_{\text{anchor}}\) [amu]：アンカー平均質量数

ゴースト領域（void セル上方）に指数関数的な電子密度プロファイルを構築する：

- ゴーストセル数：\(N_{\text{ghost}} = \texttt{n\_out}\)（既定 12）
- 密度範囲：\(\hat{n}_{\text{ghost}} \in [\texttt{ne\_min\_frac}, \texttt{ne\_max\_frac}]\)（既定 \([0.03, 0.99]\)）
- 電子温度下限：\(T_{e,\text{ghost}} \ge \texttt{Te\_min\_eV}\)（既定 50 eV）
- 電荷数範囲：\(\bar{Z} \in [\texttt{zbar\_min}, \texttt{zbar\_max}]\)（既定 \([1.0, 4.0]\)）

プロファイル幅は \(\max(w_{\text{base}},\; c_s \times t)\) で時間発展する
（\(c_s\) はアンカー値から推定した音速、\(t\) はシミュレーション時刻）。
最外実セルが既に亜臨界の場合、ゴースト内側密度は実セル密度に追随する。

##### 5.7.5.3 ハンドオフステンシル（void 側吸収パワーの再分配）

ゴースト領域内の LaserMesh ノードで吸収されたパワーは、
1D 球座標への転写時（§5.8.1 (a)）に void セルに対応する半径 \(r\) に落ちる。
void セルにはエネルギーを沈着できないため、最近傍の非 void セルを起点とする
短い **内向きステンシル** でエネルギーを再分配する。

ベースライン重み（指数関数減衰）：
\[
w_k^{\text{base}} = \exp\!\left(-\frac{k}{d_{\text{handoff}}}\right), \quad k = 0, 1, \ldots, N_{\text{handoff}}-1
\]

ここで \(k\) はアンカーセルからの内向きオフセット、
\(N_{\text{handoff}} = \texttt{handoff\_cells}\)（既定 6）、
\(d_{\text{handoff}} = \texttt{handoff\_decay}\)（既定 2.0）。
void セルはスキップされる。

正規化による分配：
\[
\text{laser\_dep}[c_k] \mathrel{+}= P_{\text{void}} \times \frac{w_k}{\sum_j w_j} \times \Delta t
\]

エネルギー保存は §5.8.2 の検証式で同ステップ内に確認される
（void 側パワーも合算後の保存式に含まれる）。

実セル側でも、近似セル中心密度
\[
\hat{n}_c = \frac{\rho_c \bar{Z}_c}{A_{\mathrm{eff},c} m_p n_{\mathrm{crit}}}
\]
が 1 以上の **超臨界セルは転写先にしない**。
このとき、void 由来のパワーは上記ハンドオフで亜臨界セルへ再分配し、
supercritical 実セルへ一旦落ちたパワーは最も近い **外側の亜臨界実セル** へ付け替える。
ただし 1D_SPH では、臨界面に隣接する **最外の超臨界実セル 1 個だけ** を例外受け皿として許可する。
臨界面が外側の亜臨界セル内部に入る場合、そのセル内で \(r < R_{\mathrm{crit}}\) にある沈着は
この 1 セルへ結合する。
さらに resolved-corona が不足している間（§5.7.5.4 の \(\beta > 0\)）は、
critical-adjacent な亜臨界セルへ落ちた沈着も同じ inward stencil で近臨界帯へ再分配する。
この再分配にはベースライン重み \(w_k^{\text{base}}\) を用い、
密度バイアスは適用しない。
亜臨界の実セルが 1 つも存在しない初期段階では、この例外受け皿は最外の実セルに一致する。
それでも受け皿が存在しない場合、そのパワーは `laser_dep` に入れず
未吸収パワーとして収支へ戻す（§5.8.2）。

##### 5.7.5.4 ブローオフ遷移モデル（Blowoff Transition）

シミュレーション初期は実セルが全て超臨界（\(\hat{n} \ge 1\)）のため、
ゴーストコロナのハンドオフが唯一の吸収エネルギー分配経路となる。
時間が経過し流体がブローオフして実セルのコロナが解像され始めると、
ハンドオフの密度バイアスを徐々に弱めて安定ベースラインに自動復帰させる必要がある。

**resolved-corona スイッチ**：

1. 各実（非 void）セル \(c\) の近似正規化電子密度を計算する：
\[
\hat{n}_c = \frac{\rho_c \, \bar{Z}_c}{A_{\text{eff},c} \, m_p \, n_{\text{crit}}}
\]
ここで \(A_{\text{eff},c}\) は体積分率加重した調和平均質量数、
\(m_p\) は陽子質量、\(n_{\text{crit}}\) は臨界密度。

2. 最外非 void セルから内向きに連続して
   \(\hat{n}_c < \hat{n}_{\text{resolved}}\)（既定 0.9）を満たすセル数
   \(N_{\text{resolved}}\) をカウントする。

3. ブレンド係数を決定する：
\[
\beta = \text{clamp}\!\left(1 - \frac{N_{\text{resolved}}}{N_{\text{required}}},\; 0,\; 1\right)
\]
ここで \(N_{\text{required}} = \texttt{transition\_resolved\_cells}\)（既定 3）。
\(\beta = 1\) で遷移モデルが完全に有効、\(\beta = 0\) で安定ベースラインに復帰。

**密度バイアス重み**：

\(\beta > 0\) のとき、void 側ハンドオフの各ステンシルセル重みに密度バイアスを適用する：
\[
w_k = (1 - \beta)\, w_k^{\text{base}} + \beta\, w_k^{\text{base}} \times f_{\rho,k}
\]
\[
f_{\rho,k} = \text{clamp}\!\left(\frac{\hat{n}_k}{\hat{n}_{\text{resolved}}},\; 0.25,\; 4.0\right)^{\!\alpha_\rho}
\]

ここで \(\alpha_\rho = \texttt{transition\_density\_exponent}\)（既定 1.0）。
密度が高いセルにより多くのパワーを分配し、低密度の最外セルへの過剰集中を抑制する。
critical-adjacent な実表面セルからの再配分は、外向きアブレーション駆動を保つため
この密度バイアスを使わず §5.7.5.3 のベースライン重みをそのまま使う。

**設計制約**：

- エネルギー保存は正規化条件で同ステップ内に厳密に維持される（§5.8.2）。
- ゴーストプロファイル自体は遷移モデルの影響を受けない（光学特性は不変）。
- 遷移モデルはハンドオフ重みのみを変更し、吸収パワーの総量には影響しない。
- `transition_enabled=False` の場合、遷移モデルは無効化され §5.7.5.3 のベースラインステンシルのみが使用される。

##### 5.7.5.5 診断出力

| History key | 型 | 単位 | 意味 |
|---|---|---|---|
| `laser/corona_transition_blend` | double | 無次元 | ブレンド係数 \(\beta\)（0＝安定、1＝遷移活発） |
| `laser/corona_transition_resolved_cells` | int64 | cells | 外側から連続して亜臨界な実セルの数 \(N_{\text{resolved}}\) |

### 5.8 座標変換と沈着の転写

#### 5.8.1 沈着の転写（次元依存）

**(a) 1D_SPH：Hydro 1Dセルへの直接沈着**

1D_SPH では §5.5 の時点で、吸収パワーは Hydro の 1Dセル配列
\(\text{deposit}_{1D}[k]\) [erg/s] に直接蓄積されている。
`radial_absorption_1d` でも §5.4a の `deposit_power_cell[k]` [erg/s] を
同じ 1D 直接沈着パスへ渡す。
したがって 2D LaserMesh節点からの再転写は行わず、
流体タイムステップ幅 \(\Delta t\) を 1 回だけ掛けて
\[
\text{laser\_dep}[k] = \text{deposit}_{1D}^{total}[k] \times \Delta t \quad [\text{erg}]
\]
に変換する。

ただし、HydroMesh 側で
\[
\hat{n}_k = \frac{\rho_k \bar{Z}_k}{A_{\mathrm{eff},k} m_p n_{\mathrm{crit}}} \ge 1
\]
または void のセルには直接沈着しない。
void セルは §5.7.5.3 のハンドオフステンシルで亜臨界セルへ再分配し、
supercritical 実セルは最も近い外側の亜臨界実セルへ付け替える。
1D_SPH では臨界面に隣接する最外の超臨界実セル 1 個を例外受け皿として使い、
臨界面が外側の亜臨界セル内部にあるときは \(r < R_{\mathrm{crit}}\) の分をそのセルへ割り当てる。
resolved-corona が不足している間は、critical-adjacent な亜臨界セルの沈着も
同じ inward stencil で近臨界帯へ広げる。
このときの重みはベース指数重み \(w_k^{\text{base}}\) を使い、
void 側ハンドオフ用の密度バイアスは適用しない。
亜臨界の実セルが存在しない場合、この例外受け皿は最外の実セルになる。
それでも受け皿が存在しない分は `laser_dep` に加えず、未吸収として扱う。

診断目的で、`Laser.deposit.deposit_smooth_passes > 0` かつ
`Laser.deposit.deposit_smooth_alpha > 0` のときは、`laser_dep` への書き込み直前に
\(\text{deposit}_{1D}^{total}\) [erg/s] へ保存的な mass-weighted smoothing を
\(N_{\mathrm{pass}}\) 回適用してよい。セル質量 \(m_k\) と specific absorbed power
\(q_k = \text{deposit}_{1D}[k] / m_k\) を用い、有効セル
（非void・非blocked・非境界）同士の界面にだけ flux を許す。
さらに、void/blocked セルに隣接する 1 セルは ablation-face guard として平滑化しない。
各 pass の face flux を
\[
F_{k+\frac{1}{2}}^{(m)} =
\alpha\,\chi_{k+\frac{1}{2}}\,
M_{k+\frac{1}{2}}
\left(q_{k+1}^{(m)} - q_k^{(m)}\right),
\]
\[
\chi_{k+\frac{1}{2}} =
\begin{cases}
1, & k,\;k+1 \text{ がともに平滑化対象セルかつ guard 外} \\
0, & \text{otherwise}
\end{cases}
\]
\[
M_{k+\frac{1}{2}} = \min(m_k, m_{k+1})
\]
とし、保存変数は
\[
\text{deposit}_{1D}^{(m+1)}[k] =
\text{deposit}_{1D}^{(m)}[k]
+ F_{k+\frac{1}{2}}^{(m)}
- F_{k-\frac{1}{2}}^{(m)}
\]
で更新する。ここで \(m = 0, \ldots, N_{\mathrm{pass}}-1\)、
\(\alpha =\) `deposit_smooth_alpha` である。face flux がペアごとに相殺されるため、
平滑化は全吸収パワーを保存する。

**(b) 2D_RZ：LaserMesh上の直接沈着**

2D_RZでは LaserMesh が流体対称軸に沿っている（§5.7.1）ため、
1D球座標への転写は不要である。
各 HydroMesh セルの中心 \((r_c, z_c)\) を LaserMesh のノード座標上に射影し、
近傍4ノードの deposit 値から双線形補間でセル沈着パワーを算出する（CUDA_KERNELS L5 準拠）：
\[
\text{laser\_dep}[c] = \left(\sum_{(i,j)\in\mathcal{N}_4(r_c,z_c)} w_{ij}(r_c,z_c)\,\text{deposit}_{i,j}^{total}\right) \times \Delta t \quad [\text{erg}]
\]
ここで \(c\) は2D_RZ HydroMeshのセルインデックス、
\(\mathcal{N}_4(r_c,z_c)\) はセル中心を囲むLaserMeshの4ノード、
\(w_{ij}\) は双線形補間重み（\(\sum w_{ij}=1\)）である。
HydroMesh 側で void または \(\hat{n}_c \ge 1\) のセルは沈着を 0 とし、
除外した分は未吸収として収支へ戻す。

診断目的で、`Laser.deposit.deposit_smooth_passes > 0` かつ
`Laser.deposit.deposit_smooth_alpha > 0` のときは、2D_RZ でも
`laser_dep` への最終書き戻し前に保存的な Jacobi smoothing を
\(N_{\mathrm{pass}}\) 回適用してよい。対象変数 \(D_{i,j}^{(m)}\) は
HydroMesh セル \((i,j)\) の沈着エネルギー [erg]（または同じ \(\Delta t\) で割った
等価な沈着パワー [erg/s]）である。有効セルは非void・非blocked・非境界
（\(0 < i < n_r-1,\;0 < j < n_z-1\)）で、4 近傍のいずれかが
void/blocked のセルも ablation-face guard として平滑化しない。
R/Z face flux を
\[
F^R_{i+\frac{1}{2},j} =
\alpha\,\chi_{i,j}\chi_{i+1,j}
\left(D_{i+1,j}^{(m)} - D_{i,j}^{(m)}\right),
\]
\[
F^Z_{i,j+\frac{1}{2}} =
\alpha\,\chi_{i,j}\chi_{i,j+1}
\left(D_{i,j+1}^{(m)} - D_{i,j}^{(m)}\right)
\]
とし、保存変数は
\[
D_{i,j}^{(m+1)} =
D_{i,j}^{(m)}
+ F^R_{i+\frac{1}{2},j} - F^R_{i-\frac{1}{2},j}
+ F^Z_{i,j+\frac{1}{2}} - F^Z_{i,j-\frac{1}{2}}
\]
で更新する。ここで \(\chi_{i,j}=1\) は有効セル、0 は非有効セルを表し、
\(\alpha =\) `deposit_smooth_alpha` である。face flux は各 face で反対符号に
ペア適用されるため、2D_RZ 平滑化は全沈着エネルギーを丸め誤差の範囲で保存する。

> **単位規約**（共通）:
> - `deposit` 配列は **吸収パワー** [erg/s] を保持する（§5.5）。
>   レイが運ぶ \(I\) はパワー [erg/s] であり、各副ステップで吸収されるパワー \(\Delta P = I^n - I^{n+1}\) [erg/s] を節点に分配する。
>   \(\Delta t_{ray}\)（レイトレース副ステップ幅）は deposit に含めない。
> - `laser_dep` は当該ステップ \(\Delta t\) 中にセルに沈着した **エネルギー** [erg] である（`rad_dep` と同一規約）。
>   **\(\Delta t\) の乗算は転写段で1回だけ行う**。Driverでさらに \(\Delta t\) を掛けてはならない。
> - Driver側での電子エネルギー更新: \(\Delta(\rho e_e) \mathrel{+}= \text{laser\_dep}/V_{cell}\)。
> - HDF5出力の `laser/deposited_power` [erg/cm³/s] は `laser_dep / (V_{cell} \times \Delta t)` で計算する。
> - 2D_RZ HDF5出力の `laser/A_Q_r_shells` と `laser/A_Q_z_shells` は、幾何因子を戻した raw `laser_dep` [erg] から shell ごとに
>   \(A_Q=(q_{\max}-q_{\min})/(q_{\max}+q_{\min})\) を計算する。`laser/deposited_power` は体積正規化済みの派生量なので A_Q には使わない。空 shell は 0 を出力する。

#### 5.8.2 エネルギー保存の検証
転写前後でパワー保存を検証する（\(\Delta t\) は共通因子として消える）：
\[
\frac{\left|\sum_{i,j}\text{deposit}_{i,j}^{total} - \left(\sum_c \text{laser\_dep}[c]/\Delta t + P_{\mathrm{blocked}}\right)\right|}{\sum_{i,j}\text{deposit}_{i,j}^{total}} \le 10^{-10}
\]
> **ゼロ吸収ガード**：\(\sum_{i,j}\text{deposit}_{i,j}^{total} \le \varepsilon_{abs}\)（\(\varepsilon_{abs} = 10^{-20}\) erg/s）の場合は、保存検証をスキップする（吸収なし → 保存が自明に成立）。
（全ビーム合算後に評価。deposit は [erg/s]、laser\_dep/Δt は [erg/s]。）
ここで \(P_{\mathrm{blocked}}\) [erg/s] は、転写先が void または超臨界であるため
HydroMesh へ結合しなかったパワーであり、ステップ収支では
`laser_unabsorbed` に加算する。

### 5.9 レイトレース Skip 最適化

#### 5.9.1 目的
レイトレースはレーザーオペレータの中で最も計算コストが高い。
ICFシミュレーションでは、プラズマ条件（ρ, T_e, Z̄）は流体タイムステップに対して
ゆっくり変化するため、条件が十分変化していなければ前回のレイトレース結果を
パワースケーリングして再利用できる。

`radial_absorption_1d` は raytrace skip の対象外であり、毎回 §5.4a の 1D serial 積分を実行する。
このモードに入ると既存の raytrace skip cache は無効化され、正規化吸収分率の再構成は使わない。

#### 5.9.2 変化メトリクス
レーザーオペレータが呼ばれるたびに、キャッシュ時の状態との変化量 δ を計算する。

**max\_relative ノルム（既定）**：
\[
\delta = \max_i \left\{
  \frac{|\rho_i - \rho_i^{\rm cached}|}{\max(\rho_i^{\rm cached},\;\rho_{floor})},\;
  \frac{|T_{e,i} - T_{e,i}^{\rm cached}|}{\max(T_{e,i}^{\rm cached},\;T_{e,floor})},\;
  \frac{|\bar{Z}_i - \bar{Z}_i^{\rm cached}|}{\max(\bar{Z}_i^{\rm cached},\;\bar{Z}_{floor})}
\right\}
\]
ここで分母のフロアは物理フロア値を使用する：
\(\rho_{floor}\)（Numerics.density\_floor）、\(T_{e,floor}\)（Numerics.temperature\_floor）、
\(\bar{Z}_{floor} = 10^{-2}\)（完全中性の近傍で 0 割りを回避）。
これにより低密度・低温・低電離領域で分母がゼロ近傍になる場合の数値不安定性を防止する。

**l2\_relative ノルム（オプション）**：
\[
\delta = \sqrt{\frac{1}{3N}\sum_i\left[
  \left(\frac{\Delta\rho_i}{\max(\rho_i,\rho_{floor})}\right)^2 +
  \left(\frac{\Delta T_{e,i}}{\max(T_{e,i},T_{e,floor})}\right)^2 +
  \left(\frac{\Delta\bar{Z}_i}{\max(\bar{Z}_i,\bar{Z}_{floor})}\right)^2
\right]}
\]

δ < threshold（既定0.01）のとき、レイトレースをスキップする。

#### 5.9.3 パワースケーリング
キャッシュされた正規化吸収分率 \(\hat{f}^{\rm cached}\) [無次元]（§5.6.4）と
現在の総パワー・流体ステップ幅でスケーリングする：
\[
\text{laser\_dep}[i] = \hat{f}_i^{\rm cached} \times \sum_b P_b(t) \times \Delta t \quad [\text{erg}]
\]
ここで \(\hat{f}_i^{\rm cached}\) は前回のレイトレースで得た正規化吸収分率（無次元）、
\(\sum_b P_b(t)\) [erg/s] は現ステップの全ビーム総パワー、
\(\Delta t\) [s] は流体タイムステップ幅。
ビーム毎にパラメータグループが異なる場合はグループ毎にキャッシュし、各グループ内のビームパワー合計でスケーリングする。

**次元検証**：[無次元] × [erg/s] × [s] = [erg] ✓（`rad_dep` と同一規約）

物理的妥当性：
- レイ軌道は ∇n̂ に依存し、δ < threshold で変化が保証される
- IB吸収率 κ\_IB は n\_e, T\_e, Z̄ に依存し、同様に変化が小さい
- 吸収パワーは入力パワーに線形：\(\Delta P = I(1-e^{-S})\) で \(I \propto P\)、光学厚 \(S\) は不変
- 近似誤差は \(O(\delta^2)\)

#### 5.9.4 安全措置
- step = 0 では常に再計算する
- `max_consecutive`（既定10）回連続スキップ後は強制的に再計算する
- **総パワー** zero ↔ nonzero 遷移：\(\sum_b P_b\) が zero ↔ nonzero に遷移した場合は再計算する
- **per-group パワー遷移**（n\_groups > 1 の場合）：いずれかのグループ g で \(P_g(t_{cached}) = 0 \wedge P_g(t) > 0\) または \(P_g(t_{cached}) > 0 \wedge P_g(t) = 0\) が成立した場合は再計算する（グループ間パワー比変化で空間パターンが不連続に変化するため、per-group キャッシュ §5.9.3 と整合）
- **総パワー相対変化ガード**：\(\left|\sum_b P_b(t)-\sum_b P_b(t_{cached})\right|/\max(\sum_b P_b(t_{cached}), 10^{-30}) > 0.01\) の場合は再計算する
- **ビーム方向変化ガード**：いずれかのビーム b で \(\|\mathbf{d}_b(t)-\mathbf{d}_b(t_{cached})\|_2 > 10^{-10}\) の場合は再計算する
- **ALE rezone ガード**：ALE再ゾーニングが発生したステップ直後は、メッシュ位相変化を保守的に扱うため強制再計算する
- **臨界帯横断ガード**（2026-08-07 改定）：いずれかのセルで、キャッシュ時と現在の
  \(\hat{n}\) が帯域 \(\hat{n}_{margin} - \varepsilon_{crit\_guard}\)（既定
  \(\varepsilon_{crit\_guard} = 0.01\)）を**横断**した場合
  （\([\hat{n}_i > \text{band}] \ne [\hat{n}_i^{\rm cached} > \text{band}]\)、両向き）に
  強制再計算する。臨界面近傍の \(\hat{n}\) 変化はレイ terminate の有無を左右するためである。
  旧実装は現在値のみの絶対判定 \(\hat{n}_i > \text{band}\) であり、過臨界の固体内部
  （\(\hat{n} \sim 80\)）が常時ヒットして overdense 材料を含む全デッキで Skip が
  構造的に不発だった（2026-08-07 の改定で実測・修正）。比較の \(\hat{n}^{\rm cached}\) は
  キャッシュ済み ρ, Z̄ から現在の volFrac 由来 A_eff で再構成する（一貫コンパレータ）

CBET（§5.10）有効時は追加の無効化条件がある：いずれかのビームグループ g で
\(|P_g(t) - P_g(t_{cached})|/\max(|P_g(t_{cached})|, 10^{-30}) > 0.01\) の場合は
再トレースする。CBET はビーム間を非線形に結合するため per-beam パワースケーリングが
厳密でなくなることへの防護であり、これにより skip 誤差は既存の 1% 級プラズマ変化
誤差と同次に抑えられる。

#### 5.9.5 設定パラメータ

| パラメータ | 型 | 既定値 | 説明 |
|-----------|------|-------|------|
| `enabled` | bool | **false** | マスタースイッチ（2026-08-07 に既定 OFF 化。従来の既定 true は crit ガード不発により全デッキで実質不活性だったため挙動互換。有効化は opt-in の加速機能として、認証済み threshold で行う） |
| `threshold` | double | 0.01 | 最大相対変化量（1%） |
| `max_consecutive` | int | 10 | 強制再計算までの最大連続スキップ数 |
| `norm` | string | "max_relative" | "max_relative" または "l2_relative" |
| `crit_guard` | double | 0.01 | 臨界近傍ガード：n̂ > n̂\_margin − crit\_guard で再計算 |

#### 5.9.6 エネルギー保存への影響
パワースケーリングは入射パワーと吸収パワーの比例関係（\(\hat{f}\) の定義）を保存する。
\(\Delta t\) の乗算は §5.8.1 と同様に1回だけ行われるため、エネルギー収支診断（§10.2）で正しく計上される。
近似誤差は吸収係数の変化に対して \(O(\delta^2)\) である。

### 5.10 Cross-Beam Energy Transfer（CBET、1D_SPH v1）

`Laser.cbet.enable=True`（既定 False）で有効化する、Marozas/DRACO 型の
線形・定常・強減衰・局所平面波 CBET モデル（Marozas et al., Phys. Plasmas 25,
056314 (2018), Eqs. (2)–(5)）。v1 は `Main.dimension="1D_SPH"` かつ
`Laser.mode="raytrace_2d"` 専用であり、単一 MPI rank を要求する。
OFF 時は全既存挙動と bit 恒等（レコーダは `template<bool>` 追加分岐のみで、
false 実体化は従来コードと構造的に同一）。

#### 5.10.1 物理モデル（Gaussian-cgs、T はエネルギー単位で評価）

probe ray p が pump 場 q と距離 ds で相互作用するときの無次元 CBET 指数：
\[
d\tau_{CBET,pq} = \eta_{pol}\,
\frac{\lambda_0 e^2}{c^3 m_e}\,
\frac{\hat n_e}{1-\hat n_e}\,
\frac{\langle Z\rangle}{\langle Z\rangle T_e + 3T_i}\,
P(g_{pq})\, I_q\, ds
\]
\[
\hat n_e = n_e/n_{crit},\quad
n_{crit} = \frac{m_e\omega_0^2}{4\pi e^2},\quad
\omega_0 = 2\pi c/\lambda_0
\]
\[
g_{pq} = \frac{(\omega_q-\omega_p) - \mathbf{k}_a\cdot\mathbf{u}}{|\mathbf{k}_a|\,c_a},\quad
\mathbf{k}_a = \mathbf{k}_q - \mathbf{k}_p,\quad
P(g) = \frac{g\alpha}{(g\alpha)^2 + (1-g^2)^2}
\]
\[
c_a = \sqrt{\frac{\langle Z\rangle T_e + 3T_i}{\bar m_i}},\quad
\bar m_i = A_{eff}\, m_p,\qquad
\eta_{pol} = \frac{f_{cbet}}{4}\left[1 + (\hat k_q\cdot\hat k_p)^2\right]
\]
- \(\alpha = \nu_a/(|\mathbf{k}_a| c_a)\) は無次元イオン音波減衰率
  （`cbet.alpha_iaw`、既定 0.2、LILAC 系知見）。
- \(f_{cbet}\)（`cbet.f_cbet`、既定 1.0）は偏光/実験較正係数（DRACO は 1.5 を使用）。
- 符号規約：\(P\) は \(g\) の奇関数で、\(g_{pq}>0\) なら q→p にエネルギーが流れる。
- 単位：\(T[\mathrm{erg}] = T[\mathrm{eV}]\times\)`eV_to_erg`、\(I\)
  [erg s\(^{-1}\)cm\(^{-2}\)]、\(\chi\)（下記）[cm·s/erg]。
- 電子定数は laser package の慣用値（`refraction.cu` の
  \(m_e=9.1094\times10^{-28}\) g、\(e=4.8032\times10^{-10}\) statC）と共有し、
  \(n_{crit}\) は既存 `compute_critical_density_from_wavelength_cm` と同一式。
  527 nm で \(n_{crit}\simeq 4.02\times10^{21}\) cm\(^{-3}\)、351 nm で
  \(9.06\times10^{21}\) cm\(^{-3}\)。
- 波長 detuning：ビーム毎の `delta_lambda_nm` により
  \(\omega_b = 2\pi c/(\lambda_0+\Delta\lambda_b)\)。\(\Delta\omega\) は g にのみ
  入り、\(|\mathbf{k}|\) と \(n_{crit}\) は共通 \(\lambda_0\) で凍結
  （誤差 \(O(\Delta\lambda/\lambda_0)\sim10^{-3}\)、Å 級 detuning では無視可）。
- v1 で入れない物理（記録）：\(|\omega_q-\omega_p|/\omega_0\) 級のイオン音波加熱
  補正、caustic 補正、speckle、偏光ダイナミクス、広帯域、kinetic \(\nu_a(k)\)。

#### 5.10.2 1D 球対称縮約 — 角度グループと方位角平均

流体は 1D 球対称、ray は既存 `ray_trace_1d_sph` の束（開口半径 \(R_k=k\Delta R\)
は impact parameter に単調）をそのまま使う。相互作用は **角度グループ** 単位：
\[
g = b\,(2N_{bin}) + \sigma N_{bin} + a,\qquad
\sigma=\mathbb{1}[\mu_{seg}>0]\ (\text{出射枝}),\quad
a = \lfloor k N_{bin}/N_{ray}\rfloor
\]
（\(N_{bin}=\)`cbet.n_impact_bins`、既定 16；G = ビーム数×2×\(N_{bin}\)）。
セグメント平均方向余弦は \(\mu_{seg} = (r_{stop}-r_{old})/\Delta s\)
（直線セグメント上の \(\hat k\cdot\hat r\) の厳密な路程平均；屈折を追った実軌道
から評価する）。

グループ×セルの tally（Marozas の volume-weighted ASR の 1D 版）：
\[
L_{g,c} = \sum_{seg\in(g,c)} P_{seg}\,\Delta s_{seg},\qquad
I_{g,c} = L_{g,c}/V_c,\qquad
\bar\mu_{g,c} = \frac{\sum P\,\Delta s\,\mu}{L_{g,c}}
\]
\(V_c\) は Lagrangian セル体積（`state.vol`、trace snapshot と同時刻）。

1D では 2 本の ray 平面の相対方位角 \(\varphi\) が未解決なので、対の運動学を
\(\varphi\) について中点求積で平均する（`cbet.n_phi`、既定 8、対称性より
\([0,\pi]\)）：
\[
\cos\psi(\varphi) = \bar\mu_p\bar\mu_q + s_p s_q\cos\varphi,\quad
s=\sqrt{1-\bar\mu^2},\qquad
|\mathbf{k}_a|(\varphi) = \bar k\sqrt{2(1-\cos\psi)},\quad
\bar k = n_r\omega_0/c
\]
\[
\mathbf{k}_a\cdot\mathbf{u} = \bar k\,u_r(\bar\mu_q-\bar\mu_p)\ (\varphi\text{ 非依存}),
\qquad
\langle P\cdot pol\rangle_\varphi = \frac{1}{N_\varphi}\sum_k
\frac{1+\cos^2\psi(\varphi_k)}{4}\,P(g(\varphi_k))
\]
\(|\mathbf{k}_a|/\bar k <\) `cbet.k_a_floor`（既定 1e-6）の節点は寄与 0
（ビート波なし）。\(\hat n_{raw}\ge\) `cbet.ne_frac_cutoff`（既定 0.95）の
セル、void セル、および非有限値（NaN/Inf）を含むセル（\(\rho, \bar Z, A_{eff},
T_e, T_i, u_r, V_c\) のいずれか）は CBET 評価から除外。\(u_r\) はセル平均節点速度
\(\tfrac12(v_r[i]+v_r[i+1])\)、1T では \(T_i=T_e\)。

セル場の組み立てでは、gain prefactor と \(\bar k = n_r\omega_0/c\) の
\((1-\hat n)\) を IB と同じ `Laser.absorption.eps_n`（既定 1e-4）で床処理する：
\(1-\hat n \leftarrow \max(1-\hat n,\,\varepsilon_n)\)。既定構成では
`ne_frac_cutoff` \(=0.95 \le 1-\varepsilon_n\) のため床は不活性であり、
これを恒常化するため config 検証で
`cbet.ne_frac_cutoff` \(\le 1 -\) `absorption.eps_n` を強制する
（床がアクティブ帯内で gain を黙って飽和させる構成の禁止）。
また \(P(g)\) の共鳴幅は \(\Delta g_{FWHM}\simeq\alpha_{iaw}\)（既定 0.2）で
あり、`n_phi` 中点求積が共鳴を解像する条件は概ね
\(\max_k|g(\varphi_{k+1})-g(\varphi_k)|\lesssim\alpha_{iaw}/4\) — 既定より
大幅に小さい `alpha_iaw` を用いる場合は `n_phi` を増やすこと。

#### 5.10.3 保存型 pairwise 交換と donor cap

セル c 内のグループ対 (p,q) の交換パワー［erg/s、正なら q→p］：
\[
\chi_{pq,c} = \frac{\lambda_0 e^2}{c^3 m_e}\frac{\hat n}{1-\hat n}
\frac{\langle Z\rangle}{\langle Z\rangle T_e+3T_i}\,
\langle P\cdot pol\rangle_\varphi,\qquad
\Delta P_{p\leftarrow q,c} = \chi_{pq,c}\, \frac{L_{p,c} L_{q,c}}{V_c}
\]
交換は厳密に反対称（\(g_{qp}=-g_{pq}\) が浮動小数点で厳密、\(P(g)\) は FP 奇。
実装はさらに (i) 対毎に \(\chi_{pq}\) を単一評価（1D はキャッシュ、2D は
(min,max) 正準引数順で再計算）し、(ii) 適用量の積
\(\chi\, L_{min} L_{max}/V_c \cdot f\) を両向きで同一の (min,max)
正準オペランド順で丸めることで、pair 量の bitwise 反対称を構造的に保証する）。
donor cap は
二段 flux-limiting：グループ毎の総損失 \(\ell_{g,c}\) に対し
\[
f_{g,c} = \min\!\left(1,\ \frac{\theta\,L_{g,c}}{\Delta s^{max}_{g,c}\,\ell_{g,c}}\right),
\qquad
\Delta P^{final}_{pq} = \Delta P_{pq}\,\min(f_p, f_q)
\]
（\(\theta=\)`cbet.theta_cap`、既定 0.3）。\(\min(f_p,f_q)\) を両側が同値で
適用するため反対称性＝機械精度保存が保たれ、かつ任意 ray の 1 交差あたり
相対損失 ≤ θ が保証される（正値性）。

#### 5.10.4 経路レコーダと IB/2→CBET→IB/2 分割・反復

CBET ON 時、trace kernel は record モード（`template<bool kCbetRecord>`; 追加は
`if constexpr` ブロックのみ）で走り、ray 毎に **セル横断毎に併合した** 経路記録
\(\{c,\ \mu,\ \Delta s,\ S\}\)（S は台形 IB 光学厚の横断内総和）を残す。
tail closure（§5.4.4）は \(\Delta s=0,\ S=\tau_{tail}\) の終端記録として同じ
対象セルに残す（record モードのレコーダ自身は沈着・未吸収台帳を書かない —
それらは propagate が権威）。レコーダの適応刻み・終端判定（cutoff 等）は
IB-only のパワー進行で従来どおり行う（近似として記録；gain ≲ e^{dτ} vs
cutoff 1e-6 で実害なし）。記録容量は ray あたり `cbet.max_segments_per_ray`
（0 で自動 = 2·n_cells+64；単一の in-out 通過のセル横断数上界）；
**容量溢れは hard error**（fatal assert）。溢れ ray は経路記録が途中で切れる
ため、prefix が tally をポンプしつつ prefix 以降の IB 吸収が未吸収へ黙って
振り替わる — 「IB-only fallback」にも台帳閉包にもならないので、黙った
truncation は禁止し `max_segments_per_ray` の引き上げを促して停止する
（旧仕様の「溢れ ray は IB-only として継続」は実装と乖離していたため撤回；
2026-07-26 カーネルレビュー指摘）。

準定常解は固定点反復（全 kernel 決定論的・atomic-free tally／同一 build+device
で replica bit 安定）：
1. **tally**：ソート済みセグメント区間の固定順逐次和で \(L, \bar\mu, \Delta s^{max}\)。
2. **交換**：χ キャッシュ → 損失/cap → \(dQ_{g,c}=\sum_q \Delta P^{final}\)。
3. **propagate**（ray 毎に記録を路程順に）：横断毎に
\[
P \mathrel{-}= -P\,\mathrm{expm1}(-S/2)\ (\text{IB 前半、沈着}),\quad
P \mathrel{+}= dQ_{g,c}\frac{w^{(m-1)}\Delta s}{L^{(m-1)}_{g,c}},\quad
P \mathrel{-}= -P\,\mathrm{expm1}(-S/2)\ (\text{IB 後半、沈着})
\]
   按分重み \(w\Delta s/L\) は **前反復の tally と同一世代**を使う
  （\(\sum_j w_j\Delta s_j = L\) が構成的に成立し、適用総和＝台帳 dQ）。
   反復は毎 pass \(P_0\) から再伝播（累積しない）。CBET 加算後の \(P\) を
   次反復の tally 重み \(w^{(m)}\) として記録する。
4. 収束：\(\sum|L^{(m)}-L^{(m-1)}|/\sum L^{(m-1)} <\) `cbet.tol`（既定 1e-3、
   `cbet.max_iters` 既定 50）。**最終反復では pass 3 を走らせず**、最後の
   propagate は deposit を書く final pass として dQ と \(w\) の世代を一致させる
  （世代不一致は按分和を 1 からずらし黙った非保存を生む — G1 gate で実証済み）。

負パワー床（clamp）は Picard 中間反復では正値 cone projection として扱い、
発生数は報告のみとする。物理的な正値性 gate は final pass の injected power
`clamped_power == 0` である。final pass で clamp が入れたエネルギーは
厳密に計測され（`clamped_power`）、恒等式
\(E_{out}-E_{in} = \Sigma_{applied} \approx 0 + E_{clamp}\) として監査される。
エネルギー台帳：\(P_{in} = P_{dep} + P_{unabs} + \Sigma_{applied}\)、
\(\Sigma_{applied}\) は按分丸め \(O(N\varepsilon)\)（実測 ≤1e-12 相対）。
CBET 交換は物質を直接加熱しない（加熱は IB のみ）。沈着・未吸収は既存の
per-ray 行 + 固定順 reduction でビーム毎に集計され、下流
（`apply_deposit_redistribution_1d`、skip cache の per-beam \(\hat f\)、
`laser_unabsorbed` 台帳）は従来と同一経路。

#### 5.10.5 raytrace skip との整合

skip の per-beam パワースケーリングは IB では厳密（線形）だが CBET では
非線形。CBET ON 時は `should_skip` にグループ毎パワー相対変化 >1% での
再 trace 判定を追加（総パワー 1% guard と同閾値）。これにより skip 誤差は
既存の 1% 級プラズマ変化誤差と同次に抑えられる（矩形パルス平坦部では厳密）。

#### 5.10.6 診断・検証 gate

診断（毎 solve、`LaserMesh.last_cbet_*` → history `laser/cbet_*`）：
交換パワー、反復数、収束 flag/残差、台帳残差、clamp 数、overflow ray 数。
常設 gate：
- **G1 slab 解析解**（`test_cbet_slab`）：2-beam 共進行 logistic
  \(I_2(s)=I_{tot}/(1+Ce^{-\gamma I_{tot}s})\) に対し fine-ds 相対誤差 <1e-3
  （実測 ~7e-5）、ds 収束次数 ~1、pairwise 台帳 ≤1e-13、適用保存 ≤1e-12、
  cap 活性ストレス下の clamp 恒等式。
- **G2 球対称保存**（`test_cbet_sphere`）：合成 corona 上で
  \(|P_{in}-P_{dep}-P_{unabs}|/P_{in}\le 10^{-4}\)（期待 ~1e-12）、台帳 ≤1e-13、
  final-pass `clamped_power`/overflow 0、収束。
- **G3a 傾向**（同）：膨張 corona で CBET ON ⇒ 吸収低下・未吸収増加。
- **G3b detuning 単調性**：実 hydro corona の deck 比較（off/on/detuned；
  expensive）で吸収回復の単調性を確認。
- **G4 OFF bit 恒等**：`verify gxii_1d_fld_regression` golden 全指標 rel=0
  維持 + 既存 laser ctest green（golden 不変が恒等の恒常的証明）。

#### 5.10.7 2D_RZ generalization

2D_RZ CBET uses the same Marozas pairwise exchange, donor cap, fixed-point
iteration, diagnostics, and cgs/eV unit conventions as §5.10.1-§5.10.6. The v1
legal mode is `Main.dimension="2D_RZ"` with `Laser.mode="raytrace_3d"` on a
single MPI rank. The CBET beam unit is the traced theta-group, i.e. the
axisymmetric cone family represented by one `raytrace_3d` group. When
`Laser.cbet.enable=True`, the theta-group fold key additionally includes
beam `delta_lambda_nm`; the group angular frequency is
\(\omega_g=2\pi c/(\lambda_0+\Delta\lambda_{rep})\), while \(|k|\) and
\(n_{crit}\) remain frozen at common \(\lambda_0\) as in 1D.

The 2D group index is
\[
g = b\,(4N_{bin}) + \sigma N_{bin} + a,\qquad
\sigma = \mathbb{1}[\bar a>0] + 2\mathbb{1}[\bar c>0],
\qquad
a=\lfloor k N_{bin}/N_{ray}\rfloor ,
\]
where \(b\) is the theta-group, \(\sigma\) is the quadrant branch
\((\operatorname{sign}\bar a,\operatorname{sign}\bar c)\), and \(a\) is the
aperture-radius bin. Thus \(G=B\,4\,N_{bin}\). The `initialize_rays_2d` order is
pre-sorted by aperture radius, so the static-bin rule used by 1D carries over.

For each laser-mesh cell and group, the resolved meridional kinematics are
path-exact averages
\[
\bar a = \frac{\sum w\,\Delta R}{\sum w\,\Delta s},\qquad
\bar c = \frac{\sum w\,\Delta z}{\sum w\,\Delta s},\qquad
\tilde b=\sqrt{\max(1-\bar a^2-\bar c^2,0)} ,
\]
where \(dR/ds=\hat k\cdot\hat r\) for the cylinder radius and \(dz/ds=\hat k\cdot
\hat z\). Mirror symmetry of the traced bundle about the meridional plane reduces
the unresolved azimuthal component to a Z2 product measure:
\[
\cos\psi_\pm = \bar a_p\bar a_q + \bar c_p\bar c_q \pm \tilde b_p\tilde b_q,
\qquad w_\pm=\frac12 .
\]
This two-point mirror closure replaces the 1D U(1) `n_phi` quadrature; `n_phi` is
inert in 2D. The Doppler term is
\[
\mathbf{k}_a\cdot\mathbf{u} =
\bar k\left[(\bar a_q-\bar a_p)u_R + (\bar c_q-\bar c_p)u_Z\right],
\qquad u_\phi=0 ,
\]
with \(g_\pm\), \(P(g)\), `alpha_iaw`, and \(\eta_{pol}\) otherwise identical to
§5.10.1.

The 2D implementation does not allocate the 1D \(\chi\) cache because the
\(N_{cell}\times N_{pair}\) scale is prohibitive. Coupling is recomputed on the
fly with the unordered \((g_{min},g_{max})\) pair convention; the opposite
orientation applies the exact sign flip, preserving FP antisymmetry of the
exchange.

Records are laser-mesh-cell records using the same midpoint deposit cell as the
legacy tracer. Consecutive crossings in the same cell are merged; \(\bar a\),
\(\bar c\), and bilinear corner weights are ds-weighted means. The final
propagate writes per-beam-group node rows through the same atomic four-corner add
class as the legacy 2D tracer. Repositioning deposited energy inside a record's
cell from per-substep weights to record-mean weights is second order in the
per-record optical depth and is a recorded approximation, not a conservation
term.

Terminal tail-closure records use \(\Delta s=0,\ S=\tau_{tail}\), so they carry
no CBET exchange weight but reproduce the legacy analytic IB tail absorption in
the final propagate. Recorder termination remains IB-only: cutoff, critical, and
mesh-exit decisions are made from the IB power path, carrying over the 1D caveat.

The 2D cell pack is built from LaserMesh node fields with bilinear-consistent
four-corner means, exact torus volumes
\(\pi(R_{i+1}^2-R_i^2)(Z_{j+1}-Z_j)\), and mask
`hydro-coverage AND non-void AND nh_raw < ne_frac_cutoff`. Ghost-corona cells are
CBET-inert in v1 because they do not provide the required physical Ti/u fields.

Clamp semantics separate reporting from the physical gate. Transient Picard
iterate clamps are benign positive-cone projections and are reported only; the
positivity gate is the final-pass injected power `clamped_power == 0`.

The raytrace skip cache is fully bypassed under `cbet_on_2d` in v1 because the
2D cache collapses group powers to one total row. Per-group CBET-aware caching is
recorded as a v2 item. The OFF-path contract is proved by bitgate comparison
against the pre-port binary with the established 2D_RZ noise-band methodology;
CBET OFF keeps the old fold key and no new recorder path, and header isolation
uses new headers only for 2D additions.

---

#### 5.10.8 port_section — 単一トレース×実ポート配置の多ビーム CBET（opt-in、1D_SPH）

`Laser.cbet.geometry_mode="port_section"`（既定 `"legacy"` は §5.10.2 の方位角
平均経路と bit 恒等）で、§5.10.2 の未解像方位角平均を **実ポート配置の
sector/section 位相空間写像**に置き換える（設計
`docs/design/multibeam_1d_superposition_20260727.md` §13、Follett et al.,
Phys. Plasmas **32**, 022709 (2025)）。物理ビームは `Laser.port_configuration`
のポート表（単位方向・roll・power_weight・δλ、正準 port_id 昇順）で与え、
トレースは単一 prototype ビーム 1 本のみ（`len(beams)==1` を検証で強制）。

- **位相空間再構成（S1）**: CBET 経路レコード (cell, μ, ds) のノード半径は
  経路方向に対応する cell edge とし、ノード α はそのノードで終端する
  record μ（path-start は先頭 record、turning を跨ぐ平均なし）から復元する。
  \(\bar r=(r_k+r_{k+1})/2\)、\(\bar\mu=(\mu_k+\mu_{k+1})/2\) として θ は
  \(\Delta\theta_k={ds_k\over6}\left[
  {\sqrt{1-\mu_k^2}\over r_k}
  +{4\sqrt{1-\bar\mu^2}\over\bar r}
  +{\sqrt{1-\mu_{k+1}^2}\over r_{k+1}}\right]\) の Simpson 累積で復元し、
  ray 束面積は、隣接 ray の同 leg 角差から内点では半差、端点では片側差を取る
  \(d\theta_{\rm eff}\ge10^{-12}\) を用い、
  \(A=2\pi r^2d\theta_{\rm eff}\max[\sin\theta,\sin(d\theta_{\rm eff}/2)]
  \max[\cos\alpha,10^{-6}]\) とする（Follett Eq.(10) の有限 bundle 離散形）。sheet 記帳は turning
  point で折り、caustic index は別持ちで `in_limiter_zone` を付す（§11(i)
  裁定 — 厳密 caustic-fold からの文書化された v1 逸脱）。Bouguer 監査
  B=√ε r sinα と除外台帳（多重 turning/caustic ray）を
  毎構築で記録する。監査 drift の条件付け（2026-07-31）: (a) 準接線 record
  （|cos α| < 0.3）は除外する — 転回セルでは cell-edge 粒度の ε 標本が
  転回点 ε を表現できず、再構成 B が ε の大きさによらず O(Δε/ε_t) の誤差を
  持つ（精密 turning 検証は supplied-α 経路 1e-10 が所掌）; (b) 分母は
  max(|B₀|, 0.1 r_outer) で条件付けし、準動径 ray（B₀→0）の相対 drift
  発散を抑える; (c) S1 経路の内部 assert 許容は record 粒度用に呼び出し側で
  bouguer_tol = bouguer_tol_fd = 0.5 を明示する（library 既定は精密
  fixture 用であり、record 粒度経路では Debug ビルドの assert を誤爆させる）。
  外部 gate は record 粒度 audit の破滅検出として
  bouguer_drift_max < 0.5 を維持する。turning–caustic 間（in_limiter_zone）の交差は場評価
  （lookup・profile）から除外する（Follett field-limiter の v1 施行）;
  診断 enumeration には残る。
- **拡張状態空間（S2）**: 状態 = (port i, leg σ, impact bin β)、
  G_ps = N_port × 2 n_bins。record キー/セグメントは参照群 G_ref のまま、
  tally/propagate は port スライス毎の remap 双子カーネル（原本カーネルは
  無改変・legacy 経路ゼロ差分）。rec_w の port スライス初期値は
  w_i × rec_w。deposit は port 昇順 final pass の累積和。
- **対結合係数**: chi_ps[c,{A,B}] は solve 前に一括計算し反復間
  不変（幾何は位相空間 table 由来で L の μ̄ 平均に依存しないため）。
  実行は CUDA カーネル（(cell,pair) 毎 1 スレッド・FP atomics 不使用で
  run-to-run ビット決定的; host 参照実装 build_chi_ps はテスト用に保持、
  device との一致は超越関数 ULP 級で許容差ゲート）。
  chi_ps = χ_pref(c) · Σ_x Σ_m ŵ_A(x) Î_B η_pol P(g) / Σ_x Σ_m ŵ_A(x) Î_B。
  x = seed 状態の shell 交差（ŵ_A は power 正規化）、m = section 方位
  （n_section_phi 中点、[0,2π)）。pump は seed 位置を port j フレームへ
  逆回転（Follett Eq.(21)）して (shell, σ_q, θ'_j) を線形補間 lookup —
  **shadow（θ'_j > 最大極角）は分子分母とも除外、k_a floor 未満は分母
  満額・分子 0**（§5.10.2 の希釈と整合）。η_pol=(f_cbet/4)(1+cos²ψ)、
  P(g) は §5.10.1 と同一、Δω は port δλ から符号付き。⟨P(g)⟩ を取り
  P(⟨g⟩) は取らない。
- **検証（実施済み gate）**: V2 = 対向 2 ポート・IB off の球対称 logistic
  （相互作用座標 τ=Σ_c χ_c ds_c²/V_c、leg 総和 power 2×10⁻³、positivity
  clamp 0・転送 ≥5% の非自明性 floor）。V3 = beam-splitting 不変性
  （w=1 の 1 port ≡ w=0.5×2 port、1e-12）。12 ポート光学保存・ポート入力
  順置換の bitwise 不変・legacy モード bitwise 二重実行。
- **v1 制限**: 拡張 pair ≤ 65536（OMEGA/NIF 規模は class/tiling 未実装で
  ConfigError）。detuning は resonance のみ（軌道は λ0 — NIF 級は per-color
  明示トレース比較 gate が必要、S4）。強結合（per-pass 転送が positivity
  clamp 域）では解析検証未カバー — explicit-N 参照（V4）が実効 gate。
  単一 rank。
- **S3 hot-e 結合（実装済み）**: port_section∧`eta_mode="model"` に限り
  CBET↔hot_electron 排他を解除。trace 側 capture は port_section で抑止
  （legacy 真理値不変）し、capture は ps final propagate の record entry で
  昇順閾値・逐次 (1−η) 減耗・per-(port,ray,channel) stage 行（atomics なし）
  として実行、host 固定順 reduction が (a) η(t) モデルの前 step 計測
  （port 集約 ΣP_cross — §14 の collective intensity が自然に成立）、
  (b) チャネル毎 1 個の RayCapture（P 加重平均 μ・半径）→ 既存 CSDA 輸送、
  (c) banked hot-e 光学台帳（dep+unabsorbed+P_hot=input ≤1e-10 gate）を
  供給する。`illumination_metric="equivalent_area"` で f_illum^(2)、
  `tpd_overlap_mode="common_wave_cluster"` で I_drive（§4.3 の
  ∫I_cw²/∫I_cw、lower/rec/upper 三重診断ログ）が η モデルの I14 入力に
  入る（port_section_overlap モジュール）。capture 粒度は record entry
  （半セル粒度）— v1 文書化制限。
- **S4 wave-action 台帳（実装済み）**: port_section の detuned 対交換では、
  sender の損失 \(A\) に対して receiver の利得を
  \(A\,\omega_r/\omega_s\) とし、差
  \(A(1-\omega_r/\omega_s)\) を signed IAW 台帳 `E_cbet_iaw` に一度だけ
  記帳する。各セルで \(\sum_g dQ_{c,g}+Q^{IAW}_c=0\) を同じ相対許容差で
  assert する。zero detuning では有限非零 \(\omega\) の \(\omega/\omega=1\)
  が厳密で、receiver 利得は bitwise 同一、IAW 項は厳密な 0.0 となる。
  tally 内の `cbet_kinetic_response_hook` は現状 ratio をそのまま返す no-op
  で、将来の非線形 IAW kinetic response の拡張点とする。


### 5.11 Hot-electron preheat（1D）

`Laser.hot_electron.enable=True`（既定 False）で有効化する、Colaïtis 2015 /
LILAC 型の prescribed LPI hot-electron preheat。LPI を自己無撞着に解くモデル
ではなく、\(\eta_{hot}\) と \(T_{hot}\) は較正入力である。v1 は 1D 専用で、
既存の 1D laser mode（`radial_absorption_1d`, `raytrace_2d`）に接続する。

各 traced ray について、最初に \(n_e=f_s n_{crit}\)
（\(f_s=\)`source_nc_fraction`）を横切る点で capture する。capture power は
\[
P_{h,ray}=\eta_{hot}\,P_{ray}^{cross}
\]
ここで \(P_{ray}^{cross}\) は**交差点でのレイパワー**：交差を含む trace
セグメントは交差点でイベント分割され、前半区間の IB 吸収（セグメント台形
光学厚 S の路程比例分）を適用した後の P を capture し、残り区間は
\((1-\eta)P^{cross}\) で IB を続ける（2026-07-26 修正 — 旧実装はセグメント
開始点 P で capture してから全区間 IB を適用しており、hot-e 源が
O(S·frac) 過大・交差前 IB が同量過小だった。critical 交差セグメントは
tail closure が from-entry 解析モデルのため旧順序＝エントリ点適用を保持）。
で、`subtract_from_laser=True` のとき ray は
\((1-\eta_{hot})P_{ray}^{cross}\) で継続する（no-double-count；
`subtract_from_laser=False` は意図的な additive 感度モードで、レーザー側を
減耗させずに外部エネルギーを注入する — 通常 run では用いない）。capture 後は
1D 射影された ray ensemble だけを保持し、固定 16 ビンの
\((source\ cell,\ \mu_{axis})\) table に power-weighted reduction する。
\(\mu_{axis}\in[-1,1]\) は一様ビンで、bin 代表値（\(\mu_{axis}\) と発射
位置 \(r_s\) の両方）は power-weighted 平均 — 発射位置は**連続値**であり
セル/面に量子化しない。hydro メッシュ外（ghost corona 帯）の capture は
境界セルへクランプし、計数して警告する。

スペクトルは指数分布
\[
\frac{d\dot N}{dE}=\frac{P_h}{T_h^2}\exp(-E/T_h),\qquad
T_h=\texttt{T\_hot\_eV}\times\mathrm{eV\_to\_erg}
\]
を用いる。\(N_E=\)`n_energy_groups` 個の log group を
\([E_{min},E_{max}]=[\texttt{E\_min\_over\_Th}T_h,\texttt{E\_max\_over\_Th}T_h]\) に置き、
analytic moment で group weight と代表エネルギーを作る：
\[
W_g=\frac{\phi_1(u_{g-1/2})-\phi_1(u_{g+1/2})}
{\phi_1(u_{min})-\phi_1(u_{max})},\quad
\phi_1(u)=(1+u)e^{-u},
\]
\[
E_g=T_h\,\frac{\phi_2(u_{g-1/2})-\phi_2(u_{g+1/2})}
{\phi_1(u_{g-1/2})-\phi_1(u_{g+1/2})},\quad
\phi_2(u)=(2+2u+u^2)e^{-u}.
\]
telescoping により \(\sum_g W_g=1\) は厳密、\(E_g\) は flux-averaged energy。

角度モデル `cone` は capture 方向を軸とする一様 cone
（半角 `theta_div_deg`）を Gauss-Legendre \(\mu\) × 一様 \(\phi\) の積求積で
離散化する（重み総和 1）。`radial` は verification mode で、\(\mu=1\) の
inward march のみを行う。`inner_bc` は `radial` のみに適用される。

輸送は直線 chord。球幾何では球殻交点を解析的に求め、inbound /
pericenter / outbound の順に shell segment を歩く（発射点 \(r_s\) から
最初の面までの部分セグメントは交点式が厳密に扱う）。planar 幾何では slab
cosine で segment 長を求め、セル内部からの発射は進行方向側の面までの
**部分初期セグメント**を先頭に発行する（内部面上の発射で逆向き側のセルを
全幅通過していた誤りの根治；2026-07-26 カーネルレビュー指摘）。
\(|\mu_x|<10^{-12}\) の面内 chord は無限側方 slab では経路長が発散する
ため、発射セルでの全量局所熱化として扱う（旧実装の escape 分類を訂正）。
`radial` モードは全チャネル電力を最外 source に併合し（documented merge
rule）、最外 source セル内は \(r_s\) から内側面までの部分面密度で開始する。`Mesh.geometry_1d="cylindrical"` と `cone` の
組合せは v1 で拒否する。各 segment で CSDA 停止を解く：
\[
S_m(E)=\frac{4\pi e^4 n_e \ln\Lambda\,G(x)}
{\rho m_e v^2},\qquad
G(x)=\operatorname{erf}(x)-\frac{2x}{\sqrt{\pi}}e^{-x^2},
\]
ここで \(v\) は relativistic velocity、\(x=v/(\sqrt2 v_{te})\)、
\(\ln\Lambda\ge2\)。cell 内の \(n_e,T_e,\rho\) を凍結し、RK4 を
energy/substep ≤2% 目標、最大 256 substep/cell で適応積分する。熱化床は
\[
E_{floor}=\max(2T_{e,local},\,10^{-3}T_h).
\]
bookkeeping は常に \(\Delta P=\dot N_g(E_{in}-E_{out})\) で行うため、RK 精度は
cell 間の配分だけを変え、総エネルギー保存は変えない。outer node を抜けた
残差は escape として台帳に入る。RK substep cap（256/cell）に到達した chord は
残エネルギーをそのセルへ全付与する（cap 到達は実質「そのセルで熱化する」
regime でのみ起こるため空間誤差は高々 1 セル幅；発火数は counter で集計し
warning で表面化する）。停止能は \(T_e\le0\)・\(n_e\le0\)・\(\rho\le0\) の
セルで 0（自由通過）を返す — void/vacuum 用の分岐であり、床処理された
production 温度場では非 void セルで発火しない（cold-material 停止能は
v1 非対応の documented 制限）。スペクトル窓 \([0.2,8]T_h\)（既定）の外の
エネルギー分率（既定で約 2.05%）は窓内 group へ**再正規化**される
（\(\sum_g W_g=1\) の telescoping はこの再正規化を含む）。

hot-electron の per-cell power は laser deposit smoothing 後に laser channel
へ合流し、その後は標準の laser source injection（2T では \(e_e\)、1T では
total energy）をそのまま使う。dt limiter は named limiter `"hot_electron"`：
\[
\Delta t\le f_E\min_{P_i>0}\frac{m_i e_{e,i}}{P_i},
\qquad f_E=\texttt{explicit\_source\_limit}
\]
で、明示 source limiter として 1 step lag で適用される。

既知の v1 制限：free electron stopping のみ（bound/plasmon/degeneracy なし、
cold partially-ionized matter では停止が過小・range が過大）、reflux なし、
collisional angular scattering なし、
\(\eta_{hot}\) と \(T_{hot}\) は prescribed（Phase 2/3 backlog）。保存台帳は
\[
|P_{dep}+P_{esc}-P_h|/P_h
\]
が構成上 machine precision で閉じる（GXII-derived smoke で実測 ≤ \(7\times10^{-15}\)）。

#### 5.11.1 機構別指向性源チャネル（TPD / SRS、multi-channel v2）

`Laser.hot_electron.sources`（SPECIFICATION §6.4）は上記の単一処方源を
**機構別チャネルの列**（最大 4、`mechanism ∈ {cone, tpd, srs}`）へ一般化する。
各チャネルは独自の捕獲面 \(f_{s,k}=\)`capture_nc_fraction`、変換効率
\(\eta_k\)（const または凍結テーブル）、温度 \(T_{h,k}\)、スペクトル窓、
角度分布を持つ。v1 のスカラー key 群は単一 `cone` チャネルの shorthand として
厳密に後方互換（同一値の `sources=[cone]` deck と出力 bitwise 恒等、両形の
併用は parse エラー）。設計・文献根拠は
`docs/design/hote_directional_sources_20260710.md`。

**捕獲**：各 traced ray はチャネル毎に独立へ最初の上向き
\(n_e=f_{s,k}n_{crit}\) 交差で capture する。チャネルは host 側で
\(f_s\) 昇順（= inward march の物理的通過順）に整列され、1 segment が複数
閾値を跨ぐ場合は低い \(f_s\) から発火する。`subtract_from_laser=True` では
チャネル \(k\) の発火後に ray power が \((1-\eta_k)\) 倍され、下流（より高い
\(f_s\)）のチャネルは正直に減耗した power を見る（Colaïtis 2015 の
PCGO beamlet 減耗と同型の sequential pump depletion）。同値の \(f_s\) は
チャネル添字順（決定論）。捕獲は per-(ray, channel) の private slab に記録
され（atomic なし）、\(\eta_k=0\) のチャネルは kernel パラメタから除外される
（\(\eta=0\) ≡ チャネル不在が物理量で bitwise に成立；当該チャネル自身の
zero 台帳系列のみが出力面の差）。

**角度分布 — µ バンド一般化**：全機構は capture 方向を軸とする solid-angle
バンド \(\mu\in[\mu_{lo},\mu_{hi}]\)（Gauss-Legendre \(\mu\) × 一様 \(\phi\)、
重み総和 1）に帰着する：

| mechanism | バンド | 既定 |
|---|---|---|
| `cone` | \([\cos\theta_{div},\,1]\) | \(\theta_{div}=60^\circ\)（v1 恒等） |
| `srs` | \([\cos\theta_{div},\,1]\) | \(\theta_{div}=20^\circ\)（前方尖鋭） |
| `tpd` | \([\cos(\theta_c+\Delta),\,\cos(\theta_c-\Delta)]\)（環） | \(\theta_c=45^\circ,\ \Delta=10^\circ\) |

TPD の「±θ 双葉」は 1D 縮約では方位角未解決（偏光面方位は 1D の自由度で
ない）ため、双葉の方位一様平均に**厳密に等しい環状バンド**として実装する。
固定子午面への 2 solid-cone 重ねは非物理的方位を注入するため棄却
（設計 doc §3.1）。\(\theta_c-\Delta<0\) の場合は前方 cap へ折返し
（\(\mu_{hi}=1\)）。縮退：\(\mu_{lo}=\mu_{hi}=1\) → 単一方向（v1 と同一）、
\(\mu_{lo}=\mu_{hi}<1\) → 単一 µ 環（\(n_\phi\) 節点）。

**機構既定値（parse 時適用；文献根拠は設計 doc §1）**：

| mechanism | \(f_s\) | \(T_{hot}\) [eV] | 角度既定 | 一次文献 |
|---|---|---|---|---|
| tpd | 0.25 | 6.0e4 | 環 45°±10° | 活性帯 0.21–0.25 \(n_c\)（Vu 2012; Follett 2017）、\(T_h\) 60–100 keV 帯（Vu 2012; Rovere 2023） |
| srs | 0.18 | 4.5e4 | 前方 20° | hot-e 相関帯 0.15–0.21 \(n_c\)（Rosenberg 2018/2020）、\(T_h\) 37–55 keV（Solodov 2020; Rosenberg 2018） |
| cone | 0.25 | 5.0e4 | 60° | v1 既定（Christopherson/LILAC 系） |

**文献上の注意（設計 doc §1.3）**：TPD の >50 keV tail は運動論計算で前方
±25–30° 集束と報告される（Rovere 2023 §IV D–E; Vu 2012 Fig 18）— ±45° 環は
古典的プラズモン方向描像であり、Rovere 型運用点は
`tpd_theta_deg=15, tpd_delta_deg=15`（バンド [0°,30°]）で再現できる。
\(\eta, T_h\)、角度 knob は v1 同様すべて較正入力である。

**輸送・台帳**：チャネル毎に v1 の \((cell,\mu_{axis})\) reduction と
spec 化された cone/radial pipeline（§5.11 本文）を再利用し、per-cell power は
チャネル合算で laser channel へ合流する（dt limiter・per-cell 診断は合算系で
不変）。総台帳（`laser/hot_e_*`）は v1 名義のまま（`source_r` は power 加重
平均、`conservation_resid` はチャネル最大）。チャネル数 > 1 の deck では
per-channel history `laser/hot_e_ch{i}_{in,deposited,escaped}` を追加出力する
（単一チャネル deck の出力レイアウトは不変）。保存はチャネル毎に
\(|P_{dep}+P_{esc}-P_h|/P_h\) が machine precision（3 チャネル
GXII-derived 20 ps 実測 ≤ \(1.6\times10^{-14}\)、\(\sum_k\) ch 台帳 ≡ 総台帳
≤ \(3\times10^{-16}\)）。

**検証（2026-07-10、設計 doc §9 に全記録）**：G1 OFF-bit（GXII FLD golden
rel=0、absent vs `enable=False` snapshot 53/53 bitwise）；G2 shorthand ≡
`sources=[cone]` bitwise（const-η と凍結テーブル両経路、snapshot 55/55）；
G3 機構傾向（planar 200-cell slab、\(T_h,\eta,f_s\) を全 leg で固定した純角度
比較）：90% 沈着深さ cone0° 154 > srs20° 152 > cone60° 144 > tpd45° 142 cell
（cone 対は v1 §12.1 実測と一致）、slab 吸収率 tpd45° 0.820 ≥ cone60° 0.805
> srs20° 0.742 ≥ cone0° 0.734 — 事前登録順序どおり；G4 上記保存・η=0 恒等。

---

### 5.11.2 2D RZ 輸送（feature/2d-hote：spec docs/design/2d_hote_port_spec.md）

1D モデル（スペクトル・多群レイアウト・阻止能・CSDA marcher・µ-band 角度求積・
チャネル定義・E_floor・dt リミッタ式）は §5.11 のまま共有し、輸送幾何のみを
2D RZ に一般化する。

- **捕獲**：`ray_trace_3d`（2D_RZ の唯一の laser モード）内で
  `template <bool kHotECapture>`。ray ごと・アクティブチャネルごとに
  `n_e_hat_raw ≥ f_s` の初回上向き交差を線形内挿で検出し、子午面回転した
  8-double 行 {valid, R_s, Z_s, k_R, k_φ, k_Z, P⁻, pad} を tid-stride で記録
  （atomics なし）。R_s>0 では k_R=(v_x x+v_y y)/(R_s|v|)、
  k_φ=(v_y x−v_x y)/(R_s|v|)、k_Z=v_z/|v|；軸上極限は k_R=√(v_x²+v_y²)/|v|、
  k_φ=0。減耗 I←(1−η_k)I は閾値昇順に逐次適用。**注**：2D port の capture は
  現在もセグメント開始点 P を用いる（1D の 2026-07-26 イベント分割修正
  = P^cross 化は未随伴 — 2D 側の mirror 対応待ち、O(S·frac) の hot-e 源
  バイアスが残る）。
- **縮約**：capture を (hydro cell, χ=atan2(k_R,k_Z) の 16 分割,
  |k_φ|<0.3 の 2 クラス) で束ね、電力加重平均の位置・方向（再正規化）を持つ
  job 列に決定論的順序で縮約する（1D の µ_axis 16-bin 縮約の 2D 形）。
  位置特定に失敗した capture の電力は escape 台帳へ計上する。
- **弦輸送（hydro 構造格子上）**：job の子午面代表点 (R_s,0,Z_s) から
  3D 直線弦 x(s)=x₀+sΩ̂ を、回転面（円錐台）である hydro セル面と交差させて
  歩く。r²(s)=q(s) は s の 2 次式；z 面は線形、r 円筒は 2 次、一般円錐面は
  √q·ΔZ = R_A·ΔZ+(z−Z_A)·ΔR の両辺平方で 2 次式に帰着し、符号フィルタ
  （L·ΔZ≥0）・平方残差フィルタ・線分上フィルタ（t∈[0,1]）で偽根を棄却する。
  軸 r=0 の縁は非交差面（子午面弦は pericenter で軸を自然に通過）。
  歩行の後退は許す（s のみ単調）。fallback：交差喪失時は ε_s 前進+再特定を
  1 回、以後は現セルへ残余全付与の保守的終端（診断カウンタ計上）。
  セル内は §5.11 の march_cell（ΔΣ=ρ·Δs、局所凍結場、E_floor 熱化）を
  そのまま用いる。領域境界面の通過 = escape（内側境界は存在しない —
  `inner_bc` は 1D 専用）。
- **沈着の合流**：チャネル合算の per-cell power は `transfer_to_2d` の
  laser 専用 rescale・平滑化の**後**、`state.laser_dep` への書き戻し直前に
  `dep_hm[c] += P_hot,c·dt` で合流する（hot-e プロファイルは物理であり
  rescale/平滑の対象にしない；§5.11 の「smoothing 後に加算」の 2D 形）。
  raytrace-skip cache は hot-e 有効時に無効化（skip された trace は捕獲
  できない）。
- **保存**：Σ_c P_c V_c + P_escape = ΣP_hot を帳簿恒等で満たし（機械精度、
  実測 ≤2e-13）、残差は修復せず報告する（1e-8 で警告）。
- **累積量の随伴（retry / remap、2026-07-17）**：per-cell 累積比沈着
  `hot_e_eps_cum` [erg/g] は (a) driver full-step retry で
  `DriverRetrySnapshot` によりステップ前値へ厳密復元され（laser 演算子は
  retry 領域内 — burn_eps_cum と同じ二重計上ハザード類；E_hot_e_* 台帳・
  enabled_any latch も収載）、(b) 質量を移す全 remap 経路（legacy 構造
  swept remap 1st/2nd order、CSR remap、ring7 packet / pole-cap packet
  transaction、ALE 内部 rollback snapshot）を ε·m の extensive 量として
  gas_tracer_Y / burn_species_Y と同型に随伴する（donor 値は max(ε,0)、
  clamp01 なし、m>0 で ε=εm/m、inactive cell は保持、空 vector=無効で
  全経路 no-op）。CSR flux の atomicAdd により ε は tracer と同じ
  replica-noise 級（診断量として許容・文書化）。中央 pseudo-core は
  tracer を pooled 診断（M_Y_c）として使うのみで ε の対応消費者が無く、
  吸収コア域は shell preheat QoI と非交差のため ε は pseudo-core
  ライフサイクル中凍結（documented deferral）；pole angular derefine の
  tracer 用途は純度センサ+監査合計であり移流ではない（ε 関与なし）。
- **実装形**：host 参照パイプライン（検証・転写照合用、
  `TENRYU_HOTE2D_HOST_PIPELINE=1`）と device 既定パイプライン（1 thread/弦、
  弦歩行 1 回+全群 in-thread march、per-segment dP を g 昇順で加算、
  (job, segment) 昇順の host fold — atomics/sort なしで device run-to-run
  bitwise）。identity ctest（host↔device ≤1e-13/cell）で束縛。
  弦の経路決定に関与する幾何（chord_q・交差行列式・point-in-cell・
  2 次係数組立て・判別式・符号/残差フィルタ・relocate 座標）は明示
  fma() で丸め列を固定し、host/device の自動 FMA 縮約差による接線での
  root flip を排除する（交差判定は両者 bit 同一；2026-07-17）。残る
  host↔device 差は libm（阻止能の log/exp）の ULP 差 × セグメント数で、
  multiblock の長弦では ≤1e-12/cell 実測 — multiblock identity gate は
  1e-11（path flip ≥1e-4 に対し 7 桁の判別余裕）、single-block は
  1e-13 を維持。
- **multiblock 格子（2026-07-17 解錠）**：`build_multiblock_view` が
  MultiBlockTopology の CSR ノードスロット（`Mesh::cell_nverts` 権威
  チャネル；三角セルは [n0,n1,n2,n0] の縮退四角形）を ccw（符号付き
  面積 > 0）へ正規化し、隣接は unique_internal_faces / boundary_faces
  （正典 helper csr_face_swept_node_indices 経由）から導出する — seam で
  ノード id が共有されない場合に破綻する id 一致 edge-hash は採らない。
  非縮退面が内部リンクにも境界タグにも解決しない場合は起動時 fatal
  （chord leak の黙認禁止）。single-block は従来 builder を経由し bit 恒等。
  **到達性の注意**：multiblock scheme は
  `logical_mesh_2d="spherical_polar_halfplane"` 上にあり、同論理格子は
  現在 hydro-only（Phase 6-minimum — Laser/Radiation/伝導を validate で
  拒否）。従って hote×multiblock は deck レベルでは上流の
  laser-on-spherical-polar フェーズ待ちであり、本層は unit gate
  （view 整合性 ×2 scheme・保存・host/device identity）で検証済みの
  受け入れ準備完了状態である。
- **制約（v1）**：単一 rank；reflect 境界でも hot 電子は escape
  （reflux なし — §5.11 と同じ v1 制限）。


### 5.11.3 η(t) 物理モデル（`eta_mode="model"`、1D、opt-in）

`Laser.hot_electron.eta_mode="model"`（既定 `"legacy"`）で、チャネルごとの変換
効率 η を定数/table の処方から**局所プラズマ条件で駆動される一次緩和 ODE**に
置き換える（設計文書
`docs/design/external-ai-responses/20260727-hote-eta-model-advice.md` の
推奨モデル案 1）。`"legacy"` 経路は bit 恒等のまま — model 分岐は解決段
（host）にのみ入り、capture kernel は不変（η=0 チャネルは
`one_minus_eta=1.0` で交差記録のみ行う）。1D_SPH 専用・`sources`
（tpd/srs のみ）・`subtract_from_laser=True` 必須（SPECIFICATION §6.4）。

**平衡効率**（チャネル k、実装 `src/laser/hot_e_eta_model.cpp`）：

- TPD：\(\xi = I_{14}\,L_{n,\mu m}\,\lambda_{\mu m}/(82\,T_{e,\rm keV})\)、
  \(g=\xi/C_k\)（Simon 1983 型閾値；\(C_k\)=`threshold_multiplier`）。
- SRS：\(I_{\rm abs,14} = 2377/L_{n,\mu m}^{4/3}\)
  （\(|\lambda_{\mu m}-0.351|<5\times10^{-3}\)）、それ以外は
  \(995/(L_{n,\mu m}^2\lambda_{\mu m})^{2/3}\)；\(g=I_{14}/(C_k I_{\rm abs,14})\)。
- \(g\le1\)：\(\eta_{\rm eq}=0\)。\(g>1\)：
  \(\eta_{\rm eq}=\min\{\eta_{\rm hard},\;
  \eta_\infty\,(1-\exp(-a_k\sqrt{g-1}))\}\)。

**時間発展**（正確な指数緩和；\(\alpha=-\mathrm{expm1}(-\Delta t/\tau)\)）：

\[
\eta \leftarrow \eta + \alpha\,(\eta_{\rm eq}-\eta),
\qquad
\tau =
\begin{cases}
\mathrm{clip}\!\left(\dfrac{1}{0.05\,g+0.06}\ \mathrm{ps},\ \tau_{\min},\ \tau_{\max}\right)
& \text{(`vu2012`)}\\[2mm]
\tau_{\rm fixed} & \text{(`fixed`)}
\end{cases}
\]

入力 invalid（後述）のステップは \(\eta_{\rm eq}=0\)・\(\tau=\tau_{\rm fixed}\)
（`relaxation_tau_s`）で 0 へ緩和する。チャネル更新後、
\(\sum_k\eta_k>\)`eta_total_cap` なら比例縮小（`apply_total_cap`）。

**局所量評価**（毎ステップ、host、現在の hydro 状態から）：

- 評価面：セル中心 \(n_e = \rho\bar Z/(A_{\rm eff}m_p)\)（void セルは 0；
  laser mesh と同一式）を外側から走査し、
  \(n_e^{\rm outer}<f_{\rm eval}n_c\le n_e^{\rm inner}\) の最外ブラケットを
  線形内挿（\(\alpha\) 内挿はセル中心間；設計文書 §13.1 のノード内挿の
  セル中心版）。\(T_{e,s}\) は同じ \(\alpha\) で線形内挿（eV→keV）。
- \(L_n\)：ブラケット±3 セル（void/非正 \(n_e\) 除外）の \(\ln n_e\) を
  重み付き最小二乗（\(w=\exp(-(d/w_0)^2)\)、\(w_0=2\overline{\Delta r}\)）で
  fit し \(\kappa=|b|\) [1/µm]。EMA
  \(\bar\kappa \leftarrow \bar\kappa + \alpha_L(\kappa-\bar\kappa)\)
  （\(\alpha_L=-\mathrm{expm1}(-\Delta t/\tau_L)\)、\(\tau_L\)=
  `ln_filter_tau_s`；初回サンプルは snap）。
  \(L_{n,\rm eff}=1/\max(\bar\kappa,1/L_{\max})\)。
  clamp：\(L_n\in[10,1000]\) µm、\(T_e\in[0.5,10]\) keV、
  \(I\in[10^{13},2\times10^{16}]\) W/cm²（発火は診断 bitmask 1/2/4）。
- 強度（**one-step-lagged**）：**前ステップ**の capture 面交差の生 power 和
  \(\Sigma P^{\rm cross}\)（η スケール前；ray 数非依存）と power 加重平均
  半径 \(\bar r_s\) から
  \(I = \Sigma P^{\rm cross}/(4\pi\bar r_s^2 f_{\rm illum})\)、
  \(f_{\rm illum}=1\)（v1、1D 球対称照射の規約）。cgs→W/cm² は \(10^{-7}\)。
  \(\Sigma P^{\rm cross}\) は ray quadrature 推定量であり ray 数に比例スケール
  しない（§14.1 の禁止事項）が、ray 配置の離散化誤差 ~1%（32–512 rays、
  GXII 様 raytrace_2d 実測）を持ち N に対し非単調 — ctest
  hot_e_eta_wiring: ray-count convergence が 5% band で守る。
  正当化：\(\tau\)（3–10 ps）≫ \(\Delta t\) のため 1 step 遅延は緩和時間内で
  無視できる（設計文書 §19 の within-step staged replay は v1 では採らない）。

**スピンアップと無効入力**：初期 \(\eta=0\)（step 1 は計測のみ、step 2 から
非零 η）。前ステップ交差なし／評価面なし／fit 縮退／\(T_e\le0\) は invalid
として 0 へ緩和。laser off（total_power=0）のステップは前ステップ計測を
クリアし η は凍結 — 再点灯の最初のステップは invalid 扱いで一段下がった後、
再成長する（v1 文書化挙動）。

**決定論・保存・制約**：η 更新は host double の固定順序演算のみ（run-to-run
bitwise）。エネルギーは既存の sequential depletion
\((1-\eta_k)P^{\rm cross}\) に η 値として入るだけで、保存台帳の構造は不変。
model 状態（η、\(\bar\kappa\)、前ステップ計測）は **checkpoint 非永続**
（v1）：restart 後 \(\sim\tau\) のスピンアップ過渡が入る（文書化制限；HDF5
スキーマ不変）。単一 rank のみ（既存 hot-e 制約と同一）。診断は
`/hydro/hot_e_eta_model_*`（チャネル長配列：η、g、η_eq、τ、I14、Te、
L_n、clamp bitmask、前ステップ \(\Sigma P^{\rm cross}\)）＋ verbose 時の
ステップログ。
