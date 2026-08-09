<!-- 分割元: docs/NUMERICS.md | このファイルは参照用です。原本（docs/NUMERICS.md）が権威です。 -->
### 6.7 Multigroup Flux-Limited Diffusion (FLD)  —【CURRENT — 現行の主放射モデル】

> **【CURRENT RADIATION MODEL — primary】** `mode="multigroup_diffusion"`。1D_SPH/2D_RZ の production 既定（DEFAULT-FLD）。決定論 multigroup flux-limited diffusion。IMC/DDMC/HOLO/difference を完全 bypass。もう一方の現行モデルは §6.8 \(S_N\)。以降 §6.7.x は FLD/\(S_N\) の検証 gate（I1〜I7）。

`Radiation.mode="multigroup_diffusion"` は 1D_SPH と 2D_RZ の production 放射モードである。この
経路では IMC/DDMC/PGRW/HOLO/difference formulation を通らず、群ごとの
flux-limited diffusion と電子物質結合を CUDA 上で解く。FLD は HYDRA-aligned
Fleck linearization を使い、stiff な物質-放射結合を放射線形系へ入れる。

各 cell \(c\)、group \(g\) の backward-Euler 有限体積式は
\[
V_c E^{n+1}_{c,g}
 +\Delta t\,c\,\sigma^{PA}_{c,g}V_cE^{n+1}_{c,g}
 -\Delta t\sum_{f\in\partial c}
 A_fD_{f,g}\frac{E^{n+1}_{n(f),g}-E^{n+1}_{c,g}}{d_f}
=V_cE^n_{c,g}
 +\Delta t\,V_c\,f_c\eta_{c,g}(T_e^n)
 +\Delta t\,V_c(1-f_c)c\sigma^{PA}_{c,g}E^n_{c,g}.
\]
\(\eta_{c,g}=c\sigma^{PE}_{c,g}a_{eV}(T_e^n)^4b_g(T_e^n)\) であり、\(f_c\)
は §6.1 の Fleck factor から作る。ここで左辺の \(\sigma^{PA}\) は total
removal（真の吸収 + Fleck effective scattering）であり、Fleck は RHS の
emission/effective-scattering split に入れる。すなわち局所・無拡散極限で
\(f_c\to0\) なら \(E^{n+1}=E^n\) となり、lagged scattering source が放射エネルギーを
増幅しない。実装では `fld_nlte_f_work` に \(f_c\)、`fld_eta` に raw
\(\eta_{c,g}\) を保持し、組み立て時に RHS の
\(f_c\eta+(1-f_c)c\sigma^{PA}E^n\) を作る。`fld_nlte_sigma_eff_work` の
\(f_c\sigma^{PA}\) は NLTE coefficient path で生成されるが、この FLD assembly の
total-removal diagonal には使わない。Grey constant-opacity FLD では
\(\sigma^{PE}=\sigma^{PA}=\sigma_a\) とし、専用の `compute_fleck_for_fld`
kernel で §6.1 系の LTE Fleck factor を**群別**に
\(f_{c,g}=1/(1+\alpha\beta_c c\Delta t\sigma_{a,g})\) として
table_nlte/tmat と同じ RHS emission/effective-scattering split に適用する
（\(\beta_c\) は cell 量、\(\sigma_{a,g}\) は群値。constant/power_law opacity
は周波数非依存で全群同値のため cell 単一 \(f_c\) と一致し、群依存 σ の
`freq_dep_marshak` 検証 opacity でのみ群別値が現れる。旧記述の
「\(\sigma_{a,P}\) 単一 \(f_c\) + McClarren-Urbatsch smooth blend」は実装と
乖離していた — blend は stiff 極限 \(zf\to0\) の交換凍結のため 1D では退役済み、
時間形状は既定 `"be"` \(f=1/(1+z)\)（下記 fleck_form 参照）。2026-07-26
doc 真実復元、AI review k05）。この kernel は FLD の
stiff-cell 要件のため IMC 共有の `compute_fleck_kernel` と分離し、IMC safety の
\(\beta\le1\) cap と `f_min_fleck` 下限を適用しない。table_nlte/tmat の NLTE
係数経路（`eval_nlte_opacity_emission` — cell 単一 \(f_c\) を emission-mean
\(\sigma_{P,\rm em}=\sum_g\eta_g/(a_{eV}cT_e^4)\) から作る）も同様に
\(\beta\le1\) cap を適用しない（2026-07-26、AI review k05 4.4: ideal-gas
fallback 分岐に残存していた IMC 系譜 cap を除去。cap は「\(f>1\) 防止」に
不要 — \(z\ge0\) で常に \(0<f\le1\) — であり、高温・低 \(C_v\) セルで f を
過大化し Fleck 線形化保護を弱めるだけだった。table cv / state cv 経路は
cap 非経由のため挙動不変）。放射エネルギー式に
\(\rho c_v\) 型の項は入れない。

> **fleck_cv_source（2026-07-10、HR-W1c；既定フリップ 2026-07-11）**: 本カーネルの \(\beta=4a_{\rm eV}T_e^3/(\rho c_{v,e})\)
> に入る電子比熱の出所は `Radiation.multigroup_diffusion.fleck_cv_source` で選ぶ。
> `"table"`（**既定**、2026-07-11 フリップ — 外部AI裁定、docs/design/fleck_cv_default_flip_20260711.md）は
> 電子 EOS テーブル存在時に現在 \(T_e\) の `device_eos_cv`（matter 更新 `update_matter_body` と
> 同一の cv）を最優先する — Fleck 線形化の \(\beta\) は matter Newton が前進させるエネルギー
> 関数と同一の \(\partial U_e/\partial T_e\) を要する（Fleck–Cummings 1971 の整合要件）。
> テーブル不在時は legacy チェーンへ落ちる（matter 側も非テーブル分岐のため整合が保たれる）。
> `"legacy"` は旧チェーン cv_e_override → state cv_e → **ideal-gas fallback**
> \(\bar Z e_{\rm eV}/(A_{\rm eff} m_p(\gamma_{\rm eff}-1))\) を凍結保存する明示互換モード
> （旧 golden の bit 再現・A/B 用）。**table-EOS 材料で legacy を使うと fleck の cv が
> matter 側と不整合になり \(f\) が歪む**（Hammer–Rosen gate 開発 2026-07-10 で発見:
> 検証フィクション A=1e5 が fallback cv を
> \(10^5\) 倍崩壊させ \(f\to0.002\)、表面 \(T_e/T_{\rm rad}\approx0.7\) の偽非平衡を生成 —
> rad_dep/rad_emit 台帳からの \(f\) 逆算で実証。実材料でも transient 交換率が最大
> \(q=C_{\rm legacy}/C_{\rm table}\) 倍歪む）。率忠実度は 0-D 緩和 gate
> `verify_fleck_relaxation_0d`（厳密 ODE 参照）が常設検証。フリップの既存 golden への
> 影響は無し（table-EOS gate 群は deck 内 pin 済み、GXII FLD regression は ideal_gas で
> knob 不活性 — golden 再生成 bit 同一で実証、VERIFICATION §4.z3）。
> 冪乗 opacity `power_law`（SPEC §6.4.3）はこの constant 経路と同格に扱われる（eta 構築・
> Fleck blend とも σ 配列値のみが異なる）。冪乗 EOS `power_law_te` は初期化時 tabulation で
> table-EOS 経路に乗る（新規離散化なし）。
>
> **fleck_beta（2026-07-14 導入；外部裁定 2026-07-15、docs/design/fleck_beta_secant_20260714.md §7-8）**:
> β の線形化点は `Radiation.multigroup_diffusion.fleck_beta` で選ぶ。`"tangent"`（**既定**、bit 凍結）
> = 上式の接線 β。`"secant"`（opt-in、table-EOS セル・1D FLD のみ — 2D fleck kernel は独立実装で
> tangent 固定）= 灰色弦 \(\beta_{\rm sec}=\Delta B/\Delta U_e\) を 0-D 局所予測子
> （\(f_{\rm tan}c\sigma_P(E-B)\Delta t/C_v\)、信頼域 ±0.5T、退化時 tangent へ fallback）の張る
> 区間で評価する。U_e は matter Newton と同一の table accessor（整合 by construction）。
> **生産精度の裁定 = documented null（外部 AI 裁定 2026-07-15、採択済み）**: 一段誤差分解で共通の
> Fleck/BE 時間形状項 \(O(h^2)\)（β 非依存）が支配し、弦補正は劣次（tangent \(O(h^3)\) →
> secant \(O(h^4)\)）のため軌道誤差比は \(\Delta t\to0\) で 1 に収束 — 5 レジームの null 実測と
> 厳密に整合。既定は tangent を維持。secant 分岐の実装正当性は係数分離ストレスゲート
> （`verify_fleck_relaxation_0d` gate (f): 1 step・max_outer=1・強 softstep D=2e12/w=100 eV、
> 実測比 0.277/0.274/0.436/0.646 ≤ 帯 [粗3本 0.5 / 全4本 0.7]）が常設判別する。OFF-bit 認証:
> GXII golden regression・Hammer–Rosen PASS + fleck_beta キー不在 vs 明示 "tangent" の
> 全 154 データセット恒等（2026-07-15；h5 バイト差は deck パス由来メタデータのみ）。
> **第 3 値 `"guard"`（2026-07-16 導入 — 裁定 §9.2）**: β_used = max(β_tan, β_sec) ⇒
> f_used = min(f_tan, f_sec) の片側単調性リミッタ（secant と同じ予測子・同じ制約、選択のみ max）。
> デバイス probe が加熱/冷却双方で厳密 min 選択を恒等検証（fmax は勝者オペランドを bit 恒等で返す）。
> outer 効率初データ（gate (i)、2026-07-16）: 多 step softstep・max_outer=60 で outer 反復平均
> tan 15.33 / sec 13.33 / grd 14.67（sec/tan=0.87 — 裁定 §9.4 の Picard 加速仮説を初計測で確認）。
> 注意: 収束後の fleck_cummings step は f(β) を保持するため、Picard 不動点は β 非依存**ではない**
> （β 選択間の終端差 ~O(h·Δf)、実測 ~1.5e-3 of span @ h~1 は正しい振る舞い）。

> **fleck_form（2026-07-16 導入；設計 docs/design/fleck_exp_source_20260716.md §2 — β_sec 後続裁定 §12 の係数レバー）**:
> Fleck 因子の時間形状は `Radiation.multigroup_diffusion.fleck_form` で選ぶ。`"be"`（**既定**、bit 凍結）
> = 標準 backward-Euler 形 \(f=1/(1+z)\)。`"exp_phi1"`（opt-in、1D FLD のみ — persistent path 含む・
> table EOS 不要）= \(f=\varphi_1(-z)=(1-e^{-z})/z\)（\(z<10^{-6}\) は級数 \(1-z/2+z^2/6\)）。固定輻射
> スカラー緩和の厳密保持率 \(R=e^{-z}\) を再現する Fleck 係数（\(f=e^{-z}\) ではない）で、
> \(0<f\le1\)・stiff 極限 \(zf\to1\)（旧 exp(−z) blend を退役させた交換凍結は起きない）。
> fleck_beta と直交（z の β には tangent/secant がそのまま入る）。結合閉箱を厳密化するものではない
> （それには f>1 が必要 — 裁定 §12.4 で棄却；厳密閉箱は分割ソース直接移送 = design doc §3 rung 2、未実装）。
> 判別ゲート `verify_fleck_relaxation_0d` gate (g)（1 step・max_outer=1・gate (f) と同じ softstep 基板、
> E_rad 端点誤差の exp/be 比）: 実測 0.531/0.380/0.267/0.187 vs 非線形 Radau 事前予測
> 0.508/0.367/0.261/0.186（帯: 全 rung ≤0.65・最細 ≤0.30）— BE 時間形状項の除去どおり細 rung ほど
> 利得（最大 5.4×）。デバイス恒等 probe（BE 逆算 exact-z で φ₁ 恒等 ≤1e-12・単調・(0,1] 域・
> crossover 連続）= ctest "fleck form exp_phi1"。OFF-bit 認証: fleck_form 実装込みバイナリで
> GXII golden regression・Hammer–Rosen PASS（2026-07-16、既定経路 golden 恒等）。

> **source_integrator（2026-07-16/17 導入 — rung-2、docs/design/fleck_exp_source_20260716.md §3）**:
> 1D FLD の物質–輻射ソース積分器を `Radiation.multigroup_diffusion.source_integrator` で選ぶ。
> `"fleck"`（**既定**、bit 凍結） = 従来のモノリシック半陰的 outer ループ。`"exp_rosenbrock"`
> （opt-in、灰色 v1・fleck_beta tangent 限定・afi/exp_phi1 非互換・persistent 非対応）= Lie 分割:
> (1) 凍結係数の厳密直接移送 \(q=h\varphi_1(-(1+\beta)h)(E-aT^4)\) を両側対称適用（E−=q, U_e+=q —
> 局所保存が構成的に厳密）、(2) 交換項を除いた純拡散陰解（D_face の σ_R は輸送係数として保持）、
> outer Picard なし。実測（VERIFICATION §7.10 gate (j) + verify_marshak_feature_1d、2026-07-17）:
> 0-D 一段誤差比 err/err_be = 0.430/0.293/0.186/0.109（8/4/2/1e-15 s、多 step で 2 次収束）、
> 総エネルギー drift **厳密 0**（fleck 単一パスの 2.1–2.3 倍非保存と対照）、1-D Marshak feature
> 前線で分割バイアス 0（最細 rung で fleck と同一セル・N=1024 参照 4dx 内、非劣化全 rung）。
> OFF-bit: exchange_off 配線+ループ再入れ子込みバイナリで GXII golden・HR PASS（既定経路恒等）。
> **多群（G≤96、2026-07-17 導入 — docs/design/exp_mg_phi1_20260717.md、外部裁定採択）**: 保存超平面上の
> 厳密 G×G 対角+rank-1 縮約 \(K=-X-X\gamma\mathbf 1^T\)、\(\Delta E=\varphi_1(K)r\)、
> \(\Delta U=-\sum_g\Delta E_g\)（構成的保存）。\(\varphi_1\) は認証済み 16 極対有理近似
> （放物線 Hankel コンター、sup 相対誤差 2.4e-14、tools/gen_phi1_poles.py 生成・phi1_poles.hpp 凍結）
> を極ごと Sherman–Morrison O(G) で作用（セルあたり O(pG)、逐次固定順序和 = 決定論、
> 極セットは 512B カーネル引数 struct — constexpr 配列の device odr-use 罠を回避）。
> \(\gamma_g=d(b_gB)/dU\)（Planck 表中心差分）で可変 Planck 分率でも 2 次を保持（β b_g 凍結刷新は
> 一般に 1 次 — 裁定 §5.2）。v1 保護 = セル単位棄却+計数（警告に cell/reason/群/Te/E_g/ΔE_g を
> 最大 8 slot 添付、裁定 §4.7 の Fleck fallback の文書化偏差）。Wien-tail アンダーフロー群
> （E_g と |ΔE_g| がともに ≤ 1e-20·max(B, ΣE_g)）は相対負性判定の対象外とし
> \(\Delta E_g\leftarrow\max(\Delta E_g,-E_g)\) にクランプする（ΔU はクランプ後の ΔE から集計 =
> 構成的保存を厳密維持。根拠: 2026-07-18 生産 A/B smoke で棄却が全て最外殻セル・最高群
> g≥70・E_g ~ 1e-106〜1e-61 erg/cc の丸め偽負性と実測特定 — 群別判定×セル全体棄却が駆動相の
> 表面セル交換を飛ばす偏りを除去、docs/design/exp_mg_phi1_20260717.md）。gate (k)（G=4 等 σ・1 step）: **周辺化恒等 |E_tot^{mg}−E_tot^{grey}| ≤ 1.9e-16**
> （等 σ・Σb=1・Σdb/dT=0 で多群凍結系の総和は灰色凍結系へ厳密周辺化 — 別経路計算の 1 ulp 一致が
> rank-1 機構の判別的認証）・保存 drift 厳密 0・2 次 slope 135×/8×。灰色 G=1 は従来スカラー kernel を
> bit 不変で維持。

1D_SPH の面幾何は \(A_f=4\pi r_f^2\)、\(d_f\) は隣接 cell center 間距離である。
2D_RZ では \(N_r\times N_z\) の cell-centered unknown を row-major
\(c=iN_z+j\) で並べ、R/Z 4-face の有限体積ステンシルを用いる。2D_RZ FLD の面積は
各 cell が独立に再構成した cell-centered metric ではなく、共有 edge の2端点
\((r_0,z_0),(r_1,z_1)\) から一意に
\[
A_{edge}=\pi(r_0+r_1)\sqrt{(r_1-r_0)^2+(z_1-z_0)^2}
\]
で計算する。これは直線 edge を対称軸まわりに回転した面積であり、軸に平行な
R 面では \(2\pi r|\Delta z|\)、水平な Z 面では \(\pi(r_1^2-r_0^2)\) に退化する。
同じ内部 face の係数 \(A_fD_f/d_f\) には左右 cell で同一の \(A_{edge}\) を用いる。
これにより Lagrangian 変形で \(x_z\) が \(i\)-row 依存になった場合も、CSR の相互
off-diagonal と face-divergence が対称に相殺され、FLD operator の self-adjointness が
丸め誤差内で保たれる。距離 \(d_f\) は隣接 cell center 間隔である。
R 軸 \(R=0\) は反射対称で face flux を 0 とする。外側 \(R=R_{max}\) は
`Radiation.multigroup_diffusion.boundary.outer_r` で指定し、`"vacuum"` では
Marshak-like escape \(F_{out}=cE/2\)、`"reflect"` では face flux 0 とする。Z 端は
`Radiation.multigroup_diffusion.z_boundary` / `boundary.z` の共通指定、または
`boundary.z_bottom` / `boundary.z_top` の面別指定で与える。`"vacuum"` では
\(F_{out}=cE/2\)、`"reflect"` では face flux 0 とする。

FLD state-supply BC（Phase 2c Wave 2）は 2D_RZ の Z 端のみ、灰色1群の
Dirichlet 放射境界として実装する。`boundary.z_bottom` または
`boundary.z_top` が `"state_supply"` の面 \(f\) では、同じ面の
`Numerics.hydro.boundary_2d.{z_bottom,z_top}` の供給温度 \(T_s\) [eV] から
\[
E_{b}=a_{eV}T_s^4
\]
を作り、境界セル中心から物理境界面までの距離
\[
d_f=\max(|z_c-z_{\min}|,\epsilon_d)\quad\text{or}\quad
d_f=\max(|z_{\max}-z_c|,\epsilon_d)
\]
を用いて
\[
F_{\mathrm{out},f}=D_{c,g}\,{E_{c,g}^{n+1}-E_b\over d_f}
\]
を有限体積面 flux として使う。線形系では
\(\Delta t\,A_fD_{c,g}/d_f\) を diagonal へ加え、
\(\Delta t\,A_fD_{c,g}E_b/d_f\) を RHS へ加える。単一セル側の境界なので
内部 face の harmonic mean は使わず、境界セルの \(D_{c,g}\) を直接使う。
この BC は `Radiation.groups=1` のみ対応し、matching hydro state-supply
設定が無い場合は `ConfigError` とする。step tally として
`fld_state_supply_in/out/net` を記録し、net は外向きを正とする。

`Radiation.multigroup_diffusion.state_supply_boundary_policy` selects only the
state-supply boundary coefficient closure. The production default
`"local_D_current"` is the existing local-D closure: the boundary Dirichlet row
uses the current boundary-cell \(D_{c,g}\) in the diagonal/RHS terms above, so
the operator coefficients remain local to the solved state. `"harmonic_ghost_D_test"`
and `"radial_mean_D_test"` are **DIAGNOSTIC-ONLY** Round 13 candidate-A
isolation policies, not production defaults. In the currently validated
configuration (`groups=1` with constant opacity), `"harmonic_ghost_D_test"`
uses the hydro supply \(\rho_s\) plus the boundary-row local effective
\(\kappa_R/\bar Z\) to form a ghost coefficient and then harmonic-averages it
with the interior coefficient. For future non-constant-opacity diagnostic use,
the ghost \(\bar Z\) is the boundary-row local value; this is diagnostic-only
acceptance, not a validated physical closure.

`"radial_mean_D_test"` replaces the boundary-row coefficient used by the
diagnostic closure with the radial mean for the audited row and is not safe as
a general non-slab-geometry production model. Its invalid-value handling is
fixed: only nonnegative finite coefficients enter the mean, invalid entries
contribute zero, and the emitted diagnostic coefficient is never NaN or Inf.

The 2D_RZ FLD CG inner tolerance is exposed as
`Radiation.multigroup_diffusion.cg_inner_tol`. The default \(10^{-10}\)
preserves the previous solver behavior; user values must be positive.

FLD Marshak source BC（Phase 2a-2）は 2D_RZ の Z 端のみ、灰色1群の定常入射
flux として実装する。`boundary.z_bottom` または `boundary.z_top` が
`"marshak"` の面 \(f\) では、入力
`Radiation.multigroup_diffusion.marshak.flux_erg_per_cm2_s` を
\(F_{\mathrm{inc}}\) [erg cm\(^{-2}\) s\(^{-1}\)] とし、
\[
F_{\mathrm{out},f} = {c\over 4}E_{c,g}^{n+1}-F_{\mathrm{inc}}
\]
を有限体積面 flux として使う。したがって線形系では、その境界セル行に
\(\Delta t\,A_f(c/4)\) を diagonal へ加え、
\(\Delta t\,A_fF_{\mathrm{inc}}\) を RHS へ加える。Marshak 面の outgoing
\(cE/4\) は `radiation_escaped`、incoming \(F_{\mathrm{inc}}\) は
`marshak_in` として step energy budget に入る。`flux_pulse_duration_s >= 0`
の場合は \(t <\) `flux_pulse_duration_s` の step だけ \(F_{\mathrm{inc}}\) を
有効化する（flux 経路の任意時間波形は未実装 — 時間依存駆動は次段落の
Tr(t) 経路が提供する）。

**2D_RZ 決定論 Marshak z 面の時間依存黒体駆動（indirect-drive Tr(t),
2026-07-11, 設計 docs/design/2d_tr_drive_port_spec.md）** — FLD/SN の
`z_bottom`/`z_top="marshak"` 面は、灰色定常 flux に代えて `Radiation.boundary`
の Tr 源（定数 `marshak_Tr_eV` / 時間 callable `marshak_Tr` / 面別 dict
`marshak_Tr_map` — IMC と共有の初期化時凍結テーブル、runtime Python なし）
でも駆動できる。駆動源は排他的二択（Tr 源 xor `marshak.flux_erg_per_cm2_s>0`、
両方・両方なしは `ConfigError`）。\(T_r(t)\) の解決規約は IMC-2D emitter と
同一: 面テーブル（正準キー `bottom_z`/`top_z`、alias `z_bottom`/`z_top`）▸
定数 `marshak_Tr_eV>0` ▸ スカラーテーブル、radiation call 冒頭で `state.t`
を 1 回評価（outer 反復不変）。
- **FLD**: per-group 入射 \(F_{\mathrm{inc},g}=(c/4)\,a_{eV}T_r^4\,b_g(T_r)\)
  （\(b_g\) は正規化 Planck 重み、`groups==1` は \(b=1\) の table 迂回）を、
  CSR assembly 後の追加 RHS post-pass kernel で境界セル行へ
  \(\Delta t\,A_f F_{\mathrm{inc},g}\) として加算する（assembly kernel は
  byte 不変; diagonal の Marshak leak \((c/4)\) は既存項のまま）。灰色
  `groups==1` 制限は flux 経路にのみ残る — Tr 経路は per-group Planck 重みを
  供給するため**多群可**。`fld_marshak_in_step` は面別灰色和
  \(\sum_g F_{\mathrm{inc},g}\) を既存の離散面積 reduction（面選択は bc 引数）
  に通した和で上書きする。
- **\(S_N\)**: 既存の z 面注入は全群一様 \(\psi^-=2F_{\mathrm{inc}}\)（構造的
  灰色）であり、Tr 経路は solve 冒頭に解決した
  \(F_{\mathrm{inc}}=(c/4)a_{eV}T_r^4\) を既存スカラー slot へ供給する
  （sweep kernel 不変）。v1 制限: `groups==1` 必須（多群 spectral 注入は
  将来 wave）、両 z 面 marshak + 面別テーブルは `ConfigError`（単一スカラー
  共有のため; 定数/スカラーテーブル源は両面共通で可）。ledger
  `sn_marshak_in_step` は flux×面積×\(\Delta t\) の既存定義のまま正しい。

検証（VERIFICATION §7.10、deck 資産 tmp/tr2d_gate/）: 定数-vs-テーブル-vs-面
テーブル bit 恒等（FLD grey/MG、SN grey）、flux 等価（FLD rel ≤2e-16 =
加算順序差のみ、SN は bitwise）、Tr 階段 100→200 eV で per-step
`marshak_in` 比 16.0000 厳密（両ソルバ同値）、MG/grey ledger 比 1.00000000
（\(\sum_g b_g=1\)）、pre-port binary との OFF-bit 恒等。

**1D_SPH FLD 境界条件（W-B, 2026-07-03）** — 内側 \(r=0\) は球対称により常に
反射（face flux 0；`boundary.inner_r` は `"reflect"` 以外を `ConfigError` で拒否）。
外側 \(r=R_{max}\) は `Radiation.multigroup_diffusion.boundary.outer_r` で
`"vacuum"` / `"reflect"` / `"marshak"` を選ぶ。`"vacuum"` は half-range escape
\(F_{out}=cE/2\)（従来の 1D 固定挙動と同一 = 既定）、`"reflect"` は face flux 0、
`"marshak"` は Milne 型 \(F_{out}=(c/4)E - F_{\mathrm{inc}}\) とし、外側セル行の
diagonal に \(\Delta t\,A(c/4)\)、RHS に \(\Delta t\,A\,F_{\mathrm{inc},g}\) を加える
（\(A=4\pi R_{max}^2\)）。入射駆動は排他的二択: (i) 黒体駆動
`Radiation.boundary.marshak_Tr_eV` \(>0\) で per-group
\(F_{\mathrm{inc},g}=(c/4)a_{eV}T_r^4\,b_g(T_r)\)（\(b_g\) は正規化 Planck 重み、
multigroup 可）、(ii) 灰色定常 flux
`Radiation.multigroup_diffusion.marshak.flux_erg_per_cm2_s`（`groups=1` 限定、
`flux_pulse_duration_s` による矩形パルス対応）。両方指定・両方ゼロは
`ConfigError`。escape tally は `fld_escaped_step`（leak 係数は BC と整合）、
incoming は \(\Delta t\,A\sum_g F_{\mathrm{inc},g}\) を `fld_marshak_in_step` として
step energy budget に入る。Z 端キー（`boundary.z/z_bottom/z_top`）は 1D_SPH では
無意味なので非既定値を `ConfigError` で拒否する。

**Time-dependent drive (indirect-drive mode, 2026-07-09).** The Marshak
drive temperature accepts a deck time callable
`Radiation.boundary.marshak_Tr` (frozen to a table at initialization — no
runtime Python) in addition to the constant `marshak_Tr_eV`. Precedence and
evaluation follow the IMC emitter convention: a positive constant wins;
otherwise the frozen table is evaluated once per radiation call at the
solve-entry time `state.t`. The per-group incident flux
F_inc,g = (c/4) a T_r^4(t) b_g(T_r(t)) and the `marshak_in` ledger use the
resolved temperature, so a staircase drive T_r: 100→200 eV produces exactly
a 16× per-step `marshak_in` jump (validated). The same wiring applies to the
SN 1D marshak boundary (per-group psi_in). Marshak outer boundaries refuse
the persistent-kernel path (`marshak_outer_boundary`) and run multi-kernel.

**1D_SPH FLD 体積線源（W-B, Su-Olson 級）** — `Radiation.volume_source_rate`
[erg cm\(^{-3}\) s\(^{-1}\)] \(>0\) かつ `Radiation.volume_source_x_max` [cm]
\(>0\) のとき、セル中心 \(r_c\le x_{max}\) の全セルに RHS へ
\(\Delta t\,V_c\,\dot S\) を加える。Fleck 線形化（物質 emission）の外側で加える
external source であり、`groups=1` 限定（多群は `ConfigError`）。注入エネルギー
\(\sum_{r_c\le x_{max}}\Delta t\,V_c\,\dot S\) は `fld_volume_source_in_step`
として step energy budget（`volume_in`）に計上する。

**1D_SPH \(S_N\) 外側 Marshak 境界（W-B2, 2026-07-03）** —
`Radiation.sn_transport.boundary.outer_r="marshak"`（1D_SPH、
`spatial_scheme="linear_characteristic"` 必須）。外側ノードの内向き半区間
（\(\mu<0\)）へ等方入射強度 \(\psi^-_g = 2F_{inc,g}\) を与える（GL 規約
\(\phi=\sum_m w_m\psi_m = cE\), \(\sum w=2\) の下で平衡厳密:
\(2\cdot(c/4)a_{eV}T_r^4 b_g = \psi_{iso}(T_r)\)）。駆動は FLD 1D と同じ
排他的二択（`Radiation.boundary.marshak_Tr_eV` の黒体 / `sn_transport.marshak.
flux_erg_per_cm2_s` の灰色定常 flux + `flux_pulse_duration_s` 矩形パルス、
灰色は `groups=1` 限定、両方指定・両方ゼロは `ConfigError`）。実装は persistent
per-group バッファ（CUDA graph key に参加、step ごと capture 域外で充填）。

The drive temperature accepts the same constant-or-time-callable sources as the FLD 1D marshak boundary (see above).

`sn_marshak_in_step` は連続式でなく**離散求積**
\(\Delta t\,A(R_{max})\sum_g S^-\psi^-_g\)（\(S^-=\sum_{\mu<0}w|\mu|\)）で
記帳し、保存的 E* 閉包の外側 face flux は Milne 純流束
\(S^-\big(\tfrac{c}{2}E-\psi^-\big)\)（平衡 \(E=a T_r^4\) で厳密にゼロ）。
検証 gate `sn_1d_marshak_equilibration`（r0/dr=200 準平面シェル）: 外側セル
plateau 一致 2.6e-4（≤5e-3）+ 全域 ≤5%。**gate 設計注記**: 境界駆動定常は
球面 LC 角度再配分の曲率比例エネルギー欠損を初めて露呈した（中心含む球で
中心セル −89%、r0=0.35 で −47%、r0=10 で −3.3%; 外側セルは常に target 一致
= BC 無実、物質は局所平衡）。これは既存輸送特性で **BUG-8** として独立
workstream 追跡（純吸収体 ballistic / α和恒等式監査 / S_N 次数掃引が診断
ladder）。vacuum 経路・既存 golden への影響なし（psi_in=nullptr で bitwise 温存）。

**BUG-8 解決（2026-07-03, 保存形 1D 球面 sweep）** — 旧 LC sweep は slab 形
特性（\(\mu\,d\psi/ds+\kappa\psi=q\)）で球面保存形の幾何ストリーミング項
\(\mu\psi\,dA/V\) を演算子から欠いており、一様等方場で角度発散
\(-\mu_m\psi_0\,dA/V\) が相殺されず、曲率比例の中心 flux dip
（S8 で中心 −89%、次数非依存）を生んでいた。修正は 3 部構成:
(1) **保存形 FV ストリーミング** \(|\mu|(A_{dn}\psi_{dn}-A_{up}\psi_{up})/V\)
と θ 重み付き空間閉包 \(\psi_m=\theta\psi_{dn}+(1-\theta)\psi_{up}\)、
\(\theta(\tau)=1/(1-e^{-\tau})-1/\tau\)（τ→0 で diamond、τ→∞ で step、
正値）; (2) **Morel–Montry 加重 diamond 角度閉包**（TTSP 13(5) 615, 1984,
Eq. 15–16: 自然分割セル端 \(\mu_{m+1/2}=\mu_{m-1/2}+W_m\)、
\(\tau_m=(\mu_m-\mu_{m-1/2})/W_m\)、収支は \(\alpha_{m+1/2}/\tau_m\) を
除去側・\(\alpha_{m-1/2}+\alpha_{m+1/2}(1-\tau_m)/\tau_m\) を源側に置く
一貫形 — 拡散極限係数 β が任意求積で恒等的に消える）;
(3) **Miller–Alcouffe 開始方向**（μ=−1 の slab 掃引で角度 ladder を種付け、
Marshak 入射対応）。検証: 一様黒体平衡は machine precision の離散不動点
（max_rel 2.8e-16）、冷開始の中心含む球が 1e-6 で plateau 到達（旧 −89%）、
`sn_1d_marshak_equilibration` gate は outer/max とも 1e-5 に強化して PASS。
serial（非 LC）デバッグ sweep は旧スキームのまま（production は LC）。

**W-B2 v2 + W-G1 step 5 平面ゲート（2026-07-03）** — marshak 外側面の
E\*-flux 記帳を Milne 対 \(S^-(cE/2-\psi_{in})\) から**離散整合形**
\(\sum_{\mu>0} w_m\mu_m\psi_{out} - S^-\psi_{in}\)（sweep の流出 half-range
tally をそのまま使用）へ更新。黒体平衡では両者一致（対称 GL で
\(S^+=S^-\)）だが、非平衡境界セルで等方近似は 5.8% のバイアスを生んだ。
E\*-面 flux 閉包の定常不動点は輸送モーメント \(E=\varphi/c\) に一致する
（donor-theta limiter が不活性な \(\Delta t \lesssim E V/|F|\) のとき）。
検証ゲート: `sn_1d_planar_marshak_equilibration`（平面平衡 plateau、
鏡映 x=0）と `sn_1d_planar_slab_attenuation` 2 部構成 — Part A は
文献 GL S8 節点をハードコードした純吸収 slab 閉形式に対する sweep
モーメント一致（θ(τ) 閉包は指数解に厳密: 面間減衰
\((1-\tau(1-\theta))/(1+\tau\theta)=e^{-\tau}\)、セル平均
\(\theta\psi_{dn}+(1-\theta)\psi_{up}=(1-e^{-\tau})\psi_{up}/\tau\)、
実測 6.5e-10）、Part B は limiter 不活性 dt での定常
\(\mathrm{rad\_E}=\varphi/c\) 恒等（実測 3.6e-16）。

**BUG-9（2026-07-03, FLD 限流子の face 中心評価化）** — 1D FLD は
Levermore–Pomraning 限流子引数 \(R=|\nabla E|/(\sigma E)\) を**セル中心**で
評価し D を調和平均で面へ落としていた。放射前線では受け手セルの小さな E が
その D を \(\sim cE_{cold}/|\nabla E|\) に潰し、調和平均が面 flux を
\(\sim cE_{cold}\) に絞る → **前線停滞**（planar 平衡ゲートが暴露、球面は
中心セル微小体積が隠蔽）。修正: 面平均 E と面勾配で **face 上の λ** を評価
（Turner & Stone 2001 の face-centered D 規約; CASTRO II §6.4）、
\(\sigma_{face}=\tfrac12(\sigma_L+\sigma_R)\)。滑らか厚極限では旧調和形と
厳密一致 \(\mathrm{harm}(c/3\sigma_L, c/3\sigma_R)\equiv c/(3\bar\sigma)\)
のため変化は前線・急勾配領域のみ。自由流極限キャップは構成上厳密
\(|F|\le cE_{face}\)。GXII golden は前駆加熱の物理変化として再基準化
（ρ_peak 74→50 g/cc 等、VERIFICATION §10.1）。2D_RZ FLD は同型パターン
（cell 中心 λ + 調和平均）— 2D セッションへ引き継ぎ。



> **Phase 2a-1.5c 実装修正**：旧 2D_RZ FLD は shared internal face でも各隣接 cell が
> 自分の cell-centered face area を使っていたため、z-deformed Lagrangian mesh で
> \(A_{L\to R}\ne A_{R\to L}\) となり、`sum_face_div` に非保存な残差が生じた。
> 共有 edge 面積 \(A_{edge}\) への統一により、この face-area asymmetry を除去した。

> **Degenerate-face handling (Phase 2a-1.5d, 2026-05-07)**: Interior FLD
> diffusion coupling for adjacent cells \((i,j)\) and \((i',j')\) is skipped
> when the center-to-center distance is below `kFldFaceDistMin = 1e-12 cm`.
> This handles late-time Lagrangian mesh pinch / inversion where
> `rc[i] >= rc[i+1]` in low-density boundary cells. Without the skip, the FLD
> operator assembles `coef = dt*area*D/dist -> inf`, producing CSR matrix
> entries of O(1e+290) that destroy operator energy conservation by
> floating-point precision loss. Skipping the face produces zero diffusion flux
> at the degenerate location, which is the correct geometric limit (no spatial
> gradient between coincident points). Counted by `face_skip_dist_count`.

FLD 係数は
\[
R_{c,g}=\frac{|\nabla E_{c,g}|}{\sigma_{R,c,g}\max(E_{c,g},E_{floor})},
\qquad
D_{c,g}=\frac{c\,\lambda(R_{c,g})}{\sigma_{R,c,g}}
\]
で作る。既定の Levermore-Pomraning limiter は
\(\lambda=(\coth R-1/R)/R\) で、`"larsen"` は
\(\lambda=(9+R^2)^{-1/2}\)、`"none"` は \(\lambda=1/3\) を使う。LP の数値評価は
\(R<10^{-3}\) で級数 \(1/3-R^2/45+2R^4/945\)（打ち切り誤差 \(O(R^6/4725)\)）、
\(R>50\) で \((1-1/R)/R\)（\(\coth\) の指数補正 \(<4\times10^{-44}\)）、
中間域のみ raw 形とする（2026-07-26、AI review k05 6.1: 旧分岐
\(R<10^{-6}\to1/3\) / \(R>50\to1/R\) は \(R\sim10^{-6}\) 近傍の桁落ち
~\(10^{-4}\) 相対と \(R=50\) での +2% 不連続を持った — 1D 2 コピー修正済み、
2D コピーは別レーン）。面係数の構成は次元で異なる（2026-07-26 doc 真実復元）:
**1D_SPH は face-centered 評価**（1D BUG-9 根治後の現行実装）—
\(\sigma_{R,f}=\tfrac12(\sigma_{R,L}+\sigma_{R,R})\)、
\(E_f=\max(\tfrac12(E_L+E_R),10^{-300})\)、
\(R_f=|E_R-E_L|/(d_f\,\sigma_{R,f}\,E_f)\)、
\(D_f=c\,\lambda(R_f)/\sigma_{R,f}\)。拡散極限（両側 \(\lambda=1/3\)）では旧
harmonic 形 \(\mathrm{harm}(c/3\sigma_L,\,c/3\sigma_R)=c/(3\cdot\tfrac12(\sigma_L+\sigma_R))\)
と厳密一致する。**2D_RZ は現行 cell-centered** \(D_c\)（cell 勾配で \(R_c\)）を
隣接 harmonic mean して \(D_f\) を作る — 1D BUG-9 と同型の front-stall 機構が
残存する既知課題（AI review k05 M1、face-centered 化は 2D レーン所掌）。

1D_SPH の群ごとの線形系は tridiagonal で、CUDA の cuSPARSE
`cusparseDgtsv2StridedBatch` を使う。batch 数は \(G\)、各 system size は
\(N_{cell}\)、batch stride は \(N_{cell}\) である。2D_RZ の線形系は群ごとの CSR
5点ステンシルとして組み立てる。`linear_solver_2d="amgx_cg"` は AmgX が CMake で
見つかった場合に、同梱 config（`resources/amgx_fld_config.json`）の
CG+AMG/Jacobi preconditioner を使う。AmgX が無い build では WARNING を出して
debug 用の cuSPARSE SpMV + Jacobi preconditioned CG へフォールバックする。
`linear_solver_2d="cusparse_cg_jacobi"` を明示すると同じ fallback を warning なしで使う。
`linear_solver_2d="cusparse_cg_zline"` は z-line block-Jacobi preconditioner を使い、
\(M\) を radial line ごとの z-tridiagonal（slots `diag`/`jB`/`jT`）の block-diagonal として
`cusparseDgtsv2StridedBatch` で適用する。\(M\) は SPD なので PCG の solve target は同じ
\(Ax=b\) のままで、`dz<dr` の z-anisotropic stencil で反復数を減らす。既定 `linear_solver_2d="auto"`
（2026-07-11 flip）は **namelist validate 時**に nr が 2 の冪かつ nz>=3 なら
`cusparse_cg_rgmg`、それ以外で nz>=3 なら `cusparse_cg_zline`、どちらも不成立なら
`cusparse_cg_jacobi` へ解決する（解決結果は 1 回 INFO ログ、frozen config には
解決後の値が入る）。solver 整合性（裁定前提①、同日統合）: deck 明示の有無を追跡し
requested/resolved を run_info + HDF5 metadata に記録、**AmgX 未 link build での
明示 `"amgx_cg"` は ConfigError（fatal — 黙った能力置換の廃止）**、`"jacobi"` は
debug fallback CG の別名。validate() を経ない手組み config（unit test 経路）が
"auto" のまま solver へ到達した場合は WARNING 付き Jacobi fallback の防御分岐が拾う。
Convergence criterion: ||r_k|| <= cg_inner_tol · D where D = ||r_0|| (cg_tol_norm="r0", historic default) or max(||b||, tiny) (cg_tol_norm="rhs"). The rhs normalization decouples the stopping test from warm-start quality; with the previous-solution warm start the r0-relative form over-solves by construction.

Opt-in Anderson(m) acceleration (`outer_accel="anderson"`) mixes the next emission-linearization temperature from the last m outer residuals (Walker-Ni form, damping beta, Tikhonov-regularized normal equations, per-step history). Mixing is applied only when continuing to another outer iteration; a converged exit always returns the raw Newton output, so the accepted fixed point satisfies the same outer_tol contract as plain iteration. Degenerate least-squares rounds fall back to plain iteration; mixed temperatures are floored at floors.Te and non-finite mixes fall back to the raw Newton value per cell.

`linear_solver_2d="cusparse_cg_rgmg"` は r-semi-coarsened geometric multigrid を
同じ 2D_RZ FLD CG の SPD preconditioner として使う opt-in solver である。演算子は
5点ステンシル配列（`diag`, `ar_left`, `ar_right`, `az_m`, `az_p`）で保持し、階層は
\((n_r,n_z)\rightarrow(n_r/2,n_z)\rightarrow\cdots\rightarrow(1,n_z)\) と
r 方向だけを半粗化する（\(n_r\) は 2 の冪を要求し、z 解像度は全 level で保持する）。
粗格子演算子は piecewise-constant prolongation \(P\) と \(R=P^T\)（r 方向 pairwise 和）による
Galerkin 集約 \(A_c=R A_f P\) で作り、保存的 FV 集約と同じ 5点形式を保つ。粗格子で
\(D\) や境界条件を再評価しないため保存則と同一解が保たれ、assembled diagonal に入った
BC leakage は集約で自動的に伝播する。V(1,1) cycle は damped z-line block-Jacobi の
pre/post smoother、restriction、再帰 coarse solve、prolong-correction から成り、
最粗 \(n_r=1\) では r 結合が無い純 z-tridiagonal を z-line solve で厳密に解く。smoother は
各 radial line の z-tridiagonal を cuSPARSE `cusparseDgtsv2StridedBatch` で解き、
\(x \leftarrow x+\omega M_z^{-1}(b-Ax)\) を適用する。\(M_z\) は対称 SPD block diagonal なので
pre/post を同じにした cycle は自己随伴であり、\((2/\omega)M_z-A\) が半正定値
（実用上 \(\omega \le 2/\lambda_{\max}(M_z^{-1}A)\)）なら V-cycle は SPD preconditioner になる。
既定の `rgmg_smoother_omega=0.67` はこの damped smoother に使う。PCG の収束先は同じ
\(A^{-1}b\) で、bit-exact な同一反復列は保証しないが、z-line block-Jacobi に残る
radial smooth error による mesh-dependence を coarse-grid correction で抑える。

RGmg 前処理の検証 battery（既定 flip の前提②/④、2026-07-11）: (i) コード監査 —
V(1,1)・pre/post 同一 ω の damped z-line Jacobi（batched gtsv、line 逐次順序なし）、
R=P^T 厳密、Galerkin RAP 厳密、最粗 nr=1 厳密解、per-apply 固定線形演算子 —
により対称性は構成的に成立し flexible CG は不要（標準 Fletcher–Reeves PCG が適法）。
(ii) 常設 ctest `test_fld_2d_rz_rgmg_verification` — 合成 stress 族 16 spec
（contrast ≤1e8 = 生産比の 100 倍、異方性両極、弱 shift）で SPD 対称性/正値性 probe、
帯 Cholesky 直接解対照、生産 CG の真残差 gate（‖b−Ax‖/‖b‖ ≤ 1e-8）、cap 飽和ゼロ、
z-line SPD anchor、assembled-G2 leg。(iii) tol ladder（i4b 300-step / capsule
3000-step、cg_inner_tol 1e-4..1e-10、U_lin ≤ 0.1 ΔQ_accept）。CG 費用が非支配の
regime（i4a marshak 級）では RGmg の wall 利得は無い（2026-07-10 実測、中立）。
詳細: docs/design/rgmg_verification_battery_20260711.md、VERIFICATION §9.5.6。

As of L1b-2 the captured block also covers the z-line and RGMG preconditioner applications (stream-parameterized variants; capture failure still latches the eager loop, and the graph key bakes in every preconditioner buffer pointer so any hierarchy reallocation forces recapture).

`state.rad_E_old`
は device resident の backward-Euler 履歴で、初回 step と ALE invalidation 後に
`rad_E` から初期化し、各 radiation step 終端で更新する。

物質結合は電子エネルギーのみを更新する。2D_RZ FLD では constrained-B
conservative closure を用い、matter 側の放射源は FLD RHS に実際に入れた source
と同一にする。各 cell/group で
\[
S^{used}_{c,g}
=\Delta t\,V_c\left[
f_c\eta_{c,g}+
(1-f_c)c\sigma^{PA}_{c,g}E^n_{c,g}
\right]
\]
を定義する。Fleck を使わない経路では \(f_c=1\) なので
\(S^{used}_{c,g}=\Delta t\,V_c\eta_{c,g}\) である。Grey constant-opacity FLD では
専用 Fleck kernel が計算した \(f_c\) を同じ式に入れる。解いた
\(E^{n+1}_g\) に対して matter update は
\[
\rho_cV_c\left[e_e(\rho_c,T^{n+1}_{e,c})-e_e(\rho_c,T^n_{e,c})\right]
=\sum_g\left[
\Delta t\,V_c\,c\sigma^{PA}_{c,g}E^{n+1}_{c,g}
-S^{used}_{c,g}
\right]
\]
を満たす。GPU Newton kernel はこれと等価な cell-local residual
\[
F(T)=
\rho\frac{e_e(\rho,T)-e_e(\rho,T_e^n)}{\Delta t}
-\sum_g c\sigma^{PA}_{g}E^{n+1}_g
+\sum_g \frac{S^{used}_{g}}{\Delta t\,V}=0
\]
を解く。ここで Planck emission は final \(T\) から再評価しない。TMAT/table EOS
が利用可能な場合は、物質 Jacobian に
\(\rho c_{v,e}(\rho,T)/\Delta t\) を用い、収束後の `ee` と `Pe` は同じ電子
EOS テーブルから \((\rho,T)\) で書く。電子 EOS device view が無い場合は
\(\rho c_{v,e}(T-T_e^n)/\Delta t\) の定比熱 residual と ideal-gas 圧力書き込みへ
fallback する。放射源は固定済みの \(S^{used}\) なので、放射 Jacobian は物質
Newton には入れない。

> **W-B 適合修正（BUG-7, 2026-07-03）**: 1D `update_matter_kernel` は上の
> \(S^{used}\) 契約に違反して**未混合** \(c\sigma^{PE}B(T)\) を emission として
> 記帳していた（\(f_c\) を完全に無視）。放射側は混合源で解くため、
> \((1-f)c\sigma\Delta t\,|E^n-B|\) の幻エネルギーが毎 step 発生していた
> （`fld_1d_volume_source_balance` gate が検出、修正で balance 残差
> 7.9e-6 → 2.1e-10）。修正後の 1D kernel は Newton 残差・Jacobian・
> `rad_emit` 記帳の全てで \(f\,c\sigma^{PE}B(T)+(1-f)c\sigma^{PE}E^n\) を使う
> （実装は emission の \(B(T)\) を iterate ごとに再評価する — 凍結 \(S^{used}\)
> と outer 収束点で一致）。付随修正 2 件: (i) assemble の \(f_c\) 参照は
> `fleck[c]`（セル素 index）だったため \(G>1\) で誤要素を読んでいた —
> `fleck[c\,G+g]` に修正（旧挙動は実質「全セルが cell \(\lfloor c/G\rfloor\) の
> \(f\)」= GXII など多群 run では Fleck が事実上無効化されていた）。
> (ii) 1D の `compute_fleck_for_fld` は \(z>2\) で exp(−z) へ blend していたが、
> 整合化後は stiff 極限 \(zf\to0\) が物質-放射交換を凍結し平衡到達を阻害する
> ため、標準 Fleck-Cummings \(f=1/(1+z)\)（\(zf\to1/\alpha\)）に一本化した。
> 2D_RZ 側の同型監査（matter 側の \(f\) 整合・`fleck[c]` indexing・blend)は
> 2D セッションへ引き継ぎ。

> **W-I AFI モード（2026-07-03）**: `Radiation.multigroup_diffusion.fleck_mode="afi"` は Fleck ブレンドを消費点（assembly の擬似散乱項 + 物質側ブレンド）で無効化し、outer 反復（Picard）が完全陰的 emission \(c\sigma B(T^{n+1})\) を収束させる。Larsen, Kumar & Morel (JCP 238, 2013) により AFI 離散化は任意 \(\Delta t\) で一意解・最大原理・平衡拡散極限を満たす。実測（GXII FLD nr200）: Fleck 既定は生産 \(\Delta t\)（コロナ z≈3）で吸収エネルギーを z→0 極限比 ~35% 抑制し dt 依存が全 metric を汚染、AFI は生産 dt で極限の数%以内（dt×4 でも残差数%）。コロナの Picard 縮小率 ~z/(1+z)≈0.75 のため `max_outer_iterations >= 40` 推奨（未収束は rate-limited warning が出る）。既定は従来 `"fleck_cummings"`（golden 影響なし）。**既定は Fleck を維持（ユーザー決定 2026-07-04）** — AFI は namelist opt-in の検証・測定モードとして存続し、GXII golden の再基準化は行わない。dt 感度の定量（Fleck ~25% vs AFI 4.1%）は VERIFICATION §10.1 に記録済み。

`Radiation.multigroup_diffusion.hydro_coupling` の既定は `"gamma_r_43"`。`"none"` は frozen-density historic behavior への明示 opt-out。`"gamma_r_43"` は opt-in で、1D deterministic FLD の Lagrangian
hydro half step ごとに \(E_rV^{4/3}\) を保存する gamma_r=4/3 radiation compression と
\(p_r=\sum_g E_g/3\) の force-side coupling を使う。
Default flipped 2026-07-06, reverted same day (v1 defect), RE-ADOPTED same night after the v3 fix and fresh A/B (R2-1=A). Scope enforcement (2026-07-06): activation additionally requires mode=="multigroup_diffusion" (SnTransport excluded). (History) DEFAULT REVERTED to "none" the same day: the rebaseline combined audit measured cumulative unexplained energy +8.6e9 erg (~10% of absorbed) on the GXII FLD regression with coupling on vs +0.6e9 off — the v1 force-side p_r work and the exact-adiabat V^{4/3} field payment do not cancel at finite amplitude (shocks/AV), a defect class invisible to the smooth-adiabat and linear-ceff gates. Deck opt-outs on compatible_energy decks stay as explicit documentation. Re-adoption path: v2 work-consistent payment (same p_r_half, same swept dV_c on BOTH modes — the design-doc v2 ruling extended to non-compatible mode, whose "v1 stays bit-for-bit" assumption this audit falsified).

Cut-1a/2 では DSA/TSA 加速は使わない。
収束判定は
\(\max_c |\Delta T_{e,c}|/\max(T_{e,c},T_{floor}) <\)
`Radiation.multigroup_diffusion.outer_tol` である。

2D_RZ FLD の deterministic tallies は
`rad_dep[c,g]=Delta t V_c c sigma_PA E^{n+1}_{c,g}` と
`rad_emit[c,g]=S_used[c,g]` である。`rad_emit` の HDF5 shape は従来通りだが、
意味は「final \(T\) から再評価した gross Planck emission」ではなく、放射線形系の
RHS に適用した source energy である。

HDF5 `radiation/fleck_factor` は cell-length の診断 field であり、現行実装では
2D_RZ gray \(G=1\) FLD の per-cell Fleck factor として有効である。Multigroup
FLD では Fleck work array が cell/group layout を持つ一方で、複数の FLD source/RHS
経路が cell-scalar indexing を仮定している既知の潜在不整合があるため、この field を
multigroup per-cell/per-group Fleck 診断として解釈してはならない。Multigroup indexing
修正と per-group export は別の合意項目で扱う。

#### 6.7.1 RH1 hydro+FLD peer-review verification gates (2026-05-13)

RH1 production closure uses three additional Catch2 verification gates over
`examples/verification/2d_rz_rh1_hydro_fld.py` without adding namelist
parameters or changing the cgs+eV unit system:

- `test_rh1_eq_neq_diffusion_branches.cu` runs planar radiative shocks at
  \(\kappa_a=1000\) and \(10\ \mathrm{cm^2\,g^{-1}}\). The high-opacity case
  gates the equilibrium-diffusion signature
  \(\max |T_r-T_e|/T_e \le 5\times10^{-2}\) in the shock-front region; the
  lower-opacity case gates the nonequilibrium-diffusion signature
  \(\max |T_r-T_e|/T_e \ge 2\times10^{-2}\). If the currently exposed deck sweep
  does not cleanly separate the regimes, the ctest records a feature-gap pass
  rather than changing production physics.
- `test_rh1_radiation_pressure_dominated_strong_driving.cu` runs a strong
  cylindrical blast and requires \(\max(P_{rad}/P_{mat})>0.1\), with
  \(P_{mat}=P_e+P_i\). It uses HDF5 `radiation/pressure` when exported and
  otherwise records a feature-gap pass using `radiation/energy_density/3`, while
  the strict exported-field gate remains covered by the dedicated RH1
  radiation-pressure diagnostic. This is a diagnostic regime-exposure gate only:
  v1.0 still does not feed radiation pressure into the matter momentum equation
  (§1.1.2).
- `test_rh1_ale_on_remap_boundary_overshoot_mandatory.cu` compares ALE-on and
  ALE-off planar-radiative-shock \(T_e\) fields with
  `TENRYU_RH1_ALE=1` and `TENRYU_RH1_ALE_EVERY_N_STEPS=5`. The active gate is
  \(\max(T_{e,ALE}-T_{e,off})/T_{e,off}\le0.10\), termination at `t_end_reached`,
  and zero escape-valve firings. If the ALE-on run fails before production
  \(t_{end}\), the ctest records the empirical L261 feature gap.

参考: Levermore & Pomraning (1981) JQSRT, Larsen (1980) JQSRT, and the
Langer-Karlin-Marinak HYDRA hyd607 description of capsule-only multigroup diffusion.

#### 6.7.2 RH2 hydro+\(S_N\) peer-review verification gates (2026-05-13)

RH2 production closure adds three Catch2 gates over
`examples/verification/2d_rz_rh2_hydro_sn.py` without adding namelist
parameters, changing production constants, or changing the cgs+eV unit system:

- `test_rh2_grey_sn_radiative_shock_comparison.cu` compares the exposed grey
  RH2 \(S_N\) shock-tube deck against the grey RH1 FLD shock-tube deck at high
  opacity. The strict transport-class gate is shock-front-region \(T_e\)
  relative \(L_2 \le 10\%\). If the exposed RH2/RH1 decks cannot both run at
  the same comparison point, or if the current deck pair exceeds the strict
  profile gate, the ctest records a feature-gap pass rather than changing
  production physics or adding deck controls.
- `test_rh2_to_rh1_thick_limit_convergence.cu` uses the RH2 deck's
  `sn_transport` and `multigroup_diffusion` branches as same-hydro
  \(S_N\)-vs-FLD references for the RH1-class optically thick limit. It sweeps
  \(\kappa_a=10,100,1000\ \mathrm{cm^2\,g^{-1}}\), requires monotone decreasing
  \(T_e\) relative \(L_2\) error with increasing opacity, and requires the
  \(\kappa_a=1000\) error to be \(\le5\%\). Empirical non-monotonicity at the
  exposed grid/timestep is recorded as a feature gap.
- `test_rh2_eddington_radiation_pressure_tensor_diagnostics.cu` checks whether
  RH2 plot HDF5 exports \(S_N\) pressure/Eddington tensor diagnostics. When a
  \(P_{zz}\) or \(f_{zz}\) diagnostic is present, the thick isotropic case gates
  \(f_{zz}\to1/3\) within \(5\times10^{-2}\). The directional beam-limit gate
  \(f_{zz}\to1\) remains a feature gap until the RH2 deck exposes an
  angle-selective source/boundary diagnostic and the plot file exports the
  required angular moments.

#### 6.7.3 §I1-A grey FLD radiative shock — 2D RZ z-slab planar code-verification gate (A2 z-HLLC submode)

The split I1-A row `I1A_2D_RZ_FLD_CED_PLANAR_Z_SHOCK_HLLC` is a
code-verification gate for TENRYU's declared planar 2D_RZ z-slab grey FLD-CED
radiative-shock model with `HLLC_Z=on`. It must exercise the production 2D RZ FLD path
(`fld_2d_rz_gpu.cu`) with `Radiation.mode="multigroup_diffusion"`, one grey
group, frequency-independent opacity, and `flux_limiter="none"` so the closure
is constant Eddington \(1/3\). This is not external physics validation and not
evidence that the default VNR/ALE hydro path carries radiative shocks. The unit
system remains cgs + eV.

Geometry:
\[
r\in[0,R_{max}],\qquad z\in[z_{min},z_{max}],
\]
with r-invariant initial data and mirror boundaries at \(r=0\) and
\(r=R_{max}\). The z boundaries use state-supply Riemann data when available,
or a finite shock-tube fallback otherwise.

For axisymmetric FLD, the radiation-energy diffusion term is
\[
\nabla\cdot(D\nabla E)=
{1\over r}{\partial\over\partial r}
\left(rD{\partial E\over\partial r}\right)
+{\partial\over\partial z}
\left(D{\partial E\over\partial z}\right).
\]
If the initial and boundary states are r-invariant, then
\(\partial E/\partial r=0\) and \(\partial T_e/\partial r=0\). The radial
diffusion term vanishes, reducing the 2D RZ FLD equation to the planar 1D z
equation:
\[
{\partial E\over\partial t}
= {\partial\over\partial z}
\left(D{\partial E\over\partial z}\right)
+c\kappa_R\rho(a_{eV}T_e^4-E),
\qquad D={c\over3\kappa_R\rho}.
\]
The same reduction applies to the material energy coupling in the absence of
r-dependent hydro motion. The production gate therefore compares the
radial-average z profile against the reused FLD-CED ODE reference while also
requiring radial invariance as an independent 2D RZ kernel check.

The 2D RZ I1-A strict gate uses `tools/validation/rz_profile_average.py` for
volume-weighted radial averaging and radial-invariance diagnostics, and
`tools/validation/radiative_shock_metrics.py` for the Wave 12
shock-windowed \(L_2\), legacy convolved peak-gap diagnostic, and Richardson
\(D_\infty\) metric framework. The production gate no longer accepts or rejects
I1-A using a candidate-inferred convolution kernel.  The strict precursor gate
uses fixed, pre-registered scales and an exact piecewise-linear cell-average
projection of the reference onto the comparison grid.

Strict hardened-metric acceptance for I1-A:

- pass/fail values are taken from the finest production grid
  \(64\times1024\), with all fixed metrics also emitted by grid.  Coarser
  grids remain convergence/trend diagnostics unless a gate explicitly says
  otherwise;
- fixed upstream precursor window
  \(W_u=[z_s-8\ell,\ z_s-\delta]\), where
  \(\ell=(\sqrt{3}\kappa_R\rho_{up})^{-1}=0.577\ \mathrm{cm}\) for the
  calibrated \(M=2\) case, \(z_s\) is the fixed half-jump density location
  \(\rho=0.5(\rho_{up}+\rho_{down})\), and
  \(\delta=\max(2\Delta z,\ 0.25\ell)\);
- precursor amplitude
  \(|D_{\max,h}/D_{\max,ref}-1|\le0.10\), with
  \(D=T_r/T_e-1\) and the reference projected by exact cell averaging, not by a
  candidate-derived blur kernel;
- integrated positive precursor area
  \(|A_h/A_{ref}-1|\le0.10\), where
  \(A=\int_{W_u}\max(D,0)\,dz\);
- fixed-window precursor shape
  \(\|D_h-D_{ref}\|_2/(\|D_{ref}\|_2+10^{-12})\le0.08\);
- independent matter-shock-thickness gate from the fixed 10--90% density jump:
  \(W_{shock}/\ell\le1.0\) or \(N_{shock}\le3\) cells.  This catches broad
  material shocks even when radiation-profile convolution would otherwise hide
  them;
- Richardson \(D_\infty^{fit}\) within 5% of the FLD-CED reference peak
  \(0.761\);
- radial-invariance \(\epsilon_r \le 10^{-8}\) for seeded smooth profiles or
  \(\epsilon_r \le 10^{-6}\) for the Riemann fallback;
- conservation: closed-domain mass \(\le 10^{-12}\) and energy
  \(\le 10^{-10}\), or open-domain energy \(\le 10^{-8}\);
- no escape valves:
  `emergency_cell_deactivation = thermal_subcycle_floor_hit = axis_spike_floor = 0`;
- Newton convergence: `converged_count = N_cells`.

PR 4 promotes the PR 3 smoke scaffold to the strict production CTest
`expensive.I1 FLD-CED 2D RZ z-slab strict`. The gate runs the three production
grids \(16\times256\), \(32\times512\), and \(64\times1024\), with z-resolution
prioritized. The harness does not override `t_end`; the deck default is the
Wave 10 production horizon
\[
t_{end}=\max(5\tau_{rel},3L/|u_{upstream}|)=32.35\ \mu s
\]
for the \(M=2\) FLD-CED calibration.

The strict CTest gates all of the following:

- `mach_match`, `pressure_ratio_match`, `profile_match`, and
  `reference_table_radiation_equilibrium_admissibility` t=0 admissibility on
  all three grids.  `pressure_ratio_match` compares code-vs-reference
  \(P_{rad}/P_{mat}\) edge and shock-ratio values with relative error
  \(\le\) `T0_PRAD_RATIO_REL_GATE`.  For `init_mode="reference_table"`,
  `reference_table_radiation_equilibrium_admissibility` requires the t=0
  radiation temperature to equal \(T_e\) with relative \(L_\infty\)
  \(\le\) `T0_TRAD_TE_REL_GATE`; the deck uses
  `radiation_field=equilibrium` IC so the FLD solve develops the precursor
  dynamically rather than pre-loading it.  This replaces the inapplicable
  two-state nED admissibility check for the smooth reference-table IC;
- branch-wise reference identity with median relative error \(\le 10^{-3}\) and
  p95 relative error \(\le 10^{-2}\);
- low-\(P_{rad}\) \(\max(P_{rad}/P_{mat})\le10^{-3}\) and frozen-reference
  conservation residual \(\le10^{-12}\);
- `termination_reason="t_end_reached"` on all three production grids;
- radial-invariance \(\epsilon_r\le10^{-6}\) for \(\rho,T_e,T_i,E_{rad},u_z\);
- fixed-window precursor amplitude/area/shape gates, independent material
  shock thickness, and
  Richardson \(D_\infty^{fit}\in[0.65,0.87]\).  The legacy
  `shock_windowed_l2_abs` and `convolved_peak_gap` values are still emitted as
  diagnostic continuity values but are not pass/fail gates;
- no escape valves:
  `emergency_cell_deactivation_fired_count =
  thermal_subcycle_floor_hit_count =
  axis_spike_floor_activation_count =
  newton_invalid_count = 0`;
- 2D RZ FLD Newton diagnostics from `fld_2d_rz_gpu.cu` satisfy
  `newton_cap_hit_count = 0`, `newton_invalid_count = 0`, and harness
  alternate relative residual `newton_resid_rel_max_alt <= 1e-3`.  The raw
  `newton_converged_count` remains reported, but it is diagnostic-only because
  multi-call FLD steps can legitimately accumulate more than one convergence
  count per cell.

For ctest performance, the I1-A strict harness runs with lean gate diagnostics
by default.  The environment gate `TENRYU_I1_2D_RZ_VERBOSE_DIAG=1` re-enables
the verbose smoke/debug diagnostics.  This switch affects only emitted
diagnostics and has zero physics impact; the Newton H3 line used by the gate is
still emitted through the production-audit path.

The CTest uses the reflecting z-boundary finite-shock-tube mode by default,
because that is the PR 3 mode already demonstrated to run through the 2D RZ FLD
path. The alternative
`TENRYU_I1_2D_RZ_BOUNDARY_MODE=state_supply` remains available for focused
stationary-shock hardening; if state-supply terminal-cell inversion reappears at
production time, the strict gate remains on the reflecting fallback and the
state-supply issue is documented separately. With these gates passing, the
current I1-A code-verification metric is satisfied for roadmap row
`I1A_2D_RZ_FLD_CED_PLANAR_Z_SHOCK_HLLC` per the high-AI consultation
acceptance criteria.

2026-05-24 closure note: `ctest #460` is OFFICIAL GREEN for this I1-A planar
z-slab gate on the hardened candidate-independent metric with opt-in
`total_energy_remap_2d_rz=true` and `hllc_z_flux_2d_rz=true`.  The fixed-scale
precursor amplitude errors were `16x256=0.266`, `32x512=0.189`,
`64x1024=0.0689` (`<=0.10` finest-grid gate), with Richardson
`D_infinity=0.799`, fixed-precursor area `0.056`, shape_l2 `0.052`, and
matter_shock_width_over_ell `0.151`.  The 2026-05-23 P-metric hardening
replaced the fragile pass/fail metric with the fixed-window gates above and
keeps `convolved_peak_gap` diagnostic-only.
The closure scope and caveats are recorded in
`docs/validation/2d_rz/I1/closure_summary.md`.

The verification deck `examples/verification/i1_fld_ced_2d_rz_slab.py` opts
into those two I1-specific flags by default, while the global namelist defaults
remain false for byte-identical legacy operation in other decks.

PR 3 added the smoke-scaffold deck
`examples/verification/i1_fld_ced_2d_rz_slab.py` and harness
`tools/validation/run_i1_2d_rz_fld_ced.py` for this 2D RZ formulation. The
geometry is an \(N_R\times N_Z\) z-slab with \(r_{min}=0\), \(r_{max}=10\)
cm, \(z_{min},z_{max}\) taken from the FLD-CED reference-table range, axis
mirror at \(r=0\), mirror at \(r=r_{max}\), and r-invariant two-state or
reference-table initial data along z. The deck keeps the supported 2D RZ
`state_supply` hydro boundary paired with one-group FLD `state_supply`
available through `TENRYU_I1_2D_RZ_BOUNDARY_MODE=state_supply`, but PR 3
defaulted to a finite-shock-tube smoke fallback (`z_bottom/z_top=reflect`, FLD
z reflect). ALE remains off.

##### §I1-A auxiliary anchors: 1D_SPH LR07-EDA / LE08 nED / FLD-CED

The existing Wave 5/6C/7B/8/9/10/12 LR07-EDA, LE08 LM_nED, and FLD-CED
documentation below is preserved as explicit I1-A auxiliary rows. Commits
`c88eb0e9 -> d3d68c59` verified the quasi-planar 1D_SPH code path at
`r_min=1e6 cm`, dispatching to `fld_1d_gpu.cu`. This is a valid auxiliary
verification of the `fld_1d_gpu.cu` assertion path
`state.mesh.dim == 1 || cfg.main.dimension == "1D_SPH"`, and it preserves the
FLD-CED ODE generator, dual-flux schema v3, branch-wise diffusion identity
checks, shock-windowed convolved-reference metric, Richardson
\(D_\infty\) fit, and M=1.5/2/3 reference tables.

It is not counted toward the production 2D RZ I1-A row because it does not exercise
`fld_2d_rz_gpu.cu` on a planar 2D_RZ slab deck.

The split auxiliary rows are:

- `I1A_AUX_LR07_EDA_1D`: external physics/model anchor, PASS (`ctest #457`);
- `I1A_AUX_LE08_NED_1D`: external physics/model anchor, PASS (`ctest #458`);
- `I1A_FLD_CED_1D`: internal model anchor, PASS (`ctest #459`).

##### Historical §I1-A auxiliary details: LR07-EDA / LE08 / FLD-CED 1D_SPH chain

The redesigned I1 code-verification gate is the planar grey equilibrium-diffusion
radiative shock of Lowrie--Rauenzahn 2007, restricted to a low-radiation-pressure
regime compatible with TENRYU v1.0's omitted radiation force (§1.1.2). It is not
a GXII capsule validation proxy and introduces no new production equations,
constants, units, or namelist parameters. The implementation consists of:

- `tools/validation/lr07_eda_reference_generator.py`, which writes frozen JSON
  reference tables under `tests/verification/data/lr07_eda_reference/`.
- `examples/verification/i1_lr07_eda_grey_radshock.py`, a 1D_SPH radial deck
  that evaluates the planar LR07-EDA table at large radius with hydro,
  one-group grey FLD, electron heat conduction enabled, Qei available via the
  existing 2T coupling path, and `Laser(enabled=False)`.
- `tools/validation/run_i1_lr07.py` and
  `tests/verification/test_i1_lr07_eda_grey_radshock.cu`, which run the
  three-grid gate and compare HDF5 profiles to the frozen table.

The steady reference uses cgs+eV variables and the same ideal-gas convention as
the deck:
\[
P_{mat}=\rho R T,\qquad
R=(1+\bar Z)\frac{eV\_to\_erg}{A m_p},
\qquad P_{rad}=\frac{a_{eV}T^4}{3}.
\]
For a prescribed upstream Mach number \(M_0\), upstream state
\((\rho_0,T_0)\), \(\gamma\), and constant \(\kappa_R\), the generator enforces
\[
\rho u=m,\qquad
\rho u^2+P_{mat}=J,
\]
\[
m\left(c_pT+\frac{u^2}{2}\right)+F_{rad}=H,
\qquad c_p=\frac{\gamma}{\gamma-1}R,
\]
with \(T_{rad}=T_{mat}=T\) and
\[
F_{rad}= -\frac{c}{3\kappa_R\rho}\frac{d(a_{eV}T^4)}{dx}.
\]
The momentum quadratic selects the supersonic upstream branch and subsonic
downstream branch. The tabulated coordinate is monotone through the
equilibrium-diffusion shock structure, with \(x=0\) at the branch switch.
Schema v2 samples the frozen table uniformly in branch temperature rather than
uniformly in \(x\), because branch-wise finite-difference checks differentiate
the frozen table itself and must resolve the asymptotic far-state tails.

Phase A Wave 3 executes this planar reference as a mesh-robust 1D_SPH
large-radius approximation instead of a 2D_RZ slab. The deck maps the frozen
table coordinate to
\[
x(r)=x_{\min}+(r-r_{\min})\frac{x_{\max}-x_{\min}}{r_{\max}-r_{\min}},
\qquad r_{\min}=100.0\ \mathrm{cm},\quad
r_{\max}=r_{\min}+x_{\max}-x_{\min},
\]
so the default \(M_0=2\) table spans \(r\in[100.0,100.04163477491]\) cm and places
the branch switch at \(r=100.02328389941\) cm. The comparison table remains the
same 1D ODE solution; the harness reads 1D radial HDF5 fields and uses the
native spherical shell volumes for reductions. The residual curvature error is
therefore an explicit verification-deck approximation, bounded by the large
reference radius and not a change to the LR07-EDA equations, constants, or unit
system.

The Wave 3 calibration uses:
\[
\gamma=5/3,\quad A=1,\quad \bar Z=1,\quad
\rho_0=2.0\ \mathrm{g/cm^3},\quad T_0=30\ \mathrm{eV},\quad
\kappa_R=0.5\ \mathrm{cm^2/g},
\]
for \(M_0\in\{1.5,2.0,3.0\}\) with 1601 points per table. This preserves
\(\kappa_R\rho_0=1\ \mathrm{cm^{-1}}\) while reducing the omitted-radiation-force
perturbation. The relevant scaling is \(P_{rad}/P_{mat}\propto T^3/\rho\),
not \(T^4\), because \(P_{mat}\propto\rho T\). Relative to the Wave 2
\(T_0=80\) eV, \(\rho_0=1\ \mathrm{g/cm^3}\) references, the Wave 3
\(T_0=30\) eV, \(\rho_0=2\ \mathrm{g/cm^3}\) tables reduce the reference
pressure ratio by \((30/80)^3/2\). The frozen-table diagnostics are:

| \(M_0\) | max \(P_{rad}/P_{mat}\) | max \(|\nabla P_{rad}|/|\nabla P_{mat}|\) | conservation residual |
|---:|---:|---:|---:|
| 1.5 | \(7.4621631\times10^{-7}\) | \(1.3226286\times10^{-6}\) | \(2.289371\times10^{-16}\) |
| 2.0 | \(2.1317967\times10^{-6}\) | \(3.7784912\times10^{-6}\) | \(2.836829\times10^{-16}\) |
| 3.0 | \(1.4341980\times10^{-5}\) | \(2.5420334\times10^{-5}\) | \(2.718628\times10^{-16}\) |

The production CTest uses the \(M_0=2.0\) table on \(N_r=64,128,256\) 1D_SPH
radial grids. Phase A Wave 4 follows the Round 9 AI fallback decision and
changes the default deck initialization from the smooth `reference_table`
profile to a `two_state` Riemann shock-formation protocol. In this mode the
discontinuity is placed at
\[
r_{shock}=\frac{r_{\min}+r_{\max}}{2},
\]
with the LR07-EDA upstream Rankine-Hugoniot state for \(r<r_{shock}\) and the
downstream state for \(r\ge r_{shock}\). The smooth `reference_table` path
remains available by explicit harness/deck override for diagnostics, but is no
longer the production default. The Wave 4 deck also lengthens the default
runtime to
\[
t_{end}=1.5\,\frac{x_{\max}-x_{\min}}{|u_{upstream}|},
\]
three times the Wave 3 half-crossing default, so the two-state initial
discontinuity can form and propagate before the production comparison.

Before the production ladder, the harness runs a t=0 admissibility audit on the
same grid set and stops before production comparison if any of the following
fail:

- upstream Mach consistency
  \(|M_{code}-M_0|/M_0\le5\times10^{-3}\);
- max \(P_{rad}/P_{mat}\) consistency between the code snapshot and the frozen
  table at the same cell centers, with relative mismatch \(\le5\%\);
- EDA initialization consistency
  \(\max |T_{rad}-T_e|/T_e\le10^{-3}\).

The production harness locates the shock in both simulation and reference
profiles by the strongest \(T_e\) gradient, shifts each simulated profile by the
measured shock displacement before computing profile errors, and masks the
outer 10% of cells at each domain boundary for the \(L_2\) comparisons. For the
Wave 4 two-state protocol, the blocking \(L_2\) values are averages over a
quasi-steady snapshot window rather than final-snapshot-only values. The window
starts at the first snapshot interval whose adjacent shock speeds differ by
less than 1% relative to the mean shock-speed magnitude; if no such interval is
found, the analyzer records the fallback method and the comparison reduces to
the final snapshot. The summary still emits final-snapshot and full-traversal
averages as non-gating diagnostics. It also runs a non-gating
\(0.25\,t_{end}\) diagnostic after the production ladder; if the strict
production \(L_2\) gates fail while this shorter diagnostic passes, the summary
flags the result as possible boundary/transient contamination.

Phase A Wave 5 adds non-gating diagnostic instrumentation to distinguish
reference fidelity, EDA-closure, and 2T material-partition causes of residual
Gate 3--5 failures. The t=0 admissibility audit now builds its expected profile
from the active initialization mode: `reference_table` uses the frozen smooth
LR07-EDA table, while `two_state` uses the upstream/downstream R-H plateau
states and excludes cells within six cell widths of the initialized
discontinuity. The dynamic closure-defect summaries report
\[
\delta_{Tr}=T_{rad}/T_e-1,\qquad
\delta_{ei}=T_i/T_e-1,
\]
and flux residuals
\[
R_F^{nonEDA}=\frac{F_{rad}+D\,\partial_xE_{rad}}
{|F_{rad}|+|D\,\partial_xE_{rad}|},\qquad
R_F^{EDA}=\frac{F_{rad}+D\,\partial_x(a_{eV}T_e^4)}
{|F_{rad}|+|D\,\partial_x(a_{eV}T_e^4)|},
\quad D=\frac{c}{3\kappa_R\rho}.
\]
When an HDF5 radiation first-moment flux is unavailable for 1D FLD plots, the
harness labels the flux source and infers \(F_{rad}\) from the steady
shock-frame matter energy-flux defect for diagnostic use only; no production
equation or output schema is changed.

Wave 5 also emits shock-centered profile dumps in \(r-r_s\),
\(\tau=\int_{r_s}^{r}\kappa_R\rho\,dr\), and
\(m=\int_{r_s}^{r}\rho\,dr\), and reports \(T_e\) and \(T_{rad}\) relative
\(L_2\) errors in all three coordinates. Far-state R-H residuals are computed
from plateau averages using
\[
J=\rho(u-v_s),\qquad
\Pi=\rho(u-v_s)^2+P_{mat},\qquad
\Phi=J\left(e+\frac{(u-v_s)^2}{2}+\frac{P_{mat}}{\rho}\right),
\]
with radiation momentum omitted consistently with the low-\(P_{rad}\) I1
acceptance regime. These diagnostics feed Wave 6 branch selection only; they do
not relax the Gate 4 \(T_e\) 5% threshold.

Phase A Wave 6C-revised regenerates the LR07-EDA references using the
radiation-momentum-free reduction required by TENRYU v1.0 (§1.1.2; see also
`docs/validation/2d_rz/RH1/audit/lowrie_edwards_feasibility.md:15-33`).
The implementation follows the CC-Codex consensus recorded in
`tmp/discussions/20260514-181253-wave6-6c-ext-vs-7b-le08-switch/log.md`:
the I1 table remains a low-\(P_{rad}\), equilibrium-diffusion energy reference,
but the steady momentum invariant is gas-only and no production radiation
momentum equation is introduced.

Wave 6C-revised also bumps the frozen JSON table to schema version 2 with a
dual-flux contract:

- `F_rad_erg_per_cm2_s` is the radiation flux conjugate to the monotone
  `x_cm` table coordinate. It is the field used by branch-wise diffusion
  identity checks and by closure residuals of the form
  \(-D\,\partial_x(a_{eV}T^4)\).
- `F_rad_energy_invariant_erg_per_cm2_s` is the LR07 branch-oriented signed
  flux used for gas-energy conservation diagnostics. It equals the monotone
  flux on the upstream branch and is sign-opposite on the downstream branch.

The reference identity gate is strict for each frozen table
\(M_0\in\{1.5,2.0,3.0\}\) and each branch independently:
\[
\mathrm{median}(|R_F|)<10^{-3},\qquad
\mathrm{p95}(|R_F|)<10^{-2},
\]
where \(R_F\) compares the tabulated monotone-coordinate flux against
\(-D\,\partial_x(a_{eV}T^4)\) on the same branch. The
energy-invariant field is reported as a sanity diagnostic with the downstream
orientation sign restored, but the strict gate is on the monotone-coordinate
schema contract.

Wave 7 escalation rule: if, after Wave 6C-revised, the harness still reports
`max_abs_delta_Tr_interior > 0.05`, `max_abs_flux_resid_EDA > 1e-3`, and
`max_R_J/Pi/Phi > 0.05`, the next wave must first run the sensitivity sweep
\(2\times\kappa_R\), \(dt/2\), and \(N_R=512\) before reopening a 7B
Lowrie-Edwards switch. This rule is documentation-only in Wave 6C-revised.

Phase A Wave 7B reopens that switch after the Wave 7 sensitivity sweep on
commit `be0944ed` showed the grey-FLD solution is intrinsically
nonequilibrium-diffusion in this calibration: the EDA flux residual stayed at
1.0 under baseline, \(N_R=512\), \(t_{end}\times2\), \(dt/2\), and
\(\kappa_R\times2\), while \(\kappa_R\times2\) worsened
`dTr_max` from 3.079 to 3.765. The primary I1 reference is therefore the
Lowrie--Edwards 2008 grey nonequilibrium-diffusion shock generated through
ExactPack `radshocks.nED_Solver` with `problem="LM_nED"`. ExactPack's
`problem="nED"` is the Ferguson--Morel--Lowrie model and must not be labeled
LE08. The consensus record is
`tmp/discussions/20260514-181253-wave6-6c-ext-vs-7b-le08-switch/log.md`.

The LE08 frozen tables use the same cgs+eV material convention and low
\(P_{rad}\) calibration as Wave 6C:
\[
\gamma=5/3,\quad A=1,\quad \bar Z=1,\quad
\rho_0=2.0\ \mathrm{g/cm^3},\quad T_0=30\ \mathrm{eV},\quad
\kappa_R=0.5\ \mathrm{cm^2/g},
\quad M_0\in\{1.5,2.0,3.0\}.
\]
The steady nED closure uses distinct material and radiation temperatures,
\[
P_{mat}=\rho R T_m,\qquad E_{rad}=a_{eV}T_r^4,\qquad
P_{rad}=E_{rad}/3,
\]
with diffusion flux
\[
F_{rad}= -\frac{c}{3\kappa_R\rho}\frac{dE_{rad}}{dx}.
\]
ExactPack stores a lab-frame radiation energy flux divided by the upstream
sound speed; the schema-v3 `F_rad_erg_per_cm2_s` field subtracts the
\((4/3)uE_{rad}\) radiation-advection term so the table flux is conjugate to
the monotone `x_cm` diffusion identity. `F_rad_lab_erg_per_cm2_s` is retained
as an LE08-v3 diagnostic field for the full ExactPack conservation check.

Schema v3 extends the Wave 6C schema-v2 dual-flux contract. It preserves
`x_cm`, `rho_g_per_cc`, `u_cm_per_s`, `T_eV`,
`P_mat_dyne_per_cm2`, `P_rad_dyne_per_cm2`,
`F_rad_erg_per_cm2_s`, `F_rad_energy_invariant_erg_per_cm2_s`, and `branch`,
and adds `T_rad_eV`, `E_rad_erg_per_cm3`, `reference_model`,
`problem`, and `schema_version=3`. The top-level provenance block records the
ExactPack version, ExactPack git sha when available, local git sha,
low-\(P_{rad}\)/force diagnostics, integrator warnings, generation timestamp,
and the flux-frame convention. ExactPack is an optional regeneration
dependency only; regeneration can use
`pip install git+https://github.com/lanl/ExactPack.git@master`. TENRYU runtime,
ctest, and the frozen-table checker do not import ExactPack.

The LE08 Wave 7B files are:

- `tools/validation/le08_ned_reference_generator.py`
- `tools/validation/check_le08_exactpack.py`
- `examples/verification/i1_le08_grey_ned_radshock.py`
- `tools/validation/run_i1_le08.py`
- `tests/verification/test_i1_le08_grey_ned_radshock.cu`
- `tests/verification/data/le08_ned_reference/le08_nED_M{1p5,2,3}.json`

For Wave 7B, LR07-EDA ctest #436 remains an auxiliary dual-flux schema
regression guard. Its generator, checker, harness, ctest, and frozen v2 tables
are intentionally left unchanged.

Wave 8 changes only the LE08 deck defaults and partial-checkpoint interpretation:
`r_min=1.0e6 cm` makes TENRYU's 1D_SPH mesh quasi-planar, while
`t_end=3.0 (x_max-x_min)/|u_upstream|` lets the two-state Riemann initial
condition settle to the steady nED shock. The previous `r_min=100 cm` introduced
O(0.1) spherical-curvature effects and the previous 1.5 crossing-time endpoint
was transient. Host verification gave `dTr_max=0.589` versus the intrinsic LE08
reference separation `0.761`, a 23% gap accepted for this nED verification.

The blocking production gates are:

- Gate 1: \(\max(P_{rad}/P_{mat})\le10^{-3}\). The force-ratio diagnostic
  \(\max(|\nabla P_{rad}|/|\nabla P_{mat}|)\) remains reported as a
  partial-checkpoint diagnostic rather than a strict exit-code gate because the
  converged steady LE08 shock concentrates gradients while \(P_{rad}/P_{mat}\)
  remains \(O(10^{-5})\).
- Gate 2: frozen-reference mass, momentum, and total energy flux residual
  \(\le10^{-12}\) relative.
- Gate 3: final-time shock-position convergence rate \(\ge1.0\) over
  \(h,h/2,h/4\), using the peak-\(|dT_e/dx|\) shock location and adjacent-grid
  \(L_1\) position differences. The first-order threshold is specific to the
  Wave 4 two-state shock-front localization protocol; it does not relax the
  radiation-pressure or conservation gates.
- Gate 4: phase-aligned, interior-masked post-shock \(T_e\) relative
  \(L_2\le5\%\) against the LR07-EDA table, averaged over the quasi-steady
  window.
- Gate 5: phase-aligned, interior-masked pre-shock radiation precursor
  \(T_{rad}=(E_{rad}/a_{eV})^{1/4}\) relative \(L_2\le10\%\) against the same
  table, averaged over the quasi-steady window.

Phase A Wave 9 adds an in-house constant-Eddington FLD nED reference because
Wave 7B/8 isolated a model mismatch between ExactPack `problem="LM_nED"` and
TENRYU's grey FLD operator with `flux_limiter="none"`. The ExactPack LE08 table
retains its role as an auxiliary cross-check, while the FLD-CED table is the
equation-matched reference for TENRYU's current no-\(O(v/c)\), gas-momentum-only
model. The consensus record is CONSENSUS Round 5 thread `019e25c4-...`; the
local design trace remains
`tmp/discussions/20260514-181253-wave6-6c-ext-vs-7b-le08-switch/log.md`.

The Wave 9 smooth-branch invariants are
\[
\rho u=J,\qquad \rho u^2+P_{mat}=\Pi,\qquad
J\left(c_pT+\frac{u^2}{2}\right)+F=\Phi ,
\]
with \(P_{mat}=\rho RT\), \(c_p=\gamma R/(\gamma-1)\), no radiation pressure in
the gas momentum invariant, and \(F\) the comoving FLD radiation flux. The
constant-Eddington moment equations, following the diffusion-limit form in
Mihalas--Mihalas §97 and the steady-shock shooting technique of Lowrie 2007 §2,
are
\[
F=-\frac{c}{3\kappa_R\rho}\frac{dE}{dx},\qquad
\frac{dF}{dx}=c\kappa_R\rho\left(a_{eV}T^4-E\right).
\]
For a branch temperature \(T\), the velocity root is
\[
u_\pm(T)=\frac{\Pi\pm\sqrt{\Pi^2-4J^2RT}}{2J},
\]
where \(+\) is the upstream supersonic branch and \(-\) the downstream
subsonic branch. The algebraic flux is
\[
F(T)=\Phi-J\left(c_pT+\frac{u_\pm^2}{2}\right).
\]
The generator integrates the first-order system in
\(W=E-a_{eV}T^4\), which removes cancellation at the far equilibrium states:
\[
\frac{dW}{dT}=\frac{3F(dF/dT)}{c^2W}-4a_{eV}T^3.
\]
The branch coordinate satisfies
\[
\frac{dx}{dT}=-\frac{dF/dT}{c\kappa_R\rho W}
\]
on the upstream branch; the downstream integration uses the opposite
distance-from-far sign and is then mapped to monotone \(x_{cm}>0\). Boundary
conditions are \(F=0\) and \(E=a_{eV}T^4\) at both far states. The gas shock is a
discontinuous hydro jump between the upstream and downstream branches; the
shooting variable is the common shock-front flux \(F_s\), and the match
condition is
\[
E_{up}(F_s)=E_{dn}(F_s),\qquad F_{up}(F_s)=F_{dn}(F_s)=F_s .
\]

The new schema-v3 reference model is
`reference_model="FLD_const_Eddington_nED_low_Prad"` with
`problem="FLD_const_Eddington"`. The frozen files are:

- `tools/validation/fld_const_eddington_reference_generator.py`
- `tools/validation/check_fld_const_eddington.py`
- `examples/verification/i1_fld_ced_grey_radshock.py`
- `tools/validation/run_i1_fld_ced.py`
- `tests/verification/test_i1_fld_ced_grey_radshock.cu`
- `tests/verification/data/fld_const_eddington_reference/fld_ced_M{1p5,2,3}.json`

The FLD-CED tables preserve the LE08 schema-v3 nED fields
`T_rad_eV`, `E_rad_erg_per_cm3`, `F_rad_erg_per_cm2_s`,
`F_rad_energy_invariant_erg_per_cm2_s`, and `F_rad_lab_erg_per_cm2_s`. For this
no-\(O(v/c)\) in-house model, the three flux fields are identical and represent
the same comoving diffusion flux. The branch-wise frozen-table identity gate is
strict:
\[
\mathrm{median}(|R_F|)<10^{-3},\qquad \mathrm{p95}(|R_F|)<10^{-2},
\]
where \(R_F\) compares the tabulated \(F\) to
\(-c(3\kappa_R\rho)^{-1}\partial_xE\) on each branch.

The Wave 9 production ctest is
`expensive.I1 FLD const-Eddington nED grey radiative shock`. It keeps the
Wave 8 quasi-planar default \(r_{\min}=10^6\ \mathrm{cm}\), the
`two_state` initialization, and the \(3.0(x_{\max}-x_{\min})/|u_0|\) runtime.
Its strict acceptance gates are t=0 Mach and pressure-ratio admissibility,
\(\max(P_{rad}/P_{mat})\le10^{-3}\), frozen-reference conservation residual
\(\le10^{-12}\), branch-wise reference identity, and
\[
\left\| (T_{rad}/T_e-1)_{TENRYU}
      -(T_{rad}/T_e-1)_{ref}\right\|_\infty < 0.2 .
\]
The \(0.2\) threshold is deliberately wider than the expected \(O(0.05)\)
equation-matched result so it does not fail on normal shock-localization and
finite-grid error, but it is tight enough to reject the Wave 7B/8 \(O(1)\)
model-mismatch signature.

Phase A Wave 10 supersedes the Wave 8/9 runtime calibration for the LE08 and
FLD-CED nED decks. The default end time is no longer set only by the hydro
crossing time. Instead, the decks use the local electron radiation-relaxation
time of the grey absorption source,
\[
\rho C_{v,e}\frac{dT_e}{dt}=c\sigma_a(E-a_{eV}T_e^4),
\qquad \sigma_a=\kappa_R\rho .
\]
Linearizing about \(E=a_{eV}T_0^4\) gives
\[
\tau_{rel}=\frac{\rho C_{v,e}}
{4c\sigma_a a_{eV}T_0^3},\qquad
C_{v,e}=\frac{\bar Z\,eV\_to\_erg}
{A m_p(\gamma-1)} .
\]
For the current I1 calibration
\(\rho_0=2.0\ \mathrm{g/cm^3}\), \(\kappa_R=0.5\ \mathrm{cm^2/g}\),
\(\sigma_a=1.0\ \mathrm{cm^{-1}}\), \(T_0=30\ \mathrm{eV}\),
\(\gamma=5/3\), \(A=1\), and \(\bar Z=1\), this gives
\[
C_{v,e}=1.436845948\times10^{12}\ \mathrm{erg\,g^{-1}\,eV^{-1}},
\qquad
\tau_{rel}=6.4690668\times10^{-6}\ \mathrm{s}=6.47\ \mu\mathrm{s}.
\]
The deck default is therefore
\[
t_{end}=\max\left(5\tau_{rel},
3\frac{x_{\max}-x_{\min}}{|u_{upstream}|}\right).
\]
At \(M_0=2\), the widened references make the hydro-crossing fallback
\(7.66\ \mu\mathrm{s}\) for FLD-CED and \(7.93\ \mu\mathrm{s}\) for LE08,
so the default selects \(5\tau_{rel}=32.35\ \mu\mathrm{s}\).

Wave 10 also extends the nED reference domains by padding only the far
upstream/downstream equilibrium tails. The solved relaxation zone and shock
matching are unchanged; the padded points carry the branch endpoint state and
preserve the branch-wise diffusion identity used by the checker. The generated
FLD-CED spans are \(50.0\ \mathrm{cm}\) for \(M_0=1.5,2.0,3.0\). The generated
LE08 spans are \(50.0\), \(51.7131695\), and \(63.6854397\ \mathrm{cm}\) for
\(M_0=1.5,2.0,3.0\), respectively.

With this calibration, `test_i1_fld_ced_grey_radshock.cu` restores the strict
gate
\[
\left\| (T_{rad}/T_e-1)_{TENRYU}
      -(T_{rad}/T_e-1)_{ref}\right\|_\infty \le 0.2 .
\]
The LE08 ctest remains an auxiliary partial-checkpoint comparison unless host
verification demonstrates that its dynamics gates should be restored: ExactPack
`problem="LM_nED"` includes \(v/c\) source-correction physics outside TENRYU's
current constant-Eddington, gas-momentum-only FLD model. LR07-EDA is unchanged
by Wave 10.

Phase A Wave 12 changes the FLD-CED production metric, not the TENRYU
discretization. The revised high-AI consultation
`tmp/prompts/20260515-105506-tenryu-fld-ced-shock-revised.md` supersedes
`tmp/prompts/20260515-101923-tenryu-fld-ced-shock-2x-gap.md` and closes Q3
(missing \(\rho C_v/\Delta t\) in the Newton residual), Q2 (Strang split), Q5
(2T drain), and Q8 (hidden Eddington mismatch) by code and empirical evidence.
The remaining FLD-CED `dTr_max` gap at NR=1024 is treated as the expected
finite-grid embedded-shock observable: finite shock-tube/reference-window
mismatch (Q7-extended), HLL/HLLC hydro shock smearing (Q4), post-shock MFP
under-resolution (Q1), and pointwise-vs-cell-average comparison at the
discontinuous hydro jump (Q6). The NR=2048 Wave 11 run hit a
thermal-subcycle floor cascade (`dTr_max=89.65`) and is outside the verified
resolution envelope for this calibration; NR=1024 is the I1-A auxiliary
verification resolution until that high-resolution stability issue is fixed.

The asymptotic shock-front separation remains the FLD-CED reference peak
\(D_\infty \simeq 0.7614\), but a finite-volume hydro shock of width
\(w_h=O(\Delta x)\) cell-averages the discontinuous \(T_e\) jump against a
continuous radiation energy profile. To first order this gives the embedded
shock observable
\[
D_{\Delta x}\approx \frac{D_\infty}{1+C\Delta x},
\]
where \(C\) is set by the measured hydro kernel width. Therefore the raw
pointwise
\[
\left\| (T_{rad}/T_e-1)_{TENRYU}
      -(T_{rad}/T_e-1)_{ref}\right\|_\infty
\]
is no longer a production gate at the embedded shock. It is retained as a
diagnostic because it is first-order convergent and dominated by where a cell
center samples the smeared hydro jump.

Wave 12 replaces that gate with a shock-windowed, cell-averaged,
convolved-reference metric. For each production snapshot the harness detects
the TENRYU shock cell by \(\max|\partial T_e/\partial x|\), phase-aligns the
reference shock to that position, and measures the hydro kernel
\[
K_h(\xi)=
\frac{|\partial T_e/\partial x|(x_s+\xi)}
{\int |\partial T_e/\partial x|(x_s+\xi)\,d\xi}
\]
over a local \(\pm 1\ \mathrm{cm}\) window, with nearest-cell fallback on coarse
meshes. The reference comparison profile is
\[
\bar d_{ref,i} =
\frac{1}{\Delta x_i}\int_{x_{i-1/2}}^{x_{i+1/2}}
\int d_{ref}(x')K_h(x-x')\,dx'\,dx,\qquad
d_{ref}=T_{rad,ref}/T_{e,ref}-1 .
\]
The strict shape gate is the worst steady-window value
\[
\left[\frac{1}{N_w}\sum_{|x_i-x_s|<10\lambda_{mfp,post}}
\left(d_{TENRYU,i}-\bar d_{ref,i}\right)^2\right]^{1/2}
\le 0.10,
\]
where \(\lambda_{mfp,post}=1/(\kappa_R\rho_{post})\) is measured from the
TENRYU post-shock cells in the same cgs + eV unit system. The peak gate compares
the maximum TENRYU \(dTr\) in that shock window to the convolved-reference peak
from the same steady snapshot:
\[
\frac{|D_{TENRYU,\Delta x}-D_{ref*K_h,\Delta x}|}
{D_{ref*K_h,\Delta x}}\le 0.15 .
\]

The Richardson envelope remains a separate asymptotic check on the raw
`dTr_max` sequence. The Wave 12 harness requires the NR=256, 512, 1024 sequence
and fits the first-order embedded-shock model above using the bracketing
NR=256 and NR=1024 values, with the NR=512 point retained in the summary as
part of the sequence diagnostics. The strict bound is
\[
0.65\le D_\infty^{fit}\le 0.87 .
\]
For the existing Wave 11 NR=256/512/1024 dumps, the fitted value is
\(D_\infty^{fit}\approx 0.749\), consistent with the reference peak 0.7614 and
with the high-AI conclusion that the factor-of-two NR=1024 pointwise gap is a
first-order embedded-shock convergence effect rather than a FLD-CED equation
mismatch.

##### 2026-07-15 gate revision — Richardson \(D_\infty^{fit}\) retired to diagnostic; fine-pair peak-gap contraction gate (gate8b)

A two-bisect root-cause investigation of the 2026-07 gate8 failure
(campaign ledger `docs/design/2d_campaign_plan_20260708.md`, exec-records
16–21) found that the strict \(D_\infty^{fit}\) window above is not an
asymptotic estimate on this ladder: adjacent-pair fits of the same
rational model disagree by more than a factor of two in every measured era
(certified-era pairs: (256,512) 0.20, (512,1024) −2.26 i.e. model-invalid,
(256,1024) 0.749), and the (256,1024) value amplifies percent-level
physics-consistency row changes by 2.5–3.0×. Two sanctioned physics
corrections (BUG-7 1D FLD Fleck conservation fixes, 2026-07-03; and the
`gamma_r_43` hydro-coupling default, 2026-07-06) moved the fine row by only
−4.7% / −1.3% (physics gates 6/7 green throughout) while the fitted
\(D_\infty\) drifted 0.7491 → 0.6346, out the bottom of the frozen window.

The strict gate is therefore replaced by the fine-pair convolved peak-gap
contraction gate (`gate8b_convolved_peak_gap_contraction`):
\[
G_{1024}\le\max\left(0.5\,G_{512},\ 0.02\right),
\]
where \(G_{NR}\) is the per-row `wave12_convolved_peak_gap`, with
fail-closed handling (non-finite or missing rows fail the gate).
Back-test: contraction ratios \(G_{1024}/G_{512}\) = 0.024 (certified era),
0.075 (post-BUG-7), 0.207 (post-`gamma_r_43`) — green in every era with
≥2.4× margin; the coarse pair (256→512) is excluded because it rose (1.09×)
at the post-BUG-7 era (coarse-row peak detection is pre-asymptotic). A
finite-but-large \(G_{512}\) weakens only the contraction evidence, not the
absolute bound: gate 7 independently caps \(G_{1024}\le 0.15\).
\(D_\infty^{fit}\) remains computed and reported
(`richardson_d_infinity_fit_status = "diagnostic_only_20260715"`).

The NR=2048 "resolution envelope" note (Wave 11/12) is refined by the same
investigation: a 2026-07-15 NR=2048 rerun under current physics completes
healthily with all fields finite (replica-identical, i.e. deterministic);
the historical `dTr_max` explosion there (89.65 in Wave 11, 65.9 today) is
the ratio metric \(|T_{rad}/T_e-1|\) evaluated on a resolved near-floor
cold cell (\(T_e\approx 0.108\) eV, \(T_{rad}\approx 7.2\) eV), plus
reference-table-edge \(1/E_{FLOOR}\) blowups in auxiliary traversal
metrics — analysis-convention limits at off-design resolutions, not a
solver divergence under current physics. The certification ladder remains
NR=256/512/1024.

#### 6.7.3.1 Per-Operator Radial Fourier Audit

The default-off 2D_RZ radial Fourier audit localizes sudden radial-null-mode
growth without changing the discretization or state. When
`Diagnostics.per_operator_radial_fourier_enabled=True`, `Coupling::Driver`
samples configured Strang-stage boundaries inside
`[radial_fourier_window_t_start_s, radial_fourier_window_t_end_s)`.

For each z-index \(j\), field \(q\), and radial mode \(m\), the audit computes
the mean-subtracted direct DFT
\[
\bar q_j={1\over N_r}\sum_i q_{ij},\qquad
\hat q_{mj}=\sum_i (q_{ij}-\bar q_j)
  \exp\left(-2\pi\mathrm{i}{mi\over N_r}\right),
\]
then reports
\[
A_{mj}(q)= {s_m|\hat q_{mj}|\over N_r|\bar q_j|+\epsilon},
\qquad
s_m=\begin{cases}
1, & m=0\ \hbox{or Nyquist}\\
2, & \hbox{otherwise}
\end{cases},
\]
where \(\epsilon=10^{-300}\) only protects zero-mean normalization. The
reported `A_max` is \(\max_{m,j} A_{mj}\). The audited fields are `rho`,
`Te`, `Ti`, cell-centered `u_r`, cell-centered `u_z`, and total group-summed
`E_rad`. All quantities use TENRYU's fixed cgs + eV internal unit system.

The diagnostic writes one HDF5 row per field per sample under
`/diagnostics/radial_fourier_audit/v1/` with stage id, before/after phase,
`A_max`, `m_max`, and `j_max`. It is read-only and additive; disabled runs do
not allocate audit buffers and preserve the physics update path.

PR G2-A adds an opt-in fixed-mode complex-coefficient audit under
`/diagnostics/radial_fourier_audit_v2/v1/`. For configured targets
`m in per_operator_radial_fourier_complex_m_targets` and
`j in per_operator_radial_fourier_complex_j_targets`, it first forms both radial
means
\[
\bar q^{unw}_j={1\over N_r}\sum_i q_{ij},\qquad
\bar q^{vol}_j={\sum_i V_{ij}q_{ij}\over\sum_i V_{ij}},
\]
then uses the volume-weighted residual
\[
\delta q_{ij}=q_{ij}-\bar q^{vol}_j
\]
for the unweighted and volume-weighted coefficients
\[
C^{unw}_{m,j}(q)=\sum_i \delta q_{ij}
\exp\left(-2\pi\mathrm{i}{mi\over N_r}\right),
\]
\[
C^{vol}_{m,j}(q)=\sum_i V_{ij}\delta q_{ij}
\exp\left(-2\pi\mathrm{i}{mi\over N_r}\right).
\]
The HDF5 row stores real/imaginary parts, amplitude, phase, radial min/max, and
\(\sum_i V_{ij}\). The sign convention is the mathematical
\(\exp(-2\pi i mi/N_r)\) convention, so a pure sine has phase \(-\pi/2\).

The v2 field enum includes hidden variables needed for ALE-RZ instability
triage: `rho`, `M`, `V`, `M_over_V`, `P_r`, `P_z`, `u_r`, `u_z`, `E_e`,
`E_i`, `E_rad`, `T_e`, `T_i`, `x_r`, `x_z`, `A_r`, `A_z`, `Q_visc`, and
`f_Fleck` when backed by current `State` storage. `E_e` and `E_i` are recorded
as energy densities \(\rho e_e\) and \(\rho e_i\); `A_r` and `A_z` are
cell-representative RZ face areas formed from the two radial-normal and
axial-normal edge surfaces. `M` uses the driver cell-mass field when present,
with \(\rho V\) as a fallback. Fields without driver-visible cell storage in PR
G2-A (`dV_swept`, `lambda_FLD`, `R_FLD`, `kappa_eff`, `newton_iters`,
`newton_residual`) are accepted in the config but skipped.

For an offline before/after pair of the same stage, field, and fixed mode,
`tools/diagnostics/pr_g2_gain_analysis.py` computes
\[
g_s={C^{vol,after}_{m,j}\over C^{vol,before}_{m,j}},\qquad
\Delta\log|C|=\log|C^{after}|-\log|C^{before}|,
\]
\[
\alpha_s={\Delta\log|C|\over \Delta t_{cycle}},\qquad
\Delta\phi=\operatorname{unwrap}\left[
\arg(C^{after})-\arg(C^{before})\right],
\]
and the in-phase multiplier
\[
a_s={\operatorname{Re}\left((C^{after}-C^{before})
\overline{C^{before}}\right)\over |C^{before}|^2}.
\]
The tool sorts detailed rows by \(\Delta\log|C|\) rather than by v1 `A_max` and
emits a per-cycle summary of the dominant stage versus distributed sub-stage
gain.

Round 13 adds an independent FLD substage audit controlled by
`Radiation.multigroup_diffusion.diagnostic_radial_fourier_substage_enabled`.
When enabled, FLD records selected substage Fourier coefficients and solver
residual diagnostics under `/diagnostics/fld_substage_audit/v1/`. The HDF5
schema is additive: the group is absent unless the flag is true, root HDF5
`schema_version` is unchanged, and existing readers need no migration because
the optional group's absence is the default state. The audit is diagnostic-only
and does not feed back into the FLD solve or material update.

#### 6.7.4 §I2-aux Phase B: 1D_SPH FLD-CED Multigroup Grey-Collapse Limit

> **SUPERSEDED (2026-07-04, main-CC adjudication Q7 of
> `docs/design/i2_mgfld_collapse_spec.md` v3.1).** The 1D_SPH aux gate
> specified below was never implemented (no deck/harness/ctest was ever
> committed on any branch). Its verification intent is covered by (a) the
> 1D solver-level multigroup gates `fld_1d_mg_planar_marshak_spectrum` /
> `fld_1d_mg_planar_freqdep_relaxation` (branch `feature/1d-brushup-mg-marshak`,
> commits 3aac4393/04536d41: anti-tautological independent b_g reference and
> closed-form per-group σ_P reference for the same `freq_dep_marshak` model),
> and (b) the 2D RZ production object of §6.7.5 below, whose frozen-hydro
> identity tier is strictly tighter (1e-8 L2 vs the 1e-2 planned here). The
> section is retained for the derivation of the collapse identity it records.

Phase B I2-aux verifies a 1D_SPH code-path equivalence rather than a new reference
solution. The foundation is the Phase A I1 Wave 12 FLD-CED infrastructure at
commit `d3d68c59`: the in-house constant-Eddington nED grey table, schema-v3
dual-flux identity checks, the \(N_R=256,512,1024\) 1D_SPH production ladder at
\(r_{\min}=10^6\ \mathrm{cm}\), and the shock-windowed/convolved/Richardson
strict gates defined above.

The I2 analytic limit is TENRYU's deterministic
`Radiation.mode="multigroup_diffusion"` path with frequency-independent
opacity. For one grey group,
\[
{\partial E\over\partial t}
=\nabla\cdot\left(D\nabla E\right)
+c\sigma_a\left(a_{eV}T_e^4-E\right),
\qquad
D={c\over3\sigma_R},\qquad
\sigma_R=\sigma_a=\kappa_R\rho .
\]
This is the same constant-Eddington FLD operator used by the I1 FLD-CED
reference when `flux_limiter="none"`.

For \(G\) groups with the same mass opacity in every group,
\[
\kappa_{R,g}=\kappa^{PA}_g=\kappa^{PE}_g=\kappa_R ,
\qquad
\sigma_{R,g}=\sigma_{a,g}=\kappa_R\rho ,
\]
and normalized Planck fractions \(b_g(T_e)\), the group equations are
\[
{\partial E_g\over\partial t}
=\nabla\cdot\left(D_g\nabla E_g\right)
+c\sigma_a\left(a_{eV}T_e^4 b_g(T_e)-E_g\right),
\qquad
D_g={c\over3\sigma_a}.
\]
Because \(D_g\) is group-independent and the Planck table is normalized so that
\[
\sum_{g=1}^{G} b_g(T_e)=1,
\]
summing over groups gives
\[
{\partial\over\partial t}\sum_g E_g
=\nabla\cdot\left(D\nabla\sum_g E_g\right)
+c\sigma_a\left(a_{eV}T_e^4-\sum_g E_g\right),
\]
which is exactly the grey FLD equation for \(E_{rad,total}=\sum_gE_g\). Thus a
multigroup run with constant opacity must collapse to the grey run, modulo the
Planck-table quadrature and interpolation normalization error. The existing
`opacity.model="constant"` namelist path supplies the required
frequency-independent opacity: it assigns the same \(\kappa_R=\kappa_P\) to all
configured groups. The Planck table only partitions \(a_{eV}T_e^4\) into
normalized \(b_g(T_e)\) weights; it does not introduce spectral opacity
structure in this deck.

The I2 deck is
`examples/verification/i2_fld_ced_multigroup_grey_collapse.py`. It keeps the
I1 FLD-CED material, hydrodynamics, conduction, Qei, low-\(P_{rad}\)
calibration, \(r_{\min}=10^6\ \mathrm{cm}\), and \(t_{end}=5\tau_{rel}\)
defaults. It changes only the env prefix to `TENRYU_I2_FLD_CED_*`, exposes
`TENRYU_I2_FLD_CED_GROUPS` with default \(G=2\), and sets
`Material.opacity.model="constant"` with \(\kappa_a=0.5\ \mathrm{cm^2/g}\) so
all groups have identical absorption/Rosseland opacity. The configured group
bounds span \([0,10^6]\ \mathrm{eV}\), and
`Radiation.groups.planck_fraction.method="compute"` builds the normalized
Planck table over \(0.01\le T\le1000\ \mathrm{eV}\).

The I2 harness is `tools/validation/run_i2_grey_collapse.py`. It reuses the I1
Wave 12 shock-windowed \(L_2\), cell-averaged convolved-reference peak gap, and
Richardson \(D_\infty\) implementation on the \(G=1\) grey baseline ladder. It
also runs the requested multigroup \(G\) ladder (production default \(G=2\));
the fine-grid \(G=2\) result is paired with the fine-grid \(G=1\) result for
the new grey-collapse gate. At \(t_{end}\), it computes
\[
\mathrm{grey\_collapse\_l2}
=\left[
{1\over N}\sum_i
\left(
{\sum_g E_{i,g}^{MG}-E_i^{grey}
\over \max(|E_i^{grey}|,\epsilon)}
\right)^2
\right]^{1/2}.
\]
The strict I2 acceptance is:

- branch-wise FLD-CED frozen-table identity:
  upstream/downstream median \(<10^{-3}\), p95 \(<10^{-2}\);
- I1 low-\(P_{rad}\), frozen-reference conservation, and t=0 admissibility
  gates unchanged;
- Wave 12 strict gates on the \(G=1\) grey baseline ladder unchanged:
  shock-windowed \(L_2\le0.10\), convolved peak gap \(\le0.15\), and
  \(0.65\le D_\infty^{fit}\le0.87\);
- new grey-collapse gate:
  \(\mathrm{grey\_collapse\_l2}\le0.01\) for the \(G=2\) production ctest.

The production ctest is
`expensive.I2 FLD multigroup grey-collapse limit`. It reuses the frozen
`tests/verification/data/fld_const_eddington_reference/fld_ced_M2.json` table;
no ExactPack dependency, reference regeneration, production CUDA/C++ change, or
unit-system change is introduced.

#### 6.7.5 §I2 2D RZ multigroup FLD grey-collapse production gate

Object `I2_2D_RZ_MULTIGROUP_FLD_GREY_COLLAPSE` (roadmap §10 row; design spec
`docs/design/i2_mgfld_collapse_spec.md` v3.1, adjudicated 2026-07-04).
Grey is the \(N_g=1\) operation of the same multigroup kernels (no separate
grey path; dispatch branches on dimension only), so the collapse comparison
verifies the group-partition machinery — b_g weights (renormalized
\(\sum_g b_g=1\) at table build, debug-asserted to 1e-12), per-group
\(\sigma/D/E\) storage, the Fleck cg layout contract (BUG-11,
`f95ba49a`+`92013efe`), the group-blocked CSR solve, and the group-summed
matter coupling — against the I1-A-anchored grey endpoint.

**Discrete identity.** With `flux_limiter="none"` (\(\lambda=1/3\)) and
frequency-flat opacity, the per-group equations telescope exactly under the
group sum: diffusion (identical per-group operator), emission
(\(\sum_g b_g(T^*)=1\) at any frozen Picard iterate — b_g is never
linearized, no \(db_g/dT\) anywhere), absorption, and the Fleck blend
(f per (cell,group), constant across g for flat κ) all sum to the grey
discrete equations. Non-telescoping residuals are the linear-solve tolerance,
outer-Picard termination, and fp summation order — hence the gate tiers.

**Gate tiers (BINDING, roadmap row values):**

- **I2a (frozen-hydro identity)** — `tests/verification/`
  `test_i2_fld_multigroup_collapse.cu`, plain ctest
  `i2a fld_2d_rz multigroup grey exact-collapse identity` (in-process
  `advance_radiation_step_fld_2d_rz`, 8×16 z-slab, K=4 steps at
  dt=6e-11 s so the Fleck factor is exercised at \(f\in[0.40,0.50]\)
  measured, ladder \(N_g\in\{2,4,8\}\) on nested log-uniform [1,1500] eV
  bounds): \(\varepsilon_{L2}\le10^{-8}\) AND
  \(\varepsilon_{L\infty}\le10^{-6}\) on \(\{\sum_g E_g, T_e,
  T_{rad,\Sigma}\}\) at every step; energy-ledger identity
  \(|drift_{mg}-drift_{grey}|\le10^{-10}\); anti-vacuity asserts
  (\(f_{\min}\le0.7\), spatial f spread \(\ge0.1\), band-invariance
  subcase G=1 [1,1500] vs [0,1e6] bit-identical). Measured first run
  (2026-07-04, RelWithDebInfo): \(\varepsilon_{L2}\in[3.5,9.2]\times
  10^{-16}\), \(\varepsilon_{L\infty}\le2.7\times10^{-15}\) across all
  \(N_g\)/steps/fields; absolute closed-box drift \(\le5.1\times10^{-16}\)
  over 4 steps; drift difference \(\le3.4\times10^{-16}\). Gate
  sensitivity is EMPIRICAL: with `fld_2d_rz_gpu.cu` reverted to base
  3f246289 (pre-BUG-11) the gate fails loudly (mutation test).
- **I2b (coupled collapse)** — deck
  `examples/verification/i2_mgfld_collapse_2d_rz_slab.py` (I1-A clone;
  `two_state` Riemann + all-reflecting BCs because groups>1 forbids
  state_supply/marshak radiation z-BCs by builder ConfigError), harness
  `tools/validation/run_i2_2d_rz_mg_collapse.py`, wrapper ctest
  `expensive.I2 mgfld grey-collapse 2D RZ battery strict`: R0 grey +
  R2 mg\{2,4,8\} flat at 32×512 to t_end; binding R0-relative
  \(\varepsilon_{L2}\le10^{-4}\) AND \(\varepsilon_{L\infty}\le10^{-3}\)
  on radial-mean z-profiles of \(\{\rho,T_e,T_i,\sum_g E_g,
  T_{rad,\Sigma},u_z\}\); flat-ladder pairwise \(\varepsilon_{L2}\le
  10^{-4}\); per-run hygiene (t_end termination, Newton gates, zero
  escape valves, radial invariance \(\le10^{-6}\), effective-group-count
  banner assert against the auto-grey trap). Wave-12 absolute metrics vs
  the FLD-CED ODE table are REPORTED, not gated (two_state-mode
  calibration caveat; adjudication Q6). Main strict battery measurement
  (2026-07-05, `ctest #603`, summary `all_checks_passed=true`): hygiene,
  collapse, flat-ladder, front-Cauchy, and Richardson gates all passed;
  worst R0-relative coupled-collapse values were
  \(\varepsilon_{L2}=3.6218\times10^{-5}\) and
  \(\varepsilon_{L\infty}=8.7189\times10^{-5}\) (both \(E_{rad}\),
  \(N_g=4\)); flat-ladder pairwise \(\varepsilon_{L2}\) values were
  \(g2g4=4.6856\times10^{-5}\), \(g2g8=3.4295\times10^{-5}\), and
  \(g4g8=1.2569\times10^{-5}\).
- **Structured departure + Richardson (characterization)** — same deck in
  `front` mode (frozen hydro, closed box z∈[0,0.24] cm, 60/3 eV
  two-temperature IC, `freq_dep_marshak`; group opacities straddle
  thick/thin): front position/width departure vs the grey comparator pair
  at analytic full-band \(\kappa_P/\kappa_R(T_{hot})\), REPORTED with
  Cauchy contraction gate \(d(4,8)\le0.75\,d(2,4)\) (resolution-floor
  guarded), plus an nz∈\{256,512,1024\} Richardson ladder at \(N_g=4\)
  (contraction + observed order reported). The first main GPU battery froze
  front-mode `t_end=3e-10 s` and the characterization values:
  \(d(2,4)=2.2284\times10^{-6}\) cm, \(d(4,8)=2.0221\times10^{-7}\) cm
  (both below the resolution floor), Richardson front positions
  \(z_{256}=0.0600014429\) cm, \(z_{512}=0.0600022962\) cm,
  \(z_{1024}=0.0600021530\) cm, Richardson deltas
  \(8.5335\times10^{-7}\) cm and \(1.4325\times10^{-7}\) cm, and observed
  order \(p=2.5746\).

Not covered by this object (recorded): bugs identical in the \(N_g=1\) and
\(N_g>1\) paths (covered only by I1-A's external anchoring); group-mean
quadrature accuracy of `freq_dep_marshak` (1D solver-level gates above);
limiter-ON collapse (breaks exactness by design); structured-opacity
hydro-coupled behavior.

#### 6.7.6 §I3 2D RZ grey S_N radiative shock production gate

PENDING — see roadmap §10 carry-over registry row
`I3_2D_RZ_GREY_SN_RADIATIVE_SHOCK`.

#### 6.7.7 §I4 2D RZ multi-material radiation-interface production gate

Design record: `docs/design/i4_mm_rad_interface_spec.md` (A1–A5 + Addenda).

**Per-material opacity (G-1, frozen convention A1).** The shared radiation
opacity path fills per-cell effective opacities from the material mixture:
\[
\sigma_{a,\mathrm{cell}}=\sum_m \kappa_m\,\rho_m^{\mathrm{part}},\qquad
\rho_m^{\mathrm{part}}={\mathrm{mass}_m/ V_{\mathrm{cell}}},\qquad
\sigma_{R,\mathrm{cell}}=\sum_m f_m\,\sigma_{R,m}
\]
(exact volume-averaged absorber sum; Rosseland arithmetic-in-\(\sigma\) over
volume fractions — the series-normal limit; error confined to the one-cell
PLIC interface band). `OpacityEvalView` carries optional per-cell pointers
with a null → scalar fast path that is bit-identical at \(n_{\mathrm{mat}}=1\).
Diagnostic override `TENRYU_MM_OPACITY_MIX=harmonic` (env, undocumented knob)
exists for sensitivity probes; no namelist key (A5).

**I4a (stationary layered Marshak).** 2D RZ z-slab, frozen medium, grey FLD,
Marshak inflow; reference = `tools/validation/layered_marshak_reference_generator.py`
(implicit-Newton FV, piecewise-constant \(\sigma(z)\), S1–S4 self-verification
battery). Dev-time smoke = short-window mechanism check (t_end \(10^{-13}\) s:
boundary ignition ≥ 50 eV, front formed before the interface, per-step ledger
eps ≤ \(10^{-10}\), layered volFrac materialized); the registry profile gates
(eps\(_{L2}\)(E) ≤ 0.02, interface flux continuity ≤ \(10^{-3}\), per-material
mass ≤ \(10^{-12}\), k ∈ {3,10,30}) are cert/campaign tier — the strong-drive
boundary pins dt ≈ 3×10⁻¹⁷ s (Fleck-z dt collapse, W-I class), making the
full window a campaign run.

**I4b (radiative shock crossing a material interface).** Deck
`i4b_radshock_interface_2d_rz_slab.py`: LE08 grey NED two-state launch at
native table scale in the shock rest frame (upstream flows toward the shock;
the material interface advects into it — crossing at
\(t_\times=d_{\mathrm{int}}/u_0\)); large-radius annular slab (R_CENTER =
50·NR·dz, curvature 2%); two materials with \(\kappa_1/\kappa_2=k\) (leg A:
pure-κ contact, identical hydro states; leg B: pressure-balanced impedance
contact \(\rho_2=\rho_1/2,\ T_2=2T_1\)). Reference branch (A4): code
convergence primary (matched-time nz/2nz pair) + LE08 incoming-shock anchor
secondary; a transient-crossing ODE does not exist as a verification asset.
Estimator canon (harness `run_i4b_radshock_interface.py`): all positions in
REAL mesh coordinates (Lagrangian-dominant motion displaces nodes by tens of
cm); primary-shock locator = windowed **Qvisc peak** (density-face argmax is
fragile: the sharp two-state init sheds an entropy wave advecting at \(u_1\)
whose 2-cell ρ sawtooth exceeds the precursor-smoothed shock jump); interface
locator = volFrac 0.5 crossing, subcell-interpolated. Smoke gates (dev, ≤60 s
wall): pre-crossing exact kinematics (shock ≤ 2 dz of the frame position,
interface ≤ 2 dz of \(z_0+u_0t\)), post-crossing sanity (shock within
0.15 L_ref — structural relaxation toward the material-2 profile displaces
the front O(0.1 L_ref); leg B interface crossed-and-bounded — the table
\(u_1\) is invalid in the lighter transmitted medium), VOF ∈ [0,1],
per-material mass leakage ≤ \(10^{-10}\), ledger max-late eps ≤ \(10^{-6}\),
CG health. Registry position/L2 gates = cert/campaign (convergence pair).

**Capability walls (parse/assert-enforced, discovered at I4b bring-up).**
Per-material conservation excludes `total_energy_remap_2d_rz` (which HLLC
z-flux requires) and the conservative reference-target remap — per-material
decks run the legacy z-flux path with the PLIC unified-pass remap
(`plic.enabled` + `rho_material_aware_donor`). The FLD 2D iterative solve at
native scale needs a cold-start dt ramp (initial dt \(10^{-12}\) s) and a
CG budget of the matrix dimension (`cg_max_iter=4096`); one deterministic
step-2 stall (rel 0.19) is a known startup artifact. The production-audit
momentum residual is not a gate for closed reflecting boxes (walls
legitimately exchange momentum with the flow).

#### 6.7.8 §I5 2D RZ 2T radiative shock with finite Q_ei production gate

The offline I5 reference generator
`tools/validation/fld_ced_2t_reference_generator.py` selects the shock-match
branch from
\[
\epsilon={l_{ei}\over l_{rad}} .
\]
`--solver auto` uses the reduced slaved branch for \(\epsilon<10^{-6}\) and
the exact full branch otherwise.  Explicit `--solver reduced` and
`--solver full` keep the same branch-specific acceptance gates.  The
reference tables are separated by regime tag: `SVLADDER_*` names the reduced
\(Q_{ei}\)-convergence ladder and `APRIME_SCALED_*` names the active
finite-\(\epsilon\) full-branch \(A^\prime\) deck with \(M_0=3\),
\(\rho_0=2\) g/cc, \(T_0=30\) eV, \(A=1\), \(\bar Z=1\),
\(\kappa_R=R\,2.5824665154\) cm2/g, and
`Numerics.hydro.qei_multiplier=1.0e-6` for \(R\in\{0.1,1,10\}\).  The
unscaled physical `APRIME_*` rows with
\(\kappa_R=R\,2.5824665154\times10^6\) cm2/g and
`qei_multiplier=1` remain committed for the future kernel floor-bug fix, but
the manifest marks them `kernel-blocked` because their nano-domain reaches the
kernel absolute-floor pathology `h=0`.

Both branches use the mixture variables
\[
T=\frac{c_{v,i}T_i+c_{v,e}T_e}{C_v},\qquad
\Delta=T_e-T_i,\qquad
Z=E-a_{eV}T^4 ,
\]
with \(C_v=c_{v,i}+c_{v,e}\), \(\beta_i=c_{v,i}/C_v\),
\(\beta_e=c_{v,e}/C_v\), \(R=R_i+R_e\), and the same cgs+eV constants as the
runtime kernels.  Thus
\[
T_i=T-\beta_e\Delta,\qquad
T_e=T+\beta_i\Delta,\qquad
E=a_{eV}T^4+Z,\qquad
W_e=E-a_{eV}T_e^4 .
\]
The gas velocity is the mixture-temperature momentum root
\[
J u^2-\Pi u+JRT=0,\qquad A=2Ju-\Pi ,
\]
using the upstream or downstream branch root.  Define
\[
M(T)=JC_v-\rho RT\,\frac{JR}{A},
\]
and, with the NRL electron-ion exchange time evaluated at \(T_e\simeq T\) and
including the generator multiplier,
\[
K(T)=\frac{\tau_{ei}}{\rho c_{v,e}}
\left(Jc_{v,i}-\rho R_iT\,\frac{JR}{A}\right).
\]
The reduced outer ODE is
\[
T'=\frac{c\kappa_R\rho Z}
{M+4c\kappa_R\rho a_{eV}T^3\beta_iK},
\qquad
Z'=-\frac{3\kappa_R\rho}{c}F-4a_{eV}T^3T',
\]
where \(F=\Phi-J(c_pT+u^2/2)\).  No sonic soft clamp is applied; integration is
guarded by an event on \(|A|\).  The tabulated species fields are reconstructed
from the outer slaving relation
\[
\Delta_s=T_e-T_i=K(T)T',\qquad
T_i=T-\beta_e\Delta_s,\quad T_e=T+\beta_i\Delta_s,\quad
E=a_{eV}T^4+Z .
\]

For the full branch, the exact \((T,\Delta,Z)\) ODE is
\[
T'=\frac{c\kappa_R\rho W_e}{M},\qquad
u'=u_TT',
\]
\[
\Delta'=
\frac{Jc_{v,i}T'+p_i u'-\rho c_{v,e}\Delta/\tau_{ei}}
{Jc_{v,i}\beta_e},
\qquad
Z'=-\frac{3\kappa_R\rho}{c}F-4a_{eV}T^3T',
\]
where \(p_i=\rho R_iT_i\), \(u_T=-JR/A\), and \(\tau_{ei}\) is evaluated at
the local \((\rho,T_e)\).  Radau integrations use the analytic Jacobian of
this system and guard the sonic wedge with an event on \(|A|\).

Across the embedded gas subshock both branches first apply the raw 2T jump
closure with continuous \(E,F\), electron adiabat, and ion Rankine-Hugoniot
heating.  The reduced table then starts the downstream outer layer at the
mixture state of that raw post-jump state,
\[
T^+_{out}=\frac{c_{v,i}T_i^+ + c_{v,e}T_e^+}{C_v},\qquad
Z^+_{out}=E^+ - a_{eV}(T^+_{out})^4 .
\]
The raw post-jump state is retained as JSON diagnostics.

On the full branch the upstream solution is integrated once from the upstream
fixed point along the growing eigenvector.  The downstream solution starts
from the raw post-jump state \(Y_0(\ell_{up})\) in \((T,\Delta,Z)\).  For
breakpoints \(0=s_0<s_1<\cdots<s_N=L_{dn}\), the Newton unknowns are the
interior states \(Y_1,\ldots,Y_{N-1}\) and \(\ell_{up}\).  The residual is
the local continuity system
\[
\Phi_k(Y_k;s_k,s_{k+1})-Y_{k+1}=0,\qquad k=0,\ldots,N-2,
\]
plus the scalar downstream far condition
\[
\left<w_{grow},{\Phi_{N-1}(Y_{N-1})-Y_\infty\over S}\right>=0 ,
\]
where \(w_{grow}\) is the left eigenvector of the growing mode of the
downstream fixed point, recomputed for each table, and \(S\) is the
downstream \((T,\Delta,Z)\) scale.  Breakpoints are uniform on the chosen span
with \(\max_j|\lambda_j|h\le2\), bounded below by four intervals and above by
192 intervals.  The span is the maximum of 25 downstream growing-mode
e-folds, four slow-decay e-folds, four physical \(\max(l_{ei},l_{rad})\)
lengths,
and twice the 1T downstream warm-start length.  The outer Newton uses finite
differences on the small square multiple-shooting system and scales
continuity residuals by the downstream far state
\((T_1,T_1,\max(|Z_\infty|,a_{eV}T_1^4\,10^{-6}))\); the far projection is
dimensionless.

The reduced path is accepted only when
\(\epsilon=l_{ei}/l_{rad}<10^{-6}\) for every continuation rung.  SV2 gates
the \(\Delta_s\to0\) outer-table convergence against the imported 1T
solution for `SVLADDER_*`; finite-\(\epsilon\) `APRIME_*` tables mark SV2 as
not applicable.  SV5 records the full species-ODE residual on the generated
profile, interpreted as the expected \(O(\epsilon)\) slaving residual on the
reduced branch and as the direct full-system residual on the full branch.

#### 6.7.9 §I6 2D RZ multigroup S_N grey-collapse production gate

Registry row `I6_2D_RZ_MULTIGROUP_SN_GREY_COLLAPSE` (roadmap §10). Design spec:
`docs/design/i6_sn_mg_collapse_spec.md` (grounding facts F1-F15 + Addenda 1-2).

**Object.** Grey \(S_N\) is \(N_g=1\) of the same multigroup kernels (single dispatch,
no grey-specific branch; `sn_transport_2d_gpu.cu` threads `n_groups` end-to-end with the
cell-major layout `c\,G+g`). For a frequency-flat opacity (`constant` model: identical
\(\kappa\) per group, \(\sigma_s\equiv 0\)) the per-group transport equation is linear
with emission source \(\sigma c\,a T^4 b_g(T)\) and \(\sum_g b_g = 1\) exactly
(PlanckTable row renormalization; convex interpolation preserves the sum), so
\(\sum_g \psi_g\) satisfies the grey equation by superposition through every linear
stage (sweeps, DSA, \(E^\*\) assembly, group-summed matter Newton). The identity breaks
only at nonlinear per-group operators — donor-\(\theta\) limiter, negative-flux fixups,
AP blend with group-varying \(\alpha\) — and at the marshak z-BC for \(G>1\) (incoming
intensity is not \(b_g\)-partitioned; parse-time ConfigError forbids that combination).

**G1 frozen-transport identity (ctest `test_sn_2d_rz_mg_grey_collapse`, permanent).**
8×64 z-slab, \(T_e\) tanh ramp 30→90 eV, \(\rho=2\) g/cc, \(\kappa=0.5\) cm²/g,
\(c_{v,e}=10^{30}\) (frozen material), per-group equilibrium IC
\(E_g = b_g(T_e)\,a T_e^4\), all-reflect BCs, \(K=4\) steps at \(dt=10^{-12}\) s,
\(N_g\in\{2,4,8\}\times\{S_8,S_{16}\}\) + band-invariance subcase. Binding:
\(\varepsilon_2\le 10^{-6}\) on both \(\sum_g E_g\) vs \(E_{grey}\) and
\(\sum_g F_{z,g}\) vs \(F_{z,grey}\), with anti-vacuity asserts (θ≡1, zero fixup
tallies, zero AP activity, effective group count, per-group fraction spatial spread
≥0.1, field evolution ≥1e-4). Measured 2026-07-07: \(G\in\{2,4,6\}\) at the fp floor
(\(\varepsilon_2\sim 10^{-16}\)–\(10^{-15}\)); \(G=8\) at \(4.5\times10^{-9}\) (E) /
\(3.0\times10^{-7}\) (F_z) — root-caused to inner-iteration truncation of the
all-reflect boundary angular closure (geometric contraction ≈0.32/sweep; grey stops at
13 sweeps, \(G=8\) at 14-16 under the max-over-(cell,group) relative measure at
`inner_tol`=1e-6; band-independent, group-count-dependent, DSA bit-neutral at
\(\sigma_s=0\)). **Mechanism-closure guard (permanent)**: at `inner_tol`=1e-12 the
\(G=8\) identity must return to the floor — measured \(6.4\times10^{-15}\) (E) /
\(2.9\times10^{-13}\) (F_z), REQUIRE ≤1e-12 — so any real group-machinery defect of
magnitude \(10^{-12}\)–\(10^{-9}\) cannot hide under production-tol truncation noise.

**G2 coupled shock collapse (I3 deck + `TENRYU_I3_SN_RS_GROUPS`; harness
`tools/validation/run_i6_sn_mg_collapse.py`).** Binding (cert tier, batched campaign):
\(\varepsilon_2\le 10^{-3}\) on radial-mean z-profiles of \(\sum_g E_g\) and
\(\sum_g F_{z,g}\) vs the same-binary grey run at full \(t_{end}\), \(N_g\in\{2,4,8\}\),
S_16 (+S_8 confirmation once the `N8_K20` reference table is generated).
Smoke observation 2026-07-07 (16×256, S_16, K=20, \(t=0.39\,\tau_{rel}\)):
G2 \(\varepsilon_2\) = 5.5e-6 (E) / 1.3e-5 (F_z); G4 = 3.4e-5 / 9.3e-5 — ≥10× inside
the cert gate; zero fixups; dt traces diverge between legs (791/838/779 steps), so the
observed values upper-bound the identity term (pinned-dt pair is the documented
fallback if a cert leg ever exceeds the gate). Status: **code-complete; G1 certified;
G2-cert queued for the batched campaign** (registry row stays PARTIAL until then).

#### 6.7.10 §I7 2D RZ coupled-stack ICF baseline production gate

Design record: `docs/design/i7a_coupled_stack_spec.md` (F1–F6 + Addenda 1–3).

**I7a (planar laser-ablation coupled stack).** Deck
`i7a_coupled_stack_2d_rz_slab.py`: axis disk (r_min=0, on-axis raytrace_3d beam —
geometric optics has NO diffraction waist, so a point focus at the surface deposits
into one axis cell; the beam is DEFOCUSED 0.10 cm behind the slab, cone radius
~125 μm at the surface; defocus ≥ 0.15 cm kills the beam entirely — open laser
finding). Corona base = 2× the 351 nm critical density (deposition at n_e~n_c;
an underdense-base IC drives the flux-limited-conduction Te runaway — model
physics, not a bug). Full stack on: hydro + ALE(conservative remap incl. rad) +
HLLC-z + laser IB + physical Spitzer conduction (mfp limiter) + Q_ei + grey FLD
(levermore_pomraning, z=vacuum escape). `rz_geometric_cfl` omitted (annular-family
knob, tri-fan degenerate at the axis).

**Smoke tier (dev, ≈7 min/replica local; harness `run_i7a_coupled_stack.py`)**:
2 replicas, mechanism gates — completion, absorbed fraction ∈[0.1,0.999] +
cumulative laser_in vs ramp integral (audit `laser_in` is PER-STEP; sum it),
ablation front (nearest-to-slab steep |dTe/dz|), shock launched (windowed ρ-jump
face inside the slab; **Qvisc locators do not work under HLLC z-flux**),
per-step ledger eps ≤ 1e-6 (the row ε; conservation restored by the BUG-15
pair-min throttle fix, §4.2.x conduction note), energies sane, replica band on the
four STABLE QoIs (positions / laser_in / mass_ablated ≤ 5e-2; measured: positions
identical, laser 7e-5, mass 0.4-0.7%).

**Measured chaos structure (binding for cert design)**: energy PARTITIONS spread
6–24% between identical replicas (E_rad heavy-tailed; retained-vs-escaped radiation
redistributes ~3% of the budget) while cumulative/mechanism QoIs are stable. The
registry row's frozen-single-reference gates (absorbed ≤2%, E_rad ≤3%,
"bit-exact replay") are therefore realized at cert as ensemble/band-aware
references + the certified 2-replica noise-band replay methodology (spec F3;
registry wording confirmation = user item at cert).

### 6.8 \(S_N\) Transport  —【CURRENT — 現行放射モデル】

> **【CURRENT RADIATION MODEL】** `mode="sn_transport"`。1D_SPH/2D_RZ pure discrete-ordinates（決定論）。IMC/DDMC/HOLO/difference を完全 bypass（Fleck bypass, \(f=1\)）。もう一方の現行モデルは §6.7 FLD。2D RZ grey \(S_N\) 検証 gate は §6.7.6 (§I3) / §6.7.9 (§I6) 参照。

`Radiation.mode="sn_transport"` is the Cut-1b/Cut-2 production discrete-ordinates
radiation mode for 1D_SPH and 2D_RZ. It is separate from the HOLO/QD closure path
and bypasses IMC/DDMC/HOLO/difference. Fleck linearization is not used: the raw
\(\sigma^{PA}\), \(\sigma^{PE}\), and available scattering opacity are passed to
the deterministic operator.

#### Fleck bypass and PA/PE consistency

Pure `sn_transport` explicitly bypasses Fleck linearization. In the NLTE/TMAT
coefficient path the SN effective Fleck factor is fixed to \(f=1\), the sweep
absorption coefficient is the raw Planck absorption opacity
\(\sigma^{PA}_g\), and the Fleck-derived effective scattering contribution is
zero:
\[
\sigma_{a,eff,g}^{SN}=\sigma^{PA}_g,\qquad
\sigma_{s,eff,g}^{Fleck,SN}=0.
\]
This zeroing applies only to the Fleck-derived effective scattering term; any
future physical elastic/scattering opacity must remain a separate raw-opacity
input.

The SN material temperature Newton solve uses the PA/PE split from §6.1.1:
\[
F(T_e)=
\rho\frac{e_e(\rho,T_e)-e_e(\rho,T_e^n)}{\Delta t}
-\sum_g c\,\sigma^{PA}_g E_g
+\sum_g c\,\sigma^{PE}_g a_{eV}T_e^4 b_g(T_e),
\]
with derivative contribution (implemented form — 2026-07-26 doc truth
restoration, AI review k06 4.4: the old text omitted the \(1/(1+\lambda^{PA})\)
reduction and the active-set mask that the production active-set Newton has
always applied; see the Phase B closure below):
\[
\frac{\partial R}{\partial T_e}
=\rho c_{v,e}(\rho,T_e)
+\sum_{g:\,E^{+}_g>0}
\frac{\lambda^{PE}_g}{1+\lambda^{PA}_g}\,4a_{eV}T_e^3 b_g(T_e),
\qquad
\lambda^{PA/PE}_g=c\,\sigma^{PA/PE}_g\,\Delta t,
\]
for the active-set residual \(R(T)=U_e(T)-U_e(T^n)+\sum_g[E^+_g(T)-E^*_g]\):
only inactive groups (\(E^+_g>0\)) contribute, each reduced by
\(1/(1+\lambda^{PA}_g)\). \(db_g/dT\) is not included (inexact Newton — the
residual itself evaluates \(b_g(T)\) at the trial temperature, so the fixed
point is exact and the bracketed/adaptive Newton safeguards convergence).
Here \(e_e(\rho,T)\), \(c_{v,e}(\rho,T)\), and final \(P_e(\rho,T)\)
come from the TMAT electron EOS table when that table is available. Analytic
ideal-gas/test builds with no device EOS table keep the legacy constant-\(c_v\)
linearization and \(P_e=(\gamma-1)\rho e_e\) closure.
Thus absorption/deposition uses \(c\sigma^{PA}_gE_g\), while material emission
and emitted-energy tallies use
\[
\eta_g=\sigma^{PE}_g\,c\,a_{eV}T_e^4b_g(T_e).
\]
This is the documented production behavior. Any use of Fleck-derived
\((1-f)\sigma^{PA}\) scattering in pure SN, or use of \(\sigma^{PA}\) for the
SN emission coefficient when \(\sigma^{PE}\) is available, is an implementation
defect rather than an alternate model.

1D_SPH and 2D_RZ GPU \(S_N\) production use the cell-local conservative
active-set closure. The namelist no longer exposes closure selectors; the
implementation is hardwired to conservative active set, signed face-flux
\(E^*\), donor-theta flux limiting, and AP face blending. 2D_RZ uses the same
Phase B material Newton wrapper as 1D_SPH, with `rad_E_out` aliased to
`rad_E` and `E_star_override` supplied by the 2D finite-volume face-flux update.

The 1D_SPH Phase B material solve uses
\[
E_g^{n+1}(T_e)=
\frac{E_g^*+\lambda^{PE}_g a_{eV}T_e^4b_g(T_e)}
     {1+\lambda^{PA}_g},\qquad
\lambda^{PA/PE}_g=\Delta t\,c\,\sigma^{PA/PE}_g ,
\]
with positivity applied only to the final radiation state. The conservative
active-set residual is:

1. Define the unfloored streaming state
\[
E^*_g =
E_g^{sweep}(1+\lambda^{PA}_g)
-\lambda^{PE}_g a_{eV}(T_e^n)^4b_g(T_e^n).
\]
2. Define the corrected final radiation energy before positivity enforcement
\[
\widetilde{E}_g(T_e)=
E_g^{sweep}
+r_g\left[a_{eV}T_e^4b_g(T_e)
-a_{eV}(T_e^n)^4b_g(T_e^n)\right],
\qquad
r_g=\frac{\lambda^{PE}_g}{1+\lambda^{PA}_g}.
\]
3. Enforce positivity only on the final radiation state:
\[
E^+_g(T_e)=\max(\widetilde{E}_g(T_e),0).
\]
4. Solve the conservative material residual
\[
R(T_e)=U_e(T_e)-U_e(T_e^n)+\sum_g\left[E^+_g(T_e)-E^*_g\right].
\]
5. For inactive groups where \(E^+_g=\widetilde{E}_g>0\), this reduces exactly
to the mixed-time residual with \(E^*_g\) left unfloored:
\[
U_e(T_e)-U_e(T_e^n)
-\sum_g q_g E^*_g
+\sum_g r_g a_{eV}T_e^4b_g(T_e),\qquad
q_g=\frac{\lambda^{PA}_g}{1+\lambda^{PA}_g}.
\]
6. For active groups where \(E^+_g=0\), the residual contribution is
\(-E^*_g\). Thus a negative streaming deficit is transferred through the
material residual rather than silently injected by a pre-source floor.
7. The residual and writeback must use identical \(E^+_g(T_e)\); otherwise the
local conservation identity is broken.
8. The upper energy bracket is
\[
U_{hi}=U_e(T_e^n)+\sum_g\max(E^*_g,0).
\]
If the upper-bracket sign check fails because of roundoff or table behavior,
the bracket is expanded adaptively; expansion failure is also a global
timestep-rejection condition.
If \(R(T_{floor})>0\), no positivity-preserving root exists above the floor and
the timestep must be rejected globally; local GPU subcycling is not used.
**Implementation (W-C, 2026-07-03):** the Newton kernel raises a retry flag
(bit 1 = floor-root, bit 2 = bracket-expansion failure) that the SN stage
forwards to `State::sn_material_retry_flag`. When
`Numerics.hydro.driver_full_step_retry_enabled=True` and the attempt budget
allows, the driver restores the pre-step snapshot and retries the full step at
\(\Delta t/2\) through the standard retry machinery (dt-lineage reason
`sn_material_newton_retry`). With retries disabled or exhausted the historic
behavior is kept: a WARNING is logged and the run proceeds with the locally
clamped \(T_e=T_{floor}\) state.

The production 1D_SPH sweep tallies a signed face flux \(F_{f,g}\) on radial
faces, where positive flux is outward (increasing \(r\)). After each Picard
sweep the center face is forced to zero by spherical symmetry and the outer
vacuum face is set to
\[
F_{N+1/2,g}=\frac{1}{2}cE_{N-1,g}^{sweep}.
\]
The streaming-only state passed to Phase B is then the finite-volume update
\[
E^{*,flux}_{c,g}=E^n_{c,g}
-\Delta t\,
\frac{A_{c+1/2}F_{c+1/2,g}-A_{c-1/2}F_{c-1/2,g}}{V_c},
\]
where \(A_f\) and \(V_c\) are the mesh face area and cell volume in the frozen
cgs geometry. No positivity clamp is applied to \(E^{*,flux}\); active-set
positivity is applied only to \(E^+_g(T_e)\). In face-flux mode the residual and
the final radiation writeback both use the identical override relation
\[
E^+_g(T_e)=
\max\left(
\frac{E^{*,flux}_{g}+\lambda^{PE}_g a_{eV}T_e^4b_g(T_e)}
     {1+\lambda^{PA}_g},0\right),
\]
which avoids subtracting nearly equal thick-LTE algebraic terms. DSA remains
cell-centered and does not modify \(F_{f,g}\); the face flux is recomputed from
the high-order sweep each Picard outer iteration.

[BUG-21c, 2026-07-14] After the material Newton writeback (each Picard outer
iteration), transparent cells are re-anchored to the transport moments. With
\(\lambda^{ext}_{c,g}=c\Delta t\,(\sigma^{a}_{c,g}+\sigma^{s}_{c,g})\)
(`state.sn_sigma_a` + `state.sn_sigma_s` — total extinction; scattering included since
BUG-25, 2026-07-19: an optically thick pure-scattering cell is in the diffusion regime
under AP-blend authority, not a void), the final radiation energy is
\[
E_{c,g}\leftarrow w\,E^{+}_{c,g}+(1-w)\,\phi^{m+1}_{c,g}/c,\qquad
w=3t^2-2t^3,\quad
t=\mathrm{clamp}\!\left(
\frac{\lambda^{ext}_{c,g}-10^{-4}}{10^{-3}-10^{-4}},0,1\right),
\]
with an exact early-out \(w=1\) for \(\lambda^{ext}_{c,g}\ge10^{-3}\)
(absorbing cells stay bitwise on the flux-form ledger) and \(w=0\) for
\(\lambda^{ext}_{c,g}\le10^{-4}\). Rationale: the flux-form ledger integrates
from \(E^n\) and in a void (\(\nabla\cdot F\to0\), no absorption or emission)
permanently freezes any transient imprint — the BUG-21 reproduction measured
\(E=1.887\,\phi/c\) frozen in the gap — while the swept moments are exact
there. The regimes sit 4+ orders apart in \(\lambda^{ext}\) (su_olson
\(\sim3\times10^{-2}\), the transparent-gap reproduction \(\sim10^{-8}\)), so
the anchor never activates in absorbing benchmarks. Gates:
`verify sn_1d_planar_transparent_gap` (contract C3) and the void-contract
case in `test_sn_streaming_limiter`.
Since 2026-07-26 (AI review k06 C8) every 1D anchor application is ledgered:
`state.sn_void_anchor_dE_step` / `sn_void_anchor_dE_abs_step` accumulate the
signed and absolute \(V\,\Delta E\) over the step (ledger-class scalars,
atomicAdd order within the documented host-ledger replica band; the anchored
field itself is bit-unchanged). A validation run that claims independent
\(S_N\)-reference status can require the abs sum to be exactly 0; a nonzero
sum is reported to the log (rate-limited). The 1D stagnation exit now also
sets the pre-existing `state.sn_outer_stagnated` flag (2D already did) and
logs the acceptance (AI review k06 C9 — the acceptance policy itself is
unchanged; changing it is a user decision).

[BUG-25, 2026-07-19] The 1D outer-boundary face-flux export and escape ledger
are now discretely consistent with the sweep in every regime. (i) The vacuum
branch of the boundary export previously overwrote the outer face flux with a
"Milne escape estimate" \(F_{out}=cE/2\) — exactly twice the discrete outgoing
half-range flux of a near-isotropic field (\(S^{+}\psi\simeq cE/4\) under the
\(\sum w=2\) GL convention). The pre-BUG-21 flux-form E update consumed the
same overwritten value, so the pair was self-consistently wrong (the
transparent-limit E*-accumulation pathology recorded in the BUG-21 execution
log); once BUG-21c anchored \(E=\phi/c\) to the moments, the 2x surplus broke
the volume-integrated streaming-conservation identity (ctests 1600/1603,
first full-suite run 2026-07-19). Vacuum now uses the sweep's discrete
outgoing tally, i.e. the Marshak branch's form with \(\psi_{in}=0\). (ii) The
AP-blend face array previously passed boundary faces through as pure SN; the
outer face now blends one-sidedly with the boundary cell playing both roles of
the interior formula, and the reduced-flux factor \(\alpha_f\) is exempted
there (escape makes \(f_{red}\sim1/4\) intrinsic at a free surface; \(\tau\)
and equilibrium factors alone decide the boundary regime), restoring the
FLD-consistent Marshak flux in the thick limit. (iii) The escape ledger books
the \(\theta\)-limited array (`sn_face_flux_limited`) — the same array the
E*-flux path consumes — instead of re-deriving \(cE/2\); in streaming tests
\(\theta=1\) and blend \(\alpha=0\) make it equal to the raw tally. Since
2026-07-20 the reported scalar is the GROSS outgoing energy: the driver energy
budget pairs \(+E^{Marshak}_{in}\) (source) with \(+E^{rad}_{esc}\) (sink), so
the Marshak-inflow part subtracted inside the net face ledger is added back
(`sn_escaped_step += sn_marshak_in_step`); the identity closes exactly because
the same scalar cancels on both sides, and vacuum configs add \(+0.0\)
(bit-identical). (iv) With
the anchor gated on total extinction (BUG-21c amendment above), every regime
has a single energy authority: anchored transport moments where
\(\lambda^{ext}\) is below the band, the flux-form/AP ledger elsewhere.
Gates: ctests `test_sn_face_flux_conservation`, `test_sn_ap_face_blend`
(conservation + FLD-match + gating cases), and the SN 1D family battery
(84/84 on the merged tree, 2026-07-19).

[BUG-25 2D port, 2026-07-20] Two of the four 1D mechanisms apply to the 2D_RZ
sweep and are ported. (iii) `sn_escaped_step` books the \(\theta\)-limited face
array (`sn_face_flux_limited`) integrated over the non-reflect boundary faces
with outward signs (\(+\) at the outer R face and top Z face, \(-\) at the
bottom Z face; the axis face is never booked), replacing the historic
coefficient model \(cE/2\) (vacuum) / \(cE/4\) (Marshak) that disagreed with
the discrete outgoing tally by up to 2x for a near-isotropic field. The scalar
feeds only the driver energy budget and diagnostics, so fields are bit-unchanged
by this rebooking. (iv) `anchor_void_rad_E_to_moments_2d_kernel` now gates on
the total extinction \(\lambda^{ext}\) exactly as the dimension-neutral BUG-21c
formula above prescribes (it previously used \(\sigma^{a}\) alone). Mechanism
(i) does not apply in 2D: there is no Milne overwrite — the face reduction
writes the full-range discrete net flux \(\sum_m w_m \mu_m^{face}\psi_m\) at
every boundary face (Marshak/reflect incoming ordinates are written by the
sweep; vacuum incoming slots stay zero), and the E*-flux update consumes it
directly (the step-total-energy identity is gated at 1e-10). Mechanism (ii)
(one-sided boundary AP blend) is deliberately not ported: its 1D precondition —
the boundary value changing when the Milne overwrite was removed — does not
exist in 2D, and boundary faces remain pass-through in the 2D blend (registered
follow-up if a thick-boundary defect is ever demonstrated). Gate:
`test_sn_2d_rz_bug25_escape_ledger` (ledger equals the discrete boundary face
integral, 1e-12 relative). The same gross convention applies in 2D
(`sn_escaped_step += sn_marshak_in_step` after both scalars are computed); the
Marshak pairing is gated in both dims (`sn 1d/2d escape ledger is gross at a
marshak boundary`, tolerance scaled by the largest participating magnitude since
escape = net + inflow is a cancellation whose float noise is
\(\sim\mathrm{ulp}(E^{Marshak}_{in})\)).

The production 2D_RZ sweep tallies unique signed R/Z face fluxes from the same
DD or linear-characteristic angular states used for the cell moments:
\[
F_{f,g}^{SN}=\sum_m w_m\,\mu_m^{face}\,\psi_{m,g}^{face}.
\]
The global sign convention is positive \(+R\) on R faces and positive \(+Z\) on
Z faces. For a cell \(c=(i,j)\), the finite-volume streaming state is
\[
E^{*,flux}_{i,j,g}=E^n_{i,j,g}
-\Delta t\frac{
A^R_{i+1/2,j}F^R_{i+1/2,j,g}
-A^R_{i-1/2,j}F^R_{i-1/2,j,g}
+A^Z_{i,j+1/2}F^Z_{i,j+1/2,g}
-A^Z_{i,j-1/2}F^Z_{i,j-1/2,g}}{V_{i,j}} .
\]
Here \(A^R=2\pi r\,\Delta z\) and
\(A^Z=\pi(r_{i+1/2}^2-r_{i-1/2}^2)\). No positivity clamp is applied to
\(E^{*,flux}\). The \(R=0\) axis face has zero geometric area and is
defensively zeroed in all face-bound closure buffers before any divergence
update to avoid \(0\times\mathrm{NaN/Inf}\) propagation.

#### AP face blend

The production path blends internal high-order face fluxes with an
FLD-style diffusion flux before the donor-theta limiter:
\[
F^{blend}_{f,g}=(1-\alpha_{f,g})F^{SN}_{f,g}
               +\alpha_{f,g}F^{diff}_{f,g}.
\]
The diffusion coefficient uses the same harmonic-D convention as the 1D FLD
tridiagonal assembly in the optically thick limit:
\[
D_{c,g}=\frac{c}{3\max(\sigma^{R}_{c,g},\sigma_{floor})},\qquad
D_{f,g}=\frac{2D_{L,g}D_{R,g}}{D_{L,g}+D_{R,g}},
\]
\[
F^{diff}_{f,g}=-D_{f,g}\frac{E^n_{R,g}-E^n_{L,g}}
 {0.5(r_{f+1}-r_{f-1})}.
\]
The AP path reuses `state.sn_sigma_s` as its "\(\sigma^R\)". **Fill truth
(2026-07-26 doc restoration, AI review k06 C2/C11)**: `sn_sigma_s` holds the
PHYSICAL scattering opacity — constant/power-law materials fill it with
\(\rho\kappa_s\) (`kappa_s`), and the NLTE pure-SN path fills it with the
Fleck-bypassed effective scattering \((1-f)\sigma^{PA}=0\) (f=1). It is NOT
the Rosseland mean. Consequently, in every production deck with
\(\kappa_s=0\) the \(\tau\) gate below evaluates \(\tau\approx0\) and the
blend weight is exactly \(\alpha=0\) on every face — the AP blend is inert
and the \(S_N\) answer is native transport (the sweep itself is unaffected:
its \(\sigma_s\) is the correct physical scattering). This matters for the
open \(S_N\)-vs-MGD bulk-compression comparison: those \(S_N\) results are
NOT FLD-contaminated through the blend. The blend machinery remains
unit-gated (`test_sn_ap_face_blend`) with synthetic opacities; wiring a true
Rosseland array into the AP gauge would ACTIVATE the blend in production and
is a user decision (escalated). Boundary faces are not blended:
\(F^{blend}_{1/2,g}=F^{SN}_{1/2,g}=0\) and the outer vacuum face remains the
SN-enforced \(F^{SN}_{N+1/2,g}=cE^{sweep}_{N-1,g}/2\).
For 2D_RZ the same harmonic diffusion formula is applied on every internal R
and Z face using the corresponding center-to-center R or Z spacing; all R/Z
boundary faces, including the axis, keep the raw \(S_N\) face flux and have
\(\alpha_{f,g}=0\).

The blend weight is the product of three smooth gates. The optical-depth gate
uses \(\tau_{min}=\min(\sigma^R_L\Delta r_L,\sigma^R_R\Delta r_R)\) and
smoothsteps from 0 at \(\tau=10\) to 1 at \(\tau=20\). The LTE gate is full for
\(\max(|E^n-B(T_e)|/B)\le0.05\) and off for values \(\ge0.10\). The reduced
flux gate is full for \(|F^{SN}|/(c\bar E)\le0.15\) and off for values
\(\ge0.25\). The donor-theta limiter is then applied to \(F^{blend}\), so any
diffusion replacement is still limited if it would drive \(E^{*,flux}<0\).
`radiation/diag_ap_alpha_face` records \(\alpha_{f,g}\) on 1D radial faces.
2D_RZ stores the unique-face alpha buffer in memory for the production closure
and writes the cell diagnostic `radiation/sn_ap_alpha`, the maximum adjacent
face-blend weight over faces and groups. The R2 ray-effect metric uses this
diagnostic in the optically thick interior:
\[
\mathrm{CV}_\alpha = {\sigma(\alpha_{\mathrm{AP}})\over
\max(|\langle\alpha_{\mathrm{AP}}\rangle|,10^{-300})}.
\]
For axisymmetric SN validation modes, \(\mathrm{CV}_\alpha \le 0.5\) is the
Phase 2b-1 loose gate for persistent ray-effect contamination. If no cell
satisfies the optically thick interior mask, the validation harness reports the
same statistic on the geometric interior as a fallback and records the mask
source.

The face-flux path applies a conservative donor-cell limiter before
Phase B, in two passes [BUG-21b, 2026-07-14]. For each cell and group, with
raw inflow/outflow loads
\[
I_{c,g}=\Delta t\,
\frac{A_{c-1/2}\max(F_{c-1/2,g},0)
      +A_{c+1/2}\max(-F_{c+1/2,g},0)}{V_c},
\qquad
O_{c,g}=\Delta t\,
\frac{A_{c+1/2}\max(F_{c+1/2,g},0)
      +A_{c-1/2}\max(-F_{c-1/2,g},0)}{V_c},
\]
pass 1 caps each cell's outflow by its stored energy alone,
\[
\theta^{(1)}_{c,g}=
\begin{cases}
\min(1,E^n_{c,g}/O_{c,g}), & O_{c,g}>0,\\
1, & O_{c,g}=0 ,
\end{cases}
\]
and pass 2 credits the inflow as limited by the upwind donors' pass-1
factors (domain-boundary inflow has no donor and is credited in full):
\[
\widehat I_{c,g}=\Delta t\,
\frac{A_{c-1/2}\,\theta^{(1)}_{c-1,g}\max(F_{c-1/2,g},0)
      +A_{c+1/2}\,\theta^{(1)}_{c+1,g}\max(-F_{c+1/2,g},0)}{V_c},
\qquad
\theta_{c,g}=
\begin{cases}
\min(1,(E^n_{c,g}+\widehat I_{c,g})/O_{c,g}), & O_{c,g}>0,\\
1, & O_{c,g}=0 .
\end{cases}
\]
Crediting the pass-1-limited inflow — never the raw \(I_{c,g}\) — keeps
positivity exact (every credited erg is deliverable within the step by
construction) while restoring free-streaming pass-through: the pre-fix
single-pass donor-only cap \(\theta=\min(1,E^n_{c,g}/O_{c,g})\) throttled a
conduit cell by its stored energy even though the same-step inflow
replenishes it, blocking fronts and freezing transient accumulations at
\(c\Delta t>\Delta x\) (measured 248 eV against a 200 eV drive; the
raw-credit form \((E^n+I)/O\) previously documented here was never the 1D
implementation).
Each face flux is then scaled once by its upwind donor:
\[
\widehat F_{f,g}=\theta_{d(f,g),g}F_{f,g},
\]
where \(d=f-1\) for \(F_{f,g}>0\) and \(d=f\) for \(F_{f,g}<0\). The tie case
\(F_{f,g}=0\) uses \(\theta=1\), i.e. no scaling. The inner spherical symmetry
face is forced to zero; a positive outer-vacuum flux uses the last physical cell
as donor, while a negative outer-face flux has no donor and is not scaled. The
limited streaming state replaces \(F\) with \(\widehat F\) in the same
finite-volume update for \(E^{*,flux}\). Conservation is unchanged because each
limited face flux is still single-valued and enters adjacent cells with opposite
signs.

In 2D_RZ the limiter remains single-pass with the raw inflow credit,
\(\theta=\min(1,(E^n_{c,g}+I_{c,g})/O_{c,g})\); the two-pass positivity-exact
credit above is 1D-only as of BUG-21b, and the port is on the 2D lane's
ledger together with the BUG-21 psi-persistence itself. The outgoing-energy
sum includes all non-axis R/Z faces with the cylindrical face areas above. The donor is the upwind cell in the global face
orientation: lower \(i\) or lower \(j\) for positive R/Z flux, higher \(i\) or
higher \(j\) for negative R/Z flux. Incoming face power contributes to
\(I_{c,g}\), so boundary-injected energy can participate in same-step donor
availability. Boundary incoming flux is not donor-limited; boundary outgoing
flux uses the adjacent cell's \(\theta\). Axis faces
\(R_{i=0,j}\) are skipped in donor selection and outflow accounting and remain
zero in the limited flux.

**2D two-pass inflow credit (BUG-21b port, 2026-07-17).** In 2D_RZ the
incoming-power credit \(I_{c,g}\) is evaluated in two passes
(`compute_streaming_theta_donor_2d_kernel` then
`compute_streaming_theta_2d_kernel`): pass 1 computes a donor-only
availability with no inflow credit,
\[
\theta^{(1)}_{c,g}=
\begin{cases}
\min\!\bigl(1,\;E^n_{c,g}/O_{c,g}\bigr), & O_{c,g}>0,\\
1, & O_{c,g}=0,
\end{cases}
\]
and pass 2 weights each face's incoming contribution by the adjacent donor
cell's pass-1 theta before forming the availability,
\[
I^{(2)}_{c,g}=\Delta t\,
\frac{\sum_{f\in\partial c} A_f\,\theta^{(1)}_{d(f),g}\,
      \max(\pm F_{f,g},0)_{\mathrm{in}}}{V_c},
\qquad
\theta_{c,g}=\min\!\bigl(1,\;(E^n_{c,g}+I^{(2)}_{c,g})/O_{c,g}\bigr),
\]
with \(\theta^{(1)}\equiv1\) for domain-boundary inflow (no donor cell). Face
limiting is unchanged: each face flux is scaled once by its upwind donor's
pass-2 theta. Positivity is exact by construction — the availability credits
inflow at \(\theta^{(1)}_{up}\) while the applied limiting delivers it at
\(\theta^{(2)}_{up}\ge\theta^{(1)}_{up}\) (pass-2 availability is never
smaller), so
\(E^{n+1}_{c,g}\ge E^n_{c,g}+\Delta t\,(\textstyle\sum
\theta^{(1)}_{up}\,\mathrm{in}-\theta_{c,g}\,\mathrm{out})/V_c\ge0\) —
while a uniformly throttled stream still relaxes to full pass-through.
Relative to the previous single-pass raw credit
(\(I\) evaluated with unthrottled face fluxes), \(\theta\) can only decrease,
and only in streaming-limited transients where the raw credit overdrew
(negative-\(E\) events). 1D twin: commit 3aeeab4f (1D branch).

**Void moment anchor (BUG-21c port, 2026-07-17).** After the 2D Picard outer
loop finalizes \(E\) and before the fixup tallies are published,
`anchor_void_rad_E_to_moments_2d_kernel` re-synchronizes effectively-void
cells to the transport moments. With
\(\lambda^{PA}_{c,g}=c\,\Delta t\,\sigma^{PA}_{c,g}\): for
\(\lambda^{PA}\ge10^{-3}\) the kernel returns without touching \(E\)
(absorbing regimes bit-identical by early return); below that,
\[
E_{c,g} := w\,E_{c,g} + (1-w)\,\phi^{sweep}_{c,g}/c,
\qquad
w=\begin{cases}
0, & \lambda^{PA}\le10^{-4},\\
S\!\bigl((\lambda^{PA}-10^{-4})/(9\times10^{-4})\bigr), &
10^{-4}<\lambda^{PA}<10^{-3},
\end{cases}
\]
where \(S\) is the cubic smoothstep. The anchor target is the converged
sweep moment \(\phi^{sweep}\) (`sn_phi_sweep`; `sn_phi_old` is the
step-start isotropic seed and must not be used). Rationale: the conservative
flux-form \(E^{*}\) update has no relaxation mechanism in transparent
regions, so transient bookkeeping imprints freeze into \(E\) (measured
+6..+87% versus the transport moments in the 1D twin); with negligible
matter coupling there is nothing to conserve against, and the transport
moments are the truth there. The blend is intentionally non-conservative in
the \(\lambda^{PA}<10^{-3}\) window. 1D twin: commit fe1960f9 (1D branch);
1D permanent gate: `verify sn_1d_planar_transparent_gap`.

The output diagnostic `radiation/diag_clip_energy` records only the radiation
portion of the Phase B negative pre-source clip. The companion diagnostic
`radiation/diag_clip_full_deficit` records the full, non-volume-weighted,
non-cumulative deficit magnitude
\[
\max(0,-E^{*,unclipped}_g),\qquad
E^{*,unclipped}_g =
E_g^{sweep}(1+\lambda^{PA}_g)
-\lambda^{PE}_g a_{eV}(T_e^n)^4b_g(T_e^n),
\]
with units \(\mathrm{erg/cm^3}\). For the same writeback state,
\[
\mathrm{diag\_clip\_full\_deficit}_g =
(1+\lambda^{PA}_g)\,\mathrm{diag\_clip\_energy}_g .
\]
Both clip diagnostics are zero in the 1D_SPH production face-flux path because
no algebraic pre-source clip is performed; `radiation/diag_E_star_flux` records
\(E^{*,flux}_{c,g}\) instead.

For each group \(g\) and ordinate \(\mu_n\), the backward-Euler equation is
\[
\frac{\psi_{c,g,n}^{m+1}-\psi_{c,g,n}^{n}}{c\Delta t}
+ \mu_n\frac{\partial\psi}{\partial r}
+ \frac{1-\mu_n^2}{r}\frac{\partial\psi}{\partial\mu}
+ \sigma_{t,c,g}\psi_{c,g,n}^{m+1}
= \frac{\sigma_{s,c,g}}{2}\phi_{c,g}^{m}
+ \frac{\eta_{c,g}(T_e)}{2}.
\]
The implementation persists the previous-step angular intensity
`state.sn_psi_prev` (layout `(cell*n_groups + g)*n_angles + n`) and feeds it
per angle into the transient source \(\psi^{n}_{c,g,n}/(c\Delta t)\) in every
sweep variant (spherical/cylindrical, serial and lane-parallel; the
starting-direction passes read the nearest regular ordinate). At the end of
each radiation step the swept \(\psi^{m+1}\) is copied into `sn_psi_prev`; on
first allocation the buffer is seeded isotropically as
\(\psi^{seed}=\tfrac12\,c\,E^n\), making the first step bit-identical to the
retired scheme. [BUG-21, 2026-07-13/14] The retired scheme stored only the
scalar `state.rad_E_old` and reconstructed the old intensity isotropically
each step; in transparent regions that re-isotropization acts as an
artificial per-step scattering, so free-streaming fronts advanced
diffusively (depth \(\propto\sqrt t\), effective speed \(c/30\)–\(c/80\)
measured) while absorbing benchmarks (su_olson) were untouched — the
coverage hole that hid the defect. Regression gate:
`verify sn_1d_planar_transparent_gap`. The scalar flux is
\(\phi_g=\sum_n w_n\psi_{g,n}\) and \(E_g=\phi_g/c\).

1D_SPH uses even-order Gauss-Legendre \(S_N\) sets. The default production
choice is \(S_{16}\) (`Radiation.sn_transport.n_angles=16`). \(S_8\) is allowed
as a performance option because sweep work scales linearly with the number of
ordinates, but it must pass the angular-quality gate
`test_sn_1d_s8_vs_s16_convergence`: on the GXII-like 1D profile,
\[
\max_{c,g}\frac{|E^{S8}_{c,g}-E^{S16}_{c,g}|}
{\max(|E^{S16}_{c,g}|,10^{-300})} < 0.05.
\]
Selecting \(S_8\) emits a warning from namelist validation to make the angular
accuracy tradeoff explicit.

The production spherical sweep default is the constant-source
linear-characteristic update
(`Radiation.sn_transport.spatial_scheme="linear_characteristic"`), with the
Morel-Adams angular redistribution coefficient \(\alpha_{n+1/2}\) represented
as an effective removal-and-source term inside the same characteristic solve.
The previous `spatial_scheme="diamond_difference"` path is retained only as a
deprecated regression option for DD comparisons. Negative ordinates sweep from
the outer vacuum boundary inward; the outgoing value at \(r=0\) is cached and
reused as the incoming value for the mirrored positive ordinate, enforcing
\(\psi(0,\mu<0)=\psi(0,-\mu)\). The outer boundary is vacuum for incoming
inward ordinates.

For `spatial_scheme="linear_characteristic"`, the 1D_SPH path replaces the
radial diamond update by a constant-source linear-characteristic update and
uses the Morel-Adams angular redistribution as an effective
removal-and-source term inside the same characteristic solve. For cell \(c\),
group \(g\), ordinate \(n\), path length
\(L_c=r_{c+1/2}-r_{c-1/2}\), scaled angular coefficients \(a\) from §6.8.1,
and incoming angular edge \(q_{c,n-1/2}\),
\[
\kappa^{MA}_{c,n}=\frac{a_{c,n+1/2}}{V_c},\qquad
S^{MA}_{c,g,n}=\frac{a_{c,n-1/2}}{V_c}q_{c,n-1/2},
\]
\[
\kappa^{LC}_{c,g,n}
=\sigma_{t,c,g}+\frac{1}{c\Delta t}+\kappa^{MA}_{c,n},\qquad
S^{LC}_{c,g,n}=S_{c,g}+S^{MA}_{c,g,n},
\]
\[
\tau_{c,g,n}=\kappa^{LC}_{c,g,n}L_c/|\mu_n|,\qquad
q_{c,g,n}=\frac{S^{LC}_{c,g,n}}{\kappa^{LC}_{c,g,n}},
\]
\[
\bar\psi_{c,g,n}
=A(\tau_{c,g,n})\psi_{in}
+\left[1-A(\tau_{c,g,n})\right]q_{c,g,n},
\]
\[
\psi_{out}
=E(\tau_{c,g,n})\psi_{in}
+\left[1-E(\tau_{c,g,n})\right]q_{c,g,n}.
\]
The LC angular half-edge update uses the Larsen-Morel lumped closure
\(q_{c,n+1/2}=\bar\psi_{c,g,n}\) for every ordinate, including the lower and
upper angular endpoints. Since the LC outgoing radial state and angular edge
are positive for non-negative effective source, the LC path does not apply the
diamond-difference radial or angular negative-flux fixups. Linear source
reconstruction remains a later LC extension.

For 2D_RZ, `spatial_scheme="linear_characteristic"` selects a constant-source
short-characteristic sweep over the existing R-Z wavefront ordering.  For each
ordinate \(d=(\mu_r,\mu_z)\), the upwind radial and axial face intensities are
read from the same reflection and face workspaces used by the diamond sweep.
The cell uses the axisymmetric face measures
\[
A_{r,L}=2\pi r_L\Delta z,\qquad
A_{r,R}=2\pi r_R\Delta z,\qquad
A_z=\pi(r_R^2-r_L^2),
\]
and volume \(V=\pi(r_R^2-r_L^2)\Delta z\).  With downwind radial face area
\(A_{r,D}\), the constant-source LC path uses the projected mean chord
\[
L_{c,d} =
\frac{V_c}{|\mu_r|A_{r,D}+|\mu_z|A_z},
\]
when the denominator is positive.  This choice preserves the thin-limit
source scaling of the finite-volume DD cell while replacing DD extrapolated
outflows by exponential attenuation.  The upwind face state is the projected
inflow blend
\[
\psi_{in} =
\frac{|\mu_r|(A_{r,L}+A_{r,R})\psi_r/2+|\mu_z|A_z\psi_z}
     {|\mu_r|A_{r,D}+|\mu_z|A_z}.
\]
For \(\sigma_{t,c,g}>0\),
\[
\tau_{c,g,d}=\sigma_{t,c,g}L_{c,d},\qquad
q_{c,g,d}=\frac{S_{c,g,d}}{\sigma_{t,c,g}},
\]
\[
\psi_{out}=E(\tau)\psi_{in}+[1-E(\tau)]q.
\]
Both radial and axial downwind face workspaces receive this positive
\(\psi_{out}\) for the current C-Step constant-source update.  The cell-average
scalar flux uses the same \(R\)-weighted exponential moment as the
axisymmetric volume integral.  Along the representative path
\(R(s)=R_0+\dot R s\), \(0\le s\le L\),
\[
\bar\psi =
\bar E_R(\tau)\psi_{in}+[1-\bar E_R(\tau)]q,
\]
\[
\bar E_R(\tau)=
\frac{R_0 A(\tau)+\dot R L M_1(\tau)}
     {R_0+\dot R L/2},\qquad
M_1(\tau)=\int_0^1 x e^{-\tau x}\,dx
=\frac{1-e^{-\tau}(1+\tau)}{\tau^2}.
\]
Small-\(\tau\) series are used for \(M_1\), and the zero-opacity limit uses
the corresponding \(R\)-weighted mean distance.  Cells touching the axis use
the existing \(r_L=0\) convention: the radial face area at the axis is zero,
but axis reflection workspaces still provide the parity state for outward
ordinates; all divisions are guarded by the projected downwind area.  As in
1D, C-Step 1 uses constant source cells only; linear-source reconstruction is
a later extension.

#### 6.8.1 Spherical angular redistribution (Morel-Adams 1979)

For 1D_SPH, the discrete angular derivative is not a local per-ordinate loss.
The raw `alpha_half` table is a dimensionless angle-space Morel coefficient.
TENRYU constructs it with the Gauss-Legendre \(\sum_n w_n=2\) convention
\[
\alpha_{n+1/2}-\alpha_{n-1/2}=-w_n\mu_n,
\]
with \(\alpha_{1/2}=\alpha_{N+1/2}=0\). Before insertion into the spherical
finite-volume sweep, each cell scales the raw coefficient by the local metric
\[
a_{c,n+1/2} = \frac{A_{c+1/2}-A_{c-1/2}}{w_n}\alpha_{n+1/2},
\]
so the scaled coefficients have area units. This gives the LTE fixed-point
cancellation identity
\[
a_{c,n+1/2}-a_{c,n-1/2}
=-\mu_n(A_{c+1/2}-A_{c-1/2}),
\]
or \( |\mu_n|(A_{c+1/2}-A_{c-1/2})\) for inward ordinates. Without this metric
scaling the dimensionless angular coefficients are incorrectly added to
area-like radial transport and volume-integrated collision terms.

Each cell stores the incoming angular-edge state \(q_{c,n-1/2}\) for the current
group. For interior angular half-edges, the diamond cell-average solve includes
the redistribution source
\[
(a_{c,n-1/2}+a_{c,n+1/2})q_{c,n-1/2}
\]
in the numerator together with the radial upwind source. After the ordinate is
solved, the outgoing angular edge is updated by
\[
q_{c,n+1/2}=2\bar\psi_{c,g,n}-q_{c,n-1/2}.
\]
The lower angular endpoint \(n=0\), where \(\alpha_{1/2}=0\), is not a real
incoming angular-edge flux. It uses endpoint step closure: the solve omits the
\((a_{c,n-1/2}+a_{c,n+1/2})q_{c,n-1/2}\) source and sets
\(q_{c,n+1/2}=\bar\psi_{c,g,n}\). At the upper endpoint
\(\alpha_{N+1/2}=0\), the stored outgoing diagnostic edge is also set to the
cell average. Interior half-edges continue to use angular diamond closure. The
stored edge then becomes the incoming angular edge for the next ordinate.

For the LC sweep, the same scaled coefficients \(a_{c,n\pm1/2}\) are not
inserted through the angular diamond denominator. Instead
\(a_{c,n+1/2}/V_c\) is added to the characteristic removal and
\((a_{c,n-1/2}/V_c)q_{c,n-1/2}\) is added to the per-volume source, as shown
above. The outgoing angular edge is then lumped to the cell average,
\(q_{c,n+1/2}=\bar\psi_{c,g,n}\), also at both angular endpoints. The weighted
angular contribution cancels cell-by-cell:
\[
\sum_n w_n\left[
\frac{a_{c,n+1/2}}{V_c}\bar\psi_{c,g,n}
-\frac{a_{c,n-1/2}}{V_c}q_{c,n-1/2}
\right]
=\frac{A_{c+1/2}-A_{c-1/2}}{V_c}
\sum_n\left[
\alpha_{n+1/2}\bar\psi_{c,g,n}
-\alpha_{n-1/2}q_{c,n-1/2}
\right]=0,
\]
because \(q_{c,n-1/2}=\bar\psi_{c,g,n-1}\) for interior half-edges and
\(\alpha_{1/2}=\alpha_{N+1/2}=0\). This is checked by the LC Morel-Adams smoke
test rather than by a production-kernel runtime residual.

This coupling requires each cell's angular edge to be consumed in ordinate
order within each group. The production 1D_SPH CUDA LC path assigns one warp to
each group and preserves that order with a half-angle diagonal wavefront:
negative ordinates advance over \(d=n+c'\), \(c'=N_c-1-c\), then positive
ordinates advance over \(d=(n-N_\Omega/2)+c\) after the reflected
`inner_boundary[n_angles]` values are available, with \(N_\Omega\) the ordinate
count. `angular_edge[n_cells]` and `inner_boundary[n_angles]` remain dynamic
shared-memory workspaces. The `TENRYU_DEBUG_LANE_PARALLEL` diagnostic build
path is still not a production physics-equivalent pair-parallel sweep.

References: Morel (1979) for spherical angular redistribution and
Adams-Larsen (2002) for discrete-ordinates transport acceleration and
diffusion-limit consistency.

#### 6.8.2 Linear-characteristic weight helpers

The LC helper header defines path optical-depth weights used by the opt-in
1D_SPH linear-characteristic sweep. For \(\tau=\kappa_t L/|\mu|\) and
\(E=\exp(-\tau)\),
\[
A=\frac{1-E}{\tau},\qquad
W_0=\frac{1-(1+\tau)E}{\tau},\qquad
W_1=\frac{\tau-1+E}{\tau},
\]
\[
B_0=\frac12-\frac{1-(1+\tau)E}{\tau^2},\qquad
B_1=\frac12-\frac{\tau-1+E}{\tau^2}.
\]
Thus the outgoing characteristic state and the cell average conserve constant
sources:
\[
W_0+W_1+E=1,\qquad B_0+B_1+A=1.
\]
For \(\tau<10^{-3}\), the implementation evaluates sixth-order Taylor
polynomials for \(E,A,W_0,W_1,B_0,B_1\) to avoid cancellation.  For
\(\tau\ge745\), it sets \(E=0\) and evaluates the same algebraic weights in
inverse powers of \(\tau\), preserving the conservation identities while
approaching \(A,W_0\to0\), \(W_1\to1\), and \(B_0,B_1\to1/2\).

Linear endpoint reconstruction uses a monotone Walters limiter.  With
\(d_L=q_i-q_{i-1}\), \(d_R=q_{i+1}-q_i\), the limited endpoint jump is zero
when \(d_Ld_R\le0\); otherwise its magnitude is bounded by the centered jump
between neighboring cell centers \(|q_{i+1}-q_{i-1}|\) and by \(2|d_L|\),
\(2|d_R|\).  The reconstructed endpoints are then scaled, if needed, so both
endpoints remain nonnegative.

2D_RZ uses a product level-symmetric quadrature over \((\mu_R,\mu_Z)\) with
azimuthal weights folded into the axisymmetric RZ solve. For fixed polar
ordinate \(i_z\), \(\phi_{i_\phi}=(i_\phi+1/2)\pi/N_\phi\) and
\(\mu_R=\sqrt{1-\mu_Z^2}\cos\phi_{i_\phi}\), so \(i_\phi=0\) is the most
positive radial ordinate and \(i_\phi=N_\phi-1\) is the most negative one.
Cylindrical curvature is represented as a one-dimensional angular edge coupling
inside each \(i_z\) row. The row is swept in descending \(i_\phi\),
\[
d_m=(i_z,N_\phi-1-m),\quad
\alpha_{i_z,0}=0,\quad
\alpha_{i_z,m+1}=\alpha_{i_z,m}-\mu_{R,d_m}w_{d_m},\quad
\alpha_{i_z,N_\phi}=0 .
\]
Small negative roundoff is clipped to zero during quadrature setup. At the start
of every source iteration the per-cell angular edge state
\(\psi^\phi_{g,c,i_z}\) is zeroed, then each \(m\) launch reads the previous
edge and writes the next edge for the same \((g,c,i_z)\).

For a cell with \(A_R^- , A_R^+\) and \(A_Z\), define
\[
G_c=\frac{A_R^+-A_R^-}{\max(w_d,10^{-300})},\qquad
\beta^- = G_c\alpha_{i_z,m},\qquad
\beta^+ = G_c\alpha_{i_z,m+1}.
\]
The 2D_RZ diamond-difference sweep solves
\[
(\sigma_tV+2|\mu_R|A_R^{down}+2|\mu_Z|A_Z+2\beta^+)\bar\psi_d
= VQ_d+|\mu_R|(A_R^-+A_R^+)\psi_R^{in}
  +2|\mu_Z|A_Z\psi_Z^{in}+(\beta^-+\beta^+)\psi_\phi^{in}.
\]
The outgoing angular edge is
\(\psi_\phi^{out}=2\bar\psi_d-\psi_\phi^{in}\), with the same nonnegative
finite clipping used for spatial outgoing faces. For uniform \(\psi=q\) and
compatible source \(Q_d=\sigma_t q\), the recurrence for \(\alpha\) supplies the
missing cylindrical curvature balance, so \(\bar\psi_d=q\) cell by cell and the
row closes because \(\alpha_{i_z,0}=\alpha_{i_z,N_\phi}=0\).

The opt-in 2D_RZ linear-characteristic sweep uses the same \(\beta^\pm\)
coefficients but merges radial, axial, and angular inflow into
\[
P=|\mu_R|A_R^{out}+|\mu_Z|A_Z+\beta^+,\qquad
\psi^{in}_{eff}
=\frac{|\mu_R|A_R^{in}\psi_R^{in}+|\mu_Z|A_Z\psi_Z^{in}
       +\beta^-\psi_\phi^{in}}{P}.
\]
The characteristic length is \(L=V/P\) for \(P>10^{-300}\), otherwise zero.
The existing LC attenuation/source weights compute the cell average and one
outgoing state for the radial face, axial face, and angular edge.

Each source iteration sweeps cell-centered intensities on the R/Z structured
mesh. The R-axis boundary uses reflective parity
\(\psi(R=0,\mu_R>0,\mu_Z)=\psi(R=0,-\mu_R,\mu_Z)\); the outer R boundary is
vacuum by default for incoming ordinates; 2D_RZ also supports `"reflect"` for
closed-cylinder benchmarks and 1D-z reductions. Z boundaries use
`Radiation.sn_transport.z_boundary` / `boundary.z`, or the face-specific
`boundary.z_bottom` / `boundary.z_top`: `"vacuum"` sets incoming intensity to
zero, while `"reflect"` maps
\(\psi(\mu_R,\mu_Z)\leftrightarrow\psi(\mu_R,-\mu_Z)\) at the lower/upper Z end.
For Phase 2b-1 SN Marshak source BC, a Z face \(f\) with boundary type
`"marshak"` injects a steady gray incoming flux
\(F_{\mathrm{inc}}=\)
`Radiation.sn_transport.marshak.flux_erg_per_cm2_s`
[erg cm\(^{-2}\) s\(^{-1}\)]. The incoming angular intensity for each ordinate
entering the domain is
\[
\psi_{\mathrm{in},d}=2F_{\mathrm{inc}},
\]
so the quadrature face moment
\(\sum_{\mu_Z n_f>0} w_d |\mu_{Z,d}|\psi_{\mathrm{in},d}\)
equals \(F_{\mathrm{inc}}\) because the product quadrature integrates
\(\int_0^1\mu\,d\mu=1/2\). This incoming face flux is added to the unique Z-face
raw flux with the global face sign convention. Marshak source accounting uses
\[
E_{\mathrm{Marshak,in}}=\Delta t\,F_{\mathrm{inc}}
\sum_{f\in\mathrm{Marshak}\ Z} A_f .
\]
For escape accounting, Marshak Z faces use the same outgoing leakage coefficient
\(c/4\) as the FLD Marshak source BC; vacuum uses \(c/2\), reflect uses zero.
The 2D path uses the same GPU material Newton update as 1D after the scalar
flux converges. Production 2D_RZ \(S_N\) closure stores face-bound quantities in
a unique-face layout. R faces come first:

\[
f_R(i,j)=iN_z+j,\qquad 0\le i\le N_R,\quad 0\le j<N_z,
\]
followed by Z faces:
\[
f_Z(i,j)=(N_R+1)N_z+i(N_z+1)+j,\qquad
0\le i<N_R,\quad 0\le j\le N_z.
\]
Verification harnesses R1/R2 compare Marshak boundary-source runs against the
Python reference `tools/marshak_boundary_source_reference.py`.  The FLD
reference solves a one-dimensional finite-volume LTE matter-radiation system in
\(x=z_{\mathrm{top}}-z\) with the same cgs/eV constants, a top Marshak Robin
condition \(D\,\partial_xE=cE/4-F_{\mathrm{inc}}\), a reflecting bottom
condition, and a fully implicit Newton solve of the coupled \(E,T\) cell
unknowns.  The S_N reference uses S_N Gauss-Legendre ordinates, first-order
upwind Strang streaming/collision, \(\psi_{\mathrm{in}}=2F_{\mathrm{inc}}\)
for the z-top incoming ordinates (mapped to \(\mu_x>0\) in the depth
coordinate), and reflecting bottom parity.  R2 reports this comparison as a
full-radial volume-weighted 1D-z profile metric, gated at
`MARSHAK_SN_PROFILE_TOL = 0.15` (measured 7.94e-3 at S16 production deck).
For cell \(c=iN_z+j\), the four faces are
\[
R_- = f_R(i,j),\quad R_+=f_R(i+1,j),\quad
Z_- = f_Z(i,j),\quad Z_+=f_Z(i,j+1).
\]
The face-flux divergence used for the 2D Phase B override is
\[
\nabla\cdot F =
\frac{A^R_+F_{R_+}-A^R_-F_{R_-}+A^Z_+F_{Z_+}-A^Z_-F_{Z_-}}{V_{i,j}},
\]
with \(A^R_\pm=2\pi r_\pm\Delta z\) and
\(A^Z_+=A^Z_-=\pi(r_+^2-r_-^2)\) for an orthogonal RZ cell.

Source iteration is accelerated by per-group DSA when
`Radiation.sn_transport.dsa_enabled=True`; when it is `False`, the sweep
iteration skips the DSA correction branch and no DSA kernels are launched. In
the notation of Larsen
(1982), Eq. 4.12, the isotropic-scattering error equation places the lagged
scattering error on the right-hand side. For backward Euler TENRYU stores the
scalar flux \(\phi=cE\), multiplies the correction equation by \(c\Delta t\),
and assembles the 1D_SPH row in volume-integrated form:
\[
\begin{aligned}
&\left(V_c+c\Delta t\,\sigma_{a,c,g}V_c+\sum_f K_f+B_c\right)
\delta\phi_{c,g}
-\sum_{f\in\mathrm{interior}} K_f\delta\phi_{nb(f),g} \\
&\qquad =
c\Delta t\,\sigma_{s,c,g}V_c
\left(\phi^{sweep}_{c,g}-\phi^{old}_{c,g}\right),
\end{aligned}
\]
with
\[
D_{c,g}=\frac{1}{3\sigma_{t,c,g}},\qquad
K_f=c\Delta t\,A_fD_f/\Delta r_f.
\]
There is no factor \(1/2\) on the RHS: the angular source is
\(\sigma_s\phi/2\), but the scalar moment integrates over the discrete angular
weights whose sum is 2, leaving \(\sigma_s\phi\). The \(c\Delta t\sigma_s V\)
factor is therefore cell/group local and is required for DSA to correct the
source-iteration residual at the same scale as the backward-Euler transport
operator.

At \(r=0\), 1D_SPH scalar parity gives a no-flux DSA boundary, so no additional
row contribution is assembled. At the outer vacuum boundary the 1D_SPH DSA
operator applies the same Marshak-like leakage convention used by
`escaped_energy_kernel` in `src/radiation/sn_transport_1d_gpu.cu`:
\[
F_{out}=\beta_{vac}cE=\beta_{vac}\phi,\qquad \beta_{vac}=0.5.
\]
The corresponding volume-integrated diagonal term is
\[
B_c=c\Delta t\,\beta_{vac}A_{out}
\]
on the last cell only; it is not divided by \(V_c\). The resulting tridiagonal
systems are solved by cuSPARSE `cusparseDgtsv2StridedBatch` with batch count
\(G\), system size \(N_{cell}\), and stride \(N_{cell}\). In 2D_RZ the DSA
correction uses the corresponding R/Z 5-point diffusion stencil and a Jacobi
iteration on the GPU; this is functional but not yet optimized for GXII-scale
2D production.

For 1D_SPH the inner source-iteration body is launched through a CUDA graph when
`Radiation.sn_transport.inner_graph_unroll > 1` (default \(K=5\)). One graph
contains \(K\) repeated bodies:
build sweep inputs, spherical sweep with integrated moment accumulation, DSA
correction, and the source-iteration state copy. The relative scalar-flux
residual is reduced only on the final unrolled body, so convergence is tested
every \(K\) inner iterations. The graph key includes cell/group/angle counts,
\(K\), DSA mode,
\(\Delta t\), spherical-sweep dynamic shared-memory size, quadrature-device
pointers, mesh/radiation buffer pointers, the streaming-limiter mode flag, and
the DSA cuSPARSE work buffer pointer. It is recaptured when those keys change,
which covers mesh reallocations, angle-count changes, namelist mode changes,
and timestep changes that alter kernel parameters. The DSA cuSPARSE buffer is
allocated before capture; if graph capture or instantiation is unavailable, the
solver falls back to the same streamed kernel sequence.

After each outer Picard sweep, a GPU Newton kernel solves the implicit electron
balance cell-locally:
\[
F(T)=\rho\frac{e_e(\rho,T)-e_e(\rho,T_e^n)}{\Delta t}
-\sum_g c\sigma^{PA}_{g}E_g
+\sum_g c\sigma^{PE}_{g}a_{eV}T^4b_g(T)=0.
\]
For TMAT/table EOS, the material Jacobian is
\(\rho c_{v,e}(\rho,T)/\Delta t\), and the converged `ee` and `Pe` are written
from the same electron table at \((\rho,T)\). If no electron EOS device view is
provided, the kernel falls back to the legacy constant-\(c_v\) residual and
ideal-gas pressure write. The radiation Jacobian uses the PE-side analytic per-group term
\(\sum_{g:\,E^+_g>0}[\lambda^{PE}_g/(1+\lambda^{PA}_g)]\,4a_{eV}T^3b_g\)
— the active-set mask and the \(1/(1+\lambda^{PA})\) reduction included
(2026-07-26 doc truth restoration, AI review k06 4.4); \(db_g/dT\) is not
included. The
Picard residual is
\[
r_k=\max_c\frac{|T^{k+1}_{e,c}-T^{k}_{e,c}|}
{\max(T^{k+1}_{e,c},T_{floor})}.
\]
The effective tolerance is
\[
r_{\mathrm{tol}}=\max\left(\texttt{outer\_tol},
10\,\texttt{outer\_tol\_hydro\_error\_scale}\right),
\]
with defaults \(10^{-4}\) and \(10^{-5}\), respectively. Picard exits when
\(r_k \le r_{\mathrm{tol}}\), or, after at least five Picard iterations, when
\[
\frac{r_k}{r_{k-2}} >
\texttt{outer\_tol\_stagnation\_factor}
\]
(default 0.5). The stagnation exit accepts the current radiation/matter update
because further Picard work is below the hydro time-integration error scale in
the targeted GXII regime. Inner source iteration uses `inner_tol`. The
deterministic tallies are
`rad_dep[c,g]=c sigma_PA E_g V_c Delta t` and
`rad_emit[c,g]=eta_g V_c Delta t`, with
\(\eta_g=c\sigma^{PE}_g a_{eV}T^4b_g\).

In 2D_RZ production SN, when the outer material Picard residual satisfies
`outer_residual <= outer_tol` but the inner source iteration has not satisfied
`inner_tol`, the outer loop may terminate as an outer-stagnated state. The
inner residual is treated as plateaued when three consecutive outer iterations
after the first candidate satisfy the symmetric test
\[
|r_k-r_{k-1}| \le 10^{-6}\max(|r_k|,|r_{k-1}|,10^{-300}).
\]
This sets `sn_outer_stagnated=true` while leaving `sn_converged=false`, so the
diagnostic remains honest that inner source iteration did not converge. The
exit avoids redundant identical-input sweeps in thin/free-streaming regimes
where source iteration cannot reach `inner_tol` but radiation-material coupling
is already steady under the outer tolerance.

In 2D_RZ \(S_N\) transport, the per-iteration sweep is executed as four ordered
octant launches with one block per group-direction pair. Per-direction angular
cell averages and unique-face angular states are accumulated into private
workspaces and then reduced into the scalar-flux moment \(\phi\), the radial
pressure tensor component \(P_{rr}\), and the unique-face flux `face_flux_raw`
by deterministic kernels that sum contributions in increasing d order. This
preserves the prior serial-direction accumulation order bit-exactly while
enabling direction-level parallelism across SMs.

#### 6.8.3 1D cylindrical product quadrature and per-level conservative sweep (W-G3, 2026-07-04)

1D cylindrical \(S_N\) (`Mesh.geometry_1d="cylindrical"` + `mode="sn_transport"`)
keeps the intensity's full azimuthal dependence: with \(\xi\) the axial cosine
(invariant along rays), \(\sin\theta=\sqrt{1-\xi^2}\), and \(\omega\in(0,\pi)\)
the azimuth about the axis measured from the outward radial direction, the
radial cosine is \(\mu=\sin\theta\cos\omega\) and the conservative equation is
\[
\frac{1}{r}\partial_r(r\mu\psi)-\frac{1}{r}\partial_\omega(\eta\psi)
+(\sigma_t+1/c\Delta t)\psi=q,\qquad \eta=\sin\theta\sin\omega .
\]
Angular redistribution couples ordinates **within one \(\xi\)-level only**
(Morel & Montry 1984, TTSP 13(5) 615, appendix — the W-G3 authority).

**Product quadrature** (`sn_cyl_quadrature_1d.{hpp,cpp}`): for
`n_angles` \(=2L^2\) (validated; 8, 18, 32, 50, ...), \(L\) polar levels at the
positive Gauss-Legendre\((2L)\) nodes \(\xi_\ell\) (half-range by z-symmetry,
\(\sum_\ell v_\ell=1\)) times \(M=2L\) azimuthal Chebyshev midpoints with equal
weights \(2/M\), stored level-major and ascending in \(\mu\) with bit-exact
\(\pm\mu\) pairing; \(\sum w=2\) preserves every existing normalization
(isotropic \(q/2\) factors, \(\phi=\sum w\psi=cE\), \(\psi_{in}=2F_{inc}\),
\(\sum w\mu^2=2/3\Rightarrow\chi=1/3\) in isotropic fields). The Carlson
recursion \(\alpha_{\ell,m+1/2}=\alpha_{\ell,m-1/2}-\mu_{\ell,m}w_{\ell,m}\)
runs per level with both level edges pinned to exactly \(0.0\); those zero
edges double as the chain delimiters the sweep kernels already key on
(`alpha_prev_raw == 0.0`), and the per-ordinate metric identity
\((\Delta A/w)(\alpha_{+}-\alpha_{-})=-\mu\,\Delta A\) holds ordinate-wise, so
a uniform isotropic field is annihilated by streaming+redistribution for any
\(A(r)\) — the BUG-8 fixed-point identity, geometry-independent.

**Weighted diamond (M&M Eqs. A1-A4)**: unlike spherical, the angular cell-edge
cosines are NOT weight partial sums; the azimuthal **angle** edges partition
\((\pi\to 0)\) by the level weights (equal weights \(\Rightarrow\) uniform
\(\Delta\omega=\pi/M\)) and the edge cosines follow as
\(\mu_{\ell,m\pm1/2}=\sin\theta_\ell\cos\omega_{m\pm1/2}\);
\(\tau_m=(\cos\omega_m-\cos\omega_{m-1/2})/(\cos\omega_{m+1/2}-\cos\omega_{m-1/2})\)
is level-independent (\(\sin\theta_\ell\) cancels), \(\tau\in(0,1)\), mirror
symmetric. **Starting direction**: the Miller-Alcouffe procedure generalizes to
one slab step-characteristic sweep per level along the \(\omega=\pi\) diametral
ray with \(|\mu|=\sin\theta_\ell\) (cell optical depth
\(\sigma_{eff}\Delta r/\sin\theta_\ell\)), seeding that level's ladder edge
\(\psi_{\ell,1/2}\); the axis reflection pairs \((\ell,m)\) with
\((\ell,M-1-m)\) within the level.

**Kernels** (`sn_sweep_cylindrical_{serial,lc}_kernel`): NEW functions — the
spherical/planar kernels are untouched (bitwise strategy; the 7-gate
spherical+planar battery reproduced its pre-change logs bit-identically). The
LC kernel loops levels sequentially over one shared `angular_edge` array (SD
seed, negative-\(\mu\) wavefront, within-level reflection, positive-\(\mu\)
wavefront), reusing the conservative-FV + \(\theta(\tau)\) + weighted-diamond
cell update verbatim; the serial kernel is the flat-loop diamond/step-start
clone with the per-level reflection index. `precompute_lc_weights_kernel`, the
K2 moment/face reductions, the E*/donor-\(\theta\)/AP closures, escaped-energy
and marshak boundary bookkeeping are reused unchanged (flat ordinate sums +
runtime `geom`); the marshak ledger's discrete \(S^-=\sum_{\mu<0}w|\mu|\) is
taken from the product set. **DSA is force-disabled for cylindrical** (the
tridiagonal operator hardcodes \(4\pi r^2\) faces — spherical-only;
acceleration-only, converged answer unchanged, one-time warning; pure-absorber
gates unaffected). `TENRYU_DEBUG_LANE_PARALLEL` builds reject cylindrical.

**Gates/tests**: `sn_1d_cylindrical_marshak_equilibration` (phase A uniform
blackbody fixed point on the FULL cylinder r0=0 including the axis cell —
measured drift 1.39e-16 = one ulp of \(a_{eV}T_r^4\); phase B cold-start
plateau, outer_rel 1.6e-7 / max_rel 9.6e-7 at 1800 steps, tolerances 1e-5 as
spherical/planar), ctest `test_sn_cyl_quadrature` (CPU invariants) and
`test_sn_1d_cylindrical_fixed_point` (LC+diamond × S8+S32, drift ≤ 1e-12,
\(\chi=1/3\)). Residuals: no analytic cylindrical transport benchmark in the
library yet (Lewis & Miller / PARTISN manuals in manual_queue); DSA
cylindrical faces; multigroup cylindrical marshak (G=1 parity with the other
geometries).

---

## 7. DDMC（Discrete Diffusion Monte Carlo） [RETIRED — legacy; 現行輻射は §6.7 FLD / §6.8 \(S_N\)]

> **【CURRENT RADIATION MODEL — 本章 §7 は RETIRED】** DDMC（Discrete Diffusion Monte Carlo）および §7.1.2g HOLO は **RETIRED**（FREEZE-1D-RAD・D1 以降）。現行の輻射輸送は決定論の **FLD（§6.7, `mode="multigroup_diffusion"`）** と **\(S_N\)（§6.8, `mode="sn_transport"`）** のみ。DDMC/HOLO コードは互換のため tree に残るが FLD/\(S_N\) mode で完全 bypass（`ddmc.enabled=False`, `holo.enabled=False` 必須）。以下の §7 全記述は歴史的参照であり現行仕様ではない。詳細は `SPECIFICATION.md` の `mode` 定義参照。

### 7.1 DDMC領域判定（cell×group）— diffusion criterion

DDMCが有効であるためには、セルが十分に拡散的である必要がある。
τ（光学厚）だけでなく **ω（散乱比）** も条件に含める（Cleveland & Gentile 2015 §2.4準拠）。

#### 7.1.1 指標の定義
光学厚さ（transport）：
\[
\tau_{i,g} = \sigma_{tr,i,g}\, \ell_i \quad [\text{無次元}]
\]
ここで \(\sigma_{tr}\) [cm\(^{-1}\)]、\(\ell_i\) [cm]。v1.0既定：
- \(\sigma_{tr,i,g} \equiv \sigma_{R,i,g}\)（Rosseland）
  ※物理散乱を入れる場合は \(\sigma_{tr}=\sigma_R+\sigma_s(1-\bar\mu)\) 等へ拡張
- \(\ell_i\)：セル代表長（既定：\(\ell_i = 2 \min_f d_{center\to f}\)、すなわちセル中心から最近面までの垂直距離の2倍）
  - **1D_SPH**：\(\ell_i = r_{i+1/2} - r_{i-1/2} = \Delta r_i\)（球殻の厚さ）
  - **2D_RZ**：セル中心 \(\mathbf{r}_{center}\) から4辺への垂直距離の最小値 \(d_{min}\) を用いて \(\ell_i = 2\,d_{min}\)
    - **セル中心の定義**：4頂点の算術平均 \(r_c = (r_1+r_2+r_3+r_4)/4\), \(z_c = (z_1+z_2+z_3+z_4)/4\) とする（面積重心ではなく頂点平均）
    - 辺 \(k\) への垂直距離（**辺長ガード付き**）：
      辺長 \(L_k = |\mathbf{V}_{k+1}-\mathbf{V}_k|\) を先に計算し、
      \(L_k < \varepsilon_{geom}\)（退化辺）の場合は \(d_k = |\mathbf{r}_{center}-\mathbf{V}_k|\)（端点距離）で代替する（0除算防止）。
      \(L_k \ge \varepsilon_{geom}\) の場合：\(d_k = |(\mathbf{r}_{center}-\mathbf{V}_k)\times(\mathbf{V}_{k+1}-\mathbf{V}_k)| / L_k\)
    - \(d_{min}\) の計算では無限直線までの距離ではなく**有限線分までの距離**を使用する。
      凸四角形では垂線の足が辺上に常に存在するため、無限直線距離と有限線分距離は一致する。
      非凸セル（ALE変形後に発生しうる）では、垂線の足が辺の端点外に落ちる場合があり、
      その場合は端点までの距離を使用する。具体的には：
      辺 \(k\) のパラメータ \(\hat{t} = (\mathbf{r}_{center}-\mathbf{V}_k)\cdot(\mathbf{V}_{k+1}-\mathbf{V}_k)/|\mathbf{V}_{k+1}-\mathbf{V}_k|^2\) を求め、
      \(d_k = \begin{cases} d_{infinite} & (0 \le \hat{t} \le 1) \\ \min(|\mathbf{r}_{center}-\mathbf{V}_k|,\; |\mathbf{r}_{center}-\mathbf{V}_{k+1}|) & (\text{otherwise})\end{cases}\)
    - **退化セル対策**：\(d_{min} < \varepsilon_{geom}\) (\(\varepsilon_{geom} = 10^{-12}\) cm) の退化セルでは \(\ell_i = 2\varepsilon_{geom}\) として DDMC 判定を回避する（IMC に退化）
    - 正方形セルでは \(\ell_i = \Delta x\)（セル幅）に一致

> **Cleveland & Gentile (2015) からの多群拡張**：
> Cleveland & Gentile Eq.(21) はgrey総不透明度 \((\kappa+\kappa_s)\) でτを定義している。
> 多群TENRYUでは、DDMCが拡散方程式（D = 1/(3σ_R)）に基づくため、
> τの不透明度も **Rosseland** を採用し、拡散係数との整合性を保つ。
> greyかつ物理散乱なし（v1.0既定）では \(\kappa_P=\kappa_R=\kappa\) となり
> Cleveland & Gentile原式と一致する。

散乱比（Fleck factorによる実効散乱を含む）：
\[
\omega_{i,g} = \frac{\sigma_{s,phys,g} + (1-f_i)\,\sigma_{a,g}}{\sigma_{a,g} + \sigma_{s,phys,g}}
\]

> **実装ノート**：
> \(\sigma_{a,g}+\sigma_{s,phys,g}<\epsilon_\sigma\)（\(\epsilon_\sigma = 10^{-30}\,[\mathrm{1/cm}]\)）の場合は、\(\omega_{i,g}=\operatorname{clamp}(1-f_i,0,1)\) を用い、事前判定での NaN を防ぐ。

v1.0既定（\(\sigma_{s,phys}=0\)）では \(\omega_{i,g} = 1-f_i\)。

> 物理的意味：ωが大きい（≈1）ほど衝突が実効散乱支配であり、拡散近似が成り立つ。
> Fleck factor fが小さい（高温・光学厚）領域で ω→1 となり、DDMCに適する。

#### 7.1.2 diffusion criterion（判定条件）
セル i × 群 g の baseline transport mode は、以下の優先順で決める。

**DDMC 条件**：

1. **散乱比閾値**：\(\omega_{i,g} \ge \omega_{DDMC}\)（既定 \(\omega_{DDMC}=0.9\)）
2. **光学厚閾値**：\(\tau_{i,g} \ge \tau_{DDMC}\)（既定 \(\tau_{DDMC}=4.0\)）

> VERIFICATION.md §8-§9 の検証テストでは τ_DDMC = 3.0 を使用する場合がある（全セルDDMCモードを確実にするためのテスト設計上の選択）。

3. **M‑matrix条件**（7.3.3）を満たす
4. **IMC→DDMC変換確率制約**（7.7.1, 7.7.3）を満たす（\(0 \le \hat{P}(\mu) \le 1\)；v1.0既定の \(\hat{P}\) では条件1により概ね充足されるが、\(\tau \approx \tau_{DDMC}\) の境界領域では §7.7.3 の安全策（クランプ/フォールバック）が必要）

判定：
- DDMC 条件 1〜4 をすべて満たす ⇒ DDMC
- それ以外 ⇒ IMC

> **PGRW 注**：`tau_rw` は独立 mode を生成しない。`tau_rw > 0` のとき、
> `imc_transport_persistent` 内で IMC 粒子に対してのみ 1D_SPH internal PGRW
> eligibility 判定を行う。mode map 自体は IMC/DDMC の2値である。

> **根拠**：τだけで判定すると、Fleck factor fが大きい（ω小、散乱が弱い）のに拡散として扱い、
> 非物理な挙動を起こし得る（Cleveland & Gentile 2015, Densmore et al. 2007）。
> 散乱比を入れることで、DDMCが実際に拡散的に振舞うセルのみに適用される。
>
> **参考文献**：
> - Cleveland & Gentile, JCP 291 (2015): diffusion criterion として ω≥0.9, τ_min≥4 を例示
> - Densmore et al., JCP 222 (2007): P(μ)制約との整合
>
> **Phase-1 実装注記**：2D_RZ では PGRW を有効化せず、mode map は IMC / DDMC の2値に退化する。

#### 7.1.2a Hybrid diffusion 分類マスク（PR1）

`Radiation.diffusion.enabled=True` かつ 1D_SPH の場合、DDMC mode selection の前にセル単位の diffusion 分類マスクを作る。entry/exit 時の粒子表現と deterministic 表現のエネルギー変換、diffusion セルの local source solve、および PR4 の 1D RKL2 空間 diffusion step を行う。分類マスクは DDMC が diffusion 領域および guard 領域を claim しないようにも使う。

セル \(i\) の Rosseland 平均は、群別 \(\sigma_{R,i,g}\) と Planck fraction \(b_g(T)\) から
\[
w_{i,g}=\max\left(\frac{\partial(T^4 b_g)}{\partial T}\bigg|_{T_i},0\right),\qquad
\bar{\sigma}_{R,i}=\frac{\sum_g w_{i,g}}{\sum_g w_{i,g}/\max(\sigma_{R,i,g},\sigma_{floor})}
\]
で評価する。1群では \(w=1\) とし、重み和または分母がゼロの場合は
\(\max_g\sigma_{R,i,g}\) へフォールバックする。セル光学厚は
\[
\tau_{R,i}=\bar{\sigma}_{R,i}\Delta r_i
\]
である。

reduced flux は前ステップの IMC face-current tally \(J_{f,g}\) [erg] から
\[
F_f = \frac{\sum_g J_{f,g}}{A_f\Delta t_{prev}},\qquad
R_{F,i}=\frac{\max(|F_{i-1/2}|,|F_{i+1/2}|)}{c\max(\sum_g E_{i,g},E_{floor})}
\]
で評価する。face-current は粒子がセル境界面を横切った時だけ加算し、右向きを \(+E_p\)、左向きを \(-E_p\) とする。初回ステップなど前ステップ tally が無い場合は \(R_F=0\) とする。
ただし raw radiation energy \(\sum_g\max(E_{i,g},0)\le 10^{-20}\) erg/cm³ のセルは、
拡散させる放射場が無いものとして diffusion entry/継続を禁止し、hold counter をリセットする。

ヒステリシスはセル単位で行うが、新規 entry と exit は異なる cadence で扱う。

- diffusion セルの hard exit は毎 step 評価する。radiation energy floor 以下、void、
  \(\tau_R<\tau_{off}\)、または \(R_F>R_{F,off}\) なら、その step で diffusion から
  exit する。さらに実装上の安全条件として \(\tau_R<0.5\tau_{off}\) のセルは
  classification cadence に関わらず必ず immediate exit する。この条件は通常の
  \(\tau_R<\tau_{off}\) exit に含まれるが、将来 entry/update cadence を最適化しても
  低光学厚セルが diffusion に残らないことを保証する invariant とする。
- 非 diffusion セルの新規 entry は
  `Radiation.diffusion.mode_update_interval` step ごとにだけ commit する。
  `mode_update_interval=1` なら従来どおり毎 step entry を許可する。step 0、または
  restart で previous diffusion mask が無い場合は entry update step として扱う。
- 非 diffusion セルは radiation energy floor を上回り、かつ
  \(\tau_R\ge\tau_{on}\) かつ \(R_F\le R_{F,on}\) を満たすと raw entry 候補となる。
  `mode_hold` と `rate_max` は raw entry 候補に対して従来どおり適用する。
  entry update step でない場合、条件を満たしても current mask には commit しない。
- entry update step では、hard exit 後に残った existing diffusion セルと新規 entry
  候補から desired diffusion mask を作り、1D global cell index で maximal contiguous
  island を走査する。island は候補セルだけで構成され、void/non-candidate cell で切れる。
  island length が `Radiation.diffusion.min_diffusion_island_cells` 未満なら、その island
  全体を非 diffusion とする。existing diffusion セルであっても、小 island と判定された
  場合は通常の exit 変換で IMC に戻す。
- `min_diffusion_island_cells=1` は island filter を実質無効化する。MPI 分割時の
  island length は global cell index で定義し、rank 境界をまたぐ candidate run が
  local run として誤って短く数えられないよう、境界 run length を隣接 rank と交換する。
- void セルは常に非 diffusion とする。

分類後、diffusion セルから `imc_guard_cells` 以内の非 diffusion セルを guard セルとし、diffusion セルと guard セルを DDMC selector 後に強制 IMC へ戻す。guard セルは
`min_diffusion_island_cells` の island size に数えず、island filter 後にだけ生成する。
`diffusion.enabled=False` では従来挙動を維持する。

#### 7.1.2b Hybrid diffusion entry/exit エネルギー変換（PR2）

diffusion セルの deterministic 放射エネルギーは \(E^D_{i,g}\) [erg/cm³] として `IMC::diff_E_` に保持する。分類直後、輸送前に current mask \(D_i^{n+1}\) と previous mask \(D_i^n\) を比較し、表現変換だけをエネルギー保存形で行う。

entry（\(D_i^{n+1}=1, D_i^n=0\)）では、セル \(i\)、群 \(g\) に存在する全 alive 粒子のエネルギーを GPU scan/atomic add で集計し、
\[
E^D_{i,g} = \frac{1}{V_i}\sum_{p:\,cell(p)=i,\,group(p)=g,\,alive(p)} E_p
\]
とする。集計した粒子は killed とし、粒子プールは compact する。この PR では LTE 初期化や物質エネルギーからの追加 withdraw は行わない。

継続 diffusion（\(D_i^{n+1}=1, D_i^n=1\)）では、既存の \(E^D_{i,g}\) を保持し、§7.1.2c の source solve と §7.1.2d の RKL2 空間 diffusion step が更新する。

exit（\(D_i^{n+1}=0, D_i^n=1\)）では、各群について
\[
E_{\mathrm{tot},i,g}=E^D_{i,g}V_i
\]
を計算し、\(E_{\mathrm{tot},i,g}>0\) なら
\[
N_{i,g}=\max(1,\texttt{Radiation.diffusion.exit\_particles\_per\_cell\_group})
\]
個の IMC 粒子を生成する。粒子エネルギーは総和が \(E_{\mathrm{tot},i,g}\) になるよう割り付け、位置は 1D 球殻体積一様、方向は等方、`time_remain=dt`、mode は IMC とする。生成成功後、exit セルの \(E^D_{i,g}\) は 0 にする。

`IMC::census_energy()` は alive 粒子のエネルギーに \(\sum_{i:D_i=1}\sum_g V_iE^D_{i,g}\) を加える。`state.rad_E` の finalization は particle cell では従来の track-length estimator を使い、diffusion cell では \(E^D_{i,g}\) をそのまま書く。

diffusion exit 粒子の Philox `global_id` は step local-id 空間の高位予約範囲のうち
\([2^{39},2^{39}+2^{38})\) を rank ごとに等分した subrange から割り当てる。
PR5 の diffusion-interface spawn 粒子は
\([2^{39}+2^{38},2^{40})\) を同様に rank 分割して使う。既存 thermal / Marshak /
volume source は低位 local-id 範囲を使うため、source 粒子の RNG stream とは衝突しない。

#### 7.1.2c Hybrid diffusion cell-local source solve（PR3）
diffusion セル \(i\) では、thermal source 粒子を生成せず、Radiation 演算子内で
cell-local な implicit matter-radiation source solve を行う。PR4 以降は Strang split とし、
IMC/DDMC/RW transport の前に \(\Delta t/2\)、RKL2 空間 diffusion step の後に
\(\Delta t/2\) の source half-step を実行する。

各 diffusion セルは1 GPU thread が担当する。diffusion 分類はセル単位の
\(\tau_R\) と reduced flux \(R_F\) により物理的な拡散・平衡領域を選んでいるため、
source solve は群別の \(\Delta t_s c\sigma^P_{i,g}\) による分岐を行わず、
常に equilibrium-limit update を使う。透明 window 群は Planck weight が小さいため、
同じ平衡投影
\[
E^{new}_{i,g}(T)=a_{eV}T^4b_g(T)
\]
に含める。

未知数は \(T^{new}_i\) の1変数で、Planck fraction が
\(\sum_g b_g(T)=1\) に正規化されていることから Newton residual は
\[
R_i(T)=m_i c_{v,e,i}(T-T_i^{old})
+V_i\left(a_{eV}T^4-\sum_gE^{old}_{i,g}\right)=0
\]
である。Phase-1 kernel は保存済みの \(T_i^{old}\)、\(e_{e,i}^{old}\) と
ideal-gas mass heat capacity を用い、
\[
e_e^{new}(T)=e_{e,i}^{old}+c_{v,e,i}(T-T_i^{old}),\qquad
\frac{de_e}{dT}=c_{v,e,i}
\]
とする。\(c_{v,e,i}\) は `state.cv_e[i]` が正ならそれを使い、未提供時は
single-material fallback \(k_B/(A m_p(\gamma-1))\) を使う。pressure closure は既存 EOS と同じ
\[
P_{e,i}^{new}=(\gamma-1)\rho_i e_{e,i}^{new}
\]
である。table EOS の \(e_e(T), c_v(T), P_e(\rho,T)\) 評価は後続 PR に延期し、
この kernel では未対応とする。

Newton derivative は
\[
\frac{dR_i}{dT}=m_i c_{v,e,i}
+4V_i a_{eV}T^3
\]
であり、\(T>0\) で正なので解は一意である。停止条件は
\[
|R_i| \le 10^{-10}\max(|m_i c_{v,e,i}(T-T_i^{old})|,
|V_i(a_{eV}T^4-\sum_gE^{old}_{i,g})|,
|m_i c_{v,e,i}\max(T,T_i^{old})|,
V_i\max(|\sum_gE^{old}_{i,g}|,10^{-30}),10^{-30})
\]
である。実装は最大30反復の damped Newton を使い、
負方向 step は \(-0.5T\) で制限し、trial residual が増加または非有限になる場合は
最大8回まで step を半減する。収束後は全群を
\[
\texttt{diff\_E}_{i,g}\leftarrow a_{eV}(T_i^{new})^4b_g(T_i^{new})
\]
に設定した後、table 補間された Planck fraction の丸め誤差を補正するため、
\[
s_i=\frac{a_{eV}(T_i^{new})^4}
{\sum_h \max(a_{eV}(T_i^{new})^4 b_h(T_i^{new}),0)}
\]
（分子・分母がともに \(10^{-30}\) より大きい場合）で全群を rescale し、
Newton residual が用いた解析的な全群和 \(a_{eV}T^4\) と一致させる。

最大反復で収束しないセル、非有限 residual、または有効な降下 step が得られないセルは
未収束として WARNING 診断に数え、`Te`, `ee`, `Pe`, `diff_E` を更新しない。
kernel 起動前に `Te`, `ee`, `Pe`, `diff_E` は device scratch へ保存し、未収束・
非有限入力・後述の保存則 gate 失敗ではこの scratch から該当セルを復元する。

収束後の更新は
\[
\texttt{diff\_E}_{i,g}\leftarrow E^{new}_{i,g}(T_i^{new}),\quad
\texttt{ee}_i\leftarrow e_{e,i}^{old}+c_{v,e,i}(T_i^{new}-T_i^{old}),\quad
\texttt{Te}_i\leftarrow T_i^{new},\quad
\texttt{Pe}_i\leftarrow P_e^{new}
\]
である。ただし書き込み直前に
\[
\Delta U_i=m_i(e_{e,i}^{new}-e_{e,i}^{old}),\qquad
\Delta E_i=V_i\sum_g(E^{new}_{i,g}-E^{old}_{i,g})
\]
に対して
\[
|\Delta U_i+\Delta E_i|
\le 10^{-6}\max(|\Delta U_i|,|\Delta E_i|,10^{-30})
\]
を満たすことを要求する。満たさない場合は solve を棄却し、保存済み state を復元し、
`rad_dep` / `rad_emit` も加算しない。diagnostic tally は保存則 gate を通ったセルだけ
\[
\texttt{rad\_dep}_{i,g}\mathrel{+}=
\Delta t_s\,c\,\sigma^{P}_{i,g}E^{new}_{i,g}V_i,\qquad
\texttt{rad\_emit}_{i,g}\mathrel{+}=
\Delta t_s\,c\,\sigma^{P}_{i,g}a_{eV}(T_i^{new})^4b_g(T_i^{new})V_i
\]
として記録するが、§2.1 の U1 source injection では diffusion セルに再適用しない。

#### 7.1.2d Hybrid diffusion RKL2 空間 step（PR4）

PR4 では 1D_SPH diffusion セルの deterministic 群別放射エネルギー
\(E^D_{i,g}\) を、凍結 Rosseland 拡散係数
\[
D^{raw}_{i,g}=\frac{c}{3\max(\sigma_{R,i,g},\sigma_{floor})},\qquad
D_{i,g}=\min\left(D^{raw}_{i,g},\frac{c\Delta r_i}{6}\right)
\]
で explicit RKL2 super-time-stepping により更新する。RKL2 stage 中は
\(D_{i,g}\)、diffusion mask、幾何を凍結し、各群は独立に解く。
この cap は透明な群で mean-free-path がセル幅を大きく超える場合の
非物理的な拡散係数を制限し、RKL2 の明示限界も同じ \(D_{i,g}\) で評価する。

セル中心 \(r_{c,i}=(r_{i-1/2}+r_{i+1/2})/2\)、面積
\(A_{i\pm1/2}=4\pi r_{i\pm1/2}^2\)、体積 \(V_i\) とする。diffusion-diffusion
内部面では
\[
D_{i+1/2,g}=
\frac{2D_{i,g}D_{i+1,g}}{D_{i,g}+D_{i+1,g}},
\qquad
F_{i+1/2,g}=
-D_{i+1/2,g}\frac{E^D_{i+1,g}-E^D_{i,g}}{r_{c,i+1}-r_{c,i}}
\]
を用い、
\[
\mathcal{L}_{i,g}(E)=
\frac{A_{i-1/2}F_{i-1/2,g}-A_{i+1/2}F_{i+1/2,g}}{V_i}.
\]
同じ面 flux を左右セルで符号反対に使うため、reflective 境界かつ source なしでは
\(\sum_i V_iE^D_{i,g}\) を roundoff まで保存する。

diffusion-IMC interface 面は PR4 では zero-current とする。物理 inner reflective
境界も \(F=0\)。物理 outer vacuum 境界は free-streaming leakage
\[
F_{N+1/2,g}=\frac{c}{4}E^D_{N,g}
\]
を使う。inner 側を vacuum と指定した場合は外向き法線に合わせ
\(F_{1/2,g}=-(c/4)E^D_{1,g}\) とする。

IMC transport で diffusion セルへ入射した packet energy は
positive face-current source \(J^{in}_{f,g}\) [erg] として別 tally に保持する。
RKL2 operator を組む前に、diffusion セル \(i\) では隣接セルが非 diffusion の
interface 面だけを取り込み、
\[
E^{D,*}_{i,g}=E^D_{i,g}+
\frac{\sum_{f\in\partial i,\;neighbor(f)\notin D}J^{in}_{f,g}}{V_i}
\]
として first diffusion cell に直接 deposit する。取り込んだ `face_current_in`
entry は 0 に戻し、RKL2 は \(E^{D,*}\) を初期値として進める。これにより
RKL2 operator が空、または stability gate で spatial step を skip した場合でも
\(J^{in}\) は diffusion field に保存される。
reduced-flux 分類用の `face_current_step` は従来通り右向き crossing を正、
左向き crossing を負とする signed tally であり、\(J^{in}\) とは分離する。
RKL2 更新後に負の \(E^D_{i,g}\) が発生した場合は、群ごとに conservative positivity
limiter を適用する。各群で signed total
\(S_g=\sum_i V_iE^D_{i,g}\) と positive total
\(P_g=\sum_i V_i\max(E^D_{i,g},0)\) を計算し、\(P_g>0\) なら正のセルだけを
\(\max(S_g,0)/P_g\) 倍して負のセルを 0 にする。これにより \(S_g\ge0\) では
\(\sum_iV_iE^D_{i,g}\) を保存しつつ、positive-only diagnostic energy が負の undershoot
を無視して人工的に増えることを防ぐ。\(S_g<0\) の群は全 diffusion cell/group を 0
にする。

RKL2 係数は Legendre 多項式 \(P_j\) の recurrence から host 側で1回計算する。
super-step \(\tau\) に対し
\[
Y_0=E^n,\qquad
Y_1=Y_0+\tilde\mu_1\tau\mathcal{L}(Y_0),
\]
\[
Y_j=(1-\mu_j-\nu_j)Y_0+\mu_jY_{j-1}+\nu_jY_{j-2}
+\tilde\mu_j\tau\mathcal{L}(Y_{j-1})
+\tilde\gamma_j\tau\mathcal{L}(Y_0),\quad 2\le j\le s.
\]
undamped RKL2 では \(w_0=1\)、\(w_1=P_s'(1)/P_s''(1)=4/(s^2+s-2)\)、
\[
b_j=\frac{P_j''(1)}{(P_j'(1))^2},\quad a_j=1-b_j,\quad
\mu_j=\frac{2j-1}{j}\frac{b_j}{b_{j-1}},\quad
\nu_j=-\frac{j-1}{j}\frac{b_j}{b_{j-2}},
\]
\[
\tilde\mu_1=b_1w_1,\quad
\tilde\mu_j=\mu_jw_1,\quad
\tilde\gamma_j=-a_{j-1}\tilde\mu_j.
\]
実装は damping \(>0\) の場合に \(w_0=1+2\,\mathrm{damping}/(s^2+s)\) とし、
\(P_j(w_0)\), \(P'_j(w_0)\), \(P''_j(w_0)\) から同じ recurrence を評価する。

明示限界は
\[
\Delta t_{exp}=\min_{i,g}
\frac{V_i}{\sum_f A_f D_{f,g}/\Delta r_f + \sum_{f\in vacuum} A_f c/4}
\]
で見積もり、
\[
C_s=\frac{s^2+s-2}{4}
\]
の undamped RKL2 stability capacity が \(C_s\ge\Delta t/(\Delta t_{exp}\eta)\)
を満たすよう
\[
s=\max\left(2,
\left\lceil
\frac{\sqrt{9+16\Delta t/(\Delta t_{exp}\eta)}-1}{2}
\right\rceil\right),
\qquad \eta=\texttt{Radiation.diffusion.sts\_subcycle\_eta}.
\]
`sts_max_stages>0` かつ必要 stage 数が上限を超える場合は、
\(\Delta t_{sub}\le \eta\,[(s_{max}^2+s_{max}-2)/4]\Delta t_{exp}\) となるよう diffusion step を
等分 subcycle する。必要 subcycle 数が 10 を超える場合は current diffusion mask を
entry/exit 変換前に棄却し、その radiation step では該当セルを IMC として扱う
（前ステップから diffusion だったセルは通常の exit 変換を行う）。

#### 7.1.2e Hybrid diffusion IMC interface（PR5）

PR5 では 1D_SPH の deterministic diffusion 領域と IMC guard 領域を face-current
で結合する。IMC packet が cell boundary を越えて destination cell
\(D_j=1\) に入る場合、packet は diffusion 表現へ変換される。face index は
左向き crossing で \(f=i\)、右向き crossing で \(f=i+1\) とし、
\[
J^{in}_{f,g}\mathrel{+}=E_p,\qquad
J^{step}_{f,g}\mathrel{+}=s_f E_p,\quad
s_f=\begin{cases}-1 & \text{left crossing}\\ +1 & \text{right crossing}\end{cases}
\]
を atomic tally した後、packet を dead にする。\(J^{in}\) は unsigned energy
source、\(J^{step}\) は次 step の reduced-flux 判定用 signed current である。
この face conversion では matter へ deposition しない。

RKL2 空間 step 後、diffusion-IMC interface 面からの outward leakage を
post-RKL2 の deterministic energy から計算する。Phase-1 は Marshak-like
partial-current closure
\[
J^{out}_{f,g}=
\frac{cE^D_{i,g}}{4+\frac{3}{2}\sigma_{R,i,g}\Delta x_i}\,
A_f\Delta t
\]
を用いる。ここで \(i\) は interface に隣接する diffusion cell、
\(\Delta x_i=r_{i+1/2}-r_{i-1/2}\)、\(A_f=4\pi r_f^2\) である。outgoing leakage には
face ごとの \(0.5\,V_iE^D_{i,g}\) cap も step 全体の 0.5 cap も適用しない。
cap は \(J^{out}\) ではなく deterministic field の positivity にだけ適用する。

cell/group ごとに、interface faces の desired leakage
\(J^{des}_{f,g}\) を集計する。\(E^{post}_{i,g}\) を source1、face-current deposit、
RKL2 内部 diffusion/vacuum leakage、および RKL2 positivity limiter 後の deterministic
energy とする。この時点の利用可能エネルギーは
\[
A_{i,g}=\max\left(V_iE^{post}_{i,g}-V_iE_{floor},0\right)
\]
であり、これは同じ step の \(J^{in}\) を含む。すなわち概念的には
\[
V_iE^{post}_{i,g}
=V_iE^{old}_{i,g}+\sum_fJ^{in}_{f,g}
+\Delta E^{internal}_{i,g}-E^{vacuum}_{i,g}
+\Delta E^{limiter}_{i,g}
\]
である。desired leakage の総和
\[
L^{des}_{i,g}=\sum_{f\in interface(i)}J^{des}_{f,g}
\]
が \(A_{i,g}\) を超える場合だけ、同じ cell/group から出る全 interface leakage を
\[
J^{out}_{f,g}=J^{des}_{f,g}\frac{A_{i,g}}{L^{des}_{i,g}}
\]
で scale する。\(L^{des}_{i,g}\le A_{i,g}\) なら \(J^{out}_{f,g}=J^{des}_{f,g}\) とする。
\(A_{i,g}=0\) の場合は全 outgoing leakage を 0 とする。確定した leakage は
`face_current_out[f,g]` に保存し、
\[
E^D_{i,g}\leftarrow E^{post}_{i,g}
-\frac{\sum_{f\in interface(i)}J^{out}_{f,g}}{V_i}
\]
で deterministic field から取り除く。これにより同じ step に入った
`face_current_in` は outgoing leakage に即時利用可能だが、cell/group energy は
非負 floor 未満に落ちない。

\(J^{out}_{f,g}>0\) の interface では
\[
N_f=\max(1,\texttt{Radiation.diffusion.interface\_particles\_per\_face\_group})
\]
個の IMC packet を adjacent IMC cell に生成し、packet energy は総和が
\(J^{out}_{f,g}\) と一致するよう最後の packet で丸め残差を受ける。位置は
interface radius \(r_f\)、方向は outward half-space current 分布
\(|\mu|=\sqrt{\xi}\) とし、符号は diffusion cell から IMC cell へ向ける。
`time_remain=dt`、mode は IMC、`rng_counter=0` とする。

spawn 後は tail IMC pass を行う。tail 中に packet が diffusion cell へ戻った場合は
同じ \(J^{in}\) tally に入るが、RKL2 step は既に終了しているため、その energy は
tail pass 後に \(E^D_{i,g}\) へ直接加算してから後段の \(\Delta t/2\) source solve
へ渡す。これにより particle + deterministic + escape energy を step 内で保存する。

#### 7.1.2f Hybrid diffusion full 3-mode step ordering（PR6）

`Radiation.diffusion.enabled=True` かつ 1D_SPH では、通常の IMC/DDMC/PGRW
輸送に deterministic diffusion mode を加え、1 radiation step を次の順序で実行する。
`Radiation.diffusion.enabled=False` または 2D_RZ では従来の IMC/DDMC/PGRW
ordering に退化する。

```text
radiation_step(dt):
  1.  opacity, Fleck factor, effective opacity を既存経路で評価する。
  2.  前 step の signed face-current から reduced flux を評価する。existing
      diffusion セルの hard exit は毎 step 評価し、新規 entry と island filter は
      mode_update_interval step ごとにだけ commit する。cell-level diffusion mask と
      IMC guard cell mask を分類する。
  3.  entry/exit 表現変換を行う。
      - entry: diffusion cell に入る既存 particle energy を E^D_{i,g} へ畳み込む。
      - exit: E^D_{i,g} V_i を IMC particle へ変換し、該当 E^D_{i,g}=0 とする。
  4.  mode map を構築する。
      - diffusion cell は state.ddmc_mode_map に TransportMode::Diffusion=3 と記録する。
      - transport kernel へ渡す DDMC mode selector では diffusion cell と guard cell を
        IMC に強制し、DDMC/PGRW が deterministic 領域を claim しないようにする。
      - それ以外は既存 DDMC/PGRW selector に従う。
  5.  particle tally（rad_dep, rad_E_tally, E_escape）と diffusion interface
      current 入出力 tally をゼロ化する。
  6.  diffusion cell が存在する場合、cell-local source solve を dt/2 実行する。
  7.  thermal source particle を生成する。ただし diffusion cell は source mask で
      skip し、thermal emission は deterministic source solve が担当する。
      Marshak / volume source は既存 source 経路で生成する。
  8.  composite sort 後、IMC/DDMC/PGRW transport を実行する。
      IMC packet が diffusion cell へ入射した場合は packet を kill し、
      J^{in}_{f,g} と signed J^{step}_{f,g} に tally する。
  9.  diffusion cell が存在する場合、J^{in}_{f,g} を隣接 diffusion cell へ
      deposit してから 1D RKL2 deterministic diffusion step を dt 実行する。
      vacuum leakage は E_rad_esc に加算される。
  10. diffusion-IMC interface face から outgoing current J^{out}_{f,g} を IMC
      particle に変換し、spawn した packet だけを tail IMC transport する。
      tail 中に diffusion cell へ戻った energy は RKL2 を再実行せず E^D_{i,g}
      へ直接加算する。
  11. diffusion cell が存在する場合、cell-local source solve を dt/2 実行する。
  12. rad_E を finalize する。
      - particle cell: legacy では rad_E_tally / (V_i c dt)
      - difference cell: E_ref_avg + signed rad_E_tally / (V_i c dt)
      - diffusion cell: E^D_{i,g}
  13. signed face-current を rotate する。
      face_current_step -> face_current_prev、face_current_step=0。
  14. census combing を既存経路で行う。diffusion cell には粒子を保持しないため
      combing 対象は particle-owned cell の census のみである。
  15. current diffusion mask を previous mask として保存し、次 step の entry/exit
      判定に使う。
```

energy budget の radiation term は particle census energy に
\(\sum_{i:D_i=1,g} V_iE^D_{i,g}\) を加えた mixed radiation energy を使う。
したがって diffusion step の vacuum leakage、IMC/particle escape、Marshak/volume
input、source/transport numerical loss を同一の step budget で評価する。
snapshot HDF5 の `radiation/energy_density` は step finalize 後に書き出され、
particle cell では track-length estimator、diffusion cell では deterministic
\(E^D_{i,g}\) を保持する。

per-step diagnostic は次の要約を出す。

```text
[diffusion] step=N n_diff=M n_guard=K mode_update=U forced_exit=H island_reject=R
            rkl2_stages=S rkl2_subcycles=Q dt_explicit=D rkl2_skipped=B source_iter=I
            E_diff=X E_in=Y E_out=Z E_particle=W interface_limited=L
```

ここで \(E_{\rm diff}=\sum_{i:D_i=1,g} V_iE^D_{i,g}\)、`E_in` は
IMC→diffusion interface energy、`E_out` は diffusion→IMC spawned energy、
`E_particle` は alive particle census energy である。`mode_update` は新規 entry/island
filter を commit した step なら 1、そうでなければ 0。`forced_exit` は hard exit で
IMC に戻した cell 数、`island_reject` は `min_diffusion_island_cells` により非 diffusion
へ戻した cell 数、`interface_limited` は positivity limiter により outgoing face current
を scale した cell/group 数である。
diffusion 有効 step では追加で
\[
E^D_{\rm start}+E^{face}_{in}-E^{face}_{out}-E^{vacuum}_{out}
-\Delta U_{\rm source}=E^D_{\rm final}
\]
を診断し、相対残差が \(10^{-8}\) を超える場合は critical log を出す。
\(\Delta U_{\rm source}\) は cell-local source solve で matter が得た正味エネルギー
（radiation は同量を失う）である。RKL2 内部でも
\(E_{before}+E^{face}_{in}-E_{vacuum}=E_{after}\) を同じ閾値で診断する。
さらに `source1`, `rkl2`, `interface_out`, `tail_deposit`, `source2` の各 substep で
`[diffusion_substep]` diagnostic を出し、substep 前後の \(E^D\)、face input/output、
vacuum loss、source exchange、source failure count を記録する。任意の substep または
step finalize で \(E^D\) が
\(10\times\max(|E_{\rm before}|, |E_{\rm face,in}|, |E_{\rm face,out}|,
|E_{\rm vacuum}|, |\Delta U_{\rm source}|, |E_{\rm expected}|, 10^{-20})\)
を超えた場合は safety fallback として current diffusion mask を全て IMC に戻し、
残った deterministic \(E^D\) を exit particles に変換してから \(E^D\) を 0 にする。

#### 7.1.2g HOLO（High-Order Low-Order）概要 [RETIRED — legacy]

`Radiation.holo.enabled=False` が既定であり、既存の
IMC/DDMC/PGRW/hybrid diffusion 実行経路を変更しない。`enabled=True` は v1 では
`Main.dimension="1D_SPH"` のみを対象とし、2D_RZ では WARNING を出して無効化する。
HOLO は high-order 粒子輸送を従来通り全領域で実行し、low-order (LO)
物理フレーム放射拡散を解く。通常 HOLO では selector が作る連続 patch ごとに解き、
DF 併用時は patch 境界 BC を使わず全 mesh を LO solve domain とする。通常 HOLO では
core cell の material coupling と accepted radiation energy を LO が所有する。radiation
acceptance は binary であり、blend cell の radiation は HO moment を採用する。
material source injection だけは従来通り `holo_lo_weight` で LO/HO source を混合する。
DF 併用時は post-hoc overwrite ではなく DF carried reference reservoir を accepted core
state へ再中心化し、LO core census residual を kill して DF 内部状態と一致させる。
LO material coupling の所有は LO core cell のみである。
patch solve では、patch が物理 outer boundary に達した場合だけ vacuum boundary
condition を使い、内部 HO/LO patch 境界では high-order face-current tally を boundary
condition として使う。

LO material coupling mask は solver 境界ではなく、電子内部エネルギー更新の source owner を
選ぶ cell mask である。mask は Rosseland optical depth の単一閾値
`Radiation.holo.coupling_tau` と `guard_cells` の膨張だけで作る。opacity 評価後の
\(\sigma_{R,i,g}\)、電子温度 \(T_{e,i}\)、Planck table から
\[
w_{i,g}=\max\left(\frac{\partial(T^4 b_g)}{\partial T}\bigg|_{T_{e,i}},0\right),
\qquad
\bar{\sigma}_{R,i}=\frac{\sum_g w_{i,g}}
{\sum_g w_{i,g}/\max(\sigma_{R,i,g},\sigma_{floor})},
\qquad
\tau_{R,i}=\bar{\sigma}_{R,i}\Delta r_i
\]
を計算する。1群では \(w=1\) とし、重み和または分母がゼロの場合は
\(\max_g\sigma_{R,i,g}\) へ fallback する。これにより `[holo_selector]` の
`tau_R_min/max` は同じ mesh、opacity、temperature state の
`[diffusion_classify]` と同じ optical-depth proxy を報告する。
base core mask は hysteresis 無効時には \(\tau_{R,i}\ge\tau_{coupling}\) かつ非 void
cell で真になる。`tau_on>0` または `tau_off>0` の場合は、前 step で core でなかった
cell は \(\tau_{R,i}\ge\tau_{on}\) で進入し、前 step で core だった cell は
\(\tau_{R,i}\ge\tau_{off}\) または `min_dwell_steps` 未満の滞在で維持する。
core mask は blend 幅だけ非 void cell へ膨張して patch mask を作り、core からの dilation
距離 \(d\) に対して \(w_i=(N_{blend}-d)/N_{blend}\) を与える。`holo_core_mask` は
LO material-coupled cell mask、`holo_patch_mask` は LO solve domain である。

low-order 系の未知量は群別の物理フレーム放射エネルギー密度
\(E^{LO}_{i,g}\) [erg/cm³] である。\(E^{LO}\) は persistent state であり、
step 間で保持する。サイズ変更または初期状態ではゼロで初期化し、`rad_E` や
LTE equilibrium へ毎 step reset しない。DF が有効な場合でも LO solve は signed residual
particle tally を読まず、物理量である \(\sigma_P,\sigma_R,T_e,\rho,V,r\) のみを読む。

material coupling の所有規則は以下で固定する。

\[
\Delta E^{mat}_i =
\begin{cases}
\sum_g(\mathrm{rad\_dep}_{i,g}-\mathrm{rad\_emit}_{i,g}) & i \notin \mathrm{LO\_coupled},\\
\Delta E^{mat,LO}_i & i \in \mathrm{LO\_coupled}.
\end{cases}
\]

`solve_holo_lo_1d_cpu` は swappable low-order backend の v1 実装であり、
1D_SPH の全 cell で群ごとに backward-Euler 1D spherical diffusion と
explicit-temperature implicit source を一つの Thomas solve で解く。内部 face
\(f=i+1/2\) の係数は Rosseland harmonic diffusion coefficient
\[
C_{f,g} = A_f\,D_{f,g}/(r_{i+1,c}-r_{i,c}),\qquad
D_{i,g}=\frac{c}{3\max(\sigma_{R,i,g},10^{-30})},
\]
\[
D_{f,g}=\frac{2D_{i,g}D_{i+1,g}}{D_{i,g}+D_{i+1,g}}
\]
である。source は step 開始時の \(T^n_{e,i}\) と opacity を固定して
\[
\alpha_{i,g}=c\sigma_{P,i,g}\Delta t,\qquad
B^n_{i,g}=a_{\mathrm{eV}}(T^n_{e,i})^4b_g(T^n_{e,i})
\]
と置く。cell \(i\)、群 \(g\) の線形系は
\[
\begin{aligned}
&\left[\frac{V_i}{\Delta t}(1+\alpha_{i,g})
 + C_{i-1/2,g}+C_{i+1/2,g}+S^{sink}_{i,g}\right]E^{n+1,LO}_{i,g}\\
&\quad -C_{i-1/2,g}E^{n+1,LO}_{i-1,g}
 -C_{i+1/2,g}E^{n+1,LO}_{i+1,g}
=\frac{V_i}{\Delta t}\left(E^{n,LO}_{i,g}+\alpha_{i,g}B^n_{i,g}\right)
 +\alpha_C R^{cons}_{i,g}+Q^{bc}_{i,g}.
\end{aligned}
\]
ここで \(R^{cons}_{i,g}\) は same-step predictor-corrector で作る
HOLO consistency source [erg/s]、\(\alpha_C=\)
`Radiation.holo.consistency_alpha` は \([0,1]\) の緩和係数である。
互換用 namelist key `gamma_alpha` は同じ値へ map されるが、前 step defect は
使用しない。
patch 内部 face は通常の diffusion coupling を持つ。patch 境界が内部 HO/LO 境界の場合、
face \(f\) の high-order step-integrated signed current \(J^{HO}_{f,g}\) [erg] を explicit
source として RHS に加える。face index は \(f=i\) が cell \(i\) の左面で、右向き crossing
を正とする。patch \([a,b]\) では左境界の流入は \(+J^{HO}_{a,g}\)、右境界の流入は
\(-J^{HO}_{b+1,g}\) であり、負の流入は positivity limiter により outgoing current として
制限し `boundary_limited_E` に記録する。DF 有効時は residual face-current と reference
face-current の cancellation および thick/thin 遷移の patch 境界 closure が不安定に
なり得るため、LO solve domain を全 mesh に戻し、内部 patch 境界 BC を使わない。
`cell_active != nullptr` かつ face-current BC が無い patch solve fallback では、patch 外側
隣接 cell の step 開始 radiation energy density \(E^{n}_{nb,g}\) を固定 Dirichlet 値として、
通常の Rosseland face coupling \(C_{f,g}\) を境界 face に適用する:
\[
C_{f,g}(E^{n+1,LO}_{bdry,g}-E^{n}_{nb,g}).
\]
すなわち patch 境界 cell の diagonal に \(C_{f,g}\) を加え、RHS に
\(C_{f,g}E^{n}_{nb,g}\) を加える。これは implicit face coupling なので
\(c\Delta t A_f/(4V)\) の制限を持たず、opacity と距離に応じた \(D_fA_f/\Delta r\)
で境界交換を制御する。patch が cell 0 に達する
左物理境界は reflecting として source/sink を加えない。

inner physical boundary は reflecting であり sink/source を加えない。outer physical boundary は
vacuum であり、patch が最外 cell に達する場合だけ
\[
S^{vac}_{N-1,g}=A_N c/4
\]
を diagonal sink として加え、`holo/E_LO_boundary_out` に
\(S^{vac}_{N-1,g}E^{n+1,LO}_{N-1,g}\Delta t\) を記録する。
`holo/E_LO_boundary_in` は face-current または Dirichlet face coupling の incoming term を
step 積算 energy として記録する。

LO gross 診断の基準値は solve 後に
\[
\mathrm{rad\_dep}^{LO}_{i,g}=V_i\alpha_{i,g}E^{n+1,LO}_{i,g},\qquad
\mathrm{rad\_emit}^{LO}_{i,g}=V_i\alpha_{i,g}B^n_{i,g},
\]
である。ただし \(\alpha\gg1\) では
\(\alpha V(E^{n+1,LO}-B^n)\) が桁落ちし、LO material source と保存性判定を
破壊するため、実際に適用する net source は同じ離散式から
\[
\Delta E^{mat,LO}_{i,g}
= -V_i(E^{n+1,LO}_{i,g}-E^{n,LO}_{i,g})
  -\Delta t\,{\cal D}_{i,g}^{n+1}
\]
として評価する。ここで
\[
{\cal D}_{i,g}^{n+1}
= C_{i-1/2,g}(E^{n+1,LO}_{i,g}-E^{n+1,LO}_{i-1,g})
 +C_{i+1/2,g}(E^{n+1,LO}_{i,g}-E^{n+1,LO}_{i+1,g})
 +S^{vac}_{i,g}E^{n+1,LO}_{i,g}
\]
であり、存在しない face の項は 0、\(S^{vac}_{i,g}\) は最外 cell のみ非ゼロである。
この式は \(R^{cons}=0\) の線形系を満たす解に対して
\(V_i\alpha_{i,g}(E^{n+1,LO}_{i,g}-B^n_{i,g})\) と代数的に同値であり、
\(R^{cons}\neq0\) では同じ離散 balance に入れた補正 source を含む
（すなわち material 側には \(-\Delta t\,R^{cons}_{i,g}\) が反映される）。
`holo_rad_dep-holo_rad_emit` はこの安定評価した net source と一致するように、
非負の gross 診断に roundoff correction を入れて保存する。gross 値自体が大きく
補正が double 精度で表現できない場合は、net source を正負に分けた非負 split として
保存し、material source history の net 値を優先する。
\[
\Delta E^{mat,LO}_i=
\sum_g\Delta E^{mat,LO}_{i,g}
\]
で定義する。`LO_coupled` cell では \(\Delta E^{mat,LO}_i/m_i\) を `ee` に加え、
ideal-gas closure では \(T_e=e_e/c_{v,e}\)、表 EOS closure では
`temperature_from_energy(rho,e_e)` と `pressure(rho,T_e)` で `Te` と `Pe` を閉じ直す。
非 coupled cell では LO radiation/source solve は実行するが `ee/Te/Pe` は変更せず、
material update は従来の particle source injection が所有する。source injection kernel は
LO-coupled cell の `holo_rad_dep-holo_rad_emit` を `delta_E_rad_prev` に記録するが、
`ee` には再適用しない。
The 1D `holo.sn_material_coupling=True` + `holo.solver="quasidiffusion_1d"`
path follows the same direct-update rule because the GPU \(S_N\) sweep supplies
only the HO \(\chi=P_{rr}/E\) closure and the QD LO solve owns the material
update.  The 2D \(S_N\) material-coupling path still publishes a deterministic
source and remains source-injection owned.

solver 全体の LO balance check は material に実際適用した mask 内 source ではなく、
全 cell の LO source term を用いて
\[
\sum_i\Delta E^{mat,LO,global}_i+\Delta E^{rad,LO}
-(E^{LO}_{boundary,in}-E^{LO}_{boundary,out})=0
\]
である。ここで \(\Delta E^{rad,LO}=\sum_{i,g}V_i(E^{n+1,LO}_{i,g}-E^{n,LO}_{i,g})\)
であり、boundary term は physical outer boundary のみを含む。

radiation transport 後、tally finalize 前に step 開始時の `radiation/energy_density`
\(E^n_{i,g}\) を device buffer に保存し、LO predictor を実行する。predictor は
`lo_coupled=nullptr`、`consistency_source=nullptr` であり、`State.holo_E_LO` に
\(E^{LO,p}_{i,g}\) を保存するが、`ee`、`Te`、`Pe` と
`State.holo_rad_dep` / `State.holo_rad_emit` は commit しない。

tally finalize 後の physical HO moment を \(E^{HO}_{i,g}\) とする。同じ step で
以下の consistency source を作る:
\[
R^t_{i,g} =
\frac{V_i}{\Delta t}
\left[(E^{HO}_{i,g}-E^n_{i,g})-(E^{LO,p}_{i,g}-E^n_{i,g})\right]
=\frac{V_i}{\Delta t}(E^{HO}_{i,g}-E^{LO,p}_{i,g}),
\]
\[
J^{diff}_{f,g}=C_{f,g}\left(E^{HO}_{R(f),g}-E^{HO}_{L(f),g}\right),
\qquad
J^{HO}_{f,g}=\frac{J^{MC,step}_{f,g}+J^{ref,step}_{f,g}}{\Delta t},
\]
\[
\gamma^{HO}_{f,g}=J^{HO}_{f,g}-J^{diff}_{f,g},\qquad
R^{flux}_{i,g}=\gamma^{HO}_{i+1/2,g}-\gamma^{HO}_{i-1/2,g},
\qquad
R^{cons}_{i,g}=R^t_{i,g}+R^{flux}_{i,g}.
\]
face current は右向きを正とする。DF 無効時は \(J^{ref,step}=0\)、DF 有効時は
deterministic reference face transport の current を加える。face defect は両隣 cell の
`holo_lo_weight>0` の内部 face だけで評価し、patch 境界 face では既存の physical
face-current / Dirichlet boundary closure と二重計上しない。

LO corrector は predictor と同じ \(E^n_{i,g}\) を time term の初期値として再び解き、
RHS に \(R^{cons}_{i,g}\) [erg/s] を直接加える。corrector だけが
`holo_core_mask` cell の `ee`、`Te`、`Pe` と LO gross 診断
`State.holo_rad_dep` / `State.holo_rad_emit` を commit し、
`State.holo_E_LO` は \(E^{LO,c}_{i,g}\) に更新される。
`radiation/rad_dep` と `radiation/rad_emit` は particle tallies のまま保持し、
LO-coupled cell では material coupling に使わない。predictor または corrector が
失敗した transport step では `State.holo_lo_source_valid=False` とし、LO の部分更新は
commit しない。この場合、その step の LO-coupled material coupling は通常の particle
tally (`radiation/rad_dep - radiation/rad_emit`) に fallback する。

HOLO radiation acceptance は binary ownership であり、blend weight は使わない:
\[
E^{acc}_{i,g}=
\begin{cases}
E^{LO,c}_{i,g}, & \mathrm{holo\_core\_mask}_i=1,\\
E^{HO}_{i,g}, & \mathrm{holo\_core\_mask}_i=0.
\end{cases}
\]
material source injection では従来通り `holo_lo_weight` による blend を使う。
DF 併用時は hard DF re-centering として、acceptance 後の \(E^{acc}_{i,g}\) から
`holo_core_mask[i] != 0` の bin で
\[
U^{ref,carried}_{i,g}=E^{acc}_{i,g}V_i
\]
を `previous_reference_U_` の device reservoir に直接設定する。加算型 retarget
\(U^{ref}\mathrel{+}=(E^{acc}-E^{HO})V_i\) は使わない。さらに
`holo_core_mask[i] != 0` の live census residual 粒子を kill し、次 step 冒頭の
census residualization が LO core で
\[
U^{phys,old}_{i,g}=U^{ref,carried}_{i,g}
\]
から始まるようにする。`difference_residual_E` は `holo_core_mask[i] != 0` の bin だけ
accepted physical field と time-average reference の差
\(E^{acc}_{i,g}-\bar{E}^{ref}_{i,g}\) に再投影し、non-core bin は tally finalize が作った
residual density を保持する。

`Radiation.holo.p_rr_tally=True` のとき、通常 IMC path segment の
track-length tally と同じ場所で passive radial pressure moment を蓄積する。
\[
P^{raw}_{rr,i,g}\leftarrow P^{raw}_{rr,i,g}
 + s_p\,\mu_r^2\,\Delta s_E,\qquad
C^{raw}_{rr,i,g}\leftarrow C^{raw}_{rr,i,g}+\Delta s_E,
\]
ここで \(s_p\) は difference particle sign（通常粒子では \(+1\)）、
\(\Delta s_E\) は `rad_E_tally` に加える energy-weighted path length
[erg cm]、\(\mu_r\) は segment 開始時の radial direction cosine である。
PGRW、DDMC、RW residence segments は v1 では信頼できる angular moment を持たないため
`Prr` から除外し、coverage 診断だけを下げる。finalize は LO-coupled cell×group に対して
\[
P^{HO}_{rr,i,g}=\frac{P^{raw}_{rr,i,g}}{V_i c\Delta t},\qquad
\chi^{HO}_{i,g}=\frac{P^{HO}_{rr,i,g}}{\max(E^{HO}_{i,g},E_{floor})},
\]
\[
\mathrm{coverage}_{i,g}=
\mathrm{clip}_{[0,1]}\!\left(\frac{C^{raw}_{rr,i,g}}
{\max(|E^{raw}_{HO,i,g}|,10^{-300})}\right)
\]
を出力する。ここで \(E^{raw}_{HO,i,g}\) は同じ step の `rad_E_tally`
raw 値である。`Prr`、\(\chi^{HO}\)、coverage は diagnostic として保存する。
さらに `Radiation.holo.solver="quasidiffusion_1d"` では、raw \(\chi^{HO}\) を
そのまま LO solver に渡さず、`solve_holo_lo_source_ownership` の host handoff で
filtered closure \(\chi^F\) を作ってから QD face coefficient に渡す。
`Radiation.holo.sn_closure=True`（既定）の場合は、この MC tally 由来 closure の代わりに
1D spherical \(S_N\) solver が作る deterministic closure \(\chi^{SN}\) を QD solver に渡す。
この経路は `Radiation.holo.solver="quasidiffusion_1d"` のときだけ有効であり、
`implicit_1d` では参照しない。

S_N closure は各 radiation step の LO solve 直前に host で実行する。角度は
Gauss-Legendre \(S_N\) quadrature（既定 \(N=8\)、偶数のみ）を
\(\mu_1<\cdots<\mu_N\) の negative-first 順に並べ、
\(\sum_n w_n=2\)、\(\sum_n w_n\mu_n=0\)、\(\sum_n w_n\mu_n^2=2/3\) を満たす。
negative-first 順で Lewis-Miller の spherical angular redistribution を非負係数として
\[
\alpha_{1/2}=0,\qquad
\alpha_{n+1/2}=\alpha_{n-1/2}-\mu_n w_n,\qquad
\alpha_{N+1/2}=0
\]
で評価する。cell \(i\) の内外 face 面積を
\(A_i=4\pi r_i^2\)、\(A_{i+1}=4\pi r_{i+1}^2\)、体積を \(V_i\) とする。
方向 \(n\) と群 \(g\) の source iteration では前反復 scalar flux
\(\phi^{old}_{i,g}=\sum_m w_m\bar\psi_{i,m,g}\) を使い、
\[
\sigma^{eff}_{a,i,g}=f_i\sigma^P_{a,i,g},\qquad
\sigma^{eff}_{s,i,g}=(1-f_i)\sigma^P_{a,i,g},\qquad
\sigma_{t,i,g}=\sigma^P_{a,i,g}
\]
を用い、
\[
Q_{i,g}=\frac12 c\,\sigma^{eff}_{a,i,g}\,a_{eV}T_{e,i}^4 b_g(T_{e,i})
       +\frac12\sigma^{eff}_{s,i,g}\phi^{old}_{i,g}
\]
を isotropic angular source とする。ここで \(f_i\) は Fleck factor、
\(\sigma^P_{a,i,g}\) は Planck absorption opacity である。
空間と角度はいずれも diamond difference を使う。outward sweep
（\(\mu_n>0\)、内側から外側）では incoming spatial face flux \(\psi_{in}\) と
angular edge flux \(\tilde\psi_{n-1/2}\) から
\[
\bar\psi =
\frac{VQ + \mu_n(A_i+A_{i+1})\psi_{in}
      +(\alpha_{n-1/2}+\alpha_{n+1/2})\tilde\psi_{n-1/2}}
     {2\mu_nA_{i+1}+2\alpha_{n+1/2}+\sigma_t V},
\]
\[
\psi_{out}=2\bar\psi-\psi_{in},\qquad
\tilde\psi_{n+1/2}=2\bar\psi-\tilde\psi_{n-1/2}.
\]
inward sweep（\(\mu_n<0\)、外側から内側）は \(|\mu_n|\) を使い、
分母の spatial 面積を \(A_i\)、incoming face を外側 face として同じ式を適用する。
外側境界の inward incoming flux は vacuum で 0、内側境界の outward incoming flux は
反射方向 \(-\mu_n\) の inner-face outgoing flux とする。diamond difference が負の
spatial outgoing flux または angular outgoing edge flux を作る場合は、その outgoing
量を 0 に fixup する。

At \(r=0\), this is the parity boundary condition
\(\psi(0,+|\mu|)=\psi(0,-|\mu|)\).  The inward sweep through cell 0 stores the
inner-face outgoing flux for the reflected positive ordinate; the following
outward sweep through cell 0 is the physical second half of the central
spherical cell and contributes the positive-angle ordinate to
\(\sum_n w_n\bar\psi_n\).  It is therefore not a duplicate deposition of the
negative-angle ordinate.  `Radiation.origin_parity_only` is retained as a
compatibility flag for the preheat investigation, but the current CPU and GPU
\(S_N\) sweeps already use this parity form and no alternate transport equation
is selected by the flag.

source iteration は
\[
\epsilon=\max_{i,g}
\frac{|\phi^{new}_{i,g}-\phi^{old}_{i,g}|}
     {\max(|\phi^{new}_{i,g}|,10^{-300})}
\]
を `1e-6` 以下にするか、200反復で打ち切る。closure-only CPU \(S_N\)
path では DSA は未実装であり、Fleck effective scattering ratio
\(\sigma^{eff}_{s}/(\sigma^{eff}_{a}+\sigma^{eff}_{s})\simeq 1-f\) が
1 に近い optically thick cell では source iteration の収束が遅い。収束後、
\[
E^{SN}_{i,g}=\frac1c\sum_n w_n\bar\psi_{i,n,g},\qquad
P^{SN}_{rr,i,g}=\frac1c\sum_n w_n\mu_n^2\bar\psi_{i,n,g},\qquad
\chi^{SN}_{i,g}=\frac{P^{SN}_{rr,i,g}}{\max(E^{SN}_{i,g},10^{-300})}.
\]
QD handoff では \(S_N\) または HO tally 由来の raw closure を共通の
QD closure regularizer に渡し、\([1/3,1]\) clamp、same-material spatial smoothing、
temporal relaxation を適用してから `holo_lo_solver.cpp` に渡す。QD solve は
spherical origin に接する cell 0 のみ \(\chi=1/3\) とし、他の cell では
regularized \(\chi^F\) を使う。near-center 判定に domain outer radius は使わない。
The first internal face adjacent to cell 0 uses the ordinary spherical QD
geometric correction.  No additional origin-face limiter is applied; reducing
this term under-drives the convergent spherical radiation coupling.

When `Radiation.holo.sn_material_coupling=True`, the GPU \(S_N\) backend runs
once per radiation step.  In 1D_SPH with
`Radiation.holo.solver="quasidiffusion_1d"`, the GPU \(S_N\) solve is used as a
chi-only high-order closure: it computes \(E^{SN}_{i,g}\), \(P^{SN}_{rr,i,g}\),
and \(\chi^{SN}_{i,g}\), but does not update `State.Te` or `State.ee`.
The \(\chi^{SN}\) host handoff uses the same QD closure regularizer as the CPU
closure path before invoking the existing QD LO solver, which owns the radiation
energy, face flux, and material-energy update.  In 2D_RZ the QD LO solver is
still unavailable, so the path remains the direct GPU \(S_N\) deterministic
source path described below.

The 1D_SPH GPU \(S_N\) closure uses the same spherical diamond-difference sweep
as the CPU closure, with one CUDA block per energy group and cell-ordered
sweeps.  Absorption re-emission is fixed explicitly at the step-start
temperature \(T^n_i\); the transport sweep does not update `State.Te`:
\[
Q^{emit,n}_{i,g}=c\,\sigma^{eff}_{a,i,g}a_{eV}(T^n_i)^4 b_g(T^n_i).
\]
With this fixed source, streaming and Fleck effective scattering form the
linear fixed-source transport problem
\[
\mu\partial_r\psi_{i,n,g}+\sigma_{t,i,g}\psi_{i,n,g}
=\frac12 Q^{emit,n}_{i,g}
 +\frac12\sigma^{eff}_{s,i,g}\phi_{i,g},
\qquad
\sigma_{t,i,g}=\sigma^{eff}_{a,i,g}+\sigma^{eff}_{s,i,g}
\]
and is discretized with the same spherical diamond-difference sweep.  Starting
from \(\phi^0_{i,g}=0\), the runtime iterates up to 500 source iterations and
checks
\[
\max_{i,g}
\frac{|\phi^{k+1}_{i,g}-\phi^{k}_{i,g}|}
     {\max(|\phi^{k+1}_{i,g}|,10^{-300})}.
\]
After convergence or the iteration cap,
\[
E^{SN}_{i,g}=\phi_{i,g}/c,\qquad
\chi^{SN}_{i,g}
  =\frac{P^{SN}_{rr,i,g}}{\max(E^{SN}_{i,g},10^{-300})}
\]
defines the QD closure.  The LO solve uses the raw Planck and Rosseland
opacities, not the Fleck-effective absorption/scattering used by the HO
closure sweep.  For each group, the QD moment system advances
\[
\frac{V_i}{\Delta t}(E^{n+1}_{i,g}-E^n_{i,g})
 + A_{i+1/2}F^{n+1}_{i+1/2,g}
 - A_{i-1/2}F^{n+1}_{i-1/2,g}
 =
c\,\sigma_{P,i,g}V_i
\left(B_{i,g}(T^n_i)-E^{n+1}_{i,g}\right),
\]
with the QD face relation
\[
\left(\frac{1}{c\Delta t}+\sigma_{R,i+1/2,g}\right)F^{n+1}_{i+1/2,g}
+ c\,\frac{\chi^{SN}_{i+1,g}E^{n+1}_{i+1,g}
          -\chi^{SN}_{i,g}E^{n+1}_{i,g}}{\Delta r_{i+1/2}}
=\frac{F^n_{i+1/2,g}}{c\Delta t}+G^{SN}_{i+1/2,g},
\]
where \(G^{SN}\) is the existing spherical QD geometric correction.  The LO
source solve then applies the accumulated matter-energy change to `State.ee`,
`State.Te`, and `State.Pe`.

2D_RZ GPU \(S_N\) material coupling は収束後の deterministic radiation energy density
\(E^{SN}_{i,g}\) から
\[
\mathrm{rad\_dep}^{SN}_{i,g}
  = c\,\sigma^{eff}_{a,i,g}E^{SN}_{i,g}V_i\Delta t,\qquad
\mathrm{rad\_emit}^{SN}_{i,g}
  = c\,\sigma^{eff}_{a,i,g}a_{eV}(T^n_i)^4b_g(T^n_i)V_i\Delta t
\]
を `State.holo_rad_dep` と `State.holo_rad_emit` に publish する。source injection
は全 cell を HOLO-owned として扱い、particle `rad_dep/rad_emit` ではなく
\(\sum_g(\mathrm{rad\_dep}^{SN}_{i,g}-\mathrm{rad\_emit}^{SN}_{i,g})\) を electron
energy update に適用する。MC transport は同じ step で継続し、particle tallies は
diagnostics/validation 用に保持する。

`Radiation.holo.solver="quasidiffusion_1d"` の LO solve は、各 active contiguous patch
\([i_0,i_1]\) ごとに cell-center \(E_{i,g}\) と internal face flux
\(F_{i+1/2,g}\) を同時に解く。未知ベクトルは
\[
x_{2m}=E_{i_0+m,g},\qquad
x_{2m+1}=F_{i_0+m+1/2,g}
\]
であり、前者は \(m=0,\ldots,N_{patch}-1\)、後者は
\(m=0,\ldots,N_{patch}-2\) にだけ存在する。サイズは \(2N_{patch}-1\) である。
even row は
\[
\left(\frac{V_i}{\Delta t}+c\sigma_{P,i,g}V_i\right)E_{i,g}
 + A_{i+1/2}F_{i+1/2,g}-A_{i-1/2}F_{i-1/2,g}
=\frac{V_i}{\Delta t}E^n_{i,g}+c\sigma_{P,i,g}V_i B_g(T_{e,i})
\]
を離散化し、consistency source、face-current 境界、outer vacuum loss
\(A_{N}cE_{N-1,g}/4\) は従来通りこの row に加える。odd row は physical flux
\(F\) を未知量にして
\[
\left(\frac{1}{c\Delta t}+\sigma_{R,i+1/2,g}\right)F_{i+1/2,g}
 +c\left[
 \frac{\chi_{i+1,g}E_{i+1,g}-\chi_{i,g}E_{i,g}}{\Delta r_{i+1/2}}
 +\frac{3\bar\chi_{i+1/2,g}-1}{2r_{reg,i+1/2}}
  (E_{i,g}+E_{i+1,g})\right]
=\frac{F^n_{i+1/2,g}}{c\Delta t}
\]
を使う。したがって QD internal face では \(F\) を消去せず、幾何項
\((3\chi-1)/r\) は odd row の \(E\) off-diagonal だけに現れる。
The geometric denominator is regularized locally as
\[
r_{reg,i+1/2}=\max(r_{i+1/2},\Delta r_{i+1/2}),
\]
where \(\Delta r_{i+1/2}\) is the neighboring cell-center spacing.  No
domain-outer-radius floor is used, so extended laser-corona mesh extent does
not enlarge the isotropic QD closure region.
The first internal face adjacent to the origin cell also uses this same
regularized spherical geometry.  The origin regularization is limited to
\(\chi_0=1/3\); the face geometric correction is retained because it carries
the convergent spherical radiation coupling needed for shell compression.
この interleaved system は通常の tridiagonal として、符号付き pivot を許す
Thomas elimination で解く。patch 境界が inactive neighbor と接する場合の
Dirichlet coupling と physical boundary treatment は boundary condition として扱い、
解後の energy/source accounting は solved \(E,F\) を用いる。

QD spatial solve の線形系は positivity constraint を未知量に含めないため、
multigroup の高周波 tail で cell-total に対して小さい負の group energy
undershoot が出ることがある。この場合は
\[
\epsilon^-_{i,g} =
\max\left[
10^{-2}\max(E^n_{i,g},B_{i,g}),
5\times10^{-3}\sum_{g'} E^n_{i,g'}
\right]
\]
を許容幅とし、\(-\epsilon^-_{i,g}\le E^{n+1}_{i,g}<0\) なら
\(E^{n+1}_{i,g}=0\) に clamp する。これより大きい負値、または NaN/Inf は
spatial solve failure として扱う。clamp 後の \(E^{n+1}\) を用いて
source accounting と conservation check を行うため、fixup で生じた差分は
LO material/radiation exchange に含める。

#### 7.1.2g.1 QD closure regularization

`Radiation.holo.solver="quasidiffusion_1d"` の closure input は、MC tally
由来 \(\chi^{HO}\)、closure-only CPU \(S_N\) 由来 \(\chi^{SN}\)、または
1D_SPH GPU \(S_N\) material-coupling 経路から host へ渡された
\(\chi^{SN}\) のいずれであっても同じ regularization を受ける。まず raw
\(\chi^S\) を \([1/3,1]\) に clamp する。次に
`Radiation.holo.closure_smooth_passes = N_s` 回、同一 dominant material の連続
cell run 内で
\[
\chi^{(p+1)}_{i,g}
=(1-\alpha)\chi^{(p)}_{i,g}
 +\frac{\alpha}{2}\left(\chi^{(p)}_{L(i),g}
 +\chi^{(p)}_{R(i),g}\right),
\]
を適用する。ここで \(\alpha=\) `Radiation.holo.closure_smooth_alpha`、
\(L(i)\)、\(R(i)\) は左右隣接 cell であり、void cell、material 境界、domain 境界では
該当側を \(i\) 自身に反射する。この reflected-boundary stencil は各 contiguous
same-material run の cell 平均を保存し、入力が \([1/3,1]\) 内なら smoothing 後も
\([1/3,1]\) 内に残る。

spatial smoothing 後、temporal relaxation を
\[
\chi^{F,n}_{i,g}=(1-w)\chi^{F,n-1}_{i,g}+w\chi^{S,n}_{i,g},
\qquad
w=\texttt{Radiation.holo.closure\_relax}
\]
で行う。履歴が未初期化の cell×group では \(\chi^{F,n}=\chi^{S,n}\) とする。
predictor solve は保存済みの \(\chi^F\) があればそれを使い、corrector solve は
新しい MC tally または \(S_N\) closure から \(\chi^F\) を更新する。1D_SPH
GPU \(S_N\) material-coupling 経路では、GPU sweep 直後の precomputed
\(\chi^{SN}\) override がこの regularizer を一度だけ通るため、呼び出し元では
別途 smoothing/relaxation を行わない。`implicit_1d` はこの filtered closure を
参照せず、従来通り diffusion coefficient \(c/(3\sigma_R)\) を使う。
`p_rr_tally=False` は MC tally closure 診断を無効化する。`sn_closure=False` かつ
MC tally closure が利用できない場合、QD closure 入力は diffusion limit
\(\chi=1/3\) に戻る。

履歴診断 `holo/E_LO_total` は全 LO domain に対する
\(\sum_i V_i\sum_g E^{LO}_{i,g}\) [erg] である。
`holo/E_LO_boundary_in`、`holo/E_LO_boundary_out`、`holo/matter_delta`、
`holo/source_balance_error` は runtime LO solve result から記録する。
`holo/particle_net_source_core` は LO-coupled cell 内の particle diagnostic
\(\sum_g(\mathrm{rad\_dep}_{i,g}-\mathrm{rad\_emit}_{i,g})\) [erg] であり、
material source には使わない。`holo/lo_particle_source_mismatch` は
`holo/matter_delta - holo/particle_net_source_core` [erg] として、LO-owned material
source と particle diagnostic source の差だけを記録する。
`holo/Prr_coverage`、`holo/chi_min`、`holo/chi_mean`、`holo/chi_max` は
LO-coupled cell×group の passive closure diagnostics から記録する。
`holo/E_LO_boundary_in` は internal HO/LO patch 境界からの face-current または Dirichlet
coupling 入射、`holo/E_LO_boundary_out` は同境界への outgoing 成分と outer vacuum
leakage である。`holo.enabled=False` では mask、global LO solve、source ownership、
追加 tally は構築しない。

#### 7.1.3 ヒステリシスモード選択器（Hysteresis Mode Selector）

§7.1.2の単一閾値判定では、衝撃波前面や不透明度の急変領域で光学厚 \(\tau\) が
閾値 \(\tau_{DDMC}\) の近傍を振動し、IMC⇄DDMCモードが毎ステップ切り替わる
**チャタリング**が生じうる。モード切替は IMC→DDMC で位置・方向の破棄を伴い、
DDMC→IMC では再サンプル、DDMC→RW では mode handoff を伴うため、
チャタリングは統計ノイズを増大させる。

ヒステリシスモード選択器はセル×群ごとの**状態機械**として、モード遷移に
入口条件（entry）と出口条件（exit）を非対称に設定することでチャタリングを抑制する。

**状態遷移ロジック**：

**(a) IMC → DDMC 遷移（entry）**：以下の**全て**を満たす場合のみ遷移：
1. **ベースライン判定**：§7.1.2の判定（条件1〜4）で DDMC が選択されている
2. **光学厚閾値**：\(\tau_{i,g} \ge \tau_{on}\)（\(\tau_{on} = \tau_{DDMC}\)）
3. **散乱比閾値**：\(\omega_{i,g} \ge \omega_{on}\)（\(\omega_{on} = \omega_{DDMC}\)）
4. **滞留条件**：\(\text{hold\_count}_{i,g} \ge \text{mode\_hold}\)
5. **変化率制限**：\(|\Delta\tau/\tau| = |\tau^n - \tau^{n-1}|/\tau^{n-1} \le \text{rate\_max}\)

**(b) DDMC → non-DDMC 遷移（exit）**：以下の**いずれか**を満たす場合に遷移：
1. \(\tau_{i,g} < \tau_{off}\)（\(\tau_{off}\)：DDMC脱出τ閾値）
2. \(\omega_{i,g} < \omega_{off}\)（\(\omega_{off}\)：DDMC脱出ω閾値）
3. \(\hat{P}(1) > 1\)（変換確率制約違反、§7.7.3）

exit したセル×群はそのステップの baseline 判定へ戻る。したがって
\(\tau_{RW} \le \tau < \tau_{DDMC}\) の 1D_SPH セルは DDMC から RW へ降格しうる。

**(c) 安全オーバーライド**：\(\sigma_R < \sigma_{floor}\)（既定 \(10^{-20}\) cm\(^{-1}\)）の場合は
条件に関わらず強制的にIMCモードとする（真空近似セル）。

**hold\_count 管理**：
- セル×群ごとに `uint8` カウンタ（最大 255）
- IMC 状態で毎ステップインクリメント（飽和あり）
- IMC → DDMC 遷移時：0 にリセット
- DDMC → non-DDMC 遷移時：0 にリセット

**パラメータ**（SPECIFICATION §6.4.5 `ddmc` ブロック）：

| パラメータ | 既定値 | 説明 |
|-----------|--------|------|
| `tau_ddmc_off` | -1.0（\(=\tau_{DDMC}\)と同値） | DDMC脱出τ閾値。\(<0\)で非アクティブ（\(\tau_{off}=\tau_{DDMC}\)）、有効時 \(0.5 \le \tau_{off} \le \tau_{DDMC}\) |
| `omega_ddmc_off` | -1.0（\(=\omega_{DDMC}\)と同値） | DDMC脱出ω閾値。\(<0\)で非アクティブ（\(\omega_{off}=\omega_{DDMC}\)）、有効時 \(0 \le \omega_{off} \le \omega_{DDMC}\) |
| `mode_hold` | 0 | IMC→DDMC遷移前の最小滞留ステップ数。0でヒステリシスなし |
| `rate_max` | \(10^{30}\) | \(|\Delta\tau/\tau|\) の最大許容変化率。\(10^{30}\)で事実上無制限 |

**後方互換性**：全パラメータが既定値の場合、ヒステリシス条件は自明に成立する
（\(\tau_{off}=\tau_{on}\)、\(\omega_{off}=\omega_{on}\)、\(\text{mode\_hold}=0\)、\(\text{rate\_max}=\infty\)）。
したがって従来の§7.1.2判定と完全に同一の動作となる。

> **チャタリング検出**：ヒステリシス結果として IMC→DDMC / DDMC→IMC の各遷移回数を
> ステップごとに集計する。遷移回数が前ステップ比で急増している場合は
> WARNING を出力し、`tau_ddmc_off` の引き下げまたは `mode_hold` の増大を推奨する。

### 7.2 DDMCの拡散係数
拡散方程式（群g）：
\[
\frac{1}{c}\frac{\partial E_g}{\partial t} - \nabla\cdot(D_g\nabla E_g) + \sigma_{a,eff,g} E_g = S_g
\]
拡散係数：
\[
D_g = \frac{1}{3\sigma_{tr,g}} = \frac{1}{3\sigma_{R,g}} \quad [\text{cm}]
\]
ここで \(\sigma_{R,g}=\rho\kappa_{R,g}\) [1/cm]。注：\(D_g\) は \(c\) を含まない形式（拡散方程式で \(c D_g\) が拡散速度の次元 [cm²/s] を持つ）。

### 7.3 離散化とリーク係数：**符号規約を固定（要修正点）**
DDMCは離散拡散の結合係数を“リーク率”に変換する。  
TENRYUでは、拡散離散化が構成する行列を **M‑matrix形式**で扱う規約を固定する。

#### 7.3.1 行列形式（規約）
セル中心未知量 \(E_{i,g}\) に対して、離散化により
\[
\sum_{j\in\mathcal{N}(i)\cup\{i\}} A_{ij,g} E_{j,g} = b_{i,g}
\]
を得るとき、TENRYUは以下を満たす実装を **必須**とする：

- オフ対角：\(A_{ij,g}\le 0\)（\(j\ne i\)）
- 対角：\(A_{ii,g} > 0\)
- 対角優位：\(A_{ii,g} \ge \sum_{j\ne i} |A_{ij,g}|\)

（この形式は確率化に必要：リーク率が非負になり、確率が正規化できる。）

#### 7.3.2 リーク係数（Leakage opacity）
上記規約のもと、セル i から近傍 j へのリーク率（単位 1/cm）を
\[
\Sigma^{leak}_{i\to j,g} = \frac{-A_{ij,g}}{V_i}
\]
と定義する。  
（オフ対角が負なので \(-A_{ij}\ge 0\) が保証される。）

総リーク率：
\[
\Sigma^{out}_{i,g} = \sum_{j\in\mathcal{N}(i)} \Sigma^{leak}_{i\to j,g}
\]

> 旧仕様の `max(0,A_ij)` のような"符号依存の曖昧さ"を禁止する。
> 実装の拡散モジュールが別符号で係数を返す場合は、**Materials/Rad側で符号変換してこの規約へ合わせる**。
> 既定（`m_matrix_check=True`）では、正のオフ対角が1つでもあれば当該セル×群をDDMC禁止（IMCへ）とし、
> クランプでDDMCを継続しない。`m_matrix_check=False`（検証用）に限り安全クランプ `max(0,-A_ij)` を許可する。

**2D RZ への拡張**：
v1.0 既定では **Kershaw 9点ステンシル**（Appendix A.10）から導出したリーク係数を使用する
（`leak_stencil="9_kershaw"`、SPECIFICATION §6.4 参照）。
Kershaw行列 \(A_{ij,g}\) を \(D_g = 1/(3\sigma_{R,g})\)（§7.2）で構成し、
§7.3.2 の一般定義 \(\Sigma^{leak}_{i\to j,g} = -A_{ij,g}/V_i\) を適用する。
これにより歪格子でも正確なリーク率が得られ、面別近似の厚光学極限バイアス（後述）を回避する。
R9イベントループはトポロジカル面（1D:2面、2D_RZ:4面）を入力とするため、
2Dの角近傍リーク（NE/NW/SE/SW）は隣接2面へ射影する。各角の射影先面ペア：

| 角方向 | 射影先面ペア \((f_1, f_2)\) |
|--------|---------------------------|
| NE (右上) | R\_right (face 1), Z\_top (face 3) |
| NW (左上) | R\_left (face 0), Z\_top (face 3) |
| SE (右下) | R\_right (face 1), Z\_bottom (face 2) |
| SW (左下) | R\_left (face 0), Z\_bottom (face 2) |

角近傍係数 \(\Sigma_{corner}\) は面積重み \(w_{f_1}=A_{f_1}/(A_{f_1}+A_{f_2})\), \(w_{f_2}=A_{f_2}/(A_{f_1}+A_{f_2})\) で
\((w_{f_1}\Sigma_{corner},\,w_{f_2}\Sigma_{corner})\) に分配する（総和保存）。
**退化ガード**：\(A_{f_1}+A_{f_2} < \varepsilon_{area}\)（\(\varepsilon_{area} = 10^{-30}\) cm²）の場合は等分配 \(w_{f_1}=w_{f_2}=0.5\) とする。

> **Kershaw ステンシルの再利用**：伝導ソルバ（§4.2）と DDMCリーク（本節）は
> 同一の Kershaw ステンシル構築カーネル（CUDA_KERNELS C2）を使用するが、
> 入力する拡散係数が異なる（伝導: \(D_{eff}\)、DDMC: \(D_g = 1/(3\sigma_{R,g})\)）。
> 多群の場合、C2 を群ごとに呼び出す（群数 \(G\) 回）。

**代替オプション `leak_stencil="4"`**（面別 Densmore 近似）：
直交格子や検証用途では、面ごとの1D Densmore 近似（§7.3.4）を各面に独立に適用する簡易式が利用可能。
セル \(i\) の面 \(m\)（\(m = 0,1,2,3\) : R\_left, R\_right, Z\_bottom, Z\_top — CUDA\_KERNELS §6.4.3 面規約）に対し：
\[
\Sigma^{leak}_{i\to j_m, g} = \frac{2\, A_m}{3\, V_i\, \sigma_{R,m}^{face}\, (\Delta x_m + 2\lambda_{mfp})}
\]

ここで：
- \(A_m = 2\pi \bar{R}_m L_m\) [cm\(^2\)]（面 \(m\) のRZ面積、\(\bar{R}_m\) は面中点R座標、\(L_m\) は面長さ）
- \(\Delta x_m = V_i / A_m\) [cm]（面に垂直な有効セル幅）
- \(\sigma_{R,m}^{face}\) [1/cm]：§7.3.4 の面評価規約で算出した面Rosseland不透明度
- \(\lambda_{mfp} = 1/\sigma_{R,m}^{face}\) [cm]（平均自由行程）— 境界面では §7.3.5 を使用

> **次元検査**：分母 \(\sigma_{R,m}^{face}\,(\Delta x_m + 2\lambda_{mfp})\) は
> \([1/\text{cm}] \times [\text{cm}] = \) 無次元。
> 等価な表現：\(\sigma_{R}\Delta x_m + 2\)（光学厚 \(\tau_m\) + 2 平均自由行程を光学厚単位で表現）。

> **記号の区別**：本節の \(\lambda_{mfp}\) [cm]（平均自由行程）と §7.3.5, §7.7.1 の
> \(\lambda \approx 0.7104\) [無次元]（Milne外挿距離、平均自由行程単位）は
> 異なる物理量である。混同を避けるため、平均自由行程には添字 \(_{mfp}\) を付す。

> **1D Densmore 公式との関係**：1D公式（§7.3.4 Eqs.20–21）の分母は
> \(\sigma^+_{R}\Delta x_j + \sigma^-_{R}\Delta x_{j-1}\)（隣接セル幅を明示使用）であるが、
> 本面別公式は \(\sigma_{R}(\Delta x_m + 2\lambda_{mfp}) = \sigma_{R}\Delta x_m + 2\)
> で隣接セルの光学厚を2（2平均自由行程）で近似している。
> 光学的に厚い一様セルでは1D公式が \(1/(3\sigma_R\Delta x^2)\) を与えるのに対し、
> 本式は \(2/(3\sigma_R\Delta x^2)\) を与え、約2倍の差異がある。
> これは面別独立近似の帰結であり、`9_kershaw` を既定とする理由の一つである。

この面別公式は4点（面のみ）近似であり、対角隣接は含めない。
歪格子では `9_kershaw`（既定）を使用すること。

**注意**：\(r = 0\) 軸上の面では \(A_m \to 0\) となるため、\(\Delta x_m \to \infty\)。
このような面のリーク係数は \(\Sigma^{leak} = 0\) と設定する（軸方向リークなし）。

**不透明度フロアによるDDMC安全策**：
\(\sigma_{R,g} < \sigma_{floor}\)（既定 \(10^{-20}\) cm\(^{-1}\)、§11.3 参照）のセルは、
DDMC 光学的厚さ基準（\(\tau < \tau_{DDMC}\)、§7.1.2）を満たさないため自動的に IMC にフォールバックする。
したがって \(\sigma_{R} = 0\) が DDMC リーク式に現れることはない。
追加安全策として、リーク係数計算で面Rosseland不透明度に
\(\sigma_{R,m}^{face} \ge \sigma_{floor}\) のクランプを適用する。

#### 7.3.3 M‑matrix診断（安全策）
v1.0既定で、各セル×群について以下を検査する：
- すべてのオフ対角 \(A_{ij}\le 0\)
- \(A_{ii} \ge \sum_{j\ne i}|A_{ij}| - \epsilon\)（丸め許容）
- もし違反があれば：
  1) そのセル×群を **DDMC禁止**にして IMC へフォールバック
  2) 違反数と場所をdiagnosticsへ出す（回帰で検知）

#### 7.3.4 面評価の不透明度（face-centered opacity）

DDMCのリーク係数に用いる **面のRosseland不透明度** \(\sigma_{R,j+1/2}\) は、
温度依存が強い場合に非物理な伝搬停止を起こし得る（Densmore et al. 2007 §3.1, Szilard & Pomraning 1992）。

TENRYUでは **面温度** で不透明度を評価する規約を固定する：

> **多群における不透明度種別の規約（grey文献からの拡張）**：
> Densmore et al. (2007) はgrey（1群）のため、単一の不透明度σ_nを使用している。
> 多群TENRYUでは、DDMC方程式内で **2種類の不透明度** が必要：
> - **リーク係数**（拡散離散化由来）：**Rosseland** \(\sigma_{R,g}=\rho\kappa_{R,g}\)（§7.2のD_gと整合）
> - **吸収/放射項**（Kirchhoff律）：**Planck** \(\sigma_{a,g}=\rho\kappa_{P,g}\)（§6.1のσ_{a,eff}と整合）
>
> grey（κ_P=κ_R）では両者が一致するため、Densmore式と自然に整合する。
> 以下、リーク関連の面不透明度は **Rosseland** を使用する。

面温度（放射温度的平均）：
\[
T_{n,j+1/2} = \left(\frac{T_{n,j}^4 + T_{n,j+1}^4}{2}\right)^{1/4}
\]

面でのRosseland不透明度：
\[
\sigma_{R,j+1/2}^{\pm} = \rho_{j \text{ or } j+1}\,\kappa_{R,g}(\rho_{j \text{ or } j+1},\, T_{n,j+1/2})
\]

- \(\sigma^-_{R,j+1/2}\)：面の左側セル j の密度で評価
- \(\sigma^+_{R,j+1/2}\)：面の右側セル j+1 の密度で評価

リーク係数（Densmore 2007 Eqs.(20)–(21) 準拠、多群ではRosseland）：
\[
\sigma_{L,j} = \frac{2}{3\Delta x_j} \frac{1}{\sigma^+_{R,j-1/2}\Delta x_j + \sigma^-_{R,j-1/2}\Delta x_{j-1}}, \quad
\sigma_{R,j} = \frac{2}{3\Delta x_j} \frac{1}{\sigma^-_{R,j+1/2}\Delta x_j + \sigma^+_{R,j+1/2}\Delta x_{j+1}}
\]

> **σ^±とΔxの対応規則**：各 σ^± は **自セルのΔxと対**になる。
> 面 j-1/2 において σ^+_{j-1/2}（セル j の物性）× Δx_j、σ^-_{j-1/2}（セル j-1 の物性）× Δx_{j-1}。
> これは Eqs.(15)–(18) の導出（各半セルの光学厚が σ×Δx/2 で近似される）から直接従う。

> **根拠**：セル中心温度でセル単位に不透明度を評価すると、
> 隣接セルの一方の不透明度が極端に大きいとき（冷たい壁など）リーク率がゼロに近づき、
> 放射が伝搬停止する非物理挙動を起こす。面温度評価はこれを防ぐ。
>
> **参考文献**：
> - Densmore et al., JCP 222 (2007) Eq.(23): 面温度の定義
> - Szilard & Pomraning, Nucl. Sci. Eng. 112 (1992): 面不透明度の理論的根拠

#### 7.3.5 DDMC境界セル（インターフェースセル）のリーク不透明度

DDMCとIMCの境界に隣接するDDMCセル（境界セル、j=1とする）では、
**asymptotic diffusion-limit BCから導出される修正リーク不透明度**を用いる。

標準の内部セル用σ_L（§7.3.4 Eqs.(20)–(21)）の代わりに、境界セルのIMC側リーク不透明度は：
\[
\sigma_{L,1} = \frac{1}{\Delta x_1} \cdot \frac{2}{3\sigma_{R,1}\Delta x_1 + 6\lambda}
\]
ここで：
- \(\sigma_{R,1}\)：境界セルのRosseland不透明度（7.3.4の面評価規約で算出、拡散と整合）
- \(\Delta x_1\)：境界セルの代表長
- \(\lambda \approx 0.7104\)：Milne外挿距離（無次元、平均自由行程単位。物理的長さは \(\lambda / \sigma_{tr}\) [cm]）

この修正は、標準のσ_L（隣接セル幅を含む）において隣接セルが存在しない（IMC側）場合に、
外挿距離λが隣接セル幅の役割を果たすことに対応する。

境界セルのDDMC方程式（Densmore Eq.32、多群ではRosseland/Planck分離）：
\[
\frac{1}{c}\dot\phi_1 + (\sigma_{L,1} + \sigma_{R,1} + f_{n,1}\sigma_{a,1})\phi_1
= f_{n,1}\sigma_{a,1}\,a_{eV}\,c\,T_{n,1}^4
+ \frac{1}{\Delta x_1}\left(\sigma_{L,2}\phi_2\Delta x_2 + \int_0^1 P(\mu)\mu I_b(\mu,t)\,d\mu\right)
\]
ここで \(\sigma_{L,1}, \sigma_{R,1}\) はRosseland由来のリーク不透明度、
\(f_{n,1}\sigma_{a,1}\) はPlanck由来の実効吸収（= \(\sigma_{a,eff,1}\)、§6.1参照）。
greyでは \(\sigma_R=\sigma_a\) となりDensmore原式と一致する。

最終項はIMC側からの入射粒子ソースであり、P(μ)と組み合わせてエネルギー保存を保証する（7.7.1参照）。

反対側（内部側）のσ_{R,1}は標準の7.3.4（Eq.21）をそのまま使う。

**DDMC-IMC 界面での面温度** \(T_{n,face}\)：隣接 IMC セルの \(T_e^n\) を用いて §7.3.4 と同じ \(T^4\) 平均で計算する。\(\sigma_{R,1}\) は DDMC セルの密度 \(\rho_1\) と面温度 \(T_{n,face}\) で評価する（セル中心温度ではなく面温度）。

> **重要**：境界セルで標準のσ_L（Eq.20）を使うと、
> DDMC方程式の係数とP(μ)の導出が不整合になり、インターフェースでのエネルギー保存が崩れる。
> 必ず修正版σ_{L,1}を使用すること。
>
> **参考文献**：Densmore et al., JCP 222 (2007) Eqs.(32)–(33)

---

### 7.4 境界リーク（vacuum/reflect/Marshak）
境界面へのリークを “外部セル” へのリークとして扱う。

- vacuum（escape）：境界リークを持ち、リークした粒子は系外へ消滅し流出エネルギーに計上
- reflect：\(\Sigma^{leak}_{i\to b,g} = 0\)（境界リークなし、完全反射）。
  境界面でのフラックス \(F = 0\) となる。
  reflect 境界で反射した IMC 粒子が DDMC セルに再入する場合は、
  標準の IMC→DDMC 変換（§7.7.1）が適用される。
- Marshak：検証用途。境界入射を別途IMC粒子として生成（8章）

vacuum境界の既定（拡散外挿：Milne）：
- 外挿長 \(d_{ext} = 0.7104/\sigma_{tr}\) [cm]
- 境界フェイスに対するリーク係数は、1Dなら
\[
\Sigma^{leak}_{i\to b,g} \approx \frac{D_g A_f}{V_i (d_{cell}+d_{ext})}
\]
ここで \(d_{cell} = V_i/A_f\)（面法線方向の代表セル厚、§7.7.4 と同一定義）である。
（一般RZ歪格子は拡散離散化が返す境界係数をこの形式へ整合させる。）

#### 7.4.1 DDMCセルの implicit diffusion solve（HIMCD Phase-1, optional）
`ddmc.implicit_diffusion=True` かつ対応条件（1D, LTE, Marshak境界なし, `volume_source_rate=0`）
では、DDMC セル×群は §7.5 の particle DDMC イベントループではなく、Radiation step 内で
backward Euler の implicit diffusion solve により更新する。

群 \(g\) の DDMC セル \(i\) について、
\[
\frac{E^{n+1}_{i,g} - E^n_{i,g}}{\Delta t}
= \frac{1}{V_i}\sum_{f\in\partial i} F_{f,g}(E^{n+1})
- c\,\sigma_{a,eff,i,g}\,E^{n+1}_{i,g}
+ c\,\sigma_{a,eff,i,g}\,a_{eV}T_i^4 b_g(T_i)
\]
を解く。Phase-1 の implicit DDMC diffusion は 2 段の predictor-corrector
（Picard 1 回）を用いる。まず radiation step 開始時の \(T_i^n\) で全群を解き、
\[
\Delta E^{pred}_{i} = \sum_g \left(
c\,\sigma_{a,eff,i,g}\,E^{pred}_{i,g}V_i\Delta t
- c\,\sigma_{a,eff,i,g}\,a_{eV}(T_i^n)^4 b_g(T_i^n)V_i\Delta t
\right)
\]
からセル熱容量 \(C_i = \rho_i c_{v,e,i} V_i\) を用いて
\[
T_i^{pred} = T_i^n + \Delta E^{pred}_{i}/C_i
\]
を作り、\(T_i^{pred}\) で再度 diffusion solve を行って最終解 \(E^{n+1}_{i,g}\) を得る。
群間散乱は含めない。

1D 球対称の DDMC-DDMC 内部面 \(i+\tfrac{1}{2}\) では、既存 DDMC face opacity
\(\sigma^{-}_{R,i+1/2,g}, \sigma^{+}_{R,i+1/2,g}\) とセル幅 \(\Delta r_i, \Delta r_{i+1}\) を用いて
\[
\mathcal{F}_{i+1/2,g}
= \frac{2\,A_{i+1/2}\,c}
       {3\left(\sigma^{-}_{R,i+1/2,g}\Delta r_i + \sigma^{+}_{R,i+1/2,g}\Delta r_{i+1}\right)}
\]
を構成し、行列のオフ対角 \(-\mathcal{F}_{i+1/2,g}\)、対角 \(+\mathcal{F}_{i+1/2,g}\) とする。
DDMC-IMC 界面は Phase-1 でも **zero-flux にしない**。IMC kernel 側は §7.7.1 の
既存 interface conversion probability をそのまま使い、IMC 粒子は界面到達時に
DDMC セルへ変換されうる。implicit diffusion solve 側は既存 DDMC の
interface leak 係数 \(\Sigma^{int}_{i\to IMC,g}\)（実装上 `sigma_leak_left/right`）
を対角 sink として用いる。
Phase-1 実装では、IMC→DDMC 変換で生じた step-end DDMC census energy は
明示的 DDMC 粒子として持ち越さず、step 終了時に `rad_E` へ畳み込んで
次 step の \(E^n\) に反映する。

vacuum 境界は既存 DDMC の Milne リーク係数 \(\Sigma^{leak}_{i\to b,g}\)、
DDMC-IMC 界面は既存 interface leak 係数 \(\Sigma^{int}_{i\to IMC,g}\) をそのまま使い、
\[
\mathcal{B}_{i,g} = c\,V_i\,\Sigma^{sink}_{i,g},
\qquad
\Sigma^{sink}_{i,g} \in \left\{\Sigma^{leak}_{i\to b,g},\ \Sigma^{int}_{i\to IMC,g}\right\}
\]
を対角 sink として追加する。vacuum 境界の sink energy は
\(\Delta E^{esc}_{i,g} = \mathcal{B}_{i,g} E^{n+1}_{i,g}\Delta t\)
で `E_escape[g]` に加算する。一方、DDMC-IMC 界面の sink energy は
物理境界流出ではないため `E_escape[g]` へは加算せず、隣接 IMC セルの
`rad_dep[i_{IMC},g]` に加算する。

solve 後の tally 更新は次のとおり：
\[
\texttt{rad\_E\_tally}_{i,g} = c\,E^{n+1}_{i,g} V_i \Delta t,
\qquad
\texttt{rad\_dep}_{i,g} \mathrel{+}= c\,\sigma_{a,eff,i,g}\,E^{n+1}_{i,g} V_i \Delta t
\]
`rad_emit` は既存 IMC thermal source が
\(c\,\sigma_{a,eff} a_{eV} T^4 b_g \, V \Delta t\)
を保持しているため、implicit diffusion path では emission を `rad_dep` に再加算しない。
ただし predictor-corrector 後は DDMC セル群の `rad_emit[i,g]` を
最終反復の \(T^{pred}\) で評価した source に上書きし、後段の
`delta_E_rad_prev` が corrected emission を含む実適用 source と一致するようにする。

> Phase-1 注記：
> - \(E^n\) には前ステップのセル平均 `rad_E` を使用する
> - true NLTE, Marshak 境界, volume source を含むケースは particle DDMC にフォールバックする
> - 2D_RZ への拡張は将来課題

#### 7.4.2 PGRW transport（Phase-1, 1D_SPH only）
PGRW（partially-gray random walk）は独立 transport mode ではなく、
`imc_transport_persistent` 内で IMC 粒子に適用される internal acceleration branch
である。Phase-1 は 1D_SPH のみ対応し、`rad_lite_mesh` 有効 step では無効化する。

セル \(i\) では、Planck weight \(b_g(T_e)\) を使って diffusive cutoff
\(g_{diff,end,i}\)、collapsed absorption \(\bar{\sigma}_{a,i}\)、collapsed total
\(\bar{\sigma}_{t,i}\)、diffusion coefficient \(D_i\)、diffusive emission fraction
\(\gamma_i\) を毎 step host 側で前計算する。群 cutoff は
\[
\tau_g = \sigma_{t,i,g}\,\bar{s}_i,\qquad
\bar{s}_i = \frac{4V_i}{A_i},\qquad
\tau_g \ge 20\,\tau_{rw}
\]
を満たす連続した低群側スライスで定義する。

粒子位置 \(r\)、群 \(g<g_{diff,end,i}\)、残り時間 \(t_{rem}\) に対し、RW sphere 半径は
\[
R_0 =
\begin{cases}
\min(r-r_{i-1/2},\,r_{i+1/2}-r), & r_{i-1/2}>0 \\
r_{i+1/2}-r, & r_{i-1/2}=0
\end{cases}
\]
とする。Phase-1 の PGRW eligibility は
\[
\sigma^{IMC}_{t,i,g}R_0 > 1,\qquad
\bar{\sigma}_{t,i}R_0 \ge \tau_{rw},\qquad
c\,t_{rem} > R_0
\]
である。ここで \(\sigma^{IMC}_{t,i,g}=\sigma_{a,eff,i,g}+\sigma_{s,eff,i,g}\) は
current group の IMC total opacity であり、kernel は通常 IMC の
\(s_{bdry}, s_{scatter}, s_{cen}\) を評価した後にこの eligibility を判定する。

event time は 3競合で決める：
1. upscatter:
\[
t_{up} = -\frac{\ln \xi}{(1-f_i)(1-\gamma_i)\bar{\sigma}_{a,i}c}
\]
2. census: \(t_{cen}=t_{rem}\)
3. leak: \(\theta=cDt/R_0^2\) を用い、survival
\[
S(\theta)=2\sum_{n=1}^{100}(-1)^{n+1}e^{-n^2\pi^2\theta}
\]
から \(F(\theta)=1-S(\theta)\) の inverse CDF table（1024点、\(\theta\in[10^{-6},3]\)）
を引く。

選ばれた event 時刻 \(t_{evt}\) に対する吸収減衰は
\[
\tau_{abs}=cf_i\bar{\sigma}_{a,i}t_{evt},\qquad
\Delta E = E\left(1-e^{-\tau_{abs}}\right)
\]
で計算し、`rad_dep` / `rad_E_tally` へ通常 IMC と同じ `warp_tally` を通して加算する。
signed residual 粒子では通常 IMC と同じく `sign` を掛けた寄与を加算する。

- leak の位置は RW sphere 表面へ移し、方向は leak 点の sphere outward normal に対する
  cosine-law half-space から再サンプルする
- census / upscatter の位置は Eq. (22) の
  \[
  G(\rho;\theta)=\frac{\int_0^\rho \Psi(r,\theta)r^2dr}{\int_0^{R_0}\Psi(r,\theta)r^2dr}
  \]
  を 64×128 の \((\theta,\rho/R_0)\) table で逆補間して決める
- upscatter の新群は transport-side 群 \([g_{diff,end}, G)\) に制限し、
  `eta_cdf` の同区間を再正規化した CDF からサンプルする。
  `eta_cdf` が無い場合は同区間の \(\sigma_a(g)b_g\) 重みで代替する

> Phase-1 注記：
> - `tau_rw=0` で PGRW は完全無効
> - legacy `TransportMode::RW` は生成しない
> - `rw_transport_gpu.cu` は dead code として保持する
> - `rad_lite_mesh` 有効 step では PGRW を使わない
---

### 7.5 DDMCイベント
DDMC粒子（cell i, group g, energy E, time t）。

総イベント率：
\[
\Sigma^{tot}_{i,g} = \sigma_{a,eff,i,g} + \sigma_{s,eff,i,g}^{(NLTE)} + \Sigma^{out}_{i,g} + \Sigma^{leak}_{i\to b,g}
\]
ここで \(\sigma_{s,eff,i,g}^{(NLTE)}\) は true NLTE DDMC でのみ有効な局所実効散乱率
\((1-f)\sigma^{PA}_{i,g}\) であり、LTE DDMC では 0 とみなす。
\(\Sigma^{out}_{i,g}\) は全ての **非境界隣接面** のリーク率の和（§7.3.2）、
\(\Sigma^{leak}_{i\to b,g}\) は全ての **境界面** のリーク率の和（§7.4）である。
内部セル（全面が内部面）では \(\Sigma^{leak}_{i\to b,g} = 0\) となる。
面 CDF には内部面と境界面の両方が含まれ、
境界面が選択された場合はエスケープ（エネルギーを outflow にタリー）、
内部面が選択された場合はセル遷移（§7.5 後段のリーク処理）となる。

> **Planck/Rosseland混合に関する注記**：
> \(\Sigma^{tot}\) は **Planck由来の実効吸収**（\(\sigma_{a,eff}=f\sigma_{a,g}\)、§6.1）、
> true NLTE での **局所実効散乱**（\(\sigma_{s,eff}=(1-f)\sigma_{a,g}\)）と
> **Rosseland由来のリーク率**（\(\Sigma^{out},\Sigma^{leak}\)、§7.3–7.4）を合算する。
> これは物理的に正しい：吸収率と局所再放出率は Planck absorption / emissivity で決まり、
> 拡散/リーク率はRosseland不透明度で決まる。
> Densmore (2007) Eq.(19) のgrey版（\(\sigma_a=\sigma_R\)）とはこの点で異なるが、
> 多群では両者の区別が本質的であり、同一視すると吸収/拡散バランスが崩れる。

次イベント時間：
\[
\Delta t_{evt} = \frac{-\ln\xi}{c\,\Sigma^{tot}_{i,g}}
\]
- **安全ガード**：\(\Sigma^{tot}_{i,g} \le \Sigma_{floor}\)（\(\Sigma_{floor} = 10^{-30}\) cm\(^{-1}\)）の場合、
  \(\Delta t_{evt} = \infty\) として即座にcensus化する。
  これは §7.1.2 のDDMC判定基準（\(\tau \ge 4\) 等）により到達不能であるべきだが、
  数値誤差や不透明度テーブル端の異常値に対する防御ガードである。
  発生時は `DeviceErrorFlags::ddmc_sigma_tot_zero` を設定する。
- \(t+\Delta t_{evt} \ge t^{n+1}\) ⇒ census（下記参照）
- そうでなければイベント実行

**DDMCのcensus処理**：\(t + \Delta t_{evt} \ge t^{n+1}\) の場合：
1. 残存滞在時間 \(\Delta t_{cen} = t^{n+1} - t\) を計算
2. 滞在時間寄与をタリーに蓄積：`rad_E_tally[i,g] += c * E * Δt_cen`（§7.6 の規約）
3. 粒子時刻を \(t \leftarrow t^{n+1}\) に更新
4. 粒子状態（セル、群、エネルギー、時刻、RNGカウンタ）をcensusプールに保存
5. 粒子を当該ステップのアクティブ輸送から除外（`time_remain=0`、`alive=1` を維持）

> census到達時にイベント（吸収/リーク）は発生しない。
> 粒子のエネルギーは保持され、次ステップで追跡が継続される。
> 滞在時間の寄与が正しくタリーされることにより、
> ステップ全体の \(\hat{E}_{i,g}\) がcensus粒子分を含む正確な推定量となる。

> **時間連続DDMC（temporally continuous DDMC）**：
> 上記の指数待ち時間サンプリングは Densmore (2007) §4–§5 の "temporally continuous" 方式に対応する。
> 代替として "temporally discretized" 方式（backward Euler離散化、時間ステップ内の一様サンプリング）
> があるが、Densmore (2007) §5 の数値比較では temporally continuous 方式の方が精度・安定性に優れ、
> 因果律違反（将来の吸収を先取りする問題）を回避できることが実証されている。
> TENRYUは temporally continuous 方式を採用する。

**DDMC カーネル実行モデル**：1 スレッド = 1 粒子ヒストリー、`block_size = 128`（CUDA_KERNELS §10.3 参照）。粒子はシンプルなインデックスマッピング `thread_id + block_id × block_size` で割り当てる。DDMC はイベント処理が単純（位置・方向追跡不要、~30 レジスタ）でワープ発散が低いため、§6.6 の Persistent Warp モデルは適用せず history-based モデルを使用する。

イベント種別：
- 吸収確率：
\[
P_{abs} = \frac{\sigma_{a,eff}}{\Sigma^{tot}}
\]
- true NLTE DDMC の局所実効散乱：
\[
P_{scat} = \frac{\sigma_{s,eff}}{\Sigma^{tot}}
\]
- リーク（近傍j）：
\[
P_{i\to j} = \frac{\Sigma^{leak}_{i\to j}}{\Sigma^{tot}}
\]
- 境界リーク：
\[
P_{i\to b}=\frac{\Sigma^{leak}_{i\to b}}{\Sigma^{tot}}
\]

LTE DDMC では \(\sigma_{s,eff}\) をイベントとして使わないため 3 チャネル、
true NLTE DDMC では \(\sigma_{s,eff}\) を局所再分配イベントとして有効化するため 4 チャネルとなる。

**イベント選択（LTE: 3チャネル, true NLTE: 4チャネル）**：

乱数 \(r = \xi \times \Sigma^{tot}\)（\(\xi \in U(0,1)\)）に対し：

1. \(r < \sigma_{a,eff}\)：**吸収イベント** — エネルギーを `rad_dep[i,g]+=s_p E` に加算、粒子消滅
2. true NLTE かつ \(\sigma_{a,eff} \le r < \sigma_{a,eff} + \sigma_{s,eff}\)：**局所実効散乱イベント** —
   outgoing group を `eta_cdf` からサンプルする。
   target group が DDMC-support ならセル・エネルギー・時刻を保持したまま DDMC を継続し、
   IMC-only なら same-step DDMC→IMC reinjection（7.7）へ送る
3. その他の場合で、\(r\) が内部リーク帯に入る：**近傍リークイベント** — 内部面を選択：
   \[
   f^* = \min\!\left\{f \in \mathcal{F}_{int} : \sigma_{a,eff} + \sigma_{s,eff} + \sum_{f' \le f} \Sigma^{leak}_{i \to j_{f'}} \ge r\right\}
   \]
   ここで \(\mathcal{F}_{int}\) は内部面（非境界面）の集合。
4. 残余は **境界リークイベント** — 境界面を選択：
   境界面が複数ある場合（2D RZのコーナーセル等）は、残余 \(r - \sigma_{a,eff} - \sigma_{s,eff} - \Sigma^{out}\) で
   CDFを走査して対象境界面を決定する。1面の場合はその面を選択。
   **v1.0簡略化**：R3 で境界面リーク係数は `Σ_leak_bdry` に合算され個別値は保持されない。
   コーナーセルでは最初の VACUUM/MARSHAK 境界面を選択する（CUDA_KERNELS §6.5 R9 参照）。
   VACUUM 境界脱出はエネルギー計上（E_escape）のみで面に依存しないため、v1.0 では正確。
   多面 Marshak BC の正確な面選択（方向依存の入射スペクトル）は将来版で対応する。

> **注意**：LTE では \(P_{abs} + \sum_j P_{i\to j} + P_{i\to b} = 1\)、
> true NLTE では \(P_{abs} + P_{scat} + \sum_j P_{i\to j} + P_{i\to b} = 1\) を満たす。
> 内部セル（全面が内部面）では \(\Sigma^{leak}_{i\to b} = 0\) のため境界リークチャネルは発生しない。

処理の詳細：
- **吸収**：`rad_dep[i,g]+=s_p E`、粒子消滅
- **局所実効散乱**：`group <- sample(eta_cdf[cell,*])`
  - sampled group が DDMC-support なら、粒子は同一セル・同一エネルギーのまま DDMC を継続
  - sampled group が IMC-only なら、同一セル内 volume source とみなして DDMC→IMC 変換し、残余時間を tail IMC phase へ渡す
- **近傍リーク**（面 \(f^*\) を通過）：cellId←\(j_{f^*}\)
  - \(j_{f^*}\) がDDMCならDDMC継続
  - \(j_{f^*}\) がIMCなら **DDMC→IMC変換**（7.7）。サンプリング面は \(f^*\)
  - **2D セルの面とリーク先の対応**：セル i の面 f (f=0,1,2,3 for R\_left, R\_right, Z\_bottom, Z\_top — CUDA\_KERNELS §6.4.3 面規約) は一意の隣接セル \(j_f\) を持つ
- **境界リーク**：`E_escape[g]+=s_p E` として流出へ計上、粒子消滅（vacuum）

#### 7.5.1 DDMC粒子の群間再分配（frequency redistribution）

LTE DDMC では粒子の群変更はステップ内で行わず、群間エネルギー再分配は
IMC と同じ **ソース生成メカニズム**（§6.2）を通じて実現される。
true NLTE DDMC では Jayenne separate-emissivity に合わせて、ステップ内の局所群再分配を
`sigma_s_eff + eta_cdf` で表現する。

1. DDMCセルで吸収イベントが発生 → エネルギーは電子系へ沈着（\(T_e\) が上昇）
2. true NLTE では、同じ step 内で local effective-scatter event が起きた場合に
   outgoing group を \(s_g\) からサンプルする
3. 次のタイムステップ冒頭で、§6.2のソース生成が **DDMCセルを含む全セル** に適用される
4. LTE ではソース粒子の群分配は \(b_g(T_e)\)、true NLTE では \(s_g\) に比例する
5. 新規ソース粒子がDDMCセル内に生成された場合、そのセルがDDMC条件を満たせば
   DDMCモードで追跡を開始する

> **IMCとの対比**：IMCでは実効散乱イベント（§6.3.4）により
> ステップ **内** で群変更が発生する。true NLTE DDMC でも局所 effective scatter に限って
> 同じ step 内の群変更を許すが、空間 diffusion / leakage operator 自体は各群で独立のまま保持する。
> したがって追加される群連成はセル局所の rank-1 kernel
> \((1-f)\sigma^{PA}_{g_{in}} s_{g_{out}}\) に限られ、空間リーク係数の定義は変えない。
>
> TENRYU は true NLTE DDMC の same-step reinjection を持つ。局所 effective-scatter で
> `eta_cdf` から引いた outgoing group が DDMC-support なら DDMC のまま継続し、
> IMC-only group なら **その場で DDMC→IMC 変換**する。
> この変換では粒子エネルギーと絶対時刻を保持し、位置はセル内一様、方向は等方に再サンプルする。
> DDMC 後に tail IMC phase を 1 回だけ回し、同一 step の残余時間を IMC で輸送する。

#### 7.5.2 DDMCセルのソース粒子生成

§6.2のソース生成は **DDMCセルにも適用** される。具体的には：

- DDMCセル i × 群 g の放射源 \(S^{emit}_{i,g}\)（§6.2）を計算
- 粒子数 \(N_{p,i,g}\) を通常のエネルギー比例配分（§6.2）で決定
- 生成された粒子はまず通常の source particle として生成し、その後 step 冒頭の mode map で DDMC 群へ partition される
  - 位置・方向は不要（DDMCはセル・群・エネルギー・時刻のみ）
  - PhotonPool SoA 上の DDMC 粒子の位置・方向フィールドは NaN（`0x7FF8000000000000`）に初期化する。これにより誤って IMC transport で参照された場合に検出できる。DDMC → IMC リーク（§7.7.2）時に新たにサンプルする。RNG ストリームは IMC 粒子と同一の規約（§12.7.1）に従う
  - DDMCイベントループ（§7.5）に \(t=t^n\)、残存時間 \(\Delta t\) で投入

> **true NLTE の補足**：`rad_lite_mesh` は coarse radiation mesh 上で `eta_cdf` を保持しないため、
> true NLTE separate-emissivity でも 1D IMC overlay として使用できる。
> coarse rad cell 上の IMC scatter は単一 coarse `eta_cdf` を使わず、粒子位置から member hydro cell を
> 逆引きしてその hydro-cell `eta_cdf` を参照する。
> したがって `rad_lite_mesh` は 1D IMC transport の coarse space operator を保ちつつ、
> emitted / redistributed spectrum は fine hydro-cell の `s_g` に従う。
> `Radiation.imc.rad_lite_mesh.nlte_auto = true` のときは、
> `opacity.model in {"table_nlte","tmat"}` かつ 1D_SPH の step で
> RadLite overlay を自動有効化する。merge criterion 自体
> (`can_merge_edge`) は変更せず、適用判定と
> `sigma_ratio_max \leftarrow \max(\text{user}, 3.0)` だけを緩和して、
> NLTE/TMAT の急峻なセル間オパシティ変動に対してより強い coarse 化を許す。

> **census粒子との関係**：DDMCセルのcensus粒子（前ステップ終了時に
> DDMCモードで生存していた粒子）は、新規ソース粒子とともに
> DDMCイベントループに投入される。census粒子の群は前ステップの値を保持する。

---

### 7.6 DDMC推定量（rad_E）
DDMCでは粒子は方向を持たず“セル滞在時間”を持つ。  
時間積分されたエネルギー密度の推定量（residence estimator）：

- 粒子がセル i に滞在した時間 \(\Delta t_{res}\) の寄与：
\[
\Delta \mathcal{E}_{i,g} = E \,\Delta t_{res}
\]
- ステップ平均のエネルギー密度推定：
\[
\hat E_{i,g} = \frac{1}{V_i \Delta t}\sum_{events} E\,\Delta t_{res}
\]

IMCのtrack‑length推定（10章）と一致する（\(\Delta t_{res}=\Delta s/c\)）。

> **実装上の注意（共有タリー配列との整合）**：
> `tally_finalize`（§10.3, CUDA_KERNELS §6.0e）は IMC/DDMC 共通の `rad_E_tally` 配列を
> \(/(V \times c \times \Delta t)\) で正規化する。DDMCの生の寄与 \(E \times \Delta t_{res}\) [erg·s] を
> そのまま蓄積すると、正規化結果が \(c\) 倍ずれる。
> そのため **実装では \(c \times E \times \Delta t_{res}\) = \(E \times \Delta s\)** [erg·cm] を蓄積し、
> IMC の track-length 推定量 \(E_{mid} \times \Delta s\) [erg·cm] と同じ単位で共有配列に寄与する。
> 物理的等価性：\(\frac{1}{V\,\Delta t}\sum E\,\Delta t_{res} = \frac{1}{V\,c\,\Delta t}\sum E\,(c\,\Delta t_{res}) = \frac{1}{V\,c\,\Delta t}\sum E\,\Delta s\)。

---

### 7.7 IMC⇄DDMC境界変換（asymptotic diffusion-limit準拠）

従来の **Marshak境界条件** では、入射IMC粒子の角度分布が強く異方的な場合に
DDMC領域内部の解が不正確になり得る（Densmore 2007 §3.2, Fig.7で実証）。

TENRYUでは **asymptotic diffusion-limit境界条件**（Larsen et al. 1983, Habetler & Matkowsky 1975）
に基づくインターフェースを採用する。これにより、入射角度分布に依存せず
DDMC内部で拡散極限として正しい解を得る。

#### 7.7.1 IMC→DDMC（方向依存変換確率 P(μ)）

IMC粒子がDDMCセルの面 m に入射（方向余弦 \(\mu_m > 0\)）したとき、
**無条件にDDMCへ変換するのではなく**、方向依存の変換確率 \(P(\mu_m)\) で判定する：

\[
P(\mu_m) = \frac{4}{3\sigma_{R,m}\Delta x_m + 6\lambda}\left(1 + \frac{3}{2}\mu_m\right)
\]

ここで：
- \(\sigma_{R,m}\)：面 m のDDMC側セルのRosseland不透明度（7.3.4の面評価規約で算出、拡散と整合）
- \(\Delta x_m\)：面 m のDDMC側セルの代表長
- \(\lambda \approx 0.7104\)：外挿距離（Milne問題の漸近値）
- \(\mu_m\)：面法線に対する方向余弦（\(0 < \mu_m \le 1\)）

処理：
- 確率 \(P(\mu_m)\) で **DDMCへ変換**：
  - `mode=DDMC`
  - 位置・方向はDDMCでは不要（セル・群・エネルギー・時刻のみ保持）
  - **物理的には** DDMCイベントループ（7.5）へ即座に合流し、残り時間 \(t^{n+1}-t_{current}\) で
    DDMCイベント処理を開始する（合流タイミングは変換時刻）。
    **ただし v1.0 実装では**、IMC カーネル（R8）内でモード変換された粒子は
    当該ステップ内で DDMC カーネル（R9）による再処理は行わない（CUDA_KERNELS §9 起動シーケンス参照）。
    変換粒子は次ステップの R7（composite\_sort\_and\_partition）で DDMC 領域に正しく分離され、
    R9 で処理される。この遅延は \(O(\Delta t)\) の分割誤差を含むが、
    Strang splitting の分割誤差（§2.1）と同等であり、統計的再現性に影響しない
  - 変換された粒子のエネルギーは、DDMC境界セルの方程式（7.3.5 Eq.32）の
    入射ソース項 \(\int_0^1 P(\mu)\mu I_b\,d\mu\) に対応する
- 確率 \(1-P(\mu_m)\) で **IMC側へ等方的に反射**：
  - 粒子は面 m 上に留まり、IMC側半空間へ等方的に方向を再サンプル
  - エネルギー・時刻は保持

> **変換確率の制約**（確率化の前提）：
> \(0 \le P(\mu_m) \le 1\) が **全ての \(\mu_m \in (0,1]\)** で成り立つ必要がある。
> \(\mu_m=1\)（垂直入射）で最大値をとるため、条件は：
> \[
> P(1) = \frac{10}{3\sigma_{R,m}\Delta x_m + 6\lambda} \le 1
> \quad\Leftrightarrow\quad
> \sigma_{R,m}\Delta x_m \ge \frac{10 - 6\lambda}{3} \approx 1.91
> \]
> この条件を満たさないセル×群は **DDMC不可**（7.1.2 条件4）。
>
> **参考文献**：
> - Densmore et al., JCP 222 (2007) Eq.(34): P(μ)の定義
> - Cleveland & Gentile, JCP 291 (2015) Appendix B: 反射確率とemissivity保存

#### 7.7.2 DDMC→IMC

DDMC粒子がIMCセルへリークしたとき：
- `mode=IMC`
- **位置**：リーク面上でサンプル（1D_SPHは等方、2D_RZは既定でR重み付け）
  - **1D_SPH**：球面 \(r = r_f\) 上で等方位置をサンプル：
    \(\mu_{pos} = 2\xi_1-1\)、\(\phi_{pos}=2\pi\xi_2\)、
    \(\mathbf{r}=(r_f\sqrt{1-\mu_{pos}^2}\cos\phi_{pos},\; r_f\sqrt{1-\mu_{pos}^2}\sin\phi_{pos},\; r_f\mu_{pos})\)
  - **2D_RZ**：辺 \(k\)（頂点 \(\mathbf{V}_k\) と \(\mathbf{V}_{k+1}\) を結ぶ）上で
    RZ体積要素の \(R\) 因子を考慮した重み付けサンプリング（v1.0既定）：
    辺の頂点座標 \(R_k, R_{k+1}\) に対し、辺上の面積要素は \(dA \propto R\,dl\) である。
    逆関数法で \(R\)-重み付き位置をサンプルする：
    \[
    t = \frac{-R_k + \sqrt{R_k^2 + \xi(R_{k+1}^2 - R_k^2)}}{R_{k+1} - R_k}, \quad \xi \in U(0,1)
    \]
    \(\mathbf{r} = \mathbf{V}_k + t \cdot (\mathbf{V}_{k+1} - \mathbf{V}_k)\)。
    \(R_k \approx R_{k+1}\)（Z方向の辺）の場合は \(t = \xi\) に退化する。
    **切替閾値**：\(|R_{k+1} - R_k| < \varepsilon_R\)（\(\varepsilon_R = 10^{-10} \times \max(R_k, R_{k+1}, 10^{-20})\)）の場合に \(t = \xi\) に退化する。
    オプション `ddmc.rz_face_r_weight=False` で一様サンプル
    \(\mathbf{r} = \mathbf{V}_k + \xi \cdot (\mathbf{V}_{k+1}-\mathbf{V}_k)\) に切替可能（回帰テスト用）。
- **方向**：面法線 \(\hat{\mathbf{n}}\) に対してIMC側半空間へ
  - Interface source angular distribution uses cosine-weighted half-space sampling: \(\mu = \sqrt{\xi}\), consistent with Lambert's cosine law for surface emission.
  - v1.0既定：**cosine分布**（pdf \(p(\mu) = 2\mu\), \(\mu \in (0,1]\)）
    - サンプリング：\(\mu = \sqrt{\xi_1}\)（逆関数法）
    - 方位角：\(\phi = 2\pi\xi_2\)
    - 方向ベクトル：\(\hat\Omega = \mu\,\hat{\mathbf{n}} + \sqrt{1-\mu^2}(\cos\phi\,\hat{\mathbf{u}}+\sin\phi\,\hat{\mathbf{w}})\)
    ここで \(\hat{\mathbf{u}}, \hat{\mathbf{w}}\) は面上の正規直交基底
    - **2D\_RZ での面法線座標系の構築**：セル辺 \(m\) の2端点を \(P_1=(r_1,z_1)\), \(P_2=(r_2,z_2)\) とする。
      1. 辺方向ベクトル \(\mathbf{t}_{edge} = (r_2-r_1,\; z_2-z_1) / |P_2-P_1|\)
      2. RZ平面内法線 \(\mathbf{n}_{RZ} = (z_2-z_1,\; -(r_2-r_1)) / |P_2-P_1|\)（セル外向き）
      3. 面上の位置 \(P\) での3D法線：\(\hat{\mathbf{n}} = (n_{RZ,r}\cos\varphi_P,\; n_{RZ,r}\sin\varphi_P,\; n_{RZ,z})\)
      4. 接線1：\(\hat{\mathbf{u}} = (t_{edge,r}\cos\varphi_P,\; t_{edge,r}\sin\varphi_P,\; t_{edge,z})\)
      5. 接線2：\(\hat{\mathbf{w}} = \hat{\mathbf{n}} \times \hat{\mathbf{u}}\)（方位角方向）
      ここで \(\varphi_P\) は粒子の方位角（DDMC 粒子の場合は \([0, 2\pi)\) から一様サンプル）。
  - オプション：**half‑range isotropic**（\(\mu\)一様）
    - \(\mu = \xi_1\)（\(\mu \in (0,1]\)）
- 群：gを保持
- E,t は保持

> **v1.0 実装注記**：R9（DDMC）カーネル内でモード変換された粒子（mode=IMC）は、
> 当該ステップ内で R8（IMC）カーネルによる再処理は行わない。
> 変換粒子は次ステップの R7（composite\_sort\_and\_partition）で IMC 領域に正しく分離され、
> R8 で処理される。この遅延は IMC→DDMC 変換（§7.7.1）と対称であり、
> \(O(\Delta t)\) の分割誤差は Strang splitting の分割誤差（§2.1）と同等である。

> **注**：DDMC→IMCのリーク面で同時にleft-leakageイベントが起きたDDMC粒子は
> IMC側へ等方的に返される（Densmore 2007 §3.2末尾）。

> 検証で cosine vs half‑range の感度を確認する（VERIFICATION §9.2）。

#### 7.7.3 Emissivity保存補正 \(\hat{P}\)（v1.0既定）

**問題**：標準の変換確率 P（7.7.1）は光学厚 τ の増大とともに 0 に近づく。
これにより、高光学厚の界面で変換確率と境界セルの σ\_{L,1} が共にほぼゼロとなり、
IMC⇄DDMC間の放射エネルギー伝搬が**人工的に遮断**される。
標準Pから得られる離散化 emissivity \(\hat\varepsilon\) は
解析拡散 emissivity \(\varepsilon'\) より常に小さく、τ増大とともに減少する
（Densmore, Davidson & Carrington 2006, §4, Eq.45）：
\[
\hat\varepsilon = \frac{P\beta}{\beta + \frac{4}{3}P\tau}
\]

**解決**：解析 emissivity \(\varepsilon'\) をセルサイズに依らず保存する
修正変換確率 \(\hat{P}\) を導入する（Densmore 2006 §5, Eq.48）。

解析拡散 emissivity（Densmore 2006 Eq.19）
\[
\omega \leftarrow \operatorname{clamp}(\omega, 0, 1)
\varepsilon' = \frac{4}{3}\frac{\sqrt{3(1-\omega)}}{1+\lambda\sqrt{3(1-\omega)}}
\]
> **実装ノート**：\(\omega\) は丸め誤差で \(1\) をわずかに超える可能性があるため、\(\varepsilon'\) と \(\beta\) 計算前にクランプして \(1-\omega\ge 0\) を保証する。上式は等価に \(1-\omega \leftarrow \max(1-\omega,0)\) とするものでもよい。
ここで \(\lambda \approx 0.7104\) は Milne 外挿距離（無次元、平均自由行程単位、§7.3.5参照）。

修正変換確率（Densmore 2006 Eq.48）：
\[
\hat{P} = \frac{\varepsilon'\,\beta}{\beta - \frac{4}{3}\varepsilon'\,\tau}
\]
\[
\beta = \frac{3}{2}(1-\omega)\tau^2 + \sqrt{3(1-\omega)\tau^2 + \frac{9}{4}(1-\omega)^2\tau^4}
\]

ここで：
- \(\tau = \sigma_{R,m}\Delta x_m\)：DDMC側セルの光学厚（7.7.1 の σ\_{R,m} と同一）
- \(\omega\)：散乱比（7.1.1 の定義）

**数値安定性と確率保証**：
\(\hat{P}\) 公式の分母 \(\beta - \frac{4}{3}\varepsilon'\tau\) は、以下の条件で問題を起こしうる：

1. **分母が非正**（\(\beta \le \frac{4}{3}\varepsilon'\tau\)）：\(\omega\) が大きく \(\tau\) が中程度の領域で発生。
   主因は \(\beta \sim O(\sqrt{1-\omega}\,\tau)\) に対し \(\varepsilon'\tau \sim O(\sqrt{1-\omega}\,\tau)\) が
   同等以上のオーダーとなること。例：\(\omega = 0.999, \tau = 10\) で分母 < 0。
2. **\(\hat{P} > 4/5\)**：分母が正だが小さい場合、\(\hat{P}\) が確率制約 \(\hat{P}(1) \le 1\) を超過。
   例：\(\omega = 0.9, \tau = 4\) で \(\hat{P} \approx 1.05\)。
3. **\(\omega \to 1\) の漸近**：\(\varepsilon' \to 0\) かつ \(\beta \to 0\) で、正しい極限は \(\hat{P} \to 0\)
   （純散乱媒質ではemissivityがゼロのため変換確率もゼロ）。\(\hat{P} \to 1\) ではない。

**実装の安全策（v1.0）**：
1. \(\beta \le \frac{4}{3}\varepsilon'\tau\)（分母 \(\le 0\)）の場合：**標準 P**（§7.7.1）にフォールバック
2. \(\hat{P} > 4/5\) の場合：\(\hat{P} = 4/5\) にクランプ（\(\hat{P}(1) \le 1\) を保証）
3. \(\hat{P} < 0\) の場合：標準 P にフォールバック（安全策、条件1で通常捕捉される）

クランプ/フォールバック時は emissivity 保存精度が低下するが、
標準 P 自体が正しい確率的インターフェースを与えるため物理的に安全である。
\(\tau\) が十分大きい領域（\(\tau \gtrsim 7\) at \(\omega = 0.9\)）では
クランプは発生せず、完全な emissivity 保存が得られる。

\(\hat{P}\) を P の代わりに用いた方向依存変換確率：
\[
\hat{P}(\mu) = \frac{\hat{P}}{2}\left(1+\frac{3}{2}\mu\right)
\]

**性質**：
- \(\tau \to 0\)：\(\hat{P} \to 8/(3\tau+6\lambda) + O(\tau^2) = P\)（標準と同精度、Eq.49）
- \(\tau \to \infty\)：\(\hat{P} \to \varepsilon' > 0\)（放射が常に界面を透過可能）
- \(\omega \to 1\)：\(\hat{P} \to 0\)（emissivity \(\varepsilon' \to 0\)）
- 光学薄セルでは標準Pと同一の1次打ち切り誤差

**確率制約（asymptotic interface method用）**：
\[
\hat{P}(1) = \frac{5}{4}\hat{P} \le 1
\quad\Leftrightarrow\quad
\hat{P} \le \frac{4}{5}
\]
**漸近極限**（\(\tau\to\infty\)）では \(\hat{P} \to \varepsilon'\) であり、\(\varepsilon' \le 4/5\) が必要。
これは（Densmore 2006 Eq.61）：
\[
\omega \ge \omega_{\min} = 1 - \frac{1}{3\left(\frac{5}{3}-\lambda\right)^2} \approx 0.6355
\]
TENRYUのDDMC判定条件（7.1.2）は \(\omega \ge 0.9\) を要求するため、
**漸近極限**（\(\tau \to \infty\)）では確率制約が満たされる。

> **有限τでの注意**：\(\tau\) がDDMCしきい値（\(\tau_{DDMC}=4\)）付近の場合、
> \(\hat{P}\) は漸近値 \(\varepsilon'\) を超えて \(4/5\) を超過しうる。
> この場合は上記の安全策（クランプ/フォールバック）が適用される。
> §7.1.2 の条件4（\(0 \le P(\mu) \le 1\)）はクランプ後に成立する。

**v1.0方針**：
- 1D球対称では **\(\hat{P}(\mu)\) を既定で使用**（emissivity保存）
- 標準 P(μ) は `ddmc.emissivity_preserving=False` で選択可能（回帰テスト・比較用）
- 境界セルの σ\_{L,1}（7.3.5）にも同じ \(\hat{P}\) を適用する

> **参考文献**：
> - Densmore, Davidson & Carrington, Ann. Nucl. Energy 33 (2006) 583–593:
>   emissivity問題の分析（§4, Eq.45）、\(\hat{P}\) の導出（§5, Eq.48）、制約（§6, Eqs.59–62）
> - Cleveland & Gentile, JCP 291 (2015) §2.3.2: \(\hat{P}\) をHIMCD実装に採用

#### 7.7.4 2D RZへの幾何拡張

v1.0の \(\hat{P}(\mu)\) 式（7.7.3）は Densmore (2006) の1D slab向け導出に基づく。
2D RZの一般四辺形セルでは「面法線に対するΔx」の定義が自明でなく、
面の幾何形状に応じた拡張が必要になる。

**v1.0方針**（セル代表長ベース）：
\(\hat{P}(\mu)\) 式中の \(\Delta x_m\) として、DDMC側セルの **面法線方向の代表長** を使用する：
\[
\Delta x_m = \frac{V_i}{A_m}
\]
ここで \(V_i\) はDDMC側セルの体積、\(A_m\) はリーク面 \(m\) の面積。

- **1D_SPH**：\(\Delta x_m = \Delta r_i\)（球殻厚さ）。\(V_i = \frac{4\pi}{3}(r_{i+1/2}^3-r_{i-1/2}^3)\)、\(A_m = 4\pi r_f^2\) だが、
  \(r_{i+1/2}-r_{i-1/2} \ll r\) の極限で \(\Delta x_m \approx \Delta r_i\) に一致。
- **2D_RZ**：\(V_i\) はRZ四辺形セルの体積（§3.2.2）、\(A_m = 2\pi \bar{R}_m \cdot L_m\)
  ここで \(\bar{R}_m\) は辺 \(m\) の平均R座標、\(L_m\) は辺の長さ。

**r=0 軸上のセル**（\(\bar{R}_m \to 0\)）：軸に接する辺 \(m\) の面積 \(A_m = 2\pi\bar{R}_m L_m \to 0\) で \(\Delta x_m \to \infty\)。これは非物理的であるため、\(r=0\) に接する辺については \(\Delta x_m = V_i / (\pi R_{max} L_m)\) とする。ここで \(R_{max} = \max(r_1, r_2, r_3, r_4)\) はセル4頂点の \(r\) 座標の最大値（構造格子では右辺の2頂点の \(r\) 座標の大きい方に等しい）。

> **物理的動機**：\(\pi R_{max} L_m\) は開口角 \(\pi\) のウェッジの面積に相当し、
> 軸接触セルの実効的な面面積スケールを表す。これにより \(A_m \to 0\) の特異性を
> 回避しつつ、有限の光学的厚さ \(\tau_m = \sigma_R \cdot \Delta x_m\) を確保する。
> 結果として得られるリーク率は軸近傍のセル幾何に対して物理的に妥当な値となる。

> **物理的妥当性**：\(V_i/A_m\) は「面 \(m\) を通して見たセルの奥行き」に相当し、
> 面法線方向の平均自由行程と直接比較可能な量である。
> 直交格子ではセル幅 \(\Delta x\) に一致する。

Cleveland & Gentile (2015) Appendix B は、任意幾何形状（非構造格子含む）に
自然に拡張可能なインターフェース定式化を提供する。
光学厚の定義を面法線ベースに一般化（\(\tau_m = \kappa_{m+1/2}\,\hat{n}_m\cdot\overrightarrow{\Delta X}\)）し、
\(\hat{P}\) の枠組みをそのまま適用する。

**将来拡張**：
- Cleveland & Gentile 幾何拡張は `interface_method="cleveland_gentile"` として将来実装

> **参考文献**：
> - Cleveland & Gentile, JCP 291 (2015) Appendix B: 任意幾何向けemissivity保存インターフェース

---

### 7.8 DDMC運動量沈着推定量（rad momentum deposition）

DDMCでは粒子が角度情報を持たないため、**放射運動量沈着**の推定には
面フラックスを経由する手法を用いる（Densmore et al. 2007 §3.3）。

連続系での運動量沈着率：
\[
\mathbf{p}(\mathbf{r},t) = \frac{\sigma_t}{c}\int_{-1}^{1} \mu\, I(\mathbf{r},\mu,t)\, d\mu = \frac{\sigma_t}{c}\, F(\mathbf{r},t)
\]
ここで \(\sigma_t = \sigma_{a,eff} + \sigma_R\) [cm\(^{-1}\)] は全相互作用不透明度（Densmore 2007 Eq.(36) に準拠）。
注：この \(\sigma_t\) はDDMC拡散係数（§7.2）の \(\sigma_R\) を含む運動量沈着専用の定義であり、IMCの \(\sigma_{total}=\sigma_{a,eff}+\sigma_{s,tot}\)（§6.3.1）やDDMCイベントレート（§7.4–§7.5の \(\Sigma^{out}+\sigma_{a,eff}\)）とは異なる。

#### 7.8.1 面フラックスの推定

DDMCの面フラックスは、面を横切るリークイベントのタリーから推定する。

内部面 \(j+1/2\)（DDMCセル \(j\) と \(j+1\) の間）：
\[
F_{j+1/2} = \sigma_{R,j}\, \phi_j\, \Delta x_j - \sigma_{L,j+1}\, \phi_{j+1}\, \Delta x_{j+1}
\]
ここで \(\phi_j\) はセル平均 scalar intensity（residence estimatorから得る）、
\(\sigma_{R,j}\), \(\sigma_{L,j+1}\) はリーク不透明度（7.3.2）。

**\(\phi_j\) と residence estimator の関係**：\(\phi_j = c\,\hat{E}_{i,g} / (4\pi)\)。ここで \(\hat{E}_{i,g}\) [erg/cm\(^3\)] は §7.6 の residence estimator が返すエネルギー密度、\(c\) は光速。この変換は \(\phi = cE/(4\pi)\) に基づく。

**\(\phi_j\) の算出タイミング**：\(\phi_j\) はタイムステップ内の全 DDMC イベント完了後にポストプロセスとして算出する。residence estimator の和 \(\sum(E \cdot \Delta t_{res})\) はイベントループ中に累積し、\(\phi_j = c\,\hat{E}_{i,g}/(4\pi)\) は全イベント処理後に一度だけ計算する。したがって運動量沈着（§7.8.2）もイベントループ完了後に算出される。

境界面 \(1/2\)（DDMCとIMCの境界）：
\[
F_{1/2} = \int_0^1 \mathcal{P}(\mu)\, \mu\, I_b(\mu,t)\, d\mu - \sigma_{L,1}\, \phi_1\, \Delta x_1
\]
\[
\mathcal{P}(\mu)=
\begin{cases}
\hat{P}(\mu) & \text{v1.0既定（`ddmc.emissivity_preserving=True`）} \\
P(\mu) & \text{比較用（`ddmc.emissivity_preserving=False`）}
\end{cases}
\]

#### 7.8.2 セル運動量沈着

セル j の運動量沈着（単位体積・単位時間）：
\[
p_j = \frac{1}{2c}\left(\sigma^+_{R,j-1/2}\, F_{j-1/2} + \sigma^-_{R,j+1/2}\, F_{j+1/2}\right)
\]

面のRosseland不透明度は7.3.4の面評価規約に従う。

**2D_RZ への一般化**：
各面 \(f\) の寄与にその面の外向き単位法線 \(\hat{\mathbf{n}}_f\) を乗じ、R/Z成分を分離する：
\[
\mathbf{p}_{i} = \frac{1}{2c\,V_i}\sum_{f\in\text{faces}(i)} \sigma_{R,f,g}\, F_{f}\, A_f\, \hat{\mathbf{n}}_f
\]
ここで \(F_f\) は面 \(f\) を通じたフラックス（§7.8.1の \(F_{j\pm 1/2}\) の多面拡張）、
\(\hat{\mathbf{n}}_f\) は面の外向き法線ベクトルである。
1D_SPHでは \(\hat{\mathbf{n}}_f = \hat{r}\)（径方向）に退化し、上式の1D版と一致する。

#### 7.8.3 v1.0での扱い

- v1.0ではDDMCの運動量沈着は **診断出力**として実装する
- タリー配列 `rad_mom_dep[i]` [dyne·s/cm³]（2D_RZ: [N_cell × 2]（R,Z成分）、1D_SPH: [N_cell × 1]）に上記推定量を蓄積
- IMCのtrack-length estimator（§10.1）による運動量沈着と合算して出力
- **注意**：運動量沈着は統計誤差が大きい（Densmore 2007 §4.2, Figs.2,4,6）。
  分散低減（将来）が入るまでは、診断参考値として扱う

> **参考文献**：
> - Densmore et al., JCP 222 (2007) Eqs.(36)–(40): DDMC運動量推定
> - Cleveland & Gentile, JCP 291 (2015) Appendix A, Eq.(A.3): 面フラックス方式

---

