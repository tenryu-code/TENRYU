<!-- 分割元: docs/NUMERICS.md | このファイルは参照用です。原本（docs/NUMERICS.md）が権威です。 -->
### 6.7 Multigroup Flux-Limited Diffusion (FLD)  —【CURRENT — 現行の主放射モデル】

> **【CURRENT RADIATION MODEL — primary】** `mode="multigroup_diffusion"`。1D_SPH/2D_RZ の production 既定（DEFAULT-FLD）。決定論 multigroup flux-limited diffusion。IMC/DDMC/HOLO/difference を完全 bypass。もう一方の現行モデルは §6.8 \(S_N\)。以降 §6.7.x は FLD/\(S_N\) の検証 gate（I1〜I7）。

`Radiation.mode="multigroup_diffusion"` は 1D_SPH と 2D_RZ の production 放射モードである。この
経路では IMC/DDMC/PGRW/HOLO/difference formulation を通らず、群ごとの
flux-limited diffusion と電子物質結合を CUDA 上で解く。FLD は HYDRA-aligned
Fleck linearization を使い、stiff な物質-放射結合を放射線形系へ入れる。

**1D の実装上の規則（2026-09-29 の監査で記載）**：
- **幾何**：面積 \(A_f\) は `Mesh.geometry_1d` に従い、球 \(4\pi r_f^2\)、円筒 \(2\pi r_f\)（単位長さ）、平板 1（単位面積）。
- **群構造**：最初の表不透明度（`tmat`・`table_nlte`、`ionmix` は `table_nlte` の経路）の材料のファイルが群の数と境界を決め、
  `Radiation.groups` と `bounds_eV` を置き換える（`group_repack_hard_xray` は群数を保って境界を再配置する）。2 つ目以降の表は
  群数が違うか、境界の相対差が \(10^{-6}\) を超えると `ConfigError`。表が無く全材料が定数不透明度で境界を与えないときは
  灰色 1 群 \([0,10^6]\) eV を自動設定する。
- **散乱**：FLD は吸収の不透明度だけで輸送し、物理散乱を持たない（`kappa_s > 0` は 2026-09-29 から `ConfigError`）。
- **物質 Newton**（Fleck 経路の電子温度の解）：反復の上限と収束の許容値に外側反復の `max_outer_iterations` と `outer_tol` を
  流用し、1 回の温度の変化を \(\pm\max(T,T_{floor})/2\) に制限する。
- **外側反復の先回り**：非パイプラインの外側反復（Anderson 以外）は、反復 \(k\) の収束判定を host が非同期に読むあいだに、
  状態を複製して反復 \(k+1\) を先に起動する。反復 \(k\) が収束していれば複製を書き戻して先回り分を捨てる（逐次の
  反復と同じ計算になる — 書き戻すのは反復が書き換える全配列（`mutation_spans`）。判定の待ち時間を隠すための実装）。

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
は周波数非依存で全群同値のため cell 単一 \(f_c\) と一致する。群依存 σ の `freq_dep_marshak` 検証 opacity は、単一材料の
デッキでは Fleck 線形化そのものを使わず（\(f=1\)）、多材料のデッキではそのセルが共有カーネルの群に共通の Planck 平均の
\(f\) を使う — 群別の \(f_{c,g}\) が群ごとに異なる値になる経路は現状無い。旧記述の
「\(\sigma_{a,P}\) 単一 \(f_c\) + McClarren-Urbatsch smooth blend」は実装と
乖離していた — blend は stiff 極限 \(zf\to0\) の交換凍結のため 1D では退役済み、
時間形状は既定 `"be"` \(f=1/(1+z)\)（下記 fleck_form 参照）。2026-07-26
doc 真実復元）。この kernel は FLD の
stiff-cell 要件のため IMC 共有の `compute_fleck_kernel` と分離し、IMC safety の
\(\beta\le1\) cap と `f_min_fleck` 下限を適用しない。table_nlte/tmat の NLTE
係数経路（`eval_nlte_opacity_emission` — cell 単一 \(f_c\) を emission-mean
\(\sigma_{P,\rm em}=\sum_g\eta_g/(a_{eV}cT_e^4)\) から作る）も同様に
\(\beta\le1\) cap を適用しない（2026-07-26： ideal-gas
fallback 分岐に残存していた IMC 系譜 cap を除去。cap は「\(f>1\) 防止」に
不要 — \(z\ge0\) で常に \(0<f\le1\) — であり、高温・低 \(C_v\) セルで f を
過大化し Fleck 線形化保護を弱めるだけだった。table cv / state cv 経路は
cap 非経由のため挙動不変）。放射エネルギー式に
\(\rho c_v\) 型の項は入れない。

> **fleck_cv_source（2026-07-10 導入；既定フリップ 2026-07-11）**: 本カーネルの \(\beta=4a_{\rm eV}T_e^3/(\rho c_{v,e})\)
> に入る電子比熱の出所は `Radiation.multigroup_diffusion.fleck_cv_source` で選ぶ。
> `"table"`（**既定**、2026-07-11 フリップ — 外部AI裁定、社内の設計メモ fleck_cv_default_flip_20260711.md）は
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
> knob 不活性 — golden 再生成 bit 同一で実証、社内の検証記録 §4.z3）。
> 冪乗 opacity `power_law`（SPEC §6.4.3）はこの constant 経路と同格に扱われる（eta 構築・
> Fleck blend とも σ 配列値のみが異なる）。冪乗 EOS `power_law_te` は初期化時 tabulation で
> table-EOS 経路に乗る（新規離散化なし）。
>
> **表不透明度のセル（2026-09-23）**: tmat/table_nlte 不透明度のセルでは Fleck 因子を NLTE
> 係数カーネル（上記の cell 単一 \(f_c\)）が作る。1D FLD ではこのカーネルにも `"table"` の比熱
> 規則（電子 EOS テーブルがあれば現在 \(T_e\) の `device_eos_cv`、セルの支配材料のテーブルを
> 優先、無ければ legacy チェーン）を適用する。2026-09-23 まで NLTE カーネルは指定によらず
> legacy チェーン（cv_e_override → state cv_e → ideal gas）を使っており、tmat 不透明度の
> デッキでは既定の `"table"` が効いていなかった（state cv_e は流体の閉包時点の表の cv で、
> 外側反復中の \(T_e\) には追従しない）。2D_RZ の FLD と S_N の呼び出しは従来どおり legacy
> チェーン。下記の secant/guard の β と `exp_phi1` の形は NLTE カーネルにも実装した（2026-09-24。
> 予測子の正味加熱は \(c(\sigma_{P,{\rm abs}}E-\sigma_{P,{\rm em}}aT^4)\)、\(U_e\) は熱容量を取るセルの電子表、
> 表が無いか step 開始の輻射が無いときは tangent。既定 tangent・be の実体は従来どおり — テンプレート
> 実体を分けている。2026-09-23〜24 は namelist で拒否、それ以前は黙って無視されていた）。
> 多材料デッキで LTE の表と周波数依存 Marshak 不透明度が支配するセルは、共有の Fleck カーネルで
> Planck 平均の吸収係数による群共通の \(f\) を使う（単一材料の表が NLTE カーネルから得る灰色の \(f\) と同じ形。
> 群ごとの \(f_g\) を使うと、同じ材料でも他の材料の有無で Fleck 因子が変わっていた）。
>
> **fleck_beta（2026-07-14 導入；外部裁定 2026-07-15、社内の設計メモ fleck_beta_secant_20260714.md §7-8）**:
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

> **fleck_form（2026-07-16 導入；社内の設計メモ fleck_exp_source_20260716.md §2 — β_sec 後続裁定 §12 の係数レバー）**:
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

> **source_integrator（2026-07-16/17 導入 — rung-2、社内の設計メモ fleck_exp_source_20260716.md §3）**:
> 1D FLD の物質–輻射ソース積分器を `Radiation.multigroup_diffusion.source_integrator` で選ぶ。
> `"fleck"`（**既定**、bit 凍結） = 従来のモノリシック半陰的 outer ループ。`"exp_rosenbrock"`
> （opt-in、灰色 v1・fleck_beta tangent 限定・afi/exp_phi1 非互換・persistent 非対応）= Lie 分割:
> (1) 凍結係数の厳密直接移送 \(q=h\varphi_1(-(1+\beta)h)(E-aT^4)\) を両側対称適用（E−=q, U_e+=q —
> 局所保存が構成的に厳密）、(2) 交換項を除いた純拡散陰解（D_face の σ_R は輸送係数として保持）、
> outer Picard なし。実測（社内の検証記録 §7.10 gate (j) + verify_marshak_feature_1d、2026-07-17）:
> 0-D 一段誤差比 err/err_be = 0.430/0.293/0.186/0.109（8/4/2/1e-15 s、多 step で 2 次収束）、
> 総エネルギー drift **厳密 0**（fleck 単一パスの 2.1–2.3 倍非保存と対照）、1-D Marshak feature
> 前線で分割バイアス 0（最細 rung で fleck と同一セル・N=1024 参照 4dx 内、非劣化全 rung）。
> OFF-bit: exchange_off 配線+ループ再入れ子込みバイナリで GXII golden・HR PASS（既定経路恒等）。
> **多群（G≤96、2026-07-17 導入 — 社内の設計メモ exp_mg_phi1_20260717.md、外部裁定採択）**: 群数の上限 96 はセルごとの
> 作業配列の大きさ（`fld_1d_gpu.cu` の `kMaxG`）。builder は `Radiation(...)` の解析時に deck の群数を、2026-09-29 からは
> 最初の不透明度表が群数を置き換えた後の最終の群数も検査する（それまでは 96 を超える表が解析時の検査をすり抜け、
> 実行時に全セルの交換が棄却された）。保存超平面上の
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
> 表面セル交換を飛ばす偏りを除去、社内の設計メモ exp_mg_phi1_20260717.md）。gate (k)（G=4 等 σ・1 step）: **周辺化恒等 |E_tot^{mg}−E_tot^{grey}| ≤ 1.9e-16**
> （等 σ・Σb=1・Σdb/dT=0 で多群凍結系の総和は灰色凍結系へ厳密周辺化 — 別経路計算の 1 ulp 一致が
> rank-1 機構の判別的認証）・保存 drift 厳密 0・2 次 slope 135×/8×。灰色 G=1 は従来スカラー kernel を
> bit 不変で維持。

1D_SPH の面幾何は \(A_f=4\pi r_f^2\)（円筒 \(2\pi r_f\)・平板 1、上の規則）、\(d_f\) は隣接 cell center 間距離である。
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

FLD state-supply BCは 2D_RZ の Z 端のみ、灰色1群の
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
and `"radial_mean_D_test"` are **DIAGNOSTIC-ONLY** policies for isolating
the diffusion-coefficient contribution, not production defaults. In the currently validated
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

FLD Marshak source BCは 2D_RZ の Z 端のみ、灰色1群の定常入射
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
2026-07-11, 社内の設計メモ 2d_tr_drive_port_spec.md）** — FLD/SN の
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
  将来拡張）、両 z 面 marshak + 面別テーブルは `ConfigError`（単一スカラー
  共有のため; 定数/スカラーテーブル源は両面共通で可）。ledger
  `sn_marshak_in_step` は flux×面積×\(\Delta t\) の既存定義のまま正しい。

検証（社内の検証記録 §7.10、deck 資産 tmp/tr2d_gate/）: 定数-vs-テーブル-vs-面
テーブル bit 恒等（FLD grey/MG、SN grey）、flux 等価（FLD rel ≤2e-16 =
加算順序差のみ、SN は bitwise）、Tr 階段 100→200 eV で per-step
`marshak_in` 比 16.0000 厳密（両ソルバ同値）、MG/grey ledger 比 1.00000000
（\(\sum_g b_g=1\)）、pre-port binary との OFF-bit 恒等。

**1D_SPH FLD 境界条件（W-B, 2026-07-03）** — 内側 \(r=0\) は球対称により常に
反射（face flux 0；`boundary.inner_r` は `"reflect"` 以外を `ConfigError` で拒否）。
外側 \(r=R_{max}\) は `Radiation.multigroup_diffusion.boundary.outer_r` で
`"vacuum"` / `"reflect"` / `"marshak"` を選ぶ。`"reflect"` は face flux 0。`"vacuum"`（既定）と
`"marshak"` は面の値 \(E_f\) に対する Robin 条件 \(F_{out}=h\,(E_f-\theta_g)\) で、`"vacuum"` は half-range
escape \(h=c/2\)、\(\theta_g=0\)、`"marshak"` は Milne 型 \(h=c/4\)、\(\theta_g=F_{\mathrm{inc},g}/h\)（駆動の放射
エネルギー密度）とする。面の値は、外側セルの中心から面までの半セルの拡散抵抗 \(d/D_0\) で消去する
（2026-09-29）。\(D_0=c\,\lambda(R_0)/\sigma_{R,0}\) は外側セル自身の Rosseland 不透明度による flux-limited 係数で、
\(R_0=|E_{0,g}-E_{1,g}|/(\Delta r\,\sigma_{R,0}\,E_{0,g})\)（\(E_{1,g}\) は内側の隣のセル、\(\Delta r\) は 2 つのセル中心の
距離、場は内部面の係数と同じ遅れた場。内部の面は 2 つのセルの平均 \(\tfrac12(E_l+E_r)\) で割るが、ここは外側セル自身の
値で割る）、\(d\) は外側セルの幅の半分で、
\[
F_{out}=h_{eff}\left(E_{0,g}-\theta_g\right),\qquad h_{eff}=\frac{h}{1+h\,d/D_0}
\]
となる。外側セル行の diagonal に \(\Delta t\,A\,h_{eff}\)、RHS に \(\Delta t\,A\,(h_{eff}/h)\,F_{\mathrm{inc},g}\) を加える
（\(A\) は外側の面の面積、球では \(4\pi R_{max}^2\)）。\(d/D_0\to0\)（外側セルが光学的に薄く、半セルの中の変化が緩やかなとき）では 2026-09-29 以前の
セル中心の値で閉じる式 \(F_{out}=h\,(E_{0,g}-\theta_g)\) に戻り、光学的に厚いセルでは面が \(\theta_g\) に保たれる
（真空では 0）。流束制限が効く \(R_0\gg1\) では \(D_0\approx c\,E_{0,g}\,\Delta r/|E_{0,g}-E_{1,g}|\) となり、光学的厚さに
よらず \(h\,d/D_0\approx(h/c)(d/\Delta r)\,|E_{0,g}-E_{1,g}|/E_{0,g}\) が残る。内側から急な勾配で放射が届く外側セル
（\(E_{0,g}\ll E_{1,g}\)）では、外側セルが満ちるまで面の結合が弱い（場は遅れた場で、外側反復の収束判定は物質温度
の変化だけを見るので、1 回目で収束した刻みではその刻みの間続く）。\(E_{0,g}=0\) で隣が正のとき、分母を \(10^{-300}\) で押さえた \(R_0\) は
\(E_{1,g}/(\Delta r\,\sigma_{R,0})\) が約 \(2\times10^{8}\)（cgs）を超えると倍精度の範囲を超えて非有限になり、\(\lambda\) は非有限の
\(R\) を 0 と読むので拡散極限の 1/3 をとる（\(R_0\) が有限に収まるときは \(\lambda\approx0\) で面はほぼ閉じる）。どちらも係数は
有限で、脱出の集計は行列と同じ \(h_{eff}\) を使うので保存は保たれる。セル中心の値で閉じる式は光学的に厚い外側セルで 1 次の誤差をもち、Marshak の流入と真空への
脱出を多く見積もっていた（社内の検証記録 §23：外側セルの光学的厚さ 0.78 の平板で、面から入った正味のエネルギーが
+18.7 %、真空への脱出が +27 %。新しい式では −4.4 %・+5.5 % で、2 次以上で同じ極限へ収束する）。持続カーネルの
経路も同じ式を使う。入射駆動は排他的二択: (i) 黒体駆動
`Radiation.boundary.marshak_Tr_eV` \(>0\) で per-group
\(F_{\mathrm{inc},g}=(c/4)a_{eV}T_r^4\,b_g(T_r)\)（\(b_g\) は正規化 Planck 重み、
multigroup 可）、(ii) 灰色定常 flux
`Radiation.multigroup_diffusion.marshak.flux_erg_per_cm2_s`（`groups=1` 限定、
`flux_pulse_duration_s` による矩形パルス対応）。両方指定・両方ゼロは
`ConfigError`。escape tally `fld_escaped_step` は面から出ていく部分流
\(\Delta t\,A\sum_g\bigl[h_{eff,g}E_{0,g}+(1-h_{eff,g}/h)\,F_{\mathrm{inc},g}\bigr]\)（その刻みの最後の解の行列と同じ
\(h_{eff}\)）、incoming は \(\Delta t\,A\sum_g F_{\mathrm{inc},g}\) を `fld_marshak_in_step` として step energy budget に
入り、両者の差が行列の境界項（正味の流出 \(\Delta t\,A\sum_g h_{eff,g}(E_{0,g}-\theta_g)\)）に等しい。Z 端キー（`boundary.z/z_bottom/z_top`）は 1D_SPH では
無意味なので非既定値を `ConfigError` で拒否する。

**Time-dependent drive (indirect-drive mode, 2026-07-09).** The Marshak
drive temperature accepts a deck time callable
`Radiation.boundary.marshak_Tr` (frozen to a table at initialization — no
runtime Python) in addition to the constant `marshak_Tr_eV`. Precedence and
evaluation follow the IMC emitter convention: a positive constant wins;
otherwise the frozen table is evaluated once per radiation call. 1D FLD / SN
(2026-09-23): the evaluation time is the midpoint of the interval the call
advances — the driver passes \(t_n+\Delta t/2\) for the single-stage
Radiation operator and \(t_n+(m+\tfrac12)\Delta t_{\rm sub}\) for thermal substep
\(m\) (the operator advances \([t_n,t_n+\Delta t]\) in both Strang and sequential
order); the historic solve-entry `state.t` \(=t_n\) lagged the drive by
\(\Delta t/2\) and held every substep at \(T_r(t_n)\). Callers that pass no drive
time (verification drivers) keep `state.t`; 2D keeps `state.t`. The per-group incident flux
F_inc,g = (c/4) a T_r^4(t) b_g(T_r(t)) and the `marshak_in` ledger use the
resolved temperature, so a staircase drive T_r: 100→200 eV produces exactly
a 16× per-step `marshak_in` jump (validated). The same wiring applies to the
SN 1D marshak boundary (per-group psi_in; F_inc,g, psi_in and the injected energy
are computed on the device in the host's rounding, the Planck fraction by
`planck_fraction_host_rounding`, the same values as the host loop of
2026-10-01). Marshak outer boundaries refuse
the persistent-kernel path (`marshak_outer_boundary`) and run multi-kernel.

**1D_SPH FLD 体積線源（W-B, Su-Olson 級）** — `Radiation.volume_source_rate`
[erg cm\(^{-3}\) s\(^{-1}\)] \(>0\) かつ `Radiation.volume_source_x_max` [cm]
\(>0\) のとき、セル中心 \(r_c\le x_{max}\) の全セルに RHS へ
\(\Delta t\,V_c\,\dot S\) を加える。Fleck 線形化（物質 emission）の外側で加える
external source であり、`groups=1` 限定（多群は `ConfigError`）。注入エネルギー
\(\sum_{r_c\le x_{max}}\Delta t\,V_c\,\dot S\) は `fld_volume_source_in_step`
として step energy budget（`volume_in`）に計上する。

**1D_SPH \(S_N\) 体積線源（2026-08-31 検証済み）** — 同じ namelist キーを
`sn_transport` でも消費する（群 0 だけに入る灰色の線源なので FLD と同じく `groups=1` 限定。多群は 2026-09-29 から
`ConfigError` — それまでは受理して線源の全量を最低の群に入れていた。`volume_source_x_max` \(>0\) も必須）: sweep の scalar source に等方寄与 \(\dot S/2\)
（GL 規約 \(\sum w=2\) の下で \(\sum_m w_m\,\dot S/2=\dot S\)）、Newton 閉包に
保持項 \(S_{\Delta t}=\Delta t\,\dot S\)（\(E^+=(E^\* + \lambda_{pe}B +
S_{\Delta t})/(1+\lambda_{pa})\)）が入り、Newton の全系エネルギー残差は
\((E^+-E^\*)+(U^+-U^n)-S_{\Delta t}=0\)（残差から \(S_{\Delta t}\) を引き忘れる
と線源セルの物質が注入分を支払わされ Te が床へ張り付く — 2026-08-31 根治）。
外側 Picard の物質基準 \(U^n\)（`sn_Te_old`）は **step 開始値に固定**
（2026-09-01）: 従来は前反復値に鎖結され、多 outer が弱結合交換を反復回数
倍に過大適用し（実測: outer=8 で物質冷却 ×4.9）、体積線源を反復毎に再注入
していた（outer=8 で源点 +25%）。固定後は追加反復が同一の保存的不動点へ
収束する（無源 1 step で outer=8 が outer=1 と bit 一致、線源つき outer=8
が参照帯に着地）ため、outer 数の制限は不要。検証 = 独立 S₈ 離散化
（`tools/su_olson_sn_reference.py`）と ξ=0.01/1.0/3.16 で 0.15%/4.8%/2.3%
一致（2026-09-23 に Newton の上側ブラケットへ \(S_{\Delta t}\) を入れた後の値。以前の
0.2%/5.3%/13.9% には、源が支配的な初期ステップで失われた注入エネルギーの分が含まれていた）（恒久 ctest `test_sn_1d_su_olson` volume-source ケース）。注入エネルギー
\(\sum_{r_c\le x_{max}}\Delta t\,V_c\,\dot S\) は `sn_volume_source_in_step`
として step energy budget（`volume_in`）に計上する（2026-09-23。従来は常に 0
で、線源つき run の保存監査に注入分が現れなかった）。線源のセル配列と注入エネルギーの和
（セル順）は device で計算し、host へは和だけを読む（2026-10-02、`sn1d_internal::volume_source`）。Newton の上側ブラケットは
\(S_{\Delta t}\) を含む（下記 §6.8 の閉包 8 項）。同じ ctest の台帳ケースが、各
solve の注入量と、物質＋放射エネルギーの変化 = 注入 − 流出（注入量の相対
\(10^{-10}\)）と、`rad_dep` が書き戻し後の \(E^{n+1}\) を使うことを検査する。

**1D_SPH \(S_N\) 外側 Marshak 境界（W-B2, 2026-07-03）** —
`Radiation.sn_transport.boundary.outer_r="marshak"`（1D_SPH。当初は `spatial_scheme="linear_characteristic"` 必須、
現在は既定の線形不連続法も受け付ける — 線形不連続法は灰色の流束駆動を、離散の半区間の流れが指定の流束になる等方強度
\(F_{inc,g}/\sum_{\mu<0}w|\mu|\) として与え、黒体駆動は下の \(2F_{inc,g}\)）。外側ノードの内向き半区間
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
= BC 無実、物質は局所平衡）。これは既存輸送特性で **別個の既知問題**として独立
workstream 追跡（純吸収体 ballistic / α和恒等式監査 / S_N 次数掃引が診断
ladder）。vacuum 経路・既存 golden への影響なし（psi_in=nullptr で bitwise 温存）。

**保存形 1D 球面 sweep による解決（2026-07-03）** — 旧 LC sweep は slab 形
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
serial（非 LC）デバッグ sweep は旧スキームのまま（当時の production は LC。1D の既定は 2026-09-25 から線形不連続法、§6.8.4）。

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

**FLD 限流子の face 中心評価化（2026-07-03）** — 1D FLD は
Levermore–Pomraning 限流子引数 \(R=|\nabla E|/(\sigma E)\) を**セル中心**で
評価し D を調和平均で面へ落としていた。放射前線では受け手セルの小さな E が
その D を \(\sim cE_{cold}/|\nabla E|\) に潰し、調和平均が面 flux を
\(\sim cE_{cold}\) に絞る → **前線停滞**（planar 平衡ゲートが暴露、球面は
中心セル微小体積が隠蔽）。修正: 面平均 E と面勾配で **face 上の λ** を評価
（Turner & Stone 2001 の face-centered D 規約; CASTRO II §6.4）、
\(\sigma_{face}=\tfrac12(\sigma_L+\sigma_R)\)（2026-09-23 からはセル幅で重みを付けた
\(\sigma_{face}=(\sigma_Lw_L+\sigma_Rw_R)/(w_L+w_R)\) — 2 つのセル中心の間の光学的厚さを直列に足した、拡散極限の面の抵抗。
幅が等しいと算術平均とビット一致。重みの無い平均は幅比 \(w\) で面の伝導を最大 \((w+1)/2\) 倍誤った。下の face-centered
の段落参照）。滑らか厚極限では旧調和形と
厳密一致 \(\mathrm{harm}(c/3\sigma_L, c/3\sigma_R)\equiv c/(3\bar\sigma)\)
のため変化は前線・急勾配領域のみ。自由流極限キャップは構成上厳密
\(|F|\le cE_{face}\)。GXII golden は前駆加熱の物理変化として再基準化
（ρ_peak 74→50 g/cc 等、社内の検証記録 §10.1）。2D_RZ FLD は同型パターン
（cell 中心 λ + 調和平均）— 2D セッションへ引き継ぎ。



> **実装修正**：旧 2D_RZ FLD は shared internal face でも各隣接 cell が
> 自分の cell-centered face area を使っていたため、z-deformed Lagrangian mesh で
> \(A_{L\to R}\ne A_{R\to L}\) となり、`sum_face_div` に非保存な残差が生じた。
> 共有 edge 面積 \(A_{edge}\) への統一により、この face-area asymmetry を除去した。

> **Degenerate-face handling (2026-05-07)**: Interior FLD
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
中間域のみ raw 形とする（2026-07-26： 旧分岐
\(R<10^{-6}\to1/3\) / \(R>50\to1/R\) は \(R\sim10^{-6}\) 近傍の桁落ち
~\(10^{-4}\) 相対と \(R=50\) での +2% 不連続を持った — 1D 2 コピー修正済み、
2D コピーは別途対応）。面係数の構成は次元で異なる（2026-07-26 doc 真実復元）:
**1D_SPH は face-centered 評価**（1D face 中心評価化後の現行実装）—
\(\sigma_{R,f}=(\sigma_{R,L}\Delta_L+\sigma_{R,R}\Delta_R)/(\Delta_L+\Delta_R)\)
（\(\Delta_{L,R}\) は両セルの幅。両セル中心間の光学的厚さ
\(\sigma_{R,L}\Delta_L/2+\sigma_{R,R}\Delta_R/2\) を中心間距離 \(d_f\) で割ったもので、
拡散極限の直列抵抗として厳密。実装は算術平均に幅差の項を足す形
\(\tfrac12(\sigma_L+\sigma_R)+\tfrac12(\sigma_L-\sigma_R)(\Delta_L-\Delta_R)/(\Delta_L+\Delta_R)\)
で、等幅なら従来の \(\tfrac12(\sigma_{R,L}+\sigma_{R,R})\) と bit 一致。2026-09-23 まで
幅の重みなしの算術平均で、幅比 \(w\) の界面では拡散極限の
面の伝導度が最大 \((w+1)/2\) 倍ずれた）、
\(E_f=\max(\tfrac12(E_L+E_R),10^{-300})\)、
\(R_f=|E_R-E_L|/(d_f\,\sigma_{R,f}\,E_f)\)、
\(D_f=c\,\lambda(R_f)/\sigma_{R,f}\)。等幅の拡散極限（両側 \(\lambda=1/3\)）では旧
harmonic 形 \(\mathrm{harm}(c/3\sigma_L,\,c/3\sigma_R)=c/(3\cdot\tfrac12(\sigma_L+\sigma_R))\)
と厳密一致する。**2D_RZ は現行 cell-centered** \(D_c\)（cell 勾配で \(R_c\)）を
隣接 harmonic mean して \(D_f\) を作る — 1D の修正前と同型の front-stall 機構が
残存する既知課題（2026-07-26 カーネルレビュー指摘、face-centered 化は 2D 側の対応範囲）。

> **真空縮退の正則化（2026-08-28）**: \(\sigma_R\) は評価・組み立ての両方で
> `radiation.multigroup_diffusion.opacity_floor`（既定 **\(10^{-6}\)**）で floor する。単位の扱いは 2 か所で異なる：
> 組み立て（面の \(\sigma\)・限流子）では \(\sigma_R\ge\) floor [cm\(^{-1}\)]（mfp 10 km）、不透明度の評価（`opacity.cu`、
> 多材料の `multimat_opacity_1d.cuh`）では質量不透明度として \(\sigma\ge\rho\cdot\)floor（floor [cm\(^2\)/g]）。void セル（\(\sigma\to 0\)）かつ一様 \(E\)（\(\nabla E=0\)）の縮退方向では
> \(R=0\to\lambda=1/3\) となり limiter が \(D=c/(3\sigma)\) の発散を止められない。旧既定
> \(10^{-100}\) では tmat opacity が void 密度で underflow すると三重対角係数が
> \(\sim 10^{98}\) に達し、非 pivoting CR（`gtsv2StridedBatch`）も QR pivoting
> （`gtsv2`）も消去中の桁落ち増幅で NaN を生成した（550 セル zoning デッキの
> 決定論的セル反転クラッシュの真因; NaN 解は下流の floor クランプ
> \(\mathrm{fmax}(\mathrm{NaN},E_{floor})=E_{floor}\) で暗黙に床値化され step 800 まで潜伏）。
> 勾配領域では \(\lambda\sim 1/R\) により \(D_f=c\,d_f E_f/|\Delta E|\) と \(\sigma\) が
> 相殺されるため、床は縮退方向にのみ作用し、光学的厚い検証系（Su-Olson・Marshak、
> \(\sigma\ge 1\)）は bit 不変。

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

**1D 外側反復の灰色加速・流束制限子の評価・void セル（2026-09-24）**

- **灰色加速**（`outer_accel="grey"`、既定 `"auto"` は 1D でこれ、2D_RZ では `"none"`）:
  外側反復 \(k\) は \(T_k\) で Fleck 係数 \(f_g\) と不透明度を評価して各群を解き、\(f_g\) を固定して物質温度
  \(T^{k+1/2}\) を解く。固定点は \(f_g(T^*)\) を持つ。\(T_k\) で線形化し（不透明度固定、
  \(f=1/(1+z)\)、\(z\propto T^3\) より \(f'_g=-3f_g(1-f_g)/T\)）、
  \(S'_g=f_g c\sigma^{pe}_g B'_g + f'_g(c\sigma^{pe}_g B_g - c\sigma^{a}_g E^n_g)\)、
  \(D=\rho c_v/\Delta t+\sum_g S'_g\)、\(w_g=S'_g/D\)、\(P=\sum_g f'_g(c\sigma^{pe}_g B_g-c\sigma^a_g E^n_g)\)、
  \(\Delta T_k=T^{k+1/2}-T_k\) とすると、固定点への補正は
  \(A_g e_g-\Delta t V w_g\sum_h c\sigma^a_h e_h=\Delta t V w_g (D-P)\Delta T_k\)、
  \(T^*-T^{k+1/2}=(\sum_h c\sigma^a_h e_h-P\Delta T_k)/D\)（\(A_g\) は反復 \(k\) の群の三重対角行列）。
  \(e_g=\xi_g\varepsilon\)、\(\xi_g\propto w_g/(1+\Delta t c\sigma^a_g)\)（無限媒質の最遅モード、
  \(\sum_g c\sigma^a_g\xi_g=1\) に規格化）として群について足すと \(\varepsilon\) の三重対角方程式 1 本になり
  （Morel, Larsen & Matzen, JQSRT 34 (1985) 243 の多群-灰色合成加速）、次の反復は
  \(T^{k+1/2}+(\varepsilon-P\Delta T_k)/D\)（\([-T/2,+T]\) に制限、電子エネルギー・圧力は物質更新と同じ閉包）から始める。
  1 群では線形化した補正そのもの（Newton 段）。補正は \(\Delta T_k\) とともに消えるので、収束解は加速なしの
  反復と同じ。スペクトルと灰色行列は反復 \(k\) の組み立て直後（群の解法が行列を上書きしうるので解く前）に作り、
  右辺と補正は反復 \(k+1\) の最初に適用する（収束して抜ける反復の状態は加速前の生の出力のまま）。
  通常ループと常駐ループで同じ device 関数（`fld_1d_grey_accel.cuh`）を使う。灰色の三重対角方程式は
  並列循環縮約（PCR、`core::pcr_solve_strided`）で解く（2026-09-25。通常ループは 1 ブロック、行を共有メモリ
  に置けないときは大域メモリの作業領域、常駐ループはグリッド全体。各行の演算は同じで、両ループの解は丸め誤差の
  範囲で一致する — 積和の融合のされ方がカーネルごとに変わりうる）。以前の 1 スレッドの Thomas 法は FP64 の割り算の依存連鎖で、RTX 4090 の GXII FLD で
  0.27 ms/step（最初の 300 ステップ）・0.55 ms/step（1.85 ns から）かかっていた（PCR で 0.036・0.069 ms/step）。
  解の丸めは変わる（方程式は同じ）。
- **流束制限子の評価**（`limiter_evaluation`、既定 `"predictor"`）: 流束制限子の \(R\) を、最初の外側反復の
  放射場（step 開始時の場で制限子を評価し、step 開始時の温度の放射で各群を解いた結果）から評価して step 内は
  固定する。`"iterate"`（従来）は毎反復その時点の放射場から評価し直すが、この遅れた固定点反復は光学的に薄く
  流れが支配的な領域で停滞・振動し、GXII 回帰デッキの初期（〜0.2 ns）で 20 回の上限に達していた（残差最大 9e-3）。
  予測子方式と灰色加速の組み合わせで、同デッキの全 step が 2 次収束する（平均 2.6 回）。
- **void セル**: void セルは真空として吸収・放射をしない（\(\sigma^a=\sigma^{pe}=\eta=0\)、物質温度は交換で
  変わらない）。拡散係数の正則化のための Rosseland 不透明度の下限はそのまま。以前は不透明度の下限
  （\(\rho\kappa_{floor}\)）と定数不透明度の混合（非 void 材料を含まないセルが先頭材料の \(\kappa\) を取っていた）
  により、ターゲット外の下限密度セルがコロナの放射を吸って 1 回の放射計算の中で 100 eV 程度まで加熱され、
  外側反復の収束を妨げていた（GXII: 未収束 238 step、平均 13.6 回 → 修正後 85 step、4.0 回）。

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
詳細: 社内の設計メモ rgmg_verification_battery_20260711.md、社内の検証記録 §9.5.6。

As of L1b-2 the captured block also covers the z-line and RGMG preconditioner applications (stream-parameterized variants; capture failure still latches the eager loop, and the graph key bakes in every preconditioner buffer pointer so any hierarchy reallocation forces recapture).

`state.rad_E_old`
は device resident の backward-Euler 履歴で、各輻射の解の開始時に `rad_E` を写す（`copy_rad_E_to_old`。前の解の後に
`gamma_r_43` の hydro 半ステップの支払いが `rad_E` を変えるので、前の解の終端の値では古い）。

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

> **W-B 適合修正（2026-07-03）**: 1D `update_matter_kernel` は上の
> \(S^{used}\) 契約に違反して**未混合** \(c\sigma^{PE}B(T)\) を emission として
> 記帳していた（\(f_c\) を完全に無視）。放射側は混合源で解くため、
> \((1-f)c\sigma\Delta t\,|E^n-B|\) の幻エネルギーが毎 step 発生していた
> （`fld_1d_volume_source_balance` gate が検出、修正で balance 残差
> 7.9e-6 → 2.1e-10）。修正後の 1D kernel は Newton 残差・Jacobian・
> `rad_emit` 記帳の全てで \(f\,c\sigma^{PE}B(T)+(1-f)c\sigma^{PA}E^n\) を使う
> （**2026-09-14 訂正**: 再放射項 \((1-f)c\sigma E^n\) の \(\sigma\) は E 方程式（上式）と同じ吸収
> 不透明度 \(\sigma^{PA}\)。2026-07-03 の実装は物質側だけ \(\sigma^{PE}\) を使っており、\(\sigma^{PE}\ne\sigma^{PA}\)
> の TMAT 表では \((1-f)c(\sigma^{PE}-\sigma^{PA})E^n\Delta t\) が毎反復消える不整合だった — §2 の
> [2026-09-14 追補 2] 同日追加を参照。`update_matter_body`（`fld_1d_gpu.cu`）・`update_matter_body_persistent`
> （`fld_1d_bodies.cuh`）・2D_RZ `update_matter_kernel`（`fld_2d_rz_gpu.cu`）の Newton 残差と `rad_emit`
> 記帳を \(\sigma^{PA}\) に統一した。）
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
> 2026-09-14: 2D_RZ の物質 Newton も同じ \(\sigma^{PE}\) 混同を持っていたため同時に \(\sigma^{PA}\) へ統一
> （2D の `rad_emit` 記帳は元から assembly の source と同じ \(\sigma^{PA}\) だった）。

> **W-I AFI モード（2026-07-03）**: `Radiation.multigroup_diffusion.fleck_mode="afi"` は Fleck ブレンドを消費点（assembly の擬似散乱項 + 物質側ブレンド）で無効化し、outer 反復（Picard）が完全陰的 emission \(c\sigma B(T^{n+1})\) を収束させる。Larsen, Kumar & Morel (JCP 238, 2013) により AFI 離散化は任意 \(\Delta t\) で一意解・最大原理・平衡拡散極限を満たす。実測（GXII FLD nr200）: Fleck 既定は生産 \(\Delta t\)（コロナ z≈3）で吸収エネルギーを z→0 極限比 ~35% 抑制し dt 依存が全 metric を汚染、AFI は生産 dt で極限の数%以内（dt×4 でも残差数%）。コロナの Picard 縮小率 ~z/(1+z)≈0.75 のため `max_outer_iterations >= 40` 推奨（未収束は rate-limited warning が出る）。既定は従来 `"fleck_cummings"`（golden 影響なし）。**既定は Fleck を維持（ユーザー決定 2026-07-04）** — AFI は namelist opt-in の検証・測定モードとして存続し、GXII golden の再基準化は行わない。dt 感度の定量（Fleck ~25% vs AFI 4.1%）は 社内の検証記録 §10.1 に記録済み。

`Radiation.multigroup_diffusion.hydro_coupling` の既定は `"gamma_r_43"`（1D の FLD）。`"none"` は frozen-density historic behavior への
明示 opt-out。`"gamma_r_43"` は 1D deterministic FLD の Lagrangian hydro half step ごとに、
\(p_r=\sum_g E_g/3\) を流体の力に加え（force-side coupling）、その力が節点にした実際の仕事 \(W_r\)（運動量更新と同じ半ステップの
位置の面積・\(\bar u\)・マスク）を輻射場から払う：セルの輻射エネルギーを
\(\sum_gE_g^{new}V^{new}=\max(0,\ \sum_gE_g^{old}V^{old}-W_r)\) とし、群の比を保って配る（0 への切り上げは \(E_{floor}\) に記帳。
`rad_gamma_coupling_bodies.cuh`、2026-07-06 の v3 — 力と仕事の共役）。名前の由来の \(E_rV^{4/3}\) を厳密に保つ断熱更新（v1）は、
有限振幅（衝撃波・人工粘性）で力の仕事と一致せず（下の履歴）、使っていない。群間のドップラー移動は無い。
Default flipped 2026-07-06, reverted same day (v1 defect), RE-ADOPTED same night after the v3 fix and fresh A/B (R2-1=A). Scope enforcement (2026-07-06): activation additionally requires mode=="multigroup_diffusion" (SnTransport excluded). (History) DEFAULT REVERTED to "none" the same day: the rebaseline combined audit measured cumulative unexplained energy +8.6e9 erg (~10% of absorbed) on the GXII FLD regression with coupling on vs +0.6e9 off — the v1 force-side p_r work and the exact-adiabat V^{4/3} field payment do not cancel at finite amplitude (shocks/AV), a defect class invisible to the smooth-adiabat and linear-ceff gates. Deck opt-outs on compatible_energy decks stay as explicit documentation. Re-adoption path: v2 work-consistent payment (same p_r_half, same swept dV_c on BOTH modes — the design-doc v2 ruling extended to non-compatible mode, whose "v1 stays bit-for-bit" assumption this audit falsified).


[2026-09-08: mesh-motion conservation]
An explicit `hydro_coupling="conservative_advection"` is available for
1D Lagrangian FLD on the host-driven split loop (no persistent loop; the 1D ALE,
which it also excluded, was retired on 2026-10-02).
It solves the passive comoving transport subproblem
`D E_g / Dt = -E_g div(u)`, without radiation pressure force/work:
`U_g = E_g V` is carried across each accepted hydro half-step and
`E_g,new = E_g,old (V_old / V_new)` is published before the next radiation
solve. Each cell/group uses its own actual volumes. There is no symmetry
projection, smoothing, clipping, or redistribution, and unchanged volumes
leave the field bitwise unchanged. Invalid volumes or nonfinite fields fail.
Owned cells are updated and the existing 1D Allgatherv restores the replicated
radiation line; full-step retry snapshots already include `rad_E`.
The default and the existing `none`/`gamma_r_43` paths remain unchanged.

This option is a reduced passive-field model, not complete moving-medium
radiation hydrodynamics. With comoving radiation pressure enabled the grey
energy law instead contains `-P_r:grad(u)`; isotropic pressure gives
`D E / Dt = -(4/3) E div(u)`, and the gas must receive the equal/opposite
discrete force work. The existing gamma mode pays the actual nodal work,
`U_new = U_old - W_r`; its continuum adiabat is not an exact finite-step
identity. Frequency-shift group coupling and a general lab-frame ALE flux
are outside the new option.

Historical `none` freezes energy density during mesh motion:
`Delta U = sum_cg E_cg (V_new - V_old)`. For a closed spherical domain with
uniform field and linear expansion ratio s this produces `U_new/U_old=s^3`.
The existing `E_rad_mesh_advection` ledger measures that change, and
`epsilon_budget` subtracts it; the adjusted epsilon is NOT a physical
closed-system conservation test. With conservative advection that mesh term
is zero to roundoff. A physical audit must show the unadjusted balance and
actual external boundary fluxes separately. A lab-frame moving-control-volume
formulation would require flux `F_lab - w E_lab` at every face; simply
freezing a nonuniform cell field is not such a remap.

Tests: `test_radiation_mesh_motion` exercises expanding/contracting planar
and spherical ideal-gas domains, signed multigroup cell energies, stationary
cells, owned windows, and active closed-domain diffusion.
`examples/verification/radiation_mesh_advection.py` provides a standard
table-free example with reflecting radiation boundaries.

DSA/TSA 加速は使わない。1D の外側反復は既定で灰色加速（`outer_accel="auto"` → `"grey"`、上の「1D 外側反復の灰色加速」）、
`"anderson"` は下記、2D_RZ の既定は加速なし。
収束判定は
\(\max_c |\Delta T_{e,c}|/\max(T_{e,c},T_{floor}) <\)
`Radiation.multigroup_diffusion.outer_tol` である。

Anderson 加速（`outer_accel="anderson"`、`anderson_m`、`anderson_beta`; 2026-09-14 に 1D_SPH へ移植、
実装は `fld_anderson.cuh`）: 2D_RZ と同じ Walker–Ni 形の混合を 1D の非パイプライン外側ループに適用する。
各反復の入口で線形化温度 \(u_k=T_e\) を \(m+1\) 段の履歴環に写し、物質更新の出力 \(g_k\) から残差
\(f_k=g_k-u_k\) を作り、収束せず次の反復へ進むときだけ最近 \(p\le m\) 個の差分 \(\Delta u_j,\Delta f_j\) で
最小二乗（Tikhonov 正則化 \(10^{-12}\,\mathrm{tr}/p\)、Cholesky）した
\(u_{k+1}=u_k+\beta f_k-\sum_j\gamma_j(\Delta u_j+\beta\Delta f_j)\) を次の線形化温度にする
（`floors.Te` で床、非有限は生の Newton 出力、退化した最小二乗は混合なし）。収束判定は生の Newton 出力に
対する上式のままで、収束時の状態は混合しない。有効時は外側ループのパイプライン化を使わない。動機:
再放射項の統一と Kirchhoff 強制（§2 [2026-09-14 追補 2]）の後も、Planck 平均不透明度の急な温度依存で
\(f(T)\) が急変する冷たいセルは逐次代入で 2 周期に落ちうる（NIF DS デッキ 1572 サイクル中 1 サイクル、
セル 157 が 0.608↔0.641 eV、\(f=0.72\leftrightarrow0.92\)）。単体テスト `test_fld_anderson` は同一係数の
線形写像（縮小・非縮小）で Anderson(1) が 1 回の混合で不動点に達することを検査する。

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
  the 2D_RZ radiation does not feed radiation pressure into the matter momentum
  equation (only the 1D FLD with `hydro_coupling="gamma_r_43"` does, §0.5 and
  §1.1.2).
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
`tools/validation/radiative_shock_metrics.py` for the
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
final runtime-calibration production horizon
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

The existing LR07-EDA, LE08 LM_nED, and FLD-CED
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

The planar-reference calibration executes this planar reference as a mesh-robust 1D_SPH
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

The planar-reference calibration uses:
\[
\gamma=5/3,\quad A=1,\quad \bar Z=1,\quad
\rho_0=2.0\ \mathrm{g/cm^3},\quad T_0=30\ \mathrm{eV},\quad
\kappa_R=0.5\ \mathrm{cm^2/g},
\]
for \(M_0\in\{1.5,2.0,3.0\}\) with 1601 points per table. This preserves
\(\kappa_R\rho_0=1\ \mathrm{cm^{-1}}\) while reducing the omitted-radiation-force
perturbation. The relevant scaling is \(P_{rad}/P_{mat}\propto T^3/\rho\),
not \(T^4\), because \(P_{mat}\propto\rho T\). Relative to the earlier
\(T_0=80\) eV, \(\rho_0=1\ \mathrm{g/cm^3}\) references, the planar-reference calibration
\(T_0=30\) eV, \(\rho_0=2\ \mathrm{g/cm^3}\) tables reduce the reference
pressure ratio by \((30/80)^3/2\). The frozen-table diagnostics are:

| \(M_0\) | max \(P_{rad}/P_{mat}\) | max \(|\nabla P_{rad}|/|\nabla P_{mat}|\) | conservation residual |
|---:|---:|---:|---:|
| 1.5 | \(7.4621631\times10^{-7}\) | \(1.3226286\times10^{-6}\) | \(2.289371\times10^{-16}\) |
| 2.0 | \(2.1317967\times10^{-6}\) | \(3.7784912\times10^{-6}\) | \(2.836829\times10^{-16}\) |
| 3.0 | \(1.4341980\times10^{-5}\) | \(2.5420334\times10^{-5}\) | \(2.718628\times10^{-16}\) |

The production CTest uses the \(M_0=2.0\) table on \(N_r=64,128,256\) 1D_SPH
radial grids. The two-state shock-front localization protocol follows an external-AI review decision and
changes the default deck initialization from the smooth `reference_table`
profile to a `two_state` Riemann shock-formation protocol. In this mode the
discontinuity is placed at
\[
r_{shock}=\frac{r_{\min}+r_{\max}}{2},
\]
with the LR07-EDA upstream Rankine-Hugoniot state for \(r<r_{shock}\) and the
downstream state for \(r\ge r_{shock}\). The smooth `reference_table` path
remains available by explicit harness/deck override for diagnostics, but is no
longer the production default. The two-state protocol deck also lengthens the default
runtime to
\[
t_{end}=1.5\,\frac{x_{\max}-x_{\min}}{|u_{upstream}|},
\]
three times the earlier half-crossing default, so the two-state initial
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
two-state shock-front localization protocol, the blocking \(L_2\) values are averages over a
quasi-steady snapshot window rather than final-snapshot-only values. The window
starts at the first snapshot interval whose adjacent shock speeds differ by
less than 1% relative to the mean shock-speed magnitude; if no such interval is
found, the analyzer records the fallback method and the comparison reduces to
the final snapshot. The summary still emits final-snapshot and full-traversal
averages as non-gating diagnostics. It also runs a non-gating
\(0.25\,t_{end}\) diagnostic after the production ladder; if the strict
production \(L_2\) gates fail while this shorter diagnostic passes, the summary
flags the result as possible boundary/transient contamination.

Non-gating diagnostic instrumentation is added to distinguish
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

The diagnostic instrumentation also emits shock-centered profile dumps in \(r-r_s\),
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
acceptance regime. These diagnostics feed the branch-selection revision only; they do
not relax the Gate 4 \(T_e\) 5% threshold.

The branch-selection revision regenerates the LR07-EDA references using the
radiation-momentum-free reduction required by TENRYU v1.0 (§1.1.2; see also
`docs/validation/2d_rz/RH1/audit/lowrie_edwards_feasibility.md:15-33`).
The implementation follows the recorded design consensus:
the I1 table remains a low-\(P_{rad}\), equilibrium-diffusion energy reference,
but the steady momentum invariant is gas-only and no production radiation
momentum equation is introduced.

The same revision also bumps the frozen JSON table to schema version 2 with a
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

The escalation rule: if, after the branch-selection revision, the harness still reports
`max_abs_delta_Tr_interior > 0.05`, `max_abs_flux_resid_EDA > 1e-3`, and
`max_R_J/Pi/Phi > 0.05`, the next revision must first run the sensitivity sweep
\(2\times\kappa_R\), \(dt/2\), and \(N_R=512\) before reopening a 7B
Lowrie-Edwards switch. This rule is documentation-only in the branch-selection revision.

The later sensitivity-sweep revision reopens that switch after the sensitivity sweep on
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
\(P_{rad}\) calibration as the branch-selection revision:
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

Schema v3 extends the earlier schema-v2 dual-flux contract. It preserves
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

The LE08 sensitivity-sweep files are:

- `tools/validation/le08_ned_reference_generator.py`
- `tools/validation/check_le08_exactpack.py`
- `examples/verification/i1_le08_grey_ned_radshock.py`
- `tools/validation/run_i1_le08.py`
- `tests/verification/test_i1_le08_grey_ned_radshock.cu`
- `tests/verification/data/le08_ned_reference/le08_nED_M{1p5,2,3}.json`

For the sensitivity-sweep revision, LR07-EDA ctest #436 remains an auxiliary dual-flux schema
regression guard. Its generator, checker, harness, ctest, and frozen v2 tables
are intentionally left unchanged.

The LE08 deck-default update changes only the deck defaults and partial-checkpoint interpretation:
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
  two-state shock-front localization protocol; it does not relax the
  radiation-pressure or conservation gates.
- Gate 4: phase-aligned, interior-masked post-shock \(T_e\) relative
  \(L_2\le5\%\) against the LR07-EDA table, averaged over the quasi-steady
  window.
- Gate 5: phase-aligned, interior-masked pre-shock radiation precursor
  \(T_{rad}=(E_{rad}/a_{eV})^{1/4}\) relative \(L_2\le10\%\) against the same
  table, averaged over the quasi-steady window.

The in-house constant-Eddington FLD nED reference was added because
the sensitivity sweep and LE08 deck-default update isolated a model mismatch between ExactPack `problem="LM_nED"` and
TENRYU's grey FLD operator with `flux_limiter="none"`. The ExactPack LE08 table
retains its role as an auxiliary cross-check, while the FLD-CED table is the
equation-matched reference for TENRYU's current no-\(O(v/c)\), gas-momentum-only
model. The consensus record is CONSENSUS thread `019e25c4-...`; the
local design trace remains
`tmp/discussions/20260514-181253-wave6-6c-ext-vs-7b-le08-switch/log.md`.

The in-house constant-Eddington reference smooth-branch invariants are
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

The in-house constant-Eddington reference production ctest is
`expensive.I1 FLD const-Eddington nED grey radiative shock`. It keeps the
The LE08 deck-default update uses quasi-planar default \(r_{\min}=10^6\ \mathrm{cm}\), the
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
finite-grid error, but it is tight enough to reject the earlier \(O(1)\)
model-mismatch signature.

The final runtime calibration supersedes the earlier runtime calibration for the LE08 and
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

The final runtime calibration also extends the nED reference domains by padding only the far
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
by the final runtime calibration.

The final FLD-CED metric revision changes the production metric, not the TENRYU
discretization. The revised high-AI consultation
`tmp/prompts/20260515-105506-tenryu-fld-ced-shock-revised.md` supersedes
`tmp/prompts/20260515-101923-tenryu-fld-ced-shock-2x-gap.md` and closes Q3
(missing \(\rho C_v/\Delta t\) in the Newton residual), Q2 (Strang split), Q5
(2T drain), and Q8 (hidden Eddington mismatch) by code and empirical evidence.
The remaining FLD-CED `dTr_max` gap at NR=1024 is treated as the expected
finite-grid embedded-shock observable: finite shock-tube/reference-window
mismatch (Q7-extended), HLL/HLLC hydro shock smearing (Q4), post-shock MFP
under-resolution (Q1), and pointwise-vs-cell-average comparison at the
discontinuous hydro jump (Q6). The earlier NR=2048 run hit a
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

The final FLD-CED metric revision replaces that gate with a shock-windowed, cell-averaged,
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
`dTr_max` sequence. The revised harness requires the NR=256, 512, 1024 sequence
and fits the first-order embedded-shock model above using the bracketing
NR=256 and NR=1024 values, with the NR=512 point retained in the summary as
part of the sequence diagnostics. The strict bound is
\[
0.65\le D_\infty^{fit}\le 0.87 .
\]
For the existing NR=256/512/1024 dumps, the fitted value is
\(D_\infty^{fit}\approx 0.749\), consistent with the reference peak 0.7614 and
with the high-AI conclusion that the factor-of-two NR=1024 pointwise gap is a
first-order embedded-shock convergence effect rather than a FLD-CED equation
mismatch.

##### 2026-07-15 gate revision — Richardson \(D_\infty^{fit}\) retired to diagnostic; fine-pair peak-gap contraction gate (gate8b)

A two-bisect root-cause investigation of the 2026-07 gate8 failure
(campaign ledger the internal design note 2d_campaign_plan_20260708.md, exec-records
16–21) found that the strict \(D_\infty^{fit}\) window above is not an
asymptotic estimate on this ladder: adjacent-pair fits of the same
rational model disagree by more than a factor of two in every measured era
(certified-era pairs: (256,512) 0.20, (512,1024) −2.26 i.e. model-invalid,
(256,1024) 0.749), and the (256,1024) value amplifies percent-level
physics-consistency row changes by 2.5–3.0×. Two sanctioned physics
corrections (1D FLD Fleck conservation fixes, 2026-07-03; and the
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
0.075 (after the 2026-07-03 conservation fix), 0.207 (post-`gamma_r_43`) — green in every era with
≥2.4× margin; the coarse pair (256→512) is excluded because it rose (1.09×)
after the 2026-07-03 conservation fix (coarse-row peak detection is pre-asymptotic). A
finite-but-large \(G_{512}\) weakens only the contraction evidence, not the
absolute bound: gate 7 independently caps \(G_{1024}\le 0.15\).
\(D_\infty^{fit}\) remains computed and reported
(`richardson_d_infinity_fit_status = "diagnostic_only_20260715"`).

The NR=2048 "resolution envelope" note from the earlier and final metric revisions is refined by the same
investigation: a 2026-07-15 NR=2048 rerun under current physics completes
healthily with all fields finite (replica-identical, i.e. deterministic);
the historical `dTr_max` explosion there (89.65 in the earlier run, 65.9 today) is
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

An independent FLD substage audit is added, controlled by
`Radiation.multigroup_diffusion.diagnostic_radial_fourier_substage_enabled`.
When enabled, FLD records selected substage Fourier coefficients and solver
residual diagnostics under `/diagnostics/fld_substage_audit/v1/`. The HDF5
schema is additive: the group is absent unless the flag is true, root HDF5
`schema_version` is unchanged, and existing readers need no migration because
the optional group's absence is the default state. The audit is diagnostic-only
and does not feed back into the FLD solve or material update.

#### 6.7.4 §I2-aux Phase B: 1D_SPH FLD-CED Multigroup Grey-Collapse Limit

> **SUPERSEDED (2026-07-04 adjudication, Q7 of
> the internal design note i2_mgfld_collapse_spec.md v3.1).** The 1D_SPH aux gate
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
solution. The foundation is the final I1 FLD-CED infrastructure at
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
shock-windowed \(L_2\), cell-averaged convolved-reference peak gap, and
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
- The strict gates on the \(G=1\) grey baseline ladder remain unchanged:
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
the internal design note i2_mgfld_collapse_spec.md v3.1, adjudicated 2026-07-04).
Grey is the \(N_g=1\) operation of the same multigroup kernels (no separate
grey path; dispatch branches on dimension only), so the collapse comparison
verifies the group-partition machinery — b_g weights (renormalized
\(\sum_g b_g=1\) at table build, debug-asserted to 1e-12), per-group
\(\sigma/D/E\) storage, the Fleck cg layout contract (
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
  3f246289 (before the Fleck-layout correction) the gate fails loudly (mutation test).
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
  banner assert against the auto-grey trap). The revised absolute metrics vs
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

Design record: the internal design note i4_mm_rad_interface_spec.md (A1–A5 + Addenda).

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
the internal design note i6_sn_mg_collapse_spec.md (grounding facts F1-F15 + Addenda 1-2).

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
`tools/validation/run_i6_sn_mg_collapse.py`).** Binding (certification runs, batched):
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

Design record: the internal design note i7a_coupled_stack_spec.md (F1–F6 + Addenda 1–3).

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
per-step ledger eps ≤ 1e-6 (the row ε; conservation restored by the
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
restoration: the old text omitted the \(1/(1+\lambda^{PA})\)
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

#### 1D_SPH multi-material decks (2026-09-24)

A 1D_SPH deck with more than one non-void material evaluates every cell's
coefficients from its own materials (`radiation/multimat_opacity_1d.cuh`,
shared with the 1D FLD solver; the opacity models are those of the FLD
multi-material path: `constant`, `none`, `tmat`, `table_nlte`, `power_law`,
`freq_dep_marshak`):

- absorption and emission: \(\sigma^{PA}_g=\rho\sum_m w_m\kappa^{PA}_{m,g}(\rho_m,T_e)\)
  and the same for \(\sigma^{PE}_g\), with the mass fractions \(w_m\) (volume
  fractions without per-material masses) and the partial densities
  \(\rho_m=\rho w_m/f_m\) for the density-dependent models; for a constant
  material \(\kappa^{PA}=\kappa^{PE}=\) `kappa_a` (the single-material S_N
  reading). `Materials.opacity_mix_rule="max"` takes the largest material
  value; `"harmonic_mass_R"` changes only the Rosseland mean, which S_N does
  not use, so its absorption is the linear mix;
- physical scattering: \(\sigma_{s,g}=\rho\sum_m w_m\kappa_{s,m}\) with
  \(\kappa_{s,m}=\) `kappa_s` for the constant, power-law and
  frequency-dependent Marshak materials and 0 for the tables (as in the
  single-material paths), bounded by \(\rho\,[\)`opacity_floor`,
  `opacity_cap`\(]\) (no floor in a cell whose dominant material is a
  table);
- emission \(\eta_g=c\,\sigma^{PE}_g a_{eV}T_e^4 b_g(T_e)\);
- a cell whose dominant material (largest volume fraction) is a non-LTE
  table takes that material's NLTE coefficients at the cell density (the
  single-material NLTE launch restricted to those cells, the dominant-material
  approximation of the FLD solver); its scattering stays the mix above;
- the material Newton closes every cell with its dominant material's electron
  table (none for an exact ideal-gas material) and `cv_e_override`
  (`radiation::cell_electron_table_selector_1d`, shared with FLD).

A deck of two identical materials in pure cells repeats the single-material
step bitwise (`test_sn_1d_multimat`).

The linear-characteristic (LC) \(S_N\) scheme — the 2D_RZ scheme, and the
1D_SPH scheme before 2026-09-25 (still selectable with
`spatial_scheme="linear_characteristic"`) — uses the cell-local conservative
active-set closure. The namelist no longer exposes closure selectors; the
implementation is hardwired to conservative active set, signed face-flux
\(E^*\), donor-theta flux limiting, and AP face blending. The 1D_SPH default
since 2026-09-25 is the linear-discontinuous scheme (§6.8.4), which has none of
the \(E^*\) flux form, donor theta, AP blending, or void anchor. 2D_RZ uses the same
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
U_{hi}=U_e(T_e^n)+\sum_g\max(E^*_g,0)+S_{\Delta t},
\]
with \(S_{\Delta t}=\Delta t\,\dot S\) the 1D external volume-source energy
(zero without a source), since \(U_e+\sum_gE^+_g=U_e^n+\sum_gE^*_g+S_{\Delta t}\)
and \(E^+_g\ge0\). Before 2026-09-23 the bracket omitted \(S_{\Delta t}\): a
source-dominated cell (\(E^*\approx0\) on its first step) had its root above
the bracket, the twenty doublings of the expansion (starting from
\(\sum_g\max(E^*_g,0)\), or \(10^{-12}\max(|U_e^n|,1)\) when that is zero)
did not reach it, and the clamp to the bracket lost the source energy (23% of
the injection in the `test_sn_1d_su_olson` ledger case).
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

The 1D_SPH LC sweep tallies a signed face flux \(F_{f,g}\) on radial
faces, where positive flux is outward (increasing \(r\)). After each Picard
sweep the center face is forced to zero by spherical symmetry and the outer
vacuum face carries the sweep's discrete outgoing tally
\(F_{N+1/2,g}=\sum_{\mu_m>0}w_m\mu_m\psi_{m,g}^{out}\) (the 2026-07-19 note
below; the earlier overwrite with the Milne estimate
\(\tfrac12cE^{sweep}_{N-1,g}\) was twice the discrete half-range flux of a
near-isotropic field).
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

[2026-07-14] After the material Newton writeback (each Picard outer
iteration), transparent cells are re-anchored to the transport moments. With
\(\lambda^{ext}_{c,g}=c\Delta t\,(\sigma^{a}_{c,g}+\sigma^{s}_{c,g})\)
(`state.sn_sigma_a` + `state.sn_sigma_s` — total extinction; scattering included since
the 2026-07-19 total-extinction correction: an optically thick pure-scattering cell is in the diffusion regime
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
permanently freezes any transient imprint — the transparent-gap reproduction measured
\(E=1.887\,\phi/c\) frozen in the gap — while the swept moments are exact
there. The regimes sit 4+ orders apart in \(\lambda^{ext}\) (su_olson
\(\sim3\times10^{-2}\), the transparent-gap reproduction \(\sim10^{-8}\)), so
the anchor never activates in absorbing benchmarks. Gates:
`verify sn_1d_planar_transparent_gap` (contract C3) and the void-contract
case in `test_sn_streaming_limiter`.
Since 2026-07-26, every 1D anchor application is ledgered:
`state.sn_void_anchor_dE_step` / `sn_void_anchor_dE_abs_step` accumulate the
signed and absolute \(V\,\Delta E\) over the step (ledger-class scalars,
atomicAdd order within the documented host-ledger replica band; the anchored
field itself is bit-unchanged). A validation run that claims independent
\(S_N\)-reference status can require the abs sum to be exactly 0; a nonzero
sum is reported to the log (rate-limited). The 1D stagnation exit now also
sets the pre-existing `state.sn_outer_stagnated` flag (2D already did) and
logs the acceptance (the 2026-07-26 kernel review left the acceptance policy
unchanged; changing it is a user decision).

[2026-07-19] The 1D outer-boundary face-flux export and escape ledger
are now discretely consistent with the sweep in every regime. (i) The vacuum
branch of the boundary export previously overwrote the outer face flux with a
"Milne escape estimate" \(F_{out}=cE/2\) — exactly twice the discrete outgoing
half-range flux of a near-isotropic field (\(S^{+}\psi\simeq cE/4\) under the
\(\sum w=2\) GL convention). The pre-anchor flux-form E update consumed the
same overwritten value, so the pair was self-consistently wrong (the
transparent-limit E*-accumulation pathology recorded in the transparent-gap execution
log); once the void moment anchor set \(E=\phi/c\) from the moments, the 2x surplus broke
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
the anchor gated on total extinction (the total-extinction amendment above), every regime
has a single energy authority: anchored transport moments where
\(\lambda^{ext}\) is below the band, the flux-form/AP ledger elsewhere.
Gates: ctests `test_sn_face_flux_conservation`, `test_sn_ap_face_blend`
(conservation + FLD-match + gating cases), and the SN 1D family battery
(84/84 on the merged tree, 2026-07-19).

[2D port, 2026-07-20] Two of the four 1D mechanisms apply to the 2D_RZ
sweep and are ported. (iii) `sn_escaped_step` books the \(\theta\)-limited face
array (`sn_face_flux_limited`) integrated over the non-reflect boundary faces
with outward signs (\(+\) at the outer R face and top Z face, \(-\) at the
bottom Z face; the axis face is never booked), replacing the historic
coefficient model \(cE/2\) (vacuum) / \(cE/4\) (Marshak) that disagreed with
the discrete outgoing tally by up to 2x for a near-isotropic field. The scalar
feeds only the driver energy budget and diagnostics, so fields are bit-unchanged
by this rebooking. (iv) `anchor_void_rad_E_to_moments_2d_kernel` now gates on
the total extinction \(\lambda^{ext}\) exactly as the dimension-neutral void-anchor
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
(2026-07-26 doc restoration)**: `sn_sigma_s` holds the
PHYSICAL scattering opacity — constant, power-law and `freq_dep_marshak`
materials fill it with \(\rho\kappa_s\) (`kappa_s`, bounded by the S_N
opacity floor/cap like the constant evaluator), and the NLTE pure-SN path
fills it with the
Fleck-bypassed effective scattering \((1-f)\sigma^{PA}=0\) (f=1). It is NOT
the Rosseland mean. (Until 2026-09-23 the 1D power-law and
`freq_dep_marshak` paths stored the evaluator's second output there — the
power-law absorption itself and the Rosseland group mean — so those
materials scattered as much as, or more than, they absorbed; the 2D stage
still zeroes `sn_sigma_s` for the analytic models.) Consequently, in every production deck with
\(\kappa_s=0\) the \(\tau\) gate below evaluates \(\tau\approx0\) and the
blend weight is exactly \(\alpha=0\) on every face — the AP blend is inert
and the \(S_N\) answer is native transport (the sweep itself is unaffected:
its \(\sigma_s\) is the correct physical scattering). This matters for the
open \(S_N\)-vs-MGD bulk-compression comparison: those \(S_N\) results are
NOT FLD-contaminated through the blend. The blend machinery remains
unit-gated (`test_sn_ap_face_blend`) with synthetic opacities; wiring a true
Rosseland array into the AP gauge would ACTIVATE the blend in production and
is a user decision (escalated). The center face is not blended
(\(F^{blend}_{1/2,g}=F^{SN}_{1/2,g}=0\)). The outer face is blended one-sidedly
with the boundary cell on both sides of the gauge: the thick limit takes the
diffusion boundary flux, the streaming limit the discrete transport tally, and
the reduced-flux detector is left out there (the free-surface escape flux has
\(F/cE\approx0.25\) intrinsically) — only the optical-depth and equilibrium
factors decide (`sn_transport_1d_gpu.cu`, blend kernel).
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
loose gate for persistent ray-effect contamination. If no cell
satisfies the optically thick interior mask, the validation harness reports the
same statistic on the geometric interior as a fallback and records the mask
source.

The face-flux path applies a conservative donor-cell limiter before
Phase B, evaluated in the order of the flow (2026-09-25). For each cell and
group, with the inflow/outflow loads
\[
I_{c,g}=\Delta t\,
\frac{A_{c-1/2}\,\theta_{c-1,g}\max(F_{c-1/2,g},0)
      +A_{c+1/2}\,\theta_{c+1,g}\max(-F_{c+1/2,g},0)}{V_c},
\qquad
O_{c,g}=\Delta t\,
\frac{A_{c+1/2}\max(F_{c+1/2,g},0)
      +A_{c-1/2}\max(-F_{c-1/2,g},0)}{V_c},
\]
(domain-boundary inflow has no donor and enters with \(\theta=1\))
\[
\theta_{c,g}=
\begin{cases}
\min(1,(E^n_{c,g}+I_{c,g})/O_{c,g}), & O_{c,g}>0,\\
1, & O_{c,g}=0 .
\end{cases}
\]
The inflow is credited with the upstream donors' final \(\theta\). Every face
flux has one direction, so the inflow of a cell comes only from cells
upstream of it and no cell is upstream of itself: one pass from left to right
sets every cell whose left face does not carry flux out of it to the left
(its inflow comes from the left or it has none; a sink has no outflow), and a
pass from right to left sets the cells whose outflow goes left and whose
inflow comes from the right (`compute_streaming_theta_ordered_kernel`, one
thread per group). Then \(E^n_c V_c+\Delta t\,(\widehat{\text{in}}-\widehat{\text{out}})\ge0\)
in every cell, and a cell that passes on what it receives is not limited.
The former two-pass credit (2026-07-14) weighted the inflow with the donors'
first-pass factors \(\min(1,E^n/O)\) instead: when \(c\Delta t\gg\Delta x\) every
first-pass factor is small, so a chain of conduit cells stayed limited to
about twice its stored energy per step and fronts stalled. Measured on the
planar Marshak slab of `sn_1d_planar_marshak_equilibration` (0.4 cm,
\(\sigma=50\,\mathrm{cm^{-1}}\), 50 eV drive, \(\Delta t=10^{-10}\) s, \(c\Delta t=3\) cm), inner-cell
temperature after 2000 steps: on 32 cells it stayed near its 1 eV start
before (8 cells heated through) and reaches 39.66 eV now, 36.23 eV on 128
cells, against 35.94 eV of the linear-discontinuous scheme on
32 and 128 cells (§6.8.4); after 3000 steps 48.57 / 47.73 against 47.65
(`test_sn_streaming_limiter`). On the GXII S_N deck (300 cells, 0.15 ns) the
peak density went from 3.80 to 3.40 g/cm³ (linear-discontinuous 3.37) and
the radiation field energy at the deck's time step from 14 894 to 11 560 erg
(at \(\Delta t=5\times10^{-15}\) s: 11 253 erg; the former limiter still gave a peak
density of 3.81 there, \(c\Delta t=1.5\,\mu\mathrm{m}\) being wider than the shell's
cells).
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
\(\theta=\min(1,(E^n_{c,g}+I_{c,g})/O_{c,g})\) with the raw inflow; the flow-ordered
credit above is 1D-only (in 2D the flux directions can form cycles, and the
two-pass form below limits conduit cells like the former 1D form — not
measured in 2D, 2026-09-25). The outgoing-energy
sum includes all non-axis R/Z faces with the cylindrical face areas above. The donor is the upwind cell in the global face
orientation: lower \(i\) or lower \(j\) for positive R/Z flux, higher \(i\) or
higher \(j\) for negative R/Z flux. Incoming face power contributes to
\(I_{c,g}\), so boundary-injected energy can participate in same-step donor
availability. Boundary incoming flux is not donor-limited; boundary outgoing
flux uses the adjacent cell's \(\theta\). Axis faces
\(R_{i=0,j}\) are skipped in donor selection and outflow accounting and remain
zero in the limited flux.

**2D two-pass inflow credit (2026-07-17).** In 2D_RZ the
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

**Void moment anchor (2026-07-17).** After the 2D Picard outer
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
retired scheme. [2026-07-13/14] The retired scheme stored only the
scalar `state.rad_E_old` and reconstructed the old intensity isotropically
each step; in transparent regions that re-isotropization acts as an
artificial per-step scattering, so free-streaming fronts advanced
diffusively (depth \(\propto\sqrt t\), effective speed \(c/30\)–\(c/80\)
measured) while absorbing benchmarks (su_olson) were untouched — the
coverage hole that hid the defect. Regression gate:
`verify sn_1d_planar_transparent_gap`. The scalar flux is
\(\phi_g=\sum_n w_n\psi_{g,n}\) and \(E_g=\phi_g/c\).
[2026-09-23] Every solve starts by copying `rad_E` into `rad_E_old` (the FLD
rule); the copy used to happen only at step 0 and at the end of each solve,
so after a checkpoint restart `rad_E_old` was 0 and the first step solved
\(E^*=0-\Delta t\,\nabla\cdot F\), losing the radiation field. `sn_psi_prev`
is part of the full-step retry snapshot (restored with its size, so an
attempt that allocated it reseeds as the failed one did) and of checkpoints
(`radiation_sn/psi_prev`); a restart keeps it when its size matches
\(n_{cells}\,n_{groups}\,n_{angles}\) and reseeds isotropically otherwise
(older checkpoints), with a logged warning.

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

The 1D_SPH default since 2026-09-25 is the linear-discontinuous scheme
(§6.8.4). The constant-source linear-characteristic update
(`Radiation.sn_transport.spatial_scheme="linear_characteristic"`, the default
before that date and the scheme this paragraph and the following ones describe)
works with the
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
For SN Marshak source BC, a Z face \(f\) with boundary type
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
iteration skips the DSA correction branch and no DSA kernels are launched.
**The 1D correction is consistent with the sweep's discretization**
(2026-09-24). The correction \(f\) of the angular flux satisfies the swept
(backward-Euler, conservative-streaming) equations with the lagged
scattering residual \(\sigma_s(\phi^{l+1/2}-\phi^l)\) as source (Larsen 1982,
Eq. 4.12). With the P1 ansatz \(\psi_m=a\Phi+b\mu_mJ\) at the cell faces
(\(a=1/\sum w\), \(b=1/\sum w\mu^2\), so \(\sum w\psi=\Phi\), \(\sum
w\mu\psi=J\)), the zeroth and first angular moments of the discrete balance
of cell \(c\) are
\[
\begin{aligned}
&A_oJ_o-A_iJ_i+\left(\sigma_a+\tfrac{1}{c\Delta t}\right)V\Phi_c
 =\sigma_sV\left(\phi^{l+1/2}_c-\phi^l_c\right),\\
&k\,(A_o\Phi_o-A_i\Phi_i)-k\,(A_o-A_i)\,\Phi_c
 +\left(\sigma_t+\tfrac{1}{c\Delta t}\right)VJ_c=0,
\end{aligned}
\]
with \(k=a\sum w\mu^2\) (\(1/3\)) and \(A_f\), \(V\) the face areas and
volume of the mesh geometry. The first moment of the angular redistribution
is \(-k(A_o-A_i)\Phi_c/V\) for any quadrature, by the discrete cancellation of
the isotropic flux that the conservative streaming form guarantees; its
\(J\) part vanishes for the symmetric quadratures (Gauss–Legendre: exactly 0
to rounding). The cell moments \(\Phi_c=\sum w\psi_m\), \(J_c=\sum
w\mu\psi_m\) come from the sweep's own spatial closure
\(\psi_m=\theta\psi_{dn}+(1-\theta)\psi_{up}\) (\(\theta\) per cell, group and
angle: `precompute_lc_weights_kernel` for the linear-characteristic sweeps,
\(1/2\) for the serial diamond sweeps), which makes them linear in the four
face moments of the cell. Boundaries: \(J=0\) at the reflecting inner face
(centre, axis, symmetry plane); no incoming correction through the outer face,
\(a\Phi S_1=bJS_2\) with \(S_1=\sum_{\mu<0}w|\mu|\), \(S_2=\sum_{\mu<0}w\mu^2\)
(the Marshak incoming intensity is data, so the vacuum condition holds for the
correction). Ordered \((\Phi_0,J_0,\dots,\Phi_N,J_N)\) the rows (inner
condition, then per cell balance and first moment, then outer condition) form
a pentadiagonal system per group, solved for all groups at once by cuSPARSE
`cusparseDgpsvInterleavedBatch` (QR, no pivoting failure on the zero diagonal
of the inner row); the cell correction is \(\Phi_c\), added to the swept flux
with the non-negativity floor (`sn_dsa_1d_gpu.cu`).

**Why (measured 2026-09-24).** The former operator was a cell-centred
diffusion equation (\(D=1/(3(\sigma_t+1/(c\Delta t)))\), harmonic face values,
a \(\tfrac12cE\) outer leakage) that is not derived from the swept closure. A
line-by-line Python replica of `sn_sweep_spherical_lc_kernel` and of that
operator gave the spectral radius of the accelerated iteration 3.5 at
\(\sigma_t\Delta r=10.5\) (S8 and S16), 5.5 at 104 and 6.1 at 500 (divergent;
the GPU test at \(\sigma_s\Delta r=10.4\) stopped at 400 iterations with
residual \(\infty\)), 0.75 at 2.9, 0.15 at 0.3; with the consistent operator
0.21, 0.21, 0.22, 0.18 and 0.18, and at most 0.29 on geometric and random
meshes, material jumps (thick core / thin shell and the reverse, cell by cell
1000 / 1), near-void cells, steady state (\(\Delta t\to\infty\)), tiny cells at
the centre and S32. `test_sn_1d_scattering_slab_dsa` checks the GPU rows
against a host dense solve of the same system (three geometries, a varying
\(\theta\)), the convergence on cells of optical thickness 10 in the three
geometries (at most 30 iterations to \(10^{-8}\), against the unaccelerated
reference), and the Anderson comparison below.

**Anderson acceleration** (`Radiation.sn_transport.inner_acceleration=
"anderson"`, 1D_SPH, 2026-09-24): the DSA-corrected source iteration
\(\phi^{l+1}=G(\phi^l)\) (sweep, DSA correction, both with the
non-negativity of the sweep) is mixed with the last `anderson_depth`
\(m\le4\) residuals \(f_j=G(\phi^j)-\phi^j\) in the Walker–Ni form of the FLD
outer iteration (`fld_anderson.cuh`, \(\beta=1\), Tikhonov-regularised normal
equations, floor 0): \(\phi^{l+1}=\phi^l+f_l-\sum_j\gamma_j(\Delta\phi_j+\Delta
f_j)\). The history restarts with every outer iteration, the unrolled graph
path is not used, and the converged iterate is \(G(\phi)\) (no mix after the
last iteration). Anderson mixing needs only evaluations of \(G\), so it keeps
working with the sweep's non-negativity fix-ups, which make \(G\) nonlinear
and rule out a Krylov solver on the sweep operator. `test_sn_1d_scattering_slab_dsa`
compares the iteration counts and the converged fields of optically thick
scattering spheres.
The pentadiagonal systems have \(2(N_{cell}+1)\) rows per group, interleaved
by group; the cuSPARSE work buffer is sized before capture. In 2D_RZ the DSA
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
the DSA cuSPARSE work buffer and pentadiagonal scratch pointers. It is recaptured when those keys change,
which covers mesh reallocations, angle-count changes, namelist mode changes,
and timestep changes that alter kernel parameters. When only \(\Delta t\) changed
(buffers and sizes unchanged), the recaptured bodies update the instantiated
graph in place (`cudaGraphExecUpdate`, same topology) instead of a new
instantiation; any other key change, or a failed update, instantiates a new
graph (2026-09-24). The DSA cuSPARSE buffer is
allocated before capture; if graph capture or instantiation is unavailable, the
solver falls back to the same streamed kernel sequence.

Without scattering (every \(\sigma_s\) entry of the outer iteration exactly
zero) the sweep source does not depend on the previous iterate, so one body
(sweep, DSA step, state copy) is the transport solution for the outer
iteration's emission. The solver then runs exactly one and records one inner
iteration with residual 0, where the source iteration repeated the same sweep
\(K\) times and then reported the residual 0 (2026-09-24; the run log states it
once). The fixup tallies (radial and angular fixup counts and artificial
absorption), which accumulate over the sweeps of a step, then count each outer
iteration's sweep once.

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
(2026-07-26 doc truth restoration); \(db_g/dT\) is not
included. The
Picard residual is
\[
r_k=\max_c\frac{|T^{k+1}_{e,c}-T^{k}_{e,c}|}
{\max(|T^{k+1}_{e,c}|,10^{-300})}
\]
（分母の下限は温度の床ではなく \(10^{-300}\) — `sn_transport_1d_gpu.cu` の `kEnergyFloor`）。
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
`rad_dep[c,g]=c sigma_PA E^{n+1}_g V_c Delta t` and
`rad_emit[c,g]=eta_g V_c Delta t`, with
\(\eta_g=c\sigma^{PE}_g a_{eV}T^4b_g\) at the final \(T\) and
\(E^{n+1}_g\) the radiation energy the matter Newton writes back (the
Newton kernel reads the written-back array explicitly; its input and output
radiation arrays are the same array and are not `__restrict__`-qualified,
2026-09-23).

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
\(A(r)\) — the spherical fixed-point identity, geometry-independent.

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
taken from the product set. **DSA runs for cylindrical** with the
\(2\pi r\) faces and the product set's \(\mu\), \(w\) and \(\theta\) in the
consistent face-moment rows (2026-09-24; the cell-centred operator ran from
2026-09-23; it was force-disabled while the tridiagonal operator hard-coded
\(4\pi r^2\) faces; acceleration only, the converged answer is unchanged). `TENRYU_DEBUG_LANE_PARALLEL` builds reject cylindrical.

**Gates/tests**: `sn_1d_cylindrical_marshak_equilibration` (phase A uniform
blackbody fixed point on the FULL cylinder r0=0 including the axis cell —
measured drift 1.39e-16 = one ulp of \(a_{eV}T_r^4\); phase B cold-start
plateau, outer_rel 1.6e-7 / max_rel 9.6e-7 at 1800 steps, tolerances 1e-5 as
spherical/planar), ctest `test_sn_cyl_quadrature` (CPU invariants) and
`test_sn_1d_cylindrical_fixed_point` (LC+diamond × S8+S32, drift ≤ 1e-12,
\(\chi=1/3\)). Residuals: no analytic cylindrical transport benchmark in the
library yet (Lewis & Miller / PARTISN manuals in manual_queue); multigroup
cylindrical marshak (G=1 parity with the other geometries).

#### 6.8.4 1D linear-discontinuous scheme (`spatial_scheme="linear_discontinuous"`, the 1D default, 2026-09-25)

**Why.** With an emission or scattering source that is constant in each cell, a 1D spatial closure puts the wrong current into an optically thick, graded medium once \(\sigma\Delta r\gtrsim1\): for an emission \(\propto1+4r^2\) in a sphere (\(\sigma=40\), S8) the linear-characteristic scheme's current at \(r=0.5\) is 3.2× the diffusion current at \(\sigma\Delta r=4\) (1.24× at 1), and in a thick scattering sphere (\(\sigma_t=10\), \(c=1-10^{-4}\), \(\sigma\Delta r=10\)) its centre value is 81 % low (measured on replicas of the sweeps, 2026-09-25). Linear in-cell emission, scattering source and electron temperature, the temperature profile carried from step to step, restore the diffusion limit; rebuilding the in-cell profile from cell means at every step instead is several times less accurate. A Marshak wave (100 eV drive, \(\sigma=40\,\mathrm{cm^{-1}}\), \(C_v=5a T_b^3\), 200 steps of \(\Delta t=0.1/c\)) on 10 cells (\(\sigma\Delta r=4\)) against 160: max \(|T-T_{fine}|/T_b\) = 0.101 (sphere) / 0.068 (slab) with this scheme, 0.695 / 0.595 with the linear-characteristic scheme (`test_sn_1d_ld_step`); an independent Python implementation of the same discretization gives 0.101394 for the sphere, the GPU result to six digits. The linear-characteristic scheme's energy update (\(E^*=E^n-\Delta t\,\nabla\cdot F\) with the face fluxes limited so that no cell ends the step with negative radiation energy) throttled the transport through cells narrower than \(c\Delta t\) until its limiter credited the inflow in the order of the flow (2026-09-25, §6.8.2): a Marshak slab (0.4 cm, \(\sigma=50\,\mathrm{cm^{-1}}\), 50 eV drive, \(\Delta t=10^{-10}\) s) heated through on 8 cells but stayed cold inside on 32 and 128 cells; now its inner-cell temperature after 2000 steps converges with refinement to this scheme's (39.66 eV on 32 cells, 36.23 on 128, against 35.94 on 32 and 128 cells here; 49.52 on 8 cells).

**Discretization.** Per cell the basis \(b_L=(r_R-r)/h\), \(b_R=(r-r_L)/h\), lumped masses \(M_j=\int b_jA\,dr\), \(N_j=\int b_jA'\,dr\) (two-point Gauss, exact) with the face area \(A=4\pi r^2\), \(2\pi r\) (per unit length) or 1 (per unit area). For ordinate \(m\) of an angular chain (§6.8.1; §6.8.3 for the cylinder's levels; the whole Gauss–Legendre set for the sphere and the slab):
\[
\begin{aligned}
&\tfrac{\mu}{h}(M_L\psi_L+M_R\psi_R)-\mu A_L\hat\psi_L+\tfrac{N_L}{w_m}\big(\alpha_{m+\frac12}\psi_{m+\frac12,L}-\alpha_{m-\frac12}\psi_{m-\frac12,L}\big)+\big(\sigma_t+\tfrac1{c\Delta t}\big)M_L\psi_L=M_L\big(q_L+\tfrac{\psi^n_L}{c\Delta t}\big),\\
&-\tfrac{\mu}{h}(M_L\psi_L+M_R\psi_R)+\mu A_R\hat\psi_R+\tfrac{N_R}{w_m}\big(\alpha_{m+\frac12}\psi_{m+\frac12,R}-\alpha_{m-\frac12}\psi_{m-\frac12,R}\big)+\big(\sigma_t+\tfrac1{c\Delta t}\big)M_R\psi_R=M_R\big(q_R+\tfrac{\psi^n_R}{c\Delta t}\big),
\end{aligned}
\]
with the upwind face traces \(\hat\psi\) (the neighbour's node on the inflow side, the cell's own on the outflow side), the weighted diamond \(\psi_m=\tau_m\psi_{m+1/2}+(1-\tau_m)\psi_{m-1/2}\) at each node and the Carlson coefficients \(\alpha\) (zero at both ends of every chain; no angular term in the slab). Since \(N_L+N_R=A_R-A_L\) and the angular terms telescope, the weighted sum over nodes and ordinates is the exact cell balance; a uniform isotropic field satisfies every row exactly (\(V/h-A_L-N_L=0\)). Each chain starts from its starting direction, \(-s\,\partial_r\psi+(\sigma_t+1/c\Delta t)\psi=q+\psi^n_{sd}/c\Delta t\) along the diameter (\(s=1\) sphere, \(\sin\theta_\ell\) cylinder level), solved as a planar linear-discontinuous transport from the outer inflow with its own history. At the centre, the axis and the inner face of the slab the outward ordinate enters with its reflection partner's outflow. Outer face: vacuum or the Marshak inflow of §6.8 — a blackbody drive enters as \(\psi_{in}=cB_g/2\), a flux drive (`marshak.flux_erg_per_cm2_s`) as \(\psi_{in}=F_{inc}/\sum_{\mu<0}w|\mu|\) so that the discrete incoming current is the specified flux (the linear-characteristic scheme keeps \(2F_{inc}\)). The GPU sweep runs one block per group (2026-10-02): the chain's ordinates one after the other, each ordinate's recursion over the cells as a scan. A cell's outflow trace is affine in its inflow trace, \(\hat\psi_{out}=a\,\hat\psi_{in}+d\), with \(a\) and \(d\) from the cell's operands and the angular edge the previous ordinate left in the cell; each thread composes the maps of its cells (one cell per thread while the cells fit one block of the one-cell kernel, whose registers bound it: 512 cells in the CUDA 12.6 sm_89 build; several contiguous cells per thread beyond), the block scans the threads' maps in the sweep's order, and each thread solves its cells with the sequential sweep's arithmetic from the inflow the scan gives its first cell. The cells' inflows come from the composed maps, so the results differ from the sequential sweep's (one warp per group, a chain's half-set of ordinates as a diagonal wavefront over the cells; `TENRYU_SN_LD_SEQUENTIAL_SWEEP=1`) by rounding: on the GXII solid S\(_N\) deck to 2.5 ns its observables differ from the sequential sweep's by as much as the sequential sweep's own runs differ when the initial time step is changed in its 12th digit. A pure absorber is attenuated by \(1/(1+\tau+\tau^2/2)\) per cell of optical depth \(\tau\) along the ordinate (the linear-characteristic sweep: \(e^{-\tau}\)): slab, optical depth 20, S8, cells with \(\sigma\Delta x=2.5\) … 0.04 (8 … 256 cells), max relative error of the cell radiation energy where it exceeds \(10^{-3}\) of the drive 1.53, 0.81, 0.25, 0.076, 0.022, 0.0073; problems dominated by the uncollided attenuation of a beam through cells of optical depth \(\gtrsim1\) are more accurate with `"linear_characteristic"`.

**Matter.** Every cell carries two electron specific energies, \(e_L=\bar e-(M_R/V)\delta\), \(e_R=\bar e+(M_L/V)\delta\) (\(V=M_L+M_R\)): `ee` \(=\bar e\) is their lumped-mass mean and the offset \(\delta=e_R-e_L\) is carried from step to step (`State::sn_ee_node_offset`, checkpoint `radiation_sn/ee_node_offset` [erg/g]; zero at the first step, after an ALE remap and after a size change; reduced where a node would fall below the energy of the temperature floor). \(T=T(e)\) from the cell material's electron EOS (the table closures of the cell-average Newton, with its tail convention; the ideal-gas closure \(e=e_{ref}+c_v(T-T_{ref})\) about the step-start `ee`, `Te`). The nodal matter equation is lumped like the transport:
\[
\rho\,(e_j-e^n_j)=\Delta t\Big(A_j-\sum_g\varepsilon_{g,j}\Big),\qquad A_j=\sum_g\sigma_{a,g}\phi_{g,j},\qquad\varepsilon_{g,j}=c\,\sigma_{pe,g}B_g(T_j).
\]

**Iteration.** Newton on the nodal temperatures: the cell opacities are evaluated at \(T(\bar e_k)\), and about \(T_k\)
\[
\varepsilon_g=\underbrace{c\sigma_{pe,g}B_g(T_k)-\chi_g\,\frac{\Delta t\sum_hc\sigma_{pe,h}B_h(T_k)+\rho(e_k-e^n)}{D}}_{\text{fixed}_g}+\underbrace{\chi_g\frac{\Delta t}{D}}_{\kappa_g}A,\qquad\chi_g=c\sigma_{pe,g}B'_g(T_k),\quad D=\rho c_v(T_k)+\Delta t\sum_h\chi_h,
\]
\(B'_g=a\,(4T^3b_g+T^4\,db_g/dT)\) with \(db_g/dT\) the derivative of the Planck table's own interpolation (\(4b_g+T\,db_g/dT=\int_{group}xf\,e^x/(e^x-1)\,dx>0\) for the Planck density \(f\); a negative interpolated value is set to 0). The per-angle source is \((\text{fixed}_g+\kappa_gA+\sigma_{s,g}\phi_g+S\delta_{g0})/2\). The coupling \((I-K)A=b\) — \(K\) the transport of every group's \(\kappa_gA\) — is solved by GMRES (restart 30, classical Gram–Schmidt with one reorthogonalization — both passes on the device — deterministic reductions; the Hessenberg columns with their Givens rotations, the start and stop tests and the back substitution run on the device by one thread in the order of the host loop they replaced (2026-10-01; each product and sum rounded on its own, the rotation's \(\sqrt{a^2+b^2}\) correctly rounded but within about \(2^{-54}\) ulp of a midpoint, where the host called `std::hypot`, which in glibc 2.39 is one ulp off the correctly rounded value in about 0.1 % of random pairs), the host reading one decision per Krylov step; each node's residual scaled by \(s_j=\sum_g\sigma_{a,g}(|\phi_{ref}|+cB_g)\); converged when the residual has dropped by `inner_tol` from the Newton iteration's first one, at most `max_inner_iterations` Krylov steps), right-preconditioned by the grey low-order correction: spectral shape \(\xi_g\propto\kappa_g/(\sigma_{a,g}+1/c\Delta t)\), \(\sum\xi_g=1\); the consistent P1 system of the sweep (the P1 ansatz \(\psi=a\Phi+b\mu J\) at both nodes inserted in every ordinate's equations, zeroth and first angular moments per node: four unknowns per cell, block tridiagonal, block Thomas) with the transport cross section \(1/\sum_g\xi_g/(\sigma_{t,g}+1/c\Delta t)\), the removal \((1-\sum_g\kappa_g)\sum_g\xi_g\sigma_{a,g}+1/c\Delta t\) and the source \(\sum_g\kappa_g\,r\); the correction of \(A\) is \(\sum_g\xi_g\sigma_{a,g}\Phi\). The preconditioner is applied in a Newton iteration only where it pays for itself (`Radiation.sn_transport.grey_preconditioner`, default `"auto"`, 2026-09-25): with the local gain \(\gamma_j=\bar\kappa_j\bar\sigma_{a,j}/\mathrm{rem}_j\) of the correction at node \(j\) (\(\bar\kappa=\sum_g\kappa_g\), \(\bar\sigma_a=\sum_g\xi_g\sigma_{a,g}\), \(\mathrm{rem}\) the removal above) — in an infinite medium the correction maps a smooth absorption-rate error \(v\) to \((1+\gamma)v\), and the unpreconditioned coupling contracts it by \(\gamma/(1+\gamma)\) per Krylov step — `"auto"` applies it when \(\max_j\gamma_j>0.1\) (a contraction above 0.091 per step) and otherwise runs GMRES unpreconditioned, skipping the P1 assembly, factorization and solves; `"on"` always applies it, `"off"` never. The converged solution and the tolerances are the same; the Krylov iterates differ, so does the rounding of the solution. In the GXII S_N deck \(\max_j\gamma_j\) stays below 0.065 (below 0.005 over the first 2000 steps); the correction saved 0.3 Krylov steps per Newton iteration late in the run (restarted from step 12000, 300 steps) and none early, and skipping it shortens the step by 3.8 ms late (8.14 to 7.01 s) and by 2.3 ms over the first 2000 steps (25.8 to 21.2 s, RTX 4090); the histories of the 300 late steps agree to \(10^{-10}\) relative. `TENRYU_SN_LD_PRECOND_AUDIT=1` logs, per Newton iteration, the largest gain, whether the preconditioner ran, the Krylov steps and the first residual ratios (read-only). Physical scattering is converged inside every transport solve by source iteration with the same P1 system per group, to 0.1 `inner_tol` (\(c=1-10^{-4}\), \(\sigma\Delta r=0.3/3/30\): 16/13/6 iterations in the sphere and the cylinder, 15/10/6 in the slab, to \(10^{-11}\)). The next linearization point of each node is its own balance temperature with the transport's absorption held (\(\rho(e(T)-e^n)+\Delta t\sum_gc\sigma_{pe,g}B_g(T)=\Delta t\max(A,0)\), safeguarded Newton), moved by at most a factor 4 per iteration (a tangent taken far below the solution overshoots by orders of magnitude when the emission dominates the heat capacity: a 5 eV → 100 eV step, measured), and by a fraction \(\omega\) of the step that halves (down to 1/16) when a step reverses the previous one without shrinking below half of it (the cell opacities are held at the point; a table-opacity corona cell at 643 eV alternated by ±20 eV and did not converge in 20 iterations without it). The iteration stops when the nodal temperatures change by at most `outer_tol` (the effective value of §6.8) and GMRES has converged. On the GPU the linearization and the nodal matter update (with its balance temperature) run one warp per node, the groups' Planck terms in parallel over the lanes and the sums over the groups in the group order; the electron EOS inversions (a bisection in \(\ln T\)) run on a warp that evaluates the next five bisection levels' 31 midpoints in parallel and descends the tree with the sequential loop's decisions (the same result); the P1 systems (the preconditioner's, and with physical scattering one per group) are assembled and eliminated once per Newton iteration, each diagonal block (with the lower neighbour eliminated) inverted by Gauss–Jordan with partial pivoting and stored as its inverse, and applied to every right-hand side by forward and backward substitution with matrix-vector products; the sweep's cell matrices of every group, ordinate and starting direction are inverted once per Newton iteration (they depend on the cross sections and the mesh only), so a cell's solve in the sweep is its right-hand side (source, inflow trace, angular edge) times the inverse, and each lane loads its next cell's operands one diagonal ahead (2026-09-25; the inverses change the rounding: the GXII S_N deck's integrated quantities differ from the Cramer's-rule and LU-substitution forms by at most \(2\times10^{-5}\) relative over 1593 steps, the iteration counts are the same).

**Accepted state.** Each iteration's nodal energies follow from the transport's own absorption and the emission that entered its final sweep, \(e_j=e^n_j+\Delta t(A_j-\sum_g(\text{fixed}_g+\kappa_gA^*_j))/\rho\): radiation and matter exchange the same energy, so the step conserves energy for any iterate (to the scattering tolerance; measured step imbalance ≤ \(3\times10^{-15}\) relative). `ee` is the lumped-mass mean, `Te` \(=T(\)`ee`\()\), `Pe` from the EOS; `rad_E` \(=\sum_jM_j\phi_j/(cV)\) (`sn_phi_old` the same mean of \(\phi\)); `rad_dep`/`rad_emit` the absorbed/emitted energy of the step; the face flux is the transport's own current (no blending with a diffusion flux, no limiter, no void-cell energy anchor); the escape is booked gross as in §6.8. A step that did not converge, or that raised a node to the floor energy, requests the driver's retry through `sn_material_retry_flag` (4 Newton, 8 GMRES, 16 scattering, 1 floor). Void cells have no absorption, emission or scattering and keep their matter energy. \(\psi^n\) (every ordinate and node, `radiation_sn/psi_prev`) and the starting-direction histories (`radiation_sn/psi_sd_prev`) are rescaled at the step start per (cell, group) so that \(\sum_jM_j\phi^n_j=cE^nV\) with \(E^n\) the radiation energy the other operators left (compression, remap, a restart), or seeded isotropic and flat in the cell (first step, a size change, a remap, no usable history).

**Namelist.** A 1D deck that does not set `spatial_scheme` runs this scheme; `inner_acceleration="anderson"` is refused with it, and `dsa_enabled`, `inner_graph_unroll`, `outer_tol_stagnation_factor` apply to the linear-characteristic scheme only. No \(S_N\) scheme implements the per-group diffusion fallback: `diffusion_fallback_mode` accepts only `"none"` (since 2026-09-29; `"per_group_hysteresis"` used to be accepted and did nothing), and `tau_diffusion_on/off` are read and unused.

**Tests.** ctest `test_sn_1d_ld` (the GPU sweep and moments against a host implementation of the same equations in the three geometries; the P1-accelerated source iteration of thick scattering against a dense direct solution: error ≤ \(2.5\times10^{-12}\)) and `test_sn_1d_ld_step` (the Planck fraction derivative against central differences, worst relative difference \(2.2\times10^{-8}\); the equilibrium fixed point in every geometry, one and three groups, with scattering: change ≤ \(3\times10^{-15}\); the step energy balance with vacuum and Marshak boundaries, scattering, void cells and a flux drive; the Marshak waves above; the slab attenuation study above; six groups with the frequency-dependent opacity (\(\sigma\) from \(10^{8}\) to 26 cm⁻¹, 5 eV matter under a 150 eV drive): every step converged, at most 139 GMRES iterations in a step, run-to-run bitwise identical). Verification: `sn_1d_analytic_marshak`, `sn_1d_su_olson`, `sn_1d_marshak_equilibration` (sphere, slab, cylinder), `sn_1d_planar_transparent_gap`, `sn_1d_origin_symmetry` and `sn_1d_e_old_transient` run this scheme; `sn_1d_planar_slab_attenuation` (the exact attenuation) and `sn_1d_spherical_lathrop_two_region` set `"linear_characteristic"`. The slab equilibration's inner-slab temperature on 8 cells matches a 128-cell run to 0.05 % from 1000 steps on (the linear-characteristic 8-cell run leads it: 49.65 eV at 3000 steps against 47.67), so its step cap is 16000 (it reaches the plateau more slowly than the linear-characteristic run it was set for).

---
