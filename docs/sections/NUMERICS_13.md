<!-- 分割元: docs/NUMERICS.md | このファイルは参照用です。原本（docs/NUMERICS.md）が権威です。 -->
## 13. 2D RZ I1-B ハイブリッド停滞アーキテクチャ（成層 1D core サブモデル）

ICF 級収束 (C≥7.2) の停滞計量のため、中心擬似コア (§9 の macro CV) を
**成層 1D 球対称 Lagrangian サブモデル**に拡張する (2026-07-02/03、
env `TENRYU_I1B_CORE_1D_SUBMODEL`、default off)。極性 butterfly シェル
メッシュは深収束の shell を保持できないことが計装付きで反証済み
(equal-μ 極 rezone / 行平滑 conv_rezone / Laplacian interior patch /
TMOP 型品質目的 patch rezone / foot+main 整形パルス — いずれも
ring 4-7 の flow 駆動フォールドに不追随) — 認証済みハイブリッドが
production 構成である。

### 13.1 サブモデル本体
- スキーム: staggered von Neumann–Richtmyer 球対称 (§3 と同族)。
  セル {m_k, e_k, Y_k}、節点 {r_i, u_i}、r_0=0 固定。人工粘性は
  c1·ρ·c_s·|Δu| + c2·ρ·Δu² (c1/c2 は 2D 側と同値既定 0.5/4.0)。
  CFL=0.25 で 2D step 内を subcycle。host 常駐 (GPU コスト零)。
- **massless outer face**: 外端フェイスは質量ゼロの運動学拘束
  (V_c 追随)。外殻セル全質量は内側節点へ lump する。質量つき face の
  速度上書きは ±7e4 erg/step 級の slosh 整流注入 (実測)、純力学 face は
  2D 境界から係留喪失 (実測 dV −7e-3 cm³ 発散) — massless が唯一
  健全。
- 2D への返り圧: `macro_core_pressure` はサブモデル外面の P+q を返す。

### 13.2 吸収 (handoff) 契約
- **GASFRONT 球対称スケジュール** (`TENRYU_I1B_SPHERICAL_ABSORB_*`):
  最外 gas unit の圧力が誕生値の PJUMP 倍に跳ねた瞬間 (=衝撃の
  cushion 進入、フォールド前・物質界面・準球状態 ε_R≈0.02-0.04) に
  gas prefix 全体を吸収する。failure 駆動吸収は折れた面を渡すため
  ε_R=0.3-0.7 となり球サブモデルと不整合 (実測)。
- **per-ring append + 半径射影**: 吸収 unit ごとに 1 殻 (radial 順)。
  到着運動エネルギーは K_r=p_r²/2M のみ運動として保持し、残差
  (角方向/分散分) は比内部エネルギーへ**明示的に熱化** (消失させない)。
  巨大 append は `TENRYU_I1B_CORE_1D_SPLIT_APPEND` で等質量サブ殻に
  分割 (pooled スラブは内部衝撃構造を持てず被圧縮体を過駆動: 実測
  +59%)。分割数系列 16/32/64 の Richardson 外挿で結合系統差 −9% を
  特性化 (50/60 Mbar で再現)。
- 運動量着地: 旧 face 節点が新殻質量の所有者となり、保存的
  質量加重マージで受ける。

### 13.3 エネルギー簿記 (単一所有 pair 契約)
- 界面仕事は両側が同一の Π·ΔV_c を記帳する (impulse 簿記は正確に
  半分を計上していた: 実測 W_2d=W_1d/2)。サブモデル内部は
  U+K vs (injected + piston work) の台帳で ≤~2.5e-4 (通常 1e-5 級)。
  1 step = 1 advance (rollback+再前進対は K を漏らす: 実測)。
- 診断: 吸収ガス部分体積はサブモデル殻から直読 (`pc.core1d_V_gas_c`
  → CR_V)。プール系 fallback はエネルギー分率 `U_gas_frac_c`
  (質量分率は混合吸収で計量を不連続化: 実測 1.6→14.7 jump)。
- 境界非球性診断: 境界ループ物理弧 (軸閉鎖除外) の Legendre ℓ≤4
  分解 (`TENRYU_I1B_CORE1D_ASPH_EVERY`、handoff 前後は常時)。

### 13.4 終端吸収と core1d 単独 tail (I1-B-R endgame)
- **rebound 期の壁の実測機構**: 吸収 walk が shell 行を消費し尽くすと
  生存 2D shell は最終 1 行のみとなり、内側行 (macro 境界) と外側行
  (物理境界) の全節点が契約 pin 済み = 修復自由度ゼロ (予測計量 TMOP は
  q_J=−0.85 の予測反転を正しく検出するが free node 0)。最終行は
  構造バックストップ (structural_max) で吸収不可のため ladder 枯渇
  abort が必然だった。
- **終端吸収** (`TENRYU_I1B_TERMINAL_ABSORB`, default off): emergency
  walk が最終行 (structural_max+1) を要求し、かつ rebound 検出
  (`pc.core1d_V_gas_c > factor × min 履歴`、factor 既定 1.02、env
  `TENRYU_I1B_TERMINAL_REBOUND_FACTOR`) の時のみ、残存全セルを
  §13.2 と同一契約 (unit depth ごと 1 殻 + 半径射影 + 明示熱化) で
  サブモデルへ吸収し、2D メッシュを凍結する。以後 t_end まで
  **core1d 単独 tail** (chunk 幅 `TENRYU_I1B_TERMINAL_TAIL_DT` 既定
  1 ps、外圧 = drive テーブル直読、rebound 期は 0) を積分する。
  2D は再ステップされない。凍結後の HDF5 最終出力は t_abs 時点の
  メッシュを保持する (時刻ラベルのみ t_end)。
- **free-outer face**: tail では 2D piston が存在しないため、外端は
  標準の質量つき VNR 自由面へ切替 (face 質量 = 外殻セルの半分、
  駆動力 = (P+q)_face − P_ext)。massless→massive の規約切替に伴う
  K 定義ジャンプは ledger の injected 項へ**明示記帳** (閉包厳密)。
  従来の実験的 dynamic_outer は外殻セル慣性を 1.5 倍計上する不整合
  (不安定の有力要因) があり、tail では使用しない。

### 13.5 remap 質量閉包ゲート
- 全 CSR remap は総質量閉包 (Σm_post−Σm_pre)/Σm_pre を無条件計測する
  (`AleRemap2DRZResult::mass_closure_rel` + step 内最悪値集約)。違反時は
  offender セル forensics (csr_closure_ledger) を常時出力。
- `TENRYU_I1B_REMAP_CLOSURE_REJECT_TOL` (>0 で有効、現状 opt-in) 設定時、
  違反 step は driver の full-step retry snapshot で棄却され、rezone
  発動系 (center-patch / per-block Winslow / 定期 axis fire) は
  `TENRYU_I1B_REZONE_CLOSURE_COOLDOWN_STEPS` (既定 50) の間停止する
  (retry は Lagrangian-only で進行)。強制修復 route の remap は直接
  検査され、汚染修復はその場で破棄される。
- 動機 (実測): center-patch Winslow fire 1 発 (invalid active-node
  updates 保持) が +1.119e-1 の総質量を捏造 (gate 系譜全 run で bit
  同値)。ゲート有効で同事象は棄却され、全走行 dM_raw が roundoff
  (3.6e-15) に回復、clean-mass gate 再測定で CR_V peak 7.689 ≥ 7.2 を
  維持 (汚染時 7.66、系統差 −9.8%→−9.4%)。root cause (Winslow
  invalid-retained 更新の受理設計) は独立残課題。

### 13.6 逸脱と登録済み残課題
- 駆動カプセル AV プロファイル c1=0.5/c2=4.0 (deck 既定 0.1/1.5 から
  の逸脱、強駆動衝撃安定化。回帰パッケージは未整備=残課題)。
- 残課題: 結合系統差 −9.8% の除去 (per-ring 到着忠実度) / env 面の
  namelist 昇格 / rebound 期 2D 分解能 (排気の角構造) は放棄を明示
  — 旧 I1-B-R #601 (Eulerian/AMR 中央 patch) は独立 capability
  milestone のまま。
  gate 証拠: docs/validation/2d_rz/I1/i1b_tier2_gate_evidence_20260703.md。

## 14. 核燃焼カーネル（nuclear burn、1D_SPH v1）
（merge train 註 2026-07-18: 統合実施 — 本 §14 採番を採用し、1d 側 §13 と内容照合の上で一本化済み。）

`Burn.enabled=True`（既定 False、SPECIFICATION §6.4.11）で有効化。設計の一次記録は
`docs/design/burn_kernel_1d_v1_design_20260710.md`（W0–W5 の測定・裁定履歴込み）。
実装は `src/burn/`（reactivity / network / deposition / partition / burn_stage）＋
`coupling/driver.cpp` の burn callback。既定 OFF は bit 恒等（§14.6）。

### 14.1 反応チャネルと反応率

v1 チャネル：DT = T(d,n)⁴He、DD 両分岐 = D(d,p)T / D(d,n)³He、D³He = ³He(d,p)⁴He
（既定 OFF）。T+T は対象外（BH fit 不在、レートは DT 比 ~10⁻² 以下）。
反応率は Bosch & Hale 1992（NF 32, 611）Eq. 12–14 の Padé パラメタ化
（Table VII 係数を `burn_constants.hpp` に凍結転写、Table VIII 8 温度×4 反応
アンカーで rel ≤ 1.5e-3 を単体ゲート G-R0 が常時検証）：
\[
\langle\sigma v\rangle = C_1\,\theta\sqrt{\xi/(m_rc^2\,T^3)}\,e^{-3\xi},\quad
\xi=(B_G^2/4\theta)^{1/3}
\]
T は **イオン温度 [keV]**（TENRYU 内部 eV → kernel 入口で 1 回だけ 1e-3 倍。
keV/eV 混同は G-R0 が桁で検出する設計）。fit 床（DT/DD 0.2 keV、D³He 0.5 keV、
**両端含む**）未満は rate=0、天井（100/190 keV）超は天井値へクランプ（記録済み仕様）。
体積反応率は \(r = n_i n_j \langle\sigma v\rangle/(1+\delta_{ij})\)（DD は 1/2）。

**遮蔽補正（v2、`Burn.screening`、既定 "none" = 係数 1.0 恒等）**：反応対ごとに
\(\langle\sigma v\rangle_{scr} = F_k\,\langle\sigma v\rangle\)、\(F_k=e^{h_k}\ge1\)。
`"salpeter"` = Salpeter 1954 弱遮蔽（電子込み 2T Debye:
\(\lambda^{-2}=4\pi e^2[\sum_s n_sZ_s^2/k_BT_i + n_e/k_BT_e]\)、
\(h=Z_iZ_je^2/(\lambda k_BT_i)\)；非縮退電子 θ_e=1、有効域 h≪1）。
`"chugunov_dewitt"` = CD 2009 (PRC 80, 014611) Appendix A4 補間
（イオン遮蔽・剛体電子背景；弱結合で Debye-Hückel A1、強結合で本文 fit へ —
ICF 燃焼域は Γ_e~0.01-0.14 の弱結合で A4 枝が operative）。両モデルの弱極限比は
解析関係 \(h_S/h_{CD}\to\sqrt{(\langle Z^2\rangle+\langle Z\rangle)/\langle Z^2\rangle}\)
（DT で √2 — 電子遮蔽の有無の設計差、ゲートはこの関係を検証する）。混合モーメント
⟨Z⟩,⟨Z²⟩ はセルの burn 種在庫（ash 込み）から。ICF 帯の大きさ:
F_CD = 1.002 (10 g/cc, 3 keV) 〜 1.08 (10³ g/cc, 1 keV)。設計・凍結参照値は
`docs/design/burn_kernel_v2_20260710.md` §B。

> **ガード（2026-07-26, AI review k14 §4.5）**: Salpeter は非有限/非正の入力
> （T_i, T_e, n_e）で全反応 F=1 に落として one-shot WARNING、指数は
> \(h \le h_{max}=2\) にクランプ（弱遮蔽模型の有効域外 — \(e^2\simeq7.4\) 倍で頭打ち、
> 超過は one-shot WARNING）。2T Debye 形は Salpeter 1954 の平衡理論の
> **TENRYU 独自 2T 拡張**であり published equilibrium result ではない（§4.2 指摘の明示）。

### 14.2 種ネットワークと Lagrangian 比在庫

種は D, T, ³He, ⁴He, p の 5 種＋中性子（台帳のみ、自由飛行逃逸）。**在庫は比在庫
\(Y_s = n_s/\rho\) [1/g] で保持**する — 連続の式 ∂n/∂t = −n∇·v + (network) の希釈項は
Lagrangian セルでは Y_s 不変性に吸収され、レート評価時に \(n_s = Y_s\rho\) を毎ステップ
再構成する。密度凍結格納は膨張セルで燃料対上限を桁破りする（W5 実測 22,800×、
gate G0 f_r ≤ 1 が常設番人）。将来の ALE/remap 結合は Y_s の質量保存 remap が前提条件。

ステップ内は温度・密度凍結の per-cell 常微分方程式を RK2（explicit midpoint）で
subcycle：\(M=\mathrm{clamp}(\lceil dt\,\max_s q_s/\max(n_s, 10^{-9}n_{tot})/\varepsilon_{dep}\rceil, 1, M_{max})\)、
\(q_s = \max(q_s^+, q_s^-)\)（**総生成と総消費の大きい方** — 2026-07-26 修正,
AI review k14 B-2: 旧実装は net \(|\dot n_s|\) を使っており、DD-bred T が DT 消費と
釣り合うセルで短い turnover が不可視だった。総量制御への修正で純消費燃料
（pure-DT deck）は bit 不変）。
制御対象は**有効チャネルの反応物種すべて（成長含む）** — 微量 bred-T（DD→DT 連鎖）の
分解能欠落は G-R1c（scipy LSODA rtol 1e-12 凍結参照との 3 checkpoint 照合 rel ≤ 1e-6）が
検出した実障害モードで、消費種限定制御は棄却済み。gross-turnover 回帰は
G-R1d（R_DDp≒R_DT の人工均衡で required substeps が飽和すること）。
\(M_{req} > M_{max}\) の飽和は黙認しない（k14 B-3）: 必要数を報告し、
\(0.9\,dt\,M_{max}/M_{req}\) を burn dt 制限（state.burn_dt_limit_s、次ステップ制御）
へ畳み込み、rate-limited WARNING を出す。current-step retry 化は driver
transaction 拡張が必要で escalate 済み（同 dt 内の当該ステップは M_max で受理される
— 精度契約は次ステップ縮小で回復する設計）。正値性は決定論的 scale-back
（θ = min n_s/(−Δn_s)、counts を先にスケールし在庫は counts から再構成 — 台帳一次主義）
＋ sub-ulp ゼロクランプ。反応 counts は RK2 と FP 同一の積で蓄積し、在庫変化との
化学量論恒等は数 ulp 帯で成立（G-R1a は counts↔He4 在庫の FP 恒等も検証）。

### 14.3 荷電粒子沈着（scheme="fraley"）

α range は Fraley 1974 fit 3d × 電子項 Coulomb-log 密度補正：
\[
\rho\lambda_\alpha(T_e,\rho) = \frac{1.5\times10^{-2}\,T_e^{5/4}}{1+8.2\times10^{-3}\,T_e^{5/4}}
\cdot\frac{1+0.17\ln T_e}{1+0.17\ln(T_e\sqrt{\rho_0/\rho})},\quad \rho_0=0.213
\]
（T_e keV、g/cm²。δ(ρ)=1→3 for solid→**絶対密度** 10⁴ g/cm³ を再現、G-K0）。
非 α 種は電子 drag 域スケーリング \(\lambda_s=\lambda_\alpha\sqrt{m_sE_s/m_\alpha E_\alpha}(Z_\alpha/Z_s)^2\)。

幾何は**点源球核**（一様媒質・直線飛行・v 線形電子 drag の前荷重減速
\(E(s)=E_0(1-s/\lambda)^2\)）：出生半径比 u=r/R_b、τ=R_b ρ̄/ρλ に対する閉形式
（asinh 1 個の初等関数、G-K1 で凍結求積参照 39 点 abs ≤ 1e-11）。体積平均は古典二分枝
\[
f(\tau)=\tfrac{3}{2}\tau-\tfrac{4}{5}\tau^2\ (\tau\le\tfrac12),\qquad
1-\tfrac{1}{4\tau}+\tfrac{1}{160\tau^3}\ (\tau\ge\tfrac12)
\]
に一致（W0 で独立導出・記号/数値検証、G-K2）。燃料域は volFrac 閾値の単一区間、
R_b = 最外燃料セル外縁、ρ̄ は外向き radial 台形 column の平均密度（一様球で厳密に ρ）。
保持分 f_pt を出生セルへ沈着、残余は荷電逃逸台帳へ（**escape = released − dep の FP 構成
＝台帳恒等が構造的**、G4 実測 ≤3e-15）。斜め chord の成層は v1 近似（設計 doc §4.4）。

### 14.4 電子/イオン分配

既定 `partition="li_petrasso"`：LP 1993（PRL 70, 3059）一般化 dE/dx
（大角散乱 1/lnΛ 補正＋x>1 集団項、量子 p_min、電子 Debye 遮蔽、lnΛ 床 2、
u²=v_t²+v_f²）を初期化時に減速積分し、
**(log T_e × log T_i × log n_e) 64×16×16 表 × 生成物 slot** に凍結
（runtime Python 不使用；slot 毎に std::async 並列 build、書込み範囲が互いに素なので
bitwise 決定的）。**field 温度は種別**（2026-07-26 修正, AI review k14 B-1/§5.1）:
電子 field の熱速度は T_e、D/T/³He イオン field は T_i（\(v_f^2=2T_f/m_f\)）。
旧実装は全 field に T_e を渡しており、\(T_e\ne T_i\) の hot-spot 形成期に
イオン stopping と e/i 分配が系統的に誤っていた。Debye 長は電子（T_e）のまま。
lookup は clamped trilinear。G-P1 = LP Table I（{6,19,32,47,64}% @ {1,5,10,20,40} keV、
T_i=T_e 対角で評価）±3 点＋Python prototype ±1 点の転写忠実帯。
残存既知事項（escalate 済み）: 積分下限 \(E_{min}=\max(1.5k_BT_e,10^{-3}E_0)\) 未満の
残差は全体平均で扱う（review §5.3）、背景組成は初期 x_D/x_T/x_He3 凍結（§5.2）。`partition="fraley"`（Eq. 4:
\(f_i=1/(1+32/T_e[\mathrm{keV}])\)、DT-α 限定、validation 強制）は rung-2 用 knob。
両者の ~8 点差（10 keV）は実物理差（LP p.3061）であり一致はむしろ危険信号（G-P2）。

### 14.5 結合・台帳・dt

演算子槽は laser 直後・radiation 前（sequential: H-C-L-**B**-R；Strang: L(dt)-**B(dt)**-
H/2-C-R-H/2、burn は全 dt 陽的源で分割しない）。ステージは host 実行（in-flight mirror
方式は inject_laser_source_terms 前例踏襲；perf 最適化は将来の別 PR）。沈着は
\(e_e{+}\!=dE_e/(\rho V)\), \(e_i{+}\!=dE_i/(\rho V)\) 後に Te/Ti/Pe/Pi を再閉包
（table EOS / cv_override / ideal の全分岐、1T は合算を e_e へ）。核融合エネルギーは
静止質量起源の**外部源**として budget の source 側 `E_burn_in` に登録（逃逸荷電/中性子は
流体に入らないので sink ではない）。W5 実測：burn 活性 6 run すべてで
epsilon_budget ≤ 6e-16。dt 制限は hot-electron 意味論
\(dt \le f_E\,e_{cell}/P_{dep}\)（lineage "burn"）＋ eps_deplete
（subcycle 飽和時は \(0.9\,dt\,M_{max}/M_{req}\) を同じ burn dt 制限へ畳み込む —
§14.2、2026-07-26）。決定論：セル独立
＋固定順縮約（bit 再現、host-device は FMA 差により rel 1e-13 ゲート）。

### 14.6 契約・制約（v1）

- 既定 OFF bit 恒等：W5 A/B（base=b07ac0c3）で field 53/53・history 164/164 bitwise、
  frozen_config 差分は additive な burn block のみ。GXII golden rel=0×6。
- persistent path は拒否（`warn_unsupported_once("burn")`）。1D_SPH+球面限定
  （validation）。HDF5 は additive（hydro/burn_*、time_state/E_burn_*）で
  kSchemaVersion 不変。checkpoint restart は burn_n_* 必須（欠損 hard error）。
- 非目標（設計 doc §1、v2 完了分を注記）：MC α は v2-D（§14.9）、中性子 in-flight 加熱は v2-E（2 線群 first-collision、SPECIFICATION §6.4 Burn.neutron_heating — 1D 専用、2D は fail-closed）で実装済み。EOS 組成 feedback
  （燃焼率 ≪1 近似、MULTI-IFE 同型）、2D、megakernel。

### 14.7 多群荷電粒子拡散（v2、`Burn.scheme="diffusion"`）

Corman-Loewe-Cooper-Winslow 1975 (NF 15, 377) の忠実実装。生成物 slot 6 種を
共有 log エネルギー格子（`diffusion_groups` 群、`diffusion_E_min_keV`〜15.5 MeV）
で追跡:
\[
\partial_t N_g = \nabla\cdot(D_g\nabla N_g) - N_g/\tau_g + N_{g+1}/\tau_{g+1} + S_g,\quad
\tau_g = t_E\tfrac{2}{3}\ln\frac{\gamma t_E+E_{g+1}^{3/2}}{\gamma t_E+E_g^{3/2}}
\]
D_g は flux-limited（加算型 limiter + Post-Wilson \(|\bar\mu|^{-1}=1+3e^{-(\lambda/2)|\nabla N/N-3.6/r|}\)、
前ステップ N で準線形化）。群カスケードは g_max→1 の逐次陰解、群毎に球面 r² FV
三重対角を cusparseDgtsv2StridedBatch（cached handle + pooled buffer、FLD 様式）で解く。
境界: 中心 reflect、外面 Milne 逃逸（1/L = 1/(0.71λ)+1/r_J、逃逸流は荷電逃逸台帳へ）。
係数 t_E（電子 drag、v≪v_te 極限の標準形）・γ（イオン drag）・λ=2vt_D（90° 偏向）は
明示 NRL 型 Coulomb log で毎セル毎ステップ評価（論文 intro の絶対値 anchor は未印字
log 処方を含むため転写対象から棄却 — log-free 恒等式 e/i∝E^{3/2}・λ∝E² と 0-D 解析
減速極限 \(E(t)=[(E_0^{3/2}+\gamma t_E)e^{-3t/2t_E}-\gamma t_E]^{2/3}\) が gate）。
**イオン Coulomb log と γ は (群, セル) 毎**に群中心エネルギーで評価する
（2026-07-26 修正, AI review k14 §6.4 — 旧実装は出生エネルギーで 1 回評価し
全群へ流用しており、最接近距離の E 依存が終端域で欠落していた）。
t_E 非正/非有限のセルは γ が有限なら純イオン drag で減速を継続
（\(\tau=(2/3)(E_{g+1}^{3/2}-E_g^{3/2})/\gamma\)、分配は全イオン — §6.5 修正;
旧実装は sink ごと消していた）。

簿記はカスケード転送構成で厳密: 出生は隣接 2 群へ数+エネルギー両保存 binning
（出生エネルギーが最上位群**中心**を超える超過分 `top_excess` は電子へ即時沈着 —
既定格子で D³He 14.663 MeV proton は 704 keV=4.80% が該当。2026-07-26 から
one-shot WARNING で定量報告する。格子再設計（product-aligned grid）は escalate 済み,
AI review k14 B-5）、
転送 1 粒子毎に (Ē_{g+1}−Ē_g) を沈着（**e/i 分配は群内のエネルギー重み付き積分
\(f_i=\frac{1}{\Delta E}\int S_i/F\,dE\)** — 2026-07-26 修正, AI review k14 B-4/§6.2:
旧実装は滞在時間重み \(\int(S_i/F^2)/\int(1/F)\) で、粗い群の e/i crossover 帯で
構造的に別の積分だった。本 scheme の分配は内在で `Burn.partition` は不使用）、
g=1 退場は Ē₁ を全イオンへ（熱化）。**在庫は比スペクトル Y_g = N_g/ρ [1/g] で持続**（§14.2 と同じ Lagrangian
希釈対策 — 密度持続は膨張系で台帳を 11% 破った実測記録あり、設計 doc §C.3）。
飛行中エネルギー E_inflight が新台帳項（released = dep + esc + ΔE_inflight、
実測 2.9e-15；ε_budget は流体側 dep のみ計上で 4.5e-16 恒常）。checkpoint は
hydro/burn_Ng_slot{0..5} [1/g] + time_state/E_burn_inflight（additive）。
cross-scheme 帯: 3 keV/ρ10/ρR0.2 で deposited fraction 比 diffusion/fraley =
0.746（採択帯 [0.60,0.90] — 直線点核 vs 拡散+Milne 逃逸+スペクトル拡散の模型差）。

### 14.8 中性子スペクトル合成診断（v2、read-only）

Brysk 1973 の二 Maxwell 平均モーメント。燃焼重み付き ⟨T_i⟩_burn・⟨v_r²⟩_burn
（DT / DDn 別、固定順セル和）から:
平均シフト \(\langle E_n\rangle - \tfrac{m_\alpha}{m_n+m_\alpha}Q =
\tfrac{m_n}{m_D+m_T}\tfrac{3}{2}\theta + \tfrac{m_\alpha}{m_n+m_\alpha}\langle K\rangle\)
（⟨K⟩ = 3T_reac−(3/2)θ、T_reac は Brysk Table 1 転写（Reac 列 = ⟨E⟩/3 と解読、
両公表 anchor 35 keV/336 keV·33/157 keV を実装前検算で再現）、log-T 補間・[1,100] keV clamp）、
熱幅 σ² = 2m_nθ⟨E_n⟩/(m_n+m_partner)、全幅は 4π 平均の流体広がり
σ_fluid² = 2m_nE_{n0}⟨v_r²⟩/3 を加算（球対称 1D の合成検出器は方向平均 —
一次モーメントは対称消失、視線スペクトルは v3/Crilly-Munro scope として設計 doc 記録）。
history `burn/neutron_{Ti_burn,mean_shift,sigma_thermal,sigma_total}_{dt,dd}`
（burn 有効 run のみ、エネルギー簿記への影響ゼロ）。

### 14.9 MC α 輸送（v2、`Burn.scheme="mc"`、統計モード）

Yuan-Moses-McKenty 2005 型の直線 CSDA Monte Carlo（1D 球面特化、角散乱なし —
偏向 λ は拡散 scheme のみ）。**停止能係数は §14.7 と同一**（corman_tE/γ 共有 —
scheme 間一致 gate が模型恒等性の検証になる）。イオン Coulomb log は
**粒子の現在エネルギー**で毎セグメント評価（2026-07-26 修正, AI review k14 B-7 —
旧実装は出生エネルギーで凍結、Bragg-peak 近傍の γ を誤っていた。RNG 消費は不変）。粒子 (r, μ, E, w, slot) は
ステップ間持続 pool（時間依存近似）、出生は殻内 r³ 一様 + 等方 μ、
**RNG は Philox / curand_init(seed^global_id, subsequence=step, offset) —
NUMERICS §12.7.1 凍結契約**（global_id = (cell·6+slot)·N_mc+sample）。
CSDA 沈着は局所瞬時レート比で e/i 分割、熱化 E≤E_min → イオン、逃逸 → 台帳。
tally は atomicAdd（統計モード — bitwise 非適用、§0.3 MC 条項が適用）。
per-particle 簿記により台帳恒等 released = dep+esc+ΔE_inflight は RNG に
依らず厳密（実測 8.5e-15、ε_budget 5.6e-16）。
**三 scheme 整合（3 keV/ρ10/ρR0.2 実測）**: deposited fraction
fraley 0.968 / mc 0.928 / diffusion 0.722 — mc（参照級）に対し fraley は
その解析近似（+4%）、diffusion は Milne 逃逸+スペクトル拡散で低め、と
物理的序列どおり。CV gate: 同 seed 5 run CV ≤ 1e-3（atomic 順序帯、§0.3 文言）
+ 異 seed 5 run CV ≤ 5%（統計収束、1/√N 傾向は PERFORMANCE 記帳）。

### 14.10 2D_RZ port（scheme="local"|"diffusion"、2026-07-11）

\`Main.dimension="2D_RZ"\` で Burn.enabled=True が有効（設計記録は
\`docs/design/2d_burn_port_spec.md\`、実装は src/burn/burn_stage_2d +
corman_diffusion_2d + driver 2D 配線）。1D との差分のみ記す：

- **scheme 行列**: 2D は \`"local"\`（全量出生セル沈着、LP/fraley 分配）と
  \`"diffusion"\`（§14.7 の Corman を 2D RZ FV へ一般化）のみ。\`"fraley"\` は
  点源球核が 1D_SPH 固有のため 2D では ConfigError、\`"mc"\` は未移植で同様。
  namelist キーは 1D と完全共有（新キーなし）。
- **種輸送と ALE remap（C-REMAP 契約）**: Y_s [1/g]（cell-major
  [n_cells×5]、host 主体）は構造格子 swept remap 本体
  （ale_remap_2d_rz_kernel / apply_hydro_face_flux）で質量 flux と同一の
  sign·dm に donor 風上で随伴（clamp なし・二次 remap 時も勾配再構成なし）。
  境界 z-flux は流出のみ随伴（供給流入は組成ゼロ）。実測：uniform-Y は
  ~80 remap events で 2.9e-15 保存・種総数 drift 0.0 厳密・blob 保存 0.0
  厳密。診断上の注意：burn dataset は最初の post-step snapshot から出現
  （burn_enabled_any latch）。
- **拒否行列（fail-closed）**: burn+ALE は conservative_remap_enabled ∧
  single_block のみ許可。拒否 = ¬conservative_remap / per_material_conservation
  / total_energy_remap_2d_rz / axis_band_managed_remap / multiblock /
  hllc_z_flux_2d_rz / force_rezone_every_n_steps>0 / reference_barrier。
- **Corman 2D**（scheme="diffusion"）: 体積重み対称 SPD 5 点 FV（面積・中心
  距離は FLD 幾何 helper の clone）。面 D は §14.7 の limiter を面入力の算術
  平均（N/ρ/lnΛ_I）で評価 — FLD のセル中心 D+調和平均とは意図的に別規約
  （RZ↔1D 球対称還元性を優先）。Post-Wilson 幾何項は 3.6/R
  （R=√(r²+z²)、面法線対数微分の ê_R 射影）で原点中心球に厳密還元。境界：
  axis/reflect=零 flux、free=面毎 Milne 1/L=1/(0.71λ)+1/R_face（同一の
  extensive 沈み込み係数を assembly 対角と逃逸 tally で共有 — 台帳恒等は
  構造的）。ソルバ：burn 自前 Jacobi-CG（5 点 stencil 直接 matvec、固定形状
  二段 reduce、warm start、rel tol 1e-10（rhs 規格）、cap 500 で fail-closed
  TENRYU_ASSERT）。飛行中スペクトル Y_g [1/g]（6 slot×G 群）は remap 時に
  ρ で N へスケール→既存 radiation plane kernel 再利用→post-remap ρ で復元
  （dm·Y_donor と代数恒等）。実測台帳：非 ALE 4.3e-16、diffusion×ALE×活性
  燃焼複合 1.38e-13、閉箱 esc_charged 0.0 厳密、10 keV で dep_e/dep_i≈5.5。
- **2T/per-material 沈着**: dE_e/dE_i の沈着は inject_burn_source_terms
  （1D 移植）+ per_material_conservation 有効時は FLD と同一の質量比配分
  （fld_2d_rz_gpu.cu の BUG-16 規約、max(...,0) clamp、Te/Ti_per_material
  キャッシュ無効化）。
- **retry 整合**: DriverRetrySnapshot が burn 在庫・累積台帳・Y_g を
  capture/restore（STRANG では burn が hydro half より先に走るため必須）。
  1D 側は同 snapshot に burn 未収載の継承ハザードあり（merge train で解消）。
- **既知の残余（v1）**: host 主体 Y と per-step mirror の perf 繰延、種は
  cell-level（per-material 種分解は v2）、Post-Wilson/Milne の非原点中心
  問題はヒューリスティック帯、Brysk 中性子診断キーは 2D では 0.0 のまま
  （スペクトル合成は未移植）。
