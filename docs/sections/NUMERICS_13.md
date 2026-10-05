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
The pooled cap is an endgame device; with
`central_pseudo_core_activation_time_s > 0`, the center evolves fully resolved
until that time and the cap pools from the resolved state using the same
operation as a swap-time build, bounding the pooled-approximation exposure
window.

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
- positivity guard が trial 節点交差を検出した場合は dt を半減し、各 retry
  で guard 前の速度から momentum kick を再計算する。受理された単一の
  final dt を速度更新、位置更新、compatible cell work、piston work の全てに
  用いる。
- **Energy admissibility guard (2026-08-24, 相談 #16 §6)**: supported/free
  経路では、候補 dt ごとに compatible な \(\Delta U_j\) の trial を節点交差
  チェックと同じ guard ループ内で評価し、いずれかのセルで
  \(e_j+\Delta U_j/m_j \le 64\,\epsilon_{mach}\max(|e_j|, \tfrac12\max(u_j^2,u_{j+1}^2), c_{s,j}^2)\)
  なら state を install せず dt を半減して再試行する (棄却 trial の
  \(\Delta U\) は受理時と式恒等のベクトルから commit — 非発火 step は
  bit 不変、gate_cap22 で end-to-end 実証)。dt が \(10^{-22}\) s を割れば
  `energy_floor` として substep を拒否し診断 dump を出す。**無記帳の
  エネルギー床は存在しない** (旧 \(e\ge10^{-30}\) clamp は O(1) 閉合残差の
  隠蔽源だった)。legacy dynamic_outer 経路 (study-only) は従来の clamp の
  まま対象外。

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
- 2D pooled overlay の界面仕事は §9 の global nodal-KE conjugate impulse
  \(-\sum_k\alpha_k\mathbf I_k\cdot\mathbf u_k^\sharp\) を一度だけ記帳する。
  core1d 側の piston ledger は実際の subcycle kick/motion と同じ final dt
  を使う独立の内部閉包診断であり、2D booking を \(-\Pi\Delta V_c\) で
  上書きしない。
- **Compatible total-energy update (W4d-7):** with
  \(\sigma_j=P_j+q_j\), pre-step face area \(A_i=4\pi r_i^2\), and nodal
  kick midpoint velocity \(\bar u_i=(u_i^-+u_i^+)/2\), each cell receives
  \[
  \Delta U_j=\Delta t\,\sigma_j
  \left(A_j\bar u_j-A_{j+1}\bar u_{j+1}\right).
  \]
  The interior face terms telescope exactly against the nodal kinetic-energy
  update. At the massive free outer face the remaining boundary work is
  \(W_{ext}=-\Delta t\,P_{ext}A_n\bar u_n\). At the coupled massless face,
  the prescribed velocity is the face velocity and the remaining flux is
  \(W_{face}=-\Delta t\,\sigma_{n-1}A_n u_{bc}\), using the same pre-step
  area family. The previous swept-volume internal-energy form had the classic
  O(1) shock-crossing energy defect: the champion reflected-shock tail measured
  a +9.5% of \(U\) closure residual invariant under substep and chunk
  refinement. The compatible attribution removes that non-convergent defect;
  \(q\) is included only through \(\sigma\), with no separate heating term.
- サブモデル内部は U+K vs (injected + piston work) の台帳で compatible
  substep の丸め誤差まで閉じる (反射衝撃の tail で実測した相対閉合残差
  −1.9e-14)。
  1 step = 1 advance (rollback+再前進対は K を漏らす: 実測)。
- 診断: 吸収ガス部分体積はサブモデル殻から直読 (`pc.core1d_V_gas_c`
  → CR_V)。プール系 fallback はエネルギー分率 `U_gas_frac_c`
  (質量分率は混合吸収で計量を不連続化: 実測 1.6→14.7 jump)。
- 境界非球性診断: 境界ループ物理弧 (軸閉鎖除外) の Legendre ℓ≤4
  分解 (`TENRYU_I1B_CORE1D_ASPH_EVERY`、handoff 前後は常時)。

### 13.4 終端吸収と core1d 単独 tail (I1-B-R terminal absorption)
- **rebound 期の壁の実測機構**: 吸収 walk が shell 行を消費し尽くすと
  生存 2D shell は最終 1 行のみとなり、内側行 (macro 境界) と外側行
  (物理境界) の全節点が契約 pin 済み = 修復自由度ゼロ (予測計量 TMOP は
  q_J=−0.85 の予測反転を正しく検出するが free node 0)。最終行は
  構造バックストップ (structural_max) で吸収不可のため ladder 枯渇
  abort が必然だった。
- **終端 takeover の廃止**: 2D メッシュを凍結して core1d tail へ移譲する
  終端吸収機構は廃止された
  (社内の設計メモ terminal_takeover_removal_20260827.md)。

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
