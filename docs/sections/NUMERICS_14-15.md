<!-- 分割元: docs/NUMERICS.md | このファイルは参照用です。原本（docs/NUMERICS.md）が権威です。 -->
## 14. 核燃焼カーネル（nuclear burn、1D_SPH v1）
（merge train 註 2026-07-18: 統合実施 — 本 §14 採番を採用し、1d 側 §13 と内容照合の上で一本化済み。）

`Burn.enabled=True`（既定 False、SPECIFICATION §6.4.11）で有効化。設計の一次記録は
社内の設計メモ burn_kernel_1d_v1_design_20260710.md（W0–W5 の測定・裁定履歴込み）。
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
社内の設計メモ burn_kernel_v2_20260710.md §B。

> **ガード（2026-07-26）**: 遮蔽は非正または NaN の入力（T_i, T_e, n_e）では全反応 F=1 に落とす（警告なし — 入口の
> 判定で遮蔽の計算そのものを飛ばす）。+∞ の入力は Salpeter の中で F=1 に落として one-shot WARNING。指数は
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
subcycle：\(M=\mathrm{clamp}(\lceil dt\,\max_s q_s/\max(n_s, 10^{-9}n_{tot}, 1\,\mathrm{cm^{-3}})/\varepsilon_{dep}\rceil, 1, M_{max})\)
（分母の最後の下限 1 cm\(^{-3}\) は実装の `network.cuh`）、
\(q_s = \max(q_s^+, q_s^-)\)（**総生成と総消費の大きい方** — 2026-07-26 修正:
旧実装は net \(|\dot n_s|\) を使っており、DD-bred T が DT 消費と
釣り合うセルで短い turnover が不可視だった。総量制御への修正で純消費燃料
（pure-DT deck）は bit 不変）。
制御対象は**有効チャネルの反応物種すべて（成長含む）** — 微量 bred-T（DD→DT 連鎖）の
分解能欠落は G-R1c（scipy LSODA rtol 1e-12 凍結参照との 3 checkpoint 照合 rel ≤ 1e-6）が
検出した実障害モードで、消費種限定制御は棄却済み。gross-turnover 回帰は
G-R1d（R_DDp≒R_DT の人工均衡で required substeps が飽和すること）。
\(M_{req} > M_{max}\) の飽和は黙認しない（2026-07-26 カーネルレビュー指摘）: 必要数を報告し、
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
（T_e keV、g/cm²。δ(ρ)=1→3 for solid→**絶対密度** 10⁴ g/cm³ を再現、G-K0）。実装の下限：\(T_e\) を 0.1 keV で、\(\rho\) を
\(10^{-12}\) g/cm³ で下から切り、Coulomb-log の密度補正の分子・分母の因子 \(1+0.17\ln(\cdot)\) をそれぞれ 0.15 で下から切る
（`alpha_rho_lambda`）。
非 α 種は電子 drag 域スケーリング \(\lambda_s=\lambda_\alpha\sqrt{m_sE_s/m_\alpha E_\alpha}(Z_\alpha/Z_s)^2\)。

**媒質の組成**（2026-09-23 是正 — 旧実装は組成によらず等モル DT の飛程を使っていた）:
fit 3d は等モル DT の値なので、\(1/(\rho\lambda)=S_e+S_i\)（電子項
\(S_e=1/(1.5\times10^{-2}T_e^{5/4})\)、イオン項 \(S_i=8.2\times10^{-3}/1.5\times10^{-2}\)）に分け、
単位質量あたりの電子数と \(\sum_j Z_j^2/A_j\) の等モル DT 比
\(f_e=(\bar Z/\bar A)/(\bar Z/\bar A)_{DT}\)、\(f_i=(\overline{Z^2/A}/\bar A)/(\overline{Z^2/A}/\bar A)_{DT}\)
（上線はイオンあたりの平均）で \(f_eS_e+f_iS_i\) とする。Coulomb-log 密度補正の ρ は DT 換算の
電子密度に当たる \(f_e\rho\) で置く。組成はステップ開始時のセルの場イオン（§14.7 と同じ定義）。
等モル DT では \(f_e=f_i=1\) が厳密に成り立ち旧式とビット同一、灰（⁴He・p）が溜まると 1 からずれる。
純 D は \(f_e=1.249\)・\(f_i=1.497\)（10 keV・固体密度で飛程 0.79 倍）。

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
**(log T_e × log T_i × log n_e) 64×16×16 表 × 10 slot**（荷電生成物 6 種と、中性子の弾性反跳 4 種 — 14 MeV・2.45 MeV の
中性子が D・T を反跳させた核、§14.11）に凍結
（runtime Python 不使用；(slot, \(T_e\) の行) を 1 つの仕事として worker の pool が並列に作る。書込み範囲が互いに素なので
bitwise 決定的。driver は時間ループと並行して表を作り、worker は他が使わない CPU 時間だけを使う）。**field 温度は種別**（2026-07-26 修正）:
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
H/2-C-R-H/2、burn は全 dt 陽的源で分割しない）。ステージは既定で GPU（`compute_burn_step_1d_device_stage`、
`burn_stage_gpu.cu`）で実行し、環境変数 `TENRYU_BURN_HOST_STAGE=1` のときだけ host の実装を使う。1D では段を device 常駐で実行する（`compute_burn_step_1d_resident`、局所沈着 2026-10-02、拡散・MC の誕生源 2026-10-02）: 入力は State の device の場、燃料在庫・\(\epsilon_{cum}\)・中性子数・診断（反応率・\(Q_e, Q_i\)）は device の写し（`State::burn_*_dev`）、燃焼域の範囲（燃料物質の一覧は device）・セル速度・燃料イオン組成（物質の表は device、物質数の上限なし）と Fraley 係数（§14.3）・\(\epsilon_{cum}\) と中性子数の更新も device（`burn_inputs_1d_gpu.cu`、積和を融合せず host のループと同じ値）で、host へ戻るのは段のスカラーだけ。拡散・MC（§14.7、§14.9）は段の誕生源とセルの燃料イオン組成を device に置き、輸送の電子密度、粒子のあるスロットの判定、輸送後の付与（中性子加熱の付与を加える）・\(\epsilon_{cum}\)・\(Q_e, Q_i\)・陽的源の dt 上限（付与の和はセル順、上限は host の `std::min` と同じ規則）も device で計算する（MC の粒子の詰め直しの添字も device の排他的接頭和）。host の写しは出力の直前に同期し、step のやり直しの退避は device の写しと同期の状態も複写する。host 段（`TENRYU_BURN_HOST_STAGE=1`）は段だけを host の写しで計算し、拡散・MC の輸送とその後は同じ device の計算を通る。沈着は
\(e_e{+}\!=dE_e/(\rho V)\), \(e_i{+}\!=dE_i/(\rho V)\) 後に Te/Ti/Pe/Pi を再閉包
（table EOS / cv_override / ideal の全分岐、1T は合算を e_e へ）。沈着と閉包はデバイス上で
レーザー沈着と同じ閉包を使い、energy_authoritative の表 EOS では表の温度上限を超えた
エネルギーを高温側の延長（\(T=T_{top}+(e-e_{top})/c_{v,top}\)、\(P=P_{top}T/T_{top}\)、§1）で
閉じる。沈着の無いセル（\(dE_e=dE_i=0\)）は触らない（2026-09-23 修正: 従来はホストで全セルを
再閉包しており、反応の無いセルの Te を逆変換の許容誤差だけ動かし、表の上限を超えたセルでは
Te を \(T_{max}\) に止めていた）。void・質量ゼロのセルへの沈着は skipped として台帳に計上し、
床注入・クランプ数はセル順の固定順で畳み込む。局所沈着のスキーム（1D は `scheme="fraley"`、2D は `"local"`。1D で
`"local"` は `ConfigError`）で燃焼域（燃料体積分率が vf_threshold を超える最初から最後のセル）に反応しうるセル（\(\rho>0\) かつ \(T_i\ge T_{floor}\)）が無い step は、ステージと沈着を実行せず診断量をゼロにする（沈着ゼロのセルは触らないので結果は実行した場合とbit 同一。2026-09-23、ホスト転送と起動の省略）。核融合エネルギーは
静止質量起源の**外部源**として budget の source 側 `E_burn_in` に登録（逃逸荷電/中性子は
流体に入らないので sink ではない）。W5 実測：burn 活性 6 run すべてで
epsilon_budget ≤ 6e-16。dt 制限は hot-electron 意味論
\(dt \le f_E\,e_{cell}/P_{dep}\)（lineage "burn"）＋ eps_deplete
（subcycle 飽和時は \(0.9\,dt\,M_{max}/M_{req}\) を同じ burn dt 制限へ畳み込む —
§14.2、2026-07-26）。決定論：セル独立
＋固定順縮約（bit 再現。host と device は FMA の違いで一致しないので許容で比べる — 反応網のカーネル単体は rel 1e-13
（`test_burn_network`）、ステージ全体は rel 1e-12（`test_burn_stage_gpu_parity`））。

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
t_E は電子とのエネルギー緩和時間 \(\dot E = -E/t_E\) で、Corman (1975) p. 380 の
\(t_E = 3m\theta_e^{3/2}/(8\sqrt{2\pi m_e}\,n_e Z^2 e^4\ln\Lambda_e)\)
（Spitzer の運動量減速時間の半分。2026-09-23 是正 — 旧実装は分母の 8 を 4 としており
t_E が 2 倍、電子加熱率が半分だった。`Burn.scheme="diffusion"`（1D・2D）と `"mc"` が共有する）。
**イオン Coulomb log と γ は (群, セル) 毎**に群中心エネルギーで評価する
（2026-07-26 修正 — 旧実装は出生エネルギーで 1 回評価し
全群へ流用しており、最接近距離の E 依存が終端域で欠落していた）。
**場イオン**（2026-09-23 是正 — 旧実装は全セルで等モル DT に固定し、イオン Coulomb log の
場イオン密度と質量だけ A=2.5、γ・λ は 2.51505 を使っていた）: γ・λ・イオン Coulomb log の場イオンは
セル毎の組成から作る。燃焼在庫の D/T/³He/⁴He/p（比在庫 \(Y_s\)、重み \(Y_s m_p\)）と、
セルの非燃料・非 void 材料（重み \(vf_m/A_m\) — 在庫の初期化と同じく材料の質量分率を体積分率で置く、
電荷は材料の Z、化合物は平均の Z で代表）を完全電離のイオンとして平均し、\(\bar A\)・\(\overline{Z^2}\)・\(\overline{Z^2/A}\) を得る:
\(n_i=\rho/(\bar A m_p)\)、γ の \(\sum_j n_jZ_j^2/m_j=n_i\overline{Z^2/A}/m_p\)、λ の
\(\sum_j n_jZ_j^2=n_i\overline{Z^2}\)、イオン Coulomb log の Debye 項 \(n_i\overline{Z^2}/kT_i\) と換算質量の
場イオン質量 \(\bar A m_p\)。面の λ は両セルの \(\sum_j n_jZ_j^2\) の平均で評価する。在庫も場の材料も無い
セル（void）は設定の燃料組成 Burn.x_D/x_T/x_He3 を使う。組成はステップ開始時の在庫で評価し、2D の
複数ランクでは所有者の値を ghost セルへ交換する。電子項 t_E は従来どおりセルの
\(n_e=\bar Z\rho/(A_{eff}m_p)\)。
t_E 非正/非有限のセルは γ が有限なら純イオン drag で減速を継続
（\(\tau=(2/3)(E_{g+1}^{3/2}-E_g^{3/2})/\gamma\)、分配は全イオン — §6.5 修正;
旧実装は sink ごと消していた）。

簿記はカスケード転送構成で厳密: 出生は隣接 2 群へ数+エネルギー両保存 binning
（出生エネルギーが最上位群**中心**を超える超過分 `top_excess` は電子へ即時沈着 —
既定格子で D³He 14.663 MeV proton は 704 keV=4.80% が該当。2026-07-26 から
one-shot WARNING で定量報告する。格子再設計（product-aligned grid）は escalate 済み、
2026-07-26 カーネルレビュー指摘）、
転送 1 粒子毎に (Ē_{g+1}−Ē_g) を沈着（**e/i 分配は群内のエネルギー重み付き積分
\(f_i=\frac{1}{\Delta E}\int S_i/F\,dE\)** — 2026-07-26 修正:
旧実装は滞在時間重み \(\int(S_i/F^2)/\int(1/F)\) で、粗い群の e/i crossover 帯で
構造的に別の積分だった。本 scheme の分配は内在で `Burn.partition` は不使用）、
g=1 退場は Ē₁ を全イオンへ（熱化）。**在庫は比スペクトル Y_g = N_g/ρ [1/g] で持続**（§14.2 と同じ Lagrangian
希釈対策 — 密度持続は膨張系で台帳を 11% 破った実測記録あり、設計 doc §C.3）。
飛行中エネルギー E_inflight が新台帳項（released = dep + esc + ΔE_inflight、
実測 2.9e-15；ε_budget は流体側 dep のみ計上で 4.5e-16 恒常）。checkpoint は
hydro/burn_Ng_slot{0..5} [1/g] + time_state/E_burn_inflight（additive）。
cross-scheme 帯: 3 keV/ρ10/ρR0.2 で deposited fraction 比 diffusion/fraley =
0.746（採択帯 [0.60,0.90] — 直線点核 vs 拡散+Milne 逃逸+スペクトル拡散の模型差）。この値は 2026-07-10 の測定で、
その後の 2026-09-23 の変更（電子とのエネルギー緩和時間の係数 8 — 電子加熱率 2 倍、セル自身のイオン組成での減速）の前の
ものであり、変更後は測り直していない。この帯を検査する試験も無い。

### 14.8 中性子スペクトル合成診断（v2、read-only）

Brysk 1973 の二 Maxwell 平均モーメント。燃焼重み付き ⟨T_i⟩_burn・⟨v_r²⟩_burn
（DT / DDn 別、固定順セル和）から:
平均シフト \(\langle E_n\rangle - \tfrac{m_\alpha}{m_n+m_\alpha}Q =
\tfrac{m_n}{m_D+m_T}\tfrac{3}{2}\theta + \tfrac{m_\alpha}{m_n+m_\alpha}\langle K\rangle\)
（⟨K⟩ = 3T_reac−(3/2)θ、T_reac は Brysk Table 1 転写（Reac 列 = ⟨E⟩/3 と解読、
両公表 anchor 35 keV/336 keV·33/157 keV を実装前検算で再現）、log-T 補間・[1,100] keV clamp）、
熱幅 σ² = 2m_nθ⟨E_n⟩/(m_n+m_partner)（ガウス分布の標準偏差。Brysk の 336/157 keV は
1/e 半値半幅 \(\sqrt2\sigma\)。実装は 2026-09-23 まで係数 1/2 を掛け、σ を半分に報告していた）、
全幅は 4π 平均の流体広がり
σ_fluid² = 2m_nE_{n0}⟨v_r²⟩/3 を加算（球対称 1D の合成検出器は方向平均 —
一次モーメントは対称消失、視線スペクトルは v3/Crilly-Munro scope として設計 doc 記録）。
history `burn/neutron_{Ti_burn,mean_shift,sigma_thermal,sigma_total}_{dt,dd}`
（burn 有効 run のみ、エネルギー簿記への影響ゼロ）。

### 14.9 MC α 輸送（v2、`Burn.scheme="mc"`、統計モード）

Yuan-Moses-McKenty 2005 型の直線 CSDA Monte Carlo（1D 球面特化、角散乱なし —
偏向 λ は拡散 scheme のみ）。**停止能係数は §14.7 と同一**（場イオンを含む。corman_tE/γ 共有 —
scheme 間一致 gate が模型恒等性の検証になる）。イオン Coulomb log は
**粒子の現在エネルギー**で毎セグメント評価（2026-07-26 修正 —
旧実装は出生エネルギーで凍結、Bragg-peak 近傍の γ を誤っていた。RNG 消費は不変）。粒子 (r, μ, E, w, slot) は
ステップ間持続 pool（時間依存近似）、出生は殻内 r³ 一様 + 等方 μ、
**RNG は Philox / curand_init(seed^global_id, subsequence=step, offset) —
NUMERICS §12.7.1 凍結契約**（global_id = (cell·6+slot)·N_mc+sample）。
CSDA 沈着は局所瞬時レート比で e/i 分割、熱化 E≤E_min → イオン、逃逸 → 台帳。
新しい粒子は pool の既存粒子の後ろに (cell, slot, sample) の順で並ぶ（(cell, slot) ごとの粒子数の排他的走査で
位置を決める）。セルへの沈着は 128 ビットの固定小数点の整数（64 ビット 2 語、下位語の桁上がりを上位語へ）に
原子的に足す: そのステップの粒子の総エネルギー E_tot = f·2^e（0.5 ≤ f < 1）に対し 1 単位 = 2^(e−100) で、
1 回の沈着の丸めは 2^(−100)·E_tot 以下。整数の和は順序に依らず、発生・逃逸・飛行中の集計も
(cell, slot) または粒子ごとの値の固定順序の和なので、同じ seed の再実行はビット一致する（2026-09-24。
それまでは浮動小数点の atomicAdd で、pool の並びもスレッドの実行順だった）。負または非有限の沈着が
あったセルは沈着を NaN で返す。
per-particle 簿記により台帳恒等 released = dep+esc+ΔE_inflight は RNG に
依らず厳密（実測 8.5e-15、ε_budget 5.6e-16）。
**三 scheme 整合（3 keV/ρ10/ρR0.2 実測、2026-07-10）**: deposited fraction
fraley 0.968 / mc 0.928 / diffusion 0.722 — mc（参照級）に対し fraley は
その解析近似（+4%）、diffusion は Milne 逃逸+スペクトル拡散で低め、と
物理的序列どおり。mc と diffusion の値は 2026-09-23 の変更（電子とのエネルギー緩和時間の係数 8、セル自身のイオン組成での
減速）の前のもので、変更後は測り直していない（§14.7 の帯と同じ）。CV gate: 同 seed 5 run CV ≤ 1e-3（§0.3 文言。2026-09-24 以降はビット一致）
+ 異 seed 5 run CV ≤ 5%（統計収束、1/√N 傾向は 社内の性能記録 記帳）。

### 14.10 2D_RZ port（scheme="local"|"diffusion"、2026-07-11）

\`Main.dimension="2D_RZ"\` で Burn.enabled=True が有効（設計記録は
社内の設計メモ 2d_burn_port_spec.md、実装は src/burn/burn_stage_2d +
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
  （fld_2d_rz_gpu.cu の per-material 沈着規約、max(...,0) clamp、Te/Ti_per_material
  キャッシュ無効化）。
- **retry 整合**: DriverRetrySnapshot が burn 在庫・累積台帳・Y_g を
  capture/restore（STRANG では burn が hydro half より先に走るため必須）。
  1D 側は同 snapshot に burn 未収載の継承ハザードあり（merge train で解消）。
- **既知の残余（v1）**: host 主体 Y と per-step mirror の perf 繰延、種は
  cell-level（per-material 種分解は v2）、Post-Wilson/Milne の非原点中心
  問題はヒューリスティック帯、Brysk 中性子診断キーは 2D では 0.0 のまま
  （スペクトル合成は未移植）。

### 14.11 中性子の最初の衝突による加熱（`Burn.neutron_heating`、v2-E、1D 専用）

`Burn.neutron_heating=True`（既定 False）で、DT-n（14.049 MeV）と DD-n（2.449 MeV）の 2 本の中性子線について、1 回の飛行の
最初の衝突だけを扱う加熱を加える（host `deposit_neutron_heating_1d`、GPU のステージは `neutron_heating_device.cuh`）。凍結した
断面積と設計の記録は 社内の設計メモ burn_kernel_v2_20260710.md §E。

- **放出**：各セル・各線で、そのステップの反応が生んだ中性子のエネルギー \(E^{emit}_{c,l}\)（反応率 × 線のエネルギー × \(V\Delta t\)）を
  セル中心の半径 \(r_0\) から等方に出す。方向は \(\mu\in[-1,1]\) の偶数次 Gauss–Legendre 求積（`neutron_heating_n_mu`、既定 16、
  [2, 64]）で、各方向の弦を殻の境界ごとにたどる（内部の節点は \(|\mu|<1\) なので衝突径数 \(b=r_0\sqrt{1-\mu^2}>0\)、中心を
  通らない。中空の格子では中心の穴を真空として素通りする）。
- **減衰**：標的は燃料の D と T だけ（燃焼在庫 \(Y_s\rho\) の数密度）で、巨視断面積 \(\Sigma=n_D\sigma_D+n_T\sigma_T\)。
  弾性散乱の断面積は 14.049 MeV で D 0.63 b・T 0.94 b、2.449 MeV で D 2.31 b・T 2.30 b。それ以外の物質（殻、⁴He、p）は透明
  （殻の kerma は v3 の課題）。殻の区間 \(\Delta s\) で衝突する割合は \(T_{rem}(1-e^{-\Sigma\Delta s})\)（\(T_{rem}\) はそこまでの透過率）。
- **沈着**：衝突した中性子のエネルギーのうち反跳核へ渡る平均の割合 \(f=\tfrac{2A}{(A+1)^2}(1-\langle\cos\theta_{CM}\rangle)\)
  （D: 14 MeV 0.3157・2.45 MeV 0.4447、T: 0.2053・0.3756）をその殻のセルへ沈着し、Li–Petrasso の分配表の反跳 slot（6〜9、§14.4）で
  電子とイオンに分ける。残りの \(1-f\) は散乱した中性子が持ち去る分として追跡しない（degraded）。衝突しなかった分は逃げる。
- **台帳**：沈着は `E_burn_in` に入る（§10.2）。放出 = 沈着 + degraded + 逃げ の恒等を毎ステップ評価し、相対残差を結果に返す。
- **制約**：1D 専用（2D_RZ は `ConfigError`）。`Burn.partition="fraley"`（DT の α の当てはめで、D・T の反跳を分けられない）
  との併用は `ConfigError`。多重散乱・減速した中性子の輸送は無い（単体試験 `test_burn_neutron_heating` のみ）。

## 15. ReALE v2: exact tessellation, conservative overlay remap, and the persistent boundary carrier

Design canon: the internal design note amm_reale_plan_20260806.md (program),
the internal design note tessellator_core_contract_20260808.md (exact core),
the internal design note boundary_carrier_c1_20260810.md (carrier; consult-29 adoption A229).
Ledger of record: `tmp/ale_p2_briefs/killer_p2b_verdict.md` (A199-A270).
Design rule (a user ruling): thresholds are machine-epsilon-derived bounds or
absolute predicates, not tunables. The current code does not meet the rule
everywhere: it reads the namelist tolerances
`Numerics.ale.reale_short_edge_collapse_rel` (default 3.0e-2),
`reale_overlay_additivity_tol` (1.0e-4), and `reale_subdomain_frac_max` (0.6,
with `reale_subdomain_rezone=True`), the environment ceiling
`TENRYU_REALE_REZONE_INTERVAL` (§15.4d), and fixed resource caps (§15.4a: 256
Newton degrees of freedom, 256 Dykstra sweeps, 64 quotient sweeps; §15.4d: 4
Lloyd iterations).

### 15.1 Exact restricted Voronoi core (`src/mesh/tessellation/`)

- Incremental global Delaunay without a super-triangle. Orientation and
  incircle first pass through semi-static Shewchuk A-bound filters widened by a
  factor of two for compiler-contraction safety. Every sign accepted as certain
  by a filter equals the exact sign; uncertain cases fall through to the
  unchanged exact-expansion evaluation and, where applicable, symbolic
  perturbation (SoS) keyed solely on stable generator ids. The exact layer keeps
  its certified running-error filter (per-op (1+4eps) slack, (1+n eps)
  magnitude compensation). The two-times widening also closes the latent
  non-conservative band in the pre-existing filtered orientation call sites;
  exact-zero and SoS tie semantics are unchanged (ledger A270 addendum).
  Morton/BRIO insertion.
- ReALE retains the final Delaunay triangulation at process-run scope and passes
  it into the next rezone as the warm state. A generator-count change resets the
  state, and stable-id matching guards site identity. The warm path uses the
  same exact predicates and deterministic ordering; exactly cocircular ties may
  select a different valid triangulation than a cold build, so landing the warm
  path may change mesh history once, but run-to-run determinism is unchanged.
  `[tess_ms] warm` reports whether the final build used a warm entry (A269-A270).
- The Voronoi dual is restricted to a simple CCW domain polyline by winding-number
  classification; boundary crossings are constructed exactly. Since C1-w2 each
  crossing carries the exact rational parameter lambda* = -phi(A)/(phi(B)-phi(A))
  on its domain edge (phi affine along the edge), with a correctly rounded double
  cache obtained by PER-COORDINATE snap of the exact point. (The response-§5
  lambda-interpolated cache was measured to have a precision cliff -
  ULP(lambda)*edge_length can exceed the separation of near-origin crossings -
  and was amended; A229 addendum.)
- Spatial prefilters (2D AABB grid + 1D z-interval buckets, sizes derived from the
  segment count) make the three O(E_domain) exact scans O(1)-ish per query; skips
  are provably outcome-neutral, certified by the byte-identity battery (A227).
- Canonical TRV1 serialization is construction-path independent (permutation cases
  assert byte identity).

### 15.1a GPU circumcenter batch (T2-GPU, opt-in, default OFF)

- `Numerics.ale.tess_gpu_dual=True` offloads the per-triangle exact
  circumcenter construction of the dual build
  (`src/mesh/tessellation/gpu_circumcenter.cu`) to the GPU. The device kernel
  replicates the host expansion pipeline operation-for-operation (explicit-fma
  two_product, tie-preferring magnitude merge, zero elision, exact power-of-two
  scaling; TU compiled `-fmad=false`), so admitted triangles return the host
  term sequences and constructed pair byte-for-byte. Semantics are therefore
  unchanged by construction; the switch is a pure venue change.
- Fixed capacities derived from the operation tree (numerators <= 48 terms,
  denominator <= 16, cap-64 working buffers). A per-triangle structural
  certificate — capacity overflow, zero/nonpositive denominator, nonfinite
  accumulate — routes that triangle to the unchanged host `circumcenter_exact`
  (the exact-fallback shortlist, processed in the deterministic class-emission
  order), so fallback and admitted triangles alike produce the host bytes.
- Device storage is a single explicitly indexed per-thread workspace (22 named
  + 8 scratch slots): measured nvcc misallocation of many live 520-byte
  aggregate locals in the fully-inlined kernel (distinct expansions sharing one
  local slot; correct under `-G` only) forbids compiler-managed FixedExp
  locals — see the internal design note t2gpu_port_design_20260815.md.
- Determinism: pure per-triangle map, no atomics, fixed-stride writes;
  host-side consumption is order-identical to the host path. Buffers come from
  the persistent device/pinned scratch pools (no per-call cudaMalloc).
- Verification: OFF keeps the pre-existing host path untouched; ON is gated by
  canonical-serialize byte identity against OFF on the production 69,626-site
  cloud and synthetic cocircular grids (member-verify path exercised), a
  forced-fallback identity case, and the `TENRYU_TESS_GPU_CC_SHADOW=1`
  per-class witness/constructed byte shadow gate.
- `Numerics.ale.tess_gpu_restrict=True` (requires `tess_gpu_dual=True`,
  `ConfigError` otherwise) additionally offloads the per-dual-node domain
  classification of the restrict stage (`gpu_classify.cu`): a certified
  brute-force winding over all domain segments computed from the
  device-resident dual term sequences, where every z-compare and orientation
  sign is accepted only through the replicated running-error ApproxBound
  filters widened by the standard factor of two. A certified winding equals
  the exact winding, and certified nodes provably touch no segment (an exact
  zero orientation is never certifiable), so their classification —
  inside/outside with empty supports — is the exact host result; every
  uncertain, boundary-touching, or fallback-backed node runs the unchanged
  host `classify_domain_point` in node order. Output is one status byte per
  node; determinism and byte identity follow as above. Gates: real-cloud and
  exact on-boundary-circumcenter serialize identities, forced-fallback and
  restrict-OFF purity cases.

### 15.2 Conservative overlay remap (Option D; `src/hydro/reale_remap.cpp`)

- For old cell i with bbox-candidate target set T(i), each pair (i,j) clips old_i
  by the bisector half-planes of generator j against ALL u in T(i)\{j}: the pieces
  form the exact nearest-generator partition of old_i (old cells tile the domain
  the targets tile; the true owner of every point is always in T(i)). Boundary
  polygon-edge half-planes are ABOLISHED (snapped sliver edges gave rounding-noise
  plane directions; leave-one-out attribution A226).
- Gates (fail-closed, rezone skip): per-cell overlay additivity within the derived
  bound 4 pi eps s^2 perimeter; certified-positive cell volumes
  vol > 8 n eps * magnitude (covers per-term rounding, accumulation, and cross-TU
  FMA-contraction variance of the token-identical volume formulas; A228).
- Cell vertex-count capacity is kMeshTopoCellStorageSlotsMaxGeneral = 16 end to
  end (the pointer overload of mesh_topo_cell_active_nverts, the 8-slot gather
  buffers, and the bespoke capacity constants were all widened in A228; a
  valence > 16 candidate is rejected fail-closed until C3 ring pages).

### 15.3 Persistent Lagrangian boundary carrier (C1, A229-A230)

Mechanism (consult-29 ruling, verified): re-extracting the domain from the mesh
boundary every rezone promotes Voronoi-boundary junctions to domain corners,
giving |B_{g+1}| ~ |B_g| + |J_g| (measured +420 vertices/generation) and unbounded
valence growth. The carrier ends this by object identity:

- The domain polyline is defined EXCLUSIVELY by N_B persistent carrier masters
  (the gen-0 compacted boundary walk; N_B = 193 on the pilot deck), which are
  ordinary Lagrangian mesh nodes tracked by index across installs.
- Boundary junctions are ephemeral slaves (carrier_edge, lambda_exact), rebuilt
  every rezone, never fed back into the domain. Measured invariant:
  n_domain_vertices = N_B exactly, every generation; mesh node count constant.
- During the Lagrangian phase, slave positions/velocities are reconstructed from
  the masters (X_s = (1-lambda) X_A + lambda X_B) after every position/velocity
  substage, and slave forces/masses are condensed to the edge masters
  (M_q = C^T M_b C, f_q = C^T f_b - the work-conjugate form; cyclic-tridiagonal
  LDL^T per boundary component; axis masters lose the radial DOF). Hook sites are
  enumerated in `tmp/ale_p2_briefs/c1w3b_integration_map.md`.
- Armed fail-closed gates (rezone skip): carrier_master_missing (each master
  appears exactly once), carrier_representability (two distinct exact junctions
  sharing one snapped double pair), carrier_provenance_conflict,
  carrier_coverage (15.4).

### 15.4 Post-remap projection and exact coverage gate (C2, A231)

- After each remap under the active carrier, boundary velocities are projected
  into the new constraint space: (C_new^T M* C_new) qdot = C_new^T p*
  (mass-orthogonal; total momentum preserved exactly by row-sums-one; solved by a
  deterministic bordered cyclic LDL^T). The nonnegative kinetic defect
  Delta-K = 1/2 (u*-u)^T M* (u*-u) is allocated per cell through corner masses
  (Sigma dE_c == Delta-K asserted at 1e-9 relative) and closed into specific ion
  internal energy - total energy is conserved by construction.
- The exact coverage gate checks, per carrier edge, that the boundary subedge
  intervals partition [0,1] in EXACT rational lambda (first==0, adjacent equality,
  last==1, nondegenerate) - evaluated where the exact lambdas live (after the
  boundary arrangement); any violation skips the rezone.

### 15.4a DV-CLP velocity projection

DV-CLP projects the remapped nodal velocity field componentwise in the nodal
mass metric subject to every accepted in-edge velocity-jump budget. Components
with `dof_count <= 256` retain the semismooth-Newton SOCP and exact
certificate path.

The Newton KKT solve is a diagonal-pivot dense LDL^T with
drop-on-zero-pivot recovery. Under `Numerics.ale.dvclp_solver_rev=1`
(default 0 = legacy bit-path), every Newton refactor entry first runs a
deterministic column-pivoted Householder QR over the active-row gradient
columns and drops linearly dependent rows before factorization. The
systematic source of such dependence is structural: a violating edge whose
velocity jump is pure-z (or pure-r) contributes a zero radial (axial)
gradient component, so uniform-vr violating cycles yield pure z-difference
rows whose signed sum telescopes to zero — exactly singular KKT systems that
the SS8-w2fix26 duplicate-support dedup cannot detect because every support
differs. Determinism: the pivot is the largest explicitly recomputed
residual norm (no downdating), ties broken by smallest original active
position; the rank cut uses the separation band `64*(m+n)*eps*max_j||g_j||`
(m active rows, n dofs), far above the rounding residue of an exactly
dependent column and far below any mesh-quality-bounded independent one.
The band guards solution quality only, not correctness: the final
feasibility scan, the homothety alpha, and all certificates evaluate edges
— never solver rows — so a mis-dropped row can only alter the Newton search
direction (a wrongly *kept* dependent row falls through to the legacy
zero-pivot drop as before). Measured on the 20-triangle-chain reproduction
(P3 harness): rev=0 collapses to alpha = 1/worst-ratio with an
FMA-dependent branch (0.419 contracted / 0.550 not); rev=1 converges to
alpha = 0.99499 in both FP regimes with zero LDL^T zero-pivot drops.
`[dvclp_comp]` reports the per-component drop count as `qrd=`;
`qr_dropped_rows` accumulates it per projection.

A component capped above 256 degrees of freedom uses this
deterministic ladder:

1. **D1 mass-metric Dykstra.** In stable edge-id order, each edge projects its
   provisional r/z jump onto the budget 2-ball and distributes the correction
   between its endpoints about their pair mass center. One two-vector Dykstra
   correction is retained per edge. The serial host sweep has a fixed resource
   cap of 256 full sweeps; after every sweep the existing exact all-edge
   feasibility scan permits an early exit. A passing result is
   `dykstra_feasible`, a feasibility-only result rather than a KKT/zero-gap
   exact projection. It then traverses the existing unit-collapse epilogue and
   post-collapse all-edge re-scan.
2. **D3 active-set KKT certificate.** A `dykstra_feasible` component then
   attempts a label upgrade to a certified exact projection (Euclidean
   edge-ball KKT form). The active set is identified by the derived band
   `jump >= budget * (1 - 8*(walk_segments+2)*eps)`, the exact mirror of the
   feasibility predicate's upper band; zero-budget edges are equality-active
   and re-verified bitwise. A dual witness is constructed by leaf elimination
   over a stable spanning forest of the active subgraph with independent r and
   z flows; cycle-closing edges, inactive edges, and axis-edge radial
   multipliers are pinned to zero. The certificate verifies stationarity per
   node with a Neumaier compensated sum under the derived envelope
   `8*(deg+2)*eps*Sum|terms|`, and the Euclidean normal-cone condition per
   active edge (the multiplier parallel to the jump and non-negative along
   it, within `16*eps` relative envelopes). The stage never modifies
   velocities: the paired Dykstra updates conserve component momentum to
   rounding, so the equality-KKT primal correction (the component translation
   null-mode) is sub-roundoff by construction and the witness is purely
   diagnostic. On full pass the component reports `solver_class = 4`; on any
   clause failure the feasibility-only label `solver_class = 1` is kept —
   never demoted — and the failing clause is counted
   (`kkt_fail_eq`/`kkt_fail_stationarity`/`kkt_fail_cone`).
3. **D4 slack-component quotient translation.** On D1 exhaustion, connected
   components of the non-violating in-edges form rigid clusters. Internal edges
   remain unchanged under a cluster translation. Each violating cut edge
   becomes the shifted-ball constraint
   `||(tau_k - tau_l) + d0_e|| <= b_e`, solved by the same mass-metric
   Dykstra pair projection on cluster masses in stable cut-edge order. A cluster
   containing an axis node has its radial translation frozen. The quotient has
   its own 64-sweep resource cap (`kDvclpQuotientSweepCap`); success uses the same unit-collapse
   epilogue and post-collapse all-edge re-scan.
4. **Homothety fallback.** If the quotient exhausts its cap or the
   post-collapse scan finds a residual violation, the pre-D1 component snapshot
   is restored bitwise and the existing homothetic completion runs.

The `[dvclp_comp]` solver classes are 0 = semismooth Newton, 1 =
Dykstra-feasible, 2 = homothety fallback, 3 = quotient-translation
feasible, and 4 = Dykstra-feasible with a passed active-set KKT
certificate (an exact projection to rounding). `dysweeps` reports D1
sweeps; `qk`, `qcut`, and `qsweeps` report quotient cluster count,
cut-edge count, and quotient sweeps; `kkt` and `lam_max` report the
per-component certificate outcome and multiplier norm. Stable
iteration order is deterministic, each mass-weighted pair update conserves
pair momentum in real arithmetic, and the existing exact-momentum closure,
final all-edge scan, and conservation ledgers remain the commit authorities.
The large-component rationale, rejected block-cut localization, observed D1
exhaustion, and completed D4 ladder are recorded in ledger A264-A268.

Consensus admission proof classes are reason-coded per consult-33 (99): exact
zero budgets (ANALYTIC_ZERO), the eq. 52'/C2 rounding-envelope certificates
(ROBUST_ZERO), and the lattice class (LATTICE_ZERO). A certified budget upper
bound below
\(g_e=\min(\mathrm{ulp}(v_{r0}),\mathrm{ulp}(v_{r1}),\mathrm{ulp}(v_{z0}),\mathrm{ulp}(v_{z1}))\)
admits no representable nonzero velocity difference, so equality is the only
representable feasible point; the minimum over endpoints and components is the
conservative cross-binade floor. Admission never depends on what the fallback
family would do (the §5.6 guard).

### 15.4b Area-weighted planar corner vectors on general cells (A234)

The AW (area-weighted) pressure corner-force kernel builds PLANAR corner
vectors per cell shape. Triangles and pentagons keep their dedicated
constructions (the pentagon uses the condensed subzonal-consistent form with
its closure check); quads keep `aw_planar_corner_vectors` (bit-preserved). All
other valences n use the same shoelace-gradient construction generalized
verbatim: with polygon orientation sigma = sign(2A) from the shoelace sum,

  S_k = (sigma/2) * ( z_{k+1} - z_{k-1},  r_{k-1} - r_{k+1} ),  k = 0..n-1

(`aw_planar_polygon_corner_vectors`; the quad helper is the n=4 instance).
Closure Sigma_k S_k = 0 holds identically (telescoping) and is asserted
per cell at rel <= 128*DBL_EPSILON of Sigma_k(|S_r|+|S_z|) (the per-term
differences are Sterbenz-exact or same-binade-rounded, and the telescoping
partial sums are bounded by the cell extent, so the true FP residual is
O(n*eps) - the margin is wide). The RZ weighting (2*pi*r_k per corner) is
shape-independent and unchanged.

History (A234): before this construction landed, general (>=6-valence) cells
fell into the quad branch. In the pre-A228 clamp era they were silently
evaluated as quads (first four vertices - planar-wrong forces); after A228's
honest nverts they read uninitialized stack (UB), proven reachable by a guard
pilot at the first post-rezone hydro step. Field-level claims from pilots in
that window are not usable as quantitative anchors; post-A234 pilots
re-baseline the march.

### 15.4c Representation-conditioned weld quotient and staged-dt gate (A246)

The exact restricted Voronoi core applies a deterministic degeneracy quotient
before exposing its final DCEL. For a represented edge $e=(a,b)$, let

\[
 \ell_e^2=(r_b-r_a)^2+(z_b-z_a)^2,\qquad
 \delta_e={u(r_a)+u(r_b)+u(z_a)+u(z_b)\over 2},
\]

where $u(q)$ is the larger of the two adjacent binary64 spacings at $q$.
For each raw pre-weld cell $c$, $A_c$ is its exact shoelace area and
$P_c=\sum_i(|\Delta r_i|+|\Delta z_i|)$ is its exact-dyadic L1 perimeter;
$h_c=2A_c/P_c$. For each node $a$, $h_a$ is the minimum over all cells
incident to that node; ties are broken by stable site id. Every measured pair
$e=(a,b)$ then uses the single canonical node-h value
$h_e=\min(h_a,h_b)$, with the same deterministic tie-break. Candidate
admission, the complete-link certificate, and the final residual audit all use
this measurement; source-edge incident cells do not provide an alternate
$h$ attribution. The cell areas, L1 perimeters, and resulting local scales are
frozen from the raw DCEL: contractions never redefine the acceptance scale.
This unification removed the source-edge/node-h/audit measurement triangle
that starved the rezone cadence (ledger A263). A pair welds exactly when

\[
 \chi_e={\ell_e^2\over\delta_e h_e}\le 4,
 \quad\hbox{equivalently}\quad
 \ell_e^2P_c\le8\delta_eA_c
\]

for the node-incident cell attaining $h_e$. Equality welds, and the comparison
is an expansion-sign decision, not a rounded quotient with a tolerance. This
robust floor is derived, not tunable: the $\pm2$-ulp application band on
$\ell_e$ and the $h$-attribution spread between the pair's two node-$h$ values
each carry a factor-2-class ambiguity, so a measured
$\chi_e\le2^2=4$ cannot be robustly distinguished from the floor. The
underlying balance point compares retaining the represented short edge,
$E_{retain}=\delta_e/\ell_e$, and collapsing it,
$E_{collapse}=\ell_e/h_e$. The L1 perimeter is deviation D1 from the L2
perimeter in the conditioning derivation: $P_{L1}\ge P_{L2}$ makes the rule
at most \(\sqrt{2}\) more conservative while keeping the decision exactly
dyadic.

The measured snap contract corrects an earlier premise without changing the
criterion. Constructed coordinates carry a few ulps of arithmetic error; they
are not necessarily correctly rounded. In the measured specimen, constructed
z was one ulp below its segment and constructed x was three ulps from the exact
rational. Nevertheless, \(\delta_e\) remains the representation-spacing bound
used by the conditioning quotient within the robust \(\chi_e\le4\) rule.

Contractions obey the carrier classes. Interior may contract only with
interior; slaves only on the same carrier, or into their own carrier endpoint
master; axis nodes only within the same axis carrier or into its axis master.
Boundary-interior, cross-carrier, axis/off-axis, and distinct master-master
pairs are forbidden. A cluster contains at most one master; otherwise the
representative is the smallest exact carrier lambda for same-carrier slaves or
the smallest canonical node key for interiors. Complete-link clustering blocks
macroscopic transitive chains. After the fixpoint, boundary intervals and the
DCEL are rebuilt and every final edge is audited. The fail-loud invariant is
two-banded: a forbidden-class or master-master pair triggers
`weld_forbidden_class_pair` / `weld_master_master` only in the strict band
\(\chi_e\le1\), where a sub-noise edge between unweldable nodes proves broken
geometry. In the extended band \(1<\chi_e\le4\) the edge is not proven noise,
so such pairs are retained silently and counted (`extended_band_forbidden`).
The final audit rejects the rezone (`weld_residual_subfloor`, through the
ordinary skip path) only for strict-band residuals; extended-band residuals of
any class are census-only diagnostics.

After remap and carrier-velocity projection, but before installing any staged
arrays, the host staged-dt gate evaluates every staged ring edge in cell/face
index order. It transcribes the base CSW98 edge-AV CFL expression using the
configured `Numerics.hydro.av_cfl_coefficient`; deviation D7 fixes the host
limiter to \(\psi=0\), conservatively biasing the bound toward rejection, and
keeps a parity-tested host copy rather than a shared device header. It also
applies the node-crossing bound
`crossing_dt_safety * edge_length / closing_speed` for positive face-direction
closing speed. The rezone commits only when the minimum staged bound is strictly
greater than `Numerics.ale.rezone_min_dt_s`; otherwise it emits a `staged_dt`
SKIP with the bound, threshold, and binding cell, leaving the installed mesh
untouched. The default is \(10^{-14}\) s, zero disables evaluation, and the
threshold is a deck-owned production requirement, never a geometric weld
tolerance.

### 15.4d Need-based rezone trigger (SS8 Track-3)

The `reale_v2` rezone fires on need, not cadence: at step \(n\) it runs when (a) any cell corner has moved at least \(h_c/2\), with \(h_c=\sqrt{A_c}\) for planar cell area \(A_c\), since the last committed rezone (a CFL-class half-cell displacement), (b) the hydro dt bound has fallen to half its post-commit reference, captured on the first step after each commit, or (c) `TENRYU_REALE_REZONE_INTERVAL` steps have elapsed since the last commit (the environment variable is now a ceiling; default `1` preserves the historical every-step behavior). An evaluation without a matching displacement reference — the first of the run, or any state whose node count differs from the last snapshot — fires unconditionally and is logged as reason `first`; the need reasons log as `disp`, `dt`, and `ceiling`. The \(1/2\) factors are derived CFL-class constants, not tunables. The displacement census is GPU-resident: one per-cell max-reduce with a deterministic atomic max; its reference is the installed post-rezone node set, refreshed device-to-device at every commit. Each firing logs `[reale_trigger]` with the reason, and the COMMIT line carries `trigger=`.

The target builder's Lloyd/CVT refinement is likewise need-based: the loop (ceiling kLloydMax = 4) exits early when every proposed site move is below that site's own coordinate representation-spacing scale \(\delta_i=\mathrm{ulp}(|r_i|)+\mathrm{ulp}(|z_i|)\) — the same derived representation-spacing family as the weld's \(\delta_e\) — since such an iteration polishes below the floating-point noise floor of the stored coordinates while paying a full tessellation. The exit is evaluated before the proposal's tessellation is built, and the iteration ceiling is unchanged.

The primary exit ahead of that floor is the d1 certified predicate-margin skip (`certify_lloyd_noop`, `src/mesh/tessellation/lloyd_skip.cpp`; derivation in the internal design note t2_d1_certified_lloyd_skip_20260815.md): before tessellating a proposal, every certificate of the current warm Delaunay triangulation — triangle orientations, interior-edge incircle tests, and hull convexity triples — is checked in pure double arithmetic for a positive filtered slack \(s=|\tilde D|-E\) (the sos_policy Shewchuk A-bounds with their 2x contraction safety) against an interval-propagated bound \(\Delta_D\) on the determinant's change under the proposed per-site displacements (product rule \(\Delta_{xy}\le U_x\Delta_y+U_y\Delta_x\) mirrored over the predicate's own expression tree); the certificate fires only when \(2\Delta_D\le s\) for every predicate, which proves no exact predicate sign can flip along the whole displacement path and hence that the proposal's Delaunay topology equals the current one — the iteration is structurally a no-op and the loop breaks without adopting it, exactly the d2 break action. The d2 floor remains as the fallback when d1 refuses (degenerate slack on exactly-cocircular or collinear sub-configurations, mismatched warm base, or the `TENRYU_LLOYD_D1_DISABLE` off-switch). Conservativeness only reduces skips, so correctness is unconditional; the certified skip matters in a measured regime the floor cannot serve: relaxed configurations whose Lloyd feedback limit-cycles at a total move near \(10^{-14}\) — above the representation floor, so d2 never fires — while the certified margin (typically \(10^{-6}\)-\(10^{-4}\) of the local spacing) fires immediately and stops the loop from paying its full four tessellations every rezone. Fires log one `[lloyd_d1]` line and are exposed as `lloyd_d1_fired`/`lloyd_d1_iteration` in the target result; no tunables are introduced.

After each committed rezone the dt controller may re-anchor the growth ladder once, upward only: the first post-commit chosen dt is raised to the minimum of all non-growth bounds and the last pre-spike chosen dt (the most recent chosen dt whose limiter was not the hydro bound). A spike edge that the rezone has removed is not a persistent constraint, so the ladder's memory of it is stale by construction; the pre-spike ceiling makes the trigger/re-anchor pair converge to the pre-spike operating point rather than oscillate, and a genuinely degraded mesh (hydro bound still low) keeps full ladder protection because the raise is bounded by the current hydro term. The lineage limiter records `rezone_reanchor` when the raise applies.

### 15.5 Known deferrals

C3 ring pages (16 becomes a storage page size, logical valence unbounded - until
then the valence gate guards); full-TEND survival characterization; hook
full-array D2H/H2D profiling; CECR Stage-2 (boundary cavities; the child lane
fixed e2_insert - non-convex patch point location - and re-attributed
hull_contact to a boundary-band sliver structural limit, A230); checkpoint IO of
the carrier (restart-incompatible meanwhile).
