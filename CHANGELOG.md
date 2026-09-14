# TENRYU β 更新履歴

配布スナップショットの更新記録です。日付はスナップショット作成日。

## 2026-09-15

### 不具合修正

- **FLD の Fleck 線形化（1D_SPH / 2D_RZ）: 物質更新の再放射項が放射側と違う不透明度を使っていた。** 放射の E 方程式は実効散乱を \((1-f)c\sigma^{PA}E^n\)（吸収不透明度）で戻すのに、物質側の Newton 残差と `rad_emit` 記帳は 2026-07-03 以来 \(\sigma^{PE}\)（放射不透明度）を使っていました。\(\sigma^{PE}\ne\sigma^{PA}\) の TMAT 表では毎反復 \((1-f)c(\sigma^{PE}-\sigma^{PA})E^n\Delta t\) のエネルギーが消え、冷たいセルの外側反復が 2 周期で振動していました。1D の物質更新・persistent loop 版・2D_RZ の物質 Newton を \(\sigma^{PA}\) に統一（灰色・定数不透明度では算術不変）。単体テスト（`test_fld_fleck_beta` の blend ケース）を追加（NUMERICS §6.7 の注記を訂正）。
- **1D hydro の compatible energy 更新で、自由境界の ghost Q が 2026-08-04 以降 0 になっていた。** 運動量式は境界セルの Q をゼロ勾配コピーした ghost を使うのに、エネルギー更新側だけ ghost が 0 になり、自由境界セルに人工粘性があると毎ステップ \(Q_{N-1}A\bar u\Delta t\) のエネルギーが生成されていました。`test_hydro_1d_step` の VNR 試験の失敗を bisect で特定し、FREE 分岐を復元。影響は `compatible_energy=True` かつ `boundary_1d="free"` の run のみ。
- **符号付き表エネルギーの許容域を 1D の全評価器で統一（energy_authoritative モード）。** cold curve が負の表エネルギー（PROPACEOS 由来表の低温側）を、伝導の増分・閉包の修復・FLD の末尾経路・Qei・注入の床分岐が 0 へ切り上げ／床へ埋め戻していたため、冷たい多材料デッキ（液体 D2 + CH）で静止セルのイオンエネルギーが毎サイクル生成されていました。許容域を \([e(\rho,T_{\rm floor}),\infty)\) に統一し（修復は非有限のみ、`floor_clamp` の拒否、Qei は両方向を床上エネルギーで制限、注入の床分岐は温度のみ）、`qei_multiplier` を hydro カーネルにも配線。legacy モードの算術は不変。
- **多材料の 1D 閉包が先頭材料の表に固定されていた。** hydro の EOS 閉包・音速・2T エネルギー更新、輻射の物質更新、Qei、レーザー／燃焼の入射、伝導の再閉包、初期化を各セルの支配材料の表で閉じるようにしました。表の密度下限未満は理想気体へ切り替えず最下行へクランプ。再閉包の表天井キャッシュは Config の表で鍵付け。
- `Materials.materials[].opacity.tmat_skip_lte_repair` が namelist 検証時の変換にしか効いておらず、実行時の表再読み込み（FLD 1D/2D・S_N 1D/2D・IMC・persistent loop・硬 X 線診断）では常に修復が適用されていました。全読み込み点へ配線。
- 表 EOS の音速テスト（`test_hydro_table_eos`）の期待値を、2026-07-26 の解析的局所微分（log 双線形補間の微分は格子節点で 1 次誤差）に合わせて更新（節点 5 %、log 中点 1 % の 2 段検査）。実装は不変。
- **IMC 用の Fleck 因子下限による Δt 制約（`Numerics.dt.f_min_fleck`）が FLD / S_N の run にも掛かっていた。** NUMERICS §2.2 (c) の根拠は Monte Carlo の分散悪化で、FLD 側は f に下限を設けない仕様なのに、駆動側は輻射が有効なら常にこの Δt を評価していました。表 EOS の電子熱容量が表の床にある冷たく光学的に厚いセル（SESAME の液体 D2、25 meV）では β が発散して Δt が 1e-21 s に落ち、計算が進まなくなります（NIF DS デッキ、Tr = 55 eV）。`Radiation.mode = imc_ddmc` 以外では評価しないようにしました（persistent loop の複製も同じ）。1D FLD の既定経路（gxii / cbet の回帰、Marshak 波、灰色 FLD の輻射衝撃波・球面 5 % 摂動、Hammer–Rosen）は状態量が bit 一致で、変わるのは履歴ファイルの診断列 `diagnostics/dt_breakdown_history/dt_rad`（候補値が +∞ になる）だけです。
- gxii 1D FLD 回帰の golden を、文書化済みの既定変更（Langdon 既定 ON など）の後に再基準化。

### 機能追加

- **1D FLD 外側反復の Anderson 加速。** `Radiation.multigroup_diffusion.outer_accel="anderson"`（`anderson_m`, `anderson_beta`）を 2D_RZ から 1D_SPH へ移植（`src/radiation/fld_anderson.cuh`、有効時は外側ループのパイプライン化を使わない）。既定 `"none"` は従来どおり。Planck 平均不透明度の温度依存が急な冷たいセルで逐次代入が 2 周期に落ちる場合の対策で、NIF DS デッキ 3 ns（1572 サイクル）の未収束を 0 にしました。単体テスト `test_fld_anderson`。
- **`Materials.materials[].opacity.tmat_kirchhoff_pe`（既定 False）。** True で TMAT 読み込み時に全ノード・全群で \(\kappa_{PE}:=\kappa_{PA}\)（Kirchhoff の法則）。PROPACEOS 由来の表は各群の Wien 裾で \(\kappa_{PE}\) が 1e-99 へアンダーフローしたり \(\kappa_{PA}\) の 10³ 倍になったりして Fleck 因子が数 meV で桁跳びし、外側反復が収束しません。LTE 表ではこれを True にします（`is_lte` 属性は変えず値だけ置換）。
- `Numerics.hydro.qei_heat_capacity`（既定 `"ideal_gas"` = 従来の算術、`"table"` = 表の \(c_{v,e}, c_{v,i}\) で Qei を計量）。表の低温電子比熱が理想値より桁で小さいとき、理想計量の Qei が \(T_e\) を過大に動かして振動する問題への対応。
- `Numerics.hydro.T_start_inactive_cells="rigid_wall"`（既定 `"passive_fill"` は不変）: セル単位の hydro 開始温度で非活性のセルを剛体壁として扱う変種。非活性セルはプラズマ端と一緒に平行移動しません。
- **張力カットオフ `Numerics.hydro.pressure_tension_cutoff`（既定 False）。** True で 1D の各 EOS 閉包の直後に、全圧 \(P_e+P_i\) が `pressure_tension_cutoff_value`（既定 0、\(\le0\) [dyn/cm²]）を下回るセルの両種の圧力を同じ比で縮めて全圧をその値にします（値 0 なら両種とも 0 で、床付き相では pdV 仕事をしません）。流体は表 EOS の cold curve の張力を支えられません — PROPACEOS 由来の液体 D2 表は初期状態（0.17 g/cc、1 meV）で −7 kbar かつ \(\partial P/\partial\rho|_T<0\) で、D2/CH 界面の膨張と燃料内部の密度波（丸め誤差の指数成長）を駆動していました。温度・エネルギーは変えず、力・仕事・人工粘性は同じ床付き圧力を読みます。NIF DS デッキの 3 ns 検証で界面変位 0.044 µm → 0、燃料内部の密度摂動なし。単体テスト `test_hydro_1d_step`（tension ケース）。
- **起動時診断: 初期状態の力学的安定性。** 各セルの支配材料のイオン表+電子表から \(\partial P_{\rm tot}/\partial\rho|_T\) を評価し、非正（スピノーダル、または非物理的な cold curve）のセル数・\(\rho, T_e\) の範囲・最小値を WARNING で報告します（状態は変えません）。床温度に固定されたセルの応答は等温で、そこでは Lagrange 格子が丸め誤差を指数的に増幅します（NUMERICS §3）。
- 1D FLD: 放射エネルギーの保存的メッシュ移流 `hydro_coupling="conservative_advection"`（opt-in、1D Lagrangian のホスト駆動ループ）。
- 等質量の自動分割 `Mesh.auto_regions`（等厚の区画を等セル数で分割し、材料境界で質量比を合わせる）。NIF 直接駆動の設計スタディ（`examples/nifds/`）は開発用ワークツリーのパスと未同梱の表に依存するため、このスナップショットには含めません。
- Studio: ヘッダ無し手書き Python デッキの取り込み（⌘O、作業ディレクトリ・環境・サーバ側実行の設定を記録、失敗時に設定を表示）、サーバのファイルブラウザに上の階層へ戻るボタン。
- TMAT: 検証済み wide D2 表のデータ監査（`tools/tmat`; 表本体は同梱しません）。

### 性能

- 1D のデバイス常駐化: ホスト側で評価していた物理（レーザー入射、伝導の再閉包、表電離、放射の投機的スナップショット）をデバイスへ移し、毎ステップの D2H を減らし、history をデータセット単位で一括書き込み。実験デッキで 6.5 ns 実行が −22.6 %、GXII −50 %、Noh −79 %（RTX 4090）。opt-in の NVTX 範囲でステップ内の遅延を帰属できます。

### ドキュメント・検証系

- NUMERICS: 符号付き表エネルギーの許容域（§2 追補）、多材料閉包の支配材料表、Fleck 再放射項の訂正と TMAT 表の \(\kappa_{PE}\) 異常（§6.7 注記）、自由境界 ghost の復元（§3）、初期状態の力学的安定性（§3）、D2 表とメッシュの保存監査。SPECIFICATION §6.4／§9.1 に新キー。
- テスト: 上記の各修正に単体テストを追加（Fleck blend、Anderson、TMAT kirchhoff、多材料の再閉包、伝導増分の恒等性、sub-floor 閉包の拒否、Qei の上限と multiplier、燃焼ゼロ注入の恒等性、剛体壁の非活性セル）。`test_hydro_1d_step` と `test_hydro_table_eos` は全件合格に復旧。
- 格子収束キャンペーン（2026-09-03/04）: 参照表の来歴と較正キーを追記し、要求モデルの既定をキャンペーン結果から較正。

## 2026-09-03

### 機能追加

- **物理由来の初期メッシュ分解能要求（1D、実験的）。** `Mesh.resolution_requirement` により、デッキのレーザー波形・波長・材料層・幾何からアブレート帯の面密度質量天井プロファイル・衝撃波分離天井・層あたり最小セル数を決定論的に見積もり、`validate` と run 開始時に判定します（`apply="report"` 既定はメッシュ不変、`apply="enforce"` は `zoning_intent` に推奨帯を注入／他形式は違反時に拒否）。run 出力に `mesh_requirement.json`、`validate --mesh-preview` に全 1D 形式の節点列と要求・判定を追加。アシスタントの `lint-deck` は要求違反を hard lint として反復フィードバックし、`generate-deck` はプロンプトに要求を提示、`zoning-report` は probe 実測との比を報告します（NUMERICS §3.1.0c、設計 docs/design/mesh_resolution_requirement_20260903.md）。
- **格子収束の実測キャンペーンと較正。** `tools/validation/mesh_convergence_campaign.py`（29 ケース×表面面密度質量の梯子、観測量抽出と収束判定、参照表出力）を追加し、その結果（`docs/validation/mesh_convergence_reference.md`、`tools/assist/data/mesh_convergence_reference.json`）から `Mesh.resolution_requirement` の既定を較正（`zones_per_scale_length` 9、強度補正 `intensity_exponent` 0.4）。
- **`dr_min` と要求天井の衝突検査。** 要求 JSON に帯ごとの `width_max_cm` と `ablation.dr_min_admissible_cm`（注入天井と両立する最大の `dr_min`）を追加。`apply="enforce"` で `zoning_intent.dr_min` がこれを超える場合は求解前に `MESH_RESOLUTION_REQUIREMENT_DR_MIN_CONFLICT` で拒否します（メッセージに許容値を明記）。アシスタントの要求サマリ・生成ガイド・スキルにも反映。
- `tools/assist`: new `docmap` verb (deterministic document map + namelist key index from the `enforce_known_keys` lists in `src/core/namelist/builder.cpp`) and `ask` verb (question answering about TENRYU from the checkout's documents and source, with automatic checks that cited paths and namelist keys exist). New role `question_answering`; read-only provider examples in `tools/assist/assistant.example.toml`. Each user runs it with their own CLI login or API key.
- Skill `tenryu-docs-qa` (`tools/assist/skills/codex/`, `.claude/skills/`) — the retrieval guidance used as the prompt head.
- Studio: question panel in the assistant view (local `assist.py ask --json`, journal progress, evidence check display).
- Export now includes `docs/site/`, `docs/OUTPUT_SCHEMA.md`, `docs/POSTPROCESSING.md`, and `.claude/skills/tenryu-docs-qa/`.

## 2026-09-02

### 機能追加

- **TENRYU Studio に LLM アシスタント統合 (実験的)。** 左ナビの新ビュー
  「アシスタント」から、同梱の実験的アシスタントハーネス `tools/assist/` を
  GUI で操作できるようになりました。
  - **デッキ生成 (LLM)**: 自然言語の仕様から 1D デッキを生成し、サーバーの
    バイナリでの validate/freeze と lint フィードバックを反復します
    (`generate-deck` ループ)。モデルが推測せず質問した場合 (UNCERTAIN) は
    回答して再生成できます。受理デッキはその場で検証・保存・実行可能で、
    実行は通常の実行履歴に入ります。生成はローカルマシンで実行され
    (プロバイダ CLI とその認証はローカル前提)、デッキ検証だけを
    `tools/assist/tenryu_remote.sh` 経由でサーバーに委ねます。既定は無効
    (assistant.toml で opt-in、`TENRYU_ASSIST_DISABLE=1` が常勝の kill
    switch)。全 LLM 呼び出しは `~/.tenryu/studio-assist/<stamp>/journal.jsonl`
    に記録されます。
  - **詳細 Lint**: 検証パネルから、現在のフォームデッキに対しメッシュ・
    ゾーニング系 lint (節点単調性・隣接幅比・autozone 質量比・intent ピン)
    を実行できます。
  - **実行診断**: 実行履歴の終了 run から、ダイジェスト (履歴系列の縮約・
    派生指標)・ゾーニング診断 (アブレート帯の質量形 lint)・ゾーニング昇格案
    を表示できます (サーバー側に python3 と tools/assist を含む
    チェックアウトが必要。ダイジェストの系列縮約には h5py)。
  - これに伴い Studio の Tauri shell 許可へ ssh/scp と並んで bash
    (ローカル実行) が追加されています。GUI マニュアル (gui/manual/) に
    「アシスタント」章を追加しました。
- `tools/assist/tenryu_remote.sh` が `TENRYU_REMOTE_SSH_OPTS` /
  `TENRYU_REMOTE_SCP_OPTS` (および標準の `RSYNC_RSH`) でポート・鍵指定等の
  接続オプションを受け取れるようになりました。未設定時の挙動は不変です。

### 不具合修正

- **配布物に `gui-common/` が含まれず GUI がビルド不能だった問題を修正。**
  Studio (gui/) は共有パッケージ `@tenryu-common` (gui-common/) を参照します
  が、これまでのスナップショットには同梱されていませんでした。今回から
  `gui-common/` と `ops/gui/` (実行スクリプトの同期テストが参照) を同梱
  します。

## 2026-08-31

### 不具合修正

- **FLD の真空拡散極限で行列が発散し NaN を生成する問題を修正。**
  `radiation.multigroup_diffusion.opacity_floor` の既定値を
  `1.0e-100` から `1.0e-6` /cm（平均自由行程 10 km）に引き上げました。
  void セル（一様放射場かつ不透明度がゼロ近傍）では flux limiter が
  拡散係数の発散を止められず、三重対角行列の係数が ~1e98 に達して
  ソルバが NaN を生成し、その NaN が下限クランプで黙って床値化されて
  長時間潜伏したのち流体セル反転として噴出していました。勾配領域では
  limiter により床は物理へ影響しません（Su-Olson 等の光学的に厚い
  検証系はビット一致のまま）。
- **多材料デッキで 2 番目以降の材料の不透明度指定が無視される問題を修正。**
  1D FLD は先頭の非 void 材料の κ を全セルに適用していました。セル毎の
  体積分率混合（2D_RZ と同方式）を 1D にも配線し、あわせて namelist
  検証を強化しました（1D 多材料×放射で不正な不透明度モデルは
  ConfigError になります）。
- **レーザー沈着の会計不整合を修正。** 全節点が臨界密度超のレーザー
  メッシュセルで吸収電力が計上のみされ沈着から漏れていた問題
  （最近接の臨界未満セルへ再配分するようフォールバックを追加）、
  および履歴データセット `energy/radiation_escaped` が累積値ではなく
  ステップ値を記録していた問題（累積化。ステップ値は
  `energy/radiation_escaped_step` に保存）を修正しました。
- **FLD ソルバ出力の非有限値検出を常設化。** ソルバ解の NaN/Inf を
  発生ステップで計数し、`Numerics.safety.nan_fatal=True`（既定）なら
  即時停止します。従来は下限クランプで黙って洗浄されていました。
- S_N の拡散形分母（DSA 前処理・AP ブレンド面流束）に 1e-6 /cm の
  正則化を追加しました。評価下限（`sn_transport.opacity_floor`）は
  減衰検証を壊さないため 1e-100 のままです。

### 機能追加

- **多材料デッキの材料別テーブル不透明度（1D FLD）。**
  各非 void 材料が自分の tmat テーブル（LTE / NLTE いずれも）または
  定数 κ を持てるようになりました。純セルは支配材料（体積分率最大）の
  テーブルを (ρ, T_e) 補間で評価し、**混合セルは寄与材料すべてを厳密
  合成します** — 質量分率重み w_m（無ければ体積分率）で
  σ = ρ Σ w_m κ_m、各材料のテーブルはその部分密度 ρ_m = ρ w_m/f_m で
  評価。NLTE テーブルは放射率≠吸収率を含めて材料別に扱われます
  （NLTE が支配のセルは支配材料近似）。恒久回帰テスト
  `test_fld_multimat_opacity` を同梱（テーブル入替 A/B・定数経路整合・
  混合セル厳密値）。
- **1D レーザープラズマ例題集 `examples/laser_plasma_1d/`（10 題）。**
  箔の衝撃波突破・二段パルス・インピーダンス整合・爆発箔・Kr 中の
  放射性衝撃波・円筒ライナー・中実球収束（3 解像度の収束スタディ付き）・
  D2 充填シェル爆縮・CBET+ホット電子コロナ・Marshak 駆動 DT 燃焼。
  各デッキに実測のコミッショニング指標と注意点を README に記載。
  ※ `.tmat.h5` テーブルはライセンス由来データのため同梱していません。
  本例題集の全デッキは固体材料の EOS に tmat テーブルを用いるため、
  実行には PrOpacEOS/SESAME からのテーブル生成が必要です
  （`tools/tmat/propaceos_to_tmat.py` 参照）。
- S_N 1D の外部体積源（`Radiation.volume_source_rate`）を検証完了し
  実験扱いを解除しました。物質側エネルギー残差に注入項が欠けて
  線源セルの電子温度が床に張り付く欠陥を修正し、独立な S₈ 離散化
  参照（`tools/su_olson_sn_reference.py`、同梱）と 0.2%/5.3%/13.9% で
  一致します。あわせて S_N の外側反復が物質基準を
  反復値に鎖結して弱結合の放射-物質交換を反復回数倍に過大適用する欠陥
  （体積源の反復毎再注入を含む）を、基準の step 開始固定で修正しました。
- レーザー転送の保存監査（環境変数 `TENRYU_LASER_TRANSFER_AUDIT=1`）と
  FLD ソルバ健全性検査（`TENRYU_FLD_SOLVE_CHECK=1`）を追加しました。

### ドキュメント

- SNB 非局所電子熱輸送の適用範囲記述を実装に一致させました
  （1D_SPH と 2D_RZ の両方で利用可能です。旧記述の「1D_SPH は
  ConfigError」はブランチ統合前の状態でした）。ほか、多材料テーブル
  不透明度まわりの実装状態注記を各所で現行化しました。

### 検証系

- 放射衝撃波検証ハーネス（`tools/validation/run_i1_fld_ced.py`）を、
  格子毎の定常相 run から前縁メトリクスを採る方式に修正しました。
  従来は自由境界からの擾乱が前縁へ到達した後の非定常状態を測って
  恒常不合格になっていました（前縁ピークの収縮判定も、固定の
  非平滑参照ピークに対する判定へ再定義）。
- `verify su_olson` を再基線しました。旧参照は退役した Monte Carlo
  モード用の離散座標輸送解で、現行の既定 FLD とは比較対象が不一致
  でした。デッキを FLD（limiter 無効 pin）へ現代化し、同一モデルの
  独立差分解（`tools/su_olson_fld_reference.py`）を参照とする実装検証
  ゲートとして PASS します。
- `verify marshak` のデッキと前提条件を現行 namelist 仕様に更新しました
  （ゲート自体は周波数連続輸送ベンチマークのため、多群 S_N 移植までは
  不合格を許容として文書化）。
- レーザー付き FLD 生産ゲート（I1 radial）を会計修正の上で PASS に
  戻しました。

## 2026-08-28

- `tenryu_lib` をテスト同梱ビルドのときのみ生成するようゲートしました
  （スナップショット構成では 264MB の共有ライブラリを作りません）。
- BUILD.md: CMake 要件 3.24→3.27、GPU アーキテクチャ自動検出の説明を
  追記しました。
- 出荷ヘッダ 1 件の内部文書引用を平易な記述に置き換えました。

## 2026-08-26

- 初回公開スナップショット（ソース+GUI+docs 一式、160 ファイル）。
