# TENRYU

TENRYU は、レーザー駆動の高エネルギー密度（HED）物理と慣性閉じ込め核融合（ICF）を対象とする、CUDA ネイティブのラグランジュ/ALE 輻射流体コードです。1D（球・円筒・平面）と 2D 軸対称（RZ）の形状を扱い、すべての計算を GPU 上で行います。入力デッキの作成から実行、結果の確認までを行う macOS アプリ「TENRYU Studio」のソースも含みます。

このリポジトリは、TENRYU の β 版（評価版）の配布用スナップショットです。

- **ドキュメント（公開サイト）**: https://tenryu-code.github.io/TENRYU/ （[日本語](https://tenryu-code.github.io/TENRYU/ja/) / [English](https://tenryu-code.github.io/TENRYU/en/)）
- **TENRYU Studio マニュアル**: [日本語](https://tenryu-code.github.io/TENRYU/gui/manual/index_ja.html) / [English](https://tenryu-code.github.io/TENRYU/gui/manual/index_en.html)
- **ビルド手順**: [BUILD.md](BUILD.md)
- **更新履歴**: [CHANGELOG.md](CHANGELOG.md)
- **不具合報告・問い合わせ**: [GitHub Issues](https://github.com/tenryu-code/TENRYU/issues)
- **ライセンス**: [LICENSE](LICENSE)（ソース公開。学術・非商用の利用は無償）、β の利用条件は [EULA.md](EULA.md)

> **English** — TENRYU is a CUDA-native Lagrangian/ALE radiation-hydrodynamics code for laser-driven high-energy-density (HED) and inertial-confinement-fusion (ICF) physics, in 1D spherical, cylindrical, or planar and 2D axisymmetric (RZ) geometry. This repository is a beta snapshot; the beta covers the 1D code and the TENRYU Studio GUI for macOS. English documentation: https://tenryu-code.github.io/TENRYU/en/ — start with [Building](https://tenryu-code.github.io/TENRYU/en/use/building.html) and [Running simulations](https://tenryu-code.github.io/TENRYU/en/use/running.html). TENRYU is source-available, not open source: running it for academic and other non-commercial purposes is free, while redistribution and commercial use are restricted (see [LICENSE](LICENSE)). Questions and reports go to [GitHub Issues](https://github.com/tenryu-code/TENRYU/issues).

## β 版について

- β のサポート対象は、1D（球・円筒・平面）のラグランジュ輻射流体計算と TENRYU Studio です。評価とフィードバックは、この 2 つを中心にお願いします。
- 2D 軸対称（RZ）のコードも含まれますが、β のサポート対象外です。
- 計算コアと Studio の仕様は、β 期間中に変わることがあります。変更は [CHANGELOG.md](CHANGELOG.md) に記録します。

## 主な機能

| 分野 | 内容 |
|---|---|
| 流体 | エネルギー整合なスタッガード格子のラグランジュ流体と衝撃波の捕捉。2D では ALE のリゾーンと保存型のリマップ |
| 輻射輸送 | 多群の流束制限拡散（FLD）と離散座標法（S<sub>N</sub>） |
| 熱伝導 | 電子・イオン熱伝導（局所の Spitzer–Härm と非局所の SNB） |
| レーザー | 屈折を含む光線追跡と逆制動輻射によるエネルギー付与、交差ビームエネルギー転送（CBET）、ホット電子の生成と輸送 |
| 物性 | 解析的または表形式の EOS・不透明度・電離 |
| 核燃焼 | 核融合反応、荷電粒子へのエネルギーの配分、中性子加熱 |
| プラズマ粘性 | Braginskii 型の運動量拡散 |
| 入出力 | 入力は Python の namelist デッキ、出力は HDF5（スナップショット、時系列の履歴、チェックポイント）。終えた run は、デッキを変えずに終了時刻を延ばして続けられます |
| 実行 | すべての計算を GPU（CUDA）で行います。単位系は cgs + eV に固定です。複数の GPU は MPI で使えます |

機能ごとの対応する次元と、既定で有効かどうかは、サイトの[物理カーネル](https://tenryu-code.github.io/TENRYU/ja/physics/)の早見表にあります。

## 動作環境

**計算サーバ**（ビルドと計算を行うマシン）

| 項目 | 要件 |
|---|---|
| OS | Linux x86_64 |
| GPU | NVIDIA（継続して検証している実機は A100 40/80 GB、RTX 4090、RTX 3090） |
| CUDA Toolkit | 12.x（配布物のビルドは 12.6 で確認） |
| C++ コンパイラ | GCC 12 以上 |
| ビルドツール | CMake 3.27 以上、Ninja |
| Python | 3.10 以上（開発用ヘッダを含む）と pybind11 |
| HDF5 | 1.12 以上 |

**TENRYU Studio**: macOS。計算サーバへ、パスワードの入力なしに（鍵認証で）`ssh` で接続できること。Studio は対話の確認ができないため、計算サーバのホスト鍵も事前に登録しておきます。アプリのビルドには Node.js と Rust のツールチェーンが要ります。

詳しい要件と事前チェックのコマンドは、サイトの[動作環境と要求スペック](https://tenryu-code.github.io/TENRYU/ja/use/requirements.html)を参照してください。

## クイックスタート

計算サーバで次を実行します。詳しい手順とトラブルシュートは [BUILD.md](BUILD.md) にあります。

```bash
git clone https://github.com/tenryu-code/TENRYU.git
cd TENRYU
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(which python3)
ninja -C build tenryu
./build/tenryu run examples/verification/sod_planar.py   # 動作確認（数秒で終わります）
```

コンパイルが `Killed` で止まるときはメモリ不足です。`ninja -C build -j 8 tenryu` のように同時実行数を減らしてください。

続けて、編集せずにそのまま完走するテンプレートで最初の計算をし、結果を図にします。

```bash
./build/tenryu validate examples/templates/template_1d_slab_radiation.py   # 実行前の検査
./build/tenryu run      examples/templates/template_1d_slab_radiation.py
python3 -m tools.tenryu_plot profile outputs/template_1d_slab_radiation   # numpy・matplotlib・h5py を使います
```

## コマンドライン

| コマンド | 内容 |
|---|---|
| `tenryu validate <deck.py>` | 入力デッキを検査します（計算はしません） |
| `tenryu run <deck.py>` | 計算を実行します |
| `tenryu run <deck.py> --restart <checkpoint>` | チェックポイントから再開します。デッキは元の run と同じものを使います |
| `tenryu run <deck.py> --restart <checkpoint> --t-end <s>` | 終了時刻を延ばして続けます（step の上限は `--max-steps <N>`） |
| `tenryu verify <name>` | 登録済みの検証問題を実行します |
| `tenryu freeze <deck.py> -o <config.json>` | デッキを解決した設定を JSON に書き出します |

デッキの書き方と出力の読み方は、サイトの[シミュレーションの実行](https://tenryu-code.github.io/TENRYU/ja/use/running.html)、[namelist リファレンス](https://tenryu-code.github.io/TENRYU/ja/use/namelist.html)、[出力と解析](https://tenryu-code.github.io/TENRYU/ja/use/outputs.html)を参照してください。

## TENRYU Studio（GUI）

Studio は macOS のデスクトップアプリです。フォームで入力デッキを作って検証し、計算サーバへ SSH で送って実行し、計算の進み具合と結果（時系列の履歴とプロファイル）を確認できます。保存したデッキは Studio で読み込み直して編集を続けられ、終えた run を延ばして続けることもできます。

Studio のソースは [gui/](gui/) にあります（アプリ単体の配布はしていません）。Mac で次のようにビルドします。

```bash
cd gui
npm ci
npm run tauri build
# gui/src-tauri/target/release/bundle/macos/TENRYU Studio.app ができます
```

使い始めの手順:

1. 計算サーバで TENRYU をビルドします（[クイックスタート](#クイックスタート)）。
2. Studio を起動し、計算サーバ（`ssh` の接続先）を登録します。
3. サーバの設定に、ビルドした `build/tenryu` の絶対パスと、実行ディレクトリ（接続するユーザーが書き込める任意のパス）を指定します。
4. 1D のプリセットを選び、条件を設定して実行します。初めは標準のプリセットから始めることを推奨します。

詳しくは [TENRYU Studio マニュアル](https://tenryu-code.github.io/TENRYU/gui/manual/index_ja.html)とサイトの [TENRYU Studio の頁](https://tenryu-code.github.io/TENRYU/ja/use/gui.html)を参照してください。

### macOS Gatekeeper について

Studio のアプリは、Apple の Developer ID による署名と公証を受けていません。ほかの人がビルドしたアプリを受け取って初めて起動するとき、macOS に起動を拒否されたら、Finder でアプリを右クリックして「開く」を選んでください。必要なら、ターミナルから次のコマンドで隔離属性を外せます。入手元と配置先を確かめてから実行してください。

```bash
xattr -cr "TENRYU Studio.app"
```

## 例題

| ディレクトリ | 内容 |
|---|---|
| [examples/templates/](examples/templates/) | 編集せずにそのまま完走する 1D のテンプレート（平板の輻射、レーザーで照射する球、間接照射） |
| [examples/laser_plasma_1d/](examples/laser_plasma_1d/) | 典型的なレーザープラズマ実験を模した 1D の例題 10 本（説明は同じディレクトリの README） |
| [examples/implosion/](examples/implosion/) | 激光 XII 号（GXII、大阪大学レーザー科学研究所の 12 ビームレーザー）の爆縮を模した 1D デッキ |
| [examples/verification/](examples/verification/) | 解析解や基準解と比べる検証用のデッキ |

一覧と使い方は、サイトの[例題とプリセット](https://tenryu-code.github.io/TENRYU/ja/use/examples.html)を参照してください。

## ドキュメント

公開サイト https://tenryu-code.github.io/TENRYU/ の構成:

| 節 | 日本語 | English | 内容 |
|---|---|---|---|
| 概要 | [概要](https://tenryu-code.github.io/TENRYU/ja/overview/) | [Overview](https://tenryu-code.github.io/TENRYU/en/overview/) | 対象範囲、アーキテクチャ、数値基盤、検証の考え方 |
| 使い方 | [TENRYU を使う](https://tenryu-code.github.io/TENRYU/ja/use/) | [Using TENRYU](https://tenryu-code.github.io/TENRYU/en/use/) | 要求環境、ビルド、実行、namelist、GUI、例題、出力 |
| 物理 | [物理カーネル](https://tenryu-code.github.io/TENRYU/ja/physics/) | [Physics kernels](https://tenryu-code.github.io/TENRYU/en/physics/) | 各物理のモデル、離散化、制御項目、検証 |
| 検証 | [検証とベンチマーク](https://tenryu-code.github.io/TENRYU/ja/verification/) | [Verification & benchmarks](https://tenryu-code.github.io/TENRYU/en/verification/) | 解析解による検査、基準解との回帰、ベンチマーク |
| 開発者向け | [開発者ガイド](https://tenryu-code.github.io/TENRYU/ja/develop/) | [Developer guide](https://tenryu-code.github.io/TENRYU/en/develop/) | 開発の規則、テスト、再現性 |
| 用語 | [記号・用語集](https://tenryu-code.github.io/TENRYU/ja/overview/nomenclature.html) | [Nomenclature](https://tenryu-code.github.io/TENRYU/en/overview/nomenclature.html) | 略語、記号、バージョンの表記 |

リポジトリ内の文書:

| ファイル | 内容 |
|---|---|
| [docs/TUTORIAL_ja.md](docs/TUTORIAL_ja.md) | 初めて使う人向けのチュートリアル（日本語） |
| [docs/SPECIFICATION.md](docs/SPECIFICATION.md) | 仕様（入力の namelist の全キー、出力、既定値、再開） |
| [docs/NUMERICS.md](docs/NUMERICS.md)、[docs/sections/](docs/sections/) | 数値解法（式と離散化） |
| [docs/OUTPUT_SCHEMA.md](docs/OUTPUT_SCHEMA.md) | HDF5 出力の構成 |
| [docs/POSTPROCESSING.md](docs/POSTPROCESSING.md) | 作図ツール `tools/tenryu_plot` の使い方 |

## リポジトリの構成

| パス | 内容 |
|---|---|
| `src/` | 計算コア（C++20 / CUDA） |
| `examples/` | 入力デッキの例 |
| `tools/` | 作図（`tenryu_plot`）、格子の設計（`mesh_planner.py`）、質問応答とデッキ作成の補助（`assist/`）などのツール |
| `gui/`、`gui-common/` | TENRYU Studio のソース（Tauri 2 + React/TypeScript） |
| `docs/` | 仕様・数値解法・出力の文書と、公開サイトのソース（`docs/site/`） |
| `CMakeLists.txt`、`cmake/` | ビルドの設定 |
| `ops/gui/` | Studio が計算サーバで計算を起動するスクリプト（アプリにも同じものが組み込まれています） |
| `retired/` | 退役したモンテカルロ輻射のコードと文書（ビルドには含まれません） |
| `.claude/skills/` | Claude Code 用のスキル（下記） |

## LLM による補助（実験的）

`tools/assist/` は、TENRYU についての質問応答や入力デッキの作成を LLM で補助する実験的なツールです。質問応答（`ask`）は、このチェックアウトの文書とソースを根拠に答えます。LLM には、利用者自身の Claude Code または Codex CLI のログイン（または API キー）を使います。

```bash
mkdir -p ~/.tenryu
cp tools/assist/assistant.example.toml ~/.tenryu/assistant.toml   # enabled = true に書き換える
python3 tools/assist/assist.py ask "質問"
```

- 質問応答に使う設定は `question_answering` の役割で、`claude_readonly` か `codex_readonly` を指定します（例の設定は `claude_readonly`）。
- 回答の末尾には、このチェックアウトと照合した `path:line` の引用が付きます。詳しくは [tools/assist/README.md](tools/assist/README.md) を参照してください。
- 使う側のマシンには `python3`（macOS では Xcode Command Line Tools）と、Claude Code か Codex CLI が要ります。
- Studio のアシスタントの画面からも同じ機能を使えます。Studio は、手元にチェックアウトを置かなくても、サーバのチェックアウトをプロファイルの SSH 接続で同期して使います（「文書とソースの取得元」→「サーバーから同期」）。

このリポジトリを Claude Code で開くと、`.claude/skills/` の次のスキルも使えます。

| スキル | 内容 |
|---|---|
| `tenryu-namelist` | 実験の条件から 1D の入力デッキを書き、ソルバーで検証します |
| `tenryu-mesh-1d` | 実験の条件から 1D の初期格子を設計します（格子収束の測定に基づく推薦） |
| `tenryu-docs-qa` | TENRYU についての質問に、文書とソースを根拠に答えます（読み取りのみ） |

## 既知の制限

- β のサポート対象は 1D と Studio です。2D 軸対称（RZ）のコードは含まれていますが、サポート対象外です。
- Studio の `polar_in_box`（2D の格子）は、初期格子のプレビューまでで、実行はできません。
- Studio のアプリは Apple の署名と公証を受けていないため、受け取ったアプリの初回の起動に手作業が要ります（[macOS Gatekeeper について](#macos-gatekeeper-について)）。
- 計算コアと Studio の仕様は、β 期間中に変わることがあります。

## フィードバック

不具合や改善の提案は [GitHub Issues](https://github.com/tenryu-code/TENRYU/issues) へお寄せください。不具合の報告には、現象、期待した動作、実行環境（GPU、CUDA、OS）、ログ、再現できる入力デッキ（`.py`）を添えてください。Studio で作ったデッキは、namelist の保存機能で `.py` に書き出せます。

機密情報、認証情報、個人情報を Issue やデッキに含めないでください。

## ライセンスと引用

- TENRYU はソース公開（source-available）のソフトウェアで、OSI の定義によるオープンソースではありません。学術目的とその他の非商用目的の実行は無償です。所属組織の中で自分の研究のために改変することは、許可なくできます。著作権者の書面による許可なく、コードを再配布すること（公開のフォークやミラーを含む）と、改変版を公開することはできません。営利企業での利用（社内の研究開発を含む）は商用利用で、有償のライセンス契約が要ります。条文は [LICENSE](LICENSE) にあります（英語が正文で、日本語の参考訳付き）。
- β 版の利用条件は [EULA.md](EULA.md)（草案）にあります。
- TENRYU を使った成果を公表するときは、EULA.md の第 3 項に従って事前に開発元へ連絡し、[LICENSE](LICENSE) の第 4 節の例のように謝辞で TENRYU に触れてください。
- 許諾と商用ライセンスの問い合わせは [GitHub Issues](https://github.com/tenryu-code/TENRYU/issues) で受け付けています。
