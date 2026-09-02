# TENRYU Studio (gui/)

TENRYU の namelist(deck) 作成とリモート実行を行う Mac GUI。
Tauri 2 + React/TypeScript の Web コア分離型 (設計は
`docs/design/gui_mac_namelist_runner_plan_20260711.md` と同 Addendum 1 が正典)。

- Web コア (`src/`) は純 Web — Linux で `npm run dev` / `npm test` 可能。
- ssh はアプリに実装しない: システム `ssh`/`scp` を Tauri plugin-shell (Mac 本番) /
  dev-bridge (`dev-bridge/server.mjs`, Linux 開発) から spawn する。
- Tauri シェル (`src-tauri/`) のビルドは Mac で行う (Linux dev 環境には
  webkit2gtk が無い)。手順は M5 の Mac ビルド手順書を参照。

## 開発 (Linux)

```bash
npm install
npm test               # vitest (deck 生成器 golden 含む)
npm run bridge &       # dev ブリッジ (127.0.0.1:5175)
npm run dev            # Vite (http://localhost:5173, /api を bridge へ proxy)
```

## アシスタント (実験的)

左ナビ「アシスタント」から tools/assist (リポジトリ同梱の LLM ハーネス) を GUI で操作できる。

- 決定論 verb (詳細 Lint / ダイジェスト / ゾーニング診断) はサーバープロファイル上で
  `python3 tools/assist/assist.py …` を実行する (サーバー側チェックアウトに tools/assist が必要)。
- LLM 生成 (generate-deck) はローカルで実行し、デッキ検証だけ
  `tools/assist/tenryu_remote.sh` 経由でサーバーのバイナリに委ねる。プロバイダ CLI と
  その認証はローカル前提。設定は assistant.toml (既定 OFF・TENRYU_ASSIST_DISABLE が
  kill switch)。作業ディレクトリは `~/.tenryu/studio-assist/<stamp>/`。
- このために Tauri shell 許可に bash が追加されている (ssh/scp と同格のローカル実行
  権限。webview は同梱コードのみを実行する)。

設計: docs/design/gui_assistant_integration_20260902.md
