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
