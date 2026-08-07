export interface TranslatedError {
  ja: string | null;
  raw: string;
}

const RULES: Array<[RegExp, (m: RegExpExecArray) => string]> = [
  [/is not a supported key \(did you mean '([^']+)'\?\)/, (m) => `未対応のキー名です — もしかして '${m[1]}' ?`],
  [/is not a supported key/, () => "未対応のキー名です (綴りを確認)"],
  [/Python execution failed/, () => "deck (Python) の実行に失敗しました — 構文・キー名を確認"],
  [/Host key verification failed/, () => "ssh ホスト鍵の検証に失敗 — 一度ターミナルから ssh 接続して known_hosts に登録してください"],
  [/Permission denied/, () => "ssh 認証に失敗 — 鍵と ssh-agent (キーチェーン) を確認してください"],
  [/Connection refused/, () => "接続が拒否されました — ホスト名・ポート・サーバー稼働を確認"],
  [/Could not resolve hostname/, () => "ホスト名を解決できません — 綴りと ~/.ssh/config を確認"],
  [/(Operation|Connection) timed out/, () => "接続タイムアウト — ネットワーク・VPN・ProxyJump を確認"],
  [/No such file or directory/, () => "ファイル/ディレクトリが見つかりません — tenryu パスと実行ディレクトリを確認"],
  [/binary not executable/, () => "tenryu バイナリが実行できません — パスと実行権限を確認"],
];

/** Best-effort Japanese translation of a TENRYU / ssh error line (raw is always kept). */
export function translateError(raw: string): TranslatedError {
  for (const [re, fmt] of RULES) {
    const m = re.exec(raw);
    if (m) return { ja: fmt(m), raw };
  }
  return { ja: null, raw };
}
