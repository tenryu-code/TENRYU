import { en } from "./en";
import { ja, type Messages } from "./ja";

export type Lang = "ja" | "en";

let currentLang: Lang = "ja";
let current: Messages = ja;

export function setLang(lang: Lang): void {
  currentLang = lang;
  current = lang === "en" ? en : ja;
}

export function getLang(): Lang {
  return currentLang;
}

export function t(): Messages {
  return current;
}
