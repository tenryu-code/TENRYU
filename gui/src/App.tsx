import { useEffect, useState } from "react";
import { isTauri } from "@tenryu-common/backend";
import { profileBinMissing } from "@tenryu-common/core/profiles";
import { isTerminal } from "@tenryu-common/core/runstate";
import { t } from "./i18n";
import { currentProfile, useApp, type SectionKey } from "./store";
import CommandPalette from "./ui/CommandPalette";
import DeckPreview from "./ui/DeckPreview";
import DeckView from "./ui/DeckView";
import HistoryView from "./ui/HistoryView";
import ServersView from "./ui/ServersView";
import ValidatePanel from "./ui/ValidatePanel";
import { Button, Select } from "@tenryu-common/ui/kit";
import { SplitPane, StatusBar, StatusItem } from "@tenryu-common/ui/shell";

const NAV_KEYS: SectionKey[] = ["presets", "basic", "materials", "mesh", "physics", "laser", "output"];

export default function App() {
  const m = t();
  const view = useApp((s) => s.view);
  const setView = useApp((s) => s.setView);
  const section = useApp((s) => s.section);
  const setSection = useApp((s) => s.setSection);
  const loadInitial = useApp((s) => s.loadInitial);
  const loadError = useApp((s) => s.loadError);
  const profiles = useApp((s) => s.profiles);
  const currentProfileId = useApp((s) => s.currentProfileId);
  const selectProfile = useApp((s) => s.selectProfile);
  const validating = useApp((s) => s.validating);
  const runValidate = useApp((s) => s.runValidate);
  const startRun = useApp((s) => s.startRun);
  const starting = useApp((s) => s.starting);
  const form = useApp((s) => s.form);
  const deckIoStatus = useApp((s) => s.deckIoStatus);
  const loadNamelist = useApp((s) => s.loadNamelist);
  const saveNamelist = useApp((s) => s.saveNamelist);
  const saveNamelistAs = useApp((s) => s.saveNamelistAs);
  const namelistPath = useApp((s) => s.namelistPath);
  const pollActiveRuns = useApp((s) => s.pollActiveRuns);
  const runs = useApp((s) => s.runs);
  const profile = useApp((s) => currentProfile(s));
  const lang = useApp((s) => s.lang);
  const setUiLang = useApp((s) => s.setUiLang);
  const [historyDeckOpen, setHistoryDeckOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const pibBlocked = form.main.dimension === "2D_RZ" && form.mesh.meshMode2d === "polar_in_box";
  const binMissing = profile !== null && profileBinMissing(profile);

  useEffect(() => {
    void loadInitial();
  }, [loadInitial]);

  useEffect(() => {
    if (!isTauri()) return;
    let disposed = false;
    void (async () => {
      const { Menu, Submenu, MenuItem, PredefinedMenuItem } = await import("@tauri-apps/api/menu");
      if (disposed) return;
      const mm = t();
      const appSub = await Submenu.new({
        text: "TENRYU Studio",
        items: [await PredefinedMenuItem.new({ item: "Quit" })],
      });
      const fileSub = await Submenu.new({
        text: mm.deckIo.menuFile,
        items: [
          await MenuItem.new({
            text: mm.deckIo.menuSave,
            accelerator: "CmdOrCtrl+S",
            action: () => { void useApp.getState().saveNamelist(); },
          }),
          await MenuItem.new({
            text: mm.deckIo.menuSaveAs,
            accelerator: "CmdOrCtrl+Shift+S",
            action: () => { void useApp.getState().saveNamelistAs(); },
          }),
          await MenuItem.new({
            text: mm.deckIo.menuLoad,
            accelerator: "CmdOrCtrl+O",
            action: () => { void useApp.getState().loadNamelist(); },
          }),
        ],
      });
      const editSub = await Submenu.new({
        text: mm.deckIo.menuEdit,
        items: [
          await PredefinedMenuItem.new({ item: "Undo" }),
          await PredefinedMenuItem.new({ item: "Redo" }),
          await PredefinedMenuItem.new({ item: "Separator" }),
          await PredefinedMenuItem.new({ item: "Cut" }),
          await PredefinedMenuItem.new({ item: "Copy" }),
          await PredefinedMenuItem.new({ item: "Paste" }),
          await PredefinedMenuItem.new({ item: "SelectAll" }),
        ],
      });
      const menu = await Menu.new({ items: [appSub, fileSub, editSub] });
      await menu.setAsAppMenu();
    })();
    return () => { disposed = true; };
  }, []);

  useEffect(() => {
    const h = setInterval(() => {
      void pollActiveRuns();
    }, 3000);
    return () => clearInterval(h);
  }, [pollActiveRuns]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
        return;
      }
      if ((e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey && e.key >= "1" && e.key <= "7") {
        e.preventDefault();
        const idx = Number(e.key) - 1;
        setView("form");
        setSection(NAV_KEYS[idx]);
        return;
      }
      if (!(e.metaKey || e.ctrlKey) || e.key !== "Enter") return;
      e.preventDefault();
      if (e.shiftKey) {
        if (pibBlocked || binMissing) return;
        void startRun();
      } else void runValidate();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [binMissing, pibBlocked, runValidate, setSection, setView, startRun]);

  return (
    <div className="grid h-full" style={{ gridTemplateRows: "44px auto minmax(0,1fr) 26px" }}>
      <header
        className="flex items-center gap-3 border-b px-4"
        style={{ borderColor: "var(--separator)", background: "var(--bg-panel)" }}
      >
        <span className="font-semibold">{m.app.title}</span>
        <div className="flex items-center gap-1">
          <Button onClick={() => void loadNamelist()}>{m.deckIo.btnLoad}</Button>
          <Button onClick={() => void saveNamelist()}>{m.deckIo.btnSave}</Button>
          <Button onClick={() => void saveNamelistAs()}>{m.deckIo.btnSaveAs}</Button>
        </div>
        {view === "history" && (
          <Button
            aria-pressed={historyDeckOpen}
            variant={historyDeckOpen ? "primary" : "secondary"}
            onClick={() => setHistoryDeckOpen((v) => !v)}
          >
            {m.run.showDeckPane}
          </Button>
        )}
        <div className="flex-1" />
        <Select
          value={lang}
          onChange={(e) => void setUiLang(e.target.value === "en" ? "en" : "ja")}
        >
          <option value="ja">{m.app.langNameJa}</option>
          <option value="en">{m.app.langNameEn}</option>
        </Select>
        <Select
          value={currentProfileId ?? ""}
          onChange={(e) => void selectProfile(e.target.value === "" ? null : e.target.value)}
        >
          <option value="">{m.server.noneSelected}</option>
          {profiles.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </Select>
        <Button title="⌘⏎ / Ctrl+Enter" onClick={() => void runValidate()} disabled={validating || !profile}>
          {validating ? m.validate.running : m.validate.run}
        </Button>
        <Button
          title={pibBlocked ? m.geo2d.pibRunGateNote : binMissing ? m.server.binMissing : "⇧⌘⏎ / Ctrl+Shift+Enter"}
          variant="primary"
          onClick={() => {
            if (pibBlocked || binMissing) return;
            void startRun();
          }}
          disabled={starting || !profile || pibBlocked || binMissing}
        >
          {starting ? m.run.starting : m.run.run}
        </Button>
      </header>

      {loadError ? (
        <div className="border-b px-4 py-1 text-xs" style={{ borderColor: "var(--separator)", color: "var(--err)" }}>
          {m.errors.bridgeDown} ({loadError})
        </div>
      ) : (
        <div />
      )}

      <div
        className="grid min-h-0"
        style={{
          gridTemplateColumns:
            view === "form" || (view === "history" && historyDeckOpen)
              ? "220px minmax(0,1fr) 440px"
              : "220px minmax(0,1fr)",
        }}
      >
        <nav
          className="flex flex-col gap-1 overflow-auto border-r p-3"
          style={{ borderColor: "var(--separator)" }}
        >
          {NAV_KEYS.map((k) => (
            <button
              key={k}
              onClick={() => setSection(k)}
              className="rounded px-2 py-1 text-left"
              style={{
                color: view === "form" && section === k ? "var(--fg)" : "var(--fg-secondary)",
                background: view === "form" && section === k ? "var(--bg-inset)" : "transparent",
              }}
            >
              {m.nav[k]}
            </button>
          ))}
          <div className="mt-2 border-t pt-2" style={{ borderColor: "var(--separator)" }}>
            <button
              onClick={() => setView("history")}
              className="w-full rounded px-2 py-1 text-left"
              style={{
                color: view === "history" ? "var(--fg)" : "var(--fg-secondary)",
                background: view === "history" ? "var(--bg-inset)" : "transparent",
              }}
            >
              {m.nav.history}
            </button>
            <button
              onClick={() => setView("servers")}
              className="w-full rounded px-2 py-1 text-left"
              style={{
                color: view === "servers" ? "var(--fg)" : "var(--fg-secondary)",
                background: view === "servers" ? "var(--bg-inset)" : "transparent",
              }}
            >
              {m.nav.servers}
            </button>
          </div>
        </nav>
        <main className="min-h-0 min-w-0 overflow-auto p-4">
          {view === "servers" ? <ServersView /> : view === "history" ? <HistoryView /> : <DeckView />}
        </main>
        {(view === "form" || (view === "history" && historyDeckOpen)) && (
          <aside
            className="min-h-0 min-w-0 overflow-hidden border-l"
            style={{ borderColor: "var(--separator)", background: "var(--bg-panel)" }}
          >
            {view === "form" ? (
              <SplitPane
                storageKey="studio.aside.split"
                direction="column"
                initialRatio={0.55}
                first={
                  <div className="h-full min-h-0 min-w-0 overflow-hidden p-3">
                    <DeckPreview />
                  </div>
                }
                second={
                  <div className="h-full min-h-0 min-w-0 overflow-auto p-3">
                    <ValidatePanel />
                  </div>
                }
              />
            ) : (
              <div className="min-h-0 min-w-0 p-3">
                <DeckPreview />
              </div>
            )}
          </aside>
        )}
      </div>
      <StatusBar>
        {deckIoStatus !== null && (
          <StatusItem tone={deckIoStatus.kind === "error" ? "err" : "muted"}>
            {deckIoStatus.kind === "saved" ? m.deckIo.saved : deckIoStatus.kind === "loaded" ? m.deckIo.loaded : ""}
            {deckIoStatus.detail}
          </StatusItem>
        )}
        {namelistPath !== null && (
          <StatusItem>
            <span style={{ fontFamily: "var(--mono)" }}>{namelistPath.split("/").pop()}</span>
          </StatusItem>
        )}
        {runs.filter((r) => !isTerminal(r.state)).slice(0, 1).map((r) => (
          <StatusItem key={r.id} tone="ok">
            {r.name} {r.lastProgress !== null ? `${Math.round(r.lastProgress.pct)}%` : m.run.states[r.state]}
          </StatusItem>
        ))}
      </StatusBar>
      <CommandPalette
        open={paletteOpen}
        placeholder={m.app.palettePlaceholder}
        onClose={() => setPaletteOpen(false)}
        commands={[
          ...NAV_KEYS.map((k, i) => ({
            id: `section-${k}`,
            label: m.nav[k],
            hint: `⌘${i + 1}`,
            run: () => {
              setView("form");
              setSection(k);
            },
          })),
          { id: "history", label: m.nav.history, run: () => setView("history") },
          { id: "servers", label: m.nav.servers, run: () => setView("servers") },
          { id: "validate", label: m.validate.run, hint: "⌘⏎", run: () => void runValidate() },
          { id: "run", label: m.run.run, hint: "⇧⌘⏎", run: () => { if (!pibBlocked && !binMissing) void startRun(); } },
          { id: "save", label: m.deckIo.menuSave, hint: "⌘S", run: () => void saveNamelist() },
          { id: "load", label: m.deckIo.menuLoad, hint: "⌘O", run: () => void loadNamelist() },
        ]}
      />
    </div>
  );
}
