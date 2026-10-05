import { useState } from "react";
import { t } from "../../i18n";
import {
  preset2dBlank,
  preset2dPolarCapsule,
  preset2dPolarSphere,
  preset2dRectLaser,
  preset2dRectSlabRad,
} from "../../core/presets";
import { PRESETS_1D, type Preset1d } from "../../core/presets1d";
import type { FormState } from "../../core/deck/formState";
import { useApp } from "../../store";
import { Badge, Button } from "@tenryu-common/ui/kit";

interface Card {
  key: string;
  title: string;
  desc: string;
  kernels?: string;
  result?: string;
  badges?: Array<{ tone: "ok" | "warn" | "muted"; text: string }>;
  build: () => FormState;
}

function card1d(preset: Preset1d): Card {
  const m = t().presets;
  const item = m.items[preset.id as keyof typeof m.items];
  const form = preset.build();
  const badges: Card["badges"] = [];
  if (form.mesh.grid1d === "recommended") badges.push({ tone: "ok", text: m.meshRecommended });
  else if (form.mesh.grid1d === "layers") badges.push({ tone: "ok", text: m.meshLayers });
  if (preset.tables.length > 0) badges.push({ tone: "warn", text: m.needsTables(preset.tables.join(", ")) });
  if (preset.heavy === true) badges.push({ tone: "warn", text: m.heavyRun });
  const result = (item as { result?: string }).result;
  return { key: preset.id, title: item.title, desc: item.desc, kernels: item.kernels, result, badges, build: preset.build };
}

export default function PresetsSection() {
  const m = t();
  const loadForm = useApp((s) => s.loadForm);
  const setSection = useApp((s) => s.setSection);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [tab, setTab] = useState<"1d" | "2d">("1d");

  const cards2d: Card[] = [
    { key: "blank2d", title: m.presets.blank2d, desc: m.presets.blank2dDesc, build: preset2dBlank },
    { key: "polarSphere2d", title: m.presets.polarSphere2d, desc: m.presets.polarSphere2dDesc, build: preset2dPolarSphere },
    { key: "polarCapsule2d", title: m.presets.polarCapsule2d, desc: m.presets.polarCapsule2dDesc, build: preset2dPolarCapsule },
    { key: "slabRad2d", title: m.presets.slabRad2d, desc: m.presets.slabRad2dDesc, build: preset2dRectSlabRad },
    { key: "laserCyl2d", title: m.presets.laserCyl2d, desc: m.presets.laserCyl2dDesc, build: preset2dRectLaser },
  ];
  const groups: Array<{ title: string | null; cards: Card[] }> =
    tab === "1d"
      ? [
          { title: m.presets.groupBasic, cards: PRESETS_1D.filter((p) => p.category === "basic").map(card1d) },
          { title: m.presets.groupOption, cards: PRESETS_1D.filter((p) => p.category === "option").map(card1d) },
        ]
      : [{ title: null, cards: cards2d }];

  const renderCard = (c: Card) => (
    <div
      key={c.key}
      className="flex flex-col gap-1 rounded border p-3"
      style={{ borderColor: "var(--separator)", background: "var(--bg-panel)" }}
    >
      <div className="font-medium">{c.title}</div>
      {c.badges !== undefined && c.badges.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {c.badges.map((b) => (
            <Badge key={b.text} tone={b.tone}>{b.text}</Badge>
          ))}
        </div>
      )}
      <p className="flex-1 text-xs" style={{ color: "var(--fg-secondary)" }}>
        {c.desc}
      </p>
      {c.kernels !== undefined && (
        <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.presets.kernelsLabel}: {c.kernels}</p>
      )}
      {c.result !== undefined && (
        <p className="text-xs" style={{ color: "var(--fg-secondary)" }}>{m.presets.resultLabel}: {c.result}</p>
      )}
      {confirming === c.key ? (
        <div className="flex flex-col gap-1">
          <p className="text-xs" style={{ color: "var(--err)" }}>
            {m.presets.confirmWarn}
          </p>
          <div className="flex gap-2">
            <Button
              variant="primary"
              onClick={() => {
                loadForm(c.build());
                setConfirming(null);
                setSection("basic");
              }}
            >
              {m.presets.confirmApply}
            </Button>
            <Button onClick={() => setConfirming(null)}>{m.presets.cancel}</Button>
          </div>
        </div>
      ) : (
        <Button variant="primary" onClick={() => setConfirming(c.key)}>
          {m.presets.apply}
        </Button>
      )}
    </div>
  );

  return (
    <div className="max-w-2xl">
      <h1 className="mb-2 text-base font-semibold">{m.nav.presets}</h1>
      <div className="mb-2 flex gap-2">
        <Button variant={tab === "1d" ? "primary" : "secondary"} onClick={() => setTab("1d")}>
          {m.presets.tab1d}
        </Button>
        <Button variant={tab === "2d" ? "primary" : "secondary"} onClick={() => setTab("2d")}>
          {m.presets.tab2d}
        </Button>
      </div>
      {tab === "1d" && <p className="mb-2 text-xs" style={{ color: "var(--fg-secondary)" }}>{m.presets.intro1d}</p>}
      {groups.map((group, i) => (
        <div key={i} className="mb-3">
          {group.title !== null && <h2 className="mb-1 text-sm font-semibold">{group.title}</h2>}
          <div className="grid grid-cols-2 gap-2">{group.cards.map(renderCard)}</div>
        </div>
      ))}
    </div>
  );
}
