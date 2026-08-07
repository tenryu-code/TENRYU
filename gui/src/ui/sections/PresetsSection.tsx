import { useState } from "react";
import { t } from "../../i18n";
import {
  preset2dBlank,
  preset2dPolarCapsule,
  preset2dPolarSphere,
  preset2dRectLaser,
  preset2dRectSlabRad,
  presetBlank,
  presetIndirectTr,
  presetLaserSphere,
  presetSlabRadiation,
} from "../../core/presets";
import type { FormState } from "../../core/deck/formState";
import { useApp } from "../../store";
import { Button } from "@tenryu-common/ui/kit";

export default function PresetsSection() {
  const m = t();
  const loadForm = useApp((s) => s.loadForm);
  const setSection = useApp((s) => s.setSection);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [tab, setTab] = useState<"1d" | "2d">("1d");

  const cards1d: Array<{ title: string; desc: string; build: () => FormState }> = [
    { title: m.presets.blank, desc: m.presets.blankDesc, build: presetBlank },
    { title: m.presets.slab, desc: m.presets.slabDesc, build: presetSlabRadiation },
    { title: m.presets.laserSphere, desc: m.presets.laserSphereDesc, build: presetLaserSphere },
    { title: m.presets.indirect, desc: m.presets.indirectDesc, build: presetIndirectTr },
  ];
  const cards2d: Array<{ title: string; desc: string; build: () => FormState }> = [
    { title: m.presets.blank2d, desc: m.presets.blank2dDesc, build: preset2dBlank },
    { title: m.presets.polarSphere2d, desc: m.presets.polarSphere2dDesc, build: preset2dPolarSphere },
    { title: m.presets.polarCapsule2d, desc: m.presets.polarCapsule2dDesc, build: preset2dPolarCapsule },
    { title: m.presets.slabRad2d, desc: m.presets.slabRad2dDesc, build: preset2dRectSlabRad },
    { title: m.presets.laserCyl2d, desc: m.presets.laserCyl2dDesc, build: preset2dRectLaser },
  ];
  const cards = tab === "1d" ? cards1d : cards2d;

  return (
    <div className="max-w-xl">
      <h1 className="mb-2 text-base font-semibold">{m.nav.presets}</h1>
      <div className="mb-2 flex gap-2">
        <Button variant={tab === "1d" ? "primary" : "secondary"} onClick={() => setTab("1d")}>
          {m.presets.tab1d}
        </Button>
        <Button variant={tab === "2d" ? "primary" : "secondary"} onClick={() => setTab("2d")}>
          {m.presets.tab2d}
        </Button>
      </div>
      <div className="grid grid-cols-2 gap-2">
        {cards.map((c) => (
          <div
            key={c.title}
            className="flex flex-col gap-1 rounded border p-3"
            style={{ borderColor: "var(--separator)", background: "var(--bg-panel)" }}
          >
            <div className="font-medium">{c.title}</div>
            <p className="flex-1 text-xs" style={{ color: "var(--fg-secondary)" }}>
              {c.desc}
            </p>
            {confirming === c.title ? (
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
              <Button variant="primary" onClick={() => setConfirming(c.title)}>
                {m.presets.apply}
              </Button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
