import { useEffect, useState } from "react";
import { FormRow, NumberInput, Select, TextInput, UnitNumberField } from "@tenryu-common/ui/kit";
import {
  CANONICAL_UNIT,
  toCanonical,
  UNIT_CHOICES,
  type Q,
  type UnitKind,
} from "../core/units";

export function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return <FormRow label={label}>{children}</FormRow>;
}

export function QInput({
  label,
  kind,
  value,
  onChange,
}: {
  label: string;
  kind: UnitKind;
  value: Q;
  onChange: (q: Q) => void;
}) {
  let cgs = "";
  try {
    cgs = `= ${String(Number(toCanonical(value, kind).toPrecision(6)))} ${CANONICAL_UNIT[kind]}`;
  } catch {
    cgs = "";
  }
  return (
    <Row label={label}>
      <UnitNumberField
        value={value.value}
        unit={value.unit}
        units={UNIT_CHOICES[kind]}
        readout={cgs}
        onValueChange={(v) => onChange({ ...value, value: v })}
        onUnitChange={(u) => onChange({ ...value, unit: u })}
      />
    </Row>
  );
}

export function NumInput({
  label,
  value,
  onChange,
  int = false,
  allowEmpty = false,
  hint,
}: {
  label: string;
  value: number | null;
  onChange: (n: number | null) => void;
  int?: boolean;
  allowEmpty?: boolean;
  hint?: string;
}) {
  return (
    <Row label={label}>
      <div className="flex flex-col gap-1">
        <NumberInput
          step={int ? 1 : "any"}
          value={value === null || !Number.isFinite(value) ? "" : value}
          onChange={(e) => {
            const raw = e.target.value;
            if (raw === "") {
              onChange(allowEmpty ? null : Number.NaN);
              return;
            }
            const n = Number(raw);
            onChange(int ? Math.trunc(n) : n);
          }}
          style={{ width: "var(--control-track-compact)" }}
        />
        {hint !== undefined && (
          <span className="text-xs" style={{ color: "var(--fg-secondary)", maxWidth: "22rem" }}>{hint}</span>
        )}
      </div>
    </Row>
  );
}

export function TextField({
  label,
  value,
  onChange,
  mono = false,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (s: string) => void;
  mono?: boolean;
  placeholder?: string;
}) {
  return (
    <Row label={label}>
      <TextInput
        type="text"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        style={{ width: "var(--control-track)", fontFamily: mono ? "var(--mono)" : undefined }}
      />
    </Row>
  );
}

export function MultiCheckField({
  label,
  options,
  selected,
  onChange,
  hint,
  undefinedSuffix,
}: {
  label: string;
  options: string[];
  selected: string[];
  onChange: (next: string[]) => void;
  hint?: string;
  undefinedSuffix?: string;
}) {
  const known = new Set(options);
  const stale = selected.filter((name) => !known.has(name));
  const toggle = (name: string, on: boolean) => {
    const set = new Set(selected);
    if (on) set.add(name);
    else set.delete(name);
    // Keep definition order for known options, then stale extras in their order.
    const next = [...options.filter((o) => set.has(o)), ...stale.filter((s) => set.has(s))];
    onChange(next);
  };
  return (
    <Row label={label}>
      <div className="flex flex-col gap-1" style={{ maxWidth: "22rem" }}>
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          {options.map((name) => (
            <label key={name} className="flex items-center gap-1 text-[13px]">
              <input
                type="checkbox"
                checked={selected.includes(name)}
                onChange={(e) => toggle(name, e.target.checked)}
              />
              <span style={{ fontFamily: "var(--mono)" }}>{name}</span>
            </label>
          ))}
          {stale.map((name) => (
            <label key={`stale-${name}`} className="flex items-center gap-1 text-[13px]" style={{ color: "var(--err)" }}>
              <input type="checkbox" checked onChange={() => toggle(name, false)} />
              <span style={{ fontFamily: "var(--mono)" }}>{name}</span>
              {undefinedSuffix !== undefined && <span className="text-xs">({undefinedSuffix})</span>}
            </label>
          ))}
        </div>
        {hint !== undefined && (
          <span className="text-xs" style={{ color: "var(--fg-secondary)" }}>{hint}</span>
        )}
      </div>
    </Row>
  );
}

export function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: Array<{ value: string; label: string }>;
  onChange: (v: string) => void;
}) {
  return (
    <Row label={label}>
      <Select value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </Select>
    </Row>
  );
}

export function SwitchField({
  label,
  checked,
  onChange,
  disabled = false,
  hint,
}: {
  label: string;
  checked: boolean;
  onChange: (b: boolean) => void;
  disabled?: boolean;
  hint?: string;
}) {
  return (
    <Row label={label}>
      <div className={`flex flex-col gap-1${disabled ? " opacity-60" : ""}`}>
        <input
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked)}
        />
        {hint !== undefined && (
          <span className="text-xs" style={{ color: "var(--fg-secondary)", maxWidth: "22rem" }}>{hint}</span>
        )}
      </div>
    </Row>
  );
}

export function BoundsInput({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number[];
  onChange: (ns: number[]) => void;
}) {
  const [raw, setRaw] = useState(value.join(", "));
  useEffect(() => {
    setRaw(value.join(", "));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(value)]);
  return (
    <Row label={label}>
      <TextInput
        type="text"
        value={raw}
        onChange={(e) => {
          setRaw(e.target.value);
          const parts = e.target.value
            .split(/[,\s]+/)
            .filter((x) => x.length > 0)
            .map(Number);
          if (parts.length > 0 && parts.every((n) => Number.isFinite(n))) {
            onChange(parts);
          }
        }}
        style={{ width: "var(--control-track)", fontFamily: "var(--mono)" }}
      />
    </Row>
  );
}
