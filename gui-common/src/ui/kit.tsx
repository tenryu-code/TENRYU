import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "danger";

const VARIANT_STYLE: Record<Variant, React.CSSProperties> = {
  primary: {
    background: "var(--action-primary-bg)",
    color: "var(--action-primary-fg)",
    border: "1px solid transparent",
  },
  secondary: {
    background: "var(--bg-panel)",
    color: "var(--fg)",
    border: "1px solid var(--separator)",
  },
  danger: { background: "transparent", color: "var(--err)", border: "1px solid var(--err)" },
};

const CONTROL_STYLE: React.CSSProperties = {
  background: "var(--bg-panel)",
  border: "1px solid var(--separator)",
  color: "var(--fg)",
  borderRadius: "var(--radius-md)",
  height: "var(--control-h)",
  fontSize: 13,
};

export function Button({
  variant = "secondary",
  style,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      {...rest}
      className={`inline-flex items-center justify-center rounded px-3 text-[13px] disabled:opacity-40 ${rest.className ?? ""}`}
      style={{ minHeight: "var(--control-h)", ...VARIANT_STYLE[variant], ...style }}
    />
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="mb-2 block">
      <div className="mb-0.5 text-xs" style={{ color: "var(--fg-secondary)" }}>
        {label}
      </div>
      {children}
    </label>
  );
}

// Horizontal form row: label track + control area. The label column sizes to
// max-content but never below --label-track, so rows align without a fixed 224px gap.
export function FormRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label
      className="mb-2 grid items-center gap-2"
      style={{ gridTemplateColumns: "minmax(var(--label-track), max-content) 1fr" }}
    >
      <span className="text-xs" style={{ color: "var(--fg-secondary)" }}>
        {label}
      </span>
      <span className="flex min-w-0 items-center gap-2">{children}</span>
    </label>
  );
}

export function TextInput(props: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={`w-full rounded border px-2 ${props.className ?? ""}`}
      style={{ ...CONTROL_STYLE, ...props.style }}
    />
  );
}

export function NumberInput(props: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      type="number"
      {...props}
      className={`rounded border px-2 ${props.className ?? ""}`}
      style={{ ...CONTROL_STYLE, ...props.style }}
    />
  );
}

export function Select(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...props}
      className={`rounded border px-2 ${props.className ?? ""}`}
      style={{ ...CONTROL_STYLE, ...props.style }}
    />
  );
}

export function Badge({ tone, children }: { tone: "ok" | "err" | "muted" | "warn"; children: ReactNode }) {
  const color =
    tone === "ok" ? "var(--ok)" : tone === "err" ? "var(--err)" : tone === "warn" ? "var(--warn)" : "var(--fg-secondary)";
  const background =
    tone === "ok" ? "var(--ok-bg)" : tone === "err" ? "var(--err-bg)" : tone === "warn" ? "var(--warn-bg)" : "transparent";
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs"
      style={{ borderColor: color, color, background }}
    >
      {children}
    </span>
  );
}

// Visual composition: numeric input + unit select + optional canonical readout.
// Domain conversion logic stays with the caller (see Studio's QInput adapter).
export function UnitNumberField({
  value,
  unit,
  units,
  readout,
  onValueChange,
  onUnitChange,
}: {
  value: number;
  unit: string;
  units: readonly string[];
  readout?: string;
  onValueChange: (value: number) => void;
  onUnitChange: (unit: string) => void;
}) {
  return (
    <span className="flex min-w-0 items-center gap-2">
      <NumberInput
        step="any"
        value={Number.isFinite(value) ? value : ""}
        onChange={(event) =>
          onValueChange(event.target.value === "" ? Number.NaN : Number(event.target.value))
        }
        style={{ width: "var(--control-track-compact)" }}
      />
      <Select value={unit} onChange={(event) => onUnitChange(event.target.value)}>
        {units.map((u) => (
          <option key={u} value={u}>
            {u}
          </option>
        ))}
      </Select>
      {readout !== undefined && (
        <span className="text-xs" style={{ color: "var(--fg-secondary)", fontFamily: "var(--mono)" }}>
          {readout}
        </span>
      )}
    </span>
  );
}

// Titled semantic group of form rows; collapsible.
export function FieldGroup({
  title,
  defaultOpen = true,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  return (
    <details
      open={defaultOpen}
      className="mb-3 rounded border"
      style={{ borderColor: "var(--separator)", background: "var(--bg-panel)" }}
    >
      <summary className="cursor-pointer select-none px-3 py-2 text-xs font-semibold">
        {title}
      </summary>
      <div className="px-3 pb-3">{children}</div>
    </details>
  );
}
