import { useRef, useState, type ButtonHTMLAttributes, type KeyboardEvent, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";

export interface SegmentedTab {
  id: string;
  label: string;
}

// Accessible tablist: ArrowLeft/ArrowRight move selection, selected tab uses the
// --selected-* tokens (never the action fill).
export function SegmentedTabs({
  tabs,
  active,
  onSelect,
  "aria-label": ariaLabel,
}: {
  tabs: SegmentedTab[];
  active: string;
  onSelect: (id: string) => void;
  "aria-label"?: string;
}) {
  const refs = useRef<Array<HTMLButtonElement | null>>([]);
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const currentIndex = tabs.findIndex((tab) => tab.id === active);
    if (currentIndex < 0) return;
    const delta = event.key === "ArrowLeft" ? -1 : 1;
    const nextIndex = (currentIndex + delta + tabs.length) % tabs.length;
    onSelect(tabs[nextIndex].id);
    refs.current[nextIndex]?.focus();
  };
  return (
    <div role="tablist" aria-label={ariaLabel} className="flex items-center gap-1" onKeyDown={onKeyDown}>
      {tabs.map((tab, index) => {
        const selected = tab.id === active;
        return (
          <button
            key={tab.id}
            ref={(node) => {
              refs.current[index] = node;
            }}
            role="tab"
            aria-selected={selected}
            tabIndex={selected ? 0 : -1}
            onClick={() => onSelect(tab.id)}
            className="rounded px-3 text-[13px]"
            style={{
              minHeight: "var(--control-h-sm)",
              border: "1px solid transparent",
              background: selected ? "var(--selected-bg)" : "transparent",
              color: selected ? "var(--selected-fg)" : "var(--fg-secondary)",
              fontWeight: selected ? 600 : 400,
            }}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}

// Generic horizontal grouping for toolbars.
export function Toolbar({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={`flex items-center gap-2 ${className ?? ""}`}>{children}</div>;
}

// Persistent context strip under the app header (source / run / field etc.).
export function ContextBar({ children }: { children: ReactNode }) {
  return (
    <div
      className="flex items-center gap-3 border-b px-4"
      style={{ minHeight: 40, background: "var(--bg-panel)", borderColor: "var(--separator)" }}
    >
      {children}
    </div>
  );
}

// Window-bottom status strip: the single home for progress, errors and readouts.
export function StatusBar({ children }: { children: ReactNode }) {
  return (
    <div
      className="flex items-center gap-4 overflow-hidden border-t px-3"
      style={{
        minHeight: 24,
        fontSize: "var(--fs-label)",
        background: "var(--bg-panel)",
        borderColor: "var(--separator)",
        color: "var(--fg-secondary)",
      }}
    >
      {children}
    </div>
  );
}

export function StatusItem({
  tone = "muted",
  children,
}: {
  tone?: "ok" | "warn" | "err" | "muted";
  children: ReactNode;
}) {
  const color =
    tone === "ok" ? "var(--ok)" : tone === "err" ? "var(--err)" : tone === "warn" ? "var(--warn)" : "var(--fg-secondary)";
  return (
    <span className="inline-flex min-w-0 items-center gap-1 whitespace-nowrap" style={{ color }}>
      {children}
    </span>
  );
}

// Collapsible inspector group (uncontrolled; defaultOpen for the initial state).
export function InspectorSection({
  title,
  defaultOpen = true,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  return (
    <details open={defaultOpen} className="border-b" style={{ borderColor: "var(--separator)" }}>
      <summary
        className="cursor-pointer select-none px-3 py-2 text-xs font-semibold"
        style={{ color: "var(--fg)" }}
      >
        {title}
      </summary>
      <div className="flex flex-col gap-2 px-3 pb-3">{children}</div>
    </details>
  );
}

// Square icon-sized button for transport controls; label is mandatory for aria.
export function IconButton({
  label,
  style,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { label: string }) {
  return (
    <button
      {...rest}
      aria-label={label}
      title={rest.title ?? label}
      className={`inline-flex items-center justify-center rounded disabled:opacity-40 ${rest.className ?? ""}`}
      style={{
        width: "var(--control-h)",
        height: "var(--control-h)",
        border: "1px solid var(--separator)",
        background: "var(--bg-panel)",
        color: "var(--fg)",
        ...style,
      }}
    />
  );
}

// Two-pane splitter with a draggable divider; ratio persists to localStorage.
export function SplitPane({
  storageKey,
  direction,
  initialRatio = 0.55,
  min = 0.2,
  max = 0.85,
  first,
  second,
}: {
  storageKey: string;
  direction: "row" | "column";
  initialRatio?: number;
  min?: number;
  max?: number;
  first: ReactNode;
  second: ReactNode;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [ratio, setRatio] = useState(() => {
    if (typeof localStorage === "undefined") return initialRatio;
    const raw = localStorage.getItem(storageKey);
    const parsed = raw === null ? Number.NaN : Number(raw);
    return Number.isFinite(parsed) && parsed >= min && parsed <= max ? parsed : initialRatio;
  });
  const row = direction === "row";
  const template = `minmax(0, ${ratio}fr) 6px minmax(0, ${1 - ratio}fr)`;
  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const onPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
    const container = containerRef.current;
    if (container === null) return;
    const rect = container.getBoundingClientRect();
    const fraction = row
      ? (event.clientX - rect.left) / rect.width
      : (event.clientY - rect.top) / rect.height;
    setRatio(Math.min(max, Math.max(min, fraction)));
  };
  const onPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    if (typeof localStorage !== "undefined") localStorage.setItem(storageKey, String(ratio));
  };
  return (
    <div
      ref={containerRef}
      className="grid min-h-0 min-w-0 h-full"
      style={row ? { gridTemplateColumns: template } : { gridTemplateRows: template }}
    >
      <div className="min-h-0 min-w-0 overflow-hidden">{first}</div>
      <div
        role="separator"
        aria-orientation={row ? "vertical" : "horizontal"}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        style={{
          cursor: row ? "col-resize" : "row-resize",
          background: "var(--separator)",
          touchAction: "none",
        }}
      />
      <div className="min-h-0 min-w-0 overflow-hidden">{second}</div>
    </div>
  );
}
