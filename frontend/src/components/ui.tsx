import type { ReactNode } from "react";
import type { ApiError } from "../services/api";

export const fmt = (v: number | null | undefined, digits = 3): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(digits);
export const pct = (v: number | null | undefined, digits = 0): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;
export const titleCase = (s: string) => s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

export function Card({ title, subtitle, actions, children, className = "" }: {
  title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`card card-pad ${className}`}>
      {(title || actions) && (
        <header className="mb-4 flex items-start justify-between gap-4">
          <div>
            {title && <h2 className="text-[15px] font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-sm text-ink-muted">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint, tone = "default" }: {
  label: string; value: ReactNode; hint?: ReactNode; tone?: "default" | "warn" | "ok";
}) {
  const toneCls = tone === "warn" ? "text-signal-warn" : tone === "ok" ? "text-signal-ok" : "text-ink";
  return (
    <div className="card card-pad">
      <div className="eyebrow">{label}</div>
      <div className={`num mt-2 text-[28px] font-semibold leading-none ${toneCls}`}>{value}</div>
      {hint && <div className="mt-2 text-xs text-ink-muted">{hint}</div>}
    </div>
  );
}

export type Tone = "neutral" | "brand" | "warn" | "alert" | "ok" | "info";
const toneMap: Record<Tone, string> = {
  neutral: "bg-canvas text-ink-soft border-line",
  brand: "bg-brand-50 text-brand-700 border-brand-100",
  warn: "bg-signal-warnbg text-signal-warn border-amber-200",
  alert: "bg-signal-alertbg text-signal-alert border-rose-200",
  ok: "bg-signal-okbg text-signal-ok border-emerald-200",
  info: "bg-blue-50 text-signal-info border-blue-200",
};

export const jobTone = (s: string): Tone => (s === "succeeded" ? "ok" : s === "failed" ? "alert" : s === "running" ? "info" : "neutral");

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: Tone }) {
  return <span className={`inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-medium ${toneMap[tone]}`}>{children}</span>;
}

export function SyntheticBadge({ show = true }: { show?: boolean }) {
  if (!show) return null;
  return <Badge tone="warn">Synthetic data</Badge>;
}

export function Notice({ tone = "warn", title, children }: { tone?: "warn" | "info" | "alert"; title?: string; children: ReactNode }) {
  const cls = tone === "alert" ? "border-rose-200 bg-signal-alertbg text-rose-900"
    : tone === "info" ? "border-blue-200 bg-blue-50 text-blue-900" : "border-amber-200 bg-signal-warnbg text-amber-900";
  return (
    <div className={`rounded-lg border px-4 py-3 text-sm ${cls}`} role="note">
      {title && <div className="mb-0.5 font-semibold">{title}</div>}
      <div className="leading-relaxed">{children}</div>
    </div>
  );
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 py-10 text-sm text-ink-muted" role="status">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-brand-500 border-t-transparent" />
      {label}…
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: ApiError | null; onRetry?: () => void }) {
  if (!error) return null;
  const unavailable = error.status === 503;
  return (
    <Notice tone={unavailable ? "info" : "alert"} title={unavailable ? "Not available yet" : error.status === 401 ? "Authentication required" : "Something went wrong"}>
      <p>{error.message}</p>
      {error.status === 401 && <p className="mt-1">Enter an API key using the key button in the header.</p>}
      {onRetry && <button className="btn-ghost mt-3" onClick={onRetry}>Retry</button>}
    </Notice>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="rounded-lg border border-dashed border-line px-4 py-8 text-center text-sm text-ink-muted">{children}</div>;
}

export function PageHeader({ title, description, children }: { title: string; description?: ReactNode; children?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold text-ink">{title}</h1>
        {description && <p className="mt-1 max-w-3xl text-sm text-ink-soft">{description}</p>}
      </div>
      {children && <div className="flex items-center gap-2">{children}</div>}
    </div>
  );
}

export function KeyValue({ items }: { items: [string, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[auto,1fr] gap-x-6 gap-y-2 text-sm">
      {items.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-ink-muted">{k}</dt>
          <dd className="num text-ink">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Tabs<T extends string>({ value, options, onChange }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void }) {
  return (
    <div className="inline-flex rounded-lg border border-line bg-white p-0.5" role="tablist">
      {options.map((o) => (
        <button key={o.value} role="tab" aria-selected={value === o.value} onClick={() => onChange(o.value)}
          className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${value === o.value ? "bg-brand-600 text-white" : "text-ink-soft hover:bg-canvas"}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export const MODALITY_COLORS: Record<string, string> = {
  acoustic: "#0f766e", linguistic: "#6366f1", conversational: "#d97706", embedding: "#0ea5e9", descriptive: "#94a3b8",
};

export function ModalityDot({ modality }: { modality: string }) {
  return <span className="inline-block h-2 w-2 rounded-full" style={{ background: MODALITY_COLORS[modality] ?? "#94a3b8" }} />;
}
