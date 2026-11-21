import { useState, type ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { useApi } from "../hooks/useApi";
import { getApiKey, setApiKey } from "../services/api";
import type { Health } from "../types/api";

const NAV = [
  { to: "/", label: "Overview", icon: "M3 12l9-8 9 8M5 10v10h14V10" },
  { to: "/participants", label: "Participants", icon: "M16 11a4 4 0 10-8 0 4 4 0 008 0zM4 21a8 8 0 0116 0" },
  { to: "/speech", label: "Speech Analysis", icon: "M3 12h2l2-6 3 12 3-9 2 5 2-2h4" },
  { to: "/models", label: "Model Intelligence", icon: "M4 19V9m6 10V5m6 14v-7m4 7H2" },
  { to: "/longitudinal", label: "Longitudinal", icon: "M3 17l5-5 4 3 7-8M14 7h5v5" },
  { to: "/explainability", label: "Explainability", icon: "M12 3v3m0 12v3M3 12h3m12 0h3M6 6l2 2m8 8l2 2M6 18l2-2m8-8l2-2" },
  { to: "/responsible-ai", label: "Responsible AI", icon: "M12 3l8 4v5c0 5-3.5 8-8 9-4.5-1-8-4-8-9V7l8-4z" },
  { to: "/monitoring", label: "System Monitoring", icon: "M4 5h16v10H4zM8 19h8M12 15v4" },
];

function Icon({ d }: { d: string }) {
  return (
    <svg viewBox="0 0 24 24" className="h-[18px] w-[18px] shrink-0" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d={d} />
    </svg>
  );
}

function ApiKeyControl() {
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState(getApiKey() ?? "");
  return (
    <div className="relative">
      <button className="btn-ghost py-1.5" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={1.8}><path d="M15 7a4 4 0 11-3.9 4.9L3 20v-3h3v-3h3l2.1-2.1A4 4 0 0115 7z" /></svg>
        {getApiKey() ? "Key set" : "API key"}
      </button>
      {open && (
        <div className="absolute right-0 z-20 mt-2 w-72 rounded-xl border border-line bg-white p-4 shadow-lg">
          <label className="eyebrow" htmlFor="apikey">X-API-Key</label>
          <input id="apikey" type="password" className="input mt-2" value={value} onChange={(e) => setValue(e.target.value)} placeholder="Leave empty if auth is disabled" />
          <p className="mt-2 text-xs text-ink-muted">Stored for this browser tab only.</p>
          <div className="mt-3 flex justify-end gap-2">
            <button className="btn-ghost" onClick={() => { setApiKey(null); setValue(""); window.location.reload(); }}>Clear</button>
            <button className="btn-primary" onClick={() => { setApiKey(value || null); window.location.reload(); }}>Save</button>
          </div>
        </div>
      )}
    </div>
  );
}

export function Layout({ children }: { children: ReactNode }) {
  const health = useApi<Health>("/health");
  const h = health.data;
  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-line bg-white lg:flex">
        <div className="flex items-center gap-2.5 px-5 py-5">
          <img src="/favicon.svg" alt="" className="h-8 w-8" />
          <div>
            <div className="text-[15px] font-semibold leading-tight text-ink">CogniTrack AI</div>
            <div className="text-[11px] text-ink-muted">Speech intelligence · research</div>
          </div>
        </div>
        <nav className="flex-1 space-y-0.5 px-3" aria-label="Main">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === "/"}
              className={({ isActive }) => `flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${isActive ? "bg-brand-50 text-brand-700" : "text-ink-soft hover:bg-canvas hover:text-ink"}`}>
              <Icon d={n.icon} />
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="m-3 rounded-lg bg-canvas p-3 text-[11px] leading-relaxed text-ink-muted">
          Research and decision-support prototype. Not a medical device; never a diagnosis.
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-10 flex items-center justify-between gap-4 border-b border-line bg-white/90 px-6 py-3 backdrop-blur">
          <nav className="flex gap-1 overflow-x-auto lg:hidden" aria-label="Main mobile">
            {NAV.map((n) => (
              <NavLink key={n.to} to={n.to} end={n.to === "/"} className={({ isActive }) => `whitespace-nowrap rounded-md px-2 py-1 text-xs ${isActive ? "bg-brand-50 text-brand-700" : "text-ink-soft"}`}>
                {n.label}
              </NavLink>
            ))}
          </nav>
          <div className="hidden items-center gap-3 text-xs text-ink-muted lg:flex">
            <span className={`inline-flex items-center gap-1.5 ${h?.status === "ok" ? "text-signal-ok" : "text-signal-alert"}`}>
              <span className={`h-2 w-2 rounded-full ${h ? (h.status === "ok" ? "bg-signal-ok" : "bg-signal-warn") : health.error ? "bg-signal-alert" : "bg-line"}`} />
              {h ? `API ${h.status}` : health.error ? "API unreachable" : "Connecting"}
            </span>
            {h && <span>Model: <span className="font-mono text-ink-soft">{h.serving_model ?? "none registered"}</span></span>}
            {h && !h.auth_enabled && <span className="rounded bg-signal-warnbg px-1.5 py-0.5 text-signal-warn">auth disabled (dev)</span>}
          </div>
          <ApiKeyControl />
        </header>
        <main className="mx-auto w-full max-w-[1400px] flex-1 px-6 py-6">{children}</main>
      </div>
    </div>
  );
}
