import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Plot } from "../charts/Plot";
import { TrendChart } from "../charts/TrendChart";
import { Badge, Card, Empty, ErrorState, fmt, Loading, Notice, PageHeader, pct } from "../components/ui";
import { useApi } from "../hooks/useApi";
import type { LongitudinalResult } from "../types/api";

interface CohortRow {
  participant_id: string; status: string; n_sessions: number;
  summary?: { n_flagged_sessions: number; statistical_change_detected: boolean; max_composite_deviation: number | null; features_with_flags: string[] };
  missing: { missing_session_indices: number[] };
}
interface Cohort {
  cohort_summary: { n_participants: number; n_analysed: number; n_insufficient_data: number; n_with_statistical_change: number };
  participants: CohortRow[]; interpretation: string; tracked_features: string[];
  synthetic_engine_validation?: { note: string; n_participants: number; n_programmed_change: number; sensitivity: number; specificity: number; false_positive_participants: number };
}

function ParticipantLongitudinal({ id }: { id: string }) {
  const { data, error, loading, reload } = useApi<LongitudinalResult>(`/participants/${id}/longitudinal`);
  const [ewmaFeature, setEwmaFeature] = useState<string | null>(null);
  if (loading) return <Loading label="Computing baseline-relative analysis" />;
  if (error) return <ErrorState error={error} onRetry={reload} />;
  const d = data!;
  const Z = d.baseline?.z_threshold ?? 2;
  const x = d.sessions.map((s) => s.session_timestamp ?? `S${s.session_index + 1}`);
  const missing = d.missing.missing_session_indices;

  return (
    <div className="space-y-6">
      <Notice tone="info" title="How to read this">{d.interpretation}</Notice>
      {(missing.length > 0 || d.missing.long_intervals.length > 0) && (
        <Notice title="Missing-data indicators">
          {missing.length > 0 && <>Missed visit(s): {missing.map((i) => `S${i + 1}`).join(", ")}. </>}
          {d.missing.long_intervals.map((g) => `Long interval of ${g.days} days after S${g.after_session_index + 1}. `)}
        </Notice>
      )}
      {d.summary && (
        <Notice tone={d.summary.statistical_change_detected ? "warn" : "info"} title={d.summary.statistical_change_detected ? "Statistical change detected (not a clinical finding)" : "No statistical change by the participant-level rule"}>
          Rule: {d.summary.decision_rule}.{d.summary.persistent_features.length > 0 && <> Persistent: {d.summary.persistent_features.map((f) => d.feature_labels[f] ?? f).join(", ")}.</>}
          {d.summary.extreme_features.length > 0 && <> Extreme: {d.summary.extreme_features.map((f) => d.feature_labels[f] ?? f).join(", ")}.</>}
        </Notice>
      )}
      {d.status !== "analysed" ? (
        <Card title="Insufficient data"><p className="text-sm text-ink-soft">{d.status_detail}</p></Card>
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-4">
            <div className="card card-pad"><div className="eyebrow">Sessions</div><div className="num mt-2 text-2xl font-semibold">{d.n_sessions}</div><div className="mt-1 text-xs text-ink-muted">baseline: first {d.baseline?.sessions.length}</div></div>
            <div className="card card-pad"><div className="eyebrow">Flagged sessions</div><div className={`num mt-2 text-2xl font-semibold ${d.summary?.n_flagged_sessions ? "text-signal-warn" : ""}`}>{d.summary?.n_flagged_sessions}</div><div className="mt-1 text-xs text-ink-muted">|z| &gt; {Z} or EWMA out of control</div></div>
            <div className="card card-pad"><div className="eyebrow">Max composite deviation</div><div className="num mt-2 text-2xl font-semibold">{fmt(d.summary?.max_composite_deviation, 2)}</div><div className="mt-1 text-xs text-ink-muted">mean |z| across tracked features</div></div>
            <div className="card card-pad"><div className="eyebrow">Change points (PELT)</div><div className="num mt-2 text-2xl font-semibold">{d.change_points?.pelt_session_indices.length ?? 0}</div><div className="mt-1 text-xs text-ink-muted">{d.change_points?.pelt_session_indices.map((i) => `S${i + 1}`).join(", ") || "none detected"}</div></div>
          </div>

          <Card title="Tracked communication measures" subtitle={`Shaded band: expected range around the personal baseline (± ${Z} × within-person SD, widened for baseline uncertainty). Red markers: flagged sessions.`}>
            <div className="grid gap-x-6 gap-y-4 md:grid-cols-2 xl:grid-cols-3">
              {d.tracked_features.map((f) => (
                <TrendChart key={f} title={d.feature_labels[f] ?? f} baselineMean={d.baseline?.mean[f]} band={(d.baseline?.scale[f] ?? 0) * Z}
                  movingAverage={d.sessions.map((s) => s.moving_average?.[f] ?? null)}
                  points={d.sessions.map((s, i) => ({ x: x[i], y: s.values[f], baseline: s.is_baseline, flagged: (s.flagged_features ?? []).includes(f) }))} />
              ))}
            </div>
          </Card>

          <div className="grid gap-6 xl:grid-cols-2">
            <Card title="Composite deviation and change probability" subtitle="Bars: mean |z| per session · line: Bayesian change-point posterior that the current regime began at that session (first point = no change)">
              <Plot height={260} data={[
                { type: "bar", x, y: d.sessions.map((s) => s.composite_deviation ?? null), name: "Composite |z|", marker: { color: d.sessions.map((s) => (s.is_baseline ? "#94a3b8" : (s.flagged_features?.length ? "#be123c" : "#0f766e"))) } },
                { type: "scatter", mode: "lines+markers", x, y: d.change_points?.bocpd_probability ?? [], name: "P(regime began)", yaxis: "y2", line: { color: "#6366f1" } },
              ]} layout={{ yaxis: { title: { text: "Mean |z|" } }, yaxis2: { overlaying: "y", side: "right", range: [0, 1], title: { text: "P(regime began)" }, showgrid: false }, margin: { l: 48, r: 48, t: 8, b: 40 } }} />
              <p className="text-xs text-ink-muted">{d.change_points?.power_note}</p>
            </Card>
            <Card title="Statistical process control (EWMA)" actions={
              <select className="input w-56 py-1.5" value={ewmaFeature ?? d.tracked_features[0]} onChange={(e) => setEwmaFeature(e.target.value)} aria-label="EWMA feature">
                {d.tracked_features.map((f) => <option key={f} value={f}>{d.feature_labels[f] ?? f}</option>)}
              </select>}>
              {(() => {
                const f = ewmaFeature ?? d.tracked_features[0];
                const e = d.ewma?.[f];
                if (!e) return <Empty>No EWMA data.</Empty>;
                return <Plot height={240} data={[
                  { type: "scatter", mode: "lines", x, y: e.ucl, line: { dash: "dash", color: "#be123c", width: 1 }, name: "UCL" },
                  { type: "scatter", mode: "lines", x, y: e.lcl, line: { dash: "dash", color: "#be123c", width: 1 }, name: "LCL" },
                  { type: "scatter", mode: "lines+markers", x, y: e.ewma, line: { color: "#0f1f2a" }, marker: { color: e.out_of_control.map((o) => (o ? "#be123c" : "#0f766e")), size: 8 }, name: "EWMA" },
                ]} layout={{ showlegend: false }} />;
              })()}
            </Card>
          </div>

          <Card title="Session-to-session comparison" subtitle="Change from previous session in units of expected visit-to-visit variation (√2 × within-person SD)">
            <div className="overflow-x-auto">
              <table className="table-base">
                <thead><tr><th>Session</th>{d.tracked_features.map((f) => <th key={f}>{d.feature_labels[f] ?? f}</th>)}<th>Anomaly</th></tr></thead>
                <tbody>
                  {d.sessions.map((s, i) => (
                    <tr key={s.session_id}>
                      <td className="num whitespace-nowrap">{x[i]} {s.is_baseline && <Badge>baseline</Badge>}</td>
                      {d.tracked_features.map((f) => {
                        const v = s.delta_from_previous?.[f];
                        const strong = v !== null && v !== undefined && Math.abs(v) > Z;
                        return <td key={f} className={`num ${strong ? "font-semibold text-signal-alert" : ""}`}>{v === undefined ? "—" : fmt(v, 2)}</td>;
                      })}
                      <td>{s.is_anomaly ? <Badge tone="warn">unusual pattern</Badge> : s.anomaly_score !== undefined ? <span className="num text-xs text-ink-muted">{fmt(s.anomaly_score, 2)}</span> : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}

export function Longitudinal() {
  const { id } = useParams();
  const navigate = useNavigate();
  const cohort = useApi<Cohort>("/longitudinal/cohort");
  const c = cohort.data;
  return (
    <>
      <PageHeader title="Longitudinal analytics" description="Changes relative to each participant's own baseline — not population thresholds.">
        {c && (
          <select className="input w-56" value={id ?? ""} onChange={(e) => navigate(`/longitudinal/${e.target.value}`)} aria-label="Participant">
            <option value="" disabled>Select participant…</option>
            {c.participants.map((p) => <option key={p.participant_id} value={p.participant_id}>{p.participant_id} ({p.n_sessions})</option>)}
          </select>
        )}
      </PageHeader>
      {id ? <ParticipantLongitudinal id={id} /> : cohort.loading ? <Loading /> : cohort.error ? <ErrorState error={cohort.error} onRetry={cohort.reload} /> : c && (
        <div className="space-y-6">
          <div className="grid gap-4 md:grid-cols-4">
            <div className="card card-pad"><div className="eyebrow">Participants</div><div className="num mt-2 text-2xl font-semibold">{c.cohort_summary.n_participants}</div></div>
            <div className="card card-pad"><div className="eyebrow">Analysed</div><div className="num mt-2 text-2xl font-semibold">{c.cohort_summary.n_analysed}</div></div>
            <div className="card card-pad"><div className="eyebrow">Insufficient data</div><div className="num mt-2 text-2xl font-semibold">{c.cohort_summary.n_insufficient_data}</div></div>
            <div className="card card-pad"><div className="eyebrow">With statistical change</div><div className="num mt-2 text-2xl font-semibold text-signal-warn">{c.cohort_summary.n_with_statistical_change}</div></div>
          </div>
          <Notice tone="info">{c.interpretation}</Notice>
          {c.synthetic_engine_validation && (
            <Card title="Engine check on synthetic trajectories" subtitle={c.synthetic_engine_validation.note}>
              <div className="flex flex-wrap gap-8 text-sm">
                <div><div className="eyebrow">Programmed change</div><div className="num text-lg font-semibold">{c.synthetic_engine_validation.n_programmed_change} / {c.synthetic_engine_validation.n_participants}</div></div>
                <div><div className="eyebrow">Sensitivity</div><div className="num text-lg font-semibold">{pct(c.synthetic_engine_validation.sensitivity)}</div></div>
                <div><div className="eyebrow">Specificity</div><div className="num text-lg font-semibold">{pct(c.synthetic_engine_validation.specificity)}</div></div>
                <div><div className="eyebrow">False-positive participants</div><div className="num text-lg font-semibold">{c.synthetic_engine_validation.false_positive_participants}</div></div>
              </div>
            </Card>
          )}
          <Card title="Cohort">
            <div className="max-h-[520px] overflow-auto">
              <table className="table-base">
                <thead className="sticky top-0 bg-white"><tr><th>Participant</th><th>Status</th><th>Sessions</th><th>Missed visits</th><th>Flagged sessions</th><th>Max |z| (composite)</th><th>Features flagged</th></tr></thead>
                <tbody>
                  {c.participants.map((p) => (
                    <tr key={p.participant_id} className="hover:bg-canvas">
                      <td><Link className="font-mono text-brand-700 hover:underline" to={`/longitudinal/${p.participant_id}`}>{p.participant_id}</Link></td>
                      <td>{p.status === "analysed" ? (p.summary?.statistical_change_detected ? <Badge tone="warn">statistical change</Badge> : <Badge tone="ok">stable</Badge>) : <Badge>insufficient data</Badge>}</td>
                      <td className="num">{p.n_sessions}</td><td className="num">{p.missing.missing_session_indices.length || "—"}</td>
                      <td className="num">{p.summary?.n_flagged_sessions ?? "—"}</td><td className="num">{fmt(p.summary?.max_composite_deviation, 2)}</td>
                      <td className="text-xs text-ink-soft">{p.summary?.features_with_flags.join(", ") || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}
    </>
  );
}
