import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Plot } from "../charts/Plot";
import { TrendChart } from "../charts/TrendChart";
import { Badge, Card, Empty, ErrorState, fmt, KeyValue, Loading, ModalityDot, PageHeader, SyntheticBadge, Tabs, titleCase } from "../components/ui";
import { useApi } from "../hooks/useApi";
import type { FeatureTrends, ParticipantDetail as Detail, ParticipantSummary } from "../types/api";

export function Participants() {
  const [search, setSearch] = useState("");
  const [query, setQuery] = useState("");
  const { data, error, loading, reload } = useApi<{ total: number; items: ParticipantSummary[] }>(
    `/participants?limit=500${query ? `&search=${encodeURIComponent(query)}` : ""}`,
  );
  return (
    <>
      <PageHeader title="Participant analytics" description="Pseudonymous participants with their session histories. Identifiers never contain names or contact details." />
      <Card>
        <form className="mb-4 flex gap-2" onSubmit={(e) => { e.preventDefault(); setQuery(search.trim()); }}>
          <input className="input max-w-xs" placeholder="Search pseudonym…" value={search} onChange={(e) => setSearch(e.target.value)} aria-label="Search participants" />
          <button className="btn-ghost" type="submit">Search</button>
        </form>
        {loading ? <Loading /> : error ? <ErrorState error={error} onRetry={reload} /> : !data?.items.length ? <Empty>No participants found.</Empty> : (
          <div className="overflow-x-auto">
            <table className="table-base">
              <thead><tr><th>Participant</th><th>Source</th><th>Dataset label</th><th>Sessions</th><th>Age band</th><th>Sex</th><th>First</th><th>Last</th></tr></thead>
              <tbody>
                {data.items.map((p) => (
                  <tr key={p.pseudonym} className="hover:bg-canvas">
                    <td><Link className="font-mono font-medium text-brand-700 hover:underline" to={`/participants/${p.pseudonym}`}>{p.pseudonym}</Link></td>
                    <td>{p.dataset_source} {p.is_synthetic && <SyntheticBadge />}</td>
                    <td>{p.cohort_label ? titleCase(p.cohort_label) : <span className="text-ink-muted">unlabelled</span>}</td>
                    <td className="num">{p.n_sessions}</td><td>{p.age_band ?? "—"}</td><td>{p.sex ?? "—"}</td>
                    <td className="num">{p.first_session ?? "—"}</td><td className="num">{p.last_session ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-3 text-xs text-ink-muted">{data.total} participants</p>
          </div>
        )}
      </Card>
    </>
  );
}

type Mod = "acoustic" | "linguistic" | "conversational";

export function ParticipantDetail() {
  const { id = "" } = useParams();
  const detail = useApi<Detail>(`/participants/${id}`);
  const [modality, setModality] = useState<Mod>("acoustic");
  const trends = useApi<FeatureTrends>(`/participants/${id}/features?modality=${modality}`);
  const timeline = useMemo(() => detail.data?.sessions ?? [], [detail.data]);

  if (detail.loading) return <Loading label="Loading participant" />;
  if (detail.error) return <ErrorState error={detail.error} onRetry={detail.reload} />;
  const d = detail.data!;
  const xLabel = (s: { session_index: number; session_date: string | null }) => s.session_date ?? `S${s.session_index + 1}`;

  return (
    <>
      <PageHeader title={d.pseudonym} description={`${d.dataset_source} · ${timeline.length} sessions`}>
        <SyntheticBadge show={d.is_synthetic} />
        <Link className="btn-ghost" to={`/longitudinal/${d.pseudonym}`}>Longitudinal analysis</Link>
      </PageHeader>

      <div className="grid gap-6 xl:grid-cols-3">
        <Card title="Profile" subtitle="Non-identifying research metadata">
          <KeyValue items={[
            ["Dataset label", d.cohort_label ? titleCase(d.cohort_label) : "unlabelled"], ["Age band", d.age_band ?? "—"],
            ["Sex", d.sex ?? "—"], ["Education", d.education_band ?? "—"],
            ["Research consent", d.consent_research ? <Badge tone="ok">recorded</Badge> : <Badge tone="alert">missing</Badge>],
          ]} />
          <p className="mt-4 text-xs text-ink-muted">Labels come from the source dataset; CogniTrack never infers clinical labels.</p>
        </Card>

        <Card title="Session timeline" className="xl:col-span-2">
          <Plot height={170} data={[{
            type: "scatter", mode: "text+markers", x: timeline.map((s) => s.session_date ?? s.session_index), y: timeline.map(() => 0),
            text: timeline.map((s) => `S${s.session_index + 1}`), textposition: "top center",
            marker: { size: 14, symbol: "circle", color: timeline.map((s) => (s.recordings.every((r) => r.quality_usable === false) ? "#be123c" : "#0f766e")) },
            hovertemplate: "%{x}<extra></extra>",
          }]} layout={{ yaxis: { visible: false, range: [-1, 1.5] }, margin: { l: 16, r: 16, t: 8, b: 32 } }} />
        </Card>
      </div>

      <Card title="Recording history" className="mt-6">
        <div className="overflow-x-auto">
          <table className="table-base">
            <thead><tr><th>Session</th><th>Date</th><th>Task</th><th>Recording</th><th>Duration</th><th>Quality</th><th>Split</th><th>Model probability</th></tr></thead>
            <tbody>
              {timeline.flatMap((s) => s.recordings.map((r) => (
                <tr key={r.recording_key}>
                  <td className="num">S{s.session_index + 1}</td><td className="num">{s.session_date ?? "—"}</td><td>{s.task ? titleCase(s.task) : "—"}</td>
                  <td><Link className="font-mono text-brand-700 hover:underline" to={`/speech/${r.recording_key}`}>{r.recording_key}</Link></td>
                  <td className="num">{r.duration_s ? `${r.duration_s.toFixed(1)} s` : "—"}</td>
                  <td>{r.quality_usable === false ? <Badge tone="alert">{r.quality_flags.join(", ") || "unusable"}</Badge>
                    : r.quality_flags.length ? <Badge tone="warn">{r.quality_flags.join(", ")}</Badge> : <Badge tone="ok">ok</Badge>}</td>
                  <td>{r.split ?? "—"}</td>
                  <td className="num">{r.latest_probability !== null ? fmt(r.latest_probability, 2) : "—"}</td>
                </tr>
              )))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card className="mt-6" title="Feature trends" subtitle="Session means of usable recordings"
        actions={<Tabs value={modality} onChange={setModality} options={[{ value: "acoustic", label: "Acoustic" }, { value: "linguistic", label: "Linguistic" }, { value: "conversational", label: "Conversational" }]} />}>
        {trends.loading ? <Loading /> : trends.error ? <ErrorState error={trends.error} /> : !trends.data?.features.length ? (
          <Empty>No {modality} features for this participant{modality === "conversational" ? " (requires speaker-attributed turns)" : ""}.</Empty>
        ) : (
          <div className="grid gap-x-6 gap-y-4 md:grid-cols-2 xl:grid-cols-3">
            {trends.data.features.slice(0, modality === "acoustic" ? 12 : 18).map((f) => (
              <div key={f.name}>
                <TrendChart title={f.label} height={170} points={trends.data!.sessions.map((s) => ({ x: xLabel(s), y: s.values[f.name] ?? null }))} />
              </div>
            ))}
          </div>
        )}
        {trends.data && modality === "acoustic" && trends.data.features.length > 12 && (
          <p className="mt-3 flex items-center gap-1.5 text-xs text-ink-muted"><ModalityDot modality="acoustic" /> Showing 12 of {trends.data.features.length} acoustic features (MFCC statistics omitted).</p>
        )}
      </Card>
    </>
  );
}
