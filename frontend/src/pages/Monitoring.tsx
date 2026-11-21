import { Plot } from "../charts/Plot";
import { Badge, Card, jobTone, Empty, ErrorState, fmt, KeyValue, Loading, ModalityDot, Notice, PageHeader, titleCase } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api, getApiKey } from "../services/api";
import type { Job, Report } from "../types/api";

interface Drift {
  generated_at: string; reference: string; current: string; n_reference: number; n_current: number; scope_note: string;
  feature_drift: { n_features: number; n_drifted: number; share_drifted: number; thresholds: { psi_alert: number; ks_alpha_bh: number };
    features: { feature: string; modality: string; psi: number; ks_statistic: number; ks_pvalue_bh: number; drifted: boolean; reference_mean: number; current_mean: number }[] };
  prediction_drift: { psi: number; ks_pvalue: number; reference_mean: number; current_mean: number; histogram: { edges: number[]; reference: number[]; current: number[] } } | null;
  evidently: { backend: string | null; html: string | null };
  model?: { experiment_id: string; dataset_version: string; trained_at: string; calibration: string };
}
interface ModelRow { experiment_key: string; model_type: string; strategy: string; is_serving: boolean; dataset_version: string; mlflow_run_id: string | null; summary_metrics: Record<string, number> | null; created_at: string }
interface Lineage { created_at: string; dataset_version: string; sources: { name: string; status: string; n_recordings?: number; message?: string }[]; n_recordings: number }
interface FeatureExtraction { n_recordings: number; n_errors: number; errors: { recording_id: string; extraction_error: string }[] }


export function Monitoring() {
  const drift = useApi<Report<Drift>>("/reports/drift");
  const jobs = useApi<Job[]>("/monitoring/jobs?limit=30");
  const models = useApi<ModelRow[]>("/models");
  const lineage = useApi<Report<Lineage>>("/reports/lineage");
  const fx = useApi<Report<FeatureExtraction>>("/reports/feature_extraction");
  const d = drift.data?.payload;
  const serving = models.data?.find((m) => m.is_serving);
  const failed = (jobs.data ?? []).filter((j) => j.status === "failed");

  return (
    <>
      <PageHeader title="System monitoring" description="Pipeline execution, data and prediction drift, model versions and processing errors." />

      <div className="grid gap-6 xl:grid-cols-3">
        <Card title="Model version">
          {models.loading ? <Loading /> : serving ? (
            <KeyValue items={[["Serving", <span className="font-mono text-xs">{serving.experiment_key}</span>], ["Dataset version", <span className="font-mono text-xs">{serving.dataset_version}</span>],
              ["MLflow run", <span className="font-mono text-xs">{serving.mlflow_run_id?.slice(0, 12) ?? "—"}</span>], ["Registered", new Date(serving.created_at).toLocaleString()],
              ["Registered models", models.data?.length ?? 0]]} />
          ) : <Empty>No model registered.</Empty>}
        </Card>
        <Card title="Pipeline lineage">
          {lineage.loading ? <Loading /> : lineage.error ? <ErrorState error={lineage.error} /> : lineage.data && (
            <div className="text-sm">
              <KeyValue items={[["Dataset version", <span className="font-mono text-xs">{lineage.data.payload.dataset_version}</span>], ["Recordings", lineage.data.payload.n_recordings], ["Ingested", new Date(lineage.data.payload.created_at).toLocaleString()]]} />
              <ul className="mt-3 space-y-1">{lineage.data.payload.sources.map((s) => <li key={s.name} className="flex items-center justify-between"><span>{s.name}</span><Badge tone={s.status === "ingested" ? "ok" : "neutral"}>{s.status}</Badge></li>)}</ul>
            </div>
          )}
        </Card>
        <Card title="Processing errors">
          {fx.data ? <p className="mb-2 text-sm">Feature extraction: <span className="num font-semibold">{fx.data.payload.n_errors}</span> of {fx.data.payload.n_recordings} recordings failed.</p> : null}
          {fx.data?.payload.errors.slice(0, 5).map((e) => <p key={e.recording_id} className="text-xs text-ink-soft"><span className="font-mono">{e.recording_id}</span>: {e.extraction_error}</p>)}
          <p className="mt-2 text-sm">Failed jobs: <span className="num font-semibold">{failed.length}</span></p>
          {failed.slice(0, 3).map((j) => <p key={j.id} className="text-xs text-signal-alert">{j.error}</p>)}
        </Card>
      </div>

      <Card title="Data drift" subtitle={d ? `${titleCase(d.current)} (n=${d.n_current}) vs ${titleCase(d.reference)} (n=${d.n_reference}) · PSI > ${d.feature_drift.thresholds.psi_alert} or BH-adjusted KS p < ${d.feature_drift.thresholds.ks_alpha_bh}` : undefined} className="mt-6"
        actions={d?.evidently.html ? <button className="btn-ghost" onClick={async () => {
          const key = getApiKey();
          const res = await fetch(api.url("/monitoring/evidently"), { headers: key ? { "X-API-Key": key } : {} });
          const url = URL.createObjectURL(await res.blob());
          window.open(url, "_blank", "noopener");
        }}>Open Evidently report</button> : undefined}>
        {drift.loading ? <Loading /> : drift.error ? <ErrorState error={drift.error} onRetry={drift.reload} /> : d && (
          <>
            <Notice tone="info">{d.scope_note}</Notice>
            <div className="mt-4 grid gap-6 xl:grid-cols-3">
              <div className="xl:col-span-2">
                <Plot height={320} data={[{
                  type: "bar", x: d.feature_drift.features.slice(0, 25).map((f) => f.feature), y: d.feature_drift.features.slice(0, 25).map((f) => f.psi),
                  marker: { color: d.feature_drift.features.slice(0, 25).map((f) => (f.drifted ? "#be123c" : "#94a3b8")) }, hovertemplate: "%{x}<br>PSI %{y:.3f}<extra></extra>",
                }]} layout={{ yaxis: { title: { text: "PSI" } }, margin: { l: 48, r: 8, t: 8, b: 120 },
                  shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: d.feature_drift.thresholds.psi_alert, y1: d.feature_drift.thresholds.psi_alert, line: { dash: "dot", color: "#be123c" } }] }} />
              </div>
              <div>
                <div className="num text-3xl font-semibold">{d.feature_drift.n_drifted}<span className="text-base font-normal text-ink-muted"> / {d.feature_drift.n_features}</span></div>
                <div className="text-sm text-ink-muted">features flagged for drift</div>
                {d.prediction_drift && (
                  <div className="mt-4">
                    <div className="eyebrow mb-1">Prediction distribution</div>
                    <Plot height={190} data={[
                      { type: "bar", name: "Reference", x: d.prediction_drift.histogram.edges.slice(0, -1).map((e) => e + 0.05), y: d.prediction_drift.histogram.reference, marker: { color: "#94a3b8" } },
                      { type: "bar", name: "Current", x: d.prediction_drift.histogram.edges.slice(0, -1).map((e) => e + 0.05), y: d.prediction_drift.histogram.current, marker: { color: "#0f766e" } },
                    ]} layout={{ barmode: "group", margin: { l: 36, r: 8, t: 4, b: 50 }, xaxis: { title: { text: "Predicted probability" } } }} />
                    <p className="text-xs text-ink-muted">Prediction PSI {fmt(d.prediction_drift.psi)} · mean {fmt(d.prediction_drift.reference_mean, 2)} → {fmt(d.prediction_drift.current_mean, 2)}</p>
                  </div>
                )}
              </div>
            </div>
            <details className="mt-3">
              <summary className="cursor-pointer text-sm font-medium text-brand-700">Drift table</summary>
              <table className="table-base mt-2"><thead><tr><th>Feature</th><th>PSI</th><th>KS</th><th>p (BH)</th><th>Ref. mean</th><th>Cur. mean</th><th /></tr></thead>
                <tbody>{d.feature_drift.features.map((f) => (
                  <tr key={f.feature}><td><span className="flex items-center gap-1.5"><ModalityDot modality={f.modality} />{f.feature}</span></td><td className="num">{fmt(f.psi)}</td><td className="num">{fmt(f.ks_statistic)}</td>
                    <td className="num">{fmt(f.ks_pvalue_bh)}</td><td className="num">{fmt(f.reference_mean)}</td><td className="num">{fmt(f.current_mean)}</td><td>{f.drifted && <Badge tone="alert">drift</Badge>}</td></tr>
                ))}</tbody></table>
            </details>
          </>
        )}
      </Card>

      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <Card title="Pipeline and processing jobs">
          {jobs.loading ? <Loading /> : jobs.error ? <ErrorState error={jobs.error} /> : !jobs.data?.length ? <Empty>No jobs recorded.</Empty> : (
            <div className="max-h-96 overflow-auto"><table className="table-base"><thead><tr><th>Job</th><th>Status</th><th>Started</th><th>Duration</th></tr></thead>
              <tbody>{jobs.data.map((j) => (
                <tr key={j.id}><td>{titleCase(j.job_type ?? "job")}<div className="font-mono text-[10px] text-ink-muted">{j.id.slice(0, 8)}</div></td><td><Badge tone={jobTone(j.status)}>{j.status}</Badge></td>
                  <td className="num text-xs">{new Date(j.created_at).toLocaleString()}</td>
                  <td className="num text-xs">{j.finished_at && j.started_at ? `${((new Date(j.finished_at).getTime() - new Date(j.started_at).getTime()) / 1000).toFixed(1)} s` : "—"}</td></tr>
              ))}</tbody></table></div>
          )}
        </Card>
        <Card title="Experiment history" subtitle="All registered model versions (MLflow holds full run details)">
          {models.loading ? <Loading /> : (
            <div className="max-h-96 overflow-auto"><table className="table-base"><thead><tr><th>Experiment</th><th>CV AUROC</th><th>Test AUROC</th></tr></thead>
              <tbody>{(models.data ?? []).map((m) => (
                <tr key={m.experiment_key}><td className="font-mono text-[12px]">{m.experiment_key} {m.is_serving && <Badge tone="brand">serving</Badge>}</td>
                  <td className="num">{fmt(m.summary_metrics?.cv_auroc ?? null)}</td><td className="num">{fmt(m.summary_metrics?.test_auroc ?? null)}</td></tr>
              ))}</tbody></table></div>
          )}
        </Card>
      </div>
    </>
  );
}
