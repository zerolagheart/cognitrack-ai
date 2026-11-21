import { Link } from "react-router-dom";
import { Plot } from "../charts/Plot";
import { Badge, Card, jobTone, Empty, ErrorState, fmt, Loading, Notice, PageHeader, Stat, SyntheticBadge, titleCase } from "../components/ui";
import { useApi } from "../hooks/useApi";
import type { Overview as OverviewData } from "../types/api";


export function Overview() {
  const { data, error, loading, reload } = useApi<OverviewData>("/overview");
  if (loading) return <Loading label="Loading overview" />;
  if (error) return <ErrorState error={error} onRetry={reload} />;
  if (!data) return null;
  const c = data.counts;
  const flags = Object.entries(data.data_quality.quality_flag_counts);
  const cls = data.data_quality.class_distribution?.participants ?? {};
  const m = data.serving_model?.metrics ?? {};

  return (
    <>
      <PageHeader title="Overview" description="Cohort, data quality and model status for the current pipeline run.">
        <SyntheticBadge show={data.synthetic_data} />
      </PageHeader>

      {data.banner && <div className="mb-6"><Notice title="Synthetic demonstration data">{data.banner}</Notice></div>}
      {c.participants === 0 && (
        <div className="mb-6"><Notice tone="info" title="No data published yet">Run <code className="font-mono">cognitrack run-all --quick</code> to generate synthetic data, train models and publish results.</Notice></div>
      )}

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <Stat label="Research participants" value={c.participants} hint={`${c.synthetic_participants} synthetic`} />
        <Stat label="Recordings" value={c.recordings} hint={`${c.sessions} assessment sessions`} />
        <Stat label="Analysed sessions" value={c.analysed_sessions} hint="with at least one usable recording" />
        <Stat label="Unusable recordings" value={c.unusable_recordings} tone={c.unusable_recordings ? "warn" : "ok"} hint="failed audio quality checks" />
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-3">
        <Card title="Serving model" subtitle="Selected by participant-grouped CV, never by test score" className="xl:col-span-1">
          {data.serving_model ? (
            <div className="space-y-3 text-sm">
              <div className="font-mono text-[13px] text-ink">{data.serving_model.experiment_id}</div>
              <div className="flex flex-wrap gap-1.5">
                {data.serving_model.modalities.map((mod) => <Badge key={mod} tone="brand">{mod}</Badge>)}
                <Badge>{data.serving_model.strategy}</Badge>
                <Badge>{data.serving_model.model}</Badge>
              </div>
              <div className="grid grid-cols-2 gap-3 pt-2">
                <div><div className="eyebrow">CV AUROC</div><div className="num text-lg font-semibold">{fmt(m.cv_auroc as number)}</div></div>
                <div><div className="eyebrow">Test AUROC</div><div className="num text-lg font-semibold">{fmt(m.test_auroc as number)}</div></div>
                <div><div className="eyebrow">Test Brier</div><div className="num text-lg font-semibold">{fmt(m.test_brier as number)}</div></div>
                <div><div className="eyebrow">Test ECE</div><div className="num text-lg font-semibold">{fmt(m.test_ece as number)}</div></div>
              </div>
              <Link to="/models" className="inline-block pt-1 text-sm font-medium text-brand-600 hover:underline">Model comparison →</Link>
            </div>
          ) : <Empty>No trained model registered.</Empty>}
        </Card>

        <Card title="Modality ablation" subtitle="Test AUROC with 95% participant-bootstrap CI (best model per modality set)" className="xl:col-span-2">
          {data.ablation?.length ? (
            <Plot height={280} data={[{
              type: "bar", orientation: "h",
              y: data.ablation.map((a) => titleCase(a.modality_set)), x: data.ablation.map((a) => a.test_auroc),
              error_x: { type: "data", symmetric: false,
                array: data.ablation.map((a) => (a.test_auroc_ci.high ?? a.test_auroc) - a.test_auroc),
                arrayminus: data.ablation.map((a) => a.test_auroc - (a.test_auroc_ci.low ?? a.test_auroc)), color: "#3d5361", thickness: 1.2 },
              marker: { color: "#0f766e" }, hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
            }]} layout={{ xaxis: { range: [0.4, 1], title: { text: "Test AUROC" } }, yaxis: { autorange: "reversed" }, margin: { l: 140, r: 16, t: 8, b: 44 },
              shapes: [{ type: "line", x0: 0.5, x1: 0.5, y0: -0.5, y1: data.ablation.length - 0.5, line: { dash: "dot", color: "#94a3b8" } }] }} />
          ) : <Empty>Ablation results appear after training.</Empty>}
        </Card>
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-3">
        <Card title="Data quality" subtitle="Validation and audio quality checks">
          <div className="mb-3 flex items-center gap-2 text-sm">
            Manifest validation:
            {data.data_quality.validation_passed === null ? <Badge>not run</Badge>
              : data.data_quality.validation_passed ? <Badge tone="ok">passed</Badge> : <Badge tone="alert">failed</Badge>}
          </div>
          {flags.length ? (
            <ul className="space-y-1.5 text-sm">
              {flags.map(([f, n]) => (
                <li key={f} className="flex justify-between"><span className="text-ink-soft">{titleCase(f)}</span><span className="num font-medium">{n}</span></li>
              ))}
            </ul>
          ) : <p className="text-sm text-ink-muted">No audio quality flags raised.</p>}
          <p className="mt-3 text-xs text-ink-muted">Missing transcripts: {data.data_quality.missing_transcripts ?? "—"}</p>
        </Card>

        <Card title="Participant labels" subtitle="Assessment labels as provided by the dataset">
          {Object.keys(cls).length ? (
            <Plot height={220} data={[{ type: "pie", hole: 0.62, labels: Object.keys(cls).map(titleCase), values: Object.values(cls),
              marker: { colors: ["#0f766e", "#6366f1", "#cbd5e1"] }, textinfo: "value", hovertemplate: "%{label}: %{value}<extra></extra>" }]}
              layout={{ margin: { l: 8, r: 8, t: 8, b: 8 }, showlegend: true, legend: { orientation: "v", x: 1, y: 0.5 } }} />
          ) : <Empty>No labels.</Empty>}
        </Card>

        <Card title="Recent processing activity">
          {data.recent_jobs.length ? (
            <ul className="divide-y divide-line text-sm">
              {data.recent_jobs.map((j) => (
                <li key={j.id} className="flex items-center justify-between py-2">
                  <div>
                    <div className="font-medium text-ink">{titleCase(j.type ?? j.job_type ?? "job")}</div>
                    <div className="text-xs text-ink-muted">{new Date(j.created_at).toLocaleString()}</div>
                  </div>
                  <Badge tone={jobTone(j.status)}>{j.status}</Badge>
                </li>
              ))}
            </ul>
          ) : <Empty>No jobs yet.</Empty>}
        </Card>
      </div>
    </>
  );
}
