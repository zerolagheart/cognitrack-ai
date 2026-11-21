import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Plot } from "../charts/Plot";
import { Badge, Card, Empty, ErrorState, fmt, Loading, MODALITY_COLORS, Notice, PageHeader, pct, SyntheticBadge, titleCase } from "../components/ui";
import { useApi } from "../hooks/useApi";
import type { Explanation, Report } from "../types/api";

interface ExplainReport {
  serving_experiment: string; banner: string | null;
  modality_permutation_importance: Record<string, { auroc_drop_mean: number; auroc_drop_std: number }>;
  feature_permutation_importance: { feature: string; label: string; auroc_drop_mean: number; auroc_drop_std: number }[];
  shap?: {
    experiment_id: string; is_serving_model: boolean; base_value: number; output_space: string;
    global_importance: { feature: string; label: string; modality: string; mean_abs_shap: number; direction: number | null }[];
    modality_importance: Record<string, number>;
    interactions: { feature_a: string; feature_b: string; mean_abs_interaction: number }[];
  };
}
interface LocalResponse {
  recording_key: string; probability: number; abstain: boolean; disclaimer: string; model_experiment_id: string;
  uncertainty: { reasons?: string[]; ensemble_std?: number | null; ood_percentile?: number | null; missing_feature_fraction?: number } | null;
  explanation: Explanation | null;
}

function LocalExplanation({ recordingKey }: { recordingKey: string }) {
  const { data, error, loading, reload } = useApi<LocalResponse>(`/recordings/${recordingKey}/explanation`);
  if (loading) return <Loading label="Explaining prediction" />;
  if (error) return <ErrorState error={error} onRetry={reload} />;
  const d = data!;
  const e = d.explanation;
  const contrib = e?.contributions ?? [];
  const modality = e?.modality_contributions ?? e?.modality_attention ?? {};
  const reasons = d.uncertainty?.reasons ?? [];
  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-4">
        <div className="card card-pad"><div className="eyebrow">Calibrated probability</div><div className="num mt-2 text-3xl font-semibold">{fmt(d.probability, 2)}</div><div className="mt-1 text-xs text-ink-muted">{d.model_experiment_id}</div></div>
        <div className="card card-pad"><div className="eyebrow">Ensemble disagreement</div><div className="num mt-2 text-2xl font-semibold">{fmt(d.uncertainty?.ensemble_std ?? null, 3)}</div><div className="mt-1 text-xs text-ink-muted">SD across models on the same features</div></div>
        <div className="card card-pad"><div className="eyebrow">Distance to training data</div><div className="num mt-2 text-2xl font-semibold">{pct(d.uncertainty?.ood_percentile ?? null)}</div><div className="mt-1 text-xs text-ink-muted">percentile of training Mahalanobis distances</div></div>
        <div className="card card-pad"><div className="eyebrow">Missing features</div><div className="num mt-2 text-2xl font-semibold">{pct(d.uncertainty?.missing_feature_fraction ?? null)}</div></div>
      </div>
      {d.abstain ? <Notice title="Uncertainty warning — refer for clinician review">{reasons.join("; ")}</Notice>
        : <Notice tone="info">No uncertainty triggers fired. This does not make the output correct; it remains a statistical estimate.</Notice>}
      <div className="grid gap-6 xl:grid-cols-3">
        <Card title="Feature contributions" subtitle={e?.method === "shap" ? `SHAP values (${e.output_space}); positive pushes toward the impairment-label pattern` : titleCase(e?.method ?? "none")} className="xl:col-span-2">
          {contrib.length ? (
            <Plot height={Math.max(260, contrib.length * 28)} data={[{
              type: "bar", orientation: "h", y: contrib.map((c) => `${c.label}${c.value !== null ? ` = ${fmt(c.value, 2)}` : ""}`), x: contrib.map((c) => c.shap),
              marker: { color: contrib.map((c) => (c.shap >= 0 ? "#be123c" : "#0f766e")) }, hovertemplate: "%{y}<br>contribution %{x:.3f}<extra></extra>",
            }]} layout={{ yaxis: { autorange: "reversed" }, margin: { l: 260, r: 16, t: 4, b: 36 }, xaxis: { title: { text: "Contribution" }, zeroline: true } }} />
          ) : <Empty>{e?.note ?? "This model type provides modality-level explanations only."}</Empty>}
        </Card>
        <Card title="Modality contributions" subtitle={e?.modality_attention ? "Attention weights" : "Sum of feature contributions per modality"}>
          {Object.keys(modality).length ? (
            <Plot height={260} data={[{ type: "bar", x: Object.keys(modality).map(titleCase), y: Object.values(modality),
              marker: { color: Object.keys(modality).map((m) => MODALITY_COLORS[m] ?? "#94a3b8") }, hovertemplate: "%{x}: %{y:.3f}<extra></extra>" }]}
              layout={{ yaxis: { zeroline: true } }} />
          ) : <Empty>No modality breakdown.</Empty>}
          {e?.warnings?.map((w) => <p key={w} className="mt-2 text-xs text-signal-warn">{w}</p>)}
        </Card>
      </div>
      <p className="text-xs text-ink-muted">{d.disclaimer} Explanations describe model behaviour, not causal clinical mechanisms.</p>
    </div>
  );
}

export function Explainability() {
  const { key } = useParams();
  const navigate = useNavigate();
  const [input, setInput] = useState(key ?? "");
  const { data, error, loading, reload } = useApi<Report<ExplainReport>>("/reports/explainability");
  const r = data?.payload;
  const shap = r?.shap;
  const perm = r ? Object.entries(r.modality_permutation_importance) : [];

  return (
    <>
      <PageHeader title="Explainability" description="Global model behaviour on held-out participants, and local explanations for individual recordings.">
        <SyntheticBadge show={!!data?.is_synthetic} />
      </PageHeader>

      <Card title="Explain a recording" className="mb-6">
        <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); if (input.trim()) navigate(`/explainability/${input.trim()}`); }}>
          <input className="input max-w-md" placeholder="Recording key" value={input} onChange={(e) => setInput(e.target.value)} />
          <button className="btn-primary" type="submit">Explain</button>
          {key && <Link className="btn-ghost" to={`/speech/${key}`}>Open speech analysis</Link>}
        </form>
      </Card>
      {key && <div className="mb-8"><LocalExplanation recordingKey={key} /></div>}

      {loading ? <Loading /> : error ? <ErrorState error={error} onRetry={reload} /> : r && (
        <div className="space-y-6">
          {r.banner && <Notice>{r.banner}</Notice>}
          <div className="grid gap-6 xl:grid-cols-3">
            <Card title="Global SHAP importance" subtitle={shap ? `${shap.experiment_id}${shap.is_serving_model ? " (serving)" : " (closest SHAP-compatible model)"}` : undefined} className="xl:col-span-2">
              {shap ? (
                <Plot height={460} data={[{
                  type: "bar", orientation: "h", y: shap.global_importance.slice(0, 20).map((g) => g.label), x: shap.global_importance.slice(0, 20).map((g) => g.mean_abs_shap),
                  marker: { color: shap.global_importance.slice(0, 20).map((g) => MODALITY_COLORS[g.modality] ?? "#94a3b8") },
                  customdata: shap.global_importance.slice(0, 20).map((g) => (g.direction === null ? "n/a" : g.direction > 0 ? "higher value → higher output" : "higher value → lower output")),
                  hovertemplate: "%{y}<br>mean |SHAP| %{x:.3f}<br>%{customdata}<extra></extra>",
                }]} layout={{ yaxis: { autorange: "reversed" }, margin: { l: 230, r: 16, t: 4, b: 36 }, xaxis: { title: { text: `Mean |SHAP| (${shap.output_space})` } } }} />
              ) : <Empty>SHAP is available for tree and linear models only.</Empty>}
            </Card>
            <div className="space-y-6">
              <Card title="Modality share of attribution" subtitle="Share of total mean |SHAP|">
                {shap ? <Plot height={220} data={[{ type: "pie", hole: 0.6, labels: Object.keys(shap.modality_importance).map(titleCase), values: Object.values(shap.modality_importance),
                  marker: { colors: Object.keys(shap.modality_importance).map((m) => MODALITY_COLORS[m] ?? "#94a3b8") }, textinfo: "percent" }]}
                  layout={{ margin: { l: 8, r: 8, t: 8, b: 8 }, legend: { orientation: "v", x: 1, y: 0.5 } }} /> : <Empty>—</Empty>}
              </Card>
              <Card title="Modality permutation importance" subtitle={`AUROC drop when a modality is shuffled · ${r.serving_experiment}`}>
                <ul className="space-y-2 text-sm">
                  {perm.map(([m, v]) => (
                    <li key={m} className="flex items-center justify-between"><span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full" style={{ background: MODALITY_COLORS[m] }} />{titleCase(m)}</span>
                      <span className="num">{fmt(v.auroc_drop_mean)} <span className="text-xs text-ink-muted">± {fmt(v.auroc_drop_std, 3)}</span></span></li>
                  ))}
                </ul>
              </Card>
            </div>
          </div>
          <div className="grid gap-6 xl:grid-cols-2">
            <Card title="Feature interactions" subtitle="Mean |SHAP interaction| (tree boosting models)">
              {shap?.interactions.length ? (
                <table className="table-base"><thead><tr><th>Feature A</th><th>Feature B</th><th>Strength</th></tr></thead>
                  <tbody>{shap.interactions.map((i) => <tr key={i.feature_a + i.feature_b}><td>{i.feature_a}</td><td>{i.feature_b}</td><td className="num">{fmt(i.mean_abs_interaction, 4)}</td></tr>)}</tbody></table>
              ) : <Empty>Interaction values are not available for this model type.</Empty>}
            </Card>
            <Card title="Feature permutation importance" subtitle="Held-out AUROC drop (top 12)">
              <table className="table-base"><thead><tr><th>Feature</th><th>AUROC drop</th></tr></thead>
                <tbody>{r.feature_permutation_importance.slice(0, 12).map((f) => (
                  <tr key={f.feature}><td>{f.label}</td><td className="num">{fmt(f.auroc_drop_mean)} {f.auroc_drop_mean <= 0 && <Badge>no effect</Badge>}</td></tr>
                ))}</tbody></table>
            </Card>
          </div>
        </div>
      )}
    </>
  );
}
