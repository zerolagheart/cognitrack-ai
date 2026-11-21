import { useEffect, useMemo, useState } from "react";
import { Plot } from "../charts/Plot";
import { Badge, Card, ErrorState, fmt, Loading, ModalityDot, MODALITY_COLORS, Notice, PageHeader, SyntheticBadge, titleCase } from "../components/ui";
import { useApi } from "../hooks/useApi";
import type { ExperimentSummary, ModelComparison, Report } from "../types/api";

interface ShapGlobal { feature: string; label: string; modality: string; mean_abs_shap: number; direction: number | null }
interface ExplainPayload { shap?: { experiment_id: string; global_importance: ShapGlobal[]; is_serving_model: boolean } }

function Ci({ e, k }: { e: ExperimentSummary; k: string }) {
  const c = e.ci?.[k];
  return <span className="text-[11px] text-ink-muted">[{fmt(c?.low ?? null, 2)}, {fmt(c?.high ?? null, 2)}]</span>;
}

export function ModelIntelligence() {
  const { data, error, loading, reload } = useApi<Report<ModelComparison>>("/reports/model_comparison");
  const explain = useApi<Report<ExplainPayload>>("/reports/explainability");
  const [selected, setSelected] = useState<string | null>(null);
  const comp = data?.payload;
  const ok = useMemo(() => (comp?.experiments ?? []).filter((e) => e.status === "ok").sort((a, b) => (b.cv_auroc ?? 0) - (a.cv_auroc ?? 0)), [comp]);
  useEffect(() => { if (comp && !selected) setSelected(comp.serving.experiment_id); }, [comp, selected]);

  if (loading) return <Loading label="Loading model comparison" />;
  if (error) return <ErrorState error={error} onRetry={reload} />;
  if (!comp) return null;
  const exp = ok.find((e) => e.experiment_id === selected) ?? ok[0];
  const skipped = comp.experiments.filter((e) => e.status !== "ok");
  const cm = exp?.test?.confusion_matrix;
  const shap = explain.data?.payload.shap;

  return (
    <>
      <PageHeader title="Model intelligence" description={`Ablation × model × fusion experiments on participant-independent splits. Dataset version ${comp.dataset_version}.`}>
        <SyntheticBadge show={comp.synthetic_data} />
      </PageHeader>
      {comp.banner && <div className="mb-6"><Notice title="Synthetic results">{comp.banner}</Notice></div>}

      <div className="grid gap-6 xl:grid-cols-3">
        <Card title="Serving model selection" subtitle="Simplest configuration within the margin of the best CV AUROC">
          <div className="font-mono text-[13px]">{comp.serving.experiment_id}</div>
          <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
            <div><div className="eyebrow">Chosen CV AUROC</div><div className="num text-lg font-semibold">{fmt(comp.serving.chosen_cv_score)}</div></div>
            <div><div className="eyebrow">Best CV AUROC</div><div className="num text-lg font-semibold">{fmt(comp.serving.best_cv_score)}</div></div>
          </div>
          <p className="mt-3 text-xs text-ink-muted">Margin {comp.serving.simplicity_margin}. Ensemble for disagreement: {comp.serving.ensemble.length} models.</p>
        </Card>
        <Card title="Does fusion help?" subtitle="Paired participant-bootstrap on the test set" className="xl:col-span-2">
          {comp.fusion_vs_best_single ? (() => {
            const f = comp.fusion_vs_best_single;
            const inconclusive = f.ci_low <= 0 && f.ci_high >= 0;
            return (
              <div className="text-sm">
                <p><span className="font-mono">{f.full}</span> vs best single modality <span className="font-mono">{f.best_single}</span></p>
                <p className="mt-2 num text-2xl font-semibold">ΔAUROC {f.delta_auroc >= 0 ? "+" : ""}{fmt(f.delta_auroc)} <span className="text-sm font-normal text-ink-muted">95% CI [{fmt(f.ci_low)}, {fmt(f.ci_high)}]</span></p>
                <p className="mt-2 text-ink-soft">{inconclusive ? "The interval includes zero: this test set does not show that multimodal fusion improves discrimination." : f.delta_auroc > 0 ? "Fusion outperforms the best single modality on this test set." : "The best single modality outperforms fusion on this test set."}</p>
              </div>
            );
          })() : <p className="text-sm text-ink-muted">Requires both a full-fusion and a single-modality experiment.</p>}
        </Card>
      </div>

      <Card title="Experiment comparison" subtitle="Test metrics are recording-level at threshold 0.5 on calibrated probabilities; brackets show 95% CIs. Click a row for curves." className="mt-6">
        <div className="max-h-[420px] overflow-auto">
          <table className="table-base">
            <thead className="sticky top-0 bg-white"><tr><th>Experiment</th><th>Modalities</th><th>CV AUROC</th><th>Test AUROC</th><th>AUPRC</th><th>Sens.</th><th>Spec.</th><th>F1</th><th>Brier</th><th>ECE</th></tr></thead>
            <tbody>
              {ok.map((e) => (
                <tr key={e.experiment_id} onClick={() => setSelected(e.experiment_id)}
                  className={`cursor-pointer ${e.experiment_id === exp?.experiment_id ? "bg-brand-50" : "hover:bg-canvas"}`}>
                  <td className="font-mono text-[12px]">{e.experiment_id}{e.experiment_id === comp.serving.experiment_id && <span className="ml-1.5"><Badge tone="brand">serving</Badge></span>}</td>
                  <td><span className="flex gap-1">{e.modalities.map((m) => <ModalityDot key={m} modality={m} />)}</span></td>
                  <td className="num font-medium">{fmt(e.cv_auroc)}</td>
                  <td className="num">{fmt(e.test?.auroc)} <Ci e={e} k="auroc" /></td>
                  <td className="num">{fmt(e.test?.auprc)}</td><td className="num">{fmt(e.test?.sensitivity, 2)}</td>
                  <td className="num">{fmt(e.test?.specificity, 2)}</td><td className="num">{fmt(e.test?.f1, 2)}</td>
                  <td className="num">{fmt(e.test?.brier)}</td><td className="num">{fmt(e.test?.ece)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {skipped.length > 0 && <p className="mt-3 text-xs text-ink-muted">Skipped: {skipped.map((s) => `${s.experiment_id} (${s.reason})`).join("; ")}</p>}
        <p className="mt-2 flex flex-wrap gap-3 text-xs text-ink-muted">{Object.entries(MODALITY_COLORS).filter(([k]) => k !== "descriptive").map(([k]) => <span key={k} className="flex items-center gap-1"><ModalityDot modality={k} />{k}</span>)}</p>
      </Card>

      {exp && (
        <div className="mt-6 grid gap-6 lg:grid-cols-2 xl:grid-cols-4">
          <Card title="ROC curve" subtitle={`AUROC ${fmt(exp.test?.auroc)}`}>
            {exp.curves?.roc ? <Plot height={250} data={[
              { x: exp.curves.roc.fpr, y: exp.curves.roc.tpr, type: "scatter", mode: "lines", line: { color: "#0f766e", width: 2, shape: "hv" }, name: "ROC" },
              { x: [0, 1], y: [0, 1], type: "scatter", mode: "lines", line: { dash: "dot", color: "#94a3b8" }, hoverinfo: "skip", name: "Chance" },
            ]} layout={{ showlegend: false, xaxis: { title: { text: "False positive rate" }, range: [0, 1] }, yaxis: { title: { text: "True positive rate" }, range: [0, 1.02] } }} /> : null}
          </Card>
          <Card title="Precision–recall" subtitle={`AUPRC ${fmt(exp.test?.auprc)} · prevalence ${fmt(exp.test?.prevalence, 2)}`}>
            {exp.curves?.pr ? <Plot height={250} data={[
              { x: exp.curves.pr.recall, y: exp.curves.pr.precision, type: "scatter", mode: "lines", line: { color: "#6366f1", width: 2, shape: "hv" } },
              { x: [0, 1], y: [exp.test?.prevalence ?? 0, exp.test?.prevalence ?? 0], type: "scatter", mode: "lines", line: { dash: "dot", color: "#94a3b8" }, hoverinfo: "skip" },
            ]} layout={{ showlegend: false, xaxis: { title: { text: "Recall" }, range: [0, 1] }, yaxis: { title: { text: "Precision" }, range: [0, 1.02] } }} /> : null}
          </Card>
          <Card title="Calibration" subtitle={`Brier ${fmt(exp.test?.brier)} · ECE ${fmt(exp.test?.ece)} (uncal. ${fmt(exp.test_uncalibrated?.ece)})`}>
            {exp.curves?.calibration && <Plot height={250} data={[
              { x: [0, 1], y: [0, 1], type: "scatter", mode: "lines", line: { dash: "dot", color: "#94a3b8" }, hoverinfo: "skip" },
              { x: exp.curves.calibration.mean_predicted, y: exp.curves.calibration.fraction_positive, type: "scatter", mode: "lines+markers",
                marker: { size: exp.curves.calibration.count.map((c) => 5 + Math.sqrt(c) * 1.5), color: "#d97706" }, line: { color: "#d97706" },
                text: exp.curves.calibration.count.map((c) => `n=${c}`), hovertemplate: "pred %{x:.2f} → obs %{y:.2f} (%{text})<extra></extra>" },
            ]} layout={{ showlegend: false, xaxis: { title: { text: "Mean predicted" }, range: [0, 1] }, yaxis: { title: { text: "Observed fraction" }, range: [0, 1.02] } }} />}
          </Card>
          <Card title="Confusion matrix" subtitle="Threshold 0.5, test recordings">
            {cm && <Plot height={250} data={[{
              type: "heatmap", z: [[cm.tp, cm.fn], [cm.fp, cm.tn]], x: ["Pred. positive", "Pred. negative"], y: ["Label positive", "Label negative"],
              colorscale: [[0, "#ecfdf8"], [1, "#0f766e"]], showscale: false, text: [[`TP ${cm.tp}`, `FN ${cm.fn}`], [`FP ${cm.fp}`, `TN ${cm.tn}`]] as unknown as string[],
              texttemplate: "%{text}", hoverinfo: "skip",
            }]} layout={{ yaxis: { autorange: "reversed" }, margin: { l: 96, r: 8, t: 8, b: 40 } }} />}
          </Card>
        </div>
      )}

      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <Card title="Modality ablation" subtitle="Best experiment per modality set">
          <Plot height={300} data={[
            { type: "bar", name: "CV AUROC", x: comp.ablation.map((a) => titleCase(a.modality_set)), y: comp.ablation.map((a) => a.cv_auroc), marker: { color: "#94a3b8" } },
            { type: "bar", name: "Test AUROC", x: comp.ablation.map((a) => titleCase(a.modality_set)), y: comp.ablation.map((a) => a.test_auroc), marker: { color: "#0f766e" },
              error_y: { type: "data", symmetric: false, array: comp.ablation.map((a) => (a.test_auroc_ci.high ?? a.test_auroc) - a.test_auroc), arrayminus: comp.ablation.map((a) => a.test_auroc - (a.test_auroc_ci.low ?? a.test_auroc)), color: "#3d5361" } },
          ]} layout={{ barmode: "group", yaxis: { range: [0.4, 1.02], title: { text: "AUROC" } } }} />
        </Card>
        <Card title="Feature importance" subtitle={shap ? `Mean |SHAP| on test set · ${shap.experiment_id}${shap.is_serving_model ? "" : " (closest SHAP-compatible model)"}` : "From the explainability report"}>
          {explain.loading ? <Loading /> : shap ? (
            <Plot height={300} data={[{
              type: "bar", orientation: "h", y: shap.global_importance.slice(0, 15).map((g) => g.label), x: shap.global_importance.slice(0, 15).map((g) => g.mean_abs_shap),
              marker: { color: shap.global_importance.slice(0, 15).map((g) => MODALITY_COLORS[g.modality] ?? "#94a3b8") }, hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
            }]} layout={{ yaxis: { autorange: "reversed" }, margin: { l: 200, r: 8, t: 4, b: 36 }, xaxis: { title: { text: "Mean |SHAP|" } } }} />
          ) : <p className="text-sm text-ink-muted">No SHAP report available.</p>}
        </Card>
      </div>
    </>
  );
}
