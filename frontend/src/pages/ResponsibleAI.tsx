import { Plot } from "../charts/Plot";
import { Badge, Card, Empty, ErrorState, fmt, Loading, Notice, PageHeader, pct, SyntheticBadge, titleCase } from "../components/ui";
import { useApi } from "../hooks/useApi";
import type { Overview, Report } from "../types/api";

interface Group { group: string; n_participants: number; prevalence: number; status: string; auroc?: number; sensitivity?: number; specificity?: number; brier?: number; ece?: number }
interface Fairness { overall: Record<string, number>; min_subgroup_size: number; attributes: Record<string, { groups: Group[]; gaps: { auroc_gap: number; sensitivity_gap: number } | null }>; banner: string | null }
interface Robustness { n_recordings: number; conditions: { condition: string; n: number; n_failed: number; abstention_rate: number; auroc?: number; mean_abs_probability_shift?: number; decision_flip_rate?: number }[] }
interface Uncertainty { abstention_rate: number; ood_rate: number; metrics_all: Record<string, number>; metrics_non_abstained: Record<string, number> | null; reason_counts: Record<string, number>; abstention_band: number[] }

const LIMITATIONS = [
  "Not a diagnostic device: outputs are statistical signals that require qualified clinical interpretation.",
  "Speech features are influenced by language, dialect, education, hearing, mood, fatigue, medication and recording conditions.",
  "Public speech corpora are small, mostly English and cross-sectional; validated performance does not transfer to new populations or tasks without re-evaluation.",
  "Longitudinal flags describe change relative to a person's own baseline, not impairment.",
  "ASR transcripts drop disfluencies and make errors that bias linguistic features.",
  "Subgroup results are withheld when samples are too small to support any fairness claim.",
];

export function ResponsibleAI() {
  const fairness = useApi<Report<Fairness>>("/reports/fairness");
  const robust = useApi<Report<Robustness>>("/reports/robustness");
  const unc = useApi<Report<Uncertainty>>("/reports/uncertainty");
  const card = useApi<Report<{ generated_markdown: string; banner: string | null }>>("/reports/model_card");
  const overview = useApi<Overview>("/overview");
  const flags = Object.entries(overview.data?.data_quality.quality_flag_counts ?? {});

  return (
    <>
      <PageHeader title="Responsible AI" description="Subgroup evaluation, robustness, uncertainty, data quality and the model card.">
        <SyntheticBadge show={!!fairness.data?.is_synthetic} />
      </PageHeader>
      <Notice tone="alert" title="Human oversight required">CogniTrack must never be used to diagnose, triage or deny care autonomously. Every output is decision support for a qualified clinician.</Notice>

      <div className="mt-6 grid gap-6 xl:grid-cols-3">
        <Card title="Uncertainty and abstention" subtitle="Test participants">
          {unc.loading ? <Loading /> : unc.error ? <ErrorState error={unc.error} /> : unc.data && (() => {
            const u = unc.data.payload;
            return (
              <div className="space-y-3 text-sm">
                <div className="grid grid-cols-2 gap-3">
                  <div><div className="eyebrow">Abstention rate</div><div className="num text-xl font-semibold">{pct(u.abstention_rate)}</div></div>
                  <div><div className="eyebrow">OOD rate</div><div className="num text-xl font-semibold">{pct(u.ood_rate)}</div></div>
                  <div><div className="eyebrow">AUROC (all)</div><div className="num text-xl font-semibold">{fmt(u.metrics_all.auroc)}</div></div>
                  <div><div className="eyebrow">AUROC (not abstained)</div><div className="num text-xl font-semibold">{fmt(u.metrics_non_abstained?.auroc ?? null)}</div></div>
                </div>
                <div className="eyebrow pt-2">Abstention triggers</div>
                <ul className="space-y-1">{Object.entries(u.reason_counts).map(([k, v]) => <li key={k} className="flex justify-between"><span className="text-ink-soft">{k}</span><span className="num">{v}</span></li>)}</ul>
                <p className="text-xs text-ink-muted">Probabilities in [{u.abstention_band.join(", ")}] are routed to review.</p>
              </div>
            );
          })()}
        </Card>
        <Card title="Robustness" subtitle="Serving model re-scored after perturbing test recordings" className="xl:col-span-2">
          {robust.loading ? <Loading /> : robust.error ? <ErrorState error={robust.error} /> : robust.data && (
            <>
              <Plot height={240} data={[
                { type: "bar", name: "AUROC", x: robust.data.payload.conditions.map((c) => titleCase(c.condition)), y: robust.data.payload.conditions.map((c) => c.auroc ?? null), marker: { color: "#0f766e" } },
                { type: "scatter", mode: "lines+markers", name: "Decision flip rate", yaxis: "y2", x: robust.data.payload.conditions.map((c) => titleCase(c.condition)), y: robust.data.payload.conditions.map((c) => c.decision_flip_rate ?? null), line: { color: "#be123c" } },
              ]} layout={{ yaxis: { range: [0, 1.02], title: { text: "AUROC" } }, yaxis2: { overlaying: "y", side: "right", range: [0, 1], title: { text: "Flip rate" }, showgrid: false }, margin: { l: 48, r: 48, t: 8, b: 90 } }} />
              <p className="mt-1 text-xs text-ink-muted">{robust.data.payload.n_recordings} test recordings; features re-extracted with the production code path for each condition.</p>
            </>
          )}
        </Card>
      </div>

      <Card title="Subgroup evaluation" subtitle={fairness.data ? `Participant level. Groups with fewer than ${fairness.data.payload.min_subgroup_size} participants are not reported.` : undefined} className="mt-6">
        {fairness.loading ? <Loading /> : fairness.error ? <ErrorState error={fairness.error} /> : fairness.data && (
          <div className="grid gap-6 lg:grid-cols-2">
            {Object.entries(fairness.data.payload.attributes).map(([attr, info]) => (
              <div key={attr}>
                <div className="mb-2 text-sm font-semibold">{titleCase(attr)}</div>
                <table className="table-base">
                  <thead><tr><th>Group</th><th>n</th><th>AUROC</th><th>Sens.</th><th>Spec.</th><th>Brier</th></tr></thead>
                  <tbody>{info.groups.map((g) => (
                    <tr key={g.group}><td>{g.group}</td><td className="num">{g.n_participants}</td>
                      {g.status === "evaluated" ? <><td className="num">{fmt(g.auroc)}</td><td className="num">{fmt(g.sensitivity, 2)}</td><td className="num">{fmt(g.specificity, 2)}</td><td className="num">{fmt(g.brier)}</td></>
                        : <td colSpan={4}><Badge>insufficient sample</Badge></td>}</tr>
                  ))}</tbody>
                </table>
                {info.gaps ? <p className="mt-1 text-xs text-ink-muted">AUROC gap {fmt(info.gaps.auroc_gap)}, sensitivity gap {fmt(info.gaps.sensitivity_gap)}</p>
                  : <p className="mt-1 text-xs text-ink-muted">Fewer than two evaluable groups: no gap reported.</p>}
              </div>
            ))}
          </div>
        )}
      </Card>

      <div className="mt-6 grid gap-6 xl:grid-cols-3">
        <Card title="Data quality warnings">
          {flags.length ? <ul className="space-y-1.5 text-sm">{flags.map(([f, n]) => <li key={f} className="flex justify-between"><span>{titleCase(f)}</span><span className="num">{n}</span></li>)}</ul> : <Empty>No quality flags.</Empty>}
        </Card>
        <Card title="Model limitations" className="xl:col-span-2">
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-ink-soft">{LIMITATIONS.map((l) => <li key={l}>{l}</li>)}</ul>
          <p className="mt-3 text-xs text-ink-muted">Full discussion: docs/LIMITATIONS.md and docs/MODEL_CARD.md.</p>
        </Card>
      </div>

      <Card title="Model card (generated)" className="mt-6">
        {card.loading ? <Loading /> : card.error ? <ErrorState error={card.error} /> : card.data && (
          <pre className="max-h-[520px] overflow-auto whitespace-pre-wrap rounded-lg bg-canvas p-4 font-mono text-xs leading-relaxed text-ink-soft">{card.data.payload.generated_markdown}</pre>
        )}
      </Card>
    </>
  );
}
