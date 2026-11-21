import type { Data } from "plotly.js";
import { Plot } from "./Plot";

export interface TrendPoint { x: string | number; y: number | null; flagged?: boolean; baseline?: boolean }

/** Per-feature session trend with optional baseline band and flagged sessions. */
export function TrendChart({ title, points, baselineMean, band, movingAverage, height = 200 }: {
  title: string; points: TrendPoint[]; baselineMean?: number; band?: number; movingAverage?: (number | null)[]; height?: number;
}) {
  const xs = points.map((p) => p.x);
  const data: Data[] = [];
  if (baselineMean !== undefined && band !== undefined && xs.length) {
    data.push({
      x: [...xs, ...xs.slice().reverse()], y: [...xs.map(() => baselineMean + band), ...xs.map(() => baselineMean - band)],
      fill: "toself", fillcolor: "rgba(15,118,110,0.08)", line: { width: 0 }, hoverinfo: "skip", showlegend: false, type: "scatter",
    });
    data.push({ x: xs, y: xs.map(() => baselineMean), mode: "lines", line: { dash: "dot", color: "#0f766e", width: 1 }, name: "Baseline", type: "scatter", hoverinfo: "skip" });
  }
  if (movingAverage) data.push({ x: xs, y: movingAverage, mode: "lines", line: { color: "#94a3b8", width: 1.5 }, name: "Moving avg", type: "scatter" });
  data.push({
    x: xs, y: points.map((p) => p.y), mode: "lines+markers", name: title, type: "scatter", connectgaps: false,
    line: { color: "#0f1f2a", width: 1.8 },
    marker: { size: points.map((p) => (p.flagged ? 11 : 7)), color: points.map((p) => (p.flagged ? "#be123c" : p.baseline ? "#0f766e" : "#0f1f2a")),
      line: { color: "#fff", width: 1.5 } },
    hovertemplate: "%{x}<br>%{y:.3f}<extra></extra>",
  });
  return (
    <div>
      <div className="mb-1 text-[13px] font-medium text-ink">{title}</div>
      <Plot height={height} data={data} layout={{ showlegend: false, margin: { l: 48, r: 8, t: 4, b: 32 } }} />
    </div>
  );
}
