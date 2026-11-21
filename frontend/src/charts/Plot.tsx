import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import type { Data, Layout } from "plotly.js";

const PlotlyPlot = createPlotlyComponent(Plotly);

export const COLORWAY = ["#0f766e", "#6366f1", "#d97706", "#be123c", "#0ea5e9", "#65a30d", "#9333ea", "#64748b"];

const baseLayout: Partial<Layout> = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "Inter, Segoe UI, system-ui, sans-serif", size: 12, color: "#3d5361" },
  colorway: COLORWAY,
  margin: { l: 52, r: 16, t: 12, b: 44 },
  xaxis: { gridcolor: "#e8eef1", zerolinecolor: "#cfdbe0", linecolor: "#cfdbe0", automargin: true },
  yaxis: { gridcolor: "#e8eef1", zerolinecolor: "#cfdbe0", linecolor: "#cfdbe0", automargin: true },
  legend: { orientation: "h", y: -0.22, x: 0, font: { size: 11 } },
  hoverlabel: { bgcolor: "#0f1f2a", bordercolor: "#0f1f2a", font: { color: "#fff", size: 12 } },
};

export function Plot({ data, layout = {}, height = 300 }: { data: Data[]; layout?: Partial<Layout>; height?: number }) {
  const merged: Partial<Layout> = {
    ...baseLayout,
    ...layout,
    xaxis: { ...baseLayout.xaxis, ...(layout.xaxis ?? {}) },
    yaxis: { ...baseLayout.yaxis, ...(layout.yaxis ?? {}) },
    height,
    autosize: true,
  };
  return (
    <PlotlyPlot
      data={data}
      layout={merged}
      config={{ displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"] }}
      useResizeHandler
      style={{ width: "100%", height }}
    />
  );
}
