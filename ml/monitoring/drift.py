"""Data and prediction drift monitoring.

Reference = features of the training partition; current = a later batch (new uploads or a held-out
cohort). Per-feature drift uses the Population Stability Index and two-sample Kolmogorov–Smirnov tests
(Benjamini–Hochberg adjusted). An Evidently HTML report is produced alongside when Evidently is installed.

This detects *distribution shift* in inputs and outputs. It does not detect concept drift (a change in
the feature→label relationship); that needs fresh ground-truth labels and is out of scope here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ml.features.registry import feature_modality
from ml.utils import get_logger

log = get_logger(__name__)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) < 10 or len(current) < 10:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_frac = np.histogram(reference, edges)[0] / len(reference)
    cur_frac = np.histogram(current, edges)[0] / len(current)
    ref_frac, cur_frac = np.clip(ref_frac, 1e-4, None), np.clip(cur_frac, 1e-4, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def _bh(pvalues: np.ndarray) -> np.ndarray:
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order] * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], cfg: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for col in columns:
        r = reference[col].to_numpy(dtype=float)
        c = current[col].to_numpy(dtype=float)
        rv, cv = r[~np.isnan(r)], c[~np.isnan(c)]
        if len(rv) < 10 or len(cv) < 10:
            continue
        ks = ks_2samp(rv, cv)
        rows.append({
            "feature": col, "modality": feature_modality(col), "psi": psi(r, c, cfg["psi_bins"]),
            # PSI is biased upward in small samples: E[PSI | no drift] ~= (bins - 1) * (1/n_ref + 1/n_cur).
            "psi_noise_floor": (cfg["psi_bins"] - 1) * (1 / len(rv) + 1 / len(cv)),
            "ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
            "reference_mean": float(rv.mean()), "current_mean": float(cv.mean()),
            "reference_missing": float(np.isnan(r).mean()), "current_missing": float(np.isnan(c).mean()),
        })
    if rows:
        adj = _bh(np.array([x["ks_pvalue"] for x in rows]))
        for x, a in zip(rows, adj, strict=True):
            x["ks_pvalue_bh"] = float(a)
            psi_excess = np.isfinite(x["psi"]) and x["psi"] > cfg["psi_alert"] + x["psi_noise_floor"]
            x["drifted"] = bool(a < cfg["ks_alpha"] or psi_excess)
    rows.sort(key=lambda x: -(x["psi"] if np.isfinite(x["psi"]) else 0))
    n_drift = sum(x["drifted"] for x in rows)
    return {
        "n_features": len(rows), "n_drifted": n_drift, "share_drifted": n_drift / len(rows) if rows else 0.0,
        "features": rows,
        "thresholds": {"psi_alert": cfg["psi_alert"], "ks_alpha_bh": cfg["ks_alpha"],
                       "psi_rule": "PSI > psi_alert + small-sample noise floor (bins-1)(1/n_ref + 1/n_cur)"},
        "caveat": "Recordings are clustered within participants, so KS p-values are anti-conservative for small cohorts.",
    }


def prediction_drift(ref_pred: np.ndarray, cur_pred: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    ks = ks_2samp(ref_pred, cur_pred) if len(ref_pred) and len(cur_pred) else None
    hist_edges = np.linspace(0, 1, 11)
    return {
        "psi": psi(ref_pred, cur_pred, cfg["psi_bins"]),
        "ks_statistic": float(ks.statistic) if ks else None, "ks_pvalue": float(ks.pvalue) if ks else None,
        "reference_mean": float(np.mean(ref_pred)), "current_mean": float(np.mean(cur_pred)),
        "reference_positive_rate": float(np.mean(ref_pred >= 0.5)), "current_positive_rate": float(np.mean(cur_pred >= 0.5)),
        "histogram": {"edges": hist_edges.tolist(),
                      "reference": (np.histogram(ref_pred, hist_edges)[0] / max(len(ref_pred), 1)).tolist(),
                      "current": (np.histogram(cur_pred, hist_edges)[0] / max(len(cur_pred), 1)).tolist()},
    }


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, out_html: Path) -> str | None:
    """Write an Evidently data-drift HTML report. Returns the backend used, or None if unavailable."""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        snapshot = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
        snapshot.save_html(str(out_html))
        return "evidently>=0.7"
    except ImportError:
        pass
    except Exception as exc:
        log.warning("Evidently (new API) report failed", extra={"ctx": {"reason": f"{type(exc).__name__}: {exc}"}})
    try:
        from evidently.metric_preset import DataDriftPreset as LegacyPreset
        from evidently.report import Report as LegacyReport

        report = LegacyReport(metrics=[LegacyPreset()])
        report.run(reference_data=reference, current_data=current)
        report.save_html(str(out_html))
        return "evidently-legacy"
    except Exception as exc:
        log.warning("Evidently report unavailable", extra={"ctx": {"reason": f"{type(exc).__name__}"}})
        return None
