"""Post-training analytics: longitudinal engine, drift monitoring, robustness and the generated model card."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ml.config import Config, resolve_path
from ml.evaluation.robustness import run_robustness
from ml.features.registry import ACOUSTIC_FEATURES, CONVERSATIONAL_FEATURES, LINGUISTIC_FEATURES
from ml.inference import Predictor
from ml.longitudinal.engine import analyse_cohort
from ml.monitoring.drift import evidently_report, feature_drift, prediction_drift
from ml.training.trainer import modelling_partitions
from ml.utils import get_logger, read_json, relpath, write_json

log = get_logger(__name__)


def run_longitudinal(cfg: Config, df: pd.DataFrame) -> dict[str, Any]:
    report = analyse_cohort(df, cfg["longitudinal"], cfg["project"]["seed"])
    report["synthetic_data"] = bool(df["is_synthetic"].all())
    # On synthetic data the programmed trajectories are known, so the engine itself can be validated.
    meta_path = resolve_path(cfg["paths"]["synthetic"]) / "metadata.csv"
    if report["synthetic_data"] and meta_path.exists():
        meta = pd.read_csv(meta_path)
        truth = meta.groupby("participant_id")["programmed_change"].first()
        detected = {p["participant_id"]: p["summary"]["statistical_change_detected"]
                    for p in report["participants"] if p["status"] == "analysed"}
        common = [pid for pid in detected if pid in truth.index]
        if common:
            y = np.array([bool(truth[p]) for p in common])
            d = np.array([detected[p] for p in common])
            report["synthetic_engine_validation"] = {
                "note": "Agreement between detected statistical change and the generator's programmed trajectories "
                        "(synthetic only; says nothing about real-world clinical change).",
                "n_participants": len(common), "n_programmed_change": int(y.sum()),
                "sensitivity": float((d & y).sum() / max(y.sum(), 1)),
                "specificity": float((~d & ~y).sum() / max((~y).sum(), 1)),
                "false_positive_participants": int((d & ~y).sum()),
            }
    write_json(cfg.path("report_data") / "longitudinal.json", report)
    log.info("longitudinal analysis written", extra={"ctx": report["cohort_summary"]})
    return report


def run_monitoring(cfg: Config, df: pd.DataFrame, current: pd.DataFrame | None = None, label: str = "test_partition") -> dict[str, Any]:
    """Drift of ``current`` (default: test partition, as a stand-in for a new batch) vs the training partition."""
    parts = modelling_partitions(df, cfg)
    reference = parts["train"]
    current = parts["test"] if current is None else current
    columns = [c for c in ACOUSTIC_FEATURES + LINGUISTIC_FEATURES + CONVERSATIONAL_FEATURES if c in reference.columns]
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(), "reference": "train_partition", "current": label,
        "n_reference": int(len(reference)), "n_current": int(len(current)),
        "feature_drift": feature_drift(reference, current, columns, cfg["monitoring"]),
        "scope_note": "Input and prediction distribution shift only; concept drift cannot be assessed without new labels.",
    }
    try:
        predictor = Predictor(cfg)
        report["prediction_drift"] = prediction_drift(predictor.bundle.predict_proba(reference), predictor.bundle.predict_proba(current), cfg["monitoring"])
        report["model"] = predictor.model_info()
    except FileNotFoundError:
        report["prediction_drift"] = None
    html = cfg.path("report_data") / "evidently_drift.html"
    backend = evidently_report(reference[columns], current[columns], html)
    report["evidently"] = {"backend": backend, "html": relpath(html) if backend else None}
    write_json(cfg.path("report_data") / "drift.json", report)
    log.info("drift report written", extra={"ctx": {"drifted": report["feature_drift"]["n_drifted"]}})
    return report


def run_robustness_report(cfg: Config, df: pd.DataFrame) -> dict[str, Any]:
    predictor = Predictor(cfg)
    report = run_robustness(modelling_partitions(df, cfg)["test"], cfg, predictor)
    report["synthetic_data"] = bool(df["is_synthetic"].all())
    write_json(cfg.path("report_data") / "robustness.json", report)
    return report


def _fmt(x: Any) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def write_model_card(cfg: Config) -> str:
    """Render docs/reports/model_card_generated.md from the latest reports (static text lives in docs/MODEL_CARD.md)."""
    rd = cfg.path("report_data")
    comp = read_json(rd / "model_comparison.json")
    fairness = read_json(rd / "fairness.json") if (rd / "fairness.json").exists() else None
    unc = read_json(rd / "uncertainty.json") if (rd / "uncertainty.json").exists() else None
    robust = read_json(rd / "robustness.json") if (rd / "robustness.json").exists() else None
    splits = read_json(rd / "splits.json") if (rd / "splits.json").exists() else None
    serving = next(e for e in comp["experiments"] if e["experiment_id"] == comp["serving"]["experiment_id"])
    lines = ["# Model card — generated section", "", f"Generated {comp['generated_at']} from the latest pipeline run.", ""]
    if comp["banner"]:
        lines += [f"> **{comp['banner']}**", ""]
    t, ci = serving["test"], serving["ci"]
    lines += ["## Serving model", "", f"- Experiment: `{serving['experiment_id']}`", f"- Model: {serving['model']} ({serving['implementation']}), "
              f"strategy: {serving['strategy']}, modalities: {', '.join(serving['modalities'])}",
              f"- Features: {serving['n_features']}; calibration: {serving['calibration']}", f"- Dataset version: `{comp['dataset_version']}`", "",
              "## Test performance (held-out participants)", "", "| Metric | Value | 95% CI |", "|---|---|---|"]
    for m in ("auroc", "auprc", "balanced_accuracy", "sensitivity", "specificity", "f1", "brier", "ece"):
        c = ci.get(m, {})
        lines.append(f"| {m} | {_fmt(t.get(m))} | {_fmt(c.get('low'))} – {_fmt(c.get('high'))} |")
    if splits:
        lines += ["", "## Data partitions", "", "| Split | Participants | Recordings | Labels (participants) |", "|---|---|---|---|"]
        for name, s in splits["summary"].items():
            lines.append(f"| {name} | {s['participants']} | {s['recordings']} | {s['labels']} |")
    if unc:
        lines += ["", "## Uncertainty and abstention", "", f"- Abstention rate on test: {_fmt(unc['abstention_rate'])}",
                  f"- Out-of-distribution rate on test: {_fmt(unc['ood_rate'])}"]
    if fairness:
        lines += ["", "## Subgroup performance (participant level)", "", f"Minimum subgroup size for reporting: {fairness['min_subgroup_size']} participants.", ""]
        for attr, info in fairness["attributes"].items():
            for g in info["groups"]:
                metric = f"AUROC {_fmt(g.get('auroc'))}, sens. {_fmt(g.get('sensitivity'))}, spec. {_fmt(g.get('specificity'))}" if g["status"] == "evaluated" else "insufficient sample — not reported"
                lines.append(f"- {attr} = {g['group']} (n={g['n_participants']}): {metric}")
    if robust:
        lines += ["", "## Robustness (re-extracted features under perturbation)", "", "| Condition | AUROC | Mean |Δp| | Flip rate | Abstention |", "|---|---|---|---|---|"]
        for c in robust["conditions"]:
            lines.append(f"| {c['condition']} | {_fmt(c.get('auroc'))} | {_fmt(c.get('mean_abs_probability_shift'))} | "
                         f"{_fmt(c.get('decision_flip_rate'))} | {_fmt(c.get('abstention_rate'))} |")
    text = "\n".join(lines) + "\n"
    (cfg.path("reports") / "model_card_generated.md").write_text(text, encoding="utf-8")
    write_json(rd / "model_card.json", {"generated_markdown": text, "serving": serving["experiment_id"], "banner": comp["banner"]})
    return text
