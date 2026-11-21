"""Stages 10–11: evaluation reports for the registered models.

Writes machine-readable reports to ``artifacts/reports`` (consumed by the API/dashboard) and a
human-readable model comparison to ``docs/reports``. Every number in these files is computed from the
current run; when the training data is synthetic the reports say so prominently.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ml.config import Config
from ml.evaluation.fairness import subgroup_metrics
from ml.evaluation.metrics import binary_metrics
from ml.explainability.explain import ShapExplainer, grouped_permutation_importance
from ml.explainability.uncertainty import assess
from ml.features.registry import label_for
from ml.training.registry import ModelRegistry
from ml.training.trainer import encode_labels, modelling_partitions
from ml.utils import get_logger, write_json

log = get_logger(__name__)

SYNTHETIC_BANNER = (
    "SYNTHETIC DATA: these results come from programmatically generated data with group differences built "
    "in by construction. They demonstrate that the pipeline works and are NOT evidence of clinical validity."
)


def paired_bootstrap_delta(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, groups: np.ndarray,
                           n_boot: int, seed: int) -> dict[str, float]:
    """AUROC(a) - AUROC(b) on the same test participants with a participant cluster bootstrap CI."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    d = np.array(deltas)
    point = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    return {"delta_auroc": float(point), "ci_low": float(np.quantile(d, 0.025)), "ci_high": float(np.quantile(d, 0.975)),
            "p_delta_le_0": float(np.mean(d <= 0))}


def _summary_row(r: dict[str, Any]) -> dict[str, Any]:
    base = {k: r[k] for k in ("experiment_id", "modality_set", "modalities", "strategy", "model", "status")}
    if r["status"] != "ok":
        return {**base, "reason": r.get("reason")}
    rec = r["test"]["recording"]
    return {
        **base, "implementation": r["implementation"], "n_features": r["n_features"], "calibration": r["calibration_method"],
        "cv_auroc": r["cv"]["auroc"], "cv_balanced_accuracy": r["cv"]["balanced_accuracy"],
        "test": {k: rec[k] for k in ("auroc", "auprc", "accuracy", "balanced_accuracy", "precision", "recall", "sensitivity",
                                      "specificity", "f1", "brier", "ece", "confusion_matrix", "n", "prevalence")},
        "test_uncalibrated": {k: r["test"]["recording_uncalibrated"][k] for k in ("brier", "ece", "auroc")},
        "test_participant": {k: r["test"]["participant"][k] for k in ("auroc", "balanced_accuracy", "sensitivity", "specificity", "brier", "n")},
        "ci": r["test"]["ci"], "curves": r["test"]["curves"], "mlflow_run_id": r.get("mlflow_run_id"),
        "duration_s": r["duration_s"],
    }


def _ablation(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for r in results.values():
        if r["status"] == "ok" and (r["modality_set"] not in best or r["cv"]["auroc"] > best[r["modality_set"]]["cv"]["auroc"]):
            best[r["modality_set"]] = r
    rows = []
    for name, r in best.items():
        rows.append({
            "modality_set": name, "modalities": r["modalities"], "best_experiment": r["experiment_id"], "cv_auroc": r["cv"]["auroc"],
            "test_auroc": r["test"]["recording"]["auroc"], "test_auroc_ci": r["test"]["ci"]["auroc"],
            "test_balanced_accuracy": r["test"]["recording"]["balanced_accuracy"], "test_brier": r["test"]["recording"]["brier"],
        })
    return sorted(rows, key=lambda x: -x["cv_auroc"])


def evaluate(cfg: Config, df: pd.DataFrame) -> dict[str, Any]:
    registry = ModelRegistry(cfg.path("models"))
    index = registry.index()
    results = index["experiments"]
    serving_id = index["serving"]["experiment_id"]
    seed = cfg["project"]["seed"]
    ecfg = cfg["evaluation"]
    out_dir = cfg.path("report_data")
    parts = modelling_partitions(df, cfg)
    test = parts["test"]
    y_te, g_te = encode_labels(test, cfg), test["participant_id"].to_numpy()
    synthetic = bool(df["is_synthetic"].all())
    banner = SYNTHETIC_BANNER if synthetic else None

    comparison = {
        "generated_at": datetime.now(UTC).isoformat(), "synthetic_data": synthetic, "banner": banner,
        "dataset_version": index["dataset_version"], "serving": index["serving"],
        "experiments": [_summary_row(r) for r in results.values()],
    }
    ablation = _ablation(results)
    # Does fusion help? Compare the full set's best experiment to the best single-modality experiment.
    singles = [a for a in ablation if len(a["modalities"]) == 1]
    full = next((a for a in ablation if a["modality_set"] == "full"), None)
    fusion_test = None
    if singles and full:
        best_single = max(singles, key=lambda a: a["cv_auroc"])
        p_full = registry.load_bundle(full["best_experiment"]).predict_proba(test)
        p_single = registry.load_bundle(best_single["best_experiment"]).predict_proba(test)
        fusion_test = {"full": full["best_experiment"], "best_single": best_single["best_experiment"],
                       **paired_bootstrap_delta(y_te, p_full, p_single, g_te, ecfg["n_bootstrap"], seed)}
    comparison["ablation"] = ablation
    comparison["fusion_vs_best_single"] = fusion_test
    write_json(out_dir / "model_comparison.json", comparison)

    # Serving-model predictions for every labelled, usable recording (split recorded; only test is unbiased).
    serving = registry.load_bundle(serving_id)
    members = [registry.load_bundle(e) for e in index["serving"]["ensemble"] if e != serving_id]
    usable = df[df["quality_usable"].fillna(False).astype(bool)].reset_index(drop=True)
    probs = serving.predict_proba(usable)
    member_probs = np.column_stack([probs] + [m.predict_proba(usable) for m in members]) if members else probs[:, None]
    ood_cols = serving.ood.columns_
    ood_dist = serving.ood.distance(usable[ood_cols])
    pred_rows = []
    for i, row in usable.iterrows():
        flags = [f for f in str(row["quality_flags"]).split(",") if f]
        unc = assess(float(probs[i]), member_probs[i].tolist(), serving.ood, usable.loc[[i], ood_cols], flags, ecfg)
        pred_rows.append({
            "recording_id": row["recording_id"], "participant_id": row["participant_id"], "split": row["split"],
            "probability": float(probs[i]), "ensemble_std": unc.ensemble_std, "ood_distance": float(ood_dist[i]),
            "abstain": unc.abstain, "abstain_reasons": "; ".join(unc.reasons), "model_experiment_id": serving_id,
        })
    predictions = pd.DataFrame(pred_rows)
    predictions.to_parquet(cfg.path("processed") / "predictions.parquet", index=False)

    test_pred = predictions[predictions["split"] == "test"].merge(
        test[["recording_id", "sex", "age_band", "education_band", "language"]], on="recording_id")
    test_pred["y_true"] = (test_pred["recording_id"].map(dict(zip(test["recording_id"], y_te, strict=True)))).astype(int)
    kept = test_pred[~test_pred["abstain"]]
    uncertainty = {
        "abstention_rate": float(test_pred["abstain"].mean()),
        "ood_rate": float((test_pred["ood_distance"] > serving.ood.threshold_).mean()),
        "metrics_all": binary_metrics(test_pred["y_true"], test_pred["probability"]),
        "metrics_non_abstained": binary_metrics(kept["y_true"], kept["probability"]) if kept["y_true"].nunique() == 2 else None,
        "reason_counts": test_pred["abstain_reasons"].str.split("; ").explode().replace("", np.nan).dropna()
        .str.replace(r"\s*\(.*\)|\s[\d.]+.*", "", regex=True).value_counts().to_dict(),
        "abstention_band": ecfg["abstention_band"],
    }
    fairness = subgroup_metrics(test_pred, min_size=ecfg["min_subgroup_size"])
    fairness["banner"] = banner

    explain: dict[str, Any] = {"serving_experiment": serving_id, "banner": banner}
    perm_groups = serving.modality_columns
    explain["modality_permutation_importance"] = grouped_permutation_importance(serving.model, test[serving.columns], y_te, perm_groups, 10, seed)
    top_cols = [c for c in serving.columns if not c.startswith("emb_")]
    feat_perm = grouped_permutation_importance(serving.model, test[serving.columns], y_te, {c: [c] for c in top_cols}, 5, seed)
    explain["feature_permutation_importance"] = sorted(
        [{"feature": k, "label": label_for(k), **v} for k, v in feat_perm.items()], key=lambda x: -x["auroc_drop_mean"])[:25]
    # SHAP on the serving model if supported; otherwise on the best SHAP-capable experiment of the same modality set.
    shap_candidates = [r for r in results.values() if r["status"] == "ok" and r["strategy"] in ("single", "early")
                       and ShapExplainer.supports(r["model"])]
    same_set = [r for r in shap_candidates if r["modality_set"] == serving.spec["modality_set"]]
    pool = same_set or shap_candidates
    if pool:
        shap_exp = serving_id if serving_id in {r["experiment_id"] for r in pool} else max(pool, key=lambda r: r["cv"]["auroc"])["experiment_id"]
        b = registry.load_bundle(shap_exp)
        explainer = ShapExplainer(b.model.base, b.spec["model"], b.background)
        X_test = test[b.columns]
        explain["shap"] = {
            "experiment_id": shap_exp, "is_serving_model": shap_exp == serving_id, "base_value": explainer.expected_value,
            "global_importance": explainer.global_importance(X_test)[:30],
            "modality_importance": explainer.modality_importance(X_test),
            "interactions": explainer.interactions(X_test),
            "output_space": "probability" if b.spec["model"] == "random_forest" else "log-odds",
        }
    write_json(out_dir / "explainability.json", explain)
    write_json(out_dir / "fairness.json", fairness)
    write_json(out_dir / "uncertainty.json", uncertainty)
    write_model_comparison_md(cfg, comparison)
    log.info("evaluation reports written", extra={"ctx": {"serving": serving_id}})
    return {"comparison": comparison, "fairness": fairness, "uncertainty": uncertainty, "explainability": explain}


def _fmt(x: Any) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def write_model_comparison_md(cfg: Config, comparison: dict[str, Any]) -> None:
    lines = ["# Model comparison report (generated)", "",
             f"Generated: {comparison['generated_at']} · dataset version `{comparison['dataset_version']}`", ""]
    if comparison["banner"]:
        lines += [f"> **{comparison['banner']}**", ""]
    s = comparison["serving"]
    lines += [f"Serving model: **{s['experiment_id']}** (selected by {s['selection_metric']} = {_fmt(s['chosen_cv_score'])}; "
              f"best CV score {_fmt(s['best_cv_score'])}; simplicity margin {s['simplicity_margin']}).", "",
              "## Modality ablation (best experiment per modality set, ranked by CV AUROC)", "",
              "| Modality set | Best experiment | CV AUROC | Test AUROC [95% CI] | Test bal. acc. | Test Brier |",
              "|---|---|---|---|---|---|"]
    for a in comparison["ablation"]:
        ci = a["test_auroc_ci"]
        lines.append(f"| {a['modality_set']} | `{a['best_experiment']}` | {_fmt(a['cv_auroc'])} | {_fmt(a['test_auroc'])} "
                     f"[{_fmt(ci['low'])}, {_fmt(ci['high'])}] | {_fmt(a['test_balanced_accuracy'])} | {_fmt(a['test_brier'])} |")
    f = comparison.get("fusion_vs_best_single")
    if f:
        lines += ["", f"**Fusion vs best single modality** (`{f['full']}` − `{f['best_single']}`): ΔAUROC = {_fmt(f['delta_auroc'])} "
                  f"[95% CI {_fmt(f['ci_low'])}, {_fmt(f['ci_high'])}], bootstrap P(Δ ≤ 0) = {_fmt(f['p_delta_le_0'])}. "
                  "A CI that includes 0 means the test set does not show that fusion helps."]
    lines += ["", "## All experiments", "",
              "| Experiment | Status | CV AUROC | Test AUROC | AUPRC | Sens. | Spec. | F1 | Brier | ECE |", "|---|---|---|---|---|---|---|---|---|---|"]
    for e in sorted(comparison["experiments"], key=lambda e: -(e.get("cv_auroc") or 0)):
        if e["status"] != "ok":
            lines.append(f"| `{e['experiment_id']}` | skipped: {e.get('reason')} | | | | | | | | |")
            continue
        t = e["test"]
        lines.append(f"| `{e['experiment_id']}` | ok | {_fmt(e['cv_auroc'])} | {_fmt(t['auroc'])} | {_fmt(t['auprc'])} | "
                     f"{_fmt(t['sensitivity'])} | {_fmt(t['specificity'])} | {_fmt(t['f1'])} | {_fmt(t['brier'])} | {_fmt(t['ece'])} |")
    lines += ["", "Metrics are recording-level on the held-out test participants at threshold 0.5 on calibrated probabilities. "
              "CIs: participant-level cluster bootstrap. See docs/MODEL_EVALUATION.md for methodology."]
    (cfg.path("reports") / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
