"""Evaluation metrics, curves, calibration and participant-level bootstrap confidence intervals.

Recordings from one participant are correlated, so confidence intervals use a *cluster bootstrap*:
participants (not recordings) are resampled with replacement. With small test sets the intervals are
wide, and the reports show them so point estimates are not over-interpreted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

CI_METRICS = ("auroc", "auprc", "balanced_accuracy", "f1", "sensitivity", "specificity", "brier", "ece")


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-width bins: sum_b (n_b/N) * |mean(y_b) - mean(p_b)|."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def calibration_curve_data(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, list[float]]:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    mean_pred, frac_pos, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            mean_pred.append(float(p[mask].mean()))
            frac_pos.append(float(y[mask].mean()))
            counts.append(int(mask.sum()))
    return {"mean_predicted": mean_pred, "fraction_positive": frac_pos, "count": counts}


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5, n_bins: int = 10) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    both_classes = len(np.unique(y)) == 2
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()) if len(y) else float("nan"),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both_classes else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if both_classes else float("nan"),
        "auprc": float(average_precision_score(y, p)) if both_classes else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p, n_bins),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": threshold,
    }


def curves(y: np.ndarray, p: np.ndarray, n_bins: int = 10, max_points: int = 200) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    out: dict[str, Any] = {"calibration": calibration_curve_data(y, np.asarray(p), n_bins)}
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        step_r = max(1, len(fpr) // max_points)
        step_p = max(1, len(prec) // max_points)
        out["roc"] = {"fpr": fpr[::step_r].tolist() + [1.0], "tpr": tpr[::step_r].tolist() + [1.0]}
        out["pr"] = {"recall": rec[::step_p].tolist(), "precision": prec[::step_p].tolist()}
    return out


def cluster_bootstrap_ci(
    y: np.ndarray, p: np.ndarray, groups: np.ndarray, n_boot: int = 300, level: float = 0.95, seed: int = 42,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Percentile CIs from resampling participants with replacement."""
    rng = np.random.default_rng(seed)
    y, p, groups = np.asarray(y).astype(int), np.asarray(p), np.asarray(groups)
    unique = np.unique(groups)
    index_by_group = {g: np.where(groups == g)[0] for g in unique}
    samples: dict[str, list[float]] = {m: [] for m in CI_METRICS}
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        m = binary_metrics(y[idx], p[idx], threshold)
        for k in CI_METRICS:
            samples[k].append(m[k])
    alpha = (1 - level) / 2
    out = {}
    for k, vals in samples.items():
        vals_arr = np.asarray([v for v in vals if np.isfinite(v)])
        out[k] = (
            {"low": float(np.quantile(vals_arr, alpha)), "high": float(np.quantile(vals_arr, 1 - alpha)), "n_boot": int(len(vals_arr))}
            if len(vals_arr) >= 20 else {"low": float("nan"), "high": float("nan"), "n_boot": int(len(vals_arr))}
        )
    return out


def participant_level(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average recording probabilities per participant (labels are participant-level)."""
    import pandas as pd

    df = pd.DataFrame({"y": y, "p": p, "g": groups}).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
    return df["y"].to_numpy(), df["p"].to_numpy(), df.index.to_numpy()
