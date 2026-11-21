"""Model explanations: SHAP (global, local, interactions), permutation importance and modality attribution.

Explanations are computed on the *uncalibrated* fitted pipeline (log-odds / margin space). Calibration
is a monotonic mapping applied afterwards, so feature rankings and signs are preserved, but SHAP values
are not probabilities. Explanations describe the model's behaviour, not causal clinical mechanisms.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from ml.features.registry import feature_modality, label_for
from ml.models.factory import TREE_MODELS, transformed_feature_names
from ml.utils import get_logger

log = get_logger(__name__)


def _normalise_shap(values: Any) -> np.ndarray:
    """Return positive-class SHAP values of shape (n, f) across shap/library versions."""
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., 1]
    return values


class ShapExplainer:
    """Wraps a fitted preprocessing+estimator pipeline."""

    def __init__(self, pipeline: Pipeline, model_name: str, background: pd.DataFrame) -> None:
        import shap

        self.pipeline = pipeline
        self.model_name = model_name
        self.pre = pipeline.named_steps["pre"]
        self.clf = pipeline.named_steps["clf"]
        self.feature_names = transformed_feature_names(pipeline)
        bg = self.pre.transform(background)
        if model_name in TREE_MODELS:
            self.explainer = shap.TreeExplainer(self.clf)
            self.kind = "tree"
        elif model_name == "logreg":
            self.explainer = shap.LinearExplainer(self.clf, bg)
            self.kind = "linear"
        else:
            raise ValueError(f"SHAP explanations are not configured for '{model_name}'; use permutation importance.")
        ev = self.explainer.expected_value
        self.expected_value = float(np.ravel(ev)[-1]) if np.ndim(ev) else float(ev)

    @staticmethod
    def supports(model_name: str) -> bool:
        return model_name in TREE_MODELS or model_name == "logreg"

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        return _normalise_shap(self.explainer.shap_values(self.pre.transform(X)))

    def global_importance(self, X: pd.DataFrame) -> list[dict[str, Any]]:
        sv = self.shap_values(X)
        Xt = self.pre.transform(X)
        mean_abs = np.abs(sv).mean(axis=0)
        out = []
        for j in np.argsort(-mean_abs):
            corr = np.corrcoef(Xt[:, j], sv[:, j])[0, 1] if np.std(Xt[:, j]) > 0 and np.std(sv[:, j]) > 0 else np.nan
            out.append({
                "feature": self.feature_names[j], "label": label_for(self.feature_names[j]),
                "modality": feature_modality(self.feature_names[j]), "mean_abs_shap": float(mean_abs[j]),
                # Sign of association between feature value and model output (direction, not causation).
                "direction": float(corr) if np.isfinite(corr) else None,
            })
        return out

    def modality_importance(self, X: pd.DataFrame) -> dict[str, float]:
        sv = np.abs(self.shap_values(X))
        totals: dict[str, float] = {}
        for j, name in enumerate(self.feature_names):
            m = feature_modality(name)
            totals[m] = totals.get(m, 0.0) + float(sv[:, j].mean())
        s = sum(totals.values()) or 1.0
        return {m: v / s for m, v in totals.items()}

    def local(self, X_row: pd.DataFrame, top_k: int = 10) -> dict[str, Any]:
        sv = self.shap_values(X_row)[0]
        xt = self.pre.transform(X_row)[0]
        order = np.argsort(-np.abs(sv))
        raw = X_row.iloc[0]
        contributions = [{
            "feature": self.feature_names[j], "label": label_for(self.feature_names[j]),
            "modality": feature_modality(self.feature_names[j]), "shap": float(sv[j]),
            "value": float(raw[self.feature_names[j]]) if self.feature_names[j] in raw.index and pd.notna(raw[self.feature_names[j]]) else None,
            "standardised_value": float(xt[j]),
        } for j in order[:top_k]]
        by_modality: dict[str, float] = {}
        for j, name in enumerate(self.feature_names):
            by_modality[feature_modality(name)] = by_modality.get(feature_modality(name), 0.0) + float(sv[j])
        return {"base_value": self.expected_value, "contributions": contributions, "modality_contributions": by_modality,
                "output_space": "probability" if self.model_name == "random_forest" else "log-odds"}

    def interactions(self, X: pd.DataFrame, top_k: int = 10, max_rows: int = 100) -> list[dict[str, Any]]:
        if self.kind != "tree" or self.model_name == "random_forest":
            return []  # RF interaction values are prohibitively slow; linear models have none
        Xt = self.pre.transform(X.iloc[:max_rows])
        try:
            inter = self.explainer.shap_interaction_values(Xt)
        except Exception as exc:  # library/version specific
            log.warning("SHAP interaction values unavailable", extra={"ctx": {"reason": str(exc)}})
            return []
        inter = np.asarray(inter[1] if isinstance(inter, list) else inter)
        if inter.ndim == 4:
            inter = inter[..., 1]
        mean_abs = np.abs(inter).mean(axis=0)
        np.fill_diagonal(mean_abs, 0)
        iu = np.triu_indices_from(mean_abs, k=1)
        order = np.argsort(-mean_abs[iu])[:top_k]
        return [{"feature_a": self.feature_names[iu[0][k]], "feature_b": self.feature_names[iu[1][k]],
                 "mean_abs_interaction": float(mean_abs[iu][k])} for k in order]


def grouped_permutation_importance(
    model: Any, X: pd.DataFrame, y: np.ndarray, groups: dict[str, list[str]], n_repeats: int = 10, seed: int = 42,
) -> dict[str, dict[str, float]]:
    """AUROC drop when all columns of a group are jointly permuted (rows shuffled together).

    With ``groups`` = modalities this measures each modality's contribution on held-out data; with
    singleton groups it equals ordinary permutation importance.
    """
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    out = {}
    for name, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        out[name] = {"auroc_drop_mean": float(np.mean(drops)), "auroc_drop_std": float(np.std(drops))}
    return out
