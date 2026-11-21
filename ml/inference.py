"""Inference service: features → calibrated probability + explanation + uncertainty assessment.

Shared by the API and the robustness experiments. Output is decision *support* for a qualified
reviewer: it never states a diagnosis, and uncertain / out-of-distribution / low-quality cases are
explicitly marked for clinician review instead of being presented as results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml.config import Config
from ml.embeddings.encoders import AudioEncoder, EncoderUnavailable, build_encoder
from ml.explainability.explain import ShapExplainer
from ml.explainability.uncertainty import assess
from ml.features.extraction import RecordingFeatures
from ml.fusion.strategies import AttentionFusionClassifier, LateFusionClassifier
from ml.training.registry import ModelRegistry
from ml.utils import get_logger

log = get_logger(__name__)

DISCLAIMER = (
    "Research decision-support output. Not a diagnosis and not a medical device. A statistical estimate "
    "from speech features that must be interpreted by a qualified clinician alongside clinical assessment."
)


@dataclass
class Prediction:
    probability: float
    model_label: str  # model output category, phrased as a screening-style signal (not a diagnosis)
    uncertainty: dict[str, Any]
    explanation: dict[str, Any]
    model: dict[str, Any]
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class Predictor:
    def __init__(self, cfg: Config, registry: ModelRegistry | None = None) -> None:
        self.cfg = cfg
        self.registry = registry or ModelRegistry(cfg.path("models"))
        self.index = self.registry.index()
        self.bundle, self.members = self.registry.serving()
        self._encoder: AudioEncoder | None = None
        self._encoder_error: str | None = None
        self._explainer: ShapExplainer | None = None

    @property
    def uses_embeddings(self) -> bool:
        return any(c.startswith("emb_") for c in self.bundle.columns)

    def model_info(self) -> dict[str, Any]:
        meta = self.bundle.metadata
        return {
            "experiment_id": self.bundle.experiment_id, "model": meta["model"], "strategy": meta["strategy"],
            "modalities": meta["modalities"], "dataset_version": meta["dataset_version"],
            "calibration": meta.get("calibration_method"), "trained_at": meta.get("trained_at"),
            "synthetic_training_data": self.index.get("synthetic_data", False),
            "embedding_encoder": self.index.get("embedding_encoder"),
        }

    def encoder(self) -> AudioEncoder | None:
        if self._encoder is None and self._encoder_error is None:
            try:
                self._encoder = build_encoder(self.cfg["embeddings"]["encoder"], self.cfg["embeddings"], str(self.cfg.path("model_cache")))
                trained_with = self.index.get("embedding_encoder")
                if trained_with and trained_with != self._encoder.version:
                    self._encoder_error = f"encoder version mismatch (model trained with {trained_with})"
                    self._encoder = None
            except (EncoderUnavailable, KeyError) as exc:
                self._encoder_error = str(exc)
        return self._encoder

    def embed(self, signal: np.ndarray, sr: int) -> dict[str, float]:
        enc = self.encoder() if self.uses_embeddings else None
        if enc is None:
            return {}
        vec = enc.encode_batch([signal], sr)[0]
        return {f"emb_{i:03d}": float(v) for i, v in enumerate(vec)}

    def _row(self, features: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame([{c: features.get(c, np.nan) for c in self.bundle.columns}], dtype=float)

    def explain(self, X_row: pd.DataFrame) -> dict[str, Any]:
        spec = self.bundle.spec
        base = self.bundle.model.base
        if isinstance(base, LateFusionClassifier):
            contrib = base.modality_contributions(X_row).iloc[0].to_dict()
            return {"method": "late_fusion_meta_coefficients", "modality_contributions": contrib, "output_space": "log-odds"}
        if isinstance(base, AttentionFusionClassifier):
            return {"method": "attention_weights", "modality_attention": base.modality_attention(X_row).iloc[0].to_dict()}
        if ShapExplainer.supports(spec["model"]):
            if self._explainer is None:
                self._explainer = ShapExplainer(base, spec["model"], self.bundle.background)
            return {"method": "shap", **self._explainer.local(X_row)}
        return {"method": "none", "note": "Local explanations are not available for this model type; see global permutation importance."}

    def predict_features(self, features: dict[str, Any], quality_flags: list[str] | None = None) -> Prediction:
        X_row = self._row(features)
        p = float(self.bundle.predict_proba(X_row)[0])
        member_p = [p] + [float(m.predict_proba(self._row(features))[0]) for m in self.members]
        handcrafted = self.bundle.ood.columns_ if self.bundle.ood is not None else []
        ood_row = pd.DataFrame([{c: features.get(c, np.nan) for c in handcrafted}], dtype=float)
        unc = assess(p, member_p, self.bundle.ood, ood_row, quality_flags or [], self.cfg["evaluation"])
        label = "elevated_signal" if p >= 0.5 else "no_elevated_signal"
        explanation = self.explain(X_row)
        if self._encoder_error and self.uses_embeddings:
            explanation["warnings"] = [f"embedding features unavailable: {self._encoder_error}"]
        return Prediction(p, label, unc.to_dict(), explanation, self.model_info())

    def predict_recording(self, rec: RecordingFeatures, signal: np.ndarray | None = None, sr: int | None = None) -> Prediction:
        feats = dict(rec.features)
        if signal is not None and sr is not None:
            feats.update(self.embed(signal, sr))
        return self.predict_features(feats, rec.quality.get("flags", []))
