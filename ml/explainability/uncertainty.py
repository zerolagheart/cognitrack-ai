"""Uncertainty, out-of-distribution detection and abstention.

A prediction is withheld from automated display ("abstain → clinician review") when any of these hold:

* the calibrated probability lies inside the configured uncertainty band;
* models trained on the same features disagree beyond a threshold (ensemble disagreement);
* the input is far from the training distribution (Mahalanobis distance above the training quantile);
* the audio failed quality checks or too many features are missing.

None of these make a prediction *correct*; they only flag cases where it is especially unreliable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class MahalanobisOOD:
    """Distance-based OOD detector on standardised, imputed features with shrinkage covariance."""

    def __init__(self, quantile: float = 0.99) -> None:
        self.quantile = quantile

    def fit(self, X: pd.DataFrame) -> MahalanobisOOD:
        self.columns_ = list(X.columns)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        self.scaler_ = StandardScaler().fit(self.imputer_.transform(X))
        Z = self.scaler_.transform(self.imputer_.transform(X))
        self.cov_ = LedoitWolf().fit(Z)
        d = self.cov_.mahalanobis(Z)
        self.threshold_ = float(np.quantile(d, self.quantile))
        self.train_distances_ = d
        return self

    def distance(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.scaler_.transform(self.imputer_.transform(X[self.columns_]))
        return self.cov_.mahalanobis(np.nan_to_num(Z))

    def percentile(self, d: np.ndarray) -> np.ndarray:
        return np.searchsorted(np.sort(self.train_distances_), d) / len(self.train_distances_)


@dataclass
class UncertaintyAssessment:
    probability: float
    ensemble_std: float | None
    ood_distance: float | None
    ood_percentile: float | None
    missing_feature_fraction: float
    quality_flags: list[str]
    reasons: list[str] = field(default_factory=list)

    @property
    def abstain(self) -> bool:
        return bool(self.reasons)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["abstain"] = self.abstain
        d["recommendation"] = "refer_for_clinician_review" if self.abstain else "display_with_caveats"
        return d


def assess(
    probability: float, member_probabilities: list[float] | None, ood: MahalanobisOOD | None, X_row: pd.DataFrame,
    quality_flags: list[str], eval_cfg: dict[str, Any], max_missing_fraction: float = 0.3,
) -> UncertaintyAssessment:
    lo, hi = eval_cfg["abstention_band"]
    std = float(np.std(member_probabilities)) if member_probabilities and len(member_probabilities) > 1 else None
    dist = pct = None
    if ood is not None:
        dist = float(ood.distance(X_row)[0])
        pct = float(ood.percentile(np.array([dist]))[0])
    missing = float(X_row.isna().mean(axis=1).iloc[0]) if X_row.shape[1] else 1.0
    a = UncertaintyAssessment(probability, std, dist, pct, missing, quality_flags)
    if lo <= probability <= hi:
        a.reasons.append(f"probability {probability:.2f} inside uncertainty band [{lo}, {hi}]")
    if std is not None and std > eval_cfg["ensemble_disagreement_threshold"]:
        a.reasons.append(f"ensemble disagreement (std {std:.2f})")
    if ood is not None and dist is not None and dist > ood.threshold_:
        a.reasons.append("input outside the training distribution (Mahalanobis distance)")
    if missing > max_missing_fraction:
        a.reasons.append(f"{missing:.0%} of model features missing")
    blocking = {"silent", "too_short", "corrupted", "clipping", "low_snr", "low_level"} & set(quality_flags)
    if blocking:
        a.reasons.append("audio quality: " + ",".join(sorted(blocking)))
    return a
