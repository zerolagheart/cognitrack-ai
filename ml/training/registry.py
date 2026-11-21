"""Local model registry: model bundles on disk plus a JSON index naming the serving model.

The backend serves from this registry (no MLflow server needed at inference time). MLflow holds the
same artifacts for experiment browsing; the ``mlflow_run_id`` in each entry links the two.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ml.utils import read_json, write_json


class ModelNotFoundError(FileNotFoundError):
    pass


class CalibratedModel:
    """Base classifier + post-hoc calibrator fitted on held-out (validation) participants."""

    def __init__(self, base: Any, method: str) -> None:
        self.base = base
        self.method = method
        self.calibrator: Any = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> CalibratedModel:
        p = self.base.predict_proba(X)[:, 1]
        if len(np.unique(y)) < 2 or self.method == "none":
            self.method = "none"
            return self
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        else:
            self.method = "sigmoid"
            self.calibrator = LogisticRegression(C=1e3).fit(self._logit(p).reshape(-1, 1), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.base.predict_proba(X)[:, 1]
        if self.calibrator is None:
            q = p
        elif self.method == "isotonic":
            q = self.calibrator.predict(p)
        else:
            q = self.calibrator.predict_proba(self._logit(p).reshape(-1, 1))[:, 1]
        return np.column_stack([1 - q, q])


@dataclass
class ModelBundle:
    experiment_id: str
    model: CalibratedModel
    columns: list[str]
    modality_columns: dict[str, list[str]]
    spec: dict[str, Any]
    ood: Any
    background: pd.DataFrame
    positive_label: str
    metadata: dict[str, Any]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X.reindex(columns=self.columns))[:, 1]


class ModelRegistry:
    def __init__(self, models_dir: Path) -> None:
        self.dir = Path(models_dir)
        self.index_path = self.dir / "registry.json"

    def index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            raise ModelNotFoundError("No trained models registered. Run: cognitrack train")
        return read_json(self.index_path)

    def write_index(self, index: dict[str, Any]) -> None:
        write_json(self.index_path, index)

    def save_bundle(self, bundle: ModelBundle) -> Path:
        out = self.dir / bundle.experiment_id
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, out / "model.joblib", compress=3)
        write_json(out / "metadata.json", bundle.metadata)
        return out

    def load_bundle(self, experiment_id: str) -> ModelBundle:
        path = self.dir / experiment_id / "model.joblib"
        if not path.exists():
            raise ModelNotFoundError(f"Model artifact for '{experiment_id}' not found at {path}")
        return joblib.load(path)

    def serving(self) -> tuple[ModelBundle, list[ModelBundle]]:
        idx = self.index()
        serving = self.load_bundle(idx["serving"]["experiment_id"])
        members = [self.load_bundle(e) for e in idx["serving"].get("ensemble", []) if e != serving.experiment_id]
        return serving, members
