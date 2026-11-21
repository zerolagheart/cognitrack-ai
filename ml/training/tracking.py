"""Thin MLflow wrapper: local SQLite backend + local artifact directory (no server required).

Browse runs with ``mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db``.

If MLflow cannot be imported or initialised, tracking degrades to a no-op with a warning and the
experiment JSON written by the trainer remains the record of truth.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ml.utils import get_logger

log = get_logger(__name__)


class Tracker:
    def __init__(self, tracking_dir: Path, experiment_name: str) -> None:
        self.enabled = False
        try:
            import mlflow

            tracking_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(f"sqlite:///{(tracking_dir / 'mlflow.db').resolve().as_posix()}")
            if mlflow.get_experiment_by_name(experiment_name) is None:
                mlflow.create_experiment(experiment_name, artifact_location=(tracking_dir / "artifacts").resolve().as_uri())
            mlflow.set_experiment(experiment_name)
            self.mlflow = mlflow
            self.enabled = True
        except Exception as exc:
            log.warning("MLflow tracking disabled", extra={"ctx": {"reason": f"{type(exc).__name__}: {exc}"}})

    @contextlib.contextmanager
    def run(self, run_name: str, tags: dict[str, str] | None = None) -> Iterator[str | None]:
        if not self.enabled:
            yield None
            return
        with self.mlflow.start_run(run_name=run_name, tags=tags) as active:
            yield active.info.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        if self.enabled:
            flat = {k: (str(v)[:500]) for k, v in params.items()}
            self.mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if self.enabled:
            clean = {k: float(v) for k, v in metrics.items() if isinstance(v, int | float) and v == v}
            self.mlflow.log_metrics(clean)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if self.enabled:
            self.mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_dict(self, obj: dict[str, Any], name: str) -> None:
        if self.enabled:
            self.mlflow.log_dict(obj, name)

    def register(self, run_id: str, artifact_path: str, model_name: str) -> str | None:
        """Register a logged artifact directory in the MLflow model registry (best effort)."""
        if not self.enabled or run_id is None:
            return None
        try:
            mv = self.mlflow.register_model(f"runs:/{run_id}/{artifact_path}", model_name)
            return str(mv.version)
        except Exception as exc:  # registry unsupported by the configured store
            log.warning("MLflow model registration skipped", extra={"ctx": {"reason": f"{type(exc).__name__}"}})
            return None
