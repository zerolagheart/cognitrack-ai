"""Experiment framework: modality ablations × models × fusion strategies with a common interface.

Protocol (identical for every experiment, so results are comparable):

1. Participant-independent train / validation / test partitions (``ml.data.splitting``).
2. Model selection evidence = participant-grouped stratified K-fold CV on the training partition.
3. Final model fitted on the full training partition.
4. Post-hoc probability calibration on the validation partition.
5. One evaluation on the untouched test partition, with participant-level cluster-bootstrap CIs.

The serving model is chosen by CV score (never by test score), preferring the *simplest* configuration
within ``SIMPLICITY_MARGIN`` of the best, so additional complexity has to earn its place.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ml.config import Config
from ml.data.splitting import assert_no_leakage, group_kfold_indices
from ml.evaluation.metrics import binary_metrics, cluster_bootstrap_ci, curves, participant_level
from ml.explainability.uncertainty import MahalanobisOOD
from ml.features.registry import columns_for
from ml.fusion.strategies import AttentionFusionClassifier, LateFusionClassifier
from ml.models.factory import build_pipeline
from ml.models.mlp import TORCH_AVAILABLE
from ml.training.registry import CalibratedModel, ModelBundle, ModelRegistry
from ml.training.tracking import Tracker
from ml.utils import dataframe_fingerprint, get_logger, set_seed

log = get_logger(__name__)

SIMPLICITY_MARGIN = 0.01
STRATEGY_RANK = {"single": 0, "early": 1, "late": 2, "attention": 3}
MODEL_RANK = {"logreg": 0, "random_forest": 1, "lightgbm": 2, "xgboost": 3, "mlp": 4, "attention": 5}


@dataclass
class ExperimentSpec:
    modality_set: str
    modalities: list[str]
    strategy: str  # single | early | late | attention
    model: str

    @property
    def experiment_id(self) -> str:
        return f"{self.modality_set}__{self.strategy}__{self.model}"

    @property
    def complexity(self) -> tuple[int, int, int]:
        return len(self.modalities), STRATEGY_RANK[self.strategy], MODEL_RANK.get(self.model, 9)


def experiment_specs(cfg: Config) -> list[ExperimentSpec]:
    tcfg = cfg["training"]
    specs = []
    for set_name, modalities in tcfg["modality_sets"].items():
        if len(modalities) == 1:
            specs += [ExperimentSpec(set_name, modalities, "single", m) for m in tcfg["models"]]
            continue
        if "early" in tcfg["fusion_strategies"]:
            specs += [ExperimentSpec(set_name, modalities, "early", m) for m in tcfg["models"]]
        if "late" in tcfg["fusion_strategies"]:
            specs += [ExperimentSpec(set_name, modalities, "late", m) for m in tcfg["late_fusion_base_models"]]
        if tcfg.get("attention_fusion") and TORCH_AVAILABLE:
            specs.append(ExperimentSpec(set_name, modalities, "attention", "attention"))
    return specs


def encode_labels(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    return (df[cfg["split"]["label_column"]] == cfg["split"]["positive_label"]).astype(int).to_numpy()


def modelling_partitions(df: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Labelled, quality-usable rows per partition (unusable audio is excluded and counted)."""
    assert_no_leakage(df)
    label_col = cfg["split"]["label_column"]
    usable = df["quality_usable"].fillna(False).astype(bool) & df[label_col].notna()
    return {s: df[usable & (df["split"] == s)].reset_index(drop=True) for s in ("train", "val", "test")}


def _make_estimator(spec: ExperimentSpec, cols_by_mod: dict[str, list[str]], y: np.ndarray, cfg: Config) -> tuple[Any, str]:
    seed = cfg["project"]["seed"]
    mlp_params = cfg["training"]["mlp"]
    if spec.strategy in ("single", "early"):
        cols = [c for cs in cols_by_mod.values() for c in cs]
        return build_pipeline(spec.model, cols, y, seed, mlp_params)
    if spec.strategy == "late":
        return LateFusionClassifier(cols_by_mod, spec.model, cfg["training"]["cv_folds"], seed, mlp_params), "sklearn-stacking"
    if spec.strategy == "attention":
        return AttentionFusionClassifier(cols_by_mod, random_state=seed), "torch"
    raise KeyError(spec.strategy)


def _fit(estimator: Any, spec: ExperimentSpec, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> Any:
    if spec.strategy in ("late", "attention"):
        return estimator.fit(X, y, groups=groups)
    if spec.model == "mlp" and TORCH_AVAILABLE:
        return estimator.fit(X, y, clf__groups=groups)
    return estimator.fit(X, y)


def _calibration_method(cfg: Config, n_val: int) -> str:
    method = cfg["training"]["calibration"]
    if method == "auto":
        return "isotonic" if n_val >= 300 else "sigmoid"  # isotonic overfits small calibration sets
    return method


def config_hash(cfg: Config) -> str:
    relevant = {k: cfg[k] for k in ("audio", "nlp", "embeddings", "split", "training", "project")}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()[:12]


def run_experiment(
    spec: ExperimentSpec, parts: dict[str, pd.DataFrame], cfg: Config, tracker: Tracker, registry: ModelRegistry,
    dataset_version: str,
) -> dict[str, Any]:
    seed = cfg["project"]["seed"]
    set_seed(seed)
    train, val, test = parts["train"], parts["val"], parts["test"]
    result: dict[str, Any] = {
        "experiment_id": spec.experiment_id, **asdict(spec), "dataset_version": dataset_version,
        "config_hash": config_hash(cfg), "seed": seed, "trained_at": datetime.now(UTC).isoformat(),
    }
    cols_by_mod = columns_for(spec.modalities, train)
    missing = [m for m in spec.modalities if m not in cols_by_mod]
    if missing:
        result.update(status="skipped", reason=f"modalities unavailable in training data: {missing}")
        log.info("experiment skipped", extra={"ctx": {"experiment": spec.experiment_id, "missing": missing}})
        return result
    cols = [c for cs in cols_by_mod.values() for c in cs]
    X_tr, X_va, X_te = train[cols], val[cols], test[cols]
    y_tr, y_va, y_te = encode_labels(train, cfg), encode_labels(val, cfg), encode_labels(test, cfg)
    g_tr, g_te = train["participant_id"].to_numpy(), test["participant_id"].to_numpy()
    t0 = time.perf_counter()

    oof = np.full(len(y_tr), np.nan)
    for tr_idx, va_idx in group_kfold_indices(g_tr, y_tr, cfg["training"]["cv_folds"], seed):
        est, _ = _make_estimator(spec, cols_by_mod, y_tr[tr_idx], cfg)
        _fit(est, spec, X_tr.iloc[tr_idx], y_tr[tr_idx], g_tr[tr_idx])
        oof[va_idx] = est.predict_proba(X_tr.iloc[va_idx])[:, 1]
    cv_metrics = binary_metrics(y_tr, oof)

    estimator, implementation = _make_estimator(spec, cols_by_mod, y_tr, cfg)
    _fit(estimator, spec, X_tr, y_tr, g_tr)
    calibrated = CalibratedModel(estimator, _calibration_method(cfg, len(y_va))).fit(X_va, y_va)
    p_va = calibrated.predict_proba(X_va)[:, 1]
    p_te = calibrated.predict_proba(X_te)[:, 1]
    p_te_uncal = estimator.predict_proba(X_te)[:, 1]
    ecfg = cfg["evaluation"]
    yp, pp, _ = participant_level(y_te, p_te, g_te)
    test_metrics = {
        "recording": binary_metrics(y_te, p_te, n_bins=ecfg["calibration_bins"]),
        "recording_uncalibrated": binary_metrics(y_te, p_te_uncal, n_bins=ecfg["calibration_bins"]),
        "participant": binary_metrics(yp, pp, n_bins=ecfg["calibration_bins"]),
        "ci": cluster_bootstrap_ci(y_te, p_te, g_te, ecfg["n_bootstrap"], ecfg["ci_level"], seed),
        "curves": curves(y_te, p_te, ecfg["calibration_bins"]),
    }
    ood = MahalanobisOOD(ecfg["ood_quantile"]).fit(X_tr[[c for c in cols if not c.startswith("emb_")] or cols])
    result.update(
        status="ok", implementation=implementation, n_features=len(cols), feature_columns=cols,
        modality_columns=cols_by_mod, calibration_method=calibrated.method,
        n_train=int(len(y_tr)), n_val=int(len(y_va)), n_test=int(len(y_te)),
        n_participants={"train": int(len(set(g_tr))), "val": int(val["participant_id"].nunique()), "test": int(len(set(g_te)))},
        cv=cv_metrics, val=binary_metrics(y_va, p_va), test=test_metrics, duration_s=round(time.perf_counter() - t0, 2),
    )

    bundle = ModelBundle(
        experiment_id=spec.experiment_id, model=calibrated, columns=cols, modality_columns=cols_by_mod, spec=asdict(spec),
        ood=ood, background=X_tr.sample(min(100, len(X_tr)), random_state=seed), positive_label=cfg["split"]["positive_label"],
        metadata={k: v for k, v in result.items() if k not in ("test",)},
    )
    artifact_dir = registry.save_bundle(bundle)
    result["artifact_path"] = artifact_dir.as_posix()

    with tracker.run(spec.experiment_id, tags={"modality_set": spec.modality_set, "strategy": spec.strategy,
                                               "dataset_version": dataset_version, "synthetic": str(bool(train["is_synthetic"].all()))}) as run_id:
        tracker.log_params({
            "model": spec.model, "strategy": spec.strategy, "modalities": ",".join(spec.modalities), "n_features": len(cols),
            "seed": seed, "dataset_version": dataset_version, "config_hash": result["config_hash"],
            "calibration": calibrated.method, "implementation": implementation, "cv_folds": cfg["training"]["cv_folds"],
        })
        tracker.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items() if not isinstance(v, dict)})
        tracker.log_metrics({f"test_{k}": v for k, v in test_metrics["recording"].items() if not isinstance(v, dict)})
        tracker.log_metrics({f"test_participant_{k}": v for k, v in test_metrics["participant"].items() if not isinstance(v, dict)})
        tracker.log_artifact(artifact_dir / "model.joblib", "model")
        tracker.log_artifact(artifact_dir / "metadata.json", "model")
        result["mlflow_run_id"] = run_id
    log.info("experiment complete", extra={"ctx": {"experiment": spec.experiment_id, "cv_auroc": round(cv_metrics["auroc"], 3),
                                                   "test_auroc": round(test_metrics["recording"]["auroc"], 3)}})
    return result


def select_serving(results: list[dict[str, Any]], metric: str = "auroc") -> dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok" and np.isfinite(r["cv"].get(metric, np.nan))]
    if not ok:
        raise RuntimeError("No experiment completed successfully; cannot select a serving model.")
    best = max(r["cv"][metric] for r in ok)
    eligible = [r for r in ok if r["cv"][metric] >= best - SIMPLICITY_MARGIN]
    chosen = min(eligible, key=lambda r: ExperimentSpec(r["modality_set"], r["modalities"], r["strategy"], r["model"]).complexity)
    ensemble = [r["experiment_id"] for r in ok if r["modality_set"] == chosen["modality_set"] and r["strategy"] in ("single", "early")]
    return {
        "experiment_id": chosen["experiment_id"], "selection_metric": f"cv_{metric}", "best_cv_score": best,
        "chosen_cv_score": chosen["cv"][metric], "simplicity_margin": SIMPLICITY_MARGIN, "ensemble": ensemble,
    }


def train_all(df: pd.DataFrame, cfg: Config, only: list[str] | None = None) -> dict[str, Any]:
    """Run all configured experiments and write the registry index. Returns the index."""
    parts = modelling_partitions(df, cfg)
    for name, part in parts.items():
        if part.empty or part[cfg["split"]["label_column"]].nunique() < 2:
            raise RuntimeError(f"Partition '{name}' lacks both classes ({len(part)} rows); cannot train/evaluate.")
    dataset_version = dataframe_fingerprint(df[["recording_id", "split", cfg["split"]["label_column"]]])
    registry = ModelRegistry(cfg.path("models"))
    tracker = Tracker(cfg.path("mlruns"), "cognitrack-ablation")
    specs = [s for s in experiment_specs(cfg) if not only or s.experiment_id in only or s.modality_set in only]
    results = [run_experiment(s, parts, cfg, tracker, registry, dataset_version) for s in specs]
    serving = select_serving(results, cfg["training"]["selection_metric"])
    index = {
        "created_at": datetime.now(UTC).isoformat(), "dataset_version": dataset_version,
        "synthetic_data": bool(df["is_synthetic"].all()), "serving": serving,
        "embedding_encoder": (df["embedding_encoder"].dropna().iloc[0]
                              if "embedding_encoder" in df and df["embedding_encoder"].notna().any() else None),
        "experiments": {r["experiment_id"]: r for r in results},
    }
    if tracker.enabled and (run_id := index["experiments"][serving["experiment_id"]].get("mlflow_run_id")):
        serving["mlflow_model_version"] = tracker.register(run_id, "model", "cognitrack-serving")
    registry.write_index(index)
    return index
