"""Publish pipeline outputs (manifest, features, models, predictions, reports) into the database.

Idempotent: entities are upserted by their natural keys, batch predictions for a model version are
replaced, and every publish appends a new version of each report.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.app.database.session import get_sessionmaker, init_db
from backend.app.models import (
    AssessmentSession,
    Experiment,
    FeatureSet,
    ModelVersion,
    MonitoringReport,
    Participant,
    Prediction,
    ProcessingJob,
    Recording,
)
from backend.app.services.common import feature_version, utcnow
from ml.config import get_config
from ml.features.registry import (
    ACOUSTIC_FEATURES,
    CONVERSATIONAL_FEATURES,
    DESCRIPTIVE_FEATURES,
    LINGUISTIC_FEATURES,
)
from ml.training.registry import ModelRegistry
from ml.utils import get_logger, read_json, sanitize_nan

log = get_logger(__name__)

REPORT_FILES = {
    "data_quality": "data_quality.json", "lineage": "lineage.json", "splits": "splits.json",
    "feature_extraction": "feature_extraction.json", "model_comparison": "model_comparison.json",
    "explainability": "explainability.json", "fairness": "fairness.json", "uncertainty": "uncertainty.json",
    "robustness": "robustness.json", "longitudinal": "longitudinal.json", "drift": "drift.json", "model_card": "model_card.json",
}
FEATURE_COLUMNS = ACOUSTIC_FEATURES + LINGUISTIC_FEATURES + CONVERSATIONAL_FEATURES + DESCRIPTIVE_FEATURES


def _val(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.generic):
        v = v.item()
        return None if isinstance(v, float) and np.isnan(v) else v
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.to_pydatetime()
    return v


def _upsert_cohort(db: Session, df: pd.DataFrame) -> dict[str, int]:
    participants = {p.pseudonym: p for p in db.scalars(select(Participant))}
    sessions = {s.session_key: s for s in db.scalars(select(AssessmentSession))}
    recordings = {r.recording_key: r for r in db.scalars(select(Recording))}
    fv = feature_version(get_config())
    value_columns = [c for c in FEATURE_COLUMNS if c in df.columns] + [c for c in df.columns if c.startswith("emb_")]
    for pid, part in df.groupby("participant_id"):
        first = part.iloc[0]
        p = participants.get(pid) or Participant(pseudonym=pid)
        p.dataset_source = first["dataset_source"]
        p.cohort_label = _val(part["assessment_label"].dropna().iloc[0]) if part["assessment_label"].notna().any() else None
        p.age_band, p.sex, p.education_band = _val(first["age_band"]), _val(first["sex"]), _val(first["education_band"])
        p.is_synthetic = bool(first["is_synthetic"])
        p.consent_research = bool(part["consent_research"].all())
        if p.id is None:
            db.add(p)
            participants[pid] = p
    db.flush()
    for _, row in df.iterrows():
        p = participants[row["participant_id"]]
        s = sessions.get(row["session_id"]) or AssessmentSession(session_key=row["session_id"], participant_id=p.id)
        s.session_index = int(row["session_index"])
        ts = _val(row["session_timestamp"])
        s.session_date = ts.date() if ts is not None else None
        s.task, s.assessment_label = _val(row["task"]), _val(row["assessment_label"])
        s.assessment_score, s.assessment_instrument = _val(row["assessment_score"]), _val(row["assessment_instrument"])
        if s.id is None:
            db.add(s)
            db.flush()
            sessions[row["session_id"]] = s
        r = recordings.get(row["recording_id"]) or Recording(recording_key=row["recording_id"], session_id=s.id, participant_id=p.id)
        r.dataset_source, r.source = row["dataset_source"], "dataset"
        r.storage_path = row["recording_path"]
        r.raw_audio_retained = True  # dataset audio stays in its licensed local location; nothing is copied
        r.waveform = None  # lazily recomputed from audio, so VAD/pause changes take effect after a publish
        r.audio_sha256, r.duration_s, r.sample_rate = _val(row["audio_sha256"]), _val(row["recording_duration"]), _val(row["sample_rate"])
        r.transcript, r.transcript_source = _val(row["transcript"]), row["transcript_source"]
        r.turns_path = _val(row["turns_path"])
        r.has_speaker_turns = bool(r.turns_path)
        r.quality_usable = bool(row["quality_usable"]) if pd.notna(row.get("quality_usable")) else None
        r.quality = sanitize_nan({
            "flags": [f for f in str(row.get("quality_flags") or "").split(",") if f],
            "snr_db": _val(row.get("quality_snr_db")), "clipping_fraction": _val(row.get("quality_clipping_fraction")),
            "silence_fraction": _val(row.get("quality_silence_fraction")), "extraction_error": _val(row.get("extraction_error")),
        })
        r.split = row.get("split")
        if r.id is None:
            db.add(r)
            db.flush()
            recordings[row["recording_id"]] = r
        values = {c: _val(row[c]) for c in value_columns}
        existing = db.scalar(select(FeatureSet).where(FeatureSet.recording_id == r.id, FeatureSet.feature_version == fv))
        fs = existing or FeatureSet(recording_id=r.id, feature_version=fv)
        fs.values = sanitize_nan(values)
        fs.backends = json.loads(row["backends"]) if isinstance(row.get("backends"), str) else None
        if existing is None:
            db.add(fs)
    db.flush()
    return {"participants": len(participants), "sessions": len(sessions), "recordings": len(recordings)}


def _upsert_models(db: Session, index: dict[str, Any]) -> dict[str, ModelVersion]:
    versions = {m.experiment_key: m for m in db.scalars(select(ModelVersion))}
    serving_id = index["serving"]["experiment_id"]
    # Set the flag through the ORM (a bulk UPDATE would leave loaded objects stale, and re-setting
    # is_serving=True on them would then emit no UPDATE).
    for key, mv in versions.items():
        mv.is_serving = key == serving_id
    for key, r in index["experiments"].items():
        mv = None
        if r["status"] == "ok":
            mv = versions.get(key) or ModelVersion(experiment_key=key)
            mv.model_type, mv.strategy, mv.modalities = r["model"], r["strategy"], r["modalities"]
            mv.dataset_version, mv.calibration = r["dataset_version"], r.get("calibration_method")
            mv.artifact_path, mv.mlflow_run_id = r["artifact_path"], r.get("mlflow_run_id")
            mv.is_serving = key == serving_id
            mv.trained_on_synthetic = bool(index.get("synthetic_data"))
            mv.summary_metrics = sanitize_nan({"cv_auroc": r["cv"]["auroc"], "test_auroc": r["test"]["recording"]["auroc"],
                                               "test_brier": r["test"]["recording"]["brier"], "test_ece": r["test"]["recording"]["ece"]})
            if mv.id is None:
                db.add(mv)
                db.flush()
                versions[key] = mv
        exp = db.scalar(select(Experiment).where(Experiment.experiment_key == key, Experiment.dataset_version == r["dataset_version"]))
        exp = exp or Experiment(experiment_key=key, dataset_version=r["dataset_version"])
        exp.model_version_id = mv.id if mv else None
        exp.status, exp.modality_set, exp.strategy, exp.model = r["status"], r["modality_set"], r["strategy"], r["model"]
        exp.seed, exp.config_hash, exp.reason = r.get("seed"), r.get("config_hash"), r.get("reason")
        exp.params = sanitize_nan({k: r.get(k) for k in ("modalities", "implementation", "n_features", "calibration_method",
                                                         "n_train", "n_val", "n_test", "n_participants")})
        exp.cv_metrics = sanitize_nan(r.get("cv"))
        exp.test_metrics = sanitize_nan({k: v for k, v in r.get("test", {}).items() if k != "curves"}) if r.get("test") else None
        exp.duration_s, exp.mlflow_run_id = r.get("duration_s"), r.get("mlflow_run_id")
        if exp.id is None:
            db.add(exp)
    db.flush()
    return versions


def _publish_predictions(db: Session, preds: pd.DataFrame, versions: dict[str, ModelVersion]) -> int:
    rec_ids = dict(db.execute(select(Recording.recording_key, Recording.id)).all())
    n = 0
    for exp_id, group in preds.groupby("model_experiment_id"):
        mv = versions[exp_id]
        db.execute(delete(Prediction).where(Prediction.model_version_id == mv.id, Prediction.origin == "batch"))
        for row in group.itertuples():
            if row.recording_id not in rec_ids:
                continue
            db.add(Prediction(
                recording_id=rec_ids[row.recording_id], model_version_id=mv.id, probability=float(row.probability),
                model_label="elevated_signal" if row.probability >= 0.5 else "no_elevated_signal", abstain=bool(row.abstain),
                uncertainty=sanitize_nan({"ensemble_std": _val(row.ensemble_std), "ood_distance": _val(row.ood_distance),
                                          "reasons": [x for x in str(row.abstain_reasons).split("; ") if x], "abstain": bool(row.abstain)}),
                origin="batch", split=row.split,
            ))
            n += 1
    db.flush()
    return n


def publish_all() -> dict[str, Any]:
    cfg = get_config()
    init_db()
    job = ProcessingJob(job_type="pipeline_publish", status="running", started_at=utcnow())
    SessionLocal = get_sessionmaker()
    with SessionLocal() as db:
        db.add(job)
        db.commit()
        try:
            processed = cfg.path("processed")
            df = pd.read_parquet(processed / "dataset.parquet")
            counts = _upsert_cohort(db, df)
            index = ModelRegistry(cfg.path("models")).index()
            versions = _upsert_models(db, index)
            preds_path = processed / "predictions.parquet"
            counts["predictions"] = _publish_predictions(db, pd.read_parquet(preds_path), versions) if preds_path.exists() else 0
            serving = versions.get(index["serving"]["experiment_id"])
            published = []
            for rtype, fname in REPORT_FILES.items():
                path = cfg.path("report_data") / fname
                if path.exists():
                    db.add(MonitoringReport(report_type=rtype, payload=read_json(path), dataset_version=index["dataset_version"],
                                            model_version_id=serving.id if serving else None, is_synthetic=bool(index.get("synthetic_data"))))
                    published.append(rtype)
            counts["reports"] = published
            job.status, job.finished_at, job.detail = "succeeded", utcnow(), sanitize_nan(counts)
            db.commit()
        except Exception as exc:
            db.rollback()
            job = db.merge(job)
            job.status, job.finished_at, job.error = "failed", utcnow(), f"{type(exc).__name__}: {exc}"
            db.commit()
            raise
    log.info("publish complete", extra={"ctx": {k: v for k, v in counts.items() if k != "reports"}})
    return counts
