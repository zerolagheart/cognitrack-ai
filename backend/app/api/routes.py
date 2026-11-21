"""HTTP API (``/api/v1``). Role requirements are declared per endpoint."""

from __future__ import annotations

import contextlib
from datetime import date
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select, text
from sqlalchemy.orm import Session, selectinload

from backend.app.core.config import Settings, get_settings
from backend.app.core.errors import NotFoundError, UnavailableError
from backend.app.core.security import Principal, require
from backend.app.database.session import get_db, get_sessionmaker
from backend.app.models import Experiment, ModelVersion, Prediction, ProcessingJob, Recording
from backend.app.schemas.api import (
    ExperimentOut,
    FeaturePredictRequest,
    HealthOut,
    JobOut,
    ModelVersionOut,
    ParticipantList,
    PredictionOut,
    RecordingOut,
    ReportOut,
    UploadOut,
)
from backend.app.services import analytics
from backend.app.services.common import audit, get_participant, get_recording, latest_features
from backend.app.services.predictor import get_predictor, reset_predictor, try_get_predictor
from backend.app.services.recordings import (
    create_upload,
    predict_existing,
    process_recording,
    save_upload,
    waveform_for,
)
from ml.config import get_config, resolve_path
from ml.data.adapters import ADAPTERS, get_adapter
from ml.features.registry import feature_modality
from ml.inference import DISCLAIMER

router = APIRouter(prefix="/api/v1")
viewer, researcher, admin = require("viewer"), require("researcher"), require("admin")
REPORT_TYPES = {"data_quality", "lineage", "splits", "feature_extraction", "model_comparison", "explainability", "fairness",
                "uncertainty", "robustness", "longitudinal", "drift", "model_card"}


def _prediction_out(p: Prediction) -> PredictionOut:
    return PredictionOut(id=p.id, recording_key=p.recording.recording_key, model_experiment_id=p.model_version.experiment_key,
                         probability=p.probability, model_label=p.model_label, abstain=p.abstain, uncertainty=p.uncertainty,
                         explanation=p.explanation, origin=p.origin, split=p.split, created_at=p.created_at, disclaimer=DISCLAIMER)


# ---------------------------------------------------------------- system
@router.get("/health", response_model=HealthOut, tags=["system"])
def health(db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> HealthOut:
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    predictor = try_get_predictor()
    return HealthOut(status="ok" if db_ok else "degraded", version="0.1.0", environment=settings.environment, database=db_ok,
                     model_loaded=predictor is not None, serving_model=predictor.bundle.experiment_id if predictor else None,
                     auth_enabled=settings.auth_enabled, disclaimer=DISCLAIMER)


@router.get("/overview", tags=["analytics"])
def overview(db: Session = Depends(get_db), _: Principal = Depends(viewer)) -> dict[str, Any]:
    return analytics.overview(db)


# ---------------------------------------------------------------- datasets
@router.get("/datasets", tags=["datasets"])
def datasets(db: Session = Depends(get_db), _: Principal = Depends(viewer)) -> dict[str, Any]:
    cfg = get_config()
    adapters = []
    for key in cfg["datasets"]:
        try:
            adapters.append({"key": key, **get_adapter(key, cfg["datasets"]).describe(), "active": key in cfg["project"]["datasets"]})
        except KeyError as exc:
            adapters.append({"key": key, "error": str(exc)})
    quality = analytics.latest_report(db, "data_quality", required=False)
    return {"adapters": adapters, "registered_adapter_types": sorted(ADAPTERS),
            "profile": quality.payload.get("profile") if quality else None,
            "dataset_version": quality.dataset_version if quality else None}


# ---------------------------------------------------------------- participants
@router.get("/participants", response_model=ParticipantList, tags=["participants"])
def participants(search: str | None = Query(None, max_length=64), limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                 db: Session = Depends(get_db), principal: Principal = Depends(researcher)) -> dict[str, Any]:
    audit(db, principal.role, "list", "participants")
    return analytics.list_participants(db, search, limit, offset)


@router.get("/participants/{pseudonym}", tags=["participants"])
def participant(pseudonym: str, db: Session = Depends(get_db), principal: Principal = Depends(researcher)) -> dict[str, Any]:
    p = get_participant(db, pseudonym)
    audit(db, principal.role, "read", "participant", pseudonym)
    return analytics.participant_detail(db, p)


@router.get("/participants/{pseudonym}/features", tags=["participants"])
def participant_features(pseudonym: str, modality: str | None = Query(None, pattern="^(acoustic|linguistic|conversational)$"),
                         db: Session = Depends(get_db), _: Principal = Depends(researcher)) -> dict[str, Any]:
    return analytics.feature_trends(db, get_participant(db, pseudonym), modality)


@router.get("/participants/{pseudonym}/longitudinal", tags=["longitudinal"])
def participant_longitudinal(pseudonym: str, db: Session = Depends(get_db), principal: Principal = Depends(researcher)) -> dict[str, Any]:
    p = get_participant(db, pseudonym)
    audit(db, principal.role, "read", "longitudinal", pseudonym)
    return analytics.live_longitudinal(db, p)


@router.get("/participants/{pseudonym}/predictions", response_model=list[PredictionOut], tags=["predictions"])
def participant_predictions(pseudonym: str, db: Session = Depends(get_db), _: Principal = Depends(researcher)) -> list[PredictionOut]:
    p = get_participant(db, pseudonym)
    rows = db.scalars(select(Prediction).join(Recording).where(Recording.participant_id == p.id)
                      .options(selectinload(Prediction.recording), selectinload(Prediction.model_version))
                      .order_by(Prediction.created_at.desc())).all()
    return [_prediction_out(r) for r in rows]


# ---------------------------------------------------------------- recordings
@router.post("/recordings/upload", response_model=UploadOut, status_code=202, tags=["recordings"])
async def upload_recording(
    background: BackgroundTasks, file: UploadFile = File(...), participant: str = Form(..., description="Pseudonymous identifier"),
    consent_confirmed: bool = Form(..., description="Documented research consent exists for this recording"),
    session_date: date | None = Form(None), task: str | None = Form(None, max_length=64),
    transcript: str | None = Form(None, max_length=20000), db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings), principal: Principal = Depends(researcher),
) -> UploadOut:
    path = await save_upload(file, settings)
    rec, job = create_upload(db, path, participant, consent_confirmed, session_date, task, transcript)
    audit(db, principal.role, "upload", "recording", rec.recording_key)
    background.add_task(_process_in_background, rec.id, job.id)
    return UploadOut(recording_key=rec.recording_key, participant=participant, session_key=rec.session.session_key,
                     job=JobOut.model_validate(job),
                     raw_audio_retention="retained" if settings.retain_raw_audio else "deleted after feature extraction")


def _process_in_background(recording_id: int, job_id: str) -> None:
    with get_sessionmaker()() as db:
        rec, job = db.get(Recording, recording_id), db.get(ProcessingJob, job_id)
        with contextlib.suppress(Exception):  # failure is recorded on the job
            process_recording(db, rec, get_settings(), job)


@router.get("/recordings/{key}", response_model=RecordingOut, tags=["recordings"])
def recording(key: str, db: Session = Depends(get_db), principal: Principal = Depends(researcher)) -> RecordingOut:
    r = get_recording(db, key)
    audit(db, principal.role, "read", "recording", key)
    feats = latest_features(r)
    latest = max(r.predictions, key=lambda p: p.created_at) if r.predictions else None
    return RecordingOut(
        recording_key=r.recording_key, participant=r.participant.pseudonym, session_key=r.session.session_key,
        session_index=r.session.session_index, session_date=r.session.session_date, dataset_source=r.dataset_source, source=r.source,
        duration_s=r.duration_s, sample_rate=r.sample_rate, transcript=r.transcript, transcript_source=r.transcript_source,
        has_speaker_turns=r.has_speaker_turns, quality_usable=r.quality_usable, quality=r.quality, split=r.split,
        raw_audio_retained=r.raw_audio_retained, features={k: v for k, v in feats.items() if not k.startswith("emb_")},
        feature_modalities={k: feature_modality(k) for k in feats if not k.startswith("emb_")},
        latest_prediction=_prediction_out(latest) if latest else None,
    )


@router.get("/recordings/{key}/waveform", tags=["recordings"])
def recording_waveform(key: str, db: Session = Depends(get_db), _: Principal = Depends(researcher)) -> dict[str, Any]:
    return waveform_for(db, get_recording(db, key))


@router.post("/recordings/{key}/process", tags=["recordings"])
def reprocess(key: str, db: Session = Depends(get_db), settings: Settings = Depends(get_settings),
              principal: Principal = Depends(researcher)) -> dict[str, Any]:
    r = get_recording(db, key)
    job = ProcessingJob(job_type="feature_extraction", status="queued", recording_id=r.id, detail={"recording_key": key})
    db.add(job)
    db.commit()
    audit(db, principal.role, "process", "recording", key)
    return process_recording(db, r, settings, job)


@router.post("/recordings/{key}/predict", response_model=PredictionOut, tags=["predictions"])
def predict(key: str, db: Session = Depends(get_db), principal: Principal = Depends(researcher)) -> PredictionOut:
    r = get_recording(db, key)
    pred = predict_existing(db, r)
    audit(db, principal.role, "predict", "recording", key)
    db.refresh(pred)
    return _prediction_out(pred)


@router.get("/recordings/{key}/explanation", tags=["explainability"])
def explanation(key: str, db: Session = Depends(get_db), _: Principal = Depends(researcher)) -> dict[str, Any]:
    r = get_recording(db, key)
    with_expl = [p for p in r.predictions if p.explanation]
    if not with_expl:
        pred = predict_existing(db, r)
        db.refresh(pred)
    else:
        pred = max(with_expl, key=lambda p: p.created_at)
    return {"recording_key": key, "probability": pred.probability, "abstain": pred.abstain, "uncertainty": pred.uncertainty,
            "explanation": pred.explanation, "model_experiment_id": pred.model_version.experiment_key, "disclaimer": DISCLAIMER}


@router.post("/predict/features", tags=["predictions"])
def predict_features(body: FeaturePredictRequest, _: Principal = Depends(researcher)) -> dict[str, Any]:
    return get_predictor().predict_features(body.features, body.quality_flags).to_dict()


@router.get("/predictions", response_model=list[PredictionOut], tags=["predictions"])
def predictions(limit: int = Query(50, ge=1, le=500), origin: str | None = Query(None, pattern="^(batch|api)$"),
                db: Session = Depends(get_db), _: Principal = Depends(researcher)) -> list[PredictionOut]:
    q = select(Prediction).options(selectinload(Prediction.recording), selectinload(Prediction.model_version))
    if origin:
        q = q.where(Prediction.origin == origin)
    return [_prediction_out(p) for p in db.scalars(q.order_by(Prediction.created_at.desc(), Prediction.id.desc()).limit(limit)).all()]


# ---------------------------------------------------------------- models & experiments
@router.get("/models", response_model=list[ModelVersionOut], tags=["models"])
def models(db: Session = Depends(get_db), _: Principal = Depends(viewer)) -> list[ModelVersion]:
    return db.scalars(select(ModelVersion).order_by(ModelVersion.is_serving.desc(), ModelVersion.experiment_key)).all()


@router.get("/models/serving", tags=["models"])
def serving_model(_: Principal = Depends(viewer)) -> dict[str, Any]:
    return get_predictor().model_info()


@router.get("/experiments", response_model=list[ExperimentOut], tags=["models"])
def experiments(db: Session = Depends(get_db), _: Principal = Depends(viewer)) -> list[Experiment]:
    return db.scalars(select(Experiment).order_by(Experiment.created_at.desc(), Experiment.experiment_key)).all()


# ---------------------------------------------------------------- reports, monitoring, responsible AI
@router.get("/reports/{report_type}", response_model=ReportOut, tags=["reports"])
def report(report_type: str, db: Session = Depends(get_db), _: Principal = Depends(viewer)) -> dict[str, Any]:
    if report_type not in REPORT_TYPES:
        raise NotFoundError(f"Unknown report type '{report_type}'. Available: {sorted(REPORT_TYPES)}")
    return analytics.report_payload(db, report_type)


@router.get("/longitudinal/cohort", tags=["longitudinal"])
def longitudinal_cohort(db: Session = Depends(get_db), _: Principal = Depends(viewer)) -> dict[str, Any]:
    payload = analytics.report_payload(db, "longitudinal")["payload"]
    rows = [{"participant_id": p["participant_id"], "status": p["status"], "n_sessions": p["n_sessions"],
             "summary": p.get("summary"), "missing": p["missing"]} for p in payload["participants"]]
    return {"cohort_summary": payload["cohort_summary"], "participants": rows, "interpretation": payload["interpretation"],
            "synthetic_engine_validation": payload.get("synthetic_engine_validation"), "tracked_features": payload["tracked_features"]}


@router.get("/monitoring/jobs", response_model=list[JobOut], tags=["monitoring"])
def jobs(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db), _: Principal = Depends(viewer)) -> list[ProcessingJob]:
    return db.scalars(select(ProcessingJob).order_by(ProcessingJob.created_at.desc()).limit(limit)).all()


@router.get("/monitoring/jobs/{job_id}", response_model=JobOut, tags=["monitoring"])
def job(job_id: str, db: Session = Depends(get_db), _: Principal = Depends(viewer)) -> ProcessingJob:
    j = db.get(ProcessingJob, job_id)
    if j is None:
        raise NotFoundError(f"Job '{job_id}' not found")
    return j


@router.get("/monitoring/evidently", tags=["monitoring"], response_class=FileResponse)
def evidently_html(db: Session = Depends(get_db), _: Principal = Depends(viewer)) -> FileResponse:
    drift = analytics.latest_report(db, "drift")
    html = (drift.payload.get("evidently") or {}).get("html")
    if not html or not resolve_path(html).exists():
        raise UnavailableError("Evidently HTML report not available (Evidently not installed or report not generated).")
    return FileResponse(resolve_path(html), media_type="text/html")


@router.post("/admin/publish", tags=["admin"])
def admin_publish(principal: Principal = Depends(admin), db: Session = Depends(get_db)) -> dict[str, Any]:
    from backend.app.services.sync import publish_all

    result = publish_all()
    reset_predictor()
    audit(db, principal.role, "publish", "pipeline")
    return result
