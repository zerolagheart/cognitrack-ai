"""Recording upload, processing, inference and waveform services."""

from __future__ import annotations

import re
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.core.config import Settings
from backend.app.core.errors import PayloadTooLarge, UnavailableError, ValidationFailed
from backend.app.models import (
    AssessmentSession,
    FeatureSet,
    ModelVersion,
    Participant,
    Prediction,
    ProcessingJob,
    Recording,
)
from backend.app.services.common import feature_version, latest_features, utcnow
from backend.app.services.predictor import try_get_predictor
from ml.audio.vad import detect_speech
from ml.config import get_config, resolve_path
from ml.features.extraction import extract_from_signal, sentence_model_for
from ml.nlp.conversation import load_turns
from ml.preprocessing.audio import AudioLoadError, load_audio
from ml.utils import file_sha256, get_logger, sanitize_nan

log = get_logger(__name__)
PSEUDONYM_RE = re.compile(r"^[A-Za-z0-9_\-]{3,64}$")
CHUNK = 1 << 20


async def save_upload(file: UploadFile, settings: Settings) -> Path:
    """Stream an upload to a server-generated filename, enforcing extension and size limits."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.allowed_extensions:
        raise ValidationFailed(f"Unsupported audio format '{ext or 'none'}'. Allowed: {sorted(settings.allowed_extensions)}")
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    upload_dir = resolve_path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{uuid.uuid4().hex}{ext}"  # never trust client-supplied names/paths
    written = 0
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(CHUNK):
                written += len(chunk)
                if written > max_bytes:
                    raise PayloadTooLarge(f"File exceeds the {settings.max_upload_mb:g} MB limit")
                out.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    if written == 0:
        dest.unlink(missing_ok=True)
        raise ValidationFailed("Uploaded file is empty")
    return dest


def waveform_summary(signal: np.ndarray, sr: int, bins: int = 1200) -> dict[str, Any]:
    """Min/max envelope + speech segments + pauses. Cannot be inverted back to audio."""
    cfg = get_config()["audio"]
    n = len(signal)
    edges = np.linspace(0, n, min(bins, max(n, 1)) + 1).astype(int)
    mins = [float(signal[a:b].min()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:], strict=True)]
    maxs = [float(signal[a:b].max()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:], strict=True)]
    activity = detect_speech(signal, sr, cfg["frame_ms"], cfg["hop_ms"], cfg["vad_relative_threshold_db"], cfg["min_pause_s"])
    pauses = [{"start": round(a[1], 3), "end": round(b[0], 3), "duration": round(b[0] - a[1], 3)}
              for a, b in zip(activity.segments, activity.segments[1:], strict=False) if b[0] - a[1] >= cfg["min_pause_s"]]
    return {"duration_s": n / sr, "time_step_s": n / sr / max(len(mins), 1), "min": [round(v, 4) for v in mins],
            "max": [round(v, 4) for v in maxs], "speech_segments": [[round(s, 3), round(e, 3)] for s, e in activity.segments],
            "pauses": pauses, "long_pause_threshold_s": cfg["long_pause_s"]}


def create_upload(
    db: Session, path: Path, pseudonym: str, consent_confirmed: bool, session_date: date | None, task: str | None,
    transcript: str | None,
) -> tuple[Recording, ProcessingJob]:
    if not consent_confirmed:
        path.unlink(missing_ok=True)
        raise ValidationFailed("Upload rejected: documented research consent must be confirmed for this recording.")
    if not PSEUDONYM_RE.match(pseudonym):
        path.unlink(missing_ok=True)
        raise ValidationFailed("Participant identifier must be a pseudonym of 3-64 letters, digits, '-' or '_' (no names or emails).")
    try:
        _, sr, native_sr = load_audio(path, get_config()["audio"]["target_sample_rate"])
    except AudioLoadError as exc:
        path.unlink(missing_ok=True)
        raise ValidationFailed(f"File is not decodable audio: {exc}") from exc
    participant = db.scalar(select(Participant).where(Participant.pseudonym == pseudonym))
    if participant is None:
        participant = Participant(pseudonym=pseudonym, dataset_source="upload", consent_research=True, consent_recorded_at=utcnow())
        db.add(participant)
        db.flush()
    elif not participant.consent_research:
        path.unlink(missing_ok=True)
        raise ValidationFailed("Participant has no recorded research consent.")
    next_index = (db.scalar(select(func.max(AssessmentSession.session_index)).where(AssessmentSession.participant_id == participant.id)) or -1) + 1
    key = f"UPL-{uuid.uuid4().hex[:10]}"
    session = AssessmentSession(session_key=f"{pseudonym}-UPL-{next_index:02d}-{key[-4:]}", participant_id=participant.id,
                                session_index=next_index, session_date=session_date or date.today(), task=task)
    db.add(session)
    db.flush()
    rec = Recording(recording_key=key, session_id=session.id, participant_id=participant.id, dataset_source="upload", source="upload",
                    storage_path=str(path), raw_audio_retained=True, audio_sha256=file_sha256(path), sample_rate=float(native_sr),
                    transcript=transcript or None, transcript_source="dataset" if transcript else "none", split="uploaded")
    db.add(rec)
    db.flush()
    job = ProcessingJob(job_type="upload_processing", status="queued", recording_id=rec.id, detail={"recording_key": key})
    db.add(job)
    db.commit()
    return rec, job


def _store_prediction(db: Session, rec: Recording, features: dict[str, Any], flags: list[str]) -> Prediction | None:
    predictor = try_get_predictor()
    if predictor is None:
        return None
    mv = db.scalar(select(ModelVersion).where(ModelVersion.experiment_key == predictor.bundle.experiment_id))
    if mv is None:
        raise UnavailableError("Serving model is not published in the database. Run `cognitrack publish`.")
    pred = predictor.predict_features(features, flags)
    row = Prediction(recording_id=rec.id, model_version_id=mv.id, probability=pred.probability, model_label=pred.model_label,
                     abstain=pred.uncertainty["abstain"], uncertainty=sanitize_nan(pred.uncertainty),
                     explanation=sanitize_nan(pred.explanation), origin="api", split=rec.split)
    db.add(row)
    return row


def process_recording(db: Session, rec: Recording, settings: Settings, job: ProcessingJob | None = None) -> dict[str, Any]:
    """Quality → features → embeddings → prediction; deletes raw upload audio unless retention is enabled."""
    cfg = get_config()
    if job:
        job.status, job.started_at = "running", utcnow()
        db.commit()
    try:
        if not rec.storage_path or not resolve_path(rec.storage_path).exists():
            raise UnavailableError("Raw audio is not available for this recording (deleted after processing or not on disk).")
        path = resolve_path(rec.storage_path)
        signal, sr, native_sr = load_audio(path, cfg["audio"]["target_sample_rate"])
        turns = load_turns(rec.turns_path)
        rf = extract_from_signal(rec.recording_key, signal, sr, native_sr, cfg, transcript=rec.transcript,
                                 transcript_source=rec.transcript_source, turns=turns, sentence_model=sentence_model_for(cfg))
        predictor = try_get_predictor()
        features = dict(rf.features)
        if predictor is not None and rf.error is None:
            features.update(predictor.embed(signal, sr))
        rec.duration_s, rec.quality_usable = len(signal) / sr, bool(rf.quality.get("usable"))
        rec.quality = sanitize_nan({"flags": rf.quality.get("flags", []), "snr_db": rf.quality.get("estimated_snr_db"),
                                    "clipping_fraction": rf.quality.get("clipping_fraction"),
                                    "silence_fraction": rf.quality.get("silence_fraction"), "extraction_error": rf.error})
        rec.waveform = waveform_summary(signal, sr)
        if rf.transcript and rf.transcript_source == "whisper":
            rec.transcript, rec.transcript_source = rf.transcript, "whisper"
        fv = feature_version(cfg)
        fs = db.scalar(select(FeatureSet).where(FeatureSet.recording_id == rec.id, FeatureSet.feature_version == fv))
        if fs is None:
            fs = FeatureSet(recording_id=rec.id, feature_version=fv)
            db.add(fs)
        fs.values, fs.backends = sanitize_nan(features), rf.backends
        pred = _store_prediction(db, rec, features, rf.quality.get("flags", [])) if rf.error is None else None
        if rec.source == "upload" and not settings.retain_raw_audio:
            path.unlink(missing_ok=True)
            rec.storage_path, rec.raw_audio_retained = None, False
        result = {"recording_key": rec.recording_key, "usable": rec.quality_usable, "error": rf.error,
                  "prediction_id": pred.id if pred else None, "model_available": predictor is not None}
        if job:
            job.status, job.finished_at, job.detail = "succeeded", utcnow(), sanitize_nan({**(job.detail or {}), **result})
        db.commit()
        if pred is not None:
            result["prediction_id"] = pred.id
        return result
    except Exception as exc:
        db.rollback()
        if job:
            job = db.merge(job)
            job.status, job.finished_at, job.error = "failed", utcnow(), f"{type(exc).__name__}: {exc}"
            db.commit()
        if rec.source == "upload" and rec.storage_path and not settings.retain_raw_audio:
            resolve_path(rec.storage_path).unlink(missing_ok=True)
        raise


def predict_existing(db: Session, rec: Recording) -> Prediction:
    features = latest_features(rec)
    if not features:
        raise UnavailableError("No features stored for this recording; process it first.")
    pred = _store_prediction(db, rec, features, (rec.quality or {}).get("flags", []))
    if pred is None:
        raise UnavailableError("No trained model is registered. Run: cognitrack run-all (or train + publish).")
    db.commit()
    return pred


def waveform_for(db: Session, rec: Recording) -> dict[str, Any]:
    if rec.waveform:
        return rec.waveform
    if not rec.storage_path or not resolve_path(rec.storage_path).exists():
        raise UnavailableError("Waveform unavailable: raw audio not retained for this recording.")
    signal, sr, _ = load_audio(resolve_path(rec.storage_path), get_config()["audio"]["target_sample_rate"])
    rec.waveform = waveform_summary(signal, sr)
    db.commit()
    return rec.waveform
