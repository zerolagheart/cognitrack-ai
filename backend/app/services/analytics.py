"""Read-side services: overview, participants, feature trends and live longitudinal analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from backend.app.models import (
    AssessmentSession,
    ModelVersion,
    MonitoringReport,
    Participant,
    Prediction,
    ProcessingJob,
    Recording,
)
from backend.app.services.common import latest_features, latest_report
from ml.config import get_config
from ml.features.registry import feature_modality, label_for
from ml.longitudinal.engine import analyse_participant, population_within_sd, session_table


def overview(db: Session) -> dict[str, Any]:
    n_part = db.scalar(select(func.count(Participant.id))) or 0
    n_rec = db.scalar(select(func.count(Recording.id))) or 0
    n_sess = db.scalar(select(func.count(AssessmentSession.id))) or 0
    analysed = db.scalar(select(func.count(func.distinct(Recording.session_id))).where(Recording.quality_usable.is_(True))) or 0
    n_synth = db.scalar(select(func.count(Participant.id)).where(Participant.is_synthetic.is_(True))) or 0
    n_pred = db.scalar(select(func.count(Prediction.id))) or 0
    unusable = db.scalar(select(func.count(Recording.id)).where(Recording.quality_usable.is_(False))) or 0
    serving = db.scalar(select(ModelVersion).where(ModelVersion.is_serving.is_(True)))
    quality = latest_report(db, "data_quality", required=False)
    comparison = latest_report(db, "model_comparison", required=False)
    jobs = db.scalars(select(ProcessingJob).order_by(ProcessingJob.created_at.desc()).limit(8)).all()
    flags: dict[str, int] = {}
    for q in db.scalars(select(Recording.quality)).all():
        for f in (q or {}).get("flags", []):
            flags[f] = flags.get(f, 0) + 1
    return {
        "counts": {"participants": n_part, "recordings": n_rec, "sessions": n_sess, "analysed_sessions": analysed,
                   "predictions": n_pred, "unusable_recordings": unusable, "synthetic_participants": n_synth},
        "synthetic_data": n_part > 0 and n_synth == n_part,
        "data_quality": {
            "quality_flag_counts": flags,
            "validation_passed": quality.payload.get("passed") if quality else None,
            "blocking_issues": quality.payload.get("blocking_issues") if quality else None,
            "missing_transcripts": (quality.payload["missing_values"]["transcript"]["missing"] if quality else None),
            "class_distribution": quality.payload.get("class_distribution") if quality else None,
        },
        "serving_model": None if serving is None else {
            "experiment_id": serving.experiment_key, "model": serving.model_type, "strategy": serving.strategy,
            "modalities": serving.modalities, "metrics": serving.summary_metrics, "trained_on_synthetic": serving.trained_on_synthetic,
        },
        "ablation": comparison.payload.get("ablation") if comparison else None,
        "banner": comparison.payload.get("banner") if comparison else None,
        "recent_jobs": [{"id": j.id, "type": j.job_type, "status": j.status, "created_at": j.created_at,
                         "finished_at": j.finished_at, "error": j.error} for j in jobs],
    }


def list_participants(db: Session, search: str | None, limit: int, offset: int) -> dict[str, Any]:
    q = select(Participant)
    if search:
        q = q.where(Participant.pseudonym.ilike(f"%{search}%"))
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = db.scalars(q.order_by(Participant.pseudonym).limit(limit).offset(offset).options(selectinload(Participant.sessions))).all()
    return {"total": total, "items": [{
        "pseudonym": p.pseudonym, "dataset_source": p.dataset_source, "cohort_label": p.cohort_label, "is_synthetic": p.is_synthetic,
        "n_sessions": len(p.sessions), "age_band": p.age_band, "sex": p.sex,
        "first_session": min((s.session_date for s in p.sessions if s.session_date), default=None),
        "last_session": max((s.session_date for s in p.sessions if s.session_date), default=None),
    } for p in rows]}


def participant_detail(db: Session, p: Participant) -> dict[str, Any]:
    recs = db.scalars(select(Recording).where(Recording.participant_id == p.id).options(
        selectinload(Recording.predictions), selectinload(Recording.session))).all()
    sessions = []
    for s in p.sessions:
        srecs = [r for r in recs if r.session_id == s.id]
        sessions.append({
            "session_key": s.session_key, "session_index": s.session_index, "session_date": s.session_date, "task": s.task,
            "assessment_label": s.assessment_label, "assessment_score": s.assessment_score, "assessment_instrument": s.assessment_instrument,
            "recordings": [{
                "recording_key": r.recording_key, "duration_s": r.duration_s, "quality_usable": r.quality_usable,
                "quality_flags": (r.quality or {}).get("flags", []), "split": r.split, "source": r.source,
                "has_transcript": bool(r.transcript), "has_speaker_turns": r.has_speaker_turns,
                "latest_probability": max(r.predictions, key=lambda x: x.created_at).probability if r.predictions else None,
            } for r in srecs],
        })
    return {"pseudonym": p.pseudonym, "dataset_source": p.dataset_source, "cohort_label": p.cohort_label, "is_synthetic": p.is_synthetic,
            "age_band": p.age_band, "sex": p.sex, "education_band": p.education_band, "consent_research": p.consent_research,
            "sessions": sessions}


def participant_feature_frame(db: Session, p: Participant) -> pd.DataFrame:
    recs = db.scalars(select(Recording).where(Recording.participant_id == p.id)
                      .options(selectinload(Recording.feature_sets), selectinload(Recording.session))).all()
    rows = []
    for r in recs:
        feats = {k: v for k, v in latest_features(r).items() if not k.startswith("emb_")}
        rows.append({"participant_id": p.pseudonym, "session_id": r.session.session_key, "session_index": r.session.session_index,
                     "session_timestamp": pd.Timestamp(r.session.session_date) if r.session.session_date else pd.NaT,
                     "recording_id": r.recording_key, "quality_usable": bool(r.quality_usable), **feats})
    return pd.DataFrame(rows)


def feature_trends(db: Session, p: Participant, modality: str | None) -> dict[str, Any]:
    df = participant_feature_frame(db, p)
    if df.empty:
        return {"participant": p.pseudonym, "sessions": [], "features": []}
    meta = {"participant_id", "session_id", "session_index", "session_timestamp", "recording_id", "quality_usable"}
    feats = [c for c in df.columns if c not in meta and feature_modality(c) != "descriptive" and df[c].notna().any()]
    if modality:
        feats = [f for f in feats if feature_modality(f) == modality]
    sessions = session_table(df, feats)
    return {
        "participant": p.pseudonym,
        "features": [{"name": f, "label": label_for(f), "modality": feature_modality(f)} for f in feats],
        "sessions": [{"session_key": r["session_id"], "session_index": int(r["session_index"]),
                      "session_date": None if pd.isna(r["session_timestamp"]) else pd.Timestamp(r["session_timestamp"]).date(),
                      "values": {f: (None if pd.isna(r[f]) else float(r[f])) for f in feats}} for _, r in sessions.iterrows()],
    }


def live_longitudinal(db: Session, p: Participant) -> dict[str, Any]:
    """Recompute the participant's baseline-relative analysis from current DB features (includes new uploads)."""
    cfg = get_config()["longitudinal"]
    df = participant_feature_frame(db, p)
    tracked = [f for f in cfg["tracked_features"] if f in df.columns]
    cohort = latest_report(db, "longitudinal", required=False)
    within = (cohort.payload.get("population_within_sd") if cohort else None) or {}
    if df.empty or not tracked:
        return {"participant_id": p.pseudonym, "status": "insufficient_data", "status_detail": "No usable features.", "sessions": []}
    sessions = session_table(df, tracked)
    if not within:
        within = population_within_sd(sessions, tracked, cfg["baseline_sessions"])
    within = {f: within.get(f, 1.0) for f in tracked}
    result = analyse_participant(sessions, tracked, within, cfg)
    if cohort:  # attach population anomaly scores computed in the batch run, where available
        batch = next((x for x in cohort.payload["participants"] if x["participant_id"] == p.pseudonym), None)
        if batch:
            scores = {s["session_id"]: (s.get("anomaly_score"), s.get("is_anomaly")) for s in batch["sessions"]}
            for s in result.get("sessions", []):
                if s["session_id"] in scores:
                    s["anomaly_score"], s["is_anomaly"] = scores[s["session_id"]]
    return result


def report_payload(db: Session, report_type: str) -> dict[str, Any]:
    r: MonitoringReport = latest_report(db, report_type)
    return {"report_type": r.report_type, "created_at": r.created_at, "dataset_version": r.dataset_version,
            "is_synthetic": r.is_synthetic, "payload": r.payload}
