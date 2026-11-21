"""Small helpers shared by services."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.errors import NotFoundError, UnavailableError
from backend.app.models import AuditEvent, MonitoringReport, Participant, Recording
from ml.config import Config
from ml.training.trainer import config_hash


def utcnow() -> datetime:
    return datetime.now(UTC)


def feature_version(cfg: Config) -> str:
    """Feature-set version = hash of extraction-relevant configuration."""
    return f"fx-{config_hash(cfg)}"


def latest_report(db: Session, report_type: str, required: bool = True) -> MonitoringReport | None:
    report = db.scalar(select(MonitoringReport).where(MonitoringReport.report_type == report_type)
                       .order_by(MonitoringReport.created_at.desc(), MonitoringReport.id.desc()).limit(1))
    if report is None and required:
        raise UnavailableError(f"No '{report_type}' report published yet. Run the pipeline, then `cognitrack publish`.")
    return report


def get_participant(db: Session, pseudonym: str) -> Participant:
    p = db.scalar(select(Participant).where(Participant.pseudonym == pseudonym))
    if p is None:
        raise NotFoundError(f"Participant '{pseudonym}' not found")
    return p


def get_recording(db: Session, key: str) -> Recording:
    r = db.scalar(select(Recording).where(Recording.recording_key == key))
    if r is None:
        raise NotFoundError(f"Recording '{key}' not found")
    return r


def audit(db: Session, role: str, action: str, resource: str, resource_id: str | None = None) -> None:
    db.add(AuditEvent(actor_role=role, action=action, resource=resource, resource_id=resource_id))
    db.commit()


def latest_features(recording: Recording) -> dict[str, Any]:
    if not recording.feature_sets:
        return {}
    return max(recording.feature_sets, key=lambda f: f.created_at).values
