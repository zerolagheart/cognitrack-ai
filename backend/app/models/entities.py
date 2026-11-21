"""Relational schema (see docs/ARCHITECTURE.md § Database for the ER diagram)."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database.session import Base, JSONType


def _now() -> datetime:
    return datetime.now(UTC)


class Participant(Base):
    """Pseudonymous research participant. Contains no direct identifiers."""

    __tablename__ = "participants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pseudonym: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    dataset_source: Mapped[str] = mapped_column(String(32), index=True)
    cohort_label: Mapped[str | None] = mapped_column(String(32))
    age_band: Mapped[str | None] = mapped_column(String(16))
    sex: Mapped[str | None] = mapped_column(String(8))
    education_band: Mapped[str | None] = mapped_column(String(16))
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_research: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_recorded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    sessions: Mapped[list[AssessmentSession]] = relationship(back_populates="participant", cascade="all, delete-orphan",
                                                             order_by="AssessmentSession.session_index")
    recordings: Mapped[list[Recording]] = relationship(back_populates="participant", cascade="all, delete-orphan")


class IdentityLink(Base):
    """Restricted re-identification link (salted hash of an external reference).

    Separated from research tables, never exposed through the API, and left empty for public or
    synthetic datasets. A production deployment would place it in a separate database/schema with
    its own access policy.
    """

    __tablename__ = "identity_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), unique=True)
    external_reference_hash: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AssessmentSession(Base):
    __tablename__ = "assessment_sessions"
    __table_args__ = (UniqueConstraint("participant_id", "session_index", name="uq_session_participant_index"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_key: Mapped[str] = mapped_column(String(96), unique=True, index=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), index=True)
    session_index: Mapped[int] = mapped_column(Integer)
    session_date: Mapped[date | None] = mapped_column(Date)
    task: Mapped[str | None] = mapped_column(String(64))
    assessment_label: Mapped[str | None] = mapped_column(String(32))
    assessment_score: Mapped[float | None] = mapped_column(Float)
    assessment_instrument: Mapped[str | None] = mapped_column(String(96))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    participant: Mapped[Participant] = relationship(back_populates="sessions")
    recordings: Mapped[list[Recording]] = relationship(back_populates="session", cascade="all, delete-orphan")


class Recording(Base):
    __tablename__ = "recordings"
    __table_args__ = (
        CheckConstraint("duration_s IS NULL OR duration_s >= 0", name="ck_recording_duration"),
        Index("ix_recordings_participant_session", "participant_id", "session_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    recording_key: Mapped[str] = mapped_column(String(96), unique=True, index=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("assessment_sessions.id", ondelete="CASCADE"))
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"))
    dataset_source: Mapped[str] = mapped_column(String(32))
    source: Mapped[str] = mapped_column(String(16), default="dataset")  # dataset | upload
    storage_path: Mapped[str | None] = mapped_column(String(512))  # None once raw audio is deleted
    raw_audio_retained: Mapped[bool] = mapped_column(Boolean, default=False)
    audio_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    duration_s: Mapped[float | None] = mapped_column(Float)
    sample_rate: Mapped[float | None] = mapped_column(Float)
    transcript: Mapped[str | None] = mapped_column(Text)
    transcript_source: Mapped[str] = mapped_column(String(16), default="none")
    has_speaker_turns: Mapped[bool] = mapped_column(Boolean, default=False)
    turns_path: Mapped[str | None] = mapped_column(String(512))
    quality_usable: Mapped[bool | None] = mapped_column(Boolean)
    quality: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    waveform: Mapped[dict[str, Any] | None] = mapped_column(JSONType)  # downsampled envelope, not reconstructable audio
    split: Mapped[str | None] = mapped_column(String(16), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    participant: Mapped[Participant] = relationship(back_populates="recordings")
    session: Mapped[AssessmentSession] = relationship(back_populates="recordings")
    feature_sets: Mapped[list[FeatureSet]] = relationship(back_populates="recording", cascade="all, delete-orphan")
    predictions: Mapped[list[Prediction]] = relationship(back_populates="recording", cascade="all, delete-orphan")


class FeatureSet(Base):
    """Extracted features for one recording and one extraction configuration."""

    __tablename__ = "feature_sets"
    __table_args__ = (UniqueConstraint("recording_id", "feature_version", name="uq_feature_recording_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    recording_id: Mapped[int] = mapped_column(ForeignKey("recordings.id", ondelete="CASCADE"), index=True)
    feature_version: Mapped[str] = mapped_column(String(64))
    values: Mapped[dict[str, Any]] = mapped_column(JSONType)
    backends: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    recording: Mapped[Recording] = relationship(back_populates="feature_sets")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    model_type: Mapped[str] = mapped_column(String(32))
    strategy: Mapped[str] = mapped_column(String(16))
    modalities: Mapped[list[str]] = mapped_column(JSONType)
    dataset_version: Mapped[str] = mapped_column(String(32))
    calibration: Mapped[str | None] = mapped_column(String(16))
    artifact_path: Mapped[str] = mapped_column(String(512))
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64))
    is_serving: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    trained_on_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    summary_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_key: Mapped[str] = mapped_column(String(128), index=True)
    dataset_version: Mapped[str] = mapped_column(String(32))
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id", ondelete="SET NULL"))
    status: Mapped[str] = mapped_column(String(16))
    modality_set: Mapped[str] = mapped_column(String(64))
    strategy: Mapped[str] = mapped_column(String(16))
    model: Mapped[str] = mapped_column(String(32))
    seed: Mapped[int | None] = mapped_column(Integer)
    config_hash: Mapped[str | None] = mapped_column(String(32))
    params: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    cv_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    test_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    reason: Mapped[str | None] = mapped_column(Text)
    duration_s: Mapped[float | None] = mapped_column(Float)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (UniqueConstraint("experiment_key", "dataset_version", name="uq_experiment_dataset"),)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("probability >= 0 AND probability <= 1", name="ck_prediction_probability"),
        Index("ix_predictions_recording_created", "recording_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    recording_id: Mapped[int] = mapped_column(ForeignKey("recordings.id", ondelete="CASCADE"))
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id", ondelete="CASCADE"), index=True)
    probability: Mapped[float] = mapped_column(Float)
    model_label: Mapped[str] = mapped_column(String(32))
    abstain: Mapped[bool] = mapped_column(Boolean, default=False)
    uncertainty: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    explanation: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    origin: Mapped[str] = mapped_column(String(16), default="batch")  # batch | api
    split: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    recording: Mapped[Recording] = relationship(back_populates="predictions")
    model_version: Mapped[ModelVersion] = relationship()


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_type: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True, default="queued")  # queued|running|succeeded|failed
    recording_id: Mapped[int | None] = mapped_column(ForeignKey("recordings.id", ondelete="SET NULL"))
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MonitoringReport(Base):
    """Versioned pipeline report payloads (drift, quality, evaluation, fairness, ...)."""

    __tablename__ = "monitoring_reports"
    __table_args__ = (Index("ix_reports_type_created", "report_type", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_type: Mapped[str] = mapped_column(String(32))
    dataset_version: Mapped[str | None] = mapped_column(String(32))
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id", ondelete="SET NULL"))
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AuditEvent(Base):
    """Access log for participant-level data (role + resource; no request bodies, no PII)."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    actor_role: Mapped[str] = mapped_column(String(16))
    action: Mapped[str] = mapped_column(String(32))
    resource: Mapped[str] = mapped_column(String(32))
    resource_id: Mapped[str | None] = mapped_column(String(96))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
