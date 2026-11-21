"""Pydantic request/response schemas.

Report endpoints return versioned JSON payloads produced by the pipeline (``ReportOut.payload``); their
structure is documented in docs/API_DOCUMENTATION.md rather than duplicated as Pydantic classes.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    database: bool
    model_loaded: bool
    serving_model: str | None
    auth_enabled: bool
    disclaimer: str


class ReportOut(BaseModel):
    report_type: str
    created_at: datetime
    dataset_version: str | None
    is_synthetic: bool
    payload: dict[str, Any]


class ParticipantSummary(BaseModel):
    pseudonym: str
    dataset_source: str
    cohort_label: str | None
    is_synthetic: bool
    n_sessions: int
    age_band: str | None
    sex: str | None
    first_session: date | None
    last_session: date | None


class ParticipantList(BaseModel):
    total: int
    items: list[ParticipantSummary]


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    recording_key: str
    model_experiment_id: str
    probability: float = Field(ge=0, le=1)
    model_label: str
    abstain: bool
    uncertainty: dict[str, Any] | None
    explanation: dict[str, Any] | None
    origin: str
    split: str | None
    created_at: datetime
    disclaimer: str


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    job_type: str
    status: str
    detail: dict[str, Any] | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class UploadOut(BaseModel):
    recording_key: str
    participant: str
    session_key: str
    job: JobOut
    raw_audio_retention: str


class RecordingOut(BaseModel):
    recording_key: str
    participant: str
    session_key: str
    session_index: int
    session_date: date | None
    dataset_source: str
    source: str
    duration_s: float | None
    sample_rate: float | None
    transcript: str | None
    transcript_source: str
    has_speaker_turns: bool
    quality_usable: bool | None
    quality: dict[str, Any] | None
    split: str | None
    raw_audio_retained: bool
    features: dict[str, Any]
    feature_modalities: dict[str, str]
    latest_prediction: PredictionOut | None


class ModelVersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    experiment_key: str
    model_type: str
    strategy: str
    modalities: list[str]
    dataset_version: str
    calibration: str | None
    mlflow_run_id: str | None
    is_serving: bool
    trained_on_synthetic: bool
    summary_metrics: dict[str, Any] | None
    created_at: datetime


class ExperimentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    experiment_key: str
    dataset_version: str
    status: str
    modality_set: str
    strategy: str
    model: str
    seed: int | None
    config_hash: str | None
    params: dict[str, Any] | None
    cv_metrics: dict[str, Any] | None
    test_metrics: dict[str, Any] | None
    reason: str | None
    duration_s: float | None
    mlflow_run_id: str | None
    created_at: datetime


class FeaturePredictRequest(BaseModel):
    """Score a pre-computed feature vector (research use, e.g. features computed offline)."""

    features: dict[str, float | None] = Field(..., min_length=1)
    quality_flags: list[str] = Field(default_factory=list)
