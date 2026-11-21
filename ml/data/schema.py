"""Canonical recording manifest schema.

Every dataset adapter must emit a DataFrame conforming to :data:`MANIFEST_SCHEMA`. Downstream stages
(feature extraction, splitting, training, longitudinal analysis) depend only on this schema, never on
dataset-specific layouts.
"""

from __future__ import annotations

try:  # pandera >= 0.24 exposes the pandas API under pandera.pandas
    import pandera.pandas as pa
except ImportError:  # pragma: no cover - older pandera
    import pandera as pa

import pandas as pd

LABEL_IMPAIRED = "cognitive_impairment"
LABEL_CONTROL = "healthy_control"
VALID_LABELS = [LABEL_IMPAIRED, LABEL_CONTROL]

# Column order of the canonical manifest. Comments document semantics (see docs/DATA_DICTIONARY.md).
MANIFEST_COLUMNS: list[str] = [
    "participant_id",  # pseudonymous, dataset-prefixed identifier
    "session_id",  # assessment session (one participant, one visit)
    "recording_id",  # unique recording identifier
    "recording_path",  # path relative to the project root
    "dataset_source",  # adapter name, e.g. synthetic / adresso / dementiabank
    "dataset_split_hint",  # official split published with the dataset, if any
    "session_index",  # 0-based visit order within participant
    "session_timestamp",  # visit date (may be relative/synthetic)
    "recording_duration",  # seconds; filled by audio quality stage
    "sample_rate",  # native sample rate (Hz)
    "task",  # elicitation task, e.g. picture_description
    "language",
    "transcript",  # dataset-provided or ASR transcript of the participant's speech
    "transcript_source",  # dataset | whisper | none
    "turns_path",  # JSON file with speaker-attributed turns (only if the dataset provides them)
    "assessment_label",  # cognitive_impairment | healthy_control | missing
    "assessment_score",  # e.g. MMSE, only where legitimately available
    "assessment_instrument",
    "age_band",
    "sex",
    "education_band",
    "consent_research",
    "is_synthetic",
    "audio_sha256",
    "provenance",  # JSON string: source files, adapter version, licence note
]

MANIFEST_SCHEMA = pa.DataFrameSchema(
    {
        "participant_id": pa.Column(str, pa.Check.str_matches(r"^[A-Za-z0-9_\-]+$"), nullable=False),
        "session_id": pa.Column(str, nullable=False),
        "recording_id": pa.Column(str, nullable=False, unique=True),
        "recording_path": pa.Column(str, nullable=False),
        "dataset_source": pa.Column(str, nullable=False),
        "dataset_split_hint": pa.Column(str, nullable=True),
        "session_index": pa.Column(int, pa.Check.ge(0), nullable=False),
        "session_timestamp": pa.Column("datetime64[ns]", nullable=True),
        "recording_duration": pa.Column(float, pa.Check.ge(0), nullable=True),
        "sample_rate": pa.Column(float, pa.Check.gt(0), nullable=True),
        "task": pa.Column(str, nullable=True),
        "language": pa.Column(str, nullable=True),
        "transcript": pa.Column(str, nullable=True),
        "transcript_source": pa.Column(str, pa.Check.isin(["dataset", "whisper", "none"]), nullable=False),
        "turns_path": pa.Column(str, nullable=True),
        "assessment_label": pa.Column(str, pa.Check.isin(VALID_LABELS), nullable=True),
        "assessment_score": pa.Column(float, nullable=True),
        "assessment_instrument": pa.Column(str, nullable=True),
        "age_band": pa.Column(str, nullable=True),
        "sex": pa.Column(str, nullable=True),
        "education_band": pa.Column(str, nullable=True),
        "consent_research": pa.Column(bool, nullable=False),
        "is_synthetic": pa.Column(bool, nullable=False),
        "audio_sha256": pa.Column(str, nullable=True),
        "provenance": pa.Column(str, nullable=False),
    },
    strict=True,
    ordered=False,
    coerce=True,
)


def empty_manifest() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype="object") for col in MANIFEST_COLUMNS})


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing optional columns as nulls, order columns and validate against the schema."""
    df = df.copy()
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["transcript_source"] = df["transcript_source"].fillna("none")
    df["consent_research"] = df["consent_research"].fillna(False).astype(bool)
    df["is_synthetic"] = df["is_synthetic"].fillna(False).astype(bool)
    df["session_timestamp"] = pd.to_datetime(df["session_timestamp"], errors="coerce")
    for col in ("recording_duration", "sample_rate", "assessment_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["session_index"] = pd.to_numeric(df["session_index"], errors="coerce").fillna(0).astype(int)
    df = df[MANIFEST_COLUMNS]
    return MANIFEST_SCHEMA.validate(df, lazy=True)
