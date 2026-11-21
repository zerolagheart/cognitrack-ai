"""Data validation and profiling for the canonical manifest.

Produces a machine-readable data-quality report used by the pipeline (to fail fast), the API (data
quality indicators) and documentation. No values are imputed here; missingness is only measured.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.config import resolve_path
from ml.data.schema import MANIFEST_SCHEMA


class DataValidationError(ValueError):
    pass


def schema_errors(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return pandera failure cases as plain dicts (empty list if valid)."""
    try:
        MANIFEST_SCHEMA.validate(df, lazy=True)
        return []
    except Exception as exc:  # pandera.errors.SchemaErrors
        cases = getattr(exc, "failure_cases", None)
        if cases is None:
            return [{"error": str(exc)}]
        return cases.astype(str).head(200).to_dict(orient="records")


def find_duplicates(df: pd.DataFrame) -> dict[str, list[str]]:
    """Duplicate recording IDs and duplicate audio content (same SHA-256 under different IDs)."""
    dup_ids = df.loc[df["recording_id"].duplicated(keep=False), "recording_id"].unique().tolist()
    hashed = df.dropna(subset=["audio_sha256"])
    dup_audio = (
        hashed.groupby("audio_sha256")["recording_id"].apply(list).loc[lambda s: s.map(len) > 1].tolist()
        if not hashed.empty
        else []
    )
    # The same audio under two different participants is a strong leakage/provenance signal.
    cross_participant = []
    for group in dup_audio:
        pids = df.loc[df["recording_id"].isin(group), "participant_id"].unique()
        if len(pids) > 1:
            cross_participant.append(group)
    return {"duplicate_recording_ids": dup_ids, "duplicate_audio_groups": dup_audio, "cross_participant_audio": cross_participant}


def missing_value_report(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    report = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        report[col] = {"missing": n_missing, "missing_fraction": round(n_missing / max(len(df), 1), 4)}
    return report


def class_distribution(df: pd.DataFrame, label_col: str = "assessment_label") -> dict[str, Any]:
    rec = df[label_col].fillna("missing").value_counts().to_dict()
    part = df.groupby("participant_id")[label_col].agg(lambda s: s.dropna().iloc[0] if s.notna().any() else "missing")
    inconsistent = (
        df.dropna(subset=[label_col]).groupby("participant_id")[label_col].nunique().loc[lambda s: s > 1].index.tolist()
    )
    return {
        "recordings": {k: int(v) for k, v in rec.items()},
        "participants": {k: int(v) for k, v in part.value_counts().to_dict().items()},
        "participants_with_changing_label": inconsistent,
    }


def missing_files(df: pd.DataFrame) -> list[str]:
    return [rid for rid, p in zip(df["recording_id"], df["recording_path"], strict=True) if not resolve_path(p).exists()]


def profile_manifest(df: pd.DataFrame) -> dict[str, Any]:
    sessions = df.groupby("participant_id")["session_id"].nunique()
    return {
        "n_recordings": int(len(df)),
        "n_participants": int(df["participant_id"].nunique()),
        "n_sessions": int(df["session_id"].nunique()),
        "sessions_per_participant": {
            "min": int(sessions.min()) if len(sessions) else 0,
            "median": float(sessions.median()) if len(sessions) else 0.0,
            "max": int(sessions.max()) if len(sessions) else 0,
        },
        "datasets": {k: int(v) for k, v in df["dataset_source"].value_counts().to_dict().items()},
        "synthetic_fraction": float(df["is_synthetic"].mean()) if len(df) else 0.0,
        "with_transcript": int(df["transcript"].notna().sum()),
        "with_speaker_turns": int(df["turns_path"].notna().sum()),
        "consented": int(df["consent_research"].sum()),
    }


def validate_manifest(df: pd.DataFrame, strict: bool = True) -> dict[str, Any]:
    """Run all checks. With ``strict``, schema errors, duplicate IDs or missing files raise."""
    report = {
        "profile": profile_manifest(df),
        "schema_errors": schema_errors(df),
        "duplicates": find_duplicates(df),
        "missing_values": missing_value_report(df),
        "class_distribution": class_distribution(df),
        "missing_files": missing_files(df),
        "non_consented_recordings": df.loc[~df["consent_research"].astype(bool), "recording_id"].tolist(),
    }
    blocking = []
    if report["schema_errors"]:
        blocking.append(f"{len(report['schema_errors'])} schema violations")
    if report["duplicates"]["duplicate_recording_ids"]:
        blocking.append("duplicate recording ids")
    if report["duplicates"]["cross_participant_audio"]:
        blocking.append("identical audio assigned to different participants")
    if report["missing_files"]:
        blocking.append(f"{len(report['missing_files'])} audio files missing")
    report["blocking_issues"] = blocking
    report["passed"] = not blocking
    if strict and blocking:
        raise DataValidationError("Manifest validation failed: " + "; ".join(blocking))
    return report
