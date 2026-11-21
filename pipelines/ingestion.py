"""Stages 1–3: dataset ingestion, schema validation and audio quality metadata.

Output: ``data/interim/manifest.parquet`` (canonical manifest with audio hash, duration and native
sample rate filled in), ``artifacts/reports/data_quality.json`` and ``artifacts/reports/lineage.json``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import soundfile as sf

from ml.config import Config, resolve_path
from ml.data.adapters import DatasetUnavailableError, get_adapter
from ml.data.schema import conform
from ml.data.synthetic import SyntheticConfig, generate_synthetic_dataset
from ml.data.validation import validate_manifest
from ml.utils import dataframe_fingerprint, file_sha256, get_logger, relpath, write_json

log = get_logger(__name__)


def generate_synthetic(cfg: Config, n_participants: int | None = None) -> pd.DataFrame:
    syn_cfg = dict(cfg["synthetic"])
    if n_participants:
        syn_cfg["n_participants"] = n_participants
    out = cfg.path("synthetic")
    meta = generate_synthetic_dataset(out, SyntheticConfig.from_config(syn_cfg, cfg["project"]["seed"]))
    log.info("synthetic dataset written", extra={"ctx": {"recordings": len(meta), "dir": relpath(out)}})
    return meta


def _audio_info(path: str) -> dict[str, Any]:
    full = resolve_path(path)
    try:
        info = sf.info(str(full))
        return {"recording_duration": float(info.duration), "sample_rate": float(info.samplerate), "audio_sha256": file_sha256(full)}
    except Exception:
        try:
            import librosa

            return {"recording_duration": float(librosa.get_duration(path=str(full))),
                    "sample_rate": float(librosa.get_samplerate(str(full))), "audio_sha256": file_sha256(full)}
        except Exception as exc:
            log.warning("could not read audio header", extra={"ctx": {"file": path, "reason": type(exc).__name__}})
            return {"recording_duration": None, "sample_rate": None, "audio_sha256": file_sha256(full) if full.exists() else None}


def ingest(cfg: Config, strict: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames, sources = [], []
    for key in cfg["project"]["datasets"]:
        adapter = get_adapter(key, cfg["datasets"])
        try:
            df = adapter.load()
        except DatasetUnavailableError as exc:
            log.warning(str(exc))
            sources.append({**adapter.describe(), "status": "unavailable", "message": str(exc)})
            continue
        frames.append(df)
        sources.append({**adapter.describe(), "status": "ingested", "n_recordings": len(df)})
    if not frames:
        raise DatasetUnavailableError(
            "No configured dataset is available. Generate demo data with `cognitrack generate-synthetic` "
            "or place a licensed dataset under data/raw/ (see docs/DATA_GOVERNANCE.md)."
        )
    manifest = pd.concat(frames, ignore_index=True)
    info = pd.DataFrame([_audio_info(p) for p in manifest["recording_path"]])
    for col in info.columns:
        manifest[col] = manifest[col].where(manifest[col].notna(), info[col]) if col in manifest else info[col]
    manifest = conform(manifest)
    report = validate_manifest(manifest, strict=strict)
    out = cfg.path("interim") / "manifest.parquet"
    manifest.to_parquet(out, index=False)
    version = dataframe_fingerprint(manifest[["recording_id", "audio_sha256", "assessment_label"]])
    report["dataset_version"] = version
    report["sources"] = sources
    write_json(cfg.path("report_data") / "data_quality.json", report)
    write_json(cfg.path("report_data") / "lineage.json", {
        "created_at": datetime.now(UTC).isoformat(), "stage": "ingest", "dataset_version": version,
        "sources": sources, "output": relpath(out), "n_recordings": len(manifest),
    })
    log.info("ingestion complete", extra={"ctx": {"recordings": len(manifest), "version": version, "passed": report["passed"]}})
    return manifest, report
