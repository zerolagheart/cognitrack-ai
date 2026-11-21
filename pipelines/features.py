"""Stages 4–7: preprocessing, feature extraction, embedding generation and feature storage.

Outputs ``data/processed/features.parquet`` and ``data/processed/embeddings.parquet``. Embeddings are
served from a content-addressed cache (``data/processed/embedding_cache.parquet``), so unchanged audio
is never re-encoded.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pandas as pd

from ml.config import Config, load_config, resolve_path
from ml.embeddings.cache import EmbeddingCache
from ml.embeddings.encoders import build_encoder
from ml.features.extraction import extract_recording, sentence_model_for
from ml.preprocessing.audio import AudioLoadError, load_audio
from ml.utils import get_logger, write_json

log = get_logger(__name__)


def _worker(args: tuple[dict[str, Any], str | None]) -> dict[str, Any]:
    row, config_path = args
    cfg = load_config(config_path)
    return extract_recording(row, cfg, sentence_model_for(cfg)).flat()


def extract_features(cfg: Config, config_path: str | None = None, workers: int = 1, limit: int | None = None) -> pd.DataFrame:
    manifest = pd.read_parquet(cfg.path("interim") / "manifest.parquet")
    if limit:
        manifest = manifest.head(limit)
    rows = manifest.to_dict(orient="records")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            flat = list(pool.map(_worker, [(r, config_path) for r in rows], chunksize=4))
    else:
        sentence_model = sentence_model_for(cfg)
        flat = []
        for i, r in enumerate(rows, 1):
            flat.append(extract_recording(r, cfg, sentence_model).flat())
            if i % 25 == 0:
                log.info("feature extraction progress", extra={"ctx": {"done": i, "total": len(rows)}})
    features = pd.DataFrame(flat)
    out = cfg.path("processed") / "features.parquet"
    features.to_parquet(out, index=False)
    errors = features[features["extraction_error"].notna()][["recording_id", "extraction_error"]]
    write_json(cfg.path("report_data") / "feature_extraction.json", {
        "n_recordings": len(features), "n_errors": len(errors), "errors": errors.to_dict(orient="records"),
        "quality_flag_counts": features["quality_flags"].str.split(",").explode().loc[lambda s: s.fillna("") != ""].value_counts().to_dict(),
        "backends": features["backends"].value_counts().to_dict(),
    })
    log.info("features written", extra={"ctx": {"rows": len(features), "errors": len(errors)}})
    return features


def extract_embeddings(cfg: Config, encoder_name: str | None = None) -> pd.DataFrame:
    emb_cfg = cfg["embeddings"]
    encoder_name = encoder_name or emb_cfg["encoder"]
    encoder = build_encoder(encoder_name, emb_cfg, str(cfg.path("model_cache")))
    manifest = pd.read_parquet(cfg.path("interim") / "manifest.parquet")
    cache = EmbeddingCache(cfg.path("processed") / "embedding_cache.parquet")
    sr = cfg["audio"]["target_sample_rate"]
    todo = [r for r in manifest.itertuples() if r.audio_sha256 and cache.get(r.audio_sha256, encoder.version) is None]
    log.info("embedding extraction", extra={"ctx": {"encoder": encoder.version, "to_compute": len(todo), "cached": len(manifest) - len(todo)}})
    batch_size = int(emb_cfg["batch_size"])
    for i in range(0, len(todo), batch_size):
        batch = todo[i: i + batch_size]
        signals, keys = [], []
        for r in batch:
            try:
                sig, _, _ = load_audio(resolve_path(r.recording_path), sr)
            except AudioLoadError as exc:
                log.warning(str(exc))
                continue
            signals.append(sig[: int(emb_cfg["max_seconds"] * sr)])
            keys.append(r.audio_sha256)
        if signals:
            for key, vec in zip(keys, encoder.encode_batch(signals, sr), strict=True):
                cache.put(key, encoder.version, vec)
        cache.flush()
    rows = []
    for r in manifest.itertuples():
        vec = cache.get(r.audio_sha256, encoder.version) if r.audio_sha256 else None
        if vec is not None:
            rows.append({"recording_id": r.recording_id, "embedding_encoder": encoder.version,
                         **{f"emb_{j:03d}": float(v) for j, v in enumerate(vec)}})
    emb = pd.DataFrame(rows)
    emb.to_parquet(cfg.path("processed") / "embeddings.parquet", index=False)
    return emb
