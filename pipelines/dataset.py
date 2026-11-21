"""Stage 8: assemble the modelling table and create participant-independent splits."""

from __future__ import annotations

import pandas as pd

from ml.config import Config
from ml.data.splitting import split_participants, split_summary
from ml.utils import get_logger, write_json

log = get_logger(__name__)


def build_dataset(cfg: Config) -> pd.DataFrame:
    manifest = pd.read_parquet(cfg.path("interim") / "manifest.parquet")
    features = pd.read_parquet(cfg.path("processed") / "features.parquet")
    df = manifest.merge(features, on="recording_id", how="left", validate="one_to_one")
    emb_path = cfg.path("processed") / "embeddings.parquet"
    if emb_path.exists():
        emb = pd.read_parquet(emb_path)
        df = df.merge(emb, on="recording_id", how="left", validate="one_to_one")
    else:
        log.warning("no embeddings found; embedding modality will be skipped (run: cognitrack embed)")
    scfg = cfg["split"]
    df = split_participants(df, scfg["label_column"], scfg["test_size"], scfg["val_size"], cfg["project"]["seed"])
    df.to_parquet(cfg.path("processed") / "dataset.parquet", index=False)
    summary = split_summary(df, scfg["label_column"])
    write_json(cfg.path("report_data") / "splits.json", {
        "summary": summary, "participant_independent": True, "seed": cfg["project"]["seed"],
        "excluded_unusable_audio": int((~df["quality_usable"].fillna(False).astype(bool)).sum()),
    })
    log.info("dataset assembled", extra={"ctx": {"rows": len(df), "splits": {k: v["participants"] for k, v in summary.items()}}})
    return df


def load_dataset(cfg: Config) -> pd.DataFrame:
    path = cfg.path("processed") / "dataset.parquet"
    if not path.exists():
        raise FileNotFoundError("data/processed/dataset.parquet not found. Run: cognitrack build-dataset")
    return pd.read_parquet(path)
