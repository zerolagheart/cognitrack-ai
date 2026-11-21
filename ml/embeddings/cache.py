"""Content-addressed embedding cache stored as Parquet.

Key = (audio SHA-256, encoder version). Re-running extraction on unchanged audio with the same encoder
version is a no-op; changing the checkpoint/revision/pooling produces new rows rather than overwriting.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


class EmbeddingCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._df = pd.read_parquet(self.path) if self.path.exists() else pd.DataFrame(
            columns=["audio_sha256", "encoder_version", "embedding", "created_at"]
        )
        self._index = {(h, v): i for i, (h, v) in enumerate(zip(self._df["audio_sha256"], self._df["encoder_version"], strict=True))}
        self._pending: list[dict] = []

    def get(self, audio_sha256: str, encoder_version: str) -> np.ndarray | None:
        idx = self._index.get((audio_sha256, encoder_version))
        if idx is None:
            for row in self._pending:
                if row["audio_sha256"] == audio_sha256 and row["encoder_version"] == encoder_version:
                    return np.asarray(row["embedding"], dtype=np.float32)
            return None
        return np.asarray(self._df.iloc[idx]["embedding"], dtype=np.float32)

    def put(self, audio_sha256: str, encoder_version: str, embedding: np.ndarray) -> None:
        self._pending.append({
            "audio_sha256": audio_sha256, "encoder_version": encoder_version,
            "embedding": embedding.astype(np.float32).tolist(), "created_at": datetime.now(UTC).isoformat(),
        })

    def flush(self) -> None:
        if not self._pending:
            return
        new = pd.DataFrame(self._pending)
        self._df = pd.concat([self._df, new], ignore_index=True) if len(self._df) else new
        self._df.to_parquet(self.path, index=False)
        self._index = {(h, v): i for i, (h, v) in enumerate(zip(self._df["audio_sha256"], self._df["encoder_version"], strict=True))}
        self._pending = []

    def __len__(self) -> int:
        return len(self._df) + len(self._pending)
