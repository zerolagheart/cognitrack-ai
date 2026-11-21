"""Local Whisper transcription via Hugging Face Transformers.

Checkpoints are never downloaded implicitly: run ``cognitrack download-models --whisper`` first.
If the checkpoint is not cached, :class:`TranscriberUnavailable` explains how to fetch it and the
pipeline continues with dataset transcripts (or none).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from ml.utils import get_logger

log = get_logger(__name__)


class TranscriberUnavailable(RuntimeError):
    pass


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@lru_cache(maxsize=1)
def _pipeline(checkpoint: str, revision: str, device: str, cache_dir: str) -> Any:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise TranscriberUnavailable("transformers/torch not installed: pip install -e .[deep]") from exc
    try:
        return pipeline(
            "automatic-speech-recognition", model=checkpoint, revision=revision, device=resolve_device(device),
            model_kwargs={"cache_dir": cache_dir, "local_files_only": True},
        )
    except OSError as exc:
        raise TranscriberUnavailable(
            f"Whisper checkpoint '{checkpoint}' is not cached locally. Run: cognitrack download-models --whisper"
        ) from exc


def transcribe(signal: np.ndarray, sr: int, cfg: dict[str, Any], cache_dir: str) -> dict[str, Any]:
    """Transcribe a mono signal. Returns ``{"text": ..., "checkpoint": ..., "revision": ...}``."""
    asr = _pipeline(cfg["checkpoint"], cfg["revision"], cfg["device"], cache_dir)
    result = asr({"raw": signal.astype(np.float32), "sampling_rate": sr}, chunk_length_s=cfg["chunk_length_s"])
    return {"text": result["text"].strip(), "checkpoint": cfg["checkpoint"], "revision": cfg["revision"]}
