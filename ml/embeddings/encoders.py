"""Audio representation encoders behind a common interface.

* :class:`SpectralStatsEncoder` — download-free, CPU-only log-mel statistics (demo / fallback mode).
* :class:`HFSpeechEncoder` — pretrained self-supervised speech models from Hugging Face (HuBERT,
  wav2vec 2.0), mean-pooled over time from a configurable hidden layer.

New encoders register in :data:`ENCODERS`. Every encoder exposes a ``version`` string (checkpoint +
revision + pooling) that is stored with cached embeddings so incompatible vectors are never mixed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from ml.nlp.transcription import resolve_device
from ml.preprocessing.audio import normalize_loudness
from ml.utils import get_logger

log = get_logger(__name__)


class EncoderUnavailable(RuntimeError):
    pass


class AudioEncoder(ABC):
    name: ClassVar[str]

    @property
    @abstractmethod
    def version(self) -> str: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def encode_batch(self, signals: list[np.ndarray], sr: int) -> np.ndarray:
        """Return an array of shape (len(signals), dim)."""


class SpectralStatsEncoder(AudioEncoder):
    """Mean and standard deviation of log-mel energies plus their deltas over voiced frames.

    Not a learned representation; it gives the embedding branch a functioning lightweight alternative
    when pretrained checkpoints are unavailable, and serves as a sanity baseline for HuBERT.
    """

    name = "spectral_stats"

    def __init__(self, n_mels: int = 64, **_: Any) -> None:
        self.n_mels = n_mels

    @property
    def version(self) -> str:
        return f"spectral_stats-v1-mels{self.n_mels}"

    @property
    def dim(self) -> int:
        return self.n_mels * 4

    def encode_batch(self, signals: list[np.ndarray], sr: int) -> np.ndarray:
        import librosa

        out = []
        for sig in signals:
            sig = normalize_loudness(sig)
            mel = librosa.power_to_db(librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=512, hop_length=160, n_mels=self.n_mels))
            energy = mel.mean(axis=0)
            active = energy > np.percentile(energy, 95) - 30
            mel = mel[:, active] if active.sum() > 5 else mel
            delta = librosa.feature.delta(mel) if mel.shape[1] >= 9 else np.zeros_like(mel)
            out.append(np.concatenate([mel.mean(1), mel.std(1), delta.mean(1), delta.std(1)]))
        return np.asarray(out, dtype=np.float32)


class HFSpeechEncoder(AudioEncoder):
    """Self-supervised speech model encoder (HuBERT / wav2vec 2.0) with mean pooling."""

    name = "hf_speech"

    def __init__(self, checkpoint: str, revision: str = "main", layer: int = -1, device: str = "auto",
                 cache_dir: str | None = None, max_seconds: float = 30.0, **_: Any) -> None:
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:
            raise EncoderUnavailable("Pretrained encoders need torch + transformers: pip install -e .[deep]") from exc
        self.checkpoint, self.revision, self.layer = checkpoint, revision, layer
        self.max_seconds = max_seconds
        self.device = resolve_device(device)
        self._torch = torch
        try:
            self.extractor = AutoFeatureExtractor.from_pretrained(checkpoint, revision=revision, cache_dir=cache_dir, local_files_only=True)
            self.model = AutoModel.from_pretrained(checkpoint, revision=revision, cache_dir=cache_dir, local_files_only=True)
        except OSError as exc:
            raise EncoderUnavailable(
                f"Checkpoint '{checkpoint}' is not cached. Run: cognitrack download-models --embeddings"
            ) from exc
        self.model.eval().to(self.device)
        self._dim = int(self.model.config.hidden_size)

    @property
    def version(self) -> str:
        return f"{self.checkpoint}@{self.revision}-layer{self.layer}-meanpool-max{self.max_seconds:g}s"

    @property
    def dim(self) -> int:
        return self._dim

    def encode_batch(self, signals: list[np.ndarray], sr: int) -> np.ndarray:
        torch = self._torch
        max_len = int(self.max_seconds * sr)
        batch = [normalize_loudness(s[:max_len]) for s in signals]
        inputs = self.extractor(batch, sampling_rate=sr, return_tensors="pt", padding=True, return_attention_mask=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[self.layer]  # (B, T, D)
        # Map the sample-level attention mask to frame level so padding is excluded from pooling.
        if "attention_mask" in inputs:
            frame_mask = self.model._get_feature_vector_attention_mask(hidden.shape[1], inputs["attention_mask"])
        else:
            frame_mask = torch.ones(hidden.shape[:2], device=hidden.device)
        mask = frame_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return pooled.cpu().numpy().astype(np.float32)


ENCODERS: dict[str, type[AudioEncoder]] = {"spectral_stats": SpectralStatsEncoder, "hubert": HFSpeechEncoder, "wav2vec2": HFSpeechEncoder}


def build_encoder(name: str, emb_cfg: dict[str, Any], cache_dir: str | None = None) -> AudioEncoder:
    if name not in ENCODERS:
        raise KeyError(f"Unknown encoder '{name}'. Available: {sorted(ENCODERS)}")
    params = dict(emb_cfg["encoders"].get(name, {}))
    if ENCODERS[name] is HFSpeechEncoder:
        params.update(device=emb_cfg["device"], cache_dir=cache_dir, max_seconds=emb_cfg["max_seconds"])
    return ENCODERS[name](**params)
