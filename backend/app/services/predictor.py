"""Process-wide cached predictor (reloaded after a publish)."""

from __future__ import annotations

import threading

from backend.app.core.errors import UnavailableError
from ml.config import get_config
from ml.inference import Predictor
from ml.training.registry import ModelNotFoundError

_lock = threading.Lock()
_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    with _lock:
        if _predictor is None:
            try:
                _predictor = Predictor(get_config())
            except ModelNotFoundError as exc:
                raise UnavailableError(str(exc)) from exc
        return _predictor


def try_get_predictor() -> Predictor | None:
    try:
        return get_predictor()
    except UnavailableError:
        return None


def reset_predictor() -> None:
    global _predictor
    with _lock:
        _predictor = None
