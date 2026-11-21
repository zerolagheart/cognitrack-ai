"""Shared helpers: structured logging, seeding, hashing and serialisation."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

_REDACT_PATTERNS = [
    # Emails and long digit sequences (phone numbers, MRNs) must never reach logs.
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<redacted-email>"),
    (re.compile(r"\b\d{7,}\b"), "<redacted-number>"),
]


def redact(text: str) -> str:
    for pattern, replacement in _REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class JsonFormatter(logging.Formatter):
    """One JSON object per line; extra fields passed via ``extra={"ctx": {...}}`` are merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        ctx = getattr(record, "ctx", None)
        if isinstance(ctx, dict):
            payload.update({k: v for k, v in ctx.items() if k not in payload})
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    root = logging.getLogger()
    if getattr(root, "_cognitrack_configured", False):
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_logs else logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.handlers = [handler]
    root.setLevel(level)
    for noisy in ("numba", "urllib3", "matplotlib", "httpx", "git", "alembic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    root._cognitrack_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_fingerprint(df: Any) -> str:
    """Stable content hash of a DataFrame, used as a dataset version reference."""
    import pandas as pd

    hashed = pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).values
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def to_jsonable(obj: Any) -> Any:
    """Round-trip through JSON, converting NaN to None so output is strict JSON."""
    return json.loads(json.dumps(sanitize_nan(obj), default=_json_default))


def sanitize_nan(obj: Any) -> Any:
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, np.floating) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [sanitize_nan(v) for v in obj]
    return obj


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=2), encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relpath(path: str | Path) -> str:
    """Path relative to the project root (POSIX style) for portable manifests."""
    from ml.config import PROJECT_ROOT

    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()
