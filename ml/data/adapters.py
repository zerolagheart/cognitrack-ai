"""Dataset adapters: convert dataset-native layouts into the canonical manifest.

Adding a dataset means writing one adapter class and registering it in :data:`ADAPTERS`; nothing else in
the codebase knows about dataset-specific layouts.

Access notes (see docs/DATA_GOVERNANCE.md):

* **ADReSSo / ADReSS-M / DementiaBank** are distributed through TalkBank/DementiaBank and require
  membership approval and agreement to their terms. They are *never* downloaded automatically and must
  never be committed. The adapters below follow the published directory layouts; because those corpora
  were not available while this code was written, the default glob patterns are configurable and the
  adapters fail with an explanatory message instead of guessing when files are missing.
* **synthetic** is produced locally by ``cognitrack generate-synthetic``.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from ml.config import resolve_path
from ml.data.chat_parser import parse_chat
from ml.data.schema import LABEL_CONTROL, LABEL_IMPAIRED, conform
from ml.utils import get_logger, relpath, write_json

log = get_logger(__name__)

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac")


class DatasetUnavailableError(RuntimeError):
    """Raised when a dataset's files are not present locally (the rest of the system keeps working)."""


class DatasetAdapter(ABC):
    name: ClassVar[str]
    licence_note: ClassVar[str]
    version: ClassVar[str] = "1.0"

    def __init__(self, root: str | Path, options: dict[str, Any] | None = None) -> None:
        self.root = resolve_path(root)
        self.options = options or {}

    def available(self) -> bool:
        return self.root.exists() and any(self.root.rglob("*"))

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "root": relpath(self.root), "available": self.available(), "licence": self.licence_note}

    def load(self) -> pd.DataFrame:
        if not self.available():
            raise DatasetUnavailableError(
                f"Dataset '{self.name}' not found at {relpath(self.root)}. {self.licence_note} "
                "Place the files there (see docs/DATA_GOVERNANCE.md) or disable the dataset in the config."
            )
        df = self._load()
        if df.empty:
            raise DatasetUnavailableError(f"Dataset '{self.name}' at {relpath(self.root)} contained no usable recordings.")
        return conform(df)

    @abstractmethod
    def _load(self) -> pd.DataFrame: ...

    def _provenance(self, **extra: Any) -> str:
        return json.dumps({"adapter": self.name, "adapter_version": self.version, "licence": self.licence_note, **extra})


class SyntheticAdapter(DatasetAdapter):
    name = "synthetic"
    licence_note = "Synthetic demonstration data generated locally; not clinical evidence."

    def available(self) -> bool:
        return (self.root / "metadata.csv").exists()

    def _load(self) -> pd.DataFrame:
        meta = pd.read_csv(self.root / "metadata.csv")
        turns_exist = meta["turns_file"].map(lambda p: resolve_path(p).exists())
        return pd.DataFrame({
            "participant_id": meta["participant_id"],
            "session_id": meta["session_id"],
            "recording_id": meta["recording_id"],
            "recording_path": meta["audio_file"],
            "dataset_source": self.name,
            "dataset_split_hint": None,
            "session_index": meta["session_index"],
            "session_timestamp": meta["session_date"],
            "task": meta["task"],
            "language": "en",
            "transcript": meta["transcript"],
            "transcript_source": "dataset",
            "turns_path": meta["turns_file"].where(turns_exist, None),
            "assessment_label": meta["group"],
            "assessment_score": meta["synthetic_score"],
            "assessment_instrument": "synthetic_score (not a clinical instrument)",
            "age_band": meta["age_band"],
            "sex": meta["sex"],
            "education_band": meta["education_band"],
            "consent_research": True,
            "is_synthetic": True,
            "provenance": [
                self._provenance(source_file=f, quality_issue=q if isinstance(q, str) else None)
                for f, q in zip(meta["audio_file"], meta["quality_issue"], strict=True)
            ],
        })


def _turns_from_segmentation_csv(seg_csv: Path, out_dir: Path, rec_id: str) -> str | None:
    """ADReSSo segmentation CSVs list speaker (PAR/INV) and begin/end in milliseconds."""
    if not seg_csv.exists():
        return None
    seg = pd.read_csv(seg_csv)
    cols = {c.lower(): c for c in seg.columns}
    if not {"speaker", "begin", "end"} <= set(cols):
        log.warning("segmentation file lacks speaker/begin/end", extra={"ctx": {"file": relpath(seg_csv)}})
        return None
    turns = [
        {"speaker": str(r[cols["speaker"]]), "start": float(r[cols["begin"]]) / 1000, "end": float(r[cols["end"]]) / 1000, "text": None}
        for _, r in seg.iterrows()
    ]
    return relpath(write_json(out_dir / f"{rec_id}.json", turns))


class ADReSSoAdapter(DatasetAdapter):
    """ADReSSo 2021 diagnosis task (speech-only; English Cookie Theft descriptions).

    Expected layout (configurable via ``options``)::

        <root>/diagnosis/train/audio/{ad,cn}/*.wav
        <root>/diagnosis/train/segmentation/{ad,cn}/*.csv
        <root>/diagnosis/train/adresso-train-mmse-scores.csv   (adressfname, mmse, dx)
        <root>/diagnosis/test-dist/audio/*.wav                  (labels only if released test labels are supplied)

    ADReSSo provides no transcripts (``transcript_source = none``) and is cross-sectional: each
    recording is treated as its own participant.
    """

    name = "adresso"
    licence_note = "ADReSSo requires DementiaBank membership and agreement to TalkBank rules."

    def _load(self) -> pd.DataFrame:
        diag = self.root / self.options.get("subdir", "diagnosis")
        scores_path = diag / "train" / self.options.get("scores_file", "adresso-train-mmse-scores.csv")
        scores = pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame(columns=["adressfname", "mmse", "dx"])
        scores = scores.set_index("adressfname") if "adressfname" in scores.columns else scores
        test_labels_path = self.options.get("test_labels_file")
        test_labels = pd.read_csv(resolve_path(test_labels_path)).set_index("ID") if test_labels_path else None
        turns_dir = resolve_path("data/interim/turns") / self.name
        rows = []
        for split in ("train", "test-dist"):
            for wav in sorted((diag / split / "audio").rglob("*")):
                if wav.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                stem = wav.stem
                label, score = None, None
                if split == "train":
                    folder = wav.parent.name.lower()
                    label = {"ad": LABEL_IMPAIRED, "cn": LABEL_CONTROL}.get(folder)
                    if stem in scores.index:
                        score = float(scores.loc[stem, "mmse"]) if pd.notna(scores.loc[stem, "mmse"]) else None
                elif test_labels is not None and stem in test_labels.index:
                    dx = str(test_labels.loc[stem].iloc[0]).lower()
                    label = {"ad": LABEL_IMPAIRED, "cn": LABEL_CONTROL}.get(dx)
                seg_csv = diag / split / "segmentation" / wav.parent.relative_to(diag / split / "audio") / f"{stem}.csv"
                rows.append({
                    "participant_id": f"ADRSO-{stem}",
                    "session_id": f"ADRSO-{stem}-S01",
                    "recording_id": f"ADRSO-{stem}",
                    "recording_path": relpath(wav),
                    "dataset_source": self.name,
                    "dataset_split_hint": split,
                    "session_index": 0,
                    "task": "picture_description",
                    "language": "en",
                    "transcript_source": "none",
                    "turns_path": _turns_from_segmentation_csv(seg_csv, turns_dir, f"ADRSO-{stem}"),
                    "assessment_label": label,
                    "assessment_score": score,
                    "assessment_instrument": "MMSE" if score is not None else None,
                    "consent_research": True,
                    "is_synthetic": False,
                    "provenance": self._provenance(source_file=relpath(wav)),
                })
        return pd.DataFrame(rows)


class ADReSSMAdapter(DatasetAdapter):
    """ADReSS-M 2023 (multilingual: English training audio, Greek evaluation audio).

    Expected layout::

        <root>/train/*.mp3   + <root>/training-groundtruth.csv (addressfname, age, gender, educ, dx, mmse)
        <root>/sample-gr/*.mp3 (Greek sample) and <root>/test-gr/*.mp3 (labels only if supplied)
    """

    name = "adress_m"
    licence_note = "ADReSS-M requires DementiaBank membership and agreement to TalkBank rules."

    def _load(self) -> pd.DataFrame:
        gt_path = self.root / self.options.get("groundtruth_file", "training-groundtruth.csv")
        gt = pd.read_csv(gt_path) if gt_path.exists() else pd.DataFrame()
        key = next((c for c in gt.columns if c.lower() in {"addressfname", "adressfname", "id"}), None)
        gt = gt.set_index(key) if key else gt
        rows = []
        for split, language in (("train", "en"), ("sample-gr", "el"), ("test-gr", "el")):
            for audio in sorted((self.root / split).glob("*")):
                if audio.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                stem = audio.stem
                rec = gt.loc[stem] if key and stem in gt.index else None
                dx = str(rec.get("dx", "")).lower() if rec is not None else ""
                label = {"ad": LABEL_IMPAIRED, "cn": LABEL_CONTROL, "control": LABEL_CONTROL}.get(dx)
                mmse = rec.get("mmse") if rec is not None else None
                age = rec.get("age") if rec is not None else None
                rows.append({
                    "participant_id": f"ADRSM-{stem}",
                    "session_id": f"ADRSM-{stem}-S01",
                    "recording_id": f"ADRSM-{stem}",
                    "recording_path": relpath(audio),
                    "dataset_source": self.name,
                    "dataset_split_hint": split,
                    "session_index": 0,
                    "task": "picture_description",
                    "language": language,
                    "transcript_source": "none",
                    "assessment_label": label,
                    "assessment_score": float(mmse) if mmse is not None and pd.notna(mmse) else None,
                    "assessment_instrument": "MMSE" if mmse is not None and pd.notna(mmse) else None,
                    "age_band": _age_band(age),
                    "sex": str(rec.get("gender")) if rec is not None and pd.notna(rec.get("gender")) else None,
                    "consent_research": True,
                    "is_synthetic": False,
                    "provenance": self._provenance(source_file=relpath(audio)),
                })
        return pd.DataFrame(rows)


class DementiaBankPittAdapter(DatasetAdapter):
    """DementiaBank Pitt corpus (CHAT transcripts + separately distributed media).

    Expected layout::

        <root>/Pitt/{Control,Dementia}/cookie/<participant>-<visit>.cha
        <root>/Pitt-media/{Control,Dementia}/cookie/<participant>-<visit>.mp3   (path configurable)

    File names encode participant and visit number, which makes Pitt the only listed corpus with
    repeated sessions per participant — it is what the longitudinal engine is designed for.
    """

    name = "dementiabank"
    licence_note = "DementiaBank requires membership approval and agreement to TalkBank rules."
    _STEM = re.compile(r"^(\d+)-(\d+)")

    def _load(self) -> pd.DataFrame:
        transcripts_root = self.root / self.options.get("transcripts_subdir", "Pitt")
        media_root = self.root / self.options.get("media_subdir", "Pitt-media")
        turns_dir = resolve_path("data/interim/turns") / self.name
        task = self.options.get("task", "cookie")
        rows = []
        for group_dir, label in (("Control", LABEL_CONTROL), ("Dementia", LABEL_IMPAIRED)):
            for cha in sorted((transcripts_root / group_dir / task).glob("*.cha")):
                match = self._STEM.match(cha.stem)
                if not match:
                    continue
                pid, visit = match.group(1), int(match.group(2))
                audio = next((media_root / group_dir / task / f"{cha.stem}{ext}" for ext in AUDIO_EXTENSIONS
                              if (media_root / group_dir / task / f"{cha.stem}{ext}").exists()), None)
                if audio is None:
                    log.warning("no media for transcript; skipped", extra={"ctx": {"transcript": relpath(cha)}})
                    continue
                parsed = parse_chat(cha)
                par = parsed.participants.get("PAR", {})
                turns = parsed.turns()
                rec_id = f"DBPITT-{pid}-V{visit}"
                rows.append({
                    "participant_id": f"DBPITT-{pid}",
                    "session_id": f"DBPITT-{pid}-V{visit}",
                    "recording_id": rec_id,
                    "recording_path": relpath(audio),
                    "dataset_source": self.name,
                    "session_index": visit,
                    "task": "picture_description",
                    "language": "en",
                    "transcript": parsed.speaker_text("PAR") or None,
                    "transcript_source": "dataset",
                    "turns_path": relpath(write_json(turns_dir / f"{rec_id}.json", turns)) if turns else None,
                    "assessment_label": label,
                    "age_band": _age_band(par.get("age", "").split(";")[0] or None),
                    "sex": {"male": "M", "female": "F"}.get(par.get("sex", "").lower()),
                    "consent_research": True,
                    "is_synthetic": False,
                    "provenance": self._provenance(source_file=relpath(cha), media=relpath(audio), chat_group=par.get("group")),
                })
        return pd.DataFrame(rows)


def _age_band(age: Any) -> str | None:
    try:
        value = int(float(age))
    except (TypeError, ValueError):
        return None
    lo = (value // 10) * 10
    return f"{lo}-{lo + 9}"


ADAPTERS: dict[str, type[DatasetAdapter]] = {
    cls.name: cls for cls in (SyntheticAdapter, ADReSSoAdapter, ADReSSMAdapter, DementiaBankPittAdapter)
}


def get_adapter(dataset_key: str, datasets_cfg: dict[str, Any]) -> DatasetAdapter:
    if dataset_key not in datasets_cfg:
        raise KeyError(f"Dataset '{dataset_key}' is not configured. Known: {sorted(datasets_cfg)}")
    spec = dict(datasets_cfg[dataset_key])
    adapter_cls = ADAPTERS[spec.pop("adapter")]
    return adapter_cls(spec.pop("root"), spec)
