import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from pandera.errors import SchemaErrors

from ml.data.schema import conform
from ml.data.splitting import LeakageError, assert_no_leakage, group_kfold_indices, split_participants
from ml.data.validation import DataValidationError, find_duplicates, validate_manifest


def _manifest(tmp_path, n_participants=10, sessions=3, label_flip=False):
    rows = []
    rng = np.random.default_rng(0)
    for p in range(n_participants):
        for s in range(sessions):
            path = tmp_path / f"p{p}s{s}.wav"
            sf.write(path, rng.standard_normal(1600).astype(np.float32) * 0.1, 16000)
            rows.append({
                "participant_id": f"T-P{p:02d}", "session_id": f"T-P{p:02d}-S{s}", "recording_id": f"T-P{p:02d}-S{s}-R1",
                "recording_path": str(path), "dataset_source": "test", "session_index": s, "transcript_source": "none",
                "assessment_label": "cognitive_impairment" if (p % 2 == 0) != (label_flip and s == 1) else "healthy_control",
                "consent_research": True, "is_synthetic": True, "audio_sha256": f"hash{p}-{s}", "provenance": json.dumps({}),
            })
    return conform(pd.DataFrame(rows))


def test_conform_fills_optional_columns_and_validates(tmp_path):
    df = _manifest(tmp_path)
    assert df["transcript"].isna().all()
    assert validate_manifest(df)["passed"]


def test_invalid_label_rejected(tmp_path):
    df = _manifest(tmp_path)
    df.loc[0, "assessment_label"] = "dementia"
    with pytest.raises(SchemaErrors):
        conform(df)


def test_duplicate_detection(tmp_path):
    df = _manifest(tmp_path)
    df.loc[1, "audio_sha256"] = df.loc[0, "audio_sha256"]  # same participant, same audio
    df.loc[5, "audio_sha256"] = df.loc[0, "audio_sha256"]  # another participant → cross-participant duplicate
    dups = find_duplicates(df)
    assert len(dups["duplicate_audio_groups"]) == 1
    assert dups["cross_participant_audio"]
    with pytest.raises(DataValidationError):
        validate_manifest(df)


def test_missing_audio_file_blocks(tmp_path):
    df = _manifest(tmp_path)
    df.loc[0, "recording_path"] = str(tmp_path / "nope.wav")
    report = validate_manifest(df, strict=False)
    assert report["missing_files"] == [df.loc[0, "recording_id"]]
    assert not report["passed"]


def test_participant_level_split_has_no_leakage(tmp_path):
    df = split_participants(_manifest(tmp_path, n_participants=20), "assessment_label", 0.2, 0.25, seed=1)
    per_pid = df.groupby("participant_id")["split"].nunique()
    assert (per_pid == 1).all()
    assert set(df["split"]) == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        assert df.loc[df.split == split, "assessment_label"].nunique() == 2


def test_leakage_detected(tmp_path):
    df = split_participants(_manifest(tmp_path), "assessment_label", 0.2, 0.25, seed=1)
    pid = df.loc[df.split == "train", "participant_id"].iloc[0]
    df.loc[df.index[(df.participant_id == pid)][0], "split"] = "test"
    with pytest.raises(LeakageError):
        assert_no_leakage(df)


def test_conflicting_participant_labels_rejected(tmp_path):
    with pytest.raises(ValueError, match="conflicting labels"):
        split_participants(_manifest(tmp_path, label_flip=True), "assessment_label", 0.2, 0.25, seed=1)


def test_group_kfold_is_participant_disjoint():
    groups = np.repeat(np.arange(12), 3)
    y = np.repeat(np.arange(12) % 2, 3)
    for tr, va in group_kfold_indices(groups, y, 4, seed=0):
        assert not set(groups[tr]) & set(groups[va])
