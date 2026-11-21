"""Participant-independent data splitting and leakage checks.

All partitions are formed at the *participant* level: every recording of a participant lands in exactly
one of train / validation / test. Splits are stratified on the participant-level label where possible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


class LeakageError(AssertionError):
    pass


def participant_labels(df: pd.DataFrame, label_col: str) -> pd.Series:
    """One label per participant (first non-missing). Participants whose label changes are rejected."""
    labelled = df.dropna(subset=[label_col])
    n_unique = labelled.groupby("participant_id")[label_col].nunique()
    if (n_unique > 1).any():
        bad = n_unique[n_unique > 1].index.tolist()[:5]
        raise ValueError(f"Participants with conflicting labels cannot be stratified: {bad}")
    return labelled.groupby("participant_id")[label_col].first()


def split_participants(
    df: pd.DataFrame, label_col: str, test_size: float, val_size: float, seed: int
) -> pd.DataFrame:
    """Return ``df`` with a ``split`` column in {train, val, test}; unlabelled rows get ``unlabelled``.

    ``val_size`` is the fraction of the *remaining* (non-test) participants.
    """
    labels = participant_labels(df, label_col)
    pids = labels.index.to_numpy()
    y = labels.to_numpy()
    stratify = y if min(pd.Series(y).value_counts()) >= 2 else None
    trainval, test = train_test_split(pids, test_size=test_size, random_state=seed, stratify=stratify)
    tv_labels = labels.loc[trainval].to_numpy()
    stratify_tv = tv_labels if min(pd.Series(tv_labels).value_counts()) >= 2 else None
    train, val = train_test_split(trainval, test_size=val_size, random_state=seed, stratify=stratify_tv)
    assignment = {**dict.fromkeys(train, "train"), **dict.fromkeys(val, "val"), **dict.fromkeys(test, "test")}
    out = df.copy()
    out["split"] = out["participant_id"].map(assignment).fillna("unlabelled")
    assert_no_leakage(out)
    return out


def group_kfold_indices(groups: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified group K-fold (participant-grouped) indices for cross-validation.

    The number of folds is capped by the participant count of the rarer class so every training fold
    contains both classes; fewer than two participants in a class is an error.
    """
    per_class = pd.DataFrame({"g": groups, "y": y}).drop_duplicates("g")["y"].value_counts()
    if len(per_class) < 2 or per_class.min() < 2:
        raise ValueError(f"Cross-validation needs >= 2 participants per class; got {per_class.to_dict()}")
    n_splits = max(2, min(n_splits, int(per_class.min())))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(np.zeros(len(y)), y, groups))
    for tr, va in folds:
        if set(groups[tr]) & set(groups[va]):
            raise LeakageError("Cross-validation fold shares participants between train and validation.")
    return folds


def assert_no_leakage(df: pd.DataFrame, split_col: str = "split") -> None:
    """Raise if any participant appears in more than one of train/val/test."""
    modelling = df[df[split_col].isin(["train", "val", "test"])]
    per_participant = modelling.groupby("participant_id")[split_col].nunique()
    leaked = per_participant[per_participant > 1].index.tolist()
    if leaked:
        raise LeakageError(f"{len(leaked)} participants appear in multiple partitions, e.g. {leaked[:5]}")
    for col in ("recording_id", "audio_sha256"):
        if col in modelling.columns:
            vals = modelling.dropna(subset=[col]).groupby(col)[split_col].nunique()
            if (vals > 1).any():
                raise LeakageError(f"Identical {col} found in multiple partitions.")


def split_summary(df: pd.DataFrame, label_col: str) -> dict:
    out = {}
    for split, part in df.groupby("split"):
        out[split] = {
            "participants": int(part["participant_id"].nunique()),
            "recordings": int(len(part)),
            "labels": {k: int(v) for k, v in part.groupby("participant_id")[label_col].first().value_counts().items()},
        }
    return out
