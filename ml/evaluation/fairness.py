"""Subgroup performance evaluation.

Metrics are computed at the participant level per subgroup of each available demographic attribute.
Subgroups with fewer participants than ``min_size`` (or lacking both classes) are reported with their
size only and marked ``insufficient_sample``: no metric is shown, and no fairness claim should be made.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.evaluation.metrics import binary_metrics

DEFAULT_ATTRIBUTES = ("sex", "age_band", "education_band", "language")


def subgroup_metrics(
    predictions: pd.DataFrame, attributes: tuple[str, ...] = DEFAULT_ATTRIBUTES, min_size: int = 20, threshold: float = 0.5,
) -> dict[str, Any]:
    """``predictions`` needs participant_id, y_true, probability and the attribute columns."""
    per_participant = predictions.groupby("participant_id").agg(
        y_true=("y_true", "first"), probability=("probability", "mean"),
        **{a: (a, "first") for a in attributes if a in predictions.columns},
    )
    overall = binary_metrics(per_participant["y_true"].to_numpy(), per_participant["probability"].to_numpy(), threshold)
    result: dict[str, Any] = {"overall": _summary(overall), "min_subgroup_size": min_size, "attributes": {}}
    for attr in attributes:
        if attr not in per_participant.columns or per_participant[attr].isna().all():
            continue
        groups = []
        for value, part in per_participant.groupby(per_participant[attr].fillna("missing")):
            y, p = part["y_true"].to_numpy(), part["probability"].to_numpy()
            entry: dict[str, Any] = {"group": str(value), "n_participants": int(len(part)), "prevalence": float(np.mean(y))}
            if len(part) < min_size or len(np.unique(y)) < 2:
                entry["status"] = "insufficient_sample"
            else:
                entry["status"] = "evaluated"
                entry.update(_summary(binary_metrics(y, p, threshold)))
            groups.append(entry)
        evaluated = [g for g in groups if g["status"] == "evaluated"]
        gap = None
        if len(evaluated) >= 2:
            gap = {
                "auroc_gap": float(max(g["auroc"] for g in evaluated) - min(g["auroc"] for g in evaluated)),
                "sensitivity_gap": float(max(g["sensitivity"] for g in evaluated) - min(g["sensitivity"] for g in evaluated)),
            }
        result["attributes"][attr] = {"groups": groups, "gaps": gap}
    return result


def _summary(m: dict[str, Any]) -> dict[str, Any]:
    return {k: m[k] for k in ("n", "auroc", "balanced_accuracy", "sensitivity", "specificity", "brier", "ece")}
