import numpy as np
import pandas as pd
import pytest

from ml.config import load_config
from ml.evaluation.fairness import subgroup_metrics
from ml.evaluation.metrics import binary_metrics, cluster_bootstrap_ci, expected_calibration_error
from ml.longitudinal.changepoint import bocpd, ewma_chart, pelt_changepoints
from ml.longitudinal.engine import analyse_cohort
from ml.monitoring.drift import feature_drift, psi


def test_binary_metrics_known_values():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])
    m = binary_metrics(y, p)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.5
    assert m["auroc"] == pytest.approx(0.75)


def test_ece_zero_for_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, np.clip(p + 0.3, 0, 1)) > 0.15


def test_cluster_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(40), 3)
    y = np.repeat(rng.integers(0, 2, 40), 3)
    p = np.clip(y * 0.4 + rng.uniform(0, 0.6, y.size), 0, 1)
    ci = cluster_bootstrap_ci(y, p, groups, n_boot=200)
    point = binary_metrics(y, p)["auroc"]
    assert ci["auroc"]["low"] <= point <= ci["auroc"]["high"]


def test_subgroups_below_minimum_are_not_reported():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"participant_id": np.arange(60), "y_true": rng.integers(0, 2, 60), "probability": rng.uniform(size=60),
                       "sex": ["F"] * 50 + ["M"] * 10})
    out = subgroup_metrics(df, attributes=("sex",), min_size=20)
    groups = {g["group"]: g for g in out["attributes"]["sex"]["groups"]}
    assert groups["F"]["status"] == "evaluated" and "auroc" in groups["F"]
    assert groups["M"]["status"] == "insufficient_sample" and "auroc" not in groups["M"]
    assert out["attributes"]["sex"]["gaps"] is None


def test_change_detection_primitives():
    x = np.array([0.0, 0.1, -0.1, 0.05, 3.0, 3.1, 2.9, 3.05])
    assert pelt_changepoints(x[:, None], penalty=1.0) == [4]
    probs = bocpd(x, hazard=0.2)
    assert int(np.argmax(probs)) == 4
    chart = ewma_chart(x, mu0=0.0, sigma=0.3)
    assert not any(chart["out_of_control"][:4]) and all(chart["out_of_control"][4:])


def _sessions(pid, values, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(values), freq="90D")
    return [{"participant_id": pid, "session_id": f"{pid}-S{i}", "recording_id": f"{pid}-S{i}-R", "session_index": i,
             "session_timestamp": d, "quality_usable": True, "speech_rate_wpm": v} for i, (v, d) in enumerate(zip(values, dates, strict=True))]


def test_longitudinal_flags_change_relative_to_own_baseline():
    cfg = dict(load_config()["longitudinal"], tracked_features=["speech_rate_wpm"])
    rng = np.random.default_rng(3)
    rows = []
    for k in range(15):  # stable participants with different personal levels
        base = rng.normal(130, 20)
        rows += _sessions(f"S{k}", list(base + rng.normal(0, 3, 5)))
    rows += _sessions("CHANGE", [150, 151, 149, 120, 110])  # high personal baseline, then a drop
    rows += _sessions("SHORT", [100, 101])  # insufficient data
    report = analyse_cohort(pd.DataFrame(rows), cfg)
    by_id = {p["participant_id"]: p for p in report["participants"]}
    assert by_id["CHANGE"]["summary"]["statistical_change_detected"]
    assert by_id["CHANGE"]["sessions"][4]["flagged_features"] == ["speech_rate_wpm"]
    assert by_id["SHORT"]["status"] == "insufficient_data"
    stable_flag_rate = np.mean([by_id[f"S{k}"]["summary"]["n_flagged_sessions"] > 0 for k in range(15)])
    assert stable_flag_rate < 0.5


def test_drift_detects_shift():
    rng = np.random.default_rng(4)
    ref = pd.DataFrame({"pause_rate_per_min": rng.normal(10, 2, 300), "f0_mean_hz": rng.normal(150, 20, 300)})
    cur = pd.DataFrame({"pause_rate_per_min": rng.normal(14, 2, 200), "f0_mean_hz": rng.normal(150, 20, 200)})
    out = feature_drift(ref, cur, list(ref.columns), load_config()["monitoring"])
    flags = {f["feature"]: f["drifted"] for f in out["features"]}
    assert flags["pause_rate_per_min"] and not flags["f0_mean_hz"]
    assert psi(ref["f0_mean_hz"].to_numpy(), ref["f0_mean_hz"].to_numpy()) == pytest.approx(0.0, abs=1e-9)
