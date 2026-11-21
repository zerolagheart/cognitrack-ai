"""Longitudinal communication analysis relative to each participant's own baseline.

For each participant: aggregate recordings per session, establish a personal baseline from the first
``baseline_sessions`` sessions, and compute deviation z-scores, moving averages, trends, EWMA control
chart signals, multivariate change points (PELT, BOCPD), population-referenced anomaly scores
(Isolation Forest on baseline-relative deviation vectors) and session-to-session comparisons.

Scale for z-scores: personal SDs from 2–3 baseline sessions are unstable, so deviations are scaled by
the *population within-person SD* (session-to-session variability among participants' baseline
sessions). This yields "how unusual is this change for a person relative to normal visit-to-visit
variation", which is the quantity of interest — not a population threshold on raw values.

Interpretation guard-rail: a flagged deviation is a statistical change in a communication measure. It
can arise from illness, fatigue, mood, medication, recording conditions, interviewer or task changes,
and it does not establish cognitive impairment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.ensemble import IsolationForest

from ml.features.registry import label_for
from ml.longitudinal.changepoint import bocpd, ewma_chart, pelt_changepoints

INTERPRETATION_NOTE = (
    "Statistical change relative to this participant's own baseline. Not a clinical finding: changes in "
    "speech measures can reflect recording conditions, health, fatigue, mood, medication or task differences. "
    "Requires review by a qualified clinician."
)


def session_table(features: pd.DataFrame, tracked: list[str]) -> pd.DataFrame:
    """One row per session (mean over usable recordings in the session)."""
    usable = features[features["quality_usable"].fillna(False).astype(bool)]
    cols = [c for c in tracked if c in usable.columns]
    agg = usable.groupby(["participant_id", "session_id"]).agg(
        session_index=("session_index", "first"), session_timestamp=("session_timestamp", "first"),
        n_recordings=("recording_id", "count"), **{c: (c, "mean") for c in cols},
    ).reset_index()
    return agg.sort_values(["participant_id", "session_index"]).reset_index(drop=True)


def population_within_sd(sessions: pd.DataFrame, tracked: list[str], baseline_sessions: int) -> dict[str, float]:
    """Pooled within-participant SD over baseline-period sessions (falls back to all sessions)."""
    out = {}
    for f in tracked:
        if f not in sessions.columns:
            continue
        base = sessions.groupby("participant_id").head(max(baseline_sessions, 2))
        resid = base[f] - base.groupby("participant_id")[f].transform("mean")
        counts = base.groupby("participant_id")[f].transform("count")
        dof = (counts > 1).sum() - base.loc[counts > 1, "participant_id"].nunique()
        sd = float(np.sqrt(np.nansum(resid[counts > 1] ** 2) / dof)) if dof > 0 else np.nan
        if not np.isfinite(sd) or sd <= 0:
            sd = float(sessions[f].std())
        out[f] = sd if np.isfinite(sd) and sd > 0 else 1.0
    return out


def _missing_sessions(part: pd.DataFrame) -> dict[str, Any]:
    idx = part["session_index"].astype(int).tolist()
    expected = list(range(min(idx), max(idx) + 1)) if idx else []
    gaps = [i for i in expected if i not in idx]
    long_gaps = []
    ts = pd.to_datetime(part["session_timestamp"])
    if ts.notna().sum() >= 3:
        intervals = ts.diff().dt.days.dropna()
        median = intervals.median()
        long_gaps = [
            {"after_session_index": int(part["session_index"].iloc[pos]), "days": int(d)}
            for pos, d in enumerate(intervals.to_numpy()) if median > 0 and d > 2 * median
        ]
    return {"missing_session_indices": gaps, "long_intervals": long_gaps}


def analyse_participant(
    part: pd.DataFrame, tracked: list[str], within_sd: dict[str, float], cfg: dict[str, Any],
) -> dict[str, Any]:
    part = part.sort_values("session_index").reset_index(drop=True)
    n = len(part)
    feats = [f for f in tracked if f in part.columns and part[f].notna().any()]
    result: dict[str, Any] = {
        "participant_id": part["participant_id"].iloc[0], "n_sessions": int(n), "tracked_features": feats,
        "feature_labels": {f: label_for(f) for f in feats}, "missing": _missing_sessions(part),
        "interpretation": INTERPRETATION_NOTE,
    }
    sessions = [{
        "session_id": r["session_id"], "session_index": int(r["session_index"]),
        "session_timestamp": None if pd.isna(r["session_timestamp"]) else pd.Timestamp(r["session_timestamp"]).date().isoformat(),
        "n_recordings": int(r["n_recordings"]), "values": {f: (None if pd.isna(r[f]) else float(r[f])) for f in feats},
    } for _, r in part.iterrows()]
    result["sessions"] = sessions
    k = cfg["baseline_sessions"]
    if n < cfg["min_sessions"] or n <= k:
        result["status"] = "insufficient_data"
        result["status_detail"] = f"{n} usable session(s); at least {max(cfg['min_sessions'], k + 1)} needed for baseline-relative analysis."
        return result

    result["status"] = "analysed"
    baseline = {f: float(part[f].iloc[:k].mean()) for f in feats}
    # A new session differs from a k-session baseline mean with SD sigma*sqrt(1 + 1/k): the baseline mean is
    # itself estimated from few sessions, and ignoring that inflates z-scores and false flags.
    scale = {f: within_sd[f] * np.sqrt(1 + 1 / max(int(part[f].iloc[:k].notna().sum()), 1)) for f in feats}
    result["baseline"] = {"sessions": [s["session_id"] for s in sessions[:k]], "mean": baseline, "scale": scale,
                          "scale_type": "population_within_person_sd_x_sqrt(1+1/k)", "z_threshold": cfg["z_threshold"]}
    Z = np.column_stack([(part[f].to_numpy(dtype=float) - baseline[f]) / scale[f] for f in feats])
    ma = part[feats].rolling(cfg["moving_average_window"], min_periods=1).mean()
    years = (pd.to_datetime(part["session_timestamp"]) - pd.to_datetime(part["session_timestamp"]).iloc[0]).dt.days / 365.25
    t = years.to_numpy() if years.notna().all() else part["session_index"].to_numpy(dtype=float)

    trends, ewma = {}, {}
    for f in feats:
        y = part[f].to_numpy(dtype=float)
        ok = ~np.isnan(y)
        if ok.sum() >= 3 and np.ptp(t[ok]) > 0:
            lr = stats.linregress(t[ok], y[ok])
            tau = stats.kendalltau(t[ok], y[ok])
            trends[f] = {"slope_per_year" if years.notna().all() else "slope_per_session": float(lr.slope),
                         "p_value": float(lr.pvalue), "kendall_tau": float(tau.statistic),
                         "slope_in_within_sd_units": float(lr.slope / within_sd[f])}
        ewma[f] = ewma_chart(y, baseline[f], scale[f], cfg["ewma_lambda"], cfg["ewma_l"])
    result["trends"] = trends
    result["ewma"] = ewma

    composite = np.nanmean(np.abs(Z), axis=1)
    for i, s in enumerate(sessions):
        s["z"] = {f: (None if np.isnan(Z[i, j]) else float(Z[i, j])) for j, f in enumerate(feats)}
        s["moving_average"] = {f: (None if pd.isna(ma[f].iloc[i]) else float(ma[f].iloc[i])) for f in feats}
        s["is_baseline"] = i < k
        s["flagged_features"] = [] if i < k else [f for j, f in enumerate(feats) if not np.isnan(Z[i, j]) and abs(Z[i, j]) > cfg["z_threshold"]]
        s["ewma_out_of_control"] = [] if i < k else [f for f in feats if ewma[f]["out_of_control"][i]]
        s["composite_deviation"] = None if np.isnan(composite[i]) else float(composite[i])
        if i > 0:
            prev = sessions[i - 1]["values"]
            s["delta_from_previous"] = {
                f: (None if s["values"][f] is None or prev[f] is None else float((s["values"][f] - prev[f]) / (within_sd[f] * np.sqrt(2))))
                for f in feats
            }

    cps = pelt_changepoints(np.nan_to_num(Z), penalty=cfg["changepoint_penalty"])
    cp_prob = bocpd(np.nan_to_num(composite - np.nanmean(composite[:k])), hazard=cfg["bocpd_hazard"])
    result["change_points"] = {
        "pelt_session_indices": [sessions[c]["session_index"] for c in cps if c < n],
        "bocpd_probability": cp_prob,
        "power_note": "Few sessions per participant: change-point methods have low power and high variance.",
    }
    flagged = [s for s in sessions if s["flagged_features"] or s["ewma_out_of_control"]]
    # Participant-level decision corrects for multiplicity (features x sessions): a single |z| > 2 among many
    # tests is expected by chance. Change is reported only if it persists, is extreme, or co-occurs with a change point.
    post = Z[k:]
    m = max(int(np.isfinite(post).sum()), 1)
    z_strong = float(norm.ppf(1 - 0.05 / (2 * m)))
    persistent = [f for j, f in enumerate(feats)
                  if max(np.sum(post[:, j] > cfg["z_threshold"]), np.sum(post[:, j] < -cfg["z_threshold"])) >= 2]
    strong = [f for j, f in enumerate(feats) if np.nanmax(np.abs(post[:, j]), initial=0) > z_strong]
    cp_supported = bool(cps) and bool(flagged)
    result["summary"] = {
        "n_flagged_sessions": len(flagged),
        "first_flagged_session_index": flagged[0]["session_index"] if flagged else None,
        "features_with_flags": sorted({f for s in flagged for f in s["flagged_features"] + s["ewma_out_of_control"]}),
        "persistent_features": persistent,
        "extreme_features": strong,
        "max_composite_deviation": float(np.nanmax(composite[k:])) if n > k else None,
        "statistical_change_detected": bool(persistent or strong or cp_supported),
        "decision_rule": (f"same-direction |z| > {cfg['z_threshold']} in >= 2 post-baseline sessions, or any |z| > {z_strong:.2f} "
                          f"(Bonferroni over {m} tests), or a PELT change point with a flagged session"),
    }
    return result


def analyse_cohort(features: pd.DataFrame, cfg: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    tracked = [f for f in cfg["tracked_features"] if f in features.columns]
    sessions = session_table(features, tracked)
    within_sd = population_within_sd(sessions, tracked, cfg["baseline_sessions"])
    participants = [analyse_participant(p, tracked, within_sd, cfg) for _, p in sessions.groupby("participant_id")]

    # Isolation Forest over baseline-relative deviation vectors of all post-baseline sessions.
    rows, keys = [], []
    for res in participants:
        if res["status"] != "analysed":
            continue
        for s in res["sessions"]:
            if not s["is_baseline"]:
                rows.append([s["z"].get(f) if s["z"].get(f) is not None else 0.0 for f in tracked])
                keys.append((res["participant_id"], s["session_id"]))
    if len(rows) >= 20:
        iso = IsolationForest(contamination=cfg["isolation_forest_contamination"], random_state=seed).fit(np.array(rows))
        scores = -iso.score_samples(np.array(rows))
        flags = iso.predict(np.array(rows)) == -1
        lookup = {k: (float(sc), bool(fl)) for k, sc, fl in zip(keys, scores, flags, strict=True)}
        for res in participants:
            for s in res["sessions"]:
                if (res["participant_id"], s["session_id"]) in lookup:
                    s["anomaly_score"], s["is_anomaly"] = lookup[(res["participant_id"], s["session_id"])]
    analysed = [p for p in participants if p["status"] == "analysed"]
    return {
        "config": {k: cfg[k] for k in cfg if k != "tracked_features"}, "tracked_features": tracked,
        "population_within_sd": within_sd,
        "cohort_summary": {
            "n_participants": len(participants), "n_analysed": len(analysed),
            "n_insufficient_data": len(participants) - len(analysed),
            "n_with_statistical_change": sum(p["summary"]["statistical_change_detected"] for p in analysed),
        },
        "participants": participants, "interpretation": INTERPRETATION_NOTE,
    }
