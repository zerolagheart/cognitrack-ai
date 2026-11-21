"""Change detection primitives for short per-participant series.

* EWMA control chart (statistical process control) relative to a personal baseline.
* PELT change-point detection via ``ruptures``.
* Bayesian online change-point detection (Adams & MacKay, 2007) with a Normal model of known variance
  and conjugate Normal prior on the mean.

Clinical speech series typically contain 2–8 sessions, so all detectors have low statistical power;
results are reported as *statistical* signals for review, never as evidence of decline.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def ewma_chart(x: np.ndarray, mu0: float, sigma: float, lam: float = 0.3, L: float = 2.7) -> dict[str, list]:
    """EWMA statistic and time-varying control limits; NaNs carry the previous statistic forward."""
    z, upper, lower, out = [], [], [], []
    prev = mu0
    for i, xi in enumerate(x, start=1):
        cur = prev if np.isnan(xi) else lam * xi + (1 - lam) * prev
        width = L * sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * i)))
        z.append(float(cur))
        upper.append(float(mu0 + width))
        lower.append(float(mu0 - width))
        out.append(bool(not np.isnan(xi) and (cur > mu0 + width or cur < mu0 - width)))
        prev = cur
    return {"ewma": z, "ucl": upper, "lcl": lower, "out_of_control": out}


def pelt_changepoints(signal: np.ndarray, penalty: float = 3.0, min_size: int = 2) -> list[int]:
    """Indices where a new segment starts (excluding 0). ``signal`` is (n_sessions, n_features)."""
    import ruptures as rpt

    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        signal = signal[:, None]
    signal = np.where(np.isnan(signal), np.nanmean(signal, axis=0, keepdims=True), signal)
    signal = np.nan_to_num(signal)
    if len(signal) < 2 * min_size:
        return []
    algo = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(signal)
    bkps = algo.predict(pen=penalty * signal.shape[1])
    return [int(b) for b in bkps[:-1]]


def bocpd(x: np.ndarray, hazard: float = 0.1, prior_mean: float = 0.0, prior_var: float = 1.0, obs_var: float = 1.0) -> list[float]:
    """Posterior over where the *current* regime began, given all sessions.

    Returns ``p`` with ``p[t]`` = P(the most recent change point occurred at session t | x_1..x_T) for
    t >= 1, and ``p[0]`` = P(no change point in the series). (Under a constant hazard the filtering
    probability P(r_t = 0) always equals the hazard, so the end-of-series run-length posterior is the
    informative summary for short series.) Missing observations are skipped.
    """
    x = np.asarray(x, dtype=float)
    obs = [(t, v) for t, v in enumerate(x) if not np.isnan(v)]
    T = len(x)
    if len(obs) < 2:
        return [float("nan")] * T
    n = len(obs)
    R = np.zeros(n + 1)  # run-length distribution after each observation; R[r]
    R[0] = 1.0
    mu, var = np.array([prior_mean]), np.array([prior_var])
    for i, (_, v) in enumerate(obs):
        pred = norm.pdf(v, loc=mu, scale=np.sqrt(var + obs_var))
        joint = R[: i + 1] * pred
        new = np.zeros(n + 1)
        new[1: i + 2] = joint * (1 - hazard)
        new[0] = joint.sum() * hazard
        R = new / (new.sum() + 1e-300)
        post_var = 1 / (1 / var + 1 / obs_var)
        post_mu = post_var * (mu / var + v / obs_var)
        mu, var = np.append(prior_mean, post_mu), np.append(prior_var, post_var)
    # Run length r = number of observations in the current regime, so it began at observation n - r.
    # r = 0 (a change *after* the last session) carries no information about observed sessions: dropped.
    out = [0.0] * T
    total = float(R[1: n + 1].sum()) or 1.0
    for r in range(1, n + 1):
        start_obs = n - r
        out[0 if start_obs == 0 else obs[start_obs][0]] += float(R[r]) / total
    return out
