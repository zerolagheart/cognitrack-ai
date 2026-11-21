# ADR-0004: Baseline-relative longitudinal analysis scaled by population within-person variability

* Status: accepted
* Date: 2026-11-11

## Context

People differ widely in speech rate, pitch and vocabulary, so population thresholds flag stable outliers and miss meaningful personal change. A personal SD estimated from 2–3 baseline sessions, however, is too unstable to scale z-scores.

## Decision

Deviation = (session value − personal baseline mean) ÷ (pooled within-person SD × √(1 + 1/k)). The SD comes from the cohort's baseline-period sessions, and the √(1+1/k) factor accounts for uncertainty in a k-session baseline mean. The engine combines:

* z-score flags;
* EWMA control charts around the personal baseline;
* PELT on the multivariate z-series;
* a Bayesian change-point posterior over where the current regime began;
* cohort Isolation Forest scores on deviation vectors.

A session is *flagged* (for display) when |z| > 2 or EWMA leaves its control limits. A participant-level *statistical change* is declared only if:

* a feature deviates in the same direction in ≥ 2 post-baseline sessions; or
* any |z| exceeds a Bonferroni threshold over all feature × session tests; or
* a PELT change point coincides with a flagged session.

This corrects for the many tests per participant. On the synthetic cohort, the uncorrected rule flagged 58% of stable participants (specificity 0.42). With the corrected rule and the noise-adaptive VAD, the latest run reaches specificity 0.91 and sensitivity 0.59 against the programmed trajectories (`longitudinal.json`). Most misses are trajectories that change only in the final session, which a persistence rule cannot yet confirm. Results are labelled as statistical change, never impairment. With fewer than `min_sessions` sessions, the engine reports `insufficient_data`.

## Consequences

Flags mean "unusual change for a person relative to normal visit-to-visit variation". The scale depends on cohort composition. On synthetic data the engine is checked against programmed trajectories.

## Alternatives considered

Mixed-effects growth models (planned once more sessions per person exist); personal SD only (unstable); population z-scores (not personalised).
