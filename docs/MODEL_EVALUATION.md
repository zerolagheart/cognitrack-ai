# Model evaluation methodology

This document defines *how* models are evaluated. It deliberately contains no headline numbers: results depend on the data in use and are regenerated each run into [`reports/model_comparison.md`](reports/model_comparison.md) and `artifacts/reports/model_comparison.json`. With the default synthetic data those numbers reflect programmed effects, not clinical validity.

## 1. Partitions

* Participants, not recordings, are split into train / validation / test (default 64 / 16 / 20%), stratified by participant label (`ml/data/splitting.py`).
* `assert_no_leakage` fails the pipeline if any participant, recording ID or audio hash appears in two partitions. Tests cover the check.
* Recordings that fail audio quality checks are excluded from modelling and counted.
* Rows without a label are excluded. Labels are never imputed.

## 2. Protocol per experiment

1. **Model selection evidence.** Stratified group K-fold on the training partition (participant-grouped; folds capped so every training fold contains both classes). Out-of-fold predictions yield the CV metrics.
2. **Fit** on the full training partition. All preprocessing (median imputation, scaling, PCA for embeddings) lives inside the pipeline.
3. **Calibrate** on the validation partition: Platt (sigmoid) below 300 recordings, isotonic otherwise.
4. **Test once** on the test partition.

Late fusion trains its meta-learner on out-of-fold base predictions, with an inner participant-grouped CV. Attention fusion early-stops on a participant-grouped internal split.

## 3. Metrics

* **Recording level, threshold 0.5:** accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1, confusion matrix.
* **Threshold-free:** AUROC, AUPRC (compare against prevalence).
* **Probabilistic:** Brier score, expected calibration error (10 equal-width bins), reliability curve. Uncalibrated Brier and ECE are reported alongside so the effect of calibration is visible.
* **Participant level:** mean recording probability per participant, the same metrics.
* **Class imbalance:** handled with class-weighted losses (`class_weight="balanced"`, `scale_pos_weight`, `pos_weight`); balanced accuracy and AUPRC are reported.

## 4. Uncertainty in the estimates

* **Cluster bootstrap:** participants are resampled with replacement (default 300 resamples); 95% percentile intervals. Intervals backed by fewer than 20 valid resamples are not reported.
* **Fusion vs best single modality:** paired cluster bootstrap of ΔAUROC on the same test participants. Fusion is only described as an improvement if the interval excludes zero.

## 5. Model selection rule

Among experiments within `SIMPLICITY_MARGIN = 0.01` of the best CV AUROC, choose the lowest complexity, ordered by: number of modalities, then strategy (single < early < late < attention), then model (logreg < RF < LightGBM < XGBoost < MLP). Test metrics never influence selection.

## 6. Ablation study

The modality sets are acoustic, linguistic, conversational, embedding, acoustic+linguistic, acoustic+embedding and full. Each is crossed with all models (early fusion), late fusion (logreg and LightGBM bases) and attention fusion. The report shows the best experiment per modality set by CV AUROC, with test AUROC and CI.

## 7. Explainability, uncertainty, fairness, robustness

| Analysis | Method | Output |
|---|---|---|
| Global importance | TreeSHAP / LinearSHAP on test set; mean \|SHAP\| plus direction (corr of value vs SHAP) | explainability.json |
| Interactions | SHAP interaction values (boosting models, ≤ 100 rows) | explainability.json |
| Modality contribution | Grouped permutation importance (AUROC drop, 10 repeats) and SHAP modality share | explainability.json |
| Selective prediction | Abstention rate; metrics on all vs non-abstained recordings | uncertainty.json |
| OOD | Ledoit-Wolf Mahalanobis on handcrafted features; 99th-percentile training threshold | uncertainty.json |
| Subgroups | Participant-level metrics per attribute; minimum 20 participants per group | fairness.json |
| Robustness | Noise at 20/10/5 dB SNR, gain −12/+6 dB, 8 kHz band-limit, 50% truncation, transcript removed, speaker turns removed; features re-extracted through production code | robustness.json |

## 8. Longitudinal engine validation

On synthetic data the generator's programmed trajectories are known. `longitudinal.json → synthetic_engine_validation` reports the sensitivity and specificity of "any statistical change detected" against "programmed change". This checks the engine's mechanics only. Real-world validation would need longitudinal cohorts with independent clinical follow-up.

## 9. What would be required for clinical validation (not done here)

* External validation on independent sites, devices and languages, with a pre-registered analysis plan.
* Prospective evaluation against a clinical reference standard.
* Calibration and decision-curve analysis at clinically motivated thresholds.
* Adequately powered subgroup analyses.
* Regulatory pathway assessment (e.g. as software as a medical device).
