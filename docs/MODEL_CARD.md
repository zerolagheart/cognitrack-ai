# Model card — CogniTrack AI speech signal model

The quantitative sections (performance, partitions, subgroups, robustness) are **generated on every pipeline run** into [`docs/reports/model_card_generated.md`](reports/model_card_generated.md). The dashboard's Responsible AI page shows them too. This file holds the static parts.

## Model details

* **Type:** binary classifier. It outputs the calibrated probability that a recording shows the speech pattern associated with the dataset label `cognitive_impairment` rather than `healthy_control`.
* **Candidates:** logistic regression, random forest, XGBoost, LightGBM and a PyTorch MLP, combined by early, late (stacking) or attention fusion over acoustic, linguistic, conversational and audio-embedding features.
* **Selection:** the simplest configuration within 0.01 of the best participant-grouped CV AUROC on the training partition.
* **Calibration:** Platt scaling (small sets) or isotonic regression, fitted on validation participants.
* **Version identity:** experiment ID + dataset version + config hash + MLflow run ID.

## Intended use

* Research on speech markers of cognitive change.
* **Decision support** for qualified clinicians and researchers: flagging recordings or trajectories that may merit closer clinical assessment.
* Teaching and engineering demonstration with synthetic data.

## Prohibited use

* Diagnosis, staging or exclusion of cognitive impairment or dementia.
* Any automated decision about care, insurance, employment, legal capacity or eligibility.
* Use without human review, or on people who have not consented.
* Use on languages, tasks, recording set-ups or populations the model has not been evaluated on.
* Presenting results trained on synthetic data as clinical evidence.

## Training data

* **Default:** synthetic data from `ml/data/synthetic.py`, with group differences and trajectories programmed in. Performance on it shows only that the pipeline works.
* **Supported:** ADReSSo, ADReSS-M and DementiaBank through adapters, once the user has legitimate access. No real-corpus model ships with this repository.

## Evaluation data

* A participant-independent held-out test partition (20% of participants by default), stratified by label.
* Metrics are reported at recording and participant level with cluster-bootstrap 95% CIs. See [MODEL_EVALUATION.md](MODEL_EVALUATION.md).

## Limitations and known failure modes

* **Recording conditions.** Microphone, room, codec and noise shift acoustic features. Jitter, shimmer and HNR are especially sensitive, and the robustness report quantifies this.
* **Population confounds.** Language, dialect, education, age, hearing, depression, fatigue, medication and speech disorders produce similar markers.
* **ASR.** Whisper drops disfluencies and errs more for accented and older speech, which biases linguistic features. ASR and manual transcripts must not be mixed.
* **Task dependence.** Picture description differs from conversation, so features do not transfer across tasks without re-validation.
* **Missing modalities.** Without transcripts or speaker turns, those features are missing and the prediction relies on the remaining modalities; the missing-feature rate feeds abstention.
* **Small data.** Public corpora are small, so CIs are wide and subgroup analysis is usually underpowered.
* **Synthetic optimism.** Synthetic data is cleaner and more separable than real speech.

## Subgroup performance

Computed per available attribute (sex, age band, education band, language) at participant level. Groups below the minimum size (default 20 participants) are reported as `insufficient_sample` **without metrics**, and no fairness claim is made from them.

## Privacy considerations

* Pseudonymous IDs only; raw uploaded audio is deleted after processing by default.
* Model bundles contain a ≤ 100-row background sample of derived training features, so treat bundles trained on licensed data as derived data.
* See [DATA_GOVERNANCE.md](DATA_GOVERNANCE.md).

## Human oversight requirements

* Every output carries a disclaimer and must be reviewed by a qualified clinician.
* An abstention or "refer for review" result is triggered when any of these hold:
  * the probability falls in [0.35, 0.65];
  * the ensemble disagrees (SD > 0.2);
  * the input is out of distribution (Mahalanobis distance > training 99th percentile);
  * more than 30% of features are missing;
  * the audio fails quality checks.
* Thresholds are configurable in `configs/default.yaml` and must be re-validated before any clinical study.
