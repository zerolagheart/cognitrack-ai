# Data governance

## 1. Dataset access and licensing

| Dataset | Access | Notes |
|---|---|---|
| ADReSSo (2021) | DementiaBank/TalkBank membership; agree to TalkBank Ground Rules | Speech-only, cross-sectional Cookie Theft descriptions with segmentation files; MMSE for training subjects |
| ADReSS-M (2023) | Same as above | English training audio, Greek evaluation audio |
| DementiaBank (Pitt) | Membership approval; media distributed separately from transcripts | CHAT transcripts with time bullets; repeated visits per participant |
| Synthetic | Generated locally | Not clinical data; always flagged `is_synthetic` |

Rules:

* **No automatic download** of any restricted corpus. Users obtain access themselves and place files under `data/raw/<Dataset>/`, which is git-ignored.
* Never commit recordings, transcripts, derived per-participant features, model checkpoints or credentials. `.gitignore` enforces this, and DVC tracks hashes only.
* Adapters record provenance (source file, adapter version, licence note) on every manifest row.
* Respect each dataset's terms on publication, redistribution and derived works. The derived feature tables of a licensed corpus count as derived data.

## 2. Pseudonymisation and separation of identity

* Research tables use dataset-prefixed pseudonyms only.
* The upload API rejects identifiers that look like names or emails: allowed characters are `[A-Za-z0-9_-]`, 3–64 long, and the UI asks for confirmation that the pseudonym is not identifying.
* `identity_links` (salted hash of an external reference) is a separate table, never exposed via the API, and empty for public and synthetic data. A production deployment should move it to a separate database with separate credentials.
* Demographics are stored only as coarse bands (age decade, education band).

## 3. Consent-aware handling

* Uploads require `consent_confirmed=true`. Participants without recorded consent cannot receive new recordings.
* Ingested datasets are covered by their collection consent and access agreement; `consent_research` records this per row.
* Participant-level endpoints require the `researcher` role, and each access writes an `audit_events` row (role, action, resource, pseudonym) with no request bodies.

## 4. Minimisation and retention

| Data | Default retention |
|---|---|
| Raw uploaded audio | **Deleted after feature extraction** (`COGNITRACK_RETAIN_RAW_AUDIO=false`) |
| Waveform | A 1,200-point min/max envelope plus speech/pause segments; cannot be inverted to audio |
| Features, predictions | Kept for longitudinal analysis; delete by cascading the participant row |
| Dataset audio | Stays in its licensed local location; nothing is copied |
| Logs | Request method, path, status and latency only; emails and long digit strings are redacted by `ml.utils.redact` |

The SHAP background sample stored in model bundles holds ≤ 100 rows of derived features from the training partition. Treat model artifacts trained on licensed data as derived data under the same terms.

## 5. Labels and clinical claims

* Labels come only from the dataset. Missing labels stay missing, and unlabelled rows are excluded from supervised training.
* The platform does not produce diagnoses. Outputs are phrased as statistical signals (`elevated_signal` or `no_elevated_signal`), and uncertain, OOD or low-quality cases are routed to clinician review.
* Longitudinal flags are statistical changes relative to a personal baseline, never evidence of impairment.

## 6. Data quality controls

* Pandera schema validation, with strict mode failing the pipeline.
* Duplicate IDs and cross-participant duplicate audio (a blocking issue).
* Missing files, missingness per column, class distribution at recording and participant level, and participants with changing labels.
* Audio quality flags. Unusable audio is excluded and counted in `splits.json`.
* Participant-independent splits, enforced by `assert_no_leakage` (participant, recording ID and audio hash).

## 7. Lineage and versioning

* `artifacts/reports/lineage.json` covers sources, dataset version and output.
* Every experiment records its dataset version, config hash and seed, plus the MLflow run.
* `dvc.yaml` declares stage dependencies and parameters; `dvc repro` reproduces outputs, and `dvc push` to a free remote (local disk or self-hosted) shares them.
