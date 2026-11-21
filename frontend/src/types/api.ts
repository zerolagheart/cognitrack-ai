// Types mirroring backend/app/schemas/api.py and the pipeline report payloads.

export type Dict<T = unknown> = Record<string, T>;

export interface Health {
  status: "ok" | "degraded";
  version: string;
  environment: string;
  database: boolean;
  model_loaded: boolean;
  serving_model: string | null;
  auth_enabled: boolean;
  disclaimer: string;
}

export interface Job {
  id: string;
  job_type?: string;
  type?: string;
  status: "queued" | "running" | "succeeded" | "failed" | string;
  detail?: Dict | null;
  error: string | null;
  created_at: string;
  started_at?: string | null;
  finished_at: string | null;
}

export interface CI { low: number | null; high: number | null; n_boot: number }

export interface AblationRow {
  modality_set: string;
  modalities: string[];
  best_experiment: string;
  cv_auroc: number;
  test_auroc: number;
  test_auroc_ci: CI;
  test_balanced_accuracy: number;
  test_brier: number;
}

export interface Overview {
  counts: {
    participants: number; recordings: number; sessions: number; analysed_sessions: number;
    predictions: number; unusable_recordings: number; synthetic_participants: number;
  };
  synthetic_data: boolean;
  data_quality: {
    quality_flag_counts: Dict<number>;
    validation_passed: boolean | null;
    blocking_issues: string[] | null;
    missing_transcripts: number | null;
    class_distribution: { recordings: Dict<number>; participants: Dict<number> } | null;
  };
  serving_model: {
    experiment_id: string; model: string; strategy: string; modalities: string[];
    metrics: Dict<number | null>; trained_on_synthetic: boolean;
  } | null;
  ablation: AblationRow[] | null;
  banner: string | null;
  recent_jobs: Job[];
}

export interface ParticipantSummary {
  pseudonym: string; dataset_source: string; cohort_label: string | null; is_synthetic: boolean;
  n_sessions: number; age_band: string | null; sex: string | null; first_session: string | null; last_session: string | null;
}

export interface RecordingBrief {
  recording_key: string; duration_s: number | null; quality_usable: boolean | null; quality_flags: string[];
  split: string | null; source: string; has_transcript: boolean; has_speaker_turns: boolean; latest_probability: number | null;
}

export interface SessionDetail {
  session_key: string; session_index: number; session_date: string | null; task: string | null;
  assessment_label: string | null; assessment_score: number | null; assessment_instrument: string | null;
  recordings: RecordingBrief[];
}

export interface ParticipantDetail {
  pseudonym: string; dataset_source: string; cohort_label: string | null; is_synthetic: boolean;
  age_band: string | null; sex: string | null; education_band: string | null; consent_research: boolean;
  sessions: SessionDetail[];
}

export interface FeatureTrends {
  participant: string;
  features: { name: string; label: string; modality: string }[];
  sessions: { session_key: string; session_index: number; session_date: string | null; values: Dict<number | null> }[];
}

export interface Prediction {
  id: number; recording_key: string; model_experiment_id: string; probability: number; model_label: string;
  abstain: boolean; uncertainty: Dict | null; explanation: Explanation | null; origin: string; split: string | null;
  created_at: string; disclaimer: string;
}

export interface Contribution {
  feature: string; label: string; modality: string; shap: number; value: number | null; standardised_value: number;
}

export interface Explanation {
  method: string;
  base_value?: number;
  contributions?: Contribution[];
  modality_contributions?: Dict<number>;
  modality_attention?: Dict<number>;
  output_space?: string;
  note?: string;
  warnings?: string[];
}

export interface RecordingDetail {
  recording_key: string; participant: string; session_key: string; session_index: number; session_date: string | null;
  dataset_source: string; source: string; duration_s: number | null; sample_rate: number | null; transcript: string | null;
  transcript_source: string; has_speaker_turns: boolean; quality_usable: boolean | null;
  quality: { flags: string[]; snr_db: number | null; clipping_fraction: number | null; silence_fraction: number | null; extraction_error: string | null } | null;
  split: string | null; raw_audio_retained: boolean; features: Dict<number | null>; feature_modalities: Dict<string>;
  latest_prediction: Prediction | null;
}

export interface Waveform {
  duration_s: number; time_step_s: number; min: number[]; max: number[];
  speech_segments: [number, number][]; pauses: { start: number; end: number; duration: number }[]; long_pause_threshold_s: number;
}

export interface LongitudinalSession {
  session_id: string; session_index: number; session_timestamp: string | null; n_recordings: number;
  values: Dict<number | null>; z?: Dict<number | null>; moving_average?: Dict<number | null>; is_baseline?: boolean;
  flagged_features?: string[]; ewma_out_of_control?: string[]; composite_deviation?: number | null;
  delta_from_previous?: Dict<number | null>; anomaly_score?: number; is_anomaly?: boolean;
}

export interface LongitudinalResult {
  participant_id: string; status: "analysed" | "insufficient_data"; status_detail?: string; n_sessions: number;
  tracked_features: string[]; feature_labels: Dict<string>;
  missing: { missing_session_indices: number[]; long_intervals: { after_session_index: number; days: number }[] };
  sessions: LongitudinalSession[];
  baseline?: { sessions: string[]; mean: Dict<number>; scale: Dict<number>; scale_type: string; z_threshold: number };
  trends?: Dict<{ slope_per_year?: number; slope_per_session?: number; p_value: number; kendall_tau: number; slope_in_within_sd_units: number }>;
  ewma?: Dict<{ ewma: number[]; ucl: number[]; lcl: number[]; out_of_control: boolean[] }>;
  change_points?: { pelt_session_indices: number[]; bocpd_probability: number[]; power_note: string };
  summary?: { n_flagged_sessions: number; first_flagged_session_index: number | null; features_with_flags: string[];
    max_composite_deviation: number | null; statistical_change_detected: boolean;
    persistent_features: string[]; extreme_features: string[]; decision_rule: string };
  interpretation: string;
}

export interface ExperimentSummary {
  experiment_id: string; modality_set: string; modalities: string[]; strategy: string; model: string; status: string;
  reason?: string; implementation?: string; n_features?: number; calibration?: string; cv_auroc?: number;
  test?: Dict<number> & { confusion_matrix: { tn: number; fp: number; fn: number; tp: number } };
  test_uncalibrated?: Dict<number>; test_participant?: Dict<number>; ci?: Dict<CI>;
  curves?: { roc?: { fpr: number[]; tpr: number[] }; pr?: { recall: number[]; precision: number[] };
    calibration: { mean_predicted: number[]; fraction_positive: number[]; count: number[] } };
  mlflow_run_id?: string | null;
}

export interface ModelComparison {
  generated_at: string; synthetic_data: boolean; banner: string | null; dataset_version: string;
  serving: { experiment_id: string; selection_metric: string; best_cv_score: number; chosen_cv_score: number; simplicity_margin: number; ensemble: string[] };
  experiments: ExperimentSummary[];
  ablation: AblationRow[];
  fusion_vs_best_single: { full: string; best_single: string; delta_auroc: number; ci_low: number; ci_high: number; p_delta_le_0: number } | null;
}

export interface Report<T = Dict> {
  report_type: string; created_at: string; dataset_version: string | null; is_synthetic: boolean; payload: T;
}
