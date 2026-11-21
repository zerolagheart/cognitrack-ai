# Model comparison report (generated)

Generated: 2026-11-12T13:12:45.444070+00:00 · dataset version `86f144e1e35d4c89`

> **SYNTHETIC DATA: these results come from programmatically generated data with group differences built in by construction. They demonstrate that the pipeline works and are NOT evidence of clinical validity.**

Serving model: **full__early__random_forest** (selected by cv_auroc = 0.931; best CV score 0.931; simplicity margin 0.01).

## Modality ablation (best experiment per modality set, ranked by CV AUROC)

| Modality set | Best experiment | CV AUROC | Test AUROC [95% CI] | Test bal. acc. | Test Brier |
|---|---|---|---|---|---|
| full | `full__early__random_forest` | 0.931 | 0.966 [0.886, 1.000] | 0.885 | 0.076 |
| acoustic_linguistic | `acoustic_linguistic__early__random_forest` | 0.913 | 0.954 [0.843, 1.000] | 0.885 | 0.072 |
| linguistic | `linguistic__single__logreg` | 0.910 | 0.958 [0.879, 1.000] | 0.892 | 0.078 |
| acoustic_embedding | `acoustic_embedding__early__random_forest` | 0.895 | 0.927 [0.801, 0.995] | 0.873 | 0.091 |
| embedding | `embedding__single__random_forest` | 0.879 | 0.930 [0.828, 0.996] | 0.862 | 0.095 |
| conversational | `conversational__single__random_forest` | 0.877 | 0.808 [0.714, 0.910] | 0.694 | 0.169 |
| acoustic | `acoustic__single__xgboost` | 0.868 | 0.914 [0.774, 0.990] | 0.851 | 0.102 |

**Fusion vs best single modality** (`full__early__random_forest` − `linguistic__single__logreg`): ΔAUROC = 0.008 [95% CI -0.023, 0.063], bootstrap P(Δ ≤ 0) = 0.430. A CI that includes 0 means the test set does not show that fusion helps.

## All experiments

| Experiment | Status | CV AUROC | Test AUROC | AUPRC | Sens. | Spec. | F1 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| `full__early__random_forest` | ok | 0.931 | 0.966 | 0.921 | 0.815 | 0.955 | 0.863 | 0.076 | 0.120 |
| `full__late__logreg` | ok | 0.919 | 0.970 | 0.936 | 0.778 | 0.955 | 0.840 | 0.073 | 0.086 |
| `full__early__mlp` | ok | 0.918 | 0.956 | 0.911 | 0.778 | 0.932 | 0.824 | 0.080 | 0.083 |
| `full__early__xgboost` | ok | 0.914 | 0.953 | 0.855 | 0.778 | 0.955 | 0.840 | 0.085 | 0.075 |
| `acoustic_linguistic__early__random_forest` | ok | 0.913 | 0.954 | 0.874 | 0.815 | 0.955 | 0.863 | 0.072 | 0.079 |
| `linguistic__single__logreg` | ok | 0.910 | 0.958 | 0.910 | 0.852 | 0.932 | 0.868 | 0.078 | 0.090 |
| `acoustic_linguistic__late__logreg` | ok | 0.910 | 0.966 | 0.922 | 0.778 | 0.955 | 0.840 | 0.071 | 0.087 |
| `acoustic_linguistic__early__xgboost` | ok | 0.909 | 0.958 | 0.890 | 0.852 | 0.955 | 0.885 | 0.067 | 0.063 |
| `full__early__lightgbm` | ok | 0.909 | 0.932 | 0.839 | 0.778 | 0.955 | 0.840 | 0.102 | 0.100 |
| `linguistic__single__mlp` | ok | 0.907 | 0.943 | 0.887 | 0.778 | 0.932 | 0.824 | 0.089 | 0.107 |
| `acoustic_linguistic__late__lightgbm` | ok | 0.907 | 0.962 | 0.925 | 0.889 | 0.932 | 0.889 | 0.075 | 0.070 |
| `linguistic__single__random_forest` | ok | 0.905 | 0.961 | 0.917 | 0.778 | 0.932 | 0.824 | 0.077 | 0.072 |
| `full__late__lightgbm` | ok | 0.901 | 0.972 | 0.945 | 0.815 | 0.955 | 0.863 | 0.073 | 0.083 |
| `acoustic_linguistic__early__lightgbm` | ok | 0.900 | 0.963 | 0.925 | 0.852 | 0.932 | 0.868 | 0.067 | 0.090 |
| `full__attention__attention` | ok | 0.899 | 0.935 | 0.864 | 0.889 | 0.909 | 0.873 | 0.088 | 0.095 |
| `linguistic__single__xgboost` | ok | 0.897 | 0.949 | 0.908 | 0.852 | 0.909 | 0.852 | 0.082 | 0.051 |
| `linguistic__single__lightgbm` | ok | 0.896 | 0.942 | 0.895 | 0.852 | 0.909 | 0.852 | 0.091 | 0.072 |
| `acoustic_embedding__early__random_forest` | ok | 0.895 | 0.927 | 0.844 | 0.815 | 0.932 | 0.846 | 0.091 | 0.065 |
| `acoustic_embedding__early__xgboost` | ok | 0.888 | 0.925 | 0.870 | 0.852 | 0.909 | 0.852 | 0.093 | 0.093 |
| `acoustic_embedding__early__lightgbm` | ok | 0.885 | 0.919 | 0.836 | 0.852 | 0.909 | 0.852 | 0.092 | 0.068 |
| `embedding__single__random_forest` | ok | 0.879 | 0.930 | 0.857 | 0.815 | 0.909 | 0.830 | 0.095 | 0.118 |
| `acoustic_linguistic__early__logreg` | ok | 0.877 | 0.944 | 0.863 | 0.815 | 0.932 | 0.846 | 0.114 | 0.116 |
| `conversational__single__random_forest` | ok | 0.877 | 0.808 | 0.723 | 0.593 | 0.795 | 0.615 | 0.169 | 0.102 |
| `full__early__logreg` | ok | 0.873 | 0.938 | 0.888 | 0.704 | 0.909 | 0.760 | 0.114 | 0.108 |
| `acoustic_embedding__late__logreg` | ok | 0.872 | 0.941 | 0.866 | 0.852 | 0.932 | 0.868 | 0.090 | 0.070 |
| `conversational__single__logreg` | ok | 0.871 | 0.848 | 0.817 | 0.407 | 0.955 | 0.550 | 0.156 | 0.077 |
| `acoustic_embedding__late__lightgbm` | ok | 0.870 | 0.949 | 0.897 | 0.815 | 0.932 | 0.846 | 0.086 | 0.075 |
| `conversational__single__xgboost` | ok | 0.869 | 0.790 | 0.731 | 0.593 | 0.818 | 0.627 | 0.179 | 0.060 |
| `embedding__single__logreg` | ok | 0.869 | 0.939 | 0.862 | 0.852 | 0.886 | 0.836 | 0.103 | 0.129 |
| `acoustic_embedding__early__logreg` | ok | 0.869 | 0.936 | 0.871 | 0.852 | 0.864 | 0.821 | 0.099 | 0.083 |
| `acoustic__single__xgboost` | ok | 0.868 | 0.914 | 0.871 | 0.815 | 0.886 | 0.815 | 0.102 | 0.065 |
| `embedding__single__xgboost` | ok | 0.861 | 0.931 | 0.877 | 0.815 | 0.932 | 0.846 | 0.095 | 0.055 |
| `acoustic__single__lightgbm` | ok | 0.860 | 0.913 | 0.872 | 0.815 | 0.909 | 0.830 | 0.097 | 0.097 |
| `acoustic__single__random_forest` | ok | 0.858 | 0.913 | 0.821 | 0.815 | 0.909 | 0.830 | 0.104 | 0.064 |
| `embedding__single__lightgbm` | ok | 0.856 | 0.934 | 0.887 | 0.815 | 0.932 | 0.846 | 0.089 | 0.075 |
| `acoustic_linguistic__early__mlp` | ok | 0.855 | 0.964 | 0.926 | 0.889 | 0.932 | 0.889 | 0.075 | 0.093 |
| `conversational__single__mlp` | ok | 0.852 | 0.850 | 0.812 | 0.593 | 0.909 | 0.681 | 0.153 | 0.069 |
| `embedding__single__mlp` | ok | 0.851 | 0.916 | 0.850 | 0.815 | 0.886 | 0.815 | 0.116 | 0.127 |
| `conversational__single__lightgbm` | ok | 0.850 | 0.782 | 0.697 | 0.667 | 0.795 | 0.667 | 0.183 | 0.066 |
| `acoustic__single__logreg` | ok | 0.840 | 0.926 | 0.877 | 0.815 | 0.864 | 0.800 | 0.107 | 0.056 |
| `acoustic_embedding__attention__attention` | ok | 0.821 | 0.884 | 0.819 | 0.815 | 0.932 | 0.846 | 0.128 | 0.158 |
| `acoustic_linguistic__attention__attention` | ok | 0.820 | 0.953 | 0.866 | 0.889 | 0.932 | 0.889 | 0.066 | 0.076 |
| `acoustic_embedding__early__mlp` | ok | 0.804 | 0.937 | 0.860 | 0.889 | 0.886 | 0.857 | 0.090 | 0.084 |
| `acoustic__single__mlp` | ok | 0.693 | 0.926 | 0.892 | 0.889 | 0.818 | 0.814 | 0.113 | 0.128 |

Metrics are recording-level on the held-out test participants at threshold 0.5 on calibrated probabilities. CIs: participant-level cluster bootstrap. See docs/MODEL_EVALUATION.md for methodology.
