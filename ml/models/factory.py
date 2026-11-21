"""Model factory: preprocessing + estimator pipelines with a uniform scikit-learn interface.

Preprocessing (median imputation, scaling and PCA for high-dimensional embeddings) lives inside the
pipeline, so it is fitted on training folds only and never sees validation/test data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features.registry import EMBEDDING_PREFIX
from ml.models.mlp import make_mlp

MODEL_NAMES = ("logreg", "random_forest", "xgboost", "lightgbm", "mlp")
TREE_MODELS = {"random_forest", "xgboost", "lightgbm"}
EMBEDDING_PCA_COMPONENTS = 16


def build_estimator(name: str, seed: int, pos_weight: float = 1.0, mlp_params: dict[str, Any] | None = None) -> tuple[Any, str]:
    """Return ``(estimator, implementation)``; class imbalance handled per library."""
    if name == "logreg":
        return LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000, random_state=seed), "sklearn"
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features="sqrt", class_weight="balanced",
                                      n_jobs=-1, random_state=seed), "sklearn"
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                             reg_lambda=1.0, scale_pos_weight=pos_weight, eval_metric="logloss", random_state=seed,
                             n_jobs=2, tree_method="hist"), "xgboost"
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05, min_child_samples=10, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.7, class_weight="balanced", random_state=seed,
                              n_jobs=2, verbose=-1), "lightgbm"
    if name == "mlp":
        if mlp_params is None:
            raise ValueError("mlp requires training.mlp parameters")
        return make_mlp(mlp_params, seed)
    raise KeyError(f"Unknown model '{name}'. Available: {MODEL_NAMES}")


def build_preprocessor(columns: list[str], n_train: int) -> ColumnTransformer:
    emb = [c for c in columns if c.startswith(EMBEDDING_PREFIX)]
    other = [c for c in columns if not c.startswith(EMBEDDING_PREFIX)]
    transformers = []
    if other:
        transformers.append(("handcrafted", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), other))
    if emb:
        n_comp = int(min(EMBEDDING_PCA_COMPONENTS, len(emb), max(2, n_train // 5)))
        transformers.append(("embedding", Pipeline([
            ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_comp, random_state=0)),
        ]), emb))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_pipeline(name: str, columns: list[str], y: np.ndarray, seed: int, mlp_params: dict[str, Any] | None = None) -> tuple[Pipeline, str]:
    pos = float(np.mean(y))
    estimator, impl = build_estimator(name, seed, pos_weight=(1 - pos) / max(pos, 1e-6), mlp_params=mlp_params)
    pipe = Pipeline([("pre", build_preprocessor(columns, len(y))), ("clf", estimator)])
    return pipe, impl


def transformed_feature_names(pipe: Pipeline) -> list[str]:
    """Names of the features entering the final estimator (PCA components are emb_pc_XX)."""
    pre: ColumnTransformer = pipe.named_steps["pre"]
    names: list[str] = []
    for name, trans, cols in pre.transformers_:
        if name == "handcrafted":
            names.extend(cols)
        elif name == "embedding":
            n = trans.named_steps["pca"].n_components_
            names.extend(f"{EMBEDDING_PREFIX}pc_{i + 1:02d}" for i in range(n))
    return names
