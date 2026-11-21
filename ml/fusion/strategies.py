"""Multimodal fusion strategies.

* **Early fusion** — concatenate modality features into one model (:func:`ml.models.factory.build_pipeline`
  with the union of columns). No extra class needed.
* **Late fusion** — :class:`LateFusionClassifier`: one model per modality; a logistic-regression
  meta-learner is trained on *out-of-fold*, participant-grouped base predictions (stacking), which avoids
  the optimistic bias of training the meta-learner on in-sample base predictions. Missing modalities are
  handled at prediction time by substituting the base-rate logit.
* **Attention fusion** — :class:`AttentionFusionClassifier`: small per-modality encoders whose outputs
  are combined by learned, input-dependent attention weights with a modality-availability mask. Trained
  end to end in PyTorch; exposes per-sample modality attention for explanation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml.data.splitting import group_kfold_indices
from ml.models.factory import build_pipeline


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class LateFusionClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, modality_columns: dict[str, list[str]], base_model: str = "logreg", n_folds: int = 5,
                 random_state: int = 42, mlp_params: dict[str, Any] | None = None) -> None:
        self.modality_columns = modality_columns
        self.base_model = base_model
        self.n_folds = n_folds
        self.random_state = random_state
        self.mlp_params = mlp_params

    def _base(self, cols: list[str], y: np.ndarray) -> Any:
        return build_pipeline(self.base_model, cols, y, self.random_state, self.mlp_params)[0]

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None = None) -> LateFusionClassifier:
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        groups = np.asarray(groups) if groups is not None else np.arange(len(y))
        folds = group_kfold_indices(groups, y, self.n_folds, self.random_state)
        modalities = list(self.modality_columns)
        oof = np.zeros((len(y), len(modalities)))
        for j, m in enumerate(modalities):
            cols = self.modality_columns[m]
            for tr, va in folds:
                model = self._base(cols, y[tr]).fit(X.iloc[tr][cols], y[tr])
                oof[va, j] = model.predict_proba(X.iloc[va][cols])[:, 1]
        self.base_models_ = {m: self._base(self.modality_columns[m], y).fit(X[self.modality_columns[m]], y) for m in modalities}
        self.prior_logit_ = float(_logit(np.array([y.mean()]))[0])
        self.meta_ = LogisticRegression(C=1.0, class_weight="balanced", random_state=self.random_state).fit(_logit(oof), y)
        self.modalities_ = modalities
        return self

    def modality_logits(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for m in self.modalities_:
            present = X[self.modality_columns[m]].notna().any(axis=1).to_numpy()
            logits = np.full(len(X), self.prior_logit_)
            if present.any():
                logits[present] = _logit(self.base_models_[m].predict_proba(X.loc[present, self.modality_columns[m]])[:, 1])
            cols[m] = logits
        return pd.DataFrame(cols, index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.meta_.predict_proba(self.modality_logits(X).to_numpy())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    def modality_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Additive log-odds contribution of each modality (meta coefficient × centred base logit)."""
        logits = self.modality_logits(X)
        coef = self.meta_.coef_[0]
        return (logits - self.prior_logit_) * coef


class AttentionFusionClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, modality_columns: dict[str, list[str]], hidden: int = 16, epochs: int = 200, lr: float = 3e-3,
                 weight_decay: float = 1e-3, dropout: float = 0.2, patience: int = 20, random_state: int = 42) -> None:
        self.modality_columns = modality_columns
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.patience = patience
        self.random_state = random_state

    def _prepare(self, X: pd.DataFrame, fit: bool) -> tuple[list[np.ndarray], np.ndarray]:
        blocks, masks = [], []
        if fit:
            self.preprocessors_ = {}
        for m in self.modalities_:
            cols = self.modality_columns[m]
            block = X[cols]
            masks.append(block.notna().any(axis=1).to_numpy())
            if fit:
                imp = SimpleImputer(strategy="median", keep_empty_features=True)
                sc = StandardScaler()
                self.preprocessors_[m] = (imp, sc)
                arr = sc.fit_transform(imp.fit_transform(block))
            else:
                imp, sc = self.preprocessors_[m]
                arr = sc.transform(imp.transform(block))
            blocks.append(np.nan_to_num(arr).astype(np.float32))
        return blocks, np.column_stack(masks).astype(np.float32)

    def _network(self) -> Any:
        # Module-level class (in a torch-only module) so fitted models can be pickled into bundles.
        from ml.fusion.attention_net import ModalityAttentionNet

        dims = [len(self.modality_columns[m]) for m in self.modalities_]
        return ModalityAttentionNet(dims, self.hidden, self.dropout)

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None = None) -> AttentionFusionClassifier:
        import torch
        from sklearn.model_selection import GroupShuffleSplit

        torch.manual_seed(self.random_state)
        self.modalities_ = list(self.modality_columns)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        y_idx = y_idx.astype(np.float32)
        groups = np.asarray(groups) if groups is not None else np.arange(len(y))
        tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=self.random_state).split(X, y_idx, groups))
        blocks, mask = self._prepare(X, fit=True)
        to_t = lambda idx: ([torch.from_numpy(b[idx]) for b in blocks], torch.from_numpy(mask[idx]), torch.from_numpy(y_idx[idx]))  # noqa: E731
        (btr, mtr, ytr), (bva, mva, yva) = to_t(tr), to_t(va)
        self.net_ = self._network()
        pos = y_idx[tr].mean()
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - pos) / max(pos, 1e-6)]))
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, wait = np.inf, None, 0
        for _ in range(self.epochs):
            self.net_.train()
            opt.zero_grad()
            # Modality dropout: randomly hide one available modality so the model learns to cope with gaps.
            drop = (torch.rand_like(mtr) < 0.15).float()
            m_train = torch.where(mtr.sum(1, keepdim=True) - drop.sum(1, keepdim=True) >= 1, mtr * (1 - drop), mtr)
            logits, _ = self.net_(btr, m_train)
            loss_fn(logits, ytr).backward()
            opt.step()
            self.net_.eval()
            with torch.no_grad():
                val = float(loss_fn(self.net_(bva, mva)[0], yva))
            if val < best - 1e-4:
                best, wait = val, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def _forward(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        import torch

        blocks, mask = self._prepare(X, fit=False)
        self.net_.eval()
        with torch.no_grad():
            logits, attn = self.net_([torch.from_numpy(b) for b in blocks], torch.from_numpy(mask))
        return 1 / (1 + np.exp(-logits.numpy())), attn.numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p, _ = self._forward(X)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    def modality_attention(self, X: pd.DataFrame) -> pd.DataFrame:
        _, attn = self._forward(X)
        return pd.DataFrame(attn, columns=self.modalities_, index=X.index)
