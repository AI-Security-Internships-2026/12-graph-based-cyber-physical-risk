"""Non-GNN baselines (Logistic Regression, MLP, one tree model) that receive the same per-
flow content features as the GraphSAGE edge classifier. For SCADANet that is
`GraphTensors.edge_attr` (built by `_build_edge_attr` in src/data/scadanet.py); for
BATADAL it is the per-window sensor feature matrix from `build_windowed_graphs`,
flattened across sensors.

Every wrapper exposes the same methods so `baseline_runner.py` can loop over model
families without branching:     fit(X_train, y_train, sample_weight=None) -> None
predict_proba(X) -> np.ndarray of shape (n,), P(class=1)     n_trainable_params() ->
Optional[int]   (None plus a reason for tree/LR)

None of these baselines receives node degree as an input feature.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False
    from sklearn.ensemble import HistGradientBoostingClassifier


# ---------------------------------------------------------------------
# Feature extraction — same content features as the GNN's edge_proj input
# ---------------------------------------------------------------------

def scadanet_content_features(edge_attr: torch.Tensor, idx: torch.Tensor) -> np.ndarray:
    """Slice GraphTensors.edge_attr at the given row positions and return
    a plain numpy array. This is deliberately a one-line pass-through
    (not a re-featurization) so there is no way for the non-GNN feature
    set to silently drift from what EdgeClassifierWithAttr's edge_proj
    sees — same tensor, same columns, same row order, only sliced."""
    return edge_attr[idx].detach().cpu().numpy()


def flatten_window_graph_features(window_graphs: List) -> np.ndarray:
    """BATADAL per-window content features for the tabular baselines
    (Experiment B3), flattened rather than aggregated.

    Each window_graphs[i].x is (n_sensors, 4) — [mean_norm, std_norm,
    zscore, sensor_type_code] per sensor (src/data/batadal.py). Sensor
    identity and order are fixed across all windows (same sensor_list
    for every window, per build_windowed_graphs), so flattening
    preserves "feature k belongs to sensor k" across the whole dataset
    rather than collapsing it into a mean/max that would throw away
    which specific sensor moved — flattening is the more information-
    preserving of the two options the issue allows, and is the one
    documented as used here per Experiment B3's requirement to document
    the choice.
    """
    return np.stack([g.x.detach().cpu().numpy().flatten() for g in window_graphs])


# ---------------------------------------------------------------------
# Baseline 1 — Logistic Regression
# ---------------------------------------------------------------------

class LogisticRegressionBaseline:
    """sklearn LogisticRegression, class_weight='balanced' by default to
    match the GNN side's inverse-frequency weighting philosophy (never
    plain unweighted — see build_class_weights docstring in models/gnn.py
    for why an unweighted objective is the wrong comparison point here).
    """

    def __init__(self, max_iter: int = 1000, C: float = 1.0, seed: int = 42):
        self.model = LogisticRegression(
            max_iter=max_iter, C=C, class_weight="balanced", random_state=seed
        )

    def fit(self, X_train, y_train, sample_weight=None):
        self.model.fit(X_train, y_train, sample_weight=sample_weight)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def n_trainable_params(self):
        # coef_ + intercept_ — reported for completeness in Table B3, but
        # not the "parameters" the issue is really contrasting (that
        # contrast is GNN-vs-GNN parameter count, per item 6's ask).
        return int(self.model.coef_.size + self.model.intercept_.size)


# ---------------------------------------------------------------------
# Baseline 2 — MLP (same training-loop shape as the GNN runners: Adam,
# weighted cross-entropy, fixed epoch budget — so "MLP vs GraphSAGE" is
# a capacity/architecture comparison, not also a training-recipe
# comparison the issue didn't ask for)
# ---------------------------------------------------------------------

class MLPClassifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


class MLPBaseline:
    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.2,
                 lr: float = 1e-3, epochs: int = 200, seed: int = 42):
        torch.manual_seed(seed)
        self.model = MLPClassifier(in_dim, hidden_dim, dropout)
        self.lr = lr
        self.epochs = epochs

    def fit(self, X_train, y_train, sample_weight=None, log_every: int = 0, log_prefix: str = ""):
        X = torch.as_tensor(X_train, dtype=torch.float32)
        y = torch.as_tensor(y_train, dtype=torch.long)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)

        if sample_weight is not None:
            w = torch.as_tensor(sample_weight, dtype=torch.float32)
        else:
            counts = torch.bincount(y, minlength=2).float()
            class_w = y.shape[0] / (2.0 * counts.clamp(min=1))
            w = class_w[y]

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            out = self.model(X)
            per_sample_loss = F.cross_entropy(out, y, reduction="none")
            loss = (per_sample_loss * w).mean()
            loss.backward()
            optimizer.step()

            if log_every and (epoch == 1 or epoch % log_every == 0 or epoch == self.epochs):
                with torch.no_grad():
                    pred = out.argmax(dim=1)
                    acc = (pred == y).float().mean().item()
                print(f"{log_prefix}epoch {epoch:4d}/{self.epochs}  loss={loss.item():.4f}  TRAIN acc={acc:.4f}")

    def predict_proba(self, X) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32)
            probs = F.softmax(self.model(X_t), dim=1)[:, 1]
        return probs.numpy()

    def n_trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------
# Baseline 3 — tree model
# ---------------------------------------------------------------------

class TreeBaseline:
    def __init__(self, max_depth: int = 6, n_estimators: int = 200,
                 learning_rate: float = 0.1, seed: int = 42):
        self.backend = "xgboost" if _HAS_XGBOOST else "hist_gradient_boosting"
        if _HAS_XGBOOST:
            self.model = xgb.XGBClassifier(
                max_depth=max_depth, n_estimators=n_estimators,
                learning_rate=learning_rate, eval_metric="logloss",
                random_state=seed, n_jobs=-1,
            )
        else:
            self.model = HistGradientBoostingClassifier(
                max_depth=max_depth, max_iter=n_estimators,
                learning_rate=learning_rate, random_state=seed,
            )

    def fit(self, X_train, y_train, sample_weight=None):
        self.model.fit(X_train, y_train, sample_weight=sample_weight)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def n_trainable_params(self):
        return None  # not a meaningful concept for a tree ensemble


def predict_labels(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (probs >= threshold).astype(np.int64)


@dataclass
class HParamGrid:
    """Small validation-only hyperparameter search space, kept in one place so
    baseline_runner.py and the reported search space read from the same source.
    """
    hidden_dim: List[int] = None
    lr: List[float] = None
    dropout: List[float] = None
    tree_max_depth: List[int] = None
    tree_n_estimators: List[int] = None
    lr_C: List[float] = None

    def __post_init__(self):
        self.hidden_dim = self.hidden_dim or [32, 64]
        self.lr = self.lr or [1e-3, 5e-3]
        self.dropout = self.dropout or [0.0, 0.2]
        self.tree_max_depth = self.tree_max_depth or [4, 6]
        self.tree_n_estimators = self.tree_n_estimators or [100, 200]
        self.lr_C = self.lr_C or [0.1, 1.0]

    def as_dict(self) -> Dict:
        return {
            "hidden_dimension": self.hidden_dim,
            "learning_rate": self.lr,
            "dropout": self.dropout,
            "tree_max_depth": self.tree_max_depth,
            "tree_n_estimators": self.tree_n_estimators,
            "logistic_regression_C": self.lr_C,
        }
