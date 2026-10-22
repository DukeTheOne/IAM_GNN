"""TAC-GB Gradient-Boosted Decision Tree Baseline from TAC-Bench.

Implements the state-of-the-art non-neural / tabular gradient-boosted link prediction
baseline for access control topologies (TAC-Bench 2025/2026). Utilizes histogram-based
gradient boosting over topological, distance, organizational, and privilege features
with balanced class weighting for severe positive-class sparsity.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-untyped]
from torch_geometric.data import HeteroData

from iam.baselines.features import EdgePairFeaturizer
from iam.baselines.metrics import (
    compute_false_negative_rate,
    compute_mean_reciprocal_rank,
    compute_pr_auc,
    compute_recall_at_k,
    compute_roc_auc,
)
from iam.generator.graph import IAMGraph

logger = logging.getLogger(__name__)


class TACGBLinkPredictor:
    """Gradient-Boosted Decision Tree baseline for link prediction and PE bridge discovery."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 150,
        max_leaf_nodes: int = 31,
        l2_regularization: float = 1.0,
        class_weight: str | dict[int, float] = "balanced",
        random_state: int = 42,
        featurizer: EdgePairFeaturizer | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.l2_regularization = l2_regularization
        self.class_weight = class_weight
        self.random_state = random_state
        self.featurizer = featurizer or EdgePairFeaturizer()

        self.classifier: HistGradientBoostingClassifier = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            l2_regularization=self.l2_regularization,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.is_fitted: bool = False

    def fit(
        self,
        graph_or_data: IAMGraph | HeteroData,
        pos_pairs: list[tuple[str, str]],
        neg_pairs: list[tuple[str, str]],
    ) -> TACGBLinkPredictor:
        """Fit the TAC-GB classifier on positive and negative candidate edge pairs.

        Args:
            graph_or_data: Training graph view G_o.
            pos_pairs: List of true positive (u, v) pairs.
            neg_pairs: List of true negative (u, v) pairs.

        Returns:
            self: The fitted predictor.
        """
        if not pos_pairs and not neg_pairs:
            raise ValueError("Cannot fit TACGBLinkPredictor with empty training pairs.")

        x_pos = (
            self.featurizer.featurize_pairs(graph_or_data, pos_pairs)
            if pos_pairs
            else np.zeros((0, self.featurizer.num_features), dtype=np.float32)
        )
        x_neg = (
            self.featurizer.featurize_pairs(graph_or_data, neg_pairs)
            if neg_pairs
            else np.zeros((0, self.featurizer.num_features), dtype=np.float32)
        )

        x_feat = np.vstack([x_pos, x_neg])
        y = np.concatenate(
            [
                np.ones(len(pos_pairs), dtype=np.int32),
                np.zeros(len(neg_pairs), dtype=np.int32),
            ]
        )

        return self.fit_from_features(x_feat, y)

    def fit_from_features(self, x_feat: np.ndarray, y: np.ndarray) -> TACGBLinkPredictor:
        """Fit the classifier directly on pre-extracted feature matrix and labels."""
        if len(y) == 0:
            raise ValueError("Training set must contain at least 1 sample.")

        unique_classes = np.unique(y)
        if len(unique_classes) == 1:
            # Degenerate case with single class
            self.classifier.fit(x_feat, y)
            self.is_fitted = True
            return self

        self.classifier.fit(x_feat, y)
        self.is_fitted = True
        return self

    def predict_proba(
        self,
        graph_or_data: IAMGraph | HeteroData,
        pairs: list[tuple[str, str]],
    ) -> np.ndarray:
        """Predict positive link probabilities for an ordered list of candidate pairs.

        Args:
            graph_or_data: Partially observed graph view G_o.
            pairs: List of candidate (u, v) pairs to evaluate.

        Returns:
            np.ndarray: 1D array of calibrated link probabilities in [0.0, 1.0].
        """
        if not pairs:
            return np.zeros(0, dtype=np.float32)

        x_feat = self.featurizer.featurize_pairs(graph_or_data, pairs)
        return self.predict_proba_from_features(x_feat)

    def predict_proba_from_features(self, x_feat: np.ndarray) -> np.ndarray:
        """Predict positive link probabilities directly from extracted feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("TACGBLinkPredictor must be fitted before predict_proba.")

        if len(x_feat) == 0:
            return np.zeros(0, dtype=np.float32)

        # Handle single-class training edge case
        if len(self.classifier.classes_) == 1:
            single_cls = self.classifier.classes_[0]
            val = 1.0 if single_cls == 1 else 0.0
            return np.full(len(x_feat), val, dtype=np.float32)

        probs: np.ndarray = self.classifier.predict_proba(x_feat)
        pos_idx = list(self.classifier.classes_).index(1) if 1 in self.classifier.classes_ else 1
        return np.asarray(probs[:, pos_idx], dtype=np.float32)

    def evaluate_metrics(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        k: int = 10,
    ) -> dict[str, float]:
        """Compute standard link prediction evaluation metrics."""
        return {
            "pr_auc": compute_pr_auc(y_true, y_score),
            "roc_auc": compute_roc_auc(y_true, y_score),
            "recall_at_k": compute_recall_at_k(y_true, y_score, k=k),
            "fnr": compute_false_negative_rate(y_true, y_score),
            "mrr": compute_mean_reciprocal_rank(y_true, y_score),
        }
