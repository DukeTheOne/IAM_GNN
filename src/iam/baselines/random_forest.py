"""Random Forest Tabular Link Prediction Baseline for IAM authorization graphs.

Employs balanced bagging ensembles over topological, shortest-path, departmental,
and privilege-oriented feature representations to score missing edges in Go.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from torch_geometric.data import HeteroData

from iam.baselines.features import EdgePairFeaturizer
from iam.generator.graph import IAMGraph


class RandomForestLinkPredictor:
    """Tabular Random Forest baseline for edge existence and PE link prediction."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 15,
        class_weight: str | dict[int, float] = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
        featurizer: EdgePairFeaturizer | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.featurizer = featurizer or EdgePairFeaturizer()

        self.classifier: RandomForestClassifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.is_fitted: bool = False

    def fit(
        self,
        graph_or_data: IAMGraph | HeteroData,
        pos_pairs: list[tuple[str, str]],
        neg_pairs: list[tuple[str, str]],
    ) -> RandomForestLinkPredictor:
        """Fit the Random Forest classifier on positive and negative candidate edge pairs.

        Args:
            graph_or_data: Training graph view Go.
            pos_pairs: List of true positive (u, v) pairs.
            neg_pairs: List of true negative (u, v) pairs.

        Returns:
            self: The fitted predictor.
        """
        if not pos_pairs and not neg_pairs:
            raise ValueError("Cannot fit RandomForestLinkPredictor with empty training pairs.")

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

    def fit_from_features(self, x_feat: np.ndarray, y: np.ndarray) -> RandomForestLinkPredictor:
        """Fit the classifier directly on pre-extracted feature matrix x_feat and labels y."""
        if len(y) == 0:
            raise ValueError("Training set must contain at least 1 sample.")

        unique_classes = np.unique(y)
        if len(unique_classes) == 1:
            # Degenerate case with single class: create dummy classifier behavior
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
        """Predict continuous probability scores in [0.0, 1.0] for candidate pairs.

        Args:
            graph_or_data: Evaluation graph view Go.
            pairs: List of candidate (u, v) pairs to evaluate.

        Returns:
            np.ndarray of shape (len(pairs),) with probability of positive link existence.
        """
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        if not self.is_fitted:
            raise RuntimeError("RandomForestLinkPredictor must be fitted before predict_proba.")

        x_feat = self.featurizer.featurize_pairs(graph_or_data, pairs)
        return self.predict_proba_from_features(x_feat)

    def predict_proba_from_features(self, x_feat: np.ndarray) -> np.ndarray:
        """Predict link probabilities directly from pre-extracted features x_feat."""
        if not self.is_fitted:
            raise RuntimeError("RandomForestLinkPredictor must be fitted before predict_proba.")
        if len(x_feat) == 0:
            return np.zeros(0, dtype=np.float32)

        classes = list(self.classifier.classes_)
        proba_matrix = self.classifier.predict_proba(x_feat)

        if 1 in classes:
            pos_idx = classes.index(1)
            return np.asarray(proba_matrix[:, pos_idx], dtype=np.float32)
        elif 1.0 in classes:
            pos_idx = classes.index(1.0)
            return np.asarray(proba_matrix[:, pos_idx], dtype=np.float32)
        else:
            # Only negative class was present during training
            return np.zeros(len(x_feat), dtype=np.float32)

    def predict(
        self,
        graph_or_data: IAMGraph | HeteroData,
        pairs: list[tuple[str, str]],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict binary edge existence decisions for candidate pairs."""
        probs = self.predict_proba(graph_or_data, pairs)
        return (probs >= threshold).astype(bool)

    @property
    def feature_importances(self) -> dict[str, float]:
        """Dictionary of feature names mapped to their Gini importance scores."""
        if not self.is_fitted:
            return dict.fromkeys(self.featurizer.feature_names, 0.0)

        raw_importances = self.classifier.feature_importances_
        return {
            name: float(imp)
            for name, imp in zip(self.featurizer.feature_names, raw_importances, strict=False)
        }

    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }
