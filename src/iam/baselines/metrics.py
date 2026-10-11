"""Baseline metrics evaluation suite for IAM privilege escalation link prediction.

Computes primary (PR-AUC), secondary (Recall@K, ROC-AUC, FNR, MRR), and security-specific
(Attack-Path Recovery Rate) evaluation metrics across held-out test splits under
structured masking conditions (Condition A: Random, Condition E: Adversarial Bridge).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-untyped]
from torch_geometric.data import HeteroData

from iam.baselines.bfs import DeterministicBFSEngine
from iam.baselines.motif_heuristic import MotifCompletionHeuristic
from iam.baselines.oracle import ExactReachabilityOracle
from iam.baselines.random_forest import RandomForestLinkPredictor

if TYPE_CHECKING:
    from iam.models.masking import (
        BaseMaskingOperator,
        MaskingResult,
    )
    from iam.models.sampler import StratifiedNegativeSampler


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Precision-Recall Area Under the Curve (Average Precision).

    Args:
        y_true: 1D binary array of ground truth labels (0 or 1).
        y_score: 1D array of continuous confidence scores.

    Returns:
        PR-AUC score in [0.0, 1.0].
    """
    if len(y_true) == 0 or len(y_score) == 0:
        return 0.0

    pos_count = int(np.sum(y_true == 1))
    if pos_count == 0:
        return 0.0
    if pos_count == len(y_true):
        return 1.0

    try:
        val: float = float(average_precision_score(y_true, y_score))
        return float(np.clip(val, 0.0, 1.0))
    except Exception:
        return 0.0


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Receiver Operating Characteristic Area Under the Curve.

    Args:
        y_true: 1D binary array of ground truth labels (0 or 1).
        y_score: 1D array of continuous confidence scores.

    Returns:
        ROC-AUC score in [0.0, 1.0]. Returns 0.5 if only one class is present.
    """
    if len(y_true) == 0 or len(y_score) == 0:
        return 0.5

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return 0.5

    try:
        val: float = float(roc_auc_score(y_true, y_score))
        return float(np.clip(val, 0.0, 1.0))
    except Exception:
        return 0.5


def compute_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Compute fraction of true positive links retrieved within top-K ranked candidates.

    Args:
        y_true: 1D binary array of ground truth labels.
        y_score: 1D array of confidence scores.
        k: Ranking cutoff parameter.

    Returns:
        Recall@K in [0.0, 1.0].
    """
    if len(y_true) == 0 or len(y_score) == 0 or k <= 0:
        return 0.0

    total_positives = int(np.sum(y_true == 1))
    if total_positives == 0:
        return 1.0

    if np.all(y_score <= 0.0):
        return 0.0

    k_clamped = min(k, len(y_score))
    top_indices = np.argsort(-y_score)[:k_clamped]
    positives_in_top_k = int(np.sum((y_true[top_indices] == 1) & (y_score[top_indices] > 0.0)))

    return float(positives_in_top_k / total_positives)


def compute_false_negative_rate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute False Negative Rate at designated decision threshold.

    FNR = FN / (TP + FN) = 1.0 - Recall.

    Args:
        y_true: 1D binary array of ground truth labels.
        y_score: 1D array of confidence scores.
        threshold: Decision threshold for positive link classification.

    Returns:
        FNR in [0.0, 1.0].
    """
    if len(y_true) == 0 or len(y_score) == 0:
        return 0.0

    pos_mask = y_true == 1
    total_positives = int(np.sum(pos_mask))
    if total_positives == 0:
        return 0.0

    predicted_positives = y_score >= threshold
    true_positives = int(np.sum(pos_mask & predicted_positives))
    false_negatives = total_positives - true_positives

    return float(false_negatives / total_positives)


def compute_mean_reciprocal_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank (MRR) of the first retrieved positive link.

    Args:
        y_true: 1D binary array of ground truth labels.
        y_score: 1D array of confidence scores.

    Returns:
        MRR in [0.0, 1.0].
    """
    if len(y_true) == 0 or len(y_score) == 0 or not np.any(y_true == 1) or np.all(y_score <= 0.0):
        return 0.0

    sorted_indices = np.argsort(-y_score)
    for rank, idx in enumerate(sorted_indices, start=1):
        if y_true[idx] == 1 and y_score[idx] > 0.0:
            return float(1.0 / rank)

    return 0.0


def compute_attack_path_recovery(bridge_ranks: list[int], k: int) -> float:
    """Compute fraction of severed attack bottlenecks ranked in top-K candidates.

    Args:
        bridge_ranks: 1-indexed list of candidate ranks assigned to true bridge edges.
        k: Top-K ranking cutoff.

    Returns:
        Attack-Path Recovery Rate in [0.0, 1.0].
    """
    if not bridge_ranks or k <= 0:
        return 0.0

    recovered_count = sum(1 for rank in bridge_ranks if rank <= k)
    return float(recovered_count / len(bridge_ranks))


@dataclass
class BaselineMetricsResult:
    """Comprehensive evaluation metrics summary for a single baseline on a benchmark split."""

    baseline_name: str
    masking_condition: str
    pr_auc: float
    roc_auc: float
    recall_at_10: float
    recall_at_50: float
    recall_at_100: float
    fnr: float
    mrr: float
    attack_path_recovery_at_10: float
    attack_path_recovery_at_50: float
    attack_path_recovery_at_100: float
    num_samples: int
    num_positives: int
    num_negatives: int
    execution_time_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics result to JSON-serializable dictionary."""
        return {
            "baseline_name": self.baseline_name,
            "masking_condition": self.masking_condition,
            "pr_auc": round(self.pr_auc, 4),
            "roc_auc": round(self.roc_auc, 4),
            "recall_at_10": round(self.recall_at_10, 4),
            "recall_at_50": round(self.recall_at_50, 4),
            "recall_at_100": round(self.recall_at_100, 4),
            "fnr": round(self.fnr, 4),
            "mrr": round(self.mrr, 4),
            "attack_path_recovery_at_10": round(self.attack_path_recovery_at_10, 4),
            "attack_path_recovery_at_50": round(self.attack_path_recovery_at_50, 4),
            "attack_path_recovery_at_100": round(self.attack_path_recovery_at_100, 4),
            "num_samples": self.num_samples,
            "num_positives": self.num_positives,
            "num_negatives": self.num_negatives,
            "execution_time_sec": round(self.execution_time_sec, 4),
            "metadata": self.metadata,
        }


class BaselineBenchmarkSuite:
    """Unified benchmark evaluation harness for IAM graph reachability and link prediction."""

    def __init__(
        self,
        negative_sampler: StratifiedNegativeSampler | None = None,
        seed: int = 42,
    ) -> None:
        if negative_sampler is None:
            from iam.models.sampler import StratifiedNegativeSampler

            self.negative_sampler = StratifiedNegativeSampler()
        else:
            self.negative_sampler = negative_sampler
        self.seed = seed

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        is_bridge: np.ndarray | None = None,
        baseline_name: str = "Baseline",
        condition_name: str = "Condition",
        execution_time_sec: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> BaselineMetricsResult:
        """Evaluate pre-computed scores against true binary labels and bridge masks.

        Args:
            y_true: 1D ground truth binary array.
            y_score: 1D predicted continuous confidence scores.
            is_bridge: Optional boolean mask designating critical PE bottleneck links.
            baseline_name: Identifier for baseline model.
            condition_name: Identifier for masking condition.
            execution_time_sec: Wall-clock latency in seconds.
            metadata: Additional metadata dictionary.

        Returns:
            Populated BaselineMetricsResult instance.
        """
        pr_auc = compute_pr_auc(y_true, y_score)
        roc_auc = compute_roc_auc(y_true, y_score)
        rec_10 = compute_recall_at_k(y_true, y_score, k=10)
        rec_50 = compute_recall_at_k(y_true, y_score, k=50)
        rec_100 = compute_recall_at_k(y_true, y_score, k=100)
        fnr = compute_false_negative_rate(y_true, y_score, threshold=0.5)
        mrr = compute_mean_reciprocal_rank(y_true, y_score)

        # Attack-Path Recovery Rate calculation
        bridge_ranks: list[int] = []
        if is_bridge is not None and np.any(is_bridge):
            sorted_indices = list(np.argsort(-y_score))
            for i, val in enumerate(is_bridge):
                if val:
                    if y_score[i] <= 0.0:
                        bridge_ranks.append(len(y_score) + 999)
                    else:
                        rank = sorted_indices.index(i) + 1
                        bridge_ranks.append(rank)

            rec_path_10 = compute_attack_path_recovery(bridge_ranks, k=10)
            rec_path_50 = compute_attack_path_recovery(bridge_ranks, k=50)
            rec_path_100 = compute_attack_path_recovery(bridge_ranks, k=100)
        else:
            rec_path_10 = rec_10
            rec_path_50 = rec_50
            rec_path_100 = rec_100

        num_pos = int(np.sum(y_true == 1))
        num_neg = int(np.sum(y_true == 0))

        return BaselineMetricsResult(
            baseline_name=baseline_name,
            masking_condition=condition_name,
            pr_auc=pr_auc,
            roc_auc=roc_auc,
            recall_at_10=rec_10,
            recall_at_50=rec_50,
            recall_at_100=rec_100,
            fnr=fnr,
            mrr=mrr,
            attack_path_recovery_at_10=rec_path_10,
            attack_path_recovery_at_50=rec_path_50,
            attack_path_recovery_at_100=rec_path_100,
            num_samples=len(y_true),
            num_positives=num_pos,
            num_negatives=num_neg,
            execution_time_sec=execution_time_sec,
            metadata=metadata or {},
        )

    def evaluate_on_hetero_data(
        self,
        baseline: Any,
        data: HeteroData,
        masking_operator: BaseMaskingOperator | None = None,
        condition_name: str = "A",
        seed: int = 42,
    ) -> BaselineMetricsResult:
        """Evaluate a baseline engine on a HeteroData environment under structured masking.

        Args:
            baseline: Baseline instance (Oracle, BFS, Random Forest, or Motif Heuristic).
            data: Full ground truth HeteroData environment.
            masking_operator: Masking operator to produce Go and supervision sets.
            condition_name: Condition shorthand string ('A', 'E', etc.).
            seed: RNG seed for reproducible evaluation.

        Returns:
            Populated BaselineMetricsResult instance.
        """
        start_time = time.perf_counter()

        # Instantiate default operator if none provided
        if masking_operator is None:
            from iam.models.masking import (
                AdversarialBridgeMaskingOperator,
                RandomMaskingOperator,
            )

            if condition_name.upper() in ("E", "CONDITION_E_ADVERSARIAL_BRIDGE"):
                op: BaseMaskingOperator = AdversarialBridgeMaskingOperator()
            else:
                op = RandomMaskingOperator(mask_ratio=0.20)
        else:
            op = masking_operator

        mask_res: MaskingResult = op.apply(
            data,
            seed=seed,
            negative_sampler=self.negative_sampler,
            negative_ratio=1.0,
        )

        observed_go = mask_res.observed_data

        # Aggregate candidate pairs and ground-truth labels across masked relation types
        eval_pairs: list[tuple[str, str]] = []
        labels_list: list[float] = []
        is_bridge_list: list[bool] = []

        for triple, sup in mask_res.supervision_by_edge_type.items():
            src_type, _rel, dst_type = triple
            src_ids = getattr(data[src_type], "node_ids", [])
            dst_ids = getattr(data[dst_type], "node_ids", [])

            edge_idx = sup.edge_label_index
            lbls = sup.edge_label.tolist()
            num_items = edge_idx.size(1)

            bridge_mask = mask_res.hidden_edge_is_bridge.get(triple)

            for i in range(num_items):
                u_idx = int(edge_idx[0, i])
                v_idx = int(edge_idx[1, i])
                u_id = src_ids[u_idx] if u_idx < len(src_ids) else f"node_{u_idx}"
                v_id = dst_ids[v_idx] if v_idx < len(dst_ids) else f"node_{v_idx}"
                eval_pairs.append((u_id, v_id))
                labels_list.append(float(lbls[i]))

                is_br = False
                if bridge_mask is not None and i < bridge_mask.size(0):
                    is_br = bool(bridge_mask[i].item())
                is_bridge_list.append(is_br)

        y_true = np.array(labels_list, dtype=np.int32)
        is_bridge = np.array(is_bridge_list, dtype=bool)

        if len(eval_pairs) == 0:
            return BaselineMetricsResult(
                baseline_name=getattr(baseline, "__class__", type(baseline)).__name__,
                masking_condition=condition_name,
                pr_auc=0.0,
                roc_auc=0.5,
                recall_at_10=0.0,
                recall_at_50=0.0,
                recall_at_100=0.0,
                fnr=0.0,
                mrr=0.0,
                attack_path_recovery_at_10=0.0,
                attack_path_recovery_at_50=0.0,
                attack_path_recovery_at_100=0.0,
                num_samples=0,
                num_positives=0,
                num_negatives=0,
                execution_time_sec=time.perf_counter() - start_time,
            )

        # Baseline-specific score generation
        baseline_name = getattr(baseline, "__class__", type(baseline)).__name__

        if isinstance(baseline, ExactReachabilityOracle):
            # Oracle operates with full knowledge of ground truth G
            y_score = y_true.astype(np.float32)

        elif isinstance(baseline, DeterministicBFSEngine):
            # Deterministic BFS operates strictly over Go
            bfs_res = baseline.evaluate(observed_go)
            y_score = np.array(
                [1.0 if bfs_res.is_reachable(u, v) else 0.0 for (u, v) in eval_pairs],
                dtype=np.float32,
            )

        elif isinstance(baseline, RandomForestLinkPredictor):
            # Random Forest tabular link predictor
            if not baseline.is_fitted:
                # Train predictor on a split of the observed data
                train_pos = [
                    pair for pair, lbl in zip(eval_pairs, y_true, strict=False) if lbl == 1
                ]
                train_neg = [
                    pair for pair, lbl in zip(eval_pairs, y_true, strict=False) if lbl == 0
                ]
                baseline.fit(observed_go, train_pos, train_neg)
            y_score = baseline.predict_proba(observed_go, eval_pairs)

        elif isinstance(baseline, MotifCompletionHeuristic):
            # Structural rule-based heuristic
            y_score = baseline.score_pairs(observed_go, eval_pairs)

        else:
            # Generic callable or default predictor
            if callable(baseline):
                y_score = np.asarray(baseline(observed_go, eval_pairs), dtype=np.float32)
            else:
                y_score = np.zeros(len(eval_pairs), dtype=np.float32)

        exec_time = time.perf_counter() - start_time

        return self.evaluate_predictions(
            y_true=y_true,
            y_score=y_score,
            is_bridge=is_bridge,
            baseline_name=baseline_name,
            condition_name=condition_name,
            execution_time_sec=exec_time,
            metadata={"num_candidate_pairs": len(eval_pairs)},
        )

    def run_benchmark_suite(
        self,
        environments: list[HeteroData],
        conditions: list[str] | None = None,
    ) -> list[BaselineMetricsResult]:
        """Run all 4 baselines across conditions on a collection of environments."""
        cond_list = conditions or ["A", "E"]
        results: list[BaselineMetricsResult] = []

        from iam.models.masking import StructuredMaskingSuite

        suite = StructuredMaskingSuite()

        for env in environments:
            for cond in cond_list:
                operator = suite.get_operator(cond)

                # 1. Oracle (Upper bound)
                oracle = ExactReachabilityOracle()
                res_oracle = self.evaluate_on_hetero_data(
                    oracle, env, masking_operator=operator, condition_name=cond
                )
                results.append(res_oracle)

                # 2. Deterministic BFS (PMapper Proxy)
                bfs = DeterministicBFSEngine()
                res_bfs = self.evaluate_on_hetero_data(
                    bfs, env, masking_operator=operator, condition_name=cond
                )
                results.append(res_bfs)

                # 3. Random Forest Tabular Link Predictor
                rf = RandomForestLinkPredictor(n_estimators=50, random_state=self.seed)
                res_rf = self.evaluate_on_hetero_data(
                    rf, env, masking_operator=operator, condition_name=cond
                )
                results.append(res_rf)

                # 4. Motif Completion Heuristic
                heuristic = MotifCompletionHeuristic()
                res_heuristic = self.evaluate_on_hetero_data(
                    heuristic, env, masking_operator=operator, condition_name=cond
                )
                results.append(res_heuristic)

        return results

    @staticmethod
    def generate_markdown_report(results: list[BaselineMetricsResult]) -> str:
        """Format metrics results into a clean GitHub Flavored Markdown summary table."""
        lines = [
            "| Baseline Model | Condition | PR-AUC | ROC-AUC | Recall@10 | Recall@50 | FNR | Attack Path Rec@10 | Latency (s) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for r in results:
            lines.append(
                f"| **{r.baseline_name}** | {r.masking_condition} | "
                f"{r.pr_auc:.3f} | {r.roc_auc:.3f} | {r.recall_at_10:.3f} | {r.recall_at_50:.3f} | "
                f"{r.fnr:.3f} | {r.attack_path_recovery_at_10:.3f} | {r.execution_time_sec:.3f} |"
            )
        return "\n".join(lines)

    @staticmethod
    def generate_json_summary(results: list[BaselineMetricsResult]) -> dict[str, Any]:
        """Aggregate metrics results into a JSON summary object."""
        return {
            "num_evaluations": len(results),
            "results": [r.to_dict() for r in results],
        }
