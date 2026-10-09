"""Test suite for Baseline Benchmark Suite and evaluation metrics.

Validates PR-AUC, ROC-AUC, Recall@K, FNR, MRR, Attack-Path Recovery,
and end-to-end multi-baseline benchmarking under Conditions A and E.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from torch_geometric.data import HeteroData

from iam.baselines.bfs import DeterministicBFSEngine
from iam.baselines.metrics import (
    BaselineBenchmarkSuite,
    BaselineMetricsResult,
    compute_attack_path_recovery,
    compute_false_negative_rate,
    compute_mean_reciprocal_rank,
    compute_pr_auc,
    compute_recall_at_k,
    compute_roc_auc,
)
from iam.baselines.motif_heuristic import MotifCompletionHeuristic
from iam.baselines.oracle import ExactReachabilityOracle
from iam.baselines.random_forest import RandomForestLinkPredictor
from iam.generator.graph import IAMGraph
from iam.generator.motifs import PassRoleLambdaMotif
from iam.generator.topology import EnterpriseTopologyConfig, EnterpriseTopologyGenerator
from iam.models.converter import IAMHeteroDataConverter
from iam.models.vocabulary import ActionVocabulary
from iam.parser import CapabilityModel, load_default_capability_model


@pytest.fixture
def capability_model() -> CapabilityModel:
    """Fixture providing loaded AWS CapabilityModel."""
    return load_default_capability_model()


@pytest.fixture
def enterprise_graph(capability_model: CapabilityModel) -> IAMGraph:
    """Fixture generating a deterministic enterprise topology with injected PE chains."""
    cfg = EnterpriseTopologyConfig(
        num_nodes=120,
        seed=42,
        num_pe_chains=2,
        include_branching_chains=True,
        num_branching_chains=1,
    )
    generator = EnterpriseTopologyGenerator(config=cfg, capability_model=capability_model)
    return generator.generate()


@pytest.fixture
def enterprise_hetero(enterprise_graph: IAMGraph) -> HeteroData:
    """Fixture converting enterprise graph to PyG HeteroData."""
    vocab = ActionVocabulary(
        actions=["iam:PassRole", "lambda:CreateFunction", "sts:AssumeRole", "s3:GetObject"]
    )
    converter = IAMHeteroDataConverter(vocabulary=vocab, include_reverse_edges=True)
    return converter.convert(enterprise_graph)


class TestMetricFunctions:
    """Unit tests for standalone metric functions."""

    def test_pr_auc_calculation(self) -> None:
        """Verify PR-AUC computation under normal, perfect, and degenerate cases."""
        y_true = np.array([1, 1, 0, 0], dtype=np.int32)
        y_perfect = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
        assert compute_pr_auc(y_true, y_perfect) == pytest.approx(1.0, abs=1e-3)

        # Degenerate cases
        assert compute_pr_auc(np.array([], dtype=np.int32), np.array([], dtype=np.float32)) == 0.0
        assert compute_pr_auc(np.array([0, 0], dtype=np.int32), np.array([0.5, 0.5])) == 0.0
        assert compute_pr_auc(np.array([1, 1], dtype=np.int32), np.array([0.5, 0.5])) == 1.0

    def test_roc_auc_calculation(self) -> None:
        """Verify ROC-AUC computation."""
        y_true = np.array([1, 1, 0, 0], dtype=np.int32)
        y_perfect = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
        assert compute_roc_auc(y_true, y_perfect) == pytest.approx(1.0, abs=1e-3)

        # Single class fallback
        assert compute_roc_auc(np.array([1, 1]), np.array([0.5, 0.5])) == 0.5
        assert compute_roc_auc(np.array([]), np.array([])) == 0.5

    def test_recall_at_k(self) -> None:
        """Verify Recall@K ranking metric."""
        y_true = np.array([1, 0, 1, 0], dtype=np.int32)
        y_scores = np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float32)

        # Top 1: index 0 (true positive) -> recall = 1/2 = 0.5
        assert compute_recall_at_k(y_true, y_scores, k=1) == pytest.approx(0.5)
        # Top 3: indices 0, 1, 2 (contains both positives) -> recall = 2/2 = 1.0
        assert compute_recall_at_k(y_true, y_scores, k=3) == pytest.approx(1.0)
        # Empty positives
        assert compute_recall_at_k(np.array([0, 0]), np.array([0.5, 0.5]), k=1) == 1.0

    def test_false_negative_rate(self) -> None:
        """Verify FNR calculation at threshold."""
        y_true = np.array([1, 1, 0, 0], dtype=np.int32)
        y_scores = np.array([0.8, 0.3, 0.1, 0.2], dtype=np.float32)

        # At threshold 0.5, 1 TP, 1 FN -> FNR = 1/2 = 0.5
        assert compute_false_negative_rate(y_true, y_scores, threshold=0.5) == pytest.approx(0.5)
        # At threshold 0.2, 2 TP, 0 FN -> FNR = 0.0
        assert compute_false_negative_rate(y_true, y_scores, threshold=0.2) == pytest.approx(0.0)

    def test_mean_reciprocal_rank(self) -> None:
        """Verify MRR calculation."""
        y_true = np.array([0, 1, 1, 0], dtype=np.int32)
        y_scores = np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float32)
        # Best prediction at rank 1 is negative (index 0). Rank 2 is positive (index 1) -> MRR = 1/2 = 0.5
        assert compute_mean_reciprocal_rank(y_true, y_scores) == pytest.approx(0.5)

        # No positives
        assert compute_mean_reciprocal_rank(np.array([0, 0]), np.array([0.5, 0.5])) == 0.0

    def test_attack_path_recovery(self) -> None:
        """Verify Attack-Path Recovery Rate calculation."""
        # 3 severed attack paths with bridge edge ranks: 5, 25, 120
        ranks = [5, 25, 120]
        assert compute_attack_path_recovery(ranks, k=10) == pytest.approx(1.0 / 3.0)
        assert compute_attack_path_recovery(ranks, k=50) == pytest.approx(2.0 / 3.0)
        assert compute_attack_path_recovery(ranks, k=150) == pytest.approx(3.0 / 3.0)
        assert compute_attack_path_recovery([], k=10) == 0.0


class TestBaselineBenchmarkSuite:
    """Integration tests for BaselineBenchmarkSuite."""

    def test_evaluate_predictions_bundle(self) -> None:
        """Verify evaluate_predictions returns complete BaselineMetricsResult."""
        suite = BaselineBenchmarkSuite()
        y_true = np.array([1, 1, 0, 0, 0], dtype=np.int32)
        y_score = np.array([0.9, 0.7, 0.3, 0.2, 0.1], dtype=np.float32)
        is_bridge = np.array([True, False, False, False, False], dtype=bool)

        res = suite.evaluate_predictions(
            y_true=y_true,
            y_score=y_score,
            is_bridge=is_bridge,
            baseline_name="TestModel",
            condition_name="ConditionA",
            execution_time_sec=0.042,
        )

        assert isinstance(res, BaselineMetricsResult)
        assert res.baseline_name == "TestModel"
        assert res.masking_condition == "ConditionA"
        assert res.pr_auc == pytest.approx(1.0, abs=1e-3)
        assert res.num_samples == 5
        assert res.num_positives == 2
        assert res.num_negatives == 3
        assert res.attack_path_recovery_at_10 == pytest.approx(1.0)

        # JSON serialization roundtrip
        d = res.to_dict()
        json_str = json.dumps(d)
        assert "TestModel" in json_str

    def test_evaluate_on_hetero_data_condition_a(self, enterprise_hetero: HeteroData) -> None:
        """Verify evaluation on HeteroData under Condition A (Random Masking)."""
        suite = BaselineBenchmarkSuite(seed=42)

        # 1. Oracle
        oracle = ExactReachabilityOracle()
        res_oracle = suite.evaluate_on_hetero_data(oracle, enterprise_hetero, condition_name="A")
        assert res_oracle.pr_auc == pytest.approx(1.0, abs=1e-3)

        # 2. Motif Heuristic
        heuristic = MotifCompletionHeuristic()
        res_heuristic = suite.evaluate_on_hetero_data(
            heuristic, enterprise_hetero, condition_name="A"
        )
        assert 0.0 <= res_heuristic.pr_auc <= 1.0
        assert res_heuristic.num_samples > 0

        # 3. Random Forest
        rf = RandomForestLinkPredictor(n_estimators=10, random_state=42)
        res_rf = suite.evaluate_on_hetero_data(rf, enterprise_hetero, condition_name="A")
        assert 0.0 <= res_rf.pr_auc <= 1.0

    def test_evaluate_on_hetero_data_condition_e_blindness(
        self, enterprise_graph: IAMGraph
    ) -> None:
        """Verify that under Condition E, Deterministic BFS drops to 0.0% attack path recovery."""
        # Inject explicit PassRole motif to guarantee at least 1 critical bridge
        motif = PassRoleLambdaMotif()
        motif.inject(enterprise_graph, suffix="cond_e_bench")

        vocab = ActionVocabulary(
            actions=["iam:PassRole", "lambda:CreateFunction", "sts:AssumeRole", "s3:GetObject"]
        )
        converter = IAMHeteroDataConverter(vocabulary=vocab, include_reverse_edges=True)
        data = converter.convert(enterprise_graph)

        suite = BaselineBenchmarkSuite(seed=42)
        bfs = DeterministicBFSEngine()
        res_bfs = suite.evaluate_on_hetero_data(bfs, data, condition_name="E")

        assert res_bfs.baseline_name == "DeterministicBFSEngine"
        # Deterministic BFS is blind to severed bridge edge in Go
        assert res_bfs.attack_path_recovery_at_10 == 0.0

    def test_run_benchmark_suite_and_reporting(self, enterprise_hetero: HeteroData) -> None:
        """Verify full benchmark suite runs across multiple baselines and formats Markdown report."""
        suite = BaselineBenchmarkSuite(seed=42)
        results = suite.run_benchmark_suite(
            environments=[enterprise_hetero],
            conditions=["A"],
        )

        # 4 baselines evaluated for 1 environment under Condition A
        assert len(results) == 4
        baseline_names = {r.baseline_name for r in results}
        assert "ExactReachabilityOracle" in baseline_names
        assert "DeterministicBFSEngine" in baseline_names
        assert "RandomForestLinkPredictor" in baseline_names
        assert "MotifCompletionHeuristic" in baseline_names

        # Markdown report validation
        md_report = BaselineBenchmarkSuite.generate_markdown_report(results)
        assert "| Baseline Model | Condition |" in md_report
        assert "ExactReachabilityOracle" in md_report

        # JSON summary validation
        json_summary = BaselineBenchmarkSuite.generate_json_summary(results)
        assert json_summary["num_evaluations"] == 4
        assert len(json_summary["results"]) == 4
