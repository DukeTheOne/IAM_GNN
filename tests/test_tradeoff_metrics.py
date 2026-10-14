"""Unit tests for multi-dimensional security-utility metrics and Pareto frontier analyzer."""

from __future__ import annotations

import csv
import io

import pytest
from scripts.run_tradeoff_experiment import build_canonical_environment

from iam.optimizer.sampler import CandidateEdge, LatentCompletionSampler
from iam.parser.capability import CapabilityModel, load_default_capability_model
from iam.verifier.metrics import (
    DominanceRelation,
    MetricSummary,
    ParetoFrontierAnalyzer,
    TradeoffMetrics,
)
from iam.verifier.tradeoff import (
    RemediationMethod,
    TradeoffConfig,
    TradeoffExperimentRunner,
)


@pytest.fixture
def capability_model() -> CapabilityModel:
    """Load default capability model."""
    return load_default_capability_model()


@pytest.fixture
def sample_metrics_dataset() -> dict[RemediationMethod, TradeoffMetrics]:
    """Create a controlled synthetic dataset of TradeoffMetrics across 6 methods."""
    return {
        RemediationMethod.ORACLE: TradeoffMetrics(
            method=RemediationMethod.ORACLE,
            method_name="Method 0 (Oracle G*)",
            level1_risk=0.0,
            level2_risk=0.0,
            level3_risk=0.0,
            is_safe_gt=True,
            is_misleading=False,
            misleading_gap=0.0,
            t1_breakage=0.0,
            t2_breakage=0.0,
            total_t1_requests=100,
            total_t2_requests=50,
            t1_denied_requests=0,
            t2_denied_requests=0,
            utility_preservation_t1=1.0,
            utility_preservation_t2=1.0,
            num_transformations=1,
            transformation_cost=1.0,
            execution_time_sec=0.015,
            execution_time_ms=15.0,
            has_side_effects=False,
            verification_status="PROVABLY_SAFE",
        ),
        RemediationMethod.NO_ACTION: TradeoffMetrics(
            method=RemediationMethod.NO_ACTION,
            method_name="Method 1 (No Action)",
            level1_risk=0.0,
            level2_risk=0.0,
            level3_risk=1.0,  # Unsafe on G*
            is_safe_gt=False,
            is_misleading=True,
            misleading_gap=1.0,
            t1_breakage=0.0,
            t2_breakage=0.0,
            total_t1_requests=100,
            total_t2_requests=50,
            t1_denied_requests=0,
            t2_denied_requests=0,
            utility_preservation_t1=1.0,
            utility_preservation_t2=1.0,
            num_transformations=0,
            transformation_cost=0.0,
            execution_time_sec=0.001,
            execution_time_ms=1.0,
            has_side_effects=False,
            verification_status="MISLEADING_SAFE",
        ),
        RemediationMethod.COARSE_DELETION: TradeoffMetrics(
            method=RemediationMethod.COARSE_DELETION,
            method_name="Method 2 (Coarse Deletion - IAMPERE)",
            level1_risk=0.0,
            level2_risk=0.0,
            level3_risk=0.0,
            is_safe_gt=True,
            is_misleading=False,
            misleading_gap=0.0,
            t1_breakage=0.25,  # High breakage on historical
            t2_breakage=0.30,  # High breakage on holdout
            total_t1_requests=100,
            total_t2_requests=50,
            t1_denied_requests=25,
            t2_denied_requests=15,
            utility_preservation_t1=0.75,
            utility_preservation_t2=0.70,
            num_transformations=2,
            transformation_cost=10.0,
            execution_time_sec=0.010,
            execution_time_ms=10.0,
            has_side_effects=False,
            verification_status="PROVABLY_SAFE",
        ),
        RemediationMethod.LOG_ONLY_TIGHTENING: TradeoffMetrics(
            method=RemediationMethod.LOG_ONLY_TIGHTENING,
            method_name="Method 3 (Log-Only Tightening - PolicyRefiner)",
            level1_risk=0.0,
            level2_risk=0.0,
            level3_risk=0.0,
            is_safe_gt=True,
            is_misleading=False,
            misleading_gap=0.0,
            t1_breakage=0.0,
            t2_breakage=0.15,  # Fragile under drift
            total_t1_requests=100,
            total_t2_requests=50,
            t1_denied_requests=0,
            t2_denied_requests=8,
            utility_preservation_t1=1.0,
            utility_preservation_t2=0.85,
            num_transformations=1,
            transformation_cost=4.0,
            execution_time_sec=0.012,
            execution_time_ms=12.0,
            has_side_effects=False,
            verification_status="PROVABLY_SAFE",
        ),
        RemediationMethod.UNCERTAINTY_AWARE: TradeoffMetrics(
            method=RemediationMethod.UNCERTAINTY_AWARE,
            method_name="Method 4b (Uncertainty-Aware Repair - Ours)",
            level1_risk=0.0,
            level2_risk=0.0,
            level3_risk=0.0,
            is_safe_gt=True,
            is_misleading=False,
            misleading_gap=0.0,
            t1_breakage=0.0,
            t2_breakage=0.0,  # Optimal utility
            total_t1_requests=100,
            total_t2_requests=50,
            t1_denied_requests=0,
            t2_denied_requests=0,
            utility_preservation_t1=1.0,
            utility_preservation_t2=1.0,
            num_transformations=1,
            transformation_cost=1.0,
            execution_time_sec=0.025,
            execution_time_ms=25.0,
            has_side_effects=False,
            verification_status="PROVABLY_SAFE",
        ),
    }


class TestTradeoffMetricsModel:
    """Tests creation, conversion, and serialization of TradeoffMetrics."""

    def test_metrics_from_dict_and_get_objective(
        self, sample_metrics_dataset: dict[RemediationMethod, TradeoffMetrics]
    ) -> None:
        m4b = sample_metrics_dataset[RemediationMethod.UNCERTAINTY_AWARE]
        assert m4b.get_objective_value("level3_risk") == 0.0
        assert m4b.get_objective_value("t2_breakage") == 0.0
        assert m4b.get_objective_value("transformation_cost") == 1.0

        with pytest.raises(ValueError, match="Unknown objective dimension"):
            m4b.get_objective_value("non_existent_objective")

        with pytest.raises(TypeError, match="is not numeric"):
            m4b.get_objective_value("method_name")

        d = m4b.to_dict()
        assert d["method"] == "uncertainty_aware"
        assert d["level3_risk"] == 0.0
        assert d["is_safe_gt"] is True
        assert d["misleading_gap"] == 0.0


class TestParetoFrontierMathematicalDominance:
    """Formal mathematical validation of Pareto dominance checks."""

    def test_pairwise_dominance_directions(self) -> None:
        """Verify strict dominance, dominated by, equivalence, and incomparability."""
        analyzer = ParetoFrontierAnalyzer(objectives=["level3_risk", "t2_breakage"])

        # Point A: (0.0, 0.0) - Ideal
        point_a = TradeoffMetrics(
            method=RemediationMethod.UNCERTAINTY_AWARE,
            method_name="A",
            level1_risk=0.0,
            level2_risk=0.0,
            level3_risk=0.0,
            is_safe_gt=True,
            is_misleading=False,
            misleading_gap=0.0,
            t1_breakage=0.0,
            t2_breakage=0.0,
            total_t1_requests=10,
            total_t2_requests=10,
            t1_denied_requests=0,
            t2_denied_requests=0,
            utility_preservation_t1=1.0,
            utility_preservation_t2=1.0,
            num_transformations=1,
            transformation_cost=1.0,
            execution_time_sec=0.01,
            execution_time_ms=10.0,
            has_side_effects=False,
            verification_status="PROVABLY_SAFE",
        )

        # Point B: (0.0, 0.2) - Same risk, strictly worse breakage
        point_b = point_a.model_copy(
            update={
                "method": RemediationMethod.LOG_ONLY_TIGHTENING,
                "method_name": "B",
                "t2_breakage": 0.2,
            }
        )

        # Point C: (1.0, 0.0) - Strictly worse risk, same breakage
        point_c = point_a.model_copy(
            update={"method": RemediationMethod.NO_ACTION, "method_name": "C", "level3_risk": 1.0}
        )

        # Point D: (0.5, 0.5) - Strictly worse risk and breakage
        point_d = point_a.model_copy(
            update={
                "method": RemediationMethod.COARSE_DELETION,
                "method_name": "D",
                "level3_risk": 0.5,
                "t2_breakage": 0.5,
            }
        )

        # Point E: (0.8, 0.0) and Point F: (0.0, 0.8) - Mutually incomparable
        point_e = point_a.model_copy(
            update={"method_name": "E", "level3_risk": 0.8, "t2_breakage": 0.0}
        )
        point_f = point_a.model_copy(
            update={"method_name": "F", "level3_risk": 0.0, "t2_breakage": 0.8}
        )

        # A dominates B, C, D
        res_ab = analyzer.compare_points(point_a, point_b)
        assert res_ab.relation == DominanceRelation.DOMINATES
        assert "strictly dominates" in res_ab.explanation

        res_ba = analyzer.compare_points(point_b, point_a)
        assert res_ba.relation == DominanceRelation.DOMINATED_BY

        res_ac = analyzer.compare_points(point_a, point_c)
        assert res_ac.relation == DominanceRelation.DOMINATES

        res_ad = analyzer.compare_points(point_a, point_d)
        assert res_ad.relation == DominanceRelation.DOMINATES

        # A and A are equivalent
        res_aa = analyzer.compare_points(point_a, point_a)
        assert res_aa.relation == DominanceRelation.EQUIVALENT

        # E and F are incomparable
        res_ef = analyzer.compare_points(point_e, point_f)
        assert res_ef.relation == DominanceRelation.INCOMPARABLE

    def test_frontier_extraction_on_dataset(
        self, sample_metrics_dataset: dict[RemediationMethod, TradeoffMetrics]
    ) -> None:
        """Verify non-dominated set extraction on the sample dataset."""
        analyzer = ParetoFrontierAnalyzer(
            objectives=["level3_risk", "t2_breakage", "transformation_cost"]
        )
        points = analyzer.compute_frontier(sample_metrics_dataset)

        by_method = {p.method: p for p in points}

        # Method 4b (Ours) must be Pareto optimal
        assert by_method[RemediationMethod.UNCERTAINTY_AWARE].is_pareto_optimal is True

        # Method 4b must dominate Coarse Deletion and Log-Only Tightening
        assert (
            RemediationMethod.COARSE_DELETION
            in by_method[RemediationMethod.UNCERTAINTY_AWARE].dominates
        )
        assert (
            RemediationMethod.LOG_ONLY_TIGHTENING
            in by_method[RemediationMethod.UNCERTAINTY_AWARE].dominates
        )

        # Coarse Deletion and Log-Only Tightening must be marked dominated
        assert by_method[RemediationMethod.COARSE_DELETION].is_pareto_optimal is False
        assert by_method[RemediationMethod.LOG_ONLY_TIGHTENING].is_pareto_optimal is False

        # Frontier methods extraction
        frontier_methods = analyzer.get_frontier_methods(sample_metrics_dataset)
        assert RemediationMethod.UNCERTAINTY_AWARE in frontier_methods
        assert RemediationMethod.COARSE_DELETION not in frontier_methods


class TestEndToEndTradeoffEvaluationAdvantage:
    """Integration test verifying Pareto frontier analyzer against live TradeoffExperimentRunner outputs."""

    def test_canonical_passrole_uncertainty_aware_advantage_certified(
        self, capability_model: CapabilityModel
    ) -> None:
        """Execute all 6 methods on canonical PassRole motif and prove Method 4b dominance."""
        env = build_canonical_environment(seed=42)

        runner = TradeoffExperimentRunner(
            capability_model=capability_model,
            config=TradeoffConfig(num_completions=15),
        )

        results = runner.run_all_methods(env, seed=42)

        # Multi-objective analyzer focusing on (Level 3 Risk, T2 Breakage, Transformation Cost)
        analyzer = ParetoFrontierAnalyzer(
            objectives=["level3_risk", "t2_breakage", "transformation_cost"]
        )

        cert = analyzer.verify_uncertainty_aware_advantage(results)

        assert cert["all_certified"] is True
        assert cert["is_safe_gt"] is True
        assert cert["is_baseline_conformant"] is True
        assert cert["is_pareto_optimal"] is True
        assert cert["dominates_no_action"] is True
        assert cert["dominates_coarse_deletion"] is True
        assert cert["dominates_log_only_tightening"] is True
        assert cert["level3_risk"] == 0.0
        assert cert["t1_breakage"] == 0.0

    def test_subthreshold_uncertainty_dominates_point_estimate(
        self, capability_model: CapabilityModel
    ) -> None:
        """Under subthreshold link prediction (p = 0.40 < 0.50), Method 4b strictly dominates Method 4a."""
        base_env = build_canonical_environment(seed=42)

        # Subthreshold probability: 0.40 < threshold 0.50
        sub_cand = CandidateEdge(
            source=base_env.candidate_edges[0].source,
            target=base_env.candidate_edges[0].target,
            relation=base_env.candidate_edges[0].relation,
            probability=0.40,
            actions=base_env.candidate_edges[0].actions,
            is_bridge=True,
        )

        sub_env = base_env.model_copy(update={"candidate_edges": [sub_cand]}, deep=True)

        runner = TradeoffExperimentRunner(
            capability_model=capability_model,
            config=TradeoffConfig(num_completions=15, point_estimate_threshold=0.50),
        )

        sampler = LatentCompletionSampler(
            observed_graph=sub_env.observed_graph, candidate_edges=[sub_cand]
        )
        completions = sampler.sample_completions(num_samples=15, seed=42)

        res_4a = runner.run_method_4a_point_estimate(sub_env, completions)
        res_4b = runner.run_method_4b_uncertainty_aware(sub_env, completions)

        # Point estimate drops edge -> residual risk on G* = 1.0
        assert res_4a.level3_risk == 1.0
        # Uncertainty-aware repairs -> risk on G* = 0.0
        assert res_4b.level3_risk == 0.0

        # Dominance comparison on (Level 3 Risk, T2 Breakage)
        analyzer = ParetoFrontierAnalyzer(objectives=["level3_risk", "t2_breakage"])
        m_4a = TradeoffMetrics.from_method_result(res_4a)
        m_4b = TradeoffMetrics.from_method_result(res_4b)

        comparison = analyzer.compare_points(m_4b, m_4a)
        assert comparison.relation == DominanceRelation.DOMINATES
        assert "strictly dominates" in comparison.explanation
        assert comparison.dimensional_deltas["level3_risk"] == -1.0


class TestMultiRunAggregationAndRendering:
    """Tests MetricSummary multi-run aggregation and Markdown/CSV/ASCII export."""

    def test_metric_summary_aggregation(
        self, sample_metrics_dataset: dict[RemediationMethod, TradeoffMetrics]
    ) -> None:
        # Create two runs with slight variations
        run1 = sample_metrics_dataset
        run2 = {
            k: v.model_copy(update={"execution_time_ms": v.execution_time_ms + 2.0})
            for k, v in sample_metrics_dataset.items()
        }

        # Need MethodResult-like or TradeoffMetrics
        summaries = MetricSummary.aggregate([run1, run2])

        assert len(summaries) == len(sample_metrics_dataset)
        m4b_summary = summaries[RemediationMethod.UNCERTAINTY_AWARE]
        assert m4b_summary.num_runs == 2
        assert m4b_summary.level3_risk_mean == 0.0
        assert m4b_summary.t1_breakage_mean == 0.0
        assert m4b_summary.pareto_optimal_rate == 1.0

        coarse_summary = summaries[RemediationMethod.COARSE_DELETION]
        assert coarse_summary.pareto_optimal_rate == 0.0

    def test_rendering_outputs(
        self, sample_metrics_dataset: dict[RemediationMethod, TradeoffMetrics]
    ) -> None:
        analyzer = ParetoFrontierAnalyzer()
        points = analyzer.compute_frontier(sample_metrics_dataset)

        # 1. Markdown Table
        md_table = analyzer.render_markdown_table(points)
        assert "| Remediation Method | Status |" in md_table
        assert "**[PARETO OPTIMAL]**" in md_table
        assert "_[DOMINATED]_" in md_table

        # 2. CSV
        csv_str = analyzer.render_csv(points)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == len(points) + 1  # Header + data
        assert rows[0][0] == "method"

        # 3. ASCII Plot
        ascii_plot = analyzer.render_ascii_tradeoff_plot(points)
        assert "Security Risk" in ascii_plot
        assert "★: Method 4b" in ascii_plot
        assert "0: Method 0" in ascii_plot
