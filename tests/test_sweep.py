"""Unit and integration test suite for Phase 2 Week 6: Multi-Seed Sweep & Gate G0.

Validates SweepConfig initialization, confidence interval calculation, feature ablation
masking, multi-model evaluation, and end-to-end fast sweep execution and artifact generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from iam.models.sweep import (
    ExperimentalSweepRunner,
    MetricSummary,
    SweepConfig,
    compute_confidence_interval,
    mask_centrality_features,
)


class TestSweepConfig:
    """Tests for SweepConfig dataclass."""

    def test_default_initialization(self) -> None:
        """Verify default configuration attributes and path normalization."""
        cfg = SweepConfig()
        assert len(cfg.seeds) == 5
        assert 101 in cfg.seeds
        assert "A_20" in cfg.conditions
        assert "E" in cfg.conditions
        assert "B" in cfg.conditions
        assert cfg.num_epochs == 25
        assert isinstance(cfg.output_dir, Path)
        assert cfg.run_ablations is True

    def test_string_path_conversion(self) -> None:
        """Verify string output directory is coerced to Path."""
        cfg = SweepConfig(output_dir="custom/sweep/dir")  # type: ignore[arg-type]
        assert isinstance(cfg.output_dir, Path)
        assert (
            str(cfg.output_dir) == "custom\\sweep\\dir" or str(cfg.output_dir) == "custom/sweep/dir"
        )


class TestConfidenceIntervals:
    """Tests for Student's t confidence interval computations."""

    def test_identical_values(self) -> None:
        """Verify zero variance for identical value arrays."""
        vals = [1.0, 1.0, 1.0, 1.0, 1.0]
        mean, std, ci_low, ci_high = compute_confidence_interval(vals)
        assert mean == 1.0
        assert std == 0.0
        assert ci_low == 1.0
        assert ci_high == 1.0

    def test_varying_values(self) -> None:
        """Verify valid interval calculation for sample distributions."""
        vals = [0.80, 0.85, 0.90, 0.82, 0.88]
        mean, std, ci_low, ci_high = compute_confidence_interval(vals)
        assert 0.84 <= mean <= 0.86
        assert std > 0.0
        assert ci_low < mean < ci_high
        assert 0.0 <= ci_low <= 1.0
        assert 0.0 <= ci_high <= 1.0

    def test_single_value(self) -> None:
        """Verify graceful single value fallback."""
        mean, std, ci_low, ci_high = compute_confidence_interval([0.75])
        assert mean == 0.75
        assert std == 0.0
        assert ci_low == 0.75
        assert ci_high == 0.75

    def test_empty_list(self) -> None:
        """Verify empty value handling."""
        mean, std, ci_low, ci_high = compute_confidence_interval([])
        assert mean == 0.0
        assert std == 0.0


class TestFeatureAblationMasking:
    """Tests for centrality zero-masking feature ablation."""

    def test_mask_centrality_features(self) -> None:
        """Verify centrality dimensions are zeroed while permissions remain intact."""
        torch.manual_seed(42)
        x_dict = {
            "User": torch.ones(3, 51),
            "Role": torch.ones(3, 52),
            "Policy": torch.ones(3, 39),
            "Group": torch.ones(3, 42),
            "Resource": torch.ones(3, 20),
        }
        ablated = mask_centrality_features(x_dict)

        # Check User: centrality [9:20] zeroed, permissions [20:] preserved
        assert (ablated["User"][:, 9:20] == 0.0).all()
        assert (ablated["User"][:, 0:9] == 1.0).all()
        assert (ablated["User"][:, 20:] == 1.0).all()

        # Check Role: centrality [10:21] zeroed, others preserved
        assert (ablated["Role"][:, 10:21] == 0.0).all()
        assert (ablated["Role"][:, 0:10] == 1.0).all()

        # Check Policy: centrality [5:8] zeroed
        assert (ablated["Policy"][:, 5:8] == 0.0).all()
        assert (ablated["Policy"][:, 0:5] == 1.0).all()


class TestExperimentalSweepRunner:
    """Integration test verifying end-to-end multi-seed sweep execution."""

    def test_fast_sweep_execution(self, tmp_path: Path) -> None:
        """Verify multi-seed sweep execution in lightweight test mode."""
        cfg = SweepConfig(
            seeds=[101, 102],
            conditions=["A_20", "E"],
            num_epochs=2,
            num_nodes_train=30,
            num_nodes_val=30,
            num_nodes_test=30,
            train_envs_per_seed=1,
            val_envs_per_seed=1,
            test_envs_per_seed=1,
            output_dir=tmp_path / "sweep_test",
            run_ablations=True,
        )

        runner = ExperimentalSweepRunner(config=cfg)
        bundle = runner.run_sweep()

        assert bundle.gate_decision in ("GO", "REFRAME", "NEGATIVE_RESULT")
        assert len(bundle.gate_rationale) > 0
        assert "E" in bundle.aggregated_metrics
        assert "A_20" in bundle.aggregated_metrics

        # Verify RGCN and baselines computed metrics
        cond_e_metrics = bundle.aggregated_metrics["E"]
        assert "Oracle" in cond_e_metrics
        assert "BFS" in cond_e_metrics
        assert "RandomForest" in cond_e_metrics
        assert "MotifHeuristic" in cond_e_metrics
        assert "RGCN_Calibrated" in cond_e_metrics
        assert "UntypedGCN" in cond_e_metrics
        assert "RGCN_PermissionsOnly" in cond_e_metrics

        rgcn_summary = cond_e_metrics["RGCN_Calibrated"]["pr_auc"]
        assert isinstance(rgcn_summary, MetricSummary)
        assert 0.0 <= rgcn_summary.mean <= 1.0

        # Verify output files generated on disk
        out_dir = tmp_path / "sweep_test"
        assert (out_dir / "sweep_results.json").exists()
        assert (out_dir / "sweep_report.md").exists()
        assert (out_dir / "sweep_conditions_comparison.png").exists()
        assert (out_dir / "sweep_ablations_comparison.png").exists()

        # Validate JSON serialization roundtrip
        with open(out_dir / "sweep_results.json", encoding="utf-8") as f:
            data = json.load(f)
            assert data["gate_decision"] == bundle.gate_decision
            assert "E" in data["aggregated_metrics"]
