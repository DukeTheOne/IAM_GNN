"""Unit and integration test suite for Step 5.5: Thin-Slice Experiment Runner.

Validates end-to-end environment synthesis, Random Forest baseline fitting,
RGCN model training and calibration, multi-baseline evaluation under Conditions E and A,
plot generation, and Markdown/JSON report synthesis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from torch_geometric.data import HeteroData

from iam.models.thin_slice import (
    ThinSliceConfig,
    ThinSliceExperimentResult,
    ThinSliceExperimentRunner,
)


class TestThinSliceConfig:
    """Tests for ThinSliceConfig dataclass."""

    def test_default_initialization(self) -> None:
        """Verify default configuration attributes and path normalization."""
        cfg = ThinSliceConfig()
        assert cfg.num_epochs == 25
        assert cfg.lr == 1e-3
        assert cfg.hidden_channels == 128
        assert cfg.num_bases == 8
        assert "E" in cfg.conditions
        assert "A" in cfg.conditions
        assert isinstance(cfg.output_dir, Path)
        assert cfg.plot_filename == "thin_slice_comparison.png"

    def test_string_path_conversion(self) -> None:
        """Verify string path is coerced to Path in post_init."""
        cfg = ThinSliceConfig(output_dir="custom/experiments/dir")  # type: ignore[arg-type]
        assert isinstance(cfg.output_dir, Path)
        assert str(cfg.output_dir) == str(Path("custom/experiments/dir"))


class TestThinSliceExperimentRunner:
    """Integration tests for ThinSliceExperimentRunner."""

    def test_generate_environments_disjointness(self) -> None:
        """Verify generated train, val, and test environments have zero entity leakage."""
        cfg = ThinSliceConfig(
            train_seeds=[11, 12],
            val_seed=21,
            test_seeds=[31],
            num_nodes_train=40,
            num_nodes_val=40,
            num_nodes_test=40,
        )
        runner = ThinSliceExperimentRunner(config=cfg)
        train_envs, val_envs, test_envs = runner.generate_environments()

        assert len(train_envs) == 2
        assert len(val_envs) == 1
        assert len(test_envs) == 1

        for env in train_envs + val_envs + test_envs:
            assert isinstance(env, HeteroData)
            assert "User" in env.node_types
            assert "Role" in env.node_types

        # Check account and org ID disjointness
        orgs = [e.organization_id for e in train_envs + val_envs + test_envs]
        assert len(orgs) == len(set(orgs))

    def test_fast_thin_slice_run(self, tmp_path: Path) -> None:
        """Verify full thin slice experiment executes cleanly and produces all expected artifacts."""
        cfg = ThinSliceConfig(
            train_seeds=[101],
            val_seed=201,
            test_seeds=[301],
            num_nodes_train=45,
            num_nodes_val=45,
            num_nodes_test=45,
            num_epochs=3,
            conditions=["E", "A"],
            output_dir=tmp_path / "thin_slice_test",
            seed=42,
        )
        runner = ThinSliceExperimentRunner(config=cfg)
        result = runner.run()

        assert isinstance(result, ThinSliceExperimentResult)
        assert result.execution_time_sec > 0.0
        assert result.calibrated_temperature > 0.0
        assert 0.0 <= result.val_ece_before <= 1.0
        assert 0.0 <= result.val_ece_after <= 1.0

        # Verify all 6 models were evaluated for both conditions
        assert len(result.metrics_results) > 0
        baseline_names = {r.baseline_name for r in result.metrics_results}
        assert "ExactReachabilityOracle" in baseline_names
        assert "DeterministicBFSEngine" in baseline_names
        assert "RandomForestLinkPredictor" in baseline_names
        assert "MotifCompletionHeuristic" in baseline_names
        assert "RGCN (Uncalibrated)" in baseline_names
        assert "RGCN (Calibrated)" in baseline_names

        # Verify Condition E deterministic BFS blindness (0.0% path recovery)
        bfs_e = [
            r
            for r in result.metrics_results
            if r.baseline_name == "DeterministicBFSEngine" and "Condition_E" in r.masking_condition
        ]
        assert len(bfs_e) > 0
        assert bfs_e[0].attack_path_recovery_at_10 == pytest.approx(0.0)

        # Verify artifacts exist on disk
        plot_path = Path(result.output_files["plot"])
        report_path = Path(result.output_files["report"])
        json_path = Path(result.output_files["json"])

        assert plot_path.exists()
        assert plot_path.stat().st_size > 1000  # Non-trivial image file

        assert report_path.exists()
        report_text = report_path.read_text(encoding="utf-8")
        assert "# Step 5.5: Thin-Slice Benchmark Report" in report_text
        assert "DeterministicBFSEngine" in report_text
        assert "RGCN (Calibrated)" in report_text
        assert "Thin-Slice Gate Decision: **GO**" in report_text

        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["num_results"] == len(result.metrics_results)
        assert "temperature" in data
