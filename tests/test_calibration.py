"""Tests for TemperatureScalingCalibrator, ECE metrics, and reliability diagrams."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from iam.generator.topology import EnterpriseTopologyConfig, EnterpriseTopologyGenerator
from iam.models.calibration import (
    CalibrationBin,
    CalibrationResult,
    TemperatureScalingCalibrator,
    compute_ece,
    compute_nll,
)
from iam.models.converter import IAMHeteroDataConverter
from iam.models.decoder import BilinearRelationPredictor, DecoderConfig
from iam.models.rgcn import HeteroRGCNEncoder, RGCNConfig, get_default_in_channels_dict
from iam.parser.capability import CapabilityModel, load_default_capability_model


@pytest.fixture
def capability_model() -> CapabilityModel:
    """Fixture providing loaded AWS CapabilityModel."""
    return load_default_capability_model()


class TestCalibrationMetrics:
    """Validates standalone ECE, MCE, and NLL metric routines."""

    def test_perfectly_calibrated_ece_near_zero(self) -> None:
        """Verify perfectly calibrated probabilities yield near-zero ECE."""
        # 100 samples with p=0.8, exactly 80 positive
        # 100 samples with p=0.2, exactly 20 positive
        probs = np.array([0.8] * 100 + [0.2] * 100)
        targets = np.array([1.0] * 80 + [0.0] * 20 + [1.0] * 20 + [0.0] * 80)

        ece, mce, bins = compute_ece(probs, targets, num_bins=10)
        assert ece < 0.01
        assert mce < 0.01
        assert len(bins) == 10

    def test_uncalibrated_severe_ece(self) -> None:
        """Verify severely overconfident probabilities yield high ECE and MCE."""
        # Predictions are 0.99 confident, but actual positive rate is only 50%
        probs = np.array([0.99] * 100)
        targets = np.array([1.0] * 50 + [0.0] * 50)

        ece, mce, bins = compute_ece(probs, targets, num_bins=10)
        # Expected gap is ~ |0.99 - 0.50| = 0.49
        assert ece > 0.45
        assert mce > 0.45
        assert bins[9].sample_count == 100

    def test_compute_nll_numerical_values(self) -> None:
        """Verify binary Negative Log-Likelihood computation."""
        probs = torch.tensor([0.9, 0.1])
        targets = torch.tensor([1.0, 0.0])
        # NLL = - (log(0.9) + log(0.9)) / 2 = -log(0.9) ~ 0.10536
        nll = compute_nll(probs, targets)
        assert abs(nll - 0.10536) < 1e-3

    def test_empty_inputs_return_zero(self) -> None:
        """Verify empty inputs return 0.0 for ECE and NLL."""
        ece, mce, bins = compute_ece(np.array([]), np.array([]))
        assert ece == 0.0
        assert mce == 0.0
        assert bins == []
        assert compute_nll(np.array([]), np.array([])) == 0.0

    def test_invalid_num_bins_raises_error(self) -> None:
        """Verify non-positive num_bins raises ValueError."""
        with pytest.raises(ValueError, match="num_bins must be positive"):
            compute_ece(np.array([0.5]), np.array([1.0]), num_bins=0)


class TestTemperatureScalingOptimization:
    """Validates temperature fitting via NLL optimization and probability scaling."""

    def test_temperature_optimization_on_overconfident_logits(self) -> None:
        """Verify fitting on overconfident logits yields T > 1 and reduces ECE."""
        torch.manual_seed(42)
        # Generate artificially overconfident logits (large magnitude) with balanced labels
        logits = torch.randn(200) * 5.0
        targets = (torch.rand(200) > 0.5).float()

        calibrator = TemperatureScalingCalibrator(initial_temperature=1.0)
        res = calibrator.fit(logits, targets)

        assert calibrator.is_fitted is True
        # Overconfidence requires softening (T > 1.0)
        assert res.temperature > 1.0
        # Calibrated ECE should be noticeably lower than uncalibrated ECE
        assert res.calibrated_ece < res.uncalibrated_ece
        assert res.calibrated_nll <= res.uncalibrated_nll

    def test_temperature_optimization_on_underconfident_logits(self) -> None:
        """Verify fitting on underconfident logits (small magnitude) yields T < 1."""
        torch.manual_seed(101)
        # Small magnitude logits near 0 (probs near 0.5) but labels are highly separable
        targets = torch.cat([torch.ones(100), torch.zeros(100)])
        logits = torch.cat([torch.ones(100) * 0.2, -torch.ones(100) * 0.2])

        calibrator = TemperatureScalingCalibrator(initial_temperature=1.0)
        res = calibrator.fit(logits, targets)

        # Underconfidence requires sharpening (T < 1.0)
        assert res.temperature < 1.0
        assert res.calibrated_nll < res.uncalibrated_nll

    def test_calibrator_predict_proba_and_calibrate_probabilities_consistency(self) -> None:
        """Verify consistency between predict_proba from logits and calibrate_probabilities."""
        calibrator = TemperatureScalingCalibrator(initial_temperature=2.0)
        logits = torch.tensor([-2.0, 0.0, 2.0])
        probs = torch.sigmoid(logits)

        cal_from_logits = calibrator.predict_proba(logits)
        cal_from_probs = calibrator.calibrate_probabilities(probs)

        torch.testing.assert_close(cal_from_logits, cal_from_probs, atol=1e-3, rtol=1e-3)

    def test_fit_on_empty_raises_error(self) -> None:
        """Verify fit on empty inputs raises ValueError."""
        calibrator = TemperatureScalingCalibrator()
        with pytest.raises(ValueError, match="Cannot fit calibrator on empty logits"):
            calibrator.fit(torch.empty(0), torch.empty(0))

    def test_invalid_initial_temperature_raises_error(self) -> None:
        """Verify non-positive initial temperature raises ValueError."""
        with pytest.raises(ValueError, match="initial_temperature must be positive"):
            TemperatureScalingCalibrator(initial_temperature=0.0)

        with pytest.raises(ValueError, match="num_bins must be positive"):
            TemperatureScalingCalibrator(num_bins=0)


class TestCalibrationReporting:
    """Validates serialization and reliability diagram reporting."""

    def test_to_dict_serialization(self) -> None:
        """Verify structured dictionary serialization."""
        bin_example = CalibrationBin(
            bin_lower=0.0,
            bin_upper=0.1,
            sample_count=10,
            mean_confidence=0.05,
            mean_accuracy=0.04,
            calibration_gap=0.01,
        )
        res = CalibrationResult(
            temperature=1.45,
            uncalibrated_ece=0.15,
            calibrated_ece=0.04,
            uncalibrated_nll=0.55,
            calibrated_nll=0.42,
            uncalibrated_mce=0.25,
            calibrated_mce=0.08,
            sample_count=100,
            bins_uncalibrated=[bin_example],
            bins_calibrated=[bin_example],
        )
        d = res.to_dict()
        assert d["temperature"] == 1.45
        assert d["uncalibrated_ece"] == 0.15
        assert d["calibrated_ece"] == 0.04
        assert d["ece_reduction_pct"] > 70.0
        assert len(d["bins_calibrated"]) == 1

    def test_generate_markdown_diagram(self) -> None:
        """Verify markdown reliability diagram produces valid table format."""
        bin_example = CalibrationBin(
            bin_lower=0.0,
            bin_upper=0.1,
            sample_count=25,
            mean_confidence=0.06,
            mean_accuracy=0.05,
            calibration_gap=0.01,
        )
        res = CalibrationResult(
            temperature=1.2,
            uncalibrated_ece=0.12,
            calibrated_ece=0.03,
            uncalibrated_nll=0.40,
            calibrated_nll=0.35,
            uncalibrated_mce=0.18,
            calibrated_mce=0.05,
            sample_count=200,
            bins_uncalibrated=[bin_example],
            bins_calibrated=[bin_example],
        )
        md = res.generate_markdown_diagram()
        assert "### Reliability Diagram & Uncertainty Calibration Summary" in md
        assert "| Confidence Bin | Samples |" in md
        assert "`[0.00, 0.10)`" in md


class TestInductiveValidationCalibration:
    """Validates calibration on an actual inductive enterprise IAM environment."""

    def test_calibration_on_inductive_environment_predictions(
        self, capability_model: CapabilityModel
    ) -> None:
        """Verify end-to-end calibration on predictions from HeteroRGCNEncoder and BilinearPredictor."""
        # 1. Generate validation environment
        topo_cfg = EnterpriseTopologyConfig(
            num_nodes=180,
            seed=777,
            num_pe_chains=2,
            account_id="123456789012",
            organization_id="o-val-calib",
        )
        generator = EnterpriseTopologyGenerator(config=topo_cfg, capability_model=capability_model)
        graph = generator.generate()

        converter = IAMHeteroDataConverter()
        data = converter.convert(graph)

        # 2. Forward through RGCN encoder and bilinear decoder
        encoder = HeteroRGCNEncoder(
            RGCNConfig(in_channels_dict=get_default_in_channels_dict(), hidden_channels=64)
        )
        decoder = BilinearRelationPredictor(DecoderConfig(hidden_channels=64))
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            h_dict = encoder(data)

            # Evaluate AssumesRole relations
            edge_type = ("User", "AssumesRole", "Role")
            pos_edges = data[edge_type].edge_index
            rel_id = decoder.get_relation_id(edge_type)

            num_pos = pos_edges.size(1)
            # Create synthetic negative pairs
            neg_src = torch.randint(0, data["User"].x.size(0), (num_pos,))
            neg_dst = torch.randint(0, data["Role"].x.size(0), (num_pos,))
            neg_edges = torch.stack([neg_src, neg_dst], dim=0)

            all_src = torch.cat([pos_edges[0], neg_edges[0]])
            all_dst = torch.cat([pos_edges[1], neg_edges[1]])
            labels = torch.cat([torch.ones(num_pos), torch.zeros(num_pos)])

            h_u = h_dict["User"][all_src]
            h_v = h_dict["Role"][all_dst]
            logits = decoder(h_u, h_v, rel_id)

        # 3. Fit calibrator on validation predictions
        calibrator = TemperatureScalingCalibrator()
        result = calibrator.fit(logits, labels)

        assert calibrator.is_fitted is True
        assert result.temperature > 0.0
        assert result.sample_count == len(labels)
        # Temperature scaling NLL must not exceed uncalibrated NLL
        assert result.calibrated_nll <= result.uncalibrated_nll + 1e-4
