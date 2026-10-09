"""Tests for FocalLoss, RGCNLinkPredictionModel, and InductiveLinkPredictionTrainer."""

from __future__ import annotations

import pytest
import torch

from iam.generator.topology import EnterpriseTopologyConfig, EnterpriseTopologyGenerator
from iam.models.converter import IAMHeteroDataConverter
from iam.models.decoder import BilinearRelationPredictor, DecoderConfig
from iam.models.masking import MaskingCondition
from iam.models.rgcn import HeteroRGCNEncoder, RGCNConfig, get_default_in_channels_dict
from iam.models.training import (
    FocalLoss,
    InductiveLinkPredictionTrainer,
    RGCNLinkPredictionModel,
    TrainingConfig,
    TrainingHistory,
)
from iam.parser.capability import CapabilityModel, load_default_capability_model


@pytest.fixture
def capability_model() -> CapabilityModel:
    """Fixture providing loaded AWS CapabilityModel."""
    return load_default_capability_model()


class TestFocalLoss:
    """Validates mathematical properties and edge cases of Focal Loss."""

    def test_easy_vs_hard_example_downweighting(self) -> None:
        """Verify easy samples produce orders of magnitude lower loss than hard samples."""
        loss_fn = FocalLoss(gamma=2.0, alpha=0.25, reduction="none")

        logits = torch.tensor([5.0, 0.2, -5.0, -0.2])
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])

        losses = loss_fn(logits, targets)

        easy_pos_loss = losses[0].item()
        hard_pos_loss = losses[1].item()
        easy_neg_loss = losses[2].item()
        hard_neg_loss = losses[3].item()

        # Easy positive has much lower loss than hard positive
        assert easy_pos_loss < hard_pos_loss / 100.0
        # Easy negative has much lower loss than hard negative
        assert easy_neg_loss < hard_neg_loss / 100.0

    def test_alpha_balancing_effect(self) -> None:
        """Verify alpha parameter balances positive vs. negative sample loss contributions."""
        loss_fn_high_alpha = FocalLoss(gamma=0.0, alpha=0.75, reduction="none")
        loss_fn_low_alpha = FocalLoss(gamma=0.0, alpha=0.25, reduction="none")

        logits = torch.tensor([0.0, 0.0])
        targets = torch.tensor([1.0, 0.0])

        l_high = loss_fn_high_alpha(logits, targets)
        l_low = loss_fn_low_alpha(logits, targets)

        # For positive sample, high alpha yields higher loss
        assert l_high[0].item() > l_low[0].item()
        # For negative sample, high alpha (1 - alpha = 0.25) yields lower loss
        assert l_high[1].item() < l_low[1].item()

    def test_reduction_options(self) -> None:
        """Verify mean, sum, and none reduction options."""
        logits = torch.tensor([1.0, -1.0])
        targets = torch.tensor([1.0, 0.0])

        l_none = FocalLoss(reduction="none")(logits, targets)
        l_mean = FocalLoss(reduction="mean")(logits, targets)
        l_sum = FocalLoss(reduction="sum")(logits, targets)

        assert l_none.shape == torch.Size([2])
        torch.testing.assert_close(l_mean, l_none.mean())
        torch.testing.assert_close(l_sum, l_none.sum())

    def test_empty_tensor_handling(self) -> None:
        """Verify empty tensor inputs return zero scalar loss."""
        loss_fn = FocalLoss()
        loss = loss_fn(torch.empty(0), torch.empty(0))
        assert loss.item() == 0.0

    def test_invalid_parameters_raise_error(self) -> None:
        """Verify invalid gamma, alpha, or reduction raise ValueError."""
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            FocalLoss(gamma=-1.0)

        with pytest.raises(ValueError, match=r"alpha must be in \(0.0, 1.0\)"):
            FocalLoss(alpha=0.0)

        with pytest.raises(ValueError, match="reduction must be one of"):
            FocalLoss(reduction="invalid")


class TestRGCNLinkPredictionModel:
    """Validates composite end-to-end link prediction model."""

    def test_composite_model_initialization_and_forward(self) -> None:
        """Verify composite model coordinates encoder and decoder modules."""
        in_channels = {"User": 8, "Role": 8}
        enc = HeteroRGCNEncoder(
            RGCNConfig(in_channels_dict=in_channels, hidden_channels=32, num_layers=2)
        )
        dec = BilinearRelationPredictor(DecoderConfig(hidden_channels=32))
        model = RGCNLinkPredictionModel(encoder=enc, decoder=dec)

        x_dict = {"User": torch.randn(4, 8), "Role": torch.randn(3, 8)}
        edge_index_dict = {
            ("User", "AssumesRole", "Role"): torch.tensor([[0, 1], [0, 1]]),
        }

        h_dict = model(x_dict=x_dict, edge_index_dict=edge_index_dict)
        assert set(h_dict.keys()) == {"User", "Role"}
        assert h_dict["User"].shape == torch.Size([4, 32])
        assert h_dict["Role"].shape == torch.Size([3, 32])
        assert model.count_parameters() > 0

    def test_hidden_dimension_mismatch_raises_error(self) -> None:
        """Verify mismatched encoder and decoder hidden channels raise ValueError."""
        enc = HeteroRGCNEncoder(RGCNConfig(hidden_channels=64))
        dec = BilinearRelationPredictor(DecoderConfig(hidden_channels=32))
        with pytest.raises(ValueError, match="must match"):
            RGCNLinkPredictionModel(encoder=enc, decoder=dec)


class TestInductiveTrainerWorkflow:
    """Validates training loop, cosine annealing scheduling, and validation evaluation."""

    def test_trainer_fit_and_early_stopping(self, capability_model: CapabilityModel) -> None:
        """Verify multi-epoch inductive training loop, evaluation, and early stopping."""
        converter = IAMHeteroDataConverter()

        # Build 2 small training environments
        g_train1 = EnterpriseTopologyGenerator(
            config=EnterpriseTopologyConfig(num_nodes=100, seed=10, num_pe_chains=2),
            capability_model=capability_model,
        ).generate()
        d_train1 = converter.convert(g_train1)

        g_train2 = EnterpriseTopologyGenerator(
            config=EnterpriseTopologyConfig(num_nodes=120, seed=20, num_pe_chains=2),
            capability_model=capability_model,
        ).generate()
        d_train2 = converter.convert(g_train2)

        # Build 1 small validation environment
        g_val = EnterpriseTopologyGenerator(
            config=EnterpriseTopologyConfig(num_nodes=100, seed=30, num_pe_chains=2),
            capability_model=capability_model,
        ).generate()
        d_val = converter.convert(g_val)

        # Initialize composite model with small hidden dimension for fast test execution
        in_channels = get_default_in_channels_dict()
        model = RGCNLinkPredictionModel(
            encoder_config=RGCNConfig(
                in_channels_dict=in_channels, hidden_channels=32, num_layers=2
            ),
            decoder_config=DecoderConfig(hidden_channels=32),
        )

        train_cfg = TrainingConfig(
            lr=5e-3,
            max_epochs=4,
            patience=2,
            negative_ratio=2.0,
            grad_clip_norm=1.0,
        )
        trainer = InductiveLinkPredictionTrainer(model=model, config=train_cfg)

        history = trainer.fit(
            train_environments=[d_train1, d_train2],
            val_environments=[d_val],
            condition=MaskingCondition.CONDITION_E_ADVERSARIAL_BRIDGE,
            seed=42,
        )

        assert isinstance(history, TrainingHistory)
        assert len(history.epoch_train_losses) <= 4
        assert len(history.epoch_val_pr_aucs) == len(history.epoch_train_losses)
        assert len(history.epoch_lrs) == len(history.epoch_train_losses)
        assert history.best_epoch >= 1
        assert history.best_val_pr_auc >= 0.0

        # Check serialization and report generation
        hist_dict = history.to_dict()
        assert "best_epoch" in hist_dict
        assert "best_val_pr_auc" in hist_dict

        report = history.generate_markdown_report()
        assert "### Inductive Training Convergence Summary" in report
        assert "Train Focal Loss" in report
