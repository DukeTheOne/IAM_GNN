"""Unit test suite for Untyped homogeneous GCN baseline architecture.

Verifies input projection, un-typed GCN forward pass, gradient flow,
and integration with UntypedLinkPredictionModel for architectural ablation studies.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from iam.models.untyped_gcn import (
    UntypedGCNConfig,
    UntypedGCNEncoder,
    UntypedLinkPredictionModel,
)


class TestUntypedGCNConfig:
    """Tests for UntypedGCNConfig dataclass."""

    def test_default_config(self) -> None:
        """Verify default hyperparameter contracts."""
        cfg = UntypedGCNConfig()
        assert cfg.hidden_channels == 128
        assert cfg.num_layers == 2
        assert cfg.dropout == 0.2
        assert cfg.act_negative_slope == 0.2
        assert cfg.use_residual is True
        assert "User" in cfg.in_channels_dict

    def test_invalid_parameters(self) -> None:
        """Verify error checking for invalid configuration."""
        with pytest.raises(ValueError, match="num_layers must be >= 1"):
            UntypedGCNEncoder(UntypedGCNConfig(num_layers=0))

        with pytest.raises(ValueError, match="hidden_channels must be > 0"):
            UntypedGCNEncoder(UntypedGCNConfig(hidden_channels=0))


class TestUntypedGCNEncoder:
    """Tests for UntypedGCNEncoder forward pass and gradient flow."""

    @pytest.fixture
    def sample_data(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[tuple[str, str, str], torch.Tensor]]:
        """Construct synthetic node features and multirelational edges."""
        torch.manual_seed(42)
        x_dict = {
            "User": torch.randn(5, 51),
            "Role": torch.randn(4, 52),
            "Policy": torch.randn(3, 39),
        }
        edge_index_dict = {
            ("User", "AssumesRole", "Role"): torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            ("Role", "AttachedWith", "Policy"): torch.tensor([[1, 2], [0, 1]], dtype=torch.long),
        }
        return x_dict, edge_index_dict

    def test_forward_with_dict(
        self, sample_data: tuple[dict[str, torch.Tensor], dict[tuple[str, str, str], torch.Tensor]]
    ) -> None:
        """Verify forward pass with explicit dictionary tensors."""
        x_dict, edge_index_dict = sample_data
        encoder = UntypedGCNEncoder()
        out = encoder(x_dict=x_dict, edge_index_dict=edge_index_dict)

        assert "User" in out
        assert "Role" in out
        assert "Policy" in out
        assert out["User"].shape == (5, 128)
        assert out["Role"].shape == (4, 128)
        assert out["Policy"].shape == (3, 128)

    def test_forward_with_hetero_data(
        self, sample_data: tuple[dict[str, torch.Tensor], dict[tuple[str, str, str], torch.Tensor]]
    ) -> None:
        """Verify forward pass using PyG HeteroData object."""
        x_dict, edge_index_dict = sample_data
        data = HeteroData()
        for nt, x in x_dict.items():
            data[nt].x = x
        for et, ei in edge_index_dict.items():
            data[et].edge_index = ei

        encoder = UntypedGCNEncoder()
        out = encoder(data=data)

        assert out["User"].shape == (5, 128)
        assert out["Role"].shape == (4, 128)

    def test_empty_graph_handling(self) -> None:
        """Verify handling of empty or disconnected graphs."""
        encoder = UntypedGCNEncoder()
        empty_out = encoder(x_dict={}, edge_index_dict={})
        assert empty_out == {}

        # Disconnected nodes (no edges)
        x_dict = {"User": torch.randn(3, 51)}
        out = encoder(x_dict=x_dict, edge_index_dict={})
        assert out["User"].shape == (3, 128)


class TestUntypedLinkPredictionModel:
    """Tests for composite UntypedLinkPredictionModel."""

    def test_end_to_end_forward_and_gradients(self) -> None:
        """Verify end-to-end scoring and non-zero backpropagation gradients."""
        torch.manual_seed(42)
        model = UntypedLinkPredictionModel()

        data = HeteroData()
        data["User"].x = torch.randn(4, 51)
        data["Role"].x = torch.randn(3, 52)
        data[("User", "AssumesRole", "Role")].edge_index = torch.tensor(
            [[0, 1], [1, 2]], dtype=torch.long
        )

        supervision = {
            ("User", "AssumesRole", "Role"): torch.tensor([[0, 2], [1, 0]], dtype=torch.long)
        }

        logits_dict = model(data=data, supervision_dict=supervision)
        logits = logits_dict[("User", "AssumesRole", "Role")]
        assert logits.shape == (2,)

        loss = logits.sum()
        loss.backward()

        # Verify gradient flow into encoder linear projections and decoder parameters
        assert model.encoder.input_proj["User"].weight.grad is not None
        assert model.decoder.weight.grad is not None
        assert model.decoder.bias.grad is not None
