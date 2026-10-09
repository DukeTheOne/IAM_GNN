"""Tests for BilinearRelationPredictor decoder head."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from iam.models.decoder import BilinearRelationPredictor, DecoderConfig
from iam.models.rgcn import HeteroRGCNEncoder, RGCNConfig


class TestDecoderConfig:
    """Validates configuration contracts and hyperparameter bounds."""

    def test_default_config(self) -> None:
        """Verify default configuration values."""
        cfg = DecoderConfig()
        assert cfg.hidden_channels == 128
        assert cfg.num_relations == 26
        assert cfg.mode == "bilinear"
        assert cfg.num_bases == 4
        assert cfg.use_bias is True
        assert cfg.dropout == 0.0

    def test_invalid_configurations_raise_error(self) -> None:
        """Verify invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="hidden_channels must be positive"):
            DecoderConfig(hidden_channels=0)

        with pytest.raises(ValueError, match="num_relations must be positive"):
            DecoderConfig(num_relations=0)

        with pytest.raises(ValueError, match="mode must be one of"):
            DecoderConfig(mode="invalid_mode")

        with pytest.raises(ValueError, match="num_bases must be >= 1"):
            DecoderConfig(mode="basis", num_bases=0)

        with pytest.raises(ValueError, match=r"dropout must be in \[0.0, 1.0\)"):
            DecoderConfig(dropout=1.0)


class TestBilinearWeightParametrization:
    """Validates tensor initialization and parameter counts across modes."""

    def test_bilinear_mode_shapes(self) -> None:
        """Verify full bilinear parameter shapes."""
        pred = BilinearRelationPredictor(
            DecoderConfig(hidden_channels=64, num_relations=10, mode="bilinear")
        )
        assert pred.W.shape == torch.Size([10, 64, 64])
        assert pred.bias is not None and pred.bias.shape == torch.Size([10])
        assert pred.count_parameters() == 10 * 64 * 64 + 10

    def test_basis_mode_shapes(self) -> None:
        """Verify basis regularized bilinear parameter shapes."""
        pred = BilinearRelationPredictor(
            DecoderConfig(hidden_channels=64, num_relations=10, mode="basis", num_bases=3)
        )
        assert pred.basis.shape == torch.Size([3, 64, 64])
        assert pred.comp.shape == torch.Size([10, 3])
        # W_r reconstructed from linear combination
        w_0 = pred.get_relation_matrix(0)
        assert w_0.shape == torch.Size([64, 64])

    def test_distmult_mode_shapes(self) -> None:
        """Verify DistMult factorized diagonal parameter shapes."""
        pred = BilinearRelationPredictor(
            DecoderConfig(hidden_channels=64, num_relations=10, mode="distmult")
        )
        assert pred.W.shape == torch.Size([10, 64])
        assert pred.count_parameters() == 10 * 64 + 10


class TestPairScoringAndLogits:
    """Validates batch pair scoring, broadcasting, and temperature scaling."""

    def test_homogeneous_pair_scoring(self) -> None:
        """Verify scoring batch of pairs with single integer relation ID."""
        pred = BilinearRelationPredictor(DecoderConfig(hidden_channels=32, num_relations=5))
        h_u = torch.randn(8, 32)
        h_v = torch.randn(8, 32)

        logits = pred(h_u, h_v, rel_ids=2)
        assert logits.shape == torch.Size([8])

        probs = pred.predict_proba(h_u, h_v, rel_ids=2)
        assert probs.shape == torch.Size([8])
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
        # Verify sigmoid connection
        expected_probs = torch.sigmoid(logits)
        torch.testing.assert_close(probs, expected_probs)

    def test_heterogeneous_pair_scoring(self) -> None:
        """Verify scoring batch of pairs with distinct per-pair relation IDs."""
        pred = BilinearRelationPredictor(DecoderConfig(hidden_channels=32, num_relations=5))
        h_u = torch.randn(10, 32)
        h_v = torch.randn(10, 32)
        rel_ids = torch.randint(0, 5, (10,))

        logits = pred(h_u, h_v, rel_ids=rel_ids)
        assert logits.shape == torch.Size([10])

        probs = pred.predict_proba(h_u, h_v, rel_ids=rel_ids)
        assert probs.shape == torch.Size([10])

    def test_temperature_scaling_effect(self) -> None:
        """Verify temperature scaling modulates prediction confidence."""
        pred = BilinearRelationPredictor(DecoderConfig(hidden_channels=16, num_relations=2))
        h_u = torch.randn(4, 16)
        h_v = torch.randn(4, 16)

        p_t1 = pred.predict_proba(h_u, h_v, rel_ids=0, temperature=1.0)
        p_t2 = pred.predict_proba(h_u, h_v, rel_ids=0, temperature=2.0)
        p_t05 = pred.predict_proba(h_u, h_v, rel_ids=0, temperature=0.5)

        # Higher T pulls probabilities closer to 0.5; lower T sharpens them towards 0 or 1
        for i in range(4):
            dist_1 = abs(p_t1[i].item() - 0.5)
            dist_2 = abs(p_t2[i].item() - 0.5)
            dist_05 = abs(p_t05[i].item() - 0.5)
            assert dist_2 <= dist_1 + 1e-5
            assert dist_05 >= dist_1 - 1e-5

    def test_temperature_zero_or_negative_raises_error(self) -> None:
        """Verify non-positive temperature raises ValueError."""
        pred = BilinearRelationPredictor()
        h = torch.randn(2, 128)
        with pytest.raises(ValueError, match="Temperature must be strictly positive"):
            pred.predict_proba(h, h, 0, temperature=0.0)

    def test_empty_tensor_scoring_returns_empty(self) -> None:
        """Verify scoring 0 pairs returns empty 1D tensor."""
        pred = BilinearRelationPredictor()
        h_empty = torch.empty((0, 128))
        logits = pred(h_empty, h_empty, rel_ids=0)
        assert logits.shape == torch.Size([0])

    def test_mismatched_dimensions_raise_value_error(self) -> None:
        """Verify mismatched batch or hidden dimensions raise ValueError."""
        pred = BilinearRelationPredictor(DecoderConfig(hidden_channels=32))
        h1 = torch.randn(4, 32)
        h2 = torch.randn(5, 32)
        with pytest.raises(ValueError, match="Mismatched batch sizes"):
            pred(h1, h2, rel_ids=0)

        h_wrong_d = torch.randn(4, 16)
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            pred(h1, h_wrong_d, rel_ids=0)


class TestAllPairsMatrixScoring:
    """Validates full candidate matrix evaluation Z in R^{N_src x N_dst}."""

    def test_matrix_scoring_matches_pointwise_eval(self) -> None:
        """Verify matrix scoring produces identical results to individual pair scoring."""
        pred = BilinearRelationPredictor(DecoderConfig(hidden_channels=32, num_relations=4))
        h_src = torch.randn(5, 32)
        h_dst = torch.randn(7, 32)

        mat_probs = pred.score_matrix(h_src, h_dst, rel_id=1)
        assert mat_probs.shape == torch.Size([5, 7])

        # Verify entry (i, j) matches pointwise pair scoring
        for i in range(5):
            for j in range(7):
                pair_prob = pred.predict_proba(h_src[i : i + 1], h_dst[j : j + 1], rel_ids=1).item()
                assert abs(mat_probs[i, j].item() - pair_prob) < 1e-5

    def test_empty_matrix_scoring(self) -> None:
        """Verify matrix scoring with empty source or destination returns empty tensor."""
        pred = BilinearRelationPredictor(DecoderConfig(hidden_channels=32))
        h_src = torch.randn(5, 32)
        h_empty = torch.empty((0, 32))
        out = pred.score_matrix(h_src, h_empty, rel_id=0)
        assert out.shape == torch.Size([5, 0])


class TestEdgeIndexScoring:
    """Validates edge index tensor scoring routines."""

    def test_score_edge_index_routine(self) -> None:
        """Verify scoring candidate edge index tensor [2, M] for a specific relation."""
        pred = BilinearRelationPredictor(DecoderConfig(hidden_channels=32))
        h_dict = {
            "User": torch.randn(10, 32),
            "Role": torch.randn(6, 32),
        }
        edge_type = ("User", "AssumesRole", "Role")
        edge_index = torch.tensor([[0, 2, 4], [1, 3, 5]])

        probs = pred.score_edge_index(h_dict, edge_type, edge_index)
        assert probs.shape == torch.Size([3])
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_score_edge_index_empty_edges(self) -> None:
        """Verify scoring empty edge index returns empty tensor."""
        pred = BilinearRelationPredictor(DecoderConfig(hidden_channels=32))
        h_dict = {"User": torch.randn(5, 32), "Role": torch.randn(3, 32)}
        edge_type = ("User", "AssumesRole", "Role")
        edge_index = torch.empty((2, 0), dtype=torch.long)

        probs = pred.score_edge_index(h_dict, edge_type, edge_index)
        assert probs.shape == torch.Size([0])


class TestEndToEndEncoderDecoderIntegration:
    """Validates end-to-end forward and backward passes from raw features to link predictions."""

    def test_end_to_end_backprop_through_encoder_and_decoder(self) -> None:
        """Verify complete gradient backpropagation through both decoder and encoder."""
        in_channels = {"User": 10, "Role": 12}
        enc = HeteroRGCNEncoder(
            RGCNConfig(in_channels_dict=in_channels, hidden_channels=32, num_layers=2)
        )
        dec = BilinearRelationPredictor(DecoderConfig(hidden_channels=32))

        x_dict = {
            "User": torch.randn(5, 10, requires_grad=True),
            "Role": torch.randn(4, 12, requires_grad=True),
        }
        edge_index_dict = {
            ("User", "AssumesRole", "Role"): torch.tensor([[0, 1, 2], [1, 2, 3]]),
        }

        # 1. Forward through encoder
        h_dict = enc(x_dict=x_dict, edge_index_dict=edge_index_dict)

        # 2. Forward through decoder for candidate links
        candidate_edges = torch.tensor([[0, 3], [2, 0]])
        probs = dec.score_edge_index(h_dict, ("User", "AssumesRole", "Role"), candidate_edges)

        # 3. Supervised BCE loss
        targets = torch.tensor([1.0, 0.0])
        loss = torch.nn.functional.binary_cross_entropy(probs, targets)

        # 4. Backward pass
        loss.backward()  # type: ignore[no-untyped-call]

        # Verify decoder received gradients
        assert dec.W.grad is not None
        assert not torch.isnan(dec.W.grad).any()

        # Verify encoder received gradients all the way to input projection weights
        user_proj = cast(torch.nn.Linear, enc.input_proj["User"])
        assert user_proj.weight.grad is not None
        assert not torch.isnan(user_proj.weight.grad).any()
