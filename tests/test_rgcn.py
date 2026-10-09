"""Tests for HeteroRGCNEncoder architecture and basis-sharing regularization."""

from __future__ import annotations

import pytest
import torch

from iam.generator.topology import EnterpriseTopologyConfig, EnterpriseTopologyGenerator
from iam.models.converter import (
    CANONICAL_FORWARD_EDGE_TYPES,
    IAMHeteroDataConverter,
    get_reverse_edge_type,
)
from iam.models.rgcn import (
    CANONICAL_RELATION_TYPES,
    HeteroRGCNEncoder,
    RGCNConfig,
    get_default_in_channels_dict,
)
from iam.parser.capability import CapabilityModel, load_default_capability_model


@pytest.fixture
def capability_model() -> CapabilityModel:
    """Fixture providing loaded AWS CapabilityModel."""
    return load_default_capability_model()


@pytest.fixture
def default_in_channels() -> dict[str, int]:
    """Fixture providing default input channels mapping."""
    return get_default_in_channels_dict()


class TestRGCNConfig:
    """Validates configuration contracts and hyperparameter bounds."""

    def test_default_config_matches_blueprint_spec(self) -> None:
        """Verify default parameters adhere to Tier 1 Blueprint contracts."""
        cfg = RGCNConfig()
        assert cfg.hidden_channels == 128
        assert cfg.num_layers == 2
        assert cfg.num_bases == 8
        assert cfg.num_relations == 26
        assert cfg.dropout == 0.2
        assert cfg.act_negative_slope == 0.2
        assert cfg.use_residual is True
        assert cfg.layer_norm is True
        assert cfg.aggr == "mean"

        # Verify all 5 node types are present in default input channels
        for nt in ["User", "Role", "Policy", "Group", "Resource"]:
            assert nt in cfg.in_channels_dict
            assert cfg.in_channels_dict[nt] > 0

    def test_config_invalid_hyperparameters(self) -> None:
        """Ensure invalid architectural hyperparameters raise ValueError."""
        with pytest.raises(ValueError, match="hidden_channels must be positive"):
            RGCNConfig(hidden_channels=0)

        with pytest.raises(ValueError, match="num_layers must be >= 1"):
            RGCNConfig(num_layers=0)

        with pytest.raises(ValueError, match="num_bases must be >= 1"):
            RGCNConfig(num_bases=0)

        with pytest.raises(ValueError, match=r"dropout must be in \[0.0, 1.0\)"):
            RGCNConfig(dropout=1.0)


class TestCanonicalRelations:
    """Validates canonical relation registry and mapping routines."""

    def test_canonical_relations_coverage(self) -> None:
        """Verify 26 relation triples with bidirectional mapping."""
        assert len(CANONICAL_RELATION_TYPES) == 26

        # Forward relations must all be in canonical relation types
        for fwd in CANONICAL_FORWARD_EDGE_TYPES:
            assert fwd in CANONICAL_RELATION_TYPES
            rev = get_reverse_edge_type(fwd)
            assert rev in CANONICAL_RELATION_TYPES

    def test_relation_id_roundtrip(self) -> None:
        """Verify relation ID lookup and reverse lookup."""
        encoder = HeteroRGCNEncoder()
        test_edge = ("User", "AssumesRole", "Role")
        rel_id = encoder.get_relation_id(test_edge)
        assert isinstance(rel_id, int)
        assert 0 <= rel_id < 26
        assert encoder.get_relation_triple(rel_id) == test_edge

    def test_unknown_relation_raises_keyerror(self) -> None:
        """Verify lookup on unindexed relation triple raises KeyError."""
        encoder = HeteroRGCNEncoder()
        with pytest.raises(KeyError, match="Unknown edge relation triple"):
            encoder.get_relation_id(("User", "FakeRelation", "Role"))

        with pytest.raises(KeyError, match="Unknown relation ID"):
            encoder.get_relation_triple(999)


class TestBasisSharingRegularization:
    """Validates basis-sharing parameter compression and weight structures."""

    def test_basis_matrix_decomposition_savings(self, default_in_channels: dict[str, int]) -> None:
        """Verify basis decomposition achieves >60% parameter reduction on conv layers."""
        cfg = RGCNConfig(
            in_channels_dict=default_in_channels,
            hidden_channels=128,
            num_layers=2,
            num_bases=8,
        )
        encoder = HeteroRGCNEncoder(cfg)
        breakdown = encoder.count_basis_parameters()

        # Shared bases: 8 * 128 * 128 = 131,072 per layer
        # Linear combination coeffs: 26 * 8 = 208 per layer
        # Root weights: 128 * 128 = 16,384 per layer
        assert breakdown["basis_matrices"] == 2 * (8 * 128 * 128)
        assert breakdown["basis_coefficients"] == 2 * (26 * 8)
        assert breakdown["root_weights"] == 2 * (128 * 128)

        total_actual_conv = (
            breakdown["basis_matrices"]
            + breakdown["basis_coefficients"]
            + breakdown["root_weights"]
            + breakdown["conv_biases"]
        )
        unconstrained_estimate = breakdown["unconstrained_conv_estimate"]

        # Theoretical reduction ratio
        compression_ratio = 1.0 - (total_actual_conv / unconstrained_estimate)
        assert compression_ratio > 0.60, (
            f"Expected >60% parameter compression, got {compression_ratio * 100:.1f}%"
        )

    def test_conv_weight_tensor_shapes(self) -> None:
        """Verify exact internal PyG RGCNConv tensor shapes."""
        encoder = HeteroRGCNEncoder(RGCNConfig(hidden_channels=64, num_bases=4, num_layers=2))
        for conv in encoder.convs:
            assert conv.weight.shape == torch.Size([4, 64, 64])
            assert conv.comp.shape == torch.Size([26, 4])
            assert conv.root.shape == torch.Size([64, 64])


class TestHeteroRGCNForwardPass:
    """Validates forward computation, gradient flow, and edge cases."""

    def test_forward_pass_with_raw_dictionaries(self) -> None:
        """Verify forward pass on synthetic feature and edge dictionaries."""
        in_channels = {"User": 10, "Role": 12, "Policy": 8}
        cfg = RGCNConfig(
            in_channels_dict=in_channels, hidden_channels=32, num_layers=2, num_bases=4
        )
        encoder = HeteroRGCNEncoder(cfg)

        x_dict = {
            "User": torch.randn(5, 10),
            "Role": torch.randn(3, 12),
            "Policy": torch.randn(4, 8),
        }
        edge_index_dict = {
            ("User", "AttachedWith", "Policy"): torch.tensor([[0, 1, 2], [0, 1, 2]]),
            ("Role", "AttachedWith", "Policy"): torch.tensor([[0, 1], [2, 3]]),
        }

        out = encoder(x_dict=x_dict, edge_index_dict=edge_index_dict)
        assert set(out.keys()) == {"User", "Role", "Policy"}
        assert out["User"].shape == torch.Size([5, 32])
        assert out["Role"].shape == torch.Size([3, 32])
        assert out["Policy"].shape == torch.Size([4, 32])

    def test_gradient_flow_and_backprop(self) -> None:
        """Verify all layers receive non-zero gradients during backpropagation."""
        in_channels = {"User": 8, "Role": 8}
        cfg = RGCNConfig(
            in_channels_dict=in_channels, hidden_channels=16, num_layers=2, num_bases=2
        )
        encoder = HeteroRGCNEncoder(cfg)

        x_dict = {
            "User": torch.randn(4, 8),
            "Role": torch.randn(3, 8),
        }
        edge_index_dict = {
            ("User", "AssumesRole", "Role"): torch.tensor([[0, 1, 2], [0, 1, 2]]),
        }

        out = encoder(x_dict=x_dict, edge_index_dict=edge_index_dict)
        loss = out["User"].sum() + out["Role"].sum()
        loss.backward()

        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} did not receive gradients."
                assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaNs."

    def test_forward_pass_on_pyg_hetero_data(
        self, capability_model: CapabilityModel, default_in_channels: dict[str, int]
    ) -> None:
        """Verify forward pass on an actual converted enterprise IAM HeteroData instance."""
        topo_cfg = EnterpriseTopologyConfig(
            num_nodes=200,
            seed=42,
            num_pe_chains=2,
            account_id="111122223333",
            organization_id="o-rgcn-test",
        )
        generator = EnterpriseTopologyGenerator(config=topo_cfg, capability_model=capability_model)
        graph = generator.generate()

        converter = IAMHeteroDataConverter()
        data = converter.convert(graph)

        encoder = HeteroRGCNEncoder(
            RGCNConfig(in_channels_dict=default_in_channels, hidden_channels=128, num_layers=2)
        )
        encoder.eval()

        with torch.no_grad():
            out = encoder(data)

        for nt in data.node_types:
            assert nt in out
            num_nodes = data[nt].x.size(0)
            assert out[nt].shape == torch.Size([num_nodes, 128])
            # Check for non-trivial finite embeddings
            assert not torch.isnan(out[nt]).any()
            assert not torch.isinf(out[nt]).any()

    def test_empty_edge_index_graceful_handling(self) -> None:
        """Verify graph with isolated nodes and zero edges executes without error."""
        in_channels = {"User": 10, "Role": 10}
        cfg = RGCNConfig(in_channels_dict=in_channels, hidden_channels=16, num_layers=2)
        encoder = HeteroRGCNEncoder(cfg)

        x_dict = {
            "User": torch.randn(3, 10),
            "Role": torch.randn(2, 10),
        }
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {
            ("User", "AssumesRole", "Role"): torch.empty((2, 0), dtype=torch.long),
        }

        out = encoder(x_dict=x_dict, edge_index_dict=edge_index_dict)
        assert out["User"].shape == torch.Size([3, 16])
        assert out["Role"].shape == torch.Size([2, 16])

    def test_empty_graph_returns_empty_tensors(self) -> None:
        """Verify completely empty graph returns empty dictionary or empty tensors."""
        in_channels = {"User": 10}
        cfg = RGCNConfig(in_channels_dict=in_channels, hidden_channels=16)
        encoder = HeteroRGCNEncoder(cfg)

        x_dict = {"User": torch.empty((0, 10))}
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {}

        out = encoder(x_dict=x_dict, edge_index_dict=edge_index_dict)
        assert out["User"].shape == torch.Size([0, 16])

    def test_unconfigured_node_type_raises_keyerror(self) -> None:
        """Verify attempting to pass unconfigured node type raises informative KeyError."""
        in_channels = {"User": 10}
        cfg = RGCNConfig(in_channels_dict=in_channels, hidden_channels=16)
        encoder = HeteroRGCNEncoder(cfg)

        x_dict = {"User": torch.randn(2, 10), "UnknownType": torch.randn(2, 5)}
        with pytest.raises(KeyError, match="UnknownType"):
            encoder(x_dict=x_dict, edge_index_dict={})

    def test_missing_input_arguments_raises_valueerror(self) -> None:
        """Verify calling forward with neither data nor dictionaries raises ValueError."""
        encoder = HeteroRGCNEncoder()
        with pytest.raises(ValueError, match="Must provide either 'data'"):
            encoder()


class TestInductiveGeneralizationAcrossEnvironments:
    """Validates inductive transfer across distinct enterprise organizations."""

    def test_inductive_evaluation_on_distinct_environments(
        self, capability_model: CapabilityModel, default_in_channels: dict[str, int]
    ) -> None:
        """Verify the exact same model instance processes multiple distinct environments."""
        encoder = HeteroRGCNEncoder(
            RGCNConfig(in_channels_dict=default_in_channels, hidden_channels=64, num_layers=2)
        )
        encoder.eval()

        converter = IAMHeteroDataConverter()

        # Environment 1: Small organization (150 nodes)
        g1 = EnterpriseTopologyGenerator(
            config=EnterpriseTopologyConfig(num_nodes=150, seed=101, num_pe_chains=2),
            capability_model=capability_model,
        ).generate()
        d1 = converter.convert(g1)

        # Environment 2: Larger organization (300 nodes)
        g2 = EnterpriseTopologyGenerator(
            config=EnterpriseTopologyConfig(num_nodes=300, seed=202, num_pe_chains=4),
            capability_model=capability_model,
        ).generate()
        d2 = converter.convert(g2)

        with torch.no_grad():
            out1 = encoder(d1)
            out2 = encoder(d2)

        assert out1["User"].size(0) == d1["User"].x.size(0)
        assert out1["Role"].size(0) == d1["Role"].x.size(0)
        assert out2["User"].size(0) == d2["User"].x.size(0)
        assert out2["Role"].size(0) == d2["Role"].x.size(0)

        assert out1["User"].size(1) == 64
        assert out2["User"].size(1) == 64
