from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
import torch

from iam.generator import (
    EnterpriseTopologyConfig,
    EnterpriseTopologyGenerator,
    EnvironmentExporter,
    GraphEdge,
    GraphNode,
    GroundTruthLabeler,
    IAMGraph,
    NodeType,
)
from iam.generator.graph import EdgeRelation
from iam.models import (
    ActionVocabulary,
    IAMHeteroDataConverter,
    get_reverse_edge_type,
    get_reverse_relation_name,
    is_reverse_edge_type,
    load_default_action_vocabulary,
    load_hetero_data,
    save_hetero_data,
)
from iam.parser import (
    CapabilityModel,
    load_default_capability_model,
)


@pytest.fixture
def capability_model() -> CapabilityModel:
    """Fixture providing loaded AWS CapabilityModel."""
    return load_default_capability_model()


@pytest.fixture
def default_vocab() -> ActionVocabulary:
    """Fixture providing default ActionVocabulary."""
    return load_default_action_vocabulary()


@pytest.fixture
def enterprise_graph_with_labels(
    capability_model: CapabilityModel,
) -> tuple[IAMGraph, Any]:
    """Fixture providing enterprise graph with computed ground-truth label set."""
    cfg = EnterpriseTopologyConfig(
        num_nodes=200,
        seed=42,
        num_pe_chains=3,
        include_branching_chains=True,
        num_branching_chains=1,
    )
    gen = EnterpriseTopologyGenerator(cfg, capability_model=capability_model)
    graph = gen.generate()

    labeler = GroundTruthLabeler(max_depth=6)
    label_set = labeler.label_graph(graph)
    return graph, label_set


class TestRelationHelpers:
    """Tests for relation naming and reverse relation computation."""

    def test_reverse_relation_name(self) -> None:
        """Verify reverse relation name prefixing and stripping."""
        assert get_reverse_relation_name("AssumesRole") == "RevAssumesRole"
        assert get_reverse_relation_name("AttachedWith") == "RevAttachedWith"
        assert get_reverse_relation_name("RevAssumesRole") == "AssumesRole"

    def test_reverse_edge_type(self) -> None:
        """Verify source/destination swap and relation reversal."""
        fwd = ("User", "AssumesRole", "Role")
        rev = get_reverse_edge_type(fwd)
        assert rev == ("Role", "RevAssumesRole", "User")

        reflexive = ("Role", "AssumesRole", "Role")
        rev_reflexive = get_reverse_edge_type(reflexive)
        assert rev_reflexive == ("Role", "RevAssumesRole", "Role")

    def test_is_reverse_edge_type(self) -> None:
        """Verify reverse edge classification."""
        assert is_reverse_edge_type(("User", "AssumesRole", "Role")) is False
        assert is_reverse_edge_type(("Role", "RevAssumesRole", "User")) is True


class TestIAMHeteroDataConverter:
    """Tests for PyG HeteroData structure generation, reverse edges, and metadata."""

    def test_minimal_graph_conversion(self, default_vocab: ActionVocabulary) -> None:
        """Verify conversion on a minimal multi-relational IAM graph."""
        graph = IAMGraph("min_test")

        user = GraphNode(
            id="user:u1",
            node_type=NodeType.USER,
            arn="arn:aws:iam::123:user/u1",
            name="u1",
            department="DevOps",
        )
        role = GraphNode(
            id="role:r1",
            node_type=NodeType.ROLE,
            arn="arn:aws:iam::123:role/r1",
            name="r1",
            department="DevOps",
        )
        edge = GraphEdge(
            source=user.id,
            target=role.id,
            relation=EdgeRelation.ASSUMES_ROLE,
            is_bridge=True,
            actions=["sts:AssumeRole"],
        )

        graph.add_node(user)
        graph.add_node(role)
        graph.add_edge(edge)

        converter = IAMHeteroDataConverter(vocabulary=default_vocab)
        data = converter.convert(graph)

        # Check nodes
        assert data["User"].num_nodes == 1
        assert data["Role"].num_nodes == 1
        assert data["User"].x.shape == (1, converter.vocabulary.vocab_size + 20)
        assert data["Role"].x.shape == (1, converter.vocabulary.vocab_size + 21)

        # Check forward edge
        fwd_triple = ("User", "AssumesRole", "Role")
        assert fwd_triple in data.edge_types
        assert data[fwd_triple].edge_index.shape == (2, 1)
        assert data[fwd_triple].edge_index[0, 0].item() == 0
        assert data[fwd_triple].edge_index[1, 0].item() == 0
        assert data[fwd_triple].edge_is_bridge[0].item() is True

        # Check reverse edge
        rev_triple = ("Role", "RevAssumesRole", "User")
        assert rev_triple in data.edge_types
        assert data[rev_triple].edge_index.shape == (2, 1)
        assert data[rev_triple].edge_index[0, 0].item() == 0
        assert data[rev_triple].edge_index[1, 0].item() == 0
        # Reverse edges are message-passing conduits, not forward attack bridges
        assert data[rev_triple].edge_is_bridge[0].item() is False

        # Edge attributes (action bitmask)
        assert hasattr(data[fwd_triple], "edge_attr")
        assumerole_idx = default_vocab.get_index("sts:AssumeRole")
        assert data[fwd_triple].edge_attr[0, assumerole_idx].item() == 1.0

    def test_reverse_edges_disabled_option(self, default_vocab: ActionVocabulary) -> None:
        """Verify converter respects include_reverse_edges=False."""
        graph = IAMGraph("no_rev")
        u = GraphNode(id="u1", node_type=NodeType.USER, arn="arn:aws:iam::123:user/u1", name="u1")
        r = GraphNode(id="r1", node_type=NodeType.ROLE, arn="arn:aws:iam::123:role/r1", name="r1")
        e = GraphEdge(source="u1", target="r1", relation=EdgeRelation.ASSUMES_ROLE)
        graph.add_nodes([u, r])
        graph.add_edge(e)

        converter = IAMHeteroDataConverter(vocabulary=default_vocab, include_reverse_edges=False)
        data = converter.convert(graph)

        assert ("User", "AssumesRole", "Role") in data.edge_types
        assert ("Role", "RevAssumesRole", "User") not in data.edge_types

    def test_enterprise_graph_conversion(
        self,
        enterprise_graph_with_labels: tuple[IAMGraph, Any],
        default_vocab: ActionVocabulary,
    ) -> None:
        """Verify full conversion on an enterprise topology graph with labels."""
        graph, label_set = enterprise_graph_with_labels
        converter = IAMHeteroDataConverter(vocabulary=default_vocab)
        data = converter.convert(graph, label_set=label_set)

        # 1. Check all canonical node types are populated
        for n_type in NodeType:
            t_str = n_type.value
            expected_count = len(graph.get_nodes_by_type(n_type))
            assert data[t_str].num_nodes == expected_count
            assert data[t_str].x.shape[0] == expected_count

        # 2. Check edge types and reverse edge symmetry
        forward_types = [t for t in data.edge_types if not is_reverse_edge_type(t)]
        for fwd_t in forward_types:
            rev_t = get_reverse_edge_type(fwd_t)
            assert rev_t in data.edge_types

            fwd_ei = data[fwd_t].edge_index
            rev_ei = data[rev_t].edge_index

            assert fwd_ei.shape[1] == rev_ei.shape[1]
            assert torch.equal(fwd_ei[0], rev_ei[1])
            assert torch.equal(fwd_ei[1], rev_ei[0])

        # 3. Check bridge flags identify injected PE attack bottlenecks
        total_bridges = 0
        for fwd_t in forward_types:
            total_bridges += int(data[fwd_t].edge_is_bridge.sum().item())
        assert total_bridges > 0, "Enterprise graph must contain labeled bridge edges"

        # 4. Check ground truth PE pairs
        assert "pe_pairs" in data
        assert data["num_pe_pairs"] == len(label_set.pe_pairs)
        assert len(data["pe_pairs"]) > 0
        for pe in data["pe_pairs"]:
            assert "source_type" in pe
            assert "source_index" in pe
            assert "target_type" in pe
            assert "target_index" in pe
            assert "hop_distance" in pe

    def test_pyg_ready_dict_conversion(
        self,
        enterprise_graph_with_labels: tuple[IAMGraph, Any],
        default_vocab: ActionVocabulary,
    ) -> None:
        """Verify fast conversion from EnvironmentExporter.to_pyg_ready_dict() payload."""
        graph, label_set = enterprise_graph_with_labels
        exporter = EnvironmentExporter()
        pyg_dict = exporter.to_pyg_ready_dict(graph, label_set)

        converter = IAMHeteroDataConverter(vocabulary=default_vocab)
        data = converter.convert_from_pyg_ready_dict(pyg_dict)

        assert data.graph_id == graph.graph_id
        for n_type in NodeType:
            expected_nodes = len(graph.get_nodes_by_type(n_type))
            assert data[n_type.value].num_nodes == expected_nodes

        assert "pe_pairs" in data
        assert data["num_pe_pairs"] == len(label_set.pe_pairs)

    def test_serialization_roundtrip(
        self,
        enterprise_graph_with_labels: tuple[IAMGraph, Any],
        default_vocab: ActionVocabulary,
        tmp_path: Path,
    ) -> None:
        """Verify saving and loading HeteroData via torch binary format."""
        graph, label_set = enterprise_graph_with_labels
        converter = IAMHeteroDataConverter(vocabulary=default_vocab)
        data = converter.convert(graph, label_set=label_set)

        save_path = tmp_path / "enterprise_graph.pt"
        save_hetero_data(data, save_path)
        assert save_path.exists()

        loaded_data = load_hetero_data(save_path)
        assert loaded_data.graph_id == data.graph_id

        # Verify node features and edge indices match exactly
        for n_type in NodeType:
            t_str = n_type.value
            assert torch.equal(data[t_str].x, loaded_data[t_str].x)

        for e_type in data.edge_types:
            assert torch.equal(data[e_type].edge_index, loaded_data[e_type].edge_index)
            assert torch.equal(data[e_type].edge_is_bridge, loaded_data[e_type].edge_is_bridge)

    def test_large_graph_conversion_performance(
        self,
        capability_model: CapabilityModel,
        default_vocab: ActionVocabulary,
    ) -> None:
        """Verify converting a large graph (N=1,000) completes well under 0.25s."""
        cfg = EnterpriseTopologyConfig(num_nodes=1000, seed=123)
        gen = EnterpriseTopologyGenerator(cfg, capability_model=capability_model)
        graph = gen.generate()

        converter = IAMHeteroDataConverter(vocabulary=default_vocab)
        start_time = time.perf_counter()
        data = converter.convert(graph)
        duration = time.perf_counter() - start_time

        assert duration < 0.25, f"Conversion took {duration:.2f}s, exceeding 0.25s limit"
        assert data["User"].num_nodes > 0
