"""Comprehensive unit and integration tests for canonical PE motifs and IAMGraph."""

import pytest

from iam.generator import (
    AssumeRoleChainMotif,
    AttachPolicyMotif,
    CreateAccessKeyMotif,
    EdgeRelation,
    GraphEdge,
    GraphNode,
    IAMGraph,
    NodeType,
    PassRoleEC2Motif,
    PassRoleLambdaMotif,
    SetDefaultPolicyVersionMotif,
    default_motif_registry,
)
from iam.parser import CapabilityModel, load_default_capability_model


class TestIAMGraphDataStructure:
    """Tests for IAMGraph node/edge indexing, serialization, and ego graph extraction."""

    def test_graph_node_and_edge_addition(self) -> None:
        graph = IAMGraph(graph_id="test_graph")
        u = GraphNode(
            id="user:alice",
            node_type=NodeType.USER,
            arn="arn:aws:iam::123456789012:user/alice",
            name="alice",
        )
        r = GraphNode(
            id="role:worker",
            node_type=NodeType.ROLE,
            arn="arn:aws:iam::123456789012:role/worker",
            name="worker",
        )
        graph.add_node(u)
        graph.add_node(r)

        assert graph.num_nodes == 2
        assert graph.get_node("user:alice") == u
        assert graph.get_node_by_arn("arn:aws:iam::123456789012:role/worker") == r
        assert len(graph.get_nodes_by_type(NodeType.USER)) == 1

        edge = GraphEdge(
            source="user:alice",
            target="role:worker",
            relation=EdgeRelation.ASSUMES_ROLE,
            is_bridge=True,
            actions=["sts:AssumeRole"],
        )
        graph.add_edge(edge)

        assert graph.num_edges == 1
        assert graph.has_edge("user:alice", "role:worker", EdgeRelation.ASSUMES_ROLE)
        assert not graph.has_edge("user:alice", "role:worker", EdgeRelation.PASSES_TO)
        assert len(graph.get_bridge_edges()) == 1

    def test_missing_node_edge_addition_fails(self) -> None:
        graph = IAMGraph()
        edge = GraphEdge(
            source="nonexistent_source",
            target="nonexistent_target",
            relation=EdgeRelation.ACTS_ON,
        )
        with pytest.raises(KeyError, match="Source node .* does not exist"):
            graph.add_edge(edge)

    def test_graph_serialization_and_deserialization(self) -> None:
        graph = IAMGraph(graph_id="serial_test")
        node = GraphNode(
            id="resource:s3-data",
            node_type=NodeType.RESOURCE,
            arn="arn:aws:s3:::company-data",
            name="company-data",
            is_high_value=True,
        )
        graph.add_node(node)

        data = graph.to_dict()
        assert data["graph_id"] == "serial_test"
        assert data["num_nodes"] == 1
        assert len(data["nodes"]) == 1

        reconstructed = IAMGraph.from_dict(data)
        assert reconstructed.graph_id == "serial_test"
        assert reconstructed.num_nodes == 1
        rec_node = reconstructed.get_node("resource:s3-data")
        assert rec_node is not None
        assert rec_node.is_high_value is True
        assert rec_node.arn == "arn:aws:s3:::company-data"

    def test_graph_cloning_and_edge_removal(self) -> None:
        graph = IAMGraph()
        n1 = GraphNode(id="u1", node_type=NodeType.USER, arn="arn:aws:iam::123:user/u1", name="u1")
        n2 = GraphNode(id="u2", node_type=NodeType.USER, arn="arn:aws:iam::123:user/u2", name="u2")
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_edge(GraphEdge(source="u1", target="u2", relation=EdgeRelation.ACTS_ON))

        cloned = graph.clone()
        assert cloned.num_nodes == 2
        assert cloned.num_edges == 1

        # Remove edge from original
        removed = graph.remove_edge("u1", "u2", EdgeRelation.ACTS_ON)
        assert removed is True
        assert graph.num_edges == 0
        # Clone remains unchanged
        assert cloned.num_edges == 1

    def test_ego_graph_extraction(self) -> None:
        graph = IAMGraph()
        nodes = [
            GraphNode(
                id=f"n{i}", node_type=NodeType.USER, arn=f"arn:aws:iam::1:user/n{i}", name=f"n{i}"
            )
            for i in range(5)
        ]
        for n in nodes:
            graph.add_node(n)

        # n0 -> n1 -> n2 -> n3 -> n4
        for i in range(4):
            graph.add_edge(
                GraphEdge(source=f"n{i}", target=f"n{i + 1}", relation=EdgeRelation.ACTS_ON)
            )

        ego = graph.get_ego_graph("n0", radius=2)
        # Should contain n0, n1, n2 (within 2 hops)
        assert ego.num_nodes == 3
        assert ego.get_node("n0") is not None
        assert ego.get_node("n1") is not None
        assert ego.get_node("n2") is not None
        assert ego.get_node("n3") is None


class TestMotifRegistry:
    """Tests verifying the dynamic motif registry."""

    def test_registry_contains_canonical_motifs(self) -> None:
        assert len(default_motif_registry) == 6
        p0_motifs = default_motif_registry.list_motifs(tier="P0")
        assert len(p0_motifs) == 4
        p1_motifs = default_motif_registry.list_motifs(tier="P1")
        assert len(p1_motifs) == 2

    def test_registry_lookup_and_unknown_key(self) -> None:
        motif = default_motif_registry.get("passrole_lambda")
        assert motif.tier == "P0"
        assert motif.required_actions == [
            "iam:PassRole",
            "lambda:CreateFunction",
            "lambda:InvokeFunction",
        ]

        with pytest.raises(KeyError, match="Unknown motif type"):
            default_motif_registry.get("nonexistent_motif")


class TestCanonicalPEMotifs:
    """Comprehensive verification of all 4 P0 and 2 P1 motifs."""

    @pytest.fixture
    def capability_model(self) -> CapabilityModel:
        return load_default_capability_model()

    def test_p0_motif_passrole_lambda(self, capability_model: CapabilityModel) -> None:
        graph = IAMGraph()
        motif = PassRoleLambdaMotif()
        instance = motif.inject(graph, suffix="test1")

        assert instance.motif_type == "passrole_lambda"
        assert instance.tier == "P0"
        assert len(instance.required_actions) == 3

        # Verify ground truth reachability and counterfactual bridge
        verification = motif.verify_ground_truth(graph, instance)
        assert verification.is_valid is True
        assert verification.path_exists is True
        assert verification.bridge_breaks_path is True
        assert len(verification.traversed_path) >= 3

        # Verify all generated policies are compliant with capability model
        for node in graph.get_nodes_by_type(NodeType.POLICY):
            if node.policy_document:
                for stmt in node.policy_document.statements:
                    errors = capability_model.validate_statement(stmt)
                    assert errors == [], f"Statement failed capability validation: {errors}"

    def test_p0_motif_create_access_key(self, capability_model: CapabilityModel) -> None:
        graph = IAMGraph()
        motif = CreateAccessKeyMotif()
        instance = motif.inject(graph, suffix="test2")

        assert instance.motif_type == "create_access_key"
        assert instance.tier == "P0"

        verification = motif.verify_ground_truth(graph, instance)
        assert verification.is_valid is True
        assert verification.path_exists is True
        assert verification.bridge_breaks_path is True

        for node in graph.get_nodes_by_type(NodeType.POLICY):
            if node.policy_document:
                for stmt in node.policy_document.statements:
                    assert capability_model.validate_statement(stmt) == []

    def test_p0_motif_attach_policy(self, capability_model: CapabilityModel) -> None:
        graph = IAMGraph()
        motif = AttachPolicyMotif()
        instance = motif.inject(graph, suffix="test3")

        assert instance.motif_type == "attach_policy"
        assert instance.tier == "P0"

        verification = motif.verify_ground_truth(graph, instance)
        assert verification.is_valid is True
        assert verification.path_exists is True
        assert verification.bridge_breaks_path is True

        for node in graph.get_nodes_by_type(NodeType.POLICY):
            if node.policy_document:
                for stmt in node.policy_document.statements:
                    assert capability_model.validate_statement(stmt) == []

    def test_p0_motif_assume_role_chain(self, capability_model: CapabilityModel) -> None:
        graph = IAMGraph()
        motif = AssumeRoleChainMotif()
        instance = motif.inject(graph, suffix="test4")

        assert instance.motif_type == "assume_role_chain"
        assert instance.tier == "P0"
        # Chain should involve intermediate nodes
        assert len(instance.intermediate_node_ids) >= 3

        verification = motif.verify_ground_truth(graph, instance)
        assert verification.is_valid is True
        assert verification.path_exists is True
        assert verification.bridge_breaks_path is True

        for node in graph.get_nodes_by_type(NodeType.POLICY):
            if node.policy_document:
                for stmt in node.policy_document.statements:
                    assert capability_model.validate_statement(stmt) == []

    def test_p1_motif_passrole_ec2(self, capability_model: CapabilityModel) -> None:
        graph = IAMGraph()
        motif = PassRoleEC2Motif()
        instance = motif.inject(graph, suffix="test5")

        assert instance.motif_type == "passrole_ec2"
        assert instance.tier == "P1"

        verification = motif.verify_ground_truth(graph, instance)
        assert verification.is_valid is True
        assert verification.path_exists is True
        assert verification.bridge_breaks_path is True

        for node in graph.get_nodes_by_type(NodeType.POLICY):
            if node.policy_document:
                for stmt in node.policy_document.statements:
                    assert capability_model.validate_statement(stmt) == []

    def test_p1_motif_set_default_policy_version(self, capability_model: CapabilityModel) -> None:
        graph = IAMGraph()
        motif = SetDefaultPolicyVersionMotif()
        instance = motif.inject(graph, suffix="test6")

        assert instance.motif_type == "set_default_policy_version"
        assert instance.tier == "P1"

        verification = motif.verify_ground_truth(graph, instance)
        assert verification.is_valid is True
        assert verification.path_exists is True
        assert verification.bridge_breaks_path is True

        for node in graph.get_nodes_by_type(NodeType.POLICY):
            if node.policy_document:
                for stmt in node.policy_document.statements:
                    assert capability_model.validate_statement(stmt) == []

    def test_multi_motif_coexistence_in_single_graph(self) -> None:
        """Verify multiple motifs can be injected into the same graph without conflict."""
        graph = IAMGraph(graph_id="multi_motif_graph")
        motifs = default_motif_registry.list_motifs()

        instances = []
        for i, motif in enumerate(motifs):
            inst = motif.inject(graph, suffix=f"batch_{i}")
            instances.append((motif, inst))

        # Graph should now contain all 6 distinct motifs
        assert len(instances) == 6
        assert graph.num_nodes > 25
        assert len(graph.get_bridge_edges()) == 6

        # Verify ground truth for every injected motif independently
        for motif, inst in instances:
            verif = motif.verify_ground_truth(graph, inst)
            assert verif.is_valid is True
            assert verif.path_exists is True
            assert verif.bridge_breaks_path is True
