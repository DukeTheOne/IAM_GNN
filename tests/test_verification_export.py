"""Comprehensive unit and integration tests for Ground-Truth Verification and Environment Export.

Tests GroundTruthLabeler, TopologicalSanityChecker, and EnvironmentExporter.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iam.generator import (
    DistributionStats,
    EdgeRelation,
    EnterpriseTopologyConfig,
    EnterpriseTopologyGenerator,
    EnvironmentExporter,
    GraphEdge,
    GraphNode,
    GroundTruthLabeler,
    GroundTruthLabelSet,
    IAMGraph,
    NodeType,
    PathEdgeWitness,
    ReachablePairWitness,
    TopologicalMetricsReport,
    TopologicalSanityChecker,
)
from iam.parser import CapabilityModel, load_default_capability_model


@pytest.fixture
def capability_model() -> CapabilityModel:
    """Fixture providing loaded AWS CapabilityModel."""
    return load_default_capability_model()


@pytest.fixture
def small_enterprise_graph(capability_model: CapabilityModel) -> IAMGraph:
    """Fixture providing a deterministic enterprise graph with canonical and branching PE chains."""
    cfg = EnterpriseTopologyConfig(
        num_nodes=150,
        seed=42,
        num_pe_chains=3,
        include_branching_chains=True,
        num_branching_chains=1,
    )
    generator = EnterpriseTopologyGenerator(config=cfg, capability_model=capability_model)
    return generator.generate()


@pytest.fixture
def synthetic_pe_chain_graph() -> IAMGraph:
    """Fixture providing a deterministic minimal graph with explicit known PE chain."""
    graph = IAMGraph(graph_id="minimal_pe_test")

    # Entrypoint user (non-admin, Interns)
    u_intern = GraphNode(
        id="user_intern",
        node_type=NodeType.USER,
        arn="arn:aws:iam::123456789012:user/intern_bob",
        name="intern_bob",
        department="Interns",
        is_admin=False,
    )
    # Intermediate role
    r_dev = GraphNode(
        id="role_dev",
        node_type=NodeType.ROLE,
        arn="arn:aws:iam::123456789012:role/dev_role",
        name="dev_role",
        department="DevOps",
        is_admin=False,
    )
    # Admin role (SecOps, Admin, High-Value)
    r_admin = GraphNode(
        id="role_admin",
        node_type=NodeType.ROLE,
        arn="arn:aws:iam::123456789012:role/admin_role",
        name="admin_role",
        department="SecOps",
        is_admin=True,
        is_high_value=True,
    )
    # High-value S3 bucket
    res_vault = GraphNode(
        id="res_vault",
        node_type=NodeType.RESOURCE,
        arn="arn:aws:s3:::corporate-vault-secrets",
        name="corporate-vault-secrets",
        department="SecOps",
        is_admin=False,
        is_high_value=True,
    )
    # Admin user
    u_admin = GraphNode(
        id="user_admin",
        node_type=NodeType.USER,
        arn="arn:aws:iam::123456789012:user/admin_alice",
        name="admin_alice",
        department="SecOps",
        is_admin=True,
    )

    graph.add_nodes([u_intern, r_dev, r_admin, res_vault, u_admin])

    # u_intern -> r_dev (AssumesRole)
    graph.add_edge(
        GraphEdge(
            source="user_intern",
            target="role_dev",
            relation=EdgeRelation.ASSUMES_ROLE,
            actions=["sts:AssumeRole"],
        )
    )
    # r_dev -> r_admin (AssumesRole, marked bridge)
    graph.add_edge(
        GraphEdge(
            source="role_dev",
            target="role_admin",
            relation=EdgeRelation.ASSUMES_ROLE,
            actions=["sts:AssumeRole"],
            is_bridge=True,
            motif_id="test_pe_chain",
        )
    )
    # r_admin -> res_vault (ActsOn)
    graph.add_edge(
        GraphEdge(
            source="role_admin",
            target="res_vault",
            relation=EdgeRelation.ACTS_ON,
            actions=["s3:GetObject"],
        )
    )
    # u_admin -> r_admin (AssumesRole)
    graph.add_edge(
        GraphEdge(
            source="user_admin",
            target="role_admin",
            relation=EdgeRelation.ASSUMES_ROLE,
            actions=["sts:AssumeRole"],
        )
    )

    return graph


class TestGroundTruthLabeler:
    """Tests for GroundTruthLabeler, reachable pair witness extraction, and intermediate edges."""

    def test_labeler_on_minimal_pe_graph(self, synthetic_pe_chain_graph: IAMGraph) -> None:
        labeler = GroundTruthLabeler(max_depth=6, find_all_paths=True)
        label_set = labeler.label_graph(synthetic_pe_chain_graph)

        assert isinstance(label_set, GroundTruthLabelSet)
        assert label_set.graph_id == "minimal_pe_test"
        assert label_set.num_reachable_pairs > 0
        assert label_set.num_pe_pairs > 0

        # Check user_intern -> role_admin PE pair
        pe_targets = {(pe.source_id, pe.target_id): pe for pe in label_set.pe_pairs}
        assert ("user_intern", "role_admin") in pe_targets
        witness = pe_targets[("user_intern", "role_admin")]
        assert isinstance(witness, ReachablePairWitness)

        assert witness.source_is_admin is False
        assert witness.target_is_admin is True
        assert witness.target_is_high_value is True
        assert witness.is_privilege_escalation is True
        assert witness.hop_distance == 2
        assert witness.path_nodes == ["user_intern", "role_dev", "role_admin"]
        assert len(witness.path_edges) == 2

        # Check edge sequence
        e1, e2 = witness.path_edges[0], witness.path_edges[1]
        assert isinstance(e1, PathEdgeWitness)
        assert isinstance(e2, PathEdgeWitness)
        assert e1.source == "user_intern"
        assert e1.target == "role_dev"
        assert e1.relation == EdgeRelation.ASSUMES_ROLE
        assert e1.is_bridge is False

        assert e2.source == "role_dev"
        assert e2.target == "role_admin"
        assert e2.relation == EdgeRelation.ASSUMES_ROLE
        assert e2.is_bridge is True
        assert e2.motif_id == "test_pe_chain"

        # Check bridge relations recorded
        assert len(witness.bridge_relations) == 1
        assert witness.bridge_relations[0] == ("role_dev", "role_admin", "AssumesRole")

    def test_target_resource_reachability(self, synthetic_pe_chain_graph: IAMGraph) -> None:
        labeler = GroundTruthLabeler(max_depth=6)
        label_set = labeler.label_graph(synthetic_pe_chain_graph)

        pe_targets = {(pe.source_id, pe.target_id): pe for pe in label_set.pe_pairs}
        # user_intern -> res_vault should be reachable in 3 hops
        assert ("user_intern", "res_vault") in pe_targets
        w_vault = pe_targets[("user_intern", "res_vault")]
        assert w_vault.hop_distance == 3
        assert w_vault.path_nodes == ["user_intern", "role_dev", "role_admin", "res_vault"]
        assert len(w_vault.path_edges) == 3
        assert w_vault.is_privilege_escalation is True

    def test_non_pe_reachability_classification(self, synthetic_pe_chain_graph: IAMGraph) -> None:
        labeler = GroundTruthLabeler(max_depth=6)
        label_set = labeler.label_graph(synthetic_pe_chain_graph)

        # user_admin is already admin, so user_admin -> role_admin is NOT a privilege escalation
        all_pairs = {(p.source_id, p.target_id): p for p in label_set.reachable_pairs}
        assert ("user_admin", "role_admin") in all_pairs
        w_admin = all_pairs[("user_admin", "role_admin")]
        assert w_admin.source_is_admin is True
        assert w_admin.is_privilege_escalation is False

        # It must NOT be included in label_set.pe_pairs
        pe_pairs_keys = {(pe.source_id, pe.target_id) for pe in label_set.pe_pairs}
        assert ("user_admin", "role_admin") not in pe_pairs_keys

    def test_depth_bounding(self, synthetic_pe_chain_graph: IAMGraph) -> None:
        # Distance to res_vault is 3 hops. With max_depth=2, res_vault should not be reached
        labeler = GroundTruthLabeler(max_depth=2)
        label_set = labeler.label_graph(synthetic_pe_chain_graph)

        reachable_keys = {(p.source_id, p.target_id) for p in label_set.reachable_pairs}
        assert ("user_intern", "role_admin") in reachable_keys  # 2 hops
        assert ("user_intern", "res_vault") not in reachable_keys  # 3 hops

    def test_explicit_source_and_target_filters(self, synthetic_pe_chain_graph: IAMGraph) -> None:
        labeler = GroundTruthLabeler()
        label_set = labeler.label_graph(
            synthetic_pe_chain_graph,
            source_ids=["user_intern"],
            target_ids=["role_admin"],
        )

        assert label_set.num_reachable_pairs == 1
        witness = label_set.reachable_pairs[0]
        assert witness.source_id == "user_intern"
        assert witness.target_id == "role_admin"

    def test_labeler_on_enterprise_graph(self, small_enterprise_graph: IAMGraph) -> None:
        labeler = GroundTruthLabeler(max_depth=8, find_all_paths=False)
        label_set = labeler.label_graph(small_enterprise_graph)

        assert label_set.graph_id == small_enterprise_graph.graph_id
        assert label_set.num_pe_pairs > 0
        assert len(label_set.pe_pairs) == label_set.num_pe_pairs
        assert label_set.computation_time_sec >= 0.0

        # Validate structural integrity of each PE pair witness
        for pe in label_set.pe_pairs:
            assert pe.is_privilege_escalation is True
            assert pe.hop_distance >= 1
            assert len(pe.path_nodes) == pe.hop_distance + 1
            assert len(pe.path_edges) == pe.hop_distance
            # Edges sequence must form a connected directed chain
            for i in range(len(pe.path_edges)):
                assert pe.path_edges[i].source == pe.path_nodes[i]
                assert pe.path_edges[i].target == pe.path_nodes[i + 1]

    def test_empty_graph_labeling(self) -> None:
        empty_graph = IAMGraph(graph_id="empty")
        labeler = GroundTruthLabeler()
        label_set = labeler.label_graph(empty_graph)

        assert label_set.num_reachable_pairs == 0
        assert label_set.num_pe_pairs == 0
        assert label_set.reachable_pairs == []
        assert label_set.pe_pairs == []


class TestTopologicalSanityChecker:
    """Tests for TopologicalSanityChecker, structural metrics, and diagnostics."""

    def test_distribution_stats_computation(self) -> None:
        empty_stat = DistributionStats.from_values([])
        assert empty_stat.min == 0.0
        assert empty_stat.max == 0.0
        assert empty_stat.mean == 0.0

        single_stat = DistributionStats.from_values([42])
        assert single_stat.min == 42.0
        assert single_stat.max == 42.0
        assert single_stat.mean == 42.0
        assert single_stat.median == 42.0
        assert single_stat.std == 0.0

        vals = [10, 20, 30, 40, 50]
        stats = DistributionStats.from_values(vals)
        assert stats.min == 10.0
        assert stats.max == 50.0
        assert stats.mean == 30.0
        assert stats.median == 30.0
        assert stats.std > 0.0

    def test_enterprise_topology_metrics(self, small_enterprise_graph: IAMGraph) -> None:
        checker = TopologicalSanityChecker()
        report = checker.compute_metrics(small_enterprise_graph)

        assert isinstance(report, TopologicalMetricsReport)
        assert report.graph_id == small_enterprise_graph.graph_id
        assert report.total_nodes == small_enterprise_graph.num_nodes
        assert report.total_edges == small_enterprise_graph.num_edges
        assert report.average_degree > 0.0

        # Degree metrics
        assert report.degrees.in_degree.max >= report.degrees.in_degree.min
        assert report.degrees.out_degree.max >= report.degrees.out_degree.min
        for rel in EdgeRelation:
            assert rel.value in report.degrees.relation_degrees

        # Connectivity
        assert report.connectivity.num_weakly_connected_components >= 1
        assert report.connectivity.largest_wcc_size > 0
        assert 0.0 < report.connectivity.giant_component_ratio <= 1.0

        # Clustering and density
        assert 0.0 <= report.clustering.average_clustering_coefficient <= 1.0
        assert 0.0 <= report.clustering.transitivity <= 1.0
        assert report.clustering.edge_density > 0.0

        # Power law and health
        assert report.admin_percentage < 25.0
        assert report.high_value_targets_count > 0
        assert report.bridge_edges_count > 0
        assert report.is_healthy is True
        assert len(report.sanity_warnings) == 0

    def test_degraded_graph_diagnostics(self) -> None:
        # 1. Zero nodes graph
        empty_g = IAMGraph(graph_id="empty_g")
        checker = TopologicalSanityChecker()
        rep_empty = checker.compute_metrics(empty_g)
        assert rep_empty.is_healthy is False
        assert any("zero nodes" in w.lower() for w in rep_empty.sanity_warnings)

        # 2. Graph without high-value targets
        no_hv_g = IAMGraph(graph_id="no_hv_g")
        u = GraphNode(
            id="u1",
            node_type=NodeType.USER,
            arn="arn:aws:iam:::user/u1",
            name="u1",
            is_admin=False,
            is_high_value=False,
        )
        r = GraphNode(
            id="r1",
            node_type=NodeType.RESOURCE,
            arn="arn:aws:s3:::b1",
            name="b1",
            is_admin=False,
            is_high_value=False,
        )
        no_hv_g.add_nodes([u, r])
        no_hv_g.add_edge(GraphEdge(source="u1", target="r1", relation=EdgeRelation.ACTS_ON))

        rep_no_hv = checker.compute_metrics(no_hv_g)
        assert rep_no_hv.is_healthy is False
        assert any("no high-value targets" in w.lower() for w in rep_no_hv.sanity_warnings)


class TestEnvironmentExporter:
    """Tests for EnvironmentExporter, serialization schemas, compression, and deserialization."""

    def test_to_structured_dict(
        self,
        small_enterprise_graph: IAMGraph,
    ) -> None:
        exporter = EnvironmentExporter(use_orjson=True)
        labeler = GroundTruthLabeler(max_depth=5)
        label_set = labeler.label_graph(small_enterprise_graph)
        checker = TopologicalSanityChecker()
        metrics = checker.compute_metrics(small_enterprise_graph)

        payload = exporter.to_structured_dict(
            graph=small_enterprise_graph,
            label_set=label_set,
            metrics_report=metrics,
        )

        assert payload["schema_version"] == "1.0.0"
        assert "graph" in payload
        assert payload["graph"]["graph_id"] == small_enterprise_graph.graph_id
        assert "ground_truth" in payload
        assert payload["ground_truth"]["num_pe_pairs"] == label_set.num_pe_pairs
        assert "topological_metrics" in payload
        assert payload["topological_metrics"]["total_nodes"] == small_enterprise_graph.num_nodes

    def test_to_networkx_node_link_dict(self, small_enterprise_graph: IAMGraph) -> None:
        exporter = EnvironmentExporter()
        nl_dict = exporter.to_networkx_node_link_dict(small_enterprise_graph)

        assert "nodes" in nl_dict
        assert "links" in nl_dict or "edges" in nl_dict
        edge_key = "edges" if "edges" in nl_dict else "links"
        assert len(nl_dict["nodes"]) == small_enterprise_graph.num_nodes
        assert len(nl_dict[edge_key]) == small_enterprise_graph.num_edges

        # Assert no non-serializable objects leaked
        json_str = json.dumps(nl_dict)
        assert len(json_str) > 0

    def test_to_pyg_ready_dict(
        self,
        small_enterprise_graph: IAMGraph,
    ) -> None:
        exporter = EnvironmentExporter()
        labeler = GroundTruthLabeler(max_depth=5)
        label_set = labeler.label_graph(small_enterprise_graph)

        pyg_dict = exporter.to_pyg_ready_dict(small_enterprise_graph, label_set)

        assert "node_mappings" in pyg_dict
        assert "edge_indices" in pyg_dict
        assert "bridge_edge_indices" in pyg_dict
        assert "pe_pairs" in pyg_dict

        # Verify node mappings per NodeType
        for nt in NodeType:
            assert nt.value in pyg_dict["node_mappings"]
            nodes = small_enterprise_graph.get_nodes_by_type(nt)
            assert len(pyg_dict["node_mappings"][nt.value]) == len(nodes)

        # Verify edge indices format [2, E]
        for _rel_key, (src_indices, dst_indices) in pyg_dict["edge_indices"].items():
            assert len(src_indices) == len(dst_indices)
            assert all(isinstance(idx, int) for idx in src_indices)
            assert all(isinstance(idx, int) for idx in dst_indices)

        # Verify bridge indices are valid offsets
        for rel_key, b_indices in pyg_dict["bridge_edge_indices"].items():
            max_idx = len(pyg_dict["edge_indices"][rel_key][0])
            for b_idx in b_indices:
                assert 0 <= b_idx < max_idx

    def test_export_import_plain_json_roundtrip(
        self,
        small_enterprise_graph: IAMGraph,
        tmp_path: Path,
    ) -> None:
        exporter = EnvironmentExporter(use_orjson=True)
        labeler = GroundTruthLabeler(max_depth=5)
        label_set = labeler.label_graph(small_enterprise_graph)
        checker = TopologicalSanityChecker()
        metrics = checker.compute_metrics(small_enterprise_graph)

        out_file = tmp_path / "env_export.json"
        res_path = exporter.export_to_file(
            graph=small_enterprise_graph,
            output_path=out_file,
            label_set=label_set,
            metrics_report=metrics,
            schema="structured",
            compress=False,
        )

        assert res_path.exists()
        assert res_path.stat().st_size > 0

        # Import back
        loaded_g, loaded_labels, loaded_metrics = exporter.import_from_file(res_path)

        assert loaded_g.graph_id == small_enterprise_graph.graph_id
        assert loaded_g.num_nodes == small_enterprise_graph.num_nodes
        assert loaded_g.num_edges == small_enterprise_graph.num_edges

        assert loaded_labels is not None
        assert loaded_labels.num_pe_pairs == label_set.num_pe_pairs
        assert loaded_labels.num_reachable_pairs == label_set.num_reachable_pairs

        assert loaded_metrics is not None
        assert loaded_metrics.total_nodes == metrics.total_nodes
        assert loaded_metrics.is_healthy == metrics.is_healthy

    def test_export_import_gzip_compressed_roundtrip(
        self,
        small_enterprise_graph: IAMGraph,
        tmp_path: Path,
    ) -> None:
        exporter = EnvironmentExporter(use_orjson=True)
        labeler = GroundTruthLabeler(max_depth=5)
        label_set = labeler.label_graph(small_enterprise_graph)
        checker = TopologicalSanityChecker()
        metrics = checker.compute_metrics(small_enterprise_graph)

        plain_file = tmp_path / "env_export.json"
        gz_file = tmp_path / "env_export.json.gz"

        exporter.export_to_file(
            graph=small_enterprise_graph,
            output_path=plain_file,
            label_set=label_set,
            metrics_report=metrics,
            compress=False,
        )

        exporter.export_to_file(
            graph=small_enterprise_graph,
            output_path=gz_file,
            label_set=label_set,
            metrics_report=metrics,
            compress=True,
        )

        assert gz_file.exists()
        # Compressed file must be significantly smaller than plain JSON
        assert gz_file.stat().st_size < plain_file.stat().st_size

        # Roundtrip decompression and deserialization
        loaded_g, loaded_labels, loaded_metrics = exporter.import_from_file(gz_file)
        assert loaded_g.num_nodes == small_enterprise_graph.num_nodes
        assert loaded_g.num_edges == small_enterprise_graph.num_edges
        assert loaded_labels is not None
        assert loaded_labels.num_pe_pairs == label_set.num_pe_pairs

    def test_export_import_networkx_schema_roundtrip(
        self,
        small_enterprise_graph: IAMGraph,
        tmp_path: Path,
    ) -> None:
        exporter = EnvironmentExporter(use_orjson=True)
        nx_file = tmp_path / "networkx_export.json"

        exporter.export_to_file(
            graph=small_enterprise_graph,
            output_path=nx_file,
            schema="networkx",
        )

        loaded_g, _, _ = exporter.import_from_file(nx_file)
        assert loaded_g.num_nodes == small_enterprise_graph.num_nodes
        assert loaded_g.num_edges == small_enterprise_graph.num_edges

    def test_fallback_without_orjson(
        self,
        small_enterprise_graph: IAMGraph,
        tmp_path: Path,
    ) -> None:
        # Test standard library json engine
        exporter = EnvironmentExporter(use_orjson=False)
        out_file = tmp_path / "fallback_export.json"

        exporter.export_to_file(
            graph=small_enterprise_graph,
            output_path=out_file,
        )

        loaded_g, _, _ = exporter.import_from_file(out_file)
        assert loaded_g.num_nodes == small_enterprise_graph.num_nodes

    def test_file_not_found_and_invalid_schema(self, tmp_path: Path) -> None:
        exporter = EnvironmentExporter()
        with pytest.raises(FileNotFoundError):
            exporter.import_from_file(tmp_path / "nonexistent.json")

        bad_file = tmp_path / "invalid.json"
        bad_file.write_text(json.dumps({"some_key": 123}), encoding="utf-8")
        with pytest.raises(ValueError, match="Unrecognized dataset export schema"):
            exporter.import_from_file(bad_file)
