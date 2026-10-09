"""Unit and integration tests for Exact Symbolic Reachability Oracle (Step 4.1).

Validates stateful reachability evaluation, multi-hop role chaining,
counterfactual bridge severance, and performance scaling on enterprise topologies.
"""

from __future__ import annotations

import time

import pytest

from iam.baselines import (
    ExactReachabilityOracle,
    OraclePathWitness,
    OracleReachabilityResult,
)
from iam.generator import (
    EnterpriseTopologyConfig,
    EnterpriseTopologyGenerator,
    IAMGraph,
    NodeType,
)
from iam.generator.motifs import AssumeRoleChainMotif, PassRoleLambdaMotif
from iam.parser import CapabilityModel, load_default_capability_model


@pytest.fixture
def capability_model() -> CapabilityModel:
    """Fixture providing loaded AWS CapabilityModel."""
    return load_default_capability_model()


@pytest.fixture
def enterprise_graph(capability_model: CapabilityModel) -> IAMGraph:
    """Fixture generating a deterministic enterprise topology with injected PE chains."""
    cfg = EnterpriseTopologyConfig(
        num_nodes=300,
        seed=42,
        num_pe_chains=2,
        include_branching_chains=True,
        num_branching_chains=1,
    )
    generator = EnterpriseTopologyGenerator(config=cfg, capability_model=capability_model)
    return generator.generate()


class TestExactReachabilityOracle:
    """Test suite for ExactReachabilityOracle."""

    def test_empty_graph_evaluation(self) -> None:
        """Verify evaluation on an empty graph returns 0 pairs."""
        graph = IAMGraph(graph_id="empty_graph")
        oracle = ExactReachabilityOracle()
        result = oracle.evaluate(graph)

        assert isinstance(result, OracleReachabilityResult)
        assert result.num_reachable_pairs == 0
        assert result.num_pe_pairs == 0
        assert len(result.reachable_pairs) == 0
        assert len(result.pe_pairs) == 0
        assert result.is_reachable("nonexistent_s", "nonexistent_t") is False
        assert result.get_witness("nonexistent_s", "nonexistent_t") is None
        assert result.get_targets_for_source("nonexistent_s") == set()

    def test_minimal_pe_motif_evaluation(self) -> None:
        """Verify reachability and PE classification on an injected canonical motif."""
        graph = IAMGraph(graph_id="minimal_pe_graph")
        motif = PassRoleLambdaMotif()
        instance = motif.inject(graph, suffix="test_cpe")

        oracle = ExactReachabilityOracle(max_depth=6)
        result = oracle.evaluate(graph)

        assert result.num_reachable_pairs > 0
        assert result.num_pe_pairs > 0

        # Verify reachability from source to high-value role / crown jewel
        source_id = instance.source_id
        target_id = instance.target_id
        assert result.is_reachable(source_id, target_id) is True

        witness = result.get_witness(source_id, target_id)
        assert witness is not None
        assert isinstance(witness, OraclePathWitness)
        assert witness.source_id == source_id
        assert witness.target_id == target_id
        assert witness.is_privilege_escalation is True
        assert witness.is_bridge_traversed is True
        assert len(witness.bridge_relations) > 0
        assert witness.hop_distance > 0
        assert len(witness.path_nodes) == witness.hop_distance + 1

        # Verify counterfactual severance
        severed = oracle.verify_counterfactual_bridge_severance(graph, witness)
        assert severed is True

    def test_multi_hop_role_chaining_oracle(self) -> None:
        """Verify multi-hop role assumption chaining reachability."""
        graph = IAMGraph(graph_id="chain_pe_graph")
        motif = AssumeRoleChainMotif()
        instance = motif.inject(graph, suffix="test_chain")

        oracle = ExactReachabilityOracle(max_depth=8)
        result = oracle.evaluate(graph)

        source_id = instance.source_id
        target_id = instance.target_id
        assert result.is_reachable(source_id, target_id) is True

        witness = result.get_witness(source_id, target_id)
        assert witness is not None
        assert witness.is_privilege_escalation is True
        assert witness.is_bridge_traversed is True
        assert "AssumesRole" in witness.path_relations

        severed = oracle.verify_counterfactual_bridge_severance(graph, witness)
        assert severed is True

    def test_depth_bounding_behavior(self) -> None:
        """Verify that max_depth cutoff stops exploration beyond hop limit."""
        graph = IAMGraph(graph_id="depth_test_graph")
        motif = AssumeRoleChainMotif()
        instance = motif.inject(graph, suffix="depth_cpe")

        # Hop distance for assume role chain is >= 3
        shallow_oracle = ExactReachabilityOracle(max_depth=1)
        shallow_result = shallow_oracle.evaluate(graph)
        assert shallow_result.is_reachable(instance.source_id, instance.target_id) is False

        deep_oracle = ExactReachabilityOracle(max_depth=8)
        deep_result = deep_oracle.evaluate(graph)
        assert deep_result.is_reachable(instance.source_id, instance.target_id) is True

    def test_explicit_source_and_target_filters(self, enterprise_graph: IAMGraph) -> None:
        """Verify filtering by explicit source_ids and target_ids."""
        users = enterprise_graph.get_nodes_by_type(NodeType.USER)
        assert len(users) >= 2
        selected_source = users[0].id

        hv_targets = enterprise_graph.get_high_value_targets()
        assert len(hv_targets) >= 1
        selected_target = hv_targets[0].id

        oracle = ExactReachabilityOracle(max_depth=8)
        result = oracle.evaluate(
            enterprise_graph,
            source_ids=[selected_source],
            target_ids=[selected_target],
        )

        for w in result.reachable_pairs:
            assert w.source_id == selected_source
            assert w.target_id == selected_target

    def test_enterprise_topology_oracle(self, enterprise_graph: IAMGraph) -> None:
        """Verify exhaustive oracle reachability on a full enterprise topology."""
        oracle = ExactReachabilityOracle(max_depth=6)
        result = oracle.evaluate(enterprise_graph)

        assert result.num_nodes == enterprise_graph.num_nodes
        assert result.num_edges == enterprise_graph.num_edges
        assert result.num_reachable_pairs > 0
        assert result.num_pe_pairs > 0
        assert result.computation_time_sec > 0.0

        # Verify that all PE pairs have valid witnesses
        for pe in result.pe_pairs:
            assert pe.is_privilege_escalation is True
            assert pe.hop_distance >= 1
            assert len(pe.path_nodes) >= 2
            assert pe.source_id == pe.path_nodes[0]
            assert pe.target_id == pe.path_nodes[-1]

    def test_counterfactual_non_bridge_witness_returns_false(self) -> None:
        """Verify that counterfactual check returns False for witnesses without bridges."""
        graph = IAMGraph(graph_id="dummy_graph")
        oracle = ExactReachabilityOracle()
        dummy_witness = OraclePathWitness(
            source_id="u1",
            source_type=NodeType.USER,
            target_id="r1",
            target_type=NodeType.ROLE,
            hop_distance=1,
            path_nodes=["u1", "r1"],
            is_bridge_traversed=False,
            bridge_relations=[],
        )
        assert oracle.verify_counterfactual_bridge_severance(graph, dummy_witness) is False

    @pytest.mark.parametrize("scale,max_sec", [(250, 0.50), (500, 1.50)])
    def test_oracle_scaling_benchmark(
        self, capability_model: CapabilityModel, scale: int, max_sec: float
    ) -> None:
        """Benchmark runtime scalability of ExactReachabilityOracle across graph sizes."""
        cfg = EnterpriseTopologyConfig(num_nodes=scale, seed=100 + scale, num_pe_chains=2)
        gen = EnterpriseTopologyGenerator(config=cfg, capability_model=capability_model)
        graph = gen.generate()

        start = time.perf_counter()
        oracle = ExactReachabilityOracle(max_depth=6)
        result = oracle.evaluate(graph)
        elapsed = time.perf_counter() - start

        assert elapsed < max_sec, f"Scale {scale}: Elapsed {elapsed:.3f}s exceeded limit {max_sec}s"
        assert result.num_reachable_pairs > 0
