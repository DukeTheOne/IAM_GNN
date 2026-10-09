"""Unit and validation tests for Deterministic BFS Reachability Engine (Step 4.2).

Validates stateful BFS traversal (PMapper proxy), cycle avoidance, depth bounding,
direct PyG HeteroData Go ingestion, and empirical recall failure (0% recall)
under Condition E Adversarial Bridge Masking.
"""

from __future__ import annotations

import time

import pytest

from iam.baselines import (
    DeterministicBFSEngine,
    ExactReachabilityOracle,
)
from iam.generator import (
    EnterpriseTopologyConfig,
    EnterpriseTopologyGenerator,
    IAMGraph,
    NodeType,
)
from iam.generator.motifs import AssumeRoleChainMotif, PassRoleLambdaMotif
from iam.models import (
    ActionVocabulary,
    IAMHeteroDataConverter,
    StructuredMaskingSuite,
)
from iam.parser import CapabilityModel, load_default_capability_model


@pytest.fixture
def capability_model() -> CapabilityModel:
    """Fixture providing loaded AWS CapabilityModel."""
    return load_default_capability_model()


@pytest.fixture
def enterprise_graph(capability_model: CapabilityModel) -> IAMGraph:
    """Fixture generating a deterministic enterprise topology with injected PE chains."""
    cfg = EnterpriseTopologyConfig(
        num_nodes=250,
        seed=42,
        num_pe_chains=2,
        include_branching_chains=True,
        num_branching_chains=1,
    )
    generator = EnterpriseTopologyGenerator(config=cfg, capability_model=capability_model)
    return generator.generate()


class TestDeterministicBFSEngine:
    """Test suite for DeterministicBFSEngine."""

    def test_empty_graph_evaluation(self) -> None:
        """Verify BFS evaluation on an empty graph returns 0 pairs."""
        graph = IAMGraph(graph_id="empty_bfs_graph")
        engine = DeterministicBFSEngine()
        result = engine.evaluate(graph)

        assert result.num_reachable_pairs == 0
        assert result.num_pe_pairs == 0
        assert len(result.reachable_pairs) == 0
        assert len(result.pe_pairs) == 0
        assert result.is_reachable("u1", "r1") is False
        assert result.get_witness("u1", "r1") is None
        assert result.get_targets_for_source("u1") == set()

    def test_bfs_on_unmasked_minimal_motif(self) -> None:
        """Verify BFS reaches target when unmasked."""
        graph = IAMGraph(graph_id="unmasked_motif_graph")
        motif = PassRoleLambdaMotif()
        instance = motif.inject(graph, suffix="bfs_unmasked")

        engine = DeterministicBFSEngine(max_depth=6)
        result = engine.evaluate(graph)

        assert result.is_reachable(instance.source_id, instance.target_id) is True
        witness = result.get_witness(instance.source_id, instance.target_id)
        assert witness is not None
        assert witness.hop_distance > 0
        assert len(witness.path_nodes) == witness.hop_distance + 1
        assert result.num_pe_pairs > 0

    def test_condition_e_blindness_failure(self) -> None:
        """Formally verify that under Condition E, BFS recall on severed attack paths drops to 0%."""
        graph = IAMGraph(graph_id="cond_e_test_graph")
        motif = PassRoleLambdaMotif()
        instance = motif.inject(graph, suffix="cond_e")

        # 1. Oracle establishes 100% reachability on unmasked graph
        oracle = ExactReachabilityOracle(max_depth=6)
        oracle_res = oracle.evaluate(graph)
        assert oracle_res.is_reachable(instance.source_id, instance.target_id) is True
        assert len(oracle_res.pe_pairs) > 0

        # 2. Apply Condition E Adversarial Bridge Masking to graph
        masking_suite = StructuredMaskingSuite()
        operator = masking_suite.get_operator("E")
        masked_graph, hidden_edges = operator.apply_to_graph(graph, seed=42)
        assert len(hidden_edges) > 0

        # 3. BFS traverses observed graph Go
        bfs_engine = DeterministicBFSEngine(max_depth=6)
        bfs_res = bfs_engine.evaluate(masked_graph)

        # 4. Formally prove BFS recall drops to 0.0 on severed attack path
        assert bfs_res.is_reachable(instance.source_id, instance.target_id) is False
        bridge_pe_pairs = [p for p in oracle_res.pe_pairs if p.is_bridge_traversed]
        recall = DeterministicBFSEngine.compute_oracle_recall(bfs_res, bridge_pe_pairs)
        assert recall == 0.0, f"Expected 0.0 recall under Condition E, got {recall}"

    def test_condition_e_on_hetero_data(self) -> None:
        """Verify BFS evaluation over PyG HeteroData Go under Condition E."""
        graph = IAMGraph(graph_id="hetero_cond_e_graph")
        motif = PassRoleLambdaMotif()
        instance = motif.inject(graph, suffix="hetero_ce")

        vocab = ActionVocabulary(
            actions=["iam:PassRole", "lambda:CreateFunction", "lambda:InvokeFunction"]
        )
        converter = IAMHeteroDataConverter(vocabulary=vocab, include_reverse_edges=True)
        hetero_data = converter.convert(graph)

        # Apply Condition E
        masking_suite = StructuredMaskingSuite()
        mask_res = masking_suite.apply_condition(data=hetero_data, condition="E", seed=42)

        bfs_engine = DeterministicBFSEngine(max_depth=6)
        bfs_res = bfs_engine.evaluate(mask_res.observed_data)

        # Target must not be reachable in observed PyG data
        assert bfs_res.is_reachable(instance.source_id, instance.target_id) is False

    def test_condition_a_random_masking_degradation(self, enterprise_graph: IAMGraph) -> None:
        """Verify that BFS recall degrades as random edge dropout probability p increases."""
        oracle = ExactReachabilityOracle(max_depth=6)
        oracle_res = oracle.evaluate(enterprise_graph)
        assert len(oracle_res.pe_pairs) > 0

        masking_suite = StructuredMaskingSuite()
        recalls: list[float] = []

        # Evaluate at p=0.0 (unmasked), p=0.20, and p=0.50
        for p in (0.0, 0.20, 0.50):
            if p == 0.0:
                masked_g = enterprise_graph
            else:
                op = masking_suite.get_operator("A", mask_ratio=p)
                masked_g, _ = op.apply_to_graph(enterprise_graph, seed=42)

            bfs_engine = DeterministicBFSEngine(max_depth=6)
            bfs_res = bfs_engine.evaluate(masked_g)
            rec = DeterministicBFSEngine.compute_oracle_recall(bfs_res, oracle_res.pe_pairs)
            recalls.append(rec)

        # Baseline at p=0.0 must be highest; p=0.50 must show substantial degradation
        assert recalls[0] >= recalls[1] >= recalls[2]
        assert recalls[0] > 0.0
        assert recalls[2] < recalls[0]

    def test_depth_bounding_cutoff(self) -> None:
        """Verify that depth cutoff prevents finding paths beyond max_depth."""
        graph = IAMGraph(graph_id="depth_bfs_graph")
        motif = AssumeRoleChainMotif()
        instance = motif.inject(graph, suffix="bfs_depth")

        # AssumeRoleChain has hop distance >= 3
        shallow_engine = DeterministicBFSEngine(max_depth=1)
        shallow_res = shallow_engine.evaluate(graph)
        assert shallow_res.is_reachable(instance.source_id, instance.target_id) is False

        deep_engine = DeterministicBFSEngine(max_depth=6)
        deep_res = deep_engine.evaluate(graph)
        assert deep_res.is_reachable(instance.source_id, instance.target_id) is True

    def test_explicit_source_and_target_filters(self, enterprise_graph: IAMGraph) -> None:
        """Verify explicit source and target filtering in BFS traversal."""
        users = enterprise_graph.get_nodes_by_type(NodeType.USER)
        targets = enterprise_graph.get_high_value_targets()
        assert len(users) >= 1 and len(targets) >= 1

        src_id = users[0].id
        tgt_id = targets[0].id

        engine = DeterministicBFSEngine(max_depth=6)
        res = engine.evaluate(enterprise_graph, source_ids=[src_id], target_ids=[tgt_id])

        for w in res.reachable_pairs:
            assert w.source_id == src_id
            assert w.target_id == tgt_id

    @pytest.mark.parametrize("scale,max_sec", [(250, 0.50), (500, 1.50)])
    def test_bfs_scaling_performance(
        self, capability_model: CapabilityModel, scale: int, max_sec: float
    ) -> None:
        """Benchmark runtime scalability of DeterministicBFSEngine."""
        cfg = EnterpriseTopologyConfig(num_nodes=scale, seed=200 + scale, num_pe_chains=2)
        gen = EnterpriseTopologyGenerator(config=cfg, capability_model=capability_model)
        graph = gen.generate()

        start = time.perf_counter()
        engine = DeterministicBFSEngine(max_depth=6)
        result = engine.evaluate(graph)
        elapsed = time.perf_counter() - start

        assert elapsed < max_sec, f"Scale {scale}: Elapsed {elapsed:.3f}s exceeded {max_sec}s"
        assert result.num_nodes == graph.num_nodes
