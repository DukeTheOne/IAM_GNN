"""Comprehensive unit and integration tests for uncertainty-guided candidate pruning.

Tests Phase 3 Week 7 (Step 7.2):
- PruningConfig initialization and validation bounds.
- ScoredCutEdge creation, keys, and serialization.
- CandidatePruner edge scoring, frequency tracking, and budget truncation (max_candidates <= 50).
- Candidate Recall and Optimality Gap metrics computation (RQ3).
- Canonical P0 PassRole+Lambda motif attack path profiling and cut extraction.
- Runtime performance verification (< 3.0 s interactive latency target).
"""

from __future__ import annotations

import math

import pytest

from iam.baselines.oracle import ExactReachabilityOracle, OraclePathWitness
from iam.generator import (
    EdgeRelation,
    GraphEdge,
    GraphNode,
    IAMGraph,
    NodeType,
    PassRoleLambdaMotif,
)
from iam.optimizer import (
    CandidateEdge,
    CandidatePruner,
    LatentCompletion,
    LatentCompletionSampler,
    PruningConfig,
    PruningResult,
    ScoredCutEdge,
    compute_candidate_recall,
    compute_optimality_gap,
)


class TestPruningConfig:
    """Tests for PruningConfig parameters and validation."""

    def test_default_config(self) -> None:
        cfg = PruningConfig()
        assert cfg.max_candidates == 50
        assert cfg.min_frequency == 0.0
        assert cfg.only_observed is True
        assert cfg.weight_frequency == 0.6
        assert cfg.weight_centrality == 0.2
        assert cfg.weight_capability == 0.2

    def test_invalid_max_candidates_raises_error(self) -> None:
        with pytest.raises(ValueError, match="max_candidates must be positive"):
            PruningConfig(max_candidates=0)

    def test_invalid_min_frequency_raises_error(self) -> None:
        with pytest.raises(ValueError, match="min_frequency must be in"):
            PruningConfig(min_frequency=-0.1)
        with pytest.raises(ValueError, match="min_frequency must be in"):
            PruningConfig(min_frequency=1.1)


class TestScoredCutEdge:
    """Tests for ScoredCutEdge data structures."""

    def test_scored_cut_edge_properties_and_serialization(self) -> None:
        edge = ScoredCutEdge(
            source="user:alice",
            target="role:worker",
            relation=EdgeRelation.PASSES_TO,
            frequency=0.85,
            path_count=12,
            capability_score=1.0,
            score=0.91,
            is_observed=True,
            actions=("iam:PassRole",),
        )
        assert edge.key == ("user:alice", "role:worker", "PassesTo")
        d = edge.to_dict()
        assert d["source"] == "user:alice"
        assert d["target"] == "role:worker"
        assert d["relation"] == "PassesTo"
        assert d["frequency"] == 0.85
        assert d["actions"] == ["iam:PassRole"]


class TestCandidatePrunerBasic:
    """Tests for basic CandidatePruner filtering, scoring, and subgraph extraction."""

    @pytest.fixture
    def linear_attack_graph(self) -> IAMGraph:
        """Graph with linear chain: s -> a -> b -> t."""
        graph = IAMGraph(graph_id="linear_attack")
        s = GraphNode(id="s", node_type=NodeType.USER, arn="arn:aws:iam::1:user/s", name="s")
        a = GraphNode(id="a", node_type=NodeType.ROLE, arn="arn:aws:iam::1:role/a", name="a")
        b = GraphNode(id="b", node_type=NodeType.ROLE, arn="arn:aws:iam::1:role/b", name="b")
        t = GraphNode(
            id="t", node_type=NodeType.ROLE, arn="arn:aws:iam::1:role/t", name="t", is_admin=True
        )
        graph.add_nodes([s, a, b, t])

        graph.add_edge(GraphEdge(source="s", target="a", relation=EdgeRelation.ASSUMES_ROLE))
        graph.add_edge(
            GraphEdge(
                source="a",
                target="b",
                relation=EdgeRelation.PASSES_TO,
                actions=["iam:PassRole"],
            )
        )
        graph.add_edge(
            GraphEdge(
                source="b",
                target="t",
                relation=EdgeRelation.ACTS_ON,
                actions=["lambda:InvokeFunction"],
            )
        )
        return graph

    def test_pruning_with_empty_completions(self, linear_attack_graph: IAMGraph) -> None:
        pruner = CandidatePruner()
        res = pruner.prune_from_completions(graph=linear_attack_graph, completions=[])
        assert isinstance(res, PruningResult)
        assert res.num_pruned_candidates == 0
        assert res.total_edges_considered == 0

    def test_pruning_truncation_to_max_candidates(self, linear_attack_graph: IAMGraph) -> None:
        cfg = PruningConfig(max_candidates=2)
        pruner = CandidatePruner(config=cfg)

        witness = OraclePathWitness(
            source_id="s",
            source_type=NodeType.USER,
            target_id="t",
            target_type=NodeType.ROLE,
            target_is_admin=True,
            hop_distance=3,
            path_nodes=["s", "a", "b", "t"],
            path_relations=["AssumesRole", "PassesTo", "ActsOn"],
            is_privilege_escalation=True,
        )
        witnesses_by_sample = [[witness], [witness], [witness]]

        res = pruner.prune_from_witnesses(
            graph=linear_attack_graph,
            witnesses_by_sample=witnesses_by_sample,
            num_samples=3,
        )
        assert res.num_pruned_candidates <= 2
        assert res.total_edges_considered == 3
        assert len(res.get_top_k(1)) == 1

    def test_min_frequency_filtering(self, linear_attack_graph: IAMGraph) -> None:
        # Edge (s, a) appears in 1/4 samples (freq 0.25)
        # Edge (a, b) appears in 4/4 samples (freq 1.0)
        w_all = OraclePathWitness(
            source_id="a",
            source_type=NodeType.ROLE,
            target_id="b",
            target_type=NodeType.ROLE,
            hop_distance=1,
            path_nodes=["a", "b"],
            path_relations=["PassesTo"],
        )
        w_rare = OraclePathWitness(
            source_id="s",
            source_type=NodeType.USER,
            target_id="a",
            target_type=NodeType.ROLE,
            hop_distance=1,
            path_nodes=["s", "a"],
            path_relations=["AssumesRole"],
        )
        witnesses_by_sample = [
            [w_all, w_rare],
            [w_all],
            [w_all],
            [w_all],
        ]

        cfg = PruningConfig(min_frequency=0.5)
        pruner = CandidatePruner(config=cfg)
        res = pruner.prune_from_witnesses(
            graph=linear_attack_graph,
            witnesses_by_sample=witnesses_by_sample,
            num_samples=4,
        )

        keys = res.candidate_edge_keys
        assert ("a", "b", "PassesTo") in keys
        assert ("s", "a", "AssumesRole") not in keys

    def test_candidate_subgraph_extraction(self, linear_attack_graph: IAMGraph) -> None:
        pruner = CandidatePruner()
        witness = OraclePathWitness(
            source_id="s",
            source_type=NodeType.USER,
            target_id="t",
            target_type=NodeType.ROLE,
            hop_distance=2,
            path_nodes=["s", "a", "b"],
            path_relations=["AssumesRole", "PassesTo"],
        )
        res = pruner.prune_from_witnesses(
            graph=linear_attack_graph,
            witnesses_by_sample=[[witness]],
            num_samples=1,
        )
        sub = res.candidate_subgraph
        assert sub is not None
        assert sub.num_nodes >= 2
        assert sub.get_node("s") is not None
        assert sub.get_node("a") is not None


class TestCandidatePrunerWithMotif:
    """Tests profiling and candidate cut extraction on canonical P0 motifs."""

    @pytest.fixture
    def passrole_lambda_completions(
        self,
    ) -> tuple[IAMGraph, list[LatentCompletion], str, str, tuple[str, str, str]]:
        graph = IAMGraph(graph_id="pe_motif_env")
        motif = PassRoleLambdaMotif()
        instance = motif.inject(graph, suffix="prune_test")

        src, dst, rel = instance.bridge_relation
        bridge_key = (src, dst, rel)

        # Sever bridge in Go to simulate Condition E
        graph.remove_edge(src, dst, EdgeRelation(rel))

        cand = CandidateEdge(
            source=src,
            target=dst,
            relation=EdgeRelation(rel),
            probability=0.85,
            is_bridge=True,
        )

        sampler = LatentCompletionSampler(observed_graph=graph, candidate_edges=[cand])
        completions = sampler.sample_completions(num_samples=30, seed=123)
        return graph, completions, instance.source_id, instance.target_id, bridge_key

    def test_pruning_extracts_observed_bottlenecks(
        self,
        passrole_lambda_completions: tuple[
            IAMGraph, list[LatentCompletion], str, str, tuple[str, str, str]
        ],
    ) -> None:
        graph, completions, source_id, target_id, bridge_key = passrole_lambda_completions

        pruner = CandidatePruner(config=PruningConfig(max_candidates=10, only_observed=True))
        oracle = ExactReachabilityOracle()

        result = pruner.prune_from_completions(
            graph=graph,
            completions=completions,
            oracle=oracle,
            source_ids=[source_id],
            target_ids=[target_id],
        )

        assert result.num_pruned_candidates > 0
        assert result.total_witnesses_analyzed > 0

        # All pruned edges must be observed edges in Go
        for edge in result.pruned_edges:
            assert edge.is_observed is True
            assert graph.has_edge(edge.source, edge.target, edge.relation)

        # The masked bridge itself is latent, so it must not be in the observed policy cut set
        assert bridge_key not in result.candidate_edge_keys

        # Subgraph contains the candidates
        sub = result.candidate_subgraph
        assert sub is not None
        assert sub.num_edges == result.num_pruned_candidates

        # Serialization check
        d = result.to_dict()
        assert d["num_pruned_candidates"] == result.num_pruned_candidates
        assert "execution_time_sec" in d


class TestCandidateRecallAndOptimalityGap:
    """Tests evaluating Candidate Recall and Optimality Gap (RQ3 metrics)."""

    def test_candidate_recall_computation(self) -> None:
        pruned_keys = {
            ("role:a", "policy:p1", "AttachedWith"),
            ("role:b", "role:admin", "PassesTo"),
            ("user:c", "role:worker", "AssumesRole"),
        }

        # 1. Perfect recall (100% of optimal cuts retained)
        optimal_1 = [("role:a", "policy:p1", "AttachedWith"), ("role:b", "role:admin", "PassesTo")]
        assert compute_candidate_recall(pruned_keys, optimal_1) == 1.0

        # 2. Partial recall (1 of 2 retained = 50%)
        optimal_2 = [
            ("role:a", "policy:p1", "AttachedWith"),
            ("role:x", "role:y", "PassesTo"),
        ]
        assert math.isclose(compute_candidate_recall(pruned_keys, optimal_2), 0.5)

        # 3. Zero recall
        optimal_3 = [("nonexistent:1", "nonexistent:2", "ActsOn")]
        assert compute_candidate_recall(pruned_keys, optimal_3) == 0.0

        # 4. Empty optimal cuts (trivially 1.0)
        assert compute_candidate_recall(pruned_keys, []) == 1.0

    def test_optimality_gap_computation(self) -> None:
        # Zero gap when costs match
        abs_gap, rel_gap = compute_optimality_gap(unpruned_cost=10.0, pruned_cost=10.0)
        assert abs_gap == 0.0
        assert rel_gap == 0.0

        # Pruned cost slightly higher (e.g. 12.0 vs 10.0)
        abs_gap, rel_gap = compute_optimality_gap(unpruned_cost=10.0, pruned_cost=12.0)
        assert abs_gap == 2.0
        assert math.isclose(rel_gap, 0.20)


class TestPruningScalability:
    """Tests verifying pruning execution latency remains well under interactive limits (< 3.0 s)."""

    def test_pruning_latency_on_multi_chain_graph(self) -> None:
        graph = IAMGraph(graph_id="scaling_test")

        # Construct a synthetic multi-chain enterprise graph with 100 nodes and 10 parallel paths
        for i in range(100):
            node = GraphNode(
                id=f"node_{i}",
                node_type=NodeType.ROLE if i > 0 else NodeType.USER,
                arn=f"arn:aws:iam::1:role/node_{i}",
                name=f"node_{i}",
                is_admin=(i == 99),
            )
            graph.add_node(node)

        # Build 10 parallel paths connecting node_0 to node_99
        witnesses: list[OraclePathWitness] = []
        for chain_idx in range(10):
            chain_nodes = [f"node_{0}"]
            chain_rels = []
            for hop in range(1, 5):
                nid = f"node_{chain_idx * 5 + hop}"
                chain_nodes.append(nid)
                chain_rels.append("PassesTo")
                graph.add_edge(
                    GraphEdge(
                        source=chain_nodes[-2],
                        target=nid,
                        relation=EdgeRelation.PASSES_TO,
                        actions=["iam:PassRole"],
                    )
                )
            chain_nodes.append("node_99")
            chain_rels.append("ActsOn")
            graph.add_edge(
                GraphEdge(
                    source=chain_nodes[-2],
                    target="node_99",
                    relation=EdgeRelation.ACTS_ON,
                    actions=["lambda:InvokeFunction"],
                )
            )

            witnesses.append(
                OraclePathWitness(
                    source_id="node_0",
                    source_type=NodeType.USER,
                    target_id="node_99",
                    target_type=NodeType.ROLE,
                    hop_distance=len(chain_rels),
                    path_nodes=chain_nodes,
                    path_relations=chain_rels,
                    is_privilege_escalation=True,
                )
            )

        # 50 samples with 10 witnesses per sample = 500 witnesses
        witnesses_by_sample = [witnesses for _ in range(50)]

        pruner = CandidatePruner(config=PruningConfig(max_candidates=50))
        result = pruner.prune_from_witnesses(
            graph=graph,
            witnesses_by_sample=witnesses_by_sample,
            num_samples=50,
        )

        assert result.num_pruned_candidates > 0
        assert result.total_witnesses_analyzed == 500
        # Latency must be far below 3.0 seconds (typically < 0.05 seconds)
        assert result.execution_time_sec < 1.0
