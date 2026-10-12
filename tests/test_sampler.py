"""Comprehensive unit and integration tests for LatentCompletionSampler and Monte Carlo reachability.

Tests Phase 3 Week 7 (Step 7.1):
- CandidateEdge creation, validation, and serialization.
- LatentCompletion instantiation and union graph consistency.
- LatentCompletionSampler Bernoulli sampling, seed reproducibility, and empirical convergence.
- Monte Carlo reachability evaluation against ExactReachabilityOracle on P0 motifs.
- compare_monte_carlo_vs_product analytical benchmarking on single chains and branching lattices.
"""

from __future__ import annotations

import math

import pytest

from iam.baselines.oracle import ExactReachabilityOracle
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
    LatentCompletion,
    LatentCompletionSampler,
    MonteCarloReachabilityResult,
    ProductComparisonResult,
    compare_monte_carlo_vs_product,
)


class TestCandidateEdge:
    """Tests for CandidateEdge data structures and constructors."""

    def test_candidate_edge_creation_and_key(self) -> None:
        edge = CandidateEdge(
            source="user:alice",
            target="role:admin",
            relation=EdgeRelation.ASSUMES_ROLE,
            probability=0.85,
            is_bridge=True,
            motif_id="motif_01",
            actions=("sts:AssumeRole",),
        )
        assert edge.source == "user:alice"
        assert edge.target == "role:admin"
        assert edge.relation == EdgeRelation.ASSUMES_ROLE
        assert edge.probability == 0.85
        assert edge.is_bridge is True
        assert edge.key == ("user:alice", "role:admin", "AssumesRole")

    def test_probability_bounds_validation(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            CandidateEdge(
                source="u",
                target="r",
                relation=EdgeRelation.ACTS_ON,
                probability=-0.1,
            )

        with pytest.raises(ValueError, match="must be in"):
            CandidateEdge(
                source="u",
                target="r",
                relation=EdgeRelation.ACTS_ON,
                probability=1.05,
            )

    def test_conversion_to_graph_edge(self) -> None:
        cand = CandidateEdge(
            source="role:dev",
            target="resource:bucket",
            relation=EdgeRelation.ACTS_ON,
            probability=0.7,
            actions=("s3:GetObject",),
            metadata={"source": "rgcn_prediction"},
        )
        ge = cand.to_graph_edge()
        assert isinstance(ge, GraphEdge)
        assert ge.source == "role:dev"
        assert ge.target == "resource:bucket"
        assert ge.relation == EdgeRelation.ACTS_ON
        assert ge.actions == ["s3:GetObject"]
        assert ge.metadata["source"] == "rgcn_prediction"

    def test_constructors_from_graph_edge_and_tuple(self) -> None:
        ge = GraphEdge(
            source="u1",
            target="r1",
            relation=EdgeRelation.PASSES_TO,
            is_bridge=True,
            actions=["iam:PassRole"],
        )
        c1 = CandidateEdge.from_graph_edge(ge, probability=0.92)
        assert c1.source == "u1"
        assert c1.probability == 0.92
        assert c1.is_bridge is True

        c2 = CandidateEdge.from_tuple(
            source="u2",
            target="r2",
            relation="AssumesRole",
            probability=0.45,
            is_bridge=False,
        )
        assert c2.relation == EdgeRelation.ASSUMES_ROLE
        assert c2.probability == 0.45


class TestLatentCompletion:
    """Tests for LatentCompletion instance properties."""

    def test_latent_completion_properties(self) -> None:
        graph = IAMGraph(graph_id="g_test")
        c1 = CandidateEdge(
            source="u", target="r1", relation=EdgeRelation.ASSUMES_ROLE, probability=0.9
        )
        c2 = CandidateEdge(
            source="r1",
            target="r2",
            relation=EdgeRelation.PASSES_TO,
            probability=0.5,
            is_bridge=True,
        )

        completion = LatentCompletion(
            sample_id=42,
            sampled_edges=[c1, c2],
            graph=graph,
            metadata={"seed": 123},
        )
        assert completion.sample_id == 42
        assert completion.num_sampled_edges == 2
        assert ("u", "r1", "AssumesRole") in completion.active_edge_keys
        assert ("r1", "r2", "PassesTo") in completion.active_edge_keys
        assert completion.has_bridge_edge is True
        assert len(completion.bridge_edges) == 1
        assert completion.bridge_edges[0] == c2


class TestLatentCompletionSampler:
    """Tests for LatentCompletionSampler sampling logic and statistical properties."""

    @pytest.fixture
    def minimal_graph(self) -> IAMGraph:
        graph = IAMGraph(graph_id="minimal_test")
        u = GraphNode(
            id="user:alice",
            node_type=NodeType.USER,
            arn="arn:aws:iam::1:user/alice",
            name="alice",
        )
        r = GraphNode(
            id="role:lambda",
            node_type=NodeType.ROLE,
            arn="arn:aws:iam::1:role/lambda",
            name="lambda",
        )
        graph.add_node(u)
        graph.add_node(r)
        return graph

    def test_empty_candidates_sampling(self, minimal_graph: IAMGraph) -> None:
        sampler = LatentCompletionSampler(observed_graph=minimal_graph)
        completions = sampler.sample_completions(num_samples=10, seed=101)
        assert len(completions) == 10
        for comp in completions:
            assert comp.num_sampled_edges == 0
            assert comp.graph.num_edges == 0

    def test_deterministic_seed_reproducibility(self, minimal_graph: IAMGraph) -> None:
        c1 = CandidateEdge(
            source="user:alice",
            target="role:lambda",
            relation=EdgeRelation.PASSES_TO,
            probability=0.6,
        )
        sampler = LatentCompletionSampler(observed_graph=minimal_graph, candidate_edges=[c1])

        run_a = sampler.sample_completions(num_samples=50, seed=999)
        run_b = sampler.sample_completions(num_samples=50, seed=999)

        for a, b in zip(run_a, run_b, strict=True):
            assert a.num_sampled_edges == b.num_sampled_edges
            assert a.active_edge_keys == b.active_edge_keys

    def test_extreme_probabilities(self, minimal_graph: IAMGraph) -> None:
        c_zero = CandidateEdge(
            source="user:alice",
            target="role:lambda",
            relation=EdgeRelation.PASSES_TO,
            probability=0.0,
        )
        c_one = CandidateEdge(
            source="user:alice",
            target="role:lambda",
            relation=EdgeRelation.ASSUMES_ROLE,
            probability=1.0,
        )
        sampler = LatentCompletionSampler(
            observed_graph=minimal_graph,
            candidate_edges=[c_zero, c_one],
        )
        completions = sampler.sample_completions(num_samples=25, seed=42)
        for comp in completions:
            assert comp.num_sampled_edges == 1
            assert ("user:alice", "role:lambda", "AssumesRole") in comp.active_edge_keys
            assert ("user:alice", "role:lambda", "PassesTo") not in comp.active_edge_keys

    def test_bernoulli_empirical_convergence(self, minimal_graph: IAMGraph) -> None:
        target_p = 0.35
        c = CandidateEdge(
            source="user:alice",
            target="role:lambda",
            relation=EdgeRelation.ASSUMES_ROLE,
            probability=target_p,
        )
        sampler = LatentCompletionSampler(observed_graph=minimal_graph, candidate_edges=[c])
        num_samples = 2000
        completions = sampler.sample_completions(num_samples=num_samples, seed=12345)

        active_count = sum(1 for comp in completions if comp.num_sampled_edges > 0)
        empirical_p = active_count / num_samples

        # 3 standard deviations = 3 * sqrt(0.35 * 0.65 / 2000) ≈ 0.032
        assert math.isclose(empirical_p, target_p, abs_tol=0.035)

    def test_invalid_candidate_endpoints_raise_keyerror(self, minimal_graph: IAMGraph) -> None:
        bad_cand = CandidateEdge(
            source="user:alice",
            target="nonexistent_role",
            relation=EdgeRelation.PASSES_TO,
            probability=1.0,
        )
        sampler = LatentCompletionSampler(observed_graph=minimal_graph, candidate_edges=[bad_cand])
        with pytest.raises(KeyError, match="not found in observed graph"):
            sampler.sample_completions(num_samples=1)

    def test_invalid_num_samples_raises_valueerror(self, minimal_graph: IAMGraph) -> None:
        sampler = LatentCompletionSampler(observed_graph=minimal_graph)
        with pytest.raises(ValueError, match="num_samples must be positive"):
            sampler.sample_completions(num_samples=0)


class TestMonteCarloReachabilityEvaluation:
    """Tests evaluating reachability across completions on canonical motifs."""

    @pytest.fixture
    def passrole_lambda_setup(self) -> tuple[IAMGraph, CandidateEdge, str, str]:
        """Setup canonical PassRole + Lambda motif with bridge masked (Condition E)."""
        graph = IAMGraph(graph_id="pe_env")
        motif = PassRoleLambdaMotif()
        instance = motif.inject(graph, suffix="test_mc")

        src, dst, rel = instance.bridge_relation
        # Sever bridge to simulate Condition E observed graph
        graph.remove_edge(src, dst, EdgeRelation(rel))

        cand = CandidateEdge(
            source=src,
            target=dst,
            relation=EdgeRelation(rel),
            probability=0.80,
            is_bridge=True,
        )
        return graph, cand, instance.source_id, instance.target_id

    def test_monte_carlo_reachability_on_masked_bridge(
        self,
        passrole_lambda_setup: tuple[IAMGraph, CandidateEdge, str, str],
    ) -> None:
        graph, cand_bridge, source_id, target_id = passrole_lambda_setup

        oracle = ExactReachabilityOracle()
        # In observed graph Go, target is disconnected
        base_res = oracle.evaluate(graph, source_ids=[source_id], target_ids=[target_id])
        assert not base_res.is_reachable(source_id, target_id)

        # Run Monte Carlo sampling with p = 0.80
        sampler = LatentCompletionSampler(observed_graph=graph, candidate_edges=[cand_bridge])
        completions, mc_res = sampler.sample_and_evaluate(
            num_samples=500,
            seed=777,
            source_ids=[source_id],
            target_ids=[target_id],
            oracle=oracle,
        )

        assert isinstance(mc_res, MonteCarloReachabilityResult)
        reach_prob = mc_res.get_reachability_probability(source_id, target_id)
        pe_prob = mc_res.get_pe_probability(source_id, target_id)

        # Empirical probability should be close to candidate probability 0.80
        assert math.isclose(reach_prob, 0.80, abs_tol=0.05)
        assert math.isclose(pe_prob, 0.80, abs_tol=0.05)
        assert mc_res.is_attack_possible(source_id, target_id)

        witnesses = mc_res.get_all_witnesses_for_pair(source_id, target_id)
        assert len(witnesses) > 0
        assert witnesses[0].is_privilege_escalation is True

        summary = mc_res.summary_dict()
        assert summary["num_samples"] == 500
        assert summary["num_pairs_evaluated"] == 1
        assert "execution_time_sec" in summary


class TestProductComparison:
    """Tests benchmarking Monte Carlo reachability against naive edge-product estimates."""

    def test_single_chain_independence_consistency(self) -> None:
        """On a single multi-hop chain, Monte Carlo reachability matches edge product within 95% CI."""
        graph = IAMGraph()
        n1 = GraphNode(id="n1", node_type=NodeType.USER, arn="arn:aws:iam::1:user/n1", name="n1")
        n2 = GraphNode(id="n2", node_type=NodeType.ROLE, arn="arn:aws:iam::1:role/n2", name="n2")
        n3 = GraphNode(id="n3", node_type=NodeType.ROLE, arn="arn:aws:iam::1:role/n3", name="n3")
        graph.add_nodes([n1, n2, n3])

        # Candidate edges: n1 -> n2 (p=0.8), n2 -> n3 (p=0.7)
        c1 = CandidateEdge(
            source="n1", target="n2", relation=EdgeRelation.ASSUMES_ROLE, probability=0.8
        )
        c2 = CandidateEdge(
            source="n2", target="n3", relation=EdgeRelation.ASSUMES_ROLE, probability=0.7
        )

        res = compare_monte_carlo_vs_product(
            graph=graph,
            candidate_edges=[c1, c2],
            source_id="n1",
            target_id="n3",
            num_samples=2000,
            seed=42,
        )

        assert isinstance(res, ProductComparisonResult)
        expected_product = 0.8 * 0.7  # 0.56
        assert math.isclose(res.product_probability, expected_product, abs_tol=1e-5)
        assert math.isclose(res.monte_carlo_probability, expected_product, abs_tol=0.035)
        assert res.is_consistent_with_independence is True
        assert res.ci_lower <= expected_product <= res.ci_upper

        d = res.to_dict()
        assert d["num_samples"] == 2000
        assert d["path_length"] == 2

    def test_diamond_lattice_multi_path_divergence(self) -> None:
        """In a diamond branching topology, Monte Carlo reachability exceeds naive product of all edges."""
        graph = IAMGraph()
        s = GraphNode(id="s", node_type=NodeType.USER, arn="arn:aws:iam::1:user/s", name="s")
        a = GraphNode(id="a", node_type=NodeType.ROLE, arn="arn:aws:iam::1:role/a", name="a")
        b = GraphNode(id="b", node_type=NodeType.ROLE, arn="arn:aws:iam::1:role/b", name="b")
        t = GraphNode(
            id="t", node_type=NodeType.ROLE, arn="arn:aws:iam::1:role/t", name="t", is_admin=True
        )
        graph.add_nodes([s, a, b, t])

        # Branch 1: s -> a (p=0.6), a -> t (p=0.6)  => P(B1) = 0.36
        # Branch 2: s -> b (p=0.6), b -> t (p=0.6)  => P(B2) = 0.36
        # P(Reach) = P(B1 ∪ B2) = 0.36 + 0.36 - 0.36*0.36 = 0.5904
        # Naive product across all 4 candidate edges = 0.6^4 = 0.1296
        c1 = CandidateEdge(
            source="s", target="a", relation=EdgeRelation.ASSUMES_ROLE, probability=0.6
        )
        c2 = CandidateEdge(
            source="a", target="t", relation=EdgeRelation.ASSUMES_ROLE, probability=0.6
        )
        c3 = CandidateEdge(
            source="s", target="b", relation=EdgeRelation.ASSUMES_ROLE, probability=0.6
        )
        c4 = CandidateEdge(
            source="b", target="t", relation=EdgeRelation.ASSUMES_ROLE, probability=0.6
        )

        res = compare_monte_carlo_vs_product(
            graph=graph,
            candidate_edges=[c1, c2, c3, c4],
            source_id="s",
            target_id="t",
            num_samples=2000,
            seed=42,
        )

        expected_mc = 0.5904
        assert math.isclose(res.monte_carlo_probability, expected_mc, abs_tol=0.035)
        # Monte Carlo reachability should strongly exceed naive 4-edge product
        assert res.monte_carlo_probability > res.product_probability
        assert "Multi-path / redundant escalation topology" in res.diagnostics
