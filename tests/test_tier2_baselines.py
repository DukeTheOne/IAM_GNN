"""Unit and integration tests for Tier 2 SOTA baselines: IAMPERE SMT and TAC-GB.

Verifies:
- IAMPERE formal SMT solver reachability and Dolev-Yao trajectory extraction.
- IAMPERE 0% recall blind spot on latent bridges under Condition E.
- TAC-GB gradient-boosted decision tree fitting, probability estimation, and metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from iam.baselines.iampere import (
    IAMPEREBaselineEngine,
    IAMPEREReachabilityResult,
)
from iam.baselines.tac_gb import TACGBLinkPredictor
from iam.generator.graph import (
    EdgeRelation,
    GraphEdge,
    GraphNode,
    IAMGraph,
    NodeType,
)
from iam.generator.motifs import PassRoleLambdaMotif
from iam.generator.topology import EnterpriseTopologyConfig, EnterpriseTopologyGenerator

# ==============================================================================
# IAMPERE Formal SMT Baseline Tests
# ==============================================================================


class TestIAMPERESMTBaseline:
    """Tests for IAMPERE SMT Dolev-Yao reachability engine."""

    @pytest.fixture
    def minimal_connected_graph(self) -> IAMGraph:
        """Create a simple chain: User -> Role -> Resource."""
        g = IAMGraph("iampere_test_env")
        u = GraphNode(
            id="user_alice",
            node_type=NodeType.USER,
            arn="arn:aws:iam::123456789012:user/alice",
            name="alice",
            is_admin=False,
        )
        r = GraphNode(
            id="role_exec",
            node_type=NodeType.ROLE,
            arn="arn:aws:iam::123456789012:role/exec",
            name="exec",
            is_admin=False,
        )
        res = GraphNode(
            id="res_crown",
            node_type=NodeType.RESOURCE,
            arn="arn:aws:s3:::crown-jewel-vault",
            name="crown-jewel-vault",
            is_high_value=True,
        )
        g.add_node(u)
        g.add_node(r)
        g.add_node(res)

        g.add_edge(
            GraphEdge(source="user_alice", target="role_exec", relation=EdgeRelation.ASSUMES_ROLE)
        )
        g.add_edge(GraphEdge(source="role_exec", target="res_crown", relation=EdgeRelation.ACTS_ON))
        return g

    def test_iampere_reachable_on_connected_chain(self, minimal_connected_graph: IAMGraph) -> None:
        engine = IAMPEREBaselineEngine(max_steps=4)
        witness = engine.check_reachability(
            minimal_connected_graph,
            source_id="user_alice",
            target_id="res_crown",
        )
        assert witness.is_reachable is True
        assert witness.status == "SAT"
        assert witness.hop_distance == 2
        assert witness.path_nodes == ["user_alice", "role_exec", "res_crown"]
        assert witness.is_privilege_escalation is True
        assert witness.smt_clauses_count > 0
        assert witness.solver_time_sec >= 0.0

    def test_iampere_unreachable_on_disconnected_pair(
        self, minimal_connected_graph: IAMGraph
    ) -> None:
        # Add an isolated node
        iso = GraphNode(
            id="user_isolated",
            node_type=NodeType.USER,
            arn="arn:aws:iam::123456789012:user/isolated",
            name="isolated",
        )
        minimal_connected_graph.add_node(iso)

        engine = IAMPEREBaselineEngine(max_steps=4)
        witness = engine.check_reachability(
            minimal_connected_graph,
            source_id="user_isolated",
            target_id="res_crown",
        )
        assert witness.is_reachable is False
        assert witness.status == "UNSAT"
        assert witness.hop_distance == 0
        assert witness.path_nodes == []

    def test_iampere_condition_e_blind_spot(self) -> None:
        """Verify that under Condition E (masked bridge), IAMPERE reports UNSAT (0% recall)."""
        graph = IAMGraph("iampere_pe_env")
        motif = PassRoleLambdaMotif()
        instance = motif.inject(graph, suffix="s_cond_e")

        # Ground truth graph has the bridge: IAMPERE proves reachable
        engine = IAMPEREBaselineEngine(max_steps=5)
        gt_witness = engine.check_reachability(graph, instance.source_id, instance.target_id)
        assert gt_witness.is_reachable is True
        assert gt_witness.status == "SAT"

        # Apply Condition E: Remove bridge edge
        graph_masked = graph.clone()
        u, v, rel_str = instance.bridge_relation
        graph_masked.remove_edge(u, v, EdgeRelation(rel_str))

        # On partially observed G_o, static SMT reports UNSAT!
        masked_witness = engine.check_reachability(
            graph_masked, instance.source_id, instance.target_id
        )
        assert masked_witness.is_reachable is False
        assert masked_witness.status == "UNSAT"

        # Predict edge probabilities under Condition E: returns 0.0 for missing bridge
        scores = engine.predict_edge_probabilities(
            graph_masked, [(instance.source_id, instance.target_id)]
        )
        assert len(scores) == 1
        assert scores[0] == 0.0

    def test_iampere_evaluate_graph(self) -> None:
        config = EnterpriseTopologyConfig(num_nodes=50, seed=42, num_pe_chains=1)
        generator = EnterpriseTopologyGenerator(config)
        graph = generator.generate()

        engine = IAMPEREBaselineEngine(max_steps=4)
        result = engine.evaluate_graph(graph)

        assert isinstance(result, IAMPEREReachabilityResult)
        assert result.num_nodes == graph.num_nodes
        assert result.total_solver_time_sec > 0.0


# ==============================================================================
# TAC-GB Gradient-Boosted Baseline Tests
# ==============================================================================


class TestTACGBBaseline:
    """Tests for TAC-GB Gradient-Boosted Trees link predictor."""

    @pytest.fixture
    def small_enterprise_graph(self) -> IAMGraph:
        config = EnterpriseTopologyConfig(num_nodes=60, seed=42, num_pe_chains=2)
        return EnterpriseTopologyGenerator(config).generate()

    def test_tac_gb_initialization(self) -> None:
        predictor = TACGBLinkPredictor(learning_rate=0.1, max_iter=50)
        assert predictor.learning_rate == 0.1
        assert predictor.max_iter == 50
        assert predictor.is_fitted is False

    def test_tac_gb_fit_and_predict(self, small_enterprise_graph: IAMGraph) -> None:
        nodes = list(small_enterprise_graph.nx_graph.nodes())
        pos_pairs = [(e.source, e.target) for e in small_enterprise_graph.get_edges()[:15]]
        neg_pairs = [
            (nodes[i], nodes[(i + 5) % len(nodes)])
            for i in range(15)
            if not small_enterprise_graph.nx_graph.has_edge(nodes[i], nodes[(i + 5) % len(nodes)])
        ]

        predictor = TACGBLinkPredictor(max_iter=30, random_state=42)
        predictor.fit(small_enterprise_graph, pos_pairs=pos_pairs, neg_pairs=neg_pairs)
        assert predictor.is_fitted is True

        # Test predict_proba
        test_pairs = pos_pairs[:5] + neg_pairs[:5]
        probs = predictor.predict_proba(small_enterprise_graph, test_pairs)
        assert isinstance(probs, np.ndarray)
        assert len(probs) == len(test_pairs)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

        # Test evaluate_metrics
        y_true = np.array([1] * 5 + [0] * 5)
        metrics = predictor.evaluate_metrics(y_true, probs, k=5)
        assert "pr_auc" in metrics
        assert "roc_auc" in metrics
        assert "recall_at_k" in metrics
        assert 0.0 <= metrics["pr_auc"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_tac_gb_fit_empty_raises(self, small_enterprise_graph: IAMGraph) -> None:
        predictor = TACGBLinkPredictor()
        with pytest.raises(ValueError, match="Cannot fit"):
            predictor.fit(small_enterprise_graph, pos_pairs=[], neg_pairs=[])

    def test_tac_gb_predict_before_fit_raises(self, small_enterprise_graph: IAMGraph) -> None:
        predictor = TACGBLinkPredictor()
        with pytest.raises(RuntimeError, match="must be fitted"):
            predictor.predict_proba(small_enterprise_graph, [("u", "v")])
