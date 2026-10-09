"""Comprehensive test suite for Classical ML and Rule-Based Baselines.

Validates EdgePairFeaturizer, RandomForestLinkPredictor, and MotifCompletionHeuristic
across IAMGraph and PyG HeteroData representations.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from iam.baselines.features import FEATURE_NAMES, EdgePairFeaturizer
from iam.baselines.motif_heuristic import MotifCompletionHeuristic
from iam.baselines.random_forest import RandomForestLinkPredictor
from iam.generator.graph import IAMGraph
from iam.generator.motifs import CreateAccessKeyMotif, PassRoleLambdaMotif
from iam.generator.topology import EnterpriseTopologyConfig, EnterpriseTopologyGenerator
from iam.models.converter import IAMHeteroDataConverter
from iam.models.vocabulary import ActionVocabulary
from iam.parser import CapabilityModel, load_default_capability_model


@pytest.fixture
def capability_model() -> CapabilityModel:
    """Fixture providing standard AWS capability taxonomy model."""
    return load_default_capability_model()


@pytest.fixture
def enterprise_graph(capability_model: CapabilityModel) -> IAMGraph:
    """Fixture generating a deterministic enterprise topology with injected PE chains."""
    cfg = EnterpriseTopologyConfig(
        num_nodes=150,
        seed=42,
        num_pe_chains=2,
        include_branching_chains=True,
        num_branching_chains=1,
    )
    generator = EnterpriseTopologyGenerator(config=cfg, capability_model=capability_model)
    return generator.generate()


@pytest.fixture
def enterprise_hetero(enterprise_graph: IAMGraph) -> HeteroData:
    """Fixture converting enterprise graph to PyG HeteroData."""
    vocab = ActionVocabulary(
        actions=["iam:PassRole", "lambda:CreateFunction", "sts:AssumeRole", "s3:GetObject"]
    )
    converter = IAMHeteroDataConverter(vocabulary=vocab, include_reverse_edges=True)
    return converter.convert(enterprise_graph)


class TestEdgePairFeaturizer:
    """Tests for EdgePairFeaturizer."""

    def test_empty_pairs(self, enterprise_graph: IAMGraph) -> None:
        """Verify empty candidate list returns empty array of proper dimensionality."""
        featurizer = EdgePairFeaturizer()
        feats = featurizer.featurize_pairs(enterprise_graph, [])
        assert feats.shape == (0, featurizer.num_features)
        assert feats.dtype == np.float32

        t_feats = featurizer.featurize_pairs_tensor(enterprise_graph, [])
        assert t_feats.shape == (0, featurizer.num_features)
        assert isinstance(t_feats, torch.Tensor)

    def test_feature_extraction_properties(self, enterprise_graph: IAMGraph) -> None:
        """Verify feature extraction on IAMGraph produces non-trivial finite features."""
        featurizer = EdgePairFeaturizer(max_path_cutoff=6)
        assert featurizer.feature_names == FEATURE_NAMES

        nodes = list(enterprise_graph.get_nodes())
        assert len(nodes) >= 10
        pairs = [(nodes[0].id, nodes[1].id), (nodes[1].id, nodes[2].id)]

        x_feat = featurizer.featurize_pairs(enterprise_graph, pairs)
        assert x_feat.shape == (2, featurizer.num_features)
        assert not np.isnan(x_feat).any()
        assert not np.isinf(x_feat).any()

        # Unknown node handling should not crash
        unknown_pairs = [("non_existent_u", "non_existent_v")]
        x_unknown = featurizer.featurize_pairs(enterprise_graph, unknown_pairs)
        assert x_unknown.shape == (1, featurizer.num_features)
        assert not np.isnan(x_unknown).any()

    def test_featurizer_on_hetero_data(self, enterprise_hetero: HeteroData) -> None:
        """Verify feature extraction functions seamlessly on PyG HeteroData."""
        featurizer = EdgePairFeaturizer()
        all_ids: list[str] = []
        for ntype in enterprise_hetero.node_types:
            all_ids.extend(getattr(enterprise_hetero[ntype], "node_ids", []))

        assert len(all_ids) >= 10
        pairs = [(all_ids[0], all_ids[1]), (all_ids[2], all_ids[3])]

        x_feat = featurizer.featurize_pairs(enterprise_hetero, pairs)
        assert x_feat.shape == (2, featurizer.num_features)
        assert not np.isnan(x_feat).any()


class TestRandomForestLinkPredictor:
    """Tests for RandomForestLinkPredictor."""

    def test_fit_and_predict_flow(self, enterprise_graph: IAMGraph) -> None:
        """Verify end-to-end fit and inference workflow."""
        rf = RandomForestLinkPredictor(n_estimators=20, max_depth=8, random_state=42)
        assert rf.is_fitted is False

        # Attempting predict before fit should raise RuntimeError
        with pytest.raises(RuntimeError):
            rf.predict_proba(enterprise_graph, [("u1", "u2")])

        # Extract true edges as positives, random non-edges as negatives
        existing_edges = list(enterprise_graph.nx_graph.edges())
        assert len(existing_edges) > 5
        pos_pairs = existing_edges[:10]

        nodes = [n.id for n in enterprise_graph.get_nodes()]
        neg_pairs = []
        for i in range(len(nodes) - 1):
            pair = (nodes[i], nodes[len(nodes) - 1 - i])
            if not enterprise_graph.nx_graph.has_edge(pair[0], pair[1]):
                neg_pairs.append(pair)
            if len(neg_pairs) >= 10:
                break

        # Train model
        fitted_rf = rf.fit(enterprise_graph, pos_pairs, neg_pairs)
        assert fitted_rf.is_fitted is True

        # Predict probabilities
        test_pairs = pos_pairs[:3] + neg_pairs[:3]
        probs = rf.predict_proba(enterprise_graph, test_pairs)
        assert len(probs) == len(test_pairs)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

        preds = rf.predict(enterprise_graph, test_pairs, threshold=0.5)
        assert preds.dtype == bool
        assert len(preds) == len(test_pairs)

        # Inspect feature importances
        importances = rf.feature_importances
        assert len(importances) == rf.featurizer.num_features
        assert sum(importances.values()) == pytest.approx(1.0, abs=1e-3)

        params = rf.get_params()
        assert params["n_estimators"] == 20

    def test_fit_from_features_separable(self) -> None:
        """Verify model fits cleanly directly on synthetic feature matrices."""
        rf = RandomForestLinkPredictor(n_estimators=10, random_state=42)
        x_mat = np.array(
            [
                [10.0, 5.0, 1.0] + [0.0] * 21,
                [12.0, 6.0, 1.0] + [0.0] * 21,
                [0.0, 0.0, 0.0] + [0.0] * 21,
                [0.0, 0.0, 0.0] + [0.0] * 21,
            ],
            dtype=np.float32,
        )
        y = np.array([1, 1, 0, 0], dtype=np.int32)

        rf.fit_from_features(x_mat, y)
        assert rf.is_fitted is True

        probs = rf.predict_proba_from_features(x_mat)
        assert len(probs) == 4
        assert probs[0] > probs[2]

    def test_random_forest_edge_cases(self, enterprise_graph: IAMGraph) -> None:
        """Verify boundary and degenerate conditions for RandomForestLinkPredictor."""
        rf = RandomForestLinkPredictor(n_estimators=5, random_state=42)
        # Unfitted feature importances
        assert all(v == 0.0 for v in rf.feature_importances.values())

        # Empty fit error
        with pytest.raises(ValueError):
            rf.fit(enterprise_graph, [], [])

        with pytest.raises(ValueError):
            rf.fit_from_features(np.zeros((0, 24)), np.array([], dtype=np.int32))

        # Single class fit
        x_single = np.ones((4, 24), dtype=np.float32)
        y_single = np.ones(4, dtype=np.int32)
        rf.fit_from_features(x_single, y_single)
        assert rf.is_fitted is True

        # Empty predict_proba
        assert len(rf.predict_proba(enterprise_graph, [])) == 0
        assert len(rf.predict_proba_from_features(np.zeros((0, 24), dtype=np.float32))) == 0


class TestMotifCompletionHeuristic:
    """Tests for MotifCompletionHeuristic."""

    def test_passrole_motif_scoring(self) -> None:
        """Verify PassRole motif candidate receives high confidence score and explanation."""
        graph = IAMGraph(graph_id="motif_test_graph")
        motif = PassRoleLambdaMotif()
        instance = motif.inject(graph, suffix="test_pr")

        heuristic = MotifCompletionHeuristic()
        bridge_u, bridge_v, _ = instance.bridge_relation
        target_pair = (bridge_u, bridge_v)
        other_pair = (bridge_v, bridge_u)

        scores = heuristic.score_pairs(graph, [target_pair, other_pair])
        assert len(scores) == 2
        # Target pair matches dangling passrole pattern, should receive high score
        assert scores[0] >= 0.65
        assert scores[0] > scores[1]

        explanation = heuristic.explain_pair(graph, bridge_u, bridge_v)
        assert explanation["final_score"] >= 0.65
        assert len(explanation["matched_rules"]) > 0
        rule_names = [r["rule"] for r in explanation["matched_rules"]]
        assert "DanglingPassRoleMotif" in rule_names

    def test_credential_minting_motif_scoring(self) -> None:
        """Verify CreateAccessKey motif candidate receives high score."""
        graph = IAMGraph(graph_id="minting_test_graph")
        motif = CreateAccessKeyMotif()
        instance = motif.inject(graph, suffix="test_key")

        heuristic = MotifCompletionHeuristic()
        bridge_u, bridge_v, _ = instance.bridge_relation
        scores = heuristic.score_pairs(graph, [(bridge_u, bridge_v)])
        assert len(scores) == 1
        assert scores[0] >= 0.60

        explanation = heuristic.explain_pair(graph, bridge_u, bridge_v)
        assert any(
            r["rule"] == "DanglingCredentialMintingMotif" for r in explanation["matched_rules"]
        )

    def test_heuristic_on_hetero_data(self, enterprise_hetero: HeteroData) -> None:
        """Verify motif heuristic functions on HeteroData Go."""
        heuristic = MotifCompletionHeuristic()
        all_ids: list[str] = []
        for ntype in enterprise_hetero.node_types:
            all_ids.extend(getattr(enterprise_hetero[ntype], "node_ids", []))

        pairs = [(all_ids[0], all_ids[1]), (all_ids[1], all_ids[2])]
        scores = heuristic.score_pairs(enterprise_hetero, pairs)
        assert len(scores) == 2
        assert (scores >= 0.0).all() and (scores <= 1.0).all()


class TestBaselineScaling:
    """Performance benchmarks ensuring sub-second inference at scale."""

    def test_batch_inference_latency(self, enterprise_graph: IAMGraph) -> None:
        """Verify 200 candidate pairs are featurized and scored in under 1 second."""
        nodes = [n.id for n in enterprise_graph.get_nodes()]
        pairs = [(nodes[i % len(nodes)], nodes[(i + 7) % len(nodes)]) for i in range(200)]

        start_time = time.perf_counter()
        featurizer = EdgePairFeaturizer()
        x_scaled = featurizer.featurize_pairs(enterprise_graph, pairs)
        feat_time = time.perf_counter() - start_time

        heuristic = MotifCompletionHeuristic()
        start_time = time.perf_counter()
        scores = heuristic.score_pairs(enterprise_graph, pairs)
        heuristic_time = time.perf_counter() - start_time

        assert x_scaled.shape == (200, featurizer.num_features)
        assert len(scores) == 200
        assert feat_time < 1.0, f"Featurization took {feat_time:.3f}s (budget 1.0s)"
        assert heuristic_time < 0.5, f"Heuristic took {heuristic_time:.3f}s (budget 0.5s)"
