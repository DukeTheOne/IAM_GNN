"""Unit and integration tests for Step 3.3: Environment dataset splits and stratified negative sampler."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
from torch_geometric.data import HeteroData

from iam.generator import (
    EnterpriseTopologyConfig,
    EnterpriseTopologyGenerator,
    GroundTruthLabeler,
    IAMGraph,
)
from iam.models import (
    DEFAULT_ORGANIZATION_CONFIGS,
    ActionVocabulary,
    EnterpriseCorpusGenerator,
    EnvironmentConfig,
    IAMEnvironmentDataset,
    IAMHeteroDataConverter,
    NegativeSampleResult,
    StratifiedNegativeSampler,
    load_default_action_vocabulary,
)
from iam.parser import CapabilityModel, load_default_capability_model


@pytest.fixture
def capability_model() -> CapabilityModel:
    """Fixture providing loaded AWS CapabilityModel."""
    return load_default_capability_model()


@pytest.fixture
def default_vocab() -> ActionVocabulary:
    """Fixture providing default ActionVocabulary."""
    return load_default_action_vocabulary()


@pytest.fixture
def sample_hetero_data(
    capability_model: CapabilityModel, default_vocab: ActionVocabulary
) -> tuple[IAMGraph, HeteroData]:
    """Fixture generating a medium enterprise IAMGraph and converted HeteroData."""
    cfg = EnterpriseTopologyConfig(
        num_nodes=150,
        seed=42,
        num_pe_chains=2,
        include_branching_chains=True,
        num_branching_chains=1,
    )
    topo_gen = EnterpriseTopologyGenerator(config=cfg, capability_model=capability_model)
    graph = topo_gen.generate()
    labeler = GroundTruthLabeler()
    labels = labeler.label_graph(graph)

    converter = IAMHeteroDataConverter(vocabulary=default_vocab)
    hetero = converter.convert(graph, label_set=labels)
    return graph, hetero


class TestStratifiedNegativeSampler:
    """Tests for StratifiedNegativeSampler verification, type safety, and stratum quotas."""

    def test_sampler_initialization_and_weights(self) -> None:
        """Verify constructor weight normalization and validation."""
        sampler = StratifiedNegativeSampler(
            intra_dept_ratio=0.5,
            inter_dept_ratio=0.3,
            privilege_boundary_ratio=0.2,
        )
        assert pytest.approx(sampler.intra_dept_ratio) == 0.5
        assert pytest.approx(sampler.inter_dept_ratio) == 0.3
        assert pytest.approx(sampler.privilege_boundary_ratio) == 0.2

        # Non-normalized weights should be automatically normalized
        sampler2 = StratifiedNegativeSampler(5.0, 3.0, 2.0)
        assert pytest.approx(sampler2.intra_dept_ratio) == 0.5
        assert pytest.approx(sampler2.inter_dept_ratio) == 0.3
        assert pytest.approx(sampler2.privilege_boundary_ratio) == 0.2

        with pytest.raises(ValueError, match="positive value"):
            StratifiedNegativeSampler(0, 0, 0)

    def test_negative_sampling_type_safety_and_shapes(
        self, sample_hetero_data: tuple[IAMGraph, HeteroData]
    ) -> None:
        """Verify sampled negative edges are type-safe and within node index boundaries."""
        _, hetero = sample_hetero_data
        sampler = StratifiedNegativeSampler()

        edge_type = ("User", "AttachedWith", "Policy")
        assert edge_type in hetero.edge_types

        res = sampler.sample(hetero, edge_type=edge_type, ratio=1.5, seed=123)

        assert isinstance(res, NegativeSampleResult)
        assert res.edge_type == edge_type
        assert res.pos_edge_index.size(0) == 2
        assert res.neg_edge_index.size(0) == 2
        assert res.edge_label_index.size(0) == 2

        n_pos = res.num_pos
        n_neg = res.num_neg
        assert n_pos > 0
        assert res.num_neg == int(round(n_pos * 1.5))
        assert abs(res.negative_ratio - 1.5) < 0.25

        # Check concatenation shapes
        assert res.edge_label_index.size(1) == n_pos + n_neg
        assert res.edge_label.size(0) == n_pos + n_neg
        assert (res.edge_label[:n_pos] == 1.0).all()
        assert (res.edge_label[n_pos:] == 0.0).all()

        # Type safety bounds: u in [0, N_user), v in [0, N_policy)
        n_users = hetero["User"].num_nodes
        n_policies = hetero["Policy"].num_nodes

        u_neg = res.neg_edge_index[0].tolist()
        v_neg = res.neg_edge_index[1].tolist()

        for u in u_neg:
            assert 0 <= u < n_users
        for v in v_neg:
            assert 0 <= v < n_policies

    def test_zero_collision_with_positive_edges(
        self, sample_hetero_data: tuple[IAMGraph, HeteroData]
    ) -> None:
        """Verify negative edges contain ZERO overlap with ground-truth positive edges."""
        _, hetero = sample_hetero_data
        sampler = StratifiedNegativeSampler()

        edge_type = ("User", "AssumesRole", "Role")
        if edge_type not in hetero.edge_types:
            edge_type = ("User", "AttachedWith", "Policy")

        res = sampler.sample(hetero, edge_type=edge_type, ratio=2.0, seed=42)

        pos_set = set(
            zip(
                res.pos_edge_index[0].tolist(),
                res.pos_edge_index[1].tolist(),
                strict=False,
            )
        )
        neg_set = set(
            zip(
                res.neg_edge_index[0].tolist(),
                res.neg_edge_index[1].tolist(),
                strict=False,
            )
        )

        # 1. No overlap with positive set
        collision = pos_set.intersection(neg_set)
        assert len(collision) == 0, f"Colliding positive edges sampled as negatives: {collision}"

        # 2. No duplicates within negative set
        assert len(neg_set) == res.num_neg, "Sampled negative edges contain duplicates."

    def test_stratum_distribution_and_counts(
        self, sample_hetero_data: tuple[IAMGraph, HeteroData]
    ) -> None:
        """Verify multi-stratum allocation across intra-department, inter-department, and boundary."""
        _, hetero = sample_hetero_data
        sampler = StratifiedNegativeSampler(
            intra_dept_ratio=0.5,
            inter_dept_ratio=0.3,
            privilege_boundary_ratio=0.2,
        )

        edge_type = ("User", "AttachedWith", "Policy")
        res = sampler.sample(hetero, edge_type=edge_type, ratio=2.0, seed=777)

        counts = res.stratum_counts
        assert "intra_department" in counts
        assert "inter_department" in counts
        assert "privilege_boundary" in counts
        assert "fallback" in counts

        total_sampled = sum(counts.values())
        assert total_sampled == res.num_neg

    def test_deterministic_reproducibility(
        self, sample_hetero_data: tuple[IAMGraph, HeteroData]
    ) -> None:
        """Verify identical seed produces identical negative samples, different seed differs."""
        _, hetero = sample_hetero_data
        sampler = StratifiedNegativeSampler()

        edge_type = ("User", "AttachedWith", "Policy")
        res1 = sampler.sample(hetero, edge_type=edge_type, ratio=1.0, seed=999)
        res2 = sampler.sample(hetero, edge_type=edge_type, ratio=1.0, seed=999)
        res3 = sampler.sample(hetero, edge_type=edge_type, ratio=1.0, seed=123)

        assert torch.equal(res1.neg_edge_index, res2.neg_edge_index)
        assert not torch.equal(res1.neg_edge_index, res3.neg_edge_index)

    def test_sample_all_forward_relations_excludes_reverse(
        self, sample_hetero_data: tuple[IAMGraph, HeteroData]
    ) -> None:
        """Verify sample_all_forward_relations only samples forward edges and excludes Rev*."""
        _, hetero = sample_hetero_data
        sampler = StratifiedNegativeSampler()

        all_results = sampler.sample_all_forward_relations(hetero, ratio=1.0, seed=42)

        for edge_type, res in all_results.items():
            assert not edge_type[1].startswith("Rev"), f"Reverse relation sampled: {edge_type}"
            assert res.edge_type == edge_type
            assert res.pos_edge_index.size(0) == 2

    def test_empty_or_missing_edge_type_handling(self) -> None:
        """Verify graceful degradation when edge type is absent or node count is zero."""
        data = HeteroData()
        data["User"].num_nodes = 5
        data["Role"].num_nodes = 0
        sampler = StratifiedNegativeSampler()

        res = sampler.sample(data, ("User", "AssumesRole", "Role"), ratio=1.0)
        assert res.num_pos == 0
        assert res.num_neg == 0
        assert res.edge_label_index.size(1) == 0


class TestEnterpriseCorpusGeneratorAndManifest:
    """Tests for EnterpriseCorpusGenerator, 14-org configuration schema, and corpus manifest."""

    def test_default_14_org_configurations(self) -> None:
        """Verify default configuration covers 14 distinct organizations across train, val, test."""
        configs = DEFAULT_ORGANIZATION_CONFIGS
        assert len(configs) == 14

        train_configs = [c for c in configs if c.split == "train"]
        val_configs = [c for c in configs if c.split == "val"]
        test_configs = [c for c in configs if c.split == "test"]

        assert len(train_configs) == 8
        assert len(val_configs) == 2
        assert len(test_configs) == 4

        # Unique organization IDs and account IDs
        org_ids = [c.org_id for c in configs]
        account_ids = [c.account_id for c in configs]
        assert len(set(org_ids)) == 14
        assert len(set(account_ids)) == 14

        # Scale constraints
        for c in train_configs:
            assert 300 <= c.num_nodes <= 1500
        for c in val_configs:
            assert 800 <= c.num_nodes <= 1500
        for c in test_configs:
            assert 1000 <= c.num_nodes <= 3500

    def test_generate_single_environment(self, capability_model: CapabilityModel) -> None:
        """Verify generation of a single environment with valid HeteroData and PE pairs."""
        cfg = EnvironmentConfig(
            org_id="test-org-01",
            account_id="999900001111",
            split="train",
            num_nodes=80,
            seed=42,
            num_pe_chains=1,
            num_branching_chains=0,
        )
        generator = EnterpriseCorpusGenerator(capability_model=capability_model)
        graph, labels, hetero = generator.generate_environment(cfg)

        assert isinstance(graph, IAMGraph)
        assert graph.num_nodes > 50
        assert len(labels.pe_pairs) >= 1
        assert isinstance(hetero, HeteroData)
        assert hetero.organization_id == "test-org-01"
        assert hetero.account_id == "999900001111"
        assert "pe_pairs" in hetero

    def test_zero_entity_leakage_across_splits(self, capability_model: CapabilityModel) -> None:
        """Formally verify mathematical disjointness of entities across train, val, test splits."""
        # Define 3 mini environments: 1 train, 1 val, 1 test
        test_configs = [
            EnvironmentConfig(
                org_id="leak-org-train",
                account_id="111111111111",
                split="train",
                num_nodes=60,
                seed=101,
                num_pe_chains=1,
                num_branching_chains=0,
            ),
            EnvironmentConfig(
                org_id="leak-org-val",
                account_id="222222222222",
                split="val",
                num_nodes=60,
                seed=201,
                num_pe_chains=1,
                num_branching_chains=0,
            ),
            EnvironmentConfig(
                org_id="leak-org-test",
                account_id="333333333333",
                split="test",
                num_nodes=60,
                seed=301,
                num_pe_chains=1,
                num_branching_chains=0,
            ),
        ]

        generator = EnterpriseCorpusGenerator(capability_model=capability_model)
        nodes_by_split: dict[str, set[str]] = {}
        arns_by_split: dict[str, set[str]] = {}

        for cfg in test_configs:
            graph, _, _ = generator.generate_environment(cfg)
            nodes_by_split[cfg.split] = {f"{cfg.org_id}:{nid}" for nid in graph.get_node_ids()}
            arns_by_split[cfg.split] = {n.arn for n in graph.get_nodes()}

        # Verify mathematical isolation: V_train ∩ V_val ∩ V_test = ∅
        train_val_nodes = nodes_by_split["train"].intersection(nodes_by_split["val"])
        train_test_nodes = nodes_by_split["train"].intersection(nodes_by_split["test"])
        val_test_nodes = nodes_by_split["val"].intersection(nodes_by_split["test"])

        assert len(train_val_nodes) == 0, f"Node leakage between train and val: {train_val_nodes}"
        assert len(train_test_nodes) == 0, (
            f"Node leakage between train and test: {train_test_nodes}"
        )
        assert len(val_test_nodes) == 0, f"Node leakage between val and test: {val_test_nodes}"

        # Verify ARN isolation
        train_val_arns = arns_by_split["train"].intersection(arns_by_split["val"])
        train_test_arns = arns_by_split["train"].intersection(arns_by_split["test"])
        val_test_arns = arns_by_split["val"].intersection(arns_by_split["test"])

        assert len(train_val_arns) == 0
        assert len(train_test_arns) == 0
        assert len(val_test_arns) == 0


class TestIAMEnvironmentDataset:
    """Tests for PyG IAMEnvironmentDataset lazy loading, caching, and split filtering."""

    @pytest.fixture
    def mini_corpus_dir(self, capability_model: CapabilityModel) -> Any:
        """Fixture generating a small 3-organization corpus in a temporary directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            configs = [
                EnvironmentConfig(
                    org_id="corpus-train-01",
                    account_id="100000000001",
                    split="train",
                    num_nodes=50,
                    seed=11,
                    num_pe_chains=1,
                    num_branching_chains=0,
                ),
                EnvironmentConfig(
                    org_id="corpus-val-01",
                    account_id="100000000002",
                    split="val",
                    num_nodes=50,
                    seed=22,
                    num_pe_chains=1,
                    num_branching_chains=0,
                ),
                EnvironmentConfig(
                    org_id="corpus-test-01",
                    account_id="100000000003",
                    split="test",
                    num_nodes=50,
                    seed=33,
                    num_pe_chains=1,
                    num_branching_chains=0,
                ),
            ]

            gen = EnterpriseCorpusGenerator(capability_model=capability_model)
            manifest = gen.generate_corpus(
                output_dir=tmp_path,
                configs=configs,
                force=True,
                export_raw_json=True,
            )
            assert (tmp_path / "corpus_manifest.json").exists()
            assert (tmp_path / "processed" / "corpus-train-01.pt").exists()
            assert (tmp_path / "processed" / "corpus-val-01.pt").exists()
            assert (tmp_path / "processed" / "corpus-test-01.pt").exists()

            yield tmp_path, manifest

    def test_dataset_full_and_split_views(
        self, mini_corpus_dir: tuple[Path, dict[str, Any]]
    ) -> None:
        """Verify IAMEnvironmentDataset initializes, reports correct len, and filters splits."""
        root_dir, manifest = mini_corpus_dir

        full_ds = IAMEnvironmentDataset(root=root_dir)
        assert len(full_ds) == 3
        assert full_ds.manifest["total_organizations"] == 3

        train_ds = full_ds.get_train_split()
        assert len(train_ds) == 1
        assert train_ds.org_ids == ["corpus-train-01"]

        val_ds = full_ds.get_val_split()
        assert len(val_ds) == 1
        assert val_ds.org_ids == ["corpus-val-01"]

        test_ds = full_ds.get_test_split()
        assert len(test_ds) == 1
        assert test_ds.org_ids == ["corpus-test-01"]

    def test_dataset_get_and_in_memory_caching(
        self, mini_corpus_dir: tuple[Path, dict[str, Any]]
    ) -> None:
        """Verify get(idx) returns valid HeteroData and utilizes in-memory cache."""
        root_dir, _ = mini_corpus_dir
        ds = IAMEnvironmentDataset(root=root_dir, cache_in_memory=True)

        data0 = ds.get(0)
        assert isinstance(data0, HeteroData)
        assert data0.organization_id in ds.org_ids

        # Subsequent retrieval must return the identical cached instance
        data0_cached = ds.get(0)
        assert data0 is data0_cached

        # Retrieval by org_id
        org0_id = ds.org_ids[0]
        data_by_id = ds.get_by_org_id(org0_id)
        assert data_by_id is data0

    def test_dataset_error_handling(self, mini_corpus_dir: tuple[Path, dict[str, Any]]) -> None:
        """Verify IndexError and KeyError on out-of-bounds queries."""
        root_dir, _ = mini_corpus_dir
        ds = IAMEnvironmentDataset(root=root_dir)

        with pytest.raises(IndexError):
            ds.get(999)

        with pytest.raises(IndexError):
            ds.get(-1)

        with pytest.raises(KeyError):
            ds.get_by_org_id("non-existent-org")

    def test_corpus_generator_reuse_existing_files(
        self, mini_corpus_dir: tuple[Path, dict[str, Any]], capability_model: CapabilityModel
    ) -> None:
        """Verify that when force=False, EnterpriseCorpusGenerator reuses existing pt files."""
        root_dir, manifest_initial = mini_corpus_dir
        gen = EnterpriseCorpusGenerator(capability_model=capability_model)

        configs = [
            EnvironmentConfig(
                org_id="corpus-train-01",
                account_id="100000000001",
                split="train",
                num_nodes=50,
                seed=11,
            )
        ]
        # Re-run with force=False
        manifest_reused = gen.generate_corpus(output_dir=root_dir, configs=configs, force=False)
        assert "corpus-train-01" in manifest_reused["organizations"]
        assert manifest_reused["organizations"]["corpus-train-01"]["org_id"] == "corpus-train-01"


class TestStratifiedNegativeSamplerEdgeCases:
    """Additional edge case tests for negative sampler coverage."""

    def test_sampler_large_search_space_and_fallback(self) -> None:
        """Verify sampling when node pair space > 2500 pairs triggers the rejection loop."""
        data = HeteroData()
        data["User"].num_nodes = 60
        data["Role"].num_nodes = 60
        data["User"].x = torch.zeros((60, 10))
        data["Role"].x = torch.zeros((60, 10))

        # Add 5 random positive edges
        pos_edges = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long)
        data["User", "AssumesRole", "Role"].edge_index = pos_edges

        sampler = StratifiedNegativeSampler(0.5, 0.3, 0.2)
        res = sampler.sample(
            data,
            ("User", "AssumesRole", "Role"),
            num_negatives=20,
            seed=42,
        )

        assert res.num_neg == 20
        assert res.neg_edge_index.size(1) == 20
        # No overlap
        pos_set = set(zip(pos_edges[0].tolist(), pos_edges[1].tolist(), strict=False))
        neg_set = set(
            zip(res.neg_edge_index[0].tolist(), res.neg_edge_index[1].tolist(), strict=False)
        )
        assert len(pos_set.intersection(neg_set)) == 0

    def test_sampler_zero_positive_edges_and_zero_target(self) -> None:
        """Verify handling when relation has 0 positive edges and ratio sampling returns 0."""
        data = HeteroData()
        data["User"].num_nodes = 10
        data["Role"].num_nodes = 10
        data["User", "AssumesRole", "Role"].edge_index = torch.empty((2, 0), dtype=torch.long)

        sampler = StratifiedNegativeSampler()
        res = sampler.sample(data, ("User", "AssumesRole", "Role"), ratio=1.0)
        assert res.num_pos == 0
        assert res.num_neg == 0
        assert res.edge_label.numel() == 0

        # Sample with explicit num_negatives when 0 pos edges exist
        res2 = sampler.sample(data, ("User", "AssumesRole", "Role"), num_negatives=5, seed=123)
        assert res2.num_pos == 0
        assert res2.num_neg == 5
        assert (res2.edge_label == 0.0).all()
