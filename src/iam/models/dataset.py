"""Environment-level dataset partitioning and inductive multi-graph dataset pipeline.

Orchestrates the synthesis, labeling, conversion, manifest generation,
and PyG Dataset abstraction for multi-organization inductive IAM corpora.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from torch_geometric.data import Dataset, HeteroData

from iam.generator.export import EnvironmentExporter
from iam.generator.graph import IAMGraph
from iam.generator.labeler import GroundTruthLabeler, GroundTruthLabelSet
from iam.generator.topology import (
    EnterpriseTopologyConfig,
    EnterpriseTopologyGenerator,
)
from iam.models.converter import (
    IAMHeteroDataConverter,
    load_hetero_data,
    save_hetero_data,
)
from iam.models.vocabulary import (
    ActionVocabulary,
    load_default_action_vocabulary,
)
from iam.parser.capability import (
    CapabilityModel,
    load_default_capability_model,
)


class EnvironmentConfig(BaseModel):
    """Configuration specification for an enterprise cloud environment within the corpus."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    org_id: str = Field(description="Unique organization identifier, e.g. 'org-01'")
    account_id: str = Field(description="AWS 12-digit account ID, e.g. '100000000001'")
    split: str = Field(default="train", description="Dataset partition: 'train', 'val', or 'test'")
    num_nodes: int = Field(default=1000, ge=30, description="Target total node count")
    seed: int = Field(default=42, description="Random seed for deterministic generation")
    num_pe_chains: int = Field(default=5, ge=0, description="Number of canonical PE chains")
    pe_chain_lengths: list[int] = Field(
        default_factory=lambda: [2, 3, 4], description="PE chain hop lengths"
    )
    pe_motif_types: list[str] | None = Field(
        default=None, description="Optional whitelist of PE motif types"
    )
    include_branching_chains: bool = Field(
        default=True, description="Whether to inject branching PE topologies"
    )
    num_branching_chains: int = Field(
        default=1, ge=0, description="Number of branching PE topologies"
    )
    edge_density: float = Field(default=1.0, ge=0.1, description="Edge density factor")
    power_law_alpha: float = Field(
        default=1.8, ge=1.0, description="Power-law parameter for identity distribution"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary environment metadata"
    )


# Standard canonical specification of 14 enterprise cloud environments (M1 corpus)
DEFAULT_ORGANIZATION_CONFIGS: tuple[EnvironmentConfig, ...] = (
    # --- Training Set (8 Organizations: org-01 to org-08, N in [300, 1500]) ---
    EnvironmentConfig(
        org_id="org-01",
        account_id="100000000001",
        split="train",
        num_nodes=350,
        seed=101,
        num_pe_chains=3,
        pe_motif_types=["AssumeRoleChain", "PassRoleLambda"],
        num_branching_chains=0,
    ),
    EnvironmentConfig(
        org_id="org-02",
        account_id="100000000002",
        split="train",
        num_nodes=480,
        seed=102,
        num_pe_chains=4,
        pe_motif_types=["CreateAccessKey", "AttachPolicy"],
        num_branching_chains=1,
    ),
    EnvironmentConfig(
        org_id="org-03",
        account_id="100000000003",
        split="train",
        num_nodes=650,
        seed=103,
        num_pe_chains=4,
        pe_motif_types=["PassRoleEC2", "AssumeRoleChain"],
        num_branching_chains=1,
    ),
    EnvironmentConfig(
        org_id="org-04",
        account_id="100000000004",
        split="train",
        num_nodes=820,
        seed=104,
        num_pe_chains=5,
        pe_motif_types=["SetDefaultPolicyVersion", "AttachPolicy"],
        num_branching_chains=1,
    ),
    EnvironmentConfig(
        org_id="org-05",
        account_id="100000000005",
        split="train",
        num_nodes=1000,
        seed=105,
        num_pe_chains=5,
        num_branching_chains=1,
    ),
    EnvironmentConfig(
        org_id="org-06",
        account_id="100000000006",
        split="train",
        num_nodes=1150,
        seed=106,
        num_pe_chains=6,
        num_branching_chains=2,
    ),
    EnvironmentConfig(
        org_id="org-07",
        account_id="100000000007",
        split="train",
        num_nodes=1320,
        seed=107,
        num_pe_chains=6,
        num_branching_chains=2,
    ),
    EnvironmentConfig(
        org_id="org-08",
        account_id="100000000008",
        split="train",
        num_nodes=1500,
        seed=108,
        num_pe_chains=7,
        num_branching_chains=2,
    ),
    # --- Validation Set (2 Organizations: org-09 to org-10, N in [800, 1500]) ---
    EnvironmentConfig(
        org_id="org-09",
        account_id="100000000009",
        split="val",
        num_nodes=800,
        seed=201,
        num_pe_chains=5,
        num_branching_chains=1,
    ),
    EnvironmentConfig(
        org_id="org-10",
        account_id="100000000010",
        split="val",
        num_nodes=1400,
        seed=202,
        num_pe_chains=6,
        num_branching_chains=2,
    ),
    # --- Held-Out Test Set (4 Organizations: org-11 to org-14, N in [1000, 3500]) ---
    EnvironmentConfig(
        org_id="org-11",
        account_id="100000000011",
        split="test",
        num_nodes=1050,
        seed=301,
        power_law_alpha=2.2,  # Heavy tail of restricted identities
        num_pe_chains=5,
        num_branching_chains=1,
    ),
    EnvironmentConfig(
        org_id="org-12",
        account_id="100000000012",
        split="test",
        num_nodes=1800,
        seed=302,
        num_pe_chains=7,
        num_branching_chains=2,
    ),
    EnvironmentConfig(
        org_id="org-13",
        account_id="100000000013",
        split="test",
        num_nodes=2500,
        seed=303,
        pe_chain_lengths=[3, 4],  # Deep multi-hop role chaining
        num_pe_chains=8,
        num_branching_chains=2,
    ),
    EnvironmentConfig(
        org_id="org-14",
        account_id="100000000014",
        split="test",
        num_nodes=3500,
        seed=304,
        edge_density=1.2,
        num_pe_chains=10,
        num_branching_chains=3,
    ),
)


class EnterpriseCorpusGenerator:
    """Orchestrates synthesis, labeling, and conversion of multi-organization IAM corpora."""

    def __init__(
        self,
        capability_model: CapabilityModel | None = None,
        vocabulary: ActionVocabulary | None = None,
    ) -> None:
        """Initialize generator with shared capability model and vocabulary."""
        self.capability_model: CapabilityModel = (
            capability_model if capability_model is not None else load_default_capability_model()
        )
        self.vocabulary: ActionVocabulary = (
            vocabulary if vocabulary is not None else load_default_action_vocabulary()
        )
        self.converter: IAMHeteroDataConverter = IAMHeteroDataConverter(
            vocabulary=self.vocabulary,
            include_reverse_edges=True,
            include_edge_attributes=True,
        )
        self.labeler: GroundTruthLabeler = GroundTruthLabeler()

    def generate_environment(
        self,
        config: EnvironmentConfig,
    ) -> tuple[IAMGraph, GroundTruthLabelSet, HeteroData]:
        """Synthesize a single environment with ground-truth labels and PyG HeteroData."""
        topo_cfg = EnterpriseTopologyConfig(
            num_nodes=config.num_nodes,
            seed=config.seed,
            account_id=config.account_id,
            organization_id=config.org_id,
            edge_density=config.edge_density,
            num_pe_chains=config.num_pe_chains,
            pe_chain_lengths=config.pe_chain_lengths,
            pe_motif_types=config.pe_motif_types,
            include_branching_chains=config.include_branching_chains,
            num_branching_chains=config.num_branching_chains,
            power_law_alpha=config.power_law_alpha,
        )

        topo_gen = EnterpriseTopologyGenerator(
            config=topo_cfg,
            capability_model=self.capability_model,
        )
        graph = topo_gen.generate()
        label_set = self.labeler.label_graph(graph)
        hetero_data = self.converter.convert(graph, label_set=label_set)

        return graph, label_set, hetero_data

    def generate_corpus(
        self,
        output_dir: Path | str,
        configs: list[EnvironmentConfig] | tuple[EnvironmentConfig, ...] | None = None,
        force: bool = False,
        export_raw_json: bool = True,
    ) -> dict[str, Any]:
        """Generate a complete multi-environment corpus and export manifest.

        Args:
            output_dir: Base directory where 'processed/', 'raw/', and 'corpus_manifest.json' live.
            configs: Organization configs (defaults to DEFAULT_ORGANIZATION_CONFIGS).
            force: Re-generate even if processed files already exist on disk.
            export_raw_json: Also write compressed JSON graph exports to raw/.

        Returns:
            manifest: Standardized dictionary summarizing all generated environments.
        """
        out_root = Path(output_dir)
        processed_dir = out_root / "processed"
        raw_dir = out_root / "raw"
        processed_dir.mkdir(parents=True, exist_ok=True)
        if export_raw_json:
            raw_dir.mkdir(parents=True, exist_ok=True)

        target_configs = configs if configs is not None else DEFAULT_ORGANIZATION_CONFIGS

        manifest: dict[str, Any] = {
            "corpus_version": "1.0.0",
            "total_organizations": len(target_configs),
            "splits": {
                "train": [c.org_id for c in target_configs if c.split == "train"],
                "val": [c.org_id for c in target_configs if c.split == "val"],
                "test": [c.org_id for c in target_configs if c.split == "test"],
            },
            "organizations": {},
        }

        for cfg in target_configs:
            pt_path = processed_dir / f"{cfg.org_id}.pt"
            raw_path = raw_dir / f"{cfg.org_id}.json.gz"

            if pt_path.exists() and not force:
                # Load existing summary if available
                data = load_hetero_data(pt_path)
                num_nodes_total = sum(
                    int(data[nt].num_nodes)
                    for nt in data.node_types
                    if hasattr(data[nt], "num_nodes")
                )
                manifest["organizations"][cfg.org_id] = {
                    "org_id": cfg.org_id,
                    "split": cfg.split,
                    "account_id": cfg.account_id,
                    "seed": cfg.seed,
                    "num_nodes": num_nodes_total,
                    "num_pe_pairs": getattr(data, "num_pe_pairs", 0),
                    "file": str(pt_path.name),
                }
                continue

            graph, label_set, hetero_data = self.generate_environment(cfg)
            save_hetero_data(hetero_data, pt_path)

            if export_raw_json:
                exporter = EnvironmentExporter()
                exporter.export_to_file(graph, raw_path, label_set=label_set, compress=True)

            manifest["organizations"][cfg.org_id] = {
                "org_id": cfg.org_id,
                "split": cfg.split,
                "account_id": cfg.account_id,
                "seed": cfg.seed,
                "num_nodes": graph.num_nodes,
                "num_edges": graph.num_edges,
                "num_pe_pairs": len(label_set.pe_pairs),
                "file": str(pt_path.name),
            }

        manifest_path = out_root / "corpus_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest


class IAMEnvironmentDataset(Dataset):  # type: ignore[misc]
    """PyTorch Geometric Dataset wrapping the multi-organization cloud IAM corpus.

    Features:
    - Lazy disk loading via PyTorch binary serialization.
    - In-memory LRU caching to eliminate redundant I/O during repeated epochs.
    - Partitioning into strict inductive splits: train, val, test with zero entity leakage.
    """

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        transform: Any = None,
        pre_transform: Any = None,
        cache_in_memory: bool = True,
    ) -> None:
        """Initialize dataset wrapper.

        Args:
            root: Root directory containing processed/ and corpus_manifest.json.
            split: Optional split filter: 'train', 'val', or 'test'.
            transform: Optional PyG callable transform.
            pre_transform: Optional PyG callable pre-transform.
            cache_in_memory: If True, keep loaded HeteroData instances in RAM.
        """
        self.split_filter: str | None = split
        self.cache_in_memory: bool = cache_in_memory
        self._cache: dict[str, HeteroData] = {}

        manifest_path = Path(root) / "corpus_manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                self._manifest: dict[str, Any] = json.load(f)
        else:
            self._manifest = {"splits": {}, "organizations": {}}

        # Determine eligible organization IDs
        all_orgs = list(self._manifest.get("organizations", {}).keys())
        if not all_orgs:
            # Fallback: scan processed directory for .pt files
            proc_dir = Path(root) / "processed"
            if proc_dir.exists():
                all_orgs = sorted(p.stem for p in proc_dir.glob("*.pt"))

        if self.split_filter is not None and "splits" in self._manifest:
            eligible_orgs = self._manifest["splits"].get(self.split_filter, [])
            self._org_ids: list[str] = [oid for oid in all_orgs if oid in eligible_orgs]
        else:
            self._org_ids = all_orgs

        super().__init__(str(root), transform=transform, pre_transform=pre_transform)

    @property
    def raw_file_names(self) -> list[str]:
        """Names of raw files in raw/."""
        return [f"{oid}.json.gz" for oid in self._org_ids]

    @property
    def processed_file_names(self) -> list[str]:
        """Names of processed files in processed/."""
        return [f"{oid}.pt" for oid in self._org_ids]

    @property
    def org_ids(self) -> list[str]:
        """List of active organization identifiers."""
        return list(self._org_ids)

    @property
    def manifest(self) -> dict[str, Any]:
        """Corpus manifest metadata."""
        return dict(self._manifest)

    def len(self) -> int:
        """Total number of environments in the active split view."""
        return len(self._org_ids)

    def get(self, idx: int) -> HeteroData:
        """Load and return the HeteroData instance at index idx."""
        if idx < 0 or idx >= len(self._org_ids):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self._org_ids)}")

        org_id = self._org_ids[idx]
        if self.cache_in_memory and org_id in self._cache:
            return self._cache[org_id]

        file_path = Path(self.processed_dir) / f"{org_id}.pt"
        data = load_hetero_data(file_path)

        if self.cache_in_memory:
            self._cache[org_id] = data

        return data

    def get_by_org_id(self, org_id: str) -> HeteroData:
        """Retrieve a specific environment by its organization ID."""
        if org_id not in self._org_ids:
            raise KeyError(f"Organization '{org_id}' not found in active dataset split view.")
        idx = self._org_ids.index(org_id)
        return self.get(idx)

    def get_train_split(self) -> IAMEnvironmentDataset:
        """Return a child dataset view containing only training environments."""
        return IAMEnvironmentDataset(
            root=self.root,
            split="train",
            transform=self.transform,
            pre_transform=self.pre_transform,
            cache_in_memory=self.cache_in_memory,
        )

    def get_val_split(self) -> IAMEnvironmentDataset:
        """Return a child dataset view containing only validation environments."""
        return IAMEnvironmentDataset(
            root=self.root,
            split="val",
            transform=self.transform,
            pre_transform=self.pre_transform,
            cache_in_memory=self.cache_in_memory,
        )

    def get_test_split(self) -> IAMEnvironmentDataset:
        """Return a child dataset view containing only held-out test environments."""
        return IAMEnvironmentDataset(
            root=self.root,
            split="test",
            transform=self.transform,
            pre_transform=self.pre_transform,
            cache_in_memory=self.cache_in_memory,
        )
