"""PyG HeteroData conversion and multirelational tensor pipeline.

Converts IAMGraph instances and pre-indexed export payloads into native PyTorch Geometric
HeteroData objects featuring typed edge stores, reverse message-passing relations,
ground-truth bridge annotations, and full index alignment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData

from iam.generator.graph import (
    GraphEdge,
    GraphNode,
    IAMGraph,
    NodeType,
)
from iam.generator.labeler import GroundTruthLabelSet
from iam.models.features import build_node_feature_dict
from iam.models.vocabulary import (
    ActionVocabulary,
    load_default_action_vocabulary,
)

# Canonical relational triples in AWS authorization multigraphs
CANONICAL_FORWARD_EDGE_TYPES: tuple[tuple[str, str, str], ...] = (
    ("User", "MemberOf", "Group"),
    ("User", "AttachedWith", "Policy"),
    ("Role", "AttachedWith", "Policy"),
    ("Group", "AttachedWith", "Policy"),
    ("User", "AssumesRole", "Role"),
    ("Role", "AssumesRole", "Role"),
    ("User", "PassesTo", "Role"),
    ("Role", "PassesTo", "Role"),
    ("Role", "PassesTo", "Resource"),
    ("Policy", "ActsOn", "Resource"),
    ("Policy", "ActsOn", "User"),
    ("Policy", "ActsOn", "Role"),
    ("Policy", "ActsOn", "Policy"),
)


def get_reverse_relation_name(rel_name: str) -> str:
    """Return the canonical reverse relation name (e.g. AssumesRole -> RevAssumesRole)."""
    if rel_name.startswith("Rev"):
        return rel_name[3:]
    return f"Rev{rel_name}"


def get_reverse_edge_type(edge_type: tuple[str, str, str]) -> tuple[str, str, str]:
    """Return the canonical reverse edge relation triple.

    Example:
        ("User", "AssumesRole", "Role") -> ("Role", "RevAssumesRole", "User")
    """
    src, rel, dst = edge_type
    return (dst, get_reverse_relation_name(rel), src)


def is_reverse_edge_type(edge_type: tuple[str, str, str]) -> bool:
    """Return True if the edge type is an auto-generated reverse message-passing relation."""
    return edge_type[1].startswith("Rev")


class IAMHeteroDataConverter:
    """Converts IAMGraph structures into validated PyG HeteroData objects."""

    def __init__(
        self,
        vocabulary: ActionVocabulary | None = None,
        include_reverse_edges: bool = True,
        include_edge_attributes: bool = True,
    ) -> None:
        """Initialize converter with action vocabulary and relation options."""
        self.vocabulary: ActionVocabulary = (
            vocabulary if vocabulary is not None else load_default_action_vocabulary()
        )
        self.include_reverse_edges: bool = include_reverse_edges
        self.include_edge_attributes: bool = include_edge_attributes

    def convert(
        self,
        graph: IAMGraph,
        label_set: GroundTruthLabelSet | None = None,
    ) -> HeteroData:
        """Convert an IAMGraph into a fully validated PyTorch Geometric HeteroData instance.

        Args:
            graph: Source IAMGraph with typed nodes and multi-relational edges.
            label_set: Optional GroundTruthLabelSet containing positive PE pairs and witness paths.

        Returns:
            hetero_data: Populated HeteroData object containing:
                - data[node_type].x: Node feature tensor [N_type, D_type]
                - data[edge_type].edge_index: COO tensor [2, E_rel]
                - data[edge_type].edge_is_bridge: Boolean tensor [E_rel] (Condition E bottleneck flag)
                - data[edge_type].edge_attr: Multi-hot action bitmask tensor [E_rel, |A_vocab|]
                - data[rev_edge_type].edge_index: Reverse relation COO tensor for message passing
                - data.pe_pairs: Ground truth PE pairs and hop distances (if label_set provided)
                - data.node_id_to_idx & data.idx_to_node_id: Bijective index mappings
        """
        # 1. Build node feature matrices and integer index mapping
        feature_dict, node_id_to_idx = build_node_feature_dict(graph, vocabulary=self.vocabulary)

        data = HeteroData()

        # 2. Populate typed node stores
        nodes_by_type: dict[str, list[GraphNode]] = {t.value: [] for t in NodeType}
        for node in graph.get_nodes():
            nodes_by_type[node.node_type.value].append(node)

        idx_to_node_id: dict[str, dict[int, str]] = {}
        for n_type_str, n_list in nodes_by_type.items():
            data[n_type_str].x = feature_dict[n_type_str]
            data[n_type_str].node_ids = [n.id for n in n_list]
            data[n_type_str].arns = [n.arn for n in n_list]
            data[n_type_str].departments = [n.department or "" for n in n_list]
            data[n_type_str].is_high_value = torch.tensor(
                [bool(n.is_high_value) for n in n_list], dtype=torch.bool
            )
            data[n_type_str].is_admin = torch.tensor(
                [bool(n.is_admin) for n in n_list], dtype=torch.bool
            )
            data[n_type_str].num_nodes = len(n_list)
            idx_to_node_id[n_type_str] = {i: n.id for i, n in enumerate(n_list)}

        # 3. Group directed edges by relation triple (src_type, rel, dst_type)
        grouped_edges: dict[tuple[str, str, str], list[GraphEdge]] = {}
        for edge in graph.get_edges():
            src_node = graph.get_node(edge.source)
            dst_node = graph.get_node(edge.target)
            if src_node is None or dst_node is None:
                continue

            rel_key = (
                src_node.node_type.value,
                edge.relation.value,
                dst_node.node_type.value,
            )
            if rel_key not in grouped_edges:
                grouped_edges[rel_key] = []
            grouped_edges[rel_key].append(edge)

        # 4. Construct edge tensors for all active forward relation triples
        forward_triples = list(grouped_edges.keys())

        for triple, edge_list in grouped_edges.items():
            src_type, _rel_str, dst_type = triple
            src_map = node_id_to_idx[src_type]
            dst_map = node_id_to_idx[dst_type]

            u_indices: list[int] = []
            v_indices: list[int] = []
            bridge_flags: list[bool] = []
            action_vectors: list[np.ndarray] = []

            for edge in edge_list:
                u_idx = src_map[edge.source]
                v_idx = dst_map[edge.target]
                u_indices.append(u_idx)
                v_indices.append(v_idx)
                bridge_flags.append(edge.is_bridge)

                if self.include_edge_attributes:
                    act_vec = self.vocabulary.encode_actions(edge.actions, expand_wildcards=True)
                    action_vectors.append(act_vec)

            edge_index_tensor = torch.tensor(
                [u_indices, v_indices],
                dtype=torch.long,
            )
            bridge_tensor = torch.tensor(bridge_flags, dtype=torch.bool)

            data[triple].edge_index = edge_index_tensor
            data[triple].edge_is_bridge = bridge_tensor

            if self.include_edge_attributes:
                data[triple].edge_attr = torch.from_numpy(
                    np.array(action_vectors, dtype=np.float32)
                )

        # 5. Construct reverse relations for bidirectional message passing
        if self.include_reverse_edges:
            for triple in forward_triples:
                rev_triple = get_reverse_edge_type(triple)
                fwd_edge_index = data[triple].edge_index

                # Invert source and destination: [v, u]
                rev_edge_index = fwd_edge_index[[1, 0], :]
                num_rev_edges = rev_edge_index.shape[1]

                data[rev_triple].edge_index = rev_edge_index
                # Reverse edges are message-passing conduits, not forward attack bridges
                data[rev_triple].edge_is_bridge = torch.zeros(num_rev_edges, dtype=torch.bool)

                if self.include_edge_attributes and hasattr(data[triple], "edge_attr"):
                    # Retain symmetric edge action capabilities
                    data[rev_triple].edge_attr = data[triple].edge_attr.clone()

        # 6. Global Metadata & Identifiers
        data.graph_id = graph.graph_id
        data.organization_id = graph.metadata.get("organization_id", "unknown")
        data.account_id = graph.metadata.get("account_id", "123456789012")
        data.metadata = dict(graph.metadata)
        data.node_id_to_idx = node_id_to_idx
        data.idx_to_node_id = idx_to_node_id

        # 7. Ground-Truth Privilege Escalation Annotations
        if label_set is not None:
            pe_pairs_list: list[dict[str, Any]] = []
            for pe in label_set.pe_pairs:
                s_node = graph.get_node(pe.source_id)
                t_node = graph.get_node(pe.target_id)
                if s_node is not None and t_node is not None:
                    s_idx = node_id_to_idx[s_node.node_type.value][s_node.id]
                    t_idx = node_id_to_idx[t_node.node_type.value][t_node.id]
                    pe_pairs_list.append(
                        {
                            "source_type": s_node.node_type.value,
                            "source_index": s_idx,
                            "source_id": s_node.id,
                            "target_type": t_node.node_type.value,
                            "target_index": t_idx,
                            "target_id": t_node.id,
                            "hop_distance": pe.hop_distance,
                        }
                    )
            data["pe_pairs"] = pe_pairs_list
            data["num_pe_pairs"] = len(pe_pairs_list)

        # 8. Structural validation via PyG built-in schema assertions
        data.validate()

        return data

    def convert_from_pyg_ready_dict(
        self,
        pyg_dict: dict[str, Any],
        feature_dict: dict[str, torch.Tensor] | None = None,
    ) -> HeteroData:
        """Fast conversion from pre-indexed dictionary generated by EnvironmentExporter.to_pyg_ready_dict()."""
        data = HeteroData()

        # 1. Node stores
        node_mappings: dict[str, dict[str, int]] = pyg_dict.get("node_mappings", {})
        num_nodes_by_type: dict[str, int] = pyg_dict.get("num_nodes_by_type", {})

        idx_to_node_id: dict[str, dict[int, str]] = {}
        for n_type, n_map in node_mappings.items():
            num_nodes = num_nodes_by_type.get(n_type, len(n_map))
            data[n_type].num_nodes = num_nodes
            data[n_type].node_ids = list(n_map.keys())
            idx_to_node_id[n_type] = {idx: nid for nid, idx in n_map.items()}

            if feature_dict is not None and n_type in feature_dict:
                data[n_type].x = feature_dict[n_type]

        # 2. Edge stores
        edge_indices_dict: dict[str, list[list[int]]] = pyg_dict.get("edge_indices", {})
        bridge_indices_dict: dict[str, list[int]] = pyg_dict.get("bridge_edge_indices", {})

        for rel_key, (u_list, v_list) in edge_indices_dict.items():
            parts = rel_key.split("__")
            if len(parts) != 3:
                continue
            triple = (parts[0], parts[1], parts[2])

            edge_index = torch.tensor([u_list, v_list], dtype=torch.long)
            num_edges = len(u_list)

            bridge_mask = torch.zeros(num_edges, dtype=torch.bool)
            bridge_positions = bridge_indices_dict.get(rel_key, [])
            for pos in bridge_positions:
                if 0 <= pos < num_edges:
                    bridge_mask[pos] = True

            data[triple].edge_index = edge_index
            data[triple].edge_is_bridge = bridge_mask

            if self.include_reverse_edges:
                rev_triple = get_reverse_edge_type(triple)
                data[rev_triple].edge_index = edge_index[[1, 0], :]
                data[rev_triple].edge_is_bridge = torch.zeros(num_edges, dtype=torch.bool)

        # 3. Graph identifiers & PE pairs
        data.graph_id = pyg_dict.get("graph_id", "default_graph")
        data.organization_id = pyg_dict.get("organization_id", "unknown")
        data.node_id_to_idx = node_mappings
        data.idx_to_node_id = idx_to_node_id

        if "pe_pairs" in pyg_dict:
            data["pe_pairs"] = pyg_dict["pe_pairs"]
            data["num_pe_pairs"] = len(pyg_dict["pe_pairs"])

        data.validate()
        return data


def save_hetero_data(data: HeteroData, path: Path | str) -> Path:
    """Save a HeteroData instance to disk using PyTorch binary serialization."""
    out_p = Path(path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_p)
    return out_p


def load_hetero_data(path: Path | str) -> HeteroData:
    """Load a HeteroData instance from disk using PyTorch binary serialization."""
    in_p = Path(path)
    if not in_p.exists():
        raise FileNotFoundError(f"HeteroData file not found: {in_p}")
    loaded = torch.load(in_p, weights_only=False)
    if not isinstance(loaded, HeteroData):
        raise TypeError(f"Expected HeteroData, got {type(loaded).__name__}")
    return loaded
