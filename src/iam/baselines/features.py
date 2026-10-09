"""Topological and attribute edge-pair featurizer for classical IAM link prediction.

Extracts invariant structural, topological, departmental, and capability-aware
feature vectors for candidate (u, v) node pairs over partially observed graphs G_o.
Operates on both IAMGraph and PyG HeteroData, strictly excluding reverse relations.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData

from iam.generator.graph import GraphNode, IAMGraph, NodeType
from iam.models.converter import is_reverse_edge_type

FEATURE_NAMES: list[str] = [
    "common_neighbors",
    "jaccard_similarity",
    "adamic_adar",
    "preferential_attachment",
    "resource_allocation",
    "directed_shortest_path",
    "is_reachable_in_go",
    "reverse_shortest_path",
    "reverse_is_reachable",
    "out_degree_u",
    "in_degree_u",
    "out_degree_v",
    "in_degree_v",
    "same_department",
    "different_department",
    "u_is_admin",
    "v_is_high_value",
    "cross_account",
    "u_is_identity",
    "v_is_role_or_resource",
    "u_has_passrole",
    "u_has_assumerole",
    "u_has_admin_privilege",
    "action_overlap_count",
]


class EdgePairFeaturizer:
    """Extracts compact, discriminative topological and semantic feature vectors for edge pairs.

    Designed for classical statistical and tabular machine learning baselines (e.g. Random Forest),
    capturing local neighborhood overlap, global directed reachability in G_o, organizational
    department affinity, and security privilege indicators.
    """

    def __init__(self, max_path_cutoff: int = 6) -> None:
        self.max_path_cutoff = max_path_cutoff
        self.feature_names: list[str] = list(FEATURE_NAMES)

    @property
    def num_features(self) -> int:
        """Total number of features in each edge pair feature vector."""
        return len(self.feature_names)

    def featurize_pairs(
        self,
        graph_or_data: IAMGraph | HeteroData,
        pairs: list[tuple[str, str]],
    ) -> np.ndarray:
        """Extract feature matrix for an ordered list of candidate (u, v) pairs.

        Args:
            graph_or_data: Partially observed graph G_o as IAMGraph or HeteroData.
            pairs: List of (source_id, target_id) string tuples to featurize.

        Returns:
            np.ndarray of shape (len(pairs), num_features) of dtype np.float32.
        """
        if not pairs:
            return np.zeros((0, self.num_features), dtype=np.float32)

        nx_g, node_map, node_actions = self._extract_graph_and_metadata(graph_or_data)
        undirected_g = nx_g.to_undirected()

        # Precompute cached lookups
        neighbors: dict[str, set[str]] = {n: set(undirected_g.neighbors(n)) for n in nx_g.nodes()}
        degrees: dict[str, int] = {n: undirected_g.degree(n) for n in nx_g.nodes()}
        out_degrees: dict[str, int] = dict(nx_g.out_degree())
        in_degrees: dict[str, int] = dict(nx_g.in_degree())

        # Forward and reverse shortest path caches
        sp_cache: dict[str, dict[str, int]] = {}
        rev_sp_cache: dict[str, dict[str, int]] = {}

        num_pairs = len(pairs)
        features = np.zeros((num_pairs, self.num_features), dtype=np.float32)

        disconnected_val = float(self.max_path_cutoff + 1)

        for i, (u, v) in enumerate(pairs):
            row = np.zeros(self.num_features, dtype=np.float32)

            u_node = node_map.get(u)
            v_node = node_map.get(v)

            # Local neighborhood overlap on undirected projection
            nbrs_u = neighbors.get(u, set())
            nbrs_v = neighbors.get(v, set())
            common_nbrs = nbrs_u & nbrs_v
            cn_count = len(common_nbrs)
            union_count = len(nbrs_u | nbrs_v)

            row[0] = float(cn_count)  # common_neighbors
            row[1] = float(cn_count / (union_count + 1e-9))  # jaccard_similarity

            # Adamic-Adar & Resource Allocation
            aa_val = 0.0
            ra_val = 0.0
            for z in common_nbrs:
                deg_z = degrees.get(z, 1)
                if deg_z > 1:
                    aa_val += 1.0 / math.log(deg_z)
                ra_val += 1.0 / (deg_z + 1e-9)
            row[2] = float(aa_val)  # adamic_adar
            row[3] = float(len(nbrs_u) * len(nbrs_v))  # preferential_attachment
            row[4] = float(ra_val)  # resource_allocation

            # Directed shortest path in G_o (u -> v)
            if u not in sp_cache:
                if u in nx_g:
                    sp_cache[u] = nx.single_source_shortest_path_length(
                        nx_g, u, cutoff=self.max_path_cutoff
                    )
                else:
                    sp_cache[u] = {}
            fwd_dist = float(sp_cache[u].get(v, disconnected_val))
            row[5] = fwd_dist  # directed_shortest_path
            row[6] = 1.0 if fwd_dist <= self.max_path_cutoff else 0.0  # is_reachable_in_go

            # Reverse shortest path in G_o (v -> u)
            if v not in rev_sp_cache:
                if v in nx_g:
                    rev_sp_cache[v] = nx.single_source_shortest_path_length(
                        nx_g, v, cutoff=self.max_path_cutoff
                    )
                else:
                    rev_sp_cache[v] = {}
            rev_dist = float(rev_sp_cache[v].get(u, disconnected_val))
            row[7] = rev_dist  # reverse_shortest_path
            row[8] = 1.0 if rev_dist <= self.max_path_cutoff else 0.0  # reverse_is_reachable

            # Degree prominence
            row[9] = float(out_degrees.get(u, 0))  # out_degree_u
            row[10] = float(in_degrees.get(u, 0))  # in_degree_u
            row[11] = float(out_degrees.get(v, 0))  # out_degree_v
            row[12] = float(in_degrees.get(v, 0))  # in_degree_v

            # Department attributes
            if u_node is not None and v_node is not None:
                dept_u = u_node.department
                dept_v = v_node.department
                if dept_u is not None and dept_v is not None:
                    row[13] = 1.0 if dept_u == dept_v else 0.0  # same_department
                    row[14] = 1.0 if dept_u != dept_v else 0.0  # different_department

                row[15] = 1.0 if u_node.is_admin else 0.0  # u_is_admin
                row[16] = (
                    1.0 if (v_node.is_high_value or v_node.is_admin) else 0.0
                )  # v_is_high_value

                if u_node.account_id and v_node.account_id:
                    row[17] = (
                        1.0 if u_node.account_id != v_node.account_id else 0.0
                    )  # cross_account

                row[18] = 1.0 if u_node.node_type in (NodeType.USER, NodeType.ROLE) else 0.0
                row[19] = 1.0 if v_node.node_type in (NodeType.ROLE, NodeType.RESOURCE) else 0.0
            else:
                # Fallback based on node ID heuristics
                u_lower = u.lower()
                v_lower = v.lower()
                row[15] = 1.0 if "admin" in u_lower or "root" in u_lower else 0.0
                row[16] = (
                    1.0 if "vault" in v_lower or "crown" in v_lower or "admin" in v_lower else 0.0
                )
                row[18] = 1.0 if "user" in u_lower or "role" in u_lower else 0.0
                row[19] = (
                    1.0 if "role" in v_lower or "resource" in v_lower or "s3" in v_lower else 0.0
                )

            # Capability action indicators
            actions_u = node_actions.get(u, set())
            actions_v = node_actions.get(v, set())

            has_passrole = (
                any("passrole" in act.lower() for act in actions_u) or "passrole" in u.lower()
            )
            has_assumerole = (
                any("assumerole" in act.lower() for act in actions_u) or "assume" in u.lower()
            )
            has_admin_act = any("*" in act or "admin" in act.lower() for act in actions_u) or (
                u_node is not None and u_node.is_admin
            )

            row[20] = 1.0 if has_passrole else 0.0
            row[21] = 1.0 if has_assumerole else 0.0
            row[22] = 1.0 if has_admin_act else 0.0
            row[23] = float(len(actions_u & actions_v))

            features[i] = row

        return features

    def featurize_pairs_tensor(
        self,
        graph_or_data: IAMGraph | HeteroData,
        pairs: list[tuple[str, str]],
    ) -> torch.Tensor:
        """Extract feature matrix and return as a PyTorch FloatTensor."""
        feat_array = self.featurize_pairs(graph_or_data, pairs)
        return torch.from_numpy(feat_array)

    def _extract_graph_and_metadata(
        self,
        graph_or_data: IAMGraph | HeteroData,
    ) -> tuple[nx.DiGraph, dict[str, GraphNode], dict[str, set[str]]]:
        """Convert input graph structure into DiGraph, GraphNode map, and node action sets."""
        if isinstance(graph_or_data, IAMGraph):
            nx_g = nx.DiGraph()
            node_map = dict(graph_or_data._nodes)
            node_actions: dict[str, set[str]] = {n: set() for n in node_map}

            for nid in node_map:
                nx_g.add_node(nid)

            for u, v, data in graph_or_data.nx_graph.edges(data=True):
                nx_g.add_edge(u, v)
                edge_acts = data.get("actions", [])
                if edge_acts:
                    node_actions.setdefault(u, set()).update(edge_acts)

            # Ingest policy actions from attached policies
            for nid, node in node_map.items():
                if node.policy_document is not None:
                    for stmt in node.policy_document.statements:
                        node_actions.setdefault(nid, set()).update(stmt.action)

            # Propagate policy actions to attached principals
            for u, v, data in graph_or_data.nx_graph.edges(data=True):
                rel = data.get("relation", "")
                if rel == "AttachedWith" and v in node_actions:
                    node_actions.setdefault(u, set()).update(node_actions[v])

            # Propagate group actions to member users
            for u, v, data in graph_or_data.nx_graph.edges(data=True):
                rel = data.get("relation", "")
                if rel == "MemberOf" and v in node_actions:
                    node_actions.setdefault(u, set()).update(node_actions[v])

            return nx_g, node_map, node_actions

        # PyG HeteroData representation
        nx_g = nx.DiGraph()
        node_map = {}
        node_actions = {}

        for node_type in graph_or_data.node_types:
            node_ids: list[str] = getattr(graph_or_data[node_type], "node_ids", [])
            arns: list[str] = getattr(graph_or_data[node_type], "arns", [])
            for i, nid in enumerate(node_ids):
                arn = arns[i] if i < len(arns) else nid
                is_admin = bool("admin" in nid.lower() or "root" in nid.lower())
                is_high_val = bool("vault" in nid.lower() or "crown" in nid.lower() or is_admin)
                dept = None
                for d in ("SecOps", "DevOps", "DataBI", "QA", "Billing", "Interns"):
                    if d.lower() in nid.lower():
                        dept = d
                        break

                gn = GraphNode(
                    id=nid,
                    node_type=NodeType(node_type),
                    arn=arn,
                    name=nid,
                    department=dept,
                    is_admin=is_admin,
                    is_high_value=is_high_val,
                )
                node_map[nid] = gn
                nx_g.add_node(nid)
                node_actions[nid] = set()

        # Connect observed edges, strictly ignoring reverse relations
        for edge_type in graph_or_data.edge_types:
            if is_reverse_edge_type(edge_type):
                continue

            src_type, rel, dst_type = edge_type
            src_ids = getattr(graph_or_data[src_type], "node_ids", [])
            dst_ids = getattr(graph_or_data[dst_type], "node_ids", [])
            edge_index = graph_or_data[edge_type].edge_index

            for i in range(edge_index.size(1)):
                u = src_ids[int(edge_index[0, i])]
                v = dst_ids[int(edge_index[1, i])]
                nx_g.add_edge(u, v, relation=rel)

        return nx_g, node_map, node_actions
