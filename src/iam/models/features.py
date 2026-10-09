"""Heterogeneous node feature representation and centrality extraction engine.

Projects IAMGraph nodes (User, Role, Policy, Group, Resource) into fixed-dimension
continuous and categorical PyTorch tensors for inductive relational GNN message passing.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch

from iam.generator.graph import (
    EdgeRelation,
    GraphNode,
    IAMGraph,
    NodeType,
)
from iam.models.vocabulary import (
    CANONICAL_DEPARTMENTS,
    CANONICAL_RESOURCE_TYPES,
    CANONICAL_SERVICES,
    ActionVocabulary,
    FeatureDimensionConfig,
    get_feature_dimension_config,
    load_default_action_vocabulary,
    one_hot_encode,
    parse_arn_service_and_type,
)
from iam.parser.schema import Effect


class CentralityExtractor:
    """High-throughput, vectorized centrality feature extraction over IAMGraph topologies.

    Computes per-relation in/out degree counts and global PageRank scores in a single
    pass, caching lookups for O(1) node-level extraction.
    """

    def __init__(self, graph: IAMGraph) -> None:
        """Initialize and precompute topological metrics across the entire graph."""
        self._graph = graph
        nx_g = graph.nx_graph
        self._num_nodes = max(1, graph.num_nodes)

        # Precompute per-relation in-degrees and out-degrees
        self._in_degrees: dict[str, dict[EdgeRelation, int]] = {
            nid: dict.fromkeys(EdgeRelation, 0) for nid in nx_g.nodes()
        }
        self._out_degrees: dict[str, dict[EdgeRelation, int]] = {
            nid: dict.fromkeys(EdgeRelation, 0) for nid in nx_g.nodes()
        }

        for u, v, data in nx_g.edges(data=True):
            rel_str = data.get("relation")
            if rel_str is not None:
                try:
                    rel = EdgeRelation(rel_str)
                    if u in self._out_degrees:
                        self._out_degrees[u][rel] += 1
                    if v in self._in_degrees:
                        self._in_degrees[v][rel] += 1
                except (ValueError, KeyError):
                    pass

        # Global PageRank computation via NetworkX
        try:
            self._pagerank: dict[str, float] = nx.pagerank(nx_g, alpha=0.85, max_iter=100, tol=1e-5)
        except Exception:
            # Fallback uniform distribution for disconnected or empty graphs
            uniform_score = 1.0 / self._num_nodes
            self._pagerank = dict.fromkeys(nx_g.nodes(), uniform_score)

    def get_relational_degree_vector(self, node_id: str) -> np.ndarray:
        """Return a 10-dimensional log1p-transformed vector of in/out degrees across all 5 relations."""
        vec = np.zeros(10, dtype=np.float32)
        in_map = self._in_degrees.get(node_id, {})
        out_map = self._out_degrees.get(node_id, {})

        relations = list(EdgeRelation)
        for i, rel in enumerate(relations):
            # In-degrees: indices 0..4
            vec[i] = np.log1p(float(in_map.get(rel, 0)))
            # Out-degrees: indices 5..9
            vec[i + 5] = np.log1p(float(out_map.get(rel, 0)))

        return vec

    def get_pagerank(self, node_id: str) -> float:
        """Return the global PageRank score for a node."""
        return float(self._pagerank.get(node_id, 1.0 / self._num_nodes))

    def get_full_centrality_vector(self, node_id: str) -> np.ndarray:
        """Return an 11-dimensional vector: 10 relational degrees + 1 PageRank score."""
        vec = np.zeros(11, dtype=np.float32)
        vec[:10] = self.get_relational_degree_vector(node_id)
        vec[10] = self.get_pagerank(node_id)
        return vec

    def get_compact_centrality(
        self,
        node_id: str,
        in_rel: EdgeRelation,
        out_rel: EdgeRelation | None = None,
    ) -> np.ndarray:
        """Return a 3-dimensional compact centrality vector: in-degree, out-degree, and PageRank."""
        vec = np.zeros(3, dtype=np.float32)
        in_map = self._in_degrees.get(node_id, {})
        out_map = self._out_degrees.get(node_id, {})

        vec[0] = np.log1p(float(in_map.get(in_rel, 0)))
        if out_rel is not None:
            vec[1] = np.log1p(float(out_map.get(out_rel, 0)))
        vec[2] = self.get_pagerank(node_id)
        return vec


class NodeFeatureExtractor:
    """Extracts typed, fixed-dimension feature vectors for individual IAM graph entities."""

    def __init__(
        self,
        graph: IAMGraph,
        vocabulary: ActionVocabulary | None = None,
        centrality_extractor: CentralityExtractor | None = None,
    ) -> None:
        """Initialize feature extractor for a specific graph instance."""
        self._graph = graph
        self._vocab = vocabulary if vocabulary is not None else load_default_action_vocabulary()
        self._centrality = (
            centrality_extractor if centrality_extractor is not None else CentralityExtractor(graph)
        )
        self._dim_config = get_feature_dimension_config(self._vocab)

        # Pre-cache policy permissions for high-throughput aggregation
        self._policy_actions_cache: dict[str, set[str]] = {}
        for p_node in graph.get_nodes_by_type(NodeType.POLICY):
            self._policy_actions_cache[p_node.id] = self._extract_raw_policy_actions(p_node)

    @property
    def vocabulary(self) -> ActionVocabulary:
        """Underlying ActionVocabulary instance."""
        return self._vocab

    @property
    def dimension_config(self) -> FeatureDimensionConfig:
        """Feature dimension specification."""
        return self._dim_config

    def _extract_raw_policy_actions(self, policy_node: GraphNode) -> set[str]:
        """Extract allowed action strings from a policy node's AST document or direct edges."""
        actions: set[str] = set()
        doc = policy_node.policy_document
        if doc is not None:
            for stmt in doc.statements:
                if stmt.effect == Effect.ALLOW and stmt.action:
                    actions.update(stmt.action)

        # Also collect actions specified directly on incident ActsOn edges
        nx_g = self._graph.nx_graph
        if policy_node.id in nx_g:
            for _u, _v, data in nx_g.out_edges(policy_node.id, data=True):
                edge_acts = data.get("actions", [])
                actions.update(edge_acts)

        return actions

    def _collect_attached_policy_actions(self, principal_id: str) -> set[str]:
        """Collect actions from all policies directly attached to a principal."""
        actions: set[str] = set()
        nx_g = self._graph.nx_graph
        if principal_id not in nx_g:
            return actions

        for _u, target_id, data in nx_g.out_edges(principal_id, data=True):
            if data.get("relation") == EdgeRelation.ATTACHED_WITH.value:
                actions.update(self._policy_actions_cache.get(target_id, set()))

        return actions

    def _collect_user_effective_actions(self, user_node: GraphNode) -> set[str]:
        """Collect effective actions from directly attached policies and group memberships."""
        actions = self._collect_attached_policy_actions(user_node.id)
        nx_g = self._graph.nx_graph
        if user_node.id not in nx_g:
            return actions

        # Traverse groups (MemberOf)
        for _u, group_id, data in nx_g.out_edges(user_node.id, data=True):
            if data.get("relation") == EdgeRelation.MEMBER_OF.value:
                # Collect policies attached to this group
                actions.update(self._collect_attached_policy_actions(group_id))

        return actions

    def extract_user_features(self, user_node: GraphNode) -> np.ndarray:
        """Extract user features.

        Dimensions:
            Department(7) + SecurityFlags(2) + Centrality(11) + ActionBitmask(|A|)
        """
        dept_onehot = one_hot_encode(user_node.department, CANONICAL_DEPARTMENTS)
        flags = np.array(
            [float(user_node.is_admin), float(user_node.is_high_value)],
            dtype=np.float32,
        )
        centrality = self._centrality.get_full_centrality_vector(user_node.id)

        effective_actions = self._collect_user_effective_actions(user_node)
        action_bitmask = self._vocab.encode_actions(effective_actions, expand_wildcards=True)

        feature_vec = np.concatenate([dept_onehot, flags, centrality, action_bitmask])
        assert len(feature_vec) == self._dim_config.user_dim, (
            f"User feature dim mismatch: expected {self._dim_config.user_dim}, got {len(feature_vec)}"
        )
        return feature_vec

    def extract_role_features(self, role_node: GraphNode) -> np.ndarray:
        """Extract role features.

        Dimensions:
            ServiceRoleFlag(1) + AdminFlag(1) + HighValue(1) + Department(7) + Centrality(11) + ActionBitmask(|A|)
        """
        # Determine if service principal role
        is_service_role = False
        if role_node.trust_policy is not None:
            for stmt in role_node.trust_policy.statements:
                if (
                    stmt.principal
                    and isinstance(stmt.principal.raw, dict)
                    and "Service" in stmt.principal.raw
                ):
                    is_service_role = True
                    break
        if role_node.metadata.get("is_service_role", False):
            is_service_role = True

        flags = np.array(
            [
                float(is_service_role),
                float(role_node.is_admin),
                float(role_node.is_high_value),
            ],
            dtype=np.float32,
        )
        dept_onehot = one_hot_encode(role_node.department, CANONICAL_DEPARTMENTS)
        centrality = self._centrality.get_full_centrality_vector(role_node.id)

        attached_actions = self._collect_attached_policy_actions(role_node.id)
        action_bitmask = self._vocab.encode_actions(attached_actions, expand_wildcards=True)

        feature_vec = np.concatenate([flags, dept_onehot, centrality, action_bitmask])
        assert len(feature_vec) == self._dim_config.role_dim, (
            f"Role feature dim mismatch: expected {self._dim_config.role_dim}, got {len(feature_vec)}"
        )
        return feature_vec

    def extract_policy_features(self, policy_node: GraphNode) -> np.ndarray:
        """Extract policy features.

        Dimensions:
            StatementStats(3) + AdminFlag(1) + WildcardResource(1) + Centrality(3) + ActionBitmask(|A|)
        """
        doc = policy_node.policy_document
        stmt_count = 0.0
        allow_count = 0.0
        deny_count = 0.0
        has_wildcard_res = False

        if doc is not None:
            stmt_count = float(len(doc.statements))
            for stmt in doc.statements:
                if stmt.effect == Effect.ALLOW:
                    allow_count += 1.0
                elif stmt.effect == Effect.DENY:
                    deny_count += 1.0
                if "*" in stmt.resource:
                    has_wildcard_res = True

        stmt_stats = np.array(
            [np.log1p(stmt_count), np.log1p(allow_count), np.log1p(deny_count)],
            dtype=np.float32,
        )
        flags = np.array(
            [float(policy_node.is_admin), float(has_wildcard_res)],
            dtype=np.float32,
        )

        # Centrality: in from principals (AttachedWith), out to resources (ActsOn), PageRank
        centrality = self._centrality.get_compact_centrality(
            policy_node.id,
            in_rel=EdgeRelation.ATTACHED_WITH,
            out_rel=EdgeRelation.ACTS_ON,
        )

        policy_actions = self._policy_actions_cache.get(policy_node.id, set())
        action_bitmask = self._vocab.encode_actions(policy_actions, expand_wildcards=True)

        feature_vec = np.concatenate([stmt_stats, flags, centrality, action_bitmask])
        assert len(feature_vec) == self._dim_config.policy_dim, (
            f"Policy feature dim mismatch: expected {self._dim_config.policy_dim}, got {len(feature_vec)}"
        )
        return feature_vec

    def extract_group_features(self, group_node: GraphNode) -> np.ndarray:
        """Extract group features.

        Dimensions:
            Department(7) + AdminFlag(1) + Centrality(3) + ActionBitmask(|A|)
        """
        dept_onehot = one_hot_encode(group_node.department, CANONICAL_DEPARTMENTS)
        flags = np.array([float(group_node.is_admin)], dtype=np.float32)

        # Centrality: in from users (MemberOf), out to policies (AttachedWith), PageRank
        centrality = self._centrality.get_compact_centrality(
            group_node.id,
            in_rel=EdgeRelation.MEMBER_OF,
            out_rel=EdgeRelation.ATTACHED_WITH,
        )

        attached_actions = self._collect_attached_policy_actions(group_node.id)
        action_bitmask = self._vocab.encode_actions(attached_actions, expand_wildcards=True)

        feature_vec = np.concatenate([dept_onehot, flags, centrality, action_bitmask])
        assert len(feature_vec) == self._dim_config.group_dim, (
            f"Group feature dim mismatch: expected {self._dim_config.group_dim}, got {len(feature_vec)}"
        )
        return feature_vec

    def extract_resource_features(self, resource_node: GraphNode) -> np.ndarray:
        """Extract resource features.

        Dimensions:
            ResourceType(8) + ServicePrefix(8) + HighValue(1) + Centrality(3)
        """
        service_prefix, res_type = parse_arn_service_and_type(resource_node.arn)
        type_onehot = one_hot_encode(res_type, CANONICAL_RESOURCE_TYPES)
        service_onehot = one_hot_encode(service_prefix, CANONICAL_SERVICES)
        flags = np.array([float(resource_node.is_high_value)], dtype=np.float32)

        # Centrality: in from policies (ActsOn), in from execution roles (PassesTo), PageRank
        centrality = self._centrality.get_compact_centrality(
            resource_node.id,
            in_rel=EdgeRelation.ACTS_ON,
            out_rel=EdgeRelation.PASSES_TO,
        )

        feature_vec = np.concatenate([type_onehot, service_onehot, flags, centrality])
        assert len(feature_vec) == self._dim_config.resource_dim, (
            f"Resource feature dim mismatch: expected {self._dim_config.resource_dim}, got {len(feature_vec)}"
        )
        return feature_vec

    def extract_node_feature(self, node: GraphNode) -> np.ndarray:
        """Extract feature vector for any node entity based on its NodeType."""
        if node.node_type == NodeType.USER:
            return self.extract_user_features(node)
        elif node.node_type == NodeType.ROLE:
            return self.extract_role_features(node)
        elif node.node_type == NodeType.POLICY:
            return self.extract_policy_features(node)
        elif node.node_type == NodeType.GROUP:
            return self.extract_group_features(node)
        elif node.node_type == NodeType.RESOURCE:
            return self.extract_resource_features(node)
        else:
            raise ValueError(f"Unsupported node type: {node.node_type}")


def build_node_feature_dict(
    graph: IAMGraph,
    vocabulary: ActionVocabulary | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, int]]]:
    """Build 2D float32 PyTorch feature tensors for each NodeType in the IAMGraph.

    The row ordering of each tensor is strictly aligned with the contiguous integer
    indices returned in node_id_to_idx, guaranteeing 100% index correspondence with
    downstream PyG HeteroData edge_index tensors.

    Returns:
        feature_dict: Mapping from node_type string ("User", "Role", etc.) to 2D tensor [N_type, D_type]
        node_id_to_idx: Mapping from node_type string to {node_id: integer_index}
    """
    vocab = vocabulary if vocabulary is not None else load_default_action_vocabulary()
    extractor = NodeFeatureExtractor(graph, vocabulary=vocab)
    dim_cfg = extractor.dimension_config

    # 1. Collect nodes partitioned by type matching EnvironmentExporter.to_pyg_ready_dict()
    nodes_by_type: dict[str, list[GraphNode]] = {t.value: [] for t in NodeType}
    for node in graph.get_nodes():
        nodes_by_type[node.node_type.value].append(node)

    # 2. Construct contiguous integer index maps
    node_id_to_idx: dict[str, dict[str, int]] = {}
    for n_type, n_list in nodes_by_type.items():
        node_id_to_idx[n_type] = {node.id: idx for idx, node in enumerate(n_list)}

    # 3. Construct feature tensors
    feature_dict: dict[str, torch.Tensor] = {}

    type_dims = {
        NodeType.USER.value: dim_cfg.user_dim,
        NodeType.ROLE.value: dim_cfg.role_dim,
        NodeType.POLICY.value: dim_cfg.policy_dim,
        NodeType.GROUP.value: dim_cfg.group_dim,
        NodeType.RESOURCE.value: dim_cfg.resource_dim,
    }

    for n_type_str, n_list in nodes_by_type.items():
        expected_dim = type_dims[n_type_str]
        if not n_list:
            # Empty tensor for node types with 0 instances
            feature_dict[n_type_str] = torch.empty((0, expected_dim), dtype=torch.float32)
            continue

        feature_matrix = np.zeros((len(n_list), expected_dim), dtype=np.float32)
        for idx, node in enumerate(n_list):
            feature_matrix[idx] = extractor.extract_node_feature(node)

        feature_dict[n_type_str] = torch.from_numpy(feature_matrix)

    return feature_dict, node_id_to_idx
