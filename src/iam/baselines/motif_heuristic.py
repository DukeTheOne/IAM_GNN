"""Rule-based motif completion heuristic baseline for IAM authorization multigraphs.

Scans partially observed graphs Go for dangling, incomplete, or severed privilege
escalation primitives (e.g. PassRole without role trust, unattached admin policies,
credential minting capabilities), scoring candidate links using domain security heuristics.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from torch_geometric.data import HeteroData

from iam.generator.graph import GraphNode, IAMGraph, NodeType
from iam.models.converter import is_reverse_edge_type


class MotifCompletionHeuristic:
    """Deterministic, rule-based motif completion heuristic.

    Detects characteristic privilege escalation motif signatures where an attacker
    possesses the prerequisite capability (PassRole, CreateAccessKey, AssumeRole, etc.)
    and a dangling candidate relationship to an escalatable target entity is plausible.
    """

    def __init__(self, default_prior: float = 0.05) -> None:
        self.default_prior = default_prior

    def score_pairs(
        self,
        graph_or_data: IAMGraph | HeteroData,
        pairs: list[tuple[str, str]],
    ) -> np.ndarray:
        """Compute heuristic completion confidence scores in [0.0, 1.0] for candidate pairs.

        Args:
            graph_or_data: Partially observed graph Go as IAMGraph or HeteroData.
            pairs: List of candidate (source_id, target_id) pairs.

        Returns:
            np.ndarray of shape (len(pairs),) with heuristic scores in [0.0, 1.0].
        """
        if not pairs:
            return np.zeros(0, dtype=np.float32)

        nx_g, node_map, node_actions = self._extract_graph_and_metadata(graph_or_data)
        undirected_g = nx_g.to_undirected()

        scores = np.zeros(len(pairs), dtype=np.float32)

        for i, (u, v) in enumerate(pairs):
            score = self._compute_pair_score(u, v, nx_g, undirected_g, node_map, node_actions)
            scores[i] = np.clip(score, 0.0, 1.0)

        return scores

    def explain_pair(
        self,
        graph_or_data: IAMGraph | HeteroData,
        source_id: str,
        target_id: str,
    ) -> dict[str, Any]:
        """Provide detailed human-readable security explanation for a specific candidate edge.

        Args:
            graph_or_data: Observed graph Go.
            source_id: Originating principal node ID.
            target_id: Destination target node ID.

        Returns:
            Dictionary detailing matched heuristic rules, evidence, and score components.
        """
        nx_g, node_map, node_actions = self._extract_graph_and_metadata(graph_or_data)
        undirected_g = nx_g.to_undirected()

        matched_rules: list[dict[str, Any]] = []
        u_node = node_map.get(source_id)
        v_node = node_map.get(target_id)
        u_acts = node_actions.get(source_id, set())
        v_acts = node_actions.get(target_id, set())

        score = self.default_prior

        # 1. Dangling PassRole
        has_passrole = (
            any("passrole" in act.lower() for act in u_acts) or "passrole" in source_id.lower()
        )
        if has_passrole and v_node is not None and v_node.node_type == NodeType.ROLE:
            rule_score = 0.85 if (v_node.is_admin or v_node.is_high_value) else 0.65
            if u_node and v_node and u_node.department and u_node.department == v_node.department:
                rule_score += 0.10
            matched_rules.append(
                {
                    "rule": "DanglingPassRoleMotif",
                    "score": rule_score,
                    "reason": f"Principal {source_id} has PassRole capability and target {target_id} is a Role.",
                }
            )
            score = max(score, rule_score)

        # 2. Dangling Credential Minting
        has_minting = (
            any(
                act.lower()
                in ("iam:createaccesskey", "iam:createloginprofile", "iam:updateaccesskey")
                for act in u_acts
            )
            or "key" in source_id.lower()
        )
        if has_minting and v_node is not None and v_node.node_type == NodeType.USER:
            rule_score = 0.85 if (v_node.is_admin or "admin" in target_id.lower()) else 0.60
            if u_node and v_node and u_node.department and u_node.department == v_node.department:
                rule_score += 0.10
            matched_rules.append(
                {
                    "rule": "DanglingCredentialMintingMotif",
                    "score": rule_score,
                    "reason": f"Principal {source_id} has credential creation capabilities targeting user {target_id}.",
                }
            )
            score = max(score, rule_score)

        # 3. Dangling Policy Attachment
        has_attach = (
            any("attach" in act.lower() or "put" in act.lower() for act in u_acts)
            or "policy" in source_id.lower()
        )
        if (
            has_attach
            and v_node is not None
            and v_node.node_type in (NodeType.POLICY, NodeType.ROLE)
        ):
            rule_score = 0.75 if (v_node.is_admin or "admin" in target_id.lower()) else 0.55
            matched_rules.append(
                {
                    "rule": "DanglingPolicyAttachmentMotif",
                    "score": rule_score,
                    "reason": f"Principal {source_id} can attach/modify policies, candidate target is {target_id}.",
                }
            )
            score = max(score, rule_score)

        # 4. Dangling Role Assumption
        has_assume = (
            any("assumerole" in act.lower() for act in u_acts) or "assume" in source_id.lower()
        )
        if has_assume and v_node is not None and v_node.node_type == NodeType.ROLE:
            rule_score = 0.70 if (v_node.is_admin or v_node.is_high_value) else 0.50
            if u_node and v_node and u_node.department and u_node.department == v_node.department:
                rule_score += 0.15
            matched_rules.append(
                {
                    "rule": "DanglingRoleAssumptionMotif",
                    "score": rule_score,
                    "reason": f"Principal {source_id} has AssumeRole privileges towards role {target_id}.",
                }
            )
            score = max(score, rule_score)

        # 5. Department Affinity & Neighborhood Overlap
        if source_id in undirected_g and target_id in undirected_g:
            common = set(undirected_g.neighbors(source_id)) & set(undirected_g.neighbors(target_id))
            if common:
                jaccard = len(common) / (
                    len(
                        set(undirected_g.neighbors(source_id))
                        | set(undirected_g.neighbors(target_id))
                    )
                    + 1e-9
                )
                topol_score = 0.30 + min(0.40, jaccard)
                matched_rules.append(
                    {
                        "rule": "NeighborhoodOverlap",
                        "score": topol_score,
                        "reason": f"Shared {len(common)} common neighbors in observed graph Go.",
                    }
                )
                score = max(score, topol_score)

        return {
            "source_id": source_id,
            "target_id": target_id,
            "final_score": float(np.clip(score, 0.0, 1.0)),
            "matched_rules": matched_rules,
            "source_actions": sorted(u_acts),
            "target_actions": sorted(v_acts),
        }

    def _compute_pair_score(
        self,
        u: str,
        v: str,
        _nx_g: nx.DiGraph,
        undirected_g: nx.Graph,
        node_map: dict[str, GraphNode],
        node_actions: dict[str, set[str]],
    ) -> float:
        """Evaluate rule matching score for a single (u, v) pair."""
        u_node = node_map.get(u)
        v_node = node_map.get(v)
        u_acts = node_actions.get(u, set())

        score = self.default_prior

        # Heuristic 1: PassRole -> Role
        has_passrole = any("passrole" in act.lower() for act in u_acts) or "passrole" in u.lower()
        if has_passrole and v_node is not None and v_node.node_type == NodeType.ROLE:
            h_score = 0.85 if (v_node.is_admin or v_node.is_high_value) else 0.65
            if u_node and u_node.department and u_node.department == v_node.department:
                h_score += 0.10
            score = max(score, h_score)

        # Heuristic 2: Credential Minting -> User
        has_minting = (
            any(
                act.lower()
                in ("iam:createaccesskey", "iam:createloginprofile", "iam:updateaccesskey")
                for act in u_acts
            )
            or "key" in u.lower()
        )
        if has_minting and v_node is not None and v_node.node_type == NodeType.USER:
            h_score = 0.85 if (v_node.is_admin or "admin" in v.lower()) else 0.60
            if u_node and u_node.department and u_node.department == v_node.department:
                h_score += 0.10
            score = max(score, h_score)

        # Heuristic 3: Policy Attachment -> Policy/Role
        has_attach = (
            any("attach" in act.lower() or "put" in act.lower() for act in u_acts)
            or "policy" in u.lower()
        )
        if (
            has_attach
            and v_node is not None
            and v_node.node_type in (NodeType.POLICY, NodeType.ROLE)
        ):
            h_score = 0.75 if (v_node.is_admin or "admin" in v.lower()) else 0.55
            score = max(score, h_score)

        # Heuristic 4: AssumeRole -> Role
        has_assume = any("assumerole" in act.lower() for act in u_acts) or "assume" in u.lower()
        if has_assume and v_node is not None and v_node.node_type == NodeType.ROLE:
            h_score = 0.70 if (v_node.is_admin or v_node.is_high_value) else 0.50
            if u_node and u_node.department and u_node.department == v_node.department:
                h_score += 0.15
            score = max(score, h_score)

        # Heuristic 5: Shared Neighborhood Overlap
        if u in undirected_g and v in undirected_g:
            nbrs_u = set(undirected_g.neighbors(u))
            nbrs_v = set(undirected_g.neighbors(v))
            common = nbrs_u & nbrs_v
            if common:
                jaccard = len(common) / (len(nbrs_u | nbrs_v) + 1e-9)
                topol_score = 0.30 + min(0.40, jaccard)
                score = max(score, topol_score)

        return score

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
