"""Exact Symbolic Reachability Oracle for Cloud IAM Authorization Graphs.

Establishes the theoretical upper-bound detection capability with omniscient
access to unmasked authorization graphs G = (V, E). Evaluates stateful reachability,
multi-hop role chaining, and produces formal proof witnesses for all privilege
escalation (PE) paths.
"""

from __future__ import annotations

import time
from collections import deque

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from iam.generator.graph import EdgeRelation, GraphNode, IAMGraph, NodeType


class OraclePathWitness(BaseModel):
    """Certified reachability and privilege escalation path witness."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_id: str = Field(description="Originating principal ID")
    source_type: NodeType = Field(description="Entity type of source principal")
    source_department: str | None = Field(default=None, description="Department of source")
    source_is_admin: bool = Field(default=False, description="Whether source is admin")
    target_id: str = Field(description="Destination target asset or role ID")
    target_type: NodeType = Field(description="Entity type of target node")
    target_department: str | None = Field(default=None, description="Department of target")
    target_is_high_value: bool = Field(default=False, description="Whether target is high-value")
    target_is_admin: bool = Field(default=False, description="Whether target is admin")
    hop_distance: int = Field(description="Number of relational edge hops along shortest path")
    path_nodes: list[str] = Field(description="Ordered sequence of node IDs from source to target")
    path_relations: list[str] = Field(
        default_factory=list, description="Ordered sequence of edge relations traversed"
    )
    is_privilege_escalation: bool = Field(
        default=False,
        description="True if path elevates privilege from non-admin to admin or high-value",
    )
    is_bridge_traversed: bool = Field(
        default=False,
        description="True if path traverses an adversarial bridge bottleneck relation",
    )
    bridge_relations: list[tuple[str, str, str]] = Field(
        default_factory=list,
        description="List of (src, rel, dst) bridge triples traversed",
    )


class OracleReachabilityResult(BaseModel):
    """Complete evaluation output produced by the Exact Symbolic Oracle."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph_id: str = Field(description="Evaluated graph identifier")
    organization_id: str = Field(default="unknown", description="Organization identifier")
    num_nodes: int = Field(description="Total nodes in evaluated graph")
    num_edges: int = Field(description="Total edges in evaluated graph")
    num_reachable_pairs: int = Field(description="Total count of reachable (source, target) pairs")
    num_pe_pairs: int = Field(description="Total count of privilege escalation pairs")
    reachable_pairs: list[OraclePathWitness] = Field(
        default_factory=list, description="All evaluated reachable path witnesses"
    )
    pe_pairs: list[OraclePathWitness] = Field(
        default_factory=list, description="Subset of witnesses representing privilege escalation"
    )
    computation_time_sec: float = Field(
        default=0.0, description="Execution runtime of oracle evaluation in seconds"
    )

    def is_reachable(self, source_id: str, target_id: str) -> bool:
        """Query if source can reach target."""
        return any(
            w.source_id == source_id and w.target_id == target_id for w in self.reachable_pairs
        )

    def get_witness(self, source_id: str, target_id: str) -> OraclePathWitness | None:
        """Retrieve path witness for a specific (source, target) pair."""
        for w in self.reachable_pairs:
            if w.source_id == source_id and w.target_id == target_id:
                return w
        return None

    def get_targets_for_source(self, source_id: str) -> set[str]:
        """Retrieve set of all target IDs reachable by a given source principal."""
        return {w.target_id for w in self.reachable_pairs if w.source_id == source_id}


class ExactReachabilityOracle:
    """Stateful exact reachability oracle with unmasked omniscient graph access.

    Serves as the theoretical detection ceiling (upper bound) against which
    all heuristic, classical ML, and neural link-prediction models are benchmarked.
    """

    def __init__(
        self,
        max_depth: int = 8,
        only_identities_as_sources: bool = True,
    ) -> None:
        """Initialize the exact oracle.

        Args:
            max_depth: Maximum path traversal search depth (hop limit).
            only_identities_as_sources: When True, only User and Role nodes are evaluated as sources.
        """
        self.max_depth = max_depth
        self.only_identities_as_sources = only_identities_as_sources

    def evaluate(
        self,
        graph: IAMGraph,
        source_ids: list[str] | None = None,
        target_ids: list[str] | None = None,
    ) -> OracleReachabilityResult:
        """Execute exhaustive exact reachability analysis over unmasked IAMGraph.

        Args:
            graph: Omniscient unmasked IAMGraph.
            source_ids: Optional explicit list of source principal IDs to evaluate.
            target_ids: Optional explicit list of target asset or role IDs to evaluate.

        Returns:
            OracleReachabilityResult containing all reachable pairs and PE witnesses.
        """
        start_time = time.perf_counter()
        nx_g = graph.nx_graph

        # 1. Resolve source candidates
        sources: list[GraphNode]
        if source_ids is not None:
            sources = [node for sid in source_ids if (node := graph.get_node(sid)) is not None]
        elif self.only_identities_as_sources:
            users = graph.get_nodes_by_type(NodeType.USER)
            roles = graph.get_nodes_by_type(NodeType.ROLE)
            sources = users + roles
        else:
            sources = graph.get_nodes()

        # 2. Resolve target candidates
        targets: list[GraphNode]
        if target_ids is not None:
            targets = [node for tid in target_ids if (node := graph.get_node(tid)) is not None]
        else:
            hv_targets = graph.get_high_value_targets()
            admin_nodes = graph.get_admin_nodes()
            # Non-policy targets (principals and resources)
            target_map: dict[str, GraphNode] = {
                n.id: n for n in (hv_targets + admin_nodes) if n.node_type != NodeType.POLICY
            }
            targets = list(target_map.values())

        target_id_set = {t.id for t in targets}

        # 3. Fast multi-source breadth-first search
        reachable_witnesses: list[OraclePathWitness] = []
        pe_witnesses: list[OraclePathWitness] = []

        for src in sources:
            if src.id not in nx_g:
                continue

            reachable_paths = self._traverse_from_source(graph, src.id, target_id_set)

            for tgt_id, path_nodes in reachable_paths.items():
                tgt_node = graph.get_node(tgt_id)
                if tgt_node is None:
                    continue

                path_relations, bridge_triples, is_bridge = self._inspect_path_edges(
                    graph, path_nodes
                )

                # Privilege escalation criteria
                reaches_admin = (not src.is_admin) and tgt_node.is_admin
                cross_tier_escalation = (
                    (not src.is_admin)
                    and tgt_node.is_high_value
                    and (tgt_node.department != src.department or tgt_node.department == "SecOps")
                    and any(r in ("AssumesRole", "PassesTo") or is_bridge for r in path_relations)
                )
                is_pe = is_bridge or reaches_admin or cross_tier_escalation

                witness = OraclePathWitness(
                    source_id=src.id,
                    source_type=src.node_type,
                    source_department=src.department,
                    source_is_admin=src.is_admin,
                    target_id=tgt_node.id,
                    target_type=tgt_node.node_type,
                    target_department=tgt_node.department,
                    target_is_high_value=tgt_node.is_high_value,
                    target_is_admin=tgt_node.is_admin,
                    hop_distance=len(path_nodes) - 1,
                    path_nodes=path_nodes,
                    path_relations=path_relations,
                    is_privilege_escalation=is_pe,
                    is_bridge_traversed=is_bridge,
                    bridge_relations=bridge_triples,
                )

                reachable_witnesses.append(witness)
                if is_pe:
                    pe_witnesses.append(witness)

        elapsed = time.perf_counter() - start_time
        org_id = str(graph.metadata.get("organization_id", "unknown"))

        return OracleReachabilityResult(
            graph_id=graph.graph_id,
            organization_id=org_id,
            num_nodes=graph.num_nodes,
            num_edges=graph.num_edges,
            num_reachable_pairs=len(reachable_witnesses),
            num_pe_pairs=len(pe_witnesses),
            reachable_pairs=reachable_witnesses,
            pe_pairs=pe_witnesses,
            computation_time_sec=elapsed,
        )

    def _traverse_from_source(
        self,
        graph: IAMGraph,
        source_id: str,
        target_ids: set[str],
    ) -> dict[str, list[str]]:
        """Traverse graph from source_id up to max_depth using BFS."""
        nx_g = graph.nx_graph
        visited: dict[str, list[str]] = {source_id: [source_id]}
        queue: deque[tuple[str, int]] = deque([(source_id, 0)])
        hits: dict[str, list[str]] = {}

        while queue:
            curr, depth = queue.popleft()

            if curr in target_ids and curr != source_id:
                hits[curr] = visited[curr]

            if depth >= self.max_depth:
                continue

            for neighbor in nx_g.successors(curr):
                if neighbor not in visited:
                    visited[neighbor] = visited[curr] + [neighbor]
                    queue.append((neighbor, depth + 1))

        return hits

    def _inspect_path_edges(
        self,
        graph: IAMGraph,
        path_nodes: list[str],
    ) -> tuple[list[str], list[tuple[str, str, str]], bool]:
        """Extract traversed edge relations and identify adversarial bridge bottleneck relations."""
        nx_g = graph.nx_graph
        relations: list[str] = []
        bridge_triples: list[tuple[str, str, str]] = []
        has_bridge = False

        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[j] if (j := i + 1) < len(path_nodes) else path_nodes[i]
            edge_data_dict = nx_g.get_edge_data(u, v, default={})

            # Check all multigraph parallel edge keys
            best_rel: str = "ConnectedTo"
            for _k, attrs in edge_data_dict.items():
                rel_val = str(attrs.get("relation", "ConnectedTo"))
                best_rel = rel_val
                is_br = bool(attrs.get("is_bridge", False))
                if is_br:
                    has_bridge = True
                    bridge_triples.append((u, rel_val, v))
                    break

            relations.append(best_rel)

        return relations, bridge_triples, has_bridge

    def verify_counterfactual_bridge_severance(
        self,
        graph: IAMGraph,
        witness: OraclePathWitness,
    ) -> bool:
        """Verify counterfactual bridge property: severing bridge relations breaks reachability.

        Returns True if the witness path traversed a bridge and removing all bridge edges
        in that witness severs reachability from source to target.
        """
        if not witness.is_bridge_traversed or not witness.bridge_relations:
            return False

        # Clone graph to test counterfactual severance
        cloned = graph.clone()
        for u, rel_name, v in witness.bridge_relations:
            try:
                rel_enum = EdgeRelation(rel_name)
                cloned.remove_edge(u, v, relation=rel_enum)
            except ValueError:
                continue

        # Check if source can still reach target in cloned graph
        nx_g = cloned.nx_graph
        has_path = nx.has_path(nx_g, witness.source_id, witness.target_id)
        return not has_path
