"""Automated ground-truth labeler and reachability witness engine.

Computes exact multi-hop reachability, extracts path witnesses (ordered intermediate edges),
identifies privilege escalation chains, and annotates critical bottleneck relations.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from iam.generator.graph import (
    EdgeRelation,
    GraphNode,
    IAMGraph,
    NodeType,
)
from iam.generator.motifs import MotifInstance
from iam.generator.topology import BranchingPEChain


class PathEdgeWitness(BaseModel):
    """A directed, typed edge witness traversed along an attack or delegation path."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    relation: EdgeRelation = Field(description="Relational edge type")
    actions: list[str] = Field(default_factory=list, description="Authorized IAM actions")
    is_bridge: bool = Field(default=False, description="Whether this edge is an adversarial bridge")
    motif_id: str | None = Field(default=None, description="PE motif ID if tagged")


class ReachablePairWitness(BaseModel):
    """Complete ground-truth witness for a reachable (source, target) pair."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_id: str = Field(description="Source principal node ID")
    source_type: NodeType = Field(description="Source entity type (User or Role)")
    source_department: str | None = Field(default=None, description="Department of source")
    source_is_admin: bool = Field(description="Whether source possesses administrative privilege")
    target_id: str = Field(description="Target asset or role node ID")
    target_type: NodeType = Field(description="Target entity type")
    target_department: str | None = Field(default=None, description="Department of target")
    target_is_high_value: bool = Field(description="Whether target is a high-value asset")
    target_is_admin: bool = Field(description="Whether target possesses admin privilege")
    is_privilege_escalation: bool = Field(
        description="True if path elevates a non-admin source to an admin/high-value target"
    )
    hop_distance: int = Field(description="Length of shortest path in edge hops")
    path_nodes: list[str] = Field(description="Ordered sequence of node IDs along shortest path")
    path_edges: list[PathEdgeWitness] = Field(
        description="Ordered sequence of intermediate edge witnesses"
    )
    bridge_relations: list[tuple[str, str, str]] = Field(
        default_factory=list, description="Bridge relations traversed along this path"
    )
    all_simple_node_paths: list[list[str]] = Field(
        default_factory=list, description="All simple node paths within depth cutoff"
    )


class GroundTruthLabelSet(BaseModel):
    """Comprehensive ground-truth reachability and privilege escalation dataset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph_id: str = Field(description="Unique graph identifier")
    organization_id: str = Field(default="unknown", description="Organization identifier")
    num_reachable_pairs: int = Field(description="Total count of reachable (s, t) pairs")
    num_pe_pairs: int = Field(description="Count of privilege escalation pairs")
    reachable_pairs: list[ReachablePairWitness] = Field(description="All evaluated reachable pairs")
    pe_pairs: list[ReachablePairWitness] = Field(
        description="Filtered subset of privilege escalation pairs"
    )
    injected_motif_instances: list[MotifInstance] = Field(
        default_factory=list, description="Injected canonical PE motifs"
    )
    branching_chain_instances: list[BranchingPEChain] = Field(
        default_factory=list, description="Injected branching PE chains"
    )
    computation_time_sec: float = Field(description="Execution runtime in seconds")


class GroundTruthLabeler:
    """High-performance ground-truth reachability and path witness labeler."""

    def __init__(
        self,
        max_depth: int = 8,
        find_all_paths: bool = True,
        max_all_paths_limit: int = 10,
        only_identities_as_sources: bool = True,
    ) -> None:
        self.max_depth = max_depth
        self.find_all_paths = find_all_paths
        self.max_all_paths_limit = max_all_paths_limit
        self.only_identities_as_sources = only_identities_as_sources

    def label_graph(
        self,
        graph: IAMGraph,
        target_ids: list[str] | None = None,
        source_ids: list[str] | None = None,
    ) -> GroundTruthLabelSet:
        """Run automated ground-truth labeling over an IAMGraph."""
        start_time = time.perf_counter()
        nx_g = graph.nx_graph

        # 1. Determine candidate sources
        sources: list[GraphNode]
        if source_ids is not None:
            sources = [node for sid in source_ids if (node := graph.get_node(sid)) is not None]
        elif self.only_identities_as_sources:
            users = graph.get_nodes_by_type(NodeType.USER)
            roles = graph.get_nodes_by_type(NodeType.ROLE)
            sources = users + roles
        else:
            sources = list(graph._nodes.values())

        # 2. Determine candidate targets
        targets: list[GraphNode]
        if target_ids is not None:
            targets = [node for tid in target_ids if (node := graph.get_node(tid)) is not None]
        else:
            hv_targets = graph.get_high_value_targets()
            admin_nodes = graph.get_admin_nodes()
            # Unique target assets/principals (excluding intermediate Policy documents)
            target_map = {
                n.id: n for n in (hv_targets + admin_nodes) if n.node_type != NodeType.POLICY
            }
            targets = list(target_map.values())

        target_id_set = {t.id for t in targets}

        # 3. Fast multi-source BFS with depth bounding
        reachable_witnesses: list[ReachablePairWitness] = []
        pe_witnesses: list[ReachablePairWitness] = []

        for src in sources:
            if not src:
                continue
            reachable_for_src = self._bfs_reachability(graph, src.id, target_id_set, self.max_depth)

            for tgt_id, path_nodes in reachable_for_src.items():
                tgt_node = graph.get_node(tgt_id)
                if not tgt_node:
                    continue

                # Build path edges and check bridge relations
                path_edges, bridge_relations = self._extract_edge_witnesses(graph, path_nodes)

                # Privilege escalation check:
                # A path is a true privilege escalation if:
                # 1. It traverses an adversarial bridge relation, OR
                # 2. A non-admin source reaches an administrative principal (role/user), OR
                # 3. A non-admin source reaches a high-value target via delegation (AssumesRole/PassesTo) across trust boundaries.
                is_bridge_traversed = len(bridge_relations) > 0 or any(
                    e.is_bridge for e in path_edges
                )
                reaches_admin = (not src.is_admin) and tgt_node.is_admin
                cross_tier_escalation = (
                    (not src.is_admin)
                    and tgt_node.is_high_value
                    and (tgt_node.department != src.department or tgt_node.department == "SecOps")
                    and any(
                        e.relation in (EdgeRelation.ASSUMES_ROLE, EdgeRelation.PASSES_TO)
                        or e.is_bridge
                        for e in path_edges
                    )
                )
                is_pe = is_bridge_traversed or reaches_admin or cross_tier_escalation

                # Optional: find all alternative simple paths
                all_paths: list[list[str]] = []
                if self.find_all_paths:
                    all_paths = self._find_simple_paths(
                        nx_g, src.id, tgt_id, self.max_depth, self.max_all_paths_limit
                    )
                else:
                    all_paths = [path_nodes]

                witness = ReachablePairWitness(
                    source_id=src.id,
                    source_type=src.node_type,
                    source_department=src.department,
                    source_is_admin=src.is_admin,
                    target_id=tgt_node.id,
                    target_type=tgt_node.node_type,
                    target_department=tgt_node.department,
                    target_is_high_value=tgt_node.is_high_value,
                    target_is_admin=tgt_node.is_admin,
                    is_privilege_escalation=is_pe,
                    hop_distance=len(path_nodes) - 1,
                    path_nodes=path_nodes,
                    path_edges=path_edges,
                    bridge_relations=bridge_relations,
                    all_simple_node_paths=all_paths,
                )

                reachable_witnesses.append(witness)
                if is_pe:
                    pe_witnesses.append(witness)

        elapsed = time.perf_counter() - start_time

        # Retrieve injected metadata if available
        injected_motifs: list[MotifInstance] = []
        branching_chains: list[BranchingPEChain] = []
        if "injected_motifs" in graph.metadata:
            for m_dict in graph.metadata["injected_motifs"]:
                injected_motifs.append(MotifInstance(**m_dict))
        if "branching_chains" in graph.metadata:
            for bc_dict in graph.metadata["branching_chains"]:
                branching_chains.append(BranchingPEChain(**bc_dict))

        org_id = str(graph.metadata.get("organization_id", "unknown"))

        return GroundTruthLabelSet(
            graph_id=graph.graph_id,
            organization_id=org_id,
            num_reachable_pairs=len(reachable_witnesses),
            num_pe_pairs=len(pe_witnesses),
            reachable_pairs=reachable_witnesses,
            pe_pairs=pe_witnesses,
            injected_motif_instances=injected_motifs,
            branching_chain_instances=branching_chains,
            computation_time_sec=elapsed,
        )

    def _bfs_reachability(
        self,
        graph: IAMGraph,
        source_id: str,
        target_ids: set[str],
        max_depth: int,
    ) -> dict[str, list[str]]:
        """Run BFS from source_id up to max_depth, returning shortest paths to hit targets."""
        nx_g = graph.nx_graph
        if source_id not in nx_g:
            return {}

        visited: dict[str, list[str]] = {source_id: [source_id]}
        queue: deque[tuple[str, int]] = deque([(source_id, 0)])
        hits: dict[str, list[str]] = {}

        while queue:
            curr, depth = queue.popleft()

            if curr in target_ids and curr != source_id:
                hits[curr] = visited[curr]

            if depth >= max_depth:
                continue

            for neighbor in nx_g.successors(curr):
                if neighbor not in visited:
                    visited[neighbor] = visited[curr] + [neighbor]
                    queue.append((neighbor, depth + 1))

        return hits

    def _extract_edge_witnesses(
        self,
        graph: IAMGraph,
        path_nodes: list[str],
    ) -> tuple[list[PathEdgeWitness], list[tuple[str, str, str]]]:
        """Extract ordered GraphEdge objects and bridge relations along path_nodes."""
        nx_g = graph.nx_graph
        path_edges: list[PathEdgeWitness] = []
        bridges: list[tuple[str, str, str]] = []

        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i + 1]

            edge_dict = nx_g.get_edge_data(u, v)
            if not edge_dict:
                continue

            # Pick the most relevant relation key (prioritizing bridge if present)
            chosen_key: str | None = None
            chosen_data: dict[str, Any] = {}
            for k, data in edge_dict.items():
                if data.get("is_bridge", False):
                    chosen_key = str(k)
                    chosen_data = data
                    break

            if chosen_key is None:
                chosen_key, chosen_data = next(iter(edge_dict.items()))

            rel_enum = EdgeRelation(chosen_key)
            is_br = bool(chosen_data.get("is_bridge", False))
            m_id = chosen_data.get("motif_id")
            acts = list(chosen_data.get("actions", []))

            witness = PathEdgeWitness(
                source=u,
                target=v,
                relation=rel_enum,
                actions=acts,
                is_bridge=is_br,
                motif_id=m_id,
            )
            path_edges.append(witness)

            if is_br:
                bridges.append((u, v, rel_enum.value))

        return path_edges, bridges

    def _find_simple_paths(
        self,
        nx_g: nx.MultiDiGraph,
        source: str,
        target: str,
        cutoff: int,
        limit: int,
    ) -> list[list[str]]:
        """Find up to limit simple node paths from source to target within cutoff hops."""
        paths: list[list[str]] = []
        try:
            for p in nx.all_simple_paths(nx_g, source, target, cutoff=cutoff):
                paths.append(list(p))
                if len(paths) >= limit:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        return paths
