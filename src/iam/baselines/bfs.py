"""Deterministic BFS Reachability Engine (PMapper / BloodHound Proxy).

Simulates stateful deterministic graph traversal over partially observed
authorization multigraphs G_o. Operates strictly without access to hidden
ground-truth edges H, enforcing cycle avoidance, role-chaining transitions,
and depth bounding (k <= 6).
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field
from torch_geometric.data import HeteroData

from iam.baselines.oracle import OraclePathWitness
from iam.generator.graph import GraphNode, IAMGraph, NodeType
from iam.models.converter import is_reverse_edge_type


class BFSPathWitness(BaseModel):
    """Certified reachability witness produced by deterministic BFS traversal."""

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
    hop_distance: int = Field(description="Number of edge hops along shortest observed path")
    path_nodes: list[str] = Field(description="Ordered sequence of node IDs in observed path")
    path_relations: list[str] = Field(
        default_factory=list, description="Ordered sequence of edge relations traversed"
    )
    is_privilege_escalation: bool = Field(
        default=False,
        description="True if path elevates privilege from non-admin to admin or high-value",
    )


class BFSReachabilityResult(BaseModel):
    """Complete evaluation output produced by DeterministicBFSEngine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph_id: str = Field(description="Evaluated graph identifier")
    num_nodes: int = Field(description="Total nodes in evaluated observed graph")
    num_edges: int = Field(description="Total edges in evaluated observed graph")
    num_reachable_pairs: int = Field(description="Total count of reachable (source, target) pairs")
    num_pe_pairs: int = Field(description="Total count of privilege escalation pairs")
    reachable_pairs: list[BFSPathWitness] = Field(
        default_factory=list, description="All evaluated reachable path witnesses"
    )
    pe_pairs: list[BFSPathWitness] = Field(
        default_factory=list, description="Subset of witnesses representing privilege escalation"
    )
    computation_time_sec: float = Field(
        default=0.0, description="Execution runtime of BFS traversal in seconds"
    )

    def is_reachable(self, source_id: str, target_id: str) -> bool:
        """Query if source can reach target in observed graph."""
        return any(
            w.source_id == source_id and w.target_id == target_id for w in self.reachable_pairs
        )

    def get_witness(self, source_id: str, target_id: str) -> BFSPathWitness | None:
        """Retrieve path witness for a specific (source, target) pair."""
        for w in self.reachable_pairs:
            if w.source_id == source_id and w.target_id == target_id:
                return w
        return None

    def get_targets_for_source(self, source_id: str) -> set[str]:
        """Retrieve set of all target IDs reachable by a given source principal."""
        return {w.target_id for w in self.reachable_pairs if w.source_id == source_id}


class DeterministicBFSEngine:
    """Stateful BFS traversal engine modeling PMapper / BloodHound over observed Go.

    Strictly traverses only visible edges in the observed graph Go.
    Under Condition E (adversarial bridge masking), critical attack paths
    are severed, causing this engine to fail by construction.
    """

    def __init__(
        self,
        max_depth: int = 6,
        only_identities_as_sources: bool = True,
    ) -> None:
        """Initialize the deterministic BFS engine.

        Args:
            max_depth: Maximum path traversal search depth (hop cutoff k <= 6).
            only_identities_as_sources: When True, only User and Role nodes are evaluated as sources.
        """
        self.max_depth = max_depth
        self.only_identities_as_sources = only_identities_as_sources

    def evaluate(
        self,
        graph_or_data: IAMGraph | HeteroData,
        source_ids: list[str] | None = None,
        target_ids: list[str] | None = None,
    ) -> BFSReachabilityResult:
        """Execute deterministic BFS traversal over observed graph or HeteroData Go.

        Args:
            graph_or_data: Observed graph as IAMGraph or PyG HeteroData.
            source_ids: Optional explicit list of source principal IDs to evaluate.
            target_ids: Optional explicit list of target asset or role IDs to evaluate.

        Returns:
            BFSReachabilityResult containing all reachable pairs and PE witnesses.
        """
        start_time = time.perf_counter()
        node_resolver: Callable[[str], GraphNode | None]

        if isinstance(graph_or_data, IAMGraph):
            nx_g = graph_or_data.nx_graph
            graph_id = graph_or_data.graph_id
            num_nodes = graph_or_data.num_nodes
            num_edges = graph_or_data.num_edges
            node_resolver = graph_or_data.get_node

            if source_ids is not None:
                sources = [
                    node for sid in source_ids if (node := graph_or_data.get_node(sid)) is not None
                ]
            elif self.only_identities_as_sources:
                sources = graph_or_data.get_nodes_by_type(
                    NodeType.USER
                ) + graph_or_data.get_nodes_by_type(NodeType.ROLE)
            else:
                sources = graph_or_data.get_nodes()

            if target_ids is not None:
                targets = [
                    node for tid in target_ids if (node := graph_or_data.get_node(tid)) is not None
                ]
            else:
                hv_targets = graph_or_data.get_high_value_targets()
                admin_nodes = graph_or_data.get_admin_nodes()
                target_map: dict[str, GraphNode] = {
                    n.id: n for n in (hv_targets + admin_nodes) if n.node_type != NodeType.POLICY
                }
                targets = list(target_map.values())

        else:
            nx_g, node_dict = self._build_digraph_from_hetero(graph_or_data)
            graph_id = getattr(graph_or_data, "graph_id", "hetero_go")
            num_nodes = nx_g.number_of_nodes()
            num_edges = nx_g.number_of_edges()

            def get_hetero_node(k: str) -> GraphNode | None:
                return node_dict.get(k)

            node_resolver = get_hetero_node

            if source_ids is not None:
                sources = [node for sid in source_ids if (node := node_dict.get(sid)) is not None]
            elif self.only_identities_as_sources:
                sources = [
                    n for n in node_dict.values() if n.node_type in (NodeType.USER, NodeType.ROLE)
                ]
            else:
                sources = list(node_dict.values())

            if target_ids is not None:
                targets = [node for tid in target_ids if (node := node_dict.get(tid)) is not None]
            else:
                targets = [
                    n
                    for n in node_dict.values()
                    if (n.is_high_value or n.is_admin) and n.node_type != NodeType.POLICY
                ]

        target_id_set = {t.id for t in targets}
        reachable_witnesses: list[BFSPathWitness] = []
        pe_witnesses: list[BFSPathWitness] = []

        for src in sources:
            if src.id not in nx_g:
                continue

            reachable_paths = self._bfs_traverse(nx_g, src.id, target_id_set)

            for tgt_id, path_nodes in reachable_paths.items():
                tgt_node = node_resolver(tgt_id)
                if tgt_node is None:
                    continue

                path_relations = self._extract_path_relations(nx_g, path_nodes)

                reaches_admin = (not src.is_admin) and tgt_node.is_admin
                cross_tier_escalation = (
                    (not src.is_admin)
                    and tgt_node.is_high_value
                    and (tgt_node.department != src.department or tgt_node.department == "SecOps")
                    and any(r in ("AssumesRole", "PassesTo") for r in path_relations)
                )
                is_pe = reaches_admin or cross_tier_escalation

                witness = BFSPathWitness(
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
                )

                reachable_witnesses.append(witness)
                if is_pe:
                    pe_witnesses.append(witness)

        elapsed = time.perf_counter() - start_time

        return BFSReachabilityResult(
            graph_id=graph_id,
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_reachable_pairs=len(reachable_witnesses),
            num_pe_pairs=len(pe_witnesses),
            reachable_pairs=reachable_witnesses,
            pe_pairs=pe_witnesses,
            computation_time_sec=elapsed,
        )

    def _bfs_traverse(
        self,
        nx_g: nx.DiGraph | nx.MultiDiGraph,
        source_id: str,
        target_ids: set[str],
    ) -> dict[str, list[str]]:
        """Run breadth-first search from source_id up to max_depth."""
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

    def _extract_path_relations(
        self,
        nx_g: nx.DiGraph | nx.MultiDiGraph,
        path_nodes: list[str],
    ) -> list[str]:
        """Extract ordered sequence of edge relations along path_nodes."""
        relations: list[str] = []
        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i + 1]
            if isinstance(nx_g, nx.MultiDiGraph):
                edge_dict = nx_g.get_edge_data(u, v, default={})
                best_rel = "ConnectedTo"
                for _k, attrs in edge_dict.items():
                    best_rel = str(attrs.get("relation", "ConnectedTo"))
                    break
                relations.append(best_rel)
            else:
                data = nx_g.get_edge_data(u, v, default={})
                relations.append(str(data.get("relation", "ConnectedTo")))
        return relations

    def _build_digraph_from_hetero(
        self,
        data: HeteroData,
    ) -> tuple[nx.DiGraph, dict[str, GraphNode]]:
        """Construct directed graph from PyG HeteroData Go, strictly excluding reverse edges."""
        nx_g = nx.DiGraph()
        node_map: dict[str, GraphNode] = {}

        for node_type in data.node_types:
            node_ids: list[str] = getattr(data[node_type], "node_ids", [])
            arns: list[str] = getattr(data[node_type], "arns", [])
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
                nx_g.add_node(nid, node_type=node_type, arn=arn)

        for edge_type in data.edge_types:
            if is_reverse_edge_type(edge_type):
                continue

            src_type, rel, dst_type = edge_type
            src_ids = getattr(data[src_type], "node_ids", [])
            dst_ids = getattr(data[dst_type], "node_ids", [])
            edge_index = data[edge_type].edge_index

            for i in range(edge_index.size(1)):
                u = src_ids[int(edge_index[0, i])]
                v = dst_ids[int(edge_index[1, i])]
                nx_g.add_edge(u, v, relation=rel)

        return nx_g, node_map

    @staticmethod
    def compute_oracle_recall(
        bfs_result: BFSReachabilityResult,
        oracle_pe_pairs: list[OraclePathWitness],
    ) -> float:
        """Compute recall of BFS on true privilege escalation paths identified by the oracle.

        Returns a value in [0.0, 1.0]. Under Condition E (Adversarial Bridge Masking),
        this recall formally drops to 0.0.
        """
        if not oracle_pe_pairs:
            return 1.0

        detected = 0
        for oracle_pe in oracle_pe_pairs:
            if bfs_result.is_reachable(oracle_pe.source_id, oracle_pe.target_id):
                detected += 1

        return float(detected / len(oracle_pe_pairs))
