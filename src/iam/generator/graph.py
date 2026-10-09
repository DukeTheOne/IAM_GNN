"""Heterogeneous graph data structures for IAM topologies and enterprise access control.

Provides typed nodes, multi-relational edges, and an optimized IAMGraph wrapper
backed by NetworkX for efficient structural queries and graph algorithms.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from iam.parser.schema import PolicyDocument


class NodeType(str, Enum):
    """Canonical node entity types in the cloud authorization multigraph."""

    USER = "User"
    ROLE = "Role"
    POLICY = "Policy"
    GROUP = "Group"
    RESOURCE = "Resource"


class EdgeRelation(str, Enum):
    """Canonical typed edge relations in the cloud authorization multigraph."""

    MEMBER_OF = "MemberOf"  # User -> Group
    ASSUMES_ROLE = "AssumesRole"  # Principal -> Role
    ATTACHED_WITH = "AttachedWith"  # Principal/Group -> Policy
    ACTS_ON = "ActsOn"  # Policy -> Resource, or Principal -> Resource/Principal
    PASSES_TO = "PassesTo"  # Principal/Role -> Role


class GraphNode(BaseModel):
    """A typed node entity within the IAM authorization multigraph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(description="Unique node identifier within the graph")
    node_type: NodeType = Field(description="Entity type")
    arn: str = Field(description="Canonical AWS ARN or synthetic entity ARN")
    name: str = Field(description="Human-readable resource or principal name")
    account_id: str = Field(default="123456789012", description="AWS 12-digit account ID")
    department: str | None = Field(default=None, description="Organizational unit/department")
    is_high_value: bool = Field(default=False, description="Whether node is a high-value asset")
    is_admin: bool = Field(default=False, description="Whether identity possesses admin privileges")
    policy_document: PolicyDocument | None = Field(
        default=None, description="Parsed AST policy for Policy nodes"
    )
    trust_policy: PolicyDocument | None = Field(
        default=None, description="AssumeRole trust policy document for Role nodes"
    )
    tags: dict[str, str] = Field(default_factory=dict, description="Resource tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary node metadata")


class GraphEdge(BaseModel):
    """A directed, typed relationship between two entities in the authorization graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    relation: EdgeRelation = Field(description="Relational edge type")
    is_bridge: bool = Field(
        default=False, description="Whether this edge is a critical PE bridge for Condition E"
    )
    motif_id: str | None = Field(
        default=None, description="Identifier of the PE motif instance if applicable"
    )
    actions: list[str] = Field(
        default_factory=list, description="Authorized actions represented by this edge"
    )
    condition: dict[str, Any] | None = Field(
        default=None, description="Optional condition constraints"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary edge metadata")

    @property
    def key(self) -> tuple[str, str, str]:
        """Triple key uniquely identifying this directed multi-edge."""
        return (self.source, self.target, self.relation.value)


class IAMGraph:
    """Optimized heterogeneous authorization multigraph backed by NetworkX.

    Maintains typed node indices and adjacency indexes for high-throughput queries,
    motif injection, and downstream PyG HeteroData conversion.
    """

    def __init__(
        self,
        graph_id: str = "default_iam_graph",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.graph_id = graph_id
        self.metadata: dict[str, Any] = metadata if metadata is not None else {}
        self._nx_graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._nodes: dict[str, GraphNode] = {}
        self._arn_to_id: dict[str, str] = {}
        self._nodes_by_type: dict[NodeType, set[str]] = {t: set() for t in NodeType}

    @property
    def nx_graph(self) -> nx.MultiDiGraph:
        """Underlying NetworkX MultiDiGraph instance."""
        return self._nx_graph

    @property
    def num_nodes(self) -> int:
        """Total node count."""
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        """Total edge count across all relations."""
        return int(self._nx_graph.number_of_edges())

    def add_node(self, node: GraphNode) -> None:
        """Add a typed node to the graph and update internal indexes."""
        self._nodes[node.id] = node
        self._arn_to_id[node.arn] = node.id
        self._nodes_by_type[node.node_type].add(node.id)

        self._nx_graph.add_node(
            node.id,
            node_type=node.node_type.value,
            arn=node.arn,
            name=node.name,
            account_id=node.account_id,
            department=node.department,
            is_high_value=node.is_high_value,
            is_admin=node.is_admin,
            tags=node.tags,
            obj=node,
        )

    def add_nodes(self, nodes: list[GraphNode]) -> None:
        """Add multiple typed nodes in batch."""
        for node in nodes:
            self.add_node(node)

    def add_edge(self, edge: GraphEdge) -> None:
        """Add a directed typed edge between two existing nodes."""
        if edge.source not in self._nodes:
            raise KeyError(f"Source node '{edge.source}' does not exist in graph.")
        if edge.target not in self._nodes:
            raise KeyError(f"Target node '{edge.target}' does not exist in graph.")

        self._nx_graph.add_edge(
            edge.source,
            edge.target,
            key=edge.relation.value,
            relation=edge.relation.value,
            is_bridge=edge.is_bridge,
            motif_id=edge.motif_id,
            actions=edge.actions,
            condition=edge.condition,
            metadata=edge.metadata,
            obj=edge,
        )

    def add_edges(self, edges: list[GraphEdge]) -> None:
        """Add multiple directed typed edges in batch."""
        for edge in edges:
            self.add_edge(edge)

    def remove_edge(self, source: str, target: str, relation: EdgeRelation) -> bool:
        """Remove an edge matching source, target, and relation."""
        if self._nx_graph.has_edge(source, target, key=relation.value):
            self._nx_graph.remove_edge(source, target, key=relation.value)
            return True
        return False

    def get_node(self, node_id: str) -> GraphNode | None:
        """Retrieve node by unique ID."""
        return self._nodes.get(node_id)

    def get_node_by_arn(self, arn: str) -> GraphNode | None:
        """Retrieve node by AWS ARN."""
        node_id = self._arn_to_id.get(arn)
        if node_id:
            return self._nodes.get(node_id)
        return None

    def get_nodes_by_type(self, node_type: NodeType) -> list[GraphNode]:
        """Retrieve all nodes of a specific NodeType."""
        return [self._nodes[nid] for nid in self._nodes_by_type[node_type]]

    def get_nodes(self) -> list[GraphNode]:
        """Retrieve all nodes in the graph in insertion order."""
        return list(self._nodes.values())

    def get_node_ids(self) -> list[str]:
        """Retrieve all node IDs in the graph in insertion order."""
        return list(self._nodes.keys())

    def get_high_value_targets(self) -> list[GraphNode]:
        """Retrieve all nodes flagged as high-value assets."""
        return [node for node in self._nodes.values() if node.is_high_value]

    def get_admin_nodes(self) -> list[GraphNode]:
        """Retrieve all identities with administrative access."""
        return [node for node in self._nodes.values() if node.is_admin]

    def get_edges(
        self,
        source: str | None = None,
        target: str | None = None,
        relation: EdgeRelation | None = None,
    ) -> list[GraphEdge]:
        """Query edges with optional filters."""
        edges: list[GraphEdge] = []
        for u, v, key, data in self._nx_graph.edges(keys=True, data=True):
            if source is not None and u != source:
                continue
            if target is not None and v != target:
                continue
            if relation is not None and key != relation.value:
                continue
            edges.append(data["obj"])
        return edges

    def get_bridge_edges(self) -> list[GraphEdge]:
        """Retrieve all edges marked as critical attack bridges (for Condition E)."""
        return [
            data["obj"]
            for _, _, _, data in self._nx_graph.edges(keys=True, data=True)
            if data.get("is_bridge", False)
        ]

    def has_edge(self, source: str, target: str, relation: EdgeRelation) -> bool:
        """Check if a specific directed typed edge exists."""
        return bool(self._nx_graph.has_edge(source, target, key=relation.value))

    def get_ego_graph(self, center_node_id: str, radius: int = 2) -> IAMGraph:
        """Extract an ego subgraph centered on center_node_id up to radius hops."""
        undirected = self._nx_graph.to_undirected(as_view=True)
        sub_nodes = set(nx.ego_graph(undirected, center_node_id, radius=radius).nodes)

        sub_iam = IAMGraph(graph_id=f"ego_{center_node_id}_r{radius}")
        for nid in sub_nodes:
            sub_iam.add_node(self._nodes[nid])

        for u, v, _key, data in self._nx_graph.edges(keys=True, data=True):
            if u in sub_nodes and v in sub_nodes:
                sub_iam.add_edge(data["obj"])

        return sub_iam

    def clone(self) -> IAMGraph:
        """Perform a deep clone of the entire IAM graph."""
        cloned = IAMGraph(
            graph_id=f"{self.graph_id}_clone",
            metadata=dict(self.metadata),
        )
        for node in self._nodes.values():
            cloned.add_node(node.model_copy(deep=True))
        for edge in self.get_edges():
            cloned.add_edge(edge.model_copy(deep=True))
        return cloned

    def to_dict(self) -> dict[str, Any]:
        """Serialize IAMGraph to a JSON-compatible dictionary."""
        nodes_list: list[dict[str, Any]] = []
        for node in self._nodes.values():
            n_dict = node.model_dump()
            if node.policy_document:
                n_dict["policy_document"] = node.policy_document.to_dict()
            if node.trust_policy:
                n_dict["trust_policy"] = node.trust_policy.to_dict()
            nodes_list.append(n_dict)

        edges_list: list[dict[str, Any]] = [edge.model_dump() for edge in self.get_edges()]

        return {
            "graph_id": self.graph_id,
            "metadata": self.metadata,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "nodes": nodes_list,
            "edges": edges_list,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IAMGraph:
        """Deserialize an IAMGraph from a dictionary."""
        graph = cls(
            graph_id=data.get("graph_id", "imported_graph"),
            metadata=data.get("metadata", {}),
        )

        for n_dict in data.get("nodes", []):
            policy_doc = None
            if n_dict.get("policy_document"):
                policy_doc = PolicyDocument.from_dict(n_dict["policy_document"])
            trust_doc = None
            if n_dict.get("trust_policy"):
                trust_doc = PolicyDocument.from_dict(n_dict["trust_policy"])

            n_dict_clean = {
                k: v for k, v in n_dict.items() if k not in ("policy_document", "trust_policy")
            }
            node = GraphNode(
                **n_dict_clean,
                policy_document=policy_doc,
                trust_policy=trust_doc,
            )
            graph.add_node(node)

        for e_dict in data.get("edges", []):
            edge = GraphEdge(**e_dict)
            graph.add_edge(edge)

        return graph
