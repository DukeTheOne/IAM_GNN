"""Topological sanity checks and structural metrics for enterprise cloud graphs.

Computes comprehensive graph-theoretic metrics including degree distributions,
clustering coefficients, strongly and weakly connected components, and authorization
health indicators.
"""

from __future__ import annotations

import statistics

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from iam.generator.graph import (
    EdgeRelation,
    IAMGraph,
    NodeType,
)


class DistributionStats(BaseModel):
    """Statistical summary of a numerical distribution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    min: float = Field(description="Minimum value")
    max: float = Field(description="Maximum value")
    mean: float = Field(description="Arithmetic mean")
    median: float = Field(description="Median value")
    std: float = Field(description="Standard deviation")

    @classmethod
    def from_values(cls, values: list[int] | list[float]) -> DistributionStats:
        """Construct distribution statistics from a list of numerical values."""
        if not values:
            return cls(min=0.0, max=0.0, mean=0.0, median=0.0, std=0.0)
        vals = [float(v) for v in values]
        mean_v = statistics.mean(vals)
        med_v = statistics.median(vals)
        std_v = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return cls(
            min=round(min(vals), 3),
            max=round(max(vals), 3),
            mean=round(mean_v, 3),
            median=round(med_v, 3),
            std=round(std_v, 3),
        )


class DegreeMetrics(BaseModel):
    """Graph degree distribution summaries."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    in_degree: DistributionStats = Field(description="In-degree distribution")
    out_degree: DistributionStats = Field(description="Out-degree distribution")
    relation_degrees: dict[str, DistributionStats] = Field(
        default_factory=dict, description="Out-degree distributions per EdgeRelation"
    )


class ConnectivityMetrics(BaseModel):
    """Graph connectivity and component metrics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_strongly_connected_components: int = Field(description="Count of SCCs")
    largest_scc_size: int = Field(description="Node count in largest SCC")
    num_weakly_connected_components: int = Field(description="Count of WCCs")
    largest_wcc_size: int = Field(description="Node count in largest WCC (giant component)")
    giant_component_ratio: float = Field(
        description="Fraction of graph nodes belonging to giant component"
    )
    num_isolated_nodes: int = Field(description="Count of nodes with zero degree")


class ClusteringMetrics(BaseModel):
    """Clustering, reciprocity, and density metrics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    average_clustering_coefficient: float = Field(
        description="Average clustering coefficient of underlying graph"
    )
    transitivity: float = Field(description="Global transitivity (fraction of closed triplets)")
    edge_density: float = Field(description="Directed edge density m / (n * (n - 1))")
    reciprocity: float = Field(description="Fraction of reciprocal directed edges")


class TopologicalMetricsReport(BaseModel):
    """Comprehensive structural health and metrics report for an authorization graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph_id: str = Field(description="Graph identifier")
    total_nodes: int = Field(description="Total node count")
    total_edges: int = Field(description="Total edge count across all relations")
    average_degree: float = Field(description="Average node degree (edges / nodes)")
    degrees: DegreeMetrics = Field(description="In-degree and out-degree statistics")
    connectivity: ConnectivityMetrics = Field(description="Component metrics")
    clustering: ClusteringMetrics = Field(description="Clustering and density metrics")
    node_type_counts: dict[str, int] = Field(description="Counts per NodeType")
    edge_relation_counts: dict[str, int] = Field(description="Counts per EdgeRelation")
    department_distribution: dict[str, int] = Field(description="Counts per department")
    admin_identities_count: int = Field(description="Identities possessing admin privileges")
    admin_percentage: float = Field(description="Percentage of identities that are admin")
    high_value_targets_count: int = Field(description="Total high-value assets")
    bridge_edges_count: int = Field(description="Edges flagged as critical bridges")
    is_healthy: bool = Field(description="True if all sanity assertions pass")
    sanity_warnings: list[str] = Field(
        default_factory=list, description="Sanity and validation diagnostics"
    )


class TopologicalSanityChecker:
    """Computes graph-theoretic metrics and performs structural validation checks."""

    def compute_metrics(self, graph: IAMGraph) -> TopologicalMetricsReport:
        """Compute exhaustive structural and authorization metrics for the graph."""
        nx_g = graph.nx_graph
        num_n = graph.num_nodes
        num_e = graph.num_edges

        # 1. Degree Distributions
        in_degrees = [d for _, d in nx_g.in_degree()]
        out_degrees = [d for _, d in nx_g.out_degree()]
        in_stat = DistributionStats.from_values(in_degrees)
        out_stat = DistributionStats.from_values(out_degrees)

        # Per-relation degrees
        rel_degrees: dict[str, DistributionStats] = {}
        for rel in EdgeRelation:
            # Subgraph for specific relation
            rel_edges = [(u, v) for u, v, k in nx_g.edges(keys=True) if k == rel.value]
            rel_g = nx.DiGraph()
            rel_g.add_nodes_from(nx_g.nodes)
            rel_g.add_edges_from(rel_edges)
            rel_out = [d for _, d in rel_g.out_degree()]
            rel_degrees[rel.value] = DistributionStats.from_values(rel_out)

        degrees = DegreeMetrics(
            in_degree=in_stat,
            out_degree=out_stat,
            relation_degrees=rel_degrees,
        )

        # 2. Connectivity Metrics
        sccs = list(nx.strongly_connected_components(nx_g))
        wccs = list(nx.weakly_connected_components(nx_g))
        largest_scc = max((len(c) for c in sccs), default=0)
        largest_wcc = max((len(c) for c in wccs), default=0)
        wcc_ratio = (largest_wcc / num_n) if num_n > 0 else 0.0

        isolated_count = sum(1 for _, d in nx_g.degree() if d == 0)

        connectivity = ConnectivityMetrics(
            num_strongly_connected_components=len(sccs),
            largest_scc_size=largest_scc,
            num_weakly_connected_components=len(wccs),
            largest_wcc_size=largest_wcc,
            giant_component_ratio=round(wcc_ratio, 3),
            num_isolated_nodes=isolated_count,
        )

        # 3. Clustering and Density Metrics
        simple_undirected = nx.Graph(nx_g)
        avg_clustering = float(nx.average_clustering(simple_undirected)) if num_n > 0 else 0.0
        transitivity_val = float(nx.transitivity(simple_undirected)) if num_n > 0 else 0.0
        edge_density = float(nx.density(nx_g))
        reciprocity_val = float(nx.reciprocity(nx_g)) if num_e > 0 else 0.0

        clustering = ClusteringMetrics(
            average_clustering_coefficient=round(avg_clustering, 4),
            transitivity=round(transitivity_val, 4),
            edge_density=round(edge_density, 5),
            reciprocity=round(reciprocity_val, 4),
        )

        # 4. Identity and Authorization Counts
        type_counts: dict[str, int] = {t.value: len(graph.get_nodes_by_type(t)) for t in NodeType}
        rel_counts: dict[str, int] = {rel.value: 0 for rel in EdgeRelation}
        for edge in graph.get_edges():
            rel_counts[edge.relation.value] += 1

        dept_dist: dict[str, int] = {}
        for node in graph._nodes.values():
            d = node.department or "Unknown"
            dept_dist[d] = dept_dist.get(d, 0) + 1

        admin_count = len(graph.get_admin_nodes())
        total_identities = type_counts[NodeType.USER.value] + type_counts[NodeType.ROLE.value]
        admin_pct = (admin_count / total_identities * 100.0) if total_identities > 0 else 0.0
        hv_count = len(graph.get_high_value_targets())
        bridge_count = len(graph.get_bridge_edges())

        # 5. Sanity Checks and Health Diagnostics
        warnings: list[str] = []
        is_healthy = True

        if num_n == 0:
            warnings.append("Graph contains zero nodes.")
            is_healthy = False
        if num_e == 0:
            warnings.append("Graph contains zero edges.")
            is_healthy = False
        if isolated_count > (num_n * 0.10):
            warnings.append(
                f"High number of isolated nodes: {isolated_count} ({isolated_count / num_n * 100:.1f}%)"
            )
            is_healthy = False
        if wcc_ratio < 0.70 and num_n >= 100:
            warnings.append(f"Giant component is fragmented; ratio is only {wcc_ratio * 100:.1f}%")
            is_healthy = False
        if admin_pct > 25.0:
            warnings.append(
                f"Admin percentage {admin_pct:.1f}% violates power-law hierarchy expectation (< 25%)"
            )
            is_healthy = False
        if hv_count == 0:
            warnings.append("Graph has no high-value targets designated.")
            is_healthy = False

        avg_deg = round(num_e / max(1, num_n), 2)

        return TopologicalMetricsReport(
            graph_id=graph.graph_id,
            total_nodes=num_n,
            total_edges=num_e,
            average_degree=avg_deg,
            degrees=degrees,
            connectivity=connectivity,
            clustering=clustering,
            node_type_counts=type_counts,
            edge_relation_counts=rel_counts,
            department_distribution=dept_dist,
            admin_identities_count=admin_count,
            admin_percentage=round(admin_pct, 2),
            high_value_targets_count=hv_count,
            bridge_edges_count=bridge_count,
            is_healthy=is_healthy,
            sanity_warnings=warnings,
        )
