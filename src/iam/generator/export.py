"""Scalable export and serialization pipeline for IAM authorization graphs.

Provides memory-efficient serialization into structured JSON, NetworkX node-link schemas,
and PyG-ready tensors/indices, with optional GZIP compression and native orjson acceleration.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import networkx as nx

from iam.generator.graph import (
    EdgeRelation,
    GraphEdge,
    GraphNode,
    IAMGraph,
    NodeType,
)
from iam.generator.labeler import GroundTruthLabelSet
from iam.generator.metrics import TopologicalMetricsReport

try:
    import orjson

    _HAS_ORJSON = True
except ImportError:
    _HAS_ORJSON = False


class EnvironmentExporter:
    """Scalable serialization engine for cloud authorization environments."""

    def __init__(self, use_orjson: bool = True) -> None:
        self.use_orjson = use_orjson and _HAS_ORJSON

    def to_structured_dict(
        self,
        graph: IAMGraph,
        label_set: GroundTruthLabelSet | None = None,
        metrics_report: TopologicalMetricsReport | None = None,
    ) -> dict[str, Any]:
        """Convert graph, ground-truth labels, and metrics into a complete JSON-serializable dict."""
        graph_dict = graph.to_dict()

        payload: dict[str, Any] = {
            "schema_version": "1.0.0",
            "graph": graph_dict,
            "organization_id": graph.metadata.get("organization_id", "unknown"),
            "account_id": graph.metadata.get("account_id", "123456789012"),
            "metadata": graph.metadata,
        }

        if label_set is not None:
            payload["ground_truth"] = label_set.model_dump()

        if metrics_report is not None:
            payload["topological_metrics"] = metrics_report.model_dump()

        return payload

    def to_networkx_node_link_dict(self, graph: IAMGraph) -> dict[str, Any]:
        """Convert IAMGraph to NetworkX canonical node-link data representation."""
        nx_g = graph.nx_graph
        # Exclude internal non-serializable python object references
        clean_g = nx.MultiDiGraph(graph_id=graph.graph_id)
        for nid, data in nx_g.nodes(data=True):
            c_data = {k: v for k, v in data.items() if k != "obj"}
            clean_g.add_node(nid, **c_data)

        for u, v, k, data in nx_g.edges(keys=True, data=True):
            c_data = {k_e: v_e for k_e, v_e in data.items() if k_e != "obj"}
            clean_g.add_edge(u, v, key=k, **c_data)

        return dict(nx.node_link_data(clean_g))

    def to_pyg_ready_dict(
        self,
        graph: IAMGraph,
        label_set: GroundTruthLabelSet | None = None,
    ) -> dict[str, Any]:
        """Convert graph into PyG-compatible index dictionaries for HeteroData conversion."""
        # 1. Map node IDs to integer indices per NodeType
        nodes_by_type: dict[str, list[str]] = {t.value: [] for t in NodeType}
        for node in graph._nodes.values():
            nodes_by_type[node.node_type.value].append(node.id)

        node_id_to_idx: dict[str, dict[str, int]] = {}
        for n_type, n_ids in nodes_by_type.items():
            node_id_to_idx[n_type] = {nid: i for i, nid in enumerate(n_ids)}

        # 2. Build edge index dictionaries per canonical relation: (src_type, rel, dst_type)
        edge_indices: dict[str, list[list[int]]] = {}
        bridge_indices: dict[str, list[int]] = {}

        for edge in graph.get_edges():
            src_node = graph.get_node(edge.source)
            dst_node = graph.get_node(edge.target)
            if not src_node or not dst_node:
                continue

            rel_key = (
                f"{src_node.node_type.value}__{edge.relation.value}__{dst_node.node_type.value}"
            )
            if rel_key not in edge_indices:
                edge_indices[rel_key] = [[], []]
                bridge_indices[rel_key] = []

            u_idx = node_id_to_idx[src_node.node_type.value][src_node.id]
            v_idx = node_id_to_idx[dst_node.node_type.value][dst_node.id]

            edge_idx_pos = len(edge_indices[rel_key][0])
            edge_indices[rel_key][0].append(u_idx)
            edge_indices[rel_key][1].append(v_idx)

            if edge.is_bridge:
                bridge_indices[rel_key].append(edge_idx_pos)

        # 3. Format ground-truth positive escalation pairs
        pe_pairs_indices: list[dict[str, Any]] = []
        if label_set is not None:
            for pe in label_set.pe_pairs:
                s_node = graph.get_node(pe.source_id)
                t_node = graph.get_node(pe.target_id)
                if s_node and t_node:
                    s_idx = node_id_to_idx[s_node.node_type.value][s_node.id]
                    t_idx = node_id_to_idx[t_node.node_type.value][t_node.id]
                    pe_pairs_indices.append(
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

        return {
            "graph_id": graph.graph_id,
            "organization_id": graph.metadata.get("organization_id", "unknown"),
            "num_nodes_by_type": {k: len(v) for k, v in nodes_by_type.items()},
            "node_mappings": node_id_to_idx,
            "edge_indices": edge_indices,
            "bridge_edge_indices": bridge_indices,
            "pe_pairs": pe_pairs_indices,
        }

    def export_to_file(
        self,
        graph: IAMGraph,
        output_path: str | Path,
        label_set: GroundTruthLabelSet | None = None,
        metrics_report: TopologicalMetricsReport | None = None,
        schema: str = "structured",
        compress: bool = False,
    ) -> Path:
        """Export graph and annotations directly to file with optional gzip compression."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if schema == "networkx":
            data = self.to_networkx_node_link_dict(graph)
        elif schema == "pyg_ready":
            data = self.to_pyg_ready_dict(graph, label_set)
        else:
            data = self.to_structured_dict(graph, label_set, metrics_report)

        is_compressed = compress or str(out_p).endswith(".gz")

        if self.use_orjson:
            json_bytes = orjson.dumps(data, option=orjson.OPT_INDENT_2)
            if is_compressed:
                with gzip.open(out_p, "wb") as f_gz:
                    f_gz.write(json_bytes)
            else:
                out_p.write_bytes(json_bytes)
        else:
            json_str = json.dumps(data, indent=2)
            if is_compressed:
                with gzip.open(out_p, "wt", encoding="utf-8") as f_gz:
                    f_gz.write(json_str)
            else:
                out_p.write_text(json_str, encoding="utf-8")

        return out_p

    def import_from_file(
        self,
        input_path: str | Path,
    ) -> tuple[IAMGraph, GroundTruthLabelSet | None, TopologicalMetricsReport | None]:
        """Deserialize IAMGraph, ground-truth label set, and metrics report from file."""
        in_p = Path(input_path)
        if not in_p.exists():
            raise FileNotFoundError(f"Export file not found: {input_path}")

        is_compressed = str(in_p).endswith(".gz")

        if self.use_orjson:
            if is_compressed:
                with gzip.open(in_p, "rb") as f_gz:
                    raw_bytes = f_gz.read()
            else:
                raw_bytes = in_p.read_bytes()
            payload = orjson.loads(raw_bytes)
        else:
            if is_compressed:
                with gzip.open(in_p, "rt", encoding="utf-8") as f_gz:
                    payload = json.load(f_gz)
            else:
                with in_p.open("r", encoding="utf-8") as f_in:
                    payload = json.load(f_in)

        # Check schema type
        if "schema_version" in payload and "graph" in payload:
            graph_dict = payload["graph"]
            graph = IAMGraph.from_dict(graph_dict)
            if "metadata" in payload:
                graph.metadata.update(payload["metadata"])

            label_set = (
                GroundTruthLabelSet(**payload["ground_truth"])
                if "ground_truth" in payload
                else None
            )
            metrics = (
                TopologicalMetricsReport(**payload["topological_metrics"])
                if "topological_metrics" in payload
                else None
            )
            return graph, label_set, metrics
        elif "nodes" in payload and ("links" in payload or "edges" in payload):
            # NetworkX node-link data
            nx_g = nx.node_link_graph(payload)
            g_meta = payload.get("graph", {}) if isinstance(payload.get("graph"), dict) else {}
            g_id = g_meta.get("graph_id") or payload.get("graph_id", "imported_nx_graph")
            graph = IAMGraph(graph_id=str(g_id))
            for nid, data in nx_g.nodes(data=True):
                node_type_val = data.get("node_type", NodeType.RESOURCE.value)
                node = GraphNode(
                    id=str(nid),
                    node_type=NodeType(node_type_val),
                    arn=data.get("arn", f"arn:aws:iam:::entity/{nid}"),
                    name=data.get("name", str(nid)),
                    account_id=data.get("account_id", "123456789012"),
                    department=data.get("department"),
                    is_high_value=data.get("is_high_value", False),
                    is_admin=data.get("is_admin", False),
                    tags=data.get("tags", {}),
                )
                graph.add_node(node)
            for u, v, data in nx_g.edges(data=True):
                rel_str = data.get("relation", EdgeRelation.ACTS_ON.value)
                edge = GraphEdge(
                    source=str(u),
                    target=str(v),
                    relation=EdgeRelation(rel_str),
                    is_bridge=data.get("is_bridge", False),
                    motif_id=data.get("motif_id"),
                    actions=data.get("actions", []),
                    condition=data.get("condition"),
                    metadata=data.get("metadata", {}),
                )
                graph.add_edge(edge)
            return graph, None, None
        elif "graph" in payload and isinstance(payload["graph"], dict):
            graph_dict = payload["graph"]
            graph = IAMGraph.from_dict(graph_dict)
            if "metadata" in payload:
                graph.metadata.update(payload["metadata"])
            return graph, None, None
        else:
            raise ValueError(f"Unrecognized dataset export schema in {input_path}")
