"""Interactive demonstration script for Phase 1 Data Pipeline & PyG Conversion.

Demonstrates end-to-end synthetic enterprise topology generation,
privilege escalation path injection, conversion to PyTorch Geometric HeteroData,
adversarial bridge masking (Condition E), and stratified negative sampling.

Usage:
    uv run python scripts/demo_pipeline.py
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch_geometric.nn import HeteroConv, SAGEConv

from iam.generator import (
    EnterpriseTopologyConfig,
    EnterpriseTopologyGenerator,
    GroundTruthLabeler,
    NodeType,
)
from iam.models import (
    IAMHeteroDataConverter,
    StratifiedNegativeSampler,
    StructuredMaskingSuite,
    load_default_action_vocabulary,
)
from iam.parser import load_default_capability_model


def print_header(title: str) -> None:
    print(f"\n{'=' * 75}")
    print(f"  {title}")
    print(f"{'=' * 75}")


def main() -> None:
    print("\n" + "#" * 75)
    print("  IAM GRAPH LEARNING - PHASE 1 PIPELINE & PYG TENSOR DEMO")
    print("#" * 75)

    # 1. Generate Synthetic Enterprise Topology
    print_header("1. Synthetic Enterprise Topology Generation")
    cap_model = load_default_capability_model()
    cfg = EnterpriseTopologyConfig(
        num_nodes=350,
        seed=42,
        num_pe_chains=2,
        include_branching_chains=True,
        num_branching_chains=1,
    )
    generator = EnterpriseTopologyGenerator(config=cfg, capability_model=cap_model)
    graph = generator.generate()

    node_stats = {
        "Users": len(graph.get_nodes_by_type(NodeType.USER)),
        "Groups": len(graph.get_nodes_by_type(NodeType.GROUP)),
        "Roles": len(graph.get_nodes_by_type(NodeType.ROLE)),
        "Policies": len(graph.get_nodes_by_type(NodeType.POLICY)),
        "Resources": len(graph.get_nodes_by_type(NodeType.RESOURCE)),
        "Total Nodes": graph.num_nodes,
        "Total Edges": graph.num_edges,
    }
    for k, v in node_stats.items():
        print(f"  {k:<16}: {v}")

    # Ground-truth reachability
    labeler = GroundTruthLabeler(max_depth=6)
    label_set = labeler.label_graph(graph)
    print(f"\n  Injected PE Escalation Pairs Found: {label_set.num_pe_pairs}")
    for i, p in enumerate(label_set.pe_pairs[:2], 1):
        print(f"    Path #{i}: {p.source_id} -> ... -> {p.target_id} (hops: {p.hop_distance})")

    # 2. PyG Heterogeneous Graph Conversion
    print_header("2. PyTorch Geometric (PyG) HeteroData Conversion")
    vocab = load_default_action_vocabulary()
    converter = IAMHeteroDataConverter(vocabulary=vocab, include_reverse_edges=True)
    hetero_data = converter.convert(graph, label_set=label_set)

    print(f"  Active Node Types in PyG HeteroData: {len(hetero_data.node_types)}")
    for nt in hetero_data.node_types:
        x_shape = list(hetero_data[nt].x.shape) if hasattr(hetero_data[nt], "x") else "No features"
        print(f"    Node [{nt:<8}]: count={hetero_data[nt].num_nodes:<4} feature_matrix={x_shape}")

    print(f"\n  Active Edge Types (Forward & Reverse): {len(hetero_data.edge_types)}")
    forward_edges = [et for et in hetero_data.edge_types if not et[1].startswith("Rev")]
    reverse_edges = [et for et in hetero_data.edge_types if et[1].startswith("Rev")]
    print(f"    Forward relations: {len(forward_edges)} types")
    for src, rel, dst in forward_edges:
        count = hetero_data[(src, rel, dst)].edge_index.size(1)
        print(f"      ({src}) -[{rel}]-> ({dst}): {count} edges")
    print(f"    Reverse relations: {len(reverse_edges)} types (for bidirectional message passing)")

    # 3. Structured Masking: Condition E (Adversarial Bridge Masking)
    print_header("3. Structured Masking: Condition E (Adversarial Bridge Excision)")
    sampler = StratifiedNegativeSampler()
    masking_res = StructuredMaskingSuite.apply_condition(
        data=hetero_data,
        condition="E",
        seed=42,
        negative_sampler=sampler,
        negative_ratio=1.0,
    )

    bridge_pe_pairs = [p for p in label_set.pe_pairs if len(p.bridge_relations) > 0]
    if bridge_pe_pairs:
        first_pe = bridge_pe_pairs[0]
        severed = StructuredMaskingSuite.verify_reachability_severed(
            original_graph=graph,
            masked_result=masking_res,
            source_id=first_pe.source_id,
            target_id=first_pe.target_id,
        )
        print(f"  Sample Injected PE Path Tested:     {first_pe.source_id} -> {first_pe.target_id}")
        print(f"  Attacker Path Severed Successfully: {severed}")

    total_hidden = sum(h.size(1) for h in masking_res.hidden_edges.values())
    print(f"\n  Excised Latent Bridge Edges (|H|): {total_hidden}")
    for et, hidden in masking_res.hidden_edges.items():
        if hidden.size(1) > 0:
            print(
                f"    Severed relation ({et[0]}) -[{et[1]}]-> ({et[2]}): {hidden.size(1)} bridge edges removed"
            )

    # 4. Stratified Negative Sampling & Supervision Tensors
    print_header("4. Stratified Negative Sampling & Link Prediction Targets")
    print("  Supervision Candidate Tensors (for GNN link prediction evaluation):")
    for et, sup in masking_res.supervision_by_edge_type.items():
        pos_count = int((sup.edge_label == 1.0).sum().item())
        neg_count = int((sup.edge_label == 0.0).sum().item())
        print(
            f"    Relation {et[1]:<14}: {sup.edge_label.size(0):<3} total candidates ({pos_count} positive, {neg_count} negative)"
        )

    # 5. Neural Message-Passing Dry Run
    print_header("5. PyG Multi-Relational Neural Message-Passing Dry Run on Go")
    observed = masking_res.observed_data
    hidden_dim = 16
    conv_dict = {}
    lin_dict = nn.ModuleDict()
    for nt in observed.node_types:
        in_dim = observed[nt].x.size(1)
        lin_dict[nt] = nn.Linear(in_dim, hidden_dim)
    for et in observed.edge_types:
        if observed[et].edge_index.size(1) > 0:
            conv_dict[et] = SAGEConv((hidden_dim, hidden_dim), hidden_dim)

    conv = HeteroConv(conv_dict, aggr="sum")
    h_dict = {nt: F.relu(lin_dict[nt](observed[nt].x)) for nt in observed.node_types}
    edge_index_dict = {et: observed[et].edge_index for et in conv_dict}
    out_dict = conv(h_dict, edge_index_dict)

    print("  Node Embeddings Output after 1-Hop Message Passing:")
    for nt, emb in out_dict.items():
        print(f"    Node [{nt:<8}]: embedding shape={list(emb.shape)}")

    # Compute sample loss on AssumesRole supervision candidates if present
    target_rel = ("User", "AssumesRole", "Role")
    if target_rel in masking_res.supervision_by_edge_type:
        sup = masking_res.supervision_by_edge_type[target_rel]
        src_emb = out_dict["User"][sup.edge_label_index[0]]
        dst_emb = out_dict["Role"][sup.edge_label_index[1]]
        logits = (src_emb * dst_emb).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(logits, sup.edge_label)
        print(f"\n  Sample Binary Cross-Entropy Loss on ({target_rel[1]}): {loss.item():.4f}")

    print("\n" + "#" * 75)
    print("  DEMO COMPLETE: Inductive PyG Pipeline verified and ready for Phase 2.")
    print("#" * 75 + "\n")


if __name__ == "__main__":
    main()
