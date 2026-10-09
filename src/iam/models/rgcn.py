"""Heterogeneous Relational Graph Convolutional Network (RGCN) encoder.

Implements multi-relational message passing over AWS IAM authorization graphs
with heterogeneous input projections, basis-sharing decomposition regularization (B=8),
layer normalization, dropout, and LeakyReLU activations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv

from iam.models.converter import (
    CANONICAL_FORWARD_EDGE_TYPES,
    get_reverse_edge_type,
)
from iam.models.vocabulary import (
    FeatureDimensionConfig,
    get_feature_dimension_config,
    load_default_action_vocabulary,
)

# Canonical 26 relation triples (13 forward + 13 reverse message-passing relations)
CANONICAL_RELATION_TYPES: tuple[tuple[str, str, str], ...] = (
    *CANONICAL_FORWARD_EDGE_TYPES,
    *(get_reverse_edge_type(edge_type) for edge_type in CANONICAL_FORWARD_EDGE_TYPES),
)

CANONICAL_NODE_TYPES: tuple[str, ...] = (
    "User",
    "Role",
    "Policy",
    "Group",
    "Resource",
)


def get_default_in_channels_dict() -> dict[str, int]:
    """Compute default input feature dimensions from the canonical AWS ActionVocabulary."""
    vocab = load_default_action_vocabulary()
    cfg: FeatureDimensionConfig = get_feature_dimension_config(vocab)
    return {
        "User": cfg.user_dim,
        "Role": cfg.role_dim,
        "Policy": cfg.policy_dim,
        "Group": cfg.group_dim,
        "Resource": cfg.resource_dim,
    }


@dataclass
class RGCNConfig:
    """Configuration contracts for HeteroRGCNEncoder architecture."""

    in_channels_dict: dict[str, int] = field(default_factory=get_default_in_channels_dict)
    hidden_channels: int = 128
    num_layers: int = 2
    num_bases: int = 8
    num_relations: int = len(CANONICAL_RELATION_TYPES)
    dropout: float = 0.2
    act_negative_slope: float = 0.2
    use_residual: bool = True
    layer_norm: bool = True
    aggr: str = "mean"

    def __post_init__(self) -> None:
        """Validate architectural hyperparameters."""
        if self.hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {self.hidden_channels}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.num_bases <= 0:
            raise ValueError(f"num_bases must be >= 1, got {self.num_bases}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")


class HeteroRGCNEncoder(nn.Module):
    """Heterogeneous Relational Graph Convolutional Network (RGCN) encoder.

    Architecture:
        1. Type-Specific Input Projection: Maps raw node features across 5 heterogeneous
           entity types to a shared hidden space d_hidden:
               h_{nt}^{(0)} = Dropout(LeakyReLU(LayerNorm(Linear(x_{nt}))))
        2. Multi-Relational Message Passing with Basis-Sharing Regularization (L=2):
               W_r^{(l)} = sum_{b=1}^B a_{r,b}^{(l)} V_b^{(l)}
               h_i^{(l+1)} = sigma(W_0^{(l)} h_i^{(l)} + sum_{r} sum_{j in N_i^r} 1/c_{i,r} W_r^{(l)} h_j^{(l)})
        3. Regularization & Residual Connections:
               Residual skip-connections, LayerNorm, LeakyReLU(0.2), and Dropout (0.2).
    """

    def __init__(self, config: RGCNConfig | None = None) -> None:
        """Initialize HeteroRGCNEncoder with hyperparameters."""
        super().__init__()
        self.config: RGCNConfig = config if config is not None else RGCNConfig()

        # Build bidirectional relation mapping
        self.relation_to_id: dict[tuple[str, str, str], int] = {
            rel: idx for idx, rel in enumerate(CANONICAL_RELATION_TYPES)
        }
        self.id_to_relation: dict[int, tuple[str, str, str]] = {
            idx: rel for rel, idx in self.relation_to_id.items()
        }

        # Ensure num_relations accommodates all registered relations
        effective_num_relations = max(self.config.num_relations, len(self.relation_to_id))

        # 1. Type-Specific Input Projections & Input LayerNorms
        self.input_proj = nn.ModuleDict()
        self.input_norm = nn.ModuleDict()
        for nt, in_dim in self.config.in_channels_dict.items():
            self.input_proj[nt] = nn.Linear(in_dim, self.config.hidden_channels)
            if self.config.layer_norm:
                self.input_norm[nt] = nn.LayerNorm(self.config.hidden_channels)

        self.input_act = nn.LeakyReLU(negative_slope=self.config.act_negative_slope)
        self.dropout = nn.Dropout(p=self.config.dropout)

        # 2. Relational Graph Convolution Layers with Basis Decomposition
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.acts = nn.ModuleList()

        for _ in range(self.config.num_layers):
            conv = RGCNConv(
                in_channels=self.config.hidden_channels,
                out_channels=self.config.hidden_channels,
                num_relations=effective_num_relations,
                num_bases=self.config.num_bases,
                aggr=self.config.aggr,
                root_weight=True,
                bias=True,
            )
            self.convs.append(conv)
            if self.config.layer_norm:
                self.layer_norms.append(nn.LayerNorm(self.config.hidden_channels))
            self.acts.append(nn.LeakyReLU(negative_slope=self.config.act_negative_slope))

    def get_relation_id(self, edge_type: tuple[str, str, str]) -> int:
        """Return the canonical integer relation ID for a typed edge triple."""
        if edge_type not in self.relation_to_id:
            raise KeyError(
                f"Unknown edge relation triple {edge_type}. "
                f"Registered relations: {list(self.relation_to_id.keys())}"
            )
        return self.relation_to_id[edge_type]

    def get_relation_triple(self, relation_id: int) -> tuple[str, str, str]:
        """Return the canonical relation triple for an integer relation ID."""
        if relation_id not in self.id_to_relation:
            raise KeyError(f"Unknown relation ID {relation_id}")
        return self.id_to_relation[relation_id]

    def _prepare_flat_graph(
        self,
        projected_x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int], dict[str, int]]:
        """Flatten heterogeneous node projections and edges into unified indexed tensors.

        Returns:
            flat_h: Unified node tensor [N_total, d_hidden]
            flat_edge_index: Shifted edge tensor [2, E_total]
            flat_edge_type: Relation type tensor [E_total]
            node_offsets: Starting global index per node type
            node_counts: Count of nodes per node type
        """
        # Determine deterministic node type ordering
        present_types: list[str] = [nt for nt in CANONICAL_NODE_TYPES if nt in projected_x_dict]
        extra_types: list[str] = sorted(
            [nt for nt in projected_x_dict if nt not in CANONICAL_NODE_TYPES]
        )
        ordered_types = present_types + extra_types

        node_offsets: dict[str, int] = {}
        node_counts: dict[str, int] = {}
        flat_h_list: list[torch.Tensor] = []
        current_offset = 0

        for nt in ordered_types:
            tensor = projected_x_dict[nt]
            count = int(tensor.size(0))
            node_offsets[nt] = current_offset
            node_counts[nt] = count
            flat_h_list.append(tensor)
            current_offset += count

        device = flat_h_list[0].device if flat_h_list else next(self.parameters()).device

        if not flat_h_list or current_offset == 0:
            flat_h = torch.empty((0, self.config.hidden_channels), device=device)
        else:
            flat_h = torch.cat(flat_h_list, dim=0)

        # Collect and shift edge indices
        shifted_edges_list: list[torch.Tensor] = []
        edge_types_list: list[torch.Tensor] = []

        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.numel() == 0 or edge_index.size(1) == 0:
                continue

            src_type, _rel_name, dst_type = edge_type
            if src_type not in node_offsets or dst_type not in node_offsets:
                continue

            rel_id = self.relation_to_id.get(edge_type)
            if rel_id is None:
                continue

            src_offset = node_offsets[src_type]
            dst_offset = node_offsets[dst_type]

            shifted_src = edge_index[0] + src_offset
            shifted_dst = edge_index[1] + dst_offset
            shifted_edge = torch.stack([shifted_src, shifted_dst], dim=0)

            rel_tensor = torch.full(
                (edge_index.size(1),),
                rel_id,
                dtype=torch.long,
                device=edge_index.device,
            )

            shifted_edges_list.append(shifted_edge)
            edge_types_list.append(rel_tensor)

        if shifted_edges_list:
            flat_edge_index = torch.cat(shifted_edges_list, dim=1)
            flat_edge_type = torch.cat(edge_types_list, dim=0)
        else:
            flat_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            flat_edge_type = torch.empty((0,), dtype=torch.long, device=device)

        return flat_h, flat_edge_index, flat_edge_type, node_offsets, node_counts

    def forward(
        self,
        data: HeteroData | None = None,
        x_dict: dict[str, torch.Tensor] | None = None,
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute multi-relational node embeddings across all entity types.

        Supports invocation either via a PyG HeteroData instance or via raw dictionaries:
            encoder(data)
            encoder(x_dict=x_dict, edge_index_dict=edge_index_dict)

        Args:
            data: Optional PyG HeteroData instance containing typed node features and edges.
            x_dict: Optional dictionary mapping node type -> feature tensor [N_type, D_type].
            edge_index_dict: Optional dictionary mapping relation triple -> edge index [2, E_rel].

        Returns:
            out_dict: Dictionary mapping node type -> hidden embedding tensor [N_type, d_hidden].
        """
        if data is not None:
            extracted_x: dict[str, torch.Tensor] = {}
            for nt in data.node_types:
                if hasattr(data[nt], "x") and data[nt].x is not None:
                    extracted_x[nt] = cast(torch.Tensor, data[nt].x)

            extracted_edges: dict[tuple[str, str, str], torch.Tensor] = {}
            for et in data.edge_types:
                if hasattr(data[et], "edge_index") and data[et].edge_index is not None:
                    extracted_edges[et] = cast(torch.Tensor, data[et].edge_index)

            x_dict = extracted_x
            edge_index_dict = extracted_edges
        elif x_dict is None or edge_index_dict is None:
            raise ValueError(
                "Must provide either 'data' (HeteroData) or both 'x_dict' and 'edge_index_dict'."
            )

        # 1. Project heterogeneous input features to shared hidden space
        projected_x_dict: dict[str, torch.Tensor] = {}
        for nt, x in x_dict.items():
            if nt not in self.input_proj:
                raise KeyError(
                    f"Node type '{nt}' not configured in in_channels_dict. "
                    f"Configured types: {list(self.input_proj.keys())}"
                )
            h = self.input_proj[nt](x)
            if self.config.layer_norm and nt in self.input_norm:
                h = self.input_norm[nt](h)
            h = self.input_act(h)
            h = self.dropout(h)
            projected_x_dict[nt] = h

        # 2. Flatten into unified global index space for RGCNConv
        flat_h, flat_edge_index, flat_edge_type, node_offsets, node_counts = (
            self._prepare_flat_graph(projected_x_dict, edge_index_dict)
        )

        # Handle empty graph edge case
        if flat_h.size(0) == 0:
            return {
                nt: torch.empty((0, self.config.hidden_channels), device=flat_h.device)
                for nt in x_dict
            }

        # 3. Apply Multi-Relational Convolutional Layers
        h_current = flat_h
        for layer_idx, conv in enumerate(self.convs):
            h_next = conv(h_current, flat_edge_index, flat_edge_type)
            if self.config.use_residual:
                h_next = h_next + h_current
            if self.config.layer_norm and layer_idx < len(self.layer_norms):
                h_next = self.layer_norms[layer_idx](h_next)
            h_next = self.acts[layer_idx](h_next)
            h_next = self.dropout(h_next)
            h_current = h_next

        # 4. Partition back into per-node-type dictionary
        out_dict: dict[str, torch.Tensor] = {}
        for nt in x_dict:
            if nt in node_offsets and nt in node_counts:
                offset = node_offsets[nt]
                count = node_counts[nt]
                out_dict[nt] = h_current[offset : offset + count]
            else:
                out_dict[nt] = torch.empty((0, self.config.hidden_channels), device=flat_h.device)

        return out_dict

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_basis_parameters(self) -> dict[str, int]:
        """Compute parameter breakdown demonstrating basis-sharing regularization savings.

        Returns:
            Dictionary with breakdown of parameter allocations:
                - 'input_projections': Linear layers mapping raw node features.
                - 'basis_matrices': Shared V_b basis matrices across relations.
                - 'basis_coefficients': Per-relation linear combination weights a_{r,b}.
                - 'root_weights': Self-loop transformation matrices W_0.
                - 'layer_norms': LayerNorm affine parameters.
                - 'total': Total parameter count.
                - 'unconstrained_conv_estimate': Theoretical parameter count without basis sharing.
        """
        input_proj_params = sum(p.numel() for p in self.input_proj.parameters())
        input_norm_params = sum(p.numel() for p in self.input_norm.parameters())

        basis_matrix_params = 0
        basis_coeff_params = 0
        root_weight_params = 0
        conv_bias_params = 0

        for conv in self.convs:
            for name, param in conv.named_parameters():
                if "weight" in name:
                    basis_matrix_params += param.numel()
                elif "comp" in name:
                    basis_coeff_params += param.numel()
                elif "root" in name:
                    root_weight_params += param.numel()
                elif "bias" in name:
                    conv_bias_params += param.numel()

        layer_norm_params = sum(p.numel() for p in self.layer_norms.parameters())

        d_h = self.config.hidden_channels
        num_rels = max(self.config.num_relations, len(self.relation_to_id))
        unconstrained_single_layer = num_rels * d_h * d_h + d_h * d_h + d_h
        unconstrained_total_conv = unconstrained_single_layer * self.config.num_layers

        return {
            "input_projections": input_proj_params + input_norm_params,
            "basis_matrices": basis_matrix_params,
            "basis_coefficients": basis_coeff_params,
            "root_weights": root_weight_params,
            "conv_biases": conv_bias_params,
            "layer_norms": layer_norm_params,
            "total": self.count_parameters(),
            "unconstrained_conv_estimate": unconstrained_total_conv,
        }
