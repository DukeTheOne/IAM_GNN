"""Untyped homogeneous GCN baseline architecture for architectural ablation studies.

Collapses multi-relational AWS IAM authorization multigraphs into homogeneous
graphs by discarding relation types and passing messages via standard 2-layer GCNConv.
Used in Step 6.3 to empirically quantify the contribution of multi-relational
basis-sharing message passing (HeteroRGCN) over untyped graph convolutions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv

from iam.models.rgcn import CANONICAL_NODE_TYPES, get_default_in_channels_dict


@dataclass
class UntypedGCNConfig:
    """Configuration hyperparameters for UntypedGCNEncoder."""

    in_channels_dict: dict[str, int] = field(default_factory=get_default_in_channels_dict)
    hidden_channels: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    act_negative_slope: float = 0.2
    use_residual: bool = True


class UntypedGCNEncoder(nn.Module):
    """Homogeneous 2-layer GCN encoder that collapses all multi-relational edges into an un-typed graph."""

    def __init__(self, config: UntypedGCNConfig | None = None) -> None:
        super().__init__()
        self.config: UntypedGCNConfig = config or UntypedGCNConfig()

        if self.config.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.config.num_layers}")
        if self.config.hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be > 0, got {self.config.hidden_channels}")

        # Heterogeneous input projection layers
        self.input_proj = nn.ModuleDict()
        self.input_norm = nn.ModuleDict()
        for node_type, in_dim in self.config.in_channels_dict.items():
            self.input_proj[node_type] = nn.Linear(in_dim, self.config.hidden_channels)
            self.input_norm[node_type] = nn.LayerNorm(self.config.hidden_channels)

        # Homogeneous GCNConv layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(self.config.num_layers):
            self.convs.append(
                GCNConv(
                    self.config.hidden_channels,
                    self.config.hidden_channels,
                    add_self_loops=True,
                )
            )
            self.layer_norms.append(nn.LayerNorm(self.config.hidden_channels))

        self.activation = nn.LeakyReLU(negative_slope=self.config.act_negative_slope)
        self.dropout = nn.Dropout(p=self.config.dropout)

    def _project_inputs(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Project heterogeneous raw node feature tensors into a common continuous embedding space."""
        projected: dict[str, torch.Tensor] = {}
        for nt, x in x_dict.items():
            if nt not in self.input_proj:
                raise KeyError(
                    f"Node type '{nt}' missing in UntypedGCNEncoder input_proj configuration. "
                    f"Configured types: {list(self.input_proj.keys())}"
                )
            h = self.input_proj[nt](x)
            h = self.input_norm[nt](h)
            h = self.activation(h)
            h = self.dropout(h)
            projected[nt] = h
        return projected

    def _flatten_graph(
        self,
        projected_x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int], dict[str, int]]:
        """Flatten heterogeneous node projections and un-typed edges into unified indexed tensors."""
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

        # Collect and shift all edge indices into a single homogeneous un-typed tensor
        shifted_edges_list: list[torch.Tensor] = []
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.numel() == 0 or edge_index.size(1) == 0:
                continue

            src_type, _rel_name, dst_type = edge_type
            if src_type not in node_offsets or dst_type not in node_offsets:
                continue

            src_offset = node_offsets[src_type]
            dst_offset = node_offsets[dst_type]

            shifted_src = edge_index[0] + src_offset
            shifted_dst = edge_index[1] + dst_offset
            shifted_edges_list.append(torch.stack([shifted_src, shifted_dst], dim=0))

        if shifted_edges_list:
            flat_edge_index = torch.cat(shifted_edges_list, dim=1)
        else:
            flat_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        return flat_h, flat_edge_index, node_offsets, node_counts

    def forward(
        self,
        data: HeteroData | None = None,
        x_dict: dict[str, torch.Tensor] | None = None,
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute homogeneous un-typed GCN embeddings across all heterogeneous entity types."""
        if data is not None:
            extracted_x: dict[str, torch.Tensor] = {}
            for node_type in data.node_types:
                if hasattr(data[node_type], "x") and data[node_type].x is not None:
                    extracted_x[node_type] = data[node_type].x

            extracted_edge: dict[tuple[str, str, str], torch.Tensor] = {}
            for edge_type in data.edge_types:
                if (
                    hasattr(data[edge_type], "edge_index")
                    and data[edge_type].edge_index is not None
                ):
                    extracted_edge[edge_type] = data[edge_type].edge_index

            active_x = extracted_x
            active_edge = extracted_edge
        elif x_dict is not None and edge_index_dict is not None:
            active_x = x_dict
            active_edge = edge_index_dict
        else:
            raise ValueError(
                "Must provide either HeteroData 'data' or both 'x_dict' and 'edge_index_dict'"
            )

        if not active_x:
            return {}

        projected = self._project_inputs(active_x)
        flat_h, flat_edge_index, node_offsets, node_counts = self._flatten_graph(
            projected, active_edge
        )

        if flat_h.size(0) == 0:
            return {
                nt: torch.empty((0, self.config.hidden_channels), device=flat_h.device)
                for nt in active_x
            }

        h = flat_h
        for layer_idx, conv in enumerate(self.convs):
            h_in = h
            h = conv(h, flat_edge_index) if flat_edge_index.size(1) > 0 else h

            h = self.layer_norms[layer_idx](h)
            h = self.activation(h)
            h = self.dropout(h)
            if self.config.use_residual and h.shape == h_in.shape:
                h = h + h_in

        # Unpack back to dictionary by node type
        out_dict: dict[str, torch.Tensor] = {}
        for nt, offset in node_offsets.items():
            count = node_counts[nt]
            out_dict[nt] = h[offset : offset + count]

        return out_dict


class UntypedRelationPredictor(nn.Module):
    """Homogeneous bilinear scoring head that scores node pairs using a single shared relation matrix."""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # Single un-typed relation transformation matrix
        self.weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute uncalibrated logits for source-destination node pairs."""
        if h_src.size(0) != h_dst.size(0):
            raise ValueError(f"Batch size mismatch: h_src {h_src.size(0)} vs h_dst {h_dst.size(0)}")
        if h_src.size(0) == 0:
            return torch.empty(0, device=h_src.device)

        h_src_drop = self.dropout(h_src)
        h_dst_drop = self.dropout(h_dst)

        # Vectorized pair scoring: (h_u W) dot h_v + b
        src_proj = torch.matmul(h_src_drop, self.weight)
        logits = (src_proj * h_dst_drop).sum(dim=-1) + self.bias
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)
        return cast(torch.Tensor, logits)


class UntypedLinkPredictionModel(nn.Module):
    """Composite Untyped GCN model pairing UntypedGCNEncoder with UntypedRelationPredictor."""

    def __init__(
        self,
        encoder_config: UntypedGCNConfig | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = UntypedGCNEncoder(config=encoder_config)
        self.decoder = UntypedRelationPredictor(
            hidden_dim=self.encoder.config.hidden_channels,
            dropout=dropout,
        )

    def encode(
        self,
        data: HeteroData | None = None,
        x_dict: dict[str, torch.Tensor] | None = None,
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute node representations using the untyped homogeneous GCN encoder."""
        return cast(
            dict[str, torch.Tensor],
            self.encoder(data=data, x_dict=x_dict, edge_index_dict=edge_index_dict),
        )

    def score_edges(
        self,
        h_dict: dict[str, torch.Tensor],
        edge_type: tuple[str, str, str],
        edge_index: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Score candidate edge index pairs."""
        src_type, _rel_name, dst_type = edge_type
        if src_type not in h_dict or dst_type not in h_dict:
            raise KeyError(f"Entity types {src_type} or {dst_type} not in embeddings dictionary")

        if edge_index.size(1) == 0:
            return torch.empty(0, device=edge_index.device)

        src_indices = edge_index[0]
        dst_indices = edge_index[1]

        h_src = h_dict[src_type][src_indices]
        h_dst = h_dict[dst_type][dst_indices]

        return cast(torch.Tensor, self.decoder(h_src, h_dst, temperature=temperature))

    def forward(
        self,
        data: HeteroData,
        supervision_dict: dict[tuple[str, str, str], torch.Tensor],
        temperature: float = 1.0,
    ) -> dict[tuple[str, str, str], torch.Tensor]:
        """End-to-end forward pass returning logits across candidate edge types."""
        h_dict = self.encode(data=data)
        logits_dict: dict[tuple[str, str, str], torch.Tensor] = {}

        for edge_type, edge_index in supervision_dict.items():
            logits_dict[edge_type] = self.score_edges(
                h_dict, edge_type, edge_index, temperature=temperature
            )

        return logits_dict
