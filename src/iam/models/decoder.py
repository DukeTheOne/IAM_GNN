r"""Action-conditioned bilinear relational link prediction decoder head.

Implements bilinear relational scoring over entity representation pairs:
    z_{uv}^{(r)} = h_u^T W_r h_v + b_r
    \hat{y}_{uv}^{(r)} = \sigma(z_{uv}^{(r)} / T)

Solves Problem 1 (Hidden Relation Prediction) under structured configuration masking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from iam.models.rgcn import CANONICAL_RELATION_TYPES


@dataclass
class DecoderConfig:
    """Configuration contracts for BilinearRelationPredictor."""

    hidden_channels: int = 128
    num_relations: int = len(CANONICAL_RELATION_TYPES)
    mode: str = "bilinear"  # Options: 'bilinear', 'basis', 'distmult'
    num_bases: int = 4
    use_bias: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Validate decoder configuration contracts."""
        if self.hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {self.hidden_channels}")
        if self.num_relations <= 0:
            raise ValueError(f"num_relations must be positive, got {self.num_relations}")
        if self.mode not in ("bilinear", "basis", "distmult"):
            raise ValueError(
                f"mode must be one of ('bilinear', 'basis', 'distmult'), got '{self.mode}'"
            )
        if self.mode == "basis" and self.num_bases <= 0:
            raise ValueError(f"num_bases must be >= 1 for basis mode, got {self.num_bases}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")


class BilinearRelationPredictor(nn.Module):
    """Action-conditioned bilinear relational link prediction decoder.

    Computes relation-specific compatibility scores between source and target node
    representations:
        z_{uv}^{(r)} = h_u^T W_r h_v + b_r
        \\hat{y}_{uv}^{(r)} = \\sigma(z_{uv}^{(r)})

    Supports:
        - Full unconstrained bilinear tensor W \\in R^{R \times d \times d} (standard)
        - Basis-sharing regularized bilinear decomposition W_r = \\sum_b \beta_{r,b} U_b
        - Diagonal DistMult factorized scoring W_r = diag(w_r)
        - Heterogeneous batch pair scoring, homogeneous relation pair scoring, and full matrix ranking.
    """

    def __init__(self, config: DecoderConfig | None = None) -> None:
        """Initialize BilinearRelationPredictor with architectural hyperparameters."""
        super().__init__()
        self.config: DecoderConfig = config if config is not None else DecoderConfig()

        d = self.config.hidden_channels
        r = self.config.num_relations

        self.relation_to_id: dict[tuple[str, str, str], int] = {
            rel: idx for idx, rel in enumerate(CANONICAL_RELATION_TYPES)
        }
        self.id_to_relation: dict[int, tuple[str, str, str]] = {
            idx: rel for rel, idx in self.relation_to_id.items()
        }

        # 1. Relation Weight Parameterization
        if self.config.mode == "bilinear":
            self.W = nn.Parameter(torch.empty(r, d, d))
            self._init_bilinear_weights()
        elif self.config.mode == "basis":
            b = self.config.num_bases
            self.basis = nn.Parameter(torch.empty(b, d, d))
            self.comp = nn.Parameter(torch.empty(r, b))
            self._init_basis_weights()
        elif self.config.mode == "distmult":
            self.W = nn.Parameter(torch.empty(r, d))
            nn.init.xavier_uniform_(self.W)

        # 2. Relation Biases
        if self.config.use_bias:
            self.bias = nn.Parameter(torch.zeros(r))
        else:
            self.register_parameter("bias", None)

        # 3. Regularization Dropout
        self.dropout = nn.Dropout(p=self.config.dropout)

    def _init_bilinear_weights(self) -> None:
        """Initialize full bilinear relation matrices with Xavier uniform scaling."""
        bound = 1.0 / math.sqrt(self.config.hidden_channels)
        nn.init.uniform_(self.W, -bound, bound)

    def _init_basis_weights(self) -> None:
        """Initialize basis matrices and relation combination coefficients."""
        bound = 1.0 / math.sqrt(self.config.hidden_channels)
        nn.init.uniform_(self.basis, -bound, bound)
        nn.init.xavier_uniform_(self.comp)

    def get_relation_id(self, edge_type: tuple[str, str, str]) -> int:
        """Resolve typed edge triple to its integer relation index."""
        if edge_type not in self.relation_to_id:
            raise KeyError(
                f"Unknown relation triple {edge_type}. "
                f"Registered relations: {list(self.relation_to_id.keys())}"
            )
        return self.relation_to_id[edge_type]

    def get_relation_matrix(self, rel_id: int) -> torch.Tensor:
        """Return the effective [d, d] (or [d]) matrix for a given relation ID."""
        if self.config.mode == "bilinear":
            return self.W[rel_id]
        if self.config.mode == "basis":
            # Linear combination: sum_b comp[rel_id, b] * basis[b]
            coeffs = self.comp[rel_id]  # [B]
            return torch.einsum("b,bde->de", coeffs, self.basis)
        # distmult
        return self.W[rel_id]

    def forward(
        self,
        h_u: torch.Tensor,
        h_v: torch.Tensor,
        rel_ids: torch.Tensor | int,
    ) -> torch.Tensor:
        """Compute raw bilinear prediction logits z_{uv}^{(r)} for candidate pairs.

        Args:
            h_u: Source node embedding tensor [M, d].
            h_v: Target node embedding tensor [M, d].
            rel_ids: Either an integer relation ID (homogeneous batch) or
                     a 1D tensor of relation IDs [M] (heterogeneous batch).

        Returns:
            logits: 1D tensor of raw bilinear scores [M].
        """
        if h_u.size(0) != h_v.size(0):
            raise ValueError(
                f"Mismatched batch sizes: h_u has {h_u.size(0)} rows, h_v has {h_v.size(0)} rows."
            )
        if h_u.size(1) != self.config.hidden_channels or h_v.size(1) != self.config.hidden_channels:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.config.hidden_channels}, "
                f"got h_u {h_u.size(1)}, h_v {h_v.size(1)}."
            )

        if h_u.size(0) == 0:
            return torch.empty((0,), dtype=torch.float32, device=h_u.device)

        h_u = self.dropout(h_u)
        h_v = self.dropout(h_v)

        # Case 1: Homogeneous batch where all M pairs evaluate the same relation r
        if isinstance(rel_ids, int):
            rel_id = rel_ids
            if self.config.mode in ("bilinear", "basis"):
                w_rel = self.get_relation_matrix(rel_id)  # [d, d]
                h_u_w = torch.matmul(h_u, w_rel)  # [M, d]
                logits = (h_u_w * h_v).sum(dim=-1)  # [M]
            else:  # distmult
                w_r = self.W[rel_id]  # [d]
                logits = (h_u * w_r * h_v).sum(dim=-1)  # [M]

            if self.bias is not None:
                logits = logits + self.bias[rel_id]

            return logits

        # Case 2: Heterogeneous batch where each pair m has distinct relation r_m
        if rel_ids.size(0) != h_u.size(0):
            raise ValueError(
                f"rel_ids length {rel_ids.size(0)} must match pair count {h_u.size(0)}."
            )

        if self.config.mode == "bilinear":
            w_sel = self.W[rel_ids]  # [M, d, d]
            logits = torch.einsum("md,mde,me->m", h_u, w_sel, h_v)
        elif self.config.mode == "basis":
            # Reconstruct batch matrices via batch einsum
            coeffs_sel = self.comp[rel_ids]  # [M, B]
            w_sel = torch.einsum("mb,bde->mde", coeffs_sel, self.basis)  # [M, d, d]
            logits = torch.einsum("md,mde,me->m", h_u, w_sel, h_v)
        else:  # distmult
            w_sel = self.W[rel_ids]  # [M, d]
            logits = (h_u * w_sel * h_v).sum(dim=-1)

        if self.bias is not None:
            logits = logits + self.bias[rel_ids]

        return logits

    def predict_proba(
        self,
        h_u: torch.Tensor,
        h_v: torch.Tensor,
        rel_ids: torch.Tensor | int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        r"""Compute sigmoid probability predictions \hat{y}_{uv} = \sigma(z_{uv} / T).

        Args:
            h_u: Source node embeddings [M, d].
            h_v: Target node embeddings [M, d].
            rel_ids: Integer relation ID or [M] tensor.
            temperature: Calibration temperature scaling factor T > 0.

        Returns:
            probabilities: 1D tensor in [0.0, 1.0] representing predicted edge likelihoods.
        """
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")

        logits = self.forward(h_u, h_v, rel_ids)
        return torch.sigmoid(logits / temperature)

    def score_matrix(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        rel_id: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute all-pairs prediction matrix Z \\in R^{N_{src} \times N_{dst}} for ranking.

        Args:
            h_src: All source entity representations [N_src, d].
            h_dst: All target entity representations [N_dst, d].
            rel_id: Integer relation ID to evaluate.
            temperature: Calibration temperature T > 0.

        Returns:
            probability_matrix: 2D tensor [N_src, N_dst] of pairwise predicted probabilities.
        """
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")

        if h_src.size(0) == 0 or h_dst.size(0) == 0:
            return torch.empty((h_src.size(0), h_dst.size(0)), device=h_src.device)

        if self.config.mode in ("bilinear", "basis"):
            w_rel = self.get_relation_matrix(rel_id)  # [d, d]
            h_src_w = torch.matmul(h_src, w_rel)  # [N_src, d]
            logits = torch.matmul(h_src_w, h_dst.t())  # [N_src, N_dst]
        else:  # distmult
            w_r = self.W[rel_id]  # [d]
            h_src_w = h_src * w_r  # [N_src, d]
            logits = torch.matmul(h_src_w, h_dst.t())  # [N_src, N_dst]

        if self.bias is not None:
            logits = logits + self.bias[rel_id]

        return torch.sigmoid(logits / temperature)

    def score_edge_index(
        self,
        h_dict: dict[str, torch.Tensor],
        edge_type: tuple[str, str, str],
        edge_index: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Score candidate edge index tensor [2, M] for a specific relation triple.

        Args:
            h_dict: Dictionary mapping node type -> embedding tensor [N_type, d].
            edge_type: Typed relation triple (src_type, rel_name, dst_type).
            edge_index: Edge index tensor [2, M].
            temperature: Calibration temperature T > 0.

        Returns:
            probabilities: 1D tensor [M] of predicted link probabilities.
        """
        src_type, _, dst_type = edge_type
        if src_type not in h_dict or dst_type not in h_dict:
            raise KeyError(f"Node types for {edge_type} not found in h_dict.")

        if edge_index.numel() == 0 or edge_index.size(1) == 0:
            return torch.empty((0,), dtype=torch.float32, device=edge_index.device)

        rel_id = self.get_relation_id(edge_type)
        src_indices = edge_index[0]
        dst_indices = edge_index[1]

        h_u = h_dict[src_type][src_indices]
        h_v = h_dict[dst_type][dst_indices]

        return self.predict_proba(h_u, h_v, rel_id, temperature=temperature)

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
