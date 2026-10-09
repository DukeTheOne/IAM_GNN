"""Formal structured masking framework for partial cloud observability.

Simulates authentic enterprise observability failures across 5 rigorous conditions:
- Condition A: Random edge deletion (P0 classical baseline)
- Condition E: Adversarial bridge bottleneck masking (P0 core PE evaluation)
- Condition B: Cross-account silo boundary masking (P1 multi-account)
- Condition C: Federated IdP identity entrypoint masking (P2 SaaS/IdP)
- Condition D: Ephemeral STS runtime session masking (P2 dynamic credentials)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData

from iam.generator.graph import EdgeRelation, GraphEdge, IAMGraph
from iam.models.converter import get_reverse_edge_type, is_reverse_edge_type
from iam.models.sampler import NegativeSampleResult, StratifiedNegativeSampler


class MaskingCondition(str, Enum):
    """The 5 formal partial observability regimes for IAM authorization multigraphs."""

    CONDITION_A_RANDOM = "condition_a_random"
    CONDITION_E_ADVERSARIAL_BRIDGE = "condition_e_adversarial_bridge"
    CONDITION_B_CROSS_ACCOUNT = "condition_b_cross_account"
    CONDITION_C_FEDERATED_IDP = "condition_c_federated_idp"
    CONDITION_D_EPHEMERAL_STS = "condition_d_ephemeral_sts"


@dataclass
class MaskingResult:
    """Encapsulates the masked observed graph G_o, hidden ground truth H, and supervision tensors."""

    condition: MaskingCondition
    observed_data: HeteroData
    hidden_edges: dict[tuple[str, str, str], torch.Tensor]
    hidden_edge_attributes: dict[tuple[str, str, str], torch.Tensor] = field(default_factory=dict)
    hidden_edge_is_bridge: dict[tuple[str, str, str], torch.Tensor] = field(default_factory=dict)
    supervision_by_edge_type: dict[tuple[str, str, str], NegativeSampleResult] = field(
        default_factory=dict
    )
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def total_observed_edges(self) -> int:
        """Total number of visible forward edges in G_o."""
        count = 0
        for edge_type in self.observed_data.edge_types:
            if not is_reverse_edge_type(edge_type):
                count += int(self.observed_data[edge_type].edge_index.size(1))
        return count

    @property
    def total_hidden_edges(self) -> int:
        """Total number of excised ground-truth edges in H."""
        return sum(int(idx.size(1)) for idx in self.hidden_edges.values())


class BaseMaskingOperator(ABC):
    """Abstract base class for all enterprise observability masking operators."""

    def __init__(self, condition: MaskingCondition) -> None:
        self.condition: MaskingCondition = condition

    @abstractmethod
    def apply(
        self,
        data: HeteroData,
        seed: int = 42,
        negative_sampler: StratifiedNegativeSampler | None = None,
        negative_ratio: float = 1.0,
    ) -> MaskingResult:
        """Apply the masking operator to a PyG HeteroData instance."""

    def apply_to_graph(
        self,
        graph: IAMGraph,
        seed: int = 42,
    ) -> tuple[IAMGraph, list[GraphEdge]]:
        """Apply masking transformation at the IAMGraph symbolic level."""
        masked_graph = IAMGraph(
            graph_id=f"{graph.graph_id}_masked_{self.condition.value}",
            metadata=dict(graph.metadata),
        )
        for node in graph.get_nodes():
            masked_graph.add_node(node)

        hidden_edges: list[GraphEdge] = []
        for edge in graph.get_edges():
            if self._should_mask_graph_edge(edge, graph, seed):
                hidden_edges.append(edge)
            else:
                masked_graph.add_edge(edge)

        return masked_graph, hidden_edges

    def _should_mask_graph_edge(
        self,
        _edge: GraphEdge,
        _graph: IAMGraph,
        _seed: int,
    ) -> bool:
        """Predicate determining if an IAMGraph edge should be hidden."""
        return False

    @staticmethod
    def _slice_hetero_edge_store(
        observed_data: HeteroData,
        triple: tuple[str, str, str],
        keep_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Excise masked edges from forward store and synchronize reverse message-passing store."""
        fwd_edge_index = observed_data[triple].edge_index
        hidden_edge_index = fwd_edge_index[:, ~keep_mask]
        observed_edge_index = fwd_edge_index[:, keep_mask]
        observed_data[triple].edge_index = observed_edge_index

        hidden_attr: torch.Tensor | None = None
        if (
            hasattr(observed_data[triple], "edge_attr")
            and observed_data[triple].edge_attr is not None
        ):
            hidden_attr = observed_data[triple].edge_attr[~keep_mask]
            observed_data[triple].edge_attr = observed_data[triple].edge_attr[keep_mask]

        hidden_bridge: torch.Tensor | None = None
        if (
            hasattr(observed_data[triple], "edge_is_bridge")
            and observed_data[triple].edge_is_bridge is not None
        ):
            hidden_bridge = observed_data[triple].edge_is_bridge[~keep_mask]
            observed_data[triple].edge_is_bridge = observed_data[triple].edge_is_bridge[keep_mask]

        # Synchronize reverse relation store to prevent backward message-passing leakage
        rev_triple = get_reverse_edge_type(triple)
        if rev_triple in observed_data.edge_types:
            observed_data[rev_triple].edge_index = observed_edge_index[[1, 0], :]
            if hasattr(observed_data[rev_triple], "edge_is_bridge"):
                observed_data[rev_triple].edge_is_bridge = torch.zeros(
                    observed_edge_index.size(1), dtype=torch.bool
                )
            if (
                hasattr(observed_data[rev_triple], "edge_attr")
                and observed_data[triple].edge_attr is not None
            ):
                observed_data[rev_triple].edge_attr = observed_data[triple].edge_attr.clone()

        return hidden_edge_index, hidden_attr, hidden_bridge


class RandomMaskingOperator(BaseMaskingOperator):
    """Condition A: Uniform random edge deletion on intermediate delegation and capability links.

    Simulates classical link prediction benchmarks by randomly omitting a fraction p of
    intermediate delegation/authorization edges, preserving node sets and benign structural memberships.
    """

    def __init__(
        self,
        mask_ratio: float = 0.20,
        protect_member_of: bool = True,
    ) -> None:
        super().__init__(MaskingCondition.CONDITION_A_RANDOM)
        if not (0.0 <= mask_ratio <= 1.0):
            raise ValueError("mask_ratio must be between 0.0 and 1.0")
        self.mask_ratio: float = mask_ratio
        self.protect_member_of: bool = protect_member_of

    def apply(
        self,
        data: HeteroData,
        seed: int = 42,
        negative_sampler: StratifiedNegativeSampler | None = None,
        negative_ratio: float = 1.0,
    ) -> MaskingResult:
        observed_data = data.clone()
        rng = np.random.default_rng(seed)

        hidden_edges: dict[tuple[str, str, str], torch.Tensor] = {}
        hidden_attrs: dict[tuple[str, str, str], torch.Tensor] = {}
        hidden_bridges: dict[tuple[str, str, str], torch.Tensor] = {}

        for triple in list(observed_data.edge_types):
            if is_reverse_edge_type(triple):
                continue
            _src, rel, _dst = triple
            if self.protect_member_of and rel == EdgeRelation.MEMBER_OF.value:
                continue

            num_edges = observed_data[triple].edge_index.size(1)
            if num_edges == 0:
                continue

            # Uniform random drop decision
            rand_vals = rng.random(num_edges)
            keep_mask = torch.from_numpy(rand_vals >= self.mask_ratio)

            if (~keep_mask).any():
                h_idx, h_attr, h_bridge = self._slice_hetero_edge_store(
                    observed_data, triple, keep_mask
                )
                hidden_edges[triple] = h_idx
                if h_attr is not None:
                    hidden_attrs[triple] = h_attr
                if h_bridge is not None:
                    hidden_bridges[triple] = h_bridge

        supervision: dict[tuple[str, str, str], NegativeSampleResult] = {}
        if negative_sampler is not None:
            for triple, h_idx in hidden_edges.items():
                res = negative_sampler.sample(
                    observed_data,
                    triple,
                    num_negatives=int(round(h_idx.size(1) * negative_ratio)),
                    seed=seed,
                )
                # Override positive evaluation edges with the exact hidden edges
                edge_label_index = torch.cat([h_idx, res.neg_edge_index], dim=1)
                edge_label = torch.cat(
                    [
                        torch.ones(h_idx.size(1), dtype=torch.float32),
                        torch.zeros(res.neg_edge_index.size(1), dtype=torch.float32),
                    ],
                    dim=0,
                )
                supervision[triple] = NegativeSampleResult(
                    edge_type=triple,
                    pos_edge_index=h_idx,
                    neg_edge_index=res.neg_edge_index,
                    edge_label_index=edge_label_index,
                    edge_label=edge_label,
                    stratum_counts=res.stratum_counts,
                )

        return MaskingResult(
            condition=self.condition,
            observed_data=observed_data,
            hidden_edges=hidden_edges,
            hidden_edge_attributes=hidden_attrs,
            hidden_edge_is_bridge=hidden_bridges,
            supervision_by_edge_type=supervision,
            provenance={
                "condition": self.condition.value,
                "mask_ratio": self.mask_ratio,
                "seed": seed,
                "protect_member_of": self.protect_member_of,
            },
        )


class AdversarialBridgeMaskingOperator(BaseMaskingOperator):
    """Condition E: Targeted masking of critical privilege escalation bottleneck relations.

    Excises relations where edge_is_bridge == True. Formal property:
    Drops deterministic BFS reachability from attacker entrypoint to high-value targets to 0%.
    Evaluates whether the GNN can reconstruct the critical missing attack step from surrounding context.
    """

    def __init__(self, bridge_mask_ratio: float = 1.0) -> None:
        super().__init__(MaskingCondition.CONDITION_E_ADVERSARIAL_BRIDGE)
        if not (0.0 <= bridge_mask_ratio <= 1.0):
            raise ValueError("bridge_mask_ratio must be between 0.0 and 1.0")
        self.bridge_mask_ratio: float = bridge_mask_ratio

    def apply(
        self,
        data: HeteroData,
        seed: int = 42,
        negative_sampler: StratifiedNegativeSampler | None = None,
        negative_ratio: float = 1.0,
    ) -> MaskingResult:
        observed_data = data.clone()
        rng = np.random.default_rng(seed)

        hidden_edges: dict[tuple[str, str, str], torch.Tensor] = {}
        hidden_attrs: dict[tuple[str, str, str], torch.Tensor] = {}
        hidden_bridges: dict[tuple[str, str, str], torch.Tensor] = {}

        total_bridges_masked = 0

        for triple in list(observed_data.edge_types):
            if is_reverse_edge_type(triple):
                continue
            if not hasattr(observed_data[triple], "edge_is_bridge"):
                continue

            bridge_mask = observed_data[triple].edge_is_bridge
            num_bridges = int(bridge_mask.sum().item())
            if num_bridges == 0:
                continue

            if self.bridge_mask_ratio < 1.0:
                # Stochastic subset of bridges
                bridge_indices = torch.where(bridge_mask)[0].tolist()
                num_to_mask = int(round(num_bridges * self.bridge_mask_ratio))
                chosen = set(rng.choice(bridge_indices, size=num_to_mask, replace=False).tolist())
                remove_mask = torch.tensor(
                    [i in chosen for i in range(bridge_mask.size(0))], dtype=torch.bool
                )
            else:
                remove_mask = bridge_mask

            keep_mask = ~remove_mask
            if remove_mask.any():
                h_idx, h_attr, h_bridge = self._slice_hetero_edge_store(
                    observed_data, triple, keep_mask
                )
                hidden_edges[triple] = h_idx
                total_bridges_masked += int(h_idx.size(1))
                if h_attr is not None:
                    hidden_attrs[triple] = h_attr
                if h_bridge is not None:
                    hidden_bridges[triple] = h_bridge

        supervision: dict[tuple[str, str, str], NegativeSampleResult] = {}
        if negative_sampler is not None:
            for triple, h_idx in hidden_edges.items():
                res = negative_sampler.sample(
                    observed_data,
                    triple,
                    num_negatives=int(round(h_idx.size(1) * negative_ratio)),
                    seed=seed,
                )
                edge_label_index = torch.cat([h_idx, res.neg_edge_index], dim=1)
                edge_label = torch.cat(
                    [
                        torch.ones(h_idx.size(1), dtype=torch.float32),
                        torch.zeros(res.neg_edge_index.size(1), dtype=torch.float32),
                    ],
                    dim=0,
                )
                supervision[triple] = NegativeSampleResult(
                    edge_type=triple,
                    pos_edge_index=h_idx,
                    neg_edge_index=res.neg_edge_index,
                    edge_label_index=edge_label_index,
                    edge_label=edge_label,
                    stratum_counts=res.stratum_counts,
                )

        return MaskingResult(
            condition=self.condition,
            observed_data=observed_data,
            hidden_edges=hidden_edges,
            hidden_edge_attributes=hidden_attrs,
            hidden_edge_is_bridge=hidden_bridges,
            supervision_by_edge_type=supervision,
            provenance={
                "condition": self.condition.value,
                "total_bridges_masked": total_bridges_masked,
                "bridge_mask_ratio": self.bridge_mask_ratio,
                "seed": seed,
            },
        )

    def _should_mask_graph_edge(
        self,
        edge: GraphEdge,
        _graph: IAMGraph,
        _seed: int,
    ) -> bool:
        return bool(edge.is_bridge)


class CrossAccountMaskingOperator(BaseMaskingOperator):
    """Condition B: Selective masking of cross-account trust boundaries and siloed accounts.

    Simulates multi-account enterprise silos where the auditor lacks read permissions for secondary
    accounts. Masks trust relations crossing into or out of external accounts.
    """

    def __init__(self, target_account_id: str | None = None) -> None:
        super().__init__(MaskingCondition.CONDITION_B_CROSS_ACCOUNT)
        self.target_account_id: str | None = target_account_id

    def apply(
        self,
        data: HeteroData,
        seed: int = 42,
        negative_sampler: StratifiedNegativeSampler | None = None,
        negative_ratio: float = 1.0,
    ) -> MaskingResult:
        observed_data = data.clone()
        primary_account = getattr(data, "account_id", "123456789012")

        hidden_edges: dict[tuple[str, str, str], torch.Tensor] = {}
        hidden_attrs: dict[tuple[str, str, str], torch.Tensor] = {}
        hidden_bridges: dict[tuple[str, str, str], torch.Tensor] = {}

        for triple in list(observed_data.edge_types):
            if is_reverse_edge_type(triple):
                continue
            src_type, rel, dst_type = triple

            edge_index = observed_data[triple].edge_index
            num_edges = edge_index.size(1)
            if num_edges == 0:
                continue

            src_arns = getattr(observed_data[src_type], "arns", [])
            dst_arns = getattr(observed_data[dst_type], "arns", [])

            # Check if endpoints cross account boundaries or belong to target silo
            remove_flags: list[bool] = []
            for i in range(num_edges):
                u_idx = int(edge_index[0, i])
                v_idx = int(edge_index[1, i])

                u_arn = src_arns[u_idx] if u_idx < len(src_arns) else ""
                v_arn = dst_arns[v_idx] if v_idx < len(dst_arns) else ""

                # Extract account digits from ARN if present
                u_acct = self._extract_account_from_arn(u_arn)
                v_acct = self._extract_account_from_arn(v_arn)

                is_cross_account = False
                if self.target_account_id:
                    is_cross_account = (
                        u_acct == self.target_account_id or v_acct == self.target_account_id
                    )
                else:
                    # Relational cross-boundary: endpoints have differing accounts or non-primary account
                    if (
                        u_acct
                        and v_acct
                        and u_acct != v_acct
                        or u_acct
                        and u_acct != primary_account
                        or v_acct
                        and v_acct != primary_account
                        or rel == EdgeRelation.ASSUMES_ROLE.value
                        and ("external" in u_arn.lower() or "partner" in u_arn.lower())
                    ):
                        is_cross_account = True

                remove_flags.append(is_cross_account)

            keep_mask = ~torch.tensor(remove_flags, dtype=torch.bool)
            if (~keep_mask).any():
                h_idx, h_attr, h_bridge = self._slice_hetero_edge_store(
                    observed_data, triple, keep_mask
                )
                hidden_edges[triple] = h_idx
                if h_attr is not None:
                    hidden_attrs[triple] = h_attr
                if h_bridge is not None:
                    hidden_bridges[triple] = h_bridge

        supervision: dict[tuple[str, str, str], NegativeSampleResult] = {}
        if negative_sampler is not None:
            for triple, h_idx in hidden_edges.items():
                res = negative_sampler.sample(
                    observed_data,
                    triple,
                    num_negatives=int(round(h_idx.size(1) * negative_ratio)),
                    seed=seed,
                )
                edge_label_index = torch.cat([h_idx, res.neg_edge_index], dim=1)
                edge_label = torch.cat(
                    [
                        torch.ones(h_idx.size(1), dtype=torch.float32),
                        torch.zeros(res.neg_edge_index.size(1), dtype=torch.float32),
                    ],
                    dim=0,
                )
                supervision[triple] = NegativeSampleResult(
                    edge_type=triple,
                    pos_edge_index=h_idx,
                    neg_edge_index=res.neg_edge_index,
                    edge_label_index=edge_label_index,
                    edge_label=edge_label,
                    stratum_counts=res.stratum_counts,
                )

        return MaskingResult(
            condition=self.condition,
            observed_data=observed_data,
            hidden_edges=hidden_edges,
            hidden_edge_attributes=hidden_attrs,
            hidden_edge_is_bridge=hidden_bridges,
            supervision_by_edge_type=supervision,
            provenance={
                "condition": self.condition.value,
                "target_account_id": self.target_account_id,
                "primary_account": primary_account,
                "seed": seed,
            },
        )

    @staticmethod
    def _extract_account_from_arn(arn: str) -> str:
        parts = arn.split(":")
        if len(parts) >= 5 and parts[4].isdigit() and len(parts[4]) == 12:
            return parts[4]
        return ""


class FederatedIdPMaskingOperator(BaseMaskingOperator):
    """Condition C: Selective masking of external identity-to-role delegation entrypoints.

    Simulates federated SaaS / OIDC / SAML entrypoints where internal trust policies are visible
    in the cloud tenant, but external IdP directory user-to-group mappings are unobservable.
    """

    def __init__(self, mask_entrypoint_users: bool = True) -> None:
        super().__init__(MaskingCondition.CONDITION_C_FEDERATED_IDP)
        self.mask_entrypoint_users: bool = mask_entrypoint_users

    def apply(
        self,
        data: HeteroData,
        seed: int = 42,
        negative_sampler: StratifiedNegativeSampler | None = None,
        negative_ratio: float = 1.0,
    ) -> MaskingResult:
        observed_data = data.clone()

        hidden_edges: dict[tuple[str, str, str], torch.Tensor] = {}
        hidden_attrs: dict[tuple[str, str, str], torch.Tensor] = {}
        hidden_bridges: dict[tuple[str, str, str], torch.Tensor] = {}

        # Target entrypoint relations: User -> Role (AssumesRole) and User -> Group (MemberOf)
        target_triples = (
            ("User", "AssumesRole", "Role"),
            ("User", "MemberOf", "Group"),
        )

        for triple in target_triples:
            if triple not in observed_data.edge_types:
                continue

            edge_index = observed_data[triple].edge_index
            num_edges = edge_index.size(1)
            if num_edges == 0:
                continue

            # In Condition C, mask entrypoints (User -> Role / Group) for designated external identities
            # Stochastically or structurally mask 50% of entrypoint bindings
            rng = np.random.default_rng(seed)
            keep_mask = torch.from_numpy(rng.random(num_edges) >= 0.50)

            if (~keep_mask).any():
                h_idx, h_attr, h_bridge = self._slice_hetero_edge_store(
                    observed_data, triple, keep_mask
                )
                hidden_edges[triple] = h_idx
                if h_attr is not None:
                    hidden_attrs[triple] = h_attr
                if h_bridge is not None:
                    hidden_bridges[triple] = h_bridge

        supervision: dict[tuple[str, str, str], NegativeSampleResult] = {}
        if negative_sampler is not None:
            for triple, h_idx in hidden_edges.items():
                res = negative_sampler.sample(
                    observed_data,
                    triple,
                    num_negatives=int(round(h_idx.size(1) * negative_ratio)),
                    seed=seed,
                )
                edge_label_index = torch.cat([h_idx, res.neg_edge_index], dim=1)
                edge_label = torch.cat(
                    [
                        torch.ones(h_idx.size(1), dtype=torch.float32),
                        torch.zeros(res.neg_edge_index.size(1), dtype=torch.float32),
                    ],
                    dim=0,
                )
                supervision[triple] = NegativeSampleResult(
                    edge_type=triple,
                    pos_edge_index=h_idx,
                    neg_edge_index=res.neg_edge_index,
                    edge_label_index=edge_label_index,
                    edge_label=edge_label,
                    stratum_counts=res.stratum_counts,
                )

        return MaskingResult(
            condition=self.condition,
            observed_data=observed_data,
            hidden_edges=hidden_edges,
            hidden_edge_attributes=hidden_attrs,
            hidden_edge_is_bridge=hidden_bridges,
            supervision_by_edge_type=supervision,
            provenance={
                "condition": self.condition.value,
                "mask_entrypoint_users": self.mask_entrypoint_users,
                "seed": seed,
            },
        )


class EphemeralSTSMaskingOperator(BaseMaskingOperator):
    """Condition D: Selective masking of runtime ephemeral STS session delegation edges.

    Hides runtime dynamic role chaining (Role -> Role via AssumesRole) and instance passes
    while retaining static policy attachments and resource targets.
    """

    def __init__(self, mask_role_chaining: bool = True) -> None:
        super().__init__(MaskingCondition.CONDITION_D_EPHEMERAL_STS)
        self.mask_role_chaining: bool = mask_role_chaining

    def apply(
        self,
        data: HeteroData,
        seed: int = 42,
        negative_sampler: StratifiedNegativeSampler | None = None,
        negative_ratio: float = 1.0,
    ) -> MaskingResult:
        observed_data = data.clone()

        hidden_edges: dict[tuple[str, str, str], torch.Tensor] = {}
        hidden_attrs: dict[tuple[str, str, str], torch.Tensor] = {}
        hidden_bridges: dict[tuple[str, str, str], torch.Tensor] = {}

        # Ephemeral session relations: Role -> Role (AssumesRole) and Role -> Role (PassesTo)
        ephemeral_triples = (
            ("Role", "AssumesRole", "Role"),
            ("Role", "PassesTo", "Role"),
        )

        for triple in ephemeral_triples:
            if triple not in observed_data.edge_types:
                continue

            edge_index = observed_data[triple].edge_index
            num_edges = edge_index.size(1)
            if num_edges == 0:
                continue

            # Excise dynamic role-chaining relations
            keep_mask = torch.zeros(num_edges, dtype=torch.bool)
            h_idx, h_attr, h_bridge = self._slice_hetero_edge_store(
                observed_data, triple, keep_mask
            )
            hidden_edges[triple] = h_idx
            if h_attr is not None:
                hidden_attrs[triple] = h_attr
            if h_bridge is not None:
                hidden_bridges[triple] = h_bridge

        supervision: dict[tuple[str, str, str], NegativeSampleResult] = {}
        if negative_sampler is not None:
            for triple, h_idx in hidden_edges.items():
                res = negative_sampler.sample(
                    observed_data,
                    triple,
                    num_negatives=int(round(h_idx.size(1) * negative_ratio)),
                    seed=seed,
                )
                edge_label_index = torch.cat([h_idx, res.neg_edge_index], dim=1)
                edge_label = torch.cat(
                    [
                        torch.ones(h_idx.size(1), dtype=torch.float32),
                        torch.zeros(res.neg_edge_index.size(1), dtype=torch.float32),
                    ],
                    dim=0,
                )
                supervision[triple] = NegativeSampleResult(
                    edge_type=triple,
                    pos_edge_index=h_idx,
                    neg_edge_index=res.neg_edge_index,
                    edge_label_index=edge_label_index,
                    edge_label=edge_label,
                    stratum_counts=res.stratum_counts,
                )

        return MaskingResult(
            condition=self.condition,
            observed_data=observed_data,
            hidden_edges=hidden_edges,
            hidden_edge_attributes=hidden_attrs,
            hidden_edge_is_bridge=hidden_bridges,
            supervision_by_edge_type=supervision,
            provenance={
                "condition": self.condition.value,
                "mask_role_chaining": self.mask_role_chaining,
                "seed": seed,
            },
        )


class StructuredMaskingSuite:
    """Factory and orchestration suite for all 5 enterprise masking conditions."""

    @staticmethod
    def get_operator(
        condition: MaskingCondition | str,
        **kwargs: Any,
    ) -> BaseMaskingOperator:
        """Instantiate the masking operator for a designated condition."""
        if isinstance(condition, str):
            key = condition.strip().upper()
            shorthand = {
                "A": MaskingCondition.CONDITION_A_RANDOM,
                "E": MaskingCondition.CONDITION_E_ADVERSARIAL_BRIDGE,
                "B": MaskingCondition.CONDITION_B_CROSS_ACCOUNT,
                "C": MaskingCondition.CONDITION_C_FEDERATED_IDP,
                "D": MaskingCondition.CONDITION_D_EPHEMERAL_STS,
            }
            cond = shorthand.get(key)
            if cond is None:
                cond = MaskingCondition(condition)
        else:
            cond = condition

        if cond == MaskingCondition.CONDITION_A_RANDOM:
            return RandomMaskingOperator(**kwargs)
        elif cond == MaskingCondition.CONDITION_E_ADVERSARIAL_BRIDGE:
            return AdversarialBridgeMaskingOperator(**kwargs)
        elif cond == MaskingCondition.CONDITION_B_CROSS_ACCOUNT:
            return CrossAccountMaskingOperator(**kwargs)
        elif cond == MaskingCondition.CONDITION_C_FEDERATED_IDP:
            return FederatedIdPMaskingOperator(**kwargs)
        elif cond == MaskingCondition.CONDITION_D_EPHEMERAL_STS:
            return EphemeralSTSMaskingOperator(**kwargs)
        raise ValueError(f"Unknown masking condition: {cond}")

    @classmethod
    def apply_condition(
        cls,
        data: HeteroData,
        condition: MaskingCondition | str,
        seed: int = 42,
        negative_sampler: StratifiedNegativeSampler | None = None,
        negative_ratio: float = 1.0,
        **kwargs: Any,
    ) -> MaskingResult:
        """Apply a specified masking condition to HeteroData in a single call."""
        operator = cls.get_operator(condition, **kwargs)
        return operator.apply(
            data=data,
            seed=seed,
            negative_sampler=negative_sampler,
            negative_ratio=negative_ratio,
        )

    @staticmethod
    def verify_reachability_severed(
        original_graph: IAMGraph,
        masked_result: MaskingResult,
        source_id: str,
        target_id: str,
    ) -> bool:
        """Verify reachability from source_id to target_id in masked observed graph."""
        nx_g = nx.DiGraph()

        # Build DiGraph from observed_data
        for node_type in masked_result.observed_data.node_types:
            n_ids = getattr(masked_result.observed_data[node_type], "node_ids", [])
            for nid in n_ids:
                nx_g.add_node(nid)

        for edge_type in masked_result.observed_data.edge_types:
            if is_reverse_edge_type(edge_type):
                continue
            src_type, _rel, dst_type = edge_type
            src_ids = getattr(masked_result.observed_data[src_type], "node_ids", [])
            dst_ids = getattr(masked_result.observed_data[dst_type], "node_ids", [])

            edge_index = masked_result.observed_data[edge_type].edge_index
            for i in range(edge_index.size(1)):
                u = src_ids[int(edge_index[0, i])]
                v = dst_ids[int(edge_index[1, i])]
                nx_g.add_edge(u, v)

        has_observed_path: bool = nx.has_path(nx_g, source_id, target_id)
        has_orig_path: bool = nx.has_path(original_graph.nx_graph, source_id, target_id)

        # Successfully severed if path existed originally but is gone in observed graph
        return bool(has_orig_path and not has_observed_path)
