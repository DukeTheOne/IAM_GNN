"""Stratified negative edge sampler for inductive multigraph link prediction.

Generates type-safe, stratified negative edges across intra-department,
inter-department, and privilege-boundary strata for enterprise IAM link prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import HeteroData

from iam.models.converter import is_reverse_edge_type


@dataclass
class NegativeSampleResult:
    """Encapsulates positive and negative edge evaluation tensors for link prediction."""

    edge_type: tuple[str, str, str]
    pos_edge_index: torch.Tensor
    neg_edge_index: torch.Tensor
    edge_label_index: torch.Tensor
    edge_label: torch.Tensor
    stratum_counts: dict[str, int]

    @property
    def num_pos(self) -> int:
        """Number of ground-truth positive edges."""
        return int(self.pos_edge_index.size(1)) if self.pos_edge_index.numel() > 0 else 0

    @property
    def num_neg(self) -> int:
        """Number of sampled negative edges."""
        return int(self.neg_edge_index.size(1)) if self.neg_edge_index.numel() > 0 else 0

    @property
    def negative_ratio(self) -> float:
        """Observed negative-to-positive edge ratio."""
        return (self.num_neg / self.num_pos) if self.num_pos > 0 else 0.0


class StratifiedNegativeSampler:
    """Samples stratified negative candidate edges for multi-relational authorization graphs.

    Partitions negative edge sampling into three distinct enterprise strata:
    1. Hard Intra-Department Negatives (default 50%):
       Pairs (u, v) sharing identical departmental organizational units but lacking
       authorization, forcing the model to learn fine-grained policy semantics rather
       than naive community clustering.
    2. Inter-Department Benign Negatives (default 30%):
       Pairs (u, v) across separate departments that lack valid authorization links.
    3. Privilege-Boundary Negatives (default 20%):
       Low-privilege identities paired with high-value assets / crown jewels where
       no legitimate authorization path exists.
    """

    def __init__(
        self,
        intra_dept_ratio: float = 0.50,
        inter_dept_ratio: float = 0.30,
        privilege_boundary_ratio: float = 0.20,
    ) -> None:
        """Initialize sampler with relative stratum weights."""
        total = intra_dept_ratio + inter_dept_ratio + privilege_boundary_ratio
        if total <= 0:
            raise ValueError("Stratum weights must sum to a positive value.")

        self.intra_dept_ratio: float = intra_dept_ratio / total
        self.inter_dept_ratio: float = inter_dept_ratio / total
        self.privilege_boundary_ratio: float = privilege_boundary_ratio / total

    def sample(
        self,
        data: HeteroData,
        edge_type: tuple[str, str, str],
        ratio: float = 1.0,
        num_negatives: int | None = None,
        seed: int | None = None,
    ) -> NegativeSampleResult:
        """Sample stratified negative edges for a specific relation triple.

        Args:
            data: PyG HeteroData instance containing typed node and edge stores.
            edge_type: Relational triple (src_type, rel, dst_type).
            ratio: Negative-to-positive ratio kappa (ignored if num_negatives is specified).
            num_negatives: Explicit number of negative edges to sample.
            seed: Optional seed for deterministic reproducibility.

        Returns:
            NegativeSampleResult containing typed pos_edge_index, neg_edge_index,
            concatenated edge_label_index, and binary edge_label tensors.
        """
        src_type, _rel_str, dst_type = edge_type
        rng = np.random.default_rng(seed)

        # 1. Validate node types exist in graph
        if src_type not in data.node_types or dst_type not in data.node_types:
            empty_idx = torch.empty((2, 0), dtype=torch.long)
            return NegativeSampleResult(
                edge_type=edge_type,
                pos_edge_index=empty_idx,
                neg_edge_index=empty_idx,
                edge_label_index=empty_idx,
                edge_label=torch.empty((0,), dtype=torch.float32),
                stratum_counts={
                    "intra_department": 0,
                    "inter_department": 0,
                    "privilege_boundary": 0,
                    "fallback": 0,
                },
            )

        n_src: int = self._get_num_nodes(data, src_type)
        n_dst: int = self._get_num_nodes(data, dst_type)

        if n_src == 0 or n_dst == 0:
            empty_idx = torch.empty((2, 0), dtype=torch.long)
            return NegativeSampleResult(
                edge_type=edge_type,
                pos_edge_index=empty_idx,
                neg_edge_index=empty_idx,
                edge_label_index=empty_idx,
                edge_label=torch.empty((0,), dtype=torch.float32),
                stratum_counts={
                    "intra_department": 0,
                    "inter_department": 0,
                    "privilege_boundary": 0,
                    "fallback": 0,
                },
            )

        # 2. Extract existing positive edges
        if edge_type in data.edge_types:
            pos_edge_index = data[edge_type].edge_index
            pos_set: set[tuple[int, int]] = set(
                zip(
                    pos_edge_index[0].tolist(),
                    pos_edge_index[1].tolist(),
                    strict=False,
                )
            )
        else:
            pos_edge_index = torch.empty((2, 0), dtype=torch.long)
            pos_set = set()

        num_pos = pos_edge_index.size(1)

        # 3. Determine target negative count K
        if num_negatives is not None:
            k_target = max(0, num_negatives)
        else:
            k_target = max(0, int(round(num_pos * ratio)))

        total_possible_pairs = n_src * n_dst
        max_possible_negatives = max(0, total_possible_pairs - len(pos_set))
        k_target = min(k_target, max_possible_negatives)

        if k_target == 0:
            neg_edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_label_index = pos_edge_index
            edge_label = torch.ones((num_pos,), dtype=torch.float32)
            return NegativeSampleResult(
                edge_type=edge_type,
                pos_edge_index=pos_edge_index,
                neg_edge_index=neg_edge_index,
                edge_label_index=edge_label_index,
                edge_label=edge_label,
                stratum_counts={
                    "intra_department": 0,
                    "inter_department": 0,
                    "privilege_boundary": 0,
                    "fallback": 0,
                },
            )

        # 4. Extract node-level stratification metadata
        src_depts = self._get_node_departments(data, src_type, n_src)
        dst_depts = self._get_node_departments(data, dst_type, n_dst)
        src_admin = self._get_node_admin_flags(data, src_type, n_src)
        dst_high_val = self._get_node_high_value_flags(data, dst_type, n_dst)

        # Index nodes by department
        src_by_dept: dict[str, list[int]] = {}
        for u, d in enumerate(src_depts):
            if d:
                src_by_dept.setdefault(d, []).append(u)

        dst_by_dept: dict[str, list[int]] = {}
        for v, d in enumerate(dst_depts):
            if d:
                dst_by_dept.setdefault(d, []).append(v)

        common_depts = [d for d in src_by_dept if d in dst_by_dept and dst_by_dept[d]]
        low_priv_src = [u for u in range(n_src) if not src_admin[u]]
        high_val_dst = [v for v in range(n_dst) if dst_high_val[v]]

        # Target quota allocation
        k_intra = int(round(k_target * self.intra_dept_ratio))
        k_priv = int(round(k_target * self.privilege_boundary_ratio))
        k_inter = k_target - k_intra - k_priv

        sampled_set: set[tuple[int, int]] = set()
        intra_pairs: list[tuple[int, int]] = []
        priv_pairs: list[tuple[int, int]] = []
        inter_pairs: list[tuple[int, int]] = []
        fallback_pairs: list[tuple[int, int]] = []

        # 5. Stratum 1: Hard Intra-Department Negatives
        if common_depts and k_intra > 0:
            max_attempts = k_intra * 25
            attempts = 0
            while len(intra_pairs) < k_intra and attempts < max_attempts:
                attempts += 1
                dept = str(rng.choice(common_depts))
                u_cand = int(rng.choice(src_by_dept[dept]))
                v_cand = int(rng.choice(dst_by_dept[dept]))
                pair = (u_cand, v_cand)
                if pair not in pos_set and pair not in sampled_set:
                    sampled_set.add(pair)
                    intra_pairs.append(pair)

        # 6. Stratum 3: Privilege-Boundary Negatives (Low-priv to High-value)
        if low_priv_src and high_val_dst and k_priv > 0:
            max_attempts = k_priv * 25
            attempts = 0
            while len(priv_pairs) < k_priv and attempts < max_attempts:
                attempts += 1
                u_cand = int(rng.choice(low_priv_src))
                v_cand = int(rng.choice(high_val_dst))
                pair = (u_cand, v_cand)
                if pair not in pos_set and pair not in sampled_set:
                    sampled_set.add(pair)
                    priv_pairs.append(pair)

        # 7. Stratum 2: Inter-Department Benign Negatives
        needed_inter = k_inter + (k_intra - len(intra_pairs)) + (k_priv - len(priv_pairs))
        if needed_inter > 0:
            max_attempts = needed_inter * 30
            attempts = 0
            while len(inter_pairs) < needed_inter and attempts < max_attempts:
                attempts += 1
                u_cand = int(rng.integers(0, n_src))
                v_cand = int(rng.integers(0, n_dst))
                # Inter-department criterion: different or unspecified departments
                if src_depts[u_cand] == dst_depts[v_cand] and src_depts[u_cand] != "":
                    continue
                pair = (u_cand, v_cand)
                if pair not in pos_set and pair not in sampled_set:
                    sampled_set.add(pair)
                    inter_pairs.append(pair)

        # 8. Fallback: Uniform valid negative sampling if quotas had deficits
        remaining_needed = k_target - len(sampled_set)
        if remaining_needed > 0:
            if total_possible_pairs <= 2500:
                # Small search space: enumerate all valid non-positive pairs
                all_candidates = [
                    (u, v)
                    for u in range(n_src)
                    for v in range(n_dst)
                    if (u, v) not in pos_set and (u, v) not in sampled_set
                ]
                if all_candidates:
                    chosen_idx = rng.choice(
                        len(all_candidates),
                        size=min(remaining_needed, len(all_candidates)),
                        replace=False,
                    )
                    for idx in chosen_idx:
                        pair = all_candidates[int(idx)]
                        sampled_set.add(pair)
                        fallback_pairs.append(pair)
            else:
                max_attempts = remaining_needed * 50
                attempts = 0
                while len(sampled_set) < k_target and attempts < max_attempts:
                    attempts += 1
                    u_cand = int(rng.integers(0, n_src))
                    v_cand = int(rng.integers(0, n_dst))
                    pair = (u_cand, v_cand)
                    if pair not in pos_set and pair not in sampled_set:
                        sampled_set.add(pair)
                        fallback_pairs.append(pair)

        # 9. Assemble negative edge tensors
        all_neg_pairs = intra_pairs + priv_pairs + inter_pairs + fallback_pairs
        if all_neg_pairs:
            u_neg = [p[0] for p in all_neg_pairs]
            v_neg = [p[1] for p in all_neg_pairs]
            neg_edge_index = torch.tensor([u_neg, v_neg], dtype=torch.long)
        else:
            neg_edge_index = torch.empty((2, 0), dtype=torch.long)

        num_neg = neg_edge_index.size(1)

        # 10. Concatenate into evaluation tensors
        if num_pos > 0 and num_neg > 0:
            edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            edge_label = torch.cat(
                [
                    torch.ones((num_pos,), dtype=torch.float32),
                    torch.zeros((num_neg,), dtype=torch.float32),
                ],
                dim=0,
            )
        elif num_pos > 0:
            edge_label_index = pos_edge_index
            edge_label = torch.ones((num_pos,), dtype=torch.float32)
        else:
            edge_label_index = neg_edge_index
            edge_label = torch.zeros((num_neg,), dtype=torch.float32)

        return NegativeSampleResult(
            edge_type=edge_type,
            pos_edge_index=pos_edge_index,
            neg_edge_index=neg_edge_index,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            stratum_counts={
                "intra_department": len(intra_pairs),
                "inter_department": len(inter_pairs),
                "privilege_boundary": len(priv_pairs),
                "fallback": len(fallback_pairs),
            },
        )

    def sample_all_forward_relations(
        self,
        data: HeteroData,
        ratio: float = 1.0,
        seed: int | None = None,
    ) -> dict[tuple[str, str, str], NegativeSampleResult]:
        """Sample negative edges across all active forward authorization relations in HeteroData.

        Reverse message-passing conduits (e.g. RevAssumesRole) are strictly excluded from
        evaluation scoring targets.
        """
        results: dict[tuple[str, str, str], NegativeSampleResult] = {}
        for edge_type in data.edge_types:
            if is_reverse_edge_type(edge_type):
                continue
            results[edge_type] = self.sample(data, edge_type, ratio=ratio, seed=seed)
        return results

    @staticmethod
    def _get_num_nodes(data: HeteroData, node_type: str) -> int:
        """Extract node count from node store."""
        if hasattr(data[node_type], "num_nodes") and data[node_type].num_nodes is not None:
            return int(data[node_type].num_nodes)
        if hasattr(data[node_type], "node_ids"):
            return len(data[node_type].node_ids)
        if hasattr(data[node_type], "x") and data[node_type].x is not None:
            return int(data[node_type].x.size(0))
        return 0

    @staticmethod
    def _get_node_departments(data: HeteroData, node_type: str, n_nodes: int) -> list[str]:
        """Extract departmental membership list."""
        if hasattr(data[node_type], "departments"):
            return list(data[node_type].departments)
        return [""] * n_nodes

    @staticmethod
    def _get_node_admin_flags(data: HeteroData, node_type: str, n_nodes: int) -> list[bool]:
        """Extract boolean admin status flags."""
        if hasattr(data[node_type], "is_admin"):
            t = data[node_type].is_admin
            if isinstance(t, torch.Tensor):
                return [bool(x) for x in t.tolist()]
            return [bool(x) for x in t]
        return [False] * n_nodes

    @staticmethod
    def _get_node_high_value_flags(data: HeteroData, node_type: str, n_nodes: int) -> list[bool]:
        """Extract boolean high value / crown jewel status flags."""
        if hasattr(data[node_type], "is_high_value"):
            t = data[node_type].is_high_value
            if isinstance(t, torch.Tensor):
                return [bool(x) for x in t.tolist()]
            return [bool(x) for x in t]
        return [False] * n_nodes
