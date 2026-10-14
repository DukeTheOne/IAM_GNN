"""Three-level completion-set symbolic safety verifier and operational replay engine.

Implements Phase 3 Week 8 (Step 8.2):
- Evaluates policy repairs across the 3-level verification boundary:
    Level 1: Observed graph G_o ⊕ P (static traversal)
    Level 2: Completion set { (G_o ∪ H_i) ⊕ P }_{i=1}^M (latent Monte Carlo reachability)
    Level 3: Ground-truth graph G* ⊕ P (synthetic counterfactual verification)
- Detects misleading safety states where Level 1 reports safe (0.0) but latent reachability remains.
- Escalation side-effect audit: verifies that patch introduces zero new (s', t') attack vectors.
- Operational workload replay: audits 100% preservation on T_1 and measures breakage rate on T_2 holdout.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from enum import Enum
from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from iam.generator.graph import (
    IAMGraph,
    NodeType,
)
from iam.optimizer.repair import (
    WorkloadConformanceChecker,
)
from iam.optimizer.sampler import LatentCompletion
from iam.optimizer.transformations import PolicyTransformation
from iam.parser.capability import CapabilityModel, load_default_capability_model
from iam.parser.evaluator import AuthRequest
from iam.verifier.workload import WorkloadEvent

logger = logging.getLogger(__name__)


class VerificationLevel(str, Enum):
    """The three conceptual levels of authorization safety verification."""

    OBSERVED_GRAPH = "observed_graph"  # Level 1: G_o ⊕ P
    COMPLETION_SET = "completion_set"  # Level 2: { (G_o ∪ H_i) ⊕ P }
    GROUND_TRUTH = "ground_truth"  # Level 3: G* ⊕ P (synthetic benchmark only)


class SafetyStatus(str, Enum):
    """Overall safety verdict produced by the completion-set verifier."""

    SAFE = "safe"  # Residual risk <= delta across all evaluated levels
    UNSAFE = "unsafe"  # Attack path remains reachable
    MISLEADING_SAFE = "misleading_safe"  # Level 1 reports safe, but Level 2 or Level 3 is unsafe


class SideEffectResult(BaseModel):
    """Result of checking whether a patch introduced new privilege escalation side-effects."""

    model_config = ConfigDict(frozen=True)

    has_side_effects: bool = Field(description="True if any new escalation pairs were discovered")
    new_escalation_pairs: list[tuple[str, str]] = Field(
        default_factory=list,
        description="List of (source_id, target_id) pairs newly reachable after patch",
    )
    num_pairs_checked: int = Field(
        default=0, description="Total number of principal-target pairs evaluated"
    )
    execution_time_sec: float = Field(default=0.0, description="Verification latency in seconds")
    details: str = Field(default="", description="Explanatory diagnostics")


class WorkloadReplayResult(BaseModel):
    """Result of replaying historical (T_1) or holdout (T_2) workload events against a repaired graph."""

    model_config = ConfigDict(frozen=True)

    window_name: str = Field(description="Name of the replayed window ('T1' or 'T2')")
    total_requests: int = Field(description="Total number of requests replayed")
    allowed_requests: int = Field(description="Count of requests successfully authorized")
    denied_requests: int = Field(description="Count of requests denied by the repaired policy set")
    breakage_rate: float = Field(description="Fraction of requests denied (denied / total)")
    denied_events: list[dict[str, Any]] = Field(
        default_factory=list, description="Detailed records of requests denied by the patch"
    )
    is_conforming: bool = Field(description="True if zero requests were denied (100% conformance)")

    def summary_dict(self) -> dict[str, Any]:
        """Serialize workload replay summary to dictionary."""
        return {
            "window_name": self.window_name,
            "total_requests": self.total_requests,
            "allowed_requests": self.allowed_requests,
            "denied_requests": self.denied_requests,
            "breakage_rate": round(self.breakage_rate, 4),
            "is_conforming": self.is_conforming,
        }


class VerificationReport(BaseModel):
    """Comprehensive multi-level safety and operational impact report for an IAM patch."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(description="Attacker entrypoint principal node ID")
    target_id: str = Field(description="High-value target asset node ID")
    level1_risk_observed: float = Field(description="Attack reachability on G_o ⊕ P (Level 1)")
    level2_risk_completion_set: float = Field(
        description="Attack reachability across completions (G_o ∪ H_i) ⊕ P (Level 2)"
    )
    level3_risk_ground_truth: float | None = Field(
        default=None, description="Attack reachability on G* ⊕ P (Level 3, synthetic only)"
    )
    level1_safe: bool = Field(description="True if Level 1 attack risk <= threshold")
    level2_safe: bool = Field(description="True if Level 2 attack risk <= threshold")
    level3_safe: bool | None = Field(
        default=None, description="True if Level 3 attack risk <= threshold (if evaluated)"
    )
    overall_status: SafetyStatus = Field(description="Overall safety classification")
    is_misleading: bool = Field(
        description="True if patch passes Level 1 (safe on G_o) but fails Level 2 or Level 3"
    )
    side_effects: SideEffectResult = Field(
        description="Audit result for unintended escalation side-effects"
    )
    t1_replay: WorkloadReplayResult | None = Field(
        default=None, description="Replay conformance result on observation window T_1"
    )
    t2_replay: WorkloadReplayResult | None = Field(
        default=None, description="Replay breakage result on holdout evaluation window T_2"
    )
    execution_time_sec: float = Field(
        default=0.0, description="Total verification latency in seconds"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional context and parameters"
    )

    @property
    def residual_risk_gap(self) -> float:
        """Discrepancy between completion set / ground truth risk and observed graph risk."""
        ref_risk = (
            self.level3_risk_ground_truth
            if self.level3_risk_ground_truth is not None
            else self.level2_risk_completion_set
        )
        return max(0.0, ref_risk - self.level1_risk_observed)

    def summary(self) -> str:
        """Generate human-readable summary of the verification report."""
        status_label = self.overall_status.value.upper()
        l3_str = (
            f"{self.level3_risk_ground_truth:.2%}"
            if self.level3_risk_ground_truth is not None
            else "N/A"
        )
        t1_str = (
            f"{self.t1_replay.allowed_requests}/{self.t1_replay.total_requests} (100% conforming)"
            if self.t1_replay and self.t1_replay.is_conforming
            else "N/A"
        )
        t2_str = (
            f"Breakage: {self.t2_replay.breakage_rate:.2%} ({self.t2_replay.denied_requests} denied)"
            if self.t2_replay
            else "N/A"
        )

        return (
            f"VerificationReport [{status_label}]:\n"
            f"  - Target Pair: {self.source_id} -> {self.target_id}\n"
            f"  - Level 1 (Observed G_o ⊕ P): Risk = {self.level1_risk_observed:.2%} (Safe: {self.level1_safe})\n"
            f"  - Level 2 (Completion Set): Risk = {self.level2_risk_completion_set:.2%} (Safe: {self.level2_safe})\n"
            f"  - Level 3 (Ground Truth G* ⊕ P): Risk = {l3_str} (Safe: {self.level3_safe})\n"
            f"  - Misleading Safe on G_o: {self.is_misleading} (Gap: {self.residual_risk_gap:.2%})\n"
            f"  - Side Effects: {self.side_effects.has_side_effects} ({len(self.side_effects.new_escalation_pairs)} new pairs)\n"
            f"  - T_1 Replay: {t1_str}\n"
            f"  - T_2 Replay: {t2_str}\n"
            f"  - Verification Latency: {self.execution_time_sec:.4f}s"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize verification report to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "level1_risk_observed": round(self.level1_risk_observed, 4),
            "level2_risk_completion_set": round(self.level2_risk_completion_set, 4),
            "level3_risk_ground_truth": (
                round(self.level3_risk_ground_truth, 4)
                if self.level3_risk_ground_truth is not None
                else None
            ),
            "level1_safe": self.level1_safe,
            "level2_safe": self.level2_safe,
            "level3_safe": self.level3_safe,
            "overall_status": self.overall_status.value,
            "is_misleading": self.is_misleading,
            "residual_risk_gap": round(self.residual_risk_gap, 4),
            "side_effects": self.side_effects.model_dump(),
            "t1_replay": self.t1_replay.summary_dict() if self.t1_replay else None,
            "t2_replay": self.t2_replay.summary_dict() if self.t2_replay else None,
            "execution_time_sec": round(self.execution_time_sec, 4),
            "metadata": self.metadata,
        }


class CompletionSetVerifier:
    """Three-level symbolic safety verifier with operational replay and side-effect checking.

    Provides formal mathematical verification that repaired IAM policy sets eliminate
    privilege escalation reachability across plausible latent completion sets without
    introducing new attack vectors or violating operational workloads.
    """

    def __init__(self, capability_model: CapabilityModel | None = None) -> None:
        """Initialize the completion-set verifier."""
        self.capability_model: CapabilityModel = (
            capability_model if capability_model is not None else load_default_capability_model()
        )
        self.conformance_checker: WorkloadConformanceChecker = WorkloadConformanceChecker(
            self.capability_model
        )

    def _resolve_node_id(self, graph: IAMGraph, identifier: str) -> str:
        """Resolve node ID from node ID or ARN."""
        node = graph.get_node(identifier) or graph.get_node_by_arn(identifier)
        if node is not None:
            return node.id
        for n in graph.get_nodes():
            if n.arn.lower() == identifier.lower():
                return n.id
        return identifier

    def replay_workload(
        self,
        graph: IAMGraph,
        workload: Sequence[AuthRequest | WorkloadEvent],
        window_name: str = "Workload",
    ) -> WorkloadReplayResult:
        """Replay authorized activity events against a repaired graph and quantify breakage.

        Args:
            graph: The repaired IAM authorization graph (G_o ⊕ P).
            workload: Sequence of AuthRequest or WorkloadEvent objects.
            window_name: Label identifying the window ('T1' or 'T2').

        Returns:
            WorkloadReplayResult documenting allowed, denied, and breakage metrics.
        """
        if not workload:
            return WorkloadReplayResult(
                window_name=window_name,
                total_requests=0,
                allowed_requests=0,
                denied_requests=0,
                breakage_rate=0.0,
                denied_events=[],
                is_conforming=True,
            )

        allowed_count = 0
        denied_records: list[dict[str, Any]] = []

        for item in workload:
            req = item.to_auth_request() if isinstance(item, WorkloadEvent) else item
            eval_res = self.conformance_checker.evaluate_request(graph, req)
            if eval_res.is_allowed:
                allowed_count += 1
            else:
                record: dict[str, Any] = {
                    "principal_arn": req.principal_arn,
                    "action": req.action,
                    "resource_arn": req.resource_arn,
                    "decision": eval_res.decision.value,
                }
                if isinstance(item, WorkloadEvent):
                    record["event_id"] = item.event_id
                    record["day_offset"] = item.day_offset
                    record["is_drift"] = item.is_drift
                    record["drift_type"] = item.drift_type.value if item.drift_type else None
                denied_records.append(record)

        total = len(workload)
        denied_count = len(denied_records)
        breakage = denied_count / float(total) if total > 0 else 0.0

        return WorkloadReplayResult(
            window_name=window_name,
            total_requests=total,
            allowed_requests=allowed_count,
            denied_requests=denied_count,
            breakage_rate=breakage,
            denied_events=denied_records,
            is_conforming=(denied_count == 0),
        )

    def check_side_effects(
        self,
        base_graph: IAMGraph,
        repaired_graph: IAMGraph,
        target_hv_resources: Sequence[str] | None = None,
    ) -> SideEffectResult:
        """Audit for unintended privilege escalation side-effects introduced by a patch.

        Verifies that no (s', t') principal-to-high-value-target reachability path
        exists in repaired_graph that did not already exist in base_graph.
        """
        start_time = time.perf_counter()
        base_nx = base_graph.nx_graph
        repaired_nx = repaired_graph.nx_graph

        # Collect candidate attacker principals (non-admin Users and Roles)
        principals = [
            n.id
            for n in repaired_graph.get_nodes()
            if n.node_type in (NodeType.USER, NodeType.ROLE) and not n.is_admin
        ]

        # Collect high-value targets (Resources or Admin Roles)
        if target_hv_resources is not None:
            targets = [self._resolve_node_id(repaired_graph, t) for t in target_hv_resources]
        else:
            targets = [
                n.id
                for n in repaired_graph.get_nodes()
                if n.is_high_value or (n.node_type == NodeType.ROLE and n.is_admin)
            ]

        new_escalation_pairs: list[tuple[str, str]] = []
        pairs_checked = 0

        for s in principals:
            for t in targets:
                if s == t:
                    continue
                pairs_checked += 1
                had_path_base = (
                    nx.has_path(base_nx, s, t)
                    if base_nx.has_node(s) and base_nx.has_node(t)
                    else False
                )
                has_path_repaired = (
                    nx.has_path(repaired_nx, s, t)
                    if repaired_nx.has_node(s) and repaired_nx.has_node(t)
                    else False
                )

                if has_path_repaired and not had_path_base:
                    # An edge/policy was somehow loosened to allow a new attack path!
                    new_escalation_pairs.append((s, t))

        elapsed = time.perf_counter() - start_time
        has_side_effects = len(new_escalation_pairs) > 0
        details = (
            f"Audited {pairs_checked} (s, t) pairs: {len(new_escalation_pairs)} new escalation vectors found."
            if has_side_effects
            else f"Audited {pairs_checked} (s, t) pairs: clean 0 side-effects verified."
        )

        return SideEffectResult(
            has_side_effects=has_side_effects,
            new_escalation_pairs=new_escalation_pairs,
            num_pairs_checked=pairs_checked,
            execution_time_sec=elapsed,
            details=details,
        )

    def verify_patch(
        self,
        observed_graph: IAMGraph,
        repaired_graph: IAMGraph,
        source_id: str,
        target_id: str,
        applied_transformations: Sequence[PolicyTransformation] | None = None,
        completions: Sequence[LatentCompletion] | None = None,
        ground_truth_graph: IAMGraph | None = None,
        t1_workload: Sequence[AuthRequest | WorkloadEvent] | None = None,
        t2_workload: Sequence[AuthRequest | WorkloadEvent] | None = None,
        check_side_effects: bool = True,
        risk_threshold: float = 0.0,
    ) -> VerificationReport:
        """Perform comprehensive three-level symbolic verification and workload replay.

        Args:
            observed_graph: Baseline partially observed authorization graph G_o.
            repaired_graph: Repaired authorization graph (G_o ⊕ P).
            source_id: Attacker entrypoint principal node ID or ARN.
            target_id: High-value asset or administrative target node ID or ARN.
            applied_transformations: Transformations comprising the synthesized patch P.
            completions: Monte Carlo latent completions {H_1, ..., H_M}.
            ground_truth_graph: Optional full ground-truth graph G* (synthetic evaluation only).
            t1_workload: Historical baseline activity trace D^{T_1}.
            t2_workload: Holdout evaluation activity trace D^{T_2} with operational drift.
            check_side_effects: Whether to audit for unintended escalation side-effects.
            risk_threshold: Maximum allowable residual attack risk delta (default: 0.0).

        Returns:
            VerificationReport containing multi-level safety verdicts, replay metrics, and diagnostics.
        """
        start_time = time.perf_counter()
        _ = applied_transformations
        s_id = self._resolve_node_id(observed_graph, source_id)
        t_id = self._resolve_node_id(observed_graph, target_id)

        # ----------------------------------------------------------------------
        # 1. Level 1 Verification: Observed Graph G_o ⊕ P
        # ----------------------------------------------------------------------
        repaired_nx = repaired_graph.nx_graph
        l1_has_path = (
            nx.has_path(repaired_nx, s_id, t_id)
            if repaired_nx.has_node(s_id) and repaired_nx.has_node(t_id)
            else False
        )
        level1_risk = 1.0 if l1_has_path else 0.0
        level1_safe = level1_risk <= risk_threshold

        # ----------------------------------------------------------------------
        # 2. Level 2 Verification: Completion Set { (G_o ∪ H_i) ⊕ P }
        # ----------------------------------------------------------------------
        comp_list = list(completions or [])
        if not comp_list:
            level2_risk = level1_risk
            level2_safe = level1_safe
        else:
            comp_reachable_count = 0
            for comp in comp_list:
                # Instantiate (G_o ⊕ P) ∪ H_i
                comp_nx = repaired_graph.nx_graph.copy()
                for ce in comp.sampled_edges:
                    comp_nx.add_edge(ce.source, ce.target, key=ce.relation.value)
                if nx.has_path(comp_nx, s_id, t_id):
                    comp_reachable_count += 1
            level2_risk = float(comp_reachable_count / len(comp_list))
            level2_safe = level2_risk <= risk_threshold

        # ----------------------------------------------------------------------
        # 3. Level 3 Verification: Ground Truth G* ⊕ P (Synthetic Evaluation Only)
        # ----------------------------------------------------------------------
        level3_risk: float | None = None
        level3_safe: bool | None = None

        if ground_truth_graph is not None:
            # Apply synthesized patch transformations P to the ground-truth graph G*
            gt_repaired = ground_truth_graph.clone()

            # 1. Transfer any modified or new policy documents from repaired_graph to gt_repaired
            for node in repaired_graph.get_nodes():
                if node.node_type == NodeType.POLICY and node.policy_document is not None:
                    obs_node = observed_graph.get_node(node.id)
                    if obs_node is None or obs_node.policy_document != node.policy_document:
                        gt_repaired.add_node(node.model_copy(deep=True))

            # 2. Synchronize edge removals: if an edge in G_o was severed/modified during repair,
            # reflect the cut in gt_repaired
            for u, v, k in list(gt_repaired.nx_graph.edges(keys=True)):
                if observed_graph.nx_graph.has_edge(
                    u, v, key=k
                ) and not repaired_graph.nx_graph.has_edge(u, v, key=k):
                    gt_repaired.nx_graph.remove_edge(u, v, key=k)

            gt_nx = gt_repaired.nx_graph
            l3_has_path = (
                nx.has_path(gt_nx, s_id, t_id)
                if gt_nx.has_node(s_id) and gt_nx.has_node(t_id)
                else False
            )
            level3_risk = 1.0 if l3_has_path else 0.0
            level3_safe = level3_risk <= risk_threshold

        # ----------------------------------------------------------------------
        # 4. Overall Safety Verdict & Misleading Safe Detection
        # ----------------------------------------------------------------------
        is_misleading = False
        overall_status = SafetyStatus.SAFE

        if level1_safe:
            if not level2_safe or (level3_safe is not None and not level3_safe):
                is_misleading = True
                overall_status = SafetyStatus.MISLEADING_SAFE
            else:
                overall_status = SafetyStatus.SAFE
        else:
            overall_status = SafetyStatus.UNSAFE

        # ----------------------------------------------------------------------
        # 5. Escalation Side-Effect Audit
        # ----------------------------------------------------------------------
        if check_side_effects:
            side_effects_res = self.check_side_effects(observed_graph, repaired_graph)
        else:
            side_effects_res = SideEffectResult(
                has_side_effects=False,
                new_escalation_pairs=[],
                num_pairs_checked=0,
                execution_time_sec=0.0,
                details="Side-effect checking disabled.",
            )

        # ----------------------------------------------------------------------
        # 6. Operational Workload Replay (T_1 & T_2)
        # ----------------------------------------------------------------------
        t1_replay_res: WorkloadReplayResult | None = None
        if t1_workload is not None:
            t1_replay_res = self.replay_workload(repaired_graph, t1_workload, window_name="T1")

        t2_replay_res: WorkloadReplayResult | None = None
        if t2_workload is not None:
            t2_replay_res = self.replay_workload(repaired_graph, t2_workload, window_name="T2")

        total_elapsed = time.perf_counter() - start_time

        return VerificationReport(
            source_id=s_id,
            target_id=t_id,
            level1_risk_observed=level1_risk,
            level2_risk_completion_set=level2_risk,
            level3_risk_ground_truth=level3_risk,
            level1_safe=level1_safe,
            level2_safe=level2_safe,
            level3_safe=level3_safe,
            overall_status=overall_status,
            is_misleading=is_misleading,
            side_effects=side_effects_res,
            t1_replay=t1_replay_res,
            t2_replay=t2_replay_res,
            execution_time_sec=total_elapsed,
            metadata={
                "num_completions_evaluated": len(comp_list),
                "has_ground_truth": ground_truth_graph is not None,
                "risk_threshold": risk_threshold,
            },
        )
