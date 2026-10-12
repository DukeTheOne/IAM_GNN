"""Capability-aware greedy least-privilege repair engine for IAM authorization graphs.

Implements Phase 3 Week 7 (Step 7.3):
- CapabilityGreedyRepairEngine: Synthesizes minimal, capability-compliant IAM policy patches.
- Admissible transformation selection: resource ARN scoping vs. action subtraction governed by C(a, r, c).
- Historical workload conformance checker: guarantees 100% preservation of D^{T_1} operations.
- Monte Carlo uncertainty optimization loop: maximizes marginal attack risk reduction per repair cost unit.
- PolicyPatchResult: Complete remediation artifact containing modified ASTs, diffs, and security proofs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from iam.generator.graph import EdgeRelation, GraphNode, IAMGraph, NodeType
from iam.optimizer.pruning import PruningConfig, PruningResult
from iam.optimizer.sampler import LatentCompletion
from iam.optimizer.transformations import (
    PolicyTransformation,
    PolicyTransformationType,
    RemoveActionTransformation,
    ScopeResourceTransformation,
    generate_policy_diff,
)
from iam.parser.capability import CapabilityModel, load_default_capability_model
from iam.parser.evaluator import AuthRequest, EvalDecision, EvalResult, PolicyEvaluator
from iam.parser.schema import Effect, PolicyDocument, Statement
from iam.parser.wildcard import (
    action_matches_pattern,
    normalize_action_name,
    resource_matches_pattern,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepairConfig:
    """Configuration governing capability-aware greedy policy repair."""

    cost_scope_resource: float = 1.0
    cost_remove_action: float = 5.0
    risk_weight_lambda: float = 100.0
    residual_risk_threshold_delta: float = 0.0
    max_iterations: int = 20
    allow_action_removal: bool = True
    prefer_scoping: bool = True
    strict_baseline_conformance: bool = True
    pruning_config: PruningConfig | None = None

    def __post_init__(self) -> None:
        """Validate repair engine hyperparameters."""
        if self.cost_scope_resource <= 0.0:
            raise ValueError(
                f"cost_scope_resource must be positive, got {self.cost_scope_resource}"
            )
        if self.cost_remove_action <= 0.0:
            raise ValueError(f"cost_remove_action must be positive, got {self.cost_remove_action}")
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {self.max_iterations}")
        if self.residual_risk_threshold_delta < 0.0 or self.residual_risk_threshold_delta > 1.0:
            raise ValueError(
                f"residual_risk_threshold_delta must be in [0.0, 1.0], got {self.residual_risk_threshold_delta}"
            )


@dataclass
class PolicyPatchResult:
    """Complete remediation output produced by CapabilityGreedyRepairEngine."""

    repaired_graph: IAMGraph
    applied_transformations: list[PolicyTransformation]
    modified_policy_ids: list[str]
    modified_policy_documents: dict[str, PolicyDocument]
    policy_diffs: dict[str, str]
    initial_attack_risk: float
    final_attack_risk: float
    total_repair_cost: float
    log_conformance_verified: bool
    is_successful: bool
    execution_time_sec: float
    num_iterations: int = 0
    denied_workload_events: list[AuthRequest] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_applied_transformations(self) -> int:
        """Count of policy transformations applied in the patch."""
        return len(self.applied_transformations)

    @property
    def risk_reduction(self) -> float:
        """Absolute reduction in Monte Carlo attack probability."""
        return max(0.0, self.initial_attack_risk - self.final_attack_risk)

    def get_diff(self, policy_id: str) -> str:
        """Retrieve unified diff for a specific modified policy."""
        return self.policy_diffs.get(policy_id, "")

    def summary(self) -> str:
        """Human-readable remediation summary."""
        status = "SUCCESS" if self.is_successful else "INCOMPLETE"
        return (
            f"PolicyPatchResult [{status}]: {self.num_applied_transformations} transformations, "
            f"cost={self.total_repair_cost:.1f}, "
            f"risk: {self.initial_attack_risk:.3f} -> {self.final_attack_risk:.3f} "
            f"(reduction={self.risk_reduction:.3f}), "
            f"log_conformance={self.log_conformance_verified}, "
            f"time={self.execution_time_sec:.3f}s"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize repair result to dictionary."""
        return {
            "is_successful": self.is_successful,
            "num_applied_transformations": self.num_applied_transformations,
            "modified_policy_ids": list(self.modified_policy_ids),
            "initial_attack_risk": round(self.initial_attack_risk, 4),
            "final_attack_risk": round(self.final_attack_risk, 4),
            "risk_reduction": round(self.risk_reduction, 4),
            "total_repair_cost": round(self.total_repair_cost, 4),
            "log_conformance_verified": self.log_conformance_verified,
            "num_iterations": self.num_iterations,
            "execution_time_sec": round(self.execution_time_sec, 4),
            "applied_transformations": [t.to_dict() for t in self.applied_transformations],
            "policy_diffs": dict(self.policy_diffs),
        }


class WorkloadConformanceChecker:
    """Historical workload conformance verifier for IAM configurations.

    Guarantees that proposed policy transformations preserve 100% of legitimate
    historical workload requests D^{T_1} by symbolically evaluating each event
    against the principal's governing policy set.
    """

    def __init__(self, capability_model: CapabilityModel | None = None) -> None:
        """Initialize conformance checker."""
        self.capability_model = (
            capability_model if capability_model is not None else load_default_capability_model()
        )

    def find_principal_node(self, graph: IAMGraph, identifier: str) -> GraphNode | None:
        """Locate a principal node in graph by ARN or unique ID."""
        node = graph.get_node_by_arn(identifier)
        if node is not None:
            return node
        node = graph.get_node(identifier)
        if node is not None:
            return node
        # Case-insensitive ARN match
        target_arn_lower = identifier.lower()
        for n in graph.get_nodes():
            if n.arn.lower() == target_arn_lower:
                return n
        return None

    def get_principal_policies(self, graph: IAMGraph, principal: GraphNode) -> list[PolicyDocument]:
        """Collect all active identity policy documents governing a principal."""
        policies: list[PolicyDocument] = []

        # 1. Direct inline policy document on principal node
        if principal.policy_document is not None:
            policies.append(principal.policy_document)

        # 2. Directly attached managed policies (u -> Policy via AttachedWith)
        for edge in graph.get_edges(source=principal.id, relation=EdgeRelation.ATTACHED_WITH):
            pol_node = graph.get_node(edge.target)
            if pol_node is not None and pol_node.policy_document is not None:
                policies.append(pol_node.policy_document)

        # 3. Group-attached policies (u -> Group via MemberOf, Group -> Policy via AttachedWith)
        for m_edge in graph.get_edges(source=principal.id, relation=EdgeRelation.MEMBER_OF):
            grp_node = graph.get_node(m_edge.target)
            if grp_node is not None:
                if grp_node.policy_document is not None:
                    policies.append(grp_node.policy_document)
                for p_edge in graph.get_edges(
                    source=grp_node.id, relation=EdgeRelation.ATTACHED_WITH
                ):
                    pol_node = graph.get_node(p_edge.target)
                    if pol_node is not None and pol_node.policy_document is not None:
                        policies.append(pol_node.policy_document)

        return policies

    def evaluate_request(self, graph: IAMGraph, request: AuthRequest) -> EvalResult:
        """Evaluate an AuthRequest against all relevant policies in the graph."""
        principal_node = self.find_principal_node(graph, request.principal_arn)
        if principal_node is None:
            return EvalResult(
                decision=EvalDecision.IMPLICIT_DENY,
                is_allowed=False,
                matched_allow_statements=[],
                matched_deny_statements=[],
            )

        # For sts:AssumeRole, verify target role trust policy
        if normalize_action_name(request.action) == "sts:assumerole":
            target_role = self.find_principal_node(graph, request.resource_arn)
            if target_role is not None and target_role.trust_policy is not None:
                trust_eval = PolicyEvaluator(policies=[target_role.trust_policy]).evaluate(request)
                if not trust_eval.is_allowed:
                    return trust_eval

        # Evaluate identity policies
        identity_policies = self.get_principal_policies(graph, principal_node)
        if not identity_policies:
            return EvalResult(
                decision=EvalDecision.IMPLICIT_DENY,
                is_allowed=False,
                matched_allow_statements=[],
                matched_deny_statements=[],
            )

        evaluator = PolicyEvaluator(policies=identity_policies)
        return evaluator.evaluate(request)

    def check_conformance(
        self, graph: IAMGraph, workload_logs: Sequence[AuthRequest]
    ) -> tuple[bool, list[AuthRequest]]:
        """Verify that every event in D^{T_1} evaluates to ALLOW in the repaired graph.

        Args:
            graph: The evaluated IAMGraph.
            workload_logs: Sequence of authorized historical AuthRequest events.

        Returns:
            Tuple of (is_conforming, list_of_denied_requests).
        """
        denied_requests: list[AuthRequest] = []
        for req in workload_logs:
            res = self.evaluate_request(graph, req)
            if not res.is_allowed:
                denied_requests.append(req)

        return len(denied_requests) == 0, denied_requests


def apply_transformation_to_graph(
    graph: IAMGraph,
    transformation: PolicyTransformation,
) -> IAMGraph:
    """Apply a policy transformation to an IAMGraph, updating policy ASTs and synchronizing edges.

    Args:
        graph: Source IAMGraph.
        transformation: PolicyTransformation to apply.

    Returns:
        Cloned IAMGraph reflecting the updated policy and pruned invalid edges.
    """
    cloned = graph.clone()

    pol_node = cloned.get_node(transformation.policy_id)
    if pol_node is None:
        pol_node = cloned.get_node_by_arn(transformation.policy_id)
    if pol_node is None:
        return cloned

    if transformation.is_trust_policy:
        if pol_node.trust_policy is not None:
            new_doc = transformation.apply(pol_node.trust_policy)
            updated_node = pol_node.model_copy(update={"trust_policy": new_doc}, deep=True)
            cloned.add_node(updated_node)
    else:
        if pol_node.policy_document is not None:
            new_doc = transformation.apply(pol_node.policy_document)
            updated_node = pol_node.model_copy(update={"policy_document": new_doc}, deep=True)
            cloned.add_node(updated_node)

    # Synchronize graph edges with updated policy permissions
    _synchronize_graph_edges(cloned, pol_node.id)
    return cloned


def _synchronize_graph_edges(graph: IAMGraph, modified_node_id: str) -> None:
    """Prune edges no longer authorized after a policy modification."""
    conformance_checker = WorkloadConformanceChecker()
    modified_node = graph.get_node(modified_node_id)
    if modified_node is None:
        return

    # 1. If modified node is a Role with updated trust policy, check incoming AssumesRole edges
    if modified_node.node_type == NodeType.ROLE and modified_node.trust_policy is not None:
        incoming_edges = graph.get_edges(
            target=modified_node.id, relation=EdgeRelation.ASSUMES_ROLE
        )
        for edge in incoming_edges:
            src_node = graph.get_node(edge.source)
            if src_node is None:
                continue
            req = AuthRequest(
                principal_arn=src_node.arn,
                action="sts:AssumeRole",
                resource_arn=modified_node.arn,
            )
            res = conformance_checker.evaluate_request(graph, req)
            if not res.is_allowed:
                graph.remove_edge(edge.source, edge.target, edge.relation)

    # 2. If modified node is a Policy, find all principals attaching this policy
    if modified_node.node_type == NodeType.POLICY:
        attached_principals: set[str] = set()

        # Direct attachments
        for edge in graph.get_edges(target=modified_node.id, relation=EdgeRelation.ATTACHED_WITH):
            src_n = graph.get_node(edge.source)
            if src_n is not None:
                if src_n.node_type in (NodeType.USER, NodeType.ROLE):
                    attached_principals.add(src_n.id)
                elif src_n.node_type == NodeType.GROUP:
                    # Group members
                    for m_edge in graph.get_edges(target=src_n.id, relation=EdgeRelation.MEMBER_OF):
                        attached_principals.add(m_edge.source)

        # Check outgoing authorization edges from all attached principals
        for p_id in attached_principals:
            p_node = graph.get_node(p_id)
            if p_node is None:
                continue

            for out_edge in graph.get_edges(source=p_id):
                target_node = graph.get_node(out_edge.target)
                if target_node is None:
                    continue

                if out_edge.relation == EdgeRelation.PASSES_TO:
                    req = AuthRequest(
                        principal_arn=p_node.arn,
                        action="iam:PassRole",
                        resource_arn=target_node.arn,
                    )
                    res = conformance_checker.evaluate_request(graph, req)
                    if not res.is_allowed:
                        graph.remove_edge(out_edge.source, out_edge.target, out_edge.relation)

                elif out_edge.relation == EdgeRelation.ACTS_ON:
                    actions_to_check = out_edge.actions or ["*"]
                    valid_actions: list[str] = []
                    for act in actions_to_check:
                        req = AuthRequest(
                            principal_arn=p_node.arn,
                            action=act,
                            resource_arn=target_node.arn,
                        )
                        res = conformance_checker.evaluate_request(graph, req)
                        if (
                            not res.is_allowed
                            and act == "iam:AttachUserPolicy"
                            and target_node.node_type == NodeType.POLICY
                        ):
                            # For AttachUserPolicy targeting a Policy node, check if user is authorized on self
                            req_self = AuthRequest(
                                principal_arn=p_node.arn,
                                action=act,
                                resource_arn=p_node.arn,
                            )
                            res_self = conformance_checker.evaluate_request(graph, req_self)
                            if res_self.is_allowed:
                                res = res_self
                        if res.is_allowed:
                            valid_actions.append(act)
                    if not valid_actions and actions_to_check:
                        graph.remove_edge(out_edge.source, out_edge.target, out_edge.relation)
                    elif len(valid_actions) < len(actions_to_check):
                        graph.remove_edge(out_edge.source, out_edge.target, out_edge.relation)
                        updated_edge = out_edge.model_copy(
                            update={"actions": valid_actions}, deep=True
                        )
                        graph.add_edge(updated_edge)

                elif out_edge.relation == EdgeRelation.ASSUMES_ROLE:
                    req = AuthRequest(
                        principal_arn=p_node.arn,
                        action="sts:AssumeRole",
                        resource_arn=target_node.arn,
                    )
                    res = conformance_checker.evaluate_request(graph, req)
                    if not res.is_allowed:
                        graph.remove_edge(out_edge.source, out_edge.target, out_edge.relation)

        # Check outgoing edges from the policy node itself (e.g. Policy -> Resource ActsOn)
        for out_edge in graph.get_edges(source=modified_node.id, relation=EdgeRelation.ACTS_ON):
            target_node = graph.get_node(out_edge.target)
            if target_node is None:
                continue
            if modified_node.policy_document is not None:
                evaluator = PolicyEvaluator(policies=[modified_node.policy_document])
                valid_acts: list[str] = []
                for act in out_edge.actions or ["*"]:
                    req = AuthRequest(
                        principal_arn="*",
                        action=act,
                        resource_arn=target_node.arn,
                    )
                    if evaluator.evaluate(req).is_allowed:
                        valid_acts.append(act)
                if not valid_acts:
                    graph.remove_edge(out_edge.source, out_edge.target, out_edge.relation)
                elif len(valid_acts) < len(out_edge.actions):
                    graph.remove_edge(out_edge.source, out_edge.target, out_edge.relation)
                    updated_edge = out_edge.model_copy(update={"actions": valid_acts}, deep=True)
                    graph.add_edge(updated_edge)


class CapabilityGreedyRepairEngine:
    """Greedy repair synthesizer with AWS Action Capability Model C(a, r, c) guarantees.

    Synthesizes minimal policy patches that eliminate privilege escalation reachability
    across latent completion sets {H_1, ..., H_M} while preserving 100% of historical
    workload operations D^{T_1}.
    """

    def __init__(
        self,
        config: RepairConfig | None = None,
        capability_model: CapabilityModel | None = None,
    ) -> None:
        """Initialize the capability greedy repair engine."""
        self.config: RepairConfig = config or RepairConfig()
        self.capability_model: CapabilityModel = (
            capability_model if capability_model is not None else load_default_capability_model()
        )
        self.conformance_checker: WorkloadConformanceChecker = WorkloadConformanceChecker(
            self.capability_model
        )

    def _evaluate_attack_risk(
        self,
        graph: IAMGraph,
        source_id: str,
        target_id: str,
        completions: list[LatentCompletion],
    ) -> float:
        """Compute Monte Carlo attack reachability P_hat across completions."""
        if not completions:
            # Single observed graph G_o evaluation
            nx_g = graph.nx_graph
            return 1.0 if nx.has_path(nx_g, source_id, target_id) else 0.0

        reachable_count = 0
        for comp in completions:
            nx_g = graph.nx_graph.copy()
            for ce in comp.sampled_edges:
                nx_g.add_edge(ce.source, ce.target, key=ce.relation.value)
            if nx.has_path(nx_g, source_id, target_id):
                reachable_count += 1

        return float(reachable_count / len(completions))

    def _resolve_node_id(self, graph: IAMGraph, identifier: str) -> str:
        """Resolve node ID from node ID or ARN."""
        node = graph.get_node(identifier)
        if node is not None:
            return node.id
        node = graph.get_node_by_arn(identifier)
        if node is not None:
            return node.id
        for n in graph.get_nodes():
            if n.arn.lower() == identifier.lower():
                return n.id
        return identifier

    def _find_authorizing_statements(
        self,
        graph: IAMGraph,
        source_node_id: str,
        target_node_id: str,
        relation: EdgeRelation,
        edge_actions: list[str],
    ) -> list[tuple[GraphNode, int, str, Statement, bool]]:
        """Identify policy nodes and statements authorizing a given cut edge.

        Returns:
            List of tuples: (policy_node, statement_index, action_name, statement, is_trust_policy).
        """
        results: list[tuple[GraphNode, int, str, Statement, bool]] = []
        src_node = graph.get_node(source_node_id)
        tgt_node = graph.get_node(target_node_id)
        if src_node is None or tgt_node is None:
            return results

        # Determine implied actions if edge actions not explicitly specified
        actions = list(edge_actions)
        if not actions:
            if relation == EdgeRelation.PASSES_TO:
                actions = ["iam:PassRole"]
            elif relation == EdgeRelation.ASSUMES_ROLE:
                actions = ["sts:AssumeRole"]

        # Case 1: Outgoing from principal (User or Role) via AttachedWith or MemberOf
        if src_node.node_type in (NodeType.USER, NodeType.ROLE):
            attached_pols: list[GraphNode] = []

            # Direct attachments
            for edge in graph.get_edges(source=src_node.id, relation=EdgeRelation.ATTACHED_WITH):
                pn = graph.get_node(edge.target)
                if pn is not None and pn.policy_document is not None:
                    attached_pols.append(pn)

            # Group attachments
            for m_edge in graph.get_edges(source=src_node.id, relation=EdgeRelation.MEMBER_OF):
                grp = graph.get_node(m_edge.target)
                if grp is not None:
                    for p_edge in graph.get_edges(
                        source=grp.id, relation=EdgeRelation.ATTACHED_WITH
                    ):
                        pn = graph.get_node(p_edge.target)
                        if pn is not None and pn.policy_document is not None:
                            attached_pols.append(pn)

            for pol_node in attached_pols:
                if pol_node.policy_document is None:
                    continue
                for idx, stmt in enumerate(pol_node.policy_document.statements):
                    if stmt.effect != Effect.ALLOW:
                        continue
                    # Match action
                    for act in actions:
                        if any(action_matches_pattern(act, pat) for pat in stmt.action) and any(
                            resource_matches_pattern(tgt_node.arn, r)
                            or resource_matches_pattern(src_node.arn, r)
                            or r == "*"
                            for r in stmt.resource
                        ):
                            results.append((pol_node, idx, act, stmt, False))

        # Case 2: Outgoing from Policy node (Policy -> Resource ActsOn)
        elif src_node.node_type == NodeType.POLICY and src_node.policy_document is not None:
            for idx, stmt in enumerate(src_node.policy_document.statements):
                if stmt.effect != Effect.ALLOW:
                    continue
                for act in actions:
                    if any(action_matches_pattern(act, pat) for pat in stmt.action) and any(
                        resource_matches_pattern(tgt_node.arn, r) for r in stmt.resource
                    ):
                        results.append((src_node, idx, act, stmt, False))

        # Case 3: Target Role trust policy for AssumesRole
        if (
            relation == EdgeRelation.ASSUMES_ROLE
            and tgt_node.node_type == NodeType.ROLE
            and tgt_node.trust_policy is not None
        ):
            for idx, stmt in enumerate(tgt_node.trust_policy.statements):
                if stmt.effect != Effect.ALLOW:
                    continue
                if (
                    any(action_matches_pattern("sts:AssumeRole", pat) for pat in stmt.action)
                    and stmt.principal is not None
                    and stmt.principal.matches(src_node.arn)
                ):
                    results.append((tgt_node, idx, "sts:AssumeRole", stmt, True))

        return results

    def _generate_candidate_transformations(
        self,
        graph: IAMGraph,
        cut_edges: list[tuple[str, str, EdgeRelation, list[str]]],
        workload_logs: Sequence[AuthRequest],
    ) -> list[PolicyTransformation]:
        """Generate candidate ScopeResource and RemoveAction transformations for cut edges."""
        transformations: list[PolicyTransformation] = []
        seen_keys: set[str] = set()

        for src_id, tgt_id, rel, acts in cut_edges:
            tgt_node = graph.get_node(tgt_id)
            tgt_arn = tgt_node.arn if tgt_node is not None else ""

            authorizing = self._find_authorizing_statements(graph, src_id, tgt_id, rel, acts)
            for pol_node, stmt_idx, act, stmt, is_trust in authorizing:
                cap = self.capability_model.get_action_capability(act)

                # 1. Candidate: ScopeResourceTransformation (Priority 1)
                if (
                    cap is not None
                    and cap.supports_resource_scoping
                    and not cap.wildcard_only
                    and not is_trust
                ):
                    src_node = graph.get_node(src_id)
                    src_arn = src_node.arn if src_node is not None else ""

                    # Find legitimate resources evidenced in historical workload D^{T_1}
                    legit_resources: set[str] = set()
                    for req in workload_logs:
                        if (
                            req.principal_arn.lower() in (src_arn.lower(), src_id.lower())
                            and action_matches_pattern(req.action, act)
                            and req.resource_arn != "*"
                            and req.resource_arn.lower() != tgt_arn.lower()
                        ):
                            legit_resources.add(req.resource_arn)

                    # Also if statement already contains non-target resources, retain legitimate ones
                    for r in stmt.resource:
                        if r != "*" and r.lower() != tgt_arn.lower():
                            legit_resources.add(r)

                    if legit_resources:
                        sorted_res = sorted(legit_resources)
                        t_key = f"scope_{pol_node.id}_{stmt_idx}_{act}_{'_'.join(sorted_res)}"
                        if t_key not in seen_keys:
                            seen_keys.add(t_key)
                            scope_trans = ScopeResourceTransformation(
                                policy_id=pol_node.id,
                                statement_index=stmt_idx,
                                action=act,
                                target_resources=sorted_res,
                                original_resources=list(stmt.resource),
                                cost=self.config.cost_scope_resource,
                                is_trust_policy=is_trust,
                            )
                            # Validate capability model admissibility
                            if scope_trans.validate_capability(self.capability_model).is_valid:
                                transformations.append(scope_trans)

                # 2. Candidate: RemoveActionTransformation (Priority 2)
                if self.config.allow_action_removal:
                    r_key = f"remove_{pol_node.id}_{stmt_idx}_{act}_{is_trust}"
                    if r_key not in seen_keys:
                        seen_keys.add(r_key)
                        remove_trans = RemoveActionTransformation(
                            policy_id=pol_node.id,
                            statement_index=stmt_idx,
                            action=act,
                            cost=self.config.cost_remove_action,
                            is_trust_policy=is_trust,
                        )
                        transformations.append(remove_trans)

        return transformations

    def repair(
        self,
        graph: IAMGraph,
        source_id: str,
        target_id: str,
        workload_logs: Sequence[AuthRequest] | None = None,
        completions: list[LatentCompletion] | None = None,
        pruning_result: PruningResult | None = None,
    ) -> PolicyPatchResult:
        """Synthesize minimal, capability-compliant least-privilege repair patch.

        Args:
            graph: Partially observed authorization graph G_o.
            source_id: Attacker entrypoint principal node ID or ARN.
            target_id: High-value asset or administrative target node ID or ARN.
            workload_logs: Historical workload activity trace D^{T_1}.
            completions: Sampled latent completions {H_1, ..., H_M}.
            pruning_result: Optional precomputed candidate pruning result.

        Returns:
            PolicyPatchResult containing repaired graph, applied transformations, and diffs.
        """
        start_time = time.perf_counter()
        logs: list[AuthRequest] = list(workload_logs or [])
        comp_list: list[LatentCompletion] = list(completions or [])

        s_id = self._resolve_node_id(graph, source_id)
        t_id = self._resolve_node_id(graph, target_id)

        # 1. Baseline log conformance check on G_o
        baseline_conforming, denied_baseline = self.conformance_checker.check_conformance(
            graph, logs
        )
        if self.config.strict_baseline_conformance and not baseline_conforming:
            raise ValueError(
                f"Historical workload D^{{T_1}} is not fully authorized on baseline graph G_o; "
                f"{len(denied_baseline)} events were denied before repair."
            )

        # 2. Initial attack risk evaluation
        initial_risk = self._evaluate_attack_risk(graph, s_id, t_id, comp_list)
        if initial_risk <= self.config.residual_risk_threshold_delta:
            logger.info(
                "Initial attack risk already satisfies threshold delta; returning identity patch."
            )
            elapsed = time.perf_counter() - start_time
            return PolicyPatchResult(
                repaired_graph=graph.clone(),
                applied_transformations=[],
                modified_policy_ids=[],
                modified_policy_documents={},
                policy_diffs={},
                initial_attack_risk=initial_risk,
                final_attack_risk=initial_risk,
                total_repair_cost=0.0,
                log_conformance_verified=baseline_conforming,
                is_successful=True,
                execution_time_sec=elapsed,
                num_iterations=0,
                denied_workload_events=[],
            )

        # 3. Extract candidate cut edges
        cut_edge_candidates: list[tuple[str, str, EdgeRelation, list[str]]] = []
        if pruning_result is not None:
            for pe in pruning_result.pruned_edges:
                cut_edge_candidates.append((pe.source, pe.target, pe.relation, list(pe.actions)))
        else:
            # Fall back to extracting cut candidates from shortest paths across completions (or G_o alone)
            eval_completions = (
                comp_list
                if comp_list
                else [LatentCompletion(sample_id=0, sampled_edges=[], graph=graph)]
            )
            seen_edge_triples: set[tuple[str, str, str]] = set()

            for comp in eval_completions:
                nx_union = graph.nx_graph.copy()
                for ce in comp.sampled_edges:
                    nx_union.add_edge(ce.source, ce.target, key=ce.relation.value)

                if nx.has_path(nx_union, s_id, t_id):
                    path_nodes = nx.shortest_path(nx_union, s_id, t_id)
                    for i in range(len(path_nodes) - 1):
                        u, v = path_nodes[i], path_nodes[i + 1]

                        # Check if edge is in G_o
                        for edge in graph.get_edges(source=u, target=v):
                            trip = (u, v, edge.relation.value)
                            if trip not in seen_edge_triples:
                                seen_edge_triples.add(trip)
                                cut_edge_candidates.append(
                                    (u, v, edge.relation, list(edge.actions))
                                )

                        # Check if edge is in sampled completion
                        for ce in comp.sampled_edges:
                            if ce.source == u and ce.target == v:
                                trip = (u, v, ce.relation.value)
                                if trip not in seen_edge_triples:
                                    seen_edge_triples.add(trip)
                                    cut_edge_candidates.append(
                                        (u, v, ce.relation, list(ce.actions))
                                    )

        # 4. Greedy Repair Loop
        current_graph = graph.clone()
        current_risk = initial_risk
        applied_transformations: list[PolicyTransformation] = []
        total_cost = 0.0

        for iteration in range(self.config.max_iterations):
            if current_risk <= self.config.residual_risk_threshold_delta:
                break

            # Generate candidate transformations
            candidates = self._generate_candidate_transformations(
                current_graph, cut_edge_candidates, logs
            )
            if not candidates:
                logger.warning("No candidate transformations generated; terminating search.")
                break

            best_transformation: PolicyTransformation | None = None
            best_spec_graph: IAMGraph | None = None
            best_spec_risk: float = current_risk
            best_score: float = -1.0

            for candidate in candidates:
                # 1. Capability validity check
                val = candidate.validate_capability(self.capability_model)
                if not val.is_valid:
                    continue

                # 2. Speculative graph update
                spec_graph = apply_transformation_to_graph(current_graph, candidate)

                # 3. 100% Historical workload conformance check
                conforms, _ = self.conformance_checker.check_conformance(spec_graph, logs)
                if not conforms:
                    # Inadmissible: breaks legitimate workload operations D^{T_1}
                    continue

                # 4. Speculative attack risk evaluation
                spec_risk = self._evaluate_attack_risk(spec_graph, s_id, t_id, comp_list)
                delta_risk = current_risk - spec_risk

                if delta_risk > 1e-6:
                    # Score = marginal attack risk reduction per unit repair cost
                    score = delta_risk / candidate.cost
                    if (
                        self.config.prefer_scoping
                        and candidate.transformation_type == PolicyTransformationType.SCOPE_RESOURCE
                    ):
                        score *= 1.1  # Slight tie-breaker preference for resource scoping

                    if score > best_score:
                        best_score = score
                        best_transformation = candidate
                        best_spec_graph = spec_graph
                        best_spec_risk = spec_risk

            if best_transformation is None or best_spec_graph is None:
                logger.info(
                    "No admissible log-conforming transformation achieved risk reduction; stopping."
                )
                break

            # Apply best transformation to current graph
            current_graph = best_spec_graph
            current_risk = best_spec_risk
            applied_transformations.append(best_transformation)
            total_cost += best_transformation.cost

            logger.info(
                f"Iteration {iteration + 1}: Applied {best_transformation.description} "
                f"(cost={best_transformation.cost:.1f}), risk: -> {current_risk:.3f}"
            )

        elapsed = time.perf_counter() - start_time

        # 5. Synthesize policy diffs and collect modified documents
        modified_policy_ids = sorted({t.policy_id for t in applied_transformations})
        modified_docs: dict[str, PolicyDocument] = {}
        policy_diffs: dict[str, str] = {}

        for pol_id in modified_policy_ids:
            orig_node = graph.get_node(pol_id) or graph.get_node_by_arn(pol_id)
            repaired_node = current_graph.get_node(pol_id) or current_graph.get_node_by_arn(pol_id)

            if orig_node is not None and repaired_node is not None:
                # Check identity policy
                if (
                    orig_node.policy_document is not None
                    and repaired_node.policy_document is not None
                ):
                    modified_docs[pol_id] = repaired_node.policy_document
                    diff_str = generate_policy_diff(
                        orig_node.policy_document,
                        repaired_node.policy_document,
                        policy_name=orig_node.name or pol_id,
                    )
                    policy_diffs[pol_id] = diff_str
                # Check trust policy
                elif orig_node.trust_policy is not None and repaired_node.trust_policy is not None:
                    modified_docs[pol_id] = repaired_node.trust_policy
                    diff_str = generate_policy_diff(
                        orig_node.trust_policy,
                        repaired_node.trust_policy,
                        policy_name=f"{orig_node.name or pol_id}_TrustPolicy",
                    )
                    policy_diffs[pol_id] = diff_str

        # Final verification of log conformance on repaired graph
        final_conforming, denied_final = self.conformance_checker.check_conformance(
            current_graph, logs
        )
        is_successful = (
            current_risk <= self.config.residual_risk_threshold_delta and final_conforming
        )

        return PolicyPatchResult(
            repaired_graph=current_graph,
            applied_transformations=applied_transformations,
            modified_policy_ids=modified_policy_ids,
            modified_policy_documents=modified_docs,
            policy_diffs=policy_diffs,
            initial_attack_risk=initial_risk,
            final_attack_risk=current_risk,
            total_repair_cost=total_cost,
            log_conformance_verified=final_conforming,
            is_successful=is_successful,
            execution_time_sec=elapsed,
            num_iterations=len(applied_transformations),
            denied_workload_events=denied_final,
        )
