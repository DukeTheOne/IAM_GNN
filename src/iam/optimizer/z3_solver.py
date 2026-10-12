"""Exact SMT and MaxSAT canonical policy repair solver using Z3.

Implements Phase 3 Week 7 (Step 7.4):
- Formal SMT decision variable encoding for policy actions (x_{action_keep}, x_{scoped_arn}).
- Hard capability constraints: x_{scoped_arn} => SupportsResourceScoping(action) in C(a, r, c).
- Hard security reachability constraints: \neg AttackReachability(s, target; G_o \\cup H_i).
- Hard historical workload conformance constraints: \bigwedge_{e \\in D^{T_1}} Authorized(e) = True.
- Soft constraint optimization via Z3 Optimize() minimizing weighted repair cost.
- Optimality gap analysis: formal mathematical baseline benchmarking greedy repair against exact MaxSAT.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import z3

from iam.generator.graph import EdgeRelation, GraphNode, IAMGraph, NodeType
from iam.optimizer.repair import (
    CapabilityGreedyRepairEngine,
    PolicyPatchResult,
    RepairConfig,
    WorkloadConformanceChecker,
    apply_transformation_to_graph,
)
from iam.optimizer.sampler import LatentCompletion
from iam.optimizer.transformations import (
    PolicyTransformation,
    RemoveActionTransformation,
    ScopeResourceTransformation,
)
from iam.parser.capability import CapabilityModel, load_default_capability_model
from iam.parser.evaluator import AuthRequest
from iam.parser.schema import Effect, Statement
from iam.parser.wildcard import (
    action_matches_pattern,
    normalize_action_name,
)

logger = logging.getLogger(__name__)


@dataclass
class Z3Decision:
    """Action-level decision output produced by Z3 SMT/MaxSAT model."""

    action: str
    keep: bool
    scoped: bool
    cost: float
    scoped_resources: list[str] = field(default_factory=list)
    transformation: PolicyTransformation | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize decision to dictionary."""
        return {
            "action": self.action,
            "keep": self.keep,
            "scoped": self.scoped,
            "cost": round(self.cost, 4),
            "scoped_resources": list(self.scoped_resources),
            "transformation": self.transformation.to_dict() if self.transformation else None,
        }


@dataclass
class Z3RepairResult:
    """Result of exact Z3 SMT/MaxSAT optimization."""

    is_sat: bool
    status: str
    total_cost: float
    decisions: dict[str, Z3Decision]
    applied_transformations: list[PolicyTransformation]
    repaired_graph: IAMGraph | None
    execution_time_sec: float
    optimality_gap_vs_greedy: float | None = None
    num_constraints: int = 0
    num_variables: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_applied_transformations(self) -> int:
        """Count of synthesized transformations."""
        return len(self.applied_transformations)

    def summary(self) -> str:
        """Human-readable summary of Z3 exact solving."""
        gap_str = (
            f", optimality_gap={self.optimality_gap_vs_greedy:.2f}"
            if self.optimality_gap_vs_greedy is not None
            else ""
        )
        return (
            f"Z3RepairResult [{self.status.upper()}]: "
            f"cost={self.total_cost:.1f}, "
            f"transformations={self.num_applied_transformations}, "
            f"time={self.execution_time_sec:.4f}s"
            f"{gap_str}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize Z3 repair result to dictionary."""
        return {
            "is_sat": self.is_sat,
            "status": self.status,
            "total_cost": round(self.total_cost, 4),
            "optimality_gap_vs_greedy": (
                round(self.optimality_gap_vs_greedy, 4)
                if self.optimality_gap_vs_greedy is not None
                else None
            ),
            "num_applied_transformations": self.num_applied_transformations,
            "num_constraints": self.num_constraints,
            "num_variables": self.num_variables,
            "execution_time_sec": round(self.execution_time_sec, 4),
            "decisions": {act: dec.to_dict() for act, dec in self.decisions.items()},
            "applied_transformations": [t.to_dict() for t in self.applied_transformations],
        }


class Z3CanonicalRepairSolver:
    """Exact SMT/MaxSAT solver for canonical IAM privilege escalation repair.

    Provides a formal mathematical baseline to verify greedy solver correctness
    and quantify any residual optimality gap on canonical privilege escalation patterns.
    """

    def __init__(
        self,
        capability_model: CapabilityModel | None = None,
        cost_scope: float = 1.0,
        cost_remove: float = 5.0,
    ) -> None:
        """Initialize Z3 canonical repair solver."""
        self.capability_model: CapabilityModel = (
            capability_model if capability_model is not None else load_default_capability_model()
        )
        self.cost_scope: float = cost_scope
        self.cost_remove: float = cost_remove
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

    def solve(
        self,
        graph: IAMGraph,
        source_id: str,
        target_id: str,
        workload_logs: Sequence[AuthRequest] | None = None,
        completions: list[LatentCompletion] | None = None,
    ) -> Z3RepairResult:
        """Solve exact MaxSAT policy repair using Z3.

        Args:
            graph: Partially observed IAM authorization graph G_o.
            source_id: Entrypoint principal node ID or ARN.
            target_id: High-value asset or admin target node ID or ARN.
            workload_logs: Historical workload operations D^{T_1}.
            completions: Optional sampled latent completions {H_1, ..., H_M}.

        Returns:
            Z3RepairResult with optimal decisions and repaired graph.
        """
        start_time = time.perf_counter()
        logs: list[AuthRequest] = list(workload_logs or [])
        comp_list: list[LatentCompletion] = list(completions or [])

        s_id = self._resolve_node_id(graph, source_id)
        t_id = self._resolve_node_id(graph, target_id)
        source_node = graph.get_node(s_id)
        target_node = graph.get_node(t_id)

        if source_node is None or target_node is None:
            return Z3RepairResult(
                is_sat=False,
                status="unsat",
                total_cost=0.0,
                decisions={},
                applied_transformations=[],
                repaired_graph=None,
                execution_time_sec=time.perf_counter() - start_time,
            )

        # 1. Discover all attack paths from s to t across completions (or Go alone)
        eval_completions = (
            comp_list
            if comp_list
            else [LatentCompletion(sample_id=0, sampled_edges=[], graph=graph)]
        )

        all_attack_paths: list[list[str]] = []
        for comp in eval_completions:
            nx_union = graph.nx_graph.copy()
            for ce in comp.sampled_edges:
                nx_union.add_edge(ce.source, ce.target, key=ce.relation.value)
            if nx.has_path(nx_union, s_id, t_id):
                # Extract all simple paths up to depth 8 to find all parallel escalation routes
                try:
                    paths = list(nx.all_simple_paths(nx_union, s_id, t_id, cutoff=8))
                    all_attack_paths.extend(paths)
                except Exception:
                    if nx.has_path(nx_union, s_id, t_id):
                        all_attack_paths.append(list(nx.shortest_path(nx_union, s_id, t_id)))

        if not all_attack_paths:
            # Already safe; no attack paths exist
            logger.info("Z3: No attack paths found; system already secure.")
            elapsed = time.perf_counter() - start_time
            return Z3RepairResult(
                is_sat=True,
                status="sat",
                total_cost=0.0,
                decisions={},
                applied_transformations=[],
                repaired_graph=graph.clone(),
                execution_time_sec=elapsed,
            )

        # 2. Extract actionable cut edges along attack paths
        actionable_edges: list[tuple[str, str, EdgeRelation, list[str]]] = []
        seen_triples: set[tuple[str, str, str]] = set()

        for path in all_attack_paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                for edge in graph.get_edges(source=u, target=v):
                    triple = (u, v, edge.relation.value)
                    if triple not in seen_triples:
                        seen_triples.add(triple)
                        actionable_edges.append((u, v, edge.relation, list(edge.actions)))

        # 3. Identify governing policy statements and actions
        # Map: action -> (pol_node, stmt_idx, is_trust, original_stmt, target_node)
        action_meta: dict[str, tuple[GraphNode, int, bool, Statement, GraphNode]] = {}

        for u, v, rel, acts in actionable_edges:
            tgt_n = graph.get_node(v)
            if tgt_n is None:
                continue

            # Determine implied action
            eff_actions = list(acts)
            if not eff_actions:
                if rel == EdgeRelation.PASSES_TO:
                    eff_actions = ["iam:PassRole"]
                elif rel == EdgeRelation.ASSUMES_ROLE:
                    eff_actions = ["sts:AssumeRole"]

            src_n = graph.get_node(u)
            if src_n is not None and src_n.node_type in (NodeType.USER, NodeType.ROLE):
                # Search attached policies
                for a_edge in graph.get_edges(source=src_n.id, relation=EdgeRelation.ATTACHED_WITH):
                    p_node = graph.get_node(a_edge.target)
                    if p_node is not None and p_node.policy_document is not None:
                        for idx, stmt in enumerate(p_node.policy_document.statements):
                            if stmt.effect != Effect.ALLOW:
                                continue
                            for act in eff_actions:
                                if any(action_matches_pattern(act, p) for p in stmt.action):
                                    norm_act = normalize_action_name(act)
                                    if norm_act not in action_meta:
                                        action_meta[norm_act] = (p_node, idx, False, stmt, tgt_n)

            # Target trust policy
            if (
                rel == EdgeRelation.ASSUMES_ROLE
                and tgt_n.node_type == NodeType.ROLE
                and tgt_n.trust_policy is not None
            ):
                for idx, stmt in enumerate(tgt_n.trust_policy.statements):
                    if stmt.effect != Effect.ALLOW:
                        continue
                    if any(action_matches_pattern("sts:AssumeRole", p) for p in stmt.action):
                        norm_act = "sts:assumerole"
                        if norm_act not in action_meta:
                            action_meta[norm_act] = (tgt_n, idx, True, stmt, tgt_n)

        # 4. Extract legitimate evidenced ARNs from workload trace D^{T_1}
        evidenced_resources: dict[str, list[str]] = {}
        for norm_act, (_pol_node, _stmt_idx, _is_trust, _stmt, tgt_n) in action_meta.items():
            legit_arns: set[str] = set()
            for req in logs:
                if (
                    action_matches_pattern(req.action, norm_act)
                    and req.resource_arn != "*"
                    and req.resource_arn.lower() != tgt_n.arn.lower()
                ):
                    legit_arns.add(req.resource_arn)
            evidenced_resources[norm_act] = sorted(legit_arns)

        # 5. Formulate Z3 SMT Decision Variables
        opt = z3.Optimize()
        vars_keep: dict[str, z3.BoolRef] = {}
        vars_scoped: dict[str, z3.BoolRef] = {}
        num_constraints = 0

        for norm_act in action_meta:
            clean_name = norm_act.replace(":", "_").replace("-", "_")
            x_keep = z3.Bool(f"keep_{clean_name}")
            x_scoped = z3.Bool(f"scoped_{clean_name}")
            vars_keep[norm_act] = x_keep
            vars_scoped[norm_act] = x_scoped

            # Consistency constraint: scoped => keep
            opt.add(z3.Implies(x_scoped, x_keep))
            num_constraints += 1

            # Capability constraint: x_scoped => SupportsResourceScoping(act) in C(a, r, c)
            cap = self.capability_model.get_action_capability(norm_act)
            can_scope = (
                cap is not None
                and cap.supports_resource_scoping
                and not cap.wildcard_only
                and len(evidenced_resources.get(norm_act, [])) > 0
            )
            if not can_scope:
                opt.add(z3.Not(x_scoped))
                num_constraints += 1

        # 6. Hard Historical Workload Conformance Constraints
        # Every action exercised in D^{T_1} must be kept (x_keep == True)
        for req in logs:
            norm_req_act = normalize_action_name(req.action)
            if norm_req_act in vars_keep:
                opt.add(vars_keep[norm_req_act] == True)  # noqa: E712
                num_constraints += 1
                # If scoped, req.resource_arn must be in evidenced resources
                if req.resource_arn not in evidenced_resources.get(norm_req_act, []):
                    opt.add(z3.Not(vars_scoped[norm_req_act]))
                    num_constraints += 1

        # 7. Hard Security Reachability Constraints
        # For each attack path, at least one critical step along the path must be deactivated
        for path in all_attack_paths:
            path_step_conditions: list[z3.BoolRef] = []

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                tgt_n = graph.get_node(v)
                tgt_arn = tgt_n.arn if tgt_n else ""

                for edge in graph.get_edges(source=u, target=v):
                    edge_acts = list(edge.actions)
                    if not edge_acts:
                        if edge.relation == EdgeRelation.PASSES_TO:
                            edge_acts = ["iam:PassRole"]
                        elif edge.relation == EdgeRelation.ASSUMES_ROLE:
                            edge_acts = ["sts:AssumeRole"]

                    for act in edge_acts:
                        norm_act = normalize_action_name(act)
                        if norm_act in vars_keep:
                            # If target is not in evidenced resources:
                            # Step is active iff action is kept AND not scoped to benign resources
                            ev_list = evidenced_resources.get(norm_act, [])
                            if tgt_arn.lower() not in [r.lower() for r in ev_list]:
                                step_active = z3.And(
                                    vars_keep[norm_act],
                                    z3.Not(vars_scoped[norm_act]),
                                )
                            else:
                                step_active = vars_keep[norm_act]
                            path_step_conditions.append(step_active)

            if path_step_conditions:
                # Attack path succeeds if ALL its steps are active
                path_active = (
                    z3.And(*path_step_conditions)
                    if len(path_step_conditions) > 1
                    else path_step_conditions[0]
                )
                # Hard constraint: attack must be broken
                opt.add(z3.Not(path_active))
                num_constraints += 1

        # 8. Soft Constraint Optimization: Minimize Weighted Cost Function
        # Cost = sum w_remove * (1 - keep) + w_scope * (keep and scoped)
        cost_expr = z3.RealVal(0)
        for norm_act in action_meta:
            x_k = vars_keep[norm_act]
            x_s = vars_scoped[norm_act]
            rem_cost = z3.If(x_k, z3.RealVal(0), z3.RealVal(int(self.cost_remove)))
            scope_cost = z3.If(
                z3.And(x_k, x_s),
                z3.RealVal(int(self.cost_scope)),
                z3.RealVal(0),
            )
            cost_expr = cost_expr + rem_cost + scope_cost

        cost_var = z3.Real("total_repair_cost")
        opt.add(cost_var == cost_expr)
        opt.minimize(cost_var)

        # 9. Solve MaxSAT
        check_status = opt.check()
        elapsed = time.perf_counter() - start_time

        if check_status != z3.sat:
            status_str = "unsat" if check_status == z3.unsat else "unknown"
            return Z3RepairResult(
                is_sat=False,
                status=status_str,
                total_cost=0.0,
                decisions={},
                applied_transformations=[],
                repaired_graph=None,
                execution_time_sec=elapsed,
                num_constraints=num_constraints,
                num_variables=len(vars_keep) * 2,
            )

        # 10. Extract Optimal Model Decisions and Synthesize Patch
        model = opt.model()
        opt_cost_val = model.evaluate(cost_var)
        if opt_cost_val is not None and hasattr(opt_cost_val, "as_decimal"):
            total_cost_float = float(opt_cost_val.as_decimal(4).replace("?", ""))
        elif opt_cost_val is not None:
            total_cost_float = float(opt_cost_val.as_long())
        else:
            total_cost_float = 0.0

        decisions: dict[str, Z3Decision] = {}
        applied_transformations: list[PolicyTransformation] = []
        repaired_graph = graph.clone()

        for norm_act, (pol_node, stmt_idx, is_trust, stmt, _tgt_n) in action_meta.items():
            k_val = bool(z3.is_true(model.evaluate(vars_keep[norm_act])))
            s_val = bool(z3.is_true(model.evaluate(vars_scoped[norm_act])))

            trans: PolicyTransformation | None = None
            action_cost = 0.0

            if not k_val:
                # Remove action
                action_cost = self.cost_remove
                trans = RemoveActionTransformation(
                    policy_id=pol_node.id,
                    statement_index=stmt_idx,
                    action=norm_act,
                    cost=self.cost_remove,
                    is_trust_policy=is_trust,
                )
            elif s_val:
                # Scope resource
                action_cost = self.cost_scope
                scoped_list = evidenced_resources.get(norm_act, [])
                trans = ScopeResourceTransformation(
                    policy_id=pol_node.id,
                    statement_index=stmt_idx,
                    action=norm_act,
                    target_resources=scoped_list,
                    original_resources=list(stmt.resource),
                    cost=self.cost_scope,
                    is_trust_policy=is_trust,
                )

            if trans is not None:
                applied_transformations.append(trans)
                repaired_graph = apply_transformation_to_graph(repaired_graph, trans)

            decisions[norm_act] = Z3Decision(
                action=norm_act,
                keep=k_val,
                scoped=s_val,
                cost=action_cost,
                scoped_resources=evidenced_resources.get(norm_act, []),
                transformation=trans,
            )

        return Z3RepairResult(
            is_sat=True,
            status="sat",
            total_cost=total_cost_float,
            decisions=decisions,
            applied_transformations=applied_transformations,
            repaired_graph=repaired_graph,
            execution_time_sec=elapsed,
            num_constraints=num_constraints,
            num_variables=len(vars_keep) * 2,
        )

    def compare_with_greedy(
        self,
        graph: IAMGraph,
        source_id: str,
        target_id: str,
        workload_logs: Sequence[AuthRequest] | None = None,
        completions: list[LatentCompletion] | None = None,
    ) -> tuple[Z3RepairResult, PolicyPatchResult, float]:
        """Execute both Z3 exact solver and greedy repair, returning the optimality gap.

        Args:
            graph: IAMGraph under evaluation.
            source_id: Originating entrypoint identity.
            target_id: Destination target asset.
            workload_logs: Historical workload activity.
            completions: Sampled latent completions.

        Returns:
            Tuple of (z3_result, greedy_result, optimality_gap).
        """
        logs = list(workload_logs or [])
        comps = list(completions or [])

        # 1. Run exact Z3 solver
        z3_res = self.solve(
            graph=graph,
            source_id=source_id,
            target_id=target_id,
            workload_logs=logs,
            completions=comps,
        )

        # 2. Run greedy heuristic solver
        greedy_engine = CapabilityGreedyRepairEngine(
            config=RepairConfig(
                cost_scope_resource=self.cost_scope,
                cost_remove_action=self.cost_remove,
            ),
            capability_model=self.capability_model,
        )
        greedy_res = greedy_engine.repair(
            graph=graph,
            source_id=source_id,
            target_id=target_id,
            workload_logs=logs,
            completions=comps,
        )

        # 3. Compute Optimality Gap: Cost(Greedy) - Cost(Z3)
        optimality_gap = max(0.0, greedy_res.total_repair_cost - z3_res.total_cost)
        z3_res.optimality_gap_vs_greedy = optimality_gap

        return z3_res, greedy_res, optimality_gap
