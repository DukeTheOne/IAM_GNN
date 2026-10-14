"""Central Tradeoff Experiment Engine (RQ2 Execution).

Evaluates the 6 remediation strategies under partial authorization observability (Condition E):
- Method 0 (Oracle): Greedy repair executed directly on ground-truth graph G*.
- Method 1 (No Action): Baseline identity patch (zero changes to Go).
- Method 2 (Coarse Deletion): Excises dangerous actions (e.g. iam:PassRole) broadly (IAMPERE-style).
- Method 3 (Log-Only Tightening): Restricts policies strictly to observed T1 logs (PolicyRefiner-style).
- Method 4a (Point-Estimate Repair): Reconstructs single graph at threshold p >= theta (default 0.50).
- Method 4b (Uncertainty-Aware Repair - Ours): Full stochastic completion sampling + capability repair.

Enforces:
- AWS Action-Resource-Condition capability model C(a, r, c).
- 100% baseline workload conformance (D^{T_1}) on valid repairs.
- Multi-dimensional evaluation: security risk (reachability on G*) vs operational utility (T2 breakage).
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from iam.generator.graph import EdgeRelation, IAMGraph
from iam.optimizer.pruning import CandidatePruner, PruningConfig
from iam.optimizer.repair import (
    CapabilityGreedyRepairEngine,
    RepairConfig,
    apply_transformation_to_graph,
)
from iam.optimizer.sampler import (
    CandidateEdge,
    LatentCompletion,
    LatentCompletionSampler,
)
from iam.optimizer.transformations import (
    PolicyTransformation,
    RemoveActionTransformation,
    ScopeResourceTransformation,
)
from iam.parser.capability import CapabilityModel, load_default_capability_model
from iam.parser.schema import Effect, PolicyDocument, Statement
from iam.parser.wildcard import action_matches_pattern, normalize_action_name
from iam.verifier.verifier import (
    CompletionSetVerifier,
    VerificationReport,
)
from iam.verifier.workload import TemporalWorkloadSplit

logger = logging.getLogger(__name__)


class RemediationMethod(str, Enum):
    """The 6 scientific remediation methods evaluated under RQ2."""

    ORACLE = "oracle"  # Method 0: Upper-bound Oracle (G*)
    NO_ACTION = "no_action"  # Method 1: No Action / Identity patch (Go)
    COARSE_DELETION = "coarse_deletion"  # Method 2: Coarse Action Deletion (IAMPERE-style)
    LOG_ONLY_TIGHTENING = (
        "log_only_tightening"  # Method 3: Log-Only Tightening (PolicyRefiner-style)
    )
    POINT_ESTIMATE = "point_estimate"  # Method 4a: Point-Estimate Repair (p >= 0.50)
    UNCERTAINTY_AWARE = "uncertainty_aware"  # Method 4b: Uncertainty-Aware Repair (Ours)


class TradeoffConfig(BaseModel):
    """Configuration governing the RQ2 tradeoff experiment."""

    num_completions: int = 20
    point_estimate_threshold: float = 0.50
    random_seeds: list[int] = Field(default_factory=lambda: [42, 43, 44])
    t1_days: float = 90.0
    t2_days: float = 30.0
    drift_prob: float = 0.20
    max_candidates: int = 50
    risk_threshold: float = 0.0
    timeout_sec: float = 5.0
    strict_baseline_conformance: bool = True


class MethodResult(BaseModel):
    """Verification and performance metrics for a single remediation method."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    method: RemediationMethod
    method_name: str
    repaired_graph: IAMGraph
    applied_transformations: list[PolicyTransformation]
    verification_report: VerificationReport

    # Core Security Risk Metrics
    level1_risk: float = Field(description="Reachability risk on Go")
    level2_risk: float = Field(description="Reachability risk on Completion Set")
    level3_risk: float = Field(description="Reachability risk on Ground Truth G*")
    is_safe_gt: bool = Field(description="Whether Level 3 risk <= threshold")
    is_misleading: bool = Field(description="Safe on Go but unsafe on G*")

    # Core Operational Utility Metrics
    t1_breakage: float = Field(description="Breakage rate on baseline T1")
    t2_breakage: float = Field(description="Breakage rate on holdout T2 with drift")
    total_t1_requests: int
    total_t2_requests: int
    t1_denied_requests: int
    t2_denied_requests: int

    # Governance & Execution Metrics
    has_side_effects: bool
    num_transformations: int
    transformation_cost: float
    execution_time_sec: float
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize method evaluation metrics to dictionary."""
        return {
            "method": self.method.value,
            "method_name": self.method_name,
            "level1_risk": round(self.level1_risk, 4),
            "level2_risk": round(self.level2_risk, 4),
            "level3_risk": round(self.level3_risk, 4),
            "is_safe_gt": self.is_safe_gt,
            "is_misleading": self.is_misleading,
            "t1_breakage": round(self.t1_breakage, 4),
            "t2_breakage": round(self.t2_breakage, 4),
            "total_t1_requests": self.total_t1_requests,
            "total_t2_requests": self.total_t2_requests,
            "t1_denied_requests": self.t1_denied_requests,
            "t2_denied_requests": self.t2_denied_requests,
            "has_side_effects": self.has_side_effects,
            "num_transformations": self.num_transformations,
            "transformation_cost": round(self.transformation_cost, 4),
            "execution_time_sec": round(self.execution_time_sec, 6),
            "details": self.details,
            "verification_status": self.verification_report.overall_status.value,
        }


class TradeoffEnvironment(BaseModel):
    """Encapsulates a benchmark environment under Condition E."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    ground_truth_graph: IAMGraph
    observed_graph: IAMGraph
    candidate_edges: list[CandidateEdge]
    source_id: str
    target_id: str
    workload_split: TemporalWorkloadSplit
    safe_roles: set[str] = Field(default_factory=set)


class TradeoffExperimentRunner:
    """Automated benchmark runner orchestrating the 6 remediation methods."""

    def __init__(
        self,
        capability_model: CapabilityModel | None = None,
        config: TradeoffConfig | None = None,
    ) -> None:
        """Initialize the tradeoff runner."""
        self.config: TradeoffConfig = config or TradeoffConfig()
        self.capability_model: CapabilityModel = (
            capability_model if capability_model is not None else load_default_capability_model()
        )
        self.verifier: CompletionSetVerifier = CompletionSetVerifier(
            capability_model=self.capability_model
        )

    # --------------------------------------------------------------------------
    # Method 0: Oracle (Ground Truth G*)
    # --------------------------------------------------------------------------
    def run_method_0_oracle(
        self,
        env: TradeoffEnvironment,
        completions: list[LatentCompletion],
    ) -> MethodResult:
        """Execute Method 0: Optimal capability repair on ground truth graph G*."""
        start_time = time.perf_counter()

        # Oracle operates on G* directly where the latent edge is observed
        repair_engine = CapabilityGreedyRepairEngine(
            config=RepairConfig(
                max_iterations=10,
                residual_risk_threshold_delta=self.config.risk_threshold,
                strict_baseline_conformance=False,
            ),
            capability_model=self.capability_model,
        )

        gt_patch = repair_engine.repair(
            graph=env.ground_truth_graph,
            source_id=env.source_id,
            target_id=env.target_id,
            workload_logs=env.workload_split.t1_requests,
            completions=None,
        )

        # Apply the synthesized transformations to Go to create the repaired graph
        repaired_go = env.observed_graph.clone()
        for trans in gt_patch.applied_transformations:
            repaired_go = apply_transformation_to_graph(repaired_go, trans)

        elapsed = time.perf_counter() - start_time

        report = self.verifier.verify_patch(
            observed_graph=env.observed_graph,
            repaired_graph=repaired_go,
            source_id=env.source_id,
            target_id=env.target_id,
            applied_transformations=gt_patch.applied_transformations,
            completions=completions,
            ground_truth_graph=env.ground_truth_graph,
            t1_workload=env.workload_split.t1_requests,
            t2_workload=env.workload_split.t2_requests,
            check_side_effects=True,
            risk_threshold=self.config.risk_threshold,
        )

        t1_rep = report.t1_replay
        t2_rep = report.t2_replay

        return MethodResult(
            method=RemediationMethod.ORACLE,
            method_name="Method 0: Oracle (G*)",
            repaired_graph=repaired_go,
            applied_transformations=gt_patch.applied_transformations,
            verification_report=report,
            level1_risk=report.level1_risk_observed,
            level2_risk=report.level2_risk_completion_set,
            level3_risk=report.level3_risk_ground_truth
            if report.level3_risk_ground_truth is not None
            else 0.0,
            is_safe_gt=bool(report.level3_safe),
            is_misleading=report.is_misleading,
            t1_breakage=t1_rep.breakage_rate if t1_rep else 0.0,
            t2_breakage=t2_rep.breakage_rate if t2_rep else 0.0,
            total_t1_requests=t1_rep.total_requests if t1_rep else 0,
            total_t2_requests=t2_rep.total_requests if t2_rep else 0,
            t1_denied_requests=t1_rep.denied_requests if t1_rep else 0,
            t2_denied_requests=t2_rep.denied_requests if t2_rep else 0,
            has_side_effects=report.side_effects.has_side_effects if report.side_effects else False,
            num_transformations=len(gt_patch.applied_transformations),
            transformation_cost=gt_patch.total_repair_cost,
            execution_time_sec=elapsed,
            details="Theoretical upper bound with zero-unobservability assumptions.",
        )

    # --------------------------------------------------------------------------
    # Method 1: No Action (Identity Patch)
    # --------------------------------------------------------------------------
    def run_method_1_no_action(
        self,
        env: TradeoffEnvironment,
        completions: list[LatentCompletion],
    ) -> MethodResult:
        """Execute Method 1: Baseline identity patch (zero changes to Go)."""
        start_time = time.perf_counter()
        repaired_go = env.observed_graph.clone()
        elapsed = time.perf_counter() - start_time

        report = self.verifier.verify_patch(
            observed_graph=env.observed_graph,
            repaired_graph=repaired_go,
            source_id=env.source_id,
            target_id=env.target_id,
            applied_transformations=[],
            completions=completions,
            ground_truth_graph=env.ground_truth_graph,
            t1_workload=env.workload_split.t1_requests,
            t2_workload=env.workload_split.t2_requests,
            check_side_effects=True,
            risk_threshold=self.config.risk_threshold,
        )

        t1_rep = report.t1_replay
        t2_rep = report.t2_replay

        return MethodResult(
            method=RemediationMethod.NO_ACTION,
            method_name="Method 1: No Action (Identity)",
            repaired_graph=repaired_go,
            applied_transformations=[],
            verification_report=report,
            level1_risk=report.level1_risk_observed,
            level2_risk=report.level2_risk_completion_set,
            level3_risk=report.level3_risk_ground_truth
            if report.level3_risk_ground_truth is not None
            else 1.0,
            is_safe_gt=bool(report.level3_safe),
            is_misleading=report.is_misleading,
            t1_breakage=t1_rep.breakage_rate if t1_rep else 0.0,
            t2_breakage=t2_rep.breakage_rate if t2_rep else 0.0,
            total_t1_requests=t1_rep.total_requests if t1_rep else 0,
            total_t2_requests=t2_rep.total_requests if t2_rep else 0,
            t1_denied_requests=t1_rep.denied_requests if t1_rep else 0,
            t2_denied_requests=t2_rep.denied_requests if t2_rep else 0,
            has_side_effects=False,
            num_transformations=0,
            transformation_cost=0.0,
            execution_time_sec=elapsed,
            details="Identity baseline: demonstrates deceptive safety on Go (Condition E).",
        )

    # --------------------------------------------------------------------------
    # Method 2: Coarse Action Deletion (IAMPERE-Style)
    # --------------------------------------------------------------------------
    def run_method_2_coarse_deletion(
        self,
        env: TradeoffEnvironment,
        completions: list[LatentCompletion],
    ) -> MethodResult:
        """Execute Method 2: Coarse least-privilege action removal (IAMPERE heuristic).

        Removes the dangerous delegation action (e.g. iam:PassRole or cut actions)
        entirely from all statements on the source principal.
        """
        start_time = time.perf_counter()
        repaired_go = env.observed_graph.clone()
        transformations: list[PolicyTransformation] = []

        # Identify policies associated with the source principal
        source_node = repaired_go.get_node(env.source_id) or repaired_go.get_node_by_arn(
            env.source_id
        )
        if source_node is None:
            raise KeyError(f"Source principal '{env.source_id}' not found in observed graph.")

        attached_pols: list[str] = []
        for edge in repaired_go.get_edges(
            source=source_node.id, relation=EdgeRelation.ATTACHED_WITH
        ):
            attached_pols.append(edge.target)

        # Candidate escalation actions to excise broadly
        target_actions = {"iam:passrole", "sts:assumerole", "lambda:createfunction"}

        total_cost = 0.0
        for pol_id in attached_pols:
            pol_node = repaired_go.get_node(pol_id)
            if pol_node is None or pol_node.policy_document is None:
                continue

            new_statements: list[Statement] = []
            for s_idx, stmt in enumerate(pol_node.policy_document.statements):
                if stmt.effect != Effect.ALLOW:
                    new_statements.append(stmt.model_copy(deep=True))
                    continue

                remaining_actions: list[str] = []
                for act in stmt.action:
                    if normalize_action_name(act) in target_actions:
                        trans = RemoveActionTransformation(
                            policy_id=pol_id,
                            statement_index=s_idx,
                            action=act,
                            cost=5.0,
                            description=f"Coarse deletion of '{act}' from {pol_id}",
                        )
                        transformations.append(trans)
                        total_cost += trans.cost
                    else:
                        remaining_actions.append(act)

                if remaining_actions:
                    new_statements.append(
                        stmt.model_copy(update={"action": remaining_actions}, deep=True)
                    )

            updated_doc = PolicyDocument(Statement=new_statements)
            repaired_go.add_node(
                pol_node.model_copy(update={"policy_document": updated_doc}, deep=True)
            )

        # Sever graph edges associated with excised actions
        for out_edge in list(repaired_go.get_edges(source=source_node.id)):
            if out_edge.relation in (EdgeRelation.PASSES_TO, EdgeRelation.ACTS_ON):
                valid_acts = [
                    a for a in out_edge.actions if normalize_action_name(a) not in target_actions
                ]
                if not valid_acts:
                    repaired_go.remove_edge(out_edge.source, out_edge.target, out_edge.relation)
                elif len(valid_acts) < len(out_edge.actions):
                    repaired_go.remove_edge(out_edge.source, out_edge.target, out_edge.relation)
                    repaired_go.add_edge(
                        out_edge.model_copy(update={"actions": valid_acts}, deep=True)
                    )

        elapsed = time.perf_counter() - start_time

        report = self.verifier.verify_patch(
            observed_graph=env.observed_graph,
            repaired_graph=repaired_go,
            source_id=env.source_id,
            target_id=env.target_id,
            applied_transformations=transformations,
            completions=completions,
            ground_truth_graph=env.ground_truth_graph,
            t1_workload=env.workload_split.t1_requests,
            t2_workload=env.workload_split.t2_requests,
            check_side_effects=True,
            risk_threshold=self.config.risk_threshold,
        )

        t1_rep = report.t1_replay
        t2_rep = report.t2_replay

        return MethodResult(
            method=RemediationMethod.COARSE_DELETION,
            method_name="Method 2: Coarse Deletion (IAMPERE)",
            repaired_graph=repaired_go,
            applied_transformations=transformations,
            verification_report=report,
            level1_risk=report.level1_risk_observed,
            level2_risk=report.level2_risk_completion_set,
            level3_risk=report.level3_risk_ground_truth
            if report.level3_risk_ground_truth is not None
            else 0.0,
            is_safe_gt=bool(report.level3_safe),
            is_misleading=report.is_misleading,
            t1_breakage=t1_rep.breakage_rate if t1_rep else 0.0,
            t2_breakage=t2_rep.breakage_rate if t2_rep else 0.0,
            total_t1_requests=t1_rep.total_requests if t1_rep else 0,
            total_t2_requests=t2_rep.total_requests if t2_rep else 0,
            t1_denied_requests=t1_rep.denied_requests if t1_rep else 0,
            t2_denied_requests=t2_rep.denied_requests if t2_rep else 0,
            has_side_effects=report.side_effects.has_side_effects if report.side_effects else False,
            num_transformations=len(transformations),
            transformation_cost=total_cost,
            execution_time_sec=elapsed,
            details="Coarse action removal: blocks attack but breaks benign workload using the action.",
        )

    # --------------------------------------------------------------------------
    # Method 3: Log-Only Tightening on T1 (PolicyRefiner-Style)
    # --------------------------------------------------------------------------
    def run_method_3_log_only_tightening(
        self,
        env: TradeoffEnvironment,
        completions: list[LatentCompletion],
    ) -> MethodResult:
        """Execute Method 3: Log-only tightening to evidenced T1 activity (PolicyRefiner heuristic).

        Synthesizes policies restricted strictly to the resources and actions evidenced
        during observation window T1. Suffers breakage when novel T2 operational drift arrives.
        """
        start_time = time.perf_counter()
        repaired_go = env.observed_graph.clone()
        transformations: list[PolicyTransformation] = []

        source_node = repaired_go.get_node(env.source_id) or repaired_go.get_node_by_arn(
            env.source_id
        )
        if source_node is None:
            raise KeyError(f"Source principal '{env.source_id}' not found.")

        t1_logs = env.workload_split.t1_requests
        total_cost = 0.0

        # Find all policies attached to source principal
        for edge in repaired_go.get_edges(
            source=source_node.id, relation=EdgeRelation.ATTACHED_WITH
        ):
            pol_node = repaired_go.get_node(edge.target)
            if pol_node is None or pol_node.policy_document is None:
                continue

            new_statements: list[Statement] = []
            for s_idx, stmt in enumerate(pol_node.policy_document.statements):
                if stmt.effect != Effect.ALLOW:
                    new_statements.append(stmt.model_copy(deep=True))
                    continue

                for act in stmt.action:
                    # Collect exact resource ARNs observed in T1 for (source, act)
                    evidenced_resources = {
                        req.resource_arn
                        for req in t1_logs
                        if req.principal_arn.lower()
                        in (source_node.arn.lower(), source_node.id.lower())
                        and action_matches_pattern(req.action, act)
                        and req.resource_arn != "*"
                    }

                    if evidenced_resources:
                        # Scope Resource strictly to evidenced T1 set
                        scoped_stmt = Statement(
                            Sid=f"{stmt.sid or 'Stmt'}_{act.replace(':', '_')}",
                            Effect=Effect.ALLOW,
                            Action=[act],
                            Resource=sorted(evidenced_resources),
                        )
                        new_statements.append(scoped_stmt)
                        trans = ScopeResourceTransformation(
                            policy_id=pol_node.id,
                            statement_index=s_idx,
                            action=act,
                            target_resources=sorted(evidenced_resources),
                            original_resources=list(stmt.resource),
                            cost=1.0,
                            description=f"Log-only tightening of '{act}' to T1 evidenced resources",
                        )
                        transformations.append(trans)
                        total_cost += trans.cost
                    else:
                        # Action was uncalled in T1: log-only tools excise uncalled action
                        trans_rem = RemoveActionTransformation(
                            policy_id=pol_node.id,
                            statement_index=s_idx,
                            action=act,
                            cost=5.0,
                            description=f"Excise uncalled action '{act}' based on T1 logs",
                        )
                        transformations.append(trans_rem)
                        total_cost += trans_rem.cost

            updated_doc = PolicyDocument(Statement=new_statements)
            repaired_go.add_node(
                pol_node.model_copy(update={"policy_document": updated_doc}, deep=True)
            )

        # Sever edges from source principal targeting resources not evidenced in T1
        t1_targets = {
            req.resource_arn.lower()
            for req in t1_logs
            if req.principal_arn.lower() in (source_node.arn.lower(), source_node.id.lower())
        }
        for out_edge in list(repaired_go.get_edges(source=source_node.id)):
            tgt_n = repaired_go.get_node(out_edge.target)
            tgt_arn = tgt_n.arn.lower() if tgt_n else ""
            if (
                out_edge.relation in (EdgeRelation.PASSES_TO, EdgeRelation.ACTS_ON)
                and tgt_arn not in t1_targets
            ):
                repaired_go.remove_edge(out_edge.source, out_edge.target, out_edge.relation)

        elapsed = time.perf_counter() - start_time

        report = self.verifier.verify_patch(
            observed_graph=env.observed_graph,
            repaired_graph=repaired_go,
            source_id=env.source_id,
            target_id=env.target_id,
            applied_transformations=transformations,
            completions=completions,
            ground_truth_graph=env.ground_truth_graph,
            t1_workload=env.workload_split.t1_requests,
            t2_workload=env.workload_split.t2_requests,
            check_side_effects=True,
            risk_threshold=self.config.risk_threshold,
        )

        t1_rep = report.t1_replay
        t2_rep = report.t2_replay
        tot_cost = sum(t.cost for t in transformations)

        return MethodResult(
            method=RemediationMethod.LOG_ONLY_TIGHTENING,
            method_name="Method 3: Log-Only Tightening (PolicyRefiner)",
            repaired_graph=repaired_go,
            applied_transformations=transformations,
            verification_report=report,
            level1_risk=report.level1_risk_observed,
            level2_risk=report.level2_risk_completion_set,
            level3_risk=report.level3_risk_ground_truth
            if report.level3_risk_ground_truth is not None
            else 0.0,
            is_safe_gt=bool(report.level3_safe),
            is_misleading=report.is_misleading,
            t1_breakage=t1_rep.breakage_rate if t1_rep else 0.0,
            t2_breakage=t2_rep.breakage_rate if t2_rep else 0.0,
            total_t1_requests=t1_rep.total_requests if t1_rep else 0,
            total_t2_requests=t2_rep.total_requests if t2_rep else 0,
            t1_denied_requests=t1_rep.denied_requests if t1_rep else 0,
            t2_denied_requests=t2_rep.denied_requests if t2_rep else 0,
            has_side_effects=report.side_effects.has_side_effects if report.side_effects else False,
            num_transformations=len(transformations),
            transformation_cost=tot_cost,
            execution_time_sec=elapsed,
            details="Log-only tightening: 0% T1 breakage, high T2 breakage on operational drift.",
        )

    # --------------------------------------------------------------------------
    # Method 4a: Point-Estimate Repair (p >= 0.50 Threshold)
    # --------------------------------------------------------------------------
    def run_method_4a_point_estimate(
        self,
        env: TradeoffEnvironment,
        completions: list[LatentCompletion],
    ) -> MethodResult:
        """Execute Method 4a: Materializes single deterministic graph from p >= theta edges.

        Fails to sever latent paths if predicted edge probability is below theta (calibration error).
        """
        start_time = time.perf_counter()

        # Materialize single deterministic graph G_hat
        ghat = env.observed_graph.clone()
        for cand in env.candidate_edges:
            if cand.probability >= self.config.point_estimate_threshold and not ghat.has_edge(
                cand.source, cand.target, cand.relation
            ):
                ghat.add_edge(cand.to_graph_edge())

        # Check if ghat contains an attack path
        s_id = env.source_id
        t_id = env.target_id
        path_exists = nx.has_path(ghat.nx_graph, s_id, t_id)

        transformations: list[PolicyTransformation] = []
        tot_cost = 0.0

        if path_exists:
            # Run repair engine on ghat
            repair_engine = CapabilityGreedyRepairEngine(
                config=RepairConfig(
                    max_iterations=10,
                    residual_risk_threshold_delta=self.config.risk_threshold,
                    strict_baseline_conformance=False,
                ),
                capability_model=self.capability_model,
            )
            patch_res = repair_engine.repair(
                graph=ghat,
                source_id=s_id,
                target_id=t_id,
                workload_logs=env.workload_split.t1_requests,
                completions=None,
            )
            transformations = patch_res.applied_transformations
            tot_cost = patch_res.total_repair_cost

        # Apply to Go
        repaired_go = env.observed_graph.clone()
        for trans in transformations:
            repaired_go = apply_transformation_to_graph(repaired_go, trans)

        elapsed = time.perf_counter() - start_time

        report = self.verifier.verify_patch(
            observed_graph=env.observed_graph,
            repaired_graph=repaired_go,
            source_id=env.source_id,
            target_id=env.target_id,
            applied_transformations=transformations,
            completions=completions,
            ground_truth_graph=env.ground_truth_graph,
            t1_workload=env.workload_split.t1_requests,
            t2_workload=env.workload_split.t2_requests,
            check_side_effects=True,
            risk_threshold=self.config.risk_threshold,
        )

        t1_rep = report.t1_replay
        t2_rep = report.t2_replay

        return MethodResult(
            method=RemediationMethod.POINT_ESTIMATE,
            method_name="Method 4a: Point-Estimate (p >= 0.50)",
            repaired_graph=repaired_go,
            applied_transformations=transformations,
            verification_report=report,
            level1_risk=report.level1_risk_observed,
            level2_risk=report.level2_risk_completion_set,
            level3_risk=report.level3_risk_ground_truth
            if report.level3_risk_ground_truth is not None
            else 1.0,
            is_safe_gt=bool(report.level3_safe),
            is_misleading=report.is_misleading,
            t1_breakage=t1_rep.breakage_rate if t1_rep else 0.0,
            t2_breakage=t2_rep.breakage_rate if t2_rep else 0.0,
            total_t1_requests=t1_rep.total_requests if t1_rep else 0,
            total_t2_requests=t2_rep.total_requests if t2_rep else 0,
            t1_denied_requests=t1_rep.denied_requests if t1_rep else 0,
            t2_denied_requests=t2_rep.denied_requests if t2_rep else 0,
            has_side_effects=report.side_effects.has_side_effects if report.side_effects else False,
            num_transformations=len(transformations),
            transformation_cost=tot_cost,
            execution_time_sec=elapsed,
            details="Point-estimate repair: brittle when probability falls below hard threshold.",
        )

    # --------------------------------------------------------------------------
    # Method 4b: Uncertainty-Aware Repair (Ours: M Completions)
    # --------------------------------------------------------------------------
    def run_method_4b_uncertainty_aware(
        self,
        env: TradeoffEnvironment,
        completions: list[LatentCompletion],
    ) -> MethodResult:
        """Execute Method 4b: Full stochastic completion sampling + capability repair.

        Evaluates latent completions, computes uncertainty-weighted candidate cut edges,
        and applies capability-guided scoping C(a, r, c) guaranteeing 100% T1 conformance.
        """
        start_time = time.perf_counter()

        # Step 1: Uncertainty-guided candidate cut pruning across completions
        pruner = CandidatePruner(
            config=PruningConfig(
                max_candidates=self.config.max_candidates,
                only_observed=True,
                weight_frequency=0.6,
                weight_centrality=0.2,
                weight_capability=0.2,
            ),
            capability_model=self.capability_model,
        )

        pruning_res = pruner.prune_from_completions(
            graph=env.observed_graph,
            completions=completions,
            source_ids=[env.source_id],
            target_ids=[env.target_id],
        )

        # Step 2: Capability-guided greedy repair
        repair_engine = CapabilityGreedyRepairEngine(
            config=RepairConfig(
                max_iterations=10,
                residual_risk_threshold_delta=self.config.risk_threshold,
                strict_baseline_conformance=self.config.strict_baseline_conformance,
            ),
            capability_model=self.capability_model,
        )

        patch_res = repair_engine.repair(
            graph=env.observed_graph,
            source_id=env.source_id,
            target_id=env.target_id,
            workload_logs=env.workload_split.t1_requests,
            completions=completions,
            pruning_result=pruning_res,
        )

        elapsed = time.perf_counter() - start_time

        report = self.verifier.verify_patch(
            observed_graph=env.observed_graph,
            repaired_graph=patch_res.repaired_graph,
            source_id=env.source_id,
            target_id=env.target_id,
            applied_transformations=patch_res.applied_transformations,
            completions=completions,
            ground_truth_graph=env.ground_truth_graph,
            t1_workload=env.workload_split.t1_requests,
            t2_workload=env.workload_split.t2_requests,
            check_side_effects=True,
            risk_threshold=self.config.risk_threshold,
        )

        t1_rep = report.t1_replay
        t2_rep = report.t2_replay

        return MethodResult(
            method=RemediationMethod.UNCERTAINTY_AWARE,
            method_name="Method 4b: Uncertainty-Aware (Ours)",
            repaired_graph=patch_res.repaired_graph,
            applied_transformations=patch_res.applied_transformations,
            verification_report=report,
            level1_risk=report.level1_risk_observed,
            level2_risk=report.level2_risk_completion_set,
            level3_risk=report.level3_risk_ground_truth
            if report.level3_risk_ground_truth is not None
            else 0.0,
            is_safe_gt=bool(report.level3_safe),
            is_misleading=report.is_misleading,
            t1_breakage=t1_rep.breakage_rate if t1_rep else 0.0,
            t2_breakage=t2_rep.breakage_rate if t2_rep else 0.0,
            total_t1_requests=t1_rep.total_requests if t1_rep else 0,
            total_t2_requests=t2_rep.total_requests if t2_rep else 0,
            t1_denied_requests=t1_rep.denied_requests if t1_rep else 0,
            t2_denied_requests=t2_rep.denied_requests if t2_rep else 0,
            has_side_effects=report.side_effects.has_side_effects if report.side_effects else False,
            num_transformations=len(patch_res.applied_transformations),
            transformation_cost=patch_res.total_repair_cost,
            execution_time_sec=elapsed,
            details="Uncertainty-aware repair: cuts latent attack distribution while preserving 100% T1.",
        )

    # --------------------------------------------------------------------------
    # Orchestrator: Run All 6 Methods
    # --------------------------------------------------------------------------
    def run_all_methods(
        self,
        env: TradeoffEnvironment,
        seed: int = 42,
    ) -> dict[RemediationMethod, MethodResult]:
        """Execute all 6 remediation strategies against a single benchmark environment."""
        # Pre-generate stochastic completions for consistency across methods
        sampler = LatentCompletionSampler(
            observed_graph=env.observed_graph,
            candidate_edges=env.candidate_edges,
        )
        completions = sampler.sample_completions(
            num_samples=self.config.num_completions,
            seed=seed,
        )

        results: dict[RemediationMethod, MethodResult] = {}

        results[RemediationMethod.ORACLE] = self.run_method_0_oracle(env, completions)
        results[RemediationMethod.NO_ACTION] = self.run_method_1_no_action(env, completions)
        results[RemediationMethod.COARSE_DELETION] = self.run_method_2_coarse_deletion(
            env, completions
        )
        results[RemediationMethod.LOG_ONLY_TIGHTENING] = self.run_method_3_log_only_tightening(
            env, completions
        )
        results[RemediationMethod.POINT_ESTIMATE] = self.run_method_4a_point_estimate(
            env, completions
        )
        results[RemediationMethod.UNCERTAINTY_AWARE] = self.run_method_4b_uncertainty_aware(
            env, completions
        )

        return results
