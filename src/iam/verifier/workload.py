"""Synthetic temporal authorization workload generator with controlled operational drift.

Implements Phase 3 Week 8 (Step 8.1):
- Generates realistic CloudTrail-like authorization activity traces D for benign identities.
- Partitions timeline into observation synthesis window T_1 and holdout evaluation window T_2.
- Injects parameterized operational drift into T_2 (resource drift and infrequent action drift).
- Guarantees 100% baseline conformance: every event in T_1 and T_2 evaluates to ALLOW on G_o.
- Quantifies temporal disjointness, drift intensity, and resource coverage.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from iam.generator.graph import (
    EdgeRelation,
    GraphNode,
    IAMGraph,
    NodeType,
)
from iam.optimizer.repair import WorkloadConformanceChecker
from iam.parser.capability import CapabilityModel, load_default_capability_model
from iam.parser.evaluator import AuthRequest
from iam.parser.schema import Effect
from iam.parser.wildcard import normalize_action_name

logger = logging.getLogger(__name__)


class DriftType(str, Enum):
    """Typology of legitimate operational drift in temporal authorization holdouts."""

    RESOURCE_DRIFT = "resource_drift"
    ACTION_DRIFT = "action_drift"


class WorkloadEvent(BaseModel):
    """An authorized historical or holdout CloudTrail authorization request event."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(description="Unique event UUID")
    principal_arn: str = Field(description="IAM principal calling the service action")
    action: str = Field(description="AWS action name (e.g. s3:GetObject)")
    resource_arn: str = Field(description="Target resource ARN evaluated")
    day_offset: float = Field(description="Temporal timestamp in fractional days from epoch")
    department: str = Field(default="", description="Originating principal department")
    service: str = Field(default="", description="AWS service prefix (e.g. s3, kms, ec2)")
    is_drift: bool = Field(default=False, description="True if event represents operational drift")
    drift_type: DriftType | None = Field(
        default=None, description="Drift classification if applicable"
    )
    context: dict[str, Any] = Field(
        default_factory=dict, description="Request contextual parameters"
    )

    def to_auth_request(self) -> AuthRequest:
        """Convert to symbolic AuthRequest for evaluator and repair engine ingestion."""
        return AuthRequest(
            principal_arn=self.principal_arn,
            action=self.action,
            resource_arn=self.resource_arn,
            context=self.context,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize workload event to dictionary."""
        return {
            "event_id": self.event_id,
            "principal_arn": self.principal_arn,
            "action": self.action,
            "resource_arn": self.resource_arn,
            "day_offset": round(self.day_offset, 2),
            "department": self.department,
            "service": self.service,
            "is_drift": self.is_drift,
            "drift_type": self.drift_type.value if self.drift_type else None,
        }


class DriftConfig(BaseModel):
    """Configuration governing operational drift injected into the T_2 holdout window."""

    model_config = ConfigDict(frozen=True)

    drift_rate: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Fraction of T_2 events exhibiting drift"
    )
    resource_drift_ratio: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Proportion of drift representing unseen resource ARNs",
    )
    action_drift_ratio: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Proportion of drift representing infrequent actions",
    )
    new_resources_per_principal: int = Field(
        default=3, ge=1, description="Number of novel resource ARNs synthesized per principal"
    )


class TemporalSplitConfig(BaseModel):
    """Configuration for generating synthetic temporal authorization workload traces."""

    model_config = ConfigDict(frozen=True)

    t1_days: int = Field(default=90, ge=1, description="Duration of observation window T_1 in days")
    t2_days: int = Field(
        default=30, ge=1, description="Duration of holdout evaluation window T_2 in days"
    )
    events_per_day: int = Field(
        default=20, ge=1, description="Average number of events generated per day"
    )
    seed: int = Field(default=42, description="Deterministic random seed for reproducibility")
    drift_config: DriftConfig = Field(
        default_factory=DriftConfig, description="Drift parameters for T_2"
    )


class TemporalWorkloadSplit(BaseModel):
    """A temporally partitioned workload dataset containing T_1 and T_2 authorization activity traces."""

    model_config = ConfigDict(frozen=True)

    t1_events: list[WorkloadEvent] = Field(
        default_factory=list, description="Synthesis window T_1 events"
    )
    t2_events: list[WorkloadEvent] = Field(
        default_factory=list, description="Holdout window T_2 events"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata and configuration parameters"
    )

    @property
    def t1_requests(self) -> list[AuthRequest]:
        """Convert T_1 events to symbolic AuthRequest sequence."""
        return [e.to_auth_request() for e in self.t1_events]

    @property
    def t2_requests(self) -> list[AuthRequest]:
        """Convert T_2 events to symbolic AuthRequest sequence."""
        return [e.to_auth_request() for e in self.t2_events]

    @property
    def num_t1(self) -> int:
        """Count of events in synthesis window T_1."""
        return len(self.t1_events)

    @property
    def num_t2(self) -> int:
        """Count of events in holdout window T_2."""
        return len(self.t2_events)

    @property
    def total_events(self) -> int:
        """Total events across both temporal windows."""
        return self.num_t1 + self.num_t2

    @property
    def drift_events(self) -> list[WorkloadEvent]:
        """List of all events in T_2 exhibiting operational drift."""
        return [e for e in self.t2_events if e.is_drift]

    @property
    def drift_count(self) -> int:
        """Count of operational drift events in T_2."""
        return len(self.drift_events)

    @property
    def drift_rate(self) -> float:
        """Empirical operational drift rate in holdout window T_2."""
        return self.drift_count / float(self.num_t2) if self.num_t2 > 0 else 0.0

    @property
    def unique_principals(self) -> set[str]:
        """Set of unique principal ARNs present in the combined workload."""
        return {e.principal_arn for e in self.t1_events}.union(
            {e.principal_arn for e in self.t2_events}
        )

    @property
    def unique_resources_t1(self) -> set[str]:
        """Set of unique resource ARNs observed during T_1."""
        return {e.resource_arn for e in self.t1_events}

    @property
    def unique_resources_t2(self) -> set[str]:
        """Set of unique resource ARNs observed during T_2."""
        return {e.resource_arn for e in self.t2_events}

    @property
    def unique_actions_t1(self) -> set[str]:
        """Set of unique actions exercised during T_1."""
        return {e.action for e in self.t1_events}

    @property
    def unique_actions_t2(self) -> set[str]:
        """Set of unique actions exercised during T_2."""
        return {e.action for e in self.t2_events}

    @property
    def new_resources_in_t2(self) -> set[str]:
        """Resource ARNs exercised in T_2 that were strictly unseen during T_1."""
        return self.unique_resources_t2.difference(self.unique_resources_t1)

    @property
    def new_actions_in_t2(self) -> set[str]:
        """Actions exercised in T_2 that were strictly unseen during T_1."""
        return self.unique_actions_t2.difference(self.unique_actions_t1)

    def summary_dict(self) -> dict[str, Any]:
        """Serialize dataset summary metrics for logging and evaluation reports."""
        return {
            "num_t1_events": self.num_t1,
            "num_t2_events": self.num_t2,
            "total_events": self.total_events,
            "num_drift_events": self.drift_count,
            "drift_rate": round(self.drift_rate, 4),
            "num_unique_principals": len(self.unique_principals),
            "num_unique_resources_t1": len(self.unique_resources_t1),
            "num_unique_resources_t2": len(self.unique_resources_t2),
            "num_new_resources_t2": len(self.new_resources_in_t2),
            "num_new_actions_t2": len(self.new_actions_in_t2),
        }


class SyntheticWorkloadGenerator:
    """Enterprise CloudTrail activity trace generator with temporal holdout and drift.

    Discovers authorized permissions across IAMGraph entities, profiles high-frequency
    and infrequent workflows, and synthesizes temporal traces satisfying 100% baseline conformance.
    """

    # Common AWS benign action taxonomy
    ROUTINE_ACTIONS: set[str] = {
        "s3:GetObject",
        "s3:ListBucket",
        "ec2:DescribeInstances",
        "ec2:DescribeSecurityGroups",
        "lambda:InvokeFunction",
        "kms:Decrypt",
        "dynamodb:GetItem",
        "dynamodb:Query",
        "logs:GetLogEvents",
        "iam:PassRole",
    }

    INFREQUENT_ACTIONS: set[str] = {
        "iam:CreateAccessKey",
        "kms:ReEncrypt",
        "kms:GenerateDataKey",
        "s3:PutObjectTagging",
        "lambda:UpdateFunctionCode",
        "ec2:StartInstances",
        "ec2:StopInstances",
        "dynamodb:PutItem",
    }

    def __init__(
        self,
        config: TemporalSplitConfig | None = None,
        capability_model: CapabilityModel | None = None,
    ) -> None:
        """Initialize synthetic workload generator with split configuration."""
        self.config: TemporalSplitConfig = config or TemporalSplitConfig()
        self.capability_model: CapabilityModel = (
            capability_model if capability_model is not None else load_default_capability_model()
        )
        self.conformance_checker: WorkloadConformanceChecker = WorkloadConformanceChecker(
            self.capability_model
        )

    def _extract_principal_operations(
        self,
        graph: IAMGraph,
        principal: GraphNode,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Discover authorized (action, resource_arn) operations for a principal in G_o.

        Returns:
            Tuple of (routine_operations, infrequent_operations).
        """
        routine_ops: list[tuple[str, str]] = []
        infrequent_ops: list[tuple[str, str]] = []
        seen_ops: set[tuple[str, str]] = set()

        for out_edge in graph.get_edges(source=principal.id):
            target_node = graph.get_node(out_edge.target)
            if target_node is None:
                continue

            # Benign baseline operational workloads for non-admin identities do not execute
            # adversarial bridge relations or access high-value / administrative targets
            if not principal.is_admin and (
                out_edge.is_bridge or target_node.is_admin or target_node.is_high_value
            ):
                continue

            actions_to_test = list(out_edge.actions)
            if out_edge.relation == EdgeRelation.PASSES_TO and not actions_to_test:
                actions_to_test = ["iam:PassRole"]
            elif out_edge.relation == EdgeRelation.ASSUMES_ROLE and not actions_to_test:
                actions_to_test = ["sts:AssumeRole"]

            for act in actions_to_test:
                req = AuthRequest(
                    principal_arn=principal.arn,
                    action=act,
                    resource_arn=target_node.arn,
                )
                if self.conformance_checker.evaluate_request(graph, req).is_allowed:
                    op_key = (act, target_node.arn)
                    if op_key not in seen_ops:
                        seen_ops.add(op_key)
                        norm_act = normalize_action_name(act)
                        if any(
                            norm_act == normalize_action_name(ia) for ia in self.INFREQUENT_ACTIONS
                        ):
                            infrequent_ops.append(op_key)
                        else:
                            routine_ops.append(op_key)

        # 2. Inspect attached identity policies
        policies = self.conformance_checker.get_principal_policies(graph, principal)
        candidate_resources = [n for n in graph.get_nodes() if n.node_type == NodeType.RESOURCE]

        for pol in policies:
            for stmt in pol.statements:
                if stmt.effect != Effect.ALLOW:
                    continue
                for act in stmt.action:
                    # Test against explicit resources in statement
                    for r_pattern in stmt.resource:
                        if r_pattern != "*" and not r_pattern.endswith("*"):
                            req = AuthRequest(
                                principal_arn=principal.arn,
                                action=act,
                                resource_arn=r_pattern,
                            )
                            if self.conformance_checker.evaluate_request(graph, req).is_allowed:
                                op_key = (act, r_pattern)
                                if op_key not in seen_ops:
                                    seen_ops.add(op_key)
                                    norm_act = normalize_action_name(act)
                                    if any(
                                        norm_act == normalize_action_name(ia)
                                        for ia in self.INFREQUENT_ACTIONS
                                    ):
                                        infrequent_ops.append(op_key)
                                    else:
                                        routine_ops.append(op_key)

                    # If wildcard or prefix, test matching resources in graph
                    if any("*" in r for r in stmt.resource):
                        for c_res in candidate_resources:
                            # Verify target resource conforms to AWS Action Capability Model C(a, r, c)
                            if not self.capability_model.validate_transformation(
                                act, c_res.arn
                            ).is_valid:
                                continue
                            if not principal.is_admin and (c_res.is_admin or c_res.is_high_value):
                                continue

                            req = AuthRequest(
                                principal_arn=principal.arn,
                                action=act,
                                resource_arn=c_res.arn,
                            )
                            if self.conformance_checker.evaluate_request(graph, req).is_allowed:
                                op_key = (act, c_res.arn)
                                if op_key not in seen_ops:
                                    seen_ops.add(op_key)
                                    norm_act = normalize_action_name(act)
                                    if any(
                                        norm_act == normalize_action_name(ia)
                                        for ia in self.INFREQUENT_ACTIONS
                                    ):
                                        infrequent_ops.append(op_key)
                                    else:
                                        routine_ops.append(op_key)

        return routine_ops, infrequent_ops

    def _synthesize_drift_resources(
        self,
        graph: IAMGraph,
        principal: GraphNode,
        routine_ops: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """Synthesize novel authorized resource ARNs to simulate T_2 resource drift.

        Creates legitimate new resource ARNs (e.g. new daily S3 keys or KMS keys)
        matching the principal's allowed action patterns that evaluate to ALLOW on G_o.
        """
        drift_ops: list[tuple[str, str]] = []
        account_id = principal.account_id or "123456789012"
        dept_slug = (principal.department or "ops").lower().replace("/", "-")
        k = self.config.drift_config.new_resources_per_principal

        for act, _base_res in routine_ops:
            service = act.split(":")[0].lower()
            for idx in range(k):
                novel_res: str | None = None
                if service == "s3":
                    # Generate new partition or bucket ARN
                    novel_res = f"arn:aws:s3:::{dept_slug}-partition-2026-q4-{idx}/report.parquet"
                elif service == "kms":
                    novel_res = (
                        f"arn:aws:kms:us-east-1:{account_id}:key/drift-key-{dept_slug}-{idx}"
                    )
                elif service == "lambda":
                    novel_res = f"arn:aws:lambda:us-east-1:{account_id}:function:drift-worker-{dept_slug}-{idx}"
                elif service == "ec2":
                    novel_res = (
                        f"arn:aws:ec2:us-east-1:{account_id}:instance/i-drift-{dept_slug}-{idx}"
                    )

                if novel_res is not None:
                    req = AuthRequest(
                        principal_arn=principal.arn,
                        action=act,
                        resource_arn=novel_res,
                    )
                    # Strictly verify this novel resource is authorized under G_o
                    if self.conformance_checker.evaluate_request(graph, req).is_allowed:
                        drift_ops.append((act, novel_res))

        return drift_ops

    def generate(self, graph: IAMGraph) -> TemporalWorkloadSplit:
        """Synthesize temporal workload split D = (D^{T_1}, D^{T_2}) with operational drift.

        Args:
            graph: Base partially observed IAM authorization graph G_o.

        Returns:
            TemporalWorkloadSplit containing T_1 and T_2 event sequences.
        """
        rng = np.random.default_rng(self.config.seed)
        principals = [n for n in graph.get_nodes() if n.node_type in (NodeType.USER, NodeType.ROLE)]

        # 1. Profile authorized operations per principal
        principal_profiles: dict[str, dict[str, Any]] = {}
        for p in principals:
            routine_ops, infrequent_ops = self._extract_principal_operations(graph, p)
            if not routine_ops and not infrequent_ops:
                continue

            drift_ops = self._synthesize_drift_resources(graph, p, routine_ops)
            principal_profiles[p.arn] = {
                "node": p,
                "routine_ops": routine_ops,
                "infrequent_ops": infrequent_ops,
                "drift_ops": drift_ops,
            }

        if not principal_profiles:
            logger.warning("No active authorized principal operations found in graph.")
            return TemporalWorkloadSplit(
                t1_events=[],
                t2_events=[],
                metadata={"config": self.config.model_dump(), "status": "empty"},
            )

        p_arns = sorted(principal_profiles.keys())
        t1_events: list[WorkloadEvent] = []
        t2_events: list[WorkloadEvent] = []
        event_counter = 0

        t1_days = self.config.t1_days
        t2_days = self.config.t2_days
        rate = self.config.events_per_day
        drift_rate = self.config.drift_config.drift_rate
        res_drift_ratio = self.config.drift_config.resource_drift_ratio

        # 2. Generate Observation Window T_1: Days [0, t1_days)
        total_t1_target = max(10, t1_days * rate)
        for _ in range(total_t1_target):
            p_arn = str(rng.choice(p_arns))
            prof = principal_profiles[p_arn]
            p_node = prof["node"]
            ops = prof["routine_ops"]
            if not ops:
                ops = prof["infrequent_ops"]
            if not ops:
                continue

            op_idx = int(rng.integers(0, len(ops)))
            act, res = ops[op_idx]
            day_offset = float(rng.uniform(0.0, float(t1_days)))

            event_counter += 1
            t1_events.append(
                WorkloadEvent(
                    event_id=f"evt-t1-{event_counter:06d}",
                    principal_arn=p_arn,
                    action=act,
                    resource_arn=res,
                    day_offset=day_offset,
                    department=p_node.department or "",
                    service=act.split(":")[0],
                    is_drift=False,
                )
            )

        # 3. Generate Holdout Window T_2: Days [t1_days, t1_days + t2_days)
        total_t2_target = max(5, t2_days * rate)
        for _ in range(total_t2_target):
            p_arn = str(rng.choice(p_arns))
            prof = principal_profiles[p_arn]
            p_node = prof["node"]

            should_drift = bool(rng.random() < drift_rate)
            drift_type: DriftType | None = None
            is_drift = False

            if should_drift:
                is_drift_resource = bool(rng.random() < res_drift_ratio)
                if is_drift_resource and prof["drift_ops"]:
                    d_idx = int(rng.integers(0, len(prof["drift_ops"])))
                    act, res = prof["drift_ops"][d_idx]
                    drift_type = DriftType.RESOURCE_DRIFT
                    is_drift = True
                elif prof["infrequent_ops"]:
                    i_idx = int(rng.integers(0, len(prof["infrequent_ops"])))
                    act, res = prof["infrequent_ops"][i_idx]
                    drift_type = DriftType.ACTION_DRIFT
                    is_drift = True
                elif prof["routine_ops"]:
                    r_idx = int(rng.integers(0, len(prof["routine_ops"])))
                    act, res = prof["routine_ops"][r_idx]
                else:
                    continue
            else:
                ops = prof["routine_ops"] or prof["infrequent_ops"]
                if not ops:
                    continue
                r_idx = int(rng.integers(0, len(ops)))
                act, res = ops[r_idx]

            day_offset = float(rng.uniform(float(t1_days), float(t1_days + t2_days)))
            event_counter += 1
            t2_events.append(
                WorkloadEvent(
                    event_id=f"evt-t2-{event_counter:06d}",
                    principal_arn=p_arn,
                    action=act,
                    resource_arn=res,
                    day_offset=day_offset,
                    department=p_node.department or "",
                    service=act.split(":")[0],
                    is_drift=is_drift,
                    drift_type=drift_type,
                )
            )

        # Sort chronologically by day_offset
        t1_events.sort(key=lambda e: e.day_offset)
        t2_events.sort(key=lambda e: e.day_offset)

        meta: dict[str, Any] = {
            "config": self.config.model_dump(),
            "graph_id": graph.graph_id,
            "seed": self.config.seed,
            "num_active_principals": len(principal_profiles),
        }

        split = TemporalWorkloadSplit(
            t1_events=t1_events,
            t2_events=t2_events,
            metadata=meta,
        )

        logger.info(
            f"Generated TemporalWorkloadSplit: {split.num_t1} T_1 events, "
            f"{split.num_t2} T_2 events ({split.drift_count} drift, {split.drift_rate:.1%})"
        )
        return split
