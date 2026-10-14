"""Comprehensive unit and integration test suite for the CompletionSetVerifier.

Tests Phase 3 Week 8 (Step 8.2):
- Three-level verification boundary:
    Level 1: Observed graph G_o ⊕ P (static traversal)
    Level 2: Completion set { (G_o ∪ H_i) ⊕ P }_{i=1}^M (latent Monte Carlo reachability)
    Level 3: Ground-truth graph G* ⊕ P (synthetic counterfactual verification)
- Detection of misleading safety states (safe on G_o, but unsafe on completion set or ground truth).
- Escalation side-effect audit: detection of newly reachable (s', t') attack vectors.
- Operational workload replay: verification of 100% T_1 preservation and T_2 drift breakage measurement.
- Serialization and diagnostic reporting.
"""

from __future__ import annotations

import pytest

from iam.generator.graph import (
    EdgeRelation,
    GraphEdge,
    GraphNode,
    IAMGraph,
    NodeType,
)
from iam.generator.motifs import PassRoleLambdaMotif
from iam.optimizer.repair import (
    CapabilityGreedyRepairEngine,
)
from iam.optimizer.sampler import (
    CandidateEdge,
    LatentCompletion,
    LatentCompletionSampler,
)
from iam.parser.capability import CapabilityModel, load_default_capability_model
from iam.parser.evaluator import AuthRequest
from iam.parser.schema import Effect, PolicyDocument, Statement
from iam.verifier.verifier import (
    CompletionSetVerifier,
    SafetyStatus,
    SideEffectResult,
    VerificationReport,
    WorkloadReplayResult,
)
from iam.verifier.workload import (
    WorkloadEvent,
)


@pytest.fixture
def capability_model() -> CapabilityModel:
    """Fixture providing loaded AWS CapabilityModel."""
    return load_default_capability_model()


@pytest.fixture
def passrole_verification_setup() -> tuple[
    IAMGraph, IAMGraph, list[LatentCompletion], str, str, str
]:
    """Setup canonical PassRole + Lambda environment with Ground Truth G* and Observed G_o."""
    gt_graph = IAMGraph("passrole_gt_env")
    motif = PassRoleLambdaMotif()
    instance = motif.inject(gt_graph, suffix="verif")

    source_node = gt_graph.get_node(instance.source_id)
    assert source_node is not None
    source_pol = gt_graph.get_node("policy:lambda-creator-verif")
    assert source_pol is not None

    # Vulnerable un-scoped PassRole on *
    vulnerable_doc = PolicyDocument(
        Statement=[
            Statement(
                Sid="AllowPassRoleWildcard",
                Effect=Effect.ALLOW,
                Action=["iam:PassRole"],
                Resource=["*"],
            ),
            Statement(
                Sid="AllowLambdaOps",
                Effect=Effect.ALLOW,
                Action=["lambda:CreateFunction", "lambda:InvokeFunction"],
                Resource=["arn:aws:lambda:us-east-1:123456789012:function:escalator-verif"],
            ),
        ]
    )
    gt_graph.add_node(source_pol.model_copy(update={"policy_document": vulnerable_doc}, deep=True))

    # Observed graph Go has downstream KMS edge severed (Condition E)
    observed_graph = gt_graph.clone()
    kms_pol_id = "policy:kms-full-access-verif"
    target_id = instance.target_id
    observed_graph.remove_edge(kms_pol_id, target_id, EdgeRelation.ACTS_ON)

    # Candidate latent edge
    cand = CandidateEdge(
        source=kms_pol_id,
        target=target_id,
        relation=EdgeRelation.ACTS_ON,
        probability=0.85,
        actions=("kms:Decrypt", "kms:GenerateDataKey"),
        is_bridge=True,
    )

    sampler = LatentCompletionSampler(observed_graph=observed_graph, candidate_edges=[cand])
    completions = sampler.sample_completions(num_samples=20, seed=42)

    legit_worker_arn = "arn:aws:iam::123456789012:role/legit-worker-verif"

    return observed_graph, gt_graph, completions, instance.source_id, target_id, legit_worker_arn


# ==============================================================================
# 1. Three-Level Symbolic Verification Tests
# ==============================================================================


class TestThreeLevelVerificationBoundary:
    """Tests evaluating safety across Level 1 (Go), Level 2 (Completions), and Level 3 (Ground Truth)."""

    def test_detection_of_misleading_safety_on_unrepaired_baseline(
        self,
        passrole_verification_setup: tuple[
            IAMGraph, IAMGraph, list[LatentCompletion], str, str, str
        ],
        capability_model: CapabilityModel,
    ) -> None:
        """Formal test of Contribution 3: Unrepaired graph is safe on G_o but unsafe on completions and GT.

        In G_o, target is disconnected (Level 1 risk = 0.0 -> Safe).
        In completion set, latent attack paths exist (Level 2 risk = ~0.85 -> Unsafe).
        In G*, full attack path exists (Level 3 risk = 1.0 -> Unsafe).
        The verifier must flag this discrepancy as MISLEADING_SAFE.
        """
        observed_graph, gt_graph, completions, source_id, target_id, _ = passrole_verification_setup

        verifier = CompletionSetVerifier(capability_model=capability_model)

        # Evaluate the unrepaired baseline graph (identity patch / no patch)
        report = verifier.verify_patch(
            observed_graph=observed_graph,
            repaired_graph=observed_graph,
            source_id=source_id,
            target_id=target_id,
            completions=completions,
            ground_truth_graph=gt_graph,
        )

        assert isinstance(report, VerificationReport)
        # Level 1: Appears completely safe on observed graph Go!
        assert report.level1_risk_observed == 0.0
        assert report.level1_safe is True

        # Level 2: Unsafe on completion set!
        assert report.level2_risk_completion_set > 0.70
        assert report.level2_safe is False

        # Level 3: Unsafe on ground truth G*!
        assert report.level3_risk_ground_truth == 1.0
        assert report.level3_safe is False

        # Must be flagged as MISLEADING_SAFE
        assert report.is_misleading is True
        assert report.overall_status == SafetyStatus.MISLEADING_SAFE
        assert report.residual_risk_gap == 1.0

    def test_repaired_patch_verifies_safe_across_all_three_levels(
        self,
        passrole_verification_setup: tuple[
            IAMGraph, IAMGraph, list[LatentCompletion], str, str, str
        ],
        capability_model: CapabilityModel,
    ) -> None:
        """Verify that capability-aware greedy repair achieves provable safety across all 3 levels."""
        (
            observed_graph,
            gt_graph,
            completions,
            source_id,
            target_id,
            legit_worker_arn,
        ) = passrole_verification_setup

        source_node = observed_graph.get_node(source_id)
        assert source_node is not None

        workload = [
            AuthRequest(
                principal_arn=source_node.arn,
                action="iam:PassRole",
                resource_arn=legit_worker_arn,
            ),
            AuthRequest(
                principal_arn=source_node.arn,
                action="lambda:CreateFunction",
                resource_arn="arn:aws:lambda:us-east-1:123456789012:function:escalator-verif",
            ),
        ]

        # Synthesize capability-compliant greedy repair patch
        engine = CapabilityGreedyRepairEngine(capability_model=capability_model)
        patch = engine.repair(
            graph=observed_graph,
            source_id=source_id,
            target_id=target_id,
            workload_logs=workload,
            completions=completions,
        )

        assert patch.is_successful is True

        # Verify synthesized patch with CompletionSetVerifier
        verifier = CompletionSetVerifier(capability_model=capability_model)
        report = verifier.verify_patch(
            observed_graph=observed_graph,
            repaired_graph=patch.repaired_graph,
            source_id=source_id,
            target_id=target_id,
            applied_transformations=patch.applied_transformations,
            completions=completions,
            ground_truth_graph=gt_graph,
            t1_workload=workload,
        )

        # All 3 levels must report 0.0 risk and safe!
        assert report.level1_risk_observed == 0.0
        assert report.level1_safe is True
        assert report.level2_risk_completion_set == 0.0
        assert report.level2_safe is True
        assert report.level3_risk_ground_truth == 0.0
        assert report.level3_safe is True

        assert report.is_misleading is False
        assert report.overall_status == SafetyStatus.SAFE
        assert report.residual_risk_gap == 0.0

        # Workload T1 must be 100% conforming
        assert report.t1_replay is not None
        assert report.t1_replay.is_conforming is True
        assert report.t1_replay.denied_requests == 0


# ==============================================================================
# 2. Escalation Side-Effect Audit Tests
# ==============================================================================


class TestEscalationSideEffectAudit:
    """Tests detecting unintended privilege escalation vectors introduced by policy modifications."""

    def test_clean_repair_introduces_zero_side_effects(
        self,
        passrole_verification_setup: tuple[
            IAMGraph, IAMGraph, list[LatentCompletion], str, str, str
        ],
        capability_model: CapabilityModel,
    ) -> None:
        """Verify that a legitimate tightening repair introduces no new attack pairs."""
        observed_graph, _, _, source_id, target_id, legit_worker_arn = passrole_verification_setup
        source_node = observed_graph.get_node(source_id)
        assert source_node is not None

        workload = [
            AuthRequest(
                principal_arn=source_node.arn,
                action="iam:PassRole",
                resource_arn=legit_worker_arn,
            )
        ]

        engine = CapabilityGreedyRepairEngine(capability_model=capability_model)
        patch = engine.repair(
            graph=observed_graph,
            source_id=source_id,
            target_id=target_id,
            workload_logs=workload,
        )

        verifier = CompletionSetVerifier(capability_model=capability_model)
        side_effect_res = verifier.check_side_effects(observed_graph, patch.repaired_graph)

        assert isinstance(side_effect_res, SideEffectResult)
        assert side_effect_res.has_side_effects is False
        assert len(side_effect_res.new_escalation_pairs) == 0
        assert side_effect_res.num_pairs_checked > 0

    def test_detection_of_introduced_side_effect_vector(
        self,
        passrole_verification_setup: tuple[
            IAMGraph, IAMGraph, list[LatentCompletion], str, str, str
        ],
        capability_model: CapabilityModel,
    ) -> None:
        """Simulate a flawed patch that accidentally connects an intern user to an admin role."""
        observed_graph, _, _, _, target_id, _ = passrole_verification_setup

        # Create an innocent bystander user (e.g. Intern)
        intern_user = GraphNode(
            id="user:innocent-intern",
            node_type=NodeType.USER,
            arn="arn:aws:iam::123456789012:user/innocent-intern",
            name="innocent-intern",
            department="Interns",
        )
        observed_graph.add_node(intern_user)

        # Bad patch: accidentally adds a trust edge from intern to lambda admin role
        bad_repaired_graph = observed_graph.clone()
        bad_repaired_graph.add_edge(
            GraphEdge(
                source=intern_user.id,
                target="role:lambda-admin-exec-verif",
                relation=EdgeRelation.ASSUMES_ROLE,
                actions=["sts:AssumeRole"],
            )
        )

        verifier = CompletionSetVerifier(capability_model=capability_model)
        side_effect_res = verifier.check_side_effects(
            observed_graph,
            bad_repaired_graph,
            target_hv_resources=["role:lambda-admin-exec-verif"],
        )

        assert side_effect_res.has_side_effects is True
        assert (
            intern_user.id,
            "role:lambda-admin-exec-verif",
        ) in side_effect_res.new_escalation_pairs


# ==============================================================================
# 3. Operational Workload Replay & Breakage Tests
# ==============================================================================


class TestOperationalWorkloadReplay:
    """Tests replaying T_1 and T_2 activity traces to evaluate operational breakage."""

    def test_workload_replay_t1_and_t2_breakage_measurement(
        self, capability_model: CapabilityModel
    ) -> None:
        """Verify replay accurately computes breakage on holdout T_2 when an action is revoked."""
        graph = IAMGraph("replay_test_env")
        user = GraphNode(
            id="user:dev",
            node_type=NodeType.USER,
            arn="arn:aws:iam::123456789012:user/dev",
            name="dev",
        )
        pol = GraphNode(
            id="policy:dev-pol",
            node_type=NodeType.POLICY,
            arn="arn:aws:iam::123456789012:policy/dev-pol",
            name="dev-pol",
            policy_document=PolicyDocument(
                Statement=[
                    Statement(
                        Effect=Effect.ALLOW,
                        Action=["s3:GetObject", "s3:ListBucket"],
                        Resource=["*"],
                    )
                ]
            ),
        )
        graph.add_nodes([user, pol])
        graph.add_edge(
            GraphEdge(source=user.id, target=pol.id, relation=EdgeRelation.ATTACHED_WITH)
        )

        # Workload T1: Routine s3:GetObject
        t1_events = [
            WorkloadEvent(
                event_id="e1",
                principal_arn=user.arn,
                action="s3:GetObject",
                resource_arn="arn:aws:s3:::bucket1/file.txt",
                day_offset=10.0,
            ),
            WorkloadEvent(
                event_id="e2",
                principal_arn=user.arn,
                action="s3:ListBucket",
                resource_arn="arn:aws:s3:::bucket1",
                day_offset=20.0,
            ),
        ]

        # Workload T2: Has drift events using s3:GetObject on a new resource and s3:ListBucket
        t2_events = [
            WorkloadEvent(
                event_id="e3",
                principal_arn=user.arn,
                action="s3:GetObject",
                resource_arn="arn:aws:s3:::bucket2_drift/new.txt",
                day_offset=95.0,
                is_drift=True,
            ),
            WorkloadEvent(
                event_id="e4",
                principal_arn=user.arn,
                action="s3:ListBucket",
                resource_arn="arn:aws:s3:::bucket2_drift",
                day_offset=105.0,
                is_drift=True,
            ),
        ]

        # Scenario A: Repaired graph scopes s3:GetObject strictly to bucket1
        # This preserves all T1 events, but breaks e3 in T2!
        scoped_graph = graph.clone()
        scoped_pol = scoped_graph.get_node(pol.id)
        assert scoped_pol is not None
        scoped_doc = PolicyDocument(
            Statement=[
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["s3:GetObject"],
                    Resource=["arn:aws:s3:::bucket1/file.txt"],
                ),
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["s3:ListBucket"],
                    Resource=["*"],
                ),
            ]
        )
        scoped_graph.add_node(
            scoped_pol.model_copy(update={"policy_document": scoped_doc}, deep=True)
        )

        verifier = CompletionSetVerifier(capability_model=capability_model)

        # Replay T1: 100% conforming!
        t1_res = verifier.replay_workload(scoped_graph, t1_events, window_name="T1")
        assert t1_res.is_conforming is True
        assert t1_res.breakage_rate == 0.0
        assert t1_res.allowed_requests == 2
        assert t1_res.denied_requests == 0

        # Replay T2: Exactly 1 event denied (e3) due to resource scoping!
        t2_res = verifier.replay_workload(scoped_graph, t2_events, window_name="T2")
        assert t2_res.is_conforming is False
        assert t2_res.allowed_requests == 1  # s3:ListBucket still allowed
        assert t2_res.denied_requests == 1  # s3:GetObject on bucket2 denied
        assert t2_res.breakage_rate == 0.50
        assert len(t2_res.denied_events) == 1
        assert t2_res.denied_events[0]["event_id"] == "e3"


# ==============================================================================
# 4. Serialization & Diagnostics Tests
# ==============================================================================


class TestVerifierSerializationAndDiagnostics:
    """Tests report formatting, serialization, and edge cases."""

    def test_verification_report_to_dict_and_summary(self) -> None:
        """Verify VerificationReport serialization and summary string."""
        side_res = SideEffectResult(
            has_side_effects=False,
            new_escalation_pairs=[],
            num_pairs_checked=10,
            execution_time_sec=0.01,
            details="Clean",
        )
        t1_res = WorkloadReplayResult(
            window_name="T1",
            total_requests=5,
            allowed_requests=5,
            denied_requests=0,
            breakage_rate=0.0,
            is_conforming=True,
        )
        t2_res = WorkloadReplayResult(
            window_name="T2",
            total_requests=5,
            allowed_requests=4,
            denied_requests=1,
            breakage_rate=0.20,
            is_conforming=False,
        )

        report = VerificationReport(
            source_id="u1",
            target_id="t1",
            level1_risk_observed=0.0,
            level2_risk_completion_set=0.0,
            level3_risk_ground_truth=0.0,
            level1_safe=True,
            level2_safe=True,
            level3_safe=True,
            overall_status=SafetyStatus.SAFE,
            is_misleading=False,
            side_effects=side_res,
            t1_replay=t1_res,
            t2_replay=t2_res,
            execution_time_sec=0.05,
        )

        d = report.to_dict()
        assert d["overall_status"] == "safe"
        assert d["is_misleading"] is False
        assert d["level1_risk_observed"] == 0.0
        assert d["residual_risk_gap"] == 0.0

        summary = report.summary()
        assert "SAFE" in summary
        assert "Level 1" in summary
        assert "Level 2" in summary
        assert "T_1 Replay" in summary
        assert "T_2 Replay" in summary
