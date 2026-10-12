"""Comprehensive unit and integration test suite for the Capability-Aware Greedy Repair Engine.

Tests Phase 3 Week 7 (Step 7.3):
- Transformation models (ScopeResourceTransformation, RemoveActionTransformation) and cost weights.
- Strict AWS Action Capability Model C(a, r, c) enforcement (scoping validity vs. wildcard-only rejection).
- Historical workload conformance checking (100% D^{T_1} preservation guarantee).
- Privilege escalation path elimination across all 4 canonical P0 motifs:
    1. PassRoleLambdaMotif
    2. CreateAccessKeyMotif
    3. AttachPolicyMotif
    4. AssumeRoleChainMotif
- Uncertainty-aware repair across sampled latent completions {H_1, ..., H_M}.
- Unified policy diff synthesis and audit logging.
"""

from __future__ import annotations

import networkx as nx
import pytest

from iam.generator.graph import EdgeRelation, GraphEdge, GraphNode, IAMGraph, NodeType
from iam.generator.motifs import (
    AssumeRoleChainMotif,
    AttachPolicyMotif,
    CreateAccessKeyMotif,
    PassRoleLambdaMotif,
)
from iam.optimizer.repair import (
    CapabilityGreedyRepairEngine,
    PolicyPatchResult,
    RepairConfig,
    WorkloadConformanceChecker,
)
from iam.optimizer.sampler import CandidateEdge, LatentCompletion
from iam.optimizer.transformations import (
    PolicyTransformationType,
    RemoveActionTransformation,
    ScopeResourceTransformation,
    generate_policy_diff,
)
from iam.parser.capability import load_default_capability_model
from iam.parser.evaluator import AuthRequest
from iam.parser.schema import Effect, PolicyDocument, Statement

# ==============================================================================
# 1. Policy Transformation Models & Cost Tests
# ==============================================================================


class TestPolicyTransformations:
    """Test transformation operators, AST updates, and diff generation."""

    def test_scope_resource_transformation_single_action(self) -> None:
        """Verify ScopeResource on a statement with a single action."""
        doc = PolicyDocument(
            Statement=[
                Statement(
                    Sid="AllowPassRole",
                    Effect=Effect.ALLOW,
                    Action=["iam:PassRole"],
                    Resource=["*"],
                )
            ]
        )
        trans = ScopeResourceTransformation(
            policy_id="policy-1",
            statement_index=0,
            action="iam:PassRole",
            target_resources=["arn:aws:iam::123456789012:role/legit-worker"],
        )
        assert trans.cost == 1.0
        assert trans.transformation_type == PolicyTransformationType.SCOPE_RESOURCE

        updated = trans.apply(doc)
        assert len(updated.statements) == 1
        assert updated.statements[0].resource == ["arn:aws:iam::123456789012:role/legit-worker"]
        assert updated.statements[0].action == ["iam:PassRole"]

    def test_scope_resource_transformation_multi_action_splits(self) -> None:
        """Verify ScopeResource splits statement when other actions exist."""
        doc = PolicyDocument(
            Statement=[
                Statement(
                    Sid="MixedStmt",
                    Effect=Effect.ALLOW,
                    Action=["iam:PassRole", "iam:ListRoles"],
                    Resource=["*"],
                )
            ]
        )
        trans = ScopeResourceTransformation(
            policy_id="policy-1",
            statement_index=0,
            action="iam:PassRole",
            target_resources=["arn:aws:iam::123456789012:role/legit-worker"],
        )
        updated = trans.apply(doc)
        assert len(updated.statements) == 2

        # First statement keeps iam:ListRoles on *
        assert updated.statements[0].action == ["iam:ListRoles"]
        assert updated.statements[0].resource == ["*"]

        # Second statement has iam:PassRole on scoped role
        assert updated.statements[1].action == ["iam:PassRole"]
        assert updated.statements[1].resource == ["arn:aws:iam::123456789012:role/legit-worker"]

    def test_scope_resource_out_of_bounds_raises(self) -> None:
        """Verify statement index out of bounds raises IndexError."""
        doc = PolicyDocument(
            Statement=[Statement(Effect=Effect.ALLOW, Action=["s3:GetObject"], Resource=["*"])]
        )
        trans = ScopeResourceTransformation(
            policy_id="policy-1",
            statement_index=5,
            action="s3:GetObject",
            target_resources=["arn:aws:s3:::my-bucket/*"],
        )
        with pytest.raises(IndexError):
            trans.apply(doc)

    def test_remove_action_single_action_removes_statement(self) -> None:
        """Verify RemoveAction removes statement entirely when last action excised."""
        doc = PolicyDocument(
            Statement=[
                Statement(
                    Sid="OnlyPassRole",
                    Effect=Effect.ALLOW,
                    Action=["iam:PassRole"],
                    Resource=["*"],
                )
            ]
        )
        trans = RemoveActionTransformation(
            policy_id="policy-1",
            statement_index=0,
            action="iam:PassRole",
        )
        assert trans.cost == 5.0
        assert trans.transformation_type == PolicyTransformationType.REMOVE_ACTION

        updated = trans.apply(doc)
        assert len(updated.statements) == 0

    def test_remove_action_multi_action_retains_other_actions(self) -> None:
        """Verify RemoveAction excises only targeted action."""
        doc = PolicyDocument(
            Statement=[
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["iam:PassRole", "iam:GetRole"],
                    Resource=["*"],
                )
            ]
        )
        trans = RemoveActionTransformation(
            policy_id="policy-1",
            statement_index=0,
            action="iam:PassRole",
        )
        updated = trans.apply(doc)
        assert len(updated.statements) == 1
        assert updated.statements[0].action == ["iam:GetRole"]

    def test_generate_policy_diff(self) -> None:
        """Verify unified policy diff formatting."""
        doc1 = PolicyDocument(
            Statement=[Statement(Effect=Effect.ALLOW, Action=["iam:PassRole"], Resource=["*"])]
        )
        doc2 = PolicyDocument(
            Statement=[
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["iam:PassRole"],
                    Resource=["arn:aws:iam::123456789012:role/worker"],
                )
            ]
        )
        diff_str = generate_policy_diff(doc1, doc2, policy_name="TestPolicy")
        assert "--- a/TestPolicy.json" in diff_str
        assert "+++ b/TestPolicy.json" in diff_str
        assert '-      "Resource": [' in diff_str or '-        "*"' in diff_str
        assert '+        "arn:aws:iam::123456789012:role/worker"' in diff_str


# ==============================================================================
# 2. AWS Capability Model Integration & Validation Tests
# ==============================================================================


class TestCapabilityEnforcement:
    """Test compliance with AWS Action Capability Model C(a, r, c)."""

    def test_valid_scoping_on_resource_level_action(self) -> None:
        """iam:PassRole supports resource scoping to role ARNs."""
        cap_model = load_default_capability_model()
        trans = ScopeResourceTransformation(
            policy_id="pol-1",
            statement_index=0,
            action="iam:PassRole",
            target_resources=["arn:aws:iam::123456789012:role/my-valid-role"],
        )
        val = trans.validate_capability(cap_model)
        assert val.is_valid is True

    def test_rejection_of_scoping_on_wildcard_only_action(self) -> None:
        """iam:ListRoles is wildcard-only; scoping to role ARN must be rejected."""
        cap_model = load_default_capability_model()
        trans = ScopeResourceTransformation(
            policy_id="pol-1",
            statement_index=0,
            action="iam:ListRoles",
            target_resources=["arn:aws:iam::123456789012:role/some-role"],
        )
        val = trans.validate_capability(cap_model)
        assert val.is_valid is False
        assert "wildcard-only" in val.reason

    def test_rejection_of_mismatched_resource_type(self) -> None:
        """iam:PassRole expects a role ARN, not an S3 bucket ARN."""
        cap_model = load_default_capability_model()
        trans = ScopeResourceTransformation(
            policy_id="pol-1",
            statement_index=0,
            action="iam:PassRole",
            target_resources=["arn:aws:s3:::some-bucket"],
        )
        val = trans.validate_capability(cap_model)
        assert val.is_valid is False
        assert "does not match supported resource types" in val.reason

    def test_empty_target_resources_rejected(self) -> None:
        """Scoping to an empty resource list is invalid."""
        cap_model = load_default_capability_model()
        trans = ScopeResourceTransformation(
            policy_id="pol-1",
            statement_index=0,
            action="iam:PassRole",
            target_resources=[],
        )
        val = trans.validate_capability(cap_model)
        assert val.is_valid is False

    def test_action_removal_always_valid(self) -> None:
        """Revoking an action is always capability compliant."""
        cap_model = load_default_capability_model()
        trans = RemoveActionTransformation(
            policy_id="pol-1",
            statement_index=0,
            action="iam:ListRoles",
        )
        val = trans.validate_capability(cap_model)
        assert val.is_valid is True


# ==============================================================================
# 3. Workload Conformance Checker Tests
# ==============================================================================


class TestWorkloadConformanceChecker:
    """Test 100% preservation guarantee on historical workload operations D^{T_1}."""

    def test_conformance_passes_for_authorized_workload(self) -> None:
        """Workload requests matching policies evaluate to ALLOW."""
        graph = IAMGraph("test_conf_graph")
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
                        Action=["iam:PassRole"],
                        Resource=["arn:aws:iam::123456789012:role/worker-1"],
                    )
                ]
            ),
        )
        graph.add_node(user)
        graph.add_node(pol)
        graph.add_edge(
            GraphEdge(source=user.id, target=pol.id, relation=EdgeRelation.ATTACHED_WITH)
        )

        checker = WorkloadConformanceChecker()
        workload = [
            AuthRequest(
                principal_arn="arn:aws:iam::123456789012:user/dev",
                action="iam:PassRole",
                resource_arn="arn:aws:iam::123456789012:role/worker-1",
            )
        ]
        conforms, denied = checker.check_conformance(graph, workload)
        assert conforms is True
        assert len(denied) == 0

    def test_conformance_fails_when_action_denied(self) -> None:
        """If policy does not authorize workload request, conformance fails."""
        graph = IAMGraph("test_conf_graph")
        user = GraphNode(
            id="user:dev",
            node_type=NodeType.USER,
            arn="arn:aws:iam::123456789012:user/dev",
            name="dev",
        )
        graph.add_node(user)

        checker = WorkloadConformanceChecker()
        workload = [
            AuthRequest(
                principal_arn="arn:aws:iam::123456789012:user/dev",
                action="s3:GetObject",
                resource_arn="arn:aws:s3:::secret-bucket/file.txt",
            )
        ]
        conforms, denied = checker.check_conformance(graph, workload)
        assert conforms is False
        assert len(denied) == 1
        assert denied[0].action == "s3:GetObject"


# ==============================================================================
# 4. Canonical P0 Motif Greedy Repair Tests
# ==============================================================================


class TestGreedyRepairP0Motifs:
    """Test full repair loop across all 4 canonical P0 privilege escalation motifs."""

    def test_repair_passrole_lambda_motif(self) -> None:
        """P0 Motif 1: iam:PassRole + lambda:CreateFunction/InvokeFunction.

        Vulnerable: User has iam:PassRole on * (or admin role).
        Legitimate Workload: Developer passes role:legit-worker.
        Attack: Attacker passes high-privilege lambda-admin-exec role to reach vault KMS key.
        Expected Repair: Scope iam:PassRole to role:legit-worker (Cost: 1.0).
        """
        graph = IAMGraph("passrole_test_graph")
        motif = PassRoleLambdaMotif()
        instance = motif.inject(graph, suffix="test")

        source_node = graph.get_node(instance.source_id)
        assert source_node is not None
        target_node = graph.get_node(instance.target_id)
        assert target_node is not None

        # Model realistic un-scoped wildcard policy allowing iam:PassRole on *
        source_pol = graph.get_node("policy:lambda-creator-test")
        assert source_pol is not None
        vulnerable_pass_doc = PolicyDocument(
            Statement=[
                Statement(Effect=Effect.ALLOW, Action=["iam:PassRole"], Resource=["*"]),
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["lambda:CreateFunction", "lambda:InvokeFunction"],
                    Resource=["arn:aws:lambda:us-east-1:123456789012:function:escalator-test"],
                ),
            ]
        )
        graph.add_node(
            source_pol.model_copy(update={"policy_document": vulnerable_pass_doc}, deep=True)
        )

        # Legitimate worker role evidenced in workload logs D^{T_1}
        legit_worker_arn = "arn:aws:iam::123456789012:role/legit-worker-test"
        workload_logs = [
            AuthRequest(
                principal_arn=source_node.arn,
                action="iam:PassRole",
                resource_arn=legit_worker_arn,
            ),
            AuthRequest(
                principal_arn=source_node.arn,
                action="lambda:CreateFunction",
                resource_arn="arn:aws:lambda:us-east-1:123456789012:function:escalator-test",
            ),
        ]

        # Verify ground truth attack path exists prior to repair
        verif = motif.verify_ground_truth(graph, instance)
        assert verif.path_exists is True

        engine = CapabilityGreedyRepairEngine()
        result = engine.repair(
            graph=graph,
            source_id=instance.source_id,
            target_id=instance.target_id,
            workload_logs=workload_logs,
        )

        assert result.is_successful is True
        assert result.initial_attack_risk == 1.0
        assert result.final_attack_risk == 0.0
        assert result.risk_reduction == 1.0
        assert result.total_repair_cost == 1.0  # Chose scoping (cost 1.0) over removal (cost 5.0)
        assert result.log_conformance_verified is True
        assert len(result.applied_transformations) == 1

        # Check transformation type
        trans = result.applied_transformations[0]
        assert isinstance(trans, ScopeResourceTransformation)
        assert trans.action == "iam:PassRole"
        assert legit_worker_arn in trans.target_resources

        # Verify bridge edge was severed in repaired graph
        repaired_nx = result.repaired_graph.nx_graph
        assert not nx.has_path(repaired_nx, instance.source_id, instance.target_id)

        # Verify policy diff generated
        assert len(result.policy_diffs) >= 1
        pol_diff = result.get_diff(trans.policy_id)
        assert legit_worker_arn in pol_diff

    def test_repair_create_access_key_motif(self) -> None:
        """P0 Motif 2: iam:CreateAccessKey on Admin User.

        Vulnerable: Principal has iam:CreateAccessKey.
        Attack: Attacker calls CreateAccessKey on admin user.
        Expected Repair: Revoke or scope iam:CreateAccessKey.
        """
        graph = IAMGraph("key_test_graph")
        motif = CreateAccessKeyMotif()
        instance = motif.inject(graph, suffix="test")

        source_node = graph.get_node(instance.source_id)
        assert source_node is not None

        # Model un-scoped policy allowing iam:CreateAccessKey on *
        source_pol = graph.get_node("policy:key-manager-test")
        assert source_pol is not None
        vulnerable_key_doc = PolicyDocument(
            Statement=[
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["iam:CreateAccessKey"],
                    Resource=["*"],
                )
            ]
        )
        graph.add_node(
            source_pol.model_copy(update={"policy_document": vulnerable_key_doc}, deep=True)
        )

        # Legitimate workload: intern only creates access keys for self
        workload_logs = [
            AuthRequest(
                principal_arn=source_node.arn,
                action="iam:CreateAccessKey",
                resource_arn=source_node.arn,
            )
        ]

        engine = CapabilityGreedyRepairEngine()
        result = engine.repair(
            graph=graph,
            source_id=instance.source_id,
            target_id=instance.target_id,
            workload_logs=workload_logs,
        )

        assert result.is_successful is True
        assert result.final_attack_risk == 0.0
        assert result.log_conformance_verified is True

        # Verify path severed
        assert not nx.has_path(
            result.repaired_graph.nx_graph, instance.source_id, instance.target_id
        )

    def test_repair_attach_policy_motif(self) -> None:
        """P0 Motif 3: iam:AttachUserPolicy with AdministratorAccess.

        Vulnerable: Principal has iam:AttachUserPolicy.
        Legitimate Workload: No workload calls to AttachUserPolicy.
        Expected Repair: RemoveActionTransformation (Cost: 5.0).
        """
        graph = IAMGraph("attach_test_graph")
        motif = AttachPolicyMotif()
        instance = motif.inject(graph, suffix="test")

        source_node = graph.get_node(instance.source_id)
        assert source_node is not None

        # Empty workload logs (unexercised privilege)
        workload_logs: list[AuthRequest] = []

        engine = CapabilityGreedyRepairEngine()
        result = engine.repair(
            graph=graph,
            source_id=instance.source_id,
            target_id=instance.target_id,
            workload_logs=workload_logs,
        )

        assert result.is_successful is True
        assert result.final_attack_risk == 0.0
        assert result.total_repair_cost == 5.0
        assert isinstance(result.applied_transformations[0], RemoveActionTransformation)

        # Verify path severed
        assert not nx.has_path(
            result.repaired_graph.nx_graph, instance.source_id, instance.target_id
        )

    def test_repair_assume_role_chain_motif(self) -> None:
        """P0 Motif 4: sts:AssumeRole Multi-Hop Chaining.

        Vulnerable: Intermediate role 1 assumes intermediate role 2 to reach target admin.
        Expected Repair: Sever chaining edge via policy or trust policy repair.
        """
        graph = IAMGraph("chain_test_graph")
        motif = AssumeRoleChainMotif()
        instance = motif.inject(graph, suffix="test", num_intermediate_roles=2)

        engine = CapabilityGreedyRepairEngine()
        result = engine.repair(
            graph=graph,
            source_id=instance.source_id,
            target_id=instance.target_id,
            workload_logs=[],
        )

        assert result.is_successful is True
        assert result.final_attack_risk == 0.0
        assert not nx.has_path(
            result.repaired_graph.nx_graph, instance.source_id, instance.target_id
        )


# ==============================================================================
# 5. Uncertainty-Aware Repair over Sampled Latent Completions
# ==============================================================================


class TestUncertaintyAwareGreedyRepair:
    """Test repair evaluated over completion distributions {H_1, ..., H_M}."""

    def test_repair_across_stochastic_latent_completions(self) -> None:
        """Verify greedy repair severs paths across multiple sampled completions."""
        graph = IAMGraph("uncertainty_graph")

        # Entrypoint user
        u1 = GraphNode(
            id="user:dev",
            node_type=NodeType.USER,
            arn="arn:aws:iam::123456789012:user/dev",
            name="dev",
        )
        # Policy attached to dev allowing iam:PassRole
        pol1 = GraphNode(
            id="policy:dev-pass",
            node_type=NodeType.POLICY,
            arn="arn:aws:iam::123456789012:policy/dev-pass",
            name="dev-pass",
            policy_document=PolicyDocument(
                Statement=[
                    Statement(
                        Effect=Effect.ALLOW,
                        Action=["iam:PassRole"],
                        Resource=["*"],
                    )
                ]
            ),
        )
        # Role 1
        r1 = GraphNode(
            id="role:worker",
            node_type=NodeType.ROLE,
            arn="arn:aws:iam::123456789012:role/worker",
            name="worker",
        )
        # High value target
        t = GraphNode(
            id="resource:vault",
            node_type=NodeType.RESOURCE,
            arn="arn:aws:kms:us-east-1:123456789012:key/vault",
            name="vault",
            is_high_value=True,
        )

        graph.add_node(u1)
        graph.add_node(pol1)
        graph.add_node(r1)
        graph.add_node(t)

        graph.add_edge(GraphEdge(source=u1.id, target=pol1.id, relation=EdgeRelation.ATTACHED_WITH))
        graph.add_edge(
            GraphEdge(
                source=u1.id,
                target=r1.id,
                relation=EdgeRelation.PASSES_TO,
                actions=["iam:PassRole"],
            )
        )

        # Latent completions: H_i connects worker -> vault with varying topologies
        latent_edge = CandidateEdge(
            source=r1.id,
            target=t.id,
            relation=EdgeRelation.ACTS_ON,
            probability=0.8,
            actions=("kms:Decrypt",),
        )

        # 5 completions where latent edge is present in 4 of them
        completions = [
            LatentCompletion(
                sample_id=i,
                sampled_edges=[latent_edge] if i < 4 else [],
                graph=graph,
            )
            for i in range(5)
        ]

        # Legitimate workload
        workload = [
            AuthRequest(
                principal_arn=u1.arn,
                action="iam:PassRole",
                resource_arn="arn:aws:iam::123456789012:role/legit-only",
            )
        ]

        engine = CapabilityGreedyRepairEngine()
        result = engine.repair(
            graph=graph,
            source_id=u1.id,
            target_id=t.id,
            workload_logs=workload,
            completions=completions,
        )

        assert result.is_successful is True
        assert result.initial_attack_risk == 0.8  # 4 / 5 completions had path
        assert result.final_attack_risk == 0.0  # Cut u1 -> r1 severs all 4 completions!
        assert result.total_repair_cost == 1.0  # Scoped PassRole


# ==============================================================================
# 6. Edge Cases & Hyperparameter Validation Tests
# ==============================================================================


class TestRepairEdgeCases:
    """Test boundary conditions, hyperparameter validation, and diagnostics."""

    def test_invalid_repair_config_raises(self) -> None:
        """Verify invalid weights and thresholds raise ValueError."""
        with pytest.raises(ValueError):
            RepairConfig(cost_scope_resource=-1.0)
        with pytest.raises(ValueError):
            RepairConfig(cost_remove_action=0.0)
        with pytest.raises(ValueError):
            RepairConfig(max_iterations=0)
        with pytest.raises(ValueError):
            RepairConfig(residual_risk_threshold_delta=1.5)

    def test_already_zero_risk_returns_identity_patch(self) -> None:
        """When source cannot reach target initially, repair returns identity patch."""
        graph = IAMGraph("empty_path_graph")
        u = GraphNode(
            id="user:u1",
            node_type=NodeType.USER,
            arn="arn:aws:iam::123456789012:user/u1",
            name="u1",
        )
        t = GraphNode(
            id="resource:t1",
            node_type=NodeType.RESOURCE,
            arn="arn:aws:s3:::t1",
            name="t1",
        )
        graph.add_node(u)
        graph.add_node(t)

        engine = CapabilityGreedyRepairEngine()
        result = engine.repair(graph=graph, source_id=u.id, target_id=t.id)

        assert result.is_successful is True
        assert result.initial_attack_risk == 0.0
        assert result.final_attack_risk == 0.0
        assert result.total_repair_cost == 0.0
        assert result.num_applied_transformations == 0

    def test_strict_baseline_conformance_failure_raises(self) -> None:
        """If baseline graph does not authorize historical workload, strict mode raises."""
        graph = IAMGraph("unauthorized_baseline_graph")
        u = GraphNode(
            id="user:u1",
            node_type=NodeType.USER,
            arn="arn:aws:iam::123456789012:user/u1",
            name="u1",
        )
        graph.add_node(u)

        workload = [
            AuthRequest(
                principal_arn=u.arn,
                action="s3:GetObject",
                resource_arn="arn:aws:s3:::bucket/key",
            )
        ]

        engine = CapabilityGreedyRepairEngine(config=RepairConfig(strict_baseline_conformance=True))
        with pytest.raises(ValueError, match="not fully authorized on baseline graph"):
            engine.repair(graph=graph, source_id=u.id, target_id=u.id, workload_logs=workload)

    def test_result_to_dict_and_summary(self) -> None:
        """Verify PolicyPatchResult serialization and summary."""
        graph = IAMGraph("dummy")
        result = PolicyPatchResult(
            repaired_graph=graph,
            applied_transformations=[],
            modified_policy_ids=["pol-1"],
            modified_policy_documents={},
            policy_diffs={"pol-1": "--- diff ---"},
            initial_attack_risk=1.0,
            final_attack_risk=0.0,
            total_repair_cost=1.0,
            log_conformance_verified=True,
            is_successful=True,
            execution_time_sec=0.042,
            num_iterations=1,
        )
        d = result.to_dict()
        assert d["is_successful"] is True
        assert d["risk_reduction"] == 1.0
        assert d["total_repair_cost"] == 1.0
        assert "SUCCESS" in result.summary()
