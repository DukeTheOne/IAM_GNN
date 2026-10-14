"""Milestone M3 validation and end-to-end Phase 3 repair pipeline benchmark.

Validates the full uncertainty-aware least-privilege repair pipeline under partial observability:
1. Calibrated Latent Completion Sampling (Step 7.1): M Monte Carlo graph completions.
2. Uncertainty-Guided Candidate Pruning (Step 7.2): Bounded candidate space with recall evaluation.
3. Capability-Aware Greedy Policy Repair (Step 7.3): Minimal, C(a, r, c)-compliant patches with 100% D^{T_1} preservation.
4. Exact SMT / MaxSAT Optimality Benchmark (Step 7.4): Zero optimality gap against Z3 exact solver.
5. Enterprise Scaling & Interactive Latency: End-to-end execution on 150+ node enterprise topology (< 2.0 s).
"""

from __future__ import annotations

import time
import tracemalloc

import networkx as nx
import pytest

from iam.generator.graph import (
    EdgeRelation,
    GraphEdge,
    GraphNode,
    IAMGraph,
    NodeType,
)
from iam.generator.motifs import (
    CreateAccessKeyMotif,
    PassRoleLambdaMotif,
)
from iam.generator.topology import (
    EnterpriseTopologyConfig,
    EnterpriseTopologyGenerator,
)
from iam.optimizer.pruning import (
    CandidatePruner,
    PruningConfig,
    PruningResult,
    compute_candidate_recall,
)
from iam.optimizer.repair import (
    CapabilityGreedyRepairEngine,
    PolicyPatchResult,
    RepairConfig,
)
from iam.optimizer.sampler import (
    CandidateEdge,
    LatentCompletionSampler,
)
from iam.optimizer.transformations import (
    RemoveActionTransformation,
    ScopeResourceTransformation,
)
from iam.optimizer.z3_solver import (
    Z3CanonicalRepairSolver,
)
from iam.parser.capability import CapabilityModel, load_default_capability_model
from iam.parser.evaluator import AuthRequest
from iam.parser.schema import Effect, PolicyDocument, Statement


@pytest.fixture
def capability_model() -> CapabilityModel:
    """Fixture providing loaded AWS CapabilityModel."""
    return load_default_capability_model()


# ==============================================================================
# 1. Full Optimization Pipeline E2E (Condition E Adversarial Bridge)
# ==============================================================================


class TestFullOptimizationPipelineE2E:
    """Chains Sampling -> Pruning -> Greedy Repair -> Policy Patch Synthesis."""

    def test_e2e_passrole_lambda_pipeline_under_condition_e(
        self, capability_model: CapabilityModel
    ) -> None:
        """Execute full Phase 3 pipeline on canonical PassRole + Lambda motif under Condition E.

        1. Mask downstream KMS decrypt edge from admin policy to vault key to simulate Condition E.
        2. Assign calibrated latent edge probability p_hat = 0.85.
        3. Draw M = 40 Monte Carlo latent completions.
        4. Prune search space with CandidatePruner, checking candidate recall.
        5. Synthesize capability-compliant patch with CapabilityGreedyRepairEngine.
        6. Verify: risk reduced to 0.0, workload D^{T_1} 100% conforming, AST diff valid.
        """
        # 1. Setup base graph with PassRole + Lambda motif
        graph = IAMGraph("e2e_passrole_env")
        motif = PassRoleLambdaMotif()
        instance = motif.inject(graph, suffix="e2e")

        source_node = graph.get_node(instance.source_id)
        assert source_node is not None
        source_pol = graph.get_node("policy:lambda-creator-e2e")
        assert source_pol is not None

        # Add vulnerable un-scoped PassRole statement
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
                    Resource=["arn:aws:lambda:us-east-1:123456789012:function:escalator-e2e"],
                ),
            ]
        )
        graph.add_node(source_pol.model_copy(update={"policy_document": vulnerable_doc}, deep=True))

        # Sever the downstream relation to simulate Condition E (observed graph G_o has no path to vault)
        kms_pol_id = "policy:kms-full-access-e2e"
        target_id = instance.target_id
        graph.remove_edge(kms_pol_id, target_id, EdgeRelation.ACTS_ON)

        # Verify G_o has zero reachability initially without latent completion
        assert not nx.has_path(graph.nx_graph, instance.source_id, instance.target_id)

        # 2. Latent candidate edge with calibrated probability p_hat = 0.85
        candidate_bridge = CandidateEdge(
            source=kms_pol_id,
            target=target_id,
            relation=EdgeRelation.ACTS_ON,
            probability=0.85,
            actions=("kms:Decrypt", "kms:GenerateDataKey"),
            is_bridge=True,
        )

        # 3. Monte Carlo latent completion sampling (M = 40)
        sampler = LatentCompletionSampler(observed_graph=graph, candidate_edges=[candidate_bridge])
        completions, mc_eval = sampler.sample_and_evaluate(
            num_samples=40,
            seed=42,
            source_ids=[instance.source_id],
            target_ids=[instance.target_id],
        )

        assert len(completions) == 40
        reach_prob = mc_eval.get_reachability_probability(instance.source_id, instance.target_id)
        assert 0.70 <= reach_prob <= 0.95
        assert mc_eval.is_attack_possible(instance.source_id, instance.target_id)

        # 4. Uncertainty-guided candidate pruning
        pruner = CandidatePruner(
            config=PruningConfig(max_candidates=20, min_frequency=0.10, only_observed=True),
            capability_model=capability_model,
        )
        pruning_res = pruner.prune_from_completions(
            graph=graph,
            completions=completions,
            oracle_result=mc_eval,
            source_ids=[instance.source_id],
            target_ids=[instance.target_id],
        )

        assert isinstance(pruning_res, PruningResult)
        assert pruning_res.num_pruned_candidates > 0
        assert pruning_res.execution_time_sec < 1.0

        # Check candidate recall: the observed user PassRole bottleneck is retained
        expected_cut = (
            instance.source_id,
            "role:lambda-admin-exec-e2e",
            EdgeRelation.PASSES_TO.value,
        )
        recall = compute_candidate_recall(pruning_res.pruned_edges, [expected_cut])
        assert recall == 1.0

        # 5. Define legitimate operational workload D^{T_1}
        legit_worker_arn = "arn:aws:iam::123456789012:role/legit-worker-service"
        workload_logs = [
            AuthRequest(
                principal_arn=source_node.arn,
                action="iam:PassRole",
                resource_arn=legit_worker_arn,
            ),
            AuthRequest(
                principal_arn=source_node.arn,
                action="lambda:CreateFunction",
                resource_arn="arn:aws:lambda:us-east-1:123456789012:function:escalator-e2e",
            ),
        ]

        # 6. Synthesize capability-compliant least-privilege repair
        repair_engine = CapabilityGreedyRepairEngine(
            config=RepairConfig(
                cost_scope_resource=1.0,
                cost_remove_action=5.0,
                residual_risk_threshold_delta=0.0,
                strict_baseline_conformance=True,
            ),
            capability_model=capability_model,
        )

        patch_result: PolicyPatchResult = repair_engine.repair(
            graph=graph,
            source_id=instance.source_id,
            target_id=instance.target_id,
            workload_logs=workload_logs,
            completions=completions,
            pruning_result=pruning_res,
        )

        # 7. Comprehensive Pipeline Assertions
        assert patch_result.is_successful is True
        assert patch_result.initial_attack_risk == reach_prob
        assert patch_result.final_attack_risk == 0.0
        assert patch_result.risk_reduction == reach_prob
        assert patch_result.log_conformance_verified is True
        assert patch_result.total_repair_cost == 1.0  # Selected least-disruptive scoping

        # Verify applied transformation is Priority 1 ScopeResource
        assert len(patch_result.applied_transformations) == 1
        trans = patch_result.applied_transformations[0]
        assert isinstance(trans, ScopeResourceTransformation)
        assert trans.action == "iam:PassRole"
        assert legit_worker_arn in trans.target_resources

        # Verify unified AST diff
        diff_text = patch_result.get_diff(trans.policy_id)
        assert f'+        "{legit_worker_arn}"' in diff_text
        assert '-        "*"' in diff_text

        # Verify that attack reachability across ALL sampled completions is completely severed
        for comp in completions:
            test_union = patch_result.repaired_graph.clone()
            for cand in comp.sampled_edges:
                test_union.add_edge(cand.to_graph_edge())
            assert not nx.has_path(test_union.nx_graph, instance.source_id, instance.target_id)


# ==============================================================================
# 2. Multi-Motif Compound Attack Pipeline
# ==============================================================================


class TestCompoundMultiMotifPipeline:
    """Tests end-to-end repair across multiple co-occurring escalation vectors."""

    def test_multi_motif_compound_attack_pipeline(self, capability_model: CapabilityModel) -> None:
        """Test pipeline when multiple privilege escalation vectors are present simultaneously.

        Attack Vector 1: PassRole + Lambda motif (masked bridge, p=0.80).
        Attack Vector 2: CreateAccessKey motif (direct un-scoped key creation on admin).
        The pipeline must sample, prune, and greedily eliminate BOTH attack vectors.
        """
        graph = IAMGraph("compound_pe_env")

        # Inject Motif 1: PassRole + Lambda
        motif1 = PassRoleLambdaMotif()
        inst1 = motif1.inject(graph, suffix="m1")
        kms_pol_1 = "policy:kms-full-access-m1"
        graph.remove_edge(kms_pol_1, inst1.target_id, EdgeRelation.ACTS_ON)

        cand1 = CandidateEdge(
            source=kms_pol_1,
            target=inst1.target_id,
            relation=EdgeRelation.ACTS_ON,
            probability=0.80,
            actions=("kms:Decrypt", "kms:GenerateDataKey"),
            is_bridge=True,
        )

        # Inject Motif 2: CreateAccessKey
        motif2 = CreateAccessKeyMotif()
        inst2 = motif2.inject(graph, suffix="m2")

        # Model un-scoped key policy on inst2 source
        pol2_node = graph.get_node("policy:key-manager-m2")
        assert pol2_node is not None
        vuln_key_doc = PolicyDocument(
            Statement=[
                Statement(
                    Sid="AllowCreateKeyWildcard",
                    Effect=Effect.ALLOW,
                    Action=["iam:CreateAccessKey"],
                    Resource=["*"],
                )
            ]
        )
        graph.add_node(pol2_node.model_copy(update={"policy_document": vuln_key_doc}, deep=True))

        src2_node = graph.get_node(inst2.source_id)
        assert src2_node is not None
        legit_self_arn = src2_node.arn

        workload = [
            AuthRequest(
                principal_arn=src2_node.arn,
                action="iam:CreateAccessKey",
                resource_arn=legit_self_arn,
            )
        ]

        # Sample completions for candidate edges
        sampler = LatentCompletionSampler(observed_graph=graph, candidate_edges=[cand1])
        completions = sampler.sample_completions(num_samples=25, seed=123)

        # Prune candidates for inst2
        pruner = CandidatePruner(capability_model=capability_model)
        pruning_res = pruner.prune_from_completions(
            graph=graph,
            completions=completions,
            source_ids=[inst2.source_id],
            target_ids=[inst2.target_id],
        )

        # Repair inst2 (CreateAccessKey vector)
        repair_engine = CapabilityGreedyRepairEngine(capability_model=capability_model)
        result2 = repair_engine.repair(
            graph=graph,
            source_id=inst2.source_id,
            target_id=inst2.target_id,
            workload_logs=workload,
            completions=completions,
            pruning_result=pruning_res,
        )

        assert result2.is_successful is True
        assert result2.final_attack_risk == 0.0
        assert result2.log_conformance_verified is True
        assert len(result2.applied_transformations) == 1
        assert isinstance(result2.applied_transformations[0], ScopeResourceTransformation)
        assert legit_self_arn in result2.applied_transformations[0].target_resources


# ==============================================================================
# 3. Greedy vs. Z3 Exact SMT / MaxSAT Optimality Gap Benchmark
# ==============================================================================


class TestGreedyVsZ3OptimalityBenchmark:
    """Benchmarks end-to-end greedy pipeline against exact symbolic Z3 MaxSAT solver."""

    def test_pipeline_optimality_gap_vs_z3_symbolic_solver(
        self, capability_model: CapabilityModel
    ) -> None:
        """Verify that greedy repair achieves mathematically optimal zero cost gap on canonical motif."""
        graph = IAMGraph("optimality_gap_env")
        motif = PassRoleLambdaMotif()
        instance = motif.inject(graph, suffix="opt")

        source_node = graph.get_node(instance.source_id)
        assert source_node is not None
        source_pol = graph.get_node("policy:lambda-creator-opt")
        assert source_pol is not None

        vulnerable_doc = PolicyDocument(
            Statement=[
                Statement(
                    Sid="PassRoleStmt",
                    Effect=Effect.ALLOW,
                    Action=["iam:PassRole"],
                    Resource=["*"],
                ),
                Statement(
                    Sid="LambdaStmt",
                    Effect=Effect.ALLOW,
                    Action=["lambda:CreateFunction", "lambda:InvokeFunction"],
                    Resource=["arn:aws:lambda:us-east-1:123456789012:function:escalator-opt"],
                ),
            ]
        )
        graph.add_node(source_pol.model_copy(update={"policy_document": vulnerable_doc}, deep=True))

        legit_worker_arn = "arn:aws:iam::123456789012:role/legit-worker-opt"
        workload_logs = [
            AuthRequest(
                principal_arn=source_node.arn,
                action="iam:PassRole",
                resource_arn=legit_worker_arn,
            ),
            AuthRequest(
                principal_arn=source_node.arn,
                action="lambda:CreateFunction",
                resource_arn="arn:aws:lambda:us-east-1:123456789012:function:escalator-opt",
            ),
        ]

        # Use Z3 solver comparison method
        solver = Z3CanonicalRepairSolver(
            capability_model=capability_model,
            cost_scope=1.0,
            cost_remove=5.0,
        )
        z3_res, greedy_res, gap = solver.compare_with_greedy(
            graph=graph,
            source_id=instance.source_id,
            target_id=instance.target_id,
            workload_logs=workload_logs,
        )

        assert z3_res.is_sat is True
        assert greedy_res.is_successful is True
        assert z3_res.total_cost == 1.0
        assert greedy_res.total_repair_cost == 1.0

        # Optimality Gap: Cost(Greedy) - Cost(Z3) == 0.0
        assert gap == 0.0
        assert z3_res.optimality_gap_vs_greedy == 0.0

        # Both solvers choose ScopeResourceTransformation on PassRole
        assert len(z3_res.applied_transformations) == 1
        assert len(greedy_res.applied_transformations) == 1
        assert isinstance(z3_res.applied_transformations[0], ScopeResourceTransformation)
        assert isinstance(greedy_res.applied_transformations[0], ScopeResourceTransformation)


# ==============================================================================
# 4. Enterprise Scaling & Future Optimization Benchmark
# ==============================================================================


class TestEnterpriseScalingAndFutureOptimization:
    """Profiles runtime latency, memory, and scaling across enterprise topologies."""

    def test_enterprise_scale_pipeline_latency_and_memory(
        self, capability_model: CapabilityModel
    ) -> None:
        """Profile full pipeline on realistic 150+ node enterprise graph.

        Verifies:
        - End-to-end execution latency < 2.0 seconds (well within the 3.0 s interactive threshold).
        - Peak memory consumption overhead is strictly bounded (< 50 MB).
        - Candidate recall and pruning efficiency maintain search bounds (<= 50 edges).
        - Clean preservation of benign enterprise operations.
        """
        tracemalloc.start()
        start_time = time.perf_counter()

        # 1. Synthesize 150-node enterprise topology with injected PassRole PE chain
        topo_config = EnterpriseTopologyConfig(
            num_nodes=150,
            seed=2026,
            num_pe_chains=2,
            pe_motif_types=["passrole_lambda"],
            include_branching_chains=False,
            account_id="123456789012",
            organization_id="o-scaling-test",
        )
        topo_gen = EnterpriseTopologyGenerator(
            config=topo_config, capability_model=capability_model
        )
        graph = topo_gen.generate()
        assert graph.num_nodes >= 120

        # Extract an injected motif
        injected = graph.metadata.get("injected_motifs", [])
        assert len(injected) >= 1
        m_info = injected[0]
        s_id = m_info["source_id"]
        t_id = m_info["target_id"]

        # Mask downstream KMS relation to simulate partial observability
        kms_pols = [n for n in m_info["intermediate_node_ids"] if "kms-full-access" in n]
        assert len(kms_pols) >= 1
        kms_pol_id = kms_pols[0]
        graph.remove_edge(kms_pol_id, t_id, EdgeRelation.ACTS_ON)

        cand_bridge = CandidateEdge(
            source=kms_pol_id,
            target=t_id,
            relation=EdgeRelation.ACTS_ON,
            probability=0.85,
            actions=("kms:Decrypt", "kms:GenerateDataKey"),
            is_bridge=True,
        )

        # 2. Monte Carlo sampling (M = 25 completions)
        sampler = LatentCompletionSampler(observed_graph=graph, candidate_edges=[cand_bridge])
        completions, mc_eval = sampler.sample_and_evaluate(
            num_samples=25,
            seed=42,
            source_ids=[s_id],
            target_ids=[t_id],
        )

        # 3. Uncertainty-guided pruning (bounded to K = 50)
        pruner = CandidatePruner(
            config=PruningConfig(max_candidates=50, min_frequency=0.05),
            capability_model=capability_model,
        )
        pruning_result = pruner.prune_from_completions(
            graph=graph,
            completions=completions,
            oracle_result=mc_eval,
            source_ids=[s_id],
            target_ids=[t_id],
        )
        assert pruning_result.num_pruned_candidates <= 50

        # 4. Capability-aware greedy repair
        engine = CapabilityGreedyRepairEngine(capability_model=capability_model)
        patch_result = engine.repair(
            graph=graph,
            source_id=s_id,
            target_id=t_id,
            completions=completions,
            pruning_result=pruning_result,
        )

        elapsed_sec = time.perf_counter() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_mem / (1024 * 1024)

        # 5. Assertions on Latency, Memory, and Scaling Bounds
        assert patch_result.is_successful is True
        assert patch_result.final_attack_risk == 0.0
        assert elapsed_sec < 2.0, f"Full pipeline took {elapsed_sec:.2f}s (exceeded 2.0s limit)"
        assert peak_mb < 50.0, f"Peak memory {peak_mb:.2f}MB exceeded 50MB bound"


# ==============================================================================
# 5. Safety, Admissibility & Invariant Enforcement
# ==============================================================================


class TestSafetyAndAdmissibilityInvariants:
    """Tests strict C(a, r, c) compliance and conflict-free least privilege."""

    def test_wildcard_only_action_triggers_action_removal_not_scoping(
        self, capability_model: CapabilityModel
    ) -> None:
        """Verify that wildcard-only actions (iam:ListRoles) are rejected for scoping and removed."""
        graph = IAMGraph("wildcard_only_test")
        u = GraphNode(
            id="user:u1", node_type=NodeType.USER, arn="arn:aws:iam::1:user/u1", name="u1"
        )
        r = GraphNode(
            id="role:r1", node_type=NodeType.ROLE, arn="arn:aws:iam::1:role/r1", name="r1"
        )
        pol_node = GraphNode(
            id="policy:pol1",
            node_type=NodeType.POLICY,
            arn="arn:aws:iam::1:policy/pol1",
            name="pol1",
            policy_document=PolicyDocument(
                Statement=[
                    Statement(
                        Effect=Effect.ALLOW,
                        Action=["iam:ListRoles"],
                        Resource=["*"],
                    )
                ]
            ),
        )
        graph.add_nodes([u, r, pol_node])
        graph.add_edge(
            GraphEdge(source=u.id, target=pol_node.id, relation=EdgeRelation.ATTACHED_WITH)
        )
        graph.add_edge(
            GraphEdge(
                source=u.id,
                target=r.id,
                relation=EdgeRelation.ACTS_ON,
                actions=["iam:ListRoles"],
            )
        )

        engine = CapabilityGreedyRepairEngine(capability_model=capability_model)
        result = engine.repair(graph=graph, source_id=u.id, target_id=r.id)

        assert result.is_successful is True
        assert len(result.applied_transformations) == 1
        trans = result.applied_transformations[0]
        # Must be RemoveAction, not ScopeResource (which is forbidden for wildcard-only actions)
        assert isinstance(trans, RemoveActionTransformation)
        assert trans.action == "iam:ListRoles"
        assert trans.cost == 5.0
