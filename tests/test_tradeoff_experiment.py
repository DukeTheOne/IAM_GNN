"""Unit and integration test suite for Central Tradeoff Experiment Engine (RQ2 Execution).

Verifies the 6 remediation strategies under Condition E adversarial bridge masking:
- Method 0: Oracle (Upper Bound)
- Method 1: No Action (Deceptive safety baseline)
- Method 2: Coarse Action Deletion (IAMPERE-style heuristic)
- Method 3: Log-Only Tightening (PolicyRefiner-style heuristic)
- Method 4a: Point-Estimate Repair (p >= 0.50 threshold)
- Method 4b: Uncertainty-Aware Repair (Ours: M completions)
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
from iam.optimizer.sampler import CandidateEdge, LatentCompletionSampler
from iam.parser.capability import CapabilityModel, load_default_capability_model
from iam.parser.schema import Effect, PolicyDocument, Statement
from iam.verifier.tradeoff import (
    MethodResult,
    RemediationMethod,
    TradeoffConfig,
    TradeoffEnvironment,
    TradeoffExperimentRunner,
)
from iam.verifier.workload import (
    DriftConfig,
    SyntheticWorkloadGenerator,
    TemporalSplitConfig,
    TemporalWorkloadSplit,
    WorkloadEvent,
)


@pytest.fixture
def capability_model() -> CapabilityModel:
    """Fixture providing loaded AWS CapabilityModel."""
    return load_default_capability_model()


@pytest.fixture
def canonical_tradeoff_setup(capability_model: CapabilityModel) -> TradeoffEnvironment:
    """Fixture providing canonical PassRole + Lambda environment under Condition E.

    Includes:
    - Ground truth graph G* with intact attack path.
    - Observed graph Go with downstream KMS edge masked (Condition E).
    - Latent candidate edge with probability 0.85.
    - Synthetic workload split (T1: 90 days, T2: 30 days) with benign PassRole calls
      and novel T2 drift events.
    """
    gt_graph = IAMGraph("tradeoff_gt_env")
    motif = PassRoleLambdaMotif()
    instance = motif.inject(gt_graph, suffix="rq2")

    source_id = instance.source_id
    target_id = instance.target_id

    # Low-privilege policy on entrypoint user with un-scoped PassRole on *
    source_pol = gt_graph.get_node("policy:lambda-creator-rq2")
    assert source_pol is not None

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
                Resource=["arn:aws:lambda:us-east-1:123456789012:function:escalator-rq2"],
            ),
        ]
    )
    gt_graph.add_node(source_pol.model_copy(update={"policy_document": vulnerable_doc}, deep=True))

    # Attach benign routine policy on entrypoint user
    routine_pol = GraphNode(
        id="policy:dev-routine-rq2",
        node_type=NodeType.POLICY,
        arn="arn:aws:iam::123456789012:policy/dev-routine-rq2",
        name="dev-routine-rq2",
        department="DevOps",
        policy_document=PolicyDocument(
            Statement=[
                Statement(
                    Sid="AllowRoutineLogs",
                    Effect=Effect.ALLOW,
                    Action=["logs:GetLogEvents"],
                    Resource=["*"],
                )
            ]
        ),
    )
    gt_graph.add_node(routine_pol)
    gt_graph.add_edge(
        GraphEdge(source=source_id, target=routine_pol.id, relation=EdgeRelation.ATTACHED_WITH)
    )

    # Observed graph Go has downstream KMS edge severed (Condition E)
    observed_graph = gt_graph.clone()
    kms_pol_id = "policy:kms-full-access-rq2"
    observed_graph.remove_edge(kms_pol_id, target_id, EdgeRelation.ACTS_ON)

    # Candidate latent edge representing the severed relation
    cand = CandidateEdge(
        source=kms_pol_id,
        target=target_id,
        relation=EdgeRelation.ACTS_ON,
        probability=0.85,
        actions=("kms:Decrypt", "kms:GenerateDataKey"),
        is_bridge=True,
    )

    legit_worker_arn = "arn:aws:iam::123456789012:role/legit-worker-rq2"

    # Synthetic workload generator
    workload_gen = SyntheticWorkloadGenerator(
        config=TemporalSplitConfig(
            t1_days=90,
            t2_days=30,
            events_per_day=3,
            seed=42,
            drift_config=DriftConfig(drift_rate=0.25, resource_drift_ratio=0.50),
        ),
        capability_model=capability_model,
    )

    # Generate synthetic workload traces
    split = workload_gen.generate(observed_graph)

    # Inject benign operational PassRole requests for legit_worker_arn into T1 and T2
    source_user = observed_graph.get_node(source_id)
    assert source_user is not None

    t1_passrole = WorkloadEvent(
        event_id="evt-benign-passrole-t1",
        principal_arn=source_user.arn,
        action="iam:PassRole",
        resource_arn=legit_worker_arn,
        day_offset=45.0,
        department="DevOps",
    )
    t2_passrole = WorkloadEvent(
        event_id="evt-benign-passrole-t2",
        principal_arn=source_user.arn,
        action="iam:PassRole",
        resource_arn=legit_worker_arn,
        day_offset=15.0,
        department="DevOps",
    )

    # Inject routine log events in T1 and novel drifted log resource in T2
    t1_routine_log = WorkloadEvent(
        event_id="evt-routine-log-t1",
        principal_arn=source_user.arn,
        action="logs:GetLogEvents",
        resource_arn="arn:aws:logs:us-east-1:123456789012:log-group:dev-app-q1:*",
        day_offset=20.0,
        department="DevOps",
    )
    t2_drift_log = WorkloadEvent(
        event_id="evt-drift-log-t2",
        principal_arn=source_user.arn,
        action="logs:GetLogEvents",
        resource_arn="arn:aws:logs:us-east-1:123456789012:log-group:dev-app-q4-new:*",
        day_offset=20.0,
        department="DevOps",
        is_drift=True,
    )

    augmented_t1 = list(split.t1_events) + [t1_passrole, t1_routine_log]
    augmented_t2 = list(split.t2_events) + [t2_passrole, t2_drift_log]

    updated_split = TemporalWorkloadSplit(
        t1_events=augmented_t1,
        t2_events=augmented_t2,
        metadata=split.metadata,
    )

    return TradeoffEnvironment(
        name="canonical_passrole_rq2",
        ground_truth_graph=gt_graph,
        observed_graph=observed_graph,
        candidate_edges=[cand],
        source_id=source_id,
        target_id=target_id,
        workload_split=updated_split,
        safe_roles={legit_worker_arn},
    )


class TestTradeoffDataModels:
    """Tests evaluating data structures and configuration serialization."""

    def test_remediation_method_enum_values(self) -> None:
        """Verify all 6 remediation methods have consistent string IDs."""
        methods = {
            RemediationMethod.ORACLE: "oracle",
            RemediationMethod.NO_ACTION: "no_action",
            RemediationMethod.COARSE_DELETION: "coarse_deletion",
            RemediationMethod.LOG_ONLY_TIGHTENING: "log_only_tightening",
            RemediationMethod.POINT_ESTIMATE: "point_estimate",
            RemediationMethod.UNCERTAINTY_AWARE: "uncertainty_aware",
        }
        for m_enum, val in methods.items():
            assert m_enum.value == val

    def test_tradeoff_config_defaults(self) -> None:
        """Verify default hyperparameter values for tradeoff evaluation."""
        cfg = TradeoffConfig()
        assert cfg.num_completions == 20
        assert cfg.point_estimate_threshold == 0.50
        assert cfg.t1_days == 90.0
        assert cfg.t2_days == 30.0
        assert cfg.risk_threshold == 0.0


class TestCanonicalPassRoleTradeoffComparison:
    """Evaluates all 6 remediation methods on the canonical PassRole motif under Condition E."""

    def test_run_all_six_methods_end_to_end(
        self,
        canonical_tradeoff_setup: TradeoffEnvironment,
        capability_model: CapabilityModel,
    ) -> None:
        """Run all 6 methods and verify their distinct security vs utility profiles."""
        runner = TradeoffExperimentRunner(
            capability_model=capability_model,
            config=TradeoffConfig(num_completions=20, risk_threshold=0.0),
        )

        results = runner.run_all_methods(canonical_tradeoff_setup, seed=42)
        assert len(results) == 6

        m0 = results[RemediationMethod.ORACLE]
        m1 = results[RemediationMethod.NO_ACTION]
        m2 = results[RemediationMethod.COARSE_DELETION]
        m3 = results[RemediationMethod.LOG_ONLY_TIGHTENING]
        m4a = results[RemediationMethod.POINT_ESTIMATE]
        m4b = results[RemediationMethod.UNCERTAINTY_AWARE]

        # ----------------------------------------------------------------------
        # Method 0: Oracle (G*)
        # ----------------------------------------------------------------------
        assert isinstance(m0, MethodResult)
        assert m0.level3_risk == 0.0
        assert m0.is_safe_gt is True
        assert m0.t1_breakage == 0.0
        assert m0.has_side_effects is False

        # ----------------------------------------------------------------------
        # Method 1: No Action (Identity)
        # ----------------------------------------------------------------------
        assert isinstance(m1, MethodResult)
        assert m1.level1_risk == 0.0  # Appears safe on Go!
        assert m1.level2_risk > 0.70  # Unsafe across completions
        assert m1.level3_risk == 1.0  # Completely unsafe on G*!
        assert m1.is_misleading is True
        assert m1.is_safe_gt is False
        assert m1.t1_breakage == 0.0
        assert m1.t2_breakage == 0.0

        # ----------------------------------------------------------------------
        # Method 2: Coarse Action Deletion (IAMPERE)
        # ----------------------------------------------------------------------
        assert isinstance(m2, MethodResult)
        assert m2.level3_risk == 0.0
        assert m2.is_safe_gt is True
        # Coarse removal of iam:PassRole must break the benign PassRole events in T1 and T2!
        assert m2.t1_denied_requests > 0
        assert m2.t1_breakage > 0.0
        assert m2.t2_denied_requests > 0
        assert m2.t2_breakage > 0.0

        # ----------------------------------------------------------------------
        # Method 3: Log-Only Tightening (PolicyRefiner)
        # ----------------------------------------------------------------------
        assert isinstance(m3, MethodResult)
        assert m3.level3_risk == 0.0  # Latent role was not in T1 logs -> scoped out
        assert m3.is_safe_gt is True
        assert m3.t1_breakage == 0.0  # Zero breakage on historical T1
        # But high breakage on T2 operational drift!
        assert m3.t2_breakage > 0.0
        assert m3.t2_denied_requests > 0

        # ----------------------------------------------------------------------
        # Method 4a: Point-Estimate Repair (p >= 0.50)
        # ----------------------------------------------------------------------
        assert isinstance(m4a, MethodResult)
        # In this setup, p = 0.85 >= 0.50, so edge was included and repaired
        assert m4a.level3_risk == 0.0
        assert m4a.is_safe_gt is True
        assert m4a.t1_breakage == 0.0

        # ----------------------------------------------------------------------
        # Method 4b: Uncertainty-Aware Repair (Ours)
        # ----------------------------------------------------------------------
        assert isinstance(m4b, MethodResult)
        assert m4b.level1_risk == 0.0
        assert m4b.level2_risk <= 0.05
        assert m4b.level3_risk == 0.0
        assert m4b.is_safe_gt is True
        assert m4b.is_misleading is False
        assert m4b.t1_breakage == 0.0  # 100% baseline conformance invariant
        assert m4b.has_side_effects is False

    def test_point_estimate_failure_under_subthreshold_probability(
        self,
        canonical_tradeoff_setup: TradeoffEnvironment,
        capability_model: CapabilityModel,
    ) -> None:
        """Verify Method 4a fails to remediate when predicted edge probability is sub-threshold (p = 0.40).

        This formally proves the vulnerability of point-estimate thresholding to model uncertainty.
        While Method 4a fails (level3_risk = 1.0), Method 4b succeeds (level3_risk = 0.0).
        """
        # Create sub-threshold candidate edge (p = 0.40 < theta = 0.50)
        sub_cand = CandidateEdge(
            source=canonical_tradeoff_setup.candidate_edges[0].source,
            target=canonical_tradeoff_setup.candidate_edges[0].target,
            relation=canonical_tradeoff_setup.candidate_edges[0].relation,
            probability=0.40,  # Below threshold 0.50!
            actions=canonical_tradeoff_setup.candidate_edges[0].actions,
            is_bridge=True,
        )

        sub_env = canonical_tradeoff_setup.model_copy(
            update={"candidate_edges": [sub_cand]}, deep=True
        )

        runner = TradeoffExperimentRunner(
            capability_model=capability_model,
            config=TradeoffConfig(num_completions=20, point_estimate_threshold=0.50),
        )

        # Pre-sample completions
        sampler = LatentCompletionSampler(
            observed_graph=sub_env.observed_graph, candidate_edges=[sub_cand]
        )
        completions = sampler.sample_completions(num_samples=20, seed=42)

        # Method 4a: Drops candidate edge because p = 0.40 < 0.50
        res_4a = runner.run_method_4a_point_estimate(sub_env, completions)
        assert res_4a.level3_risk == 1.0  # Failed to repair!
        assert res_4a.is_safe_gt is False
        assert res_4a.is_misleading is True

        # Method 4b: Accounts for uncertainty across completion set and repairs the bottleneck
        res_4b = runner.run_method_4b_uncertainty_aware(sub_env, completions)
        assert res_4b.level3_risk == 0.0  # Successfully repaired!
        assert res_4b.is_safe_gt is True
        assert res_4b.is_misleading is False

    def test_serialization_of_method_results(
        self,
        canonical_tradeoff_setup: TradeoffEnvironment,
        capability_model: CapabilityModel,
    ) -> None:
        """Verify that MethodResult correctly serializes to dictionary for JSON reports."""
        runner = TradeoffExperimentRunner(capability_model=capability_model)
        sampler = LatentCompletionSampler(
            observed_graph=canonical_tradeoff_setup.observed_graph,
            candidate_edges=canonical_tradeoff_setup.candidate_edges,
        )
        completions = sampler.sample_completions(num_samples=5, seed=42)

        res = runner.run_method_4b_uncertainty_aware(canonical_tradeoff_setup, completions)
        d = res.to_dict()

        assert d["method"] == "uncertainty_aware"
        assert d["level3_risk"] == 0.0
        assert d["is_safe_gt"] is True
        assert d["t1_breakage"] == 0.0
        assert "execution_time_sec" in d
        assert d["num_transformations"] >= 1


class TestMethodTradeoffInvariants:
    """Formal tests verifying Pareto frontier dominance properties across remediation methods."""

    def test_uncertainty_aware_dominates_identity_and_coarse(
        self,
        canonical_tradeoff_setup: TradeoffEnvironment,
        capability_model: CapabilityModel,
    ) -> None:
        """Verify Method 4b strictly dominates Method 1 and Method 2 on the security-utility plane.

        Security-Utility Dominance:
        - Safe: Level 3 Risk <= 0.0
        - Utility: T1 Breakage == 0.0 and T2 Breakage minimal.
        Method 4b is safer than Method 1 (0.0 vs 1.0).
        Method 4b has lower breakage than Method 2 (0.0% vs >0.0% T1 breakage).
        Method 4b has lower T2 breakage than Method 3 (0.0% vs >0.0% T2 breakage).
        """
        runner = TradeoffExperimentRunner(capability_model=capability_model)
        results = runner.run_all_methods(canonical_tradeoff_setup, seed=42)

        m1 = results[RemediationMethod.NO_ACTION]
        m2 = results[RemediationMethod.COARSE_DELETION]
        m3 = results[RemediationMethod.LOG_ONLY_TIGHTENING]
        m4b = results[RemediationMethod.UNCERTAINTY_AWARE]

        # Method 4b is strictly safer than Method 1 on ground truth G*
        assert m4b.level3_risk < m1.level3_risk
        assert m4b.level3_risk == 0.0
        assert m1.level3_risk == 1.0

        # Method 4b preserves historical baseline T1 (0% breakage) unlike Method 2
        assert m4b.t1_breakage == 0.0
        assert m2.t1_breakage > 0.0
        assert m4b.t1_breakage < m2.t1_breakage

        # Method 4b is resilient to T2 operational drift unlike Method 3
        assert m4b.t2_breakage < m3.t2_breakage
        assert m3.t2_breakage > 0.0


class TestEnterpriseTradeoffScaling:
    """Evaluates runtime scaling of TradeoffExperimentRunner on 150-node enterprise topology."""

    def test_tradeoff_runner_performance_on_enterprise_graph(
        self,
        capability_model: CapabilityModel,
    ) -> None:
        """Verify all methods execute and complete within strict latency limits (< 3.0s)."""
        import time

        from iam.generator.topology import EnterpriseTopologyConfig, EnterpriseTopologyGenerator

        topo_config = EnterpriseTopologyConfig(
            num_nodes=150,
            seed=2026,
            num_pe_chains=1,
            pe_motif_types=["passrole_lambda"],
            account_id="123456789012",
            organization_id="o-tradeoff-scale",
        )
        topo_gen = EnterpriseTopologyGenerator(
            config=topo_config, capability_model=capability_model
        )
        gt_graph = topo_gen.generate()

        injected = gt_graph.metadata.get("injected_motifs", [])
        assert len(injected) >= 1
        m_info = injected[0]
        s_id = m_info["source_id"]
        t_id = m_info["target_id"]

        # Sever downstream KMS edge
        kms_pols = [n for n in m_info["intermediate_node_ids"] if "kms-full-access" in n]
        assert len(kms_pols) >= 1
        kms_pol_id = kms_pols[0]

        obs_graph = gt_graph.clone()
        obs_graph.remove_edge(kms_pol_id, t_id, EdgeRelation.ACTS_ON)

        cand = CandidateEdge(
            source=kms_pol_id,
            target=t_id,
            relation=EdgeRelation.ACTS_ON,
            probability=0.85,
            actions=("kms:Decrypt", "kms:GenerateDataKey"),
            is_bridge=True,
        )

        w_gen = SyntheticWorkloadGenerator(
            config=TemporalSplitConfig(t1_days=30, t2_days=10, events_per_day=3, seed=42),
            capability_model=capability_model,
        )
        split = w_gen.generate(obs_graph)

        env = TradeoffEnvironment(
            name="enterprise_150_rq2",
            ground_truth_graph=gt_graph,
            observed_graph=obs_graph,
            candidate_edges=[cand],
            source_id=s_id,
            target_id=t_id,
            workload_split=split,
        )

        runner = TradeoffExperimentRunner(
            capability_model=capability_model,
            config=TradeoffConfig(num_completions=15),
        )

        start = time.perf_counter()
        results = runner.run_all_methods(env, seed=42)
        total_time = time.perf_counter() - start

        assert len(results) == 6
        assert total_time < 3.0  # Pipeline scalability budget
        assert results[RemediationMethod.UNCERTAINTY_AWARE].level3_risk == 0.0
        assert results[RemediationMethod.NO_ACTION].is_misleading is True
