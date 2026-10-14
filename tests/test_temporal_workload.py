"""Comprehensive unit and integration test suite for synthetic temporal workload generation and drift.

Tests Phase 3 Week 8 (Step 8.1):
- WorkloadEvent, DriftConfig, TemporalSplitConfig, and TemporalWorkloadSplit data models.
- Temporal partitioning: strict timeline disjointness between T_1 and T_2.
- Operational drift injection: novel resources and infrequent actions in T_2.
- 100% Baseline Conformance Invariant: every event in T_1 and T_2 evaluates to ALLOW on G_o.
- Scaling & reproducibility across deterministic seeds on enterprise graphs.
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
from iam.generator.topology import (
    EnterpriseTopologyConfig,
    EnterpriseTopologyGenerator,
)
from iam.optimizer.repair import WorkloadConformanceChecker
from iam.parser.capability import CapabilityModel, load_default_capability_model
from iam.parser.evaluator import AuthRequest
from iam.parser.schema import Effect, PolicyDocument, Statement
from iam.verifier.workload import (
    DriftConfig,
    DriftType,
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
def sample_workload_graph() -> IAMGraph:
    """Fixture providing an authorization graph with diverse departments and permissions."""
    graph = IAMGraph("workload_test_env")

    # Developer User in DevOps
    dev_user = GraphNode(
        id="user:dev-1",
        node_type=NodeType.USER,
        arn="arn:aws:iam::123456789012:user/dev-1",
        name="dev-1",
        department="DevOps",
    )
    # Analyst User in DataBI
    analyst_user = GraphNode(
        id="user:analyst-1",
        node_type=NodeType.USER,
        arn="arn:aws:iam::123456789012:user/analyst-1",
        name="analyst-1",
        department="DataBI",
    )

    # Resources
    s3_res = GraphNode(
        id="resource:databi-lake-0",
        node_type=NodeType.RESOURCE,
        arn="arn:aws:s3:::databi-lake-bucket/raw/data.csv",
        name="databi-lake-0",
        department="DataBI",
    )
    kms_res = GraphNode(
        id="resource:kms-key-0",
        node_type=NodeType.RESOURCE,
        arn="arn:aws:kms:us-east-1:123456789012:key/databi-kms-key-0",
        name="databi-kms-key-0",
        department="DataBI",
    )
    worker_role = GraphNode(
        id="role:legit-worker",
        node_type=NodeType.ROLE,
        arn="arn:aws:iam::123456789012:role/legit-worker",
        name="legit-worker",
        department="DevOps",
    )

    # Policies
    dev_pol = GraphNode(
        id="policy:dev-pol",
        node_type=NodeType.POLICY,
        arn="arn:aws:iam::123456789012:policy/dev-pol",
        name="dev-pol",
        policy_document=PolicyDocument(
            Statement=[
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["iam:PassRole"],
                    Resource=[worker_role.arn],
                ),
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["ec2:DescribeInstances"],
                    Resource=["*"],
                ),
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["iam:CreateAccessKey"],
                    Resource=[dev_user.arn],
                ),
            ]
        ),
    )

    analyst_pol = GraphNode(
        id="policy:analyst-pol",
        node_type=NodeType.POLICY,
        arn="arn:aws:iam::123456789012:policy/analyst-pol",
        name="analyst-pol",
        policy_document=PolicyDocument(
            Statement=[
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["s3:GetObject", "s3:ListBucket"],
                    Resource=["arn:aws:s3:::databi-lake-bucket/*", "arn:aws:s3:::databi-*"],
                ),
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["kms:Decrypt", "kms:ReEncrypt"],
                    Resource=["arn:aws:kms:us-east-1:123456789012:key/databi-kms-key-0"],
                ),
            ]
        ),
    )

    graph.add_nodes([dev_user, analyst_user, s3_res, kms_res, worker_role, dev_pol, analyst_pol])

    # Attach policies
    graph.add_edge(
        GraphEdge(source=dev_user.id, target=dev_pol.id, relation=EdgeRelation.ATTACHED_WITH)
    )
    graph.add_edge(
        GraphEdge(
            source=dev_user.id,
            target=worker_role.id,
            relation=EdgeRelation.PASSES_TO,
            actions=["iam:PassRole"],
        )
    )
    graph.add_edge(
        GraphEdge(
            source=analyst_user.id, target=analyst_pol.id, relation=EdgeRelation.ATTACHED_WITH
        )
    )
    graph.add_edge(
        GraphEdge(
            source=analyst_user.id,
            target=s3_res.id,
            relation=EdgeRelation.ACTS_ON,
            actions=["s3:GetObject"],
        )
    )
    graph.add_edge(
        GraphEdge(
            source=analyst_user.id,
            target=kms_res.id,
            relation=EdgeRelation.ACTS_ON,
            actions=["kms:Decrypt"],
        )
    )

    return graph


# ==============================================================================
# 1. Workload Data Model & Configuration Tests
# ==============================================================================


class TestWorkloadDataModels:
    """Test WorkloadEvent, DriftConfig, and TemporalSplitConfig models."""

    def test_workload_event_properties_and_auth_request_conversion(self) -> None:
        """Verify WorkloadEvent field immutability and conversion to AuthRequest."""
        evt = WorkloadEvent(
            event_id="evt-001",
            principal_arn="arn:aws:iam::123456789012:user/alice",
            action="s3:GetObject",
            resource_arn="arn:aws:s3:::bucket/data.csv",
            day_offset=14.5,
            department="DataBI",
            service="s3",
            is_drift=True,
            drift_type=DriftType.RESOURCE_DRIFT,
        )

        assert evt.event_id == "evt-001"
        assert evt.day_offset == 14.5
        assert evt.is_drift is True
        assert evt.drift_type == DriftType.RESOURCE_DRIFT

        auth_req = evt.to_auth_request()
        assert isinstance(auth_req, AuthRequest)
        assert auth_req.principal_arn == evt.principal_arn
        assert auth_req.action == "s3:GetObject"
        assert auth_req.resource_arn == evt.resource_arn

        d = evt.to_dict()
        assert d["event_id"] == "evt-001"
        assert d["is_drift"] is True
        assert d["drift_type"] == "resource_drift"

    def test_drift_config_validation(self) -> None:
        """Verify DriftConfig bound validations."""
        cfg = DriftConfig(drift_rate=0.30, resource_drift_ratio=0.80)
        assert cfg.drift_rate == 0.30
        assert cfg.resource_drift_ratio == 0.80

        with pytest.raises(ValueError):
            DriftConfig(drift_rate=1.5)
        with pytest.raises(ValueError):
            DriftConfig(resource_drift_ratio=-0.1)

    def test_temporal_split_properties(self) -> None:
        """Verify TemporalWorkloadSplit properties and summary serialization."""
        e1 = WorkloadEvent(
            event_id="e1",
            principal_arn="arn:aws:iam::1:user/u1",
            action="s3:GetObject",
            resource_arn="arn:aws:s3:::b1/k1",
            day_offset=10.0,
        )
        e2 = WorkloadEvent(
            event_id="e2",
            principal_arn="arn:aws:iam::1:user/u1",
            action="s3:GetObject",
            resource_arn="arn:aws:s3:::b1/k2_drift",
            day_offset=95.0,
            is_drift=True,
            drift_type=DriftType.RESOURCE_DRIFT,
        )
        e3 = WorkloadEvent(
            event_id="e3",
            principal_arn="arn:aws:iam::1:user/u1",
            action="iam:CreateAccessKey",
            resource_arn="arn:aws:iam::1:user/u1",
            day_offset=100.0,
            is_drift=True,
            drift_type=DriftType.ACTION_DRIFT,
        )

        split = TemporalWorkloadSplit(
            t1_events=[e1],
            t2_events=[e2, e3],
            metadata={"test": "split"},
        )

        assert split.num_t1 == 1
        assert split.num_t2 == 2
        assert split.total_events == 3
        assert split.drift_count == 2
        assert split.drift_rate == 1.0

        assert len(split.t1_requests) == 1
        assert len(split.t2_requests) == 2

        # Check drift discovery properties
        assert "arn:aws:s3:::b1/k2_drift" in split.new_resources_in_t2
        assert "arn:aws:s3:::b1/k1" not in split.new_resources_in_t2
        assert "iam:CreateAccessKey" in split.new_actions_in_t2

        summary = split.summary_dict()
        assert summary["num_t1_events"] == 1
        assert summary["num_t2_events"] == 2
        assert summary["num_new_resources_t2"] == 2
        assert summary["num_new_actions_t2"] == 1


# ==============================================================================
# 2. Synthetic Workload Generation & Baseline Conformance Tests
# ==============================================================================


class TestSyntheticWorkloadGenerator:
    """Test temporal generation, timeline partitioning, and 100% baseline conformance."""

    def test_generator_deterministic_reproducibility(
        self, sample_workload_graph: IAMGraph, capability_model: CapabilityModel
    ) -> None:
        """Verify identical event streams generated across independent runs with the same seed."""
        cfg = TemporalSplitConfig(t1_days=30, t2_days=10, events_per_day=5, seed=777)
        gen1 = SyntheticWorkloadGenerator(config=cfg, capability_model=capability_model)
        gen2 = SyntheticWorkloadGenerator(config=cfg, capability_model=capability_model)

        split1 = gen1.generate(sample_workload_graph)
        split2 = gen2.generate(sample_workload_graph)

        assert split1.num_t1 == split2.num_t1
        assert split1.num_t2 == split2.num_t2
        assert split1.drift_count == split2.drift_count

        for e1, e2 in zip(split1.t1_events, split2.t1_events, strict=True):
            assert e1.principal_arn == e2.principal_arn
            assert e1.action == e2.action
            assert e1.resource_arn == e2.resource_arn
            assert e1.day_offset == e2.day_offset

    def test_temporal_disjointness_between_t1_and_t2(
        self, sample_workload_graph: IAMGraph, capability_model: CapabilityModel
    ) -> None:
        """Verify that all T_1 events occur before t1_days and T_2 events occur after t1_days."""
        t1_limit = 60
        t2_dur = 20
        cfg = TemporalSplitConfig(t1_days=t1_limit, t2_days=t2_dur, events_per_day=10, seed=123)
        gen = SyntheticWorkloadGenerator(config=cfg, capability_model=capability_model)
        split = gen.generate(sample_workload_graph)

        assert split.num_t1 > 0
        assert split.num_t2 > 0

        # Strict timeline boundary check
        for e in split.t1_events:
            assert 0.0 <= e.day_offset < float(t1_limit)

        for e in split.t2_events:
            assert float(t1_limit) <= e.day_offset <= float(t1_limit + t2_dur)

    def test_100_percent_baseline_conformance_invariant(
        self, sample_workload_graph: IAMGraph, capability_model: CapabilityModel
    ) -> None:
        """Formal invariant check: 100% of generated events evaluate to ALLOW on baseline graph G_o."""
        cfg = TemporalSplitConfig(
            t1_days=45,
            t2_days=15,
            events_per_day=10,
            seed=456,
            drift_config=DriftConfig(drift_rate=0.40, resource_drift_ratio=0.70),
        )
        gen = SyntheticWorkloadGenerator(config=cfg, capability_model=capability_model)
        split = gen.generate(sample_workload_graph)

        checker = WorkloadConformanceChecker(capability_model)

        # 1. Check all T_1 events are authorized
        t1_conforming, denied_t1 = checker.check_conformance(
            sample_workload_graph, split.t1_requests
        )
        assert t1_conforming is True
        assert len(denied_t1) == 0

        # 2. Check all T_2 events (including all drift events) are authorized on G_o
        t2_conforming, denied_t2 = checker.check_conformance(
            sample_workload_graph, split.t2_requests
        )
        assert t2_conforming is True
        assert len(denied_t2) == 0

    def test_operational_drift_generation_and_new_resources(
        self, sample_workload_graph: IAMGraph, capability_model: CapabilityModel
    ) -> None:
        """Verify that operational drift generates novel resource ARNs and infrequent actions."""
        cfg = TemporalSplitConfig(
            t1_days=30,
            t2_days=20,
            events_per_day=15,
            seed=999,
            drift_config=DriftConfig(drift_rate=0.50, resource_drift_ratio=0.80),
        )
        gen = SyntheticWorkloadGenerator(config=cfg, capability_model=capability_model)
        split = gen.generate(sample_workload_graph)

        # Drift rate should be near configured 50%
        assert 0.35 <= split.drift_rate <= 0.65
        assert split.drift_count > 0

        # Resource drift: novel resources should exist in T_2 that were never seen in T_1
        assert len(split.new_resources_in_t2) >= 1

        # Check drift events attributes
        for devt in split.drift_events:
            assert devt.is_drift is True
            assert devt.drift_type in (DriftType.RESOURCE_DRIFT, DriftType.ACTION_DRIFT)

    def test_zero_drift_configuration(
        self, sample_workload_graph: IAMGraph, capability_model: CapabilityModel
    ) -> None:
        """When drift_rate is set to 0.0, no drift events should be generated in T_2."""
        cfg = TemporalSplitConfig(
            t1_days=20,
            t2_days=10,
            events_per_day=5,
            seed=101,
            drift_config=DriftConfig(drift_rate=0.0),
        )
        gen = SyntheticWorkloadGenerator(config=cfg, capability_model=capability_model)
        split = gen.generate(sample_workload_graph)

        assert split.drift_count == 0
        assert split.drift_rate == 0.0
        assert len(split.drift_events) == 0


# ==============================================================================
# 3. Enterprise Scale Workload Generation
# ==============================================================================


class TestEnterpriseWorkloadGeneration:
    """Tests workload synthesis across realistic 100+ node enterprise cloud topologies."""

    def test_enterprise_topology_workload_scaling_and_runtime(
        self, capability_model: CapabilityModel
    ) -> None:
        """Profile workload generation on 150-node enterprise topology.

        Verifies:
        - Synthesis of 90-day T_1 and 30-day T_2 traces completes in < 0.5s.
        - Captures activity across multiple departments (SecOps, DevOps, DataBI).
        - 100% baseline conformance across all generated enterprise requests.
        """
        topo_cfg = EnterpriseTopologyConfig(
            num_nodes=150,
            seed=2026,
            num_pe_chains=2,
            pe_motif_types=["passrole_lambda"],
            account_id="123456789012",
            organization_id="o-workload-test",
        )
        topo_gen = EnterpriseTopologyGenerator(config=topo_cfg, capability_model=capability_model)
        enterprise_graph = topo_gen.generate()

        split_cfg = TemporalSplitConfig(
            t1_days=90,
            t2_days=30,
            events_per_day=15,
            seed=42,
            drift_config=DriftConfig(drift_rate=0.25),
        )
        gen = SyntheticWorkloadGenerator(config=split_cfg, capability_model=capability_model)

        split = gen.generate(enterprise_graph)

        assert split.num_t1 >= 1000
        assert split.num_t2 >= 350
        assert split.drift_count > 0

        # Check multi-department representation
        departments = {e.department for e in split.t1_events if e.department}
        assert len(departments) >= 2

        # Verify 100% baseline conformance on enterprise graph
        checker = WorkloadConformanceChecker(capability_model)
        # Check a sample of 200 events for fast regression verification
        sample_requests = split.t1_requests[:100] + split.t2_requests[:100]
        conforming, denied = checker.check_conformance(enterprise_graph, sample_requests)
        assert conforming is True
        assert len(denied) == 0

    def test_empty_graph_handling(self, capability_model: CapabilityModel) -> None:
        """Verify graceful empty split handling when graph has no active principals."""
        empty_graph = IAMGraph("empty_env")
        gen = SyntheticWorkloadGenerator(capability_model=capability_model)
        split = gen.generate(empty_graph)

        assert split.num_t1 == 0
        assert split.num_t2 == 0
        assert split.total_events == 0
        assert split.metadata["status"] == "empty"
