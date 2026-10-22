from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from iam.dashboard.state import (
    DashboardState,
    PostureKPIs,
    SeverityTier,
    TriageRecord,
    create_demo_dashboard_state,
    create_iac_dashboard_state,
)


class TestTriageDataModels:
    """Tests for triage data structures and enums."""

    def test_severity_tier_values(self) -> None:
        assert SeverityTier.CRITICAL.value == "CRITICAL"
        assert SeverityTier.HIGH.value == "HIGH"
        assert SeverityTier.MEDIUM.value == "MEDIUM"
        assert SeverityTier.LOW.value == "LOW"

    def test_triage_record_creation(self) -> None:
        rec = TriageRecord(
            principal_id="User_dev_0",
            principal_arn="arn:aws:iam::123456789012:user/dev_0",
            principal_type="User",
            department="DevOps",
            is_admin=False,
            observed_risk=0.0,
            latent_attack_prob=0.88,
            severity=SeverityTier.CRITICAL,
            target_assets=["arn:aws:s3:::crown-jewel"],
            max_privilege_tier=3,
            recommended_action="Scope iam:PassRole",
            patch_summary="Scope resource to specific role",
            verification_status="misleading_safe",
            is_misleading=True,
        )
        assert rec.principal_id == "User_dev_0"
        assert rec.severity == SeverityTier.CRITICAL
        assert rec.is_misleading is True


class TestDashboardStateOperations:
    """Tests for DashboardState filtering, selection, and KPIs."""

    @pytest.fixture
    def sample_state(self) -> DashboardState:
        return create_demo_dashboard_state(num_nodes=100, seed=42)

    def test_demo_state_initialization(self, sample_state: DashboardState) -> None:
        assert len(sample_state.triage_records) > 0
        assert sample_state.selected_principal_id is not None
        assert sample_state.graph.num_nodes >= 80

    def test_posture_kpis_computation(self, sample_state: DashboardState) -> None:
        kpis = sample_state.get_posture_kpis()
        assert isinstance(kpis, PostureKPIs)
        assert kpis.total_identities > 0
        assert kpis.total_policies > 0
        assert kpis.flagged_identities_count >= 1
        assert kpis.pareto_dominance_certified is True

    def test_department_filtering(self, sample_state: DashboardState) -> None:
        all_records = sample_state.triage_records
        depts = {r.department for r in all_records}
        test_dept = next(iter(depts))

        filtered = sample_state.filter_triage_records(department=test_dept)
        assert all(r.department.lower() == test_dept.lower() for r in filtered)

    def test_severity_filtering(self, sample_state: DashboardState) -> None:
        filtered = sample_state.filter_triage_records(severity="CRITICAL")
        assert all(r.severity == SeverityTier.CRITICAL for r in filtered)

    def test_search_query_filtering(self, sample_state: DashboardState) -> None:
        rec = sample_state.triage_records[0]
        query = rec.principal_id[:8]

        filtered = sample_state.filter_triage_records(search_query=query)
        assert len(filtered) >= 1
        assert any(r.principal_id == rec.principal_id for r in filtered)

    def test_get_triage_dataframe(self, sample_state: DashboardState) -> None:
        df = sample_state.get_triage_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_state.triage_records)
        assert "Principal ARN" in df.columns
        assert "Latent Probability (P̂)" in df.columns
        assert "Misleading Safe?" in df.columns

    def test_empty_dataframe_schema(self, sample_state: DashboardState) -> None:
        df_empty = sample_state.get_triage_dataframe(search_query="non_existent_identity_xyz")
        assert isinstance(df_empty, pd.DataFrame)
        assert len(df_empty) == 0
        assert "Principal ARN" in df_empty.columns

    def test_principal_selection_and_simulation(self, sample_state: DashboardState) -> None:
        rec2 = sample_state.triage_records[-1]
        sample_state.select_principal(rec2.principal_id)
        assert sample_state.selected_principal_id == rec2.principal_id

        selected = sample_state.get_selected_record()
        assert selected is not None
        assert selected.principal_id == rec2.principal_id

        # Simulation toggle
        assert sample_state.simulation_active is False
        sample_state.toggle_simulation()
        assert sample_state.simulation_active is True
        sample_state.toggle_simulation(False)
        assert sample_state.simulation_active is False


class TestIaCDashboardState:
    """Tests for real-world and synthetic IaC dashboard state ingestion and triage."""

    def test_iac_state_pe_vulnerable(self, tmp_path: Path) -> None:

        state = create_iac_dashboard_state("privilege_escalation_vulnerable", storage_dir=tmp_path)
        assert state.graph.num_nodes >= 4
        assert state.selected_principal_id == "user:intern-dev"

        rec = state.get_selected_record()
        assert rec is not None
        assert rec.is_misleading is True
        assert rec.severity == SeverityTier.CRITICAL
        assert (
            "intern_escalator_policy" in rec.recommended_action
            or "PassRole" in rec.recommended_action
        )

        # Check candidate bridge edges
        assert len(state.candidate_edges) >= 1
        bridge = next(e for e in state.candidate_edges if e.is_bridge)
        assert bridge.source == "user:intern-dev"
        assert bridge.target == "role:lambda-superadmin-exec"

        # Check transformation and verification report
        trans = state.get_selected_transformations()
        assert len(trans) == 1
        assert trans[0].action == "iam:PassRole"

        report = state.get_selected_verification_report()
        assert report is not None
        assert report.is_misleading is True

    def test_iac_state_tac_bench_reference(self, tmp_path: Path) -> None:
        state = create_iac_dashboard_state("tac_bench_reference", storage_dir=tmp_path)
        assert state.graph.num_nodes >= 4
        assert state.selected_principal_id == "user:security-auditor"

        rec = state.get_selected_record()
        assert rec is not None
        assert rec.severity == SeverityTier.CRITICAL
        assert any(e.source == "role:staging-jump-role" for e in state.candidate_edges)

    def test_iac_state_enterprise_microservices(self, tmp_path: Path) -> None:
        state = create_iac_dashboard_state("enterprise_microservices", storage_dir=tmp_path)
        assert state.graph.num_nodes >= 5
        assert state.selected_principal_id == "user:lead-engineer"

        kpis = state.get_posture_kpis()
        assert kpis.total_identities >= 3
        assert kpis.total_policies >= 1
