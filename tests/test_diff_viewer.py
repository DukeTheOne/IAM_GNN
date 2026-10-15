"""Unit tests for PolicyDiffViewer, capability badge, and impact summaries."""

from __future__ import annotations

import pytest

from iam.dashboard.diff_viewer import (
    PolicyASTDiff,
    PolicyDiffViewer,
    render_capability_badge,
    render_impact_summary,
)
from iam.optimizer.transformations import (
    RemoveActionTransformation,
    ScopeResourceTransformation,
)
from iam.parser.schema import PolicyDocument
from iam.verifier.verifier import (
    SafetyStatus,
    SideEffectResult,
    VerificationReport,
    WorkloadReplayResult,
)


@pytest.fixture
def sample_policy() -> PolicyDocument:
    """Fixture providing a base wildcard policy document."""
    return PolicyDocument.from_dict(
        {
            "Version": "2012-10-17",
            "Id": "Policy_CI_Runner",
            "Statement": [
                {
                    "Sid": "PassRoleWildcard",
                    "Effect": "Allow",
                    "Action": ["iam:PassRole"],
                    "Resource": ["*"],
                },
                {
                    "Sid": "S3Read",
                    "Effect": "Allow",
                    "Action": ["s3:GetObject"],
                    "Resource": ["arn:aws:s3:::ci-cache/*"],
                },
            ],
        }
    )


class TestPolicyDiffViewer:
    """Tests for PolicyASTDiff synthesis and capability validation."""

    def test_scope_resource_diff_synthesis(self, sample_policy: PolicyDocument) -> None:
        viewer = PolicyDiffViewer()
        trans = ScopeResourceTransformation(
            policy_id="Policy_CI_Runner",
            statement_index=0,
            action="iam:PassRole",
            target_resources=["arn:aws:iam::123456789012:role/RunnerExecutionRole"],
            original_resources=["*"],
            cost=1.0,
        )

        diff = viewer.generate_diff(sample_policy, [trans])

        assert isinstance(diff, PolicyASTDiff)
        assert diff.policy_id == "Policy_CI_Runner"
        assert diff.capability_valid is True
        assert diff.actions_scoped == ["iam:PassRole"]
        assert diff.actions_removed == []
        assert "RunnerExecutionRole" in diff.repaired_json
        assert '-        "*"' in diff.unified_diff
        assert '+        "arn:aws:iam::123456789012:role/RunnerExecutionRole"' in diff.unified_diff
        assert "diff-table" in diff.side_by_side_html
        assert "RunnerExecutionRole" in diff.side_by_side_html

    def test_remove_action_diff_synthesis(self, sample_policy: PolicyDocument) -> None:
        viewer = PolicyDiffViewer()
        trans = RemoveActionTransformation(
            policy_id="Policy_CI_Runner",
            statement_index=0,
            action="iam:PassRole",
            cost=5.0,
        )

        diff = viewer.generate_diff(sample_policy, [trans])

        assert diff.capability_valid is True
        assert diff.actions_removed == ["iam:PassRole"]
        assert diff.total_cost == 5.0
        assert "diff-table" in diff.side_by_side_html

    def test_capability_badge_rendering(self) -> None:
        valid_html = render_capability_badge(True, "Admissible under AWS Authorization Model")
        assert "VALIDATED" in valid_html
        assert "#38A169" in valid_html or "border: 1px solid" in valid_html

        invalid_html = render_capability_badge(False, "Wildcard only action illegally scoped")
        assert "REJECTED" in invalid_html
        assert "#E53E3E" in invalid_html or "border: 1px solid" in invalid_html

    def test_impact_summary_with_report(self) -> None:
        report = VerificationReport(
            source_id="User_dev",
            target_id="arn:aws:s3:::crown-jewel",
            level1_risk_observed=0.0,
            level2_risk_completion_set=0.85,
            level3_risk_ground_truth=1.0,
            level1_safe=True,
            level2_safe=False,
            level3_safe=False,
            overall_status=SafetyStatus.MISLEADING_SAFE,
            is_misleading=True,
            side_effects=SideEffectResult(
                has_side_effects=False,
                new_escalation_pairs=[],
                num_pairs_checked=50,
                details="Zero new escalation pairs",
            ),
            t1_replay=WorkloadReplayResult(
                window_name="T1",
                total_requests=200,
                allowed_requests=200,
                denied_requests=0,
                breakage_rate=0.0,
                is_conforming=True,
            ),
            t2_replay=WorkloadReplayResult(
                window_name="T2",
                total_requests=100,
                allowed_requests=98,
                denied_requests=2,
                breakage_rate=0.02,
                is_conforming=False,
            ),
        )

        summary = render_impact_summary(report, num_transformations=1)
        assert summary["blast_radius_reduction"] == "100.0%"
        assert "100.0%" in summary["t1_conformance"]
        assert summary["t2_breakage_rate"] == "2.00%"
        assert summary["safety_verdict"] == "MISLEADING_SAFE"
        assert summary["is_misleading"] is True

    def test_impact_summary_none_report(self) -> None:
        summary = render_impact_summary(None, num_transformations=2)
        assert summary["blast_radius_reduction"] == "100.0%"
        assert summary["mutations_count"] == 2
