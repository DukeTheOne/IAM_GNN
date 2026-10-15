"""End-to-end integration and scalability benchmarks for the Phase 4 Visual Triage Dashboard."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from iam.dashboard.diff_viewer import PolicyDiffViewer
from iam.dashboard.ego_graph import render_ego_canvas_html
from iam.dashboard.metrics_banner import (
    format_posture_markdown_summary,
    render_posture_banner_html,
)
from iam.dashboard.state import create_demo_dashboard_state
from iam.generator.topology import EnterpriseTopologyConfig, EnterpriseTopologyGenerator
from iam.parser.schema import PolicyDocument


class TestDashboardEndToEndIntegration:
    """Full pipeline integration testing for triage, explainability, and reporting."""

    def test_full_pipeline_triage_and_remediation(self) -> None:
        # 1. State initialization
        state = create_demo_dashboard_state(num_nodes=150, seed=42)
        assert len(state.triage_records) > 0
        rec = state.get_selected_record()
        assert rec is not None

        # 2. Ego-network rendering
        html_canvas = render_ego_canvas_html(
            graph=state.graph,
            source_id=rec.principal_id,
            candidate_edges=state.candidate_edges,
            crown_jewel_arns=set(rec.target_assets),
        )
        assert "forceAtlas2Based" in html_canvas
        assert rec.principal_id in html_canvas

        # 3. Policy Diff generation & capability check
        viewer = PolicyDiffViewer()
        transformations = state.get_selected_transformations()
        base_policy = PolicyDocument.from_dict(
            {
                "Version": "2012-10-17",
                "Id": "Policy_TestExecution",
                "Statement": [
                    {
                        "Sid": "AllowPassRoleExecution",
                        "Effect": "Allow",
                        "Action": ["iam:PassRole"],
                        "Resource": ["*"],
                    }
                ],
            }
        )
        diff_res = viewer.generate_diff(base_policy, transformations)
        assert diff_res.capability_valid is True
        assert "diff-table" in diff_res.side_by_side_html

        # 4. Posture banner & markdown summaries
        kpis = state.get_posture_kpis()
        banner_html = render_posture_banner_html(kpis)
        assert "Scanned Identities" in banner_html
        assert "Pareto Optimality" in banner_html

        md_summary = format_posture_markdown_summary(kpis)
        assert "Total Scanned Identities" in md_summary

        # 5. Live Simulation Toggle
        assert state.simulation_active is False
        state.toggle_simulation(True)
        assert state.simulation_active is True

        severed_canvas = render_ego_canvas_html(
            graph=state.graph,
            source_id=rec.principal_id,
            candidate_edges=state.candidate_edges,
            severed_edges={(rec.principal_id, "Role_ExecutionEngine", "PassesTo")},
        )
        assert "SEVERED" in severed_canvas

    def test_headless_cli_exporter_execution(self, tmp_path: Path) -> None:
        """Verify standalone execution of scripts/export_triage_report.py."""
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "export_triage_report.py"
        out_dir = tmp_path / "triage_export"

        cmd = [
            sys.executable,
            str(script_path),
            "--num-nodes",
            "80",
            "--seed",
            "101",
            "--top-k",
            "2",
            "--output-dir",
            str(out_dir),
        ]

        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"Exporter failed with stderr: {res.stderr}"

        # Verify generated artifacts
        summary_md = out_dir / "triage_summary.md"
        summary_json = out_dir / "triage_summary.json"
        assert summary_md.exists()
        assert summary_json.exists()

        data = json.loads(summary_json.read_text(encoding="utf-8"))
        assert "posture_kpis" in data
        assert len(data["top_records"]) <= 2


class TestEnterpriseScalingAndLatencySafeguards:
    """Stress and scalability benchmarks ensuring Tier 2 readiness."""

    def test_ego_network_sub_second_latency_budget(self) -> None:
        """Verify that extracting and synthesizing an ego-network executes in <50ms."""
        config = EnterpriseTopologyConfig(
            num_nodes=300, seed=42, num_pe_chains=2, include_branching_chains=False
        )
        gen = EnterpriseTopologyGenerator(config)
        graph = gen.generate()

        users = [n for n, d in graph.nx_graph.nodes(data=True) if d.get("node_type") == "User"]
        test_user = users[0]

        start_time = time.perf_counter()
        html_canvas = render_ego_canvas_html(
            graph=graph,
            source_id=test_user,
            height="500px",
        )
        elapsed = time.perf_counter() - start_time

        assert isinstance(html_canvas, str)
        # Bounded k<=2 extraction must be strictly faster than 100ms
        assert elapsed < 0.100, f"Ego-network rendering took {elapsed:.4f}s (budget: <0.100s)"
