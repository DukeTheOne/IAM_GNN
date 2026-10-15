#!/usr/bin/env python3
"""Headless CLI triage exporter for cloud IAM privilege escalation.

Implements Phase 4 Week 9 (Step 9.4):
- Generates static, self-contained HTML PyVis ego-subgraphs and Markdown diff audit reports.
- Runs in headless server environments and CI/CD automated security gates without Streamlit.
- Provides fallback deliverable per project blueprint (static PyVis HTML + JSON policy diff).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from iam.dashboard.diff_viewer import (
    PolicyDiffViewer,
    render_impact_summary,
)
from iam.dashboard.ego_graph import render_ego_canvas_html
from iam.dashboard.metrics_banner import format_posture_markdown_summary
from iam.dashboard.state import create_demo_dashboard_state
from iam.parser.schema import PolicyDocument

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("export_triage_report")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the headless triage exporter."""
    parser = argparse.ArgumentParser(
        description="Headless CLI triage exporter for IAM privilege escalation."
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=150,
        help="Total node count for the synthetic enterprise topology (default: 150)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/triage_export",
        help="Target directory to save static HTML and Markdown reports (default: reports/triage_export)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of highest-risk principals to export deep-dive ego-networks for (default: 5)",
    )
    return parser.parse_args(args)


def run_export(args: argparse.Namespace) -> int:
    """Execute the headless batch triage report export.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code: 0 on success, non-zero on failure.
    """
    start_time = time.perf_counter()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing IAM state with %d nodes (seed=%d)...", args.num_nodes, args.seed)
    state = create_demo_dashboard_state(num_nodes=args.num_nodes, seed=args.seed)

    kpis = state.get_posture_kpis()
    logger.info(
        "Evaluated %d identities, %d policies. Flagged %d latent risks.",
        kpis.total_identities,
        kpis.total_policies,
        kpis.flagged_identities_count,
    )

    # Export Posture Markdown Summary
    md_summary = format_posture_markdown_summary(kpis)
    summary_path = out_dir / "triage_summary.md"

    # Compile Table of Top Flagged Identities
    top_records = state.triage_records[: args.top_k]

    table_header = (
        "\n\n### Top High-Risk Identities (Calibrated Latent Privilege Escalation)\n\n"
        "| Principal ID | Type | Department | Observed Risk (G_o) | Latent P̂ | Severity | Misleading Safe? | Target Asset |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n"
    )
    table_rows: list[str] = []
    for r in top_records:
        misleading_tag = "**YES (⚠️ Blind)**" if r.is_misleading else "No"
        target_str = r.target_assets[0] if r.target_assets else "None"
        row = (
            f"| `{r.principal_id}` | {r.principal_type} | {r.department} | "
            f"`{r.observed_risk:.2f}` | `{r.latent_attack_prob:.2f}` | "
            f"**{r.severity.value}** | {misleading_tag} | `{target_str}` |"
        )
        table_rows.append(row)

    full_md = md_summary + table_header + "\n".join(table_rows) + "\n"
    summary_path.write_text(full_md, encoding="utf-8")
    logger.info("Wrote posture summary report to: %s", summary_path)

    # Export Structured JSON
    json_data: dict[str, Any] = {
        "posture_kpis": kpis.model_dump(),
        "top_records": [r.model_dump() for r in top_records],
        "metadata": {
            "num_nodes": args.num_nodes,
            "seed": args.seed,
            "timestamp": time.time(),
        },
    }
    json_path = out_dir / "triage_summary.json"
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    logger.info("Wrote machine-readable JSON dataset to: %s", json_path)

    # Export Deep-Dive Ego-Networks and Policy Diffs for Top-K Principals
    diff_viewer = PolicyDiffViewer()
    base_policy = PolicyDocument.from_dict(
        {
            "Version": "2012-10-17",
            "Id": "Policy_DeveloperExecution",
            "Statement": [
                {
                    "Sid": "AllowPassRoleExecution",
                    "Effect": "Allow",
                    "Action": ["iam:PassRole"],
                    "Resource": ["*"],
                },
                {
                    "Sid": "AllowLambdaInvokeRoutine",
                    "Effect": "Allow",
                    "Action": ["lambda:InvokeFunction"],
                    "Resource": ["arn:aws:lambda:us-east-1:123456789012:function:test-*"],
                },
            ],
        }
    )

    for i, r in enumerate(top_records, 1):
        logger.info("Exporting ego-network [%d/%d] for: %s", i, len(top_records), r.principal_id)

        # 1. Export Self-Contained HTML PyVis Canvas
        html_canvas = render_ego_canvas_html(
            graph=state.graph,
            source_id=r.principal_id,
            candidate_edges=state.candidate_edges,
            crown_jewel_arns=set(r.target_assets),
            height="650px",
        )
        html_file = out_dir / f"ego_graph_{r.principal_id}.html"
        html_file.write_text(html_canvas, encoding="utf-8")

        # 2. Export Policy Diff
        trans_list = state.transformations.get(r.principal_id, [])
        diff_res = diff_viewer.generate_diff(base_policy, trans_list)

        diff_file = out_dir / f"policy_diff_{r.principal_id}.md"
        report = state.verification_reports.get(r.principal_id)
        impact = render_impact_summary(report, len(trans_list))

        diff_md = (
            f"# Remediation Diff: `{r.principal_id}`\n\n"
            f"- **Severity Tier:** `{r.severity.value}` (Latent Attack P̂ = `{r.latent_attack_prob:.2f}`)\n"
            f"- **Capability Status:** `{'VALID (Admissible under C(a,r,c))' if diff_res.capability_valid else 'INVALID'}`\n"
            f"- **Blast Radius Reduction:** `{impact['blast_radius_reduction']}`\n"
            f"- **Historical T_1 Conformance:** `{impact['t1_conformance']}`\n"
            f"- **Holdout T_2 Operational Breakage:** `{impact['t2_breakage_rate']}`\n\n"
            f"### Unified Policy AST Diff\n\n"
            f"```diff\n{diff_res.unified_diff}\n```\n\n"
            f"### Standalone Graph Canvas\n\n"
            f"Open [`ego_graph_{r.principal_id}.html`](./ego_graph_{r.principal_id}.html) in any browser.\n"
        )
        diff_file.write_text(diff_md, encoding="utf-8")

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Successfully exported %d triage artifacts to %s in %.3f seconds.",
        len(top_records) * 2 + 2,
        out_dir,
        elapsed,
    )
    return 0


def main() -> None:
    """CLI execution entrypoint."""
    args = parse_args()
    sys.exit(run_export(args))


if __name__ == "__main__":
    main()
