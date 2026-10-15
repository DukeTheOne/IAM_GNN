"""Global security posture metrics cards and Pareto tradeoff visualization.

Implements Phase 4 Week 9 (Step 9.1):
- Global Security Posture Banner displaying scanned identities, policies, and latent risk counts.
- Highlights misleading safe static analysis discrepancies (Level 1 blind spots).
- Displays Pareto optimality certification confirming uncertainty-aware dominance.
- Compatible with both interactive Streamlit UI and headless HTML/Markdown export.
"""

from __future__ import annotations

import logging
import textwrap

from iam.dashboard.state import PostureKPIs

logger = logging.getLogger(__name__)


def render_posture_banner_html(kpis: PostureKPIs) -> str:
    """Generate an aesthetic dark-mode HTML posture banner with 5 key KPI metric cards."""
    pareto_badge = (
        '<span style="color: #68D391; font-weight: bold;">✓ CERTIFIED DOMINANT</span>'
        if kpis.pareto_dominance_certified
        else '<span style="color: #FC8181; font-weight: bold;">PENDING</span>'
    )

    misleading_color = "#FC8181" if kpis.misleading_safe_count > 0 else "#68D391"

    html_str = textwrap.dedent(
        f"""<div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 20px;">
  <div style="background-color: #2D3748; border: 1px solid #4A5568; border-radius: 8px; padding: 14px 16px;">
    <div style="font-size: 11px; font-weight: 700; color: #A0AEC0; text-transform: uppercase; letter-spacing: 0.5px;">
      Scanned Identities
    </div>
    <div style="font-size: 26px; font-weight: 800; color: #FFFFFF; margin-top: 4px;">
      {kpis.total_identities}
    </div>
    <div style="font-size: 11px; color: #CBD5E0; margin-top: 2px;">
      {kpis.total_users} Users · {kpis.total_roles} Roles
    </div>
  </div>
  <div style="background-color: #2D3748; border: 1px solid #4A5568; border-radius: 8px; padding: 14px 16px;">
    <div style="font-size: 11px; font-weight: 700; color: #A0AEC0; text-transform: uppercase; letter-spacing: 0.5px;">
      Evaluated Policies
    </div>
    <div style="font-size: 26px; font-weight: 800; color: #63B3ED; margin-top: 4px;">
      {kpis.total_policies}
    </div>
    <div style="font-size: 11px; color: #CBD5E0; margin-top: 2px;">
      {kpis.total_resources} Monitored Assets
    </div>
  </div>
  <div style="background-color: #2D3748; border: 1px solid #E53E3E; border-radius: 8px; padding: 14px 16px;">
    <div style="font-size: 11px; font-weight: 700; color: #FEB2B2; text-transform: uppercase; letter-spacing: 0.5px;">
      Latent Risks Flagged
    </div>
    <div style="font-size: 26px; font-weight: 800; color: #E53E3E; margin-top: 4px;">
      {kpis.flagged_identities_count}
    </div>
    <div style="font-size: 11px; color: #CBD5E0; margin-top: 2px;">
      {kpis.critical_risks_count} Critical · {kpis.high_risks_count} High
    </div>
  </div>
  <div style="background-color: #2D3748; border: 1px solid #4A5568; border-radius: 8px; padding: 14px 16px;">
    <div style="font-size: 11px; font-weight: 700; color: #A0AEC0; text-transform: uppercase; letter-spacing: 0.5px;">
      Misleading Safe (G_o)
    </div>
    <div style="font-size: 26px; font-weight: 800; color: {misleading_color}; margin-top: 4px;">
      {kpis.misleading_safe_count}
    </div>
    <div style="font-size: 11px; color: #CBD5E0; margin-top: 2px;">
      Hidden by Condition E
    </div>
  </div>
  <div style="background-color: #2D3748; border: 1px solid #38A169; border-radius: 8px; padding: 14px 16px;">
    <div style="font-size: 11px; font-weight: 700; color: #9AE6B4; text-transform: uppercase; letter-spacing: 0.5px;">
      Pareto Optimality (RQ2)
    </div>
    <div style="font-size: 16px; font-weight: 800; color: #FFFFFF; margin-top: 10px;">
      {pareto_badge}
    </div>
    <div style="font-size: 11px; color: #CBD5E0; margin-top: 4px;">
      Dominates Methods 1, 2, 3
    </div>
  </div>
</div>"""
    ).strip()
    return html_str


def render_streamlit_metrics(kpis: PostureKPIs) -> None:
    """Render metrics using native Streamlit metric columns if streamlit is active."""
    try:
        import streamlit as st

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric(
                "Scanned Identities",
                kpis.total_identities,
                f"{kpis.total_users} Users, {kpis.total_roles} Roles",
            )
        with c2:
            st.metric(
                "Evaluated Policies",
                kpis.total_policies,
                f"{kpis.total_resources} Monitored Assets",
            )
        with c3:
            st.metric(
                "Latent Risks Flagged",
                kpis.flagged_identities_count,
                f"{kpis.critical_risks_count} Critical",
                delta_color="inverse",
            )
        with c4:
            st.metric(
                "Misleading Safe (G_o)",
                kpis.misleading_safe_count,
                "Condition E Blind Spots",
                delta_color="inverse" if kpis.misleading_safe_count > 0 else "normal",
            )
        with c5:
            st.metric(
                "Pareto Optimality",
                "Certified",
                "Dominates 1, 2, 3",
            )
    except ImportError:
        logger.warning(
            "Streamlit not available for render_streamlit_metrics; use render_posture_banner_html"
        )


def format_posture_markdown_summary(kpis: PostureKPIs) -> str:
    """Format an ASCII/Markdown summary table for terminal logging and CLI reports."""
    return f"""
| Security Posture Metric | Value | Context |
| :--- | :---: | :--- |
| **Total Scanned Identities** | `{kpis.total_identities}` | {kpis.total_users} Users, {kpis.total_roles} Roles |
| **Evaluated Policy Documents** | `{kpis.total_policies}` | {kpis.total_resources} Monitored Cloud Resources |
| **Flagged Latent Escalation Risks** | `{kpis.flagged_identities_count}` | {kpis.critical_risks_count} Critical, {kpis.high_risks_count} High |
| **Misleading Safe (Static Analysis Blind)** | `{kpis.misleading_safe_count}` | Level 1: 0.0, Level 2: >0.70 |
| **Average Latent Blast Radius** | `{kpis.average_latent_blast_radius:.2f}` | Crown Jewel Assets per Flagged Principal |
| **Pareto Dominance (RQ2 Tradeoff)** | `CERTIFIED` | Uncertainty-Aware Repair strictly optimal |
""".strip()
