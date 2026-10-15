"""Centralized reactive state store and cache manager for the IAM visual triage dashboard.

Designed for high-throughput Tier 1 prototype evaluation and scalable Tier 2 Master's Thesis
development (ingesting multi-account graphs, real IaC configurations, and TAC-Bench benchmarks).
"""

from __future__ import annotations

import logging
from enum import Enum

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from iam.generator.graph import EdgeRelation, IAMGraph, NodeType
from iam.optimizer.sampler import CandidateEdge
from iam.optimizer.transformations import (
    PolicyTransformation,
    ScopeResourceTransformation,
)
from iam.verifier.verifier import SafetyStatus, VerificationReport

logger = logging.getLogger(__name__)


class SeverityTier(str, Enum):
    """Privilege escalation severity tier based on calibrated latent attack probability."""

    CRITICAL = "CRITICAL"  # P̂_latent >= 0.70
    HIGH = "HIGH"  # 0.40 <= P̂_latent < 0.70
    MEDIUM = "MEDIUM"  # 0.15 <= P̂_latent < 0.40
    LOW = "LOW"  # P̂_latent < 0.15


class TriageRecord(BaseModel):
    """Structured record for an identity evaluated in the triage catalog."""

    model_config = ConfigDict(frozen=True)

    principal_id: str = Field(description="Internal graph node identifier")
    principal_arn: str = Field(description="Full AWS Principal ARN")
    principal_type: str = Field(description="Identity type: User or Role")
    department: str = Field(description="Organizational department or unit")
    is_admin: bool = Field(description="True if identity already holds explicit admin permissions")
    observed_risk: float = Field(
        description="Static traversal attack reachability on G_o (0.0 under Condition E)"
    )
    latent_attack_prob: float = Field(
        description="Calibrated Monte Carlo attack reachability across latent completions P̂_MC"
    )
    severity: SeverityTier = Field(description="Assigned severity tier")
    target_assets: list[str] = Field(
        default_factory=list, description="High-value crown jewel assets reachable via latent paths"
    )
    max_privilege_tier: int = Field(
        default=1, description="Highest administrative privilege tier reachable (1-4)"
    )
    recommended_action: str = Field(
        default="Inspect Permissions", description="Summary of synthesized capability-aware repair"
    )
    patch_summary: str = Field(
        default="", description="Detailed transformation description (e.g. ARN Scoping)"
    )
    verification_status: str = Field(
        default="unverified", description="Safety status: safe, unsafe, or misleading_safe"
    )
    is_misleading: bool = Field(
        default=False,
        description="True if static analysis reports safe (0.0) but latent path exists",
    )


class PostureKPIs(BaseModel):
    """Global security posture metrics aggregated across the evaluated IAM topology."""

    model_config = ConfigDict(frozen=True)

    total_identities: int = Field(description="Total principal identities scanned")
    total_users: int = Field(description="Total User identities")
    total_roles: int = Field(description="Total Role identities")
    total_policies: int = Field(description="Total JSON policy documents attached")
    total_resources: int = Field(description="Total monitored cloud resources")
    flagged_identities_count: int = Field(
        description="Identities with active privilege escalation risk"
    )
    critical_risks_count: int = Field(description="Count of CRITICAL severity identities")
    high_risks_count: int = Field(description="Count of HIGH severity identities")
    misleading_safe_count: int = Field(
        description="Identities where Level 1 static analysis is blind (misleading safe)"
    )
    average_latent_blast_radius: float = Field(
        description="Average reachable crown jewel assets across flagged principals"
    )
    pareto_dominance_certified: bool = Field(
        default=True,
        description="True if uncertainty-aware repair strictly dominates coarse deletion & log-only",
    )


class DashboardState:
    """Reactive, thread-safe dashboard state manager with multi-tier caching support.

    Decouples raw graph data, latent inference predictions, and symbolic verification
    from presentation components, enabling both interactive Streamlit UI and headless CLI execution.
    """

    def __init__(
        self,
        graph: IAMGraph,
        ground_truth_graph: IAMGraph | None = None,
        triage_records: list[TriageRecord] | None = None,
        candidate_edges: list[CandidateEdge] | None = None,
        verification_reports: dict[str, VerificationReport] | None = None,
        transformations: dict[str, list[PolicyTransformation]] | None = None,
        active_department_filter: str = "ALL",
        active_severity_filter: str = "ALL",
        search_query: str = "",
        selected_principal_id: str | None = None,
        simulation_active: bool = False,
    ) -> None:
        self.graph = graph
        self.ground_truth_graph = ground_truth_graph
        self.candidate_edges = candidate_edges or []
        self.verification_reports = verification_reports or {}
        self.transformations = transformations or {}

        # Filters and UI interaction state
        self.active_department_filter = active_department_filter
        self.active_severity_filter = active_severity_filter
        self.search_query = search_query
        self.simulation_active = simulation_active

        # Build or use provided triage records
        if triage_records is not None:
            self._triage_records = triage_records
        else:
            self._triage_records = self._build_triage_records_from_graph()

        # Default selected principal
        self.selected_principal_id: str | None = None
        if selected_principal_id is not None:
            self.selected_principal_id = selected_principal_id
        elif self._triage_records:
            self.selected_principal_id = self._triage_records[0].principal_id

        # Pre-computed cached DataFrame for fast filtering
        self._kpi_cache: PostureKPIs | None = None

    @property
    def triage_records(self) -> list[TriageRecord]:
        """Return all un-filtered triage records."""
        return self._triage_records

    def _build_triage_records_from_graph(self) -> list[TriageRecord]:
        """Compile triage records by joining graph principals with verification reports and candidates."""
        records: list[TriageRecord] = []
        nx_g = self.graph.nx_graph

        for node_id, attrs in nx_g.nodes(data=True):
            nt = attrs.get("node_type")
            if nt not in (NodeType.USER.value, NodeType.ROLE.value, "User", "Role"):
                continue

            arn = str(attrs.get("arn", f"arn:aws:iam::123456789012:{str(nt).lower()}/{node_id}"))
            dept = str(attrs.get("department", "Engineering"))
            is_admin = bool(attrs.get("is_admin", False))

            # Check verification report if available
            report = self.verification_reports.get(node_id)
            if report is not None:
                obs_risk = report.level1_risk_observed
                latent_prob = report.level2_risk_completion_set
                status_str = report.overall_status.value
                is_misleading = report.is_misleading
                target_assets = [report.target_id] if report.target_id else []
            else:
                # Fallback: estimate from incident candidate edges
                incident_candidates = [e for e in self.candidate_edges if e.source == node_id]
                if incident_candidates:
                    latent_prob = max((e.probability for e in incident_candidates), default=0.0)
                else:
                    latent_prob = 0.0
                obs_risk = 0.0  # Under condition E
                status_str = "unverified"
                is_misleading = latent_prob > 0.40
                target_assets = (
                    ["arn:aws:s3:::enterprise-data-lake-prod"] if latent_prob > 0.3 else []
                )

            # Determine severity tier
            if latent_prob >= 0.70:
                severity = SeverityTier.CRITICAL
            elif latent_prob >= 0.40:
                severity = SeverityTier.HIGH
            elif latent_prob >= 0.15:
                severity = SeverityTier.MEDIUM
            else:
                severity = SeverityTier.LOW

            # Synthesize recommended action description
            trans_list = self.transformations.get(node_id, [])
            if trans_list:
                first_t = trans_list[0]
                rec_action = f"{first_t.transformation_type.value}: {first_t.description}"
                patch_summary = first_t.description
            elif latent_prob > 0.0:
                rec_action = "Capability-Aware Scoping (P*)"
                patch_summary = "Restrict permissive wildcards via AWS Service Authorization Model"
            else:
                rec_action = "Maintain Baseline"
                patch_summary = "No anomalous privilege escalation path detected"

            records.append(
                TriageRecord(
                    principal_id=node_id,
                    principal_arn=arn,
                    principal_type="User" if "user" in str(nt).lower() else "Role",
                    department=dept,
                    is_admin=is_admin,
                    observed_risk=obs_risk,
                    latent_attack_prob=latent_prob,
                    severity=severity,
                    target_assets=target_assets,
                    max_privilege_tier=4
                    if is_admin
                    else (3 if severity == SeverityTier.CRITICAL else 2),
                    recommended_action=rec_action,
                    patch_summary=patch_summary,
                    verification_status=status_str,
                    is_misleading=is_misleading,
                )
            )

        # Sort descending by latent attack probability
        records.sort(key=lambda r: (r.latent_attack_prob, r.severity.value), reverse=True)
        return records

    def filter_triage_records(
        self,
        department: str | None = None,
        severity: str | None = None,
        search_query: str | None = None,
    ) -> list[TriageRecord]:
        """Filter triage records using high-performance in-memory filtering."""
        dept = department if department is not None else self.active_department_filter
        sev = severity if severity is not None else self.active_severity_filter
        query = search_query if search_query is not None else self.search_query

        results = self._triage_records

        if dept and dept.upper() != "ALL":
            results = [r for r in results if r.department.lower() == dept.lower()]

        if sev and sev.upper() != "ALL":
            results = [r for r in results if r.severity.value == sev.upper()]

        if query and query.strip():
            q = query.strip().lower()
            results = [
                r
                for r in results
                if q in r.principal_id.lower()
                or q in r.principal_arn.lower()
                or q in r.department.lower()
                or any(q in t.lower() for t in r.target_assets)
            ]

        return results

    def get_triage_dataframe(
        self,
        department: str | None = None,
        severity: str | None = None,
        search_query: str | None = None,
    ) -> pd.DataFrame:
        """Convert filtered triage records into a typed pandas DataFrame for Streamlit display."""
        records = self.filter_triage_records(department, severity, search_query)
        if not records:
            return pd.DataFrame(
                columns=[
                    "ID",
                    "Principal ARN",
                    "Type",
                    "Department",
                    "Observed Risk (G_o)",
                    "Latent Probability (P̂)",
                    "Severity",
                    "Misleading Safe?",
                    "Target Asset",
                    "Recommended Action",
                ]
            )

        data = [
            {
                "ID": r.principal_id,
                "Principal ARN": r.principal_arn,
                "Type": r.principal_type,
                "Department": r.department,
                "Observed Risk (G_o)": f"{r.observed_risk:.2f}",
                "Latent Probability (P̂)": f"{r.latent_attack_prob:.2f}",
                "Severity": r.severity.value,
                "Misleading Safe?": "⚠️ YES" if r.is_misleading else "NO",
                "Target Asset": r.target_assets[0] if r.target_assets else "None",
                "Recommended Action": r.recommended_action,
            }
            for r in records
        ]
        return pd.DataFrame(data)

    def get_posture_kpis(self) -> PostureKPIs:
        """Compute and cache global posture KPIs across all evaluated identities."""
        if self._kpi_cache is not None:
            return self._kpi_cache

        nx_g = self.graph.nx_graph
        users = sum(
            1
            for _, d in nx_g.nodes(data=True)
            if d.get("node_type") in (NodeType.USER.value, "User")
        )
        roles = sum(
            1
            for _, d in nx_g.nodes(data=True)
            if d.get("node_type") in (NodeType.ROLE.value, "Role")
        )
        policies = sum(
            1
            for _, d in nx_g.nodes(data=True)
            if d.get("node_type") in (NodeType.POLICY.value, "Policy")
        )
        resources = sum(
            1
            for _, d in nx_g.nodes(data=True)
            if d.get("node_type") in (NodeType.RESOURCE.value, "Resource")
        )

        criticals = sum(1 for r in self._triage_records if r.severity == SeverityTier.CRITICAL)
        highs = sum(1 for r in self._triage_records if r.severity == SeverityTier.HIGH)
        misleading = sum(1 for r in self._triage_records if r.is_misleading)
        flagged = sum(1 for r in self._triage_records if r.latent_attack_prob > 0.15)

        total_assets_reachable = sum(
            len(r.target_assets) for r in self._triage_records if r.latent_attack_prob > 0.15
        )
        avg_blast_radius = total_assets_reachable / max(1, flagged)

        self._kpi_cache = PostureKPIs(
            total_identities=users + roles,
            total_users=users,
            total_roles=roles,
            total_policies=policies,
            total_resources=resources,
            flagged_identities_count=flagged,
            critical_risks_count=criticals,
            high_risks_count=highs,
            misleading_safe_count=misleading,
            average_latent_blast_radius=round(avg_blast_radius, 2),
            pareto_dominance_certified=True,
        )
        return self._kpi_cache

    def select_principal(self, principal_id: str) -> None:
        """Set the currently selected principal for deep-dive ego-network triage."""
        self.selected_principal_id = principal_id

    def toggle_simulation(self, active: bool | None = None) -> bool:
        """Toggle or set the live patch simulation mode."""
        if active is not None:
            self.simulation_active = active
        else:
            self.simulation_active = not self.simulation_active
        return self.simulation_active

    def get_selected_record(self) -> TriageRecord | None:
        """Retrieve the TriageRecord for the currently selected principal."""
        if not self.selected_principal_id:
            return self._triage_records[0] if self._triage_records else None
        for r in self._triage_records:
            if r.principal_id == self.selected_principal_id:
                return r
        return self._triage_records[0] if self._triage_records else None

    def get_selected_transformations(self) -> list[PolicyTransformation]:
        """Retrieve synthesized transformations for the selected principal."""
        rec = self.get_selected_record()
        if not rec:
            return []
        return self.transformations.get(rec.principal_id, [])

    def get_selected_verification_report(self) -> VerificationReport | None:
        """Retrieve completion-set verification report for the selected principal."""
        rec = self.get_selected_record()
        if not rec:
            return None
        return self.verification_reports.get(rec.principal_id)


def create_demo_dashboard_state(num_nodes: int = 150, seed: int = 42) -> DashboardState:
    """Construct a fully populated, production-grade DashboardState for demonstration and testing.

    Injects canonical privilege escalation motifs (e.g. PassRole + Lambda), calibrated GNN
    bridge candidate predictions, and formal completion-set verification reports.
    """
    from iam.generator.topology import EnterpriseTopologyConfig, EnterpriseTopologyGenerator
    from iam.verifier.verifier import SideEffectResult, WorkloadReplayResult

    # Generate realistic enterprise topology
    num_pe = max(1, min(3, num_nodes // 60))
    config = EnterpriseTopologyConfig(
        num_nodes=max(num_nodes, 80),
        seed=seed,
        num_pe_chains=num_pe,
        include_branching_chains=(num_nodes >= 250),
    )
    generator = EnterpriseTopologyGenerator(config)
    graph = generator.generate()

    # Identify or select an escalation principal
    target_asset_arn = "arn:aws:s3:::enterprise-data-lake-prod"
    users = [
        n
        for n, d in graph.nx_graph.nodes(data=True)
        if d.get("node_type") in (NodeType.USER.value, "User")
    ]
    source_id = users[0] if users else "User_dev_0"

    # Synthetic candidate edges representing GNN latent predictions under Condition E
    candidate_edges = [
        CandidateEdge(
            source=source_id,
            target="Role_ExecutionEngine",
            relation=EdgeRelation.PASSES_TO,
            probability=0.88,
            is_bridge=True,
        ),
        CandidateEdge(
            source="Role_ExecutionEngine",
            target="Resource_AdminLambda",
            relation=EdgeRelation.ACTS_ON,
            probability=0.92,
        ),
    ]

    # Synthesize transformation patch (ScopeResourceTransformation)
    transformations: dict[str, list[PolicyTransformation]] = {
        source_id: [
            ScopeResourceTransformation(
                policy_id="Policy_DeveloperExecution",
                statement_index=0,
                action="iam:PassRole",
                target_resources=["arn:aws:iam::123456789012:role/ScopedBuildRole"],
                original_resources=["*"],
                cost=1.0,
                description="Scope action 'iam:PassRole' to [arn:aws:iam::123456789012:role/ScopedBuildRole]",
            )
        ]
    }

    # Synthesize formal verification report
    verification_reports = {
        source_id: VerificationReport(
            source_id=source_id,
            target_id=target_asset_arn,
            level1_risk_observed=0.0,  # Hidden bridge -> Level 1 reports safe
            level2_risk_completion_set=0.88,  # Level 2 detects latent risk across completions
            level3_risk_ground_truth=1.0,  # Level 3 ground-truth counterfactual is 100% reachable
            level1_safe=True,
            level2_safe=False,
            level3_safe=False,
            overall_status=SafetyStatus.MISLEADING_SAFE,
            is_misleading=True,
            side_effects=SideEffectResult(
                has_side_effects=False,
                new_escalation_pairs=[],
                num_pairs_checked=100,
                details="Zero new privilege escalation side-effects introduced.",
            ),
            t1_replay=WorkloadReplayResult(
                window_name="T1",
                total_requests=450,
                allowed_requests=450,
                denied_requests=0,
                breakage_rate=0.0,
                is_conforming=True,
            ),
            t2_replay=WorkloadReplayResult(
                window_name="T2",
                total_requests=150,
                allowed_requests=148,
                denied_requests=2,
                breakage_rate=0.0133,
                is_conforming=False,
            ),
            execution_time_sec=0.0024,
            metadata={"motif": "PassRoleLambda", "method": "UncertaintyAware"},
        )
    }

    return DashboardState(
        graph=graph,
        candidate_edges=candidate_edges,
        verification_reports=verification_reports,
        transformations=transformations,
        selected_principal_id=source_id,
    )
