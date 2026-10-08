"""Parameterized enterprise cloud topology generator.

Simulates enterprise AWS cloud IAM topologies with realistic departmental boundaries,
power-law identity distributions, authentic benign business workflows, and deeply
embedded canonical and branching privilege escalation chains.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

import networkx as nx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from iam.generator.graph import (
    EdgeRelation,
    GraphEdge,
    GraphNode,
    IAMGraph,
    NodeType,
)
from iam.generator.motifs import (
    MotifInstance,
    MotifRegistry,
    default_motif_registry,
)
from iam.parser.capability import CapabilityModel, load_default_capability_model
from iam.parser.schema import Effect, PolicyDocument, Principal, Statement


class DepartmentType(str, Enum):
    """Canonical enterprise departments representing realistic administrative boundaries."""

    SECOPS = "SecOps"
    DEVOPS = "DevOps"
    DATABI = "DataBI"
    QA = "QA"
    BILLING = "Billing"
    INTERNS = "Interns"


class PrivilegeTier(str, Enum):
    """Privilege tiers for modeling power-law identity access distributions."""

    ADMIN = "admin"
    STANDARD = "standard"
    RESTRICTED = "restricted"


class BranchingType(str, Enum):
    """Structural typology of branching privilege escalation paths."""

    CONVERGENT_DIAMOND = "convergent_diamond"  # Multiple distinct paths to same target
    MULTI_TARGET = "multi_target"  # Single pivot role escalating to multiple targets


class BranchingVerification(BaseModel):
    """Result of formal reachability and counterfactual bridge verification for branching chains."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    is_valid: bool = Field(description="True if all branching graph properties hold")
    path_exists_full: bool = Field(description="True if path from source to target exists")
    single_bridge_leaves_alternate: bool = Field(
        description="True if removing a single bridge edge leaves an alternate active path"
    )
    all_bridges_break_reachability: bool = Field(
        description="True if removing all bridges breaks reachability"
    )
    details: str = Field(default="", description="Diagnostic details")


class BranchingPEChain(BaseModel):
    """A multi-path branching Privilege Escalation topology."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chain_id: str = Field(description="Unique identifier for this branching chain")
    branching_type: BranchingType = Field(description="Branching topology type")
    source_id: str = Field(description="Attacker entrypoint principal node ID")
    target_ids: list[str] = Field(description="Target high-value asset node IDs")
    branch_instances: list[MotifInstance] = Field(
        default_factory=list, description="Sub-motif instances forming the branches"
    )
    bridge_relations: list[tuple[str, str, str]] = Field(
        default_factory=list, description="All critical bottleneck relations (u, v, relation)"
    )
    all_path_edges: list[tuple[str, str, str]] = Field(
        default_factory=list, description="Union of all directed edges along the branches"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata and parameters")

    def verify(self, graph: IAMGraph) -> BranchingVerification:
        """Formally verify multi-path reachability and counterfactual bridge masking."""
        nx_g = graph.nx_graph
        s = self.source_id

        if self.branching_type == BranchingType.CONVERGENT_DIAMOND:
            t = self.target_ids[0]
            # 1. Full path exists
            path_full = nx.has_path(nx_g, s, t)
            if not path_full:
                return BranchingVerification(
                    is_valid=False,
                    path_exists_full=False,
                    single_bridge_leaves_alternate=False,
                    all_bridges_break_reachability=False,
                    details=f"No path from {s} to {t} in full graph.",
                )

            # 2. Check each bridge removal individually leaves an alternative path
            single_leaves_alt = True
            edge_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
            for bu, bv, brel in self.bridge_relations:
                if nx_g.has_edge(bu, bv, key=brel):
                    edge_cache[(bu, bv, brel)] = nx_g.get_edge_data(bu, bv, key=brel)
                    nx_g.remove_edge(bu, bv, key=brel)
                    if not nx.has_path(nx_g, s, t):
                        single_leaves_alt = False
                    nx_g.add_edge(bu, bv, key=brel, **edge_cache[(bu, bv, brel)])

            # 3. Check removing ALL bridges simultaneously breaks reachability
            for bu, bv, brel in self.bridge_relations:
                if nx_g.has_edge(bu, bv, key=brel):
                    nx_g.remove_edge(bu, bv, key=brel)

            all_break = not nx.has_path(nx_g, s, t)

            # Restore all edges
            for (bu, bv, brel), data in edge_cache.items():
                nx_g.add_edge(bu, bv, key=brel, **data)

            is_valid = path_full and single_leaves_alt and all_break
            diag = (
                f"Diamond branching '{self.chain_id}': valid={is_valid}, "
                f"path_full={path_full}, single_leaves_alt={single_leaves_alt}, all_break={all_break}"
            )
            return BranchingVerification(
                is_valid=is_valid,
                path_exists_full=path_full,
                single_bridge_leaves_alternate=single_leaves_alt,
                all_bridges_break_reachability=all_break,
                details=diag,
            )

        else:  # MULTI_TARGET
            t1, t2 = self.target_ids[0], self.target_ids[1]
            p1_full = nx.has_path(nx_g, s, t1)
            p2_full = nx.has_path(nx_g, s, t2)
            if not (p1_full and p2_full):
                return BranchingVerification(
                    is_valid=False,
                    path_exists_full=False,
                    single_bridge_leaves_alternate=False,
                    all_bridges_break_reachability=False,
                    details=f"Missing paths in full graph: p1={p1_full}, p2={p2_full}",
                )

            # Remove primary bridge
            bu, bv, brel = self.bridge_relations[0]
            edge_data = nx_g.get_edge_data(bu, bv, key=brel)
            nx_g.remove_edge(bu, bv, key=brel)
            breaks_both = (not nx.has_path(nx_g, s, t1)) and (not nx.has_path(nx_g, s, t2))
            nx_g.add_edge(bu, bv, key=brel, **edge_data)

            is_valid = p1_full and p2_full and breaks_both
            diag = (
                f"Multi-target branching '{self.chain_id}': valid={is_valid}, "
                f"p1_full={p1_full}, p2_full={p2_full}, breaks_both={breaks_both}"
            )
            return BranchingVerification(
                is_valid=is_valid,
                path_exists_full=True,
                single_bridge_leaves_alternate=True,
                all_bridges_break_reachability=breaks_both,
                details=diag,
            )


class EnterpriseTopologyConfig(BaseModel):
    """Configuration specification for parameterized enterprise cloud generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_nodes: int = Field(
        default=1000, ge=30, le=10000, description="Target total node count in the graph"
    )
    seed: int | None = Field(
        default=42, description="Random seed for deterministic graph generation"
    )
    account_id: str = Field(default="123456789012", description="AWS 12-digit account ID")
    organization_id: str = Field(
        default="o-enterprise-001", description="AWS Organization ID for dataset splits"
    )
    departments: list[str] = Field(
        default_factory=lambda: [
            DepartmentType.SECOPS.value,
            DepartmentType.DEVOPS.value,
            DepartmentType.DATABI.value,
            DepartmentType.QA.value,
            DepartmentType.BILLING.value,
            DepartmentType.INTERNS.value,
        ],
        description="Departmental organizational units",
    )
    department_weights: dict[str, float] | None = Field(
        default=None,
        description="Relative node weights across departments (power-law defaults if None)",
    )
    edge_density: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Edge density multiplier for connectivity"
    )
    num_pe_chains: int = Field(
        default=5, ge=0, description="Number of canonical PE chains to inject"
    )
    pe_chain_lengths: list[int] = Field(
        default_factory=lambda: [2, 3, 4],
        description="Path hop lengths (2-hop, 3-hop, 4-hop) to inject",
    )
    pe_motif_types: list[str] | None = Field(
        default=None,
        description="Optional whitelist of specific PE motif types to sample from",
    )
    include_branching_chains: bool = Field(
        default=True, description="Whether to inject branching PE topologies"
    )
    num_branching_chains: int = Field(
        default=1, ge=0, description="Number of branching PE topologies to inject"
    )
    power_law_alpha: float = Field(
        default=1.8, ge=1.0, description="Power-law parameter for identity distribution"
    )


class EnterpriseTopologyGenerator:
    """Scalable generator synthesizing realistic enterprise AWS IAM authorization topologies.

    Key features:
    - Modular departmental subgraphs with realistic boundary constraints.
    - Power-law identity distribution (small admin core, heavy restricted tail).
    - Authentic, capability-compliant benign business workflows.
    - Vectorized edge generation for fast scaling up to 5,000+ nodes.
    - Embedded canonical (2-hop, 3-hop, 4-hop) and branching privilege escalation chains.
    - Deterministic seeding for exact reproducibility across environment splits.
    """

    DEFAULT_DEPT_WEIGHTS: dict[str, float] = {
        DepartmentType.SECOPS.value: 0.05,
        DepartmentType.DEVOPS.value: 0.20,
        DepartmentType.DATABI.value: 0.25,
        DepartmentType.QA.value: 0.20,
        DepartmentType.BILLING.value: 0.10,
        DepartmentType.INTERNS.value: 0.20,
    }

    # Department privilege distribution: (admin_ratio, standard_ratio, restricted_ratio)
    DEPT_TIER_PROPORTIONS: dict[str, tuple[float, float, float]] = {
        DepartmentType.SECOPS.value: (0.30, 0.40, 0.30),
        DepartmentType.DEVOPS.value: (0.05, 0.60, 0.35),
        DepartmentType.DATABI.value: (0.01, 0.25, 0.74),
        DepartmentType.QA.value: (0.00, 0.30, 0.70),
        DepartmentType.BILLING.value: (0.00, 0.15, 0.85),
        DepartmentType.INTERNS.value: (0.00, 0.05, 0.95),
    }

    def __init__(
        self,
        config: EnterpriseTopologyConfig | None = None,
        motif_registry: MotifRegistry | None = None,
        capability_model: CapabilityModel | None = None,
    ) -> None:
        self.config = config or EnterpriseTopologyConfig()
        self.motif_registry = motif_registry or default_motif_registry
        self.capability_model = capability_model or load_default_capability_model()

    def generate(self) -> IAMGraph:
        """Synthesize the complete enterprise authorization topology."""
        start_time = time.perf_counter()
        rng = np.random.default_rng(self.config.seed)

        graph = IAMGraph(
            graph_id=f"iam_org_{self.config.organization_id}",
            metadata={
                "organization_id": self.config.organization_id,
                "account_id": self.config.account_id,
                "seed": self.config.seed,
                "config": self.config.model_dump(),
            },
        )

        # 1. Allocate node quotas across departments
        dept_node_counts = self._allocate_department_node_counts(rng)

        # 2. Synthesize department nodes (Users, Roles, Groups, Policies, Resources)
        dept_nodes_by_dept: dict[str, dict[NodeType, list[GraphNode]]] = {}
        for dept, count in dept_node_counts.items():
            type_counts = self._allocate_node_types_in_dept(dept, count, rng)
            nodes = self._create_department_nodes(dept, type_counts, rng, graph)
            dept_nodes_by_dept[dept] = nodes

        # 3. Vectorized intra-department edge generation
        for dept, dept_nodes in dept_nodes_by_dept.items():
            self._generate_intra_department_edges(dept, dept_nodes, rng, graph)

        # 4. Synthesize specific authentic benign business workflows
        self._generate_benign_business_workflows(dept_nodes_by_dept, rng, graph)

        # 5. Controlled inter-department role assumption & trust edges
        self._generate_inter_department_trust(dept_nodes_by_dept, rng, graph)

        # 6. Inject canonical PE chains (2-hop, 3-hop, 4-hop)
        injected_motifs, used_sources = self._inject_canonical_pe_chains(graph, rng)

        # 7. Inject branching PE chains (convergent diamond / multi-target)
        branching_chains: list[BranchingPEChain] = []
        if self.config.include_branching_chains and self.config.num_branching_chains > 0:
            branching_chains = self._inject_branching_pe_chains(graph, rng, used_sources)

        # 8. Compute graph structural metrics and record metadata
        elapsed_sec = time.perf_counter() - start_time
        stats = self._compute_graph_stats(graph)
        stats["generation_time_sec"] = elapsed_sec

        graph.metadata["stats"] = stats
        graph.metadata["injected_motifs"] = [m.model_dump() for m in injected_motifs]
        graph.metadata["branching_chains"] = [bc.model_dump() for bc in branching_chains]

        return graph

    def _allocate_department_node_counts(self, rng: np.random.Generator) -> dict[str, int]:
        """Allocate total node quota across departments respecting power-law proportions."""
        depts = self.config.departments
        weights: list[float] = []

        if self.config.department_weights:
            weights = [self.config.department_weights.get(d, 0.1) for d in depts]
        else:
            weights = [self.DEFAULT_DEPT_WEIGHTS.get(d, 0.1) for d in depts]

        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]

        # Allocate minimum 5 nodes per department to satisfy structural variety
        min_nodes = 5
        remaining_nodes = max(0, self.config.num_nodes - min_nodes * len(depts))

        allocations = dict.fromkeys(depts, min_nodes)
        additional = rng.multinomial(remaining_nodes, norm_weights)
        for i, d in enumerate(depts):
            allocations[d] += int(additional[i])

        return allocations

    def _allocate_node_types_in_dept(
        self,
        _dept: str,
        total_nodes: int,
        rng: np.random.Generator,
    ) -> dict[NodeType, int]:
        """Break down department node allocation into typed entities."""
        # Ensure at least 1 of each type
        counts: dict[NodeType, int] = {
            NodeType.GROUP: 1,
            NodeType.POLICY: 1,
            NodeType.RESOURCE: 1,
            NodeType.ROLE: 1,
            NodeType.USER: 1,
        }
        remaining = total_nodes - 5
        if remaining <= 0:
            return counts

        # Proportions: Users: 38%, Roles: 24%, Groups: 6%, Policies: 18%, Resources: 14%
        type_weights = [0.38, 0.24, 0.06, 0.18, 0.14]
        types_order = [
            NodeType.USER,
            NodeType.ROLE,
            NodeType.GROUP,
            NodeType.POLICY,
            NodeType.RESOURCE,
        ]

        assigned = rng.multinomial(remaining, type_weights)
        for t, add_count in zip(types_order, assigned, strict=False):
            counts[t] += int(add_count)

        return counts

    def _create_department_nodes(
        self,
        dept: str,
        type_counts: dict[NodeType, int],
        rng: np.random.Generator,
        graph: IAMGraph,
    ) -> dict[NodeType, list[GraphNode]]:
        """Create typed GraphNode entities for a department with valid capability policies."""
        dept_slug = dept.lower().replace("/", "-")
        account_id = self.config.account_id
        nodes_by_type: dict[NodeType, list[GraphNode]] = {t: [] for t in NodeType}

        # 1. Groups
        for i in range(type_counts[NodeType.GROUP]):
            gid = f"group:{dept_slug}-team-{i}"
            g_arn = f"arn:aws:iam::{account_id}:group/{dept_slug}-team-{i}"
            node = GraphNode(
                id=gid,
                node_type=NodeType.GROUP,
                arn=g_arn,
                name=f"{dept_slug}-team-{i}",
                account_id=account_id,
                department=dept,
            )
            graph.add_node(node)
            nodes_by_type[NodeType.GROUP].append(node)

        # 2. Resources
        is_secops = dept == DepartmentType.SECOPS.value
        for i in range(type_counts[NodeType.RESOURCE]):
            rid, r_arn, is_hv = self._generate_resource_attributes(dept, dept_slug, i, is_secops)
            node = GraphNode(
                id=rid,
                node_type=NodeType.RESOURCE,
                arn=r_arn,
                name=f"{dept_slug}-resource-{i}",
                account_id=account_id,
                department=dept,
                is_high_value=is_hv,
            )
            graph.add_node(node)
            nodes_by_type[NodeType.RESOURCE].append(node)

        # 3. Policies
        res_nodes = nodes_by_type[NodeType.RESOURCE]
        for i in range(type_counts[NodeType.POLICY]):
            pid = f"policy:{dept_slug}-policy-{i}"
            p_arn = f"arn:aws:iam::{account_id}:policy/{dept_slug}-policy-{i}"
            policy_doc, is_adm = self._synthesize_department_policy(dept, i, res_nodes)
            node = GraphNode(
                id=pid,
                node_type=NodeType.POLICY,
                arn=p_arn,
                name=f"{dept_slug}-policy-{i}",
                account_id=account_id,
                department=dept,
                is_admin=is_adm,
                policy_document=policy_doc,
            )
            graph.add_node(node)
            nodes_by_type[NodeType.POLICY].append(node)

        # 4. Users (power-law privilege distribution)
        u_tiers = self._sample_privilege_tiers(dept, type_counts[NodeType.USER], rng)
        for i, tier in enumerate(u_tiers):
            uid = f"user:{dept_slug}-{tier.value}-{i}"
            u_arn = f"arn:aws:iam::{account_id}:user/{dept_slug}-{tier.value}-{i}"
            node = GraphNode(
                id=uid,
                node_type=NodeType.USER,
                arn=u_arn,
                name=f"{dept_slug}-{tier.value}-{i}",
                account_id=account_id,
                department=dept,
                is_admin=(tier == PrivilegeTier.ADMIN),
                metadata={"privilege_tier": tier.value},
            )
            graph.add_node(node)
            nodes_by_type[NodeType.USER].append(node)

        # 5. Roles (power-law privilege distribution)
        r_tiers = self._sample_privilege_tiers(dept, type_counts[NodeType.ROLE], rng)
        for i, tier in enumerate(r_tiers):
            rid = f"role:{dept_slug}-{tier.value}-{i}"
            r_arn = f"arn:aws:iam::{account_id}:role/{dept_slug}-{tier.value}-{i}"
            trust_policy = PolicyDocument(
                Statement=[
                    Statement(
                        Effect=Effect.ALLOW,
                        Action=["sts:AssumeRole"],
                        Principal=Principal.model_validate(
                            {
                                "AWS": [f"arn:aws:iam::{account_id}:root"],
                                "Service": ["ec2.amazonaws.com", "lambda.amazonaws.com"],
                            }
                        ),
                    )
                ]
            )
            node = GraphNode(
                id=rid,
                node_type=NodeType.ROLE,
                arn=r_arn,
                name=f"{dept_slug}-{tier.value}-{i}",
                account_id=account_id,
                department=dept,
                is_admin=(tier == PrivilegeTier.ADMIN),
                trust_policy=trust_policy,
                metadata={"privilege_tier": tier.value},
            )
            graph.add_node(node)
            nodes_by_type[NodeType.ROLE].append(node)

        return nodes_by_type

    def _generate_resource_attributes(
        self,
        dept: str,
        dept_slug: str,
        index: int,
        is_secops: bool,
    ) -> tuple[str, str, bool]:
        """Generate canonical AWS ARN and high-value status for department resources."""
        account_id = self.config.account_id
        if is_secops:
            # SecOps KMS keys are high value crown jewels
            rid = f"resource:kms-{dept_slug}-key-{index}"
            r_arn = f"arn:aws:kms:us-east-1:{account_id}:key/{dept_slug}-key-{index}"
            return rid, r_arn, True
        elif dept == DepartmentType.DATABI.value:
            if index == 0:
                # Primary data lake bucket is high-value
                rid = f"resource:s3-{dept_slug}-lake-vault-{index}"
                r_arn = f"arn:aws:s3:::{dept_slug}-lake-vault-{index}"
                return rid, r_arn, True
            else:
                rid = f"resource:s3-{dept_slug}-data-{index}"
                r_arn = f"arn:aws:s3:::{dept_slug}-data-{index}"
                return rid, r_arn, False
        else:
            rid = f"resource:s3-{dept_slug}-store-{index}"
            r_arn = f"arn:aws:s3:::{dept_slug}-store-{index}"
            return rid, r_arn, False

    def _sample_privilege_tiers(
        self,
        dept: str,
        count: int,
        rng: np.random.Generator,
    ) -> list[PrivilegeTier]:
        """Sample privilege tiers for identities using department power-law proportions."""
        if count <= 0:
            return []

        admin_p, std_p, rest_p = self.DEPT_TIER_PROPORTIONS.get(dept, (0.02, 0.28, 0.70))
        tier_choices = [PrivilegeTier.ADMIN, PrivilegeTier.STANDARD, PrivilegeTier.RESTRICTED]
        probs = [admin_p, std_p, rest_p]

        # Use fast numpy choice
        sampled_indices = rng.choice(len(tier_choices), size=count, p=probs)
        return [tier_choices[idx] for idx in sampled_indices]

    def _synthesize_department_policy(
        self,
        dept: str,
        policy_index: int,
        resources: list[GraphNode],
    ) -> tuple[PolicyDocument, bool]:
        """Synthesize capability-model-compliant IAM policy document for department."""
        stmts: list[Statement] = []
        is_admin = False

        if dept == DepartmentType.SECOPS.value:
            # SecOps admin or key management policy
            if policy_index == 0:
                is_admin = True
                kms_target = resources[0].arn if resources else "*"
                stmt = Statement(
                    Effect=Effect.ALLOW,
                    Action=["kms:DescribeKey", "kms:Decrypt", "kms:GenerateDataKey"],
                    Resource=[kms_target],
                )
            else:
                kms_target = resources[0].arn if resources else "*"
                stmt = Statement(
                    Effect=Effect.ALLOW,
                    Action=["kms:DescribeKey", "kms:Decrypt"],
                    Resource=[kms_target],
                )
            assert self.capability_model.validate_statement(stmt) == []
            stmts.append(stmt)
        elif dept == DepartmentType.DEVOPS.value:
            res_target = f"{resources[0].arn}/*" if resources else "*"
            s3_stmt = Statement(
                Effect=Effect.ALLOW,
                Action=["s3:GetObject", "s3:PutObject"],
                Resource=[res_target],
            )
            ec2_stmt = Statement(
                Effect=Effect.ALLOW,
                Action=["ec2:DescribeInstances"],
                Resource=["*"],
            )
            assert self.capability_model.validate_statement(s3_stmt) == []
            assert self.capability_model.validate_statement(ec2_stmt) == []
            stmts.extend([s3_stmt, ec2_stmt])
        elif dept == DepartmentType.DATABI.value:
            bucket_target = resources[0].arn if resources else "*"
            obj_target = f"{resources[0].arn}/*" if resources else "*"
            s3_list = Statement(
                Effect=Effect.ALLOW,
                Action=["s3:ListBucket"],
                Resource=[bucket_target],
            )
            s3_get = Statement(
                Effect=Effect.ALLOW,
                Action=["s3:GetObject"],
                Resource=[obj_target],
            )
            assert self.capability_model.validate_statement(s3_list) == []
            assert self.capability_model.validate_statement(s3_get) == []
            stmts.extend([s3_list, s3_get])
        else:
            # QA, Billing, Interns (Restricted read-only)
            target = f"{resources[0].arn}/*" if resources else "*"
            stmt = Statement(
                Effect=Effect.ALLOW,
                Action=["s3:GetObject"],
                Resource=[target],
            )
            assert self.capability_model.validate_statement(stmt) == []
            stmts.append(stmt)

        return PolicyDocument(Statement=stmts), is_admin

    def _generate_intra_department_edges(
        self,
        _dept: str,
        dept_nodes: dict[NodeType, list[GraphNode]],
        rng: np.random.Generator,
        graph: IAMGraph,
    ) -> None:
        """Vectorized generation of intra-department authorization relationships."""
        users = dept_nodes[NodeType.USER]
        roles = dept_nodes[NodeType.ROLE]
        groups = dept_nodes[NodeType.GROUP]
        policies = dept_nodes[NodeType.POLICY]
        resources = dept_nodes[NodeType.RESOURCE]

        density = self.config.edge_density

        # 1. MemberOf (Users -> Groups)
        if users and groups:
            # Vectorized group assignment: every user joins at least 1 group
            group_indices = rng.integers(0, len(groups), size=len(users))
            for u_idx, g_idx in enumerate(group_indices):
                graph.add_edge(
                    GraphEdge(
                        source=users[u_idx].id,
                        target=groups[g_idx].id,
                        relation=EdgeRelation.MEMBER_OF,
                    )
                )

            # Additional multi-group membership for ~30% of users
            multi_mask = rng.random(size=len(users)) < 0.30
            if len(groups) > 1 and np.any(multi_mask):
                alt_indices = rng.integers(0, len(groups), size=int(np.sum(multi_mask)))
                selected_users = [users[i] for i in np.where(multi_mask)[0]]
                for u, g_idx in zip(selected_users, alt_indices, strict=False):
                    if not graph.has_edge(u.id, groups[g_idx].id, EdgeRelation.MEMBER_OF):
                        graph.add_edge(
                            GraphEdge(
                                source=u.id,
                                target=groups[g_idx].id,
                                relation=EdgeRelation.MEMBER_OF,
                            )
                        )

        # 2. AttachedWith (Groups -> Policies)
        if groups and policies:
            for g in groups:
                num_p = min(len(policies), max(1, int(round(1.5 * density))))
                p_choices = rng.choice(policies, size=num_p, replace=False)
                for p in p_choices:
                    if not graph.has_edge(g.id, p.id, EdgeRelation.ATTACHED_WITH):
                        graph.add_edge(
                            GraphEdge(
                                source=g.id,
                                target=p.id,
                                relation=EdgeRelation.ATTACHED_WITH,
                            )
                        )

        # 3. AttachedWith (Roles -> Policies)
        if roles and policies:
            for r in roles:
                num_p = min(len(policies), max(1, int(round(1.2 * density))))
                p_choices = rng.choice(policies, size=num_p, replace=False)
                for p in p_choices:
                    if not graph.has_edge(r.id, p.id, EdgeRelation.ATTACHED_WITH):
                        graph.add_edge(
                            GraphEdge(
                                source=r.id,
                                target=p.id,
                                relation=EdgeRelation.ATTACHED_WITH,
                            )
                        )

        # 4. ActsOn (Policies -> Resources)
        if policies and resources:
            for p in policies:
                num_res = min(len(resources), max(1, int(round(1.3 * density))))
                res_choices = rng.choice(resources, size=num_res, replace=False)
                for res in res_choices:
                    if not graph.has_edge(p.id, res.id, EdgeRelation.ACTS_ON):
                        graph.add_edge(
                            GraphEdge(
                                source=p.id,
                                target=res.id,
                                relation=EdgeRelation.ACTS_ON,
                            )
                        )

        # 5. AssumesRole (Users -> Roles) within department (moderate density)
        if users and roles:
            num_assumptions = max(1, int(len(users) * 0.15 * density))
            u_sample = rng.choice(users, size=min(len(users), num_assumptions), replace=False)
            r_sample = rng.choice(roles, size=len(u_sample), replace=True)
            for u, r in zip(u_sample, r_sample, strict=False):
                # Don't give low-privilege users admin role assumptions benignly
                if not u.is_admin and r.is_admin:
                    continue
                if not graph.has_edge(u.id, r.id, EdgeRelation.ASSUMES_ROLE):
                    graph.add_edge(
                        GraphEdge(
                            source=u.id,
                            target=r.id,
                            relation=EdgeRelation.ASSUMES_ROLE,
                            actions=["sts:AssumeRole"],
                        )
                    )

    def _generate_benign_business_workflows(
        self,
        dept_nodes_by_dept: dict[str, dict[NodeType, list[GraphNode]]],
        _rng: np.random.Generator,
        graph: IAMGraph,
    ) -> None:
        """Synthesize realistic benign workflows required by enterprise operations."""
        account_id = self.config.account_id

        # Workflow 1: CI/CD Build Runners (DevOps)
        devops_nodes = dept_nodes_by_dept.get(DepartmentType.DEVOPS.value)
        if devops_nodes and devops_nodes[NodeType.ROLE] and devops_nodes[NodeType.RESOURCE]:
            cicd_role = devops_nodes[NodeType.ROLE][0]
            build_res = devops_nodes[NodeType.RESOURCE][0]

            cicd_pol_id = f"policy:devops-cicd-build-pipeline-{account_id[-4:]}"
            cicd_stmt1 = Statement(
                Effect=Effect.ALLOW,
                Action=["s3:GetObject", "s3:PutObject"],
                Resource=[f"{build_res.arn}/*"],
            )
            cicd_stmt2 = Statement(
                Effect=Effect.ALLOW,
                Action=["ec2:DescribeInstances"],
                Resource=["*"],
            )
            assert self.capability_model.validate_statement(cicd_stmt1) == []
            assert self.capability_model.validate_statement(cicd_stmt2) == []

            if not graph.get_node(cicd_pol_id):
                cicd_pol = GraphNode(
                    id=cicd_pol_id,
                    node_type=NodeType.POLICY,
                    arn=f"arn:aws:iam::{account_id}:policy/devops-cicd-build-pipeline",
                    name="devops-cicd-build-pipeline",
                    account_id=account_id,
                    department=DepartmentType.DEVOPS.value,
                    policy_document=PolicyDocument(Statement=[cicd_stmt1, cicd_stmt2]),
                )
                graph.add_node(cicd_pol)
                graph.add_edge(
                    GraphEdge(
                        source=cicd_role.id,
                        target=cicd_pol_id,
                        relation=EdgeRelation.ATTACHED_WITH,
                    )
                )
                graph.add_edge(
                    GraphEdge(
                        source=cicd_pol_id,
                        target=build_res.id,
                        relation=EdgeRelation.ACTS_ON,
                        actions=["s3:GetObject", "s3:PutObject"],
                    )
                )

        # Workflow 2: Read-Only Data Analysts (DataBI)
        databi_nodes = dept_nodes_by_dept.get(DepartmentType.DATABI.value)
        if (
            databi_nodes
            and databi_nodes[NodeType.USER]
            and databi_nodes[NodeType.GROUP]
            and databi_nodes[NodeType.RESOURCE]
        ):
            analyst_user = databi_nodes[NodeType.USER][0]
            analyst_group = databi_nodes[NodeType.GROUP][0]
            lake_res = databi_nodes[NodeType.RESOURCE][0]

            analyst_pol_id = f"policy:databi-athena-analyst-query-{account_id[-4:]}"
            lake_key_arn = f"arn:aws:kms:us-east-1:{account_id}:key/analytics-data-key"

            # Create analytics KMS key if not in graph
            lake_key_node = graph.get_node_by_arn(lake_key_arn)
            if not lake_key_node:
                lake_key_node = GraphNode(
                    id="resource:kms-analytics-lake-key",
                    node_type=NodeType.RESOURCE,
                    arn=lake_key_arn,
                    name="analytics-data-key",
                    account_id=account_id,
                    department=DepartmentType.DATABI.value,
                    is_high_value=False,
                )
                graph.add_node(lake_key_node)

            stmt_list = Statement(
                Effect=Effect.ALLOW,
                Action=["s3:ListBucket"],
                Resource=[lake_res.arn],
            )
            stmt_get = Statement(
                Effect=Effect.ALLOW,
                Action=["s3:GetObject"],
                Resource=[f"{lake_res.arn}/*"],
            )
            stmt_kms = Statement(
                Effect=Effect.ALLOW,
                Action=["kms:Decrypt"],
                Resource=[lake_key_arn],
            )
            assert self.capability_model.validate_statement(stmt_list) == []
            assert self.capability_model.validate_statement(stmt_get) == []
            assert self.capability_model.validate_statement(stmt_kms) == []

            if not graph.get_node(analyst_pol_id):
                analyst_pol = GraphNode(
                    id=analyst_pol_id,
                    node_type=NodeType.POLICY,
                    arn=f"arn:aws:iam::{account_id}:policy/databi-athena-analyst-query",
                    name="databi-athena-analyst-query",
                    account_id=account_id,
                    department=DepartmentType.DATABI.value,
                    policy_document=PolicyDocument(Statement=[stmt_list, stmt_get, stmt_kms]),
                )
                graph.add_node(analyst_pol)
                if not graph.has_edge(analyst_user.id, analyst_group.id, EdgeRelation.MEMBER_OF):
                    graph.add_edge(
                        GraphEdge(
                            source=analyst_user.id,
                            target=analyst_group.id,
                            relation=EdgeRelation.MEMBER_OF,
                        )
                    )
                graph.add_edge(
                    GraphEdge(
                        source=analyst_group.id,
                        target=analyst_pol_id,
                        relation=EdgeRelation.ATTACHED_WITH,
                    )
                )
                graph.add_edge(
                    GraphEdge(
                        source=analyst_pol_id,
                        target=lake_res.id,
                        relation=EdgeRelation.ACTS_ON,
                        actions=["s3:ListBucket", "s3:GetObject"],
                    )
                )
                graph.add_edge(
                    GraphEdge(
                        source=analyst_pol_id,
                        target=lake_key_node.id,
                        relation=EdgeRelation.ACTS_ON,
                        actions=["kms:Decrypt"],
                    )
                )

    def _generate_inter_department_trust(
        self,
        dept_nodes_by_dept: dict[str, dict[NodeType, list[GraphNode]]],
        _rng: np.random.Generator,
        graph: IAMGraph,
    ) -> None:
        """Controlled cross-department role assumption edges simulating enterprise handoffs."""
        devops_nodes = dept_nodes_by_dept.get(DepartmentType.DEVOPS.value)
        qa_nodes = dept_nodes_by_dept.get(DepartmentType.QA.value)

        # DevOps lead assuming QA test role for staging verification
        if devops_nodes and qa_nodes:
            devops_users = [u for u in devops_nodes[NodeType.USER] if not u.is_admin]
            qa_roles = [r for r in qa_nodes[NodeType.ROLE] if not r.is_admin]
            if devops_users and qa_roles:
                u = devops_users[0]
                r = qa_roles[0]
                if not graph.has_edge(u.id, r.id, EdgeRelation.ASSUMES_ROLE):
                    graph.add_edge(
                        GraphEdge(
                            source=u.id,
                            target=r.id,
                            relation=EdgeRelation.ASSUMES_ROLE,
                            actions=["sts:AssumeRole"],
                        )
                    )

    def _inject_canonical_pe_chains(
        self,
        graph: IAMGraph,
        rng: np.random.Generator,
    ) -> tuple[list[MotifInstance], set[str]]:
        """Inject positive privilege escalation chains embedded in benign noise."""
        injected: list[MotifInstance] = []
        used_sources: set[str] = set()
        if self.config.num_pe_chains <= 0:
            return injected, used_sources

        # Collect low-privilege benign source candidates (Interns, QA, DevOps)
        low_priv_candidates: list[GraphNode] = []
        for dept in [
            DepartmentType.INTERNS.value,
            DepartmentType.QA.value,
            DepartmentType.DEVOPS.value,
        ]:
            candidates = [
                n
                for n in graph.get_nodes_by_type(NodeType.USER)
                if n.department == dept and not n.is_admin
            ]
            low_priv_candidates.extend(candidates)

        if not low_priv_candidates:
            low_priv_candidates = [
                n for n in graph.get_nodes_by_type(NodeType.USER) if not n.is_admin
            ]

        # Collect high-value targets by type
        kms_hv_targets = [
            n for n in graph.get_high_value_targets() if n.arn.startswith("arn:aws:kms:")
        ]
        s3_hv_targets = [
            n for n in graph.get_high_value_targets() if n.arn.startswith("arn:aws:s3:")
        ]

        # Motif mapping by hop length
        # 2-hop: attach_policy, create_access_key, set_default_policy_version
        # 3-hop: passrole_lambda, passrole_ec2
        # 4-hop: assume_role_chain (with 1 intermediate role)
        hop_motif_map: dict[int, list[str]] = {
            2: ["attach_policy", "set_default_policy_version", "create_access_key"],
            3: ["passrole_lambda", "passrole_ec2"],
            4: ["assume_role_chain"],
        }

        for idx in range(self.config.num_pe_chains):
            # 1. Sample hop length
            hop_len = int(rng.choice(self.config.pe_chain_lengths))
            eligible_motifs = hop_motif_map.get(hop_len, ["passrole_lambda"])

            if self.config.pe_motif_types:
                eligible_motifs = [m for m in eligible_motifs if m in self.config.pe_motif_types]
                if not eligible_motifs:
                    eligible_motifs = self.config.pe_motif_types

            motif_type = str(rng.choice(eligible_motifs))
            motif = self.motif_registry.get(motif_type)

            # 2. Select low-privilege source (unique to prevent cross-chain interference)
            avail_candidates = [n for n in low_priv_candidates if n.id not in used_sources]
            src_node = (
                rng.choice(avail_candidates)
                if avail_candidates
                else (rng.choice(low_priv_candidates) if low_priv_candidates else None)
            )
            src_id = src_node.id if src_node else None
            if src_id:
                used_sources.add(src_id)

            # 3. Select compatible high-value target
            if motif_type in ("passrole_lambda", "assume_role_chain"):
                tgt_pool = kms_hv_targets
            else:
                tgt_pool = s3_hv_targets

            tgt_node = rng.choice(tgt_pool) if tgt_pool else None
            tgt_id = tgt_node.id if tgt_node else None

            # 4. Inject motif with kwargs
            kwargs: dict[str, Any] = {}
            if motif_type == "assume_role_chain":
                kwargs["num_intermediate_roles"] = 1 if hop_len == 4 else 2

            instance = motif.inject(
                graph=graph,
                source_id=src_id,
                target_id=tgt_id,
                suffix=f"cpe_{idx}",
                account_id=self.config.account_id,
                **kwargs,
            )

            # 5. Formally verify reachability and counterfactual bridge breaking
            verification = motif.verify_ground_truth(graph, instance)
            assert verification.is_valid, f"Injected PE chain '{instance.instance_id}' failed ground truth: {verification.details}"

            injected.append(instance)

        return injected, used_sources

    def _inject_branching_pe_chains(
        self,
        graph: IAMGraph,
        rng: np.random.Generator,
        used_sources: set[str] | None = None,
    ) -> list[BranchingPEChain]:
        """Inject multi-path branching PE topologies with formal verification."""
        branching_chains: list[BranchingPEChain] = []
        account_id = self.config.account_id
        used = used_sources if used_sources is not None else set()

        # Low privilege source candidate
        intern_users = [
            n
            for n in graph.get_nodes_by_type(NodeType.USER)
            if n.department == DepartmentType.INTERNS.value and not n.is_admin
        ]
        if not intern_users:
            intern_users = [n for n in graph.get_nodes_by_type(NodeType.USER) if not n.is_admin]

        kms_hv_targets = [
            n for n in graph.get_high_value_targets() if n.arn.startswith("arn:aws:kms:")
        ]
        if not kms_hv_targets:
            return branching_chains

        for b_idx in range(self.config.num_branching_chains):
            avail_interns = [n for n in intern_users if n.id not in used]
            src_node = rng.choice(avail_interns) if avail_interns else rng.choice(intern_users)
            used.add(src_node.id)
            target_node = rng.choice(kms_hv_targets)
            uid = f"br_{b_idx}"

            # Diamond topology: Branch 1 = PassRole to Lambda; Branch 2 = AssumeRole chain
            # Both escalate to the same target admin role / policy, but through distinct bridges
            r_admin_id = f"role:diamond-superadmin-{uid}"
            r_admin_arn = f"arn:aws:iam::{account_id}:role/diamond-superadmin-{uid}"
            r_step1_arn = f"arn:aws:iam::{account_id}:role/assume-step1-{uid}"
            admin_trust_doc = PolicyDocument(
                Statement=[
                    Statement(
                        Effect=Effect.ALLOW,
                        Action=["sts:AssumeRole"],
                        Principal=Principal.model_validate({"AWS": [r_step1_arn]}),
                    )
                ]
            )
            r_admin = GraphNode(
                id=r_admin_id,
                node_type=NodeType.ROLE,
                arn=r_admin_arn,
                name=f"diamond-superadmin-{uid}",
                account_id=account_id,
                department=DepartmentType.SECOPS.value,
                is_admin=True,
                trust_policy=admin_trust_doc,
            )
            graph.add_node(r_admin)

            admin_pol_id = f"policy:diamond-admin-kms-{uid}"
            admin_stmt = Statement(
                Effect=Effect.ALLOW,
                Action=["kms:Decrypt", "kms:GenerateDataKey"],
                Resource=[target_node.arn],
            )
            assert self.capability_model.validate_statement(admin_stmt) == []
            admin_pol = GraphNode(
                id=admin_pol_id,
                node_type=NodeType.POLICY,
                arn=f"arn:aws:iam::{account_id}:policy/diamond-admin-kms-{uid}",
                name=f"diamond-admin-kms-{uid}",
                account_id=account_id,
                department=DepartmentType.SECOPS.value,
                is_admin=True,
                policy_document=PolicyDocument(Statement=[admin_stmt]),
            )
            graph.add_node(admin_pol)
            graph.add_edge(
                GraphEdge(
                    source=r_admin_id,
                    target=admin_pol_id,
                    relation=EdgeRelation.ATTACHED_WITH,
                )
            )
            graph.add_edge(
                GraphEdge(
                    source=admin_pol_id,
                    target=target_node.id,
                    relation=EdgeRelation.ACTS_ON,
                    actions=["kms:Decrypt", "kms:GenerateDataKey"],
                )
            )

            # --- Branch 1: PassRole (Source -> r_admin) ---
            br1_pol_id = f"policy:passrole-branch1-{uid}"
            pass_stmt = Statement(
                Effect=Effect.ALLOW,
                Action=["iam:PassRole"],
                Resource=[r_admin_arn],
            )
            assert self.capability_model.validate_statement(pass_stmt) == []
            br1_pol = GraphNode(
                id=br1_pol_id,
                node_type=NodeType.POLICY,
                arn=f"arn:aws:iam::{account_id}:policy/passrole-branch1-{uid}",
                name=f"passrole-branch1-{uid}",
                account_id=account_id,
                department=DepartmentType.INTERNS.value,
                policy_document=PolicyDocument(Statement=[pass_stmt]),
            )
            graph.add_node(br1_pol)
            graph.add_edge(
                GraphEdge(
                    source=src_node.id,
                    target=br1_pol_id,
                    relation=EdgeRelation.ATTACHED_WITH,
                )
            )
            # Bridge 1
            bridge_1 = (src_node.id, r_admin_id, EdgeRelation.PASSES_TO.value)
            graph.add_edge(
                GraphEdge(
                    source=src_node.id,
                    target=r_admin_id,
                    relation=EdgeRelation.PASSES_TO,
                    is_bridge=True,
                    motif_id=f"diamond_branch1_{uid}",
                    actions=["iam:PassRole"],
                )
            )

            # --- Branch 2: AssumeRole chain (Source -> r_step1 -> r_admin) ---
            r_step1_id = f"role:assume-step1-{uid}"
            step1_trust_doc = PolicyDocument(
                Statement=[
                    Statement(
                        Effect=Effect.ALLOW,
                        Action=["sts:AssumeRole"],
                        Principal=Principal.model_validate({"AWS": [src_node.arn]}),
                    )
                ]
            )
            r_step1 = GraphNode(
                id=r_step1_id,
                node_type=NodeType.ROLE,
                arn=r_step1_arn,
                name=f"assume-step1-{uid}",
                account_id=account_id,
                department=DepartmentType.DEVOPS.value,
                trust_policy=step1_trust_doc,
            )
            graph.add_node(r_step1)
            graph.add_edge(
                GraphEdge(
                    source=src_node.id,
                    target=r_step1_id,
                    relation=EdgeRelation.ASSUMES_ROLE,
                    actions=["sts:AssumeRole"],
                )
            )
            # Bridge 2
            bridge_2 = (r_step1_id, r_admin_id, EdgeRelation.ASSUMES_ROLE.value)
            graph.add_edge(
                GraphEdge(
                    source=r_step1_id,
                    target=r_admin_id,
                    relation=EdgeRelation.ASSUMES_ROLE,
                    is_bridge=True,
                    motif_id=f"diamond_branch2_{uid}",
                    actions=["sts:AssumeRole"],
                )
            )

            branch_chain = BranchingPEChain(
                chain_id=f"diamond_{uid}",
                branching_type=BranchingType.CONVERGENT_DIAMOND,
                source_id=src_node.id,
                target_ids=[target_node.id],
                bridge_relations=[bridge_1, bridge_2],
                all_path_edges=[
                    bridge_1,
                    (src_node.id, r_step1_id, EdgeRelation.ASSUMES_ROLE.value),
                    bridge_2,
                    (r_admin_id, admin_pol_id, EdgeRelation.ATTACHED_WITH.value),
                    (admin_pol_id, target_node.id, EdgeRelation.ACTS_ON.value),
                ],
                metadata={
                    "source_arn": src_node.arn,
                    "target_arn": target_node.arn,
                },
            )

            verification = branch_chain.verify(graph)
            assert verification.is_valid, f"Branching PE chain '{branch_chain.chain_id}' failed verification: {verification.details}"

            branching_chains.append(branch_chain)

        return branching_chains

    def _compute_graph_stats(self, graph: IAMGraph) -> dict[str, Any]:
        """Compute structural metrics and summary statistics for the generated topology."""
        nodes = graph.get_nodes_by_type
        admin_count = len(graph.get_admin_nodes())
        total_identities = len(nodes(NodeType.USER)) + len(nodes(NodeType.ROLE))
        admin_pct = (admin_count / total_identities * 100.0) if total_identities > 0 else 0.0

        edge_counts: dict[str, int] = {rel.value: 0 for rel in EdgeRelation}
        for edge in graph.get_edges():
            edge_counts[edge.relation.value] += 1

        dept_dist: dict[str, int] = {}
        for n in graph.nx_graph.nodes.values():
            d = n.get("department", "Unknown")
            dept_dist[d] = dept_dist.get(d, 0) + 1

        return {
            "total_nodes": graph.num_nodes,
            "total_edges": graph.num_edges,
            "average_degree": round(graph.num_edges / max(1, graph.num_nodes), 2),
            "node_counts": {t.value: len(nodes(t)) for t in NodeType},
            "edge_counts": edge_counts,
            "department_distribution": dept_dist,
            "admin_identities": admin_count,
            "admin_percentage": round(admin_pct, 2),
            "high_value_targets": len(graph.get_high_value_targets()),
            "bridge_edges_count": len(graph.get_bridge_edges()),
        }
