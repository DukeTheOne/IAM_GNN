"""Rigorous Generator Sanity Tests for mathematical, syntactic, and semantic correctness.

Covers:
1. Syntactic & Capability Model Validity: 100% policy AST compliance across all nodes and motifs.
2. Formal Semantic Traversability: Step-by-step verification of attack paths using PolicyEvaluator.
3. False Positive Guarantees: Rigorous topological and formal IAM evaluation checks for benign identities.
4. Scalability & Memory Profiling: Runtime and heap allocation benchmarking across N in [500, 5000].
"""

from __future__ import annotations

import time
import tracemalloc

import networkx as nx
import pytest

from iam.generator import (
    DepartmentType,
    EdgeRelation,
    EnterpriseTopologyConfig,
    EnterpriseTopologyGenerator,
    EnvironmentExporter,
    GraphNode,
    GroundTruthLabeler,
    IAMGraph,
    NodeType,
)
from iam.parser import (
    AuthRequest,
    CapabilityModel,
    PolicyDocument,
    PolicyEvaluator,
    load_default_capability_model,
)


@pytest.fixture
def capability_model() -> CapabilityModel:
    """Fixture providing loaded AWS CapabilityModel."""
    return load_default_capability_model()


@pytest.fixture
def standard_enterprise_graph(capability_model: CapabilityModel) -> IAMGraph:
    """Fixture providing an enterprise graph with canonical and branching PE chains."""
    cfg = EnterpriseTopologyConfig(
        num_nodes=300,
        seed=101,
        num_pe_chains=4,
        pe_chain_lengths=[2, 3, 4],
        include_branching_chains=True,
        num_branching_chains=1,
    )
    generator = EnterpriseTopologyGenerator(config=cfg, capability_model=capability_model)
    return generator.generate()


class TestSyntacticCapabilityValidity:
    """Tests ensuring all generated policy documents and trust policies are syntactically valid and capability-compliant."""

    def test_all_policy_documents_capability_compliant(
        self,
        standard_enterprise_graph: IAMGraph,
        capability_model: CapabilityModel,
    ) -> None:
        """Verify 100% of generated Policy nodes comply with the formal CapabilityModel."""
        policy_nodes = standard_enterprise_graph.get_nodes_by_type(NodeType.POLICY)
        assert len(policy_nodes) > 0

        for pol in policy_nodes:
            assert pol.policy_document is not None, f"Policy node {pol.id} missing policy_document"
            assert len(pol.policy_document.statements) > 0
            for stmt in pol.policy_document.statements:
                errors = capability_model.validate_statement(stmt)
                assert errors == [], f"Policy {pol.id} statement invalid: {errors}"

    def test_all_role_trust_policies_syntactically_valid(
        self,
        standard_enterprise_graph: IAMGraph,
    ) -> None:
        """Verify that all Role nodes with trust policies possess valid AssumeRole statements."""
        role_nodes = standard_enterprise_graph.get_nodes_by_type(NodeType.ROLE)
        assert len(role_nodes) > 0

        roles_with_trust = [r for r in role_nodes if r.trust_policy is not None]
        assert len(roles_with_trust) > 0

        for role in roles_with_trust:
            assert role.trust_policy is not None
            for stmt in role.trust_policy.statements:
                assert stmt.effect.value == "Allow"
                assert "sts:AssumeRole" in stmt.action
                assert stmt.principal is not None

    def test_multi_seed_syntactic_robustness(
        self,
        capability_model: CapabilityModel,
    ) -> None:
        """Validate syntactic correctness across multiple randomized generator seeds."""
        for seed in [123, 456, 789]:
            cfg = EnterpriseTopologyConfig(
                num_nodes=150,
                seed=seed,
                num_pe_chains=3,
                include_branching_chains=True,
            )
            graph = EnterpriseTopologyGenerator(cfg, capability_model=capability_model).generate()
            for pol in graph.get_nodes_by_type(NodeType.POLICY):
                if pol.policy_document:
                    for stmt in pol.policy_document.statements:
                        assert capability_model.validate_statement(stmt) == []


class TestFormalTraversability:
    """Tests verifying that every step of injected PE chains is traversable under formal IAM evaluation rules."""

    def test_canonical_pe_motifs_formal_evaluation(
        self,
        standard_enterprise_graph: IAMGraph,
    ) -> None:
        """Verify each injected canonical PE motif using symbolic PolicyEvaluator."""
        injected = standard_enterprise_graph.metadata.get("injected_motifs", [])
        assert len(injected) > 0

        for m_data in injected:
            m_type = m_data["motif_type"]
            src_node = standard_enterprise_graph.get_node(m_data["source_id"])
            tgt_node = standard_enterprise_graph.get_node(m_data["target_id"])
            assert src_node is not None
            assert tgt_node is not None

            # Collect source attached policies
            src_attached_pols = [
                standard_enterprise_graph.get_node(e.target)
                for e in standard_enterprise_graph.get_edges()
                if e.source == src_node.id
                and e.relation == EdgeRelation.ATTACHED_WITH
                and standard_enterprise_graph.get_node(e.target) is not None
            ]
            src_policy_docs = [
                p.policy_document for p in src_attached_pols if p and p.policy_document
            ]

            if m_type == "assume_role_chain":
                # For role chaining, each role must permit sts:AssumeRole in its trust policy
                chain_nodes: list[GraphNode] = [
                    node
                    for nid in m_data["intermediate_node_ids"]
                    if (node := standard_enterprise_graph.get_node(nid)) is not None
                ]
                prev = src_node
                for curr in chain_nodes:
                    if curr.node_type == NodeType.ROLE and curr.trust_policy:
                        req = AuthRequest(
                            principal_arn=prev.arn,
                            action="sts:AssumeRole",
                            resource_arn=curr.arn,
                        )
                        evaluator = PolicyEvaluator(policies=[curr.trust_policy])
                        res = evaluator.evaluate(req)
                        assert res.is_allowed is True, (
                            f"AssumeRole failed for {prev.arn} -> {curr.arn}"
                        )
                    prev = curr

            elif m_type in ("passrole_lambda", "passrole_ec2"):
                # Source must have iam:PassRole authorized by an attached policy
                req = AuthRequest(
                    principal_arn=src_node.arn,
                    action="iam:PassRole",
                    resource_arn="*",
                )
                evaluator = PolicyEvaluator(policies=src_policy_docs)
                res = evaluator.evaluate(req)
                assert res.is_allowed is True, f"PassRole evaluation denied for {src_node.arn}"

            elif m_type == "attach_policy":
                # Source must have iam:AttachUserPolicy authorized
                req = AuthRequest(
                    principal_arn=src_node.arn,
                    action="iam:AttachUserPolicy",
                    resource_arn=src_node.arn,
                )
                evaluator = PolicyEvaluator(policies=src_policy_docs)
                res = evaluator.evaluate(req)
                assert res.is_allowed is True, f"AttachUserPolicy denied for {src_node.arn}"

            elif m_type == "create_access_key":
                # Source must have iam:CreateAccessKey authorized
                req = AuthRequest(
                    principal_arn=src_node.arn,
                    action="iam:CreateAccessKey",
                    resource_arn="*",
                )
                evaluator = PolicyEvaluator(policies=src_policy_docs)
                res = evaluator.evaluate(req)
                assert res.is_allowed is True, f"CreateAccessKey denied for {src_node.arn}"

            elif m_type == "set_default_policy_version":
                # Source must have iam:SetDefaultPolicyVersion authorized
                req = AuthRequest(
                    principal_arn=src_node.arn,
                    action="iam:SetDefaultPolicyVersion",
                    resource_arn="*",
                )
                evaluator = PolicyEvaluator(policies=src_policy_docs)
                res = evaluator.evaluate(req)
                assert res.is_allowed is True, f"SetDefaultPolicyVersion denied for {src_node.arn}"

    def test_branching_diamond_formal_traversability(
        self,
        standard_enterprise_graph: IAMGraph,
    ) -> None:
        """Verify both branches of convergent diamond PE chains under formal IAM evaluation."""
        branching = standard_enterprise_graph.metadata.get("branching_chains", [])
        if not branching:
            pytest.skip("No branching chains in graph")

        for b_data in branching:
            src_node = standard_enterprise_graph.get_node(b_data["source_id"])
            tgt_node = standard_enterprise_graph.get_node(b_data["target_ids"][0])
            assert src_node is not None
            assert tgt_node is not None

            # Branch 1: PassRole policy attached to source
            src_pols = [
                standard_enterprise_graph.get_node(e.target)
                for e in standard_enterprise_graph.get_edges()
                if e.source == src_node.id
                and e.relation == EdgeRelation.ATTACHED_WITH
                and standard_enterprise_graph.get_node(e.target) is not None
            ]
            src_docs = [p.policy_document for p in src_pols if p and p.policy_document]

            uid = b_data["chain_id"].replace("diamond_", "")
            r_admin = standard_enterprise_graph.get_node(f"role:diamond-superadmin-{uid}")
            assert r_admin is not None

            pass_req = AuthRequest(
                principal_arn=src_node.arn,
                action="iam:PassRole",
                resource_arn=r_admin.arn,
            )
            evaluator_br1 = PolicyEvaluator(policies=src_docs)
            assert evaluator_br1.evaluate(pass_req).is_allowed is True

            # Branch 2: AssumeRole hop to intermediate role
            r_step1 = standard_enterprise_graph.get_node(
                f"role:assume-step1-{b_data['chain_id'].replace('diamond_', '')}"
            )
            if r_step1 and r_step1.trust_policy:
                assume_req = AuthRequest(
                    principal_arn=src_node.arn,
                    action="sts:AssumeRole",
                    resource_arn=r_step1.arn,
                )
                evaluator_br2 = PolicyEvaluator(policies=[r_step1.trust_policy])
                assert evaluator_br2.evaluate(assume_req).is_allowed is True


class TestFalsePositiveGuarantees:
    """Tests ensuring benign restricted identities cannot escalate privilege."""

    def test_benign_identities_topologically_isolated_from_crown_jewels(
        self,
        standard_enterprise_graph: IAMGraph,
    ) -> None:
        """Verify restricted identities in Billing, Interns, and QA have no path to admin targets."""
        nx_g = standard_enterprise_graph.nx_graph
        injected_sources = {
            m["source_id"] for m in standard_enterprise_graph.metadata.get("injected_motifs", [])
        }
        for b in standard_enterprise_graph.metadata.get("branching_chains", []):
            injected_sources.add(b["source_id"])

        hv_targets = [n.id for n in standard_enterprise_graph.get_high_value_targets()]
        admin_roles = [
            n.id
            for n in standard_enterprise_graph.get_admin_nodes()
            if n.node_type == NodeType.ROLE
        ]

        benign_users = [
            n
            for n in standard_enterprise_graph.get_nodes_by_type(NodeType.USER)
            if n.department
            in (DepartmentType.BILLING.value, DepartmentType.INTERNS.value, DepartmentType.QA.value)
            and not n.is_admin
            and n.id not in injected_sources
        ]

        assert len(benign_users) > 0
        for user in benign_users[:15]:
            for target_id in hv_targets + admin_roles:
                assert not nx.has_path(nx_g, user.id, target_id), (
                    f"False positive topological path: {user.id} -> {target_id}"
                )

    def test_ground_truth_labeler_false_positive_exclusion(
        self,
        standard_enterprise_graph: IAMGraph,
    ) -> None:
        """Verify GroundTruthLabeler does not label any benign restricted identity as a PE source."""
        injected_sources = {
            m["source_id"] for m in standard_enterprise_graph.metadata.get("injected_motifs", [])
        }
        for b in standard_enterprise_graph.metadata.get("branching_chains", []):
            injected_sources.add(b["source_id"])

        labeler = GroundTruthLabeler(max_depth=6)
        label_set = labeler.label_graph(standard_enterprise_graph)

        restricted_depts = {
            DepartmentType.INTERNS.value,
            DepartmentType.BILLING.value,
            DepartmentType.QA.value,
        }

        for pe in label_set.pe_pairs:
            src_node = standard_enterprise_graph.get_node(pe.source_id)
            if (
                src_node
                and src_node.node_type == NodeType.USER
                and src_node.department in restricted_depts
            ):
                assert pe.source_id in injected_sources, (
                    f"GroundTruthLabeler falsely classified non-injected source {pe.source_id} in {src_node.department} as PE"
                )

    def test_benign_identities_formal_evaluation_denial(
        self,
        standard_enterprise_graph: IAMGraph,
    ) -> None:
        """Verify that evaluating admin requests on benign identities yields Implicit or Explicit Deny."""
        injected_sources = {
            m["source_id"] for m in standard_enterprise_graph.metadata.get("injected_motifs", [])
        }
        benign_users = [
            n
            for n in standard_enterprise_graph.get_nodes_by_type(NodeType.USER)
            if n.department in (DepartmentType.BILLING.value, DepartmentType.INTERNS.value)
            and not n.is_admin
            and n.id not in injected_sources
        ]

        hv_kms = next(
            (n for n in standard_enterprise_graph.get_high_value_targets() if "kms" in n.arn),
            None,
        )

        for user in benign_users[:10]:
            # Collect effective policy documents (direct + groups)
            attached_pols: list[PolicyDocument] = []
            for edge in standard_enterprise_graph.get_edges():
                if edge.source == user.id and edge.relation == EdgeRelation.ATTACHED_WITH:
                    p = standard_enterprise_graph.get_node(edge.target)
                    if p and p.policy_document:
                        attached_pols.append(p.policy_document)
                elif edge.source == user.id and edge.relation == EdgeRelation.MEMBER_OF:
                    group = standard_enterprise_graph.get_node(edge.target)
                    if group:
                        for g_edge in standard_enterprise_graph.get_edges():
                            if (
                                g_edge.source == group.id
                                and g_edge.relation == EdgeRelation.ATTACHED_WITH
                            ):
                                gp = standard_enterprise_graph.get_node(g_edge.target)
                                if gp and gp.policy_document:
                                    attached_pols.append(gp.policy_document)

            evaluator = PolicyEvaluator(policies=attached_pols)

            # 1. Deny full admin actions
            admin_req = AuthRequest(
                principal_arn=user.arn,
                action="iam:AttachUserPolicy",
                resource_arn="*",
            )
            assert evaluator.evaluate(admin_req).is_allowed is False

            # 2. Deny KMS Decrypt on high-value asset
            if hv_kms:
                kms_req = AuthRequest(
                    principal_arn=user.arn,
                    action="kms:Decrypt",
                    resource_arn=hv_kms.arn,
                )
                assert evaluator.evaluate(kms_req).is_allowed is False


class TestScalabilityAndMemoryProfiling:
    """Benchmark runtime and memory consumption across graph sizes from N=500 to N=5000."""

    @pytest.mark.parametrize(
        "num_nodes,max_sec,max_mb",
        [
            (500, 1.0, 25.0),
            (1000, 2.5, 50.0),
            (2500, 8.0, 120.0),
            (5000, 25.0, 250.0),
        ],
    )
    def test_runtime_and_memory_scaling(
        self,
        num_nodes: int,
        max_sec: float,
        max_mb: float,
        capability_model: CapabilityModel,
    ) -> None:
        """Assert both generation duration and peak heap memory stay strictly within architectural limits."""
        cfg = EnterpriseTopologyConfig(
            num_nodes=num_nodes,
            seed=42,
            num_pe_chains=3,
            include_branching_chains=True,
            num_branching_chains=1,
        )
        gen = EnterpriseTopologyGenerator(cfg, capability_model=capability_model)

        tracemalloc.start()
        start_time = time.perf_counter()
        graph = gen.generate()
        duration = time.perf_counter() - start_time
        _current, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_bytes / (1024 * 1024)
        bytes_per_node = peak_bytes / max(1, graph.num_nodes)

        # Runtime assertion
        assert duration < max_sec, (
            f"N={num_nodes} took {duration:.2f}s, exceeding maximum allowed {max_sec}s"
        )

        # Peak memory assertion
        assert peak_mb < max_mb, (
            f"N={num_nodes} allocated {peak_mb:.2f}MB, exceeding maximum allowed {max_mb}MB"
        )

        # Linear memory complexity assertion (< 50 KB per node)
        assert bytes_per_node < 50_000, (
            f"N={num_nodes} used {bytes_per_node:.0f} bytes/node, exceeding 50KB/node limit"
        )

    def test_end_to_end_pipeline_scaling_n1000(
        self,
        capability_model: CapabilityModel,
    ) -> None:
        """Verify generation, labeling, sanity checking, and export together complete in < 2.5s for N=1000."""
        cfg = EnterpriseTopologyConfig(num_nodes=1000, seed=42, num_pe_chains=3)
        gen = EnterpriseTopologyGenerator(cfg, capability_model=capability_model)

        start = time.perf_counter()
        graph = gen.generate()
        labeler = GroundTruthLabeler(max_depth=5, find_all_paths=False)
        label_set = labeler.label_graph(graph)
        exporter = EnvironmentExporter(use_orjson=True)
        pyg_data = exporter.to_pyg_ready_dict(graph, label_set)
        total_time = time.perf_counter() - start

        assert total_time < 2.5, f"Complete N=1000 pipeline took {total_time:.2f}s (> 2.5s)"
        assert label_set.num_pe_pairs > 0
        assert len(pyg_data["pe_pairs"]) == label_set.num_pe_pairs
