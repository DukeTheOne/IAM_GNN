"""Comprehensive unit and integration tests for Parameterized Enterprise Topology Generator."""

import time

import networkx as nx
import pytest

from iam.generator import (
    BranchingType,
    DepartmentType,
    EnterpriseTopologyConfig,
    EnterpriseTopologyGenerator,
    NodeType,
)
from iam.parser import CapabilityModel, load_default_capability_model


@pytest.fixture
def capability_model() -> CapabilityModel:
    """Fixture providing loaded AWS CapabilityModel."""
    return load_default_capability_model()


class TestEnterpriseTopologyConfig:
    """Tests for EnterpriseTopologyConfig validation and defaults."""

    def test_default_config(self) -> None:
        cfg = EnterpriseTopologyConfig()
        assert cfg.num_nodes == 1000
        assert cfg.seed == 42
        assert cfg.account_id == "123456789012"
        assert cfg.organization_id == "o-enterprise-001"
        assert len(cfg.departments) == 6
        assert cfg.edge_density == 1.0
        assert cfg.num_pe_chains == 5
        assert cfg.include_branching_chains is True
        assert cfg.num_branching_chains == 1

    def test_custom_config(self) -> None:
        cfg = EnterpriseTopologyConfig(
            num_nodes=250,
            seed=999,
            account_id="987654321098",
            organization_id="o-custom-org",
            departments=["SecOps", "DevOps", "DataBI"],
            edge_density=2.0,
            num_pe_chains=2,
            include_branching_chains=False,
        )
        assert cfg.num_nodes == 250
        assert cfg.seed == 999
        assert cfg.account_id == "987654321098"
        assert len(cfg.departments) == 3
        assert cfg.include_branching_chains is False


class TestEnterpriseTopologyGeneration:
    """Tests for enterprise topology synthesis, departmental boundaries, and workflows."""

    def test_deterministic_seeding(self) -> None:
        cfg1 = EnterpriseTopologyConfig(num_nodes=150, seed=12345, num_pe_chains=2)
        cfg2 = EnterpriseTopologyConfig(num_nodes=150, seed=12345, num_pe_chains=2)
        cfg3 = EnterpriseTopologyConfig(num_nodes=150, seed=54321, num_pe_chains=2)

        gen1 = EnterpriseTopologyGenerator(cfg1)
        gen2 = EnterpriseTopologyGenerator(cfg2)
        gen3 = EnterpriseTopologyGenerator(cfg3)

        g1 = gen1.generate()
        g2 = gen2.generate()
        g3 = gen3.generate()

        assert g1.num_nodes == g2.num_nodes
        assert g1.num_edges == g2.num_edges
        assert set(g1._nodes.keys()) == set(g2._nodes.keys())

        # g3 with different seed should produce distinct graph structure
        assert set(g1._nodes.keys()) != set(g3._nodes.keys())

    def test_departmental_boundaries_and_subgraphs(self) -> None:
        cfg = EnterpriseTopologyConfig(num_nodes=200, seed=42, num_pe_chains=2)
        gen = EnterpriseTopologyGenerator(cfg)
        graph = gen.generate()

        # All 6 canonical departments must be represented
        depts_in_graph = {n.department for n in graph._nodes.values()}
        for dept in [
            DepartmentType.SECOPS.value,
            DepartmentType.DEVOPS.value,
            DepartmentType.DATABI.value,
            DepartmentType.QA.value,
            DepartmentType.BILLING.value,
            DepartmentType.INTERNS.value,
        ]:
            assert dept in depts_in_graph

        # Every department must have at least 1 group, 1 policy, 1 resource, 1 user, and 1 role
        for dept in cfg.departments:
            dept_nodes = [n for n in graph._nodes.values() if n.department == dept]
            types_in_dept = {n.node_type for n in dept_nodes}
            assert NodeType.GROUP in types_in_dept, f"Missing Group in {dept}"
            assert NodeType.POLICY in types_in_dept, f"Missing Policy in {dept}"
            assert NodeType.RESOURCE in types_in_dept, f"Missing Resource in {dept}"
            assert NodeType.USER in types_in_dept, f"Missing User in {dept}"
            assert NodeType.ROLE in types_in_dept, f"Missing Role in {dept}"

    def test_power_law_identity_distribution(self) -> None:
        cfg = EnterpriseTopologyConfig(num_nodes=500, seed=42, num_pe_chains=3)
        gen = EnterpriseTopologyGenerator(cfg)
        graph = gen.generate()

        stats = graph.metadata["stats"]
        admin_pct = stats["admin_percentage"]

        # Admin identities must form a small core (< 15% total across enterprise)
        assert admin_pct <= 15.0, f"Admin percentage too high for power law: {admin_pct}%"

        # SecOps must hold the concentration of admin identities
        secops_nodes = [
            n for n in graph._nodes.values() if n.department == DepartmentType.SECOPS.value
        ]
        secops_admins = [n for n in secops_nodes if n.is_admin]
        assert len(secops_admins) >= 1

        # Interns and QA must have zero or near-zero admin identities
        intern_admins = [
            n
            for n in graph._nodes.values()
            if n.department == DepartmentType.INTERNS.value and n.is_admin
        ]
        assert len(intern_admins) == 0

    def test_benign_business_workflows_and_capability_validity(
        self, capability_model: CapabilityModel
    ) -> None:
        cfg = EnterpriseTopologyConfig(num_nodes=300, seed=42, num_pe_chains=2)
        gen = EnterpriseTopologyGenerator(cfg)
        graph = gen.generate()

        # 1. CI/CD Pipeline workflow in DevOps
        cicd_policies = [
            n
            for n in graph.get_nodes_by_type(NodeType.POLICY)
            if "devops-cicd-build-pipeline" in n.id
        ]
        assert len(cicd_policies) >= 1
        cicd_pol = cicd_policies[0]
        assert cicd_pol.policy_document is not None

        # Verify all CI/CD statements pass capability model
        for stmt in cicd_pol.policy_document.statements:
            errors = capability_model.validate_statement(stmt)
            assert errors == [], f"CI/CD statement invalid: {errors}"

        # 2. Athena Analyst query workflow in DataBI
        analyst_policies = [
            n
            for n in graph.get_nodes_by_type(NodeType.POLICY)
            if "databi-athena-analyst-query" in n.id
        ]
        assert len(analyst_policies) >= 1
        analyst_pol = analyst_policies[0]
        assert analyst_pol.policy_document is not None

        # Verify all analyst statements pass capability model
        for stmt in analyst_pol.policy_document.statements:
            errors = capability_model.validate_statement(stmt)
            assert errors == [], f"Analyst query statement invalid: {errors}"

        # 3. All policies across entire graph must be capability compliant
        for pol in graph.get_nodes_by_type(NodeType.POLICY):
            if pol.policy_document:
                for stmt in pol.policy_document.statements:
                    errors = capability_model.validate_statement(stmt)
                    assert errors == [], f"Policy {pol.id} failed capability check: {errors}"


class TestPrivilegeEscalationInjection:
    """Tests for canonical PE chain injection and branching topologies."""

    def test_canonical_pe_chain_injection_and_path_lengths(self) -> None:
        cfg = EnterpriseTopologyConfig(
            num_nodes=300,
            seed=42,
            num_pe_chains=4,
            pe_chain_lengths=[2, 3, 4],
            include_branching_chains=False,
        )
        gen = EnterpriseTopologyGenerator(cfg)
        graph = gen.generate()

        injected = graph.metadata["injected_motifs"]
        assert len(injected) == 4

        nx_g = graph.nx_graph
        for m_data in injected:
            s = m_data["source_id"]
            t = m_data["target_id"]

            # 1. Path must exist
            assert nx.has_path(nx_g, s, t), f"Path does not exist for {m_data['instance_id']}"
            path = nx.shortest_path(nx_g, s, t)
            # Edge hop length is nodes - 1
            hop_len = len(path) - 1
            assert hop_len >= 2

            # 2. Counterfactual bridge check
            bu, bv, brel = m_data["bridge_relation"]
            assert nx_g.has_edge(bu, bv, key=brel)

            # Temporarily sever bridge
            edge_data = nx_g.get_edge_data(bu, bv, key=brel)
            nx_g.remove_edge(bu, bv, key=brel)
            assert not nx.has_path(nx_g, s, t), (
                f"Bridge removal did not sever path for {m_data['instance_id']}"
            )
            # Restore bridge
            nx_g.add_edge(bu, bv, key=brel, **edge_data)

    def test_branching_pe_chain_diamond_lattice(self) -> None:
        cfg = EnterpriseTopologyConfig(
            num_nodes=250,
            seed=777,
            num_pe_chains=1,
            include_branching_chains=True,
            num_branching_chains=1,
        )
        gen = EnterpriseTopologyGenerator(cfg)
        graph = gen.generate()

        branching_data = graph.metadata["branching_chains"]
        assert len(branching_data) == 1
        b_info = branching_data[0]
        assert b_info["branching_type"] == BranchingType.CONVERGENT_DIAMOND.value

        s = b_info["source_id"]
        t = b_info["target_ids"][0]
        bridges = b_info["bridge_relations"]
        assert len(bridges) == 2

        nx_g = graph.nx_graph
        # 1. Unmasked path exists
        assert nx.has_path(nx_g, s, t)

        b1_u, b1_v, b1_rel = bridges[0]
        b2_u, b2_v, b2_rel = bridges[1]

        # 2. Severing only Bridge 1 still leaves reachability via Branch 2
        d1 = nx_g.get_edge_data(b1_u, b1_v, key=b1_rel)
        nx_g.remove_edge(b1_u, b1_v, key=b1_rel)
        assert nx.has_path(nx_g, s, t)
        nx_g.add_edge(b1_u, b1_v, key=b1_rel, **d1)

        # 3. Severing only Bridge 2 still leaves reachability via Branch 1
        d2 = nx_g.get_edge_data(b2_u, b2_v, key=b2_rel)
        nx_g.remove_edge(b2_u, b2_v, key=b2_rel)
        assert nx.has_path(nx_g, s, t)
        nx_g.add_edge(b2_u, b2_v, key=b2_rel, **d2)

        # 4. Severing BOTH bridges breaks reachability
        nx_g.remove_edge(b1_u, b1_v, key=b1_rel)
        nx_g.remove_edge(b2_u, b2_v, key=b2_rel)
        assert not nx.has_path(nx_g, s, t)
        # Restore edges
        nx_g.add_edge(b1_u, b1_v, key=b1_rel, **d1)
        nx_g.add_edge(b2_u, b2_v, key=b2_rel, **d2)

    def test_false_positive_checks_for_benign_identities(self) -> None:
        """Verify that benign restricted identities cannot reach high-value admin assets."""
        cfg = EnterpriseTopologyConfig(
            num_nodes=300,
            seed=42,
            num_pe_chains=2,
            include_branching_chains=False,
        )
        gen = EnterpriseTopologyGenerator(cfg)
        graph = gen.generate()

        nx_g = graph.nx_graph
        injected_sources = {m["source_id"] for m in graph.metadata["injected_motifs"]}
        hv_targets = [n.id for n in graph.get_high_value_targets()]

        # Check restricted users in Billing and Interns who were NOT injected
        benign_restricted_users = [
            n.id
            for n in graph.get_nodes_by_type(NodeType.USER)
            if n.department in (DepartmentType.BILLING.value, DepartmentType.INTERNS.value)
            and not n.is_admin
            and n.id not in injected_sources
        ]

        assert len(benign_restricted_users) > 0
        for u_id in benign_restricted_users[:10]:
            for t_id in hv_targets:
                assert not nx.has_path(nx_g, u_id, t_id), (
                    f"False positive reachability detected: {u_id} -> {t_id}"
                )


class TestScalabilityAndPerformanceProfiling:
    """Measure generator runtime and memory scaling across graph sizes (500 to 5,000 nodes)."""

    @pytest.mark.parametrize(
        "num_nodes,max_allowed_sec",
        [
            (100, 0.5),
            (500, 1.0),
            (1000, 2.0),
            (2500, 5.0),
            (5000, 10.0),
        ],
    )
    def test_scaling_runtime_benchmarks(self, num_nodes: int, max_allowed_sec: float) -> None:
        cfg = EnterpriseTopologyConfig(
            num_nodes=num_nodes,
            seed=42,
            num_pe_chains=3,
            include_branching_chains=True,
            num_branching_chains=1,
        )
        gen = EnterpriseTopologyGenerator(cfg)

        start = time.perf_counter()
        graph = gen.generate()
        duration = time.perf_counter() - start

        assert graph.num_nodes >= int(num_nodes * 0.90)
        assert graph.num_edges > graph.num_nodes
        assert duration < max_allowed_sec, (
            f"Generation for {num_nodes} nodes took {duration:.2f}s, exceeding {max_allowed_sec}s threshold"
        )
