"""Unit and solver verification test suite for Z3 SMT/MaxSAT exact policy repair.

Tests Phase 3 Week 7 (Step 7.4):
- SMT decision variable encoding and consistency constraints (x_{scoped} => x_{keep}).
- Capability model constraint enforcement (wildcard-only actions cannot be scoped).
- Hard security reachability constraints (attack paths severed).
- Hard workload log conformance constraints (historical operations preserved).
- Exact MaxSAT optimization via Z3 Optimize().
- Optimality gap comparison: formal verification that greedy repair achieves 0.0 gap on PassRole motif.
"""

from __future__ import annotations

import networkx as nx
import z3

from iam.generator.graph import EdgeRelation, GraphEdge, GraphNode, IAMGraph, NodeType
from iam.generator.motifs import PassRoleLambdaMotif
from iam.optimizer.transformations import ScopeResourceTransformation
from iam.optimizer.z3_solver import (
    Z3CanonicalRepairSolver,
    Z3Decision,
    Z3RepairResult,
)
from iam.parser.capability import load_default_capability_model
from iam.parser.evaluator import AuthRequest
from iam.parser.schema import Effect, PolicyDocument, Statement

# ==============================================================================
# 1. SMT Decision Variable Encoding & Constraint Tests
# ==============================================================================


class TestZ3DecisionEncoding:
    """Test Z3 boolean variables, consistency, and capability constraints."""

    def test_variable_consistency_constraint(self) -> None:
        """Scoped => Keep: an action cannot be scoped if it is removed."""
        solver = z3.Solver()
        x_keep = z3.Bool("x_keep")
        x_scoped = z3.Bool("x_scoped")

        solver.add(z3.Implies(x_scoped, x_keep))
        # Contradiction: scoped=True but keep=False
        solver.add(x_scoped == True)  # noqa: E712
        solver.add(x_keep == False)  # noqa: E712

        assert solver.check() == z3.unsat

    def test_capability_constraint_enforcement_wildcard_only(self) -> None:
        """Wildcard-only actions cannot have x_scoped == True."""
        cap_model = load_default_capability_model()
        cap = cap_model.get_action_capability("iam:ListRoles")
        assert cap is not None and cap.wildcard_only is True

        solver = z3.Solver()
        x_scoped = z3.Bool("x_scoped")
        # In Z3CanonicalRepairSolver, cap constraint forces Not(x_scoped) when wildcard-only
        can_scope = not cap.wildcard_only and cap.supports_resource_scoping
        if not can_scope:
            solver.add(z3.Not(x_scoped))

        solver.add(x_scoped == True)  # noqa: E712
        assert solver.check() == z3.unsat


# ==============================================================================
# 2. Canonical PassRole + Lambda MaxSAT Solver Tests
# ==============================================================================


class TestZ3CanonicalPassRoleSolver:
    """Test exact MaxSAT solving on canonical PassRoleLambdaMotif."""

    def test_solve_passrole_lambda_optimal_scoping(self) -> None:
        """Verify Z3 exact solver finds optimal ScopeResource transformation (cost 1.0)."""
        graph = IAMGraph("z3_passrole_graph")
        motif = PassRoleLambdaMotif()
        instance = motif.inject(graph, suffix="z3")

        source_node = graph.get_node(instance.source_id)
        assert source_node is not None

        # Model realistic un-scoped wildcard policy allowing iam:PassRole on *
        source_pol = graph.get_node("policy:lambda-creator-z3")
        assert source_pol is not None
        vulnerable_doc = PolicyDocument(
            Statement=[
                Statement(Effect=Effect.ALLOW, Action=["iam:PassRole"], Resource=["*"]),
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["lambda:CreateFunction", "lambda:InvokeFunction"],
                    Resource=["arn:aws:lambda:us-east-1:123456789012:function:escalator-z3"],
                ),
            ]
        )
        graph.add_node(source_pol.model_copy(update={"policy_document": vulnerable_doc}, deep=True))

        # Legitimate workload D^{T_1}
        legit_worker_arn = "arn:aws:iam::123456789012:role/legit-worker-z3"
        workload_logs = [
            AuthRequest(
                principal_arn=source_node.arn,
                action="iam:PassRole",
                resource_arn=legit_worker_arn,
            ),
            AuthRequest(
                principal_arn=source_node.arn,
                action="lambda:CreateFunction",
                resource_arn="arn:aws:lambda:us-east-1:123456789012:function:escalator-z3",
            ),
        ]

        solver = Z3CanonicalRepairSolver()
        result = solver.solve(
            graph=graph,
            source_id=instance.source_id,
            target_id=instance.target_id,
            workload_logs=workload_logs,
        )

        assert result.is_sat is True
        assert result.status == "sat"
        assert result.total_cost == 1.0  # Exact optimal cost: 1.0 (scoping) vs 5.0 (removal)
        assert result.num_applied_transformations == 1

        pass_decision = result.decisions.get("iam:passrole")
        assert pass_decision is not None
        assert pass_decision.keep is True
        assert pass_decision.scoped is True
        assert legit_worker_arn in pass_decision.scoped_resources

        # Verify applied transformation is ScopeResourceTransformation
        trans = result.applied_transformations[0]
        assert isinstance(trans, ScopeResourceTransformation)
        assert trans.action == "iam:passrole"

        # Verify attack path severed in repaired graph
        assert result.repaired_graph is not None
        assert not nx.has_path(
            result.repaired_graph.nx_graph, instance.source_id, instance.target_id
        )

        # Verify historical workload remains authorized
        conforms, denied = solver.conformance_checker.check_conformance(
            result.repaired_graph, workload_logs
        )
        assert conforms is True
        assert len(denied) == 0


# ==============================================================================
# 3. Action Revocation Fallback & Hard UNSAT Conflict Tests
# ==============================================================================


class TestZ3ActionRevocationAndUNSAT:
    """Test wildcard-only fallback to action removal and UNSAT conflict detection."""

    def test_wildcard_only_action_triggers_action_removal(self) -> None:
        """When an action cannot be scoped in C(a, r, c), Z3 chooses removal (cost 5.0)."""
        graph = IAMGraph("action_removal_graph")
        u = GraphNode(
            id="user:dev",
            node_type=NodeType.USER,
            arn="arn:aws:iam::123456789012:user/dev",
            name="dev",
        )
        t = GraphNode(
            id="resource:vault",
            node_type=NodeType.RESOURCE,
            arn="arn:aws:kms:us-east-1:123456789012:key/vault",
            name="vault",
            is_high_value=True,
        )
        pol = GraphNode(
            id="policy:dev-pol",
            node_type=NodeType.POLICY,
            arn="arn:aws:iam::123456789012:policy/dev-pol",
            name="dev-pol",
            policy_document=PolicyDocument(
                Statement=[
                    Statement(
                        Effect=Effect.ALLOW,
                        Action=["iam:ListRoles"],
                        Resource=["*"],
                    )
                ]
            ),
        )
        graph.add_node(u)
        graph.add_node(t)
        graph.add_node(pol)
        graph.add_edge(GraphEdge(source=u.id, target=pol.id, relation=EdgeRelation.ATTACHED_WITH))
        graph.add_edge(
            GraphEdge(
                source=u.id,
                target=t.id,
                relation=EdgeRelation.ACTS_ON,
                actions=["iam:ListRoles"],
            )
        )

        solver = Z3CanonicalRepairSolver()
        # No workload logs: dev does not need iam:ListRoles
        result = solver.solve(graph=graph, source_id=u.id, target_id=t.id, workload_logs=[])

        assert result.is_sat is True
        assert result.total_cost == 5.0  # Action removal cost
        decision = result.decisions.get("iam:listroles")
        assert decision is not None
        assert decision.keep is False
        assert decision.scoped is False

    def test_conflicting_workload_and_security_constraints_unsat(self) -> None:
        """When workload requires an un-scopable action that is essential to the attack, solver returns UNSAT."""
        graph = IAMGraph("conflict_graph")
        u = GraphNode(
            id="user:dev",
            node_type=NodeType.USER,
            arn="arn:aws:iam::123456789012:user/dev",
            name="dev",
        )
        t = GraphNode(
            id="resource:vault",
            node_type=NodeType.RESOURCE,
            arn="arn:aws:kms:us-east-1:123456789012:key/vault",
            name="vault",
            is_high_value=True,
        )
        pol = GraphNode(
            id="policy:dev-pol",
            node_type=NodeType.POLICY,
            arn="arn:aws:iam::123456789012:policy/dev-pol",
            name="dev-pol",
            policy_document=PolicyDocument(
                Statement=[
                    Statement(
                        Effect=Effect.ALLOW,
                        Action=["iam:ListRoles"],
                        Resource=["*"],
                    )
                ]
            ),
        )
        graph.add_node(u)
        graph.add_node(t)
        graph.add_node(pol)
        graph.add_edge(GraphEdge(source=u.id, target=pol.id, relation=EdgeRelation.ATTACHED_WITH))
        graph.add_edge(
            GraphEdge(
                source=u.id,
                target=t.id,
                relation=EdgeRelation.ACTS_ON,
                actions=["iam:ListRoles"],
            )
        )

        # Workload requires iam:ListRoles on * (cannot be scoped, cannot be removed)
        workload = [
            AuthRequest(
                principal_arn=u.arn,
                action="iam:ListRoles",
                resource_arn="*",
            )
        ]

        solver = Z3CanonicalRepairSolver()
        result = solver.solve(graph=graph, source_id=u.id, target_id=t.id, workload_logs=workload)

        assert result.is_sat is False
        assert result.status == "unsat"


# ==============================================================================
# 4. Greedy vs. Z3 Optimality Gap Benchmarking Tests
# ==============================================================================


class TestGreedyVsZ3OptimalityGap:
    """Benchmark greedy repair against exact Z3 MaxSAT to verify optimality gap."""

    def test_optimality_gap_is_zero_on_canonical_passrole(self) -> None:
        """Formal verification: greedy repair achieves exactly 0.0 optimality gap on PassRole motif."""
        graph = IAMGraph("gap_test_graph")
        motif = PassRoleLambdaMotif()
        instance = motif.inject(graph, suffix="gap")

        source_node = graph.get_node(instance.source_id)
        assert source_node is not None

        # Model un-scoped wildcard policy allowing iam:PassRole on *
        source_pol = graph.get_node("policy:lambda-creator-gap")
        assert source_pol is not None
        vulnerable_doc = PolicyDocument(
            Statement=[
                Statement(Effect=Effect.ALLOW, Action=["iam:PassRole"], Resource=["*"]),
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["lambda:CreateFunction", "lambda:InvokeFunction"],
                    Resource=["arn:aws:lambda:us-east-1:123456789012:function:escalator-gap"],
                ),
            ]
        )
        graph.add_node(source_pol.model_copy(update={"policy_document": vulnerable_doc}, deep=True))

        # Legitimate workload
        legit_worker_arn = "arn:aws:iam::123456789012:role/legit-worker-gap"
        workload_logs = [
            AuthRequest(
                principal_arn=source_node.arn,
                action="iam:PassRole",
                resource_arn=legit_worker_arn,
            ),
            AuthRequest(
                principal_arn=source_node.arn,
                action="lambda:CreateFunction",
                resource_arn="arn:aws:lambda:us-east-1:123456789012:function:escalator-gap",
            ),
        ]

        solver = Z3CanonicalRepairSolver()
        z3_res, greedy_res, gap = solver.compare_with_greedy(
            graph=graph,
            source_id=instance.source_id,
            target_id=instance.target_id,
            workload_logs=workload_logs,
        )

        assert z3_res.is_sat is True
        assert greedy_res.is_successful is True
        assert z3_res.total_cost == 1.0
        assert greedy_res.total_repair_cost == 1.0
        # Optimality Gap: Cost(Greedy) - Cost(Z3) == 0.0
        assert gap == 0.0
        assert z3_res.optimality_gap_vs_greedy == 0.0

        # Both solvers choose ScopeResourceTransformation on PassRole
        assert len(z3_res.applied_transformations) == 1
        assert len(greedy_res.applied_transformations) == 1
        assert isinstance(z3_res.applied_transformations[0], ScopeResourceTransformation)
        assert isinstance(greedy_res.applied_transformations[0], ScopeResourceTransformation)


# ==============================================================================
# 5. Serialization & Boundary Conditions Tests
# ==============================================================================


class TestZ3SerializationAndBoundaries:
    """Test serialization, summary generation, and boundary conditions."""

    def test_already_safe_graph_returns_zero_cost(self) -> None:
        """When source cannot reach target initially, Z3 returns sat with 0.0 cost."""
        graph = IAMGraph("already_safe_graph")
        u = GraphNode(
            id="user:u1",
            node_type=NodeType.USER,
            arn="arn:aws:iam::123456789012:user/u1",
            name="u1",
        )
        t = GraphNode(
            id="resource:t1",
            node_type=NodeType.RESOURCE,
            arn="arn:aws:s3:::t1",
            name="t1",
        )
        graph.add_node(u)
        graph.add_node(t)

        solver = Z3CanonicalRepairSolver()
        result = solver.solve(graph=graph, source_id=u.id, target_id=t.id)

        assert result.is_sat is True
        assert result.total_cost == 0.0
        assert result.num_applied_transformations == 0

    def test_invalid_node_identifiers_returns_unsat(self) -> None:
        """Non-existent source or target node returns unsat."""
        graph = IAMGraph("empty_graph")
        solver = Z3CanonicalRepairSolver()
        result = solver.solve(graph=graph, source_id="user:missing", target_id="res:missing")
        assert result.is_sat is False

    def test_result_to_dict_and_summary(self) -> None:
        """Verify Z3Decision and Z3RepairResult serialization."""
        dec = Z3Decision(
            action="iam:PassRole",
            keep=True,
            scoped=True,
            cost=1.0,
            scoped_resources=["arn:aws:iam::123456789012:role/worker"],
        )
        dec_dict = dec.to_dict()
        assert dec_dict["action"] == "iam:PassRole"
        assert dec_dict["keep"] is True
        assert dec_dict["scoped"] is True
        assert dec_dict["cost"] == 1.0

        res = Z3RepairResult(
            is_sat=True,
            status="sat",
            total_cost=1.0,
            decisions={"iam:PassRole": dec},
            applied_transformations=[],
            repaired_graph=None,
            execution_time_sec=0.015,
            optimality_gap_vs_greedy=0.0,
            num_constraints=5,
            num_variables=2,
        )
        res_dict = res.to_dict()
        assert res_dict["is_sat"] is True
        assert res_dict["optimality_gap_vs_greedy"] == 0.0
        assert "SAT" in res.summary()
