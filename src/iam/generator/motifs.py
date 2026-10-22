"""Canonical Privilege Escalation (PE) motifs with exact ground truth.

Defines high-frequency AWS privilege escalation subgraph patterns (P0 required and P1 optional),
each equipped with exact path definitions, capability-compliant IAM policy ASTs,
and deterministic ground-truth reachability verification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field

from iam.generator.graph import (
    EdgeRelation,
    GraphEdge,
    GraphNode,
    IAMGraph,
    NodeType,
)
from iam.parser.capability import load_default_capability_model
from iam.parser.schema import Effect, PolicyDocument, Principal, Statement


class MotifInstance(BaseModel):
    """A concrete instance of a Privilege Escalation motif injected into an IAMGraph."""

    instance_id: str = Field(description="Unique instance identifier")
    motif_type: str = Field(description="Motif type string identifier")
    tier: str = Field(description="Priority tier: 'P0' or 'P1'")
    source_id: str = Field(description="Attacker entrypoint principal node ID")
    target_id: str = Field(description="High-value target asset or administrative node ID")
    intermediate_node_ids: list[str] = Field(
        default_factory=list, description="Intermediate nodes along the escalation chain"
    )
    path_edges: list[tuple[str, str, str]] = Field(
        default_factory=list, description="Ordered directed edges (u, v, relation) along chain"
    )
    bridge_relation: tuple[str, str, str] = Field(
        description="The critical bottleneck relation (u, v, relation) masked under Condition E"
    )
    required_actions: list[str] = Field(
        default_factory=list, description="IAM actions exercised along the chain"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional context and parameters"
    )


class GroundTruthVerification(BaseModel):
    """Result of an exact ground-truth reachability and counterfactual analysis."""

    is_valid: bool = Field(description="True if all ground truth properties hold")
    path_exists: bool = Field(description="True if path s -> t exists in full graph")
    bridge_breaks_path: bool = Field(
        description="True if removing the bridge edge severs reachability s -> t"
    )
    traversed_path: list[str] = Field(
        default_factory=list, description="Actual node sequence traversed"
    )
    details: str = Field(default="", description="Explanatory diagnostics")


class PEMotif(ABC):
    """Abstract base class for all Privilege Escalation motifs."""

    motif_type: str
    name: str
    description: str
    tier: str
    required_actions: list[str]

    @abstractmethod
    def inject(
        self,
        graph: IAMGraph,
        source_id: str | None = None,
        target_id: str | None = None,
        suffix: str = "",
        account_id: str = "123456789012",
        **_kwargs: Any,
    ) -> MotifInstance:
        """Inject the motif subgraph into the target IAMGraph.

        Creates all required principals, policies, trust edges, and resources,
        ensuring all policies are capability-model compliant.
        """
        ...

    def verify_ground_truth(
        self,
        graph: IAMGraph,
        instance: MotifInstance,
    ) -> GroundTruthVerification:
        """Verify the exact ground-truth reachability and counterfactual bridge masking.

        Ensures:
        1. A directed escalation path exists from source_id to target_id in the observed graph.
        2. Removing the designated bridge relation severs the path.
        """
        nx_g = graph.nx_graph
        s = instance.source_id
        t = instance.target_id

        # 1. Check positive reachability
        path_exists = nx.has_path(nx_g, s, t)
        traversed: list[str] = []
        if path_exists:
            traversed = list(nx.shortest_path(nx_g, s, t))

        # 2. Check counterfactual bridge removal
        bu, bv, brel = instance.bridge_relation
        bridge_present = nx_g.has_edge(bu, bv, key=brel)

        bridge_breaks = False
        if bridge_present:
            # Temporarily remove bridge edge
            edge_data = nx_g.get_edge_data(bu, bv, key=brel)
            nx_g.remove_edge(bu, bv, key=brel)
            bridge_breaks = not nx.has_path(nx_g, s, t)
            # Restore edge
            nx_g.add_edge(bu, bv, key=brel, **edge_data)

        is_valid = path_exists and bridge_breaks
        diag = (
            f"Motif '{instance.motif_type}' ({instance.instance_id}): "
            f"path_exists={path_exists}, bridge_breaks_path={bridge_breaks}"
        )

        return GroundTruthVerification(
            is_valid=is_valid,
            path_exists=path_exists,
            bridge_breaks_path=bridge_breaks,
            traversed_path=traversed,
            details=diag,
        )


# ==============================================================================
# P0 Motifs (Required)
# ==============================================================================


class PassRoleLambdaMotif(PEMotif):
    """P0 Motif 1: iam:PassRole + lambda:CreateFunction + lambda:InvokeFunction.

    Attacker creates a Lambda function with a high-privilege execution role
    and invokes it to access high-value assets.
    Bridge relation: (source, target_role, PassesTo).
    """

    motif_type = "passrole_lambda"
    name = "iam:PassRole with Lambda Execution"
    description = (
        "Principal with iam:PassRole and lambda:CreateFunction/InvokeFunction passes "
        "a high-privilege execution role to Lambda to escalate privileges."
    )
    tier = "P0"
    required_actions = ["iam:PassRole", "lambda:CreateFunction", "lambda:InvokeFunction"]

    def inject(
        self,
        graph: IAMGraph,
        source_id: str | None = None,
        target_id: str | None = None,
        suffix: str = "",
        account_id: str = "123456789012",
        **_kwargs: Any,
    ) -> MotifInstance:
        cap_model = load_default_capability_model()
        uid = suffix or "default"

        # 1. Source principal
        if source_id is None:
            source_id = f"user:dev-junior-{uid}"
            source_node = GraphNode(
                id=source_id,
                node_type=NodeType.USER,
                arn=f"arn:aws:iam::{account_id}:user/dev-junior-{uid}",
                name=f"dev-junior-{uid}",
                account_id=account_id,
                department="DevOps",
            )
            graph.add_node(source_node)
        else:
            found_s = graph.get_node(source_id)
            if found_s is None:
                raise KeyError(f"Source node '{source_id}' not found in graph.")
            source_node = found_s

        # 2. Target high-value asset
        if target_id is None:
            target_id = f"resource:vault-kms-key-{uid}"
            target_node = GraphNode(
                id=target_id,
                node_type=NodeType.RESOURCE,
                arn=f"arn:aws:kms:us-east-1:{account_id}:key/vault-master-key-{uid}",
                name=f"vault-master-key-{uid}",
                account_id=account_id,
                department="SecOps",
                is_high_value=True,
            )
            graph.add_node(target_node)
        else:
            found = graph.get_node(target_id)
            if found is None:
                raise KeyError(f"Target node '{target_id}' not found in graph.")
            target_node = found

        # 3. Intermediate high-privilege execution role
        admin_role_id = f"role:lambda-admin-exec-{uid}"
        admin_role_arn = f"arn:aws:iam::{account_id}:role/lambda-admin-exec-{uid}"
        trust_doc = PolicyDocument(
            Statement=[
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["sts:AssumeRole"],
                    Principal=Principal.model_validate({"Service": ["lambda.amazonaws.com"]}),
                )
            ]
        )
        admin_role = GraphNode(
            id=admin_role_id,
            node_type=NodeType.ROLE,
            arn=admin_role_arn,
            name=f"lambda-admin-exec-{uid}",
            account_id=account_id,
            department="SecOps",
            is_admin=True,
            trust_policy=trust_doc,
        )
        graph.add_node(admin_role)

        # 4. Target admin policy granting access to target asset
        admin_pol_id = f"policy:kms-full-access-{uid}"
        admin_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["kms:Decrypt", "kms:GenerateDataKey"],
            Resource=[target_node.arn],
        )
        assert cap_model.validate_statement(admin_stmt) == []
        admin_pol = GraphNode(
            id=admin_pol_id,
            node_type=NodeType.POLICY,
            arn=f"arn:aws:iam::{account_id}:policy/kms-full-access-{uid}",
            name=f"kms-full-access-{uid}",
            account_id=account_id,
            department="SecOps",
            is_admin=True,
            policy_document=PolicyDocument(Statement=[admin_stmt]),
        )
        graph.add_node(admin_pol)

        # Attach admin policy to admin role
        graph.add_edge(
            GraphEdge(
                source=admin_role_id,
                target=admin_pol_id,
                relation=EdgeRelation.ATTACHED_WITH,
            )
        )
        # Policy acts on target asset
        graph.add_edge(
            GraphEdge(
                source=admin_pol_id,
                target=target_id,
                relation=EdgeRelation.ACTS_ON,
                actions=["kms:Decrypt", "kms:GenerateDataKey"],
            )
        )

        # 5. Intermediate Lambda compute resource
        lambda_res_id = f"resource:function-escalator-{uid}"
        lambda_node = GraphNode(
            id=lambda_res_id,
            node_type=NodeType.RESOURCE,
            arn=f"arn:aws:lambda:us-east-1:{account_id}:function:escalator-{uid}",
            name=f"escalator-{uid}",
            account_id=account_id,
            department="DevOps",
            metadata={"execution_role": admin_role_arn},
        )
        graph.add_node(lambda_node)

        # 6. Low-privilege policy for source principal
        source_pol_id = f"policy:lambda-creator-{uid}"
        pass_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["iam:PassRole"],
            Resource=[admin_role_arn],
        )
        lambda_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["lambda:CreateFunction", "lambda:InvokeFunction"],
            Resource=[lambda_node.arn],
        )
        assert cap_model.validate_statement(pass_stmt) == []
        assert cap_model.validate_statement(lambda_stmt) == []

        source_pol = GraphNode(
            id=source_pol_id,
            node_type=NodeType.POLICY,
            arn=f"arn:aws:iam::{account_id}:policy/lambda-creator-{uid}",
            name=f"lambda-creator-{uid}",
            account_id=account_id,
            department="DevOps",
            policy_document=PolicyDocument(Statement=[pass_stmt, lambda_stmt]),
        )
        graph.add_node(source_pol)

        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=source_pol_id,
                relation=EdgeRelation.ATTACHED_WITH,
            )
        )
        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=lambda_res_id,
                relation=EdgeRelation.ACTS_ON,
                actions=["lambda:CreateFunction", "lambda:InvokeFunction"],
            )
        )

        # 7. Bridge relation: PassesTo from source to admin role
        bridge_rel = (source_id, admin_role_id, EdgeRelation.PASSES_TO.value)
        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=admin_role_id,
                relation=EdgeRelation.PASSES_TO,
                is_bridge=True,
                motif_id=f"passrole_lambda_{uid}",
                actions=["iam:PassRole"],
            )
        )

        path_edges = [
            (source_id, admin_role_id, EdgeRelation.PASSES_TO.value),
            (admin_role_id, admin_pol_id, EdgeRelation.ATTACHED_WITH.value),
            (admin_pol_id, target_id, EdgeRelation.ACTS_ON.value),
        ]

        return MotifInstance(
            instance_id=f"passrole_lambda_{uid}",
            motif_type=self.motif_type,
            tier=self.tier,
            source_id=source_id,
            target_id=target_id,
            intermediate_node_ids=[admin_role_id, admin_pol_id, lambda_res_id],
            path_edges=path_edges,
            bridge_relation=bridge_rel,
            required_actions=self.required_actions,
        )


class CreateAccessKeyMotif(PEMotif):
    """P0 Motif 2: iam:CreateAccessKey targeting an existing high-privilege user.

    Attacker creates credentials for an admin user to inherit their full permissions.
    Bridge relation: (source, target_admin_user, ActsOn).
    """

    motif_type = "create_access_key"
    name = "iam:CreateAccessKey on Admin User"
    description = (
        "Principal with iam:CreateAccessKey calls the API on a high-privilege "
        "user, generating an access key and hijacking the identity."
    )
    tier = "P0"
    required_actions = ["iam:CreateAccessKey"]

    def inject(
        self,
        graph: IAMGraph,
        source_id: str | None = None,
        target_id: str | None = None,
        suffix: str = "",
        account_id: str = "123456789012",
        **_kwargs: Any,
    ) -> MotifInstance:
        cap_model = load_default_capability_model()
        uid = suffix or "default"

        # 1. Source principal
        if source_id is None:
            source_id = f"user:contractor-{uid}"
            source_node = GraphNode(
                id=source_id,
                node_type=NodeType.USER,
                arn=f"arn:aws:iam::{account_id}:user/contractor-{uid}",
                name=f"contractor-{uid}",
                account_id=account_id,
                department="QA",
            )
            graph.add_node(source_node)
        else:
            found_s = graph.get_node(source_id)
            if found_s is None:
                raise KeyError(f"Source node '{source_id}' not found in graph.")
            source_node = found_s

        # 2. Target high-value asset
        if target_id is None:
            target_id = f"resource:s3-vault-{uid}"
            target_node = GraphNode(
                id=target_id,
                node_type=NodeType.RESOURCE,
                arn=f"arn:aws:s3:::corporate-customer-data-{uid}",
                name=f"corporate-customer-data-{uid}",
                account_id=account_id,
                department="Billing",
                is_high_value=True,
            )
            graph.add_node(target_node)
        else:
            found = graph.get_node(target_id)
            if found is None:
                raise KeyError(f"Target node '{target_id}' not found in graph.")
            target_node = found

        # 3. High-privilege admin user
        admin_user_id = f"user:secops-admin-{uid}"
        admin_user_arn = f"arn:aws:iam::{account_id}:user/secops-admin-{uid}"
        admin_user = GraphNode(
            id=admin_user_id,
            node_type=NodeType.USER,
            arn=admin_user_arn,
            name=f"secops-admin-{uid}",
            account_id=account_id,
            department="SecOps",
            is_admin=True,
        )
        graph.add_node(admin_user)

        # 4. Admin policy attached to admin user
        admin_pol_id = f"policy:s3-admin-{uid}"
        admin_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["s3:GetObject", "s3:PutObject"],
            Resource=[f"{target_node.arn}/*"],
        )
        assert cap_model.validate_statement(admin_stmt) == []
        admin_pol = GraphNode(
            id=admin_pol_id,
            node_type=NodeType.POLICY,
            arn=f"arn:aws:iam::{account_id}:policy/s3-admin-{uid}",
            name=f"s3-admin-{uid}",
            account_id=account_id,
            department="SecOps",
            is_admin=True,
            policy_document=PolicyDocument(Statement=[admin_stmt]),
        )
        graph.add_node(admin_pol)

        graph.add_edge(
            GraphEdge(
                source=admin_user_id,
                target=admin_pol_id,
                relation=EdgeRelation.ATTACHED_WITH,
            )
        )
        graph.add_edge(
            GraphEdge(
                source=admin_pol_id,
                target=target_id,
                relation=EdgeRelation.ACTS_ON,
                actions=["s3:GetObject", "s3:PutObject"],
            )
        )

        # 5. Low-privilege policy for source
        source_pol_id = f"policy:key-manager-{uid}"
        key_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["iam:CreateAccessKey"],
            Resource=[admin_user_arn],
        )
        assert cap_model.validate_statement(key_stmt) == []
        source_pol = GraphNode(
            id=source_pol_id,
            node_type=NodeType.POLICY,
            arn=f"arn:aws:iam::{account_id}:policy/key-manager-{uid}",
            name=f"key-manager-{uid}",
            account_id=account_id,
            department="QA",
            policy_document=PolicyDocument(Statement=[key_stmt]),
        )
        graph.add_node(source_pol)

        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=source_pol_id,
                relation=EdgeRelation.ATTACHED_WITH,
            )
        )

        # 6. Bridge relation: ActsOn targeting admin user
        bridge_rel = (source_id, admin_user_id, EdgeRelation.ACTS_ON.value)
        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=admin_user_id,
                relation=EdgeRelation.ACTS_ON,
                is_bridge=True,
                motif_id=f"create_access_key_{uid}",
                actions=["iam:CreateAccessKey"],
            )
        )

        path_edges = [
            (source_id, admin_user_id, EdgeRelation.ACTS_ON.value),
            (admin_user_id, admin_pol_id, EdgeRelation.ATTACHED_WITH.value),
            (admin_pol_id, target_id, EdgeRelation.ACTS_ON.value),
        ]

        return MotifInstance(
            instance_id=f"create_access_key_{uid}",
            motif_type=self.motif_type,
            tier=self.tier,
            source_id=source_id,
            target_id=target_id,
            intermediate_node_ids=[admin_user_id, admin_pol_id],
            path_edges=path_edges,
            bridge_relation=bridge_rel,
            required_actions=self.required_actions,
        )


class AttachPolicyMotif(PEMotif):
    """P0 Motif 3: iam:AttachUserPolicy / iam:AttachRolePolicy attaching AdministratorAccess.

    Attacker has permission to attach an existing administrative policy to themselves.
    Bridge relation: (source, admin_policy, ActsOn).
    """

    motif_type = "attach_policy"
    name = "iam:AttachUserPolicy with AdministratorAccess"
    description = (
        "Principal with iam:AttachUserPolicy attaches an unrestricted or high-privilege "
        "managed policy to self to achieve organization-wide privilege escalation."
    )
    tier = "P0"
    required_actions = ["iam:AttachUserPolicy"]

    def inject(
        self,
        graph: IAMGraph,
        source_id: str | None = None,
        target_id: str | None = None,
        suffix: str = "",
        account_id: str = "123456789012",
        **_kwargs: Any,
    ) -> MotifInstance:
        cap_model = load_default_capability_model()
        uid = suffix or "default"

        # 1. Source principal
        if source_id is None:
            source_id = f"user:analyst-{uid}"
            source_node = GraphNode(
                id=source_id,
                node_type=NodeType.USER,
                arn=f"arn:aws:iam::{account_id}:user/analyst-{uid}",
                name=f"analyst-{uid}",
                account_id=account_id,
                department="Data/BI",
            )
            graph.add_node(source_node)
        else:
            found_s = graph.get_node(source_id)
            if found_s is None:
                raise KeyError(f"Source node '{source_id}' not found in graph.")
            source_node = found_s

        # 2. Target high-value asset
        if target_id is None:
            target_id = f"resource:vault-kms-key-{uid}"
            target_node = GraphNode(
                id=target_id,
                node_type=NodeType.RESOURCE,
                arn=f"arn:aws:kms:us-east-1:{account_id}:key/vault-master-key-{uid}",
                name=f"vault-master-key-{uid}",
                account_id=account_id,
                department="SecOps",
                is_high_value=True,
            )
            graph.add_node(target_node)
        else:
            found_t = graph.get_node(target_id)
            if found_t is None:
                raise KeyError(f"Target node '{target_id}' not found in graph.")
            target_node = found_t

        # 3. Existing unattached admin policy
        admin_pol_id = f"policy:admin-access-{uid}"
        admin_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["s3:GetObject", "s3:PutObject", "kms:Decrypt"],
            Resource=["*"],
        )
        assert cap_model.validate_statement(admin_stmt) == []
        admin_pol = GraphNode(
            id=admin_pol_id,
            node_type=NodeType.POLICY,
            arn=f"arn:aws:iam::{account_id}:policy/AdministratorAccess-{uid}",
            name=f"AdministratorAccess-{uid}",
            account_id=account_id,
            department="SecOps",
            is_admin=True,
            policy_document=PolicyDocument(Statement=[admin_stmt]),
        )
        graph.add_node(admin_pol)

        # Policy acts on target asset
        graph.add_edge(
            GraphEdge(
                source=admin_pol_id,
                target=target_id,
                relation=EdgeRelation.ACTS_ON,
                actions=["kms:Decrypt"],
            )
        )

        # 4. Low-privilege policy allowing source to attach policies to self
        source_pol_id = f"policy:self-attacher-{uid}"
        attach_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["iam:AttachUserPolicy"],
            Resource=[source_node.arn],
        )
        assert cap_model.validate_statement(attach_stmt) == []
        source_pol = GraphNode(
            id=source_pol_id,
            node_type=NodeType.POLICY,
            arn=f"arn:aws:iam::{account_id}:policy/self-attacher-{uid}",
            name=f"self-attacher-{uid}",
            account_id=account_id,
            department="Data/BI",
            policy_document=PolicyDocument(Statement=[attach_stmt]),
        )
        graph.add_node(source_pol)

        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=source_pol_id,
                relation=EdgeRelation.ATTACHED_WITH,
            )
        )

        # 5. Bridge relation: source can act on admin policy (attach it)
        bridge_rel = (source_id, admin_pol_id, EdgeRelation.ACTS_ON.value)
        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=admin_pol_id,
                relation=EdgeRelation.ACTS_ON,
                is_bridge=True,
                motif_id=f"attach_policy_{uid}",
                actions=["iam:AttachUserPolicy"],
            )
        )

        path_edges = [
            (source_id, admin_pol_id, EdgeRelation.ACTS_ON.value),
            (admin_pol_id, target_id, EdgeRelation.ACTS_ON.value),
        ]

        return MotifInstance(
            instance_id=f"attach_policy_{uid}",
            motif_type=self.motif_type,
            tier=self.tier,
            source_id=source_id,
            target_id=target_id,
            intermediate_node_ids=[admin_pol_id],
            path_edges=path_edges,
            bridge_relation=bridge_rel,
            required_actions=self.required_actions,
        )


class AssumeRoleChainMotif(PEMotif):
    """P0 Motif 4: sts:AssumeRole multi-hop chaining across intermediate service roles.

    Attacker hops across intermediate roles R1 -> R2 -> R_admin to reach high-value asset.
    Bridge relation: intermediate link (R1, R2, AssumesRole).
    """

    motif_type = "assume_role_chain"
    name = "sts:AssumeRole Multi-Hop Chaining"
    description = (
        "Principal chains multiple role assumptions across intermediate service roles, "
        "crossing organizational trust boundaries to escalate to target admin role."
    )
    tier = "P0"
    required_actions = ["sts:AssumeRole"]

    def inject(
        self,
        graph: IAMGraph,
        source_id: str | None = None,
        target_id: str | None = None,
        suffix: str = "",
        account_id: str = "123456789012",
        num_intermediate_roles: int = 2,
        **_kwargs: Any,
    ) -> MotifInstance:
        cap_model = load_default_capability_model()
        uid = suffix or "default"

        # 1. Source principal
        if source_id is None:
            source_id = f"user:ci-runner-{uid}"
            source_node = GraphNode(
                id=source_id,
                node_type=NodeType.USER,
                arn=f"arn:aws:iam::{account_id}:user/ci-runner-{uid}",
                name=f"ci-runner-{uid}",
                account_id=account_id,
                department="DevOps",
            )
            graph.add_node(source_node)
        else:
            found_s = graph.get_node(source_id)
            if found_s is None:
                raise KeyError(f"Source node '{source_id}' not found in graph.")
            source_node = found_s

        # 2. Target high-value asset
        if target_id is None:
            target_id = f"resource:vault-kms-key-{uid}"
            target_node = GraphNode(
                id=target_id,
                node_type=NodeType.RESOURCE,
                arn=f"arn:aws:kms:us-east-1:{account_id}:key/vault-master-key-{uid}",
                name=f"vault-master-key-{uid}",
                account_id=account_id,
                department="SecOps",
                is_high_value=True,
            )
            graph.add_node(target_node)
        else:
            found = graph.get_node(target_id)
            if found is None:
                raise KeyError(f"Target node '{target_id}' not found in graph.")
            target_node = found

        # 3. Intermediate Roles
        role_nodes: list[GraphNode] = []
        if num_intermediate_roles == 2:
            r1_id = f"role:staging-worker-{uid}"
            r1_arn = f"arn:aws:iam::{account_id}:role/staging-worker-{uid}"
            r1_trust = PolicyDocument(
                Statement=[
                    Statement(
                        Effect=Effect.ALLOW,
                        Action=["sts:AssumeRole"],
                        Principal=Principal.model_validate({"AWS": [source_node.arn]}),
                    )
                ]
            )
            r1_node = GraphNode(
                id=r1_id,
                node_type=NodeType.ROLE,
                arn=r1_arn,
                name=f"staging-worker-{uid}",
                account_id=account_id,
                department="DevOps",
                trust_policy=r1_trust,
            )
            graph.add_node(r1_node)
            role_nodes.append(r1_node)

            r2_id = f"role:ops-supervisor-{uid}"
            r2_arn = f"arn:aws:iam::{account_id}:role/ops-supervisor-{uid}"
            r2_trust = PolicyDocument(
                Statement=[
                    Statement(
                        Effect=Effect.ALLOW,
                        Action=["sts:AssumeRole"],
                        Principal=Principal.model_validate({"AWS": [r1_arn]}),
                    )
                ]
            )
            r2_node = GraphNode(
                id=r2_id,
                node_type=NodeType.ROLE,
                arn=r2_arn,
                name=f"ops-supervisor-{uid}",
                account_id=account_id,
                department="DevOps",
                trust_policy=r2_trust,
            )
            graph.add_node(r2_node)
            role_nodes.append(r2_node)
        else:
            for i in range(1, max(1, num_intermediate_roles) + 1):
                r_name = f"staging-worker-{i}-{uid}"
                r_id = f"role:{r_name}"
                r_arn = f"arn:aws:iam::{account_id}:role/{r_name}"
                prev_arn = source_node.arn if i == 1 else role_nodes[-1].arn
                r_trust = PolicyDocument(
                    Statement=[
                        Statement(
                            Effect=Effect.ALLOW,
                            Action=["sts:AssumeRole"],
                            Principal=Principal.model_validate({"AWS": [prev_arn]}),
                        )
                    ]
                )
                r_node = GraphNode(
                    id=r_id,
                    node_type=NodeType.ROLE,
                    arn=r_arn,
                    name=r_name,
                    account_id=account_id,
                    department="DevOps",
                    trust_policy=r_trust,
                )
                graph.add_node(r_node)
                role_nodes.append(r_node)

        # 4. Target Admin Role
        r_admin_id = f"role:prod-superadmin-{uid}"
        r_admin_arn = f"arn:aws:iam::{account_id}:role/prod-superadmin-{uid}"
        admin_trust = PolicyDocument(
            Statement=[
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["sts:AssumeRole"],
                    Principal=Principal.model_validate({"AWS": [role_nodes[-1].arn]}),
                )
            ]
        )
        r_admin_node = GraphNode(
            id=r_admin_id,
            node_type=NodeType.ROLE,
            arn=r_admin_arn,
            name=f"prod-superadmin-{uid}",
            account_id=account_id,
            department="SecOps",
            is_admin=True,
            trust_policy=admin_trust,
        )
        graph.add_node(r_admin_node)

        # Admin Policy attached to Admin Role
        admin_pol_id = f"policy:kms-superadmin-{uid}"
        admin_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["kms:Decrypt", "kms:GenerateDataKey"],
            Resource=[target_node.arn],
        )
        assert cap_model.validate_statement(admin_stmt) == []
        admin_pol = GraphNode(
            id=admin_pol_id,
            node_type=NodeType.POLICY,
            arn=f"arn:aws:iam::{account_id}:policy/kms-superadmin-{uid}",
            name=f"kms-superadmin-{uid}",
            account_id=account_id,
            department="SecOps",
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
                target=target_id,
                relation=EdgeRelation.ACTS_ON,
                actions=["kms:Decrypt", "kms:GenerateDataKey"],
            )
        )

        # Edges for chain: source -> R1 ... -> R_admin
        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=role_nodes[0].id,
                relation=EdgeRelation.ASSUMES_ROLE,
                actions=["sts:AssumeRole"],
            )
        )
        path_edges: list[tuple[str, str, str]] = [
            (source_id, role_nodes[0].id, EdgeRelation.ASSUMES_ROLE.value)
        ]

        if len(role_nodes) == 1:
            bridge_rel = (role_nodes[0].id, r_admin_id, EdgeRelation.ASSUMES_ROLE.value)
            graph.add_edge(
                GraphEdge(
                    source=role_nodes[0].id,
                    target=r_admin_id,
                    relation=EdgeRelation.ASSUMES_ROLE,
                    is_bridge=True,
                    motif_id=f"assume_role_chain_{uid}",
                    actions=["sts:AssumeRole"],
                )
            )
            path_edges.append((role_nodes[0].id, r_admin_id, EdgeRelation.ASSUMES_ROLE.value))
        else:
            bridge_rel = (role_nodes[0].id, role_nodes[1].id, EdgeRelation.ASSUMES_ROLE.value)
            for idx in range(len(role_nodes) - 1):
                is_br = idx == 0
                graph.add_edge(
                    GraphEdge(
                        source=role_nodes[idx].id,
                        target=role_nodes[idx + 1].id,
                        relation=EdgeRelation.ASSUMES_ROLE,
                        is_bridge=is_br,
                        motif_id=f"assume_role_chain_{uid}" if is_br else None,
                        actions=["sts:AssumeRole"],
                    )
                )
                path_edges.append(
                    (role_nodes[idx].id, role_nodes[idx + 1].id, EdgeRelation.ASSUMES_ROLE.value)
                )

            graph.add_edge(
                GraphEdge(
                    source=role_nodes[-1].id,
                    target=r_admin_id,
                    relation=EdgeRelation.ASSUMES_ROLE,
                    actions=["sts:AssumeRole"],
                )
            )
            path_edges.append((role_nodes[-1].id, r_admin_id, EdgeRelation.ASSUMES_ROLE.value))

        path_edges.extend(
            [
                (r_admin_id, admin_pol_id, EdgeRelation.ATTACHED_WITH.value),
                (admin_pol_id, target_id, EdgeRelation.ACTS_ON.value),
            ]
        )

        intermediate_ids = [r.id for r in role_nodes] + [r_admin_id, admin_pol_id]

        return MotifInstance(
            instance_id=f"assume_role_chain_{uid}",
            motif_type=self.motif_type,
            tier=self.tier,
            source_id=source_id,
            target_id=target_id,
            intermediate_node_ids=intermediate_ids,
            path_edges=path_edges,
            bridge_relation=bridge_rel,
            required_actions=self.required_actions,
        )


# ==============================================================================
# P1 Motifs (Optional / Advanced)
# ==============================================================================


class PassRoleEC2Motif(PEMotif):
    """P1 Motif 5: iam:PassRole + ec2:RunInstances (instance profile elevation).

    Attacker launches an EC2 instance with an instance profile containing an admin role.
    Bridge relation: (source, admin_role, PassesTo).
    """

    motif_type = "passrole_ec2"
    name = "iam:PassRole with EC2 RunInstances"
    description = (
        "Principal launches an EC2 instance associated with a high-privilege instance profile "
        "and uses the instance metadata service to harvest credentials."
    )
    tier = "P1"
    required_actions = ["iam:PassRole", "ec2:RunInstances"]

    def inject(
        self,
        graph: IAMGraph,
        source_id: str | None = None,
        target_id: str | None = None,
        suffix: str = "",
        account_id: str = "123456789012",
        **_kwargs: Any,
    ) -> MotifInstance:
        cap_model = load_default_capability_model()
        uid = suffix or "default"

        # 1. Source principal
        if source_id is None:
            source_id = f"user:dev-intern-{uid}"
            source_node = GraphNode(
                id=source_id,
                node_type=NodeType.USER,
                arn=f"arn:aws:iam::{account_id}:user/dev-intern-{uid}",
                name=f"dev-intern-{uid}",
                account_id=account_id,
                department="DevOps",
            )
            graph.add_node(source_node)
        else:
            found_s = graph.get_node(source_id)
            if found_s is None:
                raise KeyError(f"Source node '{source_id}' not found in graph.")
            source_node = found_s

        # 2. Target high-value asset
        if target_id is None:
            target_id = f"resource:s3-confidential-{uid}"
            target_node = GraphNode(
                id=target_id,
                node_type=NodeType.RESOURCE,
                arn=f"arn:aws:s3:::confidential-ip-repo-{uid}",
                name=f"confidential-ip-repo-{uid}",
                account_id=account_id,
                department="SecOps",
                is_high_value=True,
            )
            graph.add_node(target_node)
        else:
            found = graph.get_node(target_id)
            if found is None:
                raise KeyError(f"Target node '{target_id}' not found in graph.")
            target_node = found

        # 3. High-privilege role
        admin_role_id = f"role:ec2-admin-role-{uid}"
        admin_role_arn = f"arn:aws:iam::{account_id}:role/ec2-admin-role-{uid}"
        ec2_trust_doc = PolicyDocument(
            Statement=[
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["sts:AssumeRole"],
                    Principal=Principal.model_validate({"Service": ["ec2.amazonaws.com"]}),
                )
            ]
        )
        admin_role = GraphNode(
            id=admin_role_id,
            node_type=NodeType.ROLE,
            arn=admin_role_arn,
            name=f"ec2-admin-role-{uid}",
            account_id=account_id,
            department="SecOps",
            is_admin=True,
            trust_policy=ec2_trust_doc,
        )
        graph.add_node(admin_role)

        # 4. Admin policy
        admin_pol_id = f"policy:s3-vault-access-{uid}"
        admin_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["s3:GetObject", "s3:PutObject"],
            Resource=[f"{target_node.arn}/*"],
        )
        assert cap_model.validate_statement(admin_stmt) == []
        admin_pol = GraphNode(
            id=admin_pol_id,
            node_type=NodeType.POLICY,
            arn=f"arn:aws:iam::{account_id}:policy/s3-vault-access-{uid}",
            name=f"s3-vault-access-{uid}",
            account_id=account_id,
            department="SecOps",
            is_admin=True,
            policy_document=PolicyDocument(Statement=[admin_stmt]),
        )
        graph.add_node(admin_pol)

        graph.add_edge(
            GraphEdge(
                source=admin_role_id,
                target=admin_pol_id,
                relation=EdgeRelation.ATTACHED_WITH,
            )
        )
        graph.add_edge(
            GraphEdge(
                source=admin_pol_id,
                target=target_id,
                relation=EdgeRelation.ACTS_ON,
                actions=["s3:GetObject"],
            )
        )

        # 5. EC2 compute resource
        ec2_res_id = f"resource:ec2-instance-{uid}"
        ec2_node = GraphNode(
            id=ec2_res_id,
            node_type=NodeType.RESOURCE,
            arn=f"arn:aws:ec2:us-east-1:{account_id}:instance/i-escalator-{uid}",
            name=f"i-escalator-{uid}",
            account_id=account_id,
            department="DevOps",
            metadata={"instance_profile_role": admin_role_arn},
        )
        graph.add_node(ec2_node)

        # 6. Source policy
        source_pol_id = f"policy:ec2-launcher-{uid}"
        pass_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["iam:PassRole"],
            Resource=[admin_role_arn],
        )
        run_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["ec2:RunInstances"],
            Resource=["*"],
        )
        assert cap_model.validate_statement(pass_stmt) == []
        assert cap_model.validate_statement(run_stmt) == []
        source_pol = GraphNode(
            id=source_pol_id,
            node_type=NodeType.POLICY,
            arn=f"arn:aws:iam::{account_id}:policy/ec2-launcher-{uid}",
            name=f"ec2-launcher-{uid}",
            account_id=account_id,
            department="DevOps",
            policy_document=PolicyDocument(Statement=[pass_stmt, run_stmt]),
        )
        graph.add_node(source_pol)

        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=source_pol_id,
                relation=EdgeRelation.ATTACHED_WITH,
            )
        )
        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=ec2_res_id,
                relation=EdgeRelation.ACTS_ON,
                actions=["ec2:RunInstances"],
            )
        )

        # 7. Bridge relation: PassesTo from source to admin role
        bridge_rel = (source_id, admin_role_id, EdgeRelation.PASSES_TO.value)
        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=admin_role_id,
                relation=EdgeRelation.PASSES_TO,
                is_bridge=True,
                motif_id=f"passrole_ec2_{uid}",
                actions=["iam:PassRole"],
            )
        )

        path_edges = [
            (source_id, admin_role_id, EdgeRelation.PASSES_TO.value),
            (admin_role_id, admin_pol_id, EdgeRelation.ATTACHED_WITH.value),
            (admin_pol_id, target_id, EdgeRelation.ACTS_ON.value),
        ]

        return MotifInstance(
            instance_id=f"passrole_ec2_{uid}",
            motif_type=self.motif_type,
            tier=self.tier,
            source_id=source_id,
            target_id=target_id,
            intermediate_node_ids=[admin_role_id, admin_pol_id, ec2_res_id],
            path_edges=path_edges,
            bridge_relation=bridge_rel,
            required_actions=self.required_actions,
        )


class SetDefaultPolicyVersionMotif(PEMotif):
    """P1 Motif 6: iam:SetDefaultPolicyVersion on customer managed policy.

    Attacker rolls back an attached policy to a dormant administrative version.
    Bridge relation: (source, customer_policy, ActsOn).
    """

    motif_type = "set_default_policy_version"
    name = "iam:SetDefaultPolicyVersion Policy Rollback"
    description = (
        "Principal with iam:SetDefaultPolicyVersion switches active default version "
        "of a customer managed policy to a dormant administrative version."
    )
    tier = "P1"
    required_actions = ["iam:SetDefaultPolicyVersion"]

    def inject(
        self,
        graph: IAMGraph,
        source_id: str | None = None,
        target_id: str | None = None,
        suffix: str = "",
        account_id: str = "123456789012",
        **_kwargs: Any,
    ) -> MotifInstance:
        cap_model = load_default_capability_model()
        uid = suffix or "default"

        # 1. Source principal
        if source_id is None:
            source_id = f"user:developer-{uid}"
            source_node = GraphNode(
                id=source_id,
                node_type=NodeType.USER,
                arn=f"arn:aws:iam::{account_id}:user/developer-{uid}",
                name=f"developer-{uid}",
                account_id=account_id,
                department="DevOps",
            )
            graph.add_node(source_node)
        else:
            found_s = graph.get_node(source_id)
            if found_s is None:
                raise KeyError(f"Source node '{source_id}' not found in graph.")
            source_node = found_s

        # 2. Target high-value asset
        if target_id is None:
            target_id = f"resource:secrets-s3-{uid}"
            target_node = GraphNode(
                id=target_id,
                node_type=NodeType.RESOURCE,
                arn=f"arn:aws:s3:::production-secrets-{uid}",
                name=f"production-secrets-{uid}",
                account_id=account_id,
                department="SecOps",
                is_high_value=True,
            )
            graph.add_node(target_node)
        else:
            found = graph.get_node(target_id)
            if found is None:
                raise KeyError(f"Target node '{target_id}' not found in graph.")
            target_node = found

        # 3. Customer managed policy (currently restrictive, dormant admin version)
        cust_pol_id = f"policy:customer-service-{uid}"
        cust_pol_arn = f"arn:aws:iam::{account_id}:policy/customer-service-{uid}"
        dormant_admin_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["s3:GetObject"],
            Resource=[f"{target_node.arn}/*"],
        )
        assert cap_model.validate_statement(dormant_admin_stmt) == []
        cust_pol = GraphNode(
            id=cust_pol_id,
            node_type=NodeType.POLICY,
            arn=cust_pol_arn,
            name=f"customer-service-{uid}",
            account_id=account_id,
            department="DevOps",
            policy_document=PolicyDocument(Statement=[dormant_admin_stmt]),
            metadata={"default_version": "v2", "dormant_admin_version": "v1"},
        )
        graph.add_node(cust_pol)

        # Customer policy acts on target asset (dormant link activated upon rollback)
        graph.add_edge(
            GraphEdge(
                source=cust_pol_id,
                target=target_id,
                relation=EdgeRelation.ACTS_ON,
                actions=["s3:GetObject"],
            )
        )

        # 4. Source policy permitting SetDefaultPolicyVersion
        source_pol_id = f"policy:version-switcher-{uid}"
        version_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["iam:SetDefaultPolicyVersion"],
            Resource=[cust_pol_arn],
        )
        assert cap_model.validate_statement(version_stmt) == []
        source_pol = GraphNode(
            id=source_pol_id,
            node_type=NodeType.POLICY,
            arn=f"arn:aws:iam::{account_id}:policy/version-switcher-{uid}",
            name=f"version-switcher-{uid}",
            account_id=account_id,
            department="DevOps",
            policy_document=PolicyDocument(Statement=[version_stmt]),
        )
        graph.add_node(source_pol)

        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=source_pol_id,
                relation=EdgeRelation.ATTACHED_WITH,
            )
        )

        # 5. Bridge relation: source acts on customer policy via SetDefaultPolicyVersion
        bridge_rel = (source_id, cust_pol_id, EdgeRelation.ACTS_ON.value)
        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=cust_pol_id,
                relation=EdgeRelation.ACTS_ON,
                is_bridge=True,
                motif_id=f"set_default_policy_version_{uid}",
                actions=["iam:SetDefaultPolicyVersion"],
            )
        )

        path_edges = [
            (source_id, cust_pol_id, EdgeRelation.ACTS_ON.value),
            (cust_pol_id, target_id, EdgeRelation.ACTS_ON.value),
        ]

        return MotifInstance(
            instance_id=f"set_default_policy_version_{uid}",
            motif_type=self.motif_type,
            tier=self.tier,
            source_id=source_id,
            target_id=target_id,
            intermediate_node_ids=[cust_pol_id],
            path_edges=path_edges,
            bridge_relation=bridge_rel,
            required_actions=self.required_actions,
        )


class GlueDevEndpointMotif(PEMotif):
    """Tier 2 Motif 7: iam:PassRole + glue:CreateDevEndpoint.

    Attacker creates an AWS Glue development endpoint and passes an existing high-privilege
    Glue service role to it, executing arbitrary analytical code with that role's credentials.
    Bridge relation: (source, admin_role, PassesTo).
    """

    motif_type = "passrole_glue"
    name = "iam:PassRole with Glue CreateDevEndpoint"
    description = (
        "Principal creates an AWS Glue development endpoint associated with a high-privilege "
        "Glue service role and executes scripts in the endpoint context."
    )
    tier = "P1"
    required_actions = ["iam:PassRole", "glue:CreateDevEndpoint"]

    def inject(
        self,
        graph: IAMGraph,
        source_id: str | None = None,
        target_id: str | None = None,
        suffix: str = "",
        account_id: str = "123456789012",
        **_kwargs: Any,
    ) -> MotifInstance:
        cap_model = load_default_capability_model()
        uid = suffix or "default"

        # 1. Source principal
        if source_id is None:
            source_id = f"user:glue-dev-{uid}"
            source_node = GraphNode(
                id=source_id,
                node_type=NodeType.USER,
                arn=f"arn:aws:iam::{account_id}:user/glue-dev-{uid}",
                name=f"glue-dev-{uid}",
                account_id=account_id,
                department="DataBI",
            )
            graph.add_node(source_node)
        else:
            found_s = graph.get_node(source_id)
            if found_s is None:
                raise KeyError(f"Source node '{source_id}' not found in graph.")
            source_node = found_s

        # 2. Target high-value asset
        if target_id is None:
            target_id = f"resource:s3-lake-{uid}"
            target_node = GraphNode(
                id=target_id,
                node_type=NodeType.RESOURCE,
                arn=f"arn:aws:s3:::enterprise-data-lake-{uid}",
                name=f"enterprise-data-lake-{uid}",
                account_id=account_id,
                department="DataBI",
                is_high_value=True,
            )
            graph.add_node(target_node)
        else:
            found = graph.get_node(target_id)
            if found is None:
                raise KeyError(f"Target node '{target_id}' not found in graph.")
            target_node = found

        # 3. High-privilege Glue role
        admin_role_id = f"role:glue-admin-role-{uid}"
        admin_role_arn = f"arn:aws:iam::{account_id}:role/glue-admin-role-{uid}"
        glue_trust_doc = PolicyDocument(
            Statement=[
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["sts:AssumeRole"],
                    Principal=Principal.model_validate({"Service": ["glue.amazonaws.com"]}),
                )
            ]
        )
        admin_role = GraphNode(
            id=admin_role_id,
            node_type=NodeType.ROLE,
            arn=admin_role_arn,
            name=f"glue-admin-role-{uid}",
            account_id=account_id,
            department="DataBI",
            is_admin=True,
            trust_policy=glue_trust_doc,
        )
        graph.add_node(admin_role)

        # 4. Admin policy granting target access
        admin_pol_id = f"policy:glue-lake-access-{uid}"
        admin_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["s3:GetObject", "s3:PutObject"],
            Resource=[f"{target_node.arn}/*"],
        )
        assert cap_model.validate_statement(admin_stmt) == []
        admin_pol = GraphNode(
            id=admin_pol_id,
            node_type=NodeType.POLICY,
            arn=f"arn:aws:iam::{account_id}:policy/glue-lake-access-{uid}",
            name=f"glue-lake-access-{uid}",
            account_id=account_id,
            department="DataBI",
            is_admin=True,
            policy_document=PolicyDocument(Statement=[admin_stmt]),
        )
        graph.add_node(admin_pol)

        graph.add_edge(
            GraphEdge(
                source=admin_role_id,
                target=admin_pol_id,
                relation=EdgeRelation.ATTACHED_WITH,
            )
        )
        graph.add_edge(
            GraphEdge(
                source=admin_pol_id,
                target=target_id,
                relation=EdgeRelation.ACTS_ON,
                actions=["s3:GetObject"],
            )
        )

        # 5. Glue DevEndpoint resource
        glue_res_id = f"resource:glue-devendpoint-{uid}"
        glue_node = GraphNode(
            id=glue_res_id,
            node_type=NodeType.RESOURCE,
            arn=f"arn:aws:glue:us-east-1:{account_id}:devendpoint/endpoint-{uid}",
            name=f"endpoint-{uid}",
            account_id=account_id,
            department="DataBI",
            metadata={"dev_endpoint_role": admin_role_arn},
        )
        graph.add_node(glue_node)

        # 6. Source policy granting passrole and createdevendpoint
        source_pol_id = f"policy:glue-launcher-{uid}"
        pass_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["iam:PassRole"],
            Resource=[admin_role_arn],
        )
        create_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["glue:CreateDevEndpoint"],
            Resource=[f"arn:aws:glue:us-east-1:{account_id}:devendpoint/*"],
        )
        assert cap_model.validate_statement(pass_stmt) == []
        assert cap_model.validate_statement(create_stmt) == []
        source_pol = GraphNode(
            id=source_pol_id,
            node_type=NodeType.POLICY,
            arn=f"arn:aws:iam::{account_id}:policy/glue-launcher-{uid}",
            name=f"glue-launcher-{uid}",
            account_id=account_id,
            department="DataBI",
            policy_document=PolicyDocument(Statement=[pass_stmt, create_stmt]),
        )
        graph.add_node(source_pol)

        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=source_pol_id,
                relation=EdgeRelation.ATTACHED_WITH,
            )
        )
        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=glue_res_id,
                relation=EdgeRelation.ACTS_ON,
                actions=["glue:CreateDevEndpoint"],
            )
        )

        # 7. Bridge relation: PassesTo from source to admin role
        bridge_rel = (source_id, admin_role_id, EdgeRelation.PASSES_TO.value)
        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=admin_role_id,
                relation=EdgeRelation.PASSES_TO,
                is_bridge=True,
                motif_id=f"passrole_glue_{uid}",
                actions=["iam:PassRole"],
            )
        )

        path_edges = [
            (source_id, admin_role_id, EdgeRelation.PASSES_TO.value),
            (admin_role_id, admin_pol_id, EdgeRelation.ATTACHED_WITH.value),
            (admin_pol_id, target_id, EdgeRelation.ACTS_ON.value),
        ]

        return MotifInstance(
            instance_id=f"passrole_glue_{uid}",
            motif_type=self.motif_type,
            tier=self.tier,
            source_id=source_id,
            target_id=target_id,
            intermediate_node_ids=[admin_role_id, admin_pol_id, glue_res_id],
            path_edges=path_edges,
            bridge_relation=bridge_rel,
            required_actions=self.required_actions,
        )


class CodeBuildProjectMotif(PEMotif):
    """Tier 2 Motif 8: iam:PassRole + codebuild:StartBuild.

    Attacker triggers a CodeBuild project build passing a high-privilege service role,
    executing arbitrary build commands that access sensitive target resources.
    Bridge relation: (source, admin_role, PassesTo).
    """

    motif_type = "passrole_codebuild"
    name = "iam:PassRole with CodeBuild StartBuild"
    description = (
        "Principal triggers a CodeBuild execution associated with a high-privilege service role "
        "and executes arbitrary commands within the build container."
    )
    tier = "P1"
    required_actions = ["iam:PassRole", "codebuild:StartBuild"]

    def inject(
        self,
        graph: IAMGraph,
        source_id: str | None = None,
        target_id: str | None = None,
        suffix: str = "",
        account_id: str = "123456789012",
        **_kwargs: Any,
    ) -> MotifInstance:
        cap_model = load_default_capability_model()
        uid = suffix or "default"

        # 1. Source principal
        if source_id is None:
            source_id = f"user:build-dev-{uid}"
            source_node = GraphNode(
                id=source_id,
                node_type=NodeType.USER,
                arn=f"arn:aws:iam::{account_id}:user/build-dev-{uid}",
                name=f"build-dev-{uid}",
                account_id=account_id,
                department="DevOps",
            )
            graph.add_node(source_node)
        else:
            found_s = graph.get_node(source_id)
            if found_s is None:
                raise KeyError(f"Source node '{source_id}' not found in graph.")
            source_node = found_s

        # 2. Target high-value asset
        if target_id is None:
            target_id = f"resource:s3-vault-{uid}"
            target_node = GraphNode(
                id=target_id,
                node_type=NodeType.RESOURCE,
                arn=f"arn:aws:s3:::ci-secrets-vault-{uid}",
                name=f"ci-secrets-vault-{uid}",
                account_id=account_id,
                department="SecOps",
                is_high_value=True,
            )
            graph.add_node(target_node)
        else:
            found = graph.get_node(target_id)
            if found is None:
                raise KeyError(f"Target node '{target_id}' not found in graph.")
            target_node = found

        # 3. CodeBuild admin service role
        admin_role_id = f"role:codebuild-service-role-{uid}"
        admin_role_arn = f"arn:aws:iam::{account_id}:role/codebuild-service-role-{uid}"
        cb_trust_doc = PolicyDocument(
            Statement=[
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["sts:AssumeRole"],
                    Principal=Principal.model_validate({"Service": ["codebuild.amazonaws.com"]}),
                )
            ]
        )
        admin_role = GraphNode(
            id=admin_role_id,
            node_type=NodeType.ROLE,
            arn=admin_role_arn,
            name=f"codebuild-service-role-{uid}",
            account_id=account_id,
            department="DevOps",
            is_admin=True,
            trust_policy=cb_trust_doc,
        )
        graph.add_node(admin_role)

        # 4. Admin policy
        admin_pol_id = f"policy:codebuild-vault-access-{uid}"
        admin_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["s3:GetObject", "s3:PutObject"],
            Resource=[f"{target_node.arn}/*"],
        )
        assert cap_model.validate_statement(admin_stmt) == []
        admin_pol = GraphNode(
            id=admin_pol_id,
            node_type=NodeType.POLICY,
            arn=f"arn:aws:iam::{account_id}:policy/codebuild-vault-access-{uid}",
            name=f"codebuild-vault-access-{uid}",
            account_id=account_id,
            department="SecOps",
            is_admin=True,
            policy_document=PolicyDocument(Statement=[admin_stmt]),
        )
        graph.add_node(admin_pol)

        graph.add_edge(
            GraphEdge(
                source=admin_role_id,
                target=admin_pol_id,
                relation=EdgeRelation.ATTACHED_WITH,
            )
        )
        graph.add_edge(
            GraphEdge(
                source=admin_pol_id,
                target=target_id,
                relation=EdgeRelation.ACTS_ON,
                actions=["s3:GetObject"],
            )
        )

        # 5. CodeBuild project resource
        cb_res_id = f"resource:codebuild-project-{uid}"
        cb_node = GraphNode(
            id=cb_res_id,
            node_type=NodeType.RESOURCE,
            arn=f"arn:aws:codebuild:us-east-1:{account_id}:project/build-project-{uid}",
            name=f"build-project-{uid}",
            account_id=account_id,
            department="DevOps",
            metadata={"service_role": admin_role_arn},
        )
        graph.add_node(cb_node)

        # 6. Source policy granting passrole and startbuild
        source_pol_id = f"policy:codebuild-trigger-{uid}"
        pass_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["iam:PassRole"],
            Resource=[admin_role_arn],
        )
        start_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["codebuild:StartBuild"],
            Resource=[f"arn:aws:codebuild:us-east-1:{account_id}:project/*"],
        )
        assert cap_model.validate_statement(pass_stmt) == []
        assert cap_model.validate_statement(start_stmt) == []
        source_pol = GraphNode(
            id=source_pol_id,
            node_type=NodeType.POLICY,
            arn=f"arn:aws:iam::{account_id}:policy/codebuild-trigger-{uid}",
            name=f"codebuild-trigger-{uid}",
            account_id=account_id,
            department="DevOps",
            policy_document=PolicyDocument(Statement=[pass_stmt, start_stmt]),
        )
        graph.add_node(source_pol)

        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=source_pol_id,
                relation=EdgeRelation.ATTACHED_WITH,
            )
        )
        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=cb_res_id,
                relation=EdgeRelation.ACTS_ON,
                actions=["codebuild:StartBuild"],
            )
        )

        # 7. Bridge relation: PassesTo from source to admin role
        bridge_rel = (source_id, admin_role_id, EdgeRelation.PASSES_TO.value)
        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=admin_role_id,
                relation=EdgeRelation.PASSES_TO,
                is_bridge=True,
                motif_id=f"passrole_codebuild_{uid}",
                actions=["iam:PassRole"],
            )
        )

        path_edges = [
            (source_id, admin_role_id, EdgeRelation.PASSES_TO.value),
            (admin_role_id, admin_pol_id, EdgeRelation.ATTACHED_WITH.value),
            (admin_pol_id, target_id, EdgeRelation.ACTS_ON.value),
        ]

        return MotifInstance(
            instance_id=f"passrole_codebuild_{uid}",
            motif_type=self.motif_type,
            tier=self.tier,
            source_id=source_id,
            target_id=target_id,
            intermediate_node_ids=[admin_role_id, admin_pol_id, cb_res_id],
            path_edges=path_edges,
            bridge_relation=bridge_rel,
            required_actions=self.required_actions,
        )


class SageMakerNotebookMotif(PEMotif):
    """Tier 2 Motif 9: iam:PassRole + sagemaker:CreateNotebookInstance.

    Attacker creates a SageMaker Jupyter notebook passing a privileged execution role,
    accessing root bash shells within the notebook container to pivot to target resources.
    Bridge relation: (source, admin_role, PassesTo).
    """

    motif_type = "passrole_sagemaker"
    name = "iam:PassRole with SageMaker CreateNotebookInstance"
    description = (
        "Principal launches a SageMaker notebook instance associated with an administrative role "
        "and executes commands via the interactive terminal to harvest credentials."
    )
    tier = "P1"
    required_actions = ["iam:PassRole", "sagemaker:CreateNotebookInstance"]

    def inject(
        self,
        graph: IAMGraph,
        source_id: str | None = None,
        target_id: str | None = None,
        suffix: str = "",
        account_id: str = "123456789012",
        **_kwargs: Any,
    ) -> MotifInstance:
        cap_model = load_default_capability_model()
        uid = suffix or "default"

        # 1. Source principal
        if source_id is None:
            source_id = f"user:ml-researcher-{uid}"
            source_node = GraphNode(
                id=source_id,
                node_type=NodeType.USER,
                arn=f"arn:aws:iam::{account_id}:user/ml-researcher-{uid}",
                name=f"ml-researcher-{uid}",
                account_id=account_id,
                department="DataBI",
            )
            graph.add_node(source_node)
        else:
            found_s = graph.get_node(source_id)
            if found_s is None:
                raise KeyError(f"Source node '{source_id}' not found in graph.")
            source_node = found_s

        # 2. Target high-value asset
        if target_id is None:
            target_id = f"resource:s3-models-{uid}"
            target_node = GraphNode(
                id=target_id,
                node_type=NodeType.RESOURCE,
                arn=f"arn:aws:s3:::production-ml-models-{uid}",
                name=f"production-ml-models-{uid}",
                account_id=account_id,
                department="DataBI",
                is_high_value=True,
            )
            graph.add_node(target_node)
        else:
            found = graph.get_node(target_id)
            if found is None:
                raise KeyError(f"Target node '{target_id}' not found in graph.")
            target_node = found

        # 3. High-privilege SageMaker execution role
        admin_role_id = f"role:sagemaker-exec-role-{uid}"
        admin_role_arn = f"arn:aws:iam::{account_id}:role/sagemaker-exec-role-{uid}"
        sm_trust_doc = PolicyDocument(
            Statement=[
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["sts:AssumeRole"],
                    Principal=Principal.model_validate({"Service": ["sagemaker.amazonaws.com"]}),
                )
            ]
        )
        admin_role = GraphNode(
            id=admin_role_id,
            node_type=NodeType.ROLE,
            arn=admin_role_arn,
            name=f"sagemaker-exec-role-{uid}",
            account_id=account_id,
            department="DataBI",
            is_admin=True,
            trust_policy=sm_trust_doc,
        )
        graph.add_node(admin_role)

        # 4. Admin policy granting target access
        admin_pol_id = f"policy:sagemaker-vault-access-{uid}"
        admin_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["s3:GetObject", "s3:PutObject"],
            Resource=[f"{target_node.arn}/*"],
        )
        assert cap_model.validate_statement(admin_stmt) == []
        admin_pol = GraphNode(
            id=admin_pol_id,
            node_type=NodeType.POLICY,
            arn=f"arn:aws:iam::{account_id}:policy/sagemaker-vault-access-{uid}",
            name=f"sagemaker-vault-access-{uid}",
            account_id=account_id,
            department="DataBI",
            is_admin=True,
            policy_document=PolicyDocument(Statement=[admin_stmt]),
        )
        graph.add_node(admin_pol)

        graph.add_edge(
            GraphEdge(
                source=admin_role_id,
                target=admin_pol_id,
                relation=EdgeRelation.ATTACHED_WITH,
            )
        )
        graph.add_edge(
            GraphEdge(
                source=admin_pol_id,
                target=target_id,
                relation=EdgeRelation.ACTS_ON,
                actions=["s3:GetObject"],
            )
        )

        # 5. SageMaker notebook resource
        sm_res_id = f"resource:sagemaker-notebook-{uid}"
        sm_node = GraphNode(
            id=sm_res_id,
            node_type=NodeType.RESOURCE,
            arn=f"arn:aws:sagemaker:us-east-1:{account_id}:notebook-instance/notebook-{uid}",
            name=f"notebook-{uid}",
            account_id=account_id,
            department="DataBI",
            metadata={"execution_role": admin_role_arn},
        )
        graph.add_node(sm_node)

        # 6. Source policy granting passrole and createnotebookinstance
        source_pol_id = f"policy:sagemaker-launcher-{uid}"
        pass_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["iam:PassRole"],
            Resource=[admin_role_arn],
        )
        create_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["sagemaker:CreateNotebookInstance"],
            Resource=[f"arn:aws:sagemaker:us-east-1:{account_id}:notebook-instance/*"],
        )
        assert cap_model.validate_statement(pass_stmt) == []
        assert cap_model.validate_statement(create_stmt) == []
        source_pol = GraphNode(
            id=source_pol_id,
            node_type=NodeType.POLICY,
            arn=f"arn:aws:iam::{account_id}:policy/sagemaker-launcher-{uid}",
            name=f"sagemaker-launcher-{uid}",
            account_id=account_id,
            department="DataBI",
            policy_document=PolicyDocument(Statement=[pass_stmt, create_stmt]),
        )
        graph.add_node(source_pol)

        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=source_pol_id,
                relation=EdgeRelation.ATTACHED_WITH,
            )
        )
        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=sm_res_id,
                relation=EdgeRelation.ACTS_ON,
                actions=["sagemaker:CreateNotebookInstance"],
            )
        )

        # 7. Bridge relation: PassesTo from source to admin role
        bridge_rel = (source_id, admin_role_id, EdgeRelation.PASSES_TO.value)
        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=admin_role_id,
                relation=EdgeRelation.PASSES_TO,
                is_bridge=True,
                motif_id=f"passrole_sagemaker_{uid}",
                actions=["iam:PassRole"],
            )
        )

        path_edges = [
            (source_id, admin_role_id, EdgeRelation.PASSES_TO.value),
            (admin_role_id, admin_pol_id, EdgeRelation.ATTACHED_WITH.value),
            (admin_pol_id, target_id, EdgeRelation.ACTS_ON.value),
        ]

        return MotifInstance(
            instance_id=f"passrole_sagemaker_{uid}",
            motif_type=self.motif_type,
            tier=self.tier,
            source_id=source_id,
            target_id=target_id,
            intermediate_node_ids=[admin_role_id, admin_pol_id, sm_res_id],
            path_edges=path_edges,
            bridge_relation=bridge_rel,
            required_actions=self.required_actions,
        )


class UpdateAssumeRolePolicyMotif(PEMotif):
    """Tier 2 Motif 10: iam:UpdateAssumeRolePolicy trust modification.

    Attacker modifies the trust policy of an existing administrative role, granting themselves
    permission to assume the role via sts:AssumeRole and pivoting to crown jewels.
    Bridge relation: (source, target_role, AssumesRole).
    """

    motif_type = "update_assume_role_policy"
    name = "iam:UpdateAssumeRolePolicy trust modification"
    description = (
        "Principal updates the trust policy of an existing administrative role to add their own "
        "identity as a trusted principal, then assumes the role to escalate privileges."
    )
    tier = "P1"
    required_actions = ["iam:UpdateAssumeRolePolicy", "sts:AssumeRole"]

    def inject(
        self,
        graph: IAMGraph,
        source_id: str | None = None,
        target_id: str | None = None,
        suffix: str = "",
        account_id: str = "123456789012",
        **_kwargs: Any,
    ) -> MotifInstance:
        cap_model = load_default_capability_model()
        uid = suffix or "default"

        # 1. Source principal
        if source_id is None:
            source_id = f"user:secops-auditor-{uid}"
            source_node = GraphNode(
                id=source_id,
                node_type=NodeType.USER,
                arn=f"arn:aws:iam::{account_id}:user/secops-auditor-{uid}",
                name=f"secops-auditor-{uid}",
                account_id=account_id,
                department="SecOps",
            )
            graph.add_node(source_node)
        else:
            found_s = graph.get_node(source_id)
            if found_s is None:
                raise KeyError(f"Source node '{source_id}' not found in graph.")
            source_node = found_s

        # 2. Target high-value asset
        if target_id is None:
            target_id = f"resource:s3-compliance-vault-{uid}"
            target_node = GraphNode(
                id=target_id,
                node_type=NodeType.RESOURCE,
                arn=f"arn:aws:s3:::compliance-audit-vault-{uid}",
                name=f"compliance-audit-vault-{uid}",
                account_id=account_id,
                department="SecOps",
                is_high_value=True,
            )
            graph.add_node(target_node)
        else:
            found = graph.get_node(target_id)
            if found is None:
                raise KeyError(f"Target node '{target_id}' not found in graph.")
            target_node = found

        # 3. High-privilege administrative role to hijack
        admin_role_id = f"role:secops-privileged-admin-{uid}"
        admin_role_arn = f"arn:aws:iam::{account_id}:role/secops-privileged-admin-{uid}"
        # Initially only trusted by SecOps lead
        initial_trust = PolicyDocument(
            Statement=[
                Statement(
                    Effect=Effect.ALLOW,
                    Action=["sts:AssumeRole"],
                    Principal=Principal.model_validate(
                        {"AWS": [f"arn:aws:iam::{account_id}:root"]}
                    ),
                )
            ]
        )
        admin_role = GraphNode(
            id=admin_role_id,
            node_type=NodeType.ROLE,
            arn=admin_role_arn,
            name=f"secops-privileged-admin-{uid}",
            account_id=account_id,
            department="SecOps",
            is_admin=True,
            trust_policy=initial_trust,
        )
        graph.add_node(admin_role)

        # 4. Admin policy
        admin_pol_id = f"policy:secops-vault-admin-{uid}"
        admin_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["s3:GetObject", "s3:PutObject"],
            Resource=[f"{target_node.arn}/*"],
        )
        assert cap_model.validate_statement(admin_stmt) == []
        admin_pol = GraphNode(
            id=admin_pol_id,
            node_type=NodeType.POLICY,
            arn=f"arn:aws:iam::{account_id}:policy/secops-vault-admin-{uid}",
            name=f"secops-vault-admin-{uid}",
            account_id=account_id,
            department="SecOps",
            is_admin=True,
            policy_document=PolicyDocument(Statement=[admin_stmt]),
        )
        graph.add_node(admin_pol)

        graph.add_edge(
            GraphEdge(
                source=admin_role_id,
                target=admin_pol_id,
                relation=EdgeRelation.ATTACHED_WITH,
            )
        )
        graph.add_edge(
            GraphEdge(
                source=admin_pol_id,
                target=target_id,
                relation=EdgeRelation.ACTS_ON,
                actions=["s3:GetObject"],
            )
        )

        # 5. Source policy granting iam:UpdateAssumeRolePolicy and sts:AssumeRole
        source_pol_id = f"policy:trust-manager-{uid}"
        update_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["iam:UpdateAssumeRolePolicy", "sts:AssumeRole"],
            Resource=[admin_role_arn],
        )
        assert cap_model.validate_statement(update_stmt) == []
        source_pol = GraphNode(
            id=source_pol_id,
            node_type=NodeType.POLICY,
            arn=f"arn:aws:iam::{account_id}:policy/trust-manager-{uid}",
            name=f"trust-manager-{uid}",
            account_id=account_id,
            department="SecOps",
            policy_document=PolicyDocument(Statement=[update_stmt]),
        )
        graph.add_node(source_pol)

        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=source_pol_id,
                relation=EdgeRelation.ATTACHED_WITH,
            )
        )

        # 6. Bridge relation: AssumesRole from source to admin role
        bridge_rel = (source_id, admin_role_id, EdgeRelation.ASSUMES_ROLE.value)
        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=admin_role_id,
                relation=EdgeRelation.ASSUMES_ROLE,
                is_bridge=True,
                motif_id=f"update_assume_role_{uid}",
                actions=["iam:UpdateAssumeRolePolicy", "sts:AssumeRole"],
            )
        )

        path_edges = [
            (source_id, admin_role_id, EdgeRelation.ASSUMES_ROLE.value),
            (admin_role_id, admin_pol_id, EdgeRelation.ATTACHED_WITH.value),
            (admin_pol_id, target_id, EdgeRelation.ACTS_ON.value),
        ]

        return MotifInstance(
            instance_id=f"update_assume_role_{uid}",
            motif_type=self.motif_type,
            tier=self.tier,
            source_id=source_id,
            target_id=target_id,
            intermediate_node_ids=[admin_role_id, admin_pol_id],
            path_edges=path_edges,
            bridge_relation=bridge_rel,
            required_actions=self.required_actions,
        )


class CreatePolicyVersionMotif(PEMotif):
    """Tier 2 Motif 11: iam:CreatePolicyVersion privilege escalation.

    Attacker creates a new version of an existing policy attached to themselves, granting
    access to sensitive target resources and setting the new version as default.
    Bridge relation: (policy, target, ActsOn).
    """

    motif_type = "create_policy_version"
    name = "iam:CreatePolicyVersion privilege escalation"
    description = (
        "Principal creates a new version of an attached customer-managed policy with expanded "
        "permissions and sets it as the active default version to escalate privileges."
    )
    tier = "P1"
    required_actions = ["iam:CreatePolicyVersion", "iam:SetDefaultPolicyVersion"]

    def inject(
        self,
        graph: IAMGraph,
        source_id: str | None = None,
        target_id: str | None = None,
        suffix: str = "",
        account_id: str = "123456789012",
        **_kwargs: Any,
    ) -> MotifInstance:
        cap_model = load_default_capability_model()
        uid = suffix or "default"

        # 1. Source principal
        if source_id is None:
            source_id = f"user:app-developer-{uid}"
            source_node = GraphNode(
                id=source_id,
                node_type=NodeType.USER,
                arn=f"arn:aws:iam::{account_id}:user/app-developer-{uid}",
                name=f"app-developer-{uid}",
                account_id=account_id,
                department="DevOps",
            )
            graph.add_node(source_node)
        else:
            found_s = graph.get_node(source_id)
            if found_s is None:
                raise KeyError(f"Source node '{source_id}' not found in graph.")
            source_node = found_s

        # 2. Target high-value asset
        if target_id is None:
            target_id = f"resource:s3-vault-{uid}"
            target_node = GraphNode(
                id=target_id,
                node_type=NodeType.RESOURCE,
                arn=f"arn:aws:s3:::app-secret-vault-{uid}",
                name=f"app-secret-vault-{uid}",
                account_id=account_id,
                department="SecOps",
                is_high_value=True,
            )
            graph.add_node(target_node)
        else:
            found = graph.get_node(target_id)
            if found is None:
                raise KeyError(f"Target node '{target_id}' not found in graph.")
            target_node = found

        # 3. Customer managed policy attached to user (version 2 escalation)
        cust_pol_id = f"policy:dev-expandable-pol-{uid}"
        cust_pol_arn = f"arn:aws:iam::{account_id}:policy/dev-expandable-pol-{uid}"
        escalated_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["s3:GetObject", "s3:PutObject"],
            Resource=[f"{target_node.arn}/*"],
        )
        assert cap_model.validate_statement(escalated_stmt) == []
        cust_pol = GraphNode(
            id=cust_pol_id,
            node_type=NodeType.POLICY,
            arn=cust_pol_arn,
            name=f"dev-expandable-pol-{uid}",
            account_id=account_id,
            department="DevOps",
            policy_document=PolicyDocument(Statement=[escalated_stmt]),
        )
        graph.add_node(cust_pol)

        # 4. User policy granting permission to create policy version
        editor_pol_id = f"policy:policy-version-creator-{uid}"
        edit_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["iam:CreatePolicyVersion", "iam:SetDefaultPolicyVersion"],
            Resource=[cust_pol_arn],
        )
        assert cap_model.validate_statement(edit_stmt) == []
        editor_pol = GraphNode(
            id=editor_pol_id,
            node_type=NodeType.POLICY,
            arn=f"arn:aws:iam::{account_id}:policy/policy-version-creator-{uid}",
            name=f"policy-version-creator-{uid}",
            account_id=account_id,
            department="DevOps",
            policy_document=PolicyDocument(Statement=[edit_stmt]),
        )
        graph.add_node(editor_pol)

        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=editor_pol_id,
                relation=EdgeRelation.ATTACHED_WITH,
            )
        )
        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=cust_pol_id,
                relation=EdgeRelation.ACTS_ON,
                actions=["iam:CreatePolicyVersion"],
            )
        )
        graph.add_edge(
            GraphEdge(
                source=source_id,
                target=cust_pol_id,
                relation=EdgeRelation.ATTACHED_WITH,
            )
        )

        # 5. Bridge relation: ActsOn from customer policy to target asset
        bridge_rel = (cust_pol_id, target_id, EdgeRelation.ACTS_ON.value)
        graph.add_edge(
            GraphEdge(
                source=cust_pol_id,
                target=target_id,
                relation=EdgeRelation.ACTS_ON,
                is_bridge=True,
                motif_id=f"create_policy_version_{uid}",
                actions=["s3:GetObject"],
            )
        )

        path_edges = [
            (source_id, cust_pol_id, EdgeRelation.ATTACHED_WITH.value),
            (cust_pol_id, target_id, EdgeRelation.ACTS_ON.value),
        ]

        return MotifInstance(
            instance_id=f"create_policy_version_{uid}",
            motif_type=self.motif_type,
            tier=self.tier,
            source_id=source_id,
            target_id=target_id,
            intermediate_node_ids=[cust_pol_id],
            path_edges=path_edges,
            bridge_relation=bridge_rel,
            required_actions=self.required_actions,
        )


# ==============================================================================
# Motif Registry
# ==============================================================================


class MotifRegistry:
    """Registry pattern facilitating dynamic registration and lookup of PE motifs."""

    def __init__(self) -> None:
        self._motifs: dict[str, PEMotif] = {}
        # Pre-register all canonical motifs
        self.register(PassRoleLambdaMotif())
        self.register(CreateAccessKeyMotif())
        self.register(AttachPolicyMotif())
        self.register(AssumeRoleChainMotif())
        self.register(PassRoleEC2Motif())
        self.register(SetDefaultPolicyVersionMotif())
        self.register(GlueDevEndpointMotif())
        self.register(CodeBuildProjectMotif())
        self.register(SageMakerNotebookMotif())
        self.register(UpdateAssumeRolePolicyMotif())
        self.register(CreatePolicyVersionMotif())

    def register(self, motif: PEMotif) -> None:
        """Register a new motif handler."""
        self._motifs[motif.motif_type] = motif

    def get(self, motif_type: str) -> PEMotif:
        """Retrieve motif instance by its type string."""
        if motif_type not in self._motifs:
            raise KeyError(
                f"Unknown motif type '{motif_type}'. Available: {list(self._motifs.keys())}"
            )
        return self._motifs[motif_type]

    def list_motifs(self, tier: str | None = None) -> list[PEMotif]:
        """List all motifs, optionally filtered by tier ('P0' or 'P1')."""
        if tier is None:
            return list(self._motifs.values())
        return [m for m in self._motifs.values() if m.tier == tier]

    def __contains__(self, motif_type: str) -> bool:
        return motif_type in self._motifs

    def __len__(self) -> int:
        return len(self._motifs)


# Default global registry singleton
default_motif_registry = MotifRegistry()
