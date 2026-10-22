"""Unit and integration test suite for Terraform & IaC AWS IAM Parser and Repository Manager."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from iam.generator.graph import EdgeRelation, NodeType
from iam.parser.repository import (
    IaCRepositoryCatalog,
    IaCRepositoryManager,
    IaCRepositoryMetadata,
)
from iam.parser.schema import Effect
from iam.parser.terraform import TerraformParser


class TestIaCRepositoryCatalog:
    """Validates the curated 35-repository benchmark catalog."""

    def test_catalog_load_and_counts(self) -> None:
        """Verify default catalog loads 35 repositories across 5 architectural categories."""
        catalog = IaCRepositoryCatalog.load_default()
        assert len(catalog) == 35
        assert catalog.target_count == 35
        assert len(catalog.categories) == 5

        expected_categories = {
            "iam_identity",
            "container_kubernetes",
            "data_serverless",
            "cicd_devops",
            "tac_bench_reference",
        }
        assert set(catalog.categories) == expected_categories

    def test_catalog_lookup_and_filtering(self) -> None:
        """Verify repository lookup by ID and category filtering."""
        catalog = IaCRepositoryCatalog.load_default()

        # Lookup by ID
        iam_repo = catalog.get("tf-aws-iam")
        assert isinstance(iam_repo, IaCRepositoryMetadata)
        assert iam_repo.name == "terraform-aws-modules/terraform-aws-iam"
        assert iam_repo.category == "iam_identity"
        assert "aws_iam_role" in iam_repo.key_resources

        # Category filtering
        tac_repos = catalog.list_by_category("tac_bench_reference")
        assert len(tac_repos) >= 6
        for r in tac_repos:
            assert r.category == "tac_bench_reference"

        # Missing repo raises KeyError
        with pytest.raises(KeyError, match="not found in catalog"):
            catalog.get("nonexistent_repo")


class TestTerraformParserHCLSyntax:
    """Validates HCL2 parsing of core AWS IAM Terraform constructs."""

    @pytest.fixture
    def parser(self) -> TerraformParser:
        return TerraformParser(account_id="111122223333")

    def test_parse_user_group_and_membership(self, parser: TerraformParser) -> None:
        hcl = """
        resource "aws_iam_user" "analyst" {
          name = "data-analyst"
          path = "/analytics/"
          tags = {
            Env = "Prod"
          }
        }

        resource "aws_iam_group" "bi_team" {
          name = "bi-engineers"
        }

        resource "aws_iam_user_group_membership" "analyst_membership" {
          user   = aws_iam_user.analyst.name
          groups = [aws_iam_group.bi_team.name]
        }
        """
        env = parser.parse_string(hcl, filename="users.tf")

        assert "data-analyst" in env.users
        user = env.users["data-analyst"]
        assert user.arn == "arn:aws:iam::111122223333:user/analytics/data-analyst"
        assert user.tags == {"Env": "Prod"}

        assert "bi-engineers" in env.groups
        group = env.groups["bi-engineers"]
        assert group.arn == "arn:aws:iam::111122223333:group/bi-engineers"

        assert len(env.memberships) == 1
        assert env.memberships[0].user_name == "data-analyst"
        assert env.memberships[0].group_name == "bi-engineers"

    def test_parse_role_with_jsonencode_trust_policy(self, parser: TerraformParser) -> None:
        hcl = """
        resource "aws_iam_role" "worker_role" {
          name = "worker-task"
          assume_role_policy = jsonencode({
            Version = "2012-10-17"
            Statement = [
              {
                Action = "sts:AssumeRole"
                Effect = "Allow"
                Principal = {
                  Service = "lambda.amazonaws.com"
                }
              }
            ]
          })
        }
        """
        env = parser.parse_string(hcl, filename="roles.tf")

        assert "worker-task" in env.roles
        role = env.roles["worker-task"]
        assert role.arn == "arn:aws:iam::111122223333:role/worker-task"
        assert role.assume_role_policy is not None
        assert len(role.assume_role_policy.statements) == 1
        stmt = role.assume_role_policy.statements[0]
        assert stmt.effect == Effect.ALLOW
        assert "sts:AssumeRole" in stmt.action

    def test_parse_data_policy_document_and_managed_policy(self, parser: TerraformParser) -> None:
        hcl = """
        data "aws_iam_policy_document" "s3_access" {
          statement {
            sid     = "BucketAccess"
            effect  = "Allow"
            actions = ["s3:GetObject", "s3:PutObject"]
            resources = [
              "arn:aws:s3:::production-vault/*"
            ]
          }
        }

        resource "aws_iam_policy" "s3_policy" {
          name        = "S3ProductionVaultAccess"
          description = "Grants read/write access to vault bucket"
          policy      = data.aws_iam_policy_document.s3_access.json
        }

        resource "aws_iam_role_policy_attachment" "role_attach" {
          role       = "worker-task"
          policy_arn = aws_iam_policy.s3_policy.arn
        }
        """
        env = parser.parse_string(hcl, filename="policies.tf")

        expected_arn = "arn:aws:iam::111122223333:policy/S3ProductionVaultAccess"
        assert expected_arn in env.policies
        policy = env.policies[expected_arn]
        assert policy.name == "S3ProductionVaultAccess"
        assert len(policy.policy_document.statements) == 1
        stmt = policy.policy_document.statements[0]
        assert stmt.sid == "BucketAccess"
        assert "s3:GetObject" in stmt.action
        assert "arn:aws:s3:::production-vault/*" in stmt.resource

        assert len(env.attachments) == 1
        att = env.attachments[0]
        assert att.principal_type == "Role"
        assert att.principal_name == "worker-task"
        assert att.policy_arn == expected_arn

    def test_parse_inline_policy(self, parser: TerraformParser) -> None:
        hcl = """
        resource "aws_iam_user_policy" "admin_override" {
          name = "admin-override"
          user = "data-analyst"
          policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["*"],
      "Resource": ["*"]
    }
  ]
}
EOF
        }
        """
        env = parser.parse_string(hcl, filename="inline.tf")

        assert len(env.policies) == 1
        inline_pol = next(iter(env.policies.values()))
        assert inline_pol.is_inline is True
        assert inline_pol.name == "data-analyst-admin-override"

        assert len(env.attachments) == 1
        assert env.attachments[0].is_inline is True
        assert env.attachments[0].principal_name == "data-analyst"

    def test_parse_tf_json_format(self, parser: TerraformParser) -> None:
        """Verify parsing of Terraform JSON specification (.tf.json)."""
        raw_json = {
            "resource": [
                {
                    "aws_iam_user": {
                        "ci_bot": {
                            "name": "ci-automation-bot",
                            "path": "/bots/",
                        }
                    }
                }
            ]
        }
        env = parser.parse_json(raw_json, filename="bots.tf.json")
        assert "ci-automation-bot" in env.users
        assert (
            env.users["ci-automation-bot"].arn
            == "arn:aws:iam::111122223333:user/bots/ci-automation-bot"
        )


class TestTerraformIAMGraphConversion:
    """Validates full end-to-end conversion from Terraform IR to typed IAMGraph."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> IaCRepositoryManager:
        return IaCRepositoryManager(storage_dir=tmp_path)

    def test_enterprise_microservices_graph_conversion(
        self, manager: IaCRepositoryManager, tmp_path: Path
    ) -> None:
        """Verify multi-file microservices fixture converts cleanly into an IAMGraph."""
        fixture_dir = tmp_path / "enterprise_microservices"
        manager.generate_fixture_repository(fixture_dir, variant="enterprise_microservices")

        env, graph = manager.ingest_repository(fixture_dir, graph_id="microservices_test")

        # Verify environment counts
        assert len(env.users) == 2
        assert len(env.roles) == 2
        assert len(env.groups) == 1
        assert len(env.policies) == 1  # S3CustomerAssetsReadOnly
        assert len(env.attachments) == 2  # Backend S3 + Deployer Admin

        # Verify GraphNode counts
        assert graph.num_nodes >= 6
        user_node = graph.get_node("user:lead-engineer")
        assert user_node is not None
        assert user_node.node_type == NodeType.USER

        role_node = graph.get_node("role:app-backend-service")
        assert role_node is not None
        assert role_node.node_type == NodeType.ROLE

        group_node = graph.get_node("group:developers")
        assert group_node is not None
        assert group_node.node_type == NodeType.GROUP

        admin_pol_node = graph.get_node("policy:AdministratorAccess")
        assert admin_pol_node is not None
        assert admin_pol_node.is_admin is True

        # Verify Edges
        # User -> Group (MemberOf)
        assert graph.has_edge("user:lead-engineer", "group:developers", EdgeRelation.MEMBER_OF)

        # Deployer Role -> AdministratorAccess (AttachedWith)
        assert graph.has_edge(
            "role:ci-deployer", "policy:AdministratorAccess", EdgeRelation.ATTACHED_WITH
        )

        # Lead engineer -> Deployer Role (AssumesRole via trust policy)
        assert graph.has_edge("user:lead-engineer", "role:ci-deployer", EdgeRelation.ASSUMES_ROLE)

        # S3 Policy acts on S3 bucket resource
        s3_res_id = "resource:arn_aws_s3___corporate-customer-assets"
        assert graph.get_node(s3_res_id) is not None
        assert graph.has_edge("policy:S3CustomerAssetsReadOnly", s3_res_id, EdgeRelation.ACTS_ON)

    def test_privilege_escalation_vulnerability_detection(
        self, manager: IaCRepositoryManager, tmp_path: Path
    ) -> None:
        """Verify Terraform configuration with PassRole privilege escalation forms a traversable path in IAMGraph."""
        fixture_dir = tmp_path / "pe_vulnerable"
        manager.generate_fixture_repository(fixture_dir, variant="privilege_escalation_vulnerable")

        _, graph = manager.ingest_repository(fixture_dir, graph_id="pe_vulnerable_graph")

        intern_id = "user:intern-dev"
        exec_role_id = "role:lambda-superadmin-exec"
        admin_pol_id = "policy:CrownJewelVaultAccess"
        vault_res_id = "resource:arn_aws_s3___confidential-crown-jewels_*"

        assert graph.get_node(intern_id) is not None
        assert graph.get_node(exec_role_id) is not None
        assert graph.get_node(admin_pol_id) is not None
        assert graph.get_node(vault_res_id) is not None

        # Verify attachment: exec role -> admin policy
        assert graph.has_edge(exec_role_id, admin_pol_id, EdgeRelation.ATTACHED_WITH)

        # Verify action: admin policy -> vault resource
        assert graph.has_edge(admin_pol_id, vault_res_id, EdgeRelation.ACTS_ON)

        # Intern has policy attached granting PassRole on exec role
        intern_pol_id = "policy:intern-dev-LambdaDeployPermission"
        assert graph.get_node(intern_pol_id) is not None
        assert graph.has_edge(intern_id, intern_pol_id, EdgeRelation.ATTACHED_WITH)

    def test_tac_bench_reference_cross_account_chain(
        self, manager: IaCRepositoryManager, tmp_path: Path
    ) -> None:
        """Verify multi-hop cross-account role assumption chain parses into directed path."""
        fixture_dir = tmp_path / "tac_bench"
        manager.generate_fixture_repository(fixture_dir, variant="tac_bench_reference")

        _, graph = manager.ingest_repository(fixture_dir, graph_id="tac_bench_graph")

        auditor_id = "user:security-auditor"
        jump1_id = "role:staging-jump-role"
        jump2_id = "role:prod-gateway-role"
        kms_pol_id = "policy:VaultSecretsAccess"
        kms_key_id = "resource:arn_aws_kms_us-east-1_123456789012_key_production-vault-key"

        # Check chain edges
        assert graph.has_edge(auditor_id, jump1_id, EdgeRelation.ASSUMES_ROLE)
        assert graph.has_edge(jump1_id, jump2_id, EdgeRelation.ASSUMES_ROLE)
        assert graph.has_edge(jump2_id, kms_pol_id, EdgeRelation.ATTACHED_WITH)
        assert graph.has_edge(kms_pol_id, kms_key_id, EdgeRelation.ACTS_ON)

        # NetworkX directed path exists from auditor to KMS key
        nx_g = graph.nx_graph
        assert nx.has_path(nx_g, auditor_id, kms_key_id)
        path = nx.shortest_path(nx_g, auditor_id, kms_key_id)
        assert path == [auditor_id, jump1_id, jump2_id, kms_pol_id, kms_key_id]
