"""Repository ingestion and catalog management for real-world Terraform IAM configurations.

Provides metadata management for 35+ curated open-source AWS infrastructure repositories,
safe local cloning/caching, offline synthetic repository generation for testing,
and automated batch parsing into IAMGraph datasets for Tier 2 evaluation.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from iam.parser.terraform import TerraformIAMEnvironment, TerraformParser

if TYPE_CHECKING:
    from iam.generator.graph import IAMGraph

logger = logging.getLogger(__name__)

DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "schemas"
    / "iac_repo_catalog.json"
)


# ==============================================================================
# Catalog Data Models
# ==============================================================================


class IaCRepositoryMetadata(BaseModel):
    """Metadata describing a curated open-source IaC repository."""

    model_config = ConfigDict(frozen=True)

    repo_id: str
    category: str
    name: str
    url: str
    default_branch: str = "master"
    description: str
    complexity: str = "medium"
    key_resources: list[str] = Field(default_factory=list)


class IaCRepositoryCatalog(BaseModel):
    """Curated catalog of IaC repositories for evaluation."""

    model_config = ConfigDict(frozen=True)

    catalog_version: str
    description: str
    target_count: int
    categories: list[str]
    repositories: list[IaCRepositoryMetadata]

    @classmethod
    def load_default(cls, catalog_path: Path | None = None) -> IaCRepositoryCatalog:
        """Load catalog from JSON file."""
        target = catalog_path or DEFAULT_CATALOG_PATH
        if not target.is_file():
            raise FileNotFoundError(f"IaC catalog not found at: {target}")
        data = json.loads(target.read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def get(self, repo_id: str) -> IaCRepositoryMetadata:
        """Retrieve repository metadata by ID."""
        for repo in self.repositories:
            if repo.repo_id == repo_id:
                return repo
        raise KeyError(
            f"Repository '{repo_id}' not found in catalog. Available: {[r.repo_id for r in self.repositories]}"
        )

    def list_by_category(self, category: str) -> list[IaCRepositoryMetadata]:
        """Filter repositories by category."""
        return [r for r in self.repositories if r.category == category]

    def __len__(self) -> int:
        return len(self.repositories)


# ==============================================================================
# Ingestion and Synthesis Manager
# ==============================================================================


class IaCRepositoryManager:
    """Manages cloning, caching, offline synthesis, and batch ingestion of Terraform repositories."""

    def __init__(
        self,
        storage_dir: Path | str | None = None,
        catalog: IaCRepositoryCatalog | None = None,
        parser: TerraformParser | None = None,
    ) -> None:
        if storage_dir is None:
            self.storage_dir = (
                Path(__file__).resolve().parent.parent.parent.parent / "data" / "iac_repositories"
            )
        else:
            self.storage_dir = Path(storage_dir)

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = catalog or IaCRepositoryCatalog.load_default()
        self.parser = parser or TerraformParser()

    def clone_or_update(
        self,
        repo_id: str,
        shallow: bool = True,
        timeout_seconds: int = 60,
    ) -> Path:
        """Clone or pull an open-source repository from GitHub."""
        meta = self.catalog.get(repo_id)
        target_dir = self.storage_dir / repo_id

        if (target_dir / ".git").is_dir():
            logger.info(
                "Repository '%s' already cloned at %s, pulling latest...", repo_id, target_dir
            )
            try:
                subprocess.run(
                    ["git", "-C", str(target_dir), "pull", "--ff-only"],
                    check=False,
                    capture_output=True,
                    timeout=timeout_seconds,
                )
            except Exception as e:
                logger.warning("Failed to pull latest for '%s': %s", repo_id, e)
            return target_dir

        cmd = ["git", "clone"]
        if shallow:
            cmd.extend(["--depth", "1", "--branch", meta.default_branch])
        cmd.extend([meta.url, str(target_dir)])

        logger.info("Cloning repository '%s' from %s...", repo_id, meta.url)
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"Git clone failed for repository '{repo_id}' ({meta.url}): {err}")

        return target_dir

    def ingest_repository(
        self,
        repo_dir: Path | str,
        graph_id: str | None = None,
    ) -> tuple[TerraformIAMEnvironment, IAMGraph]:
        """Parse all Terraform code within a repository directory into an IAMGraph."""
        path = Path(repo_dir)
        env = self.parser.parse_directory(path, recursive=True)
        gid = graph_id or f"iac_{path.name}"
        graph = self.parser.to_iam_graph(env, graph_id=gid)
        return env, graph

    # --------------------------------------------------------------------------
    # Offline Synthetic Repository Generation
    # --------------------------------------------------------------------------

    def generate_fixture_repository(
        self,
        target_dir: Path | str,
        variant: str = "enterprise_microservices",
    ) -> Path:
        """Generate a complete multi-file Terraform repository fixture offline."""
        out = Path(target_dir)
        out.mkdir(parents=True, exist_ok=True)

        if variant == "enterprise_microservices":
            self._write_enterprise_microservices_fixture(out)
        elif variant == "privilege_escalation_vulnerable":
            self._write_pe_vulnerable_fixture(out)
        elif variant == "tac_bench_reference":
            self._write_tac_bench_fixture(out)
        else:
            raise ValueError(f"Unknown fixture variant: '{variant}'")

        return out

    def _write_enterprise_microservices_fixture(self, out: Path) -> None:
        """Create a realistic microservices IAM Terraform configuration."""
        # 1. main.tf
        main_tf = """# Enterprise Microservices IAM Topology
terraform {
  required_version = ">= 1.5.0"
}

# Core Users
resource "aws_iam_user" "lead_engineer" {
  name = "lead-engineer"
  path = "/engineering/"
  tags = {
    Department = "DevOps"
  }
}

resource "aws_iam_user" "qa_tester" {
  name = "qa-tester"
  path = "/engineering/"
  tags = {
    Department = "QA"
  }
}

# User Group
resource "aws_iam_group" "dev_team" {
  name = "developers"
  path = "/engineering/"
}

resource "aws_iam_user_group_membership" "dev_membership" {
  user   = aws_iam_user.lead_engineer.name
  groups = [aws_iam_group.dev_team.name]
}
"""
        (out / "main.tf").write_text(main_tf, encoding="utf-8")

        # 2. roles.tf
        roles_tf = """# Service and Deployment Roles
resource "aws_iam_role" "app_backend_role" {
  name = "app-backend-service"
  path = "/services/"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role" "deployer_role" {
  name = "ci-deployer"
  path = "/ci/"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          AWS = [
            "arn:aws:iam::123456789012:user/engineering/lead-engineer"
          ]
        }
      }
    ]
  })
}
"""
        (out / "roles.tf").write_text(roles_tf, encoding="utf-8")

        # 3. policies.tf
        policies_tf = """# Customer Managed Policies and Attachments
data "aws_iam_policy_document" "s3_reader_doc" {
  statement {
    sid       = "AllowS3Read"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [
      "arn:aws:s3:::corporate-customer-assets",
      "arn:aws:s3:::corporate-customer-assets/*"
    ]
  }
}

resource "aws_iam_policy" "s3_read_policy" {
  name        = "S3CustomerAssetsReadOnly"
  description = "Grants read access to customer asset bucket"
  policy      = data.aws_iam_policy_document.s3_reader_doc.json
}

resource "aws_iam_role_policy_attachment" "backend_s3_attach" {
  role       = aws_iam_role.app_backend_role.name
  policy_arn = aws_iam_policy.s3_read_policy.arn
}

resource "aws_iam_role_policy_attachment" "deployer_admin" {
  role       = aws_iam_role.deployer_role.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}
"""
        (out / "policies.tf").write_text(policies_tf, encoding="utf-8")

    def _write_pe_vulnerable_fixture(self, out: Path) -> None:
        """Create a Terraform configuration embedding a canonical PassRole privilege escalation vector."""
        pe_tf = """# Vulnerable Privilege Escalation Pattern: iam:PassRole with Lambda
resource "aws_iam_user" "intern_dev" {
  name = "intern-dev"
}

resource "aws_iam_role" "privileged_lambda_exec_role" {
  name = "lambda-superadmin-exec"
  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
}

resource "aws_iam_policy" "admin_access_policy" {
  name = "CrownJewelVaultAccess"
  policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::confidential-crown-jewels/*"
    }
  ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "lambda_admin_attach" {
  role       = aws_iam_role.privileged_lambda_exec_role.name
  policy_arn = aws_iam_policy.admin_access_policy.arn
}

# Intern policy granting PassRole and Lambda creation
resource "aws_iam_user_policy" "intern_escalator_policy" {
  name = "LambdaDeployPermission"
  user = aws_iam_user.intern_dev.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["iam:PassRole"]
        Resource = [aws_iam_role.privileged_lambda_exec_role.arn]
      },
      {
        Effect   = "Allow"
        Action   = ["lambda:CreateFunction", "lambda:InvokeFunction"]
        Resource = ["*"]
      }
    ]
  })
}
"""
        (out / "pe_vulnerable.tf").write_text(pe_tf, encoding="utf-8")

    def _write_tac_bench_fixture(self, out: Path) -> None:
        """Create a TAC-Bench style cross-account trust boundary configuration."""
        tac_tf = """# TAC-Bench Reference: Multi-Hop Cross-Account Delegation
resource "aws_iam_user" "auditor" {
  name = "security-auditor"
}

resource "aws_iam_role" "jump_role_1" {
  name = "staging-jump-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          AWS = ["arn:aws:iam::123456789012:user/security-auditor"]
        }
      }
    ]
  })
}

resource "aws_iam_role" "jump_role_2" {
  name = "prod-gateway-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          AWS = [aws_iam_role.jump_role_1.arn]
        }
      }
    ]
  })
}

resource "aws_iam_policy" "vault_secrets_policy" {
  name = "VaultSecretsAccess"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["kms:Decrypt"]
        Resource = ["arn:aws:kms:us-east-1:123456789012:key/production-vault-key"]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "gateway_secrets_attach" {
  role       = aws_iam_role.jump_role_2.name
  policy_arn = aws_iam_policy.vault_secrets_policy.arn
}
"""
        (out / "tac_bench.tf").write_text(tac_tf, encoding="utf-8")
