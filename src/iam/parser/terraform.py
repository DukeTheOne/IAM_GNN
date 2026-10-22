"""Terraform and Infrastructure-as-Code (IaC) parser for AWS IAM.

Extracts IAM users, roles, groups, customer-managed policies, inline policies,
policy attachments, and group memberships from HCL2 (.tf) and JSON (.tf.json)
configurations into a unified IAMGraph representation.
"""

from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hcl2
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField

from iam.parser.schema import Effect, PolicyDocument, Principal, Statement

if TYPE_CHECKING:
    from iam.generator.graph import IAMGraph

logger = logging.getLogger(__name__)


# ==============================================================================
# Intermediate Representation (IR) Models
# ==============================================================================


class TerraformUser(BaseModel):
    """Parsed IAM User from Terraform."""

    model_config = ConfigDict(frozen=True)

    name: str
    arn: str
    path: str = "/"
    tags: dict[str, str] = PydanticField(default_factory=dict)
    source_file: str | None = None


class TerraformRole(BaseModel):
    """Parsed IAM Role from Terraform."""

    model_config = ConfigDict(frozen=True)

    name: str
    arn: str
    path: str = "/"
    assume_role_policy: PolicyDocument | None = None
    tags: dict[str, str] = PydanticField(default_factory=dict)
    source_file: str | None = None


class TerraformGroup(BaseModel):
    """Parsed IAM Group from Terraform."""

    model_config = ConfigDict(frozen=True)

    name: str
    arn: str
    path: str = "/"
    source_file: str | None = None


class TerraformPolicy(BaseModel):
    """Parsed IAM Policy (Customer Managed or Inline) from Terraform."""

    model_config = ConfigDict(frozen=True)

    name: str
    arn: str
    path: str = "/"
    description: str | None = None
    policy_document: PolicyDocument
    is_inline: bool = False
    source_file: str | None = None


class TerraformAttachment(BaseModel):
    """Parsed policy attachment binding principal/group to a policy."""

    model_config = ConfigDict(frozen=True)

    principal_type: str  # "User", "Role", "Group"
    principal_name: str
    policy_arn: str
    is_inline: bool = False
    source_file: str | None = None


class TerraformMembership(BaseModel):
    """Parsed group membership binding user to group."""

    model_config = ConfigDict(frozen=True)

    user_name: str
    group_name: str
    source_file: str | None = None


@dataclass
class TerraformIAMEnvironment:
    """Aggregated IAM environment extracted from Terraform configurations."""

    users: dict[str, TerraformUser] = field(default_factory=dict)
    roles: dict[str, TerraformRole] = field(default_factory=dict)
    groups: dict[str, TerraformGroup] = field(default_factory=dict)
    policies: dict[str, TerraformPolicy] = field(default_factory=dict)
    attachments: list[TerraformAttachment] = field(default_factory=list)
    memberships: list[TerraformMembership] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    unresolved_references: list[str] = field(default_factory=list)
    symbol_table: dict[str, str] = field(default_factory=dict)
    arn_table: dict[str, str] = field(default_factory=dict)

    def merge(self, other: TerraformIAMEnvironment) -> None:
        """Merge another parsed environment into this one."""
        self.users.update(other.users)
        self.roles.update(other.roles)
        self.groups.update(other.groups)
        self.policies.update(other.policies)
        self.attachments.extend(other.attachments)
        self.memberships.extend(other.memberships)
        self.symbol_table.update(other.symbol_table)
        self.arn_table.update(other.arn_table)
        for sf in other.source_files:
            if sf not in self.source_files:
                self.source_files.append(sf)
        self.unresolved_references.extend(other.unresolved_references)

    def resolve_name(self, ref: str) -> str:
        """Resolve a Terraform reference or block label to the actual resource name."""
        clean = _clean_str(ref)
        if clean in self.symbol_table:
            return self.symbol_table[clean]
        ident = _extract_identifier(clean)
        if ident in self.symbol_table:
            return self.symbol_table[ident]
        return ident or clean

    def resolve_arn(self, ref: Any, account_id: str = "123456789012") -> str:
        """Resolve a Terraform reference or policy identifier to an AWS ARN."""
        if not ref:
            return ""
        clean = _clean_str(ref)
        if clean.startswith("arn:aws:"):
            return clean
        if clean in self.arn_table:
            return self.arn_table[clean]
        ident = _extract_identifier(clean)
        if ident in self.arn_table:
            return self.arn_table[ident]
        actual_name = self.symbol_table.get(ident, ident)
        if actual_name in self.arn_table:
            return self.arn_table[actual_name]
        return f"arn:aws:iam::{account_id}:policy/{actual_name}"


# ==============================================================================
# Helper Utilities for HCL Cleaning and Parsing
# ==============================================================================


def _clean_str(val: Any) -> str:
    """Clean string values and strip unwanted enclosing quotes."""
    if not isinstance(val, str):
        return str(val) if val is not None else ""
    s = val.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s


def _clean_data_structure(data: Any) -> Any:
    """Recursively clean strings and dictionaries parsed by python-hcl2."""
    if isinstance(data, str):
        return _clean_str(data)
    if isinstance(data, list):
        return [_clean_data_structure(item) for item in data]
    if isinstance(data, dict):
        cleaned: dict[str, Any] = {}
        for k, v in data.items():
            if k == "__is_block__":
                continue
            cleaned[_clean_str(k)] = _clean_data_structure(v)
        return cleaned
    return data


def _extract_identifier(ref: str) -> str:
    """Extract resource name from Terraform expressions like 'aws_iam_role.foo.name'."""
    clean = _clean_str(ref)
    # Remove interpolation wrapper ${...}
    if clean.startswith("${") and clean.endswith("}"):
        clean = clean[2:-1].strip()

    # Match patterns like aws_iam_role.role_name.name or aws_iam_role.role_name.arn
    parts = clean.split(".")
    if len(parts) >= 2 and parts[0] in {
        "aws_iam_role",
        "aws_iam_user",
        "aws_iam_group",
        "aws_iam_policy",
        "data",
        "var",
    }:
        if parts[0] == "data" and len(parts) >= 3:
            return parts[2]
        return parts[1]
    return clean


def _extract_json_from_heredoc(val: str) -> str:
    """Extract JSON string from heredoc markers like <<EOF ... EOF."""
    s = val.strip()
    # Strip enclosing quotes if present
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    heredoc_match = re.match(r"^<<-?([A-Za-z0-9_]+)\n(.*)\n\1$", s, re.DOTALL)
    if heredoc_match:
        return heredoc_match.group(2).strip()
    return s


# ==============================================================================
# Terraform IAM Parser
# ==============================================================================


class TerraformParser:
    """Parser for extracting AWS IAM topology and policies from Terraform files."""

    def __init__(
        self,
        account_id: str = "123456789012",
        default_region: str = "us-east-1",
    ) -> None:
        self.account_id = account_id
        self.default_region = default_region

    def parse_string(
        self,
        hcl_content: str,
        filename: str = "main.tf",
    ) -> TerraformIAMEnvironment:
        """Parse raw HCL string content into an IAM environment."""
        env = TerraformIAMEnvironment(source_files=[filename])
        try:
            raw_parsed = hcl2.load(io.StringIO(hcl_content))
            cleaned = _clean_data_structure(raw_parsed)
        except Exception as e:
            logger.warning("Failed to parse HCL in '%s': %s", filename, e)
            return env

        self._process_parsed_hcl(cleaned, env, source_file=filename)
        return env

    def parse_file(self, file_path: str | Path) -> TerraformIAMEnvironment:
        """Parse a single .tf or .tf.json file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Terraform file not found: {path}")

        content = path.read_text(encoding="utf-8")
        if path.suffix == ".json" or path.name.endswith(".tf.json"):
            return self.parse_json(content, filename=str(path))
        return self.parse_string(content, filename=str(path))

    def parse_json(
        self,
        json_content: str | dict[str, Any],
        filename: str = "main.tf.json",
    ) -> TerraformIAMEnvironment:
        """Parse Terraform JSON syntax (.tf.json)."""
        env = TerraformIAMEnvironment(source_files=[filename])
        try:
            raw_data = json.loads(json_content) if isinstance(json_content, str) else json_content
            cleaned = _clean_data_structure(raw_data)
        except Exception as e:
            logger.warning("Failed to parse JSON in '%s': %s", filename, e)
            return env

        self._process_parsed_hcl(cleaned, env, source_file=filename)
        return env

    def parse_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
    ) -> TerraformIAMEnvironment:
        """Parse all Terraform files (.tf and .tf.json) in a directory."""
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Directory not found: {path}")

        combined_env = TerraformIAMEnvironment()
        pattern = "**/*" if recursive else "*"
        for f in sorted(path.glob(pattern)):
            if f.is_file() and (f.suffix == ".tf" or f.name.endswith(".tf.json")):
                sub_env = self.parse_file(f)
                combined_env.merge(sub_env)

        return combined_env

    # --------------------------------------------------------------------------
    # Internal HCL Processing
    # --------------------------------------------------------------------------

    def _process_parsed_hcl(
        self,
        parsed: dict[str, Any],
        env: TerraformIAMEnvironment,
        source_file: str,
    ) -> None:
        """Extract IAM resources and data sources from cleaned HCL dictionary."""
        # 1. First collect data sources for aws_iam_policy_document
        data_policy_docs: dict[str, PolicyDocument] = {}
        for data_block in parsed.get("data", []):
            if isinstance(data_block, dict):
                for dtype, dinstances in data_block.items():
                    if dtype == "aws_iam_policy_document" and isinstance(dinstances, dict):
                        for dname, dbody in dinstances.items():
                            doc = self._parse_data_policy_document(dbody)
                            if doc is not None:
                                data_policy_docs[dname] = doc

        # 2. Process resources in two passes: core entities first, then relationships
        core_resources: list[tuple[str, str, dict[str, Any]]] = []
        relational_resources: list[tuple[str, str, dict[str, Any]]] = []
        for res_block in parsed.get("resource", []):
            if not isinstance(res_block, dict):
                continue
            for rtype, rinstances in res_block.items():
                if not isinstance(rinstances, dict):
                    continue
                for rname, rbody in rinstances.items():
                    if not isinstance(rbody, dict):
                        continue
                    if rtype in {
                        "aws_iam_role_policy_attachment",
                        "aws_iam_user_policy_attachment",
                        "aws_iam_group_policy_attachment",
                        "aws_iam_policy_attachment",
                        "aws_iam_user_group_membership",
                        "aws_iam_group_membership",
                    }:
                        relational_resources.append((rtype, rname, rbody))
                    else:
                        core_resources.append((rtype, rname, rbody))

        # Pass 1: Parse and register all core entities
        for rtype, rname, rbody in core_resources:
            self._parse_resource(
                res_type=rtype,
                res_name=rname,
                body=rbody,
                env=env,
                data_policy_docs=data_policy_docs,
                source_file=source_file,
            )

        # Pass 2: Parse relational resources using populated symbol table
        for rtype, rname, rbody in relational_resources:
            self._parse_resource(
                res_type=rtype,
                res_name=rname,
                body=rbody,
                env=env,
                data_policy_docs=data_policy_docs,
                source_file=source_file,
            )

    def _parse_resource(
        self,
        res_type: str,
        res_name: str,
        body: dict[str, Any],
        env: TerraformIAMEnvironment,
        data_policy_docs: dict[str, PolicyDocument],
        source_file: str,
    ) -> None:
        """Parse individual IAM resource blocks."""
        explicit_name = body.get("name")
        actual_name = _clean_str(explicit_name) if explicit_name else res_name
        path = _clean_str(body.get("path", "/"))
        tags = body.get("tags", {})
        if not isinstance(tags, dict):
            tags = {}

        def _register_symbol(rtype: str, rlabel: str, name: str, rarn: str) -> None:
            env.symbol_table[f"{rtype}.{rlabel}.name"] = name
            env.symbol_table[f"{rtype}.{rlabel}.arn"] = rarn
            env.symbol_table[f"{rtype}.{rlabel}"] = name
            env.symbol_table[rlabel] = name
            env.symbol_table[name] = name
            env.arn_table[f"{rtype}.{rlabel}.arn"] = rarn
            env.arn_table[rlabel] = rarn
            env.arn_table[name] = rarn

        if res_type == "aws_iam_user":
            arn = f"arn:aws:iam::{self.account_id}:user{path}{actual_name}".replace("//", "/")
            _register_symbol("aws_iam_user", res_name, actual_name, arn)
            env.users[actual_name] = TerraformUser(
                name=actual_name,
                arn=arn,
                path=path,
                tags={str(k): str(v) for k, v in tags.items()},
                source_file=source_file,
            )

        elif res_type == "aws_iam_role":
            arn = f"arn:aws:iam::{self.account_id}:role{path}{actual_name}".replace("//", "/")
            _register_symbol("aws_iam_role", res_name, actual_name, arn)
            assume_doc = self._resolve_policy_doc(body.get("assume_role_policy"), data_policy_docs)
            env.roles[actual_name] = TerraformRole(
                name=actual_name,
                arn=arn,
                path=path,
                assume_role_policy=assume_doc,
                tags={str(k): str(v) for k, v in tags.items()},
                source_file=source_file,
            )

        elif res_type == "aws_iam_group":
            arn = f"arn:aws:iam::{self.account_id}:group{path}{actual_name}".replace("//", "/")
            _register_symbol("aws_iam_group", res_name, actual_name, arn)
            env.groups[actual_name] = TerraformGroup(
                name=actual_name,
                arn=arn,
                path=path,
                source_file=source_file,
            )

        elif res_type == "aws_iam_policy":
            arn = f"arn:aws:iam::{self.account_id}:policy{path}{actual_name}".replace("//", "/")
            _register_symbol("aws_iam_policy", res_name, actual_name, arn)
            doc = self._resolve_policy_doc(body.get("policy"), data_policy_docs)
            if doc is not None:
                env.policies[arn] = TerraformPolicy(
                    name=actual_name,
                    arn=arn,
                    path=path,
                    description=_clean_str(body.get("description")),
                    policy_document=doc,
                    is_inline=False,
                    source_file=source_file,
                )

        elif res_type == "aws_iam_role_policy":
            role_target = env.resolve_name(str(body.get("role", "")))
            doc = self._resolve_policy_doc(body.get("policy"), data_policy_docs)
            if doc is not None and role_target:
                inline_arn = (
                    f"arn:aws:iam::{self.account_id}:policy/inline/{role_target}-{actual_name}"
                )
                env.policies[inline_arn] = TerraformPolicy(
                    name=f"{role_target}-{actual_name}",
                    arn=inline_arn,
                    policy_document=doc,
                    is_inline=True,
                    source_file=source_file,
                )
                env.attachments.append(
                    TerraformAttachment(
                        principal_type="Role",
                        principal_name=role_target,
                        policy_arn=inline_arn,
                        is_inline=True,
                        source_file=source_file,
                    )
                )

        elif res_type == "aws_iam_user_policy":
            user_target = env.resolve_name(str(body.get("user", "")))
            doc = self._resolve_policy_doc(body.get("policy"), data_policy_docs)
            if doc is not None and user_target:
                inline_arn = (
                    f"arn:aws:iam::{self.account_id}:policy/inline/{user_target}-{actual_name}"
                )
                env.policies[inline_arn] = TerraformPolicy(
                    name=f"{user_target}-{actual_name}",
                    arn=inline_arn,
                    policy_document=doc,
                    is_inline=True,
                    source_file=source_file,
                )
                env.attachments.append(
                    TerraformAttachment(
                        principal_type="User",
                        principal_name=user_target,
                        policy_arn=inline_arn,
                        is_inline=True,
                        source_file=source_file,
                    )
                )

        elif res_type == "aws_iam_group_policy":
            group_target = env.resolve_name(str(body.get("group", "")))
            doc = self._resolve_policy_doc(body.get("policy"), data_policy_docs)
            if doc is not None and group_target:
                inline_arn = (
                    f"arn:aws:iam::{self.account_id}:policy/inline/{group_target}-{actual_name}"
                )
                env.policies[inline_arn] = TerraformPolicy(
                    name=f"{group_target}-{actual_name}",
                    arn=inline_arn,
                    policy_document=doc,
                    is_inline=True,
                    source_file=source_file,
                )
                env.attachments.append(
                    TerraformAttachment(
                        principal_type="Group",
                        principal_name=group_target,
                        policy_arn=inline_arn,
                        is_inline=True,
                        source_file=source_file,
                    )
                )

        elif res_type == "aws_iam_role_policy_attachment":
            role_target = env.resolve_name(str(body.get("role", "")))
            policy_arn = env.resolve_arn(body.get("policy_arn"), self.account_id)
            if role_target and policy_arn:
                env.attachments.append(
                    TerraformAttachment(
                        principal_type="Role",
                        principal_name=role_target,
                        policy_arn=policy_arn,
                        is_inline=False,
                        source_file=source_file,
                    )
                )

        elif res_type == "aws_iam_user_policy_attachment":
            user_target = env.resolve_name(str(body.get("user", "")))
            policy_arn = env.resolve_arn(body.get("policy_arn"), self.account_id)
            if user_target and policy_arn:
                env.attachments.append(
                    TerraformAttachment(
                        principal_type="User",
                        principal_name=user_target,
                        policy_arn=policy_arn,
                        is_inline=False,
                        source_file=source_file,
                    )
                )

        elif res_type == "aws_iam_group_policy_attachment":
            group_target = env.resolve_name(str(body.get("group", "")))
            policy_arn = env.resolve_arn(body.get("policy_arn"), self.account_id)
            if group_target and policy_arn:
                env.attachments.append(
                    TerraformAttachment(
                        principal_type="Group",
                        principal_name=group_target,
                        policy_arn=policy_arn,
                        is_inline=False,
                        source_file=source_file,
                    )
                )

        elif res_type == "aws_iam_user_group_membership":
            user_target = env.resolve_name(str(body.get("user", "")))
            groups = body.get("groups", [])
            if isinstance(groups, str):
                groups = [groups]
            for g in groups:
                g_target = env.resolve_name(str(g))
                if user_target and g_target:
                    env.memberships.append(
                        TerraformMembership(
                            user_name=user_target,
                            group_name=g_target,
                            source_file=source_file,
                        )
                    )

        elif res_type == "aws_iam_group_membership":
            group_target = env.resolve_name(str(body.get("group", "")))
            users = body.get("users", [])
            if isinstance(users, str):
                users = [users]
            for u in users:
                u_target = env.resolve_name(str(u))
                if u_target and group_target:
                    env.memberships.append(
                        TerraformMembership(
                            user_name=u_target,
                            group_name=group_target,
                            source_file=source_file,
                        )
                    )

    # --------------------------------------------------------------------------
    # Policy Document & ARN Resolvers
    # --------------------------------------------------------------------------

    def _resolve_policy_arn(self, raw_arn: Any) -> str:
        """Resolve policy ARN string or resource reference."""
        if not raw_arn:
            return ""
        clean = _clean_str(raw_arn)
        if clean.startswith("arn:aws:iam:"):
            return clean

        # If it's a reference like aws_iam_policy.my_pol.arn
        target_name = _extract_identifier(clean)
        if target_name:
            return f"arn:aws:iam::{self.account_id}:policy/{target_name}"
        return clean

    def _resolve_policy_doc(
        self,
        raw: Any,
        data_policy_docs: dict[str, PolicyDocument],
    ) -> PolicyDocument | None:
        """Resolve raw policy content (JSON, heredoc, jsonencode, or data block reference)."""
        if raw is None:
            return None

        # 1. If raw is a dict directly
        if isinstance(raw, dict):
            try:
                return PolicyDocument.model_validate(raw)
            except Exception:
                pass

        val_str = _clean_str(raw)

        # 2. Check reference to data.aws_iam_policy_document.<name>.json
        data_match = re.search(r"data\.aws_iam_policy_document\.([A-Za-z0-9_-]+)\.json", val_str)
        if data_match:
            doc_name = data_match.group(1)
            if doc_name in data_policy_docs:
                return data_policy_docs[doc_name]

        # 3. Check jsonencode(...) format: ${jsonencode({ ... })}
        if "${jsonencode(" in val_str or "jsonencode(" in val_str:
            try:
                start_idx = val_str.find("jsonencode(") + len("jsonencode(")
                end_idx = val_str.rfind(")")
                if start_idx < end_idx:
                    inner = val_str[start_idx:end_idx].strip()
                    parsed = hcl2.load(io.StringIO(f"temp = {inner}"))
                    cleaned = _clean_data_structure(parsed.get("temp", {}))
                    return PolicyDocument.model_validate(cleaned)
            except Exception as e:
                logger.debug("Failed parsing jsonencode: %s", e)

        # 4. Check heredoc string
        unquoted_json = _extract_json_from_heredoc(val_str)
        if unquoted_json.startswith("{") and unquoted_json.endswith("}"):
            try:
                data = json.loads(unquoted_json)
                return PolicyDocument.model_validate(data)
            except Exception as e:
                logger.debug("Failed parsing raw JSON string: %s", e)

        return None

    def _parse_data_policy_document(self, body: dict[str, Any]) -> PolicyDocument | None:
        """Convert a data 'aws_iam_policy_document' block into a typed PolicyDocument."""
        statements_raw = body.get("statement", [])
        if isinstance(statements_raw, dict):
            statements_raw = [statements_raw]

        parsed_statements: list[Statement] = []
        for s in statements_raw:
            if not isinstance(s, dict):
                continue
            effect_str = _clean_str(s.get("effect", "Allow")).capitalize()
            effect = Effect.DENY if effect_str == "Deny" else Effect.ALLOW
            sid = _clean_str(s.get("sid")) if s.get("sid") else None

            # Actions
            actions_raw = s.get("actions", s.get("action", []))
            if isinstance(actions_raw, str):
                actions_list = [actions_raw]
            elif isinstance(actions_raw, list):
                actions_list = actions_raw
            else:
                actions_list = []
            actions = [_clean_str(a) for a in actions_list if a]

            # Resources
            resources_raw = s.get("resources", s.get("resource", []))
            if isinstance(resources_raw, str):
                resources_list = [resources_raw]
            elif isinstance(resources_raw, list):
                resources_list = resources_raw
            else:
                resources_list = []
            resources = [_clean_str(r) for r in resources_list if r]

            # Principals
            principals_dict: dict[str, list[str]] = {}
            p_blocks = s.get("principals", s.get("principal", []))
            if isinstance(p_blocks, dict):
                p_blocks_list = [p_blocks]
            elif isinstance(p_blocks, list):
                p_blocks_list = p_blocks
            else:
                p_blocks_list = []
            for pb in p_blocks_list:
                if isinstance(pb, dict):
                    ptype = _clean_str(pb.get("type", "AWS"))
                    pids = pb.get("identifiers", [])
                    pids_list = (
                        [pids]
                        if isinstance(pids, str)
                        else (pids if isinstance(pids, list) else [])
                    )
                    principals_dict[ptype] = [_clean_str(pid) for pid in pids_list]

            try:
                stmt = Statement(
                    Sid=sid,
                    Effect=effect,
                    Action=actions or ["*"],
                    Resource=resources or ["*"],
                    Principal=Principal(raw=principals_dict) if principals_dict else None,
                )
                parsed_statements.append(stmt)
            except Exception as e:
                logger.debug("Failed constructing statement from data source: %s", e)

        if not parsed_statements:
            return None
        return PolicyDocument(Statement=parsed_statements)

    # --------------------------------------------------------------------------
    # Conversion to IAMGraph
    # --------------------------------------------------------------------------

    def to_iam_graph(
        self,
        env: TerraformIAMEnvironment,
        graph_id: str = "terraform_imported_graph",
    ) -> IAMGraph:
        """Convert a parsed TerraformIAMEnvironment into a strongly typed IAMGraph."""
        from iam.generator.graph import (
            EdgeRelation,
            GraphEdge,
            GraphNode,
            IAMGraph,
            NodeType,
        )

        graph = IAMGraph(graph_id=graph_id)

        # 1. Add User nodes
        for u in env.users.values():
            node = GraphNode(
                id=f"user:{u.name}",
                node_type=NodeType.USER,
                arn=u.arn,
                name=u.name,
                account_id=self.account_id,
                tags=u.tags,
                metadata={"source_file": u.source_file} if u.source_file else {},
            )
            graph.add_node(node)

        # 2. Add Role nodes
        for r in env.roles.values():
            node = GraphNode(
                id=f"role:{r.name}",
                node_type=NodeType.ROLE,
                arn=r.arn,
                name=r.name,
                account_id=self.account_id,
                trust_policy=r.assume_role_policy,
                tags=r.tags,
                metadata={"source_file": r.source_file} if r.source_file else {},
            )
            graph.add_node(node)

        # 3. Add Group nodes
        for g in env.groups.values():
            node = GraphNode(
                id=f"group:{g.name}",
                node_type=NodeType.GROUP,
                arn=g.arn,
                name=g.name,
                account_id=self.account_id,
                metadata={"source_file": g.source_file} if g.source_file else {},
            )
            graph.add_node(node)

        # 4. Add Policy nodes
        for p in env.policies.values():
            pol_id = f"policy:{p.name}"
            # Check if policy is administrative
            is_admin = False
            for stmt in p.policy_document.statements:
                if (
                    stmt.effect == Effect.ALLOW
                    and (
                        "*" in stmt.action
                        or "administratoraccess" in [a.lower() for a in stmt.action]
                    )
                    and "*" in stmt.resource
                ):
                    is_admin = True
                    break

            node = GraphNode(
                id=pol_id,
                node_type=NodeType.POLICY,
                arn=p.arn,
                name=p.name,
                account_id=self.account_id,
                is_admin=is_admin,
                policy_document=p.policy_document,
                metadata={"is_inline": p.is_inline, "source_file": p.source_file},
            )
            graph.add_node(node)

            # Extract resource targets and create ACTS_ON edges
            for stmt in p.policy_document.statements:
                if stmt.effect != Effect.ALLOW:
                    continue
                for res_arn in stmt.resource:
                    if res_arn == "*":
                        continue
                    res_id = f"resource:{res_arn.replace(':', '_').replace('/', '_')}"
                    if graph.get_node(res_id) is None:
                        res_node = GraphNode(
                            id=res_id,
                            node_type=NodeType.RESOURCE,
                            arn=res_arn,
                            name=res_arn.split("/")[-1].split(":")[-1],
                            account_id=self.account_id,
                        )
                        graph.add_node(res_node)

                    graph.add_edge(
                        GraphEdge(
                            source=pol_id,
                            target=res_id,
                            relation=EdgeRelation.ACTS_ON,
                            actions=stmt.action,
                        )
                    )

        # 5. Add Memberships (User -> Group: MemberOf)
        for m in env.memberships:
            u_id = f"user:{m.user_name}"
            g_id = f"group:{m.group_name}"
            if graph.get_node(u_id) is not None and graph.get_node(g_id) is not None:
                graph.add_edge(
                    GraphEdge(
                        source=u_id,
                        target=g_id,
                        relation=EdgeRelation.MEMBER_OF,
                    )
                )

        # 6. Add Attachments (Principal -> Policy: AttachedWith)
        for att in env.attachments:
            p_name = env.resolve_name(att.principal_name)
            p_type = att.principal_type.lower()
            p_id = f"{p_type}:{p_name}"
            # Find matching policy node
            resolved_arn = env.resolve_arn(att.policy_arn, self.account_id)
            pol_node = graph.get_node_by_arn(resolved_arn)
            if pol_node is None:
                # Try finding by name
                pol_name = resolved_arn.split("/")[-1]
                pol_node = graph.get_node(f"policy:{pol_name}")

            # If still None and it's an AWS managed policy, create a placeholder policy node
            if pol_node is None and resolved_arn.startswith("arn:aws:iam::aws:policy/"):
                pol_name = resolved_arn.split("/")[-1]
                pol_id = f"policy:{pol_name}"
                is_admin = pol_name == "AdministratorAccess"
                pol_node = GraphNode(
                    id=pol_id,
                    node_type=NodeType.POLICY,
                    arn=resolved_arn,
                    name=pol_name,
                    account_id="aws",
                    is_admin=is_admin,
                    metadata={"is_aws_managed": True},
                )
                graph.add_node(pol_node)

            if graph.get_node(p_id) is not None and pol_node is not None:
                graph.add_edge(
                    GraphEdge(
                        source=p_id,
                        target=pol_node.id,
                        relation=EdgeRelation.ATTACHED_WITH,
                    )
                )

        # 7. Add AssumeRole trust edges from Role trust policies
        for r_node in graph.get_nodes_by_type(NodeType.ROLE):
            if r_node.trust_policy is None:
                continue
            for stmt in r_node.trust_policy.statements:
                if stmt.effect != Effect.ALLOW or not stmt.principal:
                    continue
                if isinstance(stmt.principal.raw, dict):
                    aws_principals = stmt.principal.raw.get("AWS", [])
                    for p_raw in aws_principals:
                        p_clean = _clean_str(p_raw)
                        trusted_node = graph.get_node_by_arn(p_clean)
                        if trusted_node is None:
                            resolved_name = env.resolve_name(p_clean)
                            trusted_node = graph.get_node(
                                f"user:{resolved_name}"
                            ) or graph.get_node(f"role:{resolved_name}")
                        if trusted_node is None:
                            resolved_arn = env.resolve_arn(p_clean, self.account_id)
                            trusted_node = graph.get_node_by_arn(resolved_arn)
                        if trusted_node is not None and trusted_node.id != r_node.id:
                            graph.add_edge(
                                GraphEdge(
                                    source=trusted_node.id,
                                    target=r_node.id,
                                    relation=EdgeRelation.ASSUMES_ROLE,
                                    actions=["sts:AssumeRole"],
                                )
                            )

        return graph
