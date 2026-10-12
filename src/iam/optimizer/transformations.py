"""Policy transformation primitives and cost models for IAM capability repair.

Implements Phase 3 Week 7 (Step 7.3):
- PolicyTransformation: Abstract base class for admissible policy modifications.
- ScopeResourceTransformation: Narrows Resource: "*" to specific verified target ARNs.
- RemoveActionTransformation: Excises actions when scoping is unsupported or insufficient.
- Fine-grained transformation cost model (w_scope = 1.0, w_remove = 5.0).
- AWS Action Capability Model C(a, r, c) integration and semantic validation.
- Standard unified policy diff synthesis for triage dashboard and explainability.
"""

from __future__ import annotations

import difflib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from iam.parser.capability import (
    CapabilityModel,
    TransformationValidation,
    load_default_capability_model,
)
from iam.parser.schema import PolicyDocument, Statement
from iam.parser.wildcard import normalize_action_name

logger = logging.getLogger(__name__)


class PolicyTransformationType(str, Enum):
    """Categorical type of policy transformation applied during repair."""

    SCOPE_RESOURCE = "ScopeResource"
    REMOVE_ACTION = "RemoveAction"
    INJECT_CONDITION = "InjectCondition"


@dataclass
class PolicyTransformation(ABC):
    """Abstract base class for all least-privilege IAM policy transformations."""

    policy_id: str
    statement_index: int
    cost: float
    description: str
    is_trust_policy: bool = False

    @property
    @abstractmethod
    def transformation_type(self) -> PolicyTransformationType:
        """Categorical transformation type."""
        ...

    @abstractmethod
    def apply(self, policy_doc: PolicyDocument) -> PolicyDocument:
        """Apply the transformation to a PolicyDocument and return the updated document.

        Args:
            policy_doc: Original parsed AWS IAM PolicyDocument AST.

        Returns:
            New updated PolicyDocument AST.
        """
        ...

    @abstractmethod
    def validate_capability(
        self, capability_model: CapabilityModel | None = None
    ) -> TransformationValidation:
        """Validate whether this transformation is admissible under AWS capability model C(a, r, c).

        Args:
            capability_model: Optional CapabilityModel instance; uses default if None.

        Returns:
            TransformationValidation indicating validity and diagnostic reason.
        """
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize transformation metadata to a dictionary."""
        ...


@dataclass
class ScopeResourceTransformation(PolicyTransformation):
    """Narrows Resource: '*' (or wildcards) to verified target resource ARNs.

    Priority 1 operation: Most operationally preferred transformation (cost = 1.0).
    Guarantees that actions supporting resource-level scoping are retained while
    unauthorized resource targets are excluded.
    """

    action: str = ""
    target_resources: list[str] = field(default_factory=list)
    original_resources: list[str] = field(default_factory=lambda: ["*"])
    cost: float = 1.0
    description: str = ""
    is_trust_policy: bool = False

    def __post_init__(self) -> None:
        """Initialize description if not provided."""
        if not self.description:
            res_summary = ", ".join(self.target_resources[:2])
            if len(self.target_resources) > 2:
                res_summary += f", ... (+{len(self.target_resources) - 2} more)"
            self.description = (
                f"Scope action '{self.action}' on policy '{self.policy_id}' "
                f"statement #{self.statement_index} to [{res_summary}]"
            )

    @property
    def transformation_type(self) -> PolicyTransformationType:
        """Categorical transformation type."""
        return PolicyTransformationType.SCOPE_RESOURCE

    def validate_capability(
        self, capability_model: CapabilityModel | None = None
    ) -> TransformationValidation:
        """Validate resource scoping against AWS Action Capability Model C(a, r, c)."""
        model = (
            capability_model if capability_model is not None else load_default_capability_model()
        )

        if not self.target_resources:
            return TransformationValidation(
                is_valid=False,
                reason="Target resource list cannot be empty for ScopeResourceTransformation.",
            )

        cap = model.get_action_capability(self.action)
        if cap is None:
            return TransformationValidation(
                is_valid=False,
                reason=f"Action '{self.action}' is unrecognized in capability snapshot.",
            )

        if cap.wildcard_only:
            return TransformationValidation(
                is_valid=False,
                reason=(
                    f"Action '{cap.action}' is wildcard-only in AWS; "
                    f"scoping to specific resource ARNs is semantically invalid."
                ),
            )

        if not cap.supports_resource_scoping:
            return TransformationValidation(
                is_valid=False,
                reason=(
                    f"Action '{cap.action}' does not support resource ARN scoping "
                    f"according to AWS Service Authorization Reference."
                ),
            )

        # Validate every target resource ARN matches supported resource types
        for res_arn in self.target_resources:
            val = model.validate_transformation(self.action, res_arn)
            if not val.is_valid:
                return val

        return TransformationValidation(
            is_valid=True,
            reason="Resource ARN scoping satisfies AWS capability model.",
        )

    def apply(self, policy_doc: PolicyDocument) -> PolicyDocument:
        """Apply resource scoping to target statement in PolicyDocument."""
        statements = list(policy_doc.statements)
        if self.statement_index < 0 or self.statement_index >= len(statements):
            raise IndexError(
                f"Statement index {self.statement_index} out of bounds "
                f"for policy '{self.policy_id}' with {len(statements)} statements."
            )

        target_stmt = statements[self.statement_index]
        norm_action = normalize_action_name(self.action)

        # Check if statement only contains this action
        matching_actions = [
            a for a in target_stmt.action if normalize_action_name(a) == norm_action
        ]
        other_actions = [a for a in target_stmt.action if normalize_action_name(a) != norm_action]

        if not other_actions:
            # Entire statement is dedicated to this action: narrow Resource directly
            updated_stmt = Statement(
                Sid=target_stmt.sid,
                Effect=target_stmt.effect,
                Principal=target_stmt.principal,
                NotPrincipal=target_stmt.not_principal,
                Action=target_stmt.action,
                NotAction=target_stmt.not_action,
                Resource=list(self.target_resources),
                NotResource=target_stmt.not_resource,
                Condition=dict(target_stmt.condition),
            )
            statements[self.statement_index] = updated_stmt
        else:
            # Statement contains multiple actions: split into un-scoped rest and scoped action
            remaining_stmt = Statement(
                Sid=target_stmt.sid,
                Effect=target_stmt.effect,
                Principal=target_stmt.principal,
                NotPrincipal=target_stmt.not_principal,
                Action=other_actions,
                NotAction=target_stmt.not_action,
                Resource=target_stmt.resource,
                NotResource=target_stmt.not_resource,
                Condition=dict(target_stmt.condition),
            )
            scoped_stmt = Statement(
                Sid=f"{target_stmt.sid or 'Stmt'}_Scoped" if target_stmt.sid else None,
                Effect=target_stmt.effect,
                Principal=target_stmt.principal,
                NotPrincipal=target_stmt.not_principal,
                Action=matching_actions or [self.action],
                NotAction=target_stmt.not_action,
                Resource=list(self.target_resources),
                NotResource=target_stmt.not_resource,
                Condition=dict(target_stmt.condition),
            )
            statements[self.statement_index] = remaining_stmt
            statements.append(scoped_stmt)

        return PolicyDocument(
            Version=policy_doc.version,
            Id=policy_doc.id,
            Statement=statements,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize transformation to dictionary."""
        return {
            "transformation_type": self.transformation_type.value,
            "policy_id": self.policy_id,
            "statement_index": self.statement_index,
            "action": self.action,
            "target_resources": list(self.target_resources),
            "original_resources": list(self.original_resources),
            "cost": self.cost,
            "description": self.description,
            "is_trust_policy": self.is_trust_policy,
        }


@dataclass
class RemoveActionTransformation(PolicyTransformation):
    """Excises an offending action from a policy statement.

    Priority 2 operation: More disruptive than resource scoping (cost = 5.0).
    Employed when action does not support resource scoping in C(a, r, c) (e.g., wildcard-only actions)
    or when legitimate workloads do not exercise the action.
    """

    action: str = ""
    cost: float = 5.0
    description: str = ""
    is_trust_policy: bool = False

    def __post_init__(self) -> None:
        """Initialize description if not provided."""
        if not self.description:
            self.description = (
                f"Remove action '{self.action}' from policy '{self.policy_id}' "
                f"statement #{self.statement_index}"
            )

    @property
    def transformation_type(self) -> PolicyTransformationType:
        """Categorical transformation type."""
        return PolicyTransformationType.REMOVE_ACTION

    def validate_capability(
        self, _capability_model: CapabilityModel | None = None
    ) -> TransformationValidation:
        """Action deletion is always admissible under AWS capability semantics."""
        return TransformationValidation(
            is_valid=True,
            reason="Action removal does not introduce any capability violations.",
        )

    def apply(self, policy_doc: PolicyDocument) -> PolicyDocument:
        """Remove specified action from target statement in PolicyDocument."""
        statements = list(policy_doc.statements)
        if self.statement_index < 0 or self.statement_index >= len(statements):
            raise IndexError(
                f"Statement index {self.statement_index} out of bounds "
                f"for policy '{self.policy_id}' with {len(statements)} statements."
            )

        target_stmt = statements[self.statement_index]
        norm_target = normalize_action_name(self.action)

        # Filter out action
        remaining_actions = [
            a for a in target_stmt.action if normalize_action_name(a) != norm_target
        ]

        if remaining_actions:
            # Statement still has other authorized actions
            updated_stmt = Statement(
                Sid=target_stmt.sid,
                Effect=target_stmt.effect,
                Principal=target_stmt.principal,
                NotPrincipal=target_stmt.not_principal,
                Action=remaining_actions,
                NotAction=target_stmt.not_action,
                Resource=target_stmt.resource,
                NotResource=target_stmt.not_resource,
                Condition=dict(target_stmt.condition),
            )
            statements[self.statement_index] = updated_stmt
        else:
            # Statement action list is empty; excise statement entirely
            statements.pop(self.statement_index)

        return PolicyDocument(
            Version=policy_doc.version,
            Id=policy_doc.id,
            Statement=statements,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize transformation to dictionary."""
        return {
            "transformation_type": self.transformation_type.value,
            "policy_id": self.policy_id,
            "statement_index": self.statement_index,
            "action": self.action,
            "cost": self.cost,
            "description": self.description,
            "is_trust_policy": self.is_trust_policy,
        }


def generate_policy_diff(
    original_doc: PolicyDocument,
    updated_doc: PolicyDocument,
    policy_name: str = "policy",
) -> str:
    """Generate git-style unified diff string between original and patched PolicyDocuments.

    Args:
        original_doc: Original PolicyDocument AST.
        updated_doc: Repaired PolicyDocument AST.
        policy_name: Name identifier for diff headers.

    Returns:
        Unified diff string with standard --- and +++ headers.
    """
    orig_lines = original_doc.to_json(indent=2).splitlines(keepends=True)
    updated_lines = updated_doc.to_json(indent=2).splitlines(keepends=True)

    diff = difflib.unified_diff(
        orig_lines,
        updated_lines,
        fromfile=f"a/{policy_name}.json",
        tofile=f"b/{policy_name}.json",
    )
    return "".join(diff)
