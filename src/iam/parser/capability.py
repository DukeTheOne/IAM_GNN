"""AWS Action-Resource-Condition Capability Model C(a, r, c).

Enforces valid AWS authorization semantics so that downstream policy synthesis
never generates semantically or syntactically invalid policies.
"""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from iam.parser.schema import Statement
from iam.parser.wildcard import normalize_action_name


class ActionCapability(BaseModel):
    """Specification of capabilities and constraints for a single AWS action."""

    model_config = ConfigDict(frozen=True)

    action: str
    service: str
    wildcard_only: bool = False
    resource_types: list[str] = Field(default_factory=list)
    condition_keys: list[str] = Field(default_factory=list)

    @property
    def supports_resource_scoping(self) -> bool:
        """Return True if this action supports narrowing to specific resource ARNs."""
        return not self.wildcard_only and len(self.resource_types) > 0

    def matches_resource_type(self, resource_arn: str) -> bool:
        """Check if a specific resource ARN matches the supported resource types for this action."""
        if resource_arn == "*":
            return True

        if not self.supports_resource_scoping:
            return False

        arn_lower = resource_arn.lower()

        for res_type in self.resource_types:
            r = res_type.lower()
            if r == "role" and ":role/" in arn_lower:
                return True
            if r == "user" and ":user/" in arn_lower:
                return True
            if r == "policy" and ":policy/" in arn_lower:
                return True
            if r == "function" and (":function:" in arn_lower or ":function/" in arn_lower):
                return True
            if (
                r == "bucket"
                and "arn:aws:s3:::" in arn_lower
                and "/" not in arn_lower.split(":::")[1]
            ):
                return True
            if r == "object" and "arn:aws:s3:::" in arn_lower:
                return True
            if r == "key" and ":key/" in arn_lower:
                return True
            if r == "instance" and ":instance/" in arn_lower:
                return True
            if r in ("image", "subnet", "security-group", "volume") and f":{r}/" in arn_lower:
                return True

        return False

    def supports_condition_key(self, key: str, global_keys: set[str]) -> bool:
        """Check if a condition key is valid for this action or matches a global key pattern."""
        k_lower = key.strip().lower()

        # Check action-specific condition keys
        for allowed in self.condition_keys:
            pattern = allowed.strip().lower()
            if pattern == k_lower or fnmatch.fnmatch(k_lower, pattern):
                return True

        # Check global condition keys
        for allowed_global in global_keys:
            pattern = allowed_global.strip().lower()
            if pattern == k_lower or fnmatch.fnmatch(k_lower, pattern):
                return True

        return False


class TransformationValidation(BaseModel):
    """Detailed result of a proposed policy patch validation."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool
    reason: str


class CapabilityModel:
    """The versioned AWS Action Capability Model C(a, r, c)."""

    def __init__(
        self,
        actions: Mapping[str, ActionCapability],
        global_condition_keys: set[str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the capability model."""
        self._actions: dict[str, ActionCapability] = dict(actions)
        self._global_keys: set[str] = set(global_condition_keys)
        self._metadata: dict[str, Any] = metadata or {}

    @property
    def metadata(self) -> dict[str, Any]:
        """Return snapshot metadata."""
        return dict(self._metadata)

    @property
    def registered_actions(self) -> set[str]:
        """Return all canonical registered action names."""
        return {cap.action for cap in self._actions.values()}

    def get_action_capability(self, action: str) -> ActionCapability | None:
        """Retrieve capability record for an action (case-insensitive)."""
        return self._actions.get(normalize_action_name(action))

    def is_wildcard_only(self, action: str) -> bool:
        """Check if an action requires Resource: '*'."""
        cap = self.get_action_capability(action)
        if cap is None:
            return False
        return cap.wildcard_only

    def validate_transformation(
        self,
        action: str,
        target_resource: str,
        condition: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> TransformationValidation:
        """Validate whether a proposed transformation is valid according to AWS semantics."""
        cap = self.get_action_capability(action)
        if cap is None:
            return TransformationValidation(
                is_valid=False,
                reason=f"Action '{action}' is unrecognized in capability snapshot.",
            )

        # 1. Check Resource Scoping
        if target_resource != "*":
            if cap.wildcard_only:
                return TransformationValidation(
                    is_valid=False,
                    reason=(
                        f"Action '{cap.action}' is wildcard-only in AWS; "
                        f"scoping to '{target_resource}' is semantically invalid."
                    ),
                )
            if not cap.matches_resource_type(target_resource):
                return TransformationValidation(
                    is_valid=False,
                    reason=(
                        f"Target resource '{target_resource}' does not match supported "
                        f"resource types {cap.resource_types} for action '{cap.action}'."
                    ),
                )

        # 2. Check Condition Key Injections
        if condition:
            for operator_str, key_dict in condition.items():
                for key_name in key_dict:
                    if not cap.supports_condition_key(key_name, self._global_keys):
                        return TransformationValidation(
                            is_valid=False,
                            reason=(
                                f"Condition key '{key_name}' in operator '{operator_str}' is not "
                                f"applicable to action '{cap.action}'."
                            ),
                        )

        return TransformationValidation(
            is_valid=True,
            reason="Transformation satisfies AWS authorization capabilities.",
        )

    def is_valid_transformation(
        self,
        action: str,
        target_resource: str,
        condition: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> bool:
        """Boolean convenience API checking if a proposed transformation is valid."""
        return self.validate_transformation(action, target_resource, condition).is_valid

    def validate_statement(self, statement: Statement) -> list[str]:
        """Validate all actions and resources in an existing IAM statement against the model."""
        errors: list[str] = []
        for act in statement.action:
            for res in statement.resource:
                validation = self.validate_transformation(act, res, statement.condition)
                if not validation.is_valid:
                    errors.append(validation.reason)
        return errors

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapabilityModel:
        """Construct CapabilityModel from raw dictionary representation."""
        metadata = data.get("metadata", {})
        global_keys = set(data.get("global_condition_keys", []))
        raw_actions = data.get("actions", {})

        action_caps: dict[str, ActionCapability] = {}
        for key, act_data in raw_actions.items():
            cap = ActionCapability.model_validate(act_data)
            action_caps[normalize_action_name(key)] = cap

        return cls(action_caps, global_keys, metadata)

    @classmethod
    def from_json_file(cls, file_path: str | Path) -> CapabilityModel:
        """Load CapabilityModel from a JSON file snapshot."""
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


_DEFAULT_CAPABILITY_MODEL: CapabilityModel | None = None


def load_default_capability_model() -> CapabilityModel:
    """Load the default capability snapshot stored in data/schemas/capability_snapshot.json."""
    global _DEFAULT_CAPABILITY_MODEL
    if _DEFAULT_CAPABILITY_MODEL is None:
        # Locate project root based on file path
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        snapshot_path = project_root / "data" / "schemas" / "capability_snapshot.json"
        _DEFAULT_CAPABILITY_MODEL = CapabilityModel.from_json_file(snapshot_path)
    return _DEFAULT_CAPABILITY_MODEL


def is_valid_transformation(
    action: str,
    target_resource: str,
    condition: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    """Convenience global function checking transformation validity using default capability model."""
    model = load_default_capability_model()
    return model.is_valid_transformation(action, target_resource, condition)
