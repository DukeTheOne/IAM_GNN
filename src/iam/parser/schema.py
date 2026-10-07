"""Declarative Pydantic schemas for AWS IAM JSON policy documents and statements."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Effect(str, Enum):
    """IAM Statement effect."""

    ALLOW = "Allow"
    DENY = "Deny"


class Principal(BaseModel):
    """Principal specification in resource and trust policies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw: str | dict[str, list[str]]

    @model_validator(mode="before")
    @classmethod
    def normalize_principal(cls, data: Any) -> dict[str, Any]:
        """Normalize principal representation into raw canonical dictionary or wildcard string."""
        if isinstance(data, str):
            return {"raw": data}
        if isinstance(data, dict):
            normalized: dict[str, list[str]] = {}
            for k, v in data.items():
                if isinstance(v, str):
                    normalized[k] = [v]
                elif isinstance(v, list):
                    normalized[k] = [str(item) for item in v]
                else:
                    normalized[k] = [str(v)]
            return {"raw": normalized}
        msg = f"Invalid principal format: {data!r}"
        raise ValueError(msg)

    def matches(self, principal_arn: str) -> bool:
        """Check if this principal specification matches a given principal ARN."""
        if isinstance(self.raw, str):
            return self.raw == "*" or self.raw == principal_arn

        for _category, values in self.raw.items():
            if "*" in values or principal_arn in values:
                return True
        return False


def _normalize_string_or_list(value: Any) -> list[str]:
    """Helper to convert single string or list of values into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


class Statement(BaseModel):
    """Declarative representation of an IAM Statement."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="forbid")

    sid: str | None = Field(default=None, alias="Sid")
    effect: Effect = Field(alias="Effect")
    principal: Principal | None = Field(default=None, alias="Principal")
    not_principal: Principal | None = Field(default=None, alias="NotPrincipal")
    action: list[str] = Field(default_factory=list, alias="Action")
    not_action: list[str] = Field(default_factory=list, alias="NotAction")
    resource: list[str] = Field(default_factory=list, alias="Resource")
    not_resource: list[str] = Field(default_factory=list, alias="NotResource")
    condition: dict[str, dict[str, Any]] = Field(default_factory=dict, alias="Condition")

    @field_validator("action", "not_action", "resource", "not_resource", mode="before")
    @classmethod
    def coerce_to_list(cls, value: Any) -> list[str]:
        """Allow single string or collection for action and resource fields."""
        return _normalize_string_or_list(value)

    @field_validator("principal", "not_principal", mode="before")
    @classmethod
    def coerce_principal(cls, value: Any) -> Principal | None:
        """Coerce raw principal input to Principal model."""
        if value is None:
            return None
        if isinstance(value, Principal):
            return value
        return Principal.model_validate(value)

    @model_validator(mode="after")
    def validate_action_and_resource_presence(self) -> Statement:
        """Ensure statement has at least action/not_action and resource/not_resource (unless trust policy)."""
        if not self.action and not self.not_action:
            msg = "Statement must define either 'Action' or 'NotAction'."
            raise ValueError(msg)
        if self.action and self.not_action:
            msg = "Statement cannot define both 'Action' and 'NotAction'."
            raise ValueError(msg)

        # Trust policies might omit Resource, but identity policies require Resource or NotResource
        if self.resource and self.not_resource:
            msg = "Statement cannot define both 'Resource' and 'NotResource'."
            raise ValueError(msg)
        return self


class PolicyDocument(BaseModel):
    """Complete AWS IAM Policy Document."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="forbid")

    version: str = Field(default="2012-10-17", alias="Version")
    id: str | None = Field(default=None, alias="Id")
    statements: list[Statement] = Field(default_factory=list, alias="Statement")

    @field_validator("statements", mode="before")
    @classmethod
    def coerce_statements(cls, value: Any) -> list[Any]:
        """Allow a single Statement dictionary or a list of Statement dictionaries."""
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list):
            return value
        msg = f"Invalid Statement format: expected dict or list, got {type(value).__name__}"
        raise ValueError(msg)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyDocument:
        """Parse from raw dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> PolicyDocument:
        """Parse from raw JSON string."""
        return cls.model_validate(json.loads(json_str))

    def to_dict(self) -> dict[str, Any]:
        """Export to canonical dictionary format."""
        return self.model_dump(by_alias=True, exclude_none=True)

    def to_json(self, indent: int = 2) -> str:
        """Export to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
