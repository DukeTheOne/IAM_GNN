"""Action vocabulary and categorical schemas for IAM heterogeneous graph learning.

Provides an immutable, deterministic ActionVocabulary with reserved Out-of-Vocabulary (OOV)
handling, wildcard expansion, and fixed feature dimension contracts for all entity types.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from iam.parser.capability import CapabilityModel, load_default_capability_model
from iam.parser.wildcard import WildcardResolver, normalize_action_name

# Canonical departments in enterprise topologies
CANONICAL_DEPARTMENTS: tuple[str, ...] = (
    "SecOps",
    "DevOps",
    "DataBI",
    "QA",
    "Billing",
    "Interns",
    "Other",
)

# Canonical resource categories for AWS entities
CANONICAL_RESOURCE_TYPES: tuple[str, ...] = (
    "bucket",
    "key",
    "function",
    "instance",
    "role",
    "user",
    "policy",
    "other",
)

# Canonical AWS service prefixes
CANONICAL_SERVICES: tuple[str, ...] = (
    "iam",
    "s3",
    "kms",
    "lambda",
    "ec2",
    "sts",
    "rds",
    "other",
)

# Standard sub-feature slice lengths
DEPARTMENT_FEATURE_DIM: int = len(CANONICAL_DEPARTMENTS)  # 7
RESOURCE_TYPE_FEATURE_DIM: int = len(CANONICAL_RESOURCE_TYPES)  # 8
SERVICE_PREFIX_FEATURE_DIM: int = len(CANONICAL_SERVICES)  # 8
RELATIONAL_DEGREE_DIM: int = 10  # 5 in-degrees + 5 out-degrees
CENTRALITY_FEATURE_DIM: int = RELATIONAL_DEGREE_DIM + 1  # 10 degrees + 1 PageRank = 11


def one_hot_encode(
    category: str | None,
    categories: tuple[str, ...],
    default_idx: int = -1,
) -> np.ndarray:
    """Produce a 1D float32 one-hot vector for a categorical value.

    If category is None or unrecognized, sets default_idx (defaults to last element).
    """
    vec = np.zeros(len(categories), dtype=np.float32)
    if category is not None:
        cat_lower = category.lower().strip()
        for idx, cat in enumerate(categories):
            if cat.lower() == cat_lower:
                vec[idx] = 1.0
                return vec

    # Fallback to default index
    fallback_index = default_idx if default_idx >= 0 else len(categories) - 1
    if 0 <= fallback_index < len(categories):
        vec[fallback_index] = 1.0
    return vec


def parse_arn_service_and_type(arn: str) -> tuple[str, str]:
    """Extract (service, resource_type) from an AWS ARN string.

    Examples:
        - "arn:aws:s3:::my-bucket" -> ("s3", "bucket")
        - "arn:aws:kms:us-east-1:123:key/my-key" -> ("kms", "key")
        - "arn:aws:lambda:us-east-1:123:function:my-fn" -> ("lambda", "function")
        - "arn:aws:ec2:us-east-1:123:instance/i-123" -> ("ec2", "instance")
        - "arn:aws:iam::123:role/my-role" -> ("iam", "role")
        - "arn:aws:iam::123:user/my-user" -> ("iam", "user")
        - "arn:aws:iam::123:policy/my-pol" -> ("iam", "policy")
    """
    arn_clean = arn.strip().lower()
    if not arn_clean.startswith("arn:aws:"):
        return ("other", "other")

    parts = arn_clean.split(":")
    if len(parts) < 3:
        return ("other", "other")

    service = parts[2]
    matched_service = service if service in CANONICAL_SERVICES else "other"

    # S3 special handling: arn:aws:s3:::bucket-name[/path]
    if service == "s3":
        return ("s3", "bucket")

    # Resource type extraction from remainder (parts 5 or 6)
    remainder = ":".join(parts[5:]) if len(parts) >= 6 else ""
    if not remainder and len(parts) >= 5:
        remainder = parts[4]

    matched_type = "other"
    if "/" in remainder:
        res_type_prefix = remainder.split("/")[0]
    elif ":" in remainder:
        res_type_prefix = remainder.split(":")[0]
    else:
        res_type_prefix = remainder

    for rt in CANONICAL_RESOURCE_TYPES:
        if rt == res_type_prefix or rt in remainder:
            matched_type = rt
            break

    return (matched_service, matched_type)


class ActionVocabulary:
    """Immutable, deterministic action vocabulary for IAM permission representations.

    Reserves index 0 as Out-of-Vocabulary (<OOV>) to guarantee robust inductive
    generalization across unseen services, future IaC policies, and TAC-Bench actions.
    """

    OOV_TOKEN: str = "<OOV>"
    OOV_INDEX: int = 0

    def __init__(self, actions: Iterable[str]) -> None:
        """Initialize vocabulary from an iterable of canonical AWS actions."""
        # Eliminate duplicates, sort alphabetically for exact cross-platform determinism
        unique_sorted: list[str] = sorted(
            {a.strip() for a in actions if a.strip() and a.strip() != self.OOV_TOKEN}
        )

        self._idx_to_action: list[str] = [self.OOV_TOKEN, *unique_sorted]
        self._action_to_idx: dict[str, int] = {
            self.OOV_TOKEN: self.OOV_INDEX,
            normalize_action_name(self.OOV_TOKEN): self.OOV_INDEX,
        }

        for idx, act in enumerate(unique_sorted, start=1):
            self._action_to_idx[normalize_action_name(act)] = idx
            self._action_to_idx[act] = idx

        # Resolver for expanding wildcards (e.g. s3:* or *) against this vocabulary universe
        self._wildcard_resolver = WildcardResolver(unique_sorted)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size, including <OOV> at index 0."""
        return len(self._idx_to_action)

    @property
    def canonical_actions(self) -> list[str]:
        """List of all recognized canonical actions (excluding <OOV>)."""
        return self._idx_to_action[1:]

    def get_index(self, action: str) -> int:
        """Retrieve integer index for an action name, returning 0 if Out-of-Vocabulary."""
        norm = normalize_action_name(action)
        return self._action_to_idx.get(norm, self.OOV_INDEX)

    def get_action(self, index: int) -> str:
        """Retrieve canonical action name by integer index."""
        if 0 <= index < len(self._idx_to_action):
            return self._idx_to_action[index]
        raise IndexError(f"Action index {index} out of range [0, {len(self._idx_to_action) - 1}]")

    def contains(self, action: str) -> bool:
        """Check whether an action is explicitly recognized in the known vocabulary."""
        return normalize_action_name(action) in self._action_to_idx

    def encode_action(self, action: str) -> np.ndarray:
        """Generate a 1D float32 one-hot vector for a single action."""
        vec = np.zeros(self.vocab_size, dtype=np.float32)
        vec[self.get_index(action)] = 1.0
        return vec

    def encode_actions(
        self,
        actions: Iterable[str],
        expand_wildcards: bool = True,
    ) -> np.ndarray:
        """Generate a 1D float32 multi-hot bitmask across the action vocabulary.

        If expand_wildcards is True, patterns like 's3:*' or '*' are expanded against
        the vocabulary universe. Any actions not matched in the universe set index 0 (<OOV>).
        """
        vec = np.zeros(self.vocab_size, dtype=np.float32)
        raw_actions = list(actions)
        if not raw_actions:
            return vec

        if expand_wildcards:
            # Expand actions via WildcardResolver
            expanded = self._wildcard_resolver.expand_actions(raw_actions, fallback_to_pattern=True)
            for act in expanded:
                idx = self.get_index(act)
                vec[idx] = 1.0
        else:
            for act in raw_actions:
                idx = self.get_index(act)
                vec[idx] = 1.0

        return vec

    def decode_vector(
        self,
        vector: np.ndarray,
        threshold: float = 0.5,
        include_oov: bool = False,
    ) -> list[str]:
        """Decode a multi-hot action vector back to recognized action names."""
        if len(vector) != self.vocab_size:
            raise ValueError(
                f"Vector length {len(vector)} does not match vocabulary size {self.vocab_size}"
            )

        result: list[str] = []
        start_idx = 0 if include_oov else 1
        for idx in range(start_idx, self.vocab_size):
            if vector[idx] >= threshold:
                result.append(self._idx_to_action[idx])
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize vocabulary state to a dictionary."""
        return {
            "oov_token": self.OOV_TOKEN,
            "vocab_size": self.vocab_size,
            "actions": self.canonical_actions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionVocabulary:
        """Instantiate vocabulary from a serialized dictionary."""
        actions = data.get("actions", [])
        return cls(actions)

    def save_json(self, path: Path | str) -> None:
        """Save vocabulary definition to a JSON file."""
        out_p = Path(path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> ActionVocabulary:
        """Load vocabulary definition from a JSON file."""
        in_p = Path(path)
        with in_p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_capability_model(cls, capability_model: CapabilityModel) -> ActionVocabulary:
        """Build vocabulary from a loaded AWS CapabilityModel."""
        return cls(capability_model.registered_actions)


def load_default_action_vocabulary() -> ActionVocabulary:
    """Load the canonical ActionVocabulary from the default capability snapshot."""
    cap_model = load_default_capability_model()
    return ActionVocabulary.from_capability_model(cap_model)


@dataclass(frozen=True)
class FeatureDimensionConfig:
    """Strict feature dimension contracts across all entity types."""

    action_vocab_size: int
    user_dim: int
    role_dim: int
    policy_dim: int
    group_dim: int
    resource_dim: int


def get_feature_dimension_config(vocab: ActionVocabulary) -> FeatureDimensionConfig:
    """Compute exact feature dimensionality contracts based on ActionVocabulary size.

    Contracts:
        - User: Department(7) + SecurityFlags(2) + Centrality(11) + ActionBitmask(|A|)
        - Role: ServiceRoleFlag(1) + AdminFlag(1) + HighValue(1) + Department(7) + Centrality(11) + ActionBitmask(|A|)
        - Policy: StatementStats(3) + AdminFlag(1) + WildcardResource(1) + Centrality(3) + ActionBitmask(|A|)
        - Group: Department(7) + AdminFlag(1) + Centrality(3) + ActionBitmask(|A|)
        - Resource: ResourceType(8) + ServicePrefix(8) + HighValue(1) + Centrality(3)
    """
    v_size = vocab.vocab_size

    # User: 7 (dept) + 2 (admin, high_val) + 11 (centrality) + v_size
    user_dim = DEPARTMENT_FEATURE_DIM + 2 + CENTRALITY_FEATURE_DIM + v_size

    # Role: 1 (service_role) + 1 (admin) + 1 (high_val) + 7 (dept) + 11 (centrality) + v_size
    role_dim = 1 + 1 + 1 + DEPARTMENT_FEATURE_DIM + CENTRALITY_FEATURE_DIM + v_size

    # Policy: 3 (stmt_count, allow, deny) + 1 (is_admin) + 1 (wildcard_res) + 3 (centrality: in, out, pagerank) + v_size
    policy_dim = 3 + 1 + 1 + 3 + v_size

    # Group: 7 (dept) + 1 (is_admin) + 3 (centrality: in, out, pagerank) + v_size
    group_dim = DEPARTMENT_FEATURE_DIM + 1 + 3 + v_size

    # Resource: 8 (type) + 8 (service) + 1 (high_val) + 3 (centrality: in_acts_on, in_passes_to, pagerank)
    resource_dim = RESOURCE_TYPE_FEATURE_DIM + SERVICE_PREFIX_FEATURE_DIM + 1 + 3

    return FeatureDimensionConfig(
        action_vocab_size=v_size,
        user_dim=user_dim,
        role_dim=role_dim,
        policy_dim=policy_dim,
        group_dim=group_dim,
        resource_dim=resource_dim,
    )
