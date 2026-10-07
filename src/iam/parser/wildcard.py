"""Wildcard resolution and pattern matching for AWS actions and resource ARNs."""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable


def normalize_action_name(action: str) -> str:
    """Canonicalize AWS action name by stripping whitespace and converting to lowercase for comparison."""
    return action.strip().lower()


def action_matches_pattern(action: str, pattern: str) -> bool:
    """Check if an AWS action matches a pattern (case-insensitive with glob wildcards).

    Examples:
        - action_matches_pattern("s3:GetObject", "s3:*") -> True
        - action_matches_pattern("iam:CreateAccessKey", "iam:*AccessKey*") -> True
        - action_matches_pattern("lambda:InvokeFunction", "*") -> True
        - action_matches_pattern("ec2:RunInstances", "s3:*") -> False
    """
    act = normalize_action_name(action)
    pat = normalize_action_name(pattern)

    # Fast path for universal wildcard
    if pat == "*":
        return True

    # Translate glob pattern to compiled regex
    regex_pattern = fnmatch.translate(pat)
    return bool(re.match(regex_pattern, act))


def resource_matches_pattern(resource_arn: str, pattern_arn: str) -> bool:
    """Check if a resource ARN matches a pattern (supports standard AWS ARN wildcarding).

    Examples:
        - resource_matches_pattern("arn:aws:s3:::bucket/data.csv", "*") -> True
        - resource_matches_pattern("arn:aws:s3:::bucket/data.csv", "arn:aws:s3:::bucket/*") -> True
        - resource_matches_pattern("arn:aws:iam::123:role/Admin", "arn:aws:iam::123:role/Dev") -> False
    """
    res = resource_arn.strip()
    pat = pattern_arn.strip()

    if pat == "*":
        return True

    regex_pattern = fnmatch.translate(pat)
    return bool(re.match(regex_pattern, res))


class WildcardResolver:
    """Expands action wildcards into concrete action sets against an authorization universe."""

    def __init__(self, action_universe: Iterable[str] | None = None) -> None:
        """Initialize resolver with an optional known universe of AWS actions."""
        self._action_universe: dict[str, str] = {}
        if action_universe:
            for act in action_universe:
                self._action_universe[normalize_action_name(act)] = act

    def add_actions(self, actions: Iterable[str]) -> None:
        """Register additional valid AWS actions into the known universe."""
        for act in actions:
            self._action_universe[normalize_action_name(act)] = act

    @property
    def known_actions(self) -> set[str]:
        """Return the set of canonical known actions."""
        return set(self._action_universe.values())

    def expand_actions(
        self,
        patterns: Iterable[str],
        *,
        fallback_to_pattern: bool = True,
    ) -> set[str]:
        """Expand action patterns against the known universe.

        If a pattern contains no wildcard:
            - If it exists in the universe, returns the canonical action name.
            - Otherwise returns the pattern as-is (if fallback_to_pattern is True).

        If a pattern contains wildcards (*, ?):
            - Matches against all actions in the universe.
            - If no actions match and fallback_to_pattern is True, retains pattern.
        """
        matched_actions: set[str] = set()

        for pattern in patterns:
            pat = pattern.strip()
            if not pat:
                continue

            if "*" not in pat and "?" not in pat:
                norm = normalize_action_name(pat)
                if norm in self._action_universe:
                    matched_actions.add(self._action_universe[norm])
                elif fallback_to_pattern:
                    matched_actions.add(pat)
                continue

            # Wildcard expansion against registered universe
            expanded_from_universe = {
                canonical
                for norm, canonical in self._action_universe.items()
                if action_matches_pattern(norm, pat)
            }

            if expanded_from_universe:
                matched_actions.update(expanded_from_universe)
            elif fallback_to_pattern:
                matched_actions.add(pat)

        return matched_actions

    def expand_not_actions(self, not_patterns: Iterable[str]) -> set[str]:
        """Compute the set of actions permitted under a NotAction pattern.

        NotAction represents: (Action Universe - Expanded Patterns).
        """
        excluded = self.expand_actions(not_patterns, fallback_to_pattern=False)
        return self.known_actions - excluded
