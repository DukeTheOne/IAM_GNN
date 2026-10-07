"""Symbolic authorization evaluator and explicit Deny resolution engine."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from iam.parser.conditions import evaluate_condition_block
from iam.parser.schema import Effect, PolicyDocument, Statement
from iam.parser.wildcard import action_matches_pattern, resource_matches_pattern


class EvalDecision(str, Enum):
    """Resulting decision from AWS IAM policy evaluation."""

    ALLOW = "Allow"
    EXPLICIT_DENY = "ExplicitDeny"
    IMPLICIT_DENY = "ImplicitDeny"


class AuthRequest(BaseModel):
    """Contextual authorization request evaluated against policy statements."""

    model_config = ConfigDict(frozen=True)

    principal_arn: str
    action: str
    resource_arn: str
    context: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Detailed evaluation result with contributing statement provenance."""

    model_config = ConfigDict(frozen=True)

    decision: EvalDecision
    is_allowed: bool
    matched_allow_statements: list[Statement]
    matched_deny_statements: list[Statement]


def statement_matches_request(statement: Statement, request: AuthRequest) -> bool:
    """Evaluate whether an IAM statement applies to the given authorization request."""
    # 1. Principal / NotPrincipal check (if defined in statement)
    if statement.principal is not None and not statement.principal.matches(request.principal_arn):
        return False
    if statement.not_principal is not None and statement.not_principal.matches(
        request.principal_arn
    ):
        return False

    # 2. Action / NotAction check
    if statement.action:
        action_matched = any(
            action_matches_pattern(request.action, pat) for pat in statement.action
        )
        if not action_matched:
            return False
    elif statement.not_action:
        action_excluded = any(
            action_matches_pattern(request.action, pat) for pat in statement.not_action
        )
        if action_excluded:
            return False

    # 3. Resource / NotResource check
    if statement.resource:
        resource_matched = any(
            resource_matches_pattern(request.resource_arn, pat) for pat in statement.resource
        )
        if not resource_matched:
            return False
    elif statement.not_resource:
        resource_excluded = any(
            resource_matches_pattern(request.resource_arn, pat) for pat in statement.not_resource
        )
        if resource_excluded:
            return False

    # 4. Condition check
    if statement.condition:
        # Merge built-in request context variables into context dictionary
        ctx = dict(request.context)
        ctx.setdefault("aws:PrincipalArn", request.principal_arn)
        if not evaluate_condition_block(statement.condition, ctx):
            return False

    return True


class PolicyEvaluator:
    """Evaluates authorization requests adhering to the formal AWS evaluation model."""

    def __init__(
        self,
        policies: Sequence[PolicyDocument] | None = None,
        statements: Sequence[Statement] | None = None,
    ) -> None:
        """Initialize evaluator with collections of policies or raw statements."""
        self._statements: list[Statement] = []
        if policies:
            for p in policies:
                self._statements.extend(p.statements)
        if statements:
            self._statements.extend(statements)

    @property
    def statements(self) -> list[Statement]:
        """Return all managed statements."""
        return list(self._statements)

    def add_policy(self, policy: PolicyDocument) -> None:
        """Add a policy document to the evaluator."""
        self._statements.extend(policy.statements)

    def add_statement(self, statement: Statement) -> None:
        """Add an individual statement to the evaluator."""
        self._statements.append(statement)

    def evaluate(self, request: AuthRequest) -> EvalResult:
        """Evaluate an AuthRequest against all registered statements.

        Evaluation Logic:
        1. Explicit Deny overrides any Allow.
        2. In the absence of an Explicit Deny, at least one Explicit Allow is required.
        3. If no matching statements are found, the decision is Implicit Deny.
        """
        matched_allow: list[Statement] = []
        matched_deny: list[Statement] = []

        for stmt in self._statements:
            if statement_matches_request(stmt, request):
                if stmt.effect == Effect.DENY:
                    matched_deny.append(stmt)
                elif stmt.effect == Effect.ALLOW:
                    matched_allow.append(stmt)

        if matched_deny:
            return EvalResult(
                decision=EvalDecision.EXPLICIT_DENY,
                is_allowed=False,
                matched_allow_statements=matched_allow,
                matched_deny_statements=matched_deny,
            )

        if matched_allow:
            return EvalResult(
                decision=EvalDecision.ALLOW,
                is_allowed=True,
                matched_allow_statements=matched_allow,
                matched_deny_statements=[],
            )

        return EvalResult(
            decision=EvalDecision.IMPLICIT_DENY,
            is_allowed=False,
            matched_allow_statements=[],
            matched_deny_statements=[],
        )

    def filter_authorized_actions(
        self,
        principal_arn: str,
        actions: Iterable[str],
        resource_arn: str,
        context: dict[str, Any] | None = None,
    ) -> set[str]:
        """Filter a collection of candidate actions to only those permitted for the target resource."""
        ctx = context or {}
        authorized: set[str] = set()

        for action in actions:
            req = AuthRequest(
                principal_arn=principal_arn,
                action=action,
                resource_arn=resource_arn,
                context=ctx,
            )
            res = self.evaluate(req)
            if res.is_allowed:
                authorized.add(action)

        return authorized
