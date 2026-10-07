"""IAM AST parsing, capability model C(a,r,c), and symbolic evaluation."""

from iam.parser.capability import (
    ActionCapability,
    CapabilityModel,
    TransformationValidation,
    is_valid_transformation,
    load_default_capability_model,
)
from iam.parser.conditions import evaluate_condition_block
from iam.parser.evaluator import (
    AuthRequest,
    EvalDecision,
    EvalResult,
    PolicyEvaluator,
    statement_matches_request,
)
from iam.parser.schema import Effect, PolicyDocument, Principal, Statement
from iam.parser.wildcard import (
    WildcardResolver,
    action_matches_pattern,
    normalize_action_name,
    resource_matches_pattern,
)

__all__ = [
    "ActionCapability",
    "AuthRequest",
    "CapabilityModel",
    "Effect",
    "EvalDecision",
    "EvalResult",
    "PolicyDocument",
    "PolicyEvaluator",
    "Principal",
    "Statement",
    "TransformationValidation",
    "WildcardResolver",
    "action_matches_pattern",
    "evaluate_condition_block",
    "is_valid_transformation",
    "load_default_capability_model",
    "normalize_action_name",
    "resource_matches_pattern",
    "statement_matches_request",
]
