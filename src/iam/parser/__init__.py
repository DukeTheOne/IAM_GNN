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
from iam.parser.repository import (
    IaCRepositoryCatalog,
    IaCRepositoryManager,
    IaCRepositoryMetadata,
)
from iam.parser.schema import Effect, PolicyDocument, Principal, Statement
from iam.parser.terraform import (
    TerraformAttachment,
    TerraformGroup,
    TerraformIAMEnvironment,
    TerraformMembership,
    TerraformParser,
    TerraformPolicy,
    TerraformRole,
    TerraformUser,
)
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
    "IaCRepositoryCatalog",
    "IaCRepositoryManager",
    "IaCRepositoryMetadata",
    "PolicyDocument",
    "PolicyEvaluator",
    "Principal",
    "Statement",
    "TerraformAttachment",
    "TerraformGroup",
    "TerraformIAMEnvironment",
    "TerraformMembership",
    "TerraformParser",
    "TerraformPolicy",
    "TerraformRole",
    "TerraformUser",
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
