"""Uncertainty-guided candidate pruning, greedy capability repair, and Z3 MaxSAT repair."""

from iam.optimizer.pruning import (
    CandidatePruner,
    PruningConfig,
    PruningResult,
    ScoredCutEdge,
    compute_candidate_recall,
    compute_optimality_gap,
)
from iam.optimizer.repair import (
    CapabilityGreedyRepairEngine,
    PolicyPatchResult,
    RepairConfig,
    WorkloadConformanceChecker,
    apply_transformation_to_graph,
)
from iam.optimizer.sampler import (
    CandidateEdge,
    LatentCompletion,
    LatentCompletionSampler,
    MonteCarloReachabilityResult,
    ProductComparisonResult,
    compare_monte_carlo_vs_product,
)
from iam.optimizer.transformations import (
    PolicyTransformation,
    PolicyTransformationType,
    RemoveActionTransformation,
    ScopeResourceTransformation,
    generate_policy_diff,
)
from iam.optimizer.z3_solver import (
    Z3CanonicalRepairSolver,
    Z3Decision,
    Z3RepairResult,
)

__all__ = [
    "CandidateEdge",
    "CandidatePruner",
    "CapabilityGreedyRepairEngine",
    "LatentCompletion",
    "LatentCompletionSampler",
    "MonteCarloReachabilityResult",
    "PolicyPatchResult",
    "PolicyTransformation",
    "PolicyTransformationType",
    "ProductComparisonResult",
    "PruningConfig",
    "PruningResult",
    "RemoveActionTransformation",
    "RepairConfig",
    "ScopeResourceTransformation",
    "ScoredCutEdge",
    "WorkloadConformanceChecker",
    "Z3CanonicalRepairSolver",
    "Z3Decision",
    "Z3RepairResult",
    "apply_transformation_to_graph",
    "compare_monte_carlo_vs_product",
    "compute_candidate_recall",
    "compute_optimality_gap",
    "generate_policy_diff",
]
