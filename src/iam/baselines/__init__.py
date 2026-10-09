"""Baselines module for IAM authorization multigraphs.

Provides exact reachability oracles, deterministic BFS traversal (PMapper proxy),
classical machine learning link prediction, and rule-based motif completion heuristics.
"""

from iam.baselines.bfs import (
    BFSPathWitness,
    BFSReachabilityResult,
    DeterministicBFSEngine,
)
from iam.baselines.features import (
    EdgePairFeaturizer,
)
from iam.baselines.metrics import (
    BaselineBenchmarkSuite,
    BaselineMetricsResult,
    compute_attack_path_recovery,
    compute_false_negative_rate,
    compute_mean_reciprocal_rank,
    compute_pr_auc,
    compute_recall_at_k,
    compute_roc_auc,
)
from iam.baselines.motif_heuristic import (
    MotifCompletionHeuristic,
)
from iam.baselines.oracle import (
    ExactReachabilityOracle,
    OraclePathWitness,
    OracleReachabilityResult,
)
from iam.baselines.random_forest import (
    RandomForestLinkPredictor,
)

__all__ = [
    "BFSPathWitness",
    "BFSReachabilityResult",
    "BaselineBenchmarkSuite",
    "BaselineMetricsResult",
    "DeterministicBFSEngine",
    "EdgePairFeaturizer",
    "ExactReachabilityOracle",
    "MotifCompletionHeuristic",
    "OraclePathWitness",
    "OracleReachabilityResult",
    "RandomForestLinkPredictor",
    "compute_attack_path_recovery",
    "compute_false_negative_rate",
    "compute_mean_reciprocal_rank",
    "compute_pr_auc",
    "compute_recall_at_k",
    "compute_roc_auc",
]
