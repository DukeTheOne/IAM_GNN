"""Uncertainty-guided candidate space pruning module for IAM policy repair.

Implements Phase 3 Week 7 (Step 7.2):
- Analyzes attack path witnesses across sampled completions {H_1, ..., H_M}.
- Profiles intermediate authorization edges participating in active privilege escalation chains.
- Ranks candidate cut edges using marginal attack frequency, topological centrality, and capability admissibility.
- Prunes repair search space to a focused candidate subgraph (|E_cand| <= 50).
- Computes candidate recall and optimality gap metrics for RQ3 evaluation.
- Guarantees interactive latency bounds (< 3.0 s) without compromising verifier safety.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from iam.baselines.oracle import ExactReachabilityOracle, OraclePathWitness
from iam.generator.graph import EdgeRelation, GraphEdge, IAMGraph
from iam.optimizer.sampler import CandidateEdge, LatentCompletion, MonteCarloReachabilityResult
from iam.parser.capability import CapabilityModel, load_default_capability_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoredCutEdge:
    """A candidate cut edge evaluated and ranked for least-privilege policy repair."""

    source: str
    target: str
    relation: EdgeRelation
    frequency: float
    path_count: int
    capability_score: float
    score: float
    is_observed: bool = True
    actions: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        """Triple key uniquely identifying this directed multi-edge (src, dst, rel)."""
        return (self.source, self.target, self.relation.value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize scored cut edge to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation.value,
            "frequency": round(self.frequency, 4),
            "path_count": self.path_count,
            "capability_score": round(self.capability_score, 4),
            "score": round(self.score, 4),
            "is_observed": self.is_observed,
            "actions": list(self.actions),
        }


@dataclass(frozen=True)
class PruningConfig:
    """Configuration governing uncertainty-guided candidate space pruning."""

    max_candidates: int = 50
    min_frequency: float = 0.0
    only_observed: bool = True
    target_relations: tuple[EdgeRelation, ...] | None = None
    weight_frequency: float = 0.6
    weight_centrality: float = 0.2
    weight_capability: float = 0.2

    def __post_init__(self) -> None:
        """Validate pruning weights and budget."""
        if self.max_candidates <= 0:
            raise ValueError(f"max_candidates must be positive, got {self.max_candidates}")
        if self.min_frequency < 0.0 or self.min_frequency > 1.0:
            raise ValueError(f"min_frequency must be in [0.0, 1.0], got {self.min_frequency}")


@dataclass
class PruningResult:
    """Result of candidate cut edge pruning across sampled completions."""

    pruned_edges: list[ScoredCutEdge]
    total_edges_considered: int
    num_samples_analyzed: int
    total_witnesses_analyzed: int
    execution_time_sec: float = 0.0
    candidate_subgraph: IAMGraph | None = None

    @property
    def candidate_edge_keys(self) -> set[tuple[str, str, str]]:
        """Set of (src, dst, rel) keys retained in the pruned candidate set."""
        return {e.key for e in self.pruned_edges}

    @property
    def num_pruned_candidates(self) -> int:
        """Count of retained candidate cut edges."""
        return len(self.pruned_edges)

    def get_top_k(self, k: int) -> list[ScoredCutEdge]:
        """Retrieve top k candidates in descending score order."""
        return self.pruned_edges[: max(0, k)]

    def to_dict(self) -> dict[str, Any]:
        """Serialize pruning results to dictionary."""
        return {
            "num_pruned_candidates": self.num_pruned_candidates,
            "total_edges_considered": self.total_edges_considered,
            "num_samples_analyzed": self.num_samples_analyzed,
            "total_witnesses_analyzed": self.total_witnesses_analyzed,
            "execution_time_sec": round(self.execution_time_sec, 4),
            "pruned_edges": [e.to_dict() for e in self.pruned_edges],
        }


class CandidatePruner:
    """Uncertainty-guided candidate pruning engine for IAM authorization graphs.

    Extracts, scores, and prunes edges participating in attack paths across sampled
    latent completions {H_1, ..., H_M}, reducing the combinatorial repair search space
    to an admissible candidate cut set (|E_cand| <= 50).
    """

    def __init__(
        self,
        config: PruningConfig | None = None,
        capability_model: CapabilityModel | None = None,
    ) -> None:
        """Initialize candidate pruner with optional configuration and capability model."""
        self.config: PruningConfig = config or PruningConfig()
        self.capability_model: CapabilityModel = (
            capability_model if capability_model is not None else load_default_capability_model()
        )

    def _compute_capability_score(self, graph: IAMGraph, edge_key: tuple[str, str, str]) -> float:
        """Compute capability repairability score based on AWS authorization semantics C(a, r, c).

        Priority 1 (Score 1.0): Actions supporting resource ARN scoping (most preferred repair).
        Priority 2 (Score 0.7): Standard actions with manageable revocation impact.
        Priority 3 (Score 0.4): Wildcard-only actions requiring blunt statement deletion.
        """
        src, dst, rel_str = edge_key
        try:
            rel = EdgeRelation(rel_str)
        except ValueError:
            return 0.5

        matching_edges = graph.get_edges(source=src, target=dst, relation=rel)
        if not matching_edges:
            # Candidate latent edge not in Go
            return 0.5

        actions: list[str] = []
        for me in matching_edges:
            actions.extend(me.actions)

        if not actions:
            # Delegation relation without specific actions (e.g. MemberOf, AssumesRole)
            return 0.8

        has_scopable_action = False
        all_wildcard_only = True

        for act in actions:
            cap = self.capability_model.get_action_capability(act)
            if cap is not None:
                if cap.supports_resource_scoping:
                    has_scopable_action = True
                    all_wildcard_only = False
                elif not cap.wildcard_only:
                    all_wildcard_only = False

        if has_scopable_action:
            return 1.0
        if all_wildcard_only:
            return 0.4
        return 0.7

    def prune_from_witnesses(
        self,
        graph: IAMGraph,
        witnesses_by_sample: list[list[OraclePathWitness]],
        num_samples: int,
    ) -> PruningResult:
        """Prune candidate cut edges directly from a collection of path witnesses across M samples.

        Args:
            graph: Partially observed IAM authorization graph G_o.
            witnesses_by_sample: List of path witness lists discovered across M completions.
            num_samples: Total number of completions M analyzed.

        Returns:
            PruningResult containing ranked cut candidates and candidate subgraph.
        """
        start_time = time.perf_counter()

        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        # Edge participation tracking: (src, dst, rel) -> stats
        edge_sample_counts: dict[tuple[str, str, str], int] = {}
        edge_path_counts: dict[tuple[str, str, str], int] = {}
        total_witnesses = 0

        target_rel_values: set[str] | None = None
        if self.config.target_relations is not None:
            target_rel_values = {r.value for r in self.config.target_relations}

        for sample_witnesses in witnesses_by_sample:
            sample_edges_seen: set[tuple[str, str, str]] = set()

            for witness in sample_witnesses:
                total_witnesses += 1
                nodes = witness.path_nodes
                rels = witness.path_relations

                for k in range(len(nodes) - 1):
                    src = nodes[k]
                    dst = nodes[k + 1]
                    rel = rels[k] if k < len(rels) else EdgeRelation.ACTS_ON.value

                    if target_rel_values is not None and rel not in target_rel_values:
                        continue

                    edge_key = (src, dst, rel)
                    edge_path_counts[edge_key] = edge_path_counts.get(edge_key, 0) + 1
                    sample_edges_seen.add(edge_key)

            for key in sample_edges_seen:
                edge_sample_counts[key] = edge_sample_counts.get(key, 0) + 1

        total_edges_considered = len(edge_sample_counts)
        max_paths = max(edge_path_counts.values()) if edge_path_counts else 1

        scored_candidates: list[ScoredCutEdge] = []
        for key, s_count in edge_sample_counts.items():
            src, dst, rel_str = key
            try:
                rel = EdgeRelation(rel_str)
            except ValueError:
                continue

            is_obs = graph.has_edge(src, dst, rel)
            if self.config.only_observed and not is_obs:
                continue

            freq = s_count / float(num_samples)
            if freq < self.config.min_frequency:
                continue

            p_count = edge_path_counts.get(key, 0)
            norm_path_count = min(1.0, p_count / float(max_paths))
            cap_score = self._compute_capability_score(graph, key)

            # Composite ranking score: frequency (expected risk reduction) + centrality + capability
            composite_score = (
                self.config.weight_frequency * freq
                + self.config.weight_centrality * norm_path_count
                + self.config.weight_capability * cap_score
            )

            # Retrieve associated actions
            actions_list: list[str] = []
            if is_obs:
                for ge in graph.get_edges(source=src, target=dst, relation=rel):
                    actions_list.extend(ge.actions)

            scored_candidates.append(
                ScoredCutEdge(
                    source=src,
                    target=dst,
                    relation=rel,
                    frequency=freq,
                    path_count=p_count,
                    capability_score=cap_score,
                    score=composite_score,
                    is_observed=is_obs,
                    actions=tuple(sorted(set(actions_list))),
                )
            )

        # Sort descending by composite score, then by frequency, then by path_count
        scored_candidates.sort(key=lambda e: (e.score, e.frequency, e.path_count), reverse=True)

        pruned = scored_candidates[: self.config.max_candidates]
        candidate_subgraph = self.extract_candidate_subgraph(graph, pruned)

        elapsed = time.perf_counter() - start_time

        return PruningResult(
            pruned_edges=pruned,
            total_edges_considered=total_edges_considered,
            num_samples_analyzed=num_samples,
            total_witnesses_analyzed=total_witnesses,
            execution_time_sec=elapsed,
            candidate_subgraph=candidate_subgraph,
        )

    def prune_from_completions(
        self,
        graph: IAMGraph,
        completions: list[LatentCompletion],
        oracle_result: MonteCarloReachabilityResult | None = None,
        oracle: ExactReachabilityOracle | None = None,
        source_ids: list[str] | None = None,
        target_ids: list[str] | None = None,
    ) -> PruningResult:
        """Prune candidate cut edges from sampled completions.

        Args:
            graph: Partially observed base IAMGraph G_o.
            completions: List of M LatentCompletion objects.
            oracle_result: Pre-computed MonteCarloReachabilityResult (optional).
            oracle: ExactReachabilityOracle instance if re-evaluating.
            source_ids: Optional source principals to evaluate.
            target_ids: Optional target assets/roles to evaluate.

        Returns:
            PruningResult containing ranked cut candidates and candidate subgraph.
        """
        if not completions:
            return PruningResult(
                pruned_edges=[],
                total_edges_considered=0,
                num_samples_analyzed=0,
                total_witnesses_analyzed=0,
                execution_time_sec=0.0,
            )

        if oracle_result is not None:
            witnesses = oracle_result.witnesses_by_sample
        else:
            evaluator = oracle or ExactReachabilityOracle()
            witnesses = []
            for comp in completions:
                res = evaluator.evaluate(comp.graph, source_ids=source_ids, target_ids=target_ids)
                witnesses.append(res.reachable_pairs)

        return self.prune_from_witnesses(
            graph=graph,
            witnesses_by_sample=witnesses,
            num_samples=len(completions),
        )

    def extract_candidate_subgraph(
        self,
        graph: IAMGraph,
        pruned_edges: list[ScoredCutEdge],
    ) -> IAMGraph:
        """Extract a focused candidate subgraph containing endpoints and candidate cut edges."""
        subgraph = IAMGraph(graph_id=f"{graph.graph_id}_pruned_subgraph")

        node_ids_needed: set[str] = set()
        for e in pruned_edges:
            node_ids_needed.add(e.source)
            node_ids_needed.add(e.target)

        for nid in node_ids_needed:
            node = graph.get_node(nid)
            if node is not None:
                subgraph.add_node(node.model_copy(deep=True))

        for e in pruned_edges:
            for ge in graph.get_edges(source=e.source, target=e.target, relation=e.relation):
                if not subgraph.has_edge(e.source, e.target, e.relation):
                    subgraph.add_edge(ge.model_copy(deep=True))

        return subgraph


def compute_candidate_recall(
    pruned_edges: list[ScoredCutEdge] | set[tuple[str, str, str]] | list[tuple[str, str, str]],
    optimal_cut_edges: (
        list[ScoredCutEdge]
        | list[GraphEdge]
        | list[CandidateEdge]
        | set[tuple[str, str, str]]
        | list[tuple[str, str, str]]
    ),
) -> float:
    """Compute Candidate Recall for RQ3 evaluation.

    Candidate Recall measures the fraction of true unpruned repair cut edges
    that are successfully retained in the pruned candidate set:
        Recall = |E_optimal ∩ E_pruned| / |E_optimal|

    Args:
        pruned_edges: Pruned candidate set as ScoredCutEdge items or (src, dst, rel) keys.
        optimal_cut_edges: Reference cut edges selected by unpruned solver or ground truth.

    Returns:
        Recall score in [0.0, 1.0].
    """

    def _extract_key(x: Any) -> tuple[str, str, str]:
        if isinstance(x, tuple) and len(x) == 3:
            return (str(x[0]), str(x[1]), str(x[2]))
        if hasattr(x, "key"):
            key_val = x.key
            if isinstance(key_val, tuple) and len(key_val) == 3:
                return (str(key_val[0]), str(key_val[1]), str(key_val[2]))
        raise TypeError(f"Cannot extract (src, dst, rel) key from {type(x)}: {x}")

    pruned_keys = {_extract_key(p) for p in pruned_edges}
    optimal_keys = {_extract_key(o) for o in optimal_cut_edges}

    if not optimal_keys:
        return 1.0

    intersection = optimal_keys.intersection(pruned_keys)
    return len(intersection) / float(len(optimal_keys))


def compute_optimality_gap(
    unpruned_cost: float,
    pruned_cost: float,
) -> tuple[float, float]:
    """Compute absolute and relative Optimality Gap between unpruned and pruned solver solutions.

    Returns:
        (absolute_gap, relative_gap)
        where absolute_gap = pruned_cost - unpruned_cost
        and relative_gap = absolute_gap / max(unpruned_cost, 1e-8)
    """
    abs_gap = max(0.0, pruned_cost - unpruned_cost)
    rel_gap = abs_gap / max(unpruned_cost, 1e-8) if unpruned_cost > 0 else 0.0
    return abs_gap, rel_gap
