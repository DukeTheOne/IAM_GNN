"""Latent-completion sampling and Monte Carlo reachability module under partial IAM observability.

Implements Phase 3 Week 7 (Step 7.1):
- Transforms calibrated link prediction probabilities into graph-level uncertainty.
- Ingests calibrated edge probabilities p_hat_{uv}^r in [0, 1].
- Draws M independent Bernoulli completions H_i ~ P(H | G_o).
- Constructs materialized union graphs G_i = G_o ∪ H_i.
- Evaluates Monte Carlo attack reachability P_hat[Attack(s, t) | G_o].
- Benchmarks estimation variance against naive product-of-edge-probabilities estimator.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iam.baselines.oracle import ExactReachabilityOracle, OraclePathWitness
from iam.generator.graph import EdgeRelation, GraphEdge, IAMGraph

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateEdge:
    """A candidate latent edge with an associated calibrated probability.

    Represents a potential unobserved authorization relationship (u, r, v)
    inferred by link prediction models or structural heuristics.
    """

    source: str
    target: str
    relation: EdgeRelation
    probability: float
    is_bridge: bool = False
    motif_id: str | None = None
    actions: tuple[str, ...] = ()
    condition: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and clamp candidate edge probability."""
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(
                f"CandidateEdge probability must be in [0.0, 1.0], got {self.probability}"
            )

    @property
    def key(self) -> tuple[str, str, str]:
        """Triple key uniquely identifying this candidate directed multi-edge."""
        return (self.source, self.target, self.relation.value)

    def to_graph_edge(self) -> GraphEdge:
        """Instantiate this candidate edge as a GraphEdge for IAMGraph insertion."""
        return GraphEdge(
            source=self.source,
            target=self.target,
            relation=self.relation,
            is_bridge=self.is_bridge,
            motif_id=self.motif_id,
            actions=list(self.actions),
            condition=self.condition,
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_graph_edge(
        cls,
        edge: GraphEdge,
        probability: float,
        metadata: dict[str, Any] | None = None,
    ) -> CandidateEdge:
        """Construct a CandidateEdge from an existing GraphEdge and a probability score."""
        meta = dict(edge.metadata)
        if metadata:
            meta.update(metadata)
        return cls(
            source=edge.source,
            target=edge.target,
            relation=edge.relation,
            probability=float(probability),
            is_bridge=edge.is_bridge,
            motif_id=edge.motif_id,
            actions=tuple(edge.actions),
            condition=edge.condition,
            metadata=meta,
        )

    @classmethod
    def from_tuple(
        cls,
        source: str,
        target: str,
        relation: EdgeRelation | str,
        probability: float,
        is_bridge: bool = False,
        motif_id: str | None = None,
        actions: list[str] | tuple[str, ...] | None = None,
        condition: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CandidateEdge:
        """Convenience constructor from primitive argument types."""
        rel_enum = relation if isinstance(relation, EdgeRelation) else EdgeRelation(relation)
        return cls(
            source=source,
            target=target,
            relation=rel_enum,
            probability=float(probability),
            is_bridge=is_bridge,
            motif_id=motif_id,
            actions=tuple(actions) if actions else (),
            condition=condition,
            metadata=metadata or {},
        )


@dataclass
class LatentCompletion:
    """A single sampled graph completion H_i and its materialized union G_i = G_o ∪ H_i."""

    sample_id: int
    sampled_edges: list[CandidateEdge]
    graph: IAMGraph
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_sampled_edges(self) -> int:
        """Number of candidate edges instantiated in this completion."""
        return len(self.sampled_edges)

    @property
    def active_edge_keys(self) -> set[tuple[str, str, str]]:
        """Set of (src, dst, rel) keys active in this completion."""
        return {e.key for e in self.sampled_edges}

    @property
    def has_bridge_edge(self) -> bool:
        """True if any critical attack bridge is active in this sample."""
        return any(e.is_bridge for e in self.sampled_edges)

    @property
    def bridge_edges(self) -> list[CandidateEdge]:
        """List of active bridge edges in this completion."""
        return [e for e in self.sampled_edges if e.is_bridge]


@dataclass
class MonteCarloReachabilityResult:
    """Aggregated reachability and attack frequency evaluation over M sampled completions."""

    num_samples: int
    evaluated_pairs: list[tuple[str, str]]
    reachability_frequencies: dict[tuple[str, str], float]
    pe_frequencies: dict[tuple[str, str], float]
    witnesses_by_sample: list[list[OraclePathWitness]]
    active_edges_by_sample: list[list[CandidateEdge]]
    execution_time_sec: float = 0.0

    def get_reachability_probability(self, source_id: str, target_id: str) -> float:
        """Retrieve empirical reachability probability for a (source, target) pair."""
        return self.reachability_frequencies.get((source_id, target_id), 0.0)

    def get_pe_probability(self, source_id: str, target_id: str) -> float:
        """Retrieve empirical privilege escalation probability for a (source, target) pair."""
        return self.pe_frequencies.get((source_id, target_id), 0.0)

    def is_attack_possible(
        self,
        source_id: str,
        target_id: str,
        threshold: float = 0.0,
    ) -> bool:
        """Check if empirical attack probability strictly exceeds a threshold."""
        return self.get_pe_probability(source_id, target_id) > threshold

    def get_all_witnesses_for_pair(
        self,
        source_id: str,
        target_id: str,
    ) -> list[OraclePathWitness]:
        """Collect all path witnesses discovered across samples for a specific pair."""
        witnesses: list[OraclePathWitness] = []
        for sample_witnesses in self.witnesses_by_sample:
            for w in sample_witnesses:
                if w.source_id == source_id and w.target_id == target_id:
                    witnesses.append(w)
        return witnesses

    def get_mean_reachability(self) -> float:
        """Compute average reachability probability across all evaluated pairs."""
        if not self.reachability_frequencies:
            return 0.0
        return float(np.mean(list(self.reachability_frequencies.values())))

    def get_mean_pe_frequency(self) -> float:
        """Compute average privilege escalation probability across all evaluated pairs."""
        if not self.pe_frequencies:
            return 0.0
        return float(np.mean(list(self.pe_frequencies.values())))

    def summary_dict(self) -> dict[str, Any]:
        """Serialize result to a dictionary for reporting and diagnostics."""
        return {
            "num_samples": self.num_samples,
            "num_pairs_evaluated": len(self.evaluated_pairs),
            "mean_reachability": round(self.get_mean_reachability(), 4),
            "mean_pe_frequency": round(self.get_mean_pe_frequency(), 4),
            "execution_time_sec": round(self.execution_time_sec, 3),
            "pair_frequencies": [
                {
                    "source": src,
                    "target": tgt,
                    "reachability_prob": round(self.get_reachability_probability(src, tgt), 4),
                    "pe_prob": round(self.get_pe_probability(src, tgt), 4),
                }
                for src, tgt in self.evaluated_pairs
            ],
        }


class LatentCompletionSampler:
    """Monte Carlo sampler for latent authorization graphs under partial observability.

    Ingests an observed graph G_o and candidate latent relations with calibrated probabilities,
    drawing M independent Bernoulli completions H_i ~ P(H | G_o) and evaluating path reachability.
    """

    def __init__(
        self,
        observed_graph: IAMGraph,
        candidate_edges: list[CandidateEdge] | None = None,
    ) -> None:
        """Initialize the sampler with an observed graph and optional candidate edges.

        Args:
            observed_graph: The partially observed IAM authorization graph G_o.
            candidate_edges: Optional initial list of CandidateEdge items.
        """
        self.observed_graph: IAMGraph = observed_graph
        self._candidate_edges: list[CandidateEdge] = list(candidate_edges or [])

    @property
    def candidate_edges(self) -> list[CandidateEdge]:
        """List of candidate latent edges registered in the sampler."""
        return list(self._candidate_edges)

    @property
    def num_candidates(self) -> int:
        """Count of registered candidate latent edges."""
        return len(self._candidate_edges)

    def add_candidate_edge(self, edge: CandidateEdge) -> None:
        """Register a single candidate latent edge."""
        self._candidate_edges.append(edge)

    def add_candidate_edges(self, edges: list[CandidateEdge]) -> None:
        """Register multiple candidate latent edges."""
        self._candidate_edges.extend(edges)

    def clear_candidate_edges(self) -> None:
        """Clear all registered candidate edges."""
        self._candidate_edges.clear()

    def sample_completions(
        self,
        num_samples: int = 100,
        seed: int | None = None,
    ) -> list[LatentCompletion]:
        """Draw M independent Bernoulli graph completions H_i ~ P(H | G_o).

        For each candidate edge e in E_cand with calibrated probability p_e,
        draws X_e ~ Bernoulli(p_e). If X_e = 1, the edge is instantiated into G_i = G_o ∪ H_i.

        Args:
            num_samples: Number of completion graphs M to generate (default 100).
            seed: Deterministic random seed for reproducibility.

        Returns:
            List of LatentCompletion instances containing active edges and materialized graphs.
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        rng = np.random.default_rng(seed)
        completions: list[LatentCompletion] = []

        if not self._candidate_edges:
            # Degenerate case: no latent candidate edges, all completions equal G_o
            for i in range(num_samples):
                cloned = self.observed_graph.clone()
                completions.append(
                    LatentCompletion(
                        sample_id=i,
                        sampled_edges=[],
                        graph=cloned,
                        metadata={"seed": seed, "num_candidates": 0},
                    )
                )
            return completions

        # Vectorized Bernoulli draws: shape [M, num_candidates]
        probs = np.array([e.probability for e in self._candidate_edges], dtype=np.float64)
        draws = rng.random((num_samples, len(self._candidate_edges))) < probs

        for i in range(num_samples):
            active_indices = np.where(draws[i])[0]
            sampled_edges = [self._candidate_edges[idx] for idx in active_indices]

            # Materialize G_i = G_o ∪ H_i
            cloned_graph = self.observed_graph.clone()
            cloned_graph.graph_id = f"{self.observed_graph.graph_id}_sample_{i:03d}"

            for edge in sampled_edges:
                graph_edge = edge.to_graph_edge()
                # Verify endpoints exist in graph before insertion
                if cloned_graph.get_node(graph_edge.source) is None:
                    raise KeyError(
                        f"Candidate edge source '{graph_edge.source}' not found in observed graph."
                    )
                if cloned_graph.get_node(graph_edge.target) is None:
                    raise KeyError(
                        f"Candidate edge target '{graph_edge.target}' not found in observed graph."
                    )
                cloned_graph.add_edge(graph_edge)

            completions.append(
                LatentCompletion(
                    sample_id=i,
                    sampled_edges=sampled_edges,
                    graph=cloned_graph,
                    metadata={"seed": seed, "num_candidates": len(self._candidate_edges)},
                )
            )

        return completions

    def evaluate_reachability(
        self,
        completions: list[LatentCompletion],
        source_ids: list[str] | None = None,
        target_ids: list[str] | None = None,
        oracle: ExactReachabilityOracle | None = None,
    ) -> MonteCarloReachabilityResult:
        """Evaluate attack reachability across M sampled completions via Exact Symbolic Oracle.

        Computes empirical attack frequencies:
            P_hat[Attack(s, t) | G_o] = (1/M) ∑_{i=1}^M 1[Attack(s, t; G_o ∪ H_i)]

        Args:
            completions: List of M materialized LatentCompletion objects.
            source_ids: Optional list of source principal IDs to evaluate.
            target_ids: Optional list of target asset or role IDs to evaluate.
            oracle: Optional ExactReachabilityOracle instance (instantiated if None).

        Returns:
            MonteCarloReachabilityResult containing empirical frequencies and witness trails.
        """
        if not completions:
            raise ValueError("Cannot evaluate reachability on empty completions list.")

        start_time = time.perf_counter()
        evaluator = oracle or ExactReachabilityOracle()

        num_samples = len(completions)
        witnesses_by_sample: list[list[OraclePathWitness]] = []
        active_edges_by_sample: list[list[CandidateEdge]] = []

        # Track reachability occurrences per pair
        reach_counts: dict[tuple[str, str], int] = {}
        pe_counts: dict[tuple[str, str], int] = {}
        evaluated_pairs_set: set[tuple[str, str]] = set()

        for completion in completions:
            oracle_res = evaluator.evaluate(
                completion.graph,
                source_ids=source_ids,
                target_ids=target_ids,
            )

            witnesses_by_sample.append(oracle_res.reachable_pairs)
            active_edges_by_sample.append(completion.sampled_edges)

            # Record reachable pairs in this sample
            sample_reachable_pairs: set[tuple[str, str]] = set()
            for w in oracle_res.reachable_pairs:
                pair = (w.source_id, w.target_id)
                sample_reachable_pairs.add(pair)
                evaluated_pairs_set.add(pair)
                reach_counts[pair] = reach_counts.get(pair, 0) + 1

            for w in oracle_res.pe_pairs:
                pair = (w.source_id, w.target_id)
                pe_counts[pair] = pe_counts.get(pair, 0) + 1

        # If explicit sources and targets were specified, ensure pairs are tracked even if 0 reach
        if source_ids is not None and target_ids is not None:
            for s in source_ids:
                for t in target_ids:
                    evaluated_pairs_set.add((s, t))

        # Compute empirical frequencies
        reachability_frequencies: dict[tuple[str, str], float] = {
            pair: reach_counts.get(pair, 0) / float(num_samples) for pair in evaluated_pairs_set
        }
        pe_frequencies: dict[tuple[str, str], float] = {
            pair: pe_counts.get(pair, 0) / float(num_samples) for pair in evaluated_pairs_set
        }

        execution_time = time.perf_counter() - start_time

        return MonteCarloReachabilityResult(
            num_samples=num_samples,
            evaluated_pairs=sorted(evaluated_pairs_set),
            reachability_frequencies=reachability_frequencies,
            pe_frequencies=pe_frequencies,
            witnesses_by_sample=witnesses_by_sample,
            active_edges_by_sample=active_edges_by_sample,
            execution_time_sec=execution_time,
        )

    def sample_and_evaluate(
        self,
        num_samples: int = 100,
        seed: int | None = None,
        source_ids: list[str] | None = None,
        target_ids: list[str] | None = None,
        oracle: ExactReachabilityOracle | None = None,
    ) -> tuple[list[LatentCompletion], MonteCarloReachabilityResult]:
        """Convenience method to sample M completions and evaluate reachability in one pass."""
        completions = self.sample_completions(num_samples=num_samples, seed=seed)
        results = self.evaluate_reachability(
            completions=completions,
            source_ids=source_ids,
            target_ids=target_ids,
            oracle=oracle,
        )
        return completions, results


@dataclass(frozen=True)
class ProductComparisonResult:
    """Benchmark comparing Monte Carlo reachability against naive edge-product estimates."""

    source_id: str
    target_id: str
    path_length: int
    edge_probabilities: list[float]
    product_probability: float
    monte_carlo_probability: float
    num_samples: int
    absolute_error: float
    relative_error: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    is_consistent_with_independence: bool
    diagnostics: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize benchmark comparison result to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "path_length": self.path_length,
            "edge_probabilities": [round(p, 4) for p in self.edge_probabilities],
            "product_probability": round(self.product_probability, 4),
            "monte_carlo_probability": round(self.monte_carlo_probability, 4),
            "num_samples": self.num_samples,
            "absolute_error": round(self.absolute_error, 4),
            "relative_error": round(self.relative_error, 4),
            "standard_error": round(self.standard_error, 4),
            "ci_95": (round(self.ci_lower, 4), round(self.ci_upper, 4)),
            "is_consistent_with_independence": self.is_consistent_with_independence,
            "diagnostics": self.diagnostics,
        }


def compare_monte_carlo_vs_product(
    graph: IAMGraph,
    candidate_edges: list[CandidateEdge],
    source_id: str,
    target_id: str,
    num_samples: int = 1000,
    seed: int = 42,
    oracle: ExactReachabilityOracle | None = None,
) -> ProductComparisonResult:
    """Benchmark empirical Monte Carlo reachability against naive product of edge probabilities.

    Analyzes why treating edge probabilities as independent path probabilities introduces
    error, especially when multiple alternative paths exist or when latent edges share causes.

    Args:
        graph: Partially observed base IAMGraph G_o.
        candidate_edges: Latent candidate edges with calibrated probabilities.
        source_id: Originating principal ID.
        target_id: Target asset or role ID.
        num_samples: Monte Carlo sample budget M (default 1000 for tight confidence intervals).
        seed: Random seed for sampling reproducibility.
        oracle: Optional ExactReachabilityOracle instance.

    Returns:
        ProductComparisonResult detailing absolute/relative error and confidence intervals.
    """
    sampler = LatentCompletionSampler(observed_graph=graph, candidate_edges=candidate_edges)
    completions, mc_result = sampler.sample_and_evaluate(
        num_samples=num_samples,
        seed=seed,
        source_ids=[source_id],
        target_ids=[target_id],
        oracle=oracle,
    )

    mc_prob = mc_result.get_reachability_probability(source_id, target_id)

    # Compute naive product probability: P_prod = ∏ p_e for candidate edges
    # For a path where only some candidate edges are required, extract probabilities
    edge_probs = [e.probability for e in candidate_edges]
    prod_prob = float(np.prod(edge_probs)) if edge_probs else (1.0 if mc_prob > 0 else 0.0)

    abs_err = abs(mc_prob - prod_prob)
    rel_err = abs_err / max(prod_prob, 1e-8)

    # Standard error of Bernoulli Monte Carlo estimator: SE = sqrt(p * (1 - p) / M)
    se = math.sqrt(max(mc_prob * (1.0 - mc_prob), 1e-9) / num_samples)
    ci_lower = max(0.0, mc_prob - 1.96 * se)
    ci_upper = min(1.0, mc_prob + 1.96 * se)

    # Check if naive product estimate falls within 95% confidence interval
    is_consistent = ci_lower <= prod_prob <= ci_upper

    path_len = len(candidate_edges)
    if is_consistent:
        diag = (
            f"Independent single chain: Product estimator P_prod={prod_prob:.4f} closely aligns "
            f"with Monte Carlo empirical frequency P_mc={mc_prob:.4f} (within 95% CI)."
        )
    else:
        if mc_prob > prod_prob:
            diag = (
                f"Multi-path / redundant escalation topology: Monte Carlo P_mc={mc_prob:.4f} > "
                f"Product P_prod={prod_prob:.4f}. Multi-path reachability P(A ∪ B) > P(A) · P(B)."
            )
        else:
            diag = (
                f"Topological bottleneck / constraint divergence: Monte Carlo P_mc={mc_prob:.4f} < "
                f"Product P_prod={prod_prob:.4f} (naive multiplication overestimates reachability)."
            )

    return ProductComparisonResult(
        source_id=source_id,
        target_id=target_id,
        path_length=path_len,
        edge_probabilities=edge_probs,
        product_probability=prod_prob,
        monte_carlo_probability=mc_prob,
        num_samples=num_samples,
        absolute_error=abs_err,
        relative_error=rel_err,
        standard_error=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_consistent_with_independence=is_consistent,
        diagnostics=diag,
    )
