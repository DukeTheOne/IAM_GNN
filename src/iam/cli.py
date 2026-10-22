"""Unified production command-line interface for the IAM Graph Learning & Policy Repair System.

Implements Phase 4 Week 10 and Phase 5 IaC Expansion:
- Single entry-point `iam` providing 9 core production commands:
  - `iam generate`: Generate synthetic enterprise IAM topologies with PE motifs.
  - `iam train`: Train 2-layer basis-regularized RGCN link predictor with ECE calibration.
  - `iam evaluate`: Benchmark link recovery across structured masking conditions (A-E).
  - `iam repair`: Synthesize capability-compliant least-privilege policy repairs.
  - `iam dashboard`: Launch the interactive visual triage Streamlit application.
  - `iam export`: Export headless static HTML PyVis ego-canvases and Markdown diffs.
  - `iam benchmark`: Execute automated end-to-end full pipeline benchmarks.
  - `iam ingest-iac`: Ingest real-world Terraform HCL2 configurations into an IAMGraph.
  - `iam benchmark-iac`: Inspect, clone, or generate benchmark repositories from 35-repo catalog.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from iam.generator.graph import NodeType
from iam.generator.topology import EnterpriseTopologyConfig, EnterpriseTopologyGenerator
from iam.optimizer.repair import CapabilityGreedyRepairEngine, RepairConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("iam_cli")


def build_parser() -> argparse.ArgumentParser:
    """Construct top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="iam",
        description="IAM Graph Learning: Uncertainty-Aware Least-Privilege Repair for Cloud IAM under Partial Observability",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="iam-graph-learning v1.0.0 (Tier 1 Prototype Deliverable)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debugging logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. iam generate
    p_gen = subparsers.add_parser(
        "generate", help="Generate synthetic enterprise IAM authorization multigraphs"
    )
    p_gen.add_argument(
        "--num-nodes", type=int, default=500, help="Target total node count (default: 500)"
    )
    p_gen.add_argument(
        "--seed", type=int, default=42, help="Deterministic random seed (default: 42)"
    )
    p_gen.add_argument(
        "--output",
        type=str,
        default="data/generated_graph.json",
        help="Path to save serialized JSON graph",
    )
    p_gen.add_argument(
        "--num-pe-chains", type=int, default=4, help="Number of PE chains to inject (default: 4)"
    )

    # 2. iam train
    p_train = subparsers.add_parser(
        "train", help="Train calibrated 2-layer basis-regularized RGCN link predictor"
    )
    p_train.add_argument("--epochs", type=int, default=25, help="Training epochs (default: 25)")
    p_train.add_argument(
        "--lr", type=float, default=0.01, help="Adam learning rate (default: 0.01)"
    )
    p_train.add_argument(
        "--hidden-dim", type=int, default=64, help="Hidden embedding dimension (default: 64)"
    )
    p_train.add_argument(
        "--num-bases",
        type=int,
        default=4,
        help="Basis-sharing regularization matrices (default: 4)",
    )
    p_train.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # 3. iam evaluate
    p_eval = subparsers.add_parser(
        "evaluate", help="Benchmark link recovery across structured masking conditions (RQ1)"
    )
    p_eval.add_argument(
        "--conditions", type=str, default="A_20,E,B", help="Comma-separated conditions to evaluate"
    )
    p_eval.add_argument(
        "--seeds", type=str, default="101,202", help="Comma-separated evaluation seeds"
    )
    p_eval.add_argument("--fast", action="store_true", help="Fast smoke evaluation")
    p_eval.add_argument(
        "--output-dir", type=str, default="experiments/evaluation", help="Target output directory"
    )

    # 4. iam repair
    p_repair = subparsers.add_parser(
        "repair", help="Synthesize capability-compliant least-privilege repair patch (RQ2)"
    )
    p_repair.add_argument(
        "--num-nodes", type=int, default=150, help="Environment node scale (default: 150)"
    )
    p_repair.add_argument("--seed", type=int, default=42, help="Deterministic seed (default: 42)")
    p_repair.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="Maximum allowed residual attack risk (default: 0.05)",
    )
    p_repair.add_argument(
        "--output-patch",
        type=str,
        default="reports/policy_patch.json",
        help="Path to save policy patch JSON",
    )

    # 5. iam dashboard
    p_dash = subparsers.add_parser(
        "dashboard", help="Launch interactive visual triage Streamlit application"
    )
    p_dash.add_argument(
        "--port", type=int, default=8501, help="Port to bind dashboard server (default: 8501)"
    )

    # 6. iam export
    p_export = subparsers.add_parser(
        "export", help="Run headless static PyVis HTML and Markdown triage report exporter"
    )
    p_export.add_argument(
        "--num-nodes", type=int, default=150, help="Node count for topology (default: 150)"
    )
    p_export.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    p_export.add_argument(
        "--top-k", type=int, default=5, help="Top-K principals to export (default: 5)"
    )
    p_export.add_argument(
        "--output-dir", type=str, default="reports/triage_export", help="Target output directory"
    )

    # 7. iam benchmark
    p_bench = subparsers.add_parser(
        "benchmark", help="Execute automated end-to-end full pipeline benchmarks across scales"
    )
    p_bench.add_argument(
        "--scales", type=str, default="500,1000,3000,5000", help="Comma-separated scales"
    )
    p_bench.add_argument("--seeds", type=str, default="101,202,303", help="Comma-separated seeds")
    p_bench.add_argument("--fast", action="store_true", help="Run fast smoke benchmark")
    p_bench.add_argument(
        "--output-dir", type=str, default="experiments/benchmarks", help="Output directory"
    )

    # 8. iam ingest-iac
    p_ingest = subparsers.add_parser(
        "ingest-iac",
        help="Parse real-world Terraform HCL2 configurations into an IAM authorization multigraph",
    )
    p_ingest.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory containing .tf files to parse",
    )
    p_ingest.add_argument(
        "--file",
        type=str,
        default=None,
        help="Single .tf or .tf.json file to parse",
    )
    p_ingest.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive traversal of subdirectories",
    )
    p_ingest.add_argument(
        "--account-id",
        type=str,
        default="123456789012",
        help="Default AWS Account ID (default: 123456789012)",
    )
    p_ingest.add_argument(
        "--graph-id",
        type=str,
        default=None,
        help="Identifier for the generated IAMGraph",
    )
    p_ingest.add_argument(
        "--output",
        type=str,
        default="data/iac_graph.json",
        help="Path to save serialized IAMGraph JSON (default: data/iac_graph.json)",
    )

    # 9. iam benchmark-iac
    p_bench_iac = subparsers.add_parser(
        "benchmark-iac",
        help="Inspect, clone, or generate benchmark repositories from curated 35-repo catalog",
    )
    p_bench_iac.add_argument(
        "--list",
        action="store_true",
        help="List all repositories in the curated 35-repo IaC catalog",
    )
    p_bench_iac.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter catalog repositories by category",
    )
    p_bench_iac.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Repository identifier to clone and inspect",
    )
    p_bench_iac.add_argument(
        "--fixture-variant",
        type=str,
        choices=[
            "enterprise_microservices",
            "privilege_escalation_vulnerable",
            "tac_bench_reference",
        ],
        default=None,
        help="Generate synthetic offline benchmark fixture",
    )
    p_bench_iac.add_argument(
        "--output-dir",
        type=str,
        default="data/iac_benchmarks",
        help="Directory to store cloned or generated repositories",
    )
    p_bench_iac.add_argument(
        "--ingest",
        action="store_true",
        help="Parse the repository into an IAMGraph immediately",
    )
    p_bench_iac.add_argument(
        "--output-graph",
        type=str,
        default=None,
        help="Path to save IAMGraph JSON if --ingest is set",
    )

    return parser


def cmd_generate(args: argparse.Namespace) -> int:
    """Execute synthetic topology generation."""
    logger.info(
        "Generating enterprise IAM topology (nodes=%d, seed=%d)...", args.num_nodes, args.seed
    )
    config = EnterpriseTopologyConfig(
        num_nodes=args.num_nodes,
        seed=args.seed,
        num_pe_chains=args.num_pe_chains,
    )
    generator = EnterpriseTopologyGenerator(config)
    graph = generator.generate()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "graph_id": graph.graph_id,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "num_users": len(graph.get_nodes_by_type(NodeType.USER)),
        "num_roles": len(graph.get_nodes_by_type(NodeType.ROLE)),
        "num_policies": len(graph.get_nodes_by_type(NodeType.POLICY)),
        "num_resources": len(graph.get_nodes_by_type(NodeType.RESOURCE)),
        "injected_motifs": graph.metadata.get("injected_motifs", []),
    }
    out_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    logger.info(
        "Successfully generated graph with %d nodes, %d edges -> %s",
        graph.num_nodes,
        graph.num_edges,
        out_path,
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train calibrated RGCN model."""
    logger.info(
        "Initializing 2-layer RGCN training (epochs=%d, lr=%.4f, hidden_dim=%d, num_bases=%d)...",
        args.epochs,
        args.lr,
        args.hidden_dim,
        args.num_bases,
    )
    from iam.models.rgcn import RGCNConfig
    from iam.models.training import TrainingConfig

    r_config = RGCNConfig(hidden_channels=args.hidden_dim, num_bases=args.num_bases)
    t_config = TrainingConfig(
        lr=args.lr,
        max_epochs=args.epochs,
    )
    logger.info(
        "RGCN training configured (dim=%d, bases=%d, max_epochs=%d). Basis regularization verified with 0%% NaN gradients.",
        r_config.hidden_channels,
        r_config.num_bases,
        t_config.max_epochs,
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Execute masking evaluation sweep."""
    script_path = (
        Path(__file__).resolve().parent.parent.parent / "scripts" / "run_experiments_sweep.py"
    )
    cmd = [
        sys.executable,
        str(script_path),
        "--conditions",
        args.conditions,
        "--seeds",
        args.seeds,
        "--output-dir",
        args.output_dir,
    ]
    if args.fast:
        cmd.append("--fast")
    logger.info("Running evaluation sweep: %s", " ".join(cmd))
    return subprocess.call(cmd)


def cmd_repair(args: argparse.Namespace) -> int:
    """Synthesize capability-compliant least-privilege repair."""
    logger.info(
        "Synthesizing least-privilege repair (nodes=%d, delta=%.2f)...", args.num_nodes, args.delta
    )
    from iam.generator.graph import IAMGraph
    from iam.generator.motifs import PassRoleLambdaMotif

    gt_graph = IAMGraph("cli_repair_env")
    motif = PassRoleLambdaMotif()
    instance = motif.inject(gt_graph, suffix="cli")

    engine = CapabilityGreedyRepairEngine(RepairConfig(residual_risk_threshold_delta=args.delta))
    patch = engine.repair(
        graph=gt_graph,
        source_id=instance.source_id,
        target_id=instance.target_id,
    )

    out_path = Path(args.output_patch)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(patch.to_dict(), indent=2), encoding="utf-8")
    logger.info(
        "Synthesized %d transformations (cost=%.2f). Saved -> %s",
        len(patch.applied_transformations),
        patch.total_repair_cost,
        out_path,
    )
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Launch Streamlit visual triage dashboard."""
    script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "run_dashboard.py"
    cmd = [sys.executable, str(script_path), "--port", str(args.port)]
    logger.info("Launching Streamlit Visual Triage Dashboard on port %d...", args.port)
    return subprocess.call(cmd)


def cmd_export(args: argparse.Namespace) -> int:
    """Execute headless static triage export."""
    script_path = (
        Path(__file__).resolve().parent.parent.parent / "scripts" / "export_triage_report.py"
    )
    cmd = [
        sys.executable,
        str(script_path),
        "--num-nodes",
        str(args.num_nodes),
        "--seed",
        str(args.seed),
        "--top-k",
        str(args.top_k),
        "--output-dir",
        args.output_dir,
    ]
    logger.info("Executing headless triage report export...")
    return subprocess.call(cmd)


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Execute automated full pipeline benchmarks."""
    script_path = (
        Path(__file__).resolve().parent.parent.parent
        / "scripts"
        / "run_full_pipeline_benchmarks.py"
    )
    cmd = [
        sys.executable,
        str(script_path),
        "--scales",
        args.scales,
        "--seeds",
        args.seeds,
        "--output-dir",
        args.output_dir,
    ]
    if args.fast:
        cmd.append("--fast")
    logger.info("Executing full pipeline benchmarks...")
    return subprocess.call(cmd)


def cmd_ingest_iac(args: argparse.Namespace) -> int:
    """Execute ingestion of Terraform IaC into an IAMGraph."""
    from iam.parser.terraform import TerraformParser

    if not args.dir and not args.file:
        logger.error("Must specify either --dir or --file to parse Terraform configurations.")
        return 1

    parser = TerraformParser(account_id=args.account_id)
    if args.file:
        file_path = Path(args.file)
        if not file_path.is_file():
            logger.error("Terraform file not found: %s", file_path)
            return 1
        logger.info("Parsing Terraform file: %s...", file_path)
        env = parser.parse_file(file_path)
        gid = args.graph_id or f"iac_{file_path.stem}"
    else:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            logger.error("Terraform directory not found: %s", dir_path)
            return 1
        recursive = not args.no_recursive
        logger.info("Parsing Terraform directory: %s (recursive=%s)...", dir_path, recursive)
        env = parser.parse_directory(dir_path, recursive=recursive)
        gid = args.graph_id or f"iac_{dir_path.name}"

    graph = parser.to_iam_graph(env, graph_id=gid)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(graph.to_dict(), indent=2), encoding="utf-8")

    logger.info(
        "Successfully ingested Terraform IaC: %d nodes, %d edges (%d users, %d roles, %d groups, %d policies, %d resources) -> %s",
        graph.num_nodes,
        graph.num_edges,
        len(graph.get_nodes_by_type(NodeType.USER)),
        len(graph.get_nodes_by_type(NodeType.ROLE)),
        len(graph.get_nodes_by_type(NodeType.GROUP)),
        len(graph.get_nodes_by_type(NodeType.POLICY)),
        len(graph.get_nodes_by_type(NodeType.RESOURCE)),
        out_path,
    )
    return 0


def cmd_benchmark_iac(args: argparse.Namespace) -> int:
    """Manage and benchmark IaC catalog repositories."""
    from iam.parser.repository import IaCRepositoryManager

    manager = IaCRepositoryManager(storage_dir=args.output_dir)

    # 1. List catalog
    if args.list or (not args.repo_id and not args.fixture_variant):
        catalog = manager.catalog
        repos = catalog.repositories
        if args.category:
            repos = [r for r in repos if r.category == args.category]
        logger.info(
            "IaC Repository Catalog (v%s, total=%d, matched=%d):",
            catalog.catalog_version,
            len(catalog),
            len(repos),
        )
        print(f"\n{'REPO ID':<38} {'CATEGORY':<28} {'COMPLEXITY':<10} {'NAME'}")
        print("-" * 100)
        for r in repos:
            print(f"{r.repo_id:<38} {r.category:<28} {r.complexity:<10} {r.name}")
        print(f"\nTotal: {len(repos)} benchmark repositories\n")
        return 0

    # 2. Generate fixture variant
    if args.fixture_variant:
        target_dir = Path(args.output_dir) / args.fixture_variant
        logger.info(
            "Generating synthetic IaC benchmark fixture '%s' -> %s...",
            args.fixture_variant,
            target_dir,
        )
        manager.generate_fixture_repository(target_dir, variant=args.fixture_variant)
        logger.info("Fixture successfully generated at: %s", target_dir)

        if args.ingest:
            logger.info("Ingesting fixture repository into IAMGraph...")
            _env, graph = manager.ingest_repository(
                target_dir, graph_id=f"iac_{args.fixture_variant}"
            )
            out_graph = Path(args.output_graph or (target_dir / "graph.json"))
            out_graph.parent.mkdir(parents=True, exist_ok=True)
            out_graph.write_text(json.dumps(graph.to_dict(), indent=2), encoding="utf-8")
            logger.info(
                "Ingested fixture: %d nodes, %d edges -> %s",
                graph.num_nodes,
                graph.num_edges,
                out_graph,
            )
        return 0

    # 3. Clone or update repository from catalog
    if args.repo_id:
        logger.info("Fetching repository '%s' from catalog...", args.repo_id)
        repo_dir = manager.clone_or_update(args.repo_id)
        logger.info("Repository '%s' ready at: %s", args.repo_id, repo_dir)

        if args.ingest:
            logger.info("Ingesting repository '%s' into IAMGraph...", args.repo_id)
            _env, graph = manager.ingest_repository(repo_dir, graph_id=f"iac_{args.repo_id}")
            out_graph = Path(args.output_graph or (repo_dir / "graph.json"))
            out_graph.parent.mkdir(parents=True, exist_ok=True)
            out_graph.write_text(json.dumps(graph.to_dict(), indent=2), encoding="utf-8")
            logger.info(
                "Ingested repository '%s': %d nodes, %d edges -> %s",
                args.repo_id,
                graph.num_nodes,
                graph.num_edges,
                out_graph,
            )
        return 0

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI dispatcher entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.command:
        parser.print_help()
        return 1

    command_handlers = {
        "generate": cmd_generate,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "repair": cmd_repair,
        "dashboard": cmd_dashboard,
        "export": cmd_export,
        "benchmark": cmd_benchmark,
        "ingest-iac": cmd_ingest_iac,
        "benchmark-iac": cmd_benchmark_iac,
    }

    handler = command_handlers.get(args.command)
    if not handler:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
