#!/usr/bin/env python3
"""CLI script to execute the Phase 2 Week 6 experimental sweep across seeds and masking conditions.

Usage:
    uv run python scripts/run_experiments_sweep.py
    uv run python scripts/run_experiments_sweep.py --fast
    uv run python scripts/run_experiments_sweep.py --seeds 101,202,303 --epochs 15
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from iam.models.sweep import ExperimentalSweepRunner, SweepConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sweep_cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 Week 6 multi-seed structured masking benchmark sweep and Gate G0 evaluation."
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="101,202,303,404,505",
        help="Comma-separated random seeds (default: 101,202,303,404,505)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Training epochs per seed (default: 25)",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="A_00,A_10,A_20,A_30,A_40,E,B,C,D",
        help="Comma-separated conditions to evaluate (default: A_00,A_10,A_20,A_30,A_40,E,B,C,D)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/sweep",
        help="Output directory for reports and figures (default: experiments/sweep)",
    )
    parser.add_argument(
        "--no-ablations",
        action="store_true",
        help="Disable neural and feature ablations",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast smoke benchmark with 2 seeds, 5 epochs, and reduced node counts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.fast:
        logger.info("Executing FAST benchmark mode for rapid verification...")
        seeds = [101, 202]
        epochs = 5
        conditions = ["A_20", "E", "B"]
        num_nodes_train = 40
        num_nodes_val = 35
        num_nodes_test = 45
        train_envs = 2
        val_envs = 1
        test_envs = 1
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        epochs = args.epochs
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
        num_nodes_train = 80
        num_nodes_val = 70
        num_nodes_test = 90
        train_envs = 4
        val_envs = 1
        test_envs = 2

    config = SweepConfig(
        seeds=seeds,
        conditions=conditions,
        num_epochs=epochs,
        num_nodes_train=num_nodes_train,
        num_nodes_val=num_nodes_val,
        num_nodes_test=num_nodes_test,
        train_envs_per_seed=train_envs,
        val_envs_per_seed=val_envs,
        test_envs_per_seed=test_envs,
        output_dir=Path(args.output_dir),
        run_ablations=not args.no_ablations,
    )

    logger.info(
        f"Sweep initialized: seeds={config.seeds}, conditions={config.conditions}, epochs={config.num_epochs}"
    )

    runner = ExperimentalSweepRunner(config=config)
    result_bundle = runner.run_sweep()

    logger.info("==================================================================")
    logger.info(f"SWEEP EXECUTION COMPLETE in {result_bundle.execution_time_sec:.1f} s")
    logger.info(f"GATE G0 DECISION: {result_bundle.gate_decision}")
    for r in result_bundle.gate_rationale:
        logger.info(f"  * {r}")
    logger.info(f"Report saved to: {config.output_dir / 'sweep_report.md'}")
    logger.info(f"JSON metrics saved to: {config.output_dir / 'sweep_results.json'}")
    logger.info("==================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
