"""CLI script to run Step 5.5: Thin-Slice Benchmark Experiment.

Executes the first empirical answer to the central research question (RQ1):
Comparing 2-Layer RGCN with Bilinear Decoder against Deterministic BFS,
Random Forest, and Motif Completion Heuristics under Condition E and Condition A.

Usage:
    uv run python scripts/run_thin_slice.py [--epochs 25] [--output-dir experiments/thin_slice]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from iam.models.thin_slice import ThinSliceConfig, ThinSliceExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("thin_slice_runner")


def parse_args() -> argparse.Namespace:
    """Parse command line flags."""
    parser = argparse.ArgumentParser(
        description="Run Step 5.5 Thin-Slice Experiment comparing RGCN vs Baselines on held-out environments."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs for RGCN (default: 25).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW optimizer (default: 1e-3).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden channel dimensionality for RGCN (default: 128).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/thin_slice",
        help="Output directory for generated plots, reports, and JSON summaries.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic reproducibility (default: 42).",
    )
    return parser.parse_args()


def main() -> int:
    """Execute the experiment runner and print summary table."""
    args = parse_args()

    config = ThinSliceConfig(
        num_epochs=args.epochs,
        lr=args.lr,
        hidden_channels=args.hidden_dim,
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )

    logger.info("=================================================================")
    logger.info("   STARTING STEP 5.5 THIN-SLICE EXPERIMENT RUNNER (RQ1 GATE)   ")
    logger.info("=================================================================")
    logger.info(f"Target Output Directory: {config.output_dir.resolve()}")
    logger.info(f"Epochs: {config.num_epochs} | LR: {config.lr} | Hidden: {config.hidden_channels}")

    runner = ThinSliceExperimentRunner(config=config)
    result = runner.run()

    report_path = config.output_dir / config.markdown_filename
    if report_path.exists():
        print("\n" + report_path.read_text(encoding="utf-8") + "\n")

    logger.info("=================================================================")
    logger.info(f"   EXPERIMENT COMPLETE IN {result.execution_time_sec:.2f}s   ")
    logger.info(f"   Calibrated Temperature: T* = {result.calibrated_temperature:.4f}")
    logger.info(
        f"   Val ECE Before: {result.val_ece_before:.4f} -> After: {result.val_ece_after:.4f}"
    )
    logger.info(f"   Plot Artifact: {result.output_files.get('plot')}")
    logger.info(f"   JSON Artifact: {result.output_files.get('json')}")
    logger.info(f"   Markdown Report: {result.output_files.get('report')}")
    logger.info("=================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
