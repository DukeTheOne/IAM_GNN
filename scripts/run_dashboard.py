#!/usr/bin/env python3
"""Convenience CLI launcher for the IAM Visual Triage Dashboard."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_dashboard")


def main() -> None:
    """Launch the Streamlit dashboard application."""
    repo_root = Path(__file__).resolve().parent.parent
    app_path = repo_root / "src" / "iam" / "dashboard" / "app.py"

    if not app_path.exists():
        logger.error("Dashboard app not found at: %s", app_path)
        sys.exit(1)

    # Check for streamlit binary in current virtual environment
    streamlit_bin = sys.executable.replace("python3", "streamlit").replace("python", "streamlit")
    cmd = [streamlit_bin, "run", str(app_path)] + sys.argv[1:]

    logger.info("Starting Streamlit Visual Triage Dashboard...")
    logger.info("Executing command: %s", " ".join(cmd))

    try:
        res = subprocess.run(cmd, check=False)
        sys.exit(res.returncode)
    except FileNotFoundError:
        # Fallback to python -m streamlit
        fallback_cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)] + sys.argv[1:]
        logger.info("Falling back to module execution: %s", " ".join(fallback_cmd))
        res = subprocess.run(fallback_cmd, check=False)
        sys.exit(res.returncode)


if __name__ == "__main__":
    main()
