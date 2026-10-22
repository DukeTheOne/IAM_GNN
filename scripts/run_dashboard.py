#!/usr/bin/env python3
"""Convenience CLI launcher for the IAM Visual Triage Dashboard."""

from __future__ import annotations

import argparse
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

    parser = argparse.ArgumentParser(
        description="Launch the Streamlit IAM Visual Triage Dashboard",
        add_help=False,
    )
    parser.add_argument("--port", type=int, default=None, help="Port to bind dashboard server")
    parser.add_argument("--server.port", dest="server_port", type=int, default=None)
    parser.add_argument("--host", "--address", dest="host", type=str, default=None)
    parser.add_argument("--server.address", dest="server_address", type=str, default=None)

    args, remaining = parser.parse_known_args()

    port = args.server_port or args.port or 8501
    address = args.server_address or args.host

    streamlit_args: list[str] = ["--server.port", str(port)]
    if address:
        streamlit_args.extend(["--server.address", str(address)])
    streamlit_args.extend(remaining)

    # Locate streamlit binary in current virtual environment
    bin_dir = Path(sys.executable).parent
    streamlit_bin = bin_dir / "streamlit"
    if not streamlit_bin.is_file():
        streamlit_bin = bin_dir / "streamlit.exe"

    if streamlit_bin.is_file():
        cmd = [str(streamlit_bin), "run", str(app_path)] + streamlit_args
    else:
        # Fallback to python -m streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)] + streamlit_args

    logger.info("Starting Streamlit Visual Triage Dashboard on port %d...", port)
    logger.info("Executing command: %s", " ".join(cmd))

    try:
        res = subprocess.run(cmd, check=False)
        sys.exit(res.returncode)
    except FileNotFoundError:
        fallback_cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)] + streamlit_args
        logger.info("Falling back to module execution: %s", " ".join(fallback_cmd))
        res = subprocess.run(fallback_cmd, check=False)
        sys.exit(res.returncode)


if __name__ == "__main__":
    main()
