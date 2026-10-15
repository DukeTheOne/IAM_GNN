"""Unit tests for unified production CLI interface (src/iam/cli.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iam.cli import build_parser, main


class TestIAMCommandLineInterface:
    """Tests for the unified iam command-line dispatcher and subcommands."""

    def test_parser_construction(self) -> None:
        parser = build_parser()
        assert parser.prog == "iam"
        subcommands = [
            "generate",
            "train",
            "evaluate",
            "repair",
            "dashboard",
            "export",
            "benchmark",
        ]
        for sub in subcommands:
            assert sub in parser.format_help()

    def test_main_no_arguments_shows_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main([])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "usage: iam" in captured.out or "usage: iam" in captured.err

    def test_cli_generate_command(self, tmp_path: Path) -> None:
        out_file = tmp_path / "test_graph.json"
        exit_code = main(
            ["generate", "--num-nodes", "60", "--seed", "42", "--output", str(out_file)]
        )
        assert exit_code == 0
        assert out_file.exists()

        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert "num_nodes" in data
        assert data["num_nodes"] >= 40

    def test_cli_train_command(self) -> None:
        exit_code = main(
            ["train", "--epochs", "1", "--lr", "0.01", "--hidden-dim", "16", "--num-bases", "2"]
        )
        assert exit_code == 0

    def test_cli_repair_command(self, tmp_path: Path) -> None:
        out_patch = tmp_path / "test_patch.json"
        exit_code = main(["repair", "--num-nodes", "50", "--output-patch", str(out_patch)])
        assert exit_code == 0
        assert out_patch.exists()

        patch_data = json.loads(out_patch.read_text(encoding="utf-8"))
        assert "applied_transformations" in patch_data

    def test_cli_export_command(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "export_out"
        exit_code = main(
            ["export", "--num-nodes", "50", "--top-k", "2", "--output-dir", str(out_dir)]
        )
        assert exit_code == 0
        assert (out_dir / "triage_summary.md").exists()

    def test_cli_benchmark_fast_command(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "benchmarks_out"
        exit_code = main(["benchmark", "--fast", "--output-dir", str(out_dir)])
        assert exit_code == 0
        assert (out_dir / "benchmark_report.md").exists()
